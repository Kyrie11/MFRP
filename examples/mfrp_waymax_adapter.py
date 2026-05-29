"""WOMD/Waymax adapter for MFRP same-root intervention groups.

This adapter implements the repository contract used by
``scripts/build_same_root_dataset.py``:

    build_groups(womd_pattern, split, config, max_scenarios, num_workers)
        -> Iterable[mfrp.data.schema.SameRootGroup]

It is deliberately conservative: it refuses to emit fake/log-playback response
labels.  If Waymax or the local WOMD path is unavailable it raises an actionable
error instead of creating placeholder data.  The reflection helpers cover the
public Waymax dataloader layout documented in the Waymax data tutorial; if your
installed Waymax exposes slightly different attribute names, update the small
``_extract_*`` helpers, not the MFRP schema code.
"""
from __future__ import annotations

from dataclasses import dataclass
import dataclasses
from pathlib import Path
from typing import Any, Iterable, Iterator
import glob
import math
import numpy as np

from mfrp.data.scene_schema import AgentTrackTensor, RootScene, RouteContext
from mfrp.data.schema import (
    BoundaryPair,
    CandidateValidity,
    EgoCandidate,
    SameRootGroup,
    root_scene_hash,
)
from mfrp.data.interaction_region import build_interaction_region, constant_velocity_extrapolate
from mfrp.data.label_extraction import make_response_observation, coercion_witness_label
from mfrp.data.priority import compute_priority_score


@dataclass
class VariantSpec:
    name: str
    aggressiveness: float = 1.0
    desired_speed_scale: float = 1.0
    min_gap_scale: float = 1.0


DEFAULT_VARIANTS = (
    VariantSpec("neutral_idm", aggressiveness=1.0, desired_speed_scale=1.0, min_gap_scale=1.0),
    VariantSpec("conservative_idm", aggressiveness=0.7, desired_speed_scale=0.9, min_gap_scale=1.3),
    VariantSpec("assertive_idm", aggressiveness=1.3, desired_speed_scale=1.1, min_gap_scale=0.75),
)


def _require_waymax():
    try:
        from waymax import config as wx_config  # type: ignore
        from waymax import dataloader  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Waymax is required for my_project.mfrp_waymax_adapter. Install it with\n"
            "  pip install git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax\n"
            "and make sure JAX/TensorFlow can read your local WOMD TFExample files."
        ) from e
    return wx_config, dataloader


def _as_np(x: Any) -> np.ndarray:
    try:
        import jax
        x = jax.device_get(x)
    except Exception:
        pass
    return np.asarray(x)


def _maybe_get(obj: Any, *names: str, default: Any = None) -> Any:
    cur = obj
    for name in names:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(name, default)
        else:
            cur = getattr(cur, name, default)
    return cur


def _extract_sim_trajectory(state: Any) -> Any:
    return _maybe_get(state, "sim_trajectory") or _maybe_get(state, "log_trajectory") or _maybe_get(state, "trajectory")


def _extract_states(state: Any, history_steps: int) -> tuple[np.ndarray, list[int], list[str], int, float]:
    traj = _extract_sim_trajectory(state)
    if traj is None:
        raise RuntimeError("Could not find sim/log trajectory in Waymax state")
    x = _as_np(_maybe_get(traj, "x"))
    y = _as_np(_maybe_get(traj, "y"))
    z = _as_np(_maybe_get(traj, "z", default=np.zeros_like(x)))
    length = _as_np(_maybe_get(traj, "length", default=np.full_like(x, 4.5)))
    width = _as_np(_maybe_get(traj, "width", default=np.full_like(x, 2.0)))
    yaw = _as_np(_maybe_get(traj, "yaw", default=np.zeros_like(x)))
    vel_x = _as_np(_maybe_get(traj, "vel_x", default=np.zeros_like(x)))
    vel_y = _as_np(_maybe_get(traj, "vel_y", default=np.zeros_like(x)))
    valid = _as_np(_maybe_get(traj, "valid", default=np.ones_like(x, dtype=bool)))
    # Waymax trajectories are usually [num_objects, num_timesteps].
    if x.ndim != 2:
        raise RuntimeError(f"Expected Waymax trajectory x with shape [N,T], got {x.shape}")
    T = x.shape[1]
    hist = min(history_steps, T)
    speed = np.sqrt(np.square(vel_x) + np.square(vel_y))
    states = np.stack([
        x[:, :hist], y[:, :hist], z[:, :hist], vel_x[:, :hist], vel_y[:, :hist],
        speed[:, :hist], yaw[:, :hist], length[:, :hist], width[:, :hist], valid[:, :hist].astype(np.float32),
    ], axis=-1).astype(np.float32)
    object_ids = _as_np(_maybe_get(traj, "object_id", default=np.arange(x.shape[0]))).reshape(-1).astype(int).tolist()
    obj_type_arr = _as_np(_maybe_get(traj, "object_type", default=np.zeros(x.shape[0], dtype=np.int32))).reshape(-1)
    object_types = [str(int(t)) for t in obj_type_arr]
    sdc_idx = int(_as_np(_maybe_get(state, "sdc_track_index", default=0)).reshape(())) if np.asarray(_maybe_get(state, "sdc_track_index", default=0)).size else 0
    return states, object_ids, object_types, sdc_idx, 0.1


def _scenario_id(state: Any, fallback: int) -> str:
    sid = _maybe_get(state, "scenario_id") or _maybe_get(state, "object_metadata", "scenario_id")
    if sid is None:
        return f"scenario_{fallback:08d}"
    arr = np.asarray(sid)
    if arr.shape == ():
        v = arr.item()
        return v.decode() if isinstance(v, bytes) else str(v)
    return str(arr.reshape(-1)[0])


def _extract_route_context(state: Any, ego_index: int) -> RouteContext:
    # WOMD 1.3+ may expose sdc_paths; older Waymax versions may not.
    sdc_paths = _maybe_get(state, "sdc_paths")
    polylines: list[np.ndarray] = []
    if sdc_paths is not None:
        for name in ("x", "y"):
            if _maybe_get(sdc_paths, name) is None:
                break
        else:
            x = _as_np(_maybe_get(sdc_paths, "x"))
            y = _as_np(_maybe_get(sdc_paths, "y"))
            if x.ndim >= 2:
                for p in range(min(3, x.shape[0])):
                    polylines.append(np.stack([x[p], y[p]], axis=-1).astype(np.float32))
    return RouteContext(route_polylines=polylines, metadata={"route_source": "waymax_sdc_paths" if polylines else "unavailable"})


def _extract_map_summary(state: Any) -> dict[str, Any]:
    rg = _maybe_get(state, "roadgraph_points")
    if rg is None:
        return {"map_available": False}
    out = {"map_available": True}
    for key in ("x", "y", "z", "dir_x", "dir_y", "types", "valid"):
        val = _maybe_get(rg, key)
        if val is not None:
            arr = _as_np(val)
            out[f"roadgraph_{key}_shape"] = list(arr.shape)
    return out


def _make_root_scene(state: Any, split: str, idx: int, cfg: dict[str, Any]) -> RootScene:
    history_steps = int(cfg.get("dataset", {}).get("history_steps", 11))
    states, object_ids, object_types, ego_index, dt = _extract_states(state, history_steps)
    scene_id = _scenario_id(state, idx)
    root = RootScene(
        scene_id=scene_id,
        split="val" if split in {"val", "mini"} else split if split in {"train", "test", "stress"} else "tiny_debug",
        source="womd_waymax",
        womd_version=str(cfg.get("dataset", {}).get("womd_version", "")),
        current_time_index=history_steps - 1,
        dt=dt,
        history_horizon_s=float((history_steps - 1) * dt),
        future_horizon_s=float(cfg.get("dataset", {}).get("future_horizon_s", 8.0)),
        ego_track_index=ego_index,
        agent_tracks=AgentTrackTensor(states=states, object_ids=object_ids, object_types=object_types),
        route_context=_extract_route_context(state, ego_index),
        map_features=_extract_map_summary(state),
        traffic_controls=None,
        metadata={"waymax_state_type": type(state).__name__},
    )
    return root


def _logged_future_candidate(root: RootScene, state: Any, future_steps: int) -> EgoCandidate:
    traj = _extract_sim_trajectory(state)
    ego = root.ego_track_index
    start = root.current_time_index + 1
    x = _as_np(_maybe_get(traj, "x"))[ego, start:start+future_steps]
    y = _as_np(_maybe_get(traj, "y"))[ego, start:start+future_steps]
    yaw = _as_np(_maybe_get(traj, "yaw", default=np.zeros_like(_maybe_get(traj, "x"))))[ego, start:start+future_steps]
    vx = _as_np(_maybe_get(traj, "vel_x", default=np.zeros_like(_maybe_get(traj, "x"))))[ego, start:start+future_steps]
    vy = _as_np(_maybe_get(traj, "vel_y", default=np.zeros_like(_maybe_get(traj, "x"))))[ego, start:start+future_steps]
    length = _as_np(_maybe_get(traj, "length", default=np.full_like(_maybe_get(traj, "x"), 4.5)))[ego, start:start+future_steps]
    width = _as_np(_maybe_get(traj, "width", default=np.full_like(_maybe_get(traj, "x"), 2.0)))[ego, start:start+future_steps]
    valid = _as_np(_maybe_get(traj, "valid", default=np.ones_like(_maybe_get(traj, "x"), dtype=bool)))[ego, start:start+future_steps]
    T = min(future_steps, len(x))
    if T < future_steps:
        pad = future_steps - T
        def p(a, fill=0): return np.pad(a[:T], (0, pad), constant_values=fill)
        x, y, yaw, vx, vy, length, width, valid = p(x), p(y), p(yaw), p(vx), p(vy), p(length, 4.5), p(width, 2.0), p(valid, False)
    speed = np.sqrt(vx * vx + vy * vy)
    z = np.zeros_like(x)
    states = np.stack([x, y, z, vx, vy, speed, yaw, length, width, valid.astype(np.float32)], axis=-1).astype(np.float32)
    # Transform to ego-centered t0 frame.
    origin = root.root_state(ego)
    states[:, 0] -= origin[0]
    states[:, 1] -= origin[1]
    states[:, 6] -= origin[6]
    return EgoCandidate("logged", "logged_anchor", states, nominal_cost=0.0, metadata={"is_logged_anchor": True})


def _perturb_candidate(base: EgoCandidate, cid: str, speed_scale: float, delay_steps: int, lateral_offset: float, cost: float) -> EgoCandidate:
    s = np.asarray(base.future_states_ego_frame, dtype=np.float32).copy()
    if delay_steps > 0:
        first = s[0].copy()
        s = np.concatenate([np.repeat(first[None, :], delay_steps, axis=0), s[:-delay_steps]], axis=0)
    s[:, 3:5] *= speed_scale
    s[:, 5] *= speed_scale
    if lateral_offset != 0.0:
        yaw = s[:, 6]
        normal = np.stack([-np.sin(yaw), np.cos(yaw)], axis=-1)
        s[:, :2] += normal * float(lateral_offset)
    return EgoCandidate(cid, "timing_assertiveness_perturbation", s, nominal_cost=float(cost), metadata={
        "speed_scale": speed_scale, "delay_steps": delay_steps, "lateral_offset": lateral_offset,
    })


def _generate_candidates(root: RootScene, state: Any, cfg: dict[str, Any]) -> list[EgoCandidate]:
    data = cfg.get("dataset", {})
    future_steps = int(data.get("future_steps", 80))
    base = _logged_future_candidate(root, state, future_steps)
    candidates = [base]
    speed_scales = data.get("speed_scales", [0.75, 0.9, 1.0, 1.1])
    delays = data.get("timing_delays_s", [-0.2, 0.0, 0.2, 0.4, 0.8])
    lateral_offsets = data.get("lateral_offsets_m", [0.0])
    dt = float(root.dt)
    max_k = int(data.get("candidates_per_group", 24))
    for ss in speed_scales:
        for ds in delays:
            for lo in lateral_offsets:
                if len(candidates) >= max_k:
                    return candidates
                delay_steps = max(0, int(round(float(ds) / dt)))
                cid = f"cand_v{float(ss):.2f}_d{float(ds):+.1f}_l{float(lo):+.1f}"
                if cid == "logged":
                    continue
                cost = abs(float(ss) - 1.0) + 0.1 * abs(float(ds)) + 0.05 * abs(float(lo))
                candidates.append(_perturb_candidate(base, cid, float(ss), delay_steps, float(lo), cost))
    return candidates


def _select_relevant_agents(root: RootScene, candidates: list[EgoCandidate], cfg: dict[str, Any]) -> list[int]:
    tracks = root.agent_tracks
    if tracks is None:
        return []
    radius = float(cfg.get("dataset", {}).get("relevant_radius_m", 60.0))
    ego = root.ego_track_index
    ego0 = root.root_state(ego)
    relevant: list[int] = []
    for i in range(tracks.states.shape[0]):
        if i == ego:
            continue
        st = root.root_state(i)
        if st[9] <= 0.5:
            continue
        if np.linalg.norm(st[:2] - ego0[:2]) <= radius:
            relevant.append(i)
    return relevant[: int(cfg.get("dataset", {}).get("max_agents_per_group", 8))]


def _support_query_split(candidates: list[EgoCandidate], cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    ids = [c.candidate_id for c in candidates]
    if len(ids) < 2:
        raise RuntimeError("Need at least two candidates for support/query split")
    query_fraction = float(cfg.get("dataset", {}).get("query_fraction", 0.35))
    qn = max(1, int(round(len(ids) * query_fraction)))
    query = ids[-qn:]
    support = [x for x in ids if x not in set(query)]
    if not support:
        support, query = ids[:1], ids[1:]
    return support, query


def _boundary_pairs(group: SameRootGroup) -> list[BoundaryPair]:
    pairs: list[BoundaryPair] = []
    cands = group.candidates
    for aid in group.relevant_agent_ids:
        for k in range(len(cands) - 1):
            obs_a = [o for key, o in group.observations.items() if key[0] == cands[k].candidate_id and key[1] == aid]
            obs_b = [o for key, o in group.observations.items() if key[0] == cands[k+1].candidate_id and key[1] == aid]
            if not obs_a or not obs_b:
                continue
            pa = np.mean([o.branch_probs for o in obs_a], axis=0)
            pb = np.mean([o.branch_probs for o in obs_b], axis=0)
            ba = float(np.mean([o.burden for o in obs_a]))
            bb = float(np.mean([o.burden for o in obs_b]))
            ha = float(np.mean([o.safety_margin for o in obs_a]))
            hb = float(np.mean([o.safety_margin for o in obs_b]))
            tv = 0.5 * float(np.abs(pa - pb).sum())
            dist = tv + 0.25 * abs(ba - bb) + 0.25 * abs(ha - hb)
            boundary = float(int(pa.argmax() != pb.argmax()) or int((ha > 0) != (hb > 0)))
            pairs.append(BoundaryPair(cands[k].candidate_id, cands[k+1].candidate_id, aid, dist, boundary, True))
    return pairs


def _rollout_surrogate_required_error() -> RuntimeError:
    return RuntimeError(
        "The adapter reached the rollout step without a project-specific Waymax reactive rollout implementation. "
        "Do not use logged futures as response labels.  Implement `reactive_rollout_fn` in this file for your "
        "installed Waymax version, or pass precomputed reactive rollouts through config['adapter']['rollout_cache']."
    )


def _load_cached_rollout(cache_root: str | None, scenario_id: str, candidate_id: str, variant_id: str, agent_id: int) -> np.ndarray | None:
    if not cache_root:
        return None
    p = Path(cache_root) / scenario_id / f"{candidate_id}__{variant_id}__agent{agent_id}.npy"
    if not p.exists():
        return None
    return np.load(p).astype(np.float32)


def build_groups(
    *,
    womd_pattern: str,
    split: str,
    config: dict[str, Any],
    max_scenarios: int | None = None,
    num_workers: int = 1,
) -> Iterator[SameRootGroup]:
    """Yield SameRootGroup objects from local WOMD/Waymax data.

    The safest current workflow is two-stage:
      1. run your local Waymax rollout script to populate
         config.adapter.rollout_cache with candidate/variant/agent trajectories;
      2. call this adapter to assemble schema-correct MFRP groups.

    This design prevents accidental use of log playback as response supervision.
    """
    wx_config, dataloader = _require_waymax()
    paths = sorted(glob.glob(womd_pattern))
    if not paths:
        raise FileNotFoundError(f"No WOMD TFExample files match: {womd_pattern}")
    data_cfg = config.get("dataset", {})
    adapter_cfg = config.get("adapter", {})
    rollout_cache = adapter_cfg.get("rollout_cache")
    if not rollout_cache:
        raise RuntimeError(
            "config.adapter.rollout_cache is required for this safe adapter. Populate it with real Waymax reactive "
            "rollouts first, or extend this file's rollout section for your exact Waymax env/action API."
        )
    max_objects = int(data_cfg.get("max_num_objects", 128))
    # Waymax docs show config.DatasetConfig(path=..., data_format=TFRECORD, max_num_objects=...).
    if hasattr(wx_config, "DatasetConfig"):
        ds_cfg = wx_config.DatasetConfig(
            path=",".join(paths),
            max_num_objects=max_objects,
            data_format=getattr(wx_config, "DataFormat", object()).TFRECORD if hasattr(getattr(wx_config, "DataFormat", object()), "TFRECORD") else None,
        )
    else:
        base = getattr(wx_config, "WOD_1_1_0_TRAINING")
        ds_cfg = dataclasses.replace(base, path=",".join(paths), max_num_objects=max_objects)
    iterator = dataloader.simulator_state_generator(config=ds_cfg)
    variants = [VariantSpec(**v) if isinstance(v, dict) else VariantSpec(str(v)) for v in adapter_cfg.get("variants", [])] or list(DEFAULT_VARIANTS)
    future_steps = int(data_cfg.get("future_steps", 80))
    count = 0
    for idx, state in enumerate(iterator):
        if max_scenarios is not None and count >= max_scenarios:
            break
        root = _make_root_scene(state, split, idx, config)
        candidates = _generate_candidates(root, state, config)
        agents = _select_relevant_agents(root, candidates, config)
        if not agents or len(candidates) < 2:
            continue
        root_hash = root_scene_hash(root)
        support, query = _support_query_split(candidates, config)
        group = SameRootGroup(
            scenario_id=root.scene_id,
            root_hash=root_hash,
            root_scene=root,
            candidates=candidates,
            relevant_agent_ids=agents,
            rollout_variants=[v.name for v in variants],
            metadata={
                "support_candidate_ids": support,
                "query_candidate_ids": query,
                "uses_log_playback_for_response": False,
                "adapter": "my_project.mfrp_waymax_adapter",
            },
        )
        all_obs = []
        for c in candidates:
            for aid in agents:
                hist = root.agent_tracks.states[aid]
                neutral_ref = constant_velocity_extrapolate(hist, future_steps, root.dt)
                interaction = build_interaction_region(c.future_states_ego_frame, neutral_ref, dt=root.dt)
                pr = compute_priority_score(interaction.features)
                neutral_agent_traj = neutral_ref
                if neutral_agent_traj is None:
                    continue
                for v in variants:
                    agent_traj = _load_cached_rollout(rollout_cache, root.scene_id, c.candidate_id, v.name, aid)
                    if agent_traj is None:
                        raise _rollout_surrogate_required_error()
                    n = min(len(c.future_states_ego_frame), len(agent_traj))
                    inter_feat = dict(interaction.features)
                    inter_feat.update({"tau_i_k_in": inter_feat.get("tau_i0_in", future_steps * root.dt + 1.0), "tau_i_k_out": inter_feat.get("tau_i0_out", future_steps * root.dt + 1.0)})
                    obs = make_response_observation(
                        root.scene_id, root_hash, c.candidate_id, aid, v.name,
                        c.future_states_ego_frame[:n], agent_traj[:n], neutral_agent_traj[:n],
                        inter_feat, pr.score, pr.confidence, dt=root.dt,
                    )
                    group.observations[(c.candidate_id, aid, v.name)] = obs
                    all_obs.append(obs)
        for c in candidates:
            for aid in agents:
                lab = coercion_witness_label(all_obs, c.candidate_id, aid, root_hash, root.scene_id)
                group.cw_labels[(c.candidate_id, aid)] = lab
        group.boundary_pairs = _boundary_pairs(group)
        if not group.observations:
            continue
        count += 1
        yield group
