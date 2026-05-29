"""WOMD/Waymax adapter for MFRP same-root intervention groups.

The original bundle only accepted a precomputed ``config.adapter.rollout_cache``
and raised at dataset-generation time.  This version keeps cache support, but
also performs online same-root reactive rollouts from WOMD scenarios loaded by
Waymax.  The online rollout is intentionally local and reproducible: the root
scene comes from Waymax/WOMD, the ego candidate is forcibly injected, and each
surrounding vehicle follows its logged Waymax/WOMD route with an IDM-style
reactive speed update against the counterfactual ego trajectory.  Multiple IDM
parameterizations provide the policy variants needed by the paper's ceding vs
non-ceding evidence.

The implementation does not use log playback as response supervision.  Logged
future trajectories are used only as route geometry for the reactive agents,
which is the same design assumption as Waymax's IDMRoutePolicy: follow the
agent's logged route while updating speed reactively.
"""
from __future__ import annotations

from dataclasses import dataclass
import dataclasses
from pathlib import Path
from typing import Any, Iterator
import glob
import math
import numpy as np

from mfrp.data.scene_schema import AgentTrackTensor, EgoCandidate, RootScene, RouteContext
from mfrp.data.schema import BoundaryPair, SameRootGroup, root_scene_hash
from mfrp.data.interaction_region import build_interaction_region, constant_velocity_extrapolate
from mfrp.data.label_extraction import make_response_observation
from mfrp.data.priority import compute_priority_score


@dataclass(frozen=True)
class VariantSpec:
    name: str
    aggressiveness: float = 1.0
    desired_speed_scale: float = 1.0
    min_gap_scale: float = 1.0
    safe_time_headway: float = 1.5
    max_accel: float = 2.0
    max_decel: float = 4.0


DEFAULT_VARIANTS = (
    VariantSpec("neutral_idm", aggressiveness=1.0, desired_speed_scale=1.0, min_gap_scale=1.0, safe_time_headway=1.5),
    VariantSpec("conservative_idm", aggressiveness=0.7, desired_speed_scale=0.9, min_gap_scale=1.35, safe_time_headway=2.1),
    VariantSpec("assertive_idm", aggressiveness=1.3, desired_speed_scale=1.12, min_gap_scale=0.70, safe_time_headway=0.9),
)


def _require_waymax():
    try:
        from waymax import config as wx_config  # type: ignore
        from waymax import dataloader  # type: ignore
    except Exception as e:  # pragma: no cover - depends on user's local env
        raise RuntimeError(
            "Waymax is required for this adapter. Install it with\n"
            "  pip install git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax\n"
            "and make sure JAX/TensorFlow can read your local WOMD TFExample files."
        ) from e
    return wx_config, dataloader


def _as_np(x: Any) -> np.ndarray:
    try:  # JAX DeviceArray -> ndarray
        import jax  # type: ignore
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
    # Avoid boolean-evaluating JAX/Waymax dataclasses or arrays.
    for name in ("sim_trajectory", "log_trajectory", "trajectory"):
        val = _maybe_get(state, name)
        if val is not None:
            return val
    return None


def _trajectory_arrays(state: Any) -> dict[str, np.ndarray]:
    traj = _extract_sim_trajectory(state)
    if traj is None:
        raise RuntimeError("Could not find sim/log trajectory in Waymax state")
    x = _as_np(_maybe_get(traj, "x"))
    y = _as_np(_maybe_get(traj, "y"))
    if x.ndim != 2:
        raise RuntimeError(f"Expected Waymax trajectory x with shape [N,T], got {x.shape}")
    zeros = np.zeros_like(x, dtype=np.float32)
    ones_l = np.full_like(x, 4.5, dtype=np.float32)
    ones_w = np.full_like(x, 2.0, dtype=np.float32)
    return {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "z": _as_np(_maybe_get(traj, "z", default=zeros)).astype(np.float32),
        "vx": _as_np(_maybe_get(traj, "vel_x", default=zeros)).astype(np.float32),
        "vy": _as_np(_maybe_get(traj, "vel_y", default=zeros)).astype(np.float32),
        "yaw": _as_np(_maybe_get(traj, "yaw", default=zeros)).astype(np.float32),
        "length": _as_np(_maybe_get(traj, "length", default=ones_l)).astype(np.float32),
        "width": _as_np(_maybe_get(traj, "width", default=ones_w)).astype(np.float32),
        "valid": _as_np(_maybe_get(traj, "valid", default=np.ones_like(x, dtype=bool))).astype(bool),
        "object_id": _as_np(_maybe_get(traj, "object_id", default=np.arange(x.shape[0]))).reshape(-1),
        "object_type": _as_np(_maybe_get(traj, "object_type", default=np.zeros(x.shape[0], dtype=np.int32))).reshape(-1),
    }


def _scenario_id(state: Any, fallback: int) -> str:
    sid = _maybe_get(state, "scenario_id") or _maybe_get(state, "object_metadata", "scenario_id")
    if sid is None:
        return f"scenario_{fallback:08d}"
    arr = np.asarray(sid)
    if arr.shape == ():
        v = arr.item()
        return v.decode() if isinstance(v, bytes) else str(v)
    v = arr.reshape(-1)[0]
    return v.decode() if isinstance(v, bytes) else str(v)


def _sdc_index(state: Any) -> int:
    try:
        arr = _as_np(_maybe_get(state, "sdc_track_index", default=0))
        return int(arr.reshape(())) if arr.size else 0
    except Exception:
        return 0


def _rot(xy: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, s], [-s, c]], dtype=np.float32)  # world -> ego-t0
    return np.asarray(xy, dtype=np.float32) @ R.T


def _to_ego_frame_states(world: np.ndarray, ego0: np.ndarray) -> np.ndarray:
    out = np.asarray(world, dtype=np.float32).copy()
    out_xy = _rot(out[..., :2] - ego0[:2], float(ego0[6]))
    out[..., 0:2] = out_xy
    if out.shape[-1] >= 5:
        out[..., 3:5] = _rot(out[..., 3:5], float(ego0[6]))
    if out.shape[-1] >= 7:
        out[..., 6] = out[..., 6] - float(ego0[6])
    return out


def _pack_states(arr: dict[str, np.ndarray], obj: int, sl: slice) -> np.ndarray:
    x, y = arr["x"][obj, sl], arr["y"][obj, sl]
    vx, vy = arr["vx"][obj, sl], arr["vy"][obj, sl]
    speed = np.sqrt(vx * vx + vy * vy)
    yaw = arr["yaw"][obj, sl]
    length = arr["length"][obj, sl]
    width = arr["width"][obj, sl]
    valid = arr["valid"][obj, sl].astype(np.float32)
    return np.stack([x, y, np.zeros_like(x), vx, vy, speed, yaw, length, width, valid], axis=-1).astype(np.float32)


def _extract_route_context(state: Any) -> RouteContext:
    # Current schema only carries route_features.  Keep metadata compact and safe.
    return RouteContext(route_features=None, metadata={"route_source": "waymax_womd_logged_route_geometry"})


def _extract_map_summary(state: Any) -> np.ndarray | None:
    rg = _maybe_get(state, "roadgraph_points")
    if rg is None:
        return None
    x = _maybe_get(rg, "x")
    y = _maybe_get(rg, "y")
    valid = _maybe_get(rg, "valid")
    if x is None or y is None:
        return None
    xy = np.stack([_as_np(x).reshape(-1), _as_np(y).reshape(-1)], axis=-1).astype(np.float32)
    if valid is not None:
        xy = xy[_as_np(valid).reshape(-1).astype(bool)]
    return xy[:512]


def _make_root_scene(state: Any, split: str, idx: int, cfg: dict[str, Any]) -> tuple[RootScene, dict[str, np.ndarray], np.ndarray]:
    data = cfg.get("dataset", {})
    history_steps = int(data.get("history_steps", 11))
    arr = _trajectory_arrays(state)
    N, Tall = arr["x"].shape
    hist = min(history_steps, Tall)
    ego = _sdc_index(state)
    current = hist - 1
    ego0_world = _pack_states(arr, ego, slice(current, current + 1))[0]
    hist_world = np.stack([_pack_states(arr, i, slice(0, hist)) for i in range(N)], axis=0)
    hist_ego = _to_ego_frame_states(hist_world, ego0_world)
    # Model tensors default to state_dim=5. Keep richer 10D states in agent_tracks.
    history = hist_ego[..., [0, 1, 3, 4, 6]].astype(np.float32)
    history_mask = arr["valid"][:, :hist].astype(bool)
    scene_id = _scenario_id(state, idx)
    split_norm = "val" if split in {"val", "mini"} else split if split in {"train", "test", "stress"} else "debug"
    root = RootScene(
        scene_id=scene_id,
        t0=current,
        history=history,
        history_mask=history_mask,
        ego_index=ego,
        map_features=_extract_map_summary(state),
        traffic_controls=None,
        route_features=None,
        metadata={"source": "womd_waymax", "womd_version": str(data.get("womd_version", "")), "waymax_state_type": type(state).__name__},
        dt=0.1,
        current_time_index=current,
        agent_tracks=[AgentTrackTensor(str(int(arr["object_id"][i])), hist_ego[i], mask=history_mask[i], metadata={"object_type": str(int(arr["object_type"][i]))}) for i in range(N)],
        route_context=_extract_route_context(state),
    )
    return root, arr, ego0_world


def _future_world(arr: dict[str, np.ndarray], obj: int, start: int, future_steps: int) -> np.ndarray:
    T = arr["x"].shape[1]
    sl = slice(start, min(T, start + future_steps))
    s = _pack_states(arr, obj, sl)
    if len(s) < future_steps:
        pad_n = future_steps - len(s)
        if len(s) == 0:
            last = _pack_states(arr, obj, slice(max(0, start - 1), max(0, start)))
            last = last[0] if len(last) else np.zeros(10, dtype=np.float32)
        else:
            last = s[-1]
        pads = []
        for k in range(pad_n):
            nxt = last.copy()
            nxt[0] += nxt[3] * 0.1 * (k + 1)
            nxt[1] += nxt[4] * 0.1 * (k + 1)
            nxt[9] = 0.0
            pads.append(nxt)
        s = np.concatenate([s, np.stack(pads, axis=0)], axis=0) if pads else s
    return s.astype(np.float32)


def _candidate_features(states: np.ndarray, dt: float) -> np.ndarray:
    xy = states[:, :2]
    dist = float(np.linalg.norm(xy[-1] - xy[0])) if len(xy) else 0.0
    speed = states[:, 5] if states.shape[-1] > 5 else np.linalg.norm(states[:, 3:5], axis=-1)
    acc = np.gradient(speed, dt) if len(speed) > 1 else np.zeros_like(speed)
    feat = np.zeros(20, dtype=np.float32)
    feat[:8] = [xy[0, 0], xy[0, 1], xy[-1, 0], xy[-1, 1], float(np.nanmean(speed)), float(np.nanmax(speed)), float(np.nanmean(acc)), dist]
    return feat


def _logged_future_candidate(root: RootScene, arr: dict[str, np.ndarray], ego0_world: np.ndarray, future_steps: int) -> EgoCandidate:
    ego = root.ego_index
    start = int(root.current_time_index or root.t0) + 1
    world = _future_world(arr, ego, start, future_steps)
    states = _to_ego_frame_states(world, ego0_world)
    return EgoCandidate("logged", states, features=_candidate_features(states, root.dt), nominal_cost=0.0, valid=bool(states[:, 9].any()), metadata={"is_logged_anchor": True}, family="logged_anchor")


def _shift_along_heading(states: np.ndarray, speed_scale: float, delay_steps: int, advance_steps: int, lateral_offset: float) -> np.ndarray:
    s = np.asarray(states, dtype=np.float32).copy()
    if delay_steps > 0:
        s = np.concatenate([np.repeat(s[:1], delay_steps, axis=0), s[:-delay_steps]], axis=0)
    if advance_steps > 0:
        tail = np.repeat(s[-1:], advance_steps, axis=0)
        # Constant-velocity extension for positive timing advances.
        for k in range(advance_steps):
            tail[k, 0] = s[-1, 0] + s[-1, 3] * 0.1 * (k + 1)
            tail[k, 1] = s[-1, 1] + s[-1, 4] * 0.1 * (k + 1)
        s = np.concatenate([s[advance_steps:], tail], axis=0)
    s[:, 3:5] *= float(speed_scale)
    s[:, 5] *= float(speed_scale)
    if lateral_offset != 0.0:
        yaw = s[:, 6]
        normal = np.stack([-np.sin(yaw), np.cos(yaw)], axis=-1)
        s[:, :2] += normal * float(lateral_offset)
    return s


def _perturb_candidate(base: EgoCandidate, cid: str, speed_scale: float, delay_s: float, lateral_offset: float, cost: float, dt: float) -> EgoCandidate:
    raw = np.asarray(base.trajectory, dtype=np.float32)
    delay_steps = max(0, int(round(delay_s / dt)))
    advance_steps = max(0, int(round(-delay_s / dt)))
    states = _shift_along_heading(raw, speed_scale, delay_steps, advance_steps, lateral_offset)
    return EgoCandidate(cid, states, features=_candidate_features(states, dt), nominal_cost=float(cost), valid=bool(states[:, 9].any()), metadata={"speed_scale": speed_scale, "delay_s": delay_s, "lateral_offset": lateral_offset}, family="timing_assertiveness_perturbation")


def _generate_candidates(root: RootScene, arr: dict[str, np.ndarray], ego0_world: np.ndarray, cfg: dict[str, Any]) -> list[EgoCandidate]:
    data = cfg.get("dataset", {})
    future_steps = int(data.get("future_steps", cfg.get("tensor", {}).get("future_steps", 80)))
    base = _logged_future_candidate(root, arr, ego0_world, future_steps)
    candidates = [base]
    speed_scales = data.get("speed_scales", [0.75, 0.9, 1.0, 1.1, 1.25])
    delays = data.get("timing_delays_s", [-0.4, -0.2, 0.0, 0.2, 0.4, 0.8])
    lateral_offsets = data.get("lateral_offsets_m", [0.0])
    max_k = int(data.get("candidates_per_group", 24))
    for ss in speed_scales:
        for ds in delays:
            for lo in lateral_offsets:
                if len(candidates) >= max_k:
                    return candidates
                if abs(float(ss) - 1.0) < 1e-6 and abs(float(ds)) < 1e-6 and abs(float(lo)) < 1e-6:
                    continue
                cid = f"cand_v{float(ss):.2f}_d{float(ds):+.1f}_l{float(lo):+.1f}"
                cost = abs(float(ss) - 1.0) + 0.15 * abs(float(ds)) + 0.05 * abs(float(lo))
                candidates.append(_perturb_candidate(base, cid, float(ss), float(ds), float(lo), cost, root.dt))
    return candidates


def _select_relevant_agents(root: RootScene, cfg: dict[str, Any]) -> list[str]:
    radius = float(cfg.get("dataset", {}).get("relevant_radius_m", 60.0))
    max_agents = int(cfg.get("dataset", {}).get("max_agents_per_group", 8))
    ego = int(root.ego_index)
    ego0 = root.history[ego, -1, :2]
    candidates: list[tuple[float, str]] = []
    for i in range(root.history.shape[0]):
        if i == ego or not root.history_mask[i, -1]:
            continue
        d = float(np.linalg.norm(root.history[i, -1, :2] - ego0))
        if d <= radius:
            candidates.append((d, str(i)))
    candidates.sort(key=lambda x: x[0])
    return [a for _, a in candidates[:max_agents]]


def _support_query_split(candidates: list[EgoCandidate], cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    ids = [c.candidate_id for c in candidates]
    if len(ids) < 2:
        raise RuntimeError("Need at least two candidates for support/query split")
    qf = float(cfg.get("dataset", {}).get("query_fraction", 0.35))
    qn = max(1, int(round(len(ids) * qf)))
    query = ids[-qn:]
    support = [x for x in ids if x not in set(query)]
    if not support:
        support, query = ids[:1], ids[1:]
    return support, query


def _load_cached_rollout(cache_root: str | None, scenario_id: str, candidate_id: str, variant_id: str, agent_id: str) -> np.ndarray | None:
    if not cache_root:
        return None
    for name in (f"{candidate_id}__{variant_id}__agent{agent_id}.npy", f"{candidate_id}__{variant_id}__agent{agent_id}.npz"):
        p = Path(cache_root) / scenario_id / name
        if p.exists():
            if p.suffix == ".npz":
                z = np.load(p)
                key = "trajectory" if "trajectory" in z.files else z.files[0]
                return np.asarray(z[key], dtype=np.float32)
            return np.load(p).astype(np.float32)
    return None


def _idm_reactive_rollout(agent_base: np.ndarray, ego_traj: np.ndarray, variant: VariantSpec, *, dt: float = 0.1) -> np.ndarray:
    """Route-following IDM response in ego-t0 coordinates.

    The agent follows its logged future polyline but changes its longitudinal
    speed in response to the counterfactual ego.  This is the minimum viable
    online Waymax/WOMD response supervisor: route geometry from Waymax, ego
    intervention injected, non-ego response from IDM variants.
    """
    base = np.asarray(agent_base, dtype=np.float32)
    ego = np.asarray(ego_traj, dtype=np.float32)
    T = min(len(base), len(ego))
    if T == 0:
        return base
    out = base[:T].copy()
    path_xy = base[:T, :2]
    seg = np.diff(path_xy, axis=0, prepend=path_xy[:1])
    seg_len = np.linalg.norm(seg, axis=-1)
    arclen = np.cumsum(seg_len)
    total = float(max(arclen[-1], 1e-3))
    speed0 = float(max(base[0, 5] if base.shape[-1] > 5 else np.linalg.norm(base[0, 3:5]), 0.0))
    desired = max(0.5, float(np.nanpercentile(base[:T, 5], 75) if base.shape[-1] > 5 else speed0) * variant.desired_speed_scale)
    v = speed0
    s_pos = 0.0
    # Decide whether the ego is plausibly in front at each step by projection on local route tangent.
    for t in range(T):
        idx = int(np.searchsorted(arclen, s_pos, side="left"))
        idx = min(max(idx, 0), T - 1)
        pos = path_xy[idx]
        if idx < T - 1:
            tangent = path_xy[idx + 1] - path_xy[idx]
        else:
            tangent = path_xy[idx] - path_xy[max(0, idx - 1)]
        nrm = float(np.linalg.norm(tangent))
        if nrm < 1e-4:
            yaw = float(base[idx, 6]) if base.shape[-1] > 6 else 0.0
            tangent = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        else:
            tangent = tangent / nrm
        rel = ego[t, :2] - pos
        longitudinal = float(rel @ tangent)
        lateral = float(abs(rel @ np.array([-tangent[1], tangent[0]], dtype=np.float32)))
        ego_speed = float(ego[t, 5] if ego.shape[-1] > 5 else np.linalg.norm(ego[t, 3:5]))
        lead_gap = longitudinal - 0.5 * float(base[idx, 7] if base.shape[-1] > 7 else 4.5) - 0.5 * float(ego[t, 7] if ego.shape[-1] > 7 else 4.5)
        same_corridor = lateral < (float(base[idx, 8] if base.shape[-1] > 8 else 2.0) + float(ego[t, 8] if ego.shape[-1] > 8 else 2.0) + 2.0)
        # IDM acceleration.
        a_free = variant.max_accel * (1.0 - (max(v, 0.0) / max(desired, 0.1)) ** 4)
        a = a_free
        if same_corridor and lead_gap > 0.0 and lead_gap < 45.0:
            min_gap = 2.0 * variant.min_gap_scale
            dv = v - ego_speed
            s_star = min_gap + max(0.0, v * variant.safe_time_headway + v * dv / (2.0 * math.sqrt(max(variant.max_accel * variant.max_decel, 1e-3))))
            a = variant.max_accel * (1.0 - (v / max(desired, 0.1)) ** 4 - (s_star / max(lead_gap, 0.2)) ** 2)
        a = float(np.clip(a * variant.aggressiveness, -variant.max_decel, variant.max_accel))
        v = max(0.0, v + a * dt)
        s_pos = min(total, s_pos + v * dt)
        idx = int(np.searchsorted(arclen, s_pos, side="left"))
        idx = min(max(idx, 0), T - 1)
        out[t] = base[idx]
        out[t, 5] = v
        if t > 0:
            vel = (out[t, :2] - out[t - 1, :2]) / dt
            out[t, 3:5] = vel
            if np.linalg.norm(vel) > 1e-3:
                out[t, 6] = math.atan2(float(vel[1]), float(vel[0]))
        out[t, 9] = 1.0
    return out.astype(np.float32)


def _agent_neutral_future(root: RootScene, arr: dict[str, np.ndarray], ego0_world: np.ndarray, agent_id: str, future_steps: int) -> np.ndarray:
    idx = int(agent_id)
    start = int(root.current_time_index or root.t0) + 1
    world = _future_world(arr, idx, start, future_steps)
    neutral = _to_ego_frame_states(world, ego0_world)
    if not neutral[:, 9].any():
        # Use root history only when logged future is absent.
        neutral = constant_velocity_extrapolate(root.agent_tracks[idx].states, future_steps, root.dt)
    return neutral.astype(np.float32)


def _make_agent_interaction_features(cand: EgoCandidate, neutral_ref: np.ndarray, agent_traj: np.ndarray, dt: float) -> dict[str, float]:
    base_inter = build_interaction_region(cand.trajectory, neutral_ref, dt=dt)
    resp_inter = build_interaction_region(cand.trajectory, agent_traj, dt=dt)
    feat = dict(base_inter.features)
    feat.update({
        "tau_i0_in": base_inter.tau_agent_in,
        "tau_i0_out": base_inter.tau_agent_in,
        "tau_i_k_in": resp_inter.tau_agent_in,
        "tau_i_k_out": resp_inter.tau_agent_in,
        "tau_ego_in": resp_inter.tau_ego_in,
        "entry_time_gap": base_inter.tau_agent_in - base_inter.tau_ego_in if np.isfinite(base_inter.tau_agent_in) and np.isfinite(base_inter.tau_ego_in) else 0.0,
        "min_distance": resp_inter.min_distance,
    })
    return feat


def _boundary_pairs(group: SameRootGroup) -> list[BoundaryPair]:
    pairs: list[BoundaryPair] = []
    cands = group.candidates
    for aid in group.relevant_agent_ids:
        for k in range(len(cands) - 1):
            obs_a = [o for key, o in group.observations.items() if key[0] == cands[k].candidate_id and key[1] == aid]
            obs_b = [o for key, o in group.observations.items() if key[0] == cands[k + 1].candidate_id and key[1] == aid]
            if not obs_a or not obs_b:
                continue
            pa = np.mean([o.branch_probs for o in obs_a], axis=0)
            pb = np.mean([o.branch_probs for o in obs_b], axis=0)
            ba = float(np.mean([o.burden for o in obs_a])); bb = float(np.mean([o.burden for o in obs_b]))
            ha = float(np.mean([o.safety_margin for o in obs_a])); hb = float(np.mean([o.safety_margin for o in obs_b]))
            tv = 0.5 * float(np.abs(pa - pb).sum())
            dist = tv + 0.25 * abs(ba - bb) + 0.25 * abs(ha - hb)
            pairs.append(BoundaryPair(aid, cands[k].candidate_id, cands[k + 1].candidate_id, max(dist, 1e-3)))
    return pairs


def _parse_variants(adapter_cfg: dict[str, Any]) -> list[VariantSpec]:
    raw = adapter_cfg.get("variants") or adapter_cfg.get("policy_variants") or []
    if not raw:
        return list(DEFAULT_VARIANTS)
    out: list[VariantSpec] = []
    default_by_name = {v.name: v for v in DEFAULT_VARIANTS}
    for v in raw:
        if isinstance(v, dict):
            out.append(VariantSpec(**v))
        else:
            name = str(v)
            out.append(default_by_name.get(name, VariantSpec(name)))
    return out


def build_groups(*, womd_pattern: str, split: str, config: dict[str, Any], max_scenarios: int | None = None, num_workers: int = 1) -> Iterator[SameRootGroup]:
    """Yield schema-valid SameRootGroup objects from WOMD loaded by Waymax."""
    wx_config, dataloader = _require_waymax()
    paths = sorted(glob.glob(womd_pattern))
    effective_womd_pattern = womd_pattern
    if not paths and womd_pattern.endswith(".tfrecord"):
        effective_womd_pattern = womd_pattern + "*"
    print(f"[MFRP] Matched {len(paths)} WOMD TFRecord shards with pattern: {effective_womd_pattern}")
    if not paths:
        raise FileNotFoundError(f"No WOMD TFExample files match: {womd_pattern}")
    data_cfg = config.get("dataset", {})
    adapter_cfg = config.get("adapter", {})
    rollout_cache = adapter_cfg.get("rollout_cache")
    generate_online = bool(adapter_cfg.get("generate_online_rollouts", config.get("rollout", {}).get("generate_online", True)))
    max_objects = int(data_cfg.get("max_num_objects", 128))
    if hasattr(wx_config, "DatasetConfig"):
        kwargs = {"path": effective_womd_pattern, "max_num_objects": max_objects}
        if hasattr(wx_config, "DataFormat") and hasattr(wx_config.DataFormat, "TFRECORD"):
            kwargs["data_format"] = wx_config.DataFormat.TFRECORD
        ds_cfg = wx_config.DatasetConfig(**kwargs)
    else:
        base = getattr(wx_config, "WOD_1_1_0_TRAINING")
        ds_cfg = dataclasses.replace(base, path=effective_womd_pattern, max_num_objects=max_objects)
    iterator = dataloader.simulator_state_generator(config=ds_cfg)
    variants = _parse_variants(adapter_cfg or config.get("rollout", {}))
    future_steps = int(data_cfg.get("future_steps", config.get("tensor", {}).get("future_steps", 80)))
    count = 0
    for idx, state in enumerate(iterator):
        if max_scenarios is not None and count >= max_scenarios:
            break
        root, arr, ego0_world = _make_root_scene(state, split, idx, config)
        candidates = _generate_candidates(root, arr, ego0_world, config)
        agents = _select_relevant_agents(root, config)
        if not agents or len(candidates) < 2:
            continue
        support, query = _support_query_split(candidates, config)
        group = SameRootGroup(
            root_scene=root,
            candidates=candidates,
            relevant_agent_ids=agents,
            rollout_variants=[v.name for v in variants],
            observations={},
            metadata={
                "support_candidate_ids": support,
                "query_candidate_ids": query,
                "uses_log_playback_for_response": False,
                "reactive_rollout_backend": "waymax_womd_route_idm_online" if generate_online else "cache",
                "adapter": "examples.mfrp_waymax_adapter",
            },
        )
        all_obs = []
        for c in candidates:
            c_meta = dict(c.metadata)
            c_meta.setdefault("agent_features", {})
            for aid in agents:
                neutral_ref = _agent_neutral_future(root, arr, ego0_world, aid, future_steps)
                pre_feat = _make_agent_interaction_features(c, neutral_ref, neutral_ref, root.dt)
                pr = compute_priority_score(pre_feat)
                c_meta["agent_features"][aid] = {"interaction_features": np.array([
                    pre_feat.get("tau_ego_in", 0.0), pre_feat.get("tau_i0_in", 0.0), pre_feat.get("entry_time_gap", 0.0),
                    pre_feat.get("min_distance", 0.0), pr.score, pr.confidence,
                ], dtype=np.float32)}
                for v in variants:
                    agent_traj = _load_cached_rollout(rollout_cache, root.scene_id, c.candidate_id, v.name, aid)
                    if agent_traj is None:
                        if not generate_online:
                            raise RuntimeError(
                                "Missing rollout_cache item and adapter.generate_online_rollouts=false. "
                                "Either enable online Waymax/WOMD IDM rollout or provide cache files."
                            )
                        agent_traj = _idm_reactive_rollout(neutral_ref, c.trajectory, v, dt=root.dt)
                    n = min(len(c.trajectory), len(agent_traj), len(neutral_ref))
                    inter_feat = _make_agent_interaction_features(c, neutral_ref[:n], agent_traj[:n], root.dt)
                    obs = make_response_observation(
                        root.scene_id, root.root_hash, c.candidate_id, aid, v.name,
                        c.trajectory[:n], agent_traj[:n], neutral_ref[:n], inter_feat,
                        pr.score, pr.confidence, dt=root.dt,
                    )
                    group.observations[(c.candidate_id, aid, v.name)] = obs
                    all_obs.append(obs)
            # dataclass is frozen; replace candidate metadata with preexec per-agent features.
            object.__setattr__(c, "metadata", c_meta)
        # Fill CW labels back into observations and group metadata for tensor collation.
        from mfrp.data.label_extraction import coercion_witness_label
        cw_labels: dict[tuple[str, str], Any] = {}
        for c in candidates:
            for aid in agents:
                lab = coercion_witness_label(all_obs, c.candidate_id, aid, root.root_hash, root.scene_id)
                cw_labels[(c.candidate_id, aid)] = lab
                for v in variants:
                    key = (c.candidate_id, aid, v.name)
                    obs = group.observations.get(key)
                    if obs is not None:
                        object.__setattr__(obs, "cw_soft_label", float(lab.soft_label))
                        object.__setattr__(obs, "cw_confidence", float(lab.confidence))
        group.metadata["cw_labels"] = {f"{k[0]}|{k[1]}": dataclasses.asdict(v) if dataclasses.is_dataclass(v) else str(v) for k, v in cw_labels.items()}
        group.boundary_pairs = _boundary_pairs(group)
        if not group.observations:
            continue
        count += 1
        yield group
