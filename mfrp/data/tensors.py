from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import numpy as np

from .schema import SameRootGroup, BRANCHES, BoundaryPair


def _pad1(x: np.ndarray, n: int, value: float = 0.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    out = np.full((n,), value, dtype=np.float32)
    out[: min(n, len(x))] = x[:n]
    return out


def _finite_flatten(x: Any) -> np.ndarray:
    if x is None:
        return np.zeros(0, dtype=np.float32)
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    return arr[np.isfinite(arr)]


def _root_context_features(root, n: int) -> np.ndarray:
    """Compact, deployment-safe scene context from map/traffic/route metadata.

    The previous tensorizer only copied the target agent's last state.  That made
    traffic controls, route context and map features effectively disappear even
    when the adapter extracted them.  This summary deliberately uses only root
    information available at planning time and never touches rollout labels.
    """
    feat = np.zeros(int(n), dtype=np.float32)
    pos = 0
    mf = _finite_flatten(getattr(root, "map_features", None))
    if mf.size:
        pts = np.asarray(root.map_features, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[-1] >= 2:
            xy = pts[:, :2]
            finite = np.isfinite(xy).all(axis=-1)
            xy = xy[finite]
            if xy.size:
                dist = np.linalg.norm(xy, axis=-1)
                vals = np.array([
                    len(xy) / 512.0,
                    np.nanmean(dist),
                    np.nanmin(dist),
                    np.nanpercentile(dist, 25),
                    np.nanpercentile(dist, 75),
                    np.nanmean(xy[:, 0]),
                    np.nanmean(xy[:, 1]),
                    np.nanstd(xy[:, 0]),
                    np.nanstd(xy[:, 1]),
                ], dtype=np.float32)
                take = min(n - pos, len(vals))
                if take > 0:
                    feat[pos:pos + take] = vals[:take]
                    pos += take
    tc = _finite_flatten(getattr(root, "traffic_controls", None))
    if pos < n and tc.size:
        vals = np.array([tc.mean(), tc.std(), tc.min(), tc.max(), min(tc.size, 64) / 64.0], dtype=np.float32)
        take = min(n - pos, len(vals))
        feat[pos:pos + take] = vals[:take]
        pos += take
    rf = _finite_flatten(getattr(root, "route_features", None))
    if pos < n and rf.size:
        take = min(n - pos, rf.size)
        feat[pos:pos + take] = rf[:take]
    return feat


def _agent_index(agent_id: str, fallback: int, hist_n: int) -> int:
    try:
        agent_num = int(agent_id)
    except Exception:
        agent_num = fallback
    if agent_num < 0 or agent_num >= hist_n:
        agent_num = min(fallback, max(hist_n - 1, 0))
    return int(agent_num)


def _combine_candidate_and_agent_features(cand, agent_id: str, dim: int) -> np.ndarray:
    base = _pad1(cand.features, dim)
    meta = cand.metadata if isinstance(getattr(cand, "metadata", None), dict) else {}
    agent_features = meta.get("agent_features", {})
    agent_meta = agent_features.get(agent_id, {}) if isinstance(agent_features, dict) else {}
    extra = agent_meta.get("interaction_features") if isinstance(agent_meta, dict) else None
    if extra is not None:
        extra_arr = np.asarray(extra, dtype=np.float32).reshape(-1)
        # _candidate_features uses the first 8 slots.  Put per-agent pre-exec
        # features after those slots instead of overwriting by non-zero mask; zero
        # is a meaningful value for entry gap and must be preserved.
        offset = int(meta.get("agent_feature_offset", 8))
        offset = max(0, min(offset, dim))
        take = min(dim - offset, len(extra_arr))
        if take > 0:
            base[offset:offset + take] = extra_arr[:take]
    return base


def _unpack_boundary_pair(item) -> tuple[str, str, str, float]:
    if isinstance(item, BoundaryPair):
        return item.agent_id, item.candidate_a, item.candidate_b, item.distance
    aid, ca, cb, dist = item
    return str(aid), str(ca), str(cb), float(dist)


def collate_same_root_groups(
    groups: list[SameRootGroup],
    *,
    candidate_feature_dim: int = 20,
    scene_feature_dim: int = 32,
    future_steps: int = 80,
    state_dim: int = 5,
    require_support_query_split: bool = True,
    allow_debug: bool = False,
) -> dict[str, np.ndarray]:
    if not groups:
        raise ValueError("cannot collate empty group list")
    for g in groups:
        g.validate(require_support_query_split=require_support_query_split, allow_debug=allow_debug)
    B = len(groups)
    A = max(len(g.relevant_agent_ids) for g in groups)
    K = max(len(g.candidates) for g in groups)
    R = max(len(g.rollout_variants) for g in groups)
    C = len(BRANCHES)
    Nsup = K * R
    candidate_features = np.zeros((B, A, K, candidate_feature_dim), dtype=np.float32)
    scene_features = np.zeros((B, A, scene_feature_dim), dtype=np.float32)
    candidate_valid = np.zeros((B, K), dtype=bool)
    agent_candidate_valid = np.zeros((B, A, K), dtype=bool)
    branch_probs = np.zeros((B, A, K, R, C), dtype=np.float32)
    trajectory = np.zeros((B, A, K, R, future_steps, state_dim), dtype=np.float32)
    trajectory_mask = np.zeros((B, A, K, R, future_steps), dtype=bool)
    burden = np.zeros((B, A, K, R), dtype=np.float32)
    safety_margin = np.zeros((B, A, K, R), dtype=np.float32)
    variant_valid = np.zeros((B, A, K, R), dtype=bool)
    priority_score_preexec = np.full((B, A, K), 0.5, dtype=np.float32)
    priority_confidence_preexec = np.zeros((B, A, K), dtype=np.float32)
    support_probe_mask = np.zeros((B, A, Nsup), dtype=bool)
    support_probe_features = np.zeros((B, A, Nsup, candidate_feature_dim + C + state_dim + 4), dtype=np.float32)
    query_probe_mask = np.zeros((B, A, K, R), dtype=bool)
    cw_soft_label = np.zeros((B, A, K), dtype=np.float32)
    cw_confidence = np.zeros((B, A, K), dtype=np.float32)
    nominal_cost = np.zeros((B, K), dtype=np.float32)
    root_hashes = []

    edge_a, edge_b, edge_agent, edge_batch, edge_dist = [], [], [], [], []
    for b, g in enumerate(groups):
        root_hashes.append(g.root_scene.root_hash)
        cand_index = {c.candidate_id: i for i, c in enumerate(g.candidates)}
        agent_index = {a: i for i, a in enumerate(g.relevant_agent_ids)}
        var_index = {r: i for i, r in enumerate(g.rollout_variants)}
        support_ids = set(g.metadata.get("support_candidate_ids", []))
        query_ids = set(g.metadata.get("query_candidate_ids", []))
        hist = np.asarray(g.root_scene.history, dtype=np.float32)
        hist_mask = np.asarray(g.root_scene.history_mask, dtype=bool)
        context_dim_start = min(state_dim + 2, scene_feature_dim)
        context = _root_context_features(g.root_scene, max(0, scene_feature_dim - context_dim_start))
        for a_idx, agent_id in enumerate(g.relevant_agent_ids):
            agent_num = _agent_index(agent_id, a_idx, hist.shape[0])
            valid_hist = hist[agent_num][hist_mask[agent_num]] if hist.size and hist_mask.size and agent_num < hist_mask.shape[0] else np.zeros((0, state_dim))
            if valid_hist.size:
                take = min(scene_feature_dim, valid_hist.shape[-1])
                scene_features[b, a_idx, :take] = valid_hist[-1, :take]
                if scene_feature_dim > valid_hist.shape[-1]:
                    vel = np.diff(valid_hist[:, :2], axis=0)
                    vtake = min(2, vel.shape[-1] if vel.size else 0, scene_feature_dim - valid_hist.shape[-1])
                    if vtake > 0:
                        scene_features[b, a_idx, valid_hist.shape[-1]:valid_hist.shape[-1] + vtake] = vel[-1, :vtake] if vel.size else 0
            if context.size and context_dim_start < scene_feature_dim:
                take = min(len(context), scene_feature_dim - context_dim_start)
                scene_features[b, a_idx, context_dim_start:context_dim_start + take] = context[:take]
        for k, cand in enumerate(g.candidates):
            candidate_valid[b, k] = bool(cand.valid)
            nominal_cost[b, k] = float(cand.nominal_cost)
            for a_idx, agent_id in enumerate(g.relevant_agent_ids):
                base = _combine_candidate_and_agent_features(cand, agent_id, candidate_feature_dim)
                candidate_features[b, a_idx, k] = base
                agent_candidate_valid[b, a_idx, k] = bool(cand.valid)
        for (cid, aid, rid), obs in g.observations.items():
            if cid not in cand_index or aid not in agent_index or rid not in var_index:
                continue
            k, a, r = cand_index[cid], agent_index[aid], var_index[rid]
            bp = np.asarray(obs.branch_probs, dtype=np.float32)
            branch_probs[b, a, k, r] = bp / max(1e-6, float(bp.sum()))
            yy = np.asarray(obs.trajectory, dtype=np.float32)
            mm = np.asarray(obs.trajectory_mask, dtype=bool)
            t = min(future_steps, yy.shape[0])
            d = min(state_dim, yy.shape[-1])
            trajectory[b, a, k, r, :t, :d] = yy[:t, :d]
            trajectory_mask[b, a, k, r, : min(t, len(mm))] = mm[: min(t, len(mm))]
            burden[b, a, k, r] = float(obs.burden)
            safety_margin[b, a, k, r] = float(obs.safety_margin)
            variant_valid[b, a, k, r] = bool(np.asarray(obs.trajectory_mask).any())
            priority_score_preexec[b, a, k] = float(obs.priority_score_preexec)
            priority_confidence_preexec[b, a, k] = float(obs.priority_confidence_preexec)
            cw_soft_label[b, a, k] = max(cw_soft_label[b, a, k], float(obs.cw_soft_label))
            cw_confidence[b, a, k] = max(cw_confidence[b, a, k], float(obs.cw_confidence))
            if cid in query_ids:
                query_probe_mask[b, a, k, r] = True
            if cid in support_ids:
                p = k * R + r
                support_probe_mask[b, a, p] = True
                traj_summary = trajectory[b, a, k, r, :1, :state_dim].reshape(-1)
                support_probe_features[b, a, p] = np.concatenate([
                    candidate_features[b, a, k],
                    branch_probs[b, a, k, r],
                    _pad1(traj_summary, state_dim),
                    np.array([burden[b, a, k, r], safety_margin[b, a, k, r], float(obs.high_pressure), 1.0], dtype=np.float32),
                ])
        for item in g.boundary_pairs:
            aid, ca, cb, dist = _unpack_boundary_pair(item)
            if aid in agent_index and ca in cand_index and cb in cand_index:
                edge_batch.append(b)
                edge_agent.append(agent_index[aid])
                edge_a.append(cand_index[ca])
                edge_b.append(cand_index[cb])
                edge_dist.append(max(float(dist), 1e-3))
    out: dict[str, Any] = dict(
        scene_features=scene_features,
        candidate_features=candidate_features,
        candidate_valid=candidate_valid,
        agent_candidate_valid=agent_candidate_valid,
        branch_probs=branch_probs,
        trajectory=trajectory,
        trajectory_mask=trajectory_mask,
        burden=burden,
        safety_margin=safety_margin,
        variant_valid=variant_valid,
        support_probe_features=support_probe_features,
        support_probe_mask=support_probe_mask,
        query_probe_mask=query_probe_mask,
        priority_score_preexec=priority_score_preexec,
        priority_confidence_preexec=priority_confidence_preexec,
        cw_soft_label=cw_soft_label,
        cw_confidence=cw_confidence,
        nominal_cost=nominal_cost,
        root_hashes=np.asarray(root_hashes, dtype=object),
        debug_only=np.asarray([allow_debug], dtype=bool),
    )
    if edge_batch:
        out.update(
            edge_batch=np.asarray(edge_batch, dtype=np.int64),
            edge_agent=np.asarray(edge_agent, dtype=np.int64),
            edge_a=np.asarray(edge_a, dtype=np.int64),
            edge_b=np.asarray(edge_b, dtype=np.int64),
            edge_distance=np.asarray(edge_dist, dtype=np.float32),
            edge_valid=np.ones(len(edge_batch), dtype=bool),
        )
    return out


def write_npz_shard(path: Path, batch: dict[str, np.ndarray], metadata: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **batch)
    if metadata is not None:
        path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
