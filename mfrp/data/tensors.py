"""Batch collation for MFRP same-root groups.

The collator keeps deployment inputs separate from supervision.  Deployment-safe
inputs are ``scene_features``, ``candidate_features``, masks and priority.  Labels,
rollout variants, support probes and query masks are training/evaluation only.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from mfrp.data.schema import BRANCHES, TRAJ_TARGET_DIM, SameRootGroup
from mfrp.data.interaction_region import build_interaction_region, intervention_coordinate, constant_velocity_extrapolate
from mfrp.data.priority import compute_priority_score


def _pad_or_trim(x: np.ndarray, dim: int) -> np.ndarray:
    y = np.zeros(dim, dtype=np.float32)
    n = min(dim, int(np.asarray(x).size))
    if n:
        y[:n] = np.asarray(x, dtype=np.float32).reshape(-1)[:n]
    return y


def _scene_agent_features(group: SameRootGroup, agent_id: int, dim: int) -> np.ndarray:
    vals: list[float] = [float(group.root_scene.current_time_index), float(group.root_scene.dt)]
    tracks = getattr(group.root_scene, "agent_tracks", None)
    if tracks is not None and int(agent_id) < tracks.states.shape[0]:
        hist = np.asarray(tracks.states[int(agent_id)], dtype=np.float32)
        valid = hist[:, 9] > 0.5 if hist.shape[-1] > 9 else np.ones(len(hist), bool)
        if valid.any():
            st = hist[np.where(valid)[0][-1]]
            vals.extend([float(st[0]), float(st[1]), float(st[3]), float(st[4]), float(st[5]), float(st[6]), float(st[7]), float(st[8])])
    # Intentionally do not add root_hash/scene-id fingerprints. They can create
    # train/val memorization if adjacent windows or duplicated root scenes leak
    # across splits. Use physical/map/history features only.
    return _pad_or_trim(np.asarray(vals, dtype=np.float32), dim)


def _candidate_features_and_priority(group: SameRootGroup, candidate_index: int, agent_id: int, dim: int) -> tuple[np.ndarray, float, float]:
    c = group.candidates[candidate_index]
    traj = np.asarray(c.future_states_ego_frame, dtype=np.float32)
    # Deployment-safe neutral reference: observed history -> constant-velocity extrapolation.
    # Do not copy priority/entry labels from ResponseObservation; those are label-side
    # products of rollouts and can silently leak future/diagnostic information.
    neutral = None
    tracks = getattr(group.root_scene, "agent_tracks", None)
    if tracks is not None and int(agent_id) < tracks.states.shape[0]:
        neutral = constant_velocity_extrapolate(tracks.states[int(agent_id)], len(traj), float(group.root_scene.dt))
    interaction = build_interaction_region(traj, neutral, dt=float(group.root_scene.dt))
    # Merge static/pre-execution metadata from the candidate and root scene when present.
    meta = {}
    meta.update(getattr(group.root_scene, "metadata", {}) or {})
    meta.update(getattr(c, "metadata", {}) or {})
    if isinstance(meta.get("priority_features"), dict):
        interaction.features.update(meta["priority_features"])
    pr = compute_priority_score(interaction.features)
    feat = _pad_or_trim(
        intervention_coordinate(traj, interaction, pr.score, pr.confidence, np.asarray([float(c.nominal_cost)], dtype=np.float32)),
        dim,
    )
    return feat, float(pr.score), float(pr.confidence)


def collate_same_root_groups(
    groups: list[SameRootGroup],
    future_steps: int = 80,
    candidate_feature_dim: int = 20,
    scene_feature_dim: int | None = None,
    *,
    require_support_query_split: bool = True,
    include_legacy_priority_aliases: bool = False,
) -> dict[str, np.ndarray | list[Any]]:
    B = len(groups)
    A = max((len(g.relevant_agent_ids) for g in groups), default=0)
    K = max((len(g.candidates) for g in groups), default=0)
    R = max((len(g.rollout_variants) for g in groups), default=0)
    C = len(BRANCHES)
    S = scene_feature_dim or candidate_feature_dim
    scene_features = np.zeros((B, A, S), dtype=np.float32)
    candidate_features = np.zeros((B, A, K, candidate_feature_dim), dtype=np.float32)
    candidate_valid = np.zeros((B, K), dtype=bool)
    agent_candidate_valid = np.zeros((B, A, K), dtype=bool)
    variant_valid = np.zeros((B, A, K, R), dtype=bool)
    branch_probs = np.zeros((B, A, K, R, C), dtype=np.float32)
    branch_hard = -np.ones((B, A, K, R), dtype=np.int64)
    trajectory = np.zeros((B, A, K, R, future_steps, TRAJ_TARGET_DIM), dtype=np.float32)
    trajectory_mask = np.zeros((B, A, K, R, future_steps), dtype=bool)
    burden = np.zeros((B, A, K, R), dtype=np.float32)
    hp_label = np.zeros((B, A, K, R), dtype=np.float32)
    safety_margin = np.zeros((B, A, K, R), dtype=np.float32)
    # Deployment-safe priority derived only from root-scene/candidate metadata.
    # Label-side observation priority is deliberately not copied into these arrays.
    priority_score_preexec = np.full((B, A, K), 0.5, dtype=np.float32)
    priority_confidence_preexec = np.zeros((B, A, K), dtype=np.float32)
    cw_soft_label = np.zeros((B, A, K), dtype=np.float32)
    cw_confidence = np.zeros((B, A, K), dtype=np.float32)
    support_probe_features = np.zeros((B, A, max(1, K * max(R, 1)), candidate_feature_dim + C + TRAJ_TARGET_DIM + 3), dtype=np.float32)
    support_probe_mask = np.zeros((B, A, max(1, K * max(R, 1))), dtype=bool)
    query_probe_mask = np.zeros((B, A, K, R), dtype=bool)

    # Boundary/geometry pair buffers.
    max_edges = max((len(g.boundary_pairs) for g in groups), default=0)
    edge_index = np.zeros((B, A, max_edges, 2), dtype=np.int64)
    edge_valid = np.zeros((B, A, max_edges), dtype=bool)
    response_distance = np.zeros((B, A, max_edges), dtype=np.float32)

    for b, g in enumerate(groups):
        agent_to_a = {aid: a for a, aid in enumerate(g.relevant_agent_ids)}
        cand_to_k = {c.candidate_id: k for k, c in enumerate(g.candidates)}
        var_to_r = {v: r for r, v in enumerate(g.rollout_variants)}
        support_ids = set(g.metadata.get("support_candidate_ids", []))
        query_ids = set(g.metadata.get("query_candidate_ids", []))
        if not support_ids or not query_ids:
            if require_support_query_split:
                raise ValueError(
                    f"SameRootGroup {g.scenario_id}/{g.root_hash} lacks a materialized support/query candidate split. "
                    "Run mfrp.data.support_query.split_support_query during dataset materialization, "
                    "or call collate_same_root_groups(..., require_support_query_split=False) only for smoke tests."
                )
            if not query_ids:
                query_ids = set(c.candidate_id for c in g.candidates)
        for a, aid in enumerate(g.relevant_agent_ids):
            scene_features[b, a] = _scene_agent_features(g, int(aid), S)
            for k, c in enumerate(g.candidates):
                candidate_valid[b, k] = bool(c.validity.valid)
                feat, pr_score, pr_conf = _candidate_features_and_priority(g, k, int(aid), candidate_feature_dim)
                candidate_features[b, a, k] = feat
                priority_score_preexec[b, a, k] = pr_score
                priority_confidence_preexec[b, a, k] = pr_conf
        for (cid, aid, vid), obs in g.observations.items():
            if cid not in cand_to_k or aid not in agent_to_a or vid not in var_to_r:
                continue
            k, a, r = cand_to_k[cid], agent_to_a[aid], var_to_r[vid]
            agent_candidate_valid[b, a, k] = True
            variant_valid[b, a, k, r] = bool(obs.rollout_valid)
            branch_probs[b, a, k, r] = obs.branch_probs
            branch_hard[b, a, k, r] = obs.branch_hard
            t = min(future_steps, obs.trajectory.shape[0])
            trajectory[b, a, k, r, :t] = obs.trajectory[:t]
            trajectory_mask[b, a, k, r, :t] = obs.trajectory_valid[:t]
            burden[b, a, k, r] = obs.burden
            hp_label[b, a, k, r] = obs.hp_label
            safety_margin[b, a, k, r] = obs.safety_margin
            # Do not copy obs.priority_score / obs.priority_confidence here. Those
            # values may come from rollout diagnostics and are label-side.
            if cid in query_ids:
                query_probe_mask[b, a, k, r] = bool(obs.rollout_valid)
            if cid in support_ids:
                n = k * max(R, 1) + r
                trsum = np.zeros(TRAJ_TARGET_DIM, dtype=np.float32)
                if t > 0 and trajectory_mask[b, a, k, r, :t].any():
                    trsum = trajectory[b, a, k, r, :t][trajectory_mask[b, a, k, r, :t]].mean(axis=0)
                support_probe_features[b, a, n] = np.concatenate([
                    candidate_features[b, a, k], obs.branch_probs.astype(np.float32), trsum.astype(np.float32),
                    np.asarray([obs.burden, obs.hp_label, obs.safety_margin], dtype=np.float32)
                ])
                support_probe_mask[b, a, n] = bool(obs.rollout_valid)
        for (cid, aid), lab in g.cw_labels.items():
            if cid in cand_to_k and aid in agent_to_a:
                k, a = cand_to_k[cid], agent_to_a[aid]
                cw_soft_label[b, a, k] = lab.soft_label
                cw_confidence[b, a, k] = lab.confidence
        for e, bp in enumerate(g.boundary_pairs):
            if e >= max_edges or bp.agent_id not in agent_to_a:
                continue
            if bp.candidate_id_a in cand_to_k and bp.candidate_id_b in cand_to_k:
                a = agent_to_a[bp.agent_id]
                edge_index[b, a, e] = [cand_to_k[bp.candidate_id_a], cand_to_k[bp.candidate_id_b]]
                edge_valid[b, a, e] = bool(bp.valid)
                response_distance[b, a, e] = float(bp.response_distance)
    result = {
        "scene_features": scene_features,
        "candidate_features": candidate_features,
        "candidate_valid": candidate_valid,
        "agent_candidate_valid": agent_candidate_valid,
        "variant_valid": variant_valid,
        "query_probe_mask": query_probe_mask,
        "support_probe_features": support_probe_features,
        "support_probe_mask": support_probe_mask,
        "branch_probs": branch_probs,
        "branch_hard": branch_hard,
        "trajectory": trajectory,
        "trajectory_mask": trajectory_mask,
        "burden": burden,
        "hp_label": hp_label,
        "safety_margin": safety_margin,
        "priority_score_preexec": priority_score_preexec,
        "priority_confidence_preexec": priority_confidence_preexec,
        "cw_soft_label": cw_soft_label,
        "cw_confidence": cw_confidence,
        "edge_index": edge_index,
        "edge_valid": edge_valid,
        "response_distance": response_distance,
        "groups": groups,
    }
    if include_legacy_priority_aliases:
        result["priority_score"] = priority_score_preexec
        result["priority_confidence"] = priority_confidence_preexec
    return result
