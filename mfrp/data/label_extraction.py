"""MFRP label extraction: soft branch, continuous burden, signed margin and coercion labels."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from mfrp.data.schema import BRANCHES, BRANCH_TO_INDEX, CEDING_BRANCH_IDS, ResponseObservation, CoercionWitnessLabel, normalize_branch_probs

try:  # shapely is optional but listed in project requirements.
    from shapely.geometry import Polygon
except Exception:  # pragma: no cover
    Polygon = None


@dataclass
class RobustScales:
    delay: float = 1.0
    decel: float = 1.0
    jerk: float = 1.0
    dev: float = 1.0
    trajectory: float = 1.0
    burden: float = 1.0
    margin: float = 1.0


def _finite(v: float, default: float) -> float:
    return float(v) if np.isfinite(v) else default


def _max_decel(traj: np.ndarray, dt: float) -> float:
    if traj is None or len(traj) < 2 or traj.shape[-1] <= 5:
        return 0.0
    speed = np.asarray(traj[:, 5], dtype=np.float32)
    acc = np.diff(speed) / max(dt, 1e-3)
    return float(np.max(np.maximum(-acc, 0.0))) if len(acc) else 0.0


def _rms_jerk(traj: np.ndarray, dt: float) -> float:
    if traj is None or len(traj) < 3 or traj.shape[-1] <= 5:
        return 0.0
    speed = np.asarray(traj[:, 5], dtype=np.float32)
    acc = np.diff(speed) / max(dt, 1e-3)
    jerk = np.diff(acc) / max(dt, 1e-3)
    return float(np.sqrt(np.mean(np.square(jerk)))) if len(jerk) else 0.0


def _ade(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return float(np.mean(np.linalg.norm(np.asarray(a[:n, :2]) - np.asarray(b[:n, :2]), axis=-1)))


def branch_soft_labels(
    tau_e_in: float,
    tau_i_k_in: float,
    tau_i_0_in: float,
    tau_i_k_out: float | None,
    agent_traj: np.ndarray,
    neutral_traj: np.ndarray | None,
    ego_agent_min_gap: float,
    both_enter_interaction: bool = True,
    dt: float = 0.1,
    a_brake: float = 1.5,
    a_accel: float = 1.0,
    delta_t_pass: float = 0.5,
    sigma_tau: float = 1.0,
    sigma_y: float = 3.0,
    nonconflict_gap_threshold: float = 30.0,
    eps: float = 1e-6,
) -> tuple[np.ndarray, int, bool, dict[str, float]]:
    """Return six-way soft branch probabilities in MFRP order.

    Ambiguity is represented by probability entropy and masks, never by an
    ``ambiguous`` class.
    """
    if (not both_enter_interaction) or ego_agent_min_gap > nonconflict_gap_threshold:
        raw = np.zeros(len(BRANCHES), dtype=np.float32)
        raw[BRANCH_TO_INDEX["nonconflict"]] = 1.0
        return raw, BRANCH_TO_INDEX["nonconflict"], True, {"e_nonconflict": 1.0, "ego_agent_min_gap": float(ego_agent_min_gap)}
    max_decel = _max_decel(agent_traj, dt)
    if agent_traj is not None and len(agent_traj) >= 2 and agent_traj.shape[-1] > 5:
        acc = np.diff(agent_traj[:, 5]) / max(dt, 1e-3)
        max_accel = float(np.max(acc)) if len(acc) else 0.0
    else:
        max_accel = 0.0
    ade_y = _ade(agent_traj, neutral_traj) if neutral_traj is not None else 0.0
    tau_delta = _finite(tau_i_k_in, 0.0) - _finite(tau_i_0_in, _finite(tau_i_k_in, 0.0))
    e_cede = float(tau_e_in < tau_i_k_in) * max(0.0, tau_delta)
    e_brake = max(0.0, max_decel - a_brake)
    e_accel = max(0.0, max_accel - a_accel) * float(tau_i_k_in < tau_e_in)
    e_pass = float(tau_i_k_out is not None and tau_i_k_out < tau_e_in - delta_t_pass)
    e_keep = float(np.exp(-abs(tau_delta) / max(sigma_tau, eps) - ade_y / max(sigma_y, eps)))
    e_nonconflict = 0.0
    raw = np.asarray([e_keep, e_cede, e_brake, e_accel, e_pass, e_nonconflict], dtype=np.float32)
    valid = bool(raw.sum() > eps)
    probs = normalize_branch_probs(raw, valid)
    return probs, int(probs.argmax()) if valid else -1, valid, {
        "e_keep": e_keep, "e_cede": e_cede, "e_brake": e_brake, "e_accel": e_accel,
        "e_pass": e_pass, "e_nonconflict": e_nonconflict, "max_decel": max_decel,
        "max_accel": max_accel, "ade_vs_neutral": ade_y, "tau_delta": tau_delta,
    }


def select_neutral_candidate(candidates: Iterable[Any]) -> Any:
    """Choose logged non-aggressive candidate when possible, else lowest-cost gentle candidate."""
    cands = list(candidates)
    if not cands:
        raise ValueError("no candidates available for neutral baseline")
    logged = [c for c in cands if getattr(c, "family", "") == "logged_anchor" and getattr(getattr(c, "validity", None), "valid", True)]
    pool = logged or [c for c in cands if getattr(getattr(c, "validity", None), "valid", True)] or cands
    def score(c: Any) -> float:
        traj = getattr(c, "future_states_ego_frame", getattr(c, "future_states", None))
        dec = _max_decel(traj, 0.1) if traj is not None else 0.0
        jerk = _rms_jerk(traj, 0.1) if traj is not None else 0.0
        return float(getattr(c, "nominal_cost", 0.0)) + dec + 0.1 * jerk
    return min(pool, key=score)


def baseline_relative_burden(
    agent_traj: np.ndarray,
    neutral_traj: np.ndarray,
    tau_i_k_in_censored: float,
    tau_i_0_in_censored: float,
    dt: float = 0.1,
    scales: RobustScales | dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    if scales is None:
        scales = RobustScales()
    if isinstance(scales, dict):
        scales = RobustScales(**{k: float(v) for k, v in scales.items() if k in RobustScales.__annotations__})
    weights = weights or {"delay": 1.0, "decel": 1.0, "jerk": 1.0, "dev": 1.0}
    delay = max(0.0, tau_i_k_in_censored - tau_i_0_in_censored) / max(scales.delay, 1e-6)
    dec = max(0.0, _max_decel(agent_traj, dt) - _max_decel(neutral_traj, dt)) / max(scales.decel, 1e-6)
    jerk = max(0.0, _rms_jerk(agent_traj, dt) - _rms_jerk(neutral_traj, dt)) / max(scales.jerk, 1e-6)
    dev = _ade(agent_traj, neutral_traj) / max(scales.dev, 1e-6)
    delta_b = weights["delay"] * delay + weights["decel"] * dec + weights["jerk"] * jerk + weights["dev"] * dev
    return float(delta_b), {"delay_term": delay, "decel_term": dec, "jerk_term": jerk, "dev_term": dev}


def high_pressure_label(delta_b: float, agent_traj: np.ndarray, dt: float = 0.1, eta_B: float = 1.0, a_hard: float = 3.5, j_hard: float = 5.0) -> float:
    return float(delta_b >= eta_B or _max_decel(agent_traj, dt) >= a_hard or _rms_jerk(agent_traj, dt) >= j_hard)


def _box_polygon(state: np.ndarray):
    x, y = float(state[0]), float(state[1])
    yaw = float(state[6]) if len(state) > 6 else 0.0
    length = max(float(state[7]) if len(state) > 7 else 4.5, 0.1)
    width = max(float(state[8]) if len(state) > 8 else 2.0, 0.1)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.asarray([[c, -s], [s, c]])
    corners = np.asarray([[ length/2, width/2], [ length/2, -width/2], [-length/2, -width/2], [-length/2, width/2]]) @ rot.T
    pts = corners + np.asarray([x, y])
    if Polygon is not None:
        return Polygon(pts)
    return pts


def signed_oriented_box_separation(ego_state: np.ndarray, agent_state: np.ndarray) -> float:
    """Positive when disjoint, negative approximate penetration depth."""
    if Polygon is not None:
        pe, pa = _box_polygon(ego_state), _box_polygon(agent_state)
        if not pe.is_valid or not pa.is_valid:
            return float(np.linalg.norm(np.asarray(ego_state[:2]) - np.asarray(agent_state[:2])))
        d = float(pe.distance(pa))
        if d > 0.0:
            return d
        inter = pe.intersection(pa).area
        min_dim = max(0.1, min(float(ego_state[7]) if len(ego_state)>7 else 4.5, float(ego_state[8]) if len(ego_state)>8 else 2.0, float(agent_state[7]) if len(agent_state)>7 else 4.5, float(agent_state[8]) if len(agent_state)>8 else 2.0))
        return -float(inter / min_dim)
    # Fallback: circle-like signed distance using half diagonal radii.
    re = 0.5 * np.hypot(float(ego_state[7]) if len(ego_state)>7 else 4.5, float(ego_state[8]) if len(ego_state)>8 else 2.0)
    ra = 0.5 * np.hypot(float(agent_state[7]) if len(agent_state)>7 else 4.5, float(agent_state[8]) if len(agent_state)>8 else 2.0)
    return float(np.linalg.norm(np.asarray(ego_state[:2]) - np.asarray(agent_state[:2])) - re - ra)


def signed_safety_margin(ego_traj: np.ndarray, agent_traj: np.ndarray) -> float:
    n = min(len(ego_traj), len(agent_traj))
    if n == 0:
        return float("nan")
    vals = [signed_oriented_box_separation(ego_traj[t], agent_traj[t]) for t in range(n)]
    return float(np.nanmin(vals))


def make_response_observation(
    scenario_id: str,
    root_hash: str,
    candidate_id: str,
    agent_id: int,
    variant_id: str,
    ego_traj: np.ndarray,
    agent_traj: np.ndarray,
    neutral_agent_traj: np.ndarray,
    interaction_features: dict[str, Any],
    priority_score: float,
    priority_confidence: float,
    dt: float = 0.1,
    scales: RobustScales | dict[str, float] | None = None,
    d_near: float = 1.0,
) -> ResponseObservation:
    tau_e = float(interaction_features.get("tau_e_in", 0.0))
    tau_i = float(interaction_features.get("tau_i_k_in", interaction_features.get("tau_i0_in", 0.0)))
    tau_i0 = float(interaction_features.get("tau_i0_in", tau_i))
    tau_i_out = interaction_features.get("tau_i_k_out", None)
    min_gap = float(interaction_features.get("gap", np.nanmin(np.linalg.norm(ego_traj[:min(len(ego_traj),len(agent_traj)),:2]-agent_traj[:min(len(ego_traj),len(agent_traj)),:2], axis=-1))))
    both = bool(interaction_features.get("both_enter_interaction", True))
    probs, hard, bvalid, diag_b = branch_soft_labels(tau_e, tau_i, tau_i0, tau_i_out, agent_traj, neutral_agent_traj, min_gap, both, dt)
    burden, bdiag = baseline_relative_burden(agent_traj, neutral_agent_traj, tau_i, tau_i0, dt, scales)
    hp = high_pressure_label(burden, agent_traj, dt)
    margin = signed_safety_margin(ego_traj, agent_traj)
    traj = np.stack([agent_traj[:, 0], agent_traj[:, 1], agent_traj[:, 3], agent_traj[:, 4], agent_traj[:, 6]], axis=-1).astype(np.float32)
    valid = agent_traj[:, 9] > 0.5 if agent_traj.shape[-1] > 9 else np.ones(len(agent_traj), dtype=bool)
    return ResponseObservation(
        scenario_id, root_hash, candidate_id, agent_id, variant_id,
        probs, hard, traj, valid, burden, hp, margin, bool(margin < d_near),
        priority_score, priority_confidence, dict(interaction_features), bvalid, bool(valid.any()), True,
        bool(np.isfinite(margin)), True, priority_confidence > 0.0, True, {**diag_b, **bdiag},
    )


def coercion_witness_label(
    observations: list[ResponseObservation],
    candidate_id: str,
    agent_id: int,
    root_hash: str,
    scenario_id: str,
    eta_H: float = 0.2,
    eta_B: float = 1.0,
    tau_H: float = 0.2,
    tau_B: float = 0.5,
    eps: float = 1e-6,
) -> CoercionWitnessLabel:
    obs = [o for o in observations if o.candidate_id == candidate_id and o.agent_id == agent_id and o.rollout_valid]
    if not obs:
        return CoercionWitnessLabel(scenario_id, root_hash, candidate_id, agent_id, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, {"empty": True})
    branch = np.asarray([o.branch_hard for o in obs], dtype=int)
    safe = np.asarray([o.safety_margin > 0.0 for o in obs], dtype=np.float32)
    burden = np.asarray([o.burden for o in obs], dtype=np.float32)
    w = np.asarray([float(o.branch_valid and o.margin_valid and o.burden_valid) for o in obs], dtype=np.float32)
    ceding = np.asarray([b in CEDING_BRANCH_IDS for b in branch], dtype=np.float32)
    wc = w * ceding
    wn = w * (1.0 - ceding)
    s_c = float((wc * safe).sum() / max(float(wc.sum()), eps))
    s_not_c = float((wn * safe).sum() / max(float(wn.sum()), eps))
    b_c = float((wc * burden).sum() / max(float(wc.sum()), eps))
    d_c = max(0.0, s_c - s_not_c)
    priority = float(np.mean([o.priority_score for o in obs]))
    z = float(1.0/(1.0+np.exp(-(s_c - s_not_c - eta_H)/max(tau_H, eps))) * 1.0/(1.0+np.exp(-(b_c - eta_B)/max(tau_B, eps))) * (1.0 - priority))
    both_sides = float(wc.sum() > eps and wn.sum() > eps)
    diversity = min(1.0, len(set([o.variant_id for o in obs])) / 3.0)
    prio_conf = float(np.mean([o.priority_confidence for o in obs]))
    confidence = float(np.clip(0.15 + 0.45 * both_sides + 0.25 * diversity + 0.15 * prio_conf, 0.0, 1.0))
    if not both_sides:
        confidence *= 0.25
    return CoercionWitnessLabel(scenario_id, root_hash, candidate_id, agent_id, z, confidence, s_c, s_not_c, b_c, d_c, priority, {"num_variants": len(obs), "both_sides": bool(both_sides)})


def build_cw_rank_pairs(labels: list[CoercionWitnessLabel], epsilon_D: float = 0.1, epsilon_B: float = 0.2) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for a in labels:
        for b in labels:
            if a.agent_id != b.agent_id or a.candidate_id == b.candidate_id:
                continue
            if (a.d_c - b.d_c > epsilon_D) and (a.b_c - b.b_c > epsilon_B) and min(a.confidence, b.confidence) > 0.1:
                pairs.append((a.candidate_id, b.candidate_id))
    return pairs
