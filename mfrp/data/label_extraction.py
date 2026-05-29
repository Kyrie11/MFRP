from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .schema import BRANCHES, CEDING_BRANCHES, ResponseObservation
from .interaction_region import finite_difference, signed_point_margin

BRANCH_INDEX = {b: i for i, b in enumerate(BRANCHES)}
CEDING_IDX = [BRANCH_INDEX[b] for b in CEDING_BRANCHES]
NONCEDING_IDX = [i for i, b in enumerate(BRANCHES) if b not in CEDING_BRANCHES]


def softmax_scores(scores: dict[str, float], temperature: float = 1.0) -> np.ndarray:
    z = np.array([scores.get(b, 0.0) for b in BRANCHES], dtype=np.float32) / max(temperature, 1e-6)
    z = z - np.nanmax(z)
    p = np.exp(z)
    if not np.isfinite(p).all() or p.sum() <= 0:
        p = np.ones(len(BRANCHES), dtype=np.float32)
    return p / p.sum()


def _speed(traj: np.ndarray, dt: float) -> np.ndarray:
    traj = np.asarray(traj, dtype=np.float32)
    if traj.ndim == 1:
        traj = traj[None]
    if traj.shape[-1] > 9:
        return traj[:, 9]
    if traj.shape[-1] > 3:
        return traj[:, 3]
    return np.linalg.norm(finite_difference(traj[:, :2], dt), axis=-1)


def classify_response_branch(
    *,
    ego_traj: np.ndarray,
    agent_traj: np.ndarray,
    baseline_agent_traj: np.ndarray,
    tau_ego_in: float,
    tau_agent_in: float,
    tau_agent_base_in: float,
    dt: float = 0.1,
    nonconflict_distance: float = 12.0,
) -> np.ndarray:
    speed = _speed(agent_traj, dt)
    acc = finite_difference(speed[:, None], dt).reshape(-1)
    max_brake = float(np.nanmax(-acc)) if acc.size else 0.0
    max_accel = float(np.nanmax(acc)) if acc.size else 0.0
    n = min(len(agent_traj), len(baseline_agent_traj))
    ade = float(np.nanmean(np.linalg.norm(agent_traj[:n, :2] - baseline_agent_traj[:n, :2], axis=-1))) if n else 0.0
    m = min(len(ego_traj), len(agent_traj))
    min_dist = float(np.nanmin(np.linalg.norm(ego_traj[:m, :2] - agent_traj[:m, :2], axis=-1))) if m else float("inf")
    if min_dist > nonconflict_distance:
        return softmax_scores({"nonconflict": 4.0, "keep": 1.0})
    delay = max(0.0, float(tau_agent_in - tau_agent_base_in)) if np.isfinite(tau_agent_in) and np.isfinite(tau_agent_base_in) else 0.0
    scores = {
        "cede": delay + (1.0 if tau_ego_in < tau_agent_in else 0.0),
        "brake": max(0.0, max_brake - 2.0),
        "accelerate": max(0.0, max_accel - 1.5) + (0.5 if tau_agent_in < tau_ego_in else 0.0),
        "pass": 1.5 if tau_agent_in + 0.5 < tau_ego_in else 0.0,
        "keep": float(np.exp(-delay / 1.0 - ade / 3.0)),
        "nonconflict": 0.0,
    }
    return softmax_scores(scores, temperature=0.75)


def compute_burden(
    agent_traj: np.ndarray,
    baseline_agent_traj: np.ndarray,
    *,
    tau_agent_in: float,
    tau_agent_base_in: float,
    dt: float = 0.1,
    scales: dict[str, float] | None = None,
) -> float:
    scales = {**{"delay": 1.0, "dec": 2.0, "jerk": 2.0, "dev": 3.0}, **(scales or {})}
    speed = _speed(agent_traj, dt)
    bspeed = _speed(baseline_agent_traj, dt)
    acc = finite_difference(speed[:, None], dt).reshape(-1)
    bacc = finite_difference(bspeed[:, None], dt).reshape(-1)
    jerk = finite_difference(acc[:, None], dt).reshape(-1)
    bjerk = finite_difference(bacc[:, None], dt).reshape(-1)
    n = min(len(agent_traj), len(baseline_agent_traj))
    delay = max(0.0, float(tau_agent_in - tau_agent_base_in)) / scales["delay"] if np.isfinite(tau_agent_in) and np.isfinite(tau_agent_base_in) else 0.0
    dec = max(0.0, float(np.nanmax(-acc) - np.nanmax(-bacc))) / scales["dec"] if acc.size and bacc.size else 0.0
    jr = max(0.0, float(np.sqrt(np.nanmean(jerk**2)) - np.sqrt(np.nanmean(bjerk**2)))) / scales["jerk"] if jerk.size and bjerk.size else 0.0
    dev = float(np.nanmean(np.linalg.norm(agent_traj[:n, :2] - baseline_agent_traj[:n, :2], axis=-1))) / scales["dev"] if n else 0.0
    return float(delay + dec + 0.5 * jr + 0.5 * dev)


def baseline_relative_burden(agent_traj: np.ndarray, baseline_agent_traj: np.ndarray, tau_agent_in: float, tau_agent_base_in: float, *, dt: float = 0.1, scales: dict[str, float] | None = None) -> tuple[float, dict[str, float]]:
    b = compute_burden(agent_traj, baseline_agent_traj, tau_agent_in=tau_agent_in, tau_agent_base_in=tau_agent_base_in, dt=dt, scales=scales)
    details = {"delay": max(0.0, float(tau_agent_in - tau_agent_base_in)) if np.isfinite(tau_agent_in) and np.isfinite(tau_agent_base_in) else 0.0}
    if abs(b) < 1e-7:
        b = 0.0
    return b, details


def _box_corners(state: np.ndarray) -> np.ndarray:
    x, y = float(state[0]), float(state[1])
    yaw = float(state[6] if len(state) > 6 else (state[2] if len(state) > 2 else 0.0))
    length = float(state[7] if len(state) > 7 else 4.5)
    width = float(state[8] if len(state) > 8 else 2.0)
    dx, dy = length / 2.0, width / 2.0
    local = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]], dtype=np.float32)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return local @ R.T + np.array([x, y], dtype=np.float32)


def signed_oriented_box_separation(a: np.ndarray, b: np.ndarray) -> float:
    """Signed SAT separation for two oriented rectangles. Negative means overlap."""
    ca, cb = _box_corners(np.asarray(a, dtype=np.float32)), _box_corners(np.asarray(b, dtype=np.float32))
    axes = []
    for corners in (ca, cb):
        for i in range(4):
            edge = corners[(i + 1) % 4] - corners[i]
            n = np.array([-edge[1], edge[0]], dtype=np.float32)
            norm = np.linalg.norm(n)
            if norm > 1e-8:
                axes.append(n / norm)
    sep_max = -float("inf")
    overlap_min = float("inf")
    for ax in axes:
        pa, pb = ca @ ax, cb @ ax
        gap = max(float(pb.min() - pa.max()), float(pa.min() - pb.max()))
        if gap > 0:
            sep_max = max(sep_max, gap)
        overlap = min(float(pa.max()), float(pb.max())) - max(float(pa.min()), float(pb.min()))
        overlap_min = min(overlap_min, overlap)
    return float(sep_max if sep_max > -float("inf") else -max(overlap_min, 0.0))


def high_pressure_label(burden: float, agent_traj: np.ndarray, *, dt: float = 0.1, eta_b: float = 1.0, hard_brake: float = 4.0, hard_jerk: float = 5.0) -> bool:
    speed = _speed(agent_traj, dt)
    acc = finite_difference(speed[:, None], dt).reshape(-1)
    jerk = finite_difference(acc[:, None], dt).reshape(-1)
    return bool(burden >= eta_b or (acc.size and np.nanmax(-acc) >= hard_brake) or (jerk.size and np.sqrt(np.nanmean(jerk**2)) >= hard_jerk))


def safety_margin(ego_traj: np.ndarray, agent_traj: np.ndarray, *, ego_radius: float = 2.2, agent_radius: float = 2.2) -> float:
    if ego_traj.shape[-1] >= 9 and agent_traj.shape[-1] >= 9:
        n = min(len(ego_traj), len(agent_traj))
        return float(np.nanmin([signed_oriented_box_separation(ego_traj[i], agent_traj[i]) for i in range(n)])) if n else float("nan")
    return signed_point_margin(ego_traj, agent_traj, ego_radius, agent_radius)


@dataclass(frozen=True)
class CoercionWitnessLabel:
    soft_label: float
    confidence: float
    s_c: float
    s_notc: float
    b_c: float
    ceding_count: int
    nonceding_count: int


def coercion_witness_label(
    observations: list[ResponseObservation],
    candidate_id: str,
    agent_id: str | int,
    root_hash: str | None = None,
    scene_id: str | None = None,
    *,
    eta_safe: float = 0.7,
    eta_unsafe: float = 0.4,
    eta_b: float = 1.0,
) -> CoercionWitnessLabel:
    obs = [o for o in observations if str(o.candidate_id) == str(candidate_id) and str(o.agent_id) == str(agent_id)]
    if not obs:
        return CoercionWitnessLabel(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
    pc = np.array([float(np.asarray(o.branch_probs)[CEDING_IDX].sum()) for o in obs], dtype=np.float32)
    pnc = 1.0 - pc
    safe = np.array([1.0 if o.safety_margin > 0 else 0.0 for o in obs], dtype=np.float32)
    burden = np.array([max(0.0, float(o.burden)) for o in obs], dtype=np.float32)
    c_mask = pc >= 0.5
    nc_mask = pnc > 0.5
    ceding_count = int(c_mask.sum())
    nonceding_count = int(nc_mask.sum())
    s_c = float(safe[c_mask].mean()) if ceding_count else float((pc * safe).sum() / max(pc.sum(), 1e-6))
    s_notc = float(safe[nc_mask].mean()) if nonceding_count else float((pnc * safe).sum() / max(pnc.sum(), 1e-6))
    b_c = float(burden[c_mask].mean()) if ceding_count else float((pc * burden).sum() / max(pc.sum(), 1e-6))
    dep = max(0.0, s_c - s_notc)
    label = np.clip(dep * max(0.0, b_c / max(eta_b, 1e-6)), 0.0, 1.0)
    if s_c > eta_safe and s_notc < eta_unsafe and b_c > eta_b:
        label = max(float(label), 1.0)
    # Confidence explicitly requires two-sided evidence.
    diversity = min(ceding_count, nonceding_count) / max(max(ceding_count, nonceding_count), 1)
    confidence = float(np.clip(diversity * min(1.0, len(obs) / 4.0), 0.0, 1.0))
    return CoercionWitnessLabel(float(label), confidence, s_c, s_notc, b_c, ceding_count, nonceding_count)


def make_response_observation(*args, **kwargs) -> ResponseObservation:
    """Create a ResponseObservation from either explicit fields or adapter legacy args.

    Legacy positional adapter form:
      scene_id, root_hash, candidate_id, agent_id, variant_id,
      ego_traj, agent_traj, baseline_agent_traj, interaction_features,
      priority_score, priority_confidence, dt=...
    """
    if args and len(args) >= 11:
        scene_id, root_hash, candidate_id, agent_id, variant_id, ego_traj, agent_traj, baseline_agent_traj, features, prio, prio_conf, *rest = args
        dt = float(kwargs.pop("dt", 0.1))
        tau_ego = float(features.get("tau_ego_in", features.get("tau_e_k_in", 0.0))) if isinstance(features, dict) else 0.0
        tau_agent = float(features.get("tau_i_k_in", features.get("tau_agent_in", 0.0))) if isinstance(features, dict) else 0.0
        tau_base = float(features.get("tau_i0_in", tau_agent)) if isinstance(features, dict) else tau_agent
        bp = classify_response_branch(ego_traj=ego_traj, agent_traj=agent_traj, baseline_agent_traj=baseline_agent_traj, tau_ego_in=tau_ego, tau_agent_in=tau_agent, tau_agent_base_in=tau_base, dt=dt)
        burden, _ = baseline_relative_burden(agent_traj, baseline_agent_traj, tau_agent, tau_base, dt=dt)
        margin = safety_margin(np.asarray(ego_traj), np.asarray(agent_traj))
        hp = high_pressure_label(burden, np.asarray(agent_traj), dt=dt)
        return ResponseObservation(scene_id, root_hash, candidate_id, agent_id, variant_id, bp, int(np.argmax(bp)), np.asarray(agent_traj, dtype=np.float32), np.ones(len(agent_traj), dtype=bool), burden, margin, tau_agent, hp, prio, prio_conf)
    return ResponseObservation(*args, **kwargs)
