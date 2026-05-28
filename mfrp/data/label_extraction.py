from __future__ import annotations

import numpy as np
from .schema import BRANCHES
from .interaction_region import finite_difference, signed_point_margin

BRANCH_INDEX = {b: i for i, b in enumerate(BRANCHES)}


def softmax_scores(scores: dict[str, float], temperature: float = 1.0) -> np.ndarray:
    z = np.array([scores.get(b, 0.0) for b in BRANCHES], dtype=np.float32) / max(temperature, 1e-6)
    z = z - np.nanmax(z)
    p = np.exp(z)
    if not np.isfinite(p).all() or p.sum() <= 0:
        p = np.ones(len(BRANCHES), dtype=np.float32)
    return p / p.sum()


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
    acc = finite_difference(finite_difference(agent_traj[:, 3:4] if agent_traj.shape[-1] > 3 else np.linalg.norm(finite_difference(agent_traj[:, :2], dt), axis=-1, keepdims=True), dt), dt).reshape(-1)
    max_brake = float(np.nanmax(-acc)) if acc.size else 0.0
    max_accel = float(np.nanmax(acc)) if acc.size else 0.0
    n = min(len(agent_traj), len(baseline_agent_traj))
    ade = float(np.nanmean(np.linalg.norm(agent_traj[:n, :2] - baseline_agent_traj[:n, :2], axis=-1))) if n else 0.0
    min_dist = float(np.nanmin(np.linalg.norm(ego_traj[: min(len(ego_traj), len(agent_traj)), :2] - agent_traj[: min(len(ego_traj), len(agent_traj)), :2], axis=-1)))
    if min_dist > nonconflict_distance:
        return softmax_scores({"nonconflict": 4.0, "keep": 1.0})
    scores = {
        "cede": max(0.0, tau_agent_in - tau_agent_base_in) + (1.0 if tau_ego_in < tau_agent_in else 0.0),
        "brake": max(0.0, max_brake - 2.0),
        "accelerate": max(0.0, max_accel - 1.5) + (0.5 if tau_agent_in < tau_ego_in else 0.0),
        "pass": 1.5 if tau_agent_in + 0.5 < tau_ego_in else 0.0,
        "keep": float(np.exp(-abs(tau_agent_in - tau_agent_base_in) / 1.0 - ade / 3.0)),
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
    speed = agent_traj[:, 3] if agent_traj.shape[-1] > 3 else np.linalg.norm(finite_difference(agent_traj[:, :2], dt), axis=-1)
    bspeed = baseline_agent_traj[:, 3] if baseline_agent_traj.shape[-1] > 3 else np.linalg.norm(finite_difference(baseline_agent_traj[:, :2], dt), axis=-1)
    acc = finite_difference(speed[:, None], dt).reshape(-1)
    bacc = finite_difference(bspeed[:, None], dt).reshape(-1)
    jerk = finite_difference(acc[:, None], dt).reshape(-1)
    bjerk = finite_difference(bacc[:, None], dt).reshape(-1)
    n = min(len(agent_traj), len(baseline_agent_traj))
    delay = max(0.0, float(tau_agent_in - tau_agent_base_in)) / scales["delay"]
    dec = max(0.0, float(np.nanmax(-acc) - np.nanmax(-bacc))) / scales["dec"]
    jr = max(0.0, float(np.sqrt(np.nanmean(jerk**2)) - np.sqrt(np.nanmean(bjerk**2)))) / scales["jerk"]
    dev = float(np.nanmean(np.linalg.norm(agent_traj[:n, :2] - baseline_agent_traj[:n, :2], axis=-1))) / scales["dev"] if n else 0.0
    return float(delay + dec + 0.5 * jr + 0.5 * dev)


def high_pressure_label(burden: float, agent_traj: np.ndarray, *, dt: float = 0.1, eta_b: float = 1.0, hard_brake: float = 4.0, hard_jerk: float = 5.0) -> bool:
    speed = agent_traj[:, 3] if agent_traj.shape[-1] > 3 else np.linalg.norm(finite_difference(agent_traj[:, :2], dt), axis=-1)
    acc = finite_difference(speed[:, None], dt).reshape(-1)
    jerk = finite_difference(acc[:, None], dt).reshape(-1)
    return bool(burden >= eta_b or np.nanmax(-acc) >= hard_brake or np.sqrt(np.nanmean(jerk**2)) >= hard_jerk)


def safety_margin(ego_traj: np.ndarray, agent_traj: np.ndarray, *, ego_radius: float = 2.2, agent_radius: float = 2.2) -> float:
    return signed_point_margin(ego_traj, agent_traj, ego_radius, agent_radius)
