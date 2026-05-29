from __future__ import annotations

import numpy as np


def finite_difference(x: np.ndarray, dt: float = 0.1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) < 2:
        return np.zeros_like(x)
    return np.gradient(x, dt, axis=0)


def signed_point_margin(ego_xy: np.ndarray, agent_xy: np.ndarray, ego_radius: float = 2.2, agent_radius: float = 2.2) -> float:
    """Conservative signed separation proxy. Adapters should replace this with oriented boxes."""
    ego_xy = np.asarray(ego_xy)[..., :2]
    agent_xy = np.asarray(agent_xy)[..., :2]
    n = min(len(ego_xy), len(agent_xy))
    if n == 0:
        return float("nan")
    d = np.linalg.norm(ego_xy[:n] - agent_xy[:n], axis=-1) - (ego_radius + agent_radius)
    return float(np.nanmin(d))


def closest_entry_times(ego_xy: np.ndarray, agent_xy: np.ndarray, threshold: float = 6.0, dt: float = 0.1) -> tuple[float, float, float]:
    ego_xy = np.asarray(ego_xy)[..., :2]
    agent_xy = np.asarray(agent_xy)[..., :2]
    n = min(len(ego_xy), len(agent_xy))
    if n == 0:
        return float("inf"), float("inf"), float("inf")
    d = np.linalg.norm(ego_xy[:n] - agent_xy[:n], axis=-1)
    idx = np.where(d <= threshold)[0]
    if idx.size == 0:
        j = int(np.nanargmin(d))
        return (j * dt, j * dt, float(d[j]))
    j = int(idx[0])
    return (j * dt, j * dt, float(d[j]))

from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class InteractionRegion:
    tau_ego_in: float
    tau_agent_in: float
    min_distance: float
    features: dict[str, Any]


def constant_velocity_extrapolate(history: np.ndarray, future_steps: int, dt: float = 0.1) -> np.ndarray:
    hist = np.asarray(history, dtype=np.float32)
    valid = hist[:, -1] > 0.5 if hist.ndim == 2 and hist.shape[-1] > 9 else np.ones(len(hist), dtype=bool)
    last = hist[valid][-1] if hist.ndim == 2 and valid.any() else (hist[-1] if hist.ndim == 2 and len(hist) else np.zeros(10, dtype=np.float32))
    out = np.repeat(last[None, :], int(future_steps), axis=0).astype(np.float32)
    vx = float(last[3]) if len(last) > 3 else 0.0
    vy = float(last[4]) if len(last) > 4 else 0.0
    for t in range(int(future_steps)):
        out[t, 0] = last[0] + vx * dt * (t + 1)
        out[t, 1] = last[1] + vy * dt * (t + 1)
    return out


def build_interaction_region(ego_traj: np.ndarray, agent_traj: np.ndarray, *, dt: float = 0.1, threshold: float = 6.0) -> InteractionRegion:
    tau_e, tau_a, d = closest_entry_times(ego_traj, agent_traj, threshold=threshold, dt=dt)
    features = {
        "tau_ego_in": tau_e,
        "tau_agent_in": tau_a,
        "tau_i0_in": tau_a,
        "tau_i0_out": tau_a,
        "entry_time_gap": tau_a - tau_e if np.isfinite(tau_a) and np.isfinite(tau_e) else 0.0,
        "min_distance": d,
    }
    return InteractionRegion(tau_e, tau_a, d, features)
