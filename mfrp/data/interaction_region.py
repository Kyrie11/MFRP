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
