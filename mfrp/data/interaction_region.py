from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


def finite_difference(x: np.ndarray, dt: float = 0.1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) < 2:
        return np.zeros_like(x)
    return np.gradient(x, dt, axis=0)


def _valid_xy(traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(traj, dtype=np.float32)
    xy = arr[..., :2]
    if arr.ndim >= 2 and arr.shape[-1] > 9:
        mask = np.asarray(arr[..., 9] > 0.5, dtype=bool)
    else:
        mask = np.isfinite(xy).all(axis=-1)
    mask &= np.isfinite(xy).all(axis=-1)
    return xy, mask


def signed_point_margin(ego_xy: np.ndarray, agent_xy: np.ndarray, ego_radius: float = 2.2, agent_radius: float = 2.2) -> float:
    """Conservative signed separation proxy over synchronized valid timesteps."""
    ego_xy, em = _valid_xy(ego_xy)
    agent_xy, am = _valid_xy(agent_xy)
    n = min(len(ego_xy), len(agent_xy))
    if n == 0:
        return float("nan")
    mask = em[:n] & am[:n]
    if not mask.any():
        return float("nan")
    d = np.linalg.norm(ego_xy[:n][mask] - agent_xy[:n][mask], axis=-1) - (ego_radius + agent_radius)
    return float(np.nanmin(d))


def _closest_pair_conflict_center(ego_xy: np.ndarray, agent_xy: np.ndarray, em: np.ndarray, am: np.ndarray) -> tuple[np.ndarray, float]:
    ei = np.where(em)[0]
    ai = np.where(am)[0]
    if ei.size == 0 or ai.size == 0:
        return np.zeros(2, dtype=np.float32), float("inf")
    # Full pairwise distance finds the spatial conflict point independently of synchronized time.
    de = ego_xy[ei][:, None, :] - agent_xy[ai][None, :, :]
    d2 = np.sum(de * de, axis=-1)
    flat = int(np.nanargmin(d2))
    i, j = np.unravel_index(flat, d2.shape)
    p_e = ego_xy[ei[i]]
    p_a = agent_xy[ai[j]]
    return ((p_e + p_a) * 0.5).astype(np.float32), float(np.sqrt(d2[i, j]))


def closest_entry_times(ego_xy: np.ndarray, agent_xy: np.ndarray, threshold: float = 6.0, dt: float = 0.1) -> tuple[float, float, float]:
    """Estimate separate entry times into a shared spatial conflict disk.

    The old implementation returned the same synchronized timestep for ego and
    agent, which collapsed entry_time_gap to ~0.  This version first identifies a
    shared conflict location by the closest spatial approach over all valid
    timesteps, then computes each actor's first entry time into that region.
    """
    ego_xy, em = _valid_xy(ego_xy)
    agent_xy, am = _valid_xy(agent_xy)
    n_sync = min(len(ego_xy), len(agent_xy))
    if n_sync == 0 or not em.any() or not am.any():
        return float("inf"), float("inf"), float("inf")
    center, min_pair_dist = _closest_pair_conflict_center(ego_xy, agent_xy, em, am)
    radius = max(float(threshold), 1e-3)
    de = np.linalg.norm(ego_xy - center[None, :], axis=-1)
    da = np.linalg.norm(agent_xy - center[None, :], axis=-1)
    ei = np.where((de <= radius) & em)[0]
    ai = np.where((da <= radius) & am)[0]
    tau_e = float(ei[0] * dt) if ei.size else float("inf")
    tau_a = float(ai[0] * dt) if ai.size else float("inf")
    sync_mask = em[:n_sync] & am[:n_sync]
    if sync_mask.any():
        min_sync = float(np.nanmin(np.linalg.norm(ego_xy[:n_sync][sync_mask] - agent_xy[:n_sync][sync_mask], axis=-1)))
    else:
        min_sync = min_pair_dist
    return tau_e, tau_a, min_sync


@dataclass(frozen=True)
class InteractionRegion:
    tau_ego_in: float
    tau_agent_in: float
    min_distance: float
    features: dict[str, Any]


def constant_velocity_extrapolate(history: np.ndarray, future_steps: int, dt: float = 0.1) -> np.ndarray:
    hist = np.asarray(history, dtype=np.float32)
    if hist.ndim != 2 or len(hist) == 0:
        last = np.zeros(10, dtype=np.float32)
    else:
        valid = hist[:, -1] > 0.5 if hist.shape[-1] > 9 else np.isfinite(hist[:, :2]).all(axis=-1)
        last = hist[valid][-1] if valid.any() else hist[-1]
    out = np.repeat(last[None, :], int(future_steps), axis=0).astype(np.float32)
    vx = float(last[3]) if len(last) > 3 else 0.0
    vy = float(last[4]) if len(last) > 4 else 0.0
    for t in range(int(future_steps)):
        out[t, 0] = last[0] + vx * dt * (t + 1)
        out[t, 1] = last[1] + vy * dt * (t + 1)
        if out.shape[-1] > 5:
            out[t, 5] = float(np.hypot(vx, vy))
        if out.shape[-1] > 6 and abs(vx) + abs(vy) > 1e-6:
            out[t, 6] = float(np.arctan2(vy, vx))
        if out.shape[-1] > 9:
            out[t, 9] = 1.0
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
