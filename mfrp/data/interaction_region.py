"""Interaction-region and intervention-coordinate utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class InteractionRegion:
    kind: str
    center: np.ndarray
    radius: float
    tau_e_in: float
    tau_e_out: float
    tau_i0_in: float | None
    tau_i0_out: float | None
    features: dict[str, Any]


def _entry_exit(path: np.ndarray, center: np.ndarray, radius: float, dt: float, censored_time: float) -> tuple[float, float, bool]:
    xy = np.asarray(path, dtype=np.float32)[..., :2]
    d = np.linalg.norm(xy - center[None, :], axis=-1)
    idx = np.where(d <= radius)[0]
    if len(idx) == 0:
        return censored_time, censored_time, False
    return float(idx[0] * dt), float(idx[-1] * dt), True


def constant_velocity_extrapolate(history: np.ndarray | None, steps: int, dt: float = 0.1) -> np.ndarray | None:
    """Deployment-safe neutral agent rollout from observed history only.

    WOMD labels contain future tracks, but deployment cannot use those. This helper
    extrapolates the last valid observed state with constant velocity so
    interaction-coordinate construction does not leak response labels.
    """
    if history is None or steps <= 0:
        return None
    h = np.asarray(history, dtype=np.float32)
    if h.ndim != 2 or h.shape[-1] < 2 or len(h) == 0:
        return None
    valid = h[:, 9] > 0.5 if h.shape[-1] > 9 else np.ones(len(h), dtype=bool)
    if not valid.any():
        return None
    last = h[np.where(valid)[0][-1]].copy()
    out = np.repeat(last[None, :], steps, axis=0)
    vx = float(last[3]) if h.shape[-1] > 3 and np.isfinite(last[3]) else 0.0
    vy = float(last[4]) if h.shape[-1] > 4 and np.isfinite(last[4]) else 0.0
    speed = float(last[5]) if h.shape[-1] > 5 and np.isfinite(last[5]) else float(np.hypot(vx, vy))
    if (abs(vx) + abs(vy)) < 1e-4 and speed > 0.0 and h.shape[-1] > 6:
        vx, vy = speed * float(np.cos(last[6])), speed * float(np.sin(last[6]))
    t = np.arange(1, steps + 1, dtype=np.float32) * float(dt)
    out[:, 0] = last[0] + vx * t
    out[:, 1] = last[1] + vy * t
    if h.shape[-1] > 3:
        out[:, 3] = vx
    if h.shape[-1] > 4:
        out[:, 4] = vy
    if h.shape[-1] > 5:
        out[:, 5] = float(np.hypot(vx, vy))
    if h.shape[-1] > 9:
        out[:, 9] = 1.0
    return out


def build_interaction_region(ego_future: np.ndarray, agent_ref: np.ndarray | None, dt: float = 0.1, delta_cens_s: float = 1.0, gap_threshold: float = 30.0) -> InteractionRegion:
    ego = np.asarray(ego_future, dtype=np.float32)
    if ego.ndim != 2 or ego.shape[-1] < 2:
        raise ValueError("ego_future must be [T,D>=2]")
    horizon = float(len(ego) * dt + delta_cens_s)
    if agent_ref is None or len(agent_ref) == 0:
        center = ego[min(len(ego) - 1, len(ego) // 2), :2].copy()
        te0, te1, _ = _entry_exit(ego, center, gap_threshold * 0.5, dt, horizon)
        return InteractionRegion("no_agent_reference", center, gap_threshold * 0.5, te0, te1, None, None, {"censored": True})
    agent = np.asarray(agent_ref, dtype=np.float32)
    n = min(len(ego), len(agent))
    d = np.linalg.norm(ego[:n, :2] - agent[:n, :2], axis=-1)
    j = int(np.argmin(d)) if n else 0
    center = 0.5 * (ego[j, :2] + agent[j, :2])
    radius = max(3.0, min(gap_threshold, float(d[j]) + 3.0))
    te0, te1, ego_entered = _entry_exit(ego, center, radius, dt, horizon)
    ti0, ti1, agent_entered = _entry_exit(agent, center, radius, dt, horizon)
    kind = "crossing" if float(d[j]) < 5.0 else "lead_follow_or_adjacent_gap"
    features = {
        "tau_e_in": te0, "tau_e_out": te1, "tau_i0_in": ti0, "tau_i0_out": ti1,
        "relative_gap": float(d[j]), "gap": float(d[j]), "ego_entered": ego_entered,
        "agent_entered_neutral": agent_entered, "interaction_type_crossing": kind == "crossing",
        "interaction_type_lead_follow": kind != "crossing", "censored": not (ego_entered and agent_entered),
    }
    return InteractionRegion(kind, center, radius, te0, te1, ti0, ti1, features)


def intervention_coordinate(ego_future: np.ndarray, interaction: InteractionRegion, priority_score: float = 0.5, priority_confidence: float = 1.0, extra_context: np.ndarray | None = None) -> np.ndarray:
    ego = np.asarray(ego_future, dtype=np.float32)
    first = ego[0, [0, 1, 5, 6]] if ego.shape[-1] > 6 else np.zeros(4, dtype=np.float32)
    mid = ego[len(ego) // 2, [0, 1, 5, 6]] if ego.shape[-1] > 6 else np.zeros(4, dtype=np.float32)
    last = ego[-1, [0, 1, 5, 6]] if ego.shape[-1] > 6 else np.zeros(4, dtype=np.float32)
    base = np.asarray([
        *first, *mid, *last,
        interaction.tau_e_in, interaction.tau_e_out,
        -1.0 if interaction.tau_i0_in is None else interaction.tau_i0_in,
        -1.0 if interaction.tau_i0_out is None else interaction.tau_i0_out,
        float(interaction.features.get("gap", 0.0)),
        float(priority_score), float(priority_confidence),
        float(interaction.features.get("interaction_type_crossing", False)),
        float(interaction.features.get("interaction_type_lead_follow", False)),
        float(interaction.features.get("censored", False)),
    ], dtype=np.float32)
    if extra_context is not None:
        base = np.concatenate([base, np.asarray(extra_context, dtype=np.float32).reshape(-1)], axis=0)
    return base
