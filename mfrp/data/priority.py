"""Pre-execution priority features for MFRP."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PriorityResult:
    score: float
    confidence: float
    features: np.ndarray
    feature_names: list[str]
    diagnostics: dict[str, Any]


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def priority_features(interaction_features: dict[str, Any] | None = None) -> tuple[np.ndarray, list[str], float, dict[str, Any]]:
    f = interaction_features or {}
    missing = 0
    def get(name: str, default: float = 0.0) -> float:
        nonlocal missing
        v = f.get(name, None)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            missing += 1
            return default
        return float(v)
    tau_e = get("tau_e_in", 0.0)
    tau_i0 = get("tau_i0_in", tau_e)
    route_alignment = get("route_alignment", 0.0)
    ego_protected = get("ego_protected", 0.0)
    agent_protected = get("agent_protected", 0.0)
    ego_yield = get("ego_yield_control", 0.0)
    agent_yield = get("agent_yield_control", 0.0)
    merge_priority = get("merge_priority", 0.0)
    names = [
        "bias", "agent_before_ego", "ego_before_agent", "route_alignment",
        "ego_protected", "agent_protected", "ego_yield_control", "agent_yield_control", "merge_priority",
    ]
    x = np.asarray([
        1.0,
        max(0.0, tau_e - tau_i0),
        max(0.0, tau_i0 - tau_e),
        route_alignment,
        ego_protected,
        agent_protected,
        ego_yield,
        agent_yield,
        merge_priority,
    ], dtype=np.float32)
    confidence = float(np.clip(1.0 - 0.12 * missing, 0.1, 1.0))
    return x, names, confidence, {"missing_priority_feature_count": missing}


def compute_priority_score(interaction_features: dict[str, Any] | None = None, weights: np.ndarray | None = None, missing_priority_discount: float = 0.5) -> PriorityResult:
    x, names, conf, diag = priority_features(interaction_features)
    if weights is None:
        weights = np.asarray([-0.2, -0.55, 0.45, 0.25, 0.7, -0.7, -0.55, 0.45, 0.25], dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if weights.shape != x.shape:
        raise ValueError(f"priority weights must be {x.shape}, got {weights.shape}")
    raw = float(weights @ x)
    if diag["missing_priority_feature_count"] >= 4:
        return PriorityResult(0.5, conf * missing_priority_discount, x, names, {**diag, "raw": raw, "missing_defaulted": True})
    return PriorityResult(float(np.clip(sigmoid(raw), 0.0, 1.0)), conf, x, names, {**diag, "raw": raw})
