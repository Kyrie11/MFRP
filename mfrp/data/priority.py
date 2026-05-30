from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PriorityScore:
    score: float
    confidence: float
    diagnostics: dict


def compute_preexec_priority(features: dict) -> PriorityScore:
    """Deployment-safe priority estimate from pre-execution metadata only.

    The score is ego-priority probability.  Arrival timing can contribute, but
    missing map/route/traffic-control evidence explicitly caps confidence so an
    unknown priority scene is not interpreted as strong evidence of coercion.
    """
    if not features:
        return PriorityScore(0.5, 0.20, {"reason": "missing_preexec_features"})
    gap = features.get("entry_time_gap", features.get("gap", None))
    ego_has_right_of_way = features.get("ego_has_right_of_way", None)
    agent_has_right_of_way = features.get("agent_has_right_of_way", None)
    has_route = bool(features.get("has_route_context", False))
    has_controls = bool(features.get("has_traffic_controls", False))
    score = 0.5
    conf = 0.25
    diagnostics = {"source": "preexec", "has_route_context": has_route, "has_traffic_controls": has_controls}
    if gap is not None and np.isfinite(gap):
        # Positive gap means agent is predicted to enter first -> agent priority, ego lower priority.
        score = 1.0 / (1.0 + np.exp(float(gap)))
        conf = max(conf, min(0.75, abs(float(gap)) / 2.0))
    if ego_has_right_of_way is True:
        score = max(score, 0.8)
        conf = max(conf, 0.85)
        diagnostics["right_of_way"] = "ego"
    if agent_has_right_of_way is True:
        score = min(score, 0.2)
        conf = max(conf, 0.85)
        diagnostics["right_of_way"] = "agent"
    if ego_has_right_of_way is True and agent_has_right_of_way is True:
        # Conflicting rules should not create high-confidence priority.
        score = 0.5
        conf = min(conf, 0.35)
        diagnostics["right_of_way"] = "conflict"
    if not (has_route or has_controls or ego_has_right_of_way is not None or agent_has_right_of_way is not None):
        conf = min(conf, 0.45)
        diagnostics["reason"] = "missing_route_and_traffic_control_context"
    return PriorityScore(float(np.clip(score, 0, 1)), float(np.clip(conf, 0, 1)), diagnostics)


def compute_priority_score(features: dict) -> PriorityScore:
    return compute_preexec_priority(features)
