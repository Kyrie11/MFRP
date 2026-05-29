from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PriorityScore:
    score: float
    confidence: float
    diagnostics: dict


def compute_preexec_priority(features: dict) -> PriorityScore:
    """Deployment-safe priority estimate from pre-execution metadata only."""
    if not features:
        return PriorityScore(0.5, 0.25, {"reason": "missing_preexec_features"})
    gap = features.get("entry_time_gap", features.get("gap", None))
    ego_has_right_of_way = features.get("ego_has_right_of_way", None)
    agent_has_right_of_way = features.get("agent_has_right_of_way", None)
    score = 0.5
    conf = 0.3
    if gap is not None and np.isfinite(gap):
        # Positive gap means agent is predicted to enter first -> agent priority, ego lower priority.
        score = 1.0 / (1.0 + np.exp(float(gap)))
        conf = max(conf, min(1.0, abs(float(gap)) / 2.0))
    if ego_has_right_of_way is True:
        score = max(score, 0.8); conf = max(conf, 0.8)
    if agent_has_right_of_way is True:
        score = min(score, 0.2); conf = max(conf, 0.8)
    return PriorityScore(float(np.clip(score, 0, 1)), float(np.clip(conf, 0, 1)), {"source": "preexec"})


def compute_priority_score(features: dict) -> PriorityScore:
    return compute_preexec_priority(features)
