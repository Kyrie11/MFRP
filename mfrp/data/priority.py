from __future__ import annotations

import math
from typing import Mapping, Any


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))


def compute_priority_score_preexec(features: Mapping[str, Any]) -> tuple[float, float]:
    """Deployment-safe priority score from pre-execution metadata only.

    Missing metadata returns uncertain prior rather than falsely implying low ego priority.
    """
    known = 0
    z = 0.0
    for key, weight in (
        ("ego_has_right_of_way", 1.4),
        ("agent_has_yield_or_stop", 1.1),
        ("ego_protected_lane", 0.9),
        ("ego_route_aligned", 0.5),
    ):
        if key in features:
            known += 1
            z += weight * (1.0 if bool(features[key]) else -0.6)
    for key, weight in (("agent_has_right_of_way", -1.4), ("ego_has_yield_or_stop", -1.1), ("agent_protected_lane", -0.9)):
        if key in features:
            known += 1
            z += weight * (1.0 if bool(features[key]) else -0.6)
    if "neutral_entry_gap" in features:
        known += 1
        gap = float(features["neutral_entry_gap"])
        z += max(-2.0, min(2.0, gap)) * 0.35
    if known == 0:
        return 0.5, 0.0
    confidence = min(1.0, known / 5.0)
    return sigmoid(z), confidence
