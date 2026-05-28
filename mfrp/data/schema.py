from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import hashlib
import json
import numpy as np

BRANCHES = ("keep", "cede", "brake", "accelerate", "pass", "nonconflict")
CEDING_BRANCHES = ("cede", "brake")


def stable_hash(payload: Mapping[str, Any] | str) -> str:
    if isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


@dataclass(frozen=True)
class RootScene:
    """Observed root scene only. Do not put future labels here."""

    scene_id: str
    t0: int | float
    history: np.ndarray  # [N, H, state_dim]
    history_mask: np.ndarray  # [N, H]
    ego_index: int = 0
    map_features: np.ndarray | None = None
    traffic_controls: np.ndarray | None = None
    route_features: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def root_hash(self) -> str:
        safe = {"scene_id": self.scene_id, "t0": self.t0, "ego_index": self.ego_index}
        return stable_hash(safe)


@dataclass(frozen=True)
class EgoCandidate:
    candidate_id: str
    trajectory: np.ndarray  # [T, state_dim], ego frame at t0
    features: np.ndarray  # deployment-safe a_i(u;s) base features, may be agent-independent
    nominal_cost: float = 0.0
    valid: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResponseObservation:
    candidate_id: str
    agent_id: str
    variant_id: str
    branch_probs: np.ndarray  # [6]
    trajectory: np.ndarray  # [T, state_dim]
    trajectory_mask: np.ndarray  # [T]
    burden: float
    safety_margin: float
    high_pressure: bool
    cw_soft_label: float = 0.0
    cw_confidence: float = 0.0
    priority_score_preexec: float = 0.5
    priority_confidence_preexec: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        bp = np.asarray(self.branch_probs, dtype=np.float32)
        if bp.shape != (len(BRANCHES),):
            raise ValueError(f"branch_probs must have shape {(len(BRANCHES),)}, got {bp.shape}")
        if not np.isfinite(bp).all() or bp.sum() <= 0:
            raise ValueError("branch_probs must be finite and non-empty")
        if not np.isfinite(float(self.burden)) or not np.isfinite(float(self.safety_margin)):
            raise ValueError("burden and safety_margin must be finite")
        if not (0.0 <= float(self.priority_score_preexec) <= 1.0):
            raise ValueError("priority_score_preexec must be in [0,1]")
        if not (0.0 <= float(self.priority_confidence_preexec) <= 1.0):
            raise ValueError("priority_confidence_preexec must be in [0,1]")


@dataclass
class SameRootGroup:
    root_scene: RootScene
    candidates: list[EgoCandidate]
    relevant_agent_ids: list[str]
    rollout_variants: list[str]
    observations: dict[tuple[str, str, str], ResponseObservation]
    metadata: dict[str, Any] = field(default_factory=dict)
    boundary_pairs: list[tuple[str, str, str, float]] = field(default_factory=list)  # agent_id, cand_a, cand_b, distance

    def validate(self, *, require_support_query_split: bool = True, allow_debug: bool = False) -> None:
        if not self.candidates:
            raise ValueError("SameRootGroup has no candidates")
        if not self.relevant_agent_ids:
            raise ValueError("SameRootGroup has no relevant agents")
        if not self.rollout_variants:
            raise ValueError("SameRootGroup has no rollout variants")
        if self.metadata.get("uses_log_playback_for_response") and not allow_debug:
            raise ValueError("log playback cannot be used as response supervision for paper data")
        if len(self.rollout_variants) < 2 and not allow_debug:
            raise ValueError("forced-dependence supervision needs at least two rollout variants")
        cand_ids = {c.candidate_id for c in self.candidates}
        agent_ids = set(self.relevant_agent_ids)
        variant_ids = set(self.rollout_variants)
        for key, obs in self.observations.items():
            c, a, r = key
            if c not in cand_ids or a not in agent_ids or r not in variant_ids:
                raise ValueError(f"observation key {key} is outside group ids")
            obs.validate()
        if require_support_query_split:
            support = set(self.metadata.get("support_candidate_ids", []))
            query = set(self.metadata.get("query_candidate_ids", []))
            if not support or not query:
                raise ValueError("metadata must contain non-empty support_candidate_ids and query_candidate_ids")
            if support & query:
                raise ValueError("support/query candidate ids overlap")
            if not support <= cand_ids or not query <= cand_ids:
                raise ValueError("support/query ids must be candidate ids")
