from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import hashlib
import json
import numpy as np

# Keep this as a list: several downstream configs/tests compare the exact public API.
BRANCHES = ["keep", "cede", "brake", "accelerate", "pass", "nonconflict"]
CEDING_BRANCHES = ("cede", "brake")
BRANCH_INDEX = {b: i for i, b in enumerate(BRANCHES)}


def stable_hash(payload: Mapping[str, Any] | str) -> str:
    if isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def root_scene_hash(scene_id: str, t0: int | float, ego_index: int = 0) -> str:
    return stable_hash({"scene_id": scene_id, "t0": t0, "ego_index": ego_index})


@dataclass(frozen=True)
class AgentTrackTensor:
    track_id: str
    states: np.ndarray
    mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteContext:
    route_features: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RootScene:
    """Observed root scene only. Do not put future labels here."""

    scene_id: str
    t0: int | float
    history: np.ndarray
    history_mask: np.ndarray
    ego_index: int = 0
    map_features: np.ndarray | None = None
    traffic_controls: np.ndarray | None = None
    route_features: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    dt: float = 0.1
    current_time_index: int | None = None
    agent_tracks: list[AgentTrackTensor] | None = None
    route_context: RouteContext | None = None

    @property
    def root_hash(self) -> str:
        return root_scene_hash(self.scene_id, self.t0, self.ego_index)

    def root_state(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "t0": self.t0,
            "history": self.history,
            "history_mask": self.history_mask,
            "ego_index": self.ego_index,
            "dt": self.dt,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EgoCandidate:
    candidate_id: str
    trajectory: np.ndarray
    features: np.ndarray | None = None
    nominal_cost: float = 0.0
    valid: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    family: str | None = None

    def __post_init__(self):
        if self.features is None:
            object.__setattr__(self, "features", np.zeros(20, dtype=np.float32))


@dataclass(frozen=True)
class BoundaryPair:
    agent_id: str
    candidate_a: str
    candidate_b: str
    distance: float


@dataclass(frozen=True)
class CandidateValidity:
    valid: bool
    reason: str = ""


@dataclass(init=False, frozen=True)
class ResponseObservation:
    """One rollout observation for (root, candidate, agent, variant).

    Supports both the compact current constructor:
      ResponseObservation(candidate_id, agent_id, variant_id, branch_probs, ...)
    and the legacy constructor used in older tests/adapters:
      ResponseObservation(scene_id, root_hash, candidate_id, agent_id, variant_id,
                          branch_probs, branch_hard, trajectory, ...)
    """

    candidate_id: str
    agent_id: str
    variant_id: str
    branch_probs: np.ndarray
    trajectory: np.ndarray
    trajectory_mask: np.ndarray
    burden: float
    safety_margin: float
    high_pressure: bool
    cw_soft_label: float
    cw_confidence: float
    priority_score_preexec: float
    priority_confidence_preexec: float
    diagnostics: dict[str, Any]
    scene_id: str | None
    root_hash: str | None
    branch_hard: int
    tau_agent_in: float | None

    def __init__(self, *args, **kwargs):
        scene_id = kwargs.pop("scene_id", None)
        root_hash = kwargs.pop("root_hash", None)
        branch_hard = kwargs.pop("branch_hard", None)
        tau_agent_in = kwargs.pop("tau_agent_in", None)
        diagnostics = kwargs.pop("diagnostics", {})

        if len(args) >= 15:
            # legacy positional form
            scene_id, root_hash, candidate_id, agent_id, variant_id, branch_probs, branch_hard, trajectory, trajectory_mask, burden, safety_margin, tau_agent_in, high_pressure, priority_score_preexec, priority_confidence_preexec, *rest = args
            cw_soft_label = kwargs.pop("cw_soft_label", rest[0] if len(rest) > 0 else 0.0)
            cw_confidence = kwargs.pop("cw_confidence", rest[1] if len(rest) > 1 else 0.0)
        elif len(args) >= 6:
            candidate_id, agent_id, variant_id, branch_probs, trajectory, trajectory_mask, *rest = args
            burden = kwargs.pop("burden", rest[0] if len(rest) > 0 else 0.0)
            safety_margin = kwargs.pop("safety_margin", rest[1] if len(rest) > 1 else 0.0)
            high_pressure = kwargs.pop("high_pressure", rest[2] if len(rest) > 2 else False)
            cw_soft_label = kwargs.pop("cw_soft_label", rest[3] if len(rest) > 3 else 0.0)
            cw_confidence = kwargs.pop("cw_confidence", rest[4] if len(rest) > 4 else 0.0)
            priority_score_preexec = kwargs.pop("priority_score_preexec", rest[5] if len(rest) > 5 else 0.5)
            priority_confidence_preexec = kwargs.pop("priority_confidence_preexec", rest[6] if len(rest) > 6 else 0.0)
        else:
            candidate_id = kwargs.pop("candidate_id")
            agent_id = kwargs.pop("agent_id")
            variant_id = kwargs.pop("variant_id")
            branch_probs = kwargs.pop("branch_probs")
            trajectory = kwargs.pop("trajectory")
            trajectory_mask = kwargs.pop("trajectory_mask")
            burden = kwargs.pop("burden")
            safety_margin = kwargs.pop("safety_margin")
            high_pressure = kwargs.pop("high_pressure")
            cw_soft_label = kwargs.pop("cw_soft_label", 0.0)
            cw_confidence = kwargs.pop("cw_confidence", 0.0)
            priority_score_preexec = kwargs.pop("priority_score_preexec", 0.5)
            priority_confidence_preexec = kwargs.pop("priority_confidence_preexec", 0.0)
        if kwargs:
            raise TypeError(f"Unexpected ResponseObservation kwargs: {sorted(kwargs)}")

        bp = np.asarray(branch_probs, dtype=np.float32)
        if bp.shape != (len(BRANCHES),):
            bp = np.resize(bp.reshape(-1), len(BRANCHES)).astype(np.float32)
        s = float(bp.sum())
        bp = bp / s if np.isfinite(s) and s > 0 else np.ones(len(BRANCHES), dtype=np.float32) / len(BRANCHES)
        if branch_hard is None or int(branch_hard) < 0:
            branch_hard = int(np.argmax(bp))

        object.__setattr__(self, "candidate_id", str(candidate_id))
        object.__setattr__(self, "agent_id", str(agent_id))
        object.__setattr__(self, "variant_id", str(variant_id))
        object.__setattr__(self, "branch_probs", bp)
        object.__setattr__(self, "trajectory", np.asarray(trajectory, dtype=np.float32))
        object.__setattr__(self, "trajectory_mask", np.asarray(trajectory_mask, dtype=bool))
        object.__setattr__(self, "burden", float(burden))
        object.__setattr__(self, "safety_margin", float(safety_margin))
        object.__setattr__(self, "high_pressure", bool(high_pressure))
        object.__setattr__(self, "cw_soft_label", float(cw_soft_label))
        object.__setattr__(self, "cw_confidence", float(cw_confidence))
        object.__setattr__(self, "priority_score_preexec", float(priority_score_preexec))
        object.__setattr__(self, "priority_confidence_preexec", float(priority_confidence_preexec))
        object.__setattr__(self, "diagnostics", diagnostics or {})
        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(self, "root_hash", root_hash)
        object.__setattr__(self, "branch_hard", int(branch_hard))
        object.__setattr__(self, "tau_agent_in", None if tau_agent_in is None else float(tau_agent_in))
        self.validate()

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
    boundary_pairs: list[tuple[str, str, str, float] | BoundaryPair] = field(default_factory=list)

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
