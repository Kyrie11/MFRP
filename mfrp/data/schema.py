"""Canonical MFRP data schema.

This module contains only deployment-safe schema definitions and label
containers for Mechanism-Feasible Response Planning (MFRP).  The response
primitive is O_i(u)=(M_i,Y_i,DeltaB_i,H_i): soft branch, ego-frame future
trajectory, continuous baseline-relative burden, and signed safety margin.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mfrp.data.scene_schema import RootScene

BRANCHES = ["keep", "cede", "brake", "accelerate", "pass", "nonconflict"]
BRANCH_TO_INDEX = {name: i for i, name in enumerate(BRANCHES)}
INDEX_TO_BRANCH = {i: name for name, i in BRANCH_TO_INDEX.items()}
CEDING_BRANCHES = ["cede", "brake"]
CEDING_BRANCH_IDS = [BRANCH_TO_INDEX[b] for b in CEDING_BRANCHES]
STATE_DIM = 10  # x, y, z, vx, vy, speed, yaw, length, width, valid
TRAJ_TARGET_DIM = 5  # canonical ego-centered t0 target: x, y, vx, vy, yaw
DEFAULT_DT = 0.1
DEFAULT_FUTURE_STEPS = 80


def _as_f32(x: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"expected shape {shape}, got {arr.shape}")
    return arr


def normalize_branch_probs(probs: Any, valid: bool = True) -> np.ndarray:
    arr = _as_f32(probs)
    if arr.shape != (len(BRANCHES),):
        raise ValueError(f"branch_probs must be [{len(BRANCHES)}], got {arr.shape}")
    arr = np.where(np.isfinite(arr), np.maximum(arr, 0.0), 0.0)
    s = float(arr.sum())
    if s <= 1e-8 or not valid:
        return np.full(len(BRANCHES), 1.0 / len(BRANCHES), dtype=np.float32)
    return (arr / s).astype(np.float32)


@dataclass
class CandidateValidity:
    valid: bool = True
    kinematic_feasible: bool = True
    route_feasible: bool = True
    map_feasible: bool = True
    reason: str = ""


@dataclass
class EgoCandidate:
    candidate_id: str
    family: str
    future_states_ego_frame: np.ndarray  # [T, 10], t0 ego frame
    controls: np.ndarray | None = None
    route_ids: list[int] = field(default_factory=list)
    validity: CandidateValidity = field(default_factory=CandidateValidity)
    nominal_cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.future_states_ego_frame = _as_f32(self.future_states_ego_frame)
        if self.future_states_ego_frame.ndim != 2 or self.future_states_ego_frame.shape[-1] != STATE_DIM:
            raise ValueError("future_states_ego_frame must be [T,10]")
        if self.controls is not None:
            self.controls = _as_f32(self.controls)

    @property
    def future_states(self) -> np.ndarray:
        """Compatibility alias for migrated candidate-library utilities."""
        return self.future_states_ego_frame


@dataclass
class ResponseObservation:
    scenario_id: str
    root_hash: str
    candidate_id: str
    agent_id: int
    variant_id: str
    branch_probs: np.ndarray
    branch_hard: int
    trajectory: np.ndarray
    trajectory_valid: np.ndarray
    burden: float
    hp_label: float
    safety_margin: float
    near_collision: bool
    priority_score: float
    priority_confidence: float
    interaction_features: dict[str, Any] = field(default_factory=dict)
    branch_valid: bool = True
    trajectory_valid_any: bool = True
    burden_valid: bool = True
    margin_valid: bool = True
    hp_valid: bool = True
    priority_valid: bool = True
    rollout_valid: bool = True
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.branch_probs = normalize_branch_probs(self.branch_probs, self.branch_valid)
        if self.branch_hard < 0 and self.branch_valid:
            self.branch_hard = int(self.branch_probs.argmax())
        if self.branch_hard >= len(BRANCHES):
            raise ValueError(f"branch_hard out of range: {self.branch_hard}")
        self.trajectory = _as_f32(self.trajectory)
        if self.trajectory.ndim != 2 or self.trajectory.shape[-1] != TRAJ_TARGET_DIM:
            raise ValueError("trajectory must be [T,5] in ego-centered t0 frame")
        self.trajectory_valid = np.asarray(self.trajectory_valid, dtype=bool)
        if self.trajectory_valid.shape != (self.trajectory.shape[0],):
            raise ValueError("trajectory_valid must be [T]")
        self.burden = float(self.burden)
        self.hp_label = float(self.hp_label)
        self.safety_margin = float(self.safety_margin)
        self.priority_score = float(np.clip(self.priority_score, 0.0, 1.0))
        self.priority_confidence = float(np.clip(self.priority_confidence, 0.0, 1.0))
        self.trajectory_valid_any = bool(self.trajectory_valid_any and self.trajectory_valid.any())


@dataclass
class CoercionWitnessLabel:
    scenario_id: str
    root_hash: str
    candidate_id: str
    agent_id: int
    soft_label: float
    confidence: float
    s_c: float
    s_not_c: float
    b_c: float
    d_c: float
    priority_score: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class BoundaryPair:
    candidate_id_a: str
    candidate_id_b: str
    agent_id: int
    response_distance: float
    boundary_label: float = 0.0
    valid: bool = True


@dataclass
class SameRootGroup:
    scenario_id: str
    root_hash: str
    root_scene: RootScene
    candidates: list[EgoCandidate]
    relevant_agent_ids: list[int]
    rollout_variants: list[str]
    observations: dict[tuple[str, int, str], ResponseObservation] = field(default_factory=dict)
    cw_labels: dict[tuple[str, int], CoercionWitnessLabel] = field(default_factory=dict)
    boundary_pairs: list[BoundaryPair] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.root_hash:
            self.root_hash = root_scene_hash(self.root_scene)
        for c in self.candidates:
            if any(key[0] == c.candidate_id for key in self.observations) and self.root_hash == c.candidate_id:
                raise ValueError("root_hash must not be candidate-dependent")


def root_scene_hash(root_scene: RootScene) -> str:
    """Hash only observed root-scene information, never candidates or labels."""
    payload = {
        "scene_id": getattr(root_scene, "scene_id", ""),
        "source": getattr(root_scene, "source", ""),
        "current_time_index": int(getattr(root_scene, "current_time_index", 0)),
        "dt": float(getattr(root_scene, "dt", DEFAULT_DT)),
        "ego_track_index": int(getattr(root_scene, "ego_track_index", 0)),
    }
    try:
        states = getattr(root_scene, "agent_tracks").states
        # Only history/current slice is included by construction in RootScene.
        payload["agent_shape"] = list(states.shape)
        payload["agent_checksum"] = hashlib.sha1(np.asarray(states, dtype=np.float32).tobytes()).hexdigest()
    except Exception:
        pass
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def asdict_json_safe(obj: Any) -> dict[str, Any]:
    def conv(v: Any) -> Any:
        if dataclasses.is_dataclass(v):
            return {k: conv(val) for k, val in dataclasses.asdict(v).items()}
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, dict):
            return {str(k): conv(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        return v
    return conv(obj)
