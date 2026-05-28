"""Minimal deployment-safe root-scene schema for MFRP.

Detailed WOMD/Waymax migration utilities remain in the legacy `scope` package;
this module deliberately avoids old SCOPE branch/label definitions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
import numpy as np

STATE_DIM = 10

@dataclass
class AgentTrackTensor:
    states: np.ndarray
    object_ids: list[int] = field(default_factory=list)
    object_types: list[str] = field(default_factory=list)
    def __post_init__(self) -> None:
        self.states = np.asarray(self.states, dtype=np.float32)
        if self.states.ndim != 3 or self.states.shape[-1] != STATE_DIM:
            raise ValueError('AgentTrackTensor.states must be [N,T,10]')
        n = self.states.shape[0]
        if not self.object_ids:
            self.object_ids = list(range(n))
        if not self.object_types:
            self.object_types = ['vehicle'] * n
    @property
    def valid(self) -> np.ndarray:
        return self.states[..., 9] > 0.5

@dataclass
class RouteContext:
    route_polylines: list[np.ndarray] = field(default_factory=list)
    primary_route_from_map: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    @property
    def primary_route(self) -> np.ndarray | None:
        if self.primary_route_from_map is not None:
            return np.asarray(self.primary_route_from_map, dtype=np.float32)
        return np.asarray(self.route_polylines[0], dtype=np.float32) if self.route_polylines else None

@dataclass
class RootScene:
    scene_id: str
    split: Literal['train','val','test','stress','tiny_debug']
    source: str = 'womd'
    womd_version: str = ''
    current_time_index: int = 10
    dt: float = 0.1
    history_horizon_s: float = 1.0
    future_horizon_s: float = 8.0
    ego_track_index: int = 0
    agent_tracks: AgentTrackTensor | None = None
    route_context: RouteContext = field(default_factory=RouteContext)
    map_features: Any = None
    traffic_controls: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def root_state(self, track_index: int) -> np.ndarray:
        if self.agent_tracks is None:
            raise ValueError('agent_tracks unavailable')
        return self.agent_tracks.states[track_index, -1]
