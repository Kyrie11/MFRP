"""Utilities for materializing same-root MFRP shards.

The core repository cannot know every local WOMD/Waymax storage layout, so the
paper path is an adapter contract: user code loads WOMD scenarios, runs Waymax
same-root rollouts, returns SameRootGroup objects, and this module performs
collation, split validation, metadata stamping and NPZ writing.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from mfrp.data.schema import SameRootGroup
from mfrp.data.tensors import collate_same_root_groups


def load_adapter(spec: str) -> Callable[..., Iterable[SameRootGroup] | dict[str, Any] | np.ndarray]:
    if ":" not in spec:
        raise ValueError("adapter must be formatted as module.submodule:function")
    mod_name, fn_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    if not callable(fn):
        raise TypeError(f"adapter {spec} is not callable")
    return fn


def _npz_payload(batch: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for k, v in batch.items():
        if k == "groups":
            continue
        if isinstance(v, np.ndarray):
            payload[k] = v
    return payload


def validate_groups(groups: list[SameRootGroup], *, debug_allow_no_support: bool = False) -> None:
    if not groups:
        raise ValueError("adapter returned no SameRootGroup objects")
    for g in groups:
        if not g.candidates:
            raise ValueError(f"{g.scenario_id}/{g.root_hash}: no ego candidates")
        if not g.relevant_agent_ids:
            raise ValueError(f"{g.scenario_id}/{g.root_hash}: no relevant agents")
        if not g.rollout_variants:
            raise ValueError(f"{g.scenario_id}/{g.root_hash}: no rollout variants")
        if not g.observations:
            raise ValueError(f"{g.scenario_id}/{g.root_hash}: no response observations")
        support = set(g.metadata.get("support_candidate_ids", []))
        query = set(g.metadata.get("query_candidate_ids", []))
        if not debug_allow_no_support and (not support or not query):
            raise ValueError(f"{g.scenario_id}/{g.root_hash}: missing support/query candidate split")
        if support & query:
            raise ValueError(f"{g.scenario_id}/{g.root_hash}: support/query split overlaps")
        if g.metadata.get("uses_log_playback_for_response", False):
            raise ValueError(f"{g.scenario_id}/{g.root_hash}: log playback cannot supervise MFRP response mechanisms")


def write_group_shard(groups: list[SameRootGroup], path: str | Path, *, future_steps: int = 80, candidate_feature_dim: int = 20, scene_feature_dim: int | None = None, debug_allow_no_support: bool = False) -> Path:
    validate_groups(groups, debug_allow_no_support=debug_allow_no_support)
    batch = collate_same_root_groups(
        groups,
        future_steps=future_steps,
        candidate_feature_dim=candidate_feature_dim,
        scene_feature_dim=scene_feature_dim,
        require_support_query_split=not debug_allow_no_support,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = _npz_payload(batch)
    payload["materializer_version"] = np.asarray([1], dtype=np.int64)
    payload["debug_only"] = np.asarray([bool(debug_allow_no_support)], dtype=bool)
    payload["root_hashes"] = np.asarray([g.root_hash for g in groups], dtype=object)
    payload["scenario_ids"] = np.asarray([g.scenario_id for g in groups], dtype=object)
    np.savez_compressed(out, **payload)
    meta = {
        "num_groups": len(groups),
        "scenario_ids": [g.scenario_id for g in groups],
        "root_hashes": [g.root_hash for g in groups],
        "debug_only": bool(debug_allow_no_support),
        "future_steps": int(future_steps),
        "candidate_feature_dim": int(candidate_feature_dim),
        "scene_feature_dim": int(scene_feature_dim or candidate_feature_dim),
        "support_query_required": not debug_allow_no_support,
    }
    out.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def materialize_with_adapter(adapter: Callable[..., Any], *, split: str, out_dir: str | Path, womd_pattern: str, config: dict[str, Any], max_scenarios: int | None = None, shard_size: int = 8, num_workers: int = 1, debug_allow_no_support: bool = False) -> list[Path]:
    result = adapter(womd_pattern=womd_pattern, split=split, config=config, max_scenarios=max_scenarios, num_workers=num_workers)
    if isinstance(result, dict) and "groups" in result:
        result = result["groups"]
    groups = list(result)
    validate_groups(groups, debug_allow_no_support=debug_allow_no_support)
    out_root = Path(out_dir) / split
    out_root.mkdir(parents=True, exist_ok=True)
    data_cfg = config.get("dataset", config.get("data", {}))
    future_steps = int(data_cfg.get("future_steps", config.get("future_steps", 80)))
    candidate_feature_dim = int(data_cfg.get("candidate_feature_dim", config.get("candidate_feature_dim", 20)))
    scene_feature_dim = data_cfg.get("scene_feature_dim", config.get("scene_feature_dim"))
    written: list[Path] = []
    for i in range(0, len(groups), max(1, int(shard_size))):
        shard_groups = groups[i:i + max(1, int(shard_size))]
        written.append(write_group_shard(
            shard_groups,
            out_root / f"mfrp_{split}_{len(written):05d}.npz",
            future_steps=future_steps,
            candidate_feature_dim=candidate_feature_dim,
            scene_feature_dim=scene_feature_dim,
            debug_allow_no_support=debug_allow_no_support,
        ))
    return written
