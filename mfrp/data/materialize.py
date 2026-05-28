from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .schema import SameRootGroup
from .tensors import collate_same_root_groups, write_npz_shard


def load_adapter(spec: str) -> Callable[..., Iterable[SameRootGroup]]:
    if not spec or ":" not in spec:
        raise ValueError("--adapter must be a Python callable in module:function format, e.g. my_project.mfrp_waymax_adapter:build_groups")
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, func_name)
    if not callable(fn):
        raise TypeError(f"adapter {spec} is not callable")
    return fn


def _chunks(it: Iterable[SameRootGroup], n: int):
    buf: list[SameRootGroup] = []
    for item in it:
        buf.append(item)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def materialize_with_adapter(
    adapter: Callable[..., Iterable[SameRootGroup]],
    *,
    split: str,
    out_dir: str | Path,
    womd_pattern: str,
    config: dict[str, Any],
    max_scenarios: int | None = None,
    shard_size: int = 8,
    num_workers: int = 1,
    debug_allow_no_support: bool = False,
) -> list[Path]:
    if not womd_pattern:
        raise ValueError("womd_pattern is required; refusing to create placeholder data")
    out_root = Path(out_dir) / split
    out_root.mkdir(parents=True, exist_ok=True)
    groups = adapter(womd_pattern=womd_pattern, split=split, config=config, max_scenarios=max_scenarios, num_workers=num_workers)
    written: list[Path] = []
    cfd = int(config.get("tensor", {}).get("candidate_feature_dim", config.get("model", {}).get("candidate_feature_dim", 20)))
    sfd = int(config.get("tensor", {}).get("scene_feature_dim", config.get("model", {}).get("scene_feature_dim", 32)))
    T = int(config.get("tensor", {}).get("future_steps", config.get("model", {}).get("future_steps", 80)))
    D = int(config.get("tensor", {}).get("state_dim", 5))
    for idx, chunk in enumerate(_chunks(groups, max(1, shard_size))):
        batch = collate_same_root_groups(
            chunk,
            candidate_feature_dim=cfd,
            scene_feature_dim=sfd,
            future_steps=T,
            state_dim=D,
            require_support_query_split=not debug_allow_no_support,
            allow_debug=debug_allow_no_support,
        )
        path = out_root / f"{split}_{idx:05d}.npz"
        write_npz_shard(path, batch, {"split": split, "num_groups": len(chunk), "debug_only": debug_allow_no_support})
        written.append(path)
    if not written:
        raise RuntimeError("adapter returned no SameRootGroup objects")
    return written
