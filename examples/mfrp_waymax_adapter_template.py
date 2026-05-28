"""Template only: copy into your local project and connect to WOMD/Waymax.

Run as:
  python -m scripts.build_same_root_dataset ... --adapter my_project.mfrp_waymax_adapter:build_groups

Do not import this template for paper data: it intentionally raises NotImplementedError.
"""
from __future__ import annotations

from collections.abc import Iterable
from mfrp.data.schema import SameRootGroup


def build_groups(*, womd_pattern: str, split: str, config: dict, max_scenarios: int | None, num_workers: int) -> Iterable[SameRootGroup]:
    raise NotImplementedError(
        "Implement locally: load WOMD, reset Waymax at the same t0 for every candidate, run multiple reactive variants, "
        "extract ResponseObservation labels, call split_support_query, and yield SameRootGroup objects."
    )
