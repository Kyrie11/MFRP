"""Candidate-level support/query split for MFRP."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SupportQuerySplit:
    support_candidate_ids: list[str]
    query_candidate_ids: list[str]
    support_candidate_mask: np.ndarray
    query_candidate_mask: np.ndarray
    support_probe_mask: np.ndarray | None = None
    query_probe_mask: np.ndarray | None = None


def _family(c: Any) -> str:
    return str(getattr(c, "family", getattr(c, "metadata", {}).get("family", "unknown")))


def split_support_query(candidates: list[Any], labels: Any | None = None, max_support_slots: int = 16, prob_empty_support: float = 0.4, rng: np.random.Generator | None = None, rollout_variants: list[str] | None = None) -> SupportQuerySplit:
    """Split by candidate id; all variants of a query candidate are withheld."""
    if rng is None:
        rng = np.random.default_rng()
    ids = [str(getattr(c, "candidate_id", i)) for i, c in enumerate(candidates)]
    n = len(ids)
    if n == 0:
        return SupportQuerySplit([], [], np.zeros(0, bool), np.zeros(0, bool))
    query_mask = np.zeros(n, dtype=bool)
    support_mask = np.zeros(n, dtype=bool)
    empty = rng.random() < prob_empty_support
    if empty or n == 1:
        query_mask[:] = True
    else:
        by_family: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(candidates):
            by_family[_family(c)].append(i)
        support_idx: list[int] = []
        for fam in sorted(by_family):
            support_idx.append(int(rng.choice(by_family[fam])))
        remain = [i for i in range(n) if i not in support_idx]
        rng.shuffle(remain)
        support_idx.extend(remain[: max(0, max_support_slots - len(support_idx))])
        support_idx = support_idx[:max_support_slots]
        support_mask[support_idx] = True
        query_mask = ~support_mask
        if not query_mask.any():
            # Always leave at least one candidate for query.
            j = support_idx[-1]
            support_mask[j] = False
            query_mask[j] = True
    support_ids = [ids[i] for i in range(n) if support_mask[i]]
    query_ids = [ids[i] for i in range(n) if query_mask[i]]
    R = len(rollout_variants or [])
    sp = qp = None
    if R > 0:
        sp = np.repeat(support_mask[:, None], R, axis=1)
        qp = np.repeat(query_mask[:, None], R, axis=1)
    assert not (set(support_ids) & set(query_ids))
    return SupportQuerySplit(support_ids, query_ids, support_mask, query_mask, sp, qp)
