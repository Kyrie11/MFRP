from __future__ import annotations

import random
from collections.abc import Sequence


def split_support_query(
    candidate_ids: Sequence[str],
    *,
    support_fraction: float = 0.5,
    seed: int = 0,
    strata: dict[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    """Candidate-level split. All variants of a query candidate stay out of support."""
    ids = list(dict.fromkeys(candidate_ids))
    if len(ids) < 2:
        raise ValueError("need at least two candidates for support/query split")
    rng = random.Random(seed)
    if strata:
        support, query = [], []
        by_s: dict[str, list[str]] = {}
        for cid in ids:
            by_s.setdefault(strata.get(cid, "default"), []).append(cid)
        for group in by_s.values():
            rng.shuffle(group)
            n = max(1, min(len(group) - 1, round(len(group) * support_fraction))) if len(group) > 1 else 1
            support.extend(group[:n])
            query.extend(group[n:])
        if not query:
            moved = support.pop()
            query.append(moved)
    else:
        rng.shuffle(ids)
        n = max(1, min(len(ids) - 1, round(len(ids) * support_fraction)))
        support, query = ids[:n], ids[n:]
    if set(support) & set(query):
        raise AssertionError("support/query overlap")
    return support, query
