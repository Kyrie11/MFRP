from __future__ import annotations

from dataclasses import dataclass
import random
from collections.abc import Sequence
import numpy as np


@dataclass
class SupportQuerySplit:
    support_candidate_ids: list[str]
    query_candidate_ids: list[str]
    support_probe_mask: np.ndarray
    query_probe_mask: np.ndarray

    def __iter__(self):
        yield self.support_candidate_ids
        yield self.query_candidate_ids


def _candidate_id(c) -> str:
    return str(getattr(c, "candidate_id", c))


def _candidate_family(c) -> str:
    return str(getattr(c, "family", "default") or "default")


def split_support_query(
    candidate_ids: Sequence,
    *,
    support_fraction: float = 0.5,
    seed: int = 0,
    strata: dict[str, str] | None = None,
    max_support_slots: int | None = None,
    prob_empty_support: float = 0.0,
    rng=None,
    rollout_variants: Sequence[str] | None = None,
):
    """Candidate-level split. All variants of a query candidate stay out of support.

    Returns a tuple for the current API, or a SupportQuerySplit with masks when the
    legacy mask-producing arguments are requested.
    """
    objs = list(candidate_ids)
    ids = list(dict.fromkeys(_candidate_id(c) for c in objs))
    if len(ids) < 2:
        raise ValueError("need at least two candidates for support/query split")
    use_np = rng is not None
    if rng is None:
        rr = random.Random(seed)
        shuffle = rr.shuffle
        rand = rr.random
    else:
        shuffle = lambda x: rng.shuffle(x)
        rand = rng.random
    if prob_empty_support > 0 and rand() < prob_empty_support:
        support, query = [], ids[:]
    else:
        if strata is None and any(hasattr(c, "family") for c in objs):
            strata = {_candidate_id(c): _candidate_family(c) for c in objs}
        if strata:
            support, query = [], []
            by_s: dict[str, list[str]] = {}
            for cid in ids:
                by_s.setdefault(strata.get(cid, "default"), []).append(cid)
            for group in by_s.values():
                shuffle(group)
                n = max(1, min(len(group) - 1, round(len(group) * support_fraction))) if len(group) > 1 else 1
                support.extend(group[:n]); query.extend(group[n:])
            if not query and support:
                query.append(support.pop())
        else:
            ids_shuf = ids[:]
            shuffle(ids_shuf)
            n = max(1, min(len(ids_shuf) - 1, round(len(ids_shuf) * support_fraction)))
            support, query = ids_shuf[:n], ids_shuf[n:]
        if max_support_slots is not None and len(support) > max_support_slots:
            moved = support[max_support_slots:]
            support = support[:max_support_slots]
            query = list(dict.fromkeys(query + moved))
    if set(support) & set(query):
        raise AssertionError("support/query overlap")
    if rollout_variants is None and max_support_slots is None and rng is None:
        return support, query
    R = len(rollout_variants or ["r0"])
    K = len(ids)
    support_mask = np.zeros((K, R), dtype=bool)
    query_mask = np.zeros((K, R), dtype=bool)
    for k, cid in enumerate(ids):
        if cid in support:
            support_mask[k, :] = True
        if cid in query:
            query_mask[k, :] = True
    return SupportQuerySplit(support, query, support_mask, query_mask)
