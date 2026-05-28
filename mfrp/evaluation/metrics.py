from __future__ import annotations

import numpy as np


def binary_auroc(score: np.ndarray, target: np.ndarray) -> float | None:
    score = np.asarray(score).reshape(-1)
    target = np.asarray(target).astype(bool).reshape(-1)
    mask = np.isfinite(score)
    score, target = score[mask], target[mask]
    pos = score[target]; neg = score[~target]
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    rpos = ranks[: len(pos)].sum()
    return float((rpos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def expected_calibration_error(prob: np.ndarray, target: np.ndarray, bins: int = 10) -> float | None:
    prob = np.asarray(prob).reshape(-1)
    target = np.asarray(target).reshape(-1)
    mask = np.isfinite(prob) & np.isfinite(target)
    if mask.sum() == 0:
        return None
    prob, target = prob[mask], target[mask]
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(1 / bins, 1, bins)):
        m = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if m.any():
            ece += float(m.mean() * abs(prob[m].mean() - target[m].mean()))
    return ece


def prediction_metrics(pred: dict[str, np.ndarray], truth: dict[str, np.ndarray]) -> dict[str, float | None]:
    branch_p = pred["branch_prob"]
    branch_t = truth["branch_probs"]
    mask = truth.get("query_probe_mask", truth.get("variant_valid", np.ones(branch_t.shape[:-1], dtype=bool))).astype(bool)
    ce = -(branch_t * np.log(np.expand_dims(branch_p, 3).clip(1e-8, 1))).sum(-1)
    hard_p = branch_p.argmax(-1)[..., None]
    hard_t = branch_t.argmax(-1)
    return {
        "branch_ce": float(ce[mask].mean()) if mask.any() else None,
        "branch_acc": float((hard_p == hard_t)[mask].mean()) if mask.any() else None,
    }
