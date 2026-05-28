"""Offline metrics for MFRP response prediction and mechanism risk."""
from __future__ import annotations

from typing import Any
import numpy as np
import torch
import torch.nn.functional as F


def masked_mean_np(values: np.ndarray, mask: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if values.size == 0 or not mask.any():
        return None
    return float(np.mean(values[mask]))


def binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_score = np.asarray(y_score).reshape(-1).astype(float)
    finite = np.isfinite(y_score)
    y_true, y_score = y_true[finite], y_score[finite]
    pos = y_true == 1; neg = y_true == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # Average tied ranks.
    for val in np.unique(y_score):
        idx = np.where(y_score == val)[0]
        if len(idx) > 1:
            ranks[idx] = ranks[idx].mean()
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def response_prediction_metrics(outputs: dict, batch: dict) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    pmask = batch.get("query_probe_mask", batch.get("variant_valid"))
    if pmask is None:
        return metrics
    pmask = pmask.bool()
    if "branch_probs" in batch:
        target = batch["branch_probs"].float()
        pred_log = F.log_softmax(outputs["branch_logits"], dim=-1).unsqueeze(3)
        ce = -(target * pred_log).sum(dim=-1)
        metrics["branch_ce"] = float(ce[pmask].mean().detach().cpu()) if pmask.any() else None
        hard_t = target.argmax(dim=-1)
        hard_p = outputs["branch_logits"].argmax(dim=-1).unsqueeze(3).expand_as(hard_t)
        metrics["branch_acc"] = float((hard_p[pmask] == hard_t[pmask]).float().mean().detach().cpu()) if pmask.any() else None
    if "burden" in batch:
        pred = outputs["burden_loc"].mean(dim=-1).unsqueeze(3)
        err = torch.abs(pred - batch["burden"].float())
        metrics["burden_mae"] = float(err[pmask].mean().detach().cpu()) if pmask.any() else None
    if "safety_margin" in batch:
        pred = outputs["margin_loc"].mean(dim=-1).unsqueeze(3)
        err = torch.abs(pred - batch["safety_margin"].float())
        metrics["margin_mae"] = float(err[pmask].mean().detach().cpu()) if pmask.any() else None
    if "trajectory" in batch and "trajectory_mask" in batch:
        loc = outputs["trajectory"]["loc"].mean(dim=-3).mean(dim=-3).unsqueeze(3)  # [B,A,K,1,T,D]
        tgt = batch["trajectory"].float()
        tmask = batch["trajectory_mask"].bool() & pmask.unsqueeze(-1)
        ade = torch.linalg.norm(loc[..., :2] - tgt[..., :2], dim=-1)
        metrics["traj_ade"] = float(ade[tmask].mean().detach().cpu()) if tmask.any() else None
    return metrics


def aggregate_metric_dicts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = sorted({k for r in rows for k in r})
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if not vals:
            out[k] = None
        elif isinstance(vals[0], (int, float, np.floating)):
            out[k] = float(np.mean(vals))
        else:
            out[k] = vals[-1]
    return out
