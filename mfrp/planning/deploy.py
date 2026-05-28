"""Deployment entry points. Deployment is scene-only by construction."""
from __future__ import annotations

import torch


LABEL_SIDE_KEYS = {
    "support_probe_features", "support_probe_mask", "query_probe_mask",
    "branch_probs", "branch_hard", "trajectory", "trajectory_mask",
    "burden", "hp_label", "safety_margin", "variant_valid",
    "cw_soft_label", "cw_confidence", "cw_rank_pairs", "cw_rank_valid",
    # Legacy priority values are label-side unless explicitly rebuilt as *_preexec.
    "priority_score", "priority_confidence",
    "diagnostics", "groups",
}


def scene_only_inference(model, batch: dict) -> dict:
    clean = {k: v for k, v in batch.items() if k not in LABEL_SIDE_KEYS}
    with torch.no_grad():
        return model(clean, mode="scene_only")["scene_only"]
