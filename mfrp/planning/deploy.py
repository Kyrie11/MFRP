from __future__ import annotations

LABEL_SIDE_KEYS = {
    "branch_probs", "trajectory", "trajectory_mask", "burden", "safety_margin", "variant_valid",
    "support_probe_features", "support_probe_mask", "query_probe_mask", "cw_soft_label", "cw_confidence",
    "priority_score", "priority_confidence", "groups", "observations", "rollout_diagnostics",
}


def sanitize_deployment_batch(batch: dict) -> dict:
    return {k: v for k, v in batch.items() if k not in LABEL_SIDE_KEYS}


def scene_only_inference(model, batch: dict):
    clean = sanitize_deployment_batch(batch)
    return model(clean, mode="scene_only")
