"""Shared evaluation/calibration runtime helpers for materialized MFRP NPZ shards."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from mfrp.models import MFRPModel
from mfrp.planning.estimators import per_agent_estimates, scene_mechanism_risk, uncertainty_proxy, boundary_sensitivity

MODEL_INPUT_KEYS = {
    "candidate_features", "scene_features", "priority_score_preexec", "priority_confidence_preexec",
    "agent_candidate_valid", "candidate_valid", "support_probe_features", "support_probe_mask",
}
LABEL_KEYS = {
    "query_probe_mask", "branch_probs", "branch_hard", "trajectory", "trajectory_mask", "burden",
    "hp_label", "safety_margin", "variant_valid", "cw_soft_label", "cw_confidence",
    "edge_index", "edge_valid", "response_distance"
}
LEGACY_LABEL_SIDE_KEYS = {"priority_score", "priority_confidence"}


def find_npz_shards(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_dir():
        return sorted(root.rglob("*.npz"))
    if root.suffix == ".npz":
        return [root]
    return []


def torch_batch_from_npz(path: Path, device: str = "cpu", *, include_labels: bool = True) -> dict:
    arr = np.load(path, allow_pickle=True)
    allowed = MODEL_INPUT_KEYS | (LABEL_KEYS if include_labels else set())
    batch: dict[str, torch.Tensor] = {}
    for k in arr.files:
        if k not in allowed:
            continue
        v = arr[k]
        if v.dtype == np.bool_:
            t = torch.from_numpy(v.astype(np.bool_))
        elif np.issubdtype(v.dtype, np.integer):
            t = torch.from_numpy(v.astype(np.int64))
        else:
            t = torch.from_numpy(v.astype(np.float32))
        batch[k] = t.to(device)
    # Legacy migration: old shards used label-side priority_score. Do not silently use it.
    if "priority_score_preexec" not in batch:
        if "candidate_features" in batch and batch["candidate_features"].shape[-1] >= 6:
            # intervention_coordinate writes priority score/confidence in candidate features at positions 4/5.
            batch["priority_score_preexec"] = batch["candidate_features"][..., 4].clamp(0.0, 1.0)
            batch["priority_confidence_preexec"] = batch["candidate_features"][..., 5].clamp(0.0, 1.0)
        elif any(k in arr.files for k in LEGACY_LABEL_SIDE_KEYS):
            raise ValueError(f"{path} only contains legacy label-side priority_score; rebuild the shard so priority_score_preexec is materialized from root/candidate metadata.")
    if "candidate_features" not in batch:
        raise ValueError(f"{path} lacks candidate_features")
    return batch


def make_model_from_config(mcfg: dict, *, smoke: bool = False) -> MFRPModel:
    hidden_dim = 32 if smoke else int(mcfg.get("hidden_dim", 256))
    mechanism_tokens = 4 if smoke else int(mcfg.get("mechanism_tokens", 16))
    future_steps = 8 if smoke else int(mcfg.get("future_steps", 80))
    trajectory_modes = 2 if smoke else int(mcfg.get("trajectory_modes", 6))
    return MFRPModel(
        candidate_feature_dim=int(mcfg.get("candidate_feature_dim", 20)),
        hidden_dim=hidden_dim,
        mechanism_tokens=mechanism_tokens,
        future_steps=future_steps,
        trajectory_modes=trajectory_modes,
        traj_dim=int(mcfg.get("traj_dim", 5)),
        dropout=float(mcfg.get("dropout", 0.1)),
    )


def load_checkpoint_model(checkpoint: str | Path, device: str = "cpu") -> tuple[MFRPModel, dict]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt.get("config", {})
    mcfg = cfg.get("model", cfg)
    model = make_model_from_config(mcfg).to(device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


def batch_violation_truth(batch: dict) -> torch.Tensor:
    """Return [B,K] candidate-level truth for calibration/evaluation.

    Truth is positive when any valid agent/variant has negative margin, or when a
    confident coercion witness label is positive. This keeps calibration tied to
    materialized labels rather than placeholder constants.
    """
    if "safety_margin" not in batch:
        raise ValueError("safety_margin is required to derive violation truth")
    sm = batch["safety_margin"].float()  # [B,A,K,R]
    vv = batch.get("variant_valid", torch.ones_like(sm, dtype=torch.bool)).bool()
    unsafe = ((sm < 0.0) & vv).any(dim=1).any(dim=-1)  # [B,K]
    if "cw_soft_label" in batch and "cw_confidence" in batch:
        cw = (batch["cw_soft_label"].float() >= 0.5) & (batch["cw_confidence"].float() > 0.1)
        unsafe = unsafe | cw.any(dim=1)
    return unsafe.float()


def predict_risk_and_aux(model: MFRPModel, batch: dict) -> tuple[torch.Tensor, dict]:
    with torch.no_grad():
        out = model(batch, mode="scene_only")["scene_only"]
        per_agent = per_agent_estimates(out, batch.get("priority_score_preexec"))
        agent_mask = batch.get("agent_candidate_valid")
        rho = scene_mechanism_risk(per_agent, agent_mask)
        aux = {"outputs": out, "per_agent": per_agent, "uncertainty": uncertainty_proxy(out)}
        if "edge_index" in batch:
            aux["sensitivity"] = boundary_sensitivity(out, batch["candidate_features"], batch["edge_index"], batch.get("edge_valid"))
        return rho, aux
