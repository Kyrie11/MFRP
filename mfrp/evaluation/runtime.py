from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from mfrp.models import MFRPModel


def find_npz_shards(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file() and root.suffix == ".npz":
        return [root]
    return sorted(root.rglob("*.npz")) if root.exists() else []


def torch_batch_from_npz(path: str | Path, device: str = "cpu", *, include_labels: bool = True) -> dict[str, torch.Tensor]:
    data = np.load(path, allow_pickle=True)
    out = {}
    label_keys = {"branch_probs", "trajectory", "trajectory_mask", "burden", "safety_margin", "variant_valid", "support_probe_features", "support_probe_mask", "query_probe_mask", "cw_soft_label", "cw_confidence"}
    for k in data.files:
        if k in {"root_hashes"}:
            continue
        if not include_labels and k in label_keys:
            continue
        arr = data[k]
        if arr.dtype == object:
            continue
        if arr.dtype == bool:
            out[k] = torch.from_numpy(arr.astype(np.bool_)).to(device)
        elif np.issubdtype(arr.dtype, np.integer):
            out[k] = torch.from_numpy(arr).long().to(device)
        else:
            out[k] = torch.from_numpy(arr.astype(np.float32)).to(device)
    return out


def make_model_from_config(cfg: dict, *, smoke: bool = False) -> MFRPModel:
    cfg = dict(cfg or {})
    if smoke:
        cfg = {**cfg, "hidden_dim": min(int(cfg.get("hidden_dim", 64)), 64), "future_steps": int(cfg.get("future_steps", 8))}
    return MFRPModel(
        candidate_feature_dim=int(cfg.get("candidate_feature_dim", 20)),
        scene_feature_dim=int(cfg.get("scene_feature_dim", 32)),
        probe_feature_dim=int(cfg.get("probe_feature_dim", int(cfg.get("candidate_feature_dim", 20)) + 6 + int(cfg.get("state_dim", 5)) + 4)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        mechanism_tokens=int(cfg.get("mechanism_tokens", 8)),
        branches=int(cfg.get("branches", 6)),
        future_steps=int(cfg.get("future_steps", 80)),
        state_dim=int(cfg.get("state_dim", 5)),
    )


def load_checkpoint(path: str | Path, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("config", {}).get("model", ckpt.get("config", {}))
    model = make_model_from_config(cfg)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()
    return model, ckpt
