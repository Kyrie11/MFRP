from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from mfrp.evaluation.runtime import find_npz_shards, torch_batch_from_npz, make_model_from_config
from mfrp.training import total_mfrp_loss


def _validate_paper_batch(batch: dict, shard: Path, *, allow_debug_no_support: bool = False) -> None:
    missing = [k for k in ("candidate_features", "branch_probs", "trajectory", "burden", "safety_margin", "variant_valid") if k not in batch]
    if missing:
        raise SystemExit(f"{shard} is missing required training tensors: {missing}")
    if "priority_score_preexec" not in batch:
        raise SystemExit(f"{shard} lacks priority_score_preexec. Rebuild dataset to avoid label-side priority leakage.")
    q = batch.get("query_probe_mask")
    s = batch.get("support_probe_mask")
    if not allow_debug_no_support:
        if q is None or int(q.sum().item()) == 0:
            raise SystemExit(f"{shard} has no query_probe_mask labels. Materialize support/query candidate splits first.")
        if s is None or int(s.sum().item()) == 0:
            raise SystemExit(f"{shard} has no support probes. Use --allow-debug-no-support only for smoke/debug runs.")


def main() -> None:
    p = argparse.ArgumentParser(description="Train MFRP from materialized same-root NPZ shards.")
    p.add_argument("--data", default="outputs/datasets/mfrp_womd_waymax")
    p.add_argument("--model-config", default="configs/model/mfrp_full.yaml")
    p.add_argument("--train-config", default="configs/train/mfrp_train.yaml")
    p.add_argument("--max_steps", type=int, default=0, help="If >0 with --smoke, run synthetic smoke training for this many steps.")
    p.add_argument("--smoke", action="store_true", help="Use synthetic data; never use for paper results.")
    p.add_argument("--allow-debug-no-support", action="store_true", help="Bypass support/query enforcement for debug shards only.")
    p.add_argument("--out", default="outputs/checkpoints/mfrp_best.pt")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.model_config).read_text())
    tcfg = yaml.safe_load(Path(args.train_config).read_text()) if Path(args.train_config).exists() else {}
    cfg.setdefault("loss", {}).update(tcfg.get("loss", {}))
    mcfg = cfg.get("model", {})
    model = make_model_from_config(mcfg, smoke=args.smoke).to(args.device)
    if args.smoke:
        cfg.setdefault("model", {})["hidden_dim"] = 32
        cfg.setdefault("model", {})["mechanism_tokens"] = 4
        cfg.setdefault("model", {})["future_steps"] = 8
        cfg.setdefault("model", {})["trajectory_modes"] = 2
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg.get("optimizer", {}).get("lr", 3e-4)),
        weight_decay=float(tcfg.get("optimizer", {}).get("weight_decay", 0.01)),
    )

    if args.smoke:
        steps = max(1, int(args.max_steps or 2))
        B, A, K, R, T, C = 2, 2, 3, 2, 8, 6
        F = int(mcfg.get("candidate_feature_dim", 20))
        batch = {
            "candidate_features": torch.randn(B, A, K, F, device=args.device),
            "priority_score_preexec": torch.rand(B, A, K, device=args.device),
            "priority_confidence_preexec": torch.ones(B, A, K, device=args.device),
            "branch_probs": torch.softmax(torch.randn(B, A, K, R, C, device=args.device), -1),
            "trajectory": torch.randn(B, A, K, R, T, 5, device=args.device),
            "trajectory_mask": torch.ones(B, A, K, R, T, dtype=torch.bool, device=args.device),
            "burden": torch.rand(B, A, K, R, device=args.device),
            "safety_margin": torch.randn(B, A, K, R, device=args.device),
            "variant_valid": torch.ones(B, A, K, R, dtype=torch.bool, device=args.device),
            "query_probe_mask": torch.ones(B, A, K, R, dtype=torch.bool, device=args.device),
            "support_probe_features": torch.randn(B, A, K * R, F + C + 5 + 3, device=args.device),
            "support_probe_mask": torch.ones(B, A, K * R, dtype=torch.bool, device=args.device),
            "agent_candidate_valid": torch.ones(B, A, K, dtype=torch.bool, device=args.device),
            "cw_soft_label": torch.rand(B, A, K, device=args.device),
            "cw_confidence": torch.ones(B, A, K, device=args.device),
        }
        last_loss = None
        for _ in range(steps):
            opt.zero_grad(); out = model(batch); losses = total_mfrp_loss(out, batch, cfg); losses["total"].backward(); opt.step()
            last_loss = float(losses["total"].detach().cpu())
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "config": cfg, "step": steps, "last_loss": last_loss, "debug_smoke": True}, args.out)
        print(args.out)
        return

    shards = find_npz_shards(args.data)
    if not shards:
        raise SystemExit("No materialized NPZ shards found. Run build_same_root_dataset.py with a real adapter, then diagnose_dataset.py.")
    max_steps = int(tcfg.get("max_steps", 200000)); max_epochs = int(tcfg.get("max_epochs", 50))
    step = 0; last_loss = None
    model.train()
    for epoch in range(max_epochs):
        for shard in shards:
            batch = torch_batch_from_npz(shard, args.device, include_labels=True)
            _validate_paper_batch(batch, shard, allow_debug_no_support=args.allow_debug_no_support)
            if batch["candidate_features"].shape[-1] != int(mcfg.get("candidate_feature_dim", 20)):
                raise SystemExit(f"{shard} candidate_feature_dim={batch['candidate_features'].shape[-1]} but model expects {mcfg.get('candidate_feature_dim',20)}")
            opt.zero_grad(); out = model(batch); losses = total_mfrp_loss(out, batch, cfg); losses["total"].backward(); opt.step()
            last_loss = float(losses["total"].detach().cpu()); step += 1
            if step >= max_steps:
                break
        if step >= max_steps:
            break
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg, "step": step, "last_loss": last_loss, "shards": [str(s) for s in shards]}, args.out)
    Path(args.out).with_suffix(".json").write_text(json.dumps({"step": step, "last_loss": last_loss, "num_shards": len(shards)}, indent=2))
    print(args.out)


if __name__ == "__main__":
    main()
