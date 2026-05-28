from __future__ import annotations

import argparse, json
from pathlib import Path
import torch, yaml
from mfrp.evaluation.runtime import find_npz_shards, torch_batch_from_npz, make_model_from_config
from mfrp.training import total_mfrp_loss


def _validate(batch: dict, shard: Path, *, allow_debug_no_support: bool = False) -> None:
    missing = [k for k in ("scene_features", "candidate_features", "branch_probs", "trajectory", "burden", "safety_margin", "variant_valid") if k not in batch]
    if missing: raise SystemExit(f"{shard} missing required tensors: {missing}")
    if "priority_score_preexec" not in batch: raise SystemExit(f"{shard} lacks priority_score_preexec; rebuild to avoid priority leakage.")
    if not allow_debug_no_support:
        if "query_probe_mask" not in batch or int(batch["query_probe_mask"].sum()) == 0: raise SystemExit(f"{shard} has no query labels.")
        if "support_probe_mask" not in batch or int(batch["support_probe_mask"].sum()) == 0: raise SystemExit(f"{shard} has no support probes.")


def synthetic_batch(device: str, cfg: dict) -> dict:
    m = cfg.get("model", cfg); B,A,K,R,T,C = 2,2,4,3,int(m.get("future_steps",8)),6
    F,S,D = int(m.get("candidate_feature_dim",20)), int(m.get("scene_feature_dim",32)), int(m.get("state_dim",5))
    return {
        "scene_features": torch.randn(B,A,S,device=device), "candidate_features": torch.randn(B,A,K,F,device=device),
        "priority_score_preexec": torch.rand(B,A,K,device=device), "priority_confidence_preexec": torch.ones(B,A,K,device=device),
        "branch_probs": torch.softmax(torch.randn(B,A,K,R,C,device=device),-1), "trajectory": torch.randn(B,A,K,R,T,D,device=device),
        "trajectory_mask": torch.ones(B,A,K,R,T,dtype=torch.bool,device=device), "burden": torch.rand(B,A,K,R,device=device),
        "safety_margin": torch.randn(B,A,K,R,device=device), "variant_valid": torch.ones(B,A,K,R,dtype=torch.bool,device=device),
        "query_probe_mask": torch.ones(B,A,K,R,dtype=torch.bool,device=device),
        "support_probe_features": torch.randn(B,A,K*R,F+C+D+4,device=device), "support_probe_mask": torch.ones(B,A,K*R,dtype=torch.bool,device=device),
        "agent_candidate_valid": torch.ones(B,A,K,dtype=torch.bool,device=device), "candidate_valid": torch.ones(B,K,dtype=torch.bool,device=device),
        "cw_soft_label": torch.rand(B,A,K,device=device), "cw_confidence": torch.ones(B,A,K,device=device), "nominal_cost": torch.rand(B,K,device=device),
    }


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--data", default="outputs/datasets/mfrp_womd_waymax/train"); p.add_argument("--model-config", default="configs/model/mfrp_full.yaml"); p.add_argument("--train-config", default="configs/train/mfrp_train.yaml"); p.add_argument("--max_steps", type=int, default=0); p.add_argument("--smoke", action="store_true"); p.add_argument("--allow-debug-no-support", action="store_true"); p.add_argument("--out", default="outputs/checkpoints/mfrp_best.pt"); p.add_argument("--device", default="cpu")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.model_config).read_text(encoding="utf-8")); tcfg = yaml.safe_load(Path(args.train_config).read_text(encoding="utf-8")) if Path(args.train_config).exists() else {}
    cfg.setdefault("loss", {}).update(tcfg.get("loss", {}))
    if args.smoke:
        cfg["model"]["future_steps"] = min(int(cfg["model"].get("future_steps",80)), 8)
        cfg["model"]["hidden_dim"] = min(int(cfg["model"].get("hidden_dim",128)), 64)
    model = make_model_from_config(cfg.get("model", {}), smoke=args.smoke).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg.get("optimizer", {}).get("lr", 3e-4)), weight_decay=float(tcfg.get("optimizer", {}).get("weight_decay", 1e-2)))
    step = 0; last = None
    if args.smoke:
        for _ in range(max(1, args.max_steps or 2)):
            batch = synthetic_batch(args.device, cfg); opt.zero_grad(); out = model(batch); losses = total_mfrp_loss(out,batch,cfg); losses["total"].backward(); opt.step(); step += 1; last = float(losses["total"].detach().cpu())
    else:
        shards = find_npz_shards(args.data)
        if not shards: raise SystemExit("No NPZ shards found. Build with a real adapter first.")
        for epoch in range(int(tcfg.get("max_epochs",50))):
            for shard in shards:
                batch = torch_batch_from_npz(shard,args.device,include_labels=True); _validate(batch,shard,allow_debug_no_support=args.allow_debug_no_support)
                if batch["candidate_features"].shape[-1] != int(cfg["model"].get("candidate_feature_dim",20)): raise SystemExit(f"{shard} candidate_feature_dim mismatch")
                opt.zero_grad(); out = model(batch); losses = total_mfrp_loss(out,batch,cfg); losses["total"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); step += 1; last = float(losses["total"].detach().cpu())
                if args.max_steps and step >= args.max_steps: break
            if args.max_steps and step >= args.max_steps: break
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); torch.save({"model": model.state_dict(), "config": cfg, "step": step, "last_loss": last, "debug_smoke": args.smoke}, args.out)
    Path(args.out).with_suffix(".json").write_text(json.dumps({"step": step, "last_loss": last, "debug_smoke": args.smoke}, indent=2), encoding="utf-8")
    print(args.out)

if __name__ == "__main__": main()
