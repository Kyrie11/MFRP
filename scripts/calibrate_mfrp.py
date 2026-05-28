from __future__ import annotations

import argparse
import numpy as np
import torch
from mfrp.evaluation.runtime import find_npz_shards, torch_batch_from_npz, load_checkpoint
from mfrp.planning import mechanism_estimates, fit_split_calibration


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--beta", type=float, default=0.1); ap.add_argument("--alpha", type=float, default=0.05); ap.add_argument("--device", default="cpu")
    args = ap.parse_args(); model,_ = load_checkpoint(args.checkpoint,args.device)
    rhos=[]; truths=[]
    for shard in find_npz_shards(args.data):
        batch = torch_batch_from_npz(shard,args.device,include_labels=True)
        with torch.no_grad(): out = model(batch,mode="scene_only"); est = mechanism_estimates(out,prefix="scene")
        rho = est["rho_mech"].detach().cpu().numpy()
        unsafe = (batch["safety_margin"].detach().cpu().numpy() < 0) & batch["variant_valid"].detach().cpu().numpy().astype(bool)
        truth = unsafe.any(axis=(1,3)).astype(float)  # [B,K]
        if "cw_soft_label" in batch and "cw_confidence" in batch:
            cw = ((batch["cw_soft_label"].detach().cpu().numpy() > .5) & (batch["cw_confidence"].detach().cpu().numpy() > .1)).any(axis=1)
            truth = np.maximum(truth, cw.astype(float))
        rhos.append(rho); truths.append(truth)
    if not rhos: raise SystemExit("No validation examples found")
    cal = fit_split_calibration(np.concatenate(rhos).reshape(-1), np.concatenate(truths).reshape(-1), beta=args.beta, alpha=args.alpha)
    cal.to_json(args.out); print(args.out)

if __name__ == "__main__": main()
