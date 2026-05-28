from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import torch
from mfrp.evaluation.runtime import find_npz_shards, torch_batch_from_npz, load_checkpoint
from mfrp.planning import mechanism_estimates, SplitCalibration
from mfrp.evaluation.metrics import binary_auroc, expected_calibration_error


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", required=True); ap.add_argument("--calibration", default=None); ap.add_argument("--data", required=True); ap.add_argument("--metrics", default="prediction,false_safe,boundary,calibration"); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cpu")
    args = ap.parse_args(); model,_ = load_checkpoint(args.checkpoint,args.device); cal = SplitCalibration.from_json(args.calibration) if args.calibration else None
    rows=[]
    for shard in find_npz_shards(args.data):
        batch = torch_batch_from_npz(shard,args.device,include_labels=True)
        with torch.no_grad(): out = model(batch, mode="scene_only"); est = mechanism_estimates(out,prefix="scene")
        branch_prob = out["scene_branch_prob"].cpu().numpy(); branch_true = batch["branch_probs"].cpu().numpy(); mask = batch["query_probe_mask"].cpu().numpy().astype(bool) if "query_probe_mask" in batch else batch["variant_valid"].cpu().numpy().astype(bool)
        ce = -(branch_true * np.log(np.expand_dims(branch_prob,3).clip(1e-8,1))).sum(-1)
        hard_acc = (branch_prob.argmax(-1)[...,None] == branch_true.argmax(-1))[mask].mean() if mask.any() else np.nan
        rho = est["rho_mech"].cpu().numpy(); rho_cal = cal.apply(rho) if cal else rho
        truth = ((batch["safety_margin"].cpu().numpy() < 0) & batch["variant_valid"].cpu().numpy().astype(bool)).any(axis=(1,3)).astype(float)
        rows.append({"file": str(shard), "branch_ce": float(ce[mask].mean()) if mask.any() else None, "branch_acc": float(hard_acc) if np.isfinite(hard_acc) else None, "rho_mean": float(rho.mean()), "rho_cal_mean": float(rho_cal.mean()), "truth_violation_rate": float(truth.mean()), "risk_auroc": binary_auroc(rho.reshape(-1), truth.reshape(-1)), "risk_ece": expected_calibration_error(rho_cal.reshape(-1), truth.reshape(-1))})
    if not rows: raise SystemExit("No test shards found")
    summary={k: (float(np.nanmean([r[k] for r in rows if r.get(k) is not None])) if any(r.get(k) is not None for r in rows) and isinstance(next((r[k] for r in rows if r.get(k) is not None),None),(int,float)) else None) for k in rows[0] if k != "file"}
    out={"summary": summary, "files": rows}; Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(out,indent=2),encoding="utf-8"); print(args.out)

if __name__ == "__main__": main()
