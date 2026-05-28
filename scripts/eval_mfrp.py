from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mfrp.evaluation.metrics import aggregate_metric_dicts, binary_auroc, response_prediction_metrics
from mfrp.evaluation.runtime import batch_violation_truth, find_npz_shards, load_checkpoint_model, predict_risk_and_aux, torch_batch_from_npz
from mfrp.planning.calibration import CalibrationArtifact, apply_calibration


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate MFRP prediction, mechanism risk, boundary and calibration metrics.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--calibration", default=None)
    p.add_argument("--data", required=True)
    p.add_argument("--metrics", default="prediction,false_safe,boundary,calibration")
    p.add_argument("--out", default="outputs/eval/mfrp_eval.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    requested = {m.strip() for m in args.metrics.split(",") if m.strip()}
    model, cfg = load_checkpoint_model(args.checkpoint, args.device)
    artifact = CalibrationArtifact.load(args.calibration) if args.calibration else None
    shards = find_npz_shards(args.data)
    if not shards:
        raise SystemExit("No NPZ shards found for evaluation.")

    rows: list[dict] = []
    all_rho: list[np.ndarray] = []
    all_truth: list[np.ndarray] = []
    all_kappa: list[np.ndarray] = []
    all_cw_truth: list[np.ndarray] = []
    for shard in shards:
        batch = torch_batch_from_npz(shard, args.device, include_labels=True)
        rho, aux = predict_risk_and_aux(model, batch)
        out = aux["outputs"]
        row: dict = {"shard": str(shard)}
        if "prediction" in requested:
            row.update(response_prediction_metrics(out, batch))
        truth = batch_violation_truth(batch)
        rho_np = rho.detach().cpu().numpy()
        truth_np = truth.detach().cpu().numpy()
        all_rho.append(rho_np.reshape(-1)); all_truth.append(truth_np.reshape(-1))
        if "cw_soft_label" in batch and "cw_confidence" in batch:
            cw_mask = (batch["cw_confidence"].detach().cpu().numpy().reshape(-1) > 0.1)
            if cw_mask.any():
                all_kappa.append(out["kappa"].detach().cpu().numpy().reshape(-1)[cw_mask])
                all_cw_truth.append((batch["cw_soft_label"].detach().cpu().numpy().reshape(-1)[cw_mask] >= 0.5).astype(float))
        if artifact is not None:
            cal = apply_calibration(rho_np, artifact)
            row["calibrated_risk_mean"] = float(np.mean(cal))
            row["calibrated_risk_violation_rate_at_alpha"] = float(np.mean(truth_np[cal <= args.alpha])) if np.any(cal <= args.alpha) else None
            row["calibrated_feasible_fraction_at_alpha"] = float(np.mean(cal <= args.alpha))
        if "boundary" in requested and "sensitivity" in aux:
            row["boundary_sensitivity_mean"] = float(aux["sensitivity"].detach().cpu().mean())
        rows.append(row)

    summary = aggregate_metric_dicts(rows)
    rho_flat = np.concatenate(all_rho); truth_flat = np.concatenate(all_truth)
    if "false_safe" in requested or "calibration" in requested:
        summary["risk_auroc"] = binary_auroc(truth_flat, rho_flat)
        summary["uncalibrated_risk_mean"] = float(np.mean(rho_flat))
        summary["truth_violation_rate"] = float(np.mean(truth_flat))
        summary["uncalibrated_false_safe_rate_at_alpha"] = float(np.mean(truth_flat[rho_flat <= args.alpha])) if np.any(rho_flat <= args.alpha) else None
        summary["uncalibrated_feasible_fraction_at_alpha"] = float(np.mean(rho_flat <= args.alpha))
    if all_kappa:
        summary["cw_auroc"] = binary_auroc(np.concatenate(all_cw_truth), np.concatenate(all_kappa))
    payload = {"summary": summary, "rows": rows, "config": cfg, "num_shards": len(shards)}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
