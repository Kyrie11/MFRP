from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mfrp.evaluation.runtime import batch_violation_truth, find_npz_shards, load_checkpoint_model, predict_risk_and_aux, torch_batch_from_npz
from mfrp.planning.calibration import fit_split_calibration


def main() -> None:
    p = argparse.ArgumentParser(description="Fit split calibration artifact for scene-only MFRP mechanism risk.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True, help="Validation split directory or NPZ shard")
    p.add_argument("--out", required=True)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    model, _ = load_checkpoint_model(args.checkpoint, args.device)
    shards = find_npz_shards(args.data)
    if not shards:
        raise SystemExit("No validation NPZ shards found; refusing to write an empty calibration artifact.")
    rho_all: list[np.ndarray] = []
    truth_all: list[np.ndarray] = []
    for shard in shards:
        batch = torch_batch_from_npz(shard, args.device, include_labels=True)
        rho, _ = predict_risk_and_aux(model, batch)
        truth = batch_violation_truth(batch)
        rho_all.append(rho.detach().cpu().numpy().reshape(-1))
        truth_all.append(truth.detach().cpu().numpy().reshape(-1))
    rho_np = np.concatenate(rho_all) if rho_all else np.asarray([])
    truth_np = np.concatenate(truth_all) if truth_all else np.asarray([])
    art = fit_split_calibration(rho_np, truth_np, args.beta)
    art.save(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
