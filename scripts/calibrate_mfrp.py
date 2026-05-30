from __future__ import annotations

import argparse
import numpy as np
import torch
from mfrp.evaluation.runtime import find_npz_shards, torch_batch_from_npz, load_checkpoint
from mfrp.planning import mechanism_estimates, fit_split_calibration, select_mechanism_feasible


def _candidate_violation_truth(batch: dict[str, torch.Tensor]) -> np.ndarray:
    unsafe = (batch["safety_margin"].detach().cpu().numpy() < 0) & batch["variant_valid"].detach().cpu().numpy().astype(bool)
    truth = unsafe.any(axis=(1, 3)).astype(float)  # [B,K]
    if "cw_soft_label" in batch and "cw_confidence" in batch:
        cw = ((batch["cw_soft_label"].detach().cpu().numpy() > 0.5) & (batch["cw_confidence"].detach().cpu().numpy() > 0.1)).any(axis=1)
        truth = np.maximum(truth, cw.astype(float))
    return truth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--nu-bar", type=float, default=0.30)
    ap.add_argument("--gamma-bar", type=float, default=0.60)
    ap.add_argument("--selected-action", action="store_true", help="also fit selector-level residual quantile on the action selected by the scene-only selector")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    model, _ = load_checkpoint(args.checkpoint, args.device)
    rhos: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    sel_rhos: list[np.ndarray] = []
    sel_truths: list[np.ndarray] = []
    for shard in find_npz_shards(args.data):
        batch = torch_batch_from_npz(shard, args.device, include_labels=True)
        with torch.no_grad():
            out = model(batch, mode="scene_only")
            est = mechanism_estimates(out, prefix="scene")
            if args.selected_action:
                sel = select_mechanism_feasible(out, batch, alpha=args.alpha, q_beta=0.0, nu_bar=args.nu_bar, gamma_bar=args.gamma_bar, prefix="scene")
        rho = est["rho_mech"].detach().cpu().numpy()
        truth = _candidate_violation_truth(batch)
        rhos.append(rho)
        truths.append(truth)
        if args.selected_action:
            idx = sel["selected_index"].detach().cpu().numpy().astype(int)
            b = np.arange(len(idx))
            sel_rhos.append(rho[b, idx])
            sel_truths.append(truth[b, idx])
    if not rhos:
        raise SystemExit("No validation examples found")
    cal = fit_split_calibration(np.concatenate(rhos).reshape(-1), np.concatenate(truths).reshape(-1), beta=args.beta, alpha=args.alpha)
    if args.selected_action and sel_rhos:
        selected = fit_split_calibration(np.concatenate(sel_rhos).reshape(-1), np.concatenate(sel_truths).reshape(-1), beta=args.beta, alpha=args.alpha)
        cal.selected_q_beta = selected.q_beta
    cal.to_json(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
