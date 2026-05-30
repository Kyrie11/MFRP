from __future__ import annotations

import argparse
import json
import torch
from mfrp.evaluation.runtime import load_checkpoint, torch_batch_from_npz
from mfrp.planning.deploy import scene_only_inference
from mfrp.planning.selector import select_mechanism_feasible
from mfrp.planning.calibration import SplitCalibration


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scenario", required=True, help="Deployment-ready NPZ batch")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--nu-bar", type=float, default=0.30)
    ap.add_argument("--gamma-bar", type=float, default=0.60)
    args = ap.parse_args()
    model, _ = load_checkpoint(args.checkpoint, args.device)
    batch = torch_batch_from_npz(args.scenario, args.device, include_labels=False)
    with torch.no_grad():
        out = scene_only_inference(model, batch)
        q_beta = 0.0
        if args.calibration:
            cal = SplitCalibration.from_json(args.calibration)
            q_beta = float(cal.selected_q_beta if cal.selected_q_beta is not None else cal.q_beta)
        sel = select_mechanism_feasible(out, batch, alpha=args.alpha, q_beta=q_beta, nu_bar=args.nu_bar, gamma_bar=args.gamma_bar, prefix="scene")
    result = {
        "selected_index": sel["selected_index"].detach().cpu().tolist(),
        "fallback_used": sel["fallback_used"].detach().cpu().tolist(),
        "feasible_mask": sel["feasible_mask"].detach().cpu().tolist(),
        "rho_mech_cal": sel["rho_mech_cal"].detach().cpu().tolist(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
