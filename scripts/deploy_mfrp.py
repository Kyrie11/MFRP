from __future__ import annotations

import argparse
from mfrp.evaluation.runtime import load_checkpoint, torch_batch_from_npz
from mfrp.planning.deploy import scene_only_inference


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", required=True); ap.add_argument("--scenario", required=True, help="For this reference CLI, pass a deployment-ready NPZ batch"); ap.add_argument("--device", default="cpu"); ap.add_argument("--calibration", default=None); ap.add_argument("--candidate-config", default=None)
    args = ap.parse_args(); model,_ = load_checkpoint(args.checkpoint,args.device); batch = torch_batch_from_npz(args.scenario,args.device,include_labels=False); out = scene_only_inference(model,batch)
    print({k: tuple(v.shape) for k,v in out.items() if hasattr(v,'shape') and k.endswith(('branch_prob','kappa'))})

if __name__ == "__main__": main()
