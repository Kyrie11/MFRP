from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
from mfrp.evaluation.runtime import find_npz_shards


def robust_scale(x):
    x=np.asarray(x); x=x[np.isfinite(x)]
    if x.size==0: return 1.0
    q=np.quantile(x,[.25,.75]); return float(max(1e-3,(q[1]-q[0])/1.349))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True); ap.add_argument("--out",required=True); args=ap.parse_args(); vals={"burden":[],"safety_margin":[]}
    for f in find_npz_shards(args.dataset):
        d=np.load(f,allow_pickle=True); mask=d["variant_valid"].astype(bool) if "variant_valid" in d else None
        for k in vals:
            if k in d: vals[k].append(d[k][mask] if mask is not None and mask.shape==d[k].shape else d[k].reshape(-1))
    out={k: robust_scale(np.concatenate(v)) if v else 1.0 for k,v in vals.items()}; Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(out,indent=2),encoding="utf-8"); print(args.out)
if __name__=="__main__": main()
