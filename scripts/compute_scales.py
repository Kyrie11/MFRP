from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def robust_scale(x, default=1.0):
    x=np.asarray(x, dtype=float); x=x[np.isfinite(x)]
    if x.size == 0: return float(default)
    q25,q75=np.quantile(x,[0.25,0.75]); iqr=q75-q25
    mad=np.median(np.abs(x-np.median(x)))
    s=max(float(iqr/1.349) if iqr>0 else 0.0, float(1.4826*mad) if mad>0 else 0.0, 1e-3)
    return s


def main():
    p=argparse.ArgumentParser(description='Compute train-split robust scales for MFRP labels from materialized NPZ shards.')
    p.add_argument('--dataset', required=True); p.add_argument('--out', required=True)
    args=p.parse_args(); root=Path(args.dataset); out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    burdens=[]; margins=[]
    for f in root.rglob('*.npz') if root.is_dir() else [root]:
        try: data=np.load(f, allow_pickle=True)
        except Exception: continue
        mask=data['variant_valid'].astype(bool) if 'variant_valid' in data else None
        if 'burden' in data:
            x=data['burden']; burdens.append(x[mask] if mask is not None and mask.shape==x.shape else x.reshape(-1))
        if 'safety_margin' in data:
            x=data['safety_margin']; margins.append(x[mask] if mask is not None and mask.shape==x.shape else x.reshape(-1))
    b=np.concatenate([np.asarray(x).reshape(-1) for x in burdens]) if burdens else np.asarray([])
    m=np.concatenate([np.asarray(x).reshape(-1) for x in margins]) if margins else np.asarray([])
    scales={'delay':1.0,'decel':1.0,'jerk':1.0,'dev':1.0,'trajectory':1.0,'burden':robust_scale(b),'margin':robust_scale(m),'source_dataset':args.dataset,'num_burden_values':int(b.size),'num_margin_values':int(m.size)}
    if b.size == 0 or m.size == 0:
        scales['warning']='No materialized NPZ label arrays found; defaults remain for unavailable primitive scales. Do not use this for paper results.'
    out.write_text(json.dumps(scales, indent=2)); print(out)
if __name__=='__main__': main()
