from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Any
import numpy as np

BRANCHES = ["keep", "cede", "brake", "accelerate", "pass", "nonconflict"]
CEDING = {1, 2}


def _stats(x) -> dict[str, Any]:
    x = np.asarray(x); x = x[np.isfinite(x)]
    if x.size == 0: return {"count": 0}
    return {"count": int(x.size), "mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()), "p05": float(np.quantile(x, .05)), "median": float(np.median(x)), "p95": float(np.quantile(x, .95)), "max": float(x.max())}


def diagnose_npz(path: Path) -> dict[str, Any]:
    d = np.load(path, allow_pickle=True)
    r: dict[str, Any] = {"file": str(path), "format": "npz", "keys": sorted(d.files)}
    if "candidate_features" in d: r["shape_candidate_features"] = list(d["candidate_features"].shape)
    vv = d["variant_valid"].astype(bool) if "variant_valid" in d else None
    if vv is not None:
        r["valid_rollout_variants"] = int(vv.sum()); r["variants_per_candidate_valid_stats"] = _stats(vv.sum(axis=-1).reshape(-1)) if vv.ndim == 4 else {}
    if "support_probe_mask" in d: r["support_probe_count"] = int(d["support_probe_mask"].astype(bool).sum())
    if "query_probe_mask" in d: r["query_label_count"] = int(d["query_probe_mask"].astype(bool).sum())
    if "branch_probs" in d:
        bp = d["branch_probs"]; mask = vv if vv is not None and vv.shape == bp.shape[:4] else np.ones(bp.shape[:4], bool)
        hard = bp.argmax(-1)
        total = max(1, int(mask.sum()))
        r["branch_counts"] = {BRANCHES[i]: int(((hard == i) & mask).sum()) for i in range(len(BRANCHES))}
        r["branch_fraction"] = {k: v / total for k, v in r["branch_counts"].items()}
        c = np.isin(hard, list(CEDING)) & mask; nc = (~np.isin(hard, list(CEDING))) & mask
        r["ceding_fraction"] = float(c.sum() / total); r["nonceding_fraction"] = float(nc.sum() / total)
        if mask.ndim == 4:
            both = (c.sum(-1) > 0) & (nc.sum(-1) > 0); valid = mask.sum(-1) > 0
            r["candidate_agent_with_both_ceding_and_nonceding"] = int(both.sum())
            r["candidate_agent_with_both_fraction"] = float(both.sum() / max(1, valid.sum()))
    for key in ["burden", "safety_margin", "priority_score_preexec", "priority_confidence_preexec", "cw_confidence"]:
        if key in d:
            arr = d[key]
            mask = vv if key in {"burden", "safety_margin"} and vv is not None and vv.shape == arr.shape else np.isfinite(arr)
            r[f"{key}_stats"] = _stats(arr[mask])
    if "safety_margin" in d:
        arr = d["safety_margin"]; mask = vv if vv is not None and vv.shape == arr.shape else np.isfinite(arr)
        vals = arr[mask]; r["unsafe_margin_fraction_h_lt_0"] = float((vals < 0).mean()) if vals.size else None
    if "priority_score_preexec" not in d: r["missing_priority_score_preexec"] = True
    if "priority_score" in d or "priority_confidence" in d: r["legacy_label_side_priority_present"] = True
    if "debug_only" in d: r["debug_only"] = bool(np.asarray(d["debug_only"]).reshape(-1)[0])
    if "root_hashes" in d:
        roots = [str(x) for x in np.asarray(d["root_hashes"], dtype=object).reshape(-1)]
        r["num_root_hashes"] = len(roots); r["unique_root_hashes"] = len(set(roots))
    if "edge_valid" in d: r["geometry_edges"] = int(d["edge_valid"].astype(bool).sum())
    return r


def diagnose_json(path: Path) -> dict[str, Any]:
    try: obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception: obj = None
    r = {"file": str(path), "format": path.suffix.lstrip("."), "readable": obj is not None}
    if isinstance(obj, dict) and obj.get("status") in {"spec_only", "requested"}:
        r["materialized"] = False; r["problem"] = "BUILD_SPEC only; no rollouts."
    return r


def aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    problems = []
    if not any(r.get("format") == "npz" and r.get("valid_rollout_variants", 0) > 0 for r in reports): problems.append("No materialized NPZ with valid rollout variants found.")
    for r in reports:
        if r.get("materialized") is False: problems.append(f"{r['file']}: spec only, not data.")
        if r.get("candidate_agent_with_both_fraction", 1.0) < 0.2: problems.append(f"{r['file']}: weak ceding/non-ceding diversity for forced-dependence labels.")
        if r.get("support_probe_count", 1) == 0: problems.append(f"{r['file']}: no support probes.")
        if r.get("query_label_count", 1) == 0: problems.append(f"{r['file']}: no query labels.")
        if r.get("missing_priority_score_preexec"): problems.append(f"{r['file']}: missing deployment-safe priority_score_preexec.")
        if r.get("legacy_label_side_priority_present"): problems.append(f"{r['file']}: legacy label-side priority present.")
        if r.get("debug_only"): problems.append(f"{r['file']}: debug_only shard.")
        if r.get("num_root_hashes") and r.get("unique_root_hashes") != r.get("num_root_hashes"): problems.append(f"{r['file']}: duplicate root hashes in shard.")
    return {"files_scanned": len(reports), "materialized_npz_files": sum(r.get("format") == "npz" for r in reports), "problems": problems, "paper_readiness": "pass" if not problems else "fail"}


def write_markdown(path: Path, summary: dict, reports: list[dict]) -> None:
    lines = ["# MFRP Dataset Diagnostic Report", "", f"Paper readiness: **{summary['paper_readiness']}**", ""]
    if summary["problems"]:
        lines += ["## Blocking issues", *[f"- {p}" for p in summary["problems"]], ""]
    lines.append("## Files")
    for r in reports:
        lines.append(f"### `{r['file']}`")
        for k, v in r.items():
            if k != "file": lines.append(f"- `{k}`: `{v}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--out", default="outputs/diagnostics/mfrp_dataset.json"); ap.add_argument("--markdown", default=None)
    args = ap.parse_args(); root = Path(args.dataset)
    files = sorted([*root.rglob("*.npz"), *root.rglob("*.json")]) if root.is_dir() else [root]
    reports = [diagnose_npz(f) if f.suffix == ".npz" else diagnose_json(f) for f in files]
    summary = aggregate(reports); payload = {"summary": summary, "files": reports}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = Path(args.markdown) if args.markdown else out.with_suffix(".md"); write_markdown(md, summary, reports)
    print(out); print(md)
    if summary["paper_readiness"] != "pass": raise SystemExit(2)

if __name__ == "__main__": main()
