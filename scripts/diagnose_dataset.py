from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

BRANCHES = ["keep", "cede", "brake", "accelerate", "pass", "nonconflict"]
CEDING = {1, 2}


def _safe_stats(x: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"count": 0}
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p25": float(np.quantile(x, 0.25)),
        "median": float(np.median(x)),
        "p75": float(np.quantile(x, 0.75)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x)),
    }


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def diagnose_npz(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    out: dict[str, Any] = {"file": str(path), "format": "npz", "keys": sorted(data.files)}
    if "candidate_features" in data:
        cf = data["candidate_features"]
        out["shape_candidate_features"] = list(cf.shape)
        if cf.ndim >= 4:
            out["num_groups"] = int(cf.shape[0]); out["max_agents"] = int(cf.shape[1]); out["max_candidates"] = int(cf.shape[2])
    if "variant_valid" in data:
        vv = data["variant_valid"].astype(bool)
        out["shape_variant_valid"] = list(vv.shape)
        out["valid_rollout_variants"] = int(vv.sum())
        if vv.ndim == 4:
            out["variants_per_candidate_valid_stats"] = _safe_stats(vv.sum(axis=-1).reshape(-1))
    else:
        vv = None
    if "query_probe_mask" in data and "support_probe_mask" in data:
        q = data["query_probe_mask"].astype(bool)
        s = data["support_probe_mask"].astype(bool)
        out["query_label_count"] = int(q.sum())
        out["support_probe_count"] = int(s.sum())
    if "branch_probs" in data:
        bp = data["branch_probs"]
        mask = vv if vv is not None and vv.shape == bp.shape[:4] else np.ones(bp.shape[:4], dtype=bool)
        hard = np.argmax(bp, axis=-1)
        counts = {BRANCHES[i]: int(((hard == i) & mask).sum()) for i in range(len(BRANCHES))}
        total = max(1, int(mask.sum()))
        out["branch_counts"] = counts
        out["branch_fraction"] = {k: v / total for k, v in counts.items()}
        ceding_mask = np.isin(hard, list(CEDING)) & mask
        nonceding_mask = (~np.isin(hard, list(CEDING))) & mask
        out["ceding_fraction"] = float(ceding_mask.sum() / total)
        out["nonceding_fraction"] = float(nonceding_mask.sum() / total)
        if mask.ndim == 4:
            both = (ceding_mask.sum(axis=-1) > 0) & (nonceding_mask.sum(axis=-1) > 0)
            valid_cands = mask.sum(axis=-1) > 0
            out["candidate_agent_with_both_ceding_and_nonceding"] = int(both.sum())
            out["candidate_agent_with_both_fraction"] = float(both.sum() / max(1, valid_cands.sum()))
    if "burden" in data:
        mask = vv if vv is not None and vv.shape == data["burden"].shape else np.isfinite(data["burden"])
        out["burden_stats"] = _safe_stats(data["burden"][mask])
    if "safety_margin" in data:
        mask = vv if vv is not None and vv.shape == data["safety_margin"].shape else np.isfinite(data["safety_margin"])
        vals = data["safety_margin"][mask]
        out["safety_margin_stats"] = _safe_stats(vals)
        out["unsafe_margin_fraction_h_lt_0"] = float((vals < 0).mean()) if vals.size else None
        out["near_margin_fraction_h_lt_1m"] = float((vals < 1.0).mean()) if vals.size else None
    if "priority_score_preexec" in data:
        out["priority_score_preexec_stats"] = _safe_stats(data["priority_score_preexec"].reshape(-1))
    else:
        out["missing_priority_score_preexec"] = True
    if "priority_confidence_preexec" in data:
        out["priority_confidence_preexec_stats"] = _safe_stats(data["priority_confidence_preexec"].reshape(-1))
    if "priority_score" in data or "priority_confidence" in data:
        out["legacy_label_side_priority_present"] = True
    if "debug_only" in data:
        out["debug_only"] = bool(np.asarray(data["debug_only"]).reshape(-1)[0])
    if "root_hashes" in data:
        roots = [str(x) for x in np.asarray(data["root_hashes"], dtype=object).reshape(-1)]
        out["num_root_hashes"] = len(roots)
        out["unique_root_hashes"] = len(set(roots))
    if "cw_confidence" in data:
        out["cw_confidence_stats"] = _safe_stats(data["cw_confidence"].reshape(-1))
        out["cw_effective_labels"] = int((data["cw_confidence"] > 0.1).sum())
    if "edge_valid" in data:
        out["geometry_edges"] = int(data["edge_valid"].astype(bool).sum())
    return out


def diagnose_json(path: Path) -> dict[str, Any]:
    obj = _load_json(path)
    out: dict[str, Any] = {"file": str(path), "format": path.suffix.lstrip("."), "readable": obj is not None}
    if isinstance(obj, dict):
        out["top_level_keys"] = sorted(obj.keys())[:50]
        if obj.get("status") in {"spec_only", "requested", "not_materialized", "requires WOMD/Waymax source paths for full build"}:
            out["materialized"] = False
            out["problem"] = "Only a BUILD_SPEC/request was written; no same-root response rollouts are present."
        if "metadata" in obj:
            out["metadata_keys"] = sorted(obj["metadata"].keys())[:50]
    return out


def aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    agg: dict[str, Any] = {"files_scanned": len(reports), "materialized_npz_files": sum(r.get("format") == "npz" for r in reports)}
    problems: list[str] = []
    if not any(r.get("format") == "npz" and r.get("valid_rollout_variants", 0) > 0 for r in reports):
        problems.append("No materialized NPZ with valid rollout variants found; dataset cannot train/evaluate the paper pipeline yet.")
    for r in reports:
        if r.get("candidate_agent_with_both_fraction", 1.0) < 0.2:
            problems.append(f"{r['file']}: too few candidate-agent cells contain both ceding and non-ceding variants; forced-dependence labels will be weak.")
        if r.get("support_probe_count", 1) == 0 and r.get("query_label_count", 0) > 0:
            problems.append(f"{r['file']}: has query labels but no support probes; support-adapted training/distillation is disabled for those examples.")
        if r.get("missing_priority_score_preexec"):
            problems.append(f"{r['file']}: missing deployment-safe priority_score_preexec; rebuild to avoid priority leakage.")
        if r.get("legacy_label_side_priority_present"):
            problems.append(f"{r['file']}: legacy label-side priority_score/priority_confidence present; remove or rename to *_label before training/deployment.")
        if r.get("debug_only"):
            problems.append(f"{r['file']}: debug_only shard; do not use for paper results.")
        if r.get("num_root_hashes") and r.get("unique_root_hashes") != r.get("num_root_hashes"):
            problems.append(f"{r['file']}: duplicate root_hashes inside shard; check same-root grouping and split leakage.")
        if r.get("materialized") is False:
            problems.append(f"{r['file']}: build spec only, not a dataset shard.")
    agg["problems"] = problems
    agg["paper_readiness"] = "pass" if not problems else "fail"
    return agg


def write_markdown(path: Path, summary: dict[str, Any], reports: list[dict[str, Any]]) -> None:
    lines = ["# MFRP Dataset Diagnostic Report", "", f"Paper readiness: **{summary['paper_readiness']}**", ""]
    if summary["problems"]:
        lines.append("## Blocking issues")
        for p in summary["problems"]:
            lines.append(f"- {p}")
        lines.append("")
    lines.append("## Files")
    for r in reports:
        lines.append(f"### `{r['file']}`")
        for k, v in r.items():
            if k == "file":
                continue
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose whether a generated MFRP dataset supports the paper requirements.")
    ap.add_argument("--dataset", required=True, help="Dataset root or split directory")
    ap.add_argument("--out", default="outputs/diagnostics/mfrp_dataset_diagnostics.json")
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()
    root = Path(args.dataset)
    files = sorted([*root.rglob("*.npz"), *root.rglob("*.json")]) if root.is_dir() else [root]
    reports = []
    for f in files:
        if f.suffix == ".npz":
            reports.append(diagnose_npz(f))
        elif f.suffix == ".json":
            reports.append(diagnose_json(f))
    summary = aggregate(reports)
    payload = {"summary": summary, "files": reports}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = Path(args.markdown) if args.markdown else out.with_suffix(".md")
    write_markdown(md, summary, reports)
    print(out)
    print(md)
    if summary["paper_readiness"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
