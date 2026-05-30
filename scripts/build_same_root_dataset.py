from __future__ import annotations

import argparse, json, shutil
from pathlib import Path
import yaml
from mfrp.data.materialize import load_adapter, materialize_with_adapter


def main() -> None:
    p = argparse.ArgumentParser(description="Build materialized MFRP same-root intervention-response shards.")
    p.add_argument("--config", required=True)
    p.add_argument("--split", required=True, choices=["train", "val", "test", "mini", "debug"])
    p.add_argument("--out", default="outputs/datasets/mfrp_womd_waymax")
    p.add_argument("--womd-pattern", default=None)
    p.add_argument("--adapter", default=None, help="Python adapter callable in module:function format")
    p.add_argument("--input-npz", default=None)
    p.add_argument("--max-scenarios", type=int, default=None)
    p.add_argument(
        "--max-source-scenarios",
        type=int,
        default=None,
        help=(
            "Optional hard cap on raw WOMD simulator states scanned before giving up. "
            "This is useful for quick paper checks when filters are too strict or the "
            "Waymax dataloader unexpectedly streams more than one physical TFRecord."
        ),
    )
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--shard-size", type=int, default=8)
    p.add_argument("--debug-allow-no-support", action="store_true")
    p.add_argument("--write-spec-only", action="store_true")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if cfg.get("rollout", {}).get("debug_allow_fallback", False) and not args.debug_allow_no_support:
        raise SystemExit("debug_allow_fallback=true creates debug-only data. Use --debug-allow-no-support only for smoke tests.")
    out = Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)
    spec = vars(args) | {"config": cfg}
    (out / "BUILD_SPEC.json").write_text(json.dumps({**spec, "status": "spec_only" if args.write_spec_only else "requested"}, indent=2, default=str), encoding="utf-8")
    if args.write_spec_only:
        print(f"Wrote spec only to {out}; this is not a dataset.")
        return
    if args.input_npz:
        src = Path(args.input_npz)
        files = sorted(src.rglob("*.npz")) if src.is_dir() else [src]
        if not files or not all(f.exists() and f.suffix == ".npz" for f in files):
            raise SystemExit("--input-npz must point to an NPZ shard or directory of shards")
        copied = []
        for f in files:
            dst = out / f.name
            shutil.copy2(f, dst)
            meta = f.with_suffix(".metadata.json")
            if meta.exists():
                shutil.copy2(meta, dst.with_suffix(".metadata.json"))
            copied.append(str(dst))
        (out / "BUILD_SPEC.json").write_text(json.dumps({**spec, "status": "copied_materialized_npz", "shards": copied}, indent=2, default=str), encoding="utf-8")
        print("\n".join(copied))
        return
    if not args.adapter:
        raise SystemExit("Missing --adapter. Use module:function, e.g. my_project.mfrp_waymax_adapter:build_groups. Do not use a YAML name or class name.")
    if not args.womd_pattern:
        raise SystemExit("Missing --womd-pattern. Refusing to create placeholder data.")
    adapter = load_adapter(args.adapter)
    written = materialize_with_adapter(
        adapter,
        split=args.split,
        out_dir=args.out,
        womd_pattern=args.womd_pattern,
        config=cfg,
        max_scenarios=args.max_scenarios,
        max_source_scenarios=args.max_source_scenarios,
        shard_size=args.shard_size,
        num_workers=args.num_workers,
        debug_allow_no_support=args.debug_allow_no_support,
    )
    (out / "BUILD_SPEC.json").write_text(json.dumps({**spec, "status": "materialized", "shards": [str(p) for p in written]}, indent=2, default=str), encoding="utf-8")
    print("\n".join(str(p) for p in written))


if __name__ == "__main__":
    main()
