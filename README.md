# MFRP Planner: optimized reference implementation

This package implements the code path required by the paper idea: same-root intervention-response groups, support/query operator learning, scene-only deployment, monotone forced-dependence witness, split calibration, mechanism-feasible selection, and dataset diagnostics.

## Critical adapter setting

`build_same_root_dataset --adapter` is **not** a YAML option, class name, or built-in keyword. It must be an importable Python callable in this exact form:

```bash
--adapter examples.mfrp_waymax_adapter:build_groups
```

The callable signature must be:

```python
def build_groups(*, womd_pattern: str, split: str, config: dict,
                 max_scenarios: int | None, num_workers: int) -> Iterable[SameRootGroup]:
    ...
```

The adapter must return `SameRootGroup` objects generated from real same-root WOMD/Waymax rollouts. This bundle includes `examples.mfrp_waymax_adapter:build_groups`, which loads WOMD through Waymax, injects each ego candidate, and generates same-root route-following IDM reactive responses online. Optional precomputed rollout cache files are still supported, but are no longer required.

## Recommended paper commands

```bash
python -m scripts.build_same_root_dataset \
  --config configs/data/mfrp_womd_waymax.yaml \
  --split train \
  --out outputs/datasets/mfrp_womd_waymax \
  --womd-pattern '/data/womd/motion/train/*.tfrecord*' \
  --adapter examples.mfrp_waymax_adapter:build_groups \
  --max-scenarios 10000 \
  --num-workers 8 \
  --shard-size 8
```

Use the same command for validation/test, changing `--split` and `--womd-pattern`:

```bash
--split val  --womd-pattern '/data/womd/motion/validation/*.tfrecord*'
--split test --womd-pattern '/data/womd/motion/testing/*.tfrecord*'
```

Some WOMD installations name the folders `val`/`test` instead of `validation`/`testing`; use the actual local glob. The adapter parameter remains the same `module:function` value.

Then run:

```bash
python -m scripts.diagnose_dataset --dataset outputs/datasets/mfrp_womd_waymax/train --out outputs/diagnostics/mfrp_womd_waymax/train_dataset.json --markdown outputs/diagnostics/mfrp_womd_waymax/train_dataset.md
python -m scripts.compute_scales --dataset outputs/datasets/mfrp_womd_waymax/train --out outputs/scales/mfrp_womd_waymax/mfrp_scales.json
python -m scripts.train_mfrp --data outputs/datasets/mfrp_womd_waymax/train --model-config configs/model/mfrp_full.yaml --train-config configs/train/mfrp_train.yaml --out outputs/checkpoints/mfrp_womd_waymax/mfrp_best.pt --device cuda
python -m scripts.calibrate_mfrp --checkpoint outputs/checkpoints/mfrp_womd_waymax/mfrp_best.pt --data outputs/datasets/mfrp_womd_waymax/val --out outputs/calibration/mfrp_womd_waymax/mfrp_calibration.json --beta 0.1 --alpha 0.05 --device cuda
python -m scripts.eval_mfrp --checkpoint outputs/checkpoints/mfrp_womd_waymax/mfrp_best.pt --calibration outputs/calibration/mfrp_womd_waymax/mfrp_calibration.json --data outputs/datasets/mfrp_womd_waymax/test --out outputs/eval/mfrp_womd_waymax/test_eval.json --device cuda
```

## Paper-safety invariants enforced by code

1. `BUILD_SPEC.json` is never treated as a dataset.
2. `--adapter` must be `module:function` and must return real `SameRootGroup` instances.
3. Log playback is rejected as response supervision unless the run is explicitly debug-only.
4. Non-debug materialization requires at least two rollout variants.
5. Support/query split is by candidate: query candidates never enter the support encoder.
6. Deployment uses `scene_only_inference`; label-side tensors and support probes are stripped.
7. `priority_score_preexec` is required; legacy `priority_score` is flagged by diagnostics.
8. Dataset diagnostics fail when ceding/non-ceding diversity is too weak for forced-dependence labels.

## Smoke tests

```bash
python -m py_compile $(find mfrp scripts -name '*.py')
python -m scripts.train_mfrp --smoke --max_steps 2 --out outputs/checkpoints/debug/smoke.pt
python -m pytest tests -q
```

Smoke training is only for code-path checks and must not be reported as an experiment.
