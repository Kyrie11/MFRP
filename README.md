# MFRP Planner: audit-fixed reference implementation

This package implements the paper pipeline for **Mechanism-Feasible Response Planning**: same-root intervention groups, support/query operator learning, scene-only deployment, monotone forced-dependence witness, split calibration, mechanism-feasible selection, and dataset diagnostics.

The bundled adapter is intentionally honest about its backend: it loads WOMD scenarios through Waymax and builds same-root groups with an **online WOMD-route IDM proxy**. It is no longer described as full Waymax closed-loop reset/step simulation. Full Waymax closed-loop rollouts can be supplied through a rollout cache or added as a backend, but the code will not silently treat log playback as reactive supervision.

## Critical adapter setting

`build_same_root_dataset --adapter` must be an importable Python callable in this exact form:

```bash
--adapter examples.mfrp_waymax_adapter:build_groups
```

The callable signature is:

```python
def build_groups(*, womd_pattern: str, split: str, config: dict,
                 max_scenarios: int | None, num_workers: int) -> Iterable[SameRootGroup]:
    ...
```

## Recommended command skeleton

```bash
python -m scripts.build_same_root_dataset \
  --config configs/data/mfrp_womd_waymax.yaml \
  --split train \
  --out outputs/datasets/mfrp_womd_waymax \
  --womd-pattern '/data/womd/motion/training/*.tfrecord*' \
  --adapter examples.mfrp_waymax_adapter:build_groups \
  --max-scenarios 10000 \
  --shard-size 8
```

Use official training TFRecords for `train`. For offline validation/calibration/test labels, use validation TFRecords and rely on the adapter's deterministic `scene_id` hash partitioning so `val` and `test` roles do not share identical roots:

```bash
--split val  --womd-pattern '/data/womd/motion/validation/*.tfrecord*'
--split test --womd-pattern '/data/womd/motion/validation/*.tfrecord*'
```

Official WOMD testing files do not expose the same future labels needed by this proxy label builder, so they are appropriate for deployment-style unlabeled runs rather than the supervised response-label construction here.

Then run:

```bash
python -m scripts.diagnose_dataset --dataset outputs/datasets/mfrp_womd_waymax/train --out outputs/diagnostics/train_dataset.json --markdown outputs/diagnostics/train_dataset.md
python -m scripts.compute_scales --dataset outputs/datasets/mfrp_womd_waymax/train --out outputs/scales/mfrp_scales.json
python -m scripts.train_mfrp --data outputs/datasets/mfrp_womd_waymax/train --model-config configs/model/mfrp_full.yaml --train-config configs/train/mfrp_train.yaml --out outputs/checkpoints/mfrp_best.pt --device cuda
python -m scripts.calibrate_mfrp --checkpoint outputs/checkpoints/mfrp_best.pt --data outputs/datasets/mfrp_womd_waymax/val --out outputs/calibration/split_calibration.json --beta 0.1 --alpha 0.05 --selected-action --device cuda
python -m scripts.eval_mfrp --checkpoint outputs/checkpoints/mfrp_best.pt --calibration outputs/calibration/split_calibration.json --data outputs/datasets/mfrp_womd_waymax/test --out outputs/eval/mfrp_test.json --device cuda
```

## Paper-safety invariants enforced by code

1. `BUILD_SPEC.json` is never treated as a dataset.
2. `--adapter` must be `module:function` and must return `SameRootGroup` objects.
3. Log playback is rejected as response supervision unless the run is explicitly debug-only.
4. Non-debug materialization requires multiple rollout variants.
5. Ego candidates are generated from observed root-state kinematic primitives by default; logged-future anchors require explicit `dataset.allow_logged_future_anchor: true`.
6. Speed/timing changes reintegrate trajectory positions instead of only changing velocity fields.
7. Burden is computed against the same `(agent, policy variant)` neutral-candidate rollout.
8. Priority uses pre-execution information only; missing map/route/traffic-control evidence reduces confidence.
9. Support/query split is by candidate; query candidates never enter the support encoder.
10. Deployment uses `scene_only_inference`; label-side tensors and support probes are stripped.
11. Dataset diagnostics check support/query masks, variant validity, ceding/non-ceding evidence, priority leakage, duplicate roots, and geometry edges.

## Verification

```bash
python -m py_compile $(find mfrp scripts examples tests -name '*.py')
python -m scripts.train_mfrp --smoke --max_steps 2 --out outputs/checkpoints/debug/smoke.pt --device cpu
python -m pytest -q
```

The included regression suite currently passes with 26 tests.
