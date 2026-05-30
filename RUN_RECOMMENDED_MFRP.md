# Recommended MFRP command sequence

This sequence follows the paper logic: same-root reactive data generation -> dataset diagnosis -> model training -> calibration -> testing -> deployment/selector. Replace `/path/to/...` with local WOMD/Waymax paths.

## 0. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 1. Dataset generation

The adapter builds same-root reactive data from WOMD scenarios loaded through Waymax. Ego candidates are now observation-only kinematic primitives generated from the root state, not perturbations of the logged SDC future. Surrounding vehicles are rolled with IDM-style route-following variants using a same-variant neutral baseline; this is still a WOMD-route IDM proxy rather than a full Waymax closed-loop environment. Log playback is rejected as response supervision; an optional `adapter.rollout_cache` can be used only to override the proxy rollout with precomputed reactive trajectories.

```bash
python scripts/build_same_root_dataset.py \
  --config configs/data/mfrp_womd_waymax.yaml \
  --split train \
  --out outputs/datasets/mfrp_womd_waymax \
  --womd-pattern "/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example//training/*.tfrecord" \
  --adapter examples.mfrp_waymax_adapter:build_groups \
  --max-scenarios 20000 \
  --shard-size 8

python scripts/build_same_root_dataset.py \
  --config configs/data/mfrp_womd_waymax.yaml \
  --split val \
  --out outputs/datasets/mfrp_womd_waymax \
  --womd-pattern "/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example//validation/*.tfrecord" \
  --adapter examples.mfrp_waymax_adapter:build_groups \
  --max-scenarios 2000 \
  --shard-size 8

python scripts/build_same_root_dataset.py \
  --config configs/data/mfrp_womd_waymax.yaml \
  --split test \
  --out outputs/datasets/mfrp_womd_waymax \
  --womd-pattern "/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example//validation/*.tfrecord" \
  --adapter examples.mfrp_waymax_adapter:build_groups \
  --max-scenarios 2000 \
  --shard-size 8
```

## 2. Dataset diagnosis

```bash
python scripts/diagnose_dataset.py \
  --dataset outputs/datasets/mfrp_womd_waymax/train \
  --out outputs/diagnostics/train_dataset.json \
  --markdown outputs/diagnostics/train_dataset.md

python scripts/diagnose_dataset.py \
  --dataset outputs/datasets/mfrp_womd_waymax/val \
  --out outputs/diagnostics/val_dataset.json \
  --markdown outputs/diagnostics/val_dataset.md
```

A paper-ready shard should have nonzero support/query masks, deployment-safe `priority_score_preexec`, valid rollout variants, and enough candidate-agent pairs with both ceding and non-ceding evidence.

## 3. Model training

```bash
python scripts/train_mfrp.py \
  --data outputs/datasets/mfrp_womd_waymax/train \
  --model-config configs/model/mfrp_full.yaml \
  --train-config configs/train/mfrp_train.yaml \
  --max_steps 200000 \
  --out outputs/checkpoints/mfrp_best.pt \
  --device cuda
```

Smoke check:

```bash
python scripts/train_mfrp.py --smoke --max_steps 2 --out outputs/checkpoints/mfrp_smoke.pt --device cpu
```

## 4. Calibration

```bash
python scripts/calibrate_mfrp.py \
  --checkpoint outputs/checkpoints/mfrp_best.pt \
  --data outputs/datasets/mfrp_womd_waymax/val \
  --out outputs/calibration/split_calibration.json \
  --beta 0.10 \
  --alpha 0.05 \
  --selected-action \
  --device cuda
```

## 5. Model testing

```bash
python scripts/eval_mfrp.py \
  --checkpoint outputs/checkpoints/mfrp_best.pt \
  --calibration outputs/calibration/split_calibration.json \
  --data outputs/datasets/mfrp_womd_waymax/test \
  --metrics prediction,false_safe,boundary,calibration \
  --out outputs/eval/mfrp_test.json \
  --device cuda
```

## 6. Deployment / scene-only test

Deployment sanitizes label-side tensors and support probes. Only scene/candidate/pre-execution priority features remain.

```bash
python scripts/deploy_mfrp.py \
  --checkpoint outputs/checkpoints/mfrp_best.pt \
  --scenario /path/to/deployment_ready_scene_batch.npz \
  --calibration outputs/calibration/split_calibration.json \
  --device cuda
```

## 7. Experiment switches

Use these as controlled ablation switches in YAML or launch scripts:

```yaml
surface_support: true          # false => scene-only training diagnostic
lambda_distill: 1.0            # 0.0 => no support-to-scene distillation
lambda_geo: 0.1                # 0.0 => no response-surface geometry loss
lambda_cw: 1.0                 # 0.0 => no coercion witness supervision
cw_pairwise_ranking: true      # false => BCE-only witness
planner_alpha: 0.05
planner_nu_bar: 0.30
planner_gamma_bar: 0.60        # enables boundary-sensitive rejection
risk_aggregation: noisy_or     # compare with max and sum
selector_mode: mechanism_feasible # compare with risk_only and pressure_penalty
support_query_policy: stratified_by_candidate_family
```
