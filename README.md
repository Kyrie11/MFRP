# MFRP Planner: Mechanism-Feasible Response Planning

This repository implements **MFRP (Mechanism-Feasible Response Planning)** for non-coercive autonomous driving. MFRP learns same-root intervention-response surfaces: for a fixed observed root scene and a relevant surrounding agent, the model maps an ego intervention candidate `u` to a distribution over the response primitive

```text
O_i(u) = (M_i(u), Y_i(u), ΔB_i(u), H_i(u))
```

where `M` is a six-way soft response branch, `Y` is a branch-conditioned future trajectory, `ΔB` is continuous baseline-relative induced burden, and `H` is a signed safety margin. Planning accepts candidates inside a split-calibrated mechanism-feasible set instead of hiding coercive behavior behind a generic pressure penalty.

## Current code status

The repository now contains a full **materialized-shard training/evaluation path** and refuses to silently create fake paper data. It still cannot access your private WOMD files or run your local Waymax installation inside this zip, so the WOMD/Waymax-specific scenario loader is supplied through an adapter contract:

```text
WOMD/Waymax local adapter -> SameRootGroup objects -> collate_same_root_groups -> compressed NPZ shards
```

The code is paper-safe in the following sense:

- `BUILD_SPEC.json` is not accepted as a dataset.
- empty calibration is rejected;
- `eval_mfrp.py` computes real metrics from checkpoint + NPZ shards;
- training fails if support/query masks are missing, unless explicitly marked debug;
- deployment-safe priority is named `priority_score_preexec`; legacy `priority_score` is treated as label-side and stripped from deployment;
- root hash fingerprints were removed from scene features to reduce split memorization risk.

## Repository layout

```text
mfrp/
  data/
    schema.py              # RootScene/EgoCandidate/ResponseObservation/SameRootGroup schema
    scene_schema.py        # observed root-scene containers
    interaction_region.py  # deployment-safe intervention coordinates
    label_extraction.py    # branch, ΔB, signed margin, high-pressure and CW labels
    priority.py            # pre-execution priority score and confidence
    support_query.py       # candidate-level support/query split
    tensors.py             # [B,A,K,R] batch collation; no label-side priority leakage
    materialize.py         # adapter contract and NPZ writer
  models/
    mfrp_model.py          # MFRPModel(scene_only/support_adapted/both)
    response_heads.py      # branch, trajectory, continuous burden and margin heads
    coercion_witness.py    # monotone κ_i(u) witness head
    mechanism_operator.py  # scene/support mechanism token operator
    support_encoder.py
  training/
    losses.py              # L_mech, distillation, response geometry, CW event/ranking
  planning/
    estimators.py          # p_unsafe, P_C, S_C, S_notC, B_C, D_C, rho_mech
    calibration.py         # split calibration residual q_beta; rejects empty data
    selector.py            # mechanism-feasible selection and conservative fallback
    deploy.py              # scene-only deployment sanitizer/API
  evaluation/
    runtime.py             # checkpoint loading, NPZ tensor loading, risk prediction
    metrics.py             # offline prediction/risk metrics
configs/
  data/mfrp_womd_waymax.yaml
  model/mfrp_full.yaml
  train/mfrp_train.yaml
  eval/mfrp_eval.yaml
scripts/
  build_same_root_dataset.py
  diagnose_dataset.py
  compute_scales.py
  train_mfrp.py
  calibrate_mfrp.py
  eval_mfrp.py
  deploy_mfrp.py
```

## Core invariants

1. **Same-root reset:** all candidates and rollout variants in a `SameRootGroup` must start from the identical simulator state at `t0`; only the ego intervention changes.
2. **No future leakage:** deployment inputs may include only observed root scene, ego candidate, map/route/history, and pre-execution priority features. Realized futures, support probes, query labels, rollout diagnostics, and response labels are label/evaluation data only.
3. **Candidate-level support/query split:** if a candidate is query, all its rollout variants are withheld from the support encoder.
4. **Scene-only deployment:** support-adapted tokens are training-time teachers; deployment uses `model(batch, mode="scene_only")` only.
5. **No log playback for supervision:** log playback is diagnostic only because it does not react to counterfactual ego interventions.
6. **Coercion is not high pressure alone:** a candidate is coercive only when safety depends on high-pressure ceding, plausible non-ceding responses are unsafe, and local priority does not justify ego forcing its way in.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e '.[dev]'
```

Install optional WOMD/Waymax dependencies in your local data-generation environment:

```bash
pip install -e '.[waymax]'
```

Run sanity tests:

```bash
python -m pytest tests -q
python -m py_compile $(find mfrp scripts -name '*.py')
```

## End-to-end paper pipeline

### 0. Decide experiment switches before generating data

Use `configs/data/mfrp_womd_waymax.yaml` as the paper default. For each run, record these switches in a separate config copy:

```yaml
rollout:
  backend: waymax
  ego_execution_mode: state_control
  debug_allow_fallback: false
  policy_variants:
    - neutral_idm
    - conservative_idm
    - assertive_idm
labels:
  branch_set: [keep, cede, brake, accelerate, pass, nonconflict]
  ceding_branches: [cede, brake]
planning:
  alpha: 0.05
  beta_calibration: 0.1
  nu_bar: 0.3
  gamma_bar: 2.0
```

Main-paper runs should keep `debug_allow_fallback: false`. Ablations can change support, distillation, geometry, CW, policy variants, or selector gates, but the setting must be encoded in the config filename and output directory.

Recommended output layout:

```text
outputs/
  datasets/<run_name>/{train,val,test}/
  diagnostics/<run_name>/
  scales/<run_name>/
  checkpoints/<run_name>/
  calibration/<run_name>/
  eval/<run_name>/
```

### 1. Implement the WOMD/Waymax materializer adapter

`build_same_root_dataset.py` expects a Python callable formatted as `module:function`. The callable receives keyword arguments and returns an iterable of `SameRootGroup` objects.

Minimal adapter contract:

```python
# my_project/mfrp_waymax_adapter.py
from collections.abc import Iterable
from mfrp.data.schema import SameRootGroup


def build_groups(*, womd_pattern: str, split: str, config: dict, max_scenarios: int | None, num_workers: int) -> Iterable[SameRootGroup]:
    # 1. Load WOMD scenario protos / TFExamples for the requested split.
    # 2. Reset Waymax to the same t0 state for every ego candidate.
    # 3. Execute ego candidates with multiple reactive policy variants.
    # 4. Extract ResponseObservation labels: branch_probs, trajectory, burden, margin, priority diagnostics.
    # 5. Create candidate-level support/query split via mfrp.data.support_query.split_support_query.
    # 6. Add boundary_pairs for nearby intervention candidates when available.
    # 7. Return SameRootGroup objects.
    yield group
```

Each returned `SameRootGroup` must include:

- `root_scene` with observed history/current state only;
- `candidates` with ego future trajectories in the `t0` ego frame;
- `relevant_agent_ids`;
- at least two `rollout_variants` for non-debug data;
- `observations[(candidate_id, agent_id, variant_id)]` for valid rollouts;
- `metadata["support_candidate_ids"]` and `metadata["query_candidate_ids"]` with no overlap;
- no `metadata["uses_log_playback_for_response"] = true` for paper data.

### 2. Generate materialized same-root datasets

Train split:

```bash
python -m scripts.build_same_root_dataset \
  --config configs/data/mfrp_womd_waymax.yaml \
  --split train \
  --out outputs/datasets/mfrp_womd_waymax \
  --womd-pattern '/data/womd/motion/train/*.tfrecord*' \
  --adapter my_project.mfrp_waymax_adapter:build_groups \
  --max-scenarios 10000 \
  --num-workers 8 \
  --shard-size 8
```

Validation split:

```bash
python -m scripts.build_same_root_dataset \
  --config configs/data/mfrp_womd_waymax.yaml \
  --split val \
  --out outputs/datasets/mfrp_womd_waymax \
  --womd-pattern '/data/womd/motion/val/*.tfrecord*' \
  --adapter my_project.mfrp_waymax_adapter:build_groups \
  --num-workers 8 \
  --shard-size 8
```

Test split:

```bash
python -m scripts.build_same_root_dataset \
  --config configs/data/mfrp_womd_waymax.yaml \
  --split test \
  --out outputs/datasets/mfrp_womd_waymax \
  --womd-pattern '/data/womd/motion/test/*.tfrecord*' \
  --adapter my_project.mfrp_waymax_adapter:build_groups \
  --num-workers 8 \
  --shard-size 8
```

The script writes compressed `.npz` shards plus `.metadata.json` files. It fails if the adapter returns no groups, no candidates, no rollout variants, no observations, missing support/query split, overlapping support/query sets, or log-playback response supervision.

For copying already materialized shards into this layout:

```bash
python -m scripts.build_same_root_dataset \
  --config configs/data/mfrp_womd_waymax.yaml \
  --split train \
  --out outputs/datasets/mfrp_womd_waymax \
  --input-npz /path/to/materialized/train_shards
```

`--write-spec-only` writes only `BUILD_SPEC.json`; it is useful for run planning but is not accepted by diagnostics, training, calibration, or evaluation as a real dataset.

### 3. Run dataset diagnostics for every split

Train diagnostics:

```bash
python -m scripts.diagnose_dataset \
  --dataset outputs/datasets/mfrp_womd_waymax/train \
  --out outputs/diagnostics/mfrp_womd_waymax/train_dataset.json \
  --markdown outputs/diagnostics/mfrp_womd_waymax/train_dataset.md
```

Validation and test diagnostics:

```bash
python -m scripts.diagnose_dataset \
  --dataset outputs/datasets/mfrp_womd_waymax/val \
  --out outputs/diagnostics/mfrp_womd_waymax/val_dataset.json \
  --markdown outputs/diagnostics/mfrp_womd_waymax/val_dataset.md

python -m scripts.diagnose_dataset \
  --dataset outputs/datasets/mfrp_womd_waymax/test \
  --out outputs/diagnostics/mfrp_womd_waymax/test_dataset.json \
  --markdown outputs/diagnostics/mfrp_womd_waymax/test_dataset.md
```

A paper-ready split must pass the following checks:

- at least one materialized `.npz` shard with valid rollout variants;
- non-zero `query_probe_mask` and `support_probe_mask`;
- branch diversity is not degenerate;
- enough candidate-agent cells contain both ceding and non-ceding variants;
- finite burden and safety-margin distributions;
- `priority_score_preexec` and `priority_confidence_preexec` exist;
- legacy label-side `priority_score` / `priority_confidence` are absent;
- no `debug_only` shards;
- no duplicate root hashes inside a shard;
- `BUILD_SPEC.json` is not the only artifact.

The diagnostic script exits with status `2` when a blocking issue is found. Do not train on a split that fails diagnostics unless it is an explicitly named debug run.

### 4. Compute robust scales from train only

```bash
python -m scripts.compute_scales \
  --dataset outputs/datasets/mfrp_womd_waymax/train \
  --out outputs/scales/mfrp_womd_waymax/mfrp_scales.json
```

Compute scales from the train split only. Validation/test data must never affect label scaling or normalization decisions.

### 5. Train MFRP

Paper training:

```bash
python -m scripts.train_mfrp \
  --data outputs/datasets/mfrp_womd_waymax/train \
  --model-config configs/model/mfrp_full.yaml \
  --train-config configs/train/mfrp_train.yaml \
  --out outputs/checkpoints/mfrp_womd_waymax/mfrp_best.pt \
  --device cuda
```

The trainer now enforces:

- `priority_score_preexec` must exist;
- `branch_probs`, `trajectory`, `burden`, `safety_margin`, and `variant_valid` must exist;
- query labels must exist;
- support probes must exist unless `--allow-debug-no-support` is set.

Synthetic smoke training is available only for code-path checks:

```bash
python -m scripts.train_mfrp \
  --model-config configs/model/mfrp_full.yaml \
  --train-config configs/train/mfrp_train.yaml \
  --smoke \
  --max_steps 20 \
  --out outputs/checkpoints/debug/mfrp_smoke.pt
```

Do not report smoke runs as experiments.

### 6. Fit split calibration on validation data

```bash
python -m scripts.calibrate_mfrp \
  --checkpoint outputs/checkpoints/mfrp_womd_waymax/mfrp_best.pt \
  --data outputs/datasets/mfrp_womd_waymax/val \
  --out outputs/calibration/mfrp_womd_waymax/mfrp_calibration.json \
  --beta 0.1 \
  --device cuda
```

Calibration uses scene-only predictions on validation shards. It derives candidate-level violation truth from negative safety margin and confident CW labels, then fits the residual quantile:

```text
q_beta = Quantile_{1-beta}(V(u) - rho_mech_hat(u))
rho_mech_cal(u) = min(1, rho_mech_hat(u) + q_beta)
```

If no validation examples are found, calibration raises an error and writes nothing.

### 7. Evaluate on held-out test data

```bash
python -m scripts.eval_mfrp \
  --checkpoint outputs/checkpoints/mfrp_womd_waymax/mfrp_best.pt \
  --calibration outputs/calibration/mfrp_womd_waymax/mfrp_calibration.json \
  --data outputs/datasets/mfrp_womd_waymax/test \
  --metrics prediction,false_safe,boundary,calibration \
  --out outputs/eval/mfrp_womd_waymax/test_eval.json \
  --device cuda
```

The evaluator reports available metrics such as:

- `branch_ce`, `branch_acc`;
- `burden_mae`, `margin_mae`, `traj_ade`;
- `risk_auroc`, `uncalibrated_risk_mean`, `truth_violation_rate`;
- `uncalibrated_false_safe_rate_at_alpha`, `uncalibrated_feasible_fraction_at_alpha`;
- `calibrated_risk_mean`, `calibrated_risk_violation_rate_at_alpha`, `calibrated_feasible_fraction_at_alpha`;
- `cw_auroc` when confident CW labels are present;
- `boundary_sensitivity_mean` when boundary edges exist.

Some paper tables may still require external closed-loop Waymax evaluation. Use this script for checkpoint-level response/risk evaluation; use your local Waymax harness for closed-loop rollouts and feed the selected candidate outputs into the same reporting directory.

### 8. Deployment / planning path

Use the deployment sanitizer to guarantee scene-only inference:

```python
from mfrp.planning.deploy import scene_only_inference

scene_outputs = scene_only_inference(model, batch)
```

`scene_only_inference` strips label-side keys including support/query masks, branch/trajectory/burden/margin labels, rollout validity, CW labels, diagnostics, groups, and legacy `priority_score` / `priority_confidence`. Deployment may keep only `priority_score_preexec` and `priority_confidence_preexec`, which are derived from root-scene/candidate/map metadata before rollout.

Command-line deployment stub:

```bash
python -m scripts.deploy_mfrp \
  --checkpoint outputs/checkpoints/mfrp_womd_waymax/mfrp_best.pt \
  --calibration outputs/calibration/mfrp_womd_waymax/mfrp_calibration.json \
  --scenario <scenario_source> \
  --candidate-config configs/data/mfrp_womd_waymax.yaml
```

## Mechanism-feasible selection

For each agent and candidate, MFRP estimates:

```text
p_unsafe_i(u) = P(H_i(u) < 0 | s,u,Ω_i^0)
P_C           = P(M in {cede, brake} | u)
S_C           = P(H > 0 | M in {cede, brake}, u)
S_notC        = P(H > 0 | M not in {cede, brake}, u)
B_C           = E[ΔB | M in {cede, brake}, u]
D_C           = relu(S_C - S_notC)
kappa_i(u)    = learned monotone coercion witness
p_viol_i      = p_unsafe_i + kappa_i - p_unsafe_i * kappa_i
rho_mech      = 1 - prod_i(1 - p_viol_i)
```

A candidate is feasible only if:

```text
rho_mech_cal(u) <= alpha
nu(u)           <= nu_bar
gamma(u)        <= gamma_bar
candidate_valid(u)
```

The planner chooses the lowest nominal-cost feasible candidate. If none is feasible, it may generate repair candidates such as yielding more, increasing gap, delaying entry, lowering speed, or reducing lateral commitment. If the set is still empty, conservative fallback minimizes nominal cost plus gate violations and marks `fallback_used=true` with active violations.

## Experiment switches and ablations

Use separate config/output directories for each of these:

```text
full_mfrp                  # support-adapted training + distillation + geometry + CW
wo_support                 # scene-only loss; no support adaptation
wo_distillation            # support loss but no support->scene distillation
wo_geometry                # no response geometry boundary loss
wo_cw                      # no coercion witness loss/gate
idm_only                   # one reactive policy family only
learned_sim_agent          # learned reactive agents if available
no_priority_gate           # priority features neutralized, should increase uncertainty
high_alpha / low_alpha     # selector risk threshold sensitivity
```

Main reported numbers should include at least:

1. dataset diagnostics summary for train/val/test;
2. held-out response prediction;
3. CW witness quality;
4. calibration coverage / feasible-set violation rate;
5. false-safe offline selection;
6. closed-loop Waymax planning metrics if available;
7. ablations above.

## Important implementation notes

- `priority_score_preexec` is deployment-safe; `priority_score` is legacy/label-side and should not appear in new shards.
- `collate_same_root_groups(..., require_support_query_split=True)` is the default. Use `require_support_query_split=False` only for unit tests or smoke debugging.
- `support_probe_features` intentionally include label-side response summaries; they are training/evaluation only and are stripped during deployment.
- `scene_features` do not contain root hash or scene-id fingerprints.
- The current interaction-region implementation remains a lightweight geometric proxy. For paper-quality data, the adapter should compute map/lane/swept-box conflict metadata and pass it through candidate/root metadata so priority and interaction features are pre-execution and map-aware.
- `rho_mech` uses a noisy-OR aggregation over agents. If your simulator exposes correlated joint response samples, report a correlated-risk ablation in addition to this default.

## Quick debug checklist

```bash
python -m pytest tests -q
python -m scripts.train_mfrp --smoke --max_steps 2 --out outputs/checkpoints/debug/smoke.pt
python -m scripts.build_same_root_dataset --config configs/data/mfrp_womd_waymax.yaml --split train --write-spec-only
python -m scripts.diagnose_dataset --dataset outputs/datasets/mfrp_womd_waymax/train || true
```

The final command should fail for a spec-only directory. That is expected and confirms the pipeline will not accidentally train on placeholders.
