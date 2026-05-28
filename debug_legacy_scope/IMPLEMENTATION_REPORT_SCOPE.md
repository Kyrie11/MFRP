# SCOPE Planner Implementation Report

## Scope of changes

This revision upgrades the repository from a smoke-test skeleton toward an agent-centric same-root SCOPE implementation. It prioritizes the paper-critical path: fixed-root intervention-response surfaces, `[B,A,K]` datasets, diagnostic FD/boundary labels, scene-only planning estimates, repair/fallback, experiments, calibration, and realistic README status.

## Major modified or added files

- `README.md`
- `IMPLEMENTATION_REPORT.md`
- `configs/data/womd_waymax.yaml`
- `configs/experiment/closed_loop.yaml`
- `scope/data/scene_schema.py`
- `scope/data/candidates.py`
- `scope/data/feasibility.py`
- `scope/data/labels.py`
- `scope/data/dataset.py`
- `scope/data/dataset_writer.py`
- `scope/data/dataset_diagnostics.py`
- `scope/data/rollout_waymax.py`
- `scope/models/scene_encoder.py`
- `scope/models/intervention_encoder.py`
- `scope/models/operator.py`
- `scope/models/heads.py`
- `scope/models/scope_model.py`
- `scope/models/baselines.py`
- `scope/training/support_query.py`
- `scope/training/losses.py`
- `scope/training/train.py`
- `scope/training/evaluate_response.py`
- `scope/training/calibration.py`
- `scope/planning/estimators.py`
- `scope/planning/closed_loop.py`
- `scope/experiments/run_heldout_response.py`
- `scope/experiments/run_surface_geometry_fd.py`
- `scope/experiments/run_false_safe_offline.py`
- `scope/experiments/run_closed_loop.py`
- `scope/experiments/plot_surface_slices.py`
- `tests/test_agent_centric_dataset_and_ablation.py`

## Completion status

| Item | Status |
|---|---|
| P0 diagnostics bug | Complete |
| P0 same-root policy consistency | Complete |
| P0 backend/fallback reporting and hard-fail option | Complete |
| Agent-centric `[B,A,K]` collate | Complete |
| Full observation tensors | Complete for available schema; summary fallback retained for smoke only |
| FD diagnostic labels | Complete when non-ceding rollout/neighbor evidence exists; invalid when evidence absent |
| Boundary labels | Complete |
| Ordinal burden monotonic head | Complete |
| Response heads branch+burden conditioned | Complete |
| Support-query by `(scene, agent)` | Complete |
| Manifold loss wiring | Complete |
| Validation/class weights/scheduler/mixed precision | Complete |
| Calibration | Complete validation-only implementation |
| Held-out response metrics | Complete |
| Surface geometry/FD metrics | Complete |
| False-safe offline selectors | Complete; model path plus explicit label-proxy fallback |
| Closed-loop scene-only planning | Implemented smoke/experimental loop |
| Native Waymax controlled tracking | Partial; requires local Waymax API validation |
| CARLA/stress backend | Not implemented |
| Large-scale parquet/zarr storage | Not implemented |

## Verification results

### Unit tests

```text
pytest -q
12 passed, 2 warnings in 25.17s
```

### Synthetic smoke diagnostics

```text
groups: 1
scene_count: 1
candidates_mean: 4.0
relevant_agents_mean: 3.0
valid_rollout_rate: 1.0
backend_distribution: {'reactive': 12}
policy_distribution: {'neutral_idm': 4}
branch_distribution: {'ambiguous': 7, 'maintain': 4, 'cede': 1}
burden_distribution: {'3': 12}
high_burden_cede_rate: 0.083333
fd_valid_rate: 1.0
fd_positive_rate: 0.083333
boundary_valid_rate: 1.0
boundary_positive_rate: 0.111111
main_training_ready: True
```

### Mini train/eval/calibration

```text
mini train: outputs/smoke_ckpt_final/scope.pt
held-out smoke examples: 12
branch_NLL: 1.567565390791245
branch_accuracy: 0.0
ordinal_burden_MAE: 1.6304266452789307
high_pressure_AUROC: 0.6363636363636364
FD_AUROC: 1.0
response_nll_proxy: 2474.159423828125
calibration branch_temperature: 0.7
calibration ordinal_temperature: 5.0
```

### Experiment smoke outputs

```text
surface boundary_AUROC: 1.0
surface FD_AUROC: 1.0
false-safe triples: 0 on tiny synthetic smoke data
closed-loop route_success_rate: 1.0
closed-loop fallback_rate: 1.0
closed-loop backend_distribution: {'reactive': 3}
```

## Important limitation

Synthetic/reactive smoke data proves code execution and tensor/metric correctness only. Main paper claims require a real WOMD/Waymax shard with `rollout.backend=waymax` and `rollout.fallback_to_reactive=false`, and sufficient FD/boundary positives in diagnostics.
