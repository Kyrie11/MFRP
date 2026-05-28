# MFRP optimization report

## Main theoretical issue
The paper idea is coherent, but its validity depends on a data condition that is easy to violate: forced dependence cannot be identified from a single deterministic reactive rollout. The dataset must include both ceding and non-ceding response evidence for the same root/candidate/agent cell, or the CW label must be low-confidence and excluded from witness/ranking supervision.

## Main code/data risks fixed
- Replaced placeholder/fake-data path with a strict adapter contract.
- Enforced `module:function` adapter syntax.
- Enforced candidate-level support/query split.
- Enforced `priority_score_preexec` and deployment sanitizer to avoid label-side priority leakage.
- Added support-adapted and scene-only tokens in the same model, with distillation.
- Added monotone coercion witness head.
- Added branch-conditioned burden/margin and mechanism-risk estimators.
- Added split calibration and mechanism-feasible selector.
- Added dataset diagnostics for rollout validity, ceding/non-ceding diversity, support/query masks, priority leakage, debug-only shards, and duplicate roots.
- Added smoke tests and compile checks.

## Adapter recommendation
Use `--adapter my_project.mfrp_waymax_adapter:build_groups`. Do not set `adapter` to `waymax`, a path to YAML, a class name, or the template module. The function must load real WOMD/Waymax data and return `SameRootGroup` objects.

## Remaining external requirement
The zip cannot implement your private WOMD/Waymax scenario loader because those files and local simulator installation are not available here. The included adapter template documents exactly where to connect it.
