# MFRP implementation audit and fixes

This audit compares the code against the paper algorithm in `interactive planning (17).tex`, especially the abstract, introduction, method, and appendix.

## Main alignment fixes applied

1. **Deployment feature leakage fixed.** `mfrp.data.tensors._candidate_features` no longer copies `priority_score`/`priority_confidence` from `ResponseObservation`. Those observations are label-side products of rollout extraction. Candidate features now compute a pre-execution priority score from interaction features and candidate/root metadata only.
2. **Observed-only neutral reference added.** `mfrp.data.interaction_region.constant_velocity_extrapolate` creates a deployment-safe neutral agent reference from observed history. This avoids using WOMD future labels or rollout futures to construct candidate coordinates.
3. **Training objective aligned with support/query operator learning.** `total_mfrp_loss` now makes the support-adapted query likelihood the primary mechanism NLL when support probes are available. Scene-only NLL is optional; scene-only deployment is inherited through distillation, matching the paper’s support-to-scene mechanism-token design.
4. **Scene-only deployment sanitization strengthened.** `mfrp.planning.deploy.scene_only_inference` now strips support probes, query masks, rollout validity, branch/trajectory/burden/margin labels, HP/CW labels, diagnostics, and raw groups before model inference.
5. **Real NPZ training path added.** `scripts/train_mfrp.py` now trains from materialized NPZ shards instead of only printing a placeholder message. BUILD_SPEC-only directories still fail as intended.
6. **Candidate feature dimension exposed.** `configs/model/mfrp_full.yaml` now includes `candidate_feature_dim: 24`, so full runs do not silently truncate the intervention coordinate as aggressively as the previous 20-dimensional default.

## Remaining simplifications relative to the paper

- The bundled `scripts/build_same_root_dataset.py` is still a fail-fast integration hook, not a full Waymax materializer. This is intentional because a correct materializer must reset Waymax to the same root state, execute ego interventions, run multiple reactive policy variants, and emit rollout-derived response primitives. Silently fabricating probes would invalidate the paper.
- The scene encoder is still lightweight compared with the paper: it summarizes tracks/scene vectors rather than encoding full vectorized maps, lane boundaries, crosswalks, route paths, and traffic controls with attention tokens.
- The support encoder is an MLP/set-pooling approximation rather than a full Set Transformer/slot cross-attention stack.
- The trajectory head uses a branch-conditioned Gaussian mixture proxy; burden and margin use Gaussian scalar heads. This is allowed by the appendix, but it is not a full distributional implementation with discretized logistic or calibrated quantile alternatives.
- Boundary/surface geometry supervision exists, but edge generation must be supplied by the materialized dataset.
- Closed-loop evaluation and Waymax metrics are not fully implemented in `scripts/eval_mfrp.py`; the repository now supports training from proper shards but still needs a real evaluation harness for paper numbers.

## Dataset-generation assessment

A valid paper dataset must be generated from same-root counterfactual rollouts, not from raw WOMD futures alone. WOMD TFExamples are useful for root scenes, maps, traffic controls, route context, and logged trajectories, while Waymax can reset scenes and simulate interventions. However, forced-dependence labels require response diversity: at least ceding and non-ceding variants near the same candidate/agent. A single deterministic IDM rollout, and especially log playback, is insufficient for the forced-dependence witness.

The current README and diagnostics enforce these conditions: materialized rollout variants, support/query separation by candidate, non-trivial branch diversity, both ceding and non-ceding evidence for witness labels, finite burden/margin values, priority-confidence coverage, and geometry edges.

## Deployment-input realism

Deployment-safe inputs are observed history, map/traffic-control/route context available before execution, and the ego candidate. The following are not realistic deployment inputs and must remain labels/diagnostics only: realized future trajectories, rollout response branch, burden, signed safety margin, HP/CW labels, support/query probes, future traffic-light states beyond the observed root, and any priority value inferred from the realized response. Missing traffic-control or route metadata should produce uncertain priority with lower confidence, not a default low-priority/coercive label.
