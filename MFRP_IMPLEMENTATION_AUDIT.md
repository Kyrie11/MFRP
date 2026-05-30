# MFRP implementation audit and fixes

This audit compares the implementation against `interactive_planning.tex`, especially the abstract, introduction, method, and appendix.

## Alignment now implemented

1. **Same-root group abstraction.** `SameRootGroup` keeps one root scene with multiple ego candidates, relevant agents, rollout variants, observations, support/query candidate ids, and geometry edges.
2. **No default logged-ego-future candidates.** The adapter defaults to observed-root kinematic primitives. Logged-future anchors are debug/oracle-only through `dataset.allow_logged_future_anchor: true`.
3. **Physically meaningful speed/timing perturbations.** Ego candidates reintegrate positions when speed scale or delay changes.
4. **Separate conflict-entry timing.** `build_interaction_region` estimates a shared spatial conflict region and computes ego/agent entry times separately.
5. **Same-variant neutral baseline.** Burden compares each candidate response to the neutral-candidate response under the same agent and policy variant.
6. **Deployment-safe priority.** Priority uses pre-execution features only; missing route/traffic metadata lowers confidence.
7. **Scene-only deployment.** `scene_only_inference` strips support probes and label-side tensors. `deploy_mfrp.py` runs the selector and returns selected index, fallback flag, feasible mask, and calibrated mechanism risk.
8. **Support/query separation.** Query candidates do not appear in support probes. Splitting is deterministic and stratified across candidate families.
9. **Map/traffic/route context path.** The adapter extracts available root-scene map, traffic-control, and route metadata; the tensorizer summarizes them into `scene_features`.
10. **Coercion target consistency.** Calibration/evaluation truth includes both unsafe signed margins and coercion witness labels.
11. **Regression coverage.** The test suite covers the audit blockers: speed channel, candidate reintegration, conflict timing, priority uncertainty, agent mapping/context features, non-vehicle rollout fallback, and importability.

## Still a proxy relative to the full paper

- The bundled adapter does **not** execute a full Waymax environment reset/step loop with simultaneous learned agents and Waymax metrics. It uses WOMD-loaded root states plus a route-following IDM proxy for vehicle agents.
- Logged future geometry is still used as a route/polyline source for the proxy response model. It is not used as deployment model input, not used as priority input, and not used as log-playback response supervision.
- Full vector-token map/lane/traffic-control encoding is approximated by compact scene summaries.
- Candidate feasibility uses kinematic and roadgraph-distance checks; true Waymax overlap/offroad/kinematic metrics require the full closed-loop backend.
- Scene-level risk aggregation remains the noisy-or approximation unless another aggregation mode is selected.

These remaining points should be described as simulator-proxy limitations in the paper or replaced with a full Waymax closed-loop backend before reporting strong closed-loop claims.
