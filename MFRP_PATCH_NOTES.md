# MFRP audit-driven patch notes

This patch implements the actionable fixes requested from `modifycode.md` and the paper/code consistency review.

## Fixed P0/P1/P2 implementation defects

- `mfrp/data/label_extraction.py`
  - Fixed `_speed()` to read channel 5 (`speed`) instead of channel 9 (`valid`).
  - Added invalid-timestep masking for speed, acceleration, jerk, branch, burden, high-pressure, and signed-margin calculations.
  - Propagated trajectory masks into `ResponseObservation`.
  - Attenuated coercion-witness labels by ego priority and lowered confidence when priority evidence is weak.

- `mfrp/data/interaction_region.py`
  - Replaced synchronized closest-distance timing with a shared spatial conflict-region estimate.
  - Ego and agent now receive separate `tau_in` estimates, so entry-time gap is no longer forced to zero.

- `mfrp/data/priority.py`
  - Made priority explicitly deployment-safe.
  - Missing route/traffic-control/right-of-way context now caps confidence instead of becoming false low-priority evidence.

- `examples/mfrp_waymax_adapter.py`
  - Replaced default logged-SDC-future candidate generation with observed-root kinematic primitives.
  - Speed/timing perturbations now reintegrate positions.
  - Kept logged-future candidate only behind `dataset.allow_logged_future_anchor: true`.
  - Changed burden baseline to the same `(agent, variant)` neutral-candidate rollout.
  - Changed priority features to observed-history constant-velocity extrapolation rather than logged agent future.
  - Added map summary, traffic-control summary, and route metadata extraction when available from the root state.
  - Added candidate feasibility checks for finite state, speed, acceleration, jerk, and an optional roadgraph-distance proxy.
  - Non-vehicle agents no longer use the vehicle IDM response model; they use observed-history constant-velocity fallback.
  - Replaced biased tail query split with deterministic stratified candidate split.
  - Replaced adjacent-only geometry edges with local/global intervention pairs.
  - Added deterministic validation/test subpartitioning by scenario id when both are built from official validation TFRecords.
  - Updated metadata to state that the bundled backend is a WOMD-route IDM proxy, not a full Waymax closed-loop simulator.

- `mfrp/data/tensors.py`
  - Fixed selected-agent mapping to use actual track index / agent id.
  - Preserved zero-valued interaction features instead of overwriting only nonzero values.
  - Added deployment-safe map/traffic/route context summaries into `scene_features`.
  - Added support for `BoundaryPair` dataclass objects during collation.

- `mfrp/models/mfrp_model.py`
  - Uses `priority_confidence_preexec` to shrink uncertain priority toward an uninformative prior before witness scoring.

- `mfrp/data/schema.py` / `mfrp/models/response_heads.py`
  - Added `TRAJ_TARGET_DIM = 5` so `response_heads.py` imports cleanly.

- `scripts/calibrate_mfrp.py`
  - Calibration truth now includes unsafe margins and coercion witness labels.
  - Added optional `--selected-action` calibration to estimate selector-level residuals.

- `scripts/deploy_mfrp.py`
  - Deployment now runs scene-only inference and the mechanism-feasible selector.
  - Uses selected-action calibration quantile when available.

- Configs/docs
  - Fixed `gamma_bar` to `0.60` where model `gamma` is in `[0, 1]`.
  - Enabled lateral offsets in the data config.
  - Fixed recommended val/test commands so they do not point to training TFRecords.
  - Updated docs to avoid claiming full Waymax closed-loop rollout where the code currently uses the IDM proxy.

## Added regression tests

- `test_speed_uses_speed_channel_not_valid_mask`
- `test_speed_scaling_reintegrates_positions`
- `test_conflict_region_has_distinct_entry_times_when_arrivals_differ`
- `test_collate_preserves_candidate_features_and_adds_context_and_boundary_dataclass`
- `test_priority_missing_structural_context_caps_confidence`
- `test_cw_label_is_attenuated_by_ego_priority`
- `test_nonvehicle_agent_uses_constant_velocity_not_vehicle_idm`

## Verification

Commands run from repo root:

```bash
python -m py_compile $(find mfrp scripts examples tests -name '*.py')
python -m scripts.train_mfrp --smoke --max_steps 2 --out /mnt/data/MFRP_work/smoke.pt --device cpu
python -m pytest -q
pytest -q
```

Result:

```text
26 passed
```

## Remaining limitation

The bundled adapter is a safer, non-leaky WOMD-route IDM proxy. A true simultaneous multi-agent Waymax environment reset/step closed-loop backend is still a larger integration task and is not falsely claimed as implemented.
