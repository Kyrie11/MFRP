import numpy as np

from mfrp.data.schema import RootScene, EgoCandidate, ResponseObservation, SameRootGroup, BoundaryPair, AgentTrackTensor
from mfrp.data.tensors import collate_same_root_groups
from mfrp.data.priority import compute_priority_score
from mfrp.data.label_extraction import coercion_witness_label
from examples.mfrp_waymax_adapter import _reactive_rollout_for_agent, VariantSpec


def _obs(cid, aid="1", rid="r0", branch=None, burden=0.0, margin=1.0, prio=0.5, prio_conf=1.0):
    bp = np.asarray(branch if branch is not None else [1, 0, 0, 0, 0, 0], dtype=np.float32)
    traj = np.zeros((5, 10), dtype=np.float32)
    traj[:, 5] = 2.0
    traj[:, 7:9] = [4.5, 2.0]
    traj[:, 9] = 1.0
    return ResponseObservation("s", "root", cid, aid, rid, bp, -1, traj, np.ones(5, bool), burden, margin, 1.0, burden > 1, prio, prio_conf)


def test_collate_preserves_candidate_features_and_adds_context_and_boundary_dataclass():
    hist = np.zeros((2, 3, 5), dtype=np.float32)
    mask = np.ones((2, 3), dtype=bool)
    root = RootScene(
        "s", 2, hist, mask,
        map_features=np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 1.0]], dtype=np.float32),
        traffic_controls=np.array([1, 3, 4], dtype=np.float32),
        route_features=np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )
    f0 = np.zeros(20, dtype=np.float32); f0[:8] = np.arange(1, 9)
    f1 = np.zeros(20, dtype=np.float32); f1[:8] = np.arange(11, 19)
    meta0 = {"agent_features": {"1": {"interaction_features": np.array([0.0, 2.0, -1.0], dtype=np.float32)}}, "agent_feature_offset": 8}
    c0 = EgoCandidate("c0", np.zeros((5, 10), dtype=np.float32), f0, metadata=meta0)
    c1 = EgoCandidate("c1", np.zeros((5, 10), dtype=np.float32), f1)
    obs = {}
    for c in [c0, c1]:
        for r in ["r0", "r1"]:
            obs[(c.candidate_id, "1", r)] = _obs(c.candidate_id, rid=r)
    g = SameRootGroup(root, [c0, c1], ["1"], ["r0", "r1"], obs, metadata={"support_candidate_ids": ["c0"], "query_candidate_ids": ["c1"]}, boundary_pairs=[BoundaryPair("1", "c0", "c1", 0.7)])
    batch = collate_same_root_groups([g], future_steps=5)
    np.testing.assert_allclose(batch["candidate_features"][0, 0, 0, :8], np.arange(1, 9))
    np.testing.assert_allclose(batch["candidate_features"][0, 0, 0, 8:11], [0.0, 2.0, -1.0])
    assert np.any(batch["scene_features"][0, 0, 7:] != 0)
    assert batch["edge_a"].shape[0] == 1


def test_priority_missing_structural_context_caps_confidence():
    pr = compute_priority_score({"entry_time_gap": 5.0})
    assert pr.confidence <= 0.45
    assert pr.diagnostics["reason"] == "missing_route_and_traffic_control_context"


def test_cw_label_is_attenuated_by_ego_priority():
    cede = [0, 1, 0, 0, 0, 0]
    keep = [1, 0, 0, 0, 0, 0]
    high_prio_obs = [_obs("c", "1", "r0", cede, burden=2.0, margin=1.0, prio=0.9), _obs("c", "1", "r1", keep, burden=0.0, margin=-1.0, prio=0.9)]
    low_prio_obs = [_obs("c", "1", "r0", cede, burden=2.0, margin=1.0, prio=0.1), _obs("c", "1", "r1", keep, burden=0.0, margin=-1.0, prio=0.1)]
    assert coercion_witness_label(low_prio_obs, "c", "1").soft_label > coercion_witness_label(high_prio_obs, "c", "1").soft_label


def test_nonvehicle_agent_uses_constant_velocity_not_vehicle_idm():
    hist = np.zeros((2, 3, 10), dtype=np.float32)
    hist[:, :, 7:9] = [4.5, 2.0]
    hist[:, :, 9] = 1.0
    hist[1, :, 0] = [0.0, 0.1, 0.2]
    hist[1, :, 3] = 1.0
    hist[1, :, 5] = 1.0
    mask = np.ones((2, 3), dtype=bool)
    root = RootScene("s", 2, hist[..., [0, 1, 3, 4, 6]], mask, agent_tracks=[AgentTrackTensor("ego", hist[0], mask[0], {}), AgentTrackTensor("ped", hist[1], mask[1], {"object_type": "2"})])
    route = np.zeros((10, 10), dtype=np.float32); route[:, 0] = np.linspace(0, 20, 10); route[:, 5] = 8.0; route[:, 7:9] = [4.5, 2.0]; route[:, 9] = 1.0
    ego = route.copy(); ego[:, 0] = np.linspace(0, 5, 10)
    out = _reactive_rollout_for_agent(root, "1", route, ego, VariantSpec("neutral"))
    assert np.allclose(np.diff(out[:, 0]), np.diff(out[:, 0])[0], atol=1e-5)
    assert out[-1, 0] < 2.0  # did not jump to the logged vehicle route at x=20
