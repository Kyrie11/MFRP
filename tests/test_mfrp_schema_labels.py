import numpy as np

from mfrp.data.schema import BRANCHES, ResponseObservation
from mfrp.data.label_extraction import baseline_relative_burden, signed_oriented_box_separation, coercion_witness_label
from mfrp.data.priority import compute_priority_score


def test_branch_vocab_exact():
    assert BRANCHES == ["keep", "cede", "brake", "accelerate", "pass", "nonconflict"]
    assert not ({"ambiguous", "unaffected", "follow", "maintain"} & set(BRANCHES))


def test_response_observation_fields():
    obs = ResponseObservation("s", "root", "c", 1, "r", np.ones(6), -1, np.zeros((3,5)), np.ones(3,bool), 0.0, 0.0, 2.0, False, 0.5, 1.0)
    assert np.isclose(obs.branch_probs.sum(), 1.0)
    assert obs.branch_hard >= 0
    assert obs.trajectory.shape[-1] == 5


def test_burden_baseline_relative_zero():
    traj = np.zeros((5,10), dtype=np.float32); traj[:,9] = 1; traj[:,5] = 3
    b, _ = baseline_relative_burden(traj, traj.copy(), 2.0, 2.0)
    assert b == 0.0


def test_signed_margin_positive_negative():
    a = np.array([0,0,0,0,0,0,0,4,2,1], dtype=np.float32)
    b = np.array([10,0,0,0,0,0,0,4,2,1], dtype=np.float32)
    c = np.array([0.5,0,0,0,0,0,0,4,2,1], dtype=np.float32)
    assert signed_oriented_box_separation(a,b) > 0
    assert signed_oriented_box_separation(a,c) < 0


def test_priority_missing_uncertain():
    pr = compute_priority_score({})
    assert 0.0 <= pr.score <= 1.0
    assert pr.confidence < 1.0


def test_cw_low_confidence_single_variant():
    obs = ResponseObservation("s", "root", "c", 1, "only", np.array([0,1,0,0,0,0.]), 1, np.zeros((3,5)), np.ones(3,bool), 2.0, 1.0, 1.0, False, 0.2, 1.0)
    lab = coercion_witness_label([obs], "c", 1, "root", "s")
    assert lab.confidence < 0.5
