import numpy as np

from mfrp.data.label_extraction import _speed
from mfrp.data.interaction_region import closest_entry_times
from examples.mfrp_waymax_adapter import _perturb_candidate
from mfrp.data.scene_schema import EgoCandidate


def _state_line(n=20, speed=10.0):
    s = np.zeros((n, 10), dtype=np.float32)
    s[:, 0] = np.arange(n, dtype=np.float32) * speed * 0.1
    s[:, 3] = speed
    s[:, 5] = speed
    s[:, 7] = 4.5
    s[:, 8] = 2.0
    s[:, 9] = 1.0
    return s


def test_speed_uses_speed_channel_not_valid_mask():
    traj = _state_line(5, speed=8.0)
    traj[:, 5] = np.array([8, 7, 6, 5, 4], dtype=np.float32)
    traj[:, 9] = 1.0
    np.testing.assert_allclose(_speed(traj, 0.1), [8, 7, 6, 5, 4])


def test_speed_scaling_reintegrates_positions():
    base = EgoCandidate("base", _state_line(30, speed=10.0))
    slow = _perturb_candidate(base, "slow", 0.75, 0.0, 0.0, 0.0, 0.1)
    fast = _perturb_candidate(base, "fast", 1.25, 0.0, 0.0, 0.0, 0.1)
    assert fast.trajectory[-1, 0] > slow.trajectory[-1, 0] + 1.0
    assert fast.trajectory[-1, 5] > slow.trajectory[-1, 5]


def test_conflict_region_has_distinct_entry_times_when_arrivals_differ():
    ego = np.zeros((20, 10), dtype=np.float32)
    agent = np.zeros((20, 10), dtype=np.float32)
    ego[:, 0] = np.linspace(0, 10, 20)
    agent[:, 0] = 5.0
    agent[:, 1] = np.linspace(-10, 10, 20)
    ego[:, 9] = 1.0
    agent[:, 9] = 1.0
    tau_e, tau_a, _ = closest_entry_times(ego, agent, threshold=1.0, dt=0.1)
    assert np.isfinite(tau_e) and np.isfinite(tau_a)
    assert abs(tau_e - tau_a) > 0.05
