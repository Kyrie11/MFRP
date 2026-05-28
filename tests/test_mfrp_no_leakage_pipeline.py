import numpy as np
import pytest
import torch

from mfrp.models import MFRPModel
from mfrp.planning.calibration import fit_split_calibration
from mfrp.planning.deploy import scene_only_inference


class PriorityProbeModel(MFRPModel):
    def forward(self, batch, mode="both"):
        assert "priority_score" not in batch
        assert "branch_probs" not in batch
        assert "support_probe_features" not in batch
        assert "priority_score_preexec" in batch
        return super().forward(batch, mode=mode)


def test_deployment_strips_legacy_priority_and_labels_but_keeps_preexec_priority():
    model = PriorityProbeModel(future_steps=4, hidden_dim=16, mechanism_tokens=2, trajectory_modes=2)
    batch = {
        "candidate_features": torch.randn(1, 1, 2, 20),
        "priority_score_preexec": torch.rand(1, 1, 2),
        "priority_score": torch.ones(1, 1, 2),
        "support_probe_features": torch.randn(1, 1, 3, 34),
        "support_probe_mask": torch.ones(1, 1, 3, dtype=torch.bool),
        "branch_probs": torch.ones(1, 1, 2, 1, 6) / 6,
    }
    out = scene_only_inference(model, batch)
    assert out["branch_logits"].shape == (1, 1, 2, 6)


def test_empty_calibration_is_rejected():
    with pytest.raises(ValueError):
        fit_split_calibration(np.asarray([]), np.asarray([]), beta=0.1)
