import numpy as np
import torch
from mfrp.planning.calibration import fit_split_calibration, apply_calibration
from mfrp.planning.estimators import p_unsafe_from_margin, scene_mechanism_risk
from mfrp.planning.selector import select_mechanism_feasible


def test_p_unsafe_from_margin_distribution():
    out={"branch_probs": torch.ones(1,1,2,6)/6, "margin_loc": torch.tensor([[[[1.,1,1,1,1,1],[-1,-1,-1,-1,-1,-1]]]]), "margin_log_scale": torch.zeros(1,1,2,6)}
    p=p_unsafe_from_margin(out)
    assert p[0,0,0] < 0.5
    assert p[0,0,1] > 0.5


def test_rho_mech_noisy_or_agents():
    rho=scene_mechanism_risk({"p_viol": torch.tensor([[[0.1,0.2],[0.0,0.5]]])})
    assert torch.allclose(rho, torch.tensor([[0.1,0.6]]), atol=1e-5)


def test_split_calibration_quantile():
    art=fit_split_calibration(np.array([0.1,0.2,0.8]), np.array([1,0,1]), beta=0.34)
    cal=apply_calibration(np.array([0.2]), art)
    assert cal[0] >= 0.2


def test_selector_uses_calibrated_risk_and_fallback_marks():
    class C:
        def __init__(self, cid, cost): self.candidate_id=cid; self.nominal_cost=cost
    res=select_mechanism_feasible([C('a',0),C('b',1)], np.array([0.9,0.8]), np.array([1,1]), np.array([3,3]), alpha=0.05, nu_bar=0.3, gamma_bar=2.0)
    assert res.fallback_used
    assert res.active_violations['risk']
