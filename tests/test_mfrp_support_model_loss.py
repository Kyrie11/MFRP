import torch
import numpy as np

from mfrp.data.support_query import split_support_query
from mfrp.models import MFRPModel
from mfrp.models.coercion_witness import MonotoneCoercionWitnessHead
from mfrp.training.losses import total_mfrp_loss, response_geometry_loss, distillation_loss

class C:
    def __init__(self, cid, fam):
        self.candidate_id = cid; self.family = fam; self.nominal_cost = 0.0


def make_batch(B=2,A=2,K=3,R=2,T=8,C=6):
    return {
        "candidate_features": torch.randn(B,A,K,20),
        "priority_score": torch.rand(B,A,K),
        "branch_probs": torch.softmax(torch.randn(B,A,K,R,C), -1),
        "trajectory": torch.randn(B,A,K,R,T,5),
        "trajectory_mask": torch.ones(B,A,K,R,T,dtype=torch.bool),
        "burden": torch.rand(B,A,K,R),
        "safety_margin": torch.randn(B,A,K,R),
        "variant_valid": torch.ones(B,A,K,R,dtype=torch.bool),
        "agent_candidate_valid": torch.ones(B,A,K,dtype=torch.bool),
        "cw_soft_label": torch.rand(B,A,K),
        "cw_confidence": torch.ones(B,A,K),
    }


def test_support_query_split_by_candidate():
    cands = [C("a","neutral"), C("b","assertive"), C("c","courtesy")]
    sp = split_support_query(cands, max_support_slots=2, prob_empty_support=0.0, rng=np.random.default_rng(0), rollout_variants=["r0","r1"])
    assert not (set(sp.support_candidate_ids) & set(sp.query_candidate_ids))
    for k, cid in enumerate([c.candidate_id for c in cands]):
        if cid in sp.query_candidate_ids:
            assert sp.query_probe_mask[k].all()
            assert not sp.support_probe_mask[k].any()


def test_model_output_shapes_and_loss():
    batch = make_batch(T=8)
    model = MFRPModel(future_steps=8, hidden_dim=32, mechanism_tokens=4, trajectory_modes=2)
    out = model(batch, mode="scene_only")
    y = out["scene_only"]
    assert y["branch_logits"].shape == (2,2,3,6)
    assert y["burden_loc"].shape == (2,2,3,6)
    assert y["margin_loc"].shape == (2,2,3,6)
    assert y["kappa"].shape == (2,2,3)
    loss = total_mfrp_loss(out, batch)["total"]
    assert torch.isfinite(loss)


def test_geometry_loss_uses_response_distance():
    batch = make_batch(T=8)
    batch["edge_index"] = torch.tensor([[[[0,1],[1,2]],[[0,1],[1,2]]],[[[0,1],[1,2]],[[0,1],[1,2]]]])
    batch["edge_valid"] = torch.ones(2,2,2,dtype=torch.bool)
    batch["response_distance"] = torch.ones(2,2,2)
    model = MFRPModel(future_steps=8, hidden_dim=32, mechanism_tokens=4, trajectory_modes=2)
    out = model(batch)["scene_only"]
    loss = response_geometry_loss(out, batch)
    assert torch.isfinite(loss)


def test_distillation_full_O():
    batch = make_batch(T=8)
    model = MFRPModel(future_steps=8, hidden_dim=32, mechanism_tokens=4, trajectory_modes=2)
    out = model(batch)
    d = distillation_loss(out["support_adapted"], out["scene_only"], batch["agent_candidate_valid"])
    assert torch.isfinite(d)


def test_witness_monotonicity_finite_difference():
    head = MonotoneCoercionWitnessHead(latent_dim=2, hidden_dim=4)
    with torch.no_grad():
        for p in head.flex.parameters():
            p.zero_()
    z = torch.zeros(4,2)
    base = head(torch.full((4,),0.2), torch.full((4,),0.1), torch.full((4,),0.5), torch.full((4,),0.5), torch.full((4,),0.7), torch.full((4,),0.6), z)[1]
    hi_pc = head(torch.full((4,),0.3), torch.full((4,),0.1), torch.full((4,),0.5), torch.full((4,),0.5), torch.full((4,),0.7), torch.full((4,),0.6), z)[1]
    hi_d = head(torch.full((4,),0.2), torch.full((4,),0.2), torch.full((4,),0.5), torch.full((4,),0.5), torch.full((4,),0.7), torch.full((4,),0.6), z)[1]
    hi_b = head(torch.full((4,),0.2), torch.full((4,),0.1), torch.full((4,),0.6), torch.full((4,),0.5), torch.full((4,),0.7), torch.full((4,),0.6), z)[1]
    hi_prio = head(torch.full((4,),0.2), torch.full((4,),0.1), torch.full((4,),0.5), torch.full((4,),0.7), torch.full((4,),0.7), torch.full((4,),0.6), z)[1]
    assert torch.all(hi_pc >= base)
    assert torch.all(hi_d >= base)
    assert torch.all(hi_b >= base)
    assert torch.all(hi_prio <= base)
