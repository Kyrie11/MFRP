import torch
from mfrp.models import MFRPModel
from mfrp.planning.deploy import scene_only_inference


def test_deploy_scene_only_no_support():
    model=MFRPModel(future_steps=4, hidden_dim=16, mechanism_tokens=2, trajectory_modes=2)
    batch={"candidate_features":torch.randn(1,1,2,20),"priority_score":torch.rand(1,1,2),"support_probe_features":torch.randn(1,1,3,34),"support_probe_mask":torch.ones(1,1,3,dtype=torch.bool),"branch_probs":torch.ones(1,1,2,1,6)/6,"trajectory":torch.randn(1,1,2,1,4,5),"burden":torch.zeros(1,1,2,1),"safety_margin":torch.ones(1,1,2,1)}
    out=scene_only_inference(model,batch)
    assert out['branch_logits'].shape == (1,1,2,6)
