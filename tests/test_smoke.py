from __future__ import annotations

import numpy as np
import torch
from mfrp.data.schema import RootScene, EgoCandidate, ResponseObservation, SameRootGroup
from mfrp.data.support_query import split_support_query
from mfrp.data.tensors import collate_same_root_groups
from mfrp.models import MFRPModel
from mfrp.training import total_mfrp_loss
from mfrp.planning import select_mechanism_feasible


def make_group():
    hist = np.zeros((3, 5, 5), dtype=np.float32)
    mask = np.ones((3, 5), dtype=bool)
    root = RootScene("s0", 10, hist, mask)
    cands = [EgoCandidate(f"u{k}", np.zeros((8,5), dtype=np.float32), np.random.randn(20).astype(np.float32), nominal_cost=float(k)) for k in range(4)]
    support, query = split_support_query([c.candidate_id for c in cands], seed=1)
    obs = {}
    variants = ["neutral", "assertive", "conservative"]
    for c in cands:
        for a in ["agent1", "agent2"]:
            for ri, r in enumerate(variants):
                bp = np.zeros(6, dtype=np.float32); bp[(ri + int(c.candidate_id[-1])) % 6] = 1
                obs[(c.candidate_id, a, r)] = ResponseObservation(c.candidate_id, a, r, bp, np.zeros((8,5), dtype=np.float32), np.ones(8, dtype=bool), burden=float(ri)/2, safety_margin=float(ri-1), high_pressure=ri==2, cw_soft_label=0.2*ri, cw_confidence=1.0, priority_score_preexec=0.5, priority_confidence_preexec=1.0)
    return SameRootGroup(root, cands, ["agent1","agent2"], variants, obs, metadata={"support_candidate_ids": support, "query_candidate_ids": query}, boundary_pairs=[("agent1","u0","u1",.5)])


def test_collate_model_loss_selector():
    batch_np = collate_same_root_groups([make_group()], future_steps=8)
    batch = {k: torch.from_numpy(v).float() if v.dtype.kind == 'f' else torch.from_numpy(v) for k,v in batch_np.items() if getattr(v, 'dtype', None) != object}
    model = MFRPModel(future_steps=8)
    out = model(batch)
    losses = total_mfrp_loss(out, batch, {"loss": {}})
    assert torch.isfinite(losses["total"])
    sel = select_mechanism_feasible(out, batch, alpha=1.0, nu_bar=2.0)
    assert sel["selected_index"].shape == (1,)
