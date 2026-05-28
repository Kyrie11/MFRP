"""Main MFRP model."""
from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.distributions import Normal

from mfrp.data.schema import BRANCHES, CEDING_BRANCH_IDS
from mfrp.models.mechanism_operator import MechanismTokenOperator
from mfrp.models.response_heads import ResponseHeads
from mfrp.models.coercion_witness import MonotoneCoercionWitnessHead


class MFRPModel(nn.Module):
    def __init__(self, candidate_feature_dim: int = 20, hidden_dim: int = 256, mechanism_tokens: int = 16, trajectory_modes: int = 6, future_steps: int = 80, traj_dim: int = 5, dropout: float = 0.1):
        super().__init__()
        self.operator = MechanismTokenOperator(candidate_feature_dim, hidden_dim, mechanism_tokens)
        self.heads = ResponseHeads(hidden_dim, len(BRANCHES), trajectory_modes, future_steps, traj_dim, dropout)
        self.witness = MonotoneCoercionWitnessHead(hidden_dim)

    def _complete_outputs(self, r: torch.Tensor, tokens: torch.Tensor, batch: dict) -> dict:
        out = self.heads(r)
        branch_probs = out["branch_probs"]
        margin_scale = torch.exp(out["margin_log_scale"]).clamp_min(1e-5)
        burden_loc = out["burden_loc"]
        margin_loc = out["margin_loc"]
        c_ids = torch.as_tensor(CEDING_BRANCH_IDS, device=r.device, dtype=torch.long)
        c_mask = torch.zeros(len(BRANCHES), device=r.device, dtype=torch.bool)
        c_mask[c_ids] = True
        P_C = branch_probs[..., c_mask].sum(dim=-1).clamp_min(1e-6)
        P_notC = branch_probs[..., ~c_mask].sum(dim=-1).clamp_min(1e-6)
        normal = Normal(margin_loc, margin_scale)
        safe_prob = 1.0 - normal.cdf(torch.zeros((), device=r.device))
        S_C = (branch_probs[..., c_mask] * safe_prob[..., c_mask]).sum(dim=-1) / P_C
        S_notC = (branch_probs[..., ~c_mask] * safe_prob[..., ~c_mask]).sum(dim=-1) / P_notC
        B_C = (branch_probs[..., c_mask] * burden_loc[..., c_mask]).sum(dim=-1) / P_C
        D_C = torch.relu(S_C - S_notC)
        priority = batch.get("priority_score_preexec")
        if priority is None:
            # Backward-compatible smoke-test fallback only: infer the pre-execution
            # priority that intervention_coordinate stores in candidate_features.
            # Do not read legacy label-side priority_score here.
            cand = batch.get("candidate_features")
            if cand is not None and cand.shape[-1] >= 6:
                priority = cand[..., 4].to(r.device).float().clamp(0.0, 1.0)
            else:
                priority = torch.full_like(P_C, 0.5)
        else:
            priority = priority.to(r.device).float()
        k_logits, kappa = self.witness(P_C, D_C, B_C, priority, S_C, S_notC, r)
        out.update({
            "query_embedding": r,
            "mechanism_tokens": tokens,
            "witness_features": torch.stack([P_C, S_C, S_notC, D_C, B_C, priority], dim=-1),
            "kappa_logits": k_logits,
            "kappa": kappa,
        })
        return out

    def forward(self, batch: dict, mode: Literal["scene_only", "support_adapted", "both"] = "both") -> dict:
        x = batch["candidate_features"].float()
        scene_tokens = self.operator.scene_only_tokens(x, batch.get("scene_features"))
        scene_r, scene_t = self.operator.decode_queries(x, scene_tokens)
        scene_out = self._complete_outputs(scene_r, scene_t, batch)
        if mode == "scene_only":
            return {"scene_only": scene_out, "support_adapted": None, "distill_teacher": None}
        support_out = None
        if "support_probe_features" in batch:
            support_tokens = self.operator.support_tokens(scene_tokens, batch["support_probe_features"].float(), batch.get("support_probe_mask"))
            support_r, support_t = self.operator.decode_queries(x, support_tokens)
            support_out = self._complete_outputs(support_r, support_t, batch)
        elif mode in ("support_adapted", "both"):
            support_out = scene_out
        return {"scene_only": scene_out, "support_adapted": support_out, "distill_teacher": support_out}
