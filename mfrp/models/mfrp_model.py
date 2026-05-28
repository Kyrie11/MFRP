from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .coercion_witness import MonotoneCoercionWitness


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None, dim: int) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=dim)
    m = mask.to(dtype=x.dtype).unsqueeze(-1)
    return (x * m).sum(dim=dim) / m.sum(dim=dim).clamp_min(1.0)


class MFRPModel(nn.Module):
    """Scene-only and support-adapted mechanism token operator.

    Shapes:
      scene_features: [B,A,S]
      candidate_features: [B,A,K,F]
      support_probe_features: [B,A,P,PF]
    """

    def __init__(
        self,
        *,
        candidate_feature_dim: int = 20,
        scene_feature_dim: int = 32,
        probe_feature_dim: int = 35,
        hidden_dim: int = 128,
        mechanism_tokens: int = 8,
        branches: int = 6,
        future_steps: int = 80,
        state_dim: int = 5,
    ):
        super().__init__()
        self.candidate_feature_dim = candidate_feature_dim
        self.scene_feature_dim = scene_feature_dim
        self.probe_feature_dim = probe_feature_dim
        self.hidden_dim = hidden_dim
        self.mechanism_tokens = mechanism_tokens
        self.branches = branches
        self.future_steps = future_steps
        self.state_dim = state_dim
        self.scene_encoder = nn.Sequential(nn.Linear(scene_feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.candidate_encoder = nn.Sequential(nn.Linear(candidate_feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.probe_encoder = nn.Sequential(nn.Linear(probe_feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.slots = nn.Parameter(torch.randn(mechanism_tokens, hidden_dim) * 0.02)
        self.scene_token_proj = nn.Linear(hidden_dim, mechanism_tokens * hidden_dim)
        self.support_token_proj = nn.Linear(hidden_dim, mechanism_tokens * hidden_dim)
        self.query_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.response_fuse = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.branch_head = nn.Linear(hidden_dim, branches)
        self.burden_head = nn.Linear(hidden_dim, branches * 2)
        self.margin_head = nn.Linear(hidden_dim, branches * 2)
        self.traj_head = nn.Linear(hidden_dim, branches * future_steps * state_dim)
        self.witness = MonotoneCoercionWitness(hidden_dim, hidden_dim=max(32, hidden_dim // 2))

    def _tokens(self, batch: dict[str, torch.Tensor], *, support: bool) -> torch.Tensor:
        B, A = batch["scene_features"].shape[:2]
        scene = self.scene_encoder(batch["scene_features"])
        scene_tok = self.scene_token_proj(scene).view(B, A, self.mechanism_tokens, self.hidden_dim)
        if not support or "support_probe_features" not in batch:
            return scene_tok + self.slots.view(1, 1, self.mechanism_tokens, self.hidden_dim)
        probe = self.probe_encoder(batch["support_probe_features"])
        support_summary = _masked_mean(probe, batch.get("support_probe_mask"), dim=2)
        supp_tok = self.support_token_proj(support_summary).view(B, A, self.mechanism_tokens, self.hidden_dim)
        return scene_tok + supp_tok + self.slots.view(1, 1, self.mechanism_tokens, self.hidden_dim)

    def _decode(self, batch: dict[str, torch.Tensor], tokens: torch.Tensor, prefix: str) -> dict[str, torch.Tensor]:
        cand = self.candidate_encoder(batch["candidate_features"])
        B, A, K, H = cand.shape
        q = cand.reshape(B * A, K, H)
        kv = tokens.reshape(B * A, self.mechanism_tokens, H)
        ctx, _ = self.query_attn(q, kv, kv)
        h = self.response_fuse(torch.cat([q, ctx], dim=-1)).view(B, A, K, H)
        branch_logits = self.branch_head(h)
        branch_prob = F.softmax(branch_logits, dim=-1)
        burden_raw = self.burden_head(h).view(B, A, K, self.branches, 2)
        margin_raw = self.margin_head(h).view(B, A, K, self.branches, 2)
        traj = self.traj_head(h).view(B, A, K, self.branches, self.future_steps, self.state_dim)
        burden_mu = F.softplus(burden_raw[..., 0])
        burden_sigma = F.softplus(burden_raw[..., 1]) + 1e-3
        margin_mu = margin_raw[..., 0]
        margin_sigma = F.softplus(margin_raw[..., 1]) + 1e-3
        # Branch-conditioned summaries for κ.
        ceding = torch.zeros(self.branches, dtype=torch.bool, device=h.device)
        ceding[1] = True; ceding[2] = True
        p_c = branch_prob[..., ceding].sum(dim=-1)
        p_nc = (1.0 - p_c).clamp_min(1e-6)
        normal = torch.distributions.Normal(margin_mu, margin_sigma)
        s_branch = 1.0 - normal.cdf(torch.zeros((), device=h.device))
        s_c = (branch_prob[..., ceding] * s_branch[..., ceding]).sum(-1) / p_c.clamp_min(1e-6)
        s_nc = (branch_prob[..., ~ceding] * s_branch[..., ~ceding]).sum(-1) / p_nc
        b_c = (branch_prob[..., ceding] * burden_mu[..., ceding]).sum(-1) / p_c.clamp_min(1e-6)
        d_c = (s_c - s_nc).clamp_min(0.0)
        priority = batch.get("priority_score_preexec", torch.full_like(p_c, 0.5))
        witness_features = torch.stack([p_c, s_c, s_nc, d_c, b_c, priority], dim=-1)
        kappa = self.witness(witness_features, h)
        return {
            f"{prefix}_tokens": tokens,
            f"{prefix}_latent": h,
            f"{prefix}_branch_logits": branch_logits,
            f"{prefix}_branch_prob": branch_prob,
            f"{prefix}_trajectory_mu": traj,
            f"{prefix}_burden_mu": burden_mu,
            f"{prefix}_burden_sigma": burden_sigma,
            f"{prefix}_margin_mu": margin_mu,
            f"{prefix}_margin_sigma": margin_sigma,
            f"{prefix}_P_C": p_c,
            f"{prefix}_S_C": s_c,
            f"{prefix}_S_notC": s_nc,
            f"{prefix}_B_C": b_c,
            f"{prefix}_D_C": d_c,
            f"{prefix}_kappa": kappa,
        }

    def forward(self, batch: dict[str, torch.Tensor], mode: str = "both") -> dict[str, torch.Tensor]:
        if mode not in {"both", "scene_only", "support_adapted"}:
            raise ValueError("mode must be both, scene_only, or support_adapted")
        out: dict[str, torch.Tensor] = {}
        if mode in {"both", "scene_only"}:
            out.update(self._decode(batch, self._tokens(batch, support=False), "scene"))
        if mode in {"both", "support_adapted"}:
            out.update(self._decode(batch, self._tokens(batch, support=True), "support"))
        return out
