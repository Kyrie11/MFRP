from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MonotoneCoercionWitness(nn.Module):
    """κ head monotone in ceding probability/dependence/burden, anti-monotone in priority."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.flex = nn.Sequential(nn.Linear(latent_dim + 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.raw_pos = nn.Parameter(torch.zeros(3))  # P_C, D_C, B_C
        self.raw_priority = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.tensor(-2.0))

    def forward(self, features: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        p_c = features[..., 0]
        s_c = features[..., 1]
        s_nc = features[..., 2]
        d_c = features[..., 3]
        b_c = features[..., 4]
        priority = features[..., 5]
        pos_w = F.softplus(self.raw_pos)
        prio_w = F.softplus(self.raw_priority)
        mono = pos_w[0] * p_c + pos_w[1] * d_c + pos_w[2] * b_c - prio_w * priority
        free_in = torch.cat([torch.stack([s_c, s_nc], -1), latent], dim=-1)
        return torch.sigmoid(mono + self.flex(free_in).squeeze(-1) + self.bias)


class MonotoneCoercionWitnessHead(MonotoneCoercionWitness):
    """Legacy-compatible head. Returns (logit, probability)."""

    def forward(self, p_c, d_c, b_c, priority, s_c, s_nc, latent):  # type: ignore[override]
        features = torch.stack([p_c, s_c, s_nc, d_c, b_c, priority], dim=-1)
        prob = super().forward(features, latent)
        logit = torch.logit(prob.clamp(1e-6, 1 - 1e-6))
        return logit, prob
