"""Learned monotone coercion witness for MFRP."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MonotoneCoercionWitnessHead(nn.Module):
    """Monotone in P_C, D_C, B_C and anti-monotone in priority.

    The flexible residual intentionally receives only S_C, S_notC and latent
    features so it cannot directly undo the required monotonicity constraints.
    """
    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.w_pos_raw = nn.Parameter(torch.zeros(3))
        self.w_prio_raw = nn.Parameter(torch.zeros(1))
        self.flex = nn.Sequential(nn.Linear(latent_dim + 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, P_C: torch.Tensor, D_C: torch.Tensor, B_C: torch.Tensor, priority: torch.Tensor, S_C: torch.Tensor, S_notC: torch.Tensor, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w_pos = F.softplus(self.w_pos_raw)
        w_prio = F.softplus(self.w_prio_raw)
        mono = w_pos[0] * P_C + w_pos[1] * D_C + w_pos[2] * B_C - w_prio * priority
        flex_in = torch.cat([S_C.unsqueeze(-1), S_notC.unsqueeze(-1), latent], dim=-1)
        logits = self.bias + mono + self.flex(flex_in).squeeze(-1)
        return logits, torch.sigmoid(logits)
