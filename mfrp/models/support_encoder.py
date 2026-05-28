"""Support-probe encoder for MFRP mechanism tokens."""
from __future__ import annotations

import torch
from torch import nn


class SupportEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, token_slots: int = 16):
        super().__init__()
        self.token_slots = token_slots
        self.proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.slot_queries = nn.Parameter(torch.randn(token_slots, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, probes: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # probes [B,A,N,F]
        B, A, N, _ = probes.shape
        x = self.proj(probes.reshape(B * A, N, -1))
        q = self.slot_queries.unsqueeze(0).expand(B * A, -1, -1)
        key_padding_mask = None
        any_valid = torch.ones(B * A, device=probes.device, dtype=torch.bool)
        if mask is not None:
            flat_mask = mask.reshape(B * A, N).to(probes.device).bool()
            any_valid = flat_mask.any(dim=-1)
            key_padding_mask = ~flat_mask
            # MultiheadAttention cannot handle all-masked rows. Use a dummy key,
            # then zero those rows after attention so empty support is a true no-op.
            if (~any_valid).any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[~any_valid, 0] = False
                x = x.clone()
                x[~any_valid, 0] = 0.0
        out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask)
        out = self.norm(out)
        out = out * any_valid.float().view(B * A, 1, 1)
        return out.reshape(B, A, self.token_slots, -1)
