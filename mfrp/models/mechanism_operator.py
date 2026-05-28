"""Scene-only and support-adapted mechanism token operator.

The paper separates the deployable scene-only mechanism token Ω^0 from the
support-adapted training/diagnostic token Ω^S.  This implementation accepts an
optional deployment-safe ``scene_features`` tensor.  When old smoke tests do not
provide it, the operator falls back to a candidate-pooled summary, but real
WOMD/Waymax builds should materialize scene_features from root-scene/map/history
only and keep labels/support probes out of deployment.
"""
from __future__ import annotations

import torch
from torch import nn

from mfrp.models.support_encoder import SupportEncoder


class MechanismTokenOperator(nn.Module):
    def __init__(self, candidate_feature_dim: int, hidden_dim: int = 256, token_slots: int = 16, support_feature_dim: int | None = None, scene_feature_dim: int | None = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_slots = token_slots
        self.candidate_feature_dim = candidate_feature_dim
        self.scene_feature_dim = scene_feature_dim or candidate_feature_dim
        self.candidate_proj = nn.Sequential(nn.Linear(candidate_feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.scene_proj = nn.Sequential(nn.Linear(self.scene_feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.scene_token = nn.Parameter(torch.randn(token_slots, hidden_dim) * 0.02)
        self.support_encoder = SupportEncoder(support_feature_dim or candidate_feature_dim + 6 + 5 + 3, hidden_dim, token_slots)
        self.query_proj = nn.Sequential(nn.Linear(candidate_feature_dim + hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def scene_only_tokens(self, candidate_features: torch.Tensor, scene_features: torch.Tensor | None = None) -> torch.Tensor:
        # candidate_features [B,A,K,F], scene_features optional [B,A,S]
        B, A, K, _ = candidate_features.shape
        base = self.scene_token.view(1, 1, self.token_slots, self.hidden_dim).expand(B, A, -1, -1)
        if scene_features is not None:
            pooled = self.scene_proj(scene_features.to(candidate_features.device).float())
        else:
            # Backward-compatible smoke path only; do not use for reported paper numbers.
            pooled = self.candidate_proj(candidate_features).mean(dim=2)
        return base + pooled.unsqueeze(-2)

    def support_tokens(self, scene_tokens: torch.Tensor, support_probes: torch.Tensor | None, support_mask: torch.Tensor | None) -> torch.Tensor:
        if support_probes is None:
            return scene_tokens
        return scene_tokens + self.support_encoder(support_probes, support_mask)

    def decode_queries(self, candidate_features: torch.Tensor, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, A, K, F = candidate_features.shape
        token_context = tokens.mean(dim=2, keepdim=True).expand(B, A, K, -1)
        q = self.query_proj(torch.cat([candidate_features, token_context], dim=-1)).reshape(B * A, K, -1)
        t = tokens.reshape(B * A, tokens.shape[-2], tokens.shape[-1])
        out, _ = self.cross_attn(q, t, t)
        out = self.norm(out + q).reshape(B, A, K, -1)
        return out, tokens
