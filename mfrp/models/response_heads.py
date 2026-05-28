"""MFRP response heads: soft branch, branch-conditioned trajectory, burden and margin."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from mfrp.data.schema import BRANCHES, TRAJ_TARGET_DIM


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        mods = []
        d = in_dim
        for _ in range(max(layers - 1, 0)):
            mods += [nn.Linear(d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        mods.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*mods)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BranchHead(nn.Module):
    def __init__(self, hidden_dim: int, branches: int = len(BRANCHES), dropout: float = 0.1):
        super().__init__()
        self.net = MLP(hidden_dim, hidden_dim, branches, 3, dropout)
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.net(r)


class BranchConditionedTrajectoryHead(nn.Module):
    def __init__(self, hidden_dim: int, branches: int = len(BRANCHES), modes: int = 6, future_steps: int = 80, state_dim: int = TRAJ_TARGET_DIM, dropout: float = 0.1):
        super().__init__()
        self.branches, self.modes, self.future_steps, self.state_dim = branches, modes, future_steps, state_dim
        self.mode_logits = MLP(hidden_dim, hidden_dim, branches * modes, 3, dropout)
        self.loc = MLP(hidden_dim, hidden_dim, branches * modes * future_steps * state_dim, 3, dropout)
        self.log_scale = MLP(hidden_dim, hidden_dim, branches * modes * future_steps * state_dim, 3, dropout)
    def forward(self, r: torch.Tensor) -> dict[str, torch.Tensor]:
        lead = r.shape[:-1]
        return {
            "mode_logits": self.mode_logits(r).reshape(*lead, self.branches, self.modes),
            "loc": self.loc(r).reshape(*lead, self.branches, self.modes, self.future_steps, self.state_dim),
            "log_scale": torch.clamp(self.log_scale(r).reshape(*lead, self.branches, self.modes, self.future_steps, self.state_dim), -5.0, 3.0),
        }


class GaussianScalarHead(nn.Module):
    def __init__(self, hidden_dim: int, branches: int = len(BRANCHES), dropout: float = 0.1):
        super().__init__()
        self.loc = MLP(hidden_dim, hidden_dim, branches, 3, dropout)
        self.log_scale = MLP(hidden_dim, hidden_dim, branches, 3, dropout)
    def forward(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.loc(r), torch.clamp(self.log_scale(r), -5.0, 3.0)


class ResponseHeads(nn.Module):
    def __init__(self, hidden_dim: int = 256, branches: int = len(BRANCHES), modes: int = 6, future_steps: int = 80, traj_dim: int = TRAJ_TARGET_DIM, dropout: float = 0.1):
        super().__init__()
        self.branch = BranchHead(hidden_dim, branches, dropout)
        self.trajectory = BranchConditionedTrajectoryHead(hidden_dim, branches, modes, future_steps, traj_dim, dropout)
        self.burden = GaussianScalarHead(hidden_dim, branches, dropout)
        self.margin = GaussianScalarHead(hidden_dim, branches, dropout)
    def forward(self, r: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        branch_logits = self.branch(r)
        branch_probs = F.softmax(branch_logits, dim=-1)
        burden_loc, burden_log_scale = self.burden(r)
        margin_loc, margin_log_scale = self.margin(r)
        return {
            "branch_logits": branch_logits,
            "branch_probs": branch_probs,
            "trajectory": self.trajectory(r),
            "burden_loc": burden_loc,
            "burden_log_scale": burden_log_scale,
            "margin_loc": margin_loc,
            "margin_log_scale": margin_log_scale,
        }
