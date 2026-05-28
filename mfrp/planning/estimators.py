from __future__ import annotations

import torch


def mechanism_estimates(out: dict[str, torch.Tensor], *, prefix: str = "scene") -> dict[str, torch.Tensor]:
    branch = out[f"{prefix}_branch_prob"]
    margin_mu = out[f"{prefix}_margin_mu"]
    margin_sigma = out[f"{prefix}_margin_sigma"]
    kappa = out[f"{prefix}_kappa"]
    normal = torch.distributions.Normal(margin_mu, margin_sigma)
    p_unsafe_branch = normal.cdf(torch.zeros((), device=margin_mu.device))
    p_unsafe = (branch * p_unsafe_branch).sum(-1)
    p_viol_i = p_unsafe + kappa - p_unsafe * kappa
    rho = 1.0 - torch.prod(1.0 - p_viol_i.clamp(0, 1), dim=1)
    entropy = -(branch.clamp_min(1e-6) * branch.clamp_min(1e-6).log()).sum(-1)
    # [B,K]
    nu = entropy.max(dim=1).values / torch.log(torch.tensor(branch.shape[-1], dtype=branch.dtype, device=branch.device))
    return {
        "p_unsafe_i": p_unsafe,
        "kappa_i": kappa,
        "p_viol_i": p_viol_i,
        "rho_mech": rho,
        "nu": nu,
        "P_C": out[f"{prefix}_P_C"],
        "S_C": out[f"{prefix}_S_C"],
        "S_notC": out[f"{prefix}_S_notC"],
        "B_C": out[f"{prefix}_B_C"],
        "D_C": out[f"{prefix}_D_C"],
    }
