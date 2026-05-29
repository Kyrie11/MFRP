from __future__ import annotations

import torch


def p_unsafe_from_margin(out: dict[str, torch.Tensor]) -> torch.Tensor:
    branch = out.get("branch_probs", out.get("branch_prob"))
    margin_loc = out.get("margin_loc", out.get("margin_mu"))
    if margin_loc is None and "margin_log_scale" in out:
        margin_loc = out["margin_loc"]
    scale = out.get("margin_scale")
    if scale is None:
        if "margin_log_scale" in out:
            scale = torch.exp(out["margin_log_scale"])
        elif "margin_sigma" in out:
            scale = out["margin_sigma"]
        else:
            scale = torch.ones_like(margin_loc)
    normal = torch.distributions.Normal(margin_loc, scale.clamp_min(1e-3))
    p_unsafe_branch = normal.cdf(torch.zeros((), device=margin_loc.device))
    return (branch * p_unsafe_branch).sum(-1)


def scene_mechanism_risk(est: dict[str, torch.Tensor], *, mode: str = "noisy_or") -> torch.Tensor:
    p = est.get("p_viol", est.get("p_viol_i"))
    if mode == "max":
        return p.clamp(0, 1).max(dim=1).values
    if mode == "sum":
        return p.clamp(0, 1).sum(dim=1).clamp(max=1.0)
    return 1.0 - torch.prod(1.0 - p.clamp(0, 1), dim=1)


def mechanism_estimates(out: dict[str, torch.Tensor], *, prefix: str = "scene", aggregation: str = "noisy_or") -> dict[str, torch.Tensor]:
    branch = out[f"{prefix}_branch_prob"]
    margin_mu = out[f"{prefix}_margin_mu"]
    margin_sigma = out[f"{prefix}_margin_sigma"]
    kappa = out[f"{prefix}_kappa"]
    p_unsafe = p_unsafe_from_margin({"branch_probs": branch, "margin_loc": margin_mu, "margin_scale": margin_sigma})
    p_viol_i = p_unsafe + kappa - p_unsafe * kappa
    rho = scene_mechanism_risk({"p_viol": p_viol_i}, mode=aggregation)
    entropy = -(branch.clamp_min(1e-6) * branch.clamp_min(1e-6).log()).sum(-1)
    nu = entropy.max(dim=1).values / torch.log(torch.tensor(branch.shape[-1], dtype=branch.dtype, device=branch.device))
    gamma_i = out.get(f"{prefix}_gamma", torch.zeros_like(kappa))
    gamma = gamma_i.max(dim=1).values
    return {
        "p_unsafe_i": p_unsafe,
        "kappa_i": kappa,
        "p_viol_i": p_viol_i,
        "rho_mech": rho,
        "nu": nu,
        "gamma_i": gamma_i,
        "gamma": gamma,
        "P_C": out[f"{prefix}_P_C"],
        "S_C": out[f"{prefix}_S_C"],
        "S_notC": out[f"{prefix}_S_notC"],
        "B_C": out[f"{prefix}_B_C"],
        "D_C": out[f"{prefix}_D_C"],
    }
