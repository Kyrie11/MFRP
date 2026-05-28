"""MFRP planning estimators from scene-only response surfaces."""
from __future__ import annotations

import torch
from torch.distributions import Normal

from mfrp.data.schema import BRANCHES, CEDING_BRANCH_IDS


def p_unsafe_from_margin(outputs: dict) -> torch.Tensor:
    scale = torch.exp(outputs["margin_log_scale"]).clamp_min(1e-5)
    normal = Normal(outputs["margin_loc"], scale)
    p_branch = outputs["branch_probs"]
    p_neg = normal.cdf(torch.zeros((), device=scale.device))
    return (p_branch * p_neg).sum(dim=-1)


def per_agent_estimates(outputs: dict, priority_score: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    p = outputs["branch_probs"]
    scale = torch.exp(outputs["margin_log_scale"]).clamp_min(1e-5)
    normal = Normal(outputs["margin_loc"], scale)
    safe = 1.0 - normal.cdf(torch.zeros((), device=scale.device))
    c_mask = torch.zeros(len(BRANCHES), device=p.device, dtype=torch.bool)
    c_mask[torch.as_tensor(CEDING_BRANCH_IDS, device=p.device)] = True
    P_C = p[..., c_mask].sum(dim=-1).clamp_min(1e-6)
    P_notC = p[..., ~c_mask].sum(dim=-1).clamp_min(1e-6)
    S_C = (p[..., c_mask] * safe[..., c_mask]).sum(dim=-1) / P_C
    S_notC = (p[..., ~c_mask] * safe[..., ~c_mask]).sum(dim=-1) / P_notC
    B_C = (p[..., c_mask] * outputs["burden_loc"][..., c_mask]).sum(dim=-1) / P_C
    D_C = torch.relu(S_C - S_notC)
    kappa = outputs.get("kappa")
    if kappa is None:
        kappa = torch.sigmoid(outputs["kappa_logits"])
    p_unsafe = p_unsafe_from_margin(outputs)
    p_viol = p_unsafe + kappa - p_unsafe * kappa
    return {"p_unsafe": p_unsafe, "P_C": P_C, "S_C": S_C, "S_notC": S_notC, "B_C": B_C, "D_C": D_C, "kappa": kappa, "p_viol": p_viol}


def scene_mechanism_risk(per_agent: dict[str, torch.Tensor], agent_mask: torch.Tensor | None = None) -> torch.Tensor:
    p = per_agent["p_viol"].clamp(0.0, 1.0)
    if agent_mask is not None:
        p = torch.where(agent_mask.to(p.device).bool(), p, torch.zeros_like(p))
    return 1.0 - torch.prod(1.0 - p, dim=1)


def uncertainty_proxy(outputs: dict) -> torch.Tensor:
    p = outputs["branch_probs"].clamp_min(1e-6)
    entropy = -(p * p.log()).sum(dim=-1)
    margin_var = torch.exp(2.0 * outputs["margin_log_scale"]).mean(dim=-1)
    kappa_var = outputs.get("kappa_var", outputs.get("kappa", torch.zeros_like(entropy)) * (1.0 - outputs.get("kappa", torch.zeros_like(entropy))))
    return entropy + margin_var + kappa_var


def boundary_sensitivity(outputs: dict, candidate_features: torch.Tensor, neighbor_index: torch.Tensor, neighbor_valid: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    B, A, K, _ = outputs["branch_probs"].shape
    gamma = torch.zeros((B, A, K), device=outputs["branch_probs"].device)
    if neighbor_index.numel() == 0:
        return gamma
    valid = torch.ones(neighbor_index.shape[:-1], device=gamma.device, dtype=torch.bool) if neighbor_valid is None else neighbor_valid.to(gamma.device).bool()
    b, a, e = torch.where(valid)
    if b.numel() == 0:
        return gamma
    k0 = neighbor_index[b,a,e,0].long(); k1 = neighbor_index[b,a,e,1].long()
    p0 = outputs["branch_probs"][b,a,k0]; p1 = outputs["branch_probs"][b,a,k1]
    m = 0.5 * (p0 + p1).clamp_min(1e-6)
    js = 0.5 * (p0.clamp_min(1e-6) * (p0.clamp_min(1e-6).log() - m.log())).sum(-1) + 0.5 * (p1.clamp_min(1e-6) * (p1.clamp_min(1e-6).log() - m.log())).sum(-1)
    du = torch.linalg.norm(candidate_features[b,a,k0] - candidate_features[b,a,k1], dim=-1)
    val = js / (du + eps)
    gamma[b,a,k0] = torch.maximum(gamma[b,a,k0], val)
    gamma[b,a,k1] = torch.maximum(gamma[b,a,k1], val)
    return gamma
