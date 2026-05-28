"""MFRP losses aligned with the paper primitive O=(M,Y,DeltaB,H)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.distributions import Normal


def masked_mean(value: torch.Tensor, mask: torch.Tensor | None, eps: float = 1e-6) -> torch.Tensor:
    if mask is None:
        return value.mean()
    mask = mask.to(value.device).float()
    while mask.dim() < value.dim():
        mask = mask.unsqueeze(-1)
    return (value * mask).sum() / torch.clamp(mask.sum(), min=eps)


def _variant_mean(x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    if mask is not None and x.dim() >= 4 and tuple(mask.shape) == tuple(x.shape[:4]):  # [B,A,K,R,...]
        m = mask.to(x.device).float()
        while m.dim() < x.dim():
            m = m.unsqueeze(-1)
        return (x * m).sum(dim=3) / torch.clamp(m.sum(dim=3), min=1e-6), (mask.sum(dim=3) > 0)
    if mask is None and x.dim() >= 4:
        return x.mean(dim=3), None
    return x, mask


def soft_branch_ce(outputs: dict, branch_probs: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    target, m2 = _variant_mean(branch_probs.float(), mask)
    logp = F.log_softmax(outputs["branch_logits"], dim=-1)
    return masked_mean(-(target * logp).sum(dim=-1), m2)


def gaussian_nll(value: torch.Tensor, loc: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
    scale = torch.exp(log_scale).clamp_min(1e-5)
    return 0.5 * ((value - loc) / scale).pow(2) + log_scale + 0.5 * torch.log(torch.tensor(2.0 * 3.141592653589793, device=value.device))


def branch_weighted_scalar_nll(outputs: dict, value: torch.Tensor, branch_probs: torch.Tensor, mask: torch.Tensor | None, loc_key: str, scale_key: str) -> torch.Tensor:
    target, m2 = _variant_mean(value.float(), mask)
    q, _ = _variant_mean(branch_probs.float(), mask)
    nll = gaussian_nll(target.unsqueeze(-1), outputs[loc_key], outputs[scale_key])
    return masked_mean((q * nll).sum(dim=-1), m2)


def trajectory_nll(outputs: dict, target: torch.Tensor, target_mask: torch.Tensor, branch_probs: torch.Tensor, probe_mask: torch.Tensor | None) -> torch.Tensor:
    # Stable branch/mode-weighted Gaussian mixture NLL. Targets with variants are averaged with masks first.
    if target.dim() == 6:
        pm = probe_mask if probe_mask is not None else torch.ones(target.shape[:4], device=target.device, dtype=torch.bool)
        tmask = target_mask.float() * pm.unsqueeze(-1).float()
        denom = tmask.sum(dim=3, keepdim=False).clamp_min(1e-6)
        target_m = (target * tmask.unsqueeze(-1)).sum(dim=3) / denom.unsqueeze(-1)
        time_mask = denom > 0
        q, m2 = _variant_mean(branch_probs.float(), pm)
    else:
        target_m, time_mask, q, m2 = target, target_mask.bool(), branch_probs.float(), probe_mask
    traj = outputs["trajectory"]
    loc = traj["loc"]
    log_scale = traj["log_scale"]
    mode_logits = traj["mode_logits"]
    scale = torch.exp(log_scale).clamp_min(1e-5)
    err = 0.5 * ((target_m.unsqueeze(3).unsqueeze(4) - loc) / scale).pow(2) + log_scale
    time_mask_f = time_mask.to(loc.device).float().unsqueeze(3).unsqueeze(4).unsqueeze(-1)
    nll_mode = (err * time_mask_f).sum(dim=(-1, -2)) - F.log_softmax(mode_logits, dim=-1)
    nll_branch = -torch.logsumexp(-nll_mode, dim=-1)
    return masked_mean((q * nll_branch).sum(dim=-1), m2)


def mechanism_nll(outputs: dict, batch: dict, weights: dict | None = None) -> dict[str, torch.Tensor]:
    weights = weights or {}
    pmask = batch.get("query_probe_mask", batch.get("variant_valid"))
    branch_probs = batch["branch_probs"]
    losses = {
        "L_branch": soft_branch_ce(outputs, branch_probs, pmask),
        "L_traj": trajectory_nll(outputs, batch["trajectory"].float(), batch["trajectory_mask"].bool(), branch_probs, pmask),
        "L_burden": branch_weighted_scalar_nll(outputs, batch["burden"].float(), branch_probs, pmask, "burden_loc", "burden_log_scale"),
        "L_margin": branch_weighted_scalar_nll(outputs, batch["safety_margin"].float(), branch_probs, pmask, "margin_loc", "margin_log_scale"),
    }
    losses["L_mech"] = losses["L_branch"] + weights.get("lambda_Y", 1.0) * losses["L_traj"] + weights.get("lambda_B", 1.0) * losses["L_burden"] + weights.get("lambda_H", 1.0) * losses["L_margin"]
    return losses


def gaussian_kl(loc_t: torch.Tensor, log_s_t: torch.Tensor, loc_s: torch.Tensor, log_s_s: torch.Tensor) -> torch.Tensor:
    vt = torch.exp(2 * log_s_t)
    vs = torch.exp(2 * log_s_s).clamp_min(1e-8)
    return log_s_s - log_s_t + (vt + (loc_t - loc_s).pow(2)) / (2 * vs) - 0.5


def distillation_loss(teacher: dict, student: dict, mask: torch.Tensor | None = None, detach_teacher: bool = True) -> torch.Tensor:
    if detach_teacher:
        teacher = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in teacher.items()}
    p_t = teacher["branch_probs"]
    log_s = F.log_softmax(student["branch_logits"], dim=-1)
    branch_kl = (p_t * (torch.log(p_t.clamp_min(1e-6)) - log_s)).sum(dim=-1)
    burden_kl = (p_t * gaussian_kl(teacher["burden_loc"], teacher["burden_log_scale"], student["burden_loc"], student["burden_log_scale"])).sum(dim=-1)
    margin_kl = (p_t * gaussian_kl(teacher["margin_loc"], teacher["margin_log_scale"], student["margin_loc"], student["margin_log_scale"])).sum(dim=-1)
    # Mode-matched trajectory proxy.
    tloc = teacher["trajectory"]["loc"].mean(dim=-3)
    sloc = student["trajectory"]["loc"].mean(dim=-3)
    traj_d = (p_t * (tloc - sloc).pow(2).mean(dim=(-1, -2))).sum(dim=-1)
    return masked_mean(branch_kl + burden_kl + margin_kl + traj_d, mask)


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp_min(eps); q = q.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True); q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)
    return 0.5 * (p * (p.log() - m.log())).sum(dim=-1) + 0.5 * (q * (q.log() - m.log())).sum(dim=-1)


def response_geometry_loss(outputs: dict, batch: dict, beta_M: float = 1.0, beta_Y: float = 1.0, beta_B: float = 1.0, beta_H: float = 1.0) -> torch.Tensor:
    edge_index = batch.get("edge_index")
    response_distance = batch.get("response_distance")
    if edge_index is None or response_distance is None or edge_index.numel() == 0:
        return outputs["branch_probs"].sum() * 0.0
    valid = batch.get("edge_valid", torch.ones(edge_index.shape[:-1], device=edge_index.device, dtype=torch.bool)).bool()
    b, a, e = torch.where(valid)
    if b.numel() == 0:
        return outputs["branch_probs"].sum() * 0.0
    k0 = edge_index[b, a, e, 0].long(); k1 = edge_index[b, a, e, 1].long()
    p0 = outputs["branch_probs"][b, a, k0]; p1 = outputs["branch_probs"][b, a, k1]
    Dm = js_divergence(p0, p1)
    Db = torch.abs(outputs["burden_loc"][b,a,k0].mean(-1) - outputs["burden_loc"][b,a,k1].mean(-1))
    Dh = torch.abs(outputs["margin_loc"][b,a,k0].mean(-1) - outputs["margin_loc"][b,a,k1].mean(-1))
    # loc[b,a,k] has [branch, mode, time, dim]; compare expected XY paths.
    w0 = p0 / p0.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    w1 = p1 / p1.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    y0_branch = outputs["trajectory"]["loc"][b, a, k0].mean(dim=2)  # [E,C,T,D], mean over modes
    y1_branch = outputs["trajectory"]["loc"][b, a, k1].mean(dim=2)
    y0 = (w0[..., None, None] * y0_branch).sum(dim=1)
    y1 = (w1[..., None, None] * y1_branch).sum(dim=1)
    Dy = torch.linalg.norm(y0[..., :2] - y1[..., :2], dim=-1).mean(dim=-1)
    Dp = beta_M * Dm + beta_Y * Dy + beta_B * Db + beta_H * Dh
    return torch.mean(torch.abs(Dp - response_distance.to(Dp.device)[b,a,e]))


def coercion_witness_loss(outputs: dict, batch: dict, tau_rank: float = 1.0, lambda_r: float = 0.5) -> torch.Tensor:
    target = batch.get("cw_soft_label")
    conf = batch.get("cw_confidence")
    if target is None or conf is None:
        return outputs["kappa_logits"].sum() * 0.0
    event = F.binary_cross_entropy_with_logits(outputs["kappa_logits"], target.to(outputs["kappa_logits"].device).float(), reduction="none")
    loss = masked_mean(event, conf.to(outputs["kappa_logits"].device).float())
    pairs = batch.get("cw_rank_pairs")
    valid = batch.get("cw_rank_valid")
    if pairs is not None and valid is not None and pairs.numel() > 0:
        b, a, p = torch.where(valid.bool())
        if b.numel() > 0:
            kp = pairs[b,a,p,0].long(); km = pairs[b,a,p,1].long()
            diff = outputs["kappa_logits"][b,a,kp] - outputs["kappa_logits"][b,a,km]
            loss = loss + lambda_r * F.softplus(-diff / max(tau_rank, 1e-6)).mean()
    return loss


def total_mfrp_loss(model_outputs: dict, batch: dict, cfg: dict | None = None) -> dict[str, torch.Tensor]:
    cfg = cfg or {}
    loss_cfg = cfg.get("loss", cfg)
    model_cfg = cfg.get("model", {})
    scene = model_outputs["scene_only"]
    support = model_outputs.get("support_adapted")
    losses: dict[str, torch.Tensor] = {}
    # Paper alignment: L_mech is primarily evaluated with support-adapted tokens
    # on held-out query interventions; scene-only prediction is deployable and is
    # matched to the support surface by distillation. A scene-only auxiliary NLL
    # can be enabled, but it is not the default conceptual objective.
    total = scene["branch_logits"].sum() * 0.0
    use_support = bool(model_cfg.get("use_support_adapted_loss", True)) and support is not None
    use_scene = bool(model_cfg.get("use_scene_only_loss", False)) or not use_support
    if use_support:
        sup_losses = mechanism_nll(support, batch, loss_cfg)
        total = total + sup_losses["L_mech"]
        losses.update({"support_" + k: v for k, v in sup_losses.items()})
    if use_scene:
        scene_losses = mechanism_nll(scene, batch, loss_cfg)
        total = total + scene_losses["L_mech"]
        losses.update({"scene_" + k: v for k, v in scene_losses.items()})
    if use_support and support is not scene and loss_cfg.get("lambda_d", 0.1) > 0 and bool(model_cfg.get("use_distillation", True)):
        d = distillation_loss(support, scene, batch.get("agent_candidate_valid"), loss_cfg.get("detach_teacher", True))
        total = total + loss_cfg.get("lambda_d", 0.1) * d
        losses["L_distill"] = d
    if loss_cfg.get("lambda_g", 0.02) > 0 and bool(model_cfg.get("use_geometry_loss", True)):
        g = response_geometry_loss(scene, batch)
        total = total + loss_cfg.get("lambda_g", 0.02) * g
        losses["L_geo"] = g
    if loss_cfg.get("lambda_cw", 1.0) > 0 and bool(model_cfg.get("use_coercion_witness", True)):
        cw = coercion_witness_loss(scene, batch, loss_cfg.get("tau_rank", 1.0), loss_cfg.get("lambda_r", 0.5))
        total = total + loss_cfg.get("lambda_cw", 1.0) * cw
        losses["L_cw"] = cw
    losses["total"] = total
    return losses
