from __future__ import annotations

import torch
import torch.nn.functional as F


def _query_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    q = batch.get("query_probe_mask")
    vv = batch.get("variant_valid")
    if q is None:
        q = torch.ones_like(vv, dtype=torch.bool)
    if vv is not None:
        q = q & vv.bool()
    return q.bool()


def _weighted_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x * mask.to(x.dtype)).sum() / mask.to(x.dtype).sum().clamp_min(eps)


def mechanism_nll(prefix: str, out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict) -> torch.Tensor:
    branch_logits = out[f"{prefix}_branch_logits"]
    branch_target = batch["branch_probs"].to(branch_logits.device)
    qmask = _query_mask(batch).to(branch_logits.device)
    # [B,A,K,R,C] vs logits [B,A,K,C]
    ce = -(branch_target * F.log_softmax(branch_logits, -1).unsqueeze(3)).sum(-1)
    branch_loss = _weighted_mean(ce, qmask)
    hard = branch_target.argmax(-1)
    R = hard.shape[-1]
    burden_src = out[f"{prefix}_burden_mu"].unsqueeze(3).expand(-1, -1, -1, R, -1)
    margin_src = out[f"{prefix}_margin_mu"].unsqueeze(3).expand(-1, -1, -1, R, -1)
    burden_mu = burden_src.gather(-1, hard.unsqueeze(-1)).squeeze(-1)
    margin_mu = margin_src.gather(-1, hard.unsqueeze(-1)).squeeze(-1)
    burden_loss = _weighted_mean(F.smooth_l1_loss(burden_mu, batch["burden"].to(burden_mu.device), reduction="none"), qmask)
    margin_loss = _weighted_mean(F.smooth_l1_loss(margin_mu, batch["safety_margin"].to(margin_mu.device), reduction="none"), qmask)
    traj_mu = out[f"{prefix}_trajectory_mu"]
    # Select branch trajectory [B,A,K,R,T,D]
    idx = hard[..., None, None, None].expand(*hard.shape, 1, traj_mu.shape[-2], traj_mu.shape[-1])
    pred_traj = traj_mu.unsqueeze(3).expand(-1, -1, -1, R, -1, -1, -1).gather(4, idx).squeeze(4)
    tmask = batch.get("trajectory_mask", torch.ones_like(batch["trajectory"][..., 0], dtype=torch.bool)).to(pred_traj.device) & qmask.unsqueeze(-1).to(pred_traj.device)
    traj_err = torch.linalg.norm(pred_traj[..., :2] - batch["trajectory"].to(pred_traj.device)[..., :2], dim=-1)
    traj_loss = _weighted_mean(traj_err, tmask)
    w = cfg.get("loss", cfg).get("weights", {})
    return branch_loss + float(w.get("trajectory", 1.0)) * traj_loss + float(w.get("burden", 1.0)) * burden_loss + float(w.get("margin", 1.0)) * margin_loss


def distillation_loss(out: dict[str, torch.Tensor]) -> torch.Tensor:
    if "scene_branch_prob" not in out or "support_branch_prob" not in out:
        return torch.zeros((), device=next(iter(out.values())).device)
    p = out["support_branch_prob"].detach().clamp_min(1e-6)
    qlog = out["scene_branch_prob"].clamp_min(1e-6).log()
    kl = (p * (p.log() - qlog)).sum(-1).mean()
    cont = F.smooth_l1_loss(out["scene_burden_mu"], out["support_burden_mu"].detach()) + F.smooth_l1_loss(out["scene_margin_mu"], out["support_margin_mu"].detach())
    return kl + 0.25 * cont


def geometry_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    needed = ["edge_batch", "edge_agent", "edge_a", "edge_b", "edge_distance"]
    if not all(k in batch for k in needed):
        return torch.zeros((), device=next(iter(out.values())).device)
    device = out["support_branch_prob"].device if "support_branch_prob" in out else out["scene_branch_prob"].device
    prefix = "support" if "support_branch_prob" in out else "scene"
    b = batch["edge_batch"].long().to(device); a = batch["edge_agent"].long().to(device); ca = batch["edge_a"].long().to(device); cb = batch["edge_b"].long().to(device)
    dist = batch["edge_distance"].to(device).clamp_min(1e-3)
    pa = out[f"{prefix}_branch_prob"][b, a, ca]
    pb = out[f"{prefix}_branch_prob"][b, a, cb]
    tv = 0.5 * (pa - pb).abs().sum(-1)
    ma = out[f"{prefix}_margin_mu"][b, a, ca].mean(-1)
    mb = out[f"{prefix}_margin_mu"][b, a, cb].mean(-1)
    ba = out[f"{prefix}_burden_mu"][b, a, ca].mean(-1)
    bb = out[f"{prefix}_burden_mu"][b, a, cb].mean(-1)
    pred = tv + 0.2 * (ma - mb).abs() + 0.2 * (ba - bb).abs()
    return (pred / dist).clamp(max=20).mean()


def cw_loss(prefix: str, out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if "cw_soft_label" not in batch or "cw_confidence" not in batch:
        return torch.zeros((), device=next(iter(out.values())).device)
    kappa = out[f"{prefix}_kappa"]
    target = batch["cw_soft_label"].to(kappa.device).clamp(0, 1)
    weight = batch["cw_confidence"].to(kappa.device).clamp(0, 1)
    if float(weight.sum().detach().cpu()) <= 0:
        return torch.zeros((), device=kappa.device)
    bce = F.binary_cross_entropy(kappa.clamp(1e-6, 1 - 1e-6), target, reduction="none")
    loss = _weighted_mean(bce, weight > 0) * weight[weight > 0].mean()
    return loss


def total_mfrp_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict) -> dict[str, torch.Tensor]:
    loss_cfg = cfg.get("loss", {})
    losses: dict[str, torch.Tensor] = {}
    prefix = "support" if "support_branch_logits" in out else "scene"
    losses["mech"] = mechanism_nll(prefix, out, batch, cfg)
    losses["distill"] = distillation_loss(out) * float(loss_cfg.get("lambda_distill", 1.0))
    losses["geo"] = geometry_loss(out, batch) * float(loss_cfg.get("lambda_geo", 0.1))
    losses["cw"] = cw_loss(prefix, out, batch) * float(loss_cfg.get("lambda_cw", 1.0))
    losses["total"] = losses["mech"] + losses["distill"] + losses["geo"] + losses["cw"]
    return losses
