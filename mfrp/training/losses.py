from __future__ import annotations

import torch
import torch.nn.functional as F


def _device_from(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            if torch.is_tensor(v):
                return v.device
            if isinstance(v, dict):
                d = _device_from(v)
                if d is not None:
                    return d
    return torch.device("cpu")


def _query_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    vv = batch.get("variant_valid")
    q = batch.get("query_probe_mask")
    if vv is None:
        raise KeyError("batch requires variant_valid for supervised loss")
    if q is None:
        q = torch.ones_like(vv, dtype=torch.bool)
    return (q.bool() & vv.bool())


def _weighted_mean(x: torch.Tensor, mask: torch.Tensor | None, eps: float = 1e-6) -> torch.Tensor:
    if mask is None:
        return x.mean()
    w = mask.to(dtype=x.dtype, device=x.device)
    return (x * w).sum() / w.sum().clamp_min(eps)


def _gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    sigma = sigma.clamp_min(1e-3)
    return 0.5 * ((target - mu) / sigma) ** 2 + sigma.log()


def _legacy_to_prefix(out: dict[str, torch.Tensor], prefix: str = "scene") -> dict[str, torch.Tensor]:
    if f"{prefix}_branch_logits" in out:
        return out
    # out is likely a nested compat view such as out["scene_only"].
    return {
        f"{prefix}_branch_logits": out["branch_logits"],
        f"{prefix}_branch_prob": out.get("branch_probs", F.softmax(out["branch_logits"], -1)),
        f"{prefix}_trajectory_mu": out.get("trajectory_loc"),
        f"{prefix}_burden_mu": out["burden_loc"],
        f"{prefix}_burden_sigma": out.get("burden_scale", torch.ones_like(out["burden_loc"])),
        f"{prefix}_margin_mu": out["margin_loc"],
        f"{prefix}_margin_sigma": out.get("margin_scale", torch.ones_like(out["margin_loc"])),
        f"{prefix}_kappa": out.get("kappa"),
    }


def mechanism_nll(prefix: str, out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict | None = None) -> torch.Tensor:
    cfg = cfg or {}
    branch_logits = out[f"{prefix}_branch_logits"]
    branch_target = batch["branch_probs"].to(branch_logits.device)
    qmask = _query_mask(batch).to(branch_logits.device)
    ce = -(branch_target * F.log_softmax(branch_logits, -1).unsqueeze(3)).sum(-1)
    branch_loss = _weighted_mean(ce, qmask)

    hard = branch_target.argmax(-1)
    R = hard.shape[-1]
    burden_mu_all = out[f"{prefix}_burden_mu"].unsqueeze(3).expand(-1, -1, -1, R, -1)
    margin_mu_all = out[f"{prefix}_margin_mu"].unsqueeze(3).expand(-1, -1, -1, R, -1)
    burden_s_all = out[f"{prefix}_burden_sigma"].unsqueeze(3).expand_as(burden_mu_all)
    margin_s_all = out[f"{prefix}_margin_sigma"].unsqueeze(3).expand_as(margin_mu_all)
    idx = hard.unsqueeze(-1)
    burden_mu = burden_mu_all.gather(-1, idx).squeeze(-1)
    margin_mu = margin_mu_all.gather(-1, idx).squeeze(-1)
    burden_s = burden_s_all.gather(-1, idx).squeeze(-1)
    margin_s = margin_s_all.gather(-1, idx).squeeze(-1)
    burden_loss = _weighted_mean(_gaussian_nll(burden_mu, burden_s, batch["burden"].to(branch_logits.device)), qmask)
    margin_loss = _weighted_mean(_gaussian_nll(margin_mu, margin_s, batch["safety_margin"].to(branch_logits.device)), qmask)

    traj_mu = out.get(f"{prefix}_trajectory_mu")
    if traj_mu is None:
        traj_loss = torch.zeros((), device=branch_logits.device)
    else:
        idx_t = hard[..., None, None, None].expand(*hard.shape, 1, traj_mu.shape[-2], traj_mu.shape[-1])
        pred_traj = traj_mu.unsqueeze(3).expand(-1, -1, -1, R, -1, -1, -1).gather(4, idx_t).squeeze(4)
        target_traj = batch["trajectory"].to(pred_traj.device)
        T = min(pred_traj.shape[-2], target_traj.shape[-2])
        pred_traj = pred_traj[..., :T, :]
        target_traj = target_traj[..., :T, :]
        tmask = batch.get("trajectory_mask", torch.ones_like(target_traj[..., 0], dtype=torch.bool)).to(pred_traj.device)[..., :T] & qmask.unsqueeze(-1).to(pred_traj.device)
        traj_err = torch.linalg.norm(pred_traj[..., :2] - target_traj[..., :2], dim=-1)
        traj_loss = _weighted_mean(traj_err, tmask)
    w = cfg.get("loss", cfg).get("weights", {}) if cfg else {}
    return branch_loss + float(w.get("trajectory", 1.0)) * traj_loss + float(w.get("burden", 1.0)) * burden_loss + float(w.get("margin", 1.0)) * margin_loss


def _distill_new(out: dict[str, torch.Tensor]) -> torch.Tensor:
    if "scene_branch_prob" not in out or "support_branch_prob" not in out:
        return torch.zeros((), device=_device_from(out))
    p = out["support_branch_prob"].detach().clamp_min(1e-6)
    qlog = out["scene_branch_prob"].clamp_min(1e-6).log()
    kl = (p * (p.log() - qlog)).sum(-1).mean()
    cont = (
        F.smooth_l1_loss(out["scene_burden_mu"], out["support_burden_mu"].detach())
        + F.smooth_l1_loss(out["scene_margin_mu"], out["support_margin_mu"].detach())
        + 0.25 * F.smooth_l1_loss(out["scene_trajectory_mu"], out["support_trajectory_mu"].detach())
        + F.smooth_l1_loss(out["scene_kappa"], out["support_kappa"].detach())
    )
    return kl + 0.25 * cont


def distillation_loss(*args) -> torch.Tensor:
    # New: distillation_loss(out). Legacy: distillation_loss(teacher, student, mask).
    if len(args) == 1:
        return _distill_new(args[0])
    teacher, student = args[0], args[1]
    mask = args[2] if len(args) > 2 else None
    p = teacher["branch_probs"].detach().clamp_min(1e-6)
    q = student["branch_probs"].clamp_min(1e-6)
    kl = (p * (p.log() - q.log())).sum(-1)
    if mask is not None:
        kl = _weighted_mean(kl, mask.to(kl.device))
    else:
        kl = kl.mean()
    cont = F.smooth_l1_loss(student["burden_loc"], teacher["burden_loc"].detach()) + F.smooth_l1_loss(student["margin_loc"], teacher["margin_loc"].detach())
    if "trajectory_loc" in teacher and "trajectory_loc" in student:
        cont = cont + 0.1 * F.smooth_l1_loss(student["trajectory_loc"], teacher["trajectory_loc"].detach())
    return kl + 0.25 * cont


def response_geometry_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    device = _device_from(out)
    if "edge_index" in batch and "response_distance" in batch:
        edge = batch["edge_index"].long().to(device)  # [B,A,E,2]
        valid = batch.get("edge_valid", torch.ones(edge.shape[:-1], dtype=torch.bool, device=device)).to(device)
        target = batch["response_distance"].to(device).clamp_min(0.0)
        bp = out.get("branch_probs")
        mm = out.get("margin_loc")
        bb = out.get("burden_loc")
        if bp is None:
            return torch.zeros((), device=device)
        bidx = torch.arange(edge.shape[0], device=device)[:, None, None].expand(edge.shape[:-1])
        aidx = torch.arange(edge.shape[1], device=device)[None, :, None].expand(edge.shape[:-1])
        ia, ib = edge[..., 0], edge[..., 1]
        pa, pb = bp[bidx, aidx, ia], bp[bidx, aidx, ib]
        pred = 0.5 * (pa - pb).abs().sum(-1)
        if mm is not None:
            pred = pred + 0.2 * (mm[bidx, aidx, ia].mean(-1) - mm[bidx, aidx, ib].mean(-1)).abs()
        if bb is not None:
            pred = pred + 0.2 * (bb[bidx, aidx, ia].mean(-1) - bb[bidx, aidx, ib].mean(-1)).abs()
        return _weighted_mean(F.smooth_l1_loss(pred, target, reduction="none"), valid)
    return torch.zeros((), device=device)


def geometry_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if "edge_batch" not in batch:
        if "scene_only" in out:
            return response_geometry_loss(out["scene_only"], batch)
        return response_geometry_loss(out, batch)
    prefix = "support" if "support_branch_prob" in out else "scene"
    device = out[f"{prefix}_branch_prob"].device
    b = batch["edge_batch"].long().to(device); a = batch["edge_agent"].long().to(device); ca = batch["edge_a"].long().to(device); cb = batch["edge_b"].long().to(device)
    target = batch.get("edge_response_distance", batch.get("edge_distance")).to(device).clamp_min(0.0)
    pa = out[f"{prefix}_branch_prob"][b, a, ca]
    pb = out[f"{prefix}_branch_prob"][b, a, cb]
    pred = 0.5 * (pa - pb).abs().sum(-1)
    pred = pred + 0.2 * (out[f"{prefix}_margin_mu"][b, a, ca].mean(-1) - out[f"{prefix}_margin_mu"][b, a, cb].mean(-1)).abs()
    pred = pred + 0.2 * (out[f"{prefix}_burden_mu"][b, a, ca].mean(-1) - out[f"{prefix}_burden_mu"][b, a, cb].mean(-1)).abs()
    return F.smooth_l1_loss(pred, target, reduction="mean")


def cw_loss(prefix: str, out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if "cw_soft_label" not in batch or "cw_confidence" not in batch or f"{prefix}_kappa" not in out:
        return torch.zeros((), device=_device_from(out))
    kappa = out[f"{prefix}_kappa"]
    target = batch["cw_soft_label"].to(kappa.device).clamp(0, 1)
    weight = batch["cw_confidence"].to(kappa.device).clamp(0, 1)
    if float(weight.sum().detach().cpu()) <= 0:
        return torch.zeros((), device=kappa.device)
    bce = F.binary_cross_entropy(kappa.clamp(1e-6, 1 - 1e-6), target, reduction="none")
    loss = (bce * weight).sum() / weight.sum().clamp_min(1e-6)
    # Pairwise ranking within each (B,A) over candidates when labels differ.
    diff_y = target.unsqueeze(-1) - target.unsqueeze(-2)
    diff_k = kappa.unsqueeze(-1) - kappa.unsqueeze(-2)
    pair_w = (weight.unsqueeze(-1) * weight.unsqueeze(-2) * (diff_y > 0.05)).to(kappa.dtype)
    if pair_w.sum() > 0:
        loss = loss + 0.25 * (F.softplus(-diff_k) * pair_w).sum() / pair_w.sum().clamp_min(1e-6)
    return loss


def total_mfrp_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict | None = None) -> dict[str, torch.Tensor]:
    cfg = cfg or {"loss": {}}
    loss_cfg = cfg.get("loss", {})
    losses: dict[str, torch.Tensor] = {}
    prefix = "support" if "support_branch_logits" in out else "scene"
    losses["mech"] = mechanism_nll(prefix, out, batch, cfg)
    losses["distill"] = distillation_loss(out) * float(loss_cfg.get("lambda_distill", 1.0))
    losses["geo"] = geometry_loss(out, batch) * float(loss_cfg.get("lambda_geo", 0.1))
    losses["cw"] = cw_loss(prefix, out, batch) * float(loss_cfg.get("lambda_cw", 1.0))
    losses["total"] = losses["mech"] + losses["distill"] + losses["geo"] + losses["cw"]
    return losses
