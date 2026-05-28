from __future__ import annotations

import torch
from .estimators import mechanism_estimates


def select_mechanism_feasible(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    alpha: float = 0.05,
    q_beta: float = 0.0,
    nu_bar: float = 0.3,
    gamma_bar: float | None = None,
    prefix: str = "scene",
) -> dict[str, torch.Tensor]:
    est = mechanism_estimates(out, prefix=prefix)
    rho_cal = (est["rho_mech"] + q_beta).clamp(max=1.0)
    valid = batch.get("candidate_valid")
    if valid is None:
        valid = torch.ones_like(rho_cal, dtype=torch.bool)
    feasible = (rho_cal <= alpha) & (est["nu"] <= nu_bar) & valid.bool().to(rho_cal.device)
    nominal = batch.get("nominal_cost", torch.zeros_like(rho_cal)).to(rho_cal.device)
    big = torch.full_like(nominal, 1e6)
    feas_cost = torch.where(feasible, nominal, big)
    idx = torch.argmin(feas_cost, dim=-1)
    fallback = ~feasible.any(dim=-1)
    fallback_score = nominal + 10.0 * rho_cal + est["nu"]
    fb_idx = torch.argmin(torch.where(valid.bool().to(rho_cal.device), fallback_score, big), dim=-1)
    idx = torch.where(fallback, fb_idx, idx)
    return {"selected_index": idx, "fallback_used": fallback, "feasible_mask": feasible, "rho_mech_cal": rho_cal, **est}
