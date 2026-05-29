from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from .estimators import mechanism_estimates


@dataclass
class SelectionResult:
    selected_index: int | torch.Tensor
    fallback_used: bool | torch.Tensor
    feasible_mask: np.ndarray | torch.Tensor
    rho_mech_cal: np.ndarray | torch.Tensor
    active_violations: dict

    def __getitem__(self, key):
        return getattr(self, key)

    def keys(self):
        return self.__dict__.keys()


def _legacy_select(candidates, rho, nu, gamma, *, alpha, q_beta, nu_bar, gamma_bar):
    rho = np.asarray(rho, dtype=float) + float(q_beta)
    nu = np.asarray(nu, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    valid = np.asarray([getattr(c, "valid", True) for c in candidates], dtype=bool)
    costs = np.asarray([getattr(c, "nominal_cost", 0.0) for c in candidates], dtype=float)
    risk_bad = rho > alpha
    nu_bad = nu > nu_bar
    gamma_bad = np.zeros_like(risk_bad, dtype=bool) if gamma_bar is None else gamma > gamma_bar
    feasible = valid & ~risk_bad & ~nu_bad & ~gamma_bad
    if feasible.any():
        idx = int(np.argmin(np.where(feasible, costs, np.inf)))
        fallback = False
    else:
        idx = int(np.argmin(np.where(valid, costs + 10 * rho + nu + gamma, np.inf)))
        fallback = True
    return SelectionResult(idx, fallback, feasible, rho, {"risk": bool(risk_bad.any()), "uncertainty": bool(nu_bad.any()), "boundary": bool(gamma_bad.any()), "coercion": False})


def select_mechanism_feasible(*args, alpha: float = 0.05, q_beta: float = 0.0, nu_bar: float = 0.3, gamma_bar: float | None = None, prefix: str = "scene"):
    # Legacy signature: (candidates, rho, nu, gamma, ...)
    if len(args) >= 4 and not isinstance(args[0], dict):
        return _legacy_select(args[0], args[1], args[2], args[3], alpha=alpha, q_beta=q_beta, nu_bar=nu_bar, gamma_bar=gamma_bar)

    out, batch = args[0], args[1]
    est = mechanism_estimates(out, prefix=prefix)
    rho_cal = (est["rho_mech"] + q_beta).clamp(max=1.0)
    valid = batch.get("candidate_valid")
    if valid is None:
        valid = torch.ones_like(rho_cal, dtype=torch.bool)
    valid = valid.bool().to(rho_cal.device)
    risk_bad = rho_cal > alpha
    nu_bad = est["nu"] > nu_bar
    gamma = est.get("gamma", torch.zeros_like(rho_cal))
    gamma_bad = torch.zeros_like(risk_bad) if gamma_bar is None else gamma > gamma_bar
    kappa_bad_i = est["kappa_i"] > 0.5
    coercion_bad = kappa_bad_i.any(dim=1)
    feasible = (~risk_bad) & (~nu_bad) & (~gamma_bad) & valid
    nominal = batch.get("nominal_cost", torch.zeros_like(rho_cal)).to(rho_cal.device)
    big = torch.full_like(nominal, 1e6)
    feas_cost = torch.where(feasible, nominal, big)
    idx = torch.argmin(feas_cost, dim=-1)
    fallback = ~feasible.any(dim=-1)
    fallback_score = nominal + 10.0 * rho_cal + est["nu"] + gamma
    fb_idx = torch.argmin(torch.where(valid, fallback_score, big), dim=-1)
    idx = torch.where(fallback, fb_idx, idx)
    result = {"selected_index": idx, "fallback_used": fallback, "feasible_mask": feasible, "rho_mech_cal": rho_cal, "active_violations": {"risk": risk_bad, "uncertainty": nu_bad, "boundary": gamma_bad, "coercion": coercion_bad}, **est}
    return result
