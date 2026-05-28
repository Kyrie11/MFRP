"""Mechanism-feasible selector and conservative fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SelectionResult:
    index: int
    candidate_id: str
    fallback_used: bool
    feasible_mask: np.ndarray
    active_violations: dict[str, bool]
    scores: np.ndarray


def mechanism_feasible_mask(rho_mech_cal: np.ndarray, uncertainty: np.ndarray, sensitivity: np.ndarray, candidate_valid: np.ndarray, alpha: float, nu_bar: float, gamma_bar: float) -> np.ndarray:
    return (rho_mech_cal <= alpha) & (uncertainty <= nu_bar) & (sensitivity <= gamma_bar) & candidate_valid.astype(bool)


def select_mechanism_feasible(candidates: list[Any], rho_mech_cal: np.ndarray, uncertainty: np.ndarray, sensitivity: np.ndarray, candidate_valid: np.ndarray | None = None, alpha: float = 0.05, nu_bar: float = 0.3, gamma_bar: float = 2.0, lambda_risk: float = 10.0, lambda_unc: float = 1.0, lambda_gamma: float = 1.0) -> SelectionResult:
    n = len(candidates)
    if candidate_valid is None:
        candidate_valid = np.ones(n, dtype=bool)
    rho = np.asarray(rho_mech_cal, dtype=float).reshape(n)
    nu = np.asarray(uncertainty, dtype=float).reshape(n)
    gam = np.asarray(sensitivity, dtype=float).reshape(n)
    valid = np.asarray(candidate_valid, dtype=bool).reshape(n)
    nominal = np.asarray([float(getattr(c, "nominal_cost", 0.0)) for c in candidates], dtype=float)
    feasible = mechanism_feasible_mask(rho, nu, gam, valid, alpha, nu_bar, gamma_bar)
    if feasible.any():
        masked = np.where(feasible, nominal, np.inf)
        idx = int(np.argmin(masked))
        return SelectionResult(idx, str(getattr(candidates[idx], "candidate_id", idx)), False, feasible, {}, masked)
    scores = nominal + lambda_risk * np.maximum(0.0, rho - alpha) + lambda_unc * np.maximum(0.0, nu - nu_bar) + lambda_gamma * np.maximum(0.0, gam - gamma_bar) + np.where(valid, 0.0, 1e6)
    idx = int(np.argmin(scores))
    active = {"risk": bool(rho[idx] > alpha), "uncertainty": bool(nu[idx] > nu_bar), "sensitivity": bool(gam[idx] > gamma_bar), "candidate_invalid": bool(not valid[idx])}
    return SelectionResult(idx, str(getattr(candidates[idx], "candidate_id", idx)), True, feasible, active, scores)
