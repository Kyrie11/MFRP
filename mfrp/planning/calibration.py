"""Split calibration for MFRP mechanism risk."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class CalibrationArtifact:
    beta: float
    q_beta: float
    num_examples: int
    method: str = "split_residual_quantile"

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "CalibrationArtifact":
        return CalibrationArtifact(**json.loads(Path(path).read_text(encoding="utf-8")))


def fit_split_calibration(rho_mech_hat: np.ndarray, violation_truth: np.ndarray, beta: float = 0.1) -> CalibrationArtifact:
    rho = np.asarray(rho_mech_hat, dtype=np.float64).reshape(-1)
    truth = np.asarray(violation_truth, dtype=np.float64).reshape(-1)
    if rho.shape != truth.shape:
        raise ValueError("rho_mech_hat and violation_truth must have the same shape")
    if not (0.0 < float(beta) < 1.0):
        raise ValueError("beta must be in (0, 1)")
    residual = truth - rho
    if len(residual) == 0:
        raise ValueError("cannot fit split calibration on zero examples")
    if not np.isfinite(residual).all():
        raise ValueError("calibration residuals contain non-finite values")
    q = float(np.quantile(residual, 1.0 - beta, method="higher"))
    return CalibrationArtifact(float(beta), float(max(0.0, q)), int(len(residual)))


def apply_calibration(rho_mech_hat, artifact: CalibrationArtifact):
    return np.minimum(1.0, np.asarray(rho_mech_hat) + artifact.q_beta)
