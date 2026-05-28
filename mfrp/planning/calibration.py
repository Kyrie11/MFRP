from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np


@dataclass
class SplitCalibration:
    beta: float
    q_beta: float
    alpha: float = 0.05

    def apply(self, rho_hat):
        return np.minimum(1.0, np.asarray(rho_hat) + self.q_beta)

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @staticmethod
    def from_json(path: str | Path) -> "SplitCalibration":
        return SplitCalibration(**json.loads(Path(path).read_text(encoding="utf-8")))


def fit_split_calibration(rho_hat: np.ndarray, violation_truth: np.ndarray, *, beta: float = 0.1, alpha: float = 0.05) -> SplitCalibration:
    rho_hat = np.asarray(rho_hat, dtype=np.float64).reshape(-1)
    truth = np.asarray(violation_truth, dtype=np.float64).reshape(-1)
    mask = np.isfinite(rho_hat) & np.isfinite(truth)
    if mask.sum() == 0:
        raise ValueError("empty calibration set")
    residual = truth[mask].clip(0, 1) - rho_hat[mask].clip(0, 1)
    q = float(np.quantile(residual, 1.0 - beta, method="higher"))
    return SplitCalibration(beta=float(beta), q_beta=max(0.0, q), alpha=float(alpha))
