"""
surrogate/scalers.py
====================
Input/output scalers shared by training and inference.

Previously duplicated in:
  - DataPipeline/moe_utils.py  (LogStandardInputScaler, TargetScaler)
  - Optimization/libs/moe_core.py (inline scaling in load_expert_bundle / predict_expert_batch)
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

from .config import LOG_INPUTS, INPUT_COLS


class LogStandardInputScaler:
    """Log-transform specified columns, then StandardScale.

    Handles multi-order-of-magnitude inputs (eta, sigma_y) correctly.
    Serialisable to / from a plain dict for checkpoint storage.
    """
    def __init__(self, log_cols=LOG_INPUTS, all_cols=INPUT_COLS, eps: float = 1e-6):
        self.log_cols = list(log_cols)
        self.all_cols = list(all_cols)
        self.eps      = float(eps)
        self.log_idx  = [i for i, c in enumerate(self.all_cols) if c in set(self.log_cols)]
        self.scaler   = StandardScaler()

    def _log_transform(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        for j in self.log_idx:
            X[:, j] = np.log(X[:, j] + self.eps)
        return X

    def fit(self, X: np.ndarray) -> "LogStandardInputScaler":
        self.scaler.fit(self._log_transform(X))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(self._log_transform(X))

    def to_dict(self) -> dict:
        return {
            "mean":     self.scaler.mean_,
            "scale":    self.scaler.scale_,
            "log_cols": self.log_cols,
            "all_cols": self.all_cols,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LogStandardInputScaler":
        obj = cls(log_cols=d["log_cols"], all_cols=d["all_cols"])
        obj.scaler.mean_  = np.asarray(d["mean"])
        obj.scaler.scale_ = np.asarray(d["scale"])
        return obj


class TargetScaler:
    """Simple StandardScaler wrapper for output columns."""
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, Y: np.ndarray) -> "TargetScaler":
        self.scaler.fit(Y)
        return self

    def transform(self, Y: np.ndarray) -> np.ndarray:
        return self.scaler.transform(Y)

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(Y)

    def to_dict(self) -> dict:
        return {"mean": self.scaler.mean_, "scale": self.scaler.scale_}

    @classmethod
    def from_dict(cls, d: dict) -> "TargetScaler":
        obj = cls()
        obj.scaler.mean_  = np.asarray(d["mean"])
        obj.scaler.scale_ = np.asarray(d["scale"])
        return obj
