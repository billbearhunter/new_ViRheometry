"""
DataPipeline/moe_utils.py
=========================
Shared GP model definitions, scalers, and feature engineering.
Compatible with the .pt checkpoints produced by newGP training code.
"""

import numpy as np
import torch
import gpytorch
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.likelihoods import GaussianLikelihood
from sklearn.preprocessing import StandardScaler

from dp_config import INPUT_COLS, OUTPUT_COLS, LOG_INPUTS


# ── Device / dtype ─────────────────────────────────────────────────────────────
DTYPE  = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Scalers ────────────────────────────────────────────────────────────────────

class LogStandardInputScaler:
    """Log-transform specified columns, then StandardScale.
    Handles multi-order-of-magnitude inputs (η, σ_y) correctly.
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
        self.scaler.fit(Y); return self

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


# ── GP model classes ───────────────────────────────────────────────────────────

class SingleOutputExactGP(gpytorch.models.ExactGP):
    """Exact GP for small clusters (N ≤ EXACT_THRESHOLD)."""
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
            + gpytorch.kernels.LinearKernel()
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class SingleOutputSVGP(gpytorch.models.ApproximateGP):
    """Sparse Variational GP for large clusters (N > EXACT_THRESHOLD)."""
    def __init__(self, inducing_points):
        q = inducing_points.size(0)
        vd = CholeskyVariationalDistribution(q)
        vs = VariationalStrategy(self, inducing_points, vd, learn_inducing_locations=True)
        super().__init__(vs)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
            + gpytorch.kernels.LinearKernel()
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


# ── Feature engineering ────────────────────────────────────────────────────────

def build_phi(df_or_y, W=None, H=None, eps: float = 1e-8) -> np.ndarray:
    """Build the 18-dim gating feature vector φ.

    Accepts either:
    - a DataFrame with columns x_01..x_08, width, height  (training mode)
    - a 1-D array y (8 values) + W, H scalars             (inference mode)
    """
    import pandas as pd

    if isinstance(df_or_y, pd.DataFrame):
        df = df_or_y
        Y  = df[[f"x_{i:02d}" for i in range(1, 9)]].to_numpy(dtype=np.float64)
        W_arr = df["width"].to_numpy(dtype=np.float64)
        H_arr = df["height"].to_numpy(dtype=np.float64)
    else:
        Y     = np.asarray(df_or_y, dtype=np.float64).reshape(1, -1)
        W_arr = np.full(1, W, dtype=np.float64)
        H_arr = np.full(1, H, dtype=np.float64)

    denom  = Y[:, [-1]] + eps
    Y_norm = Y / denom                         # normalised shape (8)
    dY     = np.diff(Y_norm, axis=1)           # increments (7)
    scale  = np.log(np.abs(Y[:, -1]) + eps).reshape(-1, 1)   # magnitude (1)
    g1     = np.log(np.sqrt(W_arr * H_arr) + eps).reshape(-1, 1)   # area (1)
    g2     = np.log((W_arr + eps) / (H_arr + eps)).reshape(-1, 1)  # aspect (1)

    return np.hstack([Y_norm, dY, scale, g1, g2])   # 8+7+1+1+1 = 18


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _safe_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_expert(path: str, device=None):
    """Load a trained expert from a .pt checkpoint.

    Returns (gp_kind, models, likes, x_scaler_dict, y_scaler_dict, poly_residual).
    poly_residual is None if not trained, else a dict with keys:
    {degree, powers, coef, intercept, alpha}.
    """
    device = device or DEVICE
    state  = _safe_load(path, device)

    gp_kind = state.get("gp_kind", "svgp")
    models, likes = [], []

    if gp_kind == "exact":
        tx = state["train_x"].to(device, DTYPE)
        ty = state["train_y"].to(device, DTYPE)
        for i, msd in enumerate(state["models"]):
            lk = GaussianLikelihood().to(device, DTYPE)
            lk.load_state_dict(state["likes"][i])
            m  = SingleOutputExactGP(tx, ty[:, i].contiguous(), lk).to(device, DTYPE)
            m.load_state_dict(msd)
            m.eval(); lk.eval()
            models.append(m); likes.append(lk)
    else:
        inducing = state.get("inducing")
        for i, msd in enumerate(state["models"]):
            ip = torch.tensor(np.asarray(inducing[i]), device=device, dtype=DTYPE)
            m  = SingleOutputSVGP(ip).to(device, DTYPE)
            lk = GaussianLikelihood().to(device, DTYPE)
            m.load_state_dict(msd)
            lk.load_state_dict(state["likes"][i])
            m.eval(); lk.eval()
            models.append(m); likes.append(lk)

    return (
        gp_kind,
        models,
        likes,
        state["x_scaler"],
        state["y_scaler"],
        state.get("poly_residual"),
    )


def predict_with_expert(models, likes, x_scaler_dict, y_scaler_dict,
                        poly_residual, X_raw: np.ndarray) -> np.ndarray:
    """Full inference including optional poly residual correction.

    X_raw : (N, 5)  physical-scale input  [n, eta, sigma_y, width, height]
    Returns: (N, 8) predicted displacements
    """
    # Scale inputs
    xs = LogStandardInputScaler.from_dict(x_scaler_dict)
    X_s = xs.transform(X_raw)

    # GP mean prediction
    X_t = torch.tensor(X_s, dtype=DTYPE, device=DEVICE)
    preds = []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for m, lk in zip(models, likes):
            preds.append(lk(m(X_t)).mean.detach().cpu().numpy())
    Y_s = np.stack(preds, axis=1)   # (N, 8) scaled

    # Apply poly residual correction if available
    if poly_residual is not None:
        powers = np.asarray(poly_residual["powers"], dtype=int)
        coef   = np.asarray(poly_residual["coef"],   dtype=np.float64)
        Xp     = _apply_poly_powers(X_s, powers)
        Y_s    = Y_s + Xp @ coef.T

    # Inverse-scale outputs
    ys = TargetScaler.from_dict(y_scaler_dict)
    return ys.inverse_transform(Y_s)


def _apply_poly_powers(X: np.ndarray, powers: np.ndarray) -> np.ndarray:
    """Reconstruct polynomial features from stored power matrix."""
    out = np.ones((X.shape[0], powers.shape[0]), dtype=np.float64)
    for j, row in enumerate(powers):
        for k, p in enumerate(row):
            if p > 0:
                out[:, j] *= X[:, k] ** p
    return out
