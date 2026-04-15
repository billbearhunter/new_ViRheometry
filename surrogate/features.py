"""
surrogate/features.py
=====================
Feature engineering for MoE gating and GP inputs.

Previously duplicated in:
  - DataPipeline/moe_utils.py  (build_phi, build_input_features)
  - Optimization/libs/moe_core.py  (build_phi, build_input_phi)
  - Optimization/soft_interpolate.py  (build_phi)
"""

import numpy as np


def build_phi(df_or_y, W=None, H=None, eps: float = 1e-8) -> np.ndarray:
    """Build the 18-dim gating feature vector phi.

    Accepts either:
    - a DataFrame with columns x_01..x_08, width, height  (training mode)
    - a 1-D array y (8 values) + W, H scalars             (inference mode)

    Returns: (N, 18) array
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
    Y_norm = Y / denom                                        # (8)
    dY     = np.diff(Y_norm, axis=1)                          # (7)
    scale  = np.log(np.abs(Y[:, -1]) + eps).reshape(-1, 1)    # (1)
    g1     = np.log(np.sqrt(W_arr * H_arr) + eps).reshape(-1, 1)   # (1)
    g2     = np.log((W_arr + eps) / (H_arr + eps)).reshape(-1, 1)   # (1)

    return np.hstack([Y_norm, dY, scale, g1, g2])   # 8+7+1+1+1 = 18


def build_input_features(df_or_params, W=None, H=None, eps: float = 1e-8) -> np.ndarray:
    """Build 5D input feature vector for input-space GMM gating.

    Features: (n, log(eta), log(sigma_y), W, H)

    Accepts either:
    - a DataFrame with columns n, eta, sigma_y, width, height  (training mode)
    - an array of shape (N, 3) with [n, eta, sigma_y] + W, H   (inference mode)

    Returns: (N, 5) array
    """
    import pandas as pd

    if isinstance(df_or_params, pd.DataFrame):
        df = df_or_params
        return np.column_stack([
            df["n"].values.astype(np.float64),
            np.log(df["eta"].values.astype(np.float64) + eps),
            np.log(df["sigma_y"].values.astype(np.float64) + eps),
            df["width"].values.astype(np.float64),
            df["height"].values.astype(np.float64),
        ])
    else:
        p = np.asarray(df_or_params, dtype=np.float64)
        if p.ndim == 1:
            p = p.reshape(1, -1)
        return np.column_stack([
            p[:, 0],
            np.log(p[:, 1] + eps),
            np.log(p[:, 2] + eps),
            np.full(len(p), W, dtype=np.float64),
            np.full(len(p), H, dtype=np.float64),
        ])
