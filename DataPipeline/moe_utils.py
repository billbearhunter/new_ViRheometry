"""
DataPipeline/moe_utils.py
=========================
Backward-compatible re-export from surrogate package.

All shared GP models, scalers, and feature engineering now live in:
  - surrogate/models.py    (DEVICE, DTYPE, SingleOutputExactGP, SingleOutputSVGP)
  - surrogate/scalers.py   (LogStandardInputScaler, TargetScaler)
  - surrogate/features.py  (build_phi, build_input_features)
  - surrogate/expert_io.py (load_expert, predict_with_expert)
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Re-export everything that was previously defined here
from surrogate.models import DEVICE, DTYPE, SingleOutputExactGP, SingleOutputSVGP  # noqa
from surrogate.scalers import LogStandardInputScaler, TargetScaler  # noqa
from surrogate.features import build_phi, build_input_features  # noqa
from surrogate.expert_io import (  # noqa
    load_expert_for_training as load_expert,
    _safe_torch_load as _safe_load,
)
from surrogate.predict import _apply_poly_powers  # noqa


def predict_with_expert(models, likes, x_scaler_dict, y_scaler_dict,
                        poly_residual, X_raw):
    """Full inference including optional poly residual correction.

    X_raw : (N, 5) physical-scale input [n, eta, sigma_y, width, height]
    Returns: (N, 8) predicted displacements
    """
    import numpy as np
    import torch
    import gpytorch

    xs = LogStandardInputScaler.from_dict(x_scaler_dict)
    X_s = xs.transform(X_raw)

    X_t = torch.tensor(X_s, dtype=DTYPE, device=DEVICE)
    preds = []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for m, lk in zip(models, likes):
            preds.append(lk(m(X_t)).mean.detach().cpu().numpy())
    Y_s = np.stack(preds, axis=1)

    if poly_residual is not None:
        powers = np.asarray(poly_residual["powers"], dtype=int)
        coef   = np.asarray(poly_residual["coef"],   dtype=np.float64)
        Xp     = _apply_poly_powers(X_s, powers)
        Y_s    = Y_s + Xp @ coef.T

    ys = TargetScaler.from_dict(y_scaler_dict)
    return ys.inverse_transform(Y_s)
