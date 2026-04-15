"""
libs/moe_core.py
================
Backward-compatible re-export from surrogate package.

All core MoE functionality now lives in:
  - surrogate/config.py     (bounds, constants)
  - surrogate/features.py   (build_phi, build_input_features)
  - surrogate/gating.py     (get_adaptive_weights, hierarchical_get_weights)
  - surrogate/expert_io.py  (ExpertBundle, load_expert_bundle, load_all_experts)
  - surrogate/predict.py    (predict_expert_batch, soft_predict_batch, etc.)
"""

import os
import sys
import math
import json

import numpy as np
import torch
import joblib

# Ensure repo root is on path
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Re-export from surrogate ──────────────────────────────────────────────────

from surrogate.config import (  # noqa
    MIN_N, MAX_N, MIN_ETA, MAX_ETA, MIN_SIGMA_Y, MAX_SIGMA_Y,
    PARAM_BOUNDS,
)
GLOBAL_BOUNDS = {
    "n":       (MIN_N,       MAX_N),
    "eta":     (MIN_ETA,     MAX_ETA),
    "sigma_y": (MIN_SIGMA_Y, MAX_SIGMA_Y),
}
DTYPE = torch.float32

from surrogate.models import SingleOutputExactGP as _OfflineExactGP  # noqa
from surrogate.models import SingleOutputSVGP as _OfflineSVGPModel  # noqa

from surrogate.features import (  # noqa
    build_phi,
    build_input_features as build_input_phi,
)

from surrogate.gating import (  # noqa
    get_adaptive_weights,
    hierarchical_get_weights,
)

from surrogate.expert_io import (  # noqa
    ExpertBundle,
    load_expert_bundle,
    load_all_experts,
    maybe_load_joblib,
    load_json,
)

from surrogate.predict import (  # noqa
    predict_expert_batch,
    predict_expert_variance,
    soft_predict_batch,
    dynamic_soft_predict_batch,
    _apply_poly_powers,
)

# ── Feasibility check ────────────────────────────────────────────────────────

try:
    from libs.compare_loss import mat_hw_to_PL
except ImportError:
    def mat_hw_to_PL(*args, **kwargs): return 1.0, 1.0


def check_feasibility(theta, Hcm: float, Wcm: float, base: float = 1e6) -> float:
    n, eta, sigma_y = float(theta[0]), float(theta[1]), float(theta[2])
    try:
        P0, L0 = mat_hw_to_PL(eta * 0.1, n, sigma_y * 0.1, Hcm, Wcm)
        P, L   = P0 / 10.0, L0 * 100.0
    except Exception:
        return base * 10.0
    if not (np.isfinite(P) and np.isfinite(L) and P > 0 and L > 0):
        return base
    if (P * L - sigma_y) <= 0:
        return base * 2.0
    try:
        W_val = P * (L - sigma_y / P) / eta
    except Exception:
        return base * 10.0
    if not (np.isfinite(W_val) and W_val > 0):
        return base * 3.0
    return 0.0


# ── Parameter utilities ──────────────────────────────────────────────────────

def clamp_params(theta, bounds) -> list:
    eps = 1e-5
    return [
        float(np.clip(theta[0], bounds["n"][0]       + eps, bounds["n"][1]       - eps)),
        float(np.clip(theta[1], bounds["eta"][0]     + eps, bounds["eta"][1]     - eps)),
        float(np.clip(theta[2], bounds["sigma_y"][0] + eps, bounds["sigma_y"][1] - eps)),
    ]


def default_x0(bounds) -> list:
    """Log-space midpoint for eta/sigma_y."""
    n0   = 0.5 * (bounds["n"][0] + bounds["n"][1])
    eta0 = math.exp(0.5 * (math.log(bounds["eta"][0]) + math.log(bounds["eta"][1])))
    sig0 = math.exp(0.5 * (
        math.log(max(bounds["sigma_y"][0], 1e-6)) + math.log(bounds["sigma_y"][1])
    ))
    return [n0, eta0, sig0]


# ── CMA-ES runner ────────────────────────────────────────────────────────────

def _to_log_space(theta, bounds):
    eta_lo = max(bounds["eta"][0], 1e-9)
    sig_lo = max(bounds["sigma_y"][0], 1e-9)
    return [theta[0], math.log(max(theta[1], eta_lo)), math.log(max(theta[2], sig_lo))]


def _from_log_space(z, bounds):
    return [z[0], math.exp(z[1]), math.exp(z[2])]


def run_cmaes(
    batch_loss_fn, bounds,
    x0=None, sigma0=0.5, popsize=16, maxiter=700,
    seed=42, verb_disp=1, record_iter_times=False,
):
    """Run CMA-ES optimization in log-space for eta and sigma_y."""
    import time
    import cma

    if x0 is None:
        x0 = default_x0(bounds)
    x0 = clamp_params(x0, bounds)

    z0 = _to_log_space(x0, bounds)
    eta_lo = max(bounds["eta"][0], 1e-9)
    sig_lo = max(bounds["sigma_y"][0], 1e-9)
    log_bounds = [
        [bounds["n"][0], math.log(eta_lo), math.log(sig_lo)],
        [bounds["n"][1], math.log(bounds["eta"][1]), math.log(bounds["sigma_y"][1])],
    ]

    opts = {
        "bounds":        log_bounds,
        "seed":          seed,
        "popsize":       popsize,
        "verbose":       verb_disp,
        "maxiter":       maxiter,
        "tolx":          1e-20,
        "tolfun":        1e-20,
        "tolstagnation": maxiter * 2,
    }
    es           = cma.CMAEvolutionStrategy(z0, sigma0, opts)
    loss_history = []
    iter_times   = []

    while not es.stop():
        t0 = time.time()
        z_sols = es.ask()
        orig_sols = [clamp_params(_from_log_space(z, bounds), bounds) for z in z_sols]
        losses    = batch_loss_fn(orig_sols)
        es.tell(z_sols, losses)
        if verb_disp > 0:
            es.disp()
        loss_history.append(es.result.fbest)
        iter_times.append(time.time() - t0)

    res = es.result
    theta_best = clamp_params(_from_log_space(res.xbest, bounds), bounds)
    best = [float(x) for x in theta_best], float(res.fbest), loss_history
    if record_iter_times:
        return (*best, iter_times)
    return best
