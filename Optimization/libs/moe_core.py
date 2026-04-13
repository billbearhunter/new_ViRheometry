#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
libs/moe_core.py
================
Shared core for optimize_1setup.py and optimize_2setups.py.

Previously every function below was copy-pasted in both files (~200 duplicate lines).
Both scripts now import from here instead.
"""

import os
import sys
import math
import json

import numpy as np
import torch
import joblib
import cma

try:
    import gpytorch
except ImportError as e:
    raise RuntimeError("gpytorch is required.") from e

try:
    import numpy.core as _np_core
    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = _np_core
except Exception:
    pass

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# ── Physical parameter bounds ──────────────────────────────────────────────────
MIN_N,       MAX_N       = 0.3,   1.0
MIN_ETA,     MAX_ETA     = 0.001, 300.0
MIN_SIGMA_Y, MAX_SIGMA_Y = 0.001, 400.0
GLOBAL_BOUNDS = {
    "n":       (MIN_N,       MAX_N),
    "eta":     (MIN_ETA,     MAX_ETA),
    "sigma_y": (MIN_SIGMA_Y, MAX_SIGMA_Y),
}
DTYPE = torch.float32   # match training dtype; overridden per-expert in load_expert_bundle


# ── GP model definitions ───────────────────────────────────────────────────────

class _OfflineSVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        v = gpytorch.variational.VariationalStrategy(
            self, inducing_points,
            gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0)),
            learn_inducing_locations=True,
        )
        super().__init__(v)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5) + gpytorch.kernels.LinearKernel()
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class _OfflineExactGP(gpytorch.models.ExactGP):
    def __init__(self, tx, ty, lik):
        super().__init__(tx, ty, lik)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5) + gpytorch.kernels.LinearKernel()
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


@dataclass
class ExpertBundle:
    cid:          int
    models:       List[torch.nn.Module]
    likes:        List[gpytorch.likelihoods.Likelihood]
    x_mean:       torch.Tensor
    x_scale:      torch.Tensor
    y_mean:       torch.Tensor
    y_scale:      torch.Tensor
    all_cols:     List[str]
    log_idx:      List[int]
    log_eps:      float
    poly_residual: Optional[dict] = None   # poly-residual correction dict or None
    target_mode:  str = "absolute"         # "absolute" or "diff"


# ── Gating feature vector ──────────────────────────────────────────────────────

def build_phi(y, W: float, H: float, eps: float = 1e-8) -> np.ndarray:
    """Build gating feature vector from flow distances and container geometry."""
    y = np.asarray(y, dtype=float).reshape(1, -1)
    y_norm = y / (y[:, [-1]] + eps)
    feats = [
        y_norm,
        np.diff(y_norm, axis=1),
        np.log(np.abs(y[:, -1]) + eps).reshape(-1, 1),
        np.hstack([
            np.log(np.sqrt(W * H) + eps).reshape(1, 1),
            np.log((W + eps) / (H + eps)).reshape(1, 1),
        ]),
    ]
    return np.hstack(feats)


# ── Expert selection strategies ────────────────────────────────────────────────

def get_adaptive_weights(
    gate_dict, phi, strategy="threshold",
    threshold=0.01, topk_hard=None,
    confidence_threshold=0.7, max_experts=5,
):
    """
    Compute expert IDs and weights from the GMM gate.

    Strategies
    ----------
    topk       : top-k experts with uniform weight
    threshold  : experts whose GMM probability >= threshold
    adaptive   : high-confidence → 1 expert; medium → threshold; low → top-8
    all        : all experts weighted by GMM probability
    """
    gmm    = gate_dict["gmm"]
    scaler = gate_dict.get("scaler")
    phi_in = phi.reshape(1, -1)
    if scaler is not None:
        phi_in = scaler.transform(phi_in)
    probs = gmm.predict_proba(phi_in)[0]

    if strategy == "all":
        expert_indices = np.arange(len(probs))
        weights = probs / np.sum(probs)

    elif strategy == "topk":
        k = topk_hard if topk_hard is not None else 2
        expert_indices = np.argsort(-probs)[:k]
        top_probs = probs[expert_indices]
        weights = top_probs / np.sum(top_probs)  # probability-weighted

    elif strategy == "threshold":
        mask = probs >= threshold
        if not np.any(mask):
            mask[np.argmax(probs)] = True
        expert_indices  = np.where(mask)[0]
        filtered_probs  = probs[mask]
        weights         = filtered_probs / np.sum(filtered_probs)
        if len(expert_indices) > max_experts:
            top_idx        = np.argsort(-filtered_probs)[:max_experts]
            expert_indices = expert_indices[top_idx]
            filtered_probs = filtered_probs[top_idx]
            weights        = filtered_probs / np.sum(filtered_probs)

    elif strategy == "adaptive":
        max_prob = np.max(probs)
        if max_prob > confidence_threshold:
            expert_indices = [np.argmax(probs)]
            weights        = [1.0]
        elif max_prob > 0.3:
            mask = probs >= 0.05
            if not np.any(mask):
                mask[np.argmax(probs)] = True
            expert_indices  = np.where(mask)[0]
            filtered_probs  = probs[mask]
            weights         = filtered_probs / np.sum(filtered_probs)
            if len(expert_indices) > max_experts:
                top_idx        = np.argsort(-filtered_probs)[:max_experts]
                expert_indices = expert_indices[top_idx]
                filtered_probs = filtered_probs[top_idx]
                weights        = filtered_probs / np.sum(filtered_probs)
        else:
            expert_indices = np.argsort(-probs)[:8]
            filtered_probs = probs[expert_indices]
            weights        = filtered_probs / np.sum(filtered_probs)

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    expert_ids = [int(idx) + 1 for idx in expert_indices]
    return expert_ids, np.array(weights, dtype=float)


# ── GP inference ───────────────────────────────────────────────────────────────

def _apply_poly_powers(X: np.ndarray, powers: np.ndarray) -> np.ndarray:
    """Reconstruct polynomial features from stored power matrix (no sklearn needed)."""
    out = np.ones((X.shape[0], powers.shape[0]), dtype=np.float64)
    for j, row in enumerate(powers):
        for k, p in enumerate(row):
            if p > 0:
                out[:, j] *= X[:, k] ** p
    return out


def predict_expert_batch(
    bundle: ExpertBundle,
    n_batch, eta_batch, s_batch,
    W: float, H: float, device,
) -> np.ndarray:
    """Run one expert's GP models on a batch of parameter vectors.

    Applies poly-residual correction if the bundle was trained with it.
    """
    batch_size = len(n_batch)
    if batch_size == 0:
        return np.array([])
    col_map = {c: i for i, c in enumerate(bundle.all_cols)}
    vals = np.zeros((batch_size, len(bundle.all_cols)), dtype=float)
    vals[:, col_map["n"]]       = n_batch
    vals[:, col_map["eta"]]     = eta_batch
    vals[:, col_map["sigma_y"]] = s_batch
    vals[:, col_map["width"]]   = W
    vals[:, col_map["height"]]  = H
    for j in bundle.log_idx:
        vals[:, j] = np.log(np.clip(vals[:, j] + bundle.log_eps, 1e-12, None))
    # Keep scaled inputs for poly residual (in numpy)
    vals_scaled = (vals - bundle.x_mean.cpu().numpy()) / bundle.x_scale.cpu().numpy()
    # Use the bundle's actual dtype (detected from checkpoint)
    bundle_dtype = bundle.x_mean.dtype
    xt = torch.tensor(vals_scaled, dtype=bundle_dtype, device=device)

    # GP mean predictions (scaled space)
    jitter = 1e-2 if bundle_dtype == torch.float32 else 1e-3
    preds_scaled = []
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
         gpytorch.settings.cholesky_jitter(jitter), \
         gpytorch.settings.max_cg_iterations(1000), \
         gpytorch.settings.cg_tolerance(0.01), \
         gpytorch.settings.max_preconditioner_size(100):
        for m, l in zip(bundle.models, bundle.likes):
            preds_scaled.append(l(m(xt)).mean.detach().cpu().numpy())
    Y_s = np.stack(preds_scaled, axis=1)   # (batch, 8) in scaled space

    # Apply poly residual correction in scaled space before inverse-scaling
    pr = bundle.poly_residual
    if pr is not None:
        powers = np.asarray(pr["powers"], dtype=int)
        coef   = np.asarray(pr["coef"],   dtype=np.float64)
        Xp     = _apply_poly_powers(vals_scaled, powers)
        Y_s    = Y_s + Xp @ coef.T

    # Inverse-scale: y_phys = Y_s * y_scale + y_mean
    y_mean  = bundle.y_mean.cpu().numpy()    # (1, 8)
    y_scale = bundle.y_scale.cpu().numpy()   # (1, 8)
    Y_phys  = (Y_s * y_scale + y_mean)

    # Differential target mode: D → clip(D, 0) → cumsum → Y
    if bundle.target_mode == "diff":
        Y_phys = np.maximum(Y_phys, 0.0)          # Dₖ ≥ 0 (physical: flow is monotonic)
        Y_phys = np.cumsum(Y_phys, axis=1)         # Y = cumsum(D)

    # Physical constraints (apply universally as safety net):
    # (a) flow distances are non-negative
    Y_phys = np.maximum(Y_phys, 0.0)
    # (b) flow distances are monotonically non-decreasing over time
    for i in range(1, Y_phys.shape[1]):
        Y_phys[:, i] = np.maximum(Y_phys[:, i], Y_phys[:, i - 1])

    return Y_phys.astype(float)


def soft_predict_batch(
    theta_batch, expert_bundles: Dict[int, ExpertBundle],
    expert_ids, weights,
    W: float, H: float, device,
) -> np.ndarray:
    """Weighted prediction from multiple experts for a batch of parameters."""
    batch_size = len(theta_batch)
    if batch_size == 0:
        return np.zeros((0, 1))
    n_batch   = np.array([t[0] for t in theta_batch])
    eta_batch = np.array([t[1] for t in theta_batch])
    s_batch   = np.array([t[2] for t in theta_batch])
    weighted_pred    = None
    valid_weights_sum = 0.0
    for cid, weight in zip(expert_ids, weights):
        if cid not in expert_bundles:
            continue
        bundle = expert_bundles[cid]
        try:
            pred_batch = predict_expert_batch(bundle, n_batch, eta_batch, s_batch, W, H, device)
            if weighted_pred is None:
                weighted_pred = np.zeros_like(pred_batch)
            weighted_pred     += weight * pred_batch
            valid_weights_sum += weight
        except Exception:
            continue
    if weighted_pred is None:
        raise ValueError("No valid experts for prediction")
    if valid_weights_sum > 0:
        weighted_pred /= valid_weights_sum
    return weighted_pred


# ── Parameter utilities ────────────────────────────────────────────────────────

def clamp_params(theta, bounds) -> list:
    eps = 1e-5
    return [
        float(np.clip(theta[0], bounds["n"][0]       + eps, bounds["n"][1]       - eps)),
        float(np.clip(theta[1], bounds["eta"][0]     + eps, bounds["eta"][1]     - eps)),
        float(np.clip(theta[2], bounds["sigma_y"][0] + eps, bounds["sigma_y"][1] - eps)),
    ]


def default_x0(bounds) -> list:
    """Log-space midpoint for eta/sigma_y (more appropriate than arithmetic)."""
    n0   = 0.5 * (bounds["n"][0] + bounds["n"][1])
    eta0 = math.exp(0.5 * (math.log(bounds["eta"][0]) + math.log(bounds["eta"][1])))
    sig0 = math.exp(0.5 * (
        math.log(max(bounds["sigma_y"][0], 1e-6)) + math.log(bounds["sigma_y"][1])
    ))
    return [n0, eta0, sig0]


# ── Feasibility check ──────────────────────────────────────────────────────────

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


# ── Model loading ──────────────────────────────────────────────────────────────

def _safe_torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def maybe_load_joblib(path: Optional[str]):
    if path and os.path.exists(path):
        return joblib.load(path)
    return None


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _detect_ckpt_dtype(ckpt) -> torch.dtype:
    """Detect the dtype used during training from checkpoint state dicts."""
    msds = ckpt.get("models", [])
    if msds:
        for v in msds[0].values():
            if torch.is_tensor(v) and v.is_floating_point():
                return v.dtype
    tx = ckpt.get("train_x")
    if tx is not None and torch.is_tensor(tx):
        return tx.dtype
    return DTYPE


def load_expert_bundle(path: str, device) -> ExpertBundle:
    """Load a trained GP expert from a .pt checkpoint file."""
    ckpt = _safe_torch_load(path, map_location=device)
    cid  = int(ckpt.get("cid", -1))
    dtype = _detect_ckpt_dtype(ckpt)

    def get_v(d, k, alt_k, root, root_k):
        v = None
        if d:    v = d.get(k)
        if v is None and d:    v = d.get(alt_k)
        if v is None and root: v = root.get(root_k)
        return v

    def safe_tensor(data, dt, dev):
        if torch.is_tensor(data):
            return data.detach().clone().to(dtype=dt, device=dev)
        return torch.tensor(data, dtype=dt, device=dev)

    xs, ys = ckpt.get("x_scaler", {}), ckpt.get("y_scaler", {})
    xm  = get_v(xs, "mean",  "mean_",  ckpt, "X_mean")
    xsc = get_v(xs, "scale", "scale_", ckpt, "X_scale")
    ym  = get_v(ys, "mean",  "mean_",  ckpt, "Y_mean")
    ysc = get_v(ys, "scale", "scale_", ckpt, "Y_scale")
    if xm is None:
        raise ValueError(f"Missing X_mean in {path}")

    x_mean  = safe_tensor(xm,  dtype, device).view(1, -1)
    x_scale = safe_tensor(xsc, dtype, device).view(1, -1)
    y_mean  = safe_tensor(ym,  dtype, device).view(1, -1)
    y_scale = safe_tensor(ysc, dtype, device).view(1, -1)

    models, likes = [], []
    msds = ckpt.get("models")
    lsds = ckpt.get("likes") or ckpt.get("likelihoods")
    is_exact = (ckpt.get("gp_kind", "svgp") == "exact")
    inducing = None
    if not is_exact:
        inducing = ckpt.get("inducing") or ckpt.get("inducing_points")

    if is_exact:
        tx = safe_tensor(ckpt["train_x"], dtype, device)
        ty = safe_tensor(ckpt["train_y"], dtype, device)
        for i in range(len(msds)):
            l = gpytorch.likelihoods.GaussianLikelihood().to(device, dtype)
            l.load_state_dict(lsds[i])
            m = _OfflineExactGP(tx, ty[:, i], l).to(device, dtype)
            m.load_state_dict(msds[i])
            m.eval(); l.eval()
            models.append(m); likes.append(l)
    else:
        for i in range(len(msds)):
            ip = safe_tensor(inducing[i], dtype, device)
            m  = _OfflineSVGPModel(ip).to(device, dtype)
            m.load_state_dict(msds[i])
            l = gpytorch.likelihoods.GaussianLikelihood().to(device, dtype)
            l.load_state_dict(lsds[i])
            m.eval(); l.eval()
            models.append(m); likes.append(l)

    all_cols      = xs.get("all_cols", ["n", "eta", "sigma_y", "width", "height"])
    log_cols      = set(xs.get("log_cols", ["eta", "sigma_y"]))
    log_idx       = [i for i, c in enumerate(all_cols) if c in log_cols]
    poly_residual = ckpt.get("poly_residual")   # None if not trained
    target_mode   = ckpt.get("target_mode", "absolute")   # backward compat
    return ExpertBundle(cid, models, likes, x_mean, x_scale, y_mean, y_scale,
                        all_cols, log_idx, 1e-6, poly_residual, target_mode)


# ── CMA-ES runner ──────────────────────────────────────────────────────────────

def _to_log_space(theta, bounds):
    """Transform [n, eta, sigma_y] to [n, log(eta), log(sigma_y)] for CMA-ES."""
    eta_lo = max(bounds["eta"][0], 1e-9)
    sig_lo = max(bounds["sigma_y"][0], 1e-9)
    return [theta[0], math.log(max(theta[1], eta_lo)), math.log(max(theta[2], sig_lo))]


def _from_log_space(z, bounds):
    """Transform [n, log(eta), log(sigma_y)] back to [n, eta, sigma_y]."""
    return [z[0], math.exp(z[1]), math.exp(z[2])]


def run_cmaes(
    batch_loss_fn, bounds,
    x0=None, sigma0=0.5, popsize=16, maxiter=700,
    seed=42, verb_disp=1, record_iter_times=False,
):
    """
    Run CMA-ES optimization in log-space for eta and sigma_y.

    CMA-ES searches in [n, log(eta), log(sigma_y)] space for better
    coverage of multi-order-of-magnitude parameters. The batch_loss_fn
    still receives parameters in original space [n, eta, sigma_y].

    Parameters
    ----------
    batch_loss_fn      : callable(list[theta]) -> list[float]
    record_iter_times  : if True, also return per-iteration wall-clock times

    Returns
    -------
    (theta_best, loss_best, loss_history)           if record_iter_times=False
    (theta_best, loss_best, loss_history, iter_times) if record_iter_times=True
    """
    import time
    if x0 is None:
        x0 = default_x0(bounds)
    x0 = clamp_params(x0, bounds)

    # Transform to log-space for CMA-ES
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
        # Transform back to original space for loss evaluation
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
