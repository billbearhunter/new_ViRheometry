"""
surrogate/predict.py
====================
GP expert prediction (batch inference with poly-residual correction).

Previously in Optimization/libs/moe_core.py (predict_expert_batch,
predict_expert_variance, soft_predict_batch, dynamic_soft_predict_batch).
"""

from typing import Dict, Tuple

import numpy as np
import torch
import gpytorch

from .expert_io import ExpertBundle
from .features import build_input_features
from .gating import get_adaptive_weights


def _apply_poly_powers(X: np.ndarray, powers: np.ndarray) -> np.ndarray:
    """Reconstruct polynomial features from stored power matrix."""
    out = np.ones((X.shape[0], powers.shape[0]), dtype=np.float64)
    for j, row in enumerate(powers):
        for k, p in enumerate(row):
            if p > 0:
                out[:, j] *= X[:, k] ** p
    return out


def _cg_context():
    """Context managers for CG-based solves."""
    return (
        gpytorch.settings.max_cg_iterations(1000),
        gpytorch.settings.cg_tolerance(0.01),
        gpytorch.settings.max_preconditioner_size(100),
    )


def predict_expert_batch(
    bundle: ExpertBundle,
    n_batch, eta_batch, s_batch,
    W: float, H: float, device,
) -> np.ndarray:
    """Run one expert's GP models on a batch of parameter vectors.

    Returns: (batch, 8) predicted flow distances in physical space.
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
    vals_scaled = (vals - bundle.x_mean.cpu().numpy()) / bundle.x_scale.cpu().numpy()
    bundle_dtype = bundle.x_mean.dtype
    xt = torch.tensor(vals_scaled, dtype=bundle_dtype, device=device)

    jitter = 1e-2 if bundle_dtype == torch.float32 else 1e-3
    cg1, cg2, cg3 = _cg_context()
    preds_scaled = []
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
         gpytorch.settings.cholesky_jitter(jitter), cg1, cg2, cg3:
        for m, l in zip(bundle.models, bundle.likes):
            preds_scaled.append(l(m(xt)).mean.detach().cpu().numpy())
    Y_s = np.stack(preds_scaled, axis=1)

    # Poly residual
    pr = bundle.poly_residual
    if pr is not None:
        powers = np.asarray(pr["powers"], dtype=int)
        coef   = np.asarray(pr["coef"],   dtype=np.float64)
        Xp     = _apply_poly_powers(vals_scaled, powers)
        Y_s    = Y_s + Xp @ coef.T

    # Inverse-scale
    y_mean  = bundle.y_mean.cpu().numpy()
    y_scale = bundle.y_scale.cpu().numpy()
    Y_phys  = Y_s * y_scale + y_mean

    # Differential target mode
    if bundle.target_mode == "diff":
        Y_phys = np.maximum(Y_phys, 0.0)
        Y_phys = np.cumsum(Y_phys, axis=1)

    # Physical constraints
    Y_phys = np.maximum(Y_phys, 0.0)
    for i in range(1, Y_phys.shape[1]):
        Y_phys[:, i] = np.maximum(Y_phys[:, i], Y_phys[:, i - 1])

    return Y_phys.astype(float)


def predict_expert_variance(
    bundle: ExpertBundle,
    n_batch, eta_batch, s_batch,
    W: float, H: float, device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Like predict_expert_batch but also returns GP predictive variance.

    Returns: (predictions, variances) both (N, 8).
    """
    batch_size = len(n_batch)
    if batch_size == 0:
        return np.array([]), np.array([])
    col_map = {c: i for i, c in enumerate(bundle.all_cols)}
    vals = np.zeros((batch_size, len(bundle.all_cols)), dtype=float)
    vals[:, col_map["n"]]       = n_batch
    vals[:, col_map["eta"]]     = eta_batch
    vals[:, col_map["sigma_y"]] = s_batch
    vals[:, col_map["width"]]   = W
    vals[:, col_map["height"]]  = H
    for j in bundle.log_idx:
        vals[:, j] = np.log(np.clip(vals[:, j] + bundle.log_eps, 1e-12, None))
    vals_scaled = (vals - bundle.x_mean.cpu().numpy()) / bundle.x_scale.cpu().numpy()
    bundle_dtype = bundle.x_mean.dtype
    xt = torch.tensor(vals_scaled, dtype=bundle_dtype, device=device)

    jitter = 1e-2 if bundle_dtype == torch.float32 else 1e-3
    cg1, cg2, cg3 = _cg_context()
    preds_scaled, vars_scaled = [], []
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
         gpytorch.settings.cholesky_jitter(jitter), cg1, cg2, cg3:
        for m, l in zip(bundle.models, bundle.likes):
            pred_dist = l(m(xt))
            preds_scaled.append(pred_dist.mean.detach().cpu().numpy())
            vars_scaled.append(pred_dist.variance.detach().cpu().numpy())

    Y_s = np.stack(preds_scaled, axis=1)
    V_s = np.stack(vars_scaled, axis=1)

    pr = bundle.poly_residual
    if pr is not None:
        powers = np.asarray(pr["powers"], dtype=int)
        coef   = np.asarray(pr["coef"],   dtype=np.float64)
        Xp     = _apply_poly_powers(vals_scaled, powers)
        Y_s    = Y_s + Xp @ coef.T

    y_mean  = bundle.y_mean.cpu().numpy()
    y_scale = bundle.y_scale.cpu().numpy()
    Y_phys  = Y_s * y_scale + y_mean

    if bundle.target_mode == "diff":
        Y_phys = np.maximum(Y_phys, 0.0)
        Y_phys = np.cumsum(Y_phys, axis=1)

    Y_phys = np.maximum(Y_phys, 0.0)
    for i in range(1, Y_phys.shape[1]):
        Y_phys[:, i] = np.maximum(Y_phys[:, i], Y_phys[:, i - 1])

    return Y_phys.astype(float), V_s.astype(float)


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
            pred_batch = predict_expert_batch(bundle, n_batch, eta_batch, s_batch,
                                              W, H, device)
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


def dynamic_soft_predict_batch(
    theta_batch, gate_dict, expert_cache: Dict[int, ExpertBundle],
    W: float, H: float, device,
    strategy="threshold", threshold=0.01, max_experts=5,
) -> np.ndarray:
    """Predict with per-candidate dynamic expert selection (input-space gating)."""
    batch_size = len(theta_batch)
    if batch_size == 0:
        return np.zeros((0, 8))
    preds = np.zeros((batch_size, 8))

    for i, theta in enumerate(theta_batch):
        phi = build_input_features(theta, W, H)
        expert_ids, weights = get_adaptive_weights(
            gate_dict, phi,
            strategy=strategy, threshold=threshold, max_experts=max_experts,
        )
        valid = [(eid, w) for eid, w in zip(expert_ids, weights)
                 if eid in expert_cache]
        if not valid:
            continue
        e_ids, e_wts = zip(*valid)
        e_wts = np.array(e_wts)
        e_wts /= e_wts.sum()

        pred = soft_predict_batch(
            [theta], expert_cache, list(e_ids), e_wts, W, H, device,
        )
        preds[i] = pred[0]
    return preds
