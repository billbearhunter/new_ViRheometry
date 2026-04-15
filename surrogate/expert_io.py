"""
surrogate/expert_io.py
======================
Load/save GP expert checkpoints.

Previously in:
  - DataPipeline/moe_utils.py   (load_expert)
  - Optimization/libs/moe_core.py (load_expert_bundle, ExpertBundle)
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import joblib
import gpytorch

from .models import DEVICE, DTYPE, SingleOutputExactGP, SingleOutputSVGP
from .scalers import LogStandardInputScaler, TargetScaler


@dataclass
class ExpertBundle:
    """Runtime representation of a trained GP expert."""
    cid:           int
    models:        List[torch.nn.Module]
    likes:         List[gpytorch.likelihoods.Likelihood]
    x_mean:        torch.Tensor
    x_scale:       torch.Tensor
    y_mean:        torch.Tensor
    y_scale:       torch.Tensor
    all_cols:      List[str]
    log_idx:       List[int]
    log_eps:       float
    poly_residual: Optional[dict] = None
    target_mode:   str = "absolute"


def _safe_torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


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


def load_expert_bundle(path: str, device=None) -> ExpertBundle:
    """Load a trained GP expert from a .pt checkpoint file."""
    device = device or DEVICE
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
            m = SingleOutputExactGP(tx, ty[:, i], l).to(device, dtype)
            m.load_state_dict(msds[i])
            m.eval(); l.eval()
            models.append(m); likes.append(l)
    else:
        for i in range(len(msds)):
            ip = safe_tensor(inducing[i], dtype, device)
            m  = SingleOutputSVGP(ip).to(device, dtype)
            m.load_state_dict(msds[i])
            l = gpytorch.likelihoods.GaussianLikelihood().to(device, dtype)
            l.load_state_dict(lsds[i])
            m.eval(); l.eval()
            models.append(m); likes.append(l)

    all_cols      = xs.get("all_cols", ["n", "eta", "sigma_y", "width", "height"])
    log_cols      = set(xs.get("log_cols", ["eta", "sigma_y"]))
    log_idx       = [i for i, c in enumerate(all_cols) if c in log_cols]
    poly_residual = ckpt.get("poly_residual")
    target_mode   = ckpt.get("target_mode", "absolute")
    return ExpertBundle(cid, models, likes, x_mean, x_scale, y_mean, y_scale,
                        all_cols, log_idx, 1e-6, poly_residual, target_mode)


def load_all_experts(moe_dir: str, device=None, k: int = 60) -> Dict[int, ExpertBundle]:
    """Load all available expert bundles from a workspace directory."""
    device = device or DEVICE
    cache = {}
    for cid in range(1, k + 1):
        path = os.path.join(moe_dir, f"expert_{cid}.pt")
        if os.path.exists(path):
            try:
                cache[cid] = load_expert_bundle(path, device)
            except Exception as e:
                print(f"[Warning] Expert {cid}: {e}")
    return cache


def maybe_load_joblib(path):
    if path and os.path.exists(path):
        return joblib.load(path)
    return None


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_expert_for_training(path: str, device=None):
    """Load expert checkpoint for evaluation (training-side format).

    Returns (gp_kind, models, likes, x_scaler_dict, y_scaler_dict, poly_residual).
    """
    device = device or DEVICE
    state  = _safe_torch_load(path, device)

    gp_kind = state.get("gp_kind", "svgp")
    dtype = _detect_ckpt_dtype(state)
    models, likes = [], []

    if gp_kind == "exact":
        tx = state["train_x"].to(device, dtype)
        ty = state["train_y"].to(device, dtype)
        for i, msd in enumerate(state["models"]):
            lk = gpytorch.likelihoods.GaussianLikelihood().to(device, dtype)
            lk.load_state_dict(state["likes"][i])
            m  = SingleOutputExactGP(tx, ty[:, i].contiguous(), lk).to(device, dtype)
            m.load_state_dict(msd)
            m.eval(); lk.eval()
            models.append(m); likes.append(lk)
    else:
        inducing = state.get("inducing")
        for i, msd in enumerate(state["models"]):
            ip = torch.tensor(np.asarray(inducing[i]), device=device, dtype=dtype)
            m  = SingleOutputSVGP(ip).to(device, dtype)
            lk = gpytorch.likelihoods.GaussianLikelihood().to(device, dtype)
            m.load_state_dict(msd)
            lk.load_state_dict(state["likes"][i])
            m.eval(); lk.eval()
            models.append(m); likes.append(lk)

    return (
        gp_kind, models, likes,
        state["x_scaler"], state["y_scaler"],
        state.get("poly_residual"),
    )
