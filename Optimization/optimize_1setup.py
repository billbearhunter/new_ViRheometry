#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
optimize_1setup.py (Final: GPU Batch + Fallback + Wall Clock + Force Iterations)
"""

import argparse
import os
import sys
import math
import json
import warnings
import datetime
import csv
import numpy as np
import matplotlib.pyplot as plt
import time

plt.switch_backend('Agg')

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import torch
import cma
import joblib

try:
    import sklearn
except ImportError as e:
    raise RuntimeError("scikit-learn is required.") from e
try:
    import numpy.core as _np_core
    if "numpy._core" not in sys.modules: sys.modules["numpy._core"] = _np_core
except Exception: pass
try:
    import gpytorch
except ImportError as e:
    raise RuntimeError("gpytorch is required.") from e

try:
    from libs.param import Param
    from libs.setup import Setup
    from libs.mechanism import Mechanism
    from libs.compare_loss import mat_hw_to_PL
except ImportError:
    print("\n[WARNING] 'libs' module not found. Feasibility checks will be skipped.\n")
    def mat_hw_to_PL(*args, **kwargs): return 1.0, 1.0
    class Param:
        def __init__(self, eta, n, sigma_y): self.eta, self.n, self.sigma_y = eta, n, sigma_y
    class Setup:
        def __init__(self, H, W, w): self.H, self.W, self.w = H, W, w
    class Mechanism:
        def searchNewSetup_orthognality_for_second_setup(self, m_breve, setups):
            return [None, Setup(setups[0].H, setups[0].W, 1.0)]

MIN_N, MAX_N = 0.3, 1.0
MIN_ETA, MAX_ETA = 0.001, 300.0
MIN_SIGMA_Y, MAX_SIGMA_Y = 0.0, 400.0
GLOBAL_BOUNDS = {"n": (MIN_N, MAX_N), "eta": (MIN_ETA, MAX_ETA), "sigma_y": (MIN_SIGMA_Y, MAX_SIGMA_Y)}
DTYPE = torch.float64

class _OfflineSVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        v = gpytorch.variational.VariationalStrategy(self, inducing_points, gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0)), learn_inducing_locations=True)
        super().__init__(v)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5) + gpytorch.kernels.LinearKernel())
    def forward(self, x): return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

class _OfflineExactGP(gpytorch.models.ExactGP):
    def __init__(self, tx, ty, lik):
        super().__init__(tx, ty, lik)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5) + gpytorch.kernels.LinearKernel())
    def forward(self, x): return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

@dataclass
class ExpertBundle:
    cid: int
    models: List[torch.nn.Module]
    likes: List[gpytorch.likelihoods.Likelihood]
    x_mean: torch.Tensor
    x_scale: torch.Tensor
    y_mean: torch.Tensor
    y_scale: torch.Tensor
    all_cols: List[str]
    log_idx: List[int]
    log_eps: float

def build_phi(y, W, H, eps=1e-8):
    y = np.asarray(y, dtype=float).reshape(1, -1)
    y_norm = y / (y[:, [-1]] + eps)
    feats = [y_norm, np.diff(y_norm, axis=1), np.log(np.abs(y[:, -1]) + eps).reshape(-1, 1)]
    feats.append(np.hstack([np.log(np.sqrt(W*H) + eps).reshape(1,1), np.log((W+eps)/(H+eps)).reshape(1,1)]))
    return np.hstack(feats)

def get_adaptive_weights(gate_dict, phi, strategy="threshold", threshold=0.01, 
                        topk_hard=None, confidence_threshold=0.7, max_experts=5):
    gmm = gate_dict["gmm"]
    scaler = gate_dict.get("scaler")
    phi_in = phi.reshape(1, -1)
    if scaler is not None: phi_in = scaler.transform(phi_in)
    probs = gmm.predict_proba(phi_in)[0]
    
    if strategy == "all":
        expert_indices = np.arange(len(probs))
        weights = probs / np.sum(probs)
    elif strategy == "topk":
        if topk_hard is None: topk_hard = 2
        topk_idx = np.argsort(-probs)[:topk_hard]
        expert_indices = topk_idx
        weights = np.ones(len(topk_idx)) / len(topk_idx)
    elif strategy == "threshold":
        mask = probs >= threshold
        if not np.any(mask): mask[np.argmax(probs)] = True
        expert_indices = np.where(mask)[0]
        filtered_probs = probs[mask]
        weights = filtered_probs / np.sum(filtered_probs)
        if len(expert_indices) > max_experts:
            top_idx = np.argsort(-filtered_probs)[:max_experts]
            expert_indices = expert_indices[top_idx]
            filtered_probs = filtered_probs[top_idx]
            weights = filtered_probs / np.sum(filtered_probs)
    elif strategy == "adaptive":
        max_prob = np.max(probs)
        if max_prob > confidence_threshold:
            expert_indices = [np.argmax(probs)]
            weights = [1.0]
        elif max_prob > 0.3:
            mask = probs >= 0.05
            if not np.any(mask): mask[np.argmax(probs)] = True
            expert_indices = np.where(mask)[0]
            filtered_probs = probs[mask]
            weights = filtered_probs / np.sum(filtered_probs)
            if len(expert_indices) > max_experts:
                top_idx = np.argsort(-filtered_probs)[:max_experts]
                expert_indices = expert_indices[top_idx]
                filtered_probs = filtered_probs[top_idx]
                weights = filtered_probs / np.sum(filtered_probs)
        else:
            expert_indices = np.argsort(-probs)[:8]
            filtered_probs = probs[expert_indices]
            weights = filtered_probs / np.sum(filtered_probs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    expert_ids = [int(idx) + 1 for idx in expert_indices]
    return expert_ids, np.array(weights, dtype=float)

def predict_expert_batch(bundle, n_batch, eta_batch, s_batch, W, H, device):
    batch_size = len(n_batch)
    if batch_size == 0: return np.array([])
    col_map = {c: i for i, c in enumerate(bundle.all_cols)}
    vals = np.zeros((batch_size, len(bundle.all_cols)), dtype=float)
    vals[:, col_map["n"]] = n_batch
    vals[:, col_map["eta"]] = eta_batch
    vals[:, col_map["sigma_y"]] = s_batch
    vals[:, col_map["width"]] = W
    vals[:, col_map["height"]] = H
    for j in bundle.log_idx:
        vals[:, j] = np.log(np.clip(vals[:, j] + bundle.log_eps, 1e-12, None))
    xt = torch.tensor(vals, dtype=DTYPE, device=device)
    xt = (xt - bundle.x_mean) / bundle.x_scale
    preds = []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i, (m, l) in enumerate(zip(bundle.models, bundle.likes)):
            p_raw = l(m(xt)).mean
            scale = bundle.y_scale[..., i:i+1].view(1)
            mean = bundle.y_mean[..., i:i+1].view(1)
            p_scaled = p_raw * scale + mean
            preds.append(p_scaled)
    return torch.stack(preds).t().detach().cpu().numpy().astype(float)

def soft_predict_batch(theta_batch, expert_bundles, expert_ids, weights, W, H, device):
    batch_size = len(theta_batch)
    if batch_size == 0: return np.zeros((0, 1))
    n_batch = np.array([t[0] for t in theta_batch])
    eta_batch = np.array([t[1] for t in theta_batch])
    s_batch = np.array([t[2] for t in theta_batch])
    weighted_pred = None
    valid_weights_sum = 0.0
    for cid, weight in zip(expert_ids, weights):
        if cid not in expert_bundles: continue
        bundle = expert_bundles[cid]
        try:
            pred_batch = predict_expert_batch(bundle, n_batch, eta_batch, s_batch, W, H, device)
            if weighted_pred is None: weighted_pred = np.zeros_like(pred_batch)
            weighted_pred += weight * pred_batch
            valid_weights_sum += weight
        except Exception: continue
    if weighted_pred is None: raise ValueError("No valid experts for prediction")
    if valid_weights_sum > 0: weighted_pred /= valid_weights_sum
    return weighted_pred

def soft_predict(theta, expert_bundles, expert_ids, weights, W, H, device):
    n, eta, sigma_y = theta
    individual_preds = []
    valid_experts = []
    valid_weights = []
    for cid, weight in zip(expert_ids, weights):
        if cid not in expert_bundles: continue
        bundle = expert_bundles[cid]
        try:
            pred = soft_predict_batch([[n, eta, sigma_y]], {cid: expert_bundles[cid]}, [cid], [1.0], W, H, device)
            pred = pred.flatten()
            individual_preds.append((cid, pred, weight))
            valid_experts.append(cid)
            valid_weights.append(weight)
        except Exception: continue
    if not valid_experts: raise ValueError("No valid experts")
    valid_weights = np.array(valid_weights) / np.sum(valid_weights)
    weighted_pred = np.zeros_like(individual_preds[0][1])
    for (cid, pred, weight) in zip(valid_experts, [p[1] for p in individual_preds], valid_weights):
        weighted_pred += weight * pred
    return weighted_pred, individual_preds

def _safe_torch_load(path, map_location):
    try: return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError: return torch.load(path, map_location=map_location)

def mse_loss(pred, target):
    return float(np.mean((np.asarray(pred, float) - np.asarray(target, float)) ** 2))

def load_json(path): 
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def maybe_load_joblib(path):
    if path and os.path.exists(path): return joblib.load(path)
    return None

def clamp_params(theta, bounds):
    eps = 1e-5
    return [float(np.clip(theta[0], bounds["n"][0]+eps, bounds["n"][1]-eps)),
            float(np.clip(theta[1], bounds["eta"][0]+eps, bounds["eta"][1]-eps)),
            float(np.clip(theta[2], bounds["sigma_y"][0]+eps, bounds["sigma_y"][1]-eps))]

def default_x0(bounds):
    return [0.5*(bounds["n"][0]+bounds["n"][1]), math.sqrt(bounds["eta"][0]*bounds["eta"][1]), math.sqrt(max(bounds["sigma_y"][0], 1e-12)*bounds["sigma_y"][1])]

def route_target_dis_topk(gate_dict, phi, topk=2):
    gmm = gate_dict["gmm"]
    scaler = gate_dict.get("scaler")
    phi_in = phi.reshape(1, -1)
    if scaler is not None: phi_in = scaler.transform(phi_in)
    probs = gmm.predict_proba(phi_in)[0]
    order = np.argsort(-probs)
    return [int(x)+1 for x in order[:topk]], probs[order[:topk]]

def check_feasibility(theta, Hcm, Wcm, base=1e6):
    n, eta, sigma_y = float(theta[0]), float(theta[1]), float(theta[2])
    try: P0, L0 = mat_hw_to_PL(eta * 0.1, n, sigma_y * 0.1, Hcm, Wcm); P, L = P0/10.0, L0*100.0
    except: return base * 10.0
    if not (np.isfinite(P) and np.isfinite(L) and P>0 and L>0): return base
    if (P*L - sigma_y) <= 0: return base * 2.0
    try: W_val = P*(L - sigma_y/P)/eta
    except: return base * 10.0
    if not (np.isfinite(W_val) and W_val>0): return base * 3.0
    return 0.0

def load_expert_bundle(path, device):
    ckpt = _safe_torch_load(path, map_location=device)
    cid = int(ckpt.get("cid", -1))
    def get_v(d, k, alt_k, root, root_k):
        v = None
        if d: v = d.get(k)
        if v is None and d: v = d.get(alt_k)
        if v is None and root: v = root.get(root_k)
        return v
    def safe_tensor(data, dtype, device):
        if torch.is_tensor(data): return data.detach().clone().to(dtype=dtype, device=device)
        else: return torch.tensor(data, dtype=dtype, device=device)
    xs, ys = ckpt.get("x_scaler", {}), ckpt.get("y_scaler", {})
    xm = get_v(xs, "mean", "mean_", ckpt, "X_mean")
    xsc = get_v(xs, "scale", "scale_", ckpt, "X_scale")
    ym = get_v(ys, "mean", "mean_", ckpt, "Y_mean")
    ysc = get_v(ys, "scale", "scale_", ckpt, "Y_scale")
    if xm is None: raise ValueError(f"Missing X_mean in {path}")
    x_mean = safe_tensor(xm, DTYPE, device).view(1, -1)
    x_scale = safe_tensor(xsc, DTYPE, device).view(1, -1)
    y_mean = safe_tensor(ym, DTYPE, device).view(1, -1)
    y_scale = safe_tensor(ysc, DTYPE, device).view(1, -1)
    models, likes = [], []
    msds = ckpt.get("models")
    lsds = ckpt.get("likes") or ckpt.get("likelihoods")
    is_exact = (ckpt.get("gp_kind", "svgp") == "exact")
    inducing = None
    if not is_exact: inducing = ckpt.get("inducing") or ckpt.get("inducing_points")
    if is_exact:
        tx = safe_tensor(ckpt["train_x"], DTYPE, device)
        ty = safe_tensor(ckpt["train_y"], DTYPE, device)
        for i in range(len(msds)):
            l = gpytorch.likelihoods.GaussianLikelihood().to(device, DTYPE)
            l.load_state_dict(lsds[i])
            m = _OfflineExactGP(tx, ty[:, i], l).to(device, DTYPE)
            m.load_state_dict(msds[i])
            m.eval(); l.eval(); models.append(m); likes.append(l)
    else:
        for i in range(len(msds)):
            ip = safe_tensor(inducing[i], DTYPE, device)
            m = _OfflineSVGPModel(ip).to(device, DTYPE)
            m.load_state_dict(msds[i])
            l = gpytorch.likelihoods.GaussianLikelihood().to(device, DTYPE)
            l.load_state_dict(lsds[i])
            m.eval(); l.eval(); models.append(m); likes.append(l)
    all_cols = xs.get("all_cols", ["n", "eta", "sigma_y", "width", "height"])
    log_cols = set(xs.get("log_cols", ["eta", "sigma_y"]))
    log_idx = [i for i, c in enumerate(all_cols) if c in log_cols]
    return ExpertBundle(cid, models, likes, x_mean, x_scale, y_mean, y_scale, all_cols, log_idx, 1e-6)

def run_cmaes_batch(batch_loss_fn, bounds, x0=None, sigma0=0.5, popsize=16, maxiter=700, seed=42, verb_disp=1):
    if x0 is None: x0 = default_x0(bounds)
    x0 = clamp_params(x0, bounds)
    opts = {
        "bounds": [[bounds[k][0] for k in ["n","eta","sigma_y"]], [bounds[k][1] for k in ["n","eta","sigma_y"]]],
        "seed": seed, "popsize": popsize, "verbose": verb_disp, 
        "maxiter": maxiter,
        "tolx": 1e-20,
        "tolfun": 1e-20,
        "tolstagnation": maxiter * 2
    }
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    loss_history = []
    
    while not es.stop():
        sols = es.ask()
        clamped_sols = [clamp_params(s, bounds) for s in sols]
        losses = batch_loss_fn(clamped_sols)
        es.tell(sols, losses)
        if verb_disp > 0: es.disp()
        loss_history.append(es.result.fbest)

    res = es.result
    return [float(x) for x in res.xbest], float(res.fbest), loss_history

def create_comparison_visualization(theta, strategy_info, y_target, W, H, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    ax1 = axes[0, 0]
    if 'hard' in strategy_info: strategy_info['hard']['weights'] = np.array(strategy_info['hard']['weights'])
    if 'soft' in strategy_info: strategy_info['soft']['weights'] = np.array(strategy_info['soft']['weights'])
    if 'hard' in strategy_info and 'soft' in strategy_info:
        x_hard = np.arange(len(strategy_info['hard']['expert_ids']))
        x_soft = np.arange(len(strategy_info['soft']['expert_ids']))
        ax1.bar(x_hard - 0.2, strategy_info['hard']['weights'], width=0.4, label='Hard', alpha=0.7)
        ax1.bar(x_soft + 0.2, strategy_info['soft']['weights'], width=0.4, label='Soft', alpha=0.7)
        ax1.legend()
    elif 'soft' in strategy_info:
        x_soft = np.arange(len(strategy_info['soft']['expert_ids']))
        ax1.bar(x_soft, strategy_info['soft']['weights'], width=0.4, label='Soft', alpha=0.7)
    ax1.set_title('Expert Weights')
    ax2 = axes[0, 1]
    ax2.plot(y_target, 'ko-', label='Target')
    if 'hard' in strategy_info: ax2.plot(strategy_info['hard']['prediction'], 'bs--', label='Hard')
    if 'soft' in strategy_info: ax2.plot(strategy_info['soft']['prediction'], 'r^--', label='Soft')
    ax2.legend()
    ax2.set_title('Prediction')
    ax3 = axes[0, 2]
    ax3.scatter(theta[0], theta[1], s=200, c='red', marker='*')
    ax3.set_title(f'Params: n={theta[0]:.2f}, eta={theta[1]:.2f}')
    ax3.set_yscale('log')
    
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    save_path = os.path.join(save_dir, f'boundary_comparison_{timestamp}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return save_path

def main():
    wall_clock_start = time.time()
    p = argparse.ArgumentParser()
    p.add_argument("-W1", type=float, required=True)
    p.add_argument("-H1", type=float, required=True)
    p.add_argument("-dis1", type=float, nargs=8, required=True)
    p.add_argument("--moe_dir", type=str, required=True)
    p.add_argument("--strategy", type=str, default="threshold", 
                   choices=["topk", "threshold", "adaptive", "all"])
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--confidence_threshold", type=float, default=0.7)
    p.add_argument("--max_experts", type=int, default=5)
    p.add_argument("--compare_strategies", action="store_true")
    p.add_argument("--sigma0", type=float, default=0.5)
    p.add_argument("--popsize", type=int, default=16)
    p.add_argument("--maxiter", type=int, default=700)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verb", type=int, default=1)

    args = p.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_str = f"{args.strategy}_{args.threshold}" if args.strategy != "topk" else f"topk_{args.topk}"
    save_dir = f"result_setup1_{strategy_str}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n[Info] Output directory created: {save_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")
    
    width1, height1 = float(args.W1), float(args.H1)
    y1 = np.array(args.dis1, dtype=float)

    print(f"\n========= SETUP 1 (MoE Optimization) =========")
    gate_dict = maybe_load_joblib(os.path.join(args.moe_dir, "gmm_gate.joblib"))
    moe_boxes = load_json(os.path.join(args.moe_dir, "boxes.json"))
    phi = build_phi(y1, width1, height1)
    
    strategy_info = {
        'strategy': args.strategy,
        'threshold': args.threshold,
        'topk': args.topk,
        'phi': phi.flatten()
    }
    
    expert_cache = {}
    if args.strategy == "topk":
        expert_ids, probs = route_target_dis_topk(gate_dict, phi, topk=args.topk)
        weights = np.ones(len(expert_ids)) / len(expert_ids)
        print(f"Hard TopK Experts: {expert_ids}")
    else:
        expert_ids, weights = get_adaptive_weights(
            gate_dict, phi, strategy=args.strategy,
            threshold=args.threshold, confidence_threshold=args.confidence_threshold,
            max_experts=args.max_experts
        )
        print(f"Soft Experts: {expert_ids}, Weights: {weights.round(3)}")
    
    for cid in expert_ids:
        if cid not in expert_cache:
            try:
                expert_cache[cid] = load_expert_bundle(os.path.join(args.moe_dir, f"expert_{cid}.pt"), device)
            except Exception as e:
                print(f"[Warning] Failed to load expert {cid}: {e}")

    bounds = GLOBAL_BOUNDS.copy()
    for cid in expert_ids:
        box = moe_boxes.get(str(cid)) or moe_boxes.get(cid)
        if box:
            for k in ["n", "eta", "sigma_y"]:
                lo = max(bounds[k][0], float(box[k][0]))
                hi = min(bounds[k][1], float(box[k][1]))
                if lo < hi: bounds[k] = (lo, hi)
    
    def loss_setup1_batch(thetas):
        batch_size = len(thetas)
        losses = np.zeros(batch_size)
        valid_indices = []
        valid_params = []
        for i, theta in enumerate(thetas):
            pen = check_feasibility(theta, height1, width1)
            if pen > 0.0: losses[i] = pen
            else:
                valid_indices.append(i)
                valid_params.append(theta)
        if not valid_indices: return losses.tolist()

        try:
            preds = soft_predict_batch(valid_params, expert_cache, expert_ids, weights, width1, height1, device)
            diff = preds - y1.reshape(1, -1)
            mse_vals = np.mean(diff**2, axis=1)
            for idx, val in zip(valid_indices, mse_vals): losses[idx] = val
        except RuntimeError:
            for idx, param in zip(valid_indices, valid_params):
                try:
                    pred_single = soft_predict_batch([param], expert_cache, expert_ids, weights, width1, height1, device)
                    diff = pred_single - y1.reshape(1, -1)
                    losses[idx] = np.mean(diff**2)
                except Exception: losses[idx] = 1e6
        return losses.tolist()

    print(f"\n--- Optimization Start (Popsize: {args.popsize}) ---")
    
    theta_best, loss_best, hist = run_cmaes_batch(
        loss_setup1_batch, bounds, 
        sigma0=args.sigma0, popsize=args.popsize, 
        maxiter=args.maxiter, seed=args.seed, verb_disp=args.verb
    )
    
    n_best, eta_best, sigma_best = theta_best
    print(f"Best Params: n={n_best:.6f}, eta={eta_best:.6f}, sigma_y={sigma_best:.6f}")
    
    csv_path = os.path.join(save_dir, "best_loss_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Loss"])
        writer.writerows(enumerate(hist))
    
    plt.figure()
    plt.plot(hist)
    plt.yscale("log")
    plt.title("Convergence")
    plt.savefig(os.path.join(save_dir, "best_loss_convergence.png"))
    plt.close()

    final_pred, individual_preds = soft_predict(
        theta_best, expert_cache, expert_ids, weights, width1, height1, device
    )
    strategy_info['soft'] = {
        'expert_ids': expert_ids, 'weights': weights,
        'prediction': final_pred, 'individual': individual_preds
    }
    vis_path = create_comparison_visualization(theta_best, strategy_info, y1, width1, height1, save_dir)
    print(f"Visualization saved to: {vis_path}")

    try:
        print("\n... Calculating Recommended Setup 2 ...")
        m_breve = Param(eta_best, n_best, sigma_best)
        s1 = Setup(height1, width1, 1.0)
        mech = Mechanism()
        new_setups = mech.searchNewSetup_orthognality_for_second_setup(m_breve, [s1])
        setup2 = new_setups[1]
        print(f"W2 = {setup2.W:.3f}, H2 = {setup2.H:.3f}")
        
        np.savetxt(os.path.join(save_dir, "setup1_best_x_n_eta_sigma.txt"), np.array([theta_best]))
        np.savetxt(os.path.join(save_dir, "setup1_recommended_setup2_WH.txt"), np.array([[setup2.W, setup2.H]]))
        print(f"Files saved to {save_dir}")
        
    except Exception as e:
        print(f"[Error] Mechanism search failed: {e}")
        np.savetxt(os.path.join(save_dir, "setup1_best_x_n_eta_sigma.txt"), np.array([theta_best]))

    wall_clock_end = time.time()
    total_seconds = wall_clock_end - wall_clock_start
    print(f"\n[Finished] Total Wall Clock Time: {total_seconds:.2f}s")
    
    with open(os.path.join(save_dir, "wall_clock_time.txt"), "w") as f:
        f.write(f"{total_seconds:.4f}")

if __name__ == "__main__":
    main()