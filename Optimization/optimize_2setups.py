#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
optimize_2setups.py (Final: GPU Batch + Fallback + Barrier + Clean Output + Wall Clock + Force Iterations)
"""

import argparse
import os
import sys
import math
import json
import itertools
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
except ImportError:
    raise RuntimeError("scikit-learn required")
try:
    import gpytorch
except ImportError:
    raise RuntimeError("gpytorch required")
try:
    import numpy.core as _np_core
    if "numpy._core" not in sys.modules: sys.modules["numpy._core"] = _np_core
except: pass

try:
    from libs.param import Param
    from libs.setup import Setup
    from libs.mechanism import Mechanism
    from libs.compare_loss import mat_hw_to_PL
except ImportError:
    print("[WARN] libs not found. Feasibility checks skipped.")
    def mat_hw_to_PL(*args, **kwargs): return 1.0, 1.0 
    class Param:
        def __init__(self, e, n, s): self.eta, self.n, self.sigma_y = e, n, s
    class Setup:
        def __init__(self, h, w, ww): self.H, self.W, self.w = h, w, ww
    class Mechanism:
        def searchNewSetup_orthognality_for_third_setup(self, m, s): return [None, None, Setup(s[-1].H, s[-1].W, 1.0)]

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
    if weighted_pred is None: raise ValueError("No valid experts")
    if valid_weights_sum > 0: weighted_pred /= valid_weights_sum
    return weighted_pred

def _safe_torch_load(path, map_location):
    try: return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError: return torch.load(path, map_location=map_location)

def maybe_load_joblib(path):
    if path and os.path.exists(path): return joblib.load(path)
    return None

def clamp_params(theta, bounds):
    eps = 1e-5
    return [float(np.clip(theta[0], bounds["n"][0]+eps, bounds["n"][1]-eps)),
            float(np.clip(theta[1], bounds["eta"][0]+eps, bounds["eta"][1]-eps)),
            float(np.clip(theta[2], bounds["sigma_y"][0]+eps, bounds["sigma_y"][1]-eps))]

def default_x0(bounds):
    n0 = 0.5 * (bounds["n"][0] + bounds["n"][1])
    eta0 = math.exp(0.5 * (math.log(bounds["eta"][0]) + math.log(bounds["eta"][1])))
    sig0 = math.exp(0.5 * (math.log(max(bounds["sigma_y"][0], 1e-6)) + math.log(bounds["sigma_y"][1])))
    return [n0, eta0, sig0]

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

def route_target_dis_topk(gate_dict, phi, topk=2):
    gmm = gate_dict["gmm"]
    scaler = gate_dict.get("scaler")
    phi_in = phi.reshape(1, -1)
    if scaler is not None: phi_in = scaler.transform(phi_in)
    probs = gmm.predict_proba(phi_in)[0]
    order = np.argsort(-probs)
    return [int(x) + 1 for x in order[:topk]], probs[order[:topk]]

def run_cmaes_batch_timed(loss_fn_batch, bounds, x0=None, sigma0=0.5, popsize=16, maxiter=700, seed=42, verb_disp=1):
    if x0 is None: x0 = default_x0(bounds)
    x0 = clamp_params(x0, bounds)
    opts = {
        "bounds": [[bounds[k][0] for k in ["n", "eta", "sigma_y"]], 
                  [bounds[k][1] for k in ["n", "eta", "sigma_y"]]],
        "seed": seed, "popsize": popsize, "verbose": verb_disp, 
        "maxiter": maxiter,
        "tolx": 1e-20,
        "tolfun": 1e-20,
        "tolstagnation": maxiter * 2
    }
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    loss_history = []
    iter_times = []
    
    while not es.stop():
        gen_start = time.time()
        sols = es.ask()
        clamped_sols = [clamp_params(s, bounds) for s in sols]
        losses = loss_fn_batch(clamped_sols)
        es.tell(sols, losses)
        if verb_disp > 0: es.disp()
        loss_history.append(es.result.fbest)
        iter_times.append(time.time() - gen_start)

    res = es.result
    return [float(x) for x in res.xbest], float(res.fbest), loss_history, iter_times

def main():
    wall_clock_start = time.time()
    p = argparse.ArgumentParser()
    p.add_argument("-W1", type=float, required=True)
    p.add_argument("-H1", type=float, required=True)
    p.add_argument("-dis1", type=float, nargs=8, required=True)
    p.add_argument("-W2", type=float, required=True)
    p.add_argument("-H2", type=float, required=True)
    p.add_argument("-dis2", type=float, nargs=8, required=True)
    p.add_argument("-W3", type=float, default=0.0)
    p.add_argument("-H3", type=float, default=0.0)
    p.add_argument("-dis3", type=float, nargs=8, default=[])
    
    p.add_argument("--setup1_dir", type=str, default=None, 
                        help="Path to the timestamped directory from Setup 1 containing setup1_best_x_n_eta_sigma.txt")

    p.add_argument("--moe_dir", type=str, required=True)
    p.add_argument("--strategy", type=str, default="threshold", choices=["topk", "threshold", "adaptive", "all"])
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--topk", type=int, default=1)
    p.add_argument("--confidence_threshold", type=float, default=0.7)
    p.add_argument("--max_experts", type=int, default=5)
    p.add_argument("--sigma0", type=float, default=0.5)
    p.add_argument("--popsize", type=int, default=16)
    p.add_argument("--maxiter", type=int, default=700)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verb", type=int, default=1)
    
    args = p.parse_args()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_str = f"{args.strategy}_{args.threshold}" if args.strategy != "topk" else f"topk_{args.topk}"
    save_dir = f"result_setup2_{strategy_str}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n[Info] Output directory created: {save_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    cfgs = [
        {"id": 1, "W": args.W1, "H": args.H1, "y": np.array(args.dis1, float)},
        {"id": 2, "W": args.W2, "H": args.H2, "y": np.array(args.dis2, float)}
    ]
    if args.W3 > 0 and args.H3 > 0 and len(args.dis3) == 8:
        cfgs.append({"id": 3, "W": args.W3, "H": args.H3, "y": np.array(args.dis3, float)})

    gate_dict = maybe_load_joblib(os.path.join(args.moe_dir, "gmm_gate.joblib"))
    norms = [np.mean(c["y"]**2) + 1e-9 for c in cfgs]

    needed_experts = set()
    for cfg in cfgs:
        phi = build_phi(cfg["y"], cfg["W"], cfg["H"])
        if args.strategy == "topk":
            e_ids, _ = route_target_dis_topk(gate_dict, phi, args.topk)
        else:
            e_ids, _ = get_adaptive_weights(gate_dict, phi, strategy=args.strategy, 
                                            threshold=args.threshold, confidence_threshold=args.confidence_threshold, max_experts=args.max_experts)
        needed_experts.update(e_ids)

    expert_cache = {}
    print(f"Loading {len(needed_experts)} experts...")
    for cid in needed_experts:
        try:
            path = os.path.join(args.moe_dir, f"expert_{cid}.pt")
            if os.path.exists(path): expert_cache[cid] = load_expert_bundle(path, device)
        except Exception: pass

    config_experts = [] 
    for cfg in cfgs:
        phi = build_phi(cfg["y"], cfg["W"], cfg["H"])
        if args.strategy == "topk":
            ids, _ = route_target_dis_topk(gate_dict, phi, args.topk)
            weights = np.ones(len(ids)) / len(ids)
        else:
            ids, weights = get_adaptive_weights(gate_dict, phi, strategy=args.strategy, 
                                                threshold=args.threshold, confidence_threshold=args.confidence_threshold, max_experts=args.max_experts)
        config_experts.append((ids, weights))

    bounds = GLOBAL_BOUNDS.copy()
    theta_0 = default_x0(bounds)
    
    loaded_prior = False
    if args.setup1_dir:
        prior_path = os.path.join(args.setup1_dir, "setup1_best_x_n_eta_sigma.txt")
        if os.path.exists(prior_path):
            try:
                theta_0 = np.loadtxt(prior_path)
                print(f"[Info] Loaded Setup 1 prior from: {prior_path}")
                loaded_prior = True
            except Exception as e:
                print(f"[ERROR] Failed to load prior: {e}")
        else:
            print(f"[ERROR] Setup 1 file not found: {prior_path}")
    
    if not loaded_prior:
        print("\n" + "!"*40)
        print("[WARNING] NO PRIOR LOADED! OPTIMIZATION WILL START FROM GLOBAL CENTER.")
        print("Use --setup1_dir to specify the result directory of Setup 1.")
        print("!"*40 + "\n")

    scale_n = MAX_N - MIN_N
    scale_log_eta = math.log(MAX_ETA) - math.log(MIN_ETA)
    scale_log_sig = math.log(MAX_SIGMA_Y) - math.log(max(MIN_SIGMA_Y, 1e-6))
    LAMBDA_REG = 0.05
    BARRIER_WT = 0.001 

    local_scale = {}
    for k in ["n", "eta", "sigma_y"]:
        local_scale[k] = bounds[k][1] - bounds[k][0]
        if k in ["eta", "sigma_y"]:
            low = max(bounds[k][0], 1e-9)
            local_scale[f"log_{k}"] = math.log(bounds[k][1]) - math.log(low)

    def loss_setup2_batch(thetas):
        batch_size = len(thetas)
        losses = np.zeros(batch_size)
        valid_indices = []
        valid_params = []
        
        for i, theta in enumerate(thetas):
            pen = 0.0
            for cfg in cfgs: pen += check_feasibility(theta, cfg["H"], cfg["W"])
            if pen > 0: losses[i] = pen
            else:
                valid_indices.append(i)
                valid_params.append(theta)
        
        if not valid_indices: return losses.tolist()

        v_thetas = np.array(valid_params)
        
        d_n = ((v_thetas[:,0] - theta_0[0]) / scale_n)**2
        d_eta = ((np.log(v_thetas[:,1]) - math.log(theta_0[1])) / scale_log_eta)**2
        sig_0_safe = max(theta_0[2], 1e-9)
        d_sig = ((np.log(np.maximum(v_thetas[:,2], 1e-9)) - math.log(sig_0_safe)) / scale_log_sig)**2
        prior_loss = LAMBDA_REG * (d_n + d_eta + d_sig)

        epsilon = 1e-9
        val_n = (v_thetas[:,0] - bounds["n"][0]) / local_scale["n"]
        v_safe_n = np.clip(val_n, epsilon, 1.0 - epsilon)
        barrier = -np.log(v_safe_n) - np.log(1.0 - v_safe_n)
        
        val_eta = (np.log(v_thetas[:,1]) - math.log(max(bounds["eta"][0], 1e-9))) / local_scale["log_eta"]
        v_safe_eta = np.clip(val_eta, epsilon, 1.0 - epsilon)
        barrier += -np.log(v_safe_eta) - np.log(1.0 - v_safe_eta)

        val_sig = (np.log(np.maximum(v_thetas[:,2], 1e-9)) - math.log(max(bounds["sigma_y"][0], 1e-9))) / local_scale["log_sigma_y"]
        v_safe_sig = np.clip(val_sig, epsilon, 1.0 - epsilon)
        barrier += -np.log(v_safe_sig) - np.log(1.0 - v_safe_sig)

        cpu_loss_part = prior_loss + (BARRIER_WT * barrier)

        try:
            total_nmse = np.zeros(len(valid_indices))
            for cfg_idx, cfg in enumerate(cfgs):
                e_ids, e_wts = config_experts[cfg_idx]
                preds = soft_predict_batch(valid_params, expert_cache, e_ids, e_wts, cfg["W"], cfg["H"], device)
                diff = preds - cfg["y"].reshape(1, -1)
                mse_vals = np.mean(diff**2, axis=1)
                total_nmse += mse_vals / norms[cfg_idx]
            
            final_loss = (total_nmse / len(cfgs)) + cpu_loss_part
            for idx, val in zip(valid_indices, final_loss): losses[idx] = val

        except RuntimeError: 
            for i, (idx, param) in enumerate(zip(valid_indices, valid_params)):
                try:
                    single_nmse = 0.0
                    for cfg_idx, cfg in enumerate(cfgs):
                        e_ids, e_wts = config_experts[cfg_idx]
                        p_single = soft_predict_batch([param], expert_cache, e_ids, e_wts, cfg["W"], cfg["H"], device)
                        d_single = p_single - cfg["y"].reshape(1, -1)
                        single_nmse += np.mean(d_single**2) / norms[cfg_idx]
                    
                    losses[idx] = (single_nmse / len(cfgs)) + cpu_loss_part[i]
                except Exception:
                    losses[idx] = 1e6

        return losses.tolist()

    print(f"\n--- Starting Optimization (Popsize: {args.popsize}) ---")
    
    try:
        theta_best, loss_best, hist, times = run_cmaes_batch_timed(
            loss_setup2_batch, bounds, 
            x0=list(theta_0), sigma0=args.sigma0, 
            popsize=args.popsize, maxiter=args.maxiter, 
            seed=args.seed, verb_disp=args.verb
        )
        
        print(f"Optimization finished.")
        print(f"Best Params: {theta_best}")
        
        time_csv = os.path.join(save_dir, "iteration_times.csv")
        with open(time_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Iteration", "Duration_Seconds"])
            for i, t in enumerate(times): writer.writerow([i+1, f"{t:.4f}"])
        print(f"Timing saved to {time_csv}")

        np.savetxt(os.path.join(save_dir, "setup2_best_x_n_eta_sigma.txt"), np.array([theta_best]))
        csv_path = os.path.join(save_dir, "best_loss_history.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Iteration", "Loss"])
            writer.writerows(enumerate(hist))
        
        m_breve = Param(theta_best[1], theta_best[0], theta_best[2])
        setups = [Setup(c["H"], c["W"], 1.0) for c in cfgs]
        mech = Mechanism()
        new_setups = mech.searchNewSetup_orthognality_for_third_setup(m_breve, setups)
        if new_setups and len(new_setups) > 2:
            s3 = new_setups[2]
            print(f"Recommended Setup 3: W={s3.W:.3f}, H={s3.H:.3f}")
            np.savetxt(os.path.join(save_dir, "setup2_recommended_setup3_WH.txt"), np.array([[s3.W, s3.H]]))
        
        print(f"All files saved to: {save_dir}")

    except Exception as e:
        print(f"Optimization failed: {e}")

    wall_clock_end = time.time()
    total_seconds = wall_clock_end - wall_clock_start
    print(f"\n[Finished] Total Wall Clock Time: {total_seconds:.2f}s")
    
    with open(os.path.join(save_dir, "wall_clock_time.txt"), "w") as f:
        f.write(f"{total_seconds:.4f}")

if __name__ == "__main__":
    main()