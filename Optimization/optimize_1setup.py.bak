#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
optimize_1setup.py
==================
CMA-ES parameter estimation from a single dam-break setup.

Shared infrastructure (GP models, expert loading, gating, CMA-ES runner)
lives in libs/moe_core.py to avoid code duplication with optimize_2setups.py.
"""

import argparse
import os
import datetime
import csv

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("Agg")

import torch

from libs.moe_core import (
    GLOBAL_BOUNDS,
    build_phi,
    get_adaptive_weights,
    soft_predict_batch,
    clamp_params,
    default_x0,
    check_feasibility,
    load_expert_bundle,
    load_json,
    maybe_load_joblib,
    run_cmaes,
)

# Optional: Mechanism search for recommending Setup 2
try:
    from libs.param import Param
    from libs.setup import Setup
    from libs.mechanism import Mechanism
except ImportError:
    print("[WARNING] 'libs' not found. Setup 2 recommendation will be skipped.")
    class Param:
        def __init__(self, eta, n, sigma_y): self.eta, self.n, self.sigma_y = eta, n, sigma_y
    class Setup:
        def __init__(self, H, W, w): self.H, self.W, self.w = H, W, w
    class Mechanism:
        def searchNewSetup_orthognality_for_second_setup(self, m, setups):
            return [None, Setup(setups[0].H, setups[0].W, 1.0)]


def main():
    import time
    wall_start = time.time()

    p = argparse.ArgumentParser(description="CMA-ES optimization: 1-setup")
    p.add_argument("-W1",    type=float, required=True, help="Container width [cm]")
    p.add_argument("-H1",    type=float, required=True, help="Container height [cm]")
    p.add_argument("-dis1",  type=float, nargs=8, required=True, help="8 flow distances [cm]")
    p.add_argument("--moe_dir",              type=str,   required=True)
    p.add_argument("--strategy",             type=str,   default="threshold",
                   choices=["topk", "threshold", "adaptive", "all"])
    p.add_argument("--threshold",            type=float, default=0.01)
    p.add_argument("--topk",                 type=int,   default=2)
    p.add_argument("--confidence_threshold", type=float, default=0.7)
    p.add_argument("--max_experts",          type=int,   default=5)
    p.add_argument("--sigma0",   type=float, default=0.5)
    p.add_argument("--popsize",  type=int,   default=16)
    p.add_argument("--maxiter",  type=int,   default=700)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--verb",     type=int,   default=1)
    args = p.parse_args()

    # ── Output directory ──────────────────────────────────────────────────────
    ts           = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_str = f"topk_{args.topk}" if args.strategy == "topk" \
                   else f"{args.strategy}_{args.threshold}"
    save_dir = f"result_setup1_{strategy_str}_{ts}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n[Info] Output directory: {save_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    W1 = float(args.W1)
    H1 = float(args.H1)
    y1 = np.array(args.dis1, dtype=float)

    # ── Load MoE gate ─────────────────────────────────────────────────────────
    gate_dict = maybe_load_joblib(os.path.join(args.moe_dir, "gmm_gate.joblib"))
    moe_boxes = load_json(os.path.join(args.moe_dir, "boxes.json"))
    phi       = build_phi(y1, W1, H1)

    # ── Select experts ────────────────────────────────────────────────────────
    expert_ids, weights = get_adaptive_weights(
        gate_dict, phi,
        strategy=args.strategy,
        threshold=args.threshold,
        topk_hard=args.topk,
        confidence_threshold=args.confidence_threshold,
        max_experts=args.max_experts,
    )
    print(f"Expert IDs: {expert_ids}  Weights: {np.round(weights, 3)}")

    # ── Load expert models ────────────────────────────────────────────────────
    expert_cache = {}
    for cid in expert_ids:
        try:
            expert_cache[cid] = load_expert_bundle(
                os.path.join(args.moe_dir, f"expert_{cid}.pt"), device
            )
        except Exception as e:
            print(f"[Warning] Failed to load expert {cid}: {e}")

    # ── Tighten bounds from expert boxes ─────────────────────────────────────
    bounds = GLOBAL_BOUNDS.copy()
    for cid in expert_ids:
        box = moe_boxes.get(str(cid)) or moe_boxes.get(cid)
        if box:
            for k in ["n", "eta", "sigma_y"]:
                lo = max(bounds[k][0], float(box[k][0]))
                hi = min(bounds[k][1], float(box[k][1]))
                if lo < hi:
                    bounds[k] = (lo, hi)

    # ── Loss function ─────────────────────────────────────────────────────────
    def loss_fn(thetas):
        losses = np.zeros(len(thetas))
        valid_idx, valid_params = [], []
        for i, theta in enumerate(thetas):
            pen = check_feasibility(theta, H1, W1)
            if pen > 0.0:
                losses[i] = pen
            else:
                valid_idx.append(i)
                valid_params.append(theta)
        if not valid_idx:
            return losses.tolist()
        try:
            preds    = soft_predict_batch(valid_params, expert_cache, expert_ids, weights, W1, H1, device)
            mse_vals = np.mean((preds - y1.reshape(1, -1)) ** 2, axis=1)
            for i, v in zip(valid_idx, mse_vals):
                losses[i] = v
        except RuntimeError:
            for i, param in zip(valid_idx, valid_params):
                try:
                    pred = soft_predict_batch([param], expert_cache, expert_ids, weights, W1, H1, device)
                    losses[i] = np.mean((pred - y1.reshape(1, -1)) ** 2)
                except Exception:
                    losses[i] = 1e6
        return losses.tolist()

    # ── CMA-ES ────────────────────────────────────────────────────────────────
    print(f"\n--- Optimization Start (popsize={args.popsize}) ---")
    theta_best, loss_best, hist = run_cmaes(
        loss_fn, bounds,
        sigma0=args.sigma0, popsize=args.popsize,
        maxiter=args.maxiter, seed=args.seed, verb_disp=args.verb,
    )
    n_best, eta_best, sigma_best = theta_best
    print(f"Best: n={n_best:.6f}  eta={eta_best:.6f}  sigma_y={sigma_best:.6f}  loss={loss_best:.6e}")

    # ── Save results ──────────────────────────────────────────────────────────
    np.savetxt(os.path.join(save_dir, "setup1_best_x_n_eta_sigma.txt"),
               np.array([theta_best]))

    csv_path = os.path.join(save_dir, "best_loss_history.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Iteration", "Loss"])
        w.writerows(enumerate(hist))

    plt.figure()
    plt.plot(hist)
    plt.yscale("log")
    plt.title("Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Best Loss")
    plt.savefig(os.path.join(save_dir, "best_loss_convergence.png"))
    plt.close()

    # ── Recommend Setup 2 ─────────────────────────────────────────────────────
    try:
        m_breve   = Param(eta_best, n_best, sigma_best)
        s1        = Setup(H1, W1, 1.0)
        new_setups = Mechanism().searchNewSetup_orthognality_for_second_setup(m_breve, [s1])
        s2 = new_setups[1]
        print(f"Recommended Setup 2: W={s2.W:.3f}  H={s2.H:.3f}")
        np.savetxt(os.path.join(save_dir, "setup1_recommended_setup2_WH.txt"),
                   np.array([[s2.W, s2.H]]))
    except Exception as e:
        print(f"[Error] Mechanism search failed: {e}")

    elapsed = time.time() - wall_start
    print(f"\n[Finished] Wall clock: {elapsed:.2f}s")
    with open(os.path.join(save_dir, "wall_clock_time.txt"), "w") as f:
        f.write(f"{elapsed:.4f}")


if __name__ == "__main__":
    main()