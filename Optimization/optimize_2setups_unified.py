#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""optimize_2setups_unified.py  —  joint 2-setup inverse, unified optimizer.

Log-space CMA + tightened (union) bounds + NN/prior warm-start + joint
H^PP preconditioning (sum of per-setup Hessians) + early stopping.
See Optimization/libs/cmaes_unified_core.py for design rationale.
"""
from __future__ import annotations
import argparse
import csv
import datetime
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.switch_backend("Agg")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vi_mogp.predict import HVIMoGPrBCMPredictor
from vi_mogp import config as HC
from Optimization.libs.cmaes_core import (
    PARAM_BOUNDS, check_feasibility, default_x0,
    MIN_N, MAX_N, MIN_ETA, MAX_ETA, MIN_SIGMA_Y, MAX_SIGMA_Y,
)
from Optimization.libs.cmaes_unified_core import run_cmaes_unified


def _union_bounds(b_list):
    if not b_list:
        return dict(PARAM_BOUNDS)
    out = {}
    for k in ("n", "eta", "sigma_y"):
        lo = max(min(b[k][0] for b in b_list), PARAM_BOUNDS[k][0])
        hi = min(max(b[k][1] for b in b_list), PARAM_BOUNDS[k][1])
        if hi <= lo:
            return dict(PARAM_BOUNDS)
        out[k] = (lo, hi)
    return out


def main():
    wall_start = time.time()
    default_model    = str(HC.DEFAULT_V2_MODEL)
    default_out_root = str(HC.DEFAULT_OUT)

    p = argparse.ArgumentParser(
        description="2-setup joint inverse: unified optimizer."
    )
    p.add_argument("-W1",   type=float, required=True)
    p.add_argument("-H1",   type=float, required=True)
    p.add_argument("-dis1", type=float, nargs=8, required=True)
    p.add_argument("-W2",   type=float, required=True)
    p.add_argument("-H2",   type=float, required=True)
    p.add_argument("-dis2", type=float, nargs=8, required=True)
    p.add_argument("-W3",   type=float, default=0.0)
    p.add_argument("-H3",   type=float, default=0.0)
    p.add_argument("-dis3", type=float, nargs=8, default=[])

    p.add_argument("--model", type=str, default=default_model)
    p.add_argument("--top-k-phi", dest="top_k_phi", type=int,
                   default=HC.INFER_TOP_K_PHI)
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--no-tighten-bounds", action="store_true")
    p.add_argument("--pad-frac", type=float, default=0.02)
    p.add_argument("--setup1_dir", type=str, default=None,
                   help="Directory with setup1_best_x_n_eta_sigma.txt")
    p.add_argument("--sigma0",   type=float, default=0.5)    # tighter, warm-started
    p.add_argument("--popsize",  type=int,   default=8)
    p.add_argument("--maxiter",  type=int,   default=250)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--verb",     type=int,   default=1)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--rel-tol",  type=float, default=0.003)
    p.add_argument("--out_dir",  type=str, default=None)
    args = p.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.out_dir) if args.out_dir \
               else Path(default_out_root) / f"setup2_unified_{ts}"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Info] Output directory: {save_dir}")

    cfgs = [
        {"id": 1, "W": args.W1, "H": args.H1, "y": np.array(args.dis1, float)},
        {"id": 2, "W": args.W2, "H": args.H2, "y": np.array(args.dis2, float)},
    ]
    if args.W3 > 0 and args.H3 > 0 and len(args.dis3) == 8:
        cfgs.append({"id": 3, "W": args.W3, "H": args.H3, "y": np.array(args.dis3, float)})
    print(f"[Info] #setups: {len(cfgs)}")

    predictor = HVIMoGPrBCMPredictor.load(args.model)

    per_cfg = []
    per_bounds = []
    for cfg in cfgs:
        gid, eids, wts = predictor.route_for(
            cfg["W"], cfg["H"], cfg["y"], top_k_phi=args.top_k_phi,
        )
        per_cfg.append({"geo_id": int(gid), "eids": eids, "wts": wts})
        print(f"[Route s{cfg['id']}] geo_id={gid}  eids={eids.tolist()}")
        if not args.no_tighten_bounds:
            per_bounds.append(predictor.tightened_bounds(
                eids, int(gid), PARAM_BOUNDS,
                include_baseline=True, pad_frac=args.pad_frac,
            ))
    bounds = _union_bounds(per_bounds) if per_bounds else dict(PARAM_BOUNDS)
    print(f"[Bounds] n∈{bounds['n']}  eta∈{bounds['eta']}  sigma_y∈{bounds['sigma_y']}")

    # Warm-start anchor (from setup1 if available)
    theta_0 = default_x0(bounds)
    prior_loaded = False
    if args.setup1_dir:
        prior_path = Path(args.setup1_dir) / "setup1_best_x_n_eta_sigma.txt"
        if prior_path.exists():
            theta_0 = list(np.loadtxt(prior_path))
            print(f"[Prior] loaded Setup-1: n={theta_0[0]:.4f}  "
                  f"η={theta_0[1]:.4f}  σ_y={theta_0[2]:.4f}")
            prior_loaded = True
    m_anchor = np.clip(
        np.asarray(theta_0, dtype=np.float64),
        [bounds["n"][0], bounds["eta"][0], bounds["sigma_y"][0]],
        [bounds["n"][1], bounds["eta"][1], bounds["sigma_y"][1]],
    )

    # Log-space prior regularisation (same weights as baseline for fairness)
    scale_n       = MAX_N - MIN_N
    scale_log_eta = math.log(MAX_ETA) - math.log(max(MIN_ETA, 1e-9))
    scale_log_sig = math.log(MAX_SIGMA_Y) - math.log(max(MIN_SIGMA_Y, 1e-9))
    LAMBDA_REG    = 0.05 if prior_loaded else 0.001
    norms = [max(float(np.mean(c["y"] ** 2)), 1e-12) for c in cfgs]

    def batch_loss(thetas):
        v = np.asarray(thetas, dtype=np.float64)
        B = v.shape[0]
        losses = np.zeros(B, dtype=np.float64)
        feasible = np.ones(B, dtype=bool)
        for i, th in enumerate(v):
            pen = sum(check_feasibility(th, c["H"], c["W"]) for c in cfgs)
            if pen > 0:
                losses[i] = pen
                feasible[i] = False
        if feasible.any():
            vv = v[feasible]
            total_nmse = np.zeros(len(vv), dtype=np.float64)
            try:
                for cfg_idx, cfg in enumerate(cfgs):
                    X = np.zeros((len(vv), 5), dtype=np.float64)
                    X[:, 0:3] = vv
                    X[:, 3]   = cfg["W"]
                    X[:, 4]   = cfg["H"]
                    y_pred = predictor.predict_fixed_route(
                        X, per_cfg[cfg_idx]["eids"], per_cfg[cfg_idx]["geo_id"],
                        clear_cache=False,
                        use_baseline=(not args.no_baseline),
                        phi_weights_row=per_cfg[cfg_idx].get("wts"),
                    )
                    mse = np.mean((y_pred - cfg["y"][None, :]) ** 2, axis=1)
                    total_nmse += mse / norms[cfg_idx]
                d_n   = ((vv[:, 0] - theta_0[0]) / scale_n) ** 2
                d_eta = ((np.log(np.maximum(vv[:, 1], 1e-9))
                          - math.log(max(theta_0[1], 1e-9))) / scale_log_eta) ** 2
                d_sig = ((np.log(np.maximum(vv[:, 2], 1e-9))
                          - math.log(max(theta_0[2], 1e-9))) / scale_log_sig) ** 2
                losses[feasible] = total_nmse / len(cfgs) + LAMBDA_REG * (d_n + d_eta + d_sig)
            except RuntimeError:
                for j, idx in enumerate(np.where(feasible)[0]):
                    th = vv[j]
                    try:
                        single = 0.0
                        for cfg_idx, cfg in enumerate(cfgs):
                            Xj = np.array([[*th, cfg["W"], cfg["H"]]], dtype=np.float64)
                            yp = predictor.predict_fixed_route(
                                Xj, per_cfg[cfg_idx]["eids"],
                                per_cfg[cfg_idx]["geo_id"],
                                clear_cache=False,
                                use_baseline=(not args.no_baseline),
                                phi_weights_row=per_cfg[cfg_idx].get("wts"),
                            )
                            single += float(np.mean((yp[0] - cfg["y"]) ** 2)
                                            / norms[cfg_idx])
                        losses[idx] = single / len(cfgs)
                    except Exception:
                        losses[idx] = 1e6
        return losses.tolist()

    setups_for_H = [(c["W"], c["H"]) for c in cfgs]
    print(f"\n--- CMA-ES unified (2-setup) start  "
          f"(popsize={args.popsize}, maxiter={args.maxiter}, σ0={args.sigma0:.3f}) ---")
    theta_best, loss_best, hist, diag, iter_times = run_cmaes_unified(
        batch_loss,
        setups=setups_for_H,
        m_init=m_anchor,
        bounds=bounds,
        popsize=args.popsize,
        maxiter=args.maxiter,
        sigma0=args.sigma0,
        seed=args.seed,
        verb_disp=args.verb,
        early_stop_patience=args.patience,
        early_stop_rel_tol=args.rel_tol,
        record_iter_times=True,
    )
    n_best, eta_best, sigma_best = theta_best
    print(f"\n{'='*60}")
    print(f"Best:  n={n_best:.6f}  η={eta_best:.6f}  σ_y={sigma_best:.6f}")
    print(f"Loss:  {loss_best:.6e}")
    if diag.get("hessian_available"):
        print(f"H_u eigenvalues: {['%.3e' % v for v in diag['H_u_eigvals']]}")
        print(f"Condition number: {diag['condition_number']:.3e}  (reported only)")
    print(f"Iterations: {diag['n_iter']}   (early_stopped={diag['early_stopped']})")
    print(f"Total fcalls: {diag['n_iter'] * args.popsize}")
    print(f"{'='*60}")

    np.savetxt(save_dir / "setup2_best_x_n_eta_sigma.txt", np.array([theta_best]))
    with open(save_dir / "unified_diag.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    with open(save_dir / "best_loss_history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Iteration", "Loss"])
        w.writerows(enumerate(hist))
    with open(save_dir / "iteration_times.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Iteration", "Duration_Seconds"])
        for i, t in enumerate(iter_times):
            w.writerow([i + 1, f"{t:.4f}"])

    plt.figure()
    plt.plot(hist); plt.yscale("log")
    plt.title(f"Convergence ({len(cfgs)}-setup unified)")
    plt.xlabel("Iteration"); plt.ylabel("Best Loss")
    plt.savefig(save_dir / "best_loss_convergence.png"); plt.close()

    # Flow-curve comparison per cfg
    try:
        import pandas as pd
        fc_rows = []
        for cfg_idx, cfg in enumerate(cfgs):
            X_best = np.array([[*theta_best, cfg["W"], cfg["H"]]], dtype=np.float64)
            y_pred = predictor.predict_fixed_route(
                X_best, per_cfg[cfg_idx]["eids"], per_cfg[cfg_idx]["geo_id"],
                use_baseline=(not args.no_baseline),
                phi_weights_row=per_cfg[cfg_idx].get("wts"),
            )[0]
            obs   = cfg["y"]
            resid = y_pred - obs
            print(f"  Setup {cfg['id']} (W={cfg['W']}, H={cfg['H']}):  "
                  f"RMSE={np.sqrt(np.mean(resid**2)):.6f}")
            for i in range(8):
                fc_rows.append(dict(setup=cfg["id"], W=cfg["W"], H=cfg["H"],
                                    frame=i+1, observed=obs[i],
                                    predicted=y_pred[i], residual=resid[i]))
        pd.DataFrame(fc_rows).to_csv(
            save_dir / "flowcurve_comparison.csv",
            index=False, float_format="%.8f",
        )
    except Exception as e:
        print(f"[Warning] Flow-curve saving failed: {e}")

    elapsed = time.time() - wall_start
    print(f"\n[Finished] Wall clock: {elapsed:.2f}s")
    (save_dir / "wall_clock_time.txt").write_text(f"{elapsed:.4f}")


if __name__ == "__main__":
    main()
