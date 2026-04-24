#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""optimize_1setup_unified.py  —  unified optimizer (log-space + H^PP preconditioning).

Combines strengths of optimize_1setup.py (log-space + tightened bounds +
NN warm-start) and optimize_1setup_similarity.py (H^PP preconditioning).
See Optimization/libs/cmaes_unified_core.py for the design rationale.

Same CLI as optimize_1setup.py. Outputs same artefacts + a `unified_diag.json`
with condition number / eigenvalues for paper-level uncertainty reporting.
"""
from __future__ import annotations
import argparse
import csv
import datetime
import json
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
)
from Optimization.libs.cmaes_unified_core import run_cmaes_unified

try:
    from Optimization.libs.param import Param
    from Optimization.libs.setup import Setup
    from Optimization.libs.mechanism import Mechanism
except ImportError:  # pragma: no cover
    class Param:
        def __init__(self, eta, n, sigma_y): self.eta, self.n, self.sigma_y = eta, n, sigma_y
    class Setup:
        def __init__(self, H, W, w): self.H, self.W, self.w = H, W, w
    class Mechanism:
        def searchNewSetup_orthognality_for_second_setup(self, m, setups):
            return [None, Setup(setups[0].H, setups[0].W, 1.0)]


_SETTINGS_TEMPLATE = """\
<?xml version="1.0"?>
<Optimizer>
  <setup RHO="1.0" H="{H}" W="{W}" />
  <cuboid min="-0.15 -0.15 -0.15" max="{W} {H} 4.15"
          density="1.0" cell_samples_per_dim="2"
          vel="0.0 0.0 0.0" omega="0.0 0.0 0.0" />
  <static_box min="-100 -1 -100" max="100  0 100" boundary_behavior="sticking"/>
  <static_box min="-1   0  0"   max="0   20   4"  boundary_behavior="sticking"/>
  <static_box min="-1   0 -0.3" max="{W} 20   0"  boundary_behavior="sticking"/>
  <static_box min="-1   0  4"   max="{W} 20  4.3" boundary_behavior="sticking"/>
</Optimizer>
"""


def _write_setup_dir(data_root, material, H_cm, W_cm, index):
    dir_name  = f"ref_{material}_{H_cm:.1f}_{W_cm:.1f}_{index}"
    setup_dir = Path(data_root) / dir_name
    setup_dir.mkdir(parents=True, exist_ok=True)
    (setup_dir / "settings.xml").write_text(
        _SETTINGS_TEMPLATE.format(W=f"{W_cm:.1f}", H=f"{H_cm:.1f}")
    )
    return str(setup_dir)


def make_batch_loss(predictor, y_obs, W, H, eids, geo_id, use_baseline,
                    wts=None):
    y_obs = np.asarray(y_obs, dtype=np.float64)
    norm  = max(float(np.mean(y_obs ** 2)), 1e-12)

    def batch_loss(thetas):
        thetas = np.asarray(thetas, dtype=np.float64)
        B = thetas.shape[0]
        losses = np.zeros(B, dtype=np.float64)
        feasible = np.ones(B, dtype=bool)
        for i, th in enumerate(thetas):
            pen = check_feasibility(th, H, W)
            if pen > 0:
                losses[i] = pen
                feasible[i] = False
        if feasible.any():
            v = thetas[feasible]
            X = np.zeros((len(v), 5), dtype=np.float64)
            X[:, 0:3] = v
            X[:, 3]   = W
            X[:, 4]   = H
            try:
                y_pred = predictor.predict_fixed_route(
                    X, eids, geo_id, clear_cache=False,
                    use_baseline=use_baseline,
                    phi_weights_row=wts,
                )
                nmse = np.mean((y_pred - y_obs[None, :]) ** 2, axis=1) / norm
                losses[feasible] = nmse
            except RuntimeError:
                for j, (idx, th) in enumerate(zip(np.where(feasible)[0], v)):
                    try:
                        Xj = np.array([[*th, W, H]], dtype=np.float64)
                        yp = predictor.predict_fixed_route(
                            Xj, eids, geo_id, clear_cache=False,
                            use_baseline=use_baseline,
                            phi_weights_row=wts,
                        )
                        losses[idx] = float(np.mean((yp[0] - y_obs) ** 2) / norm)
                    except Exception:
                        losses[idx] = 1e6
        return losses.tolist()

    return batch_loss


def _pick_anchor(predictor, eids, y_obs, bounds):
    nn = predictor.nearest_neighbor_theta(int(eids[0]), np.asarray(y_obs))
    if nn is not None:
        return np.array([
            float(np.clip(nn[0], bounds["n"][0], bounds["n"][1])),
            float(np.clip(nn[1], bounds["eta"][0], bounds["eta"][1])),
            float(np.clip(nn[2], bounds["sigma_y"][0], bounds["sigma_y"][1])),
        ])
    return np.asarray(default_x0(bounds), dtype=np.float64)


def main():
    wall_start = time.time()
    default_model = str(HC.DEFAULT_V2_MODEL)
    default_out_root = str(HC.DEFAULT_OUT)

    p = argparse.ArgumentParser(
        description="1-setup inverse: unified (log-space + H^PP preconditioning)."
    )
    p.add_argument("-W1",    type=float, required=True)
    p.add_argument("-H1",    type=float, required=True)
    p.add_argument("-dis1",  type=float, nargs=8, required=True)
    p.add_argument("--model", type=str, default=default_model)
    p.add_argument("--top-k-phi", dest="top_k_phi", type=int,
                   default=HC.INFER_TOP_K_PHI)
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--no-tighten-bounds", action="store_true")
    p.add_argument("--pad-frac", type=float, default=0.02)
    p.add_argument("--sigma0",   type=float, default=1.0)
    p.add_argument("--popsize",  type=int,   default=8)      # default per sweep
    p.add_argument("--maxiter",  type=int,   default=250)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--verb",     type=int,   default=1)
    p.add_argument("--patience", type=int, default=40,
                   help="Early-stop: stop if best loss stalls for K gens (0=off).")
    p.add_argument("--rel-tol",  type=float, default=0.003,
                   help="Early-stop: relative-improvement threshold (default 0.3%).")
    p.add_argument("--out_dir",  type=str, default=None)
    p.add_argument("--material",     type=str, default=None)
    p.add_argument("--data_root",    type=str, default="../data")
    p.add_argument("--setup2_index", type=int, default=2)
    args = p.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.out_dir) if args.out_dir \
               else Path(default_out_root) / f"setup1_unified_{ts}"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Info] Output directory: {save_dir}")

    W1 = float(args.W1); H1 = float(args.H1)
    y1 = np.array(args.dis1, dtype=float)
    print(f"[Info] Geometry: W={W1} cm, H={H1} cm")

    predictor = HVIMoGPrBCMPredictor.load(args.model)

    geo_id, eids, wts = predictor.route_for(
        W1, H1, y1, top_k_phi=args.top_k_phi,
    )
    print(f"[Route] geo_id={geo_id}  expert_ids={eids.tolist()}  "
          f"weights={np.round(wts, 4).tolist()}")

    # Tightened bounds (baseline-style)
    if args.no_tighten_bounds:
        bounds = dict(PARAM_BOUNDS)
    else:
        bounds = predictor.tightened_bounds(
            eids, geo_id, PARAM_BOUNDS,
            include_baseline=True, pad_frac=args.pad_frac,
        )
    print(f"[Bounds] n∈{bounds['n']}  eta∈{bounds['eta']}  sigma_y∈{bounds['sigma_y']}")

    # NN warm-start (baseline-style, also used as Hessian anchor)
    m_anchor = _pick_anchor(predictor, eids, y1, bounds)
    print(f"[Anchor] n={m_anchor[0]:.4f}  η={m_anchor[1]:.4f}  σ_y={m_anchor[2]:.4f}  "
          "(NN in top-1 expert; used as both x0 and Hessian pivot)")

    loss_fn = make_batch_loss(
        predictor, y1, W1, H1, eids, geo_id,
        use_baseline=(not args.no_baseline),
        wts=wts,
    )
    print(f"\n--- CMA-ES unified start "
          f"(popsize={args.popsize}, maxiter={args.maxiter}, σ0={args.sigma0:.3f}, "
          f"patience={args.patience}, rel_tol={args.rel_tol}) ---")
    theta_best, loss_best, hist, diag, iter_times = run_cmaes_unified(
        loss_fn,
        setups=[(W1, H1)],
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
        print(f"Condition number: {diag['condition_number']:.3e}  (reported only -- not used for search)")
    print(f"Iterations:  {diag['n_iter']}  (early_stopped={diag['early_stopped']})")
    print(f"Total fcalls: {diag['n_iter'] * args.popsize}")
    print(f"{'='*60}")

    # Save artefacts (same filenames as baseline optimiser)
    np.savetxt(save_dir / "setup1_best_x_n_eta_sigma.txt", np.array([theta_best]))
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
    plt.title("Convergence (unified, 1-setup)")
    plt.xlabel("Iteration"); plt.ylabel("Best Loss")
    plt.savefig(save_dir / "best_loss_convergence.png"); plt.close()

    # Flow-curve comparison at θ_best
    try:
        import pandas as pd
        X_best = np.array([[*theta_best, W1, H1]], dtype=np.float64)
        y_pred = predictor.predict_fixed_route(
            X_best, eids, geo_id, use_baseline=(not args.no_baseline),
            phi_weights_row=wts,
        )[0]
        resid = y_pred - y1
        fc_rows = [dict(setup=1, W=W1, H=H1, frame=i+1,
                        observed=y1[i], predicted=y_pred[i], residual=resid[i])
                   for i in range(8)]
        pd.DataFrame(fc_rows).to_csv(
            save_dir / "flowcurve_comparison.csv",
            index=False, float_format="%.8f",
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        frames = np.arange(1, 9)
        ax.plot(frames, y1,    "ko-",  label="Observed",  markersize=6)
        ax.plot(frames, y_pred,"rs--", label="Predicted (unified)", markersize=5)
        ax.fill_between(frames, y1, y_pred, alpha=0.2, color="salmon")
        rmse = np.sqrt(np.mean(resid ** 2))
        ax.set_xlabel("Frame"); ax.set_ylabel("Flow distance (cm)")
        ax.set_title(f"Setup 1 (unified): W={W1}, H={H1}")
        ax.legend(title=f"RMSE={rmse:.4f}"); ax.grid(True, alpha=0.3)
        fig.suptitle(f"n={n_best:.4f}, η={eta_best:.2f}, σ_y={sigma_best:.2f}",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(save_dir / "flowcurve_comparison.png", dpi=150); plt.close(fig)
    except Exception as e:
        print(f"[Warning] Flow-curve saving failed: {e}")

    # Setup 2 recommendation (unchanged)
    try:
        m_breve    = Param(eta_best, n_best, sigma_best)
        s1         = Setup(H1, W1, 1.0)
        new_setups = Mechanism().searchNewSetup_orthognality_for_second_setup(
            m_breve, [s1],
        )
        s2 = new_setups[1]
        print(f"\n[Mechanism] Recommended Setup 2: W={s2.W:.3f}  H={s2.H:.3f}")
        np.savetxt(save_dir / "setup1_recommended_setup2_WH.txt",
                   np.array([[s2.W, s2.H]]))
        if args.material:
            setup2_dir = _write_setup_dir(
                args.data_root, args.material, s2.H, s2.W, args.setup2_index,
            )
            print(f"           data dir: {setup2_dir}")
    except Exception as e:
        print(f"[Error] Mechanism search failed: {e}")

    elapsed = time.time() - wall_start
    print(f"\n[Finished] Wall clock: {elapsed:.2f}s")
    (save_dir / "wall_clock_time.txt").write_text(f"{elapsed:.4f}")


if __name__ == "__main__":
    main()
