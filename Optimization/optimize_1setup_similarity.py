#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
optimize_1setup_similarity.py  —  CMA-ES with Hamamichi 2023 H^PP preconditioning
==============================================================================
Parallel to `optimize_1setup.py` but drives CMA in the coordinate system
defined by the plane-Poiseuille Hessian at the warm-start material guess.

Key difference vs the baseline:
  Baseline (optimize_1setup.py):
      log-space CMA over (n, log η, log σ_y) with isotropic σ0 = 1.0.
      No information about the loss ellipsoid.
  Similarity-aware (this script):
      Computes H^PP(M_ws, S) at warm-start M_ws, eigendecomposes it in
      material-extent-normalized coordinates, and runs CMA in the
      preconditioned z-space (σ0 = 0.3 works well post-preconditioning).

Same CLI as optimize_1setup.py. Outputs same artefact names so downstream
tools (e.g. 2setup prior loading, flow-curve plotting) work unchanged.
Also saves `similarity_diag.json` with eigenvalues / preconditioner info.
"""
from __future__ import annotations
import argparse
import csv
import datetime
import json
import os
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
from Optimization.libs.cmaes_similarity_core import run_cmaes_similarity

# Mechanism-based Setup 2 recommendation (same as baseline)
try:
    from Optimization.libs.param import Param
    from Optimization.libs.setup import Setup
    from Optimization.libs.mechanism import Mechanism
except ImportError:  # pragma: no cover
    print("[WARNING] Optimization/libs not on path. Setup-2 recommendation skipped.")
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


def _pick_anchor_for_hessian(predictor, W, H, eids, geo_id, y_obs, bounds):
    """Pick a feasible anchor point for the H^PP computation.

    Try, in order:
      1. nearest-neighbor in top-1 expert's training set (warm-start)
      2. midpoint of bounds (nudged slightly to ensure P*L > sigma_y)
    """
    nn = predictor.nearest_neighbor_theta(int(eids[0]), np.asarray(y_obs))
    if nn is not None:
        n, eta, sig = float(nn[0]), float(nn[1]), float(nn[2])
        n = float(np.clip(n, bounds["n"][0], bounds["n"][1]))
        eta = float(np.clip(eta, bounds["eta"][0], bounds["eta"][1]))
        sig = float(np.clip(sig, bounds["sigma_y"][0], bounds["sigma_y"][1]))
        return np.array([n, eta, sig], dtype=np.float64)
    # fallback: midpoint
    return np.asarray(default_x0(bounds), dtype=np.float64)


def main():
    wall_start = time.time()

    default_model = str(HC.DEFAULT_V2_MODEL)
    default_out_root = str(HC.DEFAULT_OUT)

    p = argparse.ArgumentParser(
        description="CMA-ES 1-setup inverse with Hamamichi H^PP preconditioning."
    )
    p.add_argument("-W1",    type=float, required=True)
    p.add_argument("-H1",    type=float, required=True)
    p.add_argument("-dis1",  type=float, nargs=8, required=True)
    p.add_argument("--model", type=str, default=default_model)
    p.add_argument("--top-k-phi", dest="top_k_phi", type=int,
                   default=HC.INFER_TOP_K_PHI)
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--sigma0",   type=float, default=0.30,
                   help="CMA σ0 in preconditioned z-space (default 0.30)")
    p.add_argument("--popsize",  type=int,   default=16)
    p.add_argument("--maxiter",  type=int,   default=300)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--verb",     type=int,   default=1)
    p.add_argument("--no-eig-scaling", action="store_true",
                   help="Use only rotation Q of H^PP, ignore eigenvalues "
                        "(per paper footnote 15, |λ| may not match DB).")
    p.add_argument("--out_dir",  type=str, default=None)
    p.add_argument("--material",     type=str, default=None)
    p.add_argument("--data_root",    type=str, default="../data")
    p.add_argument("--setup2_index", type=int, default=2)
    args = p.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        save_dir = Path(args.out_dir)
    else:
        save_dir = Path(default_out_root) / f"setup1_sim_{ts}"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Info] Output directory: {save_dir}")

    W1 = float(args.W1); H1 = float(args.H1)
    y1 = np.array(args.dis1, dtype=float)

    # ── Load v2 surrogate ─────────────────────────────────────────────────────
    predictor = HVIMoGPrBCMPredictor.load(args.model)

    # ── Route once (fixed) ───────────────────────────────────────────────────
    geo_id, eids, wts = predictor.route_for(
        W1, H1, y1, top_k_phi=args.top_k_phi,
    )
    print(f"[Route] geo_id={geo_id}  expert_ids={eids.tolist()}  "
          f"weights={np.round(wts, 4).tolist()}")

    # ── Anchor for the Hessian ───────────────────────────────────────────────
    m_anchor = _pick_anchor_for_hessian(
        predictor, W1, H1, eids, geo_id, y1, PARAM_BOUNDS,
    )
    print(f"[Anchor] Hessian pivot: n={m_anchor[0]:.4f}  "
          f"η={m_anchor[1]:.4f}  σ_y={m_anchor[2]:.4f}  "
          f"(nearest-neighbor in top-1 expert's training data)")

    # ── Build loss + run similarity-aware CMA ───────────────────────────────
    loss_fn = make_batch_loss(
        predictor, y1, W1, H1, eids, geo_id,
        use_baseline=(not args.no_baseline),
        wts=wts,
    )
    print(f"\n--- CMA-ES (H^PP-preconditioned) start "
          f"(popsize={args.popsize}, maxiter={args.maxiter}, "
          f"σ0={args.sigma0:.3f}) ---")
    theta_best, loss_best, hist, diag, iter_times = run_cmaes_similarity(
        loss_fn,
        setups=[(W1, H1)],
        m_init=m_anchor,
        bounds=PARAM_BOUNDS,
        sigma0=args.sigma0,
        popsize=args.popsize,
        maxiter=args.maxiter,
        seed=args.seed,
        verb_disp=args.verb,
        use_eig_scaling=(not args.no_eig_scaling),
        record_iter_times=True,
    )
    n_best, eta_best, sigma_best = theta_best
    print(f"\n{'='*60}")
    print(f"Best:  n={n_best:.6f}  η={eta_best:.6f}  σ_y={sigma_best:.6f}")
    print(f"Loss:  {loss_best:.6e}")
    print(f"H_u eigenvalues: {['%.3e' % v for v in diag['H_u_eigvals']]}")
    print(f"Condition number of H_u: {diag['condition_number']:.3e}")
    print(f"{'='*60}")

    # ── Save artefacts (same filenames as the baseline optimiser) ────────────
    np.savetxt(save_dir / "setup1_best_x_n_eta_sigma.txt", np.array([theta_best]))
    with open(save_dir / "similarity_diag.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in diag.items() if k != "preconditioner_Q"}, f, indent=2)
        # save Q separately as numpy text for readability
    np.savetxt(save_dir / "preconditioner_Q.txt",
               np.asarray(diag["preconditioner_Q"]), fmt="%.8f")

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
    plt.title("Convergence (similarity-preconditioned, 1-setup)")
    plt.xlabel("Iteration"); plt.ylabel("Best Loss")
    plt.savefig(save_dir / "best_loss_convergence.png"); plt.close()

    # ── Flow-curve comparison at θ_best ──────────────────────────────────────
    try:
        import pandas as pd
        X_best = np.array([[*theta_best, W1, H1]], dtype=np.float64)
        y_pred = predictor.predict_fixed_route(
            X_best, eids, geo_id, use_baseline=(not args.no_baseline),
            phi_weights_row=wts,
        )[0]
        resid = y_pred - y1
        print(f"\n  Setup 1 (W={W1}, H={H1}):")
        print(f"    Observed:  {np.array2string(y1,    precision=4)}")
        print(f"    Predicted: {np.array2string(y_pred, precision=4)}")
        print(f"    Residual:  {np.array2string(resid, precision=4)}")
        print(f"    RMSE:      {np.sqrt(np.mean(resid**2)):.6f}")
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
        ax.plot(frames, y_pred,"rs--", label="Predicted (rBCM v2 + H^PP CMA)", markersize=5)
        ax.fill_between(frames, y1, y_pred, alpha=0.2, color="salmon")
        ax.set_xlabel("Frame"); ax.set_ylabel("Flow distance (cm)")
        ax.set_title(f"Setup 1 (similarity CMA): W={W1}, H={H1}")
        rmse = np.sqrt(np.mean(resid ** 2))
        ax.legend(title=f"RMSE={rmse:.4f}"); ax.grid(True, alpha=0.3)
        fig.suptitle(f"n={n_best:.4f}, η={eta_best:.2f}, σ_y={sigma_best:.2f}",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(save_dir / "flowcurve_comparison.png", dpi=150); plt.close(fig)
    except Exception as e:
        print(f"[Warning] Flow-curve saving failed: {e}")

    # ── Setup 2 recommendation (unchanged — Mechanism) ──────────────────────
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
