#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
optimize_2setups.py  —  CMA-ES HB inverse from TWO (or THREE) dam-break setups
==============================================================================
Joint inverse over 2 or 3 geometry/observation tuples using the HVIMoGP-rBCM
v2 surrogate. Sequel to optimize_1setup.py: Setup 2 is typically the
geometry suggested by `Mechanism.searchNewSetup_orthognality_for_second_setup`
and its observation is collected with the real apparatus (or simulator).

This script reuses the Setup-1 result as CMA x0 (soft prior), then drives the
combined NMSE across all cfgs as the objective. Each cfg routes its own
expert set (fixed per-cfg across iters).

Outputs (written to `OptimizationResults/<out_dir>/`)
-----------------------------------------------------
    setup2_best_x_n_eta_sigma.txt
    setup2_recommended_setup3_WH.txt    (optional; Mechanism Setup 3)
    setup3_settings.xml                 (optional)
    best_loss_history.csv
    iteration_times.csv
    best_loss_convergence.png
    flowcurve_comparison.csv / .png
    wall_clock_time.txt
"""
from __future__ import annotations
import argparse
import csv
import datetime
import math
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.switch_backend("Agg")

# Repo root on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vi_mogp.predict import HVIMoGPrBCMPredictor
from vi_mogp import config as HC
from Optimization.libs.cmaes_core import (
    PARAM_BOUNDS, check_feasibility, run_cmaes, default_x0,
    MIN_N, MAX_N, MIN_ETA, MAX_ETA, MIN_SIGMA_Y, MAX_SIGMA_Y,
)

# Mechanism-based Setup 3 recommendation (pure geometry, no surrogate dep)
try:
    from Optimization.libs.param import Param
    from Optimization.libs.setup import Setup
    from Optimization.libs.mechanism import Mechanism
except ImportError:  # pragma: no cover
    print("[WARNING] Optimization/libs not on path. Setup-3 recommendation skipped.")
    class Param:
        def __init__(self, eta, n, sigma_y): self.eta, self.n, self.sigma_y = eta, n, sigma_y
    class Setup:
        def __init__(self, H, W, w): self.H, self.W, self.w = H, W, w
    class Mechanism:
        def searchNewSetup_orthognality_for_third_setup(self, m, setups):
            return [None, None, Setup(setups[-1].H, setups[-1].W, 1.0)]


# ── settings.xml template for an auto-created Setup 3 data dir ──────────────
def write_settings_xml(path: str, W: float, H: float, RHO: float = 1.0):
    xml = f"""<?xml version="1.0"?>
<Optimizer>
  <path
    root_dir_path="../data/Tonkatsu_1"
    GL_render_path="../libs/3D/GLRender3d/build/GLRender3d"
    mpm_path="../libs/3D/MPM3d/AGTaichiMPM.py"
    particle_skinner_path="../libs/ParticleSkinner3DTaichi.py"
    shell_script_dir_path="../libs/3D/shellScript3d"
    GL_emulation_render_path="../libs/3D/GLEmulationRender3d/build/GLEmulationRender3d"
  />

  <setup
    RHO="{RHO}"
    H="{H}"
    W="{W}"
  />
  <cuboid min="-0.150000 -0.150000 -0.150000" max="{W} {H} 4.150000" density="{RHO}" cell_samples_per_dim="2" vel="0.0 0.0 0.0" omega="0.0 0.0 0.0" />
  <static_box min="-100.000000 -1.000000 -100.000000" max="100.000000 0.000000 100.000000" boundary_behavior="sticking"/>
  <static_box min="-1.000000 0.000000 0.000000" max="0.000000 20.000000 4.000000" boundary_behavior="sticking"/>
  <static_box min="-1.000000 0.000000 -0.300000" max="{W} 20.000000 0.000000" boundary_behavior="sticking"/>
  <static_box min="-1.000000 0.000000 4.000000" max="{W} 20.000000 4.300000" boundary_behavior="sticking"/>

</Optimizer>
"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(xml)


# ── Per-setup route & bounds union across setups ────────────────────────────
def _union_bounds(b_list):
    """Take axis-wise union of tightened-bound dicts, intersected with
    PARAM_BOUNDS. Fallback: return PARAM_BOUNDS."""
    if not b_list:
        return dict(PARAM_BOUNDS)
    out = {}
    for k in ("n", "eta", "sigma_y"):
        lo = min(b[k][0] for b in b_list)
        hi = max(b[k][1] for b in b_list)
        lo = max(lo, PARAM_BOUNDS[k][0])
        hi = min(hi, PARAM_BOUNDS[k][1])
        if hi <= lo:
            return dict(PARAM_BOUNDS)
        out[k] = (lo, hi)
    return out


def main():
    wall_start = time.time()

    default_model    = str(HC.DEFAULT_V2_MODEL)
    default_out_root = str(HC.DEFAULT_OUT)

    p = argparse.ArgumentParser(description="CMA-ES HB inverse: 2-setup (HVIMoGP-rBCM v2)")
    # geometry + obs — Setup 1 and 2 required, Setup 3 optional
    p.add_argument("-W1",   type=float, required=True)
    p.add_argument("-H1",   type=float, required=True)
    p.add_argument("-dis1", type=float, nargs=8, required=True)
    p.add_argument("-W2",   type=float, required=True)
    p.add_argument("-H2",   type=float, required=True)
    p.add_argument("-dis2", type=float, nargs=8, required=True)
    p.add_argument("-W3",   type=float, default=0.0)
    p.add_argument("-H3",   type=float, default=0.0)
    p.add_argument("-dis3", type=float, nargs=8, default=[])

    # surrogate
    p.add_argument("--model", type=str, default=default_model,
                   help=f"Path to v2 model.pt (default: {default_model})")
    p.add_argument("--top-k-phi", dest="top_k_phi", type=int,
                   default=HC.INFER_TOP_K_PHI)
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--no-tighten-bounds", action="store_true")
    p.add_argument("--pad-frac", type=float, default=0.02)

    # Setup-1 prior (warm start)
    p.add_argument("--setup1_dir", type=str, default=None,
                   help="Directory containing setup1_best_x_n_eta_sigma.txt "
                        "(used as CMA x0 and soft prior anchor)")

    # CMA-ES
    p.add_argument("--sigma0",   type=float, default=0.5)
    p.add_argument("--popsize",  type=int,   default=16)
    p.add_argument("--maxiter",  type=int,   default=700)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--verb",     type=int,   default=1)

    # output / Setup 3
    p.add_argument("--out_dir",  type=str, default=None,
                   help=f"Output directory. Default: {default_out_root}/setup2_<ts>/")
    p.add_argument("--gen_setup3_data_dir", type=str, default=None,
                   help="If given, also creates a new data directory at this "
                        "path with a settings.xml for the recommended Setup 3.")
    p.add_argument("--material_name", type=str, default="material",
                   help="Material name used in auto-generated Setup 3 dir name.")
    args = p.parse_args()

    # ── Output directory ──────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        save_dir = Path(args.out_dir)
    else:
        save_dir = Path(default_out_root) / f"setup2_{ts}"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Info] Output directory: {save_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")
    print(f"[Info] Surrogate: {args.model}")

    # ── Build cfg list ────────────────────────────────────────────────────────
    cfgs = [
        {"id": 1, "W": args.W1, "H": args.H1, "y": np.array(args.dis1, float)},
        {"id": 2, "W": args.W2, "H": args.H2, "y": np.array(args.dis2, float)},
    ]
    if args.W3 > 0 and args.H3 > 0 and len(args.dis3) == 8:
        cfgs.append({"id": 3, "W": args.W3, "H": args.H3, "y": np.array(args.dis3, float)})
    print(f"[Info] #setups: {len(cfgs)}")

    # ── Load v2 surrogate ─────────────────────────────────────────────────────
    predictor = HVIMoGPrBCMPredictor.load(args.model)

    # ── Route each cfg, collect tightened bounds ─────────────────────────────
    per_cfg = []
    per_bounds = []
    for cfg in cfgs:
        gid, eids, wts = predictor.route_for(
            cfg["W"], cfg["H"], cfg["y"], top_k_phi=args.top_k_phi,
        )
        per_cfg.append({"geo_id": int(gid), "eids": eids, "wts": wts})
        print(f"[Route s{cfg['id']}] geo_id={gid}  eids={eids.tolist()}  "
              f"wts={np.round(wts, 4).tolist()}")
        if not args.no_tighten_bounds:
            per_bounds.append(predictor.tightened_bounds(
                eids, int(gid), PARAM_BOUNDS,
                include_baseline=True, pad_frac=args.pad_frac,
            ))

    bounds = _union_bounds(per_bounds) if per_bounds else dict(PARAM_BOUNDS)
    print(f"[Bounds] n∈{bounds['n']}  eta∈{bounds['eta']}  sigma_y∈{bounds['sigma_y']}")

    # ── Warm-start x0 from Setup-1 result ────────────────────────────────────
    theta_0 = default_x0(bounds)
    prior_loaded = False
    if args.setup1_dir:
        prior_path = Path(args.setup1_dir) / "setup1_best_x_n_eta_sigma.txt"
        if prior_path.exists():
            try:
                theta_0 = list(np.loadtxt(prior_path))
                print(f"[Prior] loaded Setup-1: n={theta_0[0]:.4f}  "
                      f"η={theta_0[1]:.4f}  σ_y={theta_0[2]:.4f}")
                prior_loaded = True
            except Exception as e:
                print(f"[Warn] failed to load prior: {e}")
        else:
            print(f"[Warn] prior file not found: {prior_path}")

    # Regularisation weights (log-space) — tighter when prior loaded
    scale_n       = MAX_N - MIN_N
    scale_log_eta = math.log(MAX_ETA) - math.log(max(MIN_ETA, 1e-9))
    scale_log_sig = math.log(MAX_SIGMA_Y) - math.log(max(MIN_SIGMA_Y, 1e-9))
    LAMBDA_REG    = 0.05 if prior_loaded else 0.001

    # ── Loss ──────────────────────────────────────────────────────────────────
    norms = [max(float(np.mean(c["y"] ** 2)), 1e-12) for c in cfgs]

    def loss_fn(thetas):
        v = np.asarray(thetas, dtype=np.float64)
        B = v.shape[0]
        losses = np.zeros(B, dtype=np.float64)

        # Feasibility gate (sum across cfgs)
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

                # Log-space regularisation toward prior / midpoint
                d_n   = ((vv[:, 0] - theta_0[0]) / scale_n) ** 2
                d_eta = ((np.log(np.maximum(vv[:, 1], 1e-9))
                          - math.log(max(theta_0[1], 1e-9))) / scale_log_eta) ** 2
                d_sig = ((np.log(np.maximum(vv[:, 2], 1e-9))
                          - math.log(max(theta_0[2], 1e-9))) / scale_log_sig) ** 2
                prior_loss = LAMBDA_REG * (d_n + d_eta + d_sig)
                final = total_nmse / len(cfgs) + prior_loss
                losses[feasible] = final
            except RuntimeError:
                # Per-row fallback.
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

    # ── Run CMA ───────────────────────────────────────────────────────────────
    print(f"\n--- CMA-ES start (popsize={args.popsize}, maxiter={args.maxiter}) ---")
    theta_best, loss_best, hist, iter_times = run_cmaes(
        loss_fn, bounds,
        x0=list(theta_0), sigma0=args.sigma0,
        popsize=args.popsize, maxiter=args.maxiter,
        seed=args.seed, verb_disp=args.verb,
        record_iter_times=True,
    )
    n_best, eta_best, sigma_best = theta_best
    print(f"\n{'='*60}")
    print(f"Best:  n={n_best:.6f}  η={eta_best:.6f}  σ_y={sigma_best:.6f}")
    print(f"Loss:  {loss_best:.6e}")
    if prior_loaded:
        print(f"Prior: n={theta_0[0]:.6f}  η={theta_0[1]:.6f}  σ_y={theta_0[2]:.6f}")
    print(f"{'='*60}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    np.savetxt(save_dir / "setup2_best_x_n_eta_sigma.txt", np.array([theta_best]))

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
    plt.title(f"Convergence ({len(cfgs)}-setup)")
    plt.xlabel("Iteration"); plt.ylabel("Best Loss")
    plt.savefig(save_dir / "best_loss_convergence.png"); plt.close()

    # ── Flow-curve comparison per cfg ────────────────────────────────────────
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
            print(f"  Setup {cfg['id']} (W={cfg['W']}, H={cfg['H']}):")
            print(f"    Observed:  {np.array2string(obs,    precision=4)}")
            print(f"    Predicted: {np.array2string(y_pred, precision=4)}")
            print(f"    Residual:  {np.array2string(resid,  precision=4)}")
            print(f"    RMSE:      {np.sqrt(np.mean(resid**2)):.6f}")
            for i in range(8):
                fc_rows.append(dict(
                    setup=cfg["id"], W=cfg["W"], H=cfg["H"], frame=i + 1,
                    observed=obs[i], predicted=y_pred[i], residual=resid[i],
                ))

        fc_df = pd.DataFrame(fc_rows)
        fc_df.to_csv(save_dir / "flowcurve_comparison.csv",
                     index=False, float_format="%.8f")

        fig, axes = plt.subplots(1, len(cfgs), figsize=(6 * len(cfgs), 5), squeeze=False)
        frames = np.arange(1, 9)
        for cfg_idx, cfg in enumerate(cfgs):
            ax  = axes[0, cfg_idx]
            sub = fc_df[fc_df["setup"] == cfg["id"]]
            ax.plot(frames, sub["observed"].values,  "ko-",  label="Observed",  markersize=6)
            ax.plot(frames, sub["predicted"].values, "rs--", label="Predicted (rBCM v2)", markersize=5)
            ax.fill_between(frames, sub["observed"].values, sub["predicted"].values,
                            alpha=0.2, color="salmon")
            ax.set_xlabel("Frame"); ax.set_ylabel("Flow distance (cm)")
            ax.set_title(f"Setup {cfg['id']}: W={cfg['W']}, H={cfg['H']}")
            rmse = np.sqrt(np.mean(sub["residual"].values ** 2))
            ax.legend(title=f"RMSE={rmse:.4f}"); ax.grid(True, alpha=0.3)
        fig.suptitle(f"Flow Curves: n={n_best:.4f}, η={eta_best:.2f}, σ_y={sigma_best:.2f}",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(save_dir / "flowcurve_comparison.png", dpi=150); plt.close(fig)
    except Exception as e:
        print(f"[Warning] Flow-curve saving failed: {e}")

    # ── Setup 3 recommendation ───────────────────────────────────────────────
    try:
        m_breve = Param(theta_best[1], theta_best[0], theta_best[2])
        setups  = [Setup(c["H"], c["W"], 1.0) for c in cfgs]
        new_setups = Mechanism().searchNewSetup_orthognality_for_third_setup(
            m_breve, setups,
        )
        if new_setups and len(new_setups) > 2:
            s3 = new_setups[2]
            print(f"\n[Mechanism] Recommended Setup 3: W={s3.W:.3f}  H={s3.H:.3f}")
            np.savetxt(save_dir / "setup2_recommended_setup3_WH.txt",
                       np.array([[s3.W, s3.H]]))

            settings_template = save_dir / "setup3_settings.xml"
            write_settings_xml(str(settings_template), s3.W, s3.H)

            target_dir = (
                args.gen_setup3_data_dir if args.gen_setup3_data_dir
                else os.path.join("..", "data",
                                  f"ref_{args.material_name}_{s3.H:.1f}_{s3.W:.1f}_3")
            )
            os.makedirs(target_dir, exist_ok=True)
            write_settings_xml(os.path.join(target_dir, "settings.xml"), s3.W, s3.H)
            print(f"            data dir: {target_dir}/")
    except Exception as e:
        print(f"[Error] Mechanism search failed: {e}")

    elapsed = time.time() - wall_start
    print(f"\n[Finished] Wall clock: {elapsed:.2f}s")
    (save_dir / "wall_clock_time.txt").write_text(f"{elapsed:.4f}")


if __name__ == "__main__":
    main()
