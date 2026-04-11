#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
optimize_3setups.py
===================
CMA-ES parameter estimation from THREE dam-break setups.

Mirrors optimize_2setups.py:
  * uses MoE soft routing per setup,
  * loads Setup 1 (and optionally Setup 2) priors as warm-start + regularizer,
  * recommends Setup 4 via Mechanism.searchNewSetup_orthognality_for_forth_setup,
  * writes a `setup4_settings.xml` template and an auto-generated data dir
    for the recommended Setup 4.
"""

import argparse
import os
import math
import datetime
import csv

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("Agg")

import torch

from libs.moe_core import (
    GLOBAL_BOUNDS,
    MIN_N, MAX_N, MIN_ETA, MAX_ETA, MIN_SIGMA_Y, MAX_SIGMA_Y,
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

try:
    from libs.param import Param
    from libs.setup import Setup
    from libs.mechanism import Mechanism
except ImportError:
    print("[WARNING] 'libs' not found. Setup 4 recommendation will be skipped.")
    class Param:
        def __init__(self, eta, n, sigma_y): self.eta, self.n, self.sigma_y = eta, n, sigma_y
    class Setup:
        def __init__(self, H, W, w): self.H, self.W, self.w = H, W, w
    class Mechanism:
        def searchNewSetup_orthognality_for_forth_setup(self, m, setups):
            return [None, None, None, Setup(setups[-1].H, setups[-1].W, 1.0)]


def write_settings_xml(path: str, W: float, H: float, RHO: float = 1.0):
    """Write a dam-break settings.xml for geometry (W, H)."""
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


def main():
    import time
    wall_start = time.time()

    p = argparse.ArgumentParser(description="CMA-ES optimization: 3-setup")
    p.add_argument("-W1",   type=float, required=True)
    p.add_argument("-H1",   type=float, required=True)
    p.add_argument("-dis1", type=float, nargs=8, required=True)
    p.add_argument("-W2",   type=float, required=True)
    p.add_argument("-H2",   type=float, required=True)
    p.add_argument("-dis2", type=float, nargs=8, required=True)
    p.add_argument("-W3",   type=float, required=True)
    p.add_argument("-H3",   type=float, required=True)
    p.add_argument("-dis3", type=float, nargs=8, required=True)
    p.add_argument("--setup1_dir", type=str, default=None,
                   help="Directory from optimize_1setup.py "
                        "containing setup1_best_x_n_eta_sigma.txt (used as warm-start prior)")
    p.add_argument("--setup2_dir", type=str, default=None,
                   help="Directory from optimize_2setups.py "
                        "containing setup2_best_x_n_eta_sigma.txt (preferred prior; overrides setup1_dir)")
    p.add_argument("--moe_dir",              type=str,   required=True)
    p.add_argument("--strategy",             type=str,   default="threshold",
                   choices=["topk", "threshold", "adaptive", "all"])
    p.add_argument("--threshold",            type=float, default=0.01)
    p.add_argument("--topk",                 type=int,   default=1)
    p.add_argument("--confidence_threshold", type=float, default=0.7)
    p.add_argument("--max_experts",          type=int,   default=5)
    p.add_argument("--sigma0",   type=float, default=0.5)
    p.add_argument("--popsize",  type=int,   default=16)
    p.add_argument("--maxiter",  type=int,   default=700)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--verb",     type=int,   default=1)
    p.add_argument("--out_dir",  type=str,   default=None,
                   help="Output directory. Default: result_setup3_<strategy>_<ts>/")
    p.add_argument("--gen_setup4_data_dir", type=str, default=None,
                   help="If given, also creates a new data directory at this path with "
                        "settings.xml for the recommended Setup 4.")
    p.add_argument("--material_name", type=str, default="material",
                   help="Material name used when constructing the auto-generated Setup 4 data dir name "
                        "(default: 'material')")
    args = p.parse_args()

    # ── Output directory ──────────────────────────────────────────────────────
    ts           = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_str = f"topk_{args.topk}" if args.strategy == "topk" \
                   else f"{args.strategy}_{args.threshold}"
    save_dir = args.out_dir if args.out_dir else f"result_setup3_{strategy_str}_{ts}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n[Info] Output directory: {save_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    # ── Build config list ─────────────────────────────────────────────────────
    cfgs = [
        {"id": 1, "W": args.W1, "H": args.H1, "y": np.array(args.dis1, float)},
        {"id": 2, "W": args.W2, "H": args.H2, "y": np.array(args.dis2, float)},
        {"id": 3, "W": args.W3, "H": args.H3, "y": np.array(args.dis3, float)},
    ]

    # ── Load MoE gate and select experts per config ───────────────────────────
    gate_dict = maybe_load_joblib(os.path.join(args.moe_dir, "gmm_gate.joblib"))
    norms     = [np.mean(c["y"] ** 2) + 1e-9 for c in cfgs]

    needed_experts = set()
    config_experts = []
    for cfg in cfgs:
        phi    = build_phi(cfg["y"], cfg["W"], cfg["H"])
        ids, w = get_adaptive_weights(
            gate_dict, phi,
            strategy=args.strategy,
            threshold=args.threshold,
            topk_hard=args.topk,
            confidence_threshold=args.confidence_threshold,
            max_experts=args.max_experts,
        )
        config_experts.append((ids, w))
        needed_experts.update(ids)

    # ── Load expert models ────────────────────────────────────────────────────
    expert_cache = {}
    print(f"Loading {len(needed_experts)} experts...")
    for cid in needed_experts:
        try:
            path = os.path.join(args.moe_dir, f"expert_{cid}.pt")
            if os.path.exists(path):
                expert_cache[cid] = load_expert_bundle(path, device)
        except Exception:
            pass

    # ── Load initial guess (Setup 2 prior preferred, else Setup 1) ────────────
    bounds  = GLOBAL_BOUNDS.copy()
    theta_0 = default_x0(bounds)
    prior_loaded = False

    if args.setup2_dir:
        prior_path = os.path.join(args.setup2_dir, "setup2_best_x_n_eta_sigma.txt")
        if os.path.exists(prior_path):
            try:
                theta_0 = list(np.loadtxt(prior_path))
                print(f"[Info] Loaded Setup 2 prior: n={theta_0[0]:.4f}  eta={theta_0[1]:.4f}  sigma_y={theta_0[2]:.4f}")
                prior_loaded = True
            except Exception as e:
                print(f"[ERROR] Failed to load Setup 2 prior: {e}")
        else:
            print(f"[ERROR] Setup 2 file not found: {prior_path}")

    if not prior_loaded and args.setup1_dir:
        prior_path = os.path.join(args.setup1_dir, "setup1_best_x_n_eta_sigma.txt")
        if os.path.exists(prior_path):
            try:
                theta_0 = list(np.loadtxt(prior_path))
                print(f"[Info] Loaded Setup 1 prior: n={theta_0[0]:.4f}  eta={theta_0[1]:.4f}  sigma_y={theta_0[2]:.4f}")
                prior_loaded = True
            except Exception as e:
                print(f"[ERROR] Failed to load Setup 1 prior: {e}")
        else:
            print(f"[ERROR] Setup 1 file not found: {prior_path}")

    if not prior_loaded:
        print("\n" + "!" * 40)
        print("[WARNING] No prior given. Starting from global center.")
        print("!" * 40 + "\n")

    # ── Log-space scale factors for regularisation and barrier ────────────────
    scale_n       = MAX_N - MIN_N
    scale_log_eta = math.log(MAX_ETA) - math.log(MIN_ETA)
    scale_log_sig = math.log(MAX_SIGMA_Y) - math.log(max(MIN_SIGMA_Y, 1e-6))
    LAMBDA_REG    = 0.05
    BARRIER_WT    = 0.001

    local_scale = {}
    for k in ["n", "eta", "sigma_y"]:
        local_scale[k] = bounds[k][1] - bounds[k][0]
        if k in ["eta", "sigma_y"]:
            low = max(bounds[k][0], 1e-9)
            local_scale[f"log_{k}"] = math.log(bounds[k][1]) - math.log(low)

    # ── Loss function ─────────────────────────────────────────────────────────
    def loss_fn(thetas):
        losses = np.zeros(len(thetas))
        valid_idx, valid_params = [], []

        for i, theta in enumerate(thetas):
            pen = sum(check_feasibility(theta, cfg["H"], cfg["W"]) for cfg in cfgs)
            if pen > 0:
                losses[i] = pen
            else:
                valid_idx.append(i)
                valid_params.append(theta)

        if not valid_idx:
            return losses.tolist()

        v = np.array(valid_params)

        # Regularisation: distance from prior in log-space
        d_n   = ((v[:, 0] - theta_0[0]) / scale_n) ** 2
        d_eta = ((np.log(v[:, 1]) - math.log(theta_0[1])) / scale_log_eta) ** 2
        sig0  = max(theta_0[2], 1e-9)
        d_sig = ((np.log(np.maximum(v[:, 2], 1e-9)) - math.log(sig0)) / scale_log_sig) ** 2
        prior_loss = LAMBDA_REG * (d_n + d_eta + d_sig)

        # Log-barrier
        eps = 1e-9

        def _barrier(vals_norm):
            v_safe = np.clip(vals_norm, eps, 1.0 - eps)
            return -np.log(v_safe) - np.log(1.0 - v_safe)

        barrier = _barrier((v[:, 0] - bounds["n"][0]) / local_scale["n"])
        barrier += _barrier(
            (np.log(v[:, 1]) - math.log(max(bounds["eta"][0], 1e-9))) / local_scale["log_eta"]
        )
        barrier += _barrier(
            (np.log(np.maximum(v[:, 2], 1e-9)) - math.log(max(bounds["sigma_y"][0], 1e-9)))
            / local_scale["log_sigma_y"]
        )

        cpu_loss = prior_loss + BARRIER_WT * barrier

        try:
            total_nmse = np.zeros(len(valid_idx))
            for cfg_idx, cfg in enumerate(cfgs):
                e_ids, e_wts = config_experts[cfg_idx]
                preds    = soft_predict_batch(valid_params, expert_cache, e_ids, e_wts, cfg["W"], cfg["H"], device)
                mse_vals = np.mean((preds - cfg["y"].reshape(1, -1)) ** 2, axis=1)
                total_nmse += mse_vals / norms[cfg_idx]
            final_loss = (total_nmse / len(cfgs)) + cpu_loss
            for i, val in zip(valid_idx, final_loss):
                losses[i] = val

        except RuntimeError:
            for j, (i, param) in enumerate(zip(valid_idx, valid_params)):
                try:
                    single_nmse = 0.0
                    for cfg_idx, cfg in enumerate(cfgs):
                        e_ids, e_wts = config_experts[cfg_idx]
                        pp = soft_predict_batch([param], expert_cache, e_ids, e_wts, cfg["W"], cfg["H"], device)
                        single_nmse += np.mean((pp - cfg["y"].reshape(1, -1)) ** 2) / norms[cfg_idx]
                    losses[i] = (single_nmse / len(cfgs)) + cpu_loss[j]
                except Exception:
                    losses[i] = 1e6

        return losses.tolist()

    # ── CMA-ES ────────────────────────────────────────────────────────────────
    print(f"\n--- Optimization Start (popsize={args.popsize}) ---")
    theta_best, loss_best, hist, iter_times = run_cmaes(
        loss_fn, bounds,
        x0=list(theta_0), sigma0=args.sigma0,
        popsize=args.popsize, maxiter=args.maxiter,
        seed=args.seed, verb_disp=args.verb,
        record_iter_times=True,
    )
    print(f"Best: {theta_best}  loss={loss_best:.6e}")

    # ── Save results ──────────────────────────────────────────────────────────
    np.savetxt(os.path.join(save_dir, "setup3_best_x_n_eta_sigma.txt"),
               np.array([theta_best]))

    with open(os.path.join(save_dir, "best_loss_history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Iteration", "Loss"])
        w.writerows(enumerate(hist))

    with open(os.path.join(save_dir, "iteration_times.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Iteration", "Duration_Seconds"])
        for i, t in enumerate(iter_times):
            w.writerow([i + 1, f"{t:.4f}"])

    # ── Recommend Setup 4 ─────────────────────────────────────────────────────
    try:
        m_breve = Param(theta_best[1], theta_best[0], theta_best[2])
        setups  = [Setup(c["H"], c["W"], 1.0) for c in cfgs]
        new_setups = Mechanism().searchNewSetup_orthognality_for_forth_setup(m_breve, setups)
        if new_setups and len(new_setups) > 3:
            s4 = new_setups[3]
            print(f"Recommended Setup 4: W={s4.W:.3f}  H={s4.H:.3f}")
            np.savetxt(os.path.join(save_dir, "setup3_recommended_setup4_WH.txt"),
                       np.array([[s4.W, s4.H]]))

            # Always emit a settings.xml template inside the result dir.
            settings_template = os.path.join(save_dir, "setup4_settings.xml")
            write_settings_xml(settings_template, s4.W, s4.H)
            print(f"  Setup 4 settings.xml template: {settings_template}")

            # Optionally also create the actual data directory for Setup 4.
            if args.gen_setup4_data_dir:
                target_dir = args.gen_setup4_data_dir
            else:
                target_dir = os.path.join(
                    "..", "data",
                    f"ref_{args.material_name}_{s4.H:.1f}_{s4.W:.1f}_4"
                )

            os.makedirs(target_dir, exist_ok=True)
            write_settings_xml(os.path.join(target_dir, "settings.xml"), s4.W, s4.H)
            print(f"  Setup 4 data directory created: {target_dir}/")
            print(f"    -> contains settings.xml (W={s4.W:.3f}, H={s4.H:.3f})")
            print(f"    -> next: run Simulation/main.py with --ref {target_dir} to generate frames")
    except Exception as e:
        print(f"[Error] Mechanism search failed: {e}")

    elapsed = time.time() - wall_start
    print(f"\n[Finished] Wall clock: {elapsed:.2f}s")
    with open(os.path.join(save_dir, "wall_clock_time.txt"), "w") as f:
        f.write(f"{elapsed:.4f}")


if __name__ == "__main__":
    main()
