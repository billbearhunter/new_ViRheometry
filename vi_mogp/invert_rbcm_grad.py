"""Change 4 — gradient-based (Adam + multi-start) inverse recovery for
HVIMoGP_rBCM / GRBCM.

Why
---
CMA-ES is derivative-free and re-evaluates the forward model O(popsize ×
maxiter) ≈ 720 times per row. Our GRBCM forward is differentiable all the
way from (n, eta, sigma_y) through the log-scaler, the fixed-routed GP
experts (ExactExpert now has grad-flow enabled), and the precision-
weighted aggregation. Once we **fix the route** based on y_obs + (W, H),
re-evaluating for different θ keeps the same expert set and re-uses the
Cholesky cache exactly the same way CMA-ES does — so Adam gets the same
per-iter cost but converges in far fewer iters.

Multi-start (16 parallel) mitigates the HB identifiability surface's
(eta, sigma_y) plateau: log-likelihood is flat when eta·γ̇ⁿ ≫ σy, so
different inits land in different parts of the ridge. We return the
best.

Usage
-----
    python -m vi_mogp.invert_rbcm_grad \\
        --model Models/rbcm_v1_hotfix/model.pt \\
        --test  Optimization/moe_workspace_merged_v3_20260419/test_merged.csv \\
        --n-samples 10 --n-starts 16 --n-steps 200 --lr 0.05 --seed 0

Routing is computed ONCE per row via `predictor.route_for(W, H, y_obs)`
and the expert list is frozen for the entire optimisation. If change-2
is enabled (`--tighten-bounds`, default on) the multi-start seeds are
drawn inside the hard-union parameter box of the routed experts.
"""
from __future__ import annotations
import argparse, math, time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from vi_mogp import config as HC
from vi_mogp.predict import HVIMoGPrBCMPredictor
from vi_mogp.config import LOG_INPUTS, INPUT_COLS, LOG_EPS
from Optimization.libs.cmaes_core import PARAM_BOUNDS


OUTPUT_COLS = ["x_01", "x_02", "x_03", "x_04",
               "x_05", "x_06", "x_07", "x_08"]


# ════════════════════════════════════════════════════════════════════════════
# Torch-side scalers (mirror vi_mogp/data.py::InputScaler.transform with
# autograd flow, and OutputScaler.transform / .inverse_transform).
# ════════════════════════════════════════════════════════════════════════════

def _build_log_mask(device, dtype) -> torch.Tensor:
    """Return (5,) bool mask selecting columns in LOG_INPUTS."""
    m = [(c in LOG_INPUTS) for c in INPUT_COLS]
    return torch.tensor(m, dtype=torch.bool, device=device)


def torch_input_transform(X_phys: torch.Tensor,
                          mean: torch.Tensor,
                          std: torch.Tensor,
                          log_mask: torch.Tensor,
                          log_eps: float = LOG_EPS) -> torch.Tensor:
    """Log-then-standardise, differentiable. (B, 5) → (B, 5)."""
    X = torch.where(
        log_mask.unsqueeze(0),
        torch.log(X_phys.clamp_min(log_eps)),
        X_phys,
    )
    return (X - mean) / std


# ════════════════════════════════════════════════════════════════════════════
# Parameter reparametrisation: free z ∈ R³ → bounded θ = (n, eta, σy) via
# sigmoid. n linear, eta/σy log-linear. Keeps Adam fully unconstrained.
# ════════════════════════════════════════════════════════════════════════════

def _repar_to_theta(z: torch.Tensor,
                    bounds: dict,
                    device, dtype) -> torch.Tensor:
    """(M, 3) unconstrained → (M, 3) physical θ in `bounds`."""
    s = torch.sigmoid(z)
    n_lo, n_hi   = bounds["n"]
    e_lo, e_hi   = bounds["eta"]
    sy_lo, sy_hi = bounds["sigma_y"]

    e_lo  = max(e_lo,  1e-9)
    sy_lo = max(sy_lo, 1e-9)

    n   = n_lo + (n_hi - n_lo) * s[:, 0]
    eta = torch.exp(math.log(e_lo)  + (math.log(e_hi)  - math.log(e_lo))  * s[:, 1])
    sy  = torch.exp(math.log(sy_lo) + (math.log(sy_hi) - math.log(sy_lo)) * s[:, 2])
    return torch.stack([n, eta, sy], dim=-1)


def _theta_to_repar(theta: np.ndarray, bounds: dict) -> np.ndarray:
    """Inverse map: physical θ → unconstrained z (for seeded starts)."""
    n_lo, n_hi   = bounds["n"]
    e_lo, e_hi   = bounds["eta"]
    sy_lo, sy_hi = bounds["sigma_y"]
    e_lo  = max(e_lo,  1e-9)
    sy_lo = max(sy_lo, 1e-9)

    n   = np.clip(theta[:, 0], n_lo + 1e-6, n_hi - 1e-6)
    eta = np.clip(theta[:, 1], e_lo * 1.001, e_hi * 0.999)
    sy  = np.clip(theta[:, 2], sy_lo * 1.001, sy_hi * 0.999)

    s_n   = (n - n_lo) / (n_hi - n_lo)
    s_e   = (np.log(eta) - math.log(e_lo))  / (math.log(e_hi)  - math.log(e_lo))
    s_s   = (np.log(sy)  - math.log(sy_lo)) / (math.log(sy_hi) - math.log(sy_lo))
    s = np.clip(np.stack([s_n, s_e, s_s], axis=-1), 1e-6, 1 - 1e-6)
    return np.log(s / (1.0 - s))   # inverse sigmoid


# ════════════════════════════════════════════════════════════════════════════
# Multi-start seeding inside a box (LHS-lite: log-uniform random)
# ════════════════════════════════════════════════════════════════════════════

def _sample_starts(n_starts: int, bounds: dict,
                   rng: np.random.Generator) -> np.ndarray:
    n_lo, n_hi   = bounds["n"]
    e_lo, e_hi   = bounds["eta"]
    sy_lo, sy_hi = bounds["sigma_y"]
    e_lo  = max(e_lo,  1e-9)
    sy_lo = max(sy_lo, 1e-9)

    ns   = rng.uniform(n_lo, n_hi, n_starts)
    etas = np.exp(rng.uniform(math.log(e_lo),  math.log(e_hi),  n_starts))
    sys_ = np.exp(rng.uniform(math.log(sy_lo), math.log(sy_hi), n_starts))
    return np.stack([ns, etas, sys_], axis=-1)     # (M, 3)


# ════════════════════════════════════════════════════════════════════════════
# The gradient inverse itself
# ════════════════════════════════════════════════════════════════════════════

def invert_row_grad(predictor: HVIMoGPrBCMPredictor,
                    y_obs: np.ndarray,
                    W: float, H: float,
                    n_starts: int = 16,
                    n_steps: int = 200,
                    lr: float = 0.05,
                    seed: int = 0,
                    tighten_bounds: bool = True,
                    pad_frac: float = 0.02,
                    top_k_phi: int | None = None,
                    verbose: bool = False,
                    ) -> Tuple[np.ndarray, float, float, np.ndarray, dict]:
    """Single-row gradient inverse with multi-start Adam.

    Returns (theta_best, loss_best, dt_s, y_pred_best, stats_dict).
    """
    device = predictor.device
    dtype  = predictor.dtype
    model  = predictor.model
    xs, ys = predictor.xs, predictor.ys

    # ── 1. Route once on (W, H, y_obs) ────────────────────────────────────
    top_k = top_k_phi if top_k_phi is not None else HC.INFER_TOP_K_PHI
    geo_id, eids_1d, wts_1d = predictor.route_for(W, H, y_obs, top_k_phi=top_k)
    # GRBCM uses (B, K) expert_ids; replicate the same route across the M
    # multi-starts so every candidate sees the same experts + baseline.
    expert_ids_row = eids_1d.reshape(1, -1)                        # (1, K)
    geo_ids_row    = np.array([geo_id], dtype=np.int64)            # (1,)

    # ── 2. Tightened bounds if enabled ─────────────────────────────────────
    if tighten_bounds:
        bounds = predictor.tightened_bounds(eids_1d, geo_id, PARAM_BOUNDS,
                                            include_baseline=True,
                                            pad_frac=pad_frac)
    else:
        bounds = PARAM_BOUNDS

    # ── 3. Precompute torch-side scalers ──────────────────────────────────
    xs_mean = torch.as_tensor(xs.mean, dtype=dtype, device=device)  # (5,)
    xs_std  = torch.as_tensor(xs.std,  dtype=dtype, device=device)  # (5,)
    log_mask = _build_log_mask(device, dtype)
    ys_mean = torch.as_tensor(ys.mean, dtype=dtype, device=device)  # (8,)
    y_obs_t = torch.as_tensor(y_obs, dtype=dtype, device=device)
    # NMSE denominator matches CMA-ES's NMSE (physical y_obs norm).
    norm = float(max((y_obs ** 2).mean(), 1e-12))

    W_f, H_f = float(W), float(H)

    # ── 4. Seed M multi-starts + reparametrise to free z ──────────────────
    rng = np.random.default_rng(seed)
    M = int(n_starts)
    theta0 = _sample_starts(M, bounds, rng)                        # (M, 3)
    z_init = _theta_to_repar(theta0, bounds)                       # (M, 3)
    z = torch.as_tensor(z_init, dtype=dtype, device=device).clone()
    z.requires_grad_(True)

    opt = torch.optim.Adam([z], lr=lr)

    # ── 5. Pre-build (M, K) expert_ids / (M,) geo_ids for batched predict
    expert_ids_MB = np.tile(expert_ids_row, (M, 1))                # (M, K)
    geo_ids_MB    = np.tile(geo_ids_row,    M)                     # (M,)
    # Step-A: BGM posterior weights replicated to (M, K) for rBCM weighting
    phi_weights_MB = np.tile(np.asarray(wts_1d, dtype=np.float64).reshape(1, -1),
                             (M, 1))                                # (M, K)

    # ── 6. Inner loop: Adam steps (cache reused since route is fixed) ─────
    t0 = time.time()
    best_loss = float("inf")
    best_theta = theta0[0].copy()
    best_y_pred = np.full(8, np.nan)
    loss_hist = []

    for step in range(n_steps):
        opt.zero_grad()
        theta = _repar_to_theta(z, bounds, device, dtype)          # (M, 3)

        # Build (M, 5) physical X = [n, eta, sy, W, H], then torch-transform
        W_col = torch.full((M, 1), W_f, dtype=dtype, device=device)
        H_col = torch.full((M, 1), H_f, dtype=dtype, device=device)
        X_phys = torch.cat([theta, W_col, H_col], dim=-1)          # (M, 5)
        X_scaled = torch_input_transform(X_phys, xs_mean, xs_std, log_mask)

        # GRBCM forward — keep cache across Adam steps (clear_cache=False).
        # This is identical to CMA-ES's reuse: the routed expert set is
        # frozen, so the (N_train × N_train) Cholesky stays valid.
        mu_c, _var = model.predict_grbcm(
            X_scaled, expert_ids_MB, geo_ids_MB,
            clear_cache=False,
            phi_weights=phi_weights_MB,
        )                                                          # (M, 8)
        # mu_c is in scaled (centered) output space. y_obs in physical.
        # Centered-space numerator equals physical-space numerator because
        # both differ by the same per-column mean shift; use physical
        # denominator (norm) for NMSE consistency with CMA.
        y_pred_phys = mu_c + ys_mean                               # (M, 8)
        # Change 1: monotonicity along the 8 flow-distance columns,
        # applied in PHYSICAL space (the centered space is not monotonic
        # for slow-flow samples). cummax is a sub-differentiable op;
        # autograd picks a valid subgradient at ties.
        y_pred_phys = torch.cummax(y_pred_phys, dim=-1).values
        err2 = (y_pred_phys - y_obs_t.unsqueeze(0)) ** 2           # (M, 8)
        losses = err2.mean(dim=-1) / norm                          # (M,)
        total = losses.sum()                                       # scalar

        # Track best so far
        with torch.no_grad():
            losses_np  = losses.detach().cpu().numpy()
            theta_np   = theta.detach().cpu().numpy()
            y_pred_np  = y_pred_phys.detach().cpu().numpy()
        argm = int(np.argmin(losses_np))
        if losses_np[argm] < best_loss:
            best_loss   = float(losses_np[argm])
            best_theta  = theta_np[argm].copy()
            best_y_pred = y_pred_np[argm].copy()
        loss_hist.append(float(losses_np.min()))

        total.backward()
        opt.step()

        if verbose and (step % 25 == 0 or step == n_steps - 1):
            print(f"      step {step:3d}  best-so-far={best_loss:.3e}  "
                  f"median(batch)={np.median(losses_np):.3e}")

    dt = time.time() - t0

    # Free caches after this row — next row has a different route anyway.
    for exp in model.experts:
        exp.clear_prediction_cache()
    for b in model.baselines:
        b.clear_prediction_cache()

    stats = dict(
        n_starts=M, n_steps=n_steps, lr=lr,
        route_geo=geo_id, route_experts=eids_1d.tolist(),
        bounds_tightened=tighten_bounds,
        bounds_used=bounds,
        loss_hist=loss_hist,
    )
    return best_theta, best_loss, dt, best_y_pred, stats


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test",  required=True)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--n-starts",  type=int, default=16)
    ap.add_argument("--n-steps",   type=int, default=200)
    ap.add_argument("--lr",        type=float, default=0.05)
    ap.add_argument("--seed",      type=int, default=0)
    ap.add_argument("--no-tighten-bounds", action="store_true")
    ap.add_argument("--pad-frac",  type=float, default=0.02)
    ap.add_argument("--out",       type=str, default=None)
    ap.add_argument("--verbose",   action="store_true")
    args = ap.parse_args()

    print(f"[invert-grad] model: {args.model}")
    print(f"[invert-grad] test:  {args.test}")
    predictor = HVIMoGPrBCMPredictor.load(args.model)

    df = pd.read_csv(args.test)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(df), size=args.n_samples, replace=False)
    rows = df.iloc[idx].reset_index(drop=True)

    results = []
    total_t = 0.0
    for i, r in rows.iterrows():
        n_t, eta_t, sig_t = float(r["n"]), float(r["eta"]), float(r["sigma_y"])
        W, H = float(r["width"]), float(r["height"])
        y_obs = np.array([r[c] for c in OUTPUT_COLS], dtype=np.float64)

        theta_best, loss_best, dt, y_pred, stats = invert_row_grad(
            predictor, y_obs, W, H,
            n_starts=args.n_starts, n_steps=args.n_steps, lr=args.lr,
            seed=args.seed + int(idx[i]),  # per-sample seed
            tighten_bounds=(not args.no_tighten_bounds),
            pad_frac=args.pad_frac,
            verbose=args.verbose,
        )
        n_h, eta_h, sig_h = theta_best
        forward_nmse = float(np.mean((y_pred - y_obs) ** 2)
                             / max(float(np.mean(y_obs ** 2)), 1e-12))
        rel_n   = abs(n_h   - n_t)   / max(abs(n_t),   1e-9)
        rel_eta = abs(eta_h - eta_t) / max(abs(eta_t), 1e-9)
        rel_sig = abs(sig_h - sig_t) / max(abs(sig_t), 1e-9)

        print(f"[{i+1:2d}/{args.n_samples}]  "
              f"true  n={n_t:.3f}  η={eta_t:9.3f}  σy={sig_t:9.3f}  "
              f"(W={W},H={H})  experts={len(stats['route_experts'])}")
        print(f"       rec   n={n_h:.3f}  η={eta_h:9.3f}  σy={sig_h:9.3f}  "
              f"loss={loss_best:.3e}  fwd-NMSE={forward_nmse:.3e}  dt={dt:.1f}s")
        print(f"       relE: n={rel_n:.3f}  η={rel_eta:.3f}  σy={rel_sig:.3f}")

        results.append(dict(
            sample=int(idx[i]),
            n_true=n_t, eta_true=eta_t, sigma_true=sig_t, W=W, H=H,
            n_hat=n_h, eta_hat=eta_h, sigma_hat=sig_h,
            rel_err_n=rel_n, rel_err_eta=rel_eta, rel_err_sigma=rel_sig,
            cmaes_loss=loss_best, forward_nmse=forward_nmse,
            dt_s=dt,
            n_experts=len(stats["route_experts"]),
            bounds_tightened=bool(stats["bounds_tightened"]),
        ))
        total_t += dt

    df_r = pd.DataFrame(results)
    print("\n── summary over {} samples ──".format(len(df_r)))
    for k in ("rel_err_n", "rel_err_eta", "rel_err_sigma",
              "cmaes_loss", "forward_nmse", "dt_s"):
        s = df_r[k]
        print(f"  {k:16s}  mean={s.mean():.3e}  med={s.median():.3e}  "
              f"max={s.max():.3e}")
    print(f"  total grad time: {total_t:.1f}s  "
          f"(avg {total_t/len(df_r):.1f}s/sample)")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_r.to_csv(out_path, index=False)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
