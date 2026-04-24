"""Inverse recovery of (n, eta, sigma_y) for HVIMoGP_rBCM via CMA-ES.

Minimal test harness: sample a handful of rows from test_merged.csv, run
CMA-ES on each (using the HVIMoGPrBCMPredictor as a black-box forward
model), compare recovered vs ground-truth parameters.

Usage
-----
    python -m vi_mogp.invert_rbcm \\
        --model Models/rbcm_v1_hotfix/model.pt \\
        --test  Optimization/moe_workspace_merged_v3_20260419/test_merged.csv \\
        --n-samples 10 --popsize 12 --maxiter 60 --seed 0

Objective
---------
    loss(n, eta, sigma_y) = NMSE(y_pred, y_obs)
                          = mean((y_pred - y_obs)^2) / mean(y_obs^2)

Routing: uses y_obs as conditioning signal for the phi-GMM level-2 gate;
forward prediction feeds the CMA candidate (n, eta, sigma_y) along with
the fixed (W, H) as the 5-D input X_raw.
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from vi_mogp.predict import HVIMoGPrBCMPredictor
from Optimization.libs.cmaes_core import run_cmaes, PARAM_BOUNDS


INPUT_COLS  = ["n", "eta", "sigma_y", "width", "height"]
OUTPUT_COLS = ["x_01", "x_02", "x_03", "x_04",
               "x_05", "x_06", "x_07", "x_08"]


def make_loss_fn(predictor: HVIMoGPrBCMPredictor,
                 y_obs: np.ndarray, W: float, H: float,
                 top_k_phi: int | None = None,
                 eids: np.ndarray | None = None,
                 geo_id: int | None = None,
                 use_baseline: bool = True,
                 sigma_y_anchor: float | None = None,
                 sigma_prior_weight: float = 0.0,
                 sigma_prior_width: float = 1.0,
                 wts: np.ndarray | None = None) -> Callable:
    """Return a batch-loss callable compatible with run_cmaes.

    Each call receives a list of (n, eta, sigma_y) candidates (physical
    units) and returns a list of NMSE losses vs y_obs. We call the
    predictor ONCE per batch (GPU-efficient under our fixed predict loop).

    Optimisation A: if (eids, geo_id) are provided (pre-routed once per
    sample), use the fixed-route predict bypass that skips the gate's
    build_phi + softmax on every iter.

    use_baseline : bool
        If True (default), GRBCM aggregation with geo-baseline tie-break.
        If False, rBCM aggregation using prior variance as reference —
        removes the baseline-blend bias that hurts accuracy when the
        baseline's training data is far from the true expert's. For
        top-1 routing, rBCM ≡ single-expert mean ≈ MoE-240.

    sigma_y_anchor / sigma_prior_weight / sigma_prior_width :
        Optional soft log-space prior on σ_y pulling it toward
        `sigma_y_anchor` (typically the NN-warm-start σ_y — a data-
        honest estimate). The penalty added is::

            λ · ((log σ_y − log σ_y_anchor) / w) ²

        where λ = sigma_prior_weight (weight in NMSE-units of the
        anchor-deviation penalty), w = sigma_prior_width (log-std,
        e.g. w=1 means ±e = ±2.7×). Set λ=0 (default) to disable.

        Fixes the Herschel-Bulkley σ_y → 0 identifiability ridge: when
        the flow curve is insensitive to σ_y, CMA slides down the ridge
        to small σ_y and compensates with large η. The prior anchors
        σ_y near the training data's closest match, which is data-
        honest (it cannot be outside the training distribution).
    """
    y_obs = np.asarray(y_obs, dtype=np.float64)
    norm = max(float(np.mean(y_obs ** 2)), 1e-12)
    use_fixed_route = (eids is not None and geo_id is not None)
    use_prior = (sigma_prior_weight > 0.0 and sigma_y_anchor is not None
                 and sigma_y_anchor > 0.0)
    if use_prior:
        log_anchor = float(np.log(sigma_y_anchor))

    def batch_loss(thetas):
        thetas = np.asarray(thetas, dtype=np.float64)   # (B, 3)
        B = thetas.shape[0]
        X = np.zeros((B, 5), dtype=np.float64)
        X[:, 0:3] = thetas
        X[:, 3]   = W
        X[:, 4]   = H
        if use_fixed_route:
            # Fast path — skip phi-build + gate-softmax per iter.
            y_pred = predictor.predict_fixed_route(
                X, eids, geo_id, clear_cache=False,
                use_baseline=use_baseline,
                phi_weights_row=wts,
            )
        else:
            Y_tile = np.broadcast_to(y_obs, (B, 8)).copy()
            kwargs = dict(W=W, H=H, y_obs=Y_tile, clear_cache=False,
                          use_baseline=use_baseline)
            if top_k_phi is not None:
                kwargs["top_k_phi"] = top_k_phi
            y_pred = predictor.predict(X, **kwargs)
        err = np.mean((y_pred - y_obs[None, :]) ** 2, axis=1) / norm
        if use_prior:
            sy = np.maximum(thetas[:, 2], 1e-6)
            pen = sigma_prior_weight * ((np.log(sy) - log_anchor)
                                        / sigma_prior_width) ** 2
            err = err + pen
        return err.tolist()

    return batch_loss


def invert_row(predictor, y_obs, W, H,
               popsize=12, maxiter=60, sigma0=0.5, seed=0, verb=0,
               tighten_bounds: bool = True, pad_frac: float = 0.02,
               top_k_phi: int | None = None,
               warm_start: bool = False,
               use_baseline: bool = True,
               sigma_prior_weight: float = 0.0,
               sigma_prior_width: float = 1.0,
               use_phi_weights: bool = True):
    # Change 2: hard-union of routed-expert parameter boxes → tight CMA
    # bounds. Prevents CMA from wandering into unsupported eta≈0 / σy≈0
    # regions of the (n, eta, sigma_y) cube where the GP extrapolation
    # can return spurious zero-flow minima.
    # Optimisation A: route ONCE per row (fixed (W, H, y_obs) in a row),
    # then feed (eids, geo_id) into loss_fn so every CMA iter skips the
    # gate's phi-build + softmax.
    from vi_mogp import config as HC
    top_k = top_k_phi if top_k_phi is not None else HC.INFER_TOP_K_PHI
    geo_id, eids, wts_full = predictor.route_for(W, H, y_obs, top_k_phi=top_k)
    # Step-A: ablation switch — drop BGM posterior weighting if requested.
    wts = wts_full if use_phi_weights else None
    if tighten_bounds:
        bounds = predictor.tightened_bounds(eids, geo_id, PARAM_BOUNDS,
                                            include_baseline=True,
                                            pad_frac=pad_frac)
    else:
        bounds = PARAM_BOUNDS

    # Speed-path C: warm-start CMA from the routed expert's training
    # nearest-neighbor (output-space L2 distance). Lets us cut maxiter
    # aggressively and shrink sigma0 since we start near a data-honest
    # guess. Falls back to None (default midpoint) if NN lookup fails.
    x0 = None
    sigma_y_anchor = None
    if warm_start:
        nn = predictor.nearest_neighbor_theta(int(eids[0]), y_obs)
        if nn is not None:
            # Clip to tightened bounds so CMA's log-space transform is
            # well-defined (nearest-neighbor is guaranteed to be inside
            # the un-padded expert box, so clipping is a no-op unless
            # pad_frac < 0).
            n0 = float(np.clip(nn[0], bounds["n"][0],       bounds["n"][1]))
            e0 = float(np.clip(nn[1], bounds["eta"][0],     bounds["eta"][1]))
            s0 = float(np.clip(nn[2], bounds["sigma_y"][0], bounds["sigma_y"][1]))
            x0 = [n0, e0, s0]
            # Reuse NN σy as the soft-prior anchor (only used if
            # sigma_prior_weight > 0; otherwise the prior is off).
            sigma_y_anchor = s0

    loss_fn = make_loss_fn(predictor, y_obs, W, H,
                           top_k_phi=top_k_phi,
                           eids=eids, geo_id=geo_id,
                           use_baseline=use_baseline,
                           sigma_y_anchor=sigma_y_anchor,
                           sigma_prior_weight=sigma_prior_weight,
                           sigma_prior_width=sigma_prior_width,
                           wts=wts)

    t0 = time.time()
    theta_best, loss_best, hist = run_cmaes(
        loss_fn, bounds,
        x0=x0,
        sigma0=sigma0, popsize=popsize, maxiter=maxiter,
        seed=seed, verb_disp=verb,
    )
    dt = time.time() - t0
    # Also get the best candidate's y_pred for forward-error readout
    # (keep clear_cache=False for consistency with the loop above; we'll
    # free all caches below once the row is done).
    y_pred = predictor.predict_fixed_route(
        np.array([[*theta_best, W, H]], dtype=np.float64),
        eids, geo_id, clear_cache=False,
        use_baseline=use_baseline,
        phi_weights_row=wts,
    )[0]
    # Now that this row's inversion is done, flush every expert +
    # baseline prediction cache so GPU / CPU memory doesn't grow across
    # samples. This is cheap (~few ms) relative to the inversion itself.
    for exp in predictor.model.experts:
        exp.clear_prediction_cache()
    for b in predictor.model.baselines:
        b.clear_prediction_cache()
    return theta_best, loss_best, dt, y_pred, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test",  required=True)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--popsize",   type=int, default=12)
    ap.add_argument("--maxiter",   type=int, default=60)
    ap.add_argument("--sigma0",    type=float, default=0.5)
    ap.add_argument("--seed",      type=int, default=0)
    ap.add_argument("--out",       type=str, default=None,
                    help="Optional CSV path for per-sample results")
    ap.add_argument("--verb",      type=int, default=0)
    ap.add_argument("--no-tighten-bounds", action="store_true",
                    help="Disable per-expert box CMA bound tightening "
                         "(change 2). On by default.")
    ap.add_argument("--pad-frac",  type=float, default=0.02,
                    help="Relative pad applied to tightened bounds "
                         "(log-scale for eta/sigma_y). Default 0.02.")
    ap.add_argument("--top-k-phi", type=int, default=None,
                    help="Override INFER_TOP_K_PHI for both routing and "
                         "predict. Default None = use config value.")
    ap.add_argument("--warm-start", action="store_true",
                    help="Speed-path C: warm-start CMA x0 from the routed "
                         "expert's training nearest-neighbor (output-space "
                         "L2). Pair with smaller sigma0/maxiter for speed.")
    ap.add_argument("--no-baseline", action="store_true",
                    help="Use rBCM aggregation (no geo-baseline) instead of "
                         "GRBCM. Removes baseline-blend bias — at top-1 "
                         "routing this reduces to single-expert mean, "
                         "architecturally equivalent to MoE-240.")
    ap.add_argument("--sigma-prior-weight", type=float, default=0.0,
                    help="Soft log-space prior on σy anchored at NN-warm-"
                         "start value. 0 = off. Try 1e-5..1e-3 (same "
                         "magnitude as the NMSE loss).")
    ap.add_argument("--sigma-prior-width", type=float, default=1.0,
                    help="Log-std of the σy prior (default 1.0 = ±e). "
                         "Larger = weaker pull.")
    args = ap.parse_args()

    print(f"[invert] model: {args.model}")
    print(f"[invert] test:  {args.test}")
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

        theta_best, loss_best, dt, y_pred, _ = invert_row(
            predictor, y_obs, W, H,
            popsize=args.popsize, maxiter=args.maxiter,
            sigma0=args.sigma0, seed=args.seed, verb=args.verb,
            tighten_bounds=(not args.no_tighten_bounds),
            pad_frac=args.pad_frac,
            top_k_phi=args.top_k_phi,
            warm_start=args.warm_start,
            use_baseline=(not args.no_baseline),
            sigma_prior_weight=args.sigma_prior_weight,
            sigma_prior_width=args.sigma_prior_width,
        )
        n_h, eta_h, sig_h = theta_best
        forward_nmse = float(np.mean((y_pred - y_obs) ** 2)
                             / max(float(np.mean(y_obs ** 2)), 1e-12))
        rel_n   = abs(n_h   - n_t)   / max(abs(n_t),   1e-9)
        rel_eta = abs(eta_h - eta_t) / max(abs(eta_t), 1e-9)
        rel_sig = abs(sig_h - sig_t) / max(abs(sig_t), 1e-9)

        print(f"[{i+1:2d}/{args.n_samples}]  "
              f"true  n={n_t:.3f}  η={eta_t:9.3f}  σy={sig_t:9.3f}  "
              f"(W={W},H={H})")
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
        ))
        total_t += dt

    df_r = pd.DataFrame(results)
    print("\n── summary over {} samples ──".format(len(df_r)))
    for k in ("rel_err_n", "rel_err_eta", "rel_err_sigma",
              "cmaes_loss", "forward_nmse", "dt_s"):
        s = df_r[k]
        print(f"  {k:16s}  mean={s.mean():.3e}  med={s.median():.3e}  "
              f"max={s.max():.3e}")
    print(f"  total cma time: {total_t:.1f}s  "
          f"(avg {total_t/len(df_r):.1f}s/sample)")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_r.to_csv(out_path, index=False)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
