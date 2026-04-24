"""Step A — inverse A/B on v3 test set: weighted vs unweighted rBCM.

For N rows of test_merged.csv (each with known true theta, W, H, y_obs):
  1. Run CMA-ES inverse with use_phi_weights=True   (Step-A fix)
  2. Run CMA-ES inverse with use_phi_weights=False  (legacy bug)
  3. Compare:
     - recovered theta vs true theta (per-axis relative error)
     - forward y(theta_hat) vs y_obs (NMSE)

Output: tests/step_a_results/inverse_AB_<n>x.csv with one row per test
sample, weighted/unweighted side by side.

Same seed, same starting point, same bounds — only difference is whether
beta_k is multiplied by the BGM posterior p(k|phi,g).
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vi_mogp import config as HC
from vi_mogp.predict import HVIMoGPrBCMPredictor
from vi_mogp.invert_rbcm import invert_row, OUTPUT_COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--popsize",   type=int, default=12)
    ap.add_argument("--maxiter",   type=int, default=80)
    ap.add_argument("--sigma0",    type=float, default=0.5)
    ap.add_argument("--top-k-phi", type=int, default=4)
    ap.add_argument("--seed",      type=int, default=0)
    ap.add_argument("--warm-start", action="store_true",
                    help="NN warm-start (matches production CMA setup)")
    args = ap.parse_args()

    print(f"[load] {HC.DEFAULT_V2_MODEL}")
    predictor = HVIMoGPrBCMPredictor.load(HC.DEFAULT_V2_MODEL)

    df = pd.read_csv(HC.MERGED_TEST_CSV)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(df), size=args.n_samples, replace=False)
    rows = df.iloc[idx].reset_index(drop=True)
    print(f"[info] sampled {len(rows)} test rows from {HC.MERGED_TEST_CSV}")

    out_rows = []
    t_global = time.time()
    for i, r in rows.iterrows():
        n_t   = float(r["n"])
        eta_t = float(r["eta"])
        sig_t = float(r["sigma_y"])
        W = float(r["width"]); H = float(r["height"])
        y_obs = np.array([r[c] for c in OUTPUT_COLS], dtype=np.float64)
        norm = max(float(np.mean(y_obs ** 2)), 1e-12)

        print(f"\n--- row {i+1}/{len(rows)}  W={W} H={H}  "
              f"true: n={n_t:.4f} eta={eta_t:.3f} sy={sig_t:.3f} ---")

        # === A: weighted (Step-A fix) ===
        t0 = time.time()
        theta_W, loss_W, dt_W, ypred_W, _ = invert_row(
            predictor, y_obs, W, H,
            popsize=args.popsize, maxiter=args.maxiter,
            sigma0=args.sigma0, seed=args.seed, verb=0,
            tighten_bounds=True, pad_frac=0.02,
            top_k_phi=args.top_k_phi,
            warm_start=args.warm_start,
            use_baseline=True,
            use_phi_weights=True,
        )
        nW, eW, sW = theta_W
        fwd_W = float(np.mean((ypred_W - y_obs) ** 2) / norm)

        # === B: unweighted (legacy bug) ===
        theta_NW, loss_NW, dt_NW, ypred_NW, _ = invert_row(
            predictor, y_obs, W, H,
            popsize=args.popsize, maxiter=args.maxiter,
            sigma0=args.sigma0, seed=args.seed, verb=0,
            tighten_bounds=True, pad_frac=0.02,
            top_k_phi=args.top_k_phi,
            warm_start=args.warm_start,
            use_baseline=True,
            use_phi_weights=False,
        )
        nNW, eNW, sNW = theta_NW
        fwd_NW = float(np.mean((ypred_NW - y_obs) ** 2) / norm)

        # Per-axis relative error (use eta, sy in log space — typical scale span ~3 decades)
        def rel(a, t): return abs(a - t) / max(abs(t), 1e-9)
        def lrel(a, t):
            return abs(np.log(max(a, 1e-9)) - np.log(max(t, 1e-9))) \
                 / max(abs(np.log(max(t, 1e-9))), 1e-9)

        out_rows.append(dict(
            row=i, W=W, H=H,
            n_t=n_t, eta_t=eta_t, sig_t=sig_t,
            # weighted
            n_W=nW, eta_W=eW, sig_W=sW, fwd_nmse_W=fwd_W,
            cma_loss_W=loss_W, dt_W=dt_W,
            rel_n_W=rel(nW, n_t),
            rel_eta_W=rel(eW, eta_t),
            rel_sig_W=rel(sW, sig_t),
            lrel_eta_W=lrel(eW, eta_t),
            lrel_sig_W=lrel(sW, sig_t),
            # unweighted
            n_NW=nNW, eta_NW=eNW, sig_NW=sNW, fwd_nmse_NW=fwd_NW,
            cma_loss_NW=loss_NW, dt_NW=dt_NW,
            rel_n_NW=rel(nNW, n_t),
            rel_eta_NW=rel(eNW, eta_t),
            rel_sig_NW=rel(sNW, sig_t),
            lrel_eta_NW=lrel(eNW, eta_t),
            lrel_sig_NW=lrel(sNW, sig_t),
        ))
        print(f"  W:  n={nW:.4f}  eta={eW:.3f}  sy={sW:.3f}  "
              f"fwd_NMSE={fwd_W:.3e}  ({dt_W:.1f}s)")
        print(f"  NW: n={nNW:.4f}  eta={eNW:.3f}  sy={sNW:.3f}  "
              f"fwd_NMSE={fwd_NW:.3e}  ({dt_NW:.1f}s)")
        print(f"  delta fwd_NMSE: {fwd_W-fwd_NW:+.3e}  "
              f"({(fwd_W-fwd_NW)/max(fwd_NW,1e-12)*100:+.1f}%)")

    df_out = pd.DataFrame(out_rows)
    out_dir = REPO / "tests" / "step_a_results"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"inverse_AB_{len(out_rows)}x.csv"
    df_out.to_csv(csv_path, index=False, float_format="%.6f")

    # === Aggregate summary ===
    def agg(col_W, col_NW, name):
        a = df_out[col_W].to_numpy()
        b = df_out[col_NW].to_numpy()
        better = float(np.mean(a < b))
        print(f"  {name:18s}  W mean={a.mean():.4f}  NW mean={b.mean():.4f}  "
              f"W median={np.median(a):.4f}  NW median={np.median(b):.4f}  "
              f"frac W better={better:.2f}")

    print(f"\n{'='*78}\n  AGGREGATE  (n={len(df_out)} rows)\n{'='*78}")
    agg("fwd_nmse_W", "fwd_nmse_NW", "forward NMSE")
    agg("rel_n_W",    "rel_n_NW",    "rel err n")
    agg("lrel_eta_W", "lrel_eta_NW", "log-rel err eta")
    agg("lrel_sig_W", "lrel_sig_NW", "log-rel err sigma_y")

    print(f"\n[saved] {csv_path}")
    print(f"[total time] {time.time()-t_global:.1f}s")


if __name__ == "__main__":
    main()
