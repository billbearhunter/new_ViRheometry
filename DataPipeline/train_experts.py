"""
DataPipeline/train_experts.py
=============================
Step 3 – Train one GP expert per GMM cluster.

For each cluster{N}_train.csv produced by prepare_data.py:
  - ≤ EXACT_THRESHOLD samples → ExactGP  (8 independent models)
  - >  EXACT_THRESHOLD samples → SVGP    (8 independent models)
  - Optional poly-residual boosting if GP MaxErr > MAXERR_TARGET on val set

The resulting expert_{N}.pt files are compatible with moe_core.py in the
Optimization/ folder (poly residual is now also applied at inference — see
the fix in moe_core.py if not yet applied).

Usage
-----
# Train all clusters in workspace:
python train_experts.py --data workspace/moe_ws

# Retrain specific clusters only (e.g. hard clusters 19, 42, 46):
python train_experts.py --data workspace/moe_ws --clusters 19 42 46 --force

# Skip clusters whose .pt already exists:
python train_experts.py --data workspace/moe_ws
"""

import argparse
import contextlib
import glob
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import gpytorch
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "DataPipeline"))

from dp_config import (
    INPUT_COLS, OUTPUT_COLS, LOG_INPUTS,
    EXACT_THRESHOLD, INDUCING_POINTS, BATCH_SIZE_SVGP,
    EPOCHS_EXACT, EPOCHS_SVGP, LR_EXACT, LR_SVGP,
    MAXERR_TARGET, POLY_ALPHA, POLY_DEGREES,
    CONF_THRESHOLD,
)
from moe_utils import (
    DEVICE, DTYPE,
    SingleOutputExactGP, SingleOutputSVGP,
    LogStandardInputScaler, TargetScaler,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── GP prediction helpers ──────────────────────────────────────────────────────

def _cg_settings():
    """Context manager for CG-based solves (memory-efficient ExactGP)."""
    return gpytorch.settings.max_cg_iterations(1000), \
           gpytorch.settings.cg_tolerance(0.01), \
           gpytorch.settings.max_preconditioner_size(100)


def gp_predict_scaled(models, likes, X_s: np.ndarray) -> np.ndarray:
    X_t = torch.tensor(X_s, dtype=DTYPE, device=DEVICE)
    preds = []
    cg1, cg2, cg3 = _cg_settings()
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
         gpytorch.settings.cholesky_jitter(1e-3), cg1, cg2, cg3:
        for m, lk in zip(models, likes):
            m.eval(); lk.eval()
            preds.append(lk(m(X_t)).mean.detach().cpu().numpy())
    return np.stack(preds, axis=1)   # (N, 8)


# ── ExactGP trainer ────────────────────────────────────────────────────────────

def train_exact_gp(X_t, Y_t, epochs: int):
    models, likes = [], []
    N = X_t.size(0)
    use_cg = N > 6000
    log.info(f"    [ExactGP] N={N}, epochs={epochs}, CG={use_cg}")
    cg1, cg2, cg3 = _cg_settings()
    for i in range(Y_t.shape[1]):
        y = Y_t[:, i].contiguous()
        lk = GaussianLikelihood(noise_constraint=GreaterThan(1e-4)).to(DEVICE, DTYPE)
        lk.noise = 0.02
        m  = SingleOutputExactGP(X_t, y, lk).to(DEVICE, DTYPE)
        m.train(); lk.train()
        opt = torch.optim.Adam(m.parameters(), lr=LR_EXACT)
        mll = ExactMarginalLogLikelihood(lk, m)
        # float32 needs larger jitter to keep Cholesky stable
        jitter = 1e-2 if DTYPE == torch.float32 else 1e-3
        for _ in range(epochs):
            opt.zero_grad()
            ctx = [gpytorch.settings.cholesky_jitter(jitter),
                   gpytorch.settings.fast_computations(
                       covar_root_decomposition=False,
                       log_prob=False, solves=False)]
            if use_cg:
                ctx.extend([cg1, cg2, cg3])
            with contextlib.ExitStack() as stack:
                for c in ctx:
                    stack.enter_context(c)
                loss = -mll(m(X_t), y)
            loss.backward(); opt.step()
        m.eval(); lk.eval()
        models.append(m); likes.append(lk)
    return models, likes


# ── SVGP trainer ───────────────────────────────────────────────────────────────

def train_svgp(X_t, Y_t, epochs: int):
    models, likes = [], []
    log.info(f"    [SVGP]    N={X_t.size(0)}, epochs={epochs}")
    n_ind = min(INDUCING_POINTS, X_t.size(0))
    km    = MiniBatchKMeans(n_clusters=n_ind, batch_size=4096, n_init=3)
    km.fit(X_t.cpu().numpy())
    ind_init = torch.tensor(km.cluster_centers_, dtype=DTYPE, device=DEVICE)

    for i in range(Y_t.shape[1]):
        y  = Y_t[:, i]
        m  = SingleOutputSVGP(ind_init.clone()).to(DEVICE, DTYPE)
        lk = GaussianLikelihood().to(DEVICE, DTYPE)
        m.train(); lk.train()
        opt = torch.optim.Adam(
            [{"params": m.parameters()}, {"params": lk.parameters()}], lr=LR_SVGP
        )
        mll = VariationalELBO(lk, m, num_data=y.size(0))
        loader = DataLoader(TensorDataset(X_t, y), batch_size=BATCH_SIZE_SVGP, shuffle=True)
        for _ in range(epochs):
            for bx, by in loader:
                opt.zero_grad()
                (-mll(m(bx), by)).backward()
                opt.step()
        m.eval(); lk.eval()
        models.append(m); likes.append(lk)
    return models, likes


# ── Poly residual booster ──────────────────────────────────────────────────────

def fit_poly_residual(models, likes, x_scaler, y_scaler,
                      X_tr_s, Y_tr_s, X_val_s, Y_val_s):
    """Fit weighted Ridge regression on GP residuals. Returns poly_state or None."""

    def _unscale(Y_s):
        return y_scaler.inverse_transform(Y_s)

    Y_gp_val_s  = gp_predict_scaled(models, likes, X_val_s)
    Y_val_true  = _unscale(Y_val_s)
    Y_gp_val    = _unscale(Y_gp_val_s)
    current_err = float(np.max(np.abs(Y_gp_val - Y_val_true)))

    if current_err <= MAXERR_TARGET:
        log.info(f"    [Poly] Skipped (MaxErr={current_err:.4f} ≤ {MAXERR_TARGET})")
        return None

    log.info(f"    [Poly] Fitting (GP MaxErr={current_err:.4f} > {MAXERR_TARGET})…")

    Y_gp_tr_s  = gp_predict_scaled(models, likes, X_tr_s)
    res_tr_s   = Y_tr_s - Y_gp_tr_s
    w          = np.ones(len(res_tr_s))
    thr        = np.quantile(np.linalg.norm(res_tr_s, axis=1), 0.90)
    w[np.linalg.norm(res_tr_s, axis=1) >= thr] = 5.0

    best = None
    for deg in POLY_DEGREES:
        pf   = PolynomialFeatures(degree=deg, include_bias=True)
        Xp_t = pf.fit_transform(X_tr_s)
        ridge = Ridge(alpha=POLY_ALPHA, fit_intercept=False)
        ridge.fit(Xp_t, res_tr_s, sample_weight=w)

        Xp_v   = pf.transform(X_val_s)
        Y_mix  = _unscale(Y_gp_val_s + ridge.predict(Xp_v))
        err    = float(np.max(np.abs(Y_mix - Y_val_true)))

        if best is None or err < best["maxerr"]:
            best = {"degree": deg, "pf": pf, "ridge": ridge, "maxerr": err}

    if best and best["maxerr"] < current_err:
        log.info(f"    [Poly] Enabled! MaxErr {current_err:.4f} → {best['maxerr']:.4f}")
        return {
            "degree":    int(best["degree"]),
            "powers":    best["pf"].powers_.astype(np.int16),
            "coef":      best["ridge"].coef_.astype(np.float32),
            "intercept": np.zeros(Y_tr_s.shape[1], dtype=np.float32),
            "alpha":     float(POLY_ALPHA),
        }
    log.info("    [Poly] No improvement. Keeping pure GP.")
    return None


# ── Per-cluster pipeline ───────────────────────────────────────────────────────

def train_expert(train_csv: Path, val_csv: Path, out_pt: Path):
    log.info(f"\n>>> Cluster: {out_pt.stem}")

    df_tr = pd.read_csv(train_csv)
    if "cluster_conf" in df_tr.columns:
        df_tr = df_tr[df_tr["cluster_conf"] >= CONF_THRESHOLD]
    if len(df_tr) < 50:
        log.warning("    Skipping: < 50 training samples.")
        return

    X_tr  = df_tr[INPUT_COLS].values
    Y_tr  = df_tr[OUTPUT_COLS].values

    xs = LogStandardInputScaler(log_cols=LOG_INPUTS, all_cols=INPUT_COLS).fit(X_tr)
    ys = TargetScaler().fit(Y_tr)

    X_tr_s = xs.transform(X_tr)
    Y_tr_s = ys.transform(Y_tr)
    X_t    = torch.tensor(X_tr_s, dtype=DTYPE, device=DEVICE)
    Y_t    = torch.tensor(Y_tr_s, dtype=DTYPE, device=DEVICE)

    # Load validation set
    X_val_s = Y_val_s = None
    if val_csv.exists():
        df_vl = pd.read_csv(val_csv)
        if "cluster_conf" in df_vl.columns:
            df_vl = df_vl[df_vl["cluster_conf"] >= CONF_THRESHOLD]
        if len(df_vl) > 0:
            X_val_s = xs.transform(df_vl[INPUT_COLS].values)
            Y_val_s = ys.transform(df_vl[OUTPUT_COLS].values)

    t0 = time.time()
    if len(df_tr) <= EXACT_THRESHOLD:
        gp_kind         = "exact"
        models, likes   = train_exact_gp(X_t, Y_t, EPOCHS_EXACT)
        inducing_points = None
        train_x_save    = X_t.detach().cpu()
        train_y_save    = Y_t.detach().cpu()
    else:
        gp_kind         = "svgp"
        models, likes   = train_svgp(X_t, Y_t, EPOCHS_SVGP)
        inducing_points = [m.variational_strategy.inducing_points.detach().cpu().numpy()
                           for m in models]
        train_x_save    = None
        train_y_save    = None

    log.info(f"    Training done in {time.time()-t0:.1f}s")

    # Poly residual
    poly_state = None
    if X_val_s is not None and Y_val_s is not None:
        poly_state = fit_poly_residual(
            models, likes, xs, ys, X_tr_s, Y_tr_s, X_val_s, Y_val_s
        )

    state = {
        "gp_kind":      gp_kind,
        "models":       [m.state_dict() for m in models],
        "likes":        [l.state_dict() for l in likes],
        "x_scaler":     xs.to_dict(),
        "y_scaler":     ys.to_dict(),
        "train_x":      train_x_save,
        "train_y":      train_y_save,
        "inducing":     inducing_points,
        "poly_residual": poly_state,
    }
    torch.save(state, out_pt)
    log.info(f"    Saved → {out_pt}")

    # Free GPU memory after saving
    del models, likes, X_t, Y_t
    if train_x_save is not None:
        del train_x_save, train_y_save
    torch.cuda.empty_cache()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train GP experts for each GMM cluster.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data",     type=str, default="workspace/moe_ws",
                        help="Workspace directory from prepare_data.py")
    parser.add_argument("--clusters", type=int, nargs="*", default=None,
                        help="Train only these cluster IDs (default: all)")
    parser.add_argument("--force",    action="store_true",
                        help="Retrain even if .pt already exists")
    args = parser.parse_args()

    data_dir = Path(args.data)
    pattern  = sorted(data_dir.glob("cluster*_train.csv"))

    if not pattern:
        log.error(f"No cluster*_train.csv found in {data_dir}. Run prepare_data.py first.")
        sys.exit(1)

    for tr_csv in pattern:
        cid_str  = tr_csv.stem.replace("cluster", "").replace("_train", "")
        cid      = int(cid_str)
        if args.clusters and cid not in args.clusters:
            continue

        vl_csv = data_dir / f"cluster{cid}_val.csv"
        out_pt = data_dir / f"expert_{cid}.pt"

        if out_pt.exists() and not args.force:
            log.info(f">>> Cluster {cid}: .pt exists, skipping (use --force to retrain)")
            continue

        try:
            train_expert(tr_csv, vl_csv, out_pt)
        except Exception as exc:
            log.error(f"Cluster {cid} failed: {exc}", exc_info=True)

    log.info("\nAll clusters processed.")


if __name__ == "__main__":
    main()
