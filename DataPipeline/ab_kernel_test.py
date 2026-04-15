"""
DataPipeline/ab_kernel_test.py
===============================
A/B test different GP kernel configurations on a small set of high-error
experts, without touching the production expert_*.pt files.

For each (expert, kernel) pair:
  1. Load cluster{id}_train.csv + cluster{id}_val.csv
  2. Train a fresh 8-output GP stack with the given kernel
  3. Evaluate on test_labeled.csv points routed to this expert via hard
     argmax through the hierarchical gate (SAME routing as diagnostic)
  4. Report MaxErr, RMSE, MAE, training time

Output:
  - <workspace>/ab_kernel_results.csv     — one row per (expert, kernel)
  - <workspace>/ab_kernel_detail_<ts>/    — optional per-run detail (pred vs true)

Usage
-----
# Auto-pick top-5 highest-MaxErr experts from the current diagnostic:
python DataPipeline/ab_kernel_test.py \
    --workspace Optimization/moe_workspace \
    --diagnostic Optimization/moe_workspace/diagnostic_report_post.csv \
    --top-k 5

# Specify experts explicitly:
python DataPipeline/ab_kernel_test.py \
    --workspace Optimization/moe_workspace \
    --experts 92 52 100 185 129

# Choose kernels (comma list; see KERNEL_REGISTRY):
python DataPipeline/ab_kernel_test.py \
    --workspace Optimization/moe_workspace \
    --experts 92 52 \
    --kernels baseline,matern15,rbf,rbf_noLin

Kernel registry (extend below if needed):
    baseline              : ScaleKernel(Matern(2.5) + Linear)  -- current
    matern15              : ScaleKernel(Matern(1.5) + Linear)
    matern05              : ScaleKernel(Matern(0.5) + Linear)
    rbf                   : ScaleKernel(RBF        + Linear)
    matern25_noLin        : ScaleKernel(Matern(2.5))
    rbf_noLin             : ScaleKernel(RBF)
    matern25_strongNoise  : baseline + noise floor 1e-2 (was 1e-4)
    matern25_ard          : ScaleKernel(Matern(2.5, ard=5) + Linear)   -- per-dim lengthscales
    rbf_ard               : ScaleKernel(RBF(ard=5) + Linear)

Notes
-----
* Training uses the same diff/absolute target mode as the production expert_*.pt
  (auto-detected from the baseline checkpoint when present).
* Poly-residual is intentionally DISABLED here so we measure the GP kernel's
  raw contribution. Enable later by re-using the full train_experts.py pipeline
  on the winning kernel.
* All results include the "baseline" row for direct comparison on the exact
  same test points.
"""

import argparse
import contextlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import gpytorch
import joblib
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surrogate.config import (
    INPUT_COLS, OUTPUT_COLS, LOG_INPUTS,
    EPOCHS_EXACT, LR_EXACT, CONF_THRESHOLD,
)
from surrogate.scalers import LogStandardInputScaler, TargetScaler
from surrogate.features import build_phi

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DTYPE  = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Kernel factories ────────────────────────────────────────────────────────

def _ard_dims() -> int:
    return len(INPUT_COLS)


def _kernel_baseline():
    return gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=2.5)
        + gpytorch.kernels.LinearKernel()
    )


def _kernel_matern(nu: float):
    return gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=nu)
        + gpytorch.kernels.LinearKernel()
    )


def _kernel_rbf_lin():
    return gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel()
        + gpytorch.kernels.LinearKernel()
    )


def _kernel_matern25_noLin():
    return gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))


def _kernel_rbf_noLin():
    return gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())


def _kernel_matern25_ard():
    return gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=_ard_dims())
        + gpytorch.kernels.LinearKernel()
    )


def _kernel_rbf_ard():
    return gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(ard_num_dims=_ard_dims())
        + gpytorch.kernels.LinearKernel()
    )


KERNEL_REGISTRY: Dict[str, Callable[[], gpytorch.kernels.Kernel]] = {
    "baseline":              _kernel_baseline,
    "matern15":              lambda: _kernel_matern(1.5),
    "matern05":              lambda: _kernel_matern(0.5),
    "rbf":                   _kernel_rbf_lin,
    "matern25_noLin":        _kernel_matern25_noLin,
    "rbf_noLin":             _kernel_rbf_noLin,
    "matern25_strongNoise":  _kernel_baseline,   # differs in noise floor
    "matern25_ard":          _kernel_matern25_ard,
    "rbf_ard":               _kernel_rbf_ard,
}


# Noise floor overrides per kernel name (default 1e-4)
NOISE_FLOOR_OVERRIDES = {
    "matern25_strongNoise": 1e-2,
}


# ── Differential target helpers (mirror train_experts.py) ───────────────────

def _to_diff(Y: np.ndarray) -> np.ndarray:
    D = np.empty_like(Y)
    D[:, 0] = Y[:, 0]
    D[:, 1:] = np.diff(Y, axis=1)
    return D


def _from_diff(D: np.ndarray) -> np.ndarray:
    D_clipped = np.maximum(D, 0.0)
    return np.cumsum(D_clipped, axis=1)


# ── GP model wrapper with swappable kernel ──────────────────────────────────

class SwappableKernelGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_factory):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = kernel_factory()

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def _cg_settings():
    return (gpytorch.settings.max_cg_iterations(1000),
            gpytorch.settings.cg_tolerance(0.01),
            gpytorch.settings.max_preconditioner_size(100))


def train_8_outputs(X_t, Y_t, kernel_name: str, epochs: int = EPOCHS_EXACT):
    """Train 8 independent ExactGP outputs with the given kernel."""
    factory = KERNEL_REGISTRY[kernel_name]
    noise_floor = NOISE_FLOOR_OVERRIDES.get(kernel_name, 1e-4)

    N = X_t.size(0)
    use_cg = N > 6000
    cg1, cg2, cg3 = _cg_settings()

    models, likes = [], []
    for i in range(Y_t.shape[1]):
        y = Y_t[:, i].contiguous()
        lk = GaussianLikelihood(noise_constraint=GreaterThan(noise_floor)).to(DEVICE, DTYPE)
        lk.noise = max(0.02, noise_floor * 2)
        m  = SwappableKernelGP(X_t, y, lk, factory).to(DEVICE, DTYPE)
        m.train(); lk.train()
        opt = torch.optim.Adam(m.parameters(), lr=LR_EXACT)
        mll = ExactMarginalLogLikelihood(lk, m)
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


def predict_8_outputs(models, likes, X_s: np.ndarray) -> np.ndarray:
    X_t = torch.tensor(X_s, dtype=DTYPE, device=DEVICE)
    preds = []
    cg1, cg2, cg3 = _cg_settings()
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
         gpytorch.settings.cholesky_jitter(1e-3), cg1, cg2, cg3:
        for m, lk in zip(models, likes):
            m.eval(); lk.eval()
            preds.append(lk(m(X_t)).mean.detach().cpu().numpy())
    return np.stack(preds, axis=1)   # (N, 8)


# ── Test-set routing (reuse from diagnose_and_sample) ───────────────────────

def hard_route(gate, df: pd.DataFrame) -> np.ndarray:
    """Return 1-indexed cluster IDs via hard argmax through hierarchical gate."""
    geo_gmm     = gate["geo_gmm"]
    geo_scaler  = gate["geo_scaler"]
    phi_gmms    = gate["phi_gmms"]
    phi_scalers = gate["phi_scalers"]
    k_phi_offsets = gate["k_phi_offsets"]
    K_geo = gate["k_geo"]

    geo_feat = df[["width", "height"]].values.astype(np.float64)
    geo_labels = geo_gmm.predict(geo_scaler.transform(geo_feat))
    phi_feat = build_phi(df)

    cids = np.zeros(len(df), dtype=int)
    for g in range(K_geo):
        m = geo_labels == g
        if not np.any(m):
            continue
        ps_g = phi_scalers[g].transform(phi_feat[m])
        prob_g = phi_gmms[g].predict_proba(ps_g)
        local = np.argmax(prob_g, axis=1)
        cids[m] = k_phi_offsets[g] + local + 1
    return cids


# ── A/B driver ──────────────────────────────────────────────────────────────

def detect_diff_target(workspace: Path, expert_id: int) -> bool:
    """Read the baseline expert_{id}.pt to see its target_mode."""
    pt = workspace / f"expert_{expert_id}.pt"
    if not pt.exists():
        return False
    try:
        state = torch.load(pt, map_location="cpu", weights_only=False)
        return state.get("target_mode") == "diff"
    except Exception:
        return False


def run_one(workspace: Path, expert_id: int, kernel_name: str,
            test_mask: np.ndarray, df_test: pd.DataFrame,
            diff_target: bool) -> Dict:
    """Train with kernel_name and evaluate on df_test[test_mask]."""
    train_csv = workspace / f"cluster{expert_id}_train.csv"
    val_csv   = workspace / f"cluster{expert_id}_val.csv"
    if not train_csv.exists():
        log.warning(f"Expert {expert_id}: train CSV missing, skipping")
        return None

    df_tr = pd.read_csv(train_csv)
    if "cluster_conf" in df_tr.columns:
        df_tr = df_tr[df_tr["cluster_conf"] >= CONF_THRESHOLD]
    if len(df_tr) < 50:
        log.warning(f"Expert {expert_id}: only {len(df_tr)} training rows, skipping")
        return None

    X_tr = df_tr[INPUT_COLS].values
    Y_tr = df_tr[OUTPUT_COLS].values
    if diff_target:
        Y_tr = _to_diff(Y_tr)

    xs = LogStandardInputScaler(log_cols=LOG_INPUTS, all_cols=INPUT_COLS).fit(X_tr)
    ys = TargetScaler().fit(Y_tr)
    X_tr_s = xs.transform(X_tr)
    Y_tr_s = ys.transform(Y_tr)
    X_t = torch.tensor(X_tr_s, dtype=DTYPE, device=DEVICE)
    Y_t = torch.tensor(Y_tr_s, dtype=DTYPE, device=DEVICE)

    t0 = time.time()
    try:
        models, likes = train_8_outputs(X_t, Y_t, kernel_name)
    except Exception as exc:
        log.warning(f"Expert {expert_id} kernel {kernel_name}: train failed: {exc}")
        return {
            "expert": expert_id, "kernel": kernel_name, "status": f"train_error: {exc}",
            "n_train": len(df_tr), "n_test": int(test_mask.sum()),
            "maxerr": np.nan, "rmse": np.nan, "mae": np.nan,
            "train_sec": 0.0,
        }
    train_sec = time.time() - t0

    # Evaluate on routed test points
    n_test = int(test_mask.sum())
    if n_test == 0:
        del models, likes
        torch.cuda.empty_cache()
        return {
            "expert": expert_id, "kernel": kernel_name, "status": "no_test_samples",
            "n_train": len(df_tr), "n_test": 0,
            "maxerr": np.nan, "rmse": np.nan, "mae": np.nan,
            "train_sec": train_sec,
        }

    X_te = df_test.loc[test_mask, INPUT_COLS].values
    Y_te = df_test.loc[test_mask, OUTPUT_COLS].values  # absolute (truth)
    X_te_s = xs.transform(X_te)

    Y_pred_s = predict_8_outputs(models, likes, X_te_s)
    Y_pred   = ys.inverse_transform(Y_pred_s)
    if diff_target:
        Y_pred = _from_diff(Y_pred)

    err = Y_pred - Y_te
    abs_e = np.abs(err)
    maxerr = float(abs_e.max())
    rmse   = float(np.sqrt((err ** 2).mean()))
    mae    = float(abs_e.mean())

    # Cleanup
    del models, likes, X_t, Y_t
    torch.cuda.empty_cache()

    return {
        "expert": expert_id, "kernel": kernel_name, "status": "ok",
        "n_train": len(df_tr), "n_test": n_test,
        "maxerr": maxerr, "rmse": rmse, "mae": mae,
        "train_sec": round(train_sec, 1),
        "diff_target": diff_target,
    }


def select_top_experts(diagnostic_csv: Path, top_k: int) -> List[int]:
    df = pd.read_csv(diagnostic_csv)
    df = df[df["status"] == "ok"].sort_values("maxerr", ascending=False)
    return df.head(top_k)["expert"].astype(int).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--diagnostic", type=Path, default=None,
                    help="Diagnostic CSV to pick top-k worst experts from "
                         "(default: <workspace>/diagnostic_report_post.csv)")
    ap.add_argument("--top-k", type=int, default=5,
                    help="Pick top-k experts by MaxErr from diagnostic")
    ap.add_argument("--experts", type=int, nargs="*", default=None,
                    help="Explicit expert list (overrides --top-k)")
    ap.add_argument("--kernels", type=str,
                    default="baseline,matern15,rbf,rbf_noLin,matern25_ard,rbf_ard",
                    help="Comma-separated kernel names from KERNEL_REGISTRY")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override EPOCHS_EXACT")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output CSV (default: <workspace>/ab_kernel_results.csv)")
    args = ap.parse_args()

    # Override epoch count globally if requested
    global EPOCHS_EXACT
    if args.epochs is not None:
        EPOCHS_EXACT = args.epochs

    ws = args.workspace.resolve()

    # Kernels to test
    kernel_names = [k.strip() for k in args.kernels.split(",") if k.strip()]
    for kn in kernel_names:
        if kn not in KERNEL_REGISTRY:
            raise SystemExit(f"Unknown kernel: {kn}. Options: {list(KERNEL_REGISTRY)}")

    # Expert list
    if args.experts:
        experts = list(args.experts)
    else:
        diag_csv = args.diagnostic or (ws / "diagnostic_report_post.csv")
        if not diag_csv.exists():
            raise SystemExit(f"Diagnostic CSV not found: {diag_csv}")
        experts = select_top_experts(diag_csv, args.top_k)
        log.info(f"Picked top-{args.top_k} experts from {diag_csv}: {experts}")

    # Load gate + test set for routing
    gate = joblib.load(ws / "gmm_gate.joblib")
    df_test = pd.read_csv(ws / "test_labeled.csv")
    hard_cids = hard_route(gate, df_test)
    log.info(f"Test set: {len(df_test):,} rows; device={DEVICE}")

    rows = []
    for eid in experts:
        mask = hard_cids == eid
        diff_target = detect_diff_target(ws, eid)
        log.info(f"\n=== Expert {eid}  (n_test={int(mask.sum())}, "
                 f"diff_target={diff_target}) ===")
        for kn in kernel_names:
            log.info(f"  -- kernel: {kn}")
            r = run_one(ws, eid, kn, mask, df_test, diff_target)
            if r is None:
                continue
            log.info(f"     MaxErr={r.get('maxerr'):.4f}  RMSE={r.get('rmse'):.4f}  "
                     f"MAE={r.get('mae'):.4f}  time={r.get('train_sec')}s")
            rows.append(r)

    if not rows:
        log.error("No results produced.")
        return

    out = args.out or (ws / "ab_kernel_results.csv")
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out, index=False)
    log.info(f"\nSaved: {out}")

    # Pivot summary: experts x kernels
    print("\n=== MaxErr pivot (rows=expert, cols=kernel) ===")
    piv = df_out.pivot(index="expert", columns="kernel", values="maxerr")
    # Keep kernel column order as given
    piv = piv[[c for c in kernel_names if c in piv.columns]]
    print(piv.round(4).to_string())

    print("\n=== Δ vs baseline (negative = improvement) ===")
    if "baseline" in piv.columns:
        delta = piv.subtract(piv["baseline"], axis=0)
        print(delta.round(4).to_string())

        # Percent improvement per kernel averaged across experts
        print("\n=== Mean % change vs baseline ===")
        pct = 100.0 * delta.divide(piv["baseline"], axis=0)
        print(pct.mean().round(2).to_string())
    else:
        print("  (baseline column missing — skip comparison)")


if __name__ == "__main__":
    main()
