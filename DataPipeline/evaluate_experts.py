"""
DataPipeline/evaluate_experts.py
=================================
Step 4 – Evaluate trained GP experts on the held-out test set.

For each expert_{N}.pt in the workspace:
  - Loads the expert and applies GMM gating to route test samples
  - Computes per-output MAE, RMSE, MaxErr
  - Optionally generates parity plots (requires matplotlib)

Usage
-----
# Evaluate all experts in workspace:
python evaluate_experts.py --data workspace/moe_ws

# Save per-cluster metrics to CSV:
python evaluate_experts.py --data workspace/moe_ws --out-csv metrics.csv

# Also generate parity plots (saved as PNG):
python evaluate_experts.py --data workspace/moe_ws --plots
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "DataPipeline"))

from dp_config import INPUT_COLS, OUTPUT_COLS, CONF_THRESHOLD
from moe_utils import (
    DEVICE, DTYPE,
    load_expert, predict_with_expert,
    build_phi,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    """Return per-output and aggregate metrics."""
    err   = Y_pred - Y_true
    abs_e = np.abs(err)
    return {
        "mae":    float(abs_e.mean()),
        "rmse":   float(np.sqrt((err ** 2).mean())),
        "maxerr": float(abs_e.max()),
        "mae_per_output":    abs_e.mean(axis=0).tolist(),
        "maxerr_per_output": abs_e.max(axis=0).tolist(),
    }


# ── Parity plot ────────────────────────────────────────────────────────────────

def _parity_plot(Y_true, Y_pred, cluster_id, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping parity plots.")
        return

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.scatter(Y_true[:, i], Y_pred[:, i], s=4, alpha=0.4)
        lo = min(Y_true[:, i].min(), Y_pred[:, i].min())
        hi = max(Y_true[:, i].max(), Y_pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel(f"True x_{i+1:02d}")
        ax.set_ylabel(f"Pred x_{i+1:02d}")
        mae = float(np.abs(Y_pred[:, i] - Y_true[:, i]).mean())
        ax.set_title(f"x_{i+1:02d}  MAE={mae:.3f}")
    fig.suptitle(f"Cluster {cluster_id}", fontsize=13)
    fig.tight_layout()
    path = out_dir / f"parity_{cluster_id}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log.info(f"    Parity plot → {path}")


# ── Global MoE evaluation ─────────────────────────────────────────────────────

def evaluate_global(data_dir: Path, test_csv: Path, plots: bool) -> pd.DataFrame:
    """Route all test samples through the trained MoE and compute global metrics."""
    gmm_path = data_dir / "gmm_gate.joblib"
    if not gmm_path.exists():
        log.error(f"GMM gate not found: {gmm_path}")
        return pd.DataFrame()

    gate   = joblib.load(gmm_path)
    gmm    = gate["gmm"]
    scaler = gate["scaler"]
    k      = gate["k"]

    df_test = pd.read_csv(test_csv)
    log.info(f"Test set: {len(df_test):,} rows")

    # GMM routing
    phi  = build_phi(df_test)
    ps   = scaler.transform(phi)
    prob = gmm.predict_proba(ps)
    cids = np.argmax(prob, axis=1) + 1   # 1-indexed

    rows = []
    all_preds  = np.zeros((len(df_test), len(OUTPUT_COLS)))
    all_routed = np.zeros(len(df_test), dtype=bool)

    for cid in range(1, k + 1):
        pt_path = data_dir / f"expert_{cid}.pt"
        if not pt_path.exists():
            continue

        mask = (cids == cid)
        if not mask.any():
            continue

        X_raw = df_test.loc[mask, INPUT_COLS].values
        Y_true = df_test.loc[mask, OUTPUT_COLS].values

        try:
            gp_kind, models, likes, xs_dict, ys_dict, poly = load_expert(str(pt_path))
            Y_pred = predict_with_expert(models, likes, xs_dict, ys_dict, poly, X_raw)
        except Exception as exc:
            log.warning(f"  Cluster {cid} predict failed: {exc}")
            continue

        all_preds[mask]  = Y_pred
        all_routed[mask] = True

        m = compute_metrics(Y_true, Y_pred)
        rows.append({
            "cluster":  cid,
            "n_test":   int(mask.sum()),
            "gp_kind":  gp_kind,
            "mae":      m["mae"],
            "rmse":     m["rmse"],
            "maxerr":   m["maxerr"],
            **{f"mae_x{i+1:02d}": m["mae_per_output"][i]    for i in range(len(OUTPUT_COLS))},
            **{f"max_x{i+1:02d}": m["maxerr_per_output"][i] for i in range(len(OUTPUT_COLS))},
        })

        log.info(f"  Cluster {cid:>3}: n={mask.sum():>4}  MAE={m['mae']:.4f}  "
                 f"MaxErr={m['maxerr']:.4f}  [{gp_kind}]")

        if plots:
            _parity_plot(Y_true, Y_pred, cid, data_dir)

    # Global metrics (routed samples only)
    if all_routed.any():
        Y_true_all = df_test.loc[all_routed, OUTPUT_COLS].values
        Y_pred_all = all_preds[all_routed]
        gm = compute_metrics(Y_true_all, Y_pred_all)
        rows.append({
            "cluster":  "GLOBAL",
            "n_test":   int(all_routed.sum()),
            "gp_kind":  "mixed",
            "mae":      gm["mae"],
            "rmse":     gm["rmse"],
            "maxerr":   gm["maxerr"],
            **{f"mae_x{i+1:02d}": gm["mae_per_output"][i]    for i in range(len(OUTPUT_COLS))},
            **{f"max_x{i+1:02d}": gm["maxerr_per_output"][i] for i in range(len(OUTPUT_COLS))},
        })
        log.info(f"\n  GLOBAL  : n={all_routed.sum():>5}  MAE={gm['mae']:.4f}  "
                 f"MaxErr={gm['maxerr']:.4f}")

    return pd.DataFrame(rows)


# ── Per-cluster evaluation (uses pre-split cluster CSVs) ─────────────────────

def evaluate_per_cluster(data_dir: Path, plots: bool) -> pd.DataFrame:
    """Evaluate each expert on its own val CSV (quick sanity check)."""
    rows = []
    for pt_path in sorted(data_dir.glob("expert_*.pt")):
        cid_str = pt_path.stem.replace("expert_", "")
        cid     = int(cid_str)
        val_csv = data_dir / f"cluster{cid}_val.csv"
        if not val_csv.exists():
            continue

        df_val = pd.read_csv(val_csv)
        if len(df_val) == 0:
            continue

        X_raw  = df_val[INPUT_COLS].values
        Y_true = df_val[OUTPUT_COLS].values

        try:
            gp_kind, models, likes, xs_dict, ys_dict, poly = load_expert(str(pt_path))
            Y_pred = predict_with_expert(models, likes, xs_dict, ys_dict, poly, X_raw)
        except Exception as exc:
            log.warning(f"  Cluster {cid} failed: {exc}")
            continue

        m = compute_metrics(Y_true, Y_pred)
        rows.append({
            "cluster":  cid,
            "n_val":    len(df_val),
            "gp_kind":  gp_kind,
            "mae":      m["mae"],
            "rmse":     m["rmse"],
            "maxerr":   m["maxerr"],
            **{f"mae_x{i+1:02d}": m["mae_per_output"][i]    for i in range(len(OUTPUT_COLS))},
            **{f"max_x{i+1:02d}": m["maxerr_per_output"][i] for i in range(len(OUTPUT_COLS))},
        })
        log.info(f"  Cluster {cid:>3}: n_val={len(df_val):>4}  MAE={m['mae']:.4f}  "
                 f"MaxErr={m['maxerr']:.4f}  [poly={'yes' if poly else 'no'}]")

        if plots:
            _parity_plot(Y_true, Y_pred, cid, data_dir)

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained GP experts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data",     type=str, default="workspace/moe_ws",
                        help="Workspace directory from prepare_data.py / train_experts.py")
    parser.add_argument("--mode",     choices=["val", "test", "both"], default="both",
                        help="Evaluate on val CSVs, global test CSV, or both (default: both)")
    parser.add_argument("--out-csv",  type=str, default=None,
                        help="Save metrics table to this CSV path")
    parser.add_argument("--plots",    action="store_true",
                        help="Generate parity plots (saved as PNG in workspace dir)")
    args = parser.parse_args()

    data_dir = Path(args.data)

    frames = []

    if args.mode in ("val", "both"):
        log.info("\n=== Per-cluster val evaluation ===")
        df_val = evaluate_per_cluster(data_dir, plots=args.plots)
        if not df_val.empty:
            df_val.insert(0, "split", "val")
            frames.append(df_val)

    if args.mode in ("test", "both"):
        test_csv = data_dir / "test_labeled.csv"
        if test_csv.exists():
            log.info("\n=== Global test evaluation (MoE routing) ===")
            df_test = evaluate_global(data_dir, test_csv, plots=args.plots)
            if not df_test.empty:
                df_test.insert(0, "split", "test")
                frames.append(df_test)
        else:
            log.warning(f"test_labeled.csv not found in {data_dir} — skipping test eval.")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        if args.out_csv:
            out = Path(args.out_csv)
            combined.to_csv(out, index=False, float_format="%.6f")
            log.info(f"\nMetrics saved → {out}")
        else:
            # Print summary to terminal
            cols = ["split", "cluster", "n_test" if "n_test" in combined.columns else "n_val",
                    "gp_kind", "mae", "rmse", "maxerr"]
            cols = [c for c in cols if c in combined.columns]
            print("\n" + combined[cols].to_string(index=False))
    else:
        log.warning("No metrics computed.")

    log.info("\nEvaluation complete.")


if __name__ == "__main__":
    main()
