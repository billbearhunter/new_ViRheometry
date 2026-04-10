"""
DataPipeline/prepare_data.py
============================
Step 2 – GMM clustering + stratified train/val/test split.

Reads a merged CSV (may combine base + targeted hard-region data),
clusters on the φ feature vector, then writes per-cluster train/val CSVs
and bounding boxes used during CMA-ES optimisation.

Usage
-----
# Fresh clustering on a single CSV:
python prepare_data.py --csv workspace/data.csv --out workspace/moe_ws

# Merge base data + hard-region supplement before clustering:
python prepare_data.py --csv workspace/data.csv workspace/data_hard.csv \\
                       --out workspace/moe_ws

# Force re-clustering even if artifacts already exist:
python prepare_data.py --csv workspace/data.csv --out workspace/moe_ws --force
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "DataPipeline"))

from dp_config import (
    INPUT_COLS, OUTPUT_COLS,
    N_CLUSTERS, CONF_THRESHOLD, BOX_CONF_THRESH, OUTLIER_Z_THRESH,
    MIN_N, MAX_N, MIN_ETA, MAX_ETA, MIN_SIGMA_Y, MAX_SIGMA_Y,
)
from moe_utils import build_phi

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

GLOBAL_BOUNDS = {
    "n":       (MIN_N,       MAX_N),
    "eta":     (MIN_ETA,     MAX_ETA),
    "sigma_y": (MIN_SIGMA_Y, MAX_SIGMA_Y),
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def stratified_split(df: pd.DataFrame, seed: int = 42):
    """80 / 10 / 10 split stratified on log(η) × log(σ_y) bins.

    Falls back to unstratified split when any stratum has fewer than 2 members
    (e.g. very small datasets used for testing).
    """
    eps      = 1e-6
    log_eta  = np.log(df["eta"].values + eps)
    log_sig  = np.log(df["sigma_y"].values + eps)
    bins_eta = pd.qcut(log_eta, 10, labels=False, duplicates="drop")
    bins_sig = pd.qcut(log_sig, 10, labels=False, duplicates="drop")
    strata   = pd.Series(bins_eta).astype(str).values + "_" + pd.Series(bins_sig).astype(str).values

    def _min_count(s): return pd.Series(s).value_counts().min()

    idx = np.arange(len(df))
    # First split: train / tmp
    strat1 = strata if _min_count(strata) >= 2 else None
    idx_train, idx_tmp = train_test_split(idx, test_size=0.2, random_state=seed, stratify=strat1)
    # Second split: val / test
    strata_tmp = strata[idx_tmp]
    strat2 = strata_tmp if _min_count(strata_tmp) >= 2 else None
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.5, random_state=seed, stratify=strat2)
    return df.iloc[idx_train], df.iloc[idx_val], df.iloc[idx_test]


def remove_outliers(df: pd.DataFrame, z_thresh: float = OUTLIER_Z_THRESH) -> pd.DataFrame:
    if len(df) < 10:
        return df
    z    = np.abs(stats.zscore(df[OUTPUT_COLS]))
    mask = (z < z_thresh).all(axis=1)
    n_rm = (~mask).sum()
    if n_rm:
        log.info(f"      Removed {n_rm} outliers (|Z| > {z_thresh})")
    return df[mask]


def compute_box(df: pd.DataFrame, cluster_id: int,
                conf_threshold: float = BOX_CONF_THRESH,
                q_low: float = 0.02, q_high: float = 0.98,
                buffer: float = 0.10) -> dict:
    """Robust bounding box for a cluster (used to restrict CMA-ES search)."""
    subset = df[(df["cluster_id"] == cluster_id) & (df["cluster_conf"] >= conf_threshold)]
    if len(subset) < 50:
        return {k: list(v) for k, v in GLOBAL_BOUNDS.items()}

    box, eps = {}, 1e-6
    for col in ["n", "eta", "sigma_y"]:
        vals = subset[col].values
        if col in ("eta", "sigma_y"):
            lv    = np.log(vals + eps)
            lo, hi = np.quantile(lv, q_low), np.quantile(lv, q_high)
            span  = hi - lo
            bmin  = np.exp(lo - buffer * span)
            bmax  = np.exp(hi + buffer * span)
        else:
            lo, hi = np.quantile(vals, q_low), np.quantile(vals, q_high)
            span  = hi - lo
            bmin, bmax = lo - buffer * span, hi + buffer * span
        glo, ghi = GLOBAL_BOUNDS[col]
        box[col] = [float(max(glo, bmin)), float(min(ghi, bmax))]
    return box


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GMM clustering + data split for MoE training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--csv",    nargs="+", required=True,
                        help="Input CSV file(s). Multiple files are merged.")
    parser.add_argument("--out",    type=str, default="workspace/moe_ws",
                        help="Output workspace directory (default: workspace/moe_ws)")
    parser.add_argument("--k",      type=int, default=N_CLUSTERS,
                        help=f"Number of GMM clusters (default: {N_CLUSTERS})")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--force",  action="store_true",
                        help="Force re-clustering even if artifacts exist")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gmm_path   = out_dir / "gmm_gate.joblib"
    train_path = out_dir / "train_labeled.csv"

    if gmm_path.exists() and train_path.exists() and not args.force:
        log.info("GMM artifacts found. Use --force to overwrite. Exiting.")
        return

    # 1. Load & merge CSVs
    dfs = []
    for p in args.csv:
        df_i = pd.read_csv(p)
        log.info(f"  Loaded {len(df_i):,} rows from {p}")
        dfs.append(df_i)
    df = pd.concat(dfs, ignore_index=True).dropna(subset=INPUT_COLS + OUTPUT_COLS)
    log.info(f"Total after merge: {len(df):,} rows")

    # 2. Stratified split
    log.info("Splitting data (80/10/10)...")
    train_df, val_df, test_df = stratified_split(df, seed=args.seed)

    # 3. Train GMM on φ from training set
    log.info(f"Training GMM (K={args.k})...")
    phi_train  = build_phi(train_df)
    phi_scaler = StandardScaler().fit(phi_train)
    phi_scaled = phi_scaler.transform(phi_train)
    gmm        = GaussianMixture(
        n_components=args.k, covariance_type="diag",
        random_state=args.seed, n_init=5,
    ).fit(phi_scaled)

    # 4. Assign clusters to all splits
    def assign(frame: pd.DataFrame) -> pd.DataFrame:
        phi  = build_phi(frame)
        ps   = phi_scaler.transform(phi)
        prob = gmm.predict_proba(ps)
        out  = frame.copy()
        out["cluster_id"]   = np.argmax(prob, axis=1) + 1   # 1-indexed
        out["cluster_conf"] = np.max(prob, axis=1)
        return out

    train_lab = assign(train_df)
    val_lab   = assign(val_df)
    test_lab  = assign(test_df)

    # 5. Save GMM gate
    joblib.dump({"gmm": gmm, "scaler": phi_scaler, "k": args.k},
                gmm_path)
    log.info(f"Saved GMM gate → {gmm_path}")

    # 6. Per-cluster train/val files + bounding boxes
    boxes: dict = {}
    for cid in range(1, args.k + 1):
        mask_tr = (train_lab["cluster_id"] == cid) & (train_lab["cluster_conf"] >= CONF_THRESHOLD)
        sub_tr  = remove_outliers(train_lab[mask_tr])
        sub_tr.to_csv(out_dir / f"cluster{cid}_train.csv", index=False)

        mask_vl = (val_lab["cluster_id"] == cid) & (val_lab["cluster_conf"] >= CONF_THRESHOLD)
        sub_vl  = val_lab[mask_vl]
        sub_vl.to_csv(out_dir / f"cluster{cid}_val.csv", index=False)

        log.info(f"  Cluster {cid:>3}: train={len(sub_tr):>5}, val={len(sub_vl):>4}")
        boxes[str(cid)] = compute_box(train_lab, cid)

    with (out_dir / "boxes.json").open("w") as f:
        json.dump(boxes, f, indent=4)

    train_lab.to_csv(out_dir / "train_labeled.csv",  index=False)
    val_lab.to_csv  (out_dir / "val_labeled.csv",    index=False)
    test_lab.to_csv (out_dir / "test_labeled.csv",   index=False)

    log.info(f"\nStep 2 complete. Workspace: {out_dir}/")


if __name__ == "__main__":
    main()
