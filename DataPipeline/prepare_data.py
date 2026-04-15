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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surrogate.config import (
    INPUT_COLS, OUTPUT_COLS,
    N_CLUSTERS, CONF_THRESHOLD, BOX_CONF_THRESH, OUTLIER_Z_THRESH,
    MIN_N, MAX_N, MIN_ETA, MAX_ETA, MIN_SIGMA_Y, MAX_SIGMA_Y,
    MIN_WIDTH, MAX_WIDTH, MIN_HEIGHT, MAX_HEIGHT,
)
from surrogate.features import build_phi, build_input_features

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

GLOBAL_BOUNDS = {
    "n":       (MIN_N,       MAX_N),
    "eta":     (MIN_ETA,     MAX_ETA),
    "sigma_y": (MIN_SIGMA_Y, MAX_SIGMA_Y),
    "width":   (MIN_WIDTH,   MAX_WIDTH),
    "height":  (MIN_HEIGHT,  MAX_HEIGHT),
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
    """Robust 5D bounding box for a cluster.

    Covers (n, eta, sigma_y, width, height).  The width/height bounds are
    used by bo_collect.py to focus LHS sampling; CMA-ES only searches over
    (n, eta, sigma_y) so the extra dims don't affect the optimisation path.
    """
    subset = df[(df["cluster_id"] == cluster_id) & (df["cluster_conf"] >= conf_threshold)]
    if len(subset) < 50:
        return {k: list(v) for k, v in GLOBAL_BOUNDS.items()}

    LOG_COLS = {"eta", "sigma_y"}
    box, eps = {}, 1e-6
    for col in ["n", "eta", "sigma_y", "width", "height"]:
        vals = subset[col].values
        if col in LOG_COLS:
            lv     = np.log(vals + eps)
            lo, hi = np.quantile(lv, q_low), np.quantile(lv, q_high)
            span   = hi - lo
            bmin   = np.exp(lo - buffer * span)
            bmax   = np.exp(hi + buffer * span)
        else:
            lo, hi = np.quantile(vals, q_low), np.quantile(vals, q_high)
            span   = hi - lo
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
    parser.add_argument("--cov-type", type=str, default="full",
                        choices=["full", "diag", "tied", "spherical"],
                        help="GMM covariance type (default: full)")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--gate-space", type=str, default="phi",
                        choices=["phi", "input", "hierarchical"],
                        help="GMM gating space: 'phi' (observation features, 18D), "
                             "'input' (parameter space 5D), or "
                             "'hierarchical' (geometry groups + per-group phi)")
    parser.add_argument("--k-geo",  type=int, default=12,
                        help="Number of geometry groups for hierarchical mode (default: 12, BIC-justified)")
    parser.add_argument("--k-phi",  type=int, nargs="*", default=None,
                        help="Number of phi clusters per geometry group. "
                             "Single value = same for all groups; "
                             "multiple values = per-group (must match --k-geo). "
                             "Default: BIC-justified per group.")
    parser.add_argument("--re-threshold", type=float, default=100.0,
                        help="Stage 0: Re threshold for splashing filter (default: 100). "
                             "Set to 0 to disable.")
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

    # 1b. Stage 0: Flow regime pre-classification (Re filter)
    if args.re_threshold > 0:
        from surrogate.regime_filter import filter_splashing
        df, df_splash = filter_splashing(df, re_c=args.re_threshold)
        if len(df_splash) > 0:
            splash_path = out_dir / "splash_excluded.csv"
            df_splash.to_csv(splash_path, index=False)
            log.info(f"  Splashing samples saved → {splash_path}")

    # 2. Stratified split
    log.info("Splitting data (80/10/10)...")
    train_df, val_df, test_df = stratified_split(df, seed=args.seed)

    # 3. Train GMM
    gate_space = args.gate_space
    GMM_FIT_CAP = 30_000

    def _fit_gmm(features, k, cov_type, seed, label=""):
        scaler = StandardScaler().fit(features)
        scaled = scaler.transform(features)
        if len(scaled) > GMM_FIT_CAP:
            rng = np.random.RandomState(seed)
            fit_idx = rng.choice(len(scaled), GMM_FIT_CAP, replace=False)
            fit_data = scaled[fit_idx]
            log.info(f"  {label}Subsampling {GMM_FIT_CAP:,} / {len(scaled):,}")
        else:
            fit_data = scaled
        gmm = GaussianMixture(
            n_components=k, covariance_type=cov_type,
            random_state=seed, n_init=10, reg_covar=1e-5, max_iter=300,
        ).fit(fit_data)
        return gmm, scaler

    if gate_space == "hierarchical":
        # ── Two-stage hierarchical clustering ────────────────────────────
        K_geo = args.k_geo

        # Resolve per-group K_phi list
        if args.k_phi is None:
            # Default: uniform K_phi=20 for all groups (BIC-justified)
            k_phi_list = [20] * K_geo
        elif len(args.k_phi) == 1:
            k_phi_list = [args.k_phi[0]] * K_geo
        else:
            if len(args.k_phi) != K_geo:
                raise ValueError(f"--k-phi has {len(args.k_phi)} values but --k-geo={K_geo}")
            k_phi_list = args.k_phi

        total_k = sum(k_phi_list)
        log.info(f"HIERARCHICAL mode: K_geo={K_geo}, K_phi={k_phi_list}, total={total_k} experts")

        # Stage 1: Geometry GMM on (W, H)
        log.info("Stage 1: Clustering by geometry (W, H)...")
        geo_features = train_df[["width", "height"]].values.astype(np.float64)
        geo_gmm, geo_scaler = _fit_gmm(geo_features, K_geo, "full", args.seed, "Geo: ")

        geo_labels_train = geo_gmm.predict(geo_scaler.transform(geo_features))
        for g in range(K_geo):
            n_g = (geo_labels_train == g).sum()
            sub = train_df.iloc[geo_labels_train == g]
            log.info(f"  Geo group {g}: {n_g:,} samples, "
                     f"W=[{sub['width'].min():.1f},{sub['width'].max():.1f}], "
                     f"H=[{sub['height'].min():.1f},{sub['height'].max():.1f}]")

        # Stage 2: Per-group φ-space GMMs
        log.info("Stage 2: Per-group phi-space clustering...")
        phi_gmms = []
        phi_scalers = []
        phi_all_train = build_phi(train_df)

        for g in range(K_geo):
            K_phi_g = k_phi_list[g]
            mask_g = geo_labels_train == g
            phi_g = phi_all_train[mask_g]
            log.info(f"  Geo group {g}: fitting GMM(K={K_phi_g}) on {len(phi_g)} samples...")
            gmm_g, scaler_g = _fit_gmm(phi_g, K_phi_g, args.cov_type, args.seed + g + 1, f"G{g}: ")
            phi_gmms.append(gmm_g)
            phi_scalers.append(scaler_g)

        # Compute cumulative offsets for global cluster IDs
        # e.g. k_phi_list=[8,10,7] → offsets=[0,8,18], total=25
        k_phi_offsets = [0]
        for kp in k_phi_list[:-1]:
            k_phi_offsets.append(k_phi_offsets[-1] + kp)

        # Assign global cluster IDs
        def assign_hierarchical(frame: pd.DataFrame) -> pd.DataFrame:
            geo_feat = frame[["width", "height"]].values.astype(np.float64)
            geo_labels = geo_gmm.predict(geo_scaler.transform(geo_feat))
            phi_feat = build_phi(frame)

            cluster_ids = np.zeros(len(frame), dtype=int)
            cluster_confs = np.zeros(len(frame), dtype=float)

            for g in range(K_geo):
                mask_g = geo_labels == g
                if not np.any(mask_g):
                    continue
                phi_g = phi_feat[mask_g]
                ps_g = phi_scalers[g].transform(phi_g)
                prob_g = phi_gmms[g].predict_proba(ps_g)
                local_ids = np.argmax(prob_g, axis=1)  # 0-indexed within group
                global_ids = k_phi_offsets[g] + local_ids + 1  # 1-indexed global
                cluster_ids[mask_g] = global_ids
                cluster_confs[mask_g] = np.max(prob_g, axis=1)

            out = frame.copy()
            out["cluster_id"] = cluster_ids
            out["cluster_conf"] = cluster_confs
            return out

        train_lab = assign_hierarchical(train_df)
        val_lab = assign_hierarchical(val_df)
        test_lab = assign_hierarchical(test_df)

        # Save hierarchical gate
        joblib.dump({
            "gate_space": "hierarchical",
            "k": total_k,
            "k_geo": K_geo,
            "k_phi": k_phi_list,          # per-group list (not single int)
            "k_phi_offsets": k_phi_offsets,
            "geo_gmm": geo_gmm,
            "geo_scaler": geo_scaler,
            "phi_gmms": phi_gmms,
            "phi_scalers": phi_scalers,
        }, gmm_path)
        log.info(f"Saved hierarchical gate → {gmm_path}")
        effective_k = total_k

    else:
        # ── Original single-stage clustering ─────────────────────────────
        if gate_space == "input":
            log.info(f"Training GMM in INPUT space (n, log_η, log_σ_y, W, H) (K={args.k})...")
            phi_train = build_input_features(train_df)
        else:
            log.info(f"Training GMM in PHI space (18D observation features) (K={args.k})...")
            phi_train = build_phi(train_df)

        gmm, phi_scaler = _fit_gmm(phi_train, args.k, args.cov_type, args.seed)

        def assign(frame: pd.DataFrame) -> pd.DataFrame:
            if gate_space == "input":
                phi = build_input_features(frame)
            else:
                phi = build_phi(frame)
            ps = phi_scaler.transform(phi)
            prob = gmm.predict_proba(ps)
            out = frame.copy()
            out["cluster_id"] = np.argmax(prob, axis=1) + 1
            out["cluster_conf"] = np.max(prob, axis=1)
            return out

        train_lab = assign(train_df)
        val_lab = assign(val_df)
        test_lab = assign(test_df)

        joblib.dump({"gmm": gmm, "scaler": phi_scaler, "k": args.k,
                     "gate_space": gate_space}, gmm_path)
        log.info(f"Saved GMM gate → {gmm_path}")
        effective_k = args.k

    # 6. Per-cluster train/val files + bounding boxes
    boxes: dict = {}
    for cid in range(1, effective_k + 1):
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
