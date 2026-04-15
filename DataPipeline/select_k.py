"""
DataPipeline/select_k.py
========================
BIC / AIC screening to choose the optimal number of GMM components.

Supports three modes:
  1. phi-space BIC    (--mode phi)     — for single-stage clustering
  2. geometry BIC     (--mode geo)     — for Stage 1 K_geo selection
  3. per-group phi BIC (--mode hier)   — for Stage 2 per-group K_phi selection

Usage
-----
# Stage 1: find optimal K_geo on (W, H) space
python select_k.py --csv data.csv --mode geo --ks 2 3 4 5 6 7 8 9 10

# Stage 2: given K_geo, find per-group K_phi
python select_k.py --csv data.csv --mode hier --k-geo 5 --ks 3 5 8 10 12 15 20

# Legacy single-stage: test K on phi space
python select_k.py --csv data.csv --mode phi --ks 40 60 80 100 120
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surrogate.features import build_phi, build_input_features


def _fit_and_score(features, k, cov_type, seed, subsample, label=""):
    """Fit GMM and return BIC/AIC scores."""
    scaler = StandardScaler().fit(features)
    scaled = scaler.transform(features)

    if len(scaled) > subsample:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(scaled), subsample, replace=False)
        fit_data = scaled[idx]
        print(f"  {label}Subsampled {subsample:,} / {len(scaled):,}")
    else:
        fit_data = scaled

    t0 = time.time()
    gmm = GaussianMixture(
        n_components=k, covariance_type=cov_type,
        random_state=seed, n_init=5, reg_covar=1e-5, max_iter=300,
    ).fit(fit_data)
    elapsed = time.time() - t0

    bic = gmm.bic(fit_data)
    aic = gmm.aic(fit_data)
    return bic, aic, elapsed


def run_phi_bic(csv_path, ks, subsample, cov_type, seed):
    """Single-stage BIC screening on phi-space."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows")

    phi = build_phi(df)
    print(f"phi shape: {phi.shape} (18D)")

    print(f"\n{'K':>5}  {'BIC':>14}  {'AIC':>14}  {'time(s)':>9}")
    print("-" * 50)

    results = []
    for k in ks:
        bic, aic, t = _fit_and_score(phi, k, cov_type, seed, subsample)
        results.append({"K": k, "BIC": bic, "AIC": aic, "time_s": t})
        print(f"{k:5d}  {bic:14.0f}  {aic:14.0f}  {t:9.1f}")

    _print_summary(results, len(df))
    return results


def run_geo_bic(csv_path, ks, subsample, cov_type, seed):
    """Stage 1: BIC screening on (W, H) geometry space."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows")

    geo = df[["width", "height"]].values.astype(np.float64)
    print(f"geo features shape: {geo.shape} (2D: W, H)")

    print(f"\n{'K_geo':>5}  {'BIC':>14}  {'AIC':>14}  {'time(s)':>9}")
    print("-" * 50)

    results = []
    for k in ks:
        bic, aic, t = _fit_and_score(geo, k, cov_type, seed, subsample)
        results.append({"K": k, "BIC": bic, "AIC": aic, "time_s": t})
        print(f"{k:5d}  {bic:14.0f}  {aic:14.0f}  {t:9.1f}")

    _print_summary(results, len(df), label="K_geo")
    return results


def run_hierarchical_bic(csv_path, k_geo, ks_phi, subsample, cov_type, seed):
    """Stage 2: per-group BIC screening on phi-space within each geo group.

    First fits a geometry GMM with k_geo components, then for each group
    runs BIC screening over candidate K_phi values.
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows")

    # Stage 1: fit geometry GMM
    geo = df[["width", "height"]].values.astype(np.float64)
    geo_scaler = StandardScaler().fit(geo)
    geo_scaled = geo_scaler.transform(geo)

    geo_gmm = GaussianMixture(
        n_components=k_geo, covariance_type="full",
        random_state=seed, n_init=10, reg_covar=1e-5, max_iter=300,
    ).fit(geo_scaled)
    geo_labels = geo_gmm.predict(geo_scaled)
    print(f"Geometry GMM fitted with K_geo={k_geo}")

    # Build phi features
    phi = build_phi(df)
    print(f"phi shape: {phi.shape}")

    # Stage 2: per-group BIC
    all_results = {}
    for g in range(k_geo):
        mask = geo_labels == g
        n_g = mask.sum()
        sub = df.iloc[np.where(mask)[0]]
        w_range = f"W=[{sub['width'].min():.1f},{sub['width'].max():.1f}]"
        h_range = f"H=[{sub['height'].min():.1f},{sub['height'].max():.1f}]"
        print(f"\n{'='*60}")
        print(f"Geo group {g}: {n_g:,} samples, {w_range}, {h_range}")
        print(f"{'='*60}")

        phi_g = phi[mask]

        print(f"{'K_phi':>5}  {'BIC':>14}  {'AIC':>14}  {'time(s)':>9}")
        print("-" * 50)

        results = []
        for k in ks_phi:
            if k >= n_g:
                print(f"{k:5d}  {'SKIP (K>=N)':>14}")
                continue
            bic, aic, t = _fit_and_score(
                phi_g, k, cov_type, seed + g + 1, subsample,
                label=f"G{g}: ",
            )
            results.append({"K": k, "BIC": bic, "AIC": aic, "time_s": t})
            print(f"{k:5d}  {bic:14.0f}  {aic:14.0f}  {t:9.1f}")

        if results:
            _print_summary(results, n_g, label=f"K_phi (group {g})")
        all_results[g] = results

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY: Recommended K_phi per geometry group")
    print(f"{'='*60}")
    total_experts = 0
    for g in range(k_geo):
        if all_results.get(g):
            best = min(all_results[g], key=lambda x: x["BIC"])
            mask = geo_labels == g
            print(f"  Group {g} ({mask.sum():>5} samples): K_phi = {best['K']}")
            total_experts += best["K"]
        else:
            print(f"  Group {g}: no valid K_phi")
    print(f"  Total experts = {total_experts}")

    return all_results


def _print_summary(results, n_total, label="K"):
    best_bic = min(results, key=lambda x: x["BIC"])
    best_aic = min(results, key=lambda x: x["AIC"])
    print(f"\nBest {label} by BIC: {best_bic['K']}  (BIC={best_bic['BIC']:.0f})")
    print(f"Best {label} by AIC: {best_aic['K']}  (AIC={best_aic['AIC']:.0f})")
    print(f"  -> avg cluster size at {label}={best_bic['K']}: "
          f"{n_total // best_bic['K']:,} rows")


def main():
    parser = argparse.ArgumentParser(
        description="BIC/AIC screening for optimal GMM K",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--csv", required=True, help="Path to merged data CSV")
    parser.add_argument("--mode", default="phi", choices=["phi", "geo", "hier"],
                        help="BIC mode: 'phi' (single-stage 18D), "
                             "'geo' (Stage 1 geometry), "
                             "'hier' (Stage 2 per-group phi)")
    parser.add_argument("--ks", nargs="+", type=int,
                        default=[40, 60, 80, 100, 120],
                        help="Candidate K values (default: 40 60 80 100 120)")
    parser.add_argument("--k-geo", type=int, default=6,
                        help="Fixed K_geo for --mode hier (default: 6)")
    parser.add_argument("--subsample", type=int, default=20000,
                        help="Subsample size for GMM fit (default: 20000)")
    parser.add_argument("--cov-type", default="full",
                        choices=["full", "diag", "tied", "spherical"],
                        help="GMM covariance type (default: full)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "geo":
        if args.ks == [40, 60, 80, 100, 120]:
            args.ks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
        run_geo_bic(args.csv, args.ks, args.subsample, args.cov_type, args.seed)

    elif args.mode == "hier":
        if args.ks == [40, 60, 80, 100, 120]:
            args.ks = [3, 5, 8, 10, 12, 15, 20]
        run_hierarchical_bic(
            args.csv, args.k_geo, args.ks,
            args.subsample, args.cov_type, args.seed,
        )

    else:
        run_phi_bic(args.csv, args.ks, args.subsample, args.cov_type, args.seed)


if __name__ == "__main__":
    main()
