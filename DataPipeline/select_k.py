"""
DataPipeline/select_k.py
========================
BIC / AIC screening to choose the optimal number of GMM components (K)
for the MoE gating network.

Uses a subsample of the φ feature space (default 20 000) for speed,
since BIC rankings are stable on subsets.

Usage
-----
# Default: test K in {40,60,80,100,120} on 20K subsample, full covariance
python select_k.py --csv Optimization/moe_workspace6_data.csv

# Custom K range and subsample size
python select_k.py --csv data.csv --ks 30 40 50 60 70 80 --subsample 30000

# Diagonal covariance (faster, less accurate)
python select_k.py --csv data.csv --cov-type diag
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
sys.path.insert(0, str(ROOT / "DataPipeline"))

from moe_utils import build_phi


def run_bic_screening(csv_path: str, ks: list[int], subsample: int,
                      cov_type: str, seed: int):
    """Run BIC/AIC screening over candidate K values."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows from {csv_path}")

    # Build phi features
    phi = build_phi(df)
    print(f"phi shape: {phi.shape}")

    # Subsample for speed
    rng = np.random.RandomState(seed)
    if len(phi) > subsample:
        idx = rng.choice(len(phi), subsample, replace=False)
        phi_sub = phi[idx]
        print(f"Subsampled {subsample:,} / {len(phi):,} for GMM fit")
    else:
        phi_sub = phi

    scaler = StandardScaler().fit(phi_sub)
    phi_scaled = scaler.transform(phi_sub)

    # BIC screening
    print(f"\nTesting K = {ks}, cov_type={cov_type}")
    print(f"{'K':>5s}  {'BIC':>14s}  {'AIC':>14s}  {'time (s)':>9s}")
    print("-" * 50)

    results = []
    for k in ks:
        t0 = time.time()
        gmm = GaussianMixture(
            n_components=k, covariance_type=cov_type,
            random_state=seed, n_init=5, reg_covar=1e-5, max_iter=300,
        )
        gmm.fit(phi_scaled)
        bic = gmm.bic(phi_scaled)
        aic = gmm.aic(phi_scaled)
        elapsed = time.time() - t0
        results.append({"K": k, "BIC": bic, "AIC": aic, "time_s": elapsed})
        print(f"{k:5d}  {bic:14.0f}  {aic:14.0f}  {elapsed:9.1f}")

    # Summary
    best_bic = min(results, key=lambda x: x["BIC"])
    best_aic = min(results, key=lambda x: x["AIC"])
    print(f"\nBest K by BIC: {best_bic['K']}  (BIC={best_bic['BIC']:.0f})")
    print(f"Best K by AIC: {best_aic['K']}  (AIC={best_aic['AIC']:.0f})")
    print(f"  -> avg cluster size at K={best_bic['K']}: "
          f"{len(df) // best_bic['K']:,} rows")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="BIC/AIC screening for optimal GMM K")
    parser.add_argument("--csv", required=True, help="Path to merged data CSV")
    parser.add_argument("--ks", nargs="+", type=int,
                        default=[40, 60, 80, 100, 120],
                        help="Candidate K values (default: 40 60 80 100 120)")
    parser.add_argument("--subsample", type=int, default=20000,
                        help="Subsample size for GMM fit (default: 20000)")
    parser.add_argument("--cov-type", default="full",
                        choices=["full", "diag", "tied", "spherical"],
                        help="GMM covariance type (default: full)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_bic_screening(args.csv, args.ks, args.subsample, args.cov_type,
                      args.seed)


if __name__ == "__main__":
    main()
