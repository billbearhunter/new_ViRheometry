"""
DataPipeline/supplement_v2.py
=============================
Round-2 LHS-only supplementation for experts with post MaxErr > 0.25.

Allocation (by severity, pure LHS, no variance ranking):
  Critical  (> 3.0)      : 200 samples / expert
  High      (1.0 - 3.0)  : 150 samples / expert
  Mid       (0.5 - 1.0)  : 100 samples / expert
  Low-Mid   (0.25 - 0.5) :  60 samples / expert

Usage:
  python DataPipeline/supplement_v2.py \
      --workspace Optimization/moe_workspace \
      --out-candidates Optimization/moe_workspace/supplement_v2.candidates.csv \
      --out-results    Optimization/moe_workspace/supplement_v2.csv \
      [--candidates-only]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DataPipeline.diagnose_and_sample import (
    select_candidates_lhs,
    run_simulations,
    log,
)


TIERS = [
    ("Critical", 3.0, float("inf"), 200),
    ("High",     1.0, 3.0,          150),
    ("Mid",      0.5, 1.0,          100),
    ("LowMid",   0.25, 0.5,          60),
]


def allocate(high_csv: Path) -> pd.DataFrame:
    """Read high_maxerr_experts.csv, assign n_samples per severity."""
    df = pd.read_csv(high_csv)
    df = df[df["maxerr"] > 0.25].copy()

    def n_for(err: float) -> int:
        for _, lo, hi, n in TIERS:
            if lo <= err < hi:
                return n
        return 0

    def bucket(err: float) -> str:
        for name, lo, hi, _ in TIERS:
            if lo <= err < hi:
                return name
        return "-"

    df["n_samples"] = df["maxerr"].apply(n_for)
    df["bucket"] = df["maxerr"].apply(bucket)
    df = df.sort_values("maxerr", ascending=False).reset_index(drop=True)

    print("=== Allocation summary ===")
    for name, _, _, n in TIERS:
        sub = df[df["bucket"] == name]
        if len(sub):
            print(f"  {name:<9}: {len(sub):>3} experts x {n} = {len(sub)*n} sims")
    print(f"  {'Total':<9}: {len(df):>3} experts       = {df['n_samples'].sum()} sims")
    print()
    return df


def generate_candidates(plan: pd.DataFrame, workspace: Path) -> pd.DataFrame:
    """Generate LHS candidates for all experts in plan."""
    all_cands = []
    for _, row in plan.iterrows():
        eid = int(row["expert"])
        n = int(row["n_samples"])
        cands = select_candidates_lhs(workspace, eid, n_samples=n)
        if not cands.empty:
            all_cands.append(cands)
    if not all_cands:
        return pd.DataFrame()
    return pd.concat(all_cands, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--high-csv", type=Path, default=None,
                    help="Path to high_maxerr_experts.csv (default: workspace/high_maxerr_experts.csv)")
    ap.add_argument("--out-candidates", type=Path, required=True)
    ap.add_argument("--out-results", type=Path, required=True)
    ap.add_argument("--candidates-only", action="store_true",
                    help="Only generate candidates, do not simulate")
    ap.add_argument("--flush-every", type=int, default=20)
    args = ap.parse_args()

    workspace = args.workspace.resolve()
    high_csv = args.high_csv or (workspace / "high_maxerr_experts.csv")

    # Step 1: allocate
    plan = allocate(high_csv)
    plan.to_csv(workspace / "supplement_v2.plan.csv", index=False)
    log.info(f"Saved plan: {workspace / 'supplement_v2.plan.csv'}")

    # Step 2: generate candidates
    log.info("Generating LHS candidates...")
    cands = generate_candidates(plan, workspace)
    if cands.empty:
        log.error("No candidates generated")
        return
    args.out_candidates.parent.mkdir(parents=True, exist_ok=True)
    cands.to_csv(args.out_candidates, index=False)
    log.info(f"Saved {len(cands)} candidates -> {args.out_candidates}")

    if args.candidates_only:
        log.info("Candidates-only mode, stopping.")
        return

    # Step 3: simulate
    log.info(f"Running {len(cands)} MPM simulations (flush every {args.flush_every})...")
    run_simulations(cands, args.out_results, flush_every=args.flush_every)


if __name__ == "__main__":
    main()
