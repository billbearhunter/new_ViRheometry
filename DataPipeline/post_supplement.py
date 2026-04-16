"""
DataPipeline/post_supplement.py
================================
Post-supplement pipeline: route new sims → append to clusters → retrain → diagnose.

Usage:
  python DataPipeline/post_supplement.py \
      --workspace Optimization/moe_workspace \
      --supplement Optimization/moe_workspace/supplement_v2.csv \
      [--max-rows 4800]  # only use first N rows
"""

import argparse
import sys
import logging
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surrogate.config import INPUT_COLS, OUTPUT_COLS
from surrogate.features import build_phi

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def route_hierarchical(gate, df):
    """Route samples through hierarchical gate -> 1-indexed cluster IDs + confidence."""
    geo_gmm = gate["geo_gmm"]
    geo_scaler = gate["geo_scaler"]
    phi_gmms = gate["phi_gmms"]
    phi_scalers = gate["phi_scalers"]
    k_phi_offsets = gate["k_phi_offsets"]
    K_geo = gate["k_geo"]

    geo_feat = df[["width", "height"]].values.astype(np.float64)
    geo_labels = geo_gmm.predict(geo_scaler.transform(geo_feat))
    phi_feat = build_phi(df)

    N = len(df)
    cids = np.zeros(N, dtype=int)
    confs = np.zeros(N, dtype=float)

    for g in range(K_geo):
        mask_g = geo_labels == g
        if not np.any(mask_g):
            continue
        ps_g = phi_scalers[g].transform(phi_feat[mask_g])
        prob_g = phi_gmms[g].predict_proba(ps_g)
        local = np.argmax(prob_g, axis=1)
        cids[mask_g] = k_phi_offsets[g] + local + 1
        confs[mask_g] = prob_g[np.arange(len(local)), local]

    return cids, confs


def append_to_clusters(df_routed, workspace):
    """Append routed samples to cluster{id}_train.csv files. Returns affected cluster IDs."""
    affected = set()
    groups = df_routed.groupby("cluster_id")

    for cid, grp in groups:
        train_csv = workspace / f"cluster{int(cid)}_train.csv"
        if not train_csv.exists():
            log.warning(f"  cluster{cid}_train.csv not found, creating new")
        # Determine columns to save (match existing cluster CSV columns)
        save_cols = list(INPUT_COLS) + list(OUTPUT_COLS)
        if "cluster_conf" in grp.columns:
            save_cols.append("cluster_conf")
        existing = train_csv.exists()
        grp[save_cols].to_csv(train_csv, mode="a", header=not existing, index=False)
        affected.add(int(cid))
        log.info(f"  cluster {cid}: +{len(grp)} samples appended")

    return sorted(affected)


def retrain_experts(workspace, cluster_ids):
    """Retrain specified expert clusters using train_experts.py."""
    import subprocess
    ids_str = " ".join(str(c) for c in cluster_ids)
    cmd = (
        f"python DataPipeline/train_experts.py "
        f"--data {workspace} "
        f"--clusters {ids_str} "
        f"--force"
    )
    log.info(f"Retraining {len(cluster_ids)} experts...")
    log.info(f"  CMD: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=str(ROOT),
                            capture_output=False, text=True,
                            env={**__import__('os').environ, "PYTHONIOENCODING": "utf-8"})
    if result.returncode != 0:
        log.error(f"  Retrain failed with code {result.returncode}")
    else:
        log.info(f"  Retrain complete for {len(cluster_ids)} experts")
    return result.returncode


def run_diagnostic(workspace):
    """Run diagnose_experts and save report."""
    from DataPipeline.diagnose_and_sample import diagnose_experts
    log.info("Running post-supplement diagnostic...")
    report = diagnose_experts(workspace)
    out = workspace / "diagnostic_report_post_v2.csv"
    report.to_csv(out, index=False)
    log.info(f"Saved diagnostic -> {out}")

    ok = report[report["status"] == "ok"]
    print(f"\n=== Post Round-2 Diagnostic ===")
    print(f"Active experts: {len(ok)}")
    for thr in [0.1, 0.25, 0.5, 1.0, 3.0]:
        c = (ok["maxerr"] > thr).sum()
        print(f"  MaxErr > {thr}: {c}")
    print(f"  Mean MaxErr: {ok['maxerr'].mean():.4f}")
    print(f"  Median MaxErr: {ok['maxerr'].median():.4f}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--supplement", type=Path, required=True,
                    help="Path to supplement_v2.csv (simulation results)")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Only use first N rows from supplement CSV")
    ap.add_argument("--skip-retrain", action="store_true")
    ap.add_argument("--skip-diagnose", action="store_true")
    args = ap.parse_args()

    ws = args.workspace.resolve()
    t0 = time.time()

    # Step 1: Load supplement results
    log.info(f"Loading supplement: {args.supplement}")
    df = pd.read_csv(args.supplement)
    if args.max_rows:
        df = df.head(args.max_rows)
    log.info(f"  {len(df)} simulation results loaded")

    # Step 2: Route through hierarchical gate
    log.info("Routing through hierarchical gate...")
    gate = joblib.load(ws / "gmm_gate.joblib")
    cids, confs = route_hierarchical(gate, df)
    df["cluster_id"] = cids
    df["cluster_conf"] = confs

    # Save routed version
    routed_csv = ws / "supplement_v2_routed.csv"
    df.to_csv(routed_csv, index=False)
    log.info(f"  Routed -> {routed_csv}")

    # Routing stats
    n_clusters = df["cluster_id"].nunique()
    print(f"\n=== Routing Summary ===")
    print(f"  Total samples: {len(df)}")
    print(f"  Unique clusters hit: {n_clusters}")
    print(f"  Mean confidence: {confs.mean():.4f}")
    if "source_expert" in df.columns:
        match = (df["cluster_id"] == df["source_expert"]).sum()
        print(f"  Route-back to source: {match}/{len(df)} ({100*match/len(df):.1f}%)")

    # Step 3: Append to cluster CSVs
    log.info("Appending to cluster train CSVs...")
    affected = append_to_clusters(df, ws)
    log.info(f"  {len(affected)} clusters affected")

    # Step 4: Retrain
    if not args.skip_retrain:
        retrain_experts(ws, affected)
    else:
        log.info("Skipping retrain (--skip-retrain)")

    # Step 5: Diagnose
    if not args.skip_diagnose:
        run_diagnostic(ws)
    else:
        log.info("Skipping diagnostic (--skip-diagnose)")

    elapsed = time.time() - t0
    print(f"\n=== Pipeline complete in {elapsed/60:.1f} min ===")


if __name__ == "__main__":
    main()
