"""
DataPipeline/diagnose_and_sample.py
====================================
Active learning pipeline: diagnose sparse experts -> select high-variance
sampling points -> run headless simulations.

Three stages:
  1. Diagnose: evaluate each expert on test set routed through hierarchical gate
  2. Select:   for sparse/poor experts, use GP predictive variance to find
               the most informative candidate points (LHS in expert bounding box)
  3. Simulate: run headless MPM only for selected high-variance candidates

Usage
-----
# Diagnose only:
python diagnose_and_sample.py --workspace Optimization/moe_workspace --diagnose-only

# Select candidates for specific experts (no simulation):
python diagnose_and_sample.py --workspace Optimization/moe_workspace \
    --experts 22 221 --candidates 500 --select-per-expert 100

# Full pipeline including simulation:
python diagnose_and_sample.py --workspace Optimization/moe_workspace \
    --threshold 0.5 --candidates 500 --select-per-expert 100 \
    --simulate --output workspace/targeted_samples.csv
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surrogate.config import INPUT_COLS, OUTPUT_COLS, PARAM_BOUNDS
from surrogate.expert_io import load_expert_bundle, ExpertBundle
from surrogate.predict import predict_expert_batch, predict_expert_variance
from surrogate.features import build_phi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── Hierarchical gate routing ────────────────────────────────────────────────

def _route_hierarchical(gate, df):
    """Route samples through hierarchical gate -> 1-indexed cluster IDs."""
    geo_gmm = gate["geo_gmm"]
    geo_scaler = gate["geo_scaler"]
    phi_gmms = gate["phi_gmms"]
    phi_scalers = gate["phi_scalers"]
    k_phi_offsets = gate["k_phi_offsets"]
    K_geo = gate["k_geo"]

    geo_feat = df[["width", "height"]].values.astype(np.float64)
    geo_labels = geo_gmm.predict(geo_scaler.transform(geo_feat))
    phi_feat = build_phi(df)

    cids = np.zeros(len(df), dtype=int)
    for g in range(K_geo):
        mask_g = geo_labels == g
        if not np.any(mask_g):
            continue
        ps_g = phi_scalers[g].transform(phi_feat[mask_g])
        prob_g = phi_gmms[g].predict_proba(ps_g)
        local = np.argmax(prob_g, axis=1)
        cids[mask_g] = k_phi_offsets[g] + local + 1
    return cids


# ── Stage 1: Diagnose ───────────────────────────────────────────────────────

def diagnose_experts(workspace: Path) -> pd.DataFrame:
    """Evaluate each expert on test set routed through hierarchical gate.

    Returns DataFrame with per-expert: rmse, mae, maxerr, n_train, n_test,
    mean_variance, status.
    """
    import joblib

    gate = joblib.load(workspace / "gmm_gate.joblib")
    k = gate["k"]

    test_csv = workspace / "test_labeled.csv"
    if not test_csv.exists():
        log.error(f"test_labeled.csv not found in {workspace}")
        return pd.DataFrame()

    df_test = pd.read_csv(test_csv)
    log.info(f"Test set: {len(df_test):,} rows")

    # Route through gate
    gate_space = gate.get("gate_space", "phi")
    if gate_space == "hierarchical":
        cids = _route_hierarchical(gate, df_test)
    else:
        gmm = gate["gmm"]
        scaler = gate["scaler"]
        phi = build_phi(df_test)
        ps = scaler.transform(phi)
        cids = np.argmax(gmm.predict_proba(ps), axis=1) + 1

    rows = []
    for cid in range(1, k + 1):
        pt_path = workspace / f"expert_{cid}.pt"
        train_csv = workspace / f"cluster{cid}_train.csv"

        n_train = 0
        if train_csv.exists():
            n_train = len(pd.read_csv(train_csv))

        mask = (cids == cid)
        n_test = int(mask.sum())

        if not pt_path.exists():
            rows.append({
                "expert": cid, "n_train": n_train, "n_test": n_test,
                "mae": np.nan, "rmse": np.nan, "maxerr": np.nan,
                "mean_var": np.nan, "status": "missing",
            })
            continue

        if n_test == 0:
            rows.append({
                "expert": cid, "n_train": n_train, "n_test": 0,
                "mae": np.nan, "rmse": np.nan, "maxerr": np.nan,
                "mean_var": np.nan, "status": "no_test_samples",
            })
            continue

        X_test = df_test.loc[mask, INPUT_COLS].values
        Y_true = df_test.loc[mask, OUTPUT_COLS].values

        try:
            bundle = load_expert_bundle(str(pt_path), device="cpu")

            # Predict grouped by (W, H) since predict_expert_batch takes scalar W, H
            wh_pairs = np.unique(X_test[:, 3:5], axis=0)
            Y_pred = np.zeros_like(Y_true)
            V_pred = np.zeros_like(Y_true)

            for wh in wh_pairs:
                wh_mask = (X_test[:, 3] == wh[0]) & (X_test[:, 4] == wh[1])
                sub = X_test[wh_mask]
                p, v = predict_expert_variance(
                    bundle, sub[:, 0], sub[:, 1], sub[:, 2],
                    wh[0], wh[1], device="cpu",
                )
                Y_pred[wh_mask] = p
                V_pred[wh_mask] = v

            err = Y_pred - Y_true
            abs_e = np.abs(err)
            rows.append({
                "expert": cid, "n_train": n_train, "n_test": n_test,
                "mae": float(abs_e.mean()),
                "rmse": float(np.sqrt((err ** 2).mean())),
                "maxerr": float(abs_e.max()),
                "mean_var": float(V_pred.mean()),
                "status": "ok",
            })
        except Exception as exc:
            log.warning(f"  Expert {cid}: prediction failed - {exc}")
            rows.append({
                "expert": cid, "n_train": n_train, "n_test": n_test,
                "mae": np.nan, "rmse": np.nan, "maxerr": np.nan,
                "mean_var": np.nan, "status": f"error: {exc}",
            })

    return pd.DataFrame(rows)


# ── Stage 2: Select high-variance points via LHS + GP variance ──────────────

def _generate_lhs_in_box(box: dict, n: int, seed: int) -> np.ndarray:
    """Generate LHS candidates in an expert's bounding box.

    Returns (n, 5) array with columns [n, eta, sigma_y, width, height].
    eta and sigma_y are log-uniform; W, H are snapped to 0.5 grid.
    """
    lo = np.array([box["n"][0], box["eta"][0], box["sigma_y"][0],
                   box["width"][0], box["height"][0]])
    hi = np.array([box["n"][1], box["eta"][1], box["sigma_y"][1],
                   box["width"][1], box["height"][1]])

    sampler = qmc.LatinHypercube(d=5, seed=seed)
    u = sampler.random(n=n)

    candidates = np.zeros((n, 5))
    candidates[:, 0] = lo[0] + (hi[0] - lo[0]) * u[:, 0]
    candidates[:, 1] = np.exp(
        np.log(max(lo[1], 1e-6)) +
        (np.log(max(hi[1], 1e-6)) - np.log(max(lo[1], 1e-6))) * u[:, 1]
    )
    candidates[:, 2] = np.exp(
        np.log(max(lo[2], 1e-6)) +
        (np.log(max(hi[2], 1e-6)) - np.log(max(lo[2], 1e-6))) * u[:, 2]
    )
    candidates[:, 3] = np.clip(
        np.round((lo[3] + (hi[3] - lo[3]) * u[:, 3]) * 2) / 2, lo[3], hi[3])
    candidates[:, 4] = np.clip(
        np.round((lo[4] + (hi[4] - lo[4]) * u[:, 4]) * 2) / 2, lo[4], hi[4])
    return candidates


def select_candidates_lhs(
    workspace: Path, expert_id: int, n_samples: int = 80,
) -> pd.DataFrame:
    """Tier A: generate LHS candidates directly (no variance ranking).

    For experts with very high error whose GP is too unreliable for
    variance-based ranking.
    """
    boxes_path = workspace / "boxes.json"
    if not boxes_path.exists():
        log.warning(f"boxes.json not found in {workspace}")
        return pd.DataFrame()
    with open(boxes_path) as f:
        boxes = json.load(f)
    box = boxes.get(str(expert_id))
    if box is None:
        log.warning(f"No bounding box for expert {expert_id}")
        return pd.DataFrame()

    candidates = _generate_lhs_in_box(box, n_samples, seed=42 + expert_id)
    df = pd.DataFrame(candidates, columns=INPUT_COLS)
    df["total_variance"] = np.nan
    df["expert_id"] = expert_id
    log.info(f"  Expert {expert_id:>3}: {n_samples} LHS direct (no variance ranking)")
    return df


def select_candidates(
    workspace: Path,
    expert_id: int,
    n_candidates: int = 500,
    n_select: int = 100,
) -> pd.DataFrame:
    """Tier B/C: generate LHS candidates, rank by GP variance, select top-N."""
    boxes_path = workspace / "boxes.json"
    pt_path = workspace / f"expert_{expert_id}.pt"

    if not pt_path.exists():
        log.warning(f"Expert {expert_id} checkpoint not found")
        return pd.DataFrame()
    if not boxes_path.exists():
        log.warning(f"boxes.json not found in {workspace}")
        return pd.DataFrame()

    with open(boxes_path) as f:
        boxes = json.load(f)
    box = boxes.get(str(expert_id))
    if box is None:
        log.warning(f"No bounding box for expert {expert_id}")
        return pd.DataFrame()

    candidates = _generate_lhs_in_box(box, n_candidates, seed=42 + expert_id)

    # Load expert and compute variance per (W, H) group
    bundle = load_expert_bundle(str(pt_path), device="cpu")
    total_var = np.zeros(n_candidates)

    wh_pairs = np.unique(candidates[:, 3:5], axis=0)
    for wh in wh_pairs:
        mask = (candidates[:, 3] == wh[0]) & (candidates[:, 4] == wh[1])
        if not mask.any():
            continue
        sub = candidates[mask]
        try:
            _, V = predict_expert_variance(
                bundle, sub[:, 0], sub[:, 1], sub[:, 2],
                wh[0], wh[1], device="cpu",
            )
            total_var[mask] = V.sum(axis=1)
        except Exception as exc:
            log.warning(f"  Variance failed for expert {expert_id} W={wh[0]} H={wh[1]}: {exc}")

    top_idx = np.argsort(-total_var)[:n_select]
    selected = candidates[top_idx]

    df = pd.DataFrame(selected, columns=INPUT_COLS)
    df["total_variance"] = total_var[top_idx]
    df["expert_id"] = expert_id
    log.info(f"  Expert {expert_id:>3}: {n_select}/{n_candidates} by variance, "
             f"var=[{total_var[top_idx].min():.4f}, {total_var[top_idx].max():.4f}]")
    return df


# ── Stage 3: Run simulations ────────────────────────────────────────────────

def run_simulations(candidates: pd.DataFrame, output_csv: Path, flush_every: int = 20):
    """Run headless MPM simulation for each candidate point.

    Results are flushed to CSV every `flush_every` rows so progress
    survives interruptions.
    """
    from DataPipeline.collect_data import HeadlessSimulator

    xml_path = str(ROOT / "Simulation" / "config" / "setting.xml")
    sim = HeadlessSimulator(xml_path)
    log.info(f"Running {len(candidates)} simulations...")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    header_written = output_csv.exists()

    results_buf = []
    total_done = 0
    t0 = time.time()
    for i, row in candidates.iterrows():
        try:
            diffs = sim.run(row["n"], row["eta"], row["sigma_y"],
                            row["width"], row["height"])
            result = {c: row[c] for c in INPUT_COLS}
            for j, d in enumerate(diffs):
                result[f"x_{j+1:02d}"] = float(d)
            if "expert_id" in row:
                result["source_expert"] = int(row["expert_id"])
            results_buf.append(result)
        except Exception as exc:
            log.warning(f"  Sim failed row {i}: {exc}")
            gc.collect()

        total_done += 1

        # Incremental flush
        if len(results_buf) >= flush_every:
            _flush(results_buf, output_csv, header_written)
            header_written = True
            results_buf = []

        if total_done % flush_every == 0:
            rate = total_done / (time.time() - t0)
            eta_min = (len(candidates) - total_done) / max(rate, 1e-6) / 60
            log.info(f"  [{total_done}/{len(candidates)}] {rate:.2f}/s, "
                     f"ETA {eta_min:.0f}min")

    # Final flush
    if results_buf:
        _flush(results_buf, output_csv, header_written)

    log.info(f"Finished {total_done} sims -> {output_csv}")


def _flush(results: list, output_csv: Path, header_exists: bool):
    """Append buffered results to CSV."""
    df = pd.DataFrame(results)
    df.to_csv(output_csv, mode="a", header=not header_exists, index=False)
    log.info(f"  Flushed {len(results)} rows to {output_csv}")


# ── Main CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Active learning: diagnose sparse experts -> targeted sampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--workspace", type=str, default="Optimization/moe_workspace",
                        help="MoE workspace directory")
    parser.add_argument("--diagnose-only", action="store_true",
                        help="Only run diagnostic (no selection or simulation)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="MaxErr threshold for selecting experts to supplement")
    parser.add_argument("--experts", type=int, nargs="*", default=None,
                        help="Specific expert IDs to target (overrides threshold)")
    parser.add_argument("--candidates", type=int, default=500,
                        help="Number of LHS candidates per expert")
    parser.add_argument("--select-per-expert", type=int, default=100,
                        help="Number of high-variance points to select per expert")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV for simulation results")
    parser.add_argument("--report", type=str, default=None,
                        help="Save diagnostic report to CSV")
    parser.add_argument("--simulate", action="store_true",
                        help="Actually run simulations (default: selection only)")
    args = parser.parse_args()

    workspace = Path(args.workspace)

    # ── Stage 1: Diagnose ────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Stage 1: Diagnosing expert accuracy on test set")
    log.info("=" * 60)
    diag = diagnose_experts(workspace)

    ok = diag[diag["status"] == "ok"]
    missing = diag[diag["status"] == "missing"]
    log.info(f"\n  Total experts: {len(diag)}")
    log.info(f"  Active: {len(ok)}  Missing: {len(missing)}  "
             f"No test samples: {(diag['status'] == 'no_test_samples').sum()}")

    if not ok.empty:
        log.info(f"\n  Median MAE:    {ok['mae'].median():.4f}")
        log.info(f"  Median RMSE:   {ok['rmse'].median():.4f}")
        log.info(f"  Median MaxErr: {ok['maxerr'].median():.4f}")

    report_path = args.report or str(workspace / "diagnostic_report.csv")
    diag.to_csv(report_path, index=False, float_format="%.6f")
    log.info(f"\n  Report saved -> {report_path}")

    if args.diagnose_only:
        poor = ok[ok["maxerr"] > args.threshold].sort_values("maxerr", ascending=False)
        log.info(f"\n  Experts with MaxErr > {args.threshold}: {len(poor)}")
        cols = ["expert", "n_train", "n_test", "mae", "rmse", "maxerr", "mean_var"]
        cols = [c for c in cols if c in poor.columns]
        print("\n" + poor[cols].to_string(index=False))
        return

    # ── Stage 2: Select candidates ───────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Stage 2: Selecting high-variance candidates (LHS + GP variance)")
    log.info("=" * 60)

    if args.experts:
        target_ids = args.experts
    else:
        poor = ok[ok["maxerr"] > args.threshold].sort_values("maxerr", ascending=False)
        target_ids = poor["expert"].astype(int).tolist()

    # Also include missing experts
    for _, row in missing.iterrows():
        eid = int(row["expert"])
        if eid not in target_ids:
            target_ids.append(eid)

    log.info(f"  Targeting {len(target_ids)} experts")

    all_candidates = []
    for eid in target_ids:
        df_cand = select_candidates(
            workspace, eid,
            n_candidates=args.candidates,
            n_select=args.select_per_expert,
        )
        if not df_cand.empty:
            all_candidates.append(df_cand)

    if not all_candidates:
        log.warning("No candidates selected.")
        return

    combined = pd.concat(all_candidates, ignore_index=True)
    log.info(f"\n  Total candidates: {len(combined)} across {len(all_candidates)} experts")

    cand_path = Path(args.output or str(workspace / "targeted_candidates.csv")).with_suffix(
        ".candidates.csv"
    )
    combined.to_csv(cand_path, index=False, float_format="%.6f")
    log.info(f"  Candidate list -> {cand_path}")

    # ── Stage 3: Simulate ────────────────────────────────────────────────
    if args.simulate:
        if not args.output:
            log.error("--output is required with --simulate")
            return
        log.info("\n" + "=" * 60)
        log.info("Stage 3: Running headless simulations")
        log.info("=" * 60)
        run_simulations(combined, Path(args.output))
    else:
        log.info("\n  Simulation skipped (add --simulate to run).")


if __name__ == "__main__":
    main()
