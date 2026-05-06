"""Uniform LHS fill for any gid in v10_yshape_planB.

Problem: original training data is heavily skewed toward high eta and high
sigma_y (both sampled linearly, not in log-space).  This script runs a
Latin Hypercube Sample in LOG-space over the full (n, eta, sigma_y) domain,
then routes each result through the partition to assign sub_id.

All simulations for a given gid use W, H sampled uniformly within the gid's
WH bin -- so 100%% of results stay in the correct gid (no yield loss at the
gid level).  After routing, points are spread across all subs proportionally
to KMeans cluster size and sigma_y distribution.

Output CSV has the same columns as sub_assignments.csv plus sub_id/cluster_id
so it can be merged directly into the training pool.

Usage:
    python scripts/run_uniform_lhs_gid.py --gid 0 --n-sims 10000 --arch cuda
    python scripts/run_uniform_lhs_gid.py --gid 10 --n-sims 10000 --arch cuda
    python scripts/run_uniform_lhs_gid.py --gid 0 --n-sims 10000 --arch cpu --seed 42

Output (default):
    outputs/uniform_fill/lhs_gid{GID}_uniform.csv
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
INFILL   = Path(__file__).resolve().parents[1]   # data_infill/
PIPE     = INFILL.parent                          # pipeline_v10/
SIM_CODE = INFILL / "sim_code"

if (SIM_CODE / "DataPipeline" / "headless_mls.py").is_file():
    sys.path.insert(0, str(SIM_CODE))
else:
    # fallback: look two levels up for the repo root
    sys.path.insert(0, str(PIPE.parent))

from DataPipeline.headless_mls import HeadlessSimulatorMLS  # noqa: E402

# ---------------------------------------------------------------------------
# Geo-router bin edges (GridGeoRouter from surrogate/grid_geo.py)
# gid = wi * N_H + hi   (N_H = 5)
# ---------------------------------------------------------------------------
W_EDGES = [2.0, 3.0, 4.0, 5.0, 6.015340810, 7.000001]
H_EDGES = [2.0, 3.0, 4.0, 5.022505710, 6.048921180, 7.000001]
N_H = 5

# ---------------------------------------------------------------------------
# Physical parameter ranges (same for all gids)
# ---------------------------------------------------------------------------
N_LO,  N_HI  = 0.30,  1.00         # power-law index (linear)
ETA_LO, ETA_HI = 0.72,  300.0      # consistency [Pa.s^n] (physical)
SY_LO,  SY_HI  = 0.003, 400.0      # yield stress [Pa] (physical)
# Log-space bounds:
LE_LO = np.log(ETA_LO);  LE_HI = np.log(ETA_HI)
LS_LO = np.log(SY_LO);   LS_HI = np.log(SY_HI)

CHECKPOINT = 20
HEADER = [
    "n", "eta", "sigma_y", "width", "height",
    "x_01", "x_02", "x_03", "x_04", "x_05", "x_06", "x_07", "x_08",
    "sim_seconds", "sub_id", "cluster_id",
]


# ---------------------------------------------------------------------------
# LHS sampler
# ---------------------------------------------------------------------------
def lhs(n: int, lo: np.ndarray, hi: np.ndarray,
        rng: np.random.Generator) -> np.ndarray:
    """Maximin LHS in d dimensions."""
    d = len(lo)
    p = np.argsort(rng.uniform(0, 1, (n, d)), axis=0)
    u = (p + rng.uniform(0, 1, (n, d))) / n
    return lo + (hi - lo) * u


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def route_one(x8: np.ndarray, sy: float, st: dict) -> tuple[int, int]:
    """Route one simulated result to (sub_id, cluster_id).

    Parameters
    ----------
    x8 : array (8,)  raw outputs x_01..x_08
    sy : float       sigma_y used in simulation
    st : dict        loaded state.pkl for this gid

    Returns
    -------
    sub_id, cluster_id  (-1 on failure)
    """
    w = float(st["y8_weight"])

    # 1. Normalised curve (7-dim)
    denom = x8[7] if abs(x8[7]) > 1e-12 else 1e-12
    norm_c = (x8[1:8] - x8[0]) / denom            # shape (7,)

    # 2. Scale + PCA
    nc_sc  = st["flat_scaler_nc"].transform(norm_c.reshape(1, -1))
    pca_f  = st["flat_pca"].transform(nc_sc)        # (1, 3)

    # 3. 4-D KMeans feature: [w*log(|x8|), PC1, PC2, PC3]
    log_x8 = float(np.log(max(abs(x8[7]), 1e-9)))
    feat   = np.array([[w * log_x8,
                        float(pca_f[0, 0]),
                        float(pca_f[0, 1]),
                        float(pca_f[0, 2])]])

    # 4. Nearest KMeans centroid
    centroids = np.asarray(st["flat_centroids"])    # (K, 4)
    cluster   = int(np.argmin(np.linalg.norm(centroids - feat, axis=1)))

    # 5. sigma_y threshold within cluster
    log_sy = float(np.log(1.0 + max(sy, 0.0)))
    for cd in st["flat_clusters"]:
        if cd["cluster_id"] == cluster:
            for sub in cd["subs"]:
                if sub["sy_log_lo"] <= log_sy < sub["sy_log_hi"]:
                    return int(sub["sub_id"]), cluster
            # fallback: nearest boundary
            return int(cd["subs"][-1]["sub_id"]), cluster
    return -1, cluster


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Uniform LHS fill in log-space for a given gid")
    ap.add_argument("--gid",    type=int,  default=0,
                    help="Geo-ID to fill (0-24)")
    ap.add_argument("--n-sims", type=int,  default=10_000,
                    help="Number of simulations to run")
    ap.add_argument("--arch",   type=str,  default="cuda",
                    help="Taichi backend (cuda / cpu)")
    ap.add_argument("--seed",   type=int,  default=42)
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="Output CSV path (default: outputs/uniform_fill/lhs_gid{GID}_uniform.csv)")
    ap.add_argument("--state-pkl", type=Path, default=None,
                    help="Direct path to state.pkl (overrides --bank). "
                         "Use this on worker machines: --state-pkl worker_deps/gid10/state.pkl")
    ap.add_argument("--bank",   type=Path,
                    default=PIPE.parent.parent
                              / "Fast-Non-Newtonian-ViRheometry-via-Mixture-of-GP-Surrogates"
                              / "Models" / "v10_yshape_planB",
                    help="Path to planB model bank (ignored if --state-pkl is given)")
    a = ap.parse_args()

    # -- resolve output path
    if a.out_csv is None:
        out_dir = INFILL / "outputs" / "uniform_fill"
        a.out_csv = out_dir / f"lhs_gid{a.gid}_uniform.csv"
    a.out_csv.parent.mkdir(parents=True, exist_ok=True)

    # -- compute WH range for this gid
    wi = a.gid // N_H
    hi = a.gid %  N_H
    if wi >= len(W_EDGES) - 1 or hi >= len(H_EDGES) - 1:
        ap.error(f"gid={a.gid} out of range (max {(len(W_EDGES)-1)*(len(H_EDGES)-1)-1})")
    W_LO, W_HI = W_EDGES[wi], W_EDGES[wi + 1]
    H_LO, H_HI = H_EDGES[hi], H_EDGES[hi + 1]

    # -- load partition state
    if a.state_pkl is not None:
        state_path = a.state_pkl
    else:
        state_path = a.bank / f"state_gid_{a.gid}" / "state.pkl"
        # also check local worker_deps fallback
        if not state_path.is_file():
            local_fallback = INFILL / "worker_deps" / f"gid{a.gid}" / "state.pkl"
            if local_fallback.is_file():
                state_path = local_fallback
    if not state_path.is_file():
        ap.error(f"state.pkl not found: {state_path}\n"
                 f"  On worker machine use: --state-pkl worker_deps/gid{a.gid}/state.pkl")
    with open(state_path, "rb") as fh:
        st = pickle.load(fh)

    # -- resume: count already done
    n_done = 0
    if a.out_csv.is_file():
        try:
            import pandas as pd
            n_done = len(pd.read_csv(a.out_csv))
        except Exception:
            pass
    n_todo = max(0, a.n_sims - n_done)
    if n_todo == 0:
        print(f"Already done {n_done}/{a.n_sims}. Nothing to do.")
        return

    print(f"Uniform LHS fill  gid={a.gid}  W:[{W_LO:.3f},{W_HI:.3f}]  H:[{H_LO:.3f},{H_HI:.3f}]")
    print(f"  n_sims={a.n_sims}  done={n_done}  remaining={n_todo}")
    print(f"  n:      [{N_LO}, {N_HI}]  (linear)")
    print(f"  eta:    [{ETA_LO}, {ETA_HI}]  (log-space LHS)")
    print(f"  sigma_y:[{SY_LO}, {SY_HI}]  (log-space LHS)")
    print(f"  arch={a.arch}  seed={a.seed}")

    # -- build LHS samples (log-space for eta and sigma_y)
    rng = np.random.default_rng(a.seed + n_done)
    lo  = np.array([N_LO,  LE_LO, LS_LO, W_LO, H_LO])
    hi  = np.array([N_HI,  LE_HI, LS_HI, W_HI, H_HI])
    samples_log = lhs(n_todo, lo, hi, rng)
    # convert back to physical
    samples = np.column_stack([
        samples_log[:, 0],           # n (linear)
        np.exp(samples_log[:, 1]),   # eta
        np.exp(samples_log[:, 2]),   # sigma_y
        samples_log[:, 3],           # W (linear, already in [W_LO,W_HI])
        samples_log[:, 4],           # H (linear)
    ])

    # -- initialise simulator
    print(f"\nInitialising simulator (arch={a.arch})...")
    sim = HeadlessSimulatorMLS(arch=a.arch)

    write_header = not a.out_csv.is_file()
    t0 = time.time()
    n_ok = 0; n_fail = 0
    buf: list[dict] = []
    sub_counts: dict[int, int] = {}

    with open(a.out_csv, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEADER)
        if write_header:
            writer.writeheader()

        for i, (n_, eta_, sy_, W_, H_) in enumerate(samples):
            ts = time.time()
            try:
                y8 = sim.run(float(n_), float(eta_), float(sy_), float(W_), float(H_))
                dt = time.time() - ts

                sub_id, cid = route_one(np.array(y8, dtype=float), float(sy_), st)
                sub_counts[sub_id] = sub_counts.get(sub_id, 0) + 1

                buf.append({
                    "n": n_, "eta": eta_, "sigma_y": sy_,
                    "width": W_, "height": H_,
                    **{f"x_{j+1:02d}": float(y8[j]) for j in range(8)},
                    "sim_seconds": round(dt, 3),
                    "sub_id": sub_id,
                    "cluster_id": cid,
                })
                n_ok += 1
            except Exception as exc:
                n_fail += 1
                print(f"  [FAIL i={i}: {exc}]")
                continue

            if n_ok % CHECKPOINT == 0:
                writer.writerows(buf); fh.flush(); buf = []
                elapsed = time.time() - t0
                rate    = n_ok / max(elapsed, 1e-6)
                eta_rem = (n_todo - n_ok) / max(rate, 1e-6)
                print(f"  [{n_ok}/{n_todo}]  {rate*60:.1f} sim/min  ETA {eta_rem/60:.1f} min")

        if buf:
            writer.writerows(buf)

    elapsed = time.time() - t0
    print(f"\nDone: {n_ok} simulated, {n_fail} failed  ({elapsed/60:.1f} min total)")
    print(f"Saved -> {a.out_csv}")
    if n_ok:
        print(f"Rate: {n_ok/elapsed*60:.1f} sim/min  ({elapsed/n_ok:.2f} s/sim)")

    print("\nSub-ID distribution (new points):")
    for sid in sorted(sub_counts):
        print(f"  sub_{sid:04d}: {sub_counts[sid]:5d}")

    total_subs = len(sub_counts)
    valid = {k: v for k, v in sub_counts.items() if k >= 0}
    print(f"\nTotal subs reached: {total_subs}  (routing failures: {sub_counts.get(-1, 0)})")
    print("Next step: run merge_uniform_fill.py to add these rows to the training pool,")
    print("           then retrain affected subs with train_yshape_subs.py.")


if __name__ == "__main__":
    main()
