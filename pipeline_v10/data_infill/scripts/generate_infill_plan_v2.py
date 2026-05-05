"""Generate log-uniform infill plan from train_final.csv.

Replaces generate_cell_plan.py (which read from synthetic_splits_v2/).
Key improvements:
  - Single source CSV (train_final.csv after merge+dedup+filter)
  - Finer binning: B_n=5, B_eta=5, B_sy=6, B_W=3, B_H=3 → 1350 cells
    (η and σy get more bins because they were most non-uniform)
  - FIXED edges in log space (not data-driven), so empty regions get cells too
  - Priority weight: cells with low log(eta) and low log(sy) get 2× target
    because those regions are most under-represented and hardest to cover
  - Deficit output is compatible with run_worker.py (same column names)

Usage:
  python generate_infill_plan_v2.py \\
      --src train_final.csv \\
      --Q 400 \\
      --machines 3 \\
      --machine-weights 3,1,1 \\
      --out-csv plan/cell_plan_v2.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PIPE = Path(__file__).resolve().parents[2]   # pipeline_v10/

# ── Fixed parameter space (must match PARAM_BOUNDS in engine.py) ─────────────
N_LO,   N_HI   = 0.30,  1.00
ETA_LO, ETA_HI = 0.719, 300.0   # physical
SY_LO,  SY_HI  = 0.003, 400.0   # physical (internal units)
W_LO,   W_HI   = 2.0,   7.0
H_LO,   H_HI   = 2.0,   7.0

# ── Simulation cost model (from generate_cell_plan.py) ───────────────────────
PAD = 0.5; Z = 4.15; DX = 0.126; SPD = 2
def particles(W: float, H: float) -> float:
    return ((W + PAD) * (H + PAD) * Z) / (DX ** 3) * (SPD ** 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path,
                    default=PIPE.parent / "Fast-Non-Newtonian-ViRheometry-via-Mixture-of-GP-Surrogates"
                            / "data" / "synthetic_splits_v3p2_round2_merged" / "train_final.csv",
                    help="path to train_final.csv (merged+deduped+filtered)")
    ap.add_argument("--Q", type=int, default=400,
                    help="base target sims per cell; low-eta/sy cells get 2x")
    ap.add_argument("--machines", type=int, default=3)
    ap.add_argument("--machine-weights", type=str, default=None,
                    help="e.g. '3,1,1'")
    ap.add_argument("--B-n",   type=int, default=5)
    ap.add_argument("--B-eta", type=int, default=5)
    ap.add_argument("--B-sy",  type=int, default=6)
    ap.add_argument("--B-W",   type=int, default=3)
    ap.add_argument("--B-H",   type=int, default=3)
    ap.add_argument("--out-csv", type=Path,
                    default=PIPE / "data_infill" / "plan" / "cell_plan_v2.csv")
    a = ap.parse_args()

    # ── Load existing pool ────────────────────────────────────────────────────
    df = pd.read_csv(a.src)
    print(f"Loaded existing pool: {len(df):,} rows from {a.src.name}")

    # ── Build fixed edges in log space ────────────────────────────────────────
    ln_eta_lo, ln_eta_hi = np.log(ETA_LO), np.log(ETA_HI)
    ln_sy_lo,  ln_sy_hi  = np.log(SY_LO),  np.log(SY_HI)

    edges = {
        "n":      np.linspace(N_LO, N_HI + 1e-9, a.B_n + 1),
        "ln_eta": np.linspace(ln_eta_lo, ln_eta_hi + 1e-9, a.B_eta + 1),
        "ln_sy":  np.linspace(ln_sy_lo,  ln_sy_hi  + 1e-9, a.B_sy  + 1),
        "W":      np.linspace(W_LO, W_HI + 1e-9, a.B_W + 1),
        "H":      np.linspace(H_LO, H_HI + 1e-9, a.B_H + 1),
    }
    Bs = [a.B_n, a.B_eta, a.B_sy, a.B_W, a.B_H]
    keys = ["n", "ln_eta", "ln_sy", "W", "H"]

    # ── Bin existing data ─────────────────────────────────────────────────────
    feats = np.column_stack([
        df.n.to_numpy(),
        np.log(df.eta.clip(1e-12).to_numpy()),
        np.log(df.sigma_y.clip(1e-12).to_numpy()),
        df.width.to_numpy(),
        df.height.to_numpy(),
    ])
    bin_idx = np.zeros((len(df), 5), dtype=np.int32)
    for d, k in enumerate(keys):
        raw = np.digitize(feats[:, d], edges[k][1:-1])
        bin_idx[:, d] = np.clip(raw, 0, Bs[d] - 1)

    flat = np.zeros(len(df), dtype=np.int64)
    mult = 1
    for d in range(5):
        flat += bin_idx[:, d] * mult
        mult *= Bs[d]
    total_cells = int(mult)
    counts = np.bincount(flat, minlength=total_cells)

    # ── Print 2D marginal: log10(eta) × log10(sigma_y) ───────────────────────
    print(f"\nTotal cells: {total_cells}")
    print(f"Cells with 0 data: {(counts == 0).sum()}")
    print(f"\n=== log10(eta) x log10(sigma_y) marginal density ===")
    eta_e10 = np.log10(np.exp(edges["ln_eta"]))
    sy_e10  = np.log10(np.exp(edges["ln_sy"]))

    # marginalize over n, W, H by summing
    cnt5d = counts.reshape(Bs)   # shape (B_n, B_eta, B_sy, B_W, B_H)
    cnt_es = cnt5d.sum(axis=(0, 3, 4))  # shape (B_eta, B_sy)
    hdr = "ln_sy\\eta  " + "".join(
        f"  [{eta_e10[i]:.1f},{eta_e10[i+1]:.1f})" for i in range(a.B_eta))
    print(hdr)
    print("-" * len(hdr))
    for j in range(a.B_sy):
        row = f"[{sy_e10[j]:.1f},{sy_e10[j+1]:.1f})  "
        for i in range(a.B_eta):
            row += f"  {cnt_es[i, j]:>8,}"
        print(row)

    # ── Compute targets and deficit ───────────────────────────────────────────
    # Low eta/sy threshold: below median of log space
    ln_eta_mid = 0.5 * (ln_eta_lo + ln_eta_hi)  # natural log midpoint
    ln_sy_mid  = 0.5 * (ln_sy_lo  + ln_sy_hi)   # natural log midpoint

    rows = []
    for cell in range(total_cells):
        # decode multi-index
        tmp = cell
        idx = []
        for b in Bs:
            idx.append(tmp % b)
            tmp //= b
        i_n, i_e, i_s, i_W, i_H = idx

        # Cell center in each dimension
        n_lo   = float(edges["n"][i_n]);     n_hi   = float(edges["n"][i_n + 1])
        le_lo  = float(edges["ln_eta"][i_e]); le_hi  = float(edges["ln_eta"][i_e + 1])
        ls_lo  = float(edges["ln_sy"][i_s]);  ls_hi  = float(edges["ln_sy"][i_s + 1])
        W_lo   = float(edges["W"][i_W]);      W_hi   = float(edges["W"][i_W + 1])
        H_lo   = float(edges["H"][i_H]);      H_hi   = float(edges["H"][i_H + 1])

        # Priority: cells in LOW eta or LOW sy get 2× target
        le_mid = 0.5 * (le_lo + le_hi)
        ls_mid = 0.5 * (ls_lo + ls_hi)
        low_eta = le_mid < ln_eta_mid
        low_sy  = ls_mid < ln_sy_mid
        multiplier = 1 + int(low_eta) + int(low_sy)   # 1, 2, or 3×
        target = a.Q * multiplier

        cur = int(counts[cell])
        deficit = max(0, target - cur)
        if deficit == 0:
            continue

        W_mid = 0.5 * (W_lo + W_hi)
        H_mid = 0.5 * (H_lo + H_hi)
        est_p = particles(W_mid, H_mid)
        rows.append({
            "cell_id":      cell,
            "i_n":          i_n, "i_log_eta": i_e, "i_log_sy": i_s,
            "i_W":          i_W, "i_H":        i_H,
            "n_lo":         n_lo,   "n_hi":    n_hi,
            "log_eta_lo":   le_lo,  "log_eta_hi": le_hi,   # natural log (run_worker expects this)
            "log_sy_lo":    ls_lo,  "log_sy_hi":  ls_hi,
            "W_lo":         W_lo,   "W_hi":    W_hi,
            "H_lo":         H_lo,   "H_hi":    H_hi,
            "current_n":    cur,
            "target_n":     target,
            "priority_mult": multiplier,
            "deficit":      deficit,
            "est_particles": float(est_p),
            "est_cost":     float(deficit * est_p),
        })

    plan = pd.DataFrame(rows)
    print(f"\n=== Deficit summary ===")
    print(f"Cells needing infill: {len(plan):,}  (of {total_cells})")
    print(f"Total new sims:       {plan.deficit.sum():,}")
    print(f"  priority 1× (hi-eta & hi-sy):  "
          f"{plan[plan.priority_mult==1].deficit.sum():,}")
    print(f"  priority 2× (low-eta OR low-sy): "
          f"{plan[plan.priority_mult==2].deficit.sum():,}")
    print(f"  priority 3× (low-eta AND low-sy): "
          f"{plan[plan.priority_mult==3].deficit.sum():,}")

    # ── Assign to machines ────────────────────────────────────────────────────
    if a.machine_weights:
        weights = np.array([float(x) for x in a.machine_weights.split(",")], dtype=np.float64)
        assert len(weights) == a.machines, "--machine-weights count mismatch"
    else:
        weights = np.ones(a.machines, dtype=np.float64)
    weight_share = weights / weights.sum()
    target_cost  = plan.est_cost.sum() * weight_share

    plan = plan.sort_values("est_cost", ascending=False).reset_index(drop=True)
    machine_loads  = np.zeros(a.machines, dtype=np.float64)
    machine_assign = np.zeros(len(plan), dtype=np.int32)
    for i, row in plan.iterrows():
        deficit_ratio = (target_cost - machine_loads) / np.maximum(target_cost, 1.0)
        m = int(np.argmax(deficit_ratio))
        machine_assign[i] = m
        machine_loads[m] += float(row.est_cost)
    plan["machine_id"] = machine_assign
    plan = plan.sort_values(["machine_id", "cell_id"]).reset_index(drop=True)

    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(a.out_csv, index=False)
    print(f"\nSaved: {a.out_csv}")

    total_cost = plan.est_cost.sum()
    base_p = 350_000
    print(f"\n=== Per-machine assignment ===")
    for m in range(a.machines):
        sub = plan[plan.machine_id == m]
        avg_p = sub.est_particles.mean() if len(sub) else 0
        ref_h = sub.deficit.sum() * 2.0 * (avg_p / base_p) / 3600 if len(sub) else 0
        est_h = ref_h / weights[m]
        print(f"  machine {m} (weight {weights[m]:.0f}x): "
              f"{len(sub):>4} cells  "
              f"{sub.deficit.sum():>7,} sims  "
              f"cost share {sub.est_cost.sum()/total_cost*100:.1f}%  "
              f"~{est_h:.1f}h")


if __name__ == "__main__":
    main()
