"""
DataPipeline/soft_diagnose.py
==============================
Re-evaluate test set with SOFT-BLENDING prediction (and variance-aware).

Compares three per-test-point prediction modes:
  hard  : argmax expert only (current diagnostic)
  soft  : posterior-weighted blend  y = sum_k pi_k * f_k
  var   : variance-aware blend      y = sum_k (pi_k/sigma_k^2) * f_k / Z

Aggregates MaxErr per "hard-assigned" expert so it can be compared head-to-head
with diagnostic_report_post.csv.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surrogate.config import INPUT_COLS, OUTPUT_COLS
from surrogate.expert_io import load_expert_bundle
from surrogate.features import build_phi
from surrogate.predict import predict_expert_variance


def soft_route(gate, df, threshold=0.01, max_experts=5):
    """Return per-point list of (expert_ids, weights) and hard-argmax ids."""
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
    hard_cids = np.zeros(N, dtype=int)
    routes = []  # per-point (np.array expert_ids 1-indexed, np.array weights)

    for g in range(K_geo):
        mask_g = np.where(geo_labels == g)[0]
        if len(mask_g) == 0:
            continue
        ps_g = phi_scalers[g].transform(phi_feat[mask_g])
        prob_g = phi_gmms[g].predict_proba(ps_g)  # (n_g, K_phi_g)
        offset = k_phi_offsets[g]

        # Hard
        local_hard = np.argmax(prob_g, axis=1)
        hard_cids[mask_g] = offset + local_hard + 1

        # Soft: threshold + max_experts
        for i, row_idx in enumerate(mask_g):
            probs = prob_g[i]
            sel = np.where(probs >= threshold)[0]
            if len(sel) == 0:
                sel = np.array([int(np.argmax(probs))])
            if len(sel) > max_experts:
                top = np.argsort(-probs[sel])[:max_experts]
                sel = sel[top]
            w = probs[sel] / probs[sel].sum()
            eids = offset + sel + 1
            routes.append((eids, w))

    # routes were appended per-group-per-point — reorder to match df order
    routes_ordered = [None] * N
    idx = 0
    for g in range(K_geo):
        mask_g = np.where(geo_labels == g)[0]
        for row_idx in mask_g:
            routes_ordered[row_idx] = routes[idx]
            idx += 1
    return hard_cids, routes_ordered


def predict_for_expert(bundle, X, device="cpu"):
    """Batch-predict with (W,H) sub-grouping; returns (N,8) mean and (N,8) var."""
    N = len(X)
    Y = np.zeros((N, 8))
    V = np.zeros((N, 8))
    wh_pairs = np.unique(X[:, 3:5], axis=0)
    for wh in wh_pairs:
        m = (X[:, 3] == wh[0]) & (X[:, 4] == wh[1])
        sub = X[m]
        p, v = predict_expert_variance(
            bundle, sub[:, 0], sub[:, 1], sub[:, 2],
            wh[0], wh[1], device=device,
        )
        Y[m] = p
        V[m] = v
    return Y, V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--max-experts", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ws = args.workspace.resolve()
    gate = joblib.load(ws / "gmm_gate.joblib")
    df = pd.read_csv(ws / "test_labeled.csv")
    print(f"Test set: {len(df):,} rows")

    # Route
    hard_cids, routes = soft_route(gate, df, args.threshold, args.max_experts)

    # Collect unique experts we'll need to evaluate
    experts_needed = set()
    for eids, _ in routes:
        experts_needed.update(eids.tolist())
    print(f"Experts needed for soft eval: {len(experts_needed)}")

    # Cache predictions per expert for ALL test points (vectorized by expert)
    X_all = df[INPUT_COLS].values
    Y_all = df[OUTPUT_COLS].values
    N = len(df)

    # Instead of predicting all experts for all points (expensive), cache per-point
    # only for the experts they actually route to.
    # Build per-expert point index lists:
    eid_to_idx = {e: [] for e in experts_needed}
    for i, (eids, _) in enumerate(routes):
        for e in eids:
            eid_to_idx[int(e)].append(i)

    # Per-point predictions from its routed experts
    # pred_per_point[i] = list of (eid, y_pred_1x8, v_pred_1x8)
    pred_store = [dict() for _ in range(N)]

    for eid in sorted(experts_needed):
        pt = ws / f"expert_{eid}.pt"
        if not pt.exists():
            continue
        idx = np.array(eid_to_idx[eid], dtype=int)
        if len(idx) == 0:
            continue
        try:
            bundle = load_expert_bundle(str(pt), device="cpu")
            Y_e, V_e = predict_for_expert(bundle, X_all[idx])
            for j, i in enumerate(idx):
                pred_store[i][eid] = (Y_e[j], V_e[j])
        except Exception as exc:
            print(f"  Expert {eid}: predict failed - {exc}")
            continue

    # Now compute three predictions per test point
    Y_hard = np.zeros_like(Y_all)
    Y_soft = np.zeros_like(Y_all)
    Y_var  = np.zeros_like(Y_all)
    for i in range(N):
        eids, w = routes[i]
        ps = pred_store[i]
        # Hard: use argmax only (hard_cids)
        h = int(hard_cids[i])
        if h in ps:
            Y_hard[i] = ps[h][0]
        else:
            Y_hard[i] = np.nan
        # Soft: posterior-weighted
        y_list = []; w_list = []; v_list = []
        for e, we in zip(eids, w):
            e = int(e)
            if e in ps:
                y_list.append(ps[e][0]); w_list.append(we); v_list.append(ps[e][1])
        if len(y_list):
            ys = np.stack(y_list)
            ws_arr = np.asarray(w_list, dtype=float)
            vs = np.stack(v_list)
            ws_n = ws_arr / ws_arr.sum()
            Y_soft[i] = (ws_n[:, None] * ys).sum(axis=0)
            # Variance-aware: w_eff = pi / sigma^2, per-output dim
            inv_v = 1.0 / np.clip(vs, 1e-8, None)
            w_var = (ws_arr[:, None] * inv_v)
            w_var_n = w_var / w_var.sum(axis=0, keepdims=True)
            Y_var[i] = (w_var_n * ys).sum(axis=0)
        else:
            Y_soft[i] = np.nan
            Y_var[i] = np.nan

    # Per-expert aggregated MaxErr (group by hard-routed expert for fair comparison)
    rows = []
    for cid in np.unique(hard_cids):
        m = (hard_cids == cid)
        n_test = int(m.sum())
        if n_test == 0:
            continue
        r = {"expert": int(cid), "n_test": n_test}
        for name, Yp in [("hard", Y_hard), ("soft", Y_soft), ("var", Y_var)]:
            sub_err = np.abs(Yp[m] - Y_all[m])
            valid = ~np.isnan(sub_err).any(axis=1)
            if valid.sum() == 0:
                r[f"maxerr_{name}"] = np.nan
                r[f"mae_{name}"] = np.nan
            else:
                r[f"maxerr_{name}"] = float(sub_err[valid].max())
                r[f"mae_{name}"] = float(sub_err[valid].mean())
        rows.append(r)

    out = pd.DataFrame(rows).sort_values("expert").reset_index(drop=True)
    out_path = args.out or (ws / "soft_diagnostic_report.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Summary
    print()
    print("=== Per-expert MaxErr summary (experts that had test points) ===")
    for col in ["maxerr_hard", "maxerr_soft", "maxerr_var"]:
        s = out[col].dropna()
        print(f"  {col:<12}: mean={s.mean():.4f} median={s.median():.4f} "
              f"p95={s.quantile(0.95):.4f} max={s.max():.4f}")
    print()
    print("=== Count above thresholds ===")
    for thr in [0.1, 0.25, 0.5, 1.0, 3.0]:
        hc = (out["maxerr_hard"] > thr).sum()
        sc = (out["maxerr_soft"] > thr).sum()
        vc = (out["maxerr_var"] > thr).sum()
        print(f"  > {thr:<4}: hard={hc:3d}  soft={sc:3d}  var={vc:3d}")

    # Global test-point level MaxErr (not per-expert)
    print()
    print("=== Global per-point MaxErr across ALL test points ===")
    for name, Yp in [("hard", Y_hard), ("soft", Y_soft), ("var", Y_var)]:
        err = np.abs(Yp - Y_all)
        valid = ~np.isnan(err).any(axis=1)
        per_point_max = err[valid].max(axis=1)  # worst of 8 outputs per point
        print(f"  {name:<6}: p50={np.median(per_point_max):.4f}  "
              f"p95={np.quantile(per_point_max, 0.95):.4f}  "
              f"p99={np.quantile(per_point_max, 0.99):.4f}  "
              f"max={per_point_max.max():.4f}")


if __name__ == "__main__":
    main()
