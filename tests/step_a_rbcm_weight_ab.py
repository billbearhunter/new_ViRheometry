"""Step A — A/B test: rBCM with vs without BGM posterior weighting.

Compares the v2 model's forward predictions when the rBCM aggregation
β_k is multiplied by the BGM posterior p(k|φ,g) (new, weighted) versus
when only GP info-gain weights are used (old, unweighted, the bug).

What we test
------------
1) v3 test-set forward accuracy
   For N_SAMPLE rows of test_merged.csv:
     - y_obs   = ground-truth flow distances (used as the ROUTING signal
                 — same on both sides — so the only varying factor is
                 how β_k is computed)
     - y_pred_W  = predict(... use_phi_weights=True)
     - y_pred_NW = predict(... use_phi_weights=False)
   Report pooled RMSE/MAE and the fraction of rows where the weighted
   variant lowered RMSE.

2) 5 real materials forward error against rheometer-truth flow curves
   For each material (Tonkatsu_2, Sweet_2, Chuno_2, Tonkatsu_1, Sweet_1):
     - rheometer truth θ* (n*, η*, σy*) from HB fit
     - Use surrogate forward at θ* to get y_proxy (this is the canonical
       y_obs the v2 inverse loop would receive in production)
     - Compare y_pred_W vs y_pred_NW vs y_proxy
   Note: this measures self-consistency — does the new aggregation
   change predictions at the rheometer point? A non-zero delta says
   the BGM posterior is meaningfully redistributing β across experts;
   whether the change is BETTER requires the inverse-loop test (Step
   A.6 follow-up).

Usage
-----
    python -m tests.step_a_rbcm_weight_ab \\
        [--n-sample 500] [--top-k-phi 4]
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vi_mogp import config as HC
from vi_mogp.predict import HVIMoGPrBCMPredictor


INPUT_COLS  = ["n", "eta", "sigma_y", "width", "height"]
OUTPUT_COLS = [f"x_{i:02d}" for i in range(1, 9)]


def _eval_test_set(predictor, n_sample: int, top_k_phi: int, seed: int = 0):
    print(f"\n{'='*78}\n[1] v3 test-set A/B (n_sample={n_sample}, K={top_k_phi})\n{'='*78}")
    df = pd.read_csv(HC.MERGED_TEST_CSV)
    rng = np.random.default_rng(seed)
    if n_sample < len(df):
        idx = rng.choice(len(df), size=n_sample, replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    X = df[INPUT_COLS].to_numpy(dtype=np.float64)
    Y_true = df[OUTPUT_COLS].to_numpy(dtype=np.float64)
    W = X[:, 3]; H = X[:, 4]

    t0 = time.time()
    y_W  = predictor.predict(X, W=W, H=H, y_obs=Y_true,
                             top_k_phi=top_k_phi,
                             use_baseline=True,
                             use_phi_weights=True,
                             clear_cache=True)
    t_W = time.time() - t0
    t0 = time.time()
    y_NW = predictor.predict(X, W=W, H=H, y_obs=Y_true,
                             top_k_phi=top_k_phi,
                             use_baseline=True,
                             use_phi_weights=False,
                             clear_cache=True)
    t_NW = time.time() - t0

    err_W  = y_W  - Y_true
    err_NW = y_NW - Y_true
    rmse_W  = float(np.sqrt(np.mean(err_W  ** 2)))
    rmse_NW = float(np.sqrt(np.mean(err_NW ** 2)))
    mae_W   = float(np.mean(np.abs(err_W)))
    mae_NW  = float(np.mean(np.abs(err_NW)))
    rmse_per_W  = np.sqrt(np.mean(err_W  ** 2, axis=1))
    rmse_per_NW = np.sqrt(np.mean(err_NW ** 2, axis=1))
    frac_better_W = float(np.mean(rmse_per_W < rmse_per_NW))

    print(f"  pooled RMSE  weighted={rmse_W:.5f}   unweighted={rmse_NW:.5f} "
          f" delta={rmse_W-rmse_NW:+.5f} ({(rmse_W-rmse_NW)/rmse_NW*100:+.2f}%)")
    print(f"  pooled MAE   weighted={mae_W:.5f}    unweighted={mae_NW:.5f} "
          f" delta={mae_W-mae_NW:+.5f}")
    print(f"  fraction of rows where weighted RMSE < unweighted RMSE: "
          f"{frac_better_W:.3f}")
    print(f"  predict-time  weighted={t_W:.1f}s   unweighted={t_NW:.1f}s")
    return dict(
        n=len(df), top_k=top_k_phi,
        rmse_W=rmse_W, rmse_NW=rmse_NW,
        mae_W=mae_W, mae_NW=mae_NW,
        frac_W_better=frac_better_W,
    )


def _eval_real_materials(predictor, top_k_phi: int):
    print(f"\n{'='*78}\n[2] 5 real materials self-consistency check (K={top_k_phi})\n{'='*78}")

    # Reuse the rheometer-truth fitting logic from cluster_diagnosis.py
    sys.path.insert(0, str(REPO / "tests"))
    from cluster_diagnosis import _fit_hb, _parse_setup_xml, _find_rheo_csv, _find_rheo_csv_by_name

    materials = ["Tonkatsu_2", "Sweet_2", "Chuno_2", "Tonkatsu_1", "Sweet_1"]
    rows = []
    for m in materials:
        d = REPO / "old_result" / m
        # geometry from 2setup if present, else 1setup
        xml2 = d / "ref" / "2setup" / "settings.xml"
        xml1 = d / "ref" / "1setup" / "settings.xml"
        setup = "2setup" if xml2.exists() else "1setup"
        xml = xml2 if xml2.exists() else xml1
        if not xml.exists():
            print(f"  [skip] {m}: no settings.xml"); continue
        W, H = _parse_setup_xml(xml)

        # rheometer truth
        rheo_csv = _find_rheo_csv(d / "ref" / setup)
        if rheo_csv is None:
            rheo_csv = _find_rheo_csv_by_name(m, REPO / "FlowCurve" / "Rheo_Data")
        if rheo_csv is None:
            print(f"  [skip] {m}: no rheometer CSV"); continue
        try:
            rh = _fit_hb(rheo_csv)
        except Exception as e:
            print(f"  [skip] {m}: HB fit failed: {e}"); continue

        n_t = rh["n_rheo"]
        eta_t = rh["eta_rheo"] * 10.0    # MKS -> internal
        sy_t  = rh["sigma_y_rheo"] * 10.0
        X = np.array([[n_t, eta_t, sy_t, W, H]], dtype=np.float64)

        # First produce a stable y_obs for routing — top-1 single-expert
        # forward (matches what cluster_diagnosis does).
        y_proxy = predictor.predict(X, W=np.array([W]), H=np.array([H]),
                                    y_obs=None, top_k_phi=1,
                                    use_baseline=False,
                                    use_phi_weights=False,
                                    clear_cache=True)[0]
        # Now forward at θ* with weighted vs unweighted aggregation,
        # using y_proxy as the routing signal so both variants see the
        # same expert set (the only difference is β weighting).
        y_W = predictor.predict(X, W=np.array([W]), H=np.array([H]),
                                y_obs=y_proxy, top_k_phi=top_k_phi,
                                use_baseline=True,
                                use_phi_weights=True,
                                clear_cache=True)[0]
        y_NW = predictor.predict(X, W=np.array([W]), H=np.array([H]),
                                 y_obs=y_proxy, top_k_phi=top_k_phi,
                                 use_baseline=True,
                                 use_phi_weights=False,
                                 clear_cache=True)[0]

        delta = float(np.sqrt(np.mean((y_W - y_NW) ** 2)))
        rmse_W_to_proxy  = float(np.sqrt(np.mean((y_W  - y_proxy) ** 2)))
        rmse_NW_to_proxy = float(np.sqrt(np.mean((y_NW - y_proxy) ** 2)))
        print(f"\n  {m}:  W={W} H={H}  eta*={rh['eta_rheo']:.3f} Pa.s  "
              f"n*={n_t:.3f}  sy*={rh['sigma_y_rheo']:.3f} Pa")
        print(f"    y_proxy   = {np.array2string(y_proxy, precision=3)}")
        print(f"    y_W (weighted)   = {np.array2string(y_W,  precision=3)}")
        print(f"    y_NW (unweighted)= {np.array2string(y_NW, precision=3)}")
        print(f"    RMSE(W->proxy)={rmse_W_to_proxy:.4f}  "
              f"RMSE(NW->proxy)={rmse_NW_to_proxy:.4f}  "
              f"||W-NW||={delta:.4f}")

        rows.append(dict(material=m, W=W, H=H,
                         eta_t=eta_t, n_t=n_t, sy_t=sy_t,
                         rmse_W_proxy=rmse_W_to_proxy,
                         rmse_NW_proxy=rmse_NW_to_proxy,
                         delta_WtoNW=delta))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sample", type=int, default=500)
    ap.add_argument("--top-k-phi", type=int, default=HC.INFER_TOP_K_PHI)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[load] {HC.DEFAULT_V2_MODEL}")
    predictor = HVIMoGPrBCMPredictor.load(HC.DEFAULT_V2_MODEL)

    test_summary = _eval_test_set(predictor, args.n_sample,
                                  args.top_k_phi, args.seed)
    real_rows = _eval_real_materials(predictor, args.top_k_phi)

    out_dir = REPO / "tests" / "step_a_results"
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame([test_summary]).to_csv(out_dir / "v3_testset_AB.csv",
                                        index=False, float_format="%.6f")
    pd.DataFrame(real_rows).to_csv(out_dir / "real_materials_AB.csv",
                                   index=False, float_format="%.6f")
    print(f"\n[saved] {out_dir/'v3_testset_AB.csv'}")
    print(f"[saved] {out_dir/'real_materials_AB.csv'}")


if __name__ == "__main__":
    main()
