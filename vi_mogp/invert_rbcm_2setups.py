"""Joint N-setup inverse recovery of (n, eta, sigma_y) for HVIMoGP_rBCM.

Two-stage protocol (mirrors Optimization/optimize_2setups.py, ported to
the current hierarchical rBCM stack):

  Stage 1: single-setup CMA-ES on setup-1 alone  →  theta_1
  Stage 2: JOINT CMA-ES on all selected setups with
             - x0   = theta_1
             - sigma0_stage2 < sigma0_stage1   (tighter search)
             - soft log-space prior anchor at theta_1 (LAMBDA_REG = 0.05)
             - loss = mean_s NMSE(y_pred_s, y_obs_s) + prior + barrier

Each setup has its own (W_s, H_s), is routed through the hierarchical gate
independently (different experts per setup), and contributes an NMSE term
against its own y_obs. θ = (n, η, σy) is shared across all setups (same
material).

CMA bounds are the HARD UNION of tightened boxes across all setups
(slightly padded) — same pattern as the MoE-240 version.

Data source (real experiments)
------------------------------
  --data-dirs  data/ref_<material>_<W>_<H>[_idx]  data/ref_...  ...

Each folder must contain flow_distances.csv with 8 rows; folder-name
convention is ref_<material>_<W>_<H>[_idx] (first number = W, second = H,
per vi_mogp.invert_real.parse_wh).

Usage
-----
    # 2-setup joint on Tonkatsu2 3.0x2.5 + 5.0x2.0, rbcm_v1 frozen model:
    python -m vi_mogp.invert_rbcm_2setups \\
        --model Models/rbcm_v1/model.pt \\
        --data-dirs data/ref_Tonkatsu2_3.0_2.5 \\
                    data/ref_Tonkatsu2_5.0_2.0 \\
        --out-dir Optimization/inverse_comparison_20260421/rbcm_v1_2setup_tonkatsu2

    # Synthetic bench from test_merged.csv (stage 1 row ↔ stage 2 row from
    # the SAME substance index — pairs rows with matching (n, eta, sigma_y)
    # but differing (W, H)):
    python -m vi_mogp.invert_rbcm_2setups \\
        --model Models/rbcm_v1/model.pt \\
        --synthetic-test Optimization/moe_workspace_merged_v3_20260419/test_merged.csv \\
        --n-samples 10 --seed 0 \\
        --out Optimization/inverse_comparison_20260421/rbcm_v1_2setup_synth.csv
"""
from __future__ import annotations
import argparse, math, re, time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from vi_mogp.predict import HVIMoGPrBCMPredictor
from vi_mogp import config as HC
from Optimization.libs.cmaes_core import run_cmaes, PARAM_BOUNDS
from vi_mogp.invert_rbcm import invert_row as invert_row_single


OUTPUT_COLS = ["x_01", "x_02", "x_03", "x_04",
               "x_05", "x_06", "x_07", "x_08"]

# Folder-name convention (shared with vi_mogp/invert_real.py).
# ref_<material>_<W>_<H>[_idx]
_WH_RE = re.compile(r"ref_[A-Za-z0-9]+?_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)(?:_\d+)?$")


# ────────────────────────────────────────────────────────────────────────────
# Setup-loading helpers
# ────────────────────────────────────────────────────────────────────────────
def parse_wh_from_folder(data_dir: Path) -> Tuple[float, float]:
    """Parse (W, H) from `ref_<material>_<W>_<H>[_idx]`."""
    m = _WH_RE.match(data_dir.name)
    if not m:
        raise ValueError(f"Cannot parse W/H from folder name: {data_dir.name}")
    return float(m.group(1)), float(m.group(2))


def read_y_obs_real(data_dir: Path) -> np.ndarray:
    """Read 8 cumulative flow-distance observations from flow_distances.csv."""
    csv_path = data_dir / "flow_distances.csv"
    df = pd.read_csv(csv_path)
    y = df["distance_cm"].values.astype(np.float64)
    if y.shape[0] != 8:
        raise ValueError(f"{csv_path}: expected 8 frames, got {y.shape[0]}")
    return y


def build_real_setups(data_dirs: Sequence[Path]) -> List[dict]:
    """From a list of data folders, produce a list of setup dicts::

        {"name", "W", "H", "y_obs"}
    """
    setups = []
    for d in data_dirs:
        d = Path(d)
        W, H = parse_wh_from_folder(d)
        y = read_y_obs_real(d)
        setups.append({"name": d.name, "W": float(W), "H": float(H),
                       "y_obs": y, "source": str(d)})
    return setups


def build_synthetic_setup_pairs(df: pd.DataFrame,
                                n_pairs: int,
                                seed: int) -> List[List[dict]]:
    """For synthetic validation: pair up rows from test_merged.csv that
    share the same ground-truth (n, eta, sigma_y) but have different
    (W, H). Each returned inner-list is a list of 2 setup dicts plus the
    ground-truth θ attached as extra keys.

    Pairs are built by grouping on (n, eta, sigma_y); we keep only groups
    with ≥ 2 distinct (W, H) rows, randomly sample n_pairs of them, and
    pick 2 rows per group.
    """
    rng = np.random.default_rng(seed)
    key = ["n", "eta", "sigma_y"]
    # group by the ground-truth triple
    grouped = df.groupby(key)
    good_keys = []
    for k, sub in grouped:
        if sub[["width", "height"]].drop_duplicates().shape[0] >= 2:
            good_keys.append(k)
    if len(good_keys) == 0:
        raise RuntimeError("No rows in synthetic test CSV share (n,eta,sigma_y) "
                           "with ≥ 2 distinct (W, H).")
    take = min(n_pairs, len(good_keys))
    idx = rng.choice(len(good_keys), size=take, replace=False)
    out = []
    for i in idx:
        k = good_keys[int(i)]
        sub = grouped.get_group(k)
        # pick 2 distinct (W,H) rows
        distinct = sub.drop_duplicates(subset=["width", "height"])
        rows = distinct.sample(n=2, random_state=int(rng.integers(0, 2**31 - 1)))
        pair = []
        for _, r in rows.iterrows():
            pair.append({
                "name":    f"synth_n{r['n']:.3f}_e{r['eta']:.2f}_s{r['sigma_y']:.1f}_W{r['width']}_H{r['height']}",
                "W":       float(r["width"]),
                "H":       float(r["height"]),
                "y_obs":   np.array([r[c] for c in OUTPUT_COLS], dtype=np.float64),
                "source":  "synthetic_test_merged",
                "n_true":  float(r["n"]),
                "eta_true": float(r["eta"]),
                "sigma_true": float(r["sigma_y"]),
            })
        out.append(pair)
    return out


# ────────────────────────────────────────────────────────────────────────────
# Bounds union across setups
# ────────────────────────────────────────────────────────────────────────────
def union_tight_bounds(per_setup_bounds: List[dict],
                       base_bounds: dict) -> dict:
    """Take the hard UNION of per-setup tightened boxes (min-of-lows,
    max-of-highs per dim), then intersect with base_bounds.

    Falls back to base_bounds if the union degenerates.
    """
    if not per_setup_bounds:
        return base_bounds
    keys = ("n", "eta", "sigma_y")
    lo = {k: min(b[k][0] for b in per_setup_bounds) for k in keys}
    hi = {k: max(b[k][1] for b in per_setup_bounds) for k in keys}
    out = {k: (max(lo[k], base_bounds[k][0]),
               min(hi[k], base_bounds[k][1])) for k in keys}
    for k, (a, b) in out.items():
        if b <= a:
            return base_bounds
    return out


# ────────────────────────────────────────────────────────────────────────────
# Joint loss
# ────────────────────────────────────────────────────────────────────────────
def make_joint_loss(predictor: HVIMoGPrBCMPredictor,
                    routed: List[dict],
                    bounds: dict,
                    theta_anchor: Optional[np.ndarray],
                    lambda_reg: float = 0.05,
                    barrier_wt: float = 1e-3,
                    use_baseline: bool = True) -> Callable:
    """Build a batched CMA loss that evaluates across all setups.

    Parameters
    ----------
    routed : list of dicts each with keys {"eids","geo_id","y_obs","W","H","norm"}
        Precomputed route for each setup (W,H,y_obs are fixed across CMA).
    theta_anchor : (3,) array or None
        Log-space soft-prior anchor (typically stage-1 θ̂). None = disable
        the prior term (stage 1 uses lambda_reg=0 instead).
    lambda_reg : float
        Weight of log-space prior. 0.05 matches optimize_2setups.py default
        when a prior is loaded.
    barrier_wt : float
        Weight of the log-barrier keeping (n, log η, log σy) away from the
        CMA box walls.
    """
    MIN_N, MAX_N     = float(bounds["n"][0]),       float(bounds["n"][1])
    MIN_ETA, MAX_ETA = float(bounds["eta"][0]),     float(bounds["eta"][1])
    MIN_SY, MAX_SY   = float(bounds["sigma_y"][0]), float(bounds["sigma_y"][1])

    scale_n       = max(MAX_N - MIN_N, 1e-9)
    scale_log_eta = max(math.log(MAX_ETA) - math.log(max(MIN_ETA, 1e-9)), 1e-9)
    scale_log_sy  = max(math.log(MAX_SY)  - math.log(max(MIN_SY,  1e-9)), 1e-9)

    use_prior = (theta_anchor is not None) and (lambda_reg > 0.0)
    if use_prior:
        a_n  = float(theta_anchor[0])
        a_le = math.log(max(float(theta_anchor[1]), 1e-9))
        a_ls = math.log(max(float(theta_anchor[2]), 1e-9))

    S = len(routed)

    def batch_loss(thetas):
        thetas = np.asarray(thetas, dtype=np.float64)   # (B, 3)
        B = thetas.shape[0]
        n  = thetas[:, 0]
        eta = np.maximum(thetas[:, 1], 1e-9)
        sy  = np.maximum(thetas[:, 2], 1e-9)

        # NMSE averaged across setups
        total_nmse = np.zeros(B, dtype=np.float64)
        for s in routed:
            X = np.zeros((B, 5), dtype=np.float64)
            X[:, 0:3] = thetas
            X[:, 3]   = s["W"]
            X[:, 4]   = s["H"]
            y_pred = predictor.predict_fixed_route(
                X, s["eids"], s["geo_id"], clear_cache=False,
                use_baseline=use_baseline,
                phi_weights_row=s.get("wts"),
            )
            total_nmse += (np.mean((y_pred - s["y_obs"][None, :]) ** 2, axis=1)
                           / s["norm"])
        mean_nmse = total_nmse / S

        loss = mean_nmse

        # Log-space prior anchor at theta_anchor (stage 2 only)
        if use_prior:
            d_n  = ((n  - a_n)           / scale_n)       ** 2
            d_le = ((np.log(eta) - a_le) / scale_log_eta) ** 2
            d_ls = ((np.log(sy)  - a_ls) / scale_log_sy)  ** 2
            loss = loss + lambda_reg * (d_n + d_le + d_ls)

        # Log-barrier keeping (n, log η, log σy) inside [lo, hi]
        def _bar(u):
            u = np.clip(u, 1e-9, 1.0 - 1e-9)
            return -np.log(u) - np.log(1.0 - u)
        u_n  = (n            - MIN_N) / scale_n
        u_le = (np.log(eta)  - math.log(max(MIN_ETA, 1e-9))) / scale_log_eta
        u_ls = (np.log(sy)   - math.log(max(MIN_SY,  1e-9))) / scale_log_sy
        loss = loss + barrier_wt * (_bar(u_n) + _bar(u_le) + _bar(u_ls))

        return loss.tolist()

    return batch_loss


# ────────────────────────────────────────────────────────────────────────────
# Two-stage inverse for one "sample" (group of ≥1 setups sharing θ)
# ────────────────────────────────────────────────────────────────────────────
def two_stage_inverse(predictor: HVIMoGPrBCMPredictor,
                      setups: List[dict],
                      *,
                      popsize: int = 12,
                      maxiter_stage1: int = 25,
                      maxiter_stage2: int = 25,
                      sigma0_stage1: float = 0.25,
                      sigma0_stage2: float = 0.10,
                      seed: int = 0,
                      top_k_phi: Optional[int] = None,
                      warm_start_stage1: bool = True,
                      pad_frac: float = 0.02,
                      use_baseline: bool = True,
                      sigma_prior_weight_stage1: float = 1e-5,
                      sigma_prior_width_stage1: float = 1.0,
                      lambda_reg_stage2: float = 0.05,
                      barrier_wt_stage2: float = 1e-3,
                      verb: int = 0):
    """Return dict with both stages' results, timings, and routing info."""
    assert len(setups) >= 1, "need at least 1 setup"

    # ──  STAGE 1  ── single-setup CMA on setup[0] only (warm-start via NN)
    s0 = setups[0]
    t0 = time.time()
    theta_1, loss_1, dt_1, y_pred_1, _ = invert_row_single(
        predictor, s0["y_obs"], s0["W"], s0["H"],
        popsize=popsize, maxiter=maxiter_stage1,
        sigma0=sigma0_stage1, seed=seed, verb=verb,
        tighten_bounds=True, pad_frac=pad_frac,
        top_k_phi=top_k_phi,
        warm_start=warm_start_stage1,
        use_baseline=use_baseline,
        sigma_prior_weight=sigma_prior_weight_stage1,
        sigma_prior_width=sigma_prior_width_stage1,
    )
    theta_1 = np.asarray(theta_1, dtype=np.float64)

    # ── STAGE 2 prep: route every setup, build union bounds, pre-cache y_obs
    top_k = top_k_phi if top_k_phi is not None else HC.INFER_TOP_K_PHI
    routed: List[dict] = []
    per_setup_bounds: List[dict] = []
    for s in setups:
        geo_id, eids, wts = predictor.route_for(s["W"], s["H"], s["y_obs"],
                                                top_k_phi=top_k)
        tb = predictor.tightened_bounds(eids, geo_id, PARAM_BOUNDS,
                                        include_baseline=True,
                                        pad_frac=pad_frac)
        per_setup_bounds.append(tb)
        y_obs = np.asarray(s["y_obs"], dtype=np.float64)
        routed.append({
            "eids":   eids,
            "geo_id": int(geo_id),
            "wts":    wts,                # Step-A: BGM posterior for rBCM weighting
            "y_obs":  y_obs,
            "W":      float(s["W"]),
            "H":      float(s["H"]),
            "norm":   max(float(np.mean(y_obs ** 2)), 1e-12),
        })

    bounds = union_tight_bounds(per_setup_bounds, PARAM_BOUNDS)

    # clip stage-1 θ into the union bounds so CMA x0 is well-defined
    theta_anchor = np.array([
        float(np.clip(theta_1[0], bounds["n"][0],       bounds["n"][1])),
        float(np.clip(theta_1[1], bounds["eta"][0],     bounds["eta"][1])),
        float(np.clip(theta_1[2], bounds["sigma_y"][0], bounds["sigma_y"][1])),
    ], dtype=np.float64)

    # ── STAGE 2: joint CMA anchored at θ_1
    loss_fn2 = make_joint_loss(
        predictor, routed, bounds,
        theta_anchor=theta_anchor,
        lambda_reg=lambda_reg_stage2,
        barrier_wt=barrier_wt_stage2,
        use_baseline=use_baseline,
    )

    t2 = time.time()
    theta_2, loss_2, hist2 = run_cmaes(
        loss_fn2, bounds,
        x0=list(theta_anchor),
        sigma0=sigma0_stage2,
        popsize=popsize, maxiter=maxiter_stage2,
        seed=seed, verb_disp=verb,
    )
    dt_2 = time.time() - t2
    theta_2 = np.asarray(theta_2, dtype=np.float64)

    # Forward readouts with the stage-2 θ on every setup
    fwd = []
    for s in routed:
        X = np.array([[*theta_2, s["W"], s["H"]]], dtype=np.float64)
        y_p = predictor.predict_fixed_route(X, s["eids"], s["geo_id"],
                                            clear_cache=False,
                                            use_baseline=use_baseline,
                                            phi_weights_row=s.get("wts"))[0]
        fwd.append({
            "y_pred":       y_p,
            "forward_nmse": float(np.mean((y_p - s["y_obs"]) ** 2) / s["norm"]),
            "geo_id":       s["geo_id"],
            "eids":         s["eids"].tolist() if hasattr(s["eids"], "tolist")
                            else list(s["eids"]),
        })

    # Flush caches
    for exp in predictor.model.experts:
        exp.clear_prediction_cache()
    for b in predictor.model.baselines:
        b.clear_prediction_cache()

    return {
        "theta_stage1":  theta_1,
        "loss_stage1":   float(loss_1),
        "dt_stage1":     float(dt_1),
        "y_pred_stage1": y_pred_1,      # only for setup[0]

        "theta_stage2":  theta_2,
        "loss_stage2":   float(loss_2),
        "dt_stage2":     float(dt_2),
        "hist_stage2":   list(hist2),
        "bounds_stage2": bounds,
        "fwd":           fwd,            # len == len(setups)
        "total_dt":      float(time.time() - t0),
    }


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-dirs", nargs="+",
                     help="Real-experiment folders (>=2) following the "
                          "ref_<material>_<W>_<H>[_idx] convention. Each "
                          "must contain flow_distances.csv. All folders "
                          "are treated as setups of ONE joint sample.")
    src.add_argument("--synthetic-test",
                     help="test_merged.csv -- pair rows that share "
                          "(n,eta,sigma_y) but differ in (W,H); run N "
                          "such pairs as synthetic 2-setup benches.")

    ap.add_argument("--n-samples", type=int, default=10,
                    help="Only used with --synthetic-test.")
    ap.add_argument("--popsize",   type=int, default=12)
    ap.add_argument("--maxiter-stage1", type=int, default=25)
    ap.add_argument("--maxiter-stage2", type=int, default=25)
    ap.add_argument("--sigma0-stage1",  type=float, default=0.25)
    ap.add_argument("--sigma0-stage2",  type=float, default=0.10)
    ap.add_argument("--seed",      type=int, default=0)
    ap.add_argument("--top-k-phi", type=int, default=None)
    ap.add_argument("--pad-frac",  type=float, default=0.02)
    ap.add_argument("--no-warm-start-stage1", action="store_true")
    ap.add_argument("--no-baseline", action="store_true",
                    help="Use rBCM (no geo-baseline) instead of GRBCM.")
    ap.add_argument("--sigma-prior-weight-stage1", type=float, default=1e-5,
                    help="Soft sigma_y log-prior for stage 1 (mirrors "
                         "invert_rbcm.py default).")
    ap.add_argument("--lambda-reg-stage2", type=float, default=0.05,
                    help="Log-space prior weight for stage 2, anchored at "
                         "theta_hat_stage1 (mirrors MoE-240 optimize_2setups "
                         "default when a prior is loaded).")
    ap.add_argument("--barrier-wt-stage2", type=float, default=1e-3)
    ap.add_argument("--out",       type=str, default=None,
                    help="Per-sample CSV path (all setups flattened).")
    ap.add_argument("--out-dir",   type=str, default=None,
                    help="If given, also write setup1/2 txt files and "
                         "joint summary JSON in this folder.")
    ap.add_argument("--verb",      type=int, default=0)
    args = ap.parse_args()

    print(f"[invert-2s] model: {args.model}")
    predictor = HVIMoGPrBCMPredictor.load(args.model)

    # ── Assemble the list of "samples", each being a list of setups ─────
    if args.data_dirs:
        if len(args.data_dirs) < 2:
            print("[warn] only 1 --data-dir given; will degenerate to a "
                  "single-setup inversion (stage 2 == stage 1 objective).")
        setups = build_real_setups([Path(p) for p in args.data_dirs])
        samples = [setups]    # ONE joint sample spanning N setups
        print(f"[invert-2s] REAL data: {len(setups)} setups "
              f"({', '.join(s['name'] for s in setups)})")
    else:
        df = pd.read_csv(args.synthetic_test)
        samples = build_synthetic_setup_pairs(df, args.n_samples, args.seed)
        print(f"[invert-2s] SYNTHETIC: {len(samples)} pairs "
              f"(2 setups each) from {args.synthetic_test}")

    # ── Run ──────────────────────────────────────────────────────────────
    all_rows = []
    for si, setups in enumerate(samples):
        print(f"\n=== sample {si+1}/{len(samples)} "
              f"({len(setups)} setups) ===")
        for k, s in enumerate(setups):
            print(f"    setup[{k}] name={s['name']} "
                  f"W={s['W']} H={s['H']}")

        res = two_stage_inverse(
            predictor, setups,
            popsize=args.popsize,
            maxiter_stage1=args.maxiter_stage1,
            maxiter_stage2=args.maxiter_stage2,
            sigma0_stage1=args.sigma0_stage1,
            sigma0_stage2=args.sigma0_stage2,
            seed=args.seed,
            top_k_phi=args.top_k_phi,
            warm_start_stage1=(not args.no_warm_start_stage1),
            pad_frac=args.pad_frac,
            use_baseline=(not args.no_baseline),
            sigma_prior_weight_stage1=args.sigma_prior_weight_stage1,
            lambda_reg_stage2=args.lambda_reg_stage2,
            barrier_wt_stage2=args.barrier_wt_stage2,
            verb=args.verb,
        )
        n1, e1, s1_sy = res["theta_stage1"]
        n2, e2, s2_sy = res["theta_stage2"]
        print(f"  stage1  n={n1:.3f}  eta={e1:9.3f}  sy={s1_sy:9.3f}  "
              f"loss={res['loss_stage1']:.3e}  dt={res['dt_stage1']:.1f}s")
        print(f"  stage2  n={n2:.3f}  eta={e2:9.3f}  sy={s2_sy:9.3f}  "
              f"loss={res['loss_stage2']:.3e}  dt={res['dt_stage2']:.1f}s")
        for k, (s, fwd) in enumerate(zip(setups, res["fwd"])):
            print(f"    fwd[{k}]  setup={s['name']:<40s}  "
                  f"NMSE={fwd['forward_nmse']:.3e}  "
                  f"geo={fwd['geo_id']}  eids={fwd['eids']}")

        for k, (s, fwd) in enumerate(zip(setups, res["fwd"])):
            row = {
                "sample":     si,
                "setup_idx":  k,
                "setup_name": s["name"],
                "W":          s["W"],
                "H":          s["H"],
                "n_hat_stage1":   float(n1),
                "eta_hat_stage1": float(e1),
                "sigma_hat_stage1": float(s1_sy),
                "loss_stage1":    res["loss_stage1"],
                "dt_stage1_s":    res["dt_stage1"],
                "n_hat_stage2":   float(n2),
                "eta_hat_stage2": float(e2),
                "sigma_hat_stage2": float(s2_sy),
                "loss_stage2":    res["loss_stage2"],
                "dt_stage2_s":    res["dt_stage2"],
                "forward_nmse":   fwd["forward_nmse"],
                "geo_id":         fwd["geo_id"],
                "top1_eid":       fwd["eids"][0] if fwd["eids"] else -1,
            }
            # synthetic pairs carry ground-truth
            for tkey, colkey in (("n_true", "n_true"),
                                 ("eta_true", "eta_true"),
                                 ("sigma_true", "sigma_true")):
                if tkey in s:
                    row[colkey] = float(s[tkey])
            if "n_true" in s:
                nt, et, st = s["n_true"], s["eta_true"], s["sigma_true"]
                row["rel_err_n_stage1"]     = abs(n1 - nt) / max(abs(nt), 1e-9)
                row["rel_err_eta_stage1"]   = abs(e1 - et) / max(abs(et), 1e-9)
                row["rel_err_sigma_stage1"] = abs(s1_sy - st) / max(abs(st), 1e-9)
                row["rel_err_n_stage2"]     = abs(n2 - nt) / max(abs(nt), 1e-9)
                row["rel_err_eta_stage2"]   = abs(e2 - et) / max(abs(et), 1e-9)
                row["rel_err_sigma_stage2"] = abs(s2_sy - st) / max(abs(st), 1e-9)
            all_rows.append(row)

        # Per-sample txt dump (matches MoE-240 layout: setup1_* / setup2_*)
        if args.out_dir:
            od = Path(args.out_dir)
            sample_dir = od / f"sample_{si:03d}" if len(samples) > 1 else od
            sample_dir.mkdir(parents=True, exist_ok=True)
            np.savetxt(sample_dir / "setup1_best_x_n_eta_sigma.txt",
                       np.array([res["theta_stage1"]]))
            np.savetxt(sample_dir / "setup2_best_x_n_eta_sigma.txt",
                       np.array([res["theta_stage2"]]))
            with (sample_dir / "loss_history_stage2.csv").open("w") as f:
                f.write("iteration,loss\n")
                for i, v in enumerate(res["hist_stage2"]):
                    f.write(f"{i},{v}\n")

    # ── summary + CSV ────────────────────────────────────────────────────
    df_r = pd.DataFrame(all_rows)
    print(f"\n-- summary: {len(samples)} samples x "
          f"{len(samples[0]) if samples else 0} setups "
          f"= {len(df_r)} rows --")
    for k in ("forward_nmse", "dt_stage1_s", "dt_stage2_s",
              "rel_err_n_stage1", "rel_err_eta_stage1", "rel_err_sigma_stage1",
              "rel_err_n_stage2", "rel_err_eta_stage2", "rel_err_sigma_stage2"):
        if k not in df_r.columns:
            continue
        s = df_r[k].dropna()
        if len(s) == 0:
            continue
        print(f"  {k:28s}  mean={s.mean():.3e}  med={s.median():.3e}  "
              f"max={s.max():.3e}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_r.to_csv(out_path, index=False)
        print(f"  wrote {out_path}")
    elif args.out_dir:
        out_path = Path(args.out_dir) / "results.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_r.to_csv(out_path, index=False)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
