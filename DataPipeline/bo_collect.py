"""
DataPipeline/bo_collect.py
==========================
Step 4.5 – BO-guided targeted data collection for under-performing clusters.

Workflow
--------
1. Read evaluate_experts.py metrics CSV → find clusters whose MaxErr > threshold
2. For each bad cluster:
   a. Load its parameter bounding box from boxes.json
   b. Load existing cluster training data → hot-start a lightweight SVGP
   c. BO loop: variance-based acquisition → HeadlessSimulator → append CSV
3. After all bad clusters are done, call train_experts.py --clusters <ids>
   to retrain only those experts

Usage
-----
# Auto-detect bad clusters (MaxErr > 1.0 cm) from metrics CSV:
python bo_collect.py --data workspace/moe_ws \\
                     --metrics metrics.csv \\
                     --maxerr-thresh 1.0 \\
                     --n-per-cluster 200

# Manually specify which clusters to target:
python bo_collect.py --data workspace/moe_ws \\
                     --clusters 19 42 46 \\
                     --n-per-cluster 300

# Dry-run: show which clusters would be targeted, no simulation:
python bo_collect.py --data workspace/moe_ws --metrics metrics.csv --dry-run
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.kernels import ScaleKernel, MaternKernel, LinearKernel
from gpytorch.means import ConstantMean
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "DataPipeline"))

from dp_config import INPUT_COLS, OUTPUT_COLS, MIN_N, MAX_N, MIN_WIDTH, MAX_WIDTH, MIN_HEIGHT, MAX_HEIGHT

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32   # float32 is fine for the BO surrogate


# ── Lightweight cluster-local SVGP ────────────────────────────────────────────

class _ClusterSVGP(ApproximateGP):
    """Single-output SVGP used as within-cluster uncertainty surrogate."""
    def __init__(self, inducing: torch.Tensor, n_outputs: int):
        batch = torch.Size([n_outputs])
        vd = CholeskyVariationalDistribution(inducing.size(0), batch_shape=batch)
        ind_b = inducing.unsqueeze(0).expand(n_outputs, -1, -1)
        vs = VariationalStrategy(self, ind_b, vd, learn_inducing_locations=True)
        super().__init__(vs)
        D = inducing.size(-1)
        self.mean_module  = ConstantMean(batch_shape=batch)
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=D, batch_shape=batch)
            + LinearKernel(ard_num_dims=D, batch_shape=batch),
            batch_shape=batch,
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class ClusterSurrogate:
    """Trains a cluster-local SVGP and provides variance-based acquisition."""

    def __init__(self, n_outputs: int = 8, n_inducing: int = 128,
                 lr: float = 0.02, epochs: int = 200):
        self.n_outputs  = n_outputs
        self.n_inducing = n_inducing
        self.lr         = lr
        self.epochs     = epochs
        self.model      = None
        self.likelihood = None
        self.x_mean     = None
        self.x_std      = None

    # ── standardize ───────────────────────────────────────────────────────────
    def _fit_scaler(self, X: np.ndarray):
        """Log-standardize inputs (log for η, σ_y; linear for n, W, H)."""
        X = np.array(X, dtype=np.float32)
        Xl = X.copy()
        Xl[:, 1] = np.log(np.clip(X[:, 1], 1e-9, None))  # eta
        Xl[:, 2] = np.log(np.clip(X[:, 2], 1e-9, None))  # sigma_y
        self.x_mean = Xl.mean(axis=0)
        self.x_std  = np.clip(Xl.std(axis=0), 1e-6, None)

    def _scale(self, X: np.ndarray) -> torch.Tensor:
        X = np.array(X, dtype=np.float32)
        Xl = X.copy()
        Xl[:, 1] = np.log(np.clip(X[:, 1], 1e-9, None))
        Xl[:, 2] = np.log(np.clip(X[:, 2], 1e-9, None))
        Xs = (Xl - self.x_mean) / self.x_std
        return torch.tensor(Xs, dtype=DTYPE, device=DEVICE)

    # ── fit ───────────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, Y: np.ndarray):
        """X: (N,5), Y: (N,8)."""
        self._fit_scaler(X)
        Xt = self._scale(X)
        Yt = torch.tensor(Y, dtype=DTYPE, device=DEVICE)  # (N,8)

        m = min(self.n_inducing, len(X))
        idx = torch.randperm(len(X))[:m]
        inducing = Xt[idx].detach()

        self.model      = _ClusterSVGP(inducing, self.n_outputs).to(DEVICE, DTYPE)
        self.likelihood = GaussianLikelihood(
            batch_shape=torch.Size([self.n_outputs])
        ).to(DEVICE, DTYPE)

        self.model.train(); self.likelihood.train()
        optim = torch.optim.Adam(
            list(self.model.parameters()) + list(self.likelihood.parameters()),
            lr=self.lr,
        )
        mll   = VariationalELBO(self.likelihood, self.model, num_data=len(X))
        loader = DataLoader(TensorDataset(Xt, Yt),
                            batch_size=min(512, len(X)), shuffle=True)

        for ep in range(self.epochs):
            ep_loss = 0.0
            for xb, yb in loader:
                optim.zero_grad(set_to_none=True)
                loss = -mll(self.model(xb), yb.T).sum()
                loss.backward()
                optim.step()
                ep_loss += loss.item()
            if (ep + 1) % 50 == 0:
                log.info(f"      SVGP epoch {ep+1}/{self.epochs}  loss={ep_loss:.3f}")

        self.model.eval(); self.likelihood.eval()
        log.info(f"    SVGP fitted on N={len(X)}, M={m}")

    # ── variance-based acquisition ────────────────────────────────────────────
    def variance(self, X_cand: np.ndarray) -> np.ndarray:
        """Return summed output variance for each candidate row."""
        Xt = self._scale(X_cand)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(Xt))
            # pred.variance: (n_outputs, N_cand)
            unc = pred.variance.sum(dim=0).cpu().numpy()
        return unc  # (N_cand,)


# ── Cluster-targeted BO loop ──────────────────────────────────────────────────

def _sample_in_box(box: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n points log-uniformly in η/σ_y and linearly in n/W/H."""
    lo_n,  hi_n  = box.get("n",       [MIN_N,    MAX_N])
    lo_e,  hi_e  = box.get("eta",     [0.001,    300.0])
    lo_s,  hi_s  = box.get("sigma_y", [0.1,      400.0])
    lo_w,  hi_w  = box.get("width",   [MIN_WIDTH, MAX_WIDTH])
    lo_h,  hi_h  = box.get("height",  [MIN_HEIGHT, MAX_HEIGHT])

    n_v  = rng.uniform(lo_n, hi_n, n)
    e_v  = np.exp(rng.uniform(np.log(max(lo_e, 1e-9)), np.log(hi_e), n))
    s_v  = np.exp(rng.uniform(np.log(max(lo_s, 1e-9)), np.log(hi_s), n))
    w_v  = rng.uniform(lo_w, hi_w, n)
    h_v  = rng.uniform(lo_h, hi_h, n)
    return np.column_stack([n_v, e_v, s_v, w_v, h_v])


def bo_collect_cluster(
    cid: int,
    box: dict,
    train_csv: Path,
    out_csv: Path,
    sim,
    n_collect: int,
    n_candidates: int = 2000,
    surrogate_epochs: int = 200,
    rng: np.random.Generator = None,
):
    """Run BO data collection for one cluster."""
    if rng is None:
        rng = np.random.default_rng()

    # ── load existing cluster data as warm-start ───────────────────────────
    df_train = pd.read_csv(train_csv) if train_csv.exists() else pd.DataFrame()
    has_seed = len(df_train) >= 10

    surrogate = ClusterSurrogate(n_outputs=len(OUTPUT_COLS),
                                 epochs=surrogate_epochs)

    if has_seed:
        log.info(f"  Warm-starting SVGP from {len(df_train)} existing points")
        X_seed = df_train[INPUT_COLS].values
        Y_seed = df_train[OUTPUT_COLS].values
        surrogate.fit(X_seed, Y_seed)
    else:
        log.info("  Not enough seed data — using random acquisition for first batch")

    # ── ensure output CSV header ───────────────────────────────────────────
    if not out_csv.exists():
        pd.DataFrame(columns=INPUT_COLS + OUTPUT_COLS).to_csv(out_csv, index=False)

    collected = 0
    t0 = time.time()

    while collected < n_collect:
        # Sample candidates in cluster bounding box
        cands = _sample_in_box(box, n_candidates, rng)   # (N_cand, 5)

        if has_seed or collected > 0:
            scores = surrogate.variance(cands)
            top_idx = int(np.argmax(scores))
        else:
            top_idx = int(rng.integers(0, n_candidates))

        params = cands[top_idx]  # [n, eta, sigma_y, width, height]

        try:
            diffs = sim.run(
                float(params[0]), float(params[1]), float(params[2]),
                float(params[3]), float(params[4]),
            )
        except Exception as exc:
            log.warning(f"    Sim failed: {exc}")
            continue

        # Clip negatives (physically impossible)
        diffs = np.clip(diffs, 0.0, None)

        row = pd.DataFrame([list(params) + list(diffs)],
                           columns=INPUT_COLS + OUTPUT_COLS)
        row.to_csv(out_csv, mode="a", header=False, index=False)
        collected += 1

        # Update surrogate every 10 new points
        if collected % 10 == 0:
            df_all = pd.read_csv(out_csv)
            if len(df_all) >= 10:
                surrogate.fit(df_all[INPUT_COLS].values,
                              df_all[OUTPUT_COLS].values)
            elapsed = time.time() - t0
            rate    = collected / max(elapsed, 1)
            eta_s   = (n_collect - collected) / max(rate, 1e-6)
            log.info(f"    [{collected:>4}/{n_collect}] "
                     f"rate={rate:.2f}/s  ETA={eta_s/60:.1f}min")

    log.info(f"  Cluster {cid}: collected {collected} points → {out_csv}")


# ── Find bad clusters from metrics CSV ────────────────────────────────────────

def find_bad_clusters(metrics_csv: Path, maxerr_thresh: float,
                      split: str = "val") -> list:
    """Return list of cluster IDs whose MaxErr exceeds the threshold."""
    df = pd.read_csv(metrics_csv)
    df = df[df["split"] == split]
    df = df[df["cluster"] != "GLOBAL"]
    df["cluster"] = df["cluster"].astype(int)
    bad = df[df["maxerr"] > maxerr_thresh]["cluster"].tolist()
    return sorted(bad)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BO-guided targeted data collection for bad clusters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data",     required=True,
                        help="Workspace directory (contains boxes.json, cluster*_train.csv)")
    parser.add_argument("--metrics",  default=None,
                        help="Metrics CSV from evaluate_experts.py (for auto cluster selection)")
    parser.add_argument("--clusters", type=int, nargs="*", default=None,
                        help="Manually specify cluster IDs to target")
    parser.add_argument("--maxerr-thresh", type=float, default=1.0,
                        help="MaxErr threshold (cm) for auto bad-cluster detection (default: 1.0)")
    parser.add_argument("--n-per-cluster", type=int, default=200,
                        help="Number of new simulation points per cluster (default: 200)")
    parser.add_argument("--n-candidates",  type=int, default=2000,
                        help="Candidate pool size for each BO step (default: 2000)")
    parser.add_argument("--surrogate-epochs", type=int, default=200,
                        help="SVGP training epochs (default: 200)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print which clusters would be targeted and exit")
    args = parser.parse_args()

    data_dir = Path(args.data)
    rng      = np.random.default_rng(args.seed)

    # ── Resolve target cluster list ────────────────────────────────────────
    if args.clusters:
        target_ids = args.clusters
        log.info(f"Manually specified clusters: {target_ids}")
    elif args.metrics:
        target_ids = find_bad_clusters(
            Path(args.metrics), args.maxerr_thresh
        )
        log.info(f"Auto-detected bad clusters (MaxErr>{args.maxerr_thresh}): {target_ids}")
    else:
        parser.error("Provide either --clusters or --metrics")

    if not target_ids:
        log.info("No bad clusters found. Nothing to do.")
        return

    # ── Load boxes ─────────────────────────────────────────────────────────
    boxes_path = data_dir / "boxes.json"
    if not boxes_path.exists():
        log.error(f"boxes.json not found in {data_dir}")
        sys.exit(1)
    with open(boxes_path) as f:
        boxes = json.load(f)

    if args.dry_run:
        print(f"\nWould target {len(target_ids)} clusters:")
        for cid in target_ids:
            box = boxes.get(str(cid), {})
            print(f"  Cluster {cid:>3}: {box}")
        return

    # ── Init simulator (lazy — only imported here) ─────────────────────────
    log.info("Initialising headless simulator...")
    # Set headless before any Taichi import
    os.environ["TAICHI_HEADLESS"] = "1"
    SIM_DIR = ROOT / "Simulation"
    sys.path.insert(0, str(SIM_DIR))

    import taichi as ti
    ti.init(arch=ti.gpu, offline_cache=True,
            default_fp=ti.f32, default_ip=ti.i32)

    from simulation.taichi import AGTaichiMPM
    from simulation.xmlParser import MPMXMLData

    # Import HeadlessSimulator logic inline to avoid circular import
    XML_TEMPLATE = str(SIM_DIR / "config" / "setting.xml")

    class _Sim:
        def __init__(self):
            xml_data = MPMXMLData(XML_TEMPLATE)
            self.xml_data = xml_data
            self.mpm      = AGTaichiMPM(xml_data)
            self.mpm.changeSetUpData(xml_data)
            self.mpm.initialize()

        def run(self, n, eta, sigma_y, width, height):
            import gc
            self.xml_data.cuboidData.max          = [width, height, 4.15]
            self.xml_data.staticBoxList[2].max[0] = width
            self.xml_data.staticBoxList[3].max[0] = width
            self.xml_data.integratorData.herschel_bulkley_power = n
            self.xml_data.integratorData.eta                    = eta
            self.xml_data.integratorData.yield_stress           = sigma_y
            self.mpm.changeSetUpData(self.xml_data)
            self.mpm.initialize()
            self.mpm.py_num_saved_frames = 0
            x0, diffs = None, []
            while True:
                for _ in range(100):
                    self.mpm.step()
                    t = self.mpm.ti_iteration[None] * self.mpm.py_dt
                    if t * self.mpm.py_fps >= self.mpm.py_num_saved_frames:
                        frame = self.mpm.py_num_saved_frames
                        N     = self.mpm.ti_particle_count[None]
                        p     = self.mpm.ti_particle_x.to_numpy()[:N]
                        mx    = float(p[:, 0].max())
                        if frame == 0:
                            x0 = mx
                        elif x0 is not None and 1 <= frame <= 8:
                            diffs.append(mx - x0)
                        self.mpm.py_num_saved_frames += 1
                if self.mpm.py_num_saved_frames > self.mpm.py_max_frames:
                    gc.collect(); break
            if len(diffs) < 8:
                diffs += [0.0] * (8 - len(diffs))
            return np.array(diffs[:8], dtype=np.float32)

    sim = _Sim()
    log.info("Simulator ready.")

    # ── Run BO per bad cluster ─────────────────────────────────────────────
    for cid in target_ids:
        box = boxes.get(str(cid))
        if box is None:
            log.warning(f"Cluster {cid} not in boxes.json — skipping")
            continue

        log.info(f"\n{'='*55}")
        log.info(f"  Cluster {cid}  box={box}")
        log.info(f"{'='*55}")

        train_csv = data_dir / f"cluster{cid}_train.csv"
        out_csv   = data_dir / f"cluster{cid}_bo.csv"

        bo_collect_cluster(
            cid         = cid,
            box         = box,
            train_csv   = train_csv,
            out_csv     = out_csv,
            sim         = sim,
            n_collect   = args.n_per_cluster,
            n_candidates= args.n_candidates,
            surrogate_epochs=args.surrogate_epochs,
            rng         = rng,
        )

    log.info("\n=== BO collection done ===")
    log.info("Next step — retrain only targeted clusters:")
    ids_str = " ".join(str(c) for c in target_ids)
    log.info(f"  python DataPipeline/train_experts.py --data {data_dir} "
             f"--clusters {ids_str} --force")

    # Optionally merge bo CSVs into cluster train CSVs automatically
    log.info("\nTo merge BO data into cluster train CSVs before retraining:")
    for cid in target_ids:
        bo_csv    = data_dir / f"cluster{cid}_bo.csv"
        train_csv = data_dir / f"cluster{cid}_train.csv"
        if bo_csv.exists() and train_csv.exists():
            df_old = pd.read_csv(train_csv)
            df_bo  = pd.read_csv(bo_csv)
            # BO points are targeted to this cluster — fill missing labels so
            # train_experts.py's cluster_conf>=CONF_THRESHOLD filter keeps them.
            if "cluster_id" in df_old.columns and "cluster_id" not in df_bo.columns:
                df_bo["cluster_id"] = cid
            if "cluster_conf" in df_old.columns and "cluster_conf" not in df_bo.columns:
                df_bo["cluster_conf"] = 1.0
            df_new = pd.concat([df_old, df_bo], ignore_index=True)
            # Fill any remaining NaN in these two columns (older BO CSVs)
            if "cluster_id" in df_new.columns:
                df_new.loc[df_new["cluster_id"].isna(), "cluster_id"] = cid
                df_new["cluster_id"] = df_new["cluster_id"].astype(int)
            if "cluster_conf" in df_new.columns:
                df_new.loc[df_new["cluster_conf"].isna(), "cluster_conf"] = 1.0
            df_new.to_csv(train_csv, index=False)
            log.info(f"  Cluster {cid}: {len(df_old)} + {len(df_bo)} → {len(df_new)} rows")
            bo_csv.unlink()   # remove tmp file after merge

    log.info("Merge done. Run train_experts.py --clusters to retrain.")


if __name__ == "__main__":
    main()
