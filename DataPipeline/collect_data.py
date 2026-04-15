"""
DataPipeline/collect_data.py
============================
Headless, Windows-compatible MPM simulation runner for training data collection.

Key improvements over the original Creat_dat.py:
  - Log-uniform sampling for η and σ_y (matches the LogStandardInputScaler in training)
  - No GUI window required (TAICHI_HEADLESS=1 is set automatically)
  - Reads max-x directly from Taichi fields (no intermediate .dat I/O)
  - Append mode: safely adds rows to an existing CSV
  - Targeted mode: collect data in specific hard-region subspaces

Usage examples
--------------
# 5 000 random samples (log-uniform η / σ_y), append to existing CSV:
python collect_data.py --n-samples 5000 --out workspace/data.csv --append

# 2 000 samples in the "splashing" hard region only:
python collect_data.py --n-samples 2000 --preset splashing --out workspace/data_hard.csv

# Custom bounds: low η, narrow width
python collect_data.py --n-samples 1000 --eta-max 10 --width-max 3.5 --out workspace/data_custom.csv

# Latin Hypercube (LHS) instead of random:
python collect_data.py --n-samples 3000 --sampler lhs --out workspace/data_lhs.csv

# Multiple setups per (n, η, σ_y) triplet (like original code):
python collect_data.py --n-samples 2000 --setups-per-param 2 --out workspace/data.csv
"""

import os
import sys
import gc
import csv
import time
import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.stats import qmc

ROOT    = Path(__file__).parent.parent.resolve()
SIM_DIR = ROOT / "Simulation"

_ROOT = Path(__file__).parent.parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from surrogate.config import (
    SIM_ROOT, INPUT_COLS, OUTPUT_COLS,
    MIN_N, MAX_N, MIN_ETA, MAX_ETA, MIN_SIGMA_Y, MAX_SIGMA_Y,
    MIN_WIDTH, MAX_WIDTH, MIN_HEIGHT, MAX_HEIGHT,
    HARD_REGION_PRESETS,
)


# ── Lazy Taichi init (deferred until HeadlessSimulator is constructed) ─────────
# Must set TAICHI_HEADLESS before Taichi is imported; do it now so it is in the
# environment even if someone imports this module before constructing a simulator.
os.environ["TAICHI_HEADLESS"] = "1"

_taichi_ready = False

def _ensure_taichi():
    """Import + initialise Taichi exactly once.

    Uses ti.gpu (Metal on Mac, CUDA on NVIDIA Windows) matching the original
    Creat_dat.py. ti.init() can only be called once per process — never loop.
    """
    global _taichi_ready
    if _taichi_ready:
        return
    sys.path.insert(0, str(SIM_DIR))
    import taichi as ti
    ti.init(arch=ti.gpu, offline_cache=True,
            default_fp=ti.f32, default_ip=ti.i32)
    logging.info("Taichi initialised")
    _taichi_ready = True

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

XML_TEMPLATE = str(SIM_DIR / "config" / "setting.xml")
CSV_COLUMNS  = INPUT_COLS + OUTPUT_COLS


# ── Samplers ───────────────────────────────────────────────────────────────────

def _log_uniform(lo, hi, n, rng):
    """Log-uniform samples: more points near the lower end of the range."""
    return np.exp(rng.uniform(np.log(max(lo, 1e-9)), np.log(hi), n))


def _linear_uniform(lo, hi, n, rng):
    return rng.uniform(lo, hi, n)


def sample_parameters(n_samples: int, bounds: dict, sampler: str,
                      rng: np.random.Generator) -> np.ndarray:
    """Return (n_samples, 5) array: [n, eta, sigma_y, width, height].

    eta and sigma_y use log-uniform sampling; the rest use linear.
    """
    lo_n,   hi_n   = bounds["n"]
    lo_eta, hi_eta = bounds["eta"]
    lo_sy,  hi_sy  = bounds["sigma_y"]
    lo_w,   hi_w   = bounds["width"]
    lo_h,   hi_h   = bounds["height"]

    if sampler == "lhs":
        s = qmc.LatinHypercube(d=5, seed=int(rng.integers(0, 2**31)))
        unit = s.random(n=n_samples)   # (N, 5) in [0,1]

        def _map_log(u, lo, hi):
            return np.exp(u * (np.log(hi) - np.log(max(lo, 1e-9))) + np.log(max(lo, 1e-9)))

        n_vals   = unit[:, 0] * (hi_n   - lo_n)   + lo_n
        eta_vals = _map_log(unit[:, 1], lo_eta, hi_eta)
        sy_vals  = _map_log(unit[:, 2], lo_sy,  hi_sy)
        w_vals   = unit[:, 3] * (hi_w - lo_w) + lo_w
        h_vals   = unit[:, 4] * (hi_h - lo_h) + lo_h
    else:   # random (default)
        n_vals   = _linear_uniform(lo_n,   hi_n,   n_samples, rng)
        eta_vals = _log_uniform(lo_eta, hi_eta, n_samples, rng)
        sy_vals  = _log_uniform(lo_sy,  hi_sy,  n_samples, rng)
        w_vals   = _linear_uniform(lo_w,   hi_w,   n_samples, rng)
        h_vals   = _linear_uniform(lo_h,   hi_h,   n_samples, rng)

    return np.column_stack([n_vals, eta_vals, sy_vals, w_vals, h_vals])


# ── Headless simulation ────────────────────────────────────────────────────────

class HeadlessSimulator:
    """MPM simulator without GUI dependency.

    Reads max-x directly from Taichi fields after each saved frame,
    matching the original Creat_dat.py behaviour (is_inner_of_box
    filtering intentionally omitted for training-data consistency).
    """

    def __init__(self, xml_path: str):
        _ensure_taichi()
        sys.path.insert(0, str(SIM_DIR))
        from simulation.taichi import AGTaichiMPM
        from simulation.xmlParser import MPMXMLData
        xml_data = MPMXMLData(xml_path)
        self.xml_data = xml_data
        self.mpm      = AGTaichiMPM(xml_data)
        self.mpm.changeSetUpData(xml_data)
        self.mpm.initialize()

    def _max_x(self) -> float:
        N = self.mpm.ti_particle_count[None]
        p = self.mpm.ti_particle_x.to_numpy()[:N]
        return float(p[:, 0].max())

    def run(self, n: float, eta: float, sigma_y: float,
            width: float, height: float) -> np.ndarray:
        """Run simulation and return 8 flow-distance differences (cm)."""
        # Geometry
        self.xml_data.cuboidData.max            = [width, height, 4.15]
        self.xml_data.staticBoxList[2].max[0]   = width
        self.xml_data.staticBoxList[3].max[0]   = width

        # Material
        self.xml_data.integratorData.herschel_bulkley_power = n
        self.xml_data.integratorData.eta                    = eta
        self.xml_data.integratorData.yield_stress           = sigma_y

        self.mpm.changeSetUpData(self.xml_data)
        self.mpm.initialize()
        self.mpm.py_num_saved_frames = 0

        x0     = None
        diffs  = []

        while True:
            for _ in range(100):
                self.mpm.step()
                t = self.mpm.ti_iteration[None] * self.mpm.py_dt
                if t * self.mpm.py_fps >= self.mpm.py_num_saved_frames:
                    frame  = self.mpm.py_num_saved_frames
                    max_x  = self._max_x()

                    if frame == 0:
                        x0 = max_x
                    elif x0 is not None and 1 <= frame <= 8:
                        diffs.append(max_x - x0)

                    self.mpm.py_num_saved_frames += 1

            if self.mpm.py_num_saved_frames > self.mpm.py_max_frames:
                gc.collect()
                break

        if len(diffs) < 8:
            diffs += [0.0] * (8 - len(diffs))

        return np.array(diffs[:8], dtype=np.float32)


# ── CSV helpers ────────────────────────────────────────────────────────────────

def _init_csv(path: Path, append: bool) -> bool:
    """Return True if header already written."""
    if append and path.exists():
        return True
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(CSV_COLUMNS)
    return False


def _write_row(path: Path, row: list):
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([f"{v:.8f}" for v in row])


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Headless MPM data collection for MoE training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--n-samples",        type=int,   default=1000,
                        help="Number of parameter sets to simulate (default: 1000)")
    parser.add_argument("--setups-per-param", type=int,   default=1,
                        help="Random (W,H) setups per (n,η,σ_y) triplet (default: 1)")
    parser.add_argument("--sampler",          choices=["random", "lhs"], default="random",
                        help="Sampling strategy (default: random)")
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--out",              type=str,   default="workspace/data.csv",
                        help="Output CSV path (default: workspace/data.csv)")
    parser.add_argument("--append",           action="store_true",
                        help="Append to existing CSV instead of overwriting")

    # Parameter bound overrides
    parser.add_argument("--preset",     type=str, default=None,
                        choices=list(HARD_REGION_PRESETS.keys()),
                        help="Use a preset hard-region bound set")
    parser.add_argument("--n-min",      type=float, default=None)
    parser.add_argument("--n-max",      type=float, default=None)
    parser.add_argument("--eta-min",    type=float, default=None)
    parser.add_argument("--eta-max",    type=float, default=None)
    parser.add_argument("--sigma-min",  type=float, default=None)
    parser.add_argument("--sigma-max",  type=float, default=None)
    parser.add_argument("--width-min",  type=float, default=None)
    parser.add_argument("--width-max",  type=float, default=None)
    parser.add_argument("--height-min", type=float, default=None)
    parser.add_argument("--height-max", type=float, default=None)

    args = parser.parse_args()

    # ── Resolve bounds ─────────────────────────────────────────────────────────
    if args.preset:
        bounds = dict(HARD_REGION_PRESETS[args.preset])
        log.info(f"Using preset '{args.preset}': {bounds}")
    else:
        bounds = {
            "n":       (MIN_N,       MAX_N),
            "eta":     (MIN_ETA,     MAX_ETA),
            "sigma_y": (MIN_SIGMA_Y, MAX_SIGMA_Y),
            "width":   (MIN_WIDTH,   MAX_WIDTH),
            "height":  (MIN_HEIGHT,  MAX_HEIGHT),
        }

    # Command-line overrides
    for key, lo_arg, hi_arg in [
        ("n",       args.n_min,     args.n_max),
        ("eta",     args.eta_min,   args.eta_max),
        ("sigma_y", args.sigma_min, args.sigma_max),
        ("width",   args.width_min, args.width_max),
        ("height",  args.height_min, args.height_max),
    ]:
        lo, hi = bounds[key]
        bounds[key] = (lo_arg if lo_arg is not None else lo,
                       hi_arg if hi_arg is not None else hi)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _init_csv(out_path, args.append)

    rng = np.random.default_rng(args.seed)
    sim = HeadlessSimulator(XML_TEMPLATE)

    # Pre-sample all parameter triplets (n, η, σ_y)
    triplets = sample_parameters(args.n_samples, bounds, args.sampler, rng)

    total_rows  = 0
    t_start     = time.time()

    for idx, (n, eta, sigma_y, w0, h0) in enumerate(triplets):
        # Each triplet gets setups_per_param random (W, H) setups
        if args.setups_per_param == 1:
            wh_list = [(w0, h0)]
        else:
            wh_list = [
                (float(rng.uniform(bounds["width"][0],  bounds["width"][1])),
                 float(rng.uniform(bounds["height"][0], bounds["height"][1])))
                for _ in range(args.setups_per_param)
            ]

        for width, height in wh_list:
            try:
                diffs = sim.run(float(n), float(eta), float(sigma_y), width, height)
                row   = [n, eta, sigma_y, width, height] + diffs.tolist()
                _write_row(out_path, row)
                total_rows += 1
            except Exception as exc:
                log.warning(f"  [SKIP] param {idx}: {exc}")
                continue

        elapsed = time.time() - t_start
        rate    = total_rows / max(elapsed, 1)
        eta_s   = (args.n_samples * args.setups_per_param - total_rows) / max(rate, 1e-6)
        log.info(
            f"[{idx+1:>6}/{args.n_samples}] rows={total_rows} "
            f"rate={rate:.1f}/s  ETA={eta_s/60:.1f} min"
        )

    log.info(f"\nDone.  Wrote {total_rows} rows → {out_path}")


if __name__ == "__main__":
    main()
