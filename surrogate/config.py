"""
surrogate/config.py
====================
Single source of truth for ALL constants shared across the project.

This replaces the duplicate definitions previously scattered across:
  - DataPipeline/dp_config.py
  - Optimization/libs/moe_core.py  (MIN_N, MAX_N, etc.)
  - Optimization/libs/const.py     (CGS class)
"""

from pathlib import Path

# ── Repo layout ────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent.resolve()
SIM_ROOT   = ROOT / "Simulation"

# ── Physical parameter bounds (CGS units, matching Simulation) ─────────────────
MIN_N,       MAX_N       = 0.3,   1.0
MIN_ETA,     MAX_ETA     = 0.001, 300.0
MIN_SIGMA_Y, MAX_SIGMA_Y = 0.001, 400.0
MIN_WIDTH,   MAX_WIDTH   = 2.0,   7.0
MIN_HEIGHT,  MAX_HEIGHT  = 2.0,   7.0

PARAM_BOUNDS = {
    "n":       (MIN_N,       MAX_N),
    "eta":     (MIN_ETA,     MAX_ETA),
    "sigma_y": (MIN_SIGMA_Y, MAX_SIGMA_Y),
    "width":   (MIN_WIDTH,   MAX_WIDTH),
    "height":  (MIN_HEIGHT,  MAX_HEIGHT),
}

GRAVITY_CGS = 981.0

# ── Numerical constants ──────────────────────────────────────────────────────
LOG_EPS = 1e-8

# ── Column definitions ─────────────────────────────────────────────────────────
INPUT_COLS  = ["n", "eta", "sigma_y", "width", "height"]
OUTPUT_COLS = [f"x_{i:02d}" for i in range(1, 9)]   # x_01 … x_08
LOG_INPUTS  = ["eta", "sigma_y"]

# ── MoE / GMM clustering config ───────────────────────────────────────────────
N_CLUSTERS       = 60
CONF_THRESHOLD   = 0.6
BOX_CONF_THRESH  = 0.7
OUTLIER_Z_THRESH = 3.5

# ── GP expert training config ─────────────────────────────────────────────────
EXACT_THRESHOLD  = 8000
INDUCING_POINTS  = 2048
BATCH_SIZE_SVGP  = 512
EPOCHS_EXACT     = 600
EPOCHS_SVGP      = 500
LR_EXACT         = 0.01
LR_SVGP          = 0.02
MAXERR_TARGET    = 0.05
POLY_ALPHA       = 0.1
POLY_DEGREES     = [2, 3]

# ── Hard-region presets (for targeted data collection) ────────────────────────
HARD_REGION_PRESETS = {
    "splashing": {
        "eta":   (MIN_ETA, 10.0),
        "n":     (MIN_N,   0.55),
        "sigma_y": (MIN_SIGMA_Y, 100.0),
        "width": (MIN_WIDTH, MAX_WIDTH),
        "height": (MIN_HEIGHT, MAX_HEIGHT),
    },
    "narrow_width": {
        "eta":   (MIN_ETA, MAX_ETA),
        "n":     (MIN_N,   MAX_N),
        "sigma_y": (MIN_SIGMA_Y, MAX_SIGMA_Y),
        "width": (MIN_WIDTH, 3.5),
        "height": (MIN_HEIGHT, MAX_HEIGHT),
    },
    "pseudo_static": {
        "eta":   (100.0, MAX_ETA),
        "n":     (MIN_N, MAX_N),
        "sigma_y": (MIN_SIGMA_Y, 5.0),
        "width": (MIN_WIDTH, MAX_WIDTH),
        "height": (MIN_HEIGHT, MAX_HEIGHT),
    },
    "low_eta": {
        "eta":   (MIN_ETA, 5.0),
        "n":     (MIN_N,   MAX_N),
        "sigma_y": (MIN_SIGMA_Y, MAX_SIGMA_Y),
        "width": (MIN_WIDTH, MAX_WIDTH),
        "height": (MIN_HEIGHT, MAX_HEIGHT),
    },
}
