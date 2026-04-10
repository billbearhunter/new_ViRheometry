"""
DataPipeline/config.py
======================
Single source of truth for all parameters shared across the pipeline steps.
"""

from pathlib import Path

# ── Repo layout ────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent.resolve()   # repo root
SIM_ROOT   = ROOT / "Simulation"                       # Simulation/ folder
DATA_DIR   = Path(__file__).parent / "workspace"       # default output workspace

# ── Physical parameter bounds (CGS units, matching Simulation) ─────────────────
MIN_N,       MAX_N       = 0.3,   1.0
MIN_ETA,     MAX_ETA     = 0.001, 300.0
MIN_SIGMA_Y, MAX_SIGMA_Y = 0.001, 400.0     # avoid exact 0 for log-sampling
MIN_WIDTH,   MAX_WIDTH   = 2.0,   7.0
MIN_HEIGHT,  MAX_HEIGHT  = 2.0,   7.0

PARAM_BOUNDS = {
    "n":       (MIN_N,       MAX_N),
    "eta":     (MIN_ETA,     MAX_ETA),
    "sigma_y": (MIN_SIGMA_Y, MAX_SIGMA_Y),
    "width":   (MIN_WIDTH,   MAX_WIDTH),
    "height":  (MIN_HEIGHT,  MAX_HEIGHT),
}

# ── Hard-region presets (for targeted data collection) ────────────────────────
# These correspond to the hard clusters identified in the paper (§6.4.4).
HARD_REGION_PRESETS = {
    "splashing": {               # Cluster 19 — low-η water-like fluids
        "eta":   (MIN_ETA, 10.0),
        "n":     (MIN_N,   0.55),
        "sigma_y": (MIN_SIGMA_Y, 100.0),
        "width": (MIN_WIDTH, MAX_WIDTH),
        "height": (MIN_HEIGHT, MAX_HEIGHT),
    },
    "narrow_width": {            # W < 3.5 — high-aspect-ratio global collapse
        "eta":   (MIN_ETA, MAX_ETA),
        "n":     (MIN_N,   MAX_N),
        "sigma_y": (MIN_SIGMA_Y, MAX_SIGMA_Y),
        "width": (MIN_WIDTH, 3.5),
        "height": (MIN_HEIGHT, MAX_HEIGHT),
    },
    "pseudo_static": {           # σ_y → 0 + high η — viscous delay vs yield stop
        "eta":   (100.0, MAX_ETA),
        "n":     (MIN_N, MAX_N),
        "sigma_y": (MIN_SIGMA_Y, 5.0),
        "width": (MIN_WIDTH, MAX_WIDTH),
        "height": (MIN_HEIGHT, MAX_HEIGHT),
    },
    "low_eta": {                 # General low-viscosity under-sampled region
        "eta":   (MIN_ETA, 5.0),
        "n":     (MIN_N,   MAX_N),
        "sigma_y": (MIN_SIGMA_Y, MAX_SIGMA_Y),
        "width": (MIN_WIDTH, MAX_WIDTH),
        "height": (MIN_HEIGHT, MAX_HEIGHT),
    },
}

# ── MoE / GMM training config ──────────────────────────────────────────────────
N_CLUSTERS       = 60      # GMM components (match moe_workspace5)
CONF_THRESHOLD   = 0.6     # min GMM confidence for cluster assignment
BOX_CONF_THRESH  = 0.7     # for cluster bounding-box computation
OUTLIER_Z_THRESH = 3.5     # Z-score outlier removal threshold

# ── GP expert training config ──────────────────────────────────────────────────
EXACT_THRESHOLD  = 3000    # samples per cluster: <= use ExactGP, > use SVGP
INDUCING_POINTS  = 2048
BATCH_SIZE_SVGP  = 512
EPOCHS_EXACT     = 600
EPOCHS_SVGP      = 500
LR_EXACT         = 0.01
LR_SVGP          = 0.02
MAXERR_TARGET    = 0.05    # poly residual boosting trigger threshold
POLY_ALPHA       = 0.1
POLY_DEGREES     = [2, 3]

# ── Column definitions ─────────────────────────────────────────────────────────
INPUT_COLS  = ["n", "eta", "sigma_y", "width", "height"]
OUTPUT_COLS = [f"x_{i:02d}" for i in range(1, 9)]   # x_01 … x_08
LOG_INPUTS  = ["eta", "sigma_y"]                     # log-transformed before scaling
