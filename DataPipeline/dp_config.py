"""
DataPipeline/dp_config.py
=========================
Backward-compatible re-export of all config values from surrogate.config.

All constants are now defined in surrogate/config.py (single source of truth).
This file re-exports them so existing DataPipeline scripts continue to work
without import changes.
"""

import sys
from pathlib import Path

# Ensure repo root is importable
_ROOT = Path(__file__).parent.parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from surrogate.config import *  # noqa: F401,F403

# DataPipeline-specific paths (not in surrogate.config)
ROOT     = _ROOT
SIM_ROOT = ROOT / "Simulation"
DATA_DIR = Path(__file__).parent / "workspace"
