#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_out_dir.py
================
Adds --out_dir argument to pipeline.py, optimize_1setup.py,
optimize_2setups.py, and Simulation/main.py so that all outputs
go to a caller-specified directory instead of being hardcoded
next to the input files or in a timestamped subdirectory.

Run once from the project root:
    python3 patch_out_dir.py

Each script is backed up as <script>.bak before patching.
"""

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def backup(path: Path):
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    print(f"  backed up → {bak.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Calibration/pipeline.py
# ══════════════════════════════════════════════════════════════════════════════

def patch_pipeline():
    path = ROOT / "Calibration" / "pipeline.py"
    src  = path.read_text()
    backup(path)

    # Add --out_dir argument after --theta0
    old = '    parser.add_argument("--theta0",     default=None)'
    new = (
        '    parser.add_argument("--theta0",     default=None)\n'
        '    parser.add_argument("--out_dir",    default=None,\n'
        '                        help="Output directory. Default: same directory as --calib_img")'
    )
    assert old in src, "Pattern not found in pipeline.py (--theta0)"
    src = src.replace(old, new)

    # Replace hardcoded out_dir construction
    old2 = '    out_dir = os.path.dirname(os.path.abspath(args.calib_img))'
    new2 = (
        '    out_dir = args.out_dir if args.out_dir else os.path.dirname(os.path.abspath(args.calib_img))'
    )
    assert old2 in src, "Pattern not found in pipeline.py (out_dir construction)"
    src = src.replace(old2, new2)

    path.write_text(src)
    print(f"  patched  → Calibration/pipeline.py")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Optimization/optimize_1setup.py
# ══════════════════════════════════════════════════════════════════════════════

def patch_optimize_1():
    path = ROOT / "Optimization" / "optimize_1setup.py"
    src  = path.read_text()
    backup(path)

    # Add --out_dir argument
    old = '    p.add_argument("--verb",     type=int,   default=1)'
    new = (
        '    p.add_argument("--verb",     type=int,   default=1)\n'
        '    p.add_argument("--out_dir",  type=str,   default=None,\n'
        '                   help="Output directory. Default: result_setup1_<strategy>_<ts>/")'
    )
    assert old in src, "Pattern not found in optimize_1setup.py (--verb)"
    src = src.replace(old, new)

    # Replace timestamped save_dir construction
    old2 = (
        '    ts           = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")\n'
        '    strategy_str = f"topk_{args.topk}" if args.strategy == "topk" \\\\\n'
        '                   else f"{args.strategy}_{args.threshold}"\n'
        '    save_dir = f"result_setup1_{strategy_str}_{ts}"\n'
        '    os.makedirs(save_dir, exist_ok=True)'
    )
    new2 = (
        '    ts           = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")\n'
        '    strategy_str = f"topk_{args.topk}" if args.strategy == "topk" \\\\\n'
        '                   else f"{args.strategy}_{args.threshold}"\n'
        '    save_dir = args.out_dir if args.out_dir else f"result_setup1_{strategy_str}_{ts}"\n'
        '    os.makedirs(save_dir, exist_ok=True)'
    )
    if old2 not in src:
        # Try without escaped backslash (depends on how the file was saved)
        old2 = old2.replace('\\\\', '\\')
        new2 = new2.replace('\\\\', '\\')
    assert old2 in src, "Pattern not found in optimize_1setup.py (save_dir construction)"
    src = src.replace(old2, new2)

    path.write_text(src)
    print(f"  patched  → Optimization/optimize_1setup.py")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Optimization/optimize_2setups.py
# ══════════════════════════════════════════════════════════════════════════════

def patch_optimize_2():
    path = ROOT / "Optimization" / "optimize_2setups.py"
    src  = path.read_text()
    backup(path)

    old = '    p.add_argument("--verb",     type=int,   default=1)'
    new = (
        '    p.add_argument("--verb",     type=int,   default=1)\n'
        '    p.add_argument("--out_dir",  type=str,   default=None,\n'
        '                   help="Output directory. Default: result_setup2_<strategy>_<ts>/")'
    )
    assert old in src, "Pattern not found in optimize_2setups.py (--verb)"
    src = src.replace(old, new)

    old2 = (
        '    ts           = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")\n'
        '    strategy_str = f"topk_{args.topk}" if args.strategy == "topk" \\\\\n'
        '                   else f"{args.strategy}_{args.threshold}"\n'
        '    save_dir = f"result_setup2_{strategy_str}_{ts}"\n'
        '    os.makedirs(save_dir, exist_ok=True)'
    )
    new2 = (
        '    ts           = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")\n'
        '    strategy_str = f"topk_{args.topk}" if args.strategy == "topk" \\\\\n'
        '                   else f"{args.strategy}_{args.threshold}"\n'
        '    save_dir = args.out_dir if args.out_dir else f"result_setup2_{strategy_str}_{ts}"\n'
        '    os.makedirs(save_dir, exist_ok=True)'
    )
    if old2 not in src:
        old2 = old2.replace('\\\\', '\\')
        new2 = new2.replace('\\\\', '\\')
    assert old2 in src, "Pattern not found in optimize_2setups.py (save_dir construction)"
    src = src.replace(old2, new2)

    path.write_text(src)
    print(f"  patched  → Optimization/optimize_2setups.py")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Simulation/main.py
# ══════════════════════════════════════════════════════════════════════════════

def patch_simulation_main():
    path = ROOT / "Simulation" / "main.py"
    src  = path.read_text()
    backup(path)

    # Add --out_dir and --camera_xml arguments
    old = "    parser.add_argument('--diff_amplify', type=float, default=5.0,"
    new = (
        "    parser.add_argument('--out_dir',     type=str,   default=None,\n"
        "                        help='Output directory. Default: Simulation/results/run_<ts>/')\n"
        "    parser.add_argument('--camera_xml',  type=str,   default=None,\n"
        "                        help='Override camera_params.xml path (default: --ref/camera_params.xml)')\n"
        "    parser.add_argument('--diff_amplify', type=float, default=5.0,"
    )
    assert old in src, "Pattern not found in main.py (--diff_amplify)"
    src = src.replace(old, new)

    # Replace hardcoded results_root
    old2 = '    results_root = os.path.join("results", f"run_{timestamp}")'
    new2 = (
        '    results_root = args.out_dir if args.out_dir \\\n'
        '                   else os.path.join("results", f"run_{timestamp}")'
    )
    assert old2 in src, "Pattern not found in main.py (results_root)"
    src = src.replace(old2, new2)

    # Patch load_camera_params call to use --camera_xml if provided
    old3 = '        camera_params = load_camera_params(args.ref)'
    new3 = (
        '        cam_xml_path  = args.camera_xml if args.camera_xml else args.ref\n'
        '        camera_params = load_camera_params(cam_xml_path)'
    )
    if old3 in src:
        src = src.replace(old3, new3)

    path.write_text(src)
    print(f"  patched  → Simulation/main.py")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Write updated settings.xml template to data/ structure
# ══════════════════════════════════════════════════════════════════════════════

SETTINGS_TEMPLATE = """\
<?xml version="1.0"?>
<Optimizer>
  <!--
    settings.xml  —  container geometry for this experiment
    W = container width  [cm]
    H = container height [cm]
  -->
  <setup RHO="1.0" H="{H}" W="{W}" />
  <cuboid min="-0.15 -0.15 -0.15" max="{W} {H} 4.15"
          density="1.0" cell_samples_per_dim="2"
          vel="0.0 0.0 0.0" omega="0.0 0.0 0.0" />
  <static_box min="-100 -1 -100" max="100  0 100" boundary_behavior="sticking"/>
  <static_box min="-1   0  0"   max="0   20   4"  boundary_behavior="sticking"/>
  <static_box min="-1   0 -0.3" max="{W} 20   0"  boundary_behavior="sticking"/>
  <static_box min="-1   0  4"   max="{W} 20  4.3" boundary_behavior="sticking"/>
</Optimizer>
"""


def print_summary():
    print("""
╔══════════════════════════════════════════════════════════════╗
║  Patch complete. New --out_dir argument added to:            ║
║    Calibration/pipeline.py                                   ║
║    Optimization/optimize_1setup.py                           ║
║    Optimization/optimize_2setups.py                          ║
║    Simulation/main.py                                        ║
╠══════════════════════════════════════════════════════════════╣
║  Ideal data layout going forward:                            ║
║                                                              ║
║  data/                                                       ║
║    {material}/                                               ║
║      settings.xml      ← W, H of container                  ║
║      calib.JPG         ← photo with ChArUco board            ║
║      config_00.png     ← reference frame (flow = 0)         ║
║      config_01.png                                           ║
║      ...                                                     ║
║      config_08.png                                           ║
║                                                              ║
║  Run the full pipeline:                                      ║
║    python3 run_pipeline.py --material {material}             ║
║                                                              ║
║  Results land in:                                            ║
║    outputs/{material}/                                       ║
║      01_calibration/                                         ║
║      02_flow_distances/                                      ║
║      03_optimization/                                        ║
║      04_simulation/                                          ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print("Patching scripts...")
    try:
        patch_pipeline()
        patch_optimize_1()
        patch_optimize_2()
        patch_simulation_main()
        print_summary()
    except AssertionError as e:
        print(f"\n[ERROR] {e}")
        print("The script may have already been patched, or the source changed.")
        sys.exit(1)
