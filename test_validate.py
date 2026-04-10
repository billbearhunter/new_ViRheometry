#!/usr/bin/env python3
"""
test_validate.py — Rheometer-based validation of the ViRheometry pipeline
==========================================================================
Fits Herschel-Bulkley parameters directly from a rheometer measurement CSV,
then runs the MPM simulation with those parameters and computes frame diffs
against the real experiment images.

If the pipeline is correctly calibrated, snapdiff_*.png images should be
near-zero (i.e. simulated flow matches the observed flow).

Usage:
    python3 test_validate.py \\
        --data data/ref_Tonkatsu_6.7_3.5_1 \\
        --rheo FlowCurve/Rheo_Data/tonkatsu_20230113_2000_23C.csv

    # Custom row range for HB fitting (default: 5–19)
    python3 test_validate.py \\
        --data data/ref_Tonkatsu_6.7_3.5_1 \\
        --rheo FlowCurve/Rheo_Data/tonkatsu_20230113_2000_23C.csv \\
        --rheo-range 3 15

Outputs are saved under Simulation/results/validation_<timestamp>/.
Calibration must have been run beforehand (Calibration/results/<name>/ must exist).
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def run(cmd: list, cwd: Path = ROOT) -> int:
    print("  $", " ".join(str(c) for c in cmd))
    print()
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Validation: simulate with rheometer HB params and compare to experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data", required=True,
                        help="Material data directory (e.g. data/ref_Tonkatsu_6.7_3.5_1)")
    parser.add_argument("--rheo", required=True,
                        help="Rheometer CSV file path")
    parser.add_argument("--rheo-range", nargs=2, type=int, default=[5, 19],
                        metavar=("START", "END"),
                        help="Row index range for HB fitting (default: 5 19)")
    parser.add_argument("--diff-amplify", type=float, default=5.0,
                        help="Brightness amplification for diff images (default: 5.0)")
    args = parser.parse_args()

    data_dir      = (ROOT / args.data).resolve()
    material_name = data_dir.name
    rheo_path     = (ROOT / args.rheo).resolve()
    calib_out     = ROOT / "Calibration" / "results" / material_name
    camera_xml    = calib_out / "camera_params.xml"
    rheo_json     = calib_out / "rheo_params.json"

    # Validate inputs
    if not data_dir.is_dir():
        sys.exit(f"[error] Data directory not found: {data_dir}")
    if not rheo_path.is_file():
        sys.exit(f"[error] Rheometer CSV not found: {rheo_path}")
    if not camera_xml.exists():
        sys.exit(
            f"[error] camera_params.xml not found: {camera_xml}\n"
            f"        Run calibration first: python3 run_pipeline.py --data {args.data} --skip-extraction --skip-optimization"
        )

    print(f"\n[info] Material  : {material_name}")
    print(f"[info] Data      : {data_dir}")
    print(f"[info] Rheometer : {rheo_path}")

    # Step 1: Fit HB parameters from rheometer CSV
    print(f"\n{'='*60}")
    print("  HB Fitting from Rheometer Data")
    print(f"{'='*60}")

    calib_out.mkdir(parents=True, exist_ok=True)
    ret = run([
        sys.executable, str(ROOT / "FlowCurve" / "hb_fit.py"),
        "--file",  str(rheo_path),
        "--range", str(args.rheo_range[0]), str(args.rheo_range[1]),
        "--out",   str(rheo_json),
    ])
    if ret != 0:
        sys.exit("[error] HB fitting failed — see output above")

    with open(rheo_json) as f:
        params = json.load(f)

    # hb_fit.py outputs SI units (Pa). The simulation operates in CGS units
    # (dyne/cm²), where 1 Pa = 10 dyne/cm². Multiply η and σ_y by 10.
    eta_si     = params["eta"]
    n          = params["n"]
    sigma_y_si = params["sigma_y"]
    eta        = eta_si * 10.0
    sigma_y    = sigma_y_si * 10.0

    print(f"\n[info] HB parameters from rheometer (SI):")
    print(f"         η      = {eta_si:.4f}  Pa·s^n")
    print(f"         n      = {n:.4f}")
    print(f"         σ_y    = {sigma_y_si:.4f}  Pa")
    print(f"         R²     = {params['r2']:.6f}")
    print(f"\n[info] Converted to CGS for simulation (×10):")
    print(f"         η      = {eta:.4f}  dyne·s/cm²")
    print(f"         σ_y    = {sigma_y:.4f}  dyne/cm²")

    # Step 2: Run simulation with rheometer parameters
    print(f"\n{'='*60}")
    print("  MPM Simulation (rheometer parameters)")
    print(f"{'='*60}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = ROOT / "Simulation" / "results" / f"validation_{material_name}_{timestamp}"

    ret = run([
        sys.executable, str(ROOT / "Simulation" / "main.py"),
        "--eta",         str(eta),
        "--n",           str(n),
        "--sigma_y",     str(sigma_y),
        "--ref",         str(data_dir),
        "--camera_xml",  str(camera_xml),
        "--out_dir",     str(out_dir),
        "--diff_amplify", str(args.diff_amplify),
    ], cwd=ROOT / "Simulation")
    if ret != 0:
        sys.exit("[error] Simulation failed — see output above")

    # Summary
    print(f"\n{'='*60}")
    print("  Validation complete")
    print(f"{'='*60}")
    print(f"  Rheometer params : {rheo_json}")
    print(f"  Simulation output: {out_dir}/")
    print(f"  Check snapdiff_*.png — should be near-zero if calibration is correct.")


if __name__ == "__main__":
    main()
