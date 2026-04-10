#!/usr/bin/env python3
"""
run_pipeline.py — End-to-end ViRheometry pipeline
===================================================
Runs all steps for a single material dataset in sequence:

  Step 1  Camera calibration        Calibration/pipeline.py
  Step 2  Flow distance extraction  Calibration/extract_flow_distance.py
  Step 3  HB parameter optimization Optimization/optimize_1setup.py
  Step 4  MPM simulation (opt-in)   Simulation/main.py

Each step writes to its own directory. data/ is never modified.

Validation mode (--rheo):
  Bypasses Step 3. Fits HB parameters directly from a rheometer CSV
  (FlowCurve/hb_fit.py), then runs Step 4 with those parameters.
  The simulation diff should be near-zero if calibration is correct.

Usage:
    # Standard pipeline (Steps 1-3, optional 4)
    python3 run_pipeline.py --data data/ref_Tonkatsu_6.7_3.5_1

    # Validation mode: use rheometer ground-truth parameters
    python3 run_pipeline.py \\
        --data data/ref_Tonkatsu_6.7_3.5_1 \\
        --rheo FlowCurve/Rheo_Data/tonkatsu_20230113_2000_23C.csv \\
        --simulate

    # Skip already-completed steps (for debugging)
    python3 run_pipeline.py --data data/ref_Tonkatsu_6.7_3.5_1 \\
        --skip-calibration --skip-extraction
"""

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


# ── helpers ───────────────────────────────────────────────────────────────────

def step_banner(n, title: str):
    print(f"\n{'='*60}")
    print(f"  Step {n}: {title}")
    print(f"{'='*60}")


def run(cmd: list, cwd: Path = ROOT) -> int:
    print("  $", " ".join(str(c) for c in cmd))
    print()
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def find_calib_image(data_dir: Path) -> Path:
    for ext in ("*.JPG", "*.jpg", "*.JPEG", "*.jpeg", "*.PNG", "*.png"):
        hits = [p for p in data_dir.glob(ext) if not p.name.startswith("config_")]
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No calibration image (non-config_*.JPG/PNG) found in {data_dir}")


def read_settings(data_dir: Path):
    path = data_dir / "settings.xml"
    if not path.exists():
        sys.exit(f"[error] settings.xml not found in {data_dir}")
    setup = ET.parse(path).getroot().find("setup")
    if setup is None:
        sys.exit("[error] <setup> tag not found in settings.xml")
    return float(setup.attrib["W"]), float(setup.attrib["H"])


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end ViRheometry pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- required ----
    parser.add_argument("--data", required=True,
                        help="Material data directory (e.g. data/ref_Tonkatsu_6.7_3.5_1)")

    # ---- validation mode ----
    parser.add_argument("--rheo", default=None,
                        help="Rheometer CSV path. Enables validation mode: fits HB parameters "
                             "from measured data and uses them directly for simulation, "
                             "bypassing CMA-ES optimization. Requires --simulate.")
    parser.add_argument("--rheo-range", nargs=2, type=int, default=[5, 19],
                        metavar=("START", "END"),
                        help="Row index range for HB fitting (default: 5 19)")

    # ---- step 3 options ----
    parser.add_argument("--moe_dir", default="Optimization/moe_workspace5",
                        help="MoE model directory (default: Optimization/moe_workspace5)")
    parser.add_argument("--strategy", default="topk",
                        choices=["topk", "threshold", "adaptive", "all"],
                        help="Expert selection strategy (default: topk)")
    parser.add_argument("--topk", type=int, default=2,
                        help="Number of experts for 'topk' strategy (default: 2)")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Weight threshold for 'threshold' strategy (default: 0.01)")
    parser.add_argument("--confidence_threshold", type=float, default=0.7,
                        help="Confidence threshold for 'adaptive' strategy (default: 0.7)")
    parser.add_argument("--max_experts", type=int, default=5,
                        help="Max experts for 'threshold' strategy (default: 5)")
    parser.add_argument("--maxiter", type=int, default=700,
                        help="CMA-ES max iterations (default: 700)")

    # ---- step 4 ----
    parser.add_argument("--simulate", action="store_true",
                        help="Run MPM simulation (Step 4)")
    parser.add_argument("--eta", type=float, default=None,
                        help="Override η for simulation")
    parser.add_argument("--n", type=float, default=None,
                        help="Override n for simulation")
    parser.add_argument("--sigma_y", type=float, default=None,
                        help="Override σ_y for simulation")

    # ---- skip flags ----
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Skip Step 1 (use existing Calibration/results/<name>/)")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip Step 2 (use existing flow_distances.json)")
    parser.add_argument("--skip-optimization", action="store_true",
                        help="Skip Step 3 (requires --eta/--n/--sigma_y or --rheo for Step 4)")

    args = parser.parse_args()

    # ── validation: --rheo implies --skip-optimization --simulate ────────────
    if args.rheo and not args.simulate:
        print("[info] --rheo enables validation mode. Adding --simulate automatically.")
        args.simulate = True

    # ── derived paths ────────────────────────────────────────────
    data_dir      = (ROOT / args.data).resolve()
    material_name = data_dir.name
    calib_out     = ROOT / "Calibration" / "results" / material_name
    camera_xml    = calib_out / "camera_params.xml"
    flow_json     = calib_out / "flow_distances.json"
    moe_dir       = (ROOT / args.moe_dir).resolve()

    W, H = read_settings(data_dir)
    print(f"\n[info] Material : {material_name}")
    print(f"[info] Geometry : W={W} cm, H={H} cm")
    print(f"[info] Data     : {data_dir}")
    print(f"[info] Outputs  : {calib_out}")
    if args.rheo:
        print(f"[info] Mode     : validation (rheometer parameters)")
    else:
        print(f"[info] Mode     : inference (CMA-ES optimization)")

    # ── Step 1: Camera Calibration ───────────────────────────────
    if not args.skip_calibration:
        step_banner(1, "Camera Calibration")
        calib_img = find_calib_image(data_dir)
        target    = data_dir / "config_00.png"
        if not target.exists():
            sys.exit(f"[error] Calibration target not found: {target}")

        ret = run([
            sys.executable, str(ROOT / "Calibration" / "pipeline.py"),
            "--calib_img", str(calib_img),
            "--target",    str(target),
            "--out_dir",   str(calib_out),
        ])
        if ret != 0:
            sys.exit("[error] Step 1 failed — see output above")
    else:
        print(f"\n[skip] Step 1 — using: {calib_out}")
        if not camera_xml.exists():
            sys.exit(f"[error] --skip-calibration set but {camera_xml} not found")

    # ── Step 2: Flow Distance Extraction ─────────────────────────
    if not args.skip_extraction:
        step_banner(2, "Flow Distance Extraction")
        calib_out.mkdir(parents=True, exist_ok=True)

        ret = run([
            sys.executable, str(ROOT / "Calibration" / "extract_flow_distance.py"),
            "--dir",          str(data_dir),
            "--camera-xml",   str(camera_xml),
            "--output-csv",   str(calib_out / "flow_distances.csv"),
            "--output-json",  str(flow_json),
            "--monotonic", "--unit", "cm", "--print-dis1",
        ])
        if ret != 0:
            sys.exit("[error] Step 2 failed — see output above")
    else:
        print(f"\n[skip] Step 2 — using: {flow_json}")
        if not flow_json.exists():
            sys.exit(f"[error] --skip-extraction set but {flow_json} not found")

    # ── Step 3: HB Parameters ────────────────────────────────────
    best_eta = best_n = best_sigma_y = None

    if args.rheo:
        # Validation mode: fit from rheometer CSV
        step_banner("3 (validation)", "HB Fitting from Rheometer Data")
        rheo_path = (ROOT / args.rheo).resolve()
        if not rheo_path.exists():
            sys.exit(f"[error] Rheometer CSV not found: {rheo_path}")

        rheo_json = calib_out / "rheo_params.json"
        ret = run([
            sys.executable, str(ROOT / "FlowCurve" / "hb_fit.py"),
            "--file",  str(rheo_path),
            "--range", str(args.rheo_range[0]), str(args.rheo_range[1]),
            "--out",   str(rheo_json),
        ])
        if ret != 0:
            sys.exit("[error] HB fitting failed — see output above")

        with open(rheo_json) as f:
            rheo_params = json.load(f)
        best_eta     = rheo_params["eta"]
        best_n       = rheo_params["n"]
        best_sigma_y = rheo_params["sigma_y"]
        print(f"\n[info] Rheometer HB parameters:")
        print(f"         η      = {best_eta:.4f}  Pa·s^n")
        print(f"         n      = {best_n:.4f}")
        print(f"         σ_y    = {best_sigma_y:.4f}  Pa")
        print(f"         R²     = {rheo_params['r2']:.6f}")

    elif not args.skip_optimization:
        # Inference mode: CMA-ES optimization
        step_banner(3, "HB Parameter Optimization (CMA-ES)")

        with open(flow_json) as f:
            flow_data = json.load(f)
        distances = flow_data["distances"]
        print(f"  Flow distances: {[round(d, 4) for d in distances]}\n")

        opt_cmd = [
            sys.executable, str(ROOT / "Optimization" / "optimize_1setup.py"),
            "--moe_dir", str(moe_dir),
            "-W1", str(W), "-H1", str(H),
            "-dis1", *[str(d) for d in distances],
            "--strategy", args.strategy,
            "--maxiter", str(args.maxiter),
        ]
        if args.strategy == "topk":
            opt_cmd += ["--topk", str(args.topk)]
        elif args.strategy == "threshold":
            opt_cmd += ["--threshold", str(args.threshold),
                        "--max_experts", str(args.max_experts)]
        elif args.strategy == "adaptive":
            opt_cmd += ["--confidence_threshold", str(args.confidence_threshold)]

        ret = run(opt_cmd, cwd=ROOT / "Optimization")
        if ret != 0:
            sys.exit("[error] Step 3 failed — see output above")

        # Read result from latest result directory
        result_dirs = sorted(
            (ROOT / "Optimization").glob(f"result_setup1_{args.strategy}_*"),
            reverse=True,
        )
        if result_dirs:
            result_txt = result_dirs[0] / "setup1_result.txt"
            if result_txt.exists():
                for line in result_txt.read_text().splitlines():
                    if "eta" in line.lower():
                        best_eta = float(line.split("=")[-1].strip())
                    elif line.lower().startswith("n ") or line.lower().startswith("n="):
                        best_n = float(line.split("=")[-1].strip())
                    elif "sigma" in line.lower():
                        best_sigma_y = float(line.split("=")[-1].strip())
    else:
        print(f"\n[skip] Step 3")
        best_eta, best_n, best_sigma_y = args.eta, args.n, args.sigma_y
        if args.simulate and None in (best_eta, best_n, best_sigma_y):
            sys.exit("[error] --skip-optimization with --simulate requires --eta, --n, --sigma_y")

    # ── Step 4: MPM Simulation (optional) ────────────────────────
    if args.simulate:
        eta     = args.eta     if args.eta     is not None else best_eta
        n       = args.n       if args.n       is not None else best_n
        sigma_y = args.sigma_y if args.sigma_y is not None else best_sigma_y

        if None in (eta, n, sigma_y):
            print("\n[warn] Step 4 skipped — no HB parameters available.")
            print("       Run manually:")
            print(f"       cd Simulation && python3 main.py \\")
            print(f"           --eta <η> --n <n> --sigma_y <σ_y> \\")
            print(f"           --ref ../{args.data} \\")
            print(f"           --camera_xml {camera_xml}")
        else:
            step_banner(4, "MPM Simulation")
            ret = run([
                sys.executable, str(ROOT / "Simulation" / "main.py"),
                "--eta",        str(eta),
                "--n",          str(n),
                "--sigma_y",    str(sigma_y),
                "--ref",        str(data_dir),
                "--camera_xml", str(camera_xml),
            ], cwd=ROOT / "Simulation")
            if ret != 0:
                sys.exit("[error] Step 4 failed — see output above")
    else:
        print(f"\n[info] Step 4 (simulation) skipped. Add --simulate to run it.")

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Pipeline complete")
    print(f"{'='*60}")
    print(f"  Calibration outputs : {calib_out}/")
    if args.rheo:
        print(f"  Rheometer params    : {calib_out}/rheo_params.json")
    else:
        print(f"  Optimization result : Optimization/result_setup1_*/")
    if args.simulate:
        print(f"  Simulation result   : Simulation/results/run_*/")
        if args.rheo:
            print(f"  [check] snapdiff_*.png should be near-zero if calibration is correct")


if __name__ == "__main__":
    main()
