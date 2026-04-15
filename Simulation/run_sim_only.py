"""
run_sim_only.py
===============
Run MPM simulation for a given (n, eta, sigma_y, W, H) and output
flow distance CSV. No rendering, no camera needed.

Usage:
    python run_sim_only.py --n 0.6191 --eta 21.0 --sigma_y 53.3 --W 2.5 --H 3.0
    python run_sim_only.py --n 0.6191 --eta 21.0 --sigma_y 53.3 --W 7.0 --H 7.0
"""

import argparse
import csv
import os
import sys
import traceback

import taichi as ti

ti.init(arch=ti.gpu, offline_cache=True, default_fp=ti.f32, default_ip=ti.i32)

from config.config import XML_TEMPLATE_PATH
from simulation.taichi import MPMSimulator


def main():
    p = argparse.ArgumentParser(description="Run MPM simulation (no rendering)")
    p.add_argument("--n",       type=float, required=True)
    p.add_argument("--eta",     type=float, required=True)
    p.add_argument("--sigma_y", type=float, required=True)
    p.add_argument("--W",       type=float, required=True, help="Container width")
    p.add_argument("--H",       type=float, required=True, help="Container height")
    p.add_argument("--out_csv", type=str,   default=None,
                   help="Output CSV path (default: stdout)")
    args = p.parse_args()

    print(f"Parameters: n={args.n}, eta={args.eta}, sigma_y={args.sigma_y}, "
          f"W={args.W}, H={args.H}")

    # Initialize simulator
    simulator = MPMSimulator(XML_TEMPLATE_PATH)
    simulator.configure_geometry(width=args.W, height=args.H)

    # Run simulation
    out_dir = f"_tmp_sim_{args.W}_{args.H}"
    os.makedirs(out_dir, exist_ok=True)

    try:
        displacements = simulator.run_simulation(
            n=args.n, eta=args.eta, sigma_y=args.sigma_y,
            output_dir=out_dir,
        )
        print(f"Displacements: {displacements}")

        # Output
        row = [args.n, args.eta, args.sigma_y, args.W, args.H] + \
              [displacements[i] if i < len(displacements) else 0 for i in range(8)]

        if args.out_csv:
            with open(args.out_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["n", "eta", "sigma_y", "width", "height",
                            "x_01", "x_02", "x_03", "x_04", "x_05", "x_06", "x_07", "x_08"])
                w.writerow(row)
            print(f"Saved to {args.out_csv}")
        else:
            print(",".join([f"{v:.6f}" if isinstance(v, float) else str(v) for v in row]))

    except Exception as e:
        print(f"Simulation failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup tmp dir
        import shutil
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
