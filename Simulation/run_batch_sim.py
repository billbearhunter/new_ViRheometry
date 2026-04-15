"""
run_batch_sim.py
================
Run MPM simulations for parameter sets from a CSV file.
Each row: n, eta, sigma_y, width, height
Outputs flow distances to a new CSV.

Usage:
    python run_batch_sim.py --input candidates.csv --output results.csv
"""
import argparse, csv, os, sys, time, shutil
import numpy as np
import pandas as pd

import taichi as ti
ti.init(arch=ti.gpu, offline_cache=True, default_fp=ti.f32, default_ip=ti.i32)

from config.config import XML_TEMPLATE_PATH
from simulation.taichi import MPMSimulator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="Input CSV with n,eta,sigma_y,width,height")
    p.add_argument("--output", type=str, required=True, help="Output CSV with flow distances")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} parameter sets from {args.input}")

    simulator = MPMSimulator(XML_TEMPLATE_PATH)
    cols = ["n", "eta", "sigma_y", "width", "height",
            "x_01", "x_02", "x_03", "x_04", "x_05", "x_06", "x_07", "x_08"]

    # Write header
    with open(args.output, "w", newline="") as f:
        csv.writer(f).writerow(cols)

    done = 0
    for i, row in df.iterrows():
        n, eta, sy, W, H = float(row["n"]), float(row["eta"]), float(row["sigma_y"]), float(row["width"]), float(row["height"])
        tmp_dir = f"_tmp_batch_{i}"
        os.makedirs(tmp_dir, exist_ok=True)
        try:
            simulator.configure_geometry(width=W, height=H)
            disps = simulator.run_simulation(n=n, eta=eta, sigma_y=sy, output_dir=tmp_dir)
            out_row = [n, eta, sy, W, H] + [disps[j] if j < len(disps) else 0 for j in range(8)]
            with open(args.output, "a", newline="") as f:
                csv.writer(f).writerow(out_row)
            done += 1
        except Exception as e:
            print(f"  [{i+1}/{len(df)}] FAILED: {e}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if (done) % 50 == 0:
            print(f"  [{time.strftime('%H:%M:%S')}] {done}/{len(df)} done")

    print(f"\n[{time.strftime('%H:%M:%S')}] Complete: {done}/{len(df)} saved to {args.output}")


if __name__ == "__main__":
    main()
