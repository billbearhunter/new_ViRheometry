"""
collect_from_csv.py
===================
Run HeadlessSimulator for parameter sets from a CSV file.
Input CSV must have columns: n, eta, sigma_y, width, height

Usage:
    python collect_from_csv.py --input candidates.csv --output results.csv
"""
import argparse, csv, sys, time, logging
from pathlib import Path
import numpy as np
import pandas as pd

# Reuse HeadlessSimulator from collect_data
sys.path.insert(0, str(Path(__file__).parent))
from collect_data import HeadlessSimulator, _init_csv, _write_row, CSV_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SIM_DIR = Path(__file__).parent.parent / "Simulation"
XML_TEMPLATE = str(SIM_DIR / "config" / "setting.xml")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    log.info(f"Loaded {len(df)} parameter sets from {args.input}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    _init_csv(out, append=False)

    sim = HeadlessSimulator(XML_TEMPLATE)
    done = 0
    t0 = time.time()

    for i, row in df.iterrows():
        n, eta, sy = float(row["n"]), float(row["eta"]), float(row["sigma_y"])
        W, H = float(row["width"]), float(row["height"])
        try:
            diffs = sim.run(n, eta, sy, W, H)
            _write_row(out, [n, eta, sy, W, H] + diffs.tolist())
            done += 1
        except Exception as e:
            log.warning(f"[{i+1}/{len(df)}] FAILED: {e}")

        if (done) % 50 == 0 and done > 0:
            rate = done / (time.time() - t0)
            eta_min = (len(df) - done) / max(rate, 1e-6) / 60
            log.info(f"[{done}/{len(df)}] rate={rate:.1f}/s ETA={eta_min:.1f}min")

    log.info(f"Done. {done}/{len(df)} saved to {args.output}")


if __name__ == "__main__":
    main()
