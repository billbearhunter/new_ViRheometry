#!/usr/bin/env python3
import argparse
import random
from datetime import datetime
from pathlib import Path

# ==========================================
# Physical parameter ranges (Provided by user)
# ==========================================
MIN_ETA = 0.001
MAX_ETA = 300.0

MIN_N = 0.3
MAX_N = 1.0

MIN_SIGMA_Y = 0.0
MAX_SIGMA_Y = 400.0

# ==========================================
# Geometry ranges (kept as mm integers for consistency)
# ==========================================
# Note: 20mm = 2.0cm, 70mm = 7.0cm
DEFAULT_MIN_MM = 20
DEFAULT_MAX_MM = 70


def sample_setup1_mm(min_mm: int, max_mm: int):
    """
    Sample the first experimental setup (width, height) in millimeters.
    (Integers, inclusive range)
    """
    H_mm = random.randint(min_mm, max_mm)
    W_mm = random.randint(min_mm, max_mm)
    return W_mm, H_mm


def sample_physical_params():
    """
    Sample physical parameters (eta, n, sigma_y) using uniform distribution.
    """
    eta = random.uniform(MIN_ETA, MAX_ETA)
    n = random.uniform(MIN_N, MAX_N)
    sigma_y = random.uniform(MIN_SIGMA_Y, MAX_SIGMA_Y)
    return eta, n, sigma_y


def mm_to_cm(value_mm: int) -> float:
    """Convert millimeters to centimeters (with 0.1 cm resolution)."""
    return round(value_mm * 0.1, 2)


def main():
    parser = argparse.ArgumentParser(
        description="Propose the first dam-break setup (Setup 1) "
                    "with random Geometry AND Physical parameters."
    )
    parser.add_argument(
        "--min-mm",
        type=int,
        default=DEFAULT_MIN_MM,
        help=f"Minimum width/height in mm (default: {DEFAULT_MIN_MM} mm)",
    )
    parser.add_argument(
        "--max-mm",
        type=int,
        default=DEFAULT_MAX_MM,
        help=f"Maximum width/height in mm (default: {DEFAULT_MAX_MM} mm)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="setup1.txt",
        help="Path to save the proposed setup parameters (default: setup1.txt)",
    )

    args = parser.parse_args()

    # Use current time as seed
    random.seed(datetime.now().timestamp())

    # 1. Sample Geometry (Integer mm)
    W_mm, H_mm = sample_setup1_mm(args.min_mm, args.max_mm)
    W_cm = mm_to_cm(W_mm)
    H_cm = mm_to_cm(H_mm)

    # 2. Sample Physical Parameters (Float)
    eta, n, sigma_y = sample_physical_params()

    # Print summary
    print("===== Proposed Setup 1 (dam-break + material) =====")
    print("--- Geometry ---")
    print(f"Width  W1 = {W_cm:.2f} cm (raw {W_mm} mm)")
    print(f"Height H1 = {H_cm:.2f} cm (raw {H_mm} mm)")
    print("--- Material Properties (Random Sampled) ---")
    print(f"Eta     = {eta:.6f} Pa.s")
    print(f"n       = {n:.6f}")
    print(f"Sigma_Y = {sigma_y:.6f} Pa")
    print("---------------------------------------------------")
    print(f"Generating file: {args.out}")
    print()

    # Save to a text file
    out_path = Path(args.out)
    out_path.write_text(
        f"# Setup 1 (dam-break geometry + physical params)\n"
        f"W1_cm {W_cm:.2f}\n"
        f"H1_cm {H_cm:.2f}\n"
        f"W1_mm {W_mm}\n"
        f"H1_mm {H_mm}\n"
        f"eta {eta:.8f}\n"
        f"n {n:.8f}\n"
        f"sigma_y {sigma_y:.8f}\n"
    )
    print(f"Parameters saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()