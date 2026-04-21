#!/usr/bin/env python3
"""
test_pipeline.py — ViRheometry validation pipeline
====================================================
End-to-end validation: extract → simulate → diff (optionally 3-way vs PS).

Steps
-----
  Step 1  Extract config_00~08.png + camera_params.xml from the experiment
          video.  Calls prepare_configs.py internally.
  Step 2  Run MPM simulation with the provided HB parameters.
          Calls Simulation/main.py via subprocess after patching
          Simulation/config/setting.xml with the calibrated camera.
  Step 3  Diff each Auto cfg_i against Sim cfg_i (same index).
  Step 4  [Optional] If --ps_dir is provided, also produce 3-way diff:
            diff_sim_vs_auto/   diff_sim_vs_ps/   diff_auto_vs_ps/
          + a single overview_3way.png that side-by-sides everything.

Indexing convention
-------------------
  Sim cfg_i  = synthetic state at t = i · (1/24 s),  cfg_0 = pre-dam-break
  Auto cfg_i = video frame extracted at the same time (after frame0 shift)
  PS   cfg_i = manual Photoshop extraction at the same time
  All comparisons are done at the SAME index (cfg_0 vs cfg_0, ...).

Usage
-----
  # Minimal (extract + sim + Auto-vs-Sim diff):
  python Calibration/test_pipeline.py \\
      --data_dir data/ref_Tonkatsu_2.5_3.0_1/1st \\
      --n 0.62 --eta 20.0 --sigma_y 52.58

  # Full 3-way validation (also compares against manual PS extraction):
  python Calibration/test_pipeline.py \\
      --data_dir data/ref_Tonkatsu_2.5_3.0_1/1st \\
      --n 0.62 --eta 20.0 --sigma_y 52.58 \\
      --ps_dir data/ref_Tonkatsu_2.5_3.0_1/1st/1st_ps/exps/grays

  # Reuse already-extracted configs:
  python Calibration/test_pipeline.py \\
      --data_dir data/ref_Tonkatsu_2.5_3.0_1/1st \\
      --n 0.62 --eta 20.0 --sigma_y 52.58 \\
      --skip_extract

  # Reuse polygons (no GUI redraw):
  python Calibration/test_pipeline.py \\
      --data_dir data/ref_Tonkatsu_2.5_3.0_1/1st \\
      --n 0.62 --eta 20.0 --sigma_y 52.58 \\
      --extract_args "--roi-poly r1.npy --roi-poly-00 r2.npy"

Output
------
  <data_dir>/test_output/
    configs/                    cfg_00..08.png + camera_params.xml + ...
    sim/<n>_<eta>_<sy>/         MPM simulation result + rendered cfg_XX.png
    diff_sim_vs_auto/           diff_00..08.png  (always)
    diff_sim_vs_ps/             diff_00..08.png  (only with --ps_dir)
    diff_auto_vs_ps/            diff_00..08.png  (only with --ps_dir)
    overview_3way.png           combined 9×N grid (only with --ps_dir)
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


# ── Project layout ────────────────────────────────────────────────────────────
_HERE      = Path(__file__).parent.resolve()           # Calibration/
_PROJ_ROOT = _HERE.parent.resolve()                    # New_ViRheometry/
_SIM_MAIN  = _PROJ_ROOT / "Simulation" / "main.py"
_PREP      = _HERE / "prepare_configs.py"
_SIM_CFG   = _PROJ_ROOT / "Simulation" / "config" / "setting.xml"


# ──────────────────────────────────────────────────────────────────────────────
#  Step 1 — Extraction (delegates to prepare_configs.py)
# ──────────────────────────────────────────────────────────────────────────────

def run_extract(data_dir: str, out_dir: str, extra: list[str]) -> bool:
    print("\n" + "=" * 60)
    print("  STEP 1  Extract configs + camera_params.xml")
    print("=" * 60)
    cmd = [sys.executable, str(_PREP),
           "--data_dir", data_dir,
           "--out_dir",  out_dir] + extra
    print("[cmd]", " ".join(cmd), "\n")
    return subprocess.run(cmd, cwd=str(_PROJ_ROOT)).returncode == 0


# ──────────────────────────────────────────────────────────────────────────────
#  Step 2 — Simulation (delegates to Simulation/main.py)
#
#  The simulation reads camera_params.xml from <ref_dir> directly and passes it
#  to the renderer via MPMEmulator(camera_params=...).  setting.xml is NOT
#  patched — those two XMLs have different purposes and must stay separated.
# ──────────────────────────────────────────────────────────────────────────────

def run_simulation(ref_dir: str, n: float, eta: float, sigma_y: float,
                   sim_out_dir: str) -> bool:
    print("\n" + "=" * 60)
    print(f"  STEP 2  MPM simulation  (n={n}, η={eta}, σy={sigma_y}  CGS)")
    print("=" * 60)

    cam_xml = os.path.join(ref_dir, "camera_params.xml")
    if not os.path.isfile(cam_xml):
        print(f"[WARN] camera_params.xml not found in {ref_dir} — "
              f"renderer will fall back to setting.xml defaults.")
    else:
        print(f"[camera] {cam_xml}  (passed to renderer via main.py)")

    cmd = [sys.executable, str(_SIM_MAIN),
           "--n",       str(n),
           "--eta",     str(eta),
           "--sigma_y", str(sigma_y),
           "--ref",     ref_dir,
           "--out_dir", sim_out_dir]
    print("[cmd]", " ".join(cmd), "\n")

    return subprocess.run(cmd, cwd=str(_PROJ_ROOT / "Simulation")).returncode == 0


def _find_sim_subdir(sim_dir: str) -> str | None:
    """Find the n.nn_eta.nn_sigma.nn/ sub-directory inside sim_dir."""
    pat = re.compile(r"^\d+\.\d+_\d+\.\d+_\d+\.\d+$")
    for entry in sorted(Path(sim_dir).iterdir()):
        if entry.is_dir() and pat.match(entry.name):
            return str(entry)
    return None


# ──────────────────────────────────────────────────────────────────────────────
#  Step 3 / 4 — Diff analysis
# ──────────────────────────────────────────────────────────────────────────────

def _load_bin(path: str, target_shape=None) -> np.ndarray | None:
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    if target_shape is not None and im.shape != target_shape:
        im = cv2.resize(im, (target_shape[1], target_shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    return (im < 128).astype(np.uint8)


def _save_diff(ref: np.ndarray, test: np.ndarray, out_path: str) -> float:
    """Save both a colour-coded diff image AND a black/white XOR diff.
    Returns IoU.

    Given out_path `.../diff_XX.png`, this writes:
        .../diff_XX.png       (colour)
        .../diff_XX_bw.png    (black-and-white XOR, legacy)

    Colour legend (BGR):
        WHITE  (255,255,255) : neither has fluid (both background)
        GRAY   (128,128,128) : both agree — fluid region (intersection)
        RED    (  0,  0,255) : `ref` (a) has fluid, `test` (b) does not → a_only
        BLUE   (255,  0,  0) : `test` (b) has fluid, `ref` (a) does not → b_only

    For the Sim-vs-Auto diff where ref=Sim, test=Auto:
        RED  = Sim flowed further / is thicker than the real video
        BLUE = Video flowed further / is thicker than the sim

    B/W legend (legacy):
        BLACK (0)   : pixels that agree
        WHITE (255) : pixels that differ (XOR)
    """
    H, W = ref.shape
    # ── colour diff ──
    out_color = np.full((H, W, 3), 255, dtype=np.uint8)     # start white
    both = (ref & test).astype(bool)
    a_only = (ref & ~test).astype(bool)
    b_only = (test & ~ref).astype(bool)
    out_color[both]   = (128, 128, 128)
    out_color[a_only] = (0,   0,   255)
    out_color[b_only] = (255, 0,   0)
    cv2.imwrite(out_path, out_color)

    # ── b/w XOR diff (kept for legacy tools) ──
    root, ext = os.path.splitext(out_path)
    bw_path = root + "_bw" + ext
    out_bw = ((ref ^ test).astype(np.uint8)) * 255
    cv2.imwrite(bw_path, out_bw)

    u = (ref | test).sum()
    return float((ref & test).sum() / u) if u else 0.0


def diff_pair(name_a: str, dir_a: str,
              name_b: str, dir_b: str,
              out_dir: str, n_configs: int) -> dict:
    """Diff cfg_i in dir_a against cfg_i in dir_b at the SAME index."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"\n  {name_a:<8} vs {name_b:<8}  →  {out_dir}")
    print(f"  {'cfg':>4} {'IoU':>8} {name_a+'_fg':>10} {name_b+'_fg':>10} "
          f"{'a_only':>8} {'b_only':>8} {'bias':>10}")
    ious, sizes_a, sizes_b = [], [], []
    for i in range(n_configs):
        pa = os.path.join(dir_a, f"config_{i:02d}.png")
        pb = os.path.join(dir_b, f"config_{i:02d}.png")
        a  = _load_bin(pa)
        if a is None:
            continue
        b = _load_bin(pb, a.shape)
        if b is None:
            continue
        out = os.path.join(out_dir, f"diff_{i:02d}.png")
        iou = _save_diff(a, b, out)
        ious.append(iou); sizes_a.append(int(a.sum())); sizes_b.append(int(b.sum()))
        ao = int((a & ~b).sum()); bo = int((b & ~a).sum())
        # Bias arrow: who flowed further / is larger
        if ao + bo == 0:
            bias = "="
        elif ao >= 2 * max(bo, 1):
            bias = f"{name_a}++"
        elif bo >= 2 * max(ao, 1):
            bias = f"{name_b}++"
        elif ao > bo:
            bias = f"{name_a}+"
        elif bo > ao:
            bias = f"{name_b}+"
        else:
            bias = "="
        print(f"  {i:>4} {iou:>8.4f} {int(a.sum()):>10d} {int(b.sum()):>10d} "
              f"{ao:>8d} {bo:>8d} {bias:>10}")
    if ious:
        print(f"  {'mean':>4} {np.mean(ious):>8.4f}")
    return {"ious": ious, "sizes_a": sizes_a, "sizes_b": sizes_b}


def make_overview(out_path: str, auto_dir: str, sim_dir: str,
                  ps_dir: str, base_dir: Path, n_configs: int = 9) -> None:
    """9 rows × 6 cols overview: Auto | PS | Sim | S-A | S-P | A-P."""
    th, tw = 135, 240
    lh = 30
    cols = ['Auto', 'PS', 'Sim', 'Sim-Auto', 'Sim-PS', 'Auto-PS']
    out = np.ones((n_configs * th + lh, len(cols) * tw, 3), dtype=np.uint8) * 255
    for j, lbl in enumerate(cols):
        cv2.putText(out, lbl, (j * tw + 10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    def tile(p):
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        return cv2.resize(im, (tw, th)) if im is not None else None

    for i in range(n_configs):
        sources = [
            f"{auto_dir}/config_{i:02d}.png",
            f"{ps_dir}/config_{i:02d}.png",
            f"{sim_dir}/config_{i:02d}.png",
            f"{base_dir}/diff_sim_vs_auto/diff_{i:02d}.png",
            f"{base_dir}/diff_sim_vs_ps/diff_{i:02d}.png",
            f"{base_dir}/diff_auto_vs_ps/diff_{i:02d}.png",
        ]
        for j, src in enumerate(sources):
            im = tile(src)
            if im is not None:
                out[lh + i*th:lh + (i+1)*th, j*tw:(j+1)*tw] = im
    cv2.imwrite(out_path, out)
    print(f"\n[overview] {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ViRheometry test pipeline: extract → simulate → diff",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_dir", required=True, metavar="DIR",
                   help="Experiment dir (must contain *.mov, *.JPG, settings.xml)")

    hb = p.add_argument_group("HB parameters (CGS)")
    hb.add_argument("--n",       type=float, required=True, help="Power-law index")
    hb.add_argument("--eta",     type=float, required=True, help="Viscosity (dyne·s/cm²)")
    hb.add_argument("--sigma_y", type=float, required=True, help="Yield stress (dyne/cm²)")

    p.add_argument("--out_dir", type=str, default=None,
                   help="Base output dir (default: <data_dir>/test_output/)")

    p.add_argument("--ps_dir", type=str, default=None, metavar="DIR",
                   help="Optional manual-PS extraction dir for 3-way comparison "
                        "(must contain config_00..08.png at the SAME indexing as Auto)")
    p.add_argument("--n_configs", type=int, default=9,
                   help="Number of configs to compare (default 9)")

    p.add_argument("--skip_extract", action="store_true",
                   help="Reuse existing <out_dir>/configs/ (skip Step 1)")
    p.add_argument("--skip_sim", action="store_true",
                   help="Reuse existing <out_dir>/sim/ (skip Step 2)")
    p.add_argument("--extract_args", type=str, default="",
                   help="Extra args forwarded verbatim to prepare_configs.py")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = str(Path(args.data_dir).resolve())
    base_out = Path(args.out_dir) if args.out_dir else Path(data_dir) / "test_output"
    cfg_dir  = base_out / "configs"
    sim_dir  = base_out / "sim"
    base_out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\n  ViRheometry Test Pipeline\n{'='*60}")
    print(f"  data_dir : {data_dir}")
    print(f"  HB       : n={args.n}, η={args.eta}, σy={args.sigma_y}  (CGS)")
    print(f"  output   : {base_out}")
    if args.ps_dir:
        print(f"  ps_dir   : {args.ps_dir}  (3-way mode)")

    # ── Step 1 — Extract ─────────────────────────────────────────────────────
    if not args.skip_extract:
        extra = args.extract_args.split() if args.extract_args else []
        if not run_extract(data_dir, str(cfg_dir), extra):
            sys.exit("[ERROR] Step 1 (extraction) failed")
    else:
        print(f"\n[Step 1] skipped — using {cfg_dir}")

    # Sanity checks
    missing = [f"config_{i:02d}.png" for i in range(args.n_configs)
               if not (cfg_dir / f"config_{i:02d}.png").exists()]
    if missing:
        sys.exit(f"[ERROR] Missing configs: {missing}")
    if not (cfg_dir / "camera_params.xml").exists():
        sys.exit(f"[ERROR] Missing camera_params.xml in {cfg_dir}")

    # ── Step 2 — Simulate ────────────────────────────────────────────────────
    if not args.skip_sim:
        if not run_simulation(str(cfg_dir), args.n, args.eta, args.sigma_y,
                              str(sim_dir)):
            sys.exit("[ERROR] Step 2 (simulation) failed")
    else:
        print(f"\n[Step 2] skipped — using {sim_dir}")

    sim_sub = _find_sim_subdir(str(sim_dir))
    if sim_sub is None:
        sys.exit(f"[ERROR] No simulation sub-dir in {sim_dir}")

    # ── Step 3 — Diff Sim vs Auto (always) ───────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3  Diff analysis")
    print("=" * 60)

    sa = diff_pair("Sim", sim_sub, "Auto", str(cfg_dir),
                   str(base_out / "diff_sim_vs_auto"), args.n_configs)
    sa_mean = float(np.mean(sa["ious"])) if sa["ious"] else 0.0

    # ── Step 4 — Optional 3-way PS comparison ───────────────────────────────
    sp_mean = ap_mean = None
    if args.ps_dir:
        sp = diff_pair("Sim", sim_sub, "PS", args.ps_dir,
                       str(base_out / "diff_sim_vs_ps"), args.n_configs)
        ap = diff_pair("Auto", str(cfg_dir), "PS", args.ps_dir,
                       str(base_out / "diff_auto_vs_ps"), args.n_configs)
        sp_mean = float(np.mean(sp["ious"])) if sp["ious"] else 0.0
        ap_mean = float(np.mean(ap["ious"])) if ap["ious"] else 0.0
        make_overview(str(base_out / "overview_3way.png"),
                      str(cfg_dir), sim_sub, args.ps_dir,
                      base_out, args.n_configs)

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}\n  Summary\n{'='*60}")
    print(f"  Mean IoU (Sim vs Auto) : {sa_mean:.4f}")
    if sp_mean is not None:
        print(f"  Mean IoU (Sim vs PS)   : {sp_mean:.4f}")
        print(f"  Mean IoU (Auto vs PS)  : {ap_mean:.4f}")
    verdict = ("✓ PASS" if sa_mean >= 0.95 else
               "△ MARGINAL" if sa_mean >= 0.90 else "✗ FAIL")
    print(f"  Verdict (Sim vs Auto)  : {verdict}  (threshold IoU ≥ 0.95)")
    print(f"\n  Output : {base_out}/")


if __name__ == "__main__":
    main()
