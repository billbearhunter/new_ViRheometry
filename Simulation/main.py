import argparse
import os
import re
import csv
import traceback
import xml.etree.ElementTree as ET
import taichi as ti
from datetime import datetime
from PIL import Image
import numpy as np
from config.config import XML_TEMPLATE_PATH
from simulation.taichi import MPMSimulator
from scripts import MPM_Emulator

# 初始化 Taichi
ti.init(arch=ti.gpu, offline_cache=True, default_fp=ti.f32, default_ip=ti.i32)


# ──────────────────────────────────────────────
# XML 読み込みユーティリティ
# ──────────────────────────────────────────────

def load_settings(ref: str) -> dict:
    """ref/settings.xml から W / H を読み込む"""
    path = os.path.join(ref, "settings.xml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"settings.xml not found in ref: {ref}")
    root = ET.parse(path).getroot()
    setup = root.find("setup")
    if setup is None:
        raise ValueError("settings.xml: <setup> element not found")
    return {
        "width":  float(setup.attrib["W"]),
        "height": float(setup.attrib["H"]),
    }


def load_camera_params(ref: str) -> dict:
    """Load camera_params.xml. ref can be a file path or a directory."""
    if os.path.isdir(ref):
        path = os.path.join(ref, "camera_params.xml")
    else:
        path = ref
    if not os.path.isfile(path):
        raise FileNotFoundError(f"camera_params.xml not found: {path}")
    root = ET.parse(path).getroot()
    cam = root.find("camera")
    if cam is None:
        raise ValueError("camera_params.xml: <camera> element not found")

    eyepos = [float(v) for v in cam.attrib["eyepos"].split()]
    quat   = [float(v) for v in cam.attrib["quat"].split()]
    w, h   = [int(v)   for v in cam.attrib["window_size"].split()]
    fov    = float(cam.attrib["fov"])

    return {
        "eyepos":      eyepos,
        "quat":        quat,
        "window_size": (w, h),
        "fov":         fov,
    }


# ──────────────────────────────────────────────
# 差分処理
# ──────────────────────────────────────────────

def compute_and_save_diffs(results_root: str, ref: str, amplify: float = 5.0):
    """
    results_root 以下の config_XX.png を走査し、
    ref/config_XX.png と差分を計算して
    results_root/snapdiff_XX.png として保存する。
    """
    print("\n" + "=" * 60)
    print("Computing image diffs...")
    print(f"  Generated : {results_root}")
    print(f"  Reference : {ref}")
    print("=" * 60)

    # config_XX.png にマッチするパターン (XX は数字)
    pattern = re.compile(r'^config_(\d+)\.png$', re.IGNORECASE)

    matched = skipped = 0

    for dirpath, _, filenames in os.walk(results_root):
        for filename in sorted(filenames):
            m = pattern.match(filename)
            if not m:
                continue

            index    = m.group(1)           # 例: "01", "08"
            gen_path = os.path.join(dirpath, filename)
            ref_path = os.path.join(ref, filename)

            if not os.path.isfile(ref_path):
                print(f"  [SKIP] No reference for: {filename}")
                skipped += 1
                continue

            try:
                gen_img = Image.open(gen_path).convert("RGB")
                ref_img = Image.open(ref_path).convert("RGB")

                if gen_img.size != ref_img.size:
                    print(f"  [WARN] Size mismatch {filename}: "
                          f"gen={gen_img.size} ref={ref_img.size} → resizing ref")
                    ref_img = ref_img.resize(gen_img.size, Image.LANCZOS)

                diff_arr = np.abs(
                    np.array(gen_img, dtype=np.int16) -
                    np.array(ref_img, dtype=np.int16)
                ).astype(np.float32)
                diff_arr = np.clip(diff_arr * amplify, 0, 255).astype(np.uint8)
                diff_img = Image.fromarray(diff_arr, mode="RGB")

                # 保存先: results_root 直下に snapdiff_XX.png
                diff_filename = f"snapdiff_{index}.png"
                diff_path     = os.path.join(results_root, diff_filename)
                diff_img.save(diff_path)

                print(f"  [OK] {filename}  →  {diff_filename}")
                matched += 1

            except Exception as e:
                print(f"  [ERROR] {filename}: {e}")
                skipped += 1

    print(f"\nDiff complete: {matched} saved, {skipped} skipped.")


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run MPM simulation and render PNGs.")
    parser.add_argument('--eta',          type=float, required=True,
                        help='Viscosity parameter')
    parser.add_argument('--n',            type=float, required=True,
                        help='Power law index')
    parser.add_argument('--sigma_y',      type=float, required=True,
                        help='Yield stress')
    parser.add_argument('--ref',      type=str,   required=True,
                        help='Reference folder containing config_XX.png, '
                             'settings.xml (W/H), and camera_params.xml.')
    parser.add_argument('--out_dir',     type=str,   default=None,
                        help='Output directory. Default: Simulation/results/run_<ts>/')
    parser.add_argument('--camera_xml',  type=str,   default=None,
                        help='Path to camera_params.xml file, or directory containing it (default: --ref/camera_params.xml)')
    parser.add_argument('--diff_amplify', type=float, default=5.0,
                        help='Brightness amplification for diff images (default: 5.0)')
    args = parser.parse_args()

    # ref の存在確認
    if not os.path.isdir(args.ref):
        print(f"[ERROR] --ref '{args.ref}' does not exist.")
        return

    # ── 1. ref の XML から width / height / camera を取得 ──
    print("Loading settings from ref XMLs...")
    try:
        settings      = load_settings(args.ref)
        cam_xml_path  = args.camera_xml if args.camera_xml else args.ref
        camera_params = load_camera_params(cam_xml_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        return

    width  = settings["width"]
    height = settings["height"]
    print(f"  width={width}, height={height}")
    print(f"  camera eyepos={camera_params['eyepos']}, fov={camera_params['fov']}")

    print("\nStarting fluid simulation workflow")
    print("=" * 60)

    # ── 2. タイムスタンプ付き結果ディレクトリ作成 ──
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = args.out_dir if args.out_dir \
                   else os.path.join("results", f"run_{timestamp}")
    os.makedirs(results_root, exist_ok=True)

    # ── 3. CSV 初期化 ──
    csv_filename = os.path.join(results_root, "simulation_results.csv")
    with open(csv_filename, 'w', newline='') as f:
        headers = ["n", "eta", "sigma_y", "width", "height"] + \
                  [f"x_0{i+1}" for i in range(8)]
        csv.writer(f).writerow(headers)

    print(f"Parameters: n={args.n}, eta={args.eta}, sigma_y={args.sigma_y}, "
          f"width={width}, height={height}")

    # ── 4. サンプルディレクトリ ──
    sample_dir_name = f"{args.n:.2f}_{args.eta:.2f}_{args.sigma_y:.2f}"
    sample_dir_path = os.path.join(results_root, sample_dir_name)
    os.makedirs(sample_dir_path, exist_ok=True)

    try:
        # ── 5. シミュレーター初期化 ──
        simulator = MPMSimulator(XML_TEMPLATE_PATH)

        # ── 6. ジオメトリ設定 ──
        simulator.configure_geometry(width=width, height=height)

        # ── 7. シミュレーション実行 ──
        displacements = simulator.run_simulation(
            n=args.n,
            eta=args.eta,
            sigma_y=args.sigma_y,
            output_dir=sample_dir_path
        )
        print(f"Simulation completed. Displacements: {displacements}")

        # ── 8. CSV 書き込み ──
        with open(csv_filename, 'a', newline='') as csvfile:
            csv.writer(csvfile).writerow([
                args.n, args.eta, args.sigma_y, width, height,
                *[displacements[i] if i < len(displacements) else 0 for i in range(8)]
            ])

    except Exception as e:
        print(f"Error processing sample: {e}")
        traceback.print_exc()
        return

    # ── 9. レンダリング ──
    print("\nRendering OBJ files...")
    renderer = MPM_Emulator.MPMEmulator()
    # カメラパラメータを渡せる場合はここで設定
    # 例: renderer.set_camera(**camera_params)
    renderer.render_all()
    print("Rendering finished.")

    # ── 10. 差分画像生成 ──
    compute_and_save_diffs(
        results_root=results_root,
        ref=args.ref,
        amplify=args.diff_amplify
    )

    print("\nWorkflow finished.")


if __name__ == "__main__":
    main()