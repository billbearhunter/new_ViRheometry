#!/usr/bin/env python3
"""
Extract flow-distance sequence from binary images + camera_params.xml + settings.xml.

Minimal usage:
    python extract_flow_distance_v3.py --dir /path/to/data

Auto-discovers settings.xml, camera_params.xml, and config_*.png in the directory.
All parameters can be overridden via command line.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

@dataclass
class CameraOnPlane:
    eye_world_m: np.ndarray   # (3,)
    rot_c2w: np.ndarray       # (3, 3)
    width_px: int
    height_px: int
    fov_deg: float
    focal_px: float

    @property
    def cx(self) -> float:
        return self.width_px / 2.0

    @property
    def cy(self) -> float:
        return self.height_px / 2.0


def load_camera_from_xml(xml_path: str) -> CameraOnPlane:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    cam = root.find("camera")
    if cam is None:
        raise ValueError(f"No <camera> node in {xml_path}")

    eye_cm = np.array(cam.attrib["eyepos"].split(), dtype=float)
    eye_world_m = eye_cm / 100.0

    quat_wxyz = np.array(cam.attrib["quat"].split(), dtype=float)
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=float)
    rot_c2w = R.from_quat(quat_xyzw).as_matrix()

    width_px, height_px = [int(round(float(v))) for v in cam.attrib["window_size"].split()]
    fov_deg = float(cam.attrib["fov"])
    focal_px = height_px / (2.0 * math.tan(math.radians(fov_deg) / 2.0))

    return CameraOnPlane(
        eye_world_m=eye_world_m,
        rot_c2w=rot_c2w,
        width_px=width_px,
        height_px=height_px,
        fov_deg=fov_deg,
        focal_px=focal_px,
    )


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class SimSettings:
    width: float   # box W (cm)
    height: float  # box H (cm)

    @property
    def box_front_cm(self) -> float:
        return self.width


def load_settings_from_xml(xml_path: str) -> SimSettings:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    setup = root.find("setup")
    if setup is None:
        raise ValueError(f"No <setup> node in {xml_path}")
    return SimSettings(width=float(setup.attrib["W"]), height=float(setup.attrib["H"]))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def unit_scale(unit: str) -> float:
    return {"m": 1.0, "cm": 100.0, "mm": 1000.0}[unit]


def to_meters(value: float, unit: str) -> float:
    return float(value) / unit_scale(unit)


def normalize_2d(v: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(2)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("flow direction must be non-zero")
    return v / n


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def make_mask(image: np.ndarray, threshold: int, foreground: str) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if foreground == "black":
        mask = image < threshold
    elif foreground == "white":
        mask = image > threshold
    else:
        raise ValueError("foreground must be 'black' or 'white'")
    return mask.astype(np.uint8)


def denoise_mask(mask: np.ndarray, kernel_size: int, opening_only: bool = False) -> np.ndarray:
    if kernel_size <= 1:
        return mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if not opening_only:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    return out


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == best).astype(np.uint8)


def extract_outer_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found")
    cnt = max(contours, key=cv2.contourArea)
    return cnt[:, 0, :].astype(np.float64)


# ---------------------------------------------------------------------------
# Back-projection
# ---------------------------------------------------------------------------

def pixels_to_world_plane_y0(uv: np.ndarray, camera: CameraOnPlane) -> np.ndarray:
    u = uv[:, 0]
    v = uv[:, 1]

    x = (u - camera.cx) / camera.focal_px
    y = -(v - camera.cy) / camera.focal_px

    dirs_cam = np.stack([x, y, -np.ones_like(x)], axis=1)
    dirs_world = dirs_cam @ camera.rot_c2w.T

    denom = dirs_world[:, 1]
    valid = np.abs(denom) > 1e-12
    if not np.any(valid):
        raise ValueError("All rays are parallel to plane y=0")

    dirs_world = dirs_world[valid]
    t = -camera.eye_world_m[1] / dirs_world[:, 1]
    valid2 = t > 0
    if not np.any(valid2):
        raise ValueError("No intersections in front of camera")

    dirs_world = dirs_world[valid2]
    t = t[valid2]
    pts_world = camera.eye_world_m[None, :] + t[:, None] * dirs_world
    return pts_world


# ---------------------------------------------------------------------------
# Debug visualization
# ---------------------------------------------------------------------------

def draw_debug(
    image: np.ndarray,
    contour_uv: np.ndarray,
    far_uv: np.ndarray,
    text: str,
    save_path: str,
) -> None:
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(canvas, [contour_uv.astype(np.int32).reshape(-1, 1, 2)], -1, (0, 255, 0), 1)
    cv2.circle(canvas, tuple(np.round(far_uv).astype(int)), 4, (0, 0, 255), -1)
    cv2.putText(canvas, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(save_path, canvas)


# ---------------------------------------------------------------------------
# Auto-discover files in directory
# ---------------------------------------------------------------------------

def discover_files(dir_path: str) -> dict:
    """Auto-discover settings.xml, camera_params.xml, and config_*.png in directory."""
    d = Path(dir_path)
    if not d.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    settings = d / "settings.xml"
    if not settings.exists():
        raise FileNotFoundError(f"Not found: {settings}")

    camera = d / "camera_params.xml"
    if not camera.exists():
        raise FileNotFoundError(f"Not found: {camera}")

    images = sorted(d.glob("config_*.png"))
    if not images:
        raise FileNotFoundError(f"No config_*.png found in {d}")

    return {
        "settings": str(settings),
        "camera_xml": str(camera),
        "images": [str(p) for p in images],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract flow-distance sequence from binary images (v3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Minimal usage:
  python extract_flow_distance_v3.py --dir /path/to/data

Directory should contain settings.xml, camera_params.xml, and config_*.png.
All parameters can be overridden via command line.
""",
    )

    # ---- data directory ----
    p.add_argument("--dir", default=None,
                    help="Data directory (auto-discovers settings.xml / camera_params.xml / config_*.png)")

    # ---- manual file overrides ----
    p.add_argument("--settings", default=None, help="Override settings.xml path")
    p.add_argument("--camera-xml", default=None, help="Override camera_params.xml path")
    p.add_argument("--images", default=None, help='Override image glob, e.g. "other_*.png"')

    # ---- optional parameters ----
    p.add_argument("--threshold", type=int, default=128)
    p.add_argument("--foreground", choices=["black", "white"], default="black")
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--opening-only", action="store_true")
    p.add_argument("--no-keep-largest", action="store_true", help="Disable largest-component filtering (enabled by default)")
    p.add_argument("--flow-direction", type=float, nargs=2, default=[1.0, 0.0], metavar=("DX", "DZ"))
    p.add_argument("--box-front", type=float, default=None, help="Override box front position")
    p.add_argument("--monotonic", action="store_true")
    p.add_argument("--unit", choices=["m", "cm", "mm"], default="cm")

    # ---- output ----
    p.add_argument("--output-csv", default=None, help="Default: <dir>/flow_distances.csv")
    p.add_argument("--output-json", default=None, help="Default: <dir>/flow_distances.json")
    p.add_argument("--debug-dir", default=None)
    p.add_argument("--print-dis1", action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- file discovery ----
    if args.dir is not None:
        discovered = discover_files(args.dir)
        settings_path = args.settings or discovered["settings"]
        camera_path = args.camera_xml or discovered["camera_xml"]
        image_paths = sorted(glob.glob(args.images)) if args.images else discovered["images"]
        output_dir = args.dir
    else:
        if args.camera_xml is None or args.images is None:
            raise ValueError("Provide --dir, or both --camera-xml and --images")
        settings_path = args.settings
        camera_path = args.camera_xml
        image_paths = sorted(glob.glob(args.images))
        output_dir = "."

    if not image_paths:
        raise FileNotFoundError("No images found")

    output_csv = args.output_csv or os.path.join(output_dir, "flow_distances.csv")
    output_json = args.output_json or os.path.join(output_dir, "flow_distances.json")

    # ---- load settings ----
    settings: Optional[SimSettings] = None
    if settings_path is not None:
        settings = load_settings_from_xml(settings_path)
        print(f"[settings] W={settings.width}, H={settings.height}")

    # ---- determine box_front ----
    if args.box_front is not None:
        box_front_value = args.box_front
    elif settings is not None:
        box_front_value = settings.box_front_cm
    else:
        raise ValueError("Requires settings.xml (or --dir with settings.xml) or --box-front")
    print(f"[config]  box-front = {box_front_value} {args.unit}")

    # ---- load camera ----
    camera = load_camera_from_xml(camera_path)
    print(f"[camera]  {camera.width_px}x{camera.height_px}, fov={camera.fov_deg:.2f}°")
    print(f"[images]  {len(image_paths)} files")

    # ---- prepare parameters ----
    direction_xz = normalize_2d(args.flow_direction)
    scale = unit_scale(args.unit)
    box_front_m = to_meters(box_front_value, args.unit)
    keep_largest = not args.no_keep_largest

    if args.debug_dir is not None:
        Path(args.debug_dir).mkdir(parents=True, exist_ok=True)

    # ---- per-frame processing ----
    rows = []
    distances_out = []
    prev_distance_m = -np.inf
    origin_projection_m = box_front_m

    scale_detected = False
    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Failed to read {image_path}")
        img_w, img_h = image.shape[1], image.shape[0]

        # Auto-detect resolution scaling (e.g. 1920x1080 vs 960x540)
        if not scale_detected:
            if img_w != camera.width_px or img_h != camera.height_px:
                sx = img_w / camera.width_px
                sy = img_h / camera.height_px
                if abs(sx - sy) > 1e-6 or sx <= 0:
                    raise ValueError(
                        f"Image size {img_w}x{img_h} is not a uniform scale of "
                        f"camera {camera.width_px}x{camera.height_px}"
                    )
                print(f"[scale]   Image is {sx:.0f}x camera resolution, adjusting focal/cx/cy")
                camera.width_px = img_w
                camera.height_px = img_h
                camera.focal_px *= sx
            scale_detected = True

        mask = make_mask(image, threshold=args.threshold, foreground=args.foreground)
        mask = denoise_mask(mask, kernel_size=args.kernel_size, opening_only=args.opening_only)
        if keep_largest:
            mask = keep_largest_component(mask)

        contour_uv = extract_outer_contour(mask)
        world_pts = pixels_to_world_plane_y0(contour_uv, camera)
        world_xz = world_pts[:, [0, 2]]
        proj = world_xz @ direction_xz

        far_idx = int(np.argmax(proj))
        distance_m = float(proj[far_idx] - origin_projection_m)
        if args.monotonic:
            distance_m = max(distance_m, prev_distance_m)
        prev_distance_m = distance_m

        row = {
            "frame_index": idx + 1,
            "image": os.path.basename(image_path),
            f"distance_{args.unit}": distance_m * scale,
            f"origin_projection_{args.unit}": origin_projection_m * scale,
            f"far_projection_{args.unit}": float(proj[far_idx] * scale),
            f"far_x_{args.unit}": float(world_xz[far_idx, 0] * scale),
            f"far_z_{args.unit}": float(world_xz[far_idx, 1] * scale),
        }
        rows.append(row)
        distances_out.append(distance_m * scale)

        if args.debug_dir is not None:
            draw_debug(
                image=image,
                contour_uv=contour_uv,
                far_uv=contour_uv[far_idx],
                text=f"d = {distance_m * scale:.6f} {args.unit}",
                save_path=str(Path(args.debug_dir) / f"debug_{idx+1:02d}.png"),
            )

    # ---- output ----
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dir": args.dir,
        "camera_xml": camera_path,
        "settings_xml": settings_path,
        "images": [os.path.basename(p) for p in image_paths],
        "flow_direction_xz": direction_xz.tolist(),
        f"box_front_{args.unit}": box_front_value,
        "distances": distances_out,
        "rows": rows,
    }
    Path(output_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Done.")
    print("Distances:")
    print(" ".join(f"{d:.7f}" for d in distances_out))
    if args.print_dis1:
        print("-dis1 " + " ".join(f"{d:.7f}" for d in distances_out))


if __name__ == "__main__":
    main()