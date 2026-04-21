#!/usr/bin/env python3
"""
prepare_configs.py — ViRheometry production pipeline
======================================================
video.mov + background.JPG + settings.xml
   ↓
config_00 .. config_08.png   (9 binary masks, PS-compatible RGB PNGs)
camera_params.xml            (calibrated 3D camera)

Workflow (2–3 GUI interactions)
-------------------------------
  1. Frame picker GUI       — scrub video, pick frame 0 (dam-break frame).
                              Pre-fills with auto-detected dam-break + 1/24s shift
                              so cfg_00 = pre-dam-break, cfg_01 = initial state.

  2. Cube vertex picker GUI — on frame 0 (the same frame just picked), click 3+
                              cube corners (left wall is most reliable). solvePnP
                              uses these to refine the camera pose. ChArUco on
                              the background image gives K + a prior on R, t.

  3. ROI polygon GUI        — draw a polygon covering the container + maximum
                              puddle extent (shown on the cfg_08 frame). Used
                              for binarisation of all configs.

Then automatic:
  4. VFR-safe timestamp seeking (ffprobe) → frame_indices for cfg_00..08
  5. Binarise each frame inside ROI → config_XX.png (RGB, PR-compatible)
  6. Save camera_params.xml + Background.png + Background_mask.png + diffs

Usage
-----
  # Recommended: just point at the experiment directory.
  python Calibration/prepare_configs.py \\
      --data_dir data/ref_Tonkatsu_2.5_3.0_1/1st

  # Reuse picks from a previous run (skip all GUIs):
  python Calibration/prepare_configs.py \\
      --data_dir data/ref_Tonkatsu_2.5_3.0_1/1st \\
      --frame0 1425 \\
      --roi-poly       output/roi_polygon.npy \\
      --cube-vertices  output/cube_vertices.npy

Output structure
----------------
  <out_dir>/
    config_00 .. config_08.png   ← 9 RGB binary masks (PR/PS compatible)
    camera_params.xml             ← <setup3D><camera ...>
    Background.png                ← rendered scene + cube overlay
    Background_mask.png           ← rendered cube silhouette
    diff_combined.png / diff_binary.png  ← calibration sanity check
    settings.xml                  ← copied for the simulation
    roi_polygon.npy               ← ROI (reusable via --roi-poly)
    cube_vertices.npy             ← cube picks (reusable via --cube-vertices)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# settings.xml parser is defined in pipeline.py and re-exported here so
# callers can still do `from prepare_configs import _parse_settings_xml`.
from pipeline import _parse_settings_xml  # noqa: F401


# ── Defaults ─────────────────────────────────────────────────────────────
RAW_FPS       = 240       # iPhone slow-mo nominal frame rate
TARGET_FPS    = 24        # output rate (1 config per 1/TARGET_FPS seconds)
N_CONFIGS     = 9         # config_00 .. config_08
V_THRESH      = 128       # HSV V-channel threshold (matches PS workflow)
MOTION_THRESH = 20.0      # mean pixel diff threshold for dam-break spike
SKIN_LO       = np.array([0, 20, 120], dtype=np.uint8)   # HSV skin lower
SKIN_HI       = np.array([30, 200, 255], dtype=np.uint8) # HSV skin upper


# ── Video helpers ────────────────────────────────────────────────────────

def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    return cap


def read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray:
    """Read a specific frame by index."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Failed to read frame {idx}")
    return frame


def get_video_info(cap: cv2.VideoCapture) -> dict:
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return {"n_frames": n_frames, "fps": fps, "width": w, "height": h}


def build_timestamp_index(video_path: str) -> np.ndarray:
    """Build frame-index → timestamp mapping using ffprobe.
    Essential for VFR (variable frame rate) videos like iPhone slow-mo.
    Returns array of timestamps in seconds, indexed by frame number."""
    import subprocess, shutil

    if shutil.which("ffprobe") is None:
        return None  # ffprobe not available; fall back to index-based

    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=pts_time", "-of", "csv=p=0", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"[WARN] ffprobe failed: {result.stderr.strip()}")
        return None

    timestamps = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip().rstrip(",")
        if line:
            try:
                timestamps.append(float(line))
            except ValueError:
                continue

    if len(timestamps) < 10:
        return None

    ts = np.array(timestamps)
    diffs = np.diff(ts)
    nominal = np.median(diffs)
    irregular = np.sum(np.abs(diffs - nominal) > nominal * 0.3)
    if irregular > 5:
        print(f"[VFR] Detected variable frame rate: {irregular} irregular "
              f"intervals out of {len(diffs)}. Using timestamp-based seeking.")
    else:
        print(f"[VFR] Frame rate looks constant ({1/nominal:.1f} fps). "
              f"Timestamp index built anyway for safety.")
    return ts


def get_true_fps(video_path: str) -> float:
    """Get the nominal (max) frame rate from ffprobe r_frame_rate.
    For iPhone slow-mo this returns 240, not the average ~190."""
    import subprocess, shutil

    if shutil.which("ffprobe") is None:
        return None

    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        return None

    rate_str = result.stdout.strip()
    if "/" in rate_str:
        num, den = rate_str.split("/")
        return float(num) / float(den)
    try:
        return float(rate_str)
    except ValueError:
        return None


# ── Dam-break detection ─────────────────────────────────────────────────

def _frame_motion(cap: cv2.VideoCapture, idx_a: int, idx_b: int) -> float:
    """Mean absolute pixel diff between two frames (grayscale)."""
    ga = cv2.cvtColor(read_frame(cap, idx_a), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(read_frame(cap, idx_b), cv2.COLOR_BGR2GRAY)
    return cv2.absdiff(ga, gb).astype(float).mean()


def detect_dam_break(cap: cv2.VideoCapture, search_start: float = 0.3,
                     search_end: float = 0.95, step: int = 5) -> int:
    """
    Two-phase dam-break detection:
      Phase 1 — Coarse scan (step=5) to locate the approximate spike region.
      Phase 2 — Fine scan (step=1, frame-by-frame) around the spike to find
                the exact onset of motion, i.e. the first frame where
                consecutive-frame diff exceeds a low "onset" threshold.

    Returns the frame index where the gate begins to open (= frame 0).
    """
    info = get_video_info(cap)
    n = info["n_frames"]
    start_idx = int(n * search_start)
    end_idx = int(n * search_end)

    # ── Phase 1: coarse scan ──
    print(f"[dam-break] Phase 1: coarse scan {start_idx}..{end_idx} (step={step})")

    prev_gray = cv2.cvtColor(read_frame(cap, start_idx), cv2.COLOR_BGR2GRAY)
    motions = []

    for idx in range(start_idx + step, end_idx, step):
        curr_gray = cv2.cvtColor(read_frame(cap, idx), cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(curr_gray, prev_gray).astype(float).mean()
        motions.append((idx, diff))
        prev_gray = curr_gray

    motion_values = [m for _, m in motions]
    median_motion = float(np.median(motion_values))
    adaptive_thresh = max(MOTION_THRESH, median_motion * 5)
    print(f"[dam-break] Median motion={median_motion:.2f}, "
          f"spike threshold={adaptive_thresh:.2f}")

    # Find the coarse spike position
    spike_coarse = None
    for i, (idx, motion) in enumerate(motions):
        if motion > adaptive_thresh:
            spike_coarse = idx
            pre_spike = motions[i - 1][0] if i > 0 else start_idx
            print(f"[dam-break] Coarse spike at frame {idx} (diff={motion:.2f})")
            break

    if spike_coarse is None:
        peak_idx, peak_motion = max(motions, key=lambda x: x[1])
        print(f"[dam-break] No clear spike; using peak frame {peak_idx} "
              f"(diff={peak_motion:.2f})")
        spike_coarse = peak_idx
        pre_spike = max(start_idx, peak_idx - step)

    # ── Phase 2: fine scan (frame-by-frame) ──
    # Search backward from spike to find exact onset of motion.
    # "Onset" = first frame whose per-frame diff exceeds the quiet baseline.
    fine_start = max(0, pre_spike - step * 4)   # generous backward range
    fine_end   = min(spike_coarse + step * 2, n - 1)

    print(f"[dam-break] Phase 2: fine scan {fine_start}..{fine_end} (step=1)")

    fine_motions = []
    prev_gray = cv2.cvtColor(read_frame(cap, fine_start), cv2.COLOR_BGR2GRAY)
    for idx in range(fine_start + 1, fine_end + 1):
        curr_gray = cv2.cvtColor(read_frame(cap, idx), cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(curr_gray, prev_gray).astype(float).mean()
        fine_motions.append((idx, diff))
        prev_gray = curr_gray

    # Baseline = median of fine diffs in the quiet region (before pre_spike)
    quiet_diffs = [d for (f, d) in fine_motions if f <= pre_spike]
    if quiet_diffs:
        baseline = float(np.median(quiet_diffs))
    else:
        baseline = float(np.median([d for _, d in fine_motions[:20]]))

    # Onset threshold: baseline + 3*std, or at least 2× baseline
    quiet_std = float(np.std(quiet_diffs)) if len(quiet_diffs) > 2 else baseline
    onset_thresh = max(baseline + 3 * quiet_std, baseline * 2, 1.0)
    print(f"[dam-break] Baseline={baseline:.3f}, onset threshold={onset_thresh:.3f}")

    # Strategy: find the motion PEAK within the spike region, then set
    # frame0 to the first frame AFTER the peak where motion drops back
    # toward baseline.  This captures the moment the gate is fully open
    # and fluid begins free flow, not the moment the gate starts moving.
    onset_idx = pre_spike  # fallback
    peak_idx = pre_spike
    peak_diff = 0.0
    found_onset = False

    for idx, diff in fine_motions:
        if not found_onset and diff > onset_thresh:
            onset_idx = idx
            found_onset = True
        if found_onset and diff > peak_diff:
            peak_diff = diff
            peak_idx = idx
        # Stop scanning well after the spike ends
        if found_onset and diff < onset_thresh and idx > peak_idx + 5:
            break

    if found_onset:
        # Find where motion returns to near-baseline after the spike.
        return_idx = fine_motions[-1][0]  # fallback: end of fine scan
        for idx, diff in fine_motions:
            if idx > peak_idx and diff < onset_thresh:
                return_idx = idx
                break

        # frame0 = END of gate motion (return to baseline).
        # This is the moment the gate is FULLY lifted and the fluid
        # silhouette is a clean rectangle (before any free-flow spread).
        # Using midpoint would leave the gate half-visible in cfg_00,
        # breaking the calibration that expects fluid_w × fluid_h cube.
        dam_break_idx = return_idx

        print(f"[dam-break] Onset={onset_idx}, peak={peak_idx} "
              f"(diff={peak_diff:.3f}), return={return_idx}")
        print(f"[dam-break] Using return (gate fully lifted) → frame 0 = {dam_break_idx}")
    else:
        dam_break_idx = pre_spike
        print(f"[dam-break] No clear onset found; using pre-spike frame")

    print(f"[dam-break] Selected frame 0 = {dam_break_idx}")
    return dam_break_idx


# ── Interactive frame picker ─────────────────────────────────────────────

def pick_frame_interactive(
    cap: cv2.VideoCapture,
    hint_frame: int = 0,
    window_name: str = "Pick frame 0 (dam-break moment)",
) -> int:
    """
    Open a GUI window that lets the user scrub through the video and pick
    the dam-break frame manually (Premiere-Pro-style).

    Controls:
      ←/→  or A/D : ±1 frame
      ↑/↓  or W/S : ±10 frames
      PgUp/PgDn   : ±100 frames
      Home/End    : first / last frame
      [ / ]       : ±1 raw second
      Mouse-drag scrubber bar at bottom : jump to position
      Click on scrubber bar             : jump to position
      Enter / Space : confirm current frame as frame 0
      Esc           : abort
    Returns the selected frame index, or -1 if cancelled.
    """
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = max(cap.get(cv2.CAP_PROP_FPS), 1.0)
    current  = max(0, min(hint_frame, n_frames - 1))
    cancelled = False

    def _read(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        return f if ret else None

    frame = _read(current)
    h, w = frame.shape[:2]

    scale = 1.0
    if w > 1400:
        scale = 1400.0 / w
    disp_w, disp_h = int(w * scale), int(h * scale)
    bar_h = 24

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_w, disp_h + bar_h)

    dragging = {"on": False}

    def _idx_from_x(x):
        return max(0, min(n_frames - 1, int(x / max(1, disp_w - 1) * (n_frames - 1))))

    def mouse_cb(event, x, y, flags, param):
        nonlocal current
        if y >= disp_h:  # in scrubber bar
            if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE):
                if event == cv2.EVENT_LBUTTONDOWN:
                    dragging["on"] = True
                if event == cv2.EVENT_LBUTTONDOWN or (dragging["on"] and (flags & cv2.EVENT_FLAG_LBUTTON)):
                    new = _idx_from_x(x)
                    if new != current:
                        current = new
        if event == cv2.EVENT_LBUTTONUP:
            dragging["on"] = False

    cv2.setMouseCallback(window_name, mouse_cb)

    print("\n" + "=" * 60)
    print("  PICK FRAME 0  (dam-break moment, when gate fully lifted)")
    print("=" * 60)
    print("  ←/→ A/D     : ±1 frame")
    print("  ↑/↓ W/S     : ±10 frames")
    print("  PgUp/PgDn   : ±100 frames")
    print("  Home/End    : first / last")
    print("  [ ]         : ±1 raw second")
    print("  Mouse drag  : scrubber bar (bottom)")
    print("  Enter/Space : confirm")
    print("  Esc         : cancel")
    print(f"  Hint: auto-detected frame = {hint_frame}")
    print("=" * 60 + "\n")

    last_drawn = -1
    while True:
        if current != last_drawn:
            new_frame = _read(current)
            if new_frame is not None:
                frame = new_frame
            last_drawn = current

        canvas = cv2.resize(frame, (disp_w, disp_h))
        # Stack with scrubber bar
        bar = np.zeros((bar_h, disp_w, 3), dtype=np.uint8)
        cv2.rectangle(bar, (0, 0), (disp_w-1, bar_h-1), (60, 60, 60), -1)
        # Filled progress
        x_cur = int(current / max(1, n_frames - 1) * (disp_w - 1))
        cv2.rectangle(bar, (0, 0), (x_cur, bar_h-1), (200, 120, 0), -1)
        # Cursor mark
        cv2.line(bar, (x_cur, 0), (x_cur, bar_h-1), (255, 255, 255), 2)
        full = np.vstack([canvas, bar])

        info = (f"Frame {current}/{n_frames-1}  ({current/fps:.2f}s)  "
                f"|  Hint={hint_frame}  |  ←→=1  ↑↓=10  PgUp/Dn=100  Enter=OK")
        cv2.putText(full, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        cv2.imshow(window_name, full)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:                    # Esc
            cancelled = True; break
        elif key in (13, 32):            # Enter / Space
            break
        elif key in (83, 100, 3):        # → / D
            current = min(n_frames - 1, current + 1)
        elif key in (81, 97, 2):         # ← / A
            current = max(0, current - 1)
        elif key in (82, 119):           # ↑ / W
            current = min(n_frames - 1, current + 10)
        elif key in (84, 115):           # ↓ / S
            current = max(0, current - 10)
        elif key == 85:                  # PgUp
            current = min(n_frames - 1, current + 100)
        elif key == 86:                  # PgDn
            current = max(0, current - 100)
        elif key == 80:                  # Home
            current = 0
        elif key == 87:                  # End
            current = n_frames - 1
        elif key == ord(']'):
            current = min(n_frames - 1, current + int(round(fps)))
        elif key == ord('['):
            current = max(0, current - int(round(fps)))

    cv2.destroyWindow(window_name)
    return -1 if cancelled else current


# ── Interactive ROI polygon ──────────────────────────────────────────────

def pick_roi_polygon(
    cap: cv2.VideoCapture,
    start_frame: int,
    frame_step: int = 10,
    window_name: str = "Draw ROI (container + puddle)",
    show_scrubber: bool = True,
) -> np.ndarray:
    """
    Open a window showing a video frame. User draws a polygon around the
    entire region of interest (container + maximum puddle extent).

    Controls:
      - Left/Right arrow (or A/D): browse frames
      - Left-click: add vertex
      - Right-click: undo last vertex
      - Enter / Space: confirm polygon (minimum 3 points)
      - Esc: abort

    Returns: np.ndarray of shape (N, 2) with vertex coordinates,
             or empty array if cancelled.
    """
    points: list[tuple[int, int]] = []
    done = False
    cancelled = False
    current_idx = start_frame
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = max(cap.get(cv2.CAP_PROP_FPS), 1.0)

    def _read(idx):
        idx = max(0, min(idx, n_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        return f if ret else None

    frame = _read(current_idx)
    h, w = frame.shape[:2]

    scale = 1.0
    if w > 1400:
        scale = 1400.0 / w
    disp_w, disp_h = int(w * scale), int(h * scale)
    bar_h = 24 if show_scrubber else 0
    dragging = {"on": False}

    def _idx_from_x(x):
        return max(0, min(n_frames - 1, int(x / max(1, disp_w - 1) * (n_frames - 1))))

    def mouse_cb(event, x, y, flags, param):
        nonlocal points, current_idx
        # Scrubber bar (bottom strip)
        if show_scrubber and y >= disp_h:
            if event == cv2.EVENT_LBUTTONDOWN:
                dragging["on"] = True
                current_idx = _idx_from_x(x)
            elif event == cv2.EVENT_MOUSEMOVE and dragging["on"] and (flags & cv2.EVENT_FLAG_LBUTTON):
                current_idx = _idx_from_x(x)
            elif event == cv2.EVENT_LBUTTONUP:
                dragging["on"] = False
            return
        if event == cv2.EVENT_LBUTTONUP:
            dragging["on"] = False
        # Polygon picker (image area)
        ox, oy = int(x / scale), int(y / scale)
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((ox, oy))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if points:
                points.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_w, disp_h + bar_h)
    cv2.setMouseCallback(window_name, mouse_cb)

    print("\n" + "=" * 60)
    print("  DRAW ROI POLYGON")
    print("=" * 60)
    print("  ←/→ A/D     : ±1 step  (frame_step=" + str(frame_step) + ")")
    print("  ↑/↓ W/S     : ±10 steps")
    print("  PgUp/PgDn   : ±100 steps")
    print("  Mouse drag  : scrubber bar (bottom)")
    print("  L-click     : add vertex   |   R-click : undo last")
    print("  Enter/Space : confirm (need ≥ 3)   |   Esc : cancel")
    print("=" * 60 + "\n")

    last_drawn = -1
    while not done:
        if current_idx != last_drawn:
            new_frame = _read(current_idx)
            if new_frame is not None:
                frame = new_frame
            last_drawn = current_idx

        canvas = frame.copy()
        for i, pt in enumerate(points):
            cv2.circle(canvas, pt, 6, (0, 255, 0), -1)
            cv2.putText(canvas, str(i), (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if i > 0:
                cv2.line(canvas, points[i - 1], pt, (0, 255, 0), 2)
        if len(points) >= 3:
            cv2.line(canvas, points[-1], points[0], (0, 255, 0), 1)
            overlay = canvas.copy()
            pts_arr = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [pts_arr], (0, 0, 200))
            canvas = cv2.addWeighted(canvas, 0.7, overlay, 0.3, 0)

        info = (f"Frame {current_idx}/{n_frames-1}  ({current_idx/fps:.2f}s)  "
                f"|  Vertices: {len(points)}  |  Drag scrubber to browse")
        cv2.putText(canvas, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        disp = cv2.resize(canvas, (disp_w, disp_h))
        if show_scrubber:
            bar = np.zeros((bar_h, disp_w, 3), dtype=np.uint8)
            cv2.rectangle(bar, (0, 0), (disp_w-1, bar_h-1), (60, 60, 60), -1)
            x_cur = int(current_idx / max(1, n_frames - 1) * (disp_w - 1))
            cv2.rectangle(bar, (0, 0), (x_cur, bar_h-1), (200, 120, 0), -1)
            cv2.line(bar, (x_cur, 0), (x_cur, bar_h-1), (255, 255, 255), 2)
            disp = np.vstack([disp, bar])
        cv2.imshow(window_name, disp)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:                    # Esc
            cancelled = True; done = True
        elif key in (13, 32):            # Enter / Space
            if len(points) >= 3:
                done = True
            else:
                print("[interactive] Need at least 3 points. Keep clicking.")
        elif key in (83, 100, 3):        # → / D
            current_idx = min(current_idx + frame_step, n_frames - 1)
        elif key in (81, 97, 2):         # ← / A
            current_idx = max(current_idx - frame_step, 0)
        elif key in (82, 119):           # ↑ / W
            current_idx = min(current_idx + 10*frame_step, n_frames - 1)
        elif key in (84, 115):           # ↓ / S
            current_idx = max(current_idx - 10*frame_step, 0)
        elif key == 85:                  # PgUp
            current_idx = min(current_idx + 100*frame_step, n_frames - 1)
        elif key == 86:                  # PgDn
            current_idx = max(current_idx - 100*frame_step, 0)

    cv2.destroyWindow(window_name)

    if cancelled or len(points) < 3:
        return np.array([], dtype=np.int32)

    return np.array(points, dtype=np.int32)


def pick_polygon_on_image(
    image_bgr: np.ndarray,
    window_name: str = "Draw polygon",
    initial_poly: Optional[np.ndarray] = None,
    instructions: Optional[list] = None,
) -> np.ndarray:
    """
    Open a window on a SINGLE static image.  User draws a polygon that
    encloses a region (e.g. the CLEAN left-side container outline for
    camera calibration).

    Controls:
      L-click     : add vertex
      R-click     : undo last vertex
      C           : clear all vertices
      Enter/Space : confirm (need ≥ 3)
      Esc         : cancel

    Returns (N, 2) int32 polygon in ORIGINAL image coordinates, or empty
    array if cancelled.
    """
    h, w = image_bgr.shape[:2]
    scale = 1.0
    if w > 1400:
        scale = 1400.0 / w
    disp_w, disp_h = int(w * scale), int(h * scale)

    points: list[tuple[int, int]] = []
    if initial_poly is not None and len(initial_poly) >= 3:
        for p in initial_poly:
            points.append((int(p[0]), int(p[1])))

    done = False
    cancelled = False

    def mouse_cb(event, x, y, flags, param):
        nonlocal points
        ox, oy = int(x / scale), int(y / scale)
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((ox, oy))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_w, disp_h)
    cv2.setMouseCallback(window_name, mouse_cb)

    print("\n" + "=" * 60)
    print(f"  {window_name.upper()}")
    print("=" * 60)
    if instructions:
        for line in instructions:
            print(f"  {line}")
        print("-" * 60)
    print("  L-click     : add vertex    |  R-click : undo last")
    print("  C           : clear all")
    print("  Enter/Space : confirm (need ≥ 3)   |  Esc : cancel")
    print("=" * 60 + "\n")

    while not done:
        canvas = image_bgr.copy()
        for i, pt in enumerate(points):
            cv2.circle(canvas, pt, 6, (0, 255, 0), -1)
            cv2.putText(canvas, str(i), (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if i > 0:
                cv2.line(canvas, points[i - 1], pt, (0, 255, 0), 2)
        if len(points) >= 3:
            cv2.line(canvas, points[-1], points[0], (0, 255, 0), 1)
            overlay = canvas.copy()
            pts_arr = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [pts_arr], (0, 200, 0))
            canvas = cv2.addWeighted(canvas, 0.7, overlay, 0.3, 0)

        info = f"Vertices: {len(points)}  (need ≥ 3; Enter to confirm)"
        cv2.putText(canvas, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        disp = cv2.resize(canvas, (disp_w, disp_h))
        cv2.imshow(window_name, disp)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:                    # Esc
            cancelled = True; done = True
        elif key in (13, 32):            # Enter / Space
            if len(points) >= 3:
                done = True
            else:
                print("[poly] Need at least 3 points. Keep clicking.")
        elif key in (ord('c'), ord('C')):
            points.clear()

    cv2.destroyWindow(window_name)
    if cancelled or len(points) < 3:
        return np.array([], dtype=np.int32)
    return np.array(points, dtype=np.int32)


# ── Interactive cube-vertex picker ───────────────────────────────────────
#
# Cube coordinate system used by the simulation:
#   x axis: container width  (0 → fw)   — right
#   y axis: container height (0 → fh)   — up
#   z axis: container depth  (0 → d)    — back (away from camera)
#
# Pick order is chosen for typical sauce-tank setup, where the LEFT wall is
# always clearly visible (3 reliable corners), the FRONT-RIGHT corner is
# usually visible if the frame is good, and back-right corners are
# typically obscured by the gate/ruler. The user only needs the 3 left-wall
# points to confirm; more points (especially the 4th) make solvePnP unique.
# Pick order matches the user's NATURAL click order: top-down on the left wall
# (since the left wall is reliably visible), then jump to bottom-front-right.
# Image-y small (high on screen) → image-y large (low on screen).
_CUBE_PICK_ORDER = [
    # idx, label, 3D-coord-fn, is_required
    ("TOP-FRONT-LEFT      (highest on left wall)",       lambda fw, fh, d: (0,  fh, 0), True),
    ("TOP-BACK-LEFT       (slightly behind #1)",         lambda fw, fh, d: (0,  fh, d), True),
    ("BOTTOM-FRONT-LEFT   (low on left wall, x=y=z=0)",  lambda fw, fh, d: (0,  0,  0), True),
    ("BOTTOM-FRONT-RIGHT  (low on right side; needed)",  lambda fw, fh, d: (fw, 0,  0), False),
    # Optional extras:
    ("BOTTOM-BACK-LEFT    (rare)",                       lambda fw, fh, d: (0,  0,  d), False),
    ("TOP-FRONT-RIGHT     (rare)",                       lambda fw, fh, d: (fw, fh, 0), False),
    ("TOP-BACK-RIGHT      (rare)",                       lambda fw, fh, d: (fw, fh, d), False),
    ("BOTTOM-BACK-RIGHT   (rare)",                       lambda fw, fh, d: (fw, 0,  d), False),
]
# Edges between vertex indices (using new order):
#   0 = top-front-left, 1 = top-back-left, 2 = bot-front-left, 3 = bot-front-right
#   4 = bot-back-left,  5 = top-front-right, 6 = top-back-right, 7 = bot-back-right
_CUBE_EDGES = [
    (0, 1),  # top edge (front→back) of left wall
    (0, 2),  # left wall front vertical
    (1, 4),  # left wall back vertical
    (2, 4),  # left wall bottom edge
    (2, 3),  # bottom front edge
    (0, 5),  # top front edge
    (3, 5),  # right wall front vertical
    (5, 6),  # right wall top edge
    (1, 6),  # top back edge
    (3, 7),  # right wall bottom edge
    (5, 7),  # right wall back vertical (via 7)
    (4, 7),  # bottom back edge
    (6, 7),  # right wall back vertical
]


def pick_cube_vertices(
    bg_img: np.ndarray,
    fluid_w_m: float,
    fluid_h_m: float,
    depth_m: float = 0.04,
    window_name: str = "Click cube vertices (3 required + extras)",
) -> tuple:
    """
    Show bg_img and let the user click cube vertices in the prescribed order.
    Returns (verts_2d, verts_3d) suitable for cv2.solvePnP.

    Required (3 left-wall corners — always identifiable):
      1. bottom-front-LEFT
      2. top-front-LEFT
      3. top-back-LEFT
    Strongly recommended (disambiguates camera pose):
      4. bottom-front-RIGHT   (visible if frame is good)
    Optional extras (any visible back-right corners boost robustness).

    Right-click undoes last point. Enter confirms (≥3 points; ≥4 recommended).
    """
    h, w = bg_img.shape[:2]
    scale = 1.0
    if w > 1400:
        scale = 1400.0 / w
    disp_w, disp_h = int(w * scale), int(h * scale)

    pts2d: list[tuple[int, int]] = []
    n_required = sum(1 for o in _CUBE_PICK_ORDER if o[2])  # = 3

    def mouse_cb(event, x, y, flags, param):
        ox, oy = int(x / scale), int(y / scale)
        if event == cv2.EVENT_LBUTTONDOWN and len(pts2d) < len(_CUBE_PICK_ORDER):
            pts2d.append((ox, oy))
        elif event == cv2.EVENT_RBUTTONDOWN and pts2d:
            pts2d.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_w, disp_h)
    cv2.setMouseCallback(window_name, mouse_cb)

    print("\n" + "=" * 60)
    print("  CLICK CUBE VERTICES")
    print("=" * 60)
    print("  Container dimensions:")
    print(f"    W = {fluid_w_m*100:.2f} cm  (x axis)")
    print(f"    H = {fluid_h_m*100:.2f} cm  (y axis, vertical-up)")
    print(f"    D = {depth_m*100:.2f} cm  (z axis, depth)")
    print()
    print("  Click in order (banner in UI shows the next vertex):")
    for i, (lbl, _, req) in enumerate(_CUBE_PICK_ORDER):
        tag = "[REQUIRED]" if req else "[optional]"
        print(f"    {i+1}. {tag}  {lbl}")
    print()
    print("  L-click  : add vertex   |  R-click : undo last")
    print(f"  Enter    : confirm (need ≥ {n_required}; 4+ recommended)")
    print("  Esc      : cancel")
    print("=" * 60 + "\n")

    cancelled = False
    while True:
        canvas = bg_img.copy()
        for i, (px, py) in enumerate(pts2d):
            cv2.circle(canvas, (px, py), 8, (0, 255, 0), -1)
            cv2.putText(canvas, str(i+1), (px+12, py-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        for a, b in _CUBE_EDGES:
            if a < len(pts2d) and b < len(pts2d):
                cv2.line(canvas, pts2d[a], pts2d[b], (0, 255, 255), 1)

        if len(pts2d) < len(_CUBE_PICK_ORDER):
            lbl, _, req = _CUBE_PICK_ORDER[len(pts2d)]
            tag = "REQUIRED" if req else "optional"
            banner = f"Click vertex {len(pts2d)+1}/{len(_CUBE_PICK_ORDER)}  [{tag}]: {lbl}"
            color = (0, 165, 255) if req else (0, 200, 200)
        else:
            banner = "All vertices clicked — press Enter to confirm."
            color = (0, 255, 0)
        cv2.putText(canvas, banner, (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        if len(pts2d) >= 4:
            ready = "READY (4+ points → unique pose)"
            cclr = (0, 255, 0)
        elif len(pts2d) >= n_required:
            ready = f"OK with {len(pts2d)} (might be ambiguous; click 4th if possible)"
            cclr = (0, 200, 200)
        else:
            ready = f"Need {n_required - len(pts2d)} more required points"
            cclr = (0, 0, 255)
        cv2.putText(canvas, f"Clicked: {len(pts2d)}  |  {ready}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cclr, 2)

        disp = cv2.resize(canvas, (disp_w, disp_h))
        cv2.imshow(window_name, disp)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cancelled = True; break
        elif key in (13, 32):
            if len(pts2d) >= n_required:
                break
            else:
                print(f"[cube] Need at least {n_required} required vertices "
                      f"(got {len(pts2d)}).")

    cv2.destroyWindow(window_name)
    if cancelled or len(pts2d) < n_required:
        return None, None

    verts_2d = np.array(pts2d, dtype=np.float32)
    verts_3d = np.array(
        [_CUBE_PICK_ORDER[i][1](fluid_w_m, fluid_h_m, depth_m)
         for i in range(len(pts2d))],
        dtype=np.float32,
    )
    return verts_2d, verts_3d


# Click order — 3 required + up to 2 optional points.
# 3 required: TFL, TBL, BFL (the 3 reliably-visible left-wall corners)
# Optional 4th: BFR (front-bottom-right, off the left plane)
# Optional 5th: TBR (top-back-right, opposite diagonal to BFL)
_LEFT_FACE_LABELS = [
    "TOP-FRONT-LEFT     (TFL: top-left of left wall, near camera)",
    "TOP-BACK-LEFT      (TBL: top-left of left wall, away from camera)",
    "BOTTOM-FRONT-LEFT  (BFL: bottom of left wall, near camera — origin)",
    "BOTTOM-FRONT-RIGHT (BFR: optional; bottom of front-right opening panel)",
    "TOP-BACK-RIGHT     (TBR: optional; top of back-right edge)",
]
# 3D coords for each click; last two are off the left wall (x=fw)
_LEFT_FACE_3D_FNS = [
    lambda fw, fh, d: (0,   fh, 0),     # 1. TFL  (left wall)
    lambda fw, fh, d: (0,   fh, d),     # 2. TBL  (left wall)
    lambda fw, fh, d: (0,   0,  0),     # 3. BFL  (left wall, origin)
    lambda fw, fh, d: (fw,  0,  0),     # 4. BFR  (front-right bottom)
    lambda fw, fh, d: (fw,  fh, d),     # 5. TBR  (top-back-right corner)
]


def pick_left_face_corners(
    bg_img: np.ndarray,
    fluid_w_m: float,
    fluid_h_m: float,
    depth_m: float,
    window_name: str = "Click 3-5 cube corners (3 LEFT + up to 2 RIGHT optional)",
) -> Optional[np.ndarray]:
    """
    GUI: user clicks 3-5 cube corners.
      - 3 REQUIRED: TFL, TBL, BFL (3 reliably-visible left-wall corners,
                    coplanar on x=0).
      - up to 2 OPTIONAL: BFR (4), TBR (5) — both at x=fw, off the left
                          plane. Either one breaks coplanarity → unique
                          PnP solution; both → over-determined (best).

    Returns (N, 2) np.ndarray with 3 ≤ N ≤ 5, or None on cancel.
    """
    h, w = bg_img.shape[:2]
    scale = 1.0
    if w > 1400:
        scale = 1400.0 / w
    disp_w, disp_h = int(w * scale), int(h * scale)
    pts: list[tuple[int, int]] = []

    def cb(event, x, y, flags, param):
        ox, oy = int(x / scale), int(y / scale)
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((ox, oy))
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_w, disp_h)
    cv2.setMouseCallback(window_name, cb)

    n_required = 3
    n_max      = 5

    print("\n" + "=" * 60)
    print("  CLICK 3-5 CUBE CORNERS")
    print("=" * 60)
    print(f"  Cube dims:  W={fluid_w_m*100:.2f}cm × H={fluid_h_m*100:.2f}cm × "
          f"D={depth_m*100:.2f}cm")
    print()
    print("  Click in this order (4th is optional but recommended):")
    for i, lbl in enumerate(_LEFT_FACE_LABELS):
        tag = "[REQUIRED]" if i < n_required else "[optional]"
        print(f"    {i+1}. {tag}  {lbl}")
    print()
    print(f"  L-click=add  R-click=undo  Enter=confirm (need ≥{n_required})  Esc=cancel")
    print("=" * 60 + "\n")

    cancelled = False
    while True:
        canvas = bg_img.copy()
        for i, (px, py) in enumerate(pts):
            cv2.circle(canvas, (px, py), 8, (0, 255, 0), -1)
            cv2.putText(canvas, str(i+1), (px+12, py-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Edges connecting clicked points (use indices in the labels list):
        #   0=TFL  1=TBL  2=BFL  3=BFR  4=TBR
        edges = [
            (0, 1),  # TFL → TBL : top edge of left wall
            (0, 2),  # TFL → BFL : front edge of left wall
            (2, 3),  # BFL → BFR : bottom front edge
            (1, 4),  # TBL → TBR : top back edge
            (3, 4),  # BFR → TBR : right wall diagonal (visual aid)
        ]
        for a, b in edges:
            if a < len(pts) and b < len(pts):
                cv2.line(canvas, pts[a], pts[b], (0, 255, 255), 2)
        # Tint left-wall triangle once 3 points are clicked
        if len(pts) >= 3:
            ov = canvas.copy()
            tri = np.array([pts[0], pts[1], pts[2]], dtype=np.int32)
            cv2.fillPoly(ov, [tri], (0, 100, 200))
            canvas = cv2.addWeighted(canvas, 0.85, ov, 0.15, 0)

        if len(pts) < n_max:
            tag = "REQUIRED" if len(pts) < n_required else "optional"
            banner = (f"Click vertex {len(pts)+1}/{n_max} [{tag}]: "
                      f"{_LEFT_FACE_LABELS[len(pts)]}")
            color = (0, 165, 255) if len(pts) < n_required else (0, 200, 200)
        else:
            banner = "All 4 corners clicked — press Enter to confirm."
            color = (0, 255, 0)
        cv2.putText(canvas, banner, (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if len(pts) == 3:
            cv2.putText(canvas, "OK with 3 — but click #4 if you can!",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 200, 255), 2)

        disp = cv2.resize(canvas, (disp_w, disp_h))
        cv2.imshow(window_name, disp)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cancelled = True; break
        elif key in (13, 32):
            if len(pts) >= n_required:
                break
            else:
                print(f"[left-face] Need at least {n_required} points "
                      f"(got {len(pts)}).")

    cv2.destroyWindow(window_name)
    if cancelled or len(pts) < n_required:
        return None
    return np.array(pts, dtype=np.float32)


def solve_camera_from_left_face(
    verts_2d: np.ndarray,
    fluid_w_m: float,
    fluid_h_m: float,
    depth_m: float,
    K: np.ndarray,
    R_init: Optional[np.ndarray] = None,
    t_init: Optional[np.ndarray] = None,
) -> tuple:
    """
    Solve camera (R, t) given 3-5 cube corners.

    3D world coords (CLICK ORDER):
       0. TOP-FRONT-LEFT     = (0,  fh, 0)
       1. TOP-BACK-LEFT      = (0,  fh, d)
       2. BOTTOM-FRONT-LEFT  = (0,  0,  0)         ← origin (3 left-wall pts coplanar)
       3. BOTTOM-FRONT-RIGHT = (fw, 0,  0)         ← optional, BREAKS coplanarity
       4. TOP-BACK-RIGHT     = (fw, fh, d)         ← optional 5th, opposite diagonal

    With 4-5 points → ITERATIVE PnP with ChArUco prior, well-conditioned.
    With 3 coplanar points → ITERATIVE + prior (rank-deficient without prior).
    """
    n = len(verts_2d)
    if n < 3 or n > 5:
        return None

    verts_3d_all = np.array([
        [0,         fluid_h_m, 0       ],   # 0 TFL
        [0,         fluid_h_m, depth_m ],   # 1 TBL
        [0,         0,         0       ],   # 2 BFL  (origin)
        [fluid_w_m, 0,         0       ],   # 3 BFR  (off the left plane!)
        [fluid_w_m, fluid_h_m, depth_m ],   # 4 TBR  (opposite diagonal)
    ], dtype=np.float32)
    verts_3d = verts_3d_all[:n]

    img = verts_2d.reshape(-1, 1, 2).astype(np.float32)
    obj = verts_3d.reshape(-1, 1, 3).astype(np.float32)

    def _err(rv, tv):
        proj, _ = cv2.projectPoints(obj, rv, tv, K, None)
        return float(np.linalg.norm(proj.reshape(-1, 2) - verts_2d, axis=1).mean())

    if R_init is None or t_init is None:
        print("[left-face PnP] WARN: no ChArUco prior — solver may diverge")

    rvec_init = cv2.Rodrigues(R_init)[0].astype(np.float64) if R_init is not None \
                else np.zeros((3, 1), dtype=np.float64)
    tvec_init = t_init.reshape(3, 1).astype(np.float64) if t_init is not None \
                else np.array([[0.], [0.], [0.5]], dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        obj, img, K, None, rvec_init, tvec_init,
        useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        print(f"[left-face PnP] ITERATIVE solver failed ({n} points)")
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    err = _err(rvec, tvec)
    coplanarity = "coplanar" if n == 3 else "non-coplanar"
    print(f"[left-face PnP] {n} corners ({coplanarity}) + ChArUco prior, "
          f"reproj err = {err:.2f} px")
    return R, t


def solve_camera_from_cube(
    verts_2d: np.ndarray,
    verts_3d: np.ndarray,
    K: np.ndarray,
    R_init: Optional[np.ndarray] = None,
    t_init: Optional[np.ndarray] = None,
) -> tuple:
    """
    Solve camera (R, t) given 3-8 cube vertex correspondences.

    With 3 points (the 3 left-wall corners) the PnP is ambiguous (P3P
    returns up to 4 solutions). If R_init/t_init are provided (e.g. from
    ChArUco), we pick the solution closest to that prior; otherwise we
    pick the solution that places the camera on the "front" side of the
    cube (positive z observation).

    With ≥4 points we use ITERATIVE solver (LM-refined).

    Returns (R_3x3, t_3x1) or None on failure.
    """
    n = len(verts_2d)
    obj = verts_3d.reshape(-1, 1, 3).astype(np.float32)
    img = verts_2d.reshape(-1, 1, 2).astype(np.float32)

    def _eval(rvec, tvec):
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
        return float(np.linalg.norm(proj.reshape(-1, 2) - verts_2d, axis=1).mean())

    if n == 3:
        # P3P returns up to 4 candidate solutions
        ok, rvecs, tvecs = cv2.solveP3P(obj, img, K, None, cv2.SOLVEPNP_P3P)
        if not ok or len(rvecs) == 0:
            return None
        candidates = []
        for rv, tv in zip(rvecs, tvecs):
            R_cand, _ = cv2.Rodrigues(rv)
            t_cand    = tv.reshape(3)
            cam_z     = (-R_cand.T @ t_cand)[2]   # camera Z position
            cube_in_front = (R_cand @ verts_3d.mean(axis=0) + t_cand)[2] > 0
            err  = _eval(rv, tv)
            if R_init is not None:
                rot_diff = np.linalg.norm(R_cand - R_init)
                t_diff   = np.linalg.norm(t_cand - t_init)
                score = err + 50 * rot_diff + 5 * t_diff
            else:
                # Heuristic: camera should be in +z half-space looking -z
                score = err if cube_in_front else err + 1e6
            candidates.append((score, err, R_cand, t_cand))
        candidates.sort(key=lambda x: x[0])
        score, err, R, t = candidates[0]
        print(f"[cube-P3P] 3 vertices, picked solution {0+1}/{len(rvecs)}, "
              f"reprojection err = {err:.2f} px")
        return R, t

    # 4+ points
    if R_init is not None and t_init is not None:
        # ITERATIVE with extrinsic guess works for 4+ points
        rvec_init = cv2.Rodrigues(R_init)[0].astype(np.float64)
        tvec_init = t_init.reshape(3, 1).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            obj, img, K, None, rvec_init, tvec_init,
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
        )
    elif n >= 4:
        # No prior: use EPnP (works with 4+ points without DLT init)
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, None,
                                      flags=cv2.SOLVEPNP_EPNP)
    else:
        return None

    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    err = _eval(rvec, tvec)
    print(f"[cube-PnP] {n} vertices, mean reprojection error = {err:.2f} px")
    return R, t


# ── Binarization ─────────────────────────────────────────────────────────

def binarize_frame(
    frame: np.ndarray,
    roi_mask: np.ndarray,
    v_thresh: int = V_THRESH,
    dilate_px: int = 0,
) -> np.ndarray:
    """
    Binarize a single frame: dark pixels inside ROI → black, everything else → white.

    Simple approach:
      1. Convert to HSV
      2. Pixels with V < v_thresh AND inside roi_mask → foreground (black)
      3. Reject skin-coloured pixels (hand)
      4. Morphological cleanup
      5. Everything outside roi_mask → white

    Returns: uint8 image (255=white background, 0=black foreground).
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Skin rejection mask (hand)
    skin = cv2.inRange(hsv, SKIN_LO, SKIN_HI)
    not_skin = cv2.bitwise_not(skin)

    # Identify white paper surface (these pixels are NEVER foreground,
    # even if they are inside the ROI). Paper is bright and low-saturation.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    paper_bright = (gray > 200).astype(np.uint8) * 255

    # Dark pixels inside ROI, excluding paper surface
    dark = (hsv[:, :, 2] < v_thresh).astype(np.uint8) * 255
    dark = cv2.bitwise_and(dark, roi_mask)
    dark = cv2.bitwise_and(dark, not_skin)
    dark = cv2.bitwise_and(dark, cv2.bitwise_not(paper_bright))

    # Morphological cleanup (light close to bridge small gaps,
    # but not so aggressive that it connects the container to
    # distant dark background through empty space)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k5, iterations=2)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k5, iterations=2)

    # Clip to ROI (morph ops may expand beyond boundary)
    dark = cv2.bitwise_and(dark, roi_mask)

    # Keep largest connected component (removes noise)
    nl, lab, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    if nl > 1:
        best = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        dark = ((lab == best) * 255).astype(np.uint8)

    # Fill holes
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(dark)
    cv2.drawContours(filled, cnts, -1, 255, -1)

    # Final light cleanup
    filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, k5, iterations=1)

    # Optional outward dilation — recovers the glass-reflection rim when
    # V-thresh alone can't reach pixels brighter than v_thresh.  Clipped
    # back to ROI so it can't spill into the table.
    if dilate_px > 0:
        k_d = 2 * dilate_px + 1
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_d, k_d))
        filled = cv2.dilate(filled, kern, iterations=1)
        filled = cv2.bitwise_and(filled, roi_mask)

    # Output: white background, black foreground
    return 255 - filled


# ── Main pipeline ────────────────────────────────────────────────────────

def _resolve_data_dir(data_dir: str) -> dict:
    """
    Given an experiment directory (e.g. data/ref_Tonkatsu_2.5_3.0_1/1st),
    auto-detect:
      - video       : first *.mov file
      - bg_img      : first *.JPG / *.jpg / *.jpeg file
      - settings_xml: settings.xml in the same dir or parent dir
    Returns a dict with keys: video, bg_img, settings_xml (values may be None).
    """
    import glob
    d = Path(data_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    def _find(patterns):
        for pat in patterns:
            hits = sorted(glob.glob(str(d / pat)))
            if hits:
                return hits[0]
        return None

    video      = _find(["*.mov", "*.MOV", "*.mp4", "*.MP4"])
    bg_img     = _find(["*.JPG", "*.jpg", "*.jpeg", "*.JPEG", "*.png", "*.PNG"])
    # Prefer settings.xml in same dir; fall back to parent
    xml_local  = d / "settings.xml"
    xml_parent = d.parent / "settings.xml"
    settings_xml = str(xml_local) if xml_local.exists() else \
                   (str(xml_parent) if xml_parent.exists() else None)

    print(f"[data_dir] {data_dir}")
    print(f"  video       : {video or '(not found)'}")
    print(f"  bg_img      : {bg_img or '(not found)'}")
    print(f"  settings.xml: {settings_xml or '(not found)'}")
    return {"video": video, "bg_img": bg_img, "settings_xml": settings_xml}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Automated video -> config_XX.png + camera_params.xml pipeline for ViRheometry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Input: either --data_dir (recommended) or explicit --video ─────────
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data_dir", type=str, metavar="DIR",
                     help="Experiment directory containing *.mov, *.JPG, settings.xml. "
                          "Auto-detects video / background / dimensions. "
                          "Example: data/ref_Tonkatsu_2.5_3.0_1/1st")
    src.add_argument("--video", type=str,
                     help="Path to input video (alternative to --data_dir)")

    p.add_argument("--out_dir", type=str, default=None,
                   help="Output directory for config PNGs and camera_params.xml. "
                        "Defaults to <data_dir>/output/ when using --data_dir.")

    # ── Frame control ───────────────────────────────────────────────────────
    p.add_argument("--frame0", type=int, default=None,
                   help="Override frame 0 index (raw 240fps index). "
                        "Auto-detected if not specified.")
    p.add_argument("--auto-frame0", action="store_true",
                   help="Skip the manual frame-picker GUI and use the "
                        "auto-detected dam-break frame directly. "
                        "(Default: GUI picker opens with auto frame as hint.)")
    p.add_argument("--frame0-offset", type=int, default=0,
                   help="Shift detected frame0 by N raw frames "
                        "(positive = later, negative = earlier). "
                        "Applied after auto-detection or --frame0.")
    p.add_argument("--raw-fps", type=int, default=RAW_FPS,
                   help=f"Recording FPS (default: {RAW_FPS})")
    p.add_argument("--target-fps", type=int, default=TARGET_FPS,
                   help=f"Extraction FPS (default: {TARGET_FPS})")

    # ROI polygon (single ROI used for all configs)
    p.add_argument("--roi-poly", type=str, default=None,
                   help="Load ROI polygon from .npy (covers container + max "
                        "puddle; used for ALL configs)")
    p.add_argument("--roi-poly-00", type=str, default=None,
                   help=argparse.SUPPRESS)  # deprecated; calibration now uses cube-vertex pick on frame 0

    # Threshold tuning
    p.add_argument("--v-thresh", type=int, default=V_THRESH,
                   help=f"V-channel threshold for dark pixels (default: {V_THRESH})")
    p.add_argument("--dilate", type=int, default=0, metavar="PX",
                   help="Dilate the binary mask by PX pixels after "
                        "morphological cleanup (clipped to ROI). Useful "
                        "when V-thresh can't reach the glass-reflection rim "
                        "around the fluid. Default: 0 (disabled).")

    # Output options
    p.add_argument("--save-debug", action="store_true",
                   help="Save debug overlays alongside config PNGs")
    p.add_argument("--n-configs", type=int, default=N_CONFIGS,
                   help=f"Number of config frames to extract (default: {N_CONFIGS})")
    p.add_argument("--color-profile-ref", type=str, default=None, metavar="PNG",
                   help="Reference PNG whose ICC color profile will be copied "
                        "into all output config_XX.png files (so they can be "
                        "pasted into the same Photoshop document as PR-exported "
                        "frames without a profile-mismatch dialog).")

    # Validation
    p.add_argument("--validate", type=str, default=None, metavar="PS_DIR",
                   help="After generating configs, cross-match against PS "
                        "reference directory to verify timing alignment.")

    # ── Camera calibration (Step 4) ──────────────────────────────────────
    calib = p.add_argument_group("camera calibration (optional)")
    calib.add_argument("--bg_img", type=str, default=None,
                       help="Background PNG with ArUco/ChArUco markers. "
                            "If provided, runs camera calibration after extracting "
                            "configs and outputs camera_params.xml.")
    calib.add_argument("--fluid_w", type=float, default=None,
                       help="Fluid container width in cm (e.g. 6.7). "
                            "Required when --bg_img is used.")
    calib.add_argument("--fluid_h", type=float, default=None,
                       help="Fluid container height in cm (e.g. 3.5). "
                            "Required when --bg_img is used.")
    calib.add_argument("--settings_xml", type=str, default=None,
                       help="Path to settings.xml to read fluid_w/fluid_h from. "
                            "Alternative to specifying --fluid_w/--fluid_h directly.")
    calib.add_argument("--skip_calib", action="store_true",
                       help="Skip camera calibration even if --bg_img is provided.")
    calib.add_argument("--no-cube-pick", action="store_true",
                       help="Skip the cube-outline picker GUI entirely.")
    calib.add_argument("--cube-depth", type=float, default=4.0,
                       help="Cube depth in cm (z extent of the container). Default: 4.0")
    calib.add_argument("--cube-frame", type=int, default=None,
                       help="Frame index to show for cube-outline drawing "
                            "(default: frame 0).")
    calib.add_argument("--refine",
                       choices=["none", "left-face", "extrinsic-charuco", "edge"],
                       default="edge",
                       help="Camera refinement after ChArUco initial calibration:"
                            "  'none' → use ChArUco theta0 directly; "
                            "  'left-face' → planar PnP from 4 LEFT-face corners; "
                            "  'extrinsic-charuco' → SIFT feature match bg↔frame0; "
                            "  'edge' (default) → minimise chamfer distance between "
                            "rendered cube edges and config_00 container edges. "
                            "K fixed, only R,t adjusted.")
    calib.add_argument("--calib-poly", type=str, default=None, metavar="NPY",
                       help="Load a saved CALIBRATION polygon (.npy, shape (N,2)) "
                            "drawn on frame 0.  The polygon encloses ONLY the "
                            "clean container outline (exclude fluid tongue, hand "
                            "shadow, etc).  Pixels outside the polygon are set "
                            "to white (background) BEFORE camera calibration, "
                            "so the optimiser fits only trustworthy edges. "
                            "Original config_00.png is untouched.")
    calib.add_argument("--calib-poly-pick", action="store_true",
                       help="Open an interactive polygon picker on frame 0 "
                            "(after frame 0 is selected).  Draw around the "
                            "clean container outline, skip contaminated right "
                            "side.  Saved to <out_dir>/calib_target_poly.npy.")
    calib.add_argument("--safety-rot-deg", type=float, default=3.0,
                       help="Reject edge-refinement if rotation drift from "
                            "ChArUco exceeds this many degrees. Default: 3.0. "
                            "Relax (e.g. 5.0–6.0) when the ChArUco prior is "
                            "known to be farther off than usual.")
    calib.add_argument("--safety-t-cm", type=float, default=3.0,
                       help="Reject edge-refinement if translation drift from "
                            "ChArUco exceeds this many cm. Default: 3.0.")
    calib.add_argument("--left-face-corners", type=str, default=None, metavar="NPY",
                       help="Load saved 4-corner LEFT-face picks from .npy.")
    # Deprecated alias kept for back-compat
    calib.add_argument("--cube-vertices", type=str, default=None,
                       help=argparse.SUPPRESS)

    return p.parse_args()


def _load_icc_profile(ref_png_path: Optional[str]) -> Optional[bytes]:
    """Extract ICC profile bytes from a reference PNG, or return None."""
    if not ref_png_path:
        return None
    if not os.path.isfile(ref_png_path):
        print(f"[ICC] WARN: reference PNG not found: {ref_png_path}")
        return None
    try:
        from PIL import Image as PILImage
        with PILImage.open(ref_png_path) as im:
            icc = im.info.get("icc_profile")
        if icc:
            print(f"[ICC] Loaded {len(icc)} byte ICC profile from: {ref_png_path}")
            return icc
        else:
            print(f"[ICC] WARN: reference PNG has no ICC profile: {ref_png_path}")
            return None
    except Exception as e:
        print(f"[ICC] ERROR: {e}")
        return None


def _imwrite_with_icc(path: str, binary: np.ndarray,
                      icc_profile: Optional[bytes]) -> None:
    """
    Save a binary mask as an RGB PNG (PR-compatible format).

    Premiere Pro exports frames as RGBA PNGs; macOS/Photoshop treat
    those as BT.709. If we save as grayscale (mode 'L'), PS sees a
    different color space and prompts "profile mismatch" when both
    files are combined. To avoid this, we replicate the 0/255 mask
    across all three RGB channels so the PNG has IHDR color-type 2
    (RGB) and matches PR's output format.

    If icc_profile is provided, it is embedded as an iCCP chunk.
    """
    from PIL import Image as PILImage

    if binary.ndim == 2:
        # Replicate single channel into RGB
        rgb = np.stack([binary, binary, binary], axis=-1)  # (H, W, 3)
    else:
        # Already a color image (BGR from OpenCV) → convert to RGB
        rgb = cv2.cvtColor(binary, cv2.COLOR_BGR2RGB) if binary.shape[2] == 3 \
              else binary

    im = PILImage.fromarray(rgb.astype(np.uint8), mode="RGB")
    save_kwargs = {"format": "PNG"}
    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile
    im.save(path, **save_kwargs)


def save_debug_overlay(frame: np.ndarray, mask: np.ndarray,
                       path: str, label: str) -> None:
    """Save an overlay image showing the detected region on the original frame."""
    canvas = frame.copy()
    red_overlay = np.zeros_like(canvas)
    red_overlay[:, :, 2] = mask
    canvas = cv2.addWeighted(canvas, 0.7, red_overlay, 0.3, 0)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, cnts, -1, (0, 255, 0), 2)

    cv2.putText(canvas, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, canvas)


def main() -> None:
    args = parse_args()

    # ── Resolve --data_dir into individual paths ──────────────────────────
    if args.data_dir:
        detected = _resolve_data_dir(args.data_dir)
        if detected["video"] is None:
            sys.exit(f"[error] No video file (*.mov) found in: {args.data_dir}")
        args.video = detected["video"]
        # Only override bg_img/settings_xml if not already specified via CLI
        if args.bg_img is None:
            args.bg_img = detected["bg_img"]
        if args.settings_xml is None:
            args.settings_xml = detected["settings_xml"]
        # Default out_dir to <data_dir>/output/
        if args.out_dir is None:
            args.out_dir = str(Path(args.data_dir) / "output")

    if args.out_dir is None:
        sys.exit("[error] --out_dir is required when using --video directly")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = open_video(args.video)
    info = get_video_info(cap)
    print(f"[video] {info['width']}x{info['height']}, "
          f"{info['fps']:.1f}fps (avg), {info['n_frames']} frames")

    # ── Build timestamp index for VFR-safe seeking ──
    ts_index = build_timestamp_index(args.video)
    true_fps = get_true_fps(args.video)
    if true_fps is not None:
        print(f"[video] True (nominal) frame rate: {true_fps:.0f}fps")
    use_timestamp = ts_index is not None

    # ── Step 1: Determine frame 0 ─────────────────────────────────────────
    fps_eff = true_fps if true_fps else info["fps"]
    frames_per_config = max(1, int(round(fps_eff / args.target_fps)))

    if args.frame0 is not None:
        frame0_idx = args.frame0
        print(f"[frame0] Using user-specified frame 0 = {frame0_idx}")
    else:
        raw = detect_dam_break(cap)
        # Auto-detected return_idx = the moment the gate has fully lifted
        # (= clean initial rectangle). Use this DIRECTLY as cfg_00 so that
        # cfg_00 = initial state, matching Sim convention.
        # (For PS workflow alignment, PS cfg_(i+1) ↔ Auto cfg_i.)
        print(f"[frame0] Auto hint: dam-break return = {raw}")

        if args.auto_frame0:
            frame0_idx = raw
            print(f"[frame0] --auto-frame0 set → using {frame0_idx} without GUI")
        else:
            picked = pick_frame_interactive(cap, hint_frame=raw)
            if picked < 0:
                print("[frame0] Picker cancelled. Exiting.")
                cap.release()
                sys.exit(1)
            frame0_idx = picked
            print(f"[frame0] User picked frame 0 = {frame0_idx}")

    if args.frame0_offset != 0:
        old = frame0_idx
        frame0_idx = max(0, frame0_idx + args.frame0_offset)
        print(f"[frame0] Applied offset {args.frame0_offset:+d}: "
              f"{old} → {frame0_idx}")

    # ── Compute config frame targets ──
    # METHOD: Timestamp-based (matches Premiere Pro 24fps export)
    #   config_k is at timestamp: T0 + k / target_fps
    # This is immune to VFR dropped frames.
    if use_timestamp and frame0_idx < len(ts_index):
        t0 = ts_index[frame0_idx]  # frame0 timestamp in seconds
        config_interval = 1.0 / args.target_fps  # e.g., 1/24 = 0.04167s

        frame_indices = []
        for k in range(args.n_configs):
            target_t = t0 + k * config_interval
            # Find frame with closest timestamp
            idx = int(np.argmin(np.abs(ts_index - target_t)))
            frame_indices.append(idx)

        print(f"[timing] Timestamp-based seeking (Premiere Pro equivalent)")
        print(f"[timing] T0={t0:.6f}s, interval={config_interval*1000:.2f}ms")
        for k, idx in enumerate(frame_indices):
            target_t = t0 + k * config_interval
            actual_t = ts_index[idx]
            err_ms = (actual_t - target_t) * 1000
            print(f"  config_{k:02d}: frame[{idx}]  "
                  f"t={actual_t:.6f}s  err={err_ms:+.1f}ms")
    else:
        # Fallback: frame-index-based (for CFR videos or if ffprobe unavailable)
        effective_fps = true_fps or info["fps"]
        frame_step = max(1, round(effective_fps / args.target_fps))
        frame_indices = [frame0_idx + i * frame_step
                         for i in range(args.n_configs)]
        max_frame = info["n_frames"] - 1
        frame_indices = [min(idx, max_frame) for idx in frame_indices]
        print(f"[timing] Frame-index fallback: step={frame_step} "
              f"(fps={effective_fps:.1f})")
        print(f"[timing] ⚠  For VFR videos, install ffprobe for accurate timing")

    print(f"[frames] Config frame indices: {frame_indices}")

    h_f, w_f = info["height"], info["width"]
    poly_save  = str(out_dir / "roi_polygon.npy")

    # ── Load ICC color profile (optional, for Photoshop compatibility) ──
    icc_profile = _load_icc_profile(args.color_profile_ref)

    # ── Step 2: (optional) LEFT-FACE corner pick for PnP refinement ──────
    left_face_pts = None            # (4, 2) corners (used by refine='left-face')
    left_face_save = str(out_dir / "left_face_corners.npy")

    if args.bg_img and not args.skip_calib and (not args.no_cube_pick):
        outline_frame_idx = args.cube_frame if args.cube_frame is not None \
                            else frame0_idx

        if args.refine == "left-face":
            # ── 4-corner LEFT-FACE picker ─────────────────────────────────
            if args.left_face_corners and os.path.isfile(args.left_face_corners):
                left_face_pts = np.load(args.left_face_corners)
                print(f"[left-face] Loaded 4 corners from {args.left_face_corners}")
                if os.path.abspath(args.left_face_corners) != os.path.abspath(left_face_save):
                    np.save(left_face_save, left_face_pts)
            else:
                from_settings = (_parse_settings_xml(args.settings_xml)
                                 if args.settings_xml and os.path.isfile(args.settings_xml)
                                 else {})
                fw_cm = args.fluid_w if args.fluid_w is not None else from_settings.get("W")
                fh_cm = args.fluid_h if args.fluid_h is not None else from_settings.get("H")
                depth_cm = (from_settings.get("depth") if from_settings.get("depth") is not None
                            else args.cube_depth)
                if fh_cm is None or fw_cm is None:
                    print("[left-face] Skipping pick — fluid W/H unknown.")
                else:
                    img = read_frame(cap, outline_frame_idx)
                    print(f"\n[left-face] Click 3-4 cube corners on frame "
                          f"{outline_frame_idx}")
                    left_face_pts = pick_left_face_corners(
                        img, fw_cm/100.0, fh_cm/100.0, depth_cm/100.0,
                    )
                    if left_face_pts is None:
                        print("[left-face] cancelled.")
                    else:
                        np.save(left_face_save, left_face_pts)
                        print(f"[left-face] Saved → {left_face_save}")

    # ── Step 3: ROI polygon (for binarization of cfg_00~08) ──────────────
    if args.roi_poly is not None:
        poly = np.load(args.roi_poly)
        print(f"[ROI] Polygon loaded from {args.roi_poly} ({len(poly)} verts)")
        if os.path.abspath(args.roi_poly) != os.path.abspath(poly_save):
            np.save(poly_save, poly)
            print(f"[ROI] Copied to {poly_save}")
    else:
        last_frame_idx = frame_indices[-1]
        print(f"\n[ROI] Draw ROI on frame {last_frame_idx} (config_08, max puddle)")
        poly = pick_roi_polygon(
            cap, last_frame_idx, frame_step=10,
            window_name="ROI - container + max puddle extent",
        )
        if poly.size == 0:
            print("[ROI] Cancelled. Exiting.")
            cap.release()
            sys.exit(1)
        np.save(poly_save, poly)
        print(f"[ROI] Saved → {poly_save}")

    # Build ROI mask (used for ALL configs)
    roi_mask = np.zeros((h_f, w_f), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [poly.reshape(-1, 1, 2)], 255)
    print(f"[ROI] Polygon: {len(poly)} verts, area={int(roi_mask.sum()//255)} px")

    # ── Step 4: Process each frame (binarize + save config_XX.png) ───────
    dilate_note = f", dilate={args.dilate}px" if args.dilate > 0 else ""
    print(f"\n[process] Binarizing {args.n_configs} frames (V < {args.v_thresh}{dilate_note})...")
    for config_idx, frame_idx in enumerate(frame_indices):
        frame = read_frame(cap, frame_idx)
        binary = binarize_frame(frame, roi_mask,
                                v_thresh=args.v_thresh,
                                dilate_px=args.dilate)

        out_path = out_dir / f"config_{config_idx:02d}.png"
        _imwrite_with_icc(str(out_path), binary, icc_profile)
        black_px = (binary == 0).sum()
        total_px = binary.shape[0] * binary.shape[1]
        print(f"  config_{config_idx:02d}.png  frame={frame_idx}  "
              f"black={black_px} ({100*black_px/total_px:.1f}%)")

        if args.save_debug:
            debug_mask = (255 - binary)
            debug_path = out_dir / f"debug_{config_idx:02d}.png"
            save_debug_overlay(frame, debug_mask, str(debug_path),
                               f"config_{config_idx:02d} (frame {frame_idx})")

    cap.release()
    print(f"\n[done] {args.n_configs} config images saved to {out_dir}/")

    # Copy settings.xml into out_dir if we have it (Simulation needs it)
    if args.settings_xml and os.path.isfile(args.settings_xml):
        import shutil
        dst = out_dir / "settings.xml"
        if os.path.abspath(args.settings_xml) != os.path.abspath(str(dst)):
            shutil.copy2(args.settings_xml, dst)
            print(f"[settings] Copied settings.xml → {dst}")

    # ── Step 4b: (optional) Calibration polygon ─────────────────────────
    # User draws a polygon on frame 0 (BGR) around the CLEAN container
    # outline, excluding contaminated regions (fluid tongue, hand shadow).
    # The resulting polygon is used ONLY for camera calibration: pixels
    # outside the polygon in config_00 are forced to white (background)
    # before being fed to the calibration optimiser.
    # config_00.png on disk is NOT modified.
    calib_poly = None
    calib_poly_save = str(out_dir / "calib_target_poly.npy")
    if args.calib_poly is not None:
        calib_poly = np.load(args.calib_poly).astype(np.int32)
        print(f"[calib-poly] Loaded {args.calib_poly} ({len(calib_poly)} verts)")
        if os.path.abspath(args.calib_poly) != os.path.abspath(calib_poly_save):
            np.save(calib_poly_save, calib_poly)
            print(f"[calib-poly] Copied to {calib_poly_save}")
    elif args.calib_poly_pick and args.bg_img and not args.skip_calib:
        # Re-open video and grab frame 0 for the picker
        cap_cp = open_video(args.video)
        frame0_for_pick = read_frame(cap_cp, frame0_idx)
        cap_cp.release()
        if frame0_for_pick is not None:
            instructions = [
                "Draw ONLY around the CLEAN container outline on frame 0.",
                "Include: left-wall edges, top rim, front-bottom edge, LEFT vertices.",
                "EXCLUDE: fluid 'tongue' spilling right, hand shadow, anything outside container.",
                "Tighter polygon = cleaner calibration (but keep the container fully inside).",
            ]
            calib_poly = pick_polygon_on_image(
                frame0_for_pick,
                window_name=f"Calibration polygon on frame {frame0_idx}",
                instructions=instructions,
            )
            if len(calib_poly) >= 3:
                np.save(calib_poly_save, calib_poly)
                print(f"[calib-poly] Saved → {calib_poly_save} "
                      f"({len(calib_poly)} verts)")
            else:
                print("[calib-poly] Cancelled — using full config_00 for calibration")
                calib_poly = None

    # ── Step 5: Camera calibration ──
    # config_00.png is passed only as a sanity-check target (for IoU print +
    # diff_combined/binary overlays), NOT as an optimization target.
    if args.bg_img and not args.skip_calib:
        config00_path = str(out_dir / "config_00.png")

        # For extrinsic-charuco refinement: read frame 0 from video
        frame0_bgr = None
        if args.refine == "extrinsic-charuco":
            cap_f0 = open_video(args.video)
            frame0_bgr = read_frame(cap_f0, frame0_idx)
            cap_f0.release()

        _run_camera_calibration(
            bg_img_path=args.bg_img,
            config00_path=config00_path,
            fluid_w=args.fluid_w,
            fluid_h=args.fluid_h,
            settings_xml=args.settings_xml,
            out_dir=str(out_dir),
            icc_profile=icc_profile,
            left_face_pts=left_face_pts,
            cube_depth_cm=args.cube_depth,
            refine_mode=args.refine,
            video_h=h_f, video_w=w_f,
            frame0_bgr=frame0_bgr,
            calib_target_poly=calib_poly,
            safety_rot_deg=args.safety_rot_deg,
            safety_t_cm=args.safety_t_cm,
        )
    else:
        print(f"       Next: run pipeline.py --calib_img background.png "
              f"--target {out_dir}/config_00.png --out_dir {out_dir}/")

    # ── Optional: validate timing against PS reference ──
    if args.validate:
        if os.path.isdir(args.validate):
            validate_timing(str(out_dir), args.validate, args.n_configs)
        else:
            print(f"[WARN] --validate dir not found: {args.validate}")


# _parse_settings_xml lives in pipeline.py (imported at the top of this file).


def _run_camera_calibration(
    bg_img_path: str,
    config00_path: str,
    fluid_w: Optional[float],
    fluid_h: Optional[float],
    settings_xml: Optional[str],
    out_dir: str,
    icc_profile: Optional[bytes] = None,
    left_face_pts: Optional[np.ndarray] = None,
    cube_depth_cm: float = 4.0,
    refine_mode: str = "extrinsic-charuco",
    video_h: int = 1080, video_w: int = 1920,
    frame0_bgr: Optional[np.ndarray] = None,
    calib_target_poly: Optional[np.ndarray] = None,
    safety_rot_deg: float = 3.0,
    safety_t_cm: float = 3.0,
) -> None:
    """
    Camera calibration:
      A. ChArUco on bg_img → K, R₀, t₀  (always)
      B. Refinement (refine_mode):
           'none'                → use ChArUco directly
           'left-face'           → planar PnP (IPPE) on the 4 LEFT-face corners
           'extrinsic-charuco'   → re-detect ChArUco on frame 0 to refine R,t
                                   (K stays fixed from bg).  Compensates for
                                   camera drift between bg shot and experiment.
    """
    print("\n" + "=" * 60)
    print("  STEP 4: Camera Calibration")
    print("=" * 60)

    # ── Resolve fluid dimensions (kept in cm for clarity) ─────────────────
    fw_cm, fh_cm = fluid_w, fluid_h
    parsed_depth_cm = None
    if settings_xml and os.path.isfile(settings_xml):
        s = _parse_settings_xml(settings_xml)
        if fw_cm is None and s.get("W") is not None:
            fw_cm = s["W"]
        if fh_cm is None and s.get("H") is not None:
            fh_cm = s["H"]
        parsed_depth_cm = s.get("depth")
        print(f"[calib] settings.xml: W={s.get('W')}cm, H={s.get('H')}cm, "
              f"depth={s.get('depth')}cm")

    if fw_cm is None or fh_cm is None:
        print("[calib] ERROR: fluid dimensions unknown. "
              "Provide --fluid_w / --fluid_h or --settings_xml.")
        return

    # If user did not override --cube-depth (still default 4.0) and we parsed
    # a depth from settings.xml, use that.
    if parsed_depth_cm is not None and abs(cube_depth_cm - 4.0) < 1e-6:
        cube_depth_cm = parsed_depth_cm
        print(f"[calib] Using parsed cube depth = {cube_depth_cm:.2f} cm")

    # pipeline.py's calibrate() expects fw/fh in METERS.
    fw_m = fw_cm / 100.0
    fh_m = fh_cm / 100.0
    print(f"[calib] Fluid: W={fw_cm:.2f} cm × H={fh_cm:.2f} cm × "
          f"D={cube_depth_cm:.2f} cm")

    # ── Import calibration functions from pipeline.py ─────────────────────
    _calib_dir = os.path.dirname(os.path.abspath(__file__))
    if _calib_dir not in sys.path:
        sys.path.insert(0, _calib_dir)
    try:
        from pipeline import calibrate, save_xml, \
            render_mask_KRt, render_background_KRt, diff_visual, diff_binary
    except ImportError as e:
        print(f"[calib] ERROR: cannot import pipeline.py: {e}")
        return

    # ── Check inputs ──────────────────────────────────────────────────────
    if not os.path.isfile(bg_img_path):
        print(f"[calib] ERROR: background image not found: {bg_img_path}")
        return
    if not os.path.isfile(config00_path):
        print(f"[calib] ERROR: config_00.png not found: {config00_path}")
        return

    target = cv2.imread(config00_path, cv2.IMREAD_GRAYSCALE)
    if target is None:
        print(f"[calib] ERROR: cannot read config_00.png")
        return
    tgt_H, tgt_W = target.shape

    # ── Optional: apply user-drawn calibration polygon to target ─────────
    #   Pixels OUTSIDE the polygon are forced to white (255 = background)
    #   so the calibration optimiser only sees the clean container outline.
    #   config_00.png on disk is not touched — this is purely in-memory.
    if calib_target_poly is not None and len(calib_target_poly) >= 3:
        poly_mask = np.zeros_like(target)
        cv2.fillPoly(poly_mask, [calib_target_poly.reshape(-1, 1, 2)], 255)
        # Inside polygon: keep original; outside: force white
        target = np.where(poly_mask > 0, target, np.uint8(255))
        # Save what the calibrator sees for debugging
        debug_path = os.path.join(out_dir, "config_00_for_calib.png")
        cv2.imwrite(debug_path, target)
        inside_area = int((poly_mask > 0).sum())
        print(f"[calib] Applied calibration polygon ({len(calib_target_poly)} "
              f"verts, {inside_area} px inside)")
        print(f"[calib] Target with polygon applied → {debug_path}")

    # ── Step 4a: ChArUco calibration ─────────────────────────────────────
    print("\n[calib] Step 4a: ChArUco calibration on background image...")
    try:
        theta0, img_W, img_H = calibrate(bg_img_path, fw_m, fh_m)
    except Exception as e:
        print(f"[calib] ERROR during ChArUco calibration: {e}")
        return

    # Convert theta0 → K, R_init, t_init from ChArUco (used as PnP prior).
    try:
        from pipeline import _theta_to_K, _camera_axes
        K_init = _theta_to_K(theta0, img_W, img_H)
        R_init, t_init = None, None
        try:
            eye, C_x, C_y, C_z, _s = _camera_axes(theta0)   # 5-tuple (pipeline/OpenGL conv)
            # Convert to OpenCV convention for solvePnP:
            #   OpenCV R rows = [right; down; forward]
            #   pipeline   = [C_x; C_y(=up); C_z(=back)]
            #   so OpenCV row1 = -pipeline C_y, OpenCV row2 = -pipeline C_z
            R_init = np.vstack([C_x, -C_y, -C_z])
            t_init = -R_init @ eye
            print(f"[calib] ChArUco prior: eye={eye}, |t|={np.linalg.norm(t_init):.3f}")
        except Exception as e:
            print(f"[calib] WARN: could not derive R/t from theta0: {e}")
    except Exception as e:
        print(f"[calib] ERROR getting initial K: {e}")
        return

    K, R_mat, t = None, None, None
    depth_m = cube_depth_cm / 100.0

    # ── Step 4b: refinement (or none) ────────────────────────────────────
    if refine_mode == "extrinsic-charuco" and frame0_bgr is not None:
        from pipeline import refine_extrinsic_features
        bg_img_bgr = cv2.imread(bg_img_path)
        print(f"\n[calib] Step 4b: feature-match bg_img ↔ frame 0 → refine R, t "
              f"(K fixed from bg)")
        result = refine_extrinsic_features(
            frame0_bgr, bg_img_bgr, K_init, R_init, t_init,
            y_plane=0.0, min_inliers=8,
        )
        if result is not None:
            n_m = result["n_matches"]
            n_i = result["n_inliers"]
            err = result["reproj_err"]
            print(f"[calib] Stage 2: {n_m} SIFT matches, {n_i} PnP inliers, "
                  f"reproj = {err:.3f} px")
            # Compare with Stage 1 pose
            eye_bg  = -R_init.T @ t_init if R_init is not None else None
            eye_exp = -result["R"].T @ result["t"]
            if eye_bg is not None:
                delta = (eye_exp - eye_bg) * 100.0  # metres → cm
                dist  = np.linalg.norm(delta)
                # Rotation angle between the two poses
                R_delta = result["R"] @ R_init.T
                angle_rad = np.arccos(np.clip(
                    (np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0))
                angle_deg = np.degrees(angle_rad)
                print(f"[calib] Camera drift (bg → frame0):")
                print(f"         Δposition = ({delta[0]:+.3f}, {delta[1]:+.3f}, "
                      f"{delta[2]:+.3f}) cm  |Δ|={dist:.3f} cm")
                print(f"         Δrotation = {angle_deg:.3f}°")
            K, R_mat, t = K_init, result["R"], result["t"]
        else:
            print("[calib] Stage 2 failed (not enough feature matches) — "
                  "falling back to ChArUco bg only")

    elif refine_mode == "edge" and R_init is not None:
        from pipeline import refine_extrinsic_edge
        print(f"\n[calib] Step 4b: edge + IoU refinement of R, t "
              f"(K fixed from bg)")
        result = refine_extrinsic_edge(
            target, K_init, R_init, t_init,
            fw_m, fh_m, tgt_W, tgt_H,
            safety_rot_deg=safety_rot_deg,
            safety_t_cm=safety_t_cm,
        )
        if result is not None:
            c0 = result["cost_before"]
            c1 = result["cost_after"]
            print(f"[calib] Edge chamfer: {c0:.3f} → {c1:.3f} px "
                  f"(Δ={c1-c0:+.3f})")
            # Compare poses
            eye_bg  = -R_init.T @ t_init
            eye_new = -result["R"].T @ result["t"]
            delta = (eye_new - eye_bg) * 100.0
            dist  = np.linalg.norm(delta)
            R_delta = result["R"] @ R_init.T
            angle_rad = np.arccos(np.clip(
                (np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)
            print(f"[calib] Adjustment (ChArUco → edge):")
            print(f"         Δposition = ({delta[0]:+.3f}, {delta[1]:+.3f}, "
                  f"{delta[2]:+.3f}) cm  |Δ|={dist:.3f} cm")
            print(f"         Δrotation = {angle_deg:.3f}°")
            if c1 < c0:
                K, R_mat, t = K_init, result["R"], result["t"]
            else:
                print("[calib] Edge refinement did not improve — keeping ChArUco")
        else:
            print("[calib] Edge refinement failed — keeping ChArUco")

    elif refine_mode == "left-face" and left_face_pts is not None:
        print(f"\n[calib] Step 4b: PnP from {len(left_face_pts)} cube corners "
              f"(3 left wall + optional 1 right)")
        sol = solve_camera_from_left_face(
            left_face_pts, fw_m, fh_m, depth_m, K_init,
            R_init=R_init, t_init=t_init,
        )
        if sol is None:
            print("[calib] left-face PnP failed — falling back to ChArUco only")
        else:
            K, R_mat, t = K_init, sol[0], sol[1]

    # ── Default / fallback: use ChArUco theta0 directly ──────────────────
    if K is None:
        print("\n[calib] Step 4b: using ChArUco initial params directly "
              "(no second optimization).")
        if R_init is None:
            print("[calib] ERROR: ChArUco prior unavailable.")
            return
        K, R_mat, t = K_init, R_init, t_init

    # ── Step 4c: Save outputs ─────────────────────────────────────────────
    print("\n[calib] Step 4c: Saving outputs...")

    # camera_params.xml
    xml_path = os.path.join(out_dir, "camera_params.xml")
    save_xml(K, R_mat, t, img_W, img_H, xml_path)

    # Background_mask.png (rendered silhouette vs config_00)
    mask = render_mask_KRt(K, R_mat, t, fw_m, fh_m, tgt_W, tgt_H)
    _imwrite_with_icc(os.path.join(out_dir, "Background_mask.png"), mask, icc_profile)

    # Diff images (BGR color)
    _imwrite_with_icc(os.path.join(out_dir, "diff_combined.png"),
                      diff_visual(mask, target), icc_profile)
    _imwrite_with_icc(os.path.join(out_dir, "diff_binary.png"),
                      diff_binary(mask, target), icc_profile)

    # Background.png (rendered scene with cube overlay)
    bg_img = cv2.imread(bg_img_path)
    if bg_img is not None:
        from pipeline import render_background_KRt
        bg_out = render_background_KRt(bg_img, K, R_mat, t, fw_m, fh_m,
                                       out_w=tgt_W, out_h=tgt_H)
        cv2.imwrite(os.path.join(out_dir, "Background.png"), bg_out)

    # IoU check
    fg_mask = (mask < 128).astype(np.uint8)
    fg_tgt  = (target < 128).astype(np.uint8)
    inter = np.logical_and(fg_mask, fg_tgt).sum()
    union = np.logical_or(fg_mask, fg_tgt).sum()
    iou = inter / union if union > 0 else 0.0

    print(f"\n[calib] ✓ Done.  IoU(rendered vs config_00) = {iou:.4f}")
    print(f"[calib] Output:")
    print(f"         camera_params.xml   → {xml_path}")
    print(f"         Background_mask.png")
    print(f"         Background.png")
    print(f"         diff_combined.png / diff_binary.png")


def validate_timing(auto_dir: str, ps_dir: str, n_configs: int = 9) -> None:
    """
    Cross-match auto configs against all PS configs to detect timing offset.
    Prints an IoU matrix + best-match + recommended offset.
    """
    from PIL import Image as PILImage

    def _iou(p1: str, p2: str) -> float:
        im1 = np.array(PILImage.open(p1).convert("L").resize((960, 540)))
        im2 = np.array(PILImage.open(p2).convert("L").resize((960, 540)))
        fg1 = im1 < 128
        fg2 = im2 < 128
        inter = np.sum(fg1 & fg2)
        union = np.sum(fg1 | fg2)
        return inter / union if union > 0 else 0.0

    print("\n" + "=" * 70)
    print("  TIMING VALIDATION: Auto vs PS reference (IoU cross-match)")
    print("=" * 70)

    header = f"{'Auto':<12}"
    for j in range(n_configs):
        header += f"{'PS_'+str(j):>8}"
    header += f"  {'Best':>10}"
    print(header)
    print("-" * len(header))

    offsets = []
    for i in range(n_configs):
        auto_path = os.path.join(auto_dir, f"config_{i:02d}.png")
        if not os.path.isfile(auto_path):
            continue
        row = f"auto_{i:02d}     "
        best_j, best_iou = -1, 0.0
        for j in range(n_configs):
            ps_path = os.path.join(ps_dir, f"config_{j:02d}.png")
            if not os.path.isfile(ps_path):
                row += f"{'N/A':>8}"
                continue
            v = _iou(auto_path, ps_path)
            if v > best_iou:
                best_j, best_iou = j, v
            row += f"{v:8.3f}"
        shift = best_j - i
        offsets.append(shift)
        marker = "✓" if shift == 0 else f"Δ={shift:+d}"
        row += f"  → PS_{best_j:02d} {marker}"
        print(row)

    # Recommend offset based on early frames (config_01..04)
    early_offsets = offsets[1:5] if len(offsets) >= 5 else offsets[1:]
    if early_offsets:
        median_shift = int(np.median(early_offsets))
        if median_shift != 0:
            raw_offset = median_shift * 10  # config step → raw frames
            print(f"\n⚠  Timing drift detected: median shift = {median_shift:+d} config steps")
            print(f"   Recommended: re-run with --frame0-offset {-raw_offset}")
        else:
            print(f"\n✓  Timing looks well-aligned (median shift = 0)")
    print("=" * 70)


if __name__ == "__main__":
    main()
