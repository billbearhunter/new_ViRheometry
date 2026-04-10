"""
pipeline.py  —  Complete pipeline, no OpenGL required
======================================================
Steps:
  1. ChArUco calibration (calib_img)  -> initial theta0  [Nelder-Mead on reprojection error]
  2. DLT fine-tune against target mask -> optimal K, R, t
       2a. EPnP  : closed-form, minimizes algebraic error  (one SVD)
       2b. LM    : iterative, minimizes pixel reprojection error
                   each step solves (J^T J + lambda I) delta = J^T r
  3. Render Background_mask.png  -- directly with K, R, t (no theta conversion)
  4. Render Background.png       -- directly with K, R, t
  5. Save camera_params.xml      -- theta conversion only here (XML format)

settings.xml is read from the directory of --calib_img (data dir).
Outputs are saved to --out_dir (defaults to the same directory as --calib_img).
fluid_w / fluid_h are read automatically from settings.xml in the data dir.
  <setup W="6.3" H="4.6" .../>

Usage:
    python pipeline.py \\
        --calib_img IMG_7806.JPG \\
        --target    config_00.png \\
        [--bg_img      IMG_7806.JPG]
        [--cube_alpha  0.7]
        [--skip_calib  --theta0 "..."]
"""

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from cv2 import aruco
from scipy.optimize import minimize, least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


# ═══════════════════════════════════════════════════════════════
#  ChArUco Board setup  (identical to Calibration.py)
# ═══════════════════════════════════════════════════════════════

CHARUCOBOARD_ROWCOUNT = 5
CHARUCOBOARD_COLCOUNT = 7


def _get_aruco_dict():
    if hasattr(aruco, "Dictionary_get"):
        return aruco.Dictionary_get(aruco.DICT_5X5_100)
    return aruco.getPredefinedDictionary(aruco.DICT_5X5_100)


def _get_detector_params():
    if hasattr(aruco, "DetectorParameters_create"):
        return aruco.DetectorParameters_create()
    return aruco.DetectorParameters()


ARUCO_DICT = _get_aruco_dict()

if hasattr(aruco, "CharucoBoard_create"):
    CHARUCO_BOARD = aruco.CharucoBoard_create(
        CHARUCOBOARD_COLCOUNT, CHARUCOBOARD_ROWCOUNT, 0.04, 0.03, ARUCO_DICT)
else:
    CHARUCO_BOARD = aruco.CharucoBoard(
        (CHARUCOBOARD_COLCOUNT, CHARUCOBOARD_ROWCOUNT), 0.04, 0.03, ARUCO_DICT)


# ═══════════════════════════════════════════════════════════════
#  Projection — theta-based  (for Stage 1 calibration only)
# ═══════════════════════════════════════════════════════════════

def _camera_axes(theta):
    look_at = np.array([theta[0], 0.0, theta[1]])
    eye = theta[6] * np.array([
        math.sin(theta[2]) * math.cos(theta[3]),
        math.cos(theta[2]),
        -math.sin(theta[2]) * math.sin(theta[3]),
    ])
    C_z = eye - look_at;  C_z /= np.linalg.norm(C_z)
    C_x = np.array([C_z[2], 0.0, -C_z[0]]);  C_x /= np.linalg.norm(C_x)
    C_y = np.cross(C_z, C_x)
    s = 1.0 / (2.0 * math.tan(math.radians(theta[5] / 2.0)))
    return eye, C_x, C_y, C_z, s


def proj_batch(X0, theta, W, H, ids=None):
    pts = X0[:, ids.flatten()] if ids is not None else X0
    eye, C_x, C_y, C_z, s = _camera_axes(theta)
    E = np.outer(eye, np.ones(pts.shape[1]))
    cam = np.vstack([C_x, C_y, C_z]) @ (pts - E)
    cam[0] = -cam[0] / cam[2]
    cam[1] = -cam[1] / cam[2]
    pix = np.outer(np.array([0.5*W, 0.5*H]), np.ones(pts.shape[1])) \
          + s * H * np.array([[1,0],[0,-1]]) @ cam[:2]
    return pix


def proj_points_theta(points3d, theta, W, H):
    """theta-based projection; used only for initial correspondence matching."""
    pts = np.asarray(points3d, dtype=float).T
    eye, C_x, C_y, C_z, s = _camera_axes(theta)
    E = np.outer(eye, np.ones(pts.shape[1]))
    cam = np.vstack([C_x, C_y, C_z]) @ (pts - E)
    result = np.full((pts.shape[1], 2), np.nan)
    vis = cam[2] < 0
    result[vis, 0] = 0.5*W + s*H * (-cam[0,vis] / cam[2,vis])
    result[vis, 1] = 0.5*H + s*H * ( cam[1,vis] / cam[2,vis])
    return result


# ═══════════════════════════════════════════════════════════════
#  Projection — standard pinhole K, R, t
#  Used for all rendering after Stage 2.
#  No theta-model constraints (no forced-horizontal C_x).
# ═══════════════════════════════════════════════════════════════

def _project_KRt(points3d, K, R_mat, t):
    """
    Standard pinhole: X_c = R @ X_w + t,  p = K * X_c / X_c[2].
    Returns (N, 2); points with X_c[2] <= 0 become nan.
    """
    pts = np.asarray(points3d, dtype=float)
    X_c = (R_mat @ pts.T).T + t          # (N, 3)
    result = np.full((len(pts), 2), np.nan)
    vis = X_c[:, 2] > 0
    z = X_c[vis, 2]
    result[vis, 0] = K[0, 0] * X_c[vis, 0] / z + K[0, 2]
    result[vis, 1] = K[1, 1] * X_c[vis, 1] / z + K[1, 2]
    return result


# ═══════════════════════════════════════════════════════════════
#  Stage 1 — ChArUco Calibration
# ═══════════════════════════════════════════════════════════════

def calibrate(calib_img_path, fluid_w, fluid_h):
    """
    Detect ChArUco board and return initial theta via Nelder-Mead
    on reprojection error.
    Returns: theta (7,), img_W, img_H
    """
    img = cv2.imread(calib_img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {calib_img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_H, img_W = gray.shape

    aruco_params = _get_detector_params()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7)

    corners, ids, _ = aruco.detectMarkers(gray, dictionary=ARUCO_DICT,
                                          parameters=aruco_params)
    print(f"[calib] detected {len(corners)} markers")
    if ids is None or len(corners) == 0:
        raise RuntimeError("No ArUco markers detected")

    for c in corners:
        cv2.cornerSubPix(gray, c, (3,3), (-1,-1), criteria)

    ret, ch_corners, ch_ids = aruco.interpolateCornersCharuco(
        corners, ids, gray, CHARUCO_BOARD)
    print(f"[calib] ChArUco corners: {ret}")
    if ret is None or ret <= 10 or ch_corners is None:
        raise RuntimeError("Not enough ChArUco corners")

    X0 = np.array([
        [x + fluid_w, 0, y]
        for y in np.arange(-0.04, 0.09, 0.04)
        for x in np.arange(-0.08, 0.13, 0.04)
    ]).T

    def reprojection_error(th):
        try:
            Proj = proj_batch(X0, th, img_W, img_H, ch_ids).T
            total = 0
            for i, corner in enumerate(ch_corners):
                err = np.linalg.norm(corner - Proj[i])
                l2 = 0.5 * err**2
                if err >= 2:
                    l2 = 2 * (err - 1.0)
                total += l2
            return total / len(ch_corners)
        except Exception:
            return 1e9

    theta0 = np.array([0.10, 0.10, -math.pi/6, 2.0071, 0.0, 50.0, 0.41])
    result = minimize(reprojection_error, theta0, method="Nelder-Mead",
                      options={"disp": True, "maxiter": 1_000_000,
                               "xatol": 1e-6, "fatol": 1e-6})
    theta = result.x
    print(f"[calib] theta      = {theta}")
    print(f"[calib] reprojection error = {result.fun:.4f} px")
    return theta, img_W, img_H


# ═══════════════════════════════════════════════════════════════
#  Cube geometry
# ═══════════════════════════════════════════════════════════════

_CUBE_FACES = [
    [0,1,2,3],  # z=0
    [4,5,6,7],  # z=+
    [0,3,7,4],  # x=0
    [1,2,6,5],  # x=fw
    [0,1,5,4],  # y=0
    [3,2,6,7],  # y=fh
]
_FACE_NORMALS = [
    np.array([0,  0, -1]),
    np.array([0,  0,  1]),
    np.array([-1, 0,  0]),
    np.array([1,  0,  0]),
    np.array([0, -1,  0]),
    np.array([0,  1,  0]),
]


def _cube_verts(fluid_w, fluid_h):
    fw, fh = fluid_w, fluid_h
    return np.array([
        [0,   0,   0   ],
        [fw,  0,   0   ],
        [fw,  fh,  0   ],
        [0,   fh,  0   ],
        [0,   0,   0.04],
        [fw,  0,   0.04],
        [fw,  fh,  0.04],
        [0,   fh,  0.04],
    ], dtype=float)


# ═══════════════════════════════════════════════════════════════
#  Renderer — K, R, t based (no theta constraints)
# ═══════════════════════════════════════════════════════════════

def render_mask_KRt(K, R_mat, t, fluid_w, fluid_h, W, H):
    """
    White background, black cube mask.
    Uses standard pinhole projection with K, R, t directly.
    No theta-model horizontal constraint on C_x.
    """
    verts = _cube_verts(fluid_w, fluid_h)
    pv = _project_KRt(verts, K, R_mat, t)
    mask = np.full((H, W), 255, dtype=np.uint8)
    for face in _CUBE_FACES:
        pts = pv[face]
        if np.any(np.isnan(pts)):
            continue
        cv2.fillPoly(mask, [pts.round().astype(np.int32)], color=0)
    return mask


def render_background_KRt(bg_img_bgr, K, R_mat, t, fluid_w, fluid_h,
                           out_w=1920, out_h=1080, cube_alpha=0.7):
    """
    Background + semi-transparent white cube.
    Back-face culling + painter's algorithm.
    Uses standard pinhole K, R, t.
    """
    bg = cv2.resize(bg_img_bgr, (out_w, out_h)).astype(np.float32)
    verts = _cube_verts(fluid_w, fluid_h)
    eye = -R_mat.T @ t                   # camera center in world

    visible_faces = []
    for face, normal in zip(_CUBE_FACES, _FACE_NORMALS):
        face_center = verts[face].mean(axis=0)
        if np.dot(eye - face_center, normal) > 0:
            X_c = R_mat @ face_center + t
            depth = X_c[2]               # OpenCV Z: larger = farther
            visible_faces.append((depth, face))
    visible_faces.sort(key=lambda x: x[0], reverse=True)  # far → near

    pv = _project_KRt(verts, K, R_mat, t)
    result = bg.copy()
    for _, face in visible_faces:
        pts = pv[face]
        if np.any(np.isnan(pts)):
            continue
        face_mask = np.zeros((out_h, out_w), dtype=np.uint8)
        cv2.fillPoly(face_mask, [pts.round().astype(np.int32)], color=255)
        face_px = face_mask > 0
        result[face_px] = result[face_px] * (1.0 - cube_alpha) + 255.0 * cube_alpha

    return np.clip(result, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
#  Stage 2 — DLT Fine-Tuning
# ═══════════════════════════════════════════════════════════════

def _theta_to_K(theta, W, H):
    """Intrinsic matrix K from theta's fov. cx=W/2, cy=H/2, no skew."""
    s = 1.0 / (2.0 * math.tan(math.radians(theta[5] / 2.0)))
    f = s * H
    return np.array([[f, 0., W/2.], [0., f, H/2.], [0., 0., 1.]], dtype=np.float64)


def _get_contour(mask_gray):
    contours, _ = cv2.findContours(
        255 - mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("No contour found in target mask")
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)


def _build_correspondences(theta0, fluid_w, fluid_h, W, H, mask_gray):
    """
    For each silhouette vertex (convex hull of visible projections),
    snap to the nearest target-contour pixel.
    Returns X3d (N,3), p2d (N,2).
    """
    verts_3d = _cube_verts(fluid_w, fluid_h)
    proj = proj_points_theta(verts_3d, theta0, W, H)
    contour = _get_contour(mask_gray)

    valid_idx = [
        i for i, p in enumerate(proj)
        if not np.any(np.isnan(p)) and 0 <= p[0] <= W and 0 <= p[1] <= H
    ]
    if len(valid_idx) < 4:
        raise RuntimeError(f"Only {len(valid_idx)} cube vertices project into image.")

    hull = cv2.convexHull(
        proj[valid_idx].astype(np.float32), returnPoints=False).flatten()
    silhouette_idx = [valid_idx[h] for h in hull]

    X3d_list, p2d_list = [], []
    for i in silhouette_idx:
        dists = np.linalg.norm(contour - proj[i], axis=1)
        X3d_list.append(verts_3d[i])
        p2d_list.append(contour[int(np.argmin(dists))])

    return np.array(X3d_list), np.array(p2d_list)


def _reprojection_error_KRt(K, R_mat, t, X3d, p2d):
    """Mean pixel reprojection error."""
    errors = []
    for X, q in zip(X3d, p2d):
        X_c = R_mat @ X + t
        if X_c[2] <= 0:
            continue
        u = K[0,0] * X_c[0] / X_c[2] + K[0,2]
        v = K[1,1] * X_c[1] / X_c[2] + K[1,2]
        errors.append(np.linalg.norm(np.array([u, v]) - q))
    return float(np.mean(errors)) if errors else float('inf')


def _solve_epnp_lm(X3d, p2d, K):
    """
    Two-stage solver (fixed K):

    Stage A — EPnP (closed-form):
        Minimizes algebraic error ||A vec(P)||^2 via SVD.
        Fast; provides a good initial estimate.

    Stage B — Levenberg-Marquardt refinement:
        Minimizes pixel reprojection error:
            min  sum_i || p_i - pi(K, R, t, X_i) ||^2
        Each LM iteration solves the linear system:
            (J^T J + lambda I) delta = J^T r
        where J is the 2n x 6 Jacobian of pixel coords w.r.t. pose,
        computed analytically by OpenCV.
        Converges to sub-pixel accuracy in ~10-30 iterations.

    Returns R_mat (3,3), t (3,).
    """
    success, rvec, tvec = cv2.solvePnP(
        X3d.astype(np.float64), p2d.astype(np.float64),
        K, None, flags=cv2.SOLVEPNP_EPNP)
    if not success:
        raise RuntimeError("EPnP failed")

    R0, _ = cv2.Rodrigues(rvec)
    err_epnp = _reprojection_error_KRt(K, R0, tvec.flatten(), X3d, p2d)
    print(f"[dlt] EPnP reprojection error : {err_epnp:.4f} px")

    # LM refinement: minimize pixel reprojection error
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9)
    cv2.solvePnPRefineLM(
        X3d.astype(np.float64), p2d.astype(np.float64),
        K, None, rvec, tvec, criteria=criteria)

    R_mat, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()
    err_lm = _reprojection_error_KRt(K, R_mat, t, X3d, p2d)
    print(f"[dlt] LM   reprojection error : {err_lm:.4f} px")

    return R_mat, t




def _project_silhouette_poly(K, R_mat, t, fluid_w, fluid_h):
    """Projected convex silhouette polygon of the cube in image coordinates."""
    verts = _cube_verts(fluid_w, fluid_h)
    proj = _project_KRt(verts, K, R_mat, t)
    valid = ~np.isnan(proj).any(axis=1)
    pts = proj[valid]
    if len(pts) < 3:
        return None
    hull_idx = cv2.convexHull(pts.astype(np.float32), returnPoints=False).flatten()
    return pts[hull_idx].astype(np.float64)


def _sample_closed_polygon(poly, n_samples=384):
    """
    Resample a closed polygon with a fixed number of samples.
    This keeps the least-squares residual length constant even if the cube
    silhouette changes from 4 to 5 or 6 vertices during optimization.
    """
    if poly is None or len(poly) < 3:
        return np.empty((0, 2), dtype=np.float64)

    poly = np.asarray(poly, dtype=np.float64)
    closed = np.vstack([poly, poly[0]])
    seg = closed[1:] - closed[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    perim = float(np.sum(seg_len))
    if perim < 1e-9:
        return np.repeat(poly[:1], n_samples, axis=0)

    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    d = np.linspace(0.0, perim, n_samples, endpoint=False)
    edge_idx = np.searchsorted(cum[1:], d, side='right')
    edge_idx = np.clip(edge_idx, 0, len(seg_len) - 1)

    edge_start = closed[edge_idx]
    edge_vec = seg[edge_idx]
    edge_len = seg_len[edge_idx]
    local = d - cum[edge_idx]
    alpha = np.divide(local, edge_len, out=np.zeros_like(local), where=edge_len > 1e-12)
    return edge_start + alpha[:, None] * edge_vec


def _left_profile_from_points(points, y_centers, band=2.0):
    out = np.full(len(y_centers), np.nan, dtype=np.float64)
    if len(points) == 0:
        return out
    y = points[:, 1]
    x = points[:, 0]
    for i, yc in enumerate(y_centers):
        m = np.abs(y - yc) <= band
        if np.any(m):
            out[i] = np.min(x[m])
    return out


def _dense_left_priority_refine(K0, R0, t0, target_gray, fluid_w, fluid_h,
                                left_boost=3.5, profile_boost=5.0,
                                n_samples=384, max_nfev=200):
    """
    Dense contour refinement after EPnP+LM.

    The current pipeline only uses a few silhouette vertices for pose fitting,
    which is often good globally but leaves a visible bias on one side.
    Here we refine against a dense projected silhouette contour, with extra weight
    on the left side and on the left profile x(y), so the rendered mask hugs the
    target's left boundary much more tightly.
    """
    H, W = target_gray.shape
    target_contour = _get_contour(target_gray)

    # Sparse target samples for symmetric nearest-neighbour matching
    n_target = min(320, len(target_contour))
    idx = np.linspace(0, len(target_contour) - 1, n_target, dtype=int)
    target_sample = target_contour[idx]
    target_tree = cKDTree(target_contour)

    # Left-profile anchors sampled over the visible vertical extent of the target
    ys = target_contour[:, 1]
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))
    y_centers = np.linspace(y_min, y_max, 56)
    target_left_profile = _left_profile_from_points(target_contour, y_centers, band=2.5)

    x_min = float(np.min(target_contour[:, 0]))
    x_max = float(np.max(target_contour[:, 0]))
    span_x = max(x_max - x_min, 1.0)

    rvec0, _ = cv2.Rodrigues(R0)
    rvec0 = rvec0.flatten()
    f0 = float(K0[0, 0])
    cx0 = float(K0[0, 2])
    cy0 = float(K0[1, 2])

    def unpack(delta):
        rvec = rvec0 + delta[:3]
        t = t0 + delta[3:6]
        f = f0 * np.exp(delta[6])
        cx = cx0 + delta[7]
        cy = cy0 + delta[8]
        K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        R_mat, _ = cv2.Rodrigues(rvec)
        return K, R_mat, t

    def left_weights(points):
        alpha = np.clip((x_max - points[:, 0]) / span_x, 0.0, 1.0)
        return 1.0 + left_boost * (alpha ** 2)

    expected_len = 2 * n_samples + 2 * len(target_sample) + len(y_centers) + 9

    def residuals(delta):
        K, R_mat, t = unpack(delta)
        poly = _project_silhouette_poly(K, R_mat, t, fluid_w, fluid_h)
        samples = _sample_closed_polygon(poly, n_samples=n_samples)
        if len(samples) < 12:
            return np.full(expected_len, 1e3, dtype=np.float64)

        # One-way: rendered -> target contour
        _, nn_idx = target_tree.query(samples)
        tgt_nn = target_contour[nn_idx]
        w1 = left_weights(samples)[:, None]
        res1 = (samples - tgt_nn) * w1

        # Symmetric: target contour -> rendered contour
        render_tree = cKDTree(samples)
        _, nn_idx2 = render_tree.query(target_sample)
        rnd_nn = samples[nn_idx2]
        w2 = left_weights(target_sample)[:, None]
        res2 = 0.6 * (rnd_nn - target_sample) * w2

        # Explicit left profile x(y) anchoring
        render_left_profile = _left_profile_from_points(samples, y_centers, band=2.5)
        valid = ~np.isnan(target_left_profile) & ~np.isnan(render_left_profile)
        res_profile = np.zeros_like(target_left_profile, dtype=np.float64)
        if np.any(valid):
            res_profile[valid] = profile_boost * (render_left_profile[valid] - target_left_profile[valid])

        missing_render = ~np.isnan(target_left_profile) & np.isnan(render_left_profile)
        if np.any(missing_render):
            res_profile[missing_render] = profile_boost * 80.0

        extra_render = np.isnan(target_left_profile) & ~np.isnan(render_left_profile)
        if np.any(extra_render):
            res_profile[extra_render] = profile_boost * 20.0

        # Small regularization to avoid intrinsics drifting too far from the calibrated solution
        reg = np.array([
            0.30 * delta[6],     # log-f
            0.015 * delta[7],    # cx
            0.015 * delta[8],    # cy
            0.8  * delta[0],
            0.8  * delta[1],
            0.8  * delta[2],
            25.0 * delta[3],
            25.0 * delta[4],
            25.0 * delta[5],
        ], dtype=np.float64)

        return np.concatenate([res1.ravel(), res2.ravel(), res_profile.ravel(), reg])

    x0 = np.zeros(9, dtype=np.float64)
    lb = np.array([-0.25, -0.25, -0.25, -0.03, -0.03, -0.03, -0.15, -60.0, -60.0])
    ub = np.array([ 0.25,  0.25,  0.25,  0.03,  0.03,  0.03,  0.15,  60.0,  60.0])

    before_mask = render_mask_KRt(K0, R0, t0, fluid_w, fluid_h, W, H)
    print(f"[dense] IoU before dense refine : {_iou(before_mask, target_gray):.6f}")

    result = least_squares(
        residuals, x0, bounds=(lb, ub), method='trf', loss='soft_l1',
        f_scale=2.0, max_nfev=max_nfev, verbose=0
    )

    K_opt, R_opt, t_opt = unpack(result.x)
    after_mask = render_mask_KRt(K_opt, R_opt, t_opt, fluid_w, fluid_h, W, H)
    print(f"[dense] status={result.status}, nfev={result.nfev}, cost={result.cost:.4f}")
    print(f"[dense] IoU after  dense refine : {_iou(after_mask, target_gray):.6f}")

    return K_opt, R_opt, t_opt


def _Rt_to_theta_for_xml(R_mat, t, theta0):
    """
    Convert R, t -> theta for XML output only.
    Rendering uses K, R, t directly; this conversion is only for camera_params.xml.
    """
    eye = -R_mat.T @ t
    d       = np.linalg.norm(eye)
    theta_E = np.arccos(np.clip(eye[1] / d, -1.0, 1.0))
    sin_tE  = np.sin(theta_E)
    phi_E   = np.arctan2(
        -eye[2] / (d * sin_tE + 1e-12),
         eye[0] / (d * sin_tE + 1e-12))
    fwd = R_mat[2]
    t_scene = -eye[1] / fwd[1] if abs(fwd[1]) > 1e-8 else d
    look_at = eye + t_scene * fwd
    lx, lz  = look_at[0], look_at[2]
    cam_up  = -R_mat[1]
    psi_E   = np.arctan2(cam_up[2], cam_up[1])
    return np.array([lx, lz, theta_E, phi_E, psi_E, theta0[5], d])


def dlt_finetune(theta0, target_gray, fluid_w, fluid_h):
    """
    Stage 2: build silhouette correspondences, run EPnP + LM.
    Returns: K (3x3), R_mat (3x3), t (3,)
    """
    H, W = target_gray.shape
    K = _theta_to_K(theta0, W, H)
    print(f"[dlt] K: fx=fy={K[0,0]:.2f},  cx={K[0,2]:.1f},  cy={K[1,2]:.1f}")

    X3d, p2d = _build_correspondences(theta0, fluid_w, fluid_h, W, H, target_gray)
    print(f"[dlt] silhouette correspondences : {len(X3d)}")
    for i, (x3, p) in enumerate(zip(X3d, p2d)):
        print(f"      [{i}]  3D=({x3[0]:.4f},{x3[1]:.4f},{x3[2]:.4f})"
              f"  ->  2D=({p[0]:.1f},{p[1]:.1f})")

    R_mat, t = _solve_epnp_lm(X3d, p2d, K)

    # Stage 2b: dense silhouette refinement with extra weight on the left boundary
    K, R_mat, t = _dense_left_priority_refine(K, R_mat, t, target_gray, fluid_w, fluid_h)
    return K, R_mat, t


# ═══════════════════════════════════════════════════════════════
#  XML output  — computed directly from K, R_mat, t
#  Same format as Calibration.py; no theta conversion needed.
#
#  eyepos  = -R^T t  (camera center in world, cm)
#  quat    = from rotation matrix directly
#            Our convention: rot_cols = [C_x | C_y | C_z]
#            solvePnP rows:  R_mat    = [C_x ; -C_y ; -C_z]
#            so C_x =  R_mat[0],  C_y = -R_mat[1],  C_z = -R_mat[2]
#            rot_our = column_stack(C_x, C_y, C_z)  →  scipy quaternion
#  fov     = 2 * arctan(H / (2 * K[0,0]))   (from focal length)
# ═══════════════════════════════════════════════════════════════

def save_xml(K, R_mat, t, img_W, img_H, out_path):
    """
    Save camera_params.xml directly from K, R_mat, t.
    No intermediate theta parameterization — zero conversion loss.
    """
    # Camera center in world (meters -> cm)
    eye = -R_mat.T @ t
    eye_cm = eye * 100.0

    # Rotation: our model columns = [C_x, C_y, C_z]
    # solvePnP rows: R_mat = [C_x; -C_y; -C_z]
    C_x =  R_mat[0]
    C_y = -R_mat[1]
    C_z = -R_mat[2]
    rot_our = np.column_stack([C_x, C_y, C_z])
    rot_obj = R.from_matrix(rot_our)
    q_xyzw  = rot_obj.as_quat()                    # (x, y, z, w)
    q_wxyz  = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    # fov from focal length: K[0,0] = s*H,  s = 1/(2*tan(fov/2))
    H_px = img_H
    fov = math.degrees(2.0 * math.atan(H_px / (2.0 * K[0, 0])))

    xml = (
        '<?xml version="1.0"?>\n<setup3D>\n'
        '<camera eyepos="{:.8f} {:.8f} {:.8f}"'
        '  quat="{:.8f} {:.8f} {:.8f} {:.8f}"'
        ' window_size="{:.0f} {:.0f}"'
        ' fov="{:.5f}"/>\n</setup3D>\n'
    ).format(
        eye_cm[0], eye_cm[1], eye_cm[2],
        q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3],
        img_W / 2, img_H / 2,
        fov,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(xml)
    print(f"[save] camera_params.xml -> {out_path}")
    print(xml)


# ═══════════════════════════════════════════════════════════════
#  Diff visualization
# ═══════════════════════════════════════════════════════════════

def diff_visual(rendered, target):
    vis = cv2.cvtColor(rendered, cv2.COLOR_GRAY2BGR)
    vis[(rendered < 128) & ~(target < 128)] = [0, 0, 255]
    vis[(target < 128) & ~(rendered < 128)] = [255, 0, 0]
    return vis


def diff_binary(rendered, target):
    out = np.full_like(rendered, 255)
    out[((rendered < 128) & ~(target < 128)) |
        ((target < 128) & ~(rendered < 128))] = 0
    return out


def _iou(rendered, target):
    pred = rendered < 128;  gt = target < 128
    inter = np.count_nonzero(pred & gt)
    union = np.count_nonzero(pred | gt)
    return inter / union if union > 0 else 0.0


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ChArUco calibration -> EPnP+LM fine-tune -> render outputs")
    parser.add_argument("--calib_img",  required=True)
    parser.add_argument("--target",     required=True)
    parser.add_argument("--bg_img",     default=None)
    parser.add_argument("--cube_alpha", type=float, default=0.7)
    parser.add_argument("--skip_calib", action="store_true")
    parser.add_argument("--theta0",     default=None)
    parser.add_argument("--out_dir",    default=None,
                        help="Output directory. Default: Calibration/results/<material_name>/")
    args = parser.parse_args()

    data_dir = os.path.dirname(os.path.abspath(args.calib_img))
    if args.out_dir:
        out_dir = args.out_dir
    else:
        material_name = os.path.basename(data_dir)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(script_dir, "results", material_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[info] Output directory: {out_dir}")

    settings_path = os.path.join(data_dir, "settings.xml")
    if not os.path.exists(settings_path):
        sys.exit(f"[error] settings.xml not found in: {data_dir}")
    tree = ET.parse(settings_path)
    setup = tree.getroot().find("setup")
    if setup is None:
        sys.exit("[error] <setup> tag not found in settings.xml")
    fw = float(setup.attrib["W"]) / 100.0
    fh = float(setup.attrib["H"]) / 100.0
    print(f"[info] fluid: W={fw*100:.2f} cm, H={fh*100:.2f} cm")

    # ── Step 1 ────────────────────────────────────────────────
    if args.skip_calib:
        if args.theta0 is None:
            sys.exit("[error] --skip_calib requires --theta0")
        theta0 = np.array([float(v) for v in args.theta0.split(",")])
        tgt_tmp = cv2.imread(args.target, cv2.IMREAD_GRAYSCALE)
        img_H, img_W = tgt_tmp.shape
        print(f"[info] Skipping calibration; using provided theta0")
    else:
        print("\n========== Step 1: ChArUco Calibration ==========")
        theta0, img_W, img_H = calibrate(args.calib_img, fw, fh)

    # ── Step 2 ────────────────────────────────────────────────
    print("\n========== Step 2: EPnP + LM Fine-Tuning ==========")
    target = cv2.imread(args.target, cv2.IMREAD_GRAYSCALE)
    if target is None:
        sys.exit(f"[error] Cannot read target: {args.target}")
    tgt_H, tgt_W = target.shape

    K, R_mat, t = dlt_finetune(theta0, target, fw, fh)

    # ── Step 3 ────────────────────────────────────────────────
    print("\n========== Step 3: Render Background_mask.png ==========")
    mask = render_mask_KRt(K, R_mat, t, fw, fh, tgt_W, tgt_H)
    mask_path = os.path.join(out_dir, "Background_mask.png")
    cv2.imwrite(mask_path, mask)
    print(f"[save] Background_mask.png -> {mask_path}")
    print(f"[check] Final IoU : {_iou(mask, target):.6f}")

    cv2.imwrite(os.path.join(out_dir, "diff_combined.png"), diff_visual(mask, target))
    cv2.imwrite(os.path.join(out_dir, "diff_binary.png"),   diff_binary(mask, target))
    print(f"[check] diff_combined / diff_binary saved")

    # ── Step 4 ────────────────────────────────────────────────
    print("\n========== Step 4: Render Background.png ==========")
    bg_path = args.bg_img if args.bg_img else args.calib_img
    bg_img  = cv2.imread(bg_path)
    if bg_img is None:
        sys.exit(f"[error] Cannot read background: {bg_path}")
    bg_out = render_background_KRt(bg_img, K, R_mat, t, fw, fh,
                                   out_w=1920, out_h=1080,
                                   cube_alpha=args.cube_alpha)
    cv2.imwrite(os.path.join(out_dir, "Background.png"), bg_out)
    print(f"[save] Background.png -> {out_dir}/Background.png")

    # ── Step 5 ────────────────────────────────────────────────
    print("\n========== Step 5: Save camera_params.xml ==========")
    save_xml(K, R_mat, t, img_W, img_H, os.path.join(out_dir, "camera_params.xml"))

    print(f"[save] theta_opt.txt not saved (XML computed directly from K,R,t)")

    print("\n========== Done ==========")
    print(f"Output directory: {out_dir}/")
    print("  camera_params.xml    - camera parameters")
    print("  Background_mask.png  - cube mask (rendered with K,R,t directly)")
    print("  Background.png       - scene + semi-transparent cube")
    print("  diff_combined.png    - color diff (red=render only, blue=target only)")
    print("  diff_binary.png      - binary diff (black=diff, white=match)")
    print("  theta_opt.txt        - saved theta for --skip_calib")


if __name__ == "__main__":
    main()