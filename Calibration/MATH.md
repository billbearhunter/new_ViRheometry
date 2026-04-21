# Camera calibration — math & references

This document lists the mathematical tools used in the two-stage camera
calibration pipeline (`pipeline.py`) and where each comes from.

---

## Stage overview

```
       bg_img  ──▶ Stage 1 (ChArUco) ──▶ θ₀ ──▶ K, R₀, t₀
                                                    │
     config_00 ──▶ Stage 2 (edge refinement) ──▶ K, R*, t*   (K fixed)
```

- **Stage 1** fits a 7-parameter camera model `θ` to ChArUco board corners
  detected in the background image, using Nelder–Mead on reprojection error.
- **Stage 2** fixes the intrinsic matrix `K` obtained from `θ₀` and refines
  only the extrinsic `(R, t)` so that the rendered container silhouette
  matches the dam-break frame `config_00`. This is a two-phase optimization:
  Phase 1 minimises a chamfer distance on edge pixels (L-BFGS-B, smooth),
  Phase 2 maximises a (weighted) pixel-IoU (Nelder–Mead, non-smooth).

---

## Stage 1 — ChArUco calibration

### ChArUco markers

ChArUco = ArUco dictionary markers embedded in a chessboard. The chessboard
gives sub-pixel corner precision; the ArUco IDs give unambiguous
correspondence even under occlusion or partial visibility.

- **ArUco markers.**
  S. Garrido-Jurado, R. Muñoz-Salinas, F. J. Madrid-Cuevas, M. J.
  Marín-Jiménez. *Automatic generation and detection of highly reliable
  fiducial markers under occlusion.* Pattern Recognition 47(6):2280–2292,
  2014. https://doi.org/10.1016/j.patcog.2014.01.005

- **ArUco dictionary generation.**
  S. Garrido-Jurado, R. Muñoz-Salinas, F. J. Madrid-Cuevas, R.
  Medina-Carnicer. *Generation of fiducial marker dictionaries using mixed
  integer linear programming.* Pattern Recognition 51:481–491, 2016.
  https://doi.org/10.1016/j.patcog.2015.09.023

OpenCV implementation used: `cv2.aruco.detectMarkers` +
`cv2.aruco.interpolateCornersCharuco` (pipeline.py, `calibrate()`).

### Sub-pixel corner refinement

After the integer-pixel ChArUco corner is found, we refine with
`cv2.cornerSubPix` (gradient-based; moves each corner so that the
dot-product of the local gradient and the (corner → pixel) vector is zero).

- Rooted in Förstner's pointwise optimal point detector:
  W. Förstner, E. Gülch. *A Fast Operator for Detection and Precise
  Location of Distinct Points, Corners and Centres of Circular Features.*
  In Proc. ISPRS Intercommission Conf., 1987.

### Pinhole camera model

Standard perspective projection used throughout Stage 2 rendering
(`pipeline.py::_project_KRt`):

$$
\mathbf{X}_c = R\,\mathbf{X}_w + \mathbf{t}, \qquad
\mathbf{p} = \frac{1}{Z_c}\,K\,\mathbf{X}_c, \qquad
K = \begin{pmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix}
$$

Reference textbook:

- R. Hartley, A. Zisserman. *Multiple View Geometry in Computer Vision*,
  2nd ed. Cambridge University Press, 2003. (Pinhole model, ch. 6.)

### The 7-parameter θ model (project-specific)

Used only in Stage 1; a compact parameterization of a camera looking at a
fixed target:

$$
\theta = (c_x, c_z, \phi, \psi, \text{(unused)}, \text{fov}_\text{deg}, r)
$$

- `(c_x, 0, c_z)`  = look-at point in world frame
- `(φ, ψ)`         = spherical coordinates of the eye direction
- `r`              = eye–target distance (metres)
- `fov`            = vertical field of view (degrees)

The eye, camera axes, and focal scale are reconstructed in
`pipeline.py::_camera_axes`:

$$
\mathbf{e} = r \begin{pmatrix} \sin\phi\cos\psi \\ \cos\phi \\ -\sin\phi\sin\psi \end{pmatrix},
\quad
s = \frac{1}{2\tan(\text{fov}/2)}
$$

`(C_x, C_y, C_z)` are an orthonormal camera basis with `C_x` forced
horizontal (project convention for a level camera rig).

### Reprojection-error optimization

The Stage-1 cost is sum-squared reprojection error in pixels:

$$
\mathcal{L}_1(\theta) = \sum_{i} \|\mathbf{u}_i - \pi(\mathbf{X}_i;\theta)\|^2
$$

minimised with **Nelder–Mead** (derivative-free downhill simplex). Useful
when `π` is cheap and gradients are awkward (custom θ parameterization).

- J. A. Nelder, R. Mead. *A Simplex Method for Function Minimization.* The
  Computer Journal 7(4):308–313, 1965.
  https://doi.org/10.1093/comjnl/7.4.308
- F. Gao, L. Han. *Implementing the Nelder–Mead simplex algorithm with
  adaptive parameters.* Comput. Optim. Appl. 51:259–277, 2012.
  (SciPy's `Nelder-Mead` uses the adaptive variant when `adaptive=True`.)

---

## Stage 2 — Edge refinement (`refine_extrinsic_edge`)

### Canny edges on the target

The target silhouette edges are extracted with Canny before building the
distance field:

- J. Canny. *A Computational Approach to Edge Detection.* IEEE PAMI
  8(6):679–698, 1986. https://doi.org/10.1109/TPAMI.1986.4767851

### Euclidean distance transform (EDT)

Let $E \subset \Omega$ be the target-edge pixel set. The EDT assigns to
every pixel $p \in \Omega$ its Euclidean distance to the nearest edge:

$$
d(p) = \min_{q \in E} \|p - q\|_2
$$

Computed once per `config_00` via `scipy.ndimage.distance_transform_edt`,
which implements the exact linear-time algorithm of:

- C. R. Maurer Jr., R. Qi, V. Raghavan. *A Linear Time Algorithm for
  Computing Exact Euclidean Distance Transforms of Binary Images in
  Arbitrary Dimensions.* IEEE PAMI 25(2):265–270, 2003.
  https://doi.org/10.1109/TPAMI.2003.1177156

Classical introduction to chamfer / distance-transform matching:

- G. Borgefors. *Distance transformations in digital images.* Computer
  Vision, Graphics, and Image Processing 34(3):344–371, 1986.
  https://doi.org/10.1016/S0734-189X(86)80047-0

### Phase 1 — weighted chamfer cost

Let $P_f(R,t)$ be the projected edge pixels of the $f$-th visible cube face.
The per-face-normalised chamfer cost is

$$
\mathcal{L}_{\text{chamfer}}(R,t) = \frac{1}{|\mathcal{F}|} \sum_{f \in \mathcal{F}}
\frac{\sum_{p \in P_f(R,t)} w(p)\,d(p)}
     {\sum_{p \in P_f(R,t)} w(p)}
$$

with $d(p)$ bilinearly interpolated from the EDT and $w(p)$ a
multiplicative weight

$$
w(p) = w_{\text{left}}(p_x)\; w_{\text{top}}(p_y)\; w_{\text{bottom}}(p_y)
$$

supporting left-/top-/bottom-edge emphasis (see `README.md` for CLI flags).
Per-face normalisation prevents long edges (e.g. the bottom wall) from
dominating short ones (e.g. the top cap).

Minimised with **L-BFGS-B** on 6 DoF (Rodrigues rotation vector + translation)
inside a box ±2.5° / ±2 cm of the ChArUco prior:

- R. H. Byrd, P. Lu, J. Nocedal, C. Zhu. *A Limited Memory Algorithm for
  Bound Constrained Optimization.* SIAM J. Sci. Comput. 16(5):1190–1208,
  1995. https://doi.org/10.1137/0916069
- C. Zhu, R. H. Byrd, P. Lu, J. Nocedal. *Algorithm 778: L-BFGS-B, Fortran
  subroutines for large-scale bound-constrained optimization.* ACM TOMS
  23(4):550–560, 1997. https://doi.org/10.1145/279232.279236

### Phase 2 — weighted IoU maximisation

Let $M(R,t)$ be the rendered cube silhouette and $T$ the target foreground.
The (weighted) Jaccard index / IoU is

$$
\text{IoU}_w(R,t) = \frac{\sum_{p} w(p)\,[M(p) \wedge T(p)]}
                        {\sum_{p} w(p)\,[M(p) \vee T(p)]}
$$

starting from the Phase-1 solution with a 0.15° / 1.5 mm simplex and
maximised with **adaptive Nelder–Mead** (non-differentiable objective).

- P. Jaccard. *The distribution of the flora in the alpine zone.* New
  Phytologist 11(2):37–50, 1912.
  https://doi.org/10.1111/j.1469-8137.1912.tb05611.x

### Rodrigues rotation parameterization

Rotations are represented as a 3-vector $\mathbf{r}$ with
$\|\mathbf{r}\| = \theta$ (angle) and $\mathbf{r}/\|\mathbf{r}\|$ (axis),
converted to matrices via Rodrigues' formula:

$$
R = I + \sin\theta\,[\hat{\mathbf{r}}]_\times
     + (1-\cos\theta)\,[\hat{\mathbf{r}}]_\times^2
$$

(Implementation: `scipy.spatial.transform.Rotation.from_rotvec`.)

- O. Rodrigues. *Des lois géométriques qui régissent les déplacements
  d'un système solide dans l'espace.* J. Math. Pures Appl. 5:380–440,
  1840.

### Safety clamp

After refinement we check the deviation from the ChArUco prior:

$$
\Delta_\text{rot} = \|\mathbf{r} - \mathbf{r}_0\|,\quad
\Delta_t = \|\mathbf{t} - \mathbf{t}_0\|
$$

If $\Delta_\text{rot}$ or $\Delta_t$ exceeds safety thresholds
(`--safety_rot_deg`, `--safety_t_cm`), we keep the ChArUco prior. This
guards against large jumps driven by strong local minima in either phase.

---

## Supporting techniques

### Bilinear sampling of the distance field

Sub-pixel edge cost lookups use bilinear interpolation to keep the chamfer
objective smooth enough for L-BFGS-B:

$$
d(x,y) \approx (1-\alpha)(1-\beta)d_{00}
             + \alpha(1-\beta)d_{10}
             + (1-\alpha)\beta d_{01}
             + \alpha\beta d_{11}
$$

with $\alpha = x - \lfloor x \rfloor$, $\beta = y - \lfloor y \rfloor$.
(Implementation: `pipeline.py::_bilinear_dt`.)

### SIFT + BFMatcher (optional path, `--refine extrinsic-charuco`)

For the feature-based refinement mode (not the default), we match SIFT
descriptors between the background and frame 0 and run `cv2.solvePnP` for
a rigid-motion prior.

- D. G. Lowe. *Distinctive Image Features from Scale-Invariant Keypoints.*
  IJCV 60(2):91–110, 2004. https://doi.org/10.1023/B:VISI.0000029664.99615.94
- V. Lepetit, F. Moreno-Noguer, P. Fua. *EPnP: An Accurate O(n) Solution
  to the PnP Problem.* IJCV 81:155–166, 2009.
  https://doi.org/10.1007/s11263-008-0152-6

### Binarization pipeline (`prepare_configs.py::binarize_frame`)

The per-frame mask uses standard HSV thresholding + morphological
cleanup. The optional closing/opening and dilation steps follow the
textbook definitions in:

- R. C. Gonzalez, R. E. Woods. *Digital Image Processing*, 4th ed.
  Pearson, 2018. (Chapter 9 — morphological image processing.)

---

## TL;DR — what to cite

When writing up the pipeline, the minimum set of citations is:

1. **Garrido-Jurado et al. 2014** — ArUco markers
2. **Nelder & Mead 1965** — Stage-1 optimization
3. **Canny 1986** — target edges
4. **Maurer et al. 2003** (or **Borgefors 1986**) — distance transform
5. **Byrd et al. 1995 / Zhu et al. 1997** — L-BFGS-B
6. **Jaccard 1912** — IoU metric
7. **Hartley & Zisserman 2003** — pinhole model (textbook)
