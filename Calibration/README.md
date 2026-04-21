# Calibration pipeline

Two-step workflow: extract configs once, then refine calibration as you clean up
`config_00.png` in Photoshop.

```
┌───────────────────┐       ┌─────────────────────┐
│ Step A            │       │ Step B              │
│ prepare_configs   │─────▶ │ recalibrate_only    │◀──┐
│ (video → configs) │       │ (config_00 → calib) │   │ loop: PS-clean
└───────────────────┘       └─────────────────────┘   │       config_00,
                                      │               │       re-run B
                                      └───────────────┘
```

---

## Inputs (per experiment)

Place these in `data/ref_<sauce>_<W>_<H>_<n>/`:

| File              | Purpose                                            |
| ----------------- | -------------------------------------------------- |
| `IMG_XXXX.JPG`    | Background still with ChArUco board (camera calib) |
| `IMG_XXXX.mov`    | Dam-break video (240fps)                           |
| `settings.xml`    | Container dims + simulation setup                  |

---

## Step A — extract configs from video

```bash
python3 Calibration/prepare_configs.py --data_dir data/ref_<X>_<W>_<H>_<n> --save-debug
```

What it does:
1. Detect frame0 (dam-break start) — GUI picker opens; confirm or adjust
2. Extract 9 frames at 24fps (`config_00`…`config_08`)
3. Auto-fit ROI polygon to the max-flow frame (`config_08`), GUI to edit
4. Binarize each frame (V < v_thresh, clipped to ROI, morph cleanup, optional dilate)
5. ChArUco calibration on `IMG_XXXX.JPG` → initial `camera_params.xml`
6. Edge refinement against `config_00` (skipped if drift > safety limits)

Key flags:

| Flag               | Default | When to change                                              |
| ------------------ | ------: | ----------------------------------------------------------- |
| `--v-thresh`       |     128 | Raise (160–200) if fluid edges are missing                  |
| `--dilate PX`      |       0 | Add 3–8 px to recover glass-reflection rim when V-thresh maxes out |
| `--save-debug`     |     off | Saves `debug_XX.png` overlays — always recommended          |
| `--frame0 N`       |    auto | Skip GUI if you already know the dam-break frame            |
| `--roi-poly PATH`  |    auto | Re-use ROI from a previous run (skips GUI)                  |
| `--safety-rot-deg` |     3.0 | Bump to 5.0 when ChArUco board is far from the container    |

Outputs land in `<data_dir>/output/`:

```
config_00.png … config_08.png    # binarized, aligned to 24fps grid
debug_00.png  … debug_08.png     # raw frame + ROI + mask overlay (if --save-debug)
roi_polygon.npy                  # serialized ROI for reuse
Background_mask.png              # rendered cube silhouette
Background.png                   # bg_img projected through the camera
diff_combined.png / diff_binary.png   # mask vs render diff
camera_params.xml                # K, R, t  (setup3D XML)
settings.xml                     # copied from data_dir
```

---

## Step B — refine calibration after cleaning `config_00`

Open `config_00.png` in Photoshop, paint over shadows / spills on the right side,
save. Then:

```bash
python3 Calibration/recalibrate_only.py --data_dir data/ref_<X>_<W>_<H>_<n>
```

All the auto-detection that Step A does also works here — `--configs_dir`,
`--bg_img`, `--fluid_w/h` default from `<data_dir>/` + `settings.xml`.

### Corner / edge weighting

When you only care about aligning the **container** (left / top / bottom edges)
and don't mind the **right side** (fluid tongue, shadow, splash) being off,
boost those pixels in the chamfer + IoU objective:

```bash
python3 Calibration/recalibrate_only.py \
    --data_dir data/ref_<X>_<W>_<H>_<n> \
    --left_weight 20 --top_weight 20 --bottom_weight 20
```

- `--left_weight`  (with `--left_frac`)   : weight of pixels on the left half
- `--top_weight`   (with `--top_frac`)    : weight of pixels on the top half
- `--bottom_weight`(with `--bottom_frac`) : weight of pixels on the bottom half

They combine **multiplicatively** on a 2D grid, so the top-left corner gets
`left_weight × top_weight` and the bottom-left gets `left_weight × bottom_weight`.
Typical values: 1 (off) … 20 (strong). Fractions default to 0.5 (half bbox).

### Safety clamp

The edge refinement rejects huge jumps (default ±3° / ±3 cm from the ChArUco
prior). Relax when the ChArUco board is distant or the scene is cluttered:

```bash
    --safety_rot_deg 5.0 --safety_t_cm 5.0
```

If the clamp fires, `recalibrate_only.py` reports that the ChArUco prior was
kept; otherwise it reports the IoU before/after, drift, and writes the new
`camera_params.xml`.

---

## Typical iteration flow

```bash
# 1. Extract configs (GUI picks frame0 + ROI)
python3 Calibration/prepare_configs.py \
    --data_dir data/ref_Sweet_4.5_4.5_1 \
    --v-thresh 160 --dilate 3 --save-debug

# 2. Open output/config_00.png in Photoshop, clean shadows, save.

# 3. Refine calibration with corner weighting
python3 Calibration/recalibrate_only.py \
    --data_dir data/ref_Sweet_4.5_4.5_1 \
    --safety_rot_deg 5.0 --safety_t_cm 5.0 \
    --left_weight 20 --top_weight 20 --bottom_weight 20

# 4. Open output/diff_combined.png to verify alignment.
#    Red = mask-only, Blue = render-only, Black = overlap.
#    If misaligned, clean config_00 further and re-run step 3.
```

---

## Files in this directory

| File                        | Role                                                    |
| --------------------------- | ------------------------------------------------------- |
| `prepare_configs.py`        | **Step A** — video + bg → configs + initial calib       |
| `recalibrate_only.py`       | **Step B** — PS-cleaned config_00 → refined calib       |
| `pipeline.py`               | Shared functions (ChArUco, edge refinement, rendering)  |
| `extract_flow_distance.py`  | Post-processing (flow distance from configs)            |
| `test_pipeline.py`          | Unit tests                                              |
