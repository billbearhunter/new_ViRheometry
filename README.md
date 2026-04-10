# ViRheometry

**Virtual Rheometry via MPM Simulation** — Estimates Herschel-Bulkley rheological parameters (η, n, σ_y) from real dam-break flow experiments using CMA-ES optimization with a Mixture-of-Experts surrogate model.

> **Important:** Several scripts use relative imports and **must be run from their own subdirectory**. Every example below includes the required `cd` command — copy and run the whole block.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Data Directory Layout](#data-directory-layout)
5. [Output Layout](#output-layout)
6. [Quick Start](#quick-start)
7. [Module Reference](#module-reference)
   - [FlowCurve](#flowcurve)
   - [Calibration](#calibration)
   - [Simulation](#simulation)
   - [Optimization](#optimization)
8. [Notes](#notes)

---

## System Requirements

| Item | Requirement |
|------|-------------|
| OS | macOS (Apple Silicon / Intel) or Ubuntu/Debian Linux |
| Python | 3.11 (3.10+ acceptable) |
| CMake | 4.x |
| Eigen3 | 5.x (Homebrew default) |
| OpenCV | 4.x |
| GPU | Optional — Taichi supports CUDA (Linux) and MPS (Apple Silicon) |

---

## Installation

Run from the project root (`New_ViRheometry/`):

```bash
chmod +x build.sh
./build.sh
```

| Step | Content |
|------|---------|
| 1 | System packages: cmake, eigen, opencv, python3.11 |
| 2 | libcmaes build from source (non-critical) |
| 3 | Python packages: torch, taichi, gpytorch, cma, scikit-learn, … |
| 4 | C++ builds: `GLRender3d` and `cpp_marching_cubes` |
| 5 | Import verification |

```bash
./build.sh --py-only    # Python packages only
./build.sh --cpp-only   # C++ builds only
```

---

## Project Structure

```
New_ViRheometry/
├── build.sh
├── requirements.txt
│
├── data/                                  # Raw experiment inputs (read-only)
│   ├── ref_Okonomi_4.6_6.3_1/
│   ├── ref_Tonkatsu_5.5_2.3_1/
│   └── ref_Tonkatsu_6.7_3.5_1/
│
├── FlowCurve/                             # ★ must cd here before running
│   ├── flowcurve.py
│   ├── hb_fit.py
│   ├── param.py
│   ├── Rheo_Data/
│   └── figs/                             # ← output: PDF plots
│
├── Calibration/
│   ├── pipeline.py
│   ├── extract_flow_distance.py
│   └── results/                          # ← output: camera_params, backgrounds
│       └── ref_Tonkatsu_6.7_3.5_1/
│
├── Simulation/                            # ★ must cd here before running
│   ├── main.py
│   ├── results/                          # ← output: simulation snapshots, meshes
│   │   └── run_YYYYMMDD_HHMMSS/
│   ├── config/
│   │   ├── config.py
│   │   └── setting.xml
│   ├── simulation/
│   │   ├── taichi.py
│   │   ├── xmlParser.py
│   │   └── file_ops.py
│   ├── scripts/
│   │   ├── MPM_Emulator.py
│   │   ├── Creat_dat.py
│   │   ├── Creat_dataframe.py
│   │   └── Creat_obj.py
│   ├── GLRender3d/build/GLRender3d        # compiled binary
│   ├── ParticleSkinner3DTaichi/cpp_marching_cubes/build/cpp_marching_cubes
│   ├── model/
│   │   ├── best_model.joblib
│   │   └── target_scaler.joblib
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── tuning.py
│   ├── evaluation.py
│   └── test.py
│
└── Optimization/                          # ★ must cd here before running
    ├── optimize_1setup.py
    ├── optimize_2setups.py
    ├── propose_initial_setup.py
    ├── soft_interpolate.py
    ├── visualize_comparison.py
    ├── test_boundary_comparison.py
    ├── result_setup1_*/                   # ← output: optimization results
    ├── moe_workspace5/
    └── libs/
```

---

## Data Directory Layout

The `data/` directory contains **raw inputs only** and is never written to by any script.

```
data/ref_Tonkatsu_6.7_3.5_1/
├── settings.xml          # geometry config: W=3.5cm, H=6.7cm
├── IMG_7796.JPG          # ChArUco calibration photo
├── config_00.png         # calibration target (used by pipeline.py --target)
├── config_01.png         # reference flow snapshots
├── config_02.png
├── ...
└── config_08.png
```

`config_00.png` is the initial state (before flow starts) used for camera calibration.
`config_01.png` ~ `config_08.png` are the reference snapshots for flow distance measurement.

Available datasets:

| Directory | Material | H (cm) | W (cm) | Calib image |
|-----------|----------|--------|--------|-------------|
| `ref_Okonomi_4.6_6.3_1` | Okonomiyaki sauce | 4.6 | 6.3 | `IMG_7806.JPG` |
| `ref_Tonkatsu_5.5_2.3_1` | Tonkatsu sauce | 5.5 | 2.3 | `IMG_7799.JPG` |
| `ref_Tonkatsu_6.7_3.5_1` | Tonkatsu sauce | 6.7 | 3.5 | `IMG_7796.JPG` |

Naming convention: `ref_{Material}_{H}_{W}_{number}`

---

## Output Layout

Each processing step writes to its own directory outside `data/`:

```
Calibration/results/ref_Tonkatsu_6.7_3.5_1/
├── camera_params.xml       # calibrated camera parameters
├── Background.png          # rendered background
├── Background_mask.png     # binary foreground mask
├── diff_combined.png       # calibration quality check (color diff)
├── diff_binary.png         # calibration quality check (binary diff)
├── flow_distances.csv      # extracted flow front positions [cm]
└── flow_distances.json

Optimization/result_setup1_topk_2_YYYYMMDD_HHMMSS/
├── setup1_result.txt       # optimal η, n, σ_y
└── ...

Simulation/results/run_YYYYMMDD_HHMMSS/
├── simulation_results.csv
├── {n:.2f}_{eta:.2f}_{sigma_y:.2f}/
│   ├── config_00.png ~ config_07.png
│   └── *.obj
└── snapdiff_00.png ~ snapdiff_07.png
```

---

## Quick Start

### One-command pipeline

```bash
python3 run_pipeline.py --data data/ref_Tonkatsu_6.7_3.5_1
```

This runs Steps 1–3 automatically and keeps `data/` untouched. Add `--simulate` to also run the MPM verification (Step 4).

```bash
# Full pipeline including simulation
python3 run_pipeline.py \
    --data data/ref_Tonkatsu_6.7_3.5_1 \
    --strategy topk --topk 2 \
    --simulate

# Skip already-completed steps (useful for debugging)
python3 run_pipeline.py \
    --data data/ref_Tonkatsu_6.7_3.5_1 \
    --skip-calibration \
    --skip-extraction

# Skip optimization and run simulation with known parameters
python3 run_pipeline.py \
    --data data/ref_Tonkatsu_6.7_3.5_1 \
    --skip-calibration --skip-extraction --skip-optimization \
    --simulate --eta 26.39 --n 0.429 --sigma_y 13.05
```

`run_pipeline.py` arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | required | Material data directory |
| `--moe_dir` | `Optimization/moe_workspace5` | MoE model directory |
| `--strategy` | `topk` | Expert strategy: `topk`, `threshold`, `adaptive`, `all` |
| `--topk` | `2` | Experts for `topk` |
| `--threshold` | `0.01` | Weight threshold for `threshold` |
| `--confidence_threshold` | `0.7` | Confidence for `adaptive` |
| `--maxiter` | `700` | CMA-ES max iterations |
| `--simulate` | — | Also run MPM simulation (Step 4) |
| `--eta / --n / --sigma_y` | — | Override parameters for Step 4 |
| `--skip-calibration` | — | Skip Step 1 |
| `--skip-extraction` | — | Skip Step 2 |
| `--skip-optimization` | — | Skip Step 3 |

### Manual step-by-step

Each step can also be run individually for debugging:

```
Real experiment (photos + settings.xml in data/)
      │
      ▼
① Calibrate camera             (Calibration/pipeline.py)
      │  → Calibration/results/<name>/camera_params.xml
      │  → Calibration/results/<name>/Background.png
      ▼
② Extract flow distances       (Calibration/extract_flow_distance.py)
      │  → Calibration/results/<name>/flow_distances.csv
      ▼
③ Optimize HB parameters       (Optimization/optimize_1setup.py)
      │  → Optimization/result_setup1_*/
      ▼
④ Verify with simulation       (Simulation/main.py)      (optional)
      │  → Simulation/results/run_*/
      ▼
⑤ Compare with rheometer data  (FlowCurve/hb_fit.py + flowcurve.py)
         → FlowCurve/figs/
```

---

## Module Reference

---

### FlowCurve

> `flowcurve.py` uses `from param import Param` and **must be run from `FlowCurve/`**.
> `hb_fit.py` has no relative imports and can also be run from the project root.

#### `hb_fit.py` — Herschel-Bulkley curve fitting

Fits σ = K · γ̇ⁿ + σ_Y to Anton Paar rheometer CSV data. Prints K, n, σ_Y with ±error and R².

```bash
python3 hb_fit.py --file <CSV> [--range START END]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--file` | required | Rheometer CSV path (UTF-16, header at row 5) |
| `--range START END` | `5 19` | Row index range for fitting |

**Examples:**
```bash
cd FlowCurve
python3 hb_fit.py --file Rheo_Data/tonkatsu_20230113_2000_23C.csv
```

```bash
cd FlowCurve
python3 hb_fit.py --file Rheo_Data/Lotion_20230114_1204_23C.csv --range 3 18
```

```bash
cd FlowCurve
python3 hb_fit.py --file Rheo_Data/Chuno_20230114_1458_23C.csv
```

---

#### `flowcurve.py` — Flow curve visualization

Plots experimental data with HB model overlays.

```bash
python3 flowcurve.py \
    --file <CSV> \
    --est η n σ_Y [--est η n σ_Y ...] \
    --out <PDF> \
    [--extent_y Y_MIN Y_MAX]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--file` | required | Rheometer CSV file |
| `--est η n σ_Y` | required | HB parameters — repeat for multiple curves |
| `--out` | required | Output PDF path |
| `--extent_y Y_MIN Y_MAX` | `1e0 1e2` | Y-axis range |

**Examples:**
```bash
cd FlowCurve
python3 flowcurve.py \
    --file Rheo_Data/tonkatsu_20230113_2000_23C.csv \
    --est 208.35 0.306 95.26 \
    --est 46.314 0.636 128.527 \
    --out figs/Tonkatsu.pdf
```

```bash
cd FlowCurve
python3 flowcurve.py \
    --file Rheo_Data/Lotion_20230114_1204_23C.csv \
    --est 2.711 0.6537 8.404 \
    --est 2.929 0.621 8.393 \
    --out figs/lotion.pdf \
    --extent_y 1 200
```

```bash
cd FlowCurve
python3 flowcurve.py \
    --file Rheo_Data/Chuno_20230114_1458_23C.csv \
    --est 59.220 0.851 83.073 \
    --est 95.104 0.704 97.542 \
    --out figs/chuno.pdf
```

---

### Calibration

> `pipeline.py` and `extract_flow_distance.py` have no relative imports.
> All paths below are relative to the project root `New_ViRheometry/`.

#### `pipeline.py` — Camera calibration

ChArUco board calibration → DLT fine-tuning. Reads `settings.xml` from the directory of `--calib_img` (the data folder). Outputs `camera_params.xml`, `Background.png`, `Background_mask.png` to `--out_dir`.

```bash
python3 Calibration/pipeline.py \
    --calib_img <IMG> \
    --target    <config_00.png> \
    [--out_dir   <OUTPUT_DIR>] \
    [--bg_img <IMG>] \
    [--cube_alpha 0.7] \
    [--skip_calib --theta0 "..."]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--calib_img` | required | Photo of ChArUco board (inside data dir) |
| `--target` | required | Reference simulation snapshot (`config_00.png`) |
| `--out_dir` | `Calibration/results/<material_name>/` | Output directory |
| `--bg_img` | same as `--calib_img` | Background photo |
| `--cube_alpha` | `0.7` | Wireframe cube transparency |
| `--skip_calib` | — | Skip ChArUco step; requires `--theta0` |
| `--theta0` | — | Initial camera parameter string |

> `settings.xml` is always read from the directory of `--calib_img`, regardless of `--out_dir`.

**Examples:**
```bash
# Tonkatsu 6.7x3.5 — output auto-goes to Calibration/results/ref_Tonkatsu_6.7_3.5_1/
python3 Calibration/pipeline.py \
    --calib_img data/ref_Tonkatsu_6.7_3.5_1/IMG_7796.JPG \
    --target    data/ref_Tonkatsu_6.7_3.5_1/config_00.png
```

```bash
# Okonomiyaki
python3 Calibration/pipeline.py \
    --calib_img data/ref_Okonomi_4.6_6.3_1/IMG_7806.JPG \
    --target    data/ref_Okonomi_4.6_6.3_1/config_00.png
```

```bash
# Tonkatsu 5.5x2.3
python3 Calibration/pipeline.py \
    --calib_img data/ref_Tonkatsu_5.5_2.3_1/IMG_7799.JPG \
    --target    data/ref_Tonkatsu_5.5_2.3_1/config_00.png
```

```bash
# Custom output directory
python3 Calibration/pipeline.py \
    --calib_img data/ref_Tonkatsu_6.7_3.5_1/IMG_7796.JPG \
    --target    data/ref_Tonkatsu_6.7_3.5_1/config_00.png \
    --out_dir   /path/to/custom/output
```

```bash
# Skip ChArUco using saved camera params
python3 Calibration/pipeline.py \
    --calib_img data/ref_Tonkatsu_6.7_3.5_1/IMG_7796.JPG \
    --target    data/ref_Tonkatsu_6.7_3.5_1/config_00.png \
    --skip_calib --theta0 "..."
```

---

#### `extract_flow_distance.py` — Extract flow distances

Processes `config_01.png ~ config_08.png` to measure flow front positions. `config_00.png` is automatically excluded (it is the static initial frame). Use `--camera-xml` to point at the calibration output, and `--output-csv/json` to save results outside `data/`.

```bash
python3 Calibration/extract_flow_distance.py \
    --dir       <DATA_DIR> \
    --camera-xml <camera_params.xml> \
    --output-csv  <OUTPUT_CSV> \
    --output-json <OUTPUT_JSON> \
    [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dir` | required | Directory containing `config_*.png` and `settings.xml` |
| `--settings` | `<dir>/settings.xml` | Override `settings.xml` path |
| `--camera-xml` | `<dir>/camera_params.xml` | Path to `camera_params.xml` (file or directory) |
| `--images` | `config_*.png` | Override image glob pattern |
| `--threshold` | `128` | Binarization threshold |
| `--foreground` | `black` | Fluid color: `black` or `white` |
| `--kernel-size` | `3` | Morphological kernel size |
| `--opening-only` | — | Apply opening only (skip closing) |
| `--no-keep-largest` | — | Disable largest-component filtering |
| `--flow-direction DX DZ` | `1.0 0.0` | Flow direction vector |
| `--box-front` | auto | Override container front wall position |
| `--monotonic` | — | Enforce monotonically increasing distances |
| `--unit` | `cm` | Output unit: `m`, `cm`, or `mm` |
| `--output-csv` | `<dir>/flow_distances.csv` | Output CSV path |
| `--output-json` | `<dir>/flow_distances.json` | Output JSON path |
| `--debug-dir` | — | Save per-step debug images |
| `--print-dis1` | — | Print `-dis1` string for optimizer |

**Examples:**
```bash
# Tonkatsu 6.7x3.5 — clean output
python3 Calibration/extract_flow_distance.py \
    --dir        data/ref_Tonkatsu_6.7_3.5_1 \
    --camera-xml Calibration/results/ref_Tonkatsu_6.7_3.5_1/camera_params.xml \
    --output-csv  Calibration/results/ref_Tonkatsu_6.7_3.5_1/flow_distances.csv \
    --output-json Calibration/results/ref_Tonkatsu_6.7_3.5_1/flow_distances.json \
    --monotonic --unit cm --print-dis1
# → -dis1 1.4616 3.5214 5.6031 7.2681 8.3256 8.9129 9.2239 9.3288
```

```bash
# Okonomiyaki
python3 Calibration/extract_flow_distance.py \
    --dir        data/ref_Okonomi_4.6_6.3_1 \
    --camera-xml Calibration/results/ref_Okonomi_4.6_6.3_1/camera_params.xml \
    --output-csv  Calibration/results/ref_Okonomi_4.6_6.3_1/flow_distances.csv \
    --output-json Calibration/results/ref_Okonomi_4.6_6.3_1/flow_distances.json \
    --foreground black --monotonic --print-dis1
```

```bash
# With debug images
python3 Calibration/extract_flow_distance.py \
    --dir        data/ref_Tonkatsu_6.7_3.5_1 \
    --camera-xml Calibration/results/ref_Tonkatsu_6.7_3.5_1/camera_params.xml \
    --output-csv  Calibration/results/ref_Tonkatsu_6.7_3.5_1/flow_distances.csv \
    --output-json Calibration/results/ref_Tonkatsu_6.7_3.5_1/flow_distances.json \
    --debug-dir  Calibration/results/ref_Tonkatsu_6.7_3.5_1/debug \
    --monotonic --print-dis1
```

---

### Simulation

> `main.py` uses `from config.config`, `from simulation.taichi`, and `from scripts` — **must be run from `Simulation/`**.

#### `main.py` — Run MPM simulation and compare with reference

Runs one MPM simulation, renders PNG snapshots, and generates pixel-level diff images.

```bash
python3 main.py \
    --eta <η> --n <n> --sigma_y <σ_y> \
    --ref <REF_DIR> \
    [--diff_amplify 5.0]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--eta` | required | Viscosity η [Pa·sⁿ] |
| `--n` | required | Power-law index n [-] |
| `--sigma_y` | required | Yield stress σ_y [Pa] |
| `--ref` | required | Path to reference data directory |
| `--camera_xml` | `--ref/camera_params.xml` | Path to `camera_params.xml` file or directory (e.g. `Calibration/results/<name>/`) |
| `--diff_amplify` | `5.0` | Brightness amplification for diff images |

**Output** saved to `Simulation/results/run_YYYYMMDD_HHMMSS/`:
```
results/run_20260409_153000/
├── simulation_results.csv
├── {n:.2f}_{eta:.2f}_{sigma_y:.2f}/
│   ├── config_00.png ~ config_07.png
│   └── *.obj
└── snapdiff_00.png ~ snapdiff_07.png
```

**Examples:**
```bash
cd Simulation
python3 main.py \
    --eta 26.389 --n 0.4289 --sigma_y 13.046 \
    --ref ../data/ref_Tonkatsu_6.7_3.5_1 \
    --camera_xml ../Calibration/results/ref_Tonkatsu_6.7_3.5_1/camera_params.xml
```

```bash
cd Simulation
python3 main.py \
    --eta 3.808 --n 0.984 --sigma_y 1.007 \
    --ref ../data/ref_Tonkatsu_5.5_2.3_1 \
    --camera_xml ../Calibration/results/ref_Tonkatsu_5.5_2.3_1/camera_params.xml
```

```bash
# --camera_xml also accepts a directory
cd Simulation
python3 main.py \
    --eta 26.389 --n 0.4289 --sigma_y 13.046 \
    --ref ../data/ref_Tonkatsu_6.7_3.5_1 \
    --camera_xml ../Calibration/results/ref_Tonkatsu_6.7_3.5_1 \
    --diff_amplify 10.0
```

---

#### `test.py` — Model inference test

```bash
python3 test.py <MODEL_PATH> [--test-data <CSV>] [--single-sample] [--batch-size N] [--visualize]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `model_path` | required | Path to `.joblib` model file |
| `--test-data` | — | CSV with test samples |
| `--single-sample` | — | Run one random sample |
| `--batch-size` | `10` | Number of random samples |
| `--visualize` | — | Generate prediction scatter plots |

**Examples:**
```bash
cd Simulation
python3 test.py model/best_model.joblib --single-sample
```

```bash
cd Simulation
python3 test.py model/best_model.joblib --batch-size 20 --visualize
```

---

### Optimization

> `optimize_1setup.py` and `optimize_2setups.py` use `from libs.xxx` — **must be run from `Optimization/`**.

#### `propose_initial_setup.py` — Generate initial setup proposal

```bash
python3 propose_initial_setup.py [--min-mm N] [--max-mm N] [--out FILE]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--min-mm` | `20` | Minimum container dimension [mm] |
| `--max-mm` | `70` | Maximum container dimension [mm] |
| `--out` | `setup1.txt` | Output file path |

```bash
cd Optimization
python3 propose_initial_setup.py --min-mm 20 --max-mm 60 --out my_setup1.txt
```

---

#### `optimize_1setup.py` — CMA-ES optimization (1 setup)

Estimates (η, n, σ_y) from a **single** dam-break experiment.

```bash
python3 optimize_1setup.py \
    -W1 <W> -H1 <H> \
    -dis1 d1 d2 d3 d4 d5 d6 d7 d8 \
    --moe_dir <DIR> \
    [--strategy topk|threshold|adaptive|all] \
    [--topk N] [--threshold F] [--confidence_threshold F] [--max_experts N] \
    [--sigma0 F] [--popsize N] [--maxiter N] [--seed N] \
    [--compare_strategies]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `-W1` | required | Container width [cm] |
| `-H1` | required | Container height [cm] |
| `-dis1` | required | 8 flow distances [cm] |
| `--moe_dir` | required | MoE model directory |
| `--strategy` | `threshold` | `topk`, `threshold`, `adaptive`, or `all` |
| `--topk` | `2` | Number of experts for `topk` |
| `--threshold` | `0.01` | Weight threshold for `threshold` |
| `--confidence_threshold` | `0.7` | Confidence for `adaptive` |
| `--max_experts` | `5` | Max experts for `threshold` |
| `--sigma0` | `0.5` | CMA-ES initial step size |
| `--popsize` | `16` | CMA-ES population size |
| `--maxiter` | `700` | Max iterations |
| `--seed` | `42` | Random seed |
| `--compare_strategies` | — | Run all strategies and compare |

**Output** saved to `Optimization/result_setup1_<strategy>_<threshold>_<timestamp>/`.

**Examples:**
```bash
# Tonkatsu 6.7x3.5 — Top-2 experts
cd Optimization
python3 optimize_1setup.py \
    --moe_dir moe_workspace5 \
    -W1 3.5 -H1 6.7 \
    -dis1 1.4616 3.5214 5.6031 7.2681 8.3256 8.9129 9.2239 9.3288 \
    --strategy topk --topk 2
```

```bash
# Tonkatsu 6.7x3.5 — Adaptive gating
cd Optimization
python3 optimize_1setup.py \
    --moe_dir moe_workspace5 \
    -W1 3.5 -H1 6.7 \
    -dis1 1.4616 3.5214 5.6031 7.2681 8.3256 8.9129 9.2239 9.3288 \
    --strategy adaptive --confidence_threshold 0.7
```

```bash
# Tonkatsu 6.7x3.5 — Threshold strategy
cd Optimization
python3 optimize_1setup.py \
    --moe_dir moe_workspace5 \
    -W1 3.5 -H1 6.7 \
    -dis1 1.4616 3.5214 5.6031 7.2681 8.3256 8.9129 9.2239 9.3288 \
    --strategy threshold --threshold 0.01 --max_experts 5
```

```bash
# Compare all strategies at once
cd Optimization
python3 optimize_1setup.py \
    --moe_dir moe_workspace5 \
    -W1 3.5 -H1 6.7 \
    -dis1 1.4616 3.5214 5.6031 7.2681 8.3256 8.9129 9.2239 9.3288 \
    --compare_strategies
```

---

#### `optimize_2setups.py` — CMA-ES optimization (2 setups)

Estimates (η, n, σ_y) from **two** dam-break experiments simultaneously.

```bash
python3 optimize_2setups.py \
    -W1 <W> -H1 <H> -dis1 d1..d8 \
    -W2 <W> -H2 <H> -dis2 d1..d8 \
    --moe_dir <DIR> \
    [--setup1_dir <DIR>] \
    [same CMA-ES options as optimize_1setup.py]
```

Additional arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `-W2`, `-H2` | required | Setup 2 dimensions [cm] |
| `-dis2` | required | Setup 2 flow distances (8 values) [cm] |
| `-W3`, `-H3` | `0.0` | Setup 3 dimensions (optional) |
| `-dis3` | `[]` | Setup 3 flow distances (optional) |
| `--setup1_dir` | auto | Reuse an existing Setup 1 result |

**Examples:**
```bash
# Two camera angles from same experiment — Top-2
cd Optimization
python3 optimize_2setups.py \
    --moe_dir moe_workspace5 \
    -W1 3.5 -H1 6.7 \
    -dis1 1.4616 3.5214 5.6031 7.2681 8.3256 8.9129 9.2239 9.3288 \
    -W2 3.5 -H2 6.7 \
    -dis2 1.1970 3.7921 6.4270 8.5508 9.6890 10.1595 10.4018 10.5528 \
    --strategy topk --topk 2
```

```bash
# Adaptive gating
cd Optimization
python3 optimize_2setups.py \
    --moe_dir moe_workspace5 \
    -W1 3.5 -H1 6.7 \
    -dis1 1.4616 3.5214 5.6031 7.2681 8.3256 8.9129 9.2239 9.3288 \
    -W2 3.5 -H2 6.7 \
    -dis2 1.1970 3.7921 6.4270 8.5508 9.6890 10.1595 10.4018 10.5528 \
    --strategy adaptive --confidence_threshold 0.9
```

```bash
# Reuse a completed Setup 1 result
cd Optimization
python3 optimize_2setups.py \
    --setup1_dir result_setup1_adaptive_0.01_20260203_231654 \
    --moe_dir moe_workspace5 \
    -W1 3.5 -H1 6.7 \
    -dis1 1.4616 3.5214 5.6031 7.2681 8.3256 8.9129 9.2239 9.3288 \
    -W2 3.5 -H2 6.7 \
    -dis2 1.1970 3.7921 6.4270 8.5508 9.6890 10.1595 10.4018 10.5528 \
    --strategy adaptive --confidence_threshold 0.9
```

---

#### `test_boundary_comparison.py` — Boundary region test

No arguments. Uses `moe_workspace5` directly.

```bash
cd Optimization
python3 test_boundary_comparison.py
# Results saved to: Optimization/boundary_comparison_results/
```

---

#### `soft_interpolate.py` / `visualize_comparison.py` — Library modules

Not standalone scripts. Imported automatically by `optimize_1setup.py` and `optimize_2setups.py`.

---

## Notes

### Which directory to run from

| Script | Run from |
|--------|----------|
| `FlowCurve/flowcurve.py` | `FlowCurve/` (has `from param import`) |
| `FlowCurve/hb_fit.py` | anywhere |
| `Calibration/pipeline.py` | anywhere |
| `Calibration/extract_flow_distance.py` | anywhere |
| `Simulation/main.py` | `Simulation/` (has `from config.config`, `from simulation.taichi`) |
| `Simulation/test.py` | `Simulation/` |
| `Optimization/optimize_*.py` | `Optimization/` (has `from libs.xxx`) |
| `Optimization/propose_initial_setup.py` | `Optimization/` |

### HB parameter units

| Parameter | Symbol | Unit |
|-----------|--------|------|
| Viscosity | η | Pa·sⁿ |
| Power-law index | n | dimensionless (0.3 – 1.0) |
| Yield stress | σ_y | Pa |
| Container width/height | W, H | cm |
| Flow distance | dis | cm |

### MoE model (`moe_workspace5`)

101 Gaussian Process experts with GMM gating. Predicts 8 flow distances from (W, H, η, n, σ_y) in milliseconds, enabling fast CMA-ES evaluation without running full MPM simulations.

### `lcmaes` (non-critical)

Used only by `libs/cmaes.py` (legacy). `optimize_1setup.py` and `optimize_2setups.py` use `import cma` (pycma) and work without lcmaes.

### Taichi GPU / CPU

Default is GPU. To switch to CPU, edit `Simulation/main.py`:

```python
# GPU (default)
ti.init(arch=ti.gpu, offline_cache=True, default_fp=ti.f32, default_ip=ti.i32)

# CPU fallback
ti.init(arch=ti.cpu, offline_cache=True, default_fp=ti.f32, default_ip=ti.i32)
```
