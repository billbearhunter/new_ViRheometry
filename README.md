# ViRheometry

**Virtual Rheometry via MPM Simulation** — Estimates Herschel-Bulkley rheological parameters (η, n, σ_y) from dam-break flow experiments using CMA-ES optimization with a Mixture-of-Experts surrogate model.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Data Directory Layout](#data-directory-layout)
5. [Output Layout](#output-layout)
6. [Quick Start](#quick-start)
7. [Module Reference](#module-reference)
   - [Calibration](#calibration)
   - [Optimization](#optimization)
   - [Simulation](#simulation)
   - [FlowCurve](#flowcurve)
8. [Notes](#notes)

---

## System Requirements

| Item | Requirement |
|------|-------------|
| OS | macOS (Apple Silicon / Intel) or Ubuntu/Debian Linux |
| Python | 3.11 (3.10+ acceptable) |
| CMake | 4.x |
| Eigen3 | 5.x |
| OpenCV | 4.x |
| GPU | Optional — Taichi supports CUDA (Linux) and Metal (Apple Silicon) |

---

## Installation

Run from the project root:

```bash
chmod +x build.sh
./build.sh
```

| Step | Content |
|------|---------|
| 1 | System packages: cmake, eigen, opencv, python3.11 |
| 2 | libcmaes build from source |
| 3 | Python packages from requirements.txt |
| 4 | C++ builds: GLRender3d and cpp_marching_cubes |
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
├── run_pipeline.py                    # End-to-end pipeline entry point
│
├── data/                              # Raw experiment inputs (never modified by scripts)
│   └── ref_<Material>_<H>_<W>_<N>/
│
├── Calibration/
│   ├── pipeline.py                    # Camera calibration
│   ├── extract_flow_distance.py       # Flow distance extraction
│   └── results/                       # Calibration outputs (auto-created)
│       └── ref_<Material>_<H>_<W>_<N>/
│
├── Optimization/
│   ├── optimize_1setup.py             # CMA-ES optimization, single setup
│   ├── optimize_2setups.py            # CMA-ES optimization, two setups
│   ├── moe_workspace5/                # Pre-trained MoE surrogate model
│   │   ├── expert_0.pt ~ expert_99.pt
│   │   ├── gmm_gate.joblib
│   │   └── boxes.json
│   ├── libs/                          # Shared optimization library
│   └── result_setup1_*/              # Optimization outputs (auto-created)
│
├── Simulation/
│   ├── main.py                        # MPM simulation and rendering
│   ├── config/
│   │   ├── config.py
│   │   └── setting.xml
│   ├── simulation/
│   ├── scripts/
│   ├── GLRender3d/build/GLRender3d    # Compiled renderer
│   └── ParticleSkinner3DTaichi/cpp_marching_cubes/build/cpp_marching_cubes
│
└── FlowCurve/                         # Rheometer data analysis
    ├── flowcurve.py
    ├── hb_fit.py
    ├── param.py
    └── Rheo_Data/                     # Rheometer CSV files
```

---

## Data Directory Layout

The `data/` directory contains raw inputs only and is never written to by any script.

```
data/ref_<Material>_<H>_<W>_<N>/
├── settings.xml          # Container geometry: W [cm], H [cm]
├── <calibration>.JPG     # Photo of ChArUco board in the experimental setup
├── config_00.png         # Calibration target: rendered reference at t=0
├── config_01.png         # Reference flow snapshot at time step 1
├── ...
└── config_08.png         # Reference flow snapshot at time step 8
```

`config_00.png` is used as the calibration target for camera pose estimation.
`config_01.png` through `config_08.png` are the reference snapshots for flow distance measurement.

Directory naming convention:

```
ref_{Material}_{H:.1f}_{W:.1f}_{N}
```

Example: `ref_Tonkatsu_6.7_3.5_1` — Tonkatsu sauce, H=6.7 cm, W=3.5 cm, trial 1.

---

## Output Layout

Each step writes to its own directory outside `data/`:

```
Calibration/results/ref_<Material>_<H>_<W>_<N>/
├── camera_params.xml       # Calibrated camera parameters
├── Background.png          # Rendered background image
├── Background_mask.png     # Binary foreground mask
├── diff_combined.png       # Color diff for calibration quality check
├── diff_binary.png         # Binary diff for calibration quality check
├── flow_distances.csv      # Extracted flow front positions [cm]
└── flow_distances.json

Optimization/result_setup1_<strategy>_<timestamp>/
└── setup1_result.txt       # Optimal η, n, σ_y

Simulation/results/run_<timestamp>/
├── simulation_results.csv
├── <n>_<eta>_<sigma_y>/
│   ├── config_00.png ~ config_08.png
│   └── *.obj
└── snapdiff_00.png ~ snapdiff_08.png
```

---

## Quick Start

### One-command pipeline

Run Steps 1–3 with a single command from the project root:

```bash
python3 run_pipeline.py --data data/ref_Tonkatsu_6.7_3.5_1
```

Add `--simulate` to also run the MPM verification (Step 4):

```bash
python3 run_pipeline.py \
    --data data/ref_Tonkatsu_6.7_3.5_1 \
    --strategy topk --topk 2 \
    --simulate
```

Use `--skip-*` flags to re-run individual steps without repeating earlier ones:

```bash
# Re-run only optimization (calibration and extraction already done)
python3 run_pipeline.py \
    --data data/ref_Tonkatsu_6.7_3.5_1 \
    --skip-calibration --skip-extraction

# Run only simulation with known parameters
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
| `--topk` | `2` | Number of experts for `topk` |
| `--threshold` | `0.01` | Weight threshold for `threshold` |
| `--confidence_threshold` | `0.7` | Confidence for `adaptive` |
| `--max_experts` | `5` | Max experts for `threshold` |
| `--maxiter` | `700` | CMA-ES max iterations |
| `--simulate` | — | Run MPM simulation after optimization |
| `--eta / --n / --sigma_y` | — | Override HB parameters for simulation |
| `--skip-calibration` | — | Skip Step 1, use existing results |
| `--skip-extraction` | — | Skip Step 2, use existing results |
| `--skip-optimization` | — | Skip Step 3 |

### Pipeline overview

```
data/<material>/
      |
      v
Step 1   Camera calibration       Calibration/pipeline.py
      |  -> Calibration/results/<material>/camera_params.xml
      |
      v
Step 2   Flow distance extraction  Calibration/extract_flow_distance.py
      |  -> Calibration/results/<material>/flow_distances.csv
      |
      v
Step 3   HB parameter optimization  Optimization/optimize_1setup.py
      |  -> Optimization/result_setup1_*/
      |
      v
Step 4   MPM simulation (optional)  Simulation/main.py
         -> Simulation/results/run_*/
```

---

## Module Reference

---

### Calibration

Scripts in `Calibration/` have no relative imports and can be run from any directory.

#### `pipeline.py` — Camera calibration

ChArUco board detection → EPnP + Levenberg-Marquardt fine-tuning. Reads `settings.xml` from the data directory. Outputs camera parameters and rendered backgrounds to `--out_dir`.

```bash
python3 Calibration/pipeline.py \
    --calib_img <data_dir>/<photo>.JPG \
    --target    <data_dir>/config_00.png \
    [--out_dir  <output_dir>] \
    [--bg_img   <photo>.JPG] \
    [--cube_alpha 0.7] \
    [--skip_calib --theta0 "<values>"]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--calib_img` | required | ChArUco calibration photo |
| `--target` | required | `config_00.png` from the data directory |
| `--out_dir` | `Calibration/results/<material_name>/` | Output directory |
| `--bg_img` | same as `--calib_img` | Background photo if different from calibration photo |
| `--cube_alpha` | `0.7` | Wireframe cube transparency |
| `--skip_calib` | — | Skip ChArUco detection; requires `--theta0` |
| `--theta0` | — | Comma-separated initial camera parameter string |

`settings.xml` is always read from the directory of `--calib_img`, regardless of `--out_dir`.

**Example:**

```bash
python3 Calibration/pipeline.py \
    --calib_img data/ref_Tonkatsu_6.7_3.5_1/<photo>.JPG \
    --target    data/ref_Tonkatsu_6.7_3.5_1/config_00.png
```

---

#### `extract_flow_distance.py` — Flow distance extraction

Processes `config_01.png` through `config_08.png` to measure flow front positions in physical units. `config_00.png` is automatically excluded. Results are written to `--output-csv` and `--output-json`.

```bash
python3 Calibration/extract_flow_distance.py \
    --dir         <data_dir> \
    --camera-xml  <camera_params.xml> \
    --output-csv  <output.csv> \
    --output-json <output.json> \
    [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dir` | required | Directory containing `config_*.png` and `settings.xml` |
| `--settings` | `<dir>/settings.xml` | Override `settings.xml` path |
| `--camera-xml` | `<dir>/camera_params.xml` | Path to `camera_params.xml` (file or directory) |
| `--images` | auto (excludes `config_00.png`) | Override image glob pattern |
| `--threshold` | `128` | Binarization threshold |
| `--foreground` | `black` | Fluid color: `black` or `white` |
| `--kernel-size` | `3` | Morphological kernel size |
| `--opening-only` | — | Apply opening only, skip closing |
| `--no-keep-largest` | — | Disable largest-component filtering |
| `--flow-direction DX DZ` | `1.0 0.0` | Flow direction vector in XZ plane |
| `--box-front` | from `settings.xml` | Override container front wall position |
| `--monotonic` | — | Enforce monotonically increasing distances |
| `--unit` | `cm` | Output unit: `m`, `cm`, or `mm` |
| `--output-csv` | `<dir>/flow_distances.csv` | CSV output path |
| `--output-json` | `<dir>/flow_distances.json` | JSON output path |
| `--debug-dir` | — | Save per-frame debug images to this directory |
| `--print-dis1` | — | Print `-dis1` string ready for optimizer input |

**Example:**

```bash
python3 Calibration/extract_flow_distance.py \
    --dir         data/ref_Tonkatsu_6.7_3.5_1 \
    --camera-xml  Calibration/results/ref_Tonkatsu_6.7_3.5_1/camera_params.xml \
    --output-csv  Calibration/results/ref_Tonkatsu_6.7_3.5_1/flow_distances.csv \
    --output-json Calibration/results/ref_Tonkatsu_6.7_3.5_1/flow_distances.json \
    --monotonic --unit cm --print-dis1
```

---

### Optimization

Scripts in `Optimization/` use `from libs.xxx` and must be run from `Optimization/`.

#### `optimize_1setup.py` — CMA-ES optimization, single setup

Estimates (η, n, σ_y) from a single dam-break experiment using CMA-ES with the MoE surrogate model.

```bash
cd Optimization
python3 optimize_1setup.py \
    -W1 <W> -H1 <H> \
    -dis1 d1 d2 d3 d4 d5 d6 d7 d8 \
    --moe_dir <moe_workspace5> \
    [--strategy topk|threshold|adaptive|all] \
    [--topk N] [--threshold F] [--confidence_threshold F] [--max_experts N] \
    [--sigma0 F] [--popsize N] [--maxiter N] [--seed N]
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

Output saved to `result_setup1_<strategy>_<timestamp>/`.

**Examples:**

```bash
cd Optimization

# Top-2 experts
python3 optimize_1setup.py \
    --moe_dir moe_workspace5 \
    -W1 3.5 -H1 6.7 \
    -dis1 1.197 3.792 6.427 8.551 9.689 10.160 10.402 10.553 \
    --strategy topk --topk 2

# Threshold strategy
python3 optimize_1setup.py \
    --moe_dir moe_workspace5 \
    -W1 3.5 -H1 6.7 \
    -dis1 1.197 3.792 6.427 8.551 9.689 10.160 10.402 10.553 \
    --strategy threshold --threshold 0.01 --max_experts 5

# Adaptive strategy
python3 optimize_1setup.py \
    --moe_dir moe_workspace5 \
    -W1 3.5 -H1 6.7 \
    -dis1 1.197 3.792 6.427 8.551 9.689 10.160 10.402 10.553 \
    --strategy adaptive --confidence_threshold 0.7
```

---

#### `optimize_2setups.py` — CMA-ES optimization, two setups

Estimates (η, n, σ_y) from two dam-break experiments simultaneously, providing stronger constraints than a single setup.

```bash
cd Optimization
python3 optimize_2setups.py \
    -W1 <W> -H1 <H> -dis1 d1..d8 \
    -W2 <W> -H2 <H> -dis2 d1..d8 \
    --moe_dir <moe_workspace5> \
    [--setup1_dir <existing_result_dir>] \
    [same CMA-ES options as optimize_1setup.py]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `-W1`, `-H1` | required | Setup 1 container dimensions [cm] |
| `-dis1` | required | Setup 1 flow distances, 8 values [cm] |
| `-W2`, `-H2` | required | Setup 2 container dimensions [cm] |
| `-dis2` | required | Setup 2 flow distances, 8 values [cm] |
| `-W3`, `-H3` | `0.0` | Setup 3 dimensions (optional third setup) |
| `-dis3` | `[]` | Setup 3 flow distances (optional) |
| `--setup1_dir` | — | Reuse an existing Setup 1 result directory |

**Examples:**

```bash
cd Optimization

# Two camera angles from the same experiment
python3 optimize_2setups.py \
    --moe_dir moe_workspace5 \
    -W1 3.5 -H1 6.7 \
    -dis1 1.197 3.792 6.427 8.551 9.689 10.160 10.402 10.553 \
    -W2 3.5 -H2 6.7 \
    -dis2 1.197 3.792 6.427 8.551 9.689 10.160 10.402 10.553 \
    --strategy topk --topk 2

# Reuse an existing Setup 1 result
cd Optimization
python3 optimize_2setups.py \
    --setup1_dir result_setup1_topk_2_<timestamp> \
    --moe_dir moe_workspace5 \
    -W1 3.5 -H1 6.7 \
    -dis1 1.197 3.792 6.427 8.551 9.689 10.160 10.402 10.553 \
    -W2 3.5 -H2 6.7 \
    -dis2 1.197 3.792 6.427 8.551 9.689 10.160 10.402 10.553 \
    --strategy adaptive --confidence_threshold 0.9
```

---

### Simulation

`Simulation/main.py` uses `from config.config`, `from simulation.taichi`, and `from scripts` — it must be run from `Simulation/`.

#### `main.py` — MPM simulation and diff rendering

Runs one full MPM simulation with the given HB parameters, renders PNG snapshots, and generates pixel-level diff images against the reference.

```bash
cd Simulation
python3 main.py \
    --eta <η> --n <n> --sigma_y <σ_y> \
    --ref <data_dir> \
    [--camera_xml <camera_params.xml or directory>] \
    [--out_dir <output_dir>] \
    [--diff_amplify 5.0]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--eta` | required | Viscosity η [Pa·s^n] |
| `--n` | required | Power-law index n [-] |
| `--sigma_y` | required | Yield stress σ_y [Pa] |
| `--ref` | required | Reference data directory |
| `--camera_xml` | `--ref/camera_params.xml` | Path to `camera_params.xml` file or directory |
| `--out_dir` | `results/run_<timestamp>/` | Output directory |
| `--diff_amplify` | `5.0` | Brightness multiplier for diff images |

Output saved to `Simulation/results/run_<timestamp>/`.

**Example:**

```bash
cd Simulation
python3 main.py \
    --eta 26.389 --n 0.429 --sigma_y 13.046 \
    --ref ../data/ref_Tonkatsu_6.7_3.5_1 \
    --camera_xml ../Calibration/results/ref_Tonkatsu_6.7_3.5_1/camera_params.xml
```

---

### FlowCurve

`FlowCurve/flowcurve.py` uses `from param import Param` and must be run from `FlowCurve/`. `hb_fit.py` has no relative imports and can be run from any directory.

#### `hb_fit.py` — Herschel-Bulkley curve fitting

Fits σ = K · γ̇^n + σ_Y to Anton Paar rheometer CSV data. Prints K, n, σ_Y with errors and R².

```bash
python3 FlowCurve/hb_fit.py --file <CSV> [--range START END]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--file` | required | Rheometer CSV path (UTF-16, header at row 5) |
| `--range START END` | `5 19` | Row index range for fitting |

**Example:**

```bash
python3 FlowCurve/hb_fit.py --file FlowCurve/Rheo_Data/tonkatsu_20230113_2000_23C.csv
```

---

#### `flowcurve.py` — Flow curve visualization

Plots experimental rheometer data with one or more HB model overlays.

```bash
cd FlowCurve
python3 flowcurve.py \
    --file <CSV> \
    --est η n σ_Y [--est η n σ_Y ...] \
    --out <output.pdf> \
    [--extent_y Y_MIN Y_MAX]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--file` | required | Rheometer CSV file |
| `--est η n σ_Y` | required | HB parameters — repeat for multiple curves |
| `--out` | required | Output PDF path |
| `--extent_y Y_MIN Y_MAX` | `1e0 1e2` | Y-axis range |

**Example:**

```bash
cd FlowCurve
python3 flowcurve.py \
    --file Rheo_Data/tonkatsu_20230113_2000_23C.csv \
    --est 208.35 0.306 95.26 \
    --est 26.389 0.429 13.046 \
    --out figs/Tonkatsu.pdf
```

---

## Notes

### Which directory to run from

| Script | Run from |
|--------|----------|
| `run_pipeline.py` | project root |
| `Calibration/pipeline.py` | anywhere |
| `Calibration/extract_flow_distance.py` | anywhere |
| `FlowCurve/hb_fit.py` | anywhere |
| `FlowCurve/flowcurve.py` | `FlowCurve/` |
| `Simulation/main.py` | `Simulation/` |
| `Optimization/optimize_1setup.py` | `Optimization/` |
| `Optimization/optimize_2setups.py` | `Optimization/` |

### HB parameter units

| Parameter | Symbol | Unit |
|-----------|--------|------|
| Viscosity consistency | η | Pa·s^n |
| Power-law index | n | dimensionless (0.3–1.0) |
| Yield stress | σ_y | Pa |
| Container width / height | W, H | cm |
| Flow distance | dis | cm |

### MoE surrogate model

`Optimization/moe_workspace5/` contains 100 Gaussian Process expert networks (`expert_0.pt` to `expert_99.pt`) with a GMM gating network (`gmm_gate.joblib`). Given (W, H, η, n, σ_y), the model predicts 8 flow distances in milliseconds, enabling fast CMA-ES evaluation without running full MPM simulations.

These model files are not tracked by git and must be present locally to run optimization.

### Taichi backend

Default backend is GPU. To switch to CPU, edit `Simulation/main.py`:

```python
ti.init(arch=ti.gpu, ...)   # default
ti.init(arch=ti.cpu, ...)   # CPU fallback
```
