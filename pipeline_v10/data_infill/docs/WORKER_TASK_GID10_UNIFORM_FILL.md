# Worker Task: Uniform LHS Fill for gid=10

**Date**: 2026-05-06  
**Estimated time**: ~6-7 hours (10,000 sims at ~26 sim/min on RTX-class GPU)  
**Output**: `outputs/uniform_fill/lhs_gid10_uniform.csv`  
**Send back to**: master machine

---

## Background

The `v10_yshape_planB` GP surrogate bank has **25 geo-IDs (gids)** partitioned by
measurement geometry (W x H). gid=10 covers **W:[4.0, 5.0] cm, H:[2.0, 3.0] cm**.

Diagnostic shows gid=10 training data is **heavily skewed** toward high eta and high
sigma_y (both were originally sampled linearly, not log-uniformly):

| variable | p25 | median | p75 |
|----------|-----|--------|-----|
| log10(eta) | 1.29 → eta~20 | 2.02 → eta~104 | 2.30 → eta~200 |
| log10(sy)  | 0.67 → sy~4.6 | 2.05 → sy~112  | 2.40 → sy~251  |

3D CoV = 2.32 (ideal = 0). 106/150 cells have N<50. This script fills
the sparse low-eta / low-sigma_y region by running LHS **in log-space**.

---

## Prerequisites

1. Python 3.11 with Taichi 1.7.x, PyTorch, scikit-learn, pandas, numpy
2. CUDA GPU (RTX 3070 or better recommended)
3. This repo: `New_ViRheometry` (git pull to get latest)
4. Simulation code (HeadlessSimulatorMLS / AGTaichiMPM2) already installed
   on this machine — no extra repo needed.
5. The partition state file is bundled in this repo:
   `pipeline_v10/data_infill/worker_deps/gid10/state.pkl` (2.8 KB)

### Quick environment check
```powershell
python -c "import taichi; import torch; import sklearn; print('OK')"
```

---

## Step 1: Pull latest code

```powershell
cd New_ViRheometry
git pull
cd pipeline_v10/data_infill
```

---

## Step 2: Run the uniform LHS simulation

```powershell
python scripts/run_uniform_lhs_gid.py `
    --gid 10 `
    --n-sims 10000 `
    --arch cuda `
    --seed 2026 `
    --state-pkl worker_deps/gid10/state.pkl
```

### Parameters explained

| flag | value | meaning |
|------|-------|---------|
| `--gid` | 10 | gid=10, W:[4,5]cm H:[2,3]cm |
| `--n-sims` | 10000 | total simulations (all land in gid=10) |
| `--arch` | cuda | use GPU (change to `cpu` if no CUDA) |
| `--seed` | 2026 | random seed (different from master's seed=42 on gid=0) |

### What it does
- LHS in **log-space** for eta [0.72, 300] and sigma_y [0.003, 400]
- n sampled uniformly in [0.30, 1.00]
- W, H sampled uniformly in [4.0, 5.0] x [2.0, 3.0] (guarantees gid=10 routing)
- Each result is routed through the partition: norm_curve -> PCA -> KMeans -> sy-threshold -> sub_id
- Progress saved every 20 sims (resumable if interrupted)

### Expected output snippet
```
Uniform LHS fill  gid=10  W:[4.000,5.000]  H:[2.000,3.000]
  n_sims=10000  done=0  remaining=10000
  n:      [0.3, 1.0]  (linear)
  eta:    [0.72, 300.0]  (log-space LHS)
  sigma_y:[0.003, 400.0]  (log-space LHS)
  arch=cuda  seed=2026

Initialising simulator (arch=cuda)...
[Taichi] Starting on arch=cuda
  [20/10000]  26.x sim/min  ETA ~385 min
  ...

Done: 10000 simulated, 0 failed  (384.6 min total)
Saved -> outputs/uniform_fill/lhs_gid10_uniform.csv

Sub-ID distribution (new points):
  sub_0000:  xxx
  sub_0001:  xxx
  ...
```

### Expected time
~385 min (~6.5 hours) at 26 sim/min. The script is resumable -- if interrupted,
re-run the same command and it will skip already-simulated rows.

---

## Step 3: Verify output

```powershell
python -c "
import pandas as pd
df = pd.read_csv('outputs/uniform_fill/lhs_gid10_uniform.csv')
print(f'Rows: {len(df)}')
print(f'Subs covered: {df[\"sub_id\"].nunique()}')
print(df['sub_id'].value_counts().sort_index())
print('log10(eta) stats:')
import numpy as np
print(np.percentile(np.log10(df.eta), [5,25,50,75,95]).round(2))
print('log10(sigma_y) stats:')
print(np.percentile(np.log10(df.sigma_y.clip(0.001)), [5,25,50,75,95]).round(2))
"
```

Expected: log10(eta) and log10(sigma_y) should now be **approximately uniform**
(each quantile spaced evenly), confirming the log-space LHS worked.

---

## Step 4: Send results to master

Compress and transfer the CSV:

```powershell
# Compress
Compress-Archive outputs/uniform_fill/lhs_gid10_uniform.csv `
    outputs/uniform_fill/lhs_gid10_uniform.zip

# Transfer via whatever file-sharing method (scp, OneDrive, USB, etc.)
# File size estimate: ~10000 rows x ~20 cols = ~3 MB uncompressed, <1 MB compressed
```

---

## What master will do with this file

1. **Merge into training pool**: append rows to `state_gid_10/sub_assignments.csv`
   with `split=train`
2. **Retrain affected subs**: run `train_yshape_subs.py --gid 10` to retrain
   all sub-GPs in gid=10 that received new data
3. **Check rwrms improvement**: verify val_rwrms drops, especially for subs
   that were previously sparse

---

## Troubleshooting

### CUDA OOM
```powershell
python scripts/run_uniform_lhs_gid.py --gid 10 --n-sims 10000 --arch cpu --seed 2026
```
Slower (~5x) but no GPU required.

### "state.pkl not found"
The state file is bundled in the repo. Make sure you ran `git pull` and
the `--state-pkl` flag points to the right place:
```powershell
ls pipeline_v10\data_infill\worker_deps\gid10\state.pkl
```

### Script crashes mid-way
Just re-run the same command. It will resume from where it left off
(counts existing rows in the output CSV and generates remaining LHS points).

### Taichi version mismatch
Match the master machine: Taichi 1.7.4, Python 3.11.9.

---

## Summary

| Item | Value |
|------|-------|
| gid | 10 |
| W range | [4.0, 5.0] cm |
| H range | [2.0, 3.0] cm |
| n range | [0.30, 1.00] |
| eta range | [0.72, 300.0] Pa.s^n (log LHS) |
| sigma_y range | [0.003, 400.0] Pa (log LHS) |
| n_sims | 10,000 |
| seed | 2026 |
| ETA | ~6.5 hours (cuda) |
| output | `outputs/uniform_fill/lhs_gid10_uniform.csv` |
