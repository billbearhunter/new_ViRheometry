# Worker Task: Uniform LHS Fill for Remaining gids (1-9, 11-24)

**Date**: 2026-05-07  
**Total gids**: 23 (gid 0 reserved for master, gid 10 done)  
**Per-gid sims**: 10,000  
**Total sims**: 230,000  
**Estimated time**: ~6.5 hours per gid × 23 = **~150 hours total**  
**Output**: one CSV per gid in `outputs/uniform_fill/lhs_gid<G>_uniform.csv`

---

## Background

Same uniform LHS infill we ran for gid=10. For each gid:
- LHS in **log-space** for eta [0.72, 300] and sigma_y [0.003, 400]
- n linear in [0.30, 1.00]
- W, H sampled within the gid's geometry box (guarantees gid routing)
- Each result routed to sub_id via partition state.pkl

This fills the sparse parts of the (n, log eta, log sigma_y) space that
were under-sampled in the original linear LHS.

State.pkl files are bundled at `worker_deps/gid<G>/state.pkl` for all gids.
No dependency on the sibling `Fast-Non-Newtonian` repo.

---

## Prerequisites

1. Python 3.11 with Taichi 1.7.x, PyTorch, scikit-learn, pandas, numpy
2. CUDA GPU
3. `New_ViRheometry` repo (`git pull` to get latest)
4. Simulation code (`HeadlessSimulatorMLS` / `AGTaichiMPM2`) installed

```powershell
python -c "import taichi; import torch; import sklearn; print('OK')"
```

---

## gid → (W, H) reference

| gid | W range | H range | gid | W range | H range |
|-----|---------|---------|-----|---------|---------|
|  1  | [2.0, 3.0] | [3.0, 4.0] | 13  | [4.0, 5.0] | [5.02, 6.05] |
|  2  | [2.0, 3.0] | [4.0, 5.02] | 14 | [4.0, 5.0] | [6.05, 7.0] |
|  3  | [2.0, 3.0] | [5.02, 6.05] | 15 | [5.0, 6.02] | [2.0, 3.0] |
|  4  | [2.0, 3.0] | [6.05, 7.0] | 16  | [5.0, 6.02] | [3.0, 4.0] |
|  5  | [3.0, 4.0] | [2.0, 3.0] | 17  | [5.0, 6.02] | [4.0, 5.02] |
|  6  | [3.0, 4.0] | [3.0, 4.0] | 18  | [5.0, 6.02] | [5.02, 6.05] |
|  7  | [3.0, 4.0] | [4.0, 5.02] | 19 | [5.0, 6.02] | [6.05, 7.0] |
|  8  | [3.0, 4.0] | [5.02, 6.05] | 20 | [6.02, 7.0] | [2.0, 3.0] |
|  9  | [3.0, 4.0] | [6.05, 7.0] | 21  | [6.02, 7.0] | [3.0, 4.0] |
| 11  | [4.0, 5.0] | [3.0, 4.0] | 22  | [6.02, 7.0] | [4.0, 5.02] |
| 12  | [4.0, 5.0] | [4.0, 5.02] | 23  | [6.02, 7.0] | [5.02, 6.05] |
|     |            |             | 24  | [6.02, 7.0] | [6.05, 7.0] |

(Bigger geometries → more particles → slower per sim.)

---

## How to run

### Recommended: one gid at a time, in numerical order

```powershell
cd New_ViRheometry/pipeline_v10/data_infill
git pull
```

For **each gid G** in this list (1-9, 11-24), run:

```powershell
$G = 1     # change to next gid each iteration
python scripts/run_uniform_lhs_gid.py `
    --gid $G `
    --n-sims 10000 `
    --arch cuda `
    --seed (2000 + $G) `
    --state-pkl worker_deps/gid$G/state.pkl
```

Seed convention: `seed = 2000 + gid` (gid 1 → 2001, gid 11 → 2011, …,
gid 24 → 2024). Different from master's seed=42 (gid 0) and your own
gid=10 seed=2026, so no LHS overlap.

### Resumability

Each run is fully resumable. If interrupted, re-run the **same command** —
it counts existing rows in `outputs/uniform_fill/lhs_gid<G>_uniform.csv`
and generates only the remaining LHS points.

### Suggested schedule

Run them serially in a wrapper script if you want to leave it overnight:

```powershell
# run_all_remaining.ps1
$gids = @(1..9) + @(11..24)
foreach ($g in $gids) {
    Write-Host "=== gid=$g ==="
    python scripts/run_uniform_lhs_gid.py `
        --gid $g `
        --n-sims 10000 `
        --arch cuda `
        --seed (2000 + $g) `
        --state-pkl worker_deps/gid$g/state.pkl
}
```

Each gid takes ~6.5 hours so plan multi-day runs.

---

## Verify after each gid completes

```powershell
$G = 1
python -c @"
import pandas as pd, numpy as np
df = pd.read_csv('outputs/uniform_fill/lhs_gid$G`_uniform.csv')
print(f'rows: {len(df):,}')
print(f'subs covered: {df[\"sub_id\"].nunique()}')
print(f'log10(eta): p5={np.percentile(np.log10(df.eta),5):.2f} p95={np.percentile(np.log10(df.eta),95):.2f}')
print(f'log10(sy):  p5={np.percentile(np.log10(df.sigma_y.clip(0.001)),5):.2f} p95={np.percentile(np.log10(df.sigma_y.clip(0.001)),95):.2f}')
"@
```

p5 / p95 of log10(eta) and log10(sigma_y) should be roughly evenly spread —
that confirms the log-space LHS worked.

---

## After all gids complete

Compress the outputs and push:

```powershell
cd outputs/uniform_fill
git add lhs_gid*_uniform.csv
git commit -m "worker: uniform LHS fill gids 1-9, 11-24"
git push
```

Each CSV is ~3-5 MB, total ~80 MB. Git can handle it but if you want
zip them first:

```powershell
Compress-Archive lhs_gid*_uniform.csv worker_uniform_fill.zip
```

Master will then merge each gid into its `state_gid_<G>/sub_assignments.csv`
and decide whether to repartition + retrain.

---

## Order priority (if you want to deprioritize some)

Larger geometry → more particles → slower per sim. If GPU time is
limited, do these gids first (small, fast):

**Priority 1 (small geometry, fast):** gid 1, 2, 5, 6, 11

**Priority 2 (medium):** gid 3, 7, 12, 15, 16

**Priority 3 (large, slowest):** gid 8, 9, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24

Each priority tier is one overnight run if you commit ~30-40 hours.

---

## Troubleshooting

### CUDA OOM on large gids (W, H ≥ 6)
The simulator pre-allocates particles for the max cuboid. If you OOM on
gid 19 / 23 / 24 (largest geometry), drop to CPU:

```powershell
python scripts/run_uniform_lhs_gid.py --gid $G --n-sims 10000 --arch cpu --seed (2000 + $G) --state-pkl worker_deps/gid$G/state.pkl
```

CPU is ~5x slower (~30 hours per gid instead of 6.5).

### Script crashes mid-way
Re-run the same command. It resumes from the last checkpoint flush
(every 20 sims).

### Some gids missing state.pkl
```powershell
ls worker_deps/gid1/state.pkl  # should exist after git pull
```
If missing, master needs to push it first.

---

## Summary

| | |
|--|--|
| gids | 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 |
| sims/gid | 10,000 |
| total sims | 230,000 |
| time/gid | ~6.5 hours (cuda) |
| total time | ~150 hours (sequential) |
| output | `outputs/uniform_fill/lhs_gid<G>_uniform.csv` |
| schema | n, eta, sigma_y, width, height, x_01..x_08, sim_seconds, sub_id, cluster_id |
| seed | 2000 + gid (avoid LHS collision with master and other workers) |
