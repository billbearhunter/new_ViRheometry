# Incremental v2 → v4 retrain (v2 untouched)

This document explains how to fold hierarchical-BO infill data into the v2
surrogate to produce a v4 model **without modifying any v2 artefact**.

## Invariants preserved

After this procedure:

| Asset | Status |
|---|---|
| `Models/rbcm_v2/model.pt` | **untouched** (bit-identical) |
| `Models/full_partition/partition.pkl` | **untouched** |
| `TrainingData/moe_workspace_merged_v3_20260419/*.csv` | **untouched** |
| `vi_mogp/k_phi.json` | **untouched** |
| `Models/rbcm_v4_*/...` | **new directory created** |
| `TrainingData/moe_workspace_merged_v4_*` | **new directory created** |

## Prerequisites

1. `Optimization/hier_bo_infill_<ts>_full2000/infill_clean.csv` exists
   (from `diagnose_and_sample_hier_bo.py --max-sims 2000`).
2. The v2 partition is frozen: `Models/full_partition/partition.pkl`.

## Step-by-step

### 1. Build v4 training CSV (v3 + infill, deduplicated)

```bash
cd "C:/Users/xiong/Documents/GitHub/New_ViRheometry"

python DataPipeline/build_merged_dataset.py \
    --v3-dir      TrainingData/moe_workspace_merged_v3_20260419 \
    --infill-csv  Optimization/hier_bo_infill_<ts>_full2000/infill_clean.csv \
    --out-dir     TrainingData/moe_workspace_merged_v4_$(date +%Y%m%d) \
    --dedup-precision 4
```

Outputs:
- `<v4>/train_merged.csv`  — v3 train rows + new infill, de-duplicated to 4 decimal places
- `<v4>/val_merged.csv`    — **unchanged** from v3
- `<v4>/test_merged.csv`   — **unchanged** from v3
- `<v4>/merge_report.json` — provenance: which rows came from where

> **Why val/test unchanged**: we keep the held-out evaluation identical to
> v3/v2 so before/after error comparisons are apples-to-apples.

### 2. Route infill rows through frozen partition (assign cluster_id)

```bash
python -m vi_mogp.dump_cluster_csvs \
    --csv        TrainingData/moe_workspace_merged_v4_*/train_merged.csv \
    --partition  Models/full_partition/partition.pkl \
    --out        Models/rbcm_v4_workspace_$(date +%Y%m%d)
```

Each row in the new train CSV gets a `cluster_id` from the **frozen v2 BGM**.
Expert cells that receive new samples are written to `clusters/cluster_NNNN.csv`.

### 3. Identify experts that gained rows (mark for retrain)

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
v4 = Path("Models/rbcm_v4_workspace_<ts>")
clusters_dir = v4 / "clusters"
# cross-ref each cluster csv against v3's clusters
v3_clusters = Path("Models/rbcm_v2_workspace/clusters")  # if you have it; else compute
AFFECTED = []
for new in clusters_dir.glob("cluster_*.csv"):
    eid = int(new.stem.split("_")[1])
    v3 = v3_clusters / new.name
    n_new = len(pd.read_csv(new))
    n_old = len(pd.read_csv(v3)) if v3.exists() else 0
    if n_new > n_old:
        AFFECTED.append(eid)
        print(f"expert {eid}: {n_old} -> {n_new} (+{n_new-n_old})")
print(f"\n{len(AFFECTED)} experts need retraining")
PY
```

Alternatively: just delete expert checkpoint files that will be retrained —
`train.py` resume-skips existing `expert_NNNN.pt` files, so this is the
intended mechanism.

### 4. Copy v2 expert checkpoints into v4 workspace, DELETE affected ones

```bash
# Start v4 with v2's trained experts — train.py will skip existing, retrain missing
mkdir -p Models/rbcm_v4_workspace_<ts>/experts
mkdir -p Models/rbcm_v4_workspace_<ts>/baselines

# If v2 has a checkpoints dir (may not, depending on how v2 was trained):
#   cp Models/rbcm_v2_workspace_or_similar/experts/*.pt Models/rbcm_v4_workspace_<ts>/experts/
# Otherwise reconstruct per-expert checkpoints from the monolithic model.pt:

python - <<'PY'
from pathlib import Path
import torch
from vi_mogp.model import HVIMoGP_rBCM
m = HVIMoGP_rBCM.load("Models/rbcm_v2/model.pt")
out = Path("Models/rbcm_v4_workspace_<ts>/experts")
for j, exp in enumerate(m.experts):
    torch.save(dict(
        state_dict=exp.state_dict(),
        X_train=exp._X_train.detach().cpu(),
        Y_train=exp._Y_train.detach().cpu(),
        kernel_name=exp.kernel_name,
        loss=float("nan"),
        expert_id=int(j),
    ), out / f"expert_{j:04d}.pt")
print(f"wrote {len(m.experts)} per-expert checkpoints")
PY

# Now DELETE the affected experts so train.py retrains them
python - <<'PY'
AFFECTED = [...]  # from step 3
from pathlib import Path
for eid in AFFECTED:
    p = Path(f"Models/rbcm_v4_workspace_<ts>/experts/expert_{eid:04d}.pt")
    if p.exists():
        p.unlink()
        print(f"deleted {p}")
PY
```

### 5. Retrain (frozen partition → only missing experts fit)

```bash
python -m vi_mogp.train \
    --out              Models/rbcm_v4_workspace_<ts> \
    --load-partition   Models/full_partition/partition.pkl \
    --data             TrainingData/moe_workspace_merged_v4_<ts>
```

Only the experts whose `.pt` file is missing (the affected ones) get Adam-
trained; all others are loaded from the copied v2 checkpoints and skipped.

Expected time: 10-30 min for ~30-80 affected experts (vs 9h for full retrain).

### 6. Optional — refit poly residual on the new model

```bash
python -m vi_mogp.add_poly_residual \
    --src Models/rbcm_v4_workspace_<ts>/model.pt \
    --out Models/rbcm_v4_workspace_<ts>/model_poly.pt
```

Takes ~1-2 min. Not strictly required but usually improves per-expert RMSE
by a few percent on dense clusters.

### 7. Verify: re-run per-expert error analysis on v4

```bash
python tests/per_expert_errors.py \
    --model Models/rbcm_v4_workspace_<ts>/model_poly.pt \
    --out_root tests/per_expert_analysis_v4
```

Compare against the v2 results to quantify the improvement on the
previously-bad experts.

## Rollback

If v4 is worse than v2 on validation:

```bash
# v2 is untouched — just use model path Models/rbcm_v2/model.pt
# v4 workspace can be deleted or archived
mv Models/rbcm_v4_workspace_<ts> _archive_v4_bad_<ts>
```

No v2 artefact has been modified at any point.

## Paper claim

After v4 validates:

> "We demonstrate targeted hierarchical Bayesian infill: 2000 additional
> MPM simulations, selected by a two-level (per-geo then per-expert) BO
> loop using the frozen v2 partition, are added to the training corpus.
> Only the ~N affected experts are retrained; the geometry gate, φ-gate,
> and unchanged experts are preserved verbatim from v2. This reduces the
> aggregate per-expert RMSE from 0.17 cm (v2) to 0.XX cm (v4) while
> preserving the held-out val/test split."
