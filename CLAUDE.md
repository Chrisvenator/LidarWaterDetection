# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- **Python**: 3.12.3 (CPython) via `.venv/` (virtualenv with system site-packages)
- **Activate**: `source .venv/bin/activate`
- **Install packages**: `pip install <package>` (or `pip install --break-system-packages <package>` if needed outside venv)

The `.venv/` and `data/` directories are gitignored and not in version control.

## Project Goal

Build a **water vs. land classifier** from **full-waveform bathymetric LiDAR point cloud data** with **no labeled training data**. The approach:

1. Build a **rule-based auto-labeler** from domain knowledge
2. Use auto-labels to bootstrap a supervised ML model

## Folder Structure

```
LidarWaterDetection/
├── src/
│   ├── labeling/          # Auto-labeling pipeline (waveform-based + surface model)
│   │   ├── auto_labeler.py         ← ACTIVE: waveform-only labeler (v6), outputs labels_current.csv
│   │   ├── water_surface_model.py  ← ACTIVE: tight footprint + local adaptive surface (v8/v10)
│   │   └── river_boundary.py       ← ACTIVE: probability-field contour extraction (v10 + final boundary)
│   ├── features/          # Feature engineering
│   │   ├── feature_extractor.py    (original 35-col extractor)
│   │   ├── add_features.py         (adds 7 generalizable features → features_current.csv)
│   │   └── canopy_features.py      ← ACTIVE: echo/DTM/structure features for canopy model
│   ├── training/          # Model training scripts
│   │   ├── baseline_model.py       (XGBoost baseline helper)
│   │   ├── deep_model.py           (WaveformNet definition)
│   │   ├── train_deep.py           (deep model training)
│   │   ├── preprocess_wcn.py       ← ACTIVE: one-time prep for WCN v9 (waveform_grids_norm.npy, features_v9.csv)
│   │   ├── train_wcn_v9.py         ← ACTIVE: WCN v9 three-phase training → models/wcn_v9/
│   │   └── train_canopy.py         ← ACTIVE: canopy XGBoost (z-band bootstrap) → models/canopy_v1/
│   ├── evaluation/        # Metrics, comparison, export
│   │   ├── evaluate.py
│   │   ├── compare_models.py
│   │   └── export_predictions.py
│   ├── diagnostics/       # Exploratory / diagnostic scripts (no training)
│   │   └── diagnostic_transition_zone.py  (z-band waveform + separability analysis)
│   └── inference/
│       ├── inference_pipeline.py   (two-stage cascade inference on full 234k points)
│       └── merge_canopy.py         ← ACTIVE: merges canopy preds into v10 → labeled_pointcloud_final.csv
│
├── models/                # Trained model artifacts
│   ├── current/           (v8 surface-model outputs: v8_xgb.json, v8_deep.pt, v8_deep_stats.json, v8_metrics.json)
│   ├── wcn_v9/            ← CURRENT: wcn_pretrained.pt, wcn_finetuned.pt, wcn_refined.pt, wcn_xgb.json, wcn_stats.json
│   ├── v10/               (v10 geometry-only outputs: plots + metrics from --geometry-only run)
│   ├── canopy_v1/         ← CURRENT: canopy_xgb.json, metrics.json, diagnostic plots
│   ├── final/             ← FINAL: canopy-aware river boundary (boundary.geojson + plots)
│   ├── labeling/          (auto_labeler outputs: v6_xgb.json, v6_deep.pt, diagnostic plots)
│   └── diagnostics/       (exploratory plots: reflectance histograms, waveform clusters, etc.)
│
├── data_processed/        # Computed features, labels, waveform grids (not raw data)
│   ├── features_current.csv    (42-col feature matrix)
│   ├── features_v9.csv         ← CURRENT: 11 generalizable features + x, y, z (WCN v9 input)
│   ├── canopy_features.csv     ← CURRENT: echo/DTM/structure features (canopy model input, cached)
│   ├── labels_current.csv      (current auto-labels from auto_labeler.py)
│   ├── waveform_grids.npy      (234024 × 200 dense amplitude grids, raw)
│   └── waveform_grids_norm.npy ← CURRENT: per-sample max-normalised grids (WCN v9 input)
│
├── pointclouds/           # Final classified point clouds (open in CloudCompare)
│   ├── labeled_pointcloud_waveform_only.csv  (intermediate: auto_labeler output)
│   ├── labeled_pointcloud_current.csv        (v8 surface model output)
│   ├── labeled_pointcloud_wcn.csv            ← CURRENT: WCN v9 predictions
│   ├── labeled_pointcloud_v10.csv            ← CURRENT: v10 geometry-only output (0=land,1=water,2=uncertain,3=recon-water)
│   ├── labeled_pointcloud_canopy.csv         ← CURRENT: canopy predictions (canopy_proba, canopy_pred)
│   └── labeled_pointcloud_final.csv          ← FINAL: v10 + canopy merged (0=land,1=water,2=uncertain,3=water-under-canopy,4=canopy)
│
├── context/               # Domain knowledge, architecture spec, project notes
│   ├── architecture_recommendation.md
│   ├── knowledge_base.md
│   ├── project_spec.md
│   └── file_inventory.md
│
├── runs/                  # Per-dataset outputs (gitignored) — one tree per survey
│   └── <dataset>/{cache,models,pointclouds,plots}/ + site_profile.json, metrics.json
│
└── data/                  # RAW data (gitignored), one folder per survey
    ├── Pielach/           ← the study area this pipeline is built on
    │   ├── point_cloud_df.txt  (~19MB, 234k rows: x, y, z, reflectance_dB)
    │   └── waveform_df.txt     (~92MB, Time [SI] + Amplitude [ADC] as list strings)
    └── Inn_DeepLearning/  ← second survey; same format, very different site
        ├── point_cloud_df_inn.txt  (~19MB, 230k rows)
        └── waveform_df_inn.txt     (~795MB, full-range-gate records)
```

## Active Pipeline (run in order)

All paths are resolved relative to the repo root via `ROOT = Path(__file__).resolve().parent.parent.parent`.

```bash
source .venv/bin/activate

# Stage 1 — initial waveform-only labels (v6)
python src/labeling/auto_labeler.py
    # → data_processed/labels_current.csv
    # → pointclouds/labeled_pointcloud_waveform_only.csv
    # → models/labeling/

# Stage 2 — surface-model refinement (v8): geometry phases + V8Net/XGBoost retrain
python src/labeling/water_surface_model.py
    # → pointclouds/labeled_pointcloud_current.csv
    # → models/current/

# Stage 3 — WCN v9 training (one-time preprocessing first)
python src/training/preprocess_wcn.py
    # → data_processed/waveform_grids_norm.npy
    # → data_processed/features_v9.csv
python src/training/train_wcn_v9.py
    # → models/wcn_v9/{wcn_pretrained,wcn_finetuned,wcn_refined}.pt
    # → models/wcn_v9/wcn_xgb.json, wcn_stats.json
    # → pointclouds/labeled_pointcloud_wcn.csv

# Stage 4 — v10: geometry phases driven by WCN v9 probas (no retraining)
python src/labeling/water_surface_model.py \
    --geometry-only \
    --label-src pointclouds/labeled_pointcloud_wcn.csv \
    --out pointclouds/labeled_pointcloud_v10.csv \
    --plot-dir models/v10/
    # → pointclouds/labeled_pointcloud_v10.csv
    # → models/v10/

# Stage 5 — canopy model (needs v10: water surface = ground reference over the river)
python src/features/canopy_features.py
    # → data_processed/canopy_features.csv  (cached — use --force to rebuild)
python src/training/train_canopy.py
    # → models/canopy_v1/
    # → pointclouds/labeled_pointcloud_canopy.csv

# Stage 6 — final merge: canopy refines v10 land/uncertain classes
python src/inference/merge_canopy.py
    # → pointclouds/labeled_pointcloud_final.csv

# Stage 7 — canopy-aware river boundary (vector product)
python src/labeling/river_boundary.py \
    --input pointclouds/labeled_pointcloud_final.csv \
    --out-dir models/final
    # → models/final/boundary.geojson (+ heatmap/scatter plots)
    # canopy points excluded as evidence; water/recon-water=1.0, land=0.0,
    # uncertain=deep_proba; canopy gaps bridged by nearest-neighbour fill
```

## Running on a Different Survey

The scripts under `src/labeling/`, `src/features/`, `src/training/` are
Pielach-only by construction (hardcoded paths, absolute z constants). Use the
`lidarwater` library for any other dataset:

```bash
python scripts/run_dataset.py data/Inn_DeepLearning --profile-only   # inspect
python scripts/run_dataset.py data/Inn_DeepLearning --fit            # train + classify
```

`derive_site_config(cloud)` (`src/lidarwater/site.py`) measures the four
things the Pielach defaults actually depend on and rebases them: water level
(densest 0.1 m z-bin) shifts every absolute z threshold; a reflectance
percentile shifts the dB gates; waveform energy distribution picks
`FeatureConfig.grid_origin`; height above a per-cell ground surface decides
whether the site has canopy at all. Architecture, hyperparameters and
dimensionless ratios are untouched — on the Pielach cloud the derivation is
an exact no-op, so `pytest -m golden` still passes.

Outputs go to `runs/<dataset>/` (`Workspace`), never into the repo-level
`models/`, `pointclouds/`, `data_processed/` trees, which belong to Pielach.

**Five failure modes worth knowing** (all auto-detected):
- **Full-record waveforms.** Pielach's SVB export stores only samples around
  each echo, so `times[0]` is the first return. Inn's export digitises the
  whole range gate — ~700 samples, first ~550 pre-trigger noise. Anchoring
  the 200-bin grid at `times[0]` captured 7.8% of Inn's waveform energy;
  `grid_origin="first_return"` raises it to 87%.
- **Pulse length.** `energy_concentration` = share of energy in a fixed
  30-bin window sized to Pielach's ~60-bin pulse; the geometry stage gates
  surface candidates at `> 0.85`. Inn's returns span ~153 bins, so no point
  reaches 0.85 — the gate passed 48.6% of Pielach but 0.78% of Inn, leaving
  79 RANSAC candidates and aborting the stage. The threshold is rebased by
  percentile (0.294 on Inn → 13,481 candidates). The *window* is not
  rebased, so the feature is weaker on very different pulse lengths.
- **Full-record noise pedestal.** ~80% of a full-range-gate record is
  digitiser noise above zero, so `n_gaps`/`n_clusters`/`active_bins_ratio`
  measure the recording format, and the WCN's occupancy-mask channel goes
  all-ones. `FeatureConfig.noise_gate` drops noise-floor samples: on Inn it
  moved `n_gaps` +4.86 sigma -> -0.81, `n_clusters` likewise,
  `energy_ratio_late` +3.02 -> +0.66. `WcnConfig.standardize="site"`
  recomputes the scalar z-scoring from the cloud under test (the shipped
  stats saturated the net to 97.6% water; site stats give 68.6%) — a
  mitigation only: head agreement stays ~39% either way. Cross-site
  inference is a smoke test, not a result.
- **Reflectance excluded from the WCN.** WCN v9's 11 scalar features are all
  dimensionless by design — `reflectance_dB` is left out because its dB scale
  is a property of the scanner and survey, so it does not transfer between
  sites. On Inn that is the dominant water/land signal: measured against
  hand-labelled points, a single reflectance threshold at +0.25 dB gets 97.2%
  balanced accuracy (29/29 water, 34/36 land), and on geometric proxies
  reflectance alone scores AUC 0.973 against 0.924 for all 11 waveform
  features (0.983 combined). A model fitted for one site is only ever used
  there, so `derive_site_config` appends `reflectance_dB` to
  `WcnConfig.scalar_features` (n_scalar 11 -> 12). `wcn.predict` reads the
  feature list from the checkpoint's own stats file, so the shipped 11-feature
  Pielach model still loads unchanged.
- **Land below the water surface — the root cause of everything above.**
  Every label in the pipeline descends from `create_labels(z, zones)`, which
  takes elevation and nothing else. The WCN never sees z as a *feature*, but
  z is its entire *teaching signal*, so it learns to reproduce an elevation
  rule from waveforms — and does, at F1 0.992. Measured on Inn: 97.68% of the
  final output is reproduced by the single rule `z < 380.44 m`.
  Where elevation genuinely separates (Pielach) that is fine. Inn's bank
  rises then falls below water level, so it is wrong by construction, and the
  model is confidently wrong rather than randomly wrong.
  `BootstrapConfig(method="surface")` replaces it with an elevation-free
  bootstrap: rasterise, take each cell's top surface, keep cells whose top
  agrees with their neighbours' (a coherent sheet), and split those by the
  median reflectance of the *surface layer only*. Isolating the surface is
  what makes the reflectance histogram bimodal — over all points the deepest
  Otsu valley falls inside the water mode (-7.10 dB, 61.0% balanced accuracy
  against hand-labelled points); over surface layers it lands at -0.46 dB and
  97.2% (100% water, 94.4% land), within 0.7 dB of the hand-labelled optimum.
  Stable across 2-4 m cells. Selected by `derive_site_config`; the z-bands
  remain the default so Pielach is untouched.
  Note flatness alone is *not* enough: a gravel bar is a flat coherent sheet
  too (cell-wise flatness gives 100% water recall but ~40% on land), and
  per-point planarity is actively misleading here (water 0.61, land 0.73 —
  a k-NN ball around a water point catches riverbed returns beneath it).
- **Sites without vegetation.** The canopy stage probes how much of the cloud
  stands >3 m above the water-aware ground reference; below
  `CanopyConfig.min_canopy_frac` it returns all-zero probabilities without
  loading or training a model. Phase 3b (waterbed reconstruction) switches off
  the same way below `BedReconstructionConfig.min_canopy_frac` — with no crowns
  there is no water hidden under them, and on Inn it was inventing 17.9% of the
  cloud as a class the site does not contain.

`SurfaceConfig.resolve_uncertain` (default off) decides class 2 by the model's
own probability instead of emitting it: 97.4% -> 98.7% balanced accuracy on
Inn against 192k hand-labelled points, and 93.1% -> 96.6% on the blind 65.
Almost all residual error is abstention, not misclassification — of 7,502
missed water points, 5,395 were labelled uncertain and only 2,107 land.

`CleanupConfig.majority_filter` (default off) flips points contradicted by
their neighbours; canopy is never moved. With both it and `resolve_uncertain`
on, Inn reaches 99.3% against 192k hand labels and 98.3% on the blind 65.

**Do not attempt a cross-site model.** Measured both directions: waveform-only
transfer is chance (AUC 0.42-0.69) while within-site is 0.97-0.99, because
Pielach's strongest features (`gap_ratio`, `n_gaps`, `n_clusters`,
`max_amp_norm_by_energy`) describe its SVB export format and die or reverse on
Inn's full-record export. Only reflectance *percentile rank* transfers
(Pielach-trained on Inn hand labels: AUC 0.958), and adding waveform features
to it drops that to 0.745. Full numbers: `context/cross_site_transfer.md`.

Measured site profiles: Pielach — water 260.29 m, reflectance p5/50/95
-28.6/-22.3/-12.3 dB, energy gate 0.845, 46.1% of points >3 m above ground.
Inn — water 380.76 m (+120.5 m), reflectance +14.1 dB (p95 = +5.6 dB, i.e.
positive), energy gate 0.294, grid_origin=first_return, 1.3% of points >3 m
above ground.

## Data

Raw data at `data/` (gitignored):

| File | Size | Description |
|------|------|-------------|
| `data/Pielach/point_cloud_df.txt` | ~19 MB | ~234k rows: `x, y, z, _riegl.reflectance` (dB, negative values) |
| `data/Pielach/waveform_df.txt` | ~92 MB | Same row count: `Time [SI], Amplitude [ADC]` (stored as Python list strings) |
| `data/Inn_DeepLearning/point_cloud_df_inn.txt` | ~19 MB | 230,559 rows, same columns; z 377.5–386.4 m |
| `data/Inn_DeepLearning/waveform_df_inn.txt` | ~795 MB | Full-range-gate records: median 415 samples/row spanning t 0–700, noise floor ~35 ADC |

- Coordinates: ETRS89/UTM 33N (EPSG: 25833), local-offset (x ≈ -269, y ≈ 97, z ≈ 261–264)
- One waveform row can correspond to multiple point cloud rows (shared pulse, multiple extracted echoes)
- `Time [SI]`: sample intervals (~0.5 ns each); `Amplitude [ADC]`: raw digitizer counts
- Waveform time arrays are **non-contiguous** — gaps between clusters indicate distinct returns

## Domain Context (Critical)

**Scanner**: RIEGL VQ-840-GL, **532 nm green laser** (topo-bathymetric UAV LiDAR)
- Green light penetrates water → one pulse can yield returns from: (1) water surface, (2) water column backscatter, (3) riverbed
- Processing pipeline uses **OWP** (Online Waveform Processing) + **SVB** (Surface-Volume-Bottom) algorithm

**Study area**: Pielach River, Austria — surveyed October 2024, ~250k points at 15 cm spacing (downsampled from 1.6M)

**Elevation zones** (narrow river valley — critical for understanding feature behavior):
- z > 263.283m: 100% canopy — user-verified June 2026 (banks are tree-covered; only
  ~1% flat-ground label noise, handled by the open-sky-low exclusion)
- z 260.408–263.283m: transition — may be canopy, ground, or water; model must classify
- z ≤ 260.1m: water (surface + riverbed), 0% canopy
- Water surface sits at 259.85–260.75m (v10 local surface model)

**Key classification signals** (corrected — counter-intuitive):
- Water waveforms are **compact and early** — high `energy_concentration`, high `max_amp_norm_by_energy`
- Dry ground/gravel produces **complex multi-return** waveforms — more peaks and gaps than water
- `height_above_local_min_10m` is NOT a canopy detector in this valley — the river provides the 10m minimum for all bank points
- **Water depth masquerades as vegetation height** if the DTM (= riverbed under water)
  is the ground reference — always use max(DTM, water surface) for height features
- **Open-sky-low = not canopy**: ≤1 neighbor ≥2m overhead + <1.2m above reference is
  ground cover (grass/gravel/water); dipping branches keep their crown overhead
- **Pulse recovery**: hashing waveform rows groups shared pulses (1:1 row alignment
  with the point cloud; duplicate rows = multi-echo pulse) → echo rank features

## Model Architecture

### v6 — auto_labeler.py (waveform-only bootstrap)
Trains V6Net + XGBoost using waveform features + `reflectance_dB` — **no elevation, no geometry**.
Zone boundaries verified by manual cross-section inspection in CloudCompare.

### v8 — water_surface_model.py (surface-model refinement)
Refines v6 labels using a 5-phase geometry pipeline, then retrains V8Net + XGBoost:

- **Phase 1**: Two-tier concave-hull footprint from high-conf water anchors (eroded inward 1m)
  - Tier 1: conf ≥ 0.8, z < 259.6m (riverbed) | Tier 2: conf ≥ 0.85, z < 261.5m (surface)
- **Phase 2**: 2m-cell local surface grid, Gaussian-smoothed, RANSAC fallback for distant cells
- **Phase 3**: Inside footprint + z ≤ surface+0.3m → WATER; above → LAND
- **Phase 3b**: Waterbed reconstruction — recovers tree-over-water points missed by Phase 3 (label=3)
- **Phase 4**: Retrain V8Net + XGBoost on geometry-refined labels

**V8Net architecture** (must match `water_surface_model.py` and `inference_pipeline.py`):
`_CB(1,32,3) → _CB(32,64,5) → _CB(64,64,11) → MaxPool1d(4) → _CB(64,128,5) → AdaptiveAvgPool1d(1)`
fused with spatial MLP: `128 → 64 → 32`, head: `160 → 128 → 64 → 2`. No Dropout.

### WCN v9 — train_wcn_v9.py (current best deep model)
Three-phase training on 11 generalizable waveform features:
1. **Masked waveform autoencoder** — self-supervised, no labels (~50 epochs)
2. **Supervised fine-tuning** — confidence-weighted focal loss (~150 epochs)
3. **Pseudo-label refinement** — 2 rounds × 30 epochs

Input features: `features_v9.csv` (11 dimensionless ratios/counts). Input grids: `waveform_grids_norm.npy` (per-sample max-normalised). Deploy: `wcn_refined.pt` + `wcn_xgb.json`.

### v10 — water_surface_model.py --geometry-only (current best output)
Runs all geometry phases (1–3b) using WCN v9 probas as anchors — **no Phase 4 retraining**.
`wcn_proba` column auto-aliased to `deep_proba` for compatibility.
Output label values: 0=land, 1=water, 2=uncertain, 3=reconstructed-water (tree-over-water recovery).

**river_boundary.py** is used by v10 and v8 to draw probability-field contours at three thresholds:
- `PROB_INNER=0.65` (conservative), `PROB_CENTER=0.50`, `PROB_OUTER=0.35` (generous)

### Canopy v1 — train_canopy.py + merge_canopy.py (final stage)
XGBoost bootstrapped from z-band labels (z > 263.283 canopy, z ≤ 260.1 not) plus two
pseudo-negative rules (water-surface points; open-sky-low: n_above_2m ≤ 1 AND
height_above_ref < 1.2 m). **z is never a model feature** — features are echo rank
within pulse (waveform-row hashing recovers shared pulses), height above a
water-aware ground reference (DTM, raised to the v10 water surface), echo ratio,
points-overhead counts, and waveform shape. Runs AFTER v10 because the reference
surface needs the water model (water depth otherwise masquerades as vegetation
height). merge_canopy.py then overrides v10 land/uncertain with canopy (label 4);
water labels win conflicts. Full design + iteration log: `context/canopy_plan.md`.

## Parsing Waveform Data

Waveform columns are stored as **numpy `repr()` strings** — space-separated,
multi-line, e.g. `"[ 37  38  39 ...\n  55  56 ...]"`. This is NOT valid
Python list syntax: `ast.literal_eval` raises `SyntaxError` on it (missing
commas). Parse by extracting integers with a regex:

```python
import re
import numpy as np
import pandas as pd

NUM_RE = re.compile(r'[-+]?\d+')

def parse_array_string(s: str) -> np.ndarray:
    return np.array(NUM_RE.findall(s), dtype=np.int64)

wf = pd.read_csv('data/Pielach/waveform_df.txt')
wf['Time [SI]'] = wf['Time [SI]'].apply(parse_array_string)
wf['Amplitude [ADC]'] = wf['Amplitude [ADC]'].apply(parse_array_string)
```

For large-scale processing, use chunked reads: `pd.read_csv(..., chunksize=10_000)`.
The library equivalent is `lidarwater.io.read_pielach_txt(...)`, which does
exactly this (chunked, plus the `_riegl.reflectance` column rename).

## Code Review Standards

After completing any implementation, review the code for:
- Functions longer than 30 lines (likely doing too much — split it)
- Logic duplicated 3+ times (extract to a named helper)
- Magic numbers not assigned to a named constant
- Bare `except` or `except Exception` without re-raise or logging
- Python loops where vectorized numpy/pandas ops would work

Run /simplify before presenting code to the user.

### Python Style Rules

**Names**: `snake_case` vars/functions, `PascalCase` classes, `UPPER_CASE` module-level constants.

**Functions**: One job each. If you need "and" to describe it, split it. Max ~30 lines. Type-hint signatures:
```python
def compute_surface(grid: np.ndarray, sigma: float) -> np.ndarray:
```

**Imports**: stdlib → third-party → local. No wildcard imports (`from x import *`).

**Comments**: Explain *why*, not *what*. Self-explanatory code needs no comment.
```python
# BAD: increment counter
count += 1

# GOOD: SVB always yields surface echo first — skip it
peak_idx = peaks[1]
```

**Constants**: Magic values at module top, not buried in logic.
```python
RIVERBED_Z_MAX   = 259.6  # anchor cutoff for high-conf riverbed points
WATER_Z_MARGIN   = 0.3    # z headroom above local surface → still water
```

**Error handling**: Only at system boundaries (file I/O, raw data parsing). Don't wrap internal logic in try/except.

**Data code**: Vectorize. No row-by-row loops on DataFrames.
```python
# BAD
for i, row in df.iterrows():
    df.loc[i, 'label'] = 1 if row['z'] < thr else 0
# GOOD
df['label'] = (df['z'] < thr).astype(int)
```

**Paths**: `pathlib.Path` only, never `os.path`.

**No over-engineering**: No abstract base classes, factory patterns, or config objects unless reuse is proven. Three similar lines beat a premature abstraction.
