Here is the compressed output:

---

# Project Specification: Water vs. Land Classifier from Full-Waveform Bathymetric LiDAR

**Updated 2026-04. Actual built system.**

---

## 1. Problem Statement

Classify each LiDAR point: **water** (river surface, water column, riverbed) or **land** (dry ground, gravel, vegetation). Inputs:
- Raw waveform (time + amplitude array) per point
- 3D coordinates (x, y, z)
- Scalar reflectance value

**No labeled training data.** Workflow:
1. Build physics-based waveform auto-labeler
2. Use confident riverbed detections → river geometry
3. Propagate labels geometrically via water surface model
4. Train supervised ML (XGBoost + deep model) on propagated labels

**Current status**: Pipeline complete → `pointclouds/labeled_pointcloud_v8.csv`. XGBoost CV F1=0.913, Deep val F1=0.956 on v8 labels.

---

## 2. Data Paths and Formats

```
/home/chrisvenator/PycharmProjects/LidarWaterDetection/
├── data/                          # RAW DATA (gitignored)
│   ├── point_cloud_df.txt         # 234,024 rows, ~19 MB
│   └── waveform_df.txt            # 234,024 rows, ~92 MB
├── data_processed/                # Computed, not raw
│   ├── features_v2.csv            # 234,024 × 42 feature matrix — USE THIS
│   ├── labels_v3.csv              # Early z-threshold labels (deprecated)
│   └── waveform_grids.npy         # 234024 × 200 dense amplitude grids
├── models/
│   └── v8-surface-v2/             # CURRENT BEST MODELS
│       ├── xgb.json
│       ├── deep.pt
│       └── deep_stats.json
├── pointclouds/
│   └── labeled_pointcloud_v8.csv  # CURRENT BEST OUTPUT
└── src/                           # All code
```

### 2.1 point_cloud_df.txt

CSV. Columns: `Unnamed: 0` (index), `x`, `y`, `z`, `_riegl.reflectance`

```python
pc = pd.read_csv('data/point_cloud_df.txt')
```

Coordinate system: ETRS89/UTM 33N, EPSG:25833, local-offset residuals:
- x: −269.4 to −199.2 m
- y: 97.2 to 138.4 m
- z: 256.5 to 278.5 m (absolute elevation above geoid)
- reflectance: −30.8 to −6.3 dB (mean −21.6, std 5.0)

### 2.2 waveform_df.txt

CSV. Columns: `Unnamed: 0`, `Time [SI]`, `Amplitude [ADC]`

**Numpy array string representations.** Parse with regex:

```python
import re
import numpy as np

def parse_array_string(s):
    nums = re.findall(r'[-+]?\d+', str(s))
    return np.array([int(x) for x in nums], dtype=np.int32)

wf = pd.read_csv('data/waveform_df.txt')
times = parse_array_string(wf['Time [SI]'].iloc[i])
amps  = parse_array_string(wf['Amplitude [ADC]'].iloc[i])
```

**Units**: Time in SI (1 SI ≈ 0.5 ns); Amplitude in ADC (12-bit, range 0–8191).

**Row alignment**: `point_cloud_df.txt[i]` ↔ `waveform_df.txt[i]` exactly.

**Non-contiguous time arrays (CRITICAL)**: NOT consecutive. Scanner records windows around detected signal only:
```
Times: [37, 38, ..., 100,   ← cluster 1 (surface returns)
        173, 174, ..., 189]  ← cluster 2 (bottom returns)
                ^--- 73 SI gap (water + air column)
```

### 2.3 features_v2.csv (42 columns)

Pre-computed from `src/features/feature_extractor.py` + `src/features/add_features.py`.

Key model columns:
```
x, y, z, reflectance_dB
energy_concentration, max_amp_norm_by_energy, n_clusters, n_peaks, n_gaps,
n_samples, time_span, max_amp, mean_amp, std_amp, total_energy,
max_gap, mean_gap, total_gap, first_last_span, energy_ratio_late,
first_peak_amp, last_peak_amp, peak_amp_ratio, depth_proxy_m,
amplitude_weighted_center, active_bins_ratio,
height_above_local_min, height_above_local_min_10m, height_percentile_local,
planarity, roughness, linearity, sphericity,
height_range_local, height_std_local, z_relative, ...
```

### 2.4 waveform_grids.npy

Shape: (234024, 200), dtype=float32. Dense amplitude grid/point, origin-relative (first sample → bin 0). V8Net waveform branch input.

---

## 3. Physics

### Why green laser (532 nm) for bathymetry

Water min optical attenuation at 460–550 nm. 532 nm pulse:
1. Partially reflects off **water surface** (specular)
2. Travels through **water column** (exp. backscatter)
3. Reflects off **riverbed** (high-amp gravel return)

SVB decomposition: `PR(t) = PWS(t) + PWC(t) + PWB(t)` — surface + volume + bottom.

### Waveform signatures by type (measured, not theoretical)

| Surface | energy_concentration | n_peaks | n_gaps | Notes |
|---------|----------------------|---------|--------|-------|
| Water (riverbed) | HIGH (>0.85) | 1–2 | 1–2 | Compact early dominant peak |
| Dry gravel | LOWER | 2–4 | 2–4 | Multi-scatter from grains |
| Canopy | LOW | 3–5 | 1–3 | Multiple close returns |

**Counter-intuitive**: Water waveforms SIMPLER than dry gravel. Initial assumption (water = complex SVB multi-return) wrong. Strongest predictors: `energy_concentration`, `max_amp_norm_by_energy`, not `n_gaps`.

### Elevation zones (Pielach River, October 2024)

| Zone | z range | Description |
|------|---------|-------------|
| Canopy | > 263.3 m | Riparian vegetation |
| Dry bank/meadow | 262–263.3 m | Gravel bars, grass |
| Transition | 260.5–262 m | Uncertain — shallow margins |
| Water surface | 259.6–260.5 m | River surface returns |
| Riverbed | < 259.6 m | Gravel bottom (confident anchor points) |

River gradient: ~0.21 m/100 m over 70 m scan extent.

### Refraction and lateral shift

SVB bottom returns **laterally displaced** from surface above (Snell's law, n=1.333). Vertical propagation scheme ("if bottom at (x,y), water column at (x,y)") unreliable for surface-return points. 2D footprint approach sidesteps: work in horizontal plane.

### Key physics numbers

- 1 SI = 0.5 ns → 1 SI peak separation ≈ 5.6 cm water depth: `depth_m = Δ_SI × 0.05625`
- Min separable depth: ~20 cm (VQ-840-GL) — shallower: surface+bottom merge
- Beam footprint at 60 m AGL: ~6 cm diameter
- Scan area: Pielach River, ~20 m wide, coarse gravel bed

---

## 4. Active Pipeline

Run in order from repo root, `.venv` active:

```bash
source .venv/bin/activate

# Step 1 — Waveform-only auto-labeler (produces riverbed anchors)
python src/labeling/auto_labeler_v6.py
# Output: pointclouds/labeled_pointcloud_v6_waveform_only.csv

# Step 2 — Adaptive water surface model (current best)
python src/labeling/water_surface_model_v2.py
# Output: pointclouds/labeled_pointcloud_v8.csv
#         models/v8-surface-v2/{xgb.json, deep.pt, deep_stats.json}
#         models/v8-surface-v2/{topdown_scatter,surface_grid,crosssection}.png
```

v8 script: label generation + model training + inference + export in one pass.

### Legacy pipeline (v4 staged cascade)

```bash
python src/labeling/auto_labeler_v3.py      # z-threshold labels
python src/training/train_stage1.py          # canopy vs ground XGBoost
python src/training/train_stage2.py          # water vs dry XGBoost + CNN
python src/inference/inference_pipeline.py   # → labeled_pointcloud_v4_staged.csv
```

v4 superseded by v8. Don't use v4 labels — z-gated, models overfit to elevation.

---

## 5. Code Structure

```
src/
├── labeling/
│   ├── auto_labeler_v6.py          ← waveform-only physics labeler
│   ├── water_surface_model.py      ← v7: river footprint + RANSAC plane (superseded)
│   ├── water_surface_model_v2.py   ← v8: adaptive grid (CURRENT BEST)
│   ├── spatial_propagation.py      ← v7 alt: vertical propagation attempt (superseded)
│   ├── auto_labeler_v3.py          ← z-threshold labeler (deprecated)
│   └── auto_labeler.py             ← v1 (deprecated)
├── features/
│   ├── feature_extractor.py        ← original: x,y,z + waveform scalars + geometry
│   └── add_features.py             ← adds 7 generalizable features → features_v2.csv
├── training/
│   ├── train_stage1.py             ← v4: canopy vs ground XGBoost
│   ├── train_stage2.py             ← v4: water vs dry XGBoost + Stage2Net CNN
│   ├── train_v2.py                 ← v2: extended features (superseded)
│   ├── train_v3.py                 ← v3: no height_above features (superseded)
│   ├── baseline_model.py           ← v1 (deprecated)
│   ├── deep_model.py               ← v1 WaveformNet (deprecated)
│   └── train_deep.py               ← v1 training loop (deprecated)
├── evaluation/
│   ├── evaluate.py
│   ├── compare_models.py
│   └── export_predictions.py
├── diagnostics/
│   ├── diagnostic_water_vs_veg.py
│   ├── diagnostic_water_vs_dryground.py
│   └── inspect_zones.py
└── inference/
    └── inference_pipeline.py       ← v4 inference on 234k points
```

---

## 6. Model Architecture

### V8Net (current, n_spatial=32)

```
Waveform branch  (1D CNN on 200-bin grid):
  Conv1D(1→32, k=3)  + BN + ReLU  [padding='same']
  Conv1D(32→64, k=5) + BN + ReLU
  Conv1D(64→64, k=11)+ BN + ReLU
  MaxPool1D(4)
  Conv1D(64→128, k=5)+ BN + ReLU
  AdaptiveAvgPool1D(1) → (B, 128)

Spatial branch  (MLP on 32 scalar features):
  Linear(32→128) + BN + ReLU
  Linear(128→64) + BN + ReLU
  Linear(64→32)  + ReLU        → (B, 32)

Fusion:
  Concat → (B, 160)
  Linear(160→128) + BN + ReLU
  Linear(128→64)  + ReLU
  Linear(64→2)    → logits → Softmax
```

**No Dropout.** `Stage2Net` in `inference_pipeline.py` architecturally identical.

Loss: Focal loss with label smoothing (`gamma=2.0`, `alpha=0.65`, `smoothing=0.05`).

Optimizer: AdamW (`lr=1e-3`, `weight_decay=1e-4`) + CosineAnnealingLR (`T_max=60`).

---

## 7. Feature Sets

### ALL_FEATURES (32 total, used for both XGBoost and V8Net spatial branch)

```python
WAVEFORM_FEATURES = [
    "energy_concentration",         # KEY discriminator (water=high, gravel=low)
    "max_amp_norm_by_energy",       # KEY discriminator
    "n_clusters", "n_peaks", "n_gaps", "n_samples", "time_span",
    "max_amp", "mean_amp", "std_amp", "total_energy",
    "max_gap", "mean_gap", "total_gap", "first_last_span",
    "energy_ratio_late",
    "first_peak_amp", "last_peak_amp", "peak_amp_ratio",
    "depth_proxy_m", "amplitude_weighted_center", "active_bins_ratio",
    "reflectance_dB",
]  # 23 features

RELATIVE_FEATURES = [
    "height_above_local_min",       # ≈0 for water (river = valley floor)
    "height_percentile_local",
    "planarity", "roughness", "linearity", "sphericity",
    "height_range_local", "height_std_local",
    "z_relative",                   # z - local_mean_z
]  # 9 features
```

**Do NOT use absolute z** (`z` column) as model input. Overfits to Pielach elevation, fails other sites.

`height_above_local_min_10m` excluded: narrow valley, river provides 10 m-radius min for bank points too — meaningless local height indicator.

---

## 8. Output Files

### labeled_pointcloud_v8.csv (current best)

Columns:
```
x, y, z, reflectance_dB    — point coordinates
xgb_pred, xgb_proba        — XGBoost prediction (0=land, 1=water) + probability
deep_pred, deep_proba       — V8Net prediction + probability
ensemble                   — 0=land, 1=water, 2=uncertain (models disagree)
[scalar fields for CloudCompare: n_gaps, max_gap, n_peaks, energy_concentration, ...]
```

**Open in CloudCompare**: Colour by `ensemble`.
- 0 = land (grey/brown)
- 1 = water (blue)
- 2 = uncertain (yellow)

### labeled_pointcloud_v6_waveform_only.csv (intermediate — auto-labeler output)

Columns: `x, y, z, reflectance_dB, zone, true_label, xgb_pred, xgb_proba, deep_pred, deep_proba, ensemble, n_clusters, n_peaks, n_gaps, energy_concentration, max_amp_norm_by_energy, amplitude_weighted_center`

- ensemble: 0=land, 1=water, 2=uncertain

---

## 9. Evaluation

No ground-truth labels. Use proxy checks:

| Check | Expected |
|-------|----------|
| Spatial shape | Connected river ribbon ~15–25 m wide |
| Reflectance split | Mean reflectance(water) < Mean reflectance(land) by ≥ 3 dB |
| energy_concentration | Water cluster > land |
| z-distribution | Water concentrated at z=259–261 m |
| Footprint overlap | Water points inside/near 974 m² river polygon |
| Manual spot-check | Inspect waveforms of true/false positives |

**Spatial cross-validation** (XGBoost): 5 folds by y-strip. Best F1=0.982, worst F1=0.810. Variance from uneven class distribution (upstream vs. downstream).

---

## 10. Critical Gotchas

### Waveform physics

- **Water waveforms COMPACT, not complex.** SVB bottom-return minority of water points. Most water: high `energy_concentration` + `max_amp_norm_by_energy`, not `n_gaps`.
- **n_gaps ≥ 3 ≠ water.** v1 labeler mistake. Dry gravel has MORE gaps than water.

### Footprint construction

- Anchors: `z < 259.6 m` AND `ensemble_conf > 0.8` only. Using all confident water (incl. z=262–268 m canopy FPs) → footprint too large (66% of points, incl. meadows).
- Erode concave hull inward 0.5 m. Raw hull boundary uncertain.
- `water-like waveform` surface grid criterion needs `z ∈ [259.0, 261.5]`. Without it, canopy points with compact waveforms included → p95 = 274 m not 260 m.

### Feature engineering

- **No absolute z as model feature.** Not portable.
- `height_above_local_min` NOW safe (v8 labels not z-gated).
- `height_above_local_min_10m` NOT canopy detector in this valley — exclude.

### Model architecture consistency

`train_stage2.py`/`water_surface_model_v2.py` and `inference_pipeline.py` must define **identical** Stage2Net/V8Net architectures. No Dropout. Mismatched change → `load_state_dict()` failure.

### Data parsing

- Waveform arrays are numpy-format strings `[ 37  38  39]` — parse with `re.findall(r'[-+]?\d+', s)`, NOT `ast.literal_eval()`.
- Multiple rows can share one waveform (same pulse, multiple echoes).
- x, y are local residuals, NOT full UTM. Use relative for spatial computation.

---

## 11. Package Requirements

```
pandas>=1.5
numpy>=1.23
scipy>=1.9
scikit-learn>=1.2
xgboost>=1.7
torch>=2.0
shapely>=2.0        # for concave_hull and spatial operations
tqdm
matplotlib
```

Install:
```bash
source .venv/bin/activate
pip install pandas numpy scipy scikit-learn xgboost torch tqdm matplotlib shapely
```

Python 3.12.3, virtualenv at `.venv/` with `include-system-site-packages = true`.

---

## 12. Key Source Papers

| Paper | Importance |
|-------|------------|
| Mandlburger et al. 2025 — Pielach River showcase (DOI:10.23784/HN130-06) | THE dataset paper |
| Open benchmark dataset (DOI:10.48436/taz19-r6618) | Source data |
| Schwarz et al. 2019 — SVB algorithm | SVB decomposition theory |
| Wagner et al. 2004 — Gaussian decomposition | OWP basis |
| Pfennigbauer et al. 2014 — OWP | Online waveform processing |

For complete domain knowledge: `context/knowledge_base.md`  
For architecture details: `context/architecture_recommendation.md`