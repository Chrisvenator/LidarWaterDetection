# Project Specification: Water vs. Land Classifier from Full-Waveform Bathymetric LiDAR

**Self-contained build specification. Zero prior context required.**

---

## 1. Problem Statement

Classify each LiDAR point as **water** (river/lake surface, water column, or riverbed) or **land** (dry ground, gravel, vegetation, structures) using only:
- A raw waveform (time + amplitude array) per point
- 3D coordinates (x, y, z)
- A scalar reflectance value

**There are no labeled training data.** The workflow is:
1. Build a rule-based auto-labeler from physics
2. Generate noisy pseudo-labels (~60–70% of points get confident labels)
3. Train supervised models on pseudo-labels
4. Iterate to improve labels

---

## 2. Data Paths and Formats

```
/home/chrisvenator/PycharmProjects/LidarWaterDetection/
├── data/
│   ├── point_cloud_df.txt    # 234,024 rows, ~19 MB
│   └── waveform_df.txt       # 234,024 rows, ~92 MB
├── context/                  # knowledge base and specs
└── .venv/                    # Python 3.12 virtualenv
```

### 2.1 point_cloud_df.txt

CSV file. Columns: `Unnamed: 0` (row index), `x`, `y`, `z`, `_riegl.reflectance`

```python
import pandas as pd
pc = pd.read_csv('data/point_cloud_df.txt')
# pc.columns = ['Unnamed: 0', 'x', 'y', 'z', '_riegl.reflectance']
```

Coordinate system: ETRS89/UTM 33N, EPSG:25833  
Local offsets (these are NOT the full UTM coordinates — they are residuals after subtracting a reference):
- x range: -269.4 to -199.2 m
- y range: 97.2 to 138.4 m
- z range: 256.5 to 278.5 m (absolute elevation above geoid)
- reflectance range: -30.8 to -6.3 dB (mean -21.6, std 5.0)

### 2.2 waveform_df.txt

CSV file. Columns: `Unnamed: 0` (row index), `Time [SI]`, `Amplitude [ADC]`

**Critical**: The time and amplitude values are stored as **numpy array string representations**, not Python list strings. Use regex parsing:

```python
import re
import numpy as np

def parse_array_string(s):
    """Parse '[ 37  38  39  40 ...]' or '[37, 38, 39, ...]' to numpy array."""
    nums = re.findall(r'[-+]?\d+', str(s))
    return np.array([int(x) for x in nums], dtype=np.int32)

wf = pd.read_csv('data/waveform_df.txt')
# For a single row:
times = parse_array_string(wf['Time [SI]'].iloc[i])
amps  = parse_array_string(wf['Amplitude [ADC]'].iloc[i])
```

**Units**:
- Time: sample intervals (SI), where **1 SI ≈ 0.5 ns**
- Amplitude: ADC units, 12-bit range (0–8191)

**Row alignment**: `point_cloud_df.txt` row `i` corresponds exactly to `waveform_df.txt` row `i`.  
**Waveform sharing**: Multiple point rows can share the same waveform (one laser pulse extracted multiple echoes). Check with `wf['Time [SI]'].iloc[i] == wf['Time [SI]'].iloc[i-1]`.

**Non-contiguous time arrays (CRITICAL GOTCHA)**: Time values are NOT consecutive integers. The scanner records only windows around detected signal. A typical waveform looks like:
```
Times: [37, 38, ..., 98, 99, 100,  ← cluster 1 (surface/near returns)
        173, 174, 175, 176, ...]    ← cluster 2 (bottom/far returns)
                  ^--- gap of ~73 SI units (not recorded)
```
The **gap encodes the air travel time** from the sensor to the surface and back. Do NOT assume consecutive samples when processing.

### 2.3 Observed waveform statistics (from dataset analysis)

| Metric | Value |
|--------|-------|
| Waveform lengths | 19–101 samples (mean 59.5) |
| Max amplitude | 257–3,962 ADC (mean 2,369) |
| ADC system range | 0–8,191 (12-bit) |
| 0 time-gaps | 0.1% of waveforms |
| 1 time-gap | 3.4% |
| 2 time-gaps | 15.6% |
| 3+ time-gaps | **80.9%** |

The majority of waveforms have 3+ gaps, with the dominant pattern being a large terminal gap of **~70–83 SI units** (~35–41 ns) between early surface returns and a final return cluster. This is characteristic of the sensor height-to-ground travel time with a near-surface water body.

---

## 3. Physics Cheat Sheet

### Why green laser (532 nm) is special for water

Water has minimum optical attenuation in the 460–550 nm range. A 532 nm pulse therefore:
1. Partially reflects off the **water surface** (specular reflection)
2. Travels through the **water column** (exponentially attenuated backscatter)
3. Reflects off the **riverbed** (gravel → high reflectance)

This gives a bathymetric waveform with up to three components:
```
PR(t) = PWS(t) + PWC(t) + PWB(t)
         surface  volume   bottom
```
where PWC(t) ∝ exp(-2·k·z / cos(α_water)) — exponential decay with depth z and attenuation coefficient k.

The **NIR laser (905 nm)** is absorbed by water — it sees only the surface.

### Waveform signatures by surface type

| Surface | Peaks | Shape | Notes |
|---------|-------|-------|-------|
| Water surface | 1 | Gaussian, moderate amplitude | Specular → weak at off-nadir |
| Water column | continuous | Exponential decay after surface peak | Not a discrete peak |
| Riverbed (gravel) | 1 | Broad Gaussian, weaker than surface | Appears as second cluster in gap waveforms |
| Dry gravel/land | 1 | Narrow, high amplitude | Clean return |
| Vegetation | 3–5+ | Multiple closely spaced, varying amp | Short inter-peak gaps |
| Buildings | 1 | Sharp, very high amplitude | |

### Key physics numbers

- Speed of light in water: **225,000,000 m/s** (0.225 m/ns)
- Speed of light in air: 300,000,000 m/s
- 1 SI unit = 0.5 ns → **1 SI unit of peak separation ≈ 5.6 cm water depth** (one-way)
  - `depth_m = delta_SI * 0.5e-9 * 225e6 / 2 = delta_SI * 0.05625 m`
- Minimum separable depth: **~20 cm** (VQ-840-GL) — below this, surface and bottom echoes overlap
- Snell's law at air–water interface: n_air = 1.0, n_water = 1.333 → refraction angle changes ~0.75× at surface
- Beam footprint at 60 m AGL with 1 mrad divergence: **~6 cm diameter**
- Pielach River: ~3 m max depth, ~20 m wide, coarse gravel bed, October 2024

---

## 4. Auto-Labeling Rules (Ready to Implement)

### 4.1 Waveform feature extraction

```python
def extract_features(times, amps, min_peak_amp=100, gap_threshold=2):
    """
    Extract scalar features from a single waveform.
    times: np.array of SI time values (may be non-contiguous)
    amps:  np.array of ADC amplitude values
    """
    f = {}

    # Basic
    f['max_amp']      = int(np.max(amps))
    f['mean_amp']     = float(np.mean(amps))
    f['total_energy'] = int(np.sum(amps))
    f['n_samples']    = len(times)
    f['time_span']    = int(times[-1] - times[0]) if len(times) > 1 else 0

    # Gap analysis — KEY for water detection
    if len(times) > 1:
        diffs = np.diff(times)
        gaps  = diffs[diffs > gap_threshold]
        f['n_gaps']    = int(np.sum(diffs > gap_threshold))
        f['max_gap']   = int(np.max(diffs))
        f['total_gap'] = int(np.sum(gaps))
    else:
        f['n_gaps'] = f['max_gap'] = f['total_gap'] = 0

    # Peak detection
    peaks = [i for i in range(1, len(amps) - 1)
             if amps[i] > amps[i-1] and amps[i] > amps[i+1] and amps[i] >= min_peak_amp]
    f['n_peaks'] = len(peaks)

    if len(peaks) >= 2:
        peak_times = times[peaks]
        spacings   = np.diff(peak_times)
        f['max_peak_spacing']  = int(np.max(spacings))
        f['mean_peak_spacing'] = float(np.mean(spacings))
        f['first_last_span']   = int(peak_times[-1] - peak_times[0])
    else:
        f['max_peak_spacing'] = f['mean_peak_spacing'] = f['first_last_span'] = 0

    # Cluster structure (groups of consecutive samples separated by large gaps)
    f['n_clusters'] = 1 + f['n_gaps']  # approximate

    # Energy ratio: fraction in last 40% of time span (water column/bottom contribution)
    if f['time_span'] > 0:
        cutoff = times[0] + 0.6 * f['time_span']
        early = np.sum(amps[times <= cutoff])
        late  = np.sum(amps[times > cutoff])
        f['energy_ratio_late'] = float(late / (early + late + 1e-6))
    else:
        f['energy_ratio_late'] = 0.0

    return f
```

### 4.2 The SVB waveform signature check

```python
def is_svb_water_signature(times, amps, gap_threshold=50, min_cluster_peak=100):
    """
    Check for the Surface-Volume-Bottom signature:
      Cluster 1 (surface, SI ~37-100) + large gap (>50 SI) + Cluster 2 (bottom, SI ~150-200)
    This is HIGH CONFIDENCE water.
    """
    if len(times) < 4:
        return False
    diffs = np.diff(times)
    large_gap_idx = np.where(diffs >= gap_threshold)[0]
    if len(large_gap_idx) == 0:
        return False
    # There is at least one large gap
    split = large_gap_idx[0] + 1  # index of first sample after the large gap
    cluster1_amps = amps[:split]
    cluster2_amps = amps[split:]
    # Both clusters must have at least one meaningful amplitude
    return (np.max(cluster1_amps) >= min_cluster_peak and
            np.max(cluster2_amps) >= min_cluster_peak)
```

### 4.3 Confidence scoring function

```python
def compute_water_confidence(wf_features, reflectance_dB, z, svb_flag):
    """
    Compute water probability score [0, 1] for a single point.
    
    Returns:
        score: float in [0, 1]
        0.7+ → label WATER
        0.3- → label LAND
        0.3–0.7 → UNCERTAIN (exclude from training)
    """
    score = 0.5  # neutral prior

    # === STRONG WATER EVIDENCE (+) ===

    # SVB signature (surface cluster + large gap + bottom cluster)
    if svb_flag:
        score = max(score, 0.85)   # high-confidence water

    # Large temporal gap between sample clusters (>50 SI ≈ air+water column travel)
    if wf_features['max_gap'] >= 50:
        score += 0.20
    elif wf_features['max_gap'] >= 30:
        score += 0.10

    # Many gaps (complex multi-return structure)
    if wf_features['n_gaps'] >= 3:
        score += 0.10
    elif wf_features['n_gaps'] >= 2:
        score += 0.05

    # Significant peak separation (implies water depth)
    if wf_features['max_peak_spacing'] >= 15:   # ≥ 15 SI ≈ 0.84 m water depth
        score += 0.15
    elif wf_features['max_peak_spacing'] >= 8:   # ≥ 8 SI ≈ 0.45 m water depth
        score += 0.08

    # Low reflectance (water has low reflectance at 532 nm)
    if reflectance_dB < -25:
        score += 0.10
    elif reflectance_dB < -22:
        score += 0.03

    # Late energy (water column backscatter extends signal in time)
    if wf_features['energy_ratio_late'] > 0.35:
        score += 0.08

    # === STRONG LAND EVIDENCE (-) ===

    # Single clean echo (solid flat surface)
    if wf_features['n_peaks'] == 1 and wf_features['max_gap'] < 5:
        score -= 0.25

    # High reflectance (dry gravel/meadow)
    if reflectance_dB > -15:
        score -= 0.10
    elif reflectance_dB > -18:
        score -= 0.05

    # Very high amplitude (diffuse hard surface return)
    if wf_features['max_amp'] > 3500:
        score -= 0.08
    elif wf_features['max_amp'] > 3000:
        score -= 0.04

    # === VEGETATION REDUCTION (ambiguous — not confidently water) ===
    if (wf_features['n_peaks'] >= 3 and
            wf_features['max_peak_spacing'] < 15 and
            wf_features['time_span'] < 50):
        score -= 0.12  # closely spaced multi-peak = vegetation canopy

    return float(np.clip(score, 0.0, 1.0))
```

### 4.4 Full auto-labeling pipeline

```python
def auto_label_dataset(pc_path='data/point_cloud_df.txt',
                        wf_path='data/waveform_df.txt',
                        water_thresh=0.70,
                        land_thresh=0.30,
                        chunksize=5000):
    """
    Returns a DataFrame with columns: [index, label, confidence]
    label: 1=water, 0=land, -1=uncertain
    """
    import pandas as pd
    import numpy as np
    import re

    def parse_arr(s):
        return np.array([int(x) for x in re.findall(r'[-+]?\d+', str(s))], dtype=np.int32)

    results = []

    pc  = pd.read_csv(pc_path)
    wf  = pd.read_csv(wf_path)

    for i in range(len(pc)):
        row_pc = pc.iloc[i]
        row_wf = wf.iloc[i]

        times = parse_arr(row_wf['Time [SI]'])
        amps  = parse_arr(row_wf['Amplitude [ADC]'])

        if len(times) == 0 or len(amps) == 0:
            results.append({'index': i, 'label': -1, 'confidence': 0.5})
            continue

        feat    = extract_features(times, amps)
        svb     = is_svb_water_signature(times, amps)
        refl_dB = float(row_pc['_riegl.reflectance'])
        z       = float(row_pc['z'])

        conf = compute_water_confidence(feat, refl_dB, z, svb)

        if conf >= water_thresh:
            label = 1
        elif conf <= land_thresh:
            label = 0
        else:
            label = -1  # uncertain

        results.append({'index': i, 'label': label, 'confidence': conf,
                         **feat, 'reflectance_dB': refl_dB, 'z': z})

    return pd.DataFrame(results)
```

**Expected output**: ~60–70% of 234,024 points receive confident labels.  
- WATER labels: predominantly points with SVB signature, high n_gaps, low reflectance
- LAND labels: predominantly single-echo, high reflectance, high amplitude
- UNCERTAIN: edge cases (shallow water, wet soil, semi-submerged vegetation)

---

## 5. Model Architecture

### Option A: MVP (build first — 1–2 days)

XGBoost on ~20 scalar features. No GPU needed.

```python
import xgboost as xgb

features = [
    'max_amp', 'n_gaps', 'max_gap', 'time_span', 'n_peaks',
    'max_peak_spacing', 'total_energy', 'energy_ratio_late',
    'reflectance_dB', 'z', 'n_clusters', 'first_last_span'
]

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=3.0,  # adjust for class imbalance: land:water ratio
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
# Input:  (N_confident, len(features))
# Output: (N, 2) probabilities
```

Expected accuracy on confident labels: **~80–85%**.

### Option B: Ideal Model (1–2 weeks)

Multi-modal 1D CNN + MLP. PyTorch.

```
Waveform branch:    (N, 200) dense grid
  Conv1D(1→32, k=3)  + BN + ReLU
  Conv1D(32→64, k=5) + BN + ReLU
  Conv1D(64→64, k=11)+ BN + ReLU
  MaxPool1D(4)
  Conv1D(64→128, k=5)+ BN + ReLU
  AdaptiveAvgPool1D(1) → flatten → (N, 128)

Spatial branch:     (N, 8) scalar features
  [x_rel, y_rel, z, reflectance, planarity, roughness, height_range, height_std]
  Linear(8→64)  + BN + ReLU
  Linear(64→64) + BN + ReLU
  Linear(64→32) + ReLU → (N, 32)

Fusion head:        (N, 128+32=160)
  Linear(160→128) + BN + ReLU + Dropout(0.3)
  Linear(128→64)  + ReLU + Dropout(0.2)
  Linear(64→2)    → logits → Softmax → [P(land), P(water)]
```

**Waveform → dense grid conversion**:

```python
def waveform_to_grid(times, amps, grid_size=200, noise_floor=0.0):
    """
    Project non-contiguous waveform to fixed-length dense 1D array.
    Uses origin-relative time (t_min -> bin 0).
    """
    grid = np.full(grid_size, noise_floor, dtype=np.float32)
    t_min = int(times[0])
    for t, a in zip(times, amps):
        idx = int(t) - t_min
        if 0 <= idx < grid_size:
            grid[idx] = float(a)
    return grid  # shape (200,)
```

**Loss**: Focal loss with label smoothing (handles class imbalance + noisy labels):
```python
# FocalLossWithLabelSmoothing(gamma=2.0, alpha=0.75, smoothing=0.1)
# gamma=2.0: standard focal parameter
# alpha=0.75: upweights rare water class
# smoothing=0.1: labels become 0.1/0.9 to handle noisy auto-labels
```

**Training hyperparameters**:
```
batch_size = 512
optimizer  = AdamW(lr=1e-3, weight_decay=1e-4)
scheduler  = CosineAnnealingLR(T_max=100, eta_min=1e-5)
epochs     = 100
early_stop = patience=15 on validation focal loss
grad_clip  = 1.0
```

Expected accuracy: **~88–93%** on confident labels.

---

## 6. Code Structure

```
LidarWaterDetection/
├── auto_labeler.py        # Rules-based labeler; outputs labels.csv
├── feature_extractor.py   # Waveform + geometric feature extraction; outputs features.csv
├── baseline_model.py      # XGBoost training, evaluation, feature importance
├── deep_model.py          # PyTorch model definition (WaveformNet class)
├── train.py               # Training loop for deep model
├── evaluate.py            # Metrics, confusion matrix, spatial validation
└── data/
    ├── point_cloud_df.txt
    └── waveform_df.txt
```

### auto_labeler.py responsibilities
- Load both data files
- Call `extract_features()` per waveform
- Call `is_svb_water_signature()` per waveform
- Call `compute_water_confidence()` per point
- Threshold to {water=1, land=0, uncertain=-1}
- Save `labels.csv` with columns: index, label, confidence, all waveform features

### feature_extractor.py responsibilities
- All waveform scalar features (output of `extract_features()`)
- Geometric features: planarity, roughness, height range via k-NN PCA (k=20, radius=0.5 m)
- Dense waveform grid (200 bins) for each point → saved as NPY or HDF5
- Normalize: z-score scalar features; keep dB and SI values as-is for inspection

### baseline_model.py responsibilities
- Load features.csv + labels.csv (confident only)
- Spatial cross-validation: split by y-coordinate strips (not random) to avoid spatial leakage
- XGBoost training with scale_pos_weight for imbalance
- Output: feature importance plot, confusion matrix, classification report

### deep_model.py responsibilities
- `WaveformDataset(torch.utils.data.Dataset)`: loads dense grids + spatial features
- `WaveformNet(torch.nn.Module)`: 1D CNN + MLP architecture as specified
- `FocalLossWithLabelSmoothing`: loss function

### train.py responsibilities
- Parse arguments (model type, hyperparameters, paths)
- Load dataset, split train/val spatially
- Training loop with early stopping
- Save best checkpoint

### evaluate.py responsibilities
- Load model checkpoint
- Generate predictions for all 234,024 points
- Physical plausibility checks:
  - Do water points form spatially connected river-shaped regions?
  - Do water points have lower mean reflectance than land points?
  - Do water points show higher n_gaps than land points?
- Export labeled point cloud as CSV for visualization in CloudCompare or similar

---

## 7. Training Pipeline — Step by Step

```bash
# 0. Activate environment
cd /home/chrisvenator/PycharmProjects/LidarWaterDetection
source .venv/bin/activate
pip install xgboost scikit-learn pandas numpy scipy torch tqdm

# 1. Generate auto-labels
python auto_labeler.py \
    --pc data/point_cloud_df.txt \
    --wf data/waveform_df.txt \
    --out labels.csv \
    --water-thresh 0.70 \
    --land-thresh 0.30

# 2. Extract features (CPU-only; ~30 min for 234k points with k-NN)
python feature_extractor.py \
    --pc data/point_cloud_df.txt \
    --wf data/waveform_df.txt \
    --out features.csv \
    --grid-out waveform_grids.npy  # (234024, 200) float32 array

# 3. Train baseline (MVP)
python baseline_model.py \
    --features features.csv \
    --labels labels.csv \
    --out-model models/xgb_baseline.json

# 4. Train deep model
python train.py \
    --features features.csv \
    --grids waveform_grids.npy \
    --labels labels.csv \
    --model-out models/deep_model_best.pt \
    --epochs 100 \
    --batch-size 512

# 5. Evaluate + export labeled point cloud
python evaluate.py \
    --pc data/point_cloud_df.txt \
    --features features.csv \
    --grids waveform_grids.npy \
    --model models/deep_model_best.pt \
    --out labeled_pointcloud.csv
```

---

## 8. Evaluation (Without Ground Truth)

Since there are no labels, use these proxy evaluations:

| Check | Expected if correct |
|-------|---------------------|
| Spatial connectivity | Water points form elongated, connected ribbon ~20 m wide following the river |
| Reflectance split | Mean reflectance(water) < Mean reflectance(land) by ≥ 3 dB |
| n_gaps split | Mean n_gaps(water) > Mean n_gaps(land) |
| z-distribution | Water cluster at z ≈ 261–263 m (river surface); land at z ≈ 265–278 m |
| SVB points | All SVB-flagged points should be labeled water |
| Manual sample | Inspect 100–200 random predictions visually against waveform plots |

**Spatial cross-validation**: Divide the point cloud into N-S strips along the river (split by y-coordinate). Train on one strip, validate on adjacent strip. This tests generalization without spatial autocorrelation leakage.

**Reference**: Mandlburger et al. 2025 report vertical accuracy < 2 cm for the 3D geometry. Classification accuracy will be limited by label quality, not sensor precision.

---

## 9. Critical Gotchas and Edge Cases

### Data parsing
- **Non-contiguous time arrays**: Never assume `np.arange(times[0], times[-1])` covers all samples. Always use actual `times` array for indexing.
- **Waveform sharing**: Two adjacent rows in `waveform_df.txt` may have identical waveforms (same laser pulse, two extracted echoes). Use `wf['Time [SI]'].iloc[i] == wf['Time [SI]'].iloc[i-1]` to detect this.
- **Array string format**: `waveform_df.txt` stores numpy array strings like `[ 37  38  39]` (spaces, not commas). Use `re.findall(r'[-+]?\d+', str(s))` to parse, NOT `ast.literal_eval()`.
- **Index column**: Both files have an `Unnamed: 0` index column. Don't use it as a feature.
- **Coordinate offsets**: The x, y values are NOT full UTM coordinates. They are local residuals. Do not feed raw x, y as features — use `x - x.mean()`, `y - y.mean()` for local position, and keep absolute `z`.

### Physics / classification
- **Shallow water (< 20 cm)**: Surface and bottom echoes overlap → waveform looks like a single hard-surface return → will be mislabeled as land. These are ~20% of river points near edges. Accept this limitation.
- **Sun glint / calm water**: Smooth water surface at low scan angles → very strong specular first peak → may look like high-amplitude land. Palmer scan at ±20° reduces but doesn't eliminate this.
- **Wet soil / puddles**: Low reflectance, flat surface → passes water rules. Hard to distinguish without NIR channel. Expect ~5–10% of these to be mislabeled.
- **Semi-submerged vegetation (macrophytes)**: Has water column signature but bottom reflection is from plants, not gravel → complex waveform, high uncertainty.
- **Survey conditions (October 2024)**: Water transparency was good (one month after September 2024 flood). Different turbidity would change waveform structure. Rules tuned here may not transfer to turbid conditions without retuning the k-value assumptions.
- **Large terminal gap (~70–83 SI)**: This is the dominant feature in 80.9% of waveforms and primarily encodes the air-column travel time from sensor to water surface (at 60 m AGL, 60 m / (0.3 m/ns) / 2 × 2 (round trip) ≈ 400 ns → ~800 SI, but the waveform starts at SI ~37, not 0, so the sensor records only the return window). The gap between SI ~100 and ~173 (~73 SI ≈ 36 ns) corresponds to the water path component.

### Model training
- **Use spatial cross-validation**, NOT random split. LiDAR points are spatially autocorrelated — random split leaks spatial information.
- **Class imbalance**: The Pielach River is ~20 m wide in a ~70 m wide scene. Water is likely 20–35% of points. Use `scale_pos_weight` or focal loss.
- **Noisy labels**: Expect ~20–30% label noise from auto-labeler. Use confident learning (Cleanlab) or focal loss with label smoothing to handle this.
- **Iterative refinement**: After first model, use model predictions at high confidence (>0.90) to expand the training set and retrain. This typically improves F1 by 5–10 points.

---

## 10. Package Requirements

```
pandas>=1.5
numpy>=1.23
scipy>=1.9
scikit-learn>=1.2
xgboost>=1.7
torch>=2.0          # for deep model (CPU or GPU)
tqdm                # progress bars
matplotlib          # evaluation plots
```

Install:
```bash
source .venv/bin/activate
pip install pandas numpy scipy scikit-learn xgboost torch tqdm matplotlib
```

The `.venv` uses Python 3.12.3 with `include-system-site-packages = true`, so system-installed packages (numpy, pandas) are also available.

---

## 11. Key Source Papers

| Paper | DOI / Location | Importance |
|-------|---------------|------------|
| Mandlburger et al. 2025 — Pielach River showcase | DOI:10.23784/HN130-06 | **THE dataset paper** |
| Open benchmark dataset | DOI:10.48436/taz19-r6618 | Source data |
| Schwarz et al. 2019 — SVB algorithm | Referenced in above | SVB decomposition |
| Wagner et al. 2004 — Gaussian decomposition | Referenced in sources | OWP basis |
| Pfennigbauer et al. 2014 — OWP | Referenced in sources | Online processing |
| LiDAR Magazine — Mandlburger 2025 Parts 1–4 | `/home/chrisvenator/Documents/Uni/Topo/Papers/` | Tutorial series |

For complete domain knowledge, see:
- `context/knowledge_base.md` — full physics, features, algorithms
- `context/architecture_recommendation.md` — detailed architecture with code
- `context/file_inventory.md` — scored inventory of all lecture materials
