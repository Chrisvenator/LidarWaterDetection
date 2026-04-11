# Architecture: Water vs. Land Classifier — Implemented System

**Updated 2026-04. Reflects actual built/tested pipeline, not original design doc.**

---

## 1. Overview

Three-stage pipeline:

1. **Waveform-only auto-labeler** (v6) — label high-conf riverbed pts from physics
2. **Adaptive water surface model** (v8) — riverbed anchors → tight 2D footprint + spatially varying water surface; geometrically overrides uncertain waveform labels
3. **Supervised ML** — XGBoost + V8Net (1D-CNN + MLP) on propagated labels; ensemble

**No absolute z-threshold** in ML models. All spatial features relative (`z_relative`, `height_above_local_min`, `height_percentile_local`) → portable across rivers.

---

## 2. Label Generation Pipeline

### Stage A — Waveform-only auto-labeler (v6)

`src/labeling/auto_labeler_v6.py`

Physics rules: `energy_concentration`, `max_amp_norm_by_energy`, `n_peaks`, `n_gaps`, `depth_proxy_m`, `reflectance_dB`. No elevation gating.

- Label 1 (water): compact, early waveform (SVB, high energy concentration)
- Label 0 (land): complex multi-peak, high reflectance
- Label 2 (uncertain): edge cases

Output: `pointclouds/labeled_pointcloud_v6_waveform_only.csv`

**Counter-intuitive**: Water = compact/early — high `energy_concentration`, high `max_amp_norm_by_energy`. Dry gravel = more peaks/gaps (coarse surface scatter). SVB signature = strongest discriminator.

### Stage B — Adaptive water surface model (v8)

`src/labeling/water_surface_model_v2.py`

Four-step geometric override using v6 confident riverbed detections as anchors:

**Step 1 — Tight river footprint**
- Anchors: ensemble conf > 0.8 AND z < 259.6 m (riverbed only, not water surface returns)
- `shapely.concave_hull(points, ratio=0.2)` → tight polygon
- Erode inward 0.5 m: `hull.buffer(-0.5)` — removes uncertain boundary
- Result: ~974 m², 41.3% of 234k pts inside (vs. 66.5% naive)

**Step 2 — Adaptive 2m surface grid**
- Divide footprint into 2m × 2m cells
- Primary: p95 of z for water-like waveforms (`n_peaks ≤ 2`, `energy_concentration > 0.85`, `reflectance_dB < -15`, `z ∈ [259.0, 261.5]`) — min 5 pts/cell
- Fallback: global RANSAC plane for cells < 5 qualifying pts
- `scipy.ndimage.gaussian_filter(sigma=1)` smooth grid
- RANSAC plane: z ≈ -0.002148·x − 0.000183·y + 259.481 (gradient ~0.21 m/100m along flow)
- Grid: 37 × 22 cells, max Δz = 0.915 m

**Step 3 — Geometric classification inside footprint**
- Inside footprint AND z ≤ local_surface_z + 0.3 m → **WATER** (handles deep/turbid where waveform fails)
- Inside footprint AND z > local_surface_z + 0.3 m → **LAND** (exposed rock/gravel shelf)

**Step 4 — Outside footprint**
- v6 waveform prediction: water → **UNCERTAIN** (shallow margins), land → **LAND**

Results: water=43,867 (18.7%), land=148,985 (63.6%), uncertain=41,172 (17.6%)

---

## 3. V8Net Architecture (Implemented)

```python
class _CB(nn.Module):
    """Conv1D + BatchNorm + ReLU block."""
    def __init__(self, i, o, k):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv1d(i, o, k, padding='same', bias=False),
            nn.BatchNorm1d(o), nn.ReLU(inplace=True))
    def forward(self, x): return self.b(x)


class V8Net(nn.Module):
    def __init__(self, n_spatial=32):
        super().__init__()
        # Waveform branch: 1D CNN on 200-bin dense grid
        self.wf = nn.Sequential(
            _CB(1, 32, 3), _CB(32, 64, 5), _CB(64, 64, 11),
            nn.MaxPool1d(4), _CB(64, 128, 5), nn.AdaptiveAvgPool1d(1))
        # Spatial branch: MLP on 32 scalar features
        self.sp = nn.Sequential(
            nn.Linear(n_spatial, 128), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(True),
            nn.Linear(64, 32),         nn.ReLU(True))
        # Fusion head
        self.head = nn.Sequential(
            nn.Linear(160, 128), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Linear(128, 64),  nn.ReLU(True),
            nn.Linear(64, 2))

    def forward(self, wf, sp):
        # wf: (B, 1, 200) — waveform grid (channel-first for Conv1D)
        # sp: (B, 32)     — scalar features
        return self.head(torch.cat([self.wf(wf).squeeze(-1), self.sp(sp)], 1))
```

**CRITICAL**: No Dropout anywhere. `Stage2Net` in `inference_pipeline.py` uses same architecture. Any divergence causes `load_state_dict()` failures.

**Waveform input**: 200-bin dense grid, origin-relative (first sample = bin 0). Stored in `data_processed/waveform_grids.npy` (234024 × 200, float32).

**Z-score normalization**: Grid normalized globally (`grid_mean`, `grid_std`); spatial features per-column. Stats saved to `models/v8-surface-v2/stage2_deep_stats.json`.

---

## 4. Feature Set (32 features total)

```python
WAVEFORM_FEATURES = [
    # Energy structure
    "energy_concentration",        # KEY: water is compact (high), gravel is diffuse (low)
    "max_amp_norm_by_energy",      # KEY: water has high peak relative to total energy
    # Cluster structure
    "n_clusters", "n_peaks", "n_gaps",
    "n_samples", "time_span",
    # Amplitude statistics
    "max_amp", "mean_amp", "std_amp", "total_energy",
    # Gap statistics (water column traversal signals)
    "max_gap", "mean_gap", "total_gap", "first_last_span",
    # Energy distribution
    "energy_ratio_late",
    # Peak properties
    "first_peak_amp", "last_peak_amp", "peak_amp_ratio",
    # Depth proxy (delta_SI * 0.05625 m)
    "depth_proxy_m",
    # Waveform shape
    "amplitude_weighted_center", "active_bins_ratio",
    # Point cloud
    "reflectance_dB",
]

RELATIVE_FEATURES = [
    "height_above_local_min",      # ≈0 for water (sits at valley floor)
    "height_percentile_local",     # low for water, high for canopy
    "planarity", "roughness",      # from k-NN PCA
    "linearity", "sphericity",
    "height_range_local", "height_std_local",
    "z_relative",                  # z - local_mean_z
]
# Total: 23 + 9 = 32 features (n_spatial=32)
```

**Why RELATIVE_FEATURES now safe**: Previous versions excluded `height_above_local_min` — labels were z-gated (circular: low z → water → low height). V8 labels from geometric footprint + waveform physics (not z-thresholds) → relative height features valid, highly discriminative.

**Do NOT use absolute z as model feature** — prevents generalization to other rivers.

---

## 5. XGBoost Configuration

```python
xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=spw,  # computed as (n_land / n_water) from training set
    eval_metric='logloss',
    random_state=42,
    tree_method='hist',
)
```

Spatial cross-validation (5 folds by y-coordinate strips) prevents leakage from spatial autocorrelation.

---

## 6. Loss Function

**Focal loss with label smoothing** — handles class imbalance AND noisy auto-labels:

```python
class _Focal(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.65, smoothing=0.05):
        super().__init__()
        self.gamma = gamma; self.alpha = alpha; self.smoothing = smoothing

    def forward(self, logits, targets):
        n = logits.shape[-1]
        y = targets.float() * (1 - self.smoothing) + self.smoothing / n
        lp = F.log_softmax(logits, -1)
        p  = lp.exp()
        pt = (p * F.one_hot(targets, n)).sum(-1)
        w  = torch.where(targets == 1,
                         torch.full_like(pt, self.alpha),
                         torch.full_like(pt, 1 - self.alpha))
        ce = -(y * lp).sum(-1)
        return (w * (1 - pt) ** self.gamma * ce).mean()
```

Parameters: `gamma=2.0`, `alpha=0.65`, `smoothing=0.05`

---

## 7. Training Protocol

### Spatial Cross-Validation (XGBoost)

```python
# 5 folds split by y-coordinate (N-S strips along the river)
y_bins = pd.qcut(X['y_coord'], q=5, labels=False)
for fold in range(5):
    val_mask  = y_bins == fold
    train_mask = ~val_mask
    # Train and evaluate on hold-out strip
```

Avoids spatial leakage — LiDAR pts highly autocorrelated within ~1 m radius; random split leaks info.

### Deep Model Training

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-5)
epochs    = 60
batch_sz  = 512
patience  = 10  # early stopping on val focal loss
```

Train/val: 80/20 random (spatial strips for XGBoost CV, random split for deep model).

---

## 8. Ensemble Strategy

```python
# XGBoost + V8Net predict independently on the full point cloud
# Agreement  → use agreed prediction (high confidence)
# Disagreement on ground points → class 2 (uncertain)

xgb_pred  = (xgb_proba  >= 0.5).astype(int)  # 0=land, 1=water
deep_pred = (deep_proba >= 0.5).astype(int)

ensemble = xgb_pred.copy()
disagree = xgb_pred != deep_pred
ensemble[disagree] = 2   # uncertain
```

Final classes: **0 = land**, **1 = water**, **2 = uncertain**

---

## 9. Actual Metrics (v8 labels, models/v8-surface-v2/)

| Model | Metric | Value |
|-------|--------|-------|
| XGBoost | CV F1 (water) | 0.913 ± 0.084 |
| XGBoost | CV AUC | 0.990 |
| V8Net | Val F1 (water) | 0.956 |
| V8Net | Val AUC | ~0.995 |
| Ensemble | Agreement rate | 95.7% |
| Ensemble | Final water points | 41,442 (17.7%) |

Class distribution (`labeled_pointcloud_v8.csv`):
- 0 land: ~148,000 (63%)
- 1 water: ~41,442 (17.7%)
- 2 uncertain: ~44,000 (18.8%)

---

## 10. Key Physics Discoveries

### Counter-intuitive waveform physics

**Water = compact/early**:
- High `energy_concentration` (energy in first bins after t_min)
- High `max_amp_norm_by_energy` (dominant peak carries most signal)
- Low `n_peaks`, low `n_gaps` (clean SVB return)

**Dry gravel = complex**:
- Multiple peaks from grain-scale scatter
- More gaps than water
- Lower `energy_concentration`

Opposite of initial intuition. v1 labeler using `n_gaps ≥ 3` for water was almost entirely wrong.

### Refraction and the spatial shift problem

SVB physics: bottom returns **laterally displaced** from surface returns (Snell's law, n_water=1.333). Spatial propagation ("if bottom at x,y → water above") fails — surface offset 10-30 cm horizontally. 2D footprint + water surface beats vertical propagation.

### height_above_local_min as discriminator

Water at **valley floor** → `height_above_local_min` ≈ 0. Bank/gravel pts > 0. Strong physically meaningful discriminator, no absolute elevation needed.

**Exception**: `height_above_local_min_10m` (10 m radius) — river provides 10 m minimum for adjacent bank pts → canopy detector at large radius, not water detector.

### z_water_max and turbid-water problem

Deep/turbid sections → surface return only (no bottom penetration) → looks like land (single clean return). Footprint override solves: inside tight footprint + z ≤ surface_z + 0.3 m → WATER unconditionally.

---

## 11. Two-Stage Cascade vs. Flat Classifier

v4 had two-stage cascade (canopy → ground → water/dry), superseded by v8:
- Waveform physics for initial labeling (not absolute z)
- River footprint for geometric context
- Spatially adaptive water surface

v4 cascade (`models/v4-staged-cascade/`) still functional → `labeled_pointcloud_v4_staged.csv`. v8 preferred: physics-derived labels, not z-gated.

---

## 12. File Locations (Current Best)

| Artifact | Path |
|----------|------|
| Training labels | `data_processed/` (derived from `labeled_pointcloud_v6_waveform_only.csv`) |
| Feature matrix | `data_processed/features_v2.csv` (42 cols, 234,024 rows) |
| Waveform grids | `data_processed/waveform_grids.npy` (234024 × 200, float32) |
| XGBoost model | `models/v8-surface-v2/xgb.json` |
| Deep model weights | `models/v8-surface-v2/deep.pt` |
| Deep model stats | `models/v8-surface-v2/deep_stats.json` |
| Final point cloud | `pointclouds/labeled_pointcloud_v8.csv` |
| Diagnostic plots | `models/v8-surface-v2/{topdown_scatter,surface_grid,crosssection}.png` |

---

## 13. Waveform → Dense Grid (Implemented)

```python
def waveform_to_grid(times, amps, grid_size=200, noise_floor=0.0):
    """Origin-relative: first sample → bin 0."""
    grid = np.full(grid_size, noise_floor, dtype=np.float32)
    t_min = int(times[0])
    for t, a in zip(times, amps):
        idx = int(t) - t_min
        if 0 <= idx < grid_size:
            grid[idx] = float(a)
    return grid
```

Origin-relative: each waveform starts at bin 0. Removes sensor altitude dependency (absolute time origin), preserves within-waveform structure (peak spacing, gap sizes). 200-bin window covers all observed spans (max: ~163 bins).

---

## 14. Remaining Limitations

1. **Shallow water (< 20 cm)**: Surface + bottom echoes overlap → looks like single hard-surface return → classified as land. Unavoidable with SVB decomposition; affects ~20% of river edge pts.

2. **Calm water / specular reflection**: Low scan-angle, smooth water → strong first peak, no bottom return → may classify as land. Footprint override mitigates for pts inside river polygon.

3. **Survey-specific turbidity**: Waveform features tuned for October 2024 (good water clarity). Different turbidity changes optimal `energy_concentration` threshold.

4. **Generalization to other rivers**: Tight footprint requires confident riverbed detections as anchors. Turbid river with no bottom penetration → no anchors, footprint can't be built. Future: use a priori river centerline from external map.
