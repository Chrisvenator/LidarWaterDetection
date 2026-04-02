# Architecture: Water vs. Land Classifier — Implemented System

**Updated 2026-04. Reflects the actual built and tested pipeline, not the original design doc.**

---

## 1. Overview

The final system is a **three-stage pipeline**:

1. **Waveform-only auto-labeler** (v6) — labels high-confidence riverbed points from physics
2. **Adaptive water surface model** (v8) — uses riverbed anchors to define a tight 2D river footprint and a spatially varying water surface elevation; geometrically overrides uncertain waveform labels
3. **Supervised ML** — XGBoost + V8Net (1D-CNN + MLP) trained on the propagated labels; ensemble of both

The pipeline has **no absolute z-threshold feature** in the ML models. All spatial features are relative (z_relative, height_above_local_min, height_percentile_local), making the models more portable across different river scenes.

---

## 2. Label Generation Pipeline

### Stage A — Waveform-only auto-labeler (v6)

`src/labeling/auto_labeler_v6.py`

Physics rules based on `energy_concentration`, `max_amp_norm_by_energy`, `n_peaks`, `n_gaps`, `depth_proxy_m`, `reflectance_dB`. No elevation gating.

- Label 1 (water): compact, early waveform (SVB signature with high energy concentration)
- Label 0 (land): complex multi-peak, high reflectance
- Label 2 (uncertain): edge cases

Produces: `pointclouds/labeled_pointcloud_v6_waveform_only.csv`

**Key counter-intuitive finding**: Water waveforms are **compact and early** — high `energy_concentration`, high `max_amp_norm_by_energy`. Dry gravel produces **more peaks and gaps** than water (complex multi-return from coarse gravel surface). The SVB (Surface-Volume-Bottom) signature is the strongest discriminator.

### Stage B — Adaptive water surface model (v8)

`src/labeling/water_surface_model_v2.py`

Four-step geometric override using v6 confident riverbed detections as anchors:

**Step 1 — Tight river footprint**
- Anchors: ensemble confidence > 0.8 AND z < 259.6 m (actual riverbed only, not water surface returns)
- `shapely.concave_hull(points, ratio=0.2)` → very tight polygon
- Erode inward by 0.5 m: `hull.buffer(-0.5)` — removes uncertain boundary
- Result: ~974 m², 41.3% of 234k points inside (vs. 66.5% with the naive approach)

**Step 2 — Adaptive 2m surface grid**
- Divide footprint into 2m × 2m cells
- Primary: p95 of z for water-like waveforms (`n_peaks ≤ 2`, `energy_concentration > 0.85`, `reflectance_dB < -15`, `z ∈ [259.0, 261.5]`) — min 5 pts/cell required
- Fallback: global RANSAC plane for cells with < 5 qualifying points
- Apply `scipy.ndimage.gaussian_filter(sigma=1)` to smooth grid
- RANSAC plane coefficients: z ≈ -0.002148·x − 0.000183·y + 259.481 (gradient ~ 0.21 m/100m along flow)
- Grid: 37 × 22 cells, max Δz = 0.915 m across the footprint

**Step 3 — Geometric classification inside footprint**
- Inside footprint AND z ≤ local_surface_z + 0.3 m → **WATER** (unconditional — handles deep/turbid sections where waveform fails)
- Inside footprint AND z > local_surface_z + 0.3 m → **LAND** (exposed rock/gravel shelf)

**Step 4 — Outside footprint**
- Use v6 waveform prediction: water → **UNCERTAIN** (shallow margins), land → **LAND**

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

**CRITICAL**: No Dropout anywhere. The `Stage2Net` in `inference_pipeline.py` uses the same architecture. Any divergence causes `load_state_dict()` failures.

**Waveform input**: 200-bin dense grid, origin-relative (waveform shifted so first sample = bin 0). Stored in `data_processed/waveform_grids.npy` (shape: 234024 × 200, float32).

**Z-score normalization**: Grid normalized globally (`grid_mean`, `grid_std`); spatial features normalized per-column. Stats saved to `models/v8-surface-v2/stage2_deep_stats.json`.

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

**Why RELATIVE_FEATURES are now safe**: Previous versions excluded `height_above_local_min` because labels were z-gated (circular reasoning: low z → water → low height). With v8 labels coming from geometric footprint + waveform physics (not z-thresholds), these relative height features are valid and highly discriminative.

**Do NOT use absolute z as a model feature** — it prevents generalization to rivers at different elevations.

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

Training uses spatial cross-validation (5 folds split by y-coordinate strips) to prevent leakage from spatial autocorrelation.

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

This avoids spatial leakage — LiDAR points are highly autocorrelated within ~1 m radius, so a random split would leak information.

### Deep Model Training

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-5)
epochs    = 60
batch_sz  = 512
patience  = 10  # early stopping on val focal loss
```

Train/val split: 80/20 random (spatial strips used for XGBoost CV, random split for deep model training with waveform features).

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

**Class distribution in output** (`labeled_pointcloud_v8.csv`):
- 0 land: ~148,000 (63%)
- 1 water: ~41,442 (17.7%)
- 2 uncertain: ~44,000 (18.8%)

---

## 10. Key Physics Discoveries

### Counter-intuitive waveform physics

**Water waveforms are compact and early**, not complex:
- High `energy_concentration` (most energy in first few bins after t_min)
- High `max_amp_norm_by_energy` (dominant peak carries most signal)
- Low `n_peaks`, low `n_gaps` (clean SVB return, not scattered)

**Dry gravel waveforms are complex**:
- Multiple peaks from grain-scale scattering
- More gaps than water returns
- Lower `energy_concentration`

This is the opposite of initial intuition (which expected water to produce complex multi-return waveforms from water column traversal). The v1 labeler using `n_gaps ≥ 3` to detect water was almost entirely wrong.

### Refraction and the spatial shift problem

The SVB physics means bottom returns are **laterally displaced** from the surface returns above them (Snell's law, n_water=1.333). A spatial propagation approach ("if bottom detected at x,y, water must be above") fails because the surface is 10-30 cm horizontally offset from the bottom detection. This is why the 2D footprint + water surface approach works better than vertical propagation.

### height_above_local_min as discriminator

Water sits at the **valley floor** — its `height_above_local_min` ≈ 0 by definition (the river IS the local minimum). Bank and gravel bar points have positive values. This is a strong, physically meaningful discriminator that doesn't require knowing the absolute elevation.

**Exception**: `height_above_local_min_10m` uses a 10 m radius — in this narrow valley, the river provides the 10 m minimum for adjacent bank points too, making it a canopy detector at large radius but not a water detector.

### z_water_max and the turbid-water problem

Deep or turbid sections produce only a surface return (no bottom penetration). Without the waveform v6 anchor, these look like land (single clean return). The geometric footprint override solves this: inside the tight footprint + z ≤ surface_z + 0.3 m → WATER unconditionally.

---

## 11. Two-Stage Cascade vs. Flat Classifier

The v4 pipeline also implemented a two-stage cascade (canopy → ground → water vs. dry), but this was superseded by the v8 approach which:
- Uses waveform physics for initial labeling (not absolute z)
- Uses the river footprint for geometric context
- Handles the water surface with a spatially adaptive model

The v4 cascade (`models/v4-staged-cascade/`) is still functional and produces `labeled_pointcloud_v4_staged.csv`. The v8 approach is preferred because labels are physics-derived, not z-gated.

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

Origin-relative representation: each waveform starts at bin 0. This removes sensor altitude dependency (absolute time origin) while preserving within-waveform structure (peak spacing, gap sizes). The 200-bin window covers all observed waveform spans (max observed span: ~163 bins).

---

## 14. Remaining Limitations

1. **Shallow water (< 20 cm depth)**: Surface and bottom echoes overlap — waveform looks like a single hard-surface return. These are classified as land. Unavoidable with SVB decomposition; affects ~20% of river edge points.

2. **Calm water / specular reflection**: Low scan-angle, smooth water → very strong first peak with no bottom return → may classify as land. The footprint override mitigates this for points inside the river polygon.

3. **Survey-specific turbidity**: Waveform features were tuned for October 2024 conditions (good water clarity). Different turbidity changes the optimal `energy_concentration` threshold.

4. **Generalization to other rivers**: The tight-footprint approach requires confident riverbed detections as anchors. A turbid river with no bottom penetration would produce no anchors and the footprint could not be constructed. Future work: use a priori river centerline from external map data.
