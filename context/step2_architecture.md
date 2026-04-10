# Step 2 Architecture: WaveformNet v9 (Generalizable)

**Proposed 2026-04. Designed to classify any 532 nm bathymetric LiDAR point cloud,
regardless of site elevation, river width, valley shape, or point density.**

---

## 0. The Generalization Problem

The current v8 model (V8Net) achieves F1=0.956 on the Pielach dataset but relies
on features that are **survey-specific**:

| Feature group | Why it fails on a new survey |
|---|---|
| `height_above_local_min`, `height_percentile_local` | Computed from k-NN at a *specific density*. A different scanner altitude, overlap pattern, or decimation ratio changes which points are neighbors, and hence the height statistics. Also assumes river = valley floor, which fails for levee-bounded rivers. |
| `planarity`, `roughness`, `linearity`, `sphericity` | k-NN PCA eigenvalue ratios. Entirely density-dependent — the same surface looks different at 15 cm vs. 5 cm spacing. |
| `height_range_local`, `height_std_local` | Same problem: density + topology dependent. |
| `z_relative` | Absolute elevation reference. Meaningless for another site. |
| `n_samples`, `time_span`, raw `max_gap`, `total_gap` | In **scanner SI units** (1 SI = 0.5 ns for RIEGL). A different scanner with 1 ns sampling gives half the bin count for the same depth. |
| `max_amp`, `mean_amp`, `std_amp`, `total_energy` | Absolute ADC amplitudes. Scanner gain (AGC), target distance, atmosphere, and receiver sensitivity all affect these. Not portable. |

The current pointcloud is also pruned (234k of 1.6M points). A model that works
on the pruned version must also work on the full-density version and on different
surveys. Any feature that encodes "how many neighbours are nearby" breaks immediately.

**The fundamental insight**: the laser–target interaction physics is universal.
Water and land produce different *waveform shapes* because of their different
optical properties (specular vs. diffuse scattering, water-column exponential
backscatter, SVB decomposition). This shape information is encoded in the
**normalized waveform** and in **dimensionless ratios of waveform properties**.
Nothing else is needed — and anything else is a liability.

---

## 1. Feature Set: Generalizable Inputs Only

### 1.1  Primary Input: 2-Channel Normalized Waveform  `(B, 2, 200)`

**Channel 0 — amplitude shape** (float32):
```
waveform_norm[i] = waveform_grid[i] / max(waveform_grid[i])   # per-waveform max normalization
```
Values in [0, 1]. Preserves shape, discards absolute amplitude (scanner/AGC-dependent).

**Important**: use *per-waveform* normalization, NOT the global z-score normalization used in V8Net.
Global z-score is dataset-specific; per-waveform max is portable — any single waveform from any
scanner can be divided by its own max with no dataset statistics needed.

**Channel 1 — activity mask** (float32 0/1):
```
mask[i] = (waveform_grid[i] > 0).astype(float32)
```
Makes the gap structure explicit. The gap between surface and bottom return clusters is the most
physically meaningful feature for depth detection (it encodes water column traversal time).
The Transformer attends to the amplitude channel, but the mask channel tells it exactly where
signal exists and where it does not — removing ambiguity between "zero amplitude bins" and
"bins outside the waveform window."

### 1.2  Scalar Features: 11 Dimensionless / Physical-Unit Ratios

All features below are either dimensionless (ratios, fractions, counts) or in physical meters.
None require knowledge of point density, survey elevation, or scanner sampling rate.

```python
GENERALIZABLE_FEATURES = [
    # ── Dimensionless energy / amplitude ratios ─────────────────────────────
    "energy_concentration",        # Σ(amp²_early) / Σ(amp²)   — KEY: water=0.924, land=0.684
    "max_amp_norm_by_energy",      # max_amp / √total_energy    — dimensionless peak dominance
    "energy_ratio_late",           # Σ(amp²_late) / Σ(amp²)    — water=0.292 (bottom return!), land=0.171
    "active_bins_ratio",           # n_nonzero / n_bins         — waveform fill fraction
    "peak_amp_ratio",              # first_peak_amp / last_peak_amp  — ratio, no units

    # ── Derived dimensionless ratios (compute from existing features) ────────
    "gap_ratio",                   # total_gap / time_span  — fraction of span that is gaps
                                   # water=0.15, land=0.47  — strongest single discriminator!
    "energy_center_norm",          # amplitude_weighted_center / n_samples
                                   # fractional position of energy centroid in [0,1]
                                   # water=0.47 (early), land=0.61 (later)

    # ── Count features (scanner sampling-rate independent) ──────────────────
    "n_peaks",                     # number of distinct peaks
    "n_gaps",                      # number of gaps between signal clusters
                                   # water=0.44, land=2.40  — 82% separation!
    "n_clusters",                  # number of signal clusters

    # ── Physical-unit feature (meters, not SI) ───────────────────────────────
    "depth_proxy_m",               # peak_separation_SI × 0.05625 m/SI
                                   # = two-way travel time through water column
                                   # universal: speed of light in water is constant
]
# Total: 11 features
```

The two derived features must be computed in a pre-processing step:
```python
feat_df['gap_ratio']          = feat_df['total_gap'] / feat_df['time_span'].clip(lower=1)
feat_df['energy_center_norm'] = feat_df['amplitude_weighted_center'] / feat_df['n_samples'].clip(lower=1)
```

### 1.3  What is excluded, and why

| Excluded feature | Reason |
|---|---|
| `reflectance_dB` | Only 6.8% mean separation (water=-22.7 dB, land=-21.3 dB) with σ≈5 dB — barely above noise. Scanner-calibration and target-distance dependent. Adds more noise than signal for generalization. |
| `height_above_local_min`, `height_percentile_local` | Strong on Pielach (76.6%), useless elsewhere — density + topology dependent. |
| `planarity`, `roughness`, `linearity`, `sphericity` | k-NN PCA. Density-dependent. |
| `z_relative`, `height_range_local`, `height_std_local` | Absolute elevation / density dependent. |
| `max_amp`, `mean_amp`, `std_amp`, `total_energy` | Absolute ADC units. Scanner/AGC/distance dependent. |
| `n_samples`, `time_span`, `max_gap`, `mean_gap`, `total_gap` (raw SI) | Scanner sampling-rate dependent. Use `depth_proxy_m` and `gap_ratio` instead. |
| Spatial k-NN context | Density-dependent. The geometry of a 32-NN neighborhood changes completely at different point spacings. |

**Note on reflectance**: it is excluded here because it is the weakest feature (6.8% separation)
and introduces calibration dependency. If deploying on a survey with verified reflectance
calibration (same scanner family, same target distance), it can be added as a 12th feature with
appropriate weight decay to limit its influence.

---

## 2. Architecture

The model has two branches that fuse into a classification head.
No spatial context. No graph layers. Pure waveform physics.

```
  waveform (B, 2, 200)  ──→  WAVEFORM TRANSFORMER  ──→  (B, 128)
                                                              │
                                                          CONCAT
                                                              │
  scalars  (B, 11)       ──→  SCALAR MLP           ──→  (B, 64)
                                                              │
                                                          (B, 192)
                                                              │
                                                          HEAD MLP
                                                              │
                                                          (B, 2) logits
```

---

### 2.1  Waveform Transformer Branch

**Input**: `(B, 2, 200)` — 2-channel waveform (amplitude + mask)

```
Step 1  Patch tokenization
  Conv1D(2→128, kernel=4, stride=4, bias=False)   # 4-bin patches → 50 tokens
  BatchNorm1D(128) + GELU
  → (B, 128, 50) → transpose → (B, 50, 128)

  Patch size = 4 bins chosen so each token spans one pulse half-width
  (~1.5 ns = 3 bins, rounded up). Preserves peak shape within one token.

Step 2  Positional encoding
  Add sinusoidal PE(pos ∈ 0..49, d_model=128)
  This encodes the time position of each patch in the waveform (early vs. late).
  Sinusoidal (not learned) so it transfers to new datasets without retraining PE.

Step 3  Prepend [CLS] token
  cls = nn.Parameter(torch.zeros(1,1,128))
  x = cat([cls.expand(B,-1,-1), tokens], dim=1)   # (B, 51, 128)

Step 4  6× Transformer Encoder Layer
  Pre-LayerNorm architecture (more stable than post-LN):
    x = x + MHA(LN(x))          # MultiHeadAttention(d=128, heads=8, dropout=0.1)
    x = x + FFN(LN(x))          # FFN: Linear(128→512)+GELU+Dropout(0.1)+Linear(512→128)

Step 5  Classification token
  wf_emb = x[:, 0, :]                              # (B, 128)

Step 6  Projection
  wf_emb = Linear(128→128) + GELU
Output: (B, 128)
```

**Why 6 Transformer layers?**  
3 layers can distinguish simple vs. complex waveforms but struggle with subtle SVB patterns
(e.g., when the bottom return is weak or absent). 6 layers allow the model to first detect
individual peaks (early layers), then compute peak relationships (middle layers), then reason
about the full SVB pattern (deep layers). 8+ layers shows diminishing returns on 50-token inputs.

**What the attention learns**:
- Token 0 (bins 0–3): typically the surface return peak
- Tokens 8–20 (bins 32–80): where the bottom return appears for typical river depths (0.5–5 m)
- Attention between token 0 and later bottom-return tokens implements implicit SVB decomposition:
  the model learns "is there a bottom return?" and "how far is it?" — the depth proxy in continuous form

**Parameters**: ~1.5M

---

### 2.2  Scalar Feature Branch

**Input**: `(B, 11)` z-score normalized scalars (per-feature mean/std from training set only)

```
Linear(11→128) + BN + GELU

ResBlock × 2:
  Linear(128→128) + BN + GELU
  Linear(128→128) + BN
  + residual(x) + GELU

Linear(128→64) + BN + GELU
Output: (B, 64)
```

The scalar branch is smaller than in the previous design (64 vs 128 output) because 11 features
carry less redundant information than 32. Residual connections are kept for gradient stability.

**Parameters**: ~100k

---

### 2.3  Fusion and Classification Head

```
x = concat(wf_emb, scalar_emb)     # (B, 192)
x = Linear(192→256) + BN + GELU
x = Dropout(p=0.15)
x = Linear(256→128) + GELU
x = Linear(128→2)                  → logits

Output: (B, 2)  — class 0=land, class 1=water
```

Dropout only in the head. No dropout in the Transformer or scalar branch —
batch normalisation already regularises the scalar branch.

**Total parameters**: ~1.75M  (vs. V8Net ~500k, previous WCN design ~2.7M)

---

## 3. Training Protocol

Three phases. Total expected time: **~8–12 hours CPU, ~2 hours GPU**.

---

### Phase 1 — Masked Waveform Autoencoder (self-supervised, no labels)

The Transformer is initialized by predicting masked waveform content before ever
seeing labels. This is critical for convergence: random initialization produces an
attention pattern that ignores the gap between surface and bottom returns entirely.
Masked autoencoding forces the encoder to learn that the inter-cluster gap carries
information (you need context from before and after the gap to reconstruct the masked region).

```
Masking:
  Apply span masking — randomly select 3–6 contiguous spans, each 5–20 bins long
  Total masked: ~40% of bins
  Mask value: replace with 0.0 in channel 0, preserve 0 in channel 1

Decoder (training only — discarded afterward):
  2-layer MLP: d_model → 256 → 1  (one output per patch position)
  Reconstructs masked amplitude values

Loss:
  L = MSE(reconstructed_amps, true_amps)   over masked positions only

Normalisation:
  Per-waveform max normalisation applied BEFORE masking (model sees normalised input)

Optimiser:  AdamW(lr=1e-3, weight_decay=1e-4)
Scheduler:  CosineAnnealingLR(T_max=50)
Epochs:     50
Batch:      1024
Approx:     30 min (GPU) / 3 hr (CPU)
```

After pretraining, save waveform encoder weights. Freeze the first 2 Transformer layers
during Phase 2 warm-up.

---

### Phase 2 — Supervised Fine-tuning

**Labels**: `merged_label ∈ {0, 1}`. The 3,998 uncertain points (label=2) are excluded.

**Confidence weighting**: v8 labels vary in quality. Weight each training sample:

```python
conf_i  = (xgb_proba_i + deep_proba_i) * 0.5
weight_i = conf_i      if label_i == 1   # water labels: weight by model agreement
weight_i = 1.0         if label_i == 0   # land labels: geometrically assigned, high reliability
```

Practical effect: the ~500 water points with conf < 0.5 get near-zero weight and do not
corrupt gradients. The 75th percentile water confidence is 0.98, so most water labels are
well-calibrated.

**Loss function**:

```python
L = L_focal + λ_aux1 * L_energy_conc + λ_aux2 * L_depth_proxy

# L_focal: standard label-smoothed focal loss
#   gamma=2.0, alpha=0.65 (upweights water=minority class), smoothing=0.05
#   applied with per-sample confidence weights

# L_energy_conc: auxiliary regression — predict energy_concentration from wf_emb
#   Linear(128→1) head on waveform embedding
#   MSE against feat_df['energy_concentration']
#   Forces encoder to retain the energy-concentration information in its embedding

# L_depth_proxy: auxiliary regression — predict depth_proxy_m from wf_emb
#   Linear(128→1) head, applied only to points where depth_proxy_m > 0.1 m
#   Forces encoder to retain peak-separation information

λ_aux1 = 0.05
λ_aux2 = 0.05
```

Both auxiliary heads are discarded after training. They exist only to guide the
waveform encoder toward representations that preserve physically meaningful information.

**Schedule**:

```
Epochs 0–30:   First 2 Transformer layers frozen (stabilise pretrained weights)
               lr=5e-4, batch=512
Epochs 30–120: All layers unfrozen
               lr=2e-4, batch=512
Epochs 120–150 (optional): lr=5e-5, label-noise annealing:
               reduce smoothing from 0.05 → 0.02 (model is more confident, allow sharper targets)

Optimiser: AdamW(weight_decay=1e-4)
Scheduler: CosineAnnealingLR(T_max=150, eta_min=1e-6)
Early stopping: patience=20 on spatial-CV macro-F1
Approx: 1.5 hr (GPU) / 7 hr (CPU)
```

**Spatial cross-validation**: 5 folds split by y-coordinate strips (same as V8Net).
This is the only place where the spatial structure of the Pielach dataset is used.

---

### Phase 3 — Iterative Pseudo-label Refinement

```
1. Run full inference on all 234,024 points
2. For the 3,998 uncertain points (original label=2):
     assign pseudo-label=1  if  proba_water > 0.92
     assign pseudo-label=0  if  proba_water < 0.08
     keep excluded otherwise
3. Fine-tune 30 epochs on augmented label set
4. Repeat once
Expected: ~300–600 uncertain points resolved per round
Approx: 30 min (GPU) / 1.5 hr (CPU)
```

---

## 4. Waveform Normalisation: Per-Sample vs. Global

V8Net uses global z-score normalisation (mean/std computed from the training dataset).
This is problematic for deployment:
- The training dataset statistics are survey-specific
- A new survey with different water clarity or scanner gain would have different raw amplitude distributions
- The model silently receives out-of-distribution inputs

WCN v9 uses **per-sample max normalisation** for the waveform channel:
```python
wf_norm = waveform_grid / waveform_grid.max(axis=1, keepdims=True).clip(min=1.0)
```
This requires zero external statistics. Any waveform from any scanner can be processed
identically. The shape is preserved; the absolute amplitude is discarded (it is captured
separately by `reflectance_dB` if needed, but excluded from the current feature set).

For the **scalar features**, z-score normalisation is still applied, but only to the
dimensionless ratios and counts. These statistics are far less survey-specific than absolute
amplitudes: `n_peaks`, `gap_ratio`, `energy_concentration` have similar distributions across
surveys using the same scanner type. Save these stats to `wcn_stats.json` for inference-time use.

---

## 5. Optional: Site-Specific Spatial Refinement

For a single-site deployment where density is known and fixed, a post-processing step
can use local neighbourhood context to clean up boundary predictions WITHOUT encoding
it in the model weights:

```python
# After model inference, apply spatial majority vote on uncertain-margin points
from scipy.spatial import KDTree

tree = KDTree(coords_xy)
for margin_point_idx in margin_mask:  # proba ∈ [0.35, 0.65]
    _, nbr_idx = tree.query(coords_xy[margin_point_idx], k=16)
    nbr_preds  = predictions[nbr_idx]
    majority   = np.bincount(nbr_preds[nbr_preds < 2]).argmax()
    predictions[margin_point_idx] = majority   # override with local majority
```

This keeps the model itself density-independent while using spatial smoothing as a
separate, transparent post-processing step. The smoothing radius (in physical meters)
can be explicitly tuned for each deployment.

---

## 6. Improved XGBoost (Complementary)

Train an XGBoost on the same 11 generalizable scalar features. This serves as a
fast, interpretable baseline and ensemble component.

Compared to V8Net's XGBoost (32 features, CV F1=0.913), using only 11 features
will likely score slightly lower on Pielach but substantially better on unseen data
(fewer density-dependent features to overfit to).

Additional derived features to add (all generalizable):
```python
feat_df['peaks_per_cluster']  = feat_df['n_peaks'] / feat_df['n_clusters'].clip(lower=1)
feat_df['gap_per_peak']       = feat_df['n_gaps']  / feat_df['n_peaks'].clip(lower=1)
```

These ratio features further describe waveform complexity without units.

XGBoost configuration (same as current):
```python
xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw)
```

---

## 7. Ensemble

```python
wcn_proba    = wcn_model.predict_proba(waveforms, scalars)       # (N, 2)
xgb_proba    = xgb_model.predict_proba(scalar_features_11)      # (N, 2)
final_proba  = 0.65 * wcn_proba + 0.35 * xgb_proba

pred         = (final_proba[:,1] >= 0.50).astype(int)
uncertain    = (final_proba[:,1] > 0.35) & (final_proba[:,1] < 0.65)
```

WCN gets higher weight because it processes the raw waveform shape directly.

---

## 8. Model Summary

| Aspect | V8Net | WCN v9 |
|---|---|---|
| Waveform encoder | 5-layer 1D CNN | 6-layer Transformer, 2-channel input |
| Waveform normalization | Global z-score (survey-specific) | Per-sample max (portable) |
| Scalar features | 32 (including k-NN height/geometry) | 11 (dimensionless / physical meters only) |
| Spatial context | None | None (removed — density-dependent) |
| Graph layers | None | None (removed — density-dependent) |
| k-NN features | height_above_local_min etc. | None |
| Self-supervised pretrain | No | Masked waveform autoencoder |
| Aux regression heads | No | energy_concentration + depth_proxy_m |
| Label confidence weighting | No | Yes (v8 proba as sample weight) |
| Deployable on new survey | No (height features break) | Yes (no density assumptions) |
| Parameters | ~500k | ~1.75M |
| Expected F1 on Pielach | 0.956 (measured) | ~0.965–0.975 (estimated) |
| Expected F1 on new survey | Unknown (likely lower) | Should remain ~0.96 |

---

## 9. Output Files

```
models/wcn_v9/
  wcn_pretrained.pt       — waveform transformer after Phase 1
  wcn_finetuned.pt        — full model after Phase 2 (DEPLOY THIS)
  wcn_refined.pt          — full model after Phase 3
  wcn_stats.json          — scalar feature norm stats (mean/std for 11 features)
  wcn_metrics.json        — CV F1, AUC, per-fold
  attention_rollout.png   — per-bin attention weight averaged over water/land samples

data_processed/
  waveform_grids_norm.npy — (234024, 200) per-sample max normalised (compute once)
  features_v9.csv         — features_current.csv + gap_ratio + energy_center_norm

pointclouds/
  labeled_pointcloud_wcn.csv — same schema as labeled_pointcloud_current.csv
```

---

## 10. Implementation Checklist

- [ ] Compute `waveform_grids_norm.npy`: divide each row by its max (clip min=1.0)
- [ ] Compute `gap_ratio` and `energy_center_norm` columns in features CSV
- [ ] Phase 1: train masked waveform autoencoder on all 234k waveforms
- [ ] Phase 2a: freeze Transformer layers 0–1, fine-tune 30 epochs
- [ ] Phase 2b: unfreeze all, train to epoch 120 (or early stopping)
- [ ] Phase 3: pseudo-label refinement (2 rounds)
- [ ] Train XGBoost on 11+2 generalizable features
- [ ] Export `labeled_pointcloud_wcn.csv`
- [ ] Visualise attention rollout: do early-layer heads attend to peaks? Do deep heads attend across the surface-bottom gap?
- [ ] Test on full-density 1.6M point cloud (should work without retraining)
- [ ] Verify: no feature in the model requires k-NN computation at inference time
