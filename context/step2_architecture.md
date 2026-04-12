# Step 2 Architecture: WaveformNet v9 (Generalizable)

**Proposed 2026-04. Classifies any 532 nm bathymetric LiDAR point cloud. No site elevation, river width, valley shape, or point density assumptions.**

---

## 0. The Generalization Problem

V8Net: F1=0.956 on Pielach. Relies on **survey-specific** features:

| Feature group | Why it fails on a new survey |
|---|---|
| `height_above_local_min`, `height_percentile_local` | k-NN at *specific density*. Scanner altitude/overlap/decimation changes neighbors → changes stats. Assumes river=valley floor; fails for levee-bounded rivers. |
| `planarity`, `roughness`, `linearity`, `sphericity` | k-NN PCA eigenvalue ratios. Density-dependent — same surface differs at 15 cm vs. 5 cm spacing. |
| `height_range_local`, `height_std_local` | Density + topology dependent. |
| `z_relative` | Absolute elevation. Useless on new sites. |
| `n_samples`, `time_span`, raw `max_gap`, `total_gap` | **Scanner SI units** (1 SI = 0.5 ns for RIEGL). Different scanner (1 ns) → half bin count for same depth. |
| `max_amp`, `mean_amp`, `std_amp`, `total_energy` | Absolute ADC amplitudes. AGC, distance, atmosphere, receiver sensitivity vary. Not portable. |

Pointcloud pruned (234k of 1.6M). Model must work on full-density + new surveys. Features encoding neighbor count break immediately.

**Fundamental insight**: laser–target physics universal. Water/land differ in *waveform shape* (specular vs. diffuse, water-column backscatter, SVB). Shape → **normalized waveform** + **dimensionless ratios**. All else = liability.

---

## 1. Feature Set: Generalizable Inputs Only

### 1.1  Primary Input: 2-Channel Normalized Waveform  `(B, 2, 200)`

**Channel 0 — amplitude shape** (float32):
```
waveform_norm[i] = waveform_grid[i] / max(waveform_grid[i])   # per-waveform max normalization
```
Values in [0, 1]. Preserves shape, discards absolute amplitude (scanner/AGC-dependent).

**Important**: use *per-waveform* norm, NOT global z-score (V8Net). Global z-score = dataset-specific. Per-waveform max = portable, zero external stats.

**Channel 1 — activity mask** (float32 0/1):
```
mask[i] = (waveform_grid[i] > 0).astype(float32)
```
Gap structure explicit. Surface-bottom gap = most meaningful feature (encodes water column time). Transformer attends amplitude; mask signals where signal exists — no ambiguity between zero-amp bins and out-of-window bins.

### 1.2  Scalar Features: 11 Dimensionless / Physical-Unit Ratios

All dimensionless (ratios, fractions, counts) or physical meters. No density/elevation/sampling rate required.

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

Two derived features computed in pre-processing:
```python
feat_df['gap_ratio']          = feat_df['total_gap'] / feat_df['time_span'].clip(lower=1)
feat_df['energy_center_norm'] = feat_df['amplitude_weighted_center'] / feat_df['n_samples'].clip(lower=1)
```

### 1.3  What is excluded, and why

| Excluded feature | Reason |
|---|---|
| `reflectance_dB` | 6.8% mean sep (water=-22.7 dB, land=-21.3 dB), σ≈5 dB — near noise. Scanner-cal + distance dependent. More noise than signal. |
| `height_above_local_min`, `height_percentile_local` | Strong on Pielach (76.6%), useless elsewhere. Density + topology dependent. |
| `planarity`, `roughness`, `linearity`, `sphericity` | k-NN PCA. Density-dependent. |
| `z_relative`, `height_range_local`, `height_std_local` | Absolute elevation / density dependent. |
| `max_amp`, `mean_amp`, `std_amp`, `total_energy` | Absolute ADC units. Scanner/AGC/distance dependent. |
| `n_samples`, `time_span`, `max_gap`, `mean_gap`, `total_gap` (raw SI) | Scanner sampling-rate dependent. Use `depth_proxy_m` and `gap_ratio` instead. |
| Spatial k-NN context | Density-dependent. 32-NN geometry changes at different spacings. |

**Note on reflectance**: weakest feature (6.8% sep), adds calibration dependency. Add as 12th feature if survey has verified calibration (same scanner family, same distance) — use weight decay.

---

## 2. Architecture

Two branches → classification head. No spatial context. No graph layers. Pure waveform physics.

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
3 layers: simple vs. complex OK, subtle SVB fails (weak/absent bottom return). 6 layers: peaks (early) → relationships (mid) → full SVB (deep). 8+ = diminishing returns on 50 tokens.

**What the attention learns**:
- Token 0 (bins 0–3): surface return peak
- Tokens 8–20 (bins 32–80): bottom return for typical depths (0.5–5 m)
- Attention token 0 ↔ bottom-return tokens → implicit SVB: "bottom return present? how far?" = continuous depth proxy.

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

Smaller than prev design (64 vs 128 output): 11 features less redundant than 32. Residuals kept for gradient stability.

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

Dropout in head only. No dropout in Transformer/scalar branch — BN regularises scalar branch.

**Total parameters**: ~1.75M  (vs. V8Net ~500k, previous WCN design ~2.7M)

---

## 3. Training Protocol

Three phases. **~8–12 hr CPU, ~2 hr GPU**.

---

### Phase 1 — Masked Waveform Autoencoder (self-supervised, no labels)

Transformer predicts masked content before labels. Critical for convergence: random init ignores surface-bottom gap. Masking forces encoder to learn gap = information (needs before/after context to reconstruct).

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

Save waveform encoder weights. Freeze first 2 Transformer layers in Phase 2 warm-up.

---

### Phase 2 — Supervised Fine-tuning

**Labels**: `merged_label ∈ {0, 1}`. The 3,998 uncertain points (label=2) are excluded.

**Confidence weighting**: v8 labels vary. Per-sample weights:

```python
conf_i  = (xgb_proba_i + deep_proba_i) * 0.5
weight_i = conf_i      if label_i == 1   # water labels: weight by model agreement
weight_i = 1.0         if label_i == 0   # land labels: geometrically assigned, high reliability
```

~500 water points conf < 0.5 → near-zero weight, no gradient corruption. 75th pct water conf = 0.98 → most labels well-calibrated.

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

Aux heads discarded after training. Guide encoder to preserve physically meaningful representations.

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

**Spatial cross-validation**: 5 folds by y-coordinate strips (same as V8Net). Only use of Pielach spatial structure.

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

V8Net: global z-score (training dataset). Problematic for deployment:
- Training stats = survey-specific
- New survey (diff water clarity/gain) → diff amplitude distributions
- Model silently gets OOD inputs

WCN v9: **per-sample max norm** for waveform:
```python
wf_norm = waveform_grid / waveform_grid.max(axis=1, keepdims=True).clip(min=1.0)
```
Zero external stats. Any scanner, identical processing. Shape preserved; absolute amplitude discarded (captured by `reflectance_dB` if needed, excluded here).

**Scalar features**: z-score still applied to dimensionless ratios/counts. Less survey-specific than absolute amplitudes: `n_peaks`, `gap_ratio`, `energy_concentration` stable across same-scanner surveys. Save to `wcn_stats.json`.

---

## 5. Optional: Site-Specific Spatial Refinement

Single-site deployment (known density): post-processing cleans boundary predictions WITHOUT touching model weights:

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

Model stays density-independent. Spatial smoothing = separate, transparent step. Smoothing radius (meters) tunable per deployment.

---

## 6. Improved XGBoost (Complementary)

XGBoost on 11 generalizable features. Fast, interpretable baseline + ensemble component.

vs. V8Net XGBoost (32 features, F1=0.913): 11 features likely lower on Pielach, better on unseen data (fewer density-dependent features to overfit).

Additional derived features (all generalizable):
```python
feat_df['peaks_per_cluster']  = feat_df['n_peaks'] / feat_df['n_clusters'].clip(lower=1)
feat_df['gap_per_peak']       = feat_df['n_gaps']  / feat_df['n_peaks'].clip(lower=1)
```

Ratio features: describe waveform complexity, no units.

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

WCN higher weight: processes raw waveform shape directly.

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
- [ ] Visualise attention rollout: early heads → peaks? Deep heads → surface-bottom gap?
- [ ] Test on full-density 1.6M point cloud (should work without retraining)
- [ ] Verify: no feature in the model requires k-NN computation at inference time