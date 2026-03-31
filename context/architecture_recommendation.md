# Architecture Recommendation: Water vs. Land Classification from Full-Waveform LiDAR

*Based on knowledge extracted from EduServ 2021 course materials.*

---

## Data Format Recap

Each sample consists of:
- **Point**: (x, y, z, reflectance) — where reflectance is the Riegl calibrated backscattering coefficient γ
- **Waveform**: 1D time-series of amplitude values in ADC units, ~0.5 ns sampling interval (~150 samples per waveform typical)

Riegl reflectance (γ) is already range- and angle-corrected. This is the single most discriminative scalar feature for water vs. land based on the documents.

---

## Recommended Architecture: Two-Branch Multimodal Network

### Rationale

From the knowledge base:

1. **The waveform 1D signal** contains the physical signal of water detection:
   - Echo count (0, 1, or many)
   - Echo amplitude, width, shape (Gaussian parameters)
   - Water column produces characteristic exponential decay signature
   - Water surface echo vs. water bottom echo are temporally separated
   - These features cannot be reduced to a single scalar without information loss

2. **The point-level scalar features** (reflectance, z, neighborhood statistics) provide:
   - Calibrated reflectance γ: strongest single discriminator (water → low off-nadir, very high at nadir)
   - Height z: useful as prior (water near ground level)
   - Neighborhood-level features (density, roughness, planarity) are strong geometric discriminators

3. **The literature explicitly mentions** PointNet++ and DGCNN as state-of-the-art for ALS point cloud semantic labeling (1.10 transcript), and 1D CNNs are a natural fit for waveform classification.

### Architecture Design

```
Input:
  ├── Waveform: [N, T] where N = number of points, T = waveform length (~150 samples)
  └── Point features: [N, F] where F = {x, y, z, reflectance_γ, + optional: echo_count, amplitude, echo_width}

Branch A — Waveform Encoder (1D CNN):
  ├── Conv1D(32, kernel=7) → BN → ReLU
  ├── Conv1D(64, kernel=5) → BN → ReLU
  ├── Conv1D(128, kernel=3) → BN → ReLU
  └── GlobalMaxPool → [N, 128]   (waveform embedding per point)

Branch B — Point Feature MLP:
  ├── Linear(F → 64) → BN → ReLU
  ├── Linear(64 → 128) → BN → ReLU
  └── [N, 128]   (point feature embedding)

Fusion:
  ├── Concatenate: [N, 256]
  ├── Linear(256 → 128) → BN → ReLU → Dropout(0.3)
  ├── Linear(128 → 64) → BN → ReLU
  └── Linear(64 → 2) → Softmax   (water / land)
```

### Why This Architecture Fits

| Design Choice | Justification from Source Documents |
|---------------|-------------------------------------|
| 1D CNN for waveform | Waveforms are 1D temporal signals (~0.5 ns samples). 1D CNN captures local temporal patterns (echo peaks, width). Generalizes better than Gaussian decomposition alone. |
| Global max pooling on waveform | Extracts strongest echo characteristics regardless of position in waveform. |
| Reflectance (γ) as explicit feature | Directly mentioned as the most discriminative feature: "the deeper it gets, the lower the reflectance gets" (5.03). Riegl reflectance is already calibrated. |
| Fusion of both branches | Waveform shape + point-level reflectance are complementary: reflectance alone cannot distinguish nadir-water (high return) from land, but waveform shape can. |
| Independent point classification | Avoids the need for neighborhood aggregation, works even in data-void regions. |

---

## Alternative: PointNet++/DGCNN with Waveform Features

If neighborhood context is desired (e.g., to handle data voids):

1. **Pre-extract per-point waveform features** using the 1D CNN above → get waveform embedding per point.
2. **Concatenate** with (x, y, z, reflectance) → augmented point features.
3. **Feed into PointNet++** or **DGCNN** for spatially-aware classification.

PointNet++ (Winiwarter et al. 2018 for ALS, cited in 1.10 transcript) handles irregular point clouds and can aggregate neighborhood information, which helps:
- Fill in data voids (neighbors around a void are likely water).
- Use planarity/roughness implicitly via hierarchical grouping.

DGCNN (Widyaningrum et al. 2021, cited in 1.10 transcript) builds dynamic graph edges → useful for capturing local surface structure.

---

## Input/Output Format

### Input (per point):
```python
{
    "waveform": np.array([T]),          # T ≈ 150 amplitude values (ADC units), 0.5 ns sampling
    "reflectance": float,               # Riegl calibrated backscattering coefficient γ [dB or linear]
    "x": float,                         # UTM easting [m]
    "y": float,                         # UTM northing [m]
    "z": float,                         # ellipsoidal height [m]
}
```

### Output (per point):
```python
{
    "class": int,           # 0 = land, 1 = water
    "probability": float,   # softmax probability for water class
}
```

### Optional additional inputs (if available from Gaussian decomposition):
- Number of echoes per pulse
- Amplitude of first echo (A₁)
- Echo width of first echo (σ₁)
- Temporal position of echoes (relative to pulse)

---

## Training Plan

### Data Preparation

1. **Ground truth labeling**: Use manual annotation of the Pielach river cross-sections, or use the known water surface model (WSM) derived from the green+IR channel combination to automatically label points below the water surface as "water" and dry terrain as "land."

2. **Waveform normalization**:
   - Normalize amplitude values to [0, 1] range (divide by max ADC value).
   - Pad/truncate all waveforms to fixed length T.
   - Zero-padding is appropriate for waveforms shorter than T.

3. **Feature normalization**:
   - Reflectance γ: apply log transform if values span many orders of magnitude (e.g., log10(γ + ε)).
   - z coordinate: subtract mean height of the scene (relative height).
   - x, y: can be excluded or normalized to local patch coordinates.

4. **Class imbalance handling** (water is typically a small fraction of the scene):
   - Weighted cross-entropy loss: w_water = N_total / (2 × N_water), w_land = N_total / (2 × N_land).
   - Or: oversample the minority (water) class via random duplication.
   - Or: use focal loss (γ=2 is common).

5. **Data augmentation**:
   - Random noise addition to waveform amplitudes (simulate SNR variation).
   - Random scale of amplitude (simulate different reflectances).
   - Jitter in x, y, z positions.
   - Note: do NOT augment reflectance γ if it is already calibrated — it is a physical measurement.

### Loss Function

Weighted binary cross-entropy or focal loss, since water is a minority class.

### Training Strategy

1. Pre-train the waveform encoder (Branch A) on waveform reconstruction or auxiliary tasks if labels are scarce.
2. Fine-tune end-to-end.
3. Use Adam optimizer with learning rate 1e-3, reduce on plateau.
4. Evaluate on held-out strips (not just held-out points) to test generalization.

### Validation

- Evaluate on a different date (e.g., August 2019 turbid data vs. March 2021 clear data) to test robustness to turbidity.
- Evaluate at different scan angles to test robustness to specular effects.

---

## Whether Ensemble/Multi-Modal Approaches Help

Based on the documents:

1. **Multi-modal (waveform + point features)**: **Yes, strongly recommended.** Reflectance alone has failure modes (nadir-water looks like land; specular dropout looks like no-data). Waveform shape provides independent information.

2. **Ensemble of multiple classifiers**: The documents do not discuss this for LiDAR, but RF (mentioned as standard for LiDAR classification) could serve as a fast, interpretable baseline to compare against deep learning.

3. **Multi-wavelength fusion** (if available): NDVI between green (532 nm) and NIR (1064 nm) channels is mentioned by Morsy et al. (2017) for land cover classification. For water specifically: the NIR channel provides water surface, green provides water column — combining both gives complementary information. However, the Pielach data uses single-channel green for bathy + IR for topo. If both are available: combine.

4. **Temporal multi-date fusion**: Not discussed in source documents.

---

## Summary Recommendation

| Approach | Pros | Cons |
|----------|------|------|
| **Two-branch (1D CNN + MLP) [Primary recommendation]** | Uses full waveform + reflectance; handles all surface types; simple to implement | No spatial context |
| **PointNet++ with waveform features** | Captures spatial neighborhood; handles data voids via context | More complex; needs point neighborhood computation |
| **Random Forest on Gaussian decomp. features** | Fast, interpretable, well-understood | Loses raw waveform information; Gaussian decomp. may fail on weak echoes |
| **Riegl reflectance threshold only** | Trivial to implement | Fails at nadir water (high reflectance) and misses edge cases |

**Start with the two-branch network** as primary model. Use random forest on Gaussian decomposition features as a strong interpretable baseline. If spatial context is needed (handling data voids, edge smoothness), add PointNet++ with the waveform encoder as a feature extractor.

**Critical preprocessing steps that must not be skipped**:
1. Radiometric calibration (Riegl reflectance must be properly calibrated γ values, not raw DN).
2. Waveform normalization (per-waveform amplitude normalization to ADC max).
3. Handling of no-return points (specular water voids): these are informative features — label them as "likely water" or treat the absence of return as a distinct class/feature.