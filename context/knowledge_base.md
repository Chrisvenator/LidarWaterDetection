# Water vs. Land Classification from Full-Waveform Bathymetric LiDAR — Knowledge Base

---

## 1. Problem Overview

The goal is to train a supervised ML model to classify **water** vs. **land** points from full-waveform topo-bathymetric LiDAR point cloud data. Because no manual labels exist, the workflow is:

1. Build a **rule-based auto-labeler** grounded in domain physics (this knowledge base)
2. Apply it to generate noisy pseudo-labels on the Pielach River dataset
3. Train a supervised model (with label noise robustness) on the pseudo-labels
4. Iterate: use the model's confident predictions to refine labels

Scanner: **RIEGL VQ-840-GL** (green, 532 nm) + **RIEGL miniVUX-3UAV** (NIR, 905 nm). Dataset: ~750 m reach, pre-alpine Pielach River (Austria, Oct 2024), ~234,024 points at ~15 cm spacing.

[Source: DOI_10.23784_HN130-06.pdf, 5.01 transcript, 5.03 transcript]

---

## 2. Physics of Bathymetric LiDAR Waveforms

### 2.1 Green laser (532 nm) interaction with water

Water has **minimum optical attenuation** in blue-green spectrum (460–550 nm). 532 nm produced by frequency-doubling Nd:YAG IR laser (1064 nm). "Atmospheric window" lets green photons penetrate water column, reach bottom, return to sensor — impossible with NIR (~905/1064 nm).

Total received power = sum of water surface (PWS), water column (PWC), water bottom (PWB), background (PBK):
```
PR = PWS + PWC + PWB + PBK
```
[Source: 5.01 Topo-Bathy_Measurement principle.pdf, LIDARMagazine Part3]

### 2.2 Water surface returns (specular/diffuse reflection)

At air–water interface, **part of green signal reflected** back, **part refracted** into water column. Reflected fraction depends on:
- **Surface roughness**: Ruffled water (ripples) scatters more signal back than flat mirror.
- **Incidence angle**: Near-nadir beams hitting water can **saturate** receiver. Bathymetric scanners use **conical (Palmer) scanning at constant off-nadir ~15–20°**.
- **Specular vs. diffuse**: Water surface is **predominantly specular**. At steep off-nadir, nearly all signal reflects away → **data voids**. Only ruffled surface yields detector signal.

Backscattering solid angle omega is **very small** for specular targets → water surfaces often show data voids.

Water surface return characteristics:
- Typically **first and strongest peak** in waveform
- Relatively low amplitude (oblique scan angle)
- High spatial variability (missing in smooth/standing water)

[Source: 2.01 Multispectral_Laser Radar Equation transcript, 5.01 transcript, LIDARMagazine Part3, 5.01 Measurement principle slides]

### 2.3 Water column backscatter and exponential decay

Within water column, laser interacts with suspended sediment + water molecules. Signal **scattered and absorbed**. Received power from water column follows **exponential decay**:

```
PWC(z) ∝ exp(-2kz / cos(αW))
```

where:
- z = water depth
- k = diffuse attenuation coefficient (turbidity)
- αW = water-sided incidence angle (after refraction)

Produces **continuous, decaying signal** between surface and bottom peaks — fundamentally different from land's sharp isolated Gaussian peaks.

Exponential water column backscatter = basis for **SVB algorithm**: "Volume" component captures exponential tail, distinguishes from sharp solid surface returns.

Volume backscattering causes amplitude to drop **asymmetrically** after surface peak — distinctive signature for water column traversal.

[Source: 5.01 Topo-Bathy_Measurement principle slides, LIDARMagazine Part3, 5.03 transcript]

### 2.4 Bottom returns through water

If water clear enough, laser reaches bottom and reflects off riverbed/seabed:
- **Gravel and light sand**: high reflectivity → favors deeper penetration
- **Muddy soil or dark submerged vegetation**: low reflectivity → reduces depth
- Pielach River has **coarse gravel (2–6.3 cm)** → relatively high bottom reflectance

Bottom return: **broad Gaussian peak** appearing later in waveform, separated from surface by two-way travel time through water. Typically **weaker** than surface echo due to absorption/scattering losses.

**Peak separation** ∝ water depth:
```
depth = (delta_t_SI * 0.5e-9 * c_water) / 2
c_water = 225,000,000 m/s (225,000 km/s)
```
For 1 SI unit = 0.5 ns: delta_z_per_SI = 0.5e-9 * 225e6 / 2 ≈ 0.056 m per SI unit (water-side), ~**5.6 cm per SI unit of peak separation**.

**Shallow water problem**: Depth < ~20 cm → surface and bottom echoes **cannot be separated** (pulse length 1.5 ns ≈ 22.5 cm in water longer than two-way travel time).

[Source: 5.01 transcript, 5.02 Sensor overview slides, LIDARMagazine Part3, 1.03 transcript]

### 2.5 NIR laser (905 nm) interaction with water (for comparison)

**miniVUX-3UAV** at **905 nm**: water has very high absorption — NIR does **not penetrate** water. Instead:
- NIR → only **single water surface return** (specular reflection at air–water interface)
- NIR → **no bottom returns**
- miniVUX-3UAV = **topographic scanner** for dry land + water surface model for refraction correction of green channel

Two-channel exploitation: NIR reliably detects water surface (at nadir); green provides surface + bottom returns.

[Source: 5.01 transcript, 5.02 transcript, DOI_10.23784_HN130-06.pdf, LIDARMagazine Part3]

### 2.6 Land and vegetation waveform characteristics

**Dry land / hard surfaces** (gravel, concrete, asphalt):
- Single, narrow, roughly symmetric Gaussian peak
- High amplitude (extended target)
- Short echo width (sharp, clear return)
- No exponential decay after peak

**Vegetation** (trees, shrubs, grass):
- Multiple peaks from canopy layers, branches, ground
- Broadened echo width (vertical extent)
- Decreasing amplitude with depth
- Echo width >> hard surfaces

**Buildings/roofs**: Single sharp peak, high amplitude, short echo width

**Key waveform shape differences**:
- Water: flat/mixed pulse, exponential decay after first peak, possible separated bottom peak
- Land: distinct narrow Gaussian peak(s), no exponential decay
- Vegetation: multiple broad peaks, varying amplitudes

[Source: 1.02 transcript, 1.06 transcript, 2.02 transcript, 1.10 transcript]

### 2.7 Reflectance properties by surface type

Reflectance at 532 nm (values below for 900 nm — green reflectance generally lower):

From laser radar equation slide (λ=900 nm reference):
- White paper: up to 100%
- Snow: 80–90%
- Deciduous trees: ~60%
- Coniferous trees: ~30%
- Dry carbonate sand: ~57%
- Asphalt: ~17%
- Black rubber: ~2%

At green (532 nm):
- Most surfaces reflect **less** at 532 nm than NIR
- Water column with gravel bottom: **intermediate to low reflectance** at 532 nm
- `_riegl.reflectance` column in dB is **negative** (low-to-moderate reflectors)

Dataset analysis:
- Reflectance range: -30.8 to -6.3 dB
- Mean: -21.6 dB, Median: -22.3 dB, StdDev: ~5.0 dB
- **Lower dB = lower reflectance** → likely water/deep zones
- **Higher dB (less negative) = higher reflectance** → likely land/gravel/vegetation

Bimodal potential in histogram (~-27 to -23 dB peak, secondary ~-13 to -11 dB) may correspond to water vs. land.

[Source: 2.01 transcript, 2.01 slides, LIDARMagazine Part2, dataset analysis]

---

## 3. Waveform Processing Algorithms

### 3.1 Online Waveform Processing (OWP)

OWP (Pfennigbauer et al. 2014) processes waveforms **in real-time during flight**. Uses **reduced set of recorded waveforms around signal peaks** to extract points online:
1. Detect local maxima (peaks) above noise threshold
2. Fit Gaussian curves to peaks (Gaussian decomposition)
3. Extract per-echo attributes: range (μ), amplitude (A), echo width (σ)
4. Output as discrete 3D points with attributes

OWP standard for miniVUX-3UAV (NIR). VQ-840-GL (green) uses OWP as first processing pass.

**Limitation**: OWP may miss weak bottom returns in turbid/deep water — detection threshold suppresses weak exponential tail.

[Source: DOI_10.23784_HN130-06.pdf, 5.03 transcript, 3.1 LiDAR workflow section]

### 3.2 Surface-Volume-Bottom (SVB) algorithm

SVB (Schwarz et al. 2019): **post-processing, water-specific full-waveform algorithm** for green channel. Key algorithm unlocking deeper penetration than OWP alone.

SVB decomposes waveform into three components:
1. **Surface**: Gaussian peak at water surface reflection
2. **Volume**: Exponential decay component for water column backscattering
3. **Bottom**: Gaussian peak at riverbed/seabed reflection

Critical innovation: **exponential decomposition** (Schwarz et al. 2017) — explicitly models exponential component instead of fitting Gaussian to water column (fails since water follows exponential, not Gaussian).

Pielach 2024 results:
- OWP: standard points for clear water sections
- SVB: additional bottom points in **deeper and more turbid** areas
- Combined: full penetration of entire riverbed including ~3 m deep pools
- SVB complements OWP, extends reach in challenging conditions

`waveform_df.txt` contains raw full-waveform data needed for SVB-style analysis.

[Source: DOI_10.23784_HN130-06.pdf, 5.03 slides, 5.03 transcript]

### 3.3 Gaussian decomposition and peak fitting

Standard Gaussian decomposition (Wagner et al. 2004):
```
wf(t) = c + sum_i [ A_i * exp(-(t - mu_i)^2 / (2*sigma_i^2)) ]
```
where:
- A_i = amplitude (peak height) of echo i
- mu_i = temporal position (proportional to range) of echo i
- sigma_i = echo width (standard deviation) of echo i
- c = noise offset

**Procedure**:
1. Detect local maxima above noise threshold → determines Gaussian component count
2. Apply non-linear parameter estimation (Levenberg-Marquardt) to fit Gaussians
3. Extract per-echo: (A_i, mu_i, sigma_i)

**Available per-echo attributes**:
- Amplitude (intensity/signal strength)
- Range R (from mu_i × c/2)
- Echo width w (= sigma_i, in ns)
- Backscatter cross-section (amplitude + echo width combined)

**Echo width interpretation**:
- Small sigma = sharp, narrow peak → hard surface (road, roof, water bottom in clear shallow water)
- Large sigma = broad peak → vegetation, mixed/penetrated surfaces, water

[Source: 1.06 transcript, 1.06 slides, 2.02 transcript]

### 3.4 Exponential decomposition for water column

Water column backscatter follows:
```
PWC(z) ∝ exp(-2kz / cos(αW))
```

Waveform amplitude after surface peak decays **exponentially** with depth/time. SVB explicitly models this (Schwarz et al. 2017), separating:
- Exponential volume backscatter tail (water column)
- Superimposed Gaussian peaks (surface and bottom echoes)

In raw waveform:
- Peak from water surface
- Slowly decaying amplitude between surface and bottom peak
- Second peak (possibly weak) from bottom

Exponential decay region in time-domain = diagnostic signature of water column traversal, distinguishable from land (no such decay).

[Source: DOI_10.23784_HN130-06.pdf, 5.01 slides, LIDARMagazine Part3]

---

## 4. Feature Engineering

### 4.1 Waveform-derived features (with computation methods)

All features computed from raw time-series (Time [SI], Amplitude [ADC]):

**Amplitude features**:
- `max_amplitude`: Maximum ADC value in waveform
- `mean_amplitude`: Mean of non-zero samples
- `amplitude_at_first_peak`: ADC value at first local maximum above threshold
- `first_peak_time`: SI units of first peak
- `last_peak_time`: SI units of last peak above threshold

**Peak structure features**:
- `n_peaks`: Count of local maxima above threshold (e.g., >100 ADC)
- `peak_spacing_mean`: Mean distance between consecutive peaks (SI units)
- `max_peak_spacing`: Maximum gap between any two consecutive peaks
- `first_to_last_peak_span`: SI span from first to last peak

**Echo width (pulse broadening)**:
- `sigma_first_peak`: Gaussian sigma of first peak
- `sigma_primary_peak`: Gaussian sigma of strongest peak
- `full_width_half_max`: FWHM of strongest peak

**Energy/Area features**:
- `total_energy`: Sum of all amplitude values (AUC)
- `energy_after_first_peak`: Energy after first peak → proxy for water column backscatter
- `exponential_decay_coefficient`: Estimated k from fitting exp(-alpha*t) to post-first-peak amplitude

**Shape features**:
- `waveform_skewness`: Skewness of amplitude distribution
- `waveform_kurtosis`: Kurtosis of amplitude distribution
- `asymmetry_ratio`: Energy before vs. after peak

**Gap structure** (key for non-contiguous time arrays):
- `n_gaps`: Gaps > 2 SI units between consecutive time samples
- `max_gap_size`: Largest gap in SI units
- `total_gap_extent`: Sum of all gap sizes
- `time_span`: last_time - first_time in SI units
- `gap_ratio`: total_gap_extent / time_span

**Multi-return indicators**:
- `n_clusters`: Count of distinct continuous time-segments
- `second_cluster_max_amp`: Max amplitude in second cluster (potential bottom return)
- `bottom_surface_amplitude_ratio`: A_bottom / A_surface (if two clear peaks identified)

**Reflectance** (from point cloud, not raw waveform):
- `reflectance_dB`: From `_riegl.reflectance` column (calibrated)

[Source: 1.06 transcript, 1.06 slides, 2.02 transcript, dataset analysis]

### 4.2 Point cloud-derived features (geometric, neighborhood)

Per point, compute from k-nearest neighbors or radius search:

**Height-based**:
- `z` (absolute elevation in ETRS89/UTM 33N)
- `z_relative_to_local_mean`: Height relative to local neighborhood mean
- `height_above_lowest_neighbor`: z - min(z in radius)

**Planarity and roughness**:
- Eigenvalue decomposition of local covariance matrix → λ1 ≥ λ2 ≥ λ3
- `planarity` = (λ2 - λ3) / λ1
- `roughness` = λ3 (smallest eigenvalue, local surface roughness)
- `linearity` = (λ1 - λ2) / λ1
- `sphericity` = λ3 / λ1

**Normal vector**:
- Normal vector components (nx, ny, nz) from PCA
- `normal_z`: Z-component (close to 1.0 for flat horizontal surfaces)
- `incidence_angle`: Angle between laser beam and local surface normal

**Density**:
- `point_density_radius`: Count within radius R
- `point_density_k`: 1 / mean_distance_to_k_neighbors

**Neighborhood height statistics**:
- `height_variance_local`: Variance of z in neighborhood
- `height_range_local`: Range of z in neighborhood

[Source: 1.10 transcript, 1.06 transcript, 2.03 transcript]

### 4.3 Most discriminative features for water vs. land with green laser

Ordered by expected importance:

1. **n_gaps / n_clusters**: Water waveforms show MANY gaps (time-discontinuities) — waveform stored as clusters around detected peaks. Multiple clusters = multiple returns (surface + column + bottom). Land: fewer clusters. **Observed: 80.9% of first 2000 waveforms have 3+ gaps.**

2. **max_gap_size**: Water waveforms have characteristic large gap (~70-90 SI units, ~35-45 ns, ~3.9-5.0 m in air) between surface return cluster and deep return cluster. Large gap = light travel time through water and back.

3. **peak_spacing / time_span**: For water, total waveform time span >> land at equivalent depth (water column adds travel time).

4. **reflectance_dB**: Water: lower reflectance (more negative dB) than land (gravel, vegetation). Dataset range: -30.8 to -6.3 dB; water likely clusters in -30 to -20 dB.

5. **energy_after_first_peak / exponential_decay**: Water: significant energy after first peak (water column backscatter). Land: rapid drop.

6. **n_peaks**: Multi-peak waveforms (3+) with wide spacing → strongly indicates water (surface + intermediate + bottom). Vegetation also multi-peak → needs height info too.

7. **roughness / planarity**: Water surface smoother than land (lower roughness, higher planarity). Gravel bank rougher than water surface.

8. **height_above_lowest_neighbor**: Water points cluster in consistent elevation band near/below water surface.

9. **sigma_first_peak / echo_width**: Water bottom returns may have broader echo width (footprint widening from forward scattering).

10. **second_cluster_max_amp / bottom_surface_ratio**: Second-cluster peak well-separated from first = strong water indicator.

[Source: 5.01 transcript, 5.03 transcript, LIDARMagazine Part3, 2.01 transcript, dataset analysis]

### 4.4 How to handle variable-length waveform input

Waveforms in `waveform_df.txt`: variable lengths (min=19, max=101 samples, mean=59.5) AND non-contiguous time arrays (gaps between clusters).

Key insight: Time array is **non-contiguous** — stores only sample-blocks around detected peaks. Cannot treat as dense 1D signal. Strategies:

**Strategy A: Hand-engineered features (recommended for auto-labeler)**
- Extract scalar features as in 4.1
- Feed to traditional ML (Random Forest, XGBoost) or simple MLP
- Avoids variable-length problem entirely

**Strategy B: Fixed-length dense embedding (for deep model)**
- Define global time grid (e.g., 0–200 SI units at 0.5 ns resolution = 400 bins)
- Place amplitudes into grid cells; unfilled = 0 (or noise floor)
- Consistent 400-bin 1D vector → feed to 1D CNN
- Preserves temporal position (peaks at consistent times = consistent depth)

**Strategy C: Cluster-level representation**
- Extract per-cluster stats: [time_offset, max_amp, mean_amp, width, sigma]
- Pad to fixed cluster count (e.g., max 5)
- Masking for variable-length cluster lists (Transformer-style attention)

Strategy B recommended for deep model: preserves key discriminative signal (peak timing relative to emission time). RNNs misleadingly process "gap=0" samples as real signal.

[Source: dataset analysis, DOI_10.23784_HN130-06.pdf Figure 2 description]

---

## 5. Classification Approaches

### 5.1 Traditional ML methods

For baseline model and auto-labeler validation:
- **Random Forest**: robust, handles mixed features (waveform scalars + geometric), interpretable feature importance, handles class imbalance via `class_weight`, no normalization needed
- **Gradient Boosting (XGBoost/LightGBM)**: excellent for tabular data, fast training
- **SVM**: effective if features well-normalized

[Source: 1.10 slides, 2.03 transcript]

### 5.2 Deep learning for point clouds

Spatial classification:
- **PointNet++**: Multi-scale local feature extraction from 3D point clouds. Applied to ALS data (Winiwarter et al. 2018). Can incorporate waveform features per point.
- **DGCNN (Dynamic Graph CNN)**: Dynamic graph convolution, captures local structure. Applied to ALS (Widyaningrum et al. 2021).
- **KPConv**: Kernel point convolution; near-perfect classification with green+NIR dual-wavelength (XYZ+G+NIR vs. geometry alone). [Source: LIDARMagazine Part2]
- **3D Sparse Voxel CNN** (Schmohl et al. 2019): Voxel-based, efficient for sparse 3D data

[Source: 1.10 transcript, 1.10 slides, LIDARMagazine Part2, 2.03 transcript]

### 5.3 Deep learning for 1D waveform signals

Direct waveform classification:
- **1D CNN**: Natural choice for fixed-length 1D signal. Learns multi-scale features (narrow peaks, broad exponential tails). Multiple kernel sizes for different temporal scales.
- **RNN/LSTM**: Handles sequential data, but struggles with non-contiguous time structure and long gaps.
- **Transformer**: Self-attention handles non-contiguity if positional encoding is time-based rather than index-based.

**Recommended**: 1D CNN with time-grid embedding (Strategy B from 4.4). RNNs process "gap=0" samples as real signal — misleading. CNNs on sparse grids robust via proper padding treatment.

### 5.4 Multi-modal architectures

Dataset has both point cloud attributes and raw waveforms. Multi-modal architecture:
1. **Waveform branch**: 1D CNN or feature extractor → waveform embedding
2. **Point cloud branch**: MLP on (x, y, z, reflectance, geometric features) → spatial embedding
3. **Fusion**: Concatenate embeddings → classification head

Matches dual-sensor setup: green waveform features capture water/land physics; spatial features (z, roughness, planarity) provide geometric context.

[Source: LIDARMagazine Part2, dataset analysis]

### 5.5 Unsupervised and semi-supervised approaches

**Bootstrap approach (this project)**:
1. Auto-label with rules → ~50–70% confident labels
2. Train supervised model on confident labels
3. Apply model to remaining points → propagate labels
4. Retrain with expanded training set

**Clustering for exploration**:
- K-means or DBSCAN on (reflectance, n_gaps, max_gap, z) to find natural clusters
- UMAP/t-SNE for feature space visualization

### 5.6 Training strategies and best practices

- **Class imbalance**: Pielach River (~20 m wide) in broader landscape → water points likely minority. Use `class_weight='balanced'` or focal loss.
- **Noisy labels**: Use label smoothing, confident learning (Cleanlab), or loss-correction methods.
- **Cross-validation**: Hold out one spatial strip (not random split) to test generalization to unseen areas.
- **Feature normalization**: Z-score normalize all features except physically meaningful ones (dB values, SI units).

[Source: DOI_10.23784_HN130-06.pdf, dataset analysis, general ML best practices]

---

## 6. Auto-Labeling Rules Catalog

### 6.1 Reflectance-based rules (with thresholds)

Dataset reflectance distribution:
- Range: -30.8 to -6.3 dB
- Mean: -21.6 dB, Median: -22.3 dB
- Roughly bell-shaped, tail at higher (less negative) values

**Rule R1**: Reflectance < -25 dB → candidate WATER (low reflectance)
- Rationale: Water low reflectance at 532 nm; very negative dB → likely water surface or submerged bottom.
- Fraction: ~22% of points

**Rule R2**: Reflectance > -15 dB → candidate LAND (high reflectance)
- Rationale: Gravel, meadow, dry vegetation have higher reflectance at 532 nm than water.
- Fraction: ~9.5% of points

**Rule R3**: Reflectance -25 to -15 dB → AMBIGUOUS (requires other features)

**Caveat**: Thresholds are informed estimates. Wet soil may have low reflectance similar to water.

[Source: 2.01 transcript, 2.02 transcript, dataset analysis]

### 6.2 Waveform shape rules (with criteria)

**Rule W1** (Multi-cluster indicator): `n_gaps >= 2` AND `max_gap_size >= 50 SI units (~25 ns, ~4.2 m in air travel)`
- Rationale: Water produces distinct surface and bottom return clusters with large temporal gap = two-way travel time through air+water. Gap >50 SI units = air-column travel time, not achievable from single flat land surface.
- Dataset: 80.9% of first 2000 waveforms have 3+ gaps; typical large gap (Points 0-4) ~70-83 SI units (~35-41 ns).
- Label: **Candidate for water or deep vegetation**

**Rule W2** (Exponential tail indicator): Energy in second half of waveform cluster relative to first half > 0.3
- Rationale: Water column backscatter produces sustained energy after first peak. Land: sharp peaks decay quickly.
- Label: **Candidate for water column or bottom**

**Rule W3** (Separated double peak): Two peaks > 100 ADC with temporal separation > 15 SI units (~7.5 ns, ~0.85 m water depth)
- Rationale: Direct water surface + bottom return signature.
- Time-to-depth: 15 SI × 0.5 ns/SI = 7.5 ns → 7.5e-9 s × 225e6 m/s / 2 ≈ 0.84 m water depth
- Label: **HIGH CONFIDENCE WATER**

**Rule W4** (Single sharp peak): Only 1 peak > 100 ADC AND total waveform span < 15 SI units AND no significant energy elsewhere
- Rationale: Clean single return from solid surface (land)
- Label: **Candidate for LAND**

**Rule W5** (Very long waveform): Total time span (last_time - first_time) > 100 SI units (50 ns)
- Rationale: Long spans indicate multiple reflections at different depths. At 60 m AGL, waveform span 100+ SI units suggests multiple discrete returns.
- Note: Dataset time values range ~37 to ~188 SI units (span ~150 SI); structure heavily gapped. Large terminal gap ~70-83 SI = gap between last surface/column return and final return (possibly bottom).
- Label: **Requires interpretation with other rules**

[Source: dataset analysis, 5.01 transcript, 5.03 transcript, LIDARMagazine Part3]

### 6.3 Amplitude-based rules (with thresholds)

Dataset analysis on first 2000 waveforms:
- max_amplitude range: 257–3962 ADC
- mean max_amplitude: 2369 ADC
- ADC range: 0–8191 (12-bit ADC)

**Rule A1**: max_amplitude > 3000 ADC → candidate LAND
- Rationale: Strong returns = diffuse, high-reflectance surfaces (dry gravel/meadow at moderate incidence). Water surface at off-nadir → weaker return (specular directionality).
- Caveat: Shallow water with high-reflectance bottom can also yield strong returns.

**Rule A2**: max_amplitude < 800 ADC → candidate DEEP WATER or ambiguous
- Rationale: Weak returns → absorption (water column), deep water, or dark surfaces.

**Rule A3**: First peak amplitude > second peak by factor of 3+
- Rationale: For water, bottom return typically much weaker than surface (absorption losses in water column). Land vegetation: multiple peaks can have similar amplitudes.
- Label: Supports WATER identification when combined with spatial separation

[Source: dataset analysis, 5.01 transcript, LIDARMagazine Part3]

### 6.4 Multi-peak / echo-based rules

**Rule M1** (High peak count): `n_peaks >= 4` with at least one separation > 20 SI units
- Rationale: Water with multiple return clusters. Dataset: 5+ peaks in 18.1%, 6+ peaks in ~36% — common for water-rich Pielach scene.
- Label: Candidate WATER (peaks widely separated); candidate VEGETATION (closely spaced).

**Rule M2** (Vegetation signature): `n_peaks >= 3` AND all inter-peak gaps < 15 SI units AND total span < 50 SI units
- Rationale: Dense vegetation → multiple closely-spaced peaks within short window.
- Label: **Candidate VEGETATION (not water)**

**Rule M3** (Clean single echo): `n_peaks == 1` AND max_gap_size < 5 SI units
- Rationale: Single clean echo from solid flat surface.
- Label: **Candidate LAND (high confidence)**

**Rule M4** (Water SVB signature): Waveform has:
- First cluster: 1-2 peaks in SI range [37-80], followed by
- Large gap (50+ SI units), followed by
- Second cluster: 1-2 peaks in SI range [150-200]
- Pattern: first cluster = water surface return, gap = travel time to bottom, second cluster = bottom return
- Exactly observed in sample waveforms 0-4!
- Label: **HIGH CONFIDENCE WATER**

[Source: dataset analysis, 5.01 transcript, LIDARMagazine Part3]

### 6.5 Elevation and geometry rules

**Rule E1** (Z-based water zone): Points within known/estimated river channel elevation range
- Pielach River Z range: 256.5–278.5 m, mean 264.9 m
- Very low Z + other water indicators → high confidence water

**Rule E2** (Planarity): Points in spatially smooth regions (low local roughness) near expected water elevation → candidate WATER SURFACE
- River surface: flat → low roughness, high planarity
- Gravel bank: rough → higher roughness

**Rule E3** (Height above DTM): Points significantly BELOW terrain model → water column or bottom → candidate WATER

**Rule E4** (Z clustering): Neighboring points with similar Z within few cm → likely water surface.

[Source: 5.03 transcript, dataset analysis]

### 6.6 Spatial context rules

**Rule S1** (Spatial connectivity): Point labeled WATER + nearby neighbors also WATER → increase confidence.

**Rule S2** (NIR data voids): Points lacking corresponding NIR returns over water-elevation zones → likely water (NIR absorbed by water → NIR data absent from underwater zones). Requires NIR point cloud for comparison.

**Rule S3** (River corridor): Points within geometric footprint of river channel → prior weight toward WATER.

**Rule S4** (Vegetation height): Points > 0.5 m above DTM = vegetation or structures → weight against both water and bare land.

[Source: 5.01 transcript, 5.03 transcript, DOI_10.23784_HN130-06.pdf]

### 6.7 Combined rule confidence scoring

Assign each point water confidence score (0–1):

```python
def compute_water_confidence(point, waveform_features, geometry_features):
    score = 0.5  # baseline neutral
    
    # Strong evidence for water (+)
    if waveform_features['max_gap_size'] >= 50:
        score += 0.20
    if waveform_features['n_gaps'] >= 3:
        score += 0.10
    if waveform_features['separation_between_cluster1_cluster2'] >= 15:  # SI units
        score += 0.15
    if reflectance_dB < -25:
        score += 0.10
    
    # Strong evidence against water (-)
    if waveform_features['n_peaks'] == 1 and waveform_features['max_gap_size'] < 5:
        score -= 0.25
    if reflectance_dB > -15:
        score -= 0.10
    if waveform_features['max_amplitude'] > 3000:
        score -= 0.05
    
    # Vegetation indicators (reduce water confidence)
    if waveform_features['n_peaks'] >= 3 and waveform_features['max_peak_spacing'] < 15:
        score -= 0.10
    
    # High-confidence water signatures
    if check_svb_signature(waveform):  # Rule M4: surface cluster + large gap + bottom cluster
        score = max(score, 0.85)
    
    # Clip to [0, 1]
    return max(0.0, min(1.0, score))

# Thresholds for labeling:
# confidence >= 0.70 → label as WATER
# confidence <= 0.30 → label as LAND
# 0.30 < confidence < 0.70 → UNCERTAIN (exclude from training)
```

### 6.8 Known edge cases and failure modes

1. **Shallow water (< 20 cm depth)**: Surface and bottom echoes cannot be separated → waveform looks like single return from solid surface → may be mislabeled as LAND. Min separable depth ~20 cm for VQ-840-GL.

2. **Very smooth water surface (sun glint / calm day)**: Near-nadir specular reflection → very strong first peak, no penetration → may look like land. Palmer scan at ±20° helps, but calm inland rivers on clear days remain an issue.

3. **Wet soil / puddles**: Low reflectance at 532 nm, possible smooth surface → may be misclassified as water. Geometry (depth, roughness) needed to distinguish.

4. **Dark land surfaces** (dense wet vegetation, dark asphalt): Low reflectance → may pass reflectance-based water rules. Need multi-peak and elevation features.

5. **Semi-submerged vegetation** (macrophytes, aquatic plants): Water signature (surface + water column) but bottom return from plants, not gravel → complex waveform.

6. **Survey October 2024**: Good water clarity (one month after September 2024 flood). Different turbidity would change waveform structure significantly.

7. **Waveform-to-point mapping**: Multiple points can share same waveform (one waveform → multiple extracted echoes = multiple rows in `point_cloud_df.txt` sharing same waveform). In `waveform_df.txt`, 0.2% of consecutive rows share same waveform data (checked first 1000 rows).

[Source: DOI_10.23784_HN130-06.pdf, 5.02 slides, LIDARMagazine Part3, dataset analysis]

---

## 7. Preprocessing Pipeline

### Step 1: Load and align data
```python
# Files
# data/point_cloud_df.txt  → x, y, z, _riegl.reflectance (+ unnamed index)
# data/waveform_df.txt     → Time [SI], Amplitude [ADC] (numpy array strings)

# Alignment: row i in point_cloud_df corresponds to row i in waveform_df
# Note: Multiple point rows may share the same waveform (multi-echo case)
```

### Step 2: Parse waveform arrays
```python
import re
import numpy as np

def parse_numpy_array_string(s):
    """Parse numpy array string format: '[ 37  38  39  40 ...]'"""
    nums = re.findall(r'[-+]?\d+', str(s))
    return np.array([int(x) for x in nums])

# Apply to each row of waveform_df
times = parse_numpy_array_string(row['Time [SI]'])
amps = parse_numpy_array_string(row['Amplitude [ADC]'])
```

### Step 3: Extract waveform features
```python
def extract_waveform_features(times, amps, min_peak_amp=100, gap_threshold=2):
    """Extract scalar features from a single waveform."""
    features = {}
    
    # Basic stats
    features['max_amp'] = np.max(amps)
    features['mean_amp'] = np.mean(amps)
    features['total_energy'] = np.sum(amps)
    features['n_samples'] = len(times)
    features['time_span'] = times[-1] - times[0] if len(times) > 1 else 0
    
    # Gap analysis (key for water detection)
    if len(times) > 1:
        diffs = np.diff(times)
        gaps = diffs[diffs > gap_threshold]
        features['n_gaps'] = np.sum(diffs > gap_threshold)
        features['max_gap'] = np.max(diffs)
        features['mean_gap'] = np.mean(gaps) if len(gaps) > 0 else 0
        features['total_gap'] = np.sum(gaps)
    else:
        features['n_gaps'] = 0
        features['max_gap'] = 0
        features['mean_gap'] = 0
        features['total_gap'] = 0
    
    # Peak detection
    peaks = [i for i in range(1, len(amps)-1) 
             if amps[i] > amps[i-1] and amps[i] > amps[i+1] and amps[i] >= min_peak_amp]
    features['n_peaks'] = len(peaks)
    
    if len(peaks) >= 2:
        peak_times = [times[p] for p in peaks]
        spacings = np.diff(peak_times)
        features['max_peak_spacing'] = np.max(spacings)
        features['mean_peak_spacing'] = np.mean(spacings)
        features['first_to_last_peak_span'] = peak_times[-1] - peak_times[0]
    else:
        features['max_peak_spacing'] = 0
        features['mean_peak_spacing'] = 0
        features['first_to_last_peak_span'] = 0
    
    # Cluster analysis
    n_clusters = 1 + features['n_gaps']  # approximate
    features['n_clusters'] = n_clusters
    
    return features
```

### Step 4: Extract geometric features
```python
from sklearn.neighbors import KDTree

def extract_geometric_features(pc_df, k=20, radius=0.5):
    """Extract neighborhood-based geometric features."""
    xyz = pc_df[['x', 'y', 'z']].values
    tree = KDTree(xyz)
    
    features = []
    for i in range(len(xyz)):
        # k-nearest neighbors
        idx = tree.query([xyz[i]], k=k+1, return_distance=False)[0][1:]
        neighbors = xyz[idx]
        
        # PCA
        centered = neighbors - neighbors.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]  # descending
        lam1, lam2, lam3 = eigenvalues
        
        feat = {
            'planarity': (lam2 - lam3) / (lam1 + 1e-8),
            'roughness': lam3,
            'linearity': (lam1 - lam2) / (lam1 + 1e-8),
            'sphericity': lam3 / (lam1 + 1e-8),
            'height_range_local': neighbors[:, 2].max() - neighbors[:, 2].min(),
            'height_std_local': neighbors[:, 2].std(),
            'z_relative': xyz[i, 2] - neighbors[:, 2].mean(),
        }
        features.append(feat)
    
    return pd.DataFrame(features)
```

### Step 5: Normalize features
- Z-score normalize all features
- Keep `reflectance_dB` as-is (physically meaningful)
- Time-based features (SI units) can be kept in SI or normalized

### Step 6: Handle coordinate system
- Coordinates in ETRS89/UTM 33N (EPSG:25833) with local offsets: x ≈ -269, y ≈ 97, z ≈ 261-264
- Do NOT add large absolute x,y,z as model features — use relative/local coordinates
- z is most useful absolute feature (elevation above reference)

[Source: DOI_10.23784_HN130-06.pdf, dataset analysis]

---

## 8. Evaluation Strategy

No ground truth → indirect evaluation:

1. **Physical plausibility**: Labeled water points form spatially contiguous river-shaped regions at expected elevation?
2. **Reflectance distribution consistency**: Water class should have lower mean reflectance than land.
3. **Waveform feature distribution consistency**: Water class should have higher n_gaps, larger max_gap_size than land.
4. **Cross-validation with held-out spatial tiles**: Train on one region, predict on another (prevents spatial autocorrelation leakage).
5. **Comparison with NIR point cloud**: Water points from green channel should coincide with data voids or water surface returns from NIR channel (905 nm absorbed by water → NIR absent from underwater zones).
6. **Manual inspection of random sample**: Label ~100–200 points by visual inspection of waveform and spatial context.
7. **Precision-recall trade-off**: At different confidence thresholds, how many "uncertain" points excluded? Auto-labeler tunable.

Reference accuracy: Mandlburger et al. 2025 reports residual vertical errors **< 2 cm** for georeferenced VQ-840-GL vs. total station — 3D geometry very accurate. Classification accuracy limited by rule quality, not sensor precision.

[Source: DOI_10.23784_HN130-06.pdf]

---

## 9. The Pielach River Survey

**Survey details**:
- Location: Neubacher Au, Pielach River, Lower Austria (N 48°12'50", E 15°22'30"; WGS 84)
- Survey date: October 24–25, 2024 (one month after September 2024 flood)
- Water conditions: Good transparency, full riverbed penetration including ~3 m deep pools
- River: Pre-alpine gravel river, riffle-pool type, ~20 m wide, mean annual discharge ~7 m³/s, bed material coarse gravel (2–6.3 cm), gradient ~0.4%
- Catchment area: 590 km²
- Right-hand tributary of Danube in eastern Austria
- Study area within Natura2000 conservation area (AT1219000)
- Contains: river channel, gravel banks, riparian alluvial forest, pasture, meadow

**RIEGL VQ-840-GL (green) survey parameters**:
- Wavelength: 532 nm (green, water-penetrating)
- Flying altitude: 60 m AGL (± 5 m)
- Pulse repetition frequency: 199 kHz
- Beam divergence: 1 mrad → footprint at 60 m = 6 cm diameter
- Scan mechanism: Elliptical Palmer scan, lateral FoV ±20°, forward/backward FoV ±14°
- Point density: >50 pts/m² from sensor specs at similar parameters
- Result: Full penetration to ~3 m depth, vertical accuracy <2 cm vs. total station
- Processing: OWP + SVB algorithm (Schwarz et al. 2019)

**RIEGL miniVUX-3UAV (NIR) survey parameters**:
- Wavelength: 905 nm (NIR, topographic only)
- Flying altitude: 60 m AGL (± 14 m)
- Pulse repetition frequency: 300 kHz
- Beam divergence: 1.5 mrad (1.6 × 0.5 mrad in sensor spec table)
- Flight speed: 6 m/s
- Result: >500 pts/m², used for terrain model + water surface model for refraction correction
- Processing: OWP only

**Coordinate system**: ETRS89/UTM 33N, EPSG:25833, reference epoch 2015.0
**Georeferencing**: 8 saddle-roof reference surfaces. Applied corrections: (-1.2, 1.4, -11.7) cm for VQ-840-GL dataset.

**Waveform data format (confirmed by paper)**:
- Sample interval: ~0.5 ns per SI unit
- Amplitude: ADC units
- Figure 2 in paper: waveforms vary by depth section, ~5 samples (very shallow) to ~20+ samples (deeper), distinct multi-peak structure for water

[Source: DOI_10.23784_HN130-06.pdf, 6.02 Sensors transcript/slides, 5.03 transcript/slides]

---

## 10. Dataset Characteristics (from Phase 4 Analysis)

### Point Cloud (`data/point_cloud_df.txt`)

| Parameter | Value |
|-----------|-------|
| Total points | 234,024 |
| Columns | Unnamed:0 (index), x, y, z, _riegl.reflectance |
| X range | -269.383 to -199.156 m (local offset coordinates) |
| Y range | 97.205 to 138.395 m (local offset coordinates) |
| Z range | 256.538 to 278.453 m |
| Z mean | 264.853 m |
| Z std | 4.587 m |
| Reflectance mean | -21.6 dB |
| Reflectance std | 5.0 dB |
| Reflectance range | -30.77 to -6.32 dB |
| Reflectance 25th percentile | -25.63 dB |
| Reflectance median | -22.34 dB |
| Reflectance 75th percentile | -18.11 dB |

**Reflectance histogram** (20 bins):
```
-30.8 to -29.5 dB: 6,014 points (2.6%)
-29.5 to -28.3 dB: 8,273 points (3.5%)
-28.3 to -27.1 dB: 14,576 points (6.2%)
-27.1 to -25.9 dB: 24,391 points (10.4%)
-25.9 to -24.7 dB: 24,002 points (10.3%)
-24.7 to -23.4 dB: 21,521 points (9.2%)
-23.4 to -22.2 dB: 20,128 points (8.6%)
-22.2 to -21.0 dB: 18,616 points (8.0%)
-21.0 to -19.8 dB: 17,476 points (7.5%)
-19.8 to -18.5 dB: 15,613 points (6.7%)
-18.5 to -17.3 dB: 13,222 points (5.6%)
-17.3 to -16.1 dB: 10,726 points (4.6%)
-16.1 to -14.9 dB:  8,539 points (3.6%)
-14.9 to -13.7 dB:  8,517 points (3.6%)
-13.7 to -12.4 dB:  9,965 points (4.3%)
-12.4 to -11.2 dB:  9,213 points (3.9%)
-11.2 to -10.0 dB:  3,135 points (1.3%)
-10.0 to -8.8 dB:      75 points (0.0%)
 -8.8 to -7.5 dB:      16 points (0.0%)
 -7.5 to -6.3 dB:       6 points (0.0%)
```

Key observation: Roughly unimodal, slightly skewed toward lower (more negative) values, small tail at higher values. Possible two mixed populations: water/low-reflectance land (below -22 dB) and dry land/gravel/vegetation (above -20 dB).

### Waveform Data (`data/waveform_df.txt`)

| Parameter | Value |
|-----------|-------|
| Columns | Unnamed:0 (index), Time [SI], Amplitude [ADC] |
| Format | Numpy array strings (e.g., `[ 37  38  39  40 ...]`) |
| Rows analyzed | 2,000 |
| Valid waveforms | 2,000 |
| Waveform length (samples) | min=19, max=101, mean=59.5 |
| Max amplitude range | 257–3,962 ADC |
| Max amplitude mean | 2,369 ADC |
| ADC system range | 0–8191 (12-bit) |

**Max amplitude histogram** (first 2000):
```
257-628 ADC:   23 (1.1%)
628-998 ADC:   48 (2.4%)
998-1368 ADC: 102 (5.1%)
1368-1739 ADC: 205 (10.2%)
1739-2110 ADC: 290 (14.5%)
2110-2480 ADC: 419 (20.9%)
2480-2850 ADC: 364 (18.2%)
2850-3221 ADC: 339 (16.9%)
3221-3592 ADC: 174 (8.7%)
3592-3962 ADC:  36 (1.8%)
```

**Gap distribution** (time gaps > 2 SI units within waveform):
```
0 gaps:  3 (0.1%)
1 gaps: 67 (3.4%)
2 gaps: 312 (15.6%)
3+ gaps: 1618 (80.9%)
```

**Peak count** (peaks > 100 ADC):
```
1 peak:  7 (0.3%)
2 peaks: 47 (2.4%)
3 peaks: 119 (6.0%)
4 peaks: 244 (12.2%)
5 peaks: 320 (16.0%)
6 peaks: 363 (18.1%)
7 peaks: 352 (17.6%)
8 peaks: 258 (12.9%)
(distribution continues beyond 8)
```

**Waveform sample data (first 5 points)**:
```
Point 0: 63 samples, t=[37-179], max_amp=2836
  T: [37,38,39,...56] | gap(75→80, delta=5) | gap(93→98, delta=5) | GAP(100→173, delta=73)
  A: [62,680,2054,2836,2474,1702,1055,632,355,269,600,1078,1093,883,...] 

Point 1: 51 samples, t=[48-188], max_amp=2923
  gap(50→56,delta=6) | gap(57→66, delta=9) | GAP(98→176, delta=78)

Point 2: 55 samples, t=[47-183], max_amp=1508
  GAP(93→176, delta=83)

Point 3: 59 samples, t=[37-182], max_amp=1796
  gap(48→58, delta=10) | gap(83→94, delta=11) | GAP(103→172, delta=69)

Point 4: 49 samples, t=[38-186], max_amp=3457
  gap(42→58, delta=16) | gap(68→82, delta=14) | GAP(97→169, delta=72)
```

**Critical finding**: Every sample waveform shows characteristic large gap ~70-83 SI units after ~SI time ~93-103, followed by terminal cluster around ~SI time ~170-188. Terminal cluster = BOTTOM RETURN or distant land return. Large gap (70-83 SI × 0.5 ns/SI = 35-41 ns × 300 m/μs / 2 = 5.3-6.2 m air travel) = two-way travel to distant reflector. At 60 m flying altitude, pattern strongly suggests water-penetrating returns: surface near SI ~40-50, water column ~50-100, bottom/deep return near SI ~170-188.

**Waveform sharing**: Only 0.2% of consecutive rows share same waveform data (of first 1000 rows).

---

## 11. Gaps in Knowledge

Not found in available sources, treat as unknown:

1. **Exact reflectance threshold (dB) separating water from land** for RIEGL VQ-840-GL at 532 nm at Pielach River. Must be determined empirically.

2. **Specific ADC threshold for water vs. land peaks**: 100 ADC threshold for peak counting is heuristic; actual noise floor not specified.

3. **SVB algorithm implementation details**: Described in Schwarz et al. 2017 and 2019 (not in provided materials). Exact parameters unknown.

4. **Water vs. land ratio in dataset**: ~20 m wide river in broader landscape, but exact class balance in 234,024-point dataset unknown without labels.

5. **OWP vs. SVB classification per point**: `point_cloud_df.txt` doesn't indicate whether each point was extracted by OWP or SVB. SVB points more likely to be water-column or bottom.

6. **Turbidity and k-value for October 2024 survey**: k not given numerically. Paper states water clarity was "good."

7. **Exact ADC noise floor**: 12-bit range (0–8191). Noise floor and saturation levels not explicitly stated.

8. **Whether waveforms are pre-OWP raw or post-OWP**: `waveform_df.txt` appears to contain raw digitized waveforms (non-contiguous time arrays consistent with "sample datagrams" format in echo detection slides).

9. **Direct numerical reflectance thresholds from literature for water at 532 nm**: No specific dB threshold found in sources.

---

## 12. Source Reference Table

| Source | Type | Relevance | Key Content |
|--------|------|-----------|-------------|
| DOI_10.23784_HN130-06.pdf | Paper | **CRITICAL** | Exact paper on this dataset. VQ-840-GL specs, OWP+SVB, ETRS89/UTM 33N EPSG:25833, October 2024 Pielach survey, waveform format (0.5 ns, ADC units), vertical accuracy <2 cm |
| LIDARMagazine Part3 | Paper | Very High | ALB physics, green laser-water interaction, PWS+PWC+PWB decomposition, specular reflection, k coefficient, shallow zone problem, refraction correction |
| LIDARMagazine Part2 | Paper | High | KPConv classification with green+NIR, dual-wavelength advantage for water/land separation |
| LIDARMagazine Part4 | Paper | High | UAV LiDAR details, full-waveform digitization in survey-grade sensors |
| LIDARMagazine Part1 | Paper | Medium | LiDAR basics, ranging, waveform analysis |
| 5.01 Topo-Bathy_Measurement principle | Transcript+Slides | Very High | Water signal physics: 300,000 vs 225,000 km/h, Snell's law, exponential decay, k coefficient, Secchi depth |
| 5.02 Topo-Bathy_Sensor overview | Transcript+Slides | Very High | Sensor categories, minimum depth 20 cm, VQ-840-G specs (1-6 mrad, 50-200 kHz, 2×Secchi) |
| 5.03 Topo-Bathy_Application examples | Transcript+Slides | Very High | Pielach River examples, OWP vs SVB improvement, VQ-840-G performance |
| 6.02 UAV Sensors and platforms | Transcript+Slides | High | VQ-840-G detailed specs, miniVUX-3UAV specs |
| 1.06 Basics_Echo detection | Transcript+Slides | High | Gaussian decomposition, echo attributes, ADC range 0-8191 |
| 2.01 Multispectral_Laser Radar Equation | Transcript | High | Specular reflection physics, water surface data voids, backscattering solid angle |
| 1.05 Basics_Laser beam | Transcript | Medium | Beam divergence, footprint size, eye safety, minimum depth for surface/bottom separation |
| 1.10 Basics DTM generation | Transcript+Slides | Medium | Deep learning architectures: PointNet++, DGCNN, 3D Sparse Voxel CNN, Random Forest |
| 2.02 Multispectral Radiometric Calibration | Transcript | Medium | Reflectance calibration: amplitude × sigma ∝ received power |
| Dataset analysis (Phase 4) | Data | **CRITICAL** | 234,024 points, reflectance -30.8 to -6.3 dB, waveforms 19-101 samples, 80.9% have 3+ gaps, large terminal gaps ~70-83 SI units |