# Water vs. Land Classification from Full-Waveform Bathymetric LiDAR — Knowledge Base

---

## 1. Problem Overview

The goal is to train a supervised ML model to classify **water** vs. **land** points from full-waveform topo-bathymetric LiDAR point cloud data. Because no manual labels exist, the workflow is:

1. Build a **rule-based auto-labeler** grounded in domain physics (this knowledge base)
2. Apply it to generate noisy pseudo-labels on the Pielach River dataset
3. Train a supervised model (with label noise robustness) on the pseudo-labels
4. Iterate: use the model's confident predictions to refine labels

The scanner is a **RIEGL VQ-840-GL** (green, 532 nm) paired with a **RIEGL miniVUX-3UAV** (NIR, 905 nm). The dataset is a ~750 m reach of the pre-alpine Pielach River (Austria, October 2024), downsampled to ~234,024 points at ~15 cm spacing.

[Source: DOI_10.23784_HN130-06.pdf, 5.01 transcript, 5.03 transcript]

---

## 2. Physics of Bathymetric LiDAR Waveforms

### 2.1 Green laser (532 nm) interaction with water

Water exhibits **minimum optical attenuation** in the blue-green spectrum (460–550 nm). The 532 nm wavelength is produced by frequency-doubling a Nd:YAG infrared laser (1064 nm). This "atmospheric window" in water allows green photons to penetrate the water column, reach the bottom, and return to the sensor — which is physically impossible with NIR (~905/1064 nm) wavelengths.

Key physics: the total received power is the sum of contributions from water surface (PWS), water column (PWC), water bottom (PWB), and background (PBK):
```
PR = PWS + PWC + PWB + PBK
```
[Source: 5.01 Topo-Bathy_Measurement principle.pdf, LIDARMagazine Part3]

### 2.2 Water surface returns (specular/diffuse reflection)

At the air–water interface, **part of the green signal is reflected** back from the water surface and **part is refracted** into the water column. The fraction reflected depends on:
- **Surface roughness**: Slightly ruffled water (ripples) scatters more signal back toward the sensor than a perfectly flat mirror surface.
- **Incidence angle**: Near-nadir laser beams hitting water produce very strong backscatter (mirror-like reflection) which can **saturate** the receiver. For this reason, bathymetric scanners use **conical (Palmer) scanning at a constant off-nadir angle of ~15–20°**.
- **Specular vs. diffuse**: Water surface is a **predominantly specular reflector**. At steep off-nadir angles, nearly all signal is reflected away from the sensor, producing **data voids**. Only when the surface is slightly ruffled does the detector receive a signal.

The backscattering solid angle omega is **very small** for specular targets. This is why water surfaces often show data voids in LiDAR: the direct (specular) reflection goes away from the receiver.

The water surface returns are characterized by:
- Typically the **first and strongest peak** in the waveform
- Relatively low amplitude in the raw waveform (compared to land) because of the oblique scan angle
- High spatial variability (missing in very smooth water, especially standing water)

[Source: 2.01 Multispectral_Laser Radar Equation transcript, 5.01 transcript, LIDARMagazine Part3, 5.01 Measurement principle slides]

### 2.3 Water column backscatter and exponential decay

Within the water column, the laser interacts with suspended sediment particles and water molecules. The signal is both **scattered and absorbed**. The received power from the water column follows **exponential decay**:

```
PWC(z) ∝ exp(-2kz / cos(αW))
```

where:
- z = water depth
- k = diffuse attenuation coefficient (characterizes turbidity)
- αW = water-sided incidence angle (after refraction)

This produces a **continuous, decaying signal** between the water surface peak and the bottom peak in the waveform. This is fundamentally different from land, which produces sharp, isolated Gaussian-shaped peaks.

The exponential water column backscatter is the basis for the **SVB (Surface-Volume-Bottom) algorithm**: the "Volume" component captures this exponential tail and distinguishes it from sharp solid surface returns.

Volume backscattering causes the amplitude to drop **asymmetrically** after the water surface peak — a distinctive signature that can be used to identify water column traversal.

[Source: 5.01 Topo-Bathy_Measurement principle slides, LIDARMagazine Part3, 5.03 transcript]

### 2.4 Bottom returns through water

If water is clear enough, the laser reaches the bottom and reflects off the riverbed/seabed:
- **Gravel and light sand**: high reflectivity → favors deeper penetration
- **Muddy soil or dark submerged vegetation**: low reflectivity → reduces penetration depth
- The Pielach River has **coarse gravel (2–6.3 cm)** as bed material → relatively high bottom reflectance

The bottom return is typically a **broad Gaussian-shaped peak** that appears later in the waveform, separated from the surface return by the two-way travel time through the water column. The bottom echo is typically **weaker** than the surface echo due to absorption and scattering losses.

**Peak separation** between surface and bottom peaks is directly proportional to water depth:
```
depth = (delta_t_SI * 0.5e-9 * c_water) / 2
c_water = 225,000,000 m/s (225,000 km/s)
```
For 1 SI unit = 0.5 ns: delta_z_per_SI = 0.5e-9 * 225e6 / 2 ≈ 0.056 m per SI unit (water-side), or about **5.6 cm per SI unit of peak separation** (corrected for speed of light in water).

**Shallow water problem**: If water depth < ~20 cm (for shallow bathy systems like VQ-840-GL), the surface and bottom echoes **cannot be separated** because the pulse length (1.5 ns ≈ 22.5 cm in water) is longer than the two-way travel time. The minimum separable depth is half the pulse length in water.

[Source: 5.01 transcript, 5.02 Sensor overview slides, LIDARMagazine Part3, 1.03 transcript]

### 2.5 NIR laser (905 nm) interaction with water (for comparison)

The **miniVUX-3UAV** operates at **905 nm (near-infrared)**. At this wavelength, water has very high absorption — the NIR laser does **not penetrate** water at all. Instead:
- NIR produces only a **single water surface return** (from specular reflection at the air–water interface)
- NIR produces **no bottom returns**
- The miniVUX-3UAV is therefore used as the **topographic scanner** for dry land and to obtain the water surface model for refraction correction of the green channel data

This key difference is exploited in two-channel systems: the NIR channel reliably detects the water surface (when hitting at nadir), while the green channel provides both surface and bottom returns.

[Source: 5.01 transcript, 5.02 transcript, DOI_10.23784_HN130-06.pdf, LIDARMagazine Part3]

### 2.6 Land and vegetation waveform characteristics

**Dry land / hard surfaces** (gravel, concrete, asphalt):
- Single, narrow, roughly symmetric Gaussian-shaped peak
- High amplitude (extended target)
- Short echo width (sharp, clear return)
- No exponential decay component after peak

**Vegetation** (trees, shrubs, grass):
- Multiple peaks from canopy layers, branches, and ground
- Broadened echo width (due to vertical extent of vegetation)
- Decreasing amplitude with penetration depth
- Echo width much larger than hard surfaces

**Buildings/roofs**:
- Single sharp peak, high amplitude, short echo width

**Key waveform shape differences**:
- Water: flat/mixed pulse with exponential decay after first peak, possible separated bottom peak
- Land: distinct narrow Gaussian peak(s) with no exponential decay component
- Vegetation: multiple broad peaks with varying amplitudes

[Source: 1.02 transcript, 1.06 transcript, 2.02 transcript, 1.10 transcript]

### 2.7 Reflectance properties by surface type

Reflectance values at 532 nm (note: the values below are for 900 nm — green reflectance is generally lower for most materials):

From the laser radar equation slide (at λ=900 nm, used as reference):
- White paper: up to 100%
- Snow: 80–90%
- Deciduous trees: ~60%
- Coniferous trees: ~30%
- Dry carbonate sand: ~57%
- Asphalt: ~17%
- Black rubber: ~2%

Important context for green (532 nm):
- Most surfaces reflect **less** at 532 nm than at NIR (the green image appears "darker" overall)
- Water column with gravel bottom has **intermediate to low reflectance** at 532 nm
- The reflectance value in `_riegl.reflectance` column is in dB and is **negative** (indicating low-to-moderate reflectors)

From dataset analysis:
- Reflectance range: -30.8 to -6.3 dB
- Mean: -21.6 dB, Median: -22.3 dB, StdDev: ~5.0 dB
- **Lower dB = lower reflectance** → likely water/deep zones
- **Higher dB (less negative) = higher reflectance** → likely land/gravel/vegetation

The bimodal potential in the reflectance histogram (peaking around -27 to -23 dB and a secondary cluster around -13 to -11 dB) may correspond to water vs. land classes.

[Source: 2.01 transcript, 2.01 slides, LIDARMagazine Part2, dataset analysis]

---

## 3. Waveform Processing Algorithms

### 3.1 Online Waveform Processing (OWP)

OWP (Pfennigbauer et al. 2014) processes waveforms **in real-time during the flight**. It uses a **reduced set of the recorded waveforms around signal peaks** to extract points online. The OWP approach:
1. Detects local maxima (peaks) in the digitized waveform above a noise threshold
2. Fits Gaussian curves to detected peaks (Gaussian decomposition)
3. Extracts per-echo attributes: range (temporal position μ), amplitude (A), echo width (σ)
4. Outputs as discrete 3D points with attributes

OWP is standard for the miniVUX-3UAV (NIR) data. For the VQ-840-GL (green), OWP is used as the first processing pass.

**Limitation**: OWP may miss weak bottom returns in turbid or deep water because the detection threshold can suppress the weak exponential tail of water column backscatter.

[Source: DOI_10.23784_HN130-06.pdf, 5.03 transcript, 3.1 LiDAR workflow section]

### 3.2 Surface-Volume-Bottom (SVB) algorithm

SVB (Schwarz et al. 2019) is the **post-processing, water-specific full-waveform algorithm** applied to green channel data. It is the key algorithm that unlocks deeper penetration than OWP alone.

SVB decomposes the waveform into three components:
1. **Surface**: Gaussian peak at the water surface reflection
2. **Volume**: Exponential decay component representing water column backscattering
3. **Bottom**: Gaussian peak at the riverbed/seabed reflection

The critical innovation is the **exponential decomposition** for the water column (Schwarz et al. 2017): instead of trying to fit a Gaussian to the water column signal (which would fail because it follows exponential decay, not a Gaussian), SVB explicitly models the exponential component.

Results for Pielach 2024:
- OWP produced standard points for clear water sections
- SVB detected additional bottom points especially in **deeper and more turbid** areas
- Combined result: full penetration of entire riverbed including ~3 m deep pool areas
- SVB points complement OWP points, with SVB extending reach in challenging conditions

The waveform data in the dataset (`waveform_df.txt`) contains the raw full-waveform data needed for SVB-style analysis.

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
1. Detect local maxima above noise threshold → determines number of Gaussian components (model complexity)
2. Apply non-linear parameter estimation (Levenberg-Marquardt) to fit Gaussian curves
3. Extract attributes per echo: (A_i, mu_i, sigma_i)

**Attributes available per echo**:
- Amplitude (intensity/signal strength)
- Range R (from mu_i × c/2)
- Echo width w (= sigma_i, in ns)
- Backscatter cross-section (combination of amplitude and echo width)

**Echo width interpretation**:
- Small sigma = sharp, narrow peak → hard surface (road, roof, water bottom in clear shallow water)
- Large sigma = broad, wide peak → vegetation, mixed/penetrated surfaces, water

[Source: 1.06 transcript, 1.06 slides, 2.02 transcript]

### 3.4 Exponential decomposition for water column

The water column backscatter follows:
```
PWC(z) ∝ exp(-2kz / cos(αW))
```

This means the waveform amplitude after the water surface peak decays **exponentially** with increasing depth/time. The SVB algorithm explicitly models this exponential component (Schwarz et al. 2017), separating:
- The exponential volume backscatter tail (water column)
- Any superimposed Gaussian peaks (surface and bottom echoes)

In the raw waveform data, this appears as:
- A peak from water surface
- A slowly decaying amplitude between surface and bottom peak
- A second peak (possibly weak) from the bottom

This exponential decay region in the time-domain is a diagnostic signature of water column traversal, distinguishable from land returns which show no such decay.

[Source: DOI_10.23784_HN130-06.pdf, 5.01 slides, LIDARMagazine Part3]

---

## 4. Feature Engineering

### 4.1 Waveform-derived features (with computation methods)

All features computed from the raw time-series (Time [SI], Amplitude [ADC]):

**Amplitude features**:
- `max_amplitude`: Maximum ADC value in waveform
- `mean_amplitude`: Mean of non-zero samples
- `amplitude_at_first_peak`: ADC value at first local maximum above threshold
- `first_peak_time`: SI units of first peak
- `last_peak_time`: SI units of last peak above threshold

**Peak structure features**:
- `n_peaks`: Count of local maxima above threshold (e.g., >100 ADC)
- `peak_spacing_mean`: Mean distance between consecutive peaks (in SI units)
- `max_peak_spacing`: Maximum gap between any two consecutive peaks
- `first_to_last_peak_span`: Span in SI units from first to last peak

**Echo width (pulse broadening)**:
- `sigma_first_peak`: Gaussian sigma of first peak (Gaussian decomposition)
- `sigma_primary_peak`: Gaussian sigma of the strongest peak
- `full_width_half_max`: FWHM of strongest peak

**Energy/Area features**:
- `total_energy`: Sum of all amplitude values (Area Under Curve, AUC)
- `energy_after_first_peak`: Energy in waveform portion after first peak → proxy for water column backscatter
- `exponential_decay_coefficient`: Estimated k from fitting exp(-alpha*t) to post-first-peak amplitude

**Shape features**:
- `waveform_skewness`: Skewness of amplitude distribution across samples
- `waveform_kurtosis`: Kurtosis of amplitude distribution
- `asymmetry_ratio`: Energy before vs. after peak

**Gap structure** (key for non-contiguous time arrays):
- `n_gaps`: Number of gaps > 2 SI units between consecutive time samples
- `max_gap_size`: Largest gap in SI units between consecutive samples
- `total_gap_extent`: Sum of all gap sizes
- `time_span`: last_time - first_time in SI units
- `gap_ratio`: total_gap_extent / time_span

**Multi-return indicators**:
- `n_clusters`: Number of distinct continuous time-segments (clusters separated by gaps)
- `second_cluster_max_amp`: Maximum amplitude in second cluster (potential bottom return)
- `bottom_surface_amplitude_ratio`: A_bottom / A_surface (if two clear peaks identified)

**Reflectance** (from point cloud, not raw waveform):
- `reflectance_dB`: From `_riegl.reflectance` column (already calibrated)

[Source: 1.06 transcript, 1.06 slides, 2.02 transcript, dataset analysis]

### 4.2 Point cloud-derived features (geometric, neighborhood)

For each point, compute from k-nearest neighbors or radius search:

**Height-based**:
- `z` (absolute elevation in ETRS89/UTM 33N)
- `z_relative_to_local_mean`: Height relative to local neighborhood mean
- `height_above_lowest_neighbor`: z - min(z in radius)

**Planarity and roughness**:
- Eigenvalue decomposition of local covariance matrix → λ1 ≥ λ2 ≥ λ3
- `planarity` = (λ2 - λ3) / λ1
- `roughness` = λ3 (smallest eigenvalue, measures local surface roughness)
- `linearity` = (λ1 - λ2) / λ1
- `sphericity` = λ3 / λ1

**Normal vector**:
- Normal vector components (nx, ny, nz) from PCA of local neighborhood
- `normal_z`: Z-component of normal vector (close to 1.0 for flat horizontal surfaces)
- `incidence_angle`: Angle between laser beam and local surface normal

**Density**:
- `point_density_radius`: Count of points within radius R
- `point_density_k`: 1 / mean_distance_to_k_neighbors

**Neighborhood height statistics**:
- `height_variance_local`: Variance of z in neighborhood
- `height_range_local`: Range of z in neighborhood

[Source: 1.10 transcript, 1.06 transcript, 2.03 transcript]

### 4.3 Most discriminative features for water vs. land with green laser

Based on domain physics, the most discriminative features in order of expected importance:

1. **n_gaps / n_clusters**: Water waveforms show MANY gaps (time-discontinuities) because the waveform is stored as the cluster around each detected peak. Multiple clusters = multiple returns (surface + column + bottom). Land shows fewer distinct clusters. **Observed in data: 80.9% of first 2000 waveforms have 3+ gaps.**

2. **max_gap_size**: Water waveforms have a characteristic very large gap (~70-90 SI units, ~35-45 ns, ~3.9-5.0 m in air) between surface return cluster and the deep return cluster. This large gap = time for light to travel down through water and back.

3. **peak_spacing / time_span**: For water, the total waveform time span (first_time to last_time) is much larger than for land at equivalent depth, because the water column adds travel time.

4. **reflectance_dB**: Water has lower reflectance (more negative dB) than land (gravel, vegetation). Dataset range: -30.8 to -6.3 dB; water likely clusters in -30 to -20 dB range.

5. **energy_after_first_peak / exponential_decay**: Water shows significant energy after the first peak (water column backscatter). Land shows rapid drop.

6. **n_peaks**: Multi-peak waveforms (especially 3+) with wide spacing strongly indicate water (surface + possibly intermediate + bottom). However, vegetation also gives multi-peak, so combined with height info needed.

7. **roughness / planarity**: Water surface should be smoother than land (lower roughness, higher planarity). River bank (gravel) will be rougher than the water surface.

8. **height_above_lowest_neighbor**: Water points near or below the water surface level cluster in a consistent elevation band.

9. **sigma_first_peak / echo_width**: Water bottom returns may have broader echo width due to footprint widening in water (forward scattering).

10. **second_cluster_max_amp / bottom_surface_ratio**: Presence of a second-cluster peak well-separated from the first is a strong water indicator.

[Source: 5.01 transcript, 5.03 transcript, LIDARMagazine Part3, 2.01 transcript, dataset analysis]

### 4.4 How to handle variable-length waveform input

The waveforms in `waveform_df.txt` have variable lengths (observed: min=19, max=101 samples, mean=59.5 samples) AND non-contiguous time arrays (gaps between clusters).

Key insight: The time array is **non-contiguous** — it stores only sample-blocks around detected peaks. So you cannot simply treat it as a dense 1D signal. Two strategies:

**Strategy A: Hand-engineered features (recommended for auto-labeler)**
- Extract scalar features as listed in 4.1
- Feed to traditional ML (Random Forest, XGBoost) or simple MLP
- Avoids variable-length problem entirely

**Strategy B: Fixed-length dense embedding (for deep model)**
- Define a global time grid (e.g., 0–200 SI units at 0.5 ns resolution = 400 bins)
- Place amplitude values into grid cells; unfilled cells = 0 (or noise floor)
- Results in a consistent 400-bin 1D vector
- Can feed to 1D CNN
- Advantage: preserves temporal position information (peaks at consistent times = consistent depth)

**Strategy C: Cluster-level representation**
- Extract per-cluster statistics: [time_offset, max_amp, mean_amp, width, sigma]
- Pad to fixed number of clusters (e.g., max 5 clusters)
- Masking for variable-length cluster lists (Transformer-style attention)

Strategy B is recommended for the deep model because it preserves the key discriminative signal: the timing/position of peaks relative to the emission time.

[Source: dataset analysis, DOI_10.23784_HN130-06.pdf Figure 2 description]

---

## 5. Classification Approaches

### 5.1 Traditional ML methods

For the baseline model and auto-labeler validation:
- **Random Forest**: robust, handles mixed features (waveform scalars + geometric), interpretable feature importance, handles class imbalance via class_weight, no feature normalization needed
- **Gradient Boosting (XGBoost/LightGBM)**: excellent for tabular data, fast training
- **SVM**: effective if features are well-normalized

[Source: 1.10 slides, 2.03 transcript]

### 5.2 Deep learning for point clouds

For spatial classification:
- **PointNet++**: Multi-scale local feature extraction from 3D point clouds. Applied to ALS data (Winiwarter et al. 2018). Can incorporate waveform features per point.
- **DGCNN (Dynamic Graph CNN)**: Dynamic graph convolution; captures local structure. Applied to ALS (Widyaningrum et al. 2021).
- **KPConv**: Kernel point convolution; shown to achieve near-perfect classification with green+NIR dual-wavelength (XYZ+G+NIR vs. geometry alone). [Source: LIDARMagazine Part2]
- **3D Sparse Voxel CNN** (Schmohl et al. 2019): Voxel-based, efficient for sparse 3D data

[Source: 1.10 transcript, 1.10 slides, LIDARMagazine Part2, 2.03 transcript]

### 5.3 Deep learning for 1D waveform signals

For direct waveform classification:
- **1D CNN**: Most natural choice for fixed-length 1D signal. Can learn multi-scale features (narrow peaks, broad exponential tails). Multiple kernel sizes to capture features at different temporal scales.
- **RNN/LSTM**: Handles sequential data, but struggles with the non-contiguous time structure and long gaps.
- **Transformer**: Self-attention can attend to relevant parts of the sequence, robust to non-contiguity if positional encoding is time-based rather than position-index-based.

**Recommended approach**: 1D CNN with time-grid embedding (Strategy B from 4.4). The non-contiguous nature means RNNs process "gap=0" samples as if they were real signal — misleading. CNNs on sparse grids can be made robust via proper padding treatment.

### 5.4 Multi-modal architectures

The dataset has both point cloud attributes and raw waveforms. A multi-modal architecture:
1. **Waveform branch**: 1D CNN or feature extractor → waveform embedding vector
2. **Point cloud branch**: MLP on (x, y, z, reflectance, geometric features) → spatial embedding vector
3. **Fusion**: Concatenate embeddings → classification head

This matches the dual-sensor setup: green waveform features capture water vs. land physics; spatial features (z, roughness, planarity) provide geometric context.

[Source: LIDARMagazine Part2, dataset analysis]

### 5.5 Unsupervised and semi-supervised approaches

**Bootstrap approach (this project)**:
1. Auto-label with rules → ~50–70% confident labels
2. Train supervised model on confident labels
3. Apply model to remaining points → propagate labels
4. Retrain with expanded training set

**Clustering for exploration**:
- K-means or DBSCAN on (reflectance, n_gaps, max_gap, z) to find natural clusters
- UMAP/t-SNE for visualization of feature space

### 5.6 Training strategies and best practices

- **Class imbalance**: Pielach River is a narrow river (~20 m wide) in a broader landscape. Water points are likely a minority. Use `class_weight='balanced'` or focal loss.
- **Noisy labels**: Use label smoothing, confident learning (Cleanlab), or loss-correction methods.
- **Cross-validation**: Hold out one spatial strip (not random split) to test generalization to unseen areas.
- **Feature normalization**: Z-score normalize all features except those with physical meaning (dB values, SI units).

[Source: DOI_10.23784_HN130-06.pdf, dataset analysis, general ML best practices]

---

## 6. Auto-Labeling Rules Catalog

### 6.1 Reflectance-based rules (with thresholds)

From dataset analysis, reflectance distribution:
- Range: -30.8 to -6.3 dB
- Mean: -21.6 dB, Median: -22.3 dB
- Distribution: roughly bell-shaped with tail at higher (less negative) values

**Rule R1**: Reflectance below -25 dB → candidate WATER (LOW reflectance)
- Rationale: Water has low reflectance at 532 nm. Very negative dB values (e.g., below -25 dB) more likely indicate water surface or submerged bottom.
- Fraction of data: ~22% of points have reflectance below -25 dB

**Rule R2**: Reflectance above -15 dB → candidate LAND (HIGH reflectance)
- Rationale: Gravel, meadow, and dry vegetation have higher reflectance at 532 nm than water.
- Fraction of data: ~9.5% of points with reflectance above -15 dB

**Rule R3**: Reflectance in range -25 to -15 dB → AMBIGUOUS (requires other features)

**Important caveat**: These thresholds are informed estimates based on the dataset statistics and general physics. The actual decision boundary depends on the specific material properties and illumination conditions. Wet soil may have low reflectance similar to water.

[Source: 2.01 transcript, 2.02 transcript, dataset analysis]

### 6.2 Waveform shape rules (with criteria)

**Rule W1** (Multi-cluster indicator): `n_gaps >= 2` AND `max_gap_size >= 50 SI units (~25 ns, ~4.2 m in air travel)`
- Rationale: Water produces distinct surface and bottom return clusters with a large temporal gap between them representing the two-way travel time through air+water. The large gap (>50 SI units) corresponds to air-column travel time above the water surface, not achievable from a single flat land surface.
- From dataset: 80.9% of first 2000 waveforms have 3+ gaps; the typical large gap (Points 0-4) is ~70-83 SI units (~35-41 ns).
- Label: **Candidate for water or deep vegetation**

**Rule W2** (Exponential tail indicator): Energy in second half of waveform cluster relative to first half > 0.3
- Rationale: Water column backscatter produces sustained energy after the first peak. Land produces sharp peaks that decay quickly.
- Label: **Candidate for water column or bottom**

**Rule W3** (Separated double peak): Two peaks > 100 ADC with temporal separation > 15 SI units (~7.5 ns, ~0.85 m water depth)
- Rationale: Direct water surface + bottom return signature.
- Time-to-depth: 15 SI × 0.5 ns/SI = 7.5 ns → 7.5e-9 s × 225e6 m/s / 2 ≈ 0.84 m water depth
- Label: **HIGH CONFIDENCE WATER**

**Rule W4** (Single sharp peak): Only 1 peak > 100 ADC AND total waveform spans < 15 SI units AND no significant energy elsewhere
- Rationale: Clean single return from solid surface (land)
- Label: **Candidate for LAND**

**Rule W5** (Very long waveform): Total time span (last_time - first_time) > 100 SI units (50 ns)
- Rationale: Long time spans indicate multiple reflections at different depths. This can mean deep water, multi-layer vegetation, or buildings. At 60 m AGL, the sensor-to-ground distance corresponds to ~400 SI units, so waveform span of 100+ SI units (excluding the emission time) suggests multiple discrete returns.
- Note: In the dataset, time values range from ~37 to ~188 SI units (span ~150 SI); the structure is heavily gapped. The large terminal gap of ~70-83 SI appears to be the gap BETWEEN the last surface/column return and the final return (possibly bottom at depth).
- Label: **Requires interpretation with other rules**

[Source: dataset analysis (waveform sample analysis), 5.01 transcript, 5.03 transcript, LIDARMagazine Part3]

### 6.3 Amplitude-based rules (with thresholds)

From dataset analysis on first 2000 waveforms:
- max_amplitude range: 257–3962 ADC
- mean max_amplitude: 2369 ADC
- ADC range: 0–8191 (12-bit ADC, from echo detection slide)

**Rule A1**: max_amplitude > 3000 ADC → candidate LAND
- Rationale: Strong returns (high amplitude) typically indicate diffuse, high-reflectance surfaces like dry gravel/meadow viewed at moderate incidence angle. Water surface viewed at off-nadir gives weaker return due to specular directionality.
- Caveat: Shallow water with high-reflectance bottom can also give strong returns.

**Rule A2**: max_amplitude < 800 ADC → candidate DEEP WATER or ambiguous
- Rationale: Weak returns indicate either absorption (water column), deep water, or dark surfaces.

**Rule A3**: First peak amplitude > second peak amplitude by factor of 3 or more
- Rationale: For water, the bottom return is typically much weaker than the surface return due to absorption losses in the water column (exponential decay). For land with vegetation, multiple peaks can have similar amplitudes.
- Label: Supports WATER identification when combined with spatial separation

[Source: dataset analysis, 5.01 transcript, LIDARMagazine Part3]

### 6.4 Multi-peak / echo-based rules

**Rule M1** (High peak count): `n_peaks >= 4` with at least one peak separation > 20 SI units
- Rationale: Water with multiple return clusters. From dataset: 5+ peaks in 18.1% of waveforms, 6+ peaks in ~36% — this is common for the water-rich Pielach scene.
- Label: Candidate WATER when peaks are widely separated; candidate VEGETATION when peaks are closely spaced.

**Rule M2** (Vegetation signature): `n_peaks >= 3` AND all inter-peak gaps < 15 SI units AND total span < 50 SI units
- Rationale: Dense vegetation produces multiple closely-spaced peaks within a short time window.
- Label: **Candidate VEGETATION (not water)**

**Rule M3** (Clean single echo): `n_peaks == 1` AND max_gap_size < 5 SI units
- Rationale: Single clean echo from solid flat surface.
- Label: **Candidate LAND (high confidence)**

**Rule M4** (Water SVB signature): Waveform has:
- First cluster: 1-2 peaks in SI range [37-80], followed by
- Large gap (50+ SI units), followed by
- Second cluster: 1-2 peaks in SI range [150-200]
- This pattern: first cluster = water surface return, gap = travel time to bottom, second cluster = bottom return
- This is EXACTLY what is observed in sample waveforms 0-4 from the dataset!
- Label: **HIGH CONFIDENCE WATER**

[Source: dataset analysis, 5.01 transcript, LIDARMagazine Part3]

### 6.5 Elevation and geometry rules

**Rule E1** (Z-based water zone): Points within known or estimated river channel elevation range
- The Pielach River in the Neubacher Au study area has a specific elevation range
- From dataset: Z range is 256.5–278.5 m, mean 264.9 m
- Points with very low Z (bottom of channel) in conjunction with other water indicators → high confidence water

**Rule E2** (Planarity): Points in spatially smooth regions (low local roughness) near expected water elevation → candidate WATER SURFACE
- River surface is flat → low roughness, high planarity
- Gravel bank is rough → higher roughness

**Rule E3** (Height above DTM): Once a preliminary DTM is derived, points significantly BELOW the terrain model → indicate water column or bottom → candidate WATER

**Rule E4** (Z clustering): Water surface points should cluster in a narrow Z-range. If neighboring points all have similar Z values within a few cm → likely water surface.

[Source: 5.03 transcript, dataset analysis]

### 6.6 Spatial context rules

**Rule S1** (Spatial connectivity): Water points form spatially connected regions. If a point is labeled WATER by other rules and has nearby neighbors also labeled WATER → increase confidence.

**Rule S2** (NIR data voids): Points that lack corresponding NIR returns (from miniVUX-3UAV) over water-elevation zones → likely water (NIR is absorbed by water, creating data voids). This requires the NIR point cloud to be available for comparison.

**Rule S3** (River corridor): Points within the geometric footprint of the river channel (estimated from water surface elevation model or DEM) → prior weight toward WATER.

**Rule S4** (Vegetation height): Points at Z significantly above terrain level (> 0.5 m above DTM) are vegetation or structures → weight against both water and bare land.

[Source: 5.01 transcript, 5.03 transcript, DOI_10.23784_HN130-06.pdf]

### 6.7 Combined rule confidence scoring

Assign each point a confidence score for WATER class (0–1):

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

1. **Shallow water zone (< 20 cm depth)**: Surface and bottom echoes cannot be separated → waveform looks like a single return from solid surface → may be mislabeled as LAND. From sensor specs: minimum separable depth ~20 cm for VQ-840-GL.

2. **Very smooth water surface (sun glint / calm day)**: Near-nadir specular reflection → very strong first peak, no penetration → may look like land. Palmer scan at ±20° helps avoid nadir, but calm inland rivers on clear days are a known issue.

3. **Wet soil / puddles**: Low reflectance at 532 nm, possible smooth surface → may be misclassified as water. Need geometry (depth, roughness) to distinguish.

4. **Dark surfaces on land** (dense wet vegetation, dark asphalt): Low reflectance → may pass reflectance-based water rules. Need multi-peak and elevation features.

5. **Semi-submerged vegetation** (macrophytes, aquatic plants): Water bottom covered with vegetation → has water signature (surface + water column) but bottom return is from plants, not gravel → complex waveform.

6. **Survey conducted October 2024**: Water clarity was good (one month after September 2024 flood). Different water turbidity conditions would change the waveform structure significantly.

7. **Waveform-to-point mapping**: Multiple points can share the same waveform (one waveform → multiple extracted echoes = multiple rows in point_cloud_df.txt sharing same waveform). In waveform_df.txt, 0.2% of consecutive rows share the same waveform data (checked first 1000 rows).

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
- Keep reflectance_dB as-is (already physically meaningful)
- Time-based features (SI units) can be kept in SI or normalized

### Step 6: Handle coordinate system
- Coordinates are in ETRS89/UTM 33N (EPSG:25833) with local offsets: x ≈ -269, y ≈ 97, z ≈ 261-264
- Do NOT add large absolute x,y,z as model features — use relative/local coordinates
- z is the most useful absolute feature (elevation above reference)

[Source: DOI_10.23784_HN130-06.pdf, dataset analysis]

---

## 8. Evaluation Strategy

Without ground truth labels, evaluation must be indirect:

1. **Physical plausibility**: Do labeled water points form spatially contiguous river-shaped regions at expected elevation?
2. **Reflectance distribution consistency**: Water class should have lower mean reflectance than land class.
3. **Waveform feature distribution consistency**: Water class should have higher n_gaps, larger max_gap_size than land class.
4. **Cross-validation with held-out spatial tiles**: Train on one spatial region, predict on another (prevents data leakage from spatial autocorrelation).
5. **Comparison with NIR point cloud**: Water points from green channel should spatially coincide with data voids or water surface returns from NIR channel (905 nm is absorbed by water → NIR data is absent from underwater zones).
6. **Manual inspection of a random sample**: Label ~100–200 points by visual inspection of the waveform and spatial context.
7. **Precision-recall trade-off**: At different confidence thresholds, how many "uncertain" points are excluded? The auto-labeler should be tunable.

Reference accuracy: The Mandlburger et al. 2025 paper reports residual vertical errors **< 2 cm** for the georeferenced VQ-840-GL point cloud vs. total station measurements — so the 3D geometry is very accurate. Classification accuracy will be limited by the rule quality, not sensor precision.

[Source: DOI_10.23784_HN130-06.pdf]

---

## 9. The Pielach River Survey

**Survey details**:
- Location: Neubacher Au, Pielach River, Lower Austria (N 48°12'50", E 15°22'30"; WGS 84)
- Survey date: October 24–25, 2024 (one month after September 2024 flood)
- Water conditions: Good transparency, enabled full riverbed penetration including ~3 m deep pools
- River characteristics: Pre-alpine gravel river, riffle-pool type, ~20 m wide, mean annual discharge ~7 m³/s, bed material coarse gravel (2–6.3 cm), gradient ~0.4%
- Catchment area: 590 km²
- Right-hand tributary of the Danube in eastern Austria
- Study area within Natura2000 conservation area (AT1219000)
- Area contains: river channel, gravel banks, riparian alluvial forest, pasture, meadow

**RIEGL VQ-840-GL (green) survey parameters**:
- Wavelength: 532 nm (green, water-penetrating)
- Flying altitude: 60 m AGL (± 5 m)
- Pulse repetition frequency: 199 kHz
- Beam divergence: 1 mrad → footprint at 60 m = 6 cm diameter
- Scan mechanism: Elliptical Palmer scan, lateral FoV ±20°, forward/backward FoV ±14°
- Flight speed: not specified (but typical ~5-6 m/s for UAV)
- Point density: mentioned as >50 pts/m² from sensor specs at similar parameters
- Result: Full penetration to ~3 m depth, vertical accuracy <2 cm vs. total station
- Processing: OWP + SVB algorithm (Schwarz et al. 2019)

**RIEGL miniVUX-3UAV (NIR) survey parameters**:
- Wavelength: 905 nm (NIR, topographic only)
- Flying altitude: 60 m AGL (± 14 m)
- Pulse repetition frequency: 300 kHz
- Beam divergence: 1.5 mrad (1.6 × 0.5 mrad in sensor spec table)
- Flight speed: 6 m/s
- Result: >500 pts/m², used for terrain model and water surface model for refraction correction
- Processing: OWP only

**Coordinate system**: ETRS89/UTM 33N, EPSG:25833, reference epoch 2015.0
**Georeferencing**: 8 saddle-roof reference surfaces. Applied corrections: (-1.2, 1.4, -11.7) cm for VQ-840-GL dataset.

**Waveform data format (as confirmed by paper)**:
- Sample interval: approximately 0.5 ns per SI unit
- Amplitude: ADC (Analog-to-Digital Converter) units
- Figure 2 in the paper shows: waveforms vary by depth section, from ~5 samples (very shallow) to ~20+ samples (deeper), with distinct multi-peak structure for water

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

Key observation: The distribution shows a roughly unimodal distribution slightly skewed toward lower (more negative) values, with a small tail at higher values. There may be two populations mixed: water/low-reflectance land (below -22 dB) and dry land/gravel/vegetation (above -20 dB).

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

**Max amplitude histogram** (of first 2000):
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

**Critical finding**: Every sample waveform shows a characteristic large gap of ~70-83 SI units after approximately SI time ~93-103, followed by a terminal cluster around SI time ~170-188. This terminal cluster is the BOTTOM RETURN or a distant land return. The large gap (70-83 SI × 0.5 ns/SI = 35-41 ns × 300 m/μs / 2 = 5.3-6.2 m of air travel) represents the two-way travel distance to a distant reflector. At 60 m flying altitude, this pattern strongly suggests these are water-penetrating returns: surface near SI ~40-50, water column through ~50-100, and a bottom/deep return near SI ~170-188.

**Waveform sharing**: Only 0.2% of consecutive rows share the same waveform data (of first 1000 rows). Multi-echo assignments to the same waveform are rare but present.

---

## 11. Gaps in Knowledge

The following information was not found in the available sources and must be treated as unknown:

1. **Exact reflectance threshold (dB) separating water from land** for the RIEGL VQ-840-GL at 532 nm at the Pielach River: No specific numerical threshold found. Must be determined empirically from data.

2. **Specific ADC threshold for water vs. land peaks**: The 100 ADC threshold used for peak counting is heuristic; the actual noise floor is not specified.

3. **The SVB algorithm implementation details**: The algorithm is described in Schwarz et al. 2017 and 2019 (papers not in the provided materials). Exact parameter settings are unknown.

4. **Water vs. land ratio in the dataset**: The geographic extent and river width are known (~20 m wide river in a broader landscape), but the exact class balance in the 234,024-point dataset is unknown without labels.

5. **OWP vs. SVB classification of individual points**: The provided dataset (`point_cloud_df.txt`) does not indicate whether each point was extracted by OWP or SVB. This distinction might help identify water points (SVB points are more likely to be water-column or bottom).

6. **Turbidity and k-value for October 2024 survey**: The diffuse attenuation coefficient k is not given numerically. The paper states water clarity was "good."

7. **Exact noise floor level in ADC units**: The ADC has a 12-bit range (0–8191). The noise floor and saturation levels are not explicitly stated.

8. **Whether waveforms in the dataset are pre-OWP raw or post-OWP**: The waveform_df.txt appears to contain the raw digitized waveforms (as the time arrays are non-contiguous, consistent with the "sample datagrams" format described in the echo detection slides).

9. **Direct numerical reflectance thresholds from literature for water at 532 nm**: No specific dB threshold was found in the sources.

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
