# Water vs. Land Classification from Full-Waveform LiDAR — Knowledge Base

*Compiled from 50 academic documents (EduServ 2021 "Recent LiDAR Technologies", TU Vienna, Prof. Gottfried Mandlburger). Only information actually present in the source documents is included. No gaps have been filled with assumptions.*

---

## 1. Problem Overview

**Task**: Classify individual LiDAR points as *water* or *land* using full-waveform data.

**Data format** (from Pielach river surveys described in 5.01–5.03 transcripts and slides):
- Point cloud: x, y, z coordinates + Riegl reflectance values (calibrated backscattering coefficient γ)
- Waveforms: time (sample intervals ~0.5 ns) + amplitude (ADC units)
- Sensor: Riegl VQ-840G (Pielach UAV surveys) / Riegl VQ-820-G (manned surveys)
- Acquisition: typically March (leaf-off, clear water) for Pielach surveys

**Expected challenges (from source documents)**:
- Water surfaces are specular reflectors → signal dropout off-nadir (data voids)
- Turbidity drastically reduces penetration depth and signal strength
- Very shallow water (<20 cm) cannot be separated from surface echo
- Class imbalance: water is typically a small fraction of the scene
- Edge cases: dry gravel banks, wet soil, different depths of water

---

## 2. Physics of LiDAR Waveforms

### 2.1 How Waveforms Work

*(Sources: 1.06 transcript/slides, 1.03 transcript)*

Full waveform LiDAR records the entire temporal history of the backscattered signal, sampled at ~0.5–1 ns intervals. For each emitted pulse:

1. Laser emits a Gaussian-shaped pulse (~5 ns for topo LiDAR, ~1–1.5 ns for bathy).
2. Pulse travels through air, interacts with target(s), and returns to receiver (avalanche photodiode).
3. Receiver converts optical signal to voltage → ADC digitizes at high rate.
4. Raw data: 1.5–6 GByte/s → processed to 30–150 MByte/s after echo detection.

**Waveform attributes extracted per echo (Gaussian decomposition)**:
- **Amplitude** (Ai): peak power of the i-th echo
- **Range** (μi): temporal center = distance to target
- **Echo width** (σi): standard deviation of the Gaussian = breadth of the echo

**Echo separability**: Two targets can be separated if their distance exceeds c·τ/2, where τ is the pulse duration:
- Topo LiDAR (5 ns pulse): separability = 75 cm
- Bathy LiDAR (1–1.5 ns pulse): separability = 15–22 cm
- When two targets are closer than this → echoes merge → single broadened echo.

### 2.2 Water Surface Interaction

*(Sources: 2.01 transcript/slides, 5.01 transcript/slides, 1.09 transcript/slides)*

**Specular reflection behavior**:
- Water is a **specular reflector**: all light follows the law of reflection (incoming angle = outgoing angle).
- The backscattering solid angle ω for a specular surface is very small.
- Backscattering cross section: σ = (4π/ω) · ρ · A → for specular surfaces, almost all energy goes in one direction only.
- **At off-nadir angles**: reflected energy goes away from the receiver → **data voids** in the point cloud.
- **"Specular reflections at water surfaces often lead to data voids, which is okay, because in this case, there is no way of getting appropriate data with laser scanning."** (1.09 transcript)
- **At nadir angle**: specular reflection returns directly to receiver → **very high signal strength**.

**Wavelength-dependent behavior**:
- **Infrared (>900 nm, e.g., 1064/1550 nm)**: high absorption coefficient → IR laser is **fully absorbed at the water surface**. IR channel provides a precise water surface echo.
- **Green (532 nm)**: water absorption minimum in blue-green domain → green laser **penetrates the water column**.

**Bathymetric LiDAR radar equation** (5.01 slides):
The total return PR = PWS + PWC + PWB + PBK where:
- PWS (water surface): depends on surface albedo L0 and incidence angle. Wavy/rough water surface returns more signal back than smooth surface.
- PWC (water column): exponential attenuation e^(-2kz/cos(αW)), where k = effective attenuation coefficient.
- PWB (water bottom): depends on bottom reflectivity RB and remaining signal after attenuation.
- PBK: background radiation.

**Attenuation and Secchi depth** (5.01–5.02 transcripts):
- Effective attenuation coefficient: k [m⁻¹]
- Empirical relationship: k ≈ 1.7 / Secchi_depth
- At k = 0.1 (clear water): Secchi depth ≈ 17 m
- Maximum penetration depth: ~3× Secchi depth (deep bathy), ~1.5× Secchi depth (shallow bathy)
- Turbid water: higher k → faster signal decay → shallower penetration → data voids in turbid zones.

**Pielach river data (5.03 transcript)**:
- "The deeper it gets, the lower the reflectance gets."
- August 2019 (turbid): data voids in highly turbid areas, even with full waveform post-processing.
- March 2021 (clear water): full penetration to 3 m depth with SVB FWF post-processing.

### 2.3 Land/Vegetation Interaction

*(Sources: 1.01, 1.02, 1.06 transcripts)*

- Extended hard surfaces (roads, gravel banks, roofs): typically **single echo**, narrow echo width.
- Semi-transparent surfaces (vegetation): **multiple echoes** per pulse (first return from canopy, subsequent from branches, last from ground).
- Echo width is **broader for semi-transparent/composite targets** (e.g., shrubs, deadwood).
- Gravel/bare soil: high reflectance at NIR/SWIR wavelengths.

### 2.4 Reflectance Properties by Surface Type

*(Sources: 2.01 transcript/slides, 2.02 transcript/slides, 5.03 transcript)*

At λ ≈ 900 nm (NIR):
| Material | Reflectivity |
|----------|-------------|
| White paper | up to 100% |
| Snow | 80–90% |
| Limestone, clay | up to 75% |
| Deciduous trees | typ. 60% |
| Carbonate sand (dry) | 57% |
| Beach sands, dry | typ. 50% |
| Carbonate sand (wet) | 41% |
| Coniferous trees | typ. 30% |
| Concrete (smooth) | 24% |
| Asphalt with pebbles | 17% |
| Black rubber | 2% |
| **Water (specular)** | **Not measurable as diffuse — specular reflector** |

At multiple wavelengths (2.02 transcript, based on Pfennigbauer & Ullrich 2011):
- Green vegetation: low green (~0.5 μm), high NIR (~50%), moderate SWIR (~30%).
- Water at 532 nm (green): penetrates. At 1064 nm (NIR): absorbed at surface. At 1550 nm (SWIR): absorbed at surface.
- Riegl reflectance values: calibrated backscattering coefficient γ, derived from amplitude × echo width after range and angle correction.

---

## 3. Feature Engineering

### 3.1 Waveform-Derived Features

*(Sources: 1.06 transcript/slides, 2.01 transcript, 2.02 transcript/slides)*

**Gaussian decomposition** (Wagner et al. 2004, as cited in 1.06 transcript/slides):
```
wf(t) = c + Σ_{i=1}^{m} [Ai · exp(-(t - μi)² / σi²)]
```
Parameters extracted **per echo**:
| Feature | Symbol | Description | Relevance for water/land |
|---------|--------|-------------|-------------------------|
| Amplitude | Ai | Peak power of echo [DN] | Water: near-zero (off-nadir) or very high (nadir); land: moderate-high |
| Range | μi | Temporal center → distance [m] | Used for 3D positioning |
| Echo width | σi | Standard deviation of Gaussian [ns] | Water: typically single narrow echo; vegetation: broad |
| Baseline | c | Offset (noise level) | — |
| Backscatter cross section | σ_BSC | Combines all echo params | Material property indicator |

Additional derived features:
- **Number of echoes per pulse** (m): water (off-nadir) = 0 (void) or 1 (surface echo); vegetation = typically 2–4.
- **Echo ratio**: proportion of energy in last echo vs. first echo.
- **Backscattering coefficient** (calibrated reflectance γ):
  ```
  γ_i ∝ Ri² · Pi · σ_p,i / (Ŝ_ss · σ_ss) · Ccal · 10^(2Ri·a/10000)
  ```
  This removes range and incidence angle dependency → represents material property.
- **Area under Gaussian** (≈ Pi · σi): proxy for received power per echo.

**Note on the Pielach data**: The Riegl VQ-840G stores the full waveform (raw) and also outputs calibrated reflectance values (γ) per point. The input data includes "Riegl reflectance values" which are this calibrated backscattering coefficient.

### 3.2 Point Cloud-Derived Features

*(Sources: 1.10 transcript/slides, 2.02 transcript)*

From the per-point coordinates and attributes:
- **Height (z)**: water surface is typically at low elevation; bathymetric points below.
- **Local planarity / roughness**: water surface is planar; terrain is rough; vegetation is irregular.
- **Normal vectors**: water surface normals point nearly straight up; rough terrain normals vary.
- **Point density (local)**: data voids over water (off-nadir) → low density in water zones.
- **Number of returns per pulse**: water surface → 1 (or 0 for voids); vegetation → 2–4.
- **Neighborhood features** (from work of Weinmann et al., cited in 1.10 transcript): feature relevance and optimal neighborhood size for supervised classification.

### 3.3 Most Discriminative Features for Water vs. Land

Based on what the documents explicitly state:

1. **Calibrated reflectance (Riegl reflectance)**: Water shows very low or zero reflectance (specular, off-nadir) or very high reflectance (specular, at nadir). Dry land: moderate to high. This is the strongest single feature.
   - "The deeper it gets, the lower the reflectance gets." (5.03 transcript)
   - Cross sections of Pielach colored by reflectance: water body clearly visible as distinct zone (6.03 slides).

2. **Number of echoes per pulse**: Water surface typically produces 0 or 1 echo; vegetation: 2–4 echoes.
   - Full waveform LiDAR: 1.84 echoes/pulse in forested area (4.03 transcript).

3. **Echo width (σ)**: Hard surfaces (gravel banks, asphalt) → narrow; semi-transparent objects → broad.

4. **Amplitude**: Very low amplitude for off-nadir water returns; may be very high at nadir.

5. **NDVI variants** (multispectral data only): NDVI1064-532 and NDVI1550-532 used for land cover classification into buildings, trees, roads, grass (Morsy et al. 2017, cited in 2.03 transcript/slides). Water would show distinct NDVI values.

---

## 4. Classification Approaches

### 4.1 Traditional ML Methods

*(Sources: 1.10 transcript/slides, 2.03 slides)*

Methods mentioned for LiDAR point cloud classification:
- **Random Forest (RF)**: supervised classification with point features (1.10 transcript).
- **Support Vector Machines (SVM)**: (1.10 transcript).
- Robust interpolation with feature-dependent weights (Kraus & Pfeifer, 1998): not ML but geometry-based with feature weights; echo width used as discriminator.

For multispectral:
- NDVI-based classification (Morsy et al. 2017): NDVI1064-532 and NDVI1550-532 for urban classes (2.03 transcript/slides).

### 4.2 Deep Learning Methods

*(Sources: 1.10 transcript/slides)*

All mentioned in the context of ALS point cloud classification (specifically DTM generation / semantic labeling):

| Method | Reference | Architecture type |
|--------|-----------|------------------|
| 2D-CNN (raster-based) | Hu and Yuan, 2016 | Rasterize point cloud → apply 2D CNN |
| PointNet++ | Winiwarter et al., 2018 (ALS application) | Per-point features + hierarchical grouping |
| 3D Sparse Voxel CNN | Schmohl et al., 2019 | Voxel-based sparse 3D convolution |
| PFCN (Point-based FCN) | Jin et al., 2020 | Fully convolutional, point-based |
| DGCNN (Dynamic Graph CNN) | Widyaningrum et al., 2021 | Dynamic graph-based edges |

**Note**: These methods are referenced for general ALS point cloud classification/semantic labeling (ground vs. off-terrain). None are specifically tested for water vs. land classification in these documents. They are presented as the current state of the art in the field.

### 4.3 Recommended Architectures

*The source documents do not directly recommend architectures for water vs. land classification. The following is limited to what is directly implied by the documents:*

- For 1D waveform classification: Gaussian decomposition extracts features → these feed into any classifier.
- For point-level classification with waveform features: PointNet++ or DGCNN are mentioned as state-of-the-art.
- Raster-based (2D-CNN): requires interpolating point cloud to grid first.

**ICESat-2 photon classification** (4.04 slides): uses a surface mask with classes including "inland water" + "land" — shows that explicit water/land/ice/ocean classification is a known problem in the field and has been solved for spaceborne systems using photon counting data.

### 4.4 Training Strategies and Best Practices

*The source documents do not cover ML training strategies in detail for this specific problem. The following is limited to what is mentioned:*

- Feature relevance analysis (Weinmann et al.): useful for selecting most discriminative features.
- Echo width shown to be valuable as additional attribute weight in robust interpolation (1.10).
- Post-processing for SPL clutter: low volumetric density + low intensity → remove (4.03 transcript).

---

## 5. Preprocessing Pipeline

### 5.1 Waveform Preprocessing

*(Sources: 1.06 transcript/slides, 2.02 transcript/slides, 4.03 transcript)*

**Gaussian decomposition** (primary FWF processing method):
1. Find local maxima in waveform → determines number of echoes m.
2. Fit Gaussian curves (non-linear least squares, Levenberg-Marquardt).
3. Output per echo: Ai (amplitude), μi (range), σi (echo width).
4. Baseline c is also estimated.

**Online vs. post-processing waveform processing**:
- Standard online processing: limited depth penetration in turbid water.
- FWF post-processing (SVB = Surface-Volume-Bottom, Schwarz et al. 2019): identifies weak water bottom echoes by analyzing full waveform → deeper penetration in turbid conditions (5.03 transcript/slides).

**Radiometric calibration** (2.02 transcript/slides):
- Removes range (R²) and incidence angle dependency from raw amplitude.
- Input: amplitude Ai, echo width σi, range Ri.
- Output: backscattering coefficient γi (= Riegl reflectance).
- Must be applied independently per wavelength channel.
- Calibration constant Ccal derived from ground targets with known reflectivity.
- Atmospheric correction: Ccal · 10^(2R·a/10000) where a is atmospheric attenuation [dB/km].

### 5.2 Point Cloud Preprocessing

*(Sources: 1.09 transcript, 3.03 transcript/slides)*

- **Strip adjustment** (1.09, 3.03): minimizes discrepancies between overlapping strips via ICP-based least squares. Reduces height errors from ~10 cm (before) to ~1.7 cm (after).
- **MTA zone resolution** (1.03 transcript): resolve which pulse produced which echo when multiple pulses are in the air simultaneously.
- **Data voids over water**: specular reflections → no return → acknowledged in the literature as expected behavior, not an error.
- **SPL noise removal** (4.03 transcript): low volumetric point density + low intensity filters.

### 5.3 Data Normalization and Formatting

*(Sources: 2.02 transcript/slides)*

- Raw amplitude [DN] → calibrated reflectance γ via radiometric calibration.
- The calibrated reflectance γ values are what Riegl instruments provide as "Riegl reflectance."
- For waveform input to ML: amplitude values in ADC units; time in sample intervals (~0.5 ns).
- Refraction correction needed for water column points before accurate depth estimation (5.01 transcript/slides).

---

## 6. Domain Rules and Heuristics

### 6.1 Known Thresholds and Decision Rules

*(Sources: 2.01, 5.01, 5.02, 5.03 transcripts/slides; 1.09 transcript/slides)*

1. **Water surfaces produce data voids** when the scan angle is off-nadir. "Specular reflections at water surfaces often lead to data voids." (1.09 transcript)

2. **Water detection by wavelength**:
   - NIR laser (1064/1550 nm): absorbed at water surface → only gives water *surface* echo. Deep = no IR return.
   - Green laser (532 nm): penetrates water → gives surface + water column + bottom echoes.

3. **Reflectance gradient**: "The deeper it gets, the lower the reflectance gets." (5.03 transcript) — monotonic decrease of reflectance with water depth.

4. **Minimum detectable depth**: c·τ/2 (half pulse length). For VQ-840G (~1.5 ns): min depth ≈ 22 cm.

5. **Maximum penetration depth**: ~1.5–2× Secchi depth for shallow bathy systems. For Pielach (March, clear water): 3 m with SVB FWF post-processing.

6. **Pielach river depths**: 0–1.5 m typical, pools to 2–3 m.

7. **At nadir over water**: very high backscattering (specular return directly to sensor). Can appear as anomalously bright points over water when scan is nearly vertical.

8. **Signal decay in water column**: exponential with depth, factor e^(-2kz/cos(αW)). k ≈ 1.7/Secchi_depth.

9. **Pulse repetition rate and water**: very high PRR sensors (>1 MHz) have very short pulses → better water/ground echo separation, but also MTA issues.

### 6.2 Edge Cases and Failure Modes

*(Sources: 5.01–5.03, 4.03 transcripts; 1.09 transcript)*

1. **Turbid water**: high attenuation coefficient k → data voids even with FWF post-processing. FWF post-processing (SVB) helps recover weak echoes but cannot overcome complete absorption.

2. **Very shallow water (<20 cm)**: cannot separate water surface echo from water bottom echo with ~1.5 ns pulses.

3. **Smooth still water surface**: produces more specular return than wavy/rough water. Very smooth water → nearly complete signal dropout off-nadir.

4. **Wavy water surface**: ripples scatter light more diffusely → higher L0 → more signal back → some echoes returned even off-nadir. Echo appears as "floating" point at the water surface.

5. **Nadir-angle scan over water**: very high amplitude return from specular reflection → could be misclassified as bright land feature.

6. **Wet surfaces / puddles**: not explicitly addressed in the source documents.

7. **Gravel banks near water**: dry gravel has high reflectance; wet gravel near waterline may have reduced reflectance. The documents note this gradient.

8. **SPL data over water**: SPL (532 nm) has bathymetric capability but also produces many clutter points. Post-processing removes useful water returns along with noise (4.03 transcript).

9. **Seasonal effects**: March (leaf-off, clear water) = optimal for Pielach. August = turbid, lower penetration (5.03 transcript).

10. **Range walk** (SPL and Geiger mode): "an important issue for all the techniques using high sensitive receivers" → height estimation errors on steep slopes or bright targets (4.03 transcript). Not an issue for linear mode full waveform LiDAR.

---

## 7. Evaluation

### 7.1 Metrics to Use

*(Source: 1.09 transcript; 4.03 transcript)*

From the source documents:
- **DEM precision**: σ₀ from moving planes interpolation (standard deviation of laser points w.r.t. fitted plane in smooth areas). Values cited: waveform LiDAR ~2–3 cm, SPL ~varies.
- **Strip height difference analysis**: used for relative accuracy assessment.
- **Echo count per pulse**: mean echoes/pulse as quality metric (waveform: 1.84; SPL: 1.06).
- **Point density maps**: average pts/m² per raster cell.

*The documents do not discuss classification accuracy metrics (precision/recall/F1/IoU) for water vs. land specifically.*

### 7.2 Expected Accuracy Ranges

*(Sources: 1.09, 4.03 transcripts)*

For geometric accuracy:
- Full waveform LiDAR height precision: 1–3 cm (smooth horizontal surfaces).
- After strip adjustment: σ_mad reduced from ~8.5 cm to ~1.7 cm.
- SPL height precision: <10 cm (manufacturer spec); measured ~2–10 cm depending on slope.

*Classification accuracy ranges for water vs. land are not mentioned in the source documents.*

### 7.3 Common Pitfalls

1. Not applying radiometric calibration → amplitude is range- and angle-dependent → misleading features.
2. Not accounting for refraction → water column points have incorrect depths.
3. Confusing data voids with land (no data ≠ land; could be specular water).
4. Using pulse repetition rate as point density proxy (effective scan rate ≠ PRR for rotating polygon scanners due to unusable edge).
5. Using all echoes for point density calculation in multi-target environments (vegetation) → bias (1.09 transcript).

---

## 8. Source File Reference

| Knowledge Area | Primary Sources | Score |
|----------------|----------------|-------|
| Full waveform physics, Gaussian decomposition | 1.06 transcript, 1.06 slides | 5 |
| Water as specular reflector, data voids | 2.01 transcript, 2.01 slides | 5 |
| Bathymetric LiDAR measurement principle | 5.01 transcript, 5.01 slides | 5 |
| Topo-bathy sensor parameters | 5.02 transcript, 5.02 slides | 5 |
| Pielach river: reflectance by depth, turbidity | 5.03 transcript, 5.03 slides | 5 |
| Radiometric calibration (reflectance derivation) | 2.02 transcript, 2.02 slides | 4 |
| DL classification methods for ALS | 1.10 transcript, 1.10 slides | 4 |
| Echo width as discriminative feature | 1.10 transcript, 1.10 slides | 4 |
| Pielach UAV survey, reflectance cross-sections | 6.03 slides | 4 |
| Pulse length, waveform separability | 1.03 transcript, 1.05 transcript | 3 |
| General LiDAR crash course | 1.02 transcript, 1.02 slides | 3 |
| Quality assessment, water data voids | 1.09 transcript, 1.09 slides | 3 |
| Multispectral NDVI classification | 2.03 transcript, 2.03 slides | 3 |
| SPL vs. waveform comparison | 4.03 transcript, 4.03 slides | 3 |
| Spaceborne LiDAR (ICESat-2, GEDI water mask) | 4.04 slides | 3 |
| Laser beam model, bathy vs. topo pulses | 1.05 transcript, 1.05 slides | 3 |