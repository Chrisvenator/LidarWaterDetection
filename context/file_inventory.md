# File Inventory: LiDAR Water Detection Knowledge Base
*Generated from EduServ 2021 "Recent LiDAR Technologies" course materials by Prof. Gottfried Mandlburger, TU Vienna*

---

## Summary Table

| Filename | Type | Topic Summary | Relevance Score |
|----------|------|---------------|-----------------|
| 1.01 120.143-2026S Videos TUWEL.txt | Transcript | Course introduction; LiDAR history 1995–2021; UAV density 800 pts/m²; full waveform attributes (echo width, reflectance); bathymetric overview | 3 |
| 1.02 Basics_LiDAR crash course LectureTube.txt | Transcript | 5-min crash course: multi-sensor system (GNSS+IMU+scanner), multi-target capability, laser radar equation, ALS sensor model | 3 |
| 1.03 Basics_Ranging LectureTube.txt | Transcript | Laser ranging via round-trip time; pulse length (1–10 ns); echo separability (c·τ/2); multiple pulses in air (MTA) | 3 |
| 1.04 Basics_Scanning LectureTube.txt | Transcript | Scanning mechanisms (oscillating/rotating mirror, Palmer, Risley prism); point density patterns per scan type | 2 |
| 1.05 Basics_Laser beam LectureTube.txt | Transcript | Gaussian laser pulse model; beam divergence; topo (5 ns, 0.2 mrad) vs. bathy (1 ns, 1 mrad); Pielach 3D example | 3 |
| 1.06 Basics_Echo detection LectureTube.txt | Transcript | Full waveform vs. discrete echo; Gaussian decomposition formula; amplitude/echo-width/range attributes; backscatter cross section | 5 |
| 1.07 Basics_Direct georeferencing LectureTube.txt | Transcript | Direct georeferencing equation; GNSS+IMU+scanner components; lever arm and boresight calibration | 2 |
| 1.08 Basics_Flight planning LectureTube.txt | Transcript | Swath width formula; point density calculation; flight date impact (leaf-on vs. leaf-off) | 2 |
| 1.09 Basics_Quality assessment LectureTube.txt | Transcript | Point density/precision/accuracy metrics; specular water surface creating data voids | 3 |
| 1.10 Basics_DTM generation LectureTube.txt | Transcript | DTM/DSM filtering; DL methods (PointNet++, 2D-CNN, Voxel-CNN, DGCNN, PFCN); echo width as ground/vegetation discriminator | 4 |
| 2.01 Multispectral_Laser Radar Equation LectureTube.txt | Transcript | Laser radar equation; backscattering cross section; water as specular reflector causing data voids; reflectivity table by material | 5 |
| 2.02 Multispectral_Radiometric_Calibration LectureTube.txt | Transcript | Radiometric calibration; backscattering coefficient; waveform amplitude×sigma ≈ received power; multispectral reflectance | 4 |
| 2.03 Multispectral_Sensors_and_applications LectureTube.txt | Transcript | Multispectral sensors (Titan, VQ-1560i-DW, Chiroptera); NDVI-based land cover classification (Morsy et al. 2017) | 3 |
| 3.01 Hybrid_Sensor overview LectureTube.txt | Transcript | Hybrid LiDAR+camera systems; sensor specs; CZMIL coastal zone mapper | 2 |
| 3.02 Hybrid_LiDAR-DIM LectureTube.txt | Transcript | LiDAR vs. Dense Image Matching comparison; vegetation penetration; stereo occlusion | 2 |
| 3.03 Hybrid_Sensor orientation LectureTube.txt | Transcript | Strip adjustment workflow; ICP-based minimization; hybrid laser+image sensor orientation | 2 |
| 4.01 SPL_Measurement principle LectureTube.txt | Transcript | Linear mode vs. Geiger mode vs. Single Photon LiDAR; full waveform capability; beamlet arrays | 2 |
| 4.02 SPL_GmLiDAR and SPL sensors LectureTube.txt | Transcript | Geiger mode ITI-1000 specs; Leica SPL-100 specs; green 532 nm wavelength for bathymetry | 2 |
| 4.03 SPL_Pros and Cons LectureTube.txt | Transcript | SPL vs. waveform LiDAR comparison; waveform: 1.84 echoes/pulse, SPL: 1.06; SPL lower penetration; range walk issue | 3 |
| 5.01 Topo-Bathy_Measurement principle LectureTube.txt | Transcript | Laser bathymetry physics; green wavelength; air-water interface; signal attenuation (k coefficient); refraction; Secchi depth | 5 |
| 5.02 Topo-Bathy_Sensor overview LectureTube.txt | Transcript | ALB sensor categories (deep/shallow bathy); pulse energy/length; Secchi depth; minimum detection depth | 5 |
| 5.03 Topo-Bathy_Application examples LectureTube.txt | Transcript | Pielach river surveys; reflectance by depth; turbidity effects; SVB FWF post-processing (Schwarz 2019); SPL bathymetry tests | 5 |
| 6.01 UAV - Whats new LectureTube.txt | Transcript | UAV LiDAR overview; flash vs. scanning; corridor mapping; same sensor model as manned ALS | 2 |
| 6.02 UAV - Sensors and platforms LectureTube.txt | Transcript | UAV sensor specs; VQ-840G topo-bathy UAV scanner; 5 cm footprint at 50m AGL | 3 |
| 1.01 Basics_LiDAR timeline.pdf | Slides | LiDAR evolution timeline; point density growth; full waveform colored by reflectance/amplitude/echo width | 2 |
| 1.02 Basics_LiDAR crash course.pdf | Slides | ALS crash course; multi-echo; Laser-Radar equation; direct georeferencing equation | 3 |
| 1.03 Basics_Ranging.pdf | Slides | Range formula; pulse length (bathy 1 ns, topo 5 ns); range resolution; MTA table | 3 |
| 1.04 Basics_Scanning.pdf | Slides | Scan mechanisms; topo-bathy Riegl VQ-880-G (Palmer + rotating prism) | 2 |
| 1.05 Basics_Laser beam.pdf | Slides | Gaussian pulse model; footprint size; bathy vs. topo pulse parameters; Pielach 3D point cloud | 3 |
| 1.06 Basics_Echo detection.pdf | Slides | FWF vs. discrete; Gaussian decomp. formula; amplitude/range/echo-width/backscatter cross-section attributes | 5 |
| 1.07 Basics_Direct georeferencing.pdf | Slides | Sensor model; georeferencing | 2 |
| 1.08 Basics_Flight planning.pdf | Slides | Flight planning parameters; point density formula | 2 |
| 1.09 Basics_Quality assessment.pdf | Slides | Quality metrics; specular water reflections → data voids | 3 |
| 1.10 Basics_DTM generation.pdf | Slides | DTM filtering; ML methods (RF, SVM, 2D-CNN, PointNet++, Voxel-CNN, PFCN, DGCNN); echo width for vegetation filtering | 4 |
| 2.01 Multispectral_Laser Radar Equation.pdf | Slides | Laser radar equation; specular vs. diffuse reflection; reflectivity table (many materials); backscattering cross-section | 5 |
| 2.02 Multispectral_Radiometric_Calibration.pdf | Slides | Backscattering coefficient formula; calibration constant Ccal; multispectral reflectance at 532/1064/1550 nm | 4 |
| 2.03 Multispectral_Sensors_and_applications.pdf | Slides | Sensor overview table; NDVI classification (Morsy 2017); forest inventory with voxelization | 3 |
| 3.01 Hybrid_Sensor overview.pdf | Slides | Hybrid sensor table; CZMIL coastal zone sensor | 2 |
| 3.02 Hybrid_LiDAR-DIM.pdf | Slides | LiDAR vs. DIM data properties table | 2 |
| 3.03 Hybrid_Sensor orientation.pdf | Slides | Strip adjustment; hybrid sensor orientation workflow | 2 |
| 4.01 SPL_Measurement principle.pdf | Slides | SPL vs. linear mode; waveform vs. discrete; beamlet arrays | 2 |
| 4.02 SPL_GmLiDAR and SPL sensors.pdf | Slides | GmLiDAR ITI-1000 and SPL-100 sensor specifications | 2 |
| 4.03 SPL_Pros and Cons.pdf | Slides | SPL vs. waveform comparison; echoes per pulse map; precision maps; slope vs. precision plots | 3 |
| 4.04_SBL_GEDI_ICESat2.pdf | Slides | Spaceborne LiDAR (ICESat-2, GEDI); photon classification (noise/ground/canopy/water); GEDI waveform products | 3 |
| 5.01 Topo-Bathy_Measurement principle.pdf | Slides | Bathymetric LiDAR equation (water surface/column/bottom components); refraction (Snell's law); attenuation | 5 |
| 5.02 Topo-Bathy_Sensor overview.pdf | Slides | Deep vs. shallow bathy sensors; pulse parameters table; CZMIL water bottom classification | 5 |
| 5.03 Topo-Bathy_Application examples.pdf | Slides | Pielach river mapping; morphodynamics; UAV bathy; SVB FWF post-processing; SPL bathymetry | 5 |
| 6.01 UAV - What's new.pdf | Slides | UAV LiDAR; echo number/reflectance coloring; applications | 2 |
| 6.02 UAV - Sensors and platforms.pdf | Slides | UAV sensor table; VQ-840G bathy parameters; UAV bathy sensor table | 3 |
| 6.03 UAV - Application examples.pdf | Slides | Pielach river; Hessigheim; reflectance-colored cross sections; bathy acquisition March 2021 | 4 |

---

## Detailed Summaries

### 1.01 120.143-2026S Videos TUWEL.txt (Transcript)
Course introduction to EduServ 2021 "Recent LiDAR Technologies" by Gottfried Mandlburger, TU Vienna. Introduces the evolution of airborne LiDAR over 25 years: scan rates from 5 kHz to 6 MHz, weights from 50 kg to 1 kg, point densities from 1–2 pts/m² to 800 pts/m² (UAV). Highlights full waveform LiDAR attributes (echo width as vegetation/hard-surface discriminator, reflectance values), multispectral LiDAR, and shallow bathymetric LiDAR using the green wavelength.

### 1.02 Basics_LiDAR crash course LectureTube.txt (Transcript)
5-minute crash course on airborne LiDAR as a multi-sensor system (GNSS + IMU + laser scanner). Explains polar measurement system, multi-target capability, active illumination. Presents three basic equations: ranging (R = c·Δt/2), laser radar equation (relating transmitted/received power), and ALS sensor model for direct georeferencing.

### 1.03 Basics_Ranging LectureTube.txt (Transcript)
Explains laser ranging via round-trip time measurement. Covers pulse lengths (1–10 ns = 30 cm–3 m), echo separability distance (c·τ/2), and multiple pulses in air (MTA). Notes topographic LiDAR uses ~5 ns pulses (75 cm separability) and bathymetric uses ~1 ns (15 cm separability). Introduces MTA zone resolution as necessary post-processing step.

### 1.04 Basics_Scanning LectureTube.txt (Transcript)
Discusses scanning mechanisms: oscillating mirror (inhomogeneous density, higher at borders), rotating mirror (homogeneous density), Palmer scanner (conical, circular pattern), and Risley prism (arbitrary patterns). Includes a topo-bathy example combining rotating prism (IR) and Palmer scanner (green) in the Riegl VQ-880-G.

### 1.05 Basics_Laser beam LectureTube.txt (Transcript)
Explains the 2D Gaussian laser pulse model (temporal and lateral). Notes topo LiDAR uses 5 ns/0.2 mrad (narrow beam), bathy LiDAR uses 1 ns/1 mrad (deliberately broadened for eye safety). Shorter pulses in bathy improve separation of water surface from bottom returns. Pielach river example showing three sensor types simultaneously.

### 1.06 Basics_Echo detection LectureTube.txt (Transcript)
Compares discrete echo detection and full waveform recording. Explains digital signal chain (laser → ADC → detection → estimation). Details Gaussian decomposition (Wagner et al. 2004): fitting Gaussian curves to backscattered waveform, extracting amplitude P, echo range μ, and echo width σ per component. Shows FWF attributes at Schönbrunn palace (range, amplitude, echo width, backscatter cross section).

### 1.07 Basics_Direct georeferencing LectureTube.txt (Transcript)
Explains direct georeferencing: combining GNSS position (1–2 Hz), IMU attitude (200–500 Hz), and scanner measurements (kHz–MHz) via the ALS sensor model. Covers lever arm and boresight angle calibration parameters.

### 1.08 Basics_Flight planning LectureTube.txt (Transcript)
Covers flight planning: parallel strips with overlap, swath width (SB = 2h·tan(θ/2)), point density formula (PD = SR/(SB·v)), and flight date effects (leaf-on/off impact on vegetation penetration and DTM quality).

### 1.09 Basics_Quality assessment LectureTube.txt (Transcript)
Covers LiDAR quality metrics: point density (only last/first echoes in multi-target), precision (σ from smooth surface residuals), relative accuracy (strip differences), absolute accuracy (ground truth). Explicitly notes: "specular reflections at water surfaces often lead to data voids, which is okay, because in this case, there is no way of getting appropriate data with laser scanning."

### 1.10 Basics_DTM generation LectureTube.txt (Transcript)
Covers DSM/DTM separation via ground point filtering. Reviews methods: morphological filters, progressive TIN densification, segmentation-based, and deep learning (2D-CNN, PointNet++, 3D Sparse Voxel CNN, PFCN, DGCNN). Uses robust interpolation with echo width as an attribute weight for separating deadwood/shrub from ground.

### 2.01 Multispectral_Laser Radar Equation LectureTube.txt (Transcript)
Full derivation of the laser radar equation. Key insight for water: water surface is a specular reflector with very small backscattering solid angle ω → "data voids, because the laser pulse is directly reflected at the water surface and all the signal goes away from our receiver." At nadir, water gives high backscattering. Provides reflectivity table at ~900 nm.

### 2.02 Multispectral_Radiometric_Calibration LectureTube.txt (Transcript)
Explains radiometric calibration to remove range and incidence angle dependency. Backscattering coefficient γ = σ/(footprint area). Received power approximated by P·σ_echo (amplitude × echo width from FWF). Calibration constant derived from targets with known reflectivity. Discusses wavelength-dependent reflectance (different per channel, e.g., vegetation: high NIR, low SWIR, low green).

### 2.03 Multispectral_Sensors_and_applications LectureTube.txt (Transcript)
Overview of multispectral LiDAR sensors: Titan (532/1064/1550 nm), VQ-1560i-DW (532/1064), Chiroptera 4X (532/1064). Application: NDVI1064-532 and NDVI1550-532 for urban land cover classification (buildings, trees, roads, grass). Forest inventory with voxelization and spectral analysis.

### 3.01 Hybrid_Sensor overview LectureTube.txt (Transcript)
Reviews hybrid systems combining LiDAR and cameras. Sensors: Titan, Galaxy Prime, VQ-1562, TerrainMapper 2, CityMapper 2, CZMIL. CZMIL features classification of water bottom and water-land transition.

### 3.02 Hybrid_LiDAR-DIM LectureTube.txt (Transcript)
Detailed comparison of LiDAR and Dense Image Matching. LiDAR: active, polar, multi-target, mono-spectral, 20–50 cm spacing, 1–3 cm precision. DIM: passive, stereo, DSM only, multispectral, 5–20 cm spacing. Vegetation: DIM gives top surface, LiDAR penetrates. Proposes fusion for comprehensive 3D coverage.

### 3.03 Hybrid_Sensor orientation LectureTube.txt (Transcript)
Discusses strip adjustment workflow (ICP-based minimization of strip discrepancies) and hybrid sensor orientation (Glira 2019 PhD), which jointly adjusts LiDAR strips and image tie points, achieving ±3 cm accuracy vs. ±15 cm with separate adjustment.

### 4.01 SPL_Measurement principle LectureTube.txt (Transcript)
Explains and compares linear mode LiDAR (full waveform, 1 transmitter/1 receiver), Geiger mode (medium energy, 4096 binary receivers, first return only), and Single Photon LiDAR (100 beamlets, multi-target capable). Linear mode is the only technology capable of full waveform recording.

### 4.02 SPL_GmLiDAR and SPL sensors LectureTube.txt (Transcript)
Specs of Geiger mode ITI-1000 (4000–10000 m altitude, 50 kHz, 0.55 ns pulse, 1064 nm) and Leica SPL-100 (2000–5000 m altitude, 25–60 kHz, 532 nm green, 100 beamlets, 6 MHz effective rate, <10 cm accuracy). SPL green wavelength gives bathymetric capability.

### 4.03 SPL_Pros and Cons LectureTube.txt (Transcript)
Side-by-side comparison: SPL (Leica SPL-100, 4000 m AGL, 5 MHz, 10 strips) vs. waveform LiDAR (Riegl VQ-1560i, 750 m AGL, 1.33 MHz, 18 strips). Waveform LiDAR: 1.84 echoes/pulse, better penetration, sharper point cloud, ~2 cm precision. SPL: 1.06 echoes/pulse, noisy, range walk issue, lower penetration under leaf-on conditions.

### 5.01 Topo-Bathy_Measurement principle LectureTube.txt (Transcript)
Complete physics of laser bathymetry. Green laser (532 nm) used because water absorption minimum is in blue-green. IR laser is fully absorbed at water surface → gives precise water surface. Green enters water column, attenuates exponentially (factor e^(-2kz)), reflects from bottom. Refraction follows Snell's law. Signal velocity: air 300,000 km/h → water 225,000 km/h (n_water/n_air = 4/3). Laser radar equation split into water surface (PWS), water column (PWC(z)), and water bottom (PWB) components.

### 5.02 Topo-Bathy_Sensor overview LectureTube.txt (Transcript)
Reviews deep vs. shallow bathy ALB sensors. Deep bathy: ~3× Secchi depth, high energy (5–7 mJ), low PRR (3–10 kHz), long pulses (2–7 ns = 60 cm–2 m), large footprints (3.5 m at 500 m AGL). Shallow bathy: ~1.5× Secchi depth, low energy (0.02–0.1 mJ), high PRR (100–500 kHz), short pulses (1–2 ns = 30–60 cm), small footprints (~50 cm). Minimum measurable depth ≈ half pulse length (for 35 cm pulse → 20 cm min depth).

### 5.03 Topo-Bathy_Application examples LectureTube.txt (Transcript)
Pielach River applications. Data acquisition March 2021 (leaf-off, clear water). Riegl VQ-840G UAV sensor: >50 pts/m², 5 cm footprint. Depth 0–1.5 m typical, pools 2–3 m. Reflectance: high on dry gravel banks, decreasing with water depth. August 2019 campaign: turbid water → data voids, but SVB (Surface-Volume-Bottom) FWF post-processing (Schwarz et al. 2019) recovered 3 m penetration = 2× Secchi depth. SPL bathymetry tests in Vienna: 2.5 m (New Danube), 1.8 m (Old Danube).

### 6.01 UAV - Whats new LectureTube.txt (Transcript)
UAV LiDAR overview: centimeter-range spatial resolution, same sensor model as manned ALS, lower IMU accuracy demand. Applications: precision farming, archaeology, corridor mapping (rivers, powerlines), construction. Flash vs. scanning LiDAR tradeoffs. Panoramic scanning for canyons.

### 6.02 UAV - Sensors and platforms LectureTube.txt (Transcript)
UAV sensor overview table. VQ-840G at 50 m AGL: 5 cm footprint, 50–200 kHz, Palmer forward (14°) + sideways (20°) look, 2× Secchi depth penetration, user-definable beam divergence 1–6 mrad. Mentions UAV bathy applications for rivers.

---

## Score 3+ Files — Key Relevant Content

### 1.01 120.143-2026S Videos TUWEL.txt (Score 3)

**Waveform Physics & Signal Properties**
- Echo width: temporal breadth of the echo. Narrow for hard impenetrable surfaces (roofs, streets). Broad for semi-transparent targets (vegetation, multiple close targets).
- Echo width is "a very, very good indicator for separating ground or hard surfaces like streets and roofs" from vegetation.
- Full waveform LiDAR provides reflectance values "more or less independent from the measurement process itself."
- Solar panels: very low reflectivity. Streets: middle domain. Roofs: very high reflectance.
- Green wavelength penetrates water; IR is used for topography.

**Practical Considerations**
- UAV LiDAR at 800 pts/m² (2021 state of the art).
- Modern sensors are full waveform by default ("full waveform laser scanning is the standard").

---

### 1.02 Basics_LiDAR crash course LectureTube.txt (Score 3)

**Waveform Physics & Signal Properties**
- Single laser pulse can produce multiple echoes for semi-transparent targets (vegetation).
- One echo for extended impenetrable targets (streets, railway dam).
- Laser is monochromatic → radiometry restricted to laser wavelength.

**Classification Methods**
- Base products: 3D point cloud colored by return number or signal amplitude.

**Preprocessing**
- Direct georeferencing equation: P(t) = P0(t) + R(λ,φ)·[t + RM·p(r,α,β)]

---

### 1.03 Basics_Ranging LectureTube.txt (Score 3)

**Waveform Physics & Signal Properties**
- Pulse length for topographic LiDAR: ~5 ns (75 cm separability distance).
- Pulse length for bathymetric LiDAR: ~1 ns (15 cm separability distance).
- If two targets closer than separability distance → echoes merge → broadened signal (this is echo width broadening).
- "This broadening when we have actually two echoes, but we only see a single return peak" → important for understanding echo width in mixed surface scenarios.

**Preprocessing**
- MTA zone resolution: when PRR ≥ 240 kHz, multiple pulses in air simultaneously; must resolve which pulse produced which echo.

---

### 1.05 Basics_Laser beam LectureTube.txt (Score 3)

**Waveform Physics & Signal Properties**
- Topographic LiDAR: 5 ns pulse, 0.2 mrad divergence → 1.5 m long pulse, 12 cm wide footprint.
- Bathymetric LiDAR: 1 ns pulse, 1 mrad divergence → 30 cm long pulse, 60 cm wide footprint.
- Shorter pulses used in bathy "to get a better separation between the reflection from the water surface and the shallow water bottom."
- Green (532 nm) would produce smaller footprint than IR (1064 nm) due to wavelength, but deliberately broadened for eye safety.
- Pielach river 3D point cloud example with three sensors at different flying altitudes.

---

### 1.06 Basics_Echo detection LectureTube.txt (Score 5)

**Waveform Physics & Signal Properties**
- Full waveform recording: entire temporal history of backscattered signal, sampled at ~0.5–1 ns intervals.
- Advantage over discrete: access to sub-threshold echoes between major peaks not detected by discrete systems.
- FWF attributes per echo: temporal position (range), signal strength/amplitude/reflectance, pulse width/pulse shape deviation, backscatter coefficient.
- Signal dynamic: emitted power ~1 kW → received ~mW to nW.

**Feature Engineering — Waveform-derived Features**
- **Gaussian decomposition** (Wagner et al. 2004):
  ```
  wf(t) = c + Σ(Ai · exp(-(t - μi)² / σi²))
  ```
  - c: baseline
  - Ai: amplitude of i-th echo
  - μi: temporal position (= range) of i-th echo
  - σi: echo width (standard deviation of Gaussian)
  - m: number of echoes (must be determined first from local maxima)
- Non-linear parameter estimation via Levenberg-Marquardt framework.
- **Per-echo attributes**: Amplitude P, Range R, Echo Width w (σ).
- **Echo width**: good discriminator between low vegetation (broad) and impenetrable surfaces like gravel/roofs (narrow, blue in visualizations).
- **Backscatter cross section**: combines amplitude + range + echo width → characterizes object material.

**Preprocessing**
- Precondition for Gaussian decomp: number of targets must be known → detect local maxima first to determine model complexity.
- Digital processing chain: laser → optics → receiver (avalanche photodiode) → ADC → echo detection → parameter estimation.

---

### 1.09 Basics_Quality assessment LectureTube.txt (Score 3)

**Domain Rules & Heuristics**
- **"Specular reflections at water surfaces often lead to data voids, which is okay, because in this case, there is no way of getting appropriate data with laser scanning."** (Source: transcript)
- In multi-target environments (vegetation), only use first or last echoes for point density calculation to avoid bias.

---

### 1.10 Basics_DTM generation LectureTube.txt (Score 4)

**Feature Engineering**
- Echo width is used as a feature in robust interpolation to distinguish deadwood/shrub from ground: areas with artifacts showed "a very high echo width."
- Echo width-dependent point weights significantly improve DTM quality.

**Classification Methods (DL for LiDAR point clouds)**
- **2D-CNN (raster-based)**: Hu and Yuan, 2016 — among first to use CNNs for ALS filtering.
- **PointNet++** (single point based): Winiwarter et al., 2018 — applied to ALS data.
- **3D Sparse Voxel CNN**: Schmohl et al., 2019 — Stuttgart.
- **Point-based Fully Convolutional Network (PFCN)**: Jin et al., 2020.
- **Dynamic Graph Convolutional Neural Network (DGCNN)**: Widyaningrum et al., 2021.
- **Random Forest (RF)** and **Support Vector Machines (SVM)**: supervised classification with point features.
- Feature relevance work: Martin Weinmann et al.

**Preprocessing**
- Ground point filtering = simple version of semantic labeling.
- Include point features (echo width, amplitude) in addition to geometry.

---

### 2.01 Multispectral_Laser Radar Equation LectureTube.txt (Score 5)

**Waveform Physics & Signal Properties — CRITICAL FOR WATER CLASSIFICATION**

The laser radar equation:
```
PE = (PS · DE² · π/4) / (γ_S · R² · π/4) · σ · (DE² · π/4) / (4πR²) · η_ATM · η_SYS
```
Simplified: PE ∝ PS · A · ρ / R² · η_ATM · η_SYS (extended target)

**Backscattering solid angle ω and water:**
- General: σ = (4π/ω) · ρ · A
- Lambertian (diffuse): ω = 4π → σ = 4·ρ·A (orthogonal incidence)
- **Water surface (specular)**: ω is very small → σ is very large only in the specular direction.
- "Water surface, which is a purely specular reflector. In this case, our omega will be very, very small. So all the light follows the law of reflection... Only if the receiver would be situated in the direction of the reflected laser pulse, we would receive a signal."
- **"At water surface, we often get data voids, because the laser pulse is directly reflected at the water surface. And all the signal goes away from our receiver."**
- **"Only if the laser beam hits the water surface under a normal angle, so in the nadir direction, then we get the direct reflection and we get much of it. So we get a very high backscattering in this case. And if we are off nadir, then we won't get any signal."**

**Reflectivity table at λ ≈ 900 nm (from PDF):**
| Material | Reflectivity |
|----------|-------------|
| White paper | up to 100% |
| Snow | 80–90% |
| Beer foam | 88% |
| White masonry | 85% |
| Limestone, clay | up to 75% |
| Newspaper | 69% |
| Deciduous trees | typ. 60% |
| Carbonate sand (dry) | 57% |
| Beach sands | typ. 50% |
| Carbonate sand (wet) | 41% |
| Coniferous trees | typ. 30% |
| Rough wood pallet | 25% |
| Concrete, smooth | 24% |
| Asphalt with pebbles | 17% |
| Lava | 8% |
| Black neoprene | 5% |
| Black rubber tire | 2% |
- *Note: Water reflectivity is NOT listed — water behaves as a specular reflector, not diffuse.*
- **Wavelength dependency**: reflectivity of objects changes with wavelength. This must be considered in multispectral LiDAR.

**Domain Rules & Heuristics**
- Target size vs. received power:
  - Extended target (terrain, water surface): PE ∝ 1/R²
  - Linear target (power line): PE ∝ 1/R³
  - Point target (leaf): PE ∝ 1/R⁴

---

### 2.02 Multispectral_Radiometric_Calibration LectureTube.txt (Score 4)

**Feature Engineering — Radiometric Features**

Backscattering coefficient (calibrated reflectance):
```
γ_i = (R_i² · P̂_i · σ_p,i) / (D_r² · Ŝ_ss · σ_ss) · C_cal · 10^(2·R·a/10000)
```
- R_i: range to i-th echo
- P̂_i: amplitude of i-th echo [DN]
- σ_p,i: std. dev. of i-th echo Gaussian [ns] = echo width
- Ŝ_ss: amplitude of system waveform [DN]
- σ_ss: std. dev. of system waveform [ns]
- C_cal: calibration constant [m⁻² s⁻¹]
- a: atmospheric attenuation [dB/km]

**Key insight**: Calibrated reflectance is derived from waveform attributes: amplitude × echo width (P·σ ≈ area under Gaussian ≈ received power).

**Preprocessing**
- Raw amplitude shows drop from strip center to border (range + incidence angle effects).
- After calibration: homogeneous reflectance independent of viewing geometry.
- Calibration must be run independently per wavelength channel.
- Calibration areas: surfaces with known reflectivity (asphalt, ~17%), or measured with field reflectometer.

**Domain Rules & Heuristics**
- Green vegetation: reflects poorly in green (~0.5 μm), well in NIR (~50% at 1 μm), drops in SWIR (~30% at 1.5 μm).
- Calibrated green images appear dark (asphalt and vegetation both have low green reflectivity).
- Calibrated NIR images appear bright.

---

### 2.03 Multispectral_Sensors_and_applications LectureTube.txt (Score 3)

**Classification Methods**
- Morsy et al. (2017): "Multispectral LiDAR Data for Land Cover Classification of Urban Areas", Sensors 17(5).
  - Uses NDVI1064-532 and NDVI1550-532 for classification into: buildings, trees, roads, grass.
- NDVI using only two IR wavelengths + chlorophyll vegetation index CVI = (SWIR × NIR) / green² for forest inventory.

---

### 4.03 SPL_Pros and Cons LectureTube.txt (Score 3)

**Waveform Physics & Signal Properties**
- Full waveform LiDAR (Riegl VQ-1560i): **1.84 echoes/pulse** average in forested area.
- Single Photon LiDAR (SPL-100): **1.06 echoes/pulse** average.
- Waveform LiDAR: sharper, more consistent point cloud; much better penetration under leaf-on conditions.
- SPL: random clutter points (atmospheric spontaneous photons), low intensity. Post-processing removes clutter but may also eliminate useful points.

**Preprocessing**
- SPL clutter removal: volumetric point density filtering (clutter has low density) + low intensity filtering.

**Classification Methods**
- DEM precision measurement: moving planes interpolation (12 nearest neighbors), compute σ₀ of residuals.

---

### 5.01 Topo-Bathy_Measurement principle LectureTube.txt (Score 5)

**Waveform Physics & Signal Properties — CRITICAL**

Water surface interaction:
- Infrared (>900 nm): fully absorbed at water surface → gives precise water surface echo.
- Green (532 nm): minimum absorption → penetrates water column.
- "Absorption coefficients are quite high in all wavelength domains, except the blue and the green domain" → green window.

Water column attenuation (from bathymetric LiDAR equation):
```
PWC(z) ∝ e^(-2kz/cos(α_W))
```
- k: effective attenuation coefficient [m⁻¹]
- z: depth
- α_W: angle in water
- Empirical relationship: k = 1.7 / Secchi_depth

Water surface signal:
```
PWS ∝ PT · η_ATM · η_SYS · L0 · cos(α_L) / H²
```
- L0: albedo / reflectance of water surface (depends on surface structure/waves)
- In wavy water (tiny ripples): more signal back than smooth surface.

Water bottom:
```
PWB ∝ PT · η_ATM · η_SYS · F · (1-L0) · e^(-2kZ/cos(α_W)) · RB / (nW·H+Z)²
```
- RB: bottom reflectivity
- nW: refractive index of water ≈ 1.333

Geometric effects:
- Signal velocity in water: 225,000 km/h vs. 300,000 km/h in air.
- Beam bending at air-water interface (Snell's law).
- Apparent water bottom is shallower than real → refraction correction needed.

**Domain Rules & Heuristics**
- Green laser used for bathymetry because water absorption minimum is at blue-green wavelengths.
- IR laser does not penetrate water → used only for water surface detection.
- Full LiDAR return: PR = PWS + PWC + PWB + PBK (surface + column + bottom + background).
- Water turbidity (high k) → rapid signal decay → shallower penetration.

---

### 5.02 Topo-Bathy_Sensor overview LectureTube.txt (Score 5)

**Waveform Physics & Signal Properties**

Secchi depth and penetration performance:
- Deep bathy: ~3× Secchi depth (D_S) penetration. At k=0.1 → max depth ~50 m.
- Shallow bathy: ~1.5× Secchi depth. At k=0.1 → max depth ~25 m.
- Secchi depth empirical formula: k ≈ 1.7/D_S.

Sensor parameters affecting water detection:
- Minimum measurable depth ≈ c·τ/2 (half pulse length):
  - Deep bathy (7 ns pulse, 2 m metric length) → min depth ~1 m.
  - Shallow bathy (2 ns pulse, 60 cm metric length) → min depth ~30 cm.
  - UAV-bathy VQ-840G (1.5 ns → ~35 cm) → min depth ~20 cm.

Key shallow bathy sensor – Riegl VQ-880-G:
- Channels: green (bathy) + IR (topo)
- Green PRR: high (multiple MHz)
- Small footprints (~50 cm at 500 m AGL)
- Eye safety limits footprint size to ~50 cm minimum (green laser)

**Domain Rules & Heuristics**
- Minimum depth for topo-bathy ALB: 20 cm (shallow bathy, ~1.5 ns pulses).
- Very shallow water (<20 cm): cannot separate water surface from bottom echoes.
- Deep bathy systems cannot be used for shallow rivers.

---

### 5.03 Topo-Bathy_Application examples LectureTube.txt (Score 5)

**Waveform Physics & Signal Properties — Pielach River**
- "Colored by reflectance. So red means high reflectance here on the dry part of the area. Then we have a steep cliff here. Then we have the river. The deeper it gets, the lower the reflectance gets."
- **High reflectance → dry gravel banks/terrain; decreasing reflectance with increasing water depth** — directly observable feature for water classification.
- August 2019 campaign (turbid): data voids in highly turbid zones.
- With SVB FWF post-processing (Schwarz et al. 2019): recovered 3 m penetration depth = 2× Secchi depth.
- Turbidity significantly impacts detectability.

**Preprocessing**
- Standard online waveform processing: typically limited penetration.
- Full waveform post-processing (SVB = Surface-Volume-Bottom, Schwarz et al. 2019): recovers weak echoes from turbid water → deeper penetration.
- Data acquisition March 2021: leaf-off + clear water = optimal conditions for Pielach surveys.

**Feature Engineering**
- Riegl VQ-840G, Pielach: cross sections colored by reflectance → water body clearly visible as low-reflectance region.

**Domain Rules & Heuristics**
- River morphodynamics (30-year flood): cliff retreat of 3.5 m, thalweg shift of 25 m.
- River depth: 0–1.5 m typical, pools to 2–3 m.
- Turbid conditions: FWF post-processing essential.
- Clear water (March): standard processing sufficient.

**Practical Considerations**
- Flight date: March (leaf-off, clear water) is optimal for Pielach surveys.
- SPL bathymetry in Vienna: 2.5 m (New Danube, standing water), 1.8 m (Old Danube, standing water), ~1 m (Danube river, turbid).

---

### 6.02 UAV - Sensors and platforms LectureTube.txt (Score 3)

**Waveform Physics & Signal Properties**
- Riegl VQ-840G: green (532 nm) laser, 50–200 kHz, beam divergence 1–6 mrad (user-definable).
- At 50 m AGL with 1 mrad: footprint = 5 cm (unprecedented for bathy).
- Penetration: 2× Secchi depth.
- Forward look 14°, sideways 20° (Palmer scanner).

---

### 1.06 Basics_Echo detection.pdf (Score 5)

*Slides corresponding to transcript 1.06 — confirms all transcript content with visual examples.*

- FWF Gaussian decomposition formula explicitly shown:
  ```
  wf(t) = c + Σ(Ai · exp(-(t-μi)²/σi²))
  ```
- Full waveform attributes at Schönbrunn palace: range image, signal amplitude, echo width, backscatter cross section.
- Online waveform processing output: 30 MByte/s to 150 MByte/s.
- Raw FWF data: 1.5–6 GByte/s.

---

### 1.09 Basics_Quality assessment.pdf (Score 3)

**Domain Rules & Heuristics**
- Slide explicitly states: "specular reflections at water surfaces often lead to data voids."

---

### 1.10 Basics_DTM generation.pdf (Score 4)

**Classification Methods (slides)**
- Deep learning methods listed explicitly:
  - 2D-CNN (Hu and Yuan, 2016)
  - PointNet++ (Winiwarter et al., 2018)
  - 3D Sparse Voxel CNN (Schmohl et al., 2019)
  - PFCN (Jin et al., 2020)
  - DGCNN (Widyaningrum et al., 2021)
- Echo width from FWF analysis (Gaussian decomp) shown visually improving DTM.

---

### 2.01 Multispectral_Laser Radar Equation.pdf (Score 5)

*Slides confirming transcript content. Additional details:*

**Reflectivity table at λ = 900 nm** (from slide):
- Deciduous trees: typ. 60%
- Coniferous trees: typ. 30%
- Carbonate sand (dry): 57%, wet: 41%
- Beach sands: typ. 50%
- Concrete smooth: 24%, Asphalt: 17%
- Black rubber tire: 2%
- *Water is not listed as diffuse reflector — it is specular.*

**Specular vs. diffuse reflection schematic** (slide):
- Specular reflection: all light in one direction.
- Diffuse reflection: light into half-sphere.
- Water is specular → typically no return unless at nadir.

---

### 2.02 Multispectral_Radiometric_Calibration.pdf (Score 4)

*Slides with formulas:*

Calibration constant:
```
C_cal = (1/N_KF) Σ (10^(2·Ri·a/10000)) / (Ri² · Pi · σ_p,i) · 4·ρ_KF · cos(θi)
```

Calibrated reflectance at three wavelengths (slides show visual results):
- 532 nm: dark image (low reflectance for vegetation and asphalt)
- 1064 nm: bright image (high NIR reflectance for vegetation)
- 1550 nm: medium image

Reflectance depends on wavelength:
- From slide (based on Pfennigbauer and Ullrich 2011, Bakula 2015): shows spectral curves for different materials at 532/1064/1550 nm.

---

### 2.03 Multispectral_Sensors_and_applications.pdf (Score 3)

*Slides show:*
- NDVI classification results (Morsy et al. 2017): classes = buildings, trees, roads, grass.
- Based on NDVI1064-532 and NDVI1550-532.

---

### 4.03 SPL_Pros and Cons.pdf (Score 3)

*Slides confirm transcript content. Additional:*
- Waveform LiDAR: mean 1.84 echoes/pulse (forested area); SPL: 1.06 echoes/pulse.
- SPL: lower point density at facades and in vegetation.
- Slope vs. precision plot: waveform LiDAR shows moderate drop with slope; SPL shows more pronounced drop.

---

### 4.04_SBL_GEDI_ICESat2.pdf (Score 3)

**Classification Methods (relevant for photon/waveform classification)**

ICESat-2 (photo counting, 532 nm):
- ATL03 product: photon classification → signal vs. background photon.
- ATL03 surface mask: land ice, sea ice, **ocean, land, inland water**.
- ATL08 product: photons classified as **noise, ground, canopy, top of canopy**.
- Statistics per segment: mean, median, min, max, mode, skewness of terrain height.
- Roughness: std of terrain points about interpolated surface.

GEDI (full waveform, 1064 nm):
- GEDI01_B: geolocated waveforms.
- GEDI02_A: ground elevation, canopy top height, relative height metrics.
- GEDI02_B: canopy cover fraction, leaf area index (LAI).
- Resolution: 22 m diameter footprints, 0.5 m vertical.

**Domain Rules & Heuristics**
- ICESat-2 surface mask explicitly includes "inland water" class.
- GEDI transmits at 1064 nm → no water penetration.
- ICESat-2 transmits at 532 nm → some water penetration possible.

---

### 5.01 Topo-Bathy_Measurement principle.pdf (Score 5)

*Slides with explicit mathematical formulations:*

Bathymetric LiDAR radar equation:
```
PWS = (PT · D²π/4 · η_ATM · η_SYS · L0 · cos(αL)) / (π · H²)

PWC(z) ∝ (cos(β(φ)) · e^(-2kz/αW)) / (nW(H+z))²

PWB = (PT · D²π/4 · η_ATM · η_SYS · F · (1-L0) · e^(-2kZ/αW) · cos(αL)) / (π · RB · (nW(H+Z))²)

PR = PWS + PWC + PWB + PBK
```

Key parameters:
- L0: albedo/surface reflection factor
- k: effective attenuation coefficient
- αW: angle in water medium
- nW: refractive index of water
- RB: bottom reflectivity

Refraction (Snell's law):
```
n_air/n_water = c_water/c_air = sin(α_water)/sin(α_air)
```
- c_air = 300,000 km/h, c_water = 225,000 km/h
- Apparent water bottom is displaced (Δxy, Δz) from actual position.

---

### 5.02 Topo-Bathy_Sensor overview.pdf (Score 5)

*Slides with key parameter table for ALB sensors:*

| Sensor | Pulse energy | PRR | Pulse duration | Footprint @500m | Depth perf. |
|--------|-------------|-----|---------------|-----------------|-------------|
| LADS HD | 7 mJ | 3 kHz | ~7 ns | 270–630 cm | 3.0× SD |
| HawkEye 4X | 3 mJ (deep) | 40 kHz | 2 ns | 280–420 cm | 2.5× SD |
| CZMIL Nova | 4 mJ | 10/70/80 kHz | 3 ns | 280–700 cm | 2.6× SD |
| VQ-880-GII | — | 700/279 kHz | 1.5 ns | 40–280 cm | 1.5× SD |
| Chiroptera 4X | 0.1 mJ | 140/500 kHz | 4 ns | 120–180 cm | 1.7× SD |

Minimum depth = ~20 cm (for shallow bathy with ~1.5 ns pulse).
Maximum depth (deep bathy) = 3× Secchi depth ≈ 50 m at k=0.1.

**CZMIL sensor** features "Classification of water bottom" and "Seamless water-land-transition" capabilities.

---

### 5.03 Topo-Bathy_Application examples.pdf (Score 5)

*Slides confirm transcript content. Additional:*

- March 2021 data acquisition at Pielach: "leaf off, clear water."
- Standard waveform processing vs. SVB post-processing: SVB = Surface-Volume-Bottom (Schwarz et al. 2019) gives significantly deeper penetration.
- UAV-bathy (VQ-840G): cross sections colored by reflectance → water visible as distinct zone.
- SPL bathymetry in Vienna (cross section):
  - Danube: max penetration depth ~1 m (turbid)
  - New Danube: max penetration depth 2.5 m (standing water)
  - Old Danube: max penetration depth 1.8 m
  - Artificial pond: max penetration depth 1.8 m

---

### 6.03 UAV - Application examples.pdf (Score 4)

**Waveform Physics & Signal Properties**
- UAV-LiDAR point cloud at Hessigheim (Riegl VUX1-LR): colored by reflectance.
- Topo-bathy UAV LiDAR at Pielach river (Riegl VQ-840G): "River cross sections colored by reflectance."
- Slide caption: "Bathymetric LiDAR Data acquisition March 2021 (leaf off, clear water)."

**Domain Rules & Heuristics**
- Optimal acquisition conditions for Pielach: March, leaf-off, clear water.
- River cross sections visible from reflectance alone.