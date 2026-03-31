# File Inventory

| Filename | Directory | Type | Topic Summary | Score |
|----------|-----------|------|---------------|-------|
| 1.01 120.143-2026S Videos TUWEL.txt | Transcripts | Transcript | Course overview: 25 years of LiDAR development, UAV LiDAR, full-waveform, green wavelength for bathymetry introduced briefly. Evolution from 5 kHz to 6 MHz scan rates, weight from 50 kg to 1 kg. | 3 |
| 1.02 Basics_LiDAR crash course LectureTube.txt | Transcripts | Transcript | ALS multi-sensor system (GNSS+IMU+scanner), multi-target capability, echo width as discriminator, radiometric content. Vegetation penetration via multiple echoes. | 3 |
| 1.03 Basics_Ranging LectureTube.txt | Transcripts | Transcript | Round-trip time measurement, 1 ns = 30 cm, pulse separability = c*tau/2. Multiple pulses in air (MTA). Pulse lengths 1-10 ns typical. | 2 |
| 1.04 Basics_Scanning LectureTube.txt | Transcripts | Transcript | Scanning mechanisms: oscillating mirror, rotating mirror, Palmer scanner (conical), Risley prism. Palmer scanner key for bathymetric sensors. | 2 |
| 1.05 Basics_Laser beam LectureTube.txt | Transcripts | Transcript | Gaussian beam model, beam divergence in mrad, footprint size. Green laser ~1 mrad for bathymetry (eye safety). Pielach River example with 3 sensors mentioned. Separability of surface/bottom echoes. | 4 |
| 1.06 Basics_Echo detection LectureTube.txt | Transcripts | Transcript | Full waveform vs discrete echo digitization. Gaussian decomposition (peak fitting). Echo width, amplitude, range per echo. 0.5-1 ns sample intervals mentioned. | 4 |
| 1.07 Basics_Direct georeferencing LectureTube.txt | Transcripts | Transcript | GNSS+IMU+scanner sensor model. Lever arm and boresight calibration. Coordinate transformations. | 2 |
| 1.08 Basics_Flight planning LectureTube.txt | Transcripts | Transcript | Flight strip planning, overlap, point density calculation. Leaf-on vs leaf-off effects on penetration. | 2 |
| 1.09 Basics_Quality assessment LectureTube.txt | Transcripts | Transcript | Point density, precision, relative/absolute accuracy. Water surfaces cause data voids (specular reflection). Strip adjustment. | 3 |
| 1.10 Basics _DTM generation LectureTube.txt | Transcripts | Transcript | Ground filtering methods. Echo width as discriminator for shrub vs ground. Deep learning for point cloud classification: PointNet++, 3D sparse voxel CNN, DGCNN. | 3 |
| 2.01 Multispectral_Laser Radar Equation LectureTube.txt | Transcripts | Transcript | Laser radar equation derivation. Specular vs diffuse reflection. Water surface as purely specular reflector — data voids when off-nadir. Backscattering solid angle omega small for water. | 5 |
| 2.02 Multispectral_Radiometric_Calibration LectureTube.txt | Transcripts | Transcript | Radiometric calibration from waveform (amplitude × sigma ≈ area under curve ∝ power). Range and incidence angle dependence. Calibration constant derivation. | 3 |
| 2.03 Multispectral_Sensors_and_applications LectureTube.txt | Transcripts | Transcript | Multispectral scanners: Titan (532/1064/1550), VQ1560i DW (green+NIR). Point cloud classification using NDVI-like vegetation indices. KPConv deep learning for classification. | 3 |
| 3.01 Hybrid_Sensor overview LectureTube.txt | Transcripts | Transcript | Hybrid LiDAR+camera systems overview. Mentions VQ-840-G topo-bathymetric scanner with green laser and optional IR. | 3 |
| 3.02 Hybrid_LiDAR-DIM LectureTube.txt | Transcripts | Transcript | LiDAR vs dense image matching comparison. LiDAR penetrates vegetation; DIM gets top-most surface. Diffuse reflection needed for LiDAR. | 2 |
| 3.03 Hybrid_Sensor orientation LectureTube.txt | Transcripts | Transcript | Strip adjustment methodology for UAV LiDAR. Hybrid orientation combining LiDAR and images. | 2 |
| 4.01 SPL_Measurement principle LectureTube.txt | Transcripts | Transcript | Linear mode vs Geiger mode vs Single Photon LiDAR. Linear mode can record full waveform. SPL-100 uses 532nm (bathymetric capability). | 3 |
| 4.02 SPL_GmLiDAR and SPL sensors LectureTube.txt | Transcripts | Transcript | SPL-100 specs: 532nm, 0.08 mrad beam divergence, 25-60 kHz PRF, 100 receivers per pulse. Palmer scanner. | 3 |
| 4.03 SPL_Pros and Cons LectureTube.txt | Transcripts | Transcript | SPL vs full-waveform LiDAR comparison. Full-waveform more precise, better penetration, lower roughness on slopes. | 3 |
| 5.01 Topo-Bathy_Measurement principle LectureTube.txt | Transcripts | Transcript | Green laser bathymetry physics: water surface reflection, refraction, water column attenuation (exponential decay with coefficient k), bottom reflection. Snell's law. Signal velocity: 300,000 vs 225,000 km/h in air vs water. Two-medium problem. | 5 |
| 5.02 Topo-Bathy_Sensor overview LectureTube.txt | Transcripts | Transcript | Deep bathy (3×Secchi depth) vs shallow bathy (1.5×Secchi depth) sensors. VQ-880G: shallow only, high PRF. Riegl VQ-840G: 12 kg, Palmer scanner, ±20° lateral, ±14° forward/backward, 50-200 kHz, 1-6 mrad beam divergence, 2×Secchi depth. | 5 |
| 5.03 Topo-Bathy_Application examples LectureTube.txt | Transcripts | Transcript | Pielach River applications (fluvial geomorphology). VQ-840-G at Pielach: >50 pts/m², ~5cm point distance, depth 0-3m. FWF post-processing (SVB algorithm, Schwarz et al. 2019) improves over OWP especially in turbid/deep areas. | 5 |
| 6.01 UAV - Whats new LectureTube.txt | Transcripts | Transcript | UAV LiDAR capabilities, corridor mapping, sub-cm resolution. | 2 |
| 6.02 UAV - Sensors and platforms LectureTube.txt | Transcripts | Transcript | VQ-840-G: 12 kg, 532nm, 200 kHz, 1-6 mrad, ±20° lateral FoV, ±14° forward/backward. miniVUX: 1.8 kg, 905nm, 200 kHz, 1.6×0.5 mrad. Sensor table with all specs. | 5 |
| 1.01 Basics_LiDAR timeline.pdf | Slides | Slide | LiDAR technology timeline. Less relevant. | 2 |
| 1.02 Basics_LiDAR crash course.pdf | Slides | Slide | Basic LiDAR principles. | 2 |
| 1.03 Basics_Ranging.pdf | Slides | Slide | Ranging principles. | 2 |
| 1.04 Basics_Scanning.pdf | Slides | Slide | Scanning mechanisms. | 2 |
| 1.05 Basics_Laser beam.pdf | Slides | Slide | Beam divergence, footprint. | 3 |
| 1.06 Basics_Echo detection.pdf | Slides | Slide | Full waveform: Gaussian decomposition, echo width, amplitude, backscatter cross-section. ADC digitization shown. | 4 |
| 1.07 Basics_Direct georeferencing.pdf | Slides | Slide | Coordinate transformation. | 1 |
| 1.08 Basics_Flight planning.pdf | Slides | Slide | Flight planning. | 1 |
| 1.09 Basics_Quality assessment.pdf | Slides | Slide | QA metrics. | 2 |
| 1.10 Basics _DTM generation.pdf | Slides | Slide | DTM generation, ground filtering, deep learning for classification. | 3 |
| 2.01 Multispectral_Laser Radar Equation.pdf | Slides | Slide | Laser radar equation, specular vs diffuse reflection, water data voids. | 4 |
| 2.02 Multispectral_Radiometric_Calibration.pdf | Slides | Slide | Radiometric calibration, waveform amplitude × sigma = received power proxy. | 3 |
| 2.03 Multispectral_Sensors_and_applications.pdf | Slides | Slide | Multispectral sensors, KPConv classification. | 3 |
| 3.01 Hybrid_Sensor overview.pdf | Slides | Slide | Hybrid sensor systems. | 2 |
| 3.02 Hybrid_LiDAR-DIM.pdf | Slides | Slide | LiDAR vs DIM comparison. | 2 |
| 3.03 Hybrid_Sensor orientation.pdf | Slides | Slide | Strip adjustment. | 2 |
| 4.01 SPL_Measurement principle.pdf | Slides | Slide | SPL principles. | 3 |
| 4.02 SPL_GmLiDAR and SPL sensors.pdf | Slides | Slide | SPL-100 specs: 532nm green. | 3 |
| 4.03 SPL_Pros and Cons.pdf | Slides | Slide | SPL vs linear mode comparison. | 3 |
| 4.04_SBL_GEDI_ICESat2.pdf | Slides | Slide | Space-based LiDAR (GEDI, ICESat-2). Less relevant. | 2 |
| 5.01 Topo-Bathy_Measurement principle.pdf | Slides | Slide | Laser radar equation for bathymetry: PWS (water surface), PWC (water column, exponential decay with k), PWB (bottom). Snell's law formulas. Signal velocity: 300,000 vs 225,000 km/h. | 5 |
| 5.02 Topo-Bathy_Sensor overview.pdf | Slides | Slide | Sensor categorization: deep vs shallow bathy. Key parameters table. Minimum depth ~20 cm for shallow bathy, ~100 cm for deep bathy. VQ-840-G listed. | 5 |
| 5.03 Topo-Bathy_Application examples.pdf | Slides | Slide | Pielach River, VQ-840-G: >50 pts/m², ~5cm spacing, OWP vs SVB comparison slide, depth up to 3m. | 5 |
| 6.01 UAV - What's new.pdf | Slides | Slide | UAV LiDAR applications. | 2 |
| 6.02 UAV - Sensors and platforms.pdf | Slides | Slide | Sensor specs table, VQ-840-G detailed specs: 12kg, 532nm, 200kHz, 1-6mrad, ±40° FoV, ±50-150m altitude, 5-90cm footprint, Palmer scan, 2.0 SD depth. | 5 |
| 6.03 UAV - Application examples.pdf | Slides | Slide | UAV application examples. | 2 |
| DOI_10.23784_HN130-06.pdf | Papers | Paper | THE KEY PAPER: Mandlburger et al. 2025, "Mapping shallow inland running waters with UAV-borne photo and laser bathymetry — The Pielach River showcase." October 2024 survey with RIEGL VQ-840-GL (532nm, 60m AGL, 199 kHz, 1 mrad, elliptical Palmer scan ±20° lateral ±14° forward/backward) and miniVUX-3UAV (905nm, 60m AGL, 300 kHz). Open benchmark dataset DOI:10.48436/taz19-r6618. OWP + SVB algorithm (Schwarz et al. 2019). Refraction correction via NIR water surface model. Vertical accuracy <2cm. ETRS89/UTM 33N (EPSG:25833). Waveform sample interval ≈0.5 ns, amplitude in ADC units. | 5 |
| LIDARMagazine_Mandlburger-AirborneLidar2025_Part1.pdf | Papers | Paper | Tutorial Part I: LiDAR basics, ranging, scanning, waveform attributes. General overview of modern ALS systems. | 3 |
| LIDARMagazine_Mandlburger-AirborneLidar2025_Part2.pdf | Papers | Paper | Tutorial Part II: Integrated sensor concepts, multispectral LiDAR. Deep learning (KPConv) for classification with geometry+green+NIR outperforms geometry alone. Green+NIR dual-wavelength enables water/land separation. | 4 |
| LIDARMagazine_Mandlburger-AirborneLidar2025_Part3.pdf | Papers | Paper | Tutorial Part III: Laser bathymetry principles. Physics of green laser–water interaction. Waveform signal: PWS+PWC+PWB. Exponential decay in water column with coefficient k. Specular reflection from water surface. Minimum depth ~20cm. OWP vs FWF post-processing. VQ-840-GL at Pielach (same October 2024 dataset). Refraction correction. Bottom reflectance affects penetration (gravel=high, mud/dark vegetation=low). | 5 |
| LIDARMagazine_Mandlburger-AirborneLidar2025_Part4.pdf | Papers | Paper | Tutorial Part IV: UAV-LiDAR concepts, sensor categories, GNSS/IMU/ranging/scanning details. Green (532nm) for bathymetry, IR for topography. Full-waveform digitization in survey-grade sensors. | 4 |
