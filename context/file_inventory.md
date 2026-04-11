# File Inventory

| Filename | Directory | Type | Topic Summary | Score |
|----------|-----------|------|---------------|-------|
| 1.01 120.143-2026S Videos TUWEL.txt | Transcripts | Transcript | Course overview: 25yr LiDAR history, UAV LiDAR, full-waveform, green wavelength bathymetry intro. Evolution 5kHz→6MHz scan rate, 50kg→1kg weight. | 3 |
| 1.02 Basics_LiDAR crash course LectureTube.txt | Transcripts | Transcript | ALS multi-sensor (GNSS+IMU+scanner), multi-target, echo width discriminator, radiometric content. Veg penetration via multiple echoes. | 3 |
| 1.03 Basics_Ranging LectureTube.txt | Transcripts | Transcript | Round-trip time, 1ns=30cm, pulse separability=c*tau/2. MTA. Pulse lengths 1-10ns typical. | 2 |
| 1.04 Basics_Scanning LectureTube.txt | Transcripts | Transcript | Scan mechanisms: oscillating mirror, rotating mirror, Palmer scanner (conical), Risley prism. Palmer key for bathymetric sensors. | 2 |
| 1.05 Basics_Laser beam LectureTube.txt | Transcripts | Transcript | Gaussian beam, divergence in mrad, footprint size. Green ~1mrad bathymetry (eye safety). Pielach River 3-sensor example. Surface/bottom echo separability. | 4 |
| 1.06 Basics_Echo detection LectureTube.txt | Transcripts | Transcript | Full waveform vs discrete digitization. Gaussian decomposition (peak fitting). Echo width, amplitude, range per echo. 0.5-1ns sample intervals. | 4 |
| 1.07 Basics_Direct georeferencing LectureTube.txt | Transcripts | Transcript | GNSS+IMU+scanner sensor model. Lever arm + boresight calibration. Coord transforms. | 2 |
| 1.08 Basics_Flight planning LectureTube.txt | Transcripts | Transcript | Strip planning, overlap, point density calc. Leaf-on/off effects on penetration. | 2 |
| 1.09 Basics_Quality assessment LectureTube.txt | Transcripts | Transcript | Point density, precision, relative/absolute accuracy. Water surfaces→data voids (specular). Strip adjustment. | 3 |
| 1.10 Basics _DTM generation LectureTube.txt | Transcripts | Transcript | Ground filtering. Echo width discriminator shrub vs ground. Deep learning classification: PointNet++, 3D sparse voxel CNN, DGCNN. | 3 |
| 2.01 Multispectral_Laser Radar Equation LectureTube.txt | Transcripts | Transcript | Laser radar eq derivation. Specular vs diffuse reflection. Water surface purely specular→data voids off-nadir. Backscattering solid angle omega small for water. | 5 |
| 2.02 Multispectral_Radiometric_Calibration LectureTube.txt | Transcripts | Transcript | Radiometric cal from waveform (amplitude×sigma≈area∝power). Range + incidence angle dependence. Cal constant derivation. | 3 |
| 2.03 Multispectral_Sensors_and_applications LectureTube.txt | Transcripts | Transcript | Multispectral scanners: Titan (532/1064/1550), VQ1560i DW (green+NIR). Point cloud classification via NDVI-like veg indices. KPConv deep learning classification. | 3 |
| 3.01 Hybrid_Sensor overview LectureTube.txt | Transcripts | Transcript | Hybrid LiDAR+camera overview. VQ-840-G topo-bathy scanner: green + optional IR. | 3 |
| 3.02 Hybrid_LiDAR-DIM LectureTube.txt | Transcripts | Transcript | LiDAR vs dense image matching. LiDAR penetrates veg; DIM top-surface only. Diffuse reflection needed for LiDAR. | 2 |
| 3.03 Hybrid_Sensor orientation LectureTube.txt | Transcripts | Transcript | Strip adjustment for UAV LiDAR. Hybrid orientation combining LiDAR+images. | 2 |
| 4.01 SPL_Measurement principle LectureTube.txt | Transcripts | Transcript | Linear vs Geiger vs Single Photon LiDAR. Linear mode can record full waveform. SPL-100: 532nm (bathy capable). | 3 |
| 4.02 SPL_GmLiDAR and SPL sensors LectureTube.txt | Transcripts | Transcript | SPL-100 specs: 532nm, 0.08mrad divergence, 25-60kHz PRF, 100 receivers/pulse. Palmer scanner. | 3 |
| 4.03 SPL_Pros and Cons LectureTube.txt | Transcripts | Transcript | SPL vs full-waveform comparison. Full-waveform: more precise, better penetration, lower roughness on slopes. | 3 |
| 5.01 Topo-Bathy_Measurement principle LectureTube.txt | Transcripts | Transcript | Green laser bathy physics: water surface reflection, refraction, column attenuation (exp decay, coeff k), bottom reflection. Snell's law. Signal velocity: 300,000 vs 225,000km/h air vs water. Two-medium problem. | 5 |
| 5.02 Topo-Bathy_Sensor overview LectureTube.txt | Transcripts | Transcript | Deep bathy (3×Secchi) vs shallow (1.5×Secchi). VQ-880G: shallow only, high PRF. Riegl VQ-840G: 12kg, Palmer, ±20° lateral, ±14° fwd/bwd, 50-200kHz, 1-6mrad, 2×Secchi depth. | 5 |
| 5.03 Topo-Bathy_Application examples LectureTube.txt | Transcripts | Transcript | Pielach River fluvial geomorphology. VQ-840-G: >50pts/m², ~5cm pt spacing, depth 0-3m. FWF post-proc (SVB, Schwarz et al. 2019) beats OWP in turbid/deep areas. | 5 |
| 6.01 UAV - Whats new LectureTube.txt | Transcripts | Transcript | UAV LiDAR capabilities, corridor mapping, sub-cm resolution. | 2 |
| 6.02 UAV - Sensors and platforms LectureTube.txt | Transcripts | Transcript | VQ-840-G: 12kg, 532nm, 200kHz, 1-6mrad, ±20° lateral FoV, ±14° fwd/bwd. miniVUX: 1.8kg, 905nm, 200kHz, 1.6×0.5mrad. Full sensor spec table. | 5 |
| 1.01 Basics_LiDAR timeline.pdf | Slides | Slide | LiDAR tech timeline. Low relevance. | 2 |
| 1.02 Basics_LiDAR crash course.pdf | Slides | Slide | Basic LiDAR principles. | 2 |
| 1.03 Basics_Ranging.pdf | Slides | Slide | Ranging principles. | 2 |
| 1.04 Basics_Scanning.pdf | Slides | Slide | Scan mechanisms. | 2 |
| 1.05 Basics_Laser beam.pdf | Slides | Slide | Beam divergence, footprint. | 3 |
| 1.06 Basics_Echo detection.pdf | Slides | Slide | Full waveform: Gaussian decomp, echo width, amplitude, backscatter cross-section. ADC digitization. | 4 |
| 1.07 Basics_Direct georeferencing.pdf | Slides | Slide | Coord transform. | 1 |
| 1.08 Basics_Flight planning.pdf | Slides | Slide | Flight planning. | 1 |
| 1.09 Basics_Quality assessment.pdf | Slides | Slide | QA metrics. | 2 |
| 1.10 Basics _DTM generation.pdf | Slides | Slide | DTM gen, ground filtering, deep learning classification. | 3 |
| 2.01 Multispectral_Laser Radar Equation.pdf | Slides | Slide | Laser radar eq, specular vs diffuse, water data voids. | 4 |
| 2.02 Multispectral_Radiometric_Calibration.pdf | Slides | Slide | Radiometric cal, waveform amplitude×sigma=received power proxy. | 3 |
| 2.03 Multispectral_Sensors_and_applications.pdf | Slides | Slide | Multispectral sensors, KPConv classification. | 3 |
| 3.01 Hybrid_Sensor overview.pdf | Slides | Slide | Hybrid sensor systems. | 2 |
| 3.02 Hybrid_LiDAR-DIM.pdf | Slides | Slide | LiDAR vs DIM comparison. | 2 |
| 3.03 Hybrid_Sensor orientation.pdf | Slides | Slide | Strip adjustment. | 2 |
| 4.01 SPL_Measurement principle.pdf | Slides | Slide | SPL principles. | 3 |
| 4.02 SPL_GmLiDAR and SPL sensors.pdf | Slides | Slide | SPL-100 specs: 532nm green. | 3 |
| 4.03 SPL_Pros and Cons.pdf | Slides | Slide | SPL vs linear mode comparison. | 3 |
| 4.04_SBL_GEDI_ICESat2.pdf | Slides | Slide | Space-based LiDAR (GEDI, ICESat-2). Low relevance. | 2 |
| 5.01 Topo-Bathy_Measurement principle.pdf | Slides | Slide | Laser radar eq for bathy: PWS (water surface), PWC (column, exp decay k), PWB (bottom). Snell's law. Signal velocity: 300,000 vs 225,000km/h. | 5 |
| 5.02 Topo-Bathy_Sensor overview.pdf | Slides | Slide | Sensor categories: deep vs shallow bathy. Key params table. Min depth ~20cm shallow, ~100cm deep. VQ-840-G listed. | 5 |
| 5.03 Topo-Bathy_Application examples.pdf | Slides | Slide | Pielach River, VQ-840-G: >50pts/m², ~5cm spacing, OWP vs SVB comparison, depth ≤3m. | 5 |
| 6.01 UAV - What's new.pdf | Slides | Slide | UAV LiDAR applications. | 2 |
| 6.02 UAV - Sensors and platforms.pdf | Slides | Slide | Sensor spec table, VQ-840-G: 12kg, 532nm, 200kHz, 1-6mrad, ±40° FoV, ±50-150m altitude, 5-90cm footprint, Palmer scan, 2.0 SD depth. | 5 |
| 6.03 UAV - Application examples.pdf | Slides | Slide | UAV application examples. | 2 |
| DOI_10.23784_HN130-06.pdf | Papers | Paper | KEY PAPER: Mandlburger et al. 2025, "Mapping shallow inland running waters with UAV-borne photo and laser bathymetry — The Pielach River showcase." Oct 2024 survey: RIEGL VQ-840-GL (532nm, 60m AGL, 199kHz, 1mrad, elliptical Palmer ±20° lat ±14° fwd/bwd) + miniVUX-3UAV (905nm, 60m AGL, 300kHz). Open benchmark DOI:10.48436/taz19-r6618. OWP+SVB (Schwarz et al. 2019). Refraction correction via NIR water surface model. Vertical accuracy <2cm. ETRS89/UTM 33N (EPSG:25833). Waveform sample interval ≈0.5ns, amplitude in ADC units. | 5 |
| LIDARMagazine_Mandlburger-AirborneLidar2025_Part1.pdf | Papers | Paper | Tutorial I: LiDAR basics, ranging, scanning, waveform attrs. Modern ALS overview. | 3 |
| LIDARMagazine_Mandlburger-AirborneLidar2025_Part2.pdf | Papers | Paper | Tutorial II: Integrated sensor concepts, multispectral LiDAR. KPConv deep learning: geometry+green+NIR beats geometry alone. Green+NIR dual-wavelength→water/land separation. | 4 |
| LIDARMagazine_Mandlburger-AirborneLidar2025_Part3.pdf | Papers | Paper | Tutorial III: Laser bathy physics. Green laser–water interaction. Waveform: PWS+PWC+PWB. Exp decay k in water column. Specular reflection from surface. Min depth ~20cm. OWP vs FWF post-proc. VQ-840-GL at Pielach (Oct 2024). Refraction correction. Bottom reflectance affects penetration (gravel=high, mud/dark veg=low). | 5 |
| LIDARMagazine_Mandlburger-AirborneLidar2025_Part4.pdf | Papers | Paper | Tutorial IV: UAV-LiDAR concepts, sensor categories, GNSS/IMU/ranging/scanning. Green (532nm) bathy, IR topo. Full-waveform in survey-grade sensors. | 4 |