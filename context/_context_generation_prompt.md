I need your help building a comprehensive knowledge base for a machine learning project. The goal is to train an AI that classifies **water vs. land** from **full-waveform LiDAR point cloud data**. The input data consists of:
- A point cloud with x, y, z coordinates and Riegl reflectance values
- Corresponding waveforms for each point, given as time (in sample intervals ~0.5ns) and amplitude (in ADC units)

The domain knowledge I need is buried across 50 academic documents: 24 lecture transcriptions (.txt) and 26 slide decks (.pdf, mostly image-based). Your job is to extract, triage, and consolidate all relevant information.

## File Locations
- Transcriptions (24 .txt files): `/home/chrisvenator/Documents/Uni/Topo/Transcripts/`
- Slide decks (26 .pdf files): `/home/chrisvenator/Documents/Uni/Topo/Slides/`

## Task Overview

### Phase 1: Scan and Inventory
1. List all files in both directories with their filenames and sizes.
2. For each .txt transcription: read the file and write a 2-3 sentence summary of what the lecture covers.
3. For each .pdf slide deck: first try extracting text with `pdftotext` or `pypdf`. If there is very little text (image-heavy), use `pdftoppm` or `pdf2image` to convert key pages (first page, table of contents if present, ~every 5th page) to images, then visually inspect them to determine the topic. Write a 2-3 sentence summary.
4. Output a full inventory table as a markdown file: `filename | type (transcript/slides) | topic summary | relevance score (1-5)`

### Phase 2: Relevance Scoring
Score each file 1-5 based on how relevant it is to building a water/land classifier from full-waveform LiDAR data. Here are the topics to look for (score 5 = directly relevant, 1 = not relevant):

**Score 5 - Directly relevant:**
- Full-waveform LiDAR analysis, waveform decomposition, waveform features
- Water surface detection, bathymetric LiDAR, water body classification
- Reflectance/backscatter properties of water vs. land vs. vegetation
- Signal characteristics: amplitude, pulse width, echo shape, rise time, peak detection
- LiDAR point cloud classification methods

**Score 4 - Highly useful:**
- Deep learning or ML for point cloud classification (PointNet, 3D CNNs, random forests on LiDAR)
- Feature engineering from LiDAR data (geometric features, intensity features, neighborhood features)
- Signal processing: Gaussian decomposition, peak fitting, deconvolution
- Terrain classification, land cover classification from remote sensing

**Score 3 - Moderately useful:**
- General LiDAR principles, scanning geometry, calibration
- General remote sensing classification methods
- General deep learning architectures (CNNs, RNNs, attention) that could be adapted
- Radiometric correction, atmospheric effects on LiDAR

**Score 2 - Minor relevance:**
- General survey/geodesy concepts
- GIS and spatial data handling
- Photogrammetry (unless combined with LiDAR)

**Score 1 - Not relevant:**
- Unrelated topics (GNSS-only, leveling, cadastral survey, etc.)

### Phase 3: Deep Extraction (Score 3+ files only)
For every file scored 3 or above, do a thorough extraction. Read the full content carefully (for PDFs, convert more pages to images if needed). Extract ALL information relevant to these categories:

1. **Waveform Physics & Signal Properties**
   - How do waveforms differ for water, land, vegetation, buildings?
   - What is the expected amplitude, pulse width, number of echoes for each surface type?
   - How does reflectance relate to surface material?
   - What causes signal attenuation in water?
   - Riegl-specific information about their scanners and waveform recording

2. **Feature Engineering for Classification**
   - Which features can be extracted from waveforms? (amplitude, width, rise time, decay, skewness, kurtosis, area under curve, echo ratio, etc.)
   - Which features can be extracted from point clouds? (height, local planarity, roughness, normal vectors, density, etc.)
   - Which features are most discriminative for water vs. land?
   - Any mentioned thresholds, rules-of-thumb, or decision boundaries

3. **Classification Methods & Architectures**
   - Any ML/DL methods mentioned for LiDAR classification
   - Network architectures for point clouds or 1D signals
   - Loss functions, training strategies, data augmentation techniques
   - Accuracy metrics and benchmarks mentioned
   - Handling of class imbalance (water is usually a small portion of the scene)

4. **Preprocessing & Data Handling**
   - How to handle waveform data: alignment, normalization, resampling
   - Noise filtering, outlier removal
   - How to handle multi-return / multi-echo data
   - Coordinate systems, georeferencing considerations

5. **Domain-Specific Rules & Heuristics**
   - Any rules like "water has very low reflectance below -X dB"
   - "Water returns typically show single, weak echoes"
   - "Specular reflection causes signal dropout over water"
   - Anything about the Pielach River survey or similar river surveys

6. **Practical Considerations**
   - Scan angle effects on water detection
   - Time of day, water turbidity, surface roughness effects
   - Edge cases: wet soil, puddles, shadows, dark asphalt

### Phase 4: Compile Knowledge Base
Create a single, well-structured markdown file called `knowledge_base.md` that consolidates ALL extracted information. Structure it as follows:

```
# Water vs. Land Classification from Full-Waveform LiDAR - Knowledge Base

## 1. Problem Overview
(Brief description of the task, data format, expected challenges)

## 2. Physics of LiDAR Waveforms
### 2.1 How waveforms work
### 2.2 Water surface interaction
### 2.3 Land/vegetation interaction
### 2.4 Reflectance properties by surface type

## 3. Feature Engineering
### 3.1 Waveform-derived features
### 3.2 Point cloud-derived features
### 3.3 Most discriminative features for water vs. land

## 4. Classification Approaches
### 4.1 Traditional ML methods
### 4.2 Deep learning methods
### 4.3 Recommended architectures for this problem
### 4.4 Training strategies and best practices

## 5. Preprocessing Pipeline
### 5.1 Waveform preprocessing
### 5.2 Point cloud preprocessing
### 5.3 Data normalization and formatting

## 6. Domain Rules and Heuristics
### 6.1 Known thresholds and decision rules
### 6.2 Edge cases and failure modes

## 7. Evaluation
### 7.1 Metrics to use
### 7.2 Expected accuracy ranges
### 7.3 Common pitfalls

## 8. Source File Reference
(Table of which information came from which lecture/slide deck)
```

### Phase 5: Architecture Recommendation
Based on everything extracted, write a separate section recommending:
- The best model architecture(s) for this specific task
- Why this architecture fits (considering the data: 1D waveforms + 3D point coordinates + reflectance)
- A rough training plan
- What the input/output format should be
- Whether ensemble methods or multi-modal approaches would help

Save all outputs to `/home/chrisvenator/PycharmProjects/LidarWaterDetection/context/`:
- `file_inventory.md` (Phase 1-2 output)
- `knowledge_base.md` (Phase 3-4 output)
- `architecture_recommendation.md` (Phase 5 output)

## Important Notes
- Install any needed Python packages (`pip install pypdf pdf2image Pillow`). You may also need `poppler-utils` for `pdftoppm` (`sudo apt install poppler-utils`).
- If a PDF is entirely images and you cannot extract useful content even visually, note it in the inventory and move on.
- Be thorough. Every piece of domain knowledge matters for building an accurate classifier.
- When extracting, always note which file the information came from.
- If you find contradictory information between sources, note both versions and which source each came from.
- Do NOT hallucinate or fill in gaps with assumptions. If the documents don't cover a topic, say so explicitly in the knowledge base.