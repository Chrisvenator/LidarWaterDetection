I need your help building a comprehensive knowledge base for a machine learning project. Read this entire prompt carefully before starting. It contains critical domain information that must guide your extraction.

## Project Goal

Train an AI model that classifies **water vs. land** from **full-waveform bathymetric LiDAR point cloud data**. The model must generalize to new LiDAR scans, not just work on this one dataset. There are **NO labeled training data**. The first step is therefore to build a **rule-based auto-labeler** from domain knowledge, which will then bootstrap a supervised ML model.

## Critical Domain Context (already established)

The data comes from the **Pielach River** in eastern Austria, surveyed in **October 2024** using the following scanner:

- **Scanner:** RIEGL VQ-840-GL (topo-bathymetric UAV laser scanner)
- **Wavelength:** 532 nm (green laser). This is crucial: green lasers **penetrate water**. That means you get returns from the water surface, the water column, AND the riverbed from a single pulse. This is fundamentally different from NIR lasers (~905nm, ~1550nm) which get absorbed by water and only see the surface.
- **Scan mechanism:** Elliptical Palmer scan, lateral FoV ±20°, forward/backward FoV ±14°
- **Flying altitude:** ~60m AGL
- **Pulse repetition rate:** 199 kHz
- **Beam divergence:** 1 mrad
- **Waveform recording:** Full-waveform with ~0.5ns sample interval, amplitude in ADC units
- **Processing:** Online Waveform Processing (OWP) and Surface-Volume-Bottom (SVB) algorithm

A second scanner was also used in the same campaign:
- **Scanner:** RIEGL miniVUX-3UAV (topographic only)
- **Wavelength:** 905 nm (near-infrared). NIR lasers are absorbed by water, so they only detect the water surface. Water shows very low reflectance or signal dropout with NIR.

The dataset I have appears to come from the **green laser (VQ-840-GL)** based on the waveform characteristics. It was downsampled from 1.6 million to ~250k points at 15cm spacing. The area contains river, meadow, and vegetation.

**Key implication for waveform analysis:** Since this is a green (532nm) bathymetric laser:
- Water waveforms will often show **multiple peaks** (surface return + bottom return, sometimes water column backscatter in between)
- The **temporal spacing** between surface and bottom peaks encodes **water depth** (~0.5ns per sample interval, speed of light in water is ~0.225 m/ns)
- Land/vegetation waveforms will show different patterns (single strong return for hard surfaces, multiple returns for canopy)
- The **amplitude ratio** between surface and bottom returns, the **exponential decay** in the water column, and the **overall waveform shape** are the most discriminative features

## Data Format

The input data consists of two aligned files:

**point_cloud_df.txt** — One row per point:
- Columns: `x`, `y`, `z`, `_riegl.reflectance`
- x, y, z are in ETRS89/UTM 33N (EPSG: 25833)
- `_riegl.reflectance` is in dB (decibels), typically negative values
- Example row: x=-269.273, y=97.270, z=261.719, reflectance=-26.61 dB

**waveform_df.txt** — One row per point (aligned by index with point_cloud_df):
- Columns: `Time [SI]`, `Amplitude [ADC]`
- Each cell contains a list/array (stored as string representation)
- Time is in sample intervals (SI), where 1 SI ≈ 0.5 ns
- Amplitude is in analog-to-digital converter units (ADC)
- Note: One waveform can lead to multiple points during echo extraction, so multiple point cloud rows may share the same waveform
- The time values are NOT always contiguous. There can be gaps (the scanner only records segments around detected signal)

Example waveform (point 0):
- Time: [37, 38, 39, 40, 41, ... 176, 177, 178, 179]  ← note the gap between ~98 and 173
- Amplitude: [62, 680, 2054, 2836, 2474, ... 175, 70, 37, 30]

The gap in time values likely represents: the first cluster of samples is around the surface/near returns, and the later cluster (~173-179) could be a deeper return (bottom) or another surface.

## File Locations

- **Lecture transcriptions** (24 .txt files): `/home/chrisvenator/Documents/Uni/Topo/Transcripts/`
- **Lecture slide decks** (26 .pdf files): `/home/chrisvenator/Documents/Uni/Topo/Slides/`
- **Research papers** (PDFs): `/home/chrisvenator/Documents/Uni/Topo/Papers/`
  - This folder contains the Pielach River survey paper (DOI: 10.23784/HN130-06) and at least one other article by the professor. Read ALL PDFs in this folder. They are directly relevant.
- **Actual dataset**: `/home/chrisvenator/PycharmProjects/LidarWaterDetection/data/`
  - `point_cloud_df.txt` (~250k points)
  - `waveform_df.txt` (corresponding waveforms)
- **Output directory**: `/home/chrisvenator/PycharmProjects/LidarWaterDetection/context/`

Create the output directory if it does not exist.

---

## Task Overview

### Phase 1: Inventory and File Scanning

1. List all files in all three source directories (Transcripts, Slides, Papers) with filenames and sizes.
2. **For transcriptions (.txt):** Read each file. Write a 2-3 sentence summary of the topic.
3. **For slide decks (.pdf):** First try extracting text with `pdftotext` or `pypdf`. If there is very little text (image-heavy), use `pdftoppm` or `pdf2image` to convert key pages (first page, table of contents if present, ~every 5th page) to images. Visually inspect them to determine the topic. Write a 2-3 sentence summary.
4. **For papers (.pdf):** These are high priority. Extract text and read thoroughly. Write a detailed summary (5-10 sentences).
5. Output a full inventory as a markdown file:

| Filename | Directory | Type | Topic Summary | Relevance Score (1-5) |

### Phase 2: Relevance Scoring

Score each file 1-5 based on relevance to building a **water/land classifier from full-waveform bathymetric (green, 532nm) LiDAR data**. Use these criteria:

**Score 5 — Directly relevant:**
- Full-waveform LiDAR analysis, waveform decomposition, waveform feature extraction
- Water surface detection, bathymetric LiDAR, water body classification
- The SVB (Surface-Volume-Bottom) algorithm or OWP (Online Waveform Processing)
- Reflectance/backscatter properties of water vs. land vs. vegetation at 532nm or NIR
- Signal characteristics: amplitude, pulse width, echo shape, number of returns, peak spacing
- LiDAR point cloud classification methods
- The Pielach River survey or related river surveys
- Green laser / bathymetric laser interaction with water (refraction, absorption, scattering)

**Score 4 — Highly useful:**
- Deep learning or ML for point cloud classification (PointNet, random forests on LiDAR features, etc.)
- Feature engineering from LiDAR data (geometric features, intensity features, neighborhood features)
- Signal processing: Gaussian decomposition, peak fitting, deconvolution, exponential decomposition
- Terrain or land cover classification from remote sensing
- Refraction correction, Snell's law applied to bathymetric LiDAR

**Score 3 — Moderately useful:**
- General LiDAR principles, scanning geometry, laser-matter interaction
- General remote sensing classification methods
- General deep learning architectures (CNNs, RNNs, transformers) that could be adapted
- Radiometric correction, atmospheric effects
- Photogrammetry combined with LiDAR (photo bathymetry)

**Score 2 — Minor relevance:**
- General geodesy, coordinate systems, georeferencing
- GIS and spatial data handling
- Photogrammetry without LiDAR context

**Score 1 — Not relevant:**
- Unrelated topics (GNSS-only, leveling, cadastral survey, etc.)

### Phase 3: Deep Extraction

For every file scored **3 or above**, do a thorough content extraction. For PDFs with mostly images, convert more pages and inspect them visually. Extract ALL information relevant to these categories:

#### 3.1 Waveform Physics & Signal Properties
- How do waveforms differ for water surface, water column, riverbed, land, vegetation, buildings?
- What amplitude levels, pulse widths, number of echoes are expected for each surface type?
- How does the green (532nm) laser interact differently with water than NIR (905nm)?
- What is specular vs. diffuse reflection on water? How does scan angle affect this?
- What causes exponential decay in the water column backscatter?
- How does the SVB algorithm decompose bathymetric waveforms?
- What is OWP (Online Waveform Processing) and how does it extract points from waveforms?
- Riegl-specific information about VQ-840-GL or miniVUX-3UAV

#### 3.2 Feature Engineering for Classification
- Which features can be extracted from waveforms? (amplitude, width, rise time, fall time, skewness, kurtosis, area under curve, echo ratio, number of peaks, peak spacing, energy, etc.)
- Which features can be extracted from point clouds? (height, local planarity, roughness, normal vectors, density, eigenvalue-based features, etc.)
- Which features are most discriminative for water vs. land, specifically for green bathymetric LiDAR?
- Any mentioned thresholds, rules-of-thumb, or decision boundaries
- How to extract features from the variable-length waveform arrays

#### 3.3 Classification Methods & Architectures
- Any ML/DL methods mentioned for LiDAR classification
- Network architectures for point clouds (PointNet, PointNet++, DGCNN, etc.)
- Network architectures for 1D signal processing (1D CNNs, RNNs, transformers on sequences)
- Multi-modal architectures that combine spatial + signal features
- Loss functions, training strategies, data augmentation for point cloud or signal data
- Accuracy metrics and benchmarks for water/land classification
- Handling class imbalance (water is a smaller portion of the scene)
- Unsupervised or semi-supervised methods that could work without labels
- Knowledge distillation, distant supervision, or label bootstrapping techniques

#### 3.4 Preprocessing & Data Handling
- How to parse, align, and normalize waveform data
- Handling variable-length waveforms (padding, resampling, segmentation)
- Noise filtering, outlier removal for both point cloud and waveforms
- How to handle multi-return / multi-echo data from a single pulse
- How to handle the fact that one waveform can produce multiple extracted points
- Coordinate normalization or local reference frames

#### 3.5 Domain-Specific Rules & Heuristics for Auto-Labeling
**THIS IS THE MOST CRITICAL SECTION.** Since we have NO labels, the entire project depends on extracting rules that can automatically classify points. Look for:

- **Reflectance thresholds:** e.g. "water surface shows reflectance below X dB" or "vegetation above Y dB"
- **Waveform shape rules:** e.g. "water shows a surface peak followed by exponential decay then a bottom peak" vs. "land shows a single strong Gaussian peak"
- **Amplitude rules:** e.g. "water surface returns are typically below X ADC" or "strong returns above Y indicate solid ground"
- **Multi-peak rules:** e.g. "two peaks separated by N sample intervals indicates water of depth D"
- **Echo count rules:** e.g. "multiple returns in quick succession indicate vegetation canopy"
- **Elevation rules:** e.g. "water surfaces form locally flat areas at the lowest elevation in a corridor"
- **Geometric rules:** e.g. "water surfaces have very low local roughness and high planarity"
- **Spatial rules:** e.g. "water points cluster in elongated, connected regions following the river"
- **Any numerical values, dB thresholds, ADC thresholds, or quantitative criteria mentioned anywhere**
- **The SVB algorithm's criteria for distinguishing surface, volume, and bottom returns**

If the documents mention ANY numbers, thresholds, or quantitative criteria, extract them verbatim with source attribution. These are the most valuable pieces of information in the entire knowledge base.

#### 3.6 Practical Considerations
- Scan angle effects on water detection with green laser
- Water turbidity, depth, surface roughness effects
- Edge cases: wet soil, puddles, shadows, dark asphalt, shallow water where surface and bottom merge
- How the very shallow zone problem affects classification (surface and bottom echoes overlap)
- Season, time of day, sun glint effects

### Phase 4: Analyze the Actual Data

Before compiling the knowledge base, do a quick statistical analysis of the actual dataset to ground the extracted knowledge in reality:

```python
import pandas as pd
import numpy as np

# Load point cloud
pc = pd.read_csv('/home/chrisvenator/PycharmProjects/LidarWaterDetection/data/point_cloud_df.txt')

# Basic stats
print(f"Number of points: {len(pc)}")
print(f"\nReflectance stats:")
print(pc['_riegl.reflectance'].describe())
print(f"\nZ (elevation) stats:")
print(pc['z'].describe())

# Reflectance histogram (text-based)
hist, edges = np.histogram(pc['_riegl.reflectance'], bins=20)
for i, (count, edge) in enumerate(zip(hist, edges)):
    print(f"  {edge:.1f} to {edges[i+1]:.1f} dB: {count} points ({100*count/len(pc):.1f}%)")

# Check for bimodal distribution in reflectance (water vs land often shows two clusters)
print(f"\nElevation range: {pc['z'].min():.2f} to {pc['z'].max():.2f}")
```

Also sample a few waveforms and characterize them:
- How many peaks does a typical waveform have?
- What is the typical amplitude range?
- What do the time gaps look like (do they suggest surface + bottom returns)?
- How variable is waveform length across points?

Include these findings in the knowledge base as "Section 10: Dataset Characteristics."

### Phase 5: Compile Knowledge Base

Create a single, well-structured markdown file called `knowledge_base.md` that consolidates ALL extracted information:

```
# Water vs. Land Classification from Full-Waveform Bathymetric LiDAR — Knowledge Base

## 1. Problem Overview
(Task description, data format, scanner specs, key challenges, the no-label constraint)

## 2. Physics of Bathymetric LiDAR Waveforms
### 2.1 Green laser (532nm) interaction with water
### 2.2 Water surface returns (specular/diffuse reflection)
### 2.3 Water column backscatter and exponential decay
### 2.4 Bottom returns through water
### 2.5 NIR laser (905nm) interaction with water (for comparison)
### 2.6 Land and vegetation waveform characteristics
### 2.7 Reflectance properties by surface type

## 3. Waveform Processing Algorithms
### 3.1 Online Waveform Processing (OWP)
### 3.2 Surface-Volume-Bottom (SVB) algorithm
### 3.3 Gaussian decomposition and peak fitting
### 3.4 Exponential decomposition for water column

## 4. Feature Engineering
### 4.1 Waveform-derived features (with computation methods)
### 4.2 Point cloud-derived features (geometric, neighborhood)
### 4.3 Most discriminative features for water vs. land with green laser
### 4.4 How to handle variable-length waveform input

## 5. Classification Approaches
### 5.1 Traditional ML methods (random forests, SVM, etc.)
### 5.2 Deep learning for point clouds
### 5.3 Deep learning for 1D waveform signals
### 5.4 Multi-modal / multi-branch architectures
### 5.5 Unsupervised and semi-supervised approaches
### 5.6 Training strategies and best practices

## 6. Auto-Labeling Rules Catalog
### 6.1 Reflectance-based rules (with thresholds)
### 6.2 Waveform shape rules (with criteria)
### 6.3 Amplitude-based rules (with thresholds)
### 6.4 Multi-peak / echo-based rules
### 6.5 Elevation and geometry rules
### 6.6 Spatial context rules
### 6.7 Combined rule confidence scoring
### 6.8 Known edge cases and failure modes

## 7. Preprocessing Pipeline
### 7.1 Waveform parsing and normalization
### 7.2 Point cloud preprocessing
### 7.3 Handling the waveform-to-multipoint mapping
### 7.4 Data augmentation strategies

## 8. Evaluation Strategy
### 8.1 Metrics (accuracy, precision, recall, F1, IoU)
### 8.2 Expected accuracy ranges from literature
### 8.3 Validation without ground truth (visual, cross-validation, consistency)
### 8.4 Common pitfalls

## 9. The Pielach River Survey
### 9.1 Study area description
### 9.2 Scanner configuration and flight parameters
### 9.3 Data processing pipeline used
### 9.4 Accuracy metrics from the paper
### 9.5 Relevant findings for our classification task

## 10. Dataset Characteristics
### 10.1 Point cloud statistics (from actual data analysis)
### 10.2 Waveform statistics (from actual data analysis)
### 10.3 Reflectance distribution analysis
### 10.4 Observed waveform patterns

## 11. Gaps in Knowledge
(Topics NOT covered by the source documents that may need external research)

## 12. Source Reference Table
(Which information came from which file, with relevance scores)
```

### Phase 6: Architecture Recommendation

Based on everything extracted AND the actual data characteristics, write a detailed architecture recommendation. Consider:

**The specific nature of this problem:**
- Input per point: a variable-length 1D waveform (time + amplitude) PLUS 3D coordinates (x, y, z) PLUS scalar reflectance
- Output: binary classification (water / land)
- The waveform is the richest signal, coordinates provide spatial context
- No labels available initially, so the architecture must work well with potentially noisy auto-generated labels
- Must generalize to new scans (different rivers, different conditions)

**Address these questions:**
1. What is the recommended model architecture? Describe it layer by layer.
2. Why does this architecture fit this specific data? (not generic "CNNs are good")
3. How should the waveform branch process variable-length input? (1D CNN? RNN? Transformer? Padding + fixed-size?)
4. How should spatial features be incorporated? (separate branch? concatenation? attention?)
5. What should the input/output tensor shapes be?
6. What loss function? (consider class imbalance, noisy labels)
7. What data augmentation is appropriate for waveforms?
8. Training plan: batch size, learning rate, epochs, early stopping criteria
9. Should an ensemble of multiple models be used?
10. What is the minimum viable model (simplest thing that could work well) vs. the ideal model?
11. How to iteratively improve: train on auto-labels → predict → correct labels → retrain

**Also provide a concrete implementation plan:**
- Step 1: Rule-based auto-labeler (using the rules catalog from Section 6)
- Step 2: Feature extraction pipeline
- Step 3: Simple baseline model (e.g., random forest on hand-crafted features)
- Step 4: Deep learning model
- Step 5: Evaluation and iteration loop
- Estimated code structure (which Python files, what each does)

### Phase 7: Generate a Project Specification

Create a `project_spec.md` that is a self-contained document Claude Code could use in a future session to build the entire project. It should contain:
- Exact data paths and formats
- The auto-labeling rules (condensed)
- The chosen architecture with layer specifications
- The training pipeline steps
- Evaluation approach
- All domain knowledge needed to make correct implementation decisions
- Any important gotchas or edge cases

This document should be written so that someone (or an AI) with zero prior context could pick it up and build the classifier correctly.

---

## Output Files

Save all outputs to `/home/chrisvenator/PycharmProjects/LidarWaterDetection/context/`:

| File | Content |
|------|---------|
| `file_inventory.md` | Phase 1-2: Full file inventory with relevance scores |
| `knowledge_base.md` | Phase 3-5: Comprehensive domain knowledge |
| `architecture_recommendation.md` | Phase 6: Model architecture and training plan |
| `project_spec.md` | Phase 7: Self-contained build specification |

---

## Important Instructions

- **Install packages as needed:** `pip install pypdf pdf2image Pillow pandas numpy` (use `--break-system-packages` if needed). You may also need `sudo apt install poppler-utils` for PDF conversion.
- **Papers folder is highest priority.** Read every PDF in `/home/chrisvenator/Documents/Uni/Topo/Papers/` thoroughly. These are directly about this dataset and this research group.
- **Be thorough with image-heavy PDFs.** Convert enough pages to get useful information. If a PDF is entirely images with no extractable content even visually, note it and move on.
- **When extracting, ALWAYS note which file the information came from.** Use format: `[Source: filename.ext]`
- **If sources contradict each other**, note both versions and which source each came from.
- **Do NOT hallucinate or fill gaps with assumptions.** If the documents don't cover a topic, say so explicitly. Mark it in the "Gaps in Knowledge" section.
- **Quantitative information is gold.** Any time you find a number, threshold, dB value, accuracy percentage, or formula, extract it with exact values and source.
- **The auto-labeling rules catalog (Section 6) is the single most important output.** The entire project depends on it. Be exhaustive.
- **If you hit context limits**, save your progress to the output files and note where you left off. I will continue with: "Continue from where you left off. Check the output files to see what's done."