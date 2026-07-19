# lidarwater — Documentation

`lidarwater` (distribution name `lidar-water-detection`) classifies
full-waveform bathymetric LiDAR point clouds into **water / land / canopy /
uncertain** per point, and derives a vector **river boundary**. It was built
for topo-bathymetric UAV surveys (RIEGL VQ-840-GL, 532 nm green laser) of
narrow, tree-lined river valleys, where green-laser pulses return echoes
from the water surface, the water column, and the riverbed in a single
waveform.

It is a **library only**: you import it and call it from Python. There is no
CLI, no argparse, no script entry point.

## Documentation map

| Document | What it covers |
|---|---|
| [usage.md](usage.md) | Installing, loading data, running the pipeline, reading results, exporting, overriding configuration, training |
| [api.md](api.md) | Full reference: every public class, function, config field with its default and meaning, label semantics, artifact registry |
| [../README.md](../README.md) | Short quick-start |
| [../MIGRATION.md](../MIGRATION.md) | Mapping from the original per-script pipeline to this API, and what changed during the port |
| [../CLAUDE.md](../CLAUDE.md) | Domain background: the physics, the study area, why the algorithm looks the way it does |

## Features

**Classification pipeline** (`WaterPipeline.classify`)
- Waveform + geometric **feature extraction** from raw per-point waveforms
  (energy concentration, peak/gap structure, k-NN PCA planarity/roughness,
  local relative heights) — ~40 features per point, plus dense 200-bin
  waveform grids for the deep models.
- **WCN v9**, a transformer over waveform patches fused with 11
  generalizable scalar features, plus an XGBoost head on the same features
  — produces per-point water probabilities.
- **Geometry refinement**: concave-hull river footprint from
  high-confidence anchors, a local adaptive water-surface elevation grid
  (with RANSAC-plane fallback), footprint/surface-based classification,
  and waterbed reconstruction that recovers water points hidden under
  overhanging trees.
- **Canopy classifier**: XGBoost on echo-rank-within-pulse, water-aware
  height-above-reference, and local 3D structure features — separates tree
  canopy from ground/water without ever using absolute elevation as a
  feature.
- **Merge**: canopy predictions refine the land/uncertain classes; water
  labels always win conflicts.
- **River boundary**: iso-probability contours of the smoothed water
  probability field at three configurable thresholds (conservative /
  central / generous), as polylines.

**Data in**
- `PointCloud.from_dataframe(points_df, waveforms_df)` — bring your own
  loading; any row-aligned pair of point + waveform tables works.
- `read_pielach_txt(...)` — reader for this project's original two-file
  ASCII format.

**Data out**
- Everything in memory on a `PipelineState` (labels, probabilities,
  surface grid, footprint polygon, boundary contours, per-stage metrics).
- **LAS 1.4 / LAZ export** with ASPRS classification codes (topo-bathy
  profile by default: 40=bathymetric bottom, 41=water surface, 2=ground,
  5=high vegetation, 1=unclassified), CRS in the header, and the native
  label plus both model probabilities as Extra Bytes. Directly ingestible
  by OPALS (`opalsImport`), CloudCompare, QGIS, PDAL, LAStools.
- **GeoJSON export** of the river-boundary contours.

**Engineering**
- Every tunable is a field on a typed, validated config object
  (`PipelineConfig`) — no hardcoded thresholds. Defaults reproduce the
  values verified on the Pielach study area.
- Run the full cascade or any valid subset of stages; dependencies between
  stages are validated up front.
- Optional disk cache for the (slow) feature-extraction stage.
- Model weights resolved through a small `ArtifactResolver` seam — local
  directory today, swappable for a download-on-demand resolver later
  without touching pipeline code.
- Heavy dependency (`torch`) is an optional extra; installs without it can
  still run every XGBoost/geometry stage.
- Verified against the original pipeline's outputs on the real Pielach
  dataset (97% final-label agreement; deep-model outputs match to 1e-4 —
  see MIGRATION.md for the full parity report).

## The pipeline at a glance

```
PointCloud
   │
   ▼
FEATURES ──────────── waveform + geometric features, waveform grids
   │
   ▼
WCN ───────────────── transformer + XGBoost water probabilities
   │
   ▼
GEOMETRY ──────────── footprint, water-surface grid, classify, bed recon
   │                     labels: 0=land 1=water 2=uncertain 3=recon-water
   ▼
CANOPY ────────────── canopy probability per point
   │
   ▼
MERGE ─────────────── final label (adds 4=canopy)
   │
   ▼
BOUNDARY ──────────── river-edge contours (inner/center/outer)
```

`classify()` runs this chain with existing trained weights. `fit()` runs a
longer chain that also trains the weights (adds an AUTOLABEL bootstrap
stage before WCN). Any prefix or valid subset can be run via
`run_stages()`.
