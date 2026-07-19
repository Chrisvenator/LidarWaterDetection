# lidar-water-detection

Water / land / canopy classification for full-waveform bathymetric LiDAR
point clouds, built for topo-bathymetric UAV surveys (RIEGL VQ-840-GL, 532 nm
green laser) of narrow river valleys. Installable library, called from Python
code — no CLI.

Approach: a rule-based geometry model (concave-hull river footprint + local
adaptive water-surface grid) bootstraps a supervised waveform transformer
(WCN v9) and a canopy XGBoost classifier, with no labeled training data
required to start.

See `CLAUDE.md` for the full domain background, algorithm design, and the
original per-script pipeline this library replaces. See `MIGRATION.md` for
how the old `python src/stage_N_script.py` workflow maps onto the API below.

## Install

```bash
pip install lidar-water-detection            # inference only (XGBoost stages)
pip install "lidar-water-detection[deep]"    # + WCN v9 / V6Net / V8Net (torch)
pip install "lidar-water-detection[train]"   # + retraining support
```

Trained model weights are not bundled — point a `LocalArtifactResolver` at a
directory laid out like this repository's `models/` tree (see
`src/lidarwater/artifacts.py` for the exact filenames expected).

## Quick start

### 1. Classify a point cloud with existing trained models

```python
import pandas as pd
from lidarwater import WaterPipeline

points = pd.read_csv("data/point_cloud_df.txt")
waveforms = pd.read_csv("data/waveform_df.txt")

from lidarwater import PointCloud
cloud = PointCloud.from_dataframe(
    points.rename(columns={"_riegl.reflectance": "reflectance_dB"}), waveforms,
)

pipeline = WaterPipeline.from_local_models("models/")
state = pipeline.classify(cloud)

# state.final_label: 0=land, 1=water, 2=uncertain, 3=water-under-canopy, 4=canopy
print((state.final_label == 1).sum(), "water points")
```

### 2. Override thresholds and run a stage subset

```python
import dataclasses
from lidarwater import PipelineConfig, Stage, WaterPipeline
from lidarwater.config import BoundaryConfig

config = PipelineConfig(
    boundary=BoundaryConfig(prob_inner=0.7, prob_center=0.5, prob_outer=0.3),
)
pipeline = WaterPipeline.from_local_models("models/", config=config)

# Just the water classification, skip canopy/merge/boundary
state = pipeline.run_stages(cloud, stages=[Stage.WCN, Stage.GEOMETRY])
```

### 3. Export LAS/LAZ and hand off to OPALS

```python
from lidarwater.io import write_laz, write_geojson

state = pipeline.classify(cloud)
write_laz(state, config.output, "out/classified.laz")
write_geojson(state, config.boundary, "out/river_boundary.geojson")
```

```bash
# OPALS auto-detects LAS/LAZ; classification codes follow the ASPRS
# topo-bathy profile by default (2=ground, 5=high veg, 40=bathy bottom,
# 41=water surface, 1=unclassified for the "uncertain" class).
opalsImport -inFile out/classified.laz -outFile out/classified.odm
```

## Configuration

Every tunable that used to be a hardcoded module constant lives on
`PipelineConfig` (`src/lidarwater/config.py`), grouped by stage:
`ZoneConfig`, `FeatureConfig`, `WcnConfig`, `SurfaceConfig`,
`BoundaryConfig`, `CanopyConfig`, `OutputConfig`, `RunConfig`. Defaults
reproduce the values verified on the Pielach study area; override via
`dataclasses.replace(...)` or by constructing a nested config directly, as
in example 2 above.

## Training

`WaterPipeline.fit(cloud)` bootstraps and trains every model artifact from
raw data (needs the `train` extra). It's slower and stochastic — intended
for adapting the model to a new site, not routine use. See the `fit()`
docstring in `src/lidarwater/pipeline.py` for the exact stage chain and one
documented deviation from the original scripts (Phase 4 V8Net/XGBoost
retraining is not reproduced; see `MIGRATION.md`).

## Development

```bash
pip install -e ".[dev]"
pytest
```

`pytest -m golden` runs slow parity tests against the real Pielach dataset
and trained checkpoints (not run in CI — requires `data/` and `models/`,
both gitignored).
