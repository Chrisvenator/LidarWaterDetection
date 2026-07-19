# Usage guide

Everything here is copy-paste runnable from a Python session in a project
that has `lidar-water-detection` installed and a `models/` directory with
trained weights (this repository's own `models/` tree works as-is).

## 1. Install

```bash
pip install lidar-water-detection             # XGBoost + geometry stages only
pip install "lidar-water-detection[deep]"     # + torch (WCN transformer, V6Net)
pip install "lidar-water-detection[train]"    # + everything fit() needs
```

From this repository (development):

```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

The `deep` extra is required for the default `classify()` chain, because
the WCN stage loads a torch checkpoint. Without torch you can still run
`Stage.FEATURES` and, if you supply probabilities yourself, the geometry,
canopy, merge, and boundary stages (see §7).

**Model weights are not bundled.** You need a directory laid out like this
repository's `models/` tree (exact expected filenames: [api.md — Artifact
registry](api.md#artifact-registry)). A future release will support
fetching them; today you copy the directory.

## 2. Load your point cloud

The library's input container is `PointCloud`: per-point `x, y, z`,
reflectance (dB), and the raw full waveform (sample times + amplitudes,
variable length per point).

### From dataframes you loaded yourself (primary path)

```python
import pandas as pd
from lidarwater import PointCloud

points = pd.DataFrame({
    "x": ..., "y": ..., "z": ...,          # float coordinates
    "reflectance_dB": ...,                  # RIEGL reflectance, negative dB
})
waveforms = pd.DataFrame({
    "Time [SI]":       [...],   # per row: list/array of sample times (0.5 ns units)
    "Amplitude [ADC]": [...],   # per row: list/array of raw digitizer counts
})
# rows of `waveforms` must align 1:1 with rows of `points`

cloud = PointCloud.from_dataframe(points, waveforms)
```

Column names are parameters (`x_col=`, `reflectance_col=`, `time_col=`, …)
if yours differ. Waveform cells may be Python lists, numpy arrays, or
strings (numpy `repr()`-style strings are parsed by extracting integers).

### From the original Pielach two-file ASCII format

```python
from lidarwater.io import read_pielach_txt

cloud = read_pielach_txt("data/point_cloud_df.txt", "data/waveform_df.txt")
```

This handles the `_riegl.reflectance` column rename and the multi-line
numpy-repr waveform strings, chunked to bound memory.

## 3. Classify

```python
from lidarwater import WaterPipeline

pipeline = WaterPipeline.from_local_models("models/")
state = pipeline.classify(cloud)          # ~30 s for 234k points (GPU)
```

`classify()` runs FEATURES → WCN → GEOMETRY → CANOPY → MERGE → BOUNDARY
and returns a `PipelineState`. Nothing is written to disk.

## 4. Read the results

All results live on the returned state; the ones you usually want:

```python
state.final_label        # (N,) int8: 0=land 1=water 2=uncertain
                         #            3=water-under-canopy 4=canopy
state.wcn_proba          # (N,) float32 water probability (transformer)
state.canopy_proba       # (N,) float32 canopy probability
state.local_surface_z    # (N,) float32 estimated water-surface elevation
state.in_footprint       # (N,) bool   inside the eroded river footprint
state.boundary_contours  # {prob_level: [ (M,2) xy polyline, ...]}
state.footprint_geom     # shapely (Multi)Polygon of the river footprint
state.metrics            # per-stage summary dict (counts, areas, fractions)
```

Example — water area stats and a quick dataframe:

```python
import numpy as np
import pandas as pd

water = np.isin(state.final_label, (1, 3))    # incl. water under canopy
print(f"{water.sum():,} water points, footprint {state.footprint_geom.area:,.0f} m²")

df = pd.DataFrame({
    "x": state.cloud.x, "y": state.cloud.y, "z": state.cloud.z,
    "label": state.final_label, "p_water": state.wcn_proba,
})
```

Every intermediate artifact stays reachable too (`state.features` — the
full feature matrix, `state.merged_label` — pre-canopy labels,
`state.surface_grid`, `state.waveform_grids`, …); the complete field list
is in [api.md](api.md#pipelinestate).

## 5. Export

### LAS/LAZ (recommended hand-off to OPALS / GIS)

```python
from lidarwater.io import write_laz

write_laz(state, pipeline.config.output, "out/classified.laz")
```

Produces LAS 1.4, point format 6, EPSG:25833 WKT in the header (both
configurable), ASPRS topo-bathy classification codes, and three Extra
Bytes per point: `water_proba` (f32), `canopy_proba` (f32), `raw_label`
(u8 — the native 0–4 label, lossless). Consumers:

```bash
opalsImport -inFile out/classified.laz          # OPALS → ODM, attributes kept
# or open directly in CloudCompare / QGIS / PDAL
```

If your coordinates are in a local offset frame (the Pielach data is),
set `OutputConfig.xyz_offset` so the written file carries real UTM
coordinates — see §6.

### River boundary GeoJSON

```python
from lidarwater.io import write_geojson

write_geojson(state, pipeline.config.boundary, "out/river_boundary.geojson")
```

LineString features tagged `inner` / `center` / `outer` per probability
level, with segment index and point count in the properties.

## 6. Configure

All tunables live on `PipelineConfig` — nested frozen dataclasses, defaults
= the values verified on the Pielach site. Override by constructing the
nested config you care about, or with `dataclasses.replace`:

```python
import dataclasses
from pathlib import Path
from lidarwater import PipelineConfig, WaterPipeline
from lidarwater.config import BoundaryConfig, OutputConfig, RunConfig, LabelScheme

config = PipelineConfig(
    # tighter/looser boundary contours
    boundary=BoundaryConfig(prob_inner=0.75, prob_center=0.5, prob_outer=0.25),
    # restore real ETRS89/UTM33N coords and use plain water=9 coding
    output=OutputConfig(
        crs_epsg=25833,
        xyz_offset=(269.0, -97.0, 0.0),          # site-specific — check yours
        label_scheme=LabelScheme.ASPRS_BASIC,
    ),
    # cache the slow feature-extraction stage between runs
    run=RunConfig(cache_dir=Path(".lidarwater_cache")),
)

pipeline = WaterPipeline.from_local_models("models/", config=config)
```

Invalid combinations fail at construction time with a message saying what
to fix (e.g. boundary thresholds out of order, or a stage set missing a
dependency). The full field-by-field reference is in
[api.md](api.md#configuration-reference).

**Site adaptation warning:** `ZoneConfig` (elevation bands) and
`CanopyConfig`'s z-bounds are Pielach-specific survey values, not physics.
On a new site they must be re-derived (cross-section inspection in
CloudCompare, as documented in CLAUDE.md) before `fit()` gives sensible
bootstrap labels. `classify()` with pre-trained weights is less sensitive
but the geometry stage's z-window constants also encode the site's water
level.

## 7. Run a subset of stages

```python
from lidarwater import Stage

# Just water probabilities + geometry, skip canopy/boundary:
state = pipeline.run_stages(cloud, stages=[Stage.WCN, Stage.GEOMETRY])

# Features only (no torch needed):
state = pipeline.run_stages(cloud, stages=[Stage.FEATURES])
feature_matrix = state.features            # (N, ~40) DataFrame
```

Stages always execute in the fixed dependency order; requesting an
invalid subset (e.g. CANOPY without GEOMETRY) raises immediately.

You can also inject your own probabilities and skip the deep model
entirely — useful without torch, or to test the geometry stages against a
different classifier:

```python
state = pipeline.run_stages(cloud, stages=[Stage.FEATURES])
state.wcn_proba = my_water_probabilities.astype("float32")
state.wcn_xgb_proba = my_water_probabilities.astype("float32")

from lidarwater._stages import surface, boundary
surface.run(state, pipeline.config.surface, geometry_only=True)
boundary.run(state, pipeline.config.boundary)
```

(The `_stages` modules are importable but underscore-private: their
signatures may change between minor versions; the `WaterPipeline` surface
is the stable API.)

## 8. Train on a new site (`fit()`)

```python
pipeline = WaterPipeline.from_local_models("my_models/")   # empty dir is fine
state = pipeline.fit(cloud)
```

`fit()` bootstraps labels from `ZoneConfig` elevation bands (AUTOLABEL),
trains the v6 waveform models, runs a first geometry pass anchored on
them, trains WCN v9 (masked-autoencoder pretraining → focal-loss
fine-tuning → pseudo-label refinement) and its XGBoost head, re-runs
geometry anchored on WCN, trains the canopy XGBoost, and finishes with
merge + boundary. Trained weights land in `my_models/` via the artifact
resolver, so a subsequent `classify()` against the same directory just
works.

Caveats, honestly stated:

- Needs the `train` extra and realistically a GPU (~30 min for 234k points
  on an RTX 4090 with default epochs).
- Training is stochastic; results are seeded but not bit-reproducible.
- Verified end-to-end on the real Pielach dataset: a from-scratch `fit()`
  run completes, writes all artifacts, and its round-trip `classify()`
  agrees with the original pipeline's output on 93.3% of points. It is
  somewhat more conservative about water (18.5% vs 22.2% water fraction)
  — expected, since `fit()` skips the original's v8 retraining pass (see
  MIGRATION.md) and the original model benefited from manual iteration.
- Re-derive `ZoneConfig` for your site first (see §6 warning).

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ImportError: ... pip install 'lidar-water-detection[deep]'` | Stage needs torch — install the extra, or run a stage subset that avoids WCN/AUTOLABEL |
| `ArtifactNotFound: Artifact 'wcn_refined' not found at ...` | Resolver root doesn't contain trained weights at the expected relative path — check the [artifact registry](api.md#artifact-registry) |
| `ValueError: points (...) and waveforms (...) row counts differ` | The two input tables must be row-aligned 1:1; re-check your loader |
| `ValueError: Stage.CANOPY requires Stage.GEOMETRY ...` | Stage subset missing a dependency — add the named stage |
| `ValueError: geometry_only=True needs state.wcn_xgb_proba and state.wcn_proba` | You ran GEOMETRY without WCN (or without injecting probas, §7) |
| Classification looks shifted / everything is land | Site water level differs from Pielach — z-window constants in `SurfaceConfig`/`ZoneConfig` need adapting (§6 warning) |
| Feature extraction is slow on repeated runs | Set `RunConfig.cache_dir` — features + waveform grids are cached as parquet/npy and reused |
