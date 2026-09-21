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

cloud = read_pielach_txt("data/Pielach/point_cloud_df.txt", "data/Pielach/waveform_df.txt")
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

**Site adaptation:** `ZoneConfig`'s elevation bands, `CanopyConfig`'s
z-bounds and the geometry stage's z-windows are Pielach survey values, not
physics — they encode that site's water level. On a different survey, call
`derive_site_config` (§8) to rebase them, then override anything that still
looks wrong here.

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

## 8. Adapt to a different survey (`derive_site_config`)

The defaults are Pielach's. Rather than hand-tuning them per site,
`derive_site_config` measures what they depend on and rebases them:

```python
from lidarwater import Workspace, WaterPipeline, derive_site_config
from lidarwater.io import read_dataset_dir

cloud = read_dataset_dir("data/Inn_DeepLearning")
config, profile = derive_site_config(cloud)
print(profile.summary())
```

```
points                  230,559
water level (densest z) 380.76 m
z shift vs Pielach      +120.47 m
reflectance shift       +14.1 dB
waveform grid origin    first_return (energy in first bins 8.2%)
canopy expected         True (1.28% of points >3 m above ground)
```

| Property | Measured from | Rebases |
|---|---|---|
| Water level | densest 0.1 m elevation bin | every absolute `z` threshold across `ZoneConfig`, `FootprintConfig`, `SurfaceGridConfig`, `BoundaryConfig`, `CanopyConfig` |
| Reflectance scale | percentile matching the -15 dB Pielach gate | `reflectance_max_db`, `ransac_reflectance_max_db` |
| Waveform record type | share of energy inside the first `grid_size` samples | `FeatureConfig.grid_origin` |
| Compact-waveform gate | percentile matching Pielach's 0.85 `energy_concentration` gate | `SurfaceGridConfig.energy_concentration_min` |
| Canopy presence | share of points >3 m above a per-cell ground surface | canopy stage output |

Architecture, training hyperparameters and dimensionless ratios are never
touched. On the Pielach cloud the z and reflectance rebasing is exact and
the compact-waveform gate estimates 0.845 against the 0.85 default;
`pytest -m golden` runs against raw defaults and still passes. Override anything that looks wrong with
`dataclasses.replace` afterwards — the derived config is an ordinary
`PipelineConfig`.

### Six things it handles that would otherwise fail silently

**Full-record waveforms.** SVB-clustered exports (Pielach) store only
samples around each echo, so `times[0]` *is* the first return. Other
processing chains store the entire range gate: the Inn export digitises
~700 samples of which the first ~550 are pre-trigger noise. Anchoring the
dense grid at `times[0]` then fills it with noise — measured on the Inn
cloud, only **7.8%** of waveform energy landed in the 200-bin grid.
`grid_origin="first_return"` anchors on the first sample rising clear of
the noise floor instead, raising capture to **87%**. Selected
automatically; `grid_size` and therefore the WCN input shape are unchanged.

**Pulse length.** `energy_concentration` is the share of waveform energy in
a fixed 30-bin window (`features.ENERGY_CONCENTRATION_BINS`) sized to
Pielach's ~60-bin pulse, and the geometry stage gates water-surface
candidates at `energy_concentration > 0.85`. Inn's returns occupy a median
of 153 bins, so no Inn point can reach 0.85: the gate admitted 48.6% of the
Pielach cloud but only 0.78% of Inn's, collapsing the RANSAC candidate set
from thousands to 79 and aborting the geometry stage. The threshold is
rebased to the value passing the same share of the cloud (0.294 on Inn),
which restores 13,481 candidates. Note this rebases the *threshold*, not
the window — on a site with a very different pulse length the feature's
discriminative power is reduced even once the gate is corrected.

**Full-record noise pedestal.** A full-range-gate digitisation stores every
sample, so ~80% of each record is digitiser noise sitting well above zero.
The gap, cluster and occupancy features then describe the *recording format*
rather than the returns, and the WCN's occupancy-mask input channel — which
on an SVB export marks where the echo clusters are — becomes all-ones and
carries no information. `FeatureConfig.noise_gate` drops samples at the
noise floor, recovering the sparse echo-only form. Measured on Inn, this
pulled `n_gaps` from +4.86 sigma of the Pielach training distribution to
-0.81, `n_clusters` likewise, and `energy_ratio_late` from +3.02 to +0.66.

`WcnConfig.standardize="site"` is a companion mitigation, not a fix. The
checkpoint ships its training set's scalar mean/std, which is a property of
that survey; applying it to another site pushed 25.6% of Inn's feature
values beyond 3 sigma and saturated the network to 97.6% water. Recomputing
from the cloud under test restores dynamic range (68.6% water, and a real
probability spread rather than a wall of 1.0). It does **not** restore
accuracy: the transformer and XGBoost heads still agree on only 39% of Inn
points either way, because both learned their decision boundaries in
Pielach-normalised coordinates. Cross-site inference is a smoke test; use
`fit()` for results.

**Labels that secretly encode elevation.** This is the one that matters most.
Every label in the pipeline descends from `create_labels(z, zones)`, which
takes `z` and nothing else. The WCN never uses elevation as a *feature*, but
elevation is its entire *teaching signal* — so it learns to reproduce an
elevation rule from waveforms, and succeeds: on Inn it reached F1 0.992
against its own bootstrap, and **97.68% of the final output is reproduced by
the single rule `z < 380.44 m`**. Where elevation genuinely separates water
from land that is harmless. Where it does not, the model is confidently
wrong rather than randomly wrong, which is worse: retraining on the z-bands
*lowered* accuracy against hand-labelled points from 74.8% to 59.0%.

`BootstrapConfig(method="surface")` removes absolute elevation from the
bootstrap entirely — rasterise, take each cell's top surface, keep the cells
whose top agrees with their neighbours' (a coherent sheet), and split those
by the median reflectance of the surface layer alone. Only *relative* height
within a cell is used. Against hand-labelled points on Inn:

| bootstrap | water | land | balanced |
|---|---|---|---|
| `"zones"` (elevation bands) | 27.6% | 80.6% | 54.1% |
| `"surface"` | 100% | 94.4% | **97.2%** |

Two things make it work, and both are easy to get wrong. Isolating the
surface layer is what makes the reflectance histogram bimodal: over *all*
points the deepest Otsu valley falls inside the water mode (-7.10 dB, 61.0%),
while over surface layers it lands at -0.46 dB and 97.2% — within 0.7 dB of
the hand-labelled optimum, with no labels used. And flatness alone is not
enough, because a gravel bar is also a flat coherent sheet: cell-wise
flatness gives 100% water recall but only ~40% on land, and *per-point*
planarity is actively backwards (water 0.61, land 0.73, because a k-NN ball
around a water point also catches riverbed returns beneath it).

**Site-specific signal the shipped model cannot use.** WCN v9's 11 scalar
features are dimensionless by design; `reflectance_dB` is excluded because its
dB scale belongs to the scanner and the survey. That is right for a model meant
to travel, and wrong for one fitted to a single site — on Inn, reflectance
alone separates water from land at AUC 0.973 (against 0.924 for all 11 waveform
features; 0.983 together), and a single threshold at +0.25 dB reproduces 97.2%
of hand-labelled points. `derive_site_config` therefore appends it to
`WcnConfig.scalar_features`, taking `arch.n_scalar` from 11 to 12.
`wcn.predict` reads the feature list from the checkpoint's stats file rather
than the config, so the shipped 11-feature model still loads unchanged and a
site-fitted 12-feature one round-trips.

**Sites without vegetation.** The canopy stage checks how much of the cloud
stands more than `CanopyConfig.probe_height_m` above the water-aware ground
reference. Below `min_canopy_frac` the site is treated as canopy-free and
the stage returns all-zero probabilities without loading (or training) a
model — a bare gravel-bed reach reports no canopy rather than whatever a
canopy-trained model extrapolates. `state.metrics["canopy"]["canopy_present"]`
records the decision.

### Keeping sites apart

`Workspace` gives each dataset its own output tree, so a new survey never
writes into another's results:

```python
workspace = Workspace.for_dataset("Inn_DeepLearning").mkdirs()
# runs/Inn_DeepLearning/{cache,models,pointclouds,plots}/

pipeline = WaterPipeline(config=workspace.apply_to(config),
                         artifacts=workspace.resolver())
```

Or from the command line, which wires all of the above together:

```bash
python scripts/run_dataset.py data/Inn_DeepLearning --profile-only
python scripts/run_dataset.py data/Inn_DeepLearning              # pretrained weights
python scripts/run_dataset.py data/Inn_DeepLearning --fit        # retrain for this site
```

Classification reads the shipped `models/` tree by default; `--fit` writes
into `runs/<dataset>/models/` and never touches it.

`read_dataset_dir` locates the point-cloud/waveform pair by pattern, so
survey-specific filenames (`point_cloud_df_inn.txt`) need no extra
argument.

## 9. Train on a new site (`fit()`)

> **Fit once per survey.** Models do not transfer between sites. Waveform-only
> classification reaches AUC 0.989 (Pielach) and 0.972 (Inn) *within* a site,
> and 0.42-0.69 *across* them — chance, in both directions, and unchanged by
> site-rank normalisation or by dropping the most site-dependent features.
> Waveform shape encodes the scanner's processing chain, not only the water.
> The one transferable signal is reflectance percentile rank, and adding
> waveform features to it degrades it. See
> [context/cross_site_transfer.md](../context/cross_site_transfer.md).

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
- Run `derive_site_config` first (§8) — `fit()` bootstraps its initial
  labels from `ZoneConfig`'s absolute elevation bands, which are Pielach's
  unless rebased.

### The uncertain class

`final_label == 2` marks points the two model heads disagree on, or that fall
outside the river footprint. It is a genuine confidence signal and is emitted
by default — but any consumer that treats it as "not water" pays for it.

```python
config = dataclasses.replace(config, surface=dataclasses.replace(
    config.surface, resolve_uncertain=True))
```

or `--resolve-uncertain` on `run_dataset.py`. Each abstention is then decided
by its own water probability. Measured on Inn:

| | full 192k hand labels | blind 65 points |
|---|---|---|
| emitting uncertain | 97.4% | 93.1% |
| resolving it | **98.7%** | **96.6%** |

Water recall goes 94.9% -> 97.9% for 0.3 points of land precision. Keep the
default if you want low-confidence points flagged for review; switch it on if
you want a decision everywhere.

### Speckle inside the channel

Classification is point-by-point, so nothing in the pipeline uses the fact
that water and land are contiguous. Single points that fall the wrong side of
the decision threshold survive as isolated specks — on Inn those sit at the
water surface (-0.04 m) with water's reflectance (-4.0 dB against land's
+0.8) and a median water probability of 0.42.

```python
config = dataclasses.replace(config, cleanup=dataclasses.replace(
    config.cleanup, majority_filter=True))
```

or `--majority-filter`. Canopy is never moved — it is genuinely sparse and a
spatial majority would erase it.

| | full 192k labels | blind 65 points |
|---|---|---|
| as shipped | 97.4% | 93.1% |
| `--resolve-uncertain` | 98.7% | 96.6% |
| both flags | **99.3%** | **98.3%** |

Off by default: it will also erase genuinely small features, which is a
judgement for the caller.

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ImportError: ... pip install 'lidar-water-detection[deep]'` | Stage needs torch — install the extra, or run a stage subset that avoids WCN/AUTOLABEL |
| `ArtifactNotFound: Artifact 'wcn_refined' not found at ...` | Resolver root doesn't contain trained weights at the expected relative path — check the [artifact registry](api.md#artifact-registry) |
| `ValueError: points (...) and waveforms (...) row counts differ` | The two input tables must be row-aligned 1:1; re-check your loader |
| `ValueError: Stage.CANOPY requires Stage.GEOMETRY ...` | Stage subset missing a dependency — add the named stage |
| `ValueError: geometry_only=True needs state.wcn_xgb_proba and state.wcn_proba` | You ran GEOMETRY without WCN (or without injecting probas, §7) |
| Classification looks shifted / everything is land | Site water level differs from Pielach — run `derive_site_config` (§8) |
| Water probabilities look random on a new survey | Waveforms may be full-record digitisations; check `profile.first_bin_energy_fraction` and `grid_origin` (§8) |
| `InvalidParameterError: 'min_samples' ... Got np.float64(1.26)` | Too few surface candidates for the RANSAC plane fit — check `energy_concentration_min` against the site's distribution (§8) |
| Everything on a bare site comes back as canopy | Check `state.metrics["canopy"]["canopy_present"]`; lower `CanopyConfig.min_canopy_frac` only if the site really has vegetation |
| Feature extraction is slow on repeated runs | Set `RunConfig.cache_dir` — features + waveform grids are cached as parquet/npy and reused |
