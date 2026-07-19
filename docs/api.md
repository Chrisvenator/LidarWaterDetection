# API reference

Everything importable from `lidarwater` (the stable public surface), plus
the IO helpers under `lidarwater.io`. Modules prefixed `_` (`_stages`,
`_models`) are internal — importable, but their signatures may change
between minor versions.

Contents:
[WaterPipeline](#waterpipeline) ·
[PointCloud](#pointcloud) ·
[PipelineState](#pipelinestate) ·
[Stages & labels](#stages-and-label-semantics) ·
[Configuration](#configuration-reference) ·
[Artifacts](#artifact-resolution) ·
[IO](#io)

---

## WaterPipeline

```python
from lidarwater import WaterPipeline
```

| Member | Signature | Purpose |
|---|---|---|
| constructor | `WaterPipeline(config: PipelineConfig | None = None, artifacts: ArtifactResolver)` | `artifacts` is required; omitting it raises with instructions |
| `from_local_models` | `(models_dir, config=None) -> WaterPipeline` | Convenience: wraps `models_dir` in a `LocalArtifactResolver` |
| `classify` | `(cloud, stages=None) -> PipelineState` | Inference with existing weights. Default stages: FEATURES, WCN, GEOMETRY, CANOPY, MERGE, BOUNDARY (or `config.run.stages` if set) |
| `run_stages` | `(cloud, stages) -> PipelineState` | Explicit inference-mode subset, executed in fixed dependency order |
| `fit` | `(cloud, stages=None) -> PipelineState` | Trains all artifacts from raw data, then classifies. Adds AUTOLABEL and an extra v6-anchored geometry pass before WCN training. Stochastic; needs torch |

`classify()`/`fit()` never write result files — export explicitly via
`lidarwater.io`. Trained weights from `fit()` are written through the
artifact resolver.

---

## PointCloud

```python
from lidarwater import PointCloud
```

In-memory full-waveform cloud. Waveforms are ragged (per-point sample
count varies) and stored as flat arrays + offsets, not one object per
point.

| Attribute | Type | Meaning |
|---|---|---|
| `xyz` | `(N, 3) float64` | Coordinates |
| `reflectance_db` | `(N,) float32` | RIEGL reflectance (negative dB) |
| `waveform_times` | `(M,) int32` | All sample times, concatenated |
| `waveform_amps` | `(M,) float32` | All amplitudes, concatenated |
| `waveform_offsets` | `(N+1,) int64` | Point *i*'s samples = `[offsets[i]:offsets[i+1]]` |
| `pulse_id` | `(N,) int64` or `None` | Shared-pulse grouping. If `None`, the canopy stage derives it by hashing raw waveforms (points sharing a pulse have identical waveform rows) |

| Member | Notes |
|---|---|
| `len(cloud)`, `cloud.x / .y / .z` | Count and coordinate views |
| `waveform(i) -> (times, amps)` | One point's waveform |
| `iter_waveforms()` | Yields `(times, amps)` for every point in order |
| `PointCloud.from_dataframe(points, waveforms, *, x_col="x", y_col="y", z_col="z", reflectance_col="reflectance_dB", time_col="Time [SI]", amp_col="Amplitude [ADC]")` | Primary constructor. Row counts must match. Waveform cells may be lists, arrays, or strings (numpy-`repr` strings parsed by integer extraction) |

Construction validates shapes and raises `ValueError` on mismatches.

---

## PipelineState

Returned by every pipeline call. Stages append to it; nothing is removed,
so all intermediates stay inspectable. Fields are `None` until the
producing stage has run.

| Field | Set by | Type / meaning |
|---|---|---|
| `cloud` | constructor | The input `PointCloud` |
| `features` | FEATURES | `(N, ~40)` DataFrame — all scalar features |
| `waveform_grids` | FEATURES | `(N, 200) float32` dense amplitude grids |
| `waveform_grids_norm` | FEATURES | Per-sample max-normalised grids (WCN input) |
| `autolabel_xgb_proba`, `autolabel_deep_proba`, `autolabel_ensemble` | AUTOLABEL | v6 bootstrap outputs (fit path only) |
| `wcn_proba` | WCN | `(N,) float32` transformer water probability |
| `wcn_xgb_proba` | WCN | `(N,) float32` XGBoost water probability |
| `footprint_geom`, `footprint_raw_hull` | GEOMETRY | shapely (Multi)Polygon: eroded footprint, raw hull |
| `in_footprint` | GEOMETRY | `(N,) bool` |
| `local_surface_z` | GEOMETRY | `(N,) float32` water-surface elevation at each point |
| `surface_grid`, `surface_grid_origin`, `surface_plane_coef` | GEOMETRY | The surface elevation grid, its `(x_min, y_min)` origin, RANSAC plane `(a, b, c)` |
| `merged_label` | GEOMETRY | `(N,) int8` pre-reconstruction: 0/1/2 |
| `reconstructed_label` | GEOMETRY | `(N,) int8` post-reconstruction: 0/1/2/3 |
| `canopy_proba`, `canopy_pred` | CANOPY | `(N,) float32` / `int8` |
| `final_label` | MERGE | `(N,) int8` 0/1/2/3/4 — the headline result |
| `boundary_contours` | BOUNDARY | `{prob_level: [(M, 2) xy polylines]}` |
| `metrics` | every stage | `{stage_name: {…}}` summary counts/areas/fractions |

---

## Stages and label semantics

```python
from lidarwater import Stage
```

`Stage` values: `FEATURES`, `AUTOLABEL`, `WCN`, `GEOMETRY`, `CANOPY`,
`MERGE`, `BOUNDARY`.

Dependency rules (validated by `RunConfig` and enforced at stage runtime):

- `GEOMETRY` needs `WCN` **or** `AUTOLABEL` (probability anchors)
- `CANOPY` needs `GEOMETRY` (the water surface is its ground reference)
- `MERGE` needs `CANOPY`
- `BOUNDARY` needs `GEOMETRY` (uses `final_label` when MERGE ran, else
  `merged_label` + `wcn_proba`)

### Native labels

| Value | Meaning | Produced by |
|---|---|---|
| 0 | land | GEOMETRY |
| 1 | water (surface + riverbed) | GEOMETRY |
| 2 | uncertain | GEOMETRY |
| 3 | reconstructed water (tree-over-water recovery) | GEOMETRY (phase 3b) |
| 4 | canopy | MERGE |

### ASPRS mapping (LAS export)

Selected by `OutputConfig.label_scheme`:

| Native | `TOPO_BATHY` (default) | `ASPRS_BASIC` |
|---|---|---|
| 0 land | 2 (ground) | 2 (ground) |
| 1 water, z < local surface | 40 (bathymetric bottom) | 9 (water) |
| 1 water, z ≥ local surface | 41 (water surface) | 9 (water) |
| 2 uncertain | 1 (unclassified) | 1 (unclassified) |
| 3 recon-water | 40/41 by same z-split | 9 (water) |
| 4 canopy | 5 (high vegetation) | 5 (high vegetation) |

The native label is always preserved losslessly in the `raw_label` extra
byte, whatever scheme is chosen. `TOPO_BATHY` matches the class coding of
the Mandlburger Pielach reference dataset this project validates against.

---

## Configuration reference

```python
from lidarwater import PipelineConfig            # composes everything below
from lidarwater.config import (ZoneConfig, FeatureConfig, WcnConfig,
    SurfaceConfig, BoundaryConfig, CanopyConfig, OutputConfig, RunConfig,
    LabelScheme, Stage)
```

All configs are **frozen dataclasses**: build new ones with
`dataclasses.replace(old, field=value)`. `PipelineConfig()` with no
arguments reproduces the Pielach-verified defaults. Validation happens in
`__post_init__` and raises `ValueError` with a fix-it message.

### PipelineConfig

Fields: `zones`, `features`, `wcn`, `surface`, `boundary`, `canopy`,
`output`, `run` — each one of the classes below.

### RunConfig

| Field | Default | Meaning |
|---|---|---|
| `stages` | the classify chain | Default stage set for `classify()`; validated for dependency closure |
| `cache_dir` | `None` | If set, features + waveform grids cached here (parquet/npy) and reused |
| `plot_dir` | `None` | Reserved; currently unused (diagnostic plots were not ported) |
| `device` | `"auto"` | `"auto"` / `"cpu"` / `"cuda"` for the torch stages |

### ZoneConfig — elevation-band bootstrap labels (site-specific!)

| Field | Default | Band meaning |
|---|---|---|
| `z_underwater_max` | 259.6 | below: certain underwater |
| `z_water_surf_max` | 259.9 | to here: certain water surface |
| `z_dry_bed_min` / `z_dry_bed_max` | 260.0 / 260.4 | certain dry riverbed (land) |
| `z_banks_min` / `z_banks_max` | 260.9 / 263.1 | certain banks/meadow (land) |
| `z_canopy_min` | 263.3 | above: certain canopy (land for v6) |

Gaps between bands are excluded from training. These are Pielach survey
values — re-derive per site.

### FeatureConfig

| Field | Default | Meaning |
|---|---|---|
| `grid_size` | 200 | Waveform grid bins (must stay 200 for deployed checkpoints) |
| `knn_k` | 20 | Neighbours for PCA geometric features |
| `min_peak_adc` | 100 | Amplitude floor for peak detection |
| `gap_thresh_si` | 2 | Min time gap counting as a waveform gap |
| `local_min_radius_m` / `local_min_radius_10m` | 3.0 / 10.0 | `height_above_local_min` radii |
| `local_rank_radius_m` | 5.0 | `height_percentile_local` radius |
| `raster_cell_m` | 0.5 | Cell size for rasterised spatial ops |

### WcnConfig

`arch` (`WcnArchConfig`): `n_scalar=11, d_model=128, n_heads=8,
n_layers=6, n_patches=50`. **Must match deployed checkpoints** — change
only when training from scratch.

`train` (`WcnTrainConfig`) — fit() only: phase 1 (masked autoencoder:
`phase1_epochs=50, batch=1024, lr=1e-3, mask_ratio=0.40`), phase 2
(fine-tune: `30+120` epochs frozen/full, `lr 5e-4/2e-4`, `patience=20`),
phase 3 (pseudo-labels: `2 rounds × 30` epochs, `lr=5e-5`, thresholds
`0.92/0.08`), focal loss (`gamma=2.0, alpha=0.65, smooth=0.05`), aux-loss
weights `0.05/0.05`, `val_fraction=0.20`, `seed=42`.

### SurfaceConfig

Top-level: `water_tol_m=0.30` — inside the footprint, z ≤ surface + this
→ water.

`footprint` (`FootprintConfig`):

| Field | Default | Meaning |
|---|---|---|
| `conf` | 0.8 | Tier-1 (riverbed) anchor probability floor |
| `conf_surface` | 0.85 | Tier-2 (surface) anchor floor |
| `riverbed_z_max` | 259.6 | Tier-1 z ceiling |
| `riverbed_z_surface_max` | 261.5 | Tier-2 z ceiling |
| `hull_ratio` | 0.2 | Concave-hull tightness (lower = tighter) |
| `erosion_m` | 1.0 | Inward erosion of the hull (conservative footprint) |
| `tier2_max_dist_from_tier1_m` | 10.0 | Isolation filter for surface anchors |

`surface_grid` (`SurfaceGridConfig`): `cell_size_m=2.0`, water-like
selector (`z_lo=259.0, z_hi=261.5, n_peaks_max=2,
energy_concentration_min=0.85, reflectance_max_db=-15.0`), `z_cap=261.0`,
`min_pts_per_cell=5`, `smooth_sigma_cells=1.0`,
`max_dist_from_tier1_m=12.0` (beyond → RANSAC plane fallback),
`ransac_residual_m=0.20`, `ransac_max_rise_m=0.15`.

`bed` (`BedReconstructionConfig`): `min_pts_per_cell=3`, `max_dist_m=6.0`,
`margin_m=0.5`, `proba_min=0.95`, `recon_label=3`,
`reflectance_max_db=-15.0`, `min_peaks=3`, `planarity_min=0.30`.

### BoundaryConfig

| Field | Default | Meaning |
|---|---|---|
| `cell_size_m` | 0.5 | Probability-field raster resolution |
| `smooth_sigma_m` | 1.5 | Gaussian smoothing of the field |
| `prob_inner` / `prob_center` / `prob_outer` | 0.65 / 0.50 / 0.35 | Contour levels; must be strictly decreasing outward (validated) |
| `min_seg_len` | 20 | Discard shorter contour segments |
| `max_contours` | 3 | Keep at most this many segments per level |
| `max_fill_dist_m` | 3.0 | No-data mask distance (final-label mode) |
| `isolation_radius_m` / `min_water_support` | 3.0 / 5 | Drop isolated water strays before rasterising |
| `canopy_z_max` | 268.0 | z-based canopy exclusion (pre-merge fallback mode only) |

### CanopyConfig

| Field | Default | Meaning |
|---|---|---|
| `z_canopy_min` / `z_clear_max` | 263.283 / 260.1 | Training-label z bands (site-specific) |
| `surface_tol_m` | 0.15 | Water-surface pseudo-negative tolerance |
| `threshold` | 0.5 | Canopy decision threshold |
| `open_sky_max_above` / `low_height_max_m` | 1 / 1.2 | Open-sky-low pseudo-negative rule |
| `cell_m` | 1.0 | DTM/DSM grid cell |
| `r_local_m` | 1.0 | Structure-feature neighborhood radius |
| `above_gap_m` | 2.0 | "Crown overhead" gap |
| `dtm_percentile` | 0.05 | Robust per-cell ground elevation |
| `n_folds` | 5 | Spatial CV folds (fit only) |

### OutputConfig

| Field | Default | Meaning |
|---|---|---|
| `crs_epsg` | 25833 | Written into the LAS header (ETRS89 / UTM 33N) |
| `xyz_offset` | (0, 0, 0) | Added to coordinates on export (restore real-world frame) |
| `label_scheme` | `LabelScheme.TOPO_BATHY` | See mapping table above |

---

## Artifact resolution

```python
from lidarwater import ArtifactResolver, LocalArtifactResolver, ArtifactId, ArtifactNotFound
```

Stages never hardcode weight paths; they call
`resolver.resolve(ArtifactId.X) -> Path` (raises `ArtifactNotFound` if
missing) and training stages call `resolve_for_write(...)` (creates parent
dirs). Any object with those two methods works — the seam for a future
download-on-demand resolver. `ARTIFACT_STAGES` (in
`lidarwater.artifacts`) records which stage needs which artifact.

### Artifact registry

`LocalArtifactResolver(root)` expects, relative to `root`:

| ArtifactId | Path | Needed by |
|---|---|---|
| `WCN_REFINED` | `wcn_v9/wcn_refined.pt` | WCN |
| `WCN_XGB` | `wcn_v9/wcn_xgb.json` | WCN |
| `WCN_STATS` | `wcn_v9/wcn_stats.json` | WCN |
| `CANOPY_XGB` | `canopy_v1/canopy_xgb.json` | CANOPY |
| `V6_XGB` | `labeling/v6_xgb.json` | AUTOLABEL (fit) |
| `V6_DEEP` | `labeling/v6_deep.pt` | AUTOLABEL (fit) |
| `V6_DEEP_STATS` | `labeling/v6_deep_stats.json` | AUTOLABEL (fit) |
| `V8_XGB` / `V8_DEEP` / `V8_DEEP_STATS` | `current/v8_*.{json,pt,json}` | reserved (v8 compatibility; unused by the default chains) |

`classify()` needs only the first four.

---

## IO

```python
from lidarwater.io import read_pielach_txt, write_laz, write_geojson, classification_codes, boundary_to_geojson
```

| Function | Signature | Notes |
|---|---|---|
| `read_pielach_txt` | `(point_cloud_path, waveform_path) -> PointCloud` | Original two-file ASCII format; chunked; handles `_riegl.reflectance` rename and numpy-repr waveform strings |
| `write_laz` | `(state, output_config, path) -> None` | LAS 1.4 / point format 6; `.laz` extension → compressed. Requires `state.final_label` (run MERGE). Extra bytes: `water_proba`, `canopy_proba`, `raw_label` |
| `classification_codes` | `(state, output_config) -> (N,) uint8` | The ASPRS code array `write_laz` uses, if you want it without writing a file |
| `write_geojson` | `(state, boundary_config, path) -> None` | Boundary contours as a LineString FeatureCollection. Requires `state.boundary_contours` (run BOUNDARY) |
| `boundary_to_geojson` | `(state, boundary_config) -> dict` | Same, as an in-memory GeoJSON dict |
