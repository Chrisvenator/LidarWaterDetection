# Migration: scripts -> `lidarwater`

Maps the old `python src/stage_N_script.py` workflow (see CLAUDE.md) onto the
`lidarwater` library API, and documents the deviations introduced during the
port and how they were verified.

## Command -> API

| Old command | New call |
|---|---|
| `python src/labeling/auto_labeler.py` | `WaterPipeline.fit(cloud, stages=[Stage.FEATURES, Stage.AUTOLABEL])` |
| `python src/labeling/water_surface_model.py` (full v8) | absorbed into `fit()`'s internal v6-anchored geometry pass — no direct equivalent call (see "Not ported" below) |
| `python src/training/preprocess_wcn.py` | absorbed into the features stage (`state.waveform_grids_norm`) — no separate call needed |
| `python src/training/train_wcn_v9.py` | `WaterPipeline.fit(cloud)` (stage `Stage.WCN`) |
| `python src/labeling/water_surface_model.py --geometry-only` (v10) | `WaterPipeline.classify(cloud)` (stage `Stage.GEOMETRY`) — this is the verified, primary path |
| `python src/features/canopy_features.py` + `python src/training/train_canopy.py` | `Stage.CANOPY` (part of `classify()`/`fit()`) |
| `python src/inference/merge_canopy.py` | `Stage.MERGE` |
| `python src/labeling/river_boundary.py` | `Stage.BOUNDARY`; GeoJSON via `lidarwater.io.write_geojson` |
| reading `data/Pielach/point_cloud_df.txt` + `data/Pielach/waveform_df.txt` | `lidarwater.io.read_pielach_txt(...)` |
| `pointclouds/labeled_pointcloud_final.csv` | `state.final_label` (+ `state.cloud`, `state.wcn_proba`, `state.canopy_proba`, ...) — export via `lidarwater.io.write_laz` for a durable artifact |

For the common case — classify a cloud with existing trained models — one
call replaces the whole `apply.py` orchestration:

```python
pipeline = WaterPipeline.from_local_models("models/")
state = pipeline.classify(cloud)
```

## What changed on purpose

- **CSV round-tripping replaced by in-memory `PipelineState`.** Every stage
  output (features, probas, labels, boundary geometry) is an attribute on
  `state`, not a file. Disk caching of the features stage is available via
  `RunConfig.cache_dir` (parquet + npy, not CSV — CSV float round-tripping
  is lossy and was a source of tiny numeric drift in the original pipeline).
- **Plotting dropped.** All the diagnostic PNGs the original scripts wrote
  (training curves, feature importance, cross-sections, topdown scatters)
  are not reproduced. They were visualization, not data the pipeline
  consumes downstream — regenerate ad hoc from `state` if needed.
- **Waveform parsing fixed, not just moved.** CLAUDE.md documents parsing
  waveform columns with `ast.literal_eval`; the actual `data/Pielach/waveform_df.txt`
  stores numpy's multi-line, space-separated `repr()` output, which isn't
  valid Python list syntax and makes `ast.literal_eval` raise. The scripts
  that actually work (`feature_extractor.py`) use a regex integer extractor
  instead — `lidarwater.io.read_pielach_txt` and
  `PointCloud.from_dataframe` do the same. CLAUDE.md should be corrected.
- **A latent XGBoost footgun fixed.** The canopy stage (`canopy_features.py`
  + `train_canopy.py`) originally passed a `pandas.DataFrame` with mixed
  int64/float32/float64 columns straight into
  `XGBClassifier.predict_proba()`. XGBoost's `inplace_predict` validates
  feature order against pandas' internal block layout, not `.columns` — a
  mixed-dtype frame gets silently regrouped by dtype, so the trained
  model's stored `feature_names` order does not match what a freshly-built
  DataFrame produces even when the *logical* column order is identical.
  This worked in the original script only because training and inference
  happened to share one in-process DataFrame; loading the saved model fresh
  (as this library always does) exposed it as a hard `ValueError`. Fixed by
  converting to a single-dtype `float32` array before predicting — the
  pattern every other stage's XGBoost usage already followed.

- **A second latent bug fixed: RANSAC `min_samples`.** The surface stage
  sized each RANSAC trial with `max(0.5, 100 / n_candidates)` — a
  *fraction*, which exceeds sklearn's legal 1.0 ceiling as soon as fewer
  than 200 candidates pass the water-surface gate. Pielach always had
  thousands, so the expression only ever evaluated to 0.5 there. On a site
  where the gate is tighter it raises `InvalidParameterError` and aborts
  the geometry stage. Now clamped in `_ransac_min_samples`: no trial can
  ask for more samples than exist.

## What was intentionally not ported

**v8's Phase 4 (XGBoost/V8Net retraining in `water_surface_model.py`, no
`--geometry-only`) is not reproduced.** In the original pipeline this
retrained model's only downstream consumer was to bootstrap WCN v9's own
training labels (`ensemble`, `xgb_proba`, `deep_proba` columns fed into
`train_wcn_v9.py` via `labeled_pointcloud_wcn.csv`). `fit()` produces an
equivalent bootstrap directly from the geometry stage's Phase 1-3b output
(`geometry_only=False`, anchored on the autolabel/v6 stage's probas)
instead of retraining and saving a separate V8Net/XGBoost artifact that
nothing else reads. If you need the v8 artifacts themselves (e.g. for
comparison against historical results), run the original
`water_surface_model.py` script directly — it hasn't been removed from
`src/`.

## What must not have changed (and how that was checked)

Verified by running `WaterPipeline.classify()` against the real Pielach
dataset (`data/`) using the already-trained checkpoints in `models/`, and
diffing against `pointclouds/labeled_pointcloud_final.csv` — see
`tests/test_golden_parity.py` (`pytest -m golden`) for the exact assertions
and current numbers:

- **WCN v9 transformer inference**: matches the deployed checkpoint's output
  to <1e-3 (observed max diff 0.0001, mean diff 0.000025 across 234,024
  points) — bit-for-bit modulo floating-point op ordering.
- **Phase 1 footprint** (concave hull + erosion): exact match — same
  footprint area (1245.0 m²) and same point count inside (97,162).
- **Phase 2/3 geometry labels** (pre-canopy land/water/uncertain): match
  within 2 points out of 234,024 per class (<0.001%).
- **Final label** (post-canopy, post-merge): 97.0% point-for-point
  agreement; per-class counts within 2% of golden. The residual 3%
  concentrates at the water-surface transition zone (z 259.8-260.8 m) and
  traces to the canopy stage's DTM nearest-neighbour fill, which has
  tie-breaking sensitivity at flat/equidistant cells — not a logic
  divergence in the ported algorithm.

Training-stage (`fit()`) numerics are explicitly *not* held to this bar —
WCN v9 training involves nondeterministic GPU ops and the original scripts'
own iterative self-bootstrapping across manual runs, so exact reproduction
was never well-defined even for the original pipeline.

`fit()` was nevertheless verified functionally end-to-end on the real
Pielach dataset (2026-07-19, RTX 4090, default epochs, 30.5 min wall):
the full bootstrap chain ran without intervention, WCN val macro-F1
progressed 0.888 → 0.899 → 0.900 across the three phases, all seven
artifacts were written through the resolver, and a fresh `classify()`
against the newly trained artifacts agreed with the golden output on
**93.3%** of points (water fraction 18.5% vs golden 22.2% — more
conservative, consistent with the skipped v8 retraining pass and with the
golden model's history of manual iteration).
