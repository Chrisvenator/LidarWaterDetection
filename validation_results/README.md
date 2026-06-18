# Validation Results — water-vs-land classifier

Reference: **Mandlburger et al., Pielach River, TU Wien, Oct 2024**
(DOI 10.48436/taz19-r6618, CC-BY 4.0) — same river, epoch, RIEGL topo-bathy sensor.

Predictions validated: `pointclouds/labeled_pointcloud_final.csv`
Regenerate everything: `python src/evaluation/compile_validation_results.py`

## Coordinate registration
Our cloud is local-offset in x,y (z already true ellipsoidal height).
Offset local -> ETRS89/UTM33N (EPSG:25833): **dx = 527925.283, dy = 5340171.595**
(found by FFT cross-correlation of the river-channel mask + hard-surface refine;
water z-residual ~+0.1 m confirms datum alignment). Our AOI overlaps only the
eastern ~half of the reference AOI.

## Headline results (3 independent angles)
| truth source | metric | value |
|---|---|---|
| Classified LiDAR, point-level | F1 / IoU | 0.931 / 0.871 |
| Classified LiDAR, point-level | precision / recall | 0.923 / 0.940 |
| Classified LiDAR, 2D channel footprint | IoU / F1 | 0.597 / 0.748 |
| Surveyed points (human truth) | accuracy | 0.848 (39/46) |
| Water surface vs WSM.tif | median / RMSE | +0.321 m / 0.352 m |

## Truth class mapping (reference file 03, ASPRS)
water = {40 bathy bottom, 41 water surface, 43 submerged};
land = {2 ground, 3/4/5 vegetation}; noise {7,18} excluded.
Our labels: 0 land, 1 water, 2 uncertain, 3 water-under-canopy, 4 canopy.

## Key finding
Channel core is solid; we slightly UNDER-call water at shallow/gravel channel
margins (false-negatives cluster at the water edge — see plots/ and
tables/margin_fn_diagnostic.csv).

## Caveats
1. Overlap = eastern half of the reference AOI only (partial coverage).
2. Classified LiDAR is itself algorithm output (OWP+SVB), not human truth.
3. Surveyed points are independent human truth but sparse (46 within our coverage).

## Folder
- `reports/`  raw JSON metrics + registration
- `tables/`   CSVs: metrics summary, class histograms, confusion matrices,
              per-point matches, survey matches, margin diagnostic
- `plots/`    water-extent overlay, survey-point map, margin-FN histograms
