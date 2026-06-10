# Canopy Classifier — Plan & Progress

**Goal**: per-point canopy probability (leaves/branches vs everything else).
Handles branches hanging into water + canopy obstructing water.

## Key findings (TL;DR)

1. **z only labels, never a feature** — model uses relative features (height above
   reference, echo rank, points overhead, waveform shape), so it generalizes and can
   contradict the z bands where they are noisy.
2. **Water depth masquerades as vegetation height**: DTM under water = riverbed →
   surface points look 1.5–3m "tall". Reference must be max(DTM, water surface),
   applied everywhere (the v10 footprint erodes 1m inward and misses edge water).
3. **Open-sky-low rule**: ≤1 neighbor ≥2m overhead AND <1.2m above reference →
   ground cover, not canopy. Dipping branches survive (crown overhead). A
   flat-surface-only rule overcorrected ("rough ⇒ canopy") — negatives must cover
   the full roughness spectrum, not just the flat extreme.
4. **Echo features work**: waveform-row hashing recovers shared pulses (37% of
   points are multi-echo) → echo rank / last-echo flags, the classic geodesy signal.
5. **Final**: spatial-CV macro-F1 0.9984; merge resolved 43% of v10's uncertain
   class; all 16 user-flagged error points fixed.

## Label rule (from user, manual CloudCompare inspection)

| z band | label | n points |
|---|---|---|
| z > 263.283 | 1 = canopy (100%) | 126,482 (54.0%) |
| 260.408 < z ≤ 263.283 | unlabeled — model predicts | 49,526 (21.2%) |
| 260.1 < z ≤ 260.408 | unlabeled (user gap) — model predicts | 25,247 (10.8%) |
| z ≤ 260.1 | 0 = not canopy (water) | 32,769 (14.0%) |

**Verified against data**: HI band is genuinely vegetation-like (median planarity 0.26,
median n_peaks 5, only 1.2% flat+smooth ground-like). Label rule is sound.

**Known risk**: negative class = water only (no dry gravel/meadow negatives).
Mitigated by generalizable canopy features (echo rank, height-above-DTM, echo ratio).
If mid-band gravel gets predicted canopy → add pseudo-negative round (Stage C).

## Features (NO absolute x/y/z — must generalize)

1. **Echo features** (new, from pulse grouping): hash waveform rows → pulse_id;
   points sharing a waveform = echoes of one pulse. Per point: n_echoes, echo_rank
   (1 = highest z = first return), echo_rank_norm, is_single/is_last/is_first_multi/is_intermediate.
2. **DTM/DSM** (new, geodesy standard): 1 m grid. DTM from last+single echoes
   (p5 per cell, median-filtered, hole-filled). DSM = per-cell max.
   → height_above_dtm (CHM-style), depth_below_dsm (penetration depth).
3. **Local 3D structure** (new): r=1.0 m cylinder/sphere — echo_ratio (n_3D/n_2D,
   low = vertical dispersion = vegetation), n_above_2m (canopy cover above point),
   z_range_cyl, z_std_cyl, n_cyl.
4. **Existing** (features_current.csv): planarity, roughness, linearity, sphericity,
   height_range_local, height_std_local, height_percentile_local, n_peaks, n_gaps,
   n_clusters, time_span, energy_concentration, amplitude_weighted_center,
   active_bins_ratio, max_amp_norm_by_energy, depth_proxy_m, first_last_span, reflectance_dB.

NOT used: height_above_local_min_10m (river-in-radius bug, see memory/findings).

## Pipeline

```bash
source .venv/bin/activate
python src/features/canopy_features.py    # → data_processed/canopy_features.csv (cached, skip if exists)
python src/training/train_canopy.py       # → models/canopy_v1/ + pointclouds/labeled_pointcloud_canopy.csv
```

Model: XGBoost, 5-fold spatial CV (y-strips), scale_pos_weight for imbalance.
Output CSV columns: X,Y,Z,reflectance_dB,label_src(-1/0/1),canopy_proba,canopy_pred + key features.
CloudCompare check: color by canopy_pred / canopy_proba.

## Pipeline integration (Stage 5+6, June 2026)

Canopy runs AFTER v10 (needs the water surface as ground reference — water depth
otherwise masquerades as vegetation height; cannot run "before water").
`src/inference/merge_canopy.py` then feeds canopy back into the final water output:
v10 land/uncertain + canopy_pred=1 → label 4 (canopy); water labels (1, 3) win
conflicts (773 points, all at the water surface).
Result: 138,316 land→canopy, 4,197 uncertain→canopy (43% of v10's uncertain class
resolved). Final classes in labeled_pointcloud_final.csv:
0=land 33,934 | 1=water 48,459 | 2=uncertain 5,570 | 3=water-under-canopy 3,548 |
4=canopy 142,513. CLAUDE.md pipeline section updated (Stages 5 and 6).

## Status

- [x] Context read, z bands verified, pulse grouping confirmed feasible
- [x] Plan written (this file)
- [x] `src/features/canopy_features.py` written + run → canopy_features.csv (36 cols)
- [x] `src/training/train_canopy.py` written + run → model + labeled cloud
- [x] Diagnostics reviewed (cross-sections, proba histogram, importance)
- [x] Iteration done (2 rounds, see below)
- [ ] User manual check in CloudCompare ← NEXT

## Iteration log

**v1 round 1**: CV macro-F1 0.984. BUG: 11.7% of v10-water mid-band points predicted
canopy. Cause: DTM under water = riverbed → water depth masqueraded as vegetation
height (water surface points had height_above_dtm 1.6–2.8 m).

**v1 round 2**: two fixes:
1. `height_above_ref` = z − max(DTM, v10 water surface inside footprint);
   raw height_above_dtm excluded from model features (kept in CSV for inspection).
2. 13,950 water-surface pseudo-negatives (v10 water, in footprint, |z_above_surface|<0.15).

Result: CV macro-F1 **0.9942**, AUC 0.9999. v10-water-canopy conflicts: 2 points (0.06%).

**v1 round 3 (user feedback)**: user flagged two error clusters (both proba 1.0):
ground at (-220, 110, z≈260.5) and edge-water at (-269, 112, z≈260.2); grass at
z≈262 correctly NOT flagged (keep that way). Diagnosis: (a) bright flat ground in
canopy band = label noise pocket; (b) v10 footprint erodes 1 m inward → edge water
had no water-aware reference → riverbed-DTM masquerade again.
Fixes: water surface used EVERYWHERE (not footprint-gated; safe — surface
259.85–260.75 m sits far below canopy); first tried flat-open pseudo-negatives →
OVERCORRECTED (rough low grass flipped to canopy, 2/4 user grass points wrong).
Replaced with **open-sky-low rule**: n_above_2m ≤ 1 AND height_above_ref < 1.2 m
→ pseudo-negative (3) in transition band, excluded as noise in canopy band.
Dipping branches keep crown overhead (n_above_2m high) so they survive.

Result: CV macro-F1 **0.9984**, AUC 0.99996. ALL user-flagged points correct
(7 ground, 5 water, 4 grass → all 0). v10-water conflicts 0.17%. Full transition
band canopy fraction 23.1% (down from 34% — grass/gravel no longer flagged).
Cross-sections verified: overhanging branches at z 260–262 still detected.

## Resume instructions (if session cut)

1. Read this file. Check Status boxes.
2. Artifacts cached: canopy_features.csv exists → skip Stage A
   (`--force` to rebuild; Stage A now requires labeled_pointcloud_v10.csv for
   the water-aware reference surface).
3. Diagnostic plots in models/canopy_v1/. Final cloud: pointclouds/labeled_pointcloud_canopy.csv
   (CloudCompare: color by canopy_pred or canopy_proba; label_src = training band:
   -1 predicted, 0 water, 1 canopy, 2 water-surface pseudo-negative).
