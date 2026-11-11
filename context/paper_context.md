# Paper context — water/land classification from full-waveform bathymetric LiDAR without labels

Working document for drafting a publication from this repository.
Written 2026-10-10. Every number below has a named source in-repo; nothing here
is from memory. Items marked **[VERIFY]** are not yet established and must be
checked before they enter a draft.

---

## 1. What the repository actually did

A water-vs-land point classifier for 532 nm topo-bathymetric UAV LiDAR, built
with **no human labels at training time**. Labels are bootstrapped by a
rule-based auto-labeler from waveform physics, then used to train a deep
waveform model, then refined by a geometric water-surface model. The pipeline
was built on one survey (Pielach, Austria) and then forced to generalise to a
second, deliberately dissimilar one (Inn), which is where most of the
scientific content came from.

Two artefacts exist:
- `src/labeling/`, `src/features/`, `src/training/` — the Pielach-specific
  research pipeline (hardcoded paths, absolute z constants).
- `src/lidarwater/` — the generalised library: `derive_site_config(cloud)`
  measures four site properties and rebases the Pielach constants, so the same
  code runs on an arbitrary survey. Golden tests assert the derivation is an
  exact no-op on Pielach (`tests/test_golden_parity.py`, `pytest -m golden`).

---

## 2. Candidate theses — pick one before writing a word

**A. The negative-transfer result (recommended).**
*Waveform-shape features for water detection do not transfer between surveys,
because the strongest ones encode the vendor's waveform export format rather
than the physics. Per-site fitting with an elevation-free bootstrap is the
correct architecture.*
Strongest evidence, most surprising, hardest to get from a single dataset, and
it is a result other groups can act on. Risk: a negative result needs airtight
measurement, and n=2 surveys limits the claim (see §8).

**B. The label-free pipeline.**
*A rule-based waveform bootstrap plus geometric refinement reaches F1 0.911
(abstentions counted; 0.931 on decided points only) against an independent classified reference, with zero training labels.*
Cleanest story, easiest to write, but weakest novelty — label-free bootstrapping
is not new, and the validation is one site.

**C. The elevation-leakage result.**
*Elevation-band bootstrapping silently teaches the model an elevation rule.
97.68% of the Inn output is reproduced by the single rule `z < 380.44 m`,
despite z never being a model feature. An elevation-free surface bootstrap
fixes it: 61.0% -> 97.2% balanced accuracy.*
**Audit 2026-11-11: both figures in C are weaker than stated — see §3 and §11 Q2.
The 97.68% run is not on disk, and on the region labels a single z rule is
statistically indistinguishable from the pipeline.**
This is the sharpest single finding in the repo and the most reusable warning.
Works well as the core of A, or as a standalone short paper.

**Recommendation: A as the frame, C as the mechanism, B as the method section.**
C explains *why* the naive cross-site attempt fails before the feature analysis
even gets a say, and A is what the reader should do about it.

---

## 3. Evidence ledger

Strength column: **hard** = measured against human/independent labels,
**soft** = measured against pipeline-derived labels (partly circular),
**[VERIFY]** = not yet measured.

| Claim | Number | Source | Strength |
|---|---|---|---|
| Waveform-only classification works within a site | AUC 0.989 Pielach / 0.972 Inn (11 scalars); 0.983/0.986 (raw 200-bin grid) | `context/cross_site_transfer.md` §1 | soft; **not re-measured — no script for this experiment is committed** (§11) |
| …same, unbiased | AUC 0.902 on 65 hand-labelled Inn points (leave-one-out) | ibid. | **downgraded**: 65 points in 54 ten-metre blocks, land sample unrepresentative (§11 Q2); no committed script |
| Waveform-only transfer is chance | AUC 0.478 Pielach→Inn, 0.647 Inn→Pielach; site-rank encoding does not help (0.492 / 0.685) | ibid. §2 | **downgraded**: Inn side rests on the 65; no committed script |
| Dropping the four non-transferring features does not rescue transfer | 0.419–0.467 and 0.611–0.689 | ibid. §2 | soft; no committed script |
| Only reflectance *rank* transfers | AUC 0.958 (86.9% balanced) Pielach→Inn hand labels, reflectance rank alone | ibid. §3 | **downgraded to unsupported**: the 65's land points are 94% brighter than 0 dB, region land only 41%. A reflectance cut at +0.25 dB scores 97.2% on the 65 but 70.5% [64.3, 78.4] on the regions (§11 Q2) |
| Adding waveform features *degrades* that | 0.958 → 0.745 | ibid. §3 | **downgraded**: same biased 65; no interval; no committed script |
| Absolute dB is irreconcilable, rank nearly coincides | Pielach water/land −23.60/−13.57 dB; Inn −4.77/+0.01 dB; ranks 0.410/0.907 vs 0.484/0.879 | ibid. §3 | not re-checked; class means depend on which labels define water/land **[VERIFY]** |
| The four dead features describe the export format | `max_amp_norm_by_energy` −0.65→+0.07, `gap_ratio` −0.62→+0.13, `n_gaps` −0.56→+0.05, `n_clusters` −0.56→+0.05 (water−land shift, site-normalised) | ibid. §4 | soft (pipeline labels define the classes); not re-checked |
| What survives is consistent but weak | `energy_concentration` 0.33/0.29, `peak_amp_ratio` −0.24/−0.22, `depth_proxy_m` −0.20/−0.20 | ibid. §4 | soft; not re-checked |
| Elevation leakage | 97.68% of Inn final output reproduced by `z < 380.44 m` | `CLAUDE.md`, failure mode 5 | **downgraded**: the z-band run is not on disk, so 97.68% is unverifiable. Context it needs: the *corrected* output is itself 89.0% (base) / 94.7% (all flags) reproduced by `z < 380.77`, and against the region labels that one rule scores 96.0% [92.6, 98.3] vs the pipeline's 97.1% [96.2, 97.9] — indistinguishable. Only the 65 separate them (77.8% [68.8, 87.1] vs 93.1% [86.0, 98.6]), and that gap is the 15 "probably-land" points on one gravel bar |
| Elevation-free surface bootstrap fixes it | Otsu over all points −7.10 dB / 61.0% balanced; over surface layers −0.46 dB / 97.2% (100% water, 94.4% land); stable across 2–4 m cells | ibid. | **confirmed on the 65** (97.2%, block CI [93.0, 100], identical at 2/3/4 m). On the regions: 96.2% [94.8, 97.6] at 3 m, and *not* cell-stable (94.2% / 96.2% / 97.4% at 2/3/4 m). All-points Otsu figure not re-measured |
| Flatness alone is insufficient | cell flatness: 100% water recall, ~40% land; per-point planarity reverses (water 0.61, land 0.73) | ibid. | not re-checked |
| Pulse-length dependence of a fixed-window feature | `energy_concentration > 0.85` gate passes 48.6% of Pielach, 0.78% of Inn (153-bin vs 60-bin pulses); percentile rebase 0.294 → 13,481 candidates | ibid., failure mode 2 | hard (label-free measurement); gate 0.294 confirmed by re-run |
| Grid anchoring matters for full-record exports | energy captured 7.8% → 87% with `grid_origin="first_return"` | ibid., failure mode 1 | hard (label-free); not re-checked |
| Noise pedestal corrupts gap/cluster features | `n_gaps` +4.86σ → −0.81σ, `energy_ratio_late` +3.02 → +0.66 with `noise_gate` | ibid., failure mode 3 | hard (label-free); not re-checked |
| Final Inn accuracy, per-site fit | **Base config (all three flags off), fresh fit with current code: 99.4% balanced [99.1, 99.7] vs 180,394 region-labelled points; 96.6% [91.4, 100] vs the 65 picked points.** Same config with the stale 2026-09-06 models every earlier figure used: 97.1% [96.2, 97.9] / 93.1% [86.0, 98.6]. Flags on those stale models (dev-set numbers only): +`resolve_uncertain` 98.5% / 96.6%; +`majority_filter` 99.2% / 98.3%; +`surface_prior` 99.9% / 100% | `src/diagnostics/audit_inn_labels.py` → `runs/Inn_audit_{base,ru,ru_mf,all,refit,refit_all}/audit_labels.json` | **"99.9% vs 208k hand labels" WITHDRAWN.** 208,960 is a row sum over overlapping files; 180,394 unique points. Region labels, not point labels; dev set for all three flags. Base numbers are reportable as "agreement with hand-drawn regions"; intervals are 10 m block bootstrap. See §11 Q1–Q3 |
| Residual error is abstention, not error | de-duplicated: of 7,743 missed water points (of 136,502), 5,518 labelled uncertain, 2,225 land/canopy | `runs/Inn_audit_base/audit_labels.json` | **confirmed** (region labels; original 7,502 / 5,395 / 2,107 double-counted overlaps) |
| Pielach vs independent reference | point F1 0.931 (IoU 0.872, P 0.923 / R 0.940) on 25.4k coincident hard returns; 2D footprint IoU 0.597 (F1 0.748); water surface +0.32 m vs WSM, RMSE 0.35 m | `validation_results/validation_report.json` | **downgraded, still the best number.** F1 0.931 silently drops our 1,203 uncertain points (778 of them reference water). Counting abstention as a miss: **F1 0.911, P 0.923 / R 0.899, n = 26,645**. All matches fall in 28 ten-metre blocks; block-bootstrap F1 interval [0.842, 0.953] (0.931 version: [0.871, 0.969]). Only 11.4% of our points have a coincident reference return. Reference is OWP+SVB algorithm output, not human truth (§11 Q4–Q5) |
| Independent survey points | land control 4/4; water transects 35/42; overall 84.8% (39/46) | `validation_results/`, `survey_validation_report.json` | **confirmed it ran; too small to carry a claim.** Water 35/42, Wilson 95% [0.69, 0.92], but the 42 are 5 transects in a 39 × 22 m patch and all 7 misses are the shallow ends of 3 of them. Land 4/4, Wilson [0.51, 1.00]. 0 of 8 underwater targets fall in the cloud. Use as qualitative support for the margin bias only |
| Systematic bias | under-call water at shallow/gravel channel margins, consistent across all three validation angles | ibid. | **confirmed and diagnosed** (§11 Q5): it is the 0.5 probability guard, not the z margin or the footprint |
| Canopy stage | CV macro-F1 0.998, AUC 0.99996; mid-band canopy fraction 0.472; water sanity 0.17% false canopy | `models/canopy_v1/metrics.json` | soft (bootstrap labels); not re-checked |
| Multi-echo intuition is backwards | dry gravel/meadow produce more peaks/gaps/clusters than water; water is compact and early | `context/knowledge_base.md`, diagnostics | soft (classes defined by pipeline labels); not re-checked |

**Inn label provenance (re-measured 2026-11-11, supersedes the 2026-10-27 note).**
- *Regions ("192k" / "208k")*: seven CloudCompare cuts,
  `data/Inn_DeepLearning_labels/*.asc`. Category: **points inside regions a
  human drew** — not point-by-point labels, not propagated, not pipeline-derived.
  Evidence: every file's header line 1 is
  `//X Y Z reflectance_dB wcn_proba wcn_xgb_proba canopy_proba reconstructed_label final_label`,
  i.e. they were cut from `runs/Inn_DeepLearning/pointclouds/labeled_pointcloud_final.csv`;
  the embedded `final_label` disagrees with the region's class in thousands of
  rows (water2: 1,849 rows carry 0 and 3,356 carry 2), so membership was not
  produced by filtering on the pipeline label.
  **The counts were wrong.** 208,960 and 192,301 are sums of file lengths
  (`scripts/validate_labels.py:56-59` stacks the files; nothing de-duplicates).
  The files overlap: water1∩water2 10,849, water1∩water3 2,390,
  deep_water∩water2 15,077, deep_water∩water1 1,952. Unique points: water
  136,502 + land 43,892 = **180,394 (78.2% of the cloud, not 83%)**. deep_water
  adds only 1,332 points not already in water1/2. No point is in both a water
  and a land region.
  Dev-set status: file times are 16:59–17:01 on 2026-09-21 and the three flag
  commits follow at 17:13, 17:48, 18:10; `deep_water.asc` is stamped 17:55,
  fifteen minutes before the `surface_prior` commit whose message scores it.
  Author, 2026-11-11: the regions were drawn "looking at the reflectance and
  the geometry". So they are independent of the model's output, but **not of
  the method's cues** — the surface bootstrap splits flat sheets by
  reflectance, which is the same judgement the author made by eye. Agreement
  with them partly measures "the pipeline sees what a human sees in
  reflectance + geometry", which is weaker than agreement with ground truth.
- *The 65 picked points* (`water.txt` 29, `land.txt` 21, `proabably-land.txt`
  15; file times 2026-09-06 17:43, before the Inn site models existed at 21:07
  and before `scripts/sample_for_labeling.py` wrote its strata at 22:31 — none
  of the 65 is in any stratum file, so they were not sampled; author confirms "clicked by eye").
  **46 of 65 lie inside the regions** (25 water, 13 land, 8 probably-land), so
  the two label sets are not independent. The 19 outside are correct in every
  configuration including base, so they discriminate nothing.
  Bias: all 36 land points have `wcn_proba` < 0.004 — no land point near the
  decision boundary was picked. 34 of 36 are brighter than 0 dB; of the
  regions' land points only 41% are. All 15 probably-land points sit in one
  31 × 16 m patch (9 ten-metre blocks). 65 points occupy 54 ten-metre blocks.
  "Blind" holds only in time, and only for the base configuration:
  `resolve_uncertain` and `majority_filter` were scored on this set.

**The reference validation is still the most valuable number in the repo** —
the only one measured against a third-party classified point cloud (Mandlburger
et al., Pielach, TU Wien, 24–25 Oct 2024, DOI 10.48436/taz19-r6618, CC-BY 4.0).
Quote it as **F1 0.911 [0.84, 0.95] with abstentions counted**, not 0.931, and
say that it covers 28 ten-metre blocks of one reach.

---

## 4. Figures available now

| Figure | File | Use |
|---|---|---|
| Water-extent overlay vs reference | `validation_results/water_extent_overlay.png` | headline validation figure |
| Final classified point cloud, top-down | `models/v10/topdown_scatter.png` | method result, Pielach |
| Cross-section | `models/v10/crosssection.png` | shows surface model + bed |
| Local water surface grid | `models/v10/surface_grid.png` | method, Phase 2 |
| River boundary, probability heatmap | `models/final/boundary_heatmap.png` | vector product |
| Canopy top-down + cross-sections | `models/canopy_v1/{topdown,cross_sections}.png` | canopy stage |
| Inn classes / probability / boundary | `runs/Inn_DeepLearning/plots/*.png` | cross-site section |
| Feature importance | `models/wcn_v9/xgb_importance.png`, `models/canopy_v1/feature_importance.png` | supporting |
| WCN training curves | `models/wcn_v9/phase{1,2}*.png` | appendix |
| Reflectance / waveform diagnostics | `models/labeling/diagnostic_*.png` | motivates feature choice |

**Figures that do not exist yet and should (the paper's argument needs them):**
1. **The transfer matrix** — 2×2 trained×tested AUC, with within-site diagonal. This *is* thesis A; currently only a markdown table.
2. **Per-feature water−land shift, both sites side by side** — the §4 table as a dumbbell/slope plot. Shows "dies / reverses" at a glance.
3. **Reflectance: absolute dB vs percentile rank**, two sites overlaid. Shows why only rank transfers.
4. **Waveform archetypes** — mean normalised waveform for water vs land, Pielach vs Inn, 2×2 panel. Makes the export-format difference physical rather than statistical.
5. **The elevation-leakage figure** — Inn cross-section with the `z < 380.44 m` plane drawn across it, bank rising then falling below water level. One picture, whole argument.
6. **Bootstrap comparison** — reflectance histogram over all points (unimodal, Otsu lands inside the water mode) vs over surface layers only (bimodal, Otsu lands at −0.46 dB).

All six are derivable from data already on disk. Use the `dataviz` skill; existing plots are diagnostic-grade, not publication-grade.

---

## 5. Method, as it would appear in a paper

Stage names below match the code so the paper and repo stay reconcilable.

1. **Waveform parsing** — `lidarwater.io`: per-echo records, 200-bin dense amplitude grid, per-sample max normalisation. Grid anchoring is site-dependent (`grid_origin`).
2. **Site derivation** (`site.py`) — measure water level (densest 0.1 m z-bin), a reflectance percentile, waveform energy distribution, and fraction of cloud above a per-cell ground surface. These rebase every absolute constant. Architecture and dimensionless ratios untouched.
3. **Bootstrap** — either elevation bands (Pielach) or the elevation-free surface method (`BootstrapConfig(method="surface")`): rasterise, take each cell's top surface, keep cells whose top agrees with neighbours, split by median reflectance *of the surface layer only*.
4. **Waveform classification network** (WCN) — three-phase: masked waveform autoencoder (self-supervised), supervised fine-tuning with confidence-weighted focal loss, two rounds of pseudo-label refinement. 11 dimensionless scalars + the raw grid. `reflectance_dB` appended per-site only.
5. **Geometric refinement** — two-tier concave-hull footprint from high-confidence anchors, 2 m local water-surface grid with Gaussian smoothing and RANSAC fallback, inside-footprint + z ≤ surface + 0.3 m → water. Optional waterbed reconstruction under canopy.
6. **Canopy stage** — XGBoost on echo rank within pulse, height above a *water-aware* ground reference, points-overhead counts. Runs after the water model because water depth otherwise masquerades as vegetation height.
7. **Optional cleanup** — `resolve_uncertain`, `majority_filter`, `surface_prior`. Each is off by default and each is a measured accuracy step (§3).
8. **Vector product** — probability-field contour extraction at three thresholds (0.35 / 0.50 / 0.65), canopy excluded as evidence, gaps bridged by nearest-neighbour fill.

Spatial cross-validation throughout: 5 folds split on y-coordinate percentiles, never random, to prevent spatial leakage.

---

## 6. Related work — anchors to establish, NOT to cite from memory

Do not write a single citation without checking it. Run the `research` skill
against primary sources for each of these:

- **Mandlburger and co-workers**, topo-bathymetric UAV LiDAR, RIEGL VQ-840-GL, Pielach. The reference dataset's own authors. Non-negotiable citation, and the closest prior art. **[VERIFY]** exact publications, years, and what they claim about water-surface detection.
- **RIEGL VQ-840-GL** sensor specification, and the **OWP/SVB** (Online Waveform Processing / Surface-Volume-Bottom) processing chain. **[VERIFY]** whether SVB's export behaviour is documented publicly — the paper's central mechanism (the gap/cluster features encode the export format) depends on being able to describe it precisely and fairly.
- **Full-waveform LiDAR decomposition** — Gaussian decomposition and deconvolution literature. **[VERIFY]**
- **Water-surface and bathymetric-bottom detection** from green-wavelength LiDAR; refraction correction. **[VERIFY]**
- **Label-free / self-supervised point-cloud classification**, masked autoencoders for 1-D signals. **[VERIFY]**
- **Domain shift and negative transfer** in remote sensing; site-relative normalisation. **[VERIFY]**
- **ASPRS LAS classification** classes used by the reference (40 bathy bottom, 41 water surface, 43 submerged, 9 water). **[VERIFY]** against the LAS spec version in use.

---

## 7. Honest framing of the Inn site

The Inn survey was not a planned second study area; it was a stress test that
broke the pipeline in five distinct ways, each of which was then diagnosed and
fixed. That history is the paper's value — but write it as a systematic
generalisation study, not as a repair log. Each failure mode maps to a
mechanism:

| Failure | Mechanism | Generalisable lesson |
|---|---|---|
| Grid anchored at `times[0]` captured 7.8% of energy | SVB stores samples around echoes; full-range-gate stores everything | Waveform features must be anchored to a *detected* return, not a record start |
| Energy gate passed 0.78% of points | Fixed 30-bin window sized to a 60-bin pulse; Inn's pulses span 153 bins | Window-based features need pulse-length normalisation, not threshold rebasing |
| Gap/cluster features went flat | 80% of a full record is above-zero digitiser noise | Noise gating is a precondition, not a refinement |
| Reflectance excluded by design | dB scale is a scanner/survey property | Dimensionless-by-design can discard the dominant signal; per-site inclusion is the fix |
| 97.68% of output = an elevation rule | Bootstrap took elevation and nothing else | The teaching signal leaks even when the feature is withheld |

The fifth is the root cause of the other four mattering: a confidently wrong
model is worse than a noisy one.

---

## 8. What reviewer 2 will attack — status after the 2026-11-11 audit

1. **n = 2 surveys. — OPEN, and now fixed as a constraint.** Only Pielach and
   Inn exist; the course datasets A1–A6 have been deleted and are not
   candidates. This is a **two-survey study**. Every transfer claim must read
   "between these two exports"; `context/cross_site_transfer.md` §6 already
   concedes this. A third site remains the highest-value missing asset, and it
   would have to be newly acquired.
2. **Circular labels. — OPEN.** Within-site AUCs (0.989 / 0.972) are measured
   against pipeline-derived labels. In addition, **no script for any
   cross-site or within-site AUC in `context/cross_site_transfer.md` is
   committed** (grep for the experiment finds only prose), so none of §1–§4
   of that file can currently be regenerated. Re-run from a committed script
   before any of it is cited.
3. **Inn label sets. — RESOLVED as a question, and the answer is bad.** The
   "208k hand labels" are 180,394 unique points inside seven hand-drawn
   regions, cut from the pipeline's own output cloud, with heavy overlap
   between files. The 65 points are hand-picked, 46 of them inside those same
   regions, and their land sample is unrepresentatively bright. Neither set
   supports the AUC 0.958 vs 0.745 comparison. Author confirms the regions were drawn from
   reflectance and geometry, and the 65 were clicked by eye.
4. **Optional flags tuned on the test site. — RESOLVED by reporting base
   configuration.** Base, current code: 99.4% [99.1, 99.7] on regions, 96.6% [91.4, 100] on
   the 65 (stale models: 97.1% / 93.1%; see §11 Q3). No reach of Inn exists that the flags were not tuned on: the regions
   cover 78.2% of the cloud and were the development set, and the 19 picked
   points outside them are all correct in base already. Flag-on numbers may
   appear only as a labelled development-set ablation.
5. **"Unused" independent truth. — RESOLVED; this document was wrong.** The
   three survey CSVs were used: `src/evaluation/validate_survey_points.py`
   (commit e7c2da2, 2026-06-15) wrote
   `validation_results/reports/survey_validation_report.json` and
   `tables/survey_matches.csv` — 39/46, land 4/4, water transects 35/42, and
   0 of 8 underwater targets inside the cloud. It does not hold up as a
   headline: 5 transects in one 39 × 22 m patch, Wilson interval [0.69, 0.92]
   on water and [0.51, 1.00] on land. (Side note: the report's `by_file` block
   keys on the last word of the filename, so "control_points" is overwritten
   by "transect_points" — the 4/4 only survives in the overall totals and the
   CSV.)
6. **The margin under-call. — DIAGNOSED, not fixed.** See §11 Q5. One
   threshold explains it (the 0.5 probability guard); one threshold does not
   cleanly fix it.
7. **Reproducibility. — OPEN, and worse than listed.** Besides the data/DOI
   points: the cross-site experiments have no code (item 2); the z-band Inn run
   behind "97.68%" is not on disk; the reference set is not on disk either
   (re-download needed before the Pielach validation can be regenerated).
8. **New: the reference validation drops abstentions.** `point_level()` in
   `src/evaluation/validate_against_reference.py:179-204` scores only points we
   labelled 0/1/3/4, so the 1,203 uncertain points vanish from F1 0.931. A
   reviewer who reads the code will find it. Report 0.911.
9. **New: spatial autocorrelation.** Every accuracy in the repo was quoted
   without an interval. Point-bootstrap intervals on 180k contiguous points
   are meaninglessly narrow (±0.07 points); block-bootstrap ones are ten times
   wider. Report block intervals throughout.

---

## 9. Pre-draft checklist

- [x] Resolve item §8.3 (what are the 208k labels?). Done 2026-11-11: region
      labels, 180,394 unique, drawn from reflectance + geometry.
- [ ] Pick the thesis (§2). Blocks everything. Note thesis A's and C's key
      numbers were downgraded by the audit.
- [x] Decide the site claim: **two-site, stated honestly.** A1–A6 are deleted;
      there is no third survey (§8.1).
- [x] Run the unused survey CSVs (§8.5). Already run in June; documented and
      judged too small to carry a claim.
- [x] Re-run base-configuration Inn numbers without the three optional flags
      (§8.4). 99.4% [99.1, 99.7] regions / 96.6% [91.4, 100] picked points with
      a fresh fit; 97.1% / 93.1% with the stale September models.
- [x] Diagnose the margin under-call (§8.6). Diagnosed; fix not attempted.
- [ ] **New:** commit a script that regenerates every number in
      `context/cross_site_transfer.md`, and re-run it with block-bootstrap
      intervals. Blocks thesis A.
- [ ] **New:** build an Inn test set that is not the dev set: stratified random
      points (`scripts/sample_for_labeling.py` strata plus a uniform random
      stratum), labelled blind to pipeline output, including margins and dark
      land. Blocks any Inn accuracy claim stronger than "agreement with
      hand-drawn regions".
- [ ] **New:** re-download the reference subset and regenerate the Pielach
      validation with abstentions counted (needs author's go-ahead, ~5 GB).
- [ ] Verify every citation via `research` (§6).
- [ ] Build figures 1–6 (§4).
- [ ] Choose venue — **[VERIFY]** scope and format of candidates: *ISPRS Journal of Photogrammetry and Remote Sensing*, *Remote Sensing of Environment*, *IEEE TGRS*, MDPI *Remote Sensing*, or an ISPRS Archives conference paper for a faster, lower-stakes first outing.

---

## 10. Raw material already written

Pile these into the fragments file rather than rewriting them:
`context/cross_site_transfer.md` (114 lines, already paper-shaped),
`CLAUDE.md` failure-mode section, `context/canopy_plan.md` (design + iteration
log), `context/knowledge_base.md` (974 lines of domain notes),
`validation_results/README.md` and `reports/`, `docs/usage.md` + `docs/api.md`.

---

## 11. Audit findings (2026-11-11)

Scripts: `src/diagnostics/audit_inn_labels.py`,
`src/diagnostics/audit_pielach_validation.py`. Outputs:
`runs/Inn_audit_*/audit_labels.json`,
`runs/Pielach/audit/pielach_validation_audit.json`. All intervals are 95%
percentile bootstrap over 10 m plan-view blocks, 2,000 draws, unless marked
Wilson. Branch `paper-audit`.

**Q1 — What are the Inn "208k hand labels"? Verdict: region labels, miscounted.**
Seven regions a human drew in CloudCompare from reflectance and geometry, cut
from the pipeline's output cloud. 208,960 and 192,301 are sums over
overlapping files; the unique count is **180,394** (136,502 water, 43,892
land; 78.2% of the cloud). They were the development set for all three
optional flags. *Withdrawn:* "99.9% vs 208k hand labels", "192k hand labels",
"83% of the cloud". *Relabelled:* every number on them is "agreement with
hand-drawn regions (development set)". Detail in the provenance note under §3.

**Q2 — The 65 "blind" points. Verdict: not blind in space, biased to easy.**
Clicked by eye. 46 of 65 lie inside the regions. No land point has
`wcn_proba` above 0.004; 34 of 36 land points are brighter than 0 dB against
41% of region land. Measured consequence: a bare reflectance cut at +0.25 dB
scores 97.2% [92.9, 100] on the 65 and **70.5% [64.3, 78.4] on the regions**
(land recall 41%), because the 65 contain almost no dark vegetated land.
*Downgraded:* every reflectance-based figure on the 65 — "+0.25 dB gets 97.2%",
"reflectance rank transfers at AUC 0.958", "0.958 → 0.745", "AUC 0.902".
*Confirmed:* surface bootstrap 97.2% on the 65; it also holds on the regions
(96.2% [94.8, 97.6]), which the reflectance cut does not — so the bootstrap's
"isolate the surface layer" step is doing real work. That contrast is new and
is stronger evidence for the bootstrap than the original 65-point figure.

**Q3 — Base-configuration Inn. Verdict: 97.1% is a floor from stale models;
current code gives 99.4%; fit-to-fit spread is the dominant uncertainty.**

| models | flags | regions (n = 180,394, 199 blocks) | picked 65 (54 blocks) |
|---|---|---|---|
| `runs/Inn_DeepLearning/models` (2026-09-06) | off | 97.1% [96.2, 97.9] — water 94.3%, land 99.8% | 93.1% [86.0, 98.6] — water 25/29, land 36/36 |
| same | all three on | 99.92% [99.87, 99.95] | 100% |
| fresh `--fit`, current code, seed 42 (`runs/Inn_audit_refit`) | off | **99.4% [99.1, 99.7]** — water 99.0%, land 99.9% | **96.6% [91.4, 100]** — water 27/29, land 36/36 |
| same | all three on | 99.89% [99.84, 99.94] | 100% |

The 2026-09-06 models are the ones every published Inn figure was computed
with, and current code does not reproduce them: their recorded
`phase2_best_val_f1` is 0.954, the fresh fit's is 0.991, and the two base
outputs agree on only 95.9% of points. The fit is seeded (`WcnConfig` seed 42),
so this is code drift between 6 and 21 September, not seed noise
(a second same-seed fit, `runs/Inn_audit_refit2`, gives 99.39% [99.1, 99.6] and 96.6% — within 0.02 points of the first, though only 99.1% of labels are identical, so GPU training is not bitwise deterministic). Consequences:
- The flag ladder (97.4 → 98.7 → 99.3 → 99.9) was measured on models that the
  committed code no longer produces. With current models the three flags are
  worth **0.5 points** on the regions (99.4 → 99.9), not 2.8. Most of what the
  flags "fixed" was a weaker model.
- Abstract number: **base configuration, current code: 99.4% [99.1, 99.7]
  agreement with hand-drawn regions; 96.6% [91.4, 100] on 65 hand-picked
  points.** Both are development-era label sets; say so.
- No reach of Inn exists that the flags were not tuned on, so no flags-on
  number is reportable as a test result.
- A single z rule (`z < 380.77`) scores 96.0% [92.6, 98.3] on the regions.
  The regions are channel and bank interiors and barely contain the
  land-below-water-level case, so they cannot show the pipeline beating
  elevation; the stale-model base run did not beat it significantly at all.

**Q4 — The "unused" survey CSVs. Verdict: used in June; §8.5 and the memory
note were wrong; result too small to headline.** 39/46 overall; water transects
35/42 (Wilson [0.69, 0.92]) on 5 transects in a 39 × 22 m patch, all 7 misses
at the shallow end of 3 of them (bed heights 259.58–259.97 m); land 4/4
(Wilson [0.51, 1.00]); 0/8 underwater targets inside the cloud. Effective n is
five transects. *Downgraded* from "hard, small n" to qualitative support.

**Also found — the Pielach headline F1.** 0.931 excludes our 1,203 uncertain
points (`validate_against_reference.py:179-204`). With abstention counted as
a miss: **F1 0.911, precision 0.923, recall 0.899, n = 26,645, block interval
[0.842, 0.953]**. The matches cover only 28 ten-metre blocks. *Downgraded*
0.931 → 0.911; it remains the only third-party-referenced number.

**Q5 — The margin under-call. Verdict: one threshold explains it; it is the
0.5 probability guard.** Done from `tables/per_point_matches.csv` joined to
`pointclouds/labeled_pointcloud_{final,v10}.csv`; no download was needed.
Of 1,803 reference-water points we did not call water: 1,016 land, 778
uncertain, 9 canopy.

| | n | median z | median refl. | median `deep_proba` | median `z_above_surface` | outside footprint | above surface + 0.3 m | `deep_proba` ≤ 0.5 |
|---|---|---|---|---|---|---|---|---|
| true positives | 16,034 | 259.90 | −25.0 dB | 0.998 | −0.36 | 6.6% | 0.0% | 5.1% |
| missed, called land | 1,016 | 260.16 | −14.5 dB | 0.306 | −0.11 | 14.6% | 0.8% | **100%** |
| missed, called uncertain | 778 | 260.02 | −20.8 dB | 0.511 | −0.25 | 9.6% | 0.0% | 46.0% |

Geometry is not the cause: 85% of the land-called misses are inside the
footprint and 99% are under the surface margin. They are shallow, bright
(land-like reflectance) returns that the waveform ensemble scores below 0.5,
and the merge rule lets the ensemble veto geometry
(`src/lidarwater/_stages/surface.py:50-54`, `222-250`). `deep_proba ≤ 0.65`
covers 95.4% of all misses and 10.4% of true positives.
It is not a free fix. Among matched points geometry already accepts, lowering
the guard to 0.4 / 0.3 / 0.2 recovers 854 / 1,140 / 1,243 reference-water
points and converts 224 / 489 / 681 reference-land points to water; F1 moves
0.911 → 0.931 / 0.932 / 0.931. Choosing the guard on these matches would be
fitting to the only independent reference. And the reference at the margin is
itself OWP+SVB output: bright returns at the waterline may be wet gravel that
the reference over-calls. Not adjudicable without field truth.

**Ledger summary.**
- *Withdrawn:* 99.9% vs 208k; 192k/208k/83% counts; the flag ladder as a
  result; "+0.25 dB reflectance gets 97.2%" as a site property.
- *Downgraded:* Pielach F1 0.931 → 0.911 [0.84, 0.95]; elevation leakage
  97.68% (run not on disk; z rule matches the pipeline on regions);
  reflectance-rank transfer 0.958 and 0.958 → 0.745 (biased 65); AUC 0.902;
  survey points; every `cross_site_transfer.md` number (no committed script).
- *Confirmed:* surface bootstrap 97.2% on the 65 and 96.2% on regions;
  residual base-config error is mostly abstention; margin under-call is real
  and systematic; site-profile measurements (water level, energy gate 0.294).
- *Not checked:* failure modes 1–3 magnitudes, canopy metrics, per-feature
  shift tables.

