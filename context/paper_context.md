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
*A rule-based waveform bootstrap plus geometric refinement reaches F1 0.931
against an independent classified reference, with zero training labels.*
Cleanest story, easiest to write, but weakest novelty — label-free bootstrapping
is not new, and the validation is one site.

**C. The elevation-leakage result.**
*Elevation-band bootstrapping silently teaches the model an elevation rule.
97.68% of the Inn output is reproduced by the single rule `z < 380.44 m`,
despite z never being a model feature. An elevation-free surface bootstrap
fixes it: 61.0% -> 97.2% balanced accuracy.*
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
| Waveform-only classification works within a site | AUC 0.989 Pielach / 0.972 Inn (11 scalars); 0.983/0.986 (raw 200-bin grid) | `context/cross_site_transfer.md` §1 | soft |
| …same, unbiased | AUC 0.902 on 65 hand-labelled Inn points (leave-one-out) | ibid. | hard, small n — **effective n far below 65, see note under table** |
| Waveform-only transfer is chance | AUC 0.478 Pielach→Inn, 0.647 Inn→Pielach; site-rank encoding does not help (0.492 / 0.685) | ibid. §2 | hard / soft |
| Dropping the four non-transferring features does not rescue transfer | 0.419–0.467 and 0.611–0.689 | ibid. §2 | soft |
| Only reflectance *rank* transfers | AUC 0.958 (86.9% balanced) Pielach→Inn hand labels, reflectance rank alone | ibid. §3 | hard, small n |
| Adding waveform features *degrades* that | 0.958 → 0.745 | ibid. §3 | hard, small n |
| Absolute dB is irreconcilable, rank nearly coincides | Pielach water/land −23.60/−13.57 dB; Inn −4.77/+0.01 dB; ranks 0.410/0.907 vs 0.484/0.879 | ibid. §3 | hard |
| The four dead features describe the export format | `max_amp_norm_by_energy` −0.65→+0.07, `gap_ratio` −0.62→+0.13, `n_gaps` −0.56→+0.05, `n_clusters` −0.56→+0.05 (water−land shift, site-normalised) | ibid. §4 | hard |
| What survives is consistent but weak | `energy_concentration` 0.33/0.29, `peak_amp_ratio` −0.24/−0.22, `depth_proxy_m` −0.20/−0.20 | ibid. §4 | hard |
| Elevation leakage | 97.68% of Inn final output reproduced by `z < 380.44 m` | `CLAUDE.md`, failure mode 5 | hard |
| Elevation-free surface bootstrap fixes it | Otsu over all points −7.10 dB / 61.0% balanced; over surface layers −0.46 dB / 97.2% (100% water, 94.4% land); stable across 2–4 m cells | ibid. | hard, 65 pts |
| Flatness alone is insufficient | cell flatness: 100% water recall, ~40% land; per-point planarity reverses (water 0.61, land 0.73) | ibid. | hard |
| Pulse-length dependence of a fixed-window feature | `energy_concentration > 0.85` gate passes 48.6% of Pielach, 0.78% of Inn (153-bin vs 60-bin pulses); percentile rebase 0.294 → 13,481 candidates | ibid., failure mode 2 | hard |
| Grid anchoring matters for full-record exports | energy captured 7.8% → 87% with `grid_origin="first_return"` | ibid., failure mode 1 | hard |
| Noise pedestal corrupts gap/cluster features | `n_gaps` +4.86σ → −0.81σ, `energy_ratio_late` +3.02 → +0.66 with `noise_gate` | ibid., failure mode 3 | hard |
| Final Inn accuracy, per-site fit | 93.1% balanced (base) → 97.4% (+`resolve_uncertain` → 98.7%) → 99.3% (+`majority_filter`) → 99.9% vs 208k hand labels, 100% on blind 65 (+`surface_prior`) | `CLAUDE.md`; `runs/Inn_DeepLearning/metrics.json` | **WITHDRAWN as stated (2026-10-27).** 192k/208k = 7 region polygons cut in CloudCompare from the pipeline's own output cloud (`data/Inn_DeepLearning_labels/*.asc`, header carries `wcn_proba … final_label`). Region-level, interiors only, used as development set for all three flags. Report only as "agreement with hand-segmented regions (dev set)". Blind 65 = only held-out set. |
| Residual error is abstention, not error | of 7,502 missed water points, 5,395 labelled uncertain, 2,107 land | `CLAUDE.md` | hard |
| Pielach vs independent reference | point F1 0.931 (IoU 0.872, P 0.923 / R 0.940) on 25.4k coincident hard returns; 2D footprint IoU 0.597 (F1 0.748); water surface +0.32 m vs WSM, RMSE 0.35 m | `validation_results/validation_report.json` | **hard, independent** |
| Independent survey points | land control 4/4; water transects 35/42; overall 84.8% (39/46) | `validation_results/`, `survey_validation_report.json` | hard, small n |
| Systematic bias | under-call water at shallow/gravel channel margins, consistent across all three validation angles | ibid. | hard |
| Canopy stage | CV macro-F1 0.998, AUC 0.99996; mid-band canopy fraction 0.472; water sanity 0.17% false canopy | `models/canopy_v1/metrics.json` | soft (bootstrap labels) |
| Multi-echo intuition is backwards | dry gravel/meadow produce more peaks/gaps/clusters than water; water is compact and early | `context/knowledge_base.md`, diagnostics | hard |

**Inn label provenance (established 2026-10-27).**
- *192k / 208k*: 7 polygon regions (`data/Inn_DeepLearning_labels/*.asc`, 21 Sep)
  drawn by the author in CloudCompare by visual interpretation, exported from the
  pipeline output cloud. water1–3 = 148,409, land/canopy/dry-riverbed = 43,892
  (sum 192,301); + deep_water 16,659 = 208,960. deep_water likely overlaps water2
  (bbox inside) — double count **[VERIFY]**. Interiors only, margins excluded,
  ~7 autocorrelated regions. Every optional flag was developed against these;
  `surface_prior` specifically against deep_water. Colouring during selection
  **[VERIFY with author]**. Status: dev set, not test set.
- *Blind 65* (`water.txt` 29, `land.txt` 21, `proabably-land.txt` 15, 6 Sep,
  before the regions and before all three flags): blind in *time*, not in
  *space*. Water: 22/29 at the surface (z 380.67–380.73), 7 below it, spread
  along the channel. Land: 17/21 on the south bank inside land_and_canopy_1's
  extent, 4 inside land_and_canopy_2's; mostly z > 381 (easy), 2 below water
  level. Probably-land: all 15 in one ~30×15 m patch at water-surface height
  (x 323–354, y 19–33), which is the dry_riverbed region. So the hard land cases
  are one gravel bar, labelled "probably". Effective independent n is roughly
  a dozen sites, not 65. `resolve_uncertain` and `majority_filter` were scored
  on this set after it was used to motivate them.

**The reference validation (F1 0.931) is the single most valuable number in the
repo** — it is the only one measured against a third-party classified point
cloud (Mandlburger et al., Pielach, TU Wien, 24–25 Oct 2024,
DOI 10.48436/taz19-r6618, CC-BY 4.0). Lead with it in the abstract.

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

## 8. What reviewer 2 will attack — fix before submitting

1. **n = 2 surveys.** You cannot claim non-transferability in general from two sites; `context/cross_site_transfer.md` §6 already concedes this. Either soften to "between these two exports" throughout, or obtain a third survey. **A third site is the highest-value missing asset.** Note the repo memory references six course datasets (`data/A1–A6`) — **[VERIFY]** whether any is usable as a third site; if one is, it changes the paper's class.
2. **Circular labels.** Within-site AUCs (0.989 / 0.972) are measured against pipeline-derived labels that partly descend from reflectance and geometry. Report them as such, every time, or drop them. The hand-labelled and reference-validated numbers must carry the argument.
3. **n = 65 hand labels** carries the Inn conclusions. This is thin for AUC 0.958 vs 0.745 comparisons. **[VERIFY]** whether the 208k "hand labels" cited for the 99.9% figure are genuinely hand-labelled or propagated — if propagated, that number cannot appear in a paper as stated. **Resolve this before drafting; it is the most likely fatal flaw.**
4. **Optional flags tuned on the test site.** `resolve_uncertain`, `majority_filter`, `surface_prior` were each added after observing Inn errors and are each reported *as* Inn improvements. That is test-set fitting. Either report base-configuration numbers as the headline, or hold out a reach of the Inn survey the flags were never inspected on.
5. **Unused independent truth.** The reference dataset ships three CSVs (control points, underwater targets, transects) from total-station/RTK survey; memory records them as still unused. That is the cleanest independent check available and it is sitting on disk. Use it.
6. **The margin under-call** is a known systematic bias, consistent across three validation angles, and undiagnosed. Diagnose it (z, reflectance, `deep_proba`, `z_above_surface` at the FN points) — a one-threshold fix would strengthen the method section; leaving it undiagnosed invites the reviewer to find it.
7. **Reproducibility.** Raw data is gitignored; the reference set is 51 GB. State exactly which DOIs, which commit, which configs. `scripts/download_pielach_data.sh` exists — make the whole thing one command.

---

## 9. Pre-draft checklist

- [ ] Resolve item §8.3 (what are the 208k labels?). Blocks the results section.
- [ ] Pick the thesis (§2). Blocks everything.
- [ ] Decide the site claim: two-site honest, or find a third (§8.1).
- [ ] Run the unused survey CSVs (§8.5).
- [ ] Re-run base-configuration Inn numbers without the three optional flags (§8.4).
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
