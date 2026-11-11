# Cross-site transfer: measurements and limits

Re-measured 2026-11-11 on honest labels. Every number below is printed by one
command and stored in `runs/transfer/transfer_results.json`:

```bash
python src/diagnostics/measure_transfer.py     # ~9 min; first run adds ~9 min of feature extraction
```

The two surveys are Pielach (RIEGL VQ-840-GL, SVB-clustered waveform export,
narrow canopied valley) and Inn (full-range-gate export, wide braided river).

**Conclusion up front.** A waveform classifier fitted on one of these two
surveys does not work on the other, so per-site fitting stays the right
architecture. Two earlier claims do not survive: reflectance rank does *not*
transfer, and the export format is *not* shown to be the cause.

## 0. Method

- **Labels.** Inn: the hand-drawn regions, 180,394 points (136,502 water) in
  199 ten-metre blocks. Pielach: the Mandlburger reference matches, 26,645
  points (17,837 water) in 28 blocks; the join back to our cloud is exact
  (max distance < 1e-9 m, every match a distinct cloud row, checked on every
  run). Secondary columns, never mixed in: Inn's 65 hand-picked points
  ("65, biased") and the pipeline's own decided labels over the whole cloud
  ("pipeline": 228,454 Pielach points in 40 blocks, 223,810 Inn points in 203).
- **Model.** XGBoost, fixed hyperparameters, seed 0 (`XGB_PARAMS` in the
  script). Two consecutive full runs gave identical figures.
- **Within a site:** 5-fold cross-validation holding out whole 10 m blocks.
  **Across sites:** fit on all honest-labelled points of one site, score on
  the other's.
- **Intervals:** 95% percentile bootstrap over 10 m plan-view blocks, 2,000
  draws, the same resampling as `audit_inn_labels.py`. Differences between
  two scores on the same points use the same draws (paired).
- **Scores:** AUC, plus balanced accuracy at the model's own 0.5 threshold. A
  transferred model can rank well and still have a useless threshold, so both
  are given where it matters.
- **Why 10 m blocks.** Scalar waveform features have no spatial neighbourhood;
  leakage comes from points sharing a pulse (identical waveform rows) and from
  local land-cover continuity. Random-point folds are optimistic; 10 m removes
  that and larger blocks change nothing beyond the interval:

  | hold-out | Pielach | Inn |
  |---|---|---|
  | random points | 0.944 [0.891, 0.975] | 0.996 [0.995, 0.997] |
  | 10 m blocks | 0.890 [0.806, 0.946] | 0.993 [0.990, 0.995] |
  | 20 m blocks | 0.893 [0.808, 0.948] | 0.989 [0.983, 0.993] |
  | 40 m blocks | n/a (4 blocks) | 0.988 [0.983, 0.992] |

## 1. Waveform-only classification within one site

AUC, no reflectance, no geometry, no elevation.

| | 11 scalar features | raw 200-bin grid | pipeline labels (secondary), 11 / grid |
|---|---|---|---|
| Pielach, n = 26,645, 28 blocks | 0.890 [0.806, 0.946] | 0.888 [0.800, 0.949] | 0.980 [0.968, 0.989] / 0.975 [0.961, 0.986] |
| Inn, n = 180,394, 199 blocks | 0.993 [0.990, 0.995] | 0.999 [0.999, 1.000] | 0.988 [0.984, 0.991] / 0.997 [0.996, 0.998] |

Balanced accuracy: Pielach 0.810 [0.713, 0.883] / 0.808; Inn 0.964
[0.955, 0.973] / 0.989 [0.986, 0.991].

- It works on both sites, but on Pielach's independent reference it is
  **0.89, not 0.98–0.99**. Part of that drop is circular pipeline labels and
  part is a harder task: the reference matches are mostly water against bare
  ground near the channel (7,669 of 8,808 land points are ground class), with
  almost none of the canopy that makes whole-cloud land easy.
- The Inn figure is high, but the regions are channel and bank interiors drawn
  from reflectance and geometry and leave out the margins. Read it as "easy
  points are easy", not as a test result.
- The raw grid beats the scalars only on Inn (0.999 vs 0.993); on Pielach they
  are indistinguishable.
- Inn's 65 picked points (biased; model fitted on regions outside their
  blocks): AUC 1.000, balanced accuracy 0.972 [0.929, 1.0]. They no longer
  discriminate anything.

## 2. Waveform-only transfer

AUC on the test site's honest labels. "Gap" is the within-site score on the
same points minus the transferred one (paired interval).

| trained → tested | features | raw | site-rank | gap to within-site (raw) |
|---|---|---|---|---|
| Pielach → Inn | 11 scalars | 0.567 [0.551, 0.580] | 0.430 [0.413, 0.448] | 0.426 [0.413, 0.443] |
| | 7 (four export features dropped) | 0.454 [0.426, 0.480] | 0.442 [0.418, 0.461] | 0.539 [0.512, 0.567] |
| | raw grid | 0.647 [0.593, 0.690] | — | 0.352 [0.309, 0.406] |
| Inn → Pielach | 11 scalars | 0.702 [0.602, 0.773] | 0.725 [0.623, 0.805] | 0.188 [0.116, 0.266] |
| | 7 | 0.711 [0.606, 0.779] | 0.627 [0.552, 0.694] | 0.179 [0.118, 0.260] |
| | raw grid | 0.638 [0.590, 0.670] | — | 0.250 [0.179, 0.304] |

- **Pielach → Inn is at chance** (0.43–0.65 against 0.993–0.999 within Inn).
  Several scores are *below* 0.5 with intervals that exclude it: the
  Pielach-trained ordering is slightly inverted on Inn.
- **Inn → Pielach is weak, not chance**: 0.63–0.73, every interval above 0.5,
  and 0.17–0.26 below Pielach's own 0.89 with intervals that exclude zero.
  Balanced accuracy at the transferred threshold is 0.52–0.57, so the ranking
  signal is not usable as a classifier.
- Site-rank encoding does not help in either direction. Dropping the four
  export-format features does not help either.
- Secondary columns. On the 65 (biased), Pielach → Inn scalars give
  0.479 [0.345, 0.622] raw and 0.520 site-rank, as before. On pipeline labels
  Pielach → Inn is 0.561 / 0.475 and Inn → Pielach 0.494 / 0.441: the
  above-chance Inn → Pielach signal exists only against the reference labels.

## 3. Reflectance rank does not transfer

Everything site-rank encoded.

| | Pielach → Inn, regions | Pielach → Inn, 65 (biased) | Inn → Pielach |
|---|---|---|---|
| reflectance rank alone | 0.601 [0.486, 0.740] | 0.970 [0.906, 1.0] | 0.766 [0.633, 0.860] |
| waveform 7 + reflectance | 0.633 [0.610, 0.661] | 0.836 [0.726, 0.918] | 0.789 [0.663, 0.877] |
| waveform 11 + reflectance | 0.494 [0.441, 0.559] | 0.819 [0.698, 0.915] | 0.810 [0.683, 0.895] |
| waveform 7 alone | 0.442 [0.418, 0.461] | 0.459 [0.298, 0.617] | 0.627 [0.552, 0.694] |
| reflectance in raw dB | 0.460 [0.443, 0.478] | 0.620 [0.535, 0.702] | 0.588 [0.532, 0.645] |

- The old headline reproduces on the 65 (0.970) and **collapses on the
  regions: 0.601, interval including 0.5**. The reason is on Inn's side, not
  in the transfer: reflectance does not order water against land on the
  regions at all (its own AUC there is 0.457 [0.303, 0.577]; class means
  −6.75 dB water, −4.69 dB land), because much of the region land is dark
  vegetation. The 65 contain almost none of it.
- **"Adding waveform features degrades it" is a property of the 65.** There,
  the drop is 0.134 [0.054, 0.239] (7 features) and 0.151 [0.063, 0.258]
  (11). On the regions, adding the 7 changes nothing (−0.032 [−0.131, 0.084])
  and adding all 11 costs 0.107 [0.042, 0.187]. Inn → Pielach, adding the 11
  *helps* by 0.044 [0.009, 0.080].
- Inn → Pielach reflectance rank is above chance (0.766) because reflectance
  does separate the classes on Pielach's reference (own AUC 0.851, i.e.
  1 − 0.149).
- Class means on honest labels:

  | | water | land |
  |---|---|---|
  | Pielach, dB | −23.18 | −15.35 |
  | Inn, dB | −6.75 | −4.69 |
  | Pielach, rank | 0.404 | 0.806 |
  | Inn, rank | 0.479 | 0.532 |

  The ranks do not "nearly coincide": Inn's land sits at 0.53, not 0.88.

## 4. Per-feature shift on honest labels

Water-minus-land mean difference in whole-cloud standard deviations, native
features. Full table with each feature's own AUC is in the JSON
(`q5_feature_shifts`).

| feature | Pielach | Inn | |
|---|---|---|---|
| `max_amp_norm_by_energy` | −0.80 [−1.21, −0.33] | +0.05 [0.01, 0.08] | dies |
| `gap_ratio` | −0.60 [−0.78, −0.43] | +0.17 [0.10, 0.23] | reverses |
| `n_gaps` | −0.30 [−0.60, −0.11] | +0.07 [0.03, 0.11] | reverses, weakly |
| `n_clusters` | −0.30 [−0.60, −0.11] | +0.07 [0.03, 0.11] | same as `n_gaps` |
| `energy_ratio_late` | +0.99 [0.63, 1.26] | +0.20 [0.13, 0.26] | same sign, 5× weaker |
| `peak_amp_ratio` | −0.32 [−0.42, −0.20] | −0.22 [−0.31, −0.14] | agrees |
| `energy_concentration` | +0.09 [−0.17, 0.41] | +0.48 [0.20, 0.71] | Pielach not significant |
| `depth_proxy_m` | −0.12 [−0.33, 0.06] | −0.20 [−0.26, −0.12] | Pielach not significant |
| `active_bins_ratio` | −0.01 [−0.33, 0.24] | −0.29 [−0.39, −0.18] | Pielach not significant |
| `energy_center_norm` | +0.21 [−0.14, 0.45] | −0.10 [−0.22, 0.03] | neither significant |
| `n_peaks` | +0.06 [−0.26, 0.28] | −0.13 [−0.21, −0.04] | Pielach not significant |
| `reflectance_dB` | −1.57 [−1.99, −1.04] | −0.35 [−0.90, 0.10] | Inn not significant |

- The qualitative claim holds: the gap/cluster family and
  `max_amp_norm_by_energy` separate the classes on Pielach and are flat or
  reversed on Inn. 5 of 11 waveform features share a sign; Spearman
  correlation of the two shift vectors is 0.16.
- Only one feature has the same sign with both intervals clear of zero and a
  comparable size: `peak_amp_ratio`. `energy_ratio_late` agrees in sign.
- Pielach's strongest waveform feature on the reference is
  `energy_ratio_late` (+0.99), which the old table did not list.
- The table depends on who defines the classes. Against pipeline labels,
  Pielach's `max_amp_norm_by_energy` shift is **+0.58**, the opposite sign,
  because whole-cloud land is mostly canopy.

## 5. Format or physics?

**The test.** If the export format is why transfer fails, then putting both
surveys through the *same* export should (a) bring the gap/cluster shifts
back into line and (b) restore transfer. The script re-exports both raw
waveform sets through one function and then reads both with Pielach's
unchanged default extraction. Each rung adds one harmonisation; all
parameters are measured without labels (`runs/transfer/cache/harmonisation.json`).

1. **Echo-window export** — keep only sample runs that rise above Inn's
   pre-trigger noise ceiling (23 ADC above the record floor) and contain a
   peak of at least twice that; subtract the pedestal. This is the suggested
   "make Inn look like SVB", applied to both sites so the comparison is
   symmetric.
2. **Time axis matched** — the strongest echo is 69 samples wide at half
   maximum on Inn and 3 on Pielach, so Inn is averaged into 23-sample bins.
3. **Range gate matched** — after rung 2 an Inn record spans 7 bins (median),
   so Pielach is cut to 7 samples after its first stored sample.

This is a better test than re-exporting Inn alone because it also asks what
the *common* export does to Pielach, and because rungs 2 and 3 separate the
export from the two other recording differences that the data show.

Transfer AUC on honest labels after each rung:

| rung | P → I, 11 raw | P → I, grid | I → P, 11 raw | I → P, grid | within Pielach, 11 / grid | within Inn, 11 / grid |
|---|---|---|---|---|---|---|
| 0 native | 0.567 | 0.647 | 0.702 | 0.638 | 0.890 / 0.888 | 0.993 / 0.999 |
| 1 echo window | 0.534 | 0.782 | 0.781 | 0.545 | 0.879 / 0.881 | 0.995 / 1.000 |
| 2 + time axis | 0.559 | 0.727 | 0.704 | 0.797 | 0.879 / 0.881 | 0.942 / 0.962 |
| 3 + range gate | 0.477 | 0.623 | 0.701 | 0.775 | 0.870 / 0.876 | 0.942 / 0.962 |

Paired change against rung 0 (intervals in `q4_ladder_recovery_contrasts`):

- **Scalars do not recover.** P → I: −0.033 [−0.057, −0.004], −0.008
  [−0.041, 0.029], −0.090. I → P: +0.079 [0.041, 0.108] at rung 1, then
  +0.003 and 0.000.
- **The raw grid recovers partly, and only rung 2 helps both directions:**
  P → I +0.080 [0.047, 0.109] to 0.727; I → P +0.159 [0.091, 0.214] to 0.797
  [0.701, 0.868]. That is the time-axis match, i.e. the pulse/sampling
  difference, not the export. Rung 1 alone moves the grid in opposite
  directions (+0.136 P → I, −0.093 I → P). Even at its best, P → I sits at
  0.78 against 1.000 within Inn, with balanced accuracy 0.507.
- **The gap/cluster shifts do not line up again; they disappear.** Under the
  common export Inn's `n_gaps` has mean 0.001 — its records hold one broad
  echo — and Pielach's `gap_ratio` shift falls from −0.60 to −0.13
  [−0.42, 0.13], `n_gaps` from −0.30 to −0.06 [−0.40, 0.17]. Pielach loses
  almost nothing by it (within-site 0.890 → 0.879). So Pielach's gap/cluster
  signal lives in stored segments weaker than Inn's noise, which is a
  property of what the export keeps — but it is not what transfer was
  missing.
- Sign agreement of the eleven shifts across rungs: 5/11, 5/11, 4/11, 2/11.

**Result: the format explanation is not supported.** Harmonising the export
neither realigns the features nor restores transfer. What the data do show is
a much larger recording difference that is not an export choice: Inn's
strongest echo is 23× wider in samples (median over all records, labels not
used), and its records end 7 Pielach-equivalent samples after the first
return. Matching the time axis
recovers part of the raw-grid transfer; nothing recovers the scalars. Whether
the remainder is instrument configuration or the rivers themselves cannot be
separated with two surveys.

Limits of this test: the re-export thresholds are tied to Inn's noise (one
setting, no sweep); rung 2 costs Inn information (within-site 0.995 → 0.942);
rung 3 was chosen after seeing that Inn's records are cut short.

## 6. What this means for the architecture

- **Fit per site.** Unchanged, and now measured on honest labels.
- **Do not ship a cross-site waveform model.** Best transferred score is 0.80
  AUC with a threshold near chance.
- **There is no transferable reflectance shortcut either.** Reflectance rank
  from Pielach scores 0.60 on Inn's regions. The surface bootstrap (96.2%
  [94.8, 97.6] on the same regions, `paper_context.md` §11) is the supported
  route.

## 7. Limits of this evidence

- Two surveys. Every statement is "between these two recordings"; export,
  instrument settings and site are confounded and n = 2 cannot separate them.
- Pielach's honest labels cover 28 ten-metre blocks of one reach and are
  OWP+SVB algorithm output, not field truth. All Pielach intervals are wide.
- Inn's regions were drawn from reflectance and geometry, cover easy interiors
  and were the development set for the pipeline's optional flags. Nothing
  here was tuned on them, but they are not a blind test set.
- One model family, one hyperparameter setting, one seed. No WCN was trained.

## 8. Superseded figures

| old claim | old | now (honest labels) | why it changed |
|---|---|---|---|
| within-site, Pielach, 11 / grid | 0.989 / 0.983 | 0.890 / 0.888 | scored against pipeline labels (reproduced: 0.980 / 0.975); reference set is harder and independent |
| within-site, Inn, 11 / grid | 0.972 / 0.986 | 0.993 / 0.999 | pipeline labels then; regions now |
| Inn "unbiased" on the 65 | 0.902 | dropped | 65 biased points; now AUC 1.000, uninformative |
| Pielach → Inn waveform | 0.478 / 0.492 (65 points) | 0.567 / 0.430 (regions) | test set; conclusion unchanged |
| Inn → Pielach waveform | 0.647 / 0.685 | 0.702 / 0.725 | trained and scored on pipeline labels then; "chance" becomes "weak" |
| dropping four features | 0.419–0.467, 0.611–0.689 | 0.454 / 0.442, 0.711 / 0.627 | same; conclusion unchanged |
| reflectance rank transfers | 0.958 | 0.601 [0.486, 0.740] | measured on the 65; **withdrawn** |
| waveform degrades reflectance | 0.958 → 0.745 | 0.601 → 0.633 (7), 0.494 (11) | effect of the 65; holds only for all 11 on regions |
| ranks nearly coincide | 0.410/0.907 vs 0.484/0.879 | 0.404/0.806 vs 0.479/0.532 | pipeline-defined classes; **withdrawn** |
| shift table | −0.65/−0.62/−0.56/−0.56 → ≈0 | −0.80/−0.60/−0.30/−0.30 → +0.05/+0.17/+0.07/+0.07 | class definition; qualitative claim holds |
| "features encode the export format, hence no transfer" | asserted | not supported (§5) | first time tested |
| fit-per-site "93.1% on Inn" | 93.1% | see `paper_context.md` §11 Q3 | stale models |
