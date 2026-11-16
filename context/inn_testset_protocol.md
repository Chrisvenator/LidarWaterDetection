# Inn no-look test: protocol

Written and committed 2026-11-16, **before any fit was launched and before any
pipeline output was compared with the new labels**. Source of the task:
`context/fable_prompt_inn_testset.md`. This is a one-shot evaluation: after
this commit neither the pipeline, a config, nor this protocol changes. If
something turns out to be broken it is reported, not fixed and re-scored.

Implementation: `src/diagnostics/evaluate_inn_testset.py` (committed next,
before any fit). Outputs: `runs/inn_testset/`.

## 1. Labels

`data/Inn_DeepLearning_Labels_nolook/`, drawn by the author in CloudCompare
without looking at pipeline output. Measured before scoring, using only the
raw cloud's coordinates (`read_xyz` + nearest neighbour in
`src/diagnostics/audit_inn_labels.py`): every row matches a cloud point at
distance 0; no point carries both a water and a land label.

| file | label | unique points | share below 380.76 m |
|---|---|---|---|
| Wasser1 / Wasser2 / Wasser3 | water | 77,098 / 45,823 / 13,320 | 0.988 / 0.999 / 1.000 |
| safe-water | water | 19 | 1.000 |
| geometry-water | water | 36 | 1.000 |
| Land1 / Land2 | land | 20,628 / 22,814 | 0.000 / 0.098 |
| Landzunge | land | 2,121 | 0.408 |
| Land_below_waterline | land | 2,249 | 1.000 |
| safe-land | land | 15 (16 rows) | 0.000 |

Unique over all files above: **175,650** (129,980 water, 45,670 land).
164,824 of them are also in the old development labels (old regions or the 65
picked points); 10,826 are in neither (7,563 water, 3,263 land).

Facts recorded now because they bear on interpretation:

- The large region files carry nine columns (x y z reflectance, `wcn_proba`,
  `wcn_xgb_proba`, `canopy_proba`, `reconstructed_label`, `final_label`): they
  were cut from a pipeline output cloud. Only the first three columns are read.
  "No-look" rests on the author's statement, as it did for the old regions.
- `Land_below_waterline` (stratum C) lies **entirely inside the old
  development regions** (0 of 2,249 points new). Its labels are new; its area
  is not.
- `Landzunge`: 343 of 2,121 points are outside the old labels.
  `geometry-water`: all 36 are outside.
- **`Wasser4.asc`** (2,120 water points, 13 ten-metre blocks, 1,081 outside
  the old labels, disjoint from every file above) was saved at 10:49, four
  minutes after the task description, and is not in its file table. It is
  therefore **not** part of strata A-E. It is scored as a supplementary
  stratum (section 4) so that either reading is on the record.

## 2. Fits and configurations

- `derive_site_config(cloud)` on the Inn cloud, bootstrap `method="sheet"`
  (asserted, not assumed), from-scratch `WaterPipeline.fit()`, WCN seed 42 and
  43, torch and numpy seeded before the fit as in
  `ablate_models.fit_from_scratch`. Fresh feature cache and models under
  `runs/inn_testset/`; nothing is reused from `runs/ablation/`.
- **Primary**: all three flags off (`resolve_uncertain`, `majority_filter`,
  `surface_prior`). This is the fit's own output.
- **Secondary**: all three flags on, by `classify()` with the same fit's
  models. A flags-off `classify()` is also run and its agreement with the
  fit's output is reported as a consistency check.
- Seed 42 is the headline seed (the config default); seed 43 is a
  replication. Both are reported everywhere.

## 3. Methods scored, all on the same points

1. pipeline, primary;
2. pipeline, secondary;
3. bootstrap labels alone (`sheet`);
4. elevation rule: water iff `z < level + 0.05 m`, `level` =
   `SiteProfile.water_level_z` from `derive_site_config`;
5. reflectance rule: water iff `reflectance_dB < +0.25`.

## 4. Strata

- **A** all no-look points (175,650);
- **B** new only: A minus every point in an old region or old `.txt` file
  (10,826);
- **C** `Land_below_waterline`: land recall;
- **D** `Landzunge`: land recall;
- **E** `geometry-water`: water recall as a count out of 36 with the number
  of 10 m blocks. No interval, not called a rate.
- Supplementary, not in the task: **F** `Wasser4` alone (water recall), and
  **A+** = A plus `Wasser4`. Neither enters the headline or the criterion.

## 5. Metric

- Balanced accuracy where a stratum has both classes (A, B, A+); the recall
  of its one class otherwise.
- **Strict scoring (primary)**: a point is correct only if the method assigns
  it its true class. Pipeline class 2 (uncertain) is wrong whatever the truth;
  classes 1 and 3 are water; 0 and 4 are land. A point the bootstrap leaves
  unlabelled is wrong whatever the truth.
- **Water-versus-rest scoring (secondary, for comparison with
  `paper_context.md` sections 11-14 only)**: water = classes 1, 3; everything
  else, including uncertain and unlabelled, is "not water", so abstention on a
  land point counts as correct. The elevation and reflectance rules never
  abstain, so the two scorings coincide for them.
- Intervals: 95% percentile, 2,000 resamples of 10 m plan-view blocks
  (`audit_inn_labels.block_ids`, `measure_transfer.block_multiplicities`,
  seed 0), each cloud point counted once. Blocks are cut within each stratum.
- Paired differences pipeline minus elevation rule, on the same block draws,
  for both pipeline configurations and both seeds.
- C has 10 blocks and D has 8. A percentile bootstrap over so few blocks is
  rough; the block counts are reported next to every interval.

## 6. The claim and what refutes it

Claim: *where elevation is wrong by construction, the pipeline beats the
elevation rule.* It is tested on C and D, primary configuration, strict
scoring.

For a stratum S in {C, D} and a seed, the pipeline **beats elevation on S**
iff the paired difference (pipeline land recall minus elevation-rule land
recall) is positive and its 95% block-bootstrap interval excludes 0.

- **The claim holds** iff the pipeline beats elevation on C and on D under
  both seeds, **and** the guard below is met.
- **The claim is refuted** if, under either seed, the pipeline does not beat
  elevation on C (difference <= 0 or interval covering 0). C is the clean
  case: every point is below the water level, so the elevation rule is
  expected to score near 0 there, and failing to beat it would mean the
  pipeline reproduces elevation.
- **Mixed** (C beaten under both seeds, D not): reported as "holds for land
  well below the level, not shown for land at the level". Not rounded up to
  "holds".
- **Guard.** Beating a rule that scores ~0 on land is trivial for anything
  that calls land generously. The claim only counts if the pipeline does not
  pay for it elsewhere: on stratum A the paired difference in balanced
  accuracy (pipeline minus elevation) must not have an interval entirely
  below 0, under both seeds. E is reported next to it as counts: the
  elevation rule is expected to get all 36 by construction, and whatever the
  pipeline gets is stated as is.

The secondary configuration, the bootstrap, the reflectance rule, the
water-versus-rest scoring and strata F / A+ are reported but decide nothing.

## 7. Headline number

Stratum A, primary configuration, seed 42, strict balanced accuracy with its
block interval; stratum B next to it. Limitations to state wherever it is
quoted: 94% of A lies in the area of the old development regions (new labels,
old area); the 39,320 points nobody could label (62% within 0.5 m of the
water level, per the task description) are excluded, so the hardest margin
points are missing and the number is optimistic there.
