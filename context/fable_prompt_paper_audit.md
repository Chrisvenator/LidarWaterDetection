# Task: audit the evidence before a paper gets written

## Why

I want to publish this repo's work: label-free water/land classification
from full-waveform bathymetric LiDAR, and why waveform features don't
transfer between surveys. `context/paper_context.md` is my planning
document. It holds an evidence ledger (section 3), a list of reviewer
attacks (section 8), and a pre-draft checklist (section 9).

Several headline numbers may not hold up. Your job this session is to
establish which numbers are defensible, **before anyone writes prose**. Don't
draft any paper text. A number withdrawn now costs far less than a
retraction later, so never soften a finding to make the paper look better.

## Read first

- `context/paper_context.md`, in full.
- `CLAUDE.md`, especially "Running on a Different Survey".
- `context/cross_site_transfer.md`.
- `validation_results/README.md` and `validation_results/reports/`.

## Scope facts

- Only two surveys exist: Pielach and Inn. The course datasets A1–A6
  mentioned in `paper_context.md` §8.1 and §9 have been **deleted**. Don't
  look for them. Record that the paper is a two-survey study and update
  §8.1 and §9 to say so.
- The Mandlburger reference set (`data/mandlburger_pielach_2024/`) is **not
  on disk** right now. `scripts/download_pielach_data.sh --no-images`
  fetches the roughly 5 GB validation subset. **Ask me before downloading
  it.** Do all the work that doesn't need it first.

## Questions to settle, in this order

1. **What are the Inn "208k hand labels"?** They back the 99.9% figure.
   `data/Inn_DeepLearning_labels/` holds `.asc` region files (water1–3,
   deep_water, land_and_canopy_1–2, dry_riverbed) and three `.txt` files
   that together hold 65 rows. Trace the 208k figure, and the 192k figure
   quoted elsewhere, to the code and artefacts that produced them. Classify
   them as one of:
   - points labelled one by one by a human;
   - points inside regions a human drew, which is region labelling: still
     human, but spatially clustered and easy to get right;
   - labels propagated from a smaller set;
   - labels derived from the pipeline.

   Quote the file and line each conclusion rests on. Say what that means for
   every number in the ledger that uses them.
2. **The 65 blind points.** How were they chosen? Is the selection spatially
   biased, or biased towards easy points? Do any of them lie inside the
   `.asc` regions, so that the two label sets overlap?
3. **Base-configuration Inn numbers.** Re-run Inn through
   `scripts/run_dataset.py` with `resolve_uncertain`, `majority_filter` and
   `surface_prior` all **off**. Those three flags were added after looking at
   Inn's errors, so numbers reported with them on are fitted to the test set.
   Report balanced accuracy against each honest label set from 1 and 2, with
   n and a bootstrap confidence interval. This is the number for the abstract.
   Then, if a reach of Inn exists that none of the flags were tuned on, report
   it with the flags on as well.
4. **The "unused" survey CSVs.** `paper_context.md` §8.5 says the
   Mandlburger control, target and transect CSVs were never used. My other
   notes say `src/evaluation/validate_survey_points.py` already used them:
   84.8% overall, 39 of 46 matched, with 4 of 4 land control points and 35
   of 42 water transect points correct. Work out which is true from
   `validation_results/`, and fix whichever document is wrong. If the check
   ran, judge whether it holds up statistically: 46 points, and the
   underwater targets fall outside our cloud.
5. **The margin under-call.** All three validation angles show we label
   shallow and gravel channel margins as land. If the reference data is
   available by this point, diagnose why from the false-negative points
   (`z`, reflectance, `deep_proba`, `z_above_surface`). Say whether one
   threshold explains it. Diagnose only; don't fix it yet.

## Rules

- Measure, don't recall. Every number you write down needs a source: a
  command you ran, or a file and line.
- Don't change pipeline code or default configs. Diagnostic scripts go in
  `src/diagnostics/`, and their outputs go in `runs/` or the scratchpad.
- Outputs go to `runs/<dataset>/`, never to the repo-level `models/`,
  `pointclouds/` or `data_processed/`.
- Work on a new branch off `library-functionality`. Commit the diagnostics
  and the doc updates; don't push.

## When you're done

- Update `context/paper_context.md`:
  - section 3's strength column, for every number you checked;
  - the section 9 checkboxes;
  - section 8, marking each item resolved or still open, with the reason.
- Add a short section 11, "Audit findings (date)": each question with its
  verdict, the evidence, and which ledger numbers were withdrawn, downgraded
  or confirmed.
- Finish with a brief report to me:
  - what the defensible headline numbers are now;
  - what you withdrew;
  - whether the paper still has a results section that stands up.

  If the honest answer is "not yet", say that.

Stop and ask me only before the download, or if a label set turns out to be
something I need to explain, such as how I drew the `.asc` regions.
