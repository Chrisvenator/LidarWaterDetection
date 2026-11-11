# Paper prompts

Three prompts, run in order, each in a **fresh session** (clean context matters
more than continuity here). Prompt 0 is the one that decides whether the paper
is possible; do not skip it.

---

## Prompt 0 — resolve the blockers (run first, expect ~1 session)

```
Read context/paper_context.md in full. It is a paper-planning document I
wrote from this repo's measured results.

Do not write any prose for the paper yet. Your job this session is only
section 9's first four checklist items — the things that decide whether the
results section is defensible.

1. Section 8.3 is the blocker. Find out what the "208k hand labels" on the
   Inn survey actually are. Trace the number back to the code or artefact
   that produced it. Report one of: genuinely hand-labelled, propagated from
   a smaller hand-labelled set, or pipeline-derived. Quote the file and line
   you concluded from. If it is not genuine hand labelling, say so plainly
   and tell me which numbers in the ledger have to be withdrawn or
   relabelled as soft.

2. Do the same for the 65 hand-labelled Inn points: where are they stored,
   how were they chosen, is the selection spatially biased?

3. Re-run the Inn pipeline in base configuration — resolve_uncertain,
   majority_filter and surface_prior all OFF — and report balanced accuracy
   against whatever the honest label set turns out to be. This is the number
   that goes in the abstract, not the 99.9%.

4. Check whether data/A1 through A6 (the course datasets referenced in my
   memory) contain anything usable as a third survey: same sensor family,
   full-waveform available, a water body present. A third site would change
   what this paper can claim, so this is worth real effort.

5. Use the three unused survey CSVs in the Mandlburger reference set
   (control points, underwater targets, transects) as an independent check
   on the Pielach result. Memory says they have full UTM33N coordinates and
   ellipsoidal heights and have never been touched. Note the unicode
   direction marks in the height column.

Work through these in order. Update section 3's strength column and section 9's
checkboxes in context/paper_context.md as you establish each one. Do not
soften a finding to make the paper look better — a withdrawn number now is
cheaper than a retraction later.
```

---

## Prompt 1 — establish the argument

```
Read context/paper_context.md and the blocker findings from the previous
session.

I want to write a paper from this work. Section 2 lists three candidate
theses; the document recommends A framed by C. Before I commit, grill me on
that choice. Specifically attack:

- whether a two-survey study can support a non-transferability claim at all,
  and what the weakest honest phrasing is that the evidence does support
- whether "the features encode the export format" is actually established or
  just consistent with the data — what measurement would distinguish the two
- whether the elevation-leakage finding is novel or a known pitfall under
  another name
- what the single strongest competing explanation is for the 0.478 transfer
  AUC, and whether I have ruled it out

Then make me state the contribution in one sentence that a reviewer could
not weaken.
```

Then, once the thesis holds: `/grilling` for a full adversarial pass, or go
straight to fragments.

---

## Prompt 2 — write it

```
/writing-fragments context/paper_fragments.md
```

Open with:

```
We are drafting a paper on label-free water/land classification from
full-waveform bathymetric LiDAR, and on why waveform features do not transfer
between surveys. Read context/paper_context.md first — it holds the measured
evidence ledger, the figure inventory, and the known weaknesses.

The thesis is: <the one sentence from Prompt 1>.

Audience: remote-sensing researchers who know LiDAR but not this sensor's
processing chain. Target: <venue>.

Interview me. Pull fragments out of context/cross_site_transfer.md,
CLAUDE.md's failure-mode section and context/knowledge_base.md as raw
material, but make me say in my own words why each one matters. Do not let me
assert anything the ledger marks soft without flagging it.
```

Then shape it:

```
/writing-shape context/paper_fragments.md
```

Prerequisites to declare when it asks what the reader walks in knowing:
green-wavelength LiDAR penetrates water; full-waveform means the digitised
return is kept, not just extracted echoes; supervised classification and AUC.
Everything else — SVB, OWP, surface/volume/bottom, the export-format
distinction, site-rank encoding — must be grounded by an earlier block.

---

## Figures

After the argument is fixed, not before — the argument decides which of the
six figures in section 4 are load-bearing:

```
Read context/paper_context.md section 4. Build publication-grade versions of
figures 1-6 from the data already on disk. Use the dataviz skill. One
consistent visual system across all six: same palette, same type scale,
legible at single-column width in greyscale. Existing plots in models/ are
diagnostic-grade — do not reuse them, rebuild.
```
