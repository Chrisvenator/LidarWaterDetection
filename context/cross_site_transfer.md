# Cross-site transfer: measurements and limits

Measured September 2026 on the two surveys in `data/` — Pielach (RIEGL
VQ-840-GL, SVB-clustered waveform export, narrow canopied valley) and Inn
(same sensor family, full-range-gate export, wide braided river). All numbers
are XGBoost on the stated features, spatially cross-validated within a site
and whole-site held out across sites.

**Conclusion up front: a single waveform model spanning both surveys is not
achievable with this feature set. Per-site fitting is the correct
architecture.**

## 1. Waveform-only classification works — within one site

| | 11 scalar features | raw 200-bin grid |
|---|---|---|
| Pielach | 0.989 | 0.983 |
| Inn | 0.972 | 0.986 |

No reflectance, no geometry, no elevation. The raw grid matches or beats the
hand-engineered scalars, so the WCN's transformer branch has headroom the 11
features do not capture.

Caveat: both sites' labels are pipeline output, which partly descends from
reflectance and geometry, so these are optimistic. The unbiased figure is Inn
waveform-only scored against 65 hand-labelled points: **AUC 0.902**
(leave-one-out).

## 2. Waveform-only transfer between sites is chance

Site-rank encoding means each feature is replaced by its percentile rank
within its own cloud, which removes bodily scale differences.

| trained -> tested | raw | site-rank |
|---|---|---|
| Pielach -> Inn (65 hand points) | 0.478 | 0.492 |
| Inn -> Pielach (whole site) | 0.647 | 0.685 |

Dropping the four features whose water-vs-land shift dies or reverses between
the exports (`max_amp_norm_by_energy`, `gap_ratio`, `n_gaps`, `n_clusters` —
see §4) does **not** help: 0.419/0.467 and 0.689/0.611 respectively. The
remaining seven do not transfer either.

## 3. The only signal that transfers is reflectance, as a site-relative rank

| everything site-rank | Pielach -> Inn | Inn -> Pielach |
|---|---|---|
| waveform 7 alone | 0.467 | 0.611 |
| waveform 7 + reflectance | 0.745 | 0.761 |
| **reflectance rank alone** | **0.958** (86.9% balanced) | 0.626 |

Trained on Pielach and tested on Inn's hand labels, reflectance percentile
rank alone reaches AUC 0.958 with no Inn data at all. Adding the waveform
features *degrades* it to 0.745 — the Pielach-trained model leans on features
that are strong there and misleading on Inn.

In absolute dB the two sites are irreconcilable (Inn's water is brighter than
Pielach's land); as a rank within each cloud they nearly coincide:

| | water | land |
|---|---|---|
| Pielach, dB | -23.60 | -13.57 |
| Inn, dB | -4.77 | +0.01 |
| Pielach, rank | 0.410 | 0.907 |
| Inn, rank | 0.484 | 0.879 |

## 4. Why waveform shape does not transfer

Water-minus-land shift per feature, in site-normalised units:

| feature | Pielach | Inn | |
|---|---|---|---|
| `max_amp_norm_by_energy` | -0.65 | 0.07 | dies |
| `gap_ratio` | -0.62 | 0.13 | reverses |
| `n_gaps` | -0.56 | 0.05 | dies |
| `n_clusters` | -0.56 | 0.05 | dies |
| `energy_concentration` | 0.33 | 0.29 | agrees |
| `peak_amp_ratio` | -0.24 | -0.22 | agrees |
| `depth_proxy_m` | -0.20 | -0.20 | agrees |

Pielach's four strongest features are the gap/cluster family, which describes
the **SVB export format** — only samples around each echo are stored, so gaps
mark the boundaries between returns. Inn's full-record digitisation has no
equivalent structure, so those features go flat or reverse. What survives
across both is consistent but weak (~0.2-0.3 sd).

`FeatureConfig.noise_gate` narrows this gap (it moved `n_gaps` from +4.86 to
-0.81 sigma of the Pielach training distribution) but does not close it:
`active_bins_ratio` remains ~5 sigma apart, because the two exports are
different recordings of the same physics.

## 5. What this means for the architecture

- **Fit per site.** `derive_site_config` + `fit()` is the supported path, and
  measured 93.1% balanced accuracy on Inn against hand labels.
- **Do not ship a cross-site waveform model.** It would be, in effect, a
  reflectance-rank threshold with a network attached that actively hurts it.
- **Reflectance rank is not a better bootstrap than what exists.** The
  elevation-free surface bootstrap (`BootstrapConfig(method="surface")`)
  scores 97.2% on Inn using no second site at all, against reflectance-rank-
  from-Pielach's 86.9%.

## 6. Limits of this evidence

Two surveys cannot demonstrate or refute universality in general — they can
only measure transfer between these two. A model trained on both could
memorise "Pielach-like vs Inn-like" rather than learn invariance, and nothing
in a two-site set distinguishes those. A third held-out survey would be
needed to claim anything stronger.

The Pielach -> Inn figures rest on 65 hand-labelled points; the Inn -> Pielach
direction trains on pipeline labels that partly descend from reflectance, so
it is somewhat circular. The asymmetry between 0.958 and 0.626 is likely as
much about label quality and test-set size as about direction.
