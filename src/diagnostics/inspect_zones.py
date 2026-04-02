"""
inspect_zones.py — Purely diagnostic. No training, no labels used.

Step 1: 10 example waveforms from each of three z-bands, three rows.
Step 2: Reflectance histograms for the three zones overlaid.
Step 3: Full feature separability — z=261-263 vs z=263-266 on ALL features.

Samples from ALL points in each z-band (ignores existing labels entirely).
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

SEED = 42
rng  = np.random.default_rng(SEED)

BANDS = {
    'riverbed\n(z 258–260m)':  (258.0, 260.0, '#7f3b08'),  # dark brown
    'water surface\n(z 261–263m)': (261.0, 263.0, '#1f77b4'),  # blue
    'dry ground\n(z 263–266m)':    (263.0, 266.0, '#d62728'),  # red
}

ALL_FEATURES = [
    'max_amp', 'mean_amp', 'std_amp', 'total_energy', 'n_samples',
    'time_span', 'n_gaps', 'max_gap', 'mean_gap', 'total_gap',
    'n_peaks', 'max_peak_spacing', 'mean_peak_spacing',
    'first_last_span', 'n_clusters', 'energy_ratio_late',
    'first_peak_amp', 'last_peak_amp', 'peak_amp_ratio', 'depth_proxy_m',
    'reflectance_dB',
    'planarity', 'roughness', 'linearity', 'sphericity',
    'height_range_local', 'height_std_local', 'z_relative',
    'energy_concentration', 'amplitude_weighted_center',
    'active_bins_ratio', 'max_amp_norm_by_energy',
    'height_percentile_local',
]


# ── helpers ───────────────────────────────────────────────────────────────────

def cohens_d(a, b):
    na, nb = len(a), len(b)
    s = np.sqrt(((na-1)*a.std(ddof=1)**2 + (nb-1)*b.std(ddof=1)**2) / (na+nb-2))
    return (a.mean() - b.mean()) / s if s > 0 else 0.0

def mw_auc(a, b):
    stat, _ = stats.mannwhitneyu(a, b, alternative='two-sided')
    return stat / (len(a) * len(b))

def sep_label(auc):
    d = abs(auc - 0.5)
    if d >= 0.40: return 'EXCELLENT'
    if d >= 0.30: return 'GOOD'
    if d >= 0.15: return 'MODERATE'
    if d >= 0.05: return 'WEAK'
    return 'NONE'


# ── load ──────────────────────────────────────────────────────────────────────

print("Loading features_v2.csv …")
feat_df = pd.read_csv('features_v2.csv')
z       = feat_df['z'].values
print(f"  {len(feat_df):,} rows  z={z.min():.2f}–{z.max():.2f} m")

print("Loading waveform_grids.npy …")
grids   = np.load('waveform_grids.npy', mmap_mode='r')

# ── pull indices for each band (no label filter) ──────────────────────────────

band_idx = {}
for label, (lo, hi, _) in BANDS.items():
    mask = (z >= lo) & (z < hi)
    idx  = np.where(mask)[0]
    band_idx[label] = idx
    print(f"  {label.replace(chr(10),' '):<30}  {len(idx):,} points")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — waveform grid plots, 3 rows × 10 columns
# ═══════════════════════════════════════════════════════════════════════════════

N_EX   = 10
GRID   = 200

fig, axes = plt.subplots(3, N_EX, figsize=(28, 9),
                          gridspec_kw={'hspace': 0.65, 'wspace': 0.25})

for row, (band_label, (lo, hi, color)) in enumerate(BANDS.items()):
    idx     = band_idx[band_label]
    chosen  = rng.choice(idx, size=min(N_EX, len(idx)), replace=False)
    axes[row, 0].set_ylabel(band_label, fontsize=9, labelpad=8)

    for col, pt in enumerate(chosen):
        ax = axes[row, col]
        wf = grids[pt].astype(float)
        ax.plot(wf, color=color, linewidth=0.9)
        ax.fill_between(range(GRID), wf, alpha=0.18, color=color)

        # title: key scalar values
        r  = feat_df.iloc[pt]
        zv = r['z']
        rf = r.get('reflectance_dB', float('nan'))
        np_ = r.get('n_peaks', '?')
        ec  = r.get('energy_concentration', float('nan'))
        ax.set_title(
            f"z={zv:.2f}\nrefl={rf:.1f}  pk={np_}\nec={ec:.2f}",
            fontsize=6.5, pad=2)
        ax.set_xlim(0, GRID)
        ax.tick_params(labelsize=5)
        if col > 0:
            ax.set_yticklabels([])

fig.suptitle(
    'Waveform Grids by Elevation Zone  —  10 random points per band, NO label filter',
    fontsize=12, y=1.01)

# legend
handles = [Patch(color=c, label=lbl.replace('\n', ' '))
           for lbl, (_, _, c) in BANDS.items()]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, -0.03))

wf_path = 'models/zone_waveforms.png'
plt.savefig(wf_path, dpi=160, bbox_inches='tight')
plt.close()
print(f"\nStep 1 waveform plot → {wf_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — reflectance histograms, three zones overlaid
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

refl_col = 'reflectance_dB'
ec_col   = 'energy_concentration'

for ax, col, xlabel in [
        (axes[0], refl_col,  'Reflectance (dB)'),
        (axes[1], ec_col,    'Energy concentration (fraction in first 30 bins)')]:

    if col not in feat_df.columns:
        ax.set_visible(False)
        continue

    all_vals = []
    for band_label, (lo, hi, color) in BANDS.items():
        idx  = band_idx[band_label]
        vals = feat_df.iloc[idx][col].dropna().values.astype(float)
        all_vals.append(vals)

    lo_p = np.percentile(np.concatenate(all_vals), 1)
    hi_p = np.percentile(np.concatenate(all_vals), 99)
    bins = np.linspace(lo_p, hi_p, 60)

    for band_label, (lo, hi, color), vals in zip(BANDS.keys(), BANDS.values(), all_vals):
        short = band_label.replace('\n', ' ')
        ax.hist(vals, bins=bins, alpha=0.50, color=color,
                density=True, label=f"{short}  (n={len(vals):,})")
        ax.axvline(np.median(vals), color=color, linewidth=1.5,
                   linestyle='--', alpha=0.85)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_title(col, fontsize=10)

fig.suptitle('Distribution comparison by elevation zone  (dashed = median)',
             fontsize=11)
plt.tight_layout()
refl_path = 'models/zone_distributions.png'
plt.savefig(refl_path, dpi=160, bbox_inches='tight')
plt.close()
print(f"Step 2 distribution plot → {refl_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — separability: water surface (261–263) vs dry ground (263–266)
#           on ALL features, no label filter
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*125}")
print("STEP 3 — Separability: water surface z=261–263  vs  dry ground z=263–266")
print("         All points in each z-band, ignoring auto-labels")
print(f"{'='*125}")

keys  = list(BANDS.keys())
water_idx_sep = band_idx[keys[1]]  # 261-263
dry_idx_sep   = band_idx[keys[2]]  # 263-266

n_w = min(2000, len(water_idx_sep))
n_d = min(2000, len(dry_idx_sep))
w_sel = rng.choice(water_idx_sep, size=n_w, replace=False)
d_sel = rng.choice(dry_idx_sep,   size=n_d, replace=False)

water_df = feat_df.iloc[w_sel]
dry_df   = feat_df.iloc[d_sel]

print(f"Sample: {n_w:,} water-surface  |  {n_d:,} dry-ground")
print(f"\n{'Feature':<30} {'W mean':>10} {'W std':>9} {'D mean':>10} {'D std':>9} "
      f"{'Cohen d':>9} {'MW-AUC':>8}  Separability")
print(f"{'-'*125}")

rows = []
for feat in ALL_FEATURES:
    if feat not in water_df.columns:
        continue
    w = water_df[feat].dropna().values.astype(float)
    d = dry_df  [feat].dropna().values.astype(float)
    if len(w) < 10 or len(d) < 10:
        continue
    cd  = cohens_d(w, d)
    auc = mw_auc(w, d)
    _, p = stats.mannwhitneyu(w, d, alternative='two-sided')
    rows.append({'feature': feat, 'water_mean': w.mean(), 'water_std': w.std(),
                 'dry_mean': d.mean(), 'dry_std': d.std(),
                 'cohens_d': cd, 'mw_auc': auc, 'p_value': p,
                 'separability': sep_label(auc)})

sep_df = pd.DataFrame(rows).sort_values('mw_auc', key=lambda x: abs(x-0.5), ascending=False)

for _, r in sep_df.iterrows():
    flag = ' ***' if r['separability'] in ('EXCELLENT','GOOD') else ''
    print(f"  {r['feature']:<28} {r['water_mean']:>10.3f} {r['water_std']:>9.3f} "
          f"{r['dry_mean']:>10.3f} {r['dry_std']:>9.3f} "
          f"{r['cohens_d']:>9.2f} {r['mw_auc']:>8.3f}  {r['separability']}{flag}")

print(f"\nNote: AUC < 0.5 = water LOWER on that feature. AUC > 0.5 = water HIGHER.")

sep_df.to_csv('models/zone_separability_water_vs_dry.csv', index=False)
print(f"\nFull table → models/zone_separability_water_vs_dry.csv")

print(f"\n{'='*60}")
print("SEPARABILITY SUMMARY (water surface vs dry ground, no label bias)")
print(f"{'='*60}")
for tier in ['EXCELLENT','GOOD','MODERATE','WEAK','NONE']:
    feats = sep_df[sep_df['separability']==tier]['feature'].tolist()
    if feats:
        print(f"  {tier:<12}: {', '.join(feats)}")


# ─── Step 3b: top-feature distributions as small multiples ───────────────────

top = sep_df[sep_df['separability'].isin(['EXCELLENT','GOOD'])]['feature'].tolist()
top = top[:12]

if top:
    ncols = 4
    nrows = int(np.ceil(len(top) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(top):
        ax = axes[i]
        w_v = water_df[feat].dropna().values.astype(float)
        d_v = dry_df  [feat].dropna().values.astype(float)
        all_v = np.concatenate([w_v, d_v])
        bins  = np.linspace(np.percentile(all_v, 1), np.percentile(all_v, 99), 40)
        ax.hist(w_v, bins=bins, alpha=0.55, color='#1f77b4', density=True, label='water 261–263m')
        ax.hist(d_v, bins=bins, alpha=0.55, color='#d62728', density=True, label='dry 263–266m')
        r = sep_df[sep_df['feature']==feat].iloc[0]
        ax.set_title(f"{feat}\nAUC={r['mw_auc']:.3f}  d={r['cohens_d']:.2f}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)

    for j in range(len(top), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        'Top discriminating features: water surface (blue) vs dry ground (red)\n'
        'z=261–263m vs z=263–266m  —  no label filter, raw z-band sampling',
        fontsize=10)
    plt.tight_layout()
    top_path = 'models/zone_top_features.png'
    plt.savefig(top_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Top-feature distribution plot → {top_path}")

print("\nDone.")
