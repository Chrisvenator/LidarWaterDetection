"""
diagnostic_water_vs_veg.py

Take 500 confirmed WATER and 500 confirmed LAND-vegetation points.
  WATER      : pred_A == 1  AND  auto_label == 1
  VEGETATION : pred_A == 0  AND  auto_label == 0  AND  z > 264 m

For each waveform feature report:
  mean ± std per class, Cohen's d, and Mann-Whitney U AUC (rank-biserial).
Plot 5 example waveforms per class on the dense grid.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

WAVEFORM_FEATURES = [
    'max_amp', 'mean_amp', 'std_amp', 'total_energy', 'n_samples',
    'time_span', 'n_gaps', 'max_gap', 'mean_gap', 'total_gap',
    'n_peaks', 'max_peak_spacing', 'mean_peak_spacing',
    'first_last_span', 'n_clusters', 'energy_ratio_late',
    'first_peak_amp', 'last_peak_amp', 'peak_amp_ratio', 'depth_proxy_m',
    'reflectance_dB',           # scalar — included for completeness
]

N_SAMPLE   = 500
GRID_LEN   = 200
SEED       = 42
rng        = np.random.default_rng(SEED)


# ── helpers ──────────────────────────────────────────────────────────────────

def cohens_d(a, b):
    """Pooled-SD Cohen's d (signed: positive means a > b)."""
    na, nb  = len(a), len(b)
    pooled  = np.sqrt(((na - 1) * a.std(ddof=1)**2 +
                       (nb - 1) * b.std(ddof=1)**2) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled


def mw_auc(a, b):
    """
    Mann-Whitney U AUC = P(water > veg).
    AUC > 0.5 means 'a' tends to be larger; AUC = 1.0 → perfect separation.
    """
    stat, _ = stats.mannwhitneyu(a, b, alternative='two-sided')
    return stat / (len(a) * len(b))


def separability_label(auc):
    a = abs(auc - 0.5)      # distance from 0.5
    if a >= 0.40: return 'EXCELLENT'
    if a >= 0.30: return 'GOOD'
    if a >= 0.15: return 'MODERATE'
    if a >= 0.05: return 'WEAK'
    return 'NONE'


# ── load data ─────────────────────────────────────────────────────────────────

print("Loading all_predictions.csv …")
pred_df = pd.read_csv('all_predictions.csv')

print("Loading features.csv …")
feat_df = pd.read_csv('features.csv')

# align features into pred_df
for col in WAVEFORM_FEATURES:
    if col not in pred_df.columns and col in feat_df.columns:
        pred_df[col] = feat_df[col].values

# ── select subsets ────────────────────────────────────────────────────────────

water_mask = (pred_df['pred_A'] == 1) & (pred_df['auto_label'] == 1)
veg_mask   = (pred_df['pred_A'] == 0) & (pred_df['auto_label'] == 0) & (pred_df['z'] > 264.0)

water_idx = pred_df.index[water_mask].to_numpy()
veg_idx   = pred_df.index[veg_mask].to_numpy()

print(f"\nPool: {len(water_idx):,} confirmed water  |  {len(veg_idx):,} confirmed vegetation (z>264m)")

n_w = min(N_SAMPLE, len(water_idx))
n_v = min(N_SAMPLE, len(veg_idx))
water_sel = rng.choice(water_idx, size=n_w, replace=False)
veg_sel   = rng.choice(veg_idx,  size=n_v, replace=False)
print(f"Sampled: {n_w} water  |  {n_v} vegetation")

water_df = pred_df.loc[water_sel].reset_index(drop=True)
veg_df   = pred_df.loc[veg_sel  ].reset_index(drop=True)

# ── feature separability table ────────────────────────────────────────────────

rows = []
for feat in WAVEFORM_FEATURES:
    if feat not in water_df.columns:
        continue
    w_vals = water_df[feat].dropna().values.astype(float)
    v_vals = veg_df[feat].dropna().values.astype(float)
    if len(w_vals) < 10 or len(v_vals) < 10:
        continue

    d   = cohens_d(w_vals, v_vals)
    auc = mw_auc(w_vals, v_vals)
    _, p = stats.mannwhitneyu(w_vals, v_vals, alternative='two-sided')

    rows.append({
        'feature':         feat,
        'water_mean':      w_vals.mean(),
        'water_std':       w_vals.std(),
        'veg_mean':        v_vals.mean(),
        'veg_std':         v_vals.std(),
        'cohens_d':        d,
        'mw_auc':          auc,
        'p_value':         p,
        'separability':    separability_label(auc),
    })

sep_df = pd.DataFrame(rows).sort_values('mw_auc', key=lambda x: abs(x - 0.5), ascending=False)

print(f"\n{'='*110}")
print(f"{'Feature':<22} {'W mean':>10} {'W std':>9} {'V mean':>10} {'V std':>9} "
      f"{'Cohen d':>9} {'MW-AUC':>8} {'Separability':<14}")
print(f"{'-'*110}")
for _, r in sep_df.iterrows():
    print(f"  {r['feature']:<20} {r['water_mean']:>10.3f} {r['water_std']:>9.3f} "
          f"{r['veg_mean']:>10.3f} {r['veg_std']:>9.3f} "
          f"{r['cohens_d']:>9.2f} {r['mw_auc']:>8.3f}  {r['separability']}")

print(f"\nNote: MW-AUC = P(water > veg). "
      f"AUC > 0.7 or < 0.3 → useful signal. AUC ≈ 0.5 → overlap, not useful.")

# save table
sep_df.to_csv('models/water_vs_veg_separability.csv', index=False)
print(f"\nFull table saved to models/water_vs_veg_separability.csv")

# ── waveform plot: 5+5 examples ───────────────────────────────────────────────

print("\nLoading waveform_grids.npy for example plots …")
grids_all = np.load('waveform_grids.npy', mmap_mode='r')     # (N, 200)

# pick first 5 from each selection (already randomised)
n_ex       = 5
water_rows = water_sel[:n_ex]
veg_rows   = veg_sel[:n_ex]

fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, n_ex, hspace=0.45, wspace=0.25)

water_color = '#1f77b4'
veg_color   = '#2ca02c'

for col, row_idx in enumerate(water_rows):
    ax  = fig.add_subplot(gs[0, col])
    wf  = grids_all[row_idx]
    ax.plot(wf, color=water_color, linewidth=1.2)
    ax.fill_between(range(GRID_LEN), wf, alpha=0.25, color=water_color)
    ax.set_title(
        f"WATER #{col+1}\n"
        f"z={pred_df.loc[row_idx,'z']:.2f}m  "
        f"peaks={pred_df.loc[row_idx,'n_peaks'] if 'n_peaks' in pred_df else '?'}  "
        f"gaps={pred_df.loc[row_idx,'n_gaps'] if 'n_gaps' in pred_df else '?'}",
        fontsize=8)
    ax.set_xlabel('Grid sample', fontsize=7)
    if col == 0:
        ax.set_ylabel('Amplitude (ADC)', fontsize=8)
    ax.tick_params(labelsize=7)

for col, row_idx in enumerate(veg_rows):
    ax  = fig.add_subplot(gs[1, col])
    wf  = grids_all[row_idx]
    ax.plot(wf, color=veg_color, linewidth=1.2)
    ax.fill_between(range(GRID_LEN), wf, alpha=0.25, color=veg_color)
    ax.set_title(
        f"VEGETATION #{col+1}\n"
        f"z={pred_df.loc[row_idx,'z']:.2f}m  "
        f"peaks={pred_df.loc[row_idx,'n_peaks'] if 'n_peaks' in pred_df else '?'}  "
        f"gaps={pred_df.loc[row_idx,'n_gaps'] if 'n_gaps' in pred_df else '?'}",
        fontsize=8)
    ax.set_xlabel('Grid sample', fontsize=7)
    if col == 0:
        ax.set_ylabel('Amplitude (ADC)', fontsize=8)
    ax.tick_params(labelsize=7)

fig.suptitle('Waveform Examples: Confirmed Water (top) vs. Confirmed Vegetation z>264m (bottom)',
             fontsize=12, y=1.01)

# legend patches
from matplotlib.patches import Patch
handles = [Patch(color=water_color, label='Water'),
           Patch(color=veg_color,   label='Vegetation (z>264m)')]
fig.legend(handles=handles, loc='upper right', fontsize=9)

plot_path = 'models/water_vs_veg_waveforms.png'
plt.savefig(plot_path, dpi=180, bbox_inches='tight')
plt.close()
print(f"Waveform plot saved to {plot_path}")

# ── summary by separability tier ─────────────────────────────────────────────

print(f"\n{'='*55}")
print("SEPARABILITY SUMMARY")
print(f"{'='*55}")
for tier in ['EXCELLENT', 'GOOD', 'MODERATE', 'WEAK', 'NONE']:
    feats = sep_df[sep_df['separability'] == tier]['feature'].tolist()
    if feats:
        print(f"  {tier:<12}: {', '.join(feats)}")

print("\nDone.")
