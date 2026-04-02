"""
diagnostic_water_vs_dryground.py

The critical comparison: water vs. dry ground at SIMILAR elevations.

  WATER      : pred_A==1 AND auto_label==1 AND 258 <= z <= 263
  DRY GROUND : pred_A==0 AND auto_label==0 AND 263 <= z <= 266
               (meadow, gravel bank — right next to the river)

For every waveform feature + reflectance:
  mean ± std per class, Cohen's d, Mann-Whitney U AUC, p-value.
Plot 5 example waveforms per class from the dense grid.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

WAVEFORM_FEATURES = [
    'max_amp', 'mean_amp', 'std_amp', 'total_energy', 'n_samples',
    'time_span', 'n_gaps', 'max_gap', 'mean_gap', 'total_gap',
    'n_peaks', 'max_peak_spacing', 'mean_peak_spacing',
    'first_last_span', 'n_clusters', 'energy_ratio_late',
    'first_peak_amp', 'last_peak_amp', 'peak_amp_ratio', 'depth_proxy_m',
    'reflectance_dB',
    # New waveform shape features from features_v2.csv
    'energy_concentration', 'amplitude_weighted_center',
    'active_bins_ratio', 'max_amp_norm_by_energy',
]

N_SAMPLE  = 500
SEED      = 42
rng       = np.random.default_rng(SEED)

Z_WATER_LO  = 258.0
Z_WATER_HI  = 263.0
Z_DRY_LO    = 263.0
Z_DRY_HI    = 266.0


# ── helpers ───────────────────────────────────────────────────────────────────

def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*a.std(ddof=1)**2 + (nb-1)*b.std(ddof=1)**2) / (na+nb-2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0

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

print("Loading all_predictions.csv …")
pred_df = pd.read_csv('all_predictions.csv')

# Pull in v2 waveform shape features if available
try:
    feat_v2 = pd.read_csv('features_v2.csv',
                          usecols=['energy_concentration','amplitude_weighted_center',
                                   'active_bins_ratio','max_amp_norm_by_energy'])
    for c in feat_v2.columns:
        pred_df[c] = feat_v2[c].values
    print("  Merged waveform shape features from features_v2.csv")
except Exception as e:
    print(f"  features_v2.csv not fully available ({e}), skipping shape features")

# Pull in any missing waveform features from features.csv
try:
    feat_df = pd.read_csv('features.csv')
    for col in WAVEFORM_FEATURES:
        if col not in pred_df.columns and col in feat_df.columns:
            pred_df[col] = feat_df[col].values
except Exception:
    pass

print(f"\nZ distribution of confident labels:")
conf = pred_df['auto_label'].isin([0, 1])
print(f"  water (auto=1): z range "
      f"{pred_df.loc[pred_df['auto_label']==1,'z'].min():.2f} – "
      f"{pred_df.loc[pred_df['auto_label']==1,'z'].max():.2f}")
print(f"  land  (auto=0): z range "
      f"{pred_df.loc[pred_df['auto_label']==0,'z'].min():.2f} – "
      f"{pred_df.loc[pred_df['auto_label']==0,'z'].max():.2f}")

# ── select groups ─────────────────────────────────────────────────────────────

water_mask = (
    (pred_df['pred_A'] == 1) &
    (pred_df['auto_label'] == 1) &
    (pred_df['z'] >= Z_WATER_LO) &
    (pred_df['z'] <= Z_WATER_HI)
)
dry_mask = (
    (pred_df['pred_A'] == 0) &
    (pred_df['auto_label'] == 0) &
    (pred_df['z'] >= Z_DRY_LO) &
    (pred_df['z'] <= Z_DRY_HI)
)

water_idx = pred_df.index[water_mask].to_numpy()
dry_idx   = pred_df.index[dry_mask  ].to_numpy()

print(f"\nPool sizes after z-gate:")
print(f"  WATER  ({Z_WATER_LO}–{Z_WATER_HI} m): {len(water_idx):,} points")
print(f"  DRY    ({Z_DRY_LO}–{Z_DRY_HI} m):   {len(dry_idx):,}   points")

if len(water_idx) == 0 or len(dry_idx) == 0:
    print("\nERROR: one of the groups is empty — check z thresholds vs. data range.")
    print("Actual z ranges in data:")
    print(pred_df['z'].describe())
    raise SystemExit(1)

n_w = min(N_SAMPLE, len(water_idx))
n_d = min(N_SAMPLE, len(dry_idx))
water_sel = rng.choice(water_idx, size=n_w, replace=False)
dry_sel   = rng.choice(dry_idx,  size=n_d, replace=False)
print(f"Sampled {n_w} water, {n_d} dry ground")

water_df = pred_df.loc[water_sel].reset_index(drop=True)
dry_df   = pred_df.loc[dry_sel  ].reset_index(drop=True)

# ── separability table ────────────────────────────────────────────────────────

rows = []
for feat in WAVEFORM_FEATURES:
    if feat not in water_df.columns:
        continue
    w = water_df[feat].dropna().values.astype(float)
    d = dry_df  [feat].dropna().values.astype(float)
    if len(w) < 10 or len(d) < 10:
        continue

    cd  = cohens_d(w, d)
    auc = mw_auc(w, d)
    _, p = stats.mannwhitneyu(w, d, alternative='two-sided')

    rows.append({
        'feature':      feat,
        'water_mean':   w.mean(),
        'water_std':    w.std(),
        'dry_mean':     d.mean(),
        'dry_std':      d.std(),
        'cohens_d':     cd,
        'mw_auc':       auc,
        'p_value':      p,
        'separability': sep_label(auc),
    })

sep_df = pd.DataFrame(rows).sort_values(
    'mw_auc', key=lambda x: abs(x - 0.5), ascending=False)

print(f"\n{'='*120}")
print(f"WATER (z {Z_WATER_LO}–{Z_WATER_HI}m) vs DRY GROUND (z {Z_DRY_LO}–{Z_DRY_HI}m)")
print(f"{'='*120}")
print(f"{'Feature':<28} {'W mean':>10} {'W std':>9} {'Dry mean':>10} {'Dry std':>9} "
      f"{'Cohen d':>9} {'MW-AUC':>8}  {'Separability'}")
print(f"{'-'*120}")
for _, r in sep_df.iterrows():
    flag = ' ***' if r['separability'] in ('EXCELLENT', 'GOOD') else ''
    print(f"  {r['feature']:<26} {r['water_mean']:>10.3f} {r['water_std']:>9.3f} "
          f"{r['dry_mean']:>10.3f} {r['dry_std']:>9.3f} "
          f"{r['cohens_d']:>9.2f} {r['mw_auc']:>8.3f}  {r['separability']}{flag}")

print(f"\nMW-AUC > 0.7 or < 0.3 → useful signal. AUC ≈ 0.5 → no signal.")
print(f"AUC < 0.5 means water < dry for that feature.\n")

sep_df.to_csv('models/water_vs_dryground_separability.csv', index=False)
print(f"Table saved to models/water_vs_dryground_separability.csv")

# ── tier summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SEPARABILITY SUMMARY")
print(f"{'='*60}")
for tier in ['EXCELLENT', 'GOOD', 'MODERATE', 'WEAK', 'NONE']:
    feats = sep_df[sep_df['separability'] == tier]['feature'].tolist()
    if feats:
        print(f"  {tier:<12}: {', '.join(feats)}")

# ── waveform plots ────────────────────────────────────────────────────────────
print(f"\nLoading waveform_grids.npy …")
grids_all = np.load('waveform_grids.npy', mmap_mode='r')

n_ex      = 5
water_ex  = water_sel[:n_ex]
dry_ex    = dry_sel[:n_ex]

WATER_COL = '#1f77b4'
DRY_COL   = '#d62728'

fig = plt.figure(figsize=(20, 10))
gs  = gridspec.GridSpec(2, n_ex, hspace=0.55, wspace=0.28)

def fmt_title(row_idx, label):
    r = pred_df.loc[row_idx]
    parts = [f"z={r['z']:.2f}m"]
    for f in ['n_peaks', 'n_gaps', 'reflectance_dB', 'energy_concentration']:
        if f in r.index and pd.notna(r[f]):
            val = r[f]
            if isinstance(val, float):
                parts.append(f"{f.replace('_',' ')}={val:.2f}")
            else:
                parts.append(f"{f.replace('_',' ')}={val}")
    return f"{label}\n" + "\n".join(parts[:3])

for col, row_idx in enumerate(water_ex):
    ax = fig.add_subplot(gs[0, col])
    wf = grids_all[row_idx].astype(float)
    ax.plot(wf, color=WATER_COL, linewidth=1.2)
    ax.fill_between(range(200), wf, alpha=0.2, color=WATER_COL)
    ax.set_title(fmt_title(row_idx, f"WATER #{col+1}"), fontsize=7.5)
    ax.set_xlabel('Grid bin', fontsize=7)
    if col == 0:
        ax.set_ylabel('Amplitude (ADC)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, 200)

for col, row_idx in enumerate(dry_ex):
    ax = fig.add_subplot(gs[1, col])
    wf = grids_all[row_idx].astype(float)
    ax.plot(wf, color=DRY_COL, linewidth=1.2)
    ax.fill_between(range(200), wf, alpha=0.2, color=DRY_COL)
    ax.set_title(fmt_title(row_idx, f"DRY GROUND #{col+1}"), fontsize=7.5)
    ax.set_xlabel('Grid bin', fontsize=7)
    if col == 0:
        ax.set_ylabel('Amplitude (ADC)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, 200)

fig.suptitle(
    f'Waveform Examples: Confirmed WATER z={Z_WATER_LO}–{Z_WATER_HI}m (top, blue)\n'
    f'vs. DRY GROUND z={Z_DRY_LO}–{Z_DRY_HI}m — meadow/gravel bank (bottom, red)',
    fontsize=11, y=1.02)
fig.legend(handles=[Patch(color=WATER_COL, label='Water (river)'),
                    Patch(color=DRY_COL,   label='Dry ground (meadow/gravel)')],
           loc='upper right', fontsize=9)

plot_path = 'models/water_vs_dryground_waveforms.png'
plt.savefig(plot_path, dpi=180, bbox_inches='tight')
plt.close()
print(f"Waveform plot saved to {plot_path}")

# ── extra: distribution comparison bar chart for top features ─────────────────
top_feats = sep_df.head(10)['feature'].tolist()
n_tf = len(top_feats)
if n_tf > 0:
    fig2, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()
    for i, feat in enumerate(top_feats[:10]):
        ax = axes[i]
        w_vals = water_df[feat].dropna().values.astype(float)
        d_vals = dry_df  [feat].dropna().values.astype(float)
        all_vals = np.concatenate([w_vals, d_vals])
        bins = np.linspace(np.percentile(all_vals, 1),
                           np.percentile(all_vals, 99), 35)
        ax.hist(w_vals, bins=bins, alpha=0.55, color=WATER_COL,
                density=True, label='Water')
        ax.hist(d_vals, bins=bins, alpha=0.55, color=DRY_COL,
                density=True, label='Dry')
        row = sep_df[sep_df['feature'] == feat].iloc[0]
        ax.set_title(f"{feat}\nAUC={row['mw_auc']:.3f}  d={row['cohens_d']:.2f}",
                     fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
    for j in range(n_tf, 10):
        axes[j].set_visible(False)
    plt.suptitle('Top-10 Feature Distributions: Water vs. Dry Ground\n'
                 '(same elevation range — this is the real discrimination problem)',
                 fontsize=11)
    plt.tight_layout()
    dist_path = 'models/water_vs_dryground_distributions.png'
    plt.savefig(dist_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Distribution plot saved to {dist_path}")

print("\nDone.")
