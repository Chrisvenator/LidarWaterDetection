"""
auto_labeler_v4.py — Physics-based two-stage labeler v4.

Stage 1: Canopy / vegetation filter  (no absolute z)
──────────────────────────────────────────────────────
A point is CANOPY if:
  C1) height_above_local_min (z - min(z) within 3m radius) > CANOPY_H_ABOVE_MIN
      → the point is elevated above its immediate surroundings
  C2) height_std_local > CANOPY_STD_MIN  AND  planarity < CANOPY_PLAN_MAX
      → structurally rough/tall crown even without a large elevation drop nearby

height_above_local_min (3m radius) is precomputed in features_v2.csv — it is
equivalent to: "does any neighbor within 3m lie more than X metres below me?"

Stage 2: Water vs dry ground  (waveform cluster analysis, ground-level only)
──────────────────────────────────────────────────────────────────────────────
The 532nm green laser penetrates water.  A pulse hitting water yields:
  Return 1: water surface     (early cluster, often specular → weaker)
  Return 2: riverbed          (delayed cluster; delay ∝ 2× water depth)
  Water column: exponential decay between the two clusters.

A pulse hitting dry ground yields a single compact return cluster.

Classification rules (applied in order):

PRIMARY WATER:
  ≥2 clusters separated by gap > WATER_GAP_PRIMARY (≈84cm depth at 15 SI)
  Confidence scales with gap size.
  Bonus if first cluster weaker than last (specular surface vs bright bottom).

UNCERTAIN:
  ≥2 clusters with gap WATER_GAP_UNCERT < gap ≤ WATER_GAP_PRIMARY
  (depth signal present but shallow — could be flooded gravel, ~0–84cm)

SECONDARY WATER (single cluster):
  Single return AND reflectance_dB < WATER_SEC_REFL AND planarity > WATER_SEC_PLAN
  → specular flat surface, consistent with smooth water at near-nadir angle

DRY GROUND (default):
  Single compact cluster, no depth signal, normal reflectance

Output:
  pointclouds/labeled_pointcloud_v4.csv
  models/v4-physics-waveform/waveforms_by_class_v4.png
  models/v4-physics-waveform/distributions_v4.png
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent

# ── Stage 1 thresholds ────────────────────────────────────────────────────────
CANOPY_H_ABOVE_MIN = 2.0    # height_above_local_min > this → canopy (m)
CANOPY_STD_MIN     = 0.50   # height_std_local > this  (structural canopy C2)
CANOPY_PLAN_MAX    = 0.30   # planarity < this          (rough crown C2)

# ── Stage 2 waveform cluster thresholds ───────────────────────────────────────
MIN_AMP            = 5      # ADC amplitude: bins below this = noise / empty
MIN_CLUSTER_WIDTH  = 2      # minimum run width (SI bins) to count as a cluster
MIN_GAP            = 5      # gap ≥ this separates two clusters (SI bins)
WATER_GAP_PRIMARY  = 15     # gap > this → primary water label  (≈0.84m depth)
WATER_GAP_UNCERT   = MIN_GAP  # gap in [WATER_GAP_UNCERT, WATER_GAP_PRIMARY] → uncertain

# ── Secondary single-cluster water signal ─────────────────────────────────────
WATER_SEC_REFL     = -15.0  # reflectance_dB < this (specular = water surface)
WATER_SEC_PLAN     = 0.65   # planarity > this (flat surface)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 — canopy filter
# ═══════════════════════════════════════════════════════════════════════════════

def stage1_canopy(df: pd.DataFrame):
    """
    Returns:
      s1_label  (N,) int8   : 1=canopy, 0=ground
      s1_conf   (N,) float32: confidence in label
    """
    N = len(df)

    # height_above_local_min (3m radius) is precomputed in features_v2.csv
    # by add_features.py:  h = z - raster_min(z, radius=3.0)
    h = df['height_above_local_min'].values.astype(np.float32) \
        if 'height_above_local_min' in df.columns else np.zeros(N, np.float32)
    h_std = df['height_std_local'].values.astype(np.float32) \
            if 'height_std_local' in df.columns else np.zeros(N, np.float32)
    plan  = df['planarity'].values.astype(np.float32) \
            if 'planarity' in df.columns else np.ones(N, np.float32)

    # C1: elevated >2m above the 3m-radius local minimum
    c1 = h > CANOPY_H_ABOVE_MIN

    # C2: structurally rough crown (high height variance + low planarity)
    c2 = (h_std > CANOPY_STD_MIN) & (plan < CANOPY_PLAN_MAX)

    canopy = c1 | c2

    # Confidence: distance from threshold, clamped to [0.72, 0.97]
    s1_conf = np.where(
        canopy,
        np.clip(0.75 + 0.18 * (h - CANOPY_H_ABOVE_MIN) / 3.0, 0.72, 0.97),
        np.clip(0.75 + 0.18 * (CANOPY_H_ABOVE_MIN - h) / 3.0, 0.72, 0.97),
    ).astype(np.float32)

    s1_label = canopy.astype(np.int8)

    n_c1 = int(c1.sum());  n_c2 = int(c2.sum());  n_tot = int(canopy.sum())
    print(f"  C1 (elevation drop >2m in 3m radius): {n_c1:>7,}")
    print(f"  C2 (high height_std + low planarity) : {n_c2:>7,}")
    print(f"  Union (canopy total)                 : {n_tot:>7,}  "
          f"({100*n_tot/N:.1f}%)")
    print(f"  Ground-level                         : {N-n_tot:>7,}  "
          f"({100*(N-n_tot)/N:.1f}%)")
    print(f"  height_above_local_min stats → "
          f"p10={np.percentile(h, 10):.2f}  "
          f"p50={np.percentile(h, 50):.2f}  "
          f"p90={np.percentile(h, 90):.2f} m")

    return s1_label, s1_conf


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 — waveform cluster analysis
# ═══════════════════════════════════════════════════════════════════════════════

def parse_clusters(wf: np.ndarray):
    """
    Parse a single 200-bin waveform grid into amplitude clusters.

    Algorithm:
      1. Mark bins with amplitude >= MIN_AMP as "active".
      2. Merge active runs separated by < MIN_GAP zero-bins (surface roughness,
         not a genuine second return).
      3. Discard merged runs narrower than MIN_CLUSTER_WIDTH.
      4. Return cluster list and inter-cluster gap sizes.

    Returns:
      clusters : list of dicts  {start, end, peak_amp, peak_pos, energy, width}
      gaps     : list of int    gap widths (SI bins) between consecutive clusters
    """
    active = wf >= MIN_AMP
    T = len(active)
    if not active.any():
        return [], []

    # Run-length encode: find (start, end_exclusive) for every active run
    diff   = np.diff(active.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff ==  1)[0]
    ends   = np.where(diff == -1)[0]   # exclusive end

    # Filter by minimum cluster width
    wide = (ends - starts) >= MIN_CLUSTER_WIDTH
    starts, ends = starts[wide], ends[wide]
    if len(starts) == 0:
        return [], []

    # Merge clusters separated by a gap < MIN_GAP
    ms, me = [starts[0]], [ends[0]]
    for i in range(1, len(starts)):
        if starts[i] - me[-1] < MIN_GAP:
            me[-1] = ends[i]          # extend previous cluster
        else:
            ms.append(starts[i])
            me.append(ends[i])

    clusters, gaps = [], []
    for i, (s, e) in enumerate(zip(ms, me)):
        seg = wf[s:e]
        clusters.append({
            'start':    int(s),
            'end':      int(e),
            'peak_amp': float(seg.max()),
            'peak_pos': int(s + seg.argmax()),
            'energy':   float(seg.sum()),
            'width':    int(e - s),
        })
        if i > 0:
            gaps.append(int(ms[i] - me[i - 1]))

    return clusters, gaps


def stage2_water(df_ground: pd.DataFrame, grids: np.ndarray, ground_idx: np.ndarray):
    """
    Classify each ground-level point as water (1), dry (0), or uncertain (-1).

    Returns:
      s2_label      (M,) int8
      s2_conf       (M,) float32
      n_clusters_v4 (M,) int16   — clusters found by this labeler
      max_gap_v4    (M,) float32 — largest gap between any two clusters
      gap_12_v4     (M,) float32 — gap specifically between cluster 1 and 2
    """
    M = len(ground_idx)
    s2_label = np.full(M, -1, dtype=np.int8)
    s2_conf  = np.full(M, 0.50, dtype=np.float32)

    n_cl_out = np.zeros(M, dtype=np.int16)
    max_gap  = np.zeros(M, dtype=np.float32)
    gap_12   = np.zeros(M, dtype=np.float32)

    refl = df_ground['reflectance_dB'].values \
           if 'reflectance_dB' in df_ground.columns else np.full(M, 0.0)
    plan = df_ground['planarity'].values \
           if 'planarity'       in df_ground.columns else np.full(M, 0.5)

    for i, orig_i in enumerate(ground_idx):
        if i % 25_000 == 0 and i > 0:
            print(f"    {i:>7,}/{M:,} …")

        wf       = grids[orig_i].astype(np.float32)
        clusters, gaps = parse_clusters(wf)

        n_cl = len(clusters)
        n_cl_out[i] = n_cl

        if gaps:
            max_gap[i] = float(max(gaps))
        if len(gaps) >= 1:
            gap_12[i] = float(gaps[0])

        if n_cl == 0:
            # No signal above noise floor — leave as uncertain
            continue

        max_g = float(max(gaps)) if gaps else 0.0

        # ── Primary water: two distinct returns, deep enough gap ──────────────
        if n_cl >= 2 and max_g >= WATER_GAP_PRIMARY:
            gap_excess = max_g - WATER_GAP_PRIMARY
            c = float(np.clip(0.78 + gap_excess / 60.0, 0.78, 0.97))
            # Boost: specular surface return is weaker than the brighter bottom
            if clusters[0]['peak_amp'] < clusters[-1]['peak_amp']:
                c = min(c + 0.04, 0.97)
            s2_label[i] = 1
            s2_conf[i]  = c
            continue

        # ── Uncertain: multi-cluster but shallow gap ──────────────────────────
        if n_cl >= 2 and max_g >= WATER_GAP_UNCERT:
            s2_label[i] = -1
            s2_conf[i]  = 0.50
            continue

        # ── Single cluster: secondary signals decide ──────────────────────────
        r = float(refl[i])
        p = float(plan[i])

        if r < WATER_SEC_REFL and p > WATER_SEC_PLAN:
            # Low reflectance + high planarity → specular flat surface
            # Consistent with water surface at near-nadir return angle
            c = float(np.clip(0.68 + (WATER_SEC_REFL - r) / 25.0, 0.68, 0.85))
            s2_label[i] = 1
            s2_conf[i]  = c
        else:
            # Compact single return, normal reflectance → dry ground
            c = float(np.clip(0.70 + p * 0.18, 0.70, 0.90))
            s2_label[i] = 0
            s2_conf[i]  = c

    return s2_label, s2_conf, n_cl_out, max_gap, gap_12


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def plot_waveform_examples(out_df: pd.DataFrame, grids: np.ndarray,
                           out_dir: str, n_each: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)

    CLASSES = [
        ('water — deep\n(≥2 clusters, gap>15 SI)',
         (out_df['final_label'] == 1) & (out_df['n_clusters_v4'] >= 2) & (out_df['max_gap_v4'] >= WATER_GAP_PRIMARY),
         '#1f77b4'),
        ('water — secondary\n(1 cluster, specular)',
         (out_df['final_label'] == 1) & (out_df['n_clusters_v4'] < 2),
         '#17becf'),
        ('dry ground\n(single compact cluster)',
         out_df['final_label'] == 0,
         '#d62728'),
        ('canopy\n(elevated above 3m surroundings)',
         out_df['final_label'] == 2,
         '#2ca02c'),
        ('uncertain\n(gap 5–15 SI, shallow water?)',
         out_df['final_label'] == -1,
         '#ff7f0e'),
    ]

    valid_classes = [(label, mask, color)
                     for label, mask, color in CLASSES
                     if mask.sum() >= 2]

    n_rows = len(valid_classes)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(n_rows, n_each, figsize=(28, n_rows * 3.2),
                              gridspec_kw={'hspace': 0.75, 'wspace': 0.22})
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, (label, mask, color) in enumerate(valid_classes):
        idx = np.where(mask.values)[0]
        chosen = rng.choice(idx, size=min(n_each, len(idx)), replace=False)
        axes[row_idx, 0].set_ylabel(label, fontsize=8.5, labelpad=8)

        for col_idx, pt in enumerate(chosen):
            ax = axes[row_idx, col_idx]
            wf = grids[pt].astype(float)
            ax.plot(wf, color=color, linewidth=0.85)
            ax.fill_between(range(len(wf)), wf, alpha=0.18, color=color)

            r = out_df.iloc[pt]
            n_cl = int(r.get('n_clusters_v4', 0))
            mg   = float(r.get('max_gap_v4', 0))
            z_v  = float(r['z'])
            ax.set_title(f"ncl={n_cl} gap={mg:.0f}\nz={z_v:.2f}m",
                         fontsize=6.5, pad=2)
            ax.set_xlim(0, 200)
            ax.tick_params(labelsize=5)
            if col_idx > 0:
                ax.set_yticklabels([])

        for col_idx in range(len(chosen), n_each):
            axes[row_idx, col_idx].set_visible(False)

    fig.suptitle(
        'Waveform examples by v4 physics-based class\n'
        f'min_amp={MIN_AMP} ADC, min_gap={MIN_GAP} SI, '
        f'water_gap>{WATER_GAP_PRIMARY} SI  (1 SI ≈ 0.5 ns, depth/SI ≈ 5.6 cm)',
        fontsize=11)

    out = os.path.join(out_dir, 'waveforms_by_class_v4.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Waveform examples → {out}")


def plot_distributions(out_df: pd.DataFrame, out_dir: str):
    features = [
        ('max_gap_v4',     'Max inter-cluster gap (SI bins)',   True),
        ('n_clusters_v4',  'Clusters per waveform',             False),
        ('reflectance_dB', 'Reflectance (dB)',                  True),
    ]

    label_info = {
        1:  ('water',      '#1f77b4'),
        0:  ('dry ground', '#d62728'),
        2:  ('canopy',     '#2ca02c'),
        -1: ('uncertain',  '#ff7f0e'),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for ax, (feat, xlabel, continuous) in zip(axes, features):
        if feat not in out_df.columns:
            ax.set_visible(False)
            continue
        all_vals = []
        for lbl, (name, color) in label_info.items():
            sub = out_df[out_df['final_label'] == lbl][feat].dropna().values.astype(float)
            if len(sub) < 5:
                continue
            all_vals.append(sub)
            if continuous:
                lo, hi = np.percentile(sub, 1), np.percentile(sub, 99)
                bins = np.linspace(lo, hi, 50)
            else:
                bins = np.arange(sub.min(), sub.max() + 2)
            ax.hist(sub, bins=bins, alpha=0.45, color=color, density=True,
                    label=f"{name} (n={len(sub):,})")
            ax.axvline(np.median(sub), color=color, lw=1.4, ls='--', alpha=0.8)

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=7)
        ax.set_title(feat, fontsize=9)

    fig.suptitle('Feature distributions by v4 label  (dashed = median)', fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, 'distributions_v4.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Distributions      → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    feat_path  = ROOT / 'data_processed' / 'features_v2.csv'
    grids_path = ROOT / 'data_processed' / 'waveform_grids.npy'
    out_csv    = ROOT / 'pointclouds'    / 'labeled_pointcloud_v4.csv'
    out_dir    = ROOT / 'models'         / 'v4-physics-waveform'
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {feat_path} …")
    df = pd.read_csv(feat_path)
    N  = len(df)
    print(f"  {N:,} points × {len(df.columns)} features")

    print(f"Loading {grids_path} …")
    grids = np.load(grids_path, mmap_mode='r')
    print(f"  grid shape: {grids.shape}  dtype: {grids.dtype}")

    # ── Stage 1 ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STAGE 1 — Canopy filter  (no absolute z)")
    print(f"  C1: height_above_local_min_3m > {CANOPY_H_ABOVE_MIN}m")
    print(f"  C2: height_std_local > {CANOPY_STD_MIN}  AND  planarity < {CANOPY_PLAN_MAX}")
    print(f"{'='*60}")

    s1_label, s1_conf = stage1_canopy(df)

    ground_mask = s1_label == 0
    ground_idx  = np.where(ground_mask)[0]
    df_ground   = df.iloc[ground_idx].reset_index(drop=True)

    # ── Stage 2 ────────────────────────────────────────────────────────────────
    n_gnd = len(ground_idx)
    print(f"\n{'='*60}")
    print(f"STAGE 2 — Water vs dry ground  ({n_gnd:,} ground points)")
    print(f"  Waveform clusters: min_amp={MIN_AMP} ADC, min_width={MIN_CLUSTER_WIDTH} SI,")
    print(f"                     min_gap={MIN_GAP} SI to separate clusters")
    print(f"  Primary water  : ≥2 clusters, gap > {WATER_GAP_PRIMARY} SI (≈{WATER_GAP_PRIMARY*0.056:.2f}m depth)")
    print(f"  Uncertain      : ≥2 clusters, gap {WATER_GAP_UNCERT}–{WATER_GAP_PRIMARY} SI")
    print(f"  Secondary water: 1 cluster, refl < {WATER_SEC_REFL} dB AND plan > {WATER_SEC_PLAN}")
    print(f"{'='*60}")

    s2_label, s2_conf, n_cl, max_gap, gap_12 = stage2_water(df_ground, grids, ground_idx)

    # ── Stage 2 summary ────────────────────────────────────────────────────────
    n_water = int((s2_label == 1).sum())
    n_dry   = int((s2_label == 0).sum())
    n_unc   = int((s2_label == -1).sum())
    n_deep  = int(((n_cl >= 2) & (max_gap >= WATER_GAP_PRIMARY)).sum())
    n_sec   = int(((s2_label == 1) & (n_cl < 2)).sum())
    n_shallowgap = int(((n_cl >= 2) & (max_gap >= WATER_GAP_UNCERT)
                        & (max_gap < WATER_GAP_PRIMARY)).sum())

    print(f"\n  Water      (1): {n_water:>7,}  ({100*n_water/N:.1f}%)")
    print(f"    — primary deep-return:  {n_deep:>7,}")
    print(f"    — secondary specular:   {n_sec:>7,}")
    print(f"  Dry ground (0): {n_dry:>7,}  ({100*n_dry/N:.1f}%)")
    print(f"  Uncertain (-1): {n_unc:>7,}  ({100*n_unc/N:.1f}%)")
    print(f"\n  Waveform cluster breakdown (ground pts):")
    print(f"    ≥2 clusters, gap>{WATER_GAP_PRIMARY} SI (deep water):     {n_deep:>7,}")
    print(f"    ≥2 clusters, gap {WATER_GAP_UNCERT}–{WATER_GAP_PRIMARY} SI (uncertain):  {n_shallowgap:>7,}")
    n_1cl = int((n_cl == 1).sum())
    n_0cl = int((n_cl == 0).sum())
    print(f"    1 cluster  (dry or shallow water):      {n_1cl:>7,}")
    print(f"    0 clusters (no valid signal):           {n_0cl:>7,}")

    # z-range per label
    z_all = df['z'].values
    z_gnd = z_all[ground_idx]
    for lbl, name in [(1, 'Water'), (0, 'Dry ground'), (-1, 'Uncertain')]:
        mask = s2_label == lbl
        if mask.sum() > 0:
            zv = z_gnd[mask]
            print(f"  {name:<12} z range: {zv.min():.2f}–{zv.max():.2f}  "
                  f"median={np.median(zv):.2f} m")

    # gap stats for water labels
    water_mask = s2_label == 1
    if water_mask.sum() > 0:
        wg = max_gap[water_mask]
        print(f"\n  Water gap stats: "
              f"p10={np.percentile(wg,10):.1f}  "
              f"p50={np.percentile(wg,50):.1f}  "
              f"p90={np.percentile(wg,90):.1f} SI bins "
              f"  (depth ≈ ×{0.056:.3f}m)")

    # ── Assemble output ────────────────────────────────────────────────────────
    print(f"\nAssembling output …")

    # Full-N arrays (fill canopy/unprocessed with sentinel -2)
    s2_label_all = np.full(N, -2, dtype=np.int8)
    s2_conf_all  = np.full(N, np.nan, dtype=np.float32)
    n_cl_all     = np.zeros(N, dtype=np.int16)
    max_gap_all  = np.zeros(N, dtype=np.float32)
    gap_12_all   = np.zeros(N, dtype=np.float32)

    s2_label_all[ground_idx] = s2_label
    s2_conf_all[ground_idx]  = s2_conf
    n_cl_all[ground_idx]     = n_cl
    max_gap_all[ground_idx]  = max_gap
    gap_12_all[ground_idx]   = gap_12

    # Final label: 0=dry 1=water 2=canopy -1=uncertain
    final_label = np.where(s1_label == 1,        2,   # canopy
                  np.where(s2_label_all == 1,    1,   # water
                  np.where(s2_label_all == 0,    0,   # dry
                                                 -1   # uncertain
                  ))).astype(np.int8)

    out_df = pd.DataFrame({
        'x':             df['x'].values,
        'y':             df['y'].values,
        'z':             df['z'].values,
        'reflectance_dB': df['reflectance_dB'].values
                          if 'reflectance_dB' in df.columns else 0.0,
        # Stage 1
        's1_label':      s1_label,
        's1_conf':       np.round(s1_conf, 4),
        # Stage 2
        's2_label':      s2_label_all,     # -2=canopy -1=uncertain 0=dry 1=water
        's2_conf':       np.round(s2_conf_all, 4),
        # Waveform cluster stats (diagnostic)
        'n_clusters_v4': n_cl_all,
        'max_gap_v4':    np.round(max_gap_all, 1),
        'gap_12_v4':     np.round(gap_12_all,  1),
        # Final
        'final_label':   final_label,      # 0=dry 1=water 2=canopy -1=uncertain
    })

    # Carry through scalar fields useful in CloudCompare
    for col in ['planarity', 'roughness', 'energy_concentration',
                'height_percentile_local', 'height_std_local',
                'height_above_local_min', 'height_above_local_min_10m',
                'n_peaks', 'depth_proxy_m', 'amplitude_weighted_center',
                'max_amp_norm_by_energy', 'active_bins_ratio']:
        if col in df.columns:
            out_df[col] = df[col].values

    out_df.to_csv(out_csv, index=False)
    print(f"Saved {N:,} rows → {out_csv}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    print(f"\nGenerating plots …")
    plot_waveform_examples(out_df, grids, str(out_dir))
    plot_distributions(out_df, str(out_dir))

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FINAL LABEL DISTRIBUTION")
    print(f"{'='*60}")
    for lbl, name in [(2, 'Canopy'), (1, 'Water'), (0, 'Dry ground'), (-1, 'Uncertain')]:
        n = int((final_label == lbl).sum())
        print(f"  {name:<14} ({lbl:>2}): {n:>7,}  ({100*n/N:.1f}%)")

    print(f"\nOpen {out_csv.name} in CloudCompare:")
    print(f"  Colour by 'final_label':  -1=uncertain  0=dry  1=water  2=canopy")
    print(f"  Inspect 'n_clusters_v4'   — cluster count from waveform parsing")
    print(f"  Inspect 'max_gap_v4'      — SI bins between clusters (depth proxy)")
    print(f"  Inspect 'height_above_local_min' — Stage 1 elevation signal")
    print(f"\n  Depth estimate: depth_m ≈ gap_12_v4 × 0.056")
    print(f"\nDone.")


if __name__ == '__main__':
    main()
