"""
Diagnostic: Transition zone analysis using verified z-boundaries.

Verified boundaries (user visual inspection in CloudCompare):
  CERTAIN_WATER:  z < 261.1m
  UNCERTAIN:      261.1 <= z <= 263.1m
  CERTAIN_CANOPY: z > 263.1m

Outputs (all → models/diagnostics/):
  xy_scatter_z.png
  xy_scatter_density.png
  waveform_clusters_certain_water.png
  waveform_clusters_uncertain.png
  waveform_comparison_10each.png
  reflectance_histograms.png
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "models" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

Z_WATER_MAX = 261.1
Z_CANOPY_MIN = 263.1

# ── 1. Load point cloud ──────────────────────────────────────────────────────
print("Loading point cloud …", flush=True)
pc = pd.read_csv(DATA_DIR / "point_cloud_df.txt")
pc.columns = pc.columns.str.strip()
# normalise column names
pc = pc.rename(columns={"_riegl.reflectance": "reflectance"})
x, y, z = pc["x"].values, pc["y"].values, pc["z"].values
refl = pc["reflectance"].values
n_total = len(pc)
print(f"  {n_total:,} points loaded")

mask_water = z < Z_WATER_MAX
mask_canopy = z > Z_CANOPY_MIN
mask_uncertain = ~mask_water & ~mask_canopy

print(f"\n=== Ground-truth anchors ===")
print(f"  CERTAIN_WATER  (z < {Z_WATER_MAX}m):   {mask_water.sum():>7,}  ({100*mask_water.mean():.1f}%)")
print(f"  UNCERTAIN      ({Z_WATER_MAX}–{Z_CANOPY_MIN}m): {mask_uncertain.sum():>7,}  ({100*mask_uncertain.mean():.1f}%)")
print(f"  CERTAIN_CANOPY (z > {Z_CANOPY_MIN}m):   {mask_canopy.sum():>7,}  ({100*mask_canopy.mean():.1f}%)")

# ── 2. Plot 1: top-down scatter colored by z ────────────────────────────────
print("\nPlot 1: xy scatter colored by z …", flush=True)
fig, ax = plt.subplots(figsize=(12, 8))
sc = ax.scatter(x, y, c=z, s=0.3, cmap="viridis", rasterized=True)
plt.colorbar(sc, ax=ax, label="z (m)")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("Top-down view — colored by elevation z")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(OUT_DIR / "xy_scatter_z.png", dpi=150)
plt.close(fig)
print("  Saved xy_scatter_z.png")

# ── 3. Plot 2: point density (radius 0.3 m) ─────────────────────────────────
print("Plot 2: xy scatter colored by density (r=0.3m) …", flush=True)
xy = np.column_stack([x, y])
tree = cKDTree(xy)
counts = np.array([len(tree.query_ball_point(p, r=0.3)) for p in xy], dtype=np.float32)
print(f"  density: min={counts.min():.0f}  median={np.median(counts):.0f}  max={counts.max():.0f}")

fig, ax = plt.subplots(figsize=(12, 8))
sc = ax.scatter(x, y, c=counts, s=0.3, cmap="hot_r", rasterized=True,
                vmin=np.percentile(counts, 2), vmax=np.percentile(counts, 98))
plt.colorbar(sc, ax=ax, label="Neighbors within 0.3 m")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("Top-down view — colored by local point density (r=0.3m)")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(OUT_DIR / "xy_scatter_density.png", dpi=150)
plt.close(fig)
print("  Saved xy_scatter_density.png")

# ── 4. Parse waveforms ───────────────────────────────────────────────────────
print("\nParsing waveforms (chunked) …", flush=True)

def parse_waveform_clusters(time_arr, amp_arr, gap_threshold=3):
    """Split waveform into clusters where gap > gap_threshold SI samples."""
    if len(time_arr) == 0:
        return []
    clusters = []
    cur_t, cur_a = [time_arr[0]], [amp_arr[0]]
    for i in range(1, len(time_arr)):
        if time_arr[i] - time_arr[i-1] > gap_threshold:
            clusters.append((np.array(cur_t), np.array(cur_a)))
            cur_t, cur_a = [], []
        cur_t.append(time_arr[i])
        cur_a.append(amp_arr[i])
    clusters.append((np.array(cur_t), np.array(cur_a)))
    return clusters

# We'll collect stats for CERTAIN_WATER and UNCERTAIN
water_indices = np.where(mask_water)[0]
uncertain_indices = np.where(mask_uncertain)[0]
water_idx_set = set(water_indices.tolist())
uncertain_idx_set = set(uncertain_indices.tolist())

water_stats = []    # list of dicts
uncertain_stats = []

# Also collect up to 30 raw waveforms for each group for plotting
water_waveforms = []       # (t, a, z_val)
uncertain_waveforms = []

CHUNKSIZE = 10_000
wf_path = DATA_DIR / "waveform_df.txt"
row_offset = 0

for chunk in pd.read_csv(wf_path, chunksize=CHUNKSIZE):
    chunk = chunk.reset_index(drop=True)
    for local_i, row in chunk.iterrows():
        global_i = row_offset + local_i
        try:
            t = np.array(re.findall(r'[-+]?\d+', str(row["Time [SI]"])), dtype=np.float32)
            a = np.array(re.findall(r'[-+]?\d+', str(row["Amplitude [ADC]"])), dtype=np.float32)
            if len(t) == 0 or len(t) != len(a):
                continue
        except Exception:
            continue

        clusters = parse_waveform_clusters(t, a, gap_threshold=3)
        n_clusters = len(clusters)
        gaps = []
        peak_amps = []
        for c_t, c_a in clusters:
            peak_amps.append(float(c_a.max()))
        for ci in range(len(clusters)-1):
            g = float(clusters[ci+1][0][0] - clusters[ci][0][-1])
            gaps.append(g)

        stats = {
            "idx": global_i,
            "n_clusters": n_clusters,
            "mean_gap": float(np.mean(gaps)) if gaps else 0.0,
            "max_gap": float(np.max(gaps)) if gaps else 0.0,
            "peak_amps": peak_amps,
            "max_peak_amp": float(np.max(peak_amps)),
            "min_peak_amp": float(np.min(peak_amps)),
            "waveform_len": len(t),
        }

        if global_i in water_idx_set:
            water_stats.append(stats)
            if len(water_waveforms) < 30:
                water_waveforms.append((t, a, z[global_i]))
        elif global_i in uncertain_idx_set:
            uncertain_stats.append(stats)
            if len(uncertain_waveforms) < 80:
                uncertain_waveforms.append((t, a, z[global_i]))

    row_offset += len(chunk)
    if row_offset % 50_000 == 0:
        print(f"  processed {row_offset:,} / {n_total:,} rows …", flush=True)

print(f"  Collected {len(water_stats):,} CERTAIN_WATER waveform stats")
print(f"  Collected {len(uncertain_stats):,} UNCERTAIN waveform stats")

# ── 5. Helper: summarize cluster stats ──────────────────────────────────────
def summarize_stats(stats_list, label):
    n_clusters = [s["n_clusters"] for s in stats_list]
    mean_gaps  = [s["mean_gap"] for s in stats_list]
    max_peaks  = [s["max_peak_amp"] for s in stats_list]
    lengths    = [s["waveform_len"] for s in stats_list]
    print(f"\n  --- {label} (n={len(stats_list):,}) ---")
    print(f"  num_clusters:  mean={np.mean(n_clusters):.2f}  median={np.median(n_clusters):.1f}  "
          f"[1={sum(c==1 for c in n_clusters)/len(n_clusters)*100:.0f}%  "
          f"2={sum(c==2 for c in n_clusters)/len(n_clusters)*100:.0f}%  "
          f"3+={sum(c>=3 for c in n_clusters)/len(n_clusters)*100:.0f}%]")
    print(f"  mean_gap_SI:   mean={np.mean(mean_gaps):.1f}  median={np.median(mean_gaps):.1f}  "
          f"p95={np.percentile(mean_gaps,95):.1f}")
    print(f"  max_peak_amp:  mean={np.mean(max_peaks):.1f}  median={np.median(max_peaks):.1f}  "
          f"p95={np.percentile(max_peaks,95):.1f}")
    print(f"  waveform_len:  mean={np.mean(lengths):.1f}  median={np.median(lengths):.1f}")
    return n_clusters, mean_gaps, max_peaks

print("\n=== Waveform cluster analysis ===")
w_nc, w_mg, w_mp = summarize_stats(water_stats, "CERTAIN_WATER")
u_nc, u_mg, u_mp = summarize_stats(uncertain_stats, "UNCERTAIN")

# ── 6. Plot 3: cluster stat distributions — CERTAIN_WATER ───────────────────
def plot_cluster_distributions(nc, mg, mp, label, filename):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(nc, bins=range(1, max(nc)+2), edgecolor="white", color="steelblue")
    axes[0].set_xlabel("Number of clusters"); axes[0].set_ylabel("Count")
    axes[0].set_title(f"{label}\nCluster count distribution")

    axes[1].hist(mg, bins=50, color="orange", edgecolor="white")
    axes[1].set_xlabel("Mean inter-cluster gap (SI samples)")
    axes[1].set_title("Gap size distribution")

    axes[2].hist(mp, bins=50, color="green", edgecolor="white")
    axes[2].set_xlabel("Max peak amplitude (ADC)")
    axes[2].set_title("Max peak amplitude distribution")

    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")

print("\nPlot 3+4: cluster distributions …", flush=True)
plot_cluster_distributions(w_nc, w_mg, w_mp, "CERTAIN_WATER", "waveform_clusters_certain_water.png")
plot_cluster_distributions(u_nc, u_mg, u_mp, "UNCERTAIN", "waveform_clusters_uncertain.png")

# ── 7. Plot 5: 10-waveform comparison ───────────────────────────────────────
print("Plot 5: waveform comparison (10 each) …", flush=True)

# Classify uncertain waveforms: "similar to water" = 1 cluster; "different" = 2+ clusters
unc_single = [(t, a, zv) for t, a, zv in uncertain_waveforms
              if len(parse_waveform_clusters(t, a)) == 1]
unc_multi  = [(t, a, zv) for t, a, zv in uncertain_waveforms
              if len(parse_waveform_clusters(t, a)) >= 2]

rng = np.random.default_rng(42)

def sample_n(lst, n):
    idx = rng.choice(len(lst), size=min(n, len(lst)), replace=False)
    return [lst[i] for i in idx]

sel_water = sample_n(water_waveforms, 10)
sel_unc_single = sample_n(unc_single, 10)
sel_unc_multi  = sample_n(unc_multi, 10)

print(f"  Single-cluster uncertain: {len(unc_single)}, multi-cluster: {len(unc_multi)}")

def plot_waveform_grid(waveforms, ax_row_start, axes, title_prefix, color):
    for col_i, (t, a, zv) in enumerate(waveforms):
        ax = axes[ax_row_start][col_i]
        ax.plot(t, a, color=color, linewidth=0.8)
        ax.set_title(f"z={zv:.2f}m", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_ylim(0, None)
        if col_i == 0:
            ax.set_ylabel(title_prefix, fontsize=8)

n_cols = 10
fig, axes = plt.subplots(3, n_cols, figsize=(n_cols * 2.5, 9), sharey=False)
fig.suptitle("Waveform comparison: 10 each\n"
             "Row 1: CERTAIN_WATER  |  Row 2: UNCERTAIN single-cluster (water-like)  |  Row 3: UNCERTAIN multi-cluster (dry-like)",
             fontsize=10)

plot_waveform_grid(sel_water,      0, axes, "CERTAIN\nWATER",       "steelblue")
plot_waveform_grid(sel_unc_single, 1, axes, "UNC\nsingle-cluster",  "darkorange")
plot_waveform_grid(sel_unc_multi,  2, axes, "UNC\nmulti-cluster",   "firebrick")

# hide unused axes
for row in range(3):
    for col in range(len(sel_water if row==0 else sel_unc_single if row==1 else sel_unc_multi), n_cols):
        axes[row][col].set_visible(False)

fig.tight_layout()
fig.savefig(OUT_DIR / "waveform_comparison_10each.png", dpi=150)
plt.close(fig)
print("  Saved waveform_comparison_10each.png")

# ── 8. Plot 6: reflectance histograms ────────────────────────────────────────
print("Plot 6: reflectance histograms …", flush=True)
fig, ax = plt.subplots(figsize=(10, 5))
bins = np.linspace(refl.min(), refl.max(), 80)
ax.hist(refl[mask_water],    bins=bins, alpha=0.6, label=f"CERTAIN_WATER (n={mask_water.sum():,})",    color="steelblue",  density=True)
ax.hist(refl[mask_uncertain],bins=bins, alpha=0.5, label=f"UNCERTAIN (n={mask_uncertain.sum():,})",    color="darkorange", density=True)
ax.hist(refl[mask_canopy],   bins=bins, alpha=0.5, label=f"CERTAIN_CANOPY (n={mask_canopy.sum():,})",  color="green",      density=True)
ax.set_xlabel("Reflectance (dB)")
ax.set_ylabel("Density")
ax.set_title("Reflectance distributions by zone")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "reflectance_histograms.png", dpi=150)
plt.close(fig)
print("  Saved reflectance_histograms.png")

# ── 9. Also: density stats for UNCERTAIN zone ────────────────────────────────
print("\n=== Point density in UNCERTAIN zone ===")
unc_density = counts[mask_uncertain]
print(f"  mean={unc_density.mean():.2f}  median={np.median(unc_density):.2f}  "
      f"p5={np.percentile(unc_density,5):.0f}  p95={np.percentile(unc_density,95):.0f}")

water_density = counts[mask_water]
canopy_density = counts[mask_canopy]
print(f"\n=== Density comparison across zones ===")
print(f"  CERTAIN_WATER:  mean={water_density.mean():.2f}  median={np.median(water_density):.2f}")
print(f"  UNCERTAIN:      mean={unc_density.mean():.2f}   median={np.median(unc_density):.2f}")
print(f"  CERTAIN_CANOPY: mean={canopy_density.mean():.2f}  median={np.median(canopy_density):.2f}")

print(f"\nAll diagnostics saved to: {OUT_DIR}")
print("Done.")
