"""
add_features.py — Append new generalizable features to features.csv → features_v2.csv.

New relative elevation features (grid-rasterised, work at any absolute altitude):
  height_above_local_min     : z - min(z within 3 m radius)   water≈0, veg>0
  height_above_local_min_10m : z - min(z within 10 m radius)  broader terrain context
  height_percentile_local    : (z - local_min_5m) / (local_max_5m - local_min_5m)
                               water surface≈low, vegetation≈high

New waveform shape features (computed from waveform_grids.npy, 200-bin dense arrays):
  energy_concentration       : fraction of total energy in first 30 bins
  amplitude_weighted_center  : amplitude-weighted mean bin index (centre of mass)
  active_bins_ratio          : (bins > 0 in grid) / 200
  max_amp_norm_by_energy     : max(grid) / (sum(grid) + 1e-6)  — peak sharpness
"""

import sys
import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter, maximum_filter

CELL = 0.5      # grid cell size in metres for rasterised spatial ops
GRIDS_PATH = 'waveform_grids.npy'


# ── raster helpers ────────────────────────────────────────────────────────────

def _make_grid(x, y, z_vals, fill_max=True):
    """Rasterise z values onto a 2D grid, keeping min or max per cell."""
    x_min, y_min = x.min(), y.min()
    nx = int((x.max() - x_min) / CELL) + 2
    ny = int((y.max() - y_min) / CELL) + 2

    ix = np.clip(((x - x_min) / CELL).astype(int), 0, nx - 1)
    iy = np.clip(((y - y_min) / CELL).astype(int), 0, ny - 1)

    fill = np.inf if fill_max else -np.inf
    grid = np.full((ny, nx), fill, dtype=np.float32)
    if fill_max:
        np.minimum.at(grid, (iy, ix), z_vals)
        empty = np.isinf(grid)
        grid[empty] = float(z_vals.max())    # empty cells → global max (won't win min-filter)
    else:
        np.maximum.at(grid, (iy, ix), z_vals)
        empty = np.isinf(grid)
        grid[empty] = float(z_vals.min())

    return grid, ix, iy, x_min, y_min


def local_min_height(x, y, z, radius):
    """z - local_min within radius [m] using rasterised minimum_filter."""
    grid, ix, iy, _, _ = _make_grid(x, y, z.astype(np.float32), fill_max=True)
    k = int(np.ceil(radius / CELL)) * 2 + 1
    local_min = minimum_filter(grid, size=k, mode='nearest')
    return z.astype(np.float32) - local_min[iy, ix]


def local_normalized_rank(x, y, z, radius):
    """(z - local_min) / (local_max - local_min) within radius [m]."""
    x_min, y_min = x.min(), y.min()
    nx = int((x.max() - x_min) / CELL) + 2
    ny = int((y.max() - y_min) / CELL) + 2

    ix = np.clip(((x - x_min) / CELL).astype(int), 0, nx - 1)
    iy = np.clip(((y - y_min) / CELL).astype(int), 0, ny - 1)

    zf = z.astype(np.float32)
    gmin = np.full((ny, nx), np.inf,  dtype=np.float32)
    gmax = np.full((ny, nx), -np.inf, dtype=np.float32)
    np.minimum.at(gmin, (iy, ix), zf)
    np.maximum.at(gmax, (iy, ix), zf)

    # Fill empty cells so they don't influence the filters
    gmin = np.where(np.isinf(gmin),  zf.max(), gmin)
    gmax = np.where(np.isinf(gmax),  zf.min(), gmax)

    k = int(np.ceil(radius / CELL)) * 2 + 1
    lmin = minimum_filter(gmin, size=k, mode='nearest')[iy, ix]
    lmax = maximum_filter(gmax, size=k, mode='nearest')[iy, ix]

    rng = lmax - lmin
    return np.where(rng > 0.01, (zf - lmin) / rng, 0.5)


# ── waveform grid features ────────────────────────────────────────────────────

def waveform_shape_features(grids: np.ndarray, time_span: np.ndarray):
    """
    grids     : (N, 200) uint16/float — dense amplitude grid
    time_span : (N,)     int           — existing feature from features.csv
    Returns dict of arrays, each length N.
    """
    g = grids.astype(np.float32)
    total_e = g.sum(axis=1)                         # (N,)
    safe_e  = np.where(total_e > 0, total_e, 1.0)

    # Fraction of energy in first 30 bins
    first30 = g[:, :30].sum(axis=1)
    energy_concentration = first30 / safe_e

    # Amplitude-weighted centre of mass (bin index, 0–199)
    bins = np.arange(200, dtype=np.float32)
    amplitude_weighted_center = (g * bins[np.newaxis, :]).sum(axis=1) / safe_e

    # Non-zero bins / 200
    active_bins_ratio = (g > 0).sum(axis=1).astype(np.float32) / 200.0

    # Peak sharpness: max amplitude / total energy
    max_amp_norm_by_energy = g.max(axis=1) / safe_e

    return {
        'energy_concentration':    energy_concentration.astype(np.float32),
        'amplitude_weighted_center': amplitude_weighted_center.astype(np.float32),
        'active_bins_ratio':       active_bins_ratio.astype(np.float32),
        'max_amp_norm_by_energy':  max_amp_norm_by_energy.astype(np.float32),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    features_path = 'features.csv'
    grids_path    = GRIDS_PATH
    out_path      = 'features_v2.csv'

    print(f"Loading {features_path} …")
    feat_df = pd.read_csv(features_path)
    print(f"  {len(feat_df):,} rows × {len(feat_df.columns)} cols")

    x = feat_df['x'].values.astype(np.float32)
    y = feat_df['y'].values.astype(np.float32)
    z = feat_df['z'].values.astype(np.float32)

    # ── relative elevation ─────────────────────────────────────────────────────
    print("Computing height_above_local_min (3 m) …")
    feat_df['height_above_local_min'] = local_min_height(x, y, z, radius=3.0)

    print("Computing height_above_local_min_10m (10 m) …")
    feat_df['height_above_local_min_10m'] = local_min_height(x, y, z, radius=10.0)

    print("Computing height_percentile_local (5 m) …")
    feat_df['height_percentile_local'] = local_normalized_rank(x, y, z, radius=5.0)

    # ── waveform shape ─────────────────────────────────────────────────────────
    print(f"Loading {grids_path} …")
    grids = np.load(grids_path, mmap_mode='r')      # (N, 200)
    print(f"  shape: {grids.shape}")

    time_span = feat_df['time_span'].values
    print("Computing waveform shape features …")
    wf_feats = waveform_shape_features(grids, time_span)
    for name, arr in wf_feats.items():
        feat_df[name] = arr

    # ── save ───────────────────────────────────────────────────────────────────
    feat_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(feat_df):,} rows × {len(feat_df.columns)} cols to {out_path}")

    # quick sanity check
    new_cols = ['height_above_local_min', 'height_above_local_min_10m',
                'height_percentile_local', 'energy_concentration',
                'amplitude_weighted_center', 'active_bins_ratio',
                'max_amp_norm_by_energy']
    print("\nNew feature ranges:")
    for col in new_cols:
        vals = feat_df[col]
        print(f"  {col:<35} min={vals.min():.4f}  max={vals.max():.4f}  "
              f"mean={vals.mean():.4f}")
    print("Done.")


if __name__ == '__main__':
    main()
