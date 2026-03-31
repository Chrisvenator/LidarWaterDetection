"""
feature_extractor.py — Extract waveform + geometric features for all points.

Outputs:
  features.csv      — scalar features per point (for XGBoost)
  waveform_grids.npy — dense 1D grid per waveform, shape (N, 200), for CNN

Geometric features use k-NN PCA on the 3D point cloud:
  planarity, roughness, linearity, sphericity, height_range_local, height_std_local, z_relative
"""

import argparse
import re
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from tqdm import tqdm


GRID_SIZE   = 200   # time bins (covers SI 0 to 199, origin-relative)
KNN_K       = 20    # neighbours for geometric features
MIN_PEAK    = 100   # minimum ADC for peak detection
GAP_THRESH  = 2     # minimum time gap (SI) to count as a gap


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_array_string(s: str) -> np.ndarray:
    nums = re.findall(r'[-+]?\d+', str(s))
    return np.array(nums, dtype=np.int32)


# ---------------------------------------------------------------------------
# Waveform → scalar features  (same as auto_labeler for consistency)
# ---------------------------------------------------------------------------

def extract_waveform_features(times: np.ndarray, amps: np.ndarray) -> dict:
    f = {}
    f['max_amp']      = int(np.max(amps))
    f['mean_amp']     = float(np.mean(amps))
    f['std_amp']      = float(np.std(amps))
    f['total_energy'] = int(np.sum(amps))
    f['n_samples']    = len(times)
    f['time_span']    = int(times[-1] - times[0]) if len(times) > 1 else 0

    if len(times) > 1:
        diffs = np.diff(times.astype(np.int32))
        gaps  = diffs[diffs > GAP_THRESH]
        f['n_gaps']    = int(np.sum(diffs > GAP_THRESH))
        f['max_gap']   = int(np.max(diffs))
        f['mean_gap']  = float(np.mean(gaps)) if len(gaps) > 0 else 0.0
        f['total_gap'] = int(np.sum(gaps))
    else:
        f['n_gaps'] = f['max_gap'] = f['total_gap'] = 0
        f['mean_gap'] = 0.0

    peaks = [i for i in range(1, len(amps) - 1)
             if amps[i] > amps[i-1] and amps[i] > amps[i+1] and amps[i] >= MIN_PEAK]
    f['n_peaks'] = len(peaks)

    if len(peaks) >= 2:
        pt    = times[peaks]
        spc   = np.diff(pt.astype(np.int32))
        f['max_peak_spacing']  = int(np.max(spc))
        f['mean_peak_spacing'] = float(np.mean(spc))
        f['first_last_span']   = int(pt[-1] - pt[0])
    else:
        f['max_peak_spacing'] = f['mean_peak_spacing'] = f['first_last_span'] = 0

    f['n_clusters'] = 1 + f['n_gaps']

    if f['time_span'] > 0:
        cutoff  = times[0] + 0.6 * f['time_span']
        e_late  = int(np.sum(amps[times > cutoff]))
        f['energy_ratio_late'] = float(e_late / (f['total_energy'] + 1))
    else:
        f['energy_ratio_late'] = 0.0

    if peaks:
        f['first_peak_amp'] = int(amps[peaks[0]])
        f['last_peak_amp']  = int(amps[peaks[-1]])
    else:
        f['first_peak_amp'] = f['last_peak_amp'] = 0

    # Amplitude ratio between first and last peak (water: first >> last due to absorption)
    if f['last_peak_amp'] > 0:
        f['peak_amp_ratio'] = float(f['first_peak_amp'] / f['last_peak_amp'])
    else:
        f['peak_amp_ratio'] = 0.0

    # Water depth proxy (m)
    f['depth_proxy_m'] = round(f['max_peak_spacing'] * 0.05625, 3)

    return f


# ---------------------------------------------------------------------------
# Waveform → dense 1D grid (for CNN)
# ---------------------------------------------------------------------------

def waveform_to_grid(times: np.ndarray, amps: np.ndarray,
                     grid_size: int = GRID_SIZE) -> np.ndarray:
    """
    Project non-contiguous waveform onto a fixed-length 1D array.
    Origin-relative: t_min maps to bin 0.
    Missing bins → 0 (below noise floor).
    """
    grid  = np.zeros(grid_size, dtype=np.float32)
    t_min = int(times[0])
    for t, a in zip(times, amps):
        idx = int(t) - t_min
        if 0 <= idx < grid_size:
            grid[idx] = float(a)
    return grid


# ---------------------------------------------------------------------------
# Geometric features from k-NN PCA
# ---------------------------------------------------------------------------

def compute_geometric_features(xyz: np.ndarray, k: int = KNN_K) -> pd.DataFrame:
    """
    Vectorized k-NN PCA geometric features.
    Uses batch np.linalg.eigvalsh on (N, 3, 3) covariance stack.
    """
    n = len(xyz)
    print(f"  Building KDTree on {n:,} points …")
    tree = KDTree(xyz)

    print(f"  Querying k={k} neighbours (batch) …")
    _, indices = tree.query(xyz, k=k + 1)
    indices = indices[:, 1:]                 # (N, k) — drop self

    print(f"  Computing covariance matrices …")
    # nbrs: (N, k, 3)
    nbrs     = xyz[indices]
    means    = nbrs.mean(axis=1, keepdims=True)    # (N, 1, 3)
    centered = nbrs - means                        # (N, k, 3)

    # Batch covariance: (N, 3, 3)
    cov = np.einsum('nki,nkj->nij', centered, centered) / k

    print(f"  Computing batch eigenvalues …")
    # eigvalsh returns ascending eigenvalues; shape (N, 3)
    eigvals = np.linalg.eigvalsh(cov)          # (N, 3), ascending
    l3 = eigvals[:, 2].astype(np.float32)      # largest
    l2 = eigvals[:, 1].astype(np.float32)
    l1 = eigvals[:, 0].astype(np.float32)      # smallest (roughness)

    denom = l3 + 1e-8
    planarity  = (l2 - l1) / denom
    linearity  = (l3 - l2) / denom
    sphericity = l1 / denom
    roughness  = l1                             # smallest eigenvalue

    z_nbrs          = nbrs[:, :, 2]            # (N, k)
    height_range_loc = (z_nbrs.max(axis=1) - z_nbrs.min(axis=1)).astype(np.float32)
    height_std_loc   = z_nbrs.std(axis=1).astype(np.float32)
    z_relative       = (xyz[:, 2] - z_nbrs.mean(axis=1)).astype(np.float32)

    return pd.DataFrame({
        'planarity':          planarity,
        'roughness':          roughness,
        'linearity':          linearity,
        'sphericity':         sphericity,
        'height_range_local': height_range_loc,
        'height_std_local':   height_std_loc,
        'z_relative':         z_relative,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(pc_path: str, wf_path: str,
        out_csv: str, out_grids: str,
        skip_geo: bool) -> None:

    print(f"Loading point cloud …")
    pc = pd.read_csv(pc_path)
    print(f"  {len(pc):,} points")

    print(f"Loading waveforms …")
    wf_df = pd.read_csv(wf_path)
    print(f"  {len(wf_df):,} rows")

    time_col = 'Time [SI]'
    amp_col  = 'Amplitude [ADC]'

    # ── Step 1: Waveform features + dense grids ──────────────────────────────
    print(f"\nExtracting waveform features …")
    wf_records = []
    grids      = np.zeros((len(pc), GRID_SIZE), dtype=np.float32)

    for i in tqdm(range(len(pc)), unit='pt', ncols=80):
        try:
            times = parse_array_string(wf_df.iloc[i][time_col])
            amps  = parse_array_string(wf_df.iloc[i][amp_col])
            if len(times) == 0 or len(times) != len(amps):
                raise ValueError("bad arrays")
            feat = extract_waveform_features(times, amps)
            grids[i] = waveform_to_grid(times, amps)
        except Exception:
            feat = {k: 0 for k in [
                'max_amp', 'mean_amp', 'std_amp', 'total_energy', 'n_samples',
                'time_span', 'n_gaps', 'max_gap', 'mean_gap', 'total_gap',
                'n_peaks', 'max_peak_spacing', 'mean_peak_spacing',
                'first_last_span', 'n_clusters', 'energy_ratio_late',
                'first_peak_amp', 'last_peak_amp', 'peak_amp_ratio', 'depth_proxy_m'
            ]}
        wf_records.append(feat)

    wf_feat_df = pd.DataFrame(wf_records)

    # Save grids
    print(f"Saving waveform grids to {out_grids}  shape={grids.shape} …")
    np.save(out_grids, grids)

    # ── Step 2: Point cloud scalar features ──────────────────────────────────
    x_mean = pc['x'].mean()
    y_mean = pc['y'].mean()

    pc_feat = pd.DataFrame({
        'x_rel':          (pc['x'] - x_mean).values.astype(np.float32),
        'y_rel':          (pc['y'] - y_mean).values.astype(np.float32),
        'z':              pc['z'].values.astype(np.float32),
        'reflectance_dB': pc['_riegl.reflectance'].values.astype(np.float32),
    })

    # ── Step 3: Geometric k-NN features ──────────────────────────────────────
    if skip_geo:
        print("\nSkipping geometric features (--skip-geo flag set).")
        geo_feat = pd.DataFrame({
            'planarity':          np.zeros(len(pc), dtype=np.float32),
            'roughness':          np.zeros(len(pc), dtype=np.float32),
            'linearity':          np.zeros(len(pc), dtype=np.float32),
            'sphericity':         np.zeros(len(pc), dtype=np.float32),
            'height_range_local': np.zeros(len(pc), dtype=np.float32),
            'height_std_local':   np.zeros(len(pc), dtype=np.float32),
            'z_relative':         np.zeros(len(pc), dtype=np.float32),
        })
    else:
        print("\nComputing geometric (k-NN PCA) features …")
        xyz = pc[['x', 'y', 'z']].values.astype(np.float64)
        geo_feat = compute_geometric_features(xyz, k=KNN_K)

    # ── Combine all features ─────────────────────────────────────────────────
    combined = pd.concat([
        pc[['x', 'y', 'z', '_riegl.reflectance']].reset_index(drop=True),
        pc_feat.reset_index(drop=True),
        wf_feat_df.reset_index(drop=True),
        geo_feat.reset_index(drop=True),
    ], axis=1)

    combined.to_csv(out_csv, index=False)
    print(f"\nFeatures saved to {out_csv}  ({len(combined):,} rows × {len(combined.columns)} cols)")
    print(f"Waveform grids saved to {out_grids}  shape={grids.shape}")


def main():
    ap = argparse.ArgumentParser(description='Feature extraction for LiDAR classifier')
    ap.add_argument('--pc',        default='data/point_cloud_df.txt')
    ap.add_argument('--wf',        default='data/waveform_df.txt')
    ap.add_argument('--out',       default='features.csv')
    ap.add_argument('--grid-out',  default='waveform_grids.npy')
    ap.add_argument('--skip-geo',  action='store_true',
                    help='Skip slow k-NN geometric features (for quick testing)')
    args = ap.parse_args()
    run(args.pc, args.wf, args.out, args.grid_out, args.skip_geo)


if __name__ == '__main__':
    main()
