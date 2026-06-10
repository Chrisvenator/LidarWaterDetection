"""Canopy feature extraction: echo features, DTM/DSM heights, local 3D structure.

Groups points sharing a waveform row into pulses (multi-echo recovery), builds
1 m DTM/DSM grids, computes cylinder/sphere neighborhood stats, merges
generalizable waveform features from features_current.csv.

Output: data_processed/canopy_features.csv (cached — delete or --force to rebuild)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent.parent
POINT_CLOUD_PATH = ROOT / "data" / "point_cloud_df.txt"
WAVEFORM_PATH = ROOT / "data" / "waveform_df.txt"
FEATURES_IN_PATH = ROOT / "data_processed" / "features_current.csv"
V10_PATH = ROOT / "pointclouds" / "labeled_pointcloud_v10.csv"
OUT_PATH = ROOT / "data_processed" / "canopy_features.csv"

CELL_M = 1.0            # DTM/DSM grid cell size
R_LOCAL_M = 1.0         # cylinder/sphere radius for structure features
ABOVE_GAP_M = 2.0       # neighbor this far above a point = canopy cover above
DTM_PERCENTILE = 0.05   # robust per-cell ground elevation
CHUNK = 50_000          # neighbor-query chunk size (bounds memory)

# Generalizable columns carried over from features_current.csv
EXISTING_COLS = [
    "reflectance_dB", "planarity", "roughness", "linearity", "sphericity",
    "height_range_local", "height_std_local", "height_percentile_local",
    "n_peaks", "n_gaps", "n_clusters", "time_span", "energy_concentration",
    "amplitude_weighted_center", "active_bins_ratio", "max_amp_norm_by_energy",
    "depth_proxy_m", "first_last_span",
]


def load_pulse_ids(n_expected: int) -> np.ndarray:
    """Hash each waveform row; identical rows = echoes of the same pulse."""
    csv.field_size_limit(sys.maxsize)
    pulse_ids = np.empty(n_expected, dtype=np.int64)
    seen: dict[int, int] = {}
    with open(WAVEFORM_PATH, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for i, row in enumerate(reader):
            h = hash((row[1], row[2]))
            pulse_ids[i] = seen.setdefault(h, len(seen))
    if i + 1 != n_expected:
        raise ValueError(f"waveform rows {i + 1} != point rows {n_expected}")
    return pulse_ids


def echo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-point echo rank within its pulse (rank 1 = highest z = first return)."""
    grp = df.groupby("pulse_id")
    n_echoes = grp["z"].transform("size")
    rank = df.sort_values("z", ascending=False).groupby("pulse_id").cumcount() + 1
    rank = rank.reindex(df.index)
    out = pd.DataFrame(index=df.index)
    out["n_echoes"] = n_echoes
    out["echo_rank"] = rank
    out["echo_rank_norm"] = np.where(n_echoes > 1, (rank - 1) / (n_echoes - 1), 0.0)
    out["is_single_echo"] = (n_echoes == 1).astype(int)
    out["is_last_echo"] = ((rank == n_echoes) & (n_echoes > 1)).astype(int)
    out["is_first_multi"] = ((rank == 1) & (n_echoes > 1)).astype(int)
    out["is_intermediate"] = ((rank > 1) & (rank < n_echoes)).astype(int)
    return out


def _fill_holes(grid: np.ndarray) -> np.ndarray:
    """Fill NaN cells with the nearest valid cell value."""
    mask = np.isnan(grid)
    if mask.all():
        raise ValueError("grid has no valid cells")
    idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    return grid[tuple(idx)]


def _cell_aggregate(ix: np.ndarray, iy: np.ndarray, z: np.ndarray, shape: tuple[int, int],
                    how: str) -> np.ndarray:
    """Aggregate z per grid cell ('max' or robust low quantile)."""
    cell = ix * shape[1] + iy
    s = pd.Series(z).groupby(cell)
    agg = s.max() if how == "max" else s.quantile(DTM_PERCENTILE)
    grid = np.full(shape[0] * shape[1], np.nan)
    grid[agg.index.to_numpy()] = agg.to_numpy()
    return grid.reshape(shape)


def build_grids(df: pd.DataFrame, echo: pd.DataFrame,
                water_surface: np.ndarray) -> pd.DataFrame:
    """1 m DTM (last/single echoes, robust low) and DSM (all points, max)."""
    x0, y0 = df["x"].min(), df["y"].min()
    ix = ((df["x"] - x0) / CELL_M).astype(int).to_numpy()
    iy = ((df["y"] - y0) / CELL_M).astype(int).to_numpy()
    shape = (ix.max() + 1, iy.max() + 1)
    z = df["z"].to_numpy()

    ground = (echo["is_single_echo"] | echo["is_last_echo"]).to_numpy(bool)
    dtm = _cell_aggregate(ix[ground], iy[ground], z[ground], shape, "low")
    dtm = ndimage.median_filter(_fill_holes(dtm), size=3)
    dtm = ndimage.gaussian_filter(dtm, sigma=1.0)

    dsm = _fill_holes(_cell_aggregate(ix, iy, z, shape, "max"))
    dtm = np.minimum(dtm, dsm)

    out = pd.DataFrame(index=df.index)
    out["height_above_dtm"] = z - dtm[ix, iy]
    out["depth_below_dsm"] = dsm[ix, iy] - z
    # DTM under water = riverbed, so water depth would masquerade as vegetation
    # height — raise the reference to the water surface inside the footprint.
    out["height_above_ref"] = z - np.maximum(dtm[ix, iy], water_surface)
    return out


def load_water_surface() -> np.ndarray:
    """Per-point water surface elevation from the v10 local surface model.

    Used everywhere, not only inside the footprint: the footprint erodes 1 m
    inward and misses edge water, while the surface (259.85-260.75 m) sits far
    below any canopy, so raising the reference on land is harmless.
    """
    v10 = pd.read_csv(V10_PATH, usecols=["local_surface_z"])
    return v10["local_surface_z"].to_numpy()


def structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cylinder/sphere neighborhood stats: echo ratio, cover above, z spread."""
    xy = df[["x", "y"]].to_numpy()
    xyz = df[["x", "y", "z"]].to_numpy()
    z = xyz[:, 2]
    tree2d, tree3d = cKDTree(xy), cKDTree(xyz)
    n_3d = tree3d.query_ball_point(xyz, R_LOCAL_M, workers=-1, return_length=True)

    n = len(df)
    n_2d = np.zeros(n, dtype=int)
    n_above = np.zeros(n)
    z_range = np.zeros(n)
    z_std = np.zeros(n)
    for start in range(0, n, CHUNK):
        sl = slice(start, min(start + CHUNK, n))
        neigh = tree2d.query_ball_point(xy[sl], R_LOCAL_M, workers=-1)
        counts = np.fromiter((len(a) for a in neigh), int, sl.stop - start)
        n_2d[sl] = counts
        flat = np.concatenate(neigh).astype(int)
        owner = np.repeat(np.arange(start, sl.stop), counts)
        zn = z[flat]
        n_above[sl] = np.bincount(owner - start, zn > z[owner] + ABOVE_GAP_M,
                                  minlength=sl.stop - start)
        zsum = np.bincount(owner - start, zn, minlength=sl.stop - start)
        zsq = np.bincount(owner - start, zn ** 2, minlength=sl.stop - start)
        zmax = np.full(sl.stop - start, -np.inf)
        zmin = np.full(sl.stop - start, np.inf)
        np.maximum.at(zmax, owner - start, zn)
        np.minimum.at(zmin, owner - start, zn)
        z_range[sl] = zmax - zmin
        mean = zsum / counts
        z_std[sl] = np.sqrt(np.maximum(zsq / counts - mean ** 2, 0.0))

    out = pd.DataFrame(index=df.index)
    out["n_cyl"] = n_2d
    out["echo_ratio"] = n_3d / np.maximum(n_2d, 1)
    out["n_above_2m"] = n_above
    out["z_range_cyl"] = z_range
    out["z_std_cyl"] = z_std
    return out


def main() -> None:
    if OUT_PATH.exists() and "--force" not in sys.argv:
        print(f"{OUT_PATH} exists — skip (use --force to rebuild)")
        return

    df = pd.read_csv(POINT_CLOUD_PATH, index_col=0)
    df = df.rename(columns={"_riegl.reflectance": "reflectance_dB"})
    print(f"points: {len(df)}")

    df["pulse_id"] = load_pulse_ids(len(df))
    n_multi = (df.groupby("pulse_id")["z"].transform("size") > 1).sum()
    print(f"pulses: {df['pulse_id'].nunique()}, points in multi-echo pulses: {n_multi}")

    echo = echo_features(df)
    grids = build_grids(df, echo, load_water_surface())
    struct = structure_features(df)

    existing = pd.read_csv(FEATURES_IN_PATH, usecols=["x", *EXISTING_COLS])
    if not np.allclose(existing["x"].to_numpy(), df["x"].to_numpy()):
        raise ValueError("features_current.csv row order mismatch with point cloud")
    existing = existing.drop(columns=["x"]).set_index(df.index)

    out = pd.concat([df[["x", "y", "z"]], echo, grids, struct, existing], axis=1)
    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({out.shape[0]} rows, {out.shape[1]} cols)")


if __name__ == "__main__":
    main()
