"""Feature extraction stage: waveform scalar features + dense grids, k-NN
geometric features, and the generalizable ratios/heights derived from them.

Ports feature_extractor.py + add_features.py + the WCN-specific derived
columns from preprocess_wcn.py into one stage, operating on a PointCloud /
PipelineState instead of fixed CSV paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter, minimum_filter
from sklearn.neighbors import KDTree

from ..config import FeatureConfig
from ..types import PipelineState, PointCloud

# Columns emitted with a value of 0 when a waveform is malformed (empty or
# times/amps length mismatch) — keeps the row count aligned with the point cloud.
_ZERO_FALLBACK_COLS = [
    "max_amp", "mean_amp", "std_amp", "total_energy", "n_samples", "time_span",
    "n_gaps", "max_gap", "mean_gap", "total_gap", "n_peaks", "max_peak_spacing",
    "mean_peak_spacing", "first_last_span", "n_clusters", "energy_ratio_late",
    "first_peak_amp", "last_peak_amp", "peak_amp_ratio", "depth_proxy_m",
]

_DEPTH_PROXY_M_PER_SI = 0.05625   # c_water / 2, in SI sample units -> metres


def _extract_waveform_features(times: np.ndarray, amps: np.ndarray, config: FeatureConfig) -> dict:
    f: dict = {}
    f["max_amp"] = int(np.max(amps))
    f["mean_amp"] = float(np.mean(amps))
    f["std_amp"] = float(np.std(amps))
    f["total_energy"] = int(np.sum(amps))
    f["n_samples"] = len(times)
    f["time_span"] = int(times[-1] - times[0]) if len(times) > 1 else 0

    if len(times) > 1:
        diffs = np.diff(times.astype(np.int32))
        gaps = diffs[diffs > config.gap_thresh_si]
        f["n_gaps"] = int(np.sum(diffs > config.gap_thresh_si))
        f["max_gap"] = int(np.max(diffs))
        f["mean_gap"] = float(np.mean(gaps)) if len(gaps) > 0 else 0.0
        f["total_gap"] = int(np.sum(gaps))
    else:
        f["n_gaps"] = f["max_gap"] = f["total_gap"] = 0
        f["mean_gap"] = 0.0

    peaks = [
        i for i in range(1, len(amps) - 1)
        if amps[i] > amps[i - 1] and amps[i] > amps[i + 1] and amps[i] >= config.min_peak_adc
    ]
    f["n_peaks"] = len(peaks)

    if len(peaks) >= 2:
        pt = times[peaks]
        spacing = np.diff(pt.astype(np.int32))
        f["max_peak_spacing"] = int(np.max(spacing))
        f["mean_peak_spacing"] = float(np.mean(spacing))
        f["first_last_span"] = int(pt[-1] - pt[0])
    else:
        f["max_peak_spacing"] = f["mean_peak_spacing"] = f["first_last_span"] = 0

    f["n_clusters"] = 1 + f["n_gaps"]

    if f["time_span"] > 0:
        cutoff = times[0] + 0.6 * f["time_span"]
        e_late = int(np.sum(amps[times > cutoff]))
        f["energy_ratio_late"] = float(e_late / (f["total_energy"] + 1))
    else:
        f["energy_ratio_late"] = 0.0

    if peaks:
        f["first_peak_amp"] = int(amps[peaks[0]])
        f["last_peak_amp"] = int(amps[peaks[-1]])
    else:
        f["first_peak_amp"] = f["last_peak_amp"] = 0

    f["peak_amp_ratio"] = (
        float(f["first_peak_amp"] / f["last_peak_amp"]) if f["last_peak_amp"] > 0 else 0.0
    )
    f["depth_proxy_m"] = round(f["max_peak_spacing"] * _DEPTH_PROXY_M_PER_SI, 3)
    return f


def _waveform_to_grid(times: np.ndarray, amps: np.ndarray, grid_size: int) -> np.ndarray:
    """Project a non-contiguous waveform onto a fixed-length dense grid,
    origin-relative (first sample time maps to bin 0)."""
    grid = np.zeros(grid_size, dtype=np.float32)
    t_min = int(times[0])
    idx = times.astype(np.int64) - t_min
    valid = (idx >= 0) & (idx < grid_size)
    grid[idx[valid]] = amps[valid].astype(np.float32)
    return grid


def _extract_all_waveforms(cloud: PointCloud, config: FeatureConfig) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-point waveform scalar features + dense grids.

    Waveforms are ragged (variable samples per point) so peak/gap detection
    cannot be vectorized across points without padding to the densest
    waveform; iterates per point like the original feature_extractor.py.
    """
    n = len(cloud)
    grids = np.zeros((n, config.grid_size), dtype=np.float32)
    records: list[dict] = []
    for i, (times, amps) in enumerate(cloud.iter_waveforms()):
        if len(times) == 0 or len(times) != len(amps):
            records.append({k: 0 for k in _ZERO_FALLBACK_COLS})
            continue
        records.append(_extract_waveform_features(times, amps, config))
        grids[i] = _waveform_to_grid(times, amps, config.grid_size)
    return pd.DataFrame.from_records(records), grids


def _compute_geometric_features(xyz: np.ndarray, k: int) -> pd.DataFrame:
    """Vectorized k-NN PCA geometric features (planarity, roughness, ...)."""
    tree = KDTree(xyz)
    _, indices = tree.query(xyz, k=k + 1)
    indices = indices[:, 1:]   # drop self

    nbrs = xyz[indices]                              # (N, k, 3)
    centered = nbrs - nbrs.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / k

    eigvals = np.linalg.eigvalsh(cov)                 # ascending: (N, 3)
    l1, l2, l3 = eigvals[:, 0], eigvals[:, 1], eigvals[:, 2]
    denom = l3 + 1e-8

    z_nbrs = nbrs[:, :, 2]
    return pd.DataFrame({
        "planarity": ((l2 - l1) / denom).astype(np.float32),
        "linearity": ((l3 - l2) / denom).astype(np.float32),
        "sphericity": (l1 / denom).astype(np.float32),
        "roughness": l1.astype(np.float32),
        "height_range_local": (z_nbrs.max(axis=1) - z_nbrs.min(axis=1)).astype(np.float32),
        "height_std_local": z_nbrs.std(axis=1).astype(np.float32),
        "z_relative": (xyz[:, 2] - z_nbrs.mean(axis=1)).astype(np.float32),
    })


def _raster_grid(x: np.ndarray, y: np.ndarray, z: np.ndarray, cell: float, fill_max: bool):
    x_min, y_min = x.min(), y.min()
    nx = int((x.max() - x_min) / cell) + 2
    ny = int((y.max() - y_min) / cell) + 2
    ix = np.clip(((x - x_min) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((y - y_min) / cell).astype(int), 0, ny - 1)

    fill = np.inf if fill_max else -np.inf
    grid = np.full((ny, nx), fill, dtype=np.float32)
    if fill_max:
        np.minimum.at(grid, (iy, ix), z)
        grid[np.isinf(grid)] = float(z.max())
    else:
        np.maximum.at(grid, (iy, ix), z)
        grid[np.isinf(grid)] = float(z.min())
    return grid, ix, iy


def _local_min_height(x: np.ndarray, y: np.ndarray, z: np.ndarray, radius: float, cell: float) -> np.ndarray:
    grid, ix, iy = _raster_grid(x, y, z.astype(np.float32), cell, fill_max=True)
    k = int(np.ceil(radius / cell)) * 2 + 1
    local_min = minimum_filter(grid, size=k, mode="nearest")
    return z.astype(np.float32) - local_min[iy, ix]


def _local_normalized_rank(x: np.ndarray, y: np.ndarray, z: np.ndarray, radius: float, cell: float) -> np.ndarray:
    x_min, y_min = x.min(), y.min()
    nx = int((x.max() - x_min) / cell) + 2
    ny = int((y.max() - y_min) / cell) + 2
    ix = np.clip(((x - x_min) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((y - y_min) / cell).astype(int), 0, ny - 1)

    zf = z.astype(np.float32)
    gmin = np.full((ny, nx), np.inf, dtype=np.float32)
    gmax = np.full((ny, nx), -np.inf, dtype=np.float32)
    np.minimum.at(gmin, (iy, ix), zf)
    np.maximum.at(gmax, (iy, ix), zf)
    gmin = np.where(np.isinf(gmin), zf.max(), gmin)
    gmax = np.where(np.isinf(gmax), zf.min(), gmax)

    k = int(np.ceil(radius / cell)) * 2 + 1
    lmin = minimum_filter(gmin, size=k, mode="nearest")[iy, ix]
    lmax = maximum_filter(gmax, size=k, mode="nearest")[iy, ix]
    rng = lmax - lmin
    return np.where(rng > 0.01, (zf - lmin) / rng, 0.5)


def _waveform_shape_features(grids: np.ndarray) -> pd.DataFrame:
    g = grids.astype(np.float32)
    total_e = g.sum(axis=1)
    safe_e = np.where(total_e > 0, total_e, 1.0)
    bins = np.arange(g.shape[1], dtype=np.float32)

    return pd.DataFrame({
        "energy_concentration": (g[:, :30].sum(axis=1) / safe_e).astype(np.float32),
        "amplitude_weighted_center": ((g * bins).sum(axis=1) / safe_e).astype(np.float32),
        "active_bins_ratio": ((g > 0).sum(axis=1).astype(np.float32) / g.shape[1]),
        "max_amp_norm_by_energy": (g.max(axis=1) / safe_e).astype(np.float32),
    })


def normalize_grids(grids: np.ndarray) -> np.ndarray:
    """Per-sample max normalisation (WCN v9 input): each row divided by its
    own max amplitude, clipped to avoid dividing by zero on empty waveforms."""
    row_max = grids.max(axis=1, keepdims=True).clip(min=1.0)
    return (grids / row_max).astype(np.float32)


def run(cloud: PointCloud, config: FeatureConfig) -> PipelineState:
    """Extract all features and populate a fresh PipelineState."""
    wf_feat, grids = _extract_all_waveforms(cloud, config)
    geo_feat = _compute_geometric_features(cloud.xyz.astype(np.float64), k=config.knn_k)

    x, y, z = cloud.x.astype(np.float32), cloud.y.astype(np.float32), cloud.z.astype(np.float32)
    shape_feat = _waveform_shape_features(grids)

    features = pd.concat([
        pd.DataFrame({"x": x, "y": y, "z": z, "reflectance_dB": cloud.reflectance_db}),
        wf_feat.reset_index(drop=True),
        geo_feat.reset_index(drop=True),
        shape_feat.reset_index(drop=True),
    ], axis=1)

    features["height_above_local_min"] = _local_min_height(
        x, y, z, config.local_min_radius_m, config.raster_cell_m)
    features["height_above_local_min_10m"] = _local_min_height(
        x, y, z, config.local_min_radius_10m, config.raster_cell_m)
    features["height_percentile_local"] = _local_normalized_rank(
        x, y, z, config.local_rank_radius_m, config.raster_cell_m)

    # WCN v9's derived dimensionless ratios
    features["gap_ratio"] = (
        features["total_gap"] / features["time_span"].clip(lower=1.0)
    ).astype(np.float32)
    features["energy_center_norm"] = (
        features["amplitude_weighted_center"] / features["n_samples"].clip(lower=1.0)
    ).astype(np.float32)

    grids_norm = normalize_grids(grids)

    return PipelineState(
        cloud=cloud,
        features=features,
        waveform_grids=grids,
        waveform_grids_norm=grids_norm,
    )
