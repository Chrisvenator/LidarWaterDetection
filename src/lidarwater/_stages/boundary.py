"""River boundary stage: rasterize the water-probability field, smooth it,
and extract iso-probability contours at three thresholds (inner/center/
outer). Ports river_boundary.py's data path — plotting (boundary_heatmap.png,
boundary_nocanopy.png) is dropped; the geometry is available on
``state.boundary_contours`` and via ``lidarwater.io.write_geojson``.
"""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..config import BoundaryConfig
from ..types import PipelineState

LABEL_LAND, LABEL_WATER, LABEL_UNCERTAIN, LABEL_RECON_WATER, LABEL_CANOPY = 0, 1, 2, 3, 4


def rasterize(x: np.ndarray, y: np.ndarray, proba: np.ndarray, cell_size: float):
    """Bin proba values into a (n_y, n_x) grid; empty cells are NaN."""
    x_min, y_min = float(x.min()), float(y.min())
    n_x = int(np.ceil((x.max() - x_min) / cell_size)) + 1
    n_y = int(np.ceil((y.max() - y_min) / cell_size)) + 1

    xi = np.clip(np.floor((x - x_min) / cell_size).astype(int), 0, n_x - 1)
    yi = np.clip(np.floor((y - y_min) / cell_size).astype(int), 0, n_y - 1)

    grid_sum = np.zeros((n_y, n_x), dtype=np.float64)
    grid_cnt = np.zeros((n_y, n_x), dtype=np.int32)
    np.add.at(grid_sum, (yi, xi), proba)
    np.add.at(grid_cnt, (yi, xi), 1)

    valid = grid_cnt > 0
    grid = np.where(valid, grid_sum / np.maximum(grid_cnt, 1), np.nan)
    return grid, x_min, y_min, n_x, n_y


def fill_and_smooth(grid: np.ndarray, cell_size: float, smooth_sigma_m: float,
                    max_dist_m: float | None = None) -> np.ndarray:
    """Nearest-neighbour fill of NaN cells, then Gaussian smoothing. Cells
    farther than max_dist_m from any real data are reset to NaN afterward,
    so contours cannot wander into no-data margins."""
    nan_mask = np.isnan(grid)
    if nan_mask.any():
        dist, nearest = distance_transform_edt(nan_mask, return_indices=True)
        filled = grid.copy()
        filled[nan_mask] = grid[nearest[0][nan_mask], nearest[1][nan_mask]]
    else:
        dist = np.zeros_like(grid)
        filled = grid

    sigma_cells = smooth_sigma_m / cell_size
    smoothed = gaussian_filter(filled.astype(np.float32), sigma=sigma_cells)
    if max_dist_m is not None:
        far = dist * cell_size > max_dist_m
        smoothed[far] = np.nan
    return smoothed


def _chaikin(pts: np.ndarray, n: int = 3) -> np.ndarray:
    """Chaikin corner-cutting: smooths a polyline without shrinking it much."""
    closed = np.allclose(pts[0], pts[-1])
    for _ in range(n):
        new = []
        for i in range(len(pts) - 1):
            new.append(0.75 * pts[i] + 0.25 * pts[i + 1])
            new.append(0.25 * pts[i] + 0.75 * pts[i + 1])
        pts = np.array(new)
        if closed:
            pts = np.vstack([pts, pts[0]])
    return pts


def extract_contours(grid: np.ndarray, x_min: float, y_min: float,
                     config: BoundaryConfig) -> dict[float, list[np.ndarray]]:
    """Extract contour polylines at prob_outer/center/inner.

    Uses matplotlib's contouring algorithm purely as a numeric routine (no
    figure is shown or saved) — the only practical way to trace iso-lines
    from a smoothed probability grid without adding a separate contouring
    dependency.
    """
    levels = [config.prob_outer, config.prob_center, config.prob_inner]
    x_1d = x_min + (np.arange(grid.shape[1]) + 0.5) * config.cell_size_m
    y_1d = y_min + (np.arange(grid.shape[0]) + 0.5) * config.cell_size_m

    fig, ax = plt.subplots()
    cs = ax.contour(x_1d, y_1d, grid, levels=levels)
    plt.close(fig)

    result: dict[float, list[np.ndarray]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        all_segs = cs.allsegs

    for level, segs in zip(cs.levels, all_segs):
        long_segs = [s for s in segs if len(s) >= config.min_seg_len]
        long_segs.sort(key=len, reverse=True)
        result[float(level)] = [_chaikin(s) for s in long_segs[:config.max_contours]]
    return result


def _isolated_water(x: np.ndarray, y: np.ndarray, labels: np.ndarray, config: BoundaryConfig) -> np.ndarray:
    """Water points without enough water neighbors — reconstruction strays
    that would otherwise seed a phantom blob via nearest-neighbour fill."""
    water = np.isin(labels, (LABEL_WATER, LABEL_RECON_WATER))
    out = np.zeros(len(labels), dtype=bool)
    if not water.any():
        return out
    xy = np.column_stack([x[water], y[water]])
    support = cKDTree(xy).query_ball_point(
        xy, config.isolation_radius_m, workers=-1, return_length=True)
    out[np.flatnonzero(water)[support < config.min_water_support]] = True
    return out


def run(state: PipelineState, config: BoundaryConfig) -> PipelineState:
    """Compute river-boundary contours from the final (or v10) labels.

    Uses ``state.final_label`` if the merge stage has run (canopy-aware:
    canopy points excluded as evidence, isolated water strays dropped),
    otherwise falls back to ``state.merged_label`` + ``state.wcn_proba``
    (pre-canopy v10 evidence, z-based canopy exclusion).
    """
    x, y, z = state.cloud.x, state.cloud.y, state.cloud.z

    if state.final_label is not None:
        labels = state.final_label
        deep_proba = state.wcn_proba if state.wcn_proba is not None else np.zeros(len(labels))
        evidence = np.select(
            [np.isin(labels, (LABEL_WATER, LABEL_RECON_WATER)), labels == LABEL_LAND],
            [1.0, 0.0], default=deep_proba,
        )
        use = (labels != LABEL_CANOPY) & ~_isolated_water(x, y, labels, config)
        max_fill_dist_m = config.max_fill_dist_m
    else:
        if state.merged_label is None or state.wcn_proba is None:
            raise ValueError(
                "boundary stage needs state.final_label or "
                "(state.merged_label and state.wcn_proba) — run geometry (+ canopy/merge) first"
            )
        labels = state.merged_label
        evidence = state.wcn_proba
        use = z <= config.canopy_z_max
        max_fill_dist_m = None

    if not use.any():
        raise ValueError("no evidence points left after canopy/stray filtering")

    grid_raw, x_min, y_min, _, _ = rasterize(x[use], y[use], evidence[use], config.cell_size_m)
    grid_smooth = fill_and_smooth(grid_raw, config.cell_size_m, config.smooth_sigma_m,
                                  max_dist_m=max_fill_dist_m)
    contours = extract_contours(grid_smooth, x_min, y_min, config)

    state.boundary_contours = contours
    state.metrics["boundary"] = {
        "n_evidence_points": int(use.sum()),
        "levels": {name: len(segs) for name, segs in
                   zip(("outer", "center", "inner"),
                       (contours.get(config.prob_outer, []),
                        contours.get(config.prob_center, []),
                        contours.get(config.prob_inner, [])))},
    }
    return state
