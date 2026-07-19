"""Water-surface geometry stage: tight river footprint (Phase 1), local
adaptive surface grid (Phase 2), classification (Phase 3), waterbed
reconstruction (Phase 3b). Ports water_surface_model.py's geometry phases.

``classify()`` always runs this in "geometry-only" mode (``geometry_only=True``),
using externally supplied xgb_proba/deep_proba (WCN v9's outputs) as anchors
— this is the faithful, verified path. The original script's Phase 4
(XGBoost/V8Net retraining on these geometry labels) produced a model that
was only ever used to bootstrap WCN v9's own training labels; since
``fit()`` can source that same bootstrap directly from this stage's
``reconstructed_label`` output plus the autolabel stage's probas, Phase 4
retraining is intentionally not ported here (see MIGRATION.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.interpolate import griddata
from scipy.spatial import KDTree
from shapely import MultiPoint, concave_hull, contains_xy
from shapely.geometry import MultiPolygon
from sklearn.linear_model import LinearRegression, RANSACRegressor

from ..config import SurfaceConfig
from ..types import PipelineState

LABEL_LAND, LABEL_WATER, LABEL_UNCERTAIN = 0, 1, 2

# Feature columns build_surface_grid / bed reconstruction read directly.
_RANSAC_CANDIDATE_CONF_MIN = 0.7
_RANSAC_Z_LO, _RANSAC_Z_HI = 259.4, 260.2
_RANSAC_N_PEAKS_MAX = 3
_RANSAC_REFL_MAX_DB = -10.0


def _ensemble_from_probas(xgb_proba: np.ndarray, deep_proba: np.ndarray) -> np.ndarray:
    """0=land, 1=water, 2=uncertain (models disagree at the 0.5 threshold)."""
    xgb_pred = (xgb_proba >= 0.5).astype(np.int8)
    deep_pred = (deep_proba >= 0.5).astype(np.int8)
    ensemble = xgb_pred.copy()
    ensemble[xgb_pred != deep_pred] = LABEL_UNCERTAIN
    return ensemble


def build_tight_footprint(feat_df: pd.DataFrame, xgb_proba: np.ndarray, deep_proba: np.ndarray,
                          config: SurfaceConfig, *, geometry_only: bool,
                          ensemble: np.ndarray | None = None):
    """Concave hull of high-confidence water detections, eroded inward.

    Two-tier anchors: tier 1 = certain riverbed hits (z < riverbed_z_max),
    tier 2 = surface returns in shallow/bend sections the laser doesn't
    penetrate to the bed, filtered to within tier2_max_dist_from_tier1_m of
    a tier-1 anchor so isolated wet meadows/puddles can't spike the hull.
    """
    fp = config.footprint
    z = feat_df["z"].values
    mean_conf = (xgb_proba + deep_proba) * 0.5

    if geometry_only:
        tier1 = (deep_proba >= fp.conf) & (z < fp.riverbed_z_max)
        tier2 = (mean_conf >= fp.conf_surface) & (z < fp.riverbed_z_surface_max)
    else:
        if ensemble is None:
            raise ValueError("ensemble is required when geometry_only=False")
        tier1 = (ensemble == LABEL_WATER) & (mean_conf >= fp.conf) & (z < fp.riverbed_z_max)
        tier2 = (ensemble == LABEL_WATER) & (mean_conf >= fp.conf_surface) & (z < fp.riverbed_z_surface_max)

    tier2_only = tier2 & ~tier1
    if tier1.sum() > 0 and tier2_only.sum() > 0:
        xy1 = np.column_stack([feat_df["x"].values[tier1], feat_df["y"].values[tier1]])
        xy2 = np.column_stack([feat_df["x"].values[tier2_only], feat_df["y"].values[tier2_only]])
        dists, _ = KDTree(xy1).query(xy2)
        near = dists <= fp.tier2_max_dist_from_tier1_m
        t2_idx = np.where(tier2_only)[0]
        filtered_tier2 = np.zeros(len(feat_df), dtype=bool)
        filtered_tier2[t2_idx[near]] = True
        anchor = tier1 | filtered_tier2
    else:
        anchor = tier1 | tier2_only

    xw = feat_df["x"].values[anchor]
    yw = feat_df["y"].values[anchor]
    raw_hull = concave_hull(MultiPoint(np.column_stack([xw, yw])), ratio=fp.hull_ratio)

    if isinstance(raw_hull, MultiPolygon):
        raw_hull = raw_hull.buffer(0)   # repair topology; keeps all parts (river bend islands)

    footprint = raw_hull.buffer(-fp.erosion_m)
    if footprint.is_empty:
        footprint = raw_hull
    footprint = footprint.buffer(0)

    tier1_xy = np.column_stack([feat_df["x"].values[tier1], feat_df["y"].values[tier1]])
    return footprint, raw_hull, tier1_xy


def build_surface_grid(feat_df: pd.DataFrame, xgb_proba: np.ndarray, deep_proba: np.ndarray,
                       config: SurfaceConfig, *, geometry_only: bool,
                       tier1_xy: np.ndarray | None = None,
                       ensemble: np.ndarray | None = None):
    """Local adaptive water-surface elevation grid.

    Per cell: p95 z of water-like waveforms within max_dist_from_tier1_m of a
    tier-1 anchor, falling back to a global RANSAC plane for empty/distant
    cells, Gaussian-smoothed with an upper cap on fallback cells only.

    Returns grid_z, x_min, y_min, n_x, n_y, xi, yi, plane_coef, grid_z_pre_cap.
    """
    sg = config.surface_grid
    x_all, y_all, z_all = feat_df["x"].values, feat_df["y"].values, feat_df["z"].values

    if geometry_only:
        conf = deep_proba >= _RANSAC_CANDIDATE_CONF_MIN
    else:
        if ensemble is None:
            raise ValueError("ensemble is required when geometry_only=False")
        mean_conf = (xgb_proba + deep_proba) * 0.5
        conf = (ensemble == LABEL_WATER) & (mean_conf >= _RANSAC_CANDIDATE_CONF_MIN)

    surf_cand = (conf & (z_all >= _RANSAC_Z_LO) & (z_all <= _RANSAC_Z_HI)
                & (feat_df["n_peaks"].values <= _RANSAC_N_PEAKS_MAX)
                & (feat_df["energy_concentration"].values > sg.energy_concentration_min)
                & (feat_df["reflectance_dB"].values < _RANSAC_REFL_MAX_DB))

    ransac = RANSACRegressor(
        estimator=LinearRegression(), residual_threshold=sg.ransac_residual_m,
        min_samples=max(0.5, 100 / max(surf_cand.sum(), 1)), random_state=42,
    )
    ransac.fit(np.column_stack([x_all[surf_cand], y_all[surf_cand]]), z_all[surf_cand])
    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_
    plane_coef = (float(a), float(b), float(c))

    x_min, y_min = float(x_all.min()), float(y_all.min())
    cell = sg.cell_size_m
    n_x = int(np.ceil((x_all.max() - x_min) / cell)) + 1
    n_y = int(np.ceil((y_all.max() - y_min) / cell)) + 1
    xi = np.clip(np.floor((x_all - x_min) / cell).astype(int), 0, n_x - 1)
    yi = np.clip(np.floor((y_all - y_min) / cell).astype(int), 0, n_y - 1)
    flat_id = yi * n_x + xi

    wl_surf = ((feat_df["n_peaks"].values <= sg.n_peaks_max)
              & (feat_df["energy_concentration"].values > sg.energy_concentration_min)
              & (feat_df["reflectance_dB"].values < sg.reflectance_max_db)
              & (z_all >= sg.z_lo) & (z_all <= sg.z_hi))
    df_wl = pd.DataFrame({"flat": flat_id[wl_surf], "z": z_all[wl_surf]})
    counts = df_wl.groupby("flat")["z"].count()
    valid = counts[counts >= sg.min_pts_per_cell].index
    surf_primary = (df_wl[df_wl["flat"].isin(valid)].groupby("flat")["z"]
                    .quantile(0.95).clip(upper=sg.z_cap))

    yg, xg = np.mgrid[0:n_y, 0:n_x]
    xc = x_min + (xg + 0.5) * cell
    yc = y_min + (yg + 0.5) * cell
    grid_z = (a * xc + b * yc + c).astype(np.float32)   # baseline = global plane

    t1_tree = KDTree(tier1_xy) if tier1_xy is not None and len(tier1_xy) > 0 else None
    primary_cell_mask = np.zeros((n_y, n_x), dtype=bool)
    for flat_idx, sz in surf_primary.items():
        iy, ix = int(flat_idx) // n_x, int(flat_idx) % n_x
        if not (0 <= iy < n_y and 0 <= ix < n_x):
            continue
        if t1_tree is not None:
            xc_cell = x_min + (ix + 0.5) * cell
            yc_cell = y_min + (iy + 0.5) * cell
            dist, _ = t1_tree.query([[xc_cell, yc_cell]])
            if dist[0] > sg.max_dist_from_tier1_m:
                continue
        grid_z[iy, ix] = sz
        primary_cell_mask[iy, ix] = True

    grid_z_smooth = gaussian_filter(grid_z, sigma=sg.smooth_sigma_cells).astype(np.float32)

    # Cap RANSAC-fallback cells only — Gaussian bleed from primary cells can
    # raise neighbouring fallback cells above the true surface.
    ransac_grid = (a * xc + b * yc + c).astype(np.float32)
    cap_mask = ~primary_cell_mask
    grid_z_pre_cap = grid_z_smooth.copy()
    grid_z_smooth = np.where(
        cap_mask, np.minimum(grid_z_smooth, ransac_grid + sg.ransac_max_rise_m), grid_z_smooth)

    return grid_z_smooth, x_min, y_min, n_x, n_y, xi, yi, plane_coef, grid_z_pre_cap


def classify_points(feat_df: pd.DataFrame, in_footprint: np.ndarray, local_surface_z: np.ndarray,
                    wf_ensemble: np.ndarray, config: SurfaceConfig) -> np.ndarray:
    """Merge geometry (footprint + local surface) with the ML ensemble label.

    Inside the footprint, submerged points trust geometry unless the
    waveform model is confident it's land; near-surface points (within
    water_tol_m of the estimated surface) require ML agreement, since a
    flowing river's surface estimate can be off by 0.1-0.2 m. Outside the
    footprint, the waveform model dominates and "water" degrades to
    "uncertain" (shallow margin, might still be water).
    """
    z = feat_df["z"].values
    z_diff = z - local_surface_z
    merged = np.full(len(z), LABEL_UNCERTAIN, dtype=np.int8)

    submerged = in_footprint & (z_diff < 0.0)
    near_surface = in_footprint & (z_diff >= 0.0) & (z_diff <= config.water_tol_m)
    above_surf = in_footprint & (z_diff > config.water_tol_m)

    merged[submerged & (wf_ensemble != LABEL_LAND)] = LABEL_WATER
    merged[submerged & (wf_ensemble == LABEL_LAND)] = LABEL_LAND
    merged[near_surface & (wf_ensemble == LABEL_WATER)] = LABEL_WATER
    merged[near_surface & (wf_ensemble == LABEL_LAND)] = LABEL_LAND
    merged[near_surface & (wf_ensemble == LABEL_UNCERTAIN)] = LABEL_UNCERTAIN
    merged[above_surf] = LABEL_LAND

    outside = ~in_footprint
    merged[outside & (wf_ensemble == LABEL_LAND)] = LABEL_LAND
    merged[outside & (wf_ensemble == LABEL_WATER)] = LABEL_UNCERTAIN
    merged[outside & (wf_ensemble == LABEL_UNCERTAIN)] = LABEL_UNCERTAIN
    return merged


def build_riverbed_grid(feat_df: pd.DataFrame, merged_label: np.ndarray, deep_proba: np.ndarray,
                        x_min: float, y_min: float, n_x: int, n_y: int,
                        xi: np.ndarray, yi: np.ndarray, config: SurfaceConfig):
    """Reconstruct riverbed elevation from confirmed deep-water returns
    (5th-percentile z per cell = deepest confirmed return), nearest-neighbour
    filled where no confirmed data exists, then Gaussian-smoothed."""
    bed = config.bed
    z = feat_df["z"].values
    water_conf = (merged_label == LABEL_WATER) | (deep_proba >= bed.proba_min)
    bed_mask = water_conf & (z < config.footprint.riverbed_z_max)

    bed_coverage = np.zeros((n_y, n_x), dtype=bool)
    if bed_mask.sum() == 0:
        return np.zeros((n_y, n_x), dtype=np.float32), bed_coverage

    flat_ids = yi[bed_mask] * n_x + xi[bed_mask]
    df_bed = pd.DataFrame({"flat": flat_ids, "z": z[bed_mask]})
    counts = df_bed.groupby("flat")["z"].count()
    valid = counts[counts >= bed.min_pts_per_cell].index
    bed_primary = df_bed[df_bed["flat"].isin(valid)].groupby("flat")["z"].quantile(0.05)

    bed_grid = np.full((n_y, n_x), np.nan, dtype=np.float32)
    for flat_idx, bz in bed_primary.items():
        iy, ix = int(flat_idx) // n_x, int(flat_idx) % n_x
        if 0 <= iy < n_y and 0 <= ix < n_x:
            bed_grid[iy, ix] = float(bz)
            bed_coverage[iy, ix] = True

    known_y, known_x = np.where(bed_coverage)
    if len(known_y) > 0:
        all_y, all_x = np.mgrid[0:n_y, 0:n_x]
        filled = griddata(
            np.column_stack([known_y, known_x]), bed_grid[known_y, known_x],
            np.column_stack([all_y.ravel(), all_x.ravel()]), method="nearest",
        ).reshape(n_y, n_x).astype(np.float32)
        bed_grid = np.where(np.isnan(bed_grid), filled, bed_grid)

    bed_grid = gaussian_filter(bed_grid, sigma=config.surface_grid.smooth_sigma_cells).astype(np.float32)
    return bed_grid, bed_coverage


def apply_waterbed_reconstruction(feat_df: pd.DataFrame, merged_label: np.ndarray,
                                  local_surface_z: np.ndarray, bed_grid: np.ndarray,
                                  bed_coverage: np.ndarray, xi: np.ndarray, yi: np.ndarray,
                                  config: SurfaceConfig) -> np.ndarray:
    """Recover tree-over-water points a footprint gap or waveform confusion
    left unlabeled: any non-water point geometrically inside the water
    corridor near confirmed bed data, with water-like reflectance and a
    complex (canopy-confused) waveform, becomes reconstructed water (3)."""
    bed = config.bed
    z = feat_df["z"].values
    reflectance = feat_df["reflectance_dB"].values
    n_peaks = feat_df["n_peaks"].values

    struct_r = max(1, int(np.ceil(bed.max_dist_m / config.surface_grid.cell_size_m)))
    struct = np.ones((2 * struct_r + 1, 2 * struct_r + 1), dtype=bool)
    near_bed = binary_dilation(bed_coverage, structure=struct)[yi, xi]

    local_bed = bed_grid[yi, xi]
    below_surface = z <= local_surface_z + config.water_tol_m
    above_bed = z >= local_bed - bed.margin_m
    not_water = merged_label != LABEL_WATER
    water_like_refl = reflectance < bed.reflectance_max_db
    waveform_confused = n_peaks >= bed.min_peaks
    planar_surface = (
        feat_df["planarity"].values > bed.planarity_min if "planarity" in feat_df.columns
        else np.ones(len(feat_df), dtype=bool)
    )

    reconstructed = (not_water & below_surface & near_bed & above_bed
                     & water_like_refl & waveform_confused & planar_surface)

    new_labels = merged_label.copy()
    new_labels[reconstructed] = bed.recon_label
    return new_labels


def run(state: PipelineState, config: SurfaceConfig, *, geometry_only: bool = True) -> PipelineState:
    """Run Phases 1-3b using state.wcn_xgb_proba / state.wcn_proba as anchors
    (geometry_only=True, the classify() path) or state.autolabel_* +
    state.autolabel_ensemble (geometry_only=False, used within fit() before
    Phase 4 retraining)."""
    if state.features is None:
        raise ValueError("surface stage needs state.features — run the features stage first")

    if geometry_only:
        xgb_proba = state.wcn_xgb_proba
        deep_proba = state.wcn_proba
        if xgb_proba is None or deep_proba is None:
            raise ValueError("geometry_only=True needs state.wcn_xgb_proba and state.wcn_proba")
        ensemble = _ensemble_from_probas(xgb_proba, deep_proba)
    else:
        if (state.autolabel_xgb_proba is None or state.autolabel_deep_proba is None
                or state.autolabel_ensemble is None):
            raise ValueError(
                "geometry_only=False needs state.autolabel_xgb_proba/deep_proba/ensemble")
        xgb_proba = state.autolabel_xgb_proba
        deep_proba = state.autolabel_deep_proba
        ensemble = state.autolabel_ensemble

    feat_df = state.features

    footprint, raw_hull, tier1_xy = build_tight_footprint(
        feat_df, xgb_proba, deep_proba, config, geometry_only=geometry_only, ensemble=ensemble)
    in_footprint = contains_xy(footprint, feat_df["x"].values, feat_df["y"].values)

    grid_z, x_min, y_min, n_x, n_y, xi, yi, plane_coef, _ = build_surface_grid(
        feat_df, xgb_proba, deep_proba, config, geometry_only=geometry_only,
        tier1_xy=tier1_xy, ensemble=ensemble)
    local_surface_z = grid_z[yi, xi].astype(np.float32)

    merged_label = classify_points(feat_df, in_footprint, local_surface_z, ensemble, config)

    bed_grid, bed_coverage = build_riverbed_grid(
        feat_df, merged_label, deep_proba, x_min, y_min, n_x, n_y, xi, yi, config)
    reconstructed_label = apply_waterbed_reconstruction(
        feat_df, merged_label, local_surface_z, bed_grid, bed_coverage, xi, yi, config)

    state.footprint_geom = footprint
    state.footprint_raw_hull = raw_hull
    state.in_footprint = in_footprint
    state.local_surface_z = local_surface_z
    state.surface_grid = grid_z
    state.surface_grid_origin = (x_min, y_min)
    state.surface_plane_coef = plane_coef
    state.merged_label = merged_label
    state.reconstructed_label = reconstructed_label
    state.metrics["geometry"] = {
        "footprint_area_m2": round(float(footprint.area), 1),
        "points_inside_footprint": int(in_footprint.sum()),
        "label_counts": {
            "land": int((reconstructed_label == LABEL_LAND).sum()),
            "water": int((reconstructed_label == LABEL_WATER).sum()),
            "uncertain": int((reconstructed_label == LABEL_UNCERTAIN).sum()),
            "reconstructed_water": int((reconstructed_label == config.bed.recon_label).sum()),
        },
    }
    return state
