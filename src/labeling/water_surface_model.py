"""
water_surface_model_v2.py — Tight footprint + local adaptive surface grid (v8).

Fixes four problems from v7:
  1. Footprint too large  → tighter concave hull from high-conf (>0.8) RIVERBED
     points only (z < 259.6m), eroded inward by 0.5m.
  2. One plane can't fit a bumpy/tilted surface  → 2m-cell local surface grid,
     estimated from water-like waveforms per cell, fallback to RANSAC global
     plane, Gaussian-smoothed.
  3. Turbid/deep sections misclassified as land  → inside footprint +
     z ≤ local_surface_z + 0.3m → WATER, waveform overridden by geometry.
  4. Shallow rocky edges  → inside footprint + z > local_surface_z + 0.3m →
     LAND (exposed rock); ambiguous margin handled by outside-footprint merge.

Inputs:
  pointclouds/labeled_pointcloud_v6_waveform_only.csv
  data_processed/features_v2.csv
  data_processed/waveform_grids.npy

Outputs:
  models/v8-surface-v2/
    v8_xgb.json, v8_deep.pt, v8_deep_stats.json, v8_metrics.json
    topdown_scatter.png, crosssection.png, surface_grid.png,
    feature_importance.png, training_curve.png
  pointclouds/labeled_pointcloud_v8.csv

Merged label values:
  0 = land
  1 = water
  2 = uncertain (margin cases; outside footprint when waveform says water)
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from scipy.spatial import KDTree
from shapely import concave_hull, MultiPoint, contains_xy
from shapely.geometry import MultiPolygon, Polygon
from sklearn.linear_model import RANSACRegressor, LinearRegression
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score, classification_report

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from labeling.river_boundary import (  # noqa: E402
    CANOPY_Z_MAX, PROB_INNER, PROB_CENTER, PROB_OUTER,
    rasterize, fill_and_smooth, extract_contours, _draw_contours,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
V6_WF_CSV  = ROOT / "pointclouds"    / "labeled_pointcloud_waveform_only.csv"
FEAT_PATH  = ROOT / "data_processed" / "features_current.csv"
GRIDS_PATH = ROOT / "data_processed" / "waveform_grids.npy"
MODEL_DIR  = ROOT / "models"         / "current"
OUT_PATH   = ROOT / "pointclouds"    / "labeled_pointcloud_current.csv"

# ── Footprint parameters ───────────────────────────────────────────────────────
FOOTPRINT_CONF         = 0.8    # minimum mean(xgb_proba, deep_proba) for tier-1 anchor
FOOTPRINT_CONF_SURFACE = 0.85   # higher threshold for tier-2 surface anchors
RIVERBED_Z_MAX         = 259.6  # z < this = underwater / riverbed (v6 zone boundary)
RIVERBED_Z_SURFACE_MAX = 261.5  # z < this for tier-2 surface water anchors
HULL_RATIO             = 0.2    # concave_hull tightness  (lower = tighter)
FOOTPRINT_EROSION      = 0.5    # erode inward by this many metres (conservative)
TIER2_MAX_DIST_FROM_T1 = 10.0   # max metres a tier-2 anchor may be from any tier-1 anchor

# ── Local surface grid ─────────────────────────────────────────────────────────
CELL_SIZE         = 2.0    # metres per grid cell
SURF_Z_LO         = 259.0  # z range for surface-candidate waveforms
SURF_Z_HI         = 261.5
SURF_N_PEAKS_MAX  = 2      # simple / single-return waveforms
SURF_EC_MIN       = 0.85   # energy concentration (compact pulse = water)
SURF_REFL_MAX     = -15.0  # weak reflectance = water surface at 532 nm
SURF_Z_CAP        = 261.0  # cap estimated surface to avoid outlier cells
SURF_MIN_PTS      = 5      # min water-like pts per cell for primary estimate
SMOOTH_SIGMA      = 1.0    # Gaussian kernel width in cell units
SURF_MAX_T1_DIST  = 12.0   # surface grid cells >this many metres from any tier-1 anchor use RANSAC fallback

# ── Classification thresholds ──────────────────────────────────────────────────
WATER_TOL = 0.30   # inside footprint, z ≤ surface + this → WATER

# ── RANSAC (global fallback plane) ────────────────────────────────────────────
RANSAC_RESIDUAL   = 0.20

# ── Feature sets ──────────────────────────────────────────────────────────────
WAVEFORM_FEATURES = [
    "energy_concentration", "max_amp_norm_by_energy",
    "n_clusters", "n_peaks", "n_gaps", "n_samples", "time_span",
    "max_amp", "mean_amp", "std_amp", "total_energy",
    "max_gap", "mean_gap", "total_gap", "first_last_span",
    "energy_ratio_late", "first_peak_amp", "last_peak_amp", "peak_amp_ratio",
    "depth_proxy_m", "amplitude_weighted_center", "active_bins_ratio",
    "reflectance_dB",
]
RELATIVE_FEATURES = [
    "height_above_local_min", "height_percentile_local",
    "planarity", "roughness", "linearity", "sphericity",
    "height_range_local", "height_std_local", "z_relative",
]
ALL_FEATURES = WAVEFORM_FEATURES + RELATIVE_FEATURES


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — TIGHT RIVER FOOTPRINT
# ══════════════════════════════════════════════════════════════════════════════

def build_tight_footprint(feat_df, v6_df):
    """
    Concave hull of high-confidence water detections, eroded inward.

    Two-tier anchor selection:
      Tier 1: ensemble=1, conf ≥ FOOTPRINT_CONF, z < RIVERBED_Z_MAX
              → certain laser-hit-the-bottom points (deepest, most reliable)
      Tier 2: ensemble=1, conf ≥ FOOTPRINT_CONF_SURFACE, z < RIVERBED_Z_SURFACE_MAX
              → water-surface returns in shallow/bend sections where the laser
              does not reach the riverbed (covers river turns missed by tier-1)
    """
    mc     = (v6_df["xgb_proba"].values + v6_df["deep_proba"].values) * 0.5
    z      = feat_df["z"].values
    ens    = v6_df["ensemble"].values

    tier1  = (ens == 1) & (mc >= FOOTPRINT_CONF) & (z < RIVERBED_Z_MAX)
    tier2  = (ens == 1) & (mc >= FOOTPRINT_CONF_SURFACE) & (z < RIVERBED_Z_SURFACE_MAX)

    # Proximity filter: only keep tier-2 anchors within TIER2_MAX_DIST_FROM_T1 metres
    # of a tier-1 (riverbed) anchor.  This prevents surface water-like returns in
    # non-river areas (wet meadows, puddles) from spiking the concave hull.
    tier2_only = tier2 & ~tier1
    n_t2_raw = int(tier2_only.sum())
    if tier1.sum() > 0 and n_t2_raw > 0:
        xy1 = np.column_stack([feat_df["x"].values[tier1], feat_df["y"].values[tier1]])
        xy2 = np.column_stack([feat_df["x"].values[tier2_only], feat_df["y"].values[tier2_only]])
        tree1 = KDTree(xy1)
        dists, _ = tree1.query(xy2)
        near = dists <= TIER2_MAX_DIST_FROM_T1
        t2_idx = np.where(tier2_only)[0]
        filtered_tier2 = np.zeros(len(feat_df), dtype=bool)
        filtered_tier2[t2_idx[near]] = True
        n_removed = int((~near).sum())
        print(f"  Tier-2 proximity filter: kept {int(near.sum()):,} / {n_t2_raw:,} "
              f"(removed {n_removed} isolated surface anchors >{TIER2_MAX_DIST_FROM_T1}m from tier-1)")
        anchor = tier1 | filtered_tier2
    else:
        anchor = tier1 | tier2_only

    n_anchor = int(anchor.sum())
    print(f"  Tier-1 riverbed anchors (conf≥{FOOTPRINT_CONF}, z<{RIVERBED_Z_MAX}m): "
          f"{int(tier1.sum()):,}")
    print(f"  Tier-2 surface  anchors (conf≥{FOOTPRINT_CONF_SURFACE}, z<{RIVERBED_Z_SURFACE_MAX}m): "
          f"{int((anchor & ~tier1).sum()):,}  (after proximity filter)")
    print(f"  Total anchors: {n_anchor:,}")

    xw = feat_df["x"].values[anchor]
    yw = feat_df["y"].values[anchor]

    mp        = MultiPoint(np.column_stack([xw, yw]))
    raw_hull  = concave_hull(mp, ratio=HULL_RATIO)

    # If MultiPolygon (river bend can create disconnected islands), keep ALL parts
    # so the bend is not silently discarded.
    if isinstance(raw_hull, MultiPolygon):
        n_parts = len(raw_hull.geoms)
        print(f"  WARNING: concave hull produced {n_parts} polygons — keeping all parts")
        # union preserves the full geometry as a (possibly MultiPolygon) shape
        raw_hull = raw_hull.buffer(0)   # repairs topology; keeps all geoms

    footprint = raw_hull.buffer(-FOOTPRINT_EROSION)
    if footprint.is_empty:
        print("  WARNING: eroded hull is empty; using raw hull")
        footprint = raw_hull

    # Fix any topology after buffer
    footprint = footprint.buffer(0)

    print(f"  Concave hull area      : {raw_hull.area:,.0f} m²")
    print(f"  Tight footprint (−{FOOTPRINT_EROSION}m): {footprint.area:,.0f} m²")

    # Return tier-1 xy so the surface grid can use it as a proximity guard
    tier1_xy = np.column_stack([feat_df["x"].values[tier1],
                                 feat_df["y"].values[tier1]])
    return footprint, raw_hull, tier1_xy


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — LOCAL ADAPTIVE SURFACE GRID
# ══════════════════════════════════════════════════════════════════════════════

def build_surface_grid(feat_df, v6_df, tier1_xy=None):
    """
    Build a 2m-resolution grid of estimated water-surface elevations.

    Per cell, priority order:
      1. p95 z of water-like waveforms (n_peaks≤2, ec>0.85, refl<−15)
         within z ∈ [SURF_Z_LO, SURF_Z_HI] and ≥ SURF_MIN_PTS points,
         BUT only if the cell centre is within SURF_MAX_T1_DIST of a tier-1
         anchor (riverbed point).  Cells too far from any riverbed return
         fall back to the RANSAC plane to avoid spurious surface estimates.
      2. Global RANSAC plane value at cell centre (fallback for empty/distant cells).

    Parameters
    ----------
    tier1_xy : (N,2) float array or None
        XY coordinates of tier-1 (riverbed) anchor points.  If None, the
        proximity guard is skipped (all primary estimates are accepted).

    Returns
    -------
    grid_z   : 2D float32 array (n_y, n_x) — smoothed surface elevation
    x_min, y_min, n_x, n_y : grid origin and dimensions
    plane_coef : (a, b, c) from RANSAC fit
    """
    x_all = feat_df["x"].values
    y_all = feat_df["y"].values
    z_all = feat_df["z"].values

    # ── RANSAC global plane (same as v7, used as fallback) ─────────────────────
    mc   = (v6_df["xgb_proba"].values + v6_df["deep_proba"].values) * 0.5
    conf = (v6_df["ensemble"].values == 1) & (mc >= 0.7)
    surf_cand = (conf
                 & (z_all >= 259.4) & (z_all <= 260.2)
                 & (feat_df["n_peaks"].values <= 3)
                 & (feat_df["energy_concentration"].values > SURF_EC_MIN)
                 & (feat_df["reflectance_dB"].values < -10))
    print(f"  Global RANSAC candidates: {surf_cand.sum():,}")

    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        residual_threshold=RANSAC_RESIDUAL,
        min_samples=max(0.5, 100 / max(surf_cand.sum(), 1)),
        random_state=42,
    )
    ransac.fit(np.column_stack([x_all[surf_cand], y_all[surf_cand]]),
               z_all[surf_cand])
    a, b = ransac.estimator_.coef_
    c    = ransac.estimator_.intercept_
    plane_coef = (float(a), float(b), float(c))
    n_in = int(ransac.inlier_mask_.sum())
    print(f"  RANSAC plane: z = {a:.5f}·x + {b:.5f}·y + {c:.4f}  "
          f"({n_in} inliers)")

    # ── Grid setup ─────────────────────────────────────────────────────────────
    x_min = float(x_all.min()); y_min = float(y_all.min())
    n_x = int(np.ceil((x_all.max() - x_min) / CELL_SIZE)) + 1
    n_y = int(np.ceil((y_all.max() - y_min) / CELL_SIZE)) + 1
    print(f"  Surface grid: {n_x} × {n_y} = {n_x*n_y} cells  "
          f"({CELL_SIZE}m resolution)")

    xi = np.clip(np.floor((x_all - x_min) / CELL_SIZE).astype(int), 0, n_x-1)
    yi = np.clip(np.floor((y_all - y_min) / CELL_SIZE).astype(int), 0, n_y-1)
    flat_id = yi * n_x + xi

    # ── Primary estimate: p95 of water-like surface waveforms per cell ─────────
    wl_surf = ((feat_df["n_peaks"].values <= SURF_N_PEAKS_MAX)
               & (feat_df["energy_concentration"].values > SURF_EC_MIN)
               & (feat_df["reflectance_dB"].values < SURF_REFL_MAX)
               & (z_all >= SURF_Z_LO) & (z_all <= SURF_Z_HI))
    print(f"  Water-like surface points (z∈[{SURF_Z_LO},{SURF_Z_HI}], "
          f"n_peaks≤{SURF_N_PEAKS_MAX}): {wl_surf.sum():,}")

    df_wl = pd.DataFrame({"flat": flat_id[wl_surf], "z": z_all[wl_surf]})
    # require at least SURF_MIN_PTS per cell
    counts = df_wl.groupby("flat")["z"].count()
    valid  = counts[counts >= SURF_MIN_PTS].index
    surf_primary = df_wl[df_wl["flat"].isin(valid)].groupby("flat")["z"].quantile(0.95)
    # cap to avoid outlier cells
    surf_primary = surf_primary.clip(upper=SURF_Z_CAP)
    print(f"  Cells with primary estimate: {len(surf_primary)} / {n_x*n_y}")

    # ── Fill grid: start from RANSAC plane everywhere, override with primary ───
    yg, xg = np.mgrid[0:n_y, 0:n_x]
    # cell-centre coordinates
    xc = x_min + (xg + 0.5) * CELL_SIZE
    yc = y_min + (yg + 0.5) * CELL_SIZE
    grid_z = (a * xc + b * yc + c).astype(np.float32)   # baseline = global plane

    # ── Proximity guard: build KDTree over tier-1 anchors if provided ─────────
    if tier1_xy is not None and len(tier1_xy) > 0:
        t1_tree = KDTree(tier1_xy)
    else:
        t1_tree = None

    # ── Stamp primary estimates (skipping cells too far from tier-1 anchors) ──
    n_primary_cells = 0
    n_skipped_cells = 0
    for flat_idx, sz in surf_primary.items():
        iy = int(flat_idx) // n_x
        ix = int(flat_idx) %  n_x
        if 0 <= iy < n_y and 0 <= ix < n_x:
            if t1_tree is not None:
                xc_cell = x_min + (ix + 0.5) * CELL_SIZE
                yc_cell = y_min + (iy + 0.5) * CELL_SIZE
                dist, _ = t1_tree.query([[xc_cell, yc_cell]])
                if dist[0] > SURF_MAX_T1_DIST:
                    n_skipped_cells += 1
                    continue   # keep RANSAC plane value for this cell
            grid_z[iy, ix] = sz
            n_primary_cells += 1

    if n_skipped_cells > 0:
        print(f"  Surface proximity guard: skipped {n_skipped_cells} cells "
              f">{SURF_MAX_T1_DIST}m from tier-1 anchors (RANSAC fallback used)")
    n_plane_cells = n_x * n_y - n_primary_cells
    print(f"  Grid: {n_primary_cells} cells from local data, "
          f"{n_plane_cells} from RANSAC plane")

    # ── Gaussian smooth ────────────────────────────────────────────────────────
    grid_z_smooth = gaussian_filter(grid_z, sigma=SMOOTH_SIGMA).astype(np.float32)
    diff = np.abs(grid_z_smooth - grid_z)
    print(f"  Smoothing (σ={SMOOTH_SIGMA} cell): max Δz = {diff.max():.4f} m")

    return grid_z_smooth, x_min, y_min, n_x, n_y, xi, yi, plane_coef


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def classify(feat_df, v6_df, in_footprint, local_surface_z, wf_ensemble):
    """
    Apply the water-surface model and merge with v6 waveform predictions.

    Inside the tight footprint — geometry dominates:
      z ≤ local_surface_z + WATER_TOL  →  WATER (1)
      z >  local_surface_z + WATER_TOL →  LAND  (0)  [exposed rock / bank]

    Outside the tight footprint — waveform model dominates:
      v6 says land (0)      →  LAND      (0)
      v6 says water (1)     →  UNCERTAIN (2)  [shallow margin, might be water]
      v6 says uncertain (2) →  UNCERTAIN (2)

    Returns merged_label (int8): 0=land, 1=water, 2=uncertain
    """
    z     = feat_df["z"].values
    N     = len(z)
    z_diff = z - local_surface_z   # positive = above estimated surface

    merged = np.full(N, 2, dtype=np.int8)

    # ── Inside footprint ───────────────────────────────────────────────────────
    below_surf = in_footprint & (z_diff <= WATER_TOL)
    above_surf = in_footprint & (z_diff >  WATER_TOL)
    merged[below_surf] = 1   # WATER
    merged[above_surf] = 0   # LAND (above surface = rock / gravel / bank)

    # ── Outside footprint ──────────────────────────────────────────────────────
    outside = ~in_footprint
    merged[outside & (wf_ensemble == 0)] = 0   # land confirmed by waveform
    merged[outside & (wf_ensemble == 1)] = 2   # uncertain — margin water?
    merged[outside & (wf_ensemble == 2)] = 2   # uncertain

    # ── Stats ──────────────────────────────────────────────────────────────────
    in_count = int(in_footprint.sum())
    print(f"\n  Inside footprint: {in_count:,}  ({100*in_footprint.mean():.1f}%)")
    print(f"    z ≤ surface+{WATER_TOL}m  → water : {below_surf.sum():>8,}")
    print(f"    z >  surface+{WATER_TOL}m → land  : {above_surf.sum():>8,}")
    print(f"  Outside footprint: {int(outside.sum()):,}")
    print(f"    v6=land     → land     : {int((outside&(wf_ensemble==0)).sum()):>8,}")
    print(f"    v6=water    → uncertain: {int((outside&(wf_ensemble==1)).sum()):>8,}")
    print(f"    v6=uncertain→ uncertain: {int((outside&(wf_ensemble==2)).sum()):>8,}")

    print(f"\n  Merged label counts:")
    for lv, nm in [(0,"land"), (1,"water"), (2,"uncertain")]:
        n = int((merged==lv).sum())
        print(f"    {lv} ({nm:<12}): {n:>8,}  ({100*n/N:.1f}%)")

    return merged


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — RETRAIN XGBOOST
# ══════════════════════════════════════════════════════════════════════════════

def spatial_cv_split(y_coord, n_folds=5):
    edges = np.percentile(y_coord, np.linspace(0, 100, n_folds + 1))
    return [
        (np.where((y_coord < edges[f]) | (y_coord > edges[f+1]))[0],
         np.where((y_coord >= edges[f]) & (y_coord <= edges[f+1]))[0])
        for f in range(n_folds)
    ]


def train_xgb(feat_df, merged_labels, out_dir):
    cols    = [c for c in ALL_FEATURES if c in feat_df.columns]
    vals    = np.nan_to_num(feat_df[cols].values.astype(np.float32),
                            nan=0.0, posinf=0.0, neginf=0.0)
    y_coord = feat_df["y"].values
    train_m = (merged_labels == 0) | (merged_labels == 1)
    X_tr    = vals[train_m]
    y_tr    = (merged_labels[train_m] == 1).astype(np.int32)
    y_c_tr  = y_coord[train_m]

    n_w = int(y_tr.sum()); n_l = int((y_tr==0).sum())
    spw = round(n_l / max(n_w,1), 3)
    print(f"\n  XGBoost: {len(cols)} features | "
          f"water={n_w:,}  land={n_l:,}  spw={spw}")

    cv_f1, cv_auc = [], []
    print(f"\n  5-fold spatial CV:")
    for fold, (tr_i, va_i) in enumerate(spatial_cv_split(y_c_tr)):
        if len(va_i) == 0 or len(np.unique(y_tr[va_i])) < 2:
            continue
        clf = xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0)
        clf.fit(X_tr[tr_i], y_tr[tr_i],
                eval_set=[(X_tr[va_i], y_tr[va_i])], verbose=False)
        pr  = clf.predict_proba(X_tr[va_i])[:, 1]
        f1  = f1_score(y_tr[va_i], (pr>=0.5).astype(int),
                       average="macro", zero_division=0)
        try:    auc = roc_auc_score(y_tr[va_i], pr)
        except: auc = float("nan")
        cv_f1.append(f1); cv_auc.append(auc)
        print(f"    Fold {fold+1}: F1={f1:.3f}  AUC={auc:.3f}")

    print(f"\n  CV macro-F1 : {np.mean(cv_f1):.3f} ± {np.std(cv_f1):.3f}")
    print(f"  CV ROC-AUC  : {np.nanmean(cv_auc):.3f}")

    print(f"\n  Training final XGBoost on {len(X_tr):,} rows …")
    final = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0)
    final.fit(X_tr, y_tr, verbose=False)
    mpath = os.path.join(out_dir, "v8_xgb.json")
    final.save_model(mpath)
    print(f"  Saved → {mpath}")

    imp = (pd.DataFrame({"feature": cols,
                         "importance": final.feature_importances_})
             .sort_values("importance", ascending=False))
    print(f"\n  Top 15 features:")
    print(imp.head(15).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 8))
    t20 = imp.head(20)
    ax.barh(t20["feature"].iloc[::-1], t20["importance"].iloc[::-1], color="#3498db")
    ax.set_xlabel("XGBoost importance (gain)")
    ax.set_title("v8 Surface-v2 — XGBoost Feature Importance")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=150)
    plt.close()

    xgb_proba_all = final.predict_proba(vals)[:, 1]
    cv_res = {"macro_f1_mean": float(np.mean(cv_f1)),
              "macro_f1_std":  float(np.std(cv_f1)),
              "auc_mean":      float(np.nanmean(cv_auc)),
              "n_water": n_w, "n_land": n_l, "feature_cols": cols}
    return final, cv_res, xgb_proba_all


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4b — RETRAIN DEEP MODEL
# ══════════════════════════════════════════════════════════════════════════════

class _CB(nn.Module):
    def __init__(self, i, o, k):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv1d(i, o, k, padding="same", bias=False),
            nn.BatchNorm1d(o), nn.ReLU(inplace=True))
    def forward(self, x): return self.b(x)

class V8Net(nn.Module):
    """Same architecture as V7Net; n_spatial = len(ALL_FEATURES)."""
    def __init__(self, n_spatial):
        super().__init__()
        self.wf = nn.Sequential(
            _CB(1,32,3), _CB(32,64,5), _CB(64,64,11),
            nn.MaxPool1d(4), _CB(64,128,5), nn.AdaptiveAvgPool1d(1))
        self.sp = nn.Sequential(
            nn.Linear(n_spatial,128), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Linear(128,64),        nn.BatchNorm1d(64),  nn.ReLU(True),
            nn.Linear(64,32),         nn.ReLU(True))
        self.head = nn.Sequential(
            nn.Linear(160,128), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Linear(128,64),  nn.ReLU(True),
            nn.Linear(64,2))
    def forward(self, wf, sp):
        return self.head(torch.cat([self.wf(wf).squeeze(-1), self.sp(sp)], 1))

class _DS(Dataset):
    def __init__(self, g, s, l):
        self.g=torch.from_numpy(g).unsqueeze(1); self.s=torch.from_numpy(s)
        self.l=torch.from_numpy(l)
    def __len__(self): return len(self.l)
    def __getitem__(self, i): return self.g[i], self.s[i], self.l[i]

class _Focal(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, smooth=0.1):
        super().__init__()
        self.gamma=gamma; self.alpha=alpha; self.smooth=smooth
    def forward(self, logits, targets):
        n = logits.size(1)
        with torch.no_grad():
            st = torch.zeros_like(logits).fill_(self.smooth/n)
            st.scatter_(1, targets.unsqueeze(1), 1.0-self.smooth+self.smooth/n)
        lp   = torch.nn.functional.log_softmax(logits, 1)
        pt   = lp.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = (1-pt).pow(self.gamma)*(-(st*lp).sum(1))
        if self.alpha is not None:
            at = torch.where(targets==1,
                             torch.full_like(pt, self.alpha),
                             torch.full_like(pt, 1-self.alpha))
            loss = at * loss
        return loss.mean()


def train_deep(feat_df, grids_all, merged_labels, out_dir,
               epochs=80, batch_size=256, lr=1e-3, patience=20, val_frac=0.20):
    cols    = [c for c in ALL_FEATURES if c in feat_df.columns]
    N       = len(feat_df)
    train_m = (merged_labels==0)|(merged_labels==1)
    tr_idx  = np.where(train_m)[0]

    grids_t   = np.array(grids_all[tr_idx], dtype=np.float32)
    spatial_t = feat_df[cols].values[train_m].astype(np.float32)
    labels_t  = (merged_labels[train_m]==1).astype(np.int64)
    y_coord   = feat_df["y"].values[train_m]

    grids_t   = np.nan_to_num(grids_t,   nan=0., posinf=0., neginf=0.)
    spatial_t = np.nan_to_num(spatial_t, nan=0., posinf=0., neginf=0.)

    cutoff   = np.percentile(y_coord, 100*(1-val_frac))
    val_m    = y_coord >= cutoff
    trn_m    = ~val_m
    print(f"\n  V8Net: {len(cols)} features | "
          f"train={int(trn_m.sum()):,}  val={int(val_m.sum()):,}")

    g_mean  = float(grids_t[trn_m].mean())
    g_std   = float(grids_t[trn_m].std()) + 1e-6
    sp_mean = spatial_t[trn_m].mean(0)
    sp_std  = spatial_t[trn_m].std(0) + 1e-6
    gn = (grids_t-g_mean)/g_std; sn = (spatial_t-sp_mean)/sp_std

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_pos    = int((labels_t[trn_m]==1).sum())
    n_neg    = int(trn_m.sum())-n_pos
    alpha_f  = round(n_neg/(n_pos+n_neg), 4)
    print(f"  Device: {device}  water={n_pos:,}  land={n_neg:,}  α={alpha_f:.3f}")

    model     = V8Net(n_spatial=len(cols)).to(device)
    criterion = _Focal(gamma=2.0, alpha=alpha_f, smooth=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=5e-6)
    scaler    = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    nw = min(4, os.cpu_count() or 1)
    tr_ld = DataLoader(_DS(gn[trn_m],sn[trn_m],labels_t[trn_m]),
                       batch_size=batch_size, shuffle=True,
                       num_workers=nw, pin_memory=True)
    va_ld = DataLoader(_DS(gn[val_m],sn[val_m],labels_t[val_m]),
                       batch_size=batch_size*2, shuffle=False,
                       num_workers=nw, pin_memory=True)

    best_f1=0.; pat=0
    hist={"tl":[],"vl":[],"f1":[],"auc":[]}
    mpath=os.path.join(out_dir,"v8_deep.pt")
    print(f"\n  {'Ep':>4}  {'TrLoss':>8}  {'VaLoss':>8}  {'F1':>7}  {'AUC':>7}  LR")
    for ep in range(1, epochs+1):
        model.train(); tl=0.
        for wf,sp,lb in tr_ld:
            wf=wf.to(device,non_blocking=True); sp=sp.to(device,non_blocking=True)
            lb=lb.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=scaler.is_enabled()):
                loss=criterion(model(wf,sp),lb)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            tl+=loss.item()*len(lb)
        tl/=len(tr_ld.dataset)

        model.eval(); vl=0.
        preds_,proba_,labs_=[],[],[]
        with torch.no_grad():
            for wf,sp,lb in va_ld:
                lg=model(wf.to(device),sp.to(device))
                vl+=criterion(lg,lb.to(device)).item()*len(lb)
                pr=torch.softmax(lg,1)[:,1]
                preds_.append(lg.argmax(1).cpu().numpy())
                proba_.append(pr.cpu().numpy()); labs_.append(lb.numpy())
        vl/=len(va_ld.dataset)
        preds=np.concatenate(preds_); proba=np.concatenate(proba_)
        labs=np.concatenate(labs_)
        vf1=f1_score(labs,preds,average="macro",zero_division=0)
        try: vauc=roc_auc_score(labs,proba)
        except: vauc=float("nan")
        scheduler.step(); lr_c=optimizer.param_groups[0]["lr"]
        hist["tl"].append(tl); hist["vl"].append(vl)
        hist["f1"].append(vf1); hist["auc"].append(vauc)
        flag=""
        if vf1>best_f1:
            best_f1=vf1; pat=0; flag=" ← best"
            torch.save(model.state_dict(), mpath)
        else: pat+=1
        print(f"  {ep:>4}  {tl:>8.4f}  {vl:>8.4f}  {vf1:>7.4f}  {vauc:>7.4f}  "
              f"{lr_c:.2e}{flag}")
        if pat>=patience:
            print(f"\n  Early stopping at epoch {ep}"); break

    print(f"\n  Best val macro-F1: {best_f1:.4f}")
    model.load_state_dict(torch.load(mpath,map_location=device,weights_only=True))
    model.eval()
    preds_,labs_=[],[]
    with torch.no_grad():
        for wf,sp,lb in va_ld:
            preds_.append(model(wf.to(device),sp.to(device)).argmax(1).cpu().numpy())
            labs_.append(lb.numpy())
    print(f"\n  Validation report:")
    print(classification_report(np.concatenate(labs_),np.concatenate(preds_),
                                target_names=["land","water"],zero_division=0))

    fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4))
    ep_r=range(1,len(hist["tl"])+1)
    a1.plot(ep_r,hist["tl"],label="train"); a1.plot(ep_r,hist["vl"],label="val")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Focal Loss"); a1.legend()
    a2.plot(ep_r,hist["f1"],label="macro-F1"); a2.plot(ep_r,hist["auc"],label="AUC")
    a2.set_xlabel("Epoch"); a2.set_ylabel("Score"); a2.legend()
    plt.suptitle("V8Net — Adaptive surface-model labels")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"training_curve.png"),dpi=150)
    plt.close()

    stats={"grid_mean":g_mean,"grid_std":g_std,
           "spatial_mean":sp_mean.tolist(),"spatial_std":sp_std.tolist(),
           "spatial_cols":cols,"best_val_f1":float(best_f1)}
    with open(os.path.join(out_dir,"v8_deep_stats.json"),"w") as fh:
        json.dump(stats,fh,indent=2)
    print(f"  Model → {mpath}")

    print(f"\n  Deep inference on all {N:,} points …")
    sp_all=np.nan_to_num(feat_df[cols].values.astype(np.float32),
                          nan=0.,posinf=0.,neginf=0.)
    sn_all=(sp_all-sp_mean)/sp_std
    gf=np.nan_to_num(np.array(grids_all,dtype=np.float32),nan=0.,posinf=0.,neginf=0.)
    gn_all=(gf-g_mean)/g_std
    model.to(device).eval()
    probas=np.zeros(N,np.float32); bs=2048
    with torch.no_grad():
        for s in range(0,N,bs):
            e=min(s+bs,N)
            wfb=torch.from_numpy(gn_all[s:e]).unsqueeze(1).to(device)
            spb=torch.from_numpy(sn_all[s:e]).to(device)
            probas[s:e]=torch.softmax(model(wfb,spb),1)[:,1].cpu().numpy()
            if s%50_000==0 and s>0: print(f"    {s:>6,}/{N:,}")
    return model, stats, {"best_val_f1":float(best_f1)}, probas


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — EXPORT + PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def export_and_plot(feat_df, in_footprint, local_surface_z, merged_label,
                    xgb_proba, deep_proba, grid_z, x_min, y_min, n_x, n_y,
                    footprint_poly, raw_hull, plane_coef, out_dir):
    N = len(feat_df)
    x = feat_df["x"].values; y = feat_df["y"].values; z = feat_df["z"].values

    xgb_pred  = (xgb_proba  >= 0.5).astype(np.int8)
    deep_pred = (deep_proba >= 0.5).astype(np.int8)
    agree     = xgb_pred == deep_pred
    ensemble  = xgb_pred.copy(); ensemble[~agree] = 2

    print(f"\n  Retrained ensemble on all {N:,} points:")
    print(f"    XGBoost: water={int(xgb_pred.sum()):,}  "
          f"land={int((xgb_pred==0).sum()):,}")
    print(f"    Deep:    water={int(deep_pred.sum()):,}  "
          f"land={int((deep_pred==0).sum()):,}")
    print(f"    Agreement: {100*agree.mean():.1f}%  "
          f"({int((~agree).sum()):,} uncertain)")
    for lv,nm in [(0,"land"),(1,"water"),(2,"uncertain")]:
        n=int((ensemble==lv).sum())
        print(f"    Ensemble {lv} ({nm}): {n:,}  ({100*n/N:.1f}%)")

    # ── CSV ────────────────────────────────────────────────────────────────────
    out = pd.DataFrame({
        "x":x,"y":y,"z":z,
        "reflectance_dB": feat_df["reflectance_dB"].values
                          if "reflectance_dB" in feat_df.columns else 0.,
        "in_footprint":   in_footprint.astype(np.int8),
        "local_surface_z": np.round(local_surface_z,4),
        "z_above_surface": np.round(z-local_surface_z,4),
        "merged_label":   merged_label,
        "xgb_pred":       xgb_pred, "xgb_proba":  np.round(xgb_proba,4),
        "deep_pred":      deep_pred, "deep_proba": np.round(deep_proba,4),
        "ensemble":       ensemble,
    })
    for col in ["energy_concentration","max_amp_norm_by_energy",
                "height_above_local_min","height_percentile_local",
                "planarity","roughness","n_peaks","depth_proxy_m",
                "z_relative","amplitude_weighted_center"]:
        if col in feat_df.columns:
            out[col] = feat_df[col].values
    out.to_csv(OUT_PATH, index=False)
    print(f"\n  Saved {N:,} rows → {OUT_PATH}")

    cm = {0:"saddlebrown", 1:"steelblue", 2:"gold"}
    lm = {0:"Land", 1:"Water", 2:"Uncertain"}

    def _plot_poly(ax, geom, fill=False, fill_color=None, fill_alpha=0.12, **kwargs):
        """Plot a Polygon or MultiPolygon boundary on ax, optionally filled."""
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path as MplPath
        geoms = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
        color = kwargs.get("color", "k")
        for g in geoms:
            # Exterior boundary
            xy = np.array(g.exterior.coords)
            ax.plot(xy[:,0], xy[:,1], **kwargs)
            # Interior rings (holes)
            for interior in g.interiors:
                ixy = np.array(interior.coords)
                ax.plot(ixy[:,0], ixy[:,1], **kwargs)
            # Semi-transparent fill (exterior minus holes)
            if fill:
                verts, codes = [], []
                for ring in [g.exterior] + list(g.interiors):
                    rxy = np.array(ring.coords)
                    codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(rxy)-2) + [MplPath.CLOSEPOLY]
                    verts += list(rxy)
                patch = PathPatch(MplPath(verts, codes),
                                  facecolor=fill_color or color,
                                  alpha=fill_alpha, edgecolor="none",
                                  zorder=kwargs.get("zorder", 1) - 1)
                ax.add_patch(patch)

    # ── Plot 1: top-down scatter (non-canopy + river boundary contours) ──────────
    nc = z <= CANOPY_Z_MAX   # non-canopy mask

    mean_proba = (xgb_proba + deep_proba) / 2.0
    print("\n  Computing river boundary contours for scatter …")
    grid_raw, rb_x_min, rb_y_min, rb_n_x, rb_n_y = rasterize(x, y, mean_proba)
    grid_smooth = fill_and_smooth(grid_raw)
    rb_contours = extract_contours(grid_smooth, rb_x_min, rb_y_min)

    fig, axes = plt.subplots(1,2,figsize=(22,9))
    for ax, (arr, title) in zip(axes, [
        (merged_label[nc], f"Merged Labels — no canopy (z ≤ {CANOPY_Z_MAX}m)"),
        (ensemble[nc],     f"v8 Retrained Ensemble — no canopy (z ≤ {CANOPY_Z_MAX}m)"),
    ]):
        xs, ys = x[nc], y[nc]
        for lv in [2,0,1]:
            m = arr==lv
            if not m.any(): continue
            ax.scatter(xs[m],ys[m],c=cm[lv],s=0.4,alpha=0.6,
                       label=f"{lm[lv]} ({m.sum():,})",rasterized=True)
        # Tight footprint boundary (filled so over-extension into land is visible)
        _plot_poly(ax, footprint_poly, color="k", lw=1.5, label="Tight footprint",
                   fill=True, fill_color="steelblue", fill_alpha=0.10, zorder=6)
        # Raw hull (before erosion)
        _plot_poly(ax, raw_hull, color="grey", lw=0.8, linestyle="--",
                   label="Raw hull (pre-erosion)", zorder=5)
        # River boundary contours from mean probability field
        _draw_contours(ax, rb_contours)
        ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title(title)
        handles, labels_leg = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_leg, handles))
        ax.legend(by_label.values(), by_label.keys(), markerscale=6, fontsize=7)
    plt.suptitle(
        f"v8 Adaptive Surface Model — Top-down view\n"
        f"River boundary: inner p={PROB_INNER} (white) · "
        f"center p={PROB_CENTER} (yellow) · outer p={PROB_OUTER} (orange)",
        fontsize=11,
    )
    plt.tight_layout()
    p1=os.path.join(out_dir,"topdown_scatter.png")
    plt.savefig(p1,dpi=150); plt.close()
    print(f"  Top-down scatter  → {p1}")

    # ── Plot 2: surface grid diagnostic ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14,5))
    extent = [x_min, x_min+n_x*CELL_SIZE, y_min, y_min+n_y*CELL_SIZE]
    im = ax.imshow(grid_z, origin="lower", extent=extent,
                   aspect="auto", cmap="Blues_r",
                   vmin=grid_z.min(), vmax=grid_z.max())
    plt.colorbar(im, ax=ax, label="Estimated water surface z (m)")
    _plot_poly(ax, footprint_poly, color="k", lw=1.5, label="Tight footprint",
               fill=True, fill_color="steelblue", fill_alpha=0.10, zorder=6)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Local water surface grid ({CELL_SIZE}m cells, "
                 f"Gaussian σ={SMOOTH_SIGMA})\n"
                 f"Darker = lower surface elevation (deeper)")
    ax.legend()
    plt.tight_layout()
    p2=os.path.join(out_dir,"surface_grid.png")
    plt.savefig(p2,dpi=150); plt.close()
    print(f"  Surface grid      → {p2}")

    # ── Plot 3: cross-section ──────────────────────────────────────────────────
    y_mid   = float(np.median(y)); y_w = 5.0
    sl      = np.abs(y-y_mid) <= y_w
    print(f"\n  Cross-section: y ∈ [{y_mid-y_w:.1f},{y_mid+y_w:.1f}] m  "
          f"→ {sl.sum():,} pts")

    # surface values along slice
    a_c, b_c, c_c = plane_coef
    x_line = np.linspace(x.min(), x.max(), 300)
    z_surf_line = a_c*x_line + b_c*y_mid + c_c

    fig, axes = plt.subplots(2,1,figsize=(18,11),sharex=True)
    for ax, (arr, title) in zip(axes,[
        (merged_label,"Merged Labels"),
        (ensemble,    "v8 Retrained Ensemble"),
    ]):
        for lv in [2,0,1]:
            m = sl&(arr==lv)
            if not m.any(): continue
            ax.scatter(x[m],z[m],c=cm[lv],s=2,alpha=0.75,
                       label=f"{lm[lv]} ({m.sum():,})",rasterized=True)
        # Local surface from grid along this y strip
        xi_sl = np.clip(np.floor((x[sl]-x_min)/CELL_SIZE).astype(int),0,n_x-1)
        yi_sl = np.clip(np.floor((y[sl]-y_min)/CELL_SIZE).astype(int),0,n_y-1)
        surf_sl = grid_z[yi_sl, xi_sl]
        ax.scatter(x[sl], surf_sl, c="cyan", s=2, alpha=0.4,
                   label="Local surface estimate", zorder=7, rasterized=True)
        ax.plot(x_line, z_surf_line,   "b--", lw=1.2,
                label=f"RANSAC plane (y={y_mid:.1f}m)", zorder=8)
        ax.plot(x_line, z_surf_line+WATER_TOL, "b:", lw=0.8, alpha=0.7,
                label=f"+{WATER_TOL}m tolerance", zorder=8)
        ax.set_ylabel("z (m)"); ax.set_title(title)
        ax.legend(markerscale=4,fontsize=8,loc="upper right")
        ax.grid(True,alpha=0.2)
    axes[-1].set_xlabel("x (m)")
    plt.suptitle(f"Side-view cross-section  (y ≈ {y_mid:.1f} m ± {y_w} m)\n"
                 "Cyan = local surface grid estimate  |  "
                 "Blue dashed = global RANSAC plane",fontsize=11)
    plt.tight_layout()
    p3=os.path.join(out_dir,"crosssection.png")
    plt.savefig(p3,dpi=150); plt.close()
    print(f"  Cross-section     → {p3}")
    return ensemble


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Loading {V6_WF_CSV} …")
    v6_df   = pd.read_csv(V6_WF_CSV)
    N       = len(v6_df)
    print(f"  {N:,} points")

    print(f"Loading {FEAT_PATH} …")
    feat_df = pd.read_csv(FEAT_PATH)
    assert len(feat_df)==N, f"Row mismatch: {len(feat_df)} vs {N}"
    print(f"  {N:,} × {len(feat_df.columns)} features")

    print(f"Loading {GRIDS_PATH} …")
    grids_all = np.load(GRIDS_PATH, mmap_mode="r")
    print(f"  grids: {grids_all.shape}")

    avail   = [c for c in ALL_FEATURES if c in feat_df.columns]
    missing = [c for c in ALL_FEATURES if c not in feat_df.columns]
    print(f"\nFeatures: {len(avail)}/{len(ALL_FEATURES)} available"
          + (f"  missing: {missing}" if missing else ""))

    # ── Phase 1 ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 1 — TIGHT RIVER FOOTPRINT")
    print(f"{'='*60}")
    footprint_poly, raw_hull, tier1_xy = build_tight_footprint(feat_df, v6_df)
    in_footprint = contains_xy(footprint_poly,
                                feat_df["x"].values, feat_df["y"].values)
    print(f"  Points inside tight footprint: {in_footprint.sum():,} "
          f"({100*in_footprint.mean():.1f}%)")

    # ── Phase 2 ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 2 — LOCAL ADAPTIVE SURFACE GRID")
    print(f"{'='*60}")
    grid_z, x_min, y_min, n_x, n_y, xi, yi, plane_coef = \
        build_surface_grid(feat_df, v6_df, tier1_xy=tier1_xy)

    # Look up per-point local surface elevation
    local_surface_z = grid_z[yi, xi].astype(np.float32)

    # ── Phase 3 ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 3 — CLASSIFICATION")
    print(f"{'='*60}")
    wf_ensemble  = v6_df["ensemble"].values.astype(np.int8)
    merged_label = classify(feat_df, v6_df, in_footprint,
                             local_surface_z, wf_ensemble)

    # ── Phase 4a ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 4a — RETRAIN XGBOOST")
    print(f"{'='*60}")
    _, xgb_cv, xgb_proba_all = train_xgb(feat_df, merged_label, MODEL_DIR)

    # ── Phase 4b ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 4b — RETRAIN V8Net")
    print(f"{'='*60}")
    _, _, deep_cv, deep_proba_all = train_deep(
        feat_df, grids_all, merged_label, MODEL_DIR)

    # ── Phase 5 ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 5 — EXPORT + PLOTS")
    print(f"{'='*60}")
    final_ens = export_and_plot(
        feat_df, in_footprint, local_surface_z, merged_label,
        xgb_proba_all, deep_proba_all,
        grid_z, x_min, y_min, n_x, n_y,
        footprint_poly, raw_hull, plane_coef, MODEL_DIR)

    metrics = {
        "footprint": {
            "conf_threshold": FOOTPRINT_CONF,
            "riverbed_z_max": RIVERBED_Z_MAX,
            "hull_ratio": HULL_RATIO,
            "erosion_m": FOOTPRINT_EROSION,
            "hull_area_m2": round(raw_hull.area,1),
            "footprint_area_m2": round(footprint_poly.area,1),
            "points_inside": int(in_footprint.sum()),
            "pct_inside": round(float(100*in_footprint.mean()),1),
        },
        "surface_grid": {
            "cell_size_m": CELL_SIZE,
            "grid_shape": [int(n_y), int(n_x)],
            "smooth_sigma": SMOOTH_SIGMA,
            "plane_coef": {"a": plane_coef[0], "b": plane_coef[1],
                           "c": plane_coef[2]},
        },
        "merged_label_counts": {
            "land": int((merged_label==0).sum()),
            "water": int((merged_label==1).sum()),
            "uncertain": int((merged_label==2).sum()),
        },
        "xgb_cv": xgb_cv,
        "deep_cv": deep_cv,
        "final_ensemble": {
            "land": int((final_ens==0).sum()),
            "water": int((final_ens==1).sum()),
            "uncertain": int((final_ens==2).sum()),
        },
    }
    with open(os.path.join(MODEL_DIR,"v8_metrics.json"),"w") as fh:
        json.dump(metrics,fh,indent=2)

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"\n  River footprint   : {in_footprint.sum():,} / {N:,} points "
          f"({100*in_footprint.mean():.1f}%)")
    print(f"  Merged water      : {int((merged_label==1).sum()):,} "
          f"({100*(merged_label==1).mean():.1f}%)")
    print(f"  Merged land       : {int((merged_label==0).sum()):,}")
    print(f"  Merged uncertain  : {int((merged_label==2).sum()):,}")
    print(f"  XGBoost CV F1     : {xgb_cv['macro_f1_mean']:.3f} "
          f"± {xgb_cv['macro_f1_std']:.3f}")
    print(f"  Deep best val F1  : {deep_cv['best_val_f1']:.3f}")
    print(f"\n  Output: {OUT_PATH}")
    print(f"  Models: {MODEL_DIR}")
    print(f"\n  CloudCompare — colour by 'ensemble': "
          f"0=land (brown)  1=water (blue)  2=uncertain (gold)")
    print(f"  Also: 'in_footprint' shows the tight channel mask directly.")
    print(f"  And:  'z_above_surface' = signed distance from water surface plane.")


if __name__ == "__main__":
    main()
