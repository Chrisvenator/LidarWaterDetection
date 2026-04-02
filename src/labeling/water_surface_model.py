"""
water_surface_model.py — Water surface plane model + retrained classifier (v7).

Rationale: vertical propagation fails because refraction shifts bottom returns
laterally from the water surface above them.  Instead:
  1. Define the *river footprint* in 2-D from confident waveform-only detections.
  2. Fit a *water surface plane*  z = ax + by + c  to candidate surface returns
     inside the footprint using RANSAC (robust against outliers).
  3. Label every point: z ≤ plane(x,y) + tolerance → water; otherwise land.
  4. Merge with the v6 waveform-only predictions for a consistent final label.
  5. Retrain XGBoost + V7Net on merged labels (waveform + relative elevation,
     no absolute z).

Inputs:
  pointclouds/labeled_pointcloud_v6_waveform_only.csv  — v6 waveform predictions
  data_processed/features_v2.csv                       — full 42-col feature matrix
  data_processed/waveform_grids.npy                    — 234k × 200 grids

Outputs:
  models/v7-water-surface/
    v7ws_xgb.json, v7ws_deep.pt, v7ws_deep_stats.json, v7ws_metrics.json
    footprint_and_plane.png, feature_importance.png, training_curve.png
  pointclouds/labeled_pointcloud_v7.csv

Merged label values (in output CSV):
  0 = land
  1 = water
  2 = uncertain / unresolvable disagreement

Final ensemble column:
  0 = land, 1 = water, 2 = uncertain (XGBoost ≠ Deep)
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
from shapely import concave_hull, MultiPoint, contains_xy
from sklearn.linear_model import RANSACRegressor, LinearRegression
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score, classification_report

ROOT = Path(__file__).resolve().parent.parent.parent

# ── Paths ──────────────────────────────────────────────────────────────────────
V6_WF_CSV   = ROOT / "pointclouds"    / "labeled_pointcloud_v6_waveform_only.csv"
FEAT_PATH   = ROOT / "data_processed" / "features_v2.csv"
GRIDS_PATH  = ROOT / "data_processed" / "waveform_grids.npy"
MODEL_DIR   = ROOT / "models"         / "v7-water-surface"
OUT_PATH    = ROOT / "pointclouds"    / "labeled_pointcloud_v7.csv"

# ── Footprint parameters ───────────────────────────────────────────────────────
FOOTPRINT_RATIO      = 0.2    # concave-hull tightness (lower = tighter)
FOOTPRINT_BUFFER     = 1.5    # metres outward buffer
BOUNDARY_WIDTH       = 1.0    # metres inward erosion for "near-boundary" zone
ANCHOR_Z_MAX         = 262.0  # only use confident water z < this for footprint

# ── Water surface plane parameters ────────────────────────────────────────────
SURF_Z_LO            = 259.4  # z lower bound for surface candidate selection
SURF_Z_HI            = 260.2  # z upper bound
SURF_N_PEAKS_MAX     = 3      # simple waveforms only
SURF_EC_MIN          = 0.85   # energy concentration threshold
SURF_REFL_MAX        = -10.0  # reflectance upper bound (water is low-reflectance)
RANSAC_RESIDUAL      = 0.20   # metres — inlier threshold for plane RANSAC

# ── Classification thresholds ──────────────────────────────────────────────────
WATER_TOL            = 0.20   # z ≤ surface_z + this → water inside footprint
CANOPY_ABOVE_SURF    = 2.0    # z > surface_z + this → definitely not water surface

# ── Feature sets (same as spatial_propagation.py) ─────────────────────────────
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
    "height_above_local_min",
    "height_percentile_local",
    "planarity", "roughness", "linearity", "sphericity",
    "height_range_local", "height_std_local",
    "z_relative",
]
ALL_FEATURES = WAVEFORM_FEATURES + RELATIVE_FEATURES


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — RIVER FOOTPRINT
# ══════════════════════════════════════════════════════════════════════════════

def build_footprint(feat_df, v6_df):
    """
    Build the river channel footprint polygon.

    Uses confident waveform-only detections at z < ANCHOR_Z_MAX (riverbed /
    surface returns, excluding spurious high-z false positives).

    Returns
    -------
    footprint      : shapely Polygon — river channel + FOOTPRINT_BUFFER
    inner_footprint: shapely Polygon — footprint eroded by BOUNDARY_WIDTH
                     (points between footprint and inner_footprint are
                      'near-boundary')
    """
    mc     = (v6_df["xgb_proba"].values + v6_df["deep_proba"].values) * 0.5
    conf_w = (v6_df["ensemble"].values == 1) & (mc >= 0.7)
    z_all  = feat_df["z"].values
    anchor = conf_w & (z_all < ANCHOR_Z_MAX)

    xw = feat_df["x"].values[anchor]
    yw = feat_df["y"].values[anchor]
    n_anchor = int(anchor.sum())
    print(f"  Footprint anchors (conf water, z<{ANCHOR_Z_MAX}m): {n_anchor:,}")

    mp       = MultiPoint(np.column_stack([xw, yw]))
    raw_hull = concave_hull(mp, ratio=FOOTPRINT_RATIO)
    footprint = raw_hull.buffer(FOOTPRINT_BUFFER)

    # Erode inward for boundary zone detection
    inner = footprint.buffer(-BOUNDARY_WIDTH)
    if inner.is_empty:
        inner = footprint   # very narrow hull — skip boundary distinction

    print(f"  Concave hull area  : {raw_hull.area:,.0f} m²")
    print(f"  Footprint area (+{FOOTPRINT_BUFFER}m buffer): {footprint.area:,.0f} m²")
    return footprint, inner


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — WATER SURFACE PLANE (RANSAC)
# ══════════════════════════════════════════════════════════════════════════════

def fit_water_surface(feat_df, in_footprint):
    """
    Fit the water surface plane z = a·x + b·y + c via RANSAC.

    Candidate returns are selected by:
      - Inside the river footprint
      - z in [SURF_Z_LO, SURF_Z_HI]  (near-surface elevation band)
      - n_peaks ≤ SURF_N_PEAKS_MAX    (simple / single-return waveforms)
      - energy_concentration > SURF_EC_MIN  (compact waveform)
      - reflectance_dB < SURF_REFL_MAX      (weak specular return = water surface)

    Returns
    -------
    ransac      : fitted RANSACRegressor
    plane_coef  : (a, b, c) such that z_pred = a*x + b*y + c
    n_inliers   : number of RANSAC inliers used
    """
    z   = feat_df["z"].values
    x   = feat_df["x"].values
    y   = feat_df["y"].values
    ec  = feat_df["energy_concentration"].values
    np_ = feat_df["n_peaks"].values
    ref = feat_df["reflectance_dB"].values

    cand = (
        in_footprint
        & (z >= SURF_Z_LO) & (z <= SURF_Z_HI)
        & (np_ <= SURF_N_PEAKS_MAX)
        & (ec  >  SURF_EC_MIN)
        & (ref <  SURF_REFL_MAX)
    )
    n_cand = int(cand.sum())
    print(f"  Surface candidates: {n_cand:,}  (z∈[{SURF_Z_LO},{SURF_Z_HI}], "
          f"n_peaks≤{SURF_N_PEAKS_MAX}, ec>{SURF_EC_MIN}, refl<{SURF_REFL_MAX})")
    if n_cand < 50:
        raise RuntimeError(
            f"Too few surface candidates ({n_cand}). "
            "Check SURF_Z_LO/HI and footprint.")

    xs = x[cand]; ys = y[cand]; zs = z[cand]
    X_fit = np.column_stack([xs, ys])

    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        residual_threshold=RANSAC_RESIDUAL,
        min_samples=max(0.5, 100 / n_cand),
        random_state=42,
    )
    ransac.fit(X_fit, zs)

    n_inliers  = int(ransac.inlier_mask_.sum())
    a, b       = ransac.estimator_.coef_
    c          = ransac.estimator_.intercept_
    resid_in   = zs[ransac.inlier_mask_] - ransac.predict(X_fit[ransac.inlier_mask_])

    print(f"  RANSAC inliers: {n_inliers:,} / {n_cand:,} "
          f"({100*n_inliers/n_cand:.1f}%)")
    print(f"  Water surface plane:  z = {a:.6f}·x + {b:.6f}·y + {c:.4f}")
    print(f"  Gradient: Δz/Δx = {a*100:.4f} m/100m  "
          f"Δz/Δy = {b*100:.4f} m/100m")
    print(f"  Inlier residuals: std={resid_in.std():.4f} m  "
          f"p5={np.percentile(resid_in,5):.4f}  p95={np.percentile(resid_in,95):.4f}")

    # Evaluate over full scene extent
    x_rng = np.array([x.min(), x.max()])
    dz    = (a * (x_rng[1] - x_rng[0]))
    print(f"  Predicted surface elevation: z∈[{a*x.min()+b*y.mean()+c:.3f}, "
          f"{a*x.max()+b*y.mean()+c:.3f}] m  (Δz={dz:.3f} m over {x_rng[1]-x_rng[0]:.0f}m)")

    return ransac, (a, b, c), n_inliers


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — SURFACE-MODEL CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def classify_surface_model(feat_df, in_footprint, in_inner, surface_z):
    """
    Apply the geometric water-surface model.

    Returns surface_label (int8):
      1  = water  (inside footprint, z ≤ surface_z + WATER_TOL)
      0  = land   (outside footprint, or z > surface_z + WATER_TOL)
     -1  = near-boundary uncertain (in footprint but not in inner footprint)
    """
    z = feat_df["z"].values
    near_boundary = in_footprint & ~in_inner

    label = np.zeros(len(z), dtype=np.int8)
    # Inside footprint: water if below surface + tolerance
    label[in_footprint & (z <= surface_z + WATER_TOL)] = 1
    # near-boundary stays 0 (land) for now; we mark ambiguous ones as -1 below
    label[near_boundary] = -1          # tentative: resolve in merge step

    # Count
    n_water    = int((label ==  1).sum())
    n_boundary = int((label == -1).sum())
    n_land     = int((label ==  0).sum())
    print(f"  Surface model:")
    print(f"    Water (z ≤ surface+{WATER_TOL}m, inside footprint): {n_water:>8,}")
    print(f"    Near-boundary (±{BOUNDARY_WIDTH}m of edge):          {n_boundary:>8,}")
    print(f"    Land (outside footprint or above surface):           {n_land:>8,}")
    return label


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — MERGE SURFACE MODEL WITH V6 WAVEFORM PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

def merge_labels(surface_label, wf_ensemble, z, surface_z, in_footprint):
    """
    Combine the surface-model label with the v6 waveform-only ensemble.

    Rules (in priority order):
      1. Both agree water  → WATER (1)
      2. Both agree land   → LAND  (0)
      3. Surface=water, wf=land:
           z > surface_z + CANOPY_ABOVE_SURF → LAND  (canopy / tall veg above river)
           otherwise                          → WATER (trust surface geometry)
      4. Surface=land (outside footprint), wf=water → UNCERTAIN (2)
         (possible puddle, off-bank wetness, or waveform false positive)
      5. Near-boundary (surface=-1) + wf=water   → WATER
         Near-boundary (surface=-1) + wf=land    → LAND
         Near-boundary (surface=-1) + wf=uncert  → UNCERTAIN (2)
      6. Any remaining disagreement               → UNCERTAIN (2)

    Returns merged_label (int8): 0=land, 1=water, 2=uncertain
    """
    N = len(z)
    merged = np.full(N, 2, dtype=np.int8)   # default: uncertain

    sm_water    = surface_label == 1
    sm_land     = surface_label == 0
    sm_boundary = surface_label == -1
    wf_water    = wf_ensemble   == 1
    wf_land     = wf_ensemble   == 0
    wf_uncert   = wf_ensemble   == 2

    # Rule 1 — both agree water
    merged[sm_water & wf_water] = 1

    # Rule 2 — both agree land
    merged[sm_land & wf_land]   = 0

    # Rule 3a — surface=water, wf=land, canopy height above surface → land
    canopy_above_river = sm_water & wf_land & (z > surface_z + CANOPY_ABOVE_SURF)
    merged[canopy_above_river] = 0

    # Rule 3b — surface=water, wf=land, near surface → trust surface geometry
    near_surface = sm_water & wf_land & (z <= surface_z + CANOPY_ABOVE_SURF)
    merged[near_surface] = 1

    # Rule 4 — wf=water but surface model says land (outside footprint)
    # Keep as uncertain (already default=2)

    # Rule 5 — near-boundary resolution
    merged[sm_boundary & wf_water]  = 1
    merged[sm_boundary & wf_land]   = 0
    # sm_boundary & wf_uncert stays 2

    # Rule 6 — surface=land + wf=uncertain → land (surface is more reliable here)
    merged[sm_land & wf_uncert] = 0

    # surface=water + wf=uncertain → water
    merged[sm_water & wf_uncert] = 1

    print(f"\n  Merged labels:")
    for lv, nm in [(0, "land"), (1, "water"), (2, "uncertain")]:
        n = int((merged == lv).sum())
        print(f"    {lv} ({nm:<12}): {n:>8,}  ({100*n/N:.1f}%)")

    # Detail on rule 3
    print(f"\n  Rule 3 detail:")
    print(f"    surface=water + wf=land + canopy above ({CANOPY_ABOVE_SURF}m): "
          f"→land   {canopy_above_river.sum():,}")
    print(f"    surface=water + wf=land + near surface:                        "
          f"→water  {near_surface.sum():,}")

    return merged


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — RETRAIN XGBOOST
# ══════════════════════════════════════════════════════════════════════════════

def spatial_cv_split(y_coord, n_folds=5):
    edges = np.percentile(y_coord, np.linspace(0, 100, n_folds + 1))
    return [
        (np.where((y_coord < edges[f]) | (y_coord > edges[f + 1]))[0],
         np.where((y_coord >= edges[f]) & (y_coord <= edges[f + 1]))[0])
        for f in range(n_folds)
    ]


def train_xgb(feat_df, merged_labels, out_dir):
    """Train XGBoost on merged labels 0/1; exclude uncertain (2)."""
    cols    = [c for c in ALL_FEATURES if c in feat_df.columns]
    missing = set(ALL_FEATURES) - set(feat_df.columns)
    if missing:
        print(f"  WARNING: missing features: {sorted(missing)}")

    vals    = np.nan_to_num(feat_df[cols].values.astype(np.float32),
                            nan=0.0, posinf=0.0, neginf=0.0)
    y_coord = feat_df["y"].values
    train_m = (merged_labels == 0) | (merged_labels == 1)
    X_tr    = vals[train_m]
    y_tr    = (merged_labels[train_m] == 1).astype(np.int32)
    y_c_tr  = y_coord[train_m]

    n_w  = int(y_tr.sum())
    n_l  = int((y_tr == 0).sum())
    spw  = round(n_l / max(n_w, 1), 3)
    print(f"\n  XGBoost: {len(cols)} features | water={n_w:,}  land={n_l:,}  spw={spw}")

    splits = spatial_cv_split(y_c_tr)
    cv_f1, cv_auc = [], []
    print(f"\n  5-fold spatial CV:")
    for fold, (tr_idx, va_idx) in enumerate(splits):
        if len(va_idx) == 0 or len(np.unique(y_tr[va_idx])) < 2:
            continue
        clf = xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0,
        )
        clf.fit(X_tr[tr_idx], y_tr[tr_idx],
                eval_set=[(X_tr[va_idx], y_tr[va_idx])], verbose=False)
        pr  = clf.predict_proba(X_tr[va_idx])[:, 1]
        pd_ = (pr >= 0.5).astype(int)
        f1  = f1_score(y_tr[va_idx], pd_, average="macro", zero_division=0)
        try:    auc = roc_auc_score(y_tr[va_idx], pr)
        except: auc = float("nan")
        cv_f1.append(f1); cv_auc.append(auc)
        print(f"    Fold {fold+1}: F1={f1:.3f}  AUC={auc:.3f}")

    print(f"\n  CV macro-F1 : {np.mean(cv_f1):.3f} ± {np.std(cv_f1):.3f}")
    print(f"  CV ROC-AUC  : {np.nanmean(cv_auc):.3f}")

    print(f"\n  Training final XGBoost on {len(X_tr):,} rows …")
    final = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="logloss",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    final.fit(X_tr, y_tr, verbose=False)
    mpath = os.path.join(out_dir, "v7ws_xgb.json")
    final.save_model(mpath)
    print(f"  Model → {mpath}")

    imp = (pd.DataFrame({"feature": cols, "importance": final.feature_importances_})
             .sort_values("importance", ascending=False))
    print(f"\n  Top 15 features:")
    print(imp.head(15).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 8))
    top20 = imp.head(20)
    ax.barh(top20["feature"].iloc[::-1], top20["importance"].iloc[::-1],
            color="#3498db")
    ax.set_xlabel("XGBoost importance (gain)")
    ax.set_title("v7 Water-Surface Model — XGBoost Feature Importance\n"
                 "(waveform + relative elevation, no absolute z)")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=150)
    plt.close()

    xgb_proba_all = final.predict_proba(vals)[:, 1]
    cv_res = {
        "macro_f1_mean": float(np.mean(cv_f1)),
        "macro_f1_std":  float(np.std(cv_f1)),
        "auc_mean":      float(np.nanmean(cv_auc)),
        "n_water": n_w, "n_land": n_l, "feature_cols": cols,
    }
    return final, cols, cv_res, xgb_proba_all


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5b — RETRAIN DEEP MODEL (V7Net)
# ══════════════════════════════════════════════════════════════════════════════

class _CB(nn.Module):
    def __init__(self, i, o, k):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv1d(i, o, k, padding="same", bias=False),
            nn.BatchNorm1d(o), nn.ReLU(inplace=True))
    def forward(self, x): return self.b(x)


class V7Net(nn.Module):
    """1D-CNN on waveform grid + MLP on scalar features. No dropout."""
    def __init__(self, n_spatial):
        super().__init__()
        self.wf = nn.Sequential(
            _CB(1, 32, 3), _CB(32, 64, 5), _CB(64, 64, 11),
            nn.MaxPool1d(4), _CB(64, 128, 5), nn.AdaptiveAvgPool1d(1))
        self.sp = nn.Sequential(
            nn.Linear(n_spatial, 128), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(True),
            nn.Linear(64, 32),         nn.ReLU(True))
        self.head = nn.Sequential(
            nn.Linear(160, 128), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Linear(128, 64),  nn.ReLU(True),
            nn.Linear(64, 2))

    def forward(self, wf, sp):
        return self.head(torch.cat([self.wf(wf).squeeze(-1), self.sp(sp)], 1))


class _DS(Dataset):
    def __init__(self, g, s, l):
        self.g = torch.from_numpy(g).unsqueeze(1)
        self.s = torch.from_numpy(s)
        self.l = torch.from_numpy(l)
    def __len__(self): return len(self.l)
    def __getitem__(self, i): return self.g[i], self.s[i], self.l[i]


class _Focal(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, smooth=0.1):
        super().__init__()
        self.gamma = gamma; self.alpha = alpha; self.smooth = smooth

    def forward(self, logits, targets):
        n = logits.size(1)
        with torch.no_grad():
            st = torch.zeros_like(logits).fill_(self.smooth / n)
            st.scatter_(1, targets.unsqueeze(1),
                        1.0 - self.smooth + self.smooth / n)
        lp   = torch.nn.functional.log_softmax(logits, 1)
        pt   = lp.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = (1 - pt).pow(self.gamma) * (-(st * lp).sum(1))
        if self.alpha is not None:
            at = torch.where(targets == 1,
                             torch.full_like(pt, self.alpha),
                             torch.full_like(pt, 1 - self.alpha))
            loss = at * loss
        return loss.mean()


def train_deep(feat_df, grids_all, merged_labels, out_dir,
               epochs=80, batch_size=256, lr=1e-3, patience=20, val_frac=0.20):
    cols   = [c for c in ALL_FEATURES if c in feat_df.columns]
    N      = len(feat_df)
    train_m = (merged_labels == 0) | (merged_labels == 1)
    train_idx = np.where(train_m)[0]

    grids_t   = np.array(grids_all[train_idx], dtype=np.float32)
    spatial_t = feat_df[cols].values[train_m].astype(np.float32)
    labels_t  = (merged_labels[train_m] == 1).astype(np.int64)
    y_coord   = feat_df["y"].values[train_m]

    grids_t   = np.nan_to_num(grids_t,   nan=0.0, posinf=0.0, neginf=0.0)
    spatial_t = np.nan_to_num(spatial_t, nan=0.0, posinf=0.0, neginf=0.0)

    cutoff   = np.percentile(y_coord, 100 * (1 - val_frac))
    val_mask = y_coord >= cutoff
    trn_mask = ~val_mask
    print(f"\n  V7Net: {len(cols)} features | "
          f"train={int(trn_mask.sum()):,}  val={int(val_mask.sum()):,}")

    g_mean  = float(grids_t[trn_mask].mean())
    g_std   = float(grids_t[trn_mask].std()) + 1e-6
    sp_mean = spatial_t[trn_mask].mean(0)
    sp_std  = spatial_t[trn_mask].std(0) + 1e-6
    gn = (grids_t   - g_mean)  / g_std
    sn = (spatial_t - sp_mean) / sp_std

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_pos    = int((labels_t[trn_mask] == 1).sum())
    n_neg    = int(trn_mask.sum()) - n_pos
    alpha    = round(n_neg / (n_pos + n_neg), 4)
    print(f"  Device: {device}  |  water={n_pos:,}  land={n_neg:,}  α={alpha:.3f}")

    model     = V7Net(n_spatial=len(cols)).to(device)
    criterion = _Focal(gamma=2.0, alpha=alpha, smooth=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=5e-6)
    scaler    = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    nw    = min(4, os.cpu_count() or 1)
    tr_ld = DataLoader(_DS(gn[trn_mask], sn[trn_mask], labels_t[trn_mask]),
                       batch_size=batch_size, shuffle=True,
                       num_workers=nw, pin_memory=True)
    va_ld = DataLoader(_DS(gn[val_mask],  sn[val_mask],  labels_t[val_mask]),
                       batch_size=batch_size * 2, shuffle=False,
                       num_workers=nw, pin_memory=True)

    best_f1 = 0.0; pat = 0
    hist    = {"tl": [], "vl": [], "f1": [], "auc": []}
    mpath   = os.path.join(out_dir, "v7ws_deep.pt")

    print(f"\n  {'Ep':>4}  {'TrLoss':>8}  {'VaLoss':>8}  {'F1':>7}  {'AUC':>7}  LR")
    for ep in range(1, epochs + 1):
        model.train(); tl = 0.0
        for wf, sp, lb in tr_ld:
            wf = wf.to(device, non_blocking=True)
            sp = sp.to(device, non_blocking=True)
            lb = lb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss = criterion(model(wf, sp), lb)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            tl += loss.item() * len(lb)
        tl /= len(tr_ld.dataset)

        model.eval(); vl = 0.0
        preds_, proba_, labs_ = [], [], []
        with torch.no_grad():
            for wf, sp, lb in va_ld:
                wf = wf.to(device, non_blocking=True)
                sp = sp.to(device, non_blocking=True)
                lb = lb.to(device, non_blocking=True)
                lg  = model(wf, sp)
                vl += criterion(lg, lb).item() * len(lb)
                pr  = torch.softmax(lg, 1)[:, 1]
                preds_.append(lg.argmax(1).cpu().numpy())
                proba_.append(pr.cpu().numpy())
                labs_.append(lb.cpu().numpy())
        vl   /= len(va_ld.dataset)
        preds = np.concatenate(preds_)
        proba = np.concatenate(proba_)
        labs  = np.concatenate(labs_)
        vf1   = f1_score(labs, preds, average="macro", zero_division=0)
        try:    vauc = roc_auc_score(labs, proba)
        except: vauc = float("nan")

        scheduler.step()
        lr_cur = optimizer.param_groups[0]["lr"]
        hist["tl"].append(tl); hist["vl"].append(vl)
        hist["f1"].append(vf1); hist["auc"].append(vauc)

        flag = ""
        if vf1 > best_f1:
            best_f1 = vf1; pat = 0; flag = " ← best"
            torch.save(model.state_dict(), mpath)
        else:
            pat += 1

        print(f"  {ep:>4}  {tl:>8.4f}  {vl:>8.4f}  {vf1:>7.4f}  {vauc:>7.4f}  "
              f"{lr_cur:.2e}{flag}")
        if pat >= patience:
            print(f"\n  Early stopping at epoch {ep}")
            break

    print(f"\n  Best val macro-F1: {best_f1:.4f}")

    model.load_state_dict(torch.load(mpath, map_location=device, weights_only=True))
    model.eval()
    preds_, labs_ = [], []
    with torch.no_grad():
        for wf, sp, lb in va_ld:
            preds_.append(model(wf.to(device), sp.to(device)).argmax(1).cpu().numpy())
            labs_.append(lb.numpy())
    print(f"\n  Validation report:")
    print(classification_report(np.concatenate(labs_), np.concatenate(preds_),
                                target_names=["land", "water"], zero_division=0))

    # Training curve
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    ep_r = range(1, len(hist["tl"]) + 1)
    a1.plot(ep_r, hist["tl"], label="train"); a1.plot(ep_r, hist["vl"], label="val")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Focal Loss"); a1.legend()
    a2.plot(ep_r, hist["f1"], label="macro-F1"); a2.plot(ep_r, hist["auc"], label="AUC")
    a2.set_xlabel("Epoch"); a2.set_ylabel("Score"); a2.legend()
    plt.suptitle("V7Net — Water-surface model labels (waveform + relative elevation)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curve.png"), dpi=150)
    plt.close()

    stats = {
        "grid_mean": g_mean, "grid_std": g_std,
        "spatial_mean": sp_mean.tolist(), "spatial_std": sp_std.tolist(),
        "spatial_cols": cols, "best_val_f1": float(best_f1),
    }
    with open(os.path.join(out_dir, "v7ws_deep_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"  Model → {mpath}")

    # Inference on all N points
    print(f"\n  Deep inference on all {N:,} points …")
    sp_all = np.nan_to_num(feat_df[cols].values.astype(np.float32),
                            nan=0.0, posinf=0.0, neginf=0.0)
    sn_all = (sp_all - sp_mean) / sp_std
    grids_f = np.nan_to_num(np.array(grids_all, dtype=np.float32),
                             nan=0.0, posinf=0.0, neginf=0.0)
    gn_all  = (grids_f - g_mean) / g_std

    model.to(device).eval()
    probas = np.zeros(N, np.float32)
    bs = 2048
    with torch.no_grad():
        for s in range(0, N, bs):
            e   = min(s + bs, N)
            wfb = torch.from_numpy(gn_all[s:e]).unsqueeze(1).to(device)
            spb = torch.from_numpy(sn_all[s:e]).to(device)
            probas[s:e] = torch.softmax(model(wfb, spb), 1)[:, 1].cpu().numpy()
            if s % 50_000 == 0 and s > 0:
                print(f"    {s:>6,}/{N:,}")

    return model, stats, {"best_val_f1": float(best_f1)}, probas


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — EXPORT + PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def export_and_plot(feat_df, in_footprint, surface_z, surface_label,
                    merged_label, xgb_proba, deep_proba,
                    footprint_poly, plane_coef, out_dir):
    N = len(feat_df)
    x = feat_df["x"].values
    y = feat_df["y"].values
    z = feat_df["z"].values

    xgb_pred  = (xgb_proba  >= 0.5).astype(np.int8)
    deep_pred = (deep_proba >= 0.5).astype(np.int8)
    agree     = xgb_pred == deep_pred
    ensemble  = xgb_pred.copy()
    ensemble[~agree] = 2

    print(f"\n  Retrained ensemble on all {N:,} points:")
    print(f"    XGBoost : water={int(xgb_pred.sum()):,}  "
          f"land={int((xgb_pred==0).sum()):,}")
    print(f"    Deep    : water={int(deep_pred.sum()):,}  "
          f"land={int((deep_pred==0).sum()):,}")
    print(f"    Agreement: {100*agree.mean():.1f}%  "
          f"({int((~agree).sum()):,} uncertain)")
    for lv, nm in [(0, "land"), (1, "water"), (2, "uncertain")]:
        n = int((ensemble == lv).sum())
        print(f"    Ensemble {lv} ({nm}): {n:,}  ({100*n/N:.1f}%)")
    print(f"\n  Points inside river footprint: {int(in_footprint.sum()):,} "
          f"({100*in_footprint.mean():.1f}%)")

    # ── CSV ────────────────────────────────────────────────────────────────────
    out = pd.DataFrame({
        "x": x, "y": y, "z": z,
        "reflectance_dB": feat_df["reflectance_dB"].values
                          if "reflectance_dB" in feat_df.columns else 0.0,
        "in_footprint":   in_footprint.astype(np.int8),
        "surface_z":      np.round(surface_z, 4),
        "surface_label":  surface_label,     # 1=water,0=land,-1=boundary
        "merged_label":   merged_label,      # 0=land,1=water,2=uncertain
        "xgb_pred":       xgb_pred,
        "xgb_proba":      np.round(xgb_proba, 4),
        "deep_pred":      deep_pred,
        "deep_proba":     np.round(deep_proba, 4),
        "ensemble":       ensemble,
    })
    for col in ["energy_concentration", "max_amp_norm_by_energy",
                "height_above_local_min", "height_percentile_local",
                "planarity", "roughness", "n_peaks", "depth_proxy_m",
                "z_relative", "amplitude_weighted_center"]:
        if col in feat_df.columns:
            out[col] = feat_df[col].values
    out.to_csv(OUT_PATH, index=False)
    print(f"\n  Saved {N:,} rows → {OUT_PATH}")

    # ── Colour maps ────────────────────────────────────────────────────────────
    c_merged = {0: "saddlebrown", 1: "steelblue", 2: "gold"}
    l_merged = {0: "Land", 1: "Water", 2: "Uncertain"}
    c_ens    = {0: "saddlebrown", 1: "steelblue", 2: "gold"}

    # ── Plot 1: Footprint + water surface plane diagnostic ─────────────────────
    a_coef, b_coef, c_coef = plane_coef
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    ax = axes[0]
    # Draw all points coloured by merged label
    for lv in [2, 0, 1]:
        m = merged_label == lv
        if not m.any(): continue
        ax.scatter(x[m], y[m], c=c_merged[lv], s=0.4, alpha=0.6,
                   label=f"{l_merged[lv]} ({m.sum():,})", rasterized=True)
    # Overlay footprint boundary
    fp_xy = np.array(footprint_poly.exterior.coords)
    ax.plot(fp_xy[:, 0], fp_xy[:, 1], "k-", lw=1.2, label="River footprint", zorder=5)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Merged labels + river footprint boundary")
    ax.legend(markerscale=6, fontsize=9)

    ax = axes[1]
    for lv in [2, 0, 1]:
        m = ensemble == lv
        if not m.any(): continue
        ax.scatter(x[m], y[m], c=c_ens[lv], s=0.4, alpha=0.6,
                   label=f"{l_merged[lv]} ({m.sum():,})", rasterized=True)
    ax.plot(fp_xy[:, 0], fp_xy[:, 1], "k-", lw=1.2, label="River footprint", zorder=5)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("v7 Retrained Ensemble (XGBoost + V7Net)")
    ax.legend(markerscale=6, fontsize=9)

    plt.suptitle("v7 Water-Surface Model — Top-down view\n"
                 "(no absolute z in any model feature)", fontsize=11)
    plt.tight_layout()
    p1 = os.path.join(out_dir, "topdown_scatter.png")
    plt.savefig(p1, dpi=150); plt.close()
    print(f"  Top-down scatter  → {p1}")

    # ── Plot 2: Side-view cross-section ───────────────────────────────────────
    y_mid   = float(np.median(y))
    y_width = 5.0
    sl      = np.abs(y - y_mid) <= y_width
    x_sl    = x[sl]
    print(f"\n  Cross-section: y ∈ [{y_mid-y_width:.1f}, {y_mid+y_width:.1f}] m  "
          f"→ {sl.sum():,} points")

    fig, axes = plt.subplots(2, 1, figsize=(18, 11), sharex=True)
    for ax, (label_arr, lmap, title) in zip(axes, [
        (merged_label, l_merged, "Merged Labels (surface model + waveform)"),
        (ensemble,     l_merged, "v7 Retrained Ensemble"),
    ]):
        for lv in [2, 0, 1]:
            m = sl & (label_arr == lv)
            if not m.any(): continue
            ax.scatter(x[m], z[m], c=c_merged[lv], s=2.0, alpha=0.75,
                       label=f"{lmap[lv]} ({m.sum():,})", rasterized=True)
        # Draw the water surface plane as a line over the slice
        x_line = np.linspace(x.min(), x.max(), 300)
        z_line = a_coef * x_line + b_coef * y_mid + c_coef
        ax.plot(x_line, z_line,    "b--", lw=1.5, label=f"Surface plane (y={y_mid:.1f}m)", zorder=6)
        ax.plot(x_line, z_line + WATER_TOL, "b:", lw=1.0, alpha=0.6,
                label=f"Surface + {WATER_TOL}m tolerance", zorder=6)
        ax.set_ylabel("z (m)")
        ax.set_title(title)
        ax.legend(markerscale=4, fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("x (m)")
    plt.suptitle(
        f"Side-view cross-section  (y ≈ {y_mid:.1f} m ± {y_width} m)\n"
        "Blue dashed = RANSAC water surface plane  |  "
        "Water (blue) should sit at or below the dashed line",
        fontsize=11)
    plt.tight_layout()
    p2 = os.path.join(out_dir, "crosssection.png")
    plt.savefig(p2, dpi=150); plt.close()
    print(f"  Cross-section     → {p2}")

    return ensemble


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Loading {V6_WF_CSV} …")
    v6_df = pd.read_csv(V6_WF_CSV)
    N = len(v6_df)
    print(f"  {N:,} points")

    print(f"Loading {FEAT_PATH} …")
    feat_df = pd.read_csv(FEAT_PATH)
    assert len(feat_df) == N, f"Row mismatch: {len(feat_df)} vs {N}"
    print(f"  {N:,} × {len(feat_df.columns)} features")

    print(f"Loading {GRIDS_PATH} …")
    grids_all = np.load(GRIDS_PATH, mmap_mode="r")
    print(f"  grids: {grids_all.shape}")

    avail   = [c for c in ALL_FEATURES if c in feat_df.columns]
    missing = [c for c in ALL_FEATURES if c not in feat_df.columns]
    print(f"\nFeatures: {len(avail)}/{len(ALL_FEATURES)} available"
          + (f"  missing: {missing}" if missing else ""))

    # ── Phase 1: River footprint ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 1 — RIVER FOOTPRINT  (concave hull + buffer)")
    print(f"{'='*60}")
    footprint_poly, inner_poly = build_footprint(feat_df, v6_df)

    # Vectorised point-in-polygon tests
    print(f"\n  Testing containment for all {N:,} points …")
    in_footprint = contains_xy(footprint_poly,
                                feat_df["x"].values, feat_df["y"].values)
    in_inner     = contains_xy(inner_poly,
                                feat_df["x"].values, feat_df["y"].values)
    print(f"  Points inside footprint:          {in_footprint.sum():>8,} "
          f"({100*in_footprint.mean():.1f}%)")
    print(f"  Points in boundary zone (±{BOUNDARY_WIDTH}m): "
          f"{(in_footprint & ~in_inner).sum():>8,}")

    # ── Phase 2: Water surface plane ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 2 — WATER SURFACE PLANE  (RANSAC)")
    print(f"{'='*60}")
    ransac, plane_coef, n_inliers = fit_water_surface(feat_df, in_footprint)
    a, b, c = plane_coef
    surface_z = (a * feat_df["x"].values
               + b * feat_df["y"].values
               + c).astype(np.float32)

    # ── Phase 3: Surface model classification ───────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 3 — SURFACE MODEL CLASSIFICATION")
    print(f"{'='*60}")
    surface_label = classify_surface_model(
        feat_df, in_footprint, in_inner, surface_z)

    # ── Phase 4: Merge ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 4 — MERGE: SURFACE MODEL + v6 WAVEFORM")
    print(f"{'='*60}")
    wf_ensemble = v6_df["ensemble"].values.astype(np.int8)
    merged_label = merge_labels(
        surface_label, wf_ensemble,
        feat_df["z"].values.astype(np.float32),
        surface_z, in_footprint)

    # ── Phase 5: Retrain ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 5a — RETRAIN XGBOOST")
    print(f"{'='*60}")
    _, xgb_cols, xgb_cv, xgb_proba_all = train_xgb(
        feat_df, merged_label, MODEL_DIR)

    print(f"\n{'='*60}")
    print("PHASE 5b — RETRAIN V7Net")
    print(f"{'='*60}")
    _, deep_stats, deep_cv, deep_proba_all = train_deep(
        feat_df, grids_all, merged_label, MODEL_DIR)

    # ── Phase 6: Export + plots ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 6 — EXPORT + PLOTS")
    print(f"{'='*60}")
    final_ens = export_and_plot(
        feat_df, in_footprint, surface_z, surface_label,
        merged_label, xgb_proba_all, deep_proba_all,
        footprint_poly, plane_coef, MODEL_DIR)

    # Save metrics
    metrics = {
        "footprint": {
            "concave_hull_ratio":  FOOTPRINT_RATIO,
            "buffer_m":            FOOTPRINT_BUFFER,
            "boundary_width_m":    BOUNDARY_WIDTH,
            "anchor_z_max":        ANCHOR_Z_MAX,
            "points_inside":       int(in_footprint.sum()),
            "pct_inside":          round(float(in_footprint.mean() * 100), 1),
        },
        "water_surface_plane": {
            "a": float(a), "b": float(b), "c": float(c),
            "ransac_inliers": n_inliers,
            "ransac_threshold_m": RANSAC_RESIDUAL,
        },
        "merged_label_counts": {
            "land": int((merged_label == 0).sum()),
            "water": int((merged_label == 1).sum()),
            "uncertain": int((merged_label == 2).sum()),
        },
        "xgb_cv":  xgb_cv,
        "deep_cv": deep_cv,
        "final_ensemble_counts": {
            "land": int((final_ens == 0).sum()),
            "water": int((final_ens == 1).sum()),
            "uncertain": int((final_ens == 2).sum()),
        },
    }
    mpath = os.path.join(MODEL_DIR, "v7ws_metrics.json")
    with open(mpath, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\n  Metrics → {mpath}")

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Points inside river footprint : "
          f"{in_footprint.sum():,} / {N:,}  "
          f"({100*in_footprint.mean():.1f}%)")
    print(f"  XGBoost CV macro-F1: {xgb_cv['macro_f1_mean']:.3f} "
          f"± {xgb_cv['macro_f1_std']:.3f}")
    print(f"  Deep best val F1   : {deep_cv['best_val_f1']:.3f}")
    print(f"\n  Output CSV  : {OUT_PATH}")
    print(f"  Models dir  : {MODEL_DIR}")
    print(f"\n  Open {OUT_PATH.name} in CloudCompare — colour by 'ensemble':")
    print(f"    0 = land  (brown), 1 = water  (blue), 2 = uncertain  (gold)")
    print(f"  Also try 'merged_label' to see the pre-retrain surface-model labels.")
    print(f"  'in_footprint' colours the river channel boundary directly.")


if __name__ == "__main__":
    main()
