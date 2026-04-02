"""
spatial_propagation.py — Physics-driven label propagation + retraining (v7).

Physics principle: if a laser pulse reached the riverbed at location (x, y),
water MUST exist above it. Use confident riverbed detections from the v6
waveform-only ensemble as spatial anchors to recover water surface and water
column points that were previously left uncertain.

Inputs:
  pointclouds/labeled_pointcloud_v5_waveform_only.csv  — v6 ensemble predictions
  data_processed/features_v2.csv                       — full feature matrix (42 cols)
  data_processed/waveform_grids.npy                    — 234k × 200 waveform grids

Outputs:
  data_processed/labels_v7_propagated.csv              — propagated label array
  models/v7-propagated/
    v7_xgb.json, v7_deep.pt, v7_deep_stats.json
    v7_metrics.json, feature_importance.png, training_curve.png
  pointclouds/labeled_pointcloud_v6_propagated.csv
  models/v7-propagated/topdown_scatter.png
  models/v7-propagated/crosssection.png

Propagated label values:
  0 = land (ensemble predicted land, no water anchor nearby)
  1 = water (confident v6 ensemble OR propagated from riverbed anchor)
  2 = uncertain (no anchor in cell OR above z_water_max tolerance; models disagreed)
  3 = flagged for review (land prediction but water anchor directly below)
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
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report,
)

ROOT = Path(__file__).resolve().parent.parent.parent

# ── Paths ──────────────────────────────────────────────────────────────────────
V6_PC_PATH  = ROOT / "pointclouds"    / "labeled_pointcloud_v5_waveform_only.csv"
FEAT_PATH   = ROOT / "data_processed" / "features_v2.csv"
GRIDS_PATH  = ROOT / "data_processed" / "waveform_grids.npy"
MODEL_DIR   = ROOT / "models"         / "v7-propagated"
OUT_PATH    = ROOT / "pointclouds"    / "labeled_pointcloud_v6_propagated.csv"
LABELS_OUT  = ROOT / "data_processed" / "labels_v7_propagated.csv"

# ── Propagation parameters ────────────────────────────────────────────────────
CELL_SIZE           = 0.5   # metres — 2D grid cell side length
CONF_THRESHOLD      = 0.7   # min mean(xgb_proba, deep_proba) to count as anchor
UNCERTAIN_TOLERANCE = 0.5   # uncertain z ≤ z_water_max + this → water
LAND_REVIEW_TOL     = 0.3   # land z ≤ z_water_max + this → flagged for review

# ── Feature sets ──────────────────────────────────────────────────────────────
# Waveform-only (same discriminators as v6 — physically grounded)
WAVEFORM_FEATURES = [
    "energy_concentration", "max_amp_norm_by_energy",
    "n_clusters", "n_peaks", "n_gaps", "n_samples", "time_span",
    "max_amp", "mean_amp", "std_amp", "total_energy",
    "max_gap", "mean_gap", "total_gap", "first_last_span",
    "energy_ratio_late", "first_peak_amp", "last_peak_amp", "peak_amp_ratio",
    "depth_proxy_m", "amplitude_weighted_center", "active_bins_ratio",
    "reflectance_dB",
]

# Relative elevation + geometry: safe to use now because labels are no longer
# elevation-gated (they come from physics/propagation, not z-thresholds).
# height_above_local_min ≈ 0 for water points (at local minimum), > 0 for banks.
# NO absolute z anywhere.
RELATIVE_FEATURES = [
    "height_above_local_min",       # relative to small-radius local minimum
    "height_percentile_local",      # rank in local neighbourhood (low = at min)
    "planarity", "roughness", "linearity", "sphericity",
    "height_range_local", "height_std_local",
    "z_relative",                   # z minus tile-minimum — no absolute elevation
]

ALL_FEATURES = WAVEFORM_FEATURES + RELATIVE_FEATURES


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — SPATIAL PROPAGATION
# ══════════════════════════════════════════════════════════════════════════════

def run_spatial_propagation(v6_df):
    """
    Apply physics-based label propagation.

    Returns
    -------
    new_label : np.ndarray int8, length N
        0=land, 1=water, 2=uncertain, 3=flagged-for-review
    z_water_max_arr : np.ndarray float32, length N
        Per-point: max z of confident water anchors in the same 0.5m grid cell.
        NaN when the cell has no confident water anchor.
    """
    x          = v6_df["x"].values
    y          = v6_df["y"].values
    z          = v6_df["z"].values.astype(np.float32)
    xgb_proba  = v6_df["xgb_proba"].values.astype(np.float32)
    deep_proba = v6_df["deep_proba"].values.astype(np.float32)
    ensemble   = v6_df["ensemble"].values.astype(np.int8)   # 0=land,1=water,2=uncertain

    mean_conf = (xgb_proba + deep_proba) * 0.5

    # ── Anchor selection ───────────────────────────────────────────────────────
    conf_water = (ensemble == 1) & (mean_conf >= CONF_THRESHOLD)
    n_anchor   = int(conf_water.sum())

    print(f"  v6 ensemble breakdown:")
    print(f"    Land      (0): {int((ensemble==0).sum()):>8,}  ({100*(ensemble==0).mean():.1f}%)")
    print(f"    Water     (1): {int((ensemble==1).sum()):>8,}  ({100*(ensemble==1).mean():.1f}%)")
    print(f"    Uncertain (2): {int((ensemble==2).sum()):>8,}  ({100*(ensemble==2).mean():.1f}%)")
    print(f"  Confident water anchors (ensemble=1, mean_conf≥{CONF_THRESHOLD}): "
          f"{n_anchor:>8,}  ({100*n_anchor/len(z):.1f}%)")

    if n_anchor == 0:
        raise RuntimeError(
            "No confident water anchors found. Check v6 predictions and CONF_THRESHOLD.")

    # ── Build 2D spatial grid ──────────────────────────────────────────────────
    # Compact integer cell key: xi * large_prime + yi
    x_min = float(x.min())
    y_min = float(y.min())
    xi = np.floor((x - x_min) / CELL_SIZE).astype(np.int32)
    yi = np.floor((y - y_min) / CELL_SIZE).astype(np.int32)
    cell_id = xi.astype(np.int64) * 1_000_000 + yi.astype(np.int64)

    # z_water_max per cell — maximum z of all confident anchors in that cell
    anchor_cells = cell_id[conf_water]
    anchor_z     = z[conf_water]
    cell_z_max = (
        pd.DataFrame({"cell": anchor_cells, "z": anchor_z})
        .groupby("cell")["z"]
        .max()
        .to_dict()
    )
    n_cells_with_water = len(cell_z_max)
    print(f"  0.5m grid cells that contain confident water: {n_cells_with_water:,}")

    # Look up z_water_max for every point (NaN if cell has no anchor)
    z_water_max_arr = np.array(
        [cell_z_max.get(int(c), np.nan) for c in cell_id], dtype=np.float32
    )
    has_anchor = np.isfinite(z_water_max_arr)

    # ── Apply propagation rules ────────────────────────────────────────────────
    new_label = ensemble.copy()   # start from: 0=land, 1=water, 2=uncertain

    # Rule 1 — uncertain + cell has anchor + point not too far above water max
    uncertain  = ensemble == 2
    prop_water = uncertain & has_anchor & (z <= z_water_max_arr + UNCERTAIN_TOLERANCE)
    new_label[prop_water] = 1

    # Rule 2 — land + cell has anchor + within review tolerance above water max
    land    = ensemble == 0
    flagged = land & has_anchor & (z <= z_water_max_arr + LAND_REVIEW_TOL)
    new_label[flagged] = 3

    # ── Report ─────────────────────────────────────────────────────────────────
    n_still_unc = int((new_label == 2).sum())
    print(f"\n  Propagation results:")
    print(f"    Uncertain → Water (z ≤ z_water_max+{UNCERTAIN_TOLERANCE}m): "
          f"{prop_water.sum():>8,}")
    print(f"    Uncertain remaining (no anchor or above threshold):          "
          f"{n_still_unc:>8,}")
    print(f"    Land → Flagged for review (z ≤ z_water_max+{LAND_REVIEW_TOL}m): "
          f"{flagged.sum():>8,}")

    print(f"\n  Final propagated label counts:")
    for lv, nm in [(0, "land"), (1, "water"), (2, "uncertain"), (3, "flagged")]:
        n = int((new_label == lv).sum())
        print(f"    {lv} ({nm:<12}): {n:>8,}  ({100*n/len(new_label):.1f}%)")

    return new_label, z_water_max_arr


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — RETRAIN XGBOOST
# ══════════════════════════════════════════════════════════════════════════════

def spatial_cv_split(y_coord, n_folds=5):
    edges = np.percentile(y_coord, np.linspace(0, 100, n_folds + 1))
    return [
        (np.where((y_coord < edges[f]) | (y_coord > edges[f + 1]))[0],
         np.where((y_coord >= edges[f]) & (y_coord <= edges[f + 1]))[0])
        for f in range(n_folds)
    ]


def train_xgb(feat_df, prop_labels, out_dir):
    """
    Train XGBoost on propagated labels.
    Training set: labels 0 (land) and 1 (water).
    Excluded from training: 2 (uncertain) and 3 (flagged for review).
    Returns: (model, feature_cols, cv_metrics, xgb_proba_all_N)
    """
    cols = [c for c in ALL_FEATURES if c in feat_df.columns]
    missing = set(ALL_FEATURES) - set(feat_df.columns)
    if missing:
        print(f"  WARNING: missing features: {sorted(missing)}")

    feat_vals = feat_df[cols].values.astype(np.float32)
    feat_vals = np.nan_to_num(feat_vals, nan=0.0, posinf=0.0, neginf=0.0)
    y_coord   = feat_df["y"].values

    train_mask = (prop_labels == 0) | (prop_labels == 1)
    X_tr    = feat_vals[train_mask]
    y_tr    = (prop_labels[train_mask] == 1).astype(np.int32)
    y_c_tr  = y_coord[train_mask]

    n_w = int(y_tr.sum())
    n_l = int((y_tr == 0).sum())
    spw = round(n_l / max(n_w, 1), 3)
    print(f"\n  XGBoost: {len(cols)} features | "
          f"water={n_w:,}  land={n_l:,}  scale_pos_weight={spw}")

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

    # Final model on full training set
    print(f"\n  Training final XGBoost on {len(X_tr):,} labelled rows …")
    final = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="logloss",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    final.fit(X_tr, y_tr, verbose=False)
    mpath = os.path.join(out_dir, "v7_xgb.json")
    final.save_model(mpath)
    print(f"  Model → {mpath}")

    imp = pd.DataFrame({"feature": cols,
                        "importance": final.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    print(f"\n  Top 15 features:")
    print(imp.head(15).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 8))
    top20 = imp.head(20)
    ax.barh(top20["feature"].iloc[::-1], top20["importance"].iloc[::-1],
            color="#3498db")
    ax.set_xlabel("XGBoost importance (gain)")
    ax.set_title("v7 Propagated — XGBoost Feature Importance\n"
                 "(waveform + relative elevation, no absolute z)")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=150)
    plt.close()

    # Inference on all N points
    xgb_proba_all = final.predict_proba(feat_vals)[:, 1]

    cv_res = {
        "macro_f1_mean": float(np.mean(cv_f1)),
        "macro_f1_std":  float(np.std(cv_f1)),
        "auc_mean":      float(np.nanmean(cv_auc)),
        "n_water": n_w, "n_land": n_l,
        "feature_cols": cols,
    }
    return final, cols, cv_res, xgb_proba_all


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — RETRAIN DEEP MODEL (V7Net)
# ══════════════════════════════════════════════════════════════════════════════

class _CB(nn.Module):
    def __init__(self, i, o, k):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv1d(i, o, k, padding="same", bias=False),
            nn.BatchNorm1d(o), nn.ReLU(inplace=True))
    def forward(self, x): return self.b(x)


class V7Net(nn.Module):
    """
    1D-CNN on waveform grid (200 bins) + MLP on scalar features.
    Identical architecture to V6Net; n_spatial is larger (adds relative elevation).
    dropout=0.0 — must match between training here and any future inference.
    """
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


def train_deep(feat_df, grids_all, prop_labels, out_dir,
               epochs=80, batch_size=256, lr=1e-3, patience=20, val_frac=0.20):
    """
    Train V7Net on propagated labels (water=1, land=0).
    Excludes uncertain (2) and flagged (3) from training.
    Returns: (model, stats_dict, cv_metrics, deep_proba_all_N)
    """
    cols   = [c for c in ALL_FEATURES if c in feat_df.columns]
    N      = len(feat_df)

    train_mask = (prop_labels == 0) | (prop_labels == 1)
    train_idx  = np.where(train_mask)[0]

    grids_t   = np.array(grids_all[train_idx], dtype=np.float32)
    spatial_t = feat_df[cols].values[train_mask].astype(np.float32)
    labels_t  = (prop_labels[train_mask] == 1).astype(np.int64)
    y_coord   = feat_df["y"].values[train_mask]

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    n_pos  = int((labels_t[trn_mask] == 1).sum())
    n_neg  = int(trn_mask.sum()) - n_pos
    alpha  = round(n_neg / (n_pos + n_neg), 4)
    print(f"  water={n_pos:,}  land={n_neg:,}  focal_alpha={alpha:.3f}")

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
    mpath   = os.path.join(out_dir, "v7_deep.pt")

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
            scaler.step(optimizer)
            scaler.update()
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

    # Report on best checkpoint
    model.load_state_dict(torch.load(mpath, map_location=device, weights_only=True))
    model.eval()
    preds_, labs_ = [], []
    with torch.no_grad():
        for wf, sp, lb in va_ld:
            preds_.append(model(wf.to(device), sp.to(device)).argmax(1).cpu().numpy())
            labs_.append(lb.numpy())
    print(f"\n  Validation report (best checkpoint):")
    print(classification_report(np.concatenate(labs_), np.concatenate(preds_),
                                target_names=["land", "water"], zero_division=0))

    # Training curve
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    ep_r = range(1, len(hist["tl"]) + 1)
    a1.plot(ep_r, hist["tl"], label="train"); a1.plot(ep_r, hist["vl"], label="val")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Focal Loss"); a1.legend()
    a2.plot(ep_r, hist["f1"], label="macro-F1"); a2.plot(ep_r, hist["auc"], label="AUC")
    a2.set_xlabel("Epoch"); a2.set_ylabel("Score"); a2.legend()
    plt.suptitle("V7Net — Propagated labels (waveform + relative elevation)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curve.png"), dpi=150)
    plt.close()

    # Save normalization stats
    stats = {
        "grid_mean":    g_mean,
        "grid_std":     g_std,
        "spatial_mean": sp_mean.tolist(),
        "spatial_std":  sp_std.tolist(),
        "spatial_cols": cols,
        "best_val_f1":  float(best_f1),
    }
    spath = os.path.join(out_dir, "v7_deep_stats.json")
    with open(spath, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"  Model → {mpath}  Stats → {spath}")

    # ── Inference on all N points ──────────────────────────────────────────────
    print(f"\n  Deep inference on all {N:,} points …")
    sp_all    = feat_df[cols].values.astype(np.float32)
    sp_all    = np.nan_to_num(sp_all, nan=0.0, posinf=0.0, neginf=0.0)
    sn_all    = (sp_all - sp_mean) / sp_std

    # Load full grids into memory for batch inference (≈190 MB)
    grids_f   = np.array(grids_all, dtype=np.float32)
    grids_f   = np.nan_to_num(grids_f, nan=0.0, posinf=0.0, neginf=0.0)
    gn_all    = (grids_f - g_mean) / g_std

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
# PHASE 4 — EXPORT + PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def export_and_plot(feat_df, prop_labels, z_water_max_arr,
                    xgb_proba, deep_proba, out_dir):
    N = len(feat_df)
    x = feat_df["x"].values
    y = feat_df["y"].values
    z = feat_df["z"].values

    xgb_pred  = (xgb_proba  >= 0.5).astype(np.int8)
    deep_pred = (deep_proba >= 0.5).astype(np.int8)

    # Ensemble: agreement → prediction; disagreement → 2 (uncertain)
    agree    = xgb_pred == deep_pred
    ensemble = xgb_pred.copy()
    ensemble[~agree] = 2

    print(f"\n  Retrained model predictions on all {N:,} points:")
    print(f"    XGBoost : water={int(xgb_pred.sum()):,}  "
          f"land={int((xgb_pred==0).sum()):,}")
    print(f"    Deep    : water={int(deep_pred.sum()):,}  "
          f"land={int((deep_pred==0).sum()):,}")
    print(f"    Agreement: {100*agree.mean():.1f}%  "
          f"({int((~agree).sum()):,} uncertain)")
    for lv, nm in [(0, "land"), (1, "water"), (2, "uncertain")]:
        n = int((ensemble == lv).sum())
        print(f"    Ensemble {lv} ({nm}): {n:,}  ({100*n/N:.1f}%)")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    out = pd.DataFrame({
        "x":              x,
        "y":              y,
        "z":              z,
        "reflectance_dB": feat_df["reflectance_dB"].values
                          if "reflectance_dB" in feat_df.columns else 0.0,
        # Spatial propagation result (input to retraining)
        "prop_label":     prop_labels,      # 0=land,1=water,2=uncertain,3=flagged
        "z_water_max_cell": np.round(z_water_max_arr, 3),
        # Retrained model predictions
        "xgb_pred":   xgb_pred,
        "xgb_proba":  np.round(xgb_proba, 4),
        "deep_pred":  deep_pred,
        "deep_proba": np.round(deep_proba, 4),
        "ensemble":   ensemble,             # 0=land, 1=water, 2=uncertain
    })
    # CloudCompare scalar fields
    for col in ["energy_concentration", "max_amp_norm_by_energy",
                "height_above_local_min", "height_percentile_local",
                "planarity", "roughness", "n_peaks", "depth_proxy_m",
                "z_relative", "amplitude_weighted_center"]:
        if col in feat_df.columns:
            out[col] = feat_df[col].values

    out.to_csv(OUT_PATH, index=False)
    print(f"\n  Saved {N:,} rows → {OUT_PATH}")

    # ── Colour maps ────────────────────────────────────────────────────────────
    c_prop = {0: "saddlebrown", 1: "steelblue", 2: "gold",   3: "crimson"}
    l_prop = {0: "Land",
              1: "Water (confident + propagated)",
              2: "Uncertain",
              3: "Flagged for review"}
    c_ens  = {0: "saddlebrown", 1: "steelblue", 2: "gold"}
    l_ens  = {0: "Land", 1: "Water", 2: "Uncertain"}

    # ── Plot 1: Top-down scatter ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    ax = axes[0]
    for lv in [2, 3, 0, 1]:    # draw uncertain/flagged beneath confident classes
        m = prop_labels == lv
        if not m.any():
            continue
        ax.scatter(x[m], y[m], c=c_prop[lv], s=0.4, alpha=0.7,
                   label=f"{l_prop[lv]} ({m.sum():,})", rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Step 1 — Propagated Labels\n(physics-anchored, input to retraining)")
    ax.legend(markerscale=6, fontsize=9)

    ax = axes[1]
    for lv in [2, 0, 1]:
        m = ensemble == lv
        if not m.any():
            continue
        ax.scatter(x[m], y[m], c=c_ens[lv], s=0.4, alpha=0.7,
                   label=f"{l_ens[lv]} ({m.sum():,})", rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Step 2 — v7 Retrained Ensemble\n"
                 "(XGBoost + V7Net, waveform + relative elevation)")
    ax.legend(markerscale=6, fontsize=9)

    plt.suptitle(
        "v7 Spatial Propagation — Top-down view  "
        "(no absolute z in any model feature)",
        fontsize=11)
    plt.tight_layout()
    p1 = os.path.join(out_dir, "topdown_scatter.png")
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"  Top-down scatter  → {p1}")

    # ── Plot 2: Side-view cross-section ───────────────────────────────────────
    # Slice ±5 m around the scene's median y — cuts across the river channel
    y_mid   = float(np.median(y))
    y_width = 5.0
    slice_m = np.abs(y - y_mid) <= y_width
    n_slice = int(slice_m.sum())
    print(f"\n  Cross-section: y ∈ [{y_mid - y_width:.1f}, {y_mid + y_width:.1f}] m "
          f"→ {n_slice:,} points")

    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    for ax, (label_arr, c_map, l_map, title) in zip(axes, [
        (prop_labels, c_prop, l_prop,
         "Propagated Labels — shows vertical extent of water recovery"),
        (ensemble,    c_ens,  l_ens,
         "v7 Retrained Ensemble — final water (blue) vs land (brown)"),
    ]):
        for lv in sorted(c_map.keys(), reverse=True):   # draw water on top
            m = slice_m & (label_arr == lv)
            if not m.any():
                continue
            ax.scatter(x[m], z[m], c=c_map[lv], s=2.0, alpha=0.75,
                       label=f"{l_map[lv]} ({m.sum():,})", rasterized=True)
        ax.set_ylabel("z (m)")
        ax.set_title(title)
        ax.legend(markerscale=4, fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("x (m)")
    plt.suptitle(
        f"Side-view cross-section  (y ≈ {y_mid:.1f} m ± {y_width} m)\n"
        "Water (blue) should form a coherent horizontal band; "
        "land (brown) above and below",
        fontsize=11)
    plt.tight_layout()
    p2 = os.path.join(out_dir, "crosssection.png")
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"  Cross-section     → {p2}")

    return ensemble


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading {V6_PC_PATH} …")
    v6_df = pd.read_csv(V6_PC_PATH)
    N = len(v6_df)
    print(f"  {N:,} points  |  columns: {list(v6_df.columns)}")

    print(f"\nLoading {FEAT_PATH} …")
    feat_df = pd.read_csv(FEAT_PATH)
    if len(feat_df) != N:
        raise RuntimeError(
            f"Row count mismatch: features_v2 has {len(feat_df)}, "
            f"v6 pointcloud has {N}. "
            "Both must be the same 234k-row dataset in the same order.")
    print(f"  {len(feat_df):,} points × {len(feat_df.columns)} features")

    print(f"\nLoading {GRIDS_PATH} …")
    grids_all = np.load(GRIDS_PATH, mmap_mode="r")
    print(f"  grids shape: {grids_all.shape}")

    avail = [c for c in ALL_FEATURES if c in feat_df.columns]
    missing = [c for c in ALL_FEATURES if c not in feat_df.columns]
    print(f"\nFeature availability: {len(avail)}/{len(ALL_FEATURES)} available")
    if missing:
        print(f"  Missing: {missing}")

    # ── Phase 1: Spatial propagation ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 1 — SPATIAL PROPAGATION")
    print(f"{'='*60}")
    prop_labels, z_water_max_arr = run_spatial_propagation(v6_df)

    # Persist propagated labels
    pd.DataFrame({
        "prop_label":       prop_labels,
        "z_water_max_cell": np.round(z_water_max_arr, 3),
    }).to_csv(LABELS_OUT, index=False)
    print(f"\n  Propagated labels → {LABELS_OUT}")

    # ── Phase 2: Retrain XGBoost ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 2 — RETRAIN XGBOOST")
    print(f"{'='*60}")
    xgb_model, xgb_cols, xgb_cv, xgb_proba_all = train_xgb(
        feat_df, prop_labels, MODEL_DIR)

    # ── Phase 3: Retrain Deep Model ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 3 — RETRAIN V7Net (deep)")
    print(f"{'='*60}")
    deep_model, deep_stats, deep_cv, deep_proba_all = train_deep(
        feat_df, grids_all, prop_labels, MODEL_DIR)

    # ── Phase 4: Export + plots ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 4 — EXPORT + PLOTS")
    print(f"{'='*60}")
    final_ensemble = export_and_plot(
        feat_df, prop_labels, z_water_max_arr,
        xgb_proba_all, deep_proba_all, MODEL_DIR)

    # Save combined metrics
    metrics = {
        "propagation_params": {
            "cell_size_m":           CELL_SIZE,
            "conf_threshold":        CONF_THRESHOLD,
            "uncertain_tolerance_m": UNCERTAIN_TOLERANCE,
            "land_review_tol_m":     LAND_REVIEW_TOL,
        },
        "propagated_label_counts": {
            "land":      int((prop_labels == 0).sum()),
            "water":     int((prop_labels == 1).sum()),
            "uncertain": int((prop_labels == 2).sum()),
            "flagged":   int((prop_labels == 3).sum()),
        },
        "xgb_cv":  xgb_cv,
        "deep_cv": deep_cv,
        "final_ensemble_counts": {
            "land":      int((final_ensemble == 0).sum()),
            "water":     int((final_ensemble == 1).sum()),
            "uncertain": int((final_ensemble == 2).sum()),
        },
    }
    mpath = os.path.join(MODEL_DIR, "v7_metrics.json")
    with open(mpath, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\n  Metrics → {mpath}")

    # ── Summary ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Output CSV    : {OUT_PATH}")
    print(f"  Models dir    : {MODEL_DIR}")
    print(f"\n  Open {OUT_PATH.name} in CloudCompare:")
    print(f"    colour by 'ensemble':")
    print(f"      0 = land       (brown/grey)")
    print(f"      1 = water      (blue)")
    print(f"      2 = uncertain  (gold)")
    print(f"    colour by 'prop_label' to see propagation step separately:")
    print(f"      3 = flagged for review (red — land prediction above water anchor)")
    print(f"\n  Plots:")
    print(f"    {MODEL_DIR}/topdown_scatter.png  — x-y top-down view")
    print(f"    {MODEL_DIR}/crosssection.png     — x-z side view (vertical extent)")
    print(f"\n  XGBoost CV macro-F1 : {xgb_cv['macro_f1_mean']:.3f} "
          f"± {xgb_cv['macro_f1_std']:.3f}")
    print(f"  Deep     best val F1: {deep_cv['best_val_f1']:.3f}")


if __name__ == "__main__":
    main()
