"""
auto_labeler_v6.py — Waveform-only water/land classifier.

Labeling uses precise z-boundaries from manual cross-section inspection.
Models use ONLY waveform features + reflectance_dB.
No elevation. No neighbourhood geometry. No height_above_*. No z.

Zone definitions (cross-section verified):
  z < 259.6m          → UNDERWATER          (certain water)
  259.6 ≤ z < 259.9m  → WATER_SURFACE       (certain water)
  259.9 ≤ z < 260.0m  → gap                 (excluded)
  260.0 ≤ z < 260.4m  → DRY_RIVERBED        (certain land)
  260.4 ≤ z < 260.9m  → gap                 (excluded)
  260.9 ≤ z < 263.1m  → RIVER_BANKS_MEADOW  (certain land)
  263.1 ≤ z < 263.3m  → gap                 (excluded)
  z ≥ 263.3m          → CANOPY              (certain land)

Phases:
  1. LABEL      labels_v6.csv
  2. DIAGNOSTIC water_surface vs dry_riverbed — waveform grids, reflectance,
                Cohen's d for all features
  3. TRAIN      XGBoost + V6Net (waveform grid + waveform scalars only)
  4. EXPORT     labeled_pointcloud_v5_waveform_only.csv + plots

Outputs:
  data_processed/labels_v6.csv
  models/v6-waveform-only/diagnostic_waveforms.png
  models/v6-waveform-only/diagnostic_reflectance.png
  models/v6-waveform-only/diagnostic_cohens_d.png
  models/v6-waveform-only/feature_importance.png
  models/v6-waveform-only/training_curve.png
  models/v6-waveform-only/xy_scatter_predictions.png
  models/v6-waveform-only/v6_xgb.json
  models/v6-waveform-only/v6_deep.pt
  models/v6-waveform-only/v6_deep_stats.json
  models/v6-waveform-only/v6_metrics.json
  pointclouds/labeled_pointcloud_v5_waveform_only.csv
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
    f1_score, precision_score, recall_score, roc_auc_score,
    classification_report,
)

ROOT = Path(__file__).resolve().parent.parent.parent

# ── Paths ──────────────────────────────────────────────────────────────────────
FEAT_PATH   = ROOT / "data_processed" / "features_v2.csv"
GRIDS_PATH  = ROOT / "data_processed" / "waveform_grids.npy"
LABEL_OUT   = ROOT / "data_processed" / "labels_v6.csv"
PC_OUT      = ROOT / "pointclouds"    / "labeled_pointcloud_v5_waveform_only.csv"
MODEL_DIR   = ROOT / "models"         / "v6-waveform-only"

# ── Zone boundaries (cross-section inspection) ─────────────────────────────────
Z_UNDERWATER_MAX    = 259.6
Z_WATER_SURF_MAX    = 259.9
Z_DRY_BED_MIN       = 260.0
Z_DRY_BED_MAX       = 260.4
Z_BANKS_MIN         = 260.9
Z_BANKS_MAX         = 263.1
Z_CANOPY_MIN        = 263.3

# ── Feature sets ── NO z, NO geometry, NO height_above_* ──────────────────────
WAVEFORM_FEATURES = [
    # Primary discriminators (from diagnostic analysis)
    "energy_concentration",
    "max_amp_norm_by_energy",
    # Waveform structure
    "n_clusters", "n_peaks", "n_gaps", "n_samples", "time_span",
    "max_amp", "mean_amp", "std_amp", "total_energy",
    "max_gap", "mean_gap", "total_gap", "first_last_span",
    "energy_ratio_late", "first_peak_amp", "last_peak_amp", "peak_amp_ratio",
    "depth_proxy_m", "amplitude_weighted_center", "active_bins_ratio",
    # Scalar signal
    "reflectance_dB",
]


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — LABELING
# ══════════════════════════════════════════════════════════════════════════════

def create_labels(z):
    """
    Assign labels from z-values.
    Returns:
      label  : int8  (1=water, 0=land, -1=excluded)
      zone   : str
    """
    n     = len(z)
    label = np.full(n, -1, dtype=np.int8)
    zone  = np.full(n, "gap", dtype=object)

    # Water
    m_uw = z < Z_UNDERWATER_MAX
    m_ws = (z >= Z_UNDERWATER_MAX) & (z < Z_WATER_SURF_MAX)
    label[m_uw] = 1;  zone[m_uw] = "underwater"
    label[m_ws] = 1;  zone[m_ws] = "water_surface"

    # Land
    m_db = (z >= Z_DRY_BED_MIN) & (z < Z_DRY_BED_MAX)
    m_bk = (z >= Z_BANKS_MIN)   & (z < Z_BANKS_MAX)
    m_ca = z >= Z_CANOPY_MIN
    label[m_db] = 0;  zone[m_db] = "dry_riverbed"
    label[m_bk] = 0;  zone[m_bk] = "river_banks_meadow"
    label[m_ca] = 0;  zone[m_ca] = "canopy"

    return label, zone


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — DIAGNOSTIC: WATER_SURFACE vs DRY_RIVERBED
# ══════════════════════════════════════════════════════════════════════════════

def cohens_d(a, b):
    """Cohen's d: positive = a > b."""
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(a) - np.mean(b)) / (pooled_std + 1e-12)


def run_diagnostic(feat_df, grids_all, z, out_dir):
    print(f"\n{'='*60}")
    print("PHASE 2 — DIAGNOSTIC: WATER_SURFACE vs DRY_RIVERBED")
    print(f"{'='*60}")

    ws_mask = (z >= Z_UNDERWATER_MAX) & (z < Z_WATER_SURF_MAX)
    db_mask = (z >= Z_DRY_BED_MIN)   & (z < Z_DRY_BED_MAX)
    ws_idx  = np.where(ws_mask)[0]
    db_idx  = np.where(db_mask)[0]
    print(f"  WATER_SURFACE  ({Z_UNDERWATER_MAX}–{Z_WATER_SURF_MAX}m): {len(ws_idx):,} points")
    print(f"  DRY_RIVERBED   ({Z_DRY_BED_MIN}–{Z_DRY_BED_MAX}m):  {len(db_idx):,} points")

    rng = np.random.default_rng(42)

    # ── Plot 1: 10 waveform grids each, side by side ───────────────────────────
    ws_sample = rng.choice(ws_idx, size=min(10, len(ws_idx)), replace=False)
    db_sample = rng.choice(db_idx, size=min(10, len(db_idx)), replace=False)

    n_show = min(10, len(ws_sample), len(db_sample))
    bins   = np.arange(200)

    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 2.4, 6),
                              sharey=False, sharex=True)
    fig.suptitle(
        "Waveform Grids: WATER_SURFACE (top) vs DRY_RIVERBED (bottom)\n"
        "200-bin dense grid, amplitude vs time-bin offset",
        fontsize=10)

    ws_max_amp = grids_all[ws_sample].max()
    db_max_amp = grids_all[db_sample].max()
    global_max = max(ws_max_amp, db_max_amp)

    for col in range(n_show):
        wi = ws_sample[col]
        di = db_sample[col]

        axes[0][col].fill_between(bins, grids_all[wi], alpha=0.7, color="steelblue")
        axes[0][col].set_ylim(0, global_max * 1.05)
        axes[0][col].set_title(f"z={z[wi]:.2f}", fontsize=7)
        axes[0][col].tick_params(labelsize=5)
        if col == 0:
            axes[0][col].set_ylabel("WATER\nSURFACE", fontsize=8)

        axes[1][col].fill_between(bins, grids_all[di], alpha=0.7, color="saddlebrown")
        axes[1][col].set_ylim(0, global_max * 1.05)
        axes[1][col].set_title(f"z={z[di]:.2f}", fontsize=7)
        axes[1][col].tick_params(labelsize=5)
        if col == 0:
            axes[1][col].set_ylabel("DRY\nRIVERBED", fontsize=8)
        axes[1][col].set_xlabel("bin", fontsize=6)

    plt.tight_layout()
    p1 = os.path.join(out_dir, "diagnostic_waveforms.png")
    plt.savefig(p1, dpi=150); plt.close()
    print(f"\n  Waveform grid plot → {p1}")

    # ── Plot 2: Reflectance histograms ─────────────────────────────────────────
    ws_refl = feat_df.loc[ws_mask, "reflectance_dB"].values
    db_refl = feat_df.loc[db_mask, "reflectance_dB"].values
    all_refl = np.concatenate([ws_refl, db_refl])
    bins_r   = np.linspace(all_refl.min(), all_refl.max(), 60)
    d_refl   = cohens_d(ws_refl, db_refl)

    print(f"\n  Reflectance (dB):")
    print(f"    WATER_SURFACE : mean={ws_refl.mean():.2f}  std={ws_refl.std():.2f}  "
          f"median={np.median(ws_refl):.2f}")
    print(f"    DRY_RIVERBED  : mean={db_refl.mean():.2f}  std={db_refl.std():.2f}  "
          f"median={np.median(db_refl):.2f}")
    print(f"    Cohen's d     : {d_refl:.3f}  (|d|>0.8=large)")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(ws_refl, bins=bins_r, alpha=0.6, label=f"WATER_SURFACE (n={len(ws_refl):,})",
            color="steelblue", density=True)
    ax.hist(db_refl, bins=bins_r, alpha=0.6, label=f"DRY_RIVERBED (n={len(db_refl):,})",
            color="saddlebrown", density=True)
    ax.axvline(ws_refl.mean(), color="steelblue", lw=1.5, ls="--", alpha=0.8)
    ax.axvline(db_refl.mean(), color="saddlebrown", lw=1.5, ls="--", alpha=0.8)
    ax.set_xlabel("Reflectance (dB)")
    ax.set_ylabel("Density")
    ax.set_title(f"Reflectance distribution  —  Cohen's d = {d_refl:.3f}")
    ax.legend()
    plt.tight_layout()
    p2 = os.path.join(out_dir, "diagnostic_reflectance.png")
    plt.savefig(p2, dpi=150); plt.close()
    print(f"  Reflectance histogram → {p2}")

    # ── Plot 3: Cohen's d for all waveform features ────────────────────────────
    feat_cols = [c for c in WAVEFORM_FEATURES if c in feat_df.columns]
    ws_feat   = feat_df.loc[ws_mask, feat_cols].values.astype(np.float32)
    db_feat   = feat_df.loc[db_mask, feat_cols].values.astype(np.float32)

    ds = []
    print(f"\n  Feature separation (Cohen's d, WATER_SURFACE minus DRY_RIVERBED):")
    print(f"  {'Feature':<35} {'WS mean':>10} {'DB mean':>10} {'d':>8}")
    print(f"  {'-'*65}")
    for i, col in enumerate(feat_cols):
        ws_v = ws_feat[:, i]
        db_v = db_feat[:, i]
        d = cohens_d(ws_v, db_v)
        ds.append((col, d, ws_v.mean(), db_v.mean()))
        print(f"  {col:<35} {ws_v.mean():>10.3f} {db_v.mean():>10.3f} {d:>8.3f}")

    ds_sorted = sorted(ds, key=lambda x: abs(x[1]), reverse=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    names  = [r[0] for r in ds_sorted]
    dvals  = [r[1] for r in ds_sorted]
    colors = ["steelblue" if d > 0 else "saddlebrown" for d in dvals]
    bars   = ax.barh(names[::-1], dvals[::-1], color=colors[::-1])
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline( 0.8, color="grey", lw=0.8, ls="--", alpha=0.5, label="|d|=0.8 (large)")
    ax.axvline(-0.8, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Cohen's d  (positive = WATER_SURFACE > DRY_RIVERBED)")
    ax.set_title("Feature separation: WATER_SURFACE vs DRY_RIVERBED\n"
                 "(blue = water is higher, brown = dry is higher)")
    ax.legend(fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    p3 = os.path.join(out_dir, "diagnostic_cohens_d.png")
    plt.savefig(p3, dpi=150); plt.close()
    print(f"\n  Cohen's d plot → {p3}")

    # Summary
    large_d = [(n, d) for n, d, _, _ in ds_sorted if abs(d) >= 0.8]
    med_d   = [(n, d) for n, d, _, _ in ds_sorted if 0.5 <= abs(d) < 0.8]
    print(f"\n  Large effect (|d|≥0.8): {len(large_d)} features")
    for n, d in large_d:
        print(f"    {n:<35} d={d:.3f}")
    print(f"  Medium effect (0.5≤|d|<0.8): {len(med_d)} features")
    for n, d in med_d:
        print(f"    {n:<35} d={d:.3f}")

    separable = len(large_d) > 0 or len(med_d) > 0
    print(f"\n  SEPARABILITY VERDICT: "
          f"{'SEPARABLE — waveform signal exists' if separable else 'NOT SEPARABLE — fundamental limitation'}")
    return ds_sorted


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3a — XGBOOST
# ══════════════════════════════════════════════════════════════════════════════

def spatial_cv_split(df, n_folds=5):
    y = df["y"].values
    edges = np.percentile(y, np.linspace(0, 100, n_folds + 1))
    return [
        (np.where((y < edges[f]) | (y > edges[f + 1]))[0],
         np.where((y >= edges[f]) & (y <= edges[f + 1]))[0])
        for f in range(n_folds)
    ]


def train_xgb(df_train, out_dir):
    cols = [c for c in WAVEFORM_FEATURES if c in df_train.columns]
    miss = set(WAVEFORM_FEATURES) - set(df_train.columns)
    if miss:
        print(f"  WARNING: missing features: {miss}")

    n_w  = int((df_train["label"] == 1).sum())
    n_l  = int((df_train["label"] == 0).sum())
    spw  = round(n_l / max(n_w, 1), 3)

    print(f"\n{'='*60}")
    print(f"XGBoost  ({len(cols)} waveform features)  "
          f"water={n_w:,}  land={n_l:,}  spw={spw}")

    X = np.nan_to_num(df_train[cols].values.astype(np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    y = df_train["label"].values.astype(np.int32)

    splits  = spatial_cv_split(df_train)
    metrics = {k: [] for k in ["f1", "auc", "pw", "rw", "pl", "rl"]}

    print(f"\n  5-fold spatial CV:")
    for fold, (trn, val) in enumerate(splits):
        if len(val) == 0 or len(np.unique(y[val])) < 2:
            continue
        clf = xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0,
        )
        clf.fit(X[trn], y[trn], eval_set=[(X[val], y[val])], verbose=False)
        pr  = clf.predict_proba(X[val])[:, 1]
        pd_ = (pr >= 0.5).astype(int)
        yv  = y[val]

        metrics["f1"].append(f1_score(yv, pd_, average="macro", zero_division=0))
        metrics["pw"].append(precision_score(yv, pd_, pos_label=1, zero_division=0))
        metrics["rw"].append(recall_score(   yv, pd_, pos_label=1, zero_division=0))
        metrics["pl"].append(precision_score(yv, pd_, pos_label=0, zero_division=0))
        metrics["rl"].append(recall_score(   yv, pd_, pos_label=0, zero_division=0))
        try:    metrics["auc"].append(roc_auc_score(yv, pr))
        except: metrics["auc"].append(float("nan"))

        print(f"    Fold {fold+1}: F1={metrics['f1'][-1]:.3f}  "
              f"AUC={metrics['auc'][-1]:.3f}  "
              f"Water P={metrics['pw'][-1]:.3f} R={metrics['rw'][-1]:.3f}  "
              f"Land  P={metrics['pl'][-1]:.3f} R={metrics['rl'][-1]:.3f}")

    print(f"\n  CV macro-F1 : {np.mean(metrics['f1']):.3f} ± {np.std(metrics['f1']):.3f}")
    print(f"  CV ROC-AUC  : {np.nanmean(metrics['auc']):.3f}")
    print(f"  Water  P={np.mean(metrics['pw']):.3f}  R={np.mean(metrics['rw']):.3f}")
    print(f"  Land   P={np.mean(metrics['pl']):.3f}  R={np.mean(metrics['rl']):.3f}")

    print(f"\n  Training final XGBoost on {len(X):,} rows …")
    final = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="logloss",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    final.fit(X, y, verbose=False)
    mpath = os.path.join(out_dir, "v6_xgb.json")
    final.save_model(mpath)
    print(f"  Model → {mpath}")

    imp = pd.DataFrame({"feature": cols, "importance": final.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    print(f"\n  Top 15 features:")
    print(imp.head(15).to_string(index=False))

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(9, 7))
    top20 = imp.head(20)
    ax.barh(top20["feature"][::-1], top20["importance"][::-1], color="#3498db")
    ax.set_xlabel("XGBoost importance (gain)")
    ax.set_title("v6 Waveform-Only XGBoost — Water vs Land\nFeature Importance")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=150)
    plt.close()

    cv_res = {
        "macro_f1_mean": float(np.mean(metrics["f1"])),
        "macro_f1_std":  float(np.std(metrics["f1"])),
        "auc_mean":      float(np.nanmean(metrics["auc"])),
        "prec_water":    float(np.mean(metrics["pw"])),
        "rec_water":     float(np.mean(metrics["rw"])),
        "prec_land":     float(np.mean(metrics["pl"])),
        "rec_land":      float(np.mean(metrics["rl"])),
        "n_water":       n_w,
        "n_land":        n_l,
        "feature_cols":  cols,
    }
    return final, cols, cv_res


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3b — DEEP MODEL
# ══════════════════════════════════════════════════════════════════════════════

class _CB(nn.Module):
    def __init__(self, i, o, k):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv1d(i, o, k, padding="same", bias=False),
            nn.BatchNorm1d(o), nn.ReLU(inplace=True))
    def forward(self, x): return self.b(x)


class V6Net(nn.Module):
    """
    1D-CNN on waveform grid  +  MLP on waveform scalar features.
    No geometric or elevation inputs anywhere.
    dropout=0.0 — must match inference below.
    """
    def __init__(self, n_spatial):
        super().__init__()
        self.wf = nn.Sequential(
            _CB(1, 32, 3), _CB(32, 64, 5), _CB(64, 64, 11),
            nn.MaxPool1d(4), _CB(64, 128, 5), nn.AdaptiveAvgPool1d(1))
        self.sp = nn.Sequential(
            nn.Linear(n_spatial, 128), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(True),
            nn.Linear(64, 32),   nn.ReLU(True))
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
            st.scatter_(1, targets.unsqueeze(1), 1.0 - self.smooth + self.smooth / n)
        lp   = torch.nn.functional.log_softmax(logits, 1)
        pt   = lp.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = (1 - pt).pow(self.gamma) * (-(st * lp).sum(1))
        if self.alpha is not None:
            at = torch.where(targets == 1,
                             torch.full_like(pt, self.alpha),
                             torch.full_like(pt, 1 - self.alpha))
            loss = at * loss
        return loss.mean()


def train_deep(df_train, grids_all, train_orig_rows, out_dir,
               epochs=80, batch_size=256, lr=1e-3, patience=20,
               val_frac=0.20):
    sp_cols = [c for c in WAVEFORM_FEATURES if c in df_train.columns]

    print(f"\n{'='*60}")
    print(f"Deep V6Net  ({len(sp_cols)} waveform scalar features, no geometry)")

    grids_t   = grids_all[train_orig_rows].copy().astype(np.float32)
    spatial_t = df_train[sp_cols].values.astype(np.float32)
    labels_t  = df_train["label"].values.astype(np.int64)
    y_coord   = df_train["y"].values

    grids_t   = np.nan_to_num(grids_t,   nan=0.0, posinf=0.0, neginf=0.0)
    spatial_t = np.nan_to_num(spatial_t, nan=0.0, posinf=0.0, neginf=0.0)

    cutoff   = np.percentile(y_coord, 100 * (1 - val_frac))
    val_mask = y_coord >= cutoff
    trn_mask = ~val_mask
    print(f"  Train: {trn_mask.sum():,}   Val: {val_mask.sum():,}")

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

    model     = V6Net(n_spatial=len(sp_cols)).to(device)
    criterion = _Focal(gamma=2.0, alpha=alpha, smooth=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=5e-6)
    scaler    = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    nw    = min(4, os.cpu_count() or 1)
    tr_ld = DataLoader(_DS(gn[trn_mask], sn[trn_mask], labels_t[trn_mask]),
                       batch_size=batch_size, shuffle=True,
                       num_workers=nw, pin_memory=True)
    va_ld = DataLoader(_DS(gn[val_mask], sn[val_mask], labels_t[val_mask]),
                       batch_size=batch_size * 2, shuffle=False,
                       num_workers=nw, pin_memory=True)

    print(f"  Parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}  "
          f"alpha={alpha:.3f}")

    best_f1 = 0.0; pat = 0
    hist    = {"tl": [], "vl": [], "f1": [], "auc": []}
    mpath   = os.path.join(out_dir, "v6_deep.pt")

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
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            tl += loss.item() * len(lb)
        tl /= len(tr_ld.dataset)

        model.eval(); vl = 0.0
        preds_, proba_, labs_ = [], [], []
        with torch.no_grad():
            for wf, sp, lb in va_ld:
                wf = wf.to(device, non_blocking=True)
                sp = sp.to(device, non_blocking=True)
                lb = lb.to(device, non_blocking=True)
                lg = model(wf, sp)
                vl += criterion(lg, lb).item() * len(lb)
                pr  = torch.softmax(lg, 1)[:, 1]
                preds_.append(lg.argmax(1).cpu().numpy())
                proba_.append(pr.cpu().numpy())
                labs_.append(lb.cpu().numpy())
        vl    /= len(va_ld.dataset)
        preds  = np.concatenate(preds_)
        proba  = np.concatenate(proba_)
        labs   = np.concatenate(labs_)
        vf1    = f1_score(labs, preds, average="macro", zero_division=0)
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

    # Validation report on best checkpoint
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

    # Confusion within zones (val set only — needs zone info)
    print(f"  (Val set spans top 20% of y-coordinate strip)")

    # Training curve
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    ep_r = range(1, len(hist["tl"]) + 1)
    a1.plot(ep_r, hist["tl"], label="train"); a1.plot(ep_r, hist["vl"], label="val")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Focal Loss"); a1.legend()
    a2.plot(ep_r, hist["f1"], label="macro-F1"); a2.plot(ep_r, hist["auc"], label="AUC")
    a2.set_xlabel("Epoch"); a2.set_ylabel("Score"); a2.legend()
    plt.suptitle("V6Net — Waveform-only water vs land", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curve.png"), dpi=150)
    plt.close()

    stats = {
        "grid_mean":    g_mean,
        "grid_std":     g_std,
        "spatial_mean": sp_mean.tolist(),
        "spatial_std":  sp_std.tolist(),
        "spatial_cols": sp_cols,
        "best_val_f1":  float(best_f1),
    }
    spath = os.path.join(out_dir, "v6_deep_stats.json")
    with open(spath, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"  Model → {mpath}  Stats → {spath}")
    return model, stats, {"best_val_f1": float(best_f1)}


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — INFERENCE + EXPORT
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def deep_infer_all(feat_df, grids_all, stats, model_path, batch_size=2048):
    sp_cols = stats["spatial_cols"]
    g_mean  = float(stats["grid_mean"]); g_std = float(stats["grid_std"])
    sp_mean = np.array(stats["spatial_mean"], np.float32)
    sp_std  = np.array(stats["spatial_std"],  np.float32)

    grids_n = (grids_all.astype(np.float32) - g_mean) / g_std
    grids_n = np.nan_to_num(grids_n, nan=0.0, posinf=0.0, neginf=0.0)

    sp_vals = feat_df[sp_cols].values.astype(np.float32)
    sp_vals = np.nan_to_num(sp_vals, nan=0.0, posinf=0.0, neginf=0.0)
    sn      = (sp_vals - sp_mean) / sp_std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = V6Net(n_spatial=len(sp_cols))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()

    N = len(feat_df); probas = np.zeros(N, np.float32)
    for s in range(0, N, batch_size):
        e   = min(s + batch_size, N)
        wfb = torch.from_numpy(grids_n[s:e]).unsqueeze(1).to(device)
        spb = torch.from_numpy(sn[s:e]).to(device)
        probas[s:e] = torch.softmax(model(wfb, spb), 1)[:, 1].cpu().numpy()
        if s % 50_000 == 0 and s > 0:
            print(f"    deep: {s:>6,}/{N:,}")
    return probas


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PC_OUT.parent, exist_ok=True)

    # ── Load ───────────────────────────────────────────────────────────────────
    print(f"Loading {FEAT_PATH} …")
    feat_df = pd.read_csv(FEAT_PATH)
    N = len(feat_df)
    print(f"  {N:,} points × {len(feat_df.columns)} cols")

    print(f"Loading {GRIDS_PATH} …")
    grids_all = np.load(GRIDS_PATH, mmap_mode="r")
    print(f"  grids shape: {grids_all.shape}")

    z = feat_df["z"].values

    # ── PHASE 1: Labeling ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 1 — LABELING")
    print(f"{'='*60}")

    label, zone = create_labels(z)

    zone_order = ["underwater", "water_surface", "gap",
                  "dry_riverbed", "river_banks_meadow", "canopy"]

    print(f"\nZone breakdown ({N:,} total points):")
    print(f"  {'Zone':<28} {'Count':>8}  {'%':>6}  Label")
    print(f"  {'-'*52}")
    for zn in zone_order:
        m   = zone == zn
        cnt = m.sum()
        lv  = {-1: "excluded", 0: "land (0)", 1: "water (1)"}.get(
               int(np.unique(label[m])[0]) if m.any() else -99, "?")
        print(f"  {zn:<28} {cnt:>8,}  {100*cnt/N:>5.1f}%  {lv}")

    n_w   = int((label == 1).sum())
    n_l   = int((label == 0).sum())
    n_unk = int((label == -1).sum())
    print(f"\n  Total water (1): {n_w:>7,}  ({100*n_w/N:.1f}%)")
    print(f"  Total land  (0): {n_l:>7,}  ({100*n_l/N:.1f}%)")
    print(f"  Excluded   (-1): {n_unk:>7,}  ({100*n_unk/N:.1f}%)")
    print(f"  Class imbalance: 1:{n_l/max(n_w,1):.1f}  (land:water)")

    lab_df = pd.DataFrame({
        "label": label, "zone": zone,
        "x": feat_df["x"].values, "y": feat_df["y"].values, "z": z,
    })
    lab_df.to_csv(LABEL_OUT, index=False)
    print(f"\nLabels → {LABEL_OUT}")

    # ── PHASE 2: Diagnostic ────────────────────────────────────────────────────
    ds_sorted = run_diagnostic(feat_df, grids_all, z, MODEL_DIR)

    # ── PHASE 3: Training ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 3 — TRAINING (waveform + reflectance features only)")
    print(f"{'='*60}")

    train_mask = label != -1
    df_train   = feat_df[train_mask].copy()
    df_train["label"] = label[train_mask]
    df_train["y"]     = feat_df["y"].values[train_mask]
    train_orig_rows   = np.where(train_mask)[0]

    print(f"\nTraining set: {len(df_train):,} points  "
          f"(water={n_w:,}  land={n_l:,})")

    xgb_model, xgb_cols, xgb_cv = train_xgb(df_train, MODEL_DIR)

    deep_model, deep_stats, deep_cv = train_deep(
        df_train, grids_all, train_orig_rows, MODEL_DIR)

    # ── PHASE 4: Inference on all 234k points ─────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 4 — INFERENCE + EXPORT")
    print(f"{'='*60}")

    # XGBoost
    xgb_cols_avail = [c for c in xgb_cols if c in feat_df.columns]
    X_all     = np.nan_to_num(feat_df[xgb_cols_avail].values.astype(np.float32),
                               nan=0.0, posinf=0.0, neginf=0.0)
    xgb_m     = xgb.XGBClassifier()
    xgb_m.load_model(os.path.join(MODEL_DIR, "v6_xgb.json"))
    xgb_proba = xgb_m.predict_proba(X_all)[:, 1]
    xgb_pred  = (xgb_proba >= 0.5).astype(np.int8)

    print(f"\nXGBoost: water={int(xgb_pred.sum()):,}  "
          f"land={int((xgb_pred==0).sum()):,}")

    # Deep
    deep_proba = deep_infer_all(feat_df, grids_all, deep_stats,
                                os.path.join(MODEL_DIR, "v6_deep.pt"))
    deep_pred  = (deep_proba >= 0.5).astype(np.int8)
    print(f"Deep   : water={int(deep_pred.sum()):,}  "
          f"land={int((deep_pred==0).sum()):,}")

    # Ensemble
    agree    = xgb_pred == deep_pred
    ensemble = xgb_pred.copy()
    ensemble[~agree] = 2    # 2 = models disagree
    print(f"Ensemble agreement: {100*agree.mean():.1f}%  "
          f"({int((~agree).sum()):,} uncertain)")

    # Per-zone accuracy on labeled points (ground truth check)
    print(f"\n  Per-zone XGBoost accuracy (labeled points only):")
    print(f"  {'Zone':<28} {'N':>7}  {'True label':>10}  {'XGB water%':>10}  {'Correct%':>9}")
    print(f"  {'-'*68}")
    for zn in ["underwater", "water_surface", "dry_riverbed",
               "river_banks_meadow", "canopy"]:
        m   = zone == zn
        if not m.any():
            continue
        true_lv  = int(np.unique(label[m])[0])
        xpred_w  = xgb_pred[m].mean() * 100
        if true_lv == 1:
            correct = (xgb_pred[m] == 1).mean() * 100
        else:
            correct = (xgb_pred[m] == 0).mean() * 100
        print(f"  {zn:<28} {m.sum():>7,}  {'water' if true_lv==1 else 'land':>10}  "
              f"{xpred_w:>9.1f}%  {correct:>8.1f}%")

    # ── Export ─────────────────────────────────────────────────────────────────
    out = pd.DataFrame({
        "x":             feat_df["x"].values,
        "y":             feat_df["y"].values,
        "z":             feat_df["z"].values,
        "reflectance_dB": feat_df["reflectance_dB"].values,
        "zone":          zone,
        "true_label":    label,
        "xgb_pred":      xgb_pred,
        "xgb_proba":     np.round(xgb_proba, 4),
        "deep_pred":     deep_pred,
        "deep_proba":    np.round(deep_proba, 4),
        "ensemble":      ensemble,
    })
    for col in ["n_clusters", "n_peaks", "n_gaps", "energy_concentration",
                "max_amp_norm_by_energy", "amplitude_weighted_center", "reflectance_dB"]:
        if col in feat_df.columns and col not in out.columns:
            out[col] = feat_df[col].values

    out.to_csv(PC_OUT, index=False)
    print(f"\nSaved {N:,} rows → {PC_OUT}")

    # ── Top-down scatter: predictions ─────────────────────────────────────────
    x_all = feat_df["x"].values
    y_all = feat_df["y"].values
    fig, axes = plt.subplots(1, 2, figsize=(22, 8))
    c_map = {0: "saddlebrown", 1: "steelblue", 2: "gold"}
    l_map = {0: "Land", 1: "Water", 2: "Uncertain"}

    for ax, (pred_arr, title) in zip(axes, [
            (xgb_pred, "XGBoost (waveform-only)"),
            (ensemble,  "Ensemble")]):
        for pv in [0, 1, 2]:
            m = pred_arr == pv
            if not m.any():
                continue
            ax.scatter(x_all[m], y_all[m], c=c_map[pv], s=0.4,
                       label=f"{l_map[pv]} ({m.sum():,})",
                       rasterized=True, alpha=0.8)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title(f"v6 Waveform-Only — {title}")
        ax.legend(markerscale=6, fontsize=9, loc="upper right")

    plt.suptitle("Water vs Land predictions — NO elevation, NO geometry features",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "xy_scatter_predictions.png"), dpi=150)
    plt.close()
    print(f"Scatter → {MODEL_DIR}/xy_scatter_predictions.png")

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    print(f"\n  Training labels: {len(df_train):,}  "
          f"water={n_w:,}  land={n_l:,}  excluded={n_unk:,}")

    print(f"\n  XGBoost (waveform-only) — 5-fold spatial CV:")
    print(f"    macro-F1 : {xgb_cv['macro_f1_mean']:.3f} ± {xgb_cv['macro_f1_std']:.3f}")
    print(f"    ROC-AUC  : {xgb_cv['auc_mean']:.3f}")
    print(f"    Water P={xgb_cv['prec_water']:.3f}  R={xgb_cv['rec_water']:.3f}")
    print(f"    Land  P={xgb_cv['prec_land']:.3f}  R={xgb_cv['rec_land']:.3f}")

    print(f"\n  Deep V6Net (waveform-only):")
    print(f"    Best val macro-F1: {deep_cv['best_val_f1']:.3f}")

    print(f"\n  Ensemble on all {N:,} points:")
    print(f"    Water (1)     : {int((ensemble==1).sum()):>7,}")
    print(f"    Land  (0)     : {int((ensemble==0).sum()):>7,}")
    print(f"    Uncertain (2) : {int((ensemble==2).sum()):>7,}")

    # Top diagnostic features
    print(f"\n  Top 5 separating features (Cohen's d, water_surface vs dry_riverbed):")
    for name, d, ws_m, db_m in ds_sorted[:5]:
        print(f"    {name:<35} d={d:+.3f}  "
              f"WS={ws_m:.3f}  DB={db_m:.3f}")

    metrics = {
        "labeling": {
            "n_water": n_w, "n_land": n_l, "n_excluded": n_unk,
            "zones": {
                "underwater":         int((zone == "underwater").sum()),
                "water_surface":      int((zone == "water_surface").sum()),
                "dry_riverbed":       int((zone == "dry_riverbed").sum()),
                "river_banks_meadow": int((zone == "river_banks_meadow").sum()),
                "canopy":             int((zone == "canopy").sum()),
                "gap":                int((zone == "gap").sum()),
            }
        },
        "diagnostic": {
            "top5_features": [
                {"name": n, "cohens_d": float(d),
                 "water_surf_mean": float(wm), "dry_bed_mean": float(dm)}
                for n, d, wm, dm in ds_sorted[:5]
            ],
            "large_effect_count": int(sum(1 for _, d, _, _ in ds_sorted if abs(d) >= 0.8)),
        },
        "xgb_cv": xgb_cv,
        "deep_val": deep_cv,
        "inference": {
            "xgb_water":          int(xgb_pred.sum()),
            "deep_water":         int(deep_pred.sum()),
            "ensemble_water":     int((ensemble == 1).sum()),
            "ensemble_uncertain": int((ensemble == 2).sum()),
            "agreement_pct":      float(100 * agree.mean()),
        },
    }
    with open(os.path.join(MODEL_DIR, "v6_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nMetrics → {MODEL_DIR}/v6_metrics.json")

    print(f"\nOpen {PC_OUT} in CloudCompare:")
    print(f"  Colour by 'xgb_pred' or 'ensemble'")
    print(f"  0=land  1=water  2=uncertain")
    print(f"  Also check 'zone' (string) to verify anchor correctness")
    print("\nDone.")


if __name__ == "__main__":
    main()
