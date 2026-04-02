"""
auto_labeler_v5.py — Score-based labeling + binary water/land classifier.

Pipeline (all in one script):
  1. LABEL   — z-boundary anchors + 4-signal score for uncertain zone
  2. TRAIN   — XGBoost (spatial CV) + 1D-CNN/MLP deep model
  3. INFER   — apply to all 234k points
  4. EXPORT  — pointclouds/labeled_pointcloud_v5.csv + plots + metrics

Verified z-boundaries (CloudCompare visual inspection):
  z < 261.1m   → CERTAIN_WATER  (riverbed / below surface)
  z > 263.1m   → CERTAIN_CANOPY (tree crowns)
  261.1–263.1m → UNCERTAIN      (banks, gravel, possible water surface)

Scoring signals (derived from diagnostic analysis of waveform cluster data):
  reflectance_dB   — strongest signal; water peaks at ≈-26 dB, dry at ≈-12 dB
  n_clusters       — water: 67% single-cluster; dry: 41% have 3+ clusters
  max_amp          — water avg 1913 ADC; uncertain zone avg 2684 ADC
  n_samples        — water avg 27.6 samples; uncertain zone avg 38.0

Labels written to:
  data_processed/labels_v5.csv

Models saved to:
  models/v5-rule-scored/v5_xgb.json
  models/v5-rule-scored/v5_deep.pt
  models/v5-rule-scored/v5_deep_stats.json

Outputs:
  pointclouds/labeled_pointcloud_v5.csv
  models/v5-rule-scored/xy_scatter_labels.png
  models/v5-rule-scored/xy_scatter_predictions.png
  models/v5-rule-scored/feature_importance.png
  models/v5-rule-scored/training_curve.png
  models/v5-rule-scored/v5_metrics.json
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
LABEL_OUT   = ROOT / "data_processed" / "labels_v5.csv"
PC_OUT      = ROOT / "pointclouds"    / "labeled_pointcloud_v5.csv"
MODEL_DIR   = ROOT / "models"         / "v5-rule-scored"

# ── Z-boundary anchors (verified by visual inspection in CloudCompare) ─────────
Z_WATER_MAX  = 261.1   # z < this  → CERTAIN_WATER
Z_CANOPY_MIN = 263.1   # z > this  → CERTAIN_CANOPY
# 261.1 ≤ z ≤ 263.1    → UNCERTAIN  (classified by scoring below)

# ── Scoring thresholds (from diagnostic_transition_zone.py analysis) ──────────
# Reflectance: strongest discriminator
REFL_WATER_STRONG  = -20.0  # dB below this  → strong water (+0.25)
REFL_WATER_MILD    = -17.0  # dB below this  → mild water (+0.10)
REFL_LAND_MILD     = -17.0  # dB above this  → mild land  (-0.10)
REFL_LAND_STRONG   = -14.0  # dB above this  → strong land (-0.25)

# n_clusters: 1=compact/water-like, 3+=complex/dry-like
N_CLUST_WATER_1    = 1      # single cluster → +0.15
N_CLUST_WATER_2    = 2      # two clusters   → +0.05
N_CLUST_LAND       = 4      # 4+ clusters    → -0.15

# max_amp: water avg 1913, uncertain avg 2684
AMP_WATER_MAX      = 1500   # < 1500  → +0.10
AMP_LAND_MIN       = 3000   # > 3000  → -0.10

# n_samples: water avg 27.6, uncertain avg 38.0
SAMPLES_WATER_MAX  = 25     # < 25    → +0.05
SAMPLES_LAND_MIN   = 40     # > 40    → -0.05

# Score decision boundary
SCORE_WATER_THRESH = 0.60   # ≥ this  → label as water
SCORE_LAND_THRESH  = 0.40   # ≤ this  → label as land
# Between LAND_THRESH and WATER_THRESH → excluded from training

# ── Feature sets ──────────────────────────────────────────────────────────────
# NO absolute z. Both models see exactly these features.
XGB_FEATURES = [
    # Top diagnostic discriminators
    "energy_concentration",
    "max_amp_norm_by_energy",
    # Waveform structure
    "n_clusters", "n_peaks", "n_gaps", "n_samples", "time_span",
    "max_amp", "mean_amp", "std_amp", "total_energy",
    "max_gap", "mean_gap", "total_gap", "first_last_span",
    "first_peak_amp", "last_peak_amp", "peak_amp_ratio",
    "energy_ratio_late", "depth_proxy_m",
    "amplitude_weighted_center", "active_bins_ratio",
    # Scalar signal
    "reflectance_dB",
    # Local geometry
    "planarity", "roughness", "linearity", "sphericity",
    "height_range_local", "height_std_local",
    # Relative elevation (generalises across sites)
    "height_above_local_min",
    "height_above_local_min_10m",
    "height_percentile_local",
    "z_relative",
]

# Spatial feature subset for the deep model's MLP branch (most discriminative)
DEEP_SPATIAL = [
    "energy_concentration",
    "max_amp_norm_by_energy",
    "reflectance_dB",
    "planarity",
    "roughness",
    "height_range_local",
    "height_percentile_local",
    "amplitude_weighted_center",
    "active_bins_ratio",
    "z_relative",
    "n_clusters",
    "n_samples",
]


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — LABELING
# ══════════════════════════════════════════════════════════════════════════════

def score_uncertain_point(refl, n_clusters, max_amp, n_samples):
    """Return a water-likelihood score in [0, 1]. 0.5 = neutral."""
    score = 0.5

    # Reflectance (strongest signal)
    if refl < REFL_WATER_STRONG:
        score += 0.25
    elif refl < REFL_WATER_MILD:
        score += 0.10
    elif refl > REFL_LAND_STRONG:
        score -= 0.25
    elif refl > REFL_LAND_MILD:
        score -= 0.10

    # Waveform simplicity (cluster count)
    if n_clusters == N_CLUST_WATER_1:
        score += 0.15
    elif n_clusters == N_CLUST_WATER_2:
        score += 0.05
    elif n_clusters >= N_CLUST_LAND:
        score -= 0.15

    # Amplitude (water = dimmer)
    if max_amp < AMP_WATER_MAX:
        score += 0.10
    elif max_amp > AMP_LAND_MIN:
        score -= 0.10

    # Waveform compactness
    if n_samples < SAMPLES_WATER_MAX:
        score += 0.05
    elif n_samples > SAMPLES_LAND_MIN:
        score -= 0.05

    return float(np.clip(score, 0.0, 1.0))


def create_labels(feat_df):
    """
    Assign binary labels (1=water, 0=land, -1=excluded) to all points.

    Returns a DataFrame with columns:
      label    : 1, 0, or -1
      v5_score : float (scoring result; NaN for z-anchor points)
      anchor   : 'certain_water' | 'certain_canopy' | 'scored_water' |
                 'scored_land' | 'ambiguous'
    """
    n = len(feat_df)
    z     = feat_df["z"].values
    refl  = feat_df["reflectance_dB"].values
    nc    = feat_df["n_clusters"].values
    ma    = feat_df["max_amp"].values
    ns    = feat_df["n_samples"].values

    label   = np.full(n, -1, dtype=np.int8)
    score   = np.full(n, np.nan)
    anchor  = np.full(n, "", dtype=object)

    mask_cw = z < Z_WATER_MAX
    mask_cc = z > Z_CANOPY_MIN
    mask_uc = ~mask_cw & ~mask_cc

    # Certain zones
    label[mask_cw]  = 1;  anchor[mask_cw] = "certain_water"
    label[mask_cc]  = 0;  anchor[mask_cc] = "certain_canopy"

    # Uncertain zone — score each point
    uc_idx = np.where(mask_uc)[0]
    for i in uc_idx:
        s = score_uncertain_point(refl[i], nc[i], ma[i], ns[i])
        score[i] = s
        if s >= SCORE_WATER_THRESH:
            label[i]  = 1;  anchor[i] = "scored_water"
        elif s <= SCORE_LAND_THRESH:
            label[i]  = 0;  anchor[i] = "scored_land"
        else:
            label[i]  = -1; anchor[i] = "ambiguous"

    out = pd.DataFrame({
        "label":   label,
        "v5_score": np.round(score, 4),
        "anchor":  anchor,
        "x": feat_df["x"].values,
        "y": feat_df["y"].values,
        "z": z,
    })
    return out


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2a — XGBOOST
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
    cols = [c for c in XGB_FEATURES if c in df_train.columns]
    miss = [c for c in XGB_FEATURES if c not in df_train.columns]
    if miss:
        print(f"  WARNING: missing XGB features: {miss}")

    n_w  = int((df_train["label"] == 1).sum())
    n_l  = int((df_train["label"] == 0).sum())
    spw  = round(n_l / max(n_w, 1), 3)

    print(f"\n{'='*60}")
    print(f"XGBoost  ({len(cols)} features)  water={n_w:,}  land={n_l:,}  spw={spw}")

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
    model_path = os.path.join(out_dir, "v5_xgb.json")
    final.save_model(model_path)
    print(f"  Model → {model_path}")

    imp = pd.DataFrame({"feature": cols, "importance": final.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    print(f"\n  Top 15 features:")
    print(imp.head(15).to_string(index=False))

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(9, 8))
    SCORED_FEATS = {"reflectance_dB", "n_clusters", "max_amp", "n_samples"}
    top20 = imp.head(20)
    colors = ["#e74c3c" if f in SCORED_FEATS else "#3498db"
              for f in top20["feature"][::-1]]
    ax.barh(top20["feature"][::-1], top20["importance"][::-1], color=colors)
    ax.set_xlabel("XGBoost importance (gain)")
    ax.set_title("v5 XGBoost — Water vs Land\nFeature Importance  "
                 "(red = scoring features)")
    ax.tick_params(axis="y", labelsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#e74c3c", label="Scoring features"),
        Patch(color="#3498db", label="Other"),
    ], fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=150)
    plt.close()
    print(f"  Plot → {out_dir}/feature_importance.png")

    cv_res = {
        "macro_f1_mean":  float(np.mean(metrics["f1"])),
        "macro_f1_std":   float(np.std(metrics["f1"])),
        "auc_mean":       float(np.nanmean(metrics["auc"])),
        "prec_water":     float(np.mean(metrics["pw"])),
        "rec_water":      float(np.mean(metrics["rw"])),
        "prec_land":      float(np.mean(metrics["pl"])),
        "rec_land":       float(np.mean(metrics["rl"])),
        "n_water":        n_w,
        "n_land":         n_l,
        "feature_cols":   cols,
    }
    return final, cols, cv_res


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2b — DEEP MODEL
# ══════════════════════════════════════════════════════════════════════════════

class _CB(nn.Module):
    def __init__(self, i, o, k):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv1d(i, o, k, padding="same", bias=False),
            nn.BatchNorm1d(o), nn.ReLU(inplace=True))
    def forward(self, x): return self.b(x)


class V5Net(nn.Module):
    """
    1D-CNN branch on waveform grid  +  MLP branch on spatial features.
    No Dropout (dropout=0.0) — must match inference section below.
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


class _FlatDS(Dataset):
    def __init__(self, g, s, l):
        self.g = torch.from_numpy(g).unsqueeze(1)
        self.s = torch.from_numpy(s)
        self.l = torch.from_numpy(l)
    def __len__(self): return len(self.l)
    def __getitem__(self, i): return self.g[i], self.s[i], self.l[i]


class _FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, smooth=0.1):
        super().__init__()
        self.gamma = gamma; self.alpha = alpha; self.smooth = smooth

    def forward(self, logits, targets):
        n = logits.size(1)
        with torch.no_grad():
            st = torch.zeros_like(logits).fill_(self.smooth / n)
            st.scatter_(1, targets.unsqueeze(1), 1.0 - self.smooth + self.smooth / n)
        lp  = torch.nn.functional.log_softmax(logits, 1)
        pt  = lp.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        ce  = -(st * lp).sum(1)
        loss = (1 - pt).pow(self.gamma) * ce
        if self.alpha is not None:
            at = torch.where(targets == 1,
                             torch.full_like(pt, self.alpha),
                             torch.full_like(pt, 1 - self.alpha))
            loss = at * loss
        return loss.mean()


def train_deep(df_train, grids_all, train_orig_rows, out_dir,
               epochs=80, batch_size=256, lr=1e-3, patience=20,
               val_frac=0.20):

    sp_cols = [c for c in DEEP_SPATIAL if c in df_train.columns]
    miss = [c for c in DEEP_SPATIAL if c not in df_train.columns]
    if miss:
        print(f"  WARNING: missing deep spatial cols: {miss}")

    print(f"\n{'='*60}")
    print(f"Deep V5Net  ({len(sp_cols)} spatial features)")

    grids_t   = grids_all[train_orig_rows].copy().astype(np.float32)
    spatial_t = df_train[sp_cols].values.astype(np.float32)
    labels_t  = df_train["label"].values.astype(np.int64)
    y_t       = df_train["y"].values

    grids_t   = np.nan_to_num(grids_t,   nan=0.0, posinf=0.0, neginf=0.0)
    spatial_t = np.nan_to_num(spatial_t, nan=0.0, posinf=0.0, neginf=0.0)

    cutoff   = np.percentile(y_t, 100 * (1 - val_frac))
    val_mask = y_t >= cutoff
    trn_mask = ~val_mask
    print(f"  Train: {trn_mask.sum():,}   Val: {val_mask.sum():,}")

    g_mean  = float(grids_t[trn_mask].mean())
    g_std   = float(grids_t[trn_mask].std()) + 1e-6
    sp_mean = spatial_t[trn_mask].mean(0)
    sp_std  = spatial_t[trn_mask].std(0) + 1e-6

    gn = (grids_t   - g_mean) / g_std
    sn = (spatial_t - sp_mean) / sp_std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    n_pos  = int((labels_t[trn_mask] == 1).sum())
    n_neg  = int(trn_mask.sum()) - n_pos
    alpha  = round(n_neg / (n_pos + n_neg), 4)

    model     = V5Net(n_spatial=len(sp_cols)).to(device)
    criterion = _FocalLoss(gamma=2.0, alpha=alpha, smooth=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=5e-6)
    scaler    = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    nw    = min(4, os.cpu_count() or 1)
    tr_ld = DataLoader(_FlatDS(gn[trn_mask], sn[trn_mask], labels_t[trn_mask]),
                       batch_size=batch_size, shuffle=True,
                       num_workers=nw, pin_memory=True)
    va_ld = DataLoader(_FlatDS(gn[val_mask], sn[val_mask], labels_t[val_mask]),
                       batch_size=batch_size * 2, shuffle=False,
                       num_workers=nw, pin_memory=True)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}  "
          f"alpha={alpha:.3f}")

    best_f1 = 0.0; pat = 0
    hist = {"tl": [], "vl": [], "f1": [], "auc": []}
    mpath = os.path.join(out_dir, "v5_deep.pt")

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
                pr = torch.softmax(lg, 1)[:, 1]
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

    # Final val report on best checkpoint
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
    plt.suptitle("V5Net — water vs land  (score-labeled training set)", fontsize=10)
    plt.tight_layout()
    cp = os.path.join(out_dir, "training_curve.png")
    plt.savefig(cp, dpi=150); plt.close()
    print(f"  Curve → {cp}")

    stats = {
        "grid_mean":    g_mean,
        "grid_std":     g_std,
        "spatial_mean": sp_mean.tolist(),
        "spatial_std":  sp_std.tolist(),
        "spatial_cols": sp_cols,
        "best_val_f1":  float(best_f1),
    }
    spath = os.path.join(out_dir, "v5_deep_stats.json")
    with open(spath, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"  Model → {mpath}  Stats → {spath}")
    return model, stats, {"best_val_f1": float(best_f1)}


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — INFERENCE ON ALL 234k POINTS
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def deep_predict_all(feat_df, grids_all, stats, model_path, batch_size=2048):
    sp_cols = stats["spatial_cols"]
    g_mean  = float(stats["grid_mean"]); g_std = float(stats["grid_std"])
    sp_mean = np.array(stats["spatial_mean"], np.float32)
    sp_std  = np.array(stats["spatial_std"],  np.float32)

    grids_n  = (grids_all.astype(np.float32) - g_mean) / g_std
    grids_n  = np.nan_to_num(grids_n, nan=0.0, posinf=0.0, neginf=0.0)

    sp_vals  = feat_df[sp_cols].values.astype(np.float32)
    sp_vals  = np.nan_to_num(sp_vals, nan=0.0, posinf=0.0, neginf=0.0)
    sn       = (sp_vals - sp_mean) / sp_std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = V5Net(n_spatial=len(sp_cols))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()

    N = len(feat_df); probas = np.zeros(N, np.float32)
    for s in range(0, N, batch_size):
        e   = min(s + batch_size, N)
        wfb = torch.from_numpy(grids_n[s:e]).unsqueeze(1).to(device)
        spb = torch.from_numpy(sn[s:e]).to(device)
        probas[s:e] = torch.softmax(model(wfb, spb), 1)[:, 1].cpu().numpy()
        if s % 50_000 == 0 and s > 0:
            print(f"    deep inference: {s:>6,}/{N:,}")
    return probas


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — EXPORT + PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def scatter_by_label(x, y, c_int, cmap, vmin, vmax, title, path, labels_map):
    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(x, y, c=c_int, cmap=cmap, vmin=vmin, vmax=vmax,
                    s=0.4, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(title)

    from matplotlib.patches import Patch
    handles = [Patch(color=plt.get_cmap(cmap)((v - vmin) / (vmax - vmin)),
                     label=lbl) for v, lbl in labels_map.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PC_OUT.parent, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading {FEAT_PATH} …")
    feat_df = pd.read_csv(FEAT_PATH)
    N = len(feat_df)
    print(f"  {N:,} points × {len(feat_df.columns)} cols")

    print(f"Loading {GRIDS_PATH} …")
    grids_all = np.load(GRIDS_PATH, mmap_mode="r")
    print(f"  grids shape: {grids_all.shape}")

    # ── PHASE 1: Labeling ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 1 — LABELING")
    print(f"{'='*60}")
    lab_df = create_labels(feat_df)

    n_w   = int((lab_df["label"] == 1).sum())
    n_l   = int((lab_df["label"] == 0).sum())
    n_unk = int((lab_df["label"] == -1).sum())
    print(f"\nLabel distribution (all {N:,} points):")
    print(f"  Water   (1): {n_w:>7,}  ({100*n_w/N:.1f}%)")
    print(f"  Land    (0): {n_l:>7,}  ({100*n_l/N:.1f}%)")
    print(f"  Excluded(-1):{n_unk:>7,}  ({100*n_unk/N:.1f}%)")

    # Breakdown of uncertain zone scoring
    uc_mask = (feat_df["z"].values >= 261.1) & (feat_df["z"].values <= 263.1)
    uc_labels = lab_df.loc[uc_mask, "label"]
    print(f"\n  Uncertain zone (261.1–263.1m, {uc_mask.sum():,} points) breakdown:")
    print(f"    Scored WATER  (score≥{SCORE_WATER_THRESH}): "
          f"{int((uc_labels==1).sum()):>6,}")
    print(f"    Scored LAND   (score≤{SCORE_LAND_THRESH}): "
          f"{int((uc_labels==0).sum()):>6,}")
    print(f"    AMBIGUOUS     (0.4<s<0.6):  "
          f"{int((uc_labels==-1).sum()):>6,}  ← excluded from training")

    lab_df.to_csv(LABEL_OUT, index=False)
    print(f"\nLabels → {LABEL_OUT}")

    # ── Scatter plot: auto-labels ──────────────────────────────────────────────
    label_colors = {-1: "grey", 0: "saddlebrown", 1: "steelblue"}
    fig, ax = plt.subplots(figsize=(12, 8))
    for lv, col in label_colors.items():
        m = lab_df["label"].values == lv
        lbl = {-1: "Ambiguous/excluded", 0: "Land", 1: "Water"}[lv]
        ax.scatter(feat_df.loc[m, "x"], feat_df.loc[m, "y"],
                   c=col, s=0.4, label=lbl, rasterized=True, alpha=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"v5 Auto-labels (z-anchors + scoring)  —  "
                 f"water={n_w:,}  land={n_l:,}  excluded={n_unk:,}")
    ax.legend(markerscale=6, fontsize=9, loc="upper right")
    plt.tight_layout()
    lp = os.path.join(MODEL_DIR, "xy_scatter_labels.png")
    plt.savefig(lp, dpi=150); plt.close()
    print(f"Label scatter → {lp}")

    # ── PHASE 2: Build training set ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 2 — TRAINING")
    print(f"{'='*60}")

    train_mask = lab_df["label"].isin([0, 1])
    df_train   = feat_df[train_mask].copy()
    df_train["label"]    = lab_df.loc[train_mask, "label"].values
    df_train["y"]        = feat_df.loc[train_mask, "y"].values
    train_orig_rows      = np.where(train_mask.values)[0]

    print(f"\nTraining set: {len(df_train):,} points  "
          f"(water={int((df_train['label']==1).sum()):,}  "
          f"land={int((df_train['label']==0).sum()):,})")

    # ── Train XGBoost ──────────────────────────────────────────────────────────
    xgb_model, xgb_cols, xgb_cv = train_xgb(df_train, MODEL_DIR)

    # ── Train Deep model ───────────────────────────────────────────────────────
    deep_model, deep_stats, deep_cv = train_deep(
        df_train, grids_all, train_orig_rows, MODEL_DIR)

    # ── PHASE 3: Inference on all 234k points ─────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 3 — INFERENCE ON ALL POINTS")
    print(f"{'='*60}")

    # XGBoost inference
    xgb_cols_avail = [c for c in xgb_cols if c in feat_df.columns]
    X_all = np.nan_to_num(feat_df[xgb_cols_avail].values.astype(np.float32),
                          nan=0.0, posinf=0.0, neginf=0.0)
    xgb_m = xgb.XGBClassifier()
    xgb_m.load_model(os.path.join(MODEL_DIR, "v5_xgb.json"))
    xgb_proba = xgb_m.predict_proba(X_all)[:, 1]
    xgb_pred  = (xgb_proba >= 0.5).astype(np.int8)

    print(f"\nXGBoost predictions:")
    print(f"  Water (1): {int(xgb_pred.sum()):>7,}  ({100*xgb_pred.mean():.1f}%)")
    print(f"  Land  (0): {int((xgb_pred==0).sum()):>7,}  ({100*(1-xgb_pred.mean()):.1f}%)")

    # Deep inference
    print(f"\nDeep V5Net predictions:")
    deep_proba = deep_predict_all(feat_df, grids_all, deep_stats,
                                  os.path.join(MODEL_DIR, "v5_deep.pt"))
    deep_pred  = (deep_proba >= 0.5).astype(np.int8)
    print(f"  Water (1): {int(deep_pred.sum()):>7,}  ({100*deep_pred.mean():.1f}%)")
    print(f"  Land  (0): {int((deep_pred==0).sum()):>7,}  ({100*(1-deep_pred.mean()):.1f}%)")

    # Ensemble: agree → that prediction; disagree → uncertain (2)
    agree    = xgb_pred == deep_pred
    ensemble = xgb_pred.copy()
    ensemble[~agree] = 2    # 2 = uncertain / models disagree
    n_unc_ens = int((ensemble == 2).sum())
    print(f"\nEnsemble agreement: {int(agree.sum()):,}/{N:,} "
          f"= {100*agree.mean():.1f}%  (disagreements={n_unc_ens:,})")

    # ── PHASE 4: Export ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 4 — EXPORT + PLOTS")
    print(f"{'='*60}")

    refl_col = "reflectance_dB" if "reflectance_dB" in feat_df.columns \
               else "_riegl.reflectance"

    out = pd.DataFrame({
        "x":             feat_df["x"].values,
        "y":             feat_df["y"].values,
        "z":             feat_df["z"].values,
        "reflectance_dB": feat_df[refl_col].values,
        # Auto-labels (0=land, 1=water, -1=ambiguous)
        "label_v5":      lab_df["label"].values,
        "v5_score":      lab_df["v5_score"].values,
        "anchor":        lab_df["anchor"].values,
        # Model predictions (0=land, 1=water)
        "xgb_pred":      xgb_pred,
        "xgb_proba":     np.round(xgb_proba, 4),
        "deep_pred":     deep_pred,
        "deep_proba":    np.round(deep_proba, 4),
        # Ensemble (0=land, 1=water, 2=uncertain)
        "ensemble":      ensemble,
    })

    # Append key scalar fields for CloudCompare
    for col in ["n_gaps", "max_gap", "n_peaks", "n_clusters",
                "energy_concentration", "max_amp_norm_by_energy",
                "planarity", "roughness", "height_percentile_local",
                "amplitude_weighted_center", "z_relative",
                "height_above_local_min", "height_above_local_min_10m"]:
        if col in feat_df.columns:
            out[col] = feat_df[col].values

    out.to_csv(PC_OUT, index=False)
    print(f"\nSaved {N:,} rows → {PC_OUT}")

    # ── Scatter: XGBoost predictions ──────────────────────────────────────────
    pred_colors = {0: "saddlebrown", 1: "steelblue", 2: "gold"}
    fig, axes = plt.subplots(1, 2, figsize=(22, 8))
    for ax, (pred_arr, title_lbl) in zip(
            axes,
            [(xgb_pred,  "XGBoost"), (ensemble, "Ensemble")]):
        for pv, col in pred_colors.items():
            m = pred_arr == pv
            if not m.any():
                continue
            lbl = {0: "Land (0)", 1: "Water (1)", 2: "Uncertain (2)"}[pv]
            ax.scatter(feat_df.loc[m, "x"], feat_df.loc[m, "y"],
                       c=col, s=0.4, label=lbl, rasterized=True, alpha=0.8)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        n_water_p = int((pred_arr == 1).sum())
        ax.set_title(f"v5 {title_lbl} predictions  (water={n_water_p:,})")
        ax.legend(markerscale=6, fontsize=9, loc="upper right")
    plt.suptitle("v5 Water vs Land — Model Predictions", fontsize=11)
    plt.tight_layout()
    pp = os.path.join(MODEL_DIR, "xy_scatter_predictions.png")
    plt.savefig(pp, dpi=150); plt.close()
    print(f"Prediction scatter → {pp}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"\n  Labels (auto-labeler v5):")
    print(f"    Water   (1): {n_w:>7,}  ({100*n_w/N:.1f}%)")
    print(f"    Land    (0): {n_l:>7,}  ({100*n_l/N:.1f}%)")
    print(f"    Excluded(-1):{n_unk:>7,}  ({100*n_unk/N:.1f}%)")

    print(f"\n  XGBoost predictions (all {N:,} points):")
    print(f"    Water: {int(xgb_pred.sum()):>7,}  ({100*xgb_pred.mean():.1f}%)")
    print(f"    Land : {int((xgb_pred==0).sum()):>7,}  ({100*(1-xgb_pred.mean()):.1f}%)")
    print(f"    CV macro-F1 : {xgb_cv['macro_f1_mean']:.3f} ± {xgb_cv['macro_f1_std']:.3f}")
    print(f"    CV ROC-AUC  : {xgb_cv['auc_mean']:.3f}")
    print(f"    Water  P={xgb_cv['prec_water']:.3f}  R={xgb_cv['rec_water']:.3f}")
    print(f"    Land   P={xgb_cv['prec_land']:.3f}  R={xgb_cv['rec_land']:.3f}")

    print(f"\n  Deep V5Net predictions (all {N:,} points):")
    print(f"    Water: {int(deep_pred.sum()):>7,}  ({100*deep_pred.mean():.1f}%)")
    print(f"    Land : {int((deep_pred==0).sum()):>7,}  ({100*(1-deep_pred.mean()):.1f}%)")
    print(f"    Best val macro-F1: {deep_cv['best_val_f1']:.3f}")

    print(f"\n  Ensemble:")
    print(f"    Water     (1): {int((ensemble==1).sum()):>7,}  ({100*(ensemble==1).mean():.1f}%)")
    print(f"    Land      (0): {int((ensemble==0).sum()):>7,}  ({100*(ensemble==0).mean():.1f}%)")
    print(f"    Uncertain (2): {int((ensemble==2).sum()):>7,}  ({100*(ensemble==2).mean():.1f}%)")
    print(f"    Agreement: {100*agree.mean():.1f}%")

    # Save metrics JSON
    metrics = {
        "labeling": {
            "n_water": n_w, "n_land": n_l, "n_excluded": n_unk,
            "z_water_max": Z_WATER_MAX, "z_canopy_min": Z_CANOPY_MIN,
            "score_water_thresh": SCORE_WATER_THRESH,
            "score_land_thresh":  SCORE_LAND_THRESH,
        },
        "xgb_cv": xgb_cv,
        "deep_val": deep_cv,
        "inference": {
            "xgb_water": int(xgb_pred.sum()),
            "deep_water": int(deep_pred.sum()),
            "ensemble_water": int((ensemble==1).sum()),
            "ensemble_uncertain": int((ensemble==2).sum()),
            "agreement_pct": float(100 * agree.mean()),
        },
    }
    mpath = os.path.join(MODEL_DIR, "v5_metrics.json")
    with open(mpath, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nMetrics → {mpath}")

    print(f"\nOpen {PC_OUT} in CloudCompare:")
    print(f"  Colour by 'xgb_pred' or 'ensemble':")
    print(f"    0 = land (brown)  1 = water (blue)  2 = uncertain (yellow)")
    print(f"  Also check 'energy_concentration', 'v5_score', 'xgb_proba'")
    print("\nDone.")


if __name__ == "__main__":
    main()
