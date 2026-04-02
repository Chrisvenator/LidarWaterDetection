"""
train_v3.py — Retrain without canopy-shortcut elevation features.

The v2 models hit F1=0.98 by learning "points near local terrain minimum = water",
which just separates vegetation canopy from everything below it. The diagnostic
confirmed:
  - height_above_local_min_10m was 46% of B2 importance → canopy separator
  - energy_concentration (AUC 0.820) and max_amp_norm_by_energy (AUC 0.782)
    are the TRUE discriminators for water vs. dry ground at similar elevations

Changes from v2:
  - DROP height_above_local_min, height_above_local_min_10m  (canopy shortcut)
  - KEEP height_percentile_local  (softer local context, not a hard gate)
  - KEEP all waveform shape features  (energy_concentration is the #1 signal)
  - Deep v3 spatial branch: 10 features, zero absolute elevation

Models:
  XGBoost B3 : no absolute z, no canopy-height shortcuts
  Deep v3    : same CNN + MLP with corrected spatial feature set

Outputs:
  models/xgb_B3.json
  models/deep_v3.pt / deep_v3_stats.json / deep_v3_training_curve.png
  models/b3_feature_importance.png
  labeled_pointcloud_v3.csv
"""

import json
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              roc_auc_score, classification_report)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Feature sets ──────────────────────────────────────────────────────────────

# Features removed from v2 because they shortcut to "canopy vs ground"
REMOVED = {'height_above_local_min', 'height_above_local_min_10m'}

WAVEFORM_FEATURES = [
    'max_amp', 'mean_amp', 'std_amp', 'total_energy', 'n_samples',
    'time_span', 'n_gaps', 'max_gap', 'mean_gap', 'total_gap',
    'n_peaks', 'max_peak_spacing', 'mean_peak_spacing',
    'first_last_span', 'n_clusters', 'energy_ratio_late',
    'first_peak_amp', 'last_peak_amp', 'peak_amp_ratio', 'depth_proxy_m',
    # New waveform shape features (top discriminators from diagnostic)
    'energy_concentration',
    'amplitude_weighted_center',
    'active_bins_ratio',
    'max_amp_norm_by_energy',
]

FEAT_B3 = (
    WAVEFORM_FEATURES + [
    'reflectance_dB',
    # Geometric — no absolute z, no canopy shortcuts
    'planarity', 'roughness', 'linearity', 'sphericity',
    'height_range_local', 'height_std_local',
    'z_relative',                  # relative to dataset mean — not absolute
    'height_percentile_local',     # local rank in 5m radius — generalizes
])

# Spatial branch for deep model: 10 features, zero absolute elevation
SPATIAL_COLS_V3 = [
    'reflectance_dB',
    'z_relative',
    'planarity',
    'roughness',
    'height_range_local',
    'height_percentile_local',
    # Waveform shape — top discriminators
    'energy_concentration',
    'amplitude_weighted_center',
    'active_bins_ratio',
    'max_amp_norm_by_energy',
]

NEW_SHAPE_FEATS = {'energy_concentration', 'amplitude_weighted_center',
                   'active_bins_ratio', 'max_amp_norm_by_energy'}


# ── Spatial CV split ──────────────────────────────────────────────────────────

def spatial_cv_split(df, n_folds=5):
    y_vals = df['y'].values
    edges  = np.percentile(y_vals, np.linspace(0, 100, n_folds + 1))
    return [(np.where((y_vals < edges[f]) | (y_vals > edges[f+1]))[0],
             np.where((y_vals >= edges[f]) & (y_vals <= edges[f+1]))[0])
            for f in range(n_folds)]


# ── XGBoost B3 ────────────────────────────────────────────────────────────────

def train_xgb_b3(confident, out_dir):
    cols = [c for c in FEAT_B3 if c in confident.columns]
    missing = set(FEAT_B3) - set(confident.columns)
    if missing:
        print(f"  WARNING: missing cols: {missing}")

    print(f"\n{'='*65}")
    print(f"XGBoost Model B3  ({len(cols)} features)")
    print(f"  Removed from v2: {REMOVED}")
    print(f"  Top diagnostic features present: "
          f"{[c for c in ['energy_concentration','max_amp_norm_by_energy'] if c in cols]}")

    X = np.nan_to_num(confident[cols].values.astype(np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    y = confident['label'].values.astype(np.int32)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    spw   = round(n_neg / max(n_pos, 1), 3)
    print(f"  water={n_pos:,}  land={n_neg:,}  scale_pos_weight={spw}")

    splits = spatial_cv_split(confident)
    metrics = {k: [] for k in ['f1','auc','pw','rw','pl','rl']}

    for fold, (trn, val) in enumerate(splits):
        if len(val) == 0 or len(np.unique(y[val])) < 2:
            continue
        clf = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric='logloss',
            random_state=42, n_jobs=-1, verbosity=0,
        )
        clf.fit(X[trn], y[trn], eval_set=[(X[val], y[val])], verbose=False)
        pr = clf.predict_proba(X[val])[:, 1]
        pd_ = (pr >= 0.5).astype(int); yv = y[val]

        metrics['f1'].append( f1_score(yv, pd_, average='macro',  zero_division=0))
        metrics['pw'].append(precision_score(yv, pd_, pos_label=1, zero_division=0))
        metrics['rw'].append(recall_score(   yv, pd_, pos_label=1, zero_division=0))
        metrics['pl'].append(precision_score(yv, pd_, pos_label=0, zero_division=0))
        metrics['rl'].append(recall_score(   yv, pd_, pos_label=0, zero_division=0))
        try:    metrics['auc'].append(roc_auc_score(yv, pr))
        except: metrics['auc'].append(float('nan'))
        print(f"  Fold {fold+1}: F1={metrics['f1'][-1]:.3f}  AUC={metrics['auc'][-1]:.3f}  "
              f"W-P={metrics['pw'][-1]:.3f}  W-R={metrics['rw'][-1]:.3f}  "
              f"L-P={metrics['pl'][-1]:.3f}  L-R={metrics['rl'][-1]:.3f}")

    print(f"\n  CV macro-F1 : {np.mean(metrics['f1']):.3f} ± {np.std(metrics['f1']):.3f}")
    print(f"  CV ROC-AUC  : {np.nanmean(metrics['auc']):.3f}")
    print(f"  Water  — P={np.mean(metrics['pw']):.3f}  R={np.mean(metrics['rw']):.3f}")
    print(f"  Land   — P={np.mean(metrics['pl']):.3f}  R={np.mean(metrics['rl']):.3f}")

    print(f"\n  Training final B3 on {len(X):,} rows …")
    final = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric='logloss',
        random_state=42, n_jobs=-1, verbosity=0,
    )
    final.fit(X, y, verbose=False)
    path = os.path.join(out_dir, 'xgb_B3.json')
    final.save_model(path)
    print(f"  Saved → {path}")

    imp = pd.DataFrame({'feature': cols, 'importance': final.feature_importances_})
    imp = imp.sort_values('importance', ascending=False)
    print(f"\n  Top 15 features:")
    print(imp.head(15).to_string(index=False))

    # Importance plot — highlight waveform shape features in red
    fig, ax = plt.subplots(figsize=(9, 7))
    top20 = imp.head(20)
    colors = []
    for f in top20['feature'][::-1]:
        if f in NEW_SHAPE_FEATS:      colors.append('#e74c3c')   # new shape
        elif f in REMOVED:            colors.append('#e67e22')   # removed (shouldn't appear)
        else:                         colors.append('#2980b9')   # original
    ax.barh(top20['feature'][::-1], top20['importance'][::-1], color=colors)
    ax.set_xlabel('XGBoost importance (gain)')
    ax.set_title('Model B3 — Feature Importance\n'
                 '(red = new waveform shape, blue = original geometric/waveform)')
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(handles=[Patch(color='#e74c3c', label='New waveform shape'),
                       Patch(color='#2980b9', label='Original features')], fontsize=8)
    plt.tight_layout()
    imp_path = os.path.join(out_dir, 'b3_feature_importance.png')
    plt.savefig(imp_path, dpi=150); plt.close()
    print(f"  Importance plot → {imp_path}")

    return final, cols, {
        'macro_f1_mean':   float(np.mean(metrics['f1'])),
        'macro_f1_std':    float(np.std(metrics['f1'])),
        'auc_mean':        float(np.nanmean(metrics['auc'])),
        'prec_water_mean': float(np.mean(metrics['pw'])),
        'rec_water_mean':  float(np.mean(metrics['rw'])),
        'prec_land_mean':  float(np.mean(metrics['pl'])),
        'rec_land_mean':   float(np.mean(metrics['rl'])),
    }


# ── Deep model v3 ─────────────────────────────────────────────────────────────

class _CB(nn.Module):
    def __init__(self, i, o, k):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv1d(i, o, k, padding='same', bias=False),
            nn.BatchNorm1d(o), nn.ReLU(inplace=True))
    def forward(self, x): return self.b(x)

class WaveformNetV3(nn.Module):
    """CNN + MLP. Spatial branch explicitly fed waveform shape features."""
    def __init__(self, n_spatial, dropout=0.35):
        super().__init__()
        self.wf = nn.Sequential(
            _CB(1, 32, 3), _CB(32, 64, 5), _CB(64, 64, 11),
            nn.MaxPool1d(4), _CB(64, 128, 5), nn.AdaptiveAvgPool1d(1))
        self.sp = nn.Sequential(
            nn.Linear(n_spatial, 128), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(True),
            nn.Linear(64, 32),   nn.ReLU(True))
        self.head = nn.Sequential(
            nn.Linear(160, 128), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),  nn.ReLU(True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 2))
    def forward(self, wf, sp):
        return self.head(torch.cat([self.wf(wf).squeeze(-1), self.sp(sp)], 1))


class FlatDS(Dataset):
    def __init__(self, g, s, l):
        self.g = torch.from_numpy(g).unsqueeze(1)
        self.s = torch.from_numpy(s)
        self.l = torch.from_numpy(l)
    def __len__(self): return len(self.l)
    def __getitem__(self, i): return self.g[i], self.s[i], self.l[i]


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, smooth=0.1):
        super().__init__()
        self.gamma = gamma; self.alpha = alpha; self.smooth = smooth
    def forward(self, logits, targets):
        n = logits.size(1)
        with torch.no_grad():
            st = torch.zeros_like(logits).fill_(self.smooth / n)
            st.scatter_(1, targets.unsqueeze(1),
                        1.0 - self.smooth + self.smooth / n)
        lp  = torch.nn.functional.log_softmax(logits, 1)
        pt  = lp.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        fw  = (1 - pt).pow(self.gamma)
        ce  = -(st * lp).sum(1)
        loss = fw * ce
        if self.alpha is not None:
            at = torch.where(targets == 1,
                             torch.full_like(pt, self.alpha),
                             torch.full_like(pt, 1 - self.alpha))
            loss = at * loss
        return loss.mean()


def train_deep_v3(feat_df, labels_conf, grids_all, conf_rows, out_dir,
                  epochs=80, batch_size=512, lr=1e-3, patience=20,
                  val_frac=0.20, dropout=0.35):

    sp_cols = [c for c in SPATIAL_COLS_V3 if c in feat_df.columns]
    missing = set(SPATIAL_COLS_V3) - set(feat_df.columns)
    if missing:
        print(f"  WARNING: missing spatial cols: {missing}")

    print(f"\n{'='*65}")
    print(f"Deep Model v3  ({len(sp_cols)} spatial features, no canopy shortcuts)")
    print(f"  Spatial cols: {sp_cols}")

    grids_conf   = grids_all[conf_rows].copy().astype(np.float32)
    spatial_conf = feat_df[sp_cols].values[conf_rows].astype(np.float32)
    y_conf       = feat_df['y'].values[conf_rows]

    grids_conf   = np.nan_to_num(grids_conf,   nan=0.0, posinf=0.0, neginf=0.0)
    spatial_conf = np.nan_to_num(spatial_conf, nan=0.0, posinf=0.0, neginf=0.0)

    cutoff   = np.percentile(y_conf, 100 * (1 - val_frac))
    val_mask = y_conf >= cutoff
    trn_mask = ~val_mask
    print(f"  Train: {trn_mask.sum():,}   Val: {val_mask.sum():,}")

    g_mean = float(grids_conf[trn_mask].mean())
    g_std  = float(grids_conf[trn_mask].std()) + 1e-6
    sp_mean = spatial_conf[trn_mask].mean(0)
    sp_std  = spatial_conf[trn_mask].std(0)  + 1e-6

    gn = (grids_conf   - g_mean)  / g_std
    sn = (spatial_conf - sp_mean) / sp_std

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    n_pos_tr = int((labels_conf[trn_mask] == 1).sum())
    n_neg_tr = int(trn_mask.sum()) - n_pos_tr
    alpha    = round(n_neg_tr / (n_pos_tr + n_neg_tr), 4)

    model     = WaveformNetV3(n_spatial=len(sp_cols), dropout=dropout).to(device)
    criterion = FocalLoss(gamma=2.0, alpha=alpha, smooth=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=5e-6)
    scaler    = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    nw = min(4, os.cpu_count() or 1)
    tr_ld = DataLoader(FlatDS(gn[trn_mask], sn[trn_mask], labels_conf[trn_mask]),
                       batch_size=batch_size, shuffle=True,
                       num_workers=nw, pin_memory=True)
    va_ld = DataLoader(FlatDS(gn[val_mask], sn[val_mask], labels_conf[val_mask]),
                       batch_size=batch_size*2, shuffle=False,
                       num_workers=nw, pin_memory=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}  alpha={alpha:.3f}")

    best_f1 = 0.0; pat = 0
    hist    = {'tl': [], 'vl': [], 'f1': [], 'auc': []}
    mpath   = os.path.join(out_dir, 'deep_v3.pt')

    print(f"\n  {'Ep':>5}  {'TrLoss':>8}  {'VaLoss':>8}  {'F1':>7}  {'AUC':>7}  LR")
    for ep in range(1, epochs + 1):
        model.train(); tl = 0.0
        for wf, sp, lb in tr_ld:
            wf = wf.to(device, non_blocking=True)
            sp = sp.to(device, non_blocking=True)
            lb = lb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=scaler.is_enabled()):
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
        vl /= len(va_ld.dataset)
        preds = np.concatenate(preds_); proba = np.concatenate(proba_)
        labs  = np.concatenate(labs_)
        vf1   = f1_score(labs, preds, average='macro', zero_division=0)
        try:    vauc = roc_auc_score(labs, proba)
        except: vauc = float('nan')

        scheduler.step(); lr_cur = optimizer.param_groups[0]['lr']
        hist['tl'].append(tl); hist['vl'].append(vl)
        hist['f1'].append(vf1); hist['auc'].append(vauc)

        flag = ''
        if vf1 > best_f1:
            best_f1 = vf1; pat = 0; flag = ' ← best'
            torch.save(model.state_dict(), mpath)
        else:
            pat += 1

        print(f"  {ep:>5}  {tl:>8.4f}  {vl:>8.4f}  {vf1:>7.4f}  {vauc:>7.4f}  "
              f"{lr_cur:.2e}{flag}")

        if pat >= patience:
            print(f"\n  Early stopping at epoch {ep}")
            break

    print(f"\n  Best val macro-F1: {best_f1:.4f}")

    # Final val report with best checkpoint
    model.load_state_dict(torch.load(mpath, map_location=device, weights_only=True))
    model.eval()
    preds_, proba_, labs_ = [], [], []
    with torch.no_grad():
        for wf, sp, lb in va_ld:
            wf = wf.to(device, non_blocking=True)
            sp = sp.to(device, non_blocking=True)
            lg = model(wf, sp)
            pr = torch.softmax(lg, 1)[:, 1]
            preds_.append(lg.argmax(1).cpu().numpy())
            proba_.append(pr.cpu().numpy())
            labs_.append(lb.cpu().numpy())
    pf = np.concatenate(preds_); lf = np.concatenate(labs_)
    print(f"\n  Validation report (best checkpoint):")
    print(classification_report(lf, pf, target_names=['land','water'], zero_division=0))

    # Save norm stats
    stats = {'grid_mean': g_mean, 'grid_std': g_std,
             'spatial_mean': sp_mean.tolist(), 'spatial_std': sp_std.tolist(),
             'spatial_cols': sp_cols, 'best_val_f1': float(best_f1)}
    sp_path = os.path.join(out_dir, 'deep_v3_stats.json')
    with open(sp_path, 'w') as fh:
        json.dump(stats, fh, indent=2)

    # Training curve
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    ep_range = range(1, len(hist['tl']) + 1)
    a1.plot(ep_range, hist['tl'], label='train'); a1.plot(ep_range, hist['vl'], label='val')
    a1.set_xlabel('Epoch'); a1.set_ylabel('Focal Loss'); a1.legend()
    a2.plot(ep_range, hist['f1'], label='macro-F1'); a2.plot(ep_range, hist['auc'], label='AUC')
    a2.set_xlabel('Epoch'); a2.set_ylabel('Score'); a2.legend()
    plt.suptitle('Deep Model v3 — waveform shape features, no canopy shortcuts', fontsize=10)
    plt.tight_layout()
    cp = os.path.join(out_dir, 'deep_v3_training_curve.png')
    plt.savefig(cp, dpi=150); plt.close()

    print(f"  Model → {mpath}")
    print(f"  Stats → {sp_path}")
    print(f"  Curve → {cp}")
    return model, stats, {'best_val_f1': float(best_f1), 'spatial_cols': sp_cols}


# ── Full-dataset inference ────────────────────────────────────────────────────

def xgb_predict_full(feat_df, cols, path):
    cols_ = [c for c in cols if c in feat_df.columns]
    X = np.nan_to_num(feat_df[cols_].values.astype(np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    m = xgb.XGBClassifier(); m.load_model(path)
    pr = m.predict_proba(X)[:, 1]
    return (pr >= 0.5).astype(np.int8), pr.astype(np.float32)


@torch.no_grad()
def deep_predict_full(feat_df, grids_all, stats, path, batch_size=2048):
    sp_cols = stats['spatial_cols']
    gn = (grids_all.astype(np.float32) - stats['grid_mean']) / stats['grid_std']
    sp = feat_df[sp_cols].values.astype(np.float32)
    sp = np.nan_to_num(sp, nan=0.0, posinf=0.0, neginf=0.0)
    sn = (sp - np.array(stats['spatial_mean'], np.float32)) / \
              np.array(stats['spatial_std'],  np.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = WaveformNetV3(n_spatial=len(sp_cols), dropout=0.0)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device).eval()

    N = len(feat_df); probas = np.zeros(N, np.float32)
    for s in range(0, N, batch_size):
        e  = min(s + batch_size, N)
        wf = torch.from_numpy(gn[s:e]).unsqueeze(1).to(device)
        sp = torch.from_numpy(sn[s:e]).to(device)
        probas[s:e] = torch.softmax(model(wf, sp), 1)[:, 1].cpu().numpy()
        if s % 50_000 == 0:
            print(f"    {s:>7,}/{N:,} …")
    return (probas >= 0.5).astype(np.int8), probas


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    feat_path  = 'features_v2.csv'
    lab_path   = 'labels.csv'
    grids_path = 'waveform_grids.npy'
    out_dir    = 'models'
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {feat_path} …")
    feat_df = pd.read_csv(feat_path)
    print(f"  {len(feat_df):,} rows × {len(feat_df.columns)} cols")

    print(f"Loading {lab_path} …")
    lab_df  = pd.read_csv(lab_path)
    feat_df['label'] = lab_df['label'].values

    confident = feat_df[feat_df['label'].isin([0, 1])].copy()
    n_pos = int((confident['label'] == 1).sum())
    n_neg = int((confident['label'] == 0).sum())
    print(f"Confident: {len(confident):,}  water={n_pos:,}, land={n_neg:,}")

    conf_rows  = np.where(feat_df['label'].isin([0, 1]).values)[0]
    labels_conf = feat_df['label'].values[conf_rows].astype(np.int64)

    print(f"Loading {grids_path} …")
    grids_all = np.load(grids_path, mmap_mode='r')

    # ── Train ──────────────────────────────────────────────────────────────────
    _, xgb_cols, b3_cv = train_xgb_b3(confident, out_dir)
    _, deep_stats, deep_cv = train_deep_v3(
        feat_df, labels_conf, grids_all, conf_rows, out_dir)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("PROGRESSION SUMMARY  (no-absolute-z models)")
    print(f"{'='*75}")
    print(f"{'Model':<14} {'macro-F1':>9} {'AUC':>7} {'W-P':>7} {'W-R':>7} {'L-P':>7} {'L-R':>7}  Notes")
    print(f"{'-'*75}")
    rows = [
        ('B (v1)',   0.929, 0.988, 0.903, 0.921, 0.952, 0.951, 'no z; planarity dominant'),
        ('Deep v1',  0.938,  None,  None,  None,  None,  None, 'waveform CNN'),
        ('B2 (v2)',  0.980, 0.999, 0.962, 0.987, 0.989, 0.984, 'height_above_local_min 46%'),
        ('Deep v2',  0.985,  None,  None,  None,  None,  None, 'canopy shortcut'),
    ]
    for (name, f1, auc, wp, wr, lp, lr, note) in rows:
        def fmt(v): return f'{v:>7.3f}' if v is not None else '      —'
        print(f"  {name:<12} {f1:>9.3f} {fmt(auc)} {fmt(wp)} {fmt(wr)} {fmt(lp)} {fmt(lr)}  {note}")
    cr = b3_cv
    print(f"  {'B3 (v3)':<12} {cr['macro_f1_mean']:>9.3f} {cr['auc_mean']:>7.3f} "
          f"{cr['prec_water_mean']:>7.3f} {cr['rec_water_mean']:>7.3f} "
          f"{cr['prec_land_mean']:>7.3f} {cr['rec_land_mean']:>7.3f}  "
          f"energy_concentration dominant  ± {cr['macro_f1_std']:.3f}")
    print(f"  {'Deep v3':<12} {deep_cv['best_val_f1']:>9.3f}       —       —       —       —       —  "
          f"waveform shape features")

    # ── Export ─────────────────────────────────────────────────────────────────
    print(f"\nExporting labeled_pointcloud_v3.csv …")
    print("  Applying XGBoost B3 …")
    pb3, cb3 = xgb_predict_full(feat_df, FEAT_B3, os.path.join(out_dir, 'xgb_B3.json'))
    print(f"  B3: water={int(pb3.sum()):,}  land={len(pb3)-int(pb3.sum()):,}")

    print("  Applying Deep v3 …")
    pd3, cd3 = deep_predict_full(feat_df, grids_all, deep_stats,
                                  os.path.join(out_dir, 'deep_v3.pt'))
    print(f"  Deep v3: water={int(pd3.sum()):,}  land={len(pd3)-int(pd3.sum()):,}")

    out = pd.DataFrame({
        'x': feat_df['x'].values, 'y': feat_df['y'].values, 'z': feat_df['z'].values,
        'reflectance_dB':  feat_df['reflectance_dB'].values if 'reflectance_dB' in feat_df.columns else 0.0,
        'auto_label':      lab_df['label'].values.astype(np.int8),
        'auto_confidence': lab_df['confidence'].values.astype(np.float32),
        'pred_B3':    pb3,   'conf_B3':    np.round(cb3, 4),
        'pred_deep_v3': pd3, 'conf_deep_v3': np.round(cd3, 4),
    })
    for col in ['n_gaps','max_gap','n_peaks','depth_proxy_m','planarity','roughness',
                'height_percentile_local','energy_concentration',
                'amplitude_weighted_center','max_amp_norm_by_energy','z_relative']:
        if col in feat_df.columns:
            out[col] = feat_df[col].values

    out.to_csv('labeled_pointcloud_v3.csv', index=False)
    print(f"  Saved {len(out):,} rows to labeled_pointcloud_v3.csv")

    conf_bool = lab_df['label'].isin([0, 1]).values
    nc = conf_bool.sum()
    for name, preds in [('B3', pb3), ('deep_v3', pd3)]:
        agree = (preds[conf_bool] == lab_df['label'].values[conf_bool]).sum()
        print(f"  {name} vs auto-labels: {agree:,}/{nc:,} = {100*agree/nc:.1f}%")

    print("\nDone.")


if __name__ == '__main__':
    main()
