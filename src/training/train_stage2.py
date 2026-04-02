"""
train_stage2.py — Water vs dry ground classifier (Stage 2).

Trained ONLY on ground-level points (Stage 1 predicted as ground).
No absolute z. No height_above_local_min shortcuts.
Primary signals: energy_concentration, max_amp_norm_by_energy (from diagnostic).

Models:
  XGBoost  → models/stage2_xgb.json
  Deep CNN → models/stage2_deep.pt + models/stage2_deep_stats.json

Outputs:
  models/stage2_cv.json
  models/stage2_importance.png
  models/stage2_training_curve.png
"""

import json, os
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent.parent
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              roc_auc_score, classification_report)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Feature set ───────────────────────────────────────────────────────────────
# NO absolute z. NO height_above_local_min / _10m (canopy shortcuts).
# Focus: waveform shape + local geometry + reflectance.

XGB_FEATURES = [
    # Top diagnostic discriminators (AUC 0.78–0.82 on clean zone comparison)
    'energy_concentration',
    'max_amp_norm_by_energy',
    # Waveform structure
    'n_peaks', 'n_gaps', 'n_clusters', 'n_samples', 'time_span',
    'max_gap', 'mean_gap', 'total_gap', 'first_last_span',
    'max_amp', 'mean_amp', 'std_amp', 'total_energy',
    'first_peak_amp', 'last_peak_amp', 'peak_amp_ratio',
    'energy_ratio_late', 'depth_proxy_m',
    'amplitude_weighted_center', 'active_bins_ratio',
    # Scalar
    'reflectance_dB',
    # Local geometry — valid for ground points (no canopy to confuse)
    'planarity', 'roughness', 'linearity', 'sphericity',
    'height_range_local', 'height_std_local',
    # Soft relative position (not a hard gate)
    'height_percentile_local',
    'z_relative',
]

SPATIAL_COLS = [
    'energy_concentration',       # AUC 0.82 — primary signal
    'max_amp_norm_by_energy',     # AUC 0.78 — primary signal
    'reflectance_dB',
    'planarity',
    'roughness',
    'height_range_local',
    'height_percentile_local',
    'amplitude_weighted_center',
    'active_bins_ratio',
    'z_relative',
]


# ── Spatial CV ────────────────────────────────────────────────────────────────

def spatial_cv_split(df, n_folds=5):
    y = df['y'].values
    edges = np.percentile(y, np.linspace(0, 100, n_folds + 1))
    return [(np.where((y < edges[f]) | (y > edges[f+1]))[0],
             np.where((y >= edges[f]) & (y <= edges[f+1]))[0])
            for f in range(n_folds)]


# ── XGBoost Stage 2 ───────────────────────────────────────────────────────────

def train_xgb(df_ground, out_dir):
    cols = [c for c in XGB_FEATURES if c in df_ground.columns]
    miss = [c for c in XGB_FEATURES if c not in df_ground.columns]
    if miss:
        print(f"  WARNING: missing features: {miss}")

    print(f"\n{'='*60}")
    print(f"Stage 2 XGBoost  ({len(cols)} features, no absolute z)")
    n_w = int((df_ground['stage2_label'] == 1).sum())
    n_d = int((df_ground['stage2_label'] == 0).sum())
    print(f"  water={n_w:,}  dry_ground={n_d:,}")

    X = np.nan_to_num(df_ground[cols].values.astype(np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    y = df_ground['stage2_label'].values.astype(np.int32)
    spw = round(n_d / max(n_w, 1), 3)
    print(f"  scale_pos_weight={spw}")

    splits = spatial_cv_split(df_ground)
    metrics = {k: [] for k in ['f1', 'auc', 'pw', 'rw', 'pd', 'rd']}

    print(f"\n  5-fold spatial CV:")
    for fold, (trn, val) in enumerate(splits):
        if len(val) == 0 or len(np.unique(y[val])) < 2:
            continue
        clf = xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric='logloss',
            random_state=42, n_jobs=-1, verbosity=0,
        )
        clf.fit(X[trn], y[trn], eval_set=[(X[val], y[val])], verbose=False)
        pr = clf.predict_proba(X[val])[:, 1]
        pd_ = (pr >= 0.5).astype(int); yv = y[val]

        metrics['f1'].append( f1_score(yv, pd_, average='macro',   zero_division=0))
        metrics['pw'].append(precision_score(yv, pd_, pos_label=1, zero_division=0))
        metrics['rw'].append(recall_score(   yv, pd_, pos_label=1, zero_division=0))
        metrics['pd'].append(precision_score(yv, pd_, pos_label=0, zero_division=0))
        metrics['rd'].append(recall_score(   yv, pd_, pos_label=0, zero_division=0))
        try:    metrics['auc'].append(roc_auc_score(yv, pr))
        except: metrics['auc'].append(float('nan'))
        print(f"    Fold {fold+1}: F1={metrics['f1'][-1]:.3f}  AUC={metrics['auc'][-1]:.3f}  "
              f"Water P={metrics['pw'][-1]:.3f} R={metrics['rw'][-1]:.3f}  "
              f"Dry P={metrics['pd'][-1]:.3f} R={metrics['rd'][-1]:.3f}")

    print(f"\n  CV macro-F1 : {np.mean(metrics['f1']):.3f} ± {np.std(metrics['f1']):.3f}")
    print(f"  CV ROC-AUC  : {np.nanmean(metrics['auc']):.3f}")
    print(f"  Water  P={np.mean(metrics['pw']):.3f}  R={np.mean(metrics['rw']):.3f}")
    print(f"  Dry    P={np.mean(metrics['pd']):.3f}  R={np.mean(metrics['rd']):.3f}")

    print(f"\n  Training final XGBoost on {len(X):,} rows …")
    final = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric='logloss',
        random_state=42, n_jobs=-1, verbosity=0,
    )
    final.fit(X, y, verbose=False)
    path = os.path.join(out_dir, 'stage2_xgb.json')
    final.save_model(path)
    print(f"  Model → {path}")

    imp = pd.DataFrame({'feature': cols, 'importance': final.feature_importances_})
    imp = imp.sort_values('importance', ascending=False)
    print(f"\n  Top 12 features:")
    print(imp.head(12).to_string(index=False))

    # Plot — highlight the top diagnostic features
    TOP_DIAG = {'energy_concentration', 'max_amp_norm_by_energy'}
    fig, ax = plt.subplots(figsize=(9, 7))
    top20 = imp.head(20)
    colors = ['#e74c3c' if f in TOP_DIAG else '#3498db'
              for f in top20['feature'][::-1]]
    ax.barh(top20['feature'][::-1], top20['importance'][::-1], color=colors)
    ax.set_xlabel('XGBoost importance (gain)')
    ax.set_title('Stage 2 — Water vs Dry Ground\nFeature Importance  '
                 '(red = top diagnostic features)')
    ax.tick_params(axis='y', labelsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#e74c3c', label='Top diagnostic features'),
                       Patch(color='#3498db', label='Other')], fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'stage2_importance.png'), dpi=150)
    plt.close()

    cv_res = {
        'macro_f1_mean':    float(np.mean(metrics['f1'])),
        'macro_f1_std':     float(np.std(metrics['f1'])),
        'auc_mean':         float(np.nanmean(metrics['auc'])),
        'prec_water_mean':  float(np.mean(metrics['pw'])),
        'rec_water_mean':   float(np.mean(metrics['rw'])),
        'prec_dry_mean':    float(np.mean(metrics['pd'])),
        'rec_dry_mean':     float(np.mean(metrics['rd'])),
        'n_water':          n_w,
        'n_dry':            n_d,
        'feature_cols':     cols,
    }
    return final, cols, cv_res


# ── Deep model ────────────────────────────────────────────────────────────────

class _CB(nn.Module):
    def __init__(self, i, o, k):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv1d(i, o, k, padding='same', bias=False),
            nn.BatchNorm1d(o), nn.ReLU(inplace=True))
    def forward(self, x): return self.b(x)


class Stage2Net(nn.Module):
    """1D CNN (waveform) + MLP (spatial). Spatial branch sees top discriminators."""
    def __init__(self, n_spatial, dropout=0.0):
        super().__init__()
        self.wf = nn.Sequential(
            _CB(1, 32, 3), _CB(32, 64, 5), _CB(64, 64, 11),
            nn.MaxPool1d(4), _CB(64, 128, 5), nn.AdaptiveAvgPool1d(1))
        self.sp = nn.Sequential(
            nn.Linear(n_spatial, 128), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Linear(128, 64),  nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Linear(64, 32),   nn.ReLU(True))
        self.head = nn.Sequential(
            nn.Linear(160, 128), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Linear(128, 64),  nn.ReLU(True),
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


def train_deep(feat_df_ground, grids_all, ground_orig_rows, out_dir,
               epochs=80, batch_size=256, lr=1e-3, patience=20,
               val_frac=0.20, dropout=0.40):

    sp_cols = [c for c in SPATIAL_COLS if c in feat_df_ground.columns]
    miss = set(SPATIAL_COLS) - set(feat_df_ground.columns)
    if miss:
        print(f"  WARNING: missing spatial cols: {miss}")

    print(f"\n{'='*60}")
    print(f"Stage 2 Deep Model  ({len(sp_cols)} spatial features)")
    print(f"  Spatial cols: {sp_cols}")

    grids_g   = grids_all[ground_orig_rows].copy().astype(np.float32)
    spatial_g = feat_df_ground[sp_cols].values.astype(np.float32)
    labels_g  = feat_df_ground['stage2_label'].values.astype(np.int64)
    y_g       = feat_df_ground['y'].values

    grids_g   = np.nan_to_num(grids_g,   nan=0.0, posinf=0.0, neginf=0.0)
    spatial_g = np.nan_to_num(spatial_g, nan=0.0, posinf=0.0, neginf=0.0)

    cutoff   = np.percentile(y_g, 100 * (1 - val_frac))
    val_mask = y_g >= cutoff
    trn_mask = ~val_mask
    print(f"  Train: {trn_mask.sum():,}   Val: {val_mask.sum():,}")

    g_mean = float(grids_g[trn_mask].mean())
    g_std  = float(grids_g[trn_mask].std()) + 1e-6
    sp_mean = spatial_g[trn_mask].mean(0)
    sp_std  = spatial_g[trn_mask].std(0) + 1e-6

    gn = (grids_g   - g_mean)  / g_std
    sn = (spatial_g - sp_mean) / sp_std

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    n_pos = int((labels_g[trn_mask] == 1).sum())
    n_neg = int(trn_mask.sum()) - n_pos
    alpha = round(n_neg / (n_pos + n_neg), 4)

    model     = Stage2Net(n_spatial=len(sp_cols), dropout=dropout).to(device)
    criterion = FocalLoss(gamma=2.0, alpha=alpha, smooth=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=5e-6)
    scaler    = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    nw = min(4, os.cpu_count() or 1)
    tr_ld = DataLoader(FlatDS(gn[trn_mask], sn[trn_mask], labels_g[trn_mask]),
                       batch_size=batch_size, shuffle=True,
                       num_workers=nw, pin_memory=True)
    va_ld = DataLoader(FlatDS(gn[val_mask], sn[val_mask], labels_g[val_mask]),
                       batch_size=batch_size*2, shuffle=False,
                       num_workers=nw, pin_memory=True)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}  "
          f"alpha={alpha:.3f}")

    best_f1 = 0.0; pat = 0
    hist = {'tl': [], 'vl': [], 'f1': [], 'auc': []}
    mpath = os.path.join(out_dir, 'stage2_deep.pt')

    print(f"\n  {'Ep':>4}  {'TrLoss':>8}  {'VaLoss':>8}  {'F1':>7}  {'AUC':>7}  LR")
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

        print(f"  {ep:>4}  {tl:>8.4f}  {vl:>8.4f}  {vf1:>7.4f}  {vauc:>7.4f}  "
              f"{lr_cur:.2e}{flag}")

        if pat >= patience:
            print(f"\n  Early stopping at epoch {ep}")
            break

    print(f"\n  Best val macro-F1: {best_f1:.4f}")

    # Final val report
    model.load_state_dict(torch.load(mpath, map_location=device, weights_only=True))
    model.eval()
    preds_, labs_ = [], []
    with torch.no_grad():
        for wf, sp, lb in va_ld:
            wf = wf.to(device, non_blocking=True)
            sp = sp.to(device, non_blocking=True)
            preds_.append(model(wf, sp).argmax(1).cpu().numpy())
            labs_.append(lb.numpy())
    print(f"\n  Validation report (best checkpoint):")
    print(classification_report(np.concatenate(labs_), np.concatenate(preds_),
                                target_names=['dry_ground', 'water'], zero_division=0))

    stats = {'grid_mean': g_mean, 'grid_std': g_std,
             'spatial_mean': sp_mean.tolist(), 'spatial_std': sp_std.tolist(),
             'spatial_cols': sp_cols, 'best_val_f1': float(best_f1)}
    sp_path = os.path.join(out_dir, 'stage2_deep_stats.json')
    with open(sp_path, 'w') as fh:
        json.dump(stats, fh, indent=2)

    # Training curve
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    ep_r = range(1, len(hist['tl']) + 1)
    a1.plot(ep_r, hist['tl'], label='train'); a1.plot(ep_r, hist['vl'], label='val')
    a1.set_xlabel('Epoch'); a1.set_ylabel('Focal Loss'); a1.legend()
    a2.plot(ep_r, hist['f1'], label='macro-F1'); a2.plot(ep_r, hist['auc'], label='AUC')
    a2.set_xlabel('Epoch'); a2.set_ylabel('Score'); a2.legend()
    plt.suptitle('Stage 2 Deep Model — water vs dry ground (ground-level only)',
                 fontsize=10)
    plt.tight_layout()
    cp = os.path.join(out_dir, 'stage2_training_curve.png')
    plt.savefig(cp, dpi=150); plt.close()
    print(f"  Model → {mpath}  Stats → {sp_path}  Curve → {cp}")
    return model, stats, {'best_val_f1': float(best_f1), 'spatial_cols': sp_cols}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    feat_path  = ROOT / 'data_processed' / 'features_v2.csv'
    label_path = ROOT / 'data_processed' / 'labels_v3.csv'
    grids_path = ROOT / 'data_processed' / 'waveform_grids.npy'
    out_dir    = ROOT / 'models' / 'v4-staged-cascade'
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {feat_path} …")
    feat_df = pd.read_csv(feat_path)
    print(f"Loading {label_path} …")
    lab_df  = pd.read_csv(label_path)

    feat_df['stage1_label'] = lab_df['stage1_label'].values
    feat_df['stage2_label'] = lab_df['stage2_label'].values

    # Stage 2 training set: ground-level points with confident stage2 labels
    ground_mask = feat_df['stage1_label'] == 0
    s2_mask     = feat_df['stage2_label'].isin([0, 1])
    train_mask  = ground_mask & s2_mask

    df_ground = feat_df[train_mask].copy()
    print(f"\nStage 2 training set: {len(df_ground):,} ground-level points "
          f"(water={int((df_ground['stage2_label']==1).sum()):,}, "
          f"dry={int((df_ground['stage2_label']==0).sum()):,})")

    # Original row indices (for grid lookup)
    ground_orig_rows = np.where(train_mask.values)[0]

    print(f"Loading {grids_path} …")
    grids_all = np.load(grids_path, mmap_mode='r')

    # ── Train XGBoost ──────────────────────────────────────────────────────────
    _, xgb_cols, xgb_cv = train_xgb(df_ground, out_dir)

    # ── Train Deep model ───────────────────────────────────────────────────────
    _, deep_stats, deep_cv = train_deep(
        df_ground, grids_all, ground_orig_rows, out_dir)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STAGE 2 RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  XGBoost  macro-F1: {xgb_cv['macro_f1_mean']:.3f} ± {xgb_cv['macro_f1_std']:.3f}  "
          f"AUC: {xgb_cv['auc_mean']:.3f}")
    print(f"    Water  P={xgb_cv['prec_water_mean']:.3f}  R={xgb_cv['rec_water_mean']:.3f}")
    print(f"    Dry    P={xgb_cv['prec_dry_mean']:.3f}   R={xgb_cv['rec_dry_mean']:.3f}")
    print(f"  Deep     macro-F1: {deep_cv['best_val_f1']:.3f}  (val split)")

    cv_all = {'xgb': xgb_cv, 'deep': deep_cv}
    with open(os.path.join(out_dir, 'stage2_cv.json'), 'w') as fh:
        json.dump(cv_all, fh, indent=2)
    print(f"\nDone.")


if __name__ == '__main__':
    main()
