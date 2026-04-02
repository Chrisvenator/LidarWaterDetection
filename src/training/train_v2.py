"""
train_v2.py — Train XGBoost Model B2 + WaveformNet v2 on features_v2.csv.

Model B2 : all features EXCEPT absolute z — includes all new relative elevation
           and waveform shape features.
Deep v2  : 1D CNN waveform branch (unchanged) + MLP spatial branch with 12
           features instead of 5, still zero absolute z.

Outputs:
  models/xgb_B2.json
  models/deep_v2.pt
  models/deep_v2_stats.json
  models/deep_v2_training_curve.png
  models/b2_feature_importance.png
  labeled_pointcloud_v2.csv    ← best model predictions for CloudCompare
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


# ── Feature sets ──────────────────────────────────────────────────────────────

NEW_FEATURES = [
    'height_above_local_min',
    'height_above_local_min_10m',
    'height_percentile_local',
    'energy_concentration',
    'amplitude_weighted_center',
    'active_bins_ratio',
    'max_amp_norm_by_energy',
]

FEAT_B2 = [
    # Waveform-derived (original)
    'max_amp', 'mean_amp', 'std_amp', 'total_energy', 'n_samples',
    'time_span', 'n_gaps', 'max_gap', 'mean_gap', 'total_gap',
    'n_peaks', 'max_peak_spacing', 'mean_peak_spacing',
    'first_last_span', 'n_clusters', 'energy_ratio_late',
    'first_peak_amp', 'last_peak_amp', 'peak_amp_ratio', 'depth_proxy_m',
    # Point scalars (no absolute z)
    'reflectance_dB',
    # Geometric (no z, no z_relative)
    'planarity', 'roughness', 'linearity', 'sphericity',
    'height_range_local', 'height_std_local', 'z_relative',
] + NEW_FEATURES

SPATIAL_COLS_V2 = [
    # Original relative features (no absolute z)
    'reflectance_dB',
    'z_relative',
    'planarity',
    'roughness',
    'height_range_local',
    # New relative elevation
    'height_above_local_min',
    'height_above_local_min_10m',
    'height_percentile_local',
    # New waveform shape (explicit to MLP branch)
    'energy_concentration',
    'amplitude_weighted_center',
    'active_bins_ratio',
    'max_amp_norm_by_energy',
]


# ── Spatial CV split ──────────────────────────────────────────────────────────

def spatial_cv_split(df, n_folds=5):
    y_vals = df['y'].values
    edges  = np.percentile(y_vals, np.linspace(0, 100, n_folds + 1))
    splits = []
    for fold in range(n_folds):
        lo, hi   = edges[fold], edges[fold + 1]
        val_mask = (y_vals >= lo) & (y_vals <= hi)
        splits.append((np.where(~val_mask)[0], np.where(val_mask)[0]))
    return splits


# ── XGBoost B2 ────────────────────────────────────────────────────────────────

def train_xgb_b2(confident, out_dir):
    cols = [c for c in FEAT_B2 if c in confident.columns]
    print(f"\n{'='*60}")
    print(f"XGBoost Model B2  ({len(cols)} features, no absolute z)")

    missing = [c for c in FEAT_B2 if c not in confident.columns]
    if missing:
        print(f"  WARNING: missing {len(missing)} cols: {missing}")

    X = confident[cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = confident['label'].values.astype(np.int32)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    spw   = round(n_neg / max(n_pos, 1), 3)

    splits = spatial_cv_split(confident)
    fold_f1s, fold_aucs = [], []
    fold_prec_w, fold_rec_w = [], []
    fold_prec_l, fold_rec_l = [], []

    for fold, (trn_idx, val_idx) in enumerate(splits):
        if len(val_idx) == 0 or len(np.unique(y[val_idx])) < 2:
            continue
        clf = xgb.XGBClassifier(
            n_estimators=350, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric='logloss',
            random_state=42, n_jobs=-1, verbosity=0,
        )
        clf.fit(X[trn_idx], y[trn_idx],
                eval_set=[(X[val_idx], y[val_idx])], verbose=False)
        proba = clf.predict_proba(X[val_idx])[:, 1]
        pred  = (proba >= 0.5).astype(int)
        yv    = y[val_idx]

        fold_f1s.append(f1_score(yv, pred, average='macro', zero_division=0))
        fold_prec_w.append(precision_score(yv, pred, pos_label=1, zero_division=0))
        fold_rec_w.append( recall_score(   yv, pred, pos_label=1, zero_division=0))
        fold_prec_l.append(precision_score(yv, pred, pos_label=0, zero_division=0))
        fold_rec_l.append( recall_score(   yv, pred, pos_label=0, zero_division=0))
        try:
            fold_aucs.append(roc_auc_score(yv, proba))
        except Exception:
            fold_aucs.append(float('nan'))
        print(f"  Fold {fold+1}: F1={fold_f1s[-1]:.3f}  AUC={fold_aucs[-1]:.3f}  "
              f"W-P={fold_prec_w[-1]:.3f}  W-R={fold_rec_w[-1]:.3f}  "
              f"L-P={fold_prec_l[-1]:.3f}  L-R={fold_rec_l[-1]:.3f}")

    print(f"\n  CV macro-F1 : {np.mean(fold_f1s):.3f} ± {np.std(fold_f1s):.3f}")
    print(f"  CV ROC-AUC  : {np.nanmean(fold_aucs):.3f}")
    print(f"  Water  — P={np.mean(fold_prec_w):.3f}  R={np.mean(fold_rec_w):.3f}")
    print(f"  Land   — P={np.mean(fold_prec_l):.3f}  R={np.mean(fold_rec_l):.3f}")

    # Final model
    print(f"\n  Training final B2 on all {len(X):,} confident rows …")
    final = xgb.XGBClassifier(
        n_estimators=450, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric='logloss',
        random_state=42, n_jobs=-1, verbosity=0,
    )
    final.fit(X, y, verbose=False)
    model_path = os.path.join(out_dir, 'xgb_B2.json')
    final.save_model(model_path)
    print(f"  Model saved → {model_path}")

    imp = pd.DataFrame({'feature': cols, 'importance': final.feature_importances_})
    imp = imp.sort_values('importance', ascending=False)
    print(f"\n  Top 15 features:")
    print(imp.head(15).to_string(index=False))

    # Importance plot
    fig, ax = plt.subplots(figsize=(9, 7))
    top20 = imp.head(20)
    colors = ['#e74c3c' if f in NEW_FEATURES else '#2980b9'
              for f in top20['feature'][::-1]]
    ax.barh(top20['feature'][::-1], top20['importance'][::-1], color=colors)
    ax.set_xlabel('XGBoost importance (gain)')
    ax.set_title('Model B2 — Feature Importance\n(red = new features, blue = original)')
    ax.tick_params(axis='y', labelsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#e74c3c', label='New features'),
                       Patch(color='#2980b9', label='Original features')],
              fontsize=8)
    plt.tight_layout()
    imp_path = os.path.join(out_dir, 'b2_feature_importance.png')
    plt.savefig(imp_path, dpi=150)
    plt.close()
    print(f"  Importance plot saved → {imp_path}")

    cv_results = {
        'macro_f1_mean':   float(np.mean(fold_f1s)),
        'macro_f1_std':    float(np.std(fold_f1s)),
        'auc_mean':        float(np.nanmean(fold_aucs)),
        'prec_water_mean': float(np.mean(fold_prec_w)),
        'rec_water_mean':  float(np.mean(fold_rec_w)),
        'prec_land_mean':  float(np.mean(fold_prec_l)),
        'rec_land_mean':   float(np.mean(fold_rec_l)),
    }
    return final, cols, cv_results


# ── Deep model v2 ─────────────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding='same', bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class WaveformNetV2(nn.Module):
    """
    Same CNN backbone as v1, wider spatial MLP (12 features instead of 5).
    Dropout slightly increased to handle more features.
    """
    def __init__(self, n_spatial: int, dropout: float = 0.35):
        super().__init__()
        self.wf_branch = nn.Sequential(
            _ConvBlock(1,  32, 3),
            _ConvBlock(32, 64, 5),
            _ConvBlock(64, 64, 11),
            nn.MaxPool1d(4),
            _ConvBlock(64, 128, 5),
            nn.AdaptiveAvgPool1d(1),
        )
        self.sp_branch = nn.Sequential(
            nn.Linear(n_spatial, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + 32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 2),
        )

    def forward(self, wf, sp):
        wf_feat = self.wf_branch(wf).squeeze(-1)
        sp_feat = self.sp_branch(sp)
        return self.fusion(torch.cat([wf_feat, sp_feat], dim=1))


class FlatDataset(Dataset):
    def __init__(self, grids, spatial, labels):
        self.grids   = torch.from_numpy(grids).unsqueeze(1)
        self.spatial = torch.from_numpy(spatial)
        self.labels  = torch.from_numpy(labels)
    def __len__(self):  return len(self.labels)
    def __getitem__(self, i): return self.grids[i], self.spatial[i], self.labels[i]


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, smoothing=0.1):
        super().__init__()
        self.gamma = gamma; self.alpha = alpha; self.smoothing = smoothing
    def forward(self, logits, targets):
        n = logits.size(1)
        with torch.no_grad():
            st = torch.zeros_like(logits).fill_(self.smoothing / n)
            st.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / n)
        log_p  = torch.nn.functional.log_softmax(logits, 1)
        p_t    = log_p.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        fw     = (1 - p_t).pow(self.gamma)
        ce     = -(st * log_p).sum(1)
        loss   = fw * ce
        if self.alpha is not None:
            at = torch.where(targets == 1,
                             torch.full_like(p_t, self.alpha),
                             torch.full_like(p_t, 1 - self.alpha))
            loss = at * loss
        return loss.mean()


def train_deep_v2(feat_df, labels_conf, grids_all, conf_rows, out_dir,
                  epochs=70, batch_size=512, lr=1e-3, patience=18,
                  val_frac=0.20, dropout=0.35):
    print(f"\n{'='*60}")
    print(f"Deep Model v2  (1D CNN + MLP, {len(SPATIAL_COLS_V2)} spatial features, no absolute z)")

    # Extract arrays
    sp_available = [c for c in SPATIAL_COLS_V2 if c in feat_df.columns]
    sp_missing   = [c for c in SPATIAL_COLS_V2 if c not in feat_df.columns]
    if sp_missing:
        print(f"  WARNING: missing spatial cols: {sp_missing}")
    n_spatial = len(sp_available)

    spatial_conf = feat_df[sp_available].values[conf_rows].astype(np.float32)
    grids_conf   = grids_all[conf_rows].copy().astype(np.float32)
    y_conf       = feat_df['y'].values[conf_rows]

    spatial_conf = np.nan_to_num(spatial_conf, nan=0.0, posinf=0.0, neginf=0.0)
    grids_conf   = np.nan_to_num(grids_conf,   nan=0.0, posinf=0.0, neginf=0.0)

    # Spatial train/val split
    cutoff   = np.percentile(y_conf, 100 * (1 - val_frac))
    val_mask = y_conf >= cutoff
    trn_mask = ~val_mask
    print(f"  Train: {trn_mask.sum():,}   Val: {val_mask.sum():,}")

    # Normalise from training set
    g_mean = float(grids_conf[trn_mask].mean())
    g_std  = float(grids_conf[trn_mask].std()) + 1e-6
    sp_mean = spatial_conf[trn_mask].mean(axis=0)
    sp_std  = spatial_conf[trn_mask].std(axis=0) + 1e-6

    grids_n   = (grids_conf   - g_mean)  / g_std
    spatial_n = (spatial_conf - sp_mean) / sp_std

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    n_pos_tr = int((labels_conf[trn_mask] == 1).sum())
    n_neg_tr = int(trn_mask.sum()) - n_pos_tr
    alpha = round(n_neg_tr / (n_pos_tr + n_neg_tr), 4)

    model     = WaveformNetV2(n_spatial=n_spatial, dropout=dropout).to(device)
    criterion = FocalLoss(gamma=2.0, alpha=alpha, smoothing=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=5e-6)
    scaler    = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    n_workers = min(4, os.cpu_count() or 1)

    train_ds = FlatDataset(grids_n[trn_mask], spatial_n[trn_mask], labels_conf[trn_mask])
    val_ds   = FlatDataset(grids_n[val_mask], spatial_n[val_mask], labels_conf[val_mask])
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=n_workers, pin_memory=True)
    val_ld   = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                          num_workers=n_workers, pin_memory=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    best_f1      = 0.0
    patience_cnt = 0
    history      = {'tr_loss': [], 'va_loss': [], 'va_f1': [], 'va_auc': []}
    model_path   = os.path.join(out_dir, 'deep_v2.pt')

    print(f"\n  {'Epoch':>6}  {'TrLoss':>8}  {'VaLoss':>8}  {'F1':>7}  {'AUC':>7}  LR")
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        tr_loss = 0.0
        for wf, sp, lbl in train_ld:
            wf = wf.to(device, non_blocking=True)
            sp = sp.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=scaler.is_enabled()):
                loss = criterion(model(wf, sp), lbl)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item() * len(lbl)
        tr_loss /= len(train_ds)

        # Eval
        model.eval()
        va_loss = 0.0
        all_preds, all_proba, all_lbl = [], [], []
        with torch.no_grad():
            for wf, sp, lbl in val_ld:
                wf = wf.to(device, non_blocking=True)
                sp = sp.to(device, non_blocking=True)
                lbl = lbl.to(device, non_blocking=True)
                logits = model(wf, sp)
                va_loss += criterion(logits, lbl).item() * len(lbl)
                proba = torch.softmax(logits, 1)[:, 1]
                all_preds.append(logits.argmax(1).cpu().numpy())
                all_proba.append(proba.cpu().numpy())
                all_lbl.append(lbl.cpu().numpy())
        va_loss /= len(val_ds)
        preds = np.concatenate(all_preds)
        proba = np.concatenate(all_proba)
        lbls  = np.concatenate(all_lbl)
        va_f1 = f1_score(lbls, preds, average='macro', zero_division=0)
        try:
            va_auc = roc_auc_score(lbls, proba)
        except Exception:
            va_auc = float('nan')

        scheduler.step()
        lr_cur = optimizer.param_groups[0]['lr']
        history['tr_loss'].append(tr_loss)
        history['va_loss'].append(va_loss)
        history['va_f1'].append(va_f1)
        history['va_auc'].append(va_auc)

        flag = ''
        if va_f1 > best_f1:
            best_f1 = va_f1
            patience_cnt = 0
            flag = ' ← best'
            torch.save(model.state_dict(), model_path)
        else:
            patience_cnt += 1

        print(f"  {epoch:>6}  {tr_loss:>8.4f}  {va_loss:>8.4f}  "
              f"{va_f1:>7.4f}  {va_auc:>7.4f}  {lr_cur:.2e}{flag}")

        if patience_cnt >= patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    print(f"\n  Best val macro-F1: {best_f1:.4f}")

    # Print final validation report
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    all_preds, all_proba, all_lbl = [], [], []
    with torch.no_grad():
        for wf, sp, lbl in val_ld:
            wf = wf.to(device, non_blocking=True)
            sp = sp.to(device, non_blocking=True)
            logits = model(wf, sp)
            proba  = torch.softmax(logits, 1)[:, 1]
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_proba.append(proba.cpu().numpy())
            all_lbl.append(lbl.cpu().numpy())
    preds_fin = np.concatenate(all_preds)
    proba_fin = np.concatenate(all_proba)
    lbl_fin   = np.concatenate(all_lbl)

    print(f"\n  Final validation report (best checkpoint):")
    print(classification_report(lbl_fin, preds_fin,
                                target_names=['land', 'water'], zero_division=0))

    # Save norm stats
    stats = {
        'grid_mean':    g_mean, 'grid_std': g_std,
        'spatial_mean': sp_mean.tolist(), 'spatial_std': sp_std.tolist(),
        'spatial_cols': sp_available,
        'best_val_f1':  float(best_f1),
    }
    stats_path = os.path.join(out_dir, 'deep_v2_stats.json')
    with open(stats_path, 'w') as fh:
        json.dump(stats, fh, indent=2)

    # Training curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(history['tr_loss']) + 1)
    ax1.plot(ep, history['tr_loss'], label='train')
    ax1.plot(ep, history['va_loss'], label='val')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Focal Loss'); ax1.legend()
    ax2.plot(ep, history['va_f1'],  label='macro-F1')
    ax2.plot(ep, history['va_auc'], label='ROC-AUC')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Score'); ax2.legend()
    plt.suptitle('Deep Model v2 Training', fontsize=11)
    plt.tight_layout()
    curve_path = os.path.join(out_dir, 'deep_v2_training_curve.png')
    plt.savefig(curve_path, dpi=150)
    plt.close()

    print(f"  Model → {model_path}")
    print(f"  Stats → {stats_path}")
    print(f"  Curve → {curve_path}")

    cv_results = {
        'best_val_f1': float(best_f1),
        'n_spatial':   n_spatial,
        'spatial_cols': sp_available,
    }
    return model, stats, cv_results


# ── Inference on full 234k set ────────────────────────────────────────────────

def xgb_full_predict(feat_df, cols, model_path):
    cols_avail = [c for c in cols if c in feat_df.columns]
    X = feat_df[cols_avail].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    m = xgb.XGBClassifier()
    m.load_model(model_path)
    proba = m.predict_proba(X)[:, 1]
    return (proba >= 0.5).astype(np.int8), proba.astype(np.float32)


@torch.no_grad()
def deep_full_predict(feat_df, grids_all, stats, model_path, batch_size=2048):
    sp_cols  = stats['spatial_cols']
    g_mean   = float(stats['grid_mean'])
    g_std    = float(stats['grid_std'])
    sp_mean  = np.array(stats['spatial_mean'], dtype=np.float32)
    sp_std   = np.array(stats['spatial_std'],  dtype=np.float32)

    grids_n  = (grids_all.astype(np.float32) - g_mean) / g_std
    sp_vals  = feat_df[sp_cols].values.astype(np.float32)
    sp_vals  = np.nan_to_num(sp_vals, nan=0.0, posinf=0.0, neginf=0.0)
    sp_n     = (sp_vals - sp_mean) / sp_std

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = WaveformNetV2(n_spatial=len(sp_cols), dropout=0.0)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()

    N      = len(feat_df)
    probas = np.zeros(N, dtype=np.float32)
    for start in range(0, N, batch_size):
        end  = min(start + batch_size, N)
        wf_b = torch.from_numpy(grids_n[start:end]).unsqueeze(1).to(device)
        sp_b = torch.from_numpy(sp_n[start:end]).to(device)
        p    = torch.softmax(model(wf_b, sp_b), 1)[:, 1].cpu().numpy()
        probas[start:end] = p
        if start % 50_000 == 0:
            print(f"    {start:>7,}/{N:,} …")
    return (probas >= 0.5).astype(np.int8), probas


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    features_path = 'features_v2.csv'
    labels_path   = 'labels.csv'
    grids_path    = 'waveform_grids.npy'
    out_dir       = 'models'
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {features_path} …")
    feat_df = pd.read_csv(features_path)
    print(f"  {len(feat_df):,} rows × {len(feat_df.columns)} cols")

    print(f"Loading {labels_path} …")
    lab_df = pd.read_csv(labels_path)
    feat_df['label'] = lab_df['label'].values

    confident = feat_df[feat_df['label'].isin([0, 1])].copy()
    n_pos = int((confident['label'] == 1).sum())
    n_neg = int((confident['label'] == 0).sum())
    print(f"Confident: {len(confident):,}  water={n_pos:,}, land={n_neg:,}")

    conf_mask = feat_df['label'].isin([0, 1]).values
    conf_rows = np.where(conf_mask)[0]

    print(f"Loading {grids_path} …")
    grids_all = np.load(grids_path, mmap_mode='r')

    # ── Train B2 ──────────────────────────────────────────────────────────────
    xgb_model, xgb_cols, b2_cv = train_xgb_b2(confident, out_dir)

    # ── Train Deep v2 ─────────────────────────────────────────────────────────
    labels_conf = feat_df['label'].values[conf_rows].astype(np.int64)
    deep_model, deep_stats, deep_cv = train_deep_v2(
        feat_df, labels_conf, grids_all, conf_rows, out_dir)

    # ── Print comparison table ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("COMPARISON vs PREVIOUS BEST (no-z models)")
    print(f"{'='*70}")
    prev = {
        'B (old)':    {'F1': 0.929, 'AUC': 0.988, 'W-P': 0.903, 'W-R': 0.921, 'L-P': 0.952, 'L-R': 0.951},
        'Deep v1':    {'F1': 0.938, 'AUC': '—',   'W-P': '—',   'W-R': '—',   'L-P': '—',   'L-R': '—'},
    }
    print(f"{'Model':<14} {'macro-F1':>9} {'ROC-AUC':>9} {'W-Prec':>8} {'W-Rec':>8} {'L-Prec':>8} {'L-Rec':>8}")
    print(f"{'-'*70}")
    for name, r in prev.items():
        print(f"  {name:<12} {r['F1']:>9.3f} {str(r['AUC']):>9} "
              f"{str(r['W-P']):>8} {str(r['W-R']):>8} "
              f"{str(r['L-P']):>8} {str(r['L-R']):>8}")
    cr = b2_cv
    print(f"  {'B2 (new)':<12} {cr['macro_f1_mean']:>9.3f} {cr['auc_mean']:>9.3f} "
          f"{cr['prec_water_mean']:>8.3f} {cr['rec_water_mean']:>8.3f} "
          f"{cr['prec_land_mean']:>8.3f} {cr['rec_land_mean']:>8.3f}")
    print(f"  {'Deep v2':<12} {deep_cv['best_val_f1']:>9.3f} {'(val F1)':>9}")
    print(f"  (B2 ± std: {cr['macro_f1_std']:.3f})")

    # ── Export labeled_pointcloud_v2.csv ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("Exporting full predictions to labeled_pointcloud_v2.csv …")

    # B2 on full 234k
    print("  Applying XGBoost B2 …")
    pred_b2, conf_b2 = xgb_full_predict(
        feat_df, FEAT_B2, os.path.join(out_dir, 'xgb_B2.json'))
    n_w = int(pred_b2.sum())
    print(f"  B2: water={n_w:,}  land={len(pred_b2)-n_w:,}")

    # Deep v2 on full 234k
    print("  Applying Deep v2 …")
    pred_dv2, conf_dv2 = deep_full_predict(
        feat_df, grids_all, deep_stats,
        os.path.join(out_dir, 'deep_v2.pt'))
    n_w = int(pred_dv2.sum())
    print(f"  Deep v2: water={n_w:,}  land={len(pred_dv2)-n_w:,}")

    # Load previous model predictions if available
    out_df = pd.DataFrame()
    out_df['x']             = feat_df['x'].values
    out_df['y']             = feat_df['y'].values
    out_df['z']             = feat_df['z'].values
    out_df['reflectance_dB'] = feat_df['reflectance_dB'].values if 'reflectance_dB' in feat_df.columns else 0.0
    out_df['auto_label']    = lab_df['label'].values.astype(np.int8)
    out_df['auto_confidence'] = lab_df['confidence'].values.astype(np.float32)
    out_df['pred_B2']       = pred_b2
    out_df['conf_B2']       = np.round(conf_b2, 4)
    out_df['pred_deep_v2']  = pred_dv2
    out_df['conf_deep_v2']  = np.round(conf_dv2, 4)

    # Scalar fields for CloudCompare colouring
    for col in ['n_gaps', 'max_gap', 'n_peaks', 'max_peak_spacing', 'depth_proxy_m',
                'planarity', 'roughness', 'z_relative',
                'height_above_local_min', 'height_above_local_min_10m',
                'height_percentile_local', 'energy_concentration',
                'amplitude_weighted_center', 'active_bins_ratio']:
        if col in feat_df.columns:
            out_df[col] = feat_df[col].values

    out_path = 'labeled_pointcloud_v2.csv'
    out_df.to_csv(out_path, index=False)
    print(f"  Saved {len(out_df):,} rows to {out_path}")

    # Agreement with auto-labels
    conf_mask_bool = lab_df['label'].isin([0, 1]).values
    n_conf = conf_mask_bool.sum()
    for name, preds in [('B2', pred_b2), ('deep_v2', pred_dv2)]:
        agree = (preds[conf_mask_bool] == lab_df['label'].values[conf_mask_bool]).sum()
        print(f"  {name} agreement with v2 auto-labels: "
              f"{agree:,}/{n_conf:,} = {100*agree/n_conf:.1f}%")

    print("\nDone.")
    print(f"\nOpen labeled_pointcloud_v2.csv in CloudCompare.")
    print(f"Colour by pred_B2 or pred_deep_v2, or compare scalar fields.")


if __name__ == '__main__':
    main()
