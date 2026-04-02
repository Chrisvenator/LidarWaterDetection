"""
train_stage1.py — Canopy vs ground-level classifier (Stage 1).

This is the easy problem. Features are designed to work at any absolute elevation
by using height relative to the local 10 m neighbourhood minimum.

Features: height_above_local_min_10m, height_above_local_min, height_percentile_local,
          planarity, roughness, height_range_local, height_std_local,
          n_clusters, n_peaks, total_energy, linearity, sphericity

Output:
  models/stage1.json          — trained XGBoost model
  models/stage1_cv.json       — CV metrics
  models/stage1_importance.png
"""

import json, os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
import xgboost as xgb
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              roc_auc_score, classification_report,
                              confusion_matrix)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FEATURES = [
    # Primary — relative elevation (generalises across rivers)
    'height_above_local_min_10m',
    'height_above_local_min',
    'height_percentile_local',
    # Geometric neighbourhood
    'planarity', 'roughness', 'height_range_local', 'height_std_local',
    'linearity', 'sphericity',
    # Waveform structure (canopy = many layers)
    'n_clusters', 'n_peaks', 'total_energy', 'n_samples', 'time_span',
    'energy_concentration',   # canopy = spread, ground = compact
    'active_bins_ratio',
]


def spatial_cv_split(df, n_folds=5):
    y = df['y'].values
    edges = np.percentile(y, np.linspace(0, 100, n_folds + 1))
    return [(np.where((y < edges[f]) | (y > edges[f+1]))[0],
             np.where((y >= edges[f]) & (y <= edges[f+1]))[0])
            for f in range(n_folds)]


def main():
    feat_path  = ROOT / 'data_processed' / 'features_v2.csv'
    label_path = ROOT / 'data_processed' / 'labels_v3.csv'
    out_dir    = ROOT / 'models' / 'v4-staged-cascade'
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {feat_path} …")
    feat_df = pd.read_csv(feat_path)
    print(f"Loading {label_path} …")
    lab_df  = pd.read_csv(label_path)

    feat_df['stage1_label'] = lab_df['stage1_label'].values
    feat_df['y_coord']      = feat_df['y'].values

    # Keep only confident labels
    df = feat_df[feat_df['stage1_label'].isin([0, 1])].copy()
    n_canopy = int((df['stage1_label'] == 1).sum())
    n_ground = int((df['stage1_label'] == 0).sum())
    print(f"\nStage 1 training set: {len(df):,}  "
          f"canopy={n_canopy:,}  ground={n_ground:,}")

    cols = [c for c in FEATURES if c in df.columns]
    miss = [c for c in FEATURES if c not in df.columns]
    if miss:
        print(f"  WARNING: missing features: {miss}")

    X = np.nan_to_num(df[cols].values.astype(np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    y = df['stage1_label'].values.astype(np.int32)

    spw = round(n_ground / max(n_canopy, 1), 3)
    print(f"  scale_pos_weight={spw}  ({len(cols)} features)")

    # ── Spatial CV ─────────────────────────────────────────────────────────────
    splits = spatial_cv_split(df)
    metrics = {k: [] for k in ['f1', 'auc', 'pc', 'rc', 'pg', 'rg']}

    print(f"\n5-fold spatial CV:")
    for fold, (trn, val) in enumerate(splits):
        if len(val) == 0 or len(np.unique(y[val])) < 2:
            continue
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric='logloss',
            random_state=42, n_jobs=-1, verbosity=0,
        )
        clf.fit(X[trn], y[trn], eval_set=[(X[val], y[val])], verbose=False)
        pr = clf.predict_proba(X[val])[:, 1]
        pd_ = (pr >= 0.5).astype(int); yv = y[val]

        metrics['f1'].append( f1_score(yv, pd_, average='macro',   zero_division=0))
        metrics['pc'].append(precision_score(yv, pd_, pos_label=1, zero_division=0))
        metrics['rc'].append(recall_score(   yv, pd_, pos_label=1, zero_division=0))
        metrics['pg'].append(precision_score(yv, pd_, pos_label=0, zero_division=0))
        metrics['rg'].append(recall_score(   yv, pd_, pos_label=0, zero_division=0))
        try:    metrics['auc'].append(roc_auc_score(yv, pr))
        except: metrics['auc'].append(float('nan'))

        print(f"  Fold {fold+1}: F1={metrics['f1'][-1]:.3f}  AUC={metrics['auc'][-1]:.3f}  "
              f"Canopy P={metrics['pc'][-1]:.3f} R={metrics['rc'][-1]:.3f}  "
              f"Ground P={metrics['pg'][-1]:.3f} R={metrics['rg'][-1]:.3f}")

    print(f"\n  CV macro-F1  : {np.mean(metrics['f1']):.3f} ± {np.std(metrics['f1']):.3f}")
    print(f"  CV ROC-AUC   : {np.nanmean(metrics['auc']):.3f}")
    print(f"  Canopy  P={np.mean(metrics['pc']):.3f}  R={np.mean(metrics['rc']):.3f}")
    print(f"  Ground  P={np.mean(metrics['pg']):.3f}  R={np.mean(metrics['rg']):.3f}")

    # ── Final model ────────────────────────────────────────────────────────────
    print(f"\nTraining final Stage 1 model on all {len(X):,} confident rows …")
    final = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric='logloss',
        random_state=42, n_jobs=-1, verbosity=0,
    )
    final.fit(X, y, verbose=False)
    path = os.path.join(out_dir, 'stage1.json')
    final.save_model(path)
    print(f"Model → {path}")

    imp = pd.DataFrame({'feature': cols, 'importance': final.feature_importances_})
    imp = imp.sort_values('importance', ascending=False)
    print(f"\nTop 10 features:")
    print(imp.head(10).to_string(index=False))

    # Importance plot
    fig, ax = plt.subplots(figsize=(8, 6))
    top = imp.head(15)
    ax.barh(top['feature'][::-1], top['importance'][::-1], color='#2ecc71')
    ax.set_xlabel('XGBoost importance (gain)')
    ax.set_title('Stage 1 — Canopy vs Ground\nFeature Importance')
    ax.tick_params(axis='y', labelsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'stage1_importance.png'), dpi=150)
    plt.close()

    # Apply to full dataset to show coverage
    print(f"\nApplying Stage 1 to full {len(feat_df):,} points …")
    X_full = np.nan_to_num(feat_df[cols].values.astype(np.float32),
                            nan=0.0, posinf=0.0, neginf=0.0)
    proba_full = final.predict_proba(X_full)[:, 1]
    pred_full  = (proba_full >= 0.5).astype(np.int8)
    n_can = int(pred_full.sum())
    n_gnd = len(pred_full) - n_can
    print(f"  Predicted canopy  : {n_can:,}  ({100*n_can/len(pred_full):.1f}%)")
    print(f"  Predicted ground  : {n_gnd:,}  ({100*n_gnd/len(pred_full):.1f}%)")

    # Save CV results
    cv_out = {
        'macro_f1_mean':       float(np.mean(metrics['f1'])),
        'macro_f1_std':        float(np.std(metrics['f1'])),
        'auc_mean':            float(np.nanmean(metrics['auc'])),
        'prec_canopy_mean':    float(np.mean(metrics['pc'])),
        'rec_canopy_mean':     float(np.mean(metrics['rc'])),
        'prec_ground_mean':    float(np.mean(metrics['pg'])),
        'rec_ground_mean':     float(np.mean(metrics['rg'])),
        'n_train':             int(len(X)),
        'n_canopy':            n_canopy,
        'n_ground':            n_ground,
        'feature_cols':        cols,
        'full_pred_canopy':    n_can,
        'full_pred_ground':    n_gnd,
    }
    with open(os.path.join(out_dir, 'stage1_cv.json'), 'w') as fh:
        json.dump(cv_out, fh, indent=2)
    print(f"CV results → {os.path.join(out_dir, 'stage1_cv.json')}")
    print("\nDone.")


if __name__ == '__main__':
    main()
