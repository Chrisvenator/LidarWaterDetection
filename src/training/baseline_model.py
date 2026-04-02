"""
baseline_model.py — XGBoost water/land classifier on engineered features.

Uses spatial cross-validation (split by y-coordinate strips, not random)
to prevent spatial autocorrelation leakage.

Outputs:
  models/xgb_baseline.json   — trained XGBoost model
  models/cv_results.txt      — cross-validation metrics
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, precision_score, recall_score,
                              roc_auc_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Features used by the model (all computable without neural networks)
FEATURE_COLS = [
    # Waveform-derived
    'max_amp', 'mean_amp', 'std_amp', 'total_energy', 'n_samples',
    'time_span', 'n_gaps', 'max_gap', 'mean_gap', 'total_gap',
    'n_peaks', 'max_peak_spacing', 'mean_peak_spacing',
    'first_last_span', 'n_clusters', 'energy_ratio_late',
    'first_peak_amp', 'last_peak_amp', 'peak_amp_ratio', 'depth_proxy_m',
    # Point cloud scalar
    'z', 'reflectance_dB',
    # Geometric (if available and non-zero)
    'planarity', 'roughness', 'linearity', 'sphericity',
    'height_range_local', 'height_std_local', 'z_relative',
]


def load_data(features_path: str, labels_path: str):
    print(f"Loading features from {features_path} …")
    feat_df = pd.read_csv(features_path)
    print(f"  {len(feat_df):,} rows × {len(feat_df.columns)} cols")

    print(f"Loading labels from {labels_path} …")
    lab_df = pd.read_csv(labels_path)
    print(f"  {len(lab_df):,} rows")

    # Align on index (both files share row order; labels.csv has its own 'y' column)
    merged = feat_df.copy()
    merged['label']      = lab_df['label'].values
    merged['confidence'] = lab_df['confidence'].values
    # Use 'y' from features (original coordinate), not from labels
    # (labels also has 'y' but it's the same values — just ignore it)

    # Keep only confident labels (water=1, land=0)
    confident = merged[merged['label'].isin([0, 1])].copy()
    print(f"\nConfident labels: {len(confident):,}  "
          f"(water={int((confident['label']==1).sum()):,}, "
          f"land={int((confident['label']==0).sum()):,})")

    return confident


def spatial_cv_split(df: pd.DataFrame, n_folds: int = 5):
    """
    Split by y-coordinate strips (not random) to prevent spatial leakage.
    Returns list of (train_idx, val_idx) tuples.
    """
    y_col = 'y' if 'y' in df.columns else 'y_rel'
    y_vals = df[y_col].values
    edges  = np.percentile(y_vals, np.linspace(0, 100, n_folds + 1))
    splits = []
    for fold in range(n_folds):
        lo, hi   = edges[fold], edges[fold + 1]
        val_mask = (y_vals >= lo) & (y_vals <= hi)
        val_idx  = np.where(val_mask)[0]
        trn_idx  = np.where(~val_mask)[0]
        splits.append((trn_idx, val_idx))
    return splits


def train_and_evaluate(confident: pd.DataFrame, out_dir: str, n_folds: int):
    os.makedirs(out_dir, exist_ok=True)

    # Select feature columns that are actually present
    avail_cols = [c for c in FEATURE_COLS if c in confident.columns]
    missing    = [c for c in FEATURE_COLS if c not in confident.columns]
    if missing:
        print(f"  Warning: {len(missing)} feature columns not found: {missing[:5]}…")
    print(f"  Using {len(avail_cols)} feature columns")

    X = confident[avail_cols].values.astype(np.float32)
    y = confident['label'].values.astype(np.int32)

    # Replace NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Class weight for imbalance
    n_pos  = int(y.sum())
    n_neg  = len(y) - n_pos
    spw    = round(n_neg / max(n_pos, 1), 3)
    print(f"  Class balance → land={n_neg:,}, water={n_pos:,}, "
          f"scale_pos_weight={spw}")

    # ── Cross-validation ───────────────────────────────────────────────────
    print(f"\nRunning {n_folds}-fold spatial cross-validation …")
    splits = spatial_cv_split(confident, n_folds)

    fold_f1s    = []
    fold_aucs   = []
    oof_preds   = np.zeros(len(confident), dtype=np.float32)
    oof_labels  = y.copy()

    for fold, (trn_idx, val_idx) in enumerate(splits):
        if len(val_idx) == 0 or len(trn_idx) == 0:
            continue
        X_tr, X_va = X[trn_idx], X[val_idx]
        y_tr, y_va = y[trn_idx], y[val_idx]

        # Skip fold if validation has only one class
        if len(np.unique(y_va)) < 2:
            print(f"  Fold {fold+1}: skipped (single class in validation)")
            continue

        clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        clf.fit(X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False)

        proba = clf.predict_proba(X_va)[:, 1]
        pred  = (proba >= 0.5).astype(int)

        f1  = f1_score(y_va, pred, average='macro', zero_division=0)
        try:
            auc = roc_auc_score(y_va, proba)
        except Exception:
            auc = float('nan')

        fold_f1s.append(f1)
        fold_aucs.append(auc)
        oof_preds[val_idx] = proba

        print(f"  Fold {fold+1}/{n_folds}: "
              f"val_size={len(val_idx):,}  F1={f1:.3f}  AUC={auc:.3f}")

    print(f"\n  CV macro-F1 : {np.mean(fold_f1s):.3f} ± {np.std(fold_f1s):.3f}")
    print(f"  CV ROC-AUC  : {np.nanmean(fold_aucs):.3f} ± {np.nanstd(fold_aucs):.3f}")

    # ── Train final model on ALL confident data ────────────────────────────
    print(f"\nTraining final model on all {len(X):,} confident samples …")
    final_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    final_model.fit(X, y, verbose=False)
    model_path = os.path.join(out_dir, 'xgb_baseline.json')
    final_model.save_model(model_path)
    print(f"Model saved to {model_path}")

    # OOF metrics
    oof_pred_labels = (oof_preds >= 0.5).astype(int)
    valid_oof_mask  = oof_preds > 0   # only folds that ran
    if valid_oof_mask.sum() > 0:
        print(f"\nOut-of-fold classification report (all {valid_oof_mask.sum():,} val samples):")
        print(classification_report(oof_labels[valid_oof_mask],
                                     oof_pred_labels[valid_oof_mask],
                                     target_names=['land', 'water'],
                                     zero_division=0))
        cm = confusion_matrix(oof_labels[valid_oof_mask],
                               oof_pred_labels[valid_oof_mask])
        print(f"Confusion matrix (rows=true, cols=pred):")
        print(f"         pred_land  pred_water")
        print(f"true_land   {cm[0,0]:>7,}     {cm[0,1]:>7,}")
        print(f"true_water  {cm[1,0]:>7,}     {cm[1,1]:>7,}")

    # ── Feature importance ─────────────────────────────────────────────────
    imp     = final_model.feature_importances_
    imp_df  = pd.DataFrame({'feature': avail_cols, 'importance': imp})
    imp_df  = imp_df.sort_values('importance', ascending=False)
    print(f"\nTop 15 feature importances:")
    print(imp_df.head(15).to_string(index=False))

    # Save importance plot
    fig, ax = plt.subplots(figsize=(8, 6))
    top20   = imp_df.head(20)
    ax.barh(top20['feature'][::-1], top20['importance'][::-1])
    ax.set_xlabel('XGBoost feature importance (gain)')
    ax.set_title('Water vs. Land — Feature Importance')
    ax.tick_params(axis='y', labelsize=8)
    plt.tight_layout()
    imp_plot_path = os.path.join(out_dir, 'feature_importance.png')
    plt.savefig(imp_plot_path, dpi=150)
    plt.close()
    print(f"Feature importance plot saved to {imp_plot_path}")

    # Save full results
    results = {
        'cv_macro_f1_mean':  float(np.mean(fold_f1s)),
        'cv_macro_f1_std':   float(np.std(fold_f1s)),
        'cv_auc_mean':       float(np.nanmean(fold_aucs)),
        'cv_auc_std':        float(np.nanstd(fold_aucs)),
        'n_confident':       int(len(confident)),
        'n_water':           int(n_pos),
        'n_land':            int(n_neg),
        'scale_pos_weight':  spw,
        'feature_cols':      avail_cols,
        'top_features':      imp_df['feature'].head(10).tolist(),
    }
    res_path = os.path.join(out_dir, 'cv_results.json')
    with open(res_path, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f"CV results saved to {res_path}")

    return final_model, avail_cols


def main():
    ap = argparse.ArgumentParser(description='XGBoost baseline water/land classifier')
    ap.add_argument('--features',   default='features.csv')
    ap.add_argument('--labels',     default='labels.csv')
    ap.add_argument('--out-dir',    default='models')
    ap.add_argument('--n-folds',    type=int, default=5)
    args = ap.parse_args()

    confident = load_data(args.features, args.labels)
    train_and_evaluate(confident, args.out_dir, args.n_folds)
    print("\nDone.")


if __name__ == '__main__':
    main()
