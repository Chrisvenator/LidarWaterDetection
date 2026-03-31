"""
compare_models.py — Train three XGBoost variants with different feature sets
and compare their generalisation profiles via 5-fold spatial CV.

Models:
  A — all features (reference baseline, includes z)
  B — no elevation (drops z, z_relative — tests if geometry alone suffices)
  C — waveform + reflectance only (no k-NN geometry at all)

Outputs:
  models/xgb_A.json, xgb_B.json, xgb_C.json
  models/xgb_compare.json  — per-model CV metrics
  models/feature_importance_ABC.png
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Feature-set definitions ────────────────────────────────────────────────

FEAT_A = [           # All features (same as baseline_model.py)
    'max_amp', 'mean_amp', 'std_amp', 'total_energy', 'n_samples',
    'time_span', 'n_gaps', 'max_gap', 'mean_gap', 'total_gap',
    'n_peaks', 'max_peak_spacing', 'mean_peak_spacing',
    'first_last_span', 'n_clusters', 'energy_ratio_late',
    'first_peak_amp', 'last_peak_amp', 'peak_amp_ratio', 'depth_proxy_m',
    'z', 'reflectance_dB',
    'planarity', 'roughness', 'linearity', 'sphericity',
    'height_range_local', 'height_std_local', 'z_relative',
]

FEAT_B = [c for c in FEAT_A if c not in ('z', 'z_relative')]  # no raw elevation

FEAT_C = [           # Waveform + reflectance only (no geometry)
    'max_amp', 'mean_amp', 'std_amp', 'total_energy', 'n_samples',
    'time_span', 'n_gaps', 'max_gap', 'mean_gap', 'total_gap',
    'n_peaks', 'max_peak_spacing', 'mean_peak_spacing',
    'first_last_span', 'n_clusters', 'energy_ratio_late',
    'first_peak_amp', 'last_peak_amp', 'peak_amp_ratio', 'depth_proxy_m',
    'reflectance_dB',
]

MODELS = {
    'A': {'desc': 'All features (incl. z)',         'cols': FEAT_A},
    'B': {'desc': 'No elevation (z, z_relative)',   'cols': FEAT_B},
    'C': {'desc': 'Waveform + reflectance only',    'cols': FEAT_C},
}


def spatial_cv_split(df: pd.DataFrame, n_folds: int = 5):
    y_col  = 'y' if 'y' in df.columns else 'y_rel'
    y_vals = df[y_col].values
    edges  = np.percentile(y_vals, np.linspace(0, 100, n_folds + 1))
    splits = []
    for fold in range(n_folds):
        lo, hi   = edges[fold], edges[fold + 1]
        val_mask = (y_vals >= lo) & (y_vals <= hi)
        splits.append((np.where(~val_mask)[0], np.where(val_mask)[0]))
    return splits


def cv_model(X, y, spw, n_folds: int = 5, confident_df: pd.DataFrame = None):
    splits = spatial_cv_split(confident_df, n_folds)
    fold_f1s, fold_aucs = [], []
    fold_prec_w, fold_rec_w = [], []
    fold_prec_l, fold_rec_l = [], []

    for fold, (trn_idx, val_idx) in enumerate(splits):
        if len(val_idx) == 0 or len(trn_idx) == 0:
            continue
        X_tr, X_va = X[trn_idx], X[val_idx]
        y_tr, y_va = y[trn_idx], y[val_idx]
        if len(np.unique(y_va)) < 2:
            continue

        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric='logloss',
            random_state=42, n_jobs=-1, verbosity=0,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        proba = clf.predict_proba(X_va)[:, 1]
        pred  = (proba >= 0.5).astype(int)

        fold_f1s.append(  f1_score(y_va, pred, average='macro',    zero_division=0))
        fold_prec_w.append(precision_score(y_va, pred, pos_label=1, zero_division=0))
        fold_rec_w.append( recall_score(   y_va, pred, pos_label=1, zero_division=0))
        fold_prec_l.append(precision_score(y_va, pred, pos_label=0, zero_division=0))
        fold_rec_l.append( recall_score(   y_va, pred, pos_label=0, zero_division=0))
        try:
            fold_aucs.append(roc_auc_score(y_va, proba))
        except Exception:
            fold_aucs.append(float('nan'))

    return {
        'macro_f1_mean':  float(np.mean(fold_f1s)),
        'macro_f1_std':   float(np.std(fold_f1s)),
        'auc_mean':       float(np.nanmean(fold_aucs)),
        'prec_water_mean':float(np.mean(fold_prec_w)),
        'rec_water_mean': float(np.mean(fold_rec_w)),
        'prec_land_mean': float(np.mean(fold_prec_l)),
        'rec_land_mean':  float(np.mean(fold_rec_l)),
    }


def train_final(X, y, spw, n_estimators: int = 400):
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric='logloss',
        random_state=42, n_jobs=-1, verbosity=0,
    )
    clf.fit(X, y, verbose=False)
    return clf


def main():
    features_path = 'features.csv'
    labels_path   = 'labels.csv'
    out_dir       = 'models'
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {features_path} …")
    # pandas renames duplicate 'z' column to 'z.1' — use 'z' (first occurrence)
    feat_df = pd.read_csv(features_path)
    print(f"  {len(feat_df):,} rows × {len(feat_df.columns)} cols")

    print(f"Loading {labels_path} …")
    lab_df = pd.read_csv(labels_path)

    feat_df['label']      = lab_df['label'].values
    feat_df['confidence'] = lab_df['confidence'].values

    confident = feat_df[feat_df['label'].isin([0, 1])].copy()
    n_pos = int((confident['label'] == 1).sum())
    n_neg = int((confident['label'] == 0).sum())
    spw   = round(n_neg / max(n_pos, 1), 3)
    print(f"\nConfident: {len(confident):,}  water={n_pos:,}, land={n_neg:,}, "
          f"scale_pos_weight={spw}")

    y = confident['label'].values.astype(np.int32)

    compare_results = {}
    importances     = {}

    print(f"\n{'='*60}")
    for name, cfg in MODELS.items():
        desc = cfg['desc']
        cols = [c for c in cfg['cols'] if c in confident.columns]
        missing = [c for c in cfg['cols'] if c not in confident.columns]
        if missing:
            print(f"  Model {name}: {len(missing)} cols missing: {missing[:3]}")

        X = confident[cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"\nModel {name} — {desc}  ({len(cols)} features)")
        print(f"  Running 5-fold spatial CV …")
        cv  = cv_model(X, y, spw, n_folds=5, confident_df=confident)
        compare_results[name] = {**cv, 'desc': desc, 'n_features': len(cols)}

        print(f"  macro-F1 : {cv['macro_f1_mean']:.3f} ± {cv['macro_f1_std']:.3f}")
        print(f"  ROC-AUC  : {cv['auc_mean']:.3f}")
        print(f"  Water  — P={cv['prec_water_mean']:.3f}  R={cv['rec_water_mean']:.3f}")
        print(f"  Land   — P={cv['prec_land_mean']:.3f}  R={cv['rec_land_mean']:.3f}")

        print(f"  Training final model …")
        model = train_final(X, y, spw)
        path  = os.path.join(out_dir, f'xgb_{name}.json')
        model.save_model(path)
        print(f"  Saved → {path}")

        imp = pd.DataFrame({'feature': cols, 'importance': model.feature_importances_})
        imp = imp.sort_values('importance', ascending=False)
        importances[name] = imp
        print(f"  Top 5: {', '.join(imp['feature'].head(5).tolist())}")

    # ── Save metrics ────────────────────────────────────────────────────────
    res_path = os.path.join(out_dir, 'xgb_compare.json')
    with open(res_path, 'w') as fh:
        json.dump(compare_results, fh, indent=2)
    print(f"\nComparison metrics saved to {res_path}")

    # ── Comparison table ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'Model':<6} {'Desc':<35} {'F1':>6} {'AUC':>6} {'W-Prec':>7} {'W-Rec':>7}")
    print(f"{'-'*60}")
    for name, r in compare_results.items():
        print(f"  {name}    {r['desc']:<35} {r['macro_f1_mean']:.3f}  {r['auc_mean']:.3f}"
              f"  {r['prec_water_mean']:.3f}  {r['rec_water_mean']:.3f}")

    # ── Feature importance comparison plot ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, (name, imp) in zip(axes, importances.items()):
        top = imp.head(15)
        ax.barh(top['feature'][::-1], top['importance'][::-1])
        ax.set_title(f"Model {name} — {MODELS[name]['desc']}", fontsize=9)
        ax.set_xlabel('XGBoost importance (gain)')
        ax.tick_params(axis='y', labelsize=7)
    plt.suptitle('Feature Importance: A vs B vs C', fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'feature_importance_ABC.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Feature importance plot saved to {plot_path}")
    print("\nDone.")


if __name__ == '__main__':
    main()
