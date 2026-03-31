"""
evaluate.py — Apply trained XGBoost model to all 234k points and export
              a labeled point cloud for visualization in CloudCompare.

Output:
  labeled_pointcloud.csv — x, y, z, reflectance, label, confidence, label_source
    label: 1=water, 0=land
    label_source: 'model' (XGBoost) or 'auto' (rule-based, for comparison)
    confidence: model predicted probability of water class

  Also prints physical plausibility checks.
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


FEATURE_COLS = [
    'max_amp', 'mean_amp', 'std_amp', 'total_energy', 'n_samples',
    'time_span', 'n_gaps', 'max_gap', 'mean_gap', 'total_gap',
    'n_peaks', 'max_peak_spacing', 'mean_peak_spacing',
    'first_last_span', 'n_clusters', 'energy_ratio_late',
    'first_peak_amp', 'last_peak_amp', 'peak_amp_ratio', 'depth_proxy_m',
    'z', 'reflectance_dB',
    'planarity', 'roughness', 'linearity', 'sphericity',
    'height_range_local', 'height_std_local', 'z_relative',
]


def run(features_path: str, labels_path: str,
        model_path: str, out_path: str, threshold: float) -> None:

    # ── Load inputs ──────────────────────────────────────────────────────────
    print(f"Loading features from {features_path} …")
    feat_df = pd.read_csv(features_path)
    print(f"  {len(feat_df):,} rows × {len(feat_df.columns)} cols")

    print(f"Loading auto-labels from {labels_path} …")
    lab_df = pd.read_csv(labels_path)

    print(f"Loading model from {model_path} …")
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # ── Feature matrix ───────────────────────────────────────────────────────
    avail_cols = [c for c in FEATURE_COLS if c in feat_df.columns]
    X = feat_df[avail_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Predicting on {len(X):,} points with {len(avail_cols)} features …")
    proba = model.predict_proba(X)[:, 1]   # P(water)
    pred  = (proba >= threshold).astype(np.int8)

    # ── Assemble output ──────────────────────────────────────────────────────
    # Prefer original coordinates from features file if present
    x_col = feat_df['x'].values   if 'x' in feat_df.columns else np.zeros(len(feat_df))
    y_col = feat_df['y'].values   if 'y' in feat_df.columns else np.zeros(len(feat_df))
    z_col = feat_df['z'].values   if 'z' in feat_df.columns else np.zeros(len(feat_df))
    r_col = feat_df['reflectance_dB'].values if 'reflectance_dB' in feat_df.columns \
            else feat_df['_riegl.reflectance'].values if '_riegl.reflectance' in feat_df.columns \
            else np.zeros(len(feat_df))

    auto_label = lab_df['label'].values if 'label' in lab_df.columns else np.full(len(feat_df), -1)
    auto_conf  = lab_df['confidence'].values if 'confidence' in lab_df.columns else np.zeros(len(feat_df))

    out_df = pd.DataFrame({
        'x':             x_col,
        'y':             y_col,
        'z':             z_col,
        'reflectance_dB': r_col,
        'label':          pred.astype(np.int8),
        'confidence':     np.round(proba, 4),
        'auto_label':     auto_label.astype(np.int8),
        'auto_confidence': np.round(auto_conf, 4).astype(np.float32),
        # Extra features useful for CloudCompare scalar field colouring:
        'n_gaps':        feat_df['n_gaps'].values   if 'n_gaps'   in feat_df.columns else 0,
        'max_gap':       feat_df['max_gap'].values  if 'max_gap'  in feat_df.columns else 0,
        'n_peaks':       feat_df['n_peaks'].values  if 'n_peaks'  in feat_df.columns else 0,
        'max_peak_spacing': feat_df['max_peak_spacing'].values
                             if 'max_peak_spacing' in feat_df.columns else 0,
        'depth_proxy_m': feat_df['depth_proxy_m'].values
                          if 'depth_proxy_m' in feat_df.columns else 0.0,
    })

    out_df.to_csv(out_path, index=False)
    print(f"\nLabeled point cloud saved to {out_path}  ({len(out_df):,} rows)")

    # ── Summary statistics ───────────────────────────────────────────────────
    n_water = int(pred.sum())
    n_land  = len(pred) - n_water
    print(f"\n{'='*55}")
    print(f"PREDICTION SUMMARY (threshold={threshold})")
    print(f"{'='*55}")
    print(f"  WATER : {n_water:>7,}  ({100*n_water/len(pred):.1f}%)")
    print(f"  LAND  : {n_land:>7,}  ({100*n_land/len(pred):.1f}%)")

    # ── Physical plausibility checks ─────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"PHYSICAL PLAUSIBILITY CHECKS")
    print(f"{'='*55}")

    water_mask = pred == 1
    land_mask  = pred == 0

    # Check 1: Reflectance
    if 'reflectance_dB' in feat_df.columns:
        r_water = feat_df.loc[water_mask, 'reflectance_dB'].mean()
        r_land  = feat_df.loc[land_mask,  'reflectance_dB'].mean()
        diff    = r_land - r_water
        status  = "✓ PASS" if diff > 1.0 else "✗ FAIL"
        print(f"  [R1] Mean reflectance: water={r_water:.2f} dB, land={r_land:.2f} dB, "
              f"diff={diff:.2f} dB  {status}")
        print(f"       Expected: land > water (water absorbs/scatters green light)")

    # Check 2: n_gaps
    if 'n_gaps' in feat_df.columns:
        g_water = feat_df.loc[water_mask, 'n_gaps'].mean()
        g_land  = feat_df.loc[land_mask,  'n_gaps'].mean()
        status  = "✓ PASS" if g_water > g_land else "✗ FAIL"
        print(f"  [W1] Mean n_gaps: water={g_water:.2f}, land={g_land:.2f}  {status}")
        print(f"       Expected: water > land (multi-cluster waveforms)")

    # Check 3: Elevation
    if 'z' in feat_df.columns:
        z_water = feat_df.loc[water_mask, 'z'].mean()
        z_land  = feat_df.loc[land_mask,  'z'].mean()
        status  = "✓ PASS" if z_land > z_water + 0.5 else "? CHECK"
        print(f"  [E1] Mean elevation: water={z_water:.2f} m, land={z_land:.2f} m  {status}")
        print(f"       Expected: water at lower elevation than surrounding land")

    # Check 4: Agreement with auto-labeler on confident points
    confident_mask = np.isin(auto_label, [0, 1])
    if confident_mask.sum() > 0:
        agree = (pred[confident_mask] == auto_label[confident_mask]).sum()
        n_conf = confident_mask.sum()
        pct    = 100 * agree / n_conf
        status = "✓ PASS" if pct >= 70 else "✗ FAIL"
        print(f"  [C1] Model vs. auto-labeler agreement on {n_conf:,} confident points: "
              f"{agree:,}/{n_conf:,} = {pct:.1f}%  {status}")
        print(f"       Expected: ≥70% agreement (auto-labels are noisy but directionally correct)")

    # ── Spatial distribution plot ────────────────────────────────────────────
    if 'x' in feat_df.columns and 'y' in feat_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Panel 1: XGBoost predictions
        ax = axes[0]
        ax.scatter(feat_df.loc[land_mask, 'x'],  feat_df.loc[land_mask, 'y'],
                   c='#a0a040', s=0.2, alpha=0.3, label='Land')
        ax.scatter(feat_df.loc[water_mask, 'x'], feat_df.loc[water_mask, 'y'],
                   c='#1f77b4', s=0.2, alpha=0.6, label='Water')
        ax.set_title(f'XGBoost predictions (threshold={threshold})', fontsize=10)
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        ax.legend(markerscale=15, fontsize=8)
        ax.set_aspect('equal')

        # Panel 2: Auto-labeler confident labels
        ax = axes[1]
        auto_water = auto_label == 1
        auto_land  = auto_label == 0
        auto_unc   = auto_label == -1
        ax.scatter(feat_df.loc[auto_unc,   'x'], feat_df.loc[auto_unc,   'y'],
                   c='#cccccc', s=0.1, alpha=0.2, label='Uncertain')
        ax.scatter(feat_df.loc[auto_land,  'x'], feat_df.loc[auto_land,  'y'],
                   c='#a0a040', s=0.2, alpha=0.5, label='Land')
        ax.scatter(feat_df.loc[auto_water, 'x'], feat_df.loc[auto_water, 'y'],
                   c='#1f77b4', s=0.2, alpha=0.8, label='Water')
        ax.set_title('Auto-labeler (rule-based)', fontsize=10)
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        ax.legend(markerscale=15, fontsize=8)
        ax.set_aspect('equal')

        plt.suptitle('Pielach River — Water vs. Land Classification', fontsize=12)
        plt.tight_layout()
        plot_path = 'models/spatial_prediction.png'
        os.makedirs('models', exist_ok=True)
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"\nSpatial prediction plot saved to {plot_path}")

    print(f"\nDone. Open {out_path} in CloudCompare:")
    print(f"  File → Open → select CSV → set x/y/z and scalar fields")
    print(f"  Colour by 'label' (0=land, 1=water) or 'confidence'")


def main():
    ap = argparse.ArgumentParser(description='Evaluate XGBoost model and export labeled point cloud')
    ap.add_argument('--features',   default='features.csv')
    ap.add_argument('--labels',     default='labels.csv')
    ap.add_argument('--model',      default='models/xgb_baseline.json')
    ap.add_argument('--out',        default='labeled_pointcloud.csv')
    ap.add_argument('--threshold',  type=float, default=0.50,
                    help='Probability threshold for water class (default 0.50)')
    args = ap.parse_args()
    run(args.features, args.labels, args.model, args.out, args.threshold)


if __name__ == '__main__':
    main()
