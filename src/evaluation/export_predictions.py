"""
export_predictions.py — Apply all trained models to the full 234k-point cloud
and export a single CSV with one prediction column per model.

Columns in output:
  x, y, z, reflectance_dB
  auto_label, auto_confidence          (rule-based v2 labels)
  pred_A, conf_A                       (XGBoost Model A: all features)
  pred_B, conf_B                       (XGBoost Model B: no z / z_relative)
  pred_C, conf_C                       (XGBoost Model C: waveform + reflectance only)
  pred_deep, conf_deep                 (1D CNN + MLP, waveform-primary)
  n_gaps, max_gap, n_peaks,
  max_peak_spacing, depth_proxy_m,
  planarity, roughness, z_relative     (scalar fields for CloudCompare colouring)

Usage:
  python export_predictions.py [--no-deep]   # skip CNN if not trained
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import xgboost as xgb

# ── Feature definitions (must match compare_models.py) ──────────────────────

FEAT_A = [
    'max_amp', 'mean_amp', 'std_amp', 'total_energy', 'n_samples',
    'time_span', 'n_gaps', 'max_gap', 'mean_gap', 'total_gap',
    'n_peaks', 'max_peak_spacing', 'mean_peak_spacing',
    'first_last_span', 'n_clusters', 'energy_ratio_late',
    'first_peak_amp', 'last_peak_amp', 'peak_amp_ratio', 'depth_proxy_m',
    'z', 'reflectance_dB',
    'planarity', 'roughness', 'linearity', 'sphericity',
    'height_range_local', 'height_std_local', 'z_relative',
]

FEAT_B = [c for c in FEAT_A if c not in ('z', 'z_relative')]

FEAT_C = [
    'max_amp', 'mean_amp', 'std_amp', 'total_energy', 'n_samples',
    'time_span', 'n_gaps', 'max_gap', 'mean_gap', 'total_gap',
    'n_peaks', 'max_peak_spacing', 'mean_peak_spacing',
    'first_last_span', 'n_clusters', 'energy_ratio_late',
    'first_peak_amp', 'last_peak_amp', 'peak_amp_ratio', 'depth_proxy_m',
    'reflectance_dB',
]

SPATIAL_COLS = [
    'reflectance_dB', 'z_relative', 'planarity',
    'roughness', 'height_range_local',
]


def xgb_predict(feat_df: pd.DataFrame, feat_cols: list, model_path: str):
    """Load an XGBoost model and return (pred_label, pred_proba) arrays."""
    cols = [c for c in feat_cols if c in feat_df.columns]
    X    = feat_df[cols].values.astype(np.float32)
    X    = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(np.int8)
    return pred, proba.astype(np.float32)


def deep_predict(feat_df: pd.DataFrame, grids_path: str,
                 model_path: str, stats_path: str, batch_size: int = 1024):
    """Load WaveformNet and return (pred_label, pred_proba) arrays."""
    import torch
    from deep_model import WaveformNet

    with open(stats_path) as fh:
        stats = json.load(fh)

    g_mean   = float(stats['grid_mean'])
    g_std    = float(stats['grid_std'])
    sp_mean  = np.array(stats['spatial_mean'], dtype=np.float32)
    sp_std   = np.array(stats['spatial_std'],  dtype=np.float32)

    print(f"  Loading waveform grids from {grids_path} …")
    grids_all = np.load(grids_path, mmap_mode='r').astype(np.float32)
    grids_norm = (grids_all - g_mean) / g_std                        # (N, 200)

    sp_vals = feat_df[SPATIAL_COLS].values.astype(np.float32)
    sp_vals = np.nan_to_num(sp_vals, nan=0.0, posinf=0.0, neginf=0.0)
    sp_norm = (sp_vals - sp_mean) / sp_std

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = WaveformNet(n_spatial=len(SPATIAL_COLS), dropout=0.0)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    N       = len(feat_df)
    probas  = np.zeros(N, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end   = min(start + batch_size, N)
            wf_b  = torch.from_numpy(grids_norm[start:end]).unsqueeze(1).to(device)  # (B,1,200)
            sp_b  = torch.from_numpy(sp_norm[start:end]).to(device)                  # (B,5)
            logits = model(wf_b, sp_b)
            proba  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probas[start:end] = proba

            if start % 50_000 == 0:
                print(f"    {start:>7,}/{N:,} …")

    preds = (probas >= 0.5).astype(np.int8)
    return preds, probas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features',   default='features.csv')
    ap.add_argument('--labels',     default='labels.csv')
    ap.add_argument('--grids',      default='waveform_grids.npy')
    ap.add_argument('--models-dir', default='models')
    ap.add_argument('--out',        default='all_predictions.csv')
    ap.add_argument('--no-deep',    action='store_true',
                    help='Skip deep model (useful if not yet trained)')
    args = ap.parse_args()

    print(f"Loading {args.features} …")
    feat_df = pd.read_csv(args.features)
    print(f"  {len(feat_df):,} rows × {len(feat_df.columns)} cols")

    print(f"Loading {args.labels} …")
    lab_df = pd.read_csv(args.labels)

    # ── XGBoost predictions ──────────────────────────────────────────────────
    results = {}
    for name, cols, fname in [
        ('A', FEAT_A, 'xgb_A.json'),
        ('B', FEAT_B, 'xgb_B.json'),
        ('C', FEAT_C, 'xgb_C.json'),
    ]:
        path = os.path.join(args.models_dir, fname)
        if not os.path.exists(path):
            print(f"  Model {name} not found at {path} — skipping")
            continue
        print(f"\nApplying XGBoost Model {name} ({fname}) …")
        pred, proba = xgb_predict(feat_df, cols, path)
        results[f'pred_{name}'] = pred
        results[f'conf_{name}'] = np.round(proba, 4)
        n_water = int(pred.sum())
        print(f"  water={n_water:,}  land={len(pred)-n_water:,}  "
              f"({100*n_water/len(pred):.1f}% water)")

    # ── Deep model predictions ───────────────────────────────────────────────
    model_pt    = os.path.join(args.models_dir, 'deep_model.pt')
    stats_json  = os.path.join(args.models_dir, 'deep_model_stats.json')

    if not args.no_deep and os.path.exists(model_pt) and os.path.exists(stats_json):
        print(f"\nApplying deep model (WaveformNet) …")
        pred_deep, proba_deep = deep_predict(
            feat_df, args.grids, model_pt, stats_json)
        results['pred_deep'] = pred_deep
        results['conf_deep'] = np.round(proba_deep, 4)
        n_water = int(pred_deep.sum())
        print(f"  water={n_water:,}  land={len(pred_deep)-n_water:,}  "
              f"({100*n_water/len(pred_deep):.1f}% water)")
    elif not args.no_deep:
        print(f"\nDeep model not found at {model_pt} — skipping")

    # ── Assemble output ──────────────────────────────────────────────────────
    print(f"\nAssembling output …")

    out_df = pd.DataFrame()
    out_df['x'] = feat_df['x'].values if 'x' in feat_df.columns else 0.0
    out_df['y'] = feat_df['y'].values if 'y' in feat_df.columns else 0.0
    out_df['z'] = feat_df['z'].values if 'z' in feat_df.columns else 0.0
    out_df['reflectance_dB'] = (feat_df['reflectance_dB'].values
                                 if 'reflectance_dB' in feat_df.columns else 0.0)

    out_df['auto_label']      = lab_df['label'].values.astype(np.int8)
    out_df['auto_confidence'] = lab_df['confidence'].values.astype(np.float32)

    for key, arr in results.items():
        out_df[key] = arr

    # Extra scalar fields for CloudCompare colouring
    for col in ['n_gaps', 'max_gap', 'n_peaks', 'max_peak_spacing',
                'depth_proxy_m', 'planarity', 'roughness', 'z_relative',
                'height_range_local']:
        if col in feat_df.columns:
            out_df[col] = feat_df[col].values

    out_df.to_csv(args.out, index=False)
    print(f"\nSaved {len(out_df):,} rows to {args.out}")
    print(f"Columns: {list(out_df.columns)}")

    # ── Comparison summary ───────────────────────────────────────────────────
    pred_cols = [c for c in out_df.columns if c.startswith('pred_')]
    if len(pred_cols) > 1:
        print(f"\n{'='*60}")
        print(f"{'Model':<12} {'%Water':>8} {'Agreement w/ auto':>20}")
        print(f"{'-'*60}")
        confident_mask = out_df['auto_label'].isin([0, 1])
        n_conf = confident_mask.sum()
        for pc in pred_cols:
            n_w = int(out_df[pc].sum())
            pct = 100 * n_w / len(out_df)
            if n_conf > 0:
                agree = (out_df.loc[confident_mask, pc] ==
                         out_df.loc[confident_mask, 'auto_label']).sum()
                agree_pct = 100 * agree / n_conf
                print(f"  {pc:<10} {pct:>7.1f}%  {agree:,}/{n_conf:,} = {agree_pct:.1f}%")
            else:
                print(f"  {pc:<10} {pct:>7.1f}%")

    print(f"\nDone. Open {args.out} in CloudCompare:")
    print(f"  File → Open → CSV → set x/y/z")
    print(f"  Colour by pred_A / pred_B / pred_C / pred_deep to compare models")


if __name__ == '__main__':
    main()
