"""
inference_pipeline.py — Two-stage inference on all 234k points.

Stage 1: canopy detector  → labels canopy points, passes ground to Stage 2
Stage 2: water vs dry     → runs only on Stage-1 ground points

Final classes:
  0 = dry ground / meadow / gravel bank
  1 = water (river surface + riverbed)
  2 = canopy / vegetation

Output: labeled_pointcloud_staged.csv
  x, y, z, reflectance_dB
  stage1_pred, stage1_conf        (0=ground, 1=canopy)
  stage2_pred, stage2_conf        (0=dry, 1=water — NaN for canopy points)
  final_class, final_conf         (0=dry, 1=water, 2=canopy)
  xgb_final, deep_final           (model-specific final class for comparison)
  [scalar fields for CloudCompare colouring]
"""

import json, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn

STAGE1_FEATURES = [
    'height_above_local_min_10m', 'height_above_local_min',
    'height_percentile_local', 'planarity', 'roughness',
    'height_range_local', 'height_std_local', 'linearity', 'sphericity',
    'n_clusters', 'n_peaks', 'total_energy', 'n_samples', 'time_span',
    'energy_concentration', 'active_bins_ratio',
]

STAGE2_XGB_FEATURES = [
    'energy_concentration', 'max_amp_norm_by_energy',
    'n_peaks', 'n_gaps', 'n_clusters', 'n_samples', 'time_span',
    'max_gap', 'mean_gap', 'total_gap', 'first_last_span',
    'max_amp', 'mean_amp', 'std_amp', 'total_energy',
    'first_peak_amp', 'last_peak_amp', 'peak_amp_ratio',
    'energy_ratio_late', 'depth_proxy_m',
    'amplitude_weighted_center', 'active_bins_ratio',
    'reflectance_dB', 'planarity', 'roughness', 'linearity', 'sphericity',
    'height_range_local', 'height_std_local',
    'height_percentile_local', 'z_relative',
]

# Canopy decision threshold — conservative to avoid removing real ground points
CANOPY_THRESHOLD = 0.60


# ── Model loaders ─────────────────────────────────────────────────────────────

class _CB(nn.Module):
    def __init__(self, i, o, k):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv1d(i, o, k, padding='same', bias=False),
            nn.BatchNorm1d(o), nn.ReLU(inplace=True))
    def forward(self, x): return self.b(x)


class Stage2Net(nn.Module):
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


# ── Inference helpers ─────────────────────────────────────────────────────────

def xgb_predict(feat_df, feature_cols, model_path, batch_label=''):
    cols = [c for c in feature_cols if c in feat_df.columns]
    X = np.nan_to_num(feat_df[cols].values.astype(np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    m = xgb.XGBClassifier()
    m.load_model(model_path)
    proba = m.predict_proba(X)[:, 1]
    return proba


@torch.no_grad()
def deep_predict_subset(feat_df_sub, grids_all, orig_rows, stats, model_path,
                         batch_size=2048):
    sp_cols = stats['spatial_cols']
    g_mean  = float(stats['grid_mean']); g_std = float(stats['grid_std'])
    sp_mean = np.array(stats['spatial_mean'], np.float32)
    sp_std  = np.array(stats['spatial_std'],  np.float32)

    grids_sub  = grids_all[orig_rows].astype(np.float32)
    grids_sub  = np.nan_to_num(grids_sub, nan=0.0, posinf=0.0, neginf=0.0)
    gn         = (grids_sub - g_mean) / g_std

    sp_vals = feat_df_sub[sp_cols].values.astype(np.float32)
    sp_vals = np.nan_to_num(sp_vals, nan=0.0, posinf=0.0, neginf=0.0)
    sn      = (sp_vals - sp_mean) / sp_std

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = Stage2Net(n_spatial=len(sp_cols), dropout=0.0)
    model.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=True))
    model.to(device).eval()

    N = len(orig_rows); probas = np.zeros(N, np.float32)
    for s in range(0, N, batch_size):
        e   = min(s + batch_size, N)
        wfb = torch.from_numpy(gn[s:e]).unsqueeze(1).to(device)
        spb = torch.from_numpy(sn[s:e]).to(device)
        probas[s:e] = torch.softmax(model(wfb, spb), 1)[:, 1].cpu().numpy()
        if s % 40_000 == 0 and s > 0:
            print(f"    {s:>6,}/{N:,} …")
    return probas


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    feat_path  = ROOT / 'data_processed' / 'features_v2.csv'
    grids_path = ROOT / 'data_processed' / 'waveform_grids.npy'
    models_dir = ROOT / 'models' / 'v4-staged-cascade'
    out_path   = ROOT / 'pointclouds' / 'labeled_pointcloud_v4_staged.csv'

    print(f"Loading {feat_path} …")
    feat_df = pd.read_csv(feat_path)
    N = len(feat_df)
    print(f"  {N:,} points")

    print(f"Loading {grids_path} …")
    grids_all = np.load(grids_path, mmap_mode='r')

    # ── Stage 1: canopy detection on all points ────────────────────────────────
    s1_path = os.path.join(models_dir, 'stage1.json')
    print(f"\nStage 1: canopy detection …")
    s1_proba = xgb_predict(feat_df, STAGE1_FEATURES, s1_path)
    s1_pred  = (s1_proba >= CANOPY_THRESHOLD).astype(np.int8)
    n_can    = int(s1_pred.sum())
    n_gnd    = N - n_can
    print(f"  Canopy  : {n_can:,}  ({100*n_can/N:.1f}%)")
    print(f"  Ground  : {n_gnd:,}  ({100*n_gnd/N:.1f}%)")
    print(f"  (threshold={CANOPY_THRESHOLD} — conservative to preserve ground points)")

    # Ground point indices
    ground_mask = s1_pred == 0
    ground_rows = np.where(ground_mask)[0]
    feat_ground = feat_df.iloc[ground_rows].reset_index(drop=True)

    # ── Stage 2: water vs dry ground on ground points ─────────────────────────
    print(f"\nStage 2: water vs dry ground on {n_gnd:,} ground points …")

    # XGBoost Stage 2
    s2_xgb_path = os.path.join(models_dir, 'stage2_xgb.json')
    print(f"  Applying XGBoost …")
    s2_xgb_proba = xgb_predict(feat_ground, STAGE2_XGB_FEATURES, s2_xgb_path)
    s2_xgb_pred  = (s2_xgb_proba >= 0.5).astype(np.int8)

    # Deep model Stage 2
    s2_deep_path  = os.path.join(models_dir, 'stage2_deep.pt')
    s2_stats_path = os.path.join(models_dir, 'stage2_deep_stats.json')
    with open(s2_stats_path) as fh:
        s2_stats = json.load(fh)

    print(f"  Applying Deep model …")
    s2_deep_proba = deep_predict_subset(
        feat_ground, grids_all, ground_rows, s2_stats, s2_deep_path)
    s2_deep_pred  = (s2_deep_proba >= 0.5).astype(np.int8)

    print(f"\n  XGBoost  — water={int(s2_xgb_pred.sum()):,}  "
          f"dry={int((s2_xgb_pred==0).sum()):,}")
    print(f"  Deep     — water={int(s2_deep_pred.sum()):,}  "
          f"dry={int((s2_deep_pred==0).sum()):,}")

    # Agreement between Stage 2 models on ground points
    agree = (s2_xgb_pred == s2_deep_pred).sum()
    print(f"  XGBoost/Deep agreement on ground points: "
          f"{agree:,}/{n_gnd:,} = {100*agree/n_gnd:.1f}%")

    # ── Assemble final output ──────────────────────────────────────────────────
    print(f"\nAssembling output …")

    # Stage 1 arrays (all N points)
    stage1_pred_all = s1_pred.copy()
    stage1_conf_all = np.where(s1_pred == 1, s1_proba, 1.0 - s1_proba).astype(np.float32)

    # Stage 2 arrays (all N points, NaN where canopy)
    s2_xgb_all   = np.full(N, -1,  dtype=np.int8)
    s2_xgb_conf  = np.full(N, np.nan, dtype=np.float32)
    s2_deep_all  = np.full(N, -1,  dtype=np.int8)
    s2_deep_conf = np.full(N, np.nan, dtype=np.float32)

    s2_xgb_all[ground_rows]  = s2_xgb_pred
    s2_xgb_conf[ground_rows] = np.abs(s2_xgb_proba - 0.5) * 2   # distance from boundary
    s2_deep_all[ground_rows] = s2_deep_pred
    s2_deep_conf[ground_rows] = np.abs(s2_deep_proba - 0.5) * 2

    # Final class (XGBoost pipeline as primary)
    # 0=dry, 1=water, 2=canopy
    xgb_final  = np.where(s1_pred == 1, 2,
                  np.where(s2_xgb_all == 1, 1, 0)).astype(np.int8)
    deep_final = np.where(s1_pred == 1, 2,
                  np.where(s2_deep_all == 1, 1, 0)).astype(np.int8)

    # Ensemble: agree → confident, disagree → mark as uncertain (3)
    ensemble = xgb_final.copy()
    disagree = (xgb_final != deep_final) & (s1_pred == 0)
    ensemble[disagree] = 3   # uncertain (models disagree)
    print(f"  Model disagreement on ground points: "
          f"{int(disagree.sum()):,}  ({100*disagree.sum()/n_gnd:.1f}%)")

    out = pd.DataFrame({
        'x':             feat_df['x'].values,
        'y':             feat_df['y'].values,
        'z':             feat_df['z'].values,
        'reflectance_dB': feat_df['reflectance_dB'].values
                          if 'reflectance_dB' in feat_df.columns else 0.0,
        # Stage 1
        'stage1_pred':   stage1_pred_all,   # 0=ground, 1=canopy
        'stage1_conf':   np.round(stage1_conf_all, 4),
        # Stage 2 (NaN for canopy)
        'stage2_xgb':    s2_xgb_all,        # 0=dry, 1=water, -1=canopy
        'stage2_xgb_conf': np.round(s2_xgb_conf, 4),
        'stage2_deep':   s2_deep_all,
        'stage2_deep_conf': np.round(s2_deep_conf, 4),
        # Final
        'xgb_final':     xgb_final,         # 0=dry, 1=water, 2=canopy
        'deep_final':    deep_final,
        'ensemble':      ensemble,           # 0=dry, 1=water, 2=canopy, 3=uncertain
    })

    # Scalar fields for CloudCompare colouring
    for col in ['n_gaps', 'max_gap', 'n_peaks', 'depth_proxy_m',
                'planarity', 'roughness', 'height_percentile_local',
                'energy_concentration', 'max_amp_norm_by_energy',
                'amplitude_weighted_center', 'z_relative',
                'height_above_local_min', 'height_above_local_min_10m']:
        if col in feat_df.columns:
            out[col] = feat_df[col].values

    out.to_csv(out_path, index=False)
    print(f"\nSaved {N:,} rows to {out_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FINAL CLASSIFICATION SUMMARY")
    print(f"{'='*60}")
    for model_col, name in [('xgb_final', 'XGBoost pipeline'),
                             ('deep_final', 'Deep pipeline'),
                             ('ensemble',   'Ensemble')]:
        arr = out[model_col].values
        n0  = int((arr == 0).sum())
        n1  = int((arr == 1).sum())
        n2  = int((arr == 2).sum())
        n3  = int((arr == 3).sum())
        print(f"\n  {name}:")
        print(f"    Dry ground  (0): {n0:>7,}  ({100*n0/N:.1f}%)")
        print(f"    Water       (1): {n1:>7,}  ({100*n1/N:.1f}%)")
        print(f"    Canopy      (2): {n2:>7,}  ({100*n2/N:.1f}%)")
        if n3 > 0:
            print(f"    Uncertain   (3): {n3:>7,}  ({100*n3/N:.1f}%)")

    print(f"\nOpen {out_path} in CloudCompare:")
    print(f"  Colour by 'xgb_final' or 'ensemble':")
    print(f"    0 = dry ground (grey/brown)")
    print(f"    1 = water (blue)")
    print(f"    2 = canopy (green)")
    print(f"    3 = uncertain / models disagree (yellow)")
    print(f"\n  Also inspect 'energy_concentration' and 'stage1_conf' as scalar fields")
    print(f"  to understand where each stage is uncertain.")
    print("\nDone.")


if __name__ == '__main__':
    main()
