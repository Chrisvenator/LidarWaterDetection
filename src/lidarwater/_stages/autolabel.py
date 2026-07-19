"""Autolabel (v6) stage: waveform-only water/land bootstrap classifier.
Trains on elevation-band z-boundaries only, using waveform + reflectance
features exclusively — no geometry, no height_above_*. Used only to
bootstrap the surface-model's Phase 4 retraining in ``fit()``; ``classify()``
does not run this stage (the geometry stage consumes WCN v9 probas instead).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score

from .._models._torch_optional import require_torch
from ..artifacts import ArtifactId, ArtifactResolver
from ..config import ZoneConfig
from ..types import PipelineState

WAVEFORM_FEATURES = [
    "energy_concentration", "max_amp_norm_by_energy",
    "n_clusters", "n_peaks", "n_gaps", "n_samples", "time_span",
    "max_amp", "mean_amp", "std_amp", "total_energy",
    "max_gap", "mean_gap", "total_gap", "first_last_span",
    "energy_ratio_late", "first_peak_amp", "last_peak_amp", "peak_amp_ratio",
    "depth_proxy_m", "amplitude_weighted_center", "active_bins_ratio",
    "reflectance_dB",
]


def create_labels(z: np.ndarray, zones: ZoneConfig) -> np.ndarray:
    """1=water, 0=land, -1=excluded (gap between confidently-labeled bands)."""
    label = np.full(len(z), -1, dtype=np.int8)
    label[z < zones.z_underwater_max] = 1
    label[(z >= zones.z_underwater_max) & (z < zones.z_water_surf_max)] = 1
    label[(z >= zones.z_dry_bed_min) & (z < zones.z_dry_bed_max)] = 0
    label[(z >= zones.z_banks_min) & (z < zones.z_banks_max)] = 0
    label[z >= zones.z_canopy_min] = 0
    return label


def _select_device(device: str):
    torch = require_torch()
    return torch.device("cuda" if (device == "auto" and torch.cuda.is_available())
                        else ("cpu" if device == "auto" else device))


def predict(state: PipelineState, resolver: ArtifactResolver, device: str = "auto") -> PipelineState:
    """Load the deployed v6 XGBoost + V6Net checkpoints and run inference."""
    if state.features is None or state.waveform_grids is None:
        raise ValueError("autolabel predict needs state.features and state.waveform_grids")
    feat_df = state.features
    cols = [c for c in WAVEFORM_FEATURES if c in feat_df.columns]

    x_all = np.nan_to_num(feat_df[cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(resolver.resolve(ArtifactId.V6_XGB))
    xgb_proba = xgb_model.predict_proba(x_all)[:, 1].astype(np.float32)

    torch = require_torch()
    from .._models.nets import WaveformCnnNet

    stats = json.loads(resolver.resolve(ArtifactId.V6_DEEP_STATS).read_text())
    stat_cols = stats["spatial_cols"]
    sp = np.nan_to_num(feat_df[stat_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    sp_norm = (sp - np.array(stats["spatial_mean"], np.float32)) / np.array(stats["spatial_std"], np.float32)
    grids_norm = ((state.waveform_grids.astype(np.float32) - stats["grid_mean"]) / stats["grid_std"])
    grids_norm = np.nan_to_num(grids_norm, nan=0.0, posinf=0.0, neginf=0.0)

    dev = _select_device(device)
    model = WaveformCnnNet(n_spatial=len(stat_cols)).to(dev)
    model.load_state_dict(torch.load(resolver.resolve(ArtifactId.V6_DEEP), map_location=dev,
                                     weights_only=True))
    model.eval()

    n = len(feat_df)
    deep_proba = np.zeros(n, np.float32)
    batch = 2048
    with torch.no_grad():
        for s in range(0, n, batch):
            e = min(s + batch, n)
            wf_b = torch.from_numpy(grids_norm[s:e]).unsqueeze(1).to(dev)
            sp_b = torch.from_numpy(sp_norm[s:e]).to(dev)
            deep_proba[s:e] = torch.softmax(model(wf_b, sp_b), 1)[:, 1].cpu().numpy()

    ensemble = _ensemble(xgb_proba, deep_proba)
    state.autolabel_xgb_proba = xgb_proba
    state.autolabel_deep_proba = deep_proba
    state.autolabel_ensemble = ensemble
    state.metrics["autolabel"] = {"water_fraction": float((ensemble == 1).mean())}
    return state


def _ensemble(xgb_proba: np.ndarray, deep_proba: np.ndarray) -> np.ndarray:
    xgb_pred = (xgb_proba >= 0.5).astype(np.int8)
    deep_pred = (deep_proba >= 0.5).astype(np.int8)
    ensemble = xgb_pred.copy()
    ensemble[xgb_pred != deep_pred] = 2
    return ensemble


def _spatial_cv_split(y_coord: np.ndarray, n_folds: int = 5):
    edges = np.percentile(y_coord, np.linspace(0, 100, n_folds + 1))
    return [
        (np.where((y_coord < edges[f]) | (y_coord > edges[f + 1]))[0],
         np.where((y_coord >= edges[f]) & (y_coord <= edges[f + 1]))[0])
        for f in range(n_folds)
    ]


def _train_xgb(df_train: pd.DataFrame, cols: list[str]) -> tuple[xgb.XGBClassifier, dict]:
    x = np.nan_to_num(df_train[cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = df_train["label"].values.astype(np.int32)
    n_water, n_land = int((y == 1).sum()), int((y == 0).sum())
    spw = round(n_land / max(n_water, 1), 3)

    f1s = []
    for train_i, val_i in _spatial_cv_split(df_train["y"].values):
        if len(val_i) == 0 or len(np.unique(y[val_i])) < 2:
            continue
        clf = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                                random_state=42, n_jobs=-1, verbosity=0)
        clf.fit(x[train_i], y[train_i], verbose=False)
        pred = (clf.predict_proba(x[val_i])[:, 1] >= 0.5).astype(int)
        f1s.append(f1_score(y[val_i], pred, average="macro", zero_division=0))

    final = xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                              random_state=42, n_jobs=-1, verbosity=0)
    final.fit(x, y, verbose=False)
    return final, {"macro_f1_mean": float(np.mean(f1s)) if f1s else None, "n_water": n_water, "n_land": n_land}


def _train_deep(df_train: pd.DataFrame, grids_train: np.ndarray, cols: list[str],
                device: str, epochs: int = 80, batch_size: int = 256, lr: float = 1e-3,
                patience: int = 20, val_frac: float = 0.20):
    torch = require_torch()
    from .._models.nets import FocalLoss, WaveformCnnNet

    spatial = np.nan_to_num(df_train[cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    labels = df_train["label"].values.astype(np.int64)
    y_coord = df_train["y"].values
    grids = np.nan_to_num(grids_train.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    cutoff = np.percentile(y_coord, 100 * (1 - val_frac))
    val_mask = y_coord >= cutoff
    trn_mask = ~val_mask

    g_mean, g_std = float(grids[trn_mask].mean()), float(grids[trn_mask].std()) + 1e-6
    sp_mean, sp_std = spatial[trn_mask].mean(0), spatial[trn_mask].std(0) + 1e-6
    gn, sn = (grids - g_mean) / g_std, (spatial - sp_mean) / sp_std

    dev = _select_device(device)
    n_pos = int((labels[trn_mask] == 1).sum())
    alpha = round((int(trn_mask.sum()) - n_pos) / trn_mask.sum(), 4)

    model = WaveformCnnNet(n_spatial=len(cols)).to(dev)
    criterion = FocalLoss(2.0, alpha, 0.1).to(dev)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=5e-6)

    def loader(mask, shuffle):
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(gn[mask]).unsqueeze(1), torch.from_numpy(sn[mask]),
            torch.from_numpy(labels[mask]))
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader, val_loader = loader(trn_mask, True), loader(val_mask, False)
    best_f1, no_improve = 0.0, 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for _ in range(epochs):
        model.train()
        for wf_b, sp_b, lb_b in train_loader:
            wf_b, sp_b, lb_b = wf_b.to(dev), sp_b.to(dev), lb_b.to(dev)
            loss = criterion(model(wf_b, sp_b), lb_b)
            optim.zero_grad()
            loss.backward()
            optim.step()
        sched.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for wf_b, sp_b, lb_b in val_loader:
                logits = model(wf_b.to(dev), sp_b.to(dev))
                preds.extend(logits.argmax(1).cpu().numpy().tolist())
                trues.extend(lb_b.numpy().tolist())
        vf1 = f1_score(trues, preds, average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1, no_improve = vf1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    stats = {"grid_mean": g_mean, "grid_std": g_std, "spatial_mean": sp_mean.tolist(),
             "spatial_std": sp_std.tolist(), "spatial_cols": cols, "best_val_f1": float(best_f1)}
    return model, stats


def fit(state: PipelineState, zones: ZoneConfig, resolver: ArtifactResolver,
       device: str = "auto") -> PipelineState:
    if state.features is None or state.waveform_grids is None:
        raise ValueError("autolabel fit needs state.features and state.waveform_grids")

    feat_df = state.features
    label = create_labels(feat_df["z"].values, zones)
    train_mask = label != -1
    df_train = feat_df[train_mask].copy()
    df_train["label"] = label[train_mask]
    grids_train = state.waveform_grids[train_mask]

    cols = [c for c in WAVEFORM_FEATURES if c in feat_df.columns]
    xgb_model, xgb_metrics = _train_xgb(df_train, cols)
    xgb_model.save_model(resolver.resolve_for_write(ArtifactId.V6_XGB))

    deep_model, deep_stats = _train_deep(df_train, grids_train, cols, device)
    torch = require_torch()
    torch.save(deep_model.state_dict(), resolver.resolve_for_write(ArtifactId.V6_DEEP))
    resolver.resolve_for_write(ArtifactId.V6_DEEP_STATS).write_text(json.dumps(deep_stats, indent=2))

    state.metrics["autolabel"] = {"xgb": xgb_metrics, "deep_best_val_f1": deep_stats["best_val_f1"]}
    return predict(state, resolver, device=device)
