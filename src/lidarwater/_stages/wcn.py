"""WCN v9 stage: transformer + XGBoost on 11 generalizable scalar features.

``predict`` (used by ``classify()``) loads the deployed checkpoints and runs
inference only — this is the faithful, verified path. ``fit`` (used by
``fit()``) reproduces the three-phase training (masked-autoencoder
pretraining, focal-loss fine-tuning, pseudo-label refinement) plus the
XGBoost head; training is stochastic and self-bootstrapping in the original
pipeline, so exact numeric parity isn't guaranteed the way ``predict`` is.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score

from .._models._torch_optional import require_torch
from ..artifacts import ArtifactId, ArtifactResolver
from ..config import WcnConfig
from ..types import PipelineState

SCALAR_FEATURES = [
    "energy_concentration", "max_amp_norm_by_energy", "energy_ratio_late",
    "active_bins_ratio", "peak_amp_ratio", "gap_ratio", "energy_center_norm",
    "n_peaks", "n_gaps", "n_clusters", "depth_proxy_m",
]


def _select_device(device: str):
    torch = require_torch()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _scalars(features: pd.DataFrame) -> np.ndarray:
    return features[SCALAR_FEATURES].values.astype(np.float32)


def _normalise_scalars(scalars: np.ndarray, stats: dict | None = None):
    if stats is None:
        mean = scalars.mean(axis=0, keepdims=True).astype(np.float32)
        std = scalars.std(axis=0, keepdims=True).clip(min=1e-6).astype(np.float32)
        stats = {"mean": mean.tolist(), "std": std.tolist()}
    else:
        mean = np.array(stats["mean"], dtype=np.float32)
        std = np.array(stats["std"], dtype=np.float32)
    return ((scalars - mean) / std).astype(np.float32), stats


def _build_model(config: WcnConfig, device):
    from .._models.nets import WCNv9

    arch = config.arch
    return WCNv9(
        n_scalar=arch.n_scalar, d_model=arch.d_model, n_heads=arch.n_heads,
        n_layers=arch.n_layers, n_patches=arch.n_patches, seq_len=arch.seq_len,
    ).to(device)


def _run_inference_batch(model, wf_norm, scalars_norm, indices, device, batch_size=2048):
    torch = require_torch()
    model.eval()
    probas = np.zeros(len(indices), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(indices), batch_size):
            e = min(s + batch_size, len(indices))
            idx = indices[s:e]
            amp = wf_norm[idx]
            wf2 = np.stack([amp, (amp > 0).astype(np.float32)], axis=1)
            logits, _ = model(torch.from_numpy(wf2).to(device), torch.from_numpy(scalars_norm[idx]).to(device))
            probas[s:e] = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probas


def predict(state: PipelineState, config: WcnConfig, resolver: ArtifactResolver,
           device: str = "auto") -> PipelineState:
    """Load the deployed WCNv9 + XGBoost checkpoints and run inference on
    every point in state.features. Sets state.wcn_proba (transformer) and
    state.wcn_xgb_proba (XGBoost on the same 11 features)."""
    if state.features is None or state.waveform_grids_norm is None:
        raise ValueError("wcn predict needs state.features and state.waveform_grids_norm")

    torch = require_torch()
    device = _select_device(device)

    stats_path = resolver.resolve(ArtifactId.WCN_STATS)
    stats = json.loads(stats_path.read_text())
    scalars_norm, _ = _normalise_scalars(_scalars(state.features), stats=stats["scalar_stats"])

    model = _build_model(config, device)
    model.load_state_dict(torch.load(resolver.resolve(ArtifactId.WCN_REFINED),
                                     map_location=device, weights_only=True))
    model.eval()

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(resolver.resolve(ArtifactId.WCN_XGB))

    n = len(state.features)
    wcn_proba = _run_inference_batch(model, state.waveform_grids_norm, scalars_norm,
                                     np.arange(n), device)
    xgb_proba = xgb_model.predict_proba(scalars_norm)[:, 1].astype(np.float32)

    state.wcn_proba = wcn_proba
    state.wcn_xgb_proba = xgb_proba
    state.metrics["wcn"] = {
        "water_fraction_wcn": float((wcn_proba >= 0.5).mean()),
        "water_fraction_xgb": float((xgb_proba >= 0.5).mean()),
    }
    return state


def _spatial_cv_split(y_coord: np.ndarray, n_folds: int = 5):
    edges = np.percentile(y_coord, np.linspace(0, 100, n_folds + 1))
    folds = []
    for f in range(n_folds):
        val = np.where((y_coord >= edges[f]) & (y_coord <= edges[f + 1]))[0]
        train = np.where((y_coord < edges[f]) | (y_coord > edges[f + 1]))[0]
        folds.append((train, val))
    return folds


def _train_xgb_head(scalars_norm: np.ndarray, labels: np.ndarray, y_coord: np.ndarray) -> xgb.XGBClassifier:
    lab_mask = (labels == 0) | (labels == 1)
    x, y = scalars_norm[lab_mask], (labels[lab_mask] == 1).astype(np.int32)
    n_water, n_land = int(y.sum()), int((y == 0).sum())
    spw = round(n_land / max(n_water, 1), 3)
    model = xgb.XGBClassifier(
        n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    model.fit(x, y)
    return model


def _pretrain(model, wf_norm: np.ndarray, train_cfg, device) -> None:
    """Phase 1 — masked waveform autoencoder (self-supervised, no labels)."""
    torch = require_torch()
    from .._models.nets import MaskedWaveformAutoencoder

    mae = MaskedWaveformAutoencoder(model, train_cfg.phase1_mask_ratio).to(device)
    ds = _WaveformDataset(wf_norm)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=train_cfg.phase1_batch, shuffle=True, num_workers=2,
        pin_memory=(device.type == "cuda"))
    optim = torch.optim.AdamW(mae.parameters(), lr=train_cfg.phase1_lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=train_cfg.phase1_epochs, eta_min=1e-5)

    for _ in range(train_cfg.phase1_epochs):
        mae.train()
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            loss = mae(batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
        sched.step()


def _finetune(model, wf_norm, scalars_norm, labels, weights, train_cfg, device) -> dict:
    """Phase 2 — supervised fine-tuning with confidence-weighted focal loss."""
    torch = require_torch()
    from .._models.nets import FocalLoss

    criterion = FocalLoss(train_cfg.focal_gamma, train_cfg.focal_alpha, train_cfg.focal_smooth).to(device)
    lab_idx = np.where((labels == 0) | (labels == 1))[0]
    rng = np.random.default_rng(train_cfg.seed)
    rng.shuffle(lab_idx)
    n_val = max(1, int(len(lab_idx) * train_cfg.val_fraction))
    val_idx, train_idx = lab_idx[:n_val], lab_idx[n_val:]

    def make_loader(idx, shuffle):
        ds = _LabeledDataset(wf_norm, scalars_norm, labels, weights, idx)
        return torch.utils.data.DataLoader(ds, batch_size=train_cfg.phase2_batch, shuffle=shuffle,
                                           num_workers=2, pin_memory=(device.type == "cuda"))

    train_loader, val_loader = make_loader(train_idx, True), make_loader(val_idx, False)

    for layer in list(model.transformer.layers[:2]):
        for p in layer.parameters():
            p.requires_grad = False
    best_f1 = _run_finetune_epochs(model, train_loader, val_loader, criterion,
                                   train_cfg.phase2_epochs_frozen, train_cfg.phase2_lr_frozen, device)
    for p in model.parameters():
        p.requires_grad = True
    best_f1 = _run_finetune_epochs(model, train_loader, val_loader, criterion,
                                   train_cfg.phase2_epochs_full, train_cfg.phase2_lr_full, device,
                                   patience=train_cfg.phase2_patience, track_best=best_f1)
    return {"phase2_best_val_f1": best_f1}


def _run_finetune_epochs(model, train_loader, val_loader, criterion, epochs, lr, device,
                         patience: int | None = None, track_best: float = 0.0) -> float:
    torch = require_torch()
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=1e-6)
    best_f1, no_improve = track_best, 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for _ in range(epochs):
        model.train()
        for wf_b, sc_b, lb_b, wt_b in train_loader:
            wf_b, sc_b = wf_b.to(device), sc_b.to(device)
            lb_b, wt_b = lb_b.to(device), wt_b.to(device)
            logits, _ = model(wf_b, sc_b)
            loss = criterion(logits, lb_b, wt_b)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        sched.step()

        vf1 = _validate_f1(model, val_loader, device)
        if vf1 > best_f1:
            best_f1, no_improve = vf1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif patience is not None:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return best_f1


def _validate_f1(model, loader, device) -> float:
    torch = require_torch()
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for wf_b, sc_b, lb_b, *_ in loader:
            logits, _ = model(wf_b.to(device), sc_b.to(device))
            preds.extend((torch.softmax(logits, 1)[:, 1].cpu().numpy() >= 0.5).tolist())
            trues.extend(lb_b.numpy().tolist())
    return f1_score(trues, preds, average="macro", zero_division=0)


def _refine(model, wf_norm, scalars_norm, labels, weights, train_cfg, device) -> dict:
    """Phase 3 — iterative pseudo-labeling of uncertain points."""
    torch = require_torch()
    from .._models.nets import FocalLoss

    criterion = FocalLoss(train_cfg.focal_gamma, train_cfg.focal_alpha, train_cfg.focal_smooth).to(device)
    pseudo_labels, pseudo_weights = labels.copy(), weights.copy()
    metrics: dict = {}

    for rnd in range(1, train_cfg.phase3_rounds + 1):
        uncertain_idx = np.where(pseudo_labels == 2)[0]
        if len(uncertain_idx) == 0:
            break
        probas = _run_inference_batch(model, wf_norm, scalars_norm, uncertain_idx, device)
        new_water = uncertain_idx[probas >= train_cfg.phase3_proba_hi]
        new_land = uncertain_idx[probas <= train_cfg.phase3_proba_lo]
        pseudo_labels[new_water], pseudo_labels[new_land] = 1, 0
        pseudo_weights[new_water] = probas[probas >= train_cfg.phase3_proba_hi]
        pseudo_weights[new_land] = 1.0 - probas[probas <= train_cfg.phase3_proba_lo]

        lab_idx = np.where((pseudo_labels == 0) | (pseudo_labels == 1))[0]
        rng = np.random.default_rng(train_cfg.seed + rnd)
        rng.shuffle(lab_idx)
        n_val = max(1, int(len(lab_idx) * train_cfg.val_fraction))
        val_idx, train_idx = lab_idx[:n_val], lab_idx[n_val:]

        train_loader = torch.utils.data.DataLoader(
            _LabeledDataset(wf_norm, scalars_norm, pseudo_labels, pseudo_weights, train_idx),
            batch_size=train_cfg.phase3_batch, shuffle=True, num_workers=2,
            pin_memory=(device.type == "cuda"))
        val_loader = torch.utils.data.DataLoader(
            _LabeledDataset(wf_norm, scalars_norm, pseudo_labels, pseudo_weights, val_idx),
            batch_size=train_cfg.phase3_batch, shuffle=False, num_workers=2,
            pin_memory=(device.type == "cuda"))

        best_f1 = _run_finetune_epochs(model, train_loader, val_loader, criterion,
                                       train_cfg.phase3_epochs, train_cfg.phase3_lr, device)
        metrics[f"round{rnd}_best_f1"] = best_f1
    return metrics


class _WaveformDataset:
    def __init__(self, wf_norm: np.ndarray):
        self.wf = wf_norm

    def __len__(self):
        return len(self.wf)

    def __getitem__(self, i):
        torch = require_torch()
        amp = self.wf[i]
        mask = (amp > 0).astype(np.float32)
        return torch.from_numpy(np.stack([amp, mask], axis=0))


class _LabeledDataset:
    def __init__(self, wf_norm, scalars, labels, weights, indices):
        self.wf, self.scalars = wf_norm, scalars
        self.labels, self.weights, self.idx = labels, weights, indices

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        torch = require_torch()
        n = self.idx[i]
        amp = self.wf[n]
        wf2 = np.stack([amp, (amp > 0).astype(np.float32)], axis=0)
        return (torch.from_numpy(wf2), torch.from_numpy(self.scalars[n]),
                int(self.labels[n]), np.float32(self.weights[n]))


def fit(state: PipelineState, config: WcnConfig, resolver: ArtifactResolver,
       bootstrap_labels: np.ndarray, bootstrap_confidence: np.ndarray,
       device: str = "auto") -> PipelineState:
    """Train WCNv9 (3 phases) + the XGBoost head from bootstrap labels
    (typically the v8 surface model's ``merged_label``/mean-confidence),
    then run inference and populate state.wcn_proba / state.wcn_xgb_proba.

    Training is stochastic; unlike ``predict``, exact reproduction of the
    deployed checkpoints' numbers isn't expected. Seeded for repeatability.
    """
    torch = require_torch()
    if state.features is None or state.waveform_grids_norm is None:
        raise ValueError("wcn fit needs state.features and state.waveform_grids_norm")

    train_cfg = config.train
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    device = _select_device(device)

    scalars_norm, scalar_stats = _normalise_scalars(_scalars(state.features))
    weights = np.where(bootstrap_labels == 1, bootstrap_confidence, 1.0).astype(np.float32)

    model = _build_model(config, device)
    _pretrain(model, state.waveform_grids_norm, train_cfg, device)
    p2_metrics = _finetune(model, state.waveform_grids_norm, scalars_norm,
                           bootstrap_labels, weights, train_cfg, device)
    xgb_model = _train_xgb_head(scalars_norm, bootstrap_labels, state.features["y"].values)
    p3_metrics = _refine(model, state.waveform_grids_norm, scalars_norm,
                         bootstrap_labels, weights, train_cfg, device)

    refined_path = resolver.resolve_for_write(ArtifactId.WCN_REFINED)
    torch.save(model.state_dict(), refined_path)
    xgb_model.save_model(resolver.resolve_for_write(ArtifactId.WCN_XGB))
    stats_path = resolver.resolve_for_write(ArtifactId.WCN_STATS)
    stats_path.write_text(json.dumps({
        "scalar_stats": scalar_stats, "scalar_features": SCALAR_FEATURES,
        **p2_metrics, **p3_metrics,
    }, indent=2))

    n = len(state.features)
    state.wcn_proba = _run_inference_batch(model, state.waveform_grids_norm, scalars_norm,
                                           np.arange(n), device)
    state.wcn_xgb_proba = xgb_model.predict_proba(scalars_norm)[:, 1].astype(np.float32)
    state.metrics["wcn"] = {**p2_metrics, **p3_metrics}
    return state
