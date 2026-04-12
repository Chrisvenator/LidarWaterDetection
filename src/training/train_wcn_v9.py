"""
train_wcn_v9.py — WaveformContextNet v9 training pipeline.

Three phases:
  Phase 1  Masked waveform autoencoder  (self-supervised, no labels, ~50 epochs)
  Phase 2  Supervised fine-tuning       (confidence-weighted focal loss, ~150 epochs)
  Phase 3  Pseudo-label refinement      (2 rounds × 30 epochs)

Then trains XGBoost on the same 11 generalizable scalar features.

Prerequisites (run first):
    python src/training/preprocess_wcn.py

Inputs
------
data_processed/waveform_grids_norm.npy      (234024, 200) float32
data_processed/features_v9.csv              x, y, z + 11 features
pointclouds/labeled_pointcloud_current.csv  merged_label, xgb_proba, deep_proba

Outputs
-------
models/wcn_v9/
  wcn_pretrained.pt   waveform encoder after Phase 1
  wcn_finetuned.pt    full model after Phase 2
  wcn_refined.pt      full model after Phase 3 (deploy this)
  wcn_xgb.json        XGBoost on 11 features
  wcn_stats.json      scalar normalisation stats + metrics
  training_curves.png loss / F1 curves
  topdown_scatter.png top-down point cloud coloured by prediction
pointclouds/labeled_pointcloud_wcn.csv
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT        = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from labeling.river_boundary import (  # noqa: E402
    CANOPY_Z_MAX, PROB_INNER, PROB_CENTER, PROB_OUTER,
    rasterize, fill_and_smooth, extract_contours, _draw_contours,
)
GRIDS_PATH  = ROOT / "data_processed" / "waveform_grids_norm.npy"
FEAT_PATH   = ROOT / "data_processed" / "features_v9.csv"
LABELS_PATH = ROOT / "pointclouds"    / "labeled_pointcloud_wcn.csv"
MODEL_DIR   = ROOT / "models"         / "wcn_v9"
OUT_CSV     = ROOT / "pointclouds"    / "labeled_pointcloud_wcn.csv"

# ── Feature names (must match preprocess_wcn.py) ──────────────────────────────
SCALAR_FEATURES = [
    "energy_concentration", "max_amp_norm_by_energy", "energy_ratio_late",
    "active_bins_ratio", "peak_amp_ratio",
    "gap_ratio", "energy_center_norm",
    "n_peaks", "n_gaps", "n_clusters",
    "depth_proxy_m",
]
N_SCALAR = len(SCALAR_FEATURES)   # 11

# ── Model hyperparameters ─────────────────────────────────────────────────────
D_MODEL    = 128
N_HEADS    = 8
N_LAYERS   = 6
N_PATCHES  = 50    # 200 bins / 4-bin stride
SEQ_LEN    = N_PATCHES + 1  # +1 for CLS token

# ── Training hyperparameters ──────────────────────────────────────────────────
PHASE1_EPOCHS   = 50
PHASE1_BATCH    = 1024
PHASE1_LR       = 1e-3
PHASE1_MASK_R   = 0.40   # fraction of patches to mask

PHASE2_EPOCHS_A = 30     # frozen lower layers
PHASE2_EPOCHS_B = 120    # all unfrozen
PHASE2_BATCH    = 512
PHASE2_LR_A     = 5e-4
PHASE2_LR_B     = 2e-4
PHASE2_PATIENCE = 20     # early stopping patience on val F1

PHASE3_ROUNDS   = 2
PHASE3_EPOCHS   = 30
PHASE3_BATCH    = 512
PHASE3_LR       = 5e-5
PHASE3_PROBA_HI = 0.92   # threshold to assign pseudo-label=water
PHASE3_PROBA_LO = 0.08   # threshold to assign pseudo-label=land

FOCAL_GAMMA     = 2.0
FOCAL_ALPHA     = 0.65   # upweights minority class (water)
FOCAL_SMOOTH    = 0.05

AUX_WEIGHT_EC   = 0.05   # auxiliary loss weight for energy_concentration prediction
AUX_WEIGHT_DP   = 0.05   # auxiliary loss weight for depth_proxy_m prediction

VAL_FRACTION    = 0.20   # random 80/20 split for deep model validation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

def _sinusoidal_pe(n_pos: int, d_model: int) -> torch.Tensor:
    """Sinusoidal positional encoding. Returns (n_pos, d_model) float32."""
    pe  = torch.zeros(n_pos, d_model)
    pos = torch.arange(n_pos, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class _ResBlock(nn.Module):
    """Residual block for the scalar MLP branch."""
    def __init__(self, d: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(d, d), nn.BatchNorm1d(d), nn.GELU(),
            nn.Linear(d, d), nn.BatchNorm1d(d),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class WCNv9(nn.Module):
    """
    WaveformContextNet v9.

    Input
    -----
    wf     : (B, 2, 200)  Channel 0 = per-sample-max-normalised amplitude,
                          Channel 1 = binary activity mask (1 where amp > 0).
    scalar : (B, 11)      Z-score-normalised generalizable scalar features.

    Output
    ------
    logits : (B, 2)       Raw class logits (land=0, water=1).
    wf_emb : (B, 128)     CLS token embedding (used for auxiliary heads in training).
    """

    def __init__(self, n_scalar: int = N_SCALAR, d_model: int = D_MODEL,
                 n_heads: int = N_HEADS, n_layers: int = N_LAYERS):
        super().__init__()

        # ── Waveform branch ───────────────────────────────────────────────────
        # 2-channel → d_model patches (4-bin stride, 50 patches)
        self.patch_embed = nn.Sequential(
            nn.Conv1d(2, d_model, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Sinusoidal PE for SEQ_LEN=51 positions (pos 0=CLS, 1..50=patches)
        self.register_buffer("pos_embed", _sinusoidal_pe(SEQ_LEN, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=512, dropout=0.1,
            batch_first=True, norm_first=True,   # Pre-LayerNorm
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.wf_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())

        # ── Scalar branch ─────────────────────────────────────────────────────
        self.scalar_mlp = nn.Sequential(
            nn.Linear(n_scalar, 128), nn.BatchNorm1d(128), nn.GELU(),
            _ResBlock(128),
            _ResBlock(128),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(),
        )

        # ── Fusion head ───────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(d_model + 64, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 2),
        )

        # ── Auxiliary regression heads (training only, discarded at inference) ─
        self.aux_energy_conc = nn.Linear(d_model, 1)
        self.aux_depth_proxy = nn.Linear(d_model, 1)

        # Initialise weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_waveform(self, wf: torch.Tensor) -> torch.Tensor:
        """wf: (B, 2, 200) → CLS embedding (B, d_model)."""
        B = wf.size(0)
        x = self.patch_embed(wf)                           # (B, d_model, 50)
        x = x.permute(0, 2, 1)                            # (B, 50, d_model)
        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)                  # (B, 51, d_model)
        x   = x + self.pos_embed.unsqueeze(0)             # add sinusoidal PE
        x   = self.transformer(x)                         # (B, 51, d_model)
        return self.wf_proj(x[:, 0, :])                   # CLS → (B, d_model)

    def forward(self, wf: torch.Tensor, scalar: torch.Tensor):
        wf_emb = self.encode_waveform(wf)                 # (B, 128)
        sp_emb = self.scalar_mlp(scalar)                  # (B, 64)
        fused  = torch.cat([wf_emb, sp_emb], dim=1)      # (B, 192)
        logits = self.head(fused)                         # (B, 2)
        return logits, wf_emb


class _MaskedWaveformAE(nn.Module):
    """
    Masked Waveform Autoencoder — Phase 1 only.

    Wraps a WCNv9 instance and adds a mask token + decoder.
    Training this module updates the shared patch_embed and transformer
    inside the WCNv9 model (weights are shared by reference).
    Discard this wrapper after Phase 1; the WCNv9 retains pre-trained weights.
    """

    def __init__(self, model: WCNv9):
        super().__init__()
        self.model      = model
        self.mask_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        # Decoder: from each patch token → 4 raw amplitude bins
        self.decoder = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, 256), nn.GELU(),
            nn.Linear(256, 4),
        )
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, wf: torch.Tensor, mask_ratio: float = PHASE1_MASK_R):
        """
        wf: (B, 2, 200) → scalar reconstruction loss (MSE on masked patches).
        """
        B  = wf.size(0)
        n  = N_PATCHES  # 50

        # Embed patches (shares weights with WCNv9)
        x = self.model.patch_embed(wf)            # (B, D_MODEL, 50)
        x = x.permute(0, 2, 1)                   # (B, 50, D_MODEL)

        # Add patch positional encoding (skip CLS at position 0)
        x = x + self.model.pos_embed[1:].unsqueeze(0)  # (B, 50, D_MODEL)

        # Random patch mask (choose which patches to mask)
        n_mask     = max(1, int(mask_ratio * n))
        noise      = torch.rand(B, n, device=wf.device)
        ids_sorted = noise.argsort(dim=1)          # (B, 50) ascending
        is_masked  = torch.zeros(B, n, dtype=torch.bool, device=wf.device)
        is_masked.scatter_(1, ids_sorted[:, :n_mask], True)   # (B, 50) True=masked

        # Replace masked patches with mask token
        mask_exp = self.mask_token.expand(B, n, -1)           # (B, 50, D_MODEL)
        x = torch.where(is_masked.unsqueeze(-1), mask_exp, x)

        # Prepend CLS with its positional encoding
        cls_pe = (self.model.cls_token + self.model.pos_embed[0]).expand(B, -1, -1)
        x      = torch.cat([cls_pe, x], dim=1)   # (B, 51, D_MODEL)

        # Transformer forward (shares weights with WCNv9)
        out         = self.model.transformer(x)  # (B, 51, D_MODEL)
        patch_out   = out[:, 1:, :]              # (B, 50, D_MODEL)

        # Reconstruct amplitudes at masked patch positions
        recon       = self.decoder(patch_out).reshape(B, 200)  # (B, 200)
        target      = wf[:, 0, :]                              # (B, 200) amplitude channel

        # Loss only over masked bins
        mask_bins = is_masked.unsqueeze(-1).expand(-1, -1, 4).reshape(B, 200)
        if not mask_bins.any():
            return recon.sum() * 0.0
        return F.mse_loss(recon[mask_bins], target[mask_bins])


# ══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

class _FocalLoss(nn.Module):
    """
    Focal loss with label smoothing and per-sample confidence weighting.

    gamma   : focusing parameter (reduces weight of easy examples)
    alpha   : class weight for the positive (water) class
    smooth  : label smoothing factor
    """

    def __init__(self, gamma: float = FOCAL_GAMMA,
                 alpha: float = FOCAL_ALPHA,
                 smooth: float = FOCAL_SMOOTH):
        super().__init__()
        self.gamma  = gamma
        self.alpha  = alpha
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                weights: torch.Tensor | None = None) -> torch.Tensor:
        # logits:  (B, 2)
        # targets: (B,) int64 — 0=land, 1=water
        # weights: (B,) float — per-sample confidence weight (optional)
        n_cls = logits.size(1)
        lp = F.log_softmax(logits, dim=1)   # (B, 2)
        p  = lp.exp()

        # Smooth labels
        smooth_tgt = torch.full_like(logits, self.smooth / n_cls)
        smooth_tgt.scatter_(1, targets.unsqueeze(1),
                            1.0 - self.smooth + self.smooth / n_cls)

        # Cross-entropy with smooth labels: (B,)
        ce = -(smooth_tgt * lp).sum(dim=1)

        # Focal weight: down-weight easy examples
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)   # (B,)
        focal = (1.0 - pt) ** self.gamma

        # Class balance weight
        cls_w = torch.where(targets == 1,
                            torch.full_like(pt, self.alpha),
                            torch.full_like(pt, 1.0 - self.alpha))

        loss = focal * cls_w * ce   # (B,)

        if weights is not None:
            loss = loss * weights

        return loss.mean()


# ══════════════════════════════════════════════════════════════════════════════
# DATASETS
# ══════════════════════════════════════════════════════════════════════════════

class _WaveformDataset(Dataset):
    """Phase 1: waveforms only, no labels."""

    def __init__(self, wf_norm: np.ndarray):
        self.wf = wf_norm   # (N, 200) float32

    def __len__(self):
        return len(self.wf)

    def __getitem__(self, i):
        amp  = self.wf[i]                                          # (200,)
        mask = (amp > 0).astype(np.float32)                       # (200,)
        return torch.from_numpy(np.stack([amp, mask], axis=0))    # (2, 200)


class _LabeledDataset(Dataset):
    """
    Phase 2 / 3: labeled or pseudo-labeled points.

    Returns
    -------
    wf      : (2, 200) float32
    scalar  : (11,)    float32   z-score normalised
    label   : int      0=land, 1=water
    weight  : float    confidence weight
    aux_ec  : float    energy_concentration (auxiliary regression target)
    aux_dp  : float    depth_proxy_m        (auxiliary regression target)
    """

    def __init__(self, wf_norm: np.ndarray, scalars: np.ndarray,
                 labels: np.ndarray, weights: np.ndarray,
                 aux_ec: np.ndarray, aux_dp: np.ndarray,
                 indices: np.ndarray):
        self.wf      = wf_norm
        self.scalars = scalars
        self.labels  = labels
        self.weights = weights
        self.aux_ec  = aux_ec
        self.aux_dp  = aux_dp
        self.idx     = indices

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        n   = self.idx[i]
        amp = self.wf[n]
        wf2 = np.stack([amp, (amp > 0).astype(np.float32)], axis=0)  # (2, 200)
        return (
            torch.from_numpy(wf2),
            torch.from_numpy(self.scalars[n]),
            int(self.labels[n]),
            np.float32(self.weights[n]),
            np.float32(self.aux_ec[n]),
            np.float32(self.aux_dp[n]),
        )


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _spatial_cv_split(y_coord: np.ndarray, n_folds: int = 5):
    """5-fold spatial CV split by y-coordinate strips."""
    edges = np.percentile(y_coord, np.linspace(0, 100, n_folds + 1))
    folds = []
    for f in range(n_folds):
        val  = np.where((y_coord >= edges[f]) & (y_coord <= edges[f + 1]))[0]
        train = np.where((y_coord < edges[f]) | (y_coord > edges[f + 1]))[0]
        folds.append((train, val))
    return folds


def _normalise_scalars(scalars: np.ndarray, stats: dict | None = None):
    """
    Z-score normalise the (N, 11) scalar feature matrix.
    If stats is None, compute from data and return (normalised, stats).
    If stats provided, apply existing stats (for inference).
    """
    if stats is None:
        mean = scalars.mean(axis=0, keepdims=True).astype(np.float32)  # (1, 11)
        std  = scalars.std(axis=0, keepdims=True).clip(min=1e-6).astype(np.float32)
        stats = {"mean": mean.tolist(), "std": std.tolist()}
    else:
        mean = np.array(stats["mean"], dtype=np.float32)
        std  = np.array(stats["std"],  dtype=np.float32)

    return ((scalars - mean) / std).astype(np.float32), stats


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — MASKED WAVEFORM AUTOENCODER
# ══════════════════════════════════════════════════════════════════════════════

def phase1_pretrain(model: WCNv9, wf_norm: np.ndarray,
                    out_dir: Path, device: torch.device) -> None:
    """Pre-train the waveform encoder with masked patch prediction."""
    print("\n" + "=" * 60)
    print("PHASE 1 — Masked Waveform Autoencoder")
    print("=" * 60)

    mae      = _MaskedWaveformAE(model).to(device)
    ds       = _WaveformDataset(wf_norm)
    loader   = DataLoader(ds, batch_size=PHASE1_BATCH, shuffle=True,
                          num_workers=2, pin_memory=(device.type == "cuda"))

    optim    = torch.optim.AdamW(mae.parameters(), lr=PHASE1_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=PHASE1_EPOCHS, eta_min=1e-5)

    history = []
    for epoch in range(1, PHASE1_EPOCHS + 1):
        mae.train()
        epoch_loss = 0.0
        for batch in loader:
            wf_batch = batch.to(device, non_blocking=True)
            loss = mae(wf_batch, mask_ratio=PHASE1_MASK_R)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
            optim.step()
            epoch_loss += loss.item() * len(wf_batch)

        scheduler.step()
        avg = epoch_loss / len(ds)
        history.append(avg)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{PHASE1_EPOCHS}  "
                  f"loss={avg:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

    # Save only the WCNv9 weights (MAE decoder discarded)
    pt_path = out_dir / "wcn_pretrained.pt"
    torch.save(model.state_dict(), pt_path)
    print(f"\n  Saved pre-trained encoder → {pt_path}")

    # Plot loss
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, PHASE1_EPOCHS + 1), history, color="#2196F3")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss")
    ax.set_title("Phase 1 — Masked Waveform Reconstruction Loss")
    plt.tight_layout()
    plt.savefig(out_dir / "phase1_loss.png", dpi=150); plt.close()
    print(f"  Loss curve → {out_dir / 'phase1_loss.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — SUPERVISED FINE-TUNING
# ══════════════════════════════════════════════════════════════════════════════

def phase2_finetune(
        model: WCNv9,
        wf_norm: np.ndarray,
        scalars_norm: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        aux_ec: np.ndarray,
        aux_dp: np.ndarray,
        out_dir: Path,
        device: torch.device,
) -> dict:
    """
    Supervised fine-tuning on v8 labels.

    Returns dict with training metrics.
    """
    print("\n" + "=" * 60)
    print("PHASE 2 — Supervised Fine-tuning")
    print("=" * 60)

    criterion = _FocalLoss().to(device)

    # 80/20 random split (indices into labeled subset)
    lab_idx = np.where((labels == 0) | (labels == 1))[0]
    np.random.shuffle(lab_idx)
    n_val  = max(1, int(len(lab_idx) * VAL_FRACTION))
    val_idx   = lab_idx[:n_val]
    train_idx = lab_idx[n_val:]
    print(f"  Train: {len(train_idx):,}  Val: {len(val_idx):,}  "
          f"(water train: {(labels[train_idx]==1).sum():,}  "
          f"land train: {(labels[train_idx]==0).sum():,})")

    def _make_loader(idx, shuffle):
        ds = _LabeledDataset(wf_norm, scalars_norm, labels, weights,
                             aux_ec, aux_dp, idx)
        return DataLoader(ds, batch_size=PHASE2_BATCH, shuffle=shuffle,
                          num_workers=2, pin_memory=(device.type == "cuda"))

    train_loader = _make_loader(train_idx, True)
    val_loader   = _make_loader(val_idx,   False)

    total_epochs = PHASE2_EPOCHS_A + PHASE2_EPOCHS_B
    history      = {"train_loss": [], "val_loss": [], "val_f1": [], "val_auc": []}
    best_f1      = 0.0
    no_improve   = 0

    def _run_epoch(loader, train: bool, optim=None, freeze_lower: bool = False):
        model.train(train)
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch in loader:
            wf_b, sc_b, lb_b, wt_b, ec_b, dp_b = batch
            wf_b  = wf_b.to(device, non_blocking=True)
            sc_b  = sc_b.to(device, non_blocking=True)
            lb_b  = lb_b.to(device, non_blocking=True)
            wt_b  = wt_b.float().to(device, non_blocking=True)
            ec_b  = ec_b.float().to(device, non_blocking=True)
            dp_b  = dp_b.float().to(device, non_blocking=True)

            with torch.set_grad_enabled(train):
                logits, wf_emb = model(wf_b, sc_b)

                # Primary focal loss
                loss = criterion(logits, lb_b, wt_b)

                # Auxiliary losses (training only)
                if train:
                    aux_ec_pred = model.aux_energy_conc(wf_emb).squeeze(1)
                    aux_dp_pred = model.aux_depth_proxy(wf_emb).squeeze(1)
                    loss = (loss
                            + AUX_WEIGHT_EC * F.mse_loss(aux_ec_pred, ec_b)
                            + AUX_WEIGHT_DP * F.mse_loss(
                                aux_dp_pred[lb_b == 1], dp_b[lb_b == 1]
                            ) if (lb_b == 1).any() else loss)

            if train:
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

            total_loss += loss.item() * len(lb_b)
            proba = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
            all_preds.extend((proba >= 0.5).astype(int).tolist())
            all_labels.extend(lb_b.cpu().numpy().tolist())

        avg_loss = total_loss / len(loader.dataset)
        f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        try:    auc = roc_auc_score(all_labels, all_preds)
        except: auc = float("nan")
        return avg_loss, f1, auc

    # ── Phase 2a: freeze first 2 Transformer layers ───────────────────────────
    print(f"\n  Phase 2a: freeze lower 2 Transformer layers ({PHASE2_EPOCHS_A} epochs, lr={PHASE2_LR_A})")
    for param in list(model.transformer.layers[:2].parameters()):
        param.requires_grad = False

    optim_a   = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE2_LR_A, weight_decay=1e-4)
    sched_a   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim_a, T_max=PHASE2_EPOCHS_A, eta_min=1e-5)

    for epoch in range(1, PHASE2_EPOCHS_A + 1):
        tr_loss, _, _    = _run_epoch(train_loader, True, optim_a)
        val_loss, vf1, vauc = _run_epoch(val_loader, False)
        sched_a.step()
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(vf1)
        history["val_auc"].append(vauc)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [2a] Ep {epoch:>3}/{PHASE2_EPOCHS_A}  "
                  f"tr={tr_loss:.4f}  val={val_loss:.4f}  "
                  f"F1={vf1:.4f}  AUC={vauc:.4f}")

    # ── Phase 2b: unfreeze all ────────────────────────────────────────────────
    print(f"\n  Phase 2b: unfreeze all ({PHASE2_EPOCHS_B} epochs, lr={PHASE2_LR_B})")
    for param in model.parameters():
        param.requires_grad = True

    optim_b = torch.optim.AdamW(model.parameters(), lr=PHASE2_LR_B, weight_decay=1e-4)
    sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim_b, T_max=PHASE2_EPOCHS_B, eta_min=1e-6)

    for epoch in range(1, PHASE2_EPOCHS_B + 1):
        tr_loss, _, _       = _run_epoch(train_loader, True, optim_b)
        val_loss, vf1, vauc = _run_epoch(val_loader, False)
        sched_b.step()
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(vf1)
        history["val_auc"].append(vauc)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [2b] Ep {epoch:>3}/{PHASE2_EPOCHS_B}  "
                  f"tr={tr_loss:.4f}  val={val_loss:.4f}  "
                  f"F1={vf1:.4f}  AUC={vauc:.4f}")

        # Best checkpoint
        if vf1 > best_f1:
            best_f1    = vf1
            no_improve = 0
            torch.save(model.state_dict(), out_dir / "wcn_finetuned.pt")
        else:
            no_improve += 1
            if no_improve >= PHASE2_PATIENCE:
                print(f"  Early stopping at epoch {PHASE2_EPOCHS_A + epoch} "
                      f"(no improvement for {PHASE2_PATIENCE} epochs)")
                break

    # Reload best checkpoint
    model.load_state_dict(torch.load(out_dir / "wcn_finetuned.pt", map_location=device))
    print(f"\n  Best val F1: {best_f1:.4f}  → {out_dir / 'wcn_finetuned.pt'}")

    # Plot training curves
    _plot_training_curves(history, out_dir / "phase2_curves.png")

    return {"phase2_best_val_f1": best_f1,
            "phase2_epochs_run":  len(history["val_f1"])}


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — PSEUDO-LABEL REFINEMENT
# ══════════════════════════════════════════════════════════════════════════════

def phase3_refine(
        model: WCNv9,
        wf_norm: np.ndarray,
        scalars_norm: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        aux_ec: np.ndarray,
        aux_dp: np.ndarray,
        out_dir: Path,
        device: torch.device,
) -> dict:
    """Iterative pseudo-label refinement on uncertain points."""
    print("\n" + "=" * 60)
    print("PHASE 3 — Pseudo-label Refinement")
    print("=" * 60)

    criterion = _FocalLoss().to(device)
    pseudo_labels  = labels.copy()
    pseudo_weights = weights.copy()

    metrics = {}
    for rnd in range(1, PHASE3_ROUNDS + 1):
        print(f"\n  Round {rnd}/{PHASE3_ROUNDS}")

        # Run inference on all uncertain points
        uncertain_idx = np.where(pseudo_labels == 2)[0]
        print(f"    Uncertain points: {len(uncertain_idx):,}")

        if len(uncertain_idx) == 0:
            print("    No uncertain points — stopping.")
            break

        probas = _run_inference_batch(model, wf_norm, scalars_norm,
                                      uncertain_idx, device)

        # Assign pseudo-labels
        new_water = uncertain_idx[probas >= PHASE3_PROBA_HI]
        new_land  = uncertain_idx[probas <= PHASE3_PROBA_LO]
        pseudo_labels[new_water]  = 1
        pseudo_labels[new_land]   = 0
        pseudo_weights[new_water] = probas[probas >= PHASE3_PROBA_HI]
        pseudo_weights[new_land]  = 1.0 - probas[probas <= PHASE3_PROBA_LO]
        print(f"    New pseudo-labels → water: {len(new_water):,}  "
              f"land: {len(new_land):,}  "
              f"still uncertain: {int((pseudo_labels==2).sum()):,}")

        # Fine-tune for PHASE3_EPOCHS epochs
        lab_idx = np.where((pseudo_labels == 0) | (pseudo_labels == 1))[0]
        np.random.shuffle(lab_idx)
        n_val     = max(1, int(len(lab_idx) * VAL_FRACTION))
        val_idx   = lab_idx[:n_val]
        train_idx = lab_idx[n_val:]

        ds_tr = _LabeledDataset(wf_norm, scalars_norm, pseudo_labels,
                                pseudo_weights, aux_ec, aux_dp, train_idx)
        ds_va = _LabeledDataset(wf_norm, scalars_norm, pseudo_labels,
                                pseudo_weights, aux_ec, aux_dp, val_idx)
        loader_tr = DataLoader(ds_tr, batch_size=PHASE3_BATCH, shuffle=True,
                               num_workers=2, pin_memory=(device.type == "cuda"))
        loader_va = DataLoader(ds_va, batch_size=PHASE3_BATCH, shuffle=False,
                               num_workers=2, pin_memory=(device.type == "cuda"))

        optim = torch.optim.AdamW(model.parameters(), lr=PHASE3_LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=PHASE3_EPOCHS, eta_min=1e-6)
        best_f1_rnd = 0.0

        for ep in range(1, PHASE3_EPOCHS + 1):
            model.train()
            tr_loss = 0.0
            for batch in loader_tr:
                wf_b, sc_b, lb_b, wt_b, ec_b, dp_b = batch
                wf_b = wf_b.to(device, non_blocking=True)
                sc_b = sc_b.to(device, non_blocking=True)
                lb_b = lb_b.to(device, non_blocking=True)
                wt_b = wt_b.float().to(device, non_blocking=True)
                logits, wf_emb = model(wf_b, sc_b)
                loss = criterion(logits, lb_b, wt_b)
                optim.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                tr_loss += loss.item() * len(lb_b)
            sched.step()

            # Validate
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for batch in loader_va:
                    wf_b, sc_b, lb_b, *_ = batch
                    lg, _ = model(wf_b.to(device), sc_b.to(device))
                    preds.extend((torch.softmax(lg, 1)[:, 1].cpu().numpy() >= 0.5).tolist())
                    trues.extend(lb_b.numpy().tolist())
            vf1 = f1_score(trues, preds, average="macro", zero_division=0)

            if vf1 > best_f1_rnd:
                best_f1_rnd = vf1
                torch.save(model.state_dict(), out_dir / "wcn_refined.pt")

            if ep % 10 == 0 or ep == 1:
                print(f"    [R{rnd}] Ep {ep:>2}/{PHASE3_EPOCHS}  "
                      f"tr_loss={tr_loss/len(ds_tr):.4f}  val_F1={vf1:.4f}")

        model.load_state_dict(
            torch.load(out_dir / "wcn_refined.pt", map_location=device))
        metrics[f"round{rnd}_best_f1"] = best_f1_rnd
        print(f"    Round {rnd} best val F1: {best_f1_rnd:.4f}")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# XGBOOST
# ══════════════════════════════════════════════════════════════════════════════

def train_xgb(
        scalars_norm: np.ndarray,
        labels: np.ndarray,
        y_coord: np.ndarray,
        out_dir: Path,
) -> tuple:
    """Train XGBoost on 11 generalizable scalar features with spatial CV."""
    print("\n" + "=" * 60)
    print("XGBoost — 11 generalizable scalar features")
    print("=" * 60)

    lab_mask = (labels == 0) | (labels == 1)
    X  = scalars_norm[lab_mask]
    y  = (labels[lab_mask] == 1).astype(np.int32)
    yc = y_coord[lab_mask]

    n_w = int(y.sum()); n_l = int((y == 0).sum())
    spw = round(n_l / max(n_w, 1), 3)
    print(f"  water={n_w:,}  land={n_l:,}  scale_pos_weight={spw}")

    cv_f1, cv_auc = [], []
    print("\n  5-fold spatial CV:")
    for fold, (tr_i, va_i) in enumerate(_spatial_cv_split(yc)):
        if len(va_i) == 0 or len(np.unique(y[va_i])) < 2:
            continue
        clf = xgb.XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0,
            device="cuda" if DEVICE.type == "cuda" else "cpu",
        )
        clf.fit(X[tr_i], y[tr_i],
                eval_set=[(X[va_i], y[va_i])], verbose=False)
        pr = clf.predict_proba(X[va_i])[:, 1]
        f1 = f1_score(y[va_i], (pr >= 0.5).astype(int),
                      average="macro", zero_division=0)
        try:    auc = roc_auc_score(y[va_i], pr)
        except: auc = float("nan")
        cv_f1.append(f1); cv_auc.append(auc)
        print(f"    Fold {fold+1}: F1={f1:.3f}  AUC={auc:.3f}")

    print(f"\n  CV macro-F1 : {np.mean(cv_f1):.3f} ± {np.std(cv_f1):.3f}")
    print(f"  CV ROC-AUC  : {np.nanmean(cv_auc):.3f}")

    # Final model on all labeled data
    print(f"\n  Training final XGBoost on {len(X):,} rows …")
    final = xgb.XGBClassifier(
        n_estimators=600, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0,
        device="cuda" if DEVICE.type == "cuda" else "cpu",
    )
    final.fit(X, y, verbose=False)
    mpath = out_dir / "wcn_xgb.json"
    final.save_model(str(mpath))
    print(f"  Saved → {mpath}")

    xgb_proba = final.predict_proba(scalars_norm)[:, 1]

    # Feature importance plot
    imp = (pd.DataFrame({"feature": SCALAR_FEATURES,
                         "importance": final.feature_importances_})
             .sort_values("importance", ascending=False))
    print("\n  Feature importance:")
    print(imp.to_string(index=False))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp["feature"].iloc[::-1], imp["importance"].iloc[::-1], color="#E91E63")
    ax.set_xlabel("XGBoost gain importance")
    ax.set_title("WCN v9 XGBoost — 11 generalizable features")
    plt.tight_layout()
    plt.savefig(out_dir / "xgb_importance.png", dpi=150); plt.close()

    cv_res = {
        "xgb_cv_f1_mean": float(np.mean(cv_f1)),
        "xgb_cv_f1_std":  float(np.std(cv_f1)),
        "xgb_cv_auc_mean": float(np.nanmean(cv_auc)),
    }
    return final, xgb_proba, cv_res


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def _run_inference_batch(
        model: WCNv9,
        wf_norm: np.ndarray,
        scalars_norm: np.ndarray,
        indices: np.ndarray,
        device: torch.device,
        batch_size: int = 2048,
) -> np.ndarray:
    """Run inference on a subset of points. Returns proba_water (N,) float32."""
    model.eval()
    probas = np.zeros(len(indices), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(indices), batch_size):
            e   = min(s + batch_size, len(indices))
            idx = indices[s:e]
            amp = wf_norm[idx]                                    # (b, 200)
            wf2 = np.stack([amp, (amp > 0).astype(np.float32)],
                           axis=1)                                # (b, 2, 200)
            wf_t  = torch.from_numpy(wf2).to(device)
            sc_t  = torch.from_numpy(scalars_norm[idx]).to(device)
            lg, _ = model(wf_t, sc_t)
            probas[s:e] = torch.softmax(lg, dim=1)[:, 1].cpu().numpy()
    return probas


def run_full_inference(
        model: WCNv9,
        wf_norm: np.ndarray,
        scalars_norm: np.ndarray,
        device: torch.device,
) -> np.ndarray:
    """Run inference on all 234k points. Returns proba_water (N,) float32."""
    N = len(wf_norm)
    print(f"\n  Running WCN inference on {N:,} points …")
    all_idx = np.arange(N)
    proba   = _run_inference_batch(model, wf_norm, scalars_norm,
                                   all_idx, device, batch_size=2048)
    print(f"  Done.  water fraction: {(proba >= 0.5).mean()*100:.1f}%")
    return proba


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT AND PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def _plot_training_curves(history: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="train", color="#2196F3")
    axes[0].plot(epochs, history["val_loss"],   label="val",   color="#FF5722")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Focal Loss")
    axes[0].legend(); axes[0].set_title("Phase 2 Loss")
    axes[1].plot(epochs, history["val_f1"],  label="val F1",  color="#4CAF50")
    axes[1].plot(epochs, history["val_auc"], label="val AUC", color="#9C27B0")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
    axes[1].legend(); axes[1].set_title("Phase 2 Val Metrics")
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def export_results(
        feat_df: pd.DataFrame,
        wcn_proba: np.ndarray,
        xgb_proba: np.ndarray,
        orig_labels: np.ndarray,
        out_dir: Path,
        reconstructed_label: np.ndarray | None = None,
) -> None:
    """Save labeled_pointcloud_wcn.csv and topdown scatter plot."""
    N = len(feat_df)
    wcn_pred = (wcn_proba >= 0.5).astype(np.int8)
    xgb_pred = (xgb_proba >= 0.5).astype(np.int8)

    # Ensemble: agree → use prediction, disagree → uncertain (2)
    agree    = wcn_pred == xgb_pred
    ensemble = wcn_pred.copy()
    ensemble[~agree] = 2

    print(f"\n  WCN:      water={int(wcn_pred.sum()):,}  land={int((wcn_pred==0).sum()):,}")
    print(f"  XGBoost:  water={int(xgb_pred.sum()):,}  land={int((xgb_pred==0).sum()):,}")
    print(f"  Ensemble: water={int((ensemble==1).sum()):,}  "
          f"land={int((ensemble==0).sum()):,}  "
          f"uncertain={int((ensemble==2).sum()):,}")
    print(f"  Agreement: {100*agree.mean():.1f}%")

    out = pd.DataFrame({
        "x":             feat_df["x"].values,
        "y":             feat_df["y"].values,
        "z":             feat_df["z"].values,
        "v8_label":      orig_labels.astype(np.int8),
        "wcn_pred":      wcn_pred,
        "wcn_proba":     np.round(wcn_proba,  4),
        "xgb_pred":      xgb_pred,
        "xgb_proba":     np.round(xgb_proba,  4),
        "ensemble":      ensemble,
    })
    if reconstructed_label is not None:
        # 0=land 1=water 2=uncertain 3=recon-water (waterbed reconstruction)
        out["reconstructed_label"] = reconstructed_label
        n_recon = int((reconstructed_label == 3).sum())
        print(f"  Reconstructed water points (label=3): {n_recon:,}")
    # Append key diagnostic features for CloudCompare
    for col in ["energy_concentration", "n_gaps", "gap_ratio", "depth_proxy_m",
                "n_peaks", "energy_ratio_late"]:
        if col in feat_df.columns:
            out[col] = feat_df[col].values

    out.to_csv(OUT_CSV, index=False)
    print(f"\n  Saved {N:,} rows → {OUT_CSV}")

    x = feat_df["x"].values
    y = feat_df["y"].values
    z = feat_df["z"].values
    nc = z <= CANOPY_Z_MAX   # non-canopy mask

    cmap = {0: "saddlebrown", 1: "steelblue", 2: "gold"}
    lmap = {0: "Land", 1: "Water", 2: "Uncertain"}

    # Compute river boundary contours from wcn_proba
    print("\n  Computing river boundary contours for scatter …")
    grid_raw, x_min, y_min, n_x, n_y = rasterize(x, y, wcn_proba)
    grid_smooth = fill_and_smooth(grid_raw)
    contours = extract_contours(grid_smooth, x_min, y_min)

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    for ax, preds, title in [
        (axes[0], ensemble[nc],    f"WCN v9 Ensemble — no canopy (z ≤ {CANOPY_Z_MAX}m)"),
        (axes[1], orig_labels[nc], f"v9 Labels (input) — no canopy (z ≤ {CANOPY_Z_MAX}m)"),
    ]:
        xs, ys = x[nc], y[nc]
        for lv in [0, 1, 2]:
            m = preds == lv
            if m.any():
                ax.scatter(xs[m], ys[m], s=0.5, c=cmap[lv],
                           label=f"{lmap[lv]} ({m.sum():,})", rasterized=True)
        _draw_contours(ax, contours)
        ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title(title)
        handles, labels_leg = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_leg, handles))
        ax.legend(by_label.values(), by_label.keys(),
                  markerscale=10, loc="upper right", fontsize=7)
    plt.suptitle(
        f"River boundary — inner p={PROB_INNER} (white) · "
        f"center p={PROB_CENTER} (yellow) · outer p={PROB_OUTER} (orange)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "topdown_scatter.png", dpi=150); plt.close()
    print(f"  Plot → {out_dir / 'topdown_scatter.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def _load_labels_for_inference(N: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Load or bootstrap labels for export.

    Returns (ensemble_labels, reconstructed_label_or_None).
    Tries labeled_pointcloud_wcn.csv first, falls back to
    labeled_pointcloud_current.csv (bootstrapping stage 6 inline).
    """
    pc_cur = ROOT / "pointclouds" / "labeled_pointcloud_current.csv"
    if LABELS_PATH.exists():
        cols = ["ensemble"]
        lab_df = pd.read_csv(LABELS_PATH)
        assert len(lab_df) == N
        recon = (lab_df["reconstructed_label"].values.astype(np.int8)
                 if "reconstructed_label" in lab_df.columns else None)
        return lab_df["ensemble"].values.astype(np.int8), recon
    if pc_cur.exists():
        print(f"  labeled_pointcloud_wcn.csv not found — bootstrapping from {pc_cur.name}")
        df = pd.read_csv(pc_cur)
        assert len(df) == N
        recon = (df["reconstructed_label"].values.astype(np.int8)
                 if "reconstructed_label" in df.columns else None)
        return df["merged_label"].values.astype(np.int8), recon
    raise FileNotFoundError(
        "No label source found. Run water_surface_model.py first.")


def main():
    p = argparse.ArgumentParser(description="WCN v9 training / inference")
    p.add_argument("--no-train", action="store_true",
                   help="Skip all training — load wcn_refined.pt + wcn_xgb.json and run inference only")
    args = p.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    # ── Load data (always needed) ─────────────────────────────────────────────
    for p in [GRIDS_PATH, FEAT_PATH]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p}\nRun `python src/training/preprocess_wcn.py` first.")

    print("=" * 60)
    mode = "INFERENCE (--no-train)" if args.no_train else f"TRAINING  |  device={DEVICE}"
    print(f"WCN v9 {mode}")
    print("=" * 60)

    print(f"\nLoading waveform grids …")
    wf_norm = np.load(GRIDS_PATH, mmap_mode="r")   # (N, 200) float32
    N = len(wf_norm)
    print(f"  {N:,} × 200  ({GRIDS_PATH.stat().st_size/1e6:.0f} MB)")

    print(f"Loading features …")
    feat_df = pd.read_csv(FEAT_PATH)
    assert len(feat_df) == N, f"Row mismatch: feat={len(feat_df)}, wf={N}"
    scalars_raw = feat_df[SCALAR_FEATURES].values.astype(np.float32)
    print(f"  {N:,} × {N_SCALAR} features")

    if args.no_train:
        # ── Inference-only path ───────────────────────────────────────────────
        stats_path = MODEL_DIR / "wcn_stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"{stats_path} not found — run training first.")
        with open(stats_path) as fh:
            saved = json.load(fh)
        scalars_norm, _ = _normalise_scalars(scalars_raw,
                                             stats=saved["scalar_stats"])

        model = WCNv9(n_scalar=N_SCALAR).to(DEVICE)
        refined_path = MODEL_DIR / "wcn_refined.pt"
        model.load_state_dict(torch.load(refined_path, map_location=DEVICE,
                                         weights_only=True))
        model.eval()
        print(f"  Loaded WCNv9 from {refined_path}  (device={DEVICE})")

        xgb_m = xgb.XGBClassifier()
        xgb_path = MODEL_DIR / "wcn_xgb.json"
        xgb_m.load_model(xgb_path)
        print(f"  Loaded XGBoost from {xgb_path}")

        wcn_proba = run_full_inference(model, wf_norm, scalars_norm, DEVICE)
        xgb_proba = xgb_m.predict_proba(scalars_norm)[:, 1].astype(np.float32)

        labels, recon = _load_labels_for_inference(N)
        print(f"  Labels: land={int((labels==0).sum()):,}  "
              f"water={int((labels==1).sum()):,}  "
              f"uncertain={int((labels==2).sum()):,}")

        export_results(feat_df, wcn_proba, xgb_proba, labels, MODEL_DIR,
                       reconstructed_label=recon)

        print("\n" + "=" * 60)
        print("WCN v9 inference complete (--no-train).")
        print(f"  Output : {OUT_CSV}")
        print("=" * 60)
        return

    # ── Training path (default) ───────────────────────────────────────────────
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"{LABELS_PATH}\nRun stage 6 (bootstrap labels) first.")

    lab_df  = pd.read_csv(LABELS_PATH)
    assert len(lab_df) == N
    labels     = lab_df["ensemble"].values.astype(np.int8)
    conf       = ((lab_df["wcn_proba"].values + lab_df["xgb_proba"].values)
                  * 0.5).astype(np.float32)
    weights    = np.where(labels == 1, conf, 1.0).astype(np.float32)
    recon      = (lab_df["reconstructed_label"].values.astype(np.int8)
                  if "reconstructed_label" in lab_df.columns else None)
    print(f"  land={int((labels==0).sum()):,}  water={int((labels==1).sum()):,}  "
          f"uncertain={int((labels==2).sum()):,}")

    scalars_norm, scalar_stats = _normalise_scalars(scalars_raw)
    ec_idx = SCALAR_FEATURES.index("energy_concentration")
    dp_idx = SCALAR_FEATURES.index("depth_proxy_m")
    aux_ec = scalars_raw[:, ec_idx]
    aux_dp = scalars_raw[:, dp_idx]

    model = WCNv9(n_scalar=N_SCALAR).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nWCNv9 parameters: {n_params:,}")

    phase1_pretrain(model, wf_norm, MODEL_DIR, DEVICE)

    p2_metrics = phase2_finetune(
        model, wf_norm, scalars_norm, labels, weights,
        aux_ec, aux_dp, MODEL_DIR, DEVICE)

    xgb_model, xgb_proba, xgb_metrics = train_xgb(
        scalars_norm, labels, feat_df["y"].values, MODEL_DIR)

    p3_metrics = phase3_refine(
        model, wf_norm, scalars_norm, labels, weights,
        aux_ec, aux_dp, MODEL_DIR, DEVICE)

    refined_path = MODEL_DIR / "wcn_refined.pt"
    if refined_path.exists():
        model.load_state_dict(torch.load(refined_path, map_location=DEVICE))

    wcn_proba = run_full_inference(model, wf_norm, scalars_norm, DEVICE)
    export_results(feat_df, wcn_proba, xgb_proba, labels, MODEL_DIR,
                   reconstructed_label=recon)

    all_metrics = {
        "scalar_features": SCALAR_FEATURES,
        "n_scalar":        N_SCALAR,
        "scalar_stats":    scalar_stats,
        "n_params":        n_params,
        "device":          str(DEVICE),
        **p2_metrics,
        **xgb_metrics,
        **p3_metrics,
    }
    stats_path = MODEL_DIR / "wcn_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Stats + metrics → {stats_path}")

    print("\n" + "=" * 60)
    print("WCN v9 training complete.")
    print(f"  Models   : {MODEL_DIR}/")
    print(f"  Output   : {OUT_CSV}")
    print(f"  Best F1  : {p2_metrics.get('phase2_best_val_f1', '?'):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
