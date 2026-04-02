"""
deep_model.py — PyTorch model classes for full-waveform LiDAR water/land classification.

Architecture: 1D CNN (waveform branch) + MLP (spatial branch) → fusion head.
Raw elevation (z) is intentionally excluded from the spatial branch so the model
learns from waveform shape and relative geometry rather than absolute altitude.

Spatial branch inputs (5 features):
    reflectance_dB, z_relative, planarity, roughness, height_range_local

Usage:
    from deep_model import LidarDataset, WaveformNet, FocalLossWithLabelSmoothing
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ── Dataset ────────────────────────────────────────────────────────────────

SPATIAL_COLS = [
    'reflectance_dB',
    'z_relative',
    'planarity',
    'roughness',
    'height_range_local',
]


class LidarDataset(Dataset):
    """
    Loads pre-computed waveform grids + spatial features.

    Args:
        grids_path   : path to waveform_grids.npy  — shape (N, 200)
        features_df  : DataFrame with SPATIAL_COLS and (optionally) 'label'
        indices      : subset of row indices to use (for train/val split)
        train_stats  : dict with 'grid_mean', 'grid_std', 'spatial_mean', 'spatial_std'
                       If None, computes from this subset (use for training set).
    """

    def __init__(self, grids_path: str, features_df: pd.DataFrame,
                 indices: np.ndarray, train_stats: dict = None):
        grids_all = np.load(grids_path, mmap_mode='r')

        self.grids   = grids_all[indices].copy().astype(np.float32)   # (M, 200)
        self.spatial = features_df[SPATIAL_COLS].values[indices].astype(np.float32)

        # Replace NaN/inf
        self.grids   = np.nan_to_num(self.grids,   nan=0.0, posinf=0.0, neginf=0.0)
        self.spatial = np.nan_to_num(self.spatial, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalisation
        if train_stats is None:
            self.stats = {
                'grid_mean':    float(self.grids.mean()),
                'grid_std':     float(self.grids.std())  + 1e-6,
                'spatial_mean': self.spatial.mean(axis=0),
                'spatial_std':  self.spatial.std(axis=0) + 1e-6,
            }
        else:
            self.stats = train_stats

        self.grids   = (self.grids   - self.stats['grid_mean'])    / self.stats['grid_std']
        self.spatial = (self.spatial - self.stats['spatial_mean']) / self.stats['spatial_std']

        # Labels (−1 if not present)
        if 'label' in features_df.columns:
            self.labels = features_df['label'].values[indices].astype(np.int64)
            self.has_labels = True
        else:
            self.labels = np.full(len(indices), -1, dtype=np.int64)
            self.has_labels = False

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, idx):
        grid    = torch.from_numpy(self.grids[idx]).unsqueeze(0)   # (1, 200)
        spatial = torch.from_numpy(self.spatial[idx])               # (5,)
        label   = torch.tensor(self.labels[idx], dtype=torch.long)
        return grid, spatial, label


# ── Model ──────────────────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, padding='same'):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class WaveformNet(nn.Module):
    """
    1D CNN + MLP multi-modal classifier.

    Waveform branch input : (B, 1, 200)
    Spatial branch input  : (B, 5)
    Output                : (B, 2)  — logits [land, water]
    """

    def __init__(self, n_spatial: int = 5, dropout: float = 0.3):
        super().__init__()

        # ── Waveform branch (1D CNN) ──────────────────────────────────────
        self.wf_branch = nn.Sequential(
            _ConvBlock(1,  32, kernel=3),
            _ConvBlock(32, 64, kernel=5),
            _ConvBlock(64, 64, kernel=11),
            nn.MaxPool1d(4),                            # (B, 64, 50)
            _ConvBlock(64, 128, kernel=5),
            nn.AdaptiveAvgPool1d(1),                    # (B, 128, 1)
        )
        wf_out = 128

        # ── Spatial branch (MLP) ─────────────────────────────────────────
        self.sp_branch = nn.Sequential(
            nn.Linear(n_spatial, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        sp_out = 32

        # ── Fusion head ───────────────────────────────────────────────────
        fused = wf_out + sp_out   # 160
        self.fusion = nn.Sequential(
            nn.Linear(fused, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.67),
            nn.Linear(64, 2),
        )

    def forward(self, waveform, spatial):
        """
        waveform : (B, 1, 200)
        spatial  : (B, 5)
        returns  : (B, 2) logits
        """
        wf_feat = self.wf_branch(waveform).squeeze(-1)   # (B, 128)
        sp_feat = self.sp_branch(spatial)                  # (B, 32)
        fused   = torch.cat([wf_feat, sp_feat], dim=1)    # (B, 160)
        return self.fusion(fused)


# ── Loss ───────────────────────────────────────────────────────────────────

class FocalLossWithLabelSmoothing(nn.Module):
    """
    Focal loss (Lin et al. 2017) with label smoothing for noisy pseudo-labels.

    Args:
        gamma      : focusing parameter (default 2.0)
        alpha      : weight for positive class (water); None = no reweighting
        smoothing  : label-smoothing epsilon (default 0.1)
        reduction  : 'mean' | 'sum' | 'none'
    """

    def __init__(self, gamma: float = 2.0, alpha: float = None,
                 smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_cls = logits.size(1)
        # Label smoothing: blend one-hot with uniform
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.smoothing / n_cls)
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / n_cls)

        log_probs = F.log_softmax(logits, dim=1)
        probs     = log_probs.exp()

        # Focal weight: (1 - p_t)^gamma  where p_t = prob of correct class
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t).pow(self.gamma)

        # Cross-entropy with smooth targets
        ce = -(smooth_targets * log_probs).sum(dim=1)   # (B,)
        loss = focal_weight * ce

        # Alpha weighting
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1,
                                  torch.full_like(p_t, self.alpha),
                                  torch.full_like(p_t, 1 - self.alpha))
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss
