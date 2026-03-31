"""
train_deep.py — Train WaveformNet (1D CNN + MLP) on v2 labels.

Training strategy:
  - Spatial train/val split (bottom 80% of y-coord for training)
  - AdamW + CosineAnnealingLR
  - Focal loss with label smoothing (handles noisy pseudo-labels)
  - Best model saved by val macro-F1

Outputs:
  models/deep_model.pt         — model weights (state_dict)
  models/deep_model_stats.json — normalisation stats (needed at inference time)
  models/deep_training_curve.png
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from deep_model import WaveformNet, FocalLossWithLabelSmoothing, SPATIAL_COLS


# ── Inline dataset (avoids index-mapping complexity) ──────────────────────

class FlatDataset(Dataset):
    """Simple dataset that holds pre-loaded, pre-normalised arrays."""

    def __init__(self, grids: np.ndarray, spatial: np.ndarray, labels: np.ndarray):
        self.grids   = torch.from_numpy(grids).unsqueeze(1)    # (N, 1, 200)
        self.spatial = torch.from_numpy(spatial)                # (N, 5)
        self.labels  = torch.from_numpy(labels)                 # (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.grids[idx], self.spatial[idx], self.labels[idx]


def make_spatial_split(y_vals: np.ndarray, val_frac: float = 0.20):
    """Return (train_mask, val_mask) where val set = top val_frac by y."""
    cutoff   = np.percentile(y_vals, 100 * (1 - val_frac))
    val_mask = y_vals >= cutoff
    return ~val_mask, val_mask


def normalise(arr: np.ndarray, mean=None, std=None):
    if mean is None:
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0) + 1e-6
    return (arr - mean) / std, mean, std


def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    for wf, sp, lbl in loader:
        wf  = wf.to(device,  non_blocking=True)
        sp  = sp.to(device,  non_blocking=True)
        lbl = lbl.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=scaler.is_enabled()):
            loss = criterion(model(wf, sp), lbl)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(lbl)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_proba, all_labels = [], [], []
    for wf, sp, lbl in loader:
        wf  = wf.to(device,  non_blocking=True)
        sp  = sp.to(device,  non_blocking=True)
        lbl = lbl.to(device, non_blocking=True)
        logits = model(wf, sp)
        total_loss += criterion(logits, lbl).item() * len(lbl)
        proba = torch.softmax(logits, dim=1)[:, 1]
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_proba.append(proba.cpu().numpy())
        all_labels.append(lbl.cpu().numpy())
    preds  = np.concatenate(all_preds)
    proba  = np.concatenate(all_proba)
    labels = np.concatenate(all_labels)
    f1  = f1_score(labels, preds, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(labels, proba)
    except Exception:
        auc = float('nan')
    return total_loss / len(loader.dataset), f1, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features',   default='features.csv')
    ap.add_argument('--labels',     default='labels.csv')
    ap.add_argument('--grids',      default='waveform_grids.npy')
    ap.add_argument('--out-dir',    default='models')
    ap.add_argument('--epochs',     type=int,   default=60)
    ap.add_argument('--batch-size', type=int,   default=512)
    ap.add_argument('--lr',         type=float, default=1e-3)
    ap.add_argument('--patience',   type=int,   default=15)
    ap.add_argument('--val-frac',   type=float, default=0.20)
    ap.add_argument('--dropout',    type=float, default=0.30)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load full data arrays ────────────────────────────────────────────────
    print(f"Loading {args.features} …")
    feat_df = pd.read_csv(args.features)
    print(f"Loading {args.labels} …")
    lab_df  = pd.read_csv(args.labels)

    feat_df['label'] = lab_df['label'].values

    # Confident-only mask (row indices into 234k feat_df / waveform_grids.npy)
    conf_mask = feat_df['label'].isin([0, 1]).values
    conf_rows = np.where(conf_mask)[0]             # original row numbers
    print(f"Loading waveform grids from {args.grids} …")
    grids_all = np.load(args.grids, mmap_mode='r')   # (234024, 200)

    # Extract grids + spatial features + labels for confident rows
    grids_conf   = grids_all[conf_rows].copy().astype(np.float32)          # (N, 200)
    spatial_conf = feat_df[SPATIAL_COLS].values[conf_rows].astype(np.float32)  # (N, 5)
    labels_conf  = feat_df['label'].values[conf_rows].astype(np.int64)     # (N,)
    y_conf       = feat_df['y'].values[conf_rows]                          # for spatial split

    # Replace NaN/inf
    grids_conf   = np.nan_to_num(grids_conf,   nan=0.0, posinf=0.0, neginf=0.0)
    spatial_conf = np.nan_to_num(spatial_conf, nan=0.0, posinf=0.0, neginf=0.0)

    n_conf  = len(conf_rows)
    n_water = int((labels_conf == 1).sum())
    n_land  = n_conf - n_water
    print(f"Confident: {n_conf:,}  water={n_water:,}, land={n_land:,}")

    # ── Spatial train / val split ────────────────────────────────────────────
    trn_mask, val_mask = make_spatial_split(y_conf, val_frac=args.val_frac)
    print(f"Train: {trn_mask.sum():,}   Val: {val_mask.sum():,}")

    # ── Normalise ────────────────────────────────────────────────────────────
    # Waveform grids: single scalar mean/std (whole array)
    grid_tr   = grids_conf[trn_mask]
    g_mean    = float(grid_tr.mean())
    g_std     = float(grid_tr.std()) + 1e-6

    # Spatial: per-feature mean/std from training set
    sp_tr     = spatial_conf[trn_mask]
    sp_mean   = sp_tr.mean(axis=0)
    sp_std    = sp_tr.std(axis=0)  + 1e-6

    grids_norm   = (grids_conf   - g_mean)   / g_std
    spatial_norm = (spatial_conf - sp_mean)  / sp_std

    # ── Datasets and loaders ─────────────────────────────────────────────────
    n_workers = min(4, os.cpu_count() or 1)

    train_ds = FlatDataset(grids_norm[trn_mask],   spatial_norm[trn_mask],   labels_conf[trn_mask])
    val_ds   = FlatDataset(grids_norm[val_mask],   spatial_norm[val_mask],   labels_conf[val_mask])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=n_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=n_workers, pin_memory=True)

    # ── Model + loss + optimiser ─────────────────────────────────────────────
    n_pos_tr = int((labels_conf[trn_mask] == 1).sum())
    n_neg_tr = int(trn_mask.sum()) - n_pos_tr
    alpha    = round(n_neg_tr / (n_pos_tr + n_neg_tr), 4)   # ~ 0.61
    print(f"Focal loss alpha={alpha:.3f}  (water weight = {1-alpha:.3f})")

    model     = WaveformNet(n_spatial=len(SPATIAL_COLS), dropout=args.dropout).to(device)
    criterion = FocalLossWithLabelSmoothing(gamma=2.0, alpha=alpha,
                                            smoothing=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler    = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Training loop ────────────────────────────────────────────────────────
    best_f1      = 0.0
    patience_cnt = 0
    history      = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_auc': []}
    model_path   = os.path.join(args.out_dir, 'deep_model.pt')

    print(f"\n{'Epoch':>6}  {'TrLoss':>8}  {'VaLoss':>8}  {'F1':>7}  {'AUC':>7}  LR")
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        va_loss, va_f1, va_auc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['val_f1'].append(va_f1)
        history['val_auc'].append(va_auc)

        flag = ''
        if va_f1 > best_f1:
            best_f1      = va_f1
            patience_cnt = 0
            flag         = ' ← best'
            torch.save(model.state_dict(), model_path)
        else:
            patience_cnt += 1

        print(f"{epoch:>6}  {tr_loss:>8.4f}  {va_loss:>8.4f}  "
              f"{va_f1:>7.4f}  {va_auc:>7.4f}  {lr:.2e}{flag}")

        if patience_cnt >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
            break

    print(f"\nBest val macro-F1: {best_f1:.4f}")

    # ── Save normalisation stats ─────────────────────────────────────────────
    stats_path = os.path.join(args.out_dir, 'deep_model_stats.json')
    stats = {
        'grid_mean':    g_mean,
        'grid_std':     g_std,
        'spatial_mean': sp_mean.tolist(),
        'spatial_std':  sp_std.tolist(),
        'spatial_cols': SPATIAL_COLS,
        'best_val_f1':  float(best_f1),
    }
    with open(stats_path, 'w') as fh:
        json.dump(stats, fh, indent=2)
    print(f"Norm stats saved to {stats_path}")
    print(f"Model saved to {model_path}")

    # ── Training curve ───────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(history['train_loss']) + 1)
    ax1.plot(ep, history['train_loss'], label='train')
    ax1.plot(ep, history['val_loss'],   label='val')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Focal Loss'); ax1.legend()
    ax2.plot(ep, history['val_f1'],  label='macro-F1')
    ax2.plot(ep, history['val_auc'], label='ROC-AUC')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Score')
    ax2.set_title('Validation metrics'); ax2.legend()
    plt.tight_layout()
    curve_path = os.path.join(args.out_dir, 'deep_training_curve.png')
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"Training curve saved to {curve_path}")
    print("\nDone.")


if __name__ == '__main__':
    main()
