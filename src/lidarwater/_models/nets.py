"""Deep model architectures. Must match the shapes baked into the deployed
checkpoints (models/wcn_v9, models/labeling, models/current) — the CNN+MLP
fusion net (V6Net/V8Net/Stage2Net in the original scripts) is one identical
architecture reused across v6/v8/inference_pipeline, so it lives here once."""

from __future__ import annotations

import math

from ._torch_optional import require_torch

torch = require_torch()
nn = torch.nn
F = torch.nn.functional


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding="same", bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class WaveformCnnNet(nn.Module):
    """1D-CNN on the waveform grid fused with an MLP on spatial/scalar
    features. Identical architecture used as V6Net (waveform-only), V8Net
    (surface-model), and Stage2Net (two-stage cascade) in the original
    scripts — only the training data and n_spatial differ. No dropout: must
    match at inference."""

    def __init__(self, n_spatial: int):
        super().__init__()
        self.wf = nn.Sequential(
            _ConvBlock(1, 32, 3), _ConvBlock(32, 64, 5), _ConvBlock(64, 64, 11),
            nn.MaxPool1d(4), _ConvBlock(64, 128, 5), nn.AdaptiveAvgPool1d(1),
        )
        self.sp = nn.Sequential(
            nn.Linear(n_spatial, 128), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Linear(64, 32), nn.ReLU(True),
        )
        self.head = nn.Sequential(
            nn.Linear(160, 128), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Linear(64, 2),
        )

    def forward(self, wf, sp):
        return self.head(torch.cat([self.wf(wf).squeeze(-1), self.sp(sp)], dim=1))


def _sinusoidal_pe(n_pos: int, d_model: int):
    pe = torch.zeros(n_pos, d_model)
    pos = torch.arange(n_pos, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
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
    """WaveformContextNet v9 — transformer over waveform patches fused with
    an MLP over generalizable scalar features.

    Input
    -----
    wf     : (B, 2, 200)  channel 0 = per-sample-max-normalised amplitude,
                          channel 1 = binary activity mask (amp > 0).
    scalar : (B, n_scalar) z-score-normalised scalar features.

    Output
    ------
    logits : (B, 2)   raw class logits (land=0, water=1)
    wf_emb : (B, d_model)  CLS embedding (auxiliary heads, training only)
    """

    def __init__(self, n_scalar: int, d_model: int, n_heads: int, n_layers: int,
                 n_patches: int, seq_len: int):
        super().__init__()
        self.n_patches = n_patches
        self.d_model = d_model

        self.patch_embed = nn.Sequential(
            nn.Conv1d(2, d_model, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_buffer("pos_embed", _sinusoidal_pe(seq_len, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=512, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.wf_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())

        self.scalar_mlp = nn.Sequential(
            nn.Linear(n_scalar, 128), nn.BatchNorm1d(128), nn.GELU(),
            _ResBlock(128), _ResBlock(128),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Linear(d_model + 64, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 2),
        )

        # Auxiliary regression heads (training only, unused at inference)
        self.aux_energy_conc = nn.Linear(d_model, 1)
        self.aux_depth_proxy = nn.Linear(d_model, 1)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_waveform(self, wf):
        b = wf.size(0)
        x = self.patch_embed(wf).permute(0, 2, 1)          # (B, n_patches, d_model)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed.unsqueeze(0)
        x = self.transformer(x)
        return self.wf_proj(x[:, 0, :])

    def forward(self, wf, scalar):
        wf_emb = self.encode_waveform(wf)
        sp_emb = self.scalar_mlp(scalar)
        logits = self.head(torch.cat([wf_emb, sp_emb], dim=1))
        return logits, wf_emb


class MaskedWaveformAutoencoder(nn.Module):
    """Phase-1 self-supervised wrapper around WCNv9: masks waveform patches
    and reconstructs them, updating the shared patch_embed/transformer
    weights by reference. Discarded after pretraining."""

    def __init__(self, model: WCNv9, mask_ratio: float):
        super().__init__()
        self.model = model
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, 1, model.d_model))
        self.decoder = nn.Sequential(
            nn.LayerNorm(model.d_model),
            nn.Linear(model.d_model, 256), nn.GELU(),
            nn.Linear(256, 4),
        )
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, wf):
        b = wf.size(0)
        n = self.model.n_patches
        x = self.model.patch_embed(wf).permute(0, 2, 1)
        x = x + self.model.pos_embed[1:].unsqueeze(0)

        n_mask = max(1, int(self.mask_ratio * n))
        noise = torch.rand(b, n, device=wf.device)
        ids_sorted = noise.argsort(dim=1)
        is_masked = torch.zeros(b, n, dtype=torch.bool, device=wf.device)
        is_masked.scatter_(1, ids_sorted[:, :n_mask], True)

        mask_exp = self.mask_token.expand(b, n, -1)
        x = torch.where(is_masked.unsqueeze(-1), mask_exp, x)

        cls_pe = (self.model.cls_token + self.model.pos_embed[0]).expand(b, -1, -1)
        x = torch.cat([cls_pe, x], dim=1)
        patch_out = self.model.transformer(x)[:, 1:, :]

        recon = self.decoder(patch_out).reshape(b, 200)
        target = wf[:, 0, :]
        mask_bins = is_masked.unsqueeze(-1).expand(-1, -1, 4).reshape(b, 200)
        if not mask_bins.any():
            return recon.sum() * 0.0
        return F.mse_loss(recon[mask_bins], target[mask_bins])


class FocalLoss(nn.Module):
    """Focal loss with label smoothing and optional per-sample weighting."""

    def __init__(self, gamma: float, alpha: float | None, smooth: float):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, logits, targets, weights=None):
        n_cls = logits.size(1)
        lp = F.log_softmax(logits, dim=1)
        p = lp.exp()

        smooth_tgt = torch.full_like(logits, self.smooth / n_cls)
        smooth_tgt.scatter_(1, targets.unsqueeze(1), 1.0 - self.smooth + self.smooth / n_cls)
        ce = -(smooth_tgt * lp).sum(dim=1)

        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            cls_w = torch.where(targets == 1,
                                torch.full_like(pt, self.alpha),
                                torch.full_like(pt, 1.0 - self.alpha))
            loss = focal * cls_w * ce
        else:
            loss = focal * ce

        if weights is not None:
            loss = loss * weights
        return loss.mean()
