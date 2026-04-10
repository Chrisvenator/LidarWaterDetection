"""
preprocess_wcn.py — One-time data preparation for WCN v9 training.

Run before train_wcn_v9.py:
    python src/training/preprocess_wcn.py

Outputs
-------
data_processed/waveform_grids_norm.npy  — (234024, 200) float32, per-sample max normalised
data_processed/features_v9.csv          — 11 generalizable features + x, y, z
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent.parent
GRIDS_IN  = ROOT / "data_processed" / "waveform_grids.npy"
GRIDS_OUT = ROOT / "data_processed" / "waveform_grids_norm.npy"
FEAT_IN   = ROOT / "data_processed" / "features_current.csv"
FEAT_OUT  = ROOT / "data_processed" / "features_v9.csv"

# The 11 generalizable features used by WCN v9.
# All are dimensionless ratios, counts, or physical-unit (metres) values.
# None depend on point density or absolute elevation.
GENERALIZABLE_FEATURES = [
    # Dimensionless energy / amplitude ratios
    "energy_concentration",     # Σ(amp²_early) / Σ(amp²)        — KEY: water ≈0.924, land ≈0.684
    "max_amp_norm_by_energy",   # max_amp / √total_energy          — peak dominance
    "energy_ratio_late",        # Σ(amp²_late) / Σ(amp²)          — water=0.29 (bottom return), land=0.17
    "active_bins_ratio",        # n_nonzero / n_bins               — waveform fill fraction
    "peak_amp_ratio",           # first_peak_amp / last_peak_amp   — ratio, no units
    # Derived dimensionless ratios (computed below from raw features)
    "gap_ratio",                # total_gap / time_span            — fraction of span that is gaps
                                #   water=0.15, land=0.47 — strongest single discriminator
    "energy_center_norm",       # amplitude_weighted_center / n_samples — fractional energy centroid
                                #   water=0.47 (early), land=0.61 (later)
    # Count features (scanner sampling-rate independent)
    "n_peaks",                  # number of distinct amplitude peaks
    "n_gaps",                   # number of inter-cluster gaps     — water=0.44, land=2.40 (82% sep.)
    "n_clusters",               # number of signal clusters
    # Physical-unit feature (metres, universal)
    "depth_proxy_m",            # peak_separation_SI × 0.05625 m/SI (c_water = 225 000 km/s)
]

# Required source columns for the derived features
_REQUIRED_SOURCE = ["total_gap", "time_span", "amplitude_weighted_center", "n_samples"]


def normalise_waveforms() -> None:
    print(f"Loading waveform grids from {GRIDS_IN} …")
    grids = np.load(GRIDS_IN)                              # (N, 200) float32
    N, L  = grids.shape
    print(f"  shape={grids.shape}  dtype={grids.dtype}  "
          f"size={grids.nbytes / 1e6:.0f} MB")

    print("Per-sample max normalisation …")
    # Clip lower bound to 1 ADC unit to avoid division by zero on empty waveforms
    row_max   = grids.max(axis=1, keepdims=True).clip(min=1.0)   # (N, 1)
    grids_norm = (grids / row_max).astype(np.float32)

    # Sanity checks
    assert grids_norm.max() <= 1.0 + 1e-5, "normalisation overshoot"
    assert grids_norm.min() >= 0.0       , "negative values after normalisation"
    n_nonzero = np.count_nonzero(grids_norm)
    print(f"  Active bins: {n_nonzero:,} / {N*L:,}  "
          f"({100*n_nonzero/(N*L):.1f}%)  mean active per row: "
          f"{np.count_nonzero(grids_norm, axis=1).mean():.1f}")

    print(f"Saving {GRIDS_OUT} …")
    np.save(GRIDS_OUT, grids_norm)
    print(f"  Saved. Shape: {grids_norm.shape}")


def build_features_v9() -> None:
    print(f"\nLoading features from {FEAT_IN} …")
    feat = pd.read_csv(FEAT_IN)
    print(f"  {len(feat):,} rows × {len(feat.columns)} columns")

    # Verify required source columns exist
    missing_src = [c for c in _REQUIRED_SOURCE if c not in feat.columns]
    if missing_src:
        raise ValueError(f"Missing source columns for derived features: {missing_src}")

    # Compute derived dimensionless ratios
    feat["gap_ratio"] = (
        feat["total_gap"] / feat["time_span"].clip(lower=1.0)
    ).astype(np.float32)

    feat["energy_center_norm"] = (
        feat["amplitude_weighted_center"] / feat["n_samples"].clip(lower=1.0)
    ).astype(np.float32)

    # Verify all output features are now present
    missing_out = [f for f in GENERALIZABLE_FEATURES if f not in feat.columns]
    if missing_out:
        raise ValueError(f"Features still missing after derivation: {missing_out}")

    # Keep x, y, z for spatial cross-validation and point cloud export;
    # keep all 11 generalizable features
    out_cols = ["x", "y", "z"] + GENERALIZABLE_FEATURES
    feat_out = feat[out_cols].copy()

    # Replace any NaN / Inf with 0 (can arise from edge-case empty waveforms)
    n_bad = feat_out[GENERALIZABLE_FEATURES].isin([np.inf, -np.inf]).sum().sum()
    n_nan = feat_out[GENERALIZABLE_FEATURES].isna().sum().sum()
    if n_bad + n_nan > 0:
        print(f"  WARNING: {n_nan} NaN and {n_bad} Inf values — replacing with 0")
    feat_out[GENERALIZABLE_FEATURES] = (
        feat_out[GENERALIZABLE_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # Print statistics
    print("\n  Feature statistics (all 11 features):")
    print(f"  {'Feature':<28} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}")
    for col in GENERALIZABLE_FEATURES:
        s = feat_out[col]
        print(f"  {col:<28} {s.mean():>9.4f} {s.std():>9.4f} "
              f"{s.min():>9.4f} {s.max():>9.4f}")

    print(f"\nSaving {FEAT_OUT} …")
    feat_out.to_csv(FEAT_OUT, index=False)
    print(f"  Saved. Shape: {feat_out.shape}")


def main() -> None:
    print("=" * 60)
    print("WCN v9 — Data Preprocessing")
    print("=" * 60)
    normalise_waveforms()
    build_features_v9()
    print("\nPreprocessing complete.")
    print(f"  {GRIDS_OUT}")
    print(f"  {FEAT_OUT}")


if __name__ == "__main__":
    main()
