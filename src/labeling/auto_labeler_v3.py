"""
auto_labeler_v3.py — Two-stage rule-based labeler.

This dataset is a NARROW RIVER VALLEY. The river channel (258–264m) sits 6–10m
below surrounding banks (264–268m) and vegetation canopy (268–278m).
height_above_local_min_10m is therefore large for ALL bank/meadow points and
cannot distinguish canopy from dry ground. Use z for labeling only.

NOTE: z is used here ONLY to generate training labels.
      Neither Stage 1 nor Stage 2 model will receive z as a feature.

Stage 1 labels (canopy vs ground):
  1  = canopy / vegetation  — z > 268 m  (above v2 Z_TRANSITION)
  0  = ground-level         — z ≤ 268 m
 -1  = uncertain            — (unused — all points get a label here)

Stage 2 labels (water vs dry ground, ground-level only):
  1  = water       — z < 261 m (riverbed) or multi-signal mid-zone
  0  = dry ground  — z 265–268 m, smooth or rough bank
 -1  = uncertain   — 261–265 m transition zone (model must figure this out)

Output: labels_v3.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


# ── Stage 1 z thresholds (matches v2 auto_labeler domain knowledge) ───────────
Z_CANOPY_LO  = 268.0   # z > this → label as canopy
# Everything else → ground (no uncertain zone for Stage 1)

# ── Stage 2 thresholds ─────────────────────────────────────────────────────────
# Water — riverbed (unambiguous, all pulses that reached this depth are through water)
Z_WATER_CLEAR  = 261.0  # z < this → definitely riverbed / below water line

# Water — mid-zone: require THREE independent signals
Z_WATER_MID_HI = 264.5  # upper z bound for mid-zone water labeling
S2_PLAN_W      = 0.55   # planarity (water surface is flat)
S2_EC_W        = 0.82   # energy_concentration (compact early pulse)
S2_ROUGH_W     = 0.013  # roughness upper bound (smooth surface)
S2_HPERC_W     = 0.15   # height_percentile_local (near local minimum)

# Dry ground — bank / meadow zone above river but below canopy
Z_DRY_LO       = 265.0  # z lower bound for dry labeling
Z_DRY_HI       = 268.0  # z upper bound (above this = canopy territory)
# Within dry zone: use waveform/geometry to confirm (avoid labeling wet banks)
S2_HR_DRY      = 0.6    # height_range_local < this → smooth/flat neighbourhood
S2_PLAN_DRY_LO = 0.22   # planarity > this → some flatness (not random scatter)
S2_ROUGH_DRY   = 0.024  # rough surface → gravel / meadow texture (alternative)


def label_points(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    def col(name, default=0.0):
        return df[name].values.astype(float) if name in df.columns \
               else np.full(n, default)

    z      = col('z')
    plan   = col('planarity')
    rough  = col('roughness')
    ec     = col('energy_concentration')
    hperc  = col('height_percentile_local')
    hr     = col('height_range_local')
    npeaks = col('n_peaks')

    # ── Stage 1 ────────────────────────────────────────────────────────────────
    s1      = np.where(z > Z_CANOPY_LO, 1, 0).astype(np.int8)
    s1_conf = np.where(z > Z_CANOPY_LO,
                       np.clip((z - Z_CANOPY_LO) / 4.0, 0.70, 0.99),
                       np.clip((Z_CANOPY_LO - z) / 4.0, 0.70, 0.99)
              ).astype(np.float32)

    # ── Stage 2 ────────────────────────────────────────────────────────────────
    s2      = np.full(n, -1, dtype=np.int8)
    s2_conf = np.zeros(n, dtype=np.float32)

    ground = (s1 == 0)

    # --- Water ---
    # W1: unambiguous riverbed
    w1 = ground & (z < Z_WATER_CLEAR)

    # W2: water surface mid-zone — three independent signals required
    w2 = (ground &
          (z >= Z_WATER_CLEAR) & (z < Z_WATER_MID_HI) &
          (plan  > S2_PLAN_W) &
          (ec    > S2_EC_W)   &
          (rough < S2_ROUGH_W) &
          (hperc < S2_HPERC_W))

    water = w1 | w2
    s2[water]      = 1
    s2_conf[w1]    = np.clip(0.93 + 0.06*(Z_WATER_CLEAR - z[w1])/3.0,
                              0.88, 0.99).astype(np.float32)
    s2_conf[w2 & ~w1] = 0.86

    # --- Dry ground ---
    # D1: clearly in bank/meadow z-band, smooth-ish
    d1 = (ground &
          (z >= Z_DRY_LO) & (z < Z_DRY_HI) &
          (hr   < S2_HR_DRY) &
          (plan > S2_PLAN_DRY_LO))

    # D2: same z-band, rough texture (gravel / grass)
    d2 = (ground &
          (z >= Z_DRY_LO) & (z < Z_DRY_HI) &
          (rough > S2_ROUGH_DRY))

    dry = (d1 | d2) & ~water
    s2[dry]     = 0
    s2_conf[d1 & ~water] = np.clip(
        0.88 + 0.08*(z[d1 & ~water] - Z_DRY_LO)/2.0,
        0.82, 0.97).astype(np.float32)
    s2_conf[d2 & ~d1 & ~water] = 0.83

    # ── summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STAGE 1 — Canopy vs Ground  (z threshold = {Z_CANOPY_LO}m)")
    print(f"{'='*60}")
    print(f"  Canopy  (1): {int((s1==1).sum()):>7,}  ({100*(s1==1).mean():.1f}%)")
    print(f"  Ground  (0): {int((s1==0).sum()):>7,}  ({100*(s1==0).mean():.1f}%)")
    print(f"  z range canopy  : {z[s1==1].min():.1f} – {z[s1==1].max():.1f} m")
    print(f"  z range ground  : {z[s1==0].min():.1f} – {z[s1==0].max():.1f} m")

    print(f"\n{'='*60}")
    print(f"STAGE 2 — Water vs Dry Ground  (ground-level points only)")
    print(f"{'='*60}")
    print(f"  Water      (1): {int((s2==1).sum()):>7,}  ({100*(s2==1).mean():.1f}%)")
    print(f"  Dry ground (0): {int((s2==0).sum()):>7,}  ({100*(s2==0).mean():.1f}%)")
    print(f"  Uncertain (-1): {int((s2==-1).sum()):>7,}  ({100*(s2==-1).mean():.1f}%)")
    print(f"\n  Water   z range: {z[s2==1].min():.2f} – {z[s2==1].max():.2f} m  "
          f"median={np.median(z[s2==1]):.2f}")
    print(f"  Dry     z range: {z[s2==0].min():.2f} – {z[s2==0].max():.2f} m  "
          f"median={np.median(z[s2==0]):.2f}")

    n_w_r1 = int(w1.sum()); n_w_r2 = int((w2 & ~w1).sum())
    n_d_r1 = int((d1 & ~water).sum()); n_d_r2 = int((d2 & ~d1 & ~water).sum())
    print(f"\n  Water   breakdown: riverbed rule={n_w_r1:,}  mid-zone={n_w_r2:,}")
    print(f"  Dry     breakdown: smooth-bank={n_d_r1:,}  rough-bank={n_d_r2:,}")

    unc_zone = (s2 == -1) & (z >= 261) & (z < 265)
    print(f"\n  Uncertain in 261–265 m transition zone: {int(unc_zone.sum()):,}  "
          f"← model must classify these")

    out = pd.DataFrame({
        'index':        np.arange(n),
        'stage1_label': s1,
        'stage1_conf':  np.round(s1_conf, 4),
        'stage2_label': s2,
        'stage2_conf':  np.round(s2_conf, 4),
        'x': df['x'].values if 'x' in df.columns else 0.0,
        'y': df['y'].values if 'y' in df.columns else 0.0,
        'z': z,
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default=str(ROOT / 'data_processed' / 'features_v2.csv'))
    ap.add_argument('--out',      default=str(ROOT / 'data_processed' / 'labels_v3.csv'))
    args = ap.parse_args()

    print(f"Loading {args.features} …")
    df = pd.read_csv(args.features)
    print(f"  {len(df):,} rows × {len(df.columns)} cols")

    out = label_points(df)
    out.to_csv(args.out, index=False)
    print(f"\nSaved {len(out):,} rows to {args.out}")
    print("Done.")


if __name__ == '__main__':
    main()
