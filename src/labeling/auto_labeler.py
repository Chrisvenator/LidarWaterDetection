"""
auto_labeler.py — Rule-based water/land labeler for full-waveform bathymetric LiDAR.

v2 — fixes vegetation/water confusion via:
  1. Elevation gate   : z > 264 m + multi-cluster → vegetation (LAND), not water
  2. Planarity gate   : planarity < 0.27 + multi-cluster → vegetation; >0.50 → flat surface
  3. Height-range gate: height_range_local > 0.65 m → vegetation canopy
  4. Riverbed recovery: z < 261 m + high planarity → riverbed seen through water (WATER)
  5. Reduced gap weight: gap features only boost confidence inside the river zone (z ≤ 264)

Usage:
  python auto_labeler.py                         # waveform+elevation rules only
  python auto_labeler.py --features features.csv # also uses planarity / height_range_local

Output: labels.csv
  index, label (1=water, 0=land, -1=uncertain), confidence, waveform features
"""

import argparse
import re
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm


# ── Elevation boundaries (metres, absolute) ────────────────────────────────
Z_RIVERBED_MAX  = 261.0   # below this = riverbed / deep channel
Z_RIVER_MAX     = 264.0   # 261–264 = active river surface / shallow zone
Z_TRANSITION    = 268.0   # 264–268 = ambiguous (low meadow / gravel bank)
                           # above 268 = dry land / vegetation

# ── Planarity thresholds ───────────────────────────────────────────────────
PLAN_FLAT       = 0.50    # >= this: flat surface (riverbed, water surface, meadow)
PLAN_VEG        = 0.27    # <= this: rough / multi-layer (vegetation)

# ── Height-range thresholds ────────────────────────────────────────────────
HR_FLAT         = 0.35    # <= this: smooth neighbourhood (water / flat ground)
HR_VEG          = 0.65    # >= this: tall canopy / rough (vegetation)


# ---------------------------------------------------------------------------
# Waveform parsing
# ---------------------------------------------------------------------------

def parse_array_string(s: str) -> np.ndarray:
    """Parse '[ 37  38  39 ...]' or '[37, 38, ...]' → int32 array."""
    nums = re.findall(r'[-+]?\d+', str(s))
    return np.array(nums, dtype=np.int32)


# ---------------------------------------------------------------------------
# Waveform feature extraction  (unchanged from v1)
# ---------------------------------------------------------------------------

def extract_features(times: np.ndarray, amps: np.ndarray,
                     min_peak_amp: int = 100, gap_threshold: int = 2) -> dict:
    f = {}
    f['max_amp']      = int(np.max(amps))
    f['mean_amp']     = float(np.mean(amps))
    f['total_energy'] = int(np.sum(amps))
    f['n_samples']    = len(times)
    f['time_span']    = int(times[-1] - times[0]) if len(times) > 1 else 0

    if len(times) > 1:
        diffs = np.diff(times.astype(np.int32))
        gaps  = diffs[diffs > gap_threshold]
        f['n_gaps']    = int(np.sum(diffs > gap_threshold))
        f['max_gap']   = int(np.max(diffs))
        f['mean_gap']  = float(np.mean(gaps)) if len(gaps) > 0 else 0.0
        f['total_gap'] = int(np.sum(gaps))
    else:
        f['n_gaps'] = f['max_gap'] = f['total_gap'] = 0
        f['mean_gap'] = 0.0

    peaks = [i for i in range(1, len(amps) - 1)
             if amps[i] > amps[i - 1] and amps[i] > amps[i + 1] and amps[i] >= min_peak_amp]
    f['n_peaks'] = len(peaks)

    if len(peaks) >= 2:
        pt  = times[peaks]
        spc = np.diff(pt.astype(np.int32))
        f['max_peak_spacing']  = int(np.max(spc))
        f['mean_peak_spacing'] = float(np.mean(spc))
        f['first_last_span']   = int(pt[-1] - pt[0])
    else:
        f['max_peak_spacing'] = f['mean_peak_spacing'] = f['first_last_span'] = 0

    f['n_clusters'] = 1 + f['n_gaps']

    if f['time_span'] > 0:
        cutoff = times[0] + 0.6 * f['time_span']
        e_late = int(np.sum(amps[times > cutoff]))
        f['energy_ratio_late'] = float(e_late / (f['total_energy'] + 1))
    else:
        f['energy_ratio_late'] = 0.0

    if peaks:
        f['first_peak_amp'] = int(amps[peaks[0]])
        f['last_peak_amp']  = int(amps[peaks[-1]])
    else:
        f['first_peak_amp'] = f['last_peak_amp'] = 0

    f['depth_proxy_m'] = round(f['max_peak_spacing'] * 0.05625, 3)
    return f


# ---------------------------------------------------------------------------
# SVB signature check  (unchanged from v1)
# ---------------------------------------------------------------------------

def is_svb_water_signature(times: np.ndarray, amps: np.ndarray,
                            gap_threshold: int = 50,
                            min_cluster_peak: int = 80) -> bool:
    if len(times) < 4:
        return False
    diffs = np.diff(times.astype(np.int32))
    lg    = np.where(diffs >= gap_threshold)[0]
    if len(lg) == 0:
        return False
    split    = int(lg[0]) + 1
    c1_amps  = amps[:split]
    c2_amps  = amps[split:]
    return (len(c1_amps) > 0 and len(c2_amps) > 0 and
            int(np.max(c1_amps)) >= min_cluster_peak and
            int(np.max(c2_amps)) >= min_cluster_peak)


# ---------------------------------------------------------------------------
# Confidence scoring  (v2 — geometry-aware)
# ---------------------------------------------------------------------------

def compute_water_confidence(wf: dict,
                              reflectance_dB: float,
                              z: float,
                              svb_flag: bool,
                              planarity: float = None,
                              height_range: float = None) -> float:
    """
    Compute P(water) in [0, 1].

    Priority order of evidence (high → low):
      1. Planarity + elevation combined   (overrides almost everything)
      2. Elevation zone                   (sets baseline prior)
      3. SVB signature (only in river zone)
      4. Waveform gaps (only in river zone, reduced weight)
      5. Reflectance and amplitude        (minor adjustments)
    """
    is_multi = wf['n_gaps'] >= 2 or wf['max_gap'] >= 30
    geo_avail = planarity is not None and height_range is not None

    # ── PHASE 1: Elevation-based baseline prior ───────────────────────────
    if z < Z_RIVERBED_MAX:
        score = 0.60   # below river surface → riverbed zone, lean water
    elif z <= Z_RIVER_MAX:
        score = 0.55   # river surface / shallow zone
    elif z <= Z_TRANSITION:
        score = 0.30   # transition zone: low-lying meadow, gravel banks
    else:
        score = 0.15   # high elevation: vegetation / dry land

    # ── PHASE 2: Planarity + height-range gates (geometry-first) ─────────
    if geo_avail:
        # ── 2a. Vegetation hard gate ─────────────────────────────────────
        # Low planarity + multi-cluster waveform at any elevation = vegetation
        if planarity < PLAN_VEG and is_multi:
            return float(np.clip(score - 0.30, 0.0, 1.0))

        # ── 2b. High elevation + any geometry → strong land ──────────────
        if z > Z_RIVER_MAX:
            if planarity < PLAN_FLAT:
                # Not a flat surface at high elevation = vegetation / rough land
                score -= 0.20
            if height_range > HR_VEG:
                # Tall canopy neighbourhood
                score -= 0.20
            elif height_range > 0.5:
                score -= 0.10

        # ── 2c. Flat surface anywhere → boost toward water ────────────────
        if planarity >= PLAN_FLAT:
            if z < Z_RIVERBED_MAX:
                # High-planarity surface BELOW river level = riverbed (WATER recovery)
                score += 0.30
            elif z <= Z_RIVER_MAX:
                score += 0.20   # flat surface in river zone
            else:
                score += 0.08   # flat but high elevation: meadow, also plausible

        elif planarity >= 0.38:
            score += 0.08       # moderately flat

        # ── 2d. Smooth neighbourhood (low height range) → water boost ────
        if height_range < HR_FLAT:
            score += 0.12
        elif height_range < 0.50:
            score += 0.04

    else:
        # No geometry available: apply a conservative elevation-only adjustment
        if z > Z_TRANSITION and is_multi:
            score -= 0.15   # penalise multi-cluster at high z without geometry

    # ── PHASE 3: SVB signature — valid only in the river zone ─────────────
    if svb_flag:
        if z <= Z_RIVER_MAX:
            score = max(score, 0.78)   # high confidence water
        elif z <= Z_TRANSITION:
            score = max(score, 0.55)   # ambiguous zone: moderate boost only
        # z > Z_TRANSITION: SVB at high elevation = vegetation; no boost

    # ── PHASE 4: Waveform gaps — supporting evidence, river zone only ─────
    if z <= Z_RIVER_MAX:
        # Gaps indicate multi-cluster returns, supportive of water
        if wf['max_gap'] >= 70:
            score += 0.10
        elif wf['max_gap'] >= 50:
            score += 0.07
        elif wf['max_gap'] >= 30:
            score += 0.03

        if wf['n_gaps'] >= 3:
            score += 0.05
        elif wf['n_gaps'] >= 2:
            score += 0.02

        if wf['max_peak_spacing'] >= 15:
            score += 0.08
        elif wf['max_peak_spacing'] >= 8:
            score += 0.04

    elif z <= Z_TRANSITION:
        # Transition zone: very small weight for waveform gaps
        if wf['max_gap'] >= 70:
            score += 0.04
        if wf['max_peak_spacing'] >= 15:
            score += 0.03

    # ── PHASE 5: Single clean echo — land indicator at high elevation ──────
    is_clean_single = (wf['n_peaks'] == 1 and wf['max_gap'] < 5)
    if is_clean_single:
        if z > Z_RIVER_MAX:
            score -= 0.20   # single return at high z = dry land
        elif z <= Z_RIVERBED_MAX and (geo_avail and planarity >= PLAN_FLAT):
            pass             # flat + low z + clean = riverbed, keep water score
        else:
            score -= 0.12   # modest penalty elsewhere

    # Zero gaps anywhere
    if wf['n_gaps'] == 0 and z > Z_RIVER_MAX:
        score -= 0.10

    # ── PHASE 6: Reflectance (minor) ───────────────────────────────────────
    if reflectance_dB < -25:
        score += 0.05
    elif reflectance_dB < -22:
        score += 0.02
    if reflectance_dB > -15:
        score -= 0.08
    elif reflectance_dB > -18:
        score -= 0.04

    # High amplitude at high elevation = hard diffuse surface (meadow / road)
    if wf['max_amp'] > 3500 and z > Z_RIVER_MAX:
        score -= 0.06

    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(pc_path: str, wf_path: str, feat_path: str | None,
        out_path: str, water_thresh: float, land_thresh: float) -> None:

    print(f"Loading point cloud from {pc_path} …")
    pc = pd.read_csv(pc_path)
    print(f"  {len(pc):,} points  |  columns: {list(pc.columns)}")

    print(f"Loading waveforms from {wf_path} …")
    wf_df = pd.read_csv(wf_path)
    print(f"  {len(wf_df):,} waveform rows")

    assert len(pc) == len(wf_df), \
        f"Row count mismatch: pc={len(pc)}, wf={len(wf_df)}"

    # Optional geometric features (planarity, height_range_local)
    geo_df = None
    if feat_path:
        print(f"Loading geometric features from {feat_path} …")
        geo_df = pd.read_csv(feat_path, usecols=lambda c: c in
                             ('planarity', 'height_range_local'))
        if 'planarity' not in geo_df.columns or 'height_range_local' not in geo_df.columns:
            print("  WARNING: planarity/height_range_local not found in features file — "
                  "running without geometric gates")
            geo_df = None
        else:
            print(f"  Loaded planarity + height_range_local for {len(geo_df):,} points")

    time_col = 'Time [SI]'
    amp_col  = 'Amplitude [ADC]'

    records    = []
    n_water    = n_land = n_uncertain = n_error = 0

    print(f"\nLabeling {len(pc):,} points …")
    for i in tqdm(range(len(pc)), unit='pt', ncols=80):
        row_pc = pc.iloc[i]
        row_wf = wf_df.iloc[i]

        # Geometric features (if available)
        plan  = float(geo_df.iloc[i]['planarity'])        if geo_df is not None else None
        h_rng = float(geo_df.iloc[i]['height_range_local']) if geo_df is not None else None

        try:
            times = parse_array_string(row_wf[time_col])
            amps  = parse_array_string(row_wf[amp_col])

            if len(times) == 0 or len(amps) == 0 or len(times) != len(amps):
                raise ValueError("empty or mismatched arrays")

            feat    = extract_features(times, amps)
            svb     = is_svb_water_signature(times, amps)
            refl_dB = float(row_pc['_riegl.reflectance'])
            z_val   = float(row_pc['z'])

            conf = compute_water_confidence(feat, refl_dB, z_val, svb, plan, h_rng)

            if conf >= water_thresh:
                label = 1;  n_water += 1
            elif conf <= land_thresh:
                label = 0;  n_land += 1
            else:
                label = -1; n_uncertain += 1

            rec = {'index': i, 'label': label, 'confidence': round(conf, 4),
                   'svb_flag': int(svb), 'reflectance_dB': refl_dB,
                   'z': z_val, 'x': float(row_pc['x']), 'y': float(row_pc['y']),
                   **feat}

        except Exception:
            n_error += 1
            rec = {'index': i, 'label': -1, 'confidence': 0.5,
                   'svb_flag': 0, 'reflectance_dB': 0.0, 'z': 0.0,
                   'x': 0.0, 'y': 0.0}

        records.append(rec)

    df_out = pd.DataFrame(records)
    df_out.to_csv(out_path, index=False)

    total     = len(pc)
    confident = n_water + n_land
    print(f"\n{'='*60}")
    print(f"Results saved to {out_path}")
    print(f"{'='*60}")
    print(f"  WATER    : {n_water:>7,}  ({100*n_water/total:5.1f}%)")
    print(f"  LAND     : {n_land:>7,}  ({100*n_land/total:5.1f}%)")
    print(f"  UNCERTAIN: {n_uncertain:>7,}  ({100*n_uncertain/total:5.1f}%)")
    if n_error:
        print(f"  ERRORS   : {n_error:>7,}  ({100*n_error/total:5.1f}%)")
    print(f"  CONFIDENT: {confident:>7,}  ({100*confident/total:5.1f}%)")
    if confident > 0:
        print(f"  Water fraction of confident: {100*n_water/confident:.1f}%")

    # Per-zone breakdown
    print(f"\n  By elevation zone:")
    for lab_str, lab_val in [('WATER', 1), ('LAND', 0), ('UNCERTAIN', -1)]:
        sub = df_out[df_out['label'] == lab_val]
        if len(sub) == 0:
            continue
        rb  = (sub['z'] < Z_RIVERBED_MAX).sum()
        rs  = ((sub['z'] >= Z_RIVERBED_MAX) & (sub['z'] <= Z_RIVER_MAX)).sum()
        tr  = ((sub['z'] > Z_RIVER_MAX)     & (sub['z'] <= Z_TRANSITION)).sum()
        hi  = (sub['z'] > Z_TRANSITION).sum()
        print(f"  {lab_str:9s}: z<261={rb:>6,}  261-264={rs:>6,}  264-268={tr:>6,}  >268={hi:>6,}")
    print(f"{'='*60}")


def main():
    ap = argparse.ArgumentParser(
        description='Rule-based water/land auto-labeler (v2, geometry-aware)')
    ap.add_argument('--pc',           default='data/point_cloud_df.txt')
    ap.add_argument('--wf',           default='data/waveform_df.txt')
    ap.add_argument('--features',     default=None,
                    help='Optional path to features.csv with planarity/height_range_local')
    ap.add_argument('--out',          default='labels.csv')
    ap.add_argument('--water-thresh', type=float, default=0.70)
    ap.add_argument('--land-thresh',  type=float, default=0.30)
    args = ap.parse_args()

    run(args.pc, args.wf, args.features, args.out,
        args.water_thresh, args.land_thresh)


if __name__ == '__main__':
    main()
