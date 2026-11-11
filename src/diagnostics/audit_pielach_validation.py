"""Audit the Pielach reference validation from the artefacts already on disk.

    python src/diagnostics/audit_pielach_validation.py

Needs no reference download: it re-reads ``validation_results/tables/`` and
joins the matched points back to the classified clouds. Answers three things:

1. What the point-level F1 is once abstentions (uncertain) are counted as
   misses instead of being dropped, with a spatial block-bootstrap interval.
2. Whether the surveyed-point check holds up statistically.
3. Which single threshold, if any, explains the false negatives at the
   channel margin.

Writes ``runs/Pielach/audit/pielach_validation_audit.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent.parent
TABLES = ROOT / "validation_results" / "tables"
REGISTRATION = ROOT / "validation_results" / "reports" / "registration.json"
FINAL_CSV = ROOT / "pointclouds" / "labeled_pointcloud_final.csv"
V10_CSV = ROOT / "pointclouds" / "labeled_pointcloud_v10.csv"
OUT_DIR = ROOT / "runs" / "Pielach" / "audit"

WATER_LABELS = (1, 3)
UNCERTAIN_LABEL = 2
CANOPY_LABEL = 4
LABEL_NAMES = {0: "land", 2: "uncertain", 4: "canopy"}
BLOCK_SIZE_M = 10.0
N_BOOTSTRAP = 2000
CI_PERCENTILES = (2.5, 97.5)
WILSON_Z = 1.96
SEED = 0
WATER_Z_MARGIN = 0.3           # Phase 3 headroom above the local surface
PROBA_CUT = 0.5                # ensemble guard on the geometry override
PROBA_UNCERTAIN_MAX = 0.65     # upper edge of the probabilities we abstain on
PROBA_SWEEP = (0.2, 0.3, 0.4)  # hypothetical lower guards, for the trade-off
TRANSECT_GAP_M = 4.0           # consecutive survey points further apart start a new transect
MARGIN_FEATURES = ("Z", "reflectance_dB", "deep_proba", "z_above_surface")


def load_matches() -> pd.DataFrame:
    """Per-point reference matches joined to our feature columns."""
    matches = pd.read_csv(TABLES / "per_point_matches.csv")
    offset = json.loads(REGISTRATION.read_text())
    final, v10 = pd.read_csv(FINAL_CSV), pd.read_csv(V10_CSV)
    local = matches[["utm_x", "utm_y", "z"]].to_numpy() - [offset["dx"], offset["dy"], 0.0]
    idx = cKDTree(final[["X", "Y", "Z"]].to_numpy()).query(local)[1]
    joined = pd.concat([matches, final.iloc[idx].reset_index(drop=True)], axis=1)
    v10_idx = cKDTree(v10[["X", "Y", "Z"]].to_numpy()).query(joined[["X", "Y", "Z"]].to_numpy())[1]
    joined["in_footprint"] = v10["in_footprint"].to_numpy()[v10_idx]
    return joined


def f1_from_tallies(tallies: np.ndarray) -> np.ndarray:
    """tallies[..., 3] = tp, fp, fn."""
    return 2 * tallies[..., 0] / (2 * tallies[..., 0] + tallies[..., 1] + tallies[..., 2])


def confusion(matches: pd.DataFrame) -> dict:
    """Water-vs-not-water confusion with a 10 m block-bootstrap F1 interval."""
    ours, ref = matches.our_water.to_numpy() == 1, matches.ref_water.to_numpy() == 1
    per_point = np.column_stack([ours & ref, ours & ~ref, ~ours & ref])
    cells = np.floor(matches[["utm_x", "utm_y"]].to_numpy() / BLOCK_SIZE_M).astype(np.int64)
    blocks = np.unique(cells, axis=0, return_inverse=True)[1].ravel()
    per_block = np.zeros((blocks.max() + 1, 3))
    np.add.at(per_block, blocks, per_point)
    draws = np.random.default_rng(SEED).integers(0, len(per_block), (N_BOOTSTRAP, len(per_block)))
    tp, fp, fn = (int(v) for v in per_point.sum(axis=0))
    return {
        "n": int(len(matches)), "n_blocks": int(len(per_block)),
        "tp": tp, "fp": fp, "fn": fn, "tn": int(len(matches) - tp - fp - fn),
        "precision": tp / (tp + fp), "recall": tp / (tp + fn),
        "f1": float(f1_from_tallies(np.array([tp, fp, fn]))),
        "f1_ci95_block_bootstrap": [float(v) for v in np.percentile(
            f1_from_tallies(per_block[draws].sum(axis=1)), CI_PERCENTILES)],
    }


def point_level(matches: pd.DataFrame) -> dict:
    """The published convention drops uncertain points; the honest one keeps them."""
    decided = matches[matches.our_label != UNCERTAIN_LABEL]
    return {
        "as_published_uncertain_dropped": confusion(decided),
        "abstention_counted_as_not_water": confusion(matches),
        "uncertain_dropped": {"n": int(len(matches) - len(decided)),
                              "ref_water": int(matches.ref_water[matches.our_label == UNCERTAIN_LABEL].sum())},
    }


def margin_diagnosis(matches: pd.DataFrame) -> dict:
    """Profile reference-water points we did not call water, per label we gave."""
    water = matches[matches.ref_water == 1]
    hit = water.our_water == 1
    tests = {
        "outside_footprint": water.in_footprint == 0,
        f"z_above_surface>{WATER_Z_MARGIN}": water.z_above_surface > WATER_Z_MARGIN,
        f"deep_proba<={PROBA_CUT}": water.deep_proba <= PROBA_CUT,
        f"deep_proba<={PROBA_UNCERTAIN_MAX}": water.deep_proba <= PROBA_UNCERTAIN_MAX,
    }
    groups = {"TP": hit, "FN_all": ~hit}
    groups.update({f"FN_{name}": ~hit & (water.our_label == label)
                   for label, name in LABEL_NAMES.items()})
    out = {}
    for name, mask in groups.items():
        out[name] = {"n": int(mask.sum()),
                     "median": {f: float(water.loc[mask, f].median()) for f in MARGIN_FEATURES},
                     "share": {k: float(v[mask].mean()) for k, v in tests.items()}}
    return out


def guard_tradeoff(matches: pd.DataFrame) -> dict:
    """What lowering the probability guard would gain and cost, among matched
    points geometry already accepts (inside footprint, under the surface margin)."""
    eligible = matches[(matches.our_water == 0) & (matches.our_label != CANOPY_LABEL)
                       & (matches.in_footprint == 1)
                       & (matches.z_above_surface <= WATER_Z_MARGIN)]
    return {str(cut): {"ref_water_gained": int(((eligible.deep_proba >= cut) & (eligible.ref_water == 1)).sum()),
                       "ref_land_lost": int(((eligible.deep_proba >= cut) & (eligible.ref_water == 0)).sum())}
            for cut in PROBA_SWEEP}


def wilson(k: int, n: int) -> list[float]:
    p, z2 = k / n, WILSON_Z ** 2
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = WILSON_Z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return [float(centre - half), float(centre + half)]


def survey_check() -> dict:
    """Surveyed points: per-file tallies, Wilson intervals, transect clustering."""
    survey = pd.read_csv(TABLES / "survey_matches.csv")
    matched = survey[survey.matched == 1].copy()
    matched["correct"] = matched.our_water == matched.expect_water
    out = {"total_per_file": survey.groupby("file").size().to_dict(),
           "matched_per_file": matched.groupby("file").size().to_dict()}
    for name, part in (("water_transects", matched[matched.expect_water == 1]),
                       ("land_control", matched[matched.expect_water == 0])):
        step = np.hypot(part.easting.diff(), part.northing.diff()).fillna(0.0)
        group = (step > TRANSECT_GAP_M).cumsum()
        out[name] = {
            "n": int(len(part)), "correct": int(part.correct.sum()),
            "wilson95": wilson(int(part.correct.sum()), len(part)),
            "spatial_groups": int(group.nunique()),
            "groups_with_a_miss": int(group[~part.correct].nunique()),
            "extent_m": [float(np.ptp(part.easting)), float(np.ptp(part.northing))],
            "miss_heights": sorted(part.height[~part.correct].round(2).tolist()),
            "hit_height_range": [float(part.height[part.correct].min()),
                                 float(part.height[part.correct].max())],
        }
    return out


def main() -> int:
    matches = load_matches()
    report = {"point_level": point_level(matches),
              "margin_diagnosis": margin_diagnosis(matches),
              "guard_tradeoff": guard_tradeoff(matches),
              "survey": survey_check()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "pielach_validation_audit.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
