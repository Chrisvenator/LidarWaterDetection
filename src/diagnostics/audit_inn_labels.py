"""Audit the Inn reference labels and score a run against them honestly.

    python src/diagnostics/audit_inn_labels.py runs/Inn_audit_base

Two label sets exist under data/Inn_DeepLearning_labels/:

* ``*.asc`` — region cuts exported from CloudCompare. They overlap each
  other, so summing file lengths double-counts; this script scores each
  cloud point once.
* ``water.txt`` / ``land.txt`` / ``proabably-land.txt`` — 65 picked points.

Balanced accuracy is reported with two bootstrap intervals: resampling
points (which assumes independent points and is therefore far too narrow
for spatially contiguous labels) and resampling 10 m plan-view blocks
(which respects the clustering). Writes ``audit_labels.json`` into the
workspace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent.parent
LABEL_DIR = ROOT / "data" / "Inn_DeepLearning_labels"

WATER_LABELS = (1, 3)
UNCERTAIN_LABEL = 2
MAX_MATCH_DIST_M = 0.05        # same tolerance as scripts/validate_labels.py
BLOCK_SIZE_M = 10.0            # spatial block for the cluster bootstrap
N_BOOTSTRAP = 2000
CI_PERCENTILES = (2.5, 97.5)
SEED = 0
LOW_CONFIDENCE = 0.8           # |2p - 1| below this = the model is not sure

WATER_REGIONS = ("water1", "water2", "water3", "deep_water")
LAND_REGIONS = ("land_and_canopy_1", "land_and_canopy_2", "dry_riverbed_(land)")
BLIND_FILES = {"water.txt": True, "land.txt": False, "proabably-land.txt": False}


def read_xyz(path: Path) -> np.ndarray:
    """First three numeric columns of a CloudCompare export."""
    lines = (ln.replace(",", " ").split() for ln in path.read_text().splitlines()
             if ln.strip() and not ln.startswith(("//", "#")))
    return np.array([parts[:3] for parts in lines], dtype=float)


def match_to_cloud(tree: cKDTree, xyz: np.ndarray) -> np.ndarray:
    dist, idx = tree.query(xyz, k=1)
    if (dist > MAX_MATCH_DIST_M).any():
        raise ValueError(f"{(dist > MAX_MATCH_DIST_M).sum()} reference points are not in the cloud")
    return idx


def load_regions(tree: cKDTree) -> dict[str, np.ndarray]:
    return {name: match_to_cloud(tree, read_xyz(LABEL_DIR / f"{name}.asc"))
            for name in WATER_REGIONS + LAND_REGIONS}


def load_blind(tree: cKDTree) -> pd.DataFrame:
    frames = []
    for name, is_water in BLIND_FILES.items():
        idx = match_to_cloud(tree, read_xyz(LABEL_DIR / name))
        frames.append(pd.DataFrame({"idx": idx, "is_water": is_water, "file": name}))
    return pd.concat(frames, ignore_index=True)


def unique_points(regions: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    """Cloud indices covered by the named regions, each counted once."""
    return np.unique(np.concatenate([regions[n] for n in names]))


def region_provenance(regions: dict[str, np.ndarray], n_cloud: int) -> dict:
    """File sizes, pairwise overlaps, and the de-duplicated totals."""
    names = list(regions)
    overlaps = {f"{a}&{b}": int(np.intersect1d(regions[a], regions[b]).size)
                for i, a in enumerate(names) for b in names[i + 1:]}
    water, land = unique_points(regions, WATER_REGIONS), unique_points(regions, LAND_REGIONS)
    return {
        "rows_per_file": {n: int(len(v)) for n, v in regions.items()},
        "rows_summed": int(sum(len(v) for v in regions.values())),
        "overlaps": {k: v for k, v in overlaps.items() if v},
        "unique_water": int(water.size),
        "unique_land": int(land.size),
        "unique_total": int(water.size + land.size),
        "water_land_conflicts": int(np.intersect1d(water, land).size),
        "share_of_cloud": float((water.size + land.size) / n_cloud),
    }


def region_reference(regions: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per labelled cloud point."""
    water, land = unique_points(regions, WATER_REGIONS), unique_points(regions, LAND_REGIONS)
    return pd.DataFrame({"idx": np.concatenate([water, land]),
                         "is_water": np.repeat([True, False], [water.size, land.size])})


def block_ids(xy: np.ndarray) -> np.ndarray:
    cells = np.floor(xy / BLOCK_SIZE_M).astype(np.int64)
    return np.unique(cells, axis=0, return_inverse=True)[1].ravel()


def balanced_accuracy(tallies: np.ndarray) -> np.ndarray:
    """tallies[..., 4] = water_n, water_correct, land_n, land_correct."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.5 * (tallies[..., 1] / tallies[..., 0] + tallies[..., 3] / tallies[..., 2])


def bootstrap_interval(is_water: np.ndarray, correct: np.ndarray,
                       groups: np.ndarray, rng: np.random.Generator) -> list[float]:
    """Percentile CI of balanced accuracy, resampling whole groups."""
    per_point = np.column_stack([is_water, is_water & correct, ~is_water, ~is_water & correct])
    n_groups = int(groups.max()) + 1
    per_group = np.zeros((n_groups, 4))
    np.add.at(per_group, groups, per_point)
    draws = rng.integers(0, n_groups, size=(N_BOOTSTRAP, n_groups))
    scores = balanced_accuracy(per_group[draws].sum(axis=1))
    return [float(v) for v in np.nanpercentile(scores, CI_PERCENTILES)]


def score(reference: pd.DataFrame, cloud: pd.DataFrame) -> dict:
    """Water vs not-water; an uncertain point counts as not-water, exactly as
    scripts/validate_labels.py scores it."""
    if reference.empty or reference.is_water.nunique() < 2:
        return {"n": int(len(reference)), "note": "one class only — balanced accuracy undefined"}
    rng = np.random.default_rng(SEED)
    is_water = reference.is_water.to_numpy()
    labels = cloud.final_label.to_numpy()[reference.idx]
    correct = np.isin(labels, WATER_LABELS) == is_water
    blocks = block_ids(cloud[["x", "y"]].to_numpy()[reference.idx])
    return {
        "n": int(len(reference)), "n_water": int(is_water.sum()), "n_land": int((~is_water).sum()),
        "n_blocks": int(blocks.max()) + 1,
        "water_recall": float(correct[is_water].mean()),
        "land_recall": float(correct[~is_water].mean()),
        "balanced_accuracy": float(0.5 * (correct[is_water].mean() + correct[~is_water].mean())),
        "ci95_point_bootstrap": bootstrap_interval(is_water, correct, np.arange(len(reference)), rng),
        "ci95_block_bootstrap": bootstrap_interval(is_water, correct, blocks, rng),
        "water_called_uncertain": int((labels[is_water] == UNCERTAIN_LABEL).sum()),
        "water_called_land_or_canopy": int((~correct[is_water]).sum()
                                           - (labels[is_water] == UNCERTAIN_LABEL).sum()),
        "land_called_uncertain": int((labels[~is_water] == UNCERTAIN_LABEL).sum()),
    }


def blind_selection_bias(blind: pd.DataFrame, cloud: pd.DataFrame,
                         region_idx: np.ndarray) -> dict:
    """Are the picked points easier than the cloud they stand for?"""
    confidence = np.abs(cloud.wcn_proba.to_numpy() - 0.5) * 2
    xy = cloud[["x", "y"]].to_numpy()
    out = {"cloud_share_confidence_below_0.8": float((confidence < LOW_CONFIDENCE).mean()),
           "blind_share_confidence_below_0.8": float((confidence[blind.idx] < LOW_CONFIDENCE).mean()),
           "blind_inside_regions": int(np.isin(blind.idx, region_idx).sum()),
           "blind_blocks_10m": int(np.unique(block_ids(xy[blind.idx])).size)}
    for name, part in blind.groupby("file"):
        pts = cloud.iloc[part.idx]
        out[name] = {
            "n": int(len(part)),
            "inside_regions": int(np.isin(part.idx, region_idx).sum()),
            "x_range": [float(pts.x.min()), float(pts.x.max())],
            "y_range": [float(pts.y.min()), float(pts.y.max())],
            "z_range": [float(pts.z.min()), float(pts.z.max())],
            "wcn_proba_range": [float(pts.wcn_proba.min()), float(pts.wcn_proba.max())],
            "reflectance_range": [float(pts.reflectance_dB.min()), float(pts.reflectance_dB.max())],
            "blocks_10m": int(np.unique(block_ids(xy[part.idx])).size),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workspace", type=Path)
    args = parser.parse_args()

    cloud = pd.read_csv(args.workspace / "pointclouds" / "labeled_pointcloud_final.csv")
    tree = cKDTree(cloud[["x", "y", "z"]].to_numpy())
    regions, blind = load_regions(tree), load_blind(tree)
    reference = region_reference(regions)
    in_regions = np.isin(blind.idx, reference.idx)

    report = {
        "workspace": str(args.workspace),
        "label_counts": {int(k): int(v) for k, v in cloud.final_label.value_counts().items()},
        "region_provenance": region_provenance(regions, len(cloud)),
        "blind_selection": blind_selection_bias(blind, cloud, reference.idx.to_numpy()),
        "scores": {
            "regions_unique": score(reference, cloud),
            "blind_65": score(blind, cloud),
            "blind_inside_regions": score(blind[in_regions], cloud),
            "blind_outside_regions": score(blind[~in_regions], cloud),
        },
    }
    out = args.workspace / "audit_labels.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
