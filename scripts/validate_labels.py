"""Score a classified point cloud against hand-labelled reference points.

    python scripts/validate_labels.py runs/Inn_DeepLearning \
        --water data/Inn_DeepLearning_labels/water.txt \
        --land  data/Inn_DeepLearning_labels/land.txt \
        --land  data/Inn_DeepLearning_labels/proabably-land.txt

Reference files hold ``x y z`` (or ``x,y,z``) as the first three columns,
one point per line, as exported from CloudCompare. A leading ``//`` header
and any extra columns are ignored. Each point is matched to its nearest
classified point.

A "land" reference may legitimately contain canopy: the test is water vs
not-water, so any non-water class counts as correct for a land reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

WATER_LABELS = (1, 3)          # water, water-under-canopy
MAX_MATCH_DIST_M = 0.05        # beyond this the reference point is not in the cloud


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("workspace", type=Path, help="runs/<dataset> directory to score")
    p.add_argument("--water", type=Path, action="append", default=[],
                   help="reference file of known-water points (repeatable)")
    p.add_argument("--land", type=Path, action="append", default=[],
                   help="reference file of known-land points (repeatable)")
    p.add_argument("--column", default="final_label",
                   help="label column to score (default: final_label)")
    return p.parse_args()


def _read_xyz(path: Path) -> np.ndarray:
    """First three numeric columns, whatever the delimiter and header."""
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("//", "#")):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) >= 3:
            rows.append([float(v) for v in parts[:3]])
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def load_reference(paths: list[Path]) -> np.ndarray:
    if not paths:
        return np.empty((0, 3))
    return np.vstack([_read_xyz(p) for p in paths])


def match(tree: cKDTree, points: np.ndarray) -> tuple[np.ndarray, int]:
    if not len(points):
        return np.array([], dtype=int), 0
    dist, idx = tree.query(points, k=1)
    keep = dist <= MAX_MATCH_DIST_M
    return idx[keep], int((~keep).sum())


def report(name: str, predicted_water: np.ndarray, expect_water: bool) -> float:
    if not len(predicted_water):
        print(f"  {name:<8} no reference points")
        return float("nan")
    correct = predicted_water if expect_water else ~predicted_water
    accuracy = float(correct.mean())
    print(f"  {name:<8} {correct.sum():>4}/{len(correct):<4} correct   {accuracy:6.1%}")
    return accuracy


def main() -> int:
    args = parse_args()
    csv = args.workspace / "pointclouds" / "labeled_pointcloud_final.csv"
    cloud = pd.read_csv(csv)
    tree = cKDTree(cloud[["x", "y", "z"]].to_numpy())
    labels = cloud[args.column].to_numpy()

    print(f"scoring {csv}  ({args.column})")
    accuracies = []
    for name, paths, expect_water in (("water", args.water, True), ("land", args.land, False)):
        idx, unmatched = match(tree, load_reference(paths))
        if unmatched:
            print(f"  {name}: {unmatched} reference points beyond {MAX_MATCH_DIST_M} m — skipped")
        accuracies.append(report(name, np.isin(labels[idx], WATER_LABELS), expect_water))

    valid = [a for a in accuracies if a == a]
    if valid:
        print(f"\n  balanced accuracy {np.mean(valid):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
