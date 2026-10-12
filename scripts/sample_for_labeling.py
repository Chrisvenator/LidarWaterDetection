"""Export an ambiguity-stratified sample of points for hand labelling.

    python scripts/sample_for_labeling.py runs/Inn_DeepLearning --per-stratum 60

Uniform sampling wastes effort: most of a survey is unambiguous. This picks
points where the pipeline is least certain, so each label carries the most
information, and writes one file per stratum so they can be loaded and
split in CloudCompare.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WATER_LABELS = (1, 3)
SURFACE_BAND_M = 0.10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("workspace", type=Path)
    p.add_argument("--per-stratum", type=int, default=60)
    p.add_argument("--water-level", type=float, default=None,
                   help="default: read from the workspace's site_profile.json")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_strata(d: pd.DataFrame, water_level: float) -> dict[str, np.ndarray]:
    """Each stratum answers a question the current output cannot."""
    proba, other = d.wcn_proba.to_numpy(), d.wcn_xgb_proba.to_numpy()
    label, z = d.final_label.to_numpy(), d.z.to_numpy()
    below = z < water_level - 0.05
    return {
        # is submerged land real here, or is the model over-calling land?
        "below_water_called_land": below & (label == 0),
        # the model has no opinion
        "model_uncertain": np.abs(proba - 0.5) < 0.15,
        # the two heads contradict each other
        "heads_disagree": np.abs(proba - other) > 0.4,
        # the pipeline declined to decide
        "output_uncertain": label == 2,
        # confident calls, as a regression control
        "control_water": np.isin(label, WATER_LABELS) & (proba > 0.9),
        "control_land": (label == 0) & (proba < 0.1),
    }


def main() -> int:
    args = parse_args()
    d = pd.read_csv(args.workspace / "pointclouds" / "labeled_pointcloud_final.csv")
    water_level = args.water_level
    if water_level is None:
        import json
        water_level = json.loads(
            (args.workspace / "site_profile.json").read_text())["water_level_z"]

    out_dir = args.workspace / "to_label"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"water level {water_level:.2f} m; sampling up to {args.per_stratum} per stratum")
    for name, mask in build_strata(d, water_level).items():
        pool = np.flatnonzero(mask)
        if not len(pool):
            print(f"  {name:<26} empty")
            continue
        take = rng.choice(pool, size=min(args.per_stratum, len(pool)), replace=False)
        path = out_dir / f"{name}.txt"
        d.iloc[np.sort(take)][["x", "y", "z"]].to_csv(path, index=False, header=False)
        print(f"  {name:<26} {len(pool):>7,} available -> {len(take):>3} written  {path}")

    print(f"\nLabel each file by splitting it into water/land, then score with:\n"
          f"  python scripts/validate_labels.py {args.workspace} "
          f"--water <water files> --land <land files>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
