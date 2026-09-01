"""Run the pipeline on any survey folder, into its own output workspace.

    python scripts/run_dataset.py data/Inn_DeepLearning --profile-only
    python scripts/run_dataset.py data/Inn_DeepLearning --fit
    python scripts/run_dataset.py data/Inn_DeepLearning --models models/

Thresholds are rebased onto the dataset automatically (see
lidarwater.site.derive_site_config); everything written goes to
runs/<dataset>/ so the Pielach models/ and pointclouds/ trees stay untouched.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lidarwater import Workspace, WaterPipeline, derive_site_config   # noqa: E402
from lidarwater.artifacts import LocalArtifactResolver                # noqa: E402
from lidarwater.io import read_dataset_dir                            # noqa: E402

LABEL_NAMES = {0: "land", 1: "water", 2: "uncertain", 3: "water_under_canopy", 4: "canopy"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path, help="survey folder holding the point-cloud/waveform pair")
    p.add_argument("--name", help="workspace name (default: dataset folder name)")
    p.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    p.add_argument("--fit", action="store_true",
                   help="train this site's own models instead of applying existing ones")
    p.add_argument("--models", type=Path,
                   help="model directory to classify with (default: the workspace's own)")
    p.add_argument("--profile-only", action="store_true",
                   help="report the derived site profile and config, then exit")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return p.parse_args()


def write_outputs(state, workspace: Workspace, profile) -> Path:
    out = workspace.pointclouds_dir / "labeled_pointcloud_final.csv"
    columns = {"x": state.cloud.x, "y": state.cloud.y, "z": state.cloud.z,
               "reflectance_dB": state.cloud.reflectance_db}
    for name in ("wcn_proba", "canopy_proba", "reconstructed_label", "final_label"):
        value = getattr(state, name, None)
        if value is not None:
            columns[name] = value
    pd.DataFrame(columns).to_csv(out, index=False)

    (workspace.root / "site_profile.json").write_text(
        json.dumps(dataclasses.asdict(profile), indent=2))
    (workspace.root / "metrics.json").write_text(json.dumps(state.metrics, indent=2, default=float))
    return out


def report_labels(state) -> None:
    labels = state.final_label if state.final_label is not None else state.reconstructed_label
    if labels is None:
        return
    values, counts = np.unique(labels, return_counts=True)
    print("\nclass distribution")
    for value, count in zip(values, counts):
        print(f"  {LABEL_NAMES.get(int(value), value):<20} {count:>9,}  {count / len(labels):6.2%}")


def main() -> int:
    args = parse_args()
    workspace = Workspace.for_dataset(args.name or args.dataset.name, args.runs_dir).mkdirs()

    print(f"reading {args.dataset}")
    cloud = read_dataset_dir(args.dataset)
    config, profile = derive_site_config(cloud)
    config = workspace.apply_to(config)
    config = _with_device(config, args.device)

    print(f"\nsite profile — {args.dataset.name}\n{profile.summary()}")
    if args.profile_only:
        return 0

    resolver = LocalArtifactResolver(root=args.models) if args.models else workspace.resolver()
    pipeline = WaterPipeline(config=config, artifacts=resolver)
    state = pipeline.fit(cloud) if args.fit else pipeline.classify(cloud)

    report_labels(state)
    print(f"\nwrote {write_outputs(state, workspace, profile)}")
    return 0


def _with_device(config, device: str):
    return dataclasses.replace(config, run=dataclasses.replace(config.run, device=device))


if __name__ == "__main__":
    raise SystemExit(main())
