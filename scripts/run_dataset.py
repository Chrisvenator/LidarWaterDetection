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
from lidarwater.plots import write_all                                # noqa: E402

LABEL_NAMES = {0: "land", 1: "water", 2: "uncertain", 3: "water_under_canopy", 4: "canopy"}

# Shipped weights, trained on Pielach. Classification reads these by default;
# training never writes here — a --fit run writes into its own workspace so a
# retrain on one survey cannot overwrite another's models.
PRETRAINED_MODELS = ROOT / "models"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path, help="survey folder holding the point-cloud/waveform pair")
    p.add_argument("--name", help="workspace name (default: dataset folder name)")
    p.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    p.add_argument("--fit", action="store_true",
                   help="train this site's own models into the workspace instead of "
                        "applying the pretrained ones")
    p.add_argument("--models", type=Path,
                   help=f"model directory to classify with (default: {PRETRAINED_MODELS.name}/)")
    p.add_argument("--profile-only", action="store_true",
                   help="report the derived site profile and config, then exit")
    p.add_argument("--resolve-uncertain", action="store_true",
                   help="decide the uncertain class by the model's own probability "
                        "instead of emitting it (measured +1.3 balanced points on Inn)")
    p.add_argument("--surface-prior", action="store_true",
                   help="restore water the model rejected but the surface evidence backs "
                        "(deep water with no bed echo; +9 points there on Inn)")
    p.add_argument("--majority-filter", action="store_true",
                   help="flip points whose label contradicts their neighbours "
                        "(measured +0.6 balanced points on Inn)")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return p.parse_args()


def write_outputs(state, workspace: Workspace, profile) -> Path:
    out = workspace.pointclouds_dir / "labeled_pointcloud_final.csv"
    columns = {"x": state.cloud.x, "y": state.cloud.y, "z": state.cloud.z,
               "reflectance_dB": state.cloud.reflectance_db}
    for name in ("wcn_proba", "wcn_xgb_proba", "canopy_proba",
                 "reconstructed_label", "final_label"):
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
    if args.surface_prior:
        config = dataclasses.replace(config, cleanup=dataclasses.replace(
            config.cleanup, surface_prior=True))
    if args.majority_filter:
        config = dataclasses.replace(config, cleanup=dataclasses.replace(
            config.cleanup, majority_filter=True))
    if args.resolve_uncertain:
        config = dataclasses.replace(config, surface=dataclasses.replace(
            config.surface, resolve_uncertain=True))

    print(f"\nsite profile — {args.dataset.name}\n{profile.summary()}")
    if args.profile_only:
        return 0

    resolver = _resolver_for(args, workspace)
    print(f"\n{'training into' if args.fit else 'classifying with'} {resolver.root}")
    pipeline = WaterPipeline(config=config, artifacts=resolver)
    state = pipeline.fit(cloud) if args.fit else pipeline.classify(cloud)

    report_labels(state)
    print(f"\nwrote {write_outputs(state, workspace, profile)}")
    for figure in write_all(state, workspace.plot_dir, profile.water_level_z):
        print(f"      {figure}")
    return 0


def _resolver_for(args: argparse.Namespace, workspace: Workspace) -> LocalArtifactResolver:
    """Training writes into the workspace; classification reads the
    pretrained tree unless told otherwise."""
    if args.fit:
        return workspace.resolver()
    return LocalArtifactResolver(root=args.models or PRETRAINED_MODELS)


def _with_device(config, device: str):
    return dataclasses.replace(config, run=dataclasses.replace(config.run, device=device))


if __name__ == "__main__":
    raise SystemExit(main())
