"""Per-dataset output isolation.

One survey, one directory: cached features, trained weights, exported point
clouds and plots all land under ``runs/<dataset>/`` instead of the
repository-level ``models/`` + ``pointclouds/`` + ``data_processed/`` trees
that belong to the Pielach study area.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from .artifacts import LocalArtifactResolver
from .config import PipelineConfig

DEFAULT_RUNS_DIR = Path("runs")


@dataclasses.dataclass(frozen=True)
class Workspace:
    """Output directory tree for a single dataset."""

    root: Path

    @classmethod
    def for_dataset(cls, name: str, runs_dir: Path | str = DEFAULT_RUNS_DIR) -> "Workspace":
        return cls(root=Path(runs_dir) / name)

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def pointclouds_dir(self) -> Path:
        return self.root / "pointclouds"

    @property
    def plot_dir(self) -> Path:
        return self.root / "plots"

    def mkdirs(self) -> "Workspace":
        for path in (self.cache_dir, self.models_dir, self.pointclouds_dir, self.plot_dir):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def resolver(self) -> LocalArtifactResolver:
        return LocalArtifactResolver(root=self.models_dir)

    def apply_to(self, config: PipelineConfig) -> PipelineConfig:
        """Point a config's feature cache and plot output at this workspace,
        leaving its stage selection and device untouched."""
        return dataclasses.replace(config, run=dataclasses.replace(
            config.run, cache_dir=self.cache_dir, plot_dir=self.plot_dir))

