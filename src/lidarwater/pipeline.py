"""WaterPipeline — the public facade orchestrating stages over a PointCloud."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .artifacts import ArtifactResolver, LocalArtifactResolver
from .config import DEFAULT_CLASSIFY_STAGES, DEFAULT_FIT_STAGES, PipelineConfig, Stage
from .types import PipelineState, PointCloud

from ._stages import autolabel, boundary, canopy, features, surface, wcn


class WaterPipeline:
    """Runs the water/land/canopy classification pipeline over a PointCloud.

    ``classify()`` is the primary entry point: it runs inference only, using
    already-trained model artifacts resolved through ``artifacts``.
    ``fit()`` bootstraps and trains those artifacts from raw data — slower,
    stochastic, and intended for rebuilding the model on a new site, not for
    routine use.
    """

    def __init__(self, config: PipelineConfig | None = None,
                artifacts: ArtifactResolver | None = None):
        self.config = config or PipelineConfig()
        if artifacts is None:
            raise ValueError(
                "artifacts is required — pass a LocalArtifactResolver(root=Path(...)) "
                "pointing at a directory with trained model weights."
            )
        self.artifacts = artifacts

    @classmethod
    def from_local_models(cls, models_dir: str | Path, config: PipelineConfig | None = None) -> "WaterPipeline":
        return cls(config=config, artifacts=LocalArtifactResolver(root=Path(models_dir)))

    # ── inference ────────────────────────────────────────────────────────────

    def classify(self, cloud: PointCloud, stages: Sequence[Stage] | None = None) -> PipelineState:
        """Run the inference cascade (features -> WCN -> geometry -> canopy
        -> merge -> boundary) using existing trained artifacts."""
        stages = tuple(stages) if stages is not None else self.config.run.stages or DEFAULT_CLASSIFY_STAGES
        return self.run_stages(cloud, stages)

    def run_stages(self, cloud: PointCloud, stages: Sequence[Stage]) -> PipelineState:
        """Run an explicit subset of the inference-mode stages, in the fixed
        dependency order FEATURES -> WCN -> GEOMETRY -> CANOPY -> MERGE ->
        BOUNDARY (stages not requested are skipped)."""
        state = self._load_or_compute_features(cloud)
        requested = set(stages)
        device = self.config.run.device

        if Stage.WCN in requested:
            wcn.predict(state, self.config.wcn, self.artifacts, device=device)
        if Stage.GEOMETRY in requested:
            surface.run(state, self.config.surface, geometry_only=True)
        if Stage.CANOPY in requested:
            canopy.predict(state, self.config.canopy, self.artifacts)
        if Stage.MERGE in requested:
            canopy.merge(state)
        if Stage.BOUNDARY in requested:
            boundary.run(state, self.config.boundary)
        return state

    # ── training ─────────────────────────────────────────────────────────────

    def fit(self, cloud: PointCloud, stages: Sequence[Stage] | None = None) -> PipelineState:
        """Bootstrap and train every model artifact from raw data, then run
        the same inference cascade ``classify()`` uses.

        Stage chain: features -> autolabel (v6, z-band bootstrap) ->
        geometry (v6-anchored) -> WCN v9 (trained on that geometry's
        labels) -> geometry again (WCN-anchored, i.e. the "v10" pass) ->
        canopy -> merge -> boundary.

        Deviates from the original scripts by not retraining the
        intermediate v8 XGBoost/V8Net models (Phase 4): that model's only
        downstream use was to bootstrap WCN v9's training labels, and the
        v6-anchored geometry pass below produces an equivalent bootstrap
        directly. See MIGRATION.md.
        """
        requested = set(stages) if stages is not None else set(DEFAULT_FIT_STAGES)
        state = self._load_or_compute_features(cloud)
        device = self.config.run.device

        if Stage.AUTOLABEL in requested:
            autolabel.fit(state, self.config.zones, self.artifacts, device=device)
        if Stage.WCN in requested:
            surface.run(state, self.config.surface, geometry_only=False)   # v6-anchored bootstrap pass
            bootstrap_labels, bootstrap_confidence = _wcn_bootstrap(state)
            wcn.fit(state, self.config.wcn, self.artifacts,
                   bootstrap_labels, bootstrap_confidence, device=device)
        if Stage.GEOMETRY in requested:
            surface.run(state, self.config.surface, geometry_only=True)    # WCN-anchored ("v10") pass
        if Stage.CANOPY in requested:
            canopy.fit(state, self.config.canopy, self.artifacts)
        if Stage.MERGE in requested:
            canopy.merge(state)
        if Stage.BOUNDARY in requested:
            boundary.run(state, self.config.boundary)
        return state

    # ── caching ──────────────────────────────────────────────────────────────

    def _load_or_compute_features(self, cloud: PointCloud) -> PipelineState:
        cache_dir = self.config.run.cache_dir
        if cache_dir is not None:
            cached = _read_features_cache(cache_dir)
            if cached is not None:
                features_df, grids, grids_norm = cached
                return PipelineState(cloud=cloud, features=features_df,
                                     waveform_grids=grids, waveform_grids_norm=grids_norm)

        state = features.run(cloud, self.config.features)
        if cache_dir is not None:
            _write_features_cache(cache_dir, state)
        return state


def _wcn_bootstrap(state: PipelineState) -> tuple[np.ndarray, np.ndarray]:
    """WCN training labels/confidence from the v6-anchored geometry pass:
    reconstructed water (label 3) counts as water for training purposes."""
    assert (state.reconstructed_label is not None
            and state.autolabel_xgb_proba is not None
            and state.autolabel_deep_proba is not None)  # geometry + autolabel ran before this
    labels = state.reconstructed_label.copy()
    labels[labels == 3] = 1
    confidence = (state.autolabel_xgb_proba + state.autolabel_deep_proba) * 0.5
    return labels, confidence


_FEATURES_CACHE_NAME = "features.parquet"
_GRIDS_CACHE_NAME = "waveform_grids.npy"
_GRIDS_NORM_CACHE_NAME = "waveform_grids_norm.npy"


def _read_features_cache(cache_dir: Path):
    feat_path = cache_dir / _FEATURES_CACHE_NAME
    grids_path = cache_dir / _GRIDS_CACHE_NAME
    grids_norm_path = cache_dir / _GRIDS_NORM_CACHE_NAME
    if not (feat_path.exists() and grids_path.exists() and grids_norm_path.exists()):
        return None
    return pd.read_parquet(feat_path), np.load(grids_path), np.load(grids_norm_path)


def _write_features_cache(cache_dir: Path, state: PipelineState) -> None:
    assert (state.features is not None and state.waveform_grids is not None
            and state.waveform_grids_norm is not None)  # features stage just ran
    cache_dir.mkdir(parents=True, exist_ok=True)
    state.features.to_parquet(cache_dir / _FEATURES_CACHE_NAME)
    np.save(cache_dir / _GRIDS_CACHE_NAME, state.waveform_grids)
    np.save(cache_dir / _GRIDS_NORM_CACHE_NAME, state.waveform_grids_norm)
