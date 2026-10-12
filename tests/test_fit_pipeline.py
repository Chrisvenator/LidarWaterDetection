"""End-to-end smoke test of the training path.

``fit()`` was previously uncovered: a break in a late stage only surfaced
after the ~35 min WCN training ahead of it had already run. This exercises
the whole chain — autolabel, both geometry passes, WCN, canopy, merge,
boundary — on the synthetic cloud with epoch counts cut to the minimum, so
a structural break shows up in seconds.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from lidarwater import PipelineConfig, Stage, WaterPipeline
from lidarwater.artifacts import ArtifactId, LocalArtifactResolver
from lidarwater.config import WcnConfig, WcnTrainConfig

torch = pytest.importorskip("torch", reason="fit() needs the [deep] extra")

# Small but not degenerate: the second geometry pass anchors on the WCN's
# probabilities, so a net trained to nothing leaves it with no candidates.
_TINY_TRAINING = WcnTrainConfig(
    phase1_epochs=2, phase2_epochs_frozen=5, phase2_epochs_full=25,
    phase3_rounds=1, phase3_epochs=5, phase1_batch=256, phase2_batch=256,
    phase3_batch=256,
)

_TRAINED_ARTIFACTS = (
    ArtifactId.V6_XGB, ArtifactId.V6_DEEP, ArtifactId.V6_DEEP_STATS,
    ArtifactId.WCN_REFINED, ArtifactId.WCN_XGB, ArtifactId.WCN_STATS,
    ArtifactId.CANOPY_XGB,
)


@pytest.fixture(scope="module")
def fitted(synthetic_river_module, tmp_path_factory):
    cloud, _, _ = synthetic_river_module
    models = tmp_path_factory.mktemp("models")
    config = dataclasses.replace(
        PipelineConfig(), wcn=WcnConfig(train=_TINY_TRAINING))
    pipeline = WaterPipeline(config=config,
                             artifacts=LocalArtifactResolver(root=models))
    return pipeline.fit(cloud), models, cloud


def test_fit_runs_every_stage_end_to_end(fitted):
    state, _, cloud = fitted
    assert state.final_label is not None
    assert len(state.final_label) == len(cloud)
    assert set(np.unique(state.final_label)) <= {0, 1, 2, 3, 4}
    for stage in ("wcn", "geometry", "canopy", "merge", "boundary"):
        assert stage in state.metrics, f"{stage} stage produced no metrics"


def test_fit_writes_every_artifact_through_the_resolver(fitted):
    _, models, _ = fitted
    resolver = LocalArtifactResolver(root=models)
    for artifact in _TRAINED_ARTIFACTS:
        assert resolver.resolve(artifact).exists()


def test_fitted_artifacts_are_immediately_reusable_by_classify(fitted):
    """A fit() run must leave the workspace in a state classify() can read —
    the two paths share one artifact seam."""
    _, models, cloud = fitted
    pipeline = WaterPipeline.from_local_models(models)
    state = pipeline.classify(cloud)
    assert state.final_label is not None
    assert len(state.final_label) == len(cloud)


def test_fit_derives_its_own_scalar_stats_not_the_shipped_ones(fitted):
    """Training always standardises from the data it trains on, so the
    cross-site standardisation question does not arise for fit()."""
    import json

    _, models, _ = fitted
    stats = json.loads(LocalArtifactResolver(root=models)
                       .resolve(ArtifactId.WCN_STATS).read_text())
    assert len(stats["scalar_stats"]["mean"][0]) == WcnConfig().arch.n_scalar
