import dataclasses

import pytest

from lidarwater.config import BoundaryConfig, PipelineConfig, RunConfig, Stage


def test_pipeline_config_defaults_construct():
    config = PipelineConfig()
    assert config.zones.z_canopy_min == 263.3
    assert config.surface.water_tol_m == 0.30


def test_boundary_config_rejects_unordered_thresholds():
    with pytest.raises(ValueError, match="prob_outer < prob_center < prob_inner"):
        BoundaryConfig(prob_inner=0.3, prob_center=0.5, prob_outer=0.65)


def test_boundary_config_accepts_custom_ordered_thresholds():
    config = BoundaryConfig(prob_inner=0.8, prob_center=0.5, prob_outer=0.2)
    assert config.prob_outer < config.prob_center < config.prob_inner


def test_run_config_geometry_requires_wcn_or_autolabel():
    with pytest.raises(ValueError, match="Stage.GEOMETRY requires"):
        RunConfig(stages=(Stage.FEATURES, Stage.GEOMETRY))


def test_run_config_canopy_requires_geometry():
    with pytest.raises(ValueError, match="Stage.CANOPY requires"):
        RunConfig(stages=(Stage.FEATURES, Stage.WCN, Stage.CANOPY))


def test_run_config_merge_requires_canopy():
    with pytest.raises(ValueError, match="Stage.MERGE requires"):
        RunConfig(stages=(Stage.FEATURES, Stage.WCN, Stage.GEOMETRY, Stage.MERGE))


def test_run_config_boundary_requires_geometry():
    with pytest.raises(ValueError, match="Stage.BOUNDARY requires"):
        RunConfig(stages=(Stage.FEATURES, Stage.BOUNDARY))


def test_run_config_default_classify_stages_are_valid():
    RunConfig()  # must not raise


def test_config_is_overridable_via_replace():
    config = PipelineConfig()
    zones = dataclasses.replace(config.zones, z_canopy_min=270.0)
    replaced = dataclasses.replace(config, zones=zones)
    assert replaced.zones.z_canopy_min == 270.0
    assert config.zones.z_canopy_min == 263.3   # original untouched (frozen dataclass)
