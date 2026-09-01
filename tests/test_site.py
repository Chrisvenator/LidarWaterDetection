"""Site derivation, workspace isolation, and the dataset-portability fixes
they depend on (grid anchoring, canopy-free short-circuit, folder reading).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from lidarwater import PipelineConfig, PointCloud, Workspace, derive_site_config
from lidarwater.config import (
    GRID_ORIGIN_FIRST_RETURN,
    GRID_ORIGIN_FIRST_SAMPLE,
    FeatureConfig,
)
from lidarwater.io import read_dataset_dir
from lidarwater.site import (
    DERIVED_FIELDS,
    PIELACH_ENERGY_GATE,
    PIELACH_WATER_Z,
    _REFLECTANCE_FIELDS,
    _Z_FIELDS,
    estimate_water_level,
)


def _translate(cloud: PointCloud, dz: float = 0.0, d_reflectance: float = 0.0) -> PointCloud:
    return dataclasses.replace(
        cloud,
        xyz=cloud.xyz + np.array([0.0, 0.0, dz]),
        reflectance_db=cloud.reflectance_db + d_reflectance,
    )


def test_water_level_is_the_densest_elevation_bin():
    z = np.concatenate([np.full(1000, 380.7), np.linspace(381, 390, 100)])
    assert estimate_water_level(z) == pytest.approx(380.7, abs=0.06)


def test_derivation_is_near_identity_on_a_pielach_like_cloud(synthetic_river):
    cloud, _, _ = synthetic_river
    config, profile = derive_site_config(cloud)

    assert profile.water_level_z == pytest.approx(PIELACH_WATER_Z, abs=1.0)
    assert config.features.grid_origin == GRID_ORIGIN_FIRST_SAMPLE
    assert profile.first_bin_energy_fraction == pytest.approx(1.0)


def test_elevation_thresholds_follow_the_water_level(synthetic_river):
    cloud, _, _ = synthetic_river
    base, base_profile = derive_site_config(cloud)
    shifted, profile = derive_site_config(_translate(cloud, dz=120.0))

    assert profile.z_shift_m - base_profile.z_shift_m == pytest.approx(120.0, abs=0.11)
    for path, name in [("zones", "z_canopy_min"), ("canopy", "z_clear_max"),
                       ("boundary", "canopy_z_max")]:
        before = getattr(getattr(base, path), name)
        after = getattr(getattr(shifted, path), name)
        assert after - before == pytest.approx(120.0, abs=0.11)
    assert shifted.surface.footprint.riverbed_z_max - base.surface.footprint.riverbed_z_max \
        == pytest.approx(120.0, abs=0.11)


def test_reflectance_gates_follow_the_scanner_scale(synthetic_river):
    cloud, _, _ = synthetic_river
    base, base_profile = derive_site_config(cloud)
    shifted, profile = derive_site_config(_translate(cloud, d_reflectance=14.0))

    assert profile.reflectance_shift_db - base_profile.reflectance_shift_db \
        == pytest.approx(14.0, abs=0.5)
    assert shifted.surface.surface_grid.reflectance_max_db \
        > base.surface.surface_grid.reflectance_max_db
    assert shifted.surface.bed.reflectance_max_db - base.surface.bed.reflectance_max_db \
        == pytest.approx(14.0, abs=0.5)


def _config_fields(node, path: str = ""):
    """(dotted path, field name) for every leaf field of a nested config."""
    for field in dataclasses.fields(node):
        value = getattr(node, field.name)
        if dataclasses.is_dataclass(value):
            yield from _config_fields(value, f"{path}.{field.name}" if path else field.name)
        else:
            yield path, field.name, value


def test_every_absolute_threshold_in_the_config_is_registered_for_rebasing():
    """The rebasing registries are hand-maintained: a z or reflectance
    threshold added to config.py later would silently stay on Pielach's
    scale at every other site. Fail here instead."""
    registered = set(DERIVED_FIELDS)
    missing = []
    for path, name, value in _config_fields(PipelineConfig()):
        absolute_z = name.startswith("z_") or "_z_" in name or name.endswith("_z")
        absolute_db = "reflectance" in name and "db" in name
        if (absolute_z or absolute_db) and (path, name) not in registered:
            missing.append(f"{path}.{name} = {value}")

    assert not missing, (
        "absolute thresholds missing from site.py's DERIVED_FIELDS: {}".format(missing)
    )


def test_rebasing_registries_only_name_fields_that_exist():
    config = PipelineConfig()
    for path, name in DERIVED_FIELDS:
        node = config
        for part in path.split("."):
            node = getattr(node, part)
        assert hasattr(node, name), f"{path}.{name} is registered but does not exist"


def test_derivation_leaves_model_architecture_alone(synthetic_river):
    cloud, _, _ = synthetic_river
    config, _ = derive_site_config(_translate(cloud, dz=120.0, d_reflectance=14.0))
    assert config.wcn == PipelineConfig().wcn
    assert config.features.grid_size == PipelineConfig().features.grid_size


# ── waveform grid anchoring ──────────────────────────────────────────────────

def _full_record_cloud(n: int = 40, record_len: int = 700, return_at: int = 560) -> PointCloud:
    """Full-range-gate digitisation: a long noise floor, echo near the end."""
    rng = np.random.default_rng(0)
    times, amps = [], []
    for _ in range(n):
        t = np.arange(record_len)
        a = rng.normal(35, 2, record_len)
        a[return_at:return_at + 30] += 600 * np.exp(-np.arange(30) / 6)
        times.append(t)
        amps.append(a)
    points = pd.DataFrame({
        "x": rng.uniform(0, 10, n), "y": rng.uniform(0, 10, n),
        "z": np.full(n, 380.0), "reflectance_dB": np.full(n, -5.0),
    })
    return PointCloud.from_dataframe(
        points, pd.DataFrame({"Time [SI]": times, "Amplitude [ADC]": amps}))


def test_full_record_waveforms_switch_the_grid_to_return_anchoring():
    _, profile = derive_site_config(_full_record_cloud())
    assert profile.grid_origin == GRID_ORIGIN_FIRST_RETURN
    assert profile.first_bin_energy_fraction < 0.5


def test_return_anchoring_captures_a_late_echo_that_first_sample_misses():
    from lidarwater._stages.features import _waveform_to_grid

    cloud = _full_record_cloud()
    times, amps = next(cloud.iter_waveforms())
    first_sample = _waveform_to_grid(times, amps, FeatureConfig())
    first_return = _waveform_to_grid(
        times, amps, dataclasses.replace(FeatureConfig(), grid_origin=GRID_ORIGIN_FIRST_RETURN))

    assert first_sample.max() < 100          # noise floor only
    assert first_return.max() > 500          # the echo


def test_feature_config_rejects_an_unknown_grid_origin():
    with pytest.raises(ValueError, match="grid_origin must be one of"):
        FeatureConfig(grid_origin="middle")


# ── canopy auto-detection ────────────────────────────────────────────────────

def test_canopy_stage_reports_no_canopy_on_a_flat_site(synthetic_river):
    from lidarwater._stages import canopy, features, surface

    cloud, is_water, _ = synthetic_river
    flat = dataclasses.replace(
        cloud, xyz=np.column_stack([cloud.x, cloud.y, np.where(is_water, 259.5, 260.4)]))
    config = PipelineConfig()

    state = features.run(flat, config.features)
    state.wcn_proba = np.where(is_water, 0.95, 0.05).astype(np.float32)
    state.wcn_xgb_proba = np.where(is_water, 0.90, 0.10).astype(np.float32)
    surface.run(state, config.surface, geometry_only=True)

    canopy.predict(state, config.canopy, resolver=None)   # never loads a model

    assert state.metrics["canopy"]["canopy_present"] is False
    assert not state.canopy_pred.any()
    canopy.merge(state)
    assert not (state.final_label == 4).any()


# ── workspace + reader ───────────────────────────────────────────────────────

def test_workspace_isolates_every_output_under_one_root(tmp_path):
    workspace = Workspace.for_dataset("Inn_DeepLearning", tmp_path).mkdirs()
    assert workspace.root == tmp_path / "Inn_DeepLearning"
    for path in (workspace.cache_dir, workspace.models_dir,
                 workspace.pointclouds_dir, workspace.plot_dir):
        assert path.is_dir() and workspace.root in path.parents

    config = workspace.apply_to(PipelineConfig())
    assert config.run.cache_dir == workspace.cache_dir
    assert config.run.plot_dir == workspace.plot_dir
    assert config.run.stages == PipelineConfig().run.stages


def test_read_dataset_dir_finds_a_suffixed_file_pair(tmp_path, synthetic_river):
    cloud, _, _ = synthetic_river
    pd.DataFrame({"": range(len(cloud)), "x": cloud.x, "y": cloud.y, "z": cloud.z,
                  "_riegl.reflectance": cloud.reflectance_db}
                 ).to_csv(tmp_path / "point_cloud_df_inn.txt", index=False)
    pd.DataFrame({
        "Time [SI]": [str(t) for t, _ in cloud.iter_waveforms()],
        "Amplitude [ADC]": [str(a.astype(int)) for _, a in cloud.iter_waveforms()],
    }).to_csv(tmp_path / "waveform_df_inn.txt", index=False)

    loaded = read_dataset_dir(tmp_path)
    assert len(loaded) == len(cloud)
    assert loaded.z == pytest.approx(cloud.z)


def test_read_dataset_dir_refuses_an_ambiguous_folder(tmp_path):
    with pytest.raises(FileNotFoundError, match="expected exactly one point cloud file"):
        read_dataset_dir(tmp_path)


# ── compact-waveform gate + RANSAC candidate scarcity ────────────────────────

def test_compact_waveform_gate_is_rebased_to_the_sites_pulse_length():
    """The 0.85 gate counts energy in a 30-bin window sized to Pielach's
    pulse. A site whose returns span more bins can never reach it, so the
    threshold must drop to keep passing a comparable share of the cloud."""
    long_pulse = _full_record_cloud(record_len=700, return_at=560)
    config, profile = derive_site_config(long_pulse)

    assert profile.energy_concentration_gate < PIELACH_ENERGY_GATE
    assert config.surface.surface_grid.energy_concentration_min \
        == profile.energy_concentration_gate


def test_ransac_min_samples_stays_a_legal_sklearn_fraction():
    """sklearn rejects min_samples > 1.0 outright; scarce candidates used to
    push the 100-point target past it and abort the geometry stage."""
    from lidarwater._stages.surface import _ransac_min_samples

    for n_candidates in (0, 1, 50, 99, 100, 199, 200, 5_000, 100_000):
        value = _ransac_min_samples(n_candidates)
        assert 0.0 <= value <= 1.0, f"{n_candidates} candidates -> {value}"
    assert _ransac_min_samples(5_000) == 0.5      # plentiful: half of them
    assert _ransac_min_samples(50) == 1.0         # scarce: all of them
