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
    REFLECTANCE_FEATURE,
    STANDARDIZE_ARTIFACT,
    STANDARDIZE_SITE,
    WCN_SCALAR_FEATURES,
    FeatureConfig,
    WcnConfig,
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


def test_derivation_leaves_model_shape_and_hyperparameters_alone(synthetic_river):
    """Only the scalar input width may move, and only because the site-fitted
    feature set differs — the transformer shape and every training
    hyperparameter stay as shipped."""
    cloud, _, _ = synthetic_river
    config, _ = derive_site_config(_translate(cloud, dz=120.0, d_reflectance=14.0))
    base = PipelineConfig()

    assert config.wcn.train == base.wcn.train
    for field in ("d_model", "n_heads", "n_layers", "n_patches"):
        assert getattr(config.wcn.arch, field) == getattr(base.wcn.arch, field)
    assert config.features.grid_size == base.features.grid_size


def test_site_fitted_feature_set_adds_reflectance(synthetic_river):
    """Reflectance is the strongest water/land signal on some surveys; it is
    excluded from the shipped cross-site set only because its dB scale does
    not transfer."""
    cloud, _, _ = synthetic_river
    config, _ = derive_site_config(cloud)

    assert config.wcn.scalar_features[:-1] == WCN_SCALAR_FEATURES
    assert config.wcn.scalar_features[-1] == REFLECTANCE_FEATURE
    assert config.wcn.arch.n_scalar == len(config.wcn.scalar_features)


def test_wcn_config_rejects_a_scalar_width_that_contradicts_its_features():
    from lidarwater.config import WcnArchConfig

    with pytest.raises(ValueError, match="must match the number of"):
        WcnConfig(arch=WcnArchConfig(n_scalar=11),
                  scalar_features=(*WCN_SCALAR_FEATURES, REFLECTANCE_FEATURE))


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
    from lidarwater._stages.features import _grid_origin_time, _waveform_to_grid

    cloud = _full_record_cloud()
    times, amps = next(cloud.iter_waveforms())
    by_first_sample = FeatureConfig()
    by_first_return = dataclasses.replace(FeatureConfig(),
                                          grid_origin=GRID_ORIGIN_FIRST_RETURN)
    first_sample = _waveform_to_grid(times, amps, by_first_sample,
                                     _grid_origin_time(times, amps, by_first_sample))
    first_return = _waveform_to_grid(times, amps, by_first_return,
                                     _grid_origin_time(times, amps, by_first_return))

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
    for path in (workspace.cache_dir, workspace.models_dir, workspace.pointclouds_dir):
        assert path.is_dir() and workspace.root in path.parents

    config = workspace.apply_to(PipelineConfig())
    assert config.run.cache_dir == workspace.cache_dir
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


# ── full-record handling: noise gate + per-site standardisation ──────────────

def test_full_record_cloud_enables_the_noise_gate_and_site_standardisation():
    config, profile = derive_site_config(_full_record_cloud())
    assert profile.full_record is True
    assert config.features.noise_gate is True
    assert config.wcn.standardize == STANDARDIZE_SITE


def test_svb_style_cloud_keeps_the_deployed_checkpoint_behaviour(synthetic_river):
    cloud, _, _ = synthetic_river
    config, profile = derive_site_config(cloud)
    assert profile.full_record is False
    assert config.features.noise_gate is False
    assert config.wcn.standardize == STANDARDIZE_ARTIFACT


def test_noise_gate_drops_the_pedestal_and_keeps_the_echo():
    from lidarwater._stages.features import _gate_noise

    cloud = _full_record_cloud(record_len=700, return_at=560)
    times, amps = next(cloud.iter_waveforms())
    gated_t, gated_a = _gate_noise(times, amps,
                                   dataclasses.replace(FeatureConfig(), noise_gate=True))

    assert len(gated_t) < len(times) / 2          # most of the record was noise
    assert gated_a.max() == amps.max()            # the echo survived
    # A threshold on Gaussian noise always lets a few tail samples through;
    # what matters is that the return window dominates what is kept.
    assert (gated_t >= 500).mean() > 0.9


def test_noise_gate_keeps_the_raw_record_when_it_would_strip_everything():
    from lidarwater._stages.features import _gate_noise

    flat = np.full(50, 100.0)                      # no echo above the floor
    times = np.arange(50)
    gated_t, gated_a = _gate_noise(times, flat,
                                   dataclasses.replace(FeatureConfig(), noise_gate=True))
    assert len(gated_t) == len(times)


def test_wcn_config_rejects_an_unknown_standardisation_mode():
    with pytest.raises(ValueError, match="standardize must be one of"):
        WcnConfig(standardize="whatever")


# ── diagnostic figures ───────────────────────────────────────────────────────

def test_write_all_renders_every_figure_the_state_supports(synthetic_river, tmp_path):
    from lidarwater._stages import canopy, features, surface
    from lidarwater.plots import write_all

    cloud, is_water, _ = synthetic_river
    config = PipelineConfig()
    state = features.run(cloud, config.features)
    state.wcn_proba = np.where(is_water, 0.95, 0.05).astype(np.float32)
    state.wcn_xgb_proba = np.where(is_water, 0.90, 0.10).astype(np.float32)
    surface.run(state, config.surface, geometry_only=True)
    state.canopy_pred = np.zeros(len(cloud), dtype=np.int8)
    canopy.merge(state)

    written = write_all(state, tmp_path / "plots")
    assert {p.name for p in written} == {
        "classes_topdown.png", "cross_section.png",
        "elevation_profile.png", "water_probability.png"}
    assert all(p.stat().st_size > 0 for p in written)


def test_write_all_skips_figures_whose_inputs_are_missing(synthetic_river, tmp_path):
    """A partial stage set must not crash the plotting step."""
    from lidarwater._stages import features
    from lidarwater.plots import write_all

    cloud, _, _ = synthetic_river
    state = features.run(cloud, PipelineConfig().features)
    assert write_all(state, tmp_path / "plots") == []


@pytest.mark.parametrize("cloud_name", ["synthetic_river", "full_record"])
def test_bed_reconstruction_follows_the_measured_canopy_fraction(cloud_name, request):
    """Phase 3b recovers water hidden under crowns; with no crowns it can
    only invent a class the site does not contain. Asserted as the rule
    rather than a fixed expectation, since what a cloud measures depends on
    the cloud."""
    cloud = (request.getfixturevalue("synthetic_river")[0]
             if cloud_name == "synthetic_river" else _full_record_cloud())
    config, profile = derive_site_config(cloud)
    expected = profile.canopy_fraction_above_probe >= config.surface.bed.min_canopy_frac
    assert config.surface.bed.enabled is expected


def test_bed_reconstruction_is_skipped_when_disabled(synthetic_river):
    """With Phase 3b off, no point may carry the reconstructed-water label."""
    from lidarwater._stages import features, surface

    cloud, is_water, _ = synthetic_river
    config = PipelineConfig()
    config = dataclasses.replace(config, surface=dataclasses.replace(
        config.surface, bed=dataclasses.replace(config.surface.bed, enabled=False)))

    state = features.run(cloud, config.features)
    state.wcn_proba = np.where(is_water, 0.95, 0.05).astype(np.float32)
    state.wcn_xgb_proba = np.where(is_water, 0.90, 0.10).astype(np.float32)
    surface.run(state, config.surface, geometry_only=True)

    assert not (state.reconstructed_label == config.surface.bed.recon_label).any()
    assert state.metrics["geometry"]["label_counts"]["reconstructed_water"] == 0


def test_grid_origin_is_measured_before_the_noise_gate(monkeypatch):
    """The gate removes low samples; measuring the origin after it would drag
    bin 0 later by a few samples, differently for each point."""
    from lidarwater._stages import features as feat

    config = dataclasses.replace(FeatureConfig(), noise_gate=True,
                                 grid_origin=GRID_ORIGIN_FIRST_RETURN)
    cloud = _full_record_cloud()
    times, amps = next(cloud.iter_waveforms())

    raw_origin = feat._grid_origin_time(times, amps, config)
    gated_t, gated_a = feat._gate_noise(times, amps, config)
    assert feat._grid_origin_time(times, amps, config) == raw_origin
    # measuring on the gated record is what we must NOT do
    assert len(gated_t) < len(times)


# ── elevation-free label bootstrap ───────────────────────────────────────────

def _sheet_cloud(n: int = 6000, land_reflectance: float = 6.0) -> PointCloud:
    """A flat water sheet with a flat land bar beside it, at the SAME height.

    Elevation cannot separate these two by construction — which is the case
    the surface bootstrap exists to handle.
    """
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 60, n)
    y = rng.uniform(0, 60, n)
    is_land = x > 40
    z = np.full(n, 380.75) + rng.normal(0, 0.01, n)
    # water cells also carry riverbed returns below the surface
    bed = (~is_land) & (rng.random(n) < 0.4)
    z[bed] -= rng.uniform(0.3, 1.5, bed.sum())
    reflectance = np.where(is_land, land_reflectance, -5.0) + rng.normal(0, 0.5, n)

    times = [np.arange(20)] * n
    amps = [np.array([0, 0, 200, 800, 600, 150] + [0] * 14, dtype=float)] * n
    return PointCloud.from_dataframe(
        pd.DataFrame({"x": x, "y": y, "z": z, "reflectance_dB": reflectance}),
        pd.DataFrame({"Time [SI]": times, "Amplitude [ADC]": amps})), is_land


def test_surface_bootstrap_separates_water_from_land_at_equal_height():
    from lidarwater._stages import features
    from lidarwater._stages.autolabel import create_surface_labels
    from lidarwater.config import BOOTSTRAP_SURFACE, BootstrapConfig

    cloud, is_land = _sheet_cloud()
    feats = features.run(cloud, PipelineConfig().features).features
    labels = create_surface_labels(feats, BootstrapConfig(method=BOOTSTRAP_SURFACE))

    water_recall = (labels[~is_land] == 1).mean()
    land_recall = (labels[is_land] == 0).mean()
    assert water_recall > 0.9, f"water recall {water_recall:.1%}"
    assert land_recall > 0.9, f"land recall {land_recall:.1%}"


def test_zone_bootstrap_cannot_separate_them_at_equal_height():
    """The contrast the surface method exists for: with water and land at the
    same elevation, an elevation rule has no information at all."""
    from lidarwater._stages.autolabel import create_labels

    cloud, is_land = _sheet_cloud()
    z = cloud.z
    labels = create_labels(z, PipelineConfig().zones)
    decided = labels >= 0
    if decided.any():   # whatever it decides cannot depend on the class
        assert abs((labels[decided & ~is_land] == 1).mean()
                   - (labels[decided & is_land] == 1).mean()) < 0.2


def test_derived_config_selects_the_elevation_free_bootstrap(synthetic_river):
    from lidarwater.config import BOOTSTRAP_SURFACE, BOOTSTRAP_ZONES

    cloud, _, _ = synthetic_river
    assert PipelineConfig().bootstrap.method == BOOTSTRAP_ZONES     # Pielach default
    config, _ = derive_site_config(cloud)
    assert config.bootstrap.method == BOOTSTRAP_SURFACE


def test_bootstrap_config_rejects_an_unknown_method():
    from lidarwater.config import BootstrapConfig

    with pytest.raises(ValueError, match="bootstrap method must be one of"):
        BootstrapConfig(method="magic")
