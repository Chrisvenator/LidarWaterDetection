"""Fast, no-trained-model-required integration test: runs the geometry
pipeline (features -> surface -> canopy -> merge -> boundary) end to end on
a fabricated micro river, using ground-truth-derived stand-ins for the WCN
and canopy model outputs. Verifies shapes, state wiring, and that the
geometry phases recover the known water/canopy regions above chance,
without needing real trained checkpoints or the Pielach dataset.
"""

from __future__ import annotations

import numpy as np

from lidarwater import PipelineConfig
from lidarwater._stages import boundary, canopy, features, surface


def test_features_stage_produces_expected_columns(synthetic_river):
    cloud, _, _ = synthetic_river
    config = PipelineConfig()
    state = features.run(cloud, config.features)

    assert len(state.features) == len(cloud)
    assert state.waveform_grids.shape == (len(cloud), config.features.grid_size)
    assert state.waveform_grids_norm.max() <= 1.0 + 1e-5
    for col in ("energy_concentration", "n_peaks", "planarity", "height_above_local_min"):
        assert col in state.features.columns
        assert state.features[col].notna().all()


def test_geometry_stage_recovers_water_footprint(synthetic_river):
    cloud, is_water, _ = synthetic_river
    config = PipelineConfig()
    state = features.run(cloud, config.features)

    # Stand in for WCN v9 output — the geometry math under test doesn't
    # depend on how the probas were produced.
    state.wcn_proba = np.where(is_water, 0.95, 0.05).astype(np.float32)
    state.wcn_xgb_proba = np.where(is_water, 0.90, 0.10).astype(np.float32)

    surface.run(state, config.surface, geometry_only=True)

    assert state.in_footprint.dtype == bool
    assert state.local_surface_z.shape == (len(cloud),)
    predicted_water = np.isin(state.reconstructed_label, (1, 3))
    # Water recall well above chance, but not near 100%: the 1m footprint
    # erosion (a real, intentional conservative-footprint design choice —
    # see FootprintConfig.erosion_m) always removes a channel-edge margin,
    # and on this synthetic channel's 5m half-width that margin is a
    # substantial fraction of uniformly-random points.
    recall = predicted_water[is_water].mean()
    assert recall > 0.6, f"water recall too low: {recall:.2f}"
    # No water predicted far outside the true channel.
    false_positive_rate = predicted_water[~is_water].mean()
    assert false_positive_rate < 0.1, f"false positive rate too high: {false_positive_rate:.2f}"


def test_boundary_contours_bracket_the_channel(synthetic_river):
    cloud, is_water, _ = synthetic_river
    config = PipelineConfig()
    state = features.run(cloud, config.features)
    state.wcn_proba = np.where(is_water, 0.95, 0.05).astype(np.float32)
    state.wcn_xgb_proba = np.where(is_water, 0.90, 0.10).astype(np.float32)
    surface.run(state, config.surface, geometry_only=True)

    boundary.run(state, config.boundary)

    assert state.boundary_contours is not None
    inner = state.boundary_contours.get(config.boundary.prob_inner, [])
    outer = state.boundary_contours.get(config.boundary.prob_outer, [])
    assert len(inner) > 0 and len(outer) > 0
    # Inner (conservative) contour band should be narrower in x than the
    # outer (generous) one, on average.
    inner_x_span = np.mean([seg[:, 0].max() - seg[:, 0].min() for seg in inner])
    outer_x_span = np.mean([seg[:, 0].max() - seg[:, 0].min() for seg in outer])
    assert inner_x_span <= outer_x_span + 1.0   # small tolerance for smoothing noise


def test_canopy_and_merge_produce_final_label(synthetic_river):
    cloud, is_water, is_canopy = synthetic_river
    config = PipelineConfig()
    state = features.run(cloud, config.features)
    state.wcn_proba = np.where(is_water, 0.95, 0.05).astype(np.float32)
    state.wcn_xgb_proba = np.where(is_water, 0.90, 0.10).astype(np.float32)
    surface.run(state, config.surface, geometry_only=True)

    canopy_feat = canopy.build_canopy_features(state, config.canopy)
    assert len(canopy_feat) == len(cloud)
    assert "height_above_ref" in canopy_feat.columns

    # Stand in for the trained canopy XGBoost.
    state.canopy_proba = np.where(is_canopy, 0.9, 0.1).astype(np.float32)
    state.canopy_pred = (state.canopy_proba > config.canopy.threshold).astype(np.int8)

    canopy.merge(state)

    assert state.final_label is not None
    assert set(np.unique(state.final_label)).issubset({0, 1, 2, 3, 4})
    canopy_recall = (state.final_label[is_canopy] == 4).mean()
    assert canopy_recall > 0.8, f"canopy recall too low: {canopy_recall:.2f}"
    # Water is never overridden to canopy even where the (fabricated) canopy
    # proba disagrees, because the fixture doesn't overlap water and canopy —
    # so the water recall ceiling here is the same footprint-erosion effect
    # as test_geometry_stage_recovers_water_footprint, not a merge bug.
    assert (state.final_label[is_water] == 1).mean() > 0.6
