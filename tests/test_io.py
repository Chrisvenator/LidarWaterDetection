"""Output conformance tests: LAZ (classification codes, extra bytes, CRS)
and GeoJSON, plus PointCloud construction consistency.
"""

from __future__ import annotations

import json

import laspy
import numpy as np
import pytest

from lidarwater import PipelineConfig, PointCloud
from lidarwater._stages import boundary, canopy, features, surface
from lidarwater.config import LabelScheme, OutputConfig
from lidarwater.io import classification_codes, write_geojson, write_laz


@pytest.fixture
def classified_state(synthetic_river):
    cloud, is_water, is_canopy = synthetic_river
    config = PipelineConfig()
    state = features.run(cloud, config.features)
    state.wcn_proba = np.where(is_water, 0.95, 0.05).astype(np.float32)
    state.wcn_xgb_proba = np.where(is_water, 0.90, 0.10).astype(np.float32)
    surface.run(state, config.surface, geometry_only=True)
    state.canopy_proba = np.where(is_canopy, 0.9, 0.1).astype(np.float32)
    state.canopy_pred = (state.canopy_proba > config.canopy.threshold).astype(np.int8)
    canopy.merge(state)
    boundary.run(state, config.boundary)
    return state, config


def test_classification_codes_topo_bathy_scheme(classified_state):
    state, config = classified_state
    codes = classification_codes(state, config.output)
    assert codes.dtype == np.uint8
    valid_codes = {1, 2, 5, 40, 41}
    assert set(np.unique(codes)).issubset(valid_codes)
    # Land -> ground (2)
    assert np.all(codes[state.final_label == 0] == 2)
    # Canopy -> high vegetation (5)
    assert np.all(codes[state.final_label == 4] == 5)
    # Water -> bed (40) or surface (41), never anything else
    water_codes = codes[state.final_label == 1]
    assert set(np.unique(water_codes)).issubset({40, 41})


def test_classification_codes_asprs_basic_scheme(classified_state):
    state, config = classified_state
    basic = OutputConfig(label_scheme=LabelScheme.ASPRS_BASIC)
    codes = classification_codes(state, basic)
    assert np.all(codes[state.final_label == 1] == 9)
    assert np.all(codes[state.final_label == 0] == 2)
    assert np.all(codes[state.final_label == 4] == 5)


def test_write_laz_round_trips(classified_state, tmp_path):
    state, config = classified_state
    out_path = tmp_path / "classified.laz"
    write_laz(state, config.output, out_path)

    las = laspy.read(str(out_path))
    assert len(las.points) == len(state.cloud)
    np.testing.assert_allclose(np.asarray(las.x), state.cloud.x, atol=1e-2)
    np.testing.assert_allclose(np.asarray(las.z), state.cloud.z, atol=1e-2)

    extra_dims = set(las.point_format.extra_dimension_names)
    assert {"water_proba", "canopy_proba", "raw_label"}.issubset(extra_dims)
    np.testing.assert_array_equal(np.asarray(las.raw_label), state.final_label.astype(np.uint8))
    assert las.header.parse_crs() is not None


def test_write_laz_requires_final_label(synthetic_river, tmp_path):
    cloud, _, _ = synthetic_river
    config = PipelineConfig()
    state = features.run(cloud, config.features)
    with pytest.raises(ValueError, match="final_label"):
        write_laz(state, config.output, tmp_path / "out.laz")


def test_write_geojson_produces_valid_linestrings(classified_state, tmp_path):
    state, config = classified_state
    out_path = tmp_path / "boundary.geojson"
    write_geojson(state, config.boundary, out_path)

    fc = json.loads(out_path.read_text())
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) > 0
    names = {f["properties"]["name"] for f in fc["features"]}
    assert names <= {"inner", "center", "outer"}
    for f in fc["features"]:
        assert f["geometry"]["type"] == "LineString"
        assert len(f["geometry"]["coordinates"]) >= 2


def test_pointcloud_from_dataframe_matches_manual_construction():
    import pandas as pd

    points = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [10.0, 11.0],
                           "reflectance_dB": [-10.0, -12.0]})
    waveforms = pd.DataFrame({"Time [SI]": [[0, 1, 2], [0, 1]],
                              "Amplitude [ADC]": [[10, 20, 5], [100, 50]]})
    cloud = PointCloud.from_dataframe(points, waveforms)

    assert len(cloud) == 2
    t0, a0 = cloud.waveform(0)
    np.testing.assert_array_equal(t0, [0, 1, 2])
    np.testing.assert_array_equal(a0, [10, 20, 5])
    t1, a1 = cloud.waveform(1)
    np.testing.assert_array_equal(t1, [0, 1])
    np.testing.assert_array_equal(a1, [100, 50])


def test_pointcloud_rejects_mismatched_row_counts():
    import pandas as pd

    points = pd.DataFrame({"x": [0.0], "y": [0.0], "z": [10.0], "reflectance_dB": [-10.0]})
    waveforms = pd.DataFrame({"Time [SI]": [[0, 1], [0, 1]],
                              "Amplitude [ADC]": [[10, 20], [10, 20]]})
    with pytest.raises(ValueError, match="row counts differ"):
        PointCloud.from_dataframe(points, waveforms)
