"""Shared fixtures: a synthetic mini "river" point cloud with fabricated
waveforms, used by the fast (no-model-required) integration tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lidarwater import PointCloud

RIVER_CENTER_X = 20.0
RIVER_HALF_WIDTH = 5.0
CANOPY_DIST_FROM_CENTER = 12.0


@pytest.fixture
def synthetic_river():
    """Returns (cloud, is_water, is_canopy) for a straight synthetic channel
    along y, with land on either bank and canopy near the plot edges."""
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.uniform(0, 40, n)
    y = rng.uniform(0, 100, n)
    dist_from_center = np.abs(x - RIVER_CENTER_X)
    is_water = dist_from_center < RIVER_HALF_WIDTH
    is_canopy = (~is_water) & (dist_from_center > CANOPY_DIST_FROM_CENTER)

    z = np.where(
        is_water, 259.5 + rng.normal(0, 0.05, n),
        np.where(is_canopy, 264.0 + rng.normal(0, 0.3, n),
                 262.0 + dist_from_center * 0.05 + rng.normal(0, 0.1, n)),
    )
    reflectance = np.where(is_water, -18.0 + rng.normal(0, 1, n), -8.0 + rng.normal(0, 1, n))

    times_list, amps_list = [], []
    for water in is_water:
        if water:
            t, a = np.arange(20), np.zeros(20)
            a[2:6] = [200, 800, 600, 150]
        else:
            t, a = np.arange(40), np.zeros(40)
            a[2:5] = [150, 400, 150]
            a[15:19] = [120, 300, 280, 100]
            a[30:33] = [110, 250, 90]
        times_list.append(t)
        amps_list.append(a)

    points_df = pd.DataFrame({"x": x, "y": y, "z": z, "reflectance_dB": reflectance})
    wf_df = pd.DataFrame({"Time [SI]": times_list, "Amplitude [ADC]": amps_list})
    cloud = PointCloud.from_dataframe(points_df, wf_df)
    return cloud, is_water, is_canopy
