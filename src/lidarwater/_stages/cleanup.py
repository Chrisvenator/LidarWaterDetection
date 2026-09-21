"""Spatial regularisation of the final labels.

Classification is point-by-point, so nothing else in the pipeline uses the
fact that water and land come in contiguous regions. A point whose label
disagrees with almost all of its neighbours is far more likely to be a
threshold artifact than a real feature: measured on Inn, the isolated land
points inside the channel sit at the water surface (-0.04 m), carry water's
reflectance (-4.0 dB against land's +0.8) and have a median water
probability of 0.42 — just the wrong side of the cut.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..config import CleanupConfig
from ..types import PipelineState

LABEL_LAND, LABEL_WATER, LABEL_CANOPY = 0, 1, 4
WATER_LABELS = (1, 3)


def majority_filter(state: PipelineState, config: CleanupConfig) -> PipelineState:
    """Flip points whose label contradicts an overwhelming local majority.

    Canopy is never moved: it is a genuinely sparse class that a spatial
    majority would erase.
    """
    if state.final_label is None:
        raise ValueError("majority filter needs state.final_label — run the merge stage first")

    labels = state.final_label
    xy = state.cloud.xyz[:, :2]
    neighbours = cKDTree(xy).query(xy, k=config.k + 1, workers=-1)[1][:, 1:]

    is_water = np.isin(labels, WATER_LABELS)
    water_share = is_water[neighbours].mean(axis=1)
    land_share = (labels[neighbours] == LABEL_LAND).mean(axis=1)

    movable = labels != LABEL_CANOPY
    cleaned = labels.copy()
    cleaned[movable & (labels == LABEL_LAND) & (water_share >= config.min_agreement)] = LABEL_WATER
    cleaned[movable & is_water & (land_share >= config.min_agreement)] = LABEL_LAND

    state.final_label = cleaned
    state.metrics["cleanup"] = {
        "k": config.k,
        "min_agreement": config.min_agreement,
        "points_moved": int((cleaned != labels).sum()),
        "land_to_water": int(((labels == LABEL_LAND) & (cleaned != labels)).sum()),
        "water_to_land": int((is_water & (cleaned != labels)).sum()),
    }
    return state
