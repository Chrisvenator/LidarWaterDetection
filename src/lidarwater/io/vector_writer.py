"""GeoJSON export for river-boundary contours."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import BoundaryConfig
from ..types import PipelineState

_LEVEL_NAMES_BY_ROLE = ("outer", "center", "inner")   # ascending prob order


def _level_name(level: float, config: BoundaryConfig) -> str:
    if level == config.prob_inner:
        return "inner"
    if level == config.prob_center:
        return "center"
    if level == config.prob_outer:
        return "outer"
    return f"p{level:.2f}"


def boundary_to_geojson(state: PipelineState, config: BoundaryConfig) -> dict:
    """Build a GeoJSON FeatureCollection of the boundary contours as LineStrings."""
    if state.boundary_contours is None:
        raise ValueError("state.boundary_contours is not set — run the boundary stage first")

    features = []
    for level, segments in sorted(state.boundary_contours.items(), reverse=True):
        name = _level_name(level, config)
        for i, seg in enumerate(segments):
            features.append({
                "type": "Feature",
                "properties": {
                    "level": level,
                    "name": name,
                    "segment": i,
                    "n_points": len(seg),
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[round(float(p[0]), 4), round(float(p[1]), 4)] for p in seg],
                },
            })
    return {"type": "FeatureCollection", "features": features}


def write_geojson(state: PipelineState, config: BoundaryConfig, path: str | Path) -> None:
    fc = boundary_to_geojson(state, config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fc, indent=2))
