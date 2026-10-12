"""LAS/LAZ export with ASPRS classification codes — the primary hand-off
format to OPALS (opalsImport auto-detects LAS/LAZ) and other GIS/point-cloud
tools. Native per-point attributes (water_proba, canopy_proba, raw_label)
round-trip through OPALS as LAS Extra Bytes."""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np

from ..config import LabelScheme, OutputConfig
from ..types import PipelineState

LABEL_LAND, LABEL_WATER, LABEL_UNCERTAIN, LABEL_RECON_WATER, LABEL_CANOPY = 0, 1, 2, 3, 4

# ASPRS class codes
_ASPRS_UNCLASSIFIED = 1
_ASPRS_GROUND = 2
_ASPRS_HIGH_VEG = 5
_ASPRS_WATER = 9
_ASPRS_BATHY_BOTTOM = 40
_ASPRS_WATER_SURFACE = 41


def classification_codes(state: PipelineState, config: OutputConfig) -> np.ndarray:
    """Map the pipeline's native 0-4 label to an ASPRS classification code.

    final_label: 0=land, 1=water, 2=uncertain, 3=recon-water(under canopy), 4=canopy.
    Water/recon-water split into bed (40) vs surface (41) by z relative to the
    local water surface, when that's available (requires the geometry stage).
    """
    if state.final_label is None:
        raise ValueError("state.final_label is not set — run the merge stage first")
    label = state.final_label

    if state.local_surface_z is not None:
        is_bed = state.cloud.z < state.local_surface_z
    else:
        is_bed = np.zeros(len(label), dtype=bool)

    codes = np.full(len(label), _ASPRS_UNCLASSIFIED, dtype=np.uint8)

    if config.label_scheme == LabelScheme.TOPO_BATHY:
        codes[label == LABEL_LAND] = _ASPRS_GROUND
        codes[label == LABEL_CANOPY] = _ASPRS_HIGH_VEG
        water_like = np.isin(label, (LABEL_WATER, LABEL_RECON_WATER))
        codes[water_like & is_bed] = _ASPRS_BATHY_BOTTOM
        codes[water_like & ~is_bed] = _ASPRS_WATER_SURFACE
        # LABEL_UNCERTAIN stays _ASPRS_UNCLASSIFIED
    elif config.label_scheme == LabelScheme.ASPRS_BASIC:
        codes[label == LABEL_LAND] = _ASPRS_GROUND
        codes[label == LABEL_CANOPY] = _ASPRS_HIGH_VEG
        codes[np.isin(label, (LABEL_WATER, LABEL_RECON_WATER))] = _ASPRS_WATER
        # LABEL_UNCERTAIN stays _ASPRS_UNCLASSIFIED
    else:
        raise ValueError(f"unknown label scheme: {config.label_scheme}")

    return codes


def write_laz(state: PipelineState, config: OutputConfig, path: str | Path) -> None:
    """Write the classified cloud as LAS/LAZ (point format 6, LAS 1.4).

    Extra bytes: water_proba (f32), canopy_proba (f32), raw_label (u8, the
    native 0-4 scheme — lossless fallback alongside the ASPRS code).
    """
    if state.final_label is None:
        raise ValueError("state.final_label is not set — run the merge stage first")

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.add_crs(_epsg_to_pyproj_crs(config.crs_epsg))

    las = laspy.LasData(header)
    ox, oy, oz = config.xyz_offset
    las.x = state.cloud.x + ox
    las.y = state.cloud.y + oy
    las.z = state.cloud.z + oz
    las.classification = classification_codes(state, config)

    water_proba = (
        state.wcn_proba if state.wcn_proba is not None
        else np.full(len(state.cloud), np.nan, dtype=np.float32)
    )
    canopy_proba = (
        state.canopy_proba if state.canopy_proba is not None
        else np.full(len(state.cloud), np.nan, dtype=np.float32)
    )

    las.add_extra_dim(laspy.ExtraBytesParams(name="water_proba", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="canopy_proba", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="raw_label", type=np.uint8))
    las.water_proba = water_proba.astype(np.float32)
    las.canopy_proba = canopy_proba.astype(np.float32)
    las.raw_label = state.final_label.astype(np.uint8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(path))


def _epsg_to_pyproj_crs(epsg: int):
    import pyproj

    return pyproj.CRS.from_epsg(epsg)
