"""Core data containers: input point cloud and threaded pipeline state."""

from __future__ import annotations

import dataclasses
import re
from typing import Any

import numpy as np
import pandas as pd

_NUMBER_RE = re.compile(r"[-+]?\d+")


@dataclasses.dataclass
class PointCloud:
    """In-memory full-waveform LiDAR point cloud.

    Waveforms are ragged (samples per point vary), stored as flat arrays with
    per-point offsets rather than a list of arrays — avoids one Python object
    per point on large clouds.

    Attributes
    ----------
    xyz : (N, 3) float64
    reflectance_db : (N,) float32
    waveform_times : (M,) int32 — concatenated per-point sample times [SI]
    waveform_amps : (M,) float32 — concatenated per-point sample amplitudes [ADC]
    waveform_offsets : (N + 1,) int64 — point i's samples are
        waveform_times[waveform_offsets[i]:waveform_offsets[i + 1]]
    pulse_id : (N,) int64 or None — shared-pulse grouping for echo-rank features.
        If None, stages that need it derive it by hashing each point's raw
        waveform samples (points sharing a pulse have identical waveform rows).
    """

    xyz: np.ndarray
    reflectance_db: np.ndarray
    waveform_times: np.ndarray
    waveform_amps: np.ndarray
    waveform_offsets: np.ndarray
    pulse_id: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = len(self.xyz)
        if self.xyz.shape != (n, 3):
            raise ValueError(f"xyz must be (N, 3), got {self.xyz.shape}")
        if len(self.reflectance_db) != n:
            raise ValueError("reflectance_db length must match xyz")
        if len(self.waveform_offsets) != n + 1:
            raise ValueError("waveform_offsets must have length N + 1")
        if self.pulse_id is not None and len(self.pulse_id) != n:
            raise ValueError("pulse_id length must match xyz")

    def __len__(self) -> int:
        return len(self.xyz)

    @property
    def x(self) -> np.ndarray:
        return self.xyz[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.xyz[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self.xyz[:, 2]

    def waveform(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (times, amps) for point i."""
        lo, hi = self.waveform_offsets[i], self.waveform_offsets[i + 1]
        return self.waveform_times[lo:hi], self.waveform_amps[lo:hi]

    def iter_waveforms(self):
        """Yield (times, amps) for every point, in order."""
        for i in range(len(self)):
            yield self.waveform(i)

    @classmethod
    def from_dataframe(
        cls,
        points: pd.DataFrame,
        waveforms: pd.DataFrame,
        *,
        x_col: str = "x",
        y_col: str = "y",
        z_col: str = "z",
        reflectance_col: str = "reflectance_dB",
        time_col: str = "Time [SI]",
        amp_col: str = "Amplitude [ADC]",
    ) -> "PointCloud":
        """Build a PointCloud from a points dataframe and a row-aligned waveform
        dataframe (one waveform row per point, same order).

        Waveform columns may already hold Python lists/arrays, or stringified
        arrays (e.g. numpy's ``repr()`` output — space-separated, possibly
        multi-line; integers are extracted by regex rather than parsed as a
        Python literal, since numpy's repr isn't valid list syntax).
        """
        if len(points) != len(waveforms):
            raise ValueError(
                f"points ({len(points)}) and waveforms ({len(waveforms)}) row counts differ"
            )
        xyz = points[[x_col, y_col, z_col]].to_numpy(dtype=np.float64)
        reflectance = points[reflectance_col].to_numpy(dtype=np.float32)

        times_list = _coerce_waveform_column(waveforms[time_col])
        amps_list = _coerce_waveform_column(waveforms[amp_col])

        lengths = np.fromiter((len(t) for t in times_list), dtype=np.int64, count=len(times_list))
        offsets = np.zeros(len(times_list) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])

        flat_times = np.concatenate(times_list).astype(np.int32) if lengths.sum() else np.array([], np.int32)
        flat_amps = np.concatenate(amps_list).astype(np.float32) if lengths.sum() else np.array([], np.float32)

        return cls(
            xyz=xyz,
            reflectance_db=reflectance,
            waveform_times=flat_times,
            waveform_amps=flat_amps,
            waveform_offsets=offsets,
        )


def _coerce_waveform_column(col: pd.Series) -> list[np.ndarray]:
    sample = col.iloc[0] if len(col) else None
    if isinstance(sample, str):
        return [np.array(_NUMBER_RE.findall(v), dtype=np.int64) for v in col]
    return [np.asarray(v) for v in col]


@dataclasses.dataclass
class PipelineState:
    """Threaded state passed between pipeline stages.

    Each stage reads what it needs and sets its own outputs; nothing is
    removed, so any intermediate artifact stays reachable after the run for
    inspection (``state.features``, ``state.boundary``, ...).
    """

    cloud: PointCloud

    # features stage
    features: pd.DataFrame | None = None
    waveform_grids: np.ndarray | None = None            # (N, grid_size) raw
    waveform_grids_norm: np.ndarray | None = None        # (N, grid_size) per-sample max-normalised

    # autolabel (v6) stage — training-bootstrap only, unused by classify()
    autolabel_xgb_proba: np.ndarray | None = None
    autolabel_deep_proba: np.ndarray | None = None
    autolabel_ensemble: np.ndarray | None = None

    # wcn stage
    wcn_proba: np.ndarray | None = None
    wcn_xgb_proba: np.ndarray | None = None

    # geometry (water-surface) stage
    footprint_geom: Any = None            # shapely (Multi)Polygon
    footprint_raw_hull: Any = None
    in_footprint: np.ndarray | None = None
    local_surface_z: np.ndarray | None = None
    surface_grid: np.ndarray | None = None
    surface_grid_origin: tuple[float, float] | None = None   # (x_min, y_min)
    surface_plane_coef: tuple[float, float, float] | None = None
    merged_label: np.ndarray | None = None
    reconstructed_label: np.ndarray | None = None

    # canopy stage
    canopy_proba: np.ndarray | None = None
    canopy_pred: np.ndarray | None = None

    # merge stage
    final_label: np.ndarray | None = None

    # boundary stage
    boundary_contours: dict[float, list[np.ndarray]] | None = None

    # free-form diagnostics / stage metrics, keyed by stage name
    metrics: dict[str, dict] = dataclasses.field(default_factory=dict)
