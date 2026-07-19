"""Input adapters. ``PointCloud.from_dataframe`` is the primary constructor;
``read_pielach_txt`` is a convenience wrapper around this project's original
two-file ASCII format, kept so the existing Pielach workflow keeps working."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..types import PointCloud

_CHUNK_SIZE = 10_000
_NUMBER_RE = re.compile(r"[-+]?\d+")


def read_pielach_txt(point_cloud_path: str | Path, waveform_path: str | Path) -> PointCloud:
    """Read the original ``point_cloud_df.txt`` / ``waveform_df.txt`` pair.

    Waveform columns hold numpy's ``repr()`` of each array (space-separated,
    possibly multi-line, not valid Python list syntax) — integers are
    extracted by regex. Parsed in chunks to bound memory on the ~92 MB
    waveform file.
    """
    points = pd.read_csv(point_cloud_path)
    points = points.rename(columns={"_riegl.reflectance": "reflectance_dB"})

    times_chunks: list[np.ndarray] = []
    amps_chunks: list[np.ndarray] = []
    offsets = [0]
    for chunk in pd.read_csv(waveform_path, chunksize=_CHUNK_SIZE):
        for t_str, a_str in zip(chunk["Time [SI]"], chunk["Amplitude [ADC]"]):
            t = np.array(_NUMBER_RE.findall(t_str), dtype=np.int32)
            a = np.array(_NUMBER_RE.findall(a_str), dtype=np.float32)
            times_chunks.append(t)
            amps_chunks.append(a)
            offsets.append(offsets[-1] + len(t))

    if len(points) != len(offsets) - 1:
        raise ValueError(
            f"point cloud rows ({len(points)}) and waveform rows ({len(offsets) - 1}) differ"
        )

    flat_times = np.concatenate(times_chunks) if times_chunks else np.array([], np.int32)
    flat_amps = np.concatenate(amps_chunks) if amps_chunks else np.array([], np.float32)

    return PointCloud(
        xyz=points[["x", "y", "z"]].to_numpy(dtype=np.float64),
        reflectance_db=points["reflectance_dB"].to_numpy(dtype=np.float32),
        waveform_times=flat_times,
        waveform_amps=flat_amps,
        waveform_offsets=np.asarray(offsets, dtype=np.int64),
    )
