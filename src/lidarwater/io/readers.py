"""Input adapters. ``PointCloud.from_dataframe`` is the primary constructor;
``read_waveform_txt`` wraps the two-file ASCII export this project's surveys
ship as, and ``read_dataset_dir`` finds that pair inside a survey folder
whatever the files happen to be named."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..types import PointCloud

_CHUNK_SIZE = 10_000
_NUMBER_RE = re.compile(r"[-+]?\d+")


def read_waveform_txt(point_cloud_path: str | Path, waveform_path: str | Path) -> PointCloud:
    """Read a ``point_cloud_df.txt`` / ``waveform_df.txt`` pair.

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


_POINT_CLOUD_GLOB = "*point_cloud*.txt"
_WAVEFORM_GLOB = "*wave*form*.txt"


def read_dataset_dir(directory: str | Path) -> PointCloud:
    """Read a survey folder holding one point-cloud and one waveform file.

    Filenames vary between surveys (``point_cloud_df.txt`` vs
    ``point_cloud_df_inn.txt``), so the pair is located by pattern.
    """
    directory = Path(directory)
    points = sorted(directory.glob(_POINT_CLOUD_GLOB))
    waveforms = [p for p in sorted(directory.glob(_WAVEFORM_GLOB)) if p not in points]
    for label, found in (("point cloud", points), ("waveform", waveforms)):
        if len(found) != 1:
            raise FileNotFoundError(
                f"expected exactly one {label} file in {directory}, found {len(found)}: "
                f"{[p.name for p in found]}"
            )
    return read_waveform_txt(points[0], waveforms[0])


# Original name, kept so existing Pielach code keeps working.
read_pielach_txt = read_waveform_txt
