"""Canopy stage: echo-rank / DTM-DSM / local-structure features feeding an
XGBoost canopy classifier. Runs after the geometry stage — the DTM reference
must be raised to the local water surface inside the footprint, or water
depth masquerades as vegetation height. Ports canopy_features.py +
train_canopy.py.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import ndimage
from scipy.spatial import cKDTree
from sklearn.metrics import f1_score, roc_auc_score

from ..artifacts import ArtifactId, ArtifactResolver
from ..config import CanopyConfig
from ..types import PipelineState

LABEL_LAND, LABEL_WATER, LABEL_UNCERTAIN = 0, 1, 2
_STRUCTURE_QUERY_CHUNK = 50_000

# Feature columns carried over from the features stage. Order matters: it
# must match the deployed canopy_xgb.json's trained feature order exactly
# (XGBoost validates by name+position, not name alone), which is the
# original features_current.csv's on-disk column order — not an
# alphabetical or logically-grouped order.
_CARRIED_FEATURE_COLS = [
    "reflectance_dB", "time_span", "n_gaps", "n_peaks", "first_last_span",
    "n_clusters", "depth_proxy_m", "planarity", "roughness", "linearity",
    "sphericity", "height_range_local", "height_std_local", "height_percentile_local",
    "energy_concentration", "amplitude_weighted_center", "active_bins_ratio",
    "max_amp_norm_by_energy",
]
_NON_FEATURE_COLS = {"x", "y", "z", "height_above_dtm"}
_XGB_PARAMS = dict(
    n_estimators=500, max_depth=7, learning_rate=0.06, subsample=0.8,
    colsample_bytree=0.8, tree_method="hist", eval_metric="logloss", n_jobs=-1,
)


def _compute_pulse_id(state: PipelineState) -> np.ndarray:
    """Hash each point's raw waveform samples — points sharing a pulse have
    identical waveforms. blake2b for stability across processes."""
    if state.cloud.pulse_id is not None:
        return state.cloud.pulse_id
    cloud = state.cloud
    pulse_id = np.empty(len(cloud), dtype=np.int64)
    seen: dict[bytes, int] = {}
    for i, (times, amps) in enumerate(cloud.iter_waveforms()):
        h = hashlib.blake2b(times.tobytes() + amps.tobytes(), digest_size=16).digest()
        pulse_id[i] = seen.setdefault(h, len(seen))
    return pulse_id


def _echo_features(xyz: np.ndarray, pulse_id: np.ndarray) -> pd.DataFrame:
    """Per-point echo rank within its pulse (rank 1 = highest z = first return)."""
    df = pd.DataFrame({"z": xyz[:, 2], "pulse_id": pulse_id})
    n_echoes = df.groupby("pulse_id")["z"].transform("size")
    rank = df.sort_values("z", ascending=False).groupby("pulse_id").cumcount() + 1
    rank = rank.reindex(df.index)

    return pd.DataFrame({
        "n_echoes": n_echoes,
        "echo_rank": rank,
        "echo_rank_norm": np.where(n_echoes > 1, (rank - 1) / (n_echoes - 1), 0.0),
        "is_single_echo": (n_echoes == 1).astype(int),
        "is_last_echo": ((rank == n_echoes) & (n_echoes > 1)).astype(int),
        "is_first_multi": ((rank == 1) & (n_echoes > 1)).astype(int),
        "is_intermediate": ((rank > 1) & (rank < n_echoes)).astype(int),
    })


def _fill_holes(grid: np.ndarray) -> np.ndarray:
    mask = np.isnan(grid)
    if mask.all():
        raise ValueError("grid has no valid cells")
    idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    return grid[tuple(idx)]


def _cell_aggregate(ix, iy, z, shape, percentile: float | None) -> np.ndarray:
    cell = ix * shape[1] + iy
    grouped = pd.Series(z).groupby(cell)
    agg = grouped.max() if percentile is None else grouped.quantile(percentile)
    grid = np.full(shape[0] * shape[1], np.nan)
    grid[agg.index.to_numpy()] = agg.to_numpy()
    return grid.reshape(shape)


def _build_dtm_dsm(xyz: np.ndarray, echo: pd.DataFrame, water_surface: np.ndarray,
                   config: CanopyConfig) -> pd.DataFrame:
    """1 m DTM (last/single echoes, robust low percentile) and DSM (all
    points, max). DTM is raised to the local water surface so submerged
    depth doesn't read as vegetation height."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    x0, y0 = x.min(), y.min()
    ix = ((x - x0) / config.cell_m).astype(int)
    iy = ((y - y0) / config.cell_m).astype(int)
    shape = (ix.max() + 1, iy.max() + 1)

    ground = (echo["is_single_echo"] | echo["is_last_echo"]).to_numpy(bool)
    dtm = _cell_aggregate(ix[ground], iy[ground], z[ground], shape, config.dtm_percentile)
    dtm = ndimage.median_filter(_fill_holes(dtm), size=3)
    dtm = ndimage.gaussian_filter(dtm, sigma=1.0)

    dsm = _fill_holes(_cell_aggregate(ix, iy, z, shape, None))
    dtm = np.minimum(dtm, dsm)

    return pd.DataFrame({
        "height_above_dtm": z - dtm[ix, iy],
        "depth_below_dsm": dsm[ix, iy] - z,
        "height_above_ref": z - np.maximum(dtm[ix, iy], water_surface),
    })


def _structure_features(xyz: np.ndarray, config: CanopyConfig) -> pd.DataFrame:
    """Cylinder/sphere neighborhood stats: echo ratio, overhead cover, z spread."""
    xy, z = xyz[:, :2], xyz[:, 2]
    tree2d, tree3d = cKDTree(xy), cKDTree(xyz)
    n_3d = tree3d.query_ball_point(xyz, config.r_local_m, workers=-1, return_length=True)

    n = len(xyz)
    n_2d = np.zeros(n, dtype=int)
    n_above = np.zeros(n)
    z_range = np.zeros(n)
    z_std = np.zeros(n)
    for start in range(0, n, _STRUCTURE_QUERY_CHUNK):
        sl = slice(start, min(start + _STRUCTURE_QUERY_CHUNK, n))
        neigh = tree2d.query_ball_point(xy[sl], config.r_local_m, workers=-1)
        counts = np.fromiter((len(a) for a in neigh), int, sl.stop - start)
        n_2d[sl] = counts
        flat = np.concatenate(neigh).astype(int)
        owner = np.repeat(np.arange(start, sl.stop), counts)
        zn = z[flat]
        n_above[sl] = np.bincount(owner - start, zn > z[owner] + config.above_gap_m,
                                  minlength=sl.stop - start)
        zsum = np.bincount(owner - start, zn, minlength=sl.stop - start)
        zsq = np.bincount(owner - start, zn ** 2, minlength=sl.stop - start)
        zmax = np.full(sl.stop - start, -np.inf)
        zmin = np.full(sl.stop - start, np.inf)
        np.maximum.at(zmax, owner - start, zn)
        np.minimum.at(zmin, owner - start, zn)
        z_range[sl] = zmax - zmin
        mean = zsum / counts
        z_std[sl] = np.sqrt(np.maximum(zsq / counts - mean ** 2, 0.0))

    return pd.DataFrame({
        "n_cyl": n_2d, "echo_ratio": n_3d / np.maximum(n_2d, 1),
        "n_above_2m": n_above, "z_range_cyl": z_range, "z_std_cyl": z_std,
    })


def build_canopy_features(state: PipelineState, config: CanopyConfig) -> pd.DataFrame:
    """Assemble the canopy model's feature matrix from the point cloud +
    the features stage's carried columns + the geometry stage's local
    water surface."""
    if state.features is None:
        raise ValueError("canopy stage needs state.features — run the features stage first")
    if state.local_surface_z is None:
        raise ValueError("canopy stage needs state.local_surface_z — run the geometry stage first")

    xyz = state.cloud.xyz
    pulse_id = _compute_pulse_id(state)
    echo = _echo_features(xyz, pulse_id)
    grids = _build_dtm_dsm(xyz, echo, state.local_surface_z, config)
    struct = _structure_features(xyz, config)
    carried = state.features[_CARRIED_FEATURE_COLS].reset_index(drop=True)

    return pd.concat([
        pd.DataFrame({"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}),
        echo.reset_index(drop=True), grids.reset_index(drop=True),
        struct.reset_index(drop=True), carried,
    ], axis=1)


def _open_sky_low_mask(df: pd.DataFrame, config: CanopyConfig) -> np.ndarray:
    return ((df["n_above_2m"] <= config.open_sky_max_above)
            & (df["height_above_ref"] < config.low_height_max_m)).to_numpy()


def _make_labels(df: pd.DataFrame, state: PipelineState, config: CanopyConfig) -> np.ndarray:
    """-1 unlabeled, 0 not-canopy, 1 canopy, 2/3 pseudo-negatives."""
    z = df["z"].to_numpy()
    labels = np.full(len(z), -1, dtype=int)
    labels[z > config.z_canopy_min] = 1
    labels[z <= config.z_clear_max] = 0

    z_above_surface = df["z"].to_numpy() - state.local_surface_z
    at_surface = ((state.merged_label == LABEL_WATER) & (state.in_footprint)
                  & (np.abs(z_above_surface) < config.surface_tol_m))
    labels[(labels == -1) & at_surface] = 2

    open_sky_low = _open_sky_low_mask(df, config)
    labels[(labels == -1) & open_sky_low] = 3
    labels[(labels == 1) & open_sky_low] = -1   # canopy-band label noise — exclude
    return labels


def _xgb_input(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Single-dtype (float32) DataFrame with feat_cols order guaranteed.

    XGBoost's inplace_predict validates feature order against pandas'
    internal block layout, not ``.columns`` — a DataFrame mixing int64 and
    float32/float64 columns gets silently reordered by dtype grouping,
    which then mismatches the order the model was trained/saved with.
    """
    values = np.nan_to_num(df[feat_cols].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame(values, columns=feat_cols)


def predict(state: PipelineState, config: CanopyConfig, resolver: ArtifactResolver) -> PipelineState:
    df = build_canopy_features(state, config)
    feat_cols = [c for c in df.columns if c not in _NON_FEATURE_COLS]

    model = xgb.XGBClassifier()
    model.load_model(resolver.resolve(ArtifactId.CANOPY_XGB))
    proba = model.predict_proba(_xgb_input(df, feat_cols))[:, 1].astype(np.float32)
    pred = (proba > config.threshold).astype(np.int8)

    state.canopy_proba = proba
    state.canopy_pred = pred
    state.metrics["canopy"] = {"canopy_fraction": float(pred.mean())}
    return state


def fit(state: PipelineState, config: CanopyConfig, resolver: ArtifactResolver) -> PipelineState:
    if state.merged_label is None or state.in_footprint is None:
        raise ValueError("canopy fit needs state.merged_label/in_footprint — run geometry first")

    df = build_canopy_features(state, config)
    labels = _make_labels(df, state, config)
    feat_cols = [c for c in df.columns if c not in _NON_FEATURE_COLS]
    x, y = _xgb_input(df, feat_cols), (labels == 1).astype(int)
    labeled = labels >= 0

    pos_w = (y[labeled] == 0).sum() / (y[labeled] == 1).sum()
    model = xgb.XGBClassifier(**_XGB_PARAMS, scale_pos_weight=pos_w)
    model.fit(x[labeled], y[labeled])
    model.save_model(resolver.resolve_for_write(ArtifactId.CANOPY_XGB))

    proba = model.predict_proba(x)[:, 1].astype(np.float32)
    pred = (proba > config.threshold).astype(np.int8)
    state.canopy_proba = proba
    state.canopy_pred = pred
    state.metrics["canopy"] = {
        "n_labeled": int(labeled.sum()),
        "n_canopy_label": int((labels == 1).sum()),
        "n_clear_label": int((labels == 0).sum()),
        "canopy_fraction": float(pred.mean()),
    }
    return state


def merge(state: PipelineState) -> PipelineState:
    """Refine v10 land/uncertain with canopy predictions (final_label 4).
    Water labels (1, 3) win conflicts — the water model is authoritative at
    the water surface."""
    if state.reconstructed_label is None or state.canopy_pred is None:
        raise ValueError("merge needs state.reconstructed_label and state.canopy_pred")

    final = state.reconstructed_label.copy()
    is_canopy = state.canopy_pred == 1
    overridable = np.isin(final, (LABEL_LAND, LABEL_UNCERTAIN))
    final[overridable & is_canopy] = 4

    state.final_label = final
    state.metrics["merge"] = {
        "land_to_canopy": int(((state.reconstructed_label == LABEL_LAND) & is_canopy).sum()),
        "uncertain_to_canopy": int(((state.reconstructed_label == LABEL_UNCERTAIN) & is_canopy).sum()),
        "water_kept_despite_canopy_flag": int(
            (np.isin(state.reconstructed_label, (1, 3)) & is_canopy).sum()),
    }
    return state
