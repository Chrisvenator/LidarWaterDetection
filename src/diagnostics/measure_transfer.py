"""Re-measure within-site and cross-site waveform classification on honest labels.

    python src/diagnostics/measure_transfer.py            # every number, one command
    python src/diagnostics/measure_transfer.py --rebuild  # ignore the feature cache

Regenerates every figure in ``context/cross_site_transfer.md``. Labels:

* Inn — the hand-drawn regions (loaded by ``audit_inn_labels``), with the 65
  hand-picked points as a secondary, biased check.
* Pielach — the Mandlburger reference matches
  (``validation_results/tables/per_point_matches.csv``).
* Pipeline output appears only in rows whose label column says ``pipeline``.

Every score is XGBoost with fixed hyperparameters and seed, reported as AUC
and balanced accuracy at 0.5 with a 95% interval from resampling 10 m
plan-view blocks. Within a site the model is cross-validated over the same
blocks; across sites it is trained on one site's labelled points and tested
on the other's.

The format-or-physics ladder re-exports both surveys through one function
(keep only the samples around each echo), then matches the time axis, then
the recorded range gate, and re-measures transfer after each step.

Writes ``runs/transfer/transfer_results.json`` and prints the tables.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr
from sklearn.model_selection import GroupKFold, KFold
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import audit_inn_labels as inn_audit                                   # noqa: E402
from lidarwater import derive_site_config                               # noqa: E402
from lidarwater._stages import features as feature_stage                # noqa: E402
from lidarwater.config import (REFLECTANCE_FEATURE, WCN_SCALAR_FEATURES,  # noqa: E402
                               FeatureConfig)
from lidarwater.io import read_dataset_dir                              # noqa: E402
from lidarwater.types import PointCloud                                 # noqa: E402

PIELACH, INN = "pielach", "inn"
DATA_DIRS = {PIELACH: ROOT / "data" / "Pielach", INN: ROOT / "data" / "Inn_DeepLearning"}
PIELACH_MATCHES = ROOT / "validation_results" / "tables" / "per_point_matches.csv"
PIELACH_REGISTRATION = ROOT / "validation_results" / "reports" / "registration.json"
# Pipeline output per site: (file, xyz columns). Inn's is the fresh base-config
# fit the audit scored (scripts/run_dataset.py data/Inn_DeepLearning --fit).
PIPELINE_CLOUDS = {
    PIELACH: (ROOT / "pointclouds" / "labeled_pointcloud_final.csv", ["X", "Y", "Z"]),
    INN: (ROOT / "runs" / "Inn_audit_refit" / "pointclouds" / "labeled_pointcloud_final.csv",
          ["x", "y", "z"]),
}
OUT_DIR = ROOT / "runs" / "transfer"
CACHE_DIR = OUT_DIR / "cache"
PARAMS_JSON = CACHE_DIR / "harmonisation.json"

HONEST, PICKED, PIPELINE = "honest", "picked65 (biased)", "pipeline"
PIPELINE_WATER, PIPELINE_UNCERTAIN = (1, 3), 2
MAX_JOIN_DIST_M = 0.001        # Pielach matches are the cloud's own points, shifted

EXPORT_FORMAT_FEATURES = ("max_amp_norm_by_energy", "gap_ratio", "n_gaps", "n_clusters")
GRID = "grid200"
FEATURE_SETS = {
    "waveform11": WCN_SCALAR_FEATURES,
    "waveform7": tuple(f for f in WCN_SCALAR_FEATURES if f not in EXPORT_FORMAT_FEATURES),
    "reflectance": (REFLECTANCE_FEATURE,),
    GRID: (),
}
FEATURE_SETS["waveform7+reflectance"] = FEATURE_SETS["waveform7"] + (REFLECTANCE_FEATURE,)
FEATURE_SETS["waveform11+reflectance"] = WCN_SCALAR_FEATURES + (REFLECTANCE_FEATURE,)
SHIFT_FEATURES = WCN_SCALAR_FEATURES + (REFLECTANCE_FEATURE,)
RAW, RANK = "raw", "site-rank"

SEED = inn_audit.SEED
N_BOOTSTRAP = inn_audit.N_BOOTSTRAP
N_FOLDS = 5
CV_BLOCK_M = inn_audit.BLOCK_SIZE_M          # see block_size_sensitivity()
CV_BLOCK_SWEEP_M = (10.0, 20.0, 40.0)
DECISION_THRESHOLD = 0.5
AUC_SAMPLES = "auc_bootstrap"      # per-resample AUCs, kept in memory for paired contrasts
XGB_PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", n_jobs=16, random_state=SEED)

# Format-or-physics ladder. Each rung is (name, Pielach variant, Inn variant).
NATIVE, WINDOWED, RESCALED, GATED = "native", "echo_window", "time_rescaled", "matched_gate"
LADDER = (
    ("0 native export", NATIVE, NATIVE),
    ("1 + echo-window export", WINDOWED, WINDOWED),
    ("2 + time axis matched", WINDOWED, RESCALED),
    ("3 + range gate matched", GATED, RESCALED),
)
N_PROBE = 5000                 # records sampled to measure noise and echo width
NOISE_GUARD_SI = 20            # samples before the first return excluded from the noise probe
NOISE_CEILING_PERCENTILE = 99.0
ECHO_PEAK_FACTOR = 2.0         # an echo must peak this many noise ceilings above the floor
EXPORT_FLOOR_ADC = 26.0        # lowest amplitude either export stores


# --------------------------------------------------------------------------
# Re-exporting waveforms
# --------------------------------------------------------------------------

def run_ids(times: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Label runs of samples that are contiguous in time and agree on ``mask``."""
    breaks = (np.diff(times) > 1) | (mask[1:] != mask[:-1])
    return np.concatenate([[0], np.cumsum(breaks)])


def echo_window(times: np.ndarray, amps: np.ndarray, edge_adc: float,
                min_peak_adc: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the sample runs that hold an echo, on a common pedestal.

    This is what an SVB export looks like from the outside: each stored
    segment starts and ends where the return meets the noise.
    """
    level = amps - feature_stage._noise_floor(amps, FeatureConfig())
    above = level > edge_adc
    runs = run_ids(times, above)
    run_peak = np.zeros(runs[-1] + 1)
    np.maximum.at(run_peak, runs, np.where(above, level, 0.0))
    keep = above & (run_peak[runs] >= min(min_peak_adc, run_peak.max()))
    if not keep.any():
        return times, amps
    return times[keep], (level[keep] + EXPORT_FLOOR_ADC).astype(np.float32)


def decimate(times: np.ndarray, amps: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    """Average stored samples into bins ``factor`` samples wide."""
    bins, inverse = np.unique(times // factor, return_inverse=True)
    mean = np.bincount(inverse, weights=amps) / np.bincount(inverse)
    return bins.astype(np.int32), mean.astype(np.float32)


def truncate(times: np.ndarray, amps: np.ndarray, gate_si: int) -> tuple[np.ndarray, np.ndarray]:
    keep = times <= times[0] + gate_si
    return times[keep], amps[keep]


def map_waveforms(cloud: PointCloud, transform) -> PointCloud:
    """A copy of ``cloud`` with ``transform(times, amps)`` applied to each record."""
    records = [transform(t, a) if len(t) else (t, a) for t, a in cloud.iter_waveforms()]
    offsets = np.concatenate([[0], np.cumsum([len(t) for t, _ in records])])
    return PointCloud(xyz=cloud.xyz, reflectance_db=cloud.reflectance_db,
                      waveform_times=np.concatenate([t for t, _ in records]).astype(np.int32),
                      waveform_amps=np.concatenate([a for _, a in records]).astype(np.float32),
                      waveform_offsets=offsets.astype(np.int64))


def probe_records(cloud: PointCloud):
    """A fixed random sample of non-trivial records."""
    picks = np.random.default_rng(SEED).choice(len(cloud), N_PROBE, replace=False)
    return (rec for rec in map(cloud.waveform, picks) if len(rec[0]) > 2)


def noise_ceiling(cloud: PointCloud) -> float:
    """ADC above the record floor that pre-trigger noise stays under."""
    config, noise = FeatureConfig(), []
    for times, amps in probe_records(cloud):
        t_return = feature_stage._first_return_time(times, amps, config)
        level = amps - feature_stage._noise_floor(amps, config)
        noise.append(level[times < t_return - NOISE_GUARD_SI])
    return float(np.percentile(np.concatenate(noise), NOISE_CEILING_PERCENTILE))


def median_echo_width(cloud: PointCloud) -> float:
    """Median full width at half maximum of each record's strongest echo, in samples."""
    widths = []
    for times, amps in probe_records(cloud):
        level = amps - amps.min()
        runs = run_ids(times, level >= 0.5 * level.max())
        widths.append((runs == runs[np.argmax(level)]).sum())
    return float(np.median(widths))


@functools.lru_cache(maxsize=None)
def raw_cloud(site: str) -> PointCloud:
    print(f"  reading {DATA_DIRS[site]}", flush=True)
    return read_dataset_dir(DATA_DIRS[site])


@functools.lru_cache(maxsize=None)
def windowed_cloud(site: str, edge_adc: float, min_peak_adc: float) -> PointCloud:
    return map_waveforms(raw_cloud(site), lambda t, a: echo_window(t, a, edge_adc, min_peak_adc))


@functools.lru_cache(maxsize=None)
def harmonisation_params() -> dict:
    """Label-free measurements that fix the re-export; identical for both sites."""
    edge = noise_ceiling(raw_cloud(INN))
    widths = {site: median_echo_width(raw_cloud(site)) for site in (PIELACH, INN)}
    params = {"edge_adc": edge, "min_peak_adc": ECHO_PEAK_FACTOR * edge,
              "echo_fwhm_samples": widths,
              "time_factor": int(round(widths[INN] / widths[PIELACH]))}
    inn = variant_cloud(INN, RESCALED, params)
    first, last = inn.waveform_offsets[:-1], inn.waveform_offsets[1:] - 1
    params["gate_si"] = int(np.median(inn.waveform_times[last] - inn.waveform_times[first]))
    return params


def variant_cloud(site: str, variant: str, params: dict) -> PointCloud:
    cloud = windowed_cloud(site, params["edge_adc"], params["min_peak_adc"])
    if variant == RESCALED:
        return map_waveforms(cloud, lambda t, a: decimate(t, a, params["time_factor"]))
    if variant == GATED:
        return map_waveforms(cloud, lambda t, a: truncate(t, a, params["gate_si"]))
    return cloud


def build_variant(site: str, variant: str) -> tuple[PointCloud, FeatureConfig]:
    """Native keeps the production extraction; every re-export is read the way
    Pielach's SVB export is (grid anchored at the first stored sample, no gate)."""
    if variant == NATIVE:
        cloud = raw_cloud(site)
        return cloud, derive_site_config(cloud)[0].features
    params = harmonisation_params()
    PARAMS_JSON.write_text(json.dumps(params, indent=2))
    return variant_cloud(site, variant, params), FeatureConfig()


# --------------------------------------------------------------------------
# Sites and labels
# --------------------------------------------------------------------------

@dataclasses.dataclass(eq=False)
class Site:
    name: str
    variant: str
    table: pd.DataFrame        # x, y, z + the scalar features, one row per cloud point
    grids: np.ndarray          # (N, 200) per-sample max-normalised waveform grid

    @property
    def xyz(self) -> np.ndarray:
        return self.table[["x", "y", "z"]].to_numpy()


@dataclasses.dataclass(frozen=True, eq=False)
class LabelSet:
    name: str
    idx: np.ndarray            # cloud row of each labelled point
    is_water: np.ndarray
    xy: np.ndarray             # plan coordinates the spatial blocks are cut from

    def blocks(self, size_m: float = inn_audit.BLOCK_SIZE_M) -> np.ndarray:
        # block_ids cuts fixed 10 m cells; rescaling xy reuses it for other sizes
        return inn_audit.block_ids(self.xy * (inn_audit.BLOCK_SIZE_M / size_m))


def load_site(site: str, variant: str, rebuild: bool) -> Site:
    table_path = CACHE_DIR / f"{site}_{variant}.parquet"
    grid_path = CACHE_DIR / f"{site}_{variant}_grids.npy"
    if rebuild or not (table_path.exists() and grid_path.exists()):
        print(f"extracting features: {site} / {variant}", flush=True)
        cloud, config = build_variant(site, variant)
        state = feature_stage.run(cloud, config)
        table = state.features[list(SHIFT_FEATURES)].copy()
        table[["x", "y", "z"]] = cloud.xyz
        table.to_parquet(table_path)
        np.save(grid_path, state.waveform_grids_norm)
    return Site(site, variant, pd.read_parquet(table_path), np.load(grid_path))


def pielach_reference(xyz: np.ndarray) -> LabelSet:
    """Mandlburger matches joined back to the cloud; the join is checked, not assumed."""
    matches = pd.read_csv(PIELACH_MATCHES)
    offset = json.loads(PIELACH_REGISTRATION.read_text())
    utm = matches[["utm_x", "utm_y", "z"]].to_numpy()
    dist, idx = cKDTree(xyz).query(utm - [offset["dx"], offset["dy"], 0.0])
    if dist.max() > MAX_JOIN_DIST_M or np.unique(idx).size != len(idx):
        raise ValueError(f"Pielach reference join failed: max distance {dist.max():.4f} m, "
                         f"{np.unique(idx).size} unique rows for {len(idx)} matches")
    # blocks are cut in UTM, as audit_pielach_validation.py cuts them
    return LabelSet(HONEST, idx, matches.ref_water.to_numpy() == 1, utm[:, :2])


def inn_references(xyz: np.ndarray) -> dict[str, LabelSet]:
    tree = cKDTree(xyz)
    regions = inn_audit.region_reference(inn_audit.load_regions(tree))
    picked = inn_audit.load_blind(tree)
    return {name: LabelSet(name, ref.idx.to_numpy(), ref.is_water.to_numpy(),
                           xyz[ref.idx.to_numpy(), :2])
            for name, ref in ((HONEST, regions), (PICKED, picked))}


def pipeline_labels(site: str, xyz: np.ndarray) -> LabelSet | None:
    """The pipeline's own decided labels over the whole cloud (secondary only)."""
    path, columns = PIPELINE_CLOUDS[site]
    if not path.exists():
        print(f"  no pipeline output at {path}: pipeline-label rows skipped for {site}")
        return None
    cloud = pd.read_csv(path)
    if not np.allclose(cloud[columns].to_numpy(), xyz, atol=MAX_JOIN_DIST_M):
        raise ValueError(f"{path} is not row-aligned with the raw cloud")
    idx = np.flatnonzero(cloud.final_label.to_numpy() != PIPELINE_UNCERTAIN)
    return LabelSet(PIPELINE, idx, np.isin(cloud.final_label.to_numpy()[idx], PIPELINE_WATER),
                    xyz[idx, :2])


def load_labels(site: Site) -> dict[str, LabelSet]:
    labels = ({HONEST: pielach_reference(site.xyz)} if site.name == PIELACH
              else inn_references(site.xyz))
    pipeline = pipeline_labels(site.name, site.xyz)
    return labels if pipeline is None else {**labels, PIPELINE: pipeline}


# --------------------------------------------------------------------------
# Scoring: block-bootstrap AUC, balanced accuracy, feature shift
# --------------------------------------------------------------------------

def block_multiplicities(n_blocks: int, rng: np.random.Generator) -> np.ndarray:
    """(N_BOOTSTRAP, n_blocks) times each block is drawn — the same block
    resampling ``audit_inn_labels.bootstrap_interval`` does, kept as counts."""
    draws = rng.integers(0, n_blocks, size=(N_BOOTSTRAP, n_blocks))
    counts = np.zeros((N_BOOTSTRAP, n_blocks))
    np.add.at(counts, (np.arange(N_BOOTSTRAP)[:, None], draws), 1.0)
    return counts


def interval(samples: np.ndarray) -> list[float]:
    return [float(v) for v in np.nanpercentile(samples, inn_audit.CI_PERCENTILES)]


def pairwise_wins(is_water: np.ndarray, score: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    """wins[a, b] = (water in block a, land in block b) pairs the score orders
    correctly, ties counted half. AUC under any block reweighting follows."""
    n_blocks = int(blocks.max()) + 1
    water_score, water_block = score[is_water], blocks[is_water]
    wins = np.zeros((n_blocks, n_blocks))
    for block in np.unique(blocks[~is_water]):
        land = np.sort(score[~is_water & (blocks == block)])
        below = 0.5 * (np.searchsorted(land, water_score, "left")
                       + np.searchsorted(land, water_score, "right"))
        wins[:, block] = np.bincount(water_block, weights=below, minlength=n_blocks)
    return wins


def bootstrap_auc(is_water: np.ndarray, score: np.ndarray, blocks: np.ndarray,
                  rng: np.random.Generator) -> tuple[float, np.ndarray]:
    """AUC and its value under each block resample."""
    n_blocks = int(blocks.max()) + 1
    wins = pairwise_wins(is_water, score, blocks)
    n_water = np.bincount(blocks[is_water], minlength=n_blocks)
    n_land = np.bincount(blocks[~is_water], minlength=n_blocks)
    counts = block_multiplicities(n_blocks, rng)
    with np.errstate(invalid="ignore", divide="ignore"):
        resampled = ((counts @ wins) * counts).sum(axis=1) / ((counts @ n_water) * (counts @ n_land))
    return float(wins.sum() / (n_water.sum() * n_land.sum())), resampled


def shift_with_interval(values: np.ndarray, is_water: np.ndarray, blocks: np.ndarray,
                        scale: float, rng: np.random.Generator) -> tuple[float, list[float]]:
    """Water-minus-land mean difference in units of ``scale``."""
    per_point = np.column_stack([is_water, is_water * values, ~is_water, ~is_water * values])
    per_block = np.zeros((int(blocks.max()) + 1, 4))
    np.add.at(per_block, blocks, per_point)

    def shift(tallies: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return (tallies[..., 1] / tallies[..., 0] - tallies[..., 3] / tallies[..., 2]) / scale

    resampled = shift(block_multiplicities(len(per_block), rng) @ per_block)
    return float(shift(per_block.sum(axis=0))), interval(resampled)


def score_predictions(labels: LabelSet, proba: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    is_water, blocks = labels.is_water, labels.blocks()
    auc, auc_samples = bootstrap_auc(is_water, proba, blocks, rng)
    correct = (proba > DECISION_THRESHOLD) == is_water
    return {
        "n": int(len(is_water)), "n_water": int(is_water.sum()),
        "n_land": int((~is_water).sum()), "n_blocks": int(blocks.max()) + 1,
        "auc": auc, "auc_ci95": interval(auc_samples), AUC_SAMPLES: auc_samples,
        "balanced_accuracy": float(0.5 * (correct[is_water].mean() + correct[~is_water].mean())),
        "balanced_accuracy_ci95": inn_audit.bootstrap_interval(is_water, correct, blocks, rng),
    }


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def design(site: Site, feature_set: str, encoding: str) -> np.ndarray:
    """Feature matrix over the whole cloud. Site-rank replaces each column by
    its percentile rank within its own cloud — label-free."""
    if feature_set == GRID:
        return site.grids
    x = site.table[list(FEATURE_SETS[feature_set])].to_numpy(dtype=np.float32)
    return (rankdata(x, axis=0) / len(x)).astype(np.float32) if encoding == RANK else x


def fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    return XGBClassifier(**XGB_PARAMS).fit(x_train, y_train).predict_proba(x_test)[:, 1]


def cross_validated(x: np.ndarray, y: np.ndarray, groups: np.ndarray | None) -> np.ndarray:
    """Out-of-fold P(water). ``groups`` holds whole blocks out; None = random points."""
    splitter = (KFold(N_FOLDS, shuffle=True, random_state=SEED).split(x) if groups is None
                else GroupKFold(N_FOLDS).split(x, y, groups))
    proba = np.zeros(len(y))
    for train, test in splitter:
        proba[test] = fit_predict(x[train], y[train], x[test])
    return proba


def within_site(site: Site, labels: LabelSet, feature_set: str,
                block_m: float | None = CV_BLOCK_M) -> dict:
    x = design(site, feature_set, RAW)[labels.idx]
    groups = None if block_m is None else labels.blocks(block_m)
    proba = cross_validated(x, labels.is_water, groups)
    return {"train": site.name, "test": site.name, "labels": labels.name,
            "features": feature_set, "encoding": RAW,
            "cv": "random points" if block_m is None else f"{block_m:g} m blocks",
            **score_predictions(labels, proba)}


def transfer(train: Site, train_labels: LabelSet, test: Site, test_sets: list[LabelSet],
             feature_set: str, encoding: str) -> list[dict]:
    """Fit on one site's labelled points, score on each of the other's label sets."""
    x_test = design(test, feature_set, encoding)
    proba = fit_predict(design(train, feature_set, encoding)[train_labels.idx],
                        train_labels.is_water, x_test)
    return [{"train": train.name, "test": test.name, "labels": labels.name,
             "train_labels": train_labels.name, "features": feature_set, "encoding": encoding,
             **score_predictions(labels, proba[labels.idx])} for labels in test_sets]


def picked_within_site(site: Site, labels: dict[str, LabelSet], feature_set: str) -> dict:
    """Inn's 65 picked points, scored by a model that saw no region point from
    any 10 m block holding one of them."""
    regions, picked = labels[HONEST], labels[PICKED]
    cells = np.floor(regions.xy / CV_BLOCK_M).astype(np.int64)
    picked_cells = np.unique(np.floor(picked.xy / CV_BLOCK_M).astype(np.int64), axis=0)
    apart = ~(cells[:, None, :] == picked_cells[None, :, :]).all(axis=2).any(axis=1)
    x = design(site, feature_set, RAW)
    proba = fit_predict(x[regions.idx[apart]], regions.is_water[apart], x[picked.idx])
    return {"train": site.name, "test": site.name, "labels": PICKED, "features": feature_set,
            "encoding": RAW, "cv": "trained on regions outside the picked points' blocks",
            **score_predictions(picked, proba)}


# --------------------------------------------------------------------------
# The five questions
# --------------------------------------------------------------------------

Sites = dict[str, Site]
Labels = dict[str, dict[str, LabelSet]]
DIRECTIONS = ((PIELACH, INN), (INN, PIELACH))


def q1_within_site(sites: Sites, labels: Labels) -> list[dict]:
    rows = []
    for name, site in sites.items():
        for feature_set in ("waveform11", GRID):
            rows += [within_site(site, labels[name][kind], feature_set)
                     for kind in (HONEST, PIPELINE) if kind in labels[name]]
            if PICKED in labels[name]:
                rows.append(picked_within_site(site, labels[name], feature_set))
    return rows


def block_size_sensitivity(sites: Sites, labels: Labels) -> list[dict]:
    """Does holding out larger blocks lower the score? If not, 10 m is enough."""
    rows = []
    for name, site in sites.items():
        ref = labels[name][HONEST]
        # a size leaving fewer blocks than folds cannot be cross-validated
        sizes = [m for m in CV_BLOCK_SWEEP_M if ref.blocks(m).max() + 1 >= N_FOLDS]
        rows += [within_site(site, ref, "waveform11", block_m) for block_m in (None, *sizes)]
    return rows


def transfer_grid(sites: Sites, labels: Labels, specs: tuple[tuple[str, str], ...],
                  label_kinds: tuple[str, ...] = (HONEST, PIPELINE)) -> list[dict]:
    """Every (feature set, encoding) in ``specs``, both directions. Honest-trained
    models are also scored on the 65 picked points; pipeline-trained ones are
    scored on pipeline labels only."""
    rows = []
    for (source, target), (feature_set, encoding), kind in (
            (d, s, k) for d in DIRECTIONS for s in specs for k in label_kinds):
        if kind not in labels[source] or kind not in labels[target]:
            continue
        test_sets = [labels[target][kind]]
        if kind == HONEST and PICKED in labels[target]:
            test_sets.append(labels[target][PICKED])
        rows += transfer(sites[source], labels[source][kind], sites[target], test_sets,
                         feature_set, encoding)
    return rows


Q2_SPECS = (("waveform11", RAW), ("waveform11", RANK), ("waveform7", RAW), ("waveform7", RANK),
            (GRID, RAW))
Q3_SPECS = (("reflectance", RANK), ("waveform7+reflectance", RANK),
            ("waveform11+reflectance", RANK), ("reflectance", RAW))
LADDER_SPECS = (("waveform11", RAW), ("waveform11", RANK), (GRID, RAW))


def q4_ladder(native: Sites, labels: Labels, rebuild: bool) -> tuple[list[dict], list[dict]]:
    """Transfer, within-site score and feature shifts after each harmonisation step."""
    scores, shifts = [], []
    for rung, pielach_variant, inn_variant in LADDER:
        sites = {PIELACH: native[PIELACH] if pielach_variant == NATIVE
                 else load_site(PIELACH, pielach_variant, rebuild),
                 INN: native[INN] if inn_variant == NATIVE else load_site(INN, inn_variant, rebuild)}
        rung_rows = transfer_grid(sites, labels, LADDER_SPECS, (HONEST,))
        rung_rows += [within_site(site, labels[name][HONEST], feature_set)
                      for name, site in sites.items() for feature_set in ("waveform11", GRID)]
        scores += [{"rung": rung, **row} for row in rung_rows]
        shifts += [{"rung": rung, **row} for row in feature_shifts(sites, labels, (HONEST,))]
    return scores, shifts


def pick(rows: list[dict], **match) -> dict:
    (row,) = [r for r in rows if all(r.get(k) == v for k, v in match.items())]
    return row


def paired_delta(contrast: str, a: dict, b: dict) -> dict:
    """AUC(a) - AUC(b) on the same labelled points. Both rows were resampled
    with the same seeded block draws, so the interval is a paired one."""
    return {"contrast": contrast, "test": a["test"], "labels": a["labels"], "n": a["n"],
            "n_blocks": a["n_blocks"], "auc_a": a["auc"], "auc_b": b["auc"],
            "delta_auc": a["auc"] - b["auc"],
            "delta_auc_ci95": interval(a[AUC_SAMPLES] - b[AUC_SAMPLES])}


def transfer_gaps(within: list[dict], transfers: list[dict]) -> list[dict]:
    """Within-site ceiling minus each honest transfer score, on the test site."""
    cv = f"{CV_BLOCK_M:g} m blocks"
    rows = []
    for row in (r for r in transfers if r["labels"] == HONEST):
        ceiling = pick(within, test=row["test"], labels=HONEST, cv=cv,
                       features=GRID if row["features"] == GRID else "waveform11")
        name = (f"within-{row['test']} {ceiling['features']} minus "
                f"{row['train']}->{row['test']} {row['features']} {row['encoding']}")
        rows.append(paired_delta(name, ceiling, row))
    return rows


def reflectance_contrasts(rows: list[dict]) -> list[dict]:
    """Does adding waveform features to reflectance rank lower the score?"""
    out = []
    for base in (r for r in rows if r["features"] == "reflectance" and r["encoding"] == RANK
                 and r["labels"] != PIPELINE):
        same = {k: base[k] for k in ("train", "test", "labels", "encoding")}
        out += [paired_delta(f"{base['train']}->{base['test']} reflectance rank minus {added}",
                             base, pick(rows, features=added, **same))
                for added in ("waveform7+reflectance", "waveform11+reflectance")]
    return out


def ladder_recovery(scores: list[dict]) -> list[dict]:
    """Each rung's honest transfer score minus the native-export one."""
    transfers = [r for r in scores if r["train"] != r["test"] and r["labels"] == HONEST]
    out = []
    for row in (r for r in transfers if r["rung"] != LADDER[0][0]):
        same = {k: row[k] for k in ("train", "test", "features", "encoding")}
        name = f"{row['rung']}: {row['train']}->{row['test']} {row['features']} {row['encoding']}"
        out.append(paired_delta(name, row, pick(transfers, rung=LADDER[0][0], **same)))
    return out


def feature_shifts(sites: Sites, labels: Labels, label_kinds: tuple[str, ...]) -> list[dict]:
    """Per feature and site: class means, water-minus-land shift in whole-cloud
    standard deviations, and the feature's own AUC (above 0.5 = higher on water)."""
    rows = []
    for (name, site), kind in ((s, k) for s in sites.items() for k in label_kinds):
        if kind not in labels[name]:
            continue
        ref = labels[name][kind]
        columns = {f: site.table[f].to_numpy(dtype=float) for f in SHIFT_FEATURES}
        columns[f"{REFLECTANCE_FEATURE} ({RANK})"] = design(site, "reflectance", RANK)[:, 0]
        rows += [shift_row(name, kind, feature, column, ref) for feature, column in columns.items()]
    return rows


def shift_row(site: str, kind: str, feature: str, column: np.ndarray, ref: LabelSet) -> dict:
    rng = np.random.default_rng(SEED)
    values, blocks = column[ref.idx], ref.blocks()
    shift, shift_ci = shift_with_interval(values, ref.is_water, blocks, column.std(), rng)
    auc, auc_samples = bootstrap_auc(ref.is_water, values, blocks, rng)
    return {"site": site, "labels": kind, "feature": feature, "n": int(len(values)),
            "n_blocks": int(blocks.max()) + 1,
            "mean_water": float(values[ref.is_water].mean()),
            "mean_land": float(values[~ref.is_water].mean()),
            "shift_sd": shift, "shift_sd_ci95": shift_ci, "auc": auc, "auc_ci95": interval(auc_samples)}


def shift_agreement(shifts: list[dict]) -> list[dict]:
    """Per label kind and rung: do the two sites' waveform-feature shifts line up?"""
    frame = pd.DataFrame(shifts)
    frame = frame[frame.feature.isin(WCN_SCALAR_FEATURES)]
    rows = []
    by = [c for c in ("rung", "labels") if c in frame]
    for keys, part in frame.groupby(by, sort=False):
        wide = part.pivot(index="feature", columns="site", values="shift_sd")
        same_sign = np.sign(wide[PIELACH]) == np.sign(wide[INN])
        rows.append({**dict(zip(by, keys)),
                     "features_same_sign": f"{int(same_sign.sum())}/{len(wide)}",
                     "spearman_of_shifts": float(spearmanr(wide[PIELACH], wide[INN])[0])})
    return rows


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def with_ci(frame: pd.DataFrame, column: str, digits: int = 3) -> pd.Series:
    low, high = zip(*frame[f"{column}_ci95"])
    return (frame[column].round(digits).astype(str) + " ["
            + pd.Series(low, index=frame.index).round(digits).astype(str) + ", "
            + pd.Series(high, index=frame.index).round(digits).astype(str) + "]")


def print_scores(title: str, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    frame["AUC [95%]"] = with_ci(frame, "auc")
    frame["bal.acc [95%]"] = with_ci(frame, "balanced_accuracy")
    lead = [c for c in ("rung", "train", "test", "labels", "features", "encoding", "cv") if c in frame]
    print(f"\n=== {title} ===")
    print(frame[lead + ["n", "n_blocks", "AUC [95%]", "bal.acc [95%]"]].to_string(index=False))


def print_contrasts(title: str, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    frame["delta AUC [95%]"] = with_ci(frame, "delta_auc")
    print(f"\n=== {title} ===")
    print(frame[["contrast", "labels", "n", "n_blocks", "auc_a", "auc_b", "delta AUC [95%]"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def print_shifts(title: str, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    frame["shift sd [95%]"] = with_ci(frame, "shift_sd", 2)
    frame["own AUC [95%]"] = with_ci(frame, "auc")
    lead = [c for c in ("rung", "site", "labels", "feature") if c in frame]
    print(f"\n=== {title} ===")
    print(frame[lead + ["n", "mean_water", "mean_land", "shift sd [95%]", "own AUC [95%]"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def print_report(report: dict[str, list[dict]]) -> dict[str, list[dict]]:
    for key, rows in report.items():
        if "agreement" in key:
            print(f"\n=== {key} ===\n{pd.DataFrame(rows).to_string(index=False)}")
        elif "contrasts" in key:
            print_contrasts(key, rows)
        elif "shifts" in key:
            print_shifts(key, rows)
        else:
            print_scores(key, rows)
    return report


def metadata(labels: Labels) -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return {
        "git_commit": commit.stdout.strip(), "seed": SEED, "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_block_m": inn_audit.BLOCK_SIZE_M, "cv_block_m": CV_BLOCK_M, "n_folds": N_FOLDS,
        "xgb_params": XGB_PARAMS, "export_format_features": EXPORT_FORMAT_FEATURES,
        "harmonisation": json.loads(PARAMS_JSON.read_text()) if PARAMS_JSON.exists() else None,
        "label_sets": {site: {kind: {"n": int(len(ref.idx)), "n_water": int(ref.is_water.sum()),
                                     "n_blocks_10m": int(ref.blocks().max()) + 1}
                              for kind, ref in kinds.items()} for site, kinds in labels.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="recompute cached features")
    args = parser.parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    sites = {name: load_site(name, NATIVE, args.rebuild) for name in (PIELACH, INN)}
    labels = {name: load_labels(site) for name, site in sites.items()}
    ladder_scores, ladder_shifts = q4_ladder(sites, labels, args.rebuild)
    native_shifts = feature_shifts(sites, labels, (HONEST, PIPELINE))
    within = q1_within_site(sites, labels)
    waveform = transfer_grid(sites, labels, Q2_SPECS)
    reflectance = transfer_grid(sites, labels, Q3_SPECS)
    report = {
        "q1_within_site": within,
        "q1_block_size_sensitivity": block_size_sensitivity(sites, labels),
        "q2_waveform_transfer": waveform,
        "q2_gap_to_within_site_contrasts": transfer_gaps(within, waveform),
        "q3_reflectance_transfer": reflectance,
        "q3_waveform_added_contrasts": reflectance_contrasts(reflectance),
        "q4_ladder_scores": ladder_scores,
        "q4_ladder_recovery_contrasts": ladder_recovery(ladder_scores),
        "q4_ladder_shift_agreement": shift_agreement(ladder_shifts),
        "q4_ladder_shifts": ladder_shifts,
        "q5_feature_shifts": native_shifts,
        "q5_shift_agreement": shift_agreement(native_shifts),
    }
    report = {key: [{k: v for k, v in row.items() if k != AUC_SAMPLES} for row in rows]
              for key, rows in print_report(report).items()}
    out = OUT_DIR / "transfer_results.json"
    out.write_text(json.dumps({"meta": metadata(labels), **report}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
