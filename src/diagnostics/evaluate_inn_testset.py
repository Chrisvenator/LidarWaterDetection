"""Score the Inn pipeline once on the no-look labels.

    python src/diagnostics/evaluate_inn_testset.py --worker inn_unified_seed42_rep0
    python src/diagnostics/evaluate_inn_testset.py --worker inn_unified_seed43_rep0
    python src/diagnostics/evaluate_inn_testset.py          # score, after both fits

Implements ``context/inn_testset_protocol.md`` and nothing else: that file
fixes the labels, strata, methods, metric and the criterion that refutes the
claim "the pipeline beats an elevation rule where elevation is wrong", and it
was committed before this script ran. Do not edit either after a score exists.

A worker is one from-scratch fit of ``derive_site_config`` (sheet bootstrap,
all flags off) plus a flags-on ``classify()`` with the same models; it writes
``methods.npz`` and a ``DONE`` marker into ``runs/inn_testset/<fit>/``. Fits
take about two hours, so launch workers detached (``setsid nohup``) and wait
on the marker. Scoring writes ``testset_results.json`` and
``testset_tables.txt`` into ``runs/inn_testset/``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import ablate_models as ab                                              # noqa: E402
import audit_inn_labels as au                                           # noqa: E402
import measure_transfer as mt                                           # noqa: E402
from lidarwater import PipelineConfig, WaterPipeline, derive_site_config  # noqa: E402
from lidarwater._stages import autolabel                                # noqa: E402
from lidarwater._stages.cleanup import LABEL_CANOPY                     # noqa: E402
from lidarwater.artifacts import LocalArtifactResolver                  # noqa: E402
from lidarwater.config import BOOTSTRAP_SHEET                           # noqa: E402
from lidarwater.types import PipelineState, PointCloud                  # noqa: E402

OUT_DIR = ROOT / "runs" / "inn_testset"

NOLOOK_DIR = ROOT / "data" / "Inn_DeepLearning_Labels_nolook"
WATER_FILES = ("Wasser1", "Wasser2", "Wasser3", "safe-water", "geometry-water")
LAND_FILES = ("Land1", "Land2", "Landzunge", "Land_below_waterline", "safe-land")
# Saved after the task description was written and absent from its file
# table: scored on its own, never part of strata A-E (protocol, section 1).
LATE_WATER_FILE = "Wasser4"

FITS = (ab.FitSpec(mt.INN, ab.UNIFIED, ab.BASE_SEED, 0),
        ab.FitSpec(mt.INN, ab.UNIFIED, ab.OTHER_SEED, 0))
FIT_NAMES = tuple(spec.name for spec in FITS)
METHODS_FILE = "methods.npz"
DONE_MARKER = "DONE"

PIPELINE_OFF = "pipeline, flags off (primary)"
PIPELINE_ON = "pipeline, flags on (secondary)"
CLASSIFY_OFF = "classify() replay, flags off"      # consistency check only, never scored
BOOTSTRAP = "bootstrap labels (sheet)"
ELEVATION_MARGIN_M = 0.05
REFLECTANCE_THRESHOLD_DB = 0.25
ELEVATION = f"elevation z < level + {ELEVATION_MARGIN_M:.2f} m"
REFLECTANCE = f"reflectance < +{REFLECTANCE_THRESHOLD_DB:.2f} dB"
SCORED_METHODS = (PIPELINE_OFF, PIPELINE_ON, BOOTSTRAP, ELEVATION, REFLECTANCE)
PIPELINE_METHODS = (PIPELINE_OFF, PIPELINE_ON)

LAND, WATER = ab.LAND, ab.WATER
WATER_LABELS = ab.WATER_LABELS
LAND_LABELS = (LAND, LABEL_CANOPY)   # anything else (2, -1) is an abstention

STRICT = "strict"              # abstention is wrong whatever the truth (primary)
WATER_VS_REST = "water-vs-rest"  # abstention counts as "not water" (as in sections 11-14)
SCORINGS = (STRICT, WATER_VS_REST)

STRATUM_ALL = "A all no-look"
STRATUM_NEW = "B new only"
STRATUM_BELOW = "C Land_below_waterline"
STRATUM_SPIT = "D Landzunge"
STRATUM_SHALLOW = "E geometry-water"
STRATUM_LATE = "F Wasser4 (supplementary)"
STRATUM_ALL_LATE = "A+ all + Wasser4 (supplementary)"
COUNT_ONLY = (STRATUM_SHALLOW,)          # too few blocks for an interval
CLAIM_STRATA = (STRATUM_BELOW, STRATUM_SPIT)

BALANCED, WATER_RECALL, LAND_RECALL = "balanced_accuracy", "water_recall", "land_recall"
METRICS = (BALANCED, WATER_RECALL, LAND_RECALL)
SAMPLES = "bootstrap_samples"
REPORT_DIGITS = 4


# --------------------------------------------------------------------------
# Worker: one fit, every method's labels for the whole cloud
# --------------------------------------------------------------------------

def with_flags_on(config: PipelineConfig) -> PipelineConfig:
    return dataclasses.replace(
        config,
        surface=dataclasses.replace(config.surface, resolve_uncertain=True),
        cleanup=dataclasses.replace(config.cleanup, majority_filter=True, surface_prior=True))


def pipeline_for(spec: ab.FitSpec, config: PipelineConfig) -> WaterPipeline:
    return WaterPipeline(config, artifacts=LocalArtifactResolver(root=spec.dir / "models"))


def fitted_state(spec: ab.FitSpec, config: PipelineConfig, cloud: PointCloud) -> PipelineState:
    pipeline = pipeline_for(spec, config)
    if (spec.dir / "fit.npz").exists():
        # the fit finished but a later step did not: never train twice
        return ab.restore_fit(pipeline, spec, cloud)
    return ab.fit_from_scratch(pipeline, spec, cloud)


def run_worker(spec: ab.FitSpec) -> None:
    spec.dir.mkdir(parents=True, exist_ok=True)
    cloud = mt.raw_cloud(spec.site)
    config = ab.build_config(spec, cloud)
    if config.bootstrap.method != BOOTSTRAP_SHEET:
        raise ValueError(f"protocol asks for the sheet bootstrap, got {config.bootstrap.method}")
    level = derive_site_config(cloud)[1].water_level_z
    state = fitted_state(spec, config, cloud)
    methods = {
        PIPELINE_OFF: state.final_label,
        PIPELINE_ON: pipeline_for(spec, with_flags_on(config)).classify(cloud).final_label,
        CLASSIFY_OFF: pipeline_for(spec, config).classify(cloud).final_label,
        BOOTSTRAP: autolabel.bootstrap(state.features, config.zones, config.bootstrap),
        ELEVATION: np.where(cloud.z < level + ELEVATION_MARGIN_M, WATER, LAND),
        REFLECTANCE: np.where(cloud.reflectance_db < REFLECTANCE_THRESHOLD_DB, WATER, LAND),
    }
    np.savez_compressed(spec.dir / METHODS_FILE, xyz=cloud.xyz, water_level_z=level,
                        **{name: labels.astype(np.int8) for name, labels in methods.items()})
    (spec.dir / DONE_MARKER).touch()


# --------------------------------------------------------------------------
# Labels and strata
# --------------------------------------------------------------------------

def read_nolook(tree: cKDTree) -> dict[str, np.ndarray]:
    names = WATER_FILES + LAND_FILES + (LATE_WATER_FILE,)
    return {name: np.unique(au.match_to_cloud(tree, au.read_xyz(NOLOOK_DIR / f"{name}.asc")))
            for name in names}


def old_label_points(tree: cKDTree) -> np.ndarray:
    """Every cloud point in an old region or an old picked-point file."""
    regions = au.region_reference(au.load_regions(tree))
    return np.union1d(regions.idx.to_numpy(), au.load_blind(tree).idx.to_numpy())


def reference(name: str, water: np.ndarray, land: np.ndarray, xyz: np.ndarray) -> ab.Reference:
    if np.intersect1d(water, land).size:
        raise ValueError(f"{name}: points labelled both water and land")
    idx = np.concatenate([water, land])
    labels = mt.LabelSet(name, idx, np.repeat([True, False], [water.size, land.size]),
                         xyz[idx, :2])
    blocks = labels.blocks()
    counts = mt.block_multiplicities(int(blocks.max()) + 1, np.random.default_rng(mt.SEED))
    return ab.Reference(labels, blocks, counts)


def build_strata(xyz: np.ndarray) -> dict[str, ab.Reference]:
    tree = cKDTree(xyz)
    files = read_nolook(tree)
    water, land = au.unique_points(files, WATER_FILES), au.unique_points(files, LAND_FILES)
    old, none = old_label_points(tree), np.array([], dtype=np.int64)
    late = files[LATE_WATER_FILE]
    parts = {
        STRATUM_ALL: (water, land),
        STRATUM_NEW: (water[~np.isin(water, old)], land[~np.isin(land, old)]),
        STRATUM_BELOW: (none, files["Land_below_waterline"]),
        STRATUM_SPIT: (none, files["Landzunge"]),
        STRATUM_SHALLOW: (files["geometry-water"], none),
        STRATUM_LATE: (late, none),
        STRATUM_ALL_LATE: (np.union1d(water, late), land),
    }
    return {name: reference(name, w, l, xyz) for name, (w, l) in parts.items()}


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def is_correct(given: np.ndarray, is_water: np.ndarray, scoring: str) -> np.ndarray:
    says_water = np.isin(given, WATER_LABELS)
    says_land = np.isin(given, LAND_LABELS) if scoring == STRICT else ~says_water
    return np.where(is_water, says_water, says_land)


def recalls(tallies: np.ndarray) -> np.ndarray:
    """tallies[..., 4] as ab.block_tallies orders them -> METRICS."""
    tp, fp, fn, tn = np.moveaxis(tallies, -1, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        water, land = tp / (tp + fn), tn / (tn + fp)
    return np.stack([0.5 * (water + land), water, land], axis=-1)


def key_metric(ref: ab.Reference) -> str:
    is_water = ref.labels.is_water
    if is_water.all():
        return WATER_RECALL
    return BALANCED if is_water.any() else LAND_RECALL


def score(given: np.ndarray, ref: ab.Reference, scoring: str) -> dict:
    is_water = ref.labels.is_water
    correct = is_correct(given, is_water, scoring)
    # a wrong land point enters the tally as "called water", a wrong water
    # point as "not called water", whatever label the method actually gave
    per_block = ab.block_tallies(is_water, correct == is_water, ref.blocks)
    name = key_metric(ref)
    metric = METRICS.index(name)
    point, samples = recalls(per_block.sum(axis=0)), recalls(ref.counts @ per_block)
    abstained = ~np.isin(given, WATER_LABELS + LAND_LABELS)
    return {"n": int(len(given)), "n_water": int(is_water.sum()),
            "n_land": int((~is_water).sum()), "n_blocks": int(len(per_block)),
            "metric": name, "value": float(point[metric]),
            "value_ci95": mt.interval(samples[:, metric]),
            "water_correct": int(correct[is_water].sum()),
            "land_correct": int(correct[~is_water].sum()),
            "abstained": int(abstained.sum()), SAMPLES: samples[:, metric]}


def score_fit(fit: str, methods: dict[str, np.ndarray],
              strata: dict[str, ab.Reference]) -> list[dict]:
    return [{"fit": fit, "scoring": scoring, "stratum": stratum, "method": method,
             **score(methods[method][ref.labels.idx], ref, scoring)}
            for scoring in SCORINGS for stratum, ref in strata.items()
            for method in SCORED_METHODS]


def paired_rows(rows: list[dict]) -> list[dict]:
    """Each pipeline configuration minus the elevation rule, same block draws."""
    table = {(r["fit"], r["scoring"], r["stratum"], r["method"]): r for r in rows}
    out = []
    for (fit, scoring, stratum, method), row in table.items():
        if method not in PIPELINE_METHODS:
            continue
        rule = table[fit, scoring, stratum, ELEVATION]
        out.append({"fit": fit, "scoring": scoring, "stratum": stratum,
                    "contrast": f"{method}  minus  {ELEVATION}", "method": method,
                    "n": row["n"], "n_blocks": row["n_blocks"], "metric": row["metric"],
                    "pipeline": row["value"], "elevation": rule["value"],
                    "pipeline_correct": row["water_correct"] + row["land_correct"],
                    "elevation_correct": rule["water_correct"] + rule["land_correct"],
                    "value": row["value"] - rule["value"],
                    "value_ci95": mt.interval(row[SAMPLES] - rule[SAMPLES])})
    return out


def drop_intervals(rows: list[dict]) -> list[dict]:
    """Protocol: no interval on a stratum of three blocks."""
    return [row if row["stratum"] not in COUNT_ONLY
            else {k: v for k, v in row.items() if k != "value_ci95"} for row in rows]


# --------------------------------------------------------------------------
# The pre-registered criterion (protocol, section 6)
# --------------------------------------------------------------------------

def beats(row: dict) -> bool:
    return row["value"] > 0 and row["value_ci95"][0] > 0


def verdict(contrasts: list[dict]) -> dict:
    primary = {(r["fit"], r["stratum"]): r for r in contrasts
               if r["scoring"] == STRICT and r["method"] == PIPELINE_OFF}
    below, spit = ({fit: beats(primary[fit, stratum]) for fit in FIT_NAMES}
                   for stratum in CLAIM_STRATA)
    guard = {fit: primary[fit, STRATUM_ALL]["value_ci95"][1] >= 0 for fit in FIT_NAMES}
    if not all(below.values()):
        outcome = "refuted"
    elif all(spit.values()) and all(guard.values()):
        outcome = "holds"
    elif not all(guard.values()):
        outcome = "not counted: guard on stratum A failed"
    else:
        outcome = "mixed: holds for land well below the level, not shown for land at the level"
    return {"beats_elevation_on_C": below, "beats_elevation_on_D": spit,
            "guard_on_A": guard, "outcome": outcome}


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def consistency(fits: dict[str, dict[str, np.ndarray]]) -> dict:
    """Does classify() reproduce the fit, and how far apart are the seeds?"""
    water = {name: np.isin(methods[PIPELINE_OFF], WATER_LABELS) for name, methods in fits.items()}
    out = {f"{name}: classify() replay equals the fit's labels":
           float((methods[CLASSIFY_OFF] == methods[PIPELINE_OFF]).mean())
           for name, methods in fits.items()}
    first, second = FIT_NAMES
    out["water-mask agreement between the seeds, whole cloud"] = float(
        (water[first] == water[second]).mean())
    return out


def print_table(title: str, rows: list[dict], lead: list[str], plain: list[str]) -> None:
    frame = pd.DataFrame(rows)
    has_interval = np.array(["value_ci95" in row for row in rows])
    frame["value_ci95"] = [row.get("value_ci95", [np.nan, np.nan]) for row in rows]
    # a row without an interval is read off its *_correct columns, not as a rate
    frame["value [95%]"] = mt.with_ci(frame, "value", REPORT_DIGITS).where(
        has_interval, "counts only")
    print(f"\n=== {title} ===")
    print(frame[lead + ["n_blocks", "metric", "value [95%]"] + plain].to_string(
        index=False, float_format=lambda v: f"{v:.{REPORT_DIGITS}f}"))


def print_report(report: dict) -> None:
    for scoring in SCORINGS:
        scores = [r for r in report["scores"] if r["scoring"] == scoring]
        print_table(f"scores, {scoring} scoring", scores, ["fit", "stratum", "method"],
                    ["n_water", "water_correct", "n_land", "land_correct", "abstained"])
        deltas = [r for r in report["contrasts"] if r["scoring"] == scoring]
        print_table(f"pipeline minus elevation rule, {scoring} scoring", deltas,
                    ["fit", "stratum", "method"],
                    ["n", "pipeline_correct", "elevation_correct"])
    print("\n=== consistency ===")
    for name, value in report["consistency"].items():
        print(f"  {value:.6f}  {name}")
    print("\n=== criterion (primary configuration, strict scoring) ===")
    print(json.dumps(report["verdict"], indent=2))


def metadata(level: float) -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return {"git_commit": commit.stdout.strip(), "water_level_z": level,
            "bootstrap_seed": mt.SEED, "n_bootstrap": mt.N_BOOTSTRAP, "block_m": mt.CV_BLOCK_M,
            "fits": list(FIT_NAMES),
            "fit_metrics": {spec.name: json.loads((spec.dir / "fit_metrics.json").read_text())
                            for spec in FITS}}


def write_report(report: dict) -> None:
    tables = OUT_DIR / "testset_tables.txt"
    with tables.open("w") as handle, contextlib.redirect_stdout(handle):
        print_report(report)
    print(tables.read_text())
    out = OUT_DIR / "testset_results.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}\nwrote {tables}")


def evaluate() -> dict:
    missing = [spec.name for spec in FITS if not (spec.dir / DONE_MARKER).exists()]
    if missing:
        raise SystemExit(f"fits not finished: {missing} — run them with --worker first")
    fits = {spec.name: dict(np.load(spec.dir / METHODS_FILE)) for spec in FITS}
    first = fits[FIT_NAMES[0]]
    xyz = first["xyz"]
    if not all(np.array_equal(methods["xyz"], xyz) for methods in fits.values()):
        raise ValueError("the fits were run on different clouds")
    strata = build_strata(xyz)
    rows = [row for name, methods in fits.items() for row in score_fit(name, methods, strata)]
    contrasts = paired_rows(rows)
    scores = [{k: v for k, v in row.items() if k != SAMPLES} for row in drop_intervals(rows)]
    return {"meta": metadata(float(first["water_level_z"])),
            "scores": scores, "contrasts": drop_intervals(contrasts),
            "verdict": verdict(contrasts), "consistency": consistency(fits)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", help="run one fit, e.g. inn_unified_seed42_rep0")
    args = parser.parse_args()
    # FitSpec.dir and .cache_dir read this module global, so the fits, their
    # models and their feature cache all land under runs/inn_testset/.
    ab.OUT_DIR = OUT_DIR
    if args.worker:
        spec = ab.parse_spec(args.worker)
        if spec not in FITS:
            raise SystemExit(f"{args.worker} is not one of {FIT_NAMES}")
        run_worker(spec)
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_report(evaluate())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
