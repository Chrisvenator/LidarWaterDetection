"""Is the deep model necessary? End-to-end ablation on honest labels.

    python src/diagnostics/ablate_models.py            # every number, one command
    python src/diagnostics/ablate_models.py --rearm    # keep the fits, rebuild the arms
    python src/diagnostics/ablate_models.py --refit    # refit everything from scratch

Each entry of ``FITS`` is one from-scratch ``WaterPipeline.fit()`` in the base
configuration (all optional flags off). Every arm of a fit shares that fit's
bootstrap labels and trained models and differs only in what reaches the
geometry stage, so arm-against-arm differences inside a fit carry no training
noise. Training noise is measured separately: the same fit repeated with the
same seed (is training deterministic?) and with another seed.

Arms 1-6 are the ones the task asked for. 3b, 7a-c and 8 are additions:

* 3b swaps XGBoost for a logistic regression on the same scalars — is any
  non-linear model needed at all?
* 7a-c threshold the model probabilities with no geometry — what does the
  geometry stage add to a model?
* 8 is the WCN's own training labels (the v6-anchored geometry pass) — does
  the WCN improve on its teacher?

Arm 4 feeds the bootstrap labels to geometry as hard probabilities: water
(1, 1), land (0, 0), and unlabelled points as a disagreeing pair (XGBoost 1,
deep 0), which the stage reads as "uncertain" without letting them anchor the
footprint, the surface plane or the bed.

Water-versus-not-water is all that is scored, and the canopy merge only
relabels land and uncertain points, so no arm reruns the canopy stage.

Labels and scoring are the audits': Inn's hand-drawn regions (primary) and 65
picked points (secondary), Pielach's Mandlburger matches; abstention counts
as a miss; 95% intervals resample 10 m blocks; differences are paired.

Writes ``runs/ablation/ablation_results.json`` and ``ablation_tables.txt``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import itertools
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import measure_transfer as mt                                           # noqa: E402
from lidarwater import PipelineConfig, WaterPipeline, derive_site_config  # noqa: E402
from lidarwater import pipeline as pipeline_module                      # noqa: E402
from lidarwater._stages import autolabel, surface, wcn                  # noqa: E402
from lidarwater.artifacts import LocalArtifactResolver                  # noqa: E402
from lidarwater.config import BOOTSTRAP_SHEET, BOOTSTRAP_SURFACE        # noqa: E402
from lidarwater.site import Z_HISTOGRAM_BIN_M, estimate_water_level     # noqa: E402
from lidarwater.types import PipelineState, PointCloud                  # noqa: E402

OUT_DIR = ROOT / "runs" / "ablation"

# Pielach's shipped pipeline is PipelineConfig() (z-band bootstrap, 11
# scalars); any other survey gets derive_site_config (surface bootstrap,
# reflectance added). Pielach is also fitted the second way, because the
# z-bands were verified by eye and so are not strictly label-free.
SHIPPED, DERIVED = "shipped", "derived"
# derive_site_config with the canopy-robust sheet bootstrap, fitted by
# diagnose_bootstrap.py, not here. Both names pin their bootstrap, so each
# stays reproducible whichever one derive_site_config selects.
UNIFIED = "unified"
DERIVED_BOOTSTRAP = {DERIVED: BOOTSTRAP_SURFACE, UNIFIED: BOOTSTRAP_SHEET}
# Not a fit: the weights in models/ that every published Pielach figure came
# from, run through classify(). A yardstick for the from-scratch fits. Its v6
# checkpoint predates the library's layer names and cannot be loaded, so the
# two arms that need the WCN's training labels (3b, 8) are absent there.
PRETRAINED = "pretrained"
PRETRAINED_MODELS = ROOT / "models"
BASE_SEED = PipelineConfig().wcn.train.seed
OTHER_SEED = BASE_SEED + 1


@dataclasses.dataclass(frozen=True)
class FitSpec:
    site: str
    config: str
    seed: int
    rep: int

    @property
    def group(self) -> str:
        return f"{self.site}_{self.config}"

    @property
    def name(self) -> str:
        return f"{self.group}_seed{self.seed}_rep{self.rep}"

    @property
    def dir(self) -> Path:
        return OUT_DIR / self.name

    @property
    def cache_dir(self) -> Path:
        # the derived and unified configs extract identical features
        features = DERIVED if self.config in DERIVED_BOOTSTRAP else SHIPPED
        return OUT_DIR / "cache" / f"{self.site}_{features}"


def parse_spec(name: str) -> FitSpec:
    """Inverse of FitSpec.name, so a worker can run a fit FITS does not list."""
    site, config, seed, rep = name.split("_")
    return FitSpec(site, config, int(seed.removeprefix("seed")), int(rep.removeprefix("rep")))


FITS = (
    FitSpec(mt.PIELACH, SHIPPED, BASE_SEED, 0),
    FitSpec(mt.PIELACH, SHIPPED, BASE_SEED, 1),
    FitSpec(mt.PIELACH, SHIPPED, OTHER_SEED, 0),
    FitSpec(mt.INN, DERIVED, BASE_SEED, 0),
    FitSpec(mt.INN, DERIVED, BASE_SEED, 1),
    FitSpec(mt.INN, DERIVED, OTHER_SEED, 0),
    FitSpec(mt.PIELACH, DERIVED, BASE_SEED, 0),
    FitSpec(mt.PIELACH, PRETRAINED, BASE_SEED, 0),
)
FIT_FIELDS = ("final_label", "reconstructed_label", "wcn_proba", "wcn_xgb_proba",
              "autolabel_xgb_proba", "autolabel_deep_proba", "autolabel_ensemble")
# A fit peaks above 4 GB of GPU memory; seven side by side ran a 24 GB card
# out of memory, so the default leaves a wide margin.
DEFAULT_MAX_PARALLEL = 2

LAND, WATER, UNCERTAIN = surface.LABEL_LAND, surface.LABEL_WATER, surface.LABEL_UNCERTAIN
WATER_LABELS = mt.PIPELINE_WATER
DECISION_THRESHOLD = mt.DECISION_THRESHOLD
LOGISTIC_MAX_ITER = 1000
# Headroom above the modal water level for the elevation rule. The first is
# the top edge of the densest z-bin, i.e. it comes from the site profile's own
# definition; the rest show how much the rule depends on the choice.
ELEVATION_MARGINS_M = (0.5 * Z_HISTOGRAM_BIN_M, 0.0, Z_HISTOGRAM_BIN_M,
                       PipelineConfig().surface.water_tol_m)

ARM_FULL = "1 full pipeline (WCN+XGB -> geometry)"
ARM_DEEP = "2 WCN only -> geometry"
ARM_XGB = "3 XGBoost only -> geometry"
ARM_LOGISTIC = "3b logistic regression -> geometry"
ARM_BOOT_GEOMETRY = "4 bootstrap labels -> geometry"
ARM_BOOT = "5 bootstrap labels"
ARM_ELEVATION = "6a elevation z < level + {:.2f} m"
ARM_REFLECTANCE = "6b reflectance < Otsu"
ARM_MODELS_RAW = "7a WCN+XGB agree, no geometry"
ARM_DEEP_RAW = "7b WCN >= 0.5, no geometry"
ARM_XGB_RAW = "7c XGBoost >= 0.5, no geometry"
ARM_TEACHER = "8 WCN training labels (v6 -> geometry)"
NOISE_ARMS = (ARM_FULL, ARM_DEEP, ARM_XGB)
EXTRA_CONTRASTS = ((ARM_DEEP, ARM_XGB), (ARM_FULL, ARM_TEACHER),
                   (ARM_BOOT_GEOMETRY, ARM_BOOT), (ARM_FULL, ARM_MODELS_RAW))

METRICS = ("balanced_accuracy", "f1")
SAMPLES = "bootstrap_samples"  # per-resample metrics, kept in memory for paired contrasts
REPORT_DIGITS = 4


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def build_config(spec: FitSpec, cloud: PointCloud) -> PipelineConfig:
    base = PipelineConfig()
    if spec.config in DERIVED_BOOTSTRAP:
        base = derive_site_config(cloud)[0]
        base = dataclasses.replace(base, bootstrap=dataclasses.replace(
            base.bootstrap, method=DERIVED_BOOTSTRAP[spec.config]))
    if (base.surface.resolve_uncertain or base.cleanup.majority_filter
            or base.cleanup.surface_prior):
        raise ValueError("the ablation scores the base configuration: all flags off")
    train = dataclasses.replace(base.wcn.train, seed=spec.seed)
    return dataclasses.replace(
        base, wcn=dataclasses.replace(base.wcn, train=train),
        run=dataclasses.replace(base.run, cache_dir=spec.cache_dir))


@contextlib.contextmanager
def exclusive(lock_path: Path):
    """Fits of one group share a feature cache; only one may build it."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def features_only(pipeline: WaterPipeline, spec: FitSpec, cloud: PointCloud) -> PipelineState:
    with exclusive(spec.cache_dir.with_suffix(".lock")):
        return pipeline.run_stages(cloud, ())


def fit_from_scratch(pipeline: WaterPipeline, spec: FitSpec, cloud: PointCloud) -> PipelineState:
    features_only(pipeline, spec, cloud)
    # wcn.fit seeds itself from the config; the autolabel CNN does not, so
    # without this its initial weights would differ on every run.
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    state = pipeline.fit(cloud)
    np.savez_compressed(spec.dir / "fit.npz", **{f: getattr(state, f) for f in FIT_FIELDS})
    (spec.dir / "fit_metrics.json").write_text(json.dumps(state.metrics, indent=2, default=float))
    return state


def classify_pretrained(pipeline: WaterPipeline, spec: FitSpec,
                        cloud: PointCloud) -> PipelineState:
    features_only(pipeline, spec, cloud)
    return pipeline.classify(cloud)


def restore_fit(pipeline: WaterPipeline, spec: FitSpec, cloud: PointCloud) -> PipelineState:
    state = features_only(pipeline, spec, cloud)
    for field, value in np.load(spec.dir / "fit.npz").items():
        setattr(state, field, value)
    return state


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------

def geometry_labels(state: PipelineState, config: PipelineConfig,
                    xgb_proba: np.ndarray, deep_proba: np.ndarray) -> np.ndarray:
    """The final ("v10") geometry pass, fed the given probabilities."""
    trial = dataclasses.replace(state, wcn_xgb_proba=xgb_proba.astype(np.float32),
                                wcn_proba=deep_proba.astype(np.float32), metrics={})
    surface.run(trial, config.surface, geometry_only=True)
    return trial.reconstructed_label


def teacher_labels(state: PipelineState, config: PipelineConfig) -> np.ndarray:
    """The labels fit() trained the WCN on: its first, v6-anchored geometry pass."""
    trial = dataclasses.replace(state, metrics={})
    surface.run(trial, config.surface, geometry_only=False)
    return pipeline_module._wcn_bootstrap(trial)[0]


def logistic_proba(state: PipelineState, config: PipelineConfig,
                   teacher: np.ndarray) -> np.ndarray:
    """A linear model on the WCN's scalars and training labels."""
    scalars, _ = wcn._normalise_scalars(wcn._scalars(state.features, config.wcn.scalar_features))
    scalars = np.nan_to_num(scalars)
    decided = teacher != UNCERTAIN
    model = LogisticRegression(class_weight="balanced", max_iter=LOGISTIC_MAX_ITER)
    model.fit(scalars[decided], teacher[decided] == WATER)
    return model.predict_proba(scalars)[:, 1]


def bootstrap_as_probas(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(xgb, deep) pair per point: water (1, 1), land (0, 0), unlabelled (1, 0)."""
    return np.where(labels == LAND, 0.0, 1.0), np.where(labels == WATER, 1.0, 0.0)


def thresholded(*probas: np.ndarray) -> np.ndarray:
    """Water where every probability says so, land where none does, else uncertain."""
    votes = np.stack([p >= DECISION_THRESHOLD for p in probas])
    return np.where(votes.all(axis=0), WATER, np.where(votes.any(axis=0), UNCERTAIN, LAND))


def baseline_arms(cloud: PointCloud) -> dict[str, np.ndarray]:
    level = estimate_water_level(cloud.z)
    arms = {ARM_ELEVATION.format(margin): cloud.z < level + margin
            for margin in ELEVATION_MARGINS_M}
    arms[ARM_REFLECTANCE] = cloud.reflectance_db < autolabel._otsu(cloud.reflectance_db)
    return arms


def check_replay(state: PipelineState, config: PipelineConfig) -> None:
    replay = geometry_labels(state, config, state.wcn_xgb_proba, state.wcn_proba)
    if not np.array_equal(np.isin(replay, WATER_LABELS), np.isin(state.final_label, WATER_LABELS)):
        raise ValueError("replaying geometry on the fit's own probabilities changed its water "
                         "mask — arms would not be comparable to the full pipeline")


def teacher_arms(state: PipelineState, config: PipelineConfig) -> dict[str, np.ndarray]:
    """The arms that need the WCN's training labels, so the v6 stage's output."""
    if state.autolabel_ensemble is None:
        return {}
    teacher = teacher_labels(state, config)
    linear = logistic_proba(state, config, teacher)
    return {ARM_LOGISTIC: geometry_labels(state, config, linear, linear), ARM_TEACHER: teacher}


def build_arms(state: PipelineState, config: PipelineConfig,
               cloud: PointCloud) -> dict[str, np.ndarray]:
    check_replay(state, config)
    deep, xgb = state.wcn_proba, state.wcn_xgb_proba
    boot = autolabel.bootstrap(state.features, config.zones, config.bootstrap)
    arms = {
        ARM_FULL: state.final_label,
        ARM_DEEP: geometry_labels(state, config, deep, deep),
        ARM_XGB: geometry_labels(state, config, xgb, xgb),
        ARM_BOOT_GEOMETRY: geometry_labels(state, config, *bootstrap_as_probas(boot)),
        ARM_BOOT: boot == WATER,
        **baseline_arms(cloud),
        ARM_MODELS_RAW: thresholded(deep, xgb),
        ARM_DEEP_RAW: thresholded(deep),
        ARM_XGB_RAW: thresholded(xgb),
        **teacher_arms(state, config),
    }
    return {name: labels.astype(np.int8) for name, labels in arms.items()}


def run_worker(spec: FitSpec, refit: bool) -> None:
    spec.dir.mkdir(parents=True, exist_ok=True)
    cloud = mt.raw_cloud(spec.site)
    config = build_config(spec, cloud)
    pretrained = spec.config == PRETRAINED
    models = PRETRAINED_MODELS if pretrained else spec.dir / "models"
    pipeline = WaterPipeline(config, artifacts=LocalArtifactResolver(root=models))
    if pretrained:
        state = classify_pretrained(pipeline, spec, cloud)
    elif (spec.dir / "fit.npz").exists() and not refit:
        state = restore_fit(pipeline, spec, cloud)
    else:
        state = fit_from_scratch(pipeline, spec, cloud)
    np.savez_compressed(spec.dir / "arms.npz", xyz=cloud.xyz,
                        **build_arms(state, config, cloud))


def run_workers(fits: tuple[FitSpec, ...], refit: bool, rearm: bool,
                max_parallel: int) -> None:
    """One subprocess per fit, so none inherits another's torch state. A fit
    whose arms are already on disk is skipped unless asked for again; one
    failure does not stop the others, but it does stop the scoring."""
    todo = [s for s in fits if refit or rearm or not (s.dir / "arms.npz").exists()]
    for spec in fits:
        print(f"  {'run ' if spec in todo else 'keep'} {spec.name}", flush=True)

    def launch(spec: FitSpec) -> bool:
        spec.dir.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, __file__, "--worker", spec.name] + ["--refit"] * refit
        with (spec.dir / "worker.log").open("w") as log:
            done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        print(f"  {'FAILED' if done.returncode else 'finished'} {spec.name}", flush=True)
        return done.returncode == 0

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        succeeded = list(pool.map(launch, todo))
    failed = [spec.name for spec, ok in zip(todo, succeeded) if not ok]
    if failed:
        raise RuntimeError(f"fits failed, see their worker.log under {OUT_DIR}: {failed}")


# --------------------------------------------------------------------------
# Scoring: balanced accuracy and F1 under one shared set of block resamples
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True, eq=False)
class Reference:
    labels: mt.LabelSet
    blocks: np.ndarray
    counts: np.ndarray         # (N_BOOTSTRAP, n_blocks) block multiplicities


def load_references(site: str, xyz: np.ndarray) -> dict[str, Reference]:
    sets = ({mt.HONEST: mt.pielach_reference(xyz)} if site == mt.PIELACH
            else mt.inn_references(xyz))
    out = {}
    for name, labels in sets.items():
        blocks = labels.blocks()
        counts = mt.block_multiplicities(int(blocks.max()) + 1, np.random.default_rng(mt.SEED))
        out[name] = Reference(labels, blocks, counts)
    return out


def block_tallies(is_water: np.ndarray, called: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    """Per block: true positives, false positives, false negatives, true negatives."""
    per_point = np.column_stack([is_water & called, ~is_water & called,
                                 is_water & ~called, ~is_water & ~called])
    per_block = np.zeros((int(blocks.max()) + 1, 4))
    np.add.at(per_block, blocks, per_point)
    return per_block


def balanced_accuracy_and_f1(tallies: np.ndarray) -> np.ndarray:
    tp, fp, fn, tn = (tallies[..., i] for i in range(4))
    with np.errstate(invalid="ignore", divide="ignore"):
        balanced = 0.5 * (tp / (tp + fn) + tn / (tn + fp))
        f1 = 2 * tp / (2 * tp + fp + fn)
    return np.stack([balanced, f1], axis=-1)


def score_arm(arm_labels: np.ndarray, ref: Reference) -> dict:
    """Water (1, 3) against everything else; an uncertain point is a miss."""
    given = arm_labels[ref.labels.idx]
    per_block = block_tallies(ref.labels.is_water, np.isin(given, WATER_LABELS), ref.blocks)
    totals = per_block.sum(axis=0)
    tp, fp, fn, tn = totals
    point = balanced_accuracy_and_f1(totals)
    samples = balanced_accuracy_and_f1(ref.counts @ per_block)
    row = {"n": int(len(given)), "n_blocks": int(len(per_block)),
           "water_recall": float(tp / (tp + fn)), "land_recall": float(tn / (tn + fp)),
           "abstained": float((given == UNCERTAIN).mean()), SAMPLES: samples}
    for i, metric in enumerate(METRICS):
        row[metric] = float(point[i])
        row[f"{metric}_ci95"] = mt.interval(samples[:, i])
    return row


def paired(contrast: str, a: dict, b: dict) -> dict:
    """a - b on the same labelled points under the same block draws."""
    row = {"contrast": contrast, "n": a["n"], "n_blocks": a["n_blocks"]}
    for i, metric in enumerate(METRICS):
        row[f"delta_{metric}"] = a[metric] - b[metric]
        row[f"delta_{metric}_ci95"] = mt.interval(a[SAMPLES][:, i] - b[SAMPLES][:, i])
    return row


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

Arms = dict[str, np.ndarray]
Scores = dict[tuple[str, str, str], dict]      # (fit, label set, arm) -> score


def score_all(fits: dict[FitSpec, Arms]) -> Scores:
    scores, references = {}, {}
    for spec, arms in fits.items():
        if spec.site not in references:
            references[spec.site] = load_references(spec.site, arms["xyz"])
        for kind, ref in references[spec.site].items():
            for arm, labels in arms.items():
                if arm != "xyz":
                    scores[spec.name, kind, arm] = score_arm(labels, ref)
    return scores


def score_rows(scores: Scores) -> list[dict]:
    return [{"fit": fit, "labels": kind, "arm": arm, **score}
            for (fit, kind, arm), score in scores.items()]


def arm_contrasts(scores: Scores) -> list[dict]:
    """Every arm against the full pipeline of the same fit, plus a few
    arm-against-arm contrasts the verdict rests on."""
    rows = []
    for fit, kind in sorted({key[:2] for key in scores}):
        arms = [arm for (f, k, arm) in scores if (f, k) == (fit, kind)]
        pairs = [(arm, ARM_FULL) for arm in arms if arm != ARM_FULL]
        pairs += [pair for pair in EXTRA_CONTRASTS if set(pair) <= set(arms)]
        rows += [{"fit": fit, "labels": kind,
                  **paired(f"{a}  minus  {b}", scores[fit, kind, a], scores[fit, kind, b])}
                 for a, b in pairs]
    return rows


def noise_floor(fits: dict[FitSpec, Arms], scores: Scores) -> list[dict]:
    """The same arm from two fits of one group: what training alone moves."""
    rows = []
    for a, b in itertools.combinations(fits, 2):
        if a.group != b.group:
            continue
        kinds = sorted({kind for (fit, kind, _) in scores if fit == a.name})
        for kind, arm in itertools.product(kinds, NOISE_ARMS):
            same = np.isin(fits[a][arm], WATER_LABELS) == np.isin(fits[b][arm], WATER_LABELS)
            rows.append({"group": a.group, "labels": kind, "arm": arm,
                         "cause": "same seed" if a.seed == b.seed else "other seed",
                         "cloud_agreement": float(same.mean()),
                         **paired(f"{a.name}  minus  {b.name}",
                                  scores[a.name, kind, arm], scores[b.name, kind, arm])})
    return rows


def print_table(title: str, rows: list[dict], lead: list[str], columns: list[str],
                plain: list[str]) -> None:
    frame = pd.DataFrame(rows)
    for column in columns:
        frame[f"{column} [95%]"] = mt.with_ci(frame, column, REPORT_DIGITS)
    shown = lead + ["n", "n_blocks"] + [f"{c} [95%]" for c in columns] + plain
    print(f"\n=== {title} ===")
    print(frame[shown].to_string(index=False, float_format=lambda v: f"{v:.{REPORT_DIGITS}f}"))


def print_report(report: dict[str, list[dict]]) -> None:
    deltas = [f"delta_{metric}" for metric in METRICS]
    print_table("arms", report["arms"], ["fit", "labels", "arm"], list(METRICS),
                ["water_recall", "land_recall", "abstained"])
    print_table("paired contrasts within a fit", report["contrasts"],
                ["fit", "labels", "contrast"], deltas, [])
    print_table("noise floor: one arm, two fits", report["noise_floor"],
                ["group", "labels", "arm", "cause", "contrast"], deltas, ["cloud_agreement"])


def metadata() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return {"git_commit": commit.stdout.strip(), "bootstrap_seed": mt.SEED,
            "n_bootstrap": mt.N_BOOTSTRAP, "block_m": mt.CV_BLOCK_M,
            "fits": [dataclasses.asdict(spec) for spec in FITS],
            "fit_metrics": {spec.name: json.loads((spec.dir / "fit_metrics.json").read_text())
                            for spec in FITS if spec.config != PRETRAINED},
            "elevation_margins_m": ELEVATION_MARGINS_M}


def write_report(report: dict[str, list[dict]]) -> None:
    tables = OUT_DIR / "ablation_tables.txt"
    with tables.open("w") as handle, contextlib.redirect_stdout(handle):
        print_report(report)
    print(tables.read_text())
    slim = {key: [{k: v for k, v in row.items() if k != SAMPLES} for row in rows]
            for key, rows in report.items()}
    out = OUT_DIR / "ablation_results.json"
    out.write_text(json.dumps({"meta": metadata(), **slim}, indent=2))
    print(f"\nwrote {out}\nwrote {tables}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refit", action="store_true", help="refit even where a fit is cached")
    parser.add_argument("--rearm", action="store_true", help="rebuild the arms from cached fits")
    parser.add_argument("--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL,
                        help="fits training on the GPU at once")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        run_worker(parse_spec(args.worker), args.refit)
        return 0
    run_workers(FITS, args.refit, args.rearm, args.max_parallel)
    fits = {spec: dict(np.load(spec.dir / "arms.npz")) for spec in FITS}
    scores = score_all(fits)
    write_report({"arms": score_rows(scores), "contrasts": arm_contrasts(scores),
                  "noise_floor": noise_floor(fits, scores)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
