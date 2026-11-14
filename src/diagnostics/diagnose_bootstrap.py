"""Why does the elevation-free bootstrap collapse on Pielach, and does the
"sheet" bootstrap fix it on both surveys?

    python src/diagnostics/diagnose_bootstrap.py           # diagnosis + development signals
    python src/diagnostics/diagnose_bootstrap.py --fit     # + fit the unified configuration
    python src/diagnostics/diagnose_bootstrap.py --score   # + the frozen-design evaluation

Without ``--score`` nothing here reads the Mandlburger matches: Pielach is
judged by where the labels fall (elevation bands checked by eye, height above
ground), by coverage and by physical sanity; Inn by its hand-drawn regions,
which are the development set already.

``--fit`` fits the unified configuration (``derive_site_config`` with the
sheet bootstrap) from scratch on both sites under two seeds and reports the
same development signals for every arm of the fitted pipeline.

``--score`` scores those fits with ``ablate_models``' arms and
block-bootstrap machinery, paired against the z-band configuration and the
elevation rule. It reads the Mandlburger matches, so it is meant to be run
once, on a frozen design.

Needs the feature caches ``ablate_models.py`` leaves under
``runs/ablation/cache``. Writes ``runs/unified_bootstrap/``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import ablate_models as ab                                              # noqa: E402
import measure_transfer as mt                                           # noqa: E402
from lidarwater._stages import autolabel                                # noqa: E402
from lidarwater.config import (                                         # noqa: E402
    BOOTSTRAP_SHEET,
    BOOTSTRAP_SURFACE,
    BootstrapConfig,
    CanopyConfig,
    ZoneConfig,
)
from lidarwater.site import CANOPY_PROBE_M, estimate_water_level, height_above_ground  # noqa: E402

OUT_DIR = ROOT / "runs" / "unified_bootstrap"
SITES = (mt.PIELACH, mt.INN)
METHODS = (BOOTSTRAP_SURFACE, BOOTSTRAP_SHEET)

# Pielach's elevation bands as checked by eye in CloudCompare (CLAUDE.md,
# "Elevation zones"): at or below the first everything is water, above the
# second everything is canopy. A development proxy for Pielach only.
PIELACH_WATER_Z_MAX = CanopyConfig().z_clear_max
PIELACH_CANOPY_Z_MIN = CanopyConfig().z_canopy_min
BAND_WATER, BAND_MIDDLE, BAND_CANOPY = "water band", "transition", "canopy band"
# "At the water level" for a cell surface: within the geometry stage's own
# water tolerance of the site's modal water level.
AT_LEVEL_M = ab.PipelineConfig().surface.water_tol_m
LAYER_AT, LAYER_LOW, LAYER_HIGH = "at water level", "above it, under 3 m", "3 m or more above"
MAP_FIGSIZE = (26, 10)
MAP_HEIGHT_RANGE_M = 3.0
MAP_DPI = 80
REFLECTANCE_QUANTILES = (0.1, 0.5, 0.9)
HEIGHT_QUANTILES = (0.5, 0.95, 0.99)

# One-at-a-time departures from the frozen BootstrapConfig, to show which
# parameters the labels depend on.
SWEEPS = {
    "dense_layer_ratio": (0.3, 0.4, 0.6, 0.7, 1.0),
    "cell_m": (2.0, 4.0),
    "coherence_max_m": (0.10, 0.25),
    "sheet_filter_cells": (5, 9),
    "sheet_bin_m": (0.05, 0.20),
}

UNIFIED_FITS = tuple(ab.FitSpec(site, ab.UNIFIED, seed, 0)
                     for site in SITES for seed in (ab.BASE_SEED, ab.OTHER_SEED))
# What each unified fit is paired against: the same seed's fit of the
# configuration it would replace.
INCUMBENT = {mt.PIELACH: ab.SHIPPED, mt.INN: ab.DERIVED}
PAIRED_ARMS = (ab.ARM_FULL, ab.ARM_BOOT_GEOMETRY, ab.ARM_BOOT)


def load_features(site: str) -> pd.DataFrame:
    path = ab.OUT_DIR / "cache" / f"{site}_{ab.DERIVED}" / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run ablate_models.py once to build the caches")
    return pd.read_parquet(path)


def bootstrap_config(method: str, **overrides) -> BootstrapConfig:
    return dataclasses.replace(BootstrapConfig(), method=method, **overrides)


# --------------------------------------------------------------------------
# Diagnosis: where does a cell's surface land, and what does it look like?
# --------------------------------------------------------------------------

def cell_table(features: pd.DataFrame, method: str) -> pd.DataFrame:
    """One row per populated cell: its surface, that surface's reflectance,
    whether it is a coherent sheet, and which side of Otsu it falls."""
    config = bootstrap_config(method)
    cells = autolabel._cell_sheets if method == BOOTSTRAP_SHEET else autolabel._cell_surfaces
    surface, reflectance, _ = cells(features, config)
    level = estimate_water_level(features["z"].to_numpy())
    table = pd.DataFrame({
        "above_level": surface.ravel() - level,
        "reflectance": reflectance,
        "coherent": autolabel._sheet_coherence(surface, config) < config.coherence_max_m,
    })[np.isfinite(reflectance)]
    threshold = autolabel._otsu(table.reflectance[table.coherent].to_numpy())
    table["dark"] = table.reflectance < threshold
    table["layer"] = np.select(
        [table.above_level >= CANOPY_PROBE_M, table.above_level > AT_LEVEL_M],
        [LAYER_HIGH, LAYER_LOW], LAYER_AT)
    table.attrs["otsu"] = threshold
    return table


def layer_rows(site: str, method: str, table: pd.DataFrame) -> list[dict]:
    rows = []
    for layer, cells in table.groupby("layer"):
        low, mid, high = cells.reflectance.quantile(REFLECTANCE_QUANTILES)
        rows.append({
            "site": site, "surface": method, "lands": layer, "cells": len(cells),
            "share": len(cells) / len(table), "coherent": int(cells.coherent.sum()),
            "refl_p10": low, "refl_p50": mid, "refl_p90": high,
            "otsu_db": table.attrs["otsu"],
            "coherent_dark": int((cells.coherent & cells.dark).sum()),
            "coherent_bright": int((cells.coherent & ~cells.dark).sum()),
        })
    return rows


def cell_classes(features: pd.DataFrame, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Rasters of surface height above the water level, and of what the
    bootstrap makes of each cell: 0 not a coherent sheet, 1 bright sheet,
    2 dark sheet (water)."""
    config = bootstrap_config(method)
    cells = autolabel._cell_sheets if method == BOOTSTRAP_SHEET else autolabel._cell_surfaces
    surface, reflectance, _ = cells(features, config)
    reflectance = reflectance.reshape(surface.shape)
    coherent = (np.isfinite(reflectance)
                & (autolabel._sheet_coherence(surface, config).reshape(surface.shape)
                   < config.coherence_max_m))
    dark = reflectance < autolabel._otsu(reflectance[coherent])
    classes = np.where(np.isfinite(reflectance), coherent * (1 + dark), np.nan)
    return surface - estimate_water_level(features["z"].to_numpy()), classes


def plot_cell_maps(features: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(len(SITES), 2 * len(METHODS), figsize=MAP_FIGSIZE)
    for row, site in zip(axes, SITES):
        for pair, method in zip(row.reshape(len(METHODS), 2), METHODS):
            height, classes = cell_classes(features[site], method)
            shown = pair[0].imshow(height.T, origin="lower", cmap="coolwarm",
                                   vmin=-MAP_HEIGHT_RANGE_M, vmax=MAP_HEIGHT_RANGE_M)
            fig.colorbar(shown, ax=pair[0], shrink=0.6)
            pair[0].set_title(f"{site}, {method}: cell surface above water level [m]")
            pair[1].imshow(classes.T, origin="lower", cmap="viridis", vmin=0, vmax=2)
            pair[1].set_title("purple: no sheet, green: bright sheet, yellow: dark sheet = water")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "cell_maps.png"
    fig.savefig(path, dpi=MAP_DPI)
    plt.close(fig)
    return path


def pielach_band(z: np.ndarray) -> np.ndarray:
    return np.select([z <= PIELACH_WATER_Z_MAX, z > PIELACH_CANOPY_Z_MIN],
                     [BAND_WATER, BAND_CANOPY], BAND_MIDDLE)


def point_reflectance_rows(features: pd.DataFrame) -> list[dict]:
    """Pielach: is canopy distinguishable from water by reflectance at all?"""
    bands = pielach_band(features["z"].to_numpy())
    rows = []
    for band, values in features["reflectance_dB"].groupby(bands):
        low, mid, high = values.quantile(REFLECTANCE_QUANTILES)
        rows.append({"band": band, "points": len(values),
                     "refl_p10": low, "refl_p50": mid, "refl_p90": high})
    return rows


# --------------------------------------------------------------------------
# Development signals: where do the labels fall?
# --------------------------------------------------------------------------

def label_shares(labels: np.ndarray, mask: np.ndarray, prefix: str) -> dict:
    return {f"{prefix}_water": float((labels[mask] == 1).mean()),
            f"{prefix}_land": float((labels[mask] == 0).mean()),
            f"{prefix}_unlabelled": float((labels[mask] == -1).mean())}


def placement_row(features: pd.DataFrame, labels: np.ndarray) -> dict:
    """Label-free placement: coverage, and how high the water labels sit."""
    z = features["z"].to_numpy()
    water = labels == 1
    above_level = z[water] - estimate_water_level(z)
    above_ground = height_above_ground(features[["x", "y", "z"]].to_numpy())
    row = {"coverage": float((labels != -1).mean()), "water_share": float(water.mean()),
           "water_above_level_tol": float((above_level > AT_LEVEL_M).mean()),
           "water_above_probe_over_ground": float((above_ground[water] > CANOPY_PROBE_M).mean()),
           "tall_points_called_water": float(water[above_ground > CANOPY_PROBE_M].mean())}
    for q, value in zip(HEIGHT_QUANTILES, np.quantile(above_level, HEIGHT_QUANTILES)):
        row[f"water_above_level_p{round(q * 100)}"] = float(value)
    return row


def pielach_band_row(features: pd.DataFrame, labels: np.ndarray) -> dict:
    bands = pielach_band(features["z"].to_numpy())
    row = {}
    for band in (BAND_WATER, BAND_MIDDLE, BAND_CANOPY):
        row |= label_shares(labels, bands == band, band.split()[0])
    return row


def inn_region_row(labels: np.ndarray, reference: ab.Reference) -> dict:
    score = ab.score_arm((labels == 1).astype(np.int8), reference)
    return {key: value for key, value in score.items() if key != ab.SAMPLES}


def development_rows(features: dict[str, pd.DataFrame], reference: ab.Reference) -> list[dict]:
    rows = []
    variants = [(method, {}) for method in METHODS]
    variants += [(BOOTSTRAP_SHEET, {name: value}) for name, values in SWEEPS.items()
                 for value in values]
    for method, overrides in variants:
        config = bootstrap_config(method, **overrides)
        for site in SITES:
            labels = autolabel.bootstrap(features[site], ZoneConfig(), config)
            row = {"site": site, "bootstrap": method, "override": json.dumps(overrides),
                   **placement_row(features[site], labels)}
            row |= (pielach_band_row(features[site], labels) if site == mt.PIELACH
                    else inn_region_row(labels, reference))
            rows.append(row)
    return rows


def inn_regions(features: pd.DataFrame) -> ab.Reference:
    return ab.load_references(mt.INN, features[["x", "y", "z"]].to_numpy())[mt.HONEST]


# --------------------------------------------------------------------------
# Frozen-design evaluation (reads the Mandlburger matches)
# --------------------------------------------------------------------------

def incumbent_of(spec: ab.FitSpec) -> ab.FitSpec:
    return ab.FitSpec(spec.site, INCUMBENT[spec.site], spec.seed, 0)


def load_arms(spec: ab.FitSpec) -> ab.Arms:
    return dict(np.load(spec.dir / "arms.npz"))


def fit_signals(max_parallel: int, features: dict[str, pd.DataFrame],
                reference: ab.Reference) -> list[dict]:
    """Fit the unified configuration and say where each arm puts its water,
    next to the incumbent fit's — still without the Mandlburger matches."""
    ab.run_workers(UNIFIED_FITS, refit=False, rearm=False, max_parallel=max_parallel)
    rows = []
    for spec in UNIFIED_FITS + tuple(incumbent_of(s) for s in UNIFIED_FITS):
        for arm, labels in load_arms(spec).items():
            if arm == "xyz":
                continue
            water = np.isin(labels, ab.WATER_LABELS).astype(np.int8)
            row = {"fit": spec.name, "arm": arm, **placement_row(features[spec.site], water)}
            row |= (pielach_band_row(features[spec.site], water) if spec.site == mt.PIELACH
                    else inn_region_row(water, reference))
            rows.append(row)
    return rows


def evaluation() -> dict[str, list[dict]]:
    fits = {spec: load_arms(spec) for spec in UNIFIED_FITS}
    fits |= {incumbent_of(spec): load_arms(incumbent_of(spec)) for spec in UNIFIED_FITS}
    scores = ab.score_all(fits)
    return {"arms": ab.score_rows(scores), "contrasts": evaluation_contrasts(scores),
            "noise_floor": ab.noise_floor(fits, scores)}


def evaluation_contrasts(scores: ab.Scores) -> list[dict]:
    """Unified against the incumbent fit of the same seed, and against the
    elevation rule, on the same points under the same block draws."""
    elevation = ab.ARM_ELEVATION.format(ab.ELEVATION_MARGINS_M[0])
    rows = []
    for spec in UNIFIED_FITS:
        old = incumbent_of(spec).name
        kinds = sorted({kind for (fit, kind, _) in scores if fit == spec.name})
        for kind in kinds:
            new_full = scores[spec.name, kind, ab.ARM_FULL]
            rows += [{"fit": spec.name, "labels": kind,
                      **ab.paired(f"unified {arm}  minus  {old} {arm}",
                                  scores[spec.name, kind, arm], scores[old, kind, arm])}
                     for arm in PAIRED_ARMS]
            rows.append({"fit": spec.name, "labels": kind,
                         **ab.paired(f"unified {ab.ARM_FULL}  minus  {elevation}",
                                     new_full, scores[spec.name, kind, elevation])})
    return rows


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def print_frame(title: str, rows: list[dict]) -> None:
    print(f"\n=== {title} ===")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def print_development(rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    for site in SITES:
        shown = frame[frame.site == site].dropna(axis=1, how="all")
        shown = shown.drop(columns=[c for c in shown if c.endswith("_ci95")])
        print_frame(f"development signals, bootstrap labels: {site}", shown.to_dict("records"))


def write(name: str, printer, payload) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = OUT_DIR / f"{name}.txt"
    with text.open("w") as handle, contextlib.redirect_stdout(handle):
        printer(payload)
    print(text.read_text())
    (OUT_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2, default=float))


def print_diagnosis(report: dict) -> None:
    print_frame("where each cell's surface lands (top = 'surface', highest dense layer = 'sheet')",
                report["layers"])
    print_frame("Pielach point reflectance by eye-checked elevation band", report["pielach_points"])
    print_development(report["development"])


def diagnosis(features: dict[str, pd.DataFrame], reference: ab.Reference) -> dict:
    layers = [row for site in SITES for method in METHODS
              for row in layer_rows(site, method, cell_table(features[site], method))]
    return {"layers": layers,
            "pielach_points": point_reflectance_rows(features[mt.PIELACH]),
            "development": development_rows(features, reference)}


def print_fit_signals(rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    frame = frame.drop(columns=[c for c in frame if c.endswith("_ci95")])
    for site in SITES:
        shown = frame[frame.fit.str.startswith(site)].dropna(axis=1, how="all")
        print_frame(f"development signals, fitted pipeline arms: {site}",
                    shown.to_dict("records"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit", action="store_true",
                        help="fit the unified configuration and report development signals")
    parser.add_argument("--score", action="store_true",
                        help="score the fits (reads the Mandlburger matches; implies --fit)")
    parser.add_argument("--max-parallel", type=int, default=ab.DEFAULT_MAX_PARALLEL)
    args = parser.parse_args()

    features = {site: load_features(site) for site in SITES}
    reference = inn_regions(features[mt.INN])
    write("diagnosis", print_diagnosis, diagnosis(features, reference))
    print(f"wrote {plot_cell_maps(features)}")
    if args.fit or args.score:
        write("fit_signals", print_fit_signals,
              fit_signals(args.max_parallel, features, reference))
    if args.score:
        report = evaluation()
        slim = {key: [{k: v for k, v in row.items() if k != ab.SAMPLES} for row in rows]
                for key, rows in report.items()}
        write("evaluation", ab.print_report, slim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
