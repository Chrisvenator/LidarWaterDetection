"""Validate our water-vs-land labels against the INDEPENDENT surveyed reference
points (total-station + RTK-GNSS) from the Mandlburger Pielach dataset.

Unlike the classified LiDAR (itself algorithm output), these points are
human-measured ground truth:
  06 control_points          -> on land  -> expect NON-WATER
  07 underwater_targets      -> in river -> expect WATER
  08 underwater_transects    -> in river -> expect WATER

For each survey point we find our nearest classified point (3D, within a gate)
and check whether our label is on the correct side. Requires the coordinate
offset produced by validate_against_reference.py (validation_report.json).

Usage:
  python src/evaluation/validate_survey_points.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent.parent
REF_DIR = ROOT / "data" / "mandlburger_pielach_2024"
PRED_CSV = ROOT / "pointclouds" / "labeled_pointcloud_final.csv"
OUT_DIR = ROOT / "validation_results"
REPORT = OUT_DIR / "validation_report.json"

OUR_WATER_LABELS = (1, 3)               # water, water-under-canopy
MATCH_H = 0.75                          # m, horizontal gate (sparse cloud)
MATCH_V = 1.50                          # m, vertical gate (survey = surface/bed)

# file, delimiter, expected-is-water
SURVEY_FILES = (
    ("Pielach_2024102_control_points.csv", ";", False),
    ("Pielach_20241024_underwater_reference_targets.csv", ",", True),
    ("Pielach_20241024_underwater_transect_points.csv", ",", True),
)


def _num(s: object) -> float:
    """Parse a coordinate string, stripping unicode direction marks/spaces."""
    return float(re.sub(r"[^0-9.\-]", "", str(s)))


def load_survey(name: str, sep: str) -> np.ndarray:
    df = pd.read_csv(REF_DIR / name, sep=sep)
    df.columns = [c.strip() for c in df.columns]
    col = lambda key: next(c for c in df.columns if key in c)
    return np.column_stack([df[col("Easting")].map(_num),
                            df[col("Northing")].map(_num),
                            df[col("Height")].map(_num)])


def load_our_cloud(dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(PRED_CSV, usecols=["X", "Y", "Z", "final_label"])
    utm = np.column_stack([df["X"] + dx, df["Y"] + dy, df["Z"]])
    is_water = df["final_label"].isin(OUR_WATER_LABELS).to_numpy()
    return utm, is_water


def classify_survey(survey: np.ndarray, tree2: cKDTree, our_z: np.ndarray,
                    our_water: np.ndarray, expect_water: bool) -> dict:
    """Match each survey point to our nearest point within the gate."""
    dh, idx = tree2.query(survey[:, :2], k=1)
    zres = survey[:, 2] - our_z[idx]
    gated = (dh < MATCH_H) & (np.abs(zres) < MATCH_V)
    our_says_water = our_water[idx]
    correct = gated & (our_says_water == expect_water)
    return {"total": len(survey), "matched": int(gated.sum()),
            "correct": int(correct.sum()),
            "accuracy": float(correct.sum() / gated.sum()) if gated.sum() else 0.0,
            "_gated": gated, "_our_water": our_says_water, "_xy": survey[:, :2]}


def plot_survey(results: dict, utm: np.ndarray, is_water: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    w = utm[is_water]
    ax.scatter(w[:, 0], w[:, 1], s=1, c="#9ecae1", label="our water pts", rasterized=True)
    for name, r in results.items():
        g, xy, ours = r["_gated"], r["_xy"], r["_our_water"]
        ok = g & (ours == r["expect_water"])
        bad = g & (ours != r["expect_water"])
        marker = "s" if r["expect_water"] else "^"
        ax.scatter(xy[ok, 0], xy[ok, 1], s=55, marker=marker, c="#1a9641",
                   edgecolors="k", linewidths=0.4, label=f"{name}: correct")
        ax.scatter(xy[bad, 0], xy[bad, 1], s=55, marker=marker, c="#d7191c",
                   edgecolors="k", linewidths=0.4, label=f"{name}: WRONG")
    ax.set_aspect("equal")
    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    ax.set_title("Surveyed truth vs our labels  (square=expect-water, triangle=expect-land)")
    ax.legend(loc="upper left", fontsize=7, markerscale=1.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "survey_points_check.png", dpi=130)
    plt.close(fig)


def main() -> None:
    dx, dy = json.loads(REPORT.read_text())["offset"].values()
    utm, is_water = load_our_cloud(dx, dy)
    tree2 = cKDTree(utm[:, :2])

    results, tot, ok = {}, 0, 0
    print(f"offset dx={dx:.2f} dy={dy:.2f}\n")
    print(f"{'survey file':40s} {'expect':6s} {'n':>4s} {'matched':>8s} {'correct':>8s} {'acc':>6s}")
    for name, sep, expect_water in SURVEY_FILES:
        r = classify_survey(load_survey(name, sep), tree2, utm[:, 2], is_water, expect_water)
        r["expect_water"] = expect_water
        results[name.split("_")[-1].replace(".csv", "")] = r
        tot += r["matched"]
        ok += r["correct"]
        print(f"{name[:40]:40s} {'water' if expect_water else 'land':6s} "
              f"{r['total']:>4d} {r['matched']:>8d} {r['correct']:>8d} {r['accuracy']:>6.3f}")

    print(f"\nOVERALL surveyed-point accuracy: {ok}/{tot} = {ok/tot:.3f}" if tot else "no matches")
    plot_survey(results, utm, is_water)
    report = {"offset": {"dx": dx, "dy": dy},
              "overall": {"matched": tot, "correct": ok,
                          "accuracy": ok / tot if tot else None},
              "by_file": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                          for k, v in results.items()}}
    (OUT_DIR / "survey_validation_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {OUT_DIR/'survey_validation_report.json'}")
    print(f"plot   -> {OUT_DIR/'survey_points_check.png'}")


if __name__ == "__main__":
    main()
