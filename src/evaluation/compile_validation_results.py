"""Run the full validation against the Mandlburger Pielach reference and dump
EVERYTHING into validation_results/ — one command, complete record.

Produces:
  reports/   validation_report.json, survey_validation_report.json, registration.json
  tables/    metrics_summary.csv, reference_class_histogram.csv, our_label_histogram.csv,
             confusion_*.csv, per_point_matches.csv, survey_matches.csv,
             margin_fn_diagnostic.csv
  plots/     water_extent_overlay.png, survey_points_check.png, margin_fn_hist.png
  README.md  human-readable summary of method + all results + caveats

Usage:
  python src/evaluation/compile_validation_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_against_reference as V          # noqa: E402
import validate_survey_points as S              # noqa: E402

OUT = V.ROOT / "validation_results"
REPORTS, TABLES, PLOTS = OUT / "reports", OUT / "tables", OUT / "plots"

REF_CLASS_NAMES = {2: "ground", 3: "low_veg", 4: "med_veg", 5: "high_veg",
                   7: "low_noise", 9: "water", 18: "high_noise",
                   40: "bathy_bottom", 41: "water_surface", 43: "submerged"}
OUR_LABEL_NAMES = {0: "land", 1: "water", 2: "uncertain",
                   3: "water_under_canopy", 4: "canopy"}
MARGIN_FEATURES = ["z", "reflectance_dB", "deep_proba", "z_above_surface", "local_surface_z"]


def _mapped(cls: int) -> str:
    if cls in V.REF_WATER_CLASSES:
        return "water"
    if cls in V.REF_LAND_CLASSES:
        return "land"
    return "ignore"


def write_class_histograms(ref_cls: np.ndarray, pred: pd.DataFrame) -> None:
    vals, cnts = np.unique(ref_cls, return_counts=True)
    pd.DataFrame({"class": vals, "name": [REF_CLASS_NAMES.get(v, "?") for v in vals],
                  "count": cnts, "pct": np.round(100 * cnts / cnts.sum(), 3),
                  "maps_to": [_mapped(v) for v in vals]}
                 ).to_csv(TABLES / "reference_class_histogram.csv", index=False)
    lc = pred["final_label"].value_counts().sort_index()
    pd.DataFrame({"label": lc.index, "name": [OUR_LABEL_NAMES.get(i, "?") for i in lc.index],
                  "count": lc.values, "pct": np.round(100 * lc.values / lc.sum(), 3)}
                 ).to_csv(TABLES / "our_label_histogram.csv", index=False)


def write_confusion(name: str, m: dict) -> None:
    pd.DataFrame([["pred_water", m["tp"], m["fp"]], ["pred_land", m["fn"], m["tn"]]],
                 columns=["", "truth_water", "truth_land"]
                 ).to_csv(TABLES / f"confusion_{name}.csv", index=False)


def per_point_matches(pred: pd.DataFrame, ref_pts: np.ndarray, ref_cls: np.ndarray,
                      dx: float, dy: float) -> pd.DataFrame:
    """Full per-point match table (gated, water/land truth available)."""
    q = pred[["x", "y"]].to_numpy() + [dx, dy]
    dh, idx = cKDTree(ref_pts[:, :2]).query(q, k=1)
    zres = pred["z"].to_numpy() - ref_pts[idx, 2]
    ref_water = np.isin(ref_cls[idx], V.REF_WATER_CLASSES)
    ref_land = np.isin(ref_cls[idx], V.REF_LAND_CLASSES)
    our_water = pred["final_label"].isin(V.OUR_WATER_LABELS).to_numpy()
    gate = (dh < V.MATCH_H) & (np.abs(zres) < V.MATCH_V) & (ref_water | ref_land)
    df = pd.DataFrame({
        "utm_x": q[gate, 0], "utm_y": q[gate, 1], "z": pred["z"].to_numpy()[gate],
        "our_label": pred["final_label"].to_numpy()[gate],
        "our_water": our_water[gate].astype(int),
        "ref_class": ref_cls[idx][gate],
        "ref_water": ref_water[gate].astype(int),
        "horiz_dist": np.round(dh[gate], 3), "z_resid": np.round(zres[gate], 3)})
    df["agree"] = (df["our_water"] == df["ref_water"]).astype(int)
    df.to_csv(TABLES / "per_point_matches.csv", index=False)
    return df


def margin_diagnostic(full: pd.DataFrame, matches: pd.DataFrame) -> None:
    """Compare features of FN (ref water, we said land) vs TP (both water)."""
    fn = matches[(matches.ref_water == 1) & (matches.our_water == 0)]
    tp = matches[(matches.ref_water == 1) & (matches.our_water == 1)]
    # join feature columns back via nearest matched X/Y is overkill; recompute by index
    feats = [c for c in MARGIN_FEATURES if c in full.columns or c == "z"]
    rows = []
    full_idx = cKDTree(full[["X", "Y"]].to_numpy() +
                       [matches.attrs["dx"], matches.attrs["dy"]])
    for tag, sub in (("FN", fn), ("TP", tp)):
        _, ii = full_idx.query(sub[["utm_x", "utm_y"]].to_numpy(), k=1)
        for f in feats:
            col = full[f].to_numpy()[ii] if f != "z" else sub["z"].to_numpy()
            rows.append({"group": tag, "feature": f, "n": len(sub),
                         "median": float(np.nanmedian(col)),
                         "mean": float(np.nanmean(col)),
                         "std": float(np.nanstd(col))})
    pd.DataFrame(rows).to_csv(TABLES / "margin_fn_diagnostic.csv", index=False)

    plot_feats = [f for f in ("deep_proba", "z_above_surface", "reflectance_dB") if f in full.columns]
    fig, ax = plt.subplots(1, len(plot_feats), figsize=(5 * len(plot_feats), 4))
    ax = np.atleast_1d(ax)
    for a, f in zip(ax, plot_feats):
        for tag, sub, c in (("FN (missed water)", fn, "#d7191c"), ("TP (got water)", tp, "#1a9641")):
            _, ii = full_idx.query(sub[["utm_x", "utm_y"]].to_numpy(), k=1)
            a.hist(full[f].to_numpy()[ii], bins=30, alpha=0.6, label=tag, color=c, density=True)
        a.set_title(f); a.legend(fontsize=8)
    fig.suptitle("Margin under-call: why FN water points were labelled land")
    fig.tight_layout()
    fig.savefig(PLOTS / "margin_fn_hist.png", dpi=130)
    plt.close(fig)


def run_survey(dx: float, dy: float) -> dict:
    utm, is_water = S.load_our_cloud(dx, dy)
    tree2 = cKDTree(utm[:, :2])
    results, rows, tot, ok = {}, [], 0, 0
    for name, sep, expect in S.SURVEY_FILES:
        survey = S.load_survey(name, sep)
        r = S.classify_survey(survey, tree2, utm[:, 2], is_water, expect)
        r["expect_water"] = expect
        results[name.split("_")[-1].replace(".csv", "")] = r
        tot += r["matched"]; ok += r["correct"]
        dh, idx = tree2.query(survey[:, :2], k=1)
        for i in range(len(survey)):
            rows.append({"file": name, "easting": survey[i, 0], "northing": survey[i, 1],
                         "height": survey[i, 2], "expect_water": int(expect),
                         "matched": int(r["_gated"][i]),
                         "our_water": int(is_water[idx[i]]), "horiz_dist": round(dh[i], 3)})
    pd.DataFrame(rows).to_csv(TABLES / "survey_matches.csv", index=False)
    S.plot_survey(results, utm, is_water)
    return {"overall": {"matched": tot, "correct": ok, "accuracy": ok / tot if tot else None},
            "by_file": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                        for k, v in results.items()}}


def write_summary_table(extent: dict, point: dict, survey: dict, surf: dict) -> None:
    rows = [
        {"metric_set": "classified_point_level", "n": point["n_points"],
         "precision": point["precision"], "recall": point["recall"],
         "f1": point["f1"], "iou": point["iou"], "accuracy": point["accuracy"]},
        {"metric_set": "classified_water_extent_2d", "n": extent["n_cells"],
         "precision": extent["precision"], "recall": extent["recall"],
         "f1": extent["f1"], "iou": extent["iou"], "accuracy": extent["accuracy"]},
        {"metric_set": "surveyed_points", "n": survey["overall"]["matched"],
         "precision": "", "recall": "", "f1": "", "iou": "",
         "accuracy": survey["overall"]["accuracy"]},
        {"metric_set": "water_surface_vs_wsm_m", "n": surf["n_water_points"],
         "precision": "", "recall": "", "f1": "",
         "iou": f"rmse={surf['resid_rmse']:.3f}", "accuracy": f"median={surf['resid_median']:+.3f}"},
    ]
    pd.DataFrame(rows).to_csv(TABLES / "metrics_summary.csv", index=False)


def write_readme(dx, dy, extent, point, survey, surf) -> None:
    txt = f"""# Validation Results — water-vs-land classifier

Reference: **Mandlburger et al., Pielach River, TU Wien, Oct 2024**
(DOI 10.48436/taz19-r6618, CC-BY 4.0) — same river, epoch, RIEGL topo-bathy sensor.

Predictions validated: `pointclouds/labeled_pointcloud_final.csv`
Regenerate everything: `python src/evaluation/compile_validation_results.py`

## Coordinate registration
Our cloud is local-offset in x,y (z already true ellipsoidal height).
Offset local -> ETRS89/UTM33N (EPSG:25833): **dx = {dx:.3f}, dy = {dy:.3f}**
(found by FFT cross-correlation of the river-channel mask + hard-surface refine;
water z-residual ~+0.1 m confirms datum alignment). Our AOI overlaps only the
eastern ~half of the reference AOI.

## Headline results (3 independent angles)
| truth source | metric | value |
|---|---|---|
| Classified LiDAR, point-level | F1 / IoU | {point['f1']:.3f} / {point['iou']:.3f} |
| Classified LiDAR, point-level | precision / recall | {point['precision']:.3f} / {point['recall']:.3f} |
| Classified LiDAR, 2D channel footprint | IoU / F1 | {extent['iou']:.3f} / {extent['f1']:.3f} |
| Surveyed points (human truth) | accuracy | {survey['overall']['accuracy']:.3f} ({survey['overall']['correct']}/{survey['overall']['matched']}) |
| Water surface vs WSM.tif | median / RMSE | {surf['resid_median']:+.3f} m / {surf['resid_rmse']:.3f} m |

## Truth class mapping (reference file 03, ASPRS)
water = {{40 bathy bottom, 41 water surface, 43 submerged}};
land = {{2 ground, 3/4/5 vegetation}}; noise {{7,18}} excluded.
Our labels: 0 land, 1 water, 2 uncertain, 3 water-under-canopy, 4 canopy.

## Key finding
Channel core is solid; we slightly UNDER-call water at shallow/gravel channel
margins (false-negatives cluster at the water edge — see plots/ and
tables/margin_fn_diagnostic.csv).

## Caveats
1. Overlap = eastern half of the reference AOI only (partial coverage).
2. Classified LiDAR is itself algorithm output (OWP+SVB), not human truth.
3. Surveyed points are independent human truth but sparse (46 within our coverage).

## Folder
- `reports/`  raw JSON metrics + registration
- `tables/`   CSVs: metrics summary, class histograms, confusion matrices,
              per-point matches, survey matches, margin diagnostic
- `plots/`    water-extent overlay, survey-point map, margin-FN histograms
"""
    (OUT / "README.md").write_text(txt)


def main() -> None:
    for d in (REPORTS, TABLES, PLOTS):
        d.mkdir(parents=True, exist_ok=True)

    pred = V.load_predictions(V.PRED_CSV)
    full = pd.read_csv(V.PRED_CSV)
    ref_pts, ref_cls = V.load_reference(V.REF_LAZ)

    print("registering ...")
    dx, dy = V.register(pred, ref_pts, ref_cls)
    print("computing metrics ...")
    extent_m, grids = V.water_extent(pred, ref_pts, ref_cls, dx, dy)
    point_m = V.point_level(pred, ref_pts, ref_cls, dx, dy)
    surf_m = V.surface_vs_wsm(pred, dx, dy)

    # reports
    (REPORTS / "registration.json").write_text(json.dumps(
        {"dx": dx, "dy": dy, "method": "fft_channel_mask + hard_surface_refine",
         "crs": "EPSG:25833", "overlap": "eastern half of reference AOI"}, indent=2))
    (REPORTS / "validation_report.json").write_text(json.dumps(
        {"offset": {"dx": dx, "dy": dy}, "water_extent_2d": extent_m,
         "point_level": point_m, "surface_vs_wsm": surf_m}, indent=2))

    # plots + tables
    V.plot_overlay(grids, dx, dy)
    write_class_histograms(ref_cls, pred)
    write_confusion("water_extent_2d", extent_m)
    write_confusion("point_level", point_m)
    matches = per_point_matches(pred, ref_pts, ref_cls, dx, dy)
    matches.attrs["dx"], matches.attrs["dy"] = dx, dy
    margin_diagnostic(full, matches)
    survey = run_survey(dx, dy)
    (REPORTS / "survey_validation_report.json").write_text(json.dumps(survey, indent=2))
    write_summary_table(extent_m, point_m, survey, surf_m)
    write_readme(dx, dy, extent_m, point_m, survey, surf_m)

    # relocate the two plots the imported funcs drop in OUT root
    for p in ("water_extent_overlay.png", "survey_points_check.png"):
        src = OUT / p
        if src.exists():
            src.replace(PLOTS / p)

    print(f"\nAll validation artifacts written under: {OUT}")
    for f in sorted(OUT.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(OUT)}  ({f.stat().st_size:,} B)")


if __name__ == "__main__":
    main()
