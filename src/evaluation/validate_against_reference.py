"""Validate this repo's water-vs-land labels against the Mandlburger et al.
Pielach reference survey (TU Wien, Oct 2024, DOI 10.48436/taz19-r6618).

Pipeline:
  1. Auto-register our local-offset cloud to the reference UTM cloud:
     FFT cross-correlation of the river-channel water mask (gross offset),
     then hard-surface point-coincidence refinement (sub-decimetre).
  2. Water EXTENT validation (primary): rasterise both clouds to 0.5 m
     water masks and compare cell-by-cell over the overlap -> IoU/F1.
     Robust to point-level non-coincidence of volumetric vegetation.
  3. Point-level validation (supporting): nearest reference point within a
     horizontal+vertical gate -> per-point confusion on coincident returns.
  4. Water-surface elevation vs the reference WSM GeoTIFF.
  5. Top-down plots so the water annotation can be inspected visually.

Reference truth (file 03, topo-bathy classified LAZ), ASPRS classes:
  water = {40 bathymetric bottom, 41 water surface, 43 submerged object}
  land  = {2 ground, 3/4/5 low/med/high vegetation}
  noise = {7, 18}  -> excluded

Our final_label: 0 land, 1 water, 2 uncertain, 3 water-under-canopy, 4 canopy.

Usage:
  python src/evaluation/validate_against_reference.py
"""
from __future__ import annotations

import json
from pathlib import Path

import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent.parent
REF_DIR = ROOT / "data" / "mandlburger_pielach_2024"
REF_LAZ = REF_DIR / "Pielach_20241024_topoBathy_LiDAR.laz"
REF_WSM = REF_DIR / "Pielach_20241024_WSM.tif"
PRED_CSV = ROOT / "pointclouds" / "labeled_pointcloud_final.csv"
OUT_DIR = ROOT / "validation_results"

REF_WATER_CLASSES = (40, 41, 43)
REF_LAND_CLASSES = (2, 3, 4, 5)
REF_HARD_CLASSES = (2, 40, 41, 43)      # non-volumetric surfaces for refine

OUR_WATER_LABELS = (1, 3)               # water, water-under-canopy
OUR_LAND_LABELS = (0, 4)                # land, canopy -> non-water
OUR_HARD_LABELS = (0, 1, 3)             # exclude canopy/uncertain for refine

CELL = 0.5                              # m, raster + registration resolution
REFINE_COARSE = 0.25                    # m, coincidence refine step
REFINE_FINE = 0.05                      # m
REFINE_SPAN = 2.0                       # m, +/- around FFT optimum
MATCH_H = 0.20                          # m, horizontal NN gate (point-level)
MATCH_V = 0.50                          # m, vertical gate (point-level)
WSM_NODATA_ABOVE = 1e30
MIN_CELL_POINTS = 1                     # cell occupied if >= this many points


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["X", "Y", "Z", "final_label", "local_surface_z"])
    return df.rename(columns={"X": "x", "Y": "y", "Z": "z"})


def load_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    las = laspy.read(path)
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    return xyz, np.asarray(las.classification, dtype=np.int16)


def occupancy(x: np.ndarray, y: np.ndarray, origin: tuple[float, float],
              shape: tuple[int, int]) -> np.ndarray:
    """Binary count-grid: 1 where >= MIN_CELL_POINTS land in the cell."""
    ox, oy = origin
    nrows, ncols = shape
    col = np.floor((x - ox) / CELL).astype(int)
    row = np.floor((y - oy) / CELL).astype(int)
    keep = (col >= 0) & (col < ncols) & (row >= 0) & (row < nrows)
    grid = np.zeros((nrows, ncols), dtype=np.int32)
    np.add.at(grid, (row[keep], col[keep]), 1)
    return (grid >= MIN_CELL_POINTS).astype(np.float64)


def _fft_offset(ux: np.ndarray, uy: np.ndarray,
                rx: np.ndarray, ry: np.ndarray) -> tuple[float, float]:
    """Gross (dx,dy) aligning user water mask to reference water mask."""
    ux0, uy0 = ux.min(), uy.min()
    unc = int(np.ceil((ux.max() - ux0) / CELL)) + 1
    unr = int(np.ceil((uy.max() - uy0) / CELL)) + 1
    rx0, ry0 = rx.min(), ry.min()
    rnc = int(np.ceil((rx.max() - rx0) / CELL)) + 1
    rnr = int(np.ceil((ry.max() - ry0) / CELL)) + 1
    a = occupancy(ux, uy, (ux0, uy0), (unr, unc))
    b = occupancy(rx, ry, (rx0, ry0), (rnr, rnc))
    corr = fftconvolve(b - b.mean(), (a - a.mean())[::-1, ::-1], mode="full")
    py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
    sr, sc = py - (unr - 1), px - (unc - 1)
    return (rx0 + sc * CELL) - ux0, (ry0 + sr * CELL) - uy0


def _coincidence(pts: np.ndarray, tree: cKDTree, ref_z: np.ndarray,
                 dx: float, dy: float) -> float:
    dh, idx = tree.query(pts[:, :2] + [dx, dy], k=1)
    zres = pts[:, 2] - ref_z[idx]
    return float(((dh < MATCH_H) & (np.abs(zres) < MATCH_V * 2)).mean())


def register(pred: pd.DataFrame, ref_pts: np.ndarray,
             ref_cls: np.ndarray) -> tuple[float, float]:
    """FFT channel-mask init + hard-surface coincidence refinement."""
    uw = pred["final_label"].isin(OUR_WATER_LABELS).to_numpy()
    rw = np.isin(ref_cls, REF_WATER_CLASSES)
    dx, dy = _fft_offset(pred["x"].to_numpy()[uw], pred["y"].to_numpy()[uw],
                         ref_pts[rw, 0], ref_pts[rw, 1])

    hard = pred["final_label"].isin(OUR_HARD_LABELS).to_numpy()
    uh = pred[["x", "y", "z"]].to_numpy()[hard]
    rh = ref_pts[np.isin(ref_cls, REF_HARD_CLASSES)]
    tree = cKDTree(rh[:, :2])
    rng = np.random.default_rng(0)
    sub = uh[rng.choice(len(uh), min(20000, len(uh)), replace=False)]

    best = (_coincidence(sub, tree, rh[:, 2], dx, dy), dx, dy)
    for step, span in ((REFINE_COARSE, REFINE_SPAN), (REFINE_FINE, 0.3)):
        cx, cy = best[1], best[2]
        for ddx in np.arange(cx - span, cx + span + step, step):
            for ddy in np.arange(cy - span, cy + span + step, step):
                s = _coincidence(sub, tree, rh[:, 2], ddx, ddy)
                if s > best[0]:
                    best = (s, ddx, ddy)
    print(f"  registration coincidence={best[0]:.3f}  dx={best[1]:.3f}  dy={best[2]:.3f}")
    return best[1], best[2]


def _common_frame(ax, ay, bx, by) -> tuple[tuple[float, float], tuple[int, int]]:
    ox, oy = min(ax.min(), bx.min()), min(ay.min(), by.min())
    ncols = int(np.ceil((max(ax.max(), bx.max()) - ox) / CELL)) + 1
    nrows = int(np.ceil((max(ay.max(), by.max()) - oy) / CELL)) + 1
    return (ox, oy), (nrows, ncols)


def water_extent(pred: pd.DataFrame, ref_pts: np.ndarray, ref_cls: np.ndarray,
                 dx: float, dy: float) -> tuple[dict, dict]:
    """Cell-wise water-mask agreement over the spatial overlap."""
    ux = pred["x"].to_numpy() + dx
    uy = pred["y"].to_numpy() + dy
    uw = pred["final_label"].isin(OUR_WATER_LABELS).to_numpy()
    rw = np.isin(ref_cls, REF_WATER_CLASSES)
    rl = np.isin(ref_cls, REF_LAND_CLASSES)

    origin, shape = _common_frame(ux, uy, ref_pts[:, 0], ref_pts[:, 1])
    our_any = occupancy(ux, uy, origin, shape)
    our_water = occupancy(ux[uw], uy[uw], origin, shape)
    ref_water = occupancy(ref_pts[rw, 0], ref_pts[rw, 1], origin, shape)
    ref_land = occupancy(ref_pts[rl, 0], ref_pts[rl, 1], origin, shape)
    ref_any = ((ref_water + ref_land) > 0).astype(np.float64)

    overlap = (our_any > 0) & (ref_any > 0)
    p = our_water[overlap] > 0
    t = ref_water[overlap] > 0
    tp, fp = int((p & t).sum()), int((p & ~t).sum())
    fn, tn = int((~p & t).sum()), int((~p & ~t).sum())
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    metrics = {"n_cells": int(overlap.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
               "iou": iou, "precision": prec, "recall": rec,
               "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
               "accuracy": (tp + tn) / overlap.sum() if overlap.sum() else 0.0}
    grids = {"origin": origin, "shape": shape, "overlap": overlap,
             "our_water": our_water, "ref_water": ref_water, "ref_any": ref_any}
    return metrics, grids


def point_level(pred: pd.DataFrame, ref_pts: np.ndarray, ref_cls: np.ndarray,
                dx: float, dy: float) -> dict:
    """Per-point confusion on coincident returns (horizontal+vertical gate)."""
    q = pred[["x", "y"]].to_numpy() + [dx, dy]
    tree = cKDTree(ref_pts[:, :2])
    dh, idx = tree.query(q, k=1)
    zres = pred["z"].to_numpy() - ref_pts[idx, 2]
    gate = (dh < MATCH_H) & (np.abs(zres) < MATCH_V)
    ref_water = np.isin(ref_cls[idx], REF_WATER_CLASSES)
    ref_land = np.isin(ref_cls[idx], REF_LAND_CLASSES)
    our_water = pred["final_label"].isin(OUR_WATER_LABELS).to_numpy()
    our_land = pred["final_label"].isin(OUR_LAND_LABELS).to_numpy()

    m = gate & (ref_water | ref_land) & (our_water | our_land)
    tp = int((m & our_water & ref_water).sum())
    fp = int((m & our_water & ref_land).sum())
    fn = int((m & our_land & ref_water).sum())
    tn = int((m & our_land & ref_land).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"n_points": int(m.sum()), "gate_rate": float(gate.mean()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
            "accuracy": (tp + tn) / m.sum() if m.sum() else 0.0}


def surface_vs_wsm(pred: pd.DataFrame, dx: float, dy: float) -> dict:
    water = pred[pred["final_label"].isin(OUR_WATER_LABELS)]
    coords = list(zip(water["x"] + dx, water["y"] + dy))
    with rasterio.open(REF_WSM) as ds:
        wsm = np.array([v[0] for v in ds.sample(coords)], dtype=np.float64)
    valid = np.isfinite(wsm) & (wsm < WSM_NODATA_ABOVE) & water["local_surface_z"].notna().values
    resid = water["local_surface_z"].values[valid] - wsm[valid]
    return {"n_water_points": int(valid.sum()),
            "resid_median": float(np.median(resid)),
            "resid_mean": float(np.mean(resid)),
            "resid_rmse": float(np.sqrt(np.mean(resid ** 2)))}


def plot_overlay(grids: dict, dx: float, dy: float) -> None:
    ox, oy = grids["origin"]
    extent = [ox, ox + grids["shape"][1] * CELL, oy, oy + grids["shape"][0] * CELL]
    ours, ref = grids["our_water"], grids["ref_water"]
    overlap = grids["overlap"]

    agree = np.zeros(ours.shape)            # 0 none,1 TP,2 FP(only us),3 FN(only ref)
    agree[(ours > 0) & (ref > 0)] = 1
    agree[(ours > 0) & (ref == 0) & overlap] = 2
    agree[(ours == 0) & (ref > 0) & overlap] = 3

    fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    for a, g, t in ((ax[0], ref, "Reference water (Mandlburger)"),
                    (ax[1], ours, "Our water labels")):
        a.imshow(np.ma.masked_where(g == 0, g), origin="lower", extent=extent,
                 cmap="Blues", vmin=0, vmax=1)
        a.set_title(t)
        a.set_aspect("equal")
    cmap = plt.matplotlib.colors.ListedColormap(["white", "#2c7fb8", "#d7191c", "#fdae61"])
    ax[2].imshow(agree, origin="lower", extent=extent, cmap=cmap, vmin=0, vmax=3)
    ax[2].set_title("Agreement: blue=both  red=only ours(FP)  orange=only ref(FN)")
    ax[2].set_aspect("equal")
    for a in ax:
        a.set_xlabel("Easting [m]")
    ax[0].set_ylabel("Northing [m]")
    fig.suptitle(f"Water extent vs reference  (offset dx={dx:.2f}, dy={dy:.2f})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "water_extent_overlay.png", dpi=130)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("loading predictions + reference ...")
    pred = load_predictions(PRED_CSV)
    ref_pts, ref_cls = load_reference(REF_LAZ)

    print("auto-registering ...")
    dx, dy = register(pred, ref_pts, ref_cls)

    print("water-extent validation ...")
    extent_m, grids = water_extent(pred, ref_pts, ref_cls, dx, dy)
    print("point-level validation ...")
    point_m = point_level(pred, ref_pts, ref_cls, dx, dy)
    surf_m = surface_vs_wsm(pred, dx, dy)
    plot_overlay(grids, dx, dy)

    report = {"reference": REF_LAZ.name, "predictions": PRED_CSV.name,
              "offset": {"dx": dx, "dy": dy},
              "water_extent_2d": extent_m, "point_level": point_m, "surface_vs_wsm": surf_m}
    (OUT_DIR / "validation_report.json").write_text(json.dumps(report, indent=2))

    print("\n=== WATER EXTENT (2D, 0.5 m cells over overlap) ===")
    print(f"  cells          : {extent_m['n_cells']:,}")
    print(f"  IoU (water)    : {extent_m['iou']:.4f}")
    print(f"  precision/recall: {extent_m['precision']:.3f} / {extent_m['recall']:.3f}")
    print(f"  F1 / accuracy  : {extent_m['f1']:.3f} / {extent_m['accuracy']:.3f}")
    print(f"  TP={extent_m['tp']:,} FP={extent_m['fp']:,} FN={extent_m['fn']:,} TN={extent_m['tn']:,}")
    print("\n=== POINT-LEVEL (coincident returns) ===")
    print(f"  evaluated pts  : {point_m['n_points']:,}  (gate rate {point_m['gate_rate']*100:.1f}%)")
    print(f"  IoU / F1       : {point_m['iou']:.4f} / {point_m['f1']:.4f}")
    print(f"  precision/recall: {point_m['precision']:.3f} / {point_m['recall']:.3f}")
    print("\n=== WATER SURFACE vs WSM.tif ===")
    print(f"  n={surf_m['n_water_points']:,}  median {surf_m['resid_median']:+.3f} m  RMSE {surf_m['resid_rmse']:.3f} m")
    print(f"\nreport -> {OUT_DIR/'validation_report.json'}")
    print(f"overlay plot -> {OUT_DIR/'water_extent_overlay.png'}")


if __name__ == "__main__":
    main()
