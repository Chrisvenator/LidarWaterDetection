"""
river_boundary.py — River boundary detection from WCN v9 predictions.

Approach
--------
Rather than wrapping a concave hull around classified points (which encloses
interior land on meanders and is sensitive to outliers), this module rasterizes
the continuous water-probability field onto a 2D grid, Gaussian-smooths it, and
extracts iso-probability contour lines at three thresholds:

  PROB_INNER (0.65) — inner / conservative boundary
                      Only where the model is quite confident it's water.
  PROB_CENTER (0.50) — central boundary (model decision threshold)
  PROB_OUTER (0.35) — outer / generous boundary
                      Includes the uncertain fringe.

The band between inner and outer is the uncertainty envelope — where the true
river edge lies. A narrow band means the transition is sharp (clear bank); a
wide band means the edge is gradual (gravel bar, shallow margin).

Run
---
    python src/labeling/river_boundary.py

Outputs  (models/wcn_v9/)
--------
  boundary_heatmap.png     probability field + contour lines (diagnostic)
  boundary_nocanopy.png    no-canopy scatter with contour overlay
  boundary.geojson         all three contours as GeoJSON LineStrings
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, distance_transform_edt

ROOT      = Path(__file__).resolve().parent.parent.parent
INPUT_CSV = ROOT / "pointclouds" / "labeled_pointcloud_wcn.csv"
OUT_DIR   = ROOT / "models" / "wcn_v9"

# ── Tunable parameters ────────────────────────────────────────────────────────
CELL_SIZE    = 0.5   # grid resolution in metres
SMOOTH_SIGMA = 1.5   # Gaussian σ in metres (controls contour smoothness)

PROB_INNER  = 0.65   # conservative inner boundary
PROB_CENTER = 0.50   # central / decision-threshold boundary
PROB_OUTER  = 0.35   # generous outer boundary

MIN_SEG_LEN  = 20    # discard contour segments shorter than this many points
MAX_CONTOURS =  3    # keep at most this many segments per level (largest first)

CANOPY_Z_MAX = 268.0  # z > this = canopy (CLAUDE.md)

LEVEL_STYLE = {
    PROB_INNER:  dict(color="#ffffff", lw=2.0, ls="-",  label=f"Inner  (p={PROB_INNER})"),
    PROB_CENTER: dict(color="#ffdd00", lw=2.0, ls="-",  label=f"Center (p={PROB_CENTER})"),
    PROB_OUTER:  dict(color="#ff8800", lw=2.0, ls="--", label=f"Outer  (p={PROB_OUTER})"),
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — RASTERIZE wcn_proba TO 2D GRID
# ══════════════════════════════════════════════════════════════════════════════

def rasterize(x: np.ndarray, y: np.ndarray, proba: np.ndarray):
    """
    Bin wcn_proba values into a (n_y, n_x) grid.
    Returns grid_proba (with NaN for empty cells), x_min, y_min, n_x, n_y.
    """
    x_min, y_min = float(x.min()), float(y.min())
    n_x = int(np.ceil((x.max() - x_min) / CELL_SIZE)) + 1
    n_y = int(np.ceil((y.max() - y_min) / CELL_SIZE)) + 1

    xi = np.clip(np.floor((x - x_min) / CELL_SIZE).astype(int), 0, n_x - 1)
    yi = np.clip(np.floor((y - y_min) / CELL_SIZE).astype(int), 0, n_y - 1)

    grid_sum = np.zeros((n_y, n_x), dtype=np.float64)
    grid_cnt = np.zeros((n_y, n_x), dtype=np.int32)
    np.add.at(grid_sum, (yi, xi), proba)
    np.add.at(grid_cnt, (yi, xi), 1)

    valid = grid_cnt > 0
    grid = np.where(valid, grid_sum / np.maximum(grid_cnt, 1), np.nan)

    n_empty = int((~valid).sum())
    print(f"  Grid: {n_x} × {n_y} = {n_x*n_y:,} cells  "
          f"({100*valid.mean():.1f}% filled, {n_empty:,} empty)")
    return grid, x_min, y_min, n_x, n_y


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — FILL EMPTY CELLS + SMOOTH
# ══════════════════════════════════════════════════════════════════════════════

def fill_and_smooth(grid: np.ndarray) -> np.ndarray:
    """
    Fill NaN cells via nearest-neighbour propagation from valid cells,
    then apply Gaussian smoothing.
    """
    nan_mask = np.isnan(grid)
    if nan_mask.any():
        _, nearest = distance_transform_edt(nan_mask, return_indices=True)
        filled = grid.copy()
        filled[nan_mask] = grid[nearest[0][nan_mask], nearest[1][nan_mask]]
    else:
        filled = grid

    sigma_cells = SMOOTH_SIGMA / CELL_SIZE
    smoothed = gaussian_filter(filled.astype(np.float32), sigma=sigma_cells)
    print(f"  Smoothed with σ={SMOOTH_SIGMA}m ({sigma_cells:.1f} cells)  "
          f"proba range: [{smoothed.min():.3f}, {smoothed.max():.3f}]")
    return smoothed


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — EXTRACT CONTOURS
# ══════════════════════════════════════════════════════════════════════════════

def _chaikin(pts: np.ndarray, n: int = 3) -> np.ndarray:
    """Chaikin corner-cutting: smooths a polyline without shrinking it much."""
    closed = np.allclose(pts[0], pts[-1])
    for _ in range(n):
        new = []
        for i in range(len(pts) - 1):
            new.append(0.75 * pts[i] + 0.25 * pts[i + 1])
            new.append(0.25 * pts[i] + 0.75 * pts[i + 1])
        pts = np.array(new)
        if closed:
            pts = np.vstack([pts, pts[0]])
    return pts


def extract_contours(
        grid: np.ndarray, x_min: float, y_min: float
) -> dict[float, list[np.ndarray]]:
    """
    Extract contour polylines at PROB_INNER / PROB_CENTER / PROB_OUTER.

    Returns dict: level → list of (N, 2) world-coordinate arrays,
    sorted largest-first, clipped to MAX_CONTOURS.
    """
    levels = [PROB_OUTER, PROB_CENTER, PROB_INNER]

    x_1d = x_min + (np.arange(grid.shape[1]) + 0.5) * CELL_SIZE
    y_1d = y_min + (np.arange(grid.shape[0]) + 0.5) * CELL_SIZE

    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(x_1d, y_1d, grid, levels=levels)
    plt.close(fig_tmp)

    result: dict[float, list[np.ndarray]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        all_segs = cs.allsegs   # list[list[ndarray(N,2)]] — one list per level

    for level, segs in zip(cs.levels, all_segs):
        long_segs = [s for s in segs if len(s) >= MIN_SEG_LEN]
        long_segs.sort(key=len, reverse=True)
        smoothed = [_chaikin(s) for s in long_segs[:MAX_CONTOURS]]
        result[float(level)] = smoothed
        total_pts = sum(len(s) for s in smoothed)
        print(f"  Level {level:.2f}: {len(segs)} raw segments → "
              f"{len(smoothed)} kept  ({total_pts:,} pts after smoothing)")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def _draw_contours(ax, contours: dict, levels=None) -> None:
    """Overlay all kept contour segments on ax."""
    for level, segs in contours.items():
        if levels is not None and level not in levels:
            continue
        style = LEVEL_STYLE.get(level, {})
        label = style.get("label")
        for seg in segs:
            ax.plot(seg[:, 0], seg[:, 1],
                    color=style.get("color", "k"),
                    lw=style.get("lw", 1.5),
                    ls=style.get("ls", "-"),
                    label=label, zorder=8)
            label = None   # only label the first segment per level


def plot_heatmap(grid: np.ndarray, x_min: float, y_min: float,
                 contours: dict, out_dir: Path) -> None:
    """Probability field heatmap with contour lines."""
    n_y, n_x = grid.shape
    extent = [x_min, x_min + n_x * CELL_SIZE,
              y_min, y_min + n_y * CELL_SIZE]

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(grid, origin="lower", extent=extent,
                   cmap="RdBu", vmin=0.0, vmax=1.0,
                   aspect="equal", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="wcn_proba (water probability)")
    _draw_contours(ax, contours)

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)

    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Water probability field  "
                 f"({CELL_SIZE}m grid, σ={SMOOTH_SIGMA}m Gaussian smooth)\n"
                 f"Blue = high water probability · Red = land")
    plt.tight_layout()
    p = out_dir / "boundary_heatmap.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved → {p}")


def plot_scatter(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                 ensemble: np.ndarray, contours: dict,
                 out_dir: Path) -> None:
    """No-canopy scatter plot with contour overlay (two panels: full + zoomed)."""
    cmap = {0: "saddlebrown", 1: "steelblue", 2: "gold"}
    lmap = {0: "Land", 1: "Water", 2: "Uncertain"}

    nc = z <= CANOPY_Z_MAX

    # Bounding box of outer contour for zoom panel
    outer_segs = contours.get(PROB_OUTER, [])
    if outer_segs:
        all_pts = np.vstack(outer_segs)
        pad = 5.0
        zoom_xlim = (all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
        zoom_ylim = (all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
    else:
        zoom_xlim = zoom_ylim = None

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    for ax, apply_zoom, title in [
        (axes[0], False, f"WCN v9 — no canopy  (z ≤ {CANOPY_Z_MAX}m)"),
        (axes[1], True,  f"WCN v9 — river boundary zone"),
    ]:
        xs, ys = x[nc], y[nc]
        ps     = ensemble[nc]
        for lv in [0, 1, 2]:
            m = ps == lv
            if m.any():
                ax.scatter(xs[m], ys[m], s=0.5, c=cmap[lv],
                           label=f"{lmap[lv]} ({m.sum():,})", rasterized=True)
        _draw_contours(ax, contours)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title(title)
        if apply_zoom and zoom_xlim:
            ax.set_xlim(*zoom_xlim); ax.set_ylim(*zoom_ylim)

        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),
                  markerscale=10, loc="upper right", fontsize=7)

    plt.suptitle(f"River boundary  —  inner p={PROB_INNER} (white) · "
                 f"center p={PROB_CENTER} (yellow) · outer p={PROB_OUTER} (orange)",
                 fontsize=11)
    plt.tight_layout()
    p = out_dir / "boundary_nocanopy.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — EXPORT GeoJSON
# ══════════════════════════════════════════════════════════════════════════════

def save_geojson(contours: dict, out_dir: Path) -> None:
    """Save all contour segments as a GeoJSON FeatureCollection."""
    level_names = {
        PROB_INNER:  "inner",
        PROB_CENTER: "center",
        PROB_OUTER:  "outer",
    }
    features = []
    for level, segs in sorted(contours.items(), reverse=True):
        name = level_names.get(level, f"p{level:.2f}")
        for i, seg in enumerate(segs):
            features.append({
                "type": "Feature",
                "properties": {
                    "level":    level,
                    "name":     name,
                    "segment":  i,
                    "n_points": len(seg),
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[round(float(p[0]), 4),
                                     round(float(p[1]), 4)] for p in seg],
                },
            })

    fc = {"type": "FeatureCollection", "features": features}
    p = out_dir / "boundary.geojson"
    with open(p, "w") as f:
        json.dump(fc, f, indent=2)
    print(f"  Saved → {p}  ({len(features)} features)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("River Boundary Detection — WCN v9")
    print(f"  cell={CELL_SIZE}m  σ={SMOOTH_SIGMA}m  "
          f"levels={PROB_INNER}/{PROB_CENTER}/{PROB_OUTER}")
    print("=" * 60)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"{INPUT_CSV}\nRun train_wcn_v9.py first.")

    print(f"\nLoading {INPUT_CSV.name} …")
    df = pd.read_csv(INPUT_CSV, usecols=["x", "y", "z", "wcn_proba", "ensemble"])
    print(f"  {len(df):,} points  "
          f"land={int((df.ensemble==0).sum()):,}  "
          f"water={int((df.ensemble==1).sum()):,}  "
          f"uncertain={int((df.ensemble==2).sum()):,}")

    x        = df["x"].values
    y        = df["y"].values
    z        = df["z"].values
    proba    = df["wcn_proba"].values
    ensemble = df["ensemble"].values.astype(np.int8)

    print("\nStep 1 — Rasterizing …")
    grid_raw, x_min, y_min, n_x, n_y = rasterize(x, y, proba)

    print("\nStep 2 — Filling + smoothing …")
    grid_smooth = fill_and_smooth(grid_raw)

    print("\nStep 3 — Extracting contours …")
    contours = extract_contours(grid_smooth, x_min, y_min)

    print("\nStep 4 — Plotting …")
    plot_heatmap(grid_smooth, x_min, y_min, contours, OUT_DIR)
    plot_scatter(x, y, z, ensemble, contours, OUT_DIR)

    print("\nStep 5 — Saving GeoJSON …")
    save_geojson(contours, OUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
