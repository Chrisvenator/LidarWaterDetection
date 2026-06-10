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
    python src/labeling/river_boundary.py                       # WCN v9 probas
    python src/labeling/river_boundary.py \
        --input pointclouds/labeled_pointcloud_final.csv \
        --out-dir models/final                                  # canopy-aware

With the final merged cloud, canopy points (label 4) are excluded as evidence —
they say nothing about the water below — and the gaps they leave are bridged by
nearest-neighbour fill. Water evidence: water/recon-water = 1.0, land = 0.0,
uncertain = deep_proba.

Outputs  (out dir)
--------
  boundary_heatmap.png     probability field + contour lines (diagnostic)
  boundary_nocanopy.png    no-canopy scatter with contour overlay
  boundary.geojson         all three contours as GeoJSON LineStrings
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.spatial import cKDTree

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

# final-cloud mode guards against phantom blobs at the survey fringe
MAX_FILL_DIST_M     = 3.0  # cells farther than this from data stay NaN (no contour)
ISOLATION_RADIUS_M  = 3.0  # water point needs water neighbors within this radius …
MIN_WATER_SUPPORT   = 5    # … at least this many (incl. itself) to count as evidence

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

def fill_and_smooth(grid: np.ndarray, max_dist_m: float | None = None) -> np.ndarray:
    """
    Fill NaN cells via nearest-neighbour propagation from valid cells,
    then apply Gaussian smoothing.

    max_dist_m: if set, cells farther than this from any data cell are reset
    to NaN after smoothing, so contours cannot wander into no-data margins.
    """
    nan_mask = np.isnan(grid)
    if nan_mask.any():
        dist, nearest = distance_transform_edt(nan_mask, return_indices=True)
        filled = grid.copy()
        filled[nan_mask] = grid[nearest[0][nan_mask], nearest[1][nan_mask]]
    else:
        dist = np.zeros_like(grid)
        filled = grid

    sigma_cells = SMOOTH_SIGMA / CELL_SIZE
    smoothed = gaussian_filter(filled.astype(np.float32), sigma=sigma_cells)
    if max_dist_m is not None:
        far = dist * CELL_SIZE > max_dist_m
        smoothed[far] = np.nan
        print(f"  Masked {int(far.sum()):,} cells farther than {max_dist_m}m from data")
    print(f"  Smoothed with σ={SMOOTH_SIGMA}m ({sigma_cells:.1f} cells)  "
          f"proba range: [{np.nanmin(smoothed):.3f}, {np.nanmax(smoothed):.3f}]")
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


def plot_scatter(x: np.ndarray, y: np.ndarray, nc: np.ndarray,
                 labels: np.ndarray, contours: dict,
                 out_dir: Path, tag: str) -> None:
    """No-canopy scatter plot with contour overlay (two panels: full + zoomed)."""
    cmap = {0: "saddlebrown", 1: "steelblue", 2: "gold", 3: "navy"}
    lmap = {0: "Land", 1: "Water", 2: "Uncertain", 3: "Water under canopy"}

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
        (axes[0], False, f"{tag} — no canopy"),
        (axes[1], True,  f"{tag} — river boundary zone"),
    ]:
        xs, ys = x[nc], y[nc]
        ps     = labels[nc]
        for lv in [0, 1, 2, 3]:
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

        # Deduplicate legend (NB: don't shadow the labels parameter)
        handles, leg_labels = ax.get_legend_handles_labels()
        by_label = dict(zip(leg_labels, handles))
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

def isolated_water(x: np.ndarray, y: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Water points without enough water neighbors — reconstruction strays.

    A lone water-evidence point at the survey fringe otherwise seeds a phantom
    blob via nearest-neighbour fill (one bogus label-3 point did exactly that).
    """
    water = np.isin(labels, (1, 3))
    xy = np.column_stack([x[water], y[water]])
    support = cKDTree(xy).query_ball_point(xy, ISOLATION_RADIUS_M, workers=-1,
                                           return_length=True)
    out = np.zeros(len(labels), bool)
    out[np.flatnonzero(water)[support < MIN_WATER_SUPPORT]] = True
    if out.any():
        print(f"  Dropped {int(out.sum())} isolated water points "
              f"(<{MIN_WATER_SUPPORT} water neighbors within {ISOLATION_RADIUS_M}m)")
    return out


def load_evidence(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                       np.ndarray, np.ndarray, str]:
    """Return x, y, water evidence, class labels, evidence mask, plot tag.

    Final merged cloud: canopy and isolated water strays are masked out of the
    evidence; water/recon-water = 1.0, land = 0.0, uncertain keeps its
    deep_proba. WCN cloud: original behavior (wcn_proba, z-based canopy).
    """
    if "final_label" in pd.read_csv(path, nrows=0).columns:
        df = pd.read_csv(path, usecols=["X", "Y", "final_label", "deep_proba"])
        labels = df["final_label"].to_numpy(np.int8)
        evidence = np.select([np.isin(labels, (1, 3)), labels == 0],
                             [1.0, 0.0], default=df["deep_proba"].to_numpy())
        use = (labels != 4) & ~isolated_water(
            df["X"].to_numpy(), df["Y"].to_numpy(), labels)
        return (df["X"].to_numpy(), df["Y"].to_numpy(), evidence, labels,
                use, "final (v10 + canopy)")
    df = pd.read_csv(path, usecols=["x", "y", "z", "wcn_proba", "ensemble"])
    return (df["x"].to_numpy(), df["y"].to_numpy(),
            df["wcn_proba"].to_numpy(), df["ensemble"].to_numpy(np.int8),
            (df["z"] <= CANOPY_Z_MAX).to_numpy(), "WCN v9")


def main():
    ap = argparse.ArgumentParser(description="River boundary from proba contours")
    ap.add_argument("--input", type=Path, default=INPUT_CSV)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"River Boundary Detection — {args.input.name}")
    print(f"  cell={CELL_SIZE}m  σ={SMOOTH_SIGMA}m  "
          f"levels={PROB_INNER}/{PROB_CENTER}/{PROB_OUTER}")
    print("=" * 60)

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    x, y, evidence, labels, use, tag = load_evidence(args.input)
    counts = {v: int((labels == v).sum()) for v in np.unique(labels)}
    print(f"  {len(x):,} points  labels={counts}")

    # canopy/stray points carry no water evidence — their cells are bridged by fill
    final_mode = tag.startswith("final")
    keep = use if final_mode else np.ones(len(x), bool)

    print("\nStep 1 — Rasterizing …")
    grid_raw, x_min, y_min, n_x, n_y = rasterize(x[keep], y[keep], evidence[keep])

    print("\nStep 2 — Filling + smoothing …")
    grid_smooth = fill_and_smooth(grid_raw,
                                  max_dist_m=MAX_FILL_DIST_M if final_mode else None)

    print("\nStep 3 — Extracting contours …")
    contours = extract_contours(grid_smooth, x_min, y_min)

    print("\nStep 4 — Plotting …")
    plot_heatmap(grid_smooth, x_min, y_min, contours, args.out_dir)
    plot_scatter(x, y, use, labels, contours, args.out_dir, tag)

    print("\nStep 5 — Saving GeoJSON …")
    save_geojson(contours, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
