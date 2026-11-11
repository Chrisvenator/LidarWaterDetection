"""Diagnostic figures for a classified point cloud.

Written per run into ``runs/<dataset>/plots/``. matplotlib is imported
lazily so the library stays usable without it.

Every figure is small multiples — one class or one model per panel. A dense
scatter puts all classes in one visual field at once, and no five-hue
categorical palette clears the all-pairs colour-vision floors (measured:
normal-vision Delta E 12.9, against a floor of 15). Faceting also removes the
overplotting that makes a 230k-point five-colour map unreadable regardless.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import PipelineState

# Categorical slots, assigned in fixed order and never cycled. One class per
# panel, so identity comes from the panel title, never from colour alone.
CLASS_NAMES = {0: "land", 1: "water", 2: "uncertain", 3: "water under canopy", 4: "canopy"}
CLASS_COLORS = {0: "#eb6834", 1: "#2a78d6", 2: "#eda100", 3: "#1baf7a", 4: "#008300"}

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
CONTEXT = "#e2e1dd"          # recessive "all other points" layer
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# P(water) has a meaningful midpoint — 0.5 is the decision boundary — so the
# field is diverging, not sequential: blue for water, red for land, neutral
# grey where the model is undecided. Equal step count per arm.
DIVERGING_WATER = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#9ec5f4"]
DIVERGING_MID = "#f0efec"
DIVERGING_LAND = ["#f4b0af", "#ec7a79", "#e34948", "#b62f2e", "#7d1f1e"]

# Contour levels are distinguished by line style as well as position, so the
# three are still separable in print and for colour-vision deficiency.
CONTOUR_STYLES = {"inner": (0, ()), "center": (0, (6, 3)), "outer": (0, (2, 3))}

_POINT_SIZE = 0.4
_POINT_SIZE_SINGLE = 1.2     # single full-width panel; faceted panels stay smaller
_CROSS_SECTION_M = 6.0       # thickness of the y-slice drawn in cross-section


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=8)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=8)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=7)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(CONTEXT)


def _facet_grid(plt, n_panels: int, width: float, height: float):
    n_cols = min(3, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width * n_cols, height * n_rows),
                             facecolor=SURFACE, squeeze=False)
    return fig, axes.ravel()


def _finish(fig, axes, n_used: int, path: Path, suptitle: str) -> Path:
    for ax in axes[n_used:]:
        ax.set_visible(False)
    fig.suptitle(suptitle, color=TEXT_PRIMARY, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    _pyplot().close(fig)
    return path


def _class_facets(labels: np.ndarray, horiz: np.ndarray, vert: np.ndarray, path: Path,
                  suptitle: str, xlabel: str, ylabel: str, equal_aspect: bool) -> Path:
    """One panel per class: that class in its own hue over a grey context layer."""
    plt = _pyplot()
    present = [c for c in sorted(CLASS_NAMES) if (labels == c).any()]
    fig, axes = _facet_grid(plt, len(present), 4.6, 3.6)

    for ax, cls in zip(axes, present):
        mask = labels == cls
        ax.scatter(horiz, vert, s=_POINT_SIZE, c=CONTEXT, linewidths=0, rasterized=True)
        ax.scatter(horiz[mask], vert[mask], s=_POINT_SIZE, c=CLASS_COLORS[cls],
                   linewidths=0, rasterized=True)
        share = mask.mean()
        _style_axes(ax, f"{CLASS_NAMES[cls]}  —  {mask.sum():,} pts ({share:.1%})", xlabel, ylabel)
        if equal_aspect:
            ax.set_aspect("equal")
    return _finish(fig, axes, len(present), path, suptitle)


def plot_class_map(state: PipelineState, out_dir: Path) -> Path:
    return _class_facets(
        state.final_label, state.cloud.x, state.cloud.y,
        out_dir / "classes_topdown.png",
        "Classified point cloud — top-down, one class per panel",
        "x (m)", "y (m)", equal_aspect=True)


def plot_cross_section(state: PipelineState, out_dir: Path) -> Path:
    """Elevation against x for a slice through the middle of the survey."""
    y = state.cloud.y
    centre = float(np.median(y))
    sl = np.abs(y - centre) < _CROSS_SECTION_M / 2
    return _class_facets(
        state.final_label[sl], state.cloud.x[sl], state.cloud.z[sl],
        out_dir / "cross_section.png",
        f"Cross-section, {_CROSS_SECTION_M:g} m slice at y = {centre:.1f} m",
        "x (m)", "z (m)", equal_aspect=False)


def plot_elevation_profile(state: PipelineState, out_dir: Path,
                           water_level: float | None = None) -> Path:
    """Height above the water surface, per class — the quickest way to see a
    class sitting at a physically wrong elevation."""
    plt = _pyplot()
    z = state.cloud.z
    if water_level is None:
        water_level = _densest_bin(z)
    height = z - water_level
    labels = state.final_label
    present = [c for c in sorted(CLASS_NAMES) if (labels == c).any()]
    bins = np.linspace(np.percentile(height, 0.5), np.percentile(height, 99.5), 80)

    fig, axes = _facet_grid(plt, len(present), 4.6, 2.2)
    for ax, cls in zip(axes, present):
        ax.hist(height[labels == cls], bins=bins, color=CLASS_COLORS[cls], linewidth=0)
        ax.axvline(0.0, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
        ax.annotate("water level", (0.0, ax.get_ylim()[1]), xytext=(3, -8),
                    textcoords="offset points", color=TEXT_SECONDARY, fontsize=7)
        _style_axes(ax, CLASS_NAMES[cls], "height above water surface (m)", "points")
    return _finish(fig, axes, len(present), out_dir / "elevation_profile.png",
                   f"Elevation distribution per class (water level {water_level:.2f} m)")


def plot_water_probability(state: PipelineState, out_dir: Path) -> Path:
    """The two water heads side by side on one sequential ramp — a spatial
    view of how far apart they are."""
    from matplotlib.colors import LinearSegmentedColormap

    plt = _pyplot()
    heads = [(name, proba) for name, proba in
             (("WCN transformer", state.wcn_proba), ("XGBoost head", state.wcn_xgb_proba))
             if proba is not None]
    if not heads:
        raise ValueError("no water probabilities on the state — run the WCN stage first")

    cmap = LinearSegmentedColormap.from_list("water", SEQUENTIAL)
    fig, axes = _facet_grid(plt, len(heads), 5.2, 4.0)
    for ax, (name, proba) in zip(axes, heads):
        dots = ax.scatter(state.cloud.x, state.cloud.y, s=_POINT_SIZE, c=proba,
                          cmap=cmap, vmin=0, vmax=1, linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        _style_axes(ax, f"{name}  —  {(proba >= 0.5).mean():.1%} water", "x (m)", "y (m)")
        bar = fig.colorbar(dots, ax=ax, fraction=0.035)
        bar.set_label("P(water)", color=TEXT_SECONDARY, fontsize=8)
        bar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=7)
    return _finish(fig, axes, len(heads), out_dir / "water_probability.png",
                   "Water probability by model head")


def _diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "land_water", [*DIVERGING_LAND[::-1], DIVERGING_MID, *DIVERGING_WATER[::-1]])


def _contour_levels(config) -> dict:
    return {"outer": config.prob_outer, "center": config.prob_center,
            "inner": config.prob_inner}


def _draw_contours(ax, state, config) -> bool:
    """Overlay the boundary contours; returns whether anything was drawn."""
    drawn = False
    for name, level in _contour_levels(config).items():
        for i, seg in enumerate(state.boundary_contours.get(level, [])):
            ax.plot(seg[:, 0], seg[:, 1], color=TEXT_PRIMARY, linewidth=1.4,
                    linestyle=CONTOUR_STYLES[name],
                    label=f"P = {level:.2f} ({name})" if i == 0 else None)
            drawn = True
    return drawn


def plot_boundary_heatmap(state: PipelineState, out_dir: Path, config) -> Path:
    """The probability field the river boundary is cut from, with the cuts."""
    plt = _pyplot()
    if state.boundary_field is None:
        raise ValueError("no boundary field on the state — run the boundary stage first")

    field = state.boundary_field
    x_min, y_min = state.boundary_field_origin
    n_y, n_x = field.shape
    cell = config.cell_size_m
    extent = [x_min, x_min + n_x * cell, y_min, y_min + n_y * cell]

    fig, ax = plt.subplots(figsize=(15, 8), facecolor=SURFACE)
    im = ax.imshow(field, origin="lower", extent=extent, cmap=_diverging_cmap(),
                   vmin=0.0, vmax=1.0, aspect="equal", interpolation="nearest")
    bar = fig.colorbar(im, ax=ax, fraction=0.03)
    bar.set_label("P(water)   —   blue water, red land", color=TEXT_SECONDARY, fontsize=8)
    bar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=7)

    if _draw_contours(ax, state, config):
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    _style_axes(ax, "", "x (m)", "y (m)")
    return _finish(fig, np.array([ax]), 1, out_dir / "boundary_heatmap.png",
                   f"Water probability field and extracted boundary — {cell:g} m cells, "
                   f"{config.smooth_sigma_m:g} m smoothing")


def plot_boundary_overlay(state: PipelineState, out_dir: Path, config) -> Path:
    """Classified points under the contours, to see where the cut lands."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(15, 8), facecolor=SURFACE)
    labels = state.final_label

    for cls in sorted(CLASS_NAMES):
        mask = labels == cls
        if not mask.any():
            continue
        ax.scatter(state.cloud.x[mask], state.cloud.y[mask], s=_POINT_SIZE_SINGLE,
                   c=CLASS_COLORS[cls], linewidths=0, rasterized=True,
                   label=f"{CLASS_NAMES[cls]} ({mask.mean():.1%})")
    _draw_contours(ax, state, config)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7, markerscale=12, framealpha=0.9)
    _style_axes(ax, "", "x (m)", "y (m)")
    return _finish(fig, np.array([ax]), 1, out_dir / "boundary_overlay.png",
                   "Classified points with the extracted river boundary")


def _densest_bin(z: np.ndarray, bin_m: float = 0.1) -> float:
    edges = np.arange(z.min(), z.max() + bin_m, bin_m)
    counts, _ = np.histogram(z, bins=edges)
    return float(edges[counts.argmax()] + 0.5 * bin_m)


def write_all(state: PipelineState, out_dir: Path,
              water_level: float | None = None,
              boundary_config=None) -> list[Path]:
    """Every diagnostic figure the state supports, skipping those whose
    inputs a partial stage set did not produce."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if state.final_label is not None:
        written += [plot_class_map(state, out_dir),
                    plot_cross_section(state, out_dir),
                    plot_elevation_profile(state, out_dir, water_level)]
    if state.wcn_proba is not None or state.wcn_xgb_proba is not None:
        written.append(plot_water_probability(state, out_dir))
    if state.boundary_contours is not None and boundary_config is not None:
        if state.boundary_field is not None:
            written.append(plot_boundary_heatmap(state, out_dir, boundary_config))
        if state.final_label is not None:
            written.append(plot_boundary_overlay(state, out_dir, boundary_config))
    return written
