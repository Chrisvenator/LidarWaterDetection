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

_POINT_SIZE = 0.4
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


def _densest_bin(z: np.ndarray, bin_m: float = 0.1) -> float:
    edges = np.arange(z.min(), z.max() + bin_m, bin_m)
    counts, _ = np.histogram(z, bins=edges)
    return float(edges[counts.argmax()] + 0.5 * bin_m)


def write_all(state: PipelineState, out_dir: Path,
              water_level: float | None = None) -> list[Path]:
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
    return written
