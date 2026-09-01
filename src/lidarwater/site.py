"""Automatic per-site configuration.

Every threshold in :mod:`lidarwater.config` was tuned on the Pielach study
area, and a fair number of them are absolute metres above sea level or
absolute reflectance decibels — meaningless at any other survey. Rather than
demanding a hand-tuned config per site, :func:`derive_site_config` measures
the few properties those thresholds actually depend on and rebases the
defaults onto them:

* **water level** — the densest elevation bin, i.e. the flat water surface.
  Every absolute z threshold is re-expressed as its offset from Pielach's
  water level and re-applied at the new one.
* **reflectance scale** — the -15 dB water gate admits the lowest ~86% of
  Pielach's reflectances; the same *percentile* is taken on the new cloud,
  because scanner gain and range normalisation shift the dB scale bodily.
* **waveform record type** — SVB-clustered records start at the first echo,
  full-record digitisations start hundreds of samples of noise earlier.
  Measured from where each waveform's energy actually sits.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ._stages.canopy import _cell_aggregate, _fill_holes
from ._stages.features import ENERGY_CONCENTRATION_BINS, _waveform_to_grid
from .config import (
    GRID_ORIGIN_FIRST_RETURN,
    GRID_ORIGIN_FIRST_SAMPLE,
    FeatureConfig,
    PipelineConfig,
)
from .types import PointCloud

# The Pielach cloud all library defaults were tuned against. Absolute z
# thresholds are stored as offsets from this level, absolute reflectance
# thresholds as percentiles of this cloud's reflectance distribution.
PIELACH_WATER_Z = 260.29
# The -15 dB water gate admits the lowest 86.4% of Pielach's reflectances.
# Scanner gain shifts the dB scale bodily, so the same percentile on a new
# cloud gives the offset applied to every absolute reflectance threshold.
PIELACH_REFLECTANCE_GATE_DB = -15.0
PIELACH_REFLECTANCE_GATE_PERCENTILE = 86.4
# The 0.85 compact-waveform gate passes 48.6% of the Pielach cloud. The
# feature counts energy in a fixed 30-bin window sized to Pielach's pulse,
# so on a site whose returns span more bins no point can ever reach 0.85 —
# the gate is rebased to the value passing the same share of the cloud.
PIELACH_ENERGY_GATE = 0.85
PIELACH_ENERGY_GATE_PERCENTILE = 51.43

Z_HISTOGRAM_BIN_M = 0.10
GROUND_CELL_M = 1.0
GROUND_PERCENTILE = 0.05   # fraction, matching CanopyConfig.dtm_percentile
CANOPY_PROBE_M = 3.0
# Below this share of waveform energy inside the first grid_size bins, the
# record is a full digitisation and the grid must be anchored on the return.
FIRST_SAMPLE_ENERGY_MIN = 0.50
ENERGY_PROBE_WAVEFORMS = 2000

# Config fields holding an absolute elevation, as (nested config attribute,
# field name). Shifted as a block when the site's water level moves.
_Z_FIELDS: tuple[tuple[str, str], ...] = (
    ("zones", "z_underwater_max"), ("zones", "z_water_surf_max"),
    ("zones", "z_dry_bed_min"), ("zones", "z_dry_bed_max"),
    ("zones", "z_banks_min"), ("zones", "z_banks_max"), ("zones", "z_canopy_min"),
    ("surface.footprint", "riverbed_z_max"), ("surface.footprint", "riverbed_z_surface_max"),
    ("surface.surface_grid", "z_lo"), ("surface.surface_grid", "z_hi"),
    ("surface.surface_grid", "z_cap"), ("surface.surface_grid", "ransac_z_lo"),
    ("surface.surface_grid", "ransac_z_hi"),
    ("boundary", "canopy_z_max"),
    ("canopy", "z_canopy_min"), ("canopy", "z_clear_max"),
)

# Config fields holding an absolute reflectance in dB, rebased by percentile.
_REFLECTANCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("surface.surface_grid", "reflectance_max_db"),
    ("surface.surface_grid", "ransac_reflectance_max_db"),
    ("surface.bed", "reflectance_max_db"),
)

# Site-dependent fields derived individually rather than by a block shift.
_OTHER_DERIVED_FIELDS: tuple[tuple[str, str], ...] = (
    ("surface.surface_grid", "energy_concentration_min"),
    ("features", "grid_origin"),
)

# Everything derive_site_config() touches — the coverage test asserts no
# site-dependent config field is missing from this set.
DERIVED_FIELDS = _Z_FIELDS + _REFLECTANCE_FIELDS + _OTHER_DERIVED_FIELDS


@dataclasses.dataclass(frozen=True)
class SiteProfile:
    """What was measured off a cloud, and what it implied for the config."""

    n_points: int
    water_level_z: float
    z_shift_m: float
    reflectance_shift_db: float
    grid_origin: str
    first_bin_energy_fraction: float
    energy_concentration_gate: float
    canopy_fraction_above_probe: float
    canopy_expected: bool

    def summary(self) -> str:
        return "\n".join([
            f"points                  {self.n_points:,}",
            f"water level (densest z) {self.water_level_z:.2f} m",
            f"z shift vs Pielach      {self.z_shift_m:+.2f} m",
            f"reflectance shift       {self.reflectance_shift_db:+.1f} dB",
            f"waveform grid origin    {self.grid_origin} "
            f"(energy in first bins {self.first_bin_energy_fraction:.1%})",
            f"compact-waveform gate   {self.energy_concentration_gate:.3f} "
            f"(Pielach {PIELACH_ENERGY_GATE})",
            f"canopy expected         {self.canopy_expected} "
            f"({self.canopy_fraction_above_probe:.2%} of points >{CANOPY_PROBE_M:g} m above ground)",
        ])


def estimate_water_level(z: np.ndarray, bin_m: float = Z_HISTOGRAM_BIN_M) -> float:
    """Centre of the densest elevation bin — a flat water surface dominates
    the z histogram of any river survey."""
    edges = np.arange(z.min(), z.max() + bin_m, bin_m)
    counts, _ = np.histogram(z, bins=edges)
    return float(edges[counts.argmax()] + 0.5 * bin_m)


def height_above_ground(xyz: np.ndarray, cell_m: float = GROUND_CELL_M,
                        percentile: float = GROUND_PERCENTILE) -> np.ndarray:
    """Height above a coarse per-cell ground surface (low z percentile per
    cell, holes filled from the nearest populated cell).

    Deliberately cruder than the canopy stage's DTM — no water-surface
    correction, no smoothing — because it runs before any stage, to decide
    whether the canopy stage is worth running at all.
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    ix = ((x - x.min()) / cell_m).astype(int)
    iy = ((y - y.min()) / cell_m).astype(int)
    shape = (ix.max() + 1, iy.max() + 1)

    ground = _fill_holes(_cell_aggregate(ix, iy, z, shape, percentile))
    return z - ground[ix, iy]


def _iter_probe_waveforms(cloud: PointCloud, n_probe: int):
    """Well-formed waveforms sampled evenly across the whole cloud.

    Not the first ``n_probe`` rows: a survey file's leading rows are one
    spatial cluster, so statistics taken from them do not describe the site.
    """
    n = len(cloud)
    indices = (np.arange(n) if n <= n_probe
               else np.linspace(0, n - 1, n_probe).astype(np.int64))
    for i in indices:
        times, amps = cloud.waveform(int(i))
        if len(times) and len(times) == len(amps):
            yield times, amps


def first_bin_energy_fraction(cloud: PointCloud, grid_size: int,
                              n_probe: int = ENERGY_PROBE_WAVEFORMS) -> float:
    """Median share of waveform energy landing in the first ``grid_size``
    samples when the grid is anchored at times[0]."""
    fractions = [
        float(amps[(times - times[0]) < grid_size].sum()) / float(amps.sum())
        for times, amps in _iter_probe_waveforms(cloud, n_probe)
        if amps.sum() > 0
    ]
    return float(np.median(fractions)) if fractions else 1.0


def energy_concentration_gate(cloud: PointCloud, features: FeatureConfig,
                              n_probe: int = ENERGY_PROBE_WAVEFORMS) -> float:
    """This site's equivalent of Pielach's 0.85 compact-waveform gate.

    Needs ``features.grid_origin`` already decided — the grid anchoring
    changes which bins the energy lands in.
    """
    grids = (_waveform_to_grid(times, amps, features)
             for times, amps in _iter_probe_waveforms(cloud, n_probe))
    values = [g[:ENERGY_CONCENTRATION_BINS].sum() / g.sum() for g in grids if g.sum() > 0]
    if not values:
        return PIELACH_ENERGY_GATE
    return float(np.percentile(values, PIELACH_ENERGY_GATE_PERCENTILE))


def profile_cloud(cloud: PointCloud, base: PipelineConfig) -> SiteProfile:
    """Measure everything :func:`derive_site_config` needs off a cloud."""
    water_z = estimate_water_level(cloud.xyz[:, 2])
    energy_fraction = first_bin_energy_fraction(cloud, base.features.grid_size)
    above_probe = float(
        (height_above_ground(cloud.xyz) > CANOPY_PROBE_M).mean())
    new_gate = float(np.percentile(cloud.reflectance_db,
                                   PIELACH_REFLECTANCE_GATE_PERCENTILE))
    grid_origin = (GRID_ORIGIN_FIRST_SAMPLE if energy_fraction >= FIRST_SAMPLE_ENERGY_MIN
                   else GRID_ORIGIN_FIRST_RETURN)
    energy_gate = energy_concentration_gate(
        cloud, dataclasses.replace(base.features, grid_origin=grid_origin))
    return SiteProfile(
        n_points=len(cloud),
        water_level_z=water_z,
        z_shift_m=water_z - PIELACH_WATER_Z,
        reflectance_shift_db=new_gate - PIELACH_REFLECTANCE_GATE_DB,
        grid_origin=grid_origin,
        first_bin_energy_fraction=energy_fraction,
        energy_concentration_gate=energy_gate,
        canopy_fraction_above_probe=above_probe,
        canopy_expected=above_probe >= base.canopy.min_canopy_frac,
    )


def derive_site_config(cloud: PointCloud, base: PipelineConfig | None = None,
                       ) -> tuple[PipelineConfig, SiteProfile]:
    """Rebase ``base`` (default: the Pielach-tuned defaults) onto ``cloud``.

    Returns the adapted config and the :class:`SiteProfile` explaining every
    adjustment. Only site-dependent thresholds move — model architecture,
    training hyperparameters and dimensionless ratios are untouched.
    """
    base = base or PipelineConfig()
    profile = profile_cloud(cloud, base)

    config = _shift_fields(base, _Z_FIELDS, profile.z_shift_m)
    config = _shift_fields(config, _REFLECTANCE_FIELDS, profile.reflectance_shift_db)
    config = _replace_nested(config, "surface.surface_grid",
                             energy_concentration_min=profile.energy_concentration_gate)
    return _replace_nested(config, "features", grid_origin=profile.grid_origin), profile


def _shift_fields(config: PipelineConfig, fields: tuple[tuple[str, str], ...],
                  delta: float) -> PipelineConfig:
    """Add ``delta`` to each listed field, grouped so every nested dataclass
    is rebuilt once."""
    by_path: dict[str, dict[str, float]] = {}
    for path, name in fields:
        current = getattr(_resolve(config, path), name)
        by_path.setdefault(path, {})[name] = current + delta
    for path, updates in by_path.items():
        config = _replace_nested(config, path, **updates)
    return config


def _resolve(config: PipelineConfig, path: str):
    node = config
    for part in path.split("."):
        node = getattr(node, part)
    return node


def _replace_nested(config: PipelineConfig, path: str, **updates) -> PipelineConfig:
    """``dataclasses.replace`` through a dotted path into frozen sub-configs."""
    parts = path.split(".")
    node = dataclasses.replace(_resolve(config, path), **updates)
    for depth in range(len(parts) - 1, -1, -1):
        parent = _resolve(config, ".".join(parts[:depth])) if depth else config
        node = dataclasses.replace(parent, **{parts[depth]: node})
    return node
