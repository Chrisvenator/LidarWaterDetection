"""Configuration dataclasses for the LiDAR water-detection pipeline.

Every tunable that was a hardcoded module constant in the original scripts
lives here, grouped by stage, with defaults equal to the values verified on
the Pielach study area. Overriding a field changes only that stage's
behavior; everything else keeps its default.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from pathlib import Path


class Stage(str, Enum):
    """Pipeline stages, in their natural dependency order."""

    FEATURES = "features"
    AUTOLABEL = "autolabel"      # v6 waveform-only bootstrap (training-only path)
    WCN = "wcn"                  # WCN v9 transformer + xgboost
    GEOMETRY = "geometry"        # water-surface footprint/surface/classify/bed-recon
    CANOPY = "canopy"
    MERGE = "merge"
    BOUNDARY = "boundary"


# Stages classify() runs by default: the v10 + canopy + boundary cascade that
# CLAUDE.md documents as the current best output. AUTOLABEL is intentionally
# excluded — the geometry stage consumes WCN probas directly, not v6's.
DEFAULT_CLASSIFY_STAGES: tuple[Stage, ...] = (
    Stage.FEATURES, Stage.WCN, Stage.GEOMETRY, Stage.CANOPY, Stage.MERGE, Stage.BOUNDARY,
)

# Stages fit() runs by default: full bootstrap from raw data through v6 -> v9 training.
DEFAULT_FIT_STAGES: tuple[Stage, ...] = (
    Stage.FEATURES, Stage.AUTOLABEL, Stage.WCN, Stage.GEOMETRY,
    Stage.CANOPY, Stage.MERGE, Stage.BOUNDARY,
)


@dataclasses.dataclass(frozen=True)
class FeatureConfig:
    """Waveform + geometric feature extraction (feature_extractor / add_features)."""

    grid_size: int = 200        # time bins per waveform grid
    knn_k: int = 20             # neighbours for geometric (planarity/roughness/...) features
    min_peak_adc: int = 100     # minimum ADC for waveform peak detection
    gap_thresh_si: int = 2      # minimum time gap (SI) to count as a waveform gap
    local_min_radius_m: float = 3.0       # height_above_local_min radius
    local_min_radius_10m: float = 10.0    # height_above_local_min_10m radius
    local_rank_radius_m: float = 5.0      # height_percentile_local radius
    raster_cell_m: float = 0.5            # cell size for rasterised spatial ops


@dataclasses.dataclass(frozen=True)
class ZoneConfig:
    """Elevation-band boundaries used to bootstrap training labels (v6 / canopy).

    Verified by manual cross-section inspection on the Pielach study area;
    not physically universal — override per site.
    """

    z_underwater_max: float = 259.6
    z_water_surf_max: float = 259.9
    z_dry_bed_min: float = 260.0
    z_dry_bed_max: float = 260.4
    z_banks_min: float = 260.9
    z_banks_max: float = 263.1
    z_canopy_min: float = 263.3


@dataclasses.dataclass(frozen=True)
class WcnArchConfig:
    """WCNv9 transformer architecture. Must match deployed checkpoints — do not
    change unless retraining from scratch, since shapes are baked into the
    saved state_dict."""

    n_scalar: int = 11
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 6
    n_patches: int = 50   # 200 bins / 4-bin stride

    @property
    def seq_len(self) -> int:
        return self.n_patches + 1  # +1 for CLS token


@dataclasses.dataclass(frozen=True)
class WcnTrainConfig:
    """WCN v9 training hyperparameters (fit() path only)."""

    phase1_epochs: int = 50
    phase1_batch: int = 1024
    phase1_lr: float = 1e-3
    phase1_mask_ratio: float = 0.40

    phase2_epochs_frozen: int = 30
    phase2_epochs_full: int = 120
    phase2_batch: int = 512
    phase2_lr_frozen: float = 5e-4
    phase2_lr_full: float = 2e-4
    phase2_patience: int = 20

    phase3_rounds: int = 2
    phase3_epochs: int = 30
    phase3_batch: int = 512
    phase3_lr: float = 5e-5
    phase3_proba_hi: float = 0.92   # pseudo-label water threshold
    phase3_proba_lo: float = 0.08   # pseudo-label land threshold

    focal_gamma: float = 2.0
    focal_alpha: float = 0.65
    focal_smooth: float = 0.05

    aux_weight_energy_concentration: float = 0.05
    aux_weight_depth_proxy: float = 0.05

    val_fraction: float = 0.20
    seed: int = 42


@dataclasses.dataclass(frozen=True)
class WcnConfig:
    arch: WcnArchConfig = dataclasses.field(default_factory=WcnArchConfig)
    train: WcnTrainConfig = dataclasses.field(default_factory=WcnTrainConfig)


@dataclasses.dataclass(frozen=True)
class FootprintConfig:
    """Phase 1 — tight river footprint (concave hull of high-confidence water)."""

    conf: float = 0.8              # tier-1 anchor min mean(xgb_proba, deep_proba)
    conf_surface: float = 0.85     # tier-2 anchor threshold
    riverbed_z_max: float = 259.6  # z < this = underwater / riverbed
    riverbed_z_surface_max: float = 261.5   # z < this for tier-2 surface anchors
    hull_ratio: float = 0.2        # concave_hull tightness (lower = tighter)
    erosion_m: float = 1.0         # erode hull inward by this many metres
    tier2_max_dist_from_tier1_m: float = 10.0


@dataclasses.dataclass(frozen=True)
class SurfaceGridConfig:
    """Phase 2 — local adaptive water-surface grid."""

    cell_size_m: float = 2.0
    z_lo: float = 259.0
    z_hi: float = 261.5
    n_peaks_max: int = 2
    energy_concentration_min: float = 0.85
    reflectance_max_db: float = -15.0
    z_cap: float = 261.0
    min_pts_per_cell: int = 5
    smooth_sigma_cells: float = 1.0
    max_dist_from_tier1_m: float = 12.0    # cells farther than this use RANSAC fallback
    ransac_residual_m: float = 0.20
    ransac_max_rise_m: float = 0.15        # cap on RANSAC-fallback cells above the plane


@dataclasses.dataclass(frozen=True)
class BedReconstructionConfig:
    """Phase 3b — waterbed reconstruction (tree-over-water recovery)."""

    min_pts_per_cell: int = 3
    max_dist_m: float = 6.0          # max dist from confirmed bed data to qualify
    margin_m: float = 0.5            # z headroom below reconstructed bed
    proba_min: float = 0.95          # deep_proba threshold for high-conf bed anchor
    recon_label: int = 3
    reflectance_max_db: float = -15.0
    min_peaks: int = 3
    planarity_min: float = 0.30


@dataclasses.dataclass(frozen=True)
class SurfaceConfig:
    """Water-surface model: footprint + surface grid + classify + bed reconstruction."""

    footprint: FootprintConfig = dataclasses.field(default_factory=FootprintConfig)
    surface_grid: SurfaceGridConfig = dataclasses.field(default_factory=SurfaceGridConfig)
    bed: BedReconstructionConfig = dataclasses.field(default_factory=BedReconstructionConfig)
    water_tol_m: float = 0.30   # inside footprint, z <= surface + this -> WATER


@dataclasses.dataclass(frozen=True)
class BoundaryConfig:
    """River boundary probability-field contour extraction."""

    cell_size_m: float = 0.5
    smooth_sigma_m: float = 1.5
    prob_inner: float = 0.65
    prob_center: float = 0.50
    prob_outer: float = 0.35
    min_seg_len: int = 20
    max_contours: int = 3
    max_fill_dist_m: float = 3.0
    isolation_radius_m: float = 3.0
    min_water_support: int = 5
    canopy_z_max: float = 268.0

    def __post_init__(self) -> None:
        if not (self.prob_outer < self.prob_center < self.prob_inner):
            raise ValueError(
                "BoundaryConfig requires prob_outer < prob_center < prob_inner, got "
                f"{self.prob_outer}, {self.prob_center}, {self.prob_inner}"
            )


@dataclasses.dataclass(frozen=True)
class CanopyConfig:
    """Canopy classifier: z-band bootstrap + open-sky-low pseudo-negative rule."""

    z_canopy_min: float = 263.283   # above this = 100% canopy (training label)
    z_clear_max: float = 260.1      # below this = 0% canopy (training label)
    surface_tol_m: float = 0.15     # |z - water surface| below this -> pseudo-negative
    threshold: float = 0.5
    n_folds: int = 5
    open_sky_max_above: int = 1     # max neighbors >=2m overhead in the 1m cylinder
    low_height_max_m: float = 1.2   # max height above ground/water reference
    cell_m: float = 1.0             # DTM/DSM grid cell size
    r_local_m: float = 1.0          # cylinder/sphere neighborhood radius
    above_gap_m: float = 2.0        # neighbor this far above = canopy cover above
    dtm_percentile: float = 0.05    # robust per-cell ground elevation


class LabelScheme(str, Enum):
    """Output classification mapping. See lidarwater.io.las_writer."""

    TOPO_BATHY = "topo_bathy"   # ASPRS topo-bathy: 2=ground,5=high veg,40=bed,41=surface,1=unclassified
    ASPRS_BASIC = "asprs_basic"  # generic ASPRS: 2=ground,5=veg,9=water,1=unclassified


@dataclasses.dataclass(frozen=True)
class OutputConfig:
    crs_epsg: int = 25833
    xyz_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    label_scheme: LabelScheme = LabelScheme.TOPO_BATHY


@dataclasses.dataclass(frozen=True)
class RunConfig:
    stages: tuple[Stage, ...] = DEFAULT_CLASSIFY_STAGES
    cache_dir: Path | None = None
    plot_dir: Path | None = None
    device: str = "auto"   # "auto" | "cpu" | "cuda"

    def __post_init__(self) -> None:
        stages = set(self.stages)
        if Stage.GEOMETRY in stages and Stage.WCN not in stages and Stage.AUTOLABEL not in stages:
            raise ValueError(
                "Stage.GEOMETRY requires Stage.WCN or Stage.AUTOLABEL in the stage set "
                "to supply xgb_proba/deep_proba anchors."
            )
        if Stage.CANOPY in stages and Stage.GEOMETRY not in stages:
            raise ValueError(
                "Stage.CANOPY requires Stage.GEOMETRY (needs the local water surface "
                "as ground reference)."
            )
        if Stage.MERGE in stages and Stage.CANOPY not in stages:
            raise ValueError("Stage.MERGE requires Stage.CANOPY.")
        if Stage.BOUNDARY in stages and Stage.GEOMETRY not in stages:
            raise ValueError("Stage.BOUNDARY requires Stage.GEOMETRY.")


@dataclasses.dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration composing every stage's tunables.

    ``PipelineConfig()`` reproduces the defaults documented in CLAUDE.md for
    the Pielach study area. Override individual nested configs via
    ``dataclasses.replace`` for a different site or experiment.
    """

    zones: ZoneConfig = dataclasses.field(default_factory=ZoneConfig)
    features: FeatureConfig = dataclasses.field(default_factory=FeatureConfig)
    wcn: WcnConfig = dataclasses.field(default_factory=WcnConfig)
    surface: SurfaceConfig = dataclasses.field(default_factory=SurfaceConfig)
    boundary: BoundaryConfig = dataclasses.field(default_factory=BoundaryConfig)
    canopy: CanopyConfig = dataclasses.field(default_factory=CanopyConfig)
    output: OutputConfig = dataclasses.field(default_factory=OutputConfig)
    run: RunConfig = dataclasses.field(default_factory=RunConfig)
