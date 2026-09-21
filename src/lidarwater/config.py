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


GRID_ORIGIN_FIRST_SAMPLE = "first_sample"
GRID_ORIGIN_FIRST_RETURN = "first_return"
GRID_ORIGINS = frozenset({GRID_ORIGIN_FIRST_SAMPLE, GRID_ORIGIN_FIRST_RETURN})

# The 11 dimensionless features WCN v9 ships with. Reflectance is absent by
# design: its dB scale is a property of the scanner and the survey, so it does
# not transfer between sites. A model trained *for* one site is only ever used
# there, so derive_site_config() adds it back — on Inn it is the single
# strongest water/land feature (AUC 0.973 alone, against 0.924 for all 11).
WCN_SCALAR_FEATURES: tuple[str, ...] = (
    "energy_concentration", "max_amp_norm_by_energy", "energy_ratio_late",
    "active_bins_ratio", "peak_amp_ratio", "gap_ratio", "energy_center_norm",
    "n_peaks", "n_gaps", "n_clusters", "depth_proxy_m",
)
REFLECTANCE_FEATURE = "reflectance_dB"

STANDARDIZE_ARTIFACT = "artifact"
STANDARDIZE_SITE = "site"
STANDARDIZE_MODES = frozenset({STANDARDIZE_ARTIFACT, STANDARDIZE_SITE})


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

    # Where bin 0 of the dense grid sits. "first_sample" suits SVB-clustered
    # records (Pielach) whose stored samples already start at the first echo.
    # "first_return" is required for full-record digitisations that store the
    # whole range gate including a long pre-trigger noise floor — anchoring
    # those at times[0] fills the grid with noise. derive_site_config()
    # measures which one a cloud needs.
    grid_origin: str = "first_sample"      # "first_sample" | "first_return"
    grid_noise_percentile: float = 10.0    # amplitude percentile taken as the noise floor
    grid_return_frac: float = 0.10         # return starts at floor + this * (max - floor)

    # Full-record digitisations store the whole range gate, so most samples
    # are noise. Gating them away restores the sparse, echo-only record an
    # SVB export produces, which is what the gap/cluster/occupancy features
    # and the WCN's occupancy-mask channel assume. Set by derive_site_config.
    noise_gate: bool = False
    noise_gate_k: float = 3.0              # keep samples > floor + k * (median - floor)

    def __post_init__(self) -> None:
        if self.grid_origin not in GRID_ORIGINS:
            raise ValueError(f"grid_origin must be one of {sorted(GRID_ORIGINS)}, got {self.grid_origin!r}")

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


BOOTSTRAP_ZONES = "zones"
BOOTSTRAP_SURFACE = "surface"
BOOTSTRAP_METHODS = frozenset({BOOTSTRAP_ZONES, BOOTSTRAP_SURFACE})


@dataclasses.dataclass(frozen=True)
class BootstrapConfig:
    """How the first labels are made, before any model exists.

    ``"zones"`` is the original absolute-elevation banding (ZoneConfig): it
    assumes the site's water sits in a known height range, which is a
    property of one survey, not of water.

    ``"surface"`` uses no absolute elevation at all. It rasterises the cloud,
    takes each cell's top surface, keeps cells whose top agrees with their
    neighbours' (a coherent sheet), and splits those by the median
    reflectance of the *surface layer only* — bimodal because it is not
    diluted by riverbed returns. Measured on Inn against hand-labelled
    points: 97.2% balanced accuracy (100% water, 94.4% land), against 61.0%
    for the same threshold search over all points' reflectance.
    """

    method: str = BOOTSTRAP_ZONES
    cell_m: float = 3.0
    top_percentile: float = 0.97      # "the surface" within a cell
    surface_layer_m: float = 0.15     # points this close to the top define its reflectance
    coherence_max_m: float = 0.15     # max deviation from the neighbourhood sheet
    sheet_filter_cells: int = 7       # median-filter width defining that neighbourhood
    min_points_per_cell: int = 5

    def __post_init__(self) -> None:
        if self.method not in BOOTSTRAP_METHODS:
            raise ValueError(
                f"bootstrap method must be one of {sorted(BOOTSTRAP_METHODS)}, "
                f"got {self.method!r}")


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

    # Where the scalar z-scoring stats come from. The checkpoint ships the
    # mean/std of its training set — a property of that survey, not of the
    # model — so reusing them elsewhere pushes inputs far outside the range
    # the network ever saw. "site" recomputes them from the cloud being
    # classified. Set by derive_site_config; "artifact" keeps the deployed
    # Pielach behaviour bit-for-bit.
    standardize: str = "artifact"          # "artifact" | "site"

    # Which feature columns the scalar branch consumes. ``predict`` takes this
    # from the checkpoint's own stats file instead, so a model always sees the
    # features it was trained on.
    scalar_features: tuple[str, ...] = WCN_SCALAR_FEATURES

    def __post_init__(self) -> None:
        if self.standardize not in STANDARDIZE_MODES:
            raise ValueError(
                f"standardize must be one of {sorted(STANDARDIZE_MODES)}, got {self.standardize!r}")
        if self.arch.n_scalar != len(self.scalar_features):
            raise ValueError(
                f"arch.n_scalar ({self.arch.n_scalar}) must match the number of "
                f"scalar_features ({len(self.scalar_features)})")


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
    ransac_z_lo: float = 259.4             # z band the RANSAC plane is fitted over
    ransac_z_hi: float = 260.2
    ransac_reflectance_max_db: float = -10.0   # reflectance gate for plane candidates


@dataclasses.dataclass(frozen=True)
class BedReconstructionConfig:
    """Phase 3b — waterbed reconstruction (tree-over-water recovery)."""

    min_pts_per_cell: int = 3
    max_dist_m: float = 6.0          # max dist from confirmed bed data to qualify
    margin_m: float = 0.5            # z headroom below reconstructed bed
    proba_min: float = 0.95          # deep_proba threshold for high-conf bed anchor
    enabled: bool = True
    # Phase 3b exists to recover water points hidden under tree crowns. A site
    # with almost no canopy has no crowns to hide water under, so running it
    # only invents a class that cannot exist there.
    min_canopy_frac: float = 0.05
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
    # Where the surface grid had no nearby measurement it falls back to a
    # global plane, which on a wide channel sits below the true local surface
    # in patches. Points there were then "above surface" and forced to LAND
    # regardless of what the model said — the blocky islands on Inn, 73.6% of
    # which both model heads called water. Where the estimate is a fallback,
    # widen the tolerance and let the model decide.
    fallback_tol_m: float = 0.30   # default: same as water_tol_m, i.e. no change

    # The uncertain class (2) marks points the two model heads disagree on, or
    # that sit outside the footprint. It is a real signal, so it is emitted by
    # default — but consumers that treat it as "not water" pay for it. Measured
    # on Inn against 192k hand-labelled points: emitting it scores 97.4%
    # balanced, resolving it by the model's own probability scores 98.7%
    # (water recall 94.9% -> 97.9%, land 99.8% -> 99.5%).
    resolve_uncertain: bool = False


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

    # A site with no vegetation must yield no canopy, not whatever a
    # canopy-trained model hallucinates. Both fit() and predict() first check
    # how much of the cloud stands more than probe_height_m above the
    # water-aware ground reference; below min_canopy_frac the site is treated
    # as canopy-free and the stage short-circuits to all-zero probabilities.
    probe_height_m: float = 3.0
    min_canopy_frac: float = 0.002


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
    bootstrap: BootstrapConfig = dataclasses.field(default_factory=BootstrapConfig)
    features: FeatureConfig = dataclasses.field(default_factory=FeatureConfig)
    wcn: WcnConfig = dataclasses.field(default_factory=WcnConfig)
    surface: SurfaceConfig = dataclasses.field(default_factory=SurfaceConfig)
    boundary: BoundaryConfig = dataclasses.field(default_factory=BoundaryConfig)
    canopy: CanopyConfig = dataclasses.field(default_factory=CanopyConfig)
    output: OutputConfig = dataclasses.field(default_factory=OutputConfig)
    run: RunConfig = dataclasses.field(default_factory=RunConfig)
