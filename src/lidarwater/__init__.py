"""lidarwater — water/land/canopy classification for full-waveform
bathymetric LiDAR point clouds.

Typical usage::

    from lidarwater import WaterPipeline, PipelineConfig, PointCloud
    from lidarwater.artifacts import LocalArtifactResolver

    cloud = PointCloud.from_dataframe(points_df, waveform_df)
    config, profile = derive_site_config(cloud)      # rebase thresholds onto this site
    pipeline = WaterPipeline.from_local_models("models/", config=config)
    state = pipeline.classify(cloud)
    # state.final_label: 0=land, 1=water, 2=uncertain, 3=water-under-canopy, 4=canopy
"""

from .artifacts import ArtifactId, ArtifactNotFound, ArtifactResolver, LocalArtifactResolver
from .config import (
    BoundaryConfig,
    CanopyConfig,
    FeatureConfig,
    LabelScheme,
    OutputConfig,
    PipelineConfig,
    RunConfig,
    Stage,
    SurfaceConfig,
    WcnConfig,
    ZoneConfig,
)
from .pipeline import WaterPipeline
from .site import SiteProfile, derive_site_config
from .types import PipelineState, PointCloud
from .workspace import Workspace

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "WaterPipeline",
    "PipelineConfig",
    "PointCloud",
    "PipelineState",
    "Stage",
    "ZoneConfig",
    "FeatureConfig",
    "WcnConfig",
    "SurfaceConfig",
    "BoundaryConfig",
    "CanopyConfig",
    "OutputConfig",
    "RunConfig",
    "LabelScheme",
    "ArtifactResolver",
    "LocalArtifactResolver",
    "ArtifactId",
    "ArtifactNotFound",
    "derive_site_config",
    "SiteProfile",
    "Workspace",
]
