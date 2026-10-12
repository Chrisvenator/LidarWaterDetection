"""Model artifact resolution seam.

Stages never open a hardcoded path under ``models/``; they ask an
``ArtifactResolver`` for a named artifact. ``LocalArtifactResolver`` is the
only implementation today (points at a local directory laid out like the
current ``models/`` tree). A future download-on-demand resolver plugs in
here without touching any stage code — it only needs to implement
``resolve`` and ``resolve_for_write``.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from pathlib import Path
from typing import Protocol


class ArtifactId(str, Enum):
    """Every model weight file a stage may load or save."""

    WCN_REFINED = "wcn_refined"          # WCNv9 transformer state_dict (.pt)
    WCN_XGB = "wcn_xgb"                  # XGBoost on 11 scalar features (.json)
    WCN_STATS = "wcn_stats"              # scalar normalisation stats + metrics (.json)
    CANOPY_XGB = "canopy_xgb"            # canopy XGBoost (.json)
    V6_XGB = "v6_xgb"                    # v6 waveform-only XGBoost (.json)
    V6_DEEP = "v6_deep"                  # V6Net state_dict (.pt)
    V6_DEEP_STATS = "v6_deep_stats"      # V6Net normalisation stats (.json)
    V8_XGB = "v8_xgb"                    # v8 surface-model XGBoost (.json)
    V8_DEEP = "v8_deep"                  # V8Net state_dict (.pt)
    V8_DEEP_STATS = "v8_deep_stats"      # V8Net normalisation stats (.json)


# Which stage(s) need which artifact — lets a future downloader compute its
# fetch set from the configured RunConfig.stages without hardcoding a list.
ARTIFACT_STAGES: dict[ArtifactId, tuple[str, ...]] = {
    ArtifactId.WCN_REFINED: ("wcn",),
    ArtifactId.WCN_XGB: ("wcn",),
    ArtifactId.WCN_STATS: ("wcn",),
    ArtifactId.CANOPY_XGB: ("canopy",),
    ArtifactId.V6_XGB: ("autolabel",),
    ArtifactId.V6_DEEP: ("autolabel",),
    ArtifactId.V6_DEEP_STATS: ("autolabel",),
    ArtifactId.V8_XGB: ("geometry",),
    ArtifactId.V8_DEEP: ("geometry",),
    ArtifactId.V8_DEEP_STATS: ("geometry",),
}


class ArtifactNotFound(FileNotFoundError):
    def __init__(self, artifact_id: ArtifactId, path: Path):
        super().__init__(
            f"Artifact '{artifact_id.value}' not found at {path}. "
            f"Train it first, or point the resolver at a directory that has it."
        )
        self.artifact_id = artifact_id
        self.path = path


class ArtifactResolver(Protocol):
    """Seam between stages and wherever model weights actually live."""

    def resolve(self, artifact_id: ArtifactId) -> Path:
        """Return a path to an existing artifact, or raise ArtifactNotFound."""
        ...

    def resolve_for_write(self, artifact_id: ArtifactId) -> Path:
        """Return a path a training stage should write this artifact to
        (parent directories created as needed). Training and inference read
        through the same seam, so a trained artifact is immediately
        resolvable by future ``resolve()`` calls."""
        ...


_RELATIVE_PATHS: dict[ArtifactId, str] = {
    ArtifactId.WCN_REFINED: "wcn_v9/wcn_refined.pt",
    ArtifactId.WCN_XGB: "wcn_v9/wcn_xgb.json",
    ArtifactId.WCN_STATS: "wcn_v9/wcn_stats.json",
    ArtifactId.CANOPY_XGB: "canopy_v1/canopy_xgb.json",
    ArtifactId.V6_XGB: "labeling/v6_xgb.json",
    ArtifactId.V6_DEEP: "labeling/v6_deep.pt",
    ArtifactId.V6_DEEP_STATS: "labeling/v6_deep_stats.json",
    ArtifactId.V8_XGB: "current/v8_xgb.json",
    ArtifactId.V8_DEEP: "current/v8_deep.pt",
    ArtifactId.V8_DEEP_STATS: "current/v8_deep_stats.json",
}


@dataclasses.dataclass(frozen=True)
class LocalArtifactResolver:
    """Resolves artifacts under a local directory laid out like the
    repository's ``models/`` tree (``<root>/wcn_v9/wcn_refined.pt``, etc.)."""

    root: Path

    def resolve(self, artifact_id: ArtifactId) -> Path:
        path = self.root / _RELATIVE_PATHS[artifact_id]
        if not path.exists():
            raise ArtifactNotFound(artifact_id, path)
        return path

    def resolve_for_write(self, artifact_id: ArtifactId) -> Path:
        path = self.root / _RELATIVE_PATHS[artifact_id]
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
