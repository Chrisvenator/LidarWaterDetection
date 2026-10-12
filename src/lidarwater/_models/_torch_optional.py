"""Guarded torch import — stages needing the deep models call ``require_torch()``
first so a missing install fails with a message naming the extra, not a bare
``ModuleNotFoundError`` deep in some internal module."""

from __future__ import annotations


def require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch installed
        raise ImportError(
            "This operation needs torch, which is not installed. "
            "Install it with: pip install 'lidar-water-detection[deep]'"
        ) from exc
    return torch
