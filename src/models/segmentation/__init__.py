"""Segmentation: the shared :class:`SegmentationResult` type and SAM2, the segmenter of every stack.

The torch-heavy wrapper lives under ``realtime`` (SAM2, box-prompted) and is imported
from there directly. Only the lightweight result type is re-exported here, so
importing this package stays torch-free.
"""

from __future__ import annotations

from src.models.segmentation.types import SegmentationResult

__all__ = ["SegmentationResult"]
