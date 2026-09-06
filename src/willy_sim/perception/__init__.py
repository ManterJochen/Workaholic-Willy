"""Perception sources for the Isaac cell. They satisfy grasping's ``PerceptionSource`` Protocol.

* :mod:`ground_truth`: sources built from Isaac's ground-truth instance segmentation, which
  isolate motion from perception because the vision models never run. A single-object source
  and the multi-object dense-clutter source.
* :mod:`vision`: real-vision sources, GroundingDINO then SAM2 in the loop. A single-object
  source and the multi-object dense-clutter twin.

The camera, detector and segmenter are injected, so neither ``isaacsim`` nor torch is imported at
module top and both modules import on a machine without Isaac.
"""

from __future__ import annotations

from .ground_truth import (
    GroundTruthPerceptionSource,
    GroundTruthSegmentation,
    MultiObjectGroundTruthPerceptionSource,
    object_mask_from_frame,
)
from .vision import IsaacVisionPerceptionSource, MultiObjectVisionPerceptionSource

__all__ = [
    "GroundTruthPerceptionSource",
    "GroundTruthSegmentation",
    "MultiObjectGroundTruthPerceptionSource",
    "object_mask_from_frame",
    "IsaacVisionPerceptionSource",
    "MultiObjectVisionPerceptionSource",
]
