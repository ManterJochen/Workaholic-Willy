"""Sim-native hand-eye calibration glue.

Drives the real eye-in-hand and eye-to-hand calibrators inside Isaac: marker-pose sources (the
swappable perception seam), hemisphere viewpoint generation, and the sim-only validation helper
that compares a calibrated transform against the ground-truth oracle. The pure-math helpers run
without Isaac, because the ``isaacsim`` reads are lazily imported.
"""

from __future__ import annotations

from .hand_eye import (
    ArucoMarkerPoseSource,
    CalibrationViewpoint,
    GroundTruthMarkerPoseSource,
    MarkerPoseSource,
    generate_hemisphere_viewpoints,
    look_at_world_matrix,
    transform_delta,
)

__all__ = [
    "ArucoMarkerPoseSource",
    "CalibrationViewpoint",
    "GroundTruthMarkerPoseSource",
    "MarkerPoseSource",
    "generate_hemisphere_viewpoints",
    "look_at_world_matrix",
    "transform_delta",
]
