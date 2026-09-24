"""Sim-native hand-eye calibration glue.

Drives the real eye-in-hand and eye-to-hand calibrators inside Isaac: marker-pose sources (the
swappable perception seam), the declared station files the sim sweeps run, and the sim-only
validation helper that compares a calibrated transform against the ground-truth oracle. The pure-math helpers run
without Isaac, because the ``isaacsim`` reads are lazily imported.
"""

from __future__ import annotations

from .hand_eye import (
    ArucoMarkerPoseSource,
    GroundTruthMarkerPoseSource,
    MarkerPoseSource,
    transform_delta,
)
from .paths import STATIONS_DIR, declared_stations

__all__ = [
    "ArucoMarkerPoseSource",
    "GroundTruthMarkerPoseSource",
    "MarkerPoseSource",
    "STATIONS_DIR",
    "declared_stations",
    "transform_delta",
]
