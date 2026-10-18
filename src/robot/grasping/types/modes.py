"""The sampling each grasp mode names, the label its reports and records carry.

Each :class:`~src.robot.execution.autonomous_grasp.config.GraspMode` profile carries one: ``easy`` single_object,
``auto`` auto, ``dense_clutter`` dense_clutter. It chose the calculator's sampler until 2026-10-16, when the silhouette
stage left with its dense and geometry-first samplers (the owner's cleanup): the part's body over its support grasps
every part alike, so no stage samples by it, and a mode differs from another in its recovery alone.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["GraspSamplingMode"]


class GraspSamplingMode(StrEnum):
    """The sampling a grasp mode names.

    The values are stable strings, so they travel through JSON telemetry and a record without translation.
    """

    AUTO = "auto"
    SINGLE_OBJECT = "single_object"
    DENSE_CLUTTER = "dense_clutter"
