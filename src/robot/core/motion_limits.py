"""The speed limit every commanded motion is held to, whoever sends it (P0e E7).

``robot.motion_limits`` bounds the joint speed and acceleration a caller may ask for. The UR motion controller clamped
its own commands; the joint drive of the planned arm asked that controller to clamp for it, which a PlannedArm without
a UR motion controller cannot. One function, so the two cannot drift.
"""

from __future__ import annotations

import logging

__all__ = ["clamp_to_motion_limits"]


def clamp_to_motion_limits(
    vel: float | None,
    acc: float | None,
    *,
    max_velocity: float | None,
    max_acceleration: float | None,
    logger: logging.Logger,
) -> tuple[float | None, float | None]:
    """``(vel, acc)`` held to the configured limits, each warned about where it was lowered; ``None`` passes through."""
    if vel is not None and max_velocity is not None and vel > max_velocity:
        logger.warning("Requested velocity %.3f exceeds limit %.3f, clamping.", vel, max_velocity)
        vel = max_velocity
    if acc is not None and max_acceleration is not None and acc > max_acceleration:
        logger.warning("Requested acceleration %.3f exceeds limit %.3f, clamping.", acc, max_acceleration)
        acc = max_acceleration
    return vel, acc
