"""The speed limit every commanded motion is held to, whoever sends it (P0e E7).

``robot.motion.joint`` bounds the joint speed and acceleration of a joint move (``moveJ``, rad/s and rad/s^2),
``robot.motion.linear`` those of a straight line (``moveL``, m/s and m/s^2 of the TCP): two pairs since 2026-11-06, when
one pair served both and a line read the joints' rad/s as m/s. The UR motion controller clamped its own commands; the
joint drive of the planned arm asked that controller to clamp for it, which a PlannedArm without a UR motion controller
cannot. One function, so the two cannot drift.
"""

from __future__ import annotations

import logging
from typing import Final, Literal

__all__ = ["clamp_to_motion_limits"]

#: The units of each kind of motion's speed and acceleration, as the warning names them.
_UNITS: Final[dict[str, tuple[str, str]]] = {"joint": ("rad/s", "rad/s^2"), "linear": ("m/s", "m/s^2")}


def clamp_to_motion_limits(
    vel: float | None,
    acc: float | None,
    *,
    max_velocity: float | None,
    max_acceleration: float | None,
    logger: logging.Logger,
    kind: Literal["joint", "linear"] = "joint",
) -> tuple[float | None, float | None]:
    """``(vel, acc)`` held to the configured limits, each warned about where it was lowered; ``None`` passes through.

    Args:
        vel (float | None): The speed the caller asked for, rad/s for a joint move and m/s for a line.
        acc (float | None): The acceleration the caller asked for, rad/s^2 or m/s^2.
        max_velocity (float | None): The ceiling on ``vel``; ``None`` holds nothing.
        max_acceleration (float | None): The ceiling on ``acc``; ``None`` holds nothing.
        logger (logging.Logger): Where a lowered value is warned about.
        kind (Literal["joint", "linear"]): Which motion the pair belongs to, named with its units in the warning
            (default: "joint").

    Returns:
        tuple[float | None, float | None]: ``vel`` and ``acc``, each at most its ceiling.
    """
    speed, rate = _UNITS[kind]
    if vel is not None and max_velocity is not None and vel > max_velocity:
        logger.warning("Requested %s velocity %.3f %s exceeds limit %.3f %s, clamping.", kind, vel, speed, max_velocity,
                       speed)
        vel = max_velocity
    if acc is not None and max_acceleration is not None and acc > max_acceleration:
        logger.warning("Requested %s acceleration %.3f %s exceeds limit %.3f %s, clamping.", kind, acc, rate,
                       max_acceleration, rate)
        acc = max_acceleration
    return vel, acc
