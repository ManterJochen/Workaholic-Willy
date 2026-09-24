"""Hand guiding: an arm a person moves by hand while the software watches.

A vendor that offers it (the UR teach mode) advertises SupportsFreedrive; the hand-guided calibration and the
adjust step of fixed stations use it, and an arm without it keeps fixed stations only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class FreedriveSample:
    """One reading of a hand-guided arm: joints and speeds in radians, the TCP in millimetres and a rotation vector."""

    joints_rad: tuple[float, ...]
    joint_speeds_rad_s: tuple[float, ...]
    tcp_xyz_mm: tuple[float, float, float]
    tcp_rotvec_rad: tuple[float, float, float]
    t_s: float

    @property
    def peak_joint_speed_rad_s(self) -> float:
        """The fastest joint, which is what says whether a person still moves the arm."""
        return max((abs(v) for v in self.joint_speeds_rad_s), default=0.0)


@dataclass(frozen=True)
class ControllerPayload:
    """The payload the controller compensates for, which decides whether a freed arm floats, sinks or rises."""

    mass_kg: float
    cog_mm: tuple[float, float, float]


class FreedriveSession(Protocol):
    """An open hand-guiding session. While it is open every motion verb of the arm refuses.

    free() lets a person move the arm; hold() makes the arm hold where it stands. Leaving the session
    (the with-block) always holds the arm and gives motion back, also on an exception or Ctrl-C.
    sample() is called often (30 Hz or more) while free and keeps the crash watchdog alive.
    """

    @property
    def is_free(self) -> bool: ...

    def free(self) -> None: ...

    def hold(self) -> None: ...

    def sample(self) -> FreedriveSample: ...

    def __enter__(self) -> FreedriveSession: ...

    def __exit__(self, *exc: object) -> None: ...


@runtime_checkable
class SupportsFreedrive(Protocol):
    """An arm a person can move by hand while the software watches (a vendor capability)."""

    def freedrive(self) -> FreedriveSession: ...

    def controller_payload(self) -> ControllerPayload | None: ...
