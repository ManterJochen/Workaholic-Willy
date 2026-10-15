"""What the stack knows about one gripper vendor's driver, as one frozen value beside the driver (P0g slice 8).

Until P0g the build, the desk check, the console, the bench and the doctor each branched on ``robot.gripper.vendor``
by hand: which arm a driver needs, how it is built, what to confirm on the bench, what moves when it connects, which
SDK it imports. A new gripper meant five dispatches to find and extend, and one that was forgotten would build a
``NullGripper`` that reports every pick a success. Each driver module states those facts once, at its end, as a
:class:`GripperDriver` named ``DRIVER``, and every reader looks the vendor up
(:func:`~src.robot.grippers.registry.gripper_driver`).

The values are neutral on purpose: this package sits below the execution layer and the console, so a desk row is a
:class:`Row` the desk wraps in its own check, and a connect hazard a :class:`Hazard` the console wraps in its warning.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import RobotConfig
    from src.robot.core import Gripper, GripperVendor

    from .null import SubstitutionReason

__all__ = ["ArmFacts", "GripperDriver", "Hazard", "Refusal", "Row"]


@dataclass(frozen=True)
class ArmFacts:
    """What a gripper's driver may ask of the arm it is built for, read off the handle or stated for the arm a config
    builds: the vendor the arm reports, whether it switches digital I/O, and the controller address it names."""

    vendor: str
    has_digital_io: bool
    controller: str | None


@dataclass(frozen=True)
class Refusal:
    """Why no real end-effector can be built for this arm, and what to do about it."""

    reason: SubstitutionReason
    detail: str
    fix: str


@dataclass(frozen=True)
class Row:
    """A line of the desk check in the driver's words: its standing, what it says, what to do."""

    status: Literal["ok", "warn", "bench"]
    detail: str
    fix: str = ""


@dataclass(frozen=True)
class Hazard:
    """One thing that physically moves when this gripper connects, and what an operator does about it first."""

    what: str
    precaution: str


def _never(_robot_cfg: RobotConfig, *_rest: Any) -> None:
    return None


@dataclass(frozen=True)
class GripperDriver:
    """One gripper vendor's driver, as the build, the desk check, the console, the bench and the doctor read it."""

    vendor: GripperVendor
    #: Why this driver cannot be built on an arm with these facts, or ``None`` where it can.
    refuse: Callable[[RobotConfig, ArmFacts], Refusal | None]
    #: The driver for ``arm``, built from the tree with the facts its refusal was asked with; nothing is connected.
    build: Callable[[RobotConfig, Any, ArmFacts], Gripper]
    #: What sits on the flange when this driver actuates nothing ("nothing", a stand-in), or ``None`` for a real one.
    actuates_nothing: str | None = None
    #: What to confirm physically for this driver, which no API reports; ``None`` where nothing is actuated.
    wiring: Callable[[RobotConfig], Row] | None = None
    #: What moves when the built gripper connects, read off the built object. The third argument is where the reader
    #: asks a person at connect and how long it waits, for a hand whose connect asks where its jaws stand.
    hazards: Callable[[RobotConfig, Any, str], tuple[Hazard, ...]] = lambda _cfg, _gripper, _asked: ()
    #: Whether the travel time this driver waits on was measured, where it is the only wait; ``None`` for most.
    travel_time: Callable[[RobotConfig], Row | None] = _never
    #: Whether this driver drives jaws over the arm's digital I/O, which is what the bench's jaw verbs exercise.
    drives_jaws_over_io: bool = False
    #: Whether the hand opens only as wide as a grasp needs before it closes, because it is told a width (lever G, the
    #: owner's decision of 2026-10-13); every other hand opens fully.
    opens_as_needed: bool = False
    #: The third-party modules the driver imports, for the doctor; empty for a driver with no SDK.
    sdk_modules: tuple[str, ...] = ()
