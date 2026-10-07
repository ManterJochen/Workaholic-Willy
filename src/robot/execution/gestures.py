"""Gestures: Willy waves.

    from src.robot.execution import gestures

    waved = gestures.wave(arm, should_stop=lambda: run.stop_requested)
    waved.ok          # every swing ran and the wrist stands where it started
    print(waved)      # what each swing did, as ASCII text

A wave turns the second wrist joint (``WAVE_JOINT``, wrist 2 on a UR) :data:`WAVE_SWING_DEG` either way of where it
stands, :data:`WAVE_SWINGS` times, and back to where it started: the hand swings from side to side and the rest of the
arm keeps still. The owner wants it as the console's answer to "Hallo Willy" (2026-10-06).

**One motion, judged whole** (the owner, 2026-10-07: "das muss eine konsistente Bewegung sein"). On an arm that runs
judged joint paths (``DrivesJointPaths``, the UR driver) the whole wave goes to the arm at once through
:func:`~src.robot.execution.motion.move_through_joints_on_the_line`: every swing judged against the camera world by the
exact guard, sample by sample, before anything is sent, then the turning points sent one after the other with nothing
judged or planned between them, so the hand swings out and back without a pause. A swing that is not clear refuses
the whole wave, nothing sent. ``should_stop`` is read before the wave is sent; a halt brakes it where it is, and the
arm then stands where the brake stopped it. Swing by swing, each judged on its own, the hand stood five seconds at
every turn on the cell.

**Elsewhere every swing is a judged motion of its own.** Each one goes to the arm as a straight joint line through
:func:`~src.robot.execution.motion.move_joints_on_the_line`, and a refused swing ends the wave where the arm stands, and
so does ``should_stop`` between two swings: the wave never moves on its own to make up for a swing it did not run. An arm
with no controller (the dummy, the kinematic mock: route UNPLANNED) swings through ``move_joints``, which sets its pose;
any other arm that cannot drive a joint line alone is refused before the first swing.

The module imports nothing above ``robot.core`` and ``robot.execution.motion``, and connects nothing.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

from src.robot.core.arm_capabilities import DrivesJointLines, DrivesJointPaths
from src.robot.core.errors import RobotError
from src.robot.core.joint_positions import JointPositions

from . import motion
from .motion import MotionReport, MotionRoute

__all__ = ["WAVE_JOINT", "WAVE_SWINGS", "WAVE_SWING_DEG", "Wave", "wave", "wave_targets"]

#: The joint a wave turns: the second wrist joint (index 4, wrist 2 on a UR), which swings the hand from side to side.
WAVE_JOINT: Final[int] = 4
#: How far a swing turns that joint either way of where it stands, degrees. The Hand-E's fingertips then move about
#: 40 mm to each side.
WAVE_SWING_DEG: Final[float] = 15.0
#: How many times the hand swings out to each side before it comes back.
WAVE_SWINGS: Final[int] = 2
#: How far, radians, a joint may stand off where a wave started before a wave sent as one motion that ended early counts
#: as moved: well above what a standing arm reads, well below any swing.
_MOVED_RAD: Final[float] = math.radians(0.2)


def wave_targets(start: JointPositions, *, joint: int = WAVE_JOINT, swing_deg: float = WAVE_SWING_DEG,
                 swings: int = WAVE_SWINGS) -> tuple[JointPositions, ...]:
    """The joints a wave from ``start`` goes through, in order: ``joint`` out to one side and the other ``swings``
    times, then ``start`` again. Every other joint stays as it stands."""
    if not 0 <= joint < len(start):
        raise ValueError(f"a wave turns joint {joint}, which an arm of {len(start)} joints does not have")
    if swings < 1:
        raise ValueError(f"a wave swings at least once, not {swings} times")
    values = list(start)
    turn = math.radians(float(swing_deg))
    targets: list[JointPositions] = []
    for _ in range(swings):
        for side in (1.0, -1.0):
            swung = list(values)
            swung[joint] = values[joint] + side * turn
            targets.append(JointPositions(swung))
    targets.append(JointPositions(values))
    return tuple(targets)


@dataclass(frozen=True, slots=True)
class Wave:
    """What a wave did: the swings the arm ran, and the report of the one that ended it early, if one did."""

    #: The joints the wave went through (the last is where it started).
    targets: tuple[JointPositions, ...]
    #: The report of every motion the arm was asked for, in order; the last is the one that was refused, if any was.
    reports: tuple[MotionReport, ...]
    #: Why the wave ended before its last swing: ``"refused"`` (a swing the arm did not run), ``"stopped"``
    #: (``should_stop`` said so between two swings), or ``""`` when every swing ran.
    ended: str = ""
    #: The whole wave went to the arm as one judged joint path (``DrivesJointPaths``); its one report is ``reports``.
    as_one: bool = False
    #: A wave sent as one motion that ended early: whether the arm stood off where it started afterwards, read off the
    #: arm (``True`` where it could not be read). The controller may end a sent path part of the way along.
    left_start: bool = False

    @property
    def ok(self) -> bool:
        """Every swing ran, and the wrist stands where it started."""
        if self.as_one:
            return not self.ended and len(self.reports) == 1 and self.reports[0].ok
        return not self.ended and len(self.reports) == len(self.targets) and all(r.ok for r in self.reports)

    @property
    def moved(self) -> bool:
        """Whether the arm ran at least one swing; for a wave sent as one motion, whether it ran or left its start."""
        if self.as_one:
            return self.ok or self.left_start
        return any(report.ok for report in self.reports)

    @property
    def swings_ran(self) -> int:
        """How many swings the arm ran to their end: every one or none of a wave sent as one motion."""
        if self.as_one:
            return len(self.targets) if self.ok else 0
        return sum(1 for report in self.reports if report.ok)

    @property
    def refusal(self) -> MotionReport | None:
        """The report of the swing the arm refused, or ``None``."""
        return self.reports[-1] if self.ended == "refused" and self.reports else None

    def __str__(self) -> str:
        return self.render()

    def render(self) -> str:
        """Describe this to a person, as ASCII text without a trailing newline."""
        ran = self.swings_ran
        if self.ok:
            head = f"wave: {ran} swing(s){' as one motion' if self.as_one else ''}, back where it started"
        elif self.as_one and self.ended == "refused":
            head = ("wave: sent as one motion and ended part of the way; the arm stands where it stopped"
                    if self.left_start else "wave: refused as one motion; nothing moved")
        elif self.ended == "stopped":
            head = f"wave: stopped after {ran} of {len(self.targets)} swing(s); the arm stands where it stopped"
        else:
            head = f"wave: refused after {ran} of {len(self.targets)} swing(s); the arm stands where it stopped"
        lines = [head]
        refusal = self.refusal
        if refusal is not None:
            lines.extend(f"  {line}" for line in refusal.render().split("\n"))
        return "\n".join(lines)


def _swing(arm: Any, joints: JointPositions) -> MotionReport:
    """One swing, on the straight joint line where the arm drives one, else through ``move_joints`` where the arm has
    no controller; otherwise refused by the line verb, nothing sent."""
    if not isinstance(arm, DrivesJointLines) and motion.route_of(arm).route is MotionRoute.UNPLANNED:
        return motion.move_joints(arm, joints)
    return motion.move_joints_on_the_line(arm, joints)


def _left(arm: Any, start: list[float]) -> bool:
    """Whether ``arm`` stands off ``start`` by more than :data:`_MOVED_RAD` on any joint; ``True`` where it cannot say."""
    try:
        now = [float(v) for v in arm.get_joint_positions()]
    except (RobotError, RuntimeError, OSError):
        return True
    return len(now) != len(start) or any(abs(a - float(b)) > _MOVED_RAD for a, b in zip(now, start, strict=True))


def wave(arm: Any, *, should_stop: Callable[[], bool] = lambda: False, joint: int = WAVE_JOINT,
         swing_deg: float = WAVE_SWING_DEG, swings: int = WAVE_SWINGS) -> Wave:
    """Wave from where ``arm`` stands: as one judged motion on an arm that runs joint paths, else each swing a judged
    straight joint line, ``should_stop`` read before each.

    A swing the arm refuses ends the wave there, as does ``should_stop``; the arm then stands where the last swing it
    ran left it, and nothing is sent to take it back. Raises nothing the motion verbs end as an outcome.
    """
    start = arm.get_joint_positions()
    targets = wave_targets(JointPositions(list(start)), joint=joint, swing_deg=swing_deg, swings=swings)
    if isinstance(arm, DrivesJointPaths):
        if should_stop():
            return Wave(targets=targets, reports=(), ended="stopped", as_one=True)
        report = motion.move_through_joints_on_the_line(arm, list(targets))
        if report.ok:
            return Wave(targets=targets, reports=(report,), as_one=True)
        return Wave(targets=targets, reports=(report,), ended="refused", as_one=True,
                    left_start=_left(arm, list(start)))
    reports: list[MotionReport] = []
    for target in targets:
        if should_stop():
            return Wave(targets=targets, reports=tuple(reports), ended="stopped")
        report = _swing(arm, target)
        reports.append(report)
        if not report.ok:
            return Wave(targets=targets, reports=tuple(reports), ended="refused")
    return Wave(targets=targets, reports=tuple(reports))
