"""Gestures: Willy waves.

    from src.robot.execution import gestures

    waved = gestures.wave(arm, should_stop=lambda: run.stop_requested)
    waved.ok          # every swing ran and the wrist stands where it started
    print(waved)      # what each swing did, as ASCII text

A wave turns the second wrist joint (``WAVE_JOINT``, wrist 2 on a UR) :data:`WAVE_SWING_DEG` either way of where it
stands, :data:`WAVE_SWINGS` times, and back to where it started: the hand swings from side to side and the rest of the
arm keeps still. The owner wants it as the console's answer to "Hallo Willy" (2026-10-06).

**Every swing is a judged motion.** Each one goes to the arm as a straight joint line through
:func:`~src.robot.execution.motion.move_joints_on_the_line`: judged against the camera world by the exact guard, sample
by sample, before anything is sent, and refused, nothing sent and nothing planned around it, where it is not clear. A
refused swing ends the wave where the arm stands, and so does ``should_stop`` between two swings: the wave never moves
on its own to make up for a swing it did not run. An arm with no controller (the dummy, the kinematic mock: route
UNPLANNED) swings through ``move_joints``, which sets its pose; any other arm that cannot drive a joint line alone is
refused before the first swing.

The module imports nothing above ``robot.core`` and ``robot.execution.motion``, and connects nothing.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

from src.robot.core.arm_capabilities import DrivesJointLines
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

    @property
    def ok(self) -> bool:
        """Every swing ran, and the wrist stands where it started."""
        return not self.ended and len(self.reports) == len(self.targets) and all(r.ok for r in self.reports)

    @property
    def moved(self) -> bool:
        """Whether the arm ran at least one swing."""
        return any(report.ok for report in self.reports)

    @property
    def refusal(self) -> MotionReport | None:
        """The report of the swing the arm refused, or ``None``."""
        return self.reports[-1] if self.ended == "refused" and self.reports else None

    def __str__(self) -> str:
        return self.render()

    def render(self) -> str:
        """Describe this to a person, as ASCII text without a trailing newline."""
        ran = sum(1 for report in self.reports if report.ok)
        if self.ok:
            head = f"wave: {ran} swing(s), back where it started"
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


def wave(arm: Any, *, should_stop: Callable[[], bool] = lambda: False, joint: int = WAVE_JOINT,
         swing_deg: float = WAVE_SWING_DEG, swings: int = WAVE_SWINGS) -> Wave:
    """Wave from where ``arm`` stands: each swing a judged straight joint line, ``should_stop`` read before each.

    A swing the arm refuses ends the wave there, as does ``should_stop``; the arm then stands where the last swing it
    ran left it, and nothing is sent to take it back. Raises nothing the motion verbs end as an outcome.
    """
    start = arm.get_joint_positions()
    targets = wave_targets(JointPositions(list(start)), joint=joint, swing_deg=swing_deg, swings=swings)
    reports: list[MotionReport] = []
    for target in targets:
        if should_stop():
            return Wave(targets=targets, reports=tuple(reports), ended="stopped")
        report = _swing(arm, target)
        reports.append(report)
        if not report.ok:
            return Wave(targets=targets, reports=tuple(reports), ended="refused")
    return Wave(targets=targets, reports=tuple(reports))
