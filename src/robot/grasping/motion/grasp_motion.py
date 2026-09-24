"""How a pick moves, as a caller may tune it, and the one builder that puts the guards in.

``GraspMotion`` is what a caller may choose about the approach, the close and the lift. It carries no arm, no hand and
no guard. The pick service builds the one :class:`GraspExecutionPolicy` it drives from it, with the arm and hand it
resolved, the base frame guard where a frame resolver is wired, the dwell gate the tree asks for, and the jaws opened
to the hand's width before every approach (a hand that toggles with no sensor is asked where its jaws stand instead,
and never pulsed before the arm moves). A policy built by hand drives whatever arm it holds and carries only the
guards its builder set, so while ``policy=`` is still accepted the service refuses one whose arm or hand is not its own.

    from src.robot.grasping.motion.grasp_motion import GraspMotion

    motion = GraspMotion(standoff_mm=60.0, close_squeeze_mm=5.0)
    cell = Cell.from_robot_config(robot_config, motion=motion)          # or from_components(..., motion=motion)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

from src.contracts import UNSET, Maybe, chosen
from src.robot.core import Gripper, RobotArm
from src.robot.core.gripper import toggle_without_sensor_of
from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy

__all__ = ["GraspMotion", "build_execution_policy", "foreign_policy_refusal"]

#: Fields in millimetres or newtons: finite and not negative.
_NON_NEGATIVE = ("standoff_mm", "retreat_mm", "close_squeeze_mm", "close_force_n")


@dataclass(frozen=True, slots=True, kw_only=True)
class GraspMotion:
    """What a caller may choose about how a pick moves. A field left ``UNSET`` keeps the service's own.

    ``standoff_mm`` is the distance above the grasp the approach starts from, along the reverse approach axis, and
    ``retreat_mm`` the lift after the close. ``approach_steps`` (at least 2) and ``retreat_steps`` (at least 1) split
    them into waypoints on an arm that keeps no line. ``pre_open_width_mm`` is how far the jaws open before the
    approach, the hand's widest when unset, and never ``None``: the pre-open is what keeps an approach from arriving
    with the jaws wherever the last close left them. A hand that toggles with no sensor is the one exception, and it
    is the hand's, not this object's: it is never pulsed before the arm moves, it is asked where its jaws stand
    instead, and it takes no width, so the builder drops the pre-open for it whatever this says. ``close_squeeze_mm``
    is how far below the measured width the jaws close, ``close_speed`` (0 to 1) and ``close_force_n`` what the hand
    is asked for, and ``align_closing_to_base_x`` yaws a symmetric top-down grasp so it closes along base X.
    """

    standoff_mm: Maybe[float] = UNSET
    retreat_mm: Maybe[float] = UNSET
    approach_steps: Maybe[int] = UNSET
    retreat_steps: Maybe[int] = UNSET
    pre_open_width_mm: Maybe[float] = UNSET
    close_squeeze_mm: Maybe[float] = UNSET
    close_speed: Maybe[float] = UNSET
    close_force_n: Maybe[float] = UNSET
    align_closing_to_base_x: Maybe[bool] = UNSET

    def __post_init__(self) -> None:
        if self.pre_open_width_mm is None:
            raise TypeError(
                "GraspMotion.pre_open_width_mm is a width in mm, not None: the jaws open before every approach, and "
                "skipping that is the guard this object exists to keep; leave it unset for the hand's widest")
        for name in (*_NON_NEGATIVE, "pre_open_width_mm", "close_speed"):
            value = getattr(self, name)
            if not chosen(value):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"GraspMotion.{name} is a number, not {value!r}; leave it unset for the service's own")
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"GraspMotion.{name} must be finite and not negative, got {value!r}")
        if chosen(self.pre_open_width_mm) and self.pre_open_width_mm == 0.0:
            raise ValueError("GraspMotion.pre_open_width_mm of 0 mm opens nothing; leave it unset for the hand's widest")
        if chosen(self.close_speed) and self.close_speed > 1.0:
            raise ValueError(f"GraspMotion.close_speed is a fraction from 0 to 1, got {self.close_speed!r}")
        for name, least in (("approach_steps", 2), ("retreat_steps", 1)):
            value = getattr(self, name)
            if not chosen(value):
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"GraspMotion.{name} is a whole number, not {value!r}")
            if value < least:
                raise ValueError(f"GraspMotion.{name} must be at least {least}, got {value}")
        if chosen(self.align_closing_to_base_x) and not isinstance(self.align_closing_to_base_x, bool):
            raise TypeError(f"GraspMotion.align_closing_to_base_x is True or False, not {self.align_closing_to_base_x!r}")

    def standoff_and_retreat(self, standoff_mm: float, retreat_mm: float) -> tuple[float, float]:
        """This motion's standoff and retreat where it sets them, the service's own where it does not.

        A service reads it before it builds anything that also takes the standoff (the closed loop's second look),
        so the policy and everything beside it move to one standoff.
        """
        return (float(self.standoff_mm) if chosen(self.standoff_mm) else float(standoff_mm),
                float(self.retreat_mm) if chosen(self.retreat_mm) else float(retreat_mm))

    def to_dict(self) -> dict[str, Any]:
        """The fields a caller chose, by name; a field left unset is absent."""
        return {f.name: getattr(self, f.name) for f in fields(self) if chosen(getattr(self, f.name))}


def build_execution_policy(
    motion: GraspMotion,
    *,
    arm: RobotArm,
    gripper: Gripper | None,
    standoff_mm: float,
    retreat_mm: float,
    base_frame_required: bool,
    dwell: Any = None,
) -> GraspExecutionPolicy:
    """The one policy a pick service drives: the caller's motion, the service's arm and hand, and every guard.

    ``base_frame_required`` is whether a frame resolver is wired, so a camera frame grasp is refused before any
    motion. ``dwell`` is the tree's ``safety.dwell`` block, read by name: its steady gate holds every move until the
    arm stands still. The jaws open to ``motion.pre_open_width_mm``, else the hand's widest, before every approach,
    except on a hand that toggles with no sensor, which takes no width and is never pulsed before the arm moves: its
    pre-open is dropped and no width is checked. Refused (``ValueError``): a pre-open wider than the hand opens, and a
    pre-open on a service that drives no hand.
    """
    widest = getattr(gripper, "max_width_mm", None) if gripper is not None else None
    pre_open: float | None
    if toggle_without_sensor_of(gripper) is not None:
        pre_open = None
    elif chosen(motion.pre_open_width_mm):
        if gripper is None:
            raise ValueError(
                f"GraspMotion.pre_open_width_mm is {motion.pre_open_width_mm} mm and this service drives no hand, so "
                "there are no jaws to open; leave it unset")
        if widest is not None and float(motion.pre_open_width_mm) > float(widest) + 1e-9:
            raise ValueError(
                f"GraspMotion.pre_open_width_mm is {motion.pre_open_width_mm} mm and this hand opens "
                f"{float(widest)} mm at most")
        pre_open = float(motion.pre_open_width_mm)
    else:
        pre_open = float(widest) if widest is not None else None
    standoff, retreat = motion.standoff_and_retreat(standoff_mm, retreat_mm)
    chosen_fields: dict[str, Any] = {
        name: getattr(motion, name)
        for name in ("approach_steps", "retreat_steps", "close_squeeze_mm", "close_speed", "close_force_n",
                     "align_closing_to_base_x")
        if chosen(getattr(motion, name))
    }
    policy = GraspExecutionPolicy(
        arm=arm,
        gripper=gripper,
        standoff_mm=standoff,
        retreat_mm=retreat,
        pre_open_width_mm=pre_open,
        require_base_frame_grasp=bool(base_frame_required),
        **chosen_fields,
    )
    if dwell is not None:
        policy.require_steady_before_motion = bool(getattr(dwell, "require_steady_before_motion", False))
        policy.steady_timeout_s = float(getattr(dwell, "steady_timeout_s", 5.0))
    return policy


def foreign_policy_refusal(policy: GraspExecutionPolicy, *, arm: object, gripper: object) -> str:
    """Why a caller's own policy cannot drive this service, or the empty string when it can.

    Asked for as long as ``policy=`` is accepted: a policy drives the arm and hand it was built with, so one built
    around another arm or hand would move a robot the service never checked, and it carries none of the service's
    guards. The service's arm and hand are compared by identity.
    """
    if policy.arm is not arm:
        return (f"this policy drives its own arm ({type(policy.arm).__name__}), not the one this service built "
                f"({type(arm).__name__}); pass motion=GraspMotion(...) instead, which the service builds on its own arm "
                "with every guard")
    if policy.gripper is not gripper:
        return (f"this policy drives its own hand ({type(policy.gripper).__name__}), not the one this service built "
                f"({type(gripper).__name__}); pass motion=GraspMotion(...) instead, which the service builds on its "
                "own hand with every guard")
    return ""
