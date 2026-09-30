"""The open hand a push plans with, built from the cell's config and the gripper registry.

:class:`~src.robot.grasping.recovery.push_planner.PushHand` needs the housing's thickness along the closing axis
(``palm_thickness_mm``, 75 mm on the Hand-E), and the cell's gripper model
(:class:`~src.robot.grasping.collision.gripper_model.ParallelJawGripperModel`, or the config block it is built
from) does not carry it: only the hand's registry file does (``config/grippers/<model>.yaml``). Without it the
planner takes the housing as wide as the open fingers' outer faces, 71.6 mm on the Hand-E, 1.7 mm short on each
side. So the push takes its hand from here, never from the model alone.

Nothing here moves the arm, asks a person or touches the jaws. A cell that cannot say which hand it carries, or
carries one that does not push (a suction cup), gets a :class:`~src.robot.grasping.recovery.push_planner.PushRefusal`
before any push is planned, which the pick falls through on like any other refusal before motion.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from src.robot.grasping.recovery.push_planner import PushHand, PushRefusal

__all__ = [
    "REFUSED_PUSH_HAND_NOT_A_JAW",
    "REFUSED_PUSH_HAND_UNKNOWN",
    "push_hand_from_registry",
    "push_hand_from_robot_config",
]

#: The cell names no hand the registry holds, so nothing says how wide its housing is.
REFUSED_PUSH_HAND_UNKNOWN = "push_hand_unknown"
#: The cell's end effector is not a parallel jaw, so it has no finger to push with.
REFUSED_PUSH_HAND_NOT_A_JAW = "push_hand_not_a_jaw"

_PARALLEL_JAW = "parallel_jaw"


def push_hand_from_registry(
    gripper_model: Any, *, hand: Optional[str], open_width_mm: float,
) -> Union[PushHand, PushRefusal]:
    """The push's hand: the fingers and palm width of ``gripper_model``, the housing thickness of registry hand
    ``hand``, and the jaws open ``open_width_mm`` apart.

    ``gripper_model`` is read by field name (``finger_thickness_mm``, ``finger_width_mm``, ``fingertip_depth_mm``,
    ``finger_length_mm``, ``palm_width_mm``), so a :class:`ParallelJawGripperModel` and the
    ``grasping.gripper_geometry.parallel_jaw`` config block both serve. ``hand`` is the registry model name the
    cell carries (``robot.gripper.model``); it is looked up by model name only, as the cell's own lookup is. A
    registry hand that has no ``palm_thickness_mm`` leaves the housing to the planner's fallback, the open
    fingers' outer faces, and the registry file is where to fix that.
    """

    from src.config.grippers import load_gripper  # noqa: PLC0415 (config is read only when a hand is built)
    from src.config.loader import ConfigError  # noqa: PLC0415

    if not hand:
        return PushRefusal(
            code=REFUSED_PUSH_HAND_UNKNOWN,
            sentence=("robot.gripper.model names no hand, so the housing's width along the closing axis is not known "
                      "and no push is planned. Set robot.gripper.model to the registry hand the cell carries, for "
                      "example robotiq_hande."),
        )
    try:
        spec = load_gripper(str(hand), aliases=False)
    except ConfigError as exc:
        return PushRefusal(
            code=REFUSED_PUSH_HAND_UNKNOWN,
            sentence=f"robot.gripper.model {hand!r} is not a hand the gripper registry holds by its model name, so no "
                     f"push is planned ({str(exc).rstrip('.')}).",
        )
    palm = getattr(spec.jaw, "palm_thickness_mm", None)
    return PushHand.from_gripper_model(
        gripper_model, open_width_mm=float(open_width_mm), palm_thickness_mm=None if palm is None else float(palm),
    )


def push_hand_from_robot_config(robot: Any) -> Union[PushHand, PushRefusal]:
    """The push's hand for a loaded robot config: ``robot.grasping.gripper_geometry.parallel_jaw`` (the numbers the
    loader filled from the named hand), ``robot.gripper.max_width_mm`` as the open width (the hand's aperture; a
    toggle hand opens all the way) and the registry's ``palm_thickness_mm`` for ``robot.gripper.model``."""

    gripper = getattr(robot, "gripper", None)
    geometry = getattr(getattr(robot, "grasping", None), "gripper_geometry", None)
    if gripper is None or geometry is None:
        return PushRefusal(
            code=REFUSED_PUSH_HAND_UNKNOWN,
            sentence="The robot config carries no gripper or no gripper geometry, so the hand a push would plan with "
                     "is not known and no push is planned.",
        )
    kind = str(getattr(geometry, "kind", ""))
    if kind != _PARALLEL_JAW:
        return PushRefusal(
            code=REFUSED_PUSH_HAND_NOT_A_JAW,
            sentence=f"robot.grasping.gripper_geometry.kind is {kind!r}, not a parallel jaw: there is no finger to push "
                     "with, so no push is planned.",
        )
    return push_hand_from_registry(
        geometry.parallel_jaw, hand=getattr(gripper, "model", None), open_width_mm=float(gripper.max_width_mm),
    )
