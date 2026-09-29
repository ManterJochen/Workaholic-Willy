"""Where a camera looks from before a pick: joint positions the program declares, visited in order.

The owner, 2026-09-24: the look poses of a wrist camera are written in the Python program, as joints in degrees, and
handed to each pick; one or a list, visited in order; home from the config when none is given. They replace the viewing
pose the software used to generate from a work point, a distance and a heading, whose answer a person could only check
once the arm stood there.

A look is a :class:`~src.robot.core.JointPositions` (``JointPositions.deg`` takes the pendant's degrees, and
``python -m src.robot.drivers.ur --where`` prints them), or :data:`HOME`, the arm's own configured home. Joints rather
than a Cartesian pose, because joints are the configuration a person stood the arm in and checked: a Cartesian look
leaves the IK branch, the cable window and the camera's roll about its own axis to a solver.

A pick handed looks moves to each one with the arm's own judged verb (:func:`move_to_look`: ``move_to_joints``, the
straight joint line first and a guarded plan around it, or the gated home verb for :data:`HOME`) and perceives there. A
fixed camera's pick stops at the first look that finds something. On a camera on the wrist the pick loop fuses every
look with the ones before it and stops at the first look whose grasp is safe (``BinPickingOrchestrator.look_around``,
the owner's rule of 2026-09-29: the looks exist to find a safe grasp point, never to scan the part whole). A look a
guard or the planner refused before anything was sent is skipped and said; a pick that reaches none of its looks ends
with nothing perceived and nothing else commanded, and says which look it was. The one view a pick may generate after
its looks, and its move back to the look that saw the part, are not looks declared here: they run on the straight joint
line alone (``src.robot.execution.generated_view``). A camera that could not vouch for the cell on the way is a fault of
the cell, as it is on every other motion of a pick. Nothing is said to the hand before or between looks: a look is a
motion of the arm, and the jaws stay as the last pick left them.

A raw list of numbers is refused as a look, because its unit cannot be told: degrees read as radians is the mistake the
desk check and the UR driver already name for ``robot.home_joint_positions``, and a look is where the arm goes next.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final, Literal, Union

from src.robot.core.joint_positions import JointPositions
from src.robot.execution import motion as _motion

__all__ = ["HOME", "Look", "LookPose", "look_label", "looks_of", "move_to_look"]

#: The arm's own configured home as a look: ``robot.home_joint_positions`` (or ``home_joint_positions_deg``), reached
#: through the arm's gated home verb. What a wrist camera looks from when a program names no look.
HOME: Final = "home"

#: One place to look from.
LookPose = Union[JointPositions, Literal["home"]]
#: What a pick takes: one look, or several tried in order.
Look = Union[LookPose, Sequence[LookPose]]


def looks_of(look: Any) -> tuple[LookPose, ...]:
    """``look`` as the looks it names, in order; a list that names none, and an entry that is not a look, raise.

    Raised rather than reported, before anything moves: a look list is written in the program, and a wrong one is the
    program's error, not a state of the cell.
    """
    entries: list[Any] = [look] if isinstance(look, (JointPositions, str)) else list(look)
    if not entries:
        raise ValueError("a list of looks names no pose to look from: give one JointPositions, several, or 'home'")
    for entry in entries:
        if isinstance(entry, str) and entry != HOME:
            raise ValueError(f"a look is JointPositions or {HOME!r}, not {entry!r}")
        if not isinstance(entry, (JointPositions, str)):
            raise TypeError(
                f"a look is JointPositions or {HOME!r}, not {type(entry).__name__}: write the joints as "
                f"JointPositions.deg(...) in degrees, or JointPositions([...]) in radians, so their unit is said")
    return tuple(entries)


def look_label(pose: LookPose) -> str:
    """How a look reads in a report: ``home``, or the joints in degrees."""
    if isinstance(pose, JointPositions):
        return "(" + ", ".join(f"{v:.1f}" for v in pose.degrees()) + ") deg"
    return HOME


def move_to_look(arm: Any, pose: LookPose) -> _motion.MotionReport:
    """Move ``arm`` to ``pose`` through its own judged verb: ``move_to_joints``, or the gated home verb for :data:`HOME`."""
    if isinstance(pose, JointPositions):
        return _motion.move_joints(arm, pose)
    return _motion.home(arm)
