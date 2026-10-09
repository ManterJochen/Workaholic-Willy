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

A look the arm already stands at is not driven to (:func:`move_to_look`): where the controller is connected, the arm is
not halted and at rest, every joint reads within :data:`AT_THE_LOOK_DEG` of the look's, and the tool stands where the
arm's own kinematics put it at the look (:data:`AT_THE_LOOK_MM`), the move there would have no length, and nothing is
sent. The verb's own gates still answer first, the route, the camera world and the steady gate, and the report says the
arm was already there (EXECUTED, :data:`ALREADY_THERE`), in the log as well. The owner's cell paid 1.0 to 1.4 s for such
a move at every part (2026-10-08), because its first look is its home, and so did the bin's survey and a Restart's first
look. Anything that cannot be read, or a halted arm, moves as it always did, and the verb judges and says. Only an arm
that drives a controller, its motions planned and judged by the exact guard (the route PLANNED, its capabilities not
simulated), skips: a simulated or desk arm sets its pose and moves nothing, and its joints need not say where its tool
stands. Nothing moves either way: the frame taken there is placed by the tool pose the arm reports, which is why the
tool has to say it stands at the look as well, and the next motion builds its world and is judged as every motion is.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Final, Literal, Union

import numpy as np

from src.contracts import UNSET
from src.geometry import Pose
from src.robot.constants import create_robot_logger
from src.robot.core.camera_world import ReadsCameraWorld
from src.robot.core.capabilities import RobotCapabilities
from src.robot.core.joint_positions import JointPositions
from src.robot.core.motion_result import MotionCommand, MotionResult
from src.robot.execution import motion as _motion

__all__ = ["ALREADY_THERE", "AT_THE_LOOK_DEG", "AT_THE_LOOK_MM", "HOME", "Look", "LookPose", "look_label", "looks_of",
           "move_to_look"]

_LOG = create_robot_logger(__name__, "looks.log")

#: How near every joint has to read to a look, degrees, for the move there to have no length and nothing to be sent: far
#: above the noise of a resting UR's joint readings, far below any two looks a program declares.
AT_THE_LOOK_DEG: Final[float] = 0.05
#: How near the tool pose the arm reports has to stand to the one its own kinematics give the look, millimetres, and
#: :data:`_AT_THE_LOOK_TURN_DEG` degrees: a frame is placed by the tool pose the arm reports, so the tool has to say it
#: stands at the look too. A UR resting at its look reads its tool there to a hundredth of a millimetre.
AT_THE_LOOK_MM: Final[float] = 1.0
_AT_THE_LOOK_TURN_DEG: Final[float] = 0.1
#: What the report of a look the arm already stood at says: the move had no length, and nothing reached the controller.
ALREADY_THERE: Final[str] = "already there; nothing sent"

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
    """Move ``arm`` to ``pose`` through its own judged verb: ``move_to_joints``, or the gated home verb for :data:`HOME`.

    Where the arm already stands there (:func:`_stands_at`), nothing is sent: the verb's own gates answer as they would
    for the move, and the report is EXECUTED, :data:`ALREADY_THERE` (:func:`_already_there`).
    """
    joints = _joints_of(arm, pose)
    if joints is not None and _stands_at(arm, joints):
        return _already_there(arm, pose, joints)
    if isinstance(pose, JointPositions):
        return _motion.move_joints(arm, pose)
    return _motion.home(arm)


def _joints_of(arm: Any, pose: LookPose) -> JointPositions | None:
    """The joints ``pose`` stands ``arm`` in: the look's own, or for :data:`HOME` the home the arm says its home verb
    drives to (``home_joint_positions``, the UR's); ``None`` where the arm does not say."""
    if isinstance(pose, JointPositions):
        return pose
    try:
        home = getattr(arm, "home_joint_positions", None)
        return None if home is None else JointPositions([float(v) for v in home])
    except Exception:  # noqa: BLE001 (a home the arm cannot say is driven to, as it always was)
        return None


def _stands_at(arm: Any, joints: JointPositions) -> bool:
    """Whether ``arm`` stands at ``joints`` now, read off the arm: a controller's arm (capabilities not simulated) whose
    motions are planned and judged (the route PLANNED), connected, not halted (any latch it answers with), at rest
    (``wait_until_steady`` asked to wait for nothing), every joint within :data:`AT_THE_LOOK_DEG` of ``joints``, and its
    tool where its own kinematics put the tool at ``joints`` (``fk``), within :data:`AT_THE_LOOK_MM` and
    :data:`_AT_THE_LOOK_TURN_DEG`. A reading that raises, or is not what it says it is, is ``False``, and the look is
    driven to as it always was."""
    try:
        capabilities = getattr(arm, "capabilities", None)
        if not isinstance(capabilities, RobotCapabilities) or capabilities.is_simulated is not False:
            return False
        if _motion.route_of(arm).route is not _motion.MotionRoute.PLANNED:
            return False
        if getattr(arm, "is_connected", False) is not True:
            return False
        latch = getattr(arm, "halt_state", None)
        if callable(latch) and latch() is not None:
            return False
        steady = getattr(arm, "wait_until_steady", None)
        if not callable(steady) or steady(0.0) is not True:
            return False
        here = arm.get_joint_positions()
        if not isinstance(here, JointPositions) or here.dof != joints.dof:
            return False
        apart = np.abs(np.asarray(here.values, dtype=np.float64) - np.asarray(joints.values, dtype=np.float64))
        if not (np.all(np.isfinite(apart)) and np.all(apart <= math.radians(AT_THE_LOOK_DEG))):
            return False
        tool, there = arm.get_tcp_pose(), arm.fk(joints)
        if not isinstance(tool, Pose) or not isinstance(there, Pose):
            return False
        return (tool.distance_to(there) <= AT_THE_LOOK_MM
                and tool.angle_to(there) <= math.radians(_AT_THE_LOOK_TURN_DEG))
    except Exception:  # noqa: BLE001 (an arm that cannot say where it stands is moved, and its verb judges and says)
        return False


def _already_there(arm: Any, pose: LookPose, joints: JointPositions) -> _motion.MotionReport:
    """The report of a look ``arm`` already stands at: the verb's own gates, each as the move would have met it (the
    link, the camera world, the route, the steady gate: ``motion._Motion``, the one place they are kept), then nothing
    sent; EXECUTED with :data:`ALREADY_THERE`, said in the log. The stamp is the camera world the move would have
    carried, read as the arm stands: no refresh was made for it."""
    home = not isinstance(pose, JointPositions)
    verb = _motion.MotionVerb.HOME if home else _motion.MotionVerb.MOVE_JOINTS
    command = MotionCommand.MOVE_HOME if home else MotionCommand.MOVE_JOINTS

    def nothing_sent() -> MotionResult:
        if isinstance(arm, ReadsCameraWorld):
            return MotionResult.executed(command, target_joints=joints, message=ALREADY_THERE,
                                         camera_world=arm.camera_world_for_move(UNSET))
        return MotionResult.executed(command, target_joints=joints, message=ALREADY_THERE)

    report = _motion._Motion(arm, verb, UNSET, target_joints=joints).run(nothing_sent)
    if report.ok:
        _LOG.info("look %s: the arm already stands there, every joint within %.2f deg of it; nothing was sent",
                  look_label(pose), AT_THE_LOOK_DEG)
    return report
