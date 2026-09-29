"""The one view a wrist pick generates, and the move back to the look that saw its part: straight joint lines only.

The owner, 2026-09-29, a safety requirement (addendum 3 of the multi-view plan). When the declared looks of a wrist pick
are used up and its grasp is still not safe enough, the pick may generate one more view, as the very last resort: the
judged look turned about the vertical through the part (``unseen_side.orbit_views``), aimed at the jaw contact face of
the chosen grasp that no view showed, or at the side no look faced when there is no grasp at all. It may turn up to half
a turn, because the face behind a 45 degree look is 135 degrees round, and so it is secured the more it turns:

* smallest turn first, and at most ``unseen_side.ORBIT_MAX_MOTIONS`` motions tried;
* every candidate screened before anything moves toward it: the arm's own ``nearest_configuration``
  (``ChoosesConfigurations``: the inverse kinematics, the cable window, the workspace box, the joint limits,
  self-collision), then no joint travelling more than :data:`MAX_JOINT_TRAVEL_DEG` from where it stands, and every joint
  within :data:`CABLE_WINDOW_DEG` of home on top;
* a WARNING for every generated move past :data:`WARN_BEYOND_DEG`, of the turn about the part or of any joint, said
  once the arm drove it (a line refused before anything was sent drove nothing);
* the motion runs only on the straight joint line from where the arm stands to the screened configuration
  (``DrivesJointLines``), judged against the camera world holding every frame of the pick. A line that is not clear
  skips that turn, and nothing plans around it: no cuRobo at any angle, no retract, no detour, no fallback;
* and it runs only where that world is there to judge it: the arm's live planner world holds the frames of the pick
  (``holding``, what ``keep_out.holding_views`` yielded), no ``without_camera_world`` decline is in scope for the arm,
  and the arm's own reading of the camera world a motion would carry (``ReadsCameraWorld``) is not DECLINED, MISSING or
  UNPLANNED. An arm with no live world, or one that cannot hold the pick's frames, generates nothing.

The move back to the look that saw the part is held to the same line, the same world and the same travel cap from where
the arm stands, and a move back past :data:`WARN_BEYOND_DEG` is said as a WARNING. Where any of it fails it is refused,
nothing sent, and the approach starts from where the arm stands, planned and judged as every approach is.

Both motions are here once, for every caller. The pick loop calls them (``BinPickingOrchestrator``), and the Locator's
look around can call the same two functions: :func:`go_to_generated_view` and :func:`move_back_to_look`. Everything they
decide on is handed in, never read off a pick: the arm, whether the frames of the pick are held in its planner world,
the aim, the judged look's camera and tool, where the arm stands and where home is. What they did comes back as a value:
the turn the arm moved to, or why none; whether the move back ran, or why not. A motion that failed once it may have been
commanded, or on a controller that cannot move, is handed back as ``stopped`` for the caller to end the attempt on, with
nothing else commanded. A camera that could not vouch for the cell on the way raises ``CameraWorldUnavailable``, as on
every motion of a pick, and an arm that cannot say where it stands raises from ``nearest_configuration``.

Nothing here perceives or judges a view: the caller takes the frame where the arm now stands, as at a declared look.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from src.contracts import UNSET
from src.robot.core import NO_PLAN_FAIL_SAFE_MESSAGE, JointPositions, MotionStatus
from src.robot.core.arm_capabilities import ChoosesConfigurations, DrivesJointLines
from src.robot.core.camera_world import CameraWorldUse, ReadsCameraWorld, active_decline
from src.robot.core.errors import CameraWorldUnavailable, RobotKinematicsError
from src.robot.execution.looks import look_label
from src.robot.execution.motion import MotionOutcome, MotionReport, move_joints_on_the_line
from src.robot.grasping.multiview.unseen_side import (
    MAX_INCIDENCE_DEG,
    ORBIT_MAX_DEG,
    ORBIT_MAX_MOTIONS,
    JawFace,
    away_from_views,
    orbit_target,
    orbit_views,
)

__all__ = [
    "CABLE_WINDOW_DEG",
    "MAX_JOINT_TRAVEL_DEG",
    "REFUSED_BEFORE_SENDING",
    "WARN_BEYOND_DEG",
    "GeneratedView",
    "MovedBack",
    "ViewAim",
    "aim_at_faces",
    "aim_at_unseen_side",
    "go_to_generated_view",
    "move_back_to_look",
    "refused_before_sending",
]

_LOG = logging.getLogger(__name__)

#: The most any joint may travel on the way to a generated view, from where it stands, in degrees. The owner's figure: the
#: face behind a 45 degree look is 135 degrees round, which turns the base about as far, and 150 leaves the other joints
#: room to follow while a configuration the long way round, or a wrist wound on by most of a turn, is refused.
MAX_JOINT_TRAVEL_DEG: Final = 150.0
#: A generated move past this, of the turn about the part or of any joint, is said as a WARNING: the old bound of the
#: orbit, past which the owner wants whoever reads the log to see that the arm drove far on a view the pick made up.
WARN_BEYOND_DEG: Final = 120.0
#: How far any joint of a generated view may stand from home, in degrees: half a turn, the cable window. The arm's own
#: screen keeps its configurations inside the window where the cell keeps one; this holds it on top, for every arm.
CABLE_WINDOW_DEG: Final = 180.0
#: Rounding a bound may fall short of and still meet it: a joint exactly at a cap is at it.
_SLACK_DEG = 1e-6
#: The camera worlds an arm's motion may carry that no current camera world stands behind: a motion a pick made up
#: does not run on them.
_UNJUDGED: Final[frozenset[CameraWorldUse]] = frozenset(
    {CameraWorldUse.DECLINED, CameraWorldUse.MISSING, CameraWorldUse.UNPLANNED})

#: How a guard or the planner refuses a motion before anything is sent: the arm stands where it stood. Only these, and
#: the planner's no-plan refusal (``NO_PLAN_FAIL_SAFE_MESSAGE``), let a pick skip a motion and go on (the owner,
#: 2026-09-29). Every other failed motion may have been commanded (a controller that rejected a sent ``moveJ``, a
#: timeout, a link that dropped, a failure nobody classified), so the arm may have moved part of the way or may still be
#: moving, and the attempt ends there. ``INVALID_TARGET`` is left out: a driver names with it a waypoint refused after
#: earlier ones of the same path ran.
REFUSED_BEFORE_SENDING: Final[frozenset[MotionStatus]] = frozenset({
    MotionStatus.WORKSPACE_REJECTED,
    MotionStatus.IK_FAILED,
    MotionStatus.IK_QUALITY_REJECTED,
    MotionStatus.JOINT_LIMIT_REJECTED,
    MotionStatus.SELF_COLLISION_REJECTED,
    MotionStatus.PAYLOAD_REJECTED,
    MotionStatus.CONTINUITY_REJECTED,
})


def refused_before_sending(moved: MotionReport) -> bool:
    """Whether a failed motion was refused by a guard or the planner before anything was sent (:data:`REFUSED_BEFORE_SENDING`).

    A timeout is the planner's refusal only in its own words (``NO_PLAN_FAIL_SAFE_MESSAGE``); otherwise a sent motion ran
    out of time.
    """
    if moved.status is MotionStatus.TIMEOUT:
        return moved.message == NO_PLAN_FAIL_SAFE_MESSAGE
    return moved.status in REFUSED_BEFORE_SENDING


# ---------------------------------------------------------------------------------------------------------------------
# What a generated view is to show
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class ViewAim:
    """What a generated view is to show, and the point it turns about, BASE mm.

    ``faces`` are ``(centre, outward normal)`` pairs as ``unseen_side.orbit_views`` takes them, and ``what`` says them in
    the owner's words, for the log and the report: a contact face of the chosen grasp, or the side no look faced.
    """

    target_mm: np.ndarray
    faces: tuple[tuple[np.ndarray, np.ndarray], ...]
    what: str


def _jaw(face: JawFace) -> int:
    """The jaw a face is closed on: 1 on the face the closing axis points away from, 2 on the one it points to."""
    return 1 if face.side < 0 else 2


def aim_at_faces(missing: Sequence[JawFace], *, grasp_centre_mm: Any) -> ViewAim:
    """Aim at the jaw contact faces of the chosen grasp that no view showed, turning about the grasp's centre.

    The centre lies between the two contact faces, so a turn about it keeps both in view. ``missing`` is
    ``JawFacesSeen.missing``; an empty one names nothing to show and raises ``ValueError``.
    """
    if not missing:
        raise ValueError("aim_at_faces: both contact faces were seen, so there is no face to show")
    target = orbit_target(np.zeros((0, 3)), grasp_centre_mm)
    faces = tuple(
        (np.asarray(face.centre_mm, dtype=np.float64), np.asarray(face.normal, dtype=np.float64)) for face in missing)
    jaws = " and ".join(f"jaw {_jaw(face)}" for face in missing)
    noun = "contact face" if len(missing) == 1 else "contact faces"
    return ViewAim(target_mm=target, faces=faces, what=f"the {noun} at {jaws} of the chosen grasp")


def aim_at_unseen_side(points_base_mm: Any, *, camera_centres_mm: Sequence[Any]) -> ViewAim:
    """Aim at the side of the part no look faced, when there is no grasp: about the middle of its cloud.

    The one face is the middle of the cloud (``unseen_side.orbit_target``) facing the widest gap between the looks'
    bearings (``unseen_side.away_from_views``). Raises ``ValueError`` for a cloud with no finite point, or looks none of
    which stands to one side of it.
    """
    target = orbit_target(points_base_mm)
    normal = away_from_views(target, camera_centres_mm)
    return ViewAim(target_mm=target, faces=((target, normal),), what="the side of the part no look faced")


# ---------------------------------------------------------------------------------------------------------------------
# The generated view
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class GeneratedView:
    """What a generated view came to.

    ``azimuth_deg`` is the turn about the part the arm moved to and ``joints`` the configuration it stands at now; both
    ``None`` when it did not move. ``label`` names the view as a look is named, ``orbit +135 deg (...) deg``, also the one
    a motion was stopped on. ``skipped`` says why no view was generated, ``""`` when one was or a motion stopped it.
    ``tried`` is every turn given up on, in order, and why: screened out, or its straight joint line not clear.
    ``stopped`` is a motion that failed once it may have been commanded, or one refused on a controller that cannot move
    (``controller`` says why): the caller ends the attempt there, with nothing else commanded.
    """

    azimuth_deg: float | None = None
    joints: JointPositions | None = None
    label: str = ""
    skipped: str = ""
    tried: tuple[tuple[float, str], ...] = ()
    stopped: MotionReport | None = None
    controller: str | None = None

    @property
    def moved(self) -> bool:
        """Whether the arm stands at the generated view."""
        return self.joints is not None and self.stopped is None


def _degrees(joints: Any) -> list[float]:
    """Joints in degrees: a ``JointPositions``, or a sequence of radians (``home_joint_positions``)."""
    if isinstance(joints, JointPositions):
        return [float(value) for value in joints.degrees()]
    return [math.degrees(float(value)) for value in joints]


def _not_judged(arm: Any, *, holding: bool) -> str:
    """Why a motion ``arm`` made now would not be judged against the camera world holding every frame of the pick, or
    ``""`` when it would: the frames held (``holding``), no decline in scope for the arm, and, where the arm says what a
    motion would carry (``ReadsCameraWorld``), a camera world that is not DECLINED, MISSING or UNPLANNED."""
    if not holding:
        return ("the arm's planner world does not hold the frames of this pick (keep_out.holding_views: no live world, "
                "or none with a camera on the wrist), so the motion would not be judged against every frame the pick "
                "took")
    decline = active_decline(arm)
    if decline is not None:
        return (f"the camera world is declined for this arm ({decline.reason}), and a motion the pick made up runs "
                "judged against the camera world holding every frame of the pick or not at all")
    if isinstance(arm, ReadsCameraWorld):
        stamp = arm.camera_world_for_move(UNSET)
        if stamp.use in _UNJUDGED:
            return f"no camera world would judge the motion ({stamp.render()})"
    return ""


def _why_not(arm: Any, *, home: Any, holding: bool) -> str:
    """Why this arm generates no view at all, before any turn is asked about; ``""`` when it may."""
    if not isinstance(arm, ChoosesConfigurations):
        return (f"{type(arm).__name__} does not name the configuration a pose goes to (ChoosesConfigurations), so no "
                "turn can be screened before the arm moves")
    if not isinstance(arm, DrivesJointLines):
        return (f"{type(arm).__name__} does not drive a straight joint line alone (DrivesJointLines), and a generated "
                "view runs on nothing else")
    if home is None:
        return "the arm names no home, so the cable window about it cannot be held"
    return _not_judged(arm, holding=holding)


def _past_the_travel_cap(target: list[float], here: list[float], *, what: str) -> str:
    """Why ``target`` is not driven to from ``here``, or ``""``: a joint would travel past :data:`MAX_JOINT_TRAVEL_DEG`,
    joint by joint, in degrees. ``what`` names the motion in the reason."""
    if not target or len(target) != len(here):
        return (f"the configuration names {len(target)} joints and the arm stands on {len(here)}, so no travel can be "
                "read")
    for axis, (goal, now) in enumerate(zip(target, here, strict=True)):
        travel = abs(goal - now)
        if travel > MAX_JOINT_TRAVEL_DEG + _SLACK_DEG:
            return (f"joint {axis + 1} would travel {travel:.1f} deg from where it stands, more than the "
                    f"{MAX_JOINT_TRAVEL_DEG:g} deg {what} may turn any joint")
    return ""


def _past_the_cable_window(target: list[float], home: list[float]) -> str:
    """Why ``target`` is not driven to, or ``""``: a joint would stand more than :data:`CABLE_WINDOW_DEG` from home."""
    if len(target) != len(home):
        return f"the configuration names {len(target)} joints and home {len(home)}, so the cable window cannot be read"
    for axis, (goal, centre) in enumerate(zip(target, home, strict=True)):
        off = abs(goal - centre)
        if off > CABLE_WINDOW_DEG + _SLACK_DEG:
            return (f"joint {axis + 1} would stand {off:.1f} deg from home, past the half turn "
                    f"({CABLE_WINDOW_DEG:g} deg) the cable allows")
    return ""


def _a_long_way(target: list[float], here: list[float], *, turn: float | None = None) -> str:
    """The WARNING's words for a move past :data:`WARN_BEYOND_DEG`, of the turn about the part (``turn``, a generated
    view's) or of any joint from ``here``; ``""`` for a move within it."""
    worst = max(range(len(target)), key=lambda axis: abs(target[axis] - here[axis]))
    travel = abs(target[worst] - here[worst])
    turned = abs(turn) if turn is not None else 0.0
    if turned <= WARN_BEYOND_DEG and travel <= WARN_BEYOND_DEG:
        return ""
    about = f"a turn of {turned:.0f} deg about the part, " if turn is not None else ""
    return f"{about}joint {worst + 1} travelling {travel:.1f} deg, past {WARN_BEYOND_DEG:g} deg"


def _on_the_line(arm: Any, joints: JointPositions) -> MotionReport:
    """The straight joint line to ``joints``, and nothing else; a camera that could not vouch on the way raises."""
    moved = move_joints_on_the_line(arm, joints)
    if moved.outcome is MotionOutcome.CAMERA_WORLD_UNAVAILABLE and isinstance(
            moved.result.exception, CameraWorldUnavailable):
        raise moved.result.exception
    return moved


def _skipped(why: str, tried: Sequence[tuple[float, str]] = ()) -> GeneratedView:
    _LOG.info("no view is generated: %s", why)
    return GeneratedView(skipped=why, tried=tuple(tried))


def go_to_generated_view(
    arm: Any,
    *,
    aim: ViewAim,
    camera_to_base_mm: Any,
    tool_to_base_mm: Any,
    intrinsics: Any,
    image_size: tuple[int, int],
    here: JointPositions,
    home: Any,
    holding: bool,
    controller_stopped: Callable[[], str | None] | None = None,
    max_motions: int = ORBIT_MAX_MOTIONS,
) -> GeneratedView:
    """Move ``arm`` to the one generated view, on the straight joint line only, or say why not. Nothing is perceived.

    The candidates are the judged look turned about ``aim.target_mm`` so that some face of ``aim`` shows
    (``unseen_side.orbit_views``, up to ``ORBIT_MAX_DEG`` either way, smallest turn first): ``camera_to_base_mm`` is
    the judged frame's camera, ``tool_to_base_mm`` the tool pose stamped at its shutter, ``intrinsics`` (3x3) and
    ``image_size`` (width, height) its lens and image, all BASE millimetres. For each, in order, before anything moves
    toward it: ``arm.nearest_configuration`` (a ``RobotKinematicsError`` gives the turn up), then no joint more than
    :data:`MAX_JOINT_TRAVEL_DEG` from ``here``, where the arm stands, and none more than :data:`CABLE_WINDOW_DEG` from
    ``home`` (joints, or radians as ``home_joint_positions`` gives them). Then the straight joint line to it
    (``motion.move_joints_on_the_line``), judged by the arm against its camera world. A line refused before anything was
    sent gives the turn up and the next is tried; at most ``max_motions`` lines are driven, and a turn the screen gave
    up sent nothing and does not count. A move driven past :data:`WARN_BEYOND_DEG`, of the turn or of a joint, is said
    as a WARNING once the arm stands there.

    Nothing is generated, with the reason in ``skipped``, for an arm that does not name configurations
    (``ChoosesConfigurations``: the simulator's, the desk arm's) or does not drive a straight joint line alone
    (``DrivesJointLines``), one that names no home, one whose motion would not be judged against the camera world
    holding every frame of the pick (``holding`` false, as ``keep_out.holding_views`` yields for an arm with no live
    world or one that cannot hold the pick's frames; a ``without_camera_world`` decline in scope; a camera world the arm
    reads as DECLINED, MISSING or UNPLANNED), a judged look that cannot be turned (a camera, tool or lens that is no
    transform), or when no turn shows the aim or none of them can be driven. A
    motion that may have moved the arm, or one refused on a controller ``controller_stopped`` says cannot move, is
    handed back as ``stopped`` with no other turn tried. Raises ``CameraWorldUnavailable`` from the motion, and what
    ``nearest_configuration`` raises that is not a refusal of the pose (``RobotConnectionError``, ``FrameMismatchError``).
    """
    why = _why_not(arm, home=home, holding=holding)
    if why:
        return _skipped(why)
    try:
        views = orbit_views(
            target_mm=aim.target_mm, camera_to_base_mm=camera_to_base_mm, tool_to_base_mm=tool_to_base_mm,
            faces=aim.faces, intrinsics=intrinsics, image_size=image_size,
        )
    except ValueError as exc:
        return _skipped(f"the judged look cannot be turned: {exc}")
    if not views:
        return _skipped(f"no turn about the part within {ORBIT_MAX_DEG:g} deg shows {aim.what} at "
                        f"{MAX_INCIDENCE_DEG:g} deg of incidence or less, inside the image")
    start, centre = _degrees(here), _degrees(home)
    tried: list[tuple[float, str]] = []
    motions = 0
    for view in views:
        if motions >= max_motions:
            break
        try:
            joints = arm.nearest_configuration(view.pose())
        except RobotKinematicsError as exc:
            tried.append((view.azimuth_deg, str(exc)))
            _LOG.debug("generated view %s: screened out: %s", view.label, exc)
            continue
        target = _degrees(joints)
        refusal = _past_the_travel_cap(target, start, what="a generated view") or _past_the_cable_window(target, centre)
        label = f"{view.label} {look_label(joints)}"
        if refusal:
            tried.append((view.azimuth_deg, refusal))
            _LOG.debug("generated view %s: screened out: %s", label, refusal)
            continue
        long_way = _a_long_way(target, start, turn=view.azimuth_deg)
        motions += 1
        moved = _on_the_line(arm, joints)
        if moved.ok:
            if long_way:
                _LOG.warning("generated view %s, to show %s: the arm drove there on the straight joint line only, %s: a "
                             "long way for a view the pick made up (%d motion(s) tried)", label, aim.what, long_way,
                             motions)
            else:
                _LOG.info("generated view %s, to show %s: the arm stands there, on the straight joint line (%d "
                          "motion(s) tried)", label, aim.what, motions)
            return GeneratedView(azimuth_deg=view.azimuth_deg, joints=joints, label=label, tried=tuple(tried))
        controller = controller_stopped() if controller_stopped is not None else None
        if controller is not None or not refused_before_sending(moved):
            _LOG.warning("generated view %s: the motion failed (%s: %s)%s%s; no other turn is tried", label,
                         moved.status.value, moved.message, f", and {controller}" if controller is not None else "",
                         f"; it was to be {long_way}" if long_way else "")
            return GeneratedView(label=label, tried=tuple(tried), stopped=moved, controller=controller)
        why = (f"its straight joint line was refused before anything was sent, not clear or not admitted "
               f"({moved.status.value}: {moved.message})")
        tried.append((view.azimuth_deg, why))
        _LOG.info("generated view %s: %s; the next turn is tried", label, why)
    if motions >= max_motions:
        why = f"{motions} motion(s) tried, the most a generated view may, and none ran"
    else:
        why = f"no turn that shows {aim.what} could be driven"
    first = f"; the first, {tried[0][0]:+g} deg: {tried[0][1]}" if tried else ""
    return _skipped(f"{why} ({len(tried)} turn(s) given up){first}", tried)


# ---------------------------------------------------------------------------------------------------------------------
# The move back
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class MovedBack:
    """What the move back to the look that saw the part came to.

    ``done`` when the arm stands there again. ``refused`` says why it did not go, nothing sent: the caller approaches
    from where the arm stands. ``stopped`` is a motion that failed once it may have been commanded, or one refused on a
    controller that cannot move (``controller`` says why): the caller ends the attempt there, with nothing else
    commanded.
    """

    done: bool = False
    refused: str = ""
    stopped: MotionReport | None = None
    controller: str | None = None


def move_back_to_look(
    arm: Any, joints: JointPositions | None, *, look: str, here: JointPositions | None, holding: bool,
    controller_stopped: Callable[[], str | None] | None = None,
) -> MovedBack:
    """Move ``arm`` back to the configuration it stood at when ``look`` saw the part, on the straight joint line only.

    ``joints`` is that configuration, read where the frame was taken; ``None`` when it could not be read. ``here`` is
    where the arm stands now, read off it (``None`` when it could not say), and ``holding`` whether its planner world
    holds every frame of the pick (``keep_out.holding_views``). The line is judged by the arm against that camera world,
    and nothing plans around it. Refused (``refused``, nothing sent) for joints nobody read, for an arm that does not
    drive a straight joint line alone (``DrivesJointLines``), for a motion that would not be judged against the camera
    world holding every frame of the pick (as for the generated view), for an arm that cannot say where it stands, for a
    joint that would travel more than :data:`MAX_JOINT_TRAVEL_DEG` from ``here``, and for a line refused before anything
    was sent. A move back driven past :data:`WARN_BEYOND_DEG` on any joint is said as a WARNING. A motion that may have
    moved the arm, or one refused on a controller ``controller_stopped`` says cannot move, comes back as ``stopped``.
    Raises ``CameraWorldUnavailable``.
    """
    if joints is None:
        return MovedBack(refused=f"the joints the arm stood at for look {look} were not read")
    if not isinstance(arm, DrivesJointLines):
        return MovedBack(refused=(f"{type(arm).__name__} does not drive a straight joint line alone (DrivesJointLines),"
                                  " and the move back runs on nothing else"))
    unjudged = _not_judged(arm, holding=holding)
    if unjudged:
        return MovedBack(refused=unjudged)
    if here is None:
        return MovedBack(refused="the arm could not say where it stands, so the travel back cannot be read")
    target, start = _degrees(joints), _degrees(here)
    refusal = _past_the_travel_cap(target, start, what="the move back")
    if refusal:
        return MovedBack(refused=refusal)
    long_way = _a_long_way(target, start)
    moved = _on_the_line(arm, joints)
    if moved.ok:
        if long_way:
            _LOG.warning("moved back to look %s on the straight joint line only, %s: a long way back to the look that "
                         "saw the part", look, long_way)
        else:
            _LOG.info("moved back to look %s on the straight joint line", look)
        return MovedBack(done=True)
    controller = controller_stopped() if controller_stopped is not None else None
    if controller is not None or not refused_before_sending(moved):
        return MovedBack(stopped=moved, controller=controller)
    return MovedBack(refused=(f"its straight joint line was refused before anything was sent "
                              f"({moved.status.value}: {moved.message})"))
