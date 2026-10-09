"""Universal Robots :class:`RobotArm` implementation."""

from __future__ import annotations

import asyncio
import dataclasses
import math
import re
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING, Sequence

import numpy as np

from src.config.schema.robot import RobotConfig
from src.contracts import UNSET, Maybe, chosen
from src.geometry import Frame, FrameMismatchError, GeometryError, Pose
from src.geometry.quaternion import angle_between, from_rotation_matrix, slerp
from src.robot.constants import HOME_JOINTS_DEFAULT, UR_ARM_LOG_FILE, create_robot_logger
from src.robot.core import (
    NO_PLAN_FAIL_SAFE_MESSAGE,
    CameraWorldDecline,
    CameraWorldStamp,
    ControllerPayload,
    DigitalIOPort,
    FreedriveSession,
    JointPositions,
    MotionCommand,
    MotionResult,
    MotionStatus,
    RobotArm,
    RobotCapabilities,
    RobotConnectionError,
    RobotEmergencyStop,
    RobotKinematicsError,
    RobotMode,
    RobotMotionRejected,
    RobotStatus,
    SafetyMode,
    Wrench,
    active_decline,
    camera_world_refusal,
    resolve_camera_world,
    stamp_result,
)
from src.robot.core.arm_capabilities import HaltState, LineMotion, LineReading, PayloadModel, halted_refusal
from src.robot.core.camera_world import without_camera_world as _without_camera_world
from src.robot.core.gripper import why_not_known_open
from src.robot.safety import (
    LineSamples,
    PathSamples,
    SafetyPreflight,
    line_samples,
)
from src.robot.safety._ur_ik import NearestGoal, URBranch, nearest_goals, nearest_turn, ur_branch, ur_flange_ik
from src.robot.safety._ur_kinematics import ur_link_transforms_mm, ur_series_twin
from src.robot.safety.path_samples import (
    LINE_MAX_CHORD_OFF_MM,
    LINE_MAX_JOINT_STEP_RAD,
    MAX_PATH_SAMPLES,
    joint_path_samples,
    span_overshoot_rad,
)
from src.robot.safety.planning import (
    CuroboPlanClient,
    CuroboUnavailableError,
    StateRefusalKind,
    StateWhere,
)
from src.robot.safety.planning._curobo_protocol import PERCEIVED_PREFIX
from src.robot.safety.planning.band import (
    BAND_LEG_MAX_DEG,
    PoseScreen,
    PoseVerdict,
    admission_refusal,
    band_neighbours,
    band_sentence,
    joints_between,
    world_admission_refusal,
)
from src.robot.safety.planning.hand import planner_hand
from src.robot.safety.workspace import WorkspaceGuard

from .connection import HALT_ENDS, MoveEnd, URConnection
from .freedrive import HAND_GUIDED_MESSAGE, URFreedriveSession, raise_unless_operational
from .motion import MotionController
from .pose import URPose
from .pose_adapter import pose_to_urpose, urpose_to_pose

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from collections.abc import Callable, Sequence
    from typing import Any

    from src.robot.safety.planning.band import ExactPairs
    from src.robot.safety.planning.curobo_client import RefusedSample, SidecarIdentity
    from src.robot.safety.planning.live_world import (
        LivePlannerWorld,
        WorldRefresh,
    )
    from src.robot.safety.planning.perceived import SelfEnvelope

    from .curobo_motion import CuroboUrPlanner

__all__ = ["UR_CAPABILITIES", "URRobotArm", "ur_capabilities"]


def ur_capabilities(model: str = "ur5e") -> RobotCapabilities:
    """The capability surface of this vendor for ``model``.

    The capability record is what downstream code reads to learn which robot it is
    holding, so a hardcoded "ur5e" would make every model-keyed lookup, the DH table,
    the mesh bundle and the cuRobo config, resolve the wrong robot on a real UR3e.
    Everything except the model string is shared across the e-series.
    """
    return dataclasses.replace(UR_CAPABILITIES, model=model)


UR_CAPABILITIES = RobotCapabilities(
    vendor="ur",
    model="ur5e",
    dof=6,
    supports_joint_move=True,
    supports_linear_move=True,
    supports_async_move=True,
    has_native_fk=True,
    has_native_ik=True,
    has_force_control=True,  # backed: get_tcp_wrench / get_joint_torques via SupportsForceTorque
    is_simulated=False,
)

# UR controller mode integers to the vendor-neutral enums. UR getRobotMode runs from -1
# for no controller and 0 for disconnected up to 8 for updating firmware; getSafetyMode
# runs from 1 for normal to 9 for fault, and 10 to 13, the validate, undefined, auto-mode
# and three-position stops, map to unknown here. An unmapped integer falls back to
# unknown.
_UR_ROBOT_MODE: dict[int, RobotMode] = {
    0: RobotMode.DISCONNECTED, 1: RobotMode.CONFIRM_SAFETY, 2: RobotMode.BOOTING,
    3: RobotMode.POWER_OFF, 4: RobotMode.POWER_ON, 5: RobotMode.IDLE,
    6: RobotMode.BACKDRIVE, 7: RobotMode.RUNNING, 8: RobotMode.UPDATING_FIRMWARE,
}
_UR_SAFETY_MODE: dict[int, SafetyMode] = {
    1: SafetyMode.NORMAL, 2: SafetyMode.REDUCED, 3: SafetyMode.PROTECTIVE_STOP,
    4: SafetyMode.RECOVERY, 5: SafetyMode.SAFEGUARD_STOP, 6: SafetyMode.SYSTEM_EMERGENCY_STOP,
    7: SafetyMode.ROBOT_EMERGENCY_STOP, 8: SafetyMode.VIOLATION, 9: SafetyMode.FAULT,
}

#: Why a cuRobo ``move`` says MISSING while no live camera world is wired to the arm.
_UR_NO_LIVE_WORLD = "robot.ur.motion_planner is 'curobo' and no live camera world is wired to this arm"

#: The exact guard's parts that hang on the flange, wrist 3 with them: on a straight line of the TCP they stand where
#: the TCP puts them, whichever configuration the line starts from. The wrist cameras (``wrist_camera_<rig>``) too.
_ON_THE_FLANGE = frozenset({"gripper", "lfinger", "rfinger", "wrist_3"})
_WRIST_CAMERA_PART = "wrist_camera_"
#: The pair the exact guard names in a refusal against a fixture: ``<part>|fixture:<name>``.
_FIXTURE_PAIR = re.compile(r"(?P<part>[A-Za-z0-9_\-]+)\|fixture:")
#: What a move says that was refused only because it keeps the branch the arm holds (:meth:`URRobotArm.
#: keeping_its_branch`, the owner's R2 of 2026-10-08): a change of branch is the last resort.
_KEEPS_ITS_BRANCH = "kept to the branch the arm holds"

#: The interpreter's switch interval while a second thread judges the next leg during a watched line
#: (:meth:`URRobotArm._judging_during`). The judgement gives up and takes the GIL around its exact-guard queries, and at
#: Python's 5 ms the thread that watches the line, waking every 8 ms to read the halt, waited for it up to 451 ms on
#: Windows and 521 ms in WSL; at 0.5 ms, 1.0 and 1.6 ms, the judgement no slower (2026-10-09, a UR10 joint move of
#: 1.2 s). The interval before is put back once the last such judgement ended.
_WATCHED_SWITCH_S = 0.0005
_SWITCH_LOCK = threading.Lock()
#: How many judgements during motion run now, and the interval before the first of them shortened it.
_SWITCH_STATE: "dict[str, float]" = {"judging": 0, "before": 0.0}


def _switching_often() -> None:
    """Shorten the switch interval for a judgement that starts during a watched line (:data:`_WATCHED_SWITCH_S`)."""
    with _SWITCH_LOCK:
        if _SWITCH_STATE["judging"] == 0:
            _SWITCH_STATE["before"] = sys.getswitchinterval()
            sys.setswitchinterval(min(_SWITCH_STATE["before"], _WATCHED_SWITCH_S))
        _SWITCH_STATE["judging"] += 1


def _switching_as_before() -> None:
    """Put the switch interval back once the last judgement during a watched line ended."""
    with _SWITCH_LOCK:
        _SWITCH_STATE["judging"] = max(0, int(_SWITCH_STATE["judging"]) - 1)
        if _SWITCH_STATE["judging"] == 0 and _SWITCH_STATE["before"] > 0.0:
            sys.setswitchinterval(_SWITCH_STATE["before"])


def _planner_refusal_status(refusal: object | None) -> MotionStatus:
    """The status a cuRobo refusal reads as, by what the sidecar found.

    A joint outside the planner's limits is JOINT_LIMIT_REJECTED. The robot touching itself and the
    robot reaching into the planner's world are both SELF_COLLISION_REJECTED, the status whose
    meaning covers the declared fixtures, because :class:`MotionStatus` has no status for the world.
    A refusal the sidecar did not type keeps SELF_COLLISION_REJECTED, what every refusal read as
    before the kind was read.
    """
    if refusal is not None and getattr(refusal, "kind", None) == StateRefusalKind.JOINT_LIMIT:
        return MotionStatus.JOINT_LIMIT_REJECTED
    return MotionStatus.SELF_COLLISION_REJECTED


def _with_refusal(sentence: str, refusal: object | None) -> str:
    """``sentence``, followed by the sidecar's own account of the refusal where it gave one."""
    render = getattr(refusal, "render", None)
    return f"{sentence}. {render()}" if callable(render) else sentence


def _branch_text(branch: "URBranch | None") -> str:
    """A branch as a log line reads it, or that it has none, for a model with no DH chain."""
    return f"branch {branch.render()}" if branch is not None else "branch unknown"


def _same_branch(held: "URBranch | None", other: "URBranch | None") -> bool:
    """Whether ``other`` holds the arm as ``held`` does; an unknown branch, on a model with no DH chain, agrees."""
    return held is None or other is None or held.agrees_with(other)


def _on_the_flange(refused: MotionResult) -> bool:
    """Whether the exact guard refused a line at a part on the flange against a fixture (:data:`_ON_THE_FLANGE`, a wrist
    camera): on a straight line of the TCP such a part stands where the TCP puts it, whichever configuration the line
    starts from, so the refusal is every configuration's."""
    found = _FIXTURE_PAIR.search(str(refused.message or ""))
    part = found.group("part") if found is not None else ""
    return part in _ON_THE_FLANGE or part.startswith(_WRIST_CAMERA_PART)


def _pose_key(pose: Pose) -> "tuple[object, ...]":
    """A pose as the bytes of its position and its orientation, and its frame: two poses with the same key are the same
    goal to the last bit, whatever they are labelled."""
    return (pose.frame, np.asarray(pose.position_mm, dtype=np.float64).tobytes(),
            np.asarray(pose.quaternion_xyzw, dtype=np.float64).tobytes())


def _holds_its_world_at(arm: object, junction: str) -> bool:
    """``safety.planning_world.hold.<junction>`` (``standoff``, ``carry``, ``drop``) as ``arm`` was built with it, read as
    ``is True``; ``False`` for any other junction and for an arm that carries no such switch."""
    return junction in ("standoff", "carry", "drop") and getattr(getattr(arm, "_holds", None), junction, False) is True


def _joints_key(joints: "JointPositions | Sequence[float]") -> "tuple[object, ...]":
    """Joints as the bytes of their values: two joint targets with the same key are the same target to the last bit."""
    values = joints.tolist() if isinstance(joints, JointPositions) else joints
    return (np.asarray([float(v) for v in values], dtype=np.float64).tobytes(),)


def _off_the_line_mm(model: str, start: "Sequence[float]", end: "Sequence[float]", fractions: "Sequence[float]", *,
                     reach_mm: float) -> float:
    """How far the joint line from ``start`` to ``end`` puts the flange and what it carries off the straight line between
    where the two put the flange, at the worst of ``fractions`` of the way along it, millimetres.

    At each fraction: the flange's distance from the point that fraction of the way along the chord between the two
    flanges, plus how far it turned from the orientation that fraction of the way between theirs times ``reach_mm``, the
    furthest anything on the flange may stand from it. Read on the DH chain, as :meth:`URRobotArm._line_step_refusal`
    reads a step; ``inf`` for a model with no chain or an orientation that cannot be read, which no bound admits.
    """
    a = np.asarray([float(v) for v in start], dtype=np.float64)
    b = np.asarray([float(v) for v in end], dtype=np.float64)
    start_chain, end_chain = ur_link_transforms_mm(model, a), ur_link_transforms_mm(model, b)
    if start_chain is None or end_chain is None:
        return math.inf
    first, last = (np.asarray(chain[-1], dtype=np.float64) for chain in (start_chain, end_chain))
    worst = 0.0
    try:
        turn_a, turn_b = from_rotation_matrix(first[:3, :3]), from_rotation_matrix(last[:3, :3])
        for fraction in fractions:
            t = float(fraction)
            chain = ur_link_transforms_mm(model, a + (b - a) * t)
            if chain is None:
                return math.inf
            flange = np.asarray(chain[-1], dtype=np.float64)
            off_mm = float(np.linalg.norm(flange[:3, 3] - (first[:3, 3] + (last[:3, 3] - first[:3, 3]) * t)))
            turned = angle_between(slerp(turn_a, turn_b, t), from_rotation_matrix(flange[:3, :3]))
            worst = max(worst, off_mm + turned * float(reach_mm))
    except (GeometryError, ValueError):
        return math.inf
    return worst


@dataclasses.dataclass(frozen=True, slots=True)
class _Route:
    """The judged joint waypoints one move runs, and how they were chosen.

    ``waypoints`` starts where the arm stands (a plan's own start, which is where the planner read the arm) and ends
    on the goal; every leg between two neighbours was judged by both authorities. It is the very object they judged,
    never a copy, and it is run as it is. ``dense`` is how many waypoints cuRobo returned, two for a straight line.
    ``how`` is the route in words for the log, ``planned`` whether cuRobo planned it, which decides how a joint move
    runs it: a straight line is one ``moveJ`` to the goal, a plan one ``moveJ`` per waypoint. ``branch_change`` is
    whether it ends on another branch than the arm holds, ``overshoot_deg`` how far its worst joint swings past the span
    between its ends on the waypoints that run, nothing for a straight line (the owner's R2, 2026-10-08).
    """

    waypoints: "Sequence[Sequence[float]]"
    dense: int
    how: str
    planned: bool
    branch_change: bool = False
    overshoot_deg: float = 0.0


@dataclasses.dataclass(slots=True)
class _KeptBranch:
    """What a :meth:`URRobotArm.keeping_its_branch` block narrows, and what it refused for it.

    ``overshoot_deg`` is how far a cuRobo plan may swing a joint past its span inside the block, where that is tighter
    than ``safety.planned_motion.max_detour_deg``. ``refused`` counts the moves the block refused although the rules
    outside it would have gone on: a configuration on another branch was left untried, or a plan within the configured
    bound swung past this one. A pick reads it after each grasp it tried inside the block.
    """

    overshoot_deg: float
    refused: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class _JudgedAhead:
    """The route to a standoff a grasp judged ahead ended on (:meth:`URRobotArm.grasp_refusal_ahead`), kept for the one
    move to that standoff, which runs it as it was judged where nothing it was judged on changed since
    (:meth:`URRobotArm._drive_curobo`).

    ``route`` is the very object the exact guard judged sample by sample. ``refresh`` built the world it was judged in and
    ``planner`` holds that world. ``goal`` is the TCP pose it was judged to (:func:`_pose_key`), ``velocity`` the speed its
    goals were ranked at, ``at`` when the judgement ended (``time.monotonic``). ``state`` is what else a judgement reads
    (:meth:`URRobotArm._judgement_state`), and ``boxes`` the boxes the camera saw that the exact guard held, compared by
    identity: every refresh hands the guard a tuple of its own.
    """

    route: _Route
    refresh: "WorldRefresh"
    planner: object
    goal: "tuple[object, ...]"
    velocity: float
    at: float
    state: "tuple[object, ...]"
    boxes: "tuple[object, ...]"


@dataclasses.dataclass(frozen=True, slots=True)
class _LiftJudged:
    """A lift judged at the part from where the arm stands, as if the jaws held the part, inside a held world, on a cell
    that models no carried part (:meth:`URRobotArm.carried_line_refusal`), kept for the line up that runs it
    (:meth:`URRobotArm._drive_checked_line`).

    ``pose`` is the lift's top (:func:`_pose_key`), ``reason`` the held world's reason, ``refresh`` the refresh that built
    the held world and ``planner`` the glue holding it; ``joints`` where the arm stood, ``state`` and ``boxes`` as
    :class:`_JudgedAhead` keeps them, read while the judgement ran, the hand left unread.

    ``stands_in`` is whether the judgement's part stood in for the one the close will carry, as a lift's does; a line
    judged as it will run, the line out of a place judged while the jaws open (:meth:`URRobotArm.judge_line_ahead`), is
    not, and runs on a cell that models a carried part too.
    """

    pose: "tuple[object, ...]"
    reason: str
    refresh: "WorldRefresh"
    planner: object
    joints: "tuple[float, ...]"
    state: "tuple[object, ...]"
    boxes: "tuple[object, ...]"
    stands_in: bool = True


@dataclasses.dataclass(frozen=True, slots=True)
class _NextLegJudged:
    """The joint move a task declared after the verb running (:meth:`URRobotArm.expecting_next`), judged ahead from where
    the line before it will leave the arm, in the world a junction holds (``safety.planning_world.hold``), kept for the
    one joint move to it (:meth:`URRobotArm.move_to_joints`, :meth:`URRobotArm.move_to_home`).

    ``target`` is the declared joints as bytes (:func:`_joints_key`), ``joints`` the configuration judged (its twin inside
    the goal window), ``route`` the straight joint line judged, from ``start``, the joints the line before it was predicted
    to end at, to ``joints``. ``reason`` is the held world it was judged in, ``refresh`` the refresh that built it and
    ``planner`` the glue holding it; ``state`` and ``boxes`` as :class:`_JudgedAhead` keeps them, read where the arm stood
    still, and ``at`` when the judgement ended (``time.monotonic``).
    """

    target: "tuple[object, ...]"
    joints: JointPositions
    route: _Route
    start: "tuple[float, ...]"
    reason: str
    refresh: "WorldRefresh"
    planner: object
    state: "tuple[object, ...]"
    boxes: "tuple[object, ...]"
    at: float


@dataclasses.dataclass(frozen=True, slots=True)
class _Standing:
    """Why the planner's refusal of judged samples stands where the exact guard does not decide it (the owner's F1 for
    the robot's own pairs, and Option 1 for the boxes the camera saw).

    ``row`` is the refused sample it stands on, where the planner's report was read whole and a row of it is the
    planner's alone (``band.admission_refusal``, ``band.world_admission_refusal``), the camera's boxes cannot be set
    aside for it (a part carried, a hand not known empty and open, :meth:`URRobotArm._world_aside_refusal`), or the
    exact guard refuses it; its ``index`` counts over the samples judged. ``None`` where the refusal
    was never the guard's to decide, or no whole report came back: the refusal is then said as the planner's verdict
    said it. ``status`` is the exact guard's own, where it refused the row.
    """

    reason: str
    row: "RefusedSample | None" = None
    status: "MotionStatus | None" = None

    def said(self, sentence: str, verdict: object, count: int) -> "tuple[MotionStatus, str]":
        """The status and the message of a refusal ``sentence`` opens, on ``count`` samples the ``verdict`` judged.

        On a row: that sample, what the planner found there and why it stands, and a bound it stands on reads as a
        bound, the exact guard's refusal as the guard's own status. Else the verdict's own words and status, as every
        planner refusal read before the exact guard decided any of it (the guard's status where it refused).
        """
        refusal = getattr(verdict, "refusal", None)
        row = self.row
        if row is None:
            return (self.status or _planner_refusal_status(refusal),
                    _with_refusal(f"{sentence}: {getattr(verdict, 'reason', '')}", refusal))
        status = self.status or (MotionStatus.JOINT_LIMIT_REJECTED if not row.bound_ok
                                 else MotionStatus.SELF_COLLISION_REJECTED)
        return status, f"{sentence} at sample {row.index} of {count}: {row.findings()}; it stands: {self.reason}"


@dataclasses.dataclass(frozen=True, slots=True)
class _BandRoute:
    """What a plan out of or into the planner's cushion band came to (:meth:`URRobotArm._band_route`).

    ``route`` is ``[here, <cuRobo's plan>, goal]``, ``here`` and ``goal`` only where a straight band leg leaves or
    reaches them, or ``None`` where no route is taken. ``note`` says what was found and done, or why no route is taken;
    empty where the planner's refusal was not the band's to answer. ``stuck`` is true where it is the start that no plan
    leaves (refused for more than the band, or no leg out of it), which no other goal plans from either; a failure after
    the start was left belongs to this goal alone. ``legs`` counts the band legs at the head and at the tail of
    ``route``, which the shortening keeps as they were judged.
    """

    route: "list[list[float]] | None" = None
    note: str = ""
    stuck: bool = True
    legs: "tuple[int, int]" = (0, 0)


@dataclasses.dataclass(frozen=True, slots=True)
class _RankedGoals:
    """The configurations of one flange goal in the order the arm takes them, the branch it holds first.

    ``groups`` is the goals on the branch the arm holds (``held``), then the others, each group in the order
    :func:`~src.robot.safety._ur_ik.nearest_goals` gave it, the least time a ``moveJ`` there takes first. ``branches``
    is every goal's branch and ``names`` how a log line names it, ranked across both groups. A move screens, lines and
    plans the first group before the second (:meth:`URRobotArm._route_to_the_nearest_goal`); a caller that only asks
    where a pose goes takes the first the endpoint gate admits, in :attr:`in_order`
    (:meth:`URRobotArm.nearest_configuration`).
    """

    held: "URBranch | None"
    groups: "tuple[list[NearestGoal], list[NearestGoal]]"
    branches: "dict[NearestGoal, URBranch | None]"
    names: "dict[NearestGoal, str]"

    @property
    def in_order(self) -> "list[NearestGoal]":
        """Every goal, those on the arm's branch first."""
        return self.groups[0] + self.groups[1]


class URRobotArm(RobotArm):
    """The UR-backed arm driver.

    It satisfies the vendor-neutral :class:`RobotArm` Protocol and keeps the UR
    convenience API the current pipelines use.
    """

    def __init__(
        self,
        config: RobotConfig,
        home_joints: list[float] | None = None,
        *,
        curobo_client_factory: Callable[[], CuroboPlanClient] | None = None,
    ) -> None:
        self.config = config
        self.logger = create_robot_logger("URRobotArm", UR_ARM_LOG_FILE)

        # The halt latch lives on this connection, which the arm keeps for its life, so a disconnect and a connect of
        # the same arm keep it; brake_on_halt decides whether a halt also brakes the move in flight.
        self._conn = URConnection(
            ip=config.ur.ip,
            vel=config.motion_limits.max_velocity,
            acc=config.motion_limits.max_acceleration,
            frequency=config.ur.rtde_frequency,
            brake_on_halt=config.ur.brake_on_halt,
            # How "the arm stands still" is read: the controller's is_steady() or the joint speeds (2026-10-09).
            steady_signal=config.safety.dwell.steady_signal,
        )
        self._guard = WorkspaceGuard(config.workspace_limits)
        self._motion = MotionController(
            self._conn,
            self._guard,
            max_velocity=config.motion_limits.max_velocity,
            max_acceleration=config.motion_limits.max_acceleration,
            # The line's singularity check on the controller's FK or on the arm's own chain (2026-10-09).
            singularity_fk=config.safety.ik_quality.singularity_fk,
            arm_model=config.ur.model,
        )
        # The vendor-neutral safety preflight pipeline. It owns its own WorkspaceGuard,
        # shrunk by the configured ``limits.workspace_margin_mm``, so the typed
        # ``move()`` path enforces the tighter operator-defined margin while
        # ``MotionController`` keeps the full-box guard as a backstop for the bool path.
        self._preflight = SafetyPreflight.from_safety_config(
            config.safety, config.workspace_limits,
            # The hand the cell names and the arm the guard places it on. An exact mesh guard with no
            # hand named refuses here, at build, rather than checking whatever hand the arm bundle carries.
            hand=planner_hand(config), arm_model=config.ur.model,
        )
        # An explicit argument wins over the cell own config, which wins over the
        # UR5e-authored constant. A real cell that declares nothing still gets a home
        # pose, and move_home warns about it.
        self._home_joints = (
            list(home_joints) if home_joints
            else list(config.home_joint_positions or HOME_JOINTS_DEFAULT)
        )
        self._capabilities = ur_capabilities(config.ur.model)
        # The real-UR cuRobo binding. "ik", the default, uses the controller IK path.
        # "curobo" plans a collision-free trajectory through safety.planning and executes
        # it over ur_rtde, fail-closed.
        self._motion_planner = config.ur.motion_planner
        #: The part the gripper carries while one is attached, as (length_mm, lateral_margin_mm) from
        #: planning_world.payload. The self filter grows the hand by it; None when nothing is attached.
        self._attached_payload: tuple[float, float] | None = None
        #: Whether the planner took the part the last attach handed it.
        self._payload_in_planner = False
        #: Whether the jaws closed on a part since the last detach, whatever models it: set by every attach, also one a
        #: cell declines to model, and cleared by :meth:`detach_payload` alone. While it is set no box the camera saw is
        #: set aside in the planner's world (:meth:`_world_aside_refusal`).
        self._closed_on_part = False
        #: Why every line is judged against the world the last refresh built, with no new frame (:meth:`held_world`);
        #: ``None`` outside such a block.
        self._held_world_reason: str | None = None
        #: The route a grasp judged ahead ended on, for the one move to its standoff (:meth:`grasp_refusal_ahead`,
        #: :meth:`_drive_curobo`); ``None`` once any motion verb, refresh or held world came after it.
        self._judged_ahead: _JudgedAhead | None = None
        #: The lift judged at the part as if the jaws held the part, for the one line up that runs it
        #: (:meth:`carried_line_refusal`, :meth:`_drive_checked_line`); ``None`` once any other motion came after it.
        self._lift_judged: _LiftJudged | None = None
        #: The standoff whose route a grasp judged ahead kept the arm's branch (:func:`_pose_key`), for the move to it,
        #: which judged again keeps it too (:meth:`_drive_curobo`); ``None`` once any motion verb, refresh or held world
        #: came after it.
        self._ahead_kept_its_branch: "tuple[object, ...] | None" = None
        #: What the :meth:`keeping_its_branch` block running now narrows and counts; ``None`` outside one.
        self._keeping: _KeptBranch | None = None
        #: The refresh that vouches for the move running now where it made none itself: the one a route judged ahead was
        #: judged in (:meth:`_drive_curobo`); :meth:`move` stamps from it and forgets it.
        self._vouched_by: "WorldRefresh | None" = None
        dwell = getattr(config.safety, "dwell", None)
        #: Where the steady gate waits (``safety.dwell.gate_at``): true where this arm judges a motion first and waits for
        #: steady right before it sends it (:meth:`_send_gate`), and the verbs leave the gate to it
        #: (:attr:`gates_its_own_sends`). Only a cuRobo arm: every motion of it goes through the sends that gate.
        self._gates_sends = (self._motion_planner == "curobo" and dwell is not None
                             and bool(getattr(dwell, "require_steady_before_motion", False))
                             and str(getattr(dwell, "gate_at", "verb")) == "send")
        #: When the next leg of a pick or a place is judged (``robot.motion.judge_next_leg``), and where a world is held
        #: across a junction (``safety.planning_world.hold``).
        self._next_leg_mode = str(getattr(getattr(config, "motion", None), "judge_next_leg", "off"))
        self._holds = getattr(getattr(config.safety, "planning_world", None), "hold", None)
        #: The joint move a task declared after the verb running (:meth:`expecting_next`): its key and its joints.
        self._expected_next: "tuple[tuple[object, ...], JointPositions] | None" = None
        #: That move judged ahead (:meth:`judge_the_next_leg`), for the one joint move to it (:meth:`move_to_joints`,
        #: :meth:`move_to_home`); it outlives the lines before that move, which the start it was judged from names.
        self._next_leg_ahead: _NextLegJudged | None = None
        #: The line during whose send that move is judged on a second thread: the line's pose (:func:`_pose_key`) and the
        #: junction whose hold it is judged in (:meth:`judge_the_next_leg` with ``now`` false).
        self._next_leg_during: "tuple[tuple[object, ...], str] | None" = None
        #: Guards :attr:`_next_leg_ahead` between the thread that judges during a line and the one that sends it, and the
        #: count a judgement still running when its line ended voids itself by.
        self._next_leg_lock = threading.Lock()
        self._next_leg_epoch = 0
        #: The route judged ahead whose world :meth:`holding_the_approach` holds for the move to its standoff; ``None``
        #: outside that block.
        self._approach_held_for: _JudgedAhead | None = None
        if self._gates_sends:
            self.logger.info("safety.dwell.gate_at is 'send': a motion is judged first, while the arm settles, and the "
                             "steady gate waits right before it is sent")
        if self._next_leg_mode == "in_settles_and_motion" and not bool(config.ur.brake_on_halt):
            self.logger.warning("robot.motion.judge_next_leg is 'in_settles_and_motion' and robot.ur.brake_on_halt is "
                                "false: a leg is judged during the line before it only where a halt brakes that line, "
                                "so this arm judges ahead in the jaws' strokes alone ('in_settles')")
        #: The hand this arm carries, as the cell that brought both up handed it over (:meth:`set_hand`); ``None`` for
        #: none. The camera's boxes are set aside only while it reads empty and open (the owner, 2026-10-01).
        self._hand: object | None = None
        self._curobo_client_factory = curobo_client_factory
        self._curobo_ur: CuroboUrPlanner | None = None
        #: Guards building, starting and closing ``_curobo_ur``: a disconnect waits for a planner start in progress
        #: and closes what it started, rather than dropping a planner a start is about to hand a sidecar.
        self._planner_lock = threading.RLock()
        #: Whether :meth:`start_planner` runs now; read without the lock, for :attr:`planner_state`.
        self._planner_starting = False
        #: The flange to TCP the controller applied when this arm last connected, derived by the tool
        #: frame check on a ``polyscope`` cell; None before a connect and after a disconnect.
        self._observed_tool_frame: "np.ndarray | None" = None
        #: The cell as the cameras see it, asked immediately before every plan. Handed in by
        #: whatever composes the cell rather than built here: a driver may not reach up into
        #: perception, and this one only ever asks. `None` is the unchanged path, where the
        #: declared world is registered once and the planner keeps it for the life of the cell.
        self._live_world: "LivePlannerWorld | None" = None
        #: The hand-guiding session whose with-block is running, or None. Every motion verb refuses while
        #: one is (:meth:`_hand_guided_refusal`).
        self._freedrive: URFreedriveSession | None = None

    # ------------------------------------------------------------------
    # Tool frame (flange <-> TCP)
    # ------------------------------------------------------------------
    #
    # Every Pose crossing the RobotArm Protocol is the TCP, meaning the grasp centre.
    # What the controller calls the TCP depends on its tool register, and
    # ``gripper.tool_frame.source`` declares who owns that fact:
    #   "willy"     the controller runs a bare flange and this driver composes the
    #               transform, as the Isaac driver does. No vendor SDK call is involved.
    #   "polyscope" the controller holds it, and poses pass through untouched.
    # Either way connect() derives what the controller is really using and refuses a
    # mismatch.

    @property
    def safety_preflight(self) -> "SafetyPreflight | None":
        """The guard pipeline that judges every motion commanded through this arm's own verbs.

        What it judges differs by command, and the difference is what a caller needs to
        know. A Cartesian command is judged at its target pose; on a cuRobo path the
        planned final configuration is judged without IK quality and motion continuity.
        A joint command is judged at its target configuration by the destination guards
        only (joint limits, self-collision, payload). The middle of a planned path is
        judged sample by sample, always, whenever a planner produced one, which is what
        :attr:`plans_paths` answers. The raw transports :attr:`connection` and
        :attr:`motion` reach the controller below it.

        It implements :class:`~src.robot.safety.attestation.SafetyGated`, so a caller
        asks what this arm will refuse without reaching into `_preflight`. That matters
        for the library API: a caller supplies their own arm, which enumerating the
        driver registry cannot see, so the answer travels with the object.
        """
        return self._preflight

    @property
    def home_joint_positions(self) -> tuple[float, ...]:
        """The joints :meth:`move_home` drives to, in radians: the constructor's, else the config's, else the default.

        The joint-limit guard reads it where ``safety.joint_limits.within_half_turn_of_home`` is set,
        and centres its window on it, so the window is about the home this arm actually goes to.
        """
        return tuple(float(v) for v in self._home_joints)

    @property
    def plans_paths(self) -> bool:
        """Whether a move on this arm produces a path, and so has a middle to judge.

        Read off the arm as it is now, never off a key somebody could set: a cuRobo move
        hands over a trajectory and every configuration of it is judged, while an ik move
        interpolates and has no middle anybody could look at. The attestation prints one of
        the two, so a cell cannot claim a path check it will never run, or hide one it
        always does.
        """
        return self._motion_planner == "curobo"

    def _declared_tool_matrix(self) -> "np.ndarray | None":
        """The declared flange-to-TCP 4x4, whoever applies it. ``None`` while it is undeclared.

        Two consumers want different things and the difference is not cosmetic:
        :meth:`_pose_to_controller` needs it in ``willy`` mode alone, because in
        ``polyscope`` the controller applies it, while the cuRobo path needs it always,
        which :meth:`_pose_to_flange` sets out.
        """
        tf = self.config.gripper.tool_frame
        if tf.source == "undeclared":
            return None
        from .tool_frame import tool_frame_matrix

        return tool_frame_matrix(tf.offset_mm, tf.rotation_quat_xyzw)

    def _pose_to_flange(self, pose: Pose) -> Pose:
        """TCP to flange, meaning tool0, using the declared transform whatever ``source`` says.

        cuRobo is why this is unconditional. Its kinematic model ends at ``tool0`` by
        construction: ``scripts/curobo/build_ur_config.py`` attaches the 2F-85 collision
        spheres to ``tool0`` and the planner plans from tool0 to the pose. So it never
        consults the controller tool register, and ``polyscope`` mode does not reach it.
        Handing cuRobo the grasp centre as a tool0 goal is wrong twice over: it drives
        the flange to the grasp point, which is the 132 mm error, and it hangs the
        gripper collision spheres a whole tool length past the target, so the
        collision-free plan is computed for a gripper that is not where the planner
        thinks it is.
        """
        t = self._declared_tool_matrix()
        if t is None:
            return pose
        from src.geometry.matrix import invert_homogeneous

        m = np.asarray(pose.to_matrix(), dtype=np.float64) @ invert_homogeneous(t)
        return Pose.from_matrix(m, frame=pose.frame, label=pose.label)

    def _pose_from_controller(self, urpose: URPose, *, label: str) -> Pose:
        """Controller frame -> TCP. Identity unless this driver owns the tool frame."""
        pose = urpose_to_pose(urpose, frame=Frame.BASE, label=label)
        t = self._declared_tool_matrix() if self.config.gripper.tool_frame.source == "willy" else None
        if t is None:
            return pose
        m = np.asarray(pose.to_matrix(), dtype=np.float64) @ t
        return Pose.from_matrix(m, frame=Frame.BASE, label=label)

    def _pose_to_controller(self, pose: Pose) -> URPose:
        """TCP -> the frame the controller expects. Identity unless this driver owns the tool frame."""
        if self.config.gripper.tool_frame.source != "willy":
            return pose_to_urpose(pose)   # polyscope applies it controller-side; undeclared never moves
        return pose_to_urpose(self._pose_to_flange(pose))

    #: The size classes this driver compares across the two spellings a controller and a
    #: config use. It is the digits alone on purpose: the URSim e-Series container
    #: reports ``UR3`` for a UR3e, so a string comparison against ``ur3e`` would refuse a
    #: correctly configured cell.
    _MODEL_SIZES = ("3", "5", "10", "16", "20", "30")

    @classmethod
    def _model_size(cls, name: str | None) -> str | None:
        """Turn ``'UR3'``, ``'ur3e'`` or ``'UR5e CB3'`` into the size digits, or ``None``.

        It takes the number that follows the ``UR`` and not every digit in the string.
        Concatenating all of them looks equivalent and is not: ``'UR5e CB3'`` yields
        ``'53'``, which matches no size and disables the check silently. Any string
        without a readable ``UR<n>`` returns ``None``, which the caller treats as no
        evidence.
        """
        if not name:
            return None
        match = re.search(r"ur\s*(\d+)", str(name), flags=re.IGNORECASE)
        if match is None:
            return None
        size = match.group(1)
        return size if size in cls._MODEL_SIZES else None

    def _verify_controller_model(self) -> None:
        """Refuse where the controller says it is a different size of robot from ``ur.model``.

        ``ur.model`` keys the safety DH chain, the exact-mesh collision bundle and the
        cuRobo robot config. A UR3e planned against UR5e link lengths, a2 and a3 at
        -243.55 and -213.2 mm against -425 and -392.2, is precisely the failure that
        field exists to prevent, and without this comparison the mismatch surfaces only
        downstream, as a tool-frame refusal quoting a distance nobody can act on.

        It fails closed on evidence and open on its absence. A dashboard that will not
        answer, or answers something this driver cannot parse, yields ``None`` and is
        logged rather than refused, because a false refusal at connect is not a safe
        failure but a cell that will not start, and the tool-frame check downstream
        still catches the dangerous case on its own. Only a parsed, positive
        disagreement refuses.

        The comparison is on the size class alone. The URSim e-Series container reports
        ``UR3`` for a UR3e, so comparing whole strings would refuse a correct cell.
        """
        declared = self._model_size(self._capabilities.model)
        reported_raw = self._conn.controller_model()
        reported = self._model_size(reported_raw)
        if declared is None or reported is None:
            self.logger.info(
                "controller model not compared: config says %r, controller says %r; one of the two "
                "is not a size this driver recognises, so there is no evidence of a mismatch",
                self._capabilities.model, reported_raw,
            )
            return
        if declared == reported:
            # Say what was not checked. A size class is not a model: a UR3 and a UR3e are both
            # "UR3" to this comparison, on purpose, because the controller reports the same
            # string for both. Logging a bare "verified" for a pair this check cannot separate
            # reads as an all-clear it did not earn, and the separation is done further down by
            # _verify_tool_frame, relatively rather than by a threshold.
            twin = ur_series_twin(self._capabilities.model)
            if twin is None:
                self.logger.info(
                    "controller model verified: config %r matches the controller's %r",
                    self._capabilities.model, reported_raw,
                )
            else:
                self.logger.info(
                    "controller SIZE verified: config %r and the controller's %r are both a "
                    "UR%s. The series was NOT checked here and cannot be from this string, "
                    "because a %s controller reports %r too. The tool-frame check next tells "
                    "the two apart by asking which of them explains the controller better.",
                    self._capabilities.model, reported_raw, declared, twin, reported_raw,
                )
            return
        raise RobotConnectionError(
            f"ur.model is {self._capabilities.model!r}, but this controller reports "
            f"{reported_raw!r}: a UR{reported} is not a UR{declared}. That field keys the safety DH "
            f"chain, the collision mesh bundle and the cuRobo robot config, so every pose this stack "
            f"plans would be computed against another robot's link lengths. Point ur.model at the arm "
            f"that is actually plugged in."
        )

    def _verify_tool_frame(self) -> None:
        """Refuse to stay connected where the controller actual tool frame is not the assumed one.

        It is derived and never read:
        ``inv(base to flange from the bundled DH table) @ getForwardKinematics(q)``.
        Both halves are already on the safety-critical path, so this needs no SDK symbol
        ``move()`` does not already require. The two modes predict different answers,
        near identity for "willy", where the controller must be bare, and the declared
        transform for "polyscope", so a pass also proves which mode the cell is in
        rather than merely that some number matched.
        """
        from .tool_frame import compare_tool_frames, derive_active_tool_frame, tool_frame_matrix

        tf = self.config.gripper.tool_frame
        declared = tool_frame_matrix(tf.offset_mm, tf.rotation_quat_xyzw)
        # "willy" composes on this side, so the controller itself carries no tool.
        expected = np.eye(4, dtype=np.float64) if tf.source == "willy" else declared

        # The no-argument FK, and why. `getForwardKinematics(q)` is corrupted by any
        # preceding moveJ or moveL, because it shares the controller float registers with
        # them. Reproduced end to end: the first connect verifies at 0.00 mm, the cell
        # moves once, and the next connect is refused with a delta over a metre, turning
        # a perfectly good arm away on the second run through the check that exists to
        # catch a wrong TCP. The no-argument form is immune, at 0.000 mm in every state
        # tried, and it is also exactly what this check wants: the frame the controller
        # is applying right now.
        #
        # Steadiness is then required rather than assumed, because the pose and the
        # joints are read at two instants. A moving arm would pair a pose with joints it
        # no longer has and refuse a good cell, and a false refusal at connect is not a
        # safe failure but a cell that will not start.
        if not self._conn.is_steady():
            raise RobotConnectionError(
                "cannot verify the tool frame while the arm is moving: the check pairs the joint "
                "vector with the controller's current pose, and those must describe the same "
                "instant. Let the arm come to rest and connect again."
            )
        joints = list(self._conn.get_joint_positions())
        controller_tcp = URPose.from_ur_list(self._conn.fk_current()).to_T()
        observed = derive_active_tool_frame(self._capabilities.model, joints, controller_tcp)
        if observed is None:
            raise RobotConnectionError(
                f"cannot verify the tool frame: no bundled DH table for ur.model "
                f"{self._capabilities.model!r}, so the controller's active tool frame is not "
                f"derivable. Set a supported model, or gripper.tool_frame.source: undeclared while "
                f"this cell is being built (which refuses to connect for a different, louder reason)."
            )
        d_t, d_r = compare_tool_frames(observed, expected)
        self._refuse_if_the_series_twin_fits_better(joints, controller_tcp, expected, d_t)
        self._observed_tool_frame = None
        if d_t > tf.verify_tolerance_mm or d_r > 5.0:
            # No disconnect here, because connect() rolls the connection back for any
            # failure of this check.
            other = "polyscope" if tf.source == "willy" else "willy"
            raise RobotConnectionError(
                f"gripper.tool_frame.source is {tf.source!r}, which expects the controller's tool "
                f"register to hold {'nothing (a bare flange)' if tf.source == 'willy' else 'the declared transform'}"
                f", but the controller is actually running a tool frame {d_t:.1f} mm / {d_r:.1f} deg "
                f"away from that (tolerance {tf.verify_tolerance_mm:.1f} mm). Commanding motion now "
                f"would drive the arm off by that much on every pose. Either the pendant's TCP is not "
                f"what this config says, or the cell is really in {other!r} mode, or ur.model "
                f"({self._capabilities.model!r}) is not the arm on the other end. This is derived "
                f"from THAT model's DH table, so a wrong model produces a wrong distance here. "
                f"Derived as inv(DH flange) @ getForwardKinematics; no PolyScope read required."
            )
        # A wrist camera placed from a recorded frame is stale once the controller applies another
        # one. Checked before the frame is kept, so a refused connect keeps nothing.
        if tf.source == "polyscope":
            for body in self._preflight.wrist_bodies(self):
                stale = body.record_refusal("polyscope", observed)  # type: ignore[attr-defined]
                if stale is not None:
                    raise RobotConnectionError(stale)
        self._observed_tool_frame = np.asarray(observed, dtype=np.float64).copy()
        self.logger.info(
            "tool frame verified: source=%s, controller matches within %.2f mm / %.2f deg",
            tf.source, d_t, d_r,
        )

    @property
    def active_tool_frame(self) -> "np.ndarray | None":
        """The flange to TCP :meth:`get_tcp_pose` applies now, as a 4x4 in millimetres, or None where
        it is not known.

        On a ``willy`` cell the driver composes the declared frame itself, so it is the declared matrix,
        connected or not. On a ``polyscope`` cell the controller applies its own tool setting, so it is
        the frame the tool frame check derived from the controller at the last connect, and None before
        a connect and after a disconnect. An ``undeclared`` cell has none. An eye in hand calibration
        records this beside its solve.
        """
        source = self.config.gripper.tool_frame.source
        if source == "willy":
            return self._declared_tool_matrix()
        if source == "polyscope" and self._observed_tool_frame is not None:
            return self._observed_tool_frame.copy()
        return None

    #: How much better the twin has to explain the controller before this refuses, in mm. Small
    #: on purpose: the two hypotheses are not close. A correct cell reads about 0 mm for its own
    #: model and 8.5 mm or more for the twin (the measured ur3/ur3e minimum over 20000 poses),
    #: so anything above sensor noise separates them. This is NOT a tolerance on being right; it
    #: is the margin by which the wrong answer has to win before the cell is refused.
    _TWIN_MARGIN_MM = 1.0

    def _refuse_if_the_series_twin_fits_better(
        self, joints: list[float], controller_tcp, expected, own_d_t: float
    ) -> None:
        """Refuse where the other series of this size explains the controller better than this one does.

        The check the tolerance cannot do. ``_verify_controller_model`` compares size classes,
        because a controller reports ``UR3`` for a UR3e. ``_verify_tool_frame`` compares a
        distance against a tolerance, and over 20000 random joint vectors a ur3/ur3e swap lands
        under the default 10 mm tolerance in 12.2 % of poses, with a minimum separation of
        8.50 mm. So the two arms this cell is most likely to confuse are the two arms neither
        check can separate.

        This one is relative and therefore immune to the tolerance: the flange -> TCP transform
        is one physical thing, and deriving it through the wrong DH table moves it. Whichever
        model puts it closer to what this cell expects is the arm on the other end. A correct
        cell reads about 0 mm for itself and 8.5 mm or more for the twin; a swapped one reverses
        that, and the margin between the two is the evidence rather than either distance.

        Silent when the model has no twin (ur16e), when the twin has no DH table, or when the
        two hypotheses are within ``_TWIN_MARGIN_MM`` of each other, which is the honest reading
        of two explanations that fit equally well.
        """
        from .tool_frame import compare_tool_frames, derive_active_tool_frame

        twin = ur_series_twin(self._capabilities.model)
        if twin is None:
            return
        twin_observed = derive_active_tool_frame(twin, joints, controller_tcp)
        if twin_observed is None:
            return
        twin_d_t, _ = compare_tool_frames(twin_observed, expected)
        if twin_d_t + self._TWIN_MARGIN_MM >= own_d_t:
            return
        raise RobotConnectionError(
            f"ur.model is {self._capabilities.model!r}, and this controller is better "
            f"explained by a {twin!r}: deriving the active tool frame through the "
            f"{self._capabilities.model} DH table lands {own_d_t:.1f} mm from what this cell "
            f"expects, and through the {twin} table only {twin_d_t:.1f} mm. Those two arms "
            f"report the SAME model string to a controller dashboard, so nothing upstream can "
            f"tell them apart, and their flanges can sit as little as 8.5 mm apart, which fits "
            f"inside the tool-frame tolerance. ur.model keys the safety DH chain, the collision "
            f"mesh bundle and the cuRobo robot config, so a cell running the wrong one of this "
            f"pair plans every pose against another robot. Point ur.model at the arm that is "
            f"actually plugged in, or, if it really is a {self._capabilities.model}, check the "
            f"pendant TCP first, because a wrong tool frame can also produce this."
        )

    def connect(self) -> None:
        """Open the RTDE connection to the robot controller.

        Once the connection is up, the configured payload envelope is pushed to the
        controller through ``setPayload``, so the protective-stop calculations of the
        controller reflect the same mass and centre of gravity this stack enforces. The
        push is gated on ``config.safety.payload.enforce``: with the payload guard
        disabled, the driver does not touch the controller payload either.

        It fails closed on a contradictory payload config before opening the socket, on
        two counts. ``enforce: true`` with ``mass_kg: 0.0``, which is the shipped
        default, would push ``setPayload(0.0)`` and overwrite the controller payload for
        a mounted tool silently. ``mass_kg > 0`` with the default ``cog_mm: [0, 0, 0]``
        would declare that tool a point mass at the flange face. Both fail open in the
        dangerous direction, and both corrupt the protective-stop model of the
        controller and its gravity compensation, which is what ``get_tcp_wrench`` reads.
        A bare flange is expressed by ``enforce: false``, which never touches the
        controller payload.
        """
        payload = self.config.safety.payload
        if payload.enforce and payload.mass_kg == 0.0:
            raise RobotConnectionError(
                "safety.payload has enforce: true but mass_kg: 0.0. Connecting would push "
                "setPayload(0.0 kg) and overwrite the controller's configured payload for a mounted "
                "tool, so its protective-stop model would under-read the real mass. "
                "WEIGH THE WHOLE ASSEMBLY, not the gripper: the coupling/adapter plate, every cable and "
                "hose that rides on the wrist, and any workpiece the arm carries. A gripper's datasheet "
                "mass is not this number and no value is shipped here on purpose; a plausible default "
                "is how a cell ends up telling the controller about a tool it is not carrying. Set "
                "safety.payload.mass_kg and cog_mm from the scale, or set safety.payload.enforce: false "
                "if the flange is genuinely bare."
            )
        # The second half of the check above. A declared mass with the default centre of
        # gravity tells the controller a multi-kilogram tool is a point mass at the
        # flange face, which is physically impossible for anything bolted on. Measured:
        # at the UR3e shipped max_mass_kg of 3.0 in robot.ur3e.yaml, and the 132 mm tool
        # this project ships, that is 3.0 * 9.81 * 0.132 = 3.88 Nm of wrist torque the
        # controller does not know about. The same model backs gravity compensation, so
        # it also corrupts get_tcp_wrench(), which this driver documents as the hand-over
        # signal. cog_mm is the same flange-frame bench measurement as the tool frame, so
        # taking one without the other is the gap rather than the effort.
        if payload.enforce and payload.mass_kg > 0.0 and all(v == 0.0 for v in payload.cog_mm):
            raise RobotConnectionError(
                f"safety.payload declares mass_kg: {payload.mass_kg} but leaves cog_mm at its "
                "[0, 0, 0] default; that is the 'not measured yet' marker, and pushing it would "
                "tell the controller the tool is a point mass at the flange face. Its protective-stop "
                "model and its gravity compensation (and therefore get_tcp_wrench) would both be "
                "computed against a tool that cannot exist. Measure the CoG offset from the flange in "
                "mm, the same bench step that gives you the tool frame, and set "
                "safety.payload.cog_mm, or set safety.payload.enforce: false for a genuinely bare "
                "flange. A deliberately axis-centred tool is still not [0, 0, 0]: state its real "
                "stand-off, e.g. [0, 0, 60]."
            )
        # The tool frame is declared before a real arm is allowed to move. Silence here
        # is not a default but an unmeasured cell: every pose this stack commands is the
        # grasp centre, and with the UR factory tool register, which is the identity at
        # the flange, a top-down grasp at z=37 mm drives the flange there and the
        # fingertips about 95 mm below the table. Nothing downstream catches that,
        # because the WorkspaceGuard bounds the commanded number and no profile declares
        # a table fixture.
        tool = self.config.gripper.tool_frame
        if tool.source == "undeclared":
            raise RobotConnectionError(
                "gripper.tool_frame.source is 'undeclared': nobody has said where this gripper's "
                "grasp centre sits on the flange, so the driver cannot know whether the poses it "
                "commands mean the TCP or the flange. Measure the coupling + gripper (the same bench "
                "step as safety.payload.cog_mm) and set gripper.tool_frame.offset_mm + "
                "rotation_quat_xyzw, then choose source: 'willy' (this driver composes the transform "
                "and the controller runs a bare flange) or 'polyscope' (an operator set the TCP on the "
                "teach pendant and the driver only verifies it)."
            )
        self._conn.connect()
        if payload.enforce:
            try:
                self._conn.set_payload(payload.mass_kg, payload.cog_mm)
            except BaseException as exc:  # noqa: BLE001 (the rollback below is the whole point)
                # Roll the connection back: an unverified payload on a connected
                # controller is more dangerous than no connection at all.
                #
                # That invariant is why the catch is unconditional. Naming
                # (RuntimeError, OSError, ValueError) reads as thorough and is not.
                # An ``AttributeError`` from a build that lacks or renames
                # ``setPayload``, a ``TypeError`` from a signature change such as a
                # positional centre of gravity against a list, any vendor-specific
                # exception, and a ``KeyboardInterrupt`` mid-push would all skip the
                # rollback and leave the arm connected with a payload the
                # protective-stop model of the controller does not reflect.
                #
                # Measured on the pinned ur_rtde==1.6.5: the symbol exists, as
                # ``setPayload(mass: float, cog: list[float]) -> bool``. So the first two
                # of those are the guard for the next build rather than a suspicion about
                # this one. A handler whose only job is to fail closed does not enumerate
                # failure types.
                self.logger.error(
                    "UR setPayload failed; rolling back the connection: %s",
                    exc,
                )
                self._conn.disconnect()
                if isinstance(exc, Exception):
                    raise RobotConnectionError(
                        f"UR setPayload({payload.mass_kg} kg) failed: {exc}"
                    ) from exc
                raise  # KeyboardInterrupt and SystemExit propagate unchanged, but disconnected first
        # Last, because it needs a live connection: confirm the controller actual tool
        # frame is the one this config assumes. It fails closed.
        #
        # The rollback lives here rather than inside the check, so every failure path
        # rolls back rather than only the ones that remember to. A branch that raises
        # without disconnecting, such as an unbundled DH table for the model, leaves a
        # live RTDE control script owning the arm with the payload already pushed, while
        # the caller is told the connection failed. A long-running caller then cannot
        # recover: `URConnection.connect()` early-returns on an already-open connection,
        # so the retry skips the socket, re-pushes the payload and fails the same way
        # forever. A one-shot CLI hides that behind process exit.
        try:
            # Before the tool-frame check, because where both would fire this one names
            # the cause and that one only names a symptom. Measured: a UR3e container
            # driven with the default `ur.model: ur5e` is refused with a report that the
            # controller is running a tool frame 177.4 mm away, which sends the reader to
            # the pendant TCP. The real answer, that the DH table the check derives from
            # belongs to a different robot, is one dashboard call away.
            self._verify_controller_model()
            self._verify_tool_frame()
        except BaseException:
            self._conn.disconnect()
            raise

    def start_planner(self) -> "SidecarIdentity":
        """Start this arm's cuRobo planner through the path its first planned move takes, and say what it loaded.

        No controller is asked: the planner is built from this arm's config, so a cell can learn at a desk
        whether its planner starts. Every refusal the first move would meet is raised here, as a
        ``CuroboUnavailableError`` naming it: an undeclared planner margin, no hand, an unplaceable hand, a
        missing retract row, a descriptor built for another robot, and a combination no committed evidence
        file measured. A cell whose ``motion_planner`` is not ``curobo`` starts no planner and is refused
        saying so.
        """
        if self._motion_planner != "curobo":
            raise CuroboUnavailableError(
                f"robot.ur.motion_planner is {self._motion_planner!r}, so this cell starts no planner: its moves are "
                f"solved by the controller's IK and judged by the exact mesh guard alone"
            )
        # Under the planner lock for the whole start, about a minute on the cell: a disconnect or a stop_planner meanwhile
        # waits for it and closes the sidecar it started.
        with self._planner_lock:
            self._planner_starting = True
            try:
                return self._curobo_ur_planner().start()
            finally:
                self._planner_starting = False

    def stop_planner(self) -> None:
        """Shut down the planner :meth:`start_planner` or a move started. Idempotent, and the connection is untouched.

        A start in progress is waited for, and the sidecar it started is closed. The planner is retired, not only
        closed: a move still holding it starts no sidecar on it (:meth:`CuroboUrPlanner.retire`).
        """
        with self._planner_lock:
            self._retire_planner()

    def disconnect(self) -> None:
        """Close the RTDE connection, and shut down the cuRobo planning server where one was started.

        A planner start in progress is waited for first (the sidecar's ready timeout bounds it), and the sidecar it
        started is closed, so none is orphaned: the planner is retired, so a move still judging its route on it starts
        no sidecar either. The halt latch lives on the connection and is kept.
        """
        with self._planner_lock:
            self._retire_planner()
        self._observed_tool_frame = None
        self._conn.disconnect()

    def _retire_planner(self) -> None:
        """Shut the planner down for good and drop it, under the planner lock; a double with only ``close`` is closed."""
        self._forget_what_was_judged_ahead()
        planner = self._curobo_ur
        if planner is None:
            return
        retire = getattr(planner, "retire", None)
        if callable(retire):
            retire()
        else:
            planner.close()
        self._curobo_ur = None

    @property
    def planner_state(self) -> str:
        """Where this arm's cuRobo planner is: ``not_used``, ``off``, ``starting`` or ``ready``.

        ``not_used`` on an arm that plans nothing (``motion_planner`` ik). ``starting`` while :meth:`start_planner`
        runs, or while a move starts it. ``ready`` once a sidecar started and passed its checks. Read on every poll of
        the console's cell state: attribute reads only, so it never waits for the planner lock and never raises.
        """
        if self._motion_planner != "curobo":
            return "not_used"
        if self._planner_starting:
            return "starting"
        planner = self._curobo_ur
        if planner is None:
            return "off"
        state = getattr(planner, "state", "off")
        return state if state in ("off", "starting", "ready") else "off"

    def __enter__(self) -> URRobotArm:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._conn.is_connected

    @property
    def pose(self) -> URPose:
        """The controller's TCP register (mm, axis-angle rad): the bare flange in ``willy`` mode.

        The grasp centre every Protocol caller speaks is :meth:`get_tcp_pose`.
        """
        return self._motion.get_current_pose()

    @property
    def joints(self) -> list[float]:
        """Current joint angles in radians."""
        return self._conn.get_joint_positions()

    @property
    def tcp_raw(self) -> list[float]:
        """Raw UR TCP ``[x_m, y_m, z_m, rx, ry, rz]``."""
        return self._conn.get_tcp_pose()

    def _gate_pose(self, pose: Pose, command: MotionCommand, *, commanded: bool = True) -> "MotionResult | None":
        """The full Cartesian pipeline for one commanded pose. ``None`` means every guard passed.

        It is shared with :meth:`move`, so the bool-returning surfaces run the same
        pipeline rather than the workspace box alone. IK is pre-resolved for the same
        reason it is there: without ``target_joints`` the joint-limit, IK-quality and
        arm-against-arm guards have nothing to read and fail closed on every Cartesian
        move. A hand-guided arm is refused first. ``commanded`` false asks the same guards
        of a pose nothing is sent to (:meth:`SafetyPreflight.screen`), which leaves the
        continuity memo as it was.
        """
        refused = self._hand_guided_refusal(command, target_pose=pose)
        if refused is not None:
            return refused
        if self._preflight is None:
            return None
        target_joints: JointPositions | None = None
        current_joints: JointPositions | None = None
        if self._conn.is_connected:
            try:
                target_joints = self.ik(pose)
            except (RobotKinematicsError, RobotConnectionError) as exc:
                status = (MotionStatus.IK_FAILED if isinstance(exc, RobotKinematicsError)
                          else MotionStatus.CONNECTION_ERROR)
                return MotionResult.failed(
                    status, command, target_pose=pose, message=str(exc), exception=exc,
                )
            try:
                current_joints = self.get_joint_positions()
            except RobotConnectionError:
                current_joints = None  # the IK-jump check skips rather than fails closed
        try:
            ctx = self._preflight.context_for_pose(
                pose, command=command, target_joints=target_joints,
                current_joints=current_joints, arm=self,
            )
            decision = (self._preflight.evaluate if commanded else self._preflight.screen)(ctx)
        except RobotConnectionError as exc:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR, command, target_pose=pose,
                message=str(exc), exception=exc,
            )
        return SafetyPreflight.as_motion_result(decision, command, target_pose=pose)

    def _drive_pose(
        self,
        pose: Pose | URPose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
    ) -> bool:
        """Command a Cartesian pose with no safety gate. Callers must have gated it first."""
        return self._motion.move_to(
            self._to_controller_urpose(pose),
            linear=linear, vel=vel, acc=acc, register=register,
            workspace_pose=self._coerce_urpose(pose),
        )

    def move_to(
        self,
        pose: Pose | URPose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
    ) -> bool:
        """Move to ``pose``, gated through the full preflight pipeline.

        It accepts a vendor-neutral :class:`Pose`, which the pipelines and the API use,
        or a :class:`URPose`, which is the controller's own frame. Only tests send one:
        the calibration routine refuses anything but a Pose.

        It runs the whole pipeline and not the workspace box alone, which is all
        ``MotionController.move_to`` checks. A surface that ran the box alone would
        command motion no joint-limit, IK-quality, self-collision, fixture, payload or
        continuity guard ever saw, on a cell where :meth:`move` runs all six. ``False``
        therefore also means a guard refused, and the typed reason is logged.

        On a cuRobo cell it goes through :meth:`move`, which is what makes the two surfaces
        the same motion: the planner plans it, or the line is sampled and judged when
        ``linear`` is set. Reaching the controller directly from here would leave a cell
        configured for cuRobo with one verb that does not use it.
        """
        target = self._pose_from_any(pose)
        if self._motion_planner == "curobo":
            result = self.move(target, linear=linear, vel=vel, acc=acc, register=register)
            if not result.ok:
                self.logger.error("move_to REFUSED by %s: %s", result.status, result.message)
            return result.ok
        rejected = self._gate_pose(target, MotionCommand.MOVE_TO)
        if rejected is not None:
            self.logger.error("move_to REFUSED by %s: %s", rejected.status, rejected.message)
            return False
        return self._drive_pose(pose, linear=linear, vel=vel, acc=acc, register=register)

    def _pose_from_any(self, pose: Pose | URPose) -> Pose:
        """A BASE-frame :class:`Pose` from either accepted input, for the guards to read."""
        if isinstance(pose, URPose):
            return self._pose_from_controller(pose, label=pose.label or "move_to")
        return pose

    def is_inside_workspace(self, pose: Pose | URPose) -> bool:
        """The vendor-neutral workspace pre-check the pipelines use.

        It accepts a :class:`Pose` in the BASE frame, or a :class:`URPose`.
        """
        return self._guard.is_inside_workspace(self._coerce_urpose(pose))

    @staticmethod
    def _coerce_urpose(pose: Pose | URPose) -> URPose:
        """A :class:`Pose` as a :class:`URPose` with no tool frame applied.

        A URPose passes through. This is for the workspace box, which bounds the
        commanded TCP, both in :meth:`is_inside_workspace` and as the ``workspace_pose``
        of every controller command. It is not itself a controller command: in ``willy``
        mode the controller runs a bare flange and needs :meth:`_to_controller_urpose`.
        """
        if isinstance(pose, URPose):
            return pose
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"URRobotArm requires Frame.BASE; got {pose.frame!r}."
            )
        return pose_to_urpose(pose)

    def _to_controller_urpose(self, pose: Pose | URPose) -> URPose:
        """What the controller is told for ``pose``.

        A :class:`URPose` is already in the controller frame, which is how
        :meth:`_pose_from_any` reads it, so it passes through. A
        :class:`Pose` is the grasp centre every Protocol caller speaks, and in ``willy``
        mode it becomes the flange target before the controller sees it, the same
        conversion :meth:`ik` and :meth:`move_linear` make. The guard judges the joints
        for that flange target, so every command that reaches the controller carries the
        same target; a grasp centre sent as a flange target would land one tool length
        further along the approach.
        """
        if isinstance(pose, URPose):
            return pose
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"URRobotArm requires Frame.BASE; got {pose.frame!r}."
            )
        return self._pose_to_controller(pose)

    def move_home(self) -> bool:
        """Move to the home joint configuration, gated. ``False`` means refused; :meth:`move_to_home` says why.

        It is ``self.move_to_home().ok``, so every caller that reads a bool keeps its answer, and
        the typed refusal is in the driver's log as ``move_home REFUSED by <status>: <sentence>``.
        """
        return self.move_to_home().ok

    def move_to_home(self) -> MotionResult:
        """Move to the home joint configuration, gated, and say what the gates found (``HomesTyped``).

        This is the natural first motion on a new cell.
        ``MotionController.move_home`` checks the workspace box, logs a warning and
        proceeds anyway, so routing straight to it would leave the one commanded motion
        on the real path that no safety guard ever saw. The measured consequence on a
        UR3e: the UR5e-authored default puts the grasp centre at z = 561.9 mm against
        that cell ``workspace_limits`` z_max of 320, at 93.4% of its reach.

        So it runs the same destination guards as every other joint move, which is
        ``gate_joint_target`` with joint limits, self-collision and payload, and fails
        closed on the workspace check instead of warning past it. Both halves are
        needed and neither subsumes the other: ``gate_joint_target`` deliberately skips
        the Cartesian workspace box, because that guard does not apply to a
        point-to-point joint command, and here the box matters because the home pose is
        a place the arm will sit.

        On a cuRobo cell the whole way to the home configuration is judged as well, by
        :meth:`_judge_joint_move`: the straight line where it is clear, a cuRobo plan around it
        where it collides. The ``moveJ`` is sent here rather than through
        ``MotionController.move_home``: that method sends an unclamped command at the
        controller's own defaults, and a move that was judged has to be the move that runs. The
        configuration the judge returns is the one handed to the drive, so the configuration
        judged is the configuration sent.

        The result is the refusing gate's own: the destination guard's status relabelled
        ``MOVE_HOME``, ``WORKSPACE_REJECTED`` for a home grasp centre outside the box,
        ``CONTROLLER_REJECTED`` for a home the controller's FK could not place, the path
        judge's status, and for the drive ``CONNECTION_ERROR``, ``INVALID_TARGET`` or
        ``CONTROLLER_REJECTED``, whose message says whether ``moveJ`` had been sent. It is
        stamped as :meth:`move_to_joints` stamps its result. With neither a camera world nor a
        decline in scope on a cuRobo cell it is refused first. Every refusal is logged.
        """
        self._forget_what_was_judged_ahead()
        stamp = self._move_camera_world(UNSET)
        refused = self._refused_for_camera_world(stamp, MotionCommand.MOVE_HOME)
        if refused is not None:
            self.logger.error("move_home REFUSED by %s: %s", refused.status, refused.message)
            return refused
        joints = JointPositions(np.asarray(self._home_joints, dtype=np.float64))
        ahead = self._take_the_next_leg(joints)
        before = self._planner_refresh()
        self._vouched_by = None
        unstamped = self._move_to_home_unstamped(joints, ahead=ahead)
        after = self._planner_refresh()
        ran, self._vouched_by = self._vouched_by, None
        vouched = (after.camera_world() if after is not None and (after is not before or (ran is not None and ran is after))
                   else None)
        result = stamp_result(unstamped, self._move_camera_world(UNSET, planned=vouched))
        if not result.ok:
            self.logger.error("move_home REFUSED by %s: %s", result.status, result.message)
        return result

    def _move_to_home_unstamped(self, joints: JointPositions, *, ahead: "_NextLegJudged | None" = None) -> MotionResult:
        """The body of :meth:`move_to_home`. ``joints`` is the one home every step below reads; ``ahead`` the move home
        judged ahead (:meth:`judge_the_next_leg`), which runs as judged, after the home's own gate, where nothing it was
        judged on changed."""
        refused = self._home_refusal(joints)
        if refused is not None:
            return refused
        ran = self._run_the_next_leg(ahead, command=MotionCommand.MOVE_HOME, done="move_home")
        if ran is not None:
            return ran
        target, route, refused = self._judge_joint_move(joints, command=MotionCommand.MOVE_HOME)
        if refused is not None:
            return refused
        return self._drive_judged_joints(target, route, command=MotionCommand.MOVE_HOME, done="move_home")

    def _home_is_admissible(self, verb: str) -> bool:
        """Whether the home passes :meth:`_home_refusal`, logged under ``verb`` when it does not."""
        refused = self._home_refusal(JointPositions(np.asarray(self._home_joints, dtype=np.float64)))
        if refused is not None:
            self.logger.error("%s REFUSED by %s: %s", verb, refused.status, refused.message)
        return refused is None

    def _home_refusal(self, joints: JointPositions) -> MotionResult | None:
        """The home gate on ``joints``: the typed refusal, or ``None`` where the move may be judged as a path.

        The destination guards run first, then the workspace box on the home pose. The box
        bounds the grasp centre, so the controller FK is converted out of its TCP register
        first: in ``willy`` mode that register is the bare flange, and boxing it would pass a
        home whose grasp centre lies a tool length outside. An FK that cannot be read refuses,
        because an unchecked home is what this gate stops. An arm that is not connected has no
        FK to ask; the path judge and the drive refuse it after this.
        """
        if self._preflight is not None:
            rejected = self._preflight.gate_joint_target(joints, arm=self)
            if rejected is not None:
                message = rejected.message
                if float(np.max(np.abs(joints.values))) > 2.0 * np.pi:
                    # Every UR joint stops at one full turn either way, so a value beyond 2*pi cannot be
                    # reached in radians, and a home copied off the pendant in degrees is how one is written.
                    message += (" robot.home_joint_positions holds a value beyond 2*pi, which reads as degrees; "
                                "the key is in radians.")
                return dataclasses.replace(
                    rejected, command=MotionCommand.MOVE_HOME, target_joints=joints, message=message,
                )
        if not self._conn.is_connected:
            return None
        try:
            reported = URPose.from_ur_list(self._conn.fk(joints.tolist()), label="home")
            home_pose = self._coerce_urpose(self._pose_from_controller(reported, label="home"))
            inside = self._guard.is_inside_workspace(home_pose)
        except Exception as exc:  # noqa: BLE001 (FK unavailable must not silently wave the move through)
            return MotionResult.failed(
                MotionStatus.CONTROLLER_REJECTED, MotionCommand.MOVE_HOME, target_joints=joints,
                message=(
                    f"could not check the home pose: the controller's forward kinematics of the home "
                    f"configuration could not be read ({type(exc).__name__}: {exc}), and a home the "
                    f"workspace box never saw is not sent"
                ),
                exception=exc,
            )
        if inside:
            return None
        return MotionResult.failed(
            MotionStatus.WORKSPACE_REJECTED, MotionCommand.MOVE_HOME, target_joints=joints,
            message=self._home_outside_the_box(home_pose),
        )

    def _home_outside_the_box(self, home_pose: URPose) -> str:
        """Why a home grasp centre outside ``workspace_limits`` is refused: the box, and what it bounds."""
        box = self._guard.limits
        boxed = {
            "willy": ("The box bounds the grasp centre, the flange the controller's FK reports plus the declared "
                      "tool frame, and not the flange itself."),
            "polyscope": "The box bounds the grasp centre, the TCP the controller's FK reports with the pendant's tool.",
        }.get(self.config.gripper.tool_frame.source,
              "The box bounds the grasp centre, and with no tool frame declared that is the flange the FK reports.")
        if tuple(float(v) for v in self._home_joints) == HOME_JOINTS_DEFAULT:
            fix = ("No robot.home_joint_positions is set, so this is the shipped default, the arm standing "
                   "straight up; set one whose grasp centre lies inside this cell's box.")
        else:
            fix = ("Set robot.home_joint_positions, in radians, to a configuration whose grasp centre lies "
                   "inside this cell's box.")
        return (
            f"the home grasp centre ({home_pose.x:.1f}, {home_pose.y:.1f}, {home_pose.z:.1f}) mm is outside "
            f"workspace_limits (x {box.x_min:.1f} to {box.x_max:.1f}, y {box.y_min:.1f} to {box.y_max:.1f}, "
            f"z {box.z_min:.1f} to {box.z_max:.1f} mm). {boxed} {fix} Widening the box widens it for every "
            f"move, not only for home."
        )

    def stop(self) -> None:
        """Halt the arm: :meth:`halt` with "stop() was called". Always safe, never raises, sends nothing itself.

        Not an emergency stop; the red button is. Before the halt this sent ``stopJ`` and ``stopL`` from the calling
        thread, which a synchronous move in flight did not read until it ended (``scripts/ursim/probe_halt.py`` M0),
        and latched nothing, so the next move went out as if nothing had been pressed.
        """
        self.halt("stop() was called")

    # --- SupportsHalt ---
    def halt(self, reason: str) -> HaltState:
        """Latch the arm ("halt now"): no motion and no output from now on until :meth:`clear_halt`. Never raises.

        Nothing is sent from the calling thread. With ``robot.ur.brake_on_halt`` the move in flight is braked under
        control by the thread that sent it (``stopJ``/``stopL``); without it, as shipped, that move runs to its end and
        nothing after it is sent. The latch lives on the connection, so a disconnect and a connect of this arm keep it.
        """
        return self._conn.request_halt(reason)

    def clear_halt(self) -> None:
        """End the latch, which gives motion and the outputs back: the console's "Zelle ist frei"."""
        self._conn.clear_halt()

    def halt_state(self) -> HaltState | None:
        """The latch while it is set, else ``None``: a lock-free read that never raises."""
        return self._halt_record()

    def brakes_in_motion(self) -> bool:
        """Whether a halt brakes a move in flight here (``robot.ur.brake_on_halt``), or lets it run to its end."""
        return self._conn.brake_on_halt is True

    def _halt_record(self) -> HaltState | None:
        """The connection's latch while it is set; strict, so a connection double that answers anything is not halted."""
        read = getattr(self._conn, "halt_state", None)
        try:
            state = read() if callable(read) else None
        except Exception:  # noqa: BLE001 (a latch that cannot be read is no latch; the connection's own refuses on)
            return None
        return state if isinstance(state, HaltState) else None

    def _cancelled_by_the_halt(
        self, command: MotionCommand, verb: str, *,
        target_pose: Pose | None = None, target_joints: JointPositions | None = None,
    ) -> MotionResult:
        """The CANCELLED result of a move the halt ended, saying whether ``verb`` was sent and how it ended."""
        end = getattr(self._conn, "last_move_end", None)
        state = self._halt_record()
        reason = state.reason if state is not None else "a halt that has been cleared since"
        stop = "stopJ" if verb == "moveJ" else "stopL"
        if end is MoveEnd.BRAKED:
            took = (f", standing still {state.brake_s:.2f} s after the halt" if state is not None
                    and state.brake_s is not None else "")
            what = (f"{verb} was sent and braked under control ({stop}){took}: the arm stands where the brake stopped "
                    f"it")
        elif end is MoveEnd.BRAKE_UNCONFIRMED:
            what = (f"{verb} was sent and braked ({stop}), and the arm was not seen to stand still: if it still moves, "
                    f"press the emergency stop")
        else:
            what = f"{verb} was not sent, and nothing reached the controller"
        return MotionResult.failed(MotionStatus.CANCELLED, command, target_pose=target_pose,
                                   target_joints=target_joints, message=halted_refusal(reason, what))

    def _ended_by_the_halt(self) -> bool:
        """Whether the last move the connection was asked for was refused or braked by the halt latch."""
        return getattr(self._conn, "last_move_end", None) in HALT_ENDS

    def wait_until_steady(
        self,
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.02,
    ) -> bool:
        """Delegate to the underlying RTDE :meth:`URConnection.wait_until_steady`."""
        if not self._conn.is_connected:
            raise RobotConnectionError("wait_until_steady() requires an open connection.")
        return bool(self._conn.wait_until_steady(timeout_s, poll_interval_s))

    @property
    def gates_its_own_sends(self) -> bool:
        """Whether this arm judges a motion first and waits for steady right before it sends it (``safety.dwell.gate_at``
        ``send``, a cuRobo arm), so a verb asks it to move with no gate of its own (``execution.motion.steady_timeout_of``,
        ``GraspExecutionPolicy._drive_to``). ``False`` as shipped: every verb waits before it asks, as ever."""
        return self._gates_sends

    #: How often one motion is judged before it is refused where the arm stands elsewhere than it was judged from each time
    #: the steady gate lets it go (:meth:`_send_gate`): as it was asked, and again from where the gate found it at rest.
    _SEND_JUDGEMENTS = 3

    def _send_gate(
        self, judged_from: "Sequence[float] | None", command: MotionCommand, *,
        target_pose: Pose | None = None, target_joints: JointPositions | None = None,
    ) -> "MotionResult | bool | None":
        """The steady gate right before a send, where this arm gates its own sends (:attr:`gates_its_own_sends`).

        ``None`` where the motion may be sent now, and always on an arm that does not gate its sends. A TIMEOUT result
        where the arm did not come to rest within ``safety.dwell.steady_timeout_s``, and a CONNECTION_ERROR where the
        controller could not be asked: nothing is sent. ``True`` where it came to rest more than :data:`_AHEAD_MOVED_MM`
        from ``judged_from``, the joints the motion was judged from (the map's C4): the caller judges it again from
        where it stands. A halt that lands meanwhile is refused by the send itself, which reads the latch first.
        """
        if not self._gates_sends:
            return None
        timeout = float(self.config.safety.dwell.steady_timeout_s)
        try:
            steady = self.wait_until_steady(timeout)
        except (RobotConnectionError, RuntimeError, OSError) as exc:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR, command, target_pose=target_pose, target_joints=target_joints,
                message=(f"the steady gate before the send could not ask the controller ({type(exc).__name__}: {exc}); "
                         "nothing was sent"), exception=exc,
            )
        if not steady:
            return MotionResult.failed(
                MotionStatus.TIMEOUT, command, target_pose=target_pose, target_joints=target_joints,
                message=(f"the arm did not come to rest within {timeout:.2f} s before the motion was sent (safety.dwell, "
                         "gate_at 'send'); nothing was sent"),
            )
        if judged_from is None:
            return None
        moved = self._moved_since_mm(judged_from)
        if moved <= self._AHEAD_MOVED_MM:
            return None
        self.logger.info("the arm came to rest %s from where its motion was judged from, more than %g mm: it is judged "
                         "again from where it stands", f"{moved:.2f} mm" if math.isfinite(moved) else "somewhere that "
                         "cannot be compared", self._AHEAD_MOVED_MM)
        return True

    def _kept_moving(
        self, command: MotionCommand, *, target_pose: Pose | None = None, target_joints: JointPositions | None = None,
    ) -> MotionResult:
        """The refusal of a motion whose arm came to rest elsewhere than it was judged from at every steady gate."""
        return MotionResult.failed(
            MotionStatus.TIMEOUT, command, target_pose=target_pose, target_joints=target_joints,
            message=(f"the arm came to rest more than {self._AHEAD_MOVED_MM:g} mm from where the motion was judged from "
                     f"at each of {self._SEND_JUDGEMENTS} steady gates, so no judgement of it began where it stands; "
                     "nothing was sent"),
        )

    def move(
        self,
        pose: Pose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
        camera_world: Maybe[CameraWorldDecline] = UNSET,
    ) -> MotionResult:
        """The typed counterpart of :meth:`move_to`.

        Every commanded pose goes through the vendor-neutral :class:`SafetyPreflight`
        pipeline. A rejection from any guard produces the matching typed
        :class:`MotionStatus`, one of ``WORKSPACE_REJECTED``, ``JOINT_LIMIT_REJECTED``,
        ``IK_QUALITY_REJECTED``, ``SELF_COLLISION_REJECTED``, ``PAYLOAD_REJECTED`` and
        ``CONTINUITY_REJECTED``, so a caller branches on the precise cause without
        scraping a log. A connection fault raised by the underlying bool path is caught
        and reported as :attr:`MotionStatus.CONNECTION_ERROR`.

        The result says what stood behind the motion: UNPLANNED on the ik planner, and on cuRobo
        DECLINED for a decline (``camera_world``, or :meth:`without_camera_world`), MISSING while
        no live camera world is wired, and once one is, PLANNED when the refresh this motion made
        vouched for the cell and UNSTATED when none did, which is why it is read after the motion. A
        declined motion on an arm whose live camera world is wired is refused with ``UNSUPPORTED``
        before the planner is asked, because the planner is handed that world before every plan and
        cannot set it aside for one motion. A MISSING motion, with neither a world nor a decline, is
        refused the same way (``NO_CAMERA_WORLD_MESSAGE``), before it reads a connection or starts a
        planner.

        A route judged ahead to this very pose runs as it was judged where nothing it was judged on changed since
        (:meth:`_drive_curobo`), and the refresh it was judged in vouches for it, as the one that built a held world
        vouches for a line judged in it. Whatever this move does, nothing judged ahead outlives it.
        """
        stamp = self._move_camera_world(camera_world)
        refused = self._refused_for_camera_world(stamp, MotionCommand.MOVE_TO, target_pose=pose)
        if refused is not None:
            self._forget_what_was_judged_ahead()
            return refused
        before = self._planner_refresh()
        self._vouched_by = None
        unstamped = self._move_unstamped(pose, linear=linear, vel=vel, acc=acc, register=register)
        after = self._planner_refresh()
        ahead, self._vouched_by = self._vouched_by, None
        # Inside a held world no refresh is made for the motion: the one that built the world it was judged in vouches.
        # So does the one a route judged ahead was judged in, where that route ran (:meth:`_drive_curobo`).
        held = self._held_world_reason is not None or (ahead is not None and ahead is after)
        vouched = after.camera_world() if after is not None and (after is not before or held) else None
        return stamp_result(unstamped, self._move_camera_world(camera_world, planned=vouched))

    @contextmanager
    def held_world(self, reason: str) -> Iterator[None]:
        """Judge every line inside the block against the world the last refresh built, with no new frame.

        :class:`~src.robot.core.arm_capabilities.HoldsItsWorld`. A grasp holds it from the part until the arm is back
        up (the owner, 2026-10-06: "Ja, beides"): the lift judged before the jaws close, the lift itself and the line
        back up with the jaws open are judged against the world the line down was judged in, whose boxes the line down
        kept clear of, and whose region between the jaws stood at the part. A refresh at the part, from there, held
        boxes that one did not, and refused every way back up (the grasp bench, 2026-10-06). Every check the arm makes
        still runs: the exact guard on every sample, the planner against that world, the controller, the halt. Where no
        refresh built a world yet, a line refreshes as ever. ``reason`` is said in the log of every line judged so.

        Nothing judged ahead before the block outlives its start (:meth:`grasp_refusal_ahead`), and no lift judged inside
        it outlives its end (:meth:`carried_line_refusal`).
        """
        if not str(reason).strip():
            raise ValueError("a held world says why it is held")
        self._forget_what_was_judged_ahead()
        previous = self._held_world_reason
        self._held_world_reason = str(reason)
        try:
            yield
        finally:
            self._held_world_reason = previous
            self._lift_judged = None

    def holds_world_at(self, junction: str) -> bool:
        """Whether this arm holds its world across ``junction`` rather than taking a new frame there
        (``safety.planning_world.hold``: ``standoff``, ``carry``, ``drop``), where a live camera world is wired. A verb
        reads it before it holds one (``handling``). ``False`` as shipped."""
        return self._live_world is not None and self._holds_at(junction)

    def _holds_at(self, junction: str) -> bool:
        """The switch ``safety.planning_world.hold.<junction>``, read as ``is True`` (:func:`_holds_its_world_at`)."""
        return _holds_its_world_at(self, junction)

    @property
    def judges_next_legs(self) -> bool:
        """Whether a verb judges the next leg while the jaws' stroke is waited out (``robot.motion.judge_next_leg`` not
        ``off``, on a cuRobo arm): it then has the hand leave the stroke to it (``set_closed(wait=False)``) and asks this
        arm meanwhile (:meth:`judge_line_ahead`, :meth:`judge_the_next_leg`). ``False`` as shipped."""
        return self._motion_planner == "curobo" and self._next_leg_mode in ("in_settles", "in_settles_and_motion")

    def _judges_in_motion(self) -> bool:
        """Whether a declared joint move is judged on a second thread while the line before it runs: ``in_settles_and_
        motion``, on a send a halt brakes (``robot.ur.brake_on_halt``), whose thread watches the line and brakes it."""
        return self.judges_next_legs and self._next_leg_mode == "in_settles_and_motion" and self.brakes_in_motion()

    @contextmanager
    def holding_the_approach(self, standoff: Pose, reason: str) -> Iterator[bool]:
        """Hold the world a grasp judged ahead to ``standoff`` was judged in, for the move to the standoff and the line
        down from it (``safety.planning_world.hold.standoff``, the owner's O2 of 2026-10-09): yields ``True`` while it
        holds it, ``False`` where it holds nothing and every motion refreshes as ever.

        It holds where the switch is on and the route judged ahead to this very standoff stands ready to run as it was
        judged (:meth:`_why_judged_again`): the line down was judged ahead in that world, and the route in it too
        (:meth:`grasp_refusal_ahead`), so the line down at the standoff is judged in the world it was judged in at the
        look rather than in a new frame. The route judged ahead outlives the start of the block, for the one move it
        serves; otherwise the block is :meth:`held_world`.
        """
        if not str(reason).strip():
            raise ValueError("a held world says why it is held")
        ahead = self._judged_ahead
        planner = self._curobo_ur
        why = ("safety.planning_world.hold.standoff is false" if not self._holds_at("standoff")
               else "no route to this standoff was judged ahead" if ahead is None or ahead.goal != _pose_key(standoff)
               or planner is None else self._why_judged_again(ahead, planner, standoff, velocity=ahead.velocity))
        if why:
            if self._holds_at("standoff"):
                self.logger.info("the world is not held for the approach to %s (%s): it is refreshed as ever",
                                 standoff.label or "<unlabeled>", why)
            yield False
            return
        previous, held_for = self._held_world_reason, self._approach_held_for
        self._held_world_reason = str(reason)
        self._approach_held_for = ahead
        try:
            yield True
        finally:
            self._held_world_reason = previous
            self._approach_held_for = held_for
            self._lift_judged = None

    @contextmanager
    def expecting_next(self, joints: JointPositions) -> Iterator[None]:
        """Within the block, the joint move to ``joints`` is the leg after the verbs running in it: the carry to the bin's
        look a task makes after its pick, the return after its place.

        Where ``robot.motion.judge_next_leg`` says so and a junction holds its world (``safety.planning_world.hold``),
        a verb judges that move ahead from where its line will leave the arm (:meth:`judge_the_next_leg`), and the move
        runs as it was judged where nothing it was judged on changed. Declaring it moves nothing and judges nothing: a
        move nobody judged ahead is judged as it runs.
        """
        previous = self._expected_next
        values = JointPositions(tuple(float(v) for v in joints.tolist()))
        self._expected_next = (_joints_key(values), values)
        try:
            yield
        finally:
            self._expected_next = previous

    #: How long a joint move judged ahead in a held world stays ready (:meth:`_why_the_next_leg_is_judged_again`): the
    #: jaws' stroke, the line after it and what a pick and a task do before the move, with room, not a person's answer.
    _HELD_AHEAD_HOLDS_S = 10.0
    #: How long the line's send waits, once it ended, for the joint move judged on a second thread during it.
    _NEXT_LEG_JOIN_S = 30.0

    def judge_the_next_leg(self, after: Pose, *, junction: str, now: bool = True) -> str:
        """Judge the joint move declared next (:meth:`expecting_next`) from where the line to ``after`` will leave the
        arm, in the world held now, and keep it for that move; ``""`` where it was kept or left to the line, else why
        not, said in the log. Nothing moves and nothing is commanded.

        ``junction`` names the hold the world is held under (``carry`` at a part, ``drop`` at a place,
        ``safety.planning_world.hold``). ``now`` judges it here, where the arm stands still while a verb waits out the
        jaws' stroke (``robot.motion.judge_next_leg``): from the controller's solution of ``after`` seeded on the joints
        the arm stands at. ``now`` false leaves it to the line to ``after`` where this arm judges legs during motion
        (``in_settles_and_motion``, on a send a halt brakes): the line's send solves its end before it is sent and
        judges the move on a second thread while the line runs (:meth:`_drive_checked_line`).

        The move judged is the straight joint line from that start to the declared joints, turned onto their twin near
        it, through the destination guards and both authorities at ``safety.planned_motion.line_clearance_mm``, as the
        move would judge it as it runs. A line that is refused keeps nothing, and the move is judged as it runs. What was
        judged ahead before is another verb's, and is forgotten here.
        """
        self._next_leg_during = None
        self._drop_the_next_leg()
        why = self._why_no_next_leg(junction)
        if why:
            self.logger.info("the joint move declared next is not judged ahead at %s: %s", junction, why)
            return why
        if not now:
            if not self._judges_in_motion():
                return "this arm judges no leg during motion"
            self._next_leg_during = (_pose_key(after), junction)
            return ""
        start = self._predicted_end(after)
        if isinstance(start, str):
            self.logger.info("the joint move declared next is not judged ahead at %s: %s", junction, start)
            return start
        planner = self._curobo_ur
        return self._judge_the_next_leg_from(start, self._next_leg_ask(), state=self._judgement_state(planner),
                                             boxes=self._seen_boxes(), epoch=None)

    def _why_no_next_leg(self, junction: str) -> str:
        """Why no joint move is judged ahead at ``junction`` now, or ``""``."""
        if not self.judges_next_legs:
            return "robot.motion.judge_next_leg is off"
        if not self._holds_at(junction):
            return f"safety.planning_world.hold.{junction} is false"
        if self._expected_next is None:
            return "no joint move was declared next"
        if self._held_world_reason is None:
            return "no world is held"
        refresh = self._planner_refresh()
        if refresh is None or not refresh.ok or self._curobo_ur is None or self._preflight is None:
            return "no refresh that vouched for the cell built the world held"
        if not self._conn.is_connected:
            return "the arm is not connected"
        return ""

    def _next_leg_ask(self) -> "tuple[tuple[tuple[object, ...], JointPositions], str, WorldRefresh, object]":
        """What a judgement of the joint move declared next reads of this arm, read on the thread that sends: the move
        declared, the held world's reason, the refresh that built it and the planner holding it."""
        declared, reason, refresh = self._expected_next, self._held_world_reason, self._planner_refresh()
        assert declared is not None and reason is not None and refresh is not None  # noqa: S101 (_why_no_next_leg)
        return declared, reason, refresh, self._curobo_ur

    def _predicted_end(self, after: Pose) -> "list[float] | str":
        """The joints the line to ``after`` ends at, as the controller solves ``after`` seeded on the joints the arm
        stands at; or why it cannot say."""
        try:
            here = JointPositions([float(v) for v in self._conn.get_joint_positions()])
            return [float(v) for v in self.ik(after, seed=here).tolist()]
        except (RobotKinematicsError, RobotConnectionError, RuntimeError, OSError) as exc:
            return f"where the line to {after.label or 'its end'} ends could not be solved ({type(exc).__name__}: {exc})"

    def _judge_the_next_leg_from(
        self, start: "Sequence[float]",
        ask: "tuple[tuple[tuple[object, ...], JointPositions], str, WorldRefresh, object]", *,
        state: "tuple[object, ...]", boxes: "tuple[object, ...]", epoch: "int | None",
    ) -> str:
        """Judge the straight joint line from ``start`` to the move ``ask`` declared, in its held world, and keep it; ``""``
        where it was kept. ``state`` and ``boxes`` were read on the sending thread where the arm stood still; ``epoch``
        is the count a judgement on a second thread keeps its result under, ``None`` on the sending thread. It sends
        nothing and asks the controller nothing: a second thread runs it while the line before it is sent."""
        (key, declared), reason, refresh, planner = ask
        try:
            target, _turned = self._twin_near_the_arm(declared, start)
            assert self._preflight is not None  # noqa: S101 (_why_no_next_leg)
            refused = self._preflight.gate_joint_target(target, arm=self)
            line = [[float(v) for v in start], [float(v) for v in target.tolist()]]
            if refused is None:
                refused = self._judge_legs(planner, line, command=MotionCommand.MOVE_JOINTS,  # type: ignore[arg-type]
                                           target_joints=target, clearance_mm=self._line_clearance_mm())
        except Exception as exc:  # noqa: BLE001 (a judgement ahead that raised keeps nothing; the move is judged as it runs)
            self.logger.warning("the joint move declared next could not be judged ahead (%s: %s): it is judged as it runs",
                                type(exc).__name__, exc)
            return f"it could not be judged ahead ({type(exc).__name__}: {exc})"
        if refused is not None:
            self.logger.info("the joint move declared next, judged ahead in the world held (%s), would be refused "
                             "(%s: %s): it is judged as it runs", reason, refused.status.value, refused.message)
            return f"it would be refused ({refused.status.value}: {refused.message})"
        judged = _NextLegJudged(
            target=key, joints=target, route=self._route(line, planned=False, how="direct line judged ahead"),
            start=tuple(float(v) for v in start), reason=reason, refresh=refresh, planner=planner, state=state,
            boxes=boxes, at=time.monotonic(),
        )
        with self._next_leg_lock:
            if epoch is not None and epoch != self._next_leg_epoch:
                return "its line ended without it"
            self._next_leg_ahead = judged
        self.logger.info("the joint move declared next is judged ahead in the world held (%s), from where the line before "
                         "it ends: it runs as judged where the arm stands within %g mm of there", reason,
                         self._AHEAD_MOVED_MM)
        return ""

    def _take_the_next_leg(self, joints: JointPositions) -> "_NextLegJudged | None":
        """The joint move judged ahead, taken by the joint move to ``joints`` where it is the one declared; it serves one
        move, so it is gone either way, and a judgement still running for it is void."""
        with self._next_leg_lock:
            judged, self._next_leg_ahead = self._next_leg_ahead, None
            self._next_leg_epoch += 1
        if judged is None:
            return None
        if judged.target != _joints_key(joints):
            self.logger.info("this joint move goes elsewhere than the one judged ahead: it is judged as it runs")
            return None
        return judged

    def _drop_the_next_leg(self) -> None:
        """Forget the joint move judged ahead, and void a judgement of it still running: a motion that is not it came."""
        with self._next_leg_lock:
            self._next_leg_ahead = None
            self._next_leg_epoch += 1

    def _run_the_next_leg(
        self, ahead: "_NextLegJudged | None", *, command: MotionCommand, done: str,
        velocity: float | None = None, acceleration: float | None = None,
    ) -> MotionResult | None:
        """Run the joint move judged ahead as it was judged, where nothing it was judged on changed since
        (:meth:`_why_the_next_leg_is_judged_again`), its refresh vouching for it; ``None`` where there is none, or it is
        judged again as it runs. It goes out through :meth:`_drive_judged_joints`, the steady gate and the halt latch
        before every ``moveJ`` as on any judged move."""
        if ahead is None:
            return None
        why = self._why_the_next_leg_is_judged_again(ahead)
        if why:
            self.logger.info("%s: the joint move judged ahead is judged again: %s", done, why)
            return None
        self.logger.info("%s: the joint move judged ahead runs as it was judged, %.0f ms ago, in the world held (%s); it "
                         "is not judged a second time", done, 1000.0 * (time.monotonic() - ahead.at), ahead.reason)
        self._vouched_by = ahead.refresh
        return self._drive_judged_joints(ahead.joints, ahead.route, command=command, done=done, velocity=velocity,
                                         acceleration=acceleration)

    def _why_the_next_leg_is_judged_again(self, judged: _NextLegJudged) -> str:
        """Why the joint move judged ahead is judged again rather than run as it was judged; ``""`` where it runs.

        It runs only where all of this holds, read now: the same planner holds the world of the refresh it was judged
        in, which no refresh replaced and which vouched for the cell; at most :data:`_HELD_AHEAD_HOLDS_S` passed since;
        the arm is not halted; it stands within :data:`_AHEAD_MOVED_MM` of where the move was judged from; and what else
        the judgement read is as it was (:meth:`_judgement_state`, the halts it counted among it, and the camera's boxes
        the exact guard holds). The world held is the owner's switch (``safety.planning_world.hold``); none needs to be
        held still, since none was refreshed.
        """
        planner = self._curobo_ur
        if planner is not judged.planner:
            return "the planner is another than the one that judged it"
        refresh = getattr(planner, "last_world_refresh", None)
        if refresh is not judged.refresh or not judged.refresh.ok:
            return "the planner's world was refreshed since it was judged"
        took_s = time.monotonic() - judged.at
        if not took_s <= self._HELD_AHEAD_HOLDS_S:
            return f"{took_s:.2f} s passed since it was judged, more than {self._HELD_AHEAD_HOLDS_S:g} s"
        if self._halt_record() is not None:
            return "the arm is halted"
        moved = self._moved_since_mm(judged.start)
        if not moved <= self._AHEAD_MOVED_MM:
            return (f"the arm stands {moved:.2f} mm from where it was judged from, more than {self._AHEAD_MOVED_MM:g} mm"
                    if math.isfinite(moved) else "where the arm stands cannot be compared with where it was judged from")
        if self._judgement_state(planner) != judged.state:
            return "the carried part, the hand, the camera's boxes the planner holds or the halts changed since"
        if self._seen_boxes() is not judged.boxes:
            return "the exact guard holds other boxes the camera saw than the ones it was judged against"
        return ""

    def _judging_during(self, pose: Pose, during: "tuple[tuple[object, ...], str] | None") -> "threading.Thread | None":
        """The second thread that judges the joint move declared next while the line to ``pose`` runs, started, where that
        line is the one :meth:`judge_the_next_leg` left it to; ``None`` where none is to run. Its start is solved, and
        what it reads of the arm is read, here, before the line is sent: the thread asks the controller nothing.

        Not on a cell that models a carried part while none is carried: there the judgement reads the hand
        (``why_not_known_open``, a toggle's output read off the controller) where the camera's boxes meet the move, so
        that move is judged as it runs."""
        if during is None or during[0] != _pose_key(pose) or not self._judges_in_motion():
            return None
        why = self._why_no_next_leg(during[1])
        if not why and self._payload_config_declined() is None and self._attached_payload is None:
            why = ("this cell models a carried part and none is carried, so the judgement reads the hand off the "
                   "controller, which the second thread never asks")
        start = self._predicted_end(pose) if not why else why
        if isinstance(start, str):
            self.logger.info("the joint move declared next is not judged during the line to %s: %s",
                             pose.label or "<unlabeled>", start)
            return None
        ask = self._next_leg_ask()
        state, boxes = self._judgement_state(ask[3]), self._seen_boxes()
        with self._next_leg_lock:
            self._next_leg_epoch += 1
            epoch = self._next_leg_epoch
            self._next_leg_ahead = None
        worker = threading.Thread(target=self._judge_the_next_leg_from, args=(start, ask),
                                  kwargs={"state": state, "boxes": boxes, "epoch": epoch}, name="judge-the-next-leg",
                                  daemon=True)
        # The thread that watches the line reads the halt every 8 ms, and the judgement must not keep it waiting for
        # the GIL (:data:`_WATCHED_SWITCH_S`).
        _switching_often()
        try:
            worker.start()
        except BaseException:
            _switching_as_before()
            raise
        self.logger.info("the joint move declared next is judged on a second thread while the line to %s runs",
                         pose.label or "<unlabeled>")
        return worker

    def _joined(self, worker: "threading.Thread | None") -> None:
        """Wait for the judgement :meth:`_judging_during` started, once its line ended; a judgement still running after
        :data:`_NEXT_LEG_JOIN_S` keeps nothing, and the move is judged as it runs. The switch interval is put back
        either way."""
        if worker is None:
            return
        try:
            worker.join(self._NEXT_LEG_JOIN_S)
        finally:
            _switching_as_before()
        if worker.is_alive():
            with self._next_leg_lock:
                self._next_leg_epoch += 1
                self._next_leg_ahead = None
            self.logger.warning("the joint move declared next was still being judged %g s after its line ended: it is "
                                "judged as it runs", self._NEXT_LEG_JOIN_S)

    def judge_line_ahead(self, pose: Pose) -> MotionResult | None:
        """Judge the line to ``pose`` from where the arm stands, now, in the world held now, as the line will run, and keep
        it for that one line (:meth:`_drive_checked_line`): the line out of a place, judged while the jaws open
        (``robot.motion.judge_next_leg``, ``safety.planning_world.hold.drop``). ``None`` where the line would run, else
        the refusal it would meet, which the line meets again as it runs. Nothing moves and nothing is commanded.

        It is the line's own judgement (:meth:`_judge_linear_move`, every sample solved by the controller seeded where
        the arm stands), made early. It is kept only inside a held world, built by a refresh that vouched for the cell,
        and it runs only where nothing it read changed (:meth:`_why_the_lift_is_judged_again`): the arm within
        :data:`_AHEAD_MOVED_MM` of where it stood, the same held world, the carried part, the camera's boxes, the halts.
        """
        self._lift_judged = None
        if pose.frame is not Frame.BASE:
            return MotionResult.failed(
                MotionStatus.INVALID_TARGET, MotionCommand.MOVE_TO, target_pose=pose,
                message=f"URRobotArm.judge_line_ahead requires Frame.BASE; got {pose.frame!r}",
            )
        refused = self._refused_for_camera_world(self._move_camera_world(UNSET), MotionCommand.MOVE_TO,
                                                 target_pose=pose)
        if refused is not None:
            return refused
        joints = self._joints_now()
        judged = self._judge_linear_move(pose, command=MotionCommand.MOVE_TO, commanded=False)
        refresh = self._planner_refresh()
        if judged is None and joints is not None and self._held_world_reason is not None and refresh is not None \
                and refresh.ok:
            planner = self._curobo_ur
            self._lift_judged = _LiftJudged(
                pose=_pose_key(pose), reason=self._held_world_reason, refresh=refresh, planner=planner, joints=joints,
                state=self._judgement_state(planner, hand=False), boxes=self._seen_boxes(), stands_in=False,
            )
        self.logger.info("the line to %s judged ahead where the arm stands: %s", pose.label or "<unlabeled>",
                         ("it would run" + ("" if self._lift_judged is not None else ", and nothing is kept outside a held "
                                            "world")) if judged is None else f"it would be refused: {judged.message}")
        return judged

    def _joints_now(self) -> "tuple[float, ...] | None":
        """The joints the arm stands at, or ``None`` where they cannot be read."""
        try:
            return tuple(float(v) for v in self._conn.get_joint_positions())
        except (RobotConnectionError, RuntimeError, OSError):
            return None

    @contextmanager
    def keeping_its_branch(self, overshoot_deg: float = 15.0) -> Iterator[_KeptBranch]:
        """Route every Cartesian move inside the block on the branch the arm holds alone, every cuRobo plan swinging no
        joint more than ``overshoot_deg`` past its span; yields what the block refused for it.

        The owner's R2 (2026-10-08): a change of branch is the last resort. A pick tries every grasp of a look inside
        such a block first and leaves the grasps it refuses for their branch until after them
        (``pick_loop._try_the_grasps``). Inside it, :meth:`_route_to_the_nearest_goal` screens, lines and plans the goals
        on the arm's branch alone, a plan past ``overshoot_deg`` counts as a detour as one past
        ``safety.planned_motion.max_detour_deg`` always does, and :meth:`nearest_configuration` answers among the same
        goals. Where nothing on the branch runs while a goal on another branch, or a plan within the configured bound,
        was left untried, the move is refused before anything is sent, ``JOINT_LIMIT_REJECTED``, saying it is
        :data:`_KEEPS_ITS_BRANCH`, and counted on the yielded record (``refused``). Every judgement still runs: the block
        only narrows what may run. Nothing judged ahead before it outlives its start.
        """
        bound = float(overshoot_deg)
        if not math.isfinite(bound) or bound < 0.0:
            raise ValueError(f"a kept branch bounds a plan's overshoot by a number of degrees, got {overshoot_deg!r}")
        self._forget_what_was_judged_ahead()
        previous = self._keeping
        kept = _KeptBranch(overshoot_deg=bound)
        self._keeping = kept
        try:
            yield kept
        finally:
            self._keeping = previous

    def has_a_goal_on_its_branch(self, pose: Pose, here: "Sequence[float]") -> "bool | None":
        """Whether ``pose`` has a configuration on the branch the arm holds at ``here``, inside the goal window, read on
        the closed form alone: no screen, no planner, no controller, about a millisecond and a half. ``None`` where it
        cannot tell: a pose outside BASE, joints that are no configuration of this arm, a model with no DH chain.

        The window gap (the owner, 2026-10-08): inside the half-turn cable window, a grasp turned the cell's natural way
        round (``robot.natural_closing_axis``) can have no configuration on the arm's branch where its twin, half a turn
        about its approach, has one, wrist 3 half a turn on and the branch the same; the pick takes the twin before it
        leaves the grasp for later or changes branch (``pick_loop._closing_along``). On 2026-10-07 the natural turn
        needed wrist 3 at +90 degrees, the window about that day's home ended at +86.4, and the cell judged a shoulder and
        elbow flip. The goals are the ones a move ranks (:meth:`_goals_in_window`, :meth:`_ranked_goals`): true is a
        first group a move to ``pose`` screens before any other branch.
        """
        if pose.frame is not Frame.BASE:
            return None
        joints = [float(v) for v in here]
        if len(joints) != self.capabilities.dof or not all(math.isfinite(v) for v in joints):
            return None
        goals = self._goals_in_window(self._pose_to_flange(pose), pose, joints,
                                      velocity=float(self.config.motion_limits.max_velocity))
        if isinstance(goals, MotionResult):
            # A model with no chain says nothing; a pose out of reach, or none of whose turns lies in the window, has none.
            return None if goals.status is MotionStatus.CONTROLLER_REJECTED else False
        return bool(self._ranked_goals(joints, goals).groups[0])

    def _forget_what_was_judged_ahead(self) -> None:
        """Forget the route a grasp judged ahead, that it kept the arm's branch, and the lift judged at the part: the next
        motion is judged as it runs."""
        self._judged_ahead = None
        self._lift_judged = None
        self._ahead_kept_its_branch = None

    def _refused_for_camera_world(
        self, stamp: CameraWorldStamp, command: MotionCommand, *,
        target_pose: Pose | None = None, target_joints: JointPositions | None = None,
    ) -> MotionResult | None:
        """The ``UNSUPPORTED`` result a motion stamped ``stamp`` is refused with before its body, or ``None``.

        One rule for every verb (``core.camera_world.camera_world_refusal``): MISSING is refused, and
        DECLINED is refused where a live world is wired. Every motion verb asks this before its body, so a
        hand-guided arm is refused here first (:meth:`_hand_guided_refusal`), whatever the stamp says.
        """
        refused = self._hand_guided_refusal(command, target_pose=target_pose, target_joints=target_joints)
        if refused is not None:
            return refused
        message = camera_world_refusal(stamp, live_world_wired=self._live_world is not None)
        if message is None:
            return None
        return MotionResult.failed(
            MotionStatus.UNSUPPORTED, command, target_pose=target_pose, target_joints=target_joints,
            message=message, camera_world=stamp,
        )

    def _move_camera_world(
        self, keyword: Maybe[CameraWorldDecline], *, planned: CameraWorldStamp | None = None
    ) -> CameraWorldStamp:
        """What stands behind a ``move`` on this arm, read off the arm as it is now, never off config.

        ``planned`` is the stamp of the refresh the motion made, which only the motion can know.
        """
        curobo = self._motion_planner == "curobo"
        return resolve_camera_world(
            unplanned=None if curobo else f"robot.ur.motion_planner is {self._motion_planner!r}",
            missing=None if self._live_world is not None else _UR_NO_LIVE_WORLD,
            keyword=keyword,
            block=active_decline(self),
            planned=planned,
        )

    def _planner_refresh(self) -> "WorldRefresh | None":
        """The planner's most recent world refresh, or ``None`` before it made one or for a double.

        Compared by identity before and after a motion: a refresh that is the same object as before
        was made for an earlier motion, and must not vouch for this one.
        """
        from src.robot.safety.planning import live_world as _live

        refresh = getattr(self._curobo_ur, "last_world_refresh", None) if self._curobo_ur is not None else None
        return refresh if isinstance(refresh, _live.WorldRefresh) else None

    def _move_unstamped(
        self,
        pose: Pose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
    ) -> MotionResult:
        """The body of :meth:`move`. It stamps nothing; the public verb does."""
        if pose.frame is not Frame.BASE:
            self._forget_what_was_judged_ahead()
            return MotionResult.failed(
                MotionStatus.INVALID_TARGET,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=f"URRobotArm.move requires Frame.BASE; got {pose.frame!r}",
            )
        if self._motion_planner == "curobo":
            # Dropping `linear` here would make `move(pose, linear=True)` on a cuRobo cell
            # plan a path instead of running the line the caller asked for. It is a
            # different motion: the planner routes around obstacles and the controller
            # does not.
            if linear:
                return self._drive_checked_line(pose, vel=vel, acc=acc)
            return self._drive_curobo(pose, vel=vel, acc=acc)
        self._forget_what_was_judged_ahead()
        # Pre-resolve IK on the driver side, so the preflight pipeline sees
        # ``target_joints`` for every Cartesian command. Joint-limit, IK-quality and the
        # arm-against-arm part of the self-collision guard would otherwise fail closed as
        # unavailable on every Cartesian move.
        target_joints: JointPositions | None = None
        current_joints: JointPositions | None = None
        if self._conn.is_connected:
            try:
                target_joints = self.ik(pose)
            except RobotKinematicsError as exc:
                return MotionResult.failed(
                    MotionStatus.IK_FAILED,
                    MotionCommand.MOVE_TO,
                    target_pose=pose,
                    message=str(exc),
                    exception=exc,
                )
            except RobotConnectionError as exc:
                return MotionResult.failed(
                    MotionStatus.CONNECTION_ERROR,
                    MotionCommand.MOVE_TO,
                    target_pose=pose,
                    message=str(exc),
                    exception=exc,
                )
            try:
                current_joints = self.get_joint_positions()
            except RobotConnectionError:
                # A telemetry hiccup mid-call. The IK-jump check skips rather than
                # failing closed, because ``current_joints`` is None.
                current_joints = None
        try:
            ctx = self._preflight.context_for_pose(
                pose,
                command=MotionCommand.MOVE_TO,
                target_joints=target_joints,
                current_joints=current_joints,
                arm=self,
            )
            decision = self._preflight.evaluate(ctx)
        except RobotConnectionError as exc:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=str(exc),
                exception=exc,
            )
        rejected = SafetyPreflight.as_motion_result(
            decision, MotionCommand.MOVE_TO, target_pose=pose,
        )
        if rejected is not None:
            return rejected
        try:
            # The ungated primitive. The pipeline above has already judged this pose.
            ok = self._drive_pose(
                pose, linear=linear, vel=vel, acc=acc, register=register,
            )
        except RobotConnectionError as exc:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=str(exc),
                exception=exc,
            )
        # Attach the real rejection cause the inner controller classified, such as a
        # genuine near-singularity as IK_QUALITY_REJECTED, rather than the generic
        # CONTROLLER_REJECTED default of from_bool.
        failure_status = (
            self._motion.last_reject_status
            if self._motion.last_reject_status is not None
            else MotionStatus.CONTROLLER_REJECTED
        )
        if not ok and failure_status is MotionStatus.CANCELLED:
            # The halt refused or braked it: said as the halt, with whether it was sent.
            return self._cancelled_by_the_halt(MotionCommand.MOVE_TO, "moveL" if linear else "moveJ",
                                               target_pose=pose)
        return MotionResult.from_bool(
            ok, MotionCommand.MOVE_TO, target_pose=pose, failure_status=failure_status,
        )

    def _drive_curobo(
        self,
        pose: Pose,
        *,
        vel: float | None = None,
        acc: float | None = None,
    ) -> MotionResult:
        """Take the flange to ``pose`` on the route the arm chooses, judge the legs that will run, and run exactly those.

        In order, and nothing moves before the last of the judging has passed (:meth:`_route_to_the_nearest_goal`):

        1. The goal configuration. Every closed-form inverse kinematics solution of the flange goal is turned onto
           the full turns nearest the arm inside the planner's joint window and the joint-limit guard's (the cable
           window about home where the cell keeps one), and ranked: the branch the arm holds first, then the least
           time a ``moveJ`` there takes. Each is screened by the planner at that one configuration and by the endpoint
           gate. cuRobo never chooses a goal configuration.
        2. The straight joint line to each admissible goal, in rank order, judged by the local path gate and by the
           planner against its world, the camera's included, at ``safety.planned_motion.line_clearance_mm``. The
           first that passes runs as it is, one leg.
        3. Only where no line passes: cuRobo plans to the goals in the same order, up to
           ``_NEAREST_GOALS_PLANNED`` of them. A plan is taken where it ends on the configuration asked for and no
           joint swings more than ``safety.planned_motion.max_detour_deg`` beyond its span; its end then meets the
           controller's kinematics and the endpoint gate, and its legs, shortened to the fewest of its own waypoints,
           meet both authorities. A branch other than the arm's is tried only after every goal on it failed, and
           taking one is a warning naming both.

        Exactly the list that passed runs, one ``moveJ`` per waypoint, at the speed clamped to ``motion_limits``, and
        one log line says how it was chosen and how far each joint turns on it.

        A route a grasp judged ahead to this pose (:meth:`grasp_refusal_ahead`) runs instead of a second judgement of the
        same route, where nothing it was judged on changed since (:meth:`_why_judged_again`): the very route the exact
        guard judged sample by sample, in the world the guard and the planner still hold, from where the arm still
        stands. It serves one move, and the refresh it was judged in vouches for it (:meth:`move`). The halt is read
        before every waypoint as on any route (``CuroboUrPlanner.execute``). Anything else, and the route is judged again.
        Where the route judged ahead kept the arm's branch, the route judged again keeps it too: the other branches are
        not tried, and a move that would change branch where the judgement ahead did not is refused before anything is
        sent, ``JOINT_LIMIT_REJECTED`` (the owner's R2, 2026-10-08). Inside :meth:`keeping_its_branch` every route keeps
        it.

        It is fail-closed. An unavailable cuRobo gives ``CONTROLLER_REJECTED``; a goal out of reach ``IK_FAILED``; a
        goal with no configuration inside the window ``JOINT_LIMIT_REJECTED``; a planner that will not start from
        where the arm stands the status of what it found there; a goal every configuration of which the endpoint gate
        refuses, that gate's status; no clear line and only plans past the detour bound ``JOINT_LIMIT_REJECTED``
        naming the joint; no line and no plan ``TIMEOUT`` with the no-plan sentence, the refusal a pick may look again
        at, and the log line naming every goal and why it failed; a guard or the planner refusing a plan's end or legs
        the matching status. There is no blind motion.
        """
        ahead, kept_ahead = self._judged_ahead, self._ahead_kept_its_branch
        self._forget_what_was_judged_ahead()
        if not self._conn.is_connected:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR, MotionCommand.MOVE_TO, target_pose=pose,
                message="cuRobo move requires an open connection.",
            )
        planner = self._curobo_ur_planner()
        # cuRobo plans to tool0, so it gets the flange goal in both ownership modes. Its
        # model has no notion of the controller tool register, because it never talks to
        # the controller at all, and its 2F-85 collision spheres are attached to tool0,
        # so a grasp-centre goal would both drive the flange to the grasp point and place
        # the gripper collision geometry a whole tool length past the target. The pose
        # reported back to the caller stays the TCP pose they asked for.
        goal = self._pose_to_flange(pose)
        # Clamped once, here: the same speed ranks the goals and runs the moveJ.
        vel, acc = self._motion._clamp(vel, acc)
        # Ranking reads the speed, but a speed that is no speed is ur_rtde's to refuse at the moveJ, not a reason to
        # plan differently: one speed for every joint orders the goals the same whatever it is.
        ranked_at = (float(vel) if vel is not None and math.isfinite(float(vel)) and float(vel) > 0.0
                     else float(self.config.motion_limits.max_velocity))
        # A move to a pose is no joint move declared next: whatever was judged ahead for one is not this move's.
        self._drop_the_next_leg()
        route: "_Route | MotionResult | None" = None
        if ahead is not None:
            why = self._why_judged_again(ahead, planner, pose, velocity=ranked_at)
            if not why:
                self.logger.info(
                    "move to %s: the route judged ahead runs as it was judged, %.0f ms ago, in the world of the refresh "
                    "it was judged in (%d box(es)); it is not judged a second time",
                    pose.label or "<unlabeled>", 1000.0 * (time.monotonic() - ahead.at),
                    int(getattr(ahead.refresh, "registered", 0) or 0),
                )
                self._vouched_by = ahead.refresh
                route = ahead.route
            else:
                self.logger.info("move to %s: the route judged ahead is judged again: %s", pose.label or "<unlabeled>",
                                 why)
        for _ in range(self._SEND_JUDGEMENTS):
            if route is None:
                try:
                    route = self._route_to_the_nearest_goal(
                        planner, goal, pose, velocity=ranked_at,
                        held_only=kept_ahead is not None and kept_ahead == _pose_key(pose))
                except CuroboUnavailableError as exc:
                    return MotionResult.failed(
                        MotionStatus.CONTROLLER_REJECTED, MotionCommand.MOVE_TO, target_pose=pose,
                        message=f"cuRobo planner unavailable: {exc}", exception=exc,
                    )
            if isinstance(route, MotionResult):
                return route
            gate = self._send_gate(route.waypoints[0], MotionCommand.MOVE_TO, target_pose=pose)
            if isinstance(gate, MotionResult):
                return gate
            if gate is None:
                break
            route, self._vouched_by = None, None
        else:
            return self._kept_moving(MotionCommand.MOVE_TO, target_pose=pose)
        self._log_the_route(f"move to {pose.label or '<unlabeled>'}", route)
        return planner.execute(self._sent_waypoints(route), pose, vel=vel, acc=acc)

    def _sent_waypoints(self, route: _Route) -> "Sequence[Sequence[float]]":
        """The waypoints of ``route`` sent, one ``moveJ`` each: all of them, and where this arm gates its own sends, whose
        gate found the arm at rest within :data:`_AHEAD_MOVED_MM` of the first, the ones after it (the map's C4): a
        ``moveJ`` to the first would go to where the arm stands, and the first leg then runs from there, within the bound
        a route judged ahead runs from."""
        if self._gates_sends and len(route.waypoints) > 1:
            return route.waypoints[1:]
        return route.waypoints

    #: How long a route judged ahead stays ready for the move to its standoff: the grasp's preamble between the two took
    #: 10 ms on the cell (2026-10-08). A person asked anything in between takes longer, and the route is judged again.
    _AHEAD_HOLDS_S = 0.5
    #: How far the arm may stand from where a route or a lift judged ahead starts for it to run as judged, in the path
    #: gate's own metric (the sum of |dq| * r over the joints): ten times the encoder noise of an arm at rest, about
    #: 0.05 mm.
    _AHEAD_MOVED_MM = 0.5

    def _why_judged_again(self, ahead: _JudgedAhead, planner: object, pose: Pose, *, velocity: float) -> str:
        """Why the route judged ahead to ``pose`` is judged again rather than run as it was judged; ``""`` where it runs.

        It runs only where all of this holds, read now: the same planner holds the world of the refresh it was judged
        in, which no refresh replaced and which vouched for the cell; at most :data:`_AHEAD_HOLDS_S` passed since; no
        world is held but the one held for this very route (:meth:`holding_the_approach`, the world it was judged in);
        this move goes to the same pose, to the last bit, at the speed its goals were ranked at; the arm
        is not halted and no halt came since; it stands within :data:`_AHEAD_MOVED_MM` of where the route starts; and
        what else the judgement read is as it was: the part carried, modelled or not, by the arm and by the planner, the
        camera's boxes the planner and the exact guard hold, the hand (:meth:`_judgement_state`).
        """
        if planner is not ahead.planner:
            return "the planner is another than the one that judged it"
        refresh = getattr(planner, "last_world_refresh", None)
        if refresh is not ahead.refresh or not ahead.refresh.ok:
            return "the planner's world was refreshed since it was judged"
        took_s = time.monotonic() - ahead.at
        if not took_s <= self._AHEAD_HOLDS_S:
            return f"{took_s:.2f} s passed since it was judged, more than {self._AHEAD_HOLDS_S:g} s"
        if self._held_world_reason is not None and self._approach_held_for is not ahead:
            # The world held for this very route (:meth:`holding_the_approach`) is the one it was judged in: no other.
            return f"a world is held ({self._held_world_reason})"
        if _pose_key(pose) != ahead.goal:
            return "this move goes to another pose than the one it was judged to"
        if velocity != ahead.velocity:
            return "this move ranks its goals at another speed than the judgement did"
        if self._halt_record() is not None:
            return "the arm is halted"
        moved = self._moved_since_mm(ahead.route.waypoints[0])
        if not moved <= self._AHEAD_MOVED_MM:
            return (f"the arm stands {moved:.2f} mm from where the route starts, more than {self._AHEAD_MOVED_MM:g} mm"
                    if math.isfinite(moved) else "where the arm stands cannot be compared with where the route starts")
        if self._judgement_state(planner) != ahead.state:
            return "the carried part, the hand, the camera's boxes the planner holds or the halts changed since"
        if self._seen_boxes() is not ahead.boxes:
            return "the exact guard holds other boxes the camera saw than the ones it was judged against"
        return ""

    def _moved_since_mm(self, joints: "Sequence[float]") -> float:
        """How far the arm stands now from ``joints``, the sum of |dq| * r over the joints of the path gate; ``inf`` where
        it cannot be read."""
        radii = self._preflight.joint_radii_mm(self) if self._preflight is not None else None
        try:
            here = [float(v) for v in self._conn.get_joint_positions()]
        except (RobotConnectionError, RuntimeError, OSError):
            return math.inf
        then = [float(v) for v in joints]
        if radii is None or not len(here) == len(then) == len(radii):
            return math.inf
        moved = sum(abs(a - b) * float(r) for a, b, r in zip(here, then, radii))
        return moved if math.isfinite(moved) else math.inf

    def _judgement_state(self, planner: object, *, hand: bool = True) -> "tuple[object, ...]":
        """What a judgement of this arm reads besides its world, its joints and the exact guard's boxes, compared whole:
        the part the arm models and the planner took, whether the jaws closed on one, whether the planner may carry one
        and which camera boxes it holds, the hand where ``hand`` (``core.gripper.why_not_known_open``), and how many halts
        the connection counted."""
        seen = getattr(planner, "perceived_in_world", None)
        return (self._attached_payload, self._payload_in_planner, self._closed_on_part,
                getattr(planner, "carries_part", None), seen() if callable(seen) else None,
                why_not_known_open(self._hand) if hand else None, self._halt_count())

    def _seen_boxes(self) -> "tuple[object, ...]":
        """The boxes the camera saw that the exact guard holds now, the very tuple the last refresh handed it."""
        return tuple(self._preflight.perceived_obstacles(self)) if self._preflight is not None else ()

    def _halt_count(self) -> "int | None":
        """How many times the connection's latch was set, or ``None`` for a connection double that does not count them."""
        count = getattr(self._conn, "halt_requests", None)
        return count if isinstance(count, int) and not isinstance(count, bool) else None

    #: How many configurations of one goal cuRobo is asked to plan to, once no straight line to any of them is clear. A
    #: joint plan that fails costs cuRobo all its attempts, about 7 to 9 s a goal measured on the UR10 descriptor on this
    #: project's RTX 5080 (2026-09-23), so the count is small: a goal no plan reaches pays up to three of those.
    _NEAREST_GOALS_PLANNED = 3
    #: How close, on every joint, a joint plan has to end to the configuration it was asked for to be taken. cuRobo
    #: ends a joint plan on its goal because its optimiser works locally (0.0 rad measured on the probed legs), and
    #: its own documentation does not promise it, so it is read rather than assumed.
    _NEAREST_GOAL_END_TOL_RAD = 1e-4

    def _route_to_the_nearest_goal(
        self, planner: CuroboUrPlanner, goal: Pose, pose: Pose, *, velocity: float, held_only: bool = False,
    ) -> "_Route | MotionResult":
        """The judged route to the configuration of ``goal`` nearest the arm, or the typed refusal of the move.

        ``goal`` is the flange goal and ``pose`` the TCP pose the caller asked for, which every refusal names. Raises
        ``CuroboUnavailableError`` where the planner cannot be reached. ``held_only``, and every route inside
        :meth:`keeping_its_branch`, keeps the arm's branch: the other branches are not tried, and a move nothing on the
        branch runs while one of them was left untried is refused before anything is sent, ``JOINT_LIMIT_REJECTED``
        (:meth:`_refused_for_its_branch`); inside the block a plan also keeps to its tighter overshoot bound.

        cuRobo, handed a pose, picks a configuration for it from random seeds over the whole joint range and does not
        prefer the one nearest the arm: measured on a UR10, a wrist sweep took 8 of 22 legs the long way, and the same
        leg planned twice took different ways. So cuRobo is never handed a pose. Every closed-form solution of the
        flange goal is turned onto the full turns nearest the joints the arm stands at, inside the window both the
        planner and the joint-limit guard admit, and ranked by the branch first and the least time a ``moveJ`` there
        takes after it. The goals on the arm's branch are screened, lined and planned to first; the others only after
        all of those failed. The world is refreshed once, here, and every screen, line and plan is judged in it.

        Which goals there are and their order (:meth:`_goals_in_window`, :meth:`_ranked_goals`) is shared with
        :meth:`nearest_configuration`, so the configuration a caller is told a pose goes to is the one this route
        screens first.
        """
        try:
            here = [float(v) for v in self._conn.get_joint_positions()]
        except (RobotConnectionError, RuntimeError, OSError) as exc:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR, MotionCommand.MOVE_TO, target_pose=pose,
                message=(f"a planned move starts at the current configuration, and the controller's joint "
                         f"positions could not be read ({type(exc).__name__}: {exc}); nothing was sent"),
                exception=exc,
            )
        goals = self._goals_in_window(goal, pose, here, velocity=velocity)
        if isinstance(goals, MotionResult):
            return goals
        ranked = self._ranked_goals(here, goals)
        held, branches, groups, names = ranked.held, ranked.branches, ranked.groups, ranked.names
        kept = held_only or self._keeping is not None
        configured_deg = float(self.config.safety.planned_motion.max_detour_deg)
        bound_deg = configured_deg if self._keeping is None else min(configured_deg, self._keeping.overshoot_deg)
        if kept and not groups[0]:
            # Nothing on the arm's branch lies in the window: no world is refreshed for a move that keeps it.
            refused_for_it = self._refused_for_its_branch(pose, ranked, [], narrowed=0, held_only=held_only)
            if refused_for_it is not None:
                return refused_for_it
        refused = self._refresh_before_the_path_gate(
            near_point_mm=[float(v) for v in goal.position_mm], command=MotionCommand.MOVE_TO,
            target_pose=pose, goal_tcp_mm=self._tcp_of_pose(pose),
        )
        if refused is not None:
            return refused
        tried: list[str] = []
        end_refusals: list[MotionResult] = []
        detours: list[tuple[float, str]] = []
        narrowed = 0
        planned = 0
        for group in (groups[:1] if kept else groups):
            admissible: list[NearestGoal] = []
            for candidate in group:
                if self._mesh_first_refusal(planner, [list(candidate.joints)]) is None:
                    # Mesh first: the endpoint gate below judges the goal on the exact meshes; the spheres are not asked.
                    refused_end = self._gate_planned_config(pose, JointPositions(candidate.joints))
                    if refused_end is not None:
                        tried.append(f"{names[candidate]} refused by the endpoint gate: {refused_end.status.value}")
                        end_refusals.append(refused_end)
                        continue
                    admissible.append(candidate)
                    continue
                verdict = planner.check_joint_path([list(candidate.joints)], refresh=False)
                # A goal only the planner's padded spheres refuse, on pairs the exact guard judges and accepts (the owner,
                # 2026-09-30), or only the boxes the camera saw refuse on its world with a hand known empty and open
                # (Option 1), is the exact guard's to decide; the endpoint gate below judges it again.
                standing = (self._exact_guard_decides(planner, [list(candidate.joints)], verdict, clearance_mm=0.0,
                                                      command=MotionCommand.MOVE_TO)
                            if not verdict.valid else None)
                if standing is not None:
                    tried.append(standing.said(f"{names[candidate]} refused by the planner", verdict, 1)[1]
                                 if standing.row is not None else
                                 _with_refusal(f"{names[candidate]} refused by the planner",
                                               getattr(verdict, "refusal", None)))
                    continue
                refused_end = self._gate_planned_config(pose, JointPositions(candidate.joints))
                if refused_end is not None:
                    tried.append(f"{names[candidate]} refused by the endpoint gate: {refused_end.status.value}")
                    end_refusals.append(refused_end)
                    continue
                admissible.append(candidate)
            for candidate in admissible:
                line = [list(here), list(candidate.joints)]
                refused = self._judge_legs(planner, line, command=MotionCommand.MOVE_TO, target_pose=pose,
                                           clearance_mm=self._line_clearance_mm())
                if refused is None:
                    return self._route(line, planned=False, how=f"direct line to {names[candidate]}", tried=tried,
                                       held=held, reached=branches[candidate])
                if isinstance(refused.exception, CuroboUnavailableError):
                    return refused  # no line and no plan is judged by a planner that cannot be reached
                tried.append(f"the straight line to {names[candidate]} refused ({refused.status.value}: "
                             f"{refused.message})")
            for candidate in admissible:
                if planned >= self._NEAREST_GOALS_PLANNED:
                    break
                planned += 1
                try:
                    trajectory, why, start, legs = self._planned(planner, candidate.joints, name=names[candidate],
                                                                 here=here, command=MotionCommand.MOVE_TO,
                                                                 target_pose=pose)
                except CuroboUnavailableError:
                    raise  # a RuntimeError too, and the caller's to type as the planner lost
                except (RobotConnectionError, RuntimeError, OSError) as exc:
                    # The plan reads where the arm stands again, and a transport that fails there raises what the read
                    # at the start of the route turns into a typed result.
                    return MotionResult.failed(
                        MotionStatus.CONNECTION_ERROR, MotionCommand.MOVE_TO, target_pose=pose,
                        message=(f"the controller's joint positions could not be read to plan to {names[candidate]} "
                                 f"({type(exc).__name__}: {exc}); nothing was sent"),
                        exception=exc,
                    )
                if start is not None:
                    return self._start_refusal(start, MotionCommand.MOVE_TO, target_pose=pose, note=why)
                if trajectory is None:
                    tried.append(why)
                    continue
                over_deg, detour = self._detour(trajectory, cap_deg=bound_deg)
                if detour:
                    said = f"cuRobo's plan to {names[candidate]} {detour}"
                    tried.append(f"{said}, so it is not taken")
                    detours.append((over_deg, said))
                    # Past the kept branch's bound alone: outside the block the plan would have been taken.
                    narrowed += int(over_deg <= configured_deg)
                    continue
                # The plan ends on its goal, or nothing moves. The end check reads where cuRobo's own last
                # configuration puts the flange on the controller's kinematics, which a planner answering in another
                # base misses by a metre; the endpoint gate judges that configuration with the commanded pose in hand,
                # so the box applies as well as the joint limits, self-collision and payload. The continuity guards
                # stay skipped, as in the sim: cuRobo owns a plan's continuity, and a wrap from +pi to -pi is not a
                # blind interpolator teleporting. Cheap, and the plan's legs are judged after.
                # Two statements, not ``a or b``: a refusal is a MotionResult, which is falsy when it is not ok, so
                # ``or`` dropped the end check's refusal whenever the endpoint gate passed.
                refused = self._plan_end_refusal(goal, trajectory[-1], pose)
                if refused is None:
                    refused = self._gate_planned_config(pose, JointPositions(trajectory[-1]))
                if refused is not None:
                    return refused
                waypoints, refused = self._judged_waypoints(
                    planner, trajectory, command=MotionCommand.MOVE_TO, target_pose=pose, legs=legs,
                )
                if refused is not None:
                    return refused
                return self._route(waypoints, planned=True, dense=len(trajectory),
                                   how=f"cuRobo plan to {names[candidate]}" + (f", {why}" if why else ""),
                                   tried=tried, held=held, reached=branches[candidate])
        self.logger.error("move to %s refused, nothing was sent: %s", pose.label or "<unlabeled>",
                          "; ".join(tried) or "no goal was admissible")
        if kept:
            refused_for_it = self._refused_for_its_branch(pose, ranked, tried, narrowed=narrowed, held_only=held_only)
            if refused_for_it is not None:
                return refused_for_it
        if len(end_refusals) == len(goals):
            # Every configuration met the endpoint gate and none passed it: the end is the gate's to name, as a plan's
            # end always was, and there was nothing a planner could have been asked.
            first = end_refusals[0]
            return dataclasses.replace(first, message=(
                f"no configuration of this goal passes the endpoint gate; the nearest was refused: {first.message}"))
        if detours:
            # cuRobo found a way, and the only way it found swings a joint around: a refusal that names the joint, not
            # the no-plan sentence a pick answers by looking again at a scene that is not the problem.
            return MotionResult.failed(
                MotionStatus.JOINT_LIMIT_REJECTED, MotionCommand.MOVE_TO, target_pose=pose,
                message=(f"no straight line to this goal is clear, and no plan cuRobo found keeps every joint within "
                         f"the detour bound; the furthest: {max(detours)[1]}; nothing was sent"),
            )
        return MotionResult.failed(
            MotionStatus.TIMEOUT, MotionCommand.MOVE_TO, target_pose=pose, message=NO_PLAN_FAIL_SAFE_MESSAGE,
        )

    def _refused_for_its_branch(
        self, pose: Pose, ranked: _RankedGoals, tried: "Sequence[str]", *, narrowed: int, held_only: bool,
    ) -> "MotionResult | None":
        """The refusal of a move that keeps the arm's branch and that nothing on that branch runs, where the rules outside
        would have gone on: a configuration on another branch was left untried, or ``narrowed`` plans were refused past
        the bound of :meth:`keeping_its_branch` alone. ``None`` where neither, and the move's own refusal stands.

        ``JOINT_LIMIT_REJECTED``, nothing sent, counted on the block running (``refused``), which a pick reads to try the
        grasp again once every grasp that keeps the branch failed. ``held_only`` outside a block is the route a grasp
        judged ahead on the arm's branch, judged again by the move to its standoff (:meth:`_drive_curobo`).
        """
        others = len(ranked.groups[1])
        if not others and not narrowed:
            return None
        keeping = self._keeping
        if keeping is not None:
            keeping.refused += 1
        held = ranked.held.render() if ranked.held is not None else "unknown"
        why = "; ".join(tried) or "no configuration of this goal on it lies inside the joint window"
        left = [f"{others} configuration(s) on another branch"] if others else []
        if narrowed and keeping is not None:
            left.append(f"{narrowed} plan(s) swinging a joint more than the {keeping.overshoot_deg:g} deg a kept "
                        "branch allows")
        if keeping is not None or not held_only:  # a block keeps it; outside one only a route judged ahead does
            message = (f"this move is {_KEEPS_ITS_BRANCH} ({held}), and nothing on it runs ({why}): "
                       f"{' and '.join(left)} were left untried, a change of branch being the last resort; nothing "
                       "was sent")
        else:
            message = (f"the route judged ahead to this pose was {_KEEPS_ITS_BRANCH} ({held}), and judged again here "
                       f"nothing on that branch runs ({why}): a change of branch nobody judged ahead is not taken, "
                       f"{' and '.join(left)} were left untried, and nothing was sent")
        self.logger.warning("move to %s refused: %s", pose.label or "<unlabeled>", message)
        return MotionResult.failed(MotionStatus.JOINT_LIMIT_REJECTED, MotionCommand.MOVE_TO, target_pose=pose,
                                   message=message)

    def _goals_in_window(
        self, goal: Pose, pose: Pose, here: "Sequence[float]", *, velocity: float,
    ) -> "tuple[NearestGoal, ...] | MotionResult":
        """Every configuration of the flange ``goal`` inside the goal window, quickest first; or the typed refusal.

        The first half of the screening a Cartesian cuRobo move and :meth:`nearest_configuration` share. ``goal`` is
        the flange goal, ``pose`` the TCP pose every refusal names, ``here`` the joints the arm stands at. Every
        closed-form solution is turned onto the full turns nearest ``here`` inside :meth:`_goal_joint_window`, the
        planner's window narrowed to the joint-limit guard's (the cable window about home where the cell keeps one),
        and ordered by the least time a ``moveJ`` there takes (:func:`~src.robot.safety._ur_ik.nearest_goals`). It asks
        nothing of the controller, the planner or the world.

        The refusals: a model with no DH chain, or a controller that reported another number of joints,
        CONTROLLER_REJECTED; a pose out of reach IK_FAILED; no solution with every joint inside the window
        JOINT_LIMIT_REJECTED, naming the half turn about home where the guard keeps it.
        """
        model = str(self.config.ur.model)
        solutions = (ur_flange_ik(model, np.asarray(goal.to_matrix(), dtype=np.float64), q6_if_singular=here[-1])
                     if len(here) == self.capabilities.dof else None)
        if solutions is None:
            return MotionResult.failed(
                MotionStatus.CONTROLLER_REJECTED, MotionCommand.MOVE_TO, target_pose=pose,
                message=(f"robot.ur.model {model!r} has no DH chain to solve this goal on, or the controller reported "
                         f"{len(here)} joints, so no configuration of it can be chosen; cuRobo is never left to "
                         "choose one, and nothing was sent"),
            )
        if not solutions:
            return MotionResult.failed(
                MotionStatus.IK_FAILED, MotionCommand.MOVE_TO, target_pose=pose,
                message="the flange goal of this pose is out of the arm's reach: it has no inverse kinematics solution",
            )
        lower, upper = self._goal_joint_window()
        goals = nearest_goals(solutions, current=here, lower=lower, upper=upper, velocity=velocity)
        if not goals:
            half_turn = any(bool(getattr(g, "within_half_turn_of_home", False))
                            for g in (getattr(self._preflight, "guards", ()) or ()))
            return MotionResult.failed(
                MotionStatus.JOINT_LIMIT_REJECTED, MotionCommand.MOVE_TO, target_pose=pose,
                message=(f"none of the {len(solutions)} inverse kinematics solution(s) of this goal turns inside the "
                         "planner's and the joint-limit guard's joint window"
                         + (", which keeps half a turn either side of home "
                            "(safety.joint_limits.within_half_turn_of_home)" if half_turn else "")
                         + "; cuRobo is never left to choose a configuration outside it, and nothing was sent"),
            )
        return goals

    def _ranked_goals(self, here: "Sequence[float]", goals: "Sequence[NearestGoal]") -> _RankedGoals:
        """``goals`` in the order the arm takes them, the branch it holds first, with the names a log line gives them.

        The second half of the shared screening, read on the DH chain alone: the branch of ``here`` and of every goal
        (:func:`~src.robot.safety._ur_ik.ur_branch`), and each group kept in the order ``goals`` came in.
        """
        model = str(self.config.ur.model)
        held = ur_branch(model, here)
        branches = {g: ur_branch(model, g.joints) for g in goals}
        # The arm's own branch first, in the order nearest_goals gave; a configuration at a branch point agrees with
        # both sides of it, so out of a candle-straight home, all three branch points at once, every goal is on it.
        on_it = [g for g in goals if _same_branch(held, branches[g])]
        groups = (on_it, [g for g in goals if g not in on_it])
        names = {g: (f"goal {rank} of {len(goals)} ({_branch_text(branches[g])}, "
                     f"{math.degrees(g.largest_rad):.1f} deg at most)")
                 for rank, g in enumerate(groups[0] + groups[1], start=1)}
        return _RankedGoals(held=held, groups=groups, branches=branches, names=names)

    def _planned(
        self, planner: CuroboUrPlanner, goal: "Sequence[float]", *, name: str,
        here: "Sequence[float] | None" = None, command: MotionCommand = MotionCommand.MOVE_TO,
        target_pose: Pose | None = None, target_joints: JointPositions | None = None,
    ) -> "tuple[list[list[float]] | None, str, object | None, tuple[int, int]]":
        """cuRobo's plan to ``goal``, as ``(trajectory, how, None, legs)``, or ``(None, why, None, _)`` where it is not
        taken.

        ``(None, why, refusal, _)`` is a start the planner will not leave, which no other goal plans from either;
        ``why`` says what the two authorities found there where it can. A plan is taken where it ends within
        ``_NEAREST_GOAL_END_TOL_RAD`` of ``goal``; how far it swings is the caller's to read (:meth:`_detour`). Only the
        goal is ever chosen here; cuRobo's trajectory is never altered.

        ``here`` is where the arm stands. Given it, a start or goal the planner refuses only on self pairs the exact
        guard judges and accepts, the planner's cushion band, is planned out of or into by a straight leg of at most
        :data:`~src.robot.safety.planning.band.BAND_LEG_MAX_DEG` per joint (:meth:`_band_route`): the trajectory then
        opens at ``here`` and ends at ``goal`` with those legs, ``how`` says so, ``legs`` counts them at its head and its
        tail, for the caller to judge the trajectory whole as every plan and to keep them as they are when it shortens
        it (:meth:`_judged_waypoints`). Once a leg left the start, a plan that still fails is this goal's failure, never
        the start's. Raises ``CuroboUnavailableError``.
        """
        trajectory = planner.plan_joint(list(goal), refresh=False)
        how, legs = "", (0, 0)
        if not trajectory:
            refusal = getattr(planner, "last_refusal", None)
            banded = (self._band_route(planner, here, goal, refusal, command=command, target_pose=target_pose,
                                       target_joints=target_joints)
                      if here is not None else _BandRoute())
            if banded.route is None:
                if banded.stuck and refusal is not None and getattr(refusal, "where", None) == StateWhere.START:
                    return None, banded.note, refusal, legs
                if banded.note and not banded.stuck:
                    return None, f"{name} did not plan: {banded.note}", None, legs
                return (None, _with_refusal(f"{name} did not plan", refusal)
                        + (f"; {banded.note}" if banded.note else ""), None, legs)
            trajectory, how, legs = banded.route, banded.note, banded.legs
        off_rad = max(abs(float(a) - float(b)) for a, b in zip(trajectory[-1], goal))
        if off_rad > self._NEAREST_GOAL_END_TOL_RAD:
            return (None, f"{name} planned to a configuration {off_rad:.2g} rad from it, so the plan is not taken", None,
                    (0, 0))
        return trajectory, how, None, legs

    def _exact_guard_decides(
        self, planner: CuroboUrPlanner, configs: "Sequence[Sequence[float]]", verdict: object, *,
        clearance_mm: float, command: MotionCommand,
    ) -> "_Standing | None":
        """``None`` where every configuration of ``configs`` the planner refused is the exact guard's to decide; else why
        the planner's refusal stands, and on which sample where its report says (:class:`_Standing`).

        Two of the owner's decisions (:mod:`src.robot.safety.planning.band`). For the self pairs the exact guard judges,
        that guard decides (2026-09-30). And where only the boxes the camera saw refuse a sample on the planner's world,
        that guard decides them too: it holds the very same boxes, turned, at ``perceived_min_distance_mm`` (Option 1,
        after the guard fixes). ``configs`` are the samples ``verdict`` judged, in UR order, at ``clearance_mm`` from the
        planner's world. A refusal is left to the exact guard only where all of these hold, and anything unexpected
        leaves it standing:

        1. the planner's refusal is a self collision or its world, as it typed it, and its report agrees;
        2. this cell's self-collision guard runs its exact mesh backend (``SafetyPreflight.exact_pairs``);
        3. the planner reports every sample it refuses, with the three terms apart and every pair of links, and the
           report opens on the sample the verdict named (``CuroboUrPlanner.judge_joint_path``);
        4. no refused sample has a joint outside the planner's bounds, and every self collision is on pairs the exact
           guard judges, padded by no more than ``planner_margin_mm``, with no carried part among them
           (``band.admission_refusal``);
        5. where the planner's world refused samples: no part is carried, by the arm (modelled or not) or the planner,
           and the hand reads empty and open (:meth:`set_hand`, the owner, 2026-10-01); the exact guard holds boxes the
           camera saw; and the planner, asked again about exactly those samples with
           exactly those boxes set aside, clears its world and its bounds for every one and finds the robot itself as it
           did (:meth:`_world_aside_refusal`, :meth:`_world_set_aside`, ``band.world_admission_refusal``). The bench,
           the declared fixtures and meshes and the camera's distance field stay the planner's;
        6. the exact guard accepts every refused sample, judged here (``gate_joint_path``) at its own distances, the
           camera's turned boxes at ``perceived_min_distance_mm``, whatever the caller judged before.
        """
        refusal = getattr(verdict, "refusal", None)
        kind = getattr(refusal, "kind", None)
        if refusal is None or kind not in (StateRefusalKind.SELF_COLLISION, StateRefusalKind.WORLD):
            return _Standing("the planner's refusal is neither a self collision nor its world, as it typed it")
        exact = self._preflight.exact_pairs(self) if self._preflight is not None else None
        judge = getattr(planner, "judge_joint_path", None)
        if exact is None or not callable(judge):
            return _Standing("no exact mesh guard runs here, or the planner reports no refused samples")
        try:
            report = judge([list(c) for c in configs], clearance_mm=float(clearance_mm))
        except CuroboUnavailableError as exc:
            self.logger.warning("the planner's refusal stands: its report could not be read (%s)", exc)
            return _Standing(f"the planner's report could not be read ({exc})")
        refused = tuple(getattr(report, "refused", ()) or ())
        opens = refused[0].index if refused else None
        if (getattr(report, "checked", None) != len(configs) or opens != getattr(verdict, "first_invalid", None)
                or any(not 0 <= row.index < len(configs) for row in refused)):
            why = "the planner's report does not account for the samples its verdict judged"
            self.logger.warning("the planner's refusal stands: %s", why)
            return _Standing(why)
        if (refused[0].self_ok if kind == StateRefusalKind.SELF_COLLISION else refused[0].world_ok):
            why = f"the planner's verdict and its report disagree on what it found at sample {refused[0].index}"
            self.logger.warning("the planner's refusal stands: %s", why)
            return _Standing(why)
        in_world: list[RefusedSample] = []
        for row in refused:
            stands = admission_refusal(row, exact, margin_mm=self._planner_margin_mm(), world_set_aside=True)
            if stands is None and not row.world_ok:
                if not in_world:
                    # Asked once, at the first sample the planner's world refused: whether it can be set aside at all.
                    stands = self._world_aside_refusal(planner)
                in_world.append(row)
            if stands is not None:
                self.logger.info("the planner's refusal stands on sample %d of %d: %s", row.index, len(configs),
                                 stands)
                return _Standing(stands, row)
        if in_world:
            standing = self._world_set_aside(planner, configs, in_world, clearance_mm=clearance_mm)
            if standing is not None:
                return standing
        assert self._preflight is not None  # noqa: S101 (exact_pairs above came from it)
        judged = PathSamples(configs=tuple(tuple(float(v) for v in configs[row.index]) for row in refused),
                             step_bound_mm=float(self._preflight.path_step_mm or 0.0))
        denied = self._preflight.gate_joint_path(judged, arm=self, command=command)
        if denied is not None:
            # Which sample: each judged alone by the same guards, in path order, the first refused, in its own words.
            for row in refused:
                alone = self._exact_guard_refusal(configs[row.index], command)
                if alone is not None:
                    return _Standing(f"the exact guard refuses it too: {alone.message}", row, alone.status)
            return _Standing(f"the exact guard refuses a sample the planner refused: {denied.message}",
                             status=denied.status)
        found: list[str] = []
        on_self = [row for row in refused if not row.self_ok]
        if on_self:
            deepest = max(on_self, key=lambda row: max((pair.depth_mm for pair in row.pairs or ()), default=0.0))
            found.append(f"{len(on_self)} on pairs the exact guard judges, the deepest "
                         f"{band_sentence(deepest, exact, configs[deepest.index])}")
        if in_world:
            seen = self._preflight.perceived_obstacles(self)
            found.append(f"{len(in_world)} on its world, where only the {len(seen)} box(es) the camera saw refused them, "
                         f"which the exact guard keeps "
                         f"{float(self.config.safety.self_collision.perceived_min_distance_mm):g} mm from")
        self.logger.info(
            "the planner refuses %d of %d sample(s), %s; the exact guard accepts every one, so it decides them",
            len(refused), len(configs), "; and ".join(found),
        )
        return None

    def _world_aside_refusal(self, planner: object, *, hand: bool = True) -> "str | None":
        """Why the planner's world cannot be judged again with the boxes the camera saw set aside, or ``None``.

        Never where the exact guard holds no box the camera saw: what the planner's world refused is then the bench or
        the declared geometry, which stay the planner's. Never while a part the planner models is carried, by the arm
        (:meth:`attach_payload` handed it one since the last :meth:`detach_payload`) or by the planner, nor where the
        planner cannot say it holds none: the carried part is the planner's alone, and with the camera's boxes set
        aside nobody would judge it against them. And never unless the hand this arm carries reads empty and open
        (:meth:`set_hand`, ``core.gripper.why_not_known_open``): a toggle whose count stands and says open, a gripper
        measured fully open, whatever verb closed the jaws last (the owner, 2026-10-01). ``hand`` false leaves the hand
        unread, for the pose screen, which judges a pose for a hand known empty and open. A grasp asks whether its lift
        would run carrying before it closes (:meth:`carried_line_refusal`), so it does not close where only these boxes
        let the empty hand stand; an arm that holds a part there anyway leaves once a person opens the jaws, and the
        sentence says so.

        A cell that models no carried part (``planning_world.payload``) judges jaws closed on one as an empty hand is
        judged, the boxes set aside and the hand left unread: nobody judges such a part against the camera's boxes
        either way, and the exact guard still judges the arm and the hand (the owner, 2026-10-05: "mehr Griffe").
        """
        seen = self._preflight.perceived_obstacles(self) if self._preflight is not None else ()
        if not seen:
            return ("it reaches into the planner's world, and the exact guard holds no box the camera saw: the bench and "
                    "the declared fixtures and meshes stay the planner's")
        names = [str(box.name) for box in seen]
        if len(set(names)) != len(names) or not all(
                name.startswith(PERCEIVED_PREFIX) and len(name) > len(PERCEIVED_PREFIX) for name in names):
            return (f"the exact guard holds boxes {sorted(names)} the camera world did not name as its own, so none is "
                    "set aside and the planner's world stands")
        carried = getattr(planner, "carries_part", None)
        # A cell that models no carried part judges a closed hand as an empty one (the owner, 2026-10-05): the exact guard
        # holds the hand at full stroke, the widest it stands, and a part nobody models is judged by nobody either way.
        unmodelled = self._attached_payload is None and self._payload_config_declined() is not None
        if self._attached_payload is not None or (self._closed_on_part and not unmodelled) or carried is True:
            return ("a part is carried, and the carried part is the planner's alone, so the boxes the camera saw are not "
                    "set aside and the planner's world stands. An arm that holds a part where only those boxes refuse "
                    "the planner leaves once a person opens the jaws on it (Robot.release, which forgets the part), "
                    "empty-handed")
        if carried is not False:
            return ("the planner cannot say whether it holds a carried part, so the boxes the camera saw are not set "
                    "aside and the planner's world stands")
        why = why_not_known_open(self._hand) if hand and not unmodelled else ""
        if why:
            return (f"the hand is not known to be empty and open ({why}), so the boxes the camera saw are not set aside "
                    "and the planner's world stands")
        return None

    def _mesh_first_refusal(self, planner: object, configs: "Sequence[Sequence[float]]") -> "str | None":
        """Why the exact guard may not judge these samples alone (``safety.planned_motion.mesh_first``), or ``None``.

        ``None`` where the switch is on, an exact mesh guard runs, and it holds everything the planner would judge the
        samples against. The arm, the hand, the wrist cameras and the declared fixtures both hold; the camera's boxes
        the guard holds from the same refresh the planner's world was built from. What only the planner holds keeps it
        asked: a mesh its world declares, a distance field of the camera's points (``perceived.voxel_field_mm``), a
        carried part (attached by the arm, the jaws closed on one where the cell models it, or a planner that carries
        one or cannot say), on a cell that models a carried part a hand not known empty and open, which may hold one
        (the owner, 2026-10-01), the declared support plane, wherever a part of the arm or the hand past the
        shoulder comes within the guard's ``min_distance_mm`` of its top on the exact meshes, and the robot's own base,
        which no guard part holds and the planner's shoulder_link stands for (``planning.band``), wherever such a part
        comes within that distance of it, and on an arm whose base the guard does not know. The caller has run the
        exact guard on the samples already.
        """
        motion = getattr(self.config.safety, "planned_motion", None)
        if motion is None or not bool(getattr(motion, "mesh_first", False)):
            return "safety.planned_motion.mesh_first is off"
        # A preflight that offers no exact pairs runs no exact guard (a double built for the planner's verdict alone).
        exact = getattr(self._preflight, "exact_pairs", None) if self._preflight is not None else None
        pairs = exact(self) if callable(exact) else None
        if pairs is None:
            return "no exact mesh guard runs on this arm"
        world = getattr(self.config.safety, "planning_world", None)
        on = world is not None and bool(getattr(world, "enabled", False))
        if on and tuple(getattr(world, "meshes", None) or ()):
            return "the planner's world declares meshes the exact guard does not hold"
        perceived = getattr(world, "perceived", None) if on else None
        if perceived is not None and float(getattr(perceived, "voxel_field_mm", 0.0) or 0.0) > 0.0:
            return ("the planner holds a distance field of the camera's points (perceived.voxel_field_mm), which the "
                    "exact guard does not")
        carried = getattr(planner, "carries_part", None)
        unmodelled = self._attached_payload is None and self._payload_config_declined() is not None
        if self._attached_payload is not None or (self._closed_on_part and not unmodelled) or carried is True:
            return "a part is carried, and the carried part is the planner's alone"
        if carried is not False:
            return "the planner cannot say whether it carries a part"
        # The owner, 2026-10-01: on a cell that models a carried part, jaws nobody knows empty and open may hold one.
        why = why_not_known_open(self._hand) if not unmodelled else ""
        if why:
            return f"the hand is not known to be empty and open ({why})"
        plane = getattr(world, "support_plane", None) if on else None
        if plane is not None and configs:
            if pairs.lowest is None:
                return ("the exact guard cannot say how low the arm and the hand reach, and only the planner holds the "
                        "declared support plane")
            top = float(getattr(plane, "height_mm", 0.0) or 0.0)
            below = top + float(pairs.min_distance_mm)
            # Each configuration in turn; where the guard judges paths whole (whole_path_judge), only those it may
            # refuse, the first of them found over all of them at once (ExactPairs.first_low), and the same sentence.
            index = pairs.first_low(configs, below)
            while index < len(configs):
                low = pairs.lowest(configs[index])
                if low is None:
                    return "the exact guard cannot place the arm over the declared support plane, which only the planner holds"
                if low < below:
                    return (f"a part of the arm or the hand comes {low - top:.1f} mm over the declared support plane, "
                            f"within the guard's {float(pairs.min_distance_mm):g} mm, and only the planner holds that "
                            "plane")
                index = pairs.first_low(configs, below, start=index + 1)
        # The robot's own base: no guard part holds it, the planner's shoulder_link stands for it (review of F1,
        # 2026-09-30, the Hand-E admitted touching it), so a line that brings a part near it stays the planner's too.
        if pairs.base is None:
            return "the exact guard knows no base for this arm, and only the planner's shoulder_link stands for it"
        index = pairs.first_near_base(configs, float(pairs.min_distance_mm))
        while index < len(configs):
            near = pairs.base(configs[index])
            if near is None:
                return "the exact guard cannot place the arm beside its base, which only the planner holds"
            part, gap = near
            if gap < float(pairs.min_distance_mm):
                return (f"{part} comes {gap:.1f} mm from the robot's base, within the guard's "
                        f"{float(pairs.min_distance_mm):g} mm, and only the planner holds the base")
            index = pairs.first_near_base(configs, float(pairs.min_distance_mm), start=index + 1)
        return None

    def _world_set_aside(
        self, planner: CuroboUrPlanner, configs: "Sequence[Sequence[float]]", rows: "Sequence[RefusedSample]", *,
        clearance_mm: float,
    ) -> "_Standing | None":
        """``None`` where the planner, asked again about ``rows`` with every box the camera saw set aside, finds only those
        boxes refused them on its world; else why its world stands, and on which sample (the owner's Option 1).

        The boxes are the ones the exact guard holds now (``SafetyPreflight.perceived_obstacles``), by name; the glue asks
        with exactly those, where its last refresh handed the planner the same and none is a declared fixture's
        (``CuroboUrPlanner.judge_joint_path``). The second judgement is of exactly the samples the world refused, at the
        clearance they were refused at, and it has to account for every one of them, say it set aside exactly those
        boxes, and find the bounds and the robot itself as the first did (``band.world_admission_refusal``). The exact
        guard judges every refused sample after this, with the same boxes.
        """
        assert self._preflight is not None  # noqa: S101 (the caller found the exact guard running)
        names = tuple(sorted(str(box.name) for box in self._preflight.perceived_obstacles(self)))
        asked = [[float(v) for v in configs[row.index]] for row in rows]
        judge = getattr(planner, "judge_joint_path")
        try:
            again = judge(asked, clearance_mm=float(clearance_mm), name_pairs=False, ignore_perceived=names)
        except CuroboUnavailableError as exc:
            why = (f"the planner could not judge it again with the boxes the camera saw set aside ({exc}), so the "
                   "planner's world stands")
            self.logger.warning("the planner's refusal stands: %s", why)
            return _Standing(why, rows[0])
        second = tuple(getattr(again, "refused", ()) or ())
        if (getattr(again, "checked", None) != len(asked) or len({row.index for row in second}) != len(second)
                or any(not 0 <= row.index < len(asked) for row in second)):
            why = ("the planner's second judgement, with the boxes the camera saw set aside, does not account for the "
                   "samples it was asked about, so the planner's world stands")
            self.logger.warning("the planner's refusal stands: %s", why)
            return _Standing(why, rows[0])
        aside = tuple(getattr(again, "perceived_ignored", ()) or ())
        if aside != names:
            why = (f"the planner's second judgement set aside other boxes ({list(aside)}) than the ones the camera saw "
                   f"({list(names)}), so the planner's world stands")
            self.logger.warning("the planner's refusal stands: %s", why)
            return _Standing(why, rows[0])
        by_position = {row.index: row for row in second}
        for position, row in enumerate(rows):
            stands = world_admission_refusal(row, by_position.get(position))
            if stands is not None:
                self.logger.info("the planner's refusal stands on sample %d of %d: %s", row.index, len(configs),
                                 stands)
                return _Standing(stands, row)
        return None

    def _exact_guard_accepts(self, config: "Sequence[float]", command: MotionCommand) -> bool:
        """Whether the exact guard, with the joint-limit and payload guards, accepts one configuration: nothing moves."""
        return self._preflight is not None and self._exact_guard_refusal(config, command) is None

    def _exact_guard_refusal(self, config: "Sequence[float]", command: MotionCommand) -> "MotionResult | None":
        """The exact guard's refusal of one configuration, with the joint-limit and payload guards, or ``None``."""
        if self._preflight is None:
            return MotionResult.failed(MotionStatus.UNSUPPORTED, command, message="no safety preflight is wired")
        judged = PathSamples(configs=(tuple(float(v) for v in config),),
                             step_bound_mm=float(self._preflight.path_step_mm or 0.0))
        return self._preflight.gate_joint_path(judged, arm=self, command=command)

    #: How many of the nearest clear configurations a band start or goal tries a straight leg to, before it says none
    #: is reached.
    _BAND_LEG_TRIES = 3

    def _band_route(
        self, planner: CuroboUrPlanner, here: "Sequence[float]", goal: "Sequence[float]", refusal: object, *,
        command: MotionCommand, target_pose: Pose | None = None, target_joints: JointPositions | None = None,
    ) -> _BandRoute:
        """A plan out of or into the planner's cushion band, or why none is taken (:class:`_BandRoute`).

        Asked only where cuRobo refused the start or the goal of a joint plan as a self collision. cuRobo cannot plan
        out of or into a configuration its padded spheres refuse, so where the exact guard decides that refusal
        (:meth:`_exact_guard_decides`'s conditions, on that one configuration), the arm takes a straight leg of at most
        :data:`~src.robot.safety.planning.band.BAND_LEG_MAX_DEG` per joint to the nearest configuration both authorities
        clear (:meth:`_band_leg`), and cuRobo plans between clear configurations (``plan_joint`` with the escape as its
        start). The route is ``[here, <cuRobo's plan>, goal]``, ``here`` and ``goal`` only where a leg leaves or reaches
        them; every leg of it is judged again by the caller as every plan. An empty note where the planner's refusal was
        not the band's to answer; else the note names the pair, the planner's depth and the distance the exact meshes
        keep, and what came of it.

        The start is judged first. It is ``stuck`` where the planner refuses it for more than the band or no leg leaves
        it: no goal plans from there, and the caller says so as the start's refusal. Once a leg leaves it, or it was
        clear, whatever fails after (the goal refused for more than the band, no leg into it, no plan between the clear
        configurations) is this goal's alone, and the caller tries the next goal as after any goal that did not plan.
        """
        if (refusal is None or getattr(refusal, "kind", None) != StateRefusalKind.SELF_COLLISION
                or getattr(refusal, "where", None) not in (StateWhere.START, StateWhere.GOAL)):
            return _BandRoute()
        exact = self._preflight.exact_pairs(self) if self._preflight is not None else None
        judge = getattr(planner, "judge_joint_path", None)
        if exact is None or not callable(judge):
            return _BandRoute()
        ends = [[float(v) for v in here], [float(v) for v in goal]]
        try:
            report = judge(ends, clearance_mm=0.0)
        except CuroboUnavailableError as exc:
            return _BandRoute(note=f"the planner's report on the start and the goal could not be read ({exc})")
        rows = {row.index: row for row in getattr(report, "refused", ())}
        if (0 if getattr(refusal, "where", None) == StateWhere.START else 1) not in rows:
            # The report does not hold the end the planner refused: its typed refusal stands as it was typed.
            return _BandRoute()
        legs: dict[int, list[float]] = {}
        said: list[str] = []

        def failed(sentence: str, *, stuck: bool) -> _BandRoute:
            return _BandRoute(note="; ".join([*said, sentence]), stuck=stuck)

        for index, what in ((0, "start"), (1, "goal")):
            row = rows.get(index)
            if row is None:
                continue
            why = admission_refusal(row, exact, margin_mm=self._planner_margin_mm())
            if why is None and not self._exact_guard_accepts(ends[index], command):
                why = "the exact guard refuses it too"
            if why is not None:
                return failed(f"the planner refuses the {what}, and not only where the exact guard decides: {why}",
                              stuck=index == 0)
            found = band_sentence(row, exact, ends[index])
            leg = self._band_leg(planner, exact, ends[index], row, escape=index == 0, command=command,
                                 target_pose=target_pose, target_joints=target_joints)
            if leg is None:
                return failed(
                    f"the {what} is in the planner's cushion band: {found}; no configuration within "
                    f"{BAND_LEG_MAX_DEG:g} deg per joint of it is clear to both authorities with a straight leg "
                    + ("to it both pass, so no plan leaves it; move the arm by hand, or teach the pose where both "
                       "clear it" if index == 0 else
                       "from it both pass, so no plan reaches it; reach it on a straight line, or teach it where both "
                       "clear it"), stuck=index == 0)
            legs[index] = leg
            turn = max(abs(math.degrees(a - b)) for a, b in zip(leg, ends[index]))
            said.append(f"{'out of' if index == 0 else 'into'} the planner's cushion band at the {what} by a straight "
                        f"{turn:.1f} deg {'escape' if index == 0 else 'approach'} leg ({found})")
        if not legs:
            return _BandRoute()
        start, end = legs.get(0, ends[0]), legs.get(1, ends[1])
        plan = planner.plan_joint(end, start_ur=start, refresh=False)
        if not plan:
            return failed(_with_refusal("cuRobo found no plan between the clear configurations the band legs reach",
                                        getattr(planner, "last_refusal", None)), stuck=False)
        tol = self._NEAREST_GOAL_END_TOL_RAD
        if (max(abs(float(a) - float(b)) for a, b in zip(plan[0], start)) > tol
                or max(abs(float(a) - float(b)) for a, b in zip(plan[-1], end)) > tol):
            return failed("cuRobo's plan does not open and end on the configurations the band legs reach", stuck=False)
        route = ([ends[0]] if 0 in legs else []) + [[float(v) for v in w] for w in plan] + (
            [ends[1]] if 1 in legs else [])
        return _BandRoute(route=route, note="; ".join(said), stuck=False, legs=(int(0 in legs), int(1 in legs)))

    def _band_leg(
        self, planner: CuroboUrPlanner, exact: object, config: "Sequence[float]", row: object, *, escape: bool,
        command: MotionCommand, target_pose: Pose | None = None, target_joints: JointPositions | None = None,
    ) -> "list[float] | None":
        """The nearest configuration to a band ``config`` both authorities clear, whose straight leg both pass; or ``None``.

        The candidates turn only the joints that move one link of the planner's pairs against the other, at most
        :data:`~src.robot.safety.planning.band.BAND_LEG_MAX_DEG` each, nearest first by the path gate's measure
        (``band.band_neighbours``). The planner clears them in one request at ``safety.planned_motion.line_clearance_mm``
        from its world, with no pair left to anyone; the exact guard accepts the first of those; then the straight leg
        between ``config`` and it, out of the band for an ``escape`` and into it otherwise, is judged by both authorities
        as every straight line (:meth:`_judge_legs`). Up to :attr:`_BAND_LEG_TRIES` candidates are tried.
        """
        radii = self._preflight.joint_radii_mm(self) if self._preflight is not None else None
        pairs = tuple(getattr(row, "pairs", ()) or ())
        if radii is None or not pairs:
            return None
        candidates = band_neighbours(config, joints_between(pairs, exact), radii)  # type: ignore[arg-type]
        if not candidates:
            return None
        clearance = self._line_clearance_mm()
        report = planner.judge_joint_path([list(c) for c in candidates], clearance_mm=clearance,  # type: ignore[attr-defined]
                                          name_pairs=False)
        refused = {row.index for row in report.refused}
        tries = 0
        for index, candidate in enumerate(candidates):
            if index in refused or not self._exact_guard_accepts(candidate, command):
                continue
            leg = [list(config), list(candidate)] if escape else [list(candidate), list(config)]
            if self._judge_legs(planner, leg, command=command, target_pose=target_pose, target_joints=target_joints,
                                clearance_mm=clearance) is None:
                return [float(v) for v in candidate]
            tries += 1
            if tries >= self._BAND_LEG_TRIES:
                break
        return None

    def _start_refusal(
        self, refusal: object, command: MotionCommand, *,
        target_pose: Pose | None = None, target_joints: JointPositions | None = None, note: str = "",
    ) -> MotionResult:
        """The typed refusal of a move the planner will not start: the status of what it found where the arm stands.

        A start the planner refuses is not a goal it could not reach, and TIMEOUT with the no-plan sentence would send
        a recovery to look again at a scene that is not the problem. ``note`` is what the two authorities found there
        (:meth:`_band_route`): in the planner's cushion band, the pair, the planner's depth and the distance the exact
        meshes keep, and why no escape leg was taken.
        """
        return MotionResult.failed(
            _planner_refusal_status(refusal), command, target_pose=target_pose, target_joints=target_joints,
            message=_with_refusal(
                "the arm stands where the planner will not start from, so no plan exists from here and nothing was "
                "sent; " + (f"{note}; " if note else "") + "bring the arm out of this configuration before it plans "
                "again", refusal),
        )

    def _route(
        self, waypoints: "Sequence[Sequence[float]]", *, planned: bool, how: str, dense: int = 2,
        tried: "Sequence[str]" = (), held: "URBranch | None" = None, reached: "URBranch | None" = None,
    ) -> _Route:
        """The route that passed, saying how it was chosen; a change of branch on it is a warning naming both.

        ``planned`` says whether cuRobo planned ``waypoints`` or they are the two ends of a straight line, ``dense``
        how many waypoints cuRobo returned, and ``tried`` what failed before this route, in the order it was tried.
        """
        if tried:
            how += f", after {'; '.join(tried)}"
        changes = not _same_branch(held, reached)
        if changes:
            self.logger.warning(
                "this move changes the arm's branch from %s to %s: no configuration on the branch it holds could be "
                "reached (%s)", _branch_text(held), _branch_text(reached), "; ".join(tried) or "none was admissible",
            )
            how += f"; branch change from {_branch_text(held)} to {_branch_text(reached)}"
        overshoot_deg = 0.0
        if planned:
            try:
                overshoot_deg = max((math.degrees(v) for v in span_overshoot_rad(waypoints)), default=0.0)
            except ValueError:  # a waypoint that cannot be read: the path gate refused it before a route was made
                overshoot_deg = 0.0
        return _Route(waypoints=waypoints, dense=dense, how=how, planned=planned, branch_change=changes,
                      overshoot_deg=overshoot_deg)

    def _line_clearance_mm(self) -> float:
        """How far a straight joint line has to stay from the planner's world: ``safety.planned_motion``."""
        return float(self.config.safety.planned_motion.line_clearance_mm)

    def _planner_margin_mm(self) -> float:
        """The padding the planner puts on a pair of the links it pads, ``planner_margin_mm``: the most of its padding
        the exact guard may decide past (``band.admission_refusal``). Undeclared reads 0, and no planner starts then."""
        return float(getattr(self.config.safety.self_collision, "planner_margin_mm", 0.0) or 0.0)

    def _detour(self, waypoints: "Sequence[Sequence[float]]", *, cap_deg: "float | None" = None) -> "tuple[float, str]":
        """How far a path swings its worst joint beyond the span between its start and its goal, where that is too far.

        ``(degrees, sentence)`` where some joint swings more than ``safety.planned_motion.max_detour_deg`` past the
        span, the sentence naming the joint; ``(0.0, "")`` where every joint stays within it. The bound is read on the
        waypoints: the legs between them are straight joint lines and add nothing to it
        (:func:`~src.robot.safety.path_samples.span_overshoot_rad`). ``cap_deg`` is a tighter bound where one holds, the
        one of :meth:`keeping_its_branch`, and the sentence then names it.
        """
        from .curobo_motion import UR_ARM_JOINT_NAMES

        configured_deg = float(self.config.safety.planned_motion.max_detour_deg)
        cap_deg = configured_deg if cap_deg is None else float(cap_deg)
        try:
            overshoot = [math.degrees(v) for v in span_overshoot_rad(waypoints)]
        except ValueError:  # a waypoint that cannot be read: the path gate refuses it and says why
            return 0.0, ""
        if not overshoot or max(overshoot) <= cap_deg:
            return 0.0, ""
        worst = max(range(len(overshoot)), key=overshoot.__getitem__)
        joint = UR_ARM_JOINT_NAMES[worst] if len(overshoot) == len(UR_ARM_JOINT_NAMES) else f"joint {worst + 1}"
        # The configured bound wherever it refuses the plan too: the tighter one is the reason only where it alone does.
        bound = (f"safety.planned_motion.max_detour_deg of {configured_deg:g}"
                 if cap_deg >= configured_deg or overshoot[worst] > configured_deg
                 else f"the {cap_deg:g} deg a move {_KEEPS_ITS_BRANCH} allows")
        return overshoot[worst], (f"swings {joint} {overshoot[worst]:.1f} deg beyond the span between its start and "
                                  f"its goal, more than {bound}")

    def _goal_joint_window(self) -> tuple[list[float], list[float]]:
        """Where a chosen goal's joints may lie, in radians: the planner's window, narrowed to what the guard admits.

        The joint-limit guard's table less its margin, the numbers it refuses outside of. Where the guard keeps
        half a turn about home (``safety.joint_limits.within_half_turn_of_home``), that is the table narrowed to
        the window about this arm's home, so no goal is chosen outside it. With no guard, or no table for this
        arm, the planner's window alone; a goal outside what the guard would admit is then the endpoint gate's
        to refuse, as for any plan.
        """
        from .curobo_motion import PLANNER_JOINT_ENVELOPE_RAD

        lower, upper = list(PLANNER_JOINT_ENVELOPE_RAD[0]), list(PLANNER_JOINT_ENVELOPE_RAD[1])
        guard = next((
            candidate for candidate in (getattr(self._preflight, "guards", ()) or ())
            if getattr(candidate, "name", None) == "joint_limit" and callable(getattr(candidate, "limits_for", None))
        ), None)
        if guard is None:
            return lower, upper
        for_arm = getattr(guard, "limits_for_arm", None)
        limits = (for_arm(self) if callable(for_arm)
                  else guard.limits_for(vendor=self.capabilities.vendor, model=self.capabilities.model))
        if limits is None or len(limits[0]) != len(lower):
            return lower, upper
        margin = float(getattr(guard, "margin_deg", 0.0))
        for axis, (low_deg, high_deg) in enumerate(zip(*limits)):
            lower[axis] = max(lower[axis], math.radians(float(low_deg) + margin))
            upper[axis] = min(upper[axis], math.radians(float(high_deg) - margin))
        return lower, upper

    def _judged_waypoints(
        self, planner: CuroboUrPlanner, traj_ur: "list[list[float]]", *, command: MotionCommand,
        target_pose: Pose | None = None, target_joints: JointPositions | None = None,
        legs: "tuple[int, int]" = (0, 0),
    ) -> "tuple[Sequence[Sequence[float]], MotionResult | None]":
        """The waypoints to run, one ``moveJ`` each, with both authorities having judged their legs; or the refusal.

        cuRobo samples a plan every 25 ms, and a ``moveJ`` starts from rest and stops at its target, so running
        every sample stops the arm at every sample. The plan is shortened first to the fewest of its own
        waypoints whose legs stay within the path gate's step of it, in the gate's own metric
        (:func:`~src.robot.safety.path_samples.simplify_joint_path`); on 12 measured UR10 plans 8 became one
        ``moveJ``. The chords it draws are straight joint lines nobody planned, as the direct line is, so the
        shortened list is judged whole, by the local path gate and by the planner at
        ``safety.planned_motion.line_clearance_mm``, before it may run (found porting 85f082b to dev, 2026-09-25:
        judged at no contact, a plan that barely bends came back from the shortening as the very line refused at the
        clearance, and ran). Where either refuses, the plan as cuRobo returned it is judged by both at no contact, as
        cuRobo validated it, and where that is refused too the move is.

        What is returned is the very list that passed, and the caller hands it to ``execute`` unchanged: the
        legs judged are the legs run. The detour bound is read on it once more, although a subset of a plan's own
        waypoints that keeps both ends cannot swing further than the plan did, so the bound is stated on the list
        that runs. A cell whose preflight samples no path cannot shorten one, and its plan meets
        ``gate_planned_path`` as it always did, which refuses it.

        ``legs`` counts the straight band legs at the head and at the tail of ``traj_ur`` (:meth:`_band_route`): the
        escape out of a pose the planner's padded spheres refuse and the approach into one. Only cuRobo's part between
        them is shortened, so each runs as the one ``moveJ`` of at most
        :data:`~src.robot.safety.planning.band.BAND_LEG_MAX_DEG` per joint it was chosen and judged as, and the route's
        log line says the leg that runs.
        """
        from src.robot.safety.path_samples import simplify_joint_path

        running: "Sequence[Sequence[float]]" = traj_ur
        step = self._preflight.path_step_mm if self._preflight is not None else None
        reach = self._preflight.joint_radii_mm(self) if step is not None else None
        refused: "MotionResult | None" = None
        if step is not None and reach is not None:
            shortened: "tuple[tuple[float, ...], ...] | None"
            head, tail = legs
            try:
                shortened = (tuple(tuple(float(v) for v in w) for w in traj_ur[:head])
                             + simplify_joint_path(traj_ur[head:len(traj_ur) - tail], reach_mm=reach, tolerance_mm=step)
                             + tuple(tuple(float(v) for v in w) for w in traj_ur[len(traj_ur) - tail:]))
            except ValueError:  # a waypoint that cannot be read: the path gate below refuses it and says why
                shortened = None
            if shortened is not None and len(shortened) < len(traj_ur):
                refused = self._judge_legs(planner, shortened, command=command, target_pose=target_pose,
                                           target_joints=target_joints, clearance_mm=self._line_clearance_mm())
                if refused is None:
                    running = shortened
                else:
                    self.logger.info(
                        "the plan shortened to %d of its %d waypoints was refused (%s: %s); the plan is judged as "
                        "cuRobo returned it", len(shortened), len(traj_ur), refused.status.value, refused.message,
                    )
        if running is traj_ur:
            refused = self._judge_legs(planner, traj_ur, command=command, target_pose=target_pose,
                                       target_joints=target_joints)
            if refused is not None:
                return (), refused
        _, detour = self._detour(running)
        if detour:
            return (), MotionResult.failed(
                MotionStatus.JOINT_LIMIT_REJECTED, command, target_pose=target_pose, target_joints=target_joints,
                message=f"the path that would run {detour}; nothing was sent",
            )
        return running, None

    def _judge_legs(
        self, planner: CuroboUrPlanner, waypoints: "Sequence[Sequence[float]]", *, command: MotionCommand,
        target_pose: Pose | None = None, target_joints: JointPositions | None = None, clearance_mm: float = 0.0,
    ) -> "MotionResult | None":
        """Both authorities on the legs between neighbouring ``waypoints``; ``None`` means every leg passed.

        The local path gate first, the exact meshes and the declared fixtures, then the planner on the same samples
        against the world it holds, the camera's and the carried part's. The world was refreshed before, so the
        planner is asked with ``refresh=False`` and both judge the same cell; where no step or reach derives,
        ``gate_planned_path`` has already refused and the planner is not asked. ``clearance_mm`` is the world
        clearance the planner holds the samples to, which a straight line the arm would run instead of a plan is asked
        for (``safety.planned_motion.line_clearance_mm``); a plan's legs are judged at none, as cuRobo validated them.
        """
        refused = self._preflight.gate_planned_path(waypoints, arm=self, command=command)
        if refused is not None:
            return refused
        step = self._preflight.path_step_mm if self._preflight is not None else None
        if step is None:  # no self-collision guard: gate_planned_path already refused
            return None
        reach = self._preflight.joint_radii_mm(self)
        if reach is None:
            return None  # gate_planned_path refused this already; here it would be a second voice
        from src.robot.safety.path_samples import waypoint_path_samples

        samples = waypoint_path_samples(waypoints, reach_mm=reach, max_step_mm=step)
        why_not = self._mesh_first_refusal(planner, samples.configs)
        if why_not is None:
            # Mesh first: the exact guard just accepted every sample (gate_planned_path); the spheres are not asked.
            self.logger.info("mesh first: the exact guard alone judged %d sample(s) of %s clear; cuRobo was not asked",
                             len(samples.configs), "this straight joint line" if len(waypoints) == 2 else "these legs")
            return None
        self.logger.info("mesh first does not apply to %s (%s): cuRobo judges beside the exact guard",
                         "this straight joint line" if len(waypoints) == 2 else "these legs", why_not)
        try:
            verdict = planner.check_joint_path(samples.configs, refresh=False, clearance_mm=clearance_mm)
        except CuroboUnavailableError as exc:
            return MotionResult.failed(
                MotionStatus.CONTROLLER_REJECTED, command, target_pose=target_pose, target_joints=target_joints,
                message=f"cuRobo planner unavailable: {exc}", exception=exc,
            )
        if verdict.valid:
            return None
        # The samples the exact gate just accepted: a self collision the planner's padded spheres alone find, on pairs
        # the exact guard judges (the owner, 2026-09-30), and a world refusal only the camera's boxes make, with a hand
        # known empty and open (Option 1), are that guard's to decide.
        standing = self._exact_guard_decides(planner, samples.configs, verdict, clearance_mm=clearance_mm,
                                             command=command)
        if standing is None:
            return None
        what = "this straight joint line" if len(waypoints) == 2 else "the legs this path would run"
        status, message = standing.said(f"the planner refused {what}", verdict, len(samples.configs))
        return MotionResult.failed(status, command, target_pose=target_pose, target_joints=target_joints,
                                   message=message)

    def _log_the_route(self, what: str, route: _Route) -> None:
        """One line per move: how its route was chosen, how many waypoints ran, and how far each joint turns on them.

        The travel is read on the list that runs, leg by leg: per joint its total turn and its largest single leg, in
        degrees. A long way round shows as a joint turning far more than its end needs, its travel against the
        shortest turn to the same angle; past half a turn more it is a warning, because an unwind the joint limits
        force looks the same and a person should read which it was.
        """
        from .curobo_motion import UR_ARM_JOINT_NAMES

        path = np.asarray([[float(v) for v in w] for w in route.waypoints], dtype=np.float64)
        legs = np.abs(np.diff(path, axis=0)) if len(path) > 1 else np.zeros((1, path.shape[1]))
        travel, largest = legs.sum(axis=0), legs.max(axis=0)
        names = UR_ARM_JOINT_NAMES if len(UR_ARM_JOINT_NAMES) == len(travel) else tuple(
            f"joint {i + 1}" for i in range(len(travel)))
        per_joint = ", ".join(f"{name} {math.degrees(float(t)):.1f}/{math.degrees(float(m)):.1f}"
                              for name, t, m in zip(names, travel, largest))
        self.logger.info(
            "%s: %s; %d waypoint(s) dense, %d executed (%d leg(s)); joint travel in deg, total/largest leg: %s",
            what, route.how, route.dense, len(path), max(len(path) - 1, 0), per_joint,
        )
        shortest = np.abs((path[-1] - path[0] + np.pi) % (2.0 * np.pi) - np.pi)
        extra = travel - shortest
        if float(extra.max()) > math.pi:
            worst = int(np.argmax(extra))
            self.logger.warning(
                "%s turns %s %.1f deg where %.1f deg reaches the same angle: the long way round, or an unwind the joint "
                "limits force", what, names[worst], math.degrees(float(travel[worst])),
                math.degrees(float(shortest[worst])),
            )

    def _gate_planned_config(
        self, pose: Pose, final_joints: JointPositions
    ) -> "MotionResult | None":
        """Gate a planner's final configuration with the box and the joint guards; ``None`` = accepted."""
        return self._judge_planned_config(pose, final_joints, commanded=True)

    def _screen_planned_config(
        self, pose: Pose, final_joints: JointPositions
    ) -> "MotionResult | None":
        """:meth:`_gate_planned_config`'s verdict on a configuration nothing is sent to; ``None`` = admitted.

        The same context, guards and skips, asked through :meth:`SafetyPreflight.screen`: the continuity memo stays
        where the last commanded target left it, and a refusal is no rejected motion in the log.
        :meth:`nearest_configuration` screens with it; a move's route gates its goals with
        :meth:`_gate_planned_config`, whose acceptance is remembered because that goal is the one it drives.
        """
        return self._judge_planned_config(pose, final_joints, commanded=False)

    def _judge_planned_config(
        self, pose: Pose, final_joints: JointPositions, *, commanded: bool
    ) -> "MotionResult | None":
        """The endpoint gate on ``final_joints`` for the TCP ``pose``, commanded or only asked; ``None`` = accepted."""
        if self._preflight is None:
            return None
        current_joints: JointPositions | None = None
        if self._conn.is_connected:
            try:
                current_joints = self.get_joint_positions()
            except RobotConnectionError:
                current_joints = None
        ctx = self._preflight.context_for_pose(
            pose, command=MotionCommand.MOVE_TO, target_joints=final_joints,
            current_joints=current_joints, arm=self,
        )
        judge = self._preflight.evaluate if commanded else self._preflight.screen
        decision = judge(
            ctx, skip_guards=frozenset({"ik_quality", "motion_continuity"})
        )
        return SafetyPreflight.as_motion_result(
            decision, MotionCommand.MOVE_TO, target_pose=pose, target_joints=final_joints,
        )

    def attach_payload(self, grip_width_mm: float) -> bool:
        """Tell the planner the gripper is carrying a part about ``grip_width_mm`` across.

        The caller knows how wide the jaws closed, which is a real measurement of the
        part at the grasp line. It knows nothing about how far the part extends beyond
        it, so the length and the lateral allowance come from
        ``safety.planning_world.payload``, where an operator declares the worst case
        their cell handles.

        It returns ``False`` where nothing was attached, which includes the ordinary
        case of a cell that uses no planner. A cell that cannot model its payload
        carries on and says so.

        Every call says the jaws closed on a part, modelled or not. A part the planner
        models: from here until :meth:`detach_payload`, no box the camera saw is set aside
        in the planner's world (the owner's Option 1), because nobody judges a carried part
        against them then. A part the cell models nowhere is judged as an empty hand is
        (:meth:`_world_aside_refusal`, the owner, 2026-10-05).
        """
        self._closed_on_part = True
        cfg = getattr(getattr(self.config.safety, "planning_world", None), "payload", None)
        if cfg is None or self._payload_config_declined() is not None:
            return False
        from src.robot.safety.planning.self_envelope import carried_part_box, hand_spheres

        length = float(cfg.length_mm)
        # The self filter covers the part from here on, whatever the planner answers: a part the
        # planner could not attach is still in the gripper and still in front of the cameras.
        self._attached_payload = (length, float(cfg.lateral_margin_mm))
        self._payload_in_planner = False
        # The planner carries the same part where the hand holds it: along the approach, from the fingertips.
        kinematics = self._preflight.self_kinematics(self)
        hand = self._preflight.planner_hand(self)
        spheres = hand_spheres(hand, kinematics[0]) if kinematics is not None and chosen(hand) else None
        if spheres is None or not chosen(hand):
            self.logger.error(
                "payload NOT attached: this cell's guard places no hand model on a known arm, so there is no "
                "fingertip to hang the part from; the planner is routing as if the gripper were empty"
            )
            return False
        dims_mm, centre_mm = carried_part_box(
            hand, spheres, grip_width_mm=grip_width_mm, length_mm=length,
            lateral_margin_mm=float(cfg.lateral_margin_mm),
        )
        try:
            joints = self.get_joint_positions().tolist()
        except RobotConnectionError:
            return False
        self._payload_in_planner = bool(self._curobo_ur_planner().attach_payload(joints, dims_mm, centre_mm))
        return self._payload_in_planner

    def _payload_config_declined(self) -> str | None:
        """Why this arm's configuration models no carried part, or ``None``; reads no hand and starts no planner."""
        cfg = getattr(getattr(self.config.safety, "planning_world", None), "payload", None)
        if cfg is None or not bool(cfg.enabled):
            return ("safety.planning_world.payload.enabled is false, so neither the planner nor the self filter "
                    "models a part in the gripper")
        if self._motion_planner != "curobo":
            return (f"robot.ur.motion_planner is {self._motion_planner!r}, and only the cuRobo planner carries a "
                    "part in the gripper")
        if cfg.length_mm is None:
            return ("safety.planning_world.payload.length_mm is undeclared, so neither the planner nor the self "
                    "filter models a part in the gripper: declare how far the longest part hangs past the fingertips")
        return None

    def payload_declined_reason(self) -> str | None:
        """Why this arm models no carried part, naming the key or the hand, or ``None`` where it can.

        The configuration first, then the hand the guard places: the same checks
        :meth:`attach_payload` makes, in the same order. It starts no planner and opens no
        connection.
        """
        declined = self._payload_config_declined()
        if declined is not None:
            return declined
        from src.robot.safety.planning.self_envelope import hand_spheres

        kinematics = self._preflight.self_kinematics(self)
        hand = self._preflight.planner_hand(self)
        if kinematics is None or not chosen(hand) or hand_spheres(hand, kinematics[0]) is None:
            return ("this cell's guard places no hand model on a known arm, so there is no fingertip to hang a part "
                    "from")
        return None

    def payload_model(self) -> PayloadModel:
        """What models the part the gripper carries now: what the last attach left in force."""
        if self._attached_payload is None:
            return PayloadModel.NONE
        return PayloadModel.PLANNER_AND_FILTER if self._payload_in_planner else PayloadModel.FILTER_ONLY

    def line_motion(self) -> LineReading:
        """What a ``move(pose, linear=True)`` on this arm keeps of the line, read before moving.

        On ik the controller draws the line with ``moveL`` and only the endpoint is gated. On
        cuRobo every sample is solved and judged by the exact mesh guard and the planner, and
        one ``moveL`` runs, unless the path cannot be judged, which the preflight says in the
        sentence its gate would refuse with.
        """
        if self._motion_planner != "curobo":
            return LineReading(LineMotion.CONTROLLER_LINE, (
                f"robot.ur.motion_planner is {self._motion_planner!r}: the controller draws the line and only its "
                "end is judged"))
        if self._preflight is None:
            return LineReading(LineMotion.NOT_KEPT, "no safety preflight is wired, so the line runs unjudged")
        refusal = self._preflight.path_judge_refusal(self)
        if refusal is not None:
            return LineReading(LineMotion.NOT_KEPT, refusal)
        return LineReading(LineMotion.CHECKED, (
            "every sample of the line is solved and judged by the exact mesh guard and the planner before one moveL "
            "runs"))

    def carried_line_refusal(self, pose: Pose, *, grip_width_mm: float,
                             camera_world: Maybe[CameraWorldDecline] = UNSET,
                             start: "tuple[Pose, Sequence[float]] | None" = None) -> MotionResult | None:
        """The refusal ``move(pose, linear=True, camera_world=camera_world)`` would meet from where the arm stands if its
        jaws held a part about ``grip_width_mm`` across; ``None`` where that line would run. Nothing moves and nothing
        goes to the controller.

        :class:`~src.robot.core.arm_capabilities.JudgesCarriedLines`. A grasp asks it at the part with its jaws still
        open, of the lift it is about to make, and closes only where the lift would run (the owner, 2026-10-01). It is the
        lift's own judgement: the camera world's refusal as :meth:`move` asks it, with the decline the lift would carry
        (``camera_world``, or a block's), then :meth:`_judge_linear_move`, every sample solved by the controller, the
        exact guard on every one at its step, the planner against its whole world with the part in it as the attach
        after the close hands it over, and no box the camera saw set aside, as never while a part is carried
        (:meth:`_world_aside_refusal`). On the ik planner the end's gate alone, screened, so the continuity memo stays
        where the last commanded target left it. An arm that holds no part is handed one for the judgement
        (:meth:`attach_payload`, a cell that models none included) and forgets it after (:meth:`detach_payload`); one
        that already holds a part is judged as it stands and keeps it.

        ``start`` judges the line as if the arm stood elsewhere, the TCP pose and the joints it would stand at there
        (:meth:`grasp_refusal_ahead`): the line from that pose, its first sample solved seeded on those joints.

        A lift judged from where the arm stands that would run, inside a held world, on a cell that models no carried
        part, is kept for the line up that runs it (:meth:`_drive_checked_line`): there a part nobody models is judged as
        an empty hand is (the owner, 2026-10-05, "mehr Griffe"), so the judgement after the close reads exactly what this
        one read, in the same world, and every sample of that line was judged here.
        """
        self._lift_judged = None
        if pose.frame is not Frame.BASE:
            return MotionResult.failed(
                MotionStatus.INVALID_TARGET, MotionCommand.MOVE_TO, target_pose=pose,
                message=f"URRobotArm.carried_line_refusal requires Frame.BASE; got {pose.frame!r}",
            )
        refused = self._refused_for_camera_world(self._move_camera_world(camera_world), MotionCommand.MOVE_TO,
                                                 target_pose=pose)
        if refused is not None:
            return refused
        planner = self._curobo_ur
        holds = (self._attached_payload is not None or self._closed_on_part
                 or (planner is not None and getattr(planner, "carries_part", False) is True))
        judged: MotionResult | None = None
        kept: _LiftJudged | None = None
        try:
            if not holds:
                self.attach_payload(float(grip_width_mm))
            judged = self._judge_linear_move(pose, command=MotionCommand.MOVE_TO, commanded=False, start=start)
            if judged is None and start is None:
                kept = self._lift_to_keep(pose)
        finally:
            if not holds and not self.detach_payload():
                self.logger.warning(
                    "the part handed to the planner for a lift judged as if the jaws held it is not confirmed forgotten: "
                    "until a detach confirms it, the planner judges every motion as if a part were carried, and nothing "
                    "of the camera's world is set aside")
        self._lift_judged = kept
        self.logger.info("a lift to %s judged as if the jaws held a part %.1f mm across: %s",
                         pose.label or "<unlabeled>", float(grip_width_mm),
                         "it would run" if judged is None else f"it would be refused: {judged.message}")
        return judged

    def _lift_to_keep(self, pose: Pose) -> "_LiftJudged | None":
        """The lift to ``pose`` just judged from where the arm stands, as :meth:`_drive_checked_line` may run it; ``None``
        where the line up would not judge exactly what was judged: outside a held world or before a refresh that vouched
        built it, and on a cell that models a carried part, whose judgement after the close holds a part this one only
        stood in for. Read while the judgement's own part is still in place, the hand left unread: a cell that models no
        part reads no hand (:meth:`_world_aside_refusal`), and the jaws are open now and closed then."""
        refresh = self._planner_refresh()
        if (self._held_world_reason is None or refresh is None or not refresh.ok
                or self._payload_config_declined() is None or self._attached_payload is not None):
            return None
        try:
            joints = tuple(float(v) for v in self._conn.get_joint_positions())
        except (RobotConnectionError, RuntimeError, OSError):
            return None
        planner = self._curobo_ur
        return _LiftJudged(pose=_pose_key(pose), reason=self._held_world_reason, refresh=refresh, planner=planner,
                           joints=joints, state=self._judgement_state(planner, hand=False), boxes=self._seen_boxes())

    def _why_the_lift_is_judged_again(self, lift: _LiftJudged, pose: Pose) -> str:
        """Why the line up to ``pose`` is judged again rather than run on the lift judged at the part; ``""`` where it
        runs: in the same held world, on the refresh that built it, by the same planner, to the same pose to the last bit,
        on a cell that models no carried part where the judgement's part stood in for the one carried
        (:attr:`_LiftJudged.stands_in`), unhalted, the arm within :data:`_AHEAD_MOVED_MM` of where it stood, and what else
        the judgement read as it was (:meth:`_judgement_state`, the hand unread)."""
        if self._held_world_reason is None or self._held_world_reason != lift.reason:
            return "it is not inside the held world the lift was judged in"
        if self._planner_refresh() is not lift.refresh or self._curobo_ur is not lift.planner:
            return "the planner or its world is another than the lift was judged in"
        if _pose_key(pose) != lift.pose:
            return "this line goes to another pose than the lift judged"
        if lift.stands_in and (self._payload_config_declined() is None or self._attached_payload is not None):
            return "this cell models the carried part, which the lift judged at the part only stood in for"
        if self._halt_record() is not None:
            return "the arm is halted"
        moved = self._moved_since_mm(lift.joints)
        if not moved <= self._AHEAD_MOVED_MM:
            return (f"the arm stands {moved:.2f} mm from where the lift was judged, more than {self._AHEAD_MOVED_MM:g} mm"
                    if math.isfinite(moved) else "where the arm stands cannot be compared with where the lift was judged")
        if self._judgement_state(self._curobo_ur, hand=False) != lift.state:
            return "the carried part, the camera's boxes the planner holds or the halts changed since"
        if self._seen_boxes() is not lift.boxes:
            return "the exact guard holds other boxes the camera saw than the ones the lift was judged against"
        return ""

    def grasp_refusal_ahead(self, *, standoff: Pose, grasp: Pose, lift: Pose, grip_width_mm: float) -> str:
        """Why a grasp would be refused before its jaws close, judged now from where the arm stands; ``""`` where every
        judgement passes, and on a cell that judges nothing ahead (no cuRobo, no guard, no connection). Nothing moves.

        :class:`~src.robot.core.arm_capabilities.JudgesGraspsAhead`. The pick runs the planned move to ``standoff``, the
        very route :meth:`move` would run (:meth:`_route_to_the_nearest_goal`); the line down to ``grasp`` from the
        configuration that route ends on; and the lift to ``lift`` from the grasp's configuration, solved by the
        controller seeded on that end, as if the jaws held a part ``grip_width_mm`` across, the camera's boxes not set
        aside (:meth:`carried_line_refusal`). A blocker's pick asks it of each grasp it may take (the owner, 2026-10-03).

        They are judged in the order that judges each once (the judge chain, 2026-10-08). The line down first, from the
        configuration the route ends on as the closed form names it before any route is planned
        (:meth:`nearest_configuration`: the same goals in the same order the route takes them). Then the lift from the
        grasp's configuration, in the world the line down was judged in: both are built about the grasp, the region
        between the jaws left out there (:meth:`_judge_linear_move`), and the pick judges the lift in that world at the
        part too (:meth:`held_world`). The route last. Where it ends on that very configuration, to the last bit, each
        was judged once, with two refreshes, and the route stays ready for the move to the standoff, which runs it as it
        was judged where nothing changed since (:meth:`_drive_curobo`). Where it ends anywhere else (a cuRobo plan, another
        goal, a turn flip), where an arm link refused the line down, whose place the configuration decides, and where the
        standoff has no configuration, the line and the lift are judged again from the route's end, each in the world it
        refreshes, as before.

        A line down the exact guard refuses at a part on the flange against a fixture is every configuration's refusal
        (:func:`_on_the_flange`) and ends the judgement before the route, which plans and judges thousands of samples:
        17 s a grasp in a bin's corner, for a line then refused 0.07 mm short at wrist 3 (the grasp bench, 2026-10-06).
        The first refusal ends the judgement and is said.

        A grasp all of whose judgements pass on a route that keeps the arm's branch has the move to its standoff keep it
        too, judged again or not (:meth:`_drive_curobo`, the owner's R2 of 2026-10-08).

        With ``safety.planning_world.hold.standoff`` (the owner's O2 of 2026-10-09) the route is judged in the world the
        line down from ``near`` was judged in, where that line passed: the three are judged in one world, about the grasp
        with the region between its jaws left out, and the move to the standoff and the line down from it then hold that
        world (:meth:`holding_the_approach`) instead of taking a frame at the standoff.
        """
        if self._motion_planner != "curobo" or self._preflight is None or not self._conn.is_connected:
            return ""
        for name, asked in (("standoff", standoff), ("grasp", grasp), ("lift", lift)):
            if asked.frame is not Frame.BASE:
                return f"the {name} is not in Frame.BASE ({asked.frame!r}), so the grasp is not judged ahead"
        # Whatever was judged ahead before is another grasp's.
        self._judged_ahead = self._lift_judged = self._ahead_kept_its_branch = None
        planner = self._curobo_ur_planner()
        width = float(grip_width_mm)

        def line_from(joints: "list[float]") -> "MotionResult | None":
            return self._judge_linear_move(grasp, command=MotionCommand.MOVE_TO, commanded=False,
                                           start=(standoff, joints))

        def lift_from(joints: "list[float]", world: "WorldRefresh | None") -> str:
            """The lift from the grasp's configuration solved seeded on ``joints``, in ``world`` while it stands."""
            try:
                at_grasp = [float(v) for v in self.ik(grasp, seed=JointPositions(joints)).tolist()]
            except (RobotKinematicsError, RobotConnectionError) as exc:
                return f"the grasp has no inverse kinematics solution from the standoff: {exc}"
            held = (self.held_world("the lift judged ahead, in the world the line down to its grasp was judged in")
                    if world is not None and self._planner_refresh() is world else nullcontext())
            with held:
                carried = self.carried_line_refusal(lift, grip_width_mm=width, start=(grasp, at_grasp))
            if carried is None:
                return ""
            return (f"the lift, as if the jaws held a part {width:g} mm across, would be refused "
                    f"({carried.status.value}: {carried.message})")

        try:
            near: "list[float] | None" = [float(v) for v in self.nearest_configuration(standoff).values]
        except Exception:  # noqa: BLE001 (no early answer: the route judges the standoff and says why)
            near = None
        line = None if near is None else line_from(near)
        if line is not None and _on_the_flange(line):
            return f"the line down to the grasp would be refused ({line.status.value}: {line.message})"
        if near is not None and line is None:
            said = lift_from(near, self._planner_refresh())
            if said:
                return said
        velocity = float(self.config.motion_limits.max_velocity)
        # The owner's O2 (safety.planning_world.hold.standoff, 2026-10-09): the route is judged in the world the line down
        # was judged in, which the move to the standoff and the line down from it then hold (holding_the_approach).
        world = self._planner_refresh() if near is not None and line is None else None
        route_held: "AbstractContextManager[None]" = (
            self.held_world("the route to the standoff, judged in the world its line down was judged in")
            if _holds_its_world_at(self, "standoff") and world is not None and world.ok else nullcontext())
        try:
            with route_held:
                route = self._route_to_the_nearest_goal(planner, self._pose_to_flange(standoff), standoff,
                                                        velocity=velocity)
        except CuroboUnavailableError as exc:
            return f"the move to the standoff could not be judged: cuRobo planner unavailable: {exc}"
        if isinstance(route, MotionResult):
            return f"the move to the standoff would be refused ({route.status.value}: {route.message})"
        at_standoff = [float(v) for v in route.waypoints[-1]]
        if near is not None and line is None and at_standoff == near:
            self._keep_the_route_ahead(planner, route, standoff, velocity=velocity)
            self._ahead_kept_its_branch = None if route.branch_change else _pose_key(standoff)
            return ""
        refused = line_from(at_standoff)
        if refused is not None:
            return f"the line down to the grasp would be refused ({refused.status.value}: {refused.message})"
        said = lift_from(at_standoff, None)
        # Set last: the line and the lift judged again refresh the world, and a refresh forgets what was judged ahead.
        self._ahead_kept_its_branch = None if said or route.branch_change else _pose_key(standoff)
        return said

    def _keep_the_route_ahead(self, planner: object, route: _Route, standoff: Pose, *, velocity: float) -> None:
        """Keep ``route``, judged last of a grasp judged ahead, for the move to ``standoff`` (:meth:`_drive_curobo`):
        where no world is held and the planner's last refresh, which vouched for the cell, built the world it was judged
        in. Nothing is kept otherwise, and the move judges its route as it runs."""
        refresh = self._planner_refresh()
        # The planner's own refresh first: an identity test narrows ``refresh`` to the attribute's type for mypy, and
        # the None test after it narrows that again.
        if getattr(planner, "last_world_refresh", None) is not refresh:
            return
        if refresh is None or not refresh.ok or self._held_world_reason is not None:
            return
        self._judged_ahead = _JudgedAhead(
            route=route, refresh=refresh, planner=planner, goal=_pose_key(standoff), velocity=float(velocity),
            at=time.monotonic(), state=self._judgement_state(planner), boxes=self._seen_boxes(),
        )

    def lines_refusal_ahead(self, *, approach: Pose, lines: Sequence[Pose]) -> str:
        """Why the planned move to ``approach``, or one of the straight ``lines`` after it, would be refused, judged now
        from where the arm stands; ``""`` where every judgement passes, and on a cell that judges nothing ahead (no
        cuRobo, no guard, no connection). Nothing moves.

        :class:`~src.robot.core.arm_capabilities.JudgesLinesAhead`. The move to ``approach`` as :meth:`move` would route
        it (:meth:`_route_to_the_nearest_goal`), then each line from where the one before ends, the configuration there
        solved by the controller seeded on that end, each in the world the motion itself would refresh. A push asks it of
        its P0 and its four contact legs before it leaves the look (URSim, 2026-10-03).
        """
        if self._motion_planner != "curobo" or self._preflight is None or not self._conn.is_connected:
            return ""
        for asked in (approach, *lines):
            if asked.frame is not Frame.BASE:
                return f"{asked.label or 'a station'} is not in Frame.BASE ({asked.frame!r}), so it is not judged ahead"
        planner = self._curobo_ur_planner()
        name = approach.label or "the approach"
        try:
            route = self._route_to_the_nearest_goal(planner, self._pose_to_flange(approach), approach,
                                                    velocity=float(self.config.motion_limits.max_velocity))
        except CuroboUnavailableError as exc:
            return f"the move to {name} could not be judged: cuRobo planner unavailable: {exc}"
        if isinstance(route, MotionResult):
            return f"the move to {name} would be refused ({route.status.value}: {route.message})"
        here_pose, here = approach, [float(v) for v in route.waypoints[-1]]
        for pose in lines:
            refused = self._judge_linear_move(pose, command=MotionCommand.MOVE_TO, commanded=False,
                                              start=(here_pose, here))
            if refused is not None:
                return (f"the line to {pose.label or 'the next station'} would be refused ({refused.status.value}: "
                        f"{refused.message})")
            try:
                here = [float(v) for v in self.ik(pose, seed=JointPositions(here)).tolist()]
            except (RobotKinematicsError, RobotConnectionError) as exc:
                return f"{pose.label or 'a station'} has no inverse kinematics solution on the way: {exc}"
            here_pose = pose
        return ""

    def detach_payload(self) -> bool:
        """Forget the carried part. Safe to call when nothing was ever attached.

        The self filter forgets it here. The planner answers for its own model
        (:meth:`CuroboUrPlanner.detach_payload`): ``True`` without asking its sidecar where no attach was
        handed to it, as after a grasp that modelled nothing, and the sidecar's answer wherever one was.
        The arm forgets that its jaws closed on a part here too, the one way it does.
        """
        self._attached_payload = None
        self._payload_in_planner = False
        self._closed_on_part = False
        if self._curobo_ur is None:
            return True
        return self._curobo_ur.detach_payload()

    def _curobo_ur_planner(self) -> CuroboUrPlanner:
        """Lazily build the real-UR cuRobo execution glue bound to this arm's RTDE connection, under the planner lock.

        A halted arm keeps none it builds for a move: where a disconnect or a stop_planner dropped the glue, a move
        still in flight would get a fresh one here, and its next refresh or check would start a sidecar for a move that
        can send nothing, on an arm the console may build anew. It gets a glue retired at once instead, kept by
        nobody: every ask that would start a sidecar is refused (``CuroboUnavailableError``, which every planner call
        of a move already turns into a refusal), and its ``execute`` sends nothing past the latch. :meth:`start_planner`
        builds and keeps one, halted or not, because a person asked for it.
        """
        planner = self._curobo_ur
        if planner is not None:
            return planner
        with self._planner_lock:
            if self._curobo_ur is None and not self._planner_starting:
                halted = self._halt_record()
                if halted is not None:
                    unkept = self._build_curobo_ur_planner(keep=False)
                    unkept.retire(f"the arm is halted ({halted.reason}), so no planner is started for a move: a person "
                                  "confirms the cell is clear, and the next move starts one, or start_planner does now")
                    return unkept
            return self._build_curobo_ur_planner()

    def _build_curobo_ur_planner(self, *, keep: bool = True) -> CuroboUrPlanner:
        """The body of :meth:`_curobo_ur_planner`: built once, under the planner lock, and kept; with ``keep`` false a
        new one every call, which the arm does not hold."""
        if self._curobo_ur is None or not keep:
            from src.robot.safety.planning.reservation import PlannerReservation
            from src.robot.safety.planning.world import (
                build_planner_cuboids,
                build_planner_meshes,
            )

            from .curobo_motion import CuroboUrPlanner

            # The planner is built for this robot. A default factory taking no arguments
            # falls through to WILLY_CUROBO_ROBOT or ur5e.yml, so a real UR3e would be
            # planned against UR5e link lengths, on hardware, with nothing saying so. The
            # real-UR cuRobo path is fail-closed and never degrades to blind IK, so a
            # missing willy_{model}.yml stops the cell instead, which is the correct failure.
            #
            # The cell own geometry goes with it, or the planner routes through it.
            # Without this the planner knows its robot model and a generic table and
            # nothing about the bench, the bin walls or any fixture, so it plans straight
            # through them and the only thing that objects is the one-shot guard on the
            # final configuration, which cannot see a path. It is empty and disabled by
            # default, `safety.planning_world.enabled` turns it on, and the block refuses
            # to enable without a declared bench, because registering a world replaces
            # the planner own.
            world_cfg = getattr(self.config.safety, "planning_world", None)
            reservation = PlannerReservation.from_config(robot_cfg=self.config)
            built = CuroboUrPlanner(
                self._conn,
                client_factory=self._curobo_client_factory or self._default_curobo_client_factory(),
                vel=self.config.motion_limits.max_velocity,
                acc=self.config.motion_limits.max_acceleration,
                world_cuboids=build_planner_cuboids(
                    world_cfg, self.config.safety.self_collision.fixtures,
                    max_cuboids=reservation.cuboid_slots,
                ),
                world_meshes=build_planner_meshes(world_cfg),
                reservation=reservation,
                require_registration=bool(
                    getattr(world_cfg, "require_registration", True)
                ),
                live_world=self._live_world,
                self_envelope=self._self_envelope,
                descriptor_check=self._descriptor_check(),
                # After a sent moveJ fails the arm may have moved part of the way, and the controller's own
                # state is what says whether a stop ended it.
                controller_state=self._controller_state_text,
                # The path guard lives on this arm rather than on the planner, and the two
                # have to be looking at the same cell: a planner routing around a tote the
                # guard cannot see gives a path that avoids it and a gate that would have
                # passed one straight through.
                on_perceived_obstacles=self._preflight.set_perceived_obstacles,
            )
            # The attach spheres the reservation counts, the same number the evidence lookup names.
            if reservation.sphere_slots:
                built.enable_payload(int(reservation.sphere_slots))
            if not keep:
                return built
            self._curobo_ur = built
        return self._curobo_ur

    @property
    def live_planner_world(self) -> "LivePlannerWorld | None":
        """What this arm asks before it plans, or `None` where nothing was handed in.

        Read by whatever holds the scene, which is the pick loop: it is the only thing that
        knows which object an attempt is reaching for, and that object has to be left out of
        the world or the planner refuses to approach it.
        """
        return self._live_world

    def wrist_bodies(self) -> "tuple[Any, ...]":
        """The wrist cameras this arm carries (``body_link.WristBody``), as its guard holds them; empty for none. The
        pick loop places their housing on each grasp to choose its wrist turn (``wrist_turn``)."""
        return tuple(self._preflight.wrist_bodies(self)) if self._preflight is not None else ()

    def set_wrist_bodies(self, bodies: "Sequence[Any]") -> None:
        """Hand this arm the wrist cameras it carries (``body_link.WristBody``), before its planner or
        guard is built.

        The planner loads them on top of the config its combination evidence names, the exact mesh guard
        and the self filter hold them, and the planner start proves their cover. Refused once the planner
        was built or the guard judged a path, because a camera handed in then would be one they say they
        hold and do not. On a ``willy`` cell each body's recorded flange to TCP has to be the declared
        frame; a ``polyscope`` cell compares it at connect. An empty sequence clears them.
        """
        if self._curobo_ur is not None:
            raise RuntimeError(
                "this arm's planner was already built, so a wrist camera handed in now would not be in it: hand the "
                "cell's wrist bodies to the arm before its first planned move or planner start"
            )
        listed = tuple(bodies)
        if self.config.gripper.tool_frame.source == "willy":
            for body in listed:
                stale = body.record_refusal("willy", self._declared_tool_matrix())
                if stale is not None:
                    raise ValueError(stale)
        self._preflight.set_wrist_bodies(listed)

    def set_live_planner_world(self, world: "LivePlannerWorld | None") -> None:
        """Hand this arm the thing that answers what the cell looks like right now.

        Set before the first planned move. An arm that already built its planner passes it
        on, so a cell that wires this after connecting still gets it.
        """
        self._forget_what_was_judged_ahead()
        self._live_world = world
        if self._curobo_ur is not None:
            self._curobo_ur.set_live_world(world)

    def set_hand(self, gripper: object | None) -> None:
        """Hand this arm the hand it carries, ``None`` for none: ``connect_cell`` does once both are up.

        It is read, never commanded: where only the boxes a camera saw refuse the planner's world, they are set aside
        and the exact guard decides them only while this hand reads empty and open
        (``core.gripper.why_not_known_open``, the owner, 2026-10-01). An arm that was handed none, or a hand that cannot
        say, keeps them in, and cuRobo decides as before.
        """
        self._hand = gripper

    @property
    def camera_world_required(self) -> bool:
        """Whether every motion of this arm needs a live camera world or a decline: true on cuRobo."""
        return self._motion_planner == "curobo"

    def camera_world_for_move(self, camera_world: Maybe[CameraWorldDecline] = UNSET) -> CameraWorldStamp:
        """The camera world a :meth:`move` with ``camera_world`` would carry, read as the arm is now.

        The stamp :meth:`move` starts from before a refresh vouches for it: UNPLANNED on ik,
        MISSING on cuRobo with no live world and no decline, DECLINED for a decline. It reads no
        connection and starts no planner.
        """
        return self._move_camera_world(camera_world)

    def live_camera_world_wired(self) -> bool:
        """Whether a live camera world is wired to this arm."""
        return self._live_world is not None

    def without_camera_world(self, reason: str) -> AbstractContextManager[CameraWorldDecline]:
        """Decline the camera world for every motion this arm commands inside the ``with`` block.

        Bound to this arm alone (:func:`~src.robot.core.camera_world.without_camera_world`). A
        decline reaches both typed motions, :meth:`move` and :meth:`move_to_joints`, on the cuRobo
        planner, and on the ik planner they say UNPLANNED whatever was declined. With a live camera
        world wired a declined typed motion is refused before its path is judged, because every path
        on that arm is judged against the world.
        """
        return _without_camera_world(self, reason)

    def _self_envelope(self) -> "SelfEnvelope | None":
        """This arm's own body now, for taking the robot out of the view.

        A fixed camera watching a cell sees the robot, and a robot registered as an obstacle
        cannot move at all. The links are capsules fitted to the committed bundle, the hand is its
        sphere map and an attached part a capsule from the fingertips, placed with the guard's
        model and yaw. `None` where no model or no hand resolves or the connection cannot answer,
        and the world source then refuses to build a perceived world rather than registering the
        arm as geometry.
        """
        from src.robot.safety.planning.self_envelope import self_envelope

        try:
            joints = np.asarray(self._conn.get_joint_positions(), dtype=np.float64)
        except Exception:  # noqa: BLE001 (a cell that cannot say where it is has no self to filter)
            return None
        return self_envelope(self._preflight, self, joints, payload=self._attached_payload)

    def _default_curobo_client_factory(self) -> Callable[[], CuroboPlanClient]:
        """A cuRobo client bound to this cell robot model and to its guard planner margin.

        It mirrors the sim driver: the configured model wins, and the clearance the
        SelfCollisionGuard will demand is handed to the planner, so it stops returning
        configurations the guard would reject.
        """
        from src.robot.drivers.sim.robot_models import curobo_arm_descriptor
        from src.robot.safety.planning._curobo_body_links import BodyLinkError
        from src.robot.safety.planning.body_link import HandLink, coupling_bodies
        from src.robot.safety.planning.margin import planner_margin_refusal
        from src.robot.safety.planning.reservation import PlannerReservation
        from src.robot.safety.planning.robot.retract_table import (
            RetractMissing,
            read_retract,
        )

        # No implied margin. A UR cell that plans with cuRobo says what clearance its planner keeps,
        # or it does not plan: an undeclared margin read as 0 lets the sidecar propose paths at
        # 9.44 mm against a 10 mm guard. Read here and raised in build(), where a cell that never
        # plans never pays for it.
        margin_refusal = planner_margin_refusal(self.config)

        hand = self._preflight.planner_hand(self)
        # One descriptor per arm, with the hand this cell names added as a body link when the sidecar
        # starts. A cell that names no hand has no hand to add: the refusal is raised when the planner
        # first starts, which a move reports as CONTROLLER_REJECTED.
        robot_yml = curobo_arm_descriptor(self.config.ur.model) if chosen(hand) else None
        margin_mm = float(getattr(self.config.safety.self_collision, "planner_margin_mm", 0.0) or 0.0)
        # What the sidecar allocates, from this cell's own declaration. Without it the sidecar starts
        # with 16 boxes, no mesh and no grid whatever the cell declared, so a tote or a live scene meets
        # a planner with nowhere to put it.
        reservation = PlannerReservation.from_config(robot_cfg=self.config)

        def build() -> CuroboPlanClient:
            if margin_refusal is not None:
                raise CuroboUnavailableError(f"{margin_refusal[0]} {margin_refusal[1]}")
            if robot_yml is None or not chosen(hand):
                raise CuroboUnavailableError(
                    "robot.gripper.model is unset, so this cell has no hand to add to the planner's robot, and a "
                    "planner is never started for a hand nobody named"
                )
            try:
                link = HandLink.from_hand(hand)
            except BodyLinkError as exc:
                raise CuroboUnavailableError(str(exc)) from exc
            # The pose this pair was judged at. The descriptor is per arm and carries a fallback, while
            # the pose a cell plans from belongs to the arm and the hand on it, because a pose that
            # clears the exact meshes with one hand can be a self collision in the planner's sphere
            # model with another: on a ur3 the anchor works with the 2F-85 and refuses with the Hand-E.
            try:
                default_q = read_retract(str(self.config.ur.model), hand.model, float(hand.coupling_mm),
                                        margin_mm,
                                        placement=f"{link.placement.approach}{link.placement.closing}")
            except RetractMissing as exc:
                raise CuroboUnavailableError(str(exc)) from exc
            client = CuroboPlanClient(
                robot_config=robot_yml, self_collision_margin_mm=margin_mm,
                body_links=[link.to_dict(), *coupling_bodies(hand)],
                # The wrist cameras this arm was handed, read when the planner is built.
                wrist_body_links=[body.link() for body in self._preflight.wrist_bodies(self)],  # type: ignore[attr-defined]
                default_q=default_q,
            )
            client.reserve_world(reservation)
            return client

        return build

    def _descriptor_check(self) -> "Callable[[SidecarIdentity], str | None]":
        """What the planner asks of the sidecar it started: this arm's descriptor, no hand in it, this cell's hand added,
        and a committed evidence file that measured exactly that combination.

        The descriptor check comes first because it says which robot the sidecar loaded, and the evidence
        check then asks whether anybody measured that robot. A sidecar that reports no hashes is refused
        rather than trusted, because an unreported hash is not a hash that matched.
        """
        from src.robot.safety.planning._curobo_body_links import BodyLinkError
        from src.robot.safety.planning.body_link import HandLink, wrist_body_refusal
        from src.robot.safety.planning.evidence import planner_evidence_refusal
        from src.robot.safety.planning.hand import descriptor_refusal
        from src.robot.safety.planning.margin import declared_planner_margin
        from src.robot.safety.planning.reservation import PlannerReservation

        hand = self._preflight.planner_hand(self)
        model = str(self.config.ur.model)
        if not chosen(hand):
            return lambda identity: (
                "robot.gripper.model is unset, so nothing the planner loaded can be checked against the hand this cell "
                "carries"
            )
        try:
            link = HandLink.from_hand(hand)
        except BodyLinkError as exc:
            unplaced = str(exc)
            return lambda identity: unplaced
        sc = self.config.safety.self_collision
        declared = declared_planner_margin(sc)
        attach = PlannerReservation.from_config(robot_cfg=self.config).sphere_slots

        def check(identity: "SidecarIdentity") -> "str | None":
            said = descriptor_refusal(identity, link, arm=model)
            if said is not None:
                return said
            # The wrist cameras the sidecar loaded are exactly this arm's, and their cover is proven
            # complete: the evidence below is keyed without them.
            said = wrist_body_refusal(identity, self._preflight.wrist_bodies(self))  # type: ignore[arg-type]
            if said is not None:
                return said
            if not chosen(declared):
                return ("safety.self_collision.planner_margin_mm is undeclared, so no evidence can be looked up: "
                        "the margin is part of what a file measured")
            return planner_evidence_refusal(
                identity, hand, link, arm=model, planner_margin_mm=float(declared),
                guard_margin_mm=float(sc.min_distance_mm), attach_spheres=int(attach), mesh_dir=sc.mesh_dir,
            )

        return check

    async def amove_to(
        self,
        pose: Pose | URPose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
    ) -> bool:
        """Awaitable variant of :meth:`move_to`, gated and commanded the same way.

        On a cuRobo cell it runs the sync twin in a worker thread rather than mirroring its
        body. The plan and the sampled line are both synchronous and both talk to the
        sidecar and to the controller; a second async body beside them is how this verb
        drifted from its twin twice already.
        """
        if self._motion_planner == "curobo":
            return await asyncio.to_thread(
                self.move_to, pose, linear=linear, vel=vel, acc=acc, register=register,
            )
        rejected = self._gate_pose(self._pose_from_any(pose), MotionCommand.MOVE_TO)
        if rejected is not None:
            self.logger.error("amove_to REFUSED by %s: %s", rejected.status, rejected.message)
            return False
        return await self._motion.amove_to(
            self._to_controller_urpose(pose),
            linear=linear, vel=vel, acc=acc, register=register,
            workspace_pose=self._coerce_urpose(pose),
        )

    async def amove_home(self) -> bool:
        """The awaitable variant of :meth:`move_home`, gated the same way.

        Routing straight to ``MotionController.amove_home`` instead would log that it is
        proceeding anyway where the home pose lies outside the workspace box, and
        command the move regardless with no joint-limit, self-collision or payload check
        at all, which would leave this path strictly weaker than the method it mirrors.
        Both verbs run :meth:`move_to_home`, which runs synchronously.

        It runs the whole sync body in a worker thread rather than mirroring it: two bodies
        that have to stay the same is how this verb drifted apart from its twin in the
        first place.
        """
        return await asyncio.to_thread(self.move_home)

    def fk(self, joints: JointPositions) -> Pose:
        """Forward kinematics: :class:`JointPositions` to TCP :class:`Pose`."""
        if not self._conn.is_connected:
            raise RobotConnectionError("fk() requires an open connection.")
        if joints.dof != self.capabilities.dof:
            raise RobotKinematicsError(
                f"fk() expected {self.capabilities.dof} DoF, got {joints.dof}."
            )
        try:
            tcp_m = self._conn.fk(joints.tolist())
        except (RuntimeError, OSError) as exc:
            raise RobotKinematicsError(f"FK failed: {exc}") from exc
        urpose = URPose.from_ur_list(tcp_m, label="fk")
        return self._pose_from_controller(urpose, label="fk")

    def ik(
        self,
        pose: Pose,
        *,
        seed: JointPositions | None = None,
    ) -> JointPositions:
        """Inverse kinematics: TCP :class:`Pose` in :attr:`Frame.BASE` to joints."""
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"ik() requires pose in Frame.BASE; got {pose.frame!r}."
            )
        if not self._conn.is_connected:
            raise RobotConnectionError("ik() requires an open connection.")
        urpose = self._pose_to_controller(pose)
        seed_list = seed.tolist() if seed is not None else None
        try:
            joints = self._conn.ik(urpose.to_ur_list(), q_near=seed_list)
        except (RuntimeError, OSError) as exc:
            raise RobotKinematicsError(f"IK failed: {exc}") from exc
        if not joints or len(joints) != self.capabilities.dof:
            raise RobotKinematicsError(
                f"IK returned no valid solution for pose '{pose.label or '<unlabeled>'}'."
            )
        return JointPositions(joints)

    def nearest_configuration(self, pose: Pose) -> JointPositions:
        """The configuration a move to ``pose`` would take the arm to, as the endpoint gate judges it; nothing moves.

        :class:`~src.robot.core.arm_capabilities.ChoosesConfigurations`, for a caller that asks where the arm can
        stand before it spends a motion on it, such as the generated view of a wrist pick screening the angles of its
        orbit. It is the screening half of the route a Cartesian cuRobo move takes (:meth:`_route_to_the_nearest_goal`),
        shared and not copied, so the answer is the configuration a move to ``pose`` screens first:

        1. The joints the arm stands at, read from the controller.
        2. The flange goal of ``pose`` (:meth:`_pose_to_flange`) solved in closed form, every solution turned onto
           the full turns nearest the arm inside :meth:`_goal_joint_window`: the planner's window narrowed to the
           joint-limit guard's, which is the cable window half a turn either side of home where the cell keeps one.
        3. The branch the arm holds first, then the others, each by the least time a ``moveJ`` there takes.
        4. The first of them the endpoint gate admits (:meth:`_screen_planned_config`: the workspace box on the
           TCP, the joint limits, self-collision, the payload). The box is asked here because a joint move to the
           answer exempts it (``RobotArm.move_to_joints``), and a camera placed outside the box is still outside it.
           The box reads the TCP alone, which every configuration of ``pose`` puts in the same place, so the first
           configuration it refuses is every configuration's refusal and no other is asked.

        What a move does next is not done: no camera world is refreshed, the planner is not asked, and no line or
        path is judged, so the motion to a configuration named here is still judged whole when it runs and can be
        refused then. Nothing is commanded, so nothing is remembered as if it had been: the continuity memo the next
        move is judged from stays where the last commanded target left it, and a configuration the gate refuses is
        logged at DEBUG, not as a refused motion. It answers on an ik cell too, because the closed form needs no
        planner. One speed for every joint orders the goals the same whatever it is, so the configured maximum
        stands in for a move's; only two configurations equally near to the last bit, which serve alike, can be
        taken in the other order. Inside :meth:`keeping_its_branch` only the configurations on the branch the arm holds
        are asked, as a move there routes to no other.

        Raises ``RobotKinematicsError`` whose message says ``refused by`` what refused ``pose``: the inverse
        kinematics (out of reach, or a model with no DH chain), the joint window (no configuration inside it), the
        branch a :meth:`keeping_its_branch` block keeps (none on it inside the window), or the
        endpoint gate (the box, or every configuration refused, the first one's refusal quoted). Raises
        ``RobotConnectionError`` where the controller cannot say where the arm stands, a reading that failed or one
        that is no configuration of this arm (a joint that is not finite, another number of joints), which is no
        statement about the pose; and ``FrameMismatchError`` for a pose outside BASE, as :meth:`ik` does.
        """
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"nearest_configuration() requires pose in Frame.BASE; got {pose.frame!r}."
            )
        if not self._conn.is_connected:
            raise RobotConnectionError("nearest_configuration() requires an open connection.")
        try:
            here = [float(v) for v in self._conn.get_joint_positions()]
        except (RobotConnectionError, RuntimeError, OSError) as exc:
            raise RobotConnectionError(
                f"the configuration a pose goes to is chosen from where the arm stands, and the controller's joint "
                f"positions could not be read ({type(exc).__name__}: {exc})"
            ) from exc
        dof = self.capabilities.dof
        if len(here) != dof or not all(math.isfinite(v) for v in here):
            # A lost or garbled controller is never mistaken for an angle the arm cannot reach.
            raise RobotConnectionError(
                f"the configuration a pose goes to is chosen from where the arm stands, and the controller's joint "
                f"reading is no configuration of this {dof}-joint arm: {here!r}"
            )
        label = pose.label or "<unlabeled>"
        goals = self._goals_in_window(self._pose_to_flange(pose), pose, here,
                                      velocity=float(self.config.motion_limits.max_velocity))
        if isinstance(goals, MotionResult):
            by = "the joint window" if goals.status is MotionStatus.JOINT_LIMIT_REJECTED else "the inverse kinematics"
            raise RobotKinematicsError(f"{label}: refused by {by}: {goals.message}")
        ranked = self._ranked_goals(here, goals)
        asked = ranked.groups[0] if self._keeping is not None else ranked.in_order
        if not asked:
            raise RobotKinematicsError(
                f"{label}: refused by the branch the arm keeps: none of the {len(ranked.in_order)} configuration(s) "
                f"inside the joint window is on {_branch_text(ranked.held)}, and this move is {_KEEPS_ITS_BRANCH}")
        refusals: list[MotionResult] = []
        for candidate in asked:
            refused = self._screen_planned_config(pose, JointPositions(candidate.joints))
            if refused is None:
                self.logger.debug("nearest configuration of %s: %s, after %d refused by the endpoint gate",
                                  label, ranked.names[candidate], len(refusals))
                return JointPositions(candidate.joints)
            refusals.append(refused)
            if refused.status is MotionStatus.WORKSPACE_REJECTED:
                why = (f"the workspace box refuses the TCP, which each of its {len(asked)} configuration(s) "
                       f"inside the joint window puts in the same place, {refused.status.value}: {refused.message}")
                break
        else:
            first = refusals[0]
            why = (f"all {len(refusals)} configuration(s) inside the joint window were refused, the first in the arm's "
                   f"order {first.status.value}: {first.message}")
        self.logger.debug("no configuration for %s: refused by the endpoint gate: %s", label, why)
        raise RobotKinematicsError(f"{label}: refused by the endpoint gate: {why}")

    def screen_configuration(self, joints: JointPositions, *, ask_planner: bool = True) -> PoseScreen:
        """What the exact guard and the planner say about ``joints`` before any move goes there; nothing moves.

        The owner, 2026-09-30: a pose is screened where it is taught, where a campaign starts and at the desk, so a pose
        only the planner's padded spheres refuse is known before a pick meets it (:mod:`src.robot.safety.planning.band`).
        The configuration is the one a joint move would take, on its turn inside the goal window. The verdict:

        * ``GUARD_REFUSED``: the exact guard (with the joint-limit and payload guards) refuses it; no move goes there;
        * ``CLEAR``: the planner clears it too;
        * ``BAND``: the planner's padded spheres refuse it on pairs the exact guard judges and accepts; the arm goes there
          and out of it on straight lines, and a planned move out of it or into it takes a short escape leg;
        * ``SEEN_BOXES``: the planner's world refuses it on the boxes the camera saw alone, the robot itself at most on
          pairs the exact guard decides, and the exact guard accepts it with those boxes (Option 1): asked again as a
          move asks it (:meth:`_world_aside_refusal`, :meth:`_world_set_aside`), for a hand known empty and open: the
          screen reads no hand. Straight lines and moveL go there and out of it with such a hand; a planned move into
          it or out of it is refused, and so is any move while the hand carries a part or cannot say it stands open;
        * ``PLANNER_REFUSED``: the planner refuses it for what only it judges (its world beyond the camera's boxes, its
          bounds, the carried part, a pair the guard does not check); no move goes there while it does;
        * ``UNSCREENED``: the planner could not be asked, or ``ask_planner`` is false; the exact guard's verdict is said.

        Where it is not clear, ``nearby`` is the nearest configuration within
        :data:`~src.robot.safety.planning.band.BAND_LEG_MAX_DEG` per joint both clear: for a ``BAND`` pose the pose a
        planned move's leg goes to on its own (nothing to re-teach), for a ``SEEN_BOXES`` pose one planned moves and a
        carried part reach, for an ERROR the pose to re-teach at. The planner is started where it is not running (about a
        minute on a cell), and it holds the world it was last given: the camera's boxes of its last refresh, if any.
        """
        if self._motion_planner != "curobo" or self._preflight is None:
            return PoseScreen(PoseVerdict.UNSCREENED, (
                f"robot.ur.motion_planner is {self._motion_planner!r}"
                f"{' and no safety preflight is wired' if self._preflight is None else ''}, so no planner judges it."))
        values = [float(v) for v in joints.tolist()]
        target, _ = self._twin_near_the_arm(JointPositions(values), values)
        config = [float(v) for v in target.tolist()]
        exact = self._preflight.exact_pairs(self)
        denied = self._preflight.gate_joint_target(JointPositions(config), arm=self)
        if not ask_planner:
            said = "the exact guard refuses it" if denied is not None else "the exact guard accepts it"
            return PoseScreen(PoseVerdict.GUARD_REFUSED if denied is not None else PoseVerdict.UNSCREENED,
                              f"{said}{': ' + str(denied.message) if denied is not None else ''}; the planner was "
                              "not asked.")
        try:
            planner = self._curobo_ur_planner()
            report = planner.judge_joint_path([config], clearance_mm=0.0)
        except CuroboUnavailableError as exc:
            said = (f"the exact guard refuses it: {denied.message}" if denied is not None
                    else "the exact guard accepts it")
            return PoseScreen(PoseVerdict.GUARD_REFUSED if denied is not None else PoseVerdict.UNSCREENED,
                              f"{said}; the planner could not be asked: {exc}", planner_unavailable=True)
        row = report.refused[0] if report.refused else None
        nearby = (self._screened_nearby(planner, exact, config, row)
                  if denied is not None or row is not None else None)
        if denied is not None:
            return PoseScreen(PoseVerdict.GUARD_REFUSED, f"{denied.message}. No move goes there.", nearby=nearby)
        if row is None:
            return PoseScreen(PoseVerdict.CLEAR, "the exact guard and the planner both clear it.")
        why = (admission_refusal(row, exact, margin_mm=self._planner_margin_mm(), world_set_aside=True)
               if exact is not None else "no exact mesh guard runs here")
        if why is None and not row.world_ok:
            # A move whose sample the planner's world refuses asks it again with the camera's boxes set aside (Option 1),
            # and so does the screen, so that it says what the move does: the same conditions, at the same clearance.
            # It judges the pose, for a hand known empty and open, and reads no hand: at a desk there may be none.
            why = self._world_aside_refusal(planner, hand=False)
            if why is None:
                standing = self._world_set_aside(planner, [config], [row], clearance_mm=0.0)
                why = None if standing is None else standing.reason
            if why is None and exact is not None:
                return PoseScreen(PoseVerdict.SEEN_BOXES, self._seen_boxes_sentence(row, exact, config), nearby=nearby)
        if why is None and exact is not None:
            planned = (f"a planned move out of it or into it takes a straight leg of at most {BAND_LEG_MAX_DEG:g} deg "
                       "per joint to the nearest pose both clear first" if nearby is not None else
                       f"no pose within {BAND_LEG_MAX_DEG:g} deg per joint of it clears both, so a planned move out of "
                       "it or into it is refused: teach it where both clear, or reach it and leave it on straight "
                       "lines")
            return PoseScreen(PoseVerdict.BAND, (
                f"{band_sentence(row, exact, config)}. Straight lines run into it and out of it; {planned}."),
                nearby=nearby)
        return PoseScreen(PoseVerdict.PLANNER_REFUSED, f"{row.render()}: {why}. No move goes there while it does.",
                          nearby=nearby)

    def _seen_boxes_sentence(self, row: RefusedSample, exact: ExactPairs, config: "Sequence[float]") -> str:
        """What a ``SEEN_BOXES`` pose is and what runs there: the camera's boxes alone refuse it on the planner's world,
        the band beside that where the planner refuses the robot itself too, and the moves that run and those that don't."""
        seen = self._preflight.perceived_obstacles(self) if self._preflight is not None else ()
        keep = float(self.config.safety.self_collision.perceived_min_distance_mm)
        found = (f"the planner's world refuses it only on the {len(seen)} box(es) the camera saw, which the exact guard "
                 f"keeps {keep:g} mm from and accepts here")
        if not row.self_ok:
            found += f"; and {band_sentence(row, exact, config)}"
        return (f"{found}. Straight lines and moveL run into it and out of it with a hand known empty and open; a planned "
                "move into it or out of it is refused, and so is any move into it or out of it while the hand carries a "
                "part or cannot say it stands open, so a grasp here closes only where its lift would run carrying.")

    def _screened_nearby(
        self, planner: CuroboUrPlanner, exact: object, config: "Sequence[float]", row: object,
    ) -> "tuple[float, ...] | None":
        """The nearest configuration both authorities clear within the band leg's reach of ``config``, or ``None``.

        The joints turned are those that move one link of the planner's pairs against the other where it named some,
        every joint otherwise; the planner clears them at the line clearance, the exact guard accepts the first. It is
        the screen's word only: for a band pose where a planned move's own leg would go (nothing to re-teach), for a
        pose beside the camera's boxes one planned moves and a carried part reach, in the planner's whole world, for an
        ERROR pose the pose to re-teach at. No leg to it is judged here: the screen drives nothing.
        """
        radii = self._preflight.joint_radii_mm(self) if self._preflight is not None else None
        if radii is None:
            return None
        pairs = tuple(getattr(row, "pairs", ()) or ()) if row is not None else ()
        turned = joints_between(pairs, exact) if pairs and exact is not None else tuple(range(len(config)))  # type: ignore[arg-type]
        candidates = band_neighbours(config, turned, radii)
        if not candidates:
            return None
        try:
            report = planner.judge_joint_path([list(c) for c in candidates], clearance_mm=self._line_clearance_mm(),
                                              name_pairs=False)
        except CuroboUnavailableError:
            return None
        refused = {refusal.index for refusal in report.refused}
        for index, candidate in enumerate(candidates):
            if index not in refused and self._exact_guard_accepts(candidate, MotionCommand.MOVE_JOINTS):
                return tuple(float(v) for v in candidate)
        return None

    @property
    def capabilities(self) -> RobotCapabilities:
        """Static feature flags advertised by this driver."""
        return self._capabilities

    def get_tcp_pose(self) -> Pose:
        """Current TCP pose, tagged :attr:`Frame.BASE`."""
        if not self._conn.is_connected:
            raise RobotConnectionError("get_tcp_pose() requires an open connection.")
        urpose = self._motion.get_current_pose()
        return self._pose_from_controller(urpose, label="current")

    def get_joint_positions(self) -> JointPositions:
        """Current joint configuration as a typed vector."""
        if not self._conn.is_connected:
            raise RobotConnectionError(
                "get_joint_positions() requires an open connection."
            )
        return JointPositions(self._conn.get_joint_positions())

    def move_to_joints(
        self,
        joints: JointPositions,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
        camera_world: Maybe[CameraWorldDecline] = UNSET,
    ) -> MotionResult:
        """The typed joint move: judge it (:meth:`_judge_joint_move`), then drive exactly what was judged.

        On cuRobo the target goes onto the full turns nearest the arm inside the cable window first, the same
        pose; the straight joint line runs where both authorities pass it, and where it collides cuRobo plans
        around it. The result names the configuration that ran.

        The result says what stood behind the motion as :meth:`move` says it: UNPLANNED on the ik
        planner, and on cuRobo DECLINED for a decline, MISSING while no live camera world is wired,
        and once one is, PLANNED when the refresh this motion made vouched for the cell and UNSTATED
        when none did: the line and any plan around it are judged against that refreshed world. A
        declined joint move on an arm whose live camera world is wired is refused with ``UNSUPPORTED``
        before its path is judged, and so is a MISSING joint move, with neither a world nor a decline.
        """
        self._forget_what_was_judged_ahead()
        ahead = self._take_the_next_leg(joints)
        stamp = self._move_camera_world(camera_world)
        refused = self._refused_for_camera_world(stamp, MotionCommand.MOVE_JOINTS, target_joints=joints)
        if refused is not None:
            return refused
        before = self._planner_refresh()
        self._vouched_by = None
        unstamped = self._move_to_joints_unstamped(
            joints, velocity=velocity, acceleration=acceleration, ahead=ahead,
        )
        after = self._planner_refresh()
        ran, self._vouched_by = self._vouched_by, None
        # A move judged ahead in a held world (judge_the_next_leg) is vouched for by the refresh it was judged in, as a
        # route judged ahead is (move).
        vouched = (after.camera_world() if after is not None and (after is not before or (ran is not None and ran is after))
                   else None)
        return stamp_result(unstamped, self._move_camera_world(camera_world, planned=vouched))

    def _move_to_joints_unstamped(
        self,
        joints: JointPositions,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
        ahead: "_NextLegJudged | None" = None,
    ) -> MotionResult:
        """The body of :meth:`move_to_joints`. It stamps nothing; the public verb does. ``ahead`` is the move judged ahead
        to these joints (:meth:`judge_the_next_leg`), which runs as judged where nothing it was judged on changed."""
        ran = self._run_the_next_leg(ahead, command=MotionCommand.MOVE_JOINTS, done="move_to_joints",
                                     velocity=velocity, acceleration=acceleration)
        if ran is not None:
            return ran
        target, route, refused = self._judge_joint_move(joints, command=MotionCommand.MOVE_JOINTS)
        if refused is not None:
            return refused
        return self._drive_judged_joints(
            target, route, command=MotionCommand.MOVE_JOINTS, done="move_to_joints",
            velocity=velocity, acceleration=acceleration,
        )

    def move_through_joints_on_the_line(
        self,
        waypoints: "Sequence[JointPositions]",
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
        camera_world: Maybe[CameraWorldDecline] = UNSET,
    ) -> MotionResult:
        """Straight joint lines through every one of ``waypoints``, judged whole before anything is sent, then sent
        waypoint after waypoint: :class:`~src.robot.core.arm_capabilities.DrivesJointPaths`.

        Judged as :meth:`move_to_joints_on_the_line` judges one line, once for the whole path: one world refresh, every
        waypoint turned onto the twin nearest the one before it and through the destination guards, every leg by the
        exact guard (mesh first, ``safety.planned_motion.mesh_first``) and, where that does not apply, the planner at
        ``line_clearance_mm``. A leg that is not clear refuses the whole path, nothing sent and nothing planned around
        it. The path then runs as a judged cuRobo plan runs, one ``moveJ`` per waypoint with nothing judged between
        them, so the arm turns at each and runs on; a halt ends it where it is. No blend: a blended corner leaves the
        legs the guard judged.
        """
        self._forget_what_was_judged_ahead()
        last = waypoints[-1] if waypoints else None
        stamp = self._move_camera_world(camera_world)
        refused = self._refused_for_camera_world(stamp, MotionCommand.MOVE_JOINTS, target_joints=last)
        if refused is not None:
            return refused
        before = self._planner_refresh()
        unstamped = self._drive_joint_path(list(waypoints), velocity=velocity, acceleration=acceleration)
        after = self._planner_refresh()
        held = self._held_world_reason is not None
        vouched = after.camera_world() if after is not None and (after is not before or held) else None
        return stamp_result(unstamped, self._move_camera_world(camera_world, planned=vouched))

    def _drive_joint_path(
        self, waypoints: "list[JointPositions]", *, velocity: float | None, acceleration: float | None,
        judgements: int = 1,
    ) -> MotionResult:
        """The body of :meth:`move_through_joints_on_the_line`: judge the whole path, then send it. Where this arm gates
        its own sends, the steady gate waits right before the send, and a path whose arm came to rest more than
        :data:`_AHEAD_MOVED_MM` from where it was judged from is judged again from there (``judgements`` counts them)."""
        command = MotionCommand.MOVE_JOINTS
        if not waypoints:
            return MotionResult.failed(MotionStatus.INVALID_TARGET, command,
                                       message="a joint path needs at least one waypoint; nothing was sent")
        last = waypoints[-1]
        if self._preflight is None or self._motion_planner != "curobo":
            return MotionResult.failed(
                MotionStatus.UNSUPPORTED, command, target_joints=last,
                message=(f"a joint path only runs where every leg is judged, and nobody judges the paths of this arm "
                         f"(robot.ur.motion_planner {self._motion_planner!r}"
                         f"{', no safety preflight' if self._preflight is None else ''}); nothing was sent"),
            )
        if not self._conn.is_connected:
            return MotionResult.failed(MotionStatus.CONNECTION_ERROR, command, target_joints=last,
                                       message="a judged joint path starts at the current configuration, which needs "
                                               "an open connection.")
        try:
            here = [float(v) for v in self._conn.get_joint_positions()]
        except (RobotConnectionError, RuntimeError, OSError) as exc:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR, command, target_joints=last,
                message=(f"a judged joint path starts at the current configuration, and the controller's joint "
                         f"positions could not be read ({type(exc).__name__}: {exc}); nothing was sent"),
                exception=exc,
            )
        targets: list[JointPositions] = []
        previous = here
        for waypoint in waypoints:
            if waypoint.dof != self.capabilities.dof:
                return MotionResult.failed(
                    MotionStatus.INVALID_TARGET, command, target_joints=waypoint,
                    message=f"a joint path waypoint has {waypoint.dof} DoF, the arm {self.capabilities.dof}; nothing was "
                            "sent")
            target, _turned = self._twin_near_the_arm(waypoint, previous)
            targets.append(target)
            previous = target.tolist()
        refused = self._refresh_before_the_path_gate(
            near_point_mm=self._flange_mm(targets[-1]), command=command, target_joints=targets[-1],
            goal_tcp_mm=self._tcp_at_joints_mm(targets[-1]),
        )
        if refused is not None:
            return refused
        for target in targets:
            refused = self._preflight.gate_joint_target(target, arm=self)
            if refused is not None:
                return dataclasses.replace(refused, command=command)
        try:
            planner = self._curobo_ur_planner()
        except CuroboUnavailableError as exc:
            return MotionResult.failed(MotionStatus.CONTROLLER_REJECTED, command, target_joints=last,
                                       message=f"cuRobo planner unavailable: {exc}", exception=exc)
        path = [here] + [t.tolist() for t in targets]
        refused = self._judge_legs(planner, path, command=command, target_joints=targets[-1],
                                   clearance_mm=self._line_clearance_mm())
        if refused is not None:
            return dataclasses.replace(
                refused, message=f"{refused.message}; a joint path only: nothing is planned around it, nothing was sent")
        gate = self._send_gate(here, command, target_joints=targets[-1])
        if isinstance(gate, MotionResult):
            return gate
        if gate is not None:
            if judgements >= self._SEND_JUDGEMENTS:
                return self._kept_moving(command, target_joints=targets[-1])
            return self._drive_joint_path(waypoints, velocity=velocity, acceleration=acceleration,
                                          judgements=judgements + 1)
        self._log_the_route("move_through_joints_on_the_line",
                            self._route(path, planned=False, how=f"one joint path through {len(targets)} waypoint(s)"))
        vel, acc = self._motion._clamp(velocity, acceleration)
        # The executor of a judged plan: one moveJ per waypoint, the halt read before each, a failure saying how far.
        result = planner.execute([t.tolist() for t in targets], command=command, target_joints=targets[-1],
                                 vel=vel, acc=acc)
        return dataclasses.replace(result, message="move_through_joints_on_the_line") if result.ok else result

    def move_to_joints_on_the_line(
        self,
        joints: JointPositions,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
        camera_world: Maybe[CameraWorldDecline] = UNSET,
    ) -> MotionResult:
        """:meth:`move_to_joints` on the straight joint line only: a line that is not clear is refused, never planned around.

        :class:`~src.robot.core.arm_capabilities.DrivesJointLines`, for the motions the owner allows on the line from
        where the arm stands and nowhere else (2026-09-29): the view a wrist pick generates, and its move back to the
        look that saw the part. They must never drive through a retract pose or take a detour nobody taught, so cuRobo
        is never asked to plan one, at any angle.

        Judged exactly as :meth:`move_to_joints` judges a joint move (:meth:`_judge_joint_move`): the camera world
        refreshed with every frame the pick holds, the target on its twin inside the cable window, the destination
        guards, and the straight joint line by both authorities at ``safety.planned_motion.line_clearance_mm``. A clear
        line runs as one ``moveJ`` to the target. A line either authority refuses is refused with that refusal and the
        sentence that nothing is planned around it; nothing is sent. The camera world is stamped as on
        :meth:`move_to_joints`. An arm whose paths nobody judges, the ik planner or no preflight, refuses it
        ``UNSUPPORTED`` before anything is sent: a line nobody judged is not the line the owner allows.
        """
        self._forget_what_was_judged_ahead()
        stamp = self._move_camera_world(camera_world)
        refused = self._refused_for_camera_world(stamp, MotionCommand.MOVE_JOINTS, target_joints=joints)
        if refused is not None:
            return refused
        before = self._planner_refresh()
        target, route, unstamped = self._judge_joint_move(joints, command=MotionCommand.MOVE_JOINTS, plan_around=False)
        if unstamped is None:
            unstamped = self._drive_judged_joints(
                target, route, command=MotionCommand.MOVE_JOINTS, done="move_to_joints_on_the_line",
                velocity=velocity, acceleration=acceleration, plan_around=False,
            )
        after = self._planner_refresh()
        vouched = after.camera_world() if after is not None and after is not before else None
        return stamp_result(unstamped, self._move_camera_world(camera_world, planned=vouched))

    def _drive_judged_joints(
        self,
        joints: JointPositions,
        route: "_Route | None",
        *,
        command: MotionCommand,
        done: str,
        velocity: float | None = None,
        acceleration: float | None = None,
        plan_around: bool = True,
    ) -> MotionResult:
        """Run the joint move that was just judged, and type what the controller answered.

        ``route`` is what :meth:`_judge_joint_move` passed: ``None`` or a straight line is one ``moveJ`` to
        ``joints``, the target that was judged, and a cuRobo plan is one ``moveJ`` per waypoint of the list that was
        judged. The ungated primitives, because the move was just judged; calling the public ``move_joint`` here
        would run the same guards a second time for nothing. A drive that fails is a result and never a raise:
        :meth:`_drive_joints` carries the typed refusal on its raise, whose message says whether ``moveJ`` had been
        sent, and a raise that carries none still reads CONTROLLER_REJECTED rather than a status nobody could
        classify. ``done`` is the message of a move that ran.

        Where this arm gates its own sends (``safety.dwell.gate_at`` ``send``), the steady gate waits here, right
        before the send (:meth:`_send_gate`), and a move whose arm came to rest more than :data:`_AHEAD_MOVED_MM` from
        where it was judged from is judged again from where it stands, as it was asked (``plan_around``), before
        anything is sent.
        """
        for _ in range(self._SEND_JUDGEMENTS):
            gate = self._send_gate(None if route is None else route.waypoints[0], command, target_joints=joints)
            if isinstance(gate, MotionResult):
                return gate
            if gate is None:
                break
            joints, route, refused = self._judge_joint_move(joints, command=command, plan_around=plan_around)
            if refused is not None:
                return refused
        else:
            return self._kept_moving(command, target_joints=joints)
        if route is not None:
            self._log_the_route(done, route)
        if route is not None and route.planned:
            vel, acc = self._motion._clamp(velocity, acceleration)
            result = self._curobo_ur_planner().execute(
                self._sent_waypoints(route), command=command, target_joints=joints, vel=vel, acc=acc,
            )
            return dataclasses.replace(result, message=done) if result.ok else result
        try:
            self._drive_joints(joints, velocity=velocity, acceleration=acceleration, command=command)
        except RobotConnectionError as exc:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR, command,
                target_joints=joints, message=str(exc), exception=exc,
            )
        except RobotMotionRejected as exc:
            if isinstance(exc.result, MotionResult):
                return exc.result
            return MotionResult.failed(
                MotionStatus.CONTROLLER_REJECTED, command, target_joints=joints,
                message=f"the drive refused after the move was judged, and moveJ may have been sent: {exc}",
                exception=exc,
            )
        return MotionResult.executed(command, target_joints=joints, message=done)

    def _judge_joint_move(
        self, joints: JointPositions, *, command: MotionCommand, plan_around: bool = True,
    ) -> "tuple[JointPositions, _Route | None, MotionResult | None]":
        """Judge a joint move before anything is commanded, as ``(target, route, refusal)``.

        ``refusal`` is ``None`` where the move may run, and then ``target`` is the configuration it ends at and
        ``route`` the judged way there, for :meth:`_drive_judged_joints`. On an ik cell ``target`` is ``joints``,
        ``route`` is ``None`` and the judge is the destination gate and nothing else: no planner produced a path,
        and ``moveJ`` interpolating between two configurations is a line nothing in this repository models.

        On a cuRobo cell it is the whole path from where the arm is standing, judged twice over, because the two
        authorities see different cells. The local guards carry the declared fixtures and the exact link meshes;
        the planner carries the camera world and whatever is attached to the tool. Either one alone would miss what
        the other holds. In order:

        1. A joint outside the window the planner and the joint-limit guard admit (the cable window about home where
           the cell keeps one) goes onto its full turn nearest the arm inside it (:meth:`_twin_near_the_arm`). A full
           turn of a joint is the same pose, so this changes which number is commanded and nothing about where the
           arm ends; where it changes one, the log says so.
        2. The destination guards on that target (``gate_joint_target``): a target no turn of which the window
           admits, or one past a full turn, which reads as degrees, is refused here by the joint-limit guard before
           any line is sampled or any plan asked for.
        3. The straight joint line to it, judged by both authorities, the planner at
           ``safety.planned_motion.line_clearance_mm`` from its world. Passing, it runs as one ``moveJ``.
        4. A line that collides (``SELF_COLLISION_REJECTED``, the status of the robot, the fixtures and the
           planner's world alike) is planned around by cuRobo in joint space, to the same target. The plan is taken
           where no joint swings more than ``safety.planned_motion.max_detour_deg`` beyond its span, else the move is
           refused ``JOINT_LIMIT_REJECTED`` naming the joint, and its legs meet both authorities as a Cartesian
           plan's do. Any other refusal of the line stands.

        ``plan_around`` false is the straight joint line only (:meth:`move_to_joints_on_the_line`): step 4 is never
        taken, and a line that collides is refused as it was, with the sentence that nothing is planned around it. An
        arm that judges no line then refuses before anything else, ``UNSUPPORTED``: with no preflight, or on the ik
        planner, whose ``moveJ`` interpolates a line nothing here models.

        The connection is read first, because a path starts at the current configuration and a disconnected arm has
        none to give. A read that fails is CONNECTION_ERROR too: the transport raises ``RuntimeError`` or ``OSError``
        rather than a robot error, and it would otherwise leave the verb as a raw exception. With a live camera world,
        the world is refreshed next, before either authority judges: the local guard learns the perceived obstacles
        from that refresh, and a refresh inside the planner's check would run after the guard, which would then judge
        against the motion before.
        """
        if not plan_around and (self._preflight is None or self._motion_planner != "curobo"):
            return joints, None, MotionResult.failed(
                MotionStatus.UNSUPPORTED, command, target_joints=joints,
                message=(f"a straight joint line only runs where it is judged, and nobody judges the paths of this arm "
                         f"(robot.ur.motion_planner {self._motion_planner!r}"
                         f"{', no safety preflight' if self._preflight is None else ''}); nothing was sent"),
            )
        if self._preflight is None:
            return joints, None, None
        if self._motion_planner != "curobo":
            return joints, None, self._preflight.gate_joint_target(joints, arm=self)

        if not self._conn.is_connected:
            return joints, None, MotionResult.failed(
                MotionStatus.CONNECTION_ERROR, command, target_joints=joints,
                message="a judged joint move starts at the current configuration, which needs an "
                        "open connection.",
            )
        try:
            here = [float(v) for v in self._conn.get_joint_positions()]
        except (RobotConnectionError, RuntimeError, OSError) as exc:
            return joints, None, MotionResult.failed(
                MotionStatus.CONNECTION_ERROR, command, target_joints=joints,
                message=(f"a judged joint move starts at the current configuration, and the controller's "
                         f"joint positions could not be read ({type(exc).__name__}: {exc}); nothing was sent"),
                exception=exc,
            )
        target, turned = self._twin_near_the_arm(joints, here)
        refused = self._refresh_before_the_path_gate(
            near_point_mm=self._flange_mm(target), command=command, target_joints=target,
            goal_tcp_mm=self._tcp_at_joints_mm(target),
        )
        if refused is not None:
            return target, None, refused
        refused = self._preflight.gate_joint_target(target, arm=self)
        if refused is not None:
            if any(not abs(float(v)) <= self._TWIN_MAX_RAD for v in target.tolist()):
                refused = dataclasses.replace(refused, message=(
                    f"{refused.message} The target holds a value past a full turn, which no UR joint reaches in "
                    "radians and reads as degrees; joint targets are radians."))
            return target, None, dataclasses.replace(refused, command=command)
        planner = self._curobo_ur_planner()
        line = [here, target.tolist()]
        refused = self._judge_legs(planner, line, command=command, target_joints=target,
                                   clearance_mm=self._line_clearance_mm())
        if refused is None:
            return target, self._route(line, planned=False, how="direct line" + turned), None
        if not plan_around:
            return target, None, dataclasses.replace(
                refused, message=f"{refused.message}; a straight joint line only: nothing is planned around it")
        if refused.status is not MotionStatus.SELF_COLLISION_REJECTED:
            return target, None, refused
        try:
            trajectory, why, start, legs = self._planned(planner, target.tolist(), name="the target", here=here,
                                                         command=command, target_joints=target)
            if start is not None:
                return target, None, self._start_refusal(start, command, target_joints=target, note=why)
            if trajectory is None:
                return target, None, dataclasses.replace(
                    refused, message=f"{refused.message}; and no plan goes around it: {why}")
            _, detour = self._detour(trajectory)
            if detour:
                return target, None, MotionResult.failed(
                    MotionStatus.JOINT_LIMIT_REJECTED, command, target_joints=target,
                    message=(f"the straight line to this target is refused ({refused.message}), and cuRobo's plan "
                             f"around it {detour}; nothing was sent"),
                )
            waypoints, planned_refusal = self._judged_waypoints(
                planner, trajectory, command=command, target_joints=target, legs=legs,
            )
        except CuroboUnavailableError as exc:
            return target, None, MotionResult.failed(
                MotionStatus.CONTROLLER_REJECTED, command, target_joints=target,
                message=f"cuRobo planner unavailable: {exc}", exception=exc,
            )
        except (RobotConnectionError, RuntimeError, OSError) as exc:
            # The plan reads where the arm stands again, and a transport that fails there raises what the read above
            # turns into a typed result.
            return target, None, MotionResult.failed(
                MotionStatus.CONNECTION_ERROR, command, target_joints=target,
                message=(f"the controller's joint positions could not be read to plan around the straight line "
                         f"({type(exc).__name__}: {exc}); nothing was sent"),
                exception=exc,
            )
        if planned_refusal is not None:
            return target, None, planned_refusal
        how = f"cuRobo plan around the straight line, which was refused ({refused.status.value}: {refused.message})"
        if why:
            how += f"; {why}"
        return target, self._route(waypoints, planned=True, dense=len(trajectory), how=how + turned), None

    #: A joint value past this is past a full turn either way, which no UR joint reaches in radians: a target written in
    #: degrees by mistake. It is never turned onto a twin, so the joint-limit guard refuses it rather than the wrap
    #: making a plausible configuration out of it.
    _TWIN_MAX_RAD = 2.0 * math.pi + 1e-9

    def _twin_near_the_arm(self, joints: JointPositions, here: "Sequence[float]") -> "tuple[JointPositions, str]":
        """``joints`` with every joint outside the goal window on its full turn nearest ``here`` inside it, and what changed.

        The window is :meth:`_goal_joint_window`: the planner's, narrowed to the joint-limit guard's, which is the
        half turn either side of home where ``safety.joint_limits.within_half_turn_of_home`` is set. A full turn of a
        UR joint is the same pose, so a station stored as -200 degrees on a wrist runs as +160 inside a window of
        +-175 about home, rather than being refused for a cable it would not wind.

        What is left as given: a joint already inside the window, because that number is the turn the caller chose
        and the one the pendant will show (inside the cable window it is the only turn there is); a joint no turn of
        which lies inside the window, and a joint past a full turn either way (:data:`_TWIN_MAX_RAD`), a target written
        in degrees, both for the joint-limit guard to refuse; and a vector of the wrong length, for the drive to refuse.
        The second value is ``""`` where nothing changed, and otherwise a clause for the route's log line, which is
        also logged here.
        """
        values = [float(v) for v in joints.tolist()]
        lower, upper = self._goal_joint_window()
        if not len(values) == len(here) == len(lower):
            return joints, ""
        turned: dict[int, float] = {}
        for axis, (value, near, low, high) in enumerate(zip(values, here, lower, upper)):
            if low <= value <= high or not abs(value) <= self._TWIN_MAX_RAD:
                continue
            twin = nearest_turn(value, near=float(near), lower=low, upper=high)
            if twin is not None:
                turned[axis] = twin
        if not turned:
            return joints, ""
        from .curobo_motion import UR_ARM_JOINT_NAMES

        names = UR_ARM_JOINT_NAMES if len(values) == len(UR_ARM_JOINT_NAMES) else tuple(
            f"joint {i + 1}" for i in range(len(values)))
        said = ", ".join(f"{names[axis]} {math.degrees(values[axis]):.1f} -> {math.degrees(twin):.1f} deg"
                         for axis, twin in turned.items())
        self.logger.info("joint target turned onto the full turn inside the goal window nearest the arm, the same "
                         "pose: %s", said)
        for axis, twin in turned.items():
            values[axis] = twin
        return JointPositions(values), f"; twin wrap {said}"

    def _refresh_before_the_path_gate(
        self,
        *,
        near_point_mm: "list[float] | None",
        command: MotionCommand,
        target_joints: JointPositions | None = None,
        target_pose: Pose | None = None,
        goal_tcp_mm: "np.ndarray | None" = None,
    ) -> "MotionResult | None":
        """Refresh the live camera world before a path is judged; ``None`` means judging may start.

        ``goal_tcp_mm`` is the TCP at the path's goal, where the space between the jaws is left out
        of the world; ``None`` records why no region was placed.

        The local path guard learns the perceived obstacles from the refresh. Run inside the
        planner's check, after that guard, it would leave the guard judging this path against the
        obstacles of the motion before. It runs here instead, once, and the planner is asked with
        ``refresh=False``. Without a live world it does nothing, which is the unchanged path.

        A world that could not be refreshed refuses with CONTROLLER_REJECTED, the status of every
        planner refusal on this driver. A camera that stayed silent, blind or stale raises
        ``CameraWorldUnavailable`` out of here and out of the verb.
        """
        if self._live_world is None:
            return None
        held = self._planner_refresh() if self._held_world_reason is not None else None
        if held is not None:
            # The world the last refresh built stands (:meth:`held_world`): the planner and the guard hold it as it is.
            self.logger.info("the world is held (%s): judged against the %d box(es) the last refresh registered",
                             self._held_world_reason, int(getattr(held, "registered", 0) or 0))
            return None
        # A new world: nothing judged ahead in the one it replaces runs as judged.
        self._forget_what_was_judged_ahead()
        try:
            self._curobo_ur_planner().refresh_world(near_point_mm=near_point_mm, **self._goal_keep_out(goal_tcp_mm))
        except CuroboUnavailableError as exc:
            return MotionResult.failed(
                MotionStatus.CONTROLLER_REJECTED, command,
                target_joints=target_joints, target_pose=target_pose,
                message=f"cuRobo planner unavailable: {exc}", exception=exc,
            )
        return None

    #: How far a plan's last configuration may put the flange from the flange goal it was planned
    #: to: the tolerance the Isaac driver holds an executed cuRobo plan to (``drivers/sim/arm.py``
    #: ``_MOVE_POS_TOL_MM``, ``_MOVE_ORI_TOL_DEG``). A base half a turn out misses by about a
    #: metre and 180 degrees, far outside either.
    _PLAN_END_TOL_MM = 5.0
    _PLAN_END_TOL_DEG = 6.0

    def _plan_end_refusal(self, goal: Pose, last_joints: "Sequence[float]", pose: Pose) -> "MotionResult | None":
        """A refusal when the plan's last configuration does not put the flange on ``goal``, or ``None`` when it does.

        Read on the DH chain, which is the controller's own frame, so a planner that answers in
        another base is caught here rather than on the robot. ``goal`` is the flange goal the
        planner was handed and ``pose`` the TCP pose the caller asked for, which the refusal names.
        """
        from src.robot.safety._ur_kinematics import ur_link_transforms_mm

        model = str(self.config.ur.model)
        frames = ur_link_transforms_mm(model, np.asarray([float(v) for v in last_joints], dtype=np.float64))
        if frames is None:
            return MotionResult.failed(
                MotionStatus.CONTROLLER_REJECTED, MotionCommand.MOVE_TO, target_pose=pose,
                message=(f"robot.ur.model {model!r} has no DH chain here, so where this plan ends cannot be read, and "
                         "a plan whose end nobody read does not move"),
            )
        end = np.asarray(frames[-1], dtype=np.float64)
        target = np.asarray(goal.to_matrix(), dtype=np.float64)
        error_mm = float(np.linalg.norm(end[:3, 3] - target[:3, 3]))
        cosine = (float(np.trace(end[:3, :3].T @ target[:3, :3])) - 1.0) / 2.0
        error_deg = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        if error_mm <= self._PLAN_END_TOL_MM and error_deg <= self._PLAN_END_TOL_DEG:
            return None
        return MotionResult.failed(
            MotionStatus.CONTROLLER_REJECTED, MotionCommand.MOVE_TO, target_pose=pose,
            message=(f"the plan ends {error_mm:.1f} mm and {error_deg:.1f} deg from its goal (at most "
                     f"{self._PLAN_END_TOL_MM:g} mm and {self._PLAN_END_TOL_DEG:g} deg), read on the controller's own "
                     "kinematics, so nothing moved: a planner answering in another base than the controller's ends "
                     "here"),
        )

    def _goal_keep_out(self, goal_tcp_mm: "np.ndarray | None") -> "dict[str, Any]":
        """The goal region's keyword for a refresh, or nothing on an arm that holds no live world.

        A region is computed only where a world is refreshed, so an arm with no world keeps the
        unchanged path. An undeclared tool frame places no TCP, and the refresh records that no
        region was placed and why.
        """
        if self._live_world is None:
            return {}
        from src.robot.core.keep_out import GoalKeepOut
        from src.robot.safety.planning.self_envelope import goal_keep_out

        if goal_tcp_mm is None:
            reason = ("robot.gripper.tool_frame is undeclared, so the TCP at this goal is not known and no goal region "
                      "is placed") if self._declared_tool_matrix() is None else (
                "the TCP at this goal is not known, so no goal region is placed")
            return {"goal_keep_out": GoalKeepOut(region=None, reason=reason)}
        return {"goal_keep_out": goal_keep_out(self._preflight, self, goal_tcp_mm)}

    def _tcp_of_pose(self, pose: Pose) -> "np.ndarray | None":
        """A TCP goal as a 4x4, or ``None`` when the tool frame is undeclared and the pose is the flange's."""
        if self._declared_tool_matrix() is None:
            return None
        return np.asarray(pose.to_matrix(), dtype=np.float64)

    def _tcp_at_joints_mm(self, joints: JointPositions) -> "np.ndarray | None":
        """The TCP at ``joints``: the flange frame the chain ends in times the declared tool frame, or ``None``."""
        from src.robot.safety._ur_kinematics import ur_link_transforms_mm

        tool = self._declared_tool_matrix()
        frames = ur_link_transforms_mm(str(self.config.ur.model), np.asarray(joints.tolist(), dtype=np.float64))
        if tool is None or frames is None:
            return None
        return np.asarray(frames[-1], dtype=np.float64) @ tool

    def _flange_mm(self, joints: JointPositions) -> "list[float] | None":
        """Where the flange stands at ``joints``, in BASE millimetres; ``None`` for a model with no chain.

        The near point of a joint move's refresh: when the slot budget bites, the obstacles nearest
        the goal are the ones kept, and the goal of a joint move is where its flange ends up.
        """
        from src.robot.safety._ur_kinematics import ur_link_origins_mm

        origins = ur_link_origins_mm(
            str(self.config.ur.model), np.asarray(joints.tolist(), dtype=np.float64)
        )
        return None if origins is None else [float(v) for v in origins[-1]]

    #: How much the tool may turn along a ``willy`` mode line before it stops being a
    #: straight line for the thing that matters. The controller runs the flange straight;
    #: with the tool composed on this side, a turn swings the grasp centre off that line by
    #: up to the tool length. Half a degree: enough to absorb a pose that is nominally
    #: equal, far too little to hide a real reorientation.
    _WILLY_LINE_MAX_TURN_DEG = 0.5

    def _judge_linear_move(
        self, pose: Pose, *, command: MotionCommand, commanded: bool = True,
        start: "tuple[Pose, Sequence[float]] | None" = None,
    ) -> "MotionResult | None":
        """Judge the straight line the controller will run to ``pose``; ``None`` means it may run.

        On an ik cell this is the endpoint gate and nothing else, screened where ``commanded`` is false, for a line
        nothing is sent along (:meth:`carried_line_refusal`): the same guards, and the continuity memo stays where the
        last commanded target left it. A cuRobo line's judgement remembers nothing either way.

        On a cuRobo cell the line is sampled in the frame the controller interpolates,
        because that is the frame in which the motion is actually straight: the bare flange
        in ``willy`` mode, the declared tool in ``polyscope`` mode. Every sample is solved by
        the controller's own inverse kinematics, seeded with the previous solution and the
        first with the joints the arm stands at, so the configurations judged are the ones the
        controller will pass through rather than a second opinion from a different solver. Then
        the local guards judge all of them, and the planner judges the same list against its
        world.

        It is not free: one RTDE round trip per sample, and the samples are as dense as the
        collision margin asks for. A line is the one motion in this stack that nothing plans,
        so the alternative to paying for it is executing it unexamined.

        ``start`` is a TCP pose and the joints the arm would stand at there: the line is judged from that pose, its
        first sample seeded on those joints, as if the arm stood there (a grasp judged ahead). Never with ``commanded``.
        Such a line is solved by the controller at knots :data:`_AHEAD_KNOT_MM` apart at most rather than at every sample
        (:meth:`_judge_line_samples`): it is a prediction, and the line that runs is judged again where the arm stands,
        every sample solved.
        """
        if self._preflight is None:
            return None
        if start is not None and commanded:
            raise ValueError("a line judged from elsewhere than where the arm stands is never the line commanded")
        if self._motion_planner != "curobo":
            return self._gate_pose(pose, command, commanded=commanded)
        if not self._conn.is_connected:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR, command, target_pose=pose,
                message="a judged line starts at the current pose, which needs an open connection.",
            )

        step_mm = self._preflight.path_step_mm
        radii = self._preflight.joint_radii_mm(self)
        if step_mm is None or radii is None:
            # The same two refusals a judged joint path gives, in the same words. Asked
            # through the joint gate rather than restated here, so one place says them.
            return self._preflight.gate_planned_path(
                [[float(v) for v in self._conn.get_joint_positions()]], arm=self, command=command,
            )

        owns_tool = self.config.gripper.tool_frame.source == "willy"
        start_tcp = self.get_tcp_pose() if start is None else start[0]
        if owns_tool:
            turn_deg = float(
                np.degrees(angle_between(start_tcp.quaternion_xyzw, pose.quaternion_xyzw))
            )
            if turn_deg > self._WILLY_LINE_MAX_TURN_DEG:
                return MotionResult.failed(
                    MotionStatus.UNSUPPORTED, command, target_pose=pose,
                    message=(
                        f"this line turns the tool by {turn_deg:.2f} degrees, and in willy mode the "
                        f"controller runs the flange straight while this driver composes the tool: "
                        f"the grasp centre would travel an arc, not the line that was asked for. "
                        f"Move without the turn, or run the cell in polyscope mode, where the "
                        f"controller holds the tool and moveL draws a straight tool line"
                    ),
                )
            start_line = self._pose_to_flange(start_tcp)
            goal_line = self._pose_to_flange(pose)
        else:
            start_line, goal_line = start_tcp, pose

        try:
            samples = line_samples(
                start_line, goal_line, reach_mm=max(radii), max_step_mm=step_mm,
            )
        except ValueError as exc:
            return MotionResult.failed(
                MotionStatus.UNSUPPORTED, command, target_pose=pose, message=str(exc),
            )
        # The world is built about the line's lower end, the hand's own place there, where it passes the parts: the
        # boxes beside the fingers stay as the camera saw them and the far ones merge first (``height_map.coarsen``).
        # About the flange of the line's end, the lift's world merged the boxes beside the part it starts from, and the
        # housing stood 0.9 mm inside one the line down had kept clear (the grasp bench, 2026-10-06).
        low = pose if float(pose.position_mm[2]) <= float(start_tcp.position_mm[2]) else start_tcp
        # The region between the jaws is left out there too: a lift starts with the part between them, and a line back up
        # with the jaws open starts with them about it. At the line's end, the lift's world held the part it starts from
        # between the fingers, and refused the line back up its own descent had run (the grasp bench, 2026-10-06).
        return self._judge_line_samples(
            samples, pose=pose, owns_tool=owns_tool, radii=radii, command=command,
            seed_joints=None if start is None else start[1],
            near_point_mm=[float(v) for v in low.position_mm], keep_out_pose=low,
            knot_mm=None if start is None else self._AHEAD_KNOT_MM,
        )

    #: How far apart, at most, the controller solves a line judged ahead (``start``, never commanded), the joint line
    #: between two solutions filled at the line's own step as every step is. On nominal UR10 DH the cell's two grasp lines
    #: of 2026-10-08 (80 mm down, 100 mm up), filled between knots 12 mm apart, put the flange 26 to 33 um off the line
    #: moveL runs, against 1.7 to 2.2 um at the 3 mm a commanded line is solved at, for a quarter of the controller's round
    #: trips at 32.5 ms each (the judge chain, 2026-10-08).
    _AHEAD_KNOT_MM = 12.0
    #: How far off the line the controller runs a step between two knots may put the flange and what it carries
    #: (:func:`_off_the_line_mm`) before that step is solved at every sample, as a commanded line is: three times what the
    #: cell's lines measured, and far inside the step the path gate judges at.
    _AHEAD_KNOT_OFF_MM = 0.1

    def _line_step_refusal(
        self,
        previous: "Sequence[float]",
        solved: "Sequence[float]",
        *,
        index: int,
        count: int,
        bound_mm: float,
        pose: Pose,
        command: MotionCommand,
    ) -> "MotionResult | None":
        """Why the step from ``previous`` to ``solved`` is not part of one continuous line, or ``None`` when it is.

        A branch change is a joint jumping, or a chord off the line. It is not the sum
        |dq_j| * r_j against the step bound: that sum adds two joints turning against each other
        as if both carried the arm the same way, and on URSim a continuous lift sums to 27 mm a
        step while no link origin moves more than 10 mm. A jump is one joint turning more than a
        continuous line does. A branch change with no jump, the two elbow branches near full
        stretch, reaches the same flange pose from two configurations 0.30 rad apart, and it
        shows where the judged fill and moveL part: the joint line between them puts the flange
        off the line the controller runs.
        """
        where = "the joints the arm stands at" if index == 0 else f"sample {index}"
        turned = [abs(b - a) for a, b in zip(previous, solved)]
        joint = max(range(len(turned)), key=turned.__getitem__)
        if turned[joint] > LINE_MAX_JOINT_STEP_RAD:
            return MotionResult.failed(
                MotionStatus.IK_QUALITY_REJECTED, command, target_pose=pose,
                message=(
                    f"the controller changed IK branch between {where} and sample {index + 1} of {count} of this "
                    f"line: joint {joint + 1} turns {turned[joint]:.2f} rad in one step of at most {bound_mm:.1f} mm, "
                    f"where a continuous line turns no joint more than {LINE_MAX_JOINT_STEP_RAD} rad. The arm would "
                    f"take a route none of these samples describes"
                ),
            )
        model = str(self.config.ur.model)
        middle = [(a + b) / 2.0 for a, b in zip(previous, solved)]
        frames = [ur_link_transforms_mm(model, np.asarray(q, dtype=np.float64)) for q in (previous, solved, middle)]
        if any(chain is None for chain in frames):
            return MotionResult.failed(
                MotionStatus.UNSUPPORTED, command, target_pose=pose,
                message=(f"robot.ur.model {model!r} has no DH chain here, so whether this line's samples are one "
                         "branch cannot be read, and a line nobody read is not run"),
            )
        start, end, mid = (np.asarray(chain[-1], dtype=np.float64)[:3, 3] for chain in frames)  # type: ignore[index]
        off_mm = float(np.linalg.norm(mid - (start + end) / 2.0))
        if off_mm > LINE_MAX_CHORD_OFF_MM:
            return MotionResult.failed(
                MotionStatus.IK_QUALITY_REJECTED, command, target_pose=pose,
                message=(
                    f"the controller changed IK branch between {where} and sample {index + 1} of {count} of this "
                    f"line: no joint jumps, but the joint line between the two solutions puts the flange "
                    f"{off_mm:.2f} mm off the line moveL runs, more than {LINE_MAX_CHORD_OFF_MM} mm, so what would "
                    f"be judged is not what the controller would drive"
                ),
            )
        return None

    def _judge_line_samples(
        self,
        samples: "LineSamples",
        *,
        pose: Pose,
        owns_tool: bool,
        radii: "tuple[float, ...]",
        command: MotionCommand,
        seed_joints: "Sequence[float] | None" = None,
        near_point_mm: "Sequence[float] | None" = None,
        keep_out_pose: Pose | None = None,
        knot_mm: "float | None" = None,
    ) -> "MotionResult | None":
        """Solve every sample of a line and hand the configurations to both authorities; ``seed_joints`` stands for the
        joints the arm stands at where the line is judged from elsewhere (:meth:`_judge_linear_move`'s ``start``).
        ``near_point_mm`` is where the world is built about, the flange of ``pose`` where it is not given, and
        ``keep_out_pose`` the TCP whose region between the jaws is left out of it, ``pose`` where it is not given.

        ``knot_mm`` has the controller solve the line at knots that far apart at most, every few samples, rather than
        at every sample: the line judged ahead. The step from one knot to the next is kept as a step between two
        samples is, the joint line between their solutions filled at the line's step, so the path gate judges
        configurations no further apart than its bound. Where every sample it covers is inside the workspace box, its
        knot has a solution, it is no branch change (:meth:`_line_step_refusal`), and it puts the flange and what it
        carries no more than :data:`_AHEAD_KNOT_OFF_MM` off the line (:func:`_off_the_line_mm`). Any other step is
        solved at every sample, seeded on its start, and judged as every step of a line is, which gives its verdict as
        before.

        With ``robot.safety.ik_quality.line_ik: local`` (the owner, 2026-10-09) every sample and knot is solved here, on
        the controller's own kinematics (:meth:`_controller_chain`, :func:`~src.robot.safety._ur_ik.ur_chain_ik_nearest`),
        rather than by a round trip each, and the controller is asked at two of them, the first solved and the last,
        whether its ``getInverseKinematics`` with the same seed answers the same configuration within
        :data:`~src.robot.safety._ur_ik.CONTROLLER_AGREEMENT_RAD`. Where it does, the configurations judged are the
        controller's to that; where it does not, where a sample cannot be vouched for here (no solution, another as near
        the seed, a singularity) or where the controller's rows or its TCP cannot be read, the line is walked again from
        its start with the controller solving it, as before."""
        assert self._preflight is not None  # noqa: S101 (the caller checked)
        configs: list[tuple[float, ...]] = []
        # The line starts at the joints the arm stands at. moveL starts there whatever an
        # unseeded solution of the start pose says, so sample 0 is solved seeded on them and
        # judged against them like every later sample: a start on another branch is refused
        # rather than judged as if the arm were in it.
        seed: JointPositions = JointPositions([float(v) for v in (
            self._conn.get_joint_positions() if seed_joints is None else seed_joints)])
        bound = samples.step_bound_mm if samples.step_bound_mm > 0.0 else float(self._preflight.path_step_mm or 0.0)
        count = len(samples.poses)

        def tcp_at(index: int) -> Pose:
            sample = samples.poses[index]
            return self._pose_from_flange(sample) if owns_tool else sample

        def outside(index: int) -> "MotionResult | None":
            tcp = tcp_at(index)
            if self._guard.is_inside_workspace(self._coerce_urpose(tcp)):
                return None
            x, y, z = (float(v) for v in tcp.position_mm)
            return MotionResult.failed(
                MotionStatus.WORKSPACE_REJECTED, command, target_pose=pose,
                message=(
                    f"sample {index + 1} of {count} of this line puts the grasp "
                    f"centre at ({x:.1f}, {y:.1f}, {z:.1f}), outside workspace_limits"
                ),
            )

        class NotSolvedHere(Exception):
            """A sample the controller's chain solved here cannot vouch for: the line is walked again by the controller."""

        # line_ik: local. Each sample solved here, as (its TCP, its seed, its solution) for the two the controller is
        # asked about; ``chain`` is None where the controller solves every sample, as it always did.
        chain = self._controller_chain() if self.config.safety.ik_quality.line_ik == "local" else None
        solved_here: "list[tuple[Pose, JointPositions, JointPositions]]" = []

        def by_the_controller(tcp: Pose, start: JointPositions) -> JointPositions:
            return self.ik(tcp, seed=start)

        def here(tcp: Pose, start: JointPositions) -> JointPositions:
            from src.robot.safety._ur_ik import ur_chain_ik_nearest, ur_pose_matrix_m

            assert chain is not None  # noqa: S101 (only ever the solver where there is one)
            rows, flange_from_tcp, model = chain
            goal = ur_pose_matrix_m(self._pose_to_controller(tcp).to_ur_list()) @ flange_from_tcp
            answer = ur_chain_ik_nearest(model, rows, goal, start.tolist())
            if answer.joints is None:
                raise NotSolvedHere(answer.why_not)
            solved = JointPositions(answer.joints)
            solved_here.append((tcp, start, solved))
            return solved

        solve = by_the_controller if chain is None else here

        def solved_at(index: int, start: JointPositions) -> "JointPositions | MotionResult":
            try:
                return solve(tcp_at(index), start)
            except (RobotKinematicsError, RobotConnectionError) as exc:
                return MotionResult.failed(
                    MotionStatus.IK_FAILED, command, target_pose=pose,
                    message=(
                        f"sample {index + 1} of {count} of this line has no inverse "
                        f"kinematics solution: {exc}"
                    ),
                    exception=exc,
                )

        def kept(index: int, previous: "list[float]", solved: JointPositions,
                 filled: "PathSamples | None" = None) -> "MotionResult | None":
            # The path gate is told a bound on how far any point moves between two configurations
            # it judges, and the sum |dq_j| * r_j is the one bound that holds without the geometry.
            # Where a continuous step is more than the sum can vouch for, the joint line between
            # its two solutions is judged too, at the line's own step; the chord check before it
            # (_line_step_refusal, and _off_the_line_mm between knots) is what says that joint line
            # is the motion moveL runs.
            if bound > 0.0:
                if filled is None:
                    try:
                        filled = joint_path_samples(previous, solved.tolist(), reach_mm=radii, max_step_mm=bound)
                    except ValueError as exc:
                        return MotionResult.failed(
                            MotionStatus.UNSUPPORTED, command, target_pose=pose,
                            message=f"sample {index + 1} of {count} of this line cannot be judged: {exc}",
                        )
                configs.extend(filled.configs[1:-1])
            configs.append(tuple(solved.tolist()))
            if len(configs) > MAX_PATH_SAMPLES:
                return MotionResult.failed(
                    MotionStatus.UNSUPPORTED, command, target_pose=pose,
                    message=(
                        f"this line needs more than {MAX_PATH_SAMPLES} configurations to be judged at "
                        f"{bound:.2f} mm a step; move it in shorter lines"
                    ),
                )
            return None

        def walk(first: int, last: int, start: JointPositions) -> "JointPositions | MotionResult":
            """Samples ``first`` to ``last``, each solved seeded on the one before, ``start`` before the first."""
            for index in range(first, last + 1):
                refused = outside(index)
                if refused is not None:
                    return refused
                solved = solved_at(index, start)
                if isinstance(solved, MotionResult):
                    return solved
                previous = start.tolist()
                refused = self._line_step_refusal(
                    previous, solved.tolist(), index=index, count=count, bound_mm=bound, pose=pose, command=command,
                )
                if refused is None:
                    refused = kept(index, previous, solved)
                if refused is not None:
                    return refused
                start = solved
            return start

        def knot_to(first: int, knot: int, start: JointPositions) -> "JointPositions | MotionResult | None":
            """The step from sample ``first`` to sample ``knot``, solved at the knot alone; ``None`` where it is to be
            solved at every sample, and nothing of it was kept."""
            if knot <= first + 1 or any(outside(index) is not None for index in range(first + 1, knot + 1)):
                return None
            solved = solved_at(knot, start)
            if isinstance(solved, MotionResult):
                return None
            previous = start.tolist()
            if self._line_step_refusal(previous, solved.tolist(), index=knot, count=count, bound_mm=bound, pose=pose,
                                       command=command) is not None:
                return None
            try:
                filled = joint_path_samples(previous, solved.tolist(), reach_mm=radii, max_step_mm=bound)
            except ValueError:
                return None
            steps = len(filled.configs) - 1
            fractions = [0.5, *(k / steps for k in range(1, steps))]
            if not _off_the_line_mm(str(self.config.ur.model), previous, solved.tolist(), fractions,
                                    reach_mm=max(radii)) <= self._AHEAD_KNOT_OFF_MM:
                return None
            refused = kept(knot, previous, solved, filled)
            return solved if refused is None else refused

        every = (max(1, int(math.floor(float(knot_mm) / bound + 1e-9)))
                 if knot_mm is not None and bound > 0.0 else 1)

        def walked_line() -> "JointPositions | MotionResult":
            """The line walked from its start, at knots ``every`` samples apart where they hold, at every sample
            elsewhere; what it keeps for the gate starts empty."""
            configs.clear()
            walked: "JointPositions | MotionResult" = walk(0, 0 if every > 1 else count - 1, seed)
            last = 0 if every > 1 else count - 1
            while not isinstance(walked, MotionResult) and last < count - 1:
                knot = min(last + every, count - 1)
                stepped = knot_to(last, knot, walked)
                walked = walk(last + 1, knot, walked) if stepped is None else stepped
                last = knot
            return walked

        def disagreement() -> str:
            """Why the controller's own answer at the first or the last sample solved here is not the one solved here,
            ``""`` where both agree: its ``getInverseKinematics``, seeded as that sample was."""
            from src.robot.safety._ur_ik import CONTROLLER_AGREEMENT_RAD

            asked = (solved_here[:1] + solved_here[-1:])[:len(solved_here)]
            for which, (tcp, start, solved) in zip(("first", "last"), asked):
                try:
                    answer = self.ik(tcp, seed=start)
                except (RobotKinematicsError, RobotConnectionError) as exc:
                    return f"the controller found no solution for the {which} sample solved here ({exc})"
                apart = max(abs(a - b) for a, b in zip(answer.tolist(), solved.tolist()))
                if not apart <= CONTROLLER_AGREEMENT_RAD:
                    return (f"the controller's own solution of the {which} sample solved here stands {apart:.2e} rad "
                            f"from it, more than {CONTROLLER_AGREEMENT_RAD:g} rad")
            return ""

        if chain is None:
            walked = walked_line()
        else:
            try:
                walked = walked_line()
                why = disagreement()
            except NotSolvedHere as exc:
                why = f"a sample cannot be vouched for here: {exc}"
            if why:
                self.logger.warning("the line to %s is solved by the controller as before: %s", pose.label or "the goal",
                                    why)
                solve = by_the_controller
                walked = walked_line()
            else:
                self.logger.info("the line to %s: %d configuration(s) solved here on the controller's own kinematics, "
                                 "and the controller answered the same at %s (%d round trip(s))",
                                 pose.label or "the goal", len(solved_here),
                                 "the first and the last" if len(solved_here) > 1 else "the one" if solved_here
                                 else "none, as none was solved", min(2, len(solved_here)))
        if isinstance(walked, MotionResult):
            return walked

        judged = PathSamples(configs=tuple(configs), step_bound_mm=bound)
        refused = self._refresh_before_the_path_gate(
            near_point_mm=([float(v) for v in near_point_mm] if near_point_mm is not None
                           else [float(v) for v in self._pose_to_flange(pose).position_mm]),
            command=command, target_pose=pose,
            goal_tcp_mm=self._tcp_of_pose(pose if keep_out_pose is None else keep_out_pose),
        )
        if refused is not None:
            return refused
        refused = self._preflight.gate_joint_path(judged, arm=self, command=command)
        if refused is not None:
            return refused
        try:
            asked: "CuroboUrPlanner | None" = self._curobo_ur_planner()
        except CuroboUnavailableError:
            asked = None  # the check below says so in its own words
        why_not = self._mesh_first_refusal(asked, judged.configs) if asked is not None else "no planner to skip"
        if asked is not None and why_not is None:
            # Mesh first: the exact guard just accepted every sample of the line; the spheres are not asked.
            self.logger.info("mesh first: the exact guard alone judged %d sample(s) of the line to %s clear; cuRobo was "
                             "not asked", len(judged.configs), pose.label or "the goal")
            return None
        if asked is not None:
            self.logger.info("mesh first does not apply to the line to %s (%s): cuRobo judges beside the exact guard",
                             pose.label or "the goal", why_not)
        try:
            verdict = self._curobo_ur_planner().check_joint_path(judged.configs, refresh=False)
        except CuroboUnavailableError as exc:
            return MotionResult.failed(
                MotionStatus.CONTROLLER_REJECTED, command, target_pose=pose,
                message=f"cuRobo planner unavailable: {exc}", exception=exc,
            )
        if verdict.valid:
            return None
        # A line only: the exact guard decides the planner's self pairs it judges, and a world refusal only the camera's
        # boxes make with a hand known empty and open, never a planned way round. A grasp's lift is judged here before
        # the jaws close too, as if they held the part (carried_line_refusal).
        standing = self._exact_guard_decides(self._curobo_ur_planner(), judged.configs, verdict, clearance_mm=0.0,
                                             command=command)
        if standing is None:
            return None
        status, message = standing.said("the planner refused this line", verdict, len(judged.configs))
        return MotionResult.failed(status, command, target_pose=pose, message=message)

    def _controller_chain(self) -> "tuple[Any, np.ndarray, str] | None":
        """The controller's own kinematics, for a line solved here (``robot.safety.ik_quality.line_ik: local``): its DH
        rows (a :class:`~src.robot.safety._ur_kinematics.URDhChain`), the inverse of its active TCP, and the model
        whose nominal table the closed form starts from; ``None`` where the rows or the TCP cannot be read, and the
        controller solves the line as before.

        Both are the connection's reads, once per connection (:meth:`URConnection.controller_kinematics`,
        :meth:`URConnection.active_tcp_offset`), which log what they read or why they could not. Once per connection
        this logs how far the controller's rows put the flange from the nominal table at the joints the arm stands at:
        on a calibrated arm, the part of the gap between the nominal chain and the controller's TCP that its
        calibration explains.
        """
        from src.robot.safety._ur_ik import ur_pose_matrix_m
        from src.robot.safety._ur_kinematics import URDhChain

        from .connection import ControllerKinematics

        if not self._conn.is_connected:
            return None
        read_rows = getattr(self._conn, "controller_kinematics", None)
        read_tcp = getattr(self._conn, "active_tcp_offset", None)
        try:
            kinematics = read_rows() if callable(read_rows) else None
            tcp = read_tcp() if callable(read_tcp) else None
        except (RobotConnectionError, RuntimeError, OSError) as exc:
            self.logger.warning("the controller's kinematics could not be read (%s): lines are solved by the controller",
                                exc)
            return None
        if not isinstance(kinematics, ControllerKinematics) or not isinstance(tcp, list) or len(tcp) != 6:
            return None
        model = str(self.config.ur.model)
        try:
            rows = URDhChain(theta_rad=kinematics.theta_rad, a_m=kinematics.a_m, d_m=kinematics.d_m,
                             alpha_rad=kinematics.alpha_rad)
            flange_from_tcp = np.linalg.inv(ur_pose_matrix_m(tcp))
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.logger.warning("the controller's kinematics read as no chain to solve on (%s): lines are solved by the "
                                "controller", exc)
            return None
        if getattr(self, "_chain_described", None) is not kinematics:
            self._chain_described = kinematics
            nominal = URDhChain.nominal(model)
            try:
                here = [float(v) for v in self._conn.get_joint_positions()]
                ours, table = rows.flange_m(here), nominal.flange_m(here) if nominal is not None else None
            except (RobotConnectionError, RuntimeError, OSError, ValueError):
                table = None
            if table is not None:
                cosine = (float(np.trace(ours[:3, :3].T @ table[:3, :3])) - 1.0) / 2.0
                self.logger.info(
                    "lines are solved here on the controller's own kinematics (robot.safety.ik_quality.line_ik: local) "
                    "and checked against the controller at two samples each; at the joints the arm stands at, its rows "
                    "put the flange %.2f mm and %.3f deg from the nominal %s table",
                    1000.0 * float(np.linalg.norm(ours[:3, 3] - table[:3, 3])),
                    math.degrees(math.acos(max(-1.0, min(1.0, cosine)))), model,
                )
        return rows, flange_from_tcp, model

    def _drive_checked_line(
        self, pose: Pose, *, vel: float | None = None, acc: float | None = None
    ) -> MotionResult:
        """Judge the line, then run it as one ``moveL``. The typed half of :meth:`move_linear`.

        The command that reaches the controller is the same one an unchecked linear move
        sent, through the same `MotionController.move_to`, so its endpoint inverse kinematics
        and its singularity check still run below this. What is added is that the line it
        draws has been looked at.

        A refusal below the judge carries the cause ``MotionController`` classified, as
        :meth:`move` does, and a sentence that says whether ``moveL`` had been sent.

        The lift judged at the part as if the jaws held the part (:meth:`carried_line_refusal`) is this line's judgement
        where nothing it read changed since (:meth:`_why_the_lift_is_judged_again`): every sample of it was judged there,
        in this held world, on a cell where the part in the jaws is judged as the empty hand was. It serves one line. So
        is a line judged ahead where the arm stands, as it will run, while the jaws opened (:meth:`judge_line_ahead`).

        Where this arm gates its own sends (``safety.dwell.gate_at`` ``send``), the steady gate waits after the
        judgement, right before ``MotionController.move_to`` (:meth:`_send_gate`), and a line whose arm came to rest more
        than :data:`_AHEAD_MOVED_MM` from where it was judged from is judged again from where it stands. Where the line is
        the one a verb left the joint move declared next to (:meth:`judge_the_next_leg`), that move is judged on a second
        thread while the line runs, and the line's send waits for it once it ended (:meth:`_judging_during`).
        """
        lift = self._lift_judged
        during, self._next_leg_during = self._next_leg_during, None
        self._forget_what_was_judged_ahead()
        why = "" if lift is None else self._why_the_lift_is_judged_again(lift, pose)
        judged = lift is not None and not why
        judged_from: "Sequence[float] | None" = lift.joints if lift is not None and judged else None
        if lift is not None and judged:
            if lift.stands_in:
                self.logger.info("the line up to %s runs on the lift judged at the part as if the jaws held the part, in "
                                 "the world held (%s); it is not judged a second time", pose.label or "<unlabeled>",
                                 lift.reason)
            else:
                self.logger.info("the line to %s runs as it was judged ahead where the arm stands, in the world held "
                                 "(%s); it is not judged a second time", pose.label or "<unlabeled>", lift.reason)
        elif lift is not None:
            self.logger.info("the line up to %s is judged again: %s", pose.label or "<unlabeled>", why)
        for _ in range(self._SEND_JUDGEMENTS):
            if not judged:
                # Where the line was judged from, read for the gate at the send alone: an arm that does not gate its
                # sends reads nothing more of the controller than it always did.
                judged_from = (self._joints_now() or ()) if self._gates_sends else None
                refused = self._judge_linear_move(pose, command=MotionCommand.MOVE_TO)
                if refused is not None:
                    return refused
                judged = True
            gate = self._send_gate(judged_from, MotionCommand.MOVE_TO, target_pose=pose)
            if isinstance(gate, MotionResult):
                return gate
            if gate is None:
                break
            judged = False
        else:
            return self._kept_moving(MotionCommand.MOVE_TO, target_pose=pose)
        urpose = self._pose_to_controller(pose)
        worker = self._judging_during(pose, during)
        try:
            ok = self._motion.move_to(
                urpose, linear=True, vel=vel, acc=acc, register=False,
                workspace_pose=self._coerce_urpose(pose),
            )
        finally:
            self._joined(worker)
        if ok:
            return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)
        status = self._motion.last_reject_status
        if status is MotionStatus.CANCELLED:
            return self._cancelled_by_the_halt(MotionCommand.MOVE_TO, "moveL", target_pose=pose)
        before_movel = {
            MotionStatus.CONNECTION_ERROR: "the arm is not connected",
            MotionStatus.WORKSPACE_REJECTED: "its end is outside the controller's workspace box",
            MotionStatus.IK_FAILED: "the controller found no inverse kinematics solution for its end",
            MotionStatus.IK_QUALITY_REJECTED: "its end is near a singularity by the controller's own check",
        }
        if not isinstance(status, MotionStatus):
            status = None
        if status in before_movel:
            message = f"the judged line was refused before moveL was sent: {before_movel[status]}"
        elif status is MotionStatus.CONTROLLER_REJECTED:
            message = ("the judged line was sent as moveL and the controller reported it failed or raised; "
                       f"the arm may have moved part of the way.{self._controller_state_text()}")
        elif status is not None:
            message = f"MotionController.move_to refused the judged line as {status.value}; ur_motion.log has why"
        else:
            message = "MotionController.move_to refused the judged line and gave no cause; ur_motion.log has it"
        return MotionResult.failed(
            status if status is not None else MotionStatus.CONTROLLER_REJECTED,
            MotionCommand.MOVE_TO, target_pose=pose, message=message,
        )

    def _pose_from_flange(self, flange: Pose) -> Pose:
        """Flange (tool0) to TCP, the inverse of :meth:`_pose_to_flange`."""
        t = self._declared_tool_matrix()
        if t is None:
            return flange
        m = np.asarray(flange.to_matrix(), dtype=np.float64) @ t
        return Pose.from_matrix(m, frame=Frame.BASE, label=flange.label)

    def move_joint(
        self,
        joints: JointPositions,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> None:
        """A gated joint-space move through the UR RTDE driver.

        The Protocol promises ``RobotMotionRejected`` where a safety pre-check denies
        the move, and this is a public Protocol-level surface that sim runners and
        operator code reach directly rather than only through :meth:`move_to_joints`. It
        judges exactly what that method judges, through the same helper, and raises what
        it returns: the destination on an ik cell, the whole path on a cuRobo one, and it
        drives what was judged. On cuRobo, a move with neither a camera world nor a decline in
        scope is refused before the judge.
        """
        self._forget_what_was_judged_ahead()
        refused = self._refused_for_camera_world(
            self._move_camera_world(UNSET), MotionCommand.MOVE_JOINTS, target_joints=joints,
        )
        if refused is not None:
            raise RobotMotionRejected(f"move_joint refused: {refused.message}", result=refused)
        target, route, rejected = self._judge_joint_move(joints, command=MotionCommand.MOVE_JOINTS)
        if rejected is not None:
            if rejected.status is MotionStatus.CONNECTION_ERROR:
                raise RobotConnectionError(rejected.message or "move_joint needs a connection.")
            raise RobotMotionRejected(
                f"move_joint refused: {rejected.message}", result=rejected,
            ) from rejected.exception
        if route is None or not route.planned:
            if route is not None:
                self._log_the_route("move_joint", route)
            self._drive_joints(target, velocity=velocity, acceleration=acceleration)
            return
        result = self._drive_judged_joints(target, route, command=MotionCommand.MOVE_JOINTS, done="move_joint",
                                           velocity=velocity, acceleration=acceleration)
        if not result.ok:
            raise RobotMotionRejected(f"move_joint failed: {result.message}", result=result)

    def _drive_joints(
        self,
        joints: JointPositions,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
        command: MotionCommand = MotionCommand.MOVE_JOINTS,
    ) -> None:
        """Command the joints with no safety gate. Callers must have gated the destination first.

        Each refusal raises with its typed result, labelled ``command``, so a typed verb returns
        it instead of a status nobody could classify. The message says whether ``moveJ`` had
        been sent. Once it was, a raise or a ``False`` does not mean the arm stayed where it
        was: a controller that stops a motion midway, on a protective stop for instance, stops
        it wherever it had got to. So those two messages also carry the controller's robot and
        safety state, where it can be read.
        """
        if not self._conn.is_connected:
            raise RobotConnectionError("move_joint() requires an open connection, and moveJ was not sent.")
        if joints.dof != self.capabilities.dof:
            message = f"move_joint() expected {self.capabilities.dof} DoF, got {joints.dof}, and moveJ was not sent."
            raise RobotMotionRejected(message, result=MotionResult.failed(
                MotionStatus.INVALID_TARGET, command, target_joints=joints, message=message,
            ))
        vel, acc = self._motion._clamp(velocity, acceleration)
        try:
            ok = self._conn.moveJ(joints.tolist(), vel=vel, acc=acc)
        except (RuntimeError, OSError) as exc:
            message = (f"moveJ failed: {exc}. moveJ was sent and raised, so the arm may have moved part of the "
                       f"way.{self._controller_state_text()}")
            raise RobotMotionRejected(message, result=MotionResult.failed(
                MotionStatus.CONNECTION_ERROR, command, target_joints=joints, message=message, exception=exc,
            )) from exc
        if not ok and self._ended_by_the_halt():
            # The halt refused it or braked it: CANCELLED, not a controller refusal, and said as the halt.
            cancelled = self._cancelled_by_the_halt(command, "moveJ", target_joints=joints)
            raise RobotMotionRejected(cancelled.message, result=cancelled)
        if not ok:
            message = ("Driver reported moveJ() failure. moveJ was sent and the controller did not complete it, "
                       f"so the arm may have moved part of the way.{self._controller_state_text()}")
            raise RobotMotionRejected(message, result=MotionResult.failed(
                MotionStatus.CONTROLLER_REJECTED, command, target_joints=joints, message=message,
            ))

    def _controller_state_text(self) -> str:
        """The controller's robot and safety state as a sentence for a refusal, or ``""`` where it cannot be read.

        Best effort, and only ever read after a command failed: a state that cannot be read adds
        nothing to the refusal, and must not replace it with a second fault.
        """
        try:
            status = self.get_robot_status()
            sentence = (f" The controller reports robot mode {RobotMode(status.robot_mode).value} and safety "
                        f"mode {SafetyMode(status.safety_mode).value}")
            stops = [name for name, on in (("a protective stop", status.protective_stopped),
                                           ("an emergency stop", status.emergency_stopped)) if on is True]
            text = status.message.strip() if isinstance(status.message, str) else ""
        except Exception:  # noqa: BLE001 (best effort after a failure: no state is not a new fault)
            return ""
        if stops:
            sentence += f", with {' and '.join(stops)} active"
        if text:
            sentence += f" ({text!r})"
        return sentence + "."

    def move_linear(
        self,
        pose: Pose,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> None:
        """Cartesian move to ``pose`` in :attr:`Frame.BASE`.

        On cuRobo, a move with neither a camera world nor a decline in scope is refused before the
        connection is read.
        """
        self._forget_what_was_judged_ahead()
        refused = self._refused_for_camera_world(
            self._move_camera_world(UNSET), MotionCommand.MOVE_TO, target_pose=pose,
        )
        if refused is not None:
            raise RobotMotionRejected(f"move_linear refused: {refused.message}", result=refused)
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"move_linear() requires pose in Frame.BASE; got {pose.frame!r}."
            )
        if not self._conn.is_connected:
            raise RobotConnectionError("move_linear() requires an open connection.")
        rejected = self._judge_linear_move(pose, command=MotionCommand.MOVE_TO)
        if rejected is not None:
            raise RobotMotionRejected(
                f"move_linear refused: {rejected.message}", result=rejected,
            ) from rejected.exception
        urpose = self._pose_to_controller(pose)
        ok = self._motion.move_to(
            urpose, linear=True, vel=velocity, acc=acceleration, register=False,
            workspace_pose=self._coerce_urpose(pose),
        )
        if not ok and self._motion.last_reject_status is MotionStatus.CANCELLED:
            cancelled = self._cancelled_by_the_halt(MotionCommand.MOVE_TO, "moveL", target_pose=pose)
            raise RobotMotionRejected(f"move_linear refused: {cancelled.message}", result=cancelled)
        if not ok:
            raise RobotMotionRejected(
                f"Linear move to '{pose.label or '<unlabeled>'}' was rejected."
            )

    @property
    def connection(self) -> URConnection:
        """Underlying RTDE connection.

        Ungated: its move calls reach the controller below the SafetyPreflight.
        """
        return self._conn

    @property
    def guard(self) -> WorkspaceGuard:
        """Workspace guard instance."""
        return self._guard

    @property
    def motion(self) -> MotionController:
        """Motion controller instance.

        Ungated: its move calls check only the workspace box, IK and singularity, below
        the SafetyPreflight.
        """
        return self._motion

    # ------------------------------------------------------------------
    # The optional capabilities: SupportsDigitalIO, SupportsForceTorque,
    # SupportsRobotStatus and SupportsFreedrive. They are not on the vendor-neutral RobotArm
    # Protocol, a caller feature-checks with ``isinstance(arm, SupportsForceTorque)``, and
    # every method delegates to the RTDE interfaces in URConnection. A non-UR driver does not
    # implement them.
    # ------------------------------------------------------------------

    # --- SupportsDigitalIO ---
    def set_digital_output(
        self, pin: int, value: bool, *, port: DigitalIOPort = DigitalIOPort.STANDARD
    ) -> None:
        """Drive a controller digital output pin (bank ``port``) high/low."""
        self._conn.set_digital_out(int(pin), bool(value), str(port))

    def end_output_pulse(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> None:
        """Drive ``pin`` low to end a pulse that began before a halt: the one output write a halted arm makes.

        For a bistable valve's coil and a vacuum blow-off, whose pulse must end low however it ends; never for a toggle
        hand, whose every change of its output moves the jaws (:meth:`URConnection.end_output_pulse`).
        """
        self._conn.end_output_pulse(int(pin), str(port))

    def get_digital_input(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> bool:
        """Read a controller digital input pin (bank ``port``)."""
        return self._conn.get_digital_in(int(pin), str(port))

    def get_digital_output(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> bool:
        """Read back a controller digital output pin's commanded level (bank ``port``)."""
        return self._conn.get_digital_out(int(pin), str(port))

    def set_analog_output(self, pin: int, value: float, *, current: bool = False) -> None:
        """Set a controller analog output: voltage (V) unless ``current=True`` (A)."""
        self._conn.set_analog_out(int(pin), float(value), bool(current))

    # --- SupportsForceTorque ---
    def get_tcp_wrench(self) -> Wrench:
        """The live generalized force and torque at the TCP, in N and Nm in the BASE frame.

        It is the hand-over signal.
        """
        f = self._conn.get_tcp_force()
        return Wrench(float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[4]), float(f[5]),
                     frame=Frame.BASE)

    def get_joint_torques(self) -> tuple[float, ...]:
        """Live torque at each joint (newton-metres), base-to-tool order."""
        return tuple(float(t) for t in self._conn.get_joint_torques())

    # --- SupportsRobotStatus + enriched errors ---
    def get_robot_status(self) -> RobotStatus:
        """A snapshot of the controller live robot mode, safety mode, stop flags and text.

        It is the vendor-neutral projection of the UR ``getRobotMode``,
        ``getSafetyMode``, ``isProtectiveStopped`` and ``isEmergencyStopped``, plus the
        dashboard ``safetystatus`` text: the enriched controller state a pipeline
        surfaces instead of a bare motion rejection. ``halted`` is the arm's halt latch, read beside the controller,
        so ``is_operational`` refuses a halted arm and ``controller_operational`` keeps the controller's own answer.
        """
        halted = self._halt_record()
        return RobotStatus(
            robot_mode=_UR_ROBOT_MODE.get(self._conn.get_robot_mode(), RobotMode.UNKNOWN),
            safety_mode=_UR_SAFETY_MODE.get(self._conn.get_safety_mode(), SafetyMode.UNKNOWN),
            protective_stopped=self._conn.is_protective_stopped(),
            emergency_stopped=self._conn.is_emergency_stopped(),
            message=self._conn.dashboard_safety_status(),
            halted=halted.reason if halted is not None else "",
        )

    def quick_robot_status(self) -> RobotStatus:
        """The controller's four fields from the RTDE receive stream and the halt latch: no dashboard round trip.

        What a ready bar polls. :meth:`get_robot_status` asks the dashboard for its text on every call, a socket round
        trip; this reads only what the receive interface already holds, and ``message`` stays empty. Raises
        :class:`RobotConnectionError` on a closed connection.
        """
        if not self._conn.is_connected:
            raise RobotConnectionError("quick_robot_status() requires an open connection.")
        halted = self._halt_record()
        return RobotStatus(
            robot_mode=_UR_ROBOT_MODE.get(self._conn.get_robot_mode(), RobotMode.UNKNOWN),
            safety_mode=_UR_SAFETY_MODE.get(self._conn.get_safety_mode(), SafetyMode.UNKNOWN),
            protective_stopped=self._conn.is_protective_stopped(),
            emergency_stopped=self._conn.is_emergency_stopped(),
            message="",
            halted=halted.reason if halted is not None else "",
        )

    def recover_from_protective_stop(self) -> bool:
        """Ask the controller (via the dashboard) to release an active protective stop."""
        ok = self._conn.unlock_protective_stop()
        if ok:
            self.logger.info("Requested protective-stop release from the UR dashboard.")
        else:
            self.logger.warning("Protective-stop release unavailable (no dashboard connection).")
        return ok

    # --- SupportsFreedrive ---
    def freedrive(self) -> FreedriveSession:
        """A hand-guiding session on the UR teach mode (``SupportsFreedrive``); nothing is freed yet.

        Refused while the arm is not connected, while the controller is not operational (a stop, a
        safety mode other than normal, brakes engaged) and while a session is open. The session frees
        the arm on ``free()`` inside its with-block, and from the with-block's start to its end every
        motion verb of this arm refuses; :mod:`.freedrive` has the order of the controller calls.
        """
        self._forget_what_was_judged_ahead()
        if not self._conn.is_connected:
            raise RobotConnectionError("freedrive() requires an open connection.")
        if self._freedrive is not None:
            raise RobotMotionRejected(f"{HAND_GUIDED_MESSAGE}; leave it before opening another")
        halted = self._halt_record()
        if halted is not None:
            # Said as the halt: the controller sentence a halted status reads in raise_unless_operational would name
            # a controller that is fine.
            raise RobotMotionRejected(halted_refusal(halted.reason, "a freedrive session was not opened"))
        raise_unless_operational(self.get_robot_status(), "a freedrive session")
        return URFreedriveSession(
            self._conn, tcp_pose=self.get_tcp_pose, robot_status=self.get_robot_status,
            on_open=self._freedrive_opened, on_close=self._freedrive_closed,
        )

    def controller_payload(self) -> ControllerPayload | None:
        """The payload the controller compensates for, or ``None`` where RTDE does not report it.

        It decides whether a freed arm floats, sinks or rises, and it is the controller's own record,
        which is the one ``safety.payload`` pushed at connect only where that says ``enforce: true``.
        """
        read = self._conn.get_payload()
        if read is None:
            return None
        mass_kg, cog_mm = read
        return ControllerPayload(mass_kg=mass_kg, cog_mm=cog_mm)

    def _freedrive_opened(self, session: URFreedriveSession) -> None:
        """Record ``session`` as the open one; another open session is refused."""
        if self._freedrive is not None and self._freedrive is not session:
            raise RobotMotionRejected(f"{HAND_GUIDED_MESSAGE}; leave it before opening another")
        self._freedrive = session

    def _freedrive_closed(self, session: URFreedriveSession) -> None:
        """Forget ``session``, which gives motion back."""
        if self._freedrive is session:
            self._freedrive = None

    def _hand_guided_refusal(
        self, command: MotionCommand, *,
        target_pose: Pose | None = None, target_joints: JointPositions | None = None,
    ) -> MotionResult | None:
        """The refusal of every motion while the arm is halted or a freedrive session is open, or ``None``.

        The one check every motion verb meets: :meth:`_refused_for_camera_world` asks it for every verb,
        and :meth:`_gate_pose` for the ik path of ``move_to`` and ``amove_to``, both before a planner or
        the controller is asked anything. A halted arm is refused first, CANCELLED with nothing sent: a halt is
        no controller refusal, and a person ends it.
        """
        halted = self._halt_record()
        if halted is not None:
            return MotionResult.failed(
                MotionStatus.CANCELLED, command, target_pose=target_pose, target_joints=target_joints,
                message=halted_refusal(halted.reason, "nothing was sent to the controller"),
            )
        if self._freedrive is None:
            return None
        return MotionResult.failed(
            MotionStatus.CONTROLLER_REJECTED, command, target_pose=target_pose, target_joints=target_joints,
            message=(f"{HAND_GUIDED_MESSAGE}, so this arm does not drive itself and no command reached the "
                     f"controller; leave the session first, which holds the arm and gives motion back"),
        )

    def raise_if_stopped(self) -> None:
        """Raise an enriched :class:`RobotEmergencyStop` where the controller is stopped or faulted.

        It guards a critical operation: it reads :meth:`get_robot_status` and, where the
        arm is in a protective or emergency stop or its safety mode is a stop state,
        raises with the safety mode and the controller own text, so the operator sees
        why it stopped rather than only that it did.
        """
        status = self.get_robot_status()
        if status.is_stopped:
            detail = f": {status.message}" if status.message else ""
            raise RobotEmergencyStop(
                f"UR controller in {status.safety_mode.value} "
                f"(robot_mode={status.robot_mode.value}){detail}"
            )
