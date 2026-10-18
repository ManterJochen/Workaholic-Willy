"""The arms the core tests stand on: a PlannedArm over the fake transport, built as a vendor would, no UR driver.

P0g slice 13 moves the tests of the planning half off ``URRobotArm``. A core test built a UR, replaced its RTDE
connection with a MagicMock and read the MagicMock's calls, so it tested the core through one vendor's wire and would
have needed a copy for every vendor after it. The kit builds the same arms from ``PlannedArm`` itself: the pieces its
keyword constructor takes (P0g slice 4), a transport that answers in the stack's units and records what it was sent
(``tests/_fake_transport.py``), and the stand-ins each shared helper installed, one builder per helper:

====================================================  =======================================
the helper it replaces                                the kit builder
====================================================  =======================================
``test_ur_arm._arm``                                   :func:`bare_arm`
``test_camera_world_on_every_driver._ur_ik``           :func:`ik_arm`
``test_camera_world_on_every_driver._ur_curobo``       :func:`curobo_arm`
``test_ur_checked_linear._arm``                        :func:`line_arm`
``test_the_arm_holds_its_own_steady_gate._arm``        :func:`steady_arm`
``test_every_verb_meets_the_camera_world._ur``         :func:`live_arm`
====================================================  =======================================

The trees still name a UR model: every description this repository holds is a UR's, and the core still reads UR
geometry in the places P1 replaces. What the kit does not do is import the UR driver, so a test built on it holds
for any vendor whose transport answers the same (``tests/robot/drivers/ur/test_every_transport_keeps_the_transport_contract.py``), and
the pieces it hands ``PlannedArm`` from a tree are the ones the UR driver hands it, or the same refusal
(``tests/robot/drivers/ur/test_the_kit_builds_what_the_ur_driver_builds.py``).
Controller readings are given the way a UR reports them, ``[x_m, y_m, z_m, rx, ry, rz]``, and turned into poses by the
same arithmetic the UR transport uses (:func:`controller_pose`, :func:`controller_matrix`), so a test that moved here
reads the numbers it read before, bit for bit.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from src.config.schema.robot import RobotConfig
from src.contracts import UNSET, Maybe, chosen
from src.geometry import Frame, Pose
from src.geometry.quaternion import from_axis_angle
from src.robot.core import JointPositions, MotionCommand, MotionResult, RobotCapabilities
from src.robot.drivers.planned.arm import PlannedArm
from src.robot.safety import SafetyPreflight
from tests._fake_planner import FakePlanner
from tests._fake_transport import fake_transport

__all__ = [
    "HERE",
    "LOGGER_NAME",
    "AcceptAllPreflight",
    "AcceptingGuard",
    "PlannerDouble",
    "RecordingIk",
    "RecordingPlanner",
    "bare_arm",
    "kit_pose",
    "controller_matrix",
    "controller_pose",
    "curobo_arm",
    "ik_arm",
    "kit_tree",
    "line_arm",
    "live_arm",
    "planned_arm",
    "steady_arm",
]

#: The configuration the camera-world and hand-verb helpers stood the arm at.
HERE = (0.0, -1.5, 1.5, 0.0, 1.5, 0.0)

#: The logger every kit arm writes to; a vendor names its own, and a test that reads a vendor's log stays on it.
LOGGER_NAME = "tests.arm_kit"


def kit_pose(frame: Frame = Frame.BASE) -> Pose:
    """The grasp centre the shared helpers commanded: 400 mm out, 300 mm up, tool down."""
    return Pose(position_mm=np.array([400.0, 0.0, 300.0]), quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]), frame=frame)


class AcceptingGuard:
    """A guard that accepts everything and counts how often it was asked, so a test is about the wiring."""

    def __init__(self, name: str = "self_collision") -> None:
        self._name = name
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    def evaluate(self, ctx: Any) -> Any:
        from src.robot.safety import SafetyDecision

        self.calls += 1
        return SafetyDecision.accept(self._name, message="ok")


class AcceptAllPreflight:
    """A preflight that accepts every question, so what a test reads is the gate and not a guard."""

    def context_for_pose(self, *args: object, **kwargs: object) -> object:
        return object()

    def evaluate(self, ctx: object) -> Any:
        from src.robot.safety import SafetyDecision, SafetyReason

        return SafetyDecision(accepted=True, reason=SafetyReason.OK, guard="test", message="")

    def gate_joint_target(self, joints: object, *, arm: object) -> None:
        return None

    def path_judge_refusal(self, arm: object) -> None:
        return None


class PlannerDouble:
    """A cuRobo planner, the arm's ``_curobo``, as the route asks one: a joint-goal glue whose every straight line comes
    too close, so each move asks it for its plan, ``plan_result``, and it executes to ``execute_result``.

    A plan that has waypoints ends on the configuration it was asked for, as a cuRobo joint plan does: its last
    waypoint is the goal. The route plans joint goals and never asks ``plan(pose)`` (``test_ur_arm._FakePlanner`` until
    P0g slice 13).
    """

    def __init__(self, *, plan_result: object, execute_result: MotionResult | None = None) -> None:
        self._plan_result, self._execute_result = plan_result, execute_result
        self.last_refusal = None

    def plan_joint(self, goal: object, *, start_ur: object = None, refresh: bool = True, near_point_mm: object = None,
                   goal_keep_out: object = None) -> object:
        if isinstance(self._plan_result, Exception):
            raise self._plan_result
        if isinstance(self._plan_result, list) and self._plan_result:
            return [*self._plan_result[:-1], [float(v) for v in goal]]  # type: ignore[attr-defined]
        return self._plan_result

    def refresh_world(self, *, near_point_mm: object = None, goal_keep_out: object = None) -> None:
        """Asked before a judged joint move or line on a live camera world. The double has no world."""
        return None

    def check_joint_path(self, samples: object, *, refresh: bool = True, clearance_mm: float = 0.0) -> object:
        """Asked about a goal, a line and the legs a plan will run. It refuses the lines and nothing else."""
        from src.robot.safety.planning import JointCheckVerdict

        count = len(list(samples))  # type: ignore[call-overload]
        if clearance_mm > 0.0:
            return JointCheckVerdict(valid=False, first_invalid=0, checked=count, reason="the line grazes the tote")
        return JointCheckVerdict(valid=True, first_invalid=None, checked=count, reason="the double accepts")

    def execute(self, traj: object, pose: "Pose | None" = None, *, command: MotionCommand = MotionCommand.MOVE_TO,
                target_joints: object = None, vel: object = None, acc: object = None) -> MotionResult:
        assert self._execute_result is not None
        return self._execute_result


class RecordingPlanner:
    """A planner that records the samples a judged path asks about, and at what clearance, and answers ``verdict``
    (clear by default). It plans nothing, so a line it or the guards refuse is refused: a plan around it has its own
    tests (``tests/robot/safety/test_a_joint_target_goes_to_its_twin_and_around_what_its_line_hits.py``)."""

    def __init__(self, verdict: Any = None) -> None:
        self._verdict = verdict
        self.checked: list[list[float]] | None = None
        self.clearance_mm: float | None = None
        self.last_refusal = None

    def plan_joint(self, goal: Any, *, start_ur: Any = None, refresh: bool = True, near_point_mm: Any = None,
                   goal_keep_out: Any = None) -> None:
        return None

    def check_joint_path(self, samples: Any, *, refresh: bool = True, clearance_mm: float = 0.0) -> Any:
        from src.robot.safety.planning import JointCheckVerdict

        self.checked = [list(s) for s in samples]
        self.clearance_mm = clearance_mm
        if isinstance(self._verdict, Exception):
            raise self._verdict
        if self._verdict is None:
            return JointCheckVerdict(valid=True, first_invalid=None, checked=len(self.checked), reason="clear")
        return self._verdict


def controller_pose(tcp: Sequence[float], *, label: str = "") -> Pose:
    """A controller reading ``[x_m, y_m, z_m, rx, ry, rz]`` as a base-frame pose, as the UR transport converts one."""
    x, y, z, rx, ry, rz = (float(v) for v in tcp)
    return Pose(position_mm=np.array([x * 1000.0, y * 1000.0, z * 1000.0], dtype=np.float64),
                quaternion_xyzw=from_axis_angle(np.array([rx, ry, rz], dtype=np.float64)),
                frame=Frame.BASE, label=label or None)


def controller_matrix(tcp: Sequence[float]) -> np.ndarray:
    """A controller reading as the 4x4 in mm the UR transport's ``controller_tcp_now`` answers (``cv2.Rodrigues``)."""
    import cv2 as cv

    x, y, z, rx, ry, rz = (float(v) for v in tcp)
    rotation, _ = cv.Rodrigues(np.array([rx, ry, rz], dtype=np.float64))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [x * 1000.0, y * 1000.0, z * 1000.0]
    return matrix


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


#: What a test cell that connects declares its flange carries, the owner's cell's: every connect pushes the payload
#: since 2026-10-17 and refuses an unweighed one (the owner's cleanup took the switch that left a bare flange alone).
WEIGHED_PAYLOAD: dict[str, Any] = {"mass_kg": 1.3, "cog_mm": [0.0, 0.0, 65.0]}


def without_guards(preflight: Any, *names: str) -> Any:
    """``preflight`` without the guards called ``names``, for a test that reads something they are no part of.

    Every cell wires every guard since 2026-10-17 (the owner's cleanup took the ``enforce`` switches), so a test stands
    one down by its own hand where its tree switched it off until then: a kit arm stands at configurations the IK
    quality guard calls near-singular (``HERE``'s wrist 2 at 0), and the file reads the gate, not that guard. Returns
    ``preflight``."""
    preflight._guards = tuple(guard for guard in preflight.guards if guard.name not in names)
    return preflight


def with_the_whole_envelope(preflight: Any) -> Any:
    """``preflight`` holding its arm to the arm's whole joint envelope, as every cell's guard did by default until
    2026-10-17, for a file whose scenes stand more than half a turn from home and read something else.

    Every real arm keeps every full-turn joint within half a turn of home since (``GuardNumbers.half_turn_about_home``;
    the Isaac arm states it keeps the whole envelope), and the window has its own tests
    (``test_a_cable_is_never_wound_past_half_a_turn_of_home.py``). Returns ``preflight``."""
    from src.robot.safety.joint_limits import JointLimitGuard

    preflight._guards = tuple(JointLimitGuard(half_turn_about_home=False) if isinstance(guard, JointLimitGuard)
                              else guard for guard in preflight.guards)
    return preflight


def kit_tree(route: str = "ik", *, model: str = "ur5e", ip: str = "127.0.0.1", **blocks: Any) -> RobotConfig:
    """A cell tree on ``route`` for ``model``, the 2F-85 named, with ``blocks`` merged in (``safety={...}``)."""
    base: dict[str, Any] = {"vendor": "ur", "motion": {"planner": route}, "ur": {"ip": ip, "model": model},
                            "gripper": {"model": "robotiq_2f85"}}
    return RobotConfig.model_validate(_merge(base, blocks))


def planned_arm(
    config: RobotConfig,
    *,
    transport: Any = None,
    preflight: Maybe[Any] = UNSET,
    home_joints: Sequence[float] | None = None,
    curobo_client_factory: Maybe[Any] = UNSET,
    controller_yaw_deg: Maybe[int] = UNSET,
    mixins: Sequence[type] = (),
    tool: Maybe[Any] = UNSET,
) -> PlannedArm:
    """A ``PlannedArm`` from ``config``, as a vendor driver builds one: the description of the model the tree names,
    the preflight its safety block builds, the home the description states unless the tree or ``home_joints`` says,
    and the turn between the planner's root and the controller's base the description states unless one is handed in.

    The transport is a connected fake unless one is handed in, and the planner a :class:`FakePlanner` unless a factory
    is, so no kit arm ever starts a sidecar. ``mixins`` are capability mixins of ``drivers/planned/capabilities.py`` the
    arm lists before ``PlannedArm``, as a vendor class does (port S4: ``PlannedFreedrive``). ``tool`` is the
    ``MountedTool`` the arm is built with, the tree's when unset (P0i i5).
    """
    from src.config.arms import load_arm
    from src.robot.safety.planning.hand import planner_hand
    from src.robot.safety.workspace import WorkspaceGuard

    model = config.ur.model
    description = load_arm(model)
    capabilities = RobotCapabilities(
        vendor="ur", model=model, dof=description.dof, controller=config.ur.ip, owns_a_controller=True,
        has_native_fk=True,
        is_simulated=False,
    )
    built_preflight = preflight if chosen(preflight) else SafetyPreflight.from_safety_config(
        config.safety, config.motion.workspace, hand=planner_hand(config), arm_model=model)
    home = (list(home_joints) if home_joints is not None
            else list(config.motion.home_rad or [joint.home_rad for joint in description.joints]))
    # Any: a FakePlanner answers what the arm asks of a cuRobo client without being one.
    planner: Any = curobo_client_factory if chosen(curobo_client_factory) else (lambda: FakePlanner(arm=model))
    cls: Any = PlannedArm if not mixins else type(
        "PlannedArm[" + "+".join(m.__name__ for m in mixins) + "]", (*mixins, PlannedArm), {"__module__": __name__})
    return cls(  # type: ignore[no-any-return]
        config=config,
        logger=logging.getLogger(LOGGER_NAME),
        capabilities=capabilities,
        joint_names=tuple(joint.name for joint in description.joints),
        controller_yaw_deg=(controller_yaw_deg if chosen(controller_yaw_deg)
                            else description.base.controller_yaw_deg),
        transport=transport if transport is not None else fake_transport(),
        guard=WorkspaceGuard(config.motion.workspace),
        preflight=built_preflight,
        home_joints=home,
        curobo_client_factory=planner,
        # The tool on the flange where a test hands one in (P0i i5); else the arm reads the tree's, as a driver does.
        **({"tool": tool} if chosen(tool) else {}),
    )


def bare_arm(motion_planner: str = "ik", *, connected: bool = True) -> PlannedArm:
    """The tree's arm on ``motion_planner``, its preflight the one the tree builds (``test_ur_arm._arm``)."""
    return planned_arm(kit_tree(motion_planner), transport=fake_transport(connected=connected))


def _accepting() -> SafetyPreflight:
    return SafetyPreflight([AcceptingGuard("workspace")])


def ik_arm(joints: Sequence[float] = HERE) -> PlannedArm:
    """An ik-route arm standing at ``joints``, its IK answering them, one accepting guard (``_ur_ik``)."""
    fake = fake_transport(joints=joints)
    fake.always("controller_ik", list(joints))
    return planned_arm(kit_tree("ik"), transport=fake, preflight=_accepting())


def curobo_arm(planner: object | None = None, *, live_world: object | None = None,
               joints: Sequence[float] = HERE) -> PlannedArm:
    """A cuRobo arm planning through a planner double, its plan, path and endpoint gates stood down (``_ur_curobo``).

    The double hands back one trajectory whatever it is asked, which a real planner never does, so what a test on this
    arm reads is not where a plan ends or whether a path is judged: those have their own files.
    """
    fake = fake_transport(joints=joints)
    fake.always("controller_ik", list(joints))
    arm: Any = planned_arm(kit_tree("curobo"), transport=fake, preflight=_accepting())
    if live_world is not None:
        arm.set_live_planner_world(live_world)
    arm._curobo = planner if planner is not None else PlannerDouble(
        plan_result=[list(joints)],
        execute_result=MotionResult.executed(MotionCommand.MOVE_TO, target_pose=kit_pose(), message="curobo"),
    )
    arm._gate_planned_config = lambda pose, joints: None
    arm._plan_end_refusal = lambda goal, joints, pose: None
    arm._preflight.gate_planned_path = lambda waypoints, *, arm=None, command=None: None
    # The send gate asks where the arm came to rest in the path gate's metric, which a preflight with no path guard
    # cannot say (it refuses then): a millimetre a radian here.
    arm._preflight.joint_radii_mm = lambda _arm: (1.0,) * 6
    return arm  # type: ignore[no-any-return]


#: The Isaac cell's tool frame, approaching along flange +Y, which the checked-line and live-world fixtures carry.
_PLUS_Y_TOOL = {"offset_mm": (0.0, 132.0, 0.0), "rotation_quat_xyzw": (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476)}

_LINE_HERE = (0.0, -1.5, 1.5, 0.0, 0.0, 0.0)


class RecordingIk:
    """The arm's ``ik``, answering from a script and keeping every call (pose, seed) on itself, not on the arm."""

    def __init__(self, solutions: Iterable[Sequence[float]] | None) -> None:
        self._solutions = list(solutions) if solutions is not None else None
        self.calls: list[tuple[Pose, JointPositions | None]] = []

    def __call__(self, pose: Pose, *, seed: JointPositions | None = None) -> JointPositions:
        self.calls.append((pose, seed))
        if self._solutions is None:
            return JointPositions(_LINE_HERE)
        return JointPositions(self._solutions[min(len(self.calls), len(self._solutions)) - 1])


def line_arm(*, planner_kind: str = "curobo", tool_source: str = "controller", planner: object | None = None,
             ik_solutions: Iterable[Sequence[float]] | None = None) -> PlannedArm:
    """An arm whose controller answers IK with ``ik_solutions``, one per call, then repeats the last
    (``test_ur_checked_linear._arm``).

    Where it stands, in the frame the controller reports: 300 mm out, 300 mm up, tool down. In controller mode the
    controller holds the tool and reports the grasp centre's rotation, (0, pi, 0); in willy mode it runs a bare flange,
    which the declared frame's -90 degrees about X turns back to tool down, so it reports (0, pi/sqrt(2), -pi/sqrt(2)).
    """
    config = kit_tree(planner_kind, gripper={"tool_frame": {"source": tool_source, **_PLUS_Y_TOOL}})
    fake = fake_transport(joints=_LINE_HERE)
    here = [0.3, 0.0, 0.3, 0.0, 3.14159, 0.0]
    fake.always("controller_tcp_now", controller_matrix(here))
    fake.always("controller_fk", controller_pose(here))
    rotation = (0.0, 3.14159265, 0.0) if tool_source == "controller" else (0.0, 2.221441469079183, -2.221441469079183)
    fake.always("read_tcp", controller_pose([0.3, 0.0, 0.3, *rotation]))
    arm: Any = planned_arm(config, transport=fake)
    without_guards(arm._preflight, "ik_quality")
    arm.ik = RecordingIk(ik_solutions)
    if planner_kind == "curobo":
        arm._curobo = planner if planner is not None else RecordingPlanner()
    return arm  # type: ignore[no-any-return]


def steady_arm(*, steady: bool = True, connected: bool = True) -> PlannedArm:
    """An ik-route arm whose steady gate answers ``steady``, its preflight the steady-gate file's stand-in
    (``test_the_arm_holds_its_own_steady_gate._arm``)."""
    config = kit_tree("ik")
    fake = fake_transport(joints=HERE, connected=connected)
    fake.always("wait_until_steady", steady)
    fake.always("controller_fk", controller_pose([0.3, 0.0, 0.3, 0.0, 3.14159, 0.0]))
    arm: Any = planned_arm(config, transport=fake, preflight=AcceptAllPreflight())
    arm.ik = lambda pose, *, seed=None: JointPositions(HERE)
    return arm  # type: ignore[no-any-return]


def _recording(call: Any, label: str, events: list[str]) -> Any:
    def recorded(*args: Any, **kwargs: Any) -> Any:
        events.append(label)
        return call(*args, **kwargs)

    return recorded


def live_arm(world: Any, client: Any, events: list[str], *,
             here: Sequence[float] = (0.0, -1.5, 1.5, 0.0, 0.0, 0.0),
             there: Sequence[float] = (0.2, -1.5, 1.5, 0.0, 0.0, 0.0)) -> PlannedArm:
    """A cuRobo arm over the fake, its planner the real ``CuroboPlanner`` on ``client``, its real guard recording the
    order in ``events`` (``test_every_verb_meets_the_camera_world._ur``).

    The guard is the one the tree builds. Three of its calls are wrapped rather than replaced, so ``events`` says when
    the perceived world reached it and when each path gate ran. The plan's end is stood down: the client hands back
    one trajectory whatever it is asked, which a real planner never does, and where a plan ends has its own file.
    """
    config = kit_tree("curobo", safety={
        # The margin every cuRobo cell must declare (B1 S17); the evidence check reads it to find its file (S22).
        "self_collision": {"planner_margin_mm": 4.0},
    }, gripper={"tool_frame": {"source": "controller", **_PLUS_Y_TOOL}})
    fake = fake_transport(joints=here)
    fake.always("controller_fk", controller_pose([0.4, 0.0, 0.3, 0.0, 3.14159, 0.0]))
    fake.always("read_tcp", controller_pose([0.3, 0.0, 0.3, 0.0, 3.14159265, 0.0]))
    arm: Any = planned_arm(config, transport=fake, home_joints=there, curobo_client_factory=lambda: client)
    arm.ik = lambda pose, *, seed=None: JointPositions(here)
    arm._plan_end_refusal = lambda goal, joints, pose: None
    guard = without_guards(arm._preflight, "ik_quality")
    for name, label in (("set_perceived_obstacles", "world"), ("gate_planned_path", "gate_planned_path"),
                        ("gate_joint_path", "gate_joint_path")):
        setattr(guard, name, _recording(getattr(guard, name), label, events))
    arm.set_live_planner_world(world)
    return arm  # type: ignore[no-any-return]

