"""A camera that cannot vouch for the cell stops every verb, and the path guard judges the refreshed cell.

Owner, Step 4: a camera that stays silent, blind or stale after its fresh-frame attempts raises out of
every motion verb on a driver that plans, and nothing is commanded. The raise itself lives in the
refresh (tests/test_live_planner_world.py FreshFrameAttemptsTests); this file holds the verbs to it.

The second half is order. A joint move and a straight line on a cuRobo cell are judged by two
authorities, the local path guard and the planner, and the guard learns the perceived obstacles from the
refresh. The refresh used to run inside the planner's check, after the guard had already judged the
path, so the guard judged every path against the obstacles of the motion before. The refresh now runs
first and once, and both authorities judge the same cell.

The sim joint verbs are held here as well (owner, Q11): `move_joint`, and `move_home` through it, judge
the whole line on a cuRobo sim as the UR does, where they checked the destination alone.

And the stamp (owner, Q9): a move on a world its own refresh vouched for says PLANNED, naming the cameras
and the capture time of the oldest image, and a move no refresh vouched for keeps saying UNSTATED.

No test here starts Isaac or a sidecar. The UR arms run on a fake controller and an injected client. The
sim arms are never connected and carry a MagicMock guard, so they are non-UR doubles: every UR double in
this repository once hid a sim defect.
"""

from __future__ import annotations

import time
import unittest
from collections.abc import Sequence
from unittest.mock import MagicMock, patch

import numpy as np

from src.config.schema.robot import RobotConfig
from src.geometry import Frame, Pose
from src.robot.core import JointPositions, MotionCommand, MotionResult, MotionStatus
from src.robot.core.camera_world import CameraWorldUse
from src.robot.core.errors import CameraWorldUnavailable, RobotMotionRejected
from src.robot.drivers.sim.arm import IsaacRobotArm
from src.robot.drivers.sim.config import SimRobotConfig
from src.robot.drivers.ur import curobo_motion
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.drivers.ur.curobo_motion import UR_ARM_JOINT_NAMES
from src.robot.drivers.ur.pose import URPose
from src.robot.safety._fcl_self_collision import mesh_backend_status
from src.robot.safety._ur_kinematics import ur_link_origins_mm
from src.robot.safety.planning import JointCheckVerdict
from src.robot.safety.planning.live_world import CameraView, DepthSnapshot, LivePlannerWorld
from src.robot.safety.planning.perceived import LinkCapsule, SelfEnvelope
from src.robot.safety.planning.perceived import WorldBuildLimits
from src.robot.safety.planning.world import planner_cuboid

#: The decline these path tests run under: they read the planner and the guard, and no camera world is wired.
_PATH_ONLY = "unit double: this test reads the path judge, and no camera world is wired to the arm"

#: Clear, and where the fake controller says the arm is standing.
_HERE = (0.0, -1.5, 1.5, 0.0, 0.0, 0.0)
#: Clear, a short joint move away (tests/test_ur_checked_joint_verbs.py measured both).
_THERE = (0.2, -1.5, 1.5, 0.0, 0.0, 0.0)
_LIMITS = WorldBuildLimits(
    x_mm=(-400.0, 400.0), y_mm=(-400.0, 400.0), z_mm=(-50.0, 900.0), support_plane_top_mm=0.0
)
_BENCH = (planner_cuboid("support_plane", (0.0, 0.0, -25.0), (1600.0, 1600.0, 50.0)),)


def _mesh_backend_available() -> bool:
    return mesh_backend_status("ur5e") == "ok"


def _pose(x: float = 400.0) -> Pose:
    return Pose(
        position_mm=np.array([x, 0.0, 300.0], dtype=np.float64),
        quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
        frame=Frame.BASE,
    )


class _Camera:
    """Silent, or a bench a metre below it with nothing on it, stamped at the moment it is asked."""

    def __init__(self, *, silent: bool) -> None:
        self.silent = silent
        self.grabs = 0

    def grab_surface_depth(self) -> DepthSnapshot | None:
        self.grabs += 1
        if self.silent:
            return None
        return DepthSnapshot(
            depth_mm=np.full((40, 40), 1000.0, dtype=np.float64),
            intrinsics=np.array([[50.0, 0.0, 20.0], [0.0, 50.0, 20.0], [0.0, 0.0, 1.0]]),
            timestamp=time.time(),
        )


def _world(camera: _Camera) -> LivePlannerWorld:
    return LivePlannerWorld(
        cameras=(CameraView(name="overhead", depth_source=camera, camera_to_base=np.eye(4)),),
        declared=_BENCH,
        limits=_LIMITS,
        fresh_frame_attempts=1,
    )


class _Client:
    """A sidecar under the test's control: it confirms what it is sent, or one box fewer."""

    def __init__(self, *, joint_names: Sequence[str] | None = None, confirm_all: bool = True) -> None:
        if joint_names is not None:
            self.joint_names = list(joint_names)
        self.confirm_all = confirm_all
        self.checked: list[list[list[float]]] = []
        # What a sidecar on the ur5e descriptor with the 2F-85 added says of itself: a planner refuses one that says
        # nothing, so without it every UR verb here was refused at start before its world was asked.
        from tests._sidecar_identity import arm_identity

        self.identity = arm_identity()

    def start(self) -> None:
        return None

    def set_world(self, cuboids, meshes=None):  # noqa: ANN001, ANN201
        return len(cuboids) if self.confirm_all else max(0, len(cuboids) - 1)

    def check_joints(self, configs):  # noqa: ANN001, ANN201
        self.checked.append([[float(v) for v in c] for c in configs])
        return JointCheckVerdict(valid=True, first_invalid=None, checked=len(configs), reason="clear")

    def plan(self, start, pos_m, quat_wxyz):  # noqa: ANN001, ANN201
        return [list(start)]

    def close(self) -> None:
        return None


def _mock(value: object) -> MagicMock:
    """A double this file installed, typed as the double it is."""
    assert isinstance(value, MagicMock), value
    return value


def _recording(call, label: str, events: list[str]):  # noqa: ANN001, ANN202
    def recorded(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        events.append(label)
        return call(*args, **kwargs)

    return recorded


def _ur(world: LivePlannerWorld, client: _Client, events: list[str]) -> URRobotArm:
    """A cuRobo UR on a fake controller, its planner on `client`, its real guard recording the order.

    The guard is the one the config builds. Three of its calls are wrapped rather than replaced, so
    `events` says when the perceived world reached it and when each path gate ran.
    """
    config = RobotConfig.model_validate({
        "vendor": "ur",
        "ur": {"motion_planner": "curobo"},
        "safety": {"payload": {"enforce": False}, "ik_quality": {"enforce": False},
                   # The margin every UR cuRobo cell must declare (B1 S17). A stub client skips the factory that refuses an undeclared one,
                   # so this used to pass without it; the evidence check reads the margin to find its file (S22), and meets it here.
                   "self_collision": {"planner_margin_mm": 4.0}},
        # The Isaac cell's frame, approaching along flange +Y. From Step 4g until UM lane S23 a UR declaring the real
        # flange's +Z refused to build, which is why this fixture carries +Y; for this tool-down pose it puts the
        # flange where the +Z frame did, which is what the near-point assertions read.
        "gripper": {"model": "robotiq_2f85", "tool_frame": {
            "source": "polyscope",
            "offset_mm": (0.0, 132.0, 0.0),
            "rotation_quat_xyzw": (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476),
        }},
    })
    arm = URRobotArm(config)
    arm._conn = MagicMock()
    arm._conn.is_connected = True
    arm._conn.get_joint_positions.return_value = list(_HERE)
    arm._conn.fk.return_value = [0.4, 0.0, 0.3, 0.0, 3.14159, 0.0]
    arm._conn.moveJ.return_value = True
    arm._motion = MagicMock()
    arm._motion.move_to.return_value = True
    arm._motion._clamp.return_value = (0.5, 0.5)
    arm._motion.get_current_pose.return_value = URPose.from_ur_list([0.3, 0.0, 0.3, 0.0, 3.14159265, 0.0])
    arm.ik = lambda pose, *, seed=None: JointPositions(_HERE)  # type: ignore[method-assign]
    arm._home_joints = list(_THERE)
    arm._curobo_client_factory = lambda: client  # type: ignore[assignment, return-value]
    # And the plan's end. This client hands back one trajectory whatever it is asked, which a real planner never
    # does, and what this file reads is not where a plan ends: the UR driver refuses a plan off its goal before
    # anything moves (Step 8f), and tests/test_the_planner_sees_the_cell_where_the_controller_has_it.py holds that half.
    arm._plan_end_refusal = lambda goal, joints, pose: None  # type: ignore[method-assign]
    guard = arm._preflight
    for name, label in (
        ("set_perceived_obstacles", "world"),
        ("gate_planned_path", "gate_planned_path"),
        ("gate_joint_path", "gate_joint_path"),
    ):
        setattr(guard, name, _recording(getattr(guard, name), label, events))
    arm.set_live_planner_world(world)
    return arm


def _sim(*, client: _Client, world: LivePlannerWorld | None = None) -> IsaacRobotArm:
    """A cuRobo sim arm Isaac never starts for, on a MagicMock guard that accepts and a fake sidecar."""
    arm = IsaacRobotArm(SimRobotConfig(enabled=True, scene="placeholder.usd", robot_prim_path="/World/Robot"))
    arm._connected = True
    arm._articulation = object()  # type: ignore[assignment]
    arm._kin_solver = object()  # type: ignore[assignment]
    arm._rmpflow = object()  # type: ignore[assignment]
    arm._arm_subset = MagicMock()
    arm._arm_subset.get_joint_positions.return_value = np.asarray(_HERE, dtype=np.float64)
    arm._resolve_ik = lambda pose, **_: JointPositions(_HERE)  # type: ignore[method-assign]
    arm.get_joint_positions = lambda: JointPositions(_HERE)  # type: ignore[method-assign]
    arm.get_tcp_pose = lambda: _pose(x=300.0)  # type: ignore[method-assign]
    arm._get_curobo_client = lambda: client  # type: ignore[method-assign, assignment, return-value]
    arm._self_envelope = lambda: SelfEnvelope(  # type: ignore[method-assign]
        frames_mm=(np.eye(4),),
        capsules=(LinkCapsule(frame=0, start_mm=(0.0, 0.0, 0.0), end_mm=(0.0, 0.0, 300.0), radius_mm=135.0),),
    )
    arm._drive_joints = MagicMock()  # type: ignore[method-assign]
    arm._home_joints = JointPositions(_THERE)
    guard = MagicMock()
    guard.path_step_mm = 10.0
    guard.joint_radii_mm.return_value = (500.0,) * 6
    for name in ("gate_planned_path", "gate_joint_path", "gate_joint_target"):
        getattr(guard, name).return_value = None
    # The guard is a MagicMock, so it has to be told it places no model: the near point of a joint move then keeps the
    # configured model, unturned, as it did before the sim took the guard's yaw (Step 4h).
    guard.self_kinematics.return_value = None
    arm._preflight = guard
    arm.set_live_planner_world(world)
    return arm


def _calls(guard: MagicMock) -> list[str]:
    return [name for name, _, _ in guard.mock_calls]


class EveryUrVerbRaisesTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def test_every_ur_verb_raises_on_a_camera_that_stays_silent(self) -> None:
        verbs = {
            "move": lambda arm: arm.move(_pose()),
            "move, linear": lambda arm: arm.move(_pose(), linear=True),
            "move_to": lambda arm: arm.move_to(_pose()),
            "move_linear": lambda arm: arm.move_linear(_pose()),
            "move_to_joints": lambda arm: arm.move_to_joints(JointPositions(_THERE)),
            "move_joint": lambda arm: arm.move_joint(JointPositions(_THERE)),
            "move_home": lambda arm: arm.move_home(),
        }
        for label, verb in verbs.items():
            with self.subTest(label):
                camera = _Camera(silent=True)
                arm = _ur(_world(camera), _Client(joint_names=UR_ARM_JOINT_NAMES), [])

                with self.assertRaises(CameraWorldUnavailable):
                    verb(arm)

                self.assertEqual(camera.grabs, 2, "the first reading and its one fresh-frame attempt")
                _mock(arm._conn).moveJ.assert_not_called()
                _mock(arm._motion).move_to.assert_not_called()

    def test_a_partial_registration_is_still_an_ordinary_refusal(self) -> None:
        """The control, green before and after: a sidecar that confirms one box fewer refuses, no raise."""
        verbs = {
            "move": lambda arm: arm.move(_pose()),
            "move_to_joints": lambda arm: arm.move_to_joints(JointPositions(_THERE)),
        }
        for label, verb in verbs.items():
            with self.subTest(label):
                arm = _ur(
                    _world(_Camera(silent=False)),
                    _Client(joint_names=UR_ARM_JOINT_NAMES, confirm_all=False),
                    [],
                )

                result = verb(arm)

                self.assertIs(result.status, MotionStatus.CONTROLLER_REJECTED, result.message)
                _mock(arm._conn).moveJ.assert_not_called()


class TheUrPathGuardSeesTheRefreshedWorldTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def test_a_joint_move_refreshes_once_before_the_path_gate_near_the_goal_flange(self) -> None:
        events: list[str] = []
        arm = _ur(_world(_Camera(silent=False)), _Client(joint_names=UR_ARM_JOINT_NAMES), events)

        with patch.object(
            curobo_motion, "refresh_planner_world", wraps=curobo_motion.refresh_planner_world
        ) as refresh:
            result = arm.move_to_joints(JointPositions(_THERE))

        self.assertIs(result.status, MotionStatus.EXECUTED, result.message)
        self.assertEqual(events.count("world"), 1, events)
        self.assertLess(events.index("world"), events.index("gate_planned_path"), events)
        origins = ur_link_origins_mm(str(arm.config.ur.model), np.asarray(_THERE, dtype=np.float64))
        assert origins is not None
        np.testing.assert_allclose(refresh.call_args.kwargs["near_point_mm"], origins[-1], atol=1e-6)

    def test_a_line_refreshes_once_before_the_path_gate_near_the_goal_flange(self) -> None:
        events: list[str] = []
        arm = _ur(_world(_Camera(silent=False)), _Client(joint_names=UR_ARM_JOINT_NAMES), events)

        with patch.object(
            curobo_motion, "refresh_planner_world", wraps=curobo_motion.refresh_planner_world
        ) as refresh:
            arm.move_linear(_pose(x=400.0))

        self.assertEqual(events.count("world"), 1, events)
        self.assertLess(events.index("world"), events.index("gate_joint_path"), events)
        # The tool points down, so the flange stands its 132 mm above the grasp centre.
        np.testing.assert_allclose(
            refresh.call_args.kwargs["near_point_mm"], [400.0, 0.0, 432.0], atol=1e-6
        )


class TheSimVerbsTests(unittest.TestCase):
    def test_the_sim_verbs_raise_on_a_camera_that_stays_silent(self) -> None:
        verbs = {
            "move": lambda arm: arm.move(_pose()),
            "move, linear": lambda arm: arm.move(_pose(), linear=True),
            "move_to_joints": lambda arm: arm.move_to_joints(JointPositions(_THERE)),
            "move_joint": lambda arm: arm.move_joint(JointPositions(_THERE)),
            "move_home": lambda arm: arm.move_home(),
        }
        for label, verb in verbs.items():
            with self.subTest(label):
                camera = _Camera(silent=True)
                arm = _sim(client=_Client(), world=_world(camera))

                with self.assertRaises(CameraWorldUnavailable):
                    verb(arm)

                self.assertEqual(camera.grabs, 2)
                _mock(arm._drive_joints).assert_not_called()

    def test_the_sim_joint_check_refreshes_first(self) -> None:
        arm = _sim(client=_Client(), world=_world(_Camera(silent=False)))

        result = arm.move_to_joints(JointPositions(_THERE))

        self.assertIs(result.status, MotionStatus.EXECUTED, result.message)
        calls = _calls(_mock(arm._preflight))
        self.assertEqual(calls.count("set_perceived_obstacles"), 1, calls)
        self.assertLess(calls.index("set_perceived_obstacles"), calls.index("gate_planned_path"), calls)

    def test_the_sim_line_refreshes_before_its_path_gate(self) -> None:
        arm = _sim(client=_Client(), world=_world(_Camera(silent=False)))

        result = arm.move(_pose(x=400.0), linear=True)

        self.assertIs(result.status, MotionStatus.EXECUTED, result.message)
        calls = _calls(_mock(arm._preflight))
        self.assertEqual(calls.count("set_perceived_obstacles"), 1, calls)
        self.assertLess(calls.index("set_perceived_obstacles"), calls.index("gate_joint_path"), calls)

    def test_a_refused_sim_refresh_refuses_the_joint_move_before_any_gate(self) -> None:
        arm = _sim(client=_Client(confirm_all=False), world=_world(_Camera(silent=False)))

        result = arm.move_to_joints(JointPositions(_THERE))

        self.assertIs(result.status, MotionStatus.CONTROLLER_REJECTED, result.message)
        self.assertIn("planner world not refreshed", result.message or "")
        _mock(arm._preflight).gate_planned_path.assert_not_called()
        _mock(arm._drive_joints).assert_not_called()

    def test_a_sim_arm_that_cannot_read_its_joints_cannot_place_its_links(self) -> None:
        """The UR twin answers None here; the sim raised the articulation's error into the refresh."""
        arm = IsaacRobotArm(SimRobotConfig(enabled=True, scene="placeholder.usd", robot_prim_path="/World/Robot"))
        arm._arm_subset = MagicMock()
        arm._arm_subset.get_joint_positions.side_effect = RuntimeError("the articulation is gone")

        self.assertIsNone(arm._self_envelope())


class TheSimMoveJointIsPathCheckedTests(unittest.TestCase):
    """Owner, Q11: the public sim joint verb judges the line, not the destination alone.

    These arms plan with cuRobo and hold no world, so each test declines: without it the move is refused before the
    line is judged, and a refused path would read as the camera world's refusal.
    """

    @staticmethod
    def _refusal() -> MotionResult:
        return MotionResult.failed(
            MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_JOINTS,
            target_joints=JointPositions(_THERE), message="forearm|lfinger: mesh distance 0.367 mm",
        )

    def test_the_sim_move_joint_is_path_checked(self) -> None:
        client = _Client()
        arm = _sim(client=client)
        self.enterContext(arm.without_camera_world(_PATH_ONLY))

        arm.move_joint(JointPositions(_THERE))

        _mock(arm._preflight).gate_planned_path.assert_called_once()
        self.assertEqual(len(client.checked), 1, "the planner judged the line once")
        np.testing.assert_allclose(client.checked[0][0], _HERE, atol=1e-9)
        np.testing.assert_allclose(client.checked[0][-1], _THERE, atol=1e-9)
        _mock(arm._drive_joints).assert_called_once()

    def test_a_refused_path_raises_with_the_typed_refusal_and_drives_nothing(self) -> None:
        arm = _sim(client=_Client())
        self.enterContext(arm.without_camera_world(_PATH_ONLY))
        refusal = self._refusal()
        _mock(arm._preflight).gate_planned_path.return_value = refusal

        with self.assertRaises(RobotMotionRejected) as caught:
            arm.move_joint(JointPositions(_THERE))

        self.assertIs(caught.exception.result, refusal)
        _mock(arm._drive_joints).assert_not_called()

    def test_move_home_reports_a_refused_path_rather_than_raising(self) -> None:
        arm = _sim(client=_Client())
        self.enterContext(arm.without_camera_world(_PATH_ONLY))
        _mock(arm._preflight).gate_planned_path.return_value = self._refusal()

        self.assertFalse(arm.move_home())
        _mock(arm._drive_joints).assert_not_called()


class AMoveOnAVouchedWorldSaysPlannedTests(unittest.TestCase):
    """Owner, Q9: PLANNED names the cameras and the capture time of the refresh this motion made."""

    def test_a_ur_move_on_a_vouched_world_says_planned(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")
        arm = _ur(_world(_Camera(silent=False)), _Client(joint_names=UR_ARM_JOINT_NAMES), [])
        asked = time.time()

        result = arm.move(_pose())

        self.assertIs(result.status, MotionStatus.EXECUTED, result.message)
        stamp = result.camera_world
        self.assertIs(stamp.use, CameraWorldUse.PLANNED, stamp.render())
        self.assertEqual(stamp.cameras, ("overhead",))
        assert stamp.captured_at_s is not None
        self.assertGreaterEqual(stamp.captured_at_s, asked)
        self.assertLessEqual(stamp.captured_at_s, time.time())

    def test_a_later_move_that_refreshed_nothing_does_not_borrow_the_stamp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")
        arm = _ur(_world(_Camera(silent=False)), _Client(joint_names=UR_ARM_JOINT_NAMES), [])
        self.assertIs(arm.move(_pose()).camera_world.use, CameraWorldUse.PLANNED)
        elsewhere = Pose(
            position_mm=np.array([400.0, 0.0, 300.0]), quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
            frame=Frame.TOOL,
        )

        refused = arm.move(elsewhere)

        self.assertIs(refused.status, MotionStatus.INVALID_TARGET)
        self.assertIs(refused.camera_world.use, CameraWorldUse.UNSTATED, refused.camera_world.render())

    def test_a_refused_refresh_vouches_for_nothing(self) -> None:
        """The control, green before and after: a refresh that refused never stamps PLANNED."""
        arm = _ur(
            _world(_Camera(silent=False)), _Client(joint_names=UR_ARM_JOINT_NAMES, confirm_all=False), []
        )

        result = arm.move(_pose())

        self.assertIs(result.status, MotionStatus.CONTROLLER_REJECTED, result.message)
        self.assertIs(result.camera_world.use, CameraWorldUse.UNSTATED)

    def test_a_sim_move_on_a_vouched_world_says_planned(self) -> None:
        arm = _sim(client=_Client(), world=_world(_Camera(silent=False)))

        result = arm.move(_pose(x=400.0), linear=True)

        self.assertIs(result.status, MotionStatus.EXECUTED, result.message)
        self.assertIs(result.camera_world.use, CameraWorldUse.PLANNED, result.camera_world.render())
        self.assertEqual(result.camera_world.cameras, ("overhead",))

    def test_a_sim_joint_move_on_a_vouched_world_says_planned(self) -> None:
        """Owner, Step 4 C: the sim checks a joint move's path against the world its refresh registered."""
        arm = _sim(client=_Client(), world=_world(_Camera(silent=False)))

        result = arm.move_to_joints(JointPositions(_THERE))

        self.assertIs(result.status, MotionStatus.EXECUTED, result.message)
        self.assertIs(result.camera_world.use, CameraWorldUse.PLANNED, result.camera_world.render())
        self.assertEqual(result.camera_world.cameras, ("overhead",))

    def test_a_sim_joint_move_whose_refresh_refused_vouches_for_nothing(self) -> None:
        """The control, green before and after: a refused refresh never stamps PLANNED."""
        arm = _sim(client=_Client(confirm_all=False), world=_world(_Camera(silent=False)))

        result = arm.move_to_joints(JointPositions(_THERE))

        self.assertIs(result.status, MotionStatus.CONTROLLER_REJECTED, result.message)
        self.assertIsNot(result.camera_world.use, CameraWorldUse.PLANNED)

    def test_a_ur_joint_move_on_a_vouched_world_says_planned(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")
        arm = _ur(_world(_Camera(silent=False)), _Client(joint_names=UR_ARM_JOINT_NAMES), [])

        result = arm.move_to_joints(JointPositions(_THERE))

        self.assertIs(result.status, MotionStatus.EXECUTED, result.message)
        self.assertIs(result.camera_world.use, CameraWorldUse.PLANNED, result.camera_world.render())


class AKeepOutScopeDoesNotLoosenADeclineTests(unittest.TestCase):
    """Holding a target out of the world changes the world, never the rule that a decline on a live world is refused."""

    def test_a_declined_sim_motion_inside_a_scope_is_still_refused(self) -> None:
        from src.robot.core.camera_world import DECLINE_ON_A_LIVE_WORLD_MESSAGE, CameraWorldDecline
        from src.robot.core.keep_out import SegmentationOffer, keeping_out

        world = _world(_Camera(silent=False))
        arm = _sim(client=_Client(), world=world)
        offer = SegmentationOffer(captured_at_s=time.time(), target_points_base_mm=np.array([[0.0, 0.0, 100.0]]))
        with keeping_out(arm, offer) as scope:
            self.assertTrue(scope.world_wired)
            result = arm.move(_pose(x=300.0), camera_world=CameraWorldDecline("a declined move inside a pick"))
        self.assertIs(MotionStatus.UNSUPPORTED, result.status)
        self.assertEqual(DECLINE_ON_A_LIVE_WORLD_MESSAGE, result.message)

    @unittest.skipUnless(_mesh_backend_available(), "the UR arm's path guard needs the exact mesh engine")
    def test_a_declined_ur_motion_inside_a_scope_is_still_refused(self) -> None:
        from src.robot.core.camera_world import DECLINE_ON_A_LIVE_WORLD_MESSAGE, CameraWorldDecline
        from src.robot.core.keep_out import SegmentationOffer, keeping_out

        world = _world(_Camera(silent=False))
        arm = _ur(world, _Client(joint_names=UR_ARM_JOINT_NAMES), [])
        offer = SegmentationOffer(captured_at_s=time.time(), target_points_base_mm=np.array([[0.0, 0.0, 100.0]]))
        with keeping_out(arm, offer):
            result = arm.move(_pose(x=300.0), camera_world=CameraWorldDecline("a declined move inside a pick"))
        self.assertIs(MotionStatus.UNSUPPORTED, result.status)
        self.assertEqual(DECLINE_ON_A_LIVE_WORLD_MESSAGE, result.message)


if __name__ == "__main__":
    unittest.main()
