"""Every driver stamps the camera world on its own motions, and a decline is bound to one arm.

The owner decided that with cuRobo every motion plans against a world built from a current camera
image unless the caller declines, and that the answer travels with the motion. This file pins what each
driver says today, before any world is handed to a cell: a motion no planner planned says UNPLANNED and
why, a planned motion with no camera world says MISSING, a caller's decline says DECLINED, and a
decline on an arm whose live camera world is wired is refused before the planner is asked.

The stamp is read off the built arm, never off config: the console_dummy profile leaves the base tree's
``motion_planner: curobo`` on a dummy arm, and a dummy plans nothing.

No test here starts Isaac. The sim arms are either the mock or a non-mock arm that is never connected,
the shape ``tests/test_sim_driver_skeleton.py`` already drives.
"""

from __future__ import annotations

import inspect
import io
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from unittest.mock import MagicMock, patch

from src.config import load_robot_config
from src.contracts import UNSET
from src.geometry import Pose
from src.robot.core import (
    JointPositions,
    MotionCommand,
    MotionResult,
    MotionStatus,
    RobotArm,
)
from src.robot.core import camera_world as cw
from src.robot.core.camera_world import CameraWorldDecline, CameraWorldStamp, CameraWorldUse
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.drivers.kuka.arm import KukaRobotArm
from src.robot.drivers.sim.arm import IsaacRobotArm
from src.robot.drivers.sim.config import SimRobotConfig
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.safety import SafetyPreflight
from src.robot.safety.planning import CuroboUnavailableError
from tests.test_ungated_motion_surfaces import _AcceptingGuard, _kuka_arm
from tests.test_ur_arm import _FakePlanner, _pose
from tests.test_ur_arm import _arm as _ur_arm

_JOINTS = JointPositions([0.0, -1.5, 1.5, 0.0, 1.5, 0.0])

_UR_IK = CameraWorldStamp.unplanned("robot.ur.motion_planner is 'ik'")
_UR_NO_WORLD = CameraWorldStamp.missing(
    "robot.ur.motion_planner is 'curobo' and no live camera world is wired to this arm")
_KUKA = CameraWorldStamp.unplanned("KukaRobotArm has no planner")
_DUMMY = CameraWorldStamp.unplanned("DummyRobotArm has no planner")
_SIM_MOCK = CameraWorldStamp.unplanned("mock_mode: the kinematic mock has no planner")
_SIM_NO_WORLD = CameraWorldStamp.missing(
    "cuRobo plans this motion and no live camera world is wired to this arm")
_SIM_INTERPOLATED = CameraWorldStamp.unplanned(
    "this joint move is interpolated in joint space, and no planner plans it")
_SIM_DEGRADED = CameraWorldStamp.unplanned(
    "cuRobo is configured and unavailable, so this arm drives the ik path: no cuRobo environment "
    "on this host")

_BENCH = CameraWorldDecline("bench check, no cameras mounted")
_VOUCHED = CameraWorldStamp.planned(cameras=("overhead",), captured_at_s=100.0)


# ---------------------------------------------------------------------------------------------------
# Arms, each driving its body with nothing below it but doubles
# ---------------------------------------------------------------------------------------------------


def _ur_ik() -> URRobotArm:
    arm = _ur_arm("ik")
    arm.ik = lambda pose, **_: _JOINTS  # type: ignore[method-assign]
    arm.get_joint_positions = lambda: _JOINTS  # type: ignore[method-assign]
    arm._preflight = SafetyPreflight([_AcceptingGuard("workspace")])
    arm._drive_pose = lambda *a, **k: True  # type: ignore[method-assign]
    arm._motion.last_reject_status = None
    return arm


def _ur_curobo(planner: object | None = None, *, live_world: object | None = None) -> URRobotArm:
    arm = _ur_arm("curobo")
    arm._preflight = SafetyPreflight([_AcceptingGuard("workspace")])
    if live_world is not None:
        arm.set_live_planner_world(live_world)  # type: ignore[arg-type]
    arm._curobo_ur = planner if planner is not None else _FakePlanner(  # type: ignore[assignment]
        plan_result=[_JOINTS.tolist()],
        execute_result=MotionResult.executed(MotionCommand.MOVE_TO, target_pose=_pose(),
                                             message="curobo"),
    )
    arm._gate_planned_config = lambda pose, joints: None  # type: ignore[method-assign]
    # The path gate too. This file is about the camera world stamp, and the preflight above holds
    # one accepting stand-in rather than a self collision guard, which a judged path refuses for
    # its own good reasons (tests/test_planned_paths_are_judged.py holds that half).
    arm._preflight.gate_planned_path = (  # type: ignore[method-assign]
        lambda waypoints, *, arm=None, command=None: None
    )
    return arm


def _kuka() -> KukaRobotArm:
    arm, _ = _kuka_arm(SafetyPreflight([_AcceptingGuard()]))
    arm.ik = lambda pose, **_: _JOINTS  # type: ignore[method-assign]
    arm.get_joint_positions = lambda: _JOINTS  # type: ignore[method-assign]
    arm._drive_pose = lambda *a, **k: True  # type: ignore[method-assign]
    return arm


def _dummy() -> DummyRobotArm:
    arm = DummyRobotArm()
    arm.connect()
    return arm


def _sim_mock() -> IsaacRobotArm:
    arm = IsaacRobotArm(SimRobotConfig(enabled=True, mock_mode=True))
    arm.connect()
    return arm


def _sim_unconnected() -> IsaacRobotArm:
    """A non-mock sim arm with its connected flag forced and no articulation: Isaac never starts.

    Its cuRobo client is a tripwire, so a test that reaches the planner fails rather than spawning a
    sidecar.
    """
    arm = IsaacRobotArm(SimRobotConfig(enabled=True, scene="placeholder.usd",
                                       robot_prim_path="/World/Robot"))
    arm._connected = True
    arm._get_curobo_client = MagicMock(  # type: ignore[method-assign]
        side_effect=AssertionError("the planner was asked"))
    return arm


# ---------------------------------------------------------------------------------------------------
# 1. The stamp matrix
# ---------------------------------------------------------------------------------------------------


class TheStampMatrixTests(unittest.TestCase):
    """What each driver says for each typed verb, as literals."""

    def test_every_driver_and_verb_says_what_stood_behind_the_motion(self) -> None:
        cases = (
            ("ur ik move", lambda: _ur_ik().move(_pose()), _UR_IK),
            ("ur ik move_to_joints", lambda: _ur_ik().move_to_joints(_JOINTS), _UR_IK),
            ("ur curobo move, no live world", lambda: _ur_curobo().move(_pose()), _UR_NO_WORLD),
            ("ur curobo move_to_joints, no live world", lambda: _ur_curobo().move_to_joints(_JOINTS), _UR_NO_WORLD),
            ("kuka move", lambda: _kuka().move(_pose()), _KUKA),
            ("kuka move_to_joints", lambda: _kuka().move_to_joints(_JOINTS), _KUKA),
            ("dummy move", lambda: _dummy().move(_pose()), _DUMMY),
            ("dummy move_to_joints", lambda: _dummy().move_to_joints(_JOINTS), _DUMMY),
            ("sim mock move", lambda: _sim_mock().move(_pose()), _SIM_MOCK),
            ("sim mock move_to_joints", lambda: _sim_mock().move_to_joints(_JOINTS), _SIM_MOCK),
        )
        for label, command, expected in cases:
            with self.subTest(label):
                result = command()
                self.assertIsInstance(result, MotionResult)
                self.assertEqual(result.camera_world, expected)

    def test_a_live_world_row_whose_planner_reports_no_refresh_stays_unstated(self) -> None:
        """Nothing vouched: this planner double reports no refresh, so the row cannot say PLANNED.

        Until Step 4 this row said UNSTATED because no refresh reported cameras and a capture time. A
        refresh does now, and a move on a world its refresh vouched for says PLANNED
        (tests/test_every_verb_meets_the_camera_world.py). What stays is the control: a stamp that
        vouches needs a refresh that did.
        """
        result = _ur_curobo(live_world=object()).move(_pose())
        self.assertIs(result.status, MotionStatus.EXECUTED)
        self.assertIs(result.camera_world.use, CameraWorldUse.UNSTATED)

    def test_a_ur_joint_move_whose_refresh_vouched_says_planned(self) -> None:
        """Owner, Step 4 C: nothing plans a UR joint move, but on cuRobo its path is checked against the world its
        refresh registered, so it says PLANNED as a move does. It said UNPLANNED, "no world could be consulted"."""
        from src.robot.safety.planning.live_world import WorldRefresh, WorldVerdict

        class _Refreshing(_FakePlanner):
            last_world_refresh: object = None

            def refresh_world(self, *, near_point_mm: object = None) -> None:
                self.last_world_refresh = WorldRefresh(
                    verdict=WorldVerdict.FRESH, sent=1, registered=1, build_ms=0.0, register_ms=0.0, age_ms=5.0,
                    dropped_obstacles=0, cameras=("overhead",), captured_at_s=100.0,
                )

        result = _ur_curobo(_Refreshing(plan_result=[_JOINTS.tolist()]), live_world=object()).move_to_joints(_JOINTS)
        self.assertIs(result.status, MotionStatus.EXECUTED, result.message)
        self.assertEqual(result.camera_world, _VOUCHED)

    def test_only_the_stamp_changes_on_a_stamped_result(self) -> None:
        pose = _pose()
        result = _dummy().move(pose)
        self.assertEqual(replace(result, camera_world=CameraWorldStamp.unstated()),
                         MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose))

    def test_a_planner_double_that_returns_no_result_is_handed_back_untouched(self) -> None:
        planner = MagicMock()
        planner.plan.return_value = [[0.0] * 6, [0.1] * 6]
        arm = _ur_curobo(planner)
        self.assertIs(arm.move(_pose()), planner.execute.return_value)


class TheSimArmTests(unittest.TestCase):
    """The sim follows its own state: the mock, the missing sidecar and ``plan_joint_moves``."""

    def test_a_sidecar_that_cannot_start_refuses_the_move(self) -> None:
        """It used to drive the blind ik path instead and stamp the result UNPLANNED, saying why.

        That stamp was the honest half of a dishonest arrangement: the cell kept running, blind IK
        proposed configurations the self collision guard refused, and a validation run reported a low
        rate with nothing tying it back to a planner that never started. The move is refused now, so
        there is no motion left to stamp.
        """
        arm = _sim_unconnected()
        arm._articulation = object()  # type: ignore[assignment]
        arm._kin_solver = object()  # type: ignore[assignment]
        arm._rmpflow = object()  # type: ignore[assignment]
        arm._resolve_ik = lambda pose: _JOINTS  # type: ignore[method-assign]
        arm._get_curobo_client = MagicMock(  # type: ignore[method-assign]
            side_effect=CuroboUnavailableError("no cuRobo environment on this host"))
        drove = []
        arm._drive_to_target = lambda pose, joints: drove.append(pose)  # type: ignore[method-assign]
        result = arm.move(_pose())
        self.assertIs(result.status, MotionStatus.CONTROLLER_REJECTED)
        self.assertIn("cuRobo planner unavailable", result.message or "")
        self.assertEqual([], drove, "the arm drove the blind path anyway")

    def test_the_arm_no_longer_carries_a_degrade_latch(self) -> None:
        """A flag nobody can set is a state nobody can be in, which is the point of the change."""
        arm = _sim_unconnected()
        for name in ("curobo_degraded", "curobo_degraded_reason", "_curobo_unavailable"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(arm, name))

    def test_an_rmpflow_arm_says_unplanned_and_names_the_policy(self) -> None:
        """RMPflow is a reactive policy on the sim arm, not a planner that consults a camera world, so
        its motions say UNPLANNED like every other motion no planner planned, and a decline does not
        turn that into DECLINED."""
        arm = _sim_unconnected()
        arm._motion_planner = "rmpflow"
        expected = CameraWorldStamp.unplanned(
            "the RMPflow policy drives this arm: a reactive policy consults no camera world")
        self.assertEqual(arm.move(_pose()).camera_world, expected)
        declined = arm.move(_pose(), camera_world=CameraWorldDecline("bench rehearsal"))
        self.assertEqual(declined.camera_world, expected)

    def test_a_curobo_arm_with_no_live_world_says_missing(self) -> None:
        result = _sim_unconnected().move(_pose())
        self.assertIs(result.status, MotionStatus.CONNECTION_ERROR)
        self.assertEqual(result.camera_world, _SIM_NO_WORLD)

    def test_a_curobo_joint_move_is_judged_against_the_world_planned_or_not(self) -> None:
        """Owner, Step 4 C. A cuRobo joint move's path is checked against the live world whether or not
        ``plan_joint_moves`` plans it, so with no world wired it says MISSING either way; the interpolated one said
        UNPLANNED, which the check against the world has contradicted since Step 4e. A sim on another planner
        interpolates and consults no world, and says so: the control, green before and after."""
        interpolated = _sim_unconnected()
        interpolated._drive_joints = lambda joints, **_: None  # type: ignore[method-assign]
        planned = _sim_unconnected()
        planned._plan_joint_moves = True
        planned._drive_joints = lambda joints, **_: None  # type: ignore[method-assign]
        other = _sim_unconnected()
        other._motion_planner = "ik"
        other._drive_joints = lambda joints, **_: None  # type: ignore[method-assign]
        self.assertEqual(interpolated.move_to_joints(_JOINTS).camera_world, _SIM_NO_WORLD)
        self.assertEqual(planned.move_to_joints(_JOINTS).camera_world, _SIM_NO_WORLD)
        self.assertEqual(other.move_to_joints(_JOINTS).camera_world, _SIM_INTERPOLATED)


# ---------------------------------------------------------------------------------------------------
# 2. Precedence
# ---------------------------------------------------------------------------------------------------


class PrecedenceTests(unittest.TestCase):

    def test_the_pure_rule(self) -> None:
        keyword, block = CameraWorldDecline("keyword"), CameraWorldDecline("block")
        rows = (
            (dict(unplanned="no planner", missing="m", keyword=keyword, block=block),
             CameraWorldStamp.unplanned("no planner")),
            (dict(unplanned=None, missing="m", keyword=keyword, block=block),
             CameraWorldStamp.declined(keyword)),
            (dict(unplanned=None, missing="m", keyword=UNSET, block=block),
             CameraWorldStamp.declined(block)),
            (dict(unplanned=None, missing="m", keyword=UNSET, block=None),
             CameraWorldStamp.missing("m")),
            (dict(unplanned=None, missing=None, keyword=UNSET, block=block, planned=_VOUCHED),
             CameraWorldStamp.declined(block)),
            (dict(unplanned=None, missing=None, keyword=UNSET, block=None, planned=_VOUCHED),
             _VOUCHED),
            (dict(unplanned=None, missing=None, keyword=UNSET, block=None),
             CameraWorldStamp.unstated()),
        )
        for arguments, expected in rows:
            with self.subTest(**{key: str(value) for key, value in arguments.items()}):
                self.assertEqual(cw.resolve_camera_world(**arguments), expected)

    def test_a_keyword_beats_a_block(self) -> None:
        arm = _ur_curobo()
        with arm.without_camera_world("the block"):
            result = arm.move(_pose(), camera_world=CameraWorldDecline("the keyword"))
        self.assertEqual(result.camera_world, CameraWorldStamp.declined(CameraWorldDecline("the keyword")))

    def test_a_block_on_one_arm_does_not_stamp_another(self) -> None:
        declined, beside = _ur_curobo(), _ur_curobo()
        with declined.without_camera_world(_BENCH.reason):
            inside, other = declined.move(_pose()), beside.move(_pose())
        self.assertEqual(inside.camera_world, CameraWorldStamp.declined(_BENCH))
        self.assertEqual(other.camera_world, _UR_NO_WORLD)

    def test_the_innermost_block_wins_and_each_ends_with_its_with(self) -> None:
        arm = _ur_curobo()
        with arm.without_camera_world("outer"):
            with arm.without_camera_world("inner"):
                innermost = arm.move(_pose())
            outer = arm.move(_pose())
        after = arm.move(_pose())
        self.assertEqual(innermost.camera_world.reason, "inner")
        self.assertEqual(outer.camera_world.reason, "outer")
        self.assertEqual(after.camera_world, _UR_NO_WORLD)

    def test_a_block_ends_when_its_body_raises(self) -> None:
        arm = _ur_curobo()
        with self.assertRaises(RuntimeError), arm.without_camera_world(_BENCH.reason):
            raise RuntimeError("the body failed")
        self.assertEqual(arm.move(_pose()).camera_world, _UR_NO_WORLD)

    def test_a_blank_reason_is_refused_when_the_block_is_asked_for(self) -> None:
        for arm in (_ur_curobo(), _dummy()):
            with self.subTest(type(arm).__name__), self.assertRaises(ValueError):
                arm.without_camera_world("   ")

    def test_a_decline_is_a_decline_object_not_a_reason_or_none(self) -> None:
        """Matched on the message: an arm without the keyword raises TypeError too, for another reason."""
        for arm in (_ur_curobo(), _dummy()):
            for value in ("bench", None):
                with self.subTest(arm=type(arm).__name__, value=value), \
                        self.assertRaisesRegex(TypeError, "is a CameraWorldDecline, not"):
                    arm.move(_pose(), camera_world=value)  # type: ignore[arg-type]

    def test_unplanned_wins_over_a_decline(self) -> None:
        cases = (
            ("ur ik move", lambda: _ur_ik().move(_pose(), camera_world=_BENCH), _UR_IK),
            ("ur ik move_to_joints", lambda: _ur_ik().move_to_joints(_JOINTS, camera_world=_BENCH), _UR_IK),
            ("dummy move in a block", lambda: _in_block(_dummy(), "move"), _DUMMY),
            ("sim mock move", lambda: _sim_mock().move(_pose(), camera_world=_BENCH), _SIM_MOCK),
        )
        for label, command, expected in cases:
            with self.subTest(label):
                self.assertEqual(command().camera_world, expected)


def _in_block(arm: DummyRobotArm, verb: str) -> MotionResult:
    with arm.without_camera_world(_BENCH.reason):
        return arm.move(_pose()) if verb == "move" else arm.move_to_joints(_JOINTS)


# ---------------------------------------------------------------------------------------------------
# 3. A decline on an arm whose live camera world is wired
# ---------------------------------------------------------------------------------------------------


class ADeclineOnALiveWorldIsRefusedTests(unittest.TestCase):

    def test_the_ur_refuses_before_the_planner_is_asked(self) -> None:
        for label, enter in (("keyword", False), ("block", True)):
            with self.subTest(label):
                planner = MagicMock()
                arm = _ur_curobo(planner, live_world=object())
                pose = _pose()
                if enter:
                    with arm.without_camera_world(_BENCH.reason):
                        result = arm.move(pose)
                else:
                    result = arm.move(pose, camera_world=_BENCH)
                planner.plan.assert_not_called()
                planner.execute.assert_not_called()
                self.assertIs(result.status, MotionStatus.UNSUPPORTED)
                self.assertIs(result.command, MotionCommand.MOVE_TO)
                self.assertIs(result.target_pose, pose)
                self.assertEqual(result.message, cw.DECLINE_ON_A_LIVE_WORLD_MESSAGE)
                self.assertEqual(result.camera_world, CameraWorldStamp.declined(_BENCH))

    def test_the_ur_refuses_a_declined_joint_move_before_its_path_is_judged(self) -> None:
        """Owner, Step 4 C: the path of a cuRobo joint move is checked against the live world, which a decline cannot
        set aside for one motion. It ran and said UNPLANNED."""
        for label, enter in (("keyword", False), ("block", True)):
            with self.subTest(label):
                planner = MagicMock()
                arm = _ur_curobo(planner, live_world=object())
                if enter:
                    with arm.without_camera_world(_BENCH.reason):
                        result = arm.move_to_joints(_JOINTS)
                else:
                    result = arm.move_to_joints(_JOINTS, camera_world=_BENCH)
                planner.refresh_world.assert_not_called()
                arm._conn.moveJ.assert_not_called()  # type: ignore[attr-defined]
                self.assertIs(result.status, MotionStatus.UNSUPPORTED)
                self.assertIs(result.command, MotionCommand.MOVE_JOINTS)
                self.assertEqual(result.message, cw.DECLINE_ON_A_LIVE_WORLD_MESSAGE)
                self.assertEqual(result.camera_world, CameraWorldStamp.declined(_BENCH))

    def test_the_sim_refuses_before_the_planner_is_asked(self) -> None:
        arm = _sim_unconnected()
        arm.set_live_planner_world(object())  # type: ignore[arg-type]
        result = arm.move(_pose(), camera_world=_BENCH)
        arm._get_curobo_client.assert_not_called()  # type: ignore[attr-defined]
        self.assertIs(result.status, MotionStatus.UNSUPPORTED)
        self.assertEqual(result.camera_world, CameraWorldStamp.declined(_BENCH))

        joints = _sim_unconnected()
        joints._plan_joint_moves = True
        joints.set_live_planner_world(object())  # type: ignore[arg-type]
        joints._drive_joints = MagicMock()  # type: ignore[method-assign]
        refused = joints.move_to_joints(_JOINTS, camera_world=_BENCH)
        joints._drive_joints.assert_not_called()
        self.assertIs(refused.status, MotionStatus.UNSUPPORTED)
        self.assertIs(refused.command, MotionCommand.MOVE_JOINTS)
        self.assertEqual(refused.camera_world, CameraWorldStamp.declined(_BENCH))

    def test_the_message_is_one_ascii_sentence_pair(self) -> None:
        self.assertEqual(
            cw.DECLINE_ON_A_LIVE_WORLD_MESSAGE,
            "Refused before planning: this motion declines the camera world, and the planner cannot "
            "set aside the live camera world wired to this arm for one motion. Nothing moved.",
        )


# ---------------------------------------------------------------------------------------------------
# 4. The stamp comes from the arm, not from config
# ---------------------------------------------------------------------------------------------------


class TheStampComesFromTheArmTests(unittest.TestCase):

    def test_a_console_dummy_rehearsal_reads_unplanned_although_its_tree_says_curobo(self) -> None:
        tree = load_robot_config(profile="console_dummy")
        self.assertEqual((str(tree.vendor), tree.ur.motion_planner), ("dummy", "curobo"))

        from src.robot.execution.real_cell.__main__ import main

        seen: list[MotionResult] = []
        original = DummyRobotArm.move

        def recording(self: DummyRobotArm, pose: Pose, **kwargs: object) -> MotionResult:
            result = original(self, pose, **kwargs)  # type: ignore[arg-type]
            seen.append(result)
            return result

        with patch.object(DummyRobotArm, "move", recording), redirect_stdout(io.StringIO()):
            code = main(["--rehearse", "--runs", "1", "--profile", "console_dummy"])
        self.assertEqual(code, 0)
        self.assertTrue(seen, "the rehearsal commanded no typed motion")
        self.assertEqual({result.camera_world for result in seen}, {_DUMMY})


# ---------------------------------------------------------------------------------------------------
# The capability and the keyword
# ---------------------------------------------------------------------------------------------------


class TheCapabilityTests(unittest.TestCase):

    def test_every_driver_offers_the_block_and_the_protocol_does_not_demand_it(self) -> None:
        for arm in (_ur_curobo(), _kuka(), _sim_mock(), _dummy()):
            with self.subTest(type(arm).__name__):
                self.assertIsInstance(arm, cw.DeclinesCameraWorld)
        self.assertNotIsInstance(object(), cw.DeclinesCameraWorld)
        self.assertFalse(hasattr(RobotArm, "without_camera_world"))

    def test_both_typed_verbs_take_the_decline_as_a_keyword_that_defaults_to_unset(self) -> None:
        for owner in (RobotArm, URRobotArm, KukaRobotArm, IsaacRobotArm, DummyRobotArm):
            for verb in ("move", "move_to_joints"):
                with self.subTest(owner=owner.__name__, verb=verb):
                    parameter = inspect.signature(getattr(owner, verb)).parameters["camera_world"]
                    self.assertIs(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
                    self.assertIs(parameter.default, UNSET)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
