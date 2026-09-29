"""A UR arm names the joint configuration a pose goes to, before anything moves.

The generated view of a wrist pick orbits the camera about the part and asks, angle by angle, whether the arm can stand
there: whether the pose has a configuration at all, whether one lies inside the half-turn cable window about home, and
whether the endpoint gate admits it. Asking by moving costs a judged motion each, and a cuRobo plan that fails about 7
to 9 s, so the arm is asked first: ``nearest_configuration(pose)``, the capability
:class:`~src.robot.core.arm_capabilities.ChoosesConfigurations`.

It is the screening half of the route a Cartesian cuRobo move takes to its nearest goal, shared and not copied: the
joints from the controller, the flange goal of the TCP pose solved in closed form, every solution turned onto the full
turns nearest the arm inside the planner's window narrowed to the cable window about home, the arm's own branch first,
and the first configuration the endpoint gate admits (the workspace box on the TCP, the joint limits, self-collision,
payload). No world is refreshed and nothing moves. A pose with no answer raises ``RobotKinematicsError`` naming the
inverse kinematics, the window or the gate.

What this file holds, with the planner double of the route tests: which configuration is named and in what order the
gate is asked, on the branch the arm stands on; that a TCP pose is solved as its flange goal and boxed as the TCP; each
refusal and what it names, and a controller reading that is no configuration refused as the controller's; that
neither the planner, the world nor the controller's motion is asked anything, and that asking leaves the continuity
memo and the warning log as they were, since nothing was commanded; that a move to the pose ends on the configuration
named; and that only the UR arm says it can. Self-collision is not enforced on these arms: without an exact-mesh engine
(CI, macOS, a box whose policy blocks the Coal DLL) the capsule fallback refuses this UR10 at its own home, link 3 14 mm
into the tool, and the collision half of the gate has its own tests.
"""

from __future__ import annotations

import math
import unittest
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from src.config.schema.robot import RobotConfig
from src.geometry import Frame, FrameMismatchError, Pose
from src.robot.core import (
    JointPositions,
    MotionCommand,
    MotionResult,
    MotionStatus,
    RobotArm,
    RobotConnectionError,
    RobotKinematicsError,
)
from src.robot.core.arm_capabilities import ChoosesConfigurations
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.drivers.kuka.arm import KukaRobotArm
from src.robot.drivers.sim.arm import IsaacRobotArm
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.drivers.ur.curobo_motion import PLANNER_JOINT_ENVELOPE_RAD
from src.robot.safety._ur_ik import NearestGoal, nearest_goals, ur_branch, ur_flange_ik
from src.robot.safety._ur_kinematics import ur_link_transforms_mm
from tests._plan_end import OPEN_WORKSPACE, pose_where_it_ends
from tests._route_planner import RoutePlanner

_DECLINED = "unit double: this file reads which configuration a pose goes to, and no camera world is wired"

#: The owner's arm: a UR10 whose cable window is half a turn either side of this home (the cable-window tests' home).
_HOME = (0.2, -1.4, 1.3, -1.5, -1.57, 0.1)
#: The arm stands at home with wrist 3 wound on to 150 deg.
_HERE = [*_HOME[:5], math.radians(150.0)]
#: A pose whose configuration on the arm's branch holds wrist 3 at -160 deg, 310 deg back from where it stands. The
#: same angle one turn on, 200 deg, is only 50 deg away and past the seam half a turn behind home (180.7 deg less the
#: 5 deg margin); the quickest configuration by time is on another shoulder.
_THERE = [0.5, -1.3, 1.2, -1.4, -1.5, math.radians(-160.0)]
#: The arm standing elbow down, on another branch than home's (elbow up), inside the cable window.
_ELBOW_DOWN = [0.5, -0.15, -1.2, -0.15, -1.5, math.radians(-160.0)]
#: A few degrees on from it on every joint, still elbow down: a pose with a configuration on either elbow.
_ELBOW_DOWN_THERE = [0.55, -0.1, -1.25, -0.1, -1.45, math.radians(-155.0)]
_SAFETY: dict[str, Any] = {
    "payload": {"enforce": False},
    "self_collision": {"enforce": False},
    "joint_limits": {"within_half_turn_of_home": True},
}
#: The owner's hand: a Hand-E whose TCP this driver owns (``source: willy``), 150 mm out along the flange axis and
#: turned 45 deg about it, so the flange goal of a pose is neither the pose nor its translate.
_HANDE: dict[str, Any] = {
    "model": "robotiq_hande", "coupling_plates": [],
    "tool_frame": {"source": "willy", "offset_mm": [0.0, 0.0, 150.0],
                   "rotation_quat_xyzw": [0.0, 0.0, math.sin(math.radians(22.5)), math.cos(math.radians(22.5))]},
}


def _arm(planner: RoutePlanner, *, safety: "dict[str, Any] | None" = None,
         workspace: "dict[str, float] | None" = None, gripper: "dict[str, Any] | None" = None,
         **more: Any) -> URRobotArm:
    arm = URRobotArm(RobotConfig.model_validate({
        "vendor": "ur", "ur": {"model": "ur10", "motion_planner": "curobo"},
        "safety": safety if safety is not None else _SAFETY,
        "gripper": gripper if gripper is not None else {"model": "robotiq_2f85"},
        "workspace_limits": workspace if workspace is not None else OPEN_WORKSPACE,
        "home_joint_positions": _HOME, **more,
    }))
    arm._conn = MagicMock()
    arm._conn.is_connected = True
    arm._conn.get_joint_positions.return_value = list(planner.here)
    arm._curobo_ur = planner  # type: ignore[assignment]
    return arm


def _gate_spy(arm: URRobotArm, *, refuse: "Any" = None) -> "list[tuple[float, ...]]":
    """Record every configuration the endpoint gate is asked about, screened (``nearest_configuration``) or gated
    before a move drives it; ``refuse(joints)`` true refuses it in both, else the real gate answers."""
    asked: list[tuple[float, ...]] = []

    def spy(real: Any) -> Any:
        def gate(pose: Any, joints: JointPositions) -> "MotionResult | None":
            asked.append(tuple(float(v) for v in joints.tolist()))
            if refuse is not None and refuse(joints.tolist()):
                return MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO,
                                           message="the spy")
            return real(pose, joints)
        return gate

    arm._screen_planned_config = spy(arm._screen_planned_config)  # type: ignore[method-assign]
    arm._gate_planned_config = spy(arm._gate_planned_config)  # type: ignore[method-assign]
    return asked


def _goals(arm: URRobotArm, end: "list[float]", here: "list[float]", *,
           window: "tuple[Any, Any] | None" = None) -> "list[NearestGoal]":
    """The configurations of ``end``'s pose in ``window`` (the arm's goal window by default), by time alone."""
    frames = ur_link_transforms_mm("ur10", np.asarray(end, dtype=np.float64))
    assert frames is not None
    solutions = ur_flange_ik("ur10", frames[-1], q6_if_singular=here[-1])
    assert solutions
    lower, upper = window if window is not None else arm._goal_joint_window()
    return list(nearest_goals(solutions, current=here, lower=lower, upper=upper, velocity=1.0))


def _on_the_arms_branch(joints: "Any", here: "list[float]") -> bool:
    held, other = ur_branch("ur10", here), ur_branch("ur10", list(joints))
    assert held is not None and other is not None
    return held.agrees_with(other)


def _near(a: "Any", b: "Any", tol: float = 1e-6) -> bool:
    return max(abs(float(x) - float(y)) for x, y in zip(a, b)) < tol


def _flange_mm(joints: "list[float]") -> np.ndarray:
    """Where the flange stands at ``joints``, a 4x4 in BASE millimetres, on the DH chain alone."""
    frames = ur_link_transforms_mm("ur10", np.asarray(joints, dtype=np.float64))
    assert frames is not None
    return np.asarray(frames[-1], dtype=np.float64)


class TheConfigurationNamedTests(unittest.TestCase):
    def test_nearest_on_the_arms_branch_inside_the_cable_window(self) -> None:
        planner = RoutePlanner(here=_HERE)
        arm = _arm(planner)
        asked = _gate_spy(arm)
        named = arm.nearest_configuration(pose_where_it_ends(arm, _THERE))

        self.assertIsInstance(named, JointPositions)
        np.testing.assert_allclose(named.tolist(), _THERE, atol=1e-6)
        lower, upper = arm._goal_joint_window()
        for axis, value in enumerate(named.tolist()):
            self.assertTrue(lower[axis] <= value <= upper[axis], f"axis {axis} named at {math.degrees(value):.1f}")
        self.assertTrue(_on_the_arms_branch(named.tolist(), _HERE))
        self.assertEqual(1, len(asked), "the first configuration on the arm's branch passed, so no other was asked")
        # What makes it the arm's choice: the quickest configuration is on another branch, and without the cable
        # window the nearest turn on this one is wrist 3 at 200 deg, past the seam behind home.
        quickest = _goals(arm, _THERE, _HERE)[0]
        self.assertFalse(_on_the_arms_branch(quickest.joints, _HERE))
        unwound = _goals(arm, _THERE, _HERE, window=PLANNER_JOINT_ENVELOPE_RAD)[0]
        self.assertTrue(_on_the_arms_branch(unwound.joints, _HERE))
        self.assertAlmostEqual(200.0, math.degrees(unwound.joints[5]), places=6)

    def test_the_branch_held_first_is_the_one_the_arm_stands_on(self) -> None:
        """Standing elbow down, the arm is named the configuration elbow down; the same pose from home, elbow up.

        The branch comes from the joints the controller reports, not from home: an arm that stood on home's branch
        whatever it read would name the elbow-up configuration from both."""
        home_branch, standing = ur_branch("ur10", list(_HOME)), ur_branch("ur10", _ELBOW_DOWN)
        assert home_branch is not None and standing is not None
        self.assertFalse(home_branch.agrees_with(standing), "the test needs the arm on another branch than home's")
        down = _arm(RoutePlanner(here=_ELBOW_DOWN))
        pose = pose_where_it_ends(down, _ELBOW_DOWN_THERE)
        named = down.nearest_configuration(pose)
        np.testing.assert_allclose(named.tolist(), _ELBOW_DOWN_THERE, atol=1e-6)
        self.assertTrue(_on_the_arms_branch(named.tolist(), _ELBOW_DOWN))
        # The control: from home the same pose goes elbow up, so the answer above was the reading's.
        from_home = _arm(RoutePlanner(here=list(_HOME))).nearest_configuration(pose)
        self.assertTrue(_on_the_arms_branch(from_home.tolist(), list(_HOME)))
        self.assertFalse(_near(from_home.tolist(), _ELBOW_DOWN_THERE, 1e-3))

    def test_nothing_moves_and_no_world_is_refreshed(self) -> None:
        planner = RoutePlanner(here=_HERE)
        arm = _arm(planner)
        arm._live_world = MagicMock()  # a refresh would reach the planner double, and it records every call
        arm.nearest_configuration(pose_where_it_ends(arm, _THERE))
        self.assertEqual([], planner.calls, "the planner was asked something")
        self.assertEqual([], arm._live_world.mock_calls)
        conn: Any = arm._conn
        self.assertEqual({"get_joint_positions"}, {call[0] for call in conn.method_calls},
                         "the controller was asked something other than where the arm stands")

    def test_a_configuration_the_gate_refuses_gives_way_to_the_next_in_the_arms_order(self) -> None:
        """The only configuration on the arm's branch is refused; the quickest of the others is named next."""
        planner = RoutePlanner(here=_HERE)
        arm = _arm(planner)
        on_branch = [g for g in _goals(arm, _THERE, _HERE) if _on_the_arms_branch(g.joints, _HERE)]
        self.assertEqual(1, len(on_branch), "the test needs exactly one configuration on the arm's branch")
        asked = _gate_spy(arm, refuse=lambda joints: _near(joints, on_branch[0].joints, 1e-9))
        named = arm.nearest_configuration(pose_where_it_ends(arm, _THERE))
        quickest = _goals(arm, _THERE, _HERE)[0]
        np.testing.assert_allclose(named.tolist(), quickest.joints, atol=1e-9)
        self.assertEqual(2, len(asked))
        np.testing.assert_allclose(asked[0], on_branch[0].joints, atol=1e-9)
        np.testing.assert_allclose(asked[1], quickest.joints, atol=1e-9)


class ATcpPoseTests(unittest.TestCase):
    """The pose asked about is the TCP's: its flange goal is what is solved, and the TCP is what the box judges."""

    def test_the_configuration_named_puts_the_tcp_on_the_pose(self) -> None:
        arm = _arm(RoutePlanner(here=_HERE), gripper=_HANDE)
        self.assertIsNotNone(arm._declared_tool_matrix(), "the test needs a declared tool frame")
        pose = pose_where_it_ends(arm, _THERE)
        self.assertGreater(float(np.linalg.norm(np.asarray(pose.position_mm) - _flange_mm(_THERE)[:3, 3])), 100.0)
        named = arm.nearest_configuration(pose)
        np.testing.assert_allclose(named.tolist(), _THERE, atol=1e-6)
        np.testing.assert_allclose(pose_where_it_ends(arm, named.tolist()).to_matrix(), pose.to_matrix(), atol=1e-6)

    def test_the_box_judges_the_tcp_and_not_the_flange(self) -> None:
        """A box between the TCP and the flange: the pose is admitted where the TCP is inside it, whatever the flange."""
        probe = _arm(RoutePlanner(here=_HERE), gripper=_HANDE)
        tcp_z = float(pose_where_it_ends(probe, _THERE).position_mm[2])
        flange_z = float(_flange_mm(_THERE)[2, 3])
        self.assertGreater(flange_z - tcp_z, 100.0, "the test needs the flange well above the TCP")
        between = (tcp_z + flange_z) / 2.0  # 75 mm from either, past the box's 20 mm margin
        tcp_inside = _arm(RoutePlanner(here=_HERE), gripper=_HANDE, workspace={**OPEN_WORKSPACE, "z_max": between})
        np.testing.assert_allclose(
            tcp_inside.nearest_configuration(pose_where_it_ends(tcp_inside, _THERE)).tolist(), _THERE, atol=1e-6)
        flange_inside = _arm(RoutePlanner(here=_HERE), gripper=_HANDE, workspace={**OPEN_WORKSPACE, "z_min": between},
                             safe_pose={"x": 0.0, "y": -300.0, "z": 1000.0})  # the config keeps it in the box
        with self.assertRaises(RobotKinematicsError) as caught:
            flange_inside.nearest_configuration(pose_where_it_ends(flange_inside, _THERE))
        self.assertIn(MotionStatus.WORKSPACE_REJECTED.value, str(caught.exception))


class WhatIsRefusedTests(unittest.TestCase):
    def test_out_of_reach_is_refused_as_ik(self) -> None:
        planner = RoutePlanner(here=_HERE)
        arm = _arm(planner)
        asked = _gate_spy(arm)
        near = pose_where_it_ends(arm, _THERE)
        far = type(near)(position_mm=np.asarray(near.position_mm) * 5.0, quaternion_xyzw=near.quaternion_xyzw,
                         frame=near.frame, label="five times out")
        with self.assertRaises(RobotKinematicsError) as caught:
            arm.nearest_configuration(far)
        self.assertIn("refused by the inverse kinematics", str(caught.exception))
        self.assertIn("out of the arm's reach", str(caught.exception))
        self.assertIn("five times out", str(caught.exception))
        self.assertEqual([], asked)
        self.assertEqual([], planner.calls)

    def test_every_configuration_outside_the_window_is_refused(self) -> None:
        """Wrist 3 limited to +-30 deg: the pose has configurations, and every one turns wrist 3 outside."""
        safety = {**_SAFETY, "joint_limits": {"within_half_turn_of_home": True,
                                              "min_deg": [-360.0] * 5 + [-30.0], "max_deg": [360.0] * 5 + [30.0]}}
        planner = RoutePlanner(here=list(_HOME))
        arm = _arm(planner, safety=safety)
        asked = _gate_spy(arm)
        end = [*_THERE[:5], 2.0]
        self.assertTrue(_goals(arm, end, list(_HOME), window=PLANNER_JOINT_ENVELOPE_RAD),
                        "the pose has to be reachable without the window, or this reads the inverse kinematics")
        with self.assertRaises(RobotKinematicsError) as caught:
            arm.nearest_configuration(pose_where_it_ends(arm, end))
        self.assertIn("refused by the joint window", str(caught.exception))
        self.assertIn("half a turn either side of home", str(caught.exception))
        self.assertEqual([], asked, "a configuration outside the window reached the gate")
        self.assertEqual([], planner.calls)

    def test_outside_the_workspace_box_is_refused(self) -> None:
        """The real gate: the box on the TCP refuses the pose, the same pose inside it is named.

        The box reads the TCP alone, which every configuration of the pose puts in the same place, so the first
        configuration it refuses is every configuration's refusal and no other is asked."""
        planner = RoutePlanner(here=_HERE)
        pose_x = float(pose_where_it_ends(_arm(planner), _THERE).position_mm[0])
        self.assertLess(pose_x, -100.0)
        boxed = _arm(planner, workspace={**OPEN_WORKSPACE, "x_min": 0.0})
        asked = _gate_spy(boxed)
        self.assertGreater(len(_goals(boxed, _THERE, _HERE)), 1, "the test needs more than one configuration")
        with self.assertRaises(RobotKinematicsError) as caught:
            boxed.nearest_configuration(pose_where_it_ends(boxed, _THERE))
        said = str(caught.exception)
        self.assertIn("refused by the endpoint gate", said)
        self.assertIn(MotionStatus.WORKSPACE_REJECTED.value, said)
        self.assertIn("outside workspace box", said)
        self.assertEqual(1, len(asked), "a box that refused the TCP was asked again about the same TCP")
        self.assertEqual([], planner.calls)
        # The control: the same pose and the same guards in an open box.
        np.testing.assert_allclose(_arm(planner).nearest_configuration(
            pose_where_it_ends(boxed, _THERE)).tolist(), _THERE, atol=1e-6)

    def test_an_arm_that_cannot_read_where_it_stands_says_so_rather_than_naming_a_configuration(self) -> None:
        planner = RoutePlanner(here=_HERE)
        arm = _arm(planner)
        pose = pose_where_it_ends(arm, _THERE)
        conn: Any = arm._conn
        conn.get_joint_positions.side_effect = RuntimeError("RTDE receive timed out")
        with self.assertRaises(RobotConnectionError) as caught:
            arm.nearest_configuration(pose)
        self.assertIn("RTDE receive timed out", str(caught.exception))
        # Closed: refused before the controller is asked, whatever it would have answered.
        conn.get_joint_positions.side_effect = None
        conn.get_joint_positions.reset_mock()
        conn.is_connected = False
        with self.assertRaises(RobotConnectionError) as caught:
            arm.nearest_configuration(pose)
        self.assertIn("open connection", str(caught.exception))
        conn.get_joint_positions.assert_not_called()

    def test_a_reading_that_is_no_configuration_is_the_controllers_fault_and_not_the_poses(self) -> None:
        """A joint reading with a value that is not finite, or with another number of joints, says nothing about
        the pose: it is refused as the controller's, so a caller never skips a pose it could have stood at."""
        readings = {
            "not a number": [math.nan] * 6,
            "an infinite wrist": [*_HERE[:5], math.inf],
            "five joints": _HERE[:5],
            "none": [],
        }
        for case, reading in readings.items():
            with self.subTest(case=case):
                planner = RoutePlanner(here=_HERE)
                arm = _arm(planner)
                pose = pose_where_it_ends(arm, _THERE)
                asked = _gate_spy(arm)
                conn: Any = arm._conn
                conn.get_joint_positions.return_value = reading
                with self.assertRaises(RobotConnectionError) as caught:
                    arm.nearest_configuration(pose)
                self.assertIn("joint reading", str(caught.exception))
                self.assertEqual([], asked)

    def test_a_pose_outside_base_is_refused_as_ik_refuses_it(self) -> None:
        """Refused on its frame alone, before the controller, the inverse kinematics or the gate is asked."""
        planner = RoutePlanner(here=_HERE)
        arm = _arm(planner)
        asked = _gate_spy(arm)
        base = pose_where_it_ends(arm, _THERE)
        tool = Pose(position_mm=base.position_mm, quaternion_xyzw=base.quaternion_xyzw, frame=Frame.TOOL)
        with self.assertRaises(FrameMismatchError) as caught:
            arm.nearest_configuration(tool)
        self.assertIn("nearest_configuration", str(caught.exception))
        conn: Any = arm._conn
        conn.get_joint_positions.assert_not_called()
        self.assertEqual([], asked)
        self.assertEqual([], planner.calls)


class AskingCommandsNothingTests(unittest.TestCase):
    """A configuration named is no target sent: the safety state and the warning log stay as they were."""

    def test_the_continuity_memo_is_left_as_it_was(self) -> None:
        """The next move's continuity is judged from the last target sent, not from a configuration only asked about.

        The memo holds home; the arm is asked about a pose a quarter turn round the base; a target 10 mm from home is
        still one continuous step from the last target, which it would not be from the one asked about."""
        # The IK-quality guard probes the Jacobian through the controller, which this double has not got.
        arm = _arm(RoutePlanner(here=list(_HOME)), safety={**_SAFETY, "ik_quality": {"enforce": False}})
        preflight = arm._preflight
        assert preflight is not None
        self.assertIn("motion_continuity", preflight.guard_names)
        at_home = JointPositions(list(_HOME))
        home = pose_where_it_ends(arm, _HOME)
        sent = preflight.evaluate(preflight.context_for_pose(
            home, target_joints=at_home, current_joints=at_home, arm=arm))
        self.assertTrue(sent.accepted, sent.message)
        memo = (preflight._last_target_pose, preflight._last_target_joints)

        round_the_base = [_HOME[0] + math.pi / 2.0, *_HOME[1:]]
        named = arm.nearest_configuration(pose_where_it_ends(arm, round_the_base))
        np.testing.assert_allclose(named.tolist(), round_the_base, atol=1e-6)
        self.assertIs(memo[0], preflight._last_target_pose)
        self.assertIs(memo[1], preflight._last_target_joints)

        nudged = Pose(position_mm=np.asarray(home.position_mm) + np.array([10.0, 0.0, 0.0]),
                      quaternion_xyzw=home.quaternion_xyzw, frame=Frame.BASE, label="10 mm on from home")
        after = preflight.evaluate(preflight.context_for_pose(
            nudged, target_joints=at_home, current_joints=at_home, arm=arm))
        self.assertTrue(after.accepted, after.message)

    def test_a_configuration_refused_is_no_refused_motion_in_the_log(self) -> None:
        """Nothing was commanded, so the preflight writes no 'rejected motion' warning, and the box says once that
        the pose is outside it."""
        planner = RoutePlanner(here=_HERE)
        boxed = _arm(planner, workspace={**OPEN_WORKSPACE, "x_min": 0.0})
        pose = pose_where_it_ends(boxed, _THERE)
        with self.assertNoLogs("SafetyPreflight", level="WARNING"):
            with self.assertLogs("WorkspaceGuard", level="WARNING") as box:
                with self.assertRaises(RobotKinematicsError):
                    boxed.nearest_configuration(pose)
        self.assertEqual(1, len(box.records), [r.getMessage() for r in box.records])


class TheRouteTests(unittest.TestCase):
    def test_a_move_to_the_pose_ends_on_the_configuration_it_named(self) -> None:
        """One screening half for both: whichever configuration the gate admits, the move runs to the one named."""
        cases = {
            "the arm's branch": lambda joints: False,
            "the next, the arm's refused": lambda joints: _near(joints, _THERE, 1e-6),
        }
        for case, refuse in cases.items():
            with self.subTest(case=case):
                planner = RoutePlanner(here=_HERE)
                arm = _arm(planner)
                _gate_spy(arm, refuse=refuse)
                arm._preflight.gate_planned_path = (  # type: ignore[method-assign]
                    lambda waypoints, *, arm=None, command=None: None)  # the path gate has its own tests
                pose = pose_where_it_ends(arm, _THERE)
                named = arm.nearest_configuration(pose)
                with arm.without_camera_world(_DECLINED):
                    result = arm.move(pose)
                self.assertTrue(result.ok, result.message)
                (execute,) = planner.named("execute")
                np.testing.assert_allclose(execute[1][-1], named.tolist(), atol=1e-12)


class WhichArmsNameAConfigurationTests(unittest.TestCase):
    def test_only_the_ur_arm_names_a_configuration(self) -> None:
        """The dummy's IK is a stand-in that must never place a camera; the KUKA and Isaac arms do not say yet."""
        self.assertIsInstance(_arm(RoutePlanner(here=_HERE)), ChoosesConfigurations)
        self.assertNotIsInstance(DummyRobotArm(), ChoosesConfigurations)
        for driver in (KukaRobotArm, IsaacRobotArm, DummyRobotArm):
            with self.subTest(driver=driver.__name__):
                self.assertFalse(hasattr(driver, "nearest_configuration"))
        self.assertFalse(hasattr(RobotArm, "nearest_configuration"),
                         "a member on RobotArm would reach every driver that subclasses it as a stub")


if __name__ == "__main__":
    unittest.main()
