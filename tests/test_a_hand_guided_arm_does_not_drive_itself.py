"""While a freedrive session is open, every motion verb of a planned arm refuses, and says the arm is hand-guided.

The owner's decision of 2026-09-24: hand guiding is a vendor capability (``SupportsFreedrive``). A vendor offers it by listing
``PlannedFreedrive`` where its transport offers ``ControllerFreedrive`` (the UR does; ``tests/test_ur_io_force_status.py``
holds that on the UR arm itself), and the sim and dummy arms do not. A person has their hands on a freed arm, so from the
with-block's start to its end nothing the arm is asked to do may drive it: the refusal is the first thing every verb
meets, on the ik path and on the cuRobo path, before a planner, a guard or the controller is asked anything, and it says
nothing was sent. The arm here is a planned arm over the fake transport with the mixin a vendor class lists; the session
itself is pinned in ``tests/test_a_hand_guided_arm_is_held_on_every_way_out.py``.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import (
    ArmFault,
    ControllerPayload,
    FreedriveSample,
    JointPositions,
    MotionCommand,
    MotionSent,
    MotionStatus,
    RobotConnectionError,
    RobotMode,
    RobotMotionRejected,
    RobotStatus,
    SafetyMode,
    SupportsFreedrive,
)
from src.robot.drivers.planned.capabilities import PlannedFreedrive
from src.robot.core.freedrive import HAND_GUIDED_MESSAGE
from tests._arm_kit import kit_tree, planned_arm
from tests._fake_transport import fake_transport

_JOINTS = JointPositions([0.0, -1.5, 1.5, 0.0, 1.5, 0.0])
_STOPPED = RobotStatus(RobotMode.RUNNING, SafetyMode.PROTECTIVE_STOP, protective_stopped=True, emergency_stopped=False,
                       message="Safetystatus: PROTECTIVE_STOP", program_running=True)
_BRAKED = RobotStatus(RobotMode.IDLE, SafetyMode.NORMAL, protective_stopped=False, emergency_stopped=False,
                      program_running=True)
_NO_PROGRAM = RobotStatus(RobotMode.RUNNING, SafetyMode.NORMAL, protective_stopped=False, emergency_stopped=False,
                          program_running=False)
#: What the fake records when anything is sent to move the arm.
_MOTION_VERBS = ("run_trajectory", "move_pose")


def _pose() -> Pose:
    return Pose(position_mm=np.array([400.0, 0.0, 300.0]), quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
                frame=Frame.BASE)


class TheTypesTests(unittest.TestCase):
    def test_a_sample_says_its_fastest_joint(self) -> None:
        sample = FreedriveSample(joints_rad=(0.0,) * 6, joint_speeds_rad_s=(0.0, -0.3, 0.1, 0.0, 0.0, 0.2),
                                 tcp_xyz_mm=(400.0, 0.0, 300.0), tcp_rotvec_rad=(0.0, 3.14, 0.0), t_s=1.0)
        self.assertAlmostEqual(0.3, sample.peak_joint_speed_rad_s)
        still = FreedriveSample(joints_rad=(), joint_speeds_rad_s=(), tcp_xyz_mm=(0.0, 0.0, 0.0),
                                tcp_rotvec_rad=(0.0, 0.0, 0.0), t_s=0.0)
        self.assertEqual(0.0, still.peak_joint_speed_rad_s)

    def test_a_payload_is_a_mass_and_a_centre_of_gravity_in_millimetres(self) -> None:
        payload = ControllerPayload(mass_kg=1.9, cog_mm=(1.0, -2.0, 60.0))
        self.assertEqual(ControllerPayload(1.9, (1.0, -2.0, 60.0)), payload)


class _Tripwire:
    """Every name it hands out fails the test when called: a refused motion reaches none of them."""

    def __init__(self) -> None:
        self.reached: list[str] = []

    def __call__(self, name: str):  # noqa: ANN204
        def reached(*args: object, **kwargs: object) -> None:
            self.reached.append(name)
            raise AssertionError(f"{name} was reached by a motion of a hand-guided arm")
        return reached


def _arm(route: str = "ik", *, status: "RobotStatus | None" = None, connected: bool = True,
         **tree: Any) -> "tuple[Any, Any, _Tripwire]":
    """A hand-guidable planned arm on ``route`` over a fake transport in ``status``, its drive paths tripwired."""
    transport = fake_transport(connected=connected)
    if status is not None:
        transport.always("read_status", status)
    arm = planned_arm(kit_tree(route, **tree), transport=transport, mixins=(PlannedFreedrive,))
    trip = _Tripwire()
    for name in ("_drive_pose", "_drive_joints", "_home_refusal", "_judge_joint_move", "_judge_linear_move",
                 "_curobo_planner", "ik"):
        setattr(arm, name, trip(name))
    return arm, transport, trip


def _sent(transport: Any) -> list[str]:
    return [call.verb for call in transport.tape if call.verb in _MOTION_VERBS]


class EveryVerbRefusesTests(unittest.TestCase):
    def _assert_refused(self, arm: Any) -> None:
        for name, result, command in (
            ("move", arm.move(_pose()), MotionCommand.MOVE_TO),
            ("move linear", arm.move(_pose(), linear=True), MotionCommand.MOVE_TO),
            ("move_to_joints", arm.move_to_joints(_JOINTS), MotionCommand.MOVE_JOINTS),
            ("move_to_joints_on_the_line", arm.move_to_joints_on_the_line(_JOINTS), MotionCommand.MOVE_JOINTS),
            ("move_to_home", arm.move_to_home(), MotionCommand.MOVE_HOME),
        ):
            with self.subTest(verb=name):
                self.assertIs(result.status, MotionStatus.CONTROLLER_REJECTED, result.message)
                self.assertIs(result.command, command)
                self.assertIn(HAND_GUIDED_MESSAGE, result.message or "")
                self.assertIn("no command reached the controller", result.message or "")
                self.assertIs(MotionSent.NOTHING, result.sent)
        self.assertFalse(arm.move_home())
        with self.subTest(verb="move_linear"), self.assertRaises(RobotMotionRejected) as caught:
            arm.move_linear(_pose())
        self.assertIn(HAND_GUIDED_MESSAGE, str(caught.exception))
        self.assertIs(getattr(caught.exception.result, "status", None), MotionStatus.CONTROLLER_REJECTED)

    def test_every_verb_of_an_ik_arm_refuses_while_the_session_is_open_free_or_held(self) -> None:
        arm, transport, trip = _arm("ik")
        with arm.freedrive() as session:
            self._assert_refused(arm)
            session.free()
            self._assert_refused(arm)
            session.hold()
            self._assert_refused(arm)
        self.assertEqual(trip.reached, [])
        self.assertEqual([], _sent(transport))
        self.assertEqual([], transport.sent)
        # Nor was the steady gate asked: a UR in teach mode never answers steady, so a refusal below the gate would
        # wait out its timeout and come back as TIMEOUT (aurora's review of the port, 2026-09-25).
        self.assertEqual([], transport.calls("wait_until_steady"))

    def test_every_verb_of_a_curobo_arm_refuses_before_the_planner_or_the_camera_world(self) -> None:
        arm, transport, trip = _arm("curobo")
        with arm.freedrive() as session:
            session.free()
            self._assert_refused(arm)  # no decline in scope: the hand-guided refusal comes before the camera world's
            with arm.without_camera_world("bench check, no cameras mounted"):
                self._assert_refused(arm)
        self.assertEqual(trip.reached, [])
        self.assertIsNone(arm._curobo, "no planner was built")
        self.assertEqual([], _sent(transport))
        self.assertEqual([], transport.calls("wait_until_steady"))

    def test_leaving_the_session_gives_motion_back(self) -> None:
        """Inside the session the home move never reaches its own gate; after it, it does again."""
        from src.robot.core import MotionResult

        arm, transport, _ = _arm("ik")
        reached: list[object] = []

        def home_gate(joints: object) -> MotionResult:
            reached.append(joints)
            return MotionResult.failed(MotionStatus.JOINT_LIMIT_REJECTED, MotionCommand.MOVE_HOME,
                                       message="the test's home gate")

        arm._home_refusal = home_gate
        with arm.freedrive() as session:
            session.free()
            self.assertIs(MotionStatus.CONTROLLER_REJECTED, arm.move_to_home().status)
            self.assertEqual([], reached)
        result = arm.move_to_home()
        self.assertEqual(1, len(reached), "the verb did not reach its own gate after the session")
        self.assertEqual("the test's home gate", result.message)
        self.assertEqual(1, len(transport.calls("restart_control_program")))

    def test_an_arm_that_offers_no_hand_guiding_never_refuses_for_it(self) -> None:
        """THE CONTROL: a plain planned arm has no session to record, so the check is never what refuses."""
        arm = planned_arm(kit_tree("ik"), transport=fake_transport())
        self.assertNotIsInstance(arm, SupportsFreedrive)
        self.assertIsNone(arm._hand_guided_refusal(MotionCommand.MOVE_TO))


class OpeningASessionTests(unittest.TestCase):
    def test_the_mixin_offers_hand_guiding_and_the_sim_and_dummy_arms_do_not(self) -> None:
        from src.robot.drivers.dummy.arm import DummyRobotArm
        from src.robot.drivers.sim.arm import IsaacRobotArm
        from src.robot.drivers.sim.config import SimRobotConfig

        arm, _, _ = _arm()
        self.assertIsInstance(arm, SupportsFreedrive)
        self.assertNotIsInstance(IsaacRobotArm(SimRobotConfig(enabled=True, mock_mode=True)), SupportsFreedrive)
        self.assertNotIsInstance(DummyRobotArm(), SupportsFreedrive)

    def test_nothing_is_freed_when_the_session_is_handed_out(self) -> None:
        arm, transport, _ = _arm()
        session = arm.freedrive()
        self.assertFalse(session.is_free)
        self.assertEqual([], transport.calls("begin_hand_guiding"))
        self.assertEqual([], transport.calls("arm_watchdog"))
        # Open only inside its with-block, which is what holds the arm again at its end.
        self.assertIsNone(arm._hand_guided_refusal(MotionCommand.MOVE_TO))

    def test_an_arm_that_is_not_connected_is_refused(self) -> None:
        arm, _, _ = _arm(connected=False)
        with self.assertRaisesRegex(RobotConnectionError, "open connection"):
            arm.freedrive()

    def test_a_stopped_controller_is_refused_as_a_fault(self) -> None:
        arm, _, _ = _arm(status=_STOPPED)
        with self.assertRaisesRegex(ArmFault, "protective_stop"):
            arm.freedrive()

    def test_an_arm_with_its_brakes_engaged_is_refused(self) -> None:
        arm, _, _ = _arm(status=_BRAKED)
        with self.assertRaisesRegex(RobotMotionRejected, "robot mode idle"):
            arm.freedrive()

    def test_an_arm_whose_control_program_stopped_is_refused(self) -> None:
        arm, _, _ = _arm(status=_NO_PROGRAM)
        with self.assertRaisesRegex(RobotMotionRejected, "control program running"):
            arm.freedrive()

    def test_a_second_session_is_refused_while_one_is_open(self) -> None:
        arm, _, _ = _arm()
        waiting = arm.freedrive()
        with arm.freedrive():
            with self.assertRaisesRegex(RobotMotionRejected, HAND_GUIDED_MESSAGE):
                arm.freedrive()
            with self.assertRaisesRegex(RobotMotionRejected, HAND_GUIDED_MESSAGE):
                waiting.__enter__()
        self.assertIsNone(arm._freedrive)
        with arm.freedrive():  # and one after the other is fine
            pass


class ControllerPayloadTests(unittest.TestCase):
    def test_the_controller_payload_is_the_transports(self) -> None:
        arm, transport, _ = _arm()
        transport.always("read_payload", ControllerPayload(mass_kg=1.9, cog_mm=(0.0, 0.0, 60.0)))
        self.assertEqual(arm.controller_payload(), ControllerPayload(mass_kg=1.9, cog_mm=(0.0, 0.0, 60.0)))

    def test_a_controller_that_does_not_report_it_reads_none(self) -> None:
        arm, _, _ = _arm()
        self.assertIsNone(arm.controller_payload())


class TheSampleIsTheArmsTcpTests(unittest.TestCase):
    def test_a_sample_reads_the_tcp_with_the_declared_tool_frame_composed(self) -> None:
        """On a ``willy`` cell the controller runs a bare flange; the sample is the grasp centre, as ``get_tcp_pose``."""
        arm, transport, _ = _arm("ik", gripper={"model": "robotiq_2f85", "tool_frame": {
            "source": "willy", "offset_mm": [0.0, 0.0, 150.0], "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]}})
        # The flange 0.6 m up, pointing straight down (180 degrees about base X).
        transport.always("read_tcp", Pose(position_mm=np.array([400.0, -100.0, 600.0]),
                                          quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]), frame=Frame.BASE))
        with arm.freedrive() as session:
            sample = session.sample()
        np.testing.assert_allclose(sample.tcp_xyz_mm, (400.0, -100.0, 450.0), atol=1e-9)
        np.testing.assert_allclose(sample.tcp_xyz_mm, arm.get_tcp_pose().position_mm, atol=1e-9)
        np.testing.assert_allclose(sample.joints_rad, transport.joints, atol=0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
