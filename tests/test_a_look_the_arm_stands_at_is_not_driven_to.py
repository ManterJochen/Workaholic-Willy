"""A look the arm already stands at is not driven to: nothing is sent, the report says so, and the camera looks there.

The owner's cell, 2026-10-08: its first look is its home, so every part began with a move of no length, 1.0 to 1.4 s
of world, judge and controller for nothing; the bin's survey and a Restart's first look paid it too.
``looks.move_to_look`` now reads the arm first: a controller's arm whose motions are planned and judged, connected, not
halted, at rest, every joint within 0.05 degrees of the look and its tool where its own kinematics put the look's tool
answers EXECUTED, "already there; nothing sent", said in the log, after the verb's own gates. A look 0.1 degrees away is
driven to; so is one the arm cannot say it stands at, and a halted arm meets its verb, which refuses as before. A desk
arm, whose joints need not say where its tool stands, moves as it always did.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import JointPositions, MotionCommand, MotionResult, MotionStatus
from src.robot.core.arm_capabilities import LineMotion, LineReading
from src.robot.core.camera_world import CameraWorldStamp
from src.robot.core.capabilities import RobotCapabilities
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.execution.looks import ALREADY_THERE, HOME, move_to_look
from src.robot.execution.motion import MotionOutcome, MotionVerb
from tests._task_fakes import HOME_JOINTS, ScriptedLocator, TaskArm, bin_object, motions, run


def _key(joints: JointPositions) -> tuple[float, ...]:
    return tuple(round(v, 1) for v in joints.degrees())


def _turned(joints: JointPositions, joint: int, by_deg: float) -> JointPositions:
    degrees = list(joints.degrees())
    degrees[joint] += by_deg
    return JointPositions.deg(*degrees)


#: A look the arm does not stand at, and where it puts the tool.
LOOK = JointPositions.deg(10.0, -90.0, -110.0, -60.0, 90.0, 0.0)
AT_LOOK = Pose.tool_down(150.0, -650.0, 450.0, label="look")


def _arm(log: list[Any], **keywords: Any) -> TaskArm:
    """The task's UR-like arm, standing at its home, whose kinematics place the home and :data:`LOOK`."""
    arm = TaskArm(log, fk_table={_key(LOOK): AT_LOOK}, **keywords)
    return arm


class ALookTheArmStandsAtTests(unittest.TestCase):
    def test_nothing_is_sent_and_the_report_says_the_arm_was_already_there(self) -> None:
        log: list[Any] = []
        arm = _arm(log)
        with self.assertLogs("src.robot.execution.looks", "INFO") as said:
            moved = move_to_look(arm, HOME_JOINTS)

        self.assertIs(MotionOutcome.EXECUTED, moved.outcome)
        self.assertTrue(moved.ok)
        self.assertIs(MotionVerb.MOVE_JOINTS, moved.verb)
        self.assertEqual(ALREADY_THERE, moved.message)
        self.assertIs(MotionCommand.MOVE_JOINTS, moved.result.command)
        self.assertEqual(HOME_JOINTS, moved.result.target_joints)
        self.assertEqual([], motions(log), "a move of no length was sent")
        self.assertTrue(any("already stands there" in line for line in said.output), said.output)

    def test_the_home_look_is_read_off_the_arms_own_home(self) -> None:
        log: list[Any] = []
        moved = move_to_look(_arm(log), HOME)

        self.assertIs(MotionVerb.HOME, moved.verb)
        self.assertEqual(ALREADY_THERE, moved.message)
        self.assertIs(MotionCommand.MOVE_HOME, moved.result.command)
        self.assertEqual([], motions(log))

    def test_within_five_hundredths_of_a_degree_it_stands_there(self) -> None:
        log: list[Any] = []
        moved = move_to_look(_arm(log), _turned(HOME_JOINTS, 3, 0.04))

        self.assertEqual(ALREADY_THERE, moved.message)
        self.assertEqual([], motions(log))

    def test_a_look_a_tenth_of_a_degree_away_is_driven_to(self) -> None:
        for joint in range(6):
            with self.subTest(joint=joint):
                log: list[Any] = []
                look = _turned(HOME_JOINTS, joint, 0.1)
                moved = move_to_look(_arm(log), look)

                self.assertIs(MotionOutcome.EXECUTED, moved.outcome)
                self.assertNotEqual(ALREADY_THERE, moved.message)
                self.assertEqual([("joints", _key(look))], motions(log))

    def test_a_look_elsewhere_is_driven_to_and_the_next_ask_finds_the_arm_there(self) -> None:
        log: list[Any] = []
        arm = _arm(log)
        first = move_to_look(arm, LOOK)
        again = move_to_look(arm, LOOK)

        self.assertNotEqual(ALREADY_THERE, first.message)
        self.assertEqual(ALREADY_THERE, again.message)
        self.assertEqual([("joints", _key(LOOK))], motions(log))


class WhatTheArmCannotSayIsDrivenToTests(unittest.TestCase):
    def test_a_halted_arm_standing_at_its_look_meets_its_verb_which_refuses_as_before(self) -> None:
        log: list[Any] = []
        arm = _arm(log)
        arm.halted = "a person pressed Halt"
        moved = move_to_look(arm, HOME_JOINTS)

        self.assertFalse(moved.ok)
        self.assertIs(MotionStatus.CANCELLED, moved.status)
        self.assertIn(("refused_halted", "joints"), log)

    def test_an_arm_not_at_rest_is_driven_to_its_look(self) -> None:
        log: list[Any] = []
        arm = _arm(log)
        # Still settling: asked to wait for nothing, it is not at rest yet.
        arm.wait_until_steady = lambda timeout_s=5.0, poll_interval_s=0.02: timeout_s > 0.0  # type: ignore
        moved = move_to_look(arm, HOME_JOINTS)

        self.assertNotEqual(ALREADY_THERE, moved.message)
        self.assertEqual([("joints", _key(HOME_JOINTS))], motions(log))

    def test_an_arm_whose_tool_stands_elsewhere_is_driven_to_its_look(self) -> None:
        log: list[Any] = []
        arm = _arm(log)
        arm.move(Pose.tool_down(0.0, -500.0, 300.0, label="lower"))
        moved = move_to_look(arm, HOME_JOINTS)

        self.assertNotEqual(ALREADY_THERE, moved.message)
        self.assertEqual(("joints", _key(HOME_JOINTS)), motions(log)[-1])

    def test_an_arm_whose_readings_raise_is_driven_to_its_look(self) -> None:
        for reading in ("get_joint_positions", "get_tcp_pose", "fk", "wait_until_steady", "halt_state"):
            with self.subTest(reading=reading):
                log: list[Any] = []
                arm = _arm(log)

                def broken(*_args: Any, **_kwargs: Any) -> Any:
                    raise RuntimeError("the receive stream stopped")

                setattr(arm, reading, broken)
                move_to_look(arm, HOME_JOINTS)
                self.assertEqual([("joints", _key(HOME_JOINTS))], motions(log))

    def test_an_arm_not_connected_is_refused_by_its_verb_as_before(self) -> None:
        log: list[Any] = []
        arm = _arm(log)
        arm.is_connected = False
        moved = move_to_look(arm, HOME_JOINTS)

        self.assertIs(MotionOutcome.REFUSED, moved.outcome)
        self.assertIs(MotionStatus.CONNECTION_ERROR, moved.status)
        self.assertEqual([], motions(log))

    def test_a_camera_world_the_move_would_be_refused_for_refuses_the_look_as_before(self) -> None:
        class _NoWorld(TaskArm):
            def camera_world_for_move(self, camera_world: Any = None) -> CameraWorldStamp:
                return CameraWorldStamp.missing("no live camera world is wired")

            def live_camera_world_wired(self) -> bool:
                return False

        log: list[Any] = []
        moved = move_to_look(_NoWorld(log), HOME_JOINTS)

        self.assertIs(MotionOutcome.REFUSED, moved.outcome)
        self.assertNotEqual(ALREADY_THERE, moved.message)
        self.assertEqual([], motions(log))

    def test_a_desk_arm_at_its_look_moves_as_it_always_did(self) -> None:
        arm = DummyRobotArm()
        arm.connect()
        arm.move_to_joints(HOME_JOINTS)
        moved = move_to_look(arm, HOME_JOINTS)

        self.assertTrue(moved.ok)
        self.assertNotEqual(ALREADY_THERE, moved.message)


class TheSurveyAndARestartTests(unittest.TestCase):
    def test_a_survey_from_the_look_the_arm_stands_at_sends_nothing_and_locates_there(self) -> None:
        from src.robot.execution.place_target import survey

        log: list[Any] = []
        arm = _arm(log)
        locator = ScriptedLocator({"blue bin": [bin_object()]}, arm=arm, log=log)
        found = survey(arm, [locator], "blue bin", object_phrase="", looks=(HOME_JOINTS,))

        self.assertIsNotNone(found.kept)
        self.assertEqual([], motions(log))
        self.assertEqual(["blue bin"], locator.asked)

    def test_a_restart_goes_home_then_surveys_its_home_look_with_no_move_of_no_length(self) -> None:
        from src.robot.execution.task import PlaceAt

        log: list[Any] = []
        arm = _arm(log)
        locator = ScriptedLocator({"blue bin": [bin_object()]}, arm=arm, log=log)
        ran = run(["part"], place=PlaceAt(camera="blue bin"), arm=arm, wrist=True, looks=(HOME_JOINTS,),
                  locators=[locator], first_motion="return")

        self.assertEqual(("home",), motions(ran.log)[0])
        survey = ran.log.index(("event", "task.survey_started"))
        part = ran.log.index(("event", "task.part_started"))
        self.assertEqual([], motions(ran.log[survey:part]), "the survey drove to the home the arm stood at")
        self.assertIn("blue bin", locator.asked)


# ---------------------------------------------------------------------------------------------------------------------
# A pick from the look the arm stands at
# ---------------------------------------------------------------------------------------------------------------------

_CONTROLLER = RobotCapabilities(vendor="ur", model="ur10", dof=6, supports_joint_move=True, supports_linear_move=True,
                                supports_async_move=False, has_native_fk=True, has_native_ik=True,
                                has_force_control=False, is_simulated=False)


class _ControllerArm(DummyRobotArm):
    """A UR as a wrist pick meets it: a controller's arm, its motions planned and judged, its joints and its tool kept
    together (a joint move puts the tool where ``table`` says, a Cartesian move leaves every look), every motion
    written on ``events``."""

    plans_paths = True

    def __init__(self, events: list[Any], table: "dict[tuple[float, ...], Pose]", start: JointPositions) -> None:
        super().__init__(initial_pose=table[_key(start)])
        self.events = events
        self.table = dict(table)
        self.connect()
        self._joints = start

    @property
    def capabilities(self) -> RobotCapabilities:
        return _CONTROLLER

    def line_motion(self) -> LineReading:
        return LineReading(LineMotion.CHECKED, "cuRobo and the exact guard judge every sample")

    def fk(self, joints: JointPositions) -> Pose:
        return self.table[_key(joints)]

    def move_to_joints(self, joints: JointPositions, **_keywords: Any) -> MotionResult:
        self.events.append(("joints", _key(joints)))
        self._joints = joints
        self._tcp = self.table[_key(joints)]
        return MotionResult.executed(MotionCommand.MOVE_JOINTS, target_joints=joints, message="move_to_joints")

    def move(self, pose: Any, **keywords: Any) -> MotionResult:
        self.events.append(("move", tuple(round(float(v), 1) for v in pose.position_mm)))
        self._joints = JointPositions.deg(0.0, -45.0, -45.0, -45.0, 45.0, 0.0)
        return super().move(pose, **keywords)


class APickFromTheLookTheArmStandsAtTests(unittest.TestCase):
    def test_the_pick_perceives_there_with_nothing_sent_and_drives_there_again_after_its_grasp(self) -> None:
        from tests.test_a_pick_looks_from_the_joints_its_program_declares import GRASP_MM, LOOK_A, _Calculator, \
            _Camera, _Hand
        from src.geometry import Transform
        from src.robot.execution.autonomous_grasp import AutonomousGraspService
        from src.robot.grasping.motion.frame_resolver import EyeInHandFrameResolver

        events: list[Any] = []
        at_a = Pose(position_mm=np.array([GRASP_MM[0], GRASP_MM[1], 600.0]),
                    quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]), frame=Frame.BASE, label="look a")
        arm = _ControllerArm(events, {_key(LOOK_A): at_a}, LOOK_A)
        service = AutonomousGraspService.from_components(
            arm=arm, calculator=_Calculator(),  # type: ignore[arg-type]
            perception=_Camera(events, [True]), gripper=_Hand(events),  # type: ignore[arg-type]
            frame_resolver=EyeInHandFrameResolver(t_cam_to_tool=Transform.from_matrix(
                np.eye(4), from_frame=Frame.CAMERA, to_frame=Frame.TOOL)), max_attempts=1)

        first = service.pick(look=[LOOK_A])
        self.assertTrue(first.succeeded, first.failure_summary())
        self.assertEqual("perceive", events[0], "a move of no length was sent before the look perceived")
        self.assertEqual(1, len(first.looks))

        del events[:]
        second = service.pick(look=[LOOK_A])
        self.assertTrue(second.succeeded, second.failure_summary())
        self.assertEqual([("joints", _key(LOOK_A)), "perceive"], events[:2], "the grasp left the look, so it is "
                                                                          "driven to again")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
