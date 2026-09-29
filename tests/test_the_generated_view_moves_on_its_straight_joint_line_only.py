"""The one view a wrist pick generates moves on the straight joint line only, screened and capped, or not at all.

The owner, 2026-09-29 (addendum 3 of the multi-view plan), a safety requirement: the generated view is the very last
resort, after every declared look, aimed at the jaw contact face of the chosen grasp that no view showed, or at the side
no look faced when there is no grasp. Its candidates come smallest turn first (``unseen_side.orbit_views``), and every
one is screened before anything moves: the arm's own ``nearest_configuration`` (inverse kinematics, the cable window,
the workspace box, the joint limits, self-collision), a travel of at most 150 degrees for any joint from where the arm
stands, and half a turn from home for every joint on top. A move past 120 degrees is said as a WARNING. The motion runs
only on the straight joint line from where the arm stands to the screened configuration, judged against the camera
world holding every frame of the pick; a line that is not clear skips that angle, and cuRobo is never asked to plan
around it, at any angle. At most ``ORBIT_MAX_MOTIONS`` motions are tried; a turn the screen gives up sends nothing and
does not count. An arm whose world does not hold the frames of the pick, has no world, or declines it generates nothing.
The move back to the look that saw the part is held to the same line and the same world, and to the same travel cap from
where the arm stands, a WARNING past 120 degrees: where any of it fails it is refused, and the approach starts where the
arm stands.

These tests drive ``src.robot.execution.generated_view`` on a desk arm that names a configuration for every angle,
drives the joint lines it is asked to, and fails the test the moment anything asks its planner. The part is the 40 mm
cube of ``tests/_wrist_views.py`` seen from its +x side 45 degrees up, so its -x face, jaw 1's, is 140 degrees round.
"""

from __future__ import annotations

import math
import unittest
from collections.abc import Callable
from typing import Any

import numpy as np

from src.geometry import Pose
from src.robot.core import NO_PLAN_FAIL_SAFE_MESSAGE, JointPositions, MotionCommand, MotionResult, MotionStatus
from src.robot.core.errors import CameraWorldUnavailable, RobotConnectionError, RobotKinematicsError
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.execution import motion
from src.robot.execution.generated_view import (
    CABLE_WINDOW_DEG,
    MAX_JOINT_TRAVEL_DEG,
    WARN_BEYOND_DEG,
    aim_at_faces,
    aim_at_unseen_side,
    go_to_generated_view,
    move_back_to_look,
    refused_before_sending,
)
from src.robot.grasping.multiview.unseen_side import ORBIT_MAX_MOTIONS, JawFace, away_from_views, orbit_target
from tests._wrist_views import CUBE_CENTRE, HEIGHT, WIDTH, K, camera_looking_at, joints_key, tool_for

_LOGGER = "src.robot.execution.generated_view"
#: Where the arm stands: the look from the cube's +x side, base joint at 0.
_HERE_DEG = (0.0, -90.0, -110.0, -60.0, 90.0, 0.0)
#: The cube's -x face, jaw 1's for a grasp closing along base x.
_JAW_1 = JawFace(side=-1, centre_mm=(-20.0, -700.0, 20.0), normal=(-1.0, 0.0, 0.0), points=0, cells_seen=0,
                 cells_touchable=48)
#: A face 60 degrees round from the look, which a turn of 20 degrees shows.
_SIXTY_ROUND = JawFace(side=1, centre_mm=(10.0, -682.7, 20.0), normal=(0.5, math.sqrt(0.75), 0.0), points=0,
                       cells_seen=0, cells_touchable=48)
#: "Where the arm stands", read off the arm.
_ARM = object()


def _bearing(pose: Pose) -> float:
    """The tool's bearing round the cube's vertical, in degrees."""
    x, y, _ = (float(v) for v in pose.position_mm)
    return math.degrees(math.atan2(y - CUBE_CENTRE[1], x - CUBE_CENTRE[0]))


_LOOK_CAMERA = camera_looking_at(bearing_deg=0.0)
_LOOK_TOOL = tool_for(_LOOK_CAMERA)
#: The smallest turns that show jaw 1's face from the look, either way round. A camera 45 degrees over the cube's centre
#: meets the incidence limit on the -x face at 135 degrees in closed form; the face's own centre lies 20 mm further round
#: and the camera beside the tool, so the limit is met one step on (``unseen_side.orbit_views`` with this lens).
_FIRST, _SECOND = 140.0, -140.0


def _turned_to(degrees: float) -> tuple[float, ...]:
    """The configuration the desk arm names for a turn of ``degrees`` from the look, in degrees."""
    return joints_key(JointPositions.deg(_HERE_DEG[0] + degrees, *_HERE_DEG[1:]))


def _azimuth(pose: Pose) -> float:
    """How far round the cube's vertical ``pose`` is turned from the look, in degrees, (-180, 180]."""
    turned = (_bearing(pose) - _bearing(_LOOK_TOOL)) % 360.0
    return round(turned - 360.0 if turned > 180.0 else turned, 3)


def _base_turned(pose: Pose) -> JointPositions:
    """The configuration an orbit turns to: the base joint turned as far as the orbit, the rest as the look stands."""
    return JointPositions.deg(_HERE_DEG[0] + _azimuth(pose), *_HERE_DEG[1:])


class _Arm(DummyRobotArm):
    """A desk arm that names a configuration for every pose and drives the joint lines it is asked to.

    ``configure`` names the configuration of a pose (the base joint turned by the orbit, by default); ``unreachable``
    holds the azimuths it refuses as a UR's inverse kinematics refuses a pose out of reach; ``blocked`` the targets
    whose straight joint line the planner world finds not clear; ``answers`` a result or a raise for a target, as a
    drive answers once it was asked. ``lines`` is every straight joint line it was asked to drive, from where it stood
    to where it went, in degrees. Its ``move_to_joints``, the verb that plans around a line, fails the test.
    """

    def __init__(self, test: unittest.TestCase, *, configure: Callable[[Pose], JointPositions] = _base_turned,
                 home_deg: tuple[float, ...] = _HERE_DEG) -> None:
        super().__init__()
        self.connect()
        self.test = test
        self._joints = JointPositions.deg(*_HERE_DEG)
        self.home_joint_positions = tuple(JointPositions.deg(*home_deg).tolist())
        self.configure = configure
        self.unreachable: set[float] = set()
        self.blocked: set[tuple[float, ...]] = set()
        self.answers: dict[tuple[float, ...], Any] = {}
        self.screened: list[float] = []
        self.lines: list[tuple[tuple[float, ...], tuple[float, ...]]] = []

    def nearest_configuration(self, pose: Pose) -> JointPositions:
        azimuth = _azimuth(pose)
        self.screened.append(azimuth)
        if azimuth in self.unreachable:
            raise RobotKinematicsError(f"orbit {azimuth:+g} deg: refused by the inverse kinematics: out of reach")
        return self.configure(pose)

    def move_to_joints_on_the_line(self, joints: JointPositions) -> MotionResult:
        key = joints_key(joints)
        self.lines.append((joints_key(self._joints), key))
        answer = self.answers.get(key)
        if isinstance(answer, BaseException):
            raise answer
        if answer is not None:
            status, message = answer
            return MotionResult.failed(status, MotionCommand.MOVE_JOINTS, target_joints=joints, message=message)
        if key in self.blocked:
            return MotionResult.failed(
                MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_JOINTS, target_joints=joints,
                message="the planner refused this straight joint line: it passes 4 mm from a held frame's box; "
                        "nothing is planned around it")
        self._joints = joints
        return MotionResult.executed(MotionCommand.MOVE_JOINTS, target_joints=joints,
                                     message="move_to_joints_on_the_line")

    def move_to_joints(self, joints: JointPositions, **_: Any) -> MotionResult:
        self.test.fail(f"the planner was asked for {joints_key(joints)}: move_to_joints plans around a line")
        raise AssertionError  # unreachable: fail raises


class _NoLineArm(DummyRobotArm):
    """Names configurations and drives no straight joint line of its own: its joint verb may plan around one."""

    def __init__(self) -> None:
        super().__init__()
        self.connect()
        self.home_joint_positions = tuple(JointPositions.deg(*_HERE_DEG).tolist())

    def nearest_configuration(self, pose: Pose) -> JointPositions:
        return _base_turned(pose)


def _go(arm: Any, *, faces: tuple[JawFace, ...] = (_JAW_1,), holding: bool = True, home: Any = "arm",
        controller_stopped: Callable[[], str | None] | None = None) -> Any:
    return go_to_generated_view(
        arm, aim=aim_at_faces(faces, grasp_centre_mm=CUBE_CENTRE),
        camera_to_base_mm=_LOOK_CAMERA, tool_to_base_mm=_LOOK_TOOL.to_matrix(), intrinsics=K,
        image_size=(WIDTH, HEIGHT), here=arm.get_joint_positions(),
        home=arm.home_joint_positions if home == "arm" else home, holding=holding,
        controller_stopped=controller_stopped,
    )


def _deg(joints: JointPositions | None) -> tuple[float, ...]:
    assert joints is not None
    return joints_key(joints)


# ---------------------------------------------------------------------------------------------------------------------
# The generated view: screened, capped, and on the straight joint line only
# ---------------------------------------------------------------------------------------------------------------------


class TheGeneratedViewTests(unittest.TestCase):
    def test_the_smallest_angle_that_shows_the_face_runs_on_the_straight_joint_line(self) -> None:
        arm = _Arm(self)

        with self.assertLogs(_LOGGER, level="INFO"):
            view = _go(arm)

        self.assertTrue(view.moved, view.skipped)
        self.assertEqual(_FIRST, view.azimuth_deg, "the smallest turn that shows the -x face is 140 degrees")
        self.assertEqual(_turned_to(_FIRST), _deg(view.joints))
        # The executed joint path: one straight joint line, from where the arm stood to the screened configuration.
        self.assertEqual([(_HERE_DEG, _deg(view.joints))], arm.lines)
        self.assertEqual([_FIRST], arm.screened, "a candidate was screened past the one that moved")
        self.assertTrue(view.label.startswith("orbit +140 deg ("), view.label)
        self.assertEqual("", view.skipped)

    def test_a_candidate_whose_line_is_not_clear_is_skipped_and_the_next_angle_tried(self) -> None:
        arm = _Arm(self)
        arm.blocked.add(_turned_to(_FIRST))

        view = _go(arm)

        self.assertTrue(view.moved, view.skipped)
        self.assertEqual(_SECOND, view.azimuth_deg)
        self.assertEqual([_FIRST, _SECOND], [end[0] for _, end in arm.lines])
        ((turn, why),) = view.tried
        self.assertEqual(_FIRST, turn)
        self.assertIn("not clear", why)
        self.assertIn("nothing is planned around it", why)

    def test_an_angle_the_arm_cannot_stand_at_is_screened_out_before_anything_moves(self) -> None:
        arm = _Arm(self)
        arm.unreachable.add(_FIRST)

        view = _go(arm)

        self.assertEqual(_SECOND, view.azimuth_deg)
        self.assertEqual([_FIRST, _SECOND], arm.screened)
        self.assertEqual(1, len(arm.lines), "an angle the screen refused was driven")
        self.assertIn("inverse kinematics", view.tried[0][1])

    def test_no_joint_travels_more_than_150_degrees_from_where_it_stands(self) -> None:
        self.assertEqual(150.0, MAX_JOINT_TRAVEL_DEG)

        def wound(pose: Pose) -> JointPositions:
            turned = _base_turned(pose)
            if _azimuth(pose) > 0.0:  # wrist 3 wound a further 151 degrees on every positive turn
                values = turned.degrees()
                return JointPositions.deg(*values[:5], values[5] + 151.0)
            return turned

        arm = _Arm(self, configure=wound)

        view = _go(arm)

        self.assertEqual(_SECOND, view.azimuth_deg)
        self.assertEqual(1, len(arm.lines), "a configuration past the travel cap was driven")
        self.assertIn("joint 6 would travel 151.0 deg", view.tried[0][1])
        self.assertIn("150", view.tried[0][1])

    def test_the_cable_window_about_home_holds_on_top(self) -> None:
        """Home at -60 degrees on the base: +140 lies 200 degrees from it, past half a turn; -140 lies 80 from it."""
        self.assertEqual(180.0, CABLE_WINDOW_DEG)
        arm = _Arm(self, home_deg=(-60.0, *_HERE_DEG[1:]))

        view = _go(arm)

        self.assertEqual(_SECOND, view.azimuth_deg)
        self.assertEqual(1, len(arm.lines))
        self.assertIn("joint 1 would stand 200.0 deg from home", view.tried[0][1])

    def test_every_generated_move_beyond_120_degrees_is_a_warning(self) -> None:
        self.assertEqual(120.0, WARN_BEYOND_DEG)
        arm = _Arm(self)

        with self.assertLogs(_LOGGER, level="WARNING") as said:
            _go(arm)

        (warning,) = [line for line in said.output if line.startswith("WARNING")]
        self.assertIn("orbit +140 deg", warning)
        self.assertIn("140.0 deg", warning)

    def test_a_small_turn_is_not_a_warning(self) -> None:
        """A face 60 degrees round from the look shows at a 20 degree turn: no warning."""
        arm = _Arm(self)

        with self.assertNoLogs(_LOGGER, level="WARNING"):
            view = _go(arm, faces=(_SIXTY_ROUND,))

        self.assertEqual(20.0, view.azimuth_deg)

    def test_a_joint_travelling_past_120_degrees_on_a_small_turn_is_a_warning(self) -> None:
        """A 20 degree turn about the part on which wrist 3 winds on by 130 degrees: the joint's travel is said."""
        def wound(pose: Pose) -> JointPositions:
            values = _base_turned(pose).degrees()
            return JointPositions.deg(*values[:5], values[5] + 130.0)

        arm = _Arm(self, configure=wound)

        with self.assertLogs(_LOGGER, level="WARNING") as said:
            view = _go(arm, faces=(_SIXTY_ROUND,))

        self.assertEqual(20.0, view.azimuth_deg)
        (warning,) = [line for line in said.output if line.startswith("WARNING")]
        self.assertIn("joint 6 travelling 130.0 deg", warning)
        self.assertIn("a turn of 20 deg", warning)

    def test_a_turn_past_120_degrees_is_a_warning_however_little_the_joints_travel(self) -> None:
        """An arm whose base turns half as far as the view turns about the part: the turn is said."""
        def half(pose: Pose) -> JointPositions:
            return JointPositions.deg(_HERE_DEG[0] + _azimuth(pose) / 2.0, *_HERE_DEG[1:])

        arm = _Arm(self, configure=half)

        with self.assertLogs(_LOGGER, level="WARNING") as said:
            view = _go(arm)

        self.assertEqual(_FIRST, view.azimuth_deg)
        (warning,) = [line for line in said.output if line.startswith("WARNING")]
        self.assertIn("a turn of 140 deg", warning)
        self.assertIn("joint 1 travelling 70.0 deg", warning)

    def test_a_long_turn_whose_line_is_refused_is_no_warning_of_a_drive(self) -> None:
        """Nothing drove: every straight line round the part is refused before anything was sent, and the log does not
        say a long move that never happened."""
        arm = _Arm(self)
        arm.blocked = _AllKeys()

        with self.assertNoLogs(_LOGGER, level="WARNING"):
            view = _go(arm)

        self.assertFalse(view.moved)
        self.assertEqual(3, len(arm.lines))

    def test_at_most_three_motions_are_tried(self) -> None:
        self.assertEqual(3, ORBIT_MAX_MOTIONS)
        arm = _Arm(self, configure=lambda pose: _base_turned(pose))
        # Every straight joint line is refused: the planner world is not clear anywhere round the part.
        arm.blocked = _AllKeys()

        view = _go(arm)

        self.assertFalse(view.moved)
        self.assertEqual(3, len(arm.lines), "more motions were tried than a generated view may")
        self.assertIn("3 motion(s) tried", view.skipped)

    def test_turns_the_screen_gives_up_do_not_count_against_the_motions(self) -> None:
        """``ORBIT_MAX_MOTIONS`` counts straight joint lines driven. A turn the screen gives up sends nothing, so four
        turns out of reach still leave the fifth its motion."""
        given_up: list[float] = []

        def four_out_of_reach(pose: Pose) -> JointPositions:
            if len(given_up) < 4:
                given_up.append(_azimuth(pose))
                raise RobotKinematicsError(f"orbit {_azimuth(pose):+g} deg: out of reach")
            return _base_turned(pose)

        arm = _Arm(self, configure=four_out_of_reach)

        view = _go(arm)

        self.assertTrue(view.moved, view.skipped)
        self.assertEqual(5, len(arm.screened))
        self.assertEqual(1, len(arm.lines))
        self.assertEqual(given_up, [turn for turn, _ in view.tried])

    def test_a_motion_that_may_have_moved_stops_the_view_and_nothing_else_is_commanded(self) -> None:
        arm = _Arm(self)
        arm.answers[_turned_to(_FIRST)] = (
            MotionStatus.CONTROLLER_REJECTED, "the drive refused after the move was judged, and moveJ may have been sent")

        view = _go(arm)

        self.assertFalse(view.moved)
        self.assertIsNotNone(view.stopped)
        self.assertIs(MotionStatus.CONTROLLER_REJECTED, view.stopped.status)
        self.assertEqual(1, len(arm.lines), "another angle was driven after a motion that may have moved the arm")

    def test_a_controller_that_stopped_ends_the_view_at_the_first_refusal(self) -> None:
        arm = _Arm(self)
        arm.blocked.add(_turned_to(_FIRST))

        view = _go(arm, controller_stopped=lambda: "the controller cannot move: protective stop")

        self.assertIsNotNone(view.stopped)
        self.assertEqual("the controller cannot move: protective stop", view.controller)
        self.assertEqual(1, len(arm.lines))

    def test_a_camera_that_cannot_vouch_on_the_way_is_a_fault(self) -> None:
        arm = _Arm(self)
        arm.answers[_turned_to(_FIRST)] = CameraWorldUnavailable(
            camera="wrist", verdict="blind", attempts=4, reason="the wrist camera could not vouch for the cell")

        with self.assertRaises(CameraWorldUnavailable):
            _go(arm)

    def test_a_controller_that_cannot_say_where_the_arm_stands_is_a_fault(self) -> None:
        arm = _Arm(self)

        def lost(pose: Pose) -> JointPositions:
            raise RobotConnectionError("the controller's joint positions could not be read")

        arm.configure = lost

        with self.assertRaises(RobotConnectionError):
            _go(arm)
        self.assertEqual([], arm.lines)

    def test_an_arm_that_does_not_choose_configurations_generates_nothing_and_says_why(self) -> None:
        """The sim's IsaacRobotArm and the desk arm name no configuration: nothing moves, and the reason is said."""
        arm = DummyRobotArm()
        arm.connect()

        with self.assertLogs(_LOGGER, level="INFO") as said:
            view = _go(arm, home=JointPositions.deg(*_HERE_DEG))

        self.assertFalse(view.moved)
        self.assertIn("ChoosesConfigurations", view.skipped)
        self.assertTrue(any("ChoosesConfigurations" in line for line in said.output), said.output)

    def test_an_arm_that_cannot_drive_a_straight_joint_line_alone_generates_nothing(self) -> None:
        view = _go(_NoLineArm())

        self.assertFalse(view.moved)
        self.assertIn("DrivesJointLines", view.skipped)

    def test_a_world_that_does_not_hold_the_frames_of_the_pick_generates_nothing(self) -> None:
        """A world that holds no frame of the pick, or no live world at all: a generated view is judged against the
        world holding every frame of the pick, or it does not run."""
        for name, world in (("a world that does not hold them", object()), ("no live world", None)):
            with self.subTest(name):
                arm = _Arm(self)
                if world is not None:
                    arm.live_planner_world = world  # type: ignore[attr-defined]

                view = _go(arm, holding=False)

                self.assertFalse(view.moved)
                self.assertIn("does not hold the frames of this pick", view.skipped)
                self.assertEqual([], arm.lines)
                self.assertEqual([], arm.screened, "a turn was screened for a view that could not be judged")
                self.assertTrue(_go(arm, holding=True).moved)

    def test_a_camera_world_declined_for_the_arm_generates_nothing(self) -> None:
        """A program's ``without_camera_world`` block: whatever the caller says is held, nothing it made up moves."""
        arm = _Arm(self)

        with arm.without_camera_world("bench: no camera is wired to the planner"):
            view = _go(arm, holding=True)

        self.assertFalse(view.moved)
        self.assertIn("declined", view.skipped)
        self.assertIn("bench: no camera is wired to the planner", view.skipped)
        self.assertEqual([], arm.lines)

    def test_an_arm_that_names_no_home_generates_nothing(self) -> None:
        arm = _Arm(self)

        view = _go(arm, home=None)

        self.assertFalse(view.moved)
        self.assertIn("home", view.skipped)
        self.assertEqual([], arm.lines)

    def test_a_look_that_cannot_be_turned_generates_nothing(self) -> None:
        arm = _Arm(self)

        view = go_to_generated_view(
            arm, aim=aim_at_faces((_JAW_1,), grasp_centre_mm=CUBE_CENTRE), camera_to_base_mm=np.eye(3),
            tool_to_base_mm=_LOOK_TOOL.to_matrix(), intrinsics=K, image_size=(WIDTH, HEIGHT),
            here=arm.get_joint_positions(), home=arm.home_joint_positions, holding=True)

        self.assertFalse(view.moved)
        self.assertIn("cannot be turned", view.skipped)
        self.assertEqual([], arm.screened)

    def test_a_face_no_turn_shows_generates_nothing(self) -> None:
        """A bottom face is shown by no turn about the vertical."""
        arm = _Arm(self)
        bottom = JawFace(side=-1, centre_mm=(0.0, -700.0, 0.0), normal=(0.0, 0.0, -1.0), points=0, cells_seen=0,
                         cells_touchable=48)

        view = _go(arm, faces=(bottom,))

        self.assertFalse(view.moved)
        self.assertIn("no turn", view.skipped)
        self.assertEqual([], arm.screened)


class _AllKeys(set):  # type: ignore[type-arg]
    """Every target is in it."""

    def __contains__(self, _item: object) -> bool:
        return True


# ---------------------------------------------------------------------------------------------------------------------
# The aim, and what counts as refused before anything was sent
# ---------------------------------------------------------------------------------------------------------------------


class TheAimTests(unittest.TestCase):
    def test_a_grasp_aims_at_its_missing_contact_faces_about_its_centre(self) -> None:
        aim = aim_at_faces((_JAW_1,), grasp_centre_mm=CUBE_CENTRE)

        np.testing.assert_allclose(aim.target_mm, CUBE_CENTRE)
        ((centre, normal),) = aim.faces
        np.testing.assert_allclose(centre, _JAW_1.centre_mm)
        np.testing.assert_allclose(normal, _JAW_1.normal)
        self.assertEqual("the contact face at jaw 1 of the chosen grasp", aim.what)

    def test_no_grasp_aims_at_the_side_no_look_faced(self) -> None:
        cloud = np.array([[x, y, z] for x in (-20.0, 20.0) for y in (-720.0, -680.0) for z in (0.0, 40.0)])
        centres = [_LOOK_CAMERA[:3, 3]]

        aim = aim_at_unseen_side(cloud, camera_centres_mm=centres)

        target = orbit_target(cloud)
        np.testing.assert_allclose(aim.target_mm, target)
        ((centre, normal),) = aim.faces
        np.testing.assert_allclose(centre, target)
        np.testing.assert_allclose(normal, away_from_views(target, centres))
        self.assertEqual("the side of the part no look faced", aim.what)

    def test_only_a_guard_or_the_planner_refusing_before_anything_was_sent_is_a_skip(self) -> None:
        def report(status: MotionStatus, message: str = "") -> Any:
            return motion.MotionReport(
                motion.MotionVerb.MOVE_JOINTS, motion.MotionOutcome.MOTION_REFUSED,
                motion.RouteReading(motion.MotionRoute.PLANNED, "test"),
                MotionResult.failed(status, MotionCommand.MOVE_JOINTS, message=message))

        for status in (MotionStatus.WORKSPACE_REJECTED, MotionStatus.IK_FAILED, MotionStatus.IK_QUALITY_REJECTED,
                       MotionStatus.JOINT_LIMIT_REJECTED, MotionStatus.SELF_COLLISION_REJECTED,
                       MotionStatus.PAYLOAD_REJECTED, MotionStatus.CONTINUITY_REJECTED):
            with self.subTest(status=status.value):
                self.assertTrue(refused_before_sending(report(status)))
        self.assertTrue(refused_before_sending(report(MotionStatus.TIMEOUT, NO_PLAN_FAIL_SAFE_MESSAGE)))
        for status in (MotionStatus.CONTROLLER_REJECTED, MotionStatus.CONNECTION_ERROR, MotionStatus.UNKNOWN,
                       MotionStatus.INVALID_TARGET, MotionStatus.TIMEOUT, MotionStatus.UNSUPPORTED):
            with self.subTest(status=status.value):
                self.assertFalse(refused_before_sending(report(status, "moveJ may have been sent")))


# ---------------------------------------------------------------------------------------------------------------------
# The move back to the look that saw the part
# ---------------------------------------------------------------------------------------------------------------------


def _back(arm: Any, joints: JointPositions | None, *, here: Any = _ARM, holding: bool = True) -> Any:
    return move_back_to_look(arm, joints, look="the look", here=arm.get_joint_positions() if here is _ARM else here,
                             holding=holding)


def _base_at(degrees: float) -> JointPositions:
    """The look with its base joint at ``degrees``."""
    return JointPositions.deg(degrees, *_HERE_DEG[1:])


class TheMoveBackTests(unittest.TestCase):
    #: The look 140 degrees round on the base from where the arm stands: inside the travel cap, past the warning.
    _LOOK = _base_at(140.0)

    def test_the_move_back_runs_on_the_straight_joint_line_and_never_plans(self) -> None:
        arm = _Arm(self)

        with self.assertLogs(_LOGGER, level="WARNING") as said:
            back = _back(arm, self._LOOK)

        self.assertTrue(back.done, back.refused)
        self.assertEqual([(_HERE_DEG, _deg(self._LOOK))], arm.lines)
        (warning,) = [line for line in said.output if line.startswith("WARNING")]
        self.assertIn("joint 1 travelling 140.0 deg", warning)

    def test_a_short_move_back_is_no_warning(self) -> None:
        arm = _Arm(self)

        with self.assertNoLogs(_LOGGER, level="WARNING"):
            back = _back(arm, _base_at(20.0))

        self.assertTrue(back.done, back.refused)

    def test_a_move_back_past_the_travel_cap_is_refused_before_anything_moves(self) -> None:
        arm = _Arm(self)

        back = _back(arm, _base_at(170.0))

        self.assertFalse(back.done)
        self.assertIsNone(back.stopped)
        self.assertIn("joint 1 would travel 170.0 deg", back.refused)
        self.assertIn("150", back.refused)
        self.assertEqual([], arm.lines)

    def test_a_move_back_not_judged_against_the_frames_of_the_pick_is_refused(self) -> None:
        """The world does not hold them, or the camera world is declined for the arm: nothing is sent."""
        arm = _Arm(self)

        not_held = _back(arm, self._LOOK, holding=False)
        with arm.without_camera_world("bench: no camera is wired to the planner"):
            declined = _back(arm, self._LOOK)

        self.assertIn("does not hold the frames of this pick", not_held.refused)
        self.assertIn("declined", declined.refused)
        self.assertEqual([], arm.lines)

    def test_a_move_back_from_where_nobody_read_is_refused(self) -> None:
        arm = _Arm(self)

        back = _back(arm, self._LOOK, here=None)

        self.assertFalse(back.done)
        self.assertIn("where it stands", back.refused)
        self.assertEqual([], arm.lines)

    def test_a_move_back_whose_line_is_not_clear_is_refused_and_says_why(self) -> None:
        arm = _Arm(self)
        arm.blocked.add(_deg(self._LOOK))

        back = _back(arm, self._LOOK)

        self.assertFalse(back.done)
        self.assertIsNone(back.stopped)
        self.assertIn("self_collision_rejected", back.refused)
        self.assertIn("nothing is planned around it", back.refused)
        self.assertEqual(_HERE_DEG, _deg(arm.get_joint_positions()), "the arm moved on a refused move back")

    def test_a_move_back_that_may_have_moved_is_a_stop(self) -> None:
        arm = _Arm(self)
        arm.answers[_deg(self._LOOK)] = (MotionStatus.TIMEOUT, "moveJ did not complete within its budget")

        back = _back(arm, self._LOOK)

        self.assertFalse(back.done)
        self.assertIsNotNone(back.stopped)
        self.assertEqual("", back.refused)

    def test_a_move_back_to_joints_nobody_read_is_refused(self) -> None:
        arm = _Arm(self)

        back = _back(arm, None)

        self.assertFalse(back.done)
        self.assertIn("were not read", back.refused)
        self.assertEqual([], arm.lines)

    def test_an_arm_that_cannot_drive_a_straight_joint_line_alone_is_not_moved_back(self) -> None:
        back = _back(_NoLineArm(), self._LOOK)

        self.assertFalse(back.done)
        self.assertIn("DrivesJointLines", back.refused)


class TheMotionVerbTests(unittest.TestCase):
    def test_an_arm_without_the_verb_is_refused_before_any_command(self) -> None:
        arm = _NoLineArm()

        moved = motion.move_joints_on_the_line(arm, JointPositions.deg(*_HERE_DEG))

        self.assertIs(motion.MotionOutcome.REFUSED, moved.outcome)
        self.assertIs(MotionStatus.UNSUPPORTED, moved.status)
        self.assertIn("DrivesJointLines", moved.message)

    def test_a_declined_camera_world_refuses_the_line_before_any_command(self) -> None:
        """The line only runs judged against the camera world: a decline in scope for the arm refuses it."""
        arm = _Arm(self)

        with arm.without_camera_world("bench: no camera is wired to the planner"):
            moved = motion.move_joints_on_the_line(arm, _base_at(20.0))

        self.assertIs(motion.MotionOutcome.REFUSED, moved.outcome)
        self.assertIs(MotionStatus.UNSUPPORTED, moved.status)
        self.assertIn("declined", moved.message)
        self.assertEqual([], arm.lines)


if __name__ == "__main__":
    unittest.main()
