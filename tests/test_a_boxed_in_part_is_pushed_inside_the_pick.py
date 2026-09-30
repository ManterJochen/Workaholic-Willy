"""A boxed-in part is pushed inside the pick and then picked: the recovery of 2026-09-29 and 2026-09-30, end to end.

The owner's cell in miniature: a UR10 with cuRobo judging every straight line against the camera world (the arm double
here says so, ``LineMotion.CHECKED``), the Robotiq Hand-E as ``jaw_io`` single_toggle on tool DO0, where every change of
DO0 moves the jaws once and the program keeps the count (the real :class:`JawIOGripper` on a recording tool I/O,
connected once with a person saying "open", and nobody asked after that: any question fails the test), and the tilted
D415 on the wrist looking at a 40 mm part on the bench from three declared looks (``tests/_wrist_views.py``). A 30 mm
post stands 22 mm off the part's -x side at its -y end, so every grasp candidate of the part collides with something
(``ALL_COLLIDED``, which the calculator double reports as both real calculators do, with ``RESCAN_RECOMMENDED``).

Everything runs through ``PickRun`` or the console's run body (``RunRegistry._drive``) on the real service, its recovery
loop and the real pick loop; the calculator, the policy and the arm are doubles. The arm moves the part on the push leg,
so the look taken again after the push sees it where it was pushed, and the calculator grasps it there.

Pinned:

* dense_clutter pushes the part once (P0 a plain move like a grasp approach, then the four judged lines at the owner's
  speeds: down 50 mm/s, push 25 mm/s at 0.1 m/s^2, back 50 mm/s, up 50 mm/s), the arm goes back to the look it stood at,
  looks again, and the part is picked from there. DO0 is never written, the count never changes and nobody is asked;
  the budgets count the push by the plan's own centre; the part and its path stay out of the planner world;
* from the one view the pick generated (every declared look ranked the part ``ALL_COLLIDED``), the arm comes back on
  the straight joint line only, never through the planner's detour (the owner's rule of 2026-09-29 for a configuration
  the pick made up), and a line that does not run stops there;
* ``auto`` never pushes; a jaw count that says closed never pushes and never asks, and the pick goes on; nor does a part
  whose grasps were not refused for a collision (``no_valid_grasp``, no candidate);
* a count nobody can vouch for when the push reads it (DO0 switched at the pendant, or unreadable) ends the pick as a
  gripper fault, as the toggle's own refusal ends one (the owner, 2026-09-30): nothing more moves, nobody is asked, and
  the campaign and the console run stop on it;
* a refusal before anything moved falls through (a look-less pick rescans where it stands), with no motion; the down leg
  refused before it was sent sends the arm back to the look and falls through, and counts against the budgets, as
  every push that commanded a motion does; a stop asked for while the push is planned moves nothing;
* a controller found stopped before the push ends the pick with nothing commanded, as a stopped controller ends it
  anywhere, and the campaign with it;
* a refusal after contact, a failed move back, a protective stop mid-push and a camera world that cannot vouch mid-push
  stop where the arm is: nothing more moves, nothing clears a stop, the service says ``unsafe_recovery_refused``
  (``controller_stopped`` for the stop), the campaign and the console run end there (``needs_person``), and the service
  starts no further pick until a person decides (a new campaign, or ``acknowledge_needs_person``);
* budgets across picks: 1 per part, 2 per pick (over every re-pick of one pick), 5 per campaign; ``next_target``'s zones
  live for the pick they were made in and the next two;
* ``next_target`` skips the failed part, same label only, runs the looks again for the next part (the owner, 2026-09-30:
  declared looks with the early stop), and stops with the zones' sentence when only skipped parts remain; a rescan of
  the same part never repeats the looks; an empty bench is ``no_target`` whatever recovery tried;
* ``push_mm``: 51 and 9 refused, 40 taken when the cell's ceiling allows it, refused where it does not; the config's
  ``recovery.fixture.push_distance_mm`` when unset; through ``PickRun`` here, and through the console API in
  ``tests/test_api_pick_push_distance.py`` (only a ``test_api_*`` file may import the web framework).

The re-pick after a push on the real ``GraspExecutionPolicy``, which opens nothing and closes the toggle once, is pinned in
``tests/test_a_push_and_its_re_pick_leave_the_toggle_alone.py``.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Any
from unittest import mock

import numpy as np

from src.config.schema.robot.grasping_schema import GraspingSupportConfig
from src.geometry import Pose
from src.robot.core import (
    JointPositions,
    MotionCommand,
    MotionResult,
    MotionStatus,
    RobotError,
    RobotMode,
    RobotStatus,
    SafetyMode,
)
from src.robot.core.arm_capabilities import LineMotion, LineReading
from src.robot.core.errors import CameraWorldUnavailable
from src.robot.execution.autonomous_grasp import (
    AutonomousGraspOutcome,
    AutonomousGraspService,
    EffectiveGraspingConfig,
    EffectiveRecoveryOrchestratorConfig,
    GraspMode,
)
from src.robot.execution.pick_run import PickOutcome, PickRun, Recording
from src.robot.grasping.motion.frame_resolver import EyeInHandFrameResolver
from src.robot.grasping.recovery.push_budgets import PushBudgets
from src.robot.grasping.recovery.push_gate import PushCampaign, PushCell
from src.robot.grasping.recovery.push_planner import AxisBox, PushHand
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from tests._wrist_views import CUBE, Box, LookingArm, WristCamera, camera_to_tool, joints_key
from tests.test_a_toggle_asks_where_its_jaws_stand import Person, _changes, _toggle
from tests.test_a_wrist_pick_looks_until_its_grasp_is_safe import (
    HAND_E,
    LOOK_MINUS_X,
    LOOK_PLUS_X,
    LOOK_PLUS_Y,
    _keys,
    _Policy,
    _poses,
    _turned,
    _World,
)

#: A 30 mm post 22 mm off the part's -x face, at its -y end: within 25 mm of the part, so a neighbour left the fingers
#: no room, and far enough aside that the open hand, kept 25 mm from it, fits beside the part.
POST = Box((-62.0, -722.0, 0.0), (-42.0, -714.0, 30.0), "post")
#: The Hand-E as the push sweeps it (config/grippers/robotiq_hande.yaml): fingers, palm and housing.
PUSH_HAND = PushHand(finger_thickness_mm=10.80, finger_width_mm=29.24, fingertip_depth_mm=10.45,
                     finger_length_mm=36.53, palm_width_mm=75.0, open_width_mm=49.99, palm_thickness_mm=75.0)
#: The workspace, its floor well under the bench: the shipped 100 mm z_min refuses every push, as it should.
WORKSPACE = AxisBox((-800.0, -1000.0, -100.0), (800.0, 800.0, 600.0))
#: The recovery fixture the config declares: the push box it only narrows, and the ceiling of a push, 50 mm.
FIXTURE = ((0.0, -700.0, 100.0), (300.0, 300.0, 300.0), 50.0)
#: A declared container's interior around the bench, the push box in place of the table the looks saw: what lets the
#: one 45 degree view of a pick handed no look plan a push at all.
CONTAINER = AxisBox((-200.0, -900.0, -10.0), (200.0, -500.0, 200.0))
LOOKS = (LOOK_PLUS_X, LOOK_MINUS_X, LOOK_PLUS_Y)
PUSH_LABELS = ["push_p0", "push_down", "push_push", "push_back", "push_up"]

_RUNNING = RobotStatus(robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.NORMAL, protective_stopped=False,
                       emergency_stopped=False)
_PROTECTIVE = RobotStatus(robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.PROTECTIVE_STOP,
                          protective_stopped=True, emergency_stopped=False, message="Safetystatus: PROTECTIVE_STOP")


def _refused(status: MotionStatus, message: str = "refused") -> MotionResult:
    return MotionResult.failed(status, MotionCommand.MOVE_TO, message=message)


class _Bench(WristCamera):
    """The wrist D415 over the bench, whose part a push moves: :meth:`push` shifts it by what the push leg drove past
    the 10 mm contact gap, as a finger would. A ``stuck`` part does not budge."""

    stuck: bool = False

    def push(self, start_mm: Any, end_mm: Any) -> None:
        if self.stuck:
            return
        run = np.asarray(end_mm, dtype=np.float64)[:2] - np.asarray(start_mm, dtype=np.float64)[:2]
        length = float(np.linalg.norm(run))
        shift = run / length * max(0.0, length - 10.0)
        moved = []
        for box in self.boxes:
            if box.label == "part" and box.holds(np.array([[0.0, -700.0, 20.0]]), margin_mm=0.0).any():
                box = Box((box.low[0] + shift[0], box.low[1] + shift[1], box.low[2]),
                          (box.high[0] + shift[0], box.high[1] + shift[1], box.high[2]), box.label)
            moved.append(box)
        self.boxes = tuple(moved)

    def centre_of(self, label: str) -> np.ndarray:
        box = next(box for box in self.boxes if box.label == label)
        return (np.asarray(box.low) + np.asarray(box.high)) / 2.0

    @property
    def pushed(self) -> bool:
        return not any(box.label == "part" and box.low[:2] == CUBE.low[:2] for box in self.boxes)


class _PushingArm(LookingArm):
    """A cuRobo UR as the pick meets it: it judges every line (``LineMotion.CHECKED``, ``plans_paths``), reports its
    controller, holds the frames of a pick in its live world (``_World``) and writes every motion down.

    ``pushes`` is every call of the typed ``move`` (the push's), by label, with its keywords. ``answers`` maps a label
    to what that motion does instead of arriving: a :class:`MotionResult`, an exception, or ``"stop"`` (the controller
    protective-stops during it). ``back_refused`` is what the joint move back to the look answers once a push moved the
    arm (P0 and on): a :class:`MotionResult`, or an exception it raises; ``back_stops`` protective-stops the controller as
    it fails. ``status`` is what the controller says, or an exception its read raises. Clearing a stop fails the test:
    only a person does that."""

    plans_paths = True

    def __init__(self, test: unittest.TestCase, *, answers: dict[str, Any] | None = None) -> None:
        super().__init__(_poses())
        self.test = test
        self.live_planner_world = _World()
        self.camera: _Bench | None = None
        self.pushes: list[tuple[str, tuple[float, ...], dict[str, Any]]] = []
        self.answers = dict(answers or {})
        self.status: RobotStatus | BaseException = _RUNNING
        self.back_refused: MotionResult | BaseException | None = None
        self.back_stops = False
        self._pushed = False
        self._left = False

    def line_motion(self) -> LineReading:
        return LineReading(LineMotion.CHECKED, "cuRobo judges every sample of a straight line before it drives it")

    def get_robot_status(self) -> RobotStatus:
        if isinstance(self.status, BaseException):
            raise self.status
        return self.status

    def recover_from_protective_stop(self) -> bool:
        self.test.fail("nothing clears a protective stop but a person")
        return False

    def move(self, pose: Any, **keywords: Any) -> MotionResult:
        label = str(pose.label)
        self.pushes.append((label, tuple(round(float(v), 1) for v in pose.position_mm), dict(keywords)))
        self.motions.append(("move", label))
        self._left = self._left or label.startswith("push_")
        answer = self.answers.get(label)
        if answer == "stop":
            self.status = _PROTECTIVE
            return _refused(MotionStatus.CONTROLLER_REJECTED, "Driver reported moveL() failure")
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, MotionResult):
            return answer
        if label == "push_push" and self.camera is not None:
            self.camera.push(self._tcp.position_mm, pose.position_mm)
            self._pushed = True
        self._tcp = pose
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)

    def move_to_joints(self, joints: Any, **keywords: Any) -> MotionResult:
        if self._left and self.back_refused is not None:
            self.motions.append(("joints", joints_key(joints)))
            if self.back_stops:
                self.status = _PROTECTIVE
            if isinstance(self.back_refused, BaseException):
                raise self.back_refused
            return self.back_refused
        return super().move_to_joints(joints, **keywords)

    def push_motions(self) -> list[str]:
        return [label for label, _position, _keywords in self.pushes]


class _OrbitingPushingArm(_PushingArm):
    """The pushing arm, which also screens and drives the one view a wrist pick generates, as the owner's UR does
    (``nearest_configuration``, ``move_to_joints_on_the_line``): the base joint turned as far round the part as the tool
    stands from the +x look, the rest of the +x look. Its planner verb, ``move_to_joints``, may run to a declared look
    only: a configuration the pick made up is never driven through the planner's detour (the owner, 2026-09-29), and
    asking fails the test. Every straight joint line is written down (``("line", key)``); ``back_line`` is what one
    answers once a push moved the arm."""

    def __init__(self, test: unittest.TestCase, *, answers: dict[str, Any] | None = None) -> None:
        super().__init__(test, answers=answers)
        self.home_joint_positions = tuple(LOOK_PLUS_X.tolist())
        self.declared = set(self.table)
        self.back_line: MotionResult | None = None

    def nearest_configuration(self, pose: Pose) -> JointPositions:
        joints = JointPositions.deg(_turned(pose), *LOOK_PLUS_X.degrees()[1:])
        self.table[joints_key(joints)] = pose
        return joints

    def move_to_joints_on_the_line(self, joints: JointPositions) -> MotionResult:
        key = joints_key(joints)
        self.motions.append(("line", key))
        if self._left and self.back_line is not None:
            return self.back_line
        self._joints = joints
        self._tcp = self.table[key]
        return MotionResult.executed(MotionCommand.MOVE_JOINTS, target_joints=joints,
                                     message="move_to_joints_on_the_line")

    def move_to_joints(self, joints: Any, **keywords: Any) -> MotionResult:
        key = joints_key(joints)
        if key not in self.declared:
            self.test.fail(f"the planner was asked for {key}: a configuration the pick made up runs on the straight "
                           "joint line only")
        return super().move_to_joints(joints, **keywords)

    def lines(self) -> list[tuple[float, ...]]:
        return [key for kind, key in self.motions if kind == "line"]


class _Jammed:
    """Every candidate of the part collides while it stands where it stood (``ALL_COLLIDED`` with ``RESCAN_RECOMMENDED``,
    as both calculators report a collision), unless ``freed_on`` names the frame; once pushed, or on a frame
    ``freed_on`` names, a valid grasp at its centre, straight down and closing along base x, and on a frame
    ``uncertain_on`` names the same grasp with ``RESCAN_RECOMMENDED``. ``stuck`` keeps it jammed after a push too.
    Another label gets no candidate. ``on_frame`` is called with each frame number as it is ranked."""

    render_debug_images = False

    def __init__(self, camera: _Bench, *, stuck: bool = False, freed_on: tuple[int, ...] = (),
                 uncertain_on: tuple[int, ...] = (), on_frame: Any = None) -> None:
        self.camera = camera
        self.stuck = stuck
        self.freed_on = freed_on
        self.uncertain_on = uncertain_on
        self.on_frame = on_frame
        self.calls: list[tuple[int, str, dict[str, Any]]] = []

    def compute_result(self, seg: Any, *_args: Any, **kwargs: Any) -> GraspResult:
        number = len(self.camera.taken) - 1
        label = str(getattr(seg, "label", ""))
        self.calls.append((number, label, dict(kwargs)))
        if self.on_frame is not None:
            self.on_frame(number)
        uncertain = number in self.uncertain_on
        if label == "part" and (uncertain or number in self.freed_on or (self.camera.pushed and not self.stuck)):
            centre = self._centre(seg)
            grasp = GraspPoint(position=centre, approach=np.array([0.0, 0.0, -1.0]), axis=np.array([1.0, 0.0, 0.0]),
                               grip_width_mm=40.0, score=0.9, frame=GraspFrame.BASE, label=label)
            return GraspResult(candidates=(grasp,), top_score=0.9,
                               reasons=(GraspFailureReason.RESCAN_RECOMMENDED,) if uncertain else ())
        if label == "part":
            return GraspResult(reasons=(GraspFailureReason.ALL_COLLIDED, GraspFailureReason.RESCAN_RECOMMENDED))
        return GraspResult(reasons=(GraspFailureReason.NO_CANDIDATES_GENERATED, GraspFailureReason.RESCAN_RECOMMENDED))

    def _centre(self, seg: Any) -> np.ndarray:
        parts = [box for box in self.camera.boxes if box.label == "part"]
        return (np.asarray(parts[0].low) + np.asarray(parts[0].high)) / 2.0 if len(parts) == 1 else \
            self.camera.centre_of("part")

    def frames_asked_about(self, label: str) -> list[int]:
        return sorted({number for number, named, _ in self.calls if named == label})


class _PushCell:
    """The service of the owner's dense_clutter cell on the pushing arm, the wrist D415, the jammed calculator, the
    recording policy and the toggle hand, with recovery armed: ``allowed`` actions, two at most, the fixture declared.

    ``arm`` is the arm double's class; ``do0_high`` leaves tool DO0 HIGH with the jaws open at the connect, as the owner
    leaves it after opening them at the pendant; ``push_distance_mm`` is the config's push; ``container`` a declared
    container's interior, which bounds a push in place of the table the looks saw."""

    def __init__(self, test: unittest.TestCase, *, mode: GraspMode = GraspMode.DENSE_CLUTTER,
                 allowed: tuple[str, ...] = ("rescan", "next_target", "nudge_target"), boxes: tuple[Box, ...] = (
                     CUBE, POST), answers: dict[str, Any] | None = None, fixture: Any = FIXTURE,
                 approach_validation: bool = False, max_actions: int = 2, max_attempts: int = 3,
                 arm: Any = None, do0_high: bool = False, push_distance_mm: float = 30.0,
                 container: AxisBox | None = None, **calculator: Any) -> None:
        self.events: list[Any] = []
        self.jaws = _toggle(self.events, Person("open"))
        if do0_high:
            self.jaws._io.do[0] = True  # noqa: SLF001 (opened at the pendant: DO0 left HIGH with the jaws open)
        self.jaws.connect()
        self.nobody = Person()
        self.jaws._ask = self.nobody  # noqa: SLF001 (any question from here on fails the test)
        self.connected_writes = len(self.events)
        self.arm = (arm or _PushingArm)(test, answers=answers)
        self.camera = _Bench(self.arm, boxes=boxes)
        self.arm.camera = self.camera
        self.calculator = _Jammed(self.camera, **calculator)
        self.policy = _Policy(self.arm)
        self.policy.gripper = self.jaws  # type: ignore[attr-defined]
        self.service = AutonomousGraspService.from_components(
            arm=self.arm, calculator=self.calculator, perception=self.camera,  # type: ignore[arg-type]
            mode=mode, policy=self.policy, frame_resolver=EyeInHandFrameResolver(t_cam_to_tool=camera_to_tool()),
            max_attempts=max_attempts, gripper=self.jaws)  # type: ignore[arg-type]
        self.orchestrator = self.service.runtime.orchestrator
        self.orchestrator.gripper_model = HAND_E
        self.orchestrator.primary_camera_id = "wrist"
        self.orchestrator.support_config = GraspingSupportConfig(height_mm=0.0, refine_from_target=False)
        self.orchestrator.target_label = "part"
        self.service.effective_config = EffectiveGraspingConfig(
            default_mode=mode, max_attempts=max_attempts, approach_validation_enabled=approach_validation,
            recovery_orchestrator=EffectiveRecoveryOrchestratorConfig(
                enabled=True, allowed_actions=allowed, max_actions=max_actions, fixture=fixture,
                push_distance_mm=push_distance_mm))
        self.service.push_cell = PushCell(hand=PUSH_HAND, workspace=WORKSPACE, hand_clearance_mm=25.0,
                                          container_interior=container)

    def campaign(self, *, runs: int = 1, **keywords: Any) -> Any:
        return PickRun.from_service(self.service, runs=runs, recording=Recording.off(), look=list(LOOKS),
                                    **keywords).execute()

    def joint_moves(self) -> list[tuple[float, ...]]:
        return [key for kind, key in self.arm.motions if kind == "joints"]

    def assert_the_jaws_untouched(self, test: unittest.TestCase) -> None:
        """No write to tool DO0 since the connect, changed or not (on a DO0 standing HIGH a stray LOW moves the jaws),
        the count unchanged and still vouched for, and nobody asked."""
        writes = [event for event in self.events[self.connected_writes:]
                  if isinstance(event, tuple) and event[:2] == ("DO", 0)]
        test.assertEqual([], writes, "DO0 was written")
        test.assertEqual([], _changes(self.events[self.connected_writes:]), "DO0 changed")
        test.assertEqual(0, self.jaws.commands_sent, "the program's count of DO0 changes moved")
        test.assertFalse(self.jaws.jaws_closed)
        test.assertFalse(self.jaws.edge_unknown, "the count no longer stands")
        test.assertEqual("", self.jaws.why_jaws_unknown(), "nobody can say where the jaws stand")
        test.assertEqual([], self.nobody.asked, "a person was asked")

    def logged_world(self) -> "_LoggingWorld":
        """Swap the arm's live world for one that writes its offers and forgets onto the arm's motion log."""
        world = _LoggingWorld(self.arm.motions)
        self.arm.live_planner_world = world
        return world


class _LoggingWorld(_World):
    """The live planner world, writing each offer (``("offer", n points)``) and each forget (``("forget",)``) onto the
    log the arm writes its motions onto, so what the world held at each motion is readable in order."""

    def __init__(self, log: list[Any]) -> None:
        super().__init__()
        self.log = log

    def offer_segmentation(self, **offered: Any) -> None:
        super().offer_segmentation(**offered)
        points = offered.get("target_points_base_mm")
        self.log.append(("offer", 0 if points is None else len(points)))

    def forget_segmentation(self) -> None:
        self.log.append(("forget",))


class _SaysNothingOfItsJaws:
    """A connected parallel jaw that neither counts its changes nor measures its width: it keeps no count a push could
    find unknown, and it never says its jaws stand open."""

    is_connected = True
    min_width_mm = 0.0
    max_width_mm = 50.0

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def activate(self) -> None:
        return None

    def set_width_mm(self, width_mm: float, *, speed: float | None = None, force: float | None = None) -> None:
        raise AssertionError("the jaws were commanded")

    def get_width_mm(self) -> float:
        return self.max_width_mm


class APushedPartIsPickedTests(unittest.TestCase):
    def test_a_boxed_in_part_is_pushed_once_and_then_picked(self) -> None:
        cell = _PushCell(self)

        with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
            run = cell.campaign()

        (attempt,) = run.attempts
        self.assertTrue(attempt.passed, run.render())
        # The push: P0 a plain move, like a grasp approach; then the four judged lines at the owner's speeds.
        self.assertEqual(PUSH_LABELS, cell.arm.push_motions())
        (p0, *legs) = cell.arm.pushes
        self.assertEqual({}, p0[2], "P0 is the arm's plain move, as a grasp approach")
        self.assertEqual([{"linear": True, "vel": 0.05, "acc": 0.1}, {"linear": True, "vel": 0.025, "acc": 0.1},
                          {"linear": True, "vel": 0.05, "acc": 0.1}, {"linear": True, "vel": 0.05, "acc": 0.1}],
                         [keywords for _label, _position, keywords in legs])
        # The looks, once each; then back to the look the arm stood at, and that look taken again.
        self.assertEqual(_keys(*LOOKS, LOOK_PLUS_Y), cell.joint_moves())
        self.assertEqual(4, len(cell.camera.taken))
        self.assertEqual(joints_key(cell.arm.get_joint_positions()), _keys(LOOK_PLUS_Y)[0])
        # Picked from the look taken after the push, at the part where it was pushed.
        (grasp,) = cell.policy.executed
        np.testing.assert_allclose(grasp.position, cell.camera.centre_of("part"))
        self.assertGreater(float(np.linalg.norm(cell.camera.centre_of("part")[:2] - np.array([0.0, -700.0]))), 25.0)
        self.assertEqual(_keys(LOOK_PLUS_Y), [joints_key(j) for j in cell.policy.joints_at])
        self.assertEqual([3], cell.calculator.frames_asked_about("part")[-1:], "the grasp was not ranked after the push")
        # What the earlier looks saw of the part where it stood left the views the look after the push was fused with.
        cloud = cell.service.looked_around.judged.target_cloud_base_mm
        pushed_box = next(box for box in cell.camera.boxes if box.label == "part")
        stale = CUBE.holds(cloud, margin_mm=1.0) & ~pushed_box.holds(cloud, margin_mm=3.0)
        self.assertEqual(0, int(stale.sum()), "the look after the push was fused with the part where it stood")
        cell.assert_the_jaws_untouched(self)
        # What the pick says it did.
        report = run.last
        self.assertIs(AutonomousGraspOutcome.SUCCEEDED, report.outcome)
        self.assertFalse(report.needs_person)
        (push,) = report.telemetry["pushes"]
        self.assertEqual(("all_collided", "pushed", True), (push["trigger"], push["code"], push["motion_started"]))
        self.assertTrue(push["looked_again"].endswith("after the push"))
        self.assertEqual(["push", "executed"], [a.action for a in report.pick_report.attempts])
        self.assertEqual("pushed", report.pick_report.attempts[0].push)
        self.assertEqual([("nudge_target", "recovered_success", True)],
                         [(row["action"], row["outcome"], row["executed"]) for row in report.recovery_actions])
        self.assertIn("pushed the part 30 mm", report.render())
        # Counted once, by the plan's own centre: the one the check was given.
        (record,) = cell.service.campaign.budgets.records
        self.assertEqual(tuple(push["part_centre_mm"]), tuple(round(v, 2) for v in record.centre_mm))
        self.assertGreater(record.push_length_mm, 29.0)

    def test_the_part_is_kept_out_of_the_world_along_its_push_and_after_it(self) -> None:
        """The pick offers its target to the planner world; the push holds the part and its swept path out while it
        moves; the scope's close forgets every offer, so the swept part is offered again for the move back; and the
        re-pick's target, the pushed part, is offered with where it stood and its path, since the frames the world holds
        from before the push still show it there."""
        cell = _PushCell(self)

        cell.campaign()

        offers = [offer["target_points_base_mm"] for offer in cell.arm.live_planner_world.offers]
        self.assertEqual(4, len(offers), "the target, the push's hold, the move back's, the re-pick's")
        first, held, back, repick = offers
        direction = np.asarray(cell.service.looked_around.judged.result.best.position[:2]) - np.array([0.0, -700.0])
        direction /= np.linalg.norm(direction)
        reach = [float(np.max(points[:, :2] @ direction)) for points in (first, held, back, repick)]
        self.assertGreater(reach[1], reach[0] + 25.0, "the push did not keep the part's path out of the world")
        self.assertAlmostEqual(reach[1], reach[2], places=6)
        self.assertGreaterEqual(reach[3], reach[2] - 1e-6)
        self.assertLess(float(np.min(repick[:, :2] @ direction)), float(np.min(first[:, :2] @ direction)) + 1.0,
                        "the re-pick's offer left out where the part stood")
        self.assertEqual(["hold", "forget"], cell.arm.live_planner_world.calls)

    def test_a_part_beside_where_a_pushed_part_landed_is_not_taken_for_it(self) -> None:
        """Only the pushed part is offered with where it stood and its path: a target whose footprint holds the landing
        the push predicted. A small part of the label a finger's width off that landing is another part, and offering
        it with the pushed part's path would take the pushed part itself out of the world while the arm picks beside
        it."""
        from types import SimpleNamespace

        cell = _PushCell(self)
        loop = cell.orchestrator
        stood = _box_points((-20.0, -720.0, 0.0), (20.0, -680.0, 40.0))
        landed = stood + np.array([30.0, 0.0, 0.0])
        loop._pushed_parts = [(np.vstack([stood, landed]), (30.0, -700.0))]  # noqa: SLF001

        self.assertIsNotNone(loop._swept_by_a_push(SimpleNamespace(target_points_base_mm=landed)),  # noqa: SLF001
                             "the pushed part, where it landed, was not taken for itself")
        beside = _box_points((47.0, -708.0, 0.0), (63.0, -692.0, 16.0))  # 16 mm across, 25 mm off the landing
        self.assertIsNone(loop._swept_by_a_push(SimpleNamespace(target_points_base_mm=beside)),  # noqa: SLF001
                          "a part beside the landing was taken for the pushed part")


class APushFromTheViewThePickGeneratedTests(unittest.TestCase):
    """Every declared look ranks the part ``ALL_COLLIDED``, so the looks end at the one view the pick generates (the
    owner, 2026-09-29), turned to the side no look faced, and the push starts there. The arm comes back to that view on
    the straight joint line only: a configuration the pick made up is never driven through the planner's detour, after
    a push as before it, and a line that does not run leaves the arm where it stopped, a person to decide."""

    def test_the_arm_comes_back_to_the_generated_view_on_the_straight_joint_line_only(self) -> None:
        cell = _PushCell(self, arm=_OrbitingPushingArm)

        with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
            report = cell.service.pick(look=list(LOOKS))

        self.assertIs(AutonomousGraspOutcome.SUCCEEDED, report.outcome, report.render())
        (there, back) = cell.arm.lines()
        self.assertEqual(there, back, "not back to the view the push started from")
        self.assertEqual(_keys(*LOOKS), cell.joint_moves(), "the planner's verb ran somewhere the pick made up")
        order = [entry[1] if entry[0] == "move" else "line" for entry in cell.arm.motions if entry[0] in ("line", "move")]
        self.assertEqual(["line", *PUSH_LABELS, "line"], order)
        (push,) = report.telemetry["pushes"]
        self.assertEqual("pushed", push["code"])
        self.assertTrue(push["looked_again"].endswith("after the push"))
        cell.assert_the_jaws_untouched(self)

    def test_a_straight_joint_line_back_that_does_not_run_stops_where_the_arm_stands(self) -> None:
        cell = _PushCell(self, arm=_OrbitingPushingArm)
        cell.arm.back_line = MotionResult.failed(
            MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_JOINTS,
            message="the planner refused this straight joint line: it passes 4 mm from a held frame's box")

        run = cell.campaign(runs=2)

        self.assertEqual([PickOutcome.RAISED, PickOutcome.CANCELLED], [a.outcome for a in run.attempts], run.render())
        report = run.last
        self.assertIs(AutonomousGraspOutcome.UNSAFE_RECOVERY_REFUSED, report.outcome)
        self.assertTrue(report.needs_person)
        self.assertIn("straight joint line", report.telemetry["push_stopped"])
        self.assertEqual(PUSH_LABELS, cell.arm.push_motions())
        self.assertEqual(2, len(cell.arm.lines()), "a line was tried again")
        self.assertEqual(_keys(*LOOKS), cell.joint_moves(), "the planner went round the refused line")
        self.assertEqual(4, len(cell.camera.taken), "a further pick looked again")
        self.assertEqual([], cell.policy.executed)
        cell.assert_the_jaws_untouched(self)


class AControllerFoundStoppedBeforeThePushTests(unittest.TestCase):
    """The push reads the controller before its first motion. Found stopped (a protective stop while the arm stood at
    the last look) or unreadable, nothing is pushed, and the pick ends there as a stopped controller ends it anywhere
    (``controller_not_operational``): the recovery has nothing to act on, no look is driven again on a controller that
    cannot move, and the campaign stops."""

    def test_a_controller_found_stopped_before_the_push_ends_the_pick_and_the_campaign(self) -> None:
        for kind, status in (("protective_stop", _PROTECTIVE), ("unreadable", RobotError("the dashboard did not answer"))):
            with self.subTest(controller=kind):
                holder: dict[str, Any] = {}

                def stop(number: int, status: Any = status) -> None:
                    # The controller stops while the arm stands at the last look, as its grasp is ranked.
                    if number == 2 and not holder.get("done"):
                        holder["done"] = True
                        holder["cell"].arm.status = status

                cell = _PushCell(self, on_frame=stop)
                holder["cell"] = cell

                run = cell.campaign(runs=2)

                self.assertEqual([], cell.arm.pushes, "a push moved on a controller that cannot move")
                self.assertEqual(_keys(*LOOKS), cell.joint_moves(), "a look was driven on a controller that cannot move")
                self.assertEqual(3, len(cell.camera.taken))
                self.assertEqual([PickOutcome.RAISED, PickOutcome.CANCELLED], [a.outcome for a in run.attempts],
                                 run.render())
                self.assertIn("the controller cannot move", run.attempts[0].detail)
                report = run.last
                self.assertTrue(report.controller_stopped)
                self.assertIs(AutonomousGraspOutcome.CANCELLED, report.outcome)
                self.assertFalse(report.needs_person)
                last = report.pick_report.attempts[-1]
                self.assertEqual(("push", "refused_controller_stopped", (GraspFailureReason.CONTROLLER_NOT_OPERATIONAL,)),
                                 (last.action, last.push, last.reasons))
                self.assertEqual([], report.telemetry["recovery_trail_actions"])
                cell.assert_the_jaws_untouched(self)


class AJawCountNobodyVouchesForBeforeThePushTests(unittest.TestCase):
    """The push reads the toggle's count before its first motion (the owner's Hand-E, single_toggle on tool DO0). Found
    unknown, nobody can say where the jaws stand: DO0 switched at the pendant while the arm stood at the last look, or
    its level unreadable then. Nothing is pushed, and the pick ends there as a gripper fault, as the toggle's own refusal
    ends a pick (the owner, 2026-09-30): nothing more moves (no next_target, no look driven again), nobody is asked, DO0
    is not written, the report says why (``gripper_fault``), and the campaign stops on it. A count that says closed
    stays a plain refusal the pick falls through on (:meth:`WhatNeverPushesTests.test_a_closed_jaw_count_never_pushes_
    and_never_asks`)."""

    @staticmethod
    def _spoiled_at_the_last_look(test: unittest.TestCase, kind: str) -> _PushCell:
        """The cell, whose count nobody can vouch for from the moment the last look is ranked: ``switched`` flips DO0
        as the pendant would, ``unreadable`` makes every read of its level fail. ``cell.spoiled_at`` is where the log
        stood then."""
        holder: dict[str, Any] = {}

        def spoil(number: int) -> None:
            if number != 2 or holder.get("done"):
                return
            holder["done"] = True
            cell = holder["cell"]
            io = cell.jaws._io  # noqa: SLF001 (the tool I/O the hand switches)
            if kind == "switched":
                io.do[0] = not io.do.get(0, False)
            else:
                def unreadable(pin: int, **_keywords: Any) -> bool:
                    raise RobotError(f"the controller did not answer the read of tool output {pin}")

                io.get_digital_output = unreadable
            cell.spoiled_at = len(cell.events)

        cell = _PushCell(test, on_frame=spoil)
        holder["cell"] = cell
        return cell

    @staticmethod
    def _said(kind: str) -> str:
        return "somebody switched it" if kind == "switched" else "could not be read"

    def test_a_count_nobody_vouches_for_ends_the_pick_as_a_gripper_fault_and_the_campaign(self) -> None:
        from src.robot.grasping.loop.pick_loop import PickOutcome as LoopOutcome

        for kind in ("switched", "unreadable"):
            with self.subTest(count=kind):
                cell = self._spoiled_at_the_last_look(self, kind)

                with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
                    run = cell.campaign(runs=2)

                self.assertEqual([], cell.arm.pushes, "a push moved on jaws nobody vouched open")
                self.assertEqual(_keys(*LOOKS), cell.joint_moves(), "the arm moved after the push found the count unknown")
                self.assertEqual(3, len(cell.camera.taken))
                self.assertEqual([PickOutcome.RAISED, PickOutcome.CANCELLED], [a.outcome for a in run.attempts],
                                 run.render())
                self.assertIn("the gripper needs a person, so the campaign stops", run.attempts[0].detail)
                self.assertIn(self._said(kind), run.attempts[0].detail)
                report = run.last
                self.assertIs(AutonomousGraspOutcome.EXECUTION_FAILED, report.outcome)
                self.assertIs(LoopOutcome.GRIPPER_FAULT, report.pick_report.outcome)
                self.assertIn("nobody can say where the jaws stand", report.gripper_fault.lower())
                self.assertIn(self._said(kind), report.gripper_fault)
                self.assertFalse(report.needs_person)
                self.assertFalse(report.controller_stopped)
                last = report.pick_report.attempts[-1]
                self.assertEqual(("push", "refused_jaws_unknown", ()), (last.action, last.push, last.reasons))
                (push,) = report.telemetry["pushes"]
                self.assertEqual(("refused_jaws_unknown", False), (push["code"], push["arm_moved"]))
                self.assertEqual([], report.telemetry["recovery_trail_actions"], "a recovery ran after the fault")
                self.assertEqual((), report.recovery_actions)
                self.assertIn("gripper    needs a person:", report.render())
                self.assertEqual([], cell.nobody.asked, "a person was asked")
                self.assertEqual([], [event for event in cell.events[cell.spoiled_at:]  # type: ignore[attr-defined]
                                      if isinstance(event, tuple) and event[:2] == ("DO", 0)], "DO0 was written")
                self.assertEqual(0, cell.jaws.commands_sent)
                self.assertEqual([], cell.policy.executed)

    def test_a_console_run_ends_on_it_asking_nobody(self) -> None:
        from api.runs import RunState
        from tests.test_a_console_run_stops_and_looks_as_a_campaign_does import _drive

        for kind in ("switched", "unreadable"):
            with self.subTest(count=kind):
                cell = self._spoiled_at_the_last_look(self, kind)
                cell.service.configured_looks = LOOKS

                with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
                    run, hub = _drive(cell.service, picks=2)

                self.assertIs(RunState.FAILED, run.state)
                self.assertEqual(1, run.attempted, "the run picked again on a hand nobody vouches for")
                self.assertIn("the gripper needs a person, so the run stops", run.error)
                self.assertIn(self._said(kind), run.error)
                self.assertEqual([], cell.arm.pushes)
                self.assertEqual(_keys(*LOOKS), cell.joint_moves(), "the arm moved after the push found the count unknown")
                self.assertEqual([], cell.nobody.asked, "a person was asked")
                self.assertEqual(0, cell.jaws.commands_sent)
                (said,) = [event for event in hub.since(run.id, 0)[0]
                           if event.type == "pick.attempt_finished" and event.data.get("action") == "push"]
                self.assertEqual("refused_jaws_unknown", said.data["push"])
                self.assertIn("Nobody can say where the jaws stand", said.human)


class TheServiceStartsNoPickAfterAPushStoppedTests(unittest.TestCase):
    """A push that stopped where the arm stands needs a person. A program that calls ``pick()`` itself, as ``PickRun``
    and the console do not, gets no further pick from the service until a person decided (B1's handover: no further
    pick and no further motion after a stopped push): nothing is perceived, asked or commanded meanwhile."""

    def test_a_bare_pick_after_a_push_that_stopped_commands_nothing_until_a_person_decides(self) -> None:
        cell = _PushCell(self, answers={"push_back": _refused(MotionStatus.WORKSPACE_REJECTED, "sample 3 leaves the box")})
        first = cell.service.pick(look=list(LOOKS))
        self.assertTrue(first.needs_person)
        motions, frames = list(cell.arm.motions), len(cell.camera.taken)

        second = cell.service.pick(look=list(LOOKS))

        self.assertTrue(second.needs_person)
        self.assertIs(AutonomousGraspOutcome.UNSAFE_RECOVERY_REFUSED, second.outcome)
        self.assertIsNone(second.pick_report)
        self.assertEqual(motions, cell.arm.motions, "the arm left where the push stopped it")
        self.assertEqual(frames, len(cell.camera.taken), "a further pick looked")
        self.assertIn("The back motion did not finish", cell.service.stopped_where_the_arm_stands)
        self.assertIn("stopped where the arm stands", second.failure_summary())
        cell.assert_the_jaws_untouched(self)
        # A person decided: the next pick starts.
        cell.service.acknowledge_needs_person()
        self.assertEqual("", cell.service.stopped_where_the_arm_stands)
        cell.service.pick(look=list(LOOKS))
        self.assertGreater(len(cell.arm.motions), len(motions))

    def test_a_new_campaign_is_a_person_s_decision_as_well(self) -> None:
        cell = _PushCell(self, answers={"push_back": _refused(MotionStatus.WORKSPACE_REJECTED, "sample 3 leaves the box")})
        cell.service.pick(look=list(LOOKS))
        self.assertTrue(cell.service.stopped_where_the_arm_stands)

        run = cell.campaign()

        self.assertEqual("", cell.service.stopped_where_the_arm_stands)
        self.assertTrue(run.attempts[0].passed, run.render())


class ABlockedApproachTriggersThePushWhereApproachValidationRunsTests(unittest.TestCase):
    """Every approach sweep of the part's grasp blocked (``approach_path_blocked``, nothing moved) is a trigger of the
    push too, but only where approach validation runs (owner, 2026-09-29), under the same neighbour evidence."""

    def _blocked_once(self, cell: _PushCell) -> list[Any]:
        from src.robot.grasping.motion.execution_policy import PolicyOutcome, PolicyReport

        executed = cell.orchestrator._execute  # noqa: SLF001 (the approach check's answer, scripted)
        answers: list[Any] = []

        def execute(result: Any) -> Any:
            if not answers:
                answers.append("blocked")
                return PolicyReport(outcome=PolicyOutcome.APPROACH_PATH_BLOCKED)
            answers.append("executed")
            return executed(result)

        cell.orchestrator._execute = execute  # type: ignore[method-assign]
        return answers

    def test_a_blocked_approach_pushes_the_part_and_picks_it_where_validation_runs(self) -> None:
        # The grasp is uncertain at the first two looks, so all three are fused, and safe at the third.
        cell = _PushCell(self, approach_validation=True, uncertain_on=(0, 1), freed_on=(2,))
        answers = self._blocked_once(cell)

        run = cell.campaign()

        self.assertTrue(run.attempts[0].passed, run.render())
        self.assertEqual(["blocked", "executed"], answers)
        self.assertEqual(PUSH_LABELS, cell.arm.push_motions())
        (push,) = run.last.telemetry["pushes"]
        self.assertEqual(("approach_path_blocked", "pushed"), (push["trigger"], push["code"]))
        self.assertEqual(_keys(*LOOKS, LOOK_PLUS_Y), cell.joint_moves(), "not back to the look it stood at")
        cell.assert_the_jaws_untouched(self)

    def test_a_blocked_approach_never_pushes_where_validation_does_not_run(self) -> None:
        cell = _PushCell(self, approach_validation=False, uncertain_on=(0, 1), freed_on=(2,))
        answers = self._blocked_once(cell)

        run = cell.campaign()

        self.assertEqual(["blocked"], answers)
        self.assertEqual([], cell.arm.pushes)
        self.assertIn("approach_path_blocked", str(run.last.telemetry["low_level_outcome"]).lower())


class WhatNeverPushesTests(unittest.TestCase):
    def test_auto_never_pushes(self) -> None:
        cell = _PushCell(self, mode=GraspMode.AUTO, allowed=("rescan", "nudge_target"))

        run = cell.campaign()

        self.assertFalse(run.attempts[0].passed)
        self.assertEqual([], cell.arm.pushes, "auto pushed")
        self.assertEqual((), cell.service.runtime.orchestrator.pushes)
        self.assertEqual(_keys(*LOOKS), cell.joint_moves(), "a rescan drove the looks again")
        cell.assert_the_jaws_untouched(self)

    def test_a_closed_jaw_count_never_pushes_and_never_asks(self) -> None:
        """The count is read, never switched and never asked about: a count that says closed (the one change a close
        sends, made as the last look is ranked), and the push is refused before anything moves. The pick goes on to its
        next action, which asks nobody either. A count nobody can vouch for ends the pick instead
        (:class:`AJawCountNobodyVouchesForBeforeThePushTests`)."""
        holder: dict[str, Any] = {}

        def spoil(number: int) -> None:
            cell = holder["cell"]
            if number != 2 or holder.get("done"):
                return
            holder["done"] = True
            cell.jaws.set_closed(True)
            holder["writes"] = len(cell.events)

        cell = _PushCell(self, allowed=("rescan", "nudge_target"), on_frame=spoil)
        holder["cell"] = cell

        with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
            report = cell.service.pick(look=list(LOOKS))

        self.assertEqual([], cell.arm.pushes, "the arm moved for a push on jaws the count says closed")
        self.assertEqual([], cell.nobody.asked, "a person was asked")
        self.assertEqual(holder["writes"], len(cell.events), "the push wrote DO0")
        (push,) = cell.service.runtime.orchestrator.pushes
        self.assertEqual("refused_jaws_closed", push.code)
        self.assertFalse(push.motion_started)
        self.assertEqual([], cell.policy.executed)
        self.assertIs(AutonomousGraspOutcome.NO_VALID_GRASP, report.outcome)
        self.assertEqual("", report.gripper_fault)
        self.assertFalse(report.needs_person)
        self.assertIn("not pushed (all_collided): refused_jaws_closed", report.render())

    def test_a_gripper_that_cannot_say_where_its_jaws_stand_is_no_push_and_the_pick_goes_on(self) -> None:
        """Only a hand that toggles keeps a count a push can find unknown. A gripper that neither counts its changes nor
        measures its width never says its jaws stand open: no push (``refused_jaws_unknown``), nothing moves for it, and
        the pick goes on to its next action as on any refusal before motion, with no gripper fault."""
        cell = _PushCell(self, allowed=("rescan", "nudge_target"))
        cell.orchestrator.gripper = _SaysNothingOfItsJaws()

        report = cell.service.pick(look=list(LOOKS))

        self.assertEqual([], cell.arm.pushes)
        (push,) = cell.orchestrator.pushes
        self.assertEqual("refused_jaws_unknown", push.code)
        self.assertIn("neither counts its changes nor measures its width", push.sentence)
        self.assertIs(AutonomousGraspOutcome.NO_VALID_GRASP, report.outcome, report.render())
        self.assertEqual("", report.gripper_fault)
        self.assertEqual(_keys(*LOOKS), cell.joint_moves())

    def test_a_part_whose_grasps_no_collision_refused_is_never_pushed(self) -> None:
        """The push answers a neighbour in the way (``ALL_COLLIDED``) and nothing else: no valid grasp, or no candidate
        at all, is never pushed (the owner, 2026-09-29), however close the post stands."""
        for reasons in ((GraspFailureReason.NO_VALID_GRASP,),
                        (GraspFailureReason.NO_CANDIDATES_GENERATED, GraspFailureReason.RESCAN_RECOMMENDED)):
            with self.subTest(reasons=reasons):
                cell = _PushCell(self, allowed=("rescan", "nudge_target"))

                def compute(seg: Any, *args: Any, reasons: Any = reasons, cell: Any = cell, **kwargs: Any) -> GraspResult:
                    cell.calculator.calls.append((len(cell.camera.taken) - 1, str(seg.label), dict(kwargs)))
                    return GraspResult(reasons=reasons)

                cell.calculator.compute_result = compute  # type: ignore[method-assign]

                report = cell.service.pick(look=list(LOOKS))

                self.assertEqual([], cell.arm.pushes)
                self.assertEqual((), cell.orchestrator.pushes, "a push was considered")
                self.assertNotIn("pushes", report.telemetry)
                cell.assert_the_jaws_untouched(self)


class ARefusalBeforeMotionFallsThroughTests(unittest.TestCase):
    def test_a_refusal_before_anything_moved_falls_through_to_rescan_with_no_motion(self) -> None:
        """A pick handed no look rescans where it stands. From its one 45 degree view the part's shadow hides the
        table where it would land, so the planner refuses (the fail-closed reading of the owner's rule), nothing moves,
        and the pick goes on to that rescan: a fresh frame from where the arm stands, and the same refusal."""
        cell = _PushCell(self, allowed=("rescan", "nudge_target"))
        cell.arm.move_to_joints(LOOK_PLUS_X)  # where the arm stands; the pick is handed no look
        cell.arm.motions.clear()
        where = cell.arm.get_tcp_pose()

        report = cell.service.pick()

        self.assertEqual([], cell.arm.pushes, "the arm was asked to move for a push the planner refused")
        self.assertIs(where, cell.arm.get_tcp_pose(), "the arm moved")
        self.assertEqual([], cell.joint_moves(), "the arm was sent somewhere")
        actions = [attempt.action for attempt in report.pick_report.attempts]
        self.assertEqual(["rescan", "rescan", "rescan"], actions, "the pick did not rescan where it stood")
        pushes = cell.service.runtime.orchestrator.pushes
        self.assertEqual("no_free_direction", pushes[0].code)
        self.assertIn("landing_over_unseen_table", pushes[0].sentence)
        self.assertTrue(all(not push.motion_started for push in pushes))
        self.assertFalse(report.needs_person)
        self.assertEqual([], cell.policy.executed)
        cell.assert_the_jaws_untouched(self)

    def test_a_refusal_before_anything_moved_goes_on_to_the_next_action_with_no_motion(self) -> None:
        """A pick handed looks has had its rescans (its looks). Its push refused before anything moved (the campaign's
        pushes spent: no call to the arm at all; P0 refused before anything was sent: the one call, refused) moves
        nothing, and the pick goes on to its next action, which next_target is here: the looks again, for another part
        of the label, and none is left."""
        spent = PushCampaign(distance_mm=30.0, budgets=PushBudgets(per_campaign=0))
        for name, answers, campaign, code in (
                ("budget", {}, spent, "campaign_budget_spent"),
                ("p0", {"push_p0": _refused(MotionStatus.WORKSPACE_REJECTED, "outside")}, None,
                 "refused_approach_not_sent")):
            with self.subTest(refused=name):
                cell = _PushCell(self, answers=answers)
                if campaign is not None:
                    cell.service._campaign = campaign  # noqa: SLF001 (a campaign that has spent its pushes)

                report = cell.service.pick(look=list(LOOKS))

                self.assertEqual([] if name == "budget" else ["push_p0"], cell.arm.push_motions())
                self.assertEqual(_keys(*LOOKS, *LOOKS), cell.joint_moves(), "the arm was sent back or somewhere else")
                self.assertEqual(["next_target"], report.telemetry["recovery_trail_actions"])
                self.assertEqual(code, report.telemetry["pushes"][0]["code"])
                self.assertFalse(report.telemetry["pushes"][0]["motion_started"])
                self.assertFalse(report.needs_person)
                self.assertEqual([], cell.policy.executed)
                cell.assert_the_jaws_untouched(self)

    def test_the_down_leg_refused_before_it_was_sent_sends_the_arm_back_to_the_look_and_falls_through(self) -> None:
        """The arm is in the air at P0 and nothing touched the part (the lead's ruling of 2026-09-29, pending the
        owner): back to the look like a grasp approach, and the pick goes on as on any refusal before motion. Nothing
        needs a person. The round trip commanded motion, so it counts against the budgets (``PushBudgets.record``), by
        where the part still stands: the next pick of the campaign does not drive it again."""
        cell = _PushCell(self, allowed=("rescan", "nudge_target"),
                         answers={"push_down": _refused(MotionStatus.SELF_COLLISION_REJECTED, "rfinger 4 mm off the post")})

        run = cell.campaign(runs=2)

        self.assertEqual(["push_p0", "push_down"], cell.arm.push_motions(), "the part was driven twice")
        self.assertEqual(_keys(*LOOKS, LOOK_PLUS_Y, *LOOKS), cell.joint_moves(), "not back to the look after P0")
        self.assertEqual([PickOutcome.FAILED, PickOutcome.FAILED], [a.outcome for a in run.attempts], run.render())
        pushes = [push["code"] for push in run.last.telemetry["pushes"]]
        self.assertEqual(["part_already_pushed"], pushes)
        self.assertFalse(run.last.needs_person)
        self.assertEqual([], cell.policy.executed)
        (record,) = cell.service.campaign.budgets.records
        self.assertIsNone(record.landing_mm, "nothing touched the part, so it stands where it stood")
        cell.assert_the_jaws_untouched(self)

    def test_a_pick_handed_no_look_drives_p0_once_for_a_part_whose_down_leg_is_refused(self) -> None:
        """A wrist pick handed no look rescans where it stands, attempt after attempt; with a declared container its
        one view can plan a push. The down leg refused before it was sent brings the arm back each time, and the round
        trip counts against the part's budget: one per part, however often the pick rescans."""
        cell = _PushCell(self, allowed=("rescan", "nudge_target"), container=CONTAINER,
                         answers={"push_down": _refused(MotionStatus.SELF_COLLISION_REJECTED, "rfinger 4 mm off the post")})
        cell.arm.move_to_joints(LOOK_PLUS_Y)  # where the arm stands; the pick is handed no look
        cell.arm.motions.clear()

        report = cell.service.pick()

        self.assertEqual(["push_p0", "push_down"], cell.arm.push_motions(), "P0 was driven again for the same part")
        self.assertEqual(_keys(LOOK_PLUS_Y), cell.joint_moves())
        codes = [push["code"] for push in report.telemetry["pushes"]]
        self.assertEqual("refused_down_not_sent", codes[0])
        self.assertIn("part_already_pushed", codes[1:])
        self.assertFalse(report.needs_person)
        cell.assert_the_jaws_untouched(self)

    def test_the_target_is_offered_again_before_the_arm_goes_back_from_p0(self) -> None:
        """The push's keep-out forgets every offer as it closes (the live world's rule): the pick's target goes back
        out of the world before the move back from P0, whose first stretch starts right above the part (B1's caller
        rule)."""
        cell = _PushCell(self, allowed=("rescan", "nudge_target"),
                         answers={"push_down": _refused(MotionStatus.SELF_COLLISION_REJECTED, "rfinger 4 mm off the post")})
        cell.logged_world()

        cell.service.pick(look=list(LOOKS))

        log = cell.arm.motions
        down = log.index(("move", "push_down"))
        back = next(index for index in range(down, len(log)) if log[index][0] == "joints")
        between = [entry[0] for entry in log[down + 1:back]]
        self.assertIn("forget", between, "the push's keep-out did not close before the move back")
        self.assertEqual("offer", between[-1], "the move back ran with the part not offered to the world")

    def test_a_stop_asked_for_while_the_push_is_planned_moves_nothing(self) -> None:
        """The operator's Stop (``should_cancel``) arrives after the looks asked for it the last time, while the push
        is planned: the push is refused before its first motion, and the pick ends ``cancelled`` with nothing more
        moved, as a stop between two attempts ends it."""
        from src.robot.grasping.recovery import push_planner

        cell = _PushCell(self)
        asked = {"stop": False}
        cell.service.set_cancel_check(lambda: asked["stop"])
        plan_push = push_planner.plan_push

        def planned_while_stop_is_pressed(**keywords: Any) -> Any:
            asked["stop"] = True
            return plan_push(**keywords)

        with mock.patch("src.robot.grasping.recovery.push_planner.plan_push", side_effect=planned_while_stop_is_pressed):
            report = cell.service.pick(look=list(LOOKS))

        self.assertEqual([], cell.arm.pushes, "the push began after the Stop")
        self.assertEqual(_keys(*LOOKS), cell.joint_moves(), "the arm moved after the Stop")
        self.assertIs(AutonomousGraspOutcome.CANCELLED, report.outcome)
        self.assertEqual("refused_stop_requested", report.telemetry["pushes"][0]["code"])
        self.assertFalse(report.needs_person)
        cell.assert_the_jaws_untouched(self)


class AStopAfterContactEndsTheCampaignTests(unittest.TestCase):
    """A failure of P0 once it may have been sent, a refusal or a failure of any leg once the down leg was sent, a
    failed move back to the look and a protective stop end the push where the arm is: nothing more is commanded,
    nothing clears a stop, and the campaign ends (B1's handover, the lead's ruling of 2026-09-29): the next pick would
    drive the arm back to its look."""

    def _assert_the_campaign_ended(self, cell: _PushCell, run: Any, *, motions: list[str], joints: int) -> None:
        self.assertEqual(motions, cell.arm.push_motions(), "a motion followed the stop")
        self.assertEqual(joints, len(cell.joint_moves()), "the arm was sent back to a look after the stop")
        self.assertEqual([PickOutcome.RAISED, PickOutcome.CANCELLED], [a.outcome for a in run.attempts], run.render())
        self.assertEqual(3, len(cell.camera.taken), "a further pick looked again")
        self.assertEqual([], cell.policy.executed)
        cell.assert_the_jaws_untouched(self)

    def test_a_failure_of_p0_once_sent_or_of_any_leg_after_the_down_leg_was_sent_stops_there(self) -> None:
        cases = {
            "push_p0": _refused(MotionStatus.CONTROLLER_REJECTED, "Driver reported moveJ() failure"),
            "push_down": RobotError("the line raised"),
            "push_push": _refused(MotionStatus.SELF_COLLISION_REJECTED, "rfinger 4 mm off the post"),
            "push_back": _refused(MotionStatus.WORKSPACE_REJECTED, "sample 3 leaves the box"),
            "push_up": _refused(MotionStatus.CONTROLLER_REJECTED, "Driver reported moveL() failure"),
        }
        for leg, answer in cases.items():
            with self.subTest(leg=leg):
                cell = _PushCell(self, answers={leg: answer})

                run = cell.campaign(runs=2)

                self._assert_the_campaign_ended(cell, run, motions=PUSH_LABELS[:PUSH_LABELS.index(leg) + 1], joints=3)
                report = run.last
                self.assertIs(AutonomousGraspOutcome.UNSAFE_RECOVERY_REFUSED, report.outcome)
                self.assertTrue(report.needs_person)
                self.assertIn("a recovery stopped where the arm stands", run.attempts[0].detail)
                self.assertIn("a person decides", run.attempts[0].detail)
                (push,) = report.telemetry["pushes"]
                self.assertEqual((True, True), (push["stopped"], push["motion_started"]))
                self.assertEqual(("nudge_target", "unsafe_recovery_refused", False),
                                 tuple(report.recovery_actions[-1][key] for key in ("action", "outcome", "executed")))
                self.assertEqual(1, len(cell.service.campaign.budgets.records), "a push that moved was not counted")
                self.assertIn("push       stopped where the arm stands (all_collided): unsafe_recovery_refused",
                              report.render())

    def test_a_move_back_to_the_look_that_fails_after_the_push_stops_there(self) -> None:
        cell = _PushCell(self)
        cell.arm.back_refused = MotionResult.failed(MotionStatus.CONTROLLER_REJECTED, MotionCommand.MOVE_JOINTS,
                                                    message="Driver reported moveJ() failure")

        run = cell.campaign(runs=2)

        self._assert_the_campaign_ended(cell, run, motions=PUSH_LABELS, joints=4)
        self.assertIs(AutonomousGraspOutcome.UNSAFE_RECOVERY_REFUSED, run.last.outcome)
        self.assertIn("the move back to the look", run.last.telemetry["push_stopped"])

    def test_a_protective_stop_mid_push_gets_no_automatic_recovery(self) -> None:
        cell = _PushCell(self, answers={"push_push": "stop"})

        run = cell.campaign(runs=2)

        self._assert_the_campaign_ended(cell, run, motions=PUSH_LABELS[:3], joints=3)
        report = run.last
        self.assertTrue(report.controller_stopped)
        self.assertTrue(report.needs_person)
        self.assertIs(AutonomousGraspOutcome.CANCELLED, report.outcome)
        self.assertIn("the controller cannot move", run.attempts[0].detail)
        self.assertEqual("controller_not_operational", report.pick_report.attempts[-1].push)

    def test_a_protective_stop_on_the_move_back_is_the_controller_s(self) -> None:
        """The move back to the look fails on a controller that protective-stopped under it: the controller is asked
        after the failure, the push says so (``controller_not_operational``), and nothing clears the stop."""
        cell = _PushCell(self)
        cell.arm.back_refused = MotionResult.failed(MotionStatus.CONTROLLER_REJECTED, MotionCommand.MOVE_JOINTS,
                                                    message="Driver reported moveJ() failure")
        cell.arm.back_stops = True

        run = cell.campaign(runs=2)

        self._assert_the_campaign_ended(cell, run, motions=PUSH_LABELS, joints=4)
        report = run.last
        self.assertTrue(report.controller_stopped)
        self.assertTrue(report.needs_person)
        self.assertIs(AutonomousGraspOutcome.CANCELLED, report.outcome)
        (push,) = report.telemetry["pushes"]
        self.assertEqual(("controller_not_operational", True, "back_to_the_look"),
                         (push["code"], push["controller_stopped"], push["leg"]))
        self.assertIn("protective_stop=True", report.telemetry["push_stopped"])

    def test_a_failed_move_back_after_the_down_leg_was_refused_says_nothing_touched_the_part(self) -> None:
        """The down leg refused before it was sent is a fall-through, and the move back from P0 then fails: a stop
        where the arm stands like any other, said as what happened (the arm went to P0, nothing touched the part),
        counted against the budgets, and a row of the record's recovery."""
        cell = _PushCell(self, answers={"push_down": _refused(MotionStatus.SELF_COLLISION_REJECTED, "4 mm off the post")})
        cell.arm.back_refused = MotionResult.failed(MotionStatus.CONTROLLER_REJECTED, MotionCommand.MOVE_JOINTS,
                                                    message="Driver reported moveJ() failure")

        run = cell.campaign(runs=2)

        self._assert_the_campaign_ended(cell, run, motions=["push_p0", "push_down"], joints=4)
        report = run.last
        self.assertIs(AutonomousGraspOutcome.UNSAFE_RECOVERY_REFUSED, report.outcome)
        self.assertTrue(report.needs_person)
        said = report.telemetry["push_stopped"]
        self.assertNotIn("The part was pushed", said)
        self.assertIn("nothing touched it", said)
        (push,) = report.telemetry["pushes"]
        self.assertEqual((True, False, True), (push["arm_moved"], push["motion_started"], push["stopped"]))
        self.assertEqual(("nudge_target", "unsafe_recovery_refused", False),
                         tuple(report.recovery_actions[-1][key] for key in ("action", "outcome", "executed")))
        (record,) = cell.service.campaign.budgets.records
        self.assertIsNone(record.landing_mm)

    def test_a_camera_world_that_cannot_vouch_mid_push_is_a_push_that_stopped_where_the_arm_stands(self) -> None:
        """A camera world that cannot vouch for the cell raises out of the pick as a fault of the cell, as on every
        motion of a pick, during the push or on the move back after it. The pick's report says the push stopped where
        the arm stands (``needs_person``), the push counts against the budgets, nothing more moves, and the next pick
        of the service does not start until a person decides."""
        blind = CameraWorldUnavailable(camera="wrist", verdict="blind", attempts=4,
                                       reason="the wrist camera could not vouch for the cell")
        for where, motions, joints in (("push_down", ["push_p0", "push_down"], 3), ("back", PUSH_LABELS, 4)):
            with self.subTest(raised=where):
                cell = _PushCell(self, answers={"push_down": blind} if where == "push_down" else {})
                if where == "back":
                    cell.arm.back_refused = blind

                report = cell.service.pick(look=list(LOOKS))

                self.assertIs(blind, report.fault)
                self.assertTrue(report.needs_person)
                self.assertIn("could not vouch", report.telemetry["push_stopped"])
                (push,) = report.telemetry["pushes"]
                self.assertEqual((True, True), (push["stopped"], push["arm_moved"]))
                self.assertEqual(1, len(cell.service.campaign.budgets.records), "the push was not counted")
                self.assertEqual(motions, cell.arm.push_motions())
                self.assertEqual(joints, len(cell.joint_moves()))
                # The next pick of the service starts nothing.
                again = cell.service.pick(look=list(LOOKS))
                self.assertTrue(again.needs_person)
                self.assertEqual(motions, cell.arm.push_motions())
                self.assertEqual(joints, len(cell.joint_moves()), "the arm left where the push stopped it")
                cell.assert_the_jaws_untouched(self)

    def test_a_console_run_pushes_the_part_and_picks_it_asking_nobody(self) -> None:
        from api.runs import RunState
        from tests.test_a_console_run_stops_and_looks_as_a_campaign_does import _drive

        cell = _PushCell(self)
        cell.service.configured_looks = LOOKS

        with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
            run, hub = _drive(cell.service, picks=1)

        self.assertIs(RunState.FINISHED, run.state, run.error)
        self.assertEqual(1, run.succeeded)
        self.assertEqual(PUSH_LABELS, cell.arm.push_motions())
        finished = [event.data for event in hub.since(run.id, 0)[0] if event.data.get("action") == "push"]
        self.assertEqual("pushed", finished[0]["push"])
        cell.assert_the_jaws_untouched(self)

    def test_a_console_run_ends_on_a_push_that_stopped(self) -> None:
        from api.runs import RunState
        from tests.test_a_console_run_stops_and_looks_as_a_campaign_does import _drive

        cell = _PushCell(self, answers={"push_back": _refused(MotionStatus.WORKSPACE_REJECTED, "leaves the box")})
        cell.service.configured_looks = LOOKS

        run, hub = _drive(cell.service, picks=2)

        self.assertIs(RunState.FAILED, run.state)
        self.assertEqual(1, run.attempted, "the run picked again after the push stopped")
        self.assertIn("a recovery stopped where the arm stands", run.error)
        self.assertEqual(PUSH_LABELS[:4], cell.arm.push_motions())
        self.assertEqual(3, len(cell.joint_moves()))
        cell.assert_the_jaws_untouched(self)
        # What the operator reads for the push: its own words, where the arm stopped, never "Nothing moved".
        (said,) = [event for event in hub.since(run.id, 0)[0]
                   if event.type == "pick.attempt_finished" and event.data.get("action") == "push"]
        self.assertNotIn("Nothing moved", said.human)
        self.assertIn("The back motion did not finish", said.human)
        self.assertIn("a person decides", said.human)

    def test_a_console_run_says_what_the_push_did_and_carries_its_distance(self) -> None:
        from api.runs import Run
        from tests.test_a_console_run_stops_and_looks_as_a_campaign_does import _registry_and_console

        cell = _PushCell(self)
        cell.service.configured_looks = LOOKS
        registry, console, hub = _registry_and_console(cell.service)
        run = Run(id="run-console", prompt="", requested_picks=1, push_mm=40.0)
        console.active_run_id = run.id

        registry._drive(console, run)

        events = hub.since(run.id, 0)[0]
        (started,) = [event for event in events if event.type == "run_started"]
        self.assertEqual(40.0, started.data["push_mm"])
        (pushed,) = [event for event in events
                     if event.type == "pick.attempt_finished" and event.data.get("action") == "push"]
        self.assertNotIn("Nothing moved", pushed.human)
        self.assertIn("The part was pushed 40 mm", pushed.human)


class TheBudgetsTests(unittest.TestCase):
    """1 push per part, 2 per pick, 5 per campaign (owner, 2026-09-29), counted by the service's campaign across its
    picks. A spent budget means no push, never a stop: the pick goes on without it."""

    def test_one_push_per_part_across_the_picks_of_a_campaign(self) -> None:
        """A part that did not budge is the part that was pushed: the look taken again after the push and the next
        pick both find it where it stood, jammed, and neither pushes it again."""
        cell = _PushCell(self, allowed=("rescan", "nudge_target"), stuck=True)
        cell.camera.stuck = True

        run = cell.campaign(runs=2)

        self.assertEqual(PUSH_LABELS, cell.arm.push_motions(), "the part was pushed twice")
        self.assertEqual([PickOutcome.FAILED, PickOutcome.FAILED], [a.outcome for a in run.attempts], run.render())
        codes = [push["code"] for push in run.last.telemetry["pushes"]]
        self.assertEqual(["part_already_pushed"], codes)
        self.assertEqual(1, cell.service.campaign.budgets.pushes_this_campaign)

    def test_two_pushes_per_pick(self) -> None:
        holder: dict[str, Any] = {}

        def two_already(number: int) -> None:
            # Two other parts of this pick were pushed before this one, far from it.
            if number == 0:
                budgets = holder["cell"].service.campaign.budgets
                for x in (400.0, -400.0):
                    budgets.record(centre_mm=(x, -300.0, 20.0), landing_mm=(x, -270.0, 20.0))

        cell = _PushCell(self, allowed=("rescan", "nudge_target"), on_frame=two_already)
        holder["cell"] = cell

        report = cell.service.pick(look=list(LOOKS))

        self.assertEqual([], cell.arm.pushes)
        self.assertEqual("pick_budget_spent", cell.service.runtime.orchestrator.pushes[0].code)
        self.assertFalse(report.needs_person)
        # The next pick has its own two again.
        cell.camera.taken.clear()
        cell.calculator.on_frame = None
        report = cell.service.pick(look=list(LOOKS))
        self.assertEqual(PUSH_LABELS, cell.arm.push_motions())
        self.assertIs(AutonomousGraspOutcome.SUCCEEDED, report.outcome)

    def test_two_pushes_per_pick_hold_over_every_re_pick_of_one_pick(self) -> None:
        """One pick is one pick for the budgets however often its recovery picks again: two pushes of this pick spent,
        the recovery loop's rescan picks again, and that re-pick pushes nothing either. A pick handed no look rescans
        where it stands, and a declared container lets its one view plan the push the budget refuses."""
        holder: dict[str, Any] = {}

        def two_already(number: int) -> None:
            # Two other parts of this pick were pushed before this one, far from it.
            if number == 0:
                budgets = holder["cell"].service.campaign.budgets
                for x in (400.0, -400.0):
                    budgets.record(centre_mm=(x, -300.0, 20.0), landing_mm=(x, -270.0, 20.0))

        cell = _PushCell(self, allowed=("rescan", "nudge_target"), container=CONTAINER, on_frame=two_already)
        holder["cell"] = cell
        cell.arm.move_to_joints(LOOK_PLUS_Y)  # where the arm stands; the pick is handed no look
        cell.arm.motions.clear()

        report = cell.service.pick()

        self.assertIn("rescan", report.telemetry["recovery_trail_actions"], "the recovery did not pick again")
        self.assertEqual([], cell.arm.pushes, "a re-pick of the pick pushed a third time")
        codes = [push["code"] for push in report.telemetry["pushes"]]
        self.assertGreaterEqual(len(codes), 6, "the re-pick considered no push")
        self.assertEqual({"pick_budget_spent", "refused_no_attempt_left"}, set(codes))
        self.assertEqual(2, len(cell.service.campaign.budgets.records))

    def test_five_pushes_per_campaign_and_a_new_campaign_starts_afresh(self) -> None:
        cell = _PushCell(self, allowed=("rescan", "nudge_target"))
        campaign = cell.service.start_campaign()
        for number in range(5):
            campaign.budgets.record(centre_mm=(400.0 - 150.0 * number, -300.0, 20.0))

        report = cell.service.pick(look=list(LOOKS))

        self.assertEqual([], cell.arm.pushes)
        self.assertEqual("campaign_budget_spent", cell.service.runtime.orchestrator.pushes[0].code)
        self.assertFalse(report.needs_person)
        self.assertIs(AutonomousGraspOutcome.NO_VALID_GRASP, report.outcome)
        # PickRun starts a campaign of its own: its pushes are its own.
        cell.camera.taken.clear()
        run = cell.campaign()
        self.assertTrue(run.attempts[0].passed, run.render())
        self.assertEqual(PUSH_LABELS, cell.arm.push_motions())


class NextTargetTests(unittest.TestCase):
    """``next_target`` is a rescan that skips the part that failed: same label only, this pick and the next two. The
    owner (2026-09-30): it runs the look sequence again for the new part, declared looks with the early stop; a rescan
    of the same part never repeats the looks."""

    #: A second part of the label 80 mm along +x of the first, which no grasp can take in the first pick.
    OTHER = Box((60.0, -720.0, 0.0), (100.0, -680.0, 40.0), "part")

    def test_next_target_runs_the_looks_again_for_another_part_of_the_label(self) -> None:
        cell = _PushCell(self, mode=GraspMode.AUTO, allowed=("next_target",), boxes=(self.OTHER, CUBE))
        # The other part gets a grasp from the second pick's first frame on; the first never does.
        calculator = cell.calculator

        def compute(seg: Any, *args: Any, **kwargs: Any) -> GraspResult:
            number = len(cell.camera.taken) - 1
            centre = _centre_of_mask(seg, cell)
            calculator.calls.append((number, str(seg.label), dict(kwargs)))
            if centre[0] > 40.0 and number >= 3:
                grasp = GraspPoint(position=centre, approach=np.array([0.0, 0.0, -1.0]), axis=np.array([1.0, 0.0, 0.0]),
                                   grip_width_mm=40.0, score=0.9, frame=GraspFrame.BASE, label="part")
                return GraspResult(candidates=(grasp,), top_score=0.9)
            return GraspResult(reasons=(GraspFailureReason.ALL_COLLIDED, GraspFailureReason.RESCAN_RECOMMENDED))

        calculator.compute_result = compute  # type: ignore[method-assign]

        report = cell.service.pick(look=list(LOOKS))

        self.assertIs(AutonomousGraspOutcome.SUCCEEDED, report.outcome, report.render())
        # The first pick drives every look; next_target runs them again and stops at the first safe grasp.
        self.assertEqual(_keys(*LOOKS, LOOK_PLUS_X), cell.joint_moves())
        (grasp,) = cell.policy.executed
        self.assertGreater(float(grasp.position[0]), 40.0, "not the other part of the label")
        self.assertEqual(["next_target"], report.telemetry["recovery_trail_actions"])
        (zone,) = cell.service.campaign.zones.zones("part")
        self.assertLess(float(np.hypot(zone.centre_xy_mm[0], zone.centre_xy_mm[1] + 700.0)), 10.0)
        self.assertEqual([], cell.arm.pushes)

    def test_only_skipped_parts_left_stops_with_the_sentence(self) -> None:
        cell = _PushCell(self, mode=GraspMode.AUTO, allowed=("next_target",))

        run = cell.campaign()

        self.assertEqual(_keys(*LOOKS, *LOOKS), cell.joint_moves(), "next_target did not run the looks again")
        self.assertEqual([0, 1, 2], cell.calculator.frames_asked_about("part"), "a skipped part was ranked again")
        (attempt,) = run.attempts
        self.assertFalse(attempt.passed)
        self.assertIn("is being skipped, so the pick stops here", attempt.detail)
        self.assertEqual([], cell.policy.executed)
        last = run.last.pick_report.attempts[-1]
        self.assertEqual(("exhausted", ()), (last.action, last.reasons))
        self.assertIn("'part'", last.excluded)

    def test_a_skipped_part_is_of_one_label_only(self) -> None:
        """A zone applies only to parts of the label it was made for: a post standing where the skipped part stood is no
        more a target than before, and never picked in its place."""
        cell = _PushCell(self, mode=GraspMode.AUTO, allowed=("next_target",))
        cell.service.start_campaign()
        cell.service.campaign.start_pick()
        cell.service.campaign.zones.exclude(label="post", centre_mm=(0.0, -700.0, 20.0))

        cell.service.pick(look=list(LOOKS))

        self.assertIn(0, cell.calculator.frames_asked_about("part"), "a zone of another label skipped the part")
        self.assertEqual([], cell.calculator.frames_asked_about("post"), "the post became a target")

    def test_a_skipped_part_is_skipped_in_its_pick_and_the_next_two_and_not_after(self) -> None:
        """A zone lives for the pick it was made in and the next two (the owner, 2026-09-29): the part skipped in the
        first pick of a campaign is not ranked in the second or the third, which stop with the zones' sentence, and is
        ranked again in the fourth."""
        cell = _PushCell(self, mode=GraspMode.AUTO, allowed=("next_target",))

        run = cell.campaign(runs=4)

        self.assertEqual(4, len(run.attempts), run.render())
        # Pick 1 ranks it on its three looks and skips it on the looks next_target runs again (frames 3 to 5); picks 2
        # and 3 skip it (6 to 11); pick 4 ranks it again (12 to 14).
        self.assertEqual([0, 1, 2, 12, 13, 14], cell.calculator.frames_asked_about("part"))
        for attempt in run.attempts[1:3]:
            self.assertIn("is being skipped", attempt.detail)

    def test_a_rescan_of_the_same_part_never_repeats_the_looks(self) -> None:
        cell = _PushCell(self, mode=GraspMode.AUTO, allowed=("rescan",))

        report = cell.service.pick(look=list(LOOKS))

        self.assertEqual(_keys(*LOOKS), cell.joint_moves())
        self.assertEqual(3, len(cell.camera.taken))
        self.assertNotIn("rescan", report.telemetry["recovery_trail_actions"])


class ThePushDistanceTests(unittest.TestCase):
    def test_push_mm_51_and_9_are_refused_and_40_is_taken_where_the_ceiling_allows(self) -> None:
        cell = _PushCell(self)
        for asked in (51.0, 9.0):
            with self.subTest(push_mm=asked):
                with self.assertRaises(ValueError) as caught:
                    PickRun.from_service(cell.service, runs=1, recording=Recording.off(), push_mm=asked)
                self.assertIn("50 mm" if asked > 50 else "10 mm", str(caught.exception))
                with self.assertRaises(ValueError):
                    cell.service.push_distance(asked)

        run = cell.campaign(push_mm=40.0)

        self.assertTrue(run.attempts[0].passed, run.render())
        self.assertEqual(40.0, cell.service.campaign.distance_mm)
        (push,) = run.last.telemetry["pushes"]
        self.assertEqual(40.0, push["distance_mm"])

    def test_the_config_s_push_distance_is_the_push_when_nobody_asks(self) -> None:
        """``recovery.fixture.push_distance_mm`` reaches the service through the builders, and a campaign that names no
        distance pushes that far."""
        from src.config.schema.robot import RobotConfig
        from src.robot.grasping.recovery import push_planner
        from tests.test_grasping_config_wiring import _calc_and_perception

        cfg = RobotConfig(vendor="dummy", gripper={"vendor": "none"}, grasping={
            "default_mode": "dense_clutter",
            "recovery": {"enabled": True, "allowed_actions": ["rescan", "nudge_target"], "fixture": {
                "center_mm": [0.0, -700.0, 150.0], "half_extents_mm": [300.0, 300.0, 300.0], "max_nudge_mm": 50.0,
                "push_distance_mm": 20.0}}})
        calculator, perception = _calc_and_perception()
        service = AutonomousGraspService.from_robot_config(cfg, calculator=calculator, perception=perception)

        self.assertEqual(20.0, service.effective_config.recovery_orchestrator.push_distance_mm)
        self.assertEqual(20.0, service.push_distance())
        self.assertEqual(20.0, service.start_campaign().distance_mm)
        self.assertEqual(40.0, service.push_distance(40.0), "a request up to the ceiling is taken as asked")

        # The owner's cell, its config pushing 20 mm, and a campaign that asks for no distance.
        cell = _PushCell(self, push_distance_mm=20.0)
        with mock.patch("src.robot.grasping.recovery.push_planner.plan_push", wraps=push_planner.plan_push) as planned:
            cell.campaign()
        self.assertEqual(20.0, cell.service.campaign.distance_mm)
        self.assertEqual(20.0, planned.call_args.kwargs["push_distance_mm"])

    def test_a_push_above_the_cells_ceiling_is_refused_before_anything_moves(self) -> None:
        cell = _PushCell(self, fixture=(FIXTURE[0], FIXTURE[1], 30.0))

        run = cell.campaign(push_mm=40.0)

        self.assertEqual((), run.attempts)
        self.assertIn("push_mm refused", run.error)
        self.assertIn("the cell allows 30 mm", run.error)
        self.assertEqual(1, run.exit_code)
        self.assertEqual([], cell.arm.motions)
        self.assertEqual(30.0, cell.service.push_distance())


class WhatThePushNeedsBeforeItPlansTests(unittest.TestCase):
    def test_no_neighbour_within_25_mm_is_no_push_and_the_planner_is_not_asked(self) -> None:
        """The trigger's evidence is read off what the looks saw before anything is planned: a post 30 mm off the part
        left the fingers room, so its collisions are not the neighbour's doing, and nothing is pushed."""
        far = Box((-80.0, -722.0, 0.0), (-50.0, -714.0, 30.0), "post")
        cell = _PushCell(self, boxes=(CUBE, far), allowed=("rescan", "nudge_target"))

        with mock.patch("src.robot.grasping.recovery.push_planner.plan_push",
                        side_effect=AssertionError("the planner was asked")):
            cell.service.pick(look=list(LOOKS))

        self.assertEqual([], cell.arm.pushes)
        (push,) = cell.service.runtime.orchestrator.pushes
        self.assertEqual("no_blocking_neighbour", push.code)
        self.assertIn("the nearest is 30 mm away", push.sentence)

    def test_a_push_needs_an_attempt_left_to_pick_the_part_from_after_it(self) -> None:
        cell = _PushCell(self, allowed=("rescan", "nudge_target"), max_attempts=1)

        report = cell.service.pick(look=list(LOOKS))

        self.assertEqual([], cell.arm.pushes)
        (push,) = cell.service.runtime.orchestrator.pushes
        self.assertEqual("refused_no_attempt_left", push.code)
        self.assertIs(AutonomousGraspOutcome.NO_VALID_GRASP, report.outcome)

    def test_the_planner_keeps_the_hand_the_clearance_the_arm_s_line_judge_keeps(self) -> None:
        """The cell's clearance reaches the planner: the camera world's margin_mm plus line_clearance_mm, 25 mm on the
        owner's config. A push whose finger passes 15 mm from a post, inside the box the world grows around it, is one
        the arm's line judge would refuse: at 25 mm it is not planned, where the 10 mm floor alone would plan it."""
        from src.config.loader import load_robot_config, reload_config
        from src.robot.grasping.recovery import push_planner
        from tests.test_a_push_moves_only_when_every_guard_holds import HAND_E as BLOCK_HAND
        from tests.test_a_push_moves_only_when_every_guard_holds import PART, TABLE
        from tests.test_a_push_moves_only_when_every_guard_holds import POST as NEAR_POST
        from tests.test_a_push_moves_only_when_every_guard_holds import WORKSPACE as BLOCK_WORKSPACE

        reload_config()
        self.addCleanup(reload_config)
        owners = PushCell.from_robot_config(load_robot_config(None, profile="ur10,hande"))
        assert isinstance(owners, PushCell), owners
        self.assertEqual(25.0, owners.hand_clearance_mm)
        self.assertEqual(75.0, owners.hand.palm_thickness_mm)

        def plan(clearance: float) -> Any:
            return push_planner.plan_push(
                target_points_mm=PART, neighbour_points_mm=NEAR_POST, support_normal=(0.0, 0.0, 1.0),
                support_offset_mm=0.0, workspace=BLOCK_WORKSPACE, hand=BLOCK_HAND, table_points_mm=TABLE,
                hand_clearance_mm=clearance)

        self.assertIsInstance(plan(10.0), push_planner.PushPlan)
        self.assertIsInstance(plan(5.0), push_planner.PushPlan, "a clearance under the floor narrowed something")
        refused = plan(owners.hand_clearance_mm)
        self.assertEqual("no_free_direction", refused.code)
        self.assertEqual({"hand_blocked"}, {candidate.verdict for candidate in refused.candidates})
        with self.assertRaises(ValueError):
            plan(float("nan"))

        cell = _PushCell(self)
        with mock.patch("src.robot.grasping.recovery.push_planner.plan_push", wraps=push_planner.plan_push) as planned:
            cell.campaign()
        self.assertEqual(25.0, planned.call_args.kwargs["hand_clearance_mm"])
        self.assertEqual(30.0, planned.call_args.kwargs["push_distance_mm"])
        self.assertIsNotNone(planned.call_args.kwargs["operator_box"])

    def test_a_neighbour_only_another_look_saw_is_evidence_too(self) -> None:
        """The trigger's neighbour evidence is read off what every look saw (the fused neighbour cloud the candidate
        filter reads), not only off the look the part was judged on: a post that look did not see still stands."""
        from types import SimpleNamespace

        from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator

        part = _box_points((-20.0, -720.0, 0.0), (20.0, -680.0, 40.0))
        post = _box_points(POST.low, POST.high)
        judged = SimpleNamespace(look=SimpleNamespace(clouds=(part,)),
                                 fused_scene=SimpleNamespace(neighbour_for=lambda index: post if index == 0 else None))

        seen = BinPickingOrchestrator._neighbour_points_of(judged, 0)  # type: ignore[arg-type]  # noqa: SLF001

        np.testing.assert_allclose(post, seen)


class TheServiceSaysWhatTheRecoveryCameToTests(unittest.TestCase):
    def test_a_loop_that_ran_out_of_actions_is_recovery_exhausted(self) -> None:
        other = Box((60.0, -720.0, 0.0), (100.0, -680.0, 40.0), "part")
        cell = _PushCell(self, mode=GraspMode.AUTO, allowed=("next_target",), boxes=(other, CUBE), max_actions=1)

        run = cell.campaign()

        report = run.last
        self.assertIs(AutonomousGraspOutcome.RECOVERY_EXHAUSTED, report.outcome)
        self.assertEqual("exhausted_budget", report.telemetry["recovery_trail_terminal_reason"])
        self.assertEqual([("next_target", "completed")],
                         [(row["action"], row["outcome"]) for row in report.recovery_actions])
        self.assertEqual("recovery_exhausted", run.attempts[0].reported)
        self.assertFalse(report.needs_person)

    def test_an_empty_bench_is_no_target_whatever_recovery_tried(self) -> None:
        """Nothing perceived is no dead loop: a pick that finds nothing reports ``no_target`` once its recovery
        rescanned or skipped, as a pick without recovery does (the KPI roll-up counts ``recovery_exhausted`` as a dead
        loop, and a campaign that empties the bench is not one)."""
        for mode, allowed, looks in ((GraspMode.AUTO, ("rescan",), False),
                                     (GraspMode.AUTO, ("rescan", "next_target"), False),
                                     (GraspMode.DENSE_CLUTTER, ("rescan", "next_target", "nudge_target"), True)):
            with self.subTest(allowed=allowed, looks=looks):
                cell = _PushCell(self, mode=mode, allowed=allowed, boxes=())
                if looks:
                    report = cell.service.pick(look=list(LOOKS))
                else:
                    cell.arm.move_to_joints(LOOK_PLUS_X)  # where the arm stands; the pick is handed no look
                    report = cell.service.pick()

                self.assertIs(AutonomousGraspOutcome.NO_TARGET, report.outcome, report.render())
                self.assertEqual([], cell.arm.pushes)

    def test_a_report_that_needs_a_person_gives_the_loop_nothing_to_recover_from(self) -> None:
        from types import SimpleNamespace

        from src.robot.execution.autonomous_grasp.config import _profile_for
        from src.robot.execution.autonomous_grasp.report import AutonomousGraspReport
        from src.robot.execution.autonomous_grasp.service import _RecoveryReportAdapter

        attempt = SimpleNamespace(reasons=(GraspFailureReason.ALL_COLLIDED,), action="push", motion_status=None,
                                  motion_message=None)
        stopped = AutonomousGraspReport(
            outcome=AutonomousGraspOutcome.UNSAFE_RECOVERY_REFUSED, mode=GraspMode.DENSE_CLUTTER,
            profile=_profile_for(GraspMode.DENSE_CLUTTER), pick_report=SimpleNamespace(attempts=(attempt,)),
            telemetry={"push_stopped": "the back leg was refused"})
        self.assertTrue(stopped.needs_person)
        self.assertEqual((), _RecoveryReportAdapter(stopped).failure_reasons)
        going_on = replace(stopped, outcome=AutonomousGraspOutcome.NO_VALID_GRASP, telemetry={})
        self.assertFalse(going_on.needs_person)
        self.assertEqual((GraspFailureReason.ALL_COLLIDED,), _RecoveryReportAdapter(going_on).failure_reasons)


class ACalculatorOfEitherKindTriggersThePushTests(unittest.TestCase):
    """Both calculators map their rejections to ``ALL_COLLIDED`` through the same counters, and the push reads the
    reason, not the calculator: the pick loop pushes on either one's collision."""

    def _assert_pushed_on(self, result: GraspResult) -> None:
        self.assertIn(GraspFailureReason.ALL_COLLIDED, result.reasons)
        cell = _PushCell(self, allowed=("rescan", "nudge_target"), stuck=True)

        def compute(seg: Any, *args: Any, **kwargs: Any) -> GraspResult:
            cell.calculator.calls.append((len(cell.camera.taken) - 1, str(seg.label), dict(kwargs)))
            return result

        cell.calculator.compute_result = compute  # type: ignore[method-assign]
        cell.service.pick(look=list(LOOKS))
        self.assertEqual(PUSH_LABELS, cell.arm.push_motions())
        self.assertEqual("all_collided", cell.service.runtime.orchestrator.pushes[0].trigger)

    def test_the_geometric_calculator_s_collision_triggers_it(self) -> None:
        from src.robot.grasping.generation.calculator import GraspCalculator

        reasons = GraspCalculator._derive_failure_reasons(  # noqa: SLF001 (the one mapping the analytic path uses)
            candidates=[], telemetry={"rejected_collision": 12}, had_filters=True)
        self._assert_pushed_on(GraspResult(reasons=reasons))

    def test_the_deep_calculator_s_collision_triggers_it(self) -> None:
        from src.robot.grasping.collision import rejection_reasons

        self._assert_pushed_on(GraspResult(reasons=rejection_reasons({"rejected_collision": 12.0})))


def _box_points(low: Any, high: Any, *, step_mm: float = 4.0) -> np.ndarray:
    """The points of a box's top and sides, every ``step_mm``, BASE mm: a part as the looks place it."""
    (x0, y0, z0), (x1, y1, z1) = (tuple(float(v) for v in low), tuple(float(v) for v in high))
    xs, ys, zs = (np.arange(a, b + 1e-9, step_mm) for a, b in ((x0, x1), (y0, y1), (z0, z1)))
    grid = np.array([(x, y, z) for x in xs for y in ys for z in zs], dtype=np.float64)
    on_surface = ((np.isclose(grid[:, 0], xs[0]) | np.isclose(grid[:, 0], xs[-1])
                   | np.isclose(grid[:, 1], ys[0]) | np.isclose(grid[:, 1], ys[-1]) | np.isclose(grid[:, 2], zs[-1])))
    return grid[on_surface]


def _centre_of_mask(seg: Any, cell: _PushCell) -> np.ndarray:
    """The BASE centre of the box a segmentation of the last frame shows, read off the scene by its mask."""
    frame = cell.camera.taken[-1]
    from tests._wrist_views import mount, render

    _depth, hit = render(frame.to_matrix() @ mount(), cell.camera.boxes)
    for index, box in enumerate(cell.camera.boxes):
        if np.array_equal(np.asarray(seg.mask), hit == index):
            return (np.asarray(box.low) + np.asarray(box.high)) / 2.0
    raise AssertionError("no box of the scene shows this mask")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
