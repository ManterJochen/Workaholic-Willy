"""A wrist pick end to end: a campaign and a console run on a wrist cell, every look through the pick loop.

The integration of 2026-09-29 (multi-view plan, addendum 1 to 3). The three tracks were built apart: the pick loop looks
around and generates its one view (``BinPickingOrchestrator.look_around``, ``src/robot/execution/generated_view.py``),
the service hands its looks and ``both_faces`` to the loop and reports what they came to, and ``PickRun`` and the console
hand every pick its looks. These tests run them together, through ``PickRun.from_service`` and the console's run body,
on one wrist cell of doubles:

* two declared looks, a safe grasp at the first: the looking ends there, and the arm is sent nowhere else;
* an uncertain grasp at the first: the second look, fused with the first, and the grasp ranked on both;
* nothing safe after the declared looks: the one generated view, on the straight joint line only (the planner fails the
  test when it is asked for anything but a declared look), and the pick goes on from it; or, where the view does not
  see the part, the move back to the look that did, on the straight line again;
* ``both_faces`` with a contact face no view showed: ``no_valid_grasp``, nothing gripped, the face named; and
  ``both_faces`` with a motion that turns every grasp off the faces that were judged: refused before anything moves;
* recovery armed, as the owner's ``dense_clutter`` preset arms it: a pick handed looks has had its rescans, so its
  looks and its one generated view run once per pick; a pick handed no look still rescans where it stands;
* a console run of the same cell never asks at the terminal, and says what the looks came to on its event, a look the
  arm did not reach included;
* a fixed camera's campaign is the one it always was.

The scene is ``tests/_wrist_views.py`` (a 40 mm cube, the owner's tilted D415 ray cast from where the tool stands); the
arm, the calculator and the policy are the pick loop's doubles (``tests/test_a_wrist_pick_looks_until_its_grasp_is_safe``):
an arm that names the configuration of a turn about the cube and drives straight joint lines, a calculator that grasps
the cube on the frames a test names, straight down and closing along base x, and a policy that records where the arm
stood when it was handed the grasp.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from src.geometry import Frame, Transform
from src.robot.core import JointPositions, MotionCommand, MotionResult, MotionStatus
from src.robot.execution.autonomous_grasp import (
    AutonomousGraspOutcome,
    AutonomousGraspService,
    EffectiveGraspingConfig,
    EffectiveRecoveryOrchestratorConfig,
    GraspMode,
)
from src.robot.execution.pick_run import PickOutcome, PickRun, Recording
from src.robot.grasping.motion.frame_resolver import EyeInHandFrameResolver, StaticCameraToBaseResolver
from src.robot.grasping.types.perception import PerceptionFrame
from tests._wrist_views import CUBE, Box, WristCamera, base_points, camera_looking_at, camera_to_tool, joints_key
from tests.test_a_wrist_pick_hands_its_looks_to_the_pick_loop import _Watching
from tests.test_a_wrist_pick_looks_until_its_grasp_is_safe import (
    HAND_E,
    LOOK_ELSEWHERE,
    LOOK_MINUS_X,
    LOOK_PLUS_X,
    LOOK_PLUS_Y,
    _keys,
    _label,
    _OrbitingArm,
    _Policy,
)

#: A part like the cube 400 mm along +y of it, which the look from elsewhere sees and the others do not.
OTHER = Box((-20.0, -320.0, 0.0), (20.0, -280.0, 30.0), "part")


@dataclass(eq=False)
class _Camera(WristCamera):
    """The wrist D415, which from frame ``blind_from`` on segments nothing: a view that does not see the part."""

    blind_from: int | None = None

    def acquire(self) -> PerceptionFrame:
        frame = super().acquire()
        if self.blind_from is not None and len(self.taken) - 1 >= self.blind_from:
            return replace(frame, segmentations=())
        return frame


class _WristCell:
    """The pick service on the orbiting arm, its wrist camera, the watching calculator and the recording policy."""

    def __init__(self, test: unittest.TestCase, *, grasps_on: tuple[int, ...] | None = None,
                 uncertain_on: tuple[int, ...] = (), boxes: tuple[Box, ...] = (CUBE,), blind_from: int | None = None,
                 fixed: bool = False, gripper: Any = None, mode: GraspMode = GraspMode.EASY) -> None:
        self.arm = _OrbitingArm(test)
        self.camera = _Camera(self.arm, boxes=boxes, blind_from=blind_from)
        self.calculator = _Watching(self.camera, grasps_on=grasps_on, uncertain_on=uncertain_on)
        self.policy = _Policy(self.arm)
        # The service drives only a policy of its own arm and hand (`foreign_policy_refusal`).
        self.policy.gripper = gripper  # type: ignore[attr-defined]
        resolver: Any = (StaticCameraToBaseResolver(transform=Transform.from_matrix(
            camera_looking_at(bearing_deg=0.0), from_frame=Frame.CAMERA, to_frame=Frame.BASE)) if fixed
            else EyeInHandFrameResolver(t_cam_to_tool=camera_to_tool()))
        self.service = AutonomousGraspService.from_components(
            arm=self.arm, calculator=self.calculator, perception=self.camera,  # type: ignore[arg-type]
            mode=mode, policy=self.policy, frame_resolver=resolver, max_attempts=1,  # type: ignore[arg-type]
            **({"gripper": gripper} if gripper is not None else {}),
        )
        self.orchestrator = self.service.runtime.orchestrator
        self.orchestrator.gripper_model = HAND_E
        self.orchestrator.primary_camera_id = "wrist"
        self.calculator.orchestrator = self.orchestrator

    def campaign(self, *looks: Any, **keywords: Any) -> Any:
        return PickRun.from_service(self.service, runs=1, recording=keywords.pop("recording", Recording.off()),
                                    look=list(looks), **keywords).execute()

    def approached_from(self) -> list[tuple[float, ...]]:
        """Where the arm stood each time it was handed a grasp: the start of its approach."""
        return [joints_key(joints) for joints in self.policy.joints_at]

    def recovering(self) -> None:
        """Arm the recovery loop as the owner's ``dense_clutter`` preset does: rescan allowed, two actions at most."""
        self.service.effective_config = EffectiveGraspingConfig(
            default_mode=GraspMode.AUTO, max_attempts=1,
            recovery_orchestrator=EffectiveRecoveryOrchestratorConfig(
                enabled=True, allowed_actions=("rescan",), max_actions=2))

    def rejecting(self, look: JointPositions) -> None:
        """The controller rejects the motion to ``look`` once it may have been sent (``CONTROLLER_REJECTED``)."""
        declared = self.arm.move_to_joints

        def move_to_joints(joints: JointPositions, **keywords: Any) -> MotionResult:
            if joints_key(joints) != joints_key(look):
                return declared(joints, **keywords)
            self.arm.motions.append(("joints", joints_key(joints)))
            return MotionResult.failed(MotionStatus.CONTROLLER_REJECTED, MotionCommand.MOVE_JOINTS,
                                       target_joints=joints, message="Driver reported moveJ() failure")

        self.arm.move_to_joints = move_to_joints  # type: ignore[method-assign]


class APickRunOnAWristCellTests(unittest.TestCase):
    def test_a_safe_grasp_at_the_first_look_ends_the_looking_with_no_second_motion(self) -> None:
        cell = _WristCell(self)

        run = cell.campaign(LOOK_PLUS_X, LOOK_MINUS_X)

        (attempt,) = run.attempts
        self.assertTrue(attempt.passed, run.render())
        self.assertEqual([("joints", _keys(LOOK_PLUS_X)[0])], cell.arm.motions, "the arm was sent on after a safe grasp")
        self.assertEqual((_label(LOOK_PLUS_X),), attempt.looks)
        self.assertEqual((_label(LOOK_PLUS_X),), attempt.looks_fused)
        self.assertIsNone(attempt.generated_view_deg)
        self.assertEqual([], cell.arm.screened, "a view was generated after a safe grasp")
        self.assertEqual(_keys(LOOK_PLUS_X), cell.approached_from())
        self.assertEqual(["hold", "forget"], cell.arm.live_planner_world.calls,
                         "the frames of the pick were not held for it, or not let go when it ended")
        self.assertEqual([True], cell.policy.held_at, "the approach was not judged against the frames of the pick")

    def test_an_uncertain_first_look_goes_on_to_the_second_and_the_grasp_is_ranked_on_both_fused(self) -> None:
        cell = _WristCell(self, grasps_on=(1,), uncertain_on=(0,))

        run = cell.campaign(LOOK_PLUS_X, LOOK_MINUS_X)

        (attempt,) = run.attempts
        self.assertTrue(attempt.passed, run.render())
        self.assertEqual([("joints", _keys(LOOK_PLUS_X)[0]), ("joints", _keys(LOOK_MINUS_X)[0])], cell.arm.motions)
        self.assertEqual((_label(LOOK_PLUS_X), _label(LOOK_MINUS_X)), attempt.looks)
        self.assertEqual((_label(LOOK_MINUS_X), _label(LOOK_PLUS_X)), attempt.looks_fused)
        self.assertEqual((True, True), attempt.jaw_faces_seen, "the fused looks did not show both contact faces")
        (ranked,) = cell.calculator.calls_on(1)
        cloud = base_points(ranked, "geometry_points_base_mm")
        self.assertGreater(int(np.sum(cloud[:, 0] > 19.0)), 100, "the first look's +x face was not fused in")
        self.assertGreater(int(np.sum(cloud[:, 0] < -19.0)), 100, "the second look's -x face was not fused in")
        self.assertIsNone(attempt.generated_view_deg)
        self.assertEqual(_keys(LOOK_MINUS_X), cell.approached_from())

    def test_nothing_safe_after_the_declared_looks_generates_one_view_on_the_straight_line_and_picks_from_it(
            self) -> None:
        """Neither declared look gives a certain grasp; the +y look shows neither jaw face, so the view turns from it to
        the face at jaw 1 that no look showed, 50 degrees on round the part, where the base stands at 140. The arm's
        planner fails the test if it is asked for the view: the straight joint line is all that runs."""
        cell = _WristCell(self, grasps_on=(2,), uncertain_on=(0, 1))

        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "picks.jsonl")
            run = cell.campaign(LOOK_PLUS_X, LOOK_PLUS_Y, recording=Recording.to_file(path))
            (record,) = [json.loads(line) for line in Path(path).read_text().splitlines()]

        (attempt,) = run.attempts
        self.assertTrue(attempt.passed, run.render())
        self.assertEqual(50.0, record["extra"]["generated_view_deg"], "the record does not say the generated view")
        turned = cell.arm.turned_to(140.0)
        self.assertEqual([("joints", _keys(LOOK_PLUS_X)[0]), ("joints", _keys(LOOK_PLUS_Y)[0]), ("line", turned)],
                         cell.arm.motions)
        self.assertEqual([(_keys(LOOK_PLUS_Y)[0], turned)], cell.arm.lines, "not the straight line from where it stood")
        self.assertEqual(50.0, attempt.generated_view_deg)
        generated = attempt.looks[-1]
        self.assertTrue(generated.startswith("orbit +50 deg ("), attempt.looks)
        self.assertEqual((_label(LOOK_PLUS_X), _label(LOOK_PLUS_Y), generated), attempt.looks)
        self.assertEqual(generated, attempt.looks_fused[0], "the grasp was not ranked on the generated view")
        self.assertEqual((True, True), attempt.jaw_faces_seen)
        self.assertEqual([turned], cell.approached_from())
        self.assertIn("generated one view, turned +50 deg", attempt.render())

    def test_a_generated_view_that_does_not_see_the_part_moves_back_to_the_look_that_did(self) -> None:
        """The first look's grasp is uncertain and the look from elsewhere misses the part; the view generated from the
        first look sees nothing there. So the grasp stays the first look's, and the arm goes back to it, 140 degrees
        on the base, on the straight joint line again, and approaches from there."""
        cell = _WristCell(self, grasps_on=(), uncertain_on=(0,), boxes=(CUBE, OTHER), blind_from=2)

        with self.assertLogs("src.robot.execution.generated_view", level="WARNING") as said:
            run = cell.campaign(LOOK_PLUS_X, LOOK_ELSEWHERE)

        (attempt,) = run.attempts
        self.assertTrue(attempt.passed, run.render())
        turned = cell.arm.turned_to(140.0)
        plus_x, elsewhere = _keys(LOOK_PLUS_X, LOOK_ELSEWHERE)
        self.assertEqual([("joints", plus_x), ("joints", elsewhere), ("line", turned), ("line", plus_x)],
                         cell.arm.motions)
        self.assertEqual([(elsewhere, turned), (turned, plus_x)], cell.arm.lines)
        self.assertEqual(140.0, attempt.generated_view_deg)
        self.assertEqual([plus_x], cell.approached_from(), "the approach did not start from the look that saw the part")
        self.assertEqual((_label(LOOK_PLUS_X),), attempt.looks_fused)
        self.assertTrue(any("moved back to look" in line and "past 120 deg" in line for line in said.output),
                        said.output)

    def test_a_move_back_refused_is_said_on_the_report_and_the_approach_starts_where_the_arm_stands(self) -> None:
        """As above, and the straight joint line back to the +x look is not clear: nothing is planned around it, the
        approach starts from the generated view where the arm stands, and the report says why it did not go back."""
        cell = _WristCell(self, grasps_on=(), uncertain_on=(0,), boxes=(CUBE, OTHER), blind_from=2)
        (plus_x,) = _keys(LOOK_PLUS_X)
        cell.arm.blocked.add(plus_x)

        report = cell.service.pick(look=[LOOK_PLUS_X, LOOK_ELSEWHERE])

        self.assertIs(AutonomousGraspOutcome.SUCCEEDED, report.outcome, report.render())
        turned = cell.arm.turned_to(140.0)
        self.assertEqual([("line", turned), ("line", plus_x)], [m for m in cell.arm.motions if m[0] == "line"])
        self.assertEqual([turned], cell.approached_from(), "the approach did not start where the arm stood")
        self.assertIn("refused before anything was sent", report.telemetry["move_back_refused"])
        self.assertIn("the move back was refused (its straight joint line was refused", report.render())

    def test_both_faces_with_a_contact_face_no_view_showed_grips_nothing_and_names_the_face(self) -> None:
        """Every straight line round the part is refused, so no view shows the face at jaw 1: with both faces asked
        for the pick ends ``no_valid_grasp`` with nothing gripped, the face named on the attempt and the record."""
        cell = _WristCell(self)
        cell.arm.blocked = {cell.arm.turned_to(float(turn)) for turn in range(-180, 181, 5)}
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "picks.jsonl")
            run = cell.campaign(LOOK_PLUS_X, both_faces=True, recording=Recording.to_file(path))
            (record,) = [json.loads(line) for line in Path(path).read_text().splitlines()]

        (attempt,) = run.attempts
        self.assertFalse(attempt.passed)
        self.assertEqual("no_valid_grasp", attempt.reported)
        self.assertIn("the contact face at jaw 1 of the chosen grasp was not seen", attempt.detail)
        self.assertNotIn("jaw 2 of the chosen grasp", attempt.detail)
        self.assertEqual([], cell.policy.executed, "a grasp whose contact face nobody saw was gripped")
        self.assertEqual(3, len(cell.arm.lines), "more generated motions were tried than three")
        self.assertIsNone(attempt.generated_view_deg)
        self.assertEqual("no_valid_grasp", record["final_outcome"])
        self.assertIs(True, record["extra"]["both_faces"])
        self.assertEqual([False, True], record["extra"]["jaw_faces_seen"])
        self.assertNotIn("generated_view_deg", record["extra"])

    def test_both_faces_with_a_motion_that_turns_the_grasp_is_refused_before_anything_moves(self) -> None:
        """``both_faces`` grips only the grasp whose contact faces were judged; a motion that yaws every grasp about
        base Z onto base X (``GraspMotion(align_closing_to_base_x=True)``) would close the jaws on faces nobody judged.
        The two together are the program's error: the campaign stops on it before the arm moves or the camera looks, a
        fixed camera's move to its first look included."""
        for fixed in (False, True):
            with self.subTest(fixed=fixed):
                cell = _WristCell(self, fixed=fixed)
                cell.policy.align_closing_to_base_x = True  # type: ignore[attr-defined]

                run = cell.campaign(LOOK_PLUS_X, LOOK_MINUS_X, both_faces=True)

                (attempt,) = run.attempts
                self.assertIs(PickOutcome.RAISED, attempt.outcome)
                self.assertIn("align_closing_to_base_x", attempt.detail)
                self.assertEqual([], cell.arm.motions, "the arm moved for a pick that could grip only unjudged faces")
                self.assertEqual([], cell.camera.taken)
                self.assertEqual([], cell.arm.live_planner_world.calls)
                self.assertEqual([], cell.policy.executed)
                self.assertTrue(cell.campaign(LOOK_PLUS_X).attempts[0].passed,
                                "the control: without both faces it picks")

    def test_a_fixed_camera_campaign_is_the_one_it_always_was(self) -> None:
        cell = _WristCell(self, fixed=True)

        run = PickRun.from_service(cell.service, runs=2, recording=Recording.off()).execute()

        self.assertEqual(2, run.succeeded, run.render())
        self.assertEqual([], cell.arm.motions, "a fixed camera's pick moved the arm to look")
        self.assertEqual([], cell.arm.screened)
        self.assertEqual([], cell.arm.live_planner_world.calls, "a fixed camera's pick held frames")
        self.assertEqual([((), False), ((), False)], cell.calculator.held)
        for attempt in run.attempts:
            self.assertEqual(((), (), None, None), (attempt.looks, attempt.looks_fused, attempt.jaw_faces_seen,
                                                    attempt.generated_view_deg))
        self.assertEqual(2, len(cell.camera.taken), "one frame per pick, and no other")


class ARecoveryOnAWristCellTests(unittest.TestCase):
    """The owner's ``dense_clutter`` preset arms the recovery loop with ``rescan``, which re-runs the pick. A pick
    handed looks has had its rescans: its looks, each fused, and the one view they generated, the very last resort
    (addendum 3). So recovery never runs them again for it: the looks and the generated view run once per ``pick()``,
    and the arm's planner fails the test if it is asked for a declared look twice. A pick handed no look still rescans
    where it stands, as it always did."""

    def test_a_pick_with_no_grasp_drives_its_looks_and_its_one_generated_view_once(self) -> None:
        cell = _WristCell(self, grasps_on=(), mode=GraspMode.AUTO)
        cell.recovering()

        report = cell.service.pick(look=[LOOK_PLUS_X, LOOK_PLUS_Y])

        self.assertIn("recovery_trail_terminal_reason", report.telemetry, "the control: recovery was not armed")
        self.assertNotEqual(AutonomousGraspOutcome.SUCCEEDED, report.outcome)
        self.assertEqual(["joints", "joints", "line"], [kind for kind, _ in cell.arm.motions],
                         "the looks or the generated view were driven again")
        self.assertEqual(3, len(cell.camera.taken))
        self.assertEqual(["hold", "forget"], cell.arm.live_planner_world.calls)
        self.assertNotIn("rescan", report.telemetry["recovery_trail_actions"])
        self.assertEqual(3, len(report.looks))
        self.assertTrue(report.looks[-1].startswith("orbit "), report.looks)
        self.assertIsNotNone(report.generated_view_deg)

    def test_a_pick_both_faces_refused_is_not_driven_round_again(self) -> None:
        """Every straight line round the part is refused, so no view shows the face at jaw 1: the pick ends
        ``no_valid_grasp``, which the recovery loop maps to a rescan, and is not looked at again."""
        cell = _WristCell(self, mode=GraspMode.AUTO)
        cell.arm.blocked = {cell.arm.turned_to(float(turn)) for turn in range(-180, 181, 5)}
        cell.recovering()

        report = cell.service.pick(look=[LOOK_PLUS_X], both_faces=True)

        self.assertIs(AutonomousGraspOutcome.NO_VALID_GRASP, report.outcome)
        self.assertEqual(1, len([kind for kind, _ in cell.arm.motions if kind == "joints"]))
        self.assertEqual(3, len(cell.arm.lines), "the generated view's turns were tried again")
        self.assertEqual(["hold", "forget"], cell.arm.live_planner_world.calls)
        self.assertEqual([], cell.policy.executed)
        self.assertNotIn("rescan", report.telemetry["recovery_trail_actions"])
        self.assertIn("the contact face at jaw 1 of the chosen grasp was not seen", report.failure_summary())

    def test_a_pick_handed_no_look_still_rescans_where_it_stands(self) -> None:
        cell = _WristCell(self, grasps_on=(), mode=GraspMode.AUTO)
        cell.recovering()

        report = cell.service.pick()

        self.assertEqual([], cell.arm.motions)
        self.assertEqual(["rescan", "rescan"], report.telemetry["recovery_trail_actions"])
        self.assertEqual(3, len(cell.camera.taken), "one frame per pick, and one per rescan")


class AConsoleRunOnAWristCellTests(unittest.TestCase):
    def test_it_never_asks_at_the_terminal_and_says_what_the_looks_came_to(self) -> None:
        """The console hands every pick the cell profile's looks, and runs the fast rule (no ``both_faces``). The whole
        wrist path, the looks, the generated view and the approach, runs under the console's ``asking_nobody`` with the
        owner's toggle hand, and nobody is asked: not the hand's person, not the terminal."""
        from api.runs import RunState, _toggle_of
        from tests.test_a_console_run_stops_and_looks_as_a_campaign_does import _drive
        from tests.test_a_toggle_asks_where_its_jaws_stand import _NO_TERMINAL, Person, _toggle

        events: list[Any] = []
        jaws = _toggle(events, Person(""))
        jaws.connect()
        nobody = Person()
        jaws._ask = nobody  # noqa: SLF001 (any question from here on fails the test)
        cell = _WristCell(self, grasps_on=(2,), uncertain_on=(0, 1), gripper=jaws)
        cell.service.configured_looks = (LOOK_PLUS_X, LOOK_PLUS_Y)
        self.assertIs(jaws, _toggle_of(cell.service), "the control: the run's hand is the toggle")

        with mock.patch("sys.stdin", _NO_TERMINAL), \
                mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")) as typed:
            run, hub = _drive(cell.service, picks=1)

        self.assertIs(RunState.FINISHED, run.state, run.error)
        self.assertEqual(1, run.succeeded)
        self.assertEqual([], nobody.asked)
        typed.assert_not_called()
        self.assertEqual({False}, {asked for _looks, asked in cell.calculator.held}, "the console asked for both faces")
        (result,) = [event.data for event in hub.since(run.id, 0)[0] if event.type == "pick_result"]
        generated = result["looks"][-1]
        self.assertEqual([_label(LOOK_PLUS_X), _label(LOOK_PLUS_Y), generated], result["looks"])
        self.assertTrue(generated.startswith("orbit +50 deg ("), result["looks"])
        self.assertEqual(generated, result["looks_fused"][0])
        self.assertEqual(50.0, result["generated_view_deg"])
        self.assertEqual([True, True], result["jaw_faces_seen"])
        self.assertIsInstance(result["hand_eye_gap_mm"], float, "the hand-eye check of the looks is not said")
        self.assertNotIn("refused_look", result)
        self.assertEqual([("line", cell.arm.turned_to(140.0))],
                         [motion for motion in cell.arm.motions if motion[0] == "line"])

    def test_a_look_the_arm_did_not_reach_is_named_on_the_event(self) -> None:
        """The controller rejects the motion to the first look once it may have been sent: the pick ends there with
        nothing else commanded, and ``pick_result`` names the look."""
        from tests.test_a_console_run_stops_and_looks_as_a_campaign_does import _drive

        cell = _WristCell(self)
        cell.service.configured_looks = (LOOK_PLUS_X, LOOK_PLUS_Y)
        cell.rejecting(LOOK_PLUS_X)

        run, hub = _drive(cell.service, picks=1)

        (result,) = [event.data for event in hub.since(run.id, 0)[0] if event.type == "pick_result"]
        self.assertEqual(_label(LOOK_PLUS_X), result["refused_look"])
        self.assertEqual([("joints", _keys(LOOK_PLUS_X)[0])], cell.arm.motions, "the arm was sent on after the look")
        self.assertEqual([], cell.camera.taken)
        self.assertEqual(0, run.succeeded)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
