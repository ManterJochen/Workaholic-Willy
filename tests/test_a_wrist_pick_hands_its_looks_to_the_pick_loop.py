"""The pick service hands a wrist camera's looks to the pick loop, and reports what they came to (plan step 5).

The owner's decisions of 2026-09-28 and 2026-09-29: a wrist pick looks from the looks its program declares, in order,
and every look is fused with the ones before it; the looks exist to find a safe grasp point, never to scan the part
whole. The pick loop does the looking (``BinPickingOrchestrator.look_around``). The service no longer moves the arm to
each look itself and stops at the first look that finds something: it hands its looks to the loop (``orch.looks``,
``orch.both_faces``), takes them back when the pick ends, whatever ended it, and stamps its report from what the looks
came to (``orch.looked_around``): the looks visited, the looks fused into the grasp, whether each jaw contact face of
the chosen grasp was seen, and how far apart the looks measure the part (the hand-eye check).

A fixed camera's pick is the one it always was unless ``both_faces`` is set: the service's own loop over any looks it
is handed, the first look that finds something ending it. ``both_faces`` is the owner's switch for safety-critical
processes, a keyword of ``pick`` and of both ``PickRun`` doors; the console and the API keep it off. It means one thing
on every camera: a fixed camera, which cannot look around, grips only a grasp its cameras showed both contact faces of.
``record_views`` keeps each pick's looks for training, off by default.

The scene is ``tests/_wrist_views.py``: a 40 mm cube on the bench, ray cast through the wrist D415 from the tool pose
each look puts the arm at, and stamped with it. The service is the real one (``from_components``) with an eye-in-hand
resolver; the calculator grasps the cube on the frames a test names, straight down and closing along base x; the
policy records what it was asked to execute and moves nothing.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from src.geometry import Frame, Transform
from src.robot.core import JointPositions
from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome, AutonomousGraspService, GraspMode
from src.robot.execution.pick_run import PickRun, Recording
from src.robot.grasping.collision.gripper_model import ParallelJawGripperModel
from src.robot.grasping.decision import DecisionEngine, DecisionPolicy
from src.robot.grasping.motion.execution_policy import PolicyOutcome, PolicyReport
from src.robot.grasping.motion.frame_resolver import EyeInHandFrameResolver, StaticCameraToBaseResolver
from src.robot.grasping.types.perception import PerceptionFrame
from tests._wrist_views import (
    K,
    LookCalculator,
    LookingArm,
    WristCamera,
    camera_looking_at,
    camera_to_tool,
    mount,
    tool_for,
)

#: The looks a program declares: the cube from its +x side, its -x side and its +y side, 45 degrees up.
LOOK_PLUS_X = JointPositions.deg(0.0, -90.0, -110.0, -60.0, 90.0, 0.0)
LOOK_MINUS_X = JointPositions.deg(180.0, -90.0, -110.0, -60.0, 90.0, 0.0)
LOOK_PLUS_Y = JointPositions.deg(90.0, -90.0, -110.0, -60.0, 90.0, 0.0)
#: The Hand-E's contact patch (config/grippers/robotiq_hande.yaml), the owner's hand.
HAND_E = ParallelJawGripperModel(finger_width_mm=29.24, pad_length_mm=20.91, pad_ahead_mm=10.45)


def _label(joints: JointPositions) -> str:
    return "(" + ", ".join(f"{v:.1f}" for v in joints.degrees()) + ") deg"


def _key(joints: JointPositions) -> tuple[float, ...]:
    return tuple(round(v, 1) for v in joints.degrees())


def _poses() -> dict[JointPositions, Any]:
    return {
        LOOK_PLUS_X: tool_for(camera_looking_at(bearing_deg=0.0)),
        LOOK_MINUS_X: tool_for(camera_looking_at(bearing_deg=180.0)),
        LOOK_PLUS_Y: tool_for(camera_looking_at(bearing_deg=90.0)),
    }


class _Policy:
    """Executes nothing and says it executed, on the service's own arm: what the pick asked for is what is read."""

    def __init__(self, arm: Any) -> None:
        self.arm = arm
        self.gripper = None
        self.executed: list[Any] = []

    def execute(self, grasp: Any) -> PolicyReport:
        self.executed.append(grasp)
        return PolicyReport(outcome=PolicyOutcome.EXECUTED)


class _Watching(LookCalculator):
    """The scene's calculator, which also writes down the looks and the switch the pick loop held at every call."""

    orchestrator: Any = None

    def __init__(self, camera: WristCamera, **keywords: Any) -> None:
        super().__init__(camera, **keywords)
        self.held: list[tuple[tuple[Any, ...], bool]] = []

    def compute_result(self, seg: Any, *args: Any, **kwargs: Any) -> Any:
        if self.orchestrator is not None:
            self.held.append((tuple(self.orchestrator.looks), bool(self.orchestrator.both_faces)))
        return super().compute_result(seg, *args, **kwargs)


class _Cell:
    """The pick service on the looking arm, its wrist camera, the watching calculator and the recording policy."""

    def __init__(self, *, grasps_on: tuple[int, ...] | None = None, refuse: tuple[JointPositions, ...] = (),
                 unvouched: tuple[JointPositions, ...] = (), fixed: bool = False, deciding: bool = False,
                 max_attempts: int = 1) -> None:
        self.arm = LookingArm(_poses(), refuse=refuse, unvouched=unvouched)
        self.camera = WristCamera(self.arm)
        self.calculator = _Watching(self.camera, grasps_on=grasps_on)
        self.policy = _Policy(self.arm)
        resolver: Any = (StaticCameraToBaseResolver(transform=Transform.from_matrix(
            camera_looking_at(bearing_deg=0.0), from_frame=Frame.CAMERA, to_frame=Frame.BASE)) if fixed
            else EyeInHandFrameResolver(t_cam_to_tool=camera_to_tool()))
        wiring: dict[str, Any] = {}
        if deciding:
            policy = DecisionPolicy(enabled=True)
            wiring = {"decision_policy": policy, "decision_engine": DecisionEngine(policy=policy)}
        self.service = AutonomousGraspService.from_components(
            arm=self.arm, calculator=self.calculator, perception=self.camera,  # type: ignore[arg-type]
            mode=GraspMode.AUTO if deciding else GraspMode.EASY, policy=self.policy,  # type: ignore[arg-type]
            frame_resolver=resolver, max_attempts=max_attempts, **wiring,
        )
        self.orchestrator = self.service.runtime.orchestrator
        self.orchestrator.gripper_model = HAND_E
        self.orchestrator.primary_camera_id = "wrist"
        self.calculator.orchestrator = self.orchestrator

    def joints_moved_to(self) -> list[tuple[float, ...]]:
        return [key for kind, key in self.arm.motions if kind == "joints"]


def _looks_handed_back(test: unittest.TestCase, orchestrator: Any) -> None:
    """The pick loop holds no look of a pick that ended: no looks, the switch off, nothing held open."""
    test.assertEqual((), tuple(orchestrator.looks), "the pick loop kept the looks of a pick that ended")
    test.assertIs(False, orchestrator.both_faces, "the pick loop kept the switch of a pick that ended")
    test.assertIsNone(orchestrator._open_looked, "the looks of a pick that ended are still held open")  # noqa: SLF001
    test.assertIsNone(orchestrator._pending_looked)  # noqa: SLF001


# ---------------------------------------------------------------------------------------------------------------------
# The service hands its looks to the pick loop
# ---------------------------------------------------------------------------------------------------------------------


class TheServiceHandsItsLooksToThePickLoopTests(unittest.TestCase):
    def test_a_look_without_a_grasp_goes_on_to_the_next_and_the_looks_are_fused(self) -> None:
        """Red before: the service moved to the first look itself, the pick there saw the part and ranked no grasp,
        and that ended the looking there (an object was found)."""
        cell = _Cell(grasps_on=(1,))

        report = cell.service.pick(look=[LOOK_PLUS_X, LOOK_MINUS_X])

        self.assertIs(AutonomousGraspOutcome.SUCCEEDED, report.outcome, report.failure_summary())
        self.assertEqual([_key(LOOK_PLUS_X), _key(LOOK_MINUS_X)], cell.joints_moved_to())
        self.assertEqual(2, len(cell.camera.taken), "one frame per look, and no other")
        self.assertEqual((_label(LOOK_PLUS_X), _label(LOOK_MINUS_X)), report.looks)
        self.assertEqual((_label(LOOK_MINUS_X), _label(LOOK_PLUS_X)), report.looks_fused,
                         "the look the grasp was ranked on comes first")
        self.assertEqual((f"wrist@{_label(LOOK_MINUS_X)}", f"wrist@{_label(LOOK_PLUS_X)}"), report.fused_views)
        # The pick loop was handed the looks, and held them while it ranked each one.
        self.assertEqual({((LOOK_PLUS_X, LOOK_MINUS_X), False)}, set(cell.calculator.held))
        self.assertEqual(1, len(cell.policy.executed))

    def test_a_safe_grasp_at_the_first_look_ends_the_looking_there(self) -> None:
        cell = _Cell()

        report = cell.service.pick(look=[LOOK_PLUS_X, LOOK_MINUS_X])

        self.assertTrue(report.succeeded, report.failure_summary())
        self.assertEqual([_key(LOOK_PLUS_X)], cell.joints_moved_to(), "the arm was sent on after a safe grasp")
        self.assertEqual((_label(LOOK_PLUS_X),), report.looks)
        self.assertEqual((_label(LOOK_PLUS_X),), report.looks_fused)
        self.assertEqual((False, True), report.jaw_faces_seen, "from +x only the face at jaw 2 is seen")
        self.assertIsNone(report.hand_eye_gap_mm, "one look measures no gap between looks")

    def test_no_grasp_from_any_look_is_reported_with_every_look_visited(self) -> None:
        cell = _Cell(grasps_on=())

        report = cell.service.pick(look=[LOOK_PLUS_X, LOOK_MINUS_X])

        self.assertIs(AutonomousGraspOutcome.NO_VALID_GRASP, report.outcome)
        self.assertEqual((_label(LOOK_PLUS_X), _label(LOOK_MINUS_X)), report.looks)
        self.assertIn("looked from 2", report.failure_summary())
        self.assertEqual([], cell.policy.executed)

    def test_a_pick_hands_its_looks_back_whatever_ended_it_and_the_next_pick_inherits_nothing(self) -> None:
        """Red before: ``pick`` took no ``both_faces``. The looks and the switch are the pick's, not the cell's: the
        service takes them back from the pick loop when the pick ends, and a pick handed none perceives where the
        arm stands, with the switch off."""
        cell = _Cell(grasps_on=(1,))

        first = cell.service.pick(look=[LOOK_PLUS_X, LOOK_MINUS_X], both_faces=True)
        _looks_handed_back(self, cell.orchestrator)
        moved = len(cell.arm.motions)
        second = cell.service.pick()

        self.assertTrue(first.succeeded, first.failure_summary())
        self.assertEqual({((LOOK_PLUS_X, LOOK_MINUS_X), True)}, set(cell.calculator.held[:2]))
        self.assertEqual([((), False)], cell.calculator.held[2:], "the second pick inherited the first's looks")
        self.assertEqual(moved, len(cell.arm.motions), "a pick handed no look moved the arm to look")
        self.assertEqual((), second.looks)
        self.assertEqual((), second.looks_fused)
        _looks_handed_back(self, cell.orchestrator)

    def test_a_pick_a_fault_ended_hands_its_looks_back_too(self) -> None:
        cell = _Cell(grasps_on=(1,), unvouched=(LOOK_MINUS_X,))

        report = cell.service.pick(look=[LOOK_PLUS_X, LOOK_MINUS_X], both_faces=True)

        self.assertIsNotNone(report.fault, "a camera that could not vouch on the way is a fault of the cell")
        _looks_handed_back(self, cell.orchestrator)

    def test_a_pick_that_raised_hands_its_looks_back_too(self) -> None:
        cell = _Cell(grasps_on=(1,))

        with mock.patch.object(cell.camera, "acquire", side_effect=TypeError("a programmer's error")), \
                self.assertRaises(TypeError):
            cell.service.pick(look=[LOOK_PLUS_X, LOOK_MINUS_X], both_faces=True)

        _looks_handed_back(self, cell.orchestrator)

    def test_a_stop_before_the_first_look_moves_nothing(self) -> None:
        cell = _Cell()
        asked: list[bool] = []
        # Not at the check that opens the pick; from the next question on.
        cell.service.set_cancel_check(lambda: bool(asked.append(True)) or len(asked) > 1)

        report = cell.service.pick(look=[LOOK_PLUS_X, LOOK_MINUS_X])

        self.assertIs(AutonomousGraspOutcome.CANCELLED, report.outcome)
        self.assertEqual([], cell.arm.motions)
        self.assertEqual((), report.looks)
        self.assertTrue(report.telemetry["cancelled_before_start"])
        _looks_handed_back(self, cell.orchestrator)

    def test_a_stop_between_looks_names_the_looks_visited(self) -> None:
        cell = _Cell(grasps_on=())
        cell.service.set_cancel_check(lambda: len(cell.camera.taken) > 0)

        report = cell.service.pick(look=[LOOK_PLUS_X, LOOK_MINUS_X])

        self.assertIs(AutonomousGraspOutcome.CANCELLED, report.outcome)
        self.assertEqual([_key(LOOK_PLUS_X)], cell.joints_moved_to())
        self.assertEqual((_label(LOOK_PLUS_X),), report.looks)
        self.assertIs(False, report.telemetry["cancelled_before_start"])

    def test_a_pick_that_reaches_no_look_says_which_it_did_not_reach(self) -> None:
        cell = _Cell(refuse=(LOOK_PLUS_X, LOOK_MINUS_X))

        report = cell.service.pick(look=[LOOK_PLUS_X, LOOK_MINUS_X])

        self.assertIs(AutonomousGraspOutcome.EXECUTION_FAILED, report.outcome)
        self.assertEqual([], cell.camera.taken, "a frame was taken although no look was reached")
        self.assertEqual((), report.looks)
        self.assertEqual(_label(LOOK_PLUS_X), report.telemetry["refused_look"])
        self.assertIn("outside the cable window", report.telemetry["look_refused"])
        self.assertIn(f"look {_label(LOOK_PLUS_X)} not reached", report.failure_summary())
        self.assertIn("outside the cable window", report.failure_summary())


class BothJawFacesTests(unittest.TestCase):
    """``both_faces``: the owner's switch for safety-critical processes, off by default, handed to the pick loop."""

    def test_off_by_default_one_face_seen_is_enough_to_stop(self) -> None:
        cell = _Cell()

        report = cell.service.pick(look=[LOOK_PLUS_X, LOOK_MINUS_X, LOOK_PLUS_Y])

        self.assertEqual([_key(LOOK_PLUS_X)], cell.joints_moved_to())
        self.assertEqual((False, True), report.jaw_faces_seen)

    def test_on_the_pick_looks_until_both_contact_faces_were_seen(self) -> None:
        cell = _Cell()

        report = cell.service.pick(look=[LOOK_PLUS_X, LOOK_MINUS_X, LOOK_PLUS_Y], both_faces=True)

        self.assertTrue(report.succeeded, report.failure_summary())
        self.assertEqual([_key(LOOK_PLUS_X), _key(LOOK_MINUS_X)], cell.joints_moved_to())
        self.assertEqual((True, True), report.jaw_faces_seen)

    def test_a_campaign_hands_the_switch_to_every_pick_and_the_default_leaves_it_off(self) -> None:
        for asked, expected in ((True, True), (False, False)):
            with self.subTest(both_faces=asked):
                cell = _Cell()
                run = PickRun.from_service(cell.service, runs=2, recording=Recording.off(),
                                           look=[LOOK_PLUS_X, LOOK_MINUS_X], both_faces=asked).execute()

                self.assertEqual(2, run.succeeded, run.render())
                self.assertEqual({expected}, {held for _looks, held in cell.calculator.held})

    def test_a_fixed_camera_takes_the_switch_and_grips_only_what_its_cameras_showed_both_faces_of(self) -> None:
        """The switch means one thing on every camera (integration of 2026-09-29, the pick loop's fixed-camera ending
        and the Locator's alike): a fixed camera cannot look around, so from the cube's +x side alone the face at jaw 1
        stays unseen, and nothing is gripped. The switch is handed to its pick loop and taken back."""
        cell = _Cell(fixed=True)

        report = cell.service.pick(both_faces=True)

        self.assertIs(AutonomousGraspOutcome.NO_VALID_GRASP, report.outcome, report.failure_summary())
        self.assertEqual([], cell.policy.executed, "a grasp whose contact face nobody saw was gripped")
        self.assertEqual([((), True)], cell.calculator.held, "a fixed camera's pick loop was not handed the switch")
        self.assertIs(False, cell.orchestrator.both_faces, "the switch outlived the pick")
        self.assertIsNone(report.jaw_faces_seen)
        self.assertIs(True, report.both_faces)
        self.assertIn("the contact face at jaw 1 of the chosen grasp was not seen", report.failure_summary())

    def test_a_fixed_camera_without_the_switch_is_the_pick_it_was(self) -> None:
        cell = _Cell(fixed=True)

        report = cell.service.pick()

        self.assertTrue(report.succeeded, report.failure_summary())
        self.assertEqual([((), False)], cell.calculator.held, "a fixed camera's pick loop was handed looks")
        self.assertIsNone(report.jaw_faces_seen)
        self.assertIs(False, report.both_faces)

    def test_on_the_decision_path_a_grasp_whose_faces_were_not_both_seen_is_not_gripped(self) -> None:
        """Red before: the decision layer decided ``GRASP_NOW`` on the looks' grasp and the service handed it on
        without asking whether both contact faces were seen, so a pick that asked for both gripped with the face at
        jaw 1 unseen. Addendum 1: with the switch on and no look showing both, do not grip; ``NO_VALID_GRASP``, naming
        the face."""
        cell = _Cell(deciding=True)
        with tempfile.TemporaryDirectory() as folder, \
                self.assertLogs("src.robot.execution.autonomous_grasp.service", "WARNING") as said:
            cell.service.enable_record_logging(Path(folder) / "picks.jsonl")
            report = cell.service.pick(look=[LOOK_PLUS_X, LOOK_PLUS_Y], both_faces=True)
            (record,) = [json.loads(line) for line in (Path(folder) / "picks.jsonl").read_text().splitlines()]

        self.assertTrue(any("the contact face at jaw 1 of the chosen grasp was not seen" in line
                            and "does not grip" in line for line in said.output), said.output)
        self.assertIs(AutonomousGraspOutcome.NO_VALID_GRASP, report.outcome, report.failure_summary())
        self.assertEqual([], cell.policy.executed, "a grasp whose contact face nobody saw was gripped")
        self.assertEqual([_key(LOOK_PLUS_X), _key(LOOK_PLUS_Y)], cell.joints_moved_to(), "the arm was sent on")
        self.assertIsNotNone(report.decision, "the engine's own word is kept beside the refusal")
        self.assertEqual((False, True), report.jaw_faces_seen)
        self.assertIs(True, report.both_faces)
        self.assertIn("the contact face at jaw 1 of the chosen grasp was not seen", report.failure_summary())
        self.assertNotIn("jaw 2 of the chosen grasp", report.failure_summary())
        self.assertEqual("no_valid_grasp", record["final_outcome"])
        self.assertIs(True, record["extra"]["both_faces"])
        _looks_handed_back(self, cell.orchestrator)

    def test_on_the_decision_path_both_faces_seen_or_not_asked_for_grips(self) -> None:
        """The controls: the switch refuses only a grasp whose faces were not both seen, and off it changes nothing."""
        for looks, asked, faces in (([LOOK_PLUS_X, LOOK_MINUS_X], True, (True, True)),
                                    ([LOOK_PLUS_X, LOOK_MINUS_X], False, (False, True))):
            with self.subTest(both_faces=asked):
                cell = _Cell(deciding=True)

                report = cell.service.pick(look=looks, both_faces=asked)

                self.assertIs(AutonomousGraspOutcome.SUCCEEDED, report.outcome, report.failure_summary())
                self.assertEqual(faces, report.jaw_faces_seen)
                self.assertEqual(1, len(cell.policy.executed))
                self.assertIs(asked, report.both_faces)

    def test_a_hand_with_no_jaw_faces_has_none_to_see(self) -> None:
        """A suction cup closes on no two faces: the switch asks nothing of it, on the decision path as in the loop."""
        from src.robot.grasping.collision.gripper_model import SuctionCupGripperModel

        cell = _Cell(deciding=True)
        cell.orchestrator.gripper_model = SuctionCupGripperModel()

        report = cell.service.pick(look=[LOOK_PLUS_X, LOOK_PLUS_Y], both_faces=True)

        self.assertIs(AutonomousGraspOutcome.SUCCEEDED, report.outcome, report.failure_summary())
        self.assertEqual([_key(LOOK_PLUS_X)], cell.joints_moved_to())
        self.assertIsNone(report.jaw_faces_seen)
        self.assertEqual(1, len(cell.policy.executed))

    def test_on_the_open_loop_a_grasp_whose_faces_were_not_both_seen_is_not_gripped(self) -> None:
        """The pick loop's fail-closed ending (addendum 1), on the path ``PickRun`` and the console take: with the switch
        on and no look showing both contact faces (this arm generates no view), the loop ends the pick
        ``NO_VALID_GRASP`` and grips nothing, and the report names the face. The decision path refuses the same grasp
        itself (the test above)."""
        cell = _Cell()

        report = cell.service.pick(look=[LOOK_PLUS_X, LOOK_PLUS_Y], both_faces=True)

        self.assertEqual([], cell.policy.executed, "a grasp whose contact face nobody saw was gripped")
        self.assertIs(AutonomousGraspOutcome.NO_VALID_GRASP, report.outcome)
        self.assertIn("the contact face at jaw 1 of the chosen grasp was not seen", report.failure_summary())
        self.assertNotIn("jaw 2 of the chosen grasp", report.failure_summary())
        self.assertIs(True, report.both_faces)
        _looks_handed_back(self, cell.orchestrator)


# ---------------------------------------------------------------------------------------------------------------------
# The decision layer decides on the fused looks
# ---------------------------------------------------------------------------------------------------------------------


class TheDecisionLayerDecidesOnTheFusedLooksTests(unittest.TestCase):
    def test_a_grasp_now_on_the_looks_is_picked_without_looking_again(self) -> None:
        """Red before: the decision layer decided on the first look's frame alone (an object and no grasp) and the
        service stopped looking there."""
        cell = _Cell(grasps_on=(1,), deciding=True)

        report = cell.service.pick(look=[LOOK_PLUS_X, LOOK_MINUS_X])

        self.assertIs(AutonomousGraspOutcome.SUCCEEDED, report.outcome, report.failure_summary())
        self.assertIsNotNone(report.decision)
        self.assertEqual([_key(LOOK_PLUS_X), _key(LOOK_MINUS_X)], cell.joints_moved_to())
        self.assertEqual(2, len(cell.camera.taken), "the pick looked around a second time after the decision")
        self.assertEqual((_label(LOOK_PLUS_X), _label(LOOK_MINUS_X)), report.looks)
        self.assertEqual((_label(LOOK_MINUS_X), _label(LOOK_PLUS_X)), report.looks_fused)
        self.assertEqual(1, len(cell.policy.executed))
        _looks_handed_back(self, cell.orchestrator)

    def test_no_grasp_from_any_look_is_decided_on_the_last_look_judged(self) -> None:
        cell = _Cell(grasps_on=(), deciding=True)

        report = cell.service.pick(look=[LOOK_PLUS_X, LOOK_MINUS_X])

        self.assertIs(AutonomousGraspOutcome.DECISION_RECOVER_PENDING, report.outcome)
        self.assertEqual([_key(LOOK_PLUS_X), _key(LOOK_MINUS_X)], cell.joints_moved_to())
        self.assertEqual((_label(LOOK_PLUS_X), _label(LOOK_MINUS_X)), report.looks)
        self.assertEqual(1, report.telemetry["n_segmentations"], "what the judged look's frame held")
        self.assertIn("no_candidates_generated", report.telemetry["reasons"])
        self.assertEqual([], cell.policy.executed)
        _looks_handed_back(self, cell.orchestrator)

    def test_a_pick_that_reaches_no_look_is_refused_as_before(self) -> None:
        cell = _Cell(refuse=(LOOK_PLUS_X, LOOK_MINUS_X), deciding=True)

        report = cell.service.pick(look=[LOOK_PLUS_X, LOOK_MINUS_X])

        self.assertIs(AutonomousGraspOutcome.EXECUTION_FAILED, report.outcome)
        self.assertIsNone(report.pick_report)
        self.assertEqual([], cell.camera.taken)
        self.assertEqual("look", report.telemetry["stage"])
        self.assertIn("outside the cable window", report.failure_summary())
        _looks_handed_back(self, cell.orchestrator)

    def test_a_stop_between_looks_is_a_cancel(self) -> None:
        cell = _Cell(grasps_on=(), deciding=True)
        cell.service.set_cancel_check(lambda: len(cell.camera.taken) > 0)

        report = cell.service.pick(look=[LOOK_PLUS_X, LOOK_MINUS_X])

        self.assertIs(AutonomousGraspOutcome.CANCELLED, report.outcome)
        self.assertEqual([_key(LOOK_PLUS_X)], cell.joints_moved_to())
        self.assertEqual((_label(LOOK_PLUS_X),), report.looks)


# ---------------------------------------------------------------------------------------------------------------------
# A fixed camera, and a wrist pick handed no look
# ---------------------------------------------------------------------------------------------------------------------


class AFixedCameraPicksAsItDidTests(unittest.TestCase):
    def test_a_fixed_camera_handed_looks_is_the_services_own_loop(self) -> None:
        """THE CONTROL: the service moves to each look itself and the first look that finds something ends it; the
        pick loop is never handed a look."""
        cell = _Cell(fixed=True, grasps_on=())

        report = cell.service.pick(look=[LOOK_PLUS_X, LOOK_MINUS_X])

        self.assertIs(AutonomousGraspOutcome.NO_VALID_GRASP, report.outcome)
        self.assertEqual([_key(LOOK_PLUS_X)], cell.joints_moved_to(), "a part found and not grasped ends it there")
        self.assertEqual({((), False)}, set(cell.calculator.held))
        self.assertEqual((_label(LOOK_PLUS_X),), report.looks)
        self.assertEqual((), report.looks_fused)
        self.assertIsNone(report.jaw_faces_seen)
        self.assertIsNone(report.hand_eye_gap_mm)


class AWristPickHandedNoLookTests(unittest.TestCase):
    """The decision of plan A1.6 for a pick handed no look at all: it perceives where the arm stands and, told not to
    move the arm, keeps the one retry it has, a fresh frame there (``PickRun`` and the console hand a wrist camera the
    configured looks or home, so this is a program's own ``pick()``)."""

    def test_it_perceives_where_the_arm_stands_and_rescans_there(self) -> None:
        cell = _Cell(grasps_on=(1,), max_attempts=3)

        report = cell.service.pick()

        self.assertTrue(report.succeeded, report.failure_summary())
        self.assertEqual([], cell.arm.motions, "a pick handed no look moved the arm")
        self.assertEqual(2, len(cell.camera.taken))
        self.assertEqual((), report.looks, "a pick handed no look names none")
        self.assertEqual(["rescan", "executed"], [a.action for a in report.pick_report.attempts])


# ---------------------------------------------------------------------------------------------------------------------
# What the report, the campaign and the record say
# ---------------------------------------------------------------------------------------------------------------------


class TheReportSaysWhatTheLooksCameToTests(unittest.TestCase):
    def test_render_and_to_dict_name_the_looks_the_faces_and_the_hand_eye(self) -> None:
        cell = _Cell(grasps_on=(1,))

        report = cell.service.pick(look=[LOOK_PLUS_X, LOOK_MINUS_X])

        self.assertEqual((True, True), report.jaw_faces_seen)
        self.assertIsInstance(report.hand_eye_gap_mm, float)
        self.assertLess(report.hand_eye_gap_mm, 2.0)
        text = report.render()
        self.assertIn(f"fused {_label(LOOK_MINUS_X)} + {_label(LOOK_PLUS_X)}", text)
        self.assertIn("contact faces of the chosen grasp: jaw 1 seen, jaw 2 seen", text)
        self.assertIn("hand-eye", text)
        wire = json.loads(json.dumps(report.to_dict()))
        self.assertEqual([_label(LOOK_PLUS_X), _label(LOOK_MINUS_X)], wire["looks"])
        self.assertEqual([_label(LOOK_MINUS_X), _label(LOOK_PLUS_X)], wire["looks_fused"])
        self.assertEqual([True, True], wire["jaw_faces_seen"])
        self.assertAlmostEqual(report.hand_eye_gap_mm, wire["hand_eye_gap_mm"])

    def test_a_contact_face_not_seen_is_said_in_the_owners_words(self) -> None:
        cell = _Cell()

        report = cell.service.pick(look=[LOOK_PLUS_X])

        self.assertIn("contact faces of the chosen grasp: jaw 1 not seen, jaw 2 seen", report.render())
        self.assertNotIn("scanned", report.render())

    def test_a_contact_face_not_seen_is_given_as_the_cause_only_where_both_faces_were_asked(self) -> None:
        """Red before: every failed wrist pick said a contact face not seen in its failure line, the console's reason,
        though off the switch a face not seen ends no pick (a 45-degree look nearly always misses one)."""
        cell = _Cell()
        seen = cell.service.pick(look=[LOOK_PLUS_X])
        self.assertEqual((False, True), seen.jaw_faces_seen)
        sentence = "the contact face at jaw 1 of the chosen grasp was not seen"

        for outcome, asked, said in ((AutonomousGraspOutcome.EXECUTION_FAILED, False, False),
                                     (AutonomousGraspOutcome.NO_VALID_GRASP, False, False),
                                     (AutonomousGraspOutcome.EXECUTION_FAILED, True, False),
                                     (AutonomousGraspOutcome.NO_VALID_GRASP, True, True)):
            with self.subTest(outcome=outcome, both_faces=asked):
                report = replace(seen, outcome=outcome, both_faces=asked)

                self.assertEqual(said, sentence in report.failure_summary(), report.failure_summary())
                self.assertIn("contact faces of the chosen grasp: jaw 1 not seen, jaw 2 seen", report.render())
                self.assertEqual(asked, report.to_dict()["both_faces"])

    def test_a_campaign_attempt_carries_the_looks_fused_the_faces_and_the_hand_eye(self) -> None:
        cell = _Cell(grasps_on=(1,))

        run = PickRun.from_service(cell.service, runs=1, recording=Recording.off(),
                                   look=[LOOK_PLUS_X, LOOK_MINUS_X]).execute()

        (attempt,) = run.attempts
        self.assertEqual((_label(LOOK_MINUS_X), _label(LOOK_PLUS_X)), attempt.looks_fused)
        self.assertEqual((True, True), attempt.jaw_faces_seen)
        self.assertLess(attempt.hand_eye_gap_mm, 2.0)
        self.assertIn(f"fused {_label(LOOK_MINUS_X)} + {_label(LOOK_PLUS_X)}", attempt.render())
        wire = attempt.to_dict()
        self.assertEqual([_label(LOOK_MINUS_X), _label(LOOK_PLUS_X)], wire["looks_fused"])
        self.assertEqual([True, True], wire["jaw_faces_seen"])

    def test_the_record_carries_the_looks_and_a_fixed_camera_record_none_of_them(self) -> None:
        keys = {"looks_visited", "looks_fused", "jaw_faces_seen", "hand_eye_gap_mm"}
        with tempfile.TemporaryDirectory() as folder:
            wrist, fixed = _Cell(grasps_on=(1,)), _Cell(fixed=True)
            wrist.service.enable_record_logging(Path(folder) / "wrist.jsonl")
            fixed.service.enable_record_logging(Path(folder) / "fixed.jsonl")

            wrist.service.pick(look=[LOOK_PLUS_X, LOOK_MINUS_X])
            fixed.service.pick()

            (on_the_wrist,) = [json.loads(line) for line in (Path(folder) / "wrist.jsonl").read_text().splitlines()]
            (fixed_record,) = [json.loads(line) for line in (Path(folder) / "fixed.jsonl").read_text().splitlines()]
        extra = on_the_wrist["extra"]
        self.assertEqual([_label(LOOK_PLUS_X), _label(LOOK_MINUS_X)], extra["looks_visited"])
        self.assertEqual([_label(LOOK_MINUS_X), _label(LOOK_PLUS_X)], extra["looks_fused"])
        self.assertEqual([True, True], extra["jaw_faces_seen"])
        self.assertLess(extra["hand_eye_gap_mm"], 2.0)
        self.assertNotIn("both_faces", extra, "a pick that did not ask for both faces says it did")
        self.assertEqual(set(), keys & set(fixed_record["extra"]), "a fixed camera's record gained look keys")

    def test_a_fixed_camera_record_names_only_the_looks_it_reached(self) -> None:
        """Red before: a fixed camera's look the arm did not reach was recorded among the looks visited. Its service
        loop names the looks it moved to, the one not reached last (``telemetry['look_refused']``)."""
        with tempfile.TemporaryDirectory() as folder:
            reached, refused = _Cell(fixed=True), _Cell(fixed=True, refuse=(LOOK_MINUS_X,))
            reached.service.enable_record_logging(Path(folder) / "reached.jsonl")
            refused.service.enable_record_logging(Path(folder) / "refused.jsonl")

            reached.service.pick(look=[LOOK_PLUS_X])
            not_reached = refused.service.pick(look=[LOOK_MINUS_X])

            (reached_record,) = [json.loads(line) for line in (Path(folder) / "reached.jsonl").read_text().splitlines()]
            (refused_record,) = [json.loads(line) for line in (Path(folder) / "refused.jsonl").read_text().splitlines()]
        self.assertEqual([_label(LOOK_PLUS_X)], reached_record["extra"]["looks_visited"])
        self.assertEqual((_label(LOOK_MINUS_X),), not_reached.looks, "the fixed camera's report is as it was")
        self.assertNotIn("looks_visited", refused_record["extra"], "a look the arm did not reach was recorded visited")


class ALookRefusedAndSkippedIsSaidTests(unittest.TestCase):
    """A look the planner refused before anything was sent is skipped and the pick goes on (addendum 2). Red before:
    only the log said so; the report, the record and the console named only the looks visited."""

    def test_a_pick_that_went_on_names_the_look_it_skipped_and_why(self) -> None:
        cell = _Cell(refuse=(LOOK_MINUS_X,))

        report = cell.service.pick(look=[LOOK_MINUS_X, LOOK_PLUS_X])

        self.assertTrue(report.succeeded, report.failure_summary())
        self.assertEqual((_label(LOOK_PLUS_X),), report.looks)
        (skipped,) = report.telemetry["looks_skipped"]
        self.assertTrue(skipped.startswith(f"{_label(LOOK_MINUS_X)}: "), skipped)
        self.assertIn("outside the cable window", skipped)
        self.assertIn(f"skipped    look {_label(LOOK_MINUS_X)}: ", report.render())
        self.assertNotIn("refused_look", report.telemetry, "a look skipped is not a look that ended the pick")

    def test_a_pick_that_failed_after_a_skipped_look_names_it(self) -> None:
        cell = _Cell(refuse=(LOOK_MINUS_X,), grasps_on=())

        report = cell.service.pick(look=[LOOK_MINUS_X, LOOK_PLUS_X])

        self.assertIs(AutonomousGraspOutcome.NO_VALID_GRASP, report.outcome)
        self.assertIn(f"skipped look {_label(LOOK_MINUS_X)}: ", report.failure_summary())
        self.assertIn("outside the cable window", report.failure_summary())

    def test_a_pick_that_reached_no_look_names_the_refusal_that_ended_it_and_skips_nothing(self) -> None:
        cell = _Cell(refuse=(LOOK_PLUS_X, LOOK_MINUS_X))

        report = cell.service.pick(look=[LOOK_PLUS_X, LOOK_MINUS_X])

        self.assertIs(AutonomousGraspOutcome.EXECUTION_FAILED, report.outcome)
        self.assertEqual(_label(LOOK_PLUS_X), report.telemetry["refused_look"])
        self.assertNotIn("looks_skipped", report.telemetry)
        self.assertNotIn("skipped", report.failure_summary())


class APickSaysNothingOfTheLastPicksLooksTests(unittest.TestCase):
    def test_a_pick_that_ends_before_it_looks_leaves_no_looks_to_read(self) -> None:
        """Red before: a pick that ended before its first look (a stop asked for at the pick's own check, or before
        the first look) left the last pick's looks on ``service.looked_around``, where a program keeping its views
        would have read the last pick's frames as this pick's."""
        for checks_passed in (0, 1):
            with self.subTest(stopped_at="the pick's own check" if checks_passed == 0 else "the first look"):
                cell = _Cell()
                cell.service.pick(look=[LOOK_PLUS_X])
                self.assertIsNotNone(cell.service.looked_around)
                asked: list[bool] = []
                cell.service.set_cancel_check(
                    lambda asked=asked, passed=checks_passed: bool(asked.append(True)) or len(asked) > passed)

                report = cell.service.pick(look=[LOOK_PLUS_X])

                self.assertIs(AutonomousGraspOutcome.CANCELLED, report.outcome)
                self.assertIsNone(cell.service.looked_around, "the last pick's looks read as this pick's")


# ---------------------------------------------------------------------------------------------------------------------
# The views of a pick, kept for training (addendum 7.7)
# ---------------------------------------------------------------------------------------------------------------------


class TheViewsOfAPickAreKeptOnlyWhenAskedTests(unittest.TestCase):
    def test_a_campaign_asked_to_keeps_each_picks_looks_where_it_says(self) -> None:
        # A file none of whose looks a camera recorded for research is format 1, byte for byte (format 2, 2026-10-09).
        from src.robot.execution.record_views import RECORD_VIEWS_FORMAT_WITHOUT_RESEARCH

        cell = _Cell(grasps_on=(1,))
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch("src.robot.execution.record_views.RECORD_VIEWS_DIR", folder):
            run = PickRun.from_service(cell.service, runs=1, recording=Recording.off(),
                                       look=[LOOK_PLUS_X, LOOK_MINUS_X], record_views=True).execute()
            (attempt,) = run.attempts
            written = Path(attempt.views_file)
            self.assertEqual([written], sorted(Path(folder).glob("*.npz")))
            self.assertIn(written.name, attempt.render())
            with np.load(written, allow_pickle=False) as views:
                self.assertEqual(RECORD_VIEWS_FORMAT_WITHOUT_RESEARCH, int(views["format"]))
                self.assertEqual([_label(LOOK_PLUS_X), _label(LOOK_MINUS_X)], views["labels"].tolist())
                self.assertEqual([f"wrist@{_label(LOOK_PLUS_X)}", f"wrist@{_label(LOOK_MINUS_X)}"],
                                 views["views"].tolist())
                for index, taken in enumerate(cell.camera.taken):
                    self.assertEqual((360, 640, 3), views[f"rgb_{index}"].shape)
                    self.assertEqual((360, 640), views[f"depth_{index}"].shape)
                    np.testing.assert_allclose(K, views[f"intrinsics_{index}"])
                    np.testing.assert_allclose(taken.to_matrix(), views[f"tool_to_base_{index}"], atol=1e-6)
                    np.testing.assert_allclose(taken.to_matrix() @ mount(), views[f"camera_to_base_{index}"],
                                               atol=1e-6)
                cloud = views["target_cloud_base_mm"]
                self.assertGreater(len(cloud), 1000)
                self.assertTrue(np.all(np.abs(cloud[:, 0]) < 25.0), "the fused target cloud is not the cube's")
        self.assertEqual(1, run.succeeded, run.render())

    def test_off_by_default_nothing_is_written(self) -> None:
        cell = _Cell(grasps_on=(1,))
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch("src.robot.execution.record_views.RECORD_VIEWS_DIR", folder):
            run = PickRun.from_service(cell.service, runs=1, recording=Recording.off(),
                                       look=[LOOK_PLUS_X, LOOK_MINUS_X]).execute()
            self.assertEqual([], list(Path(folder).iterdir()))
        self.assertEqual("", run.attempts[0].views_file)

    def test_a_fixed_camera_has_no_looks_to_keep(self) -> None:
        cell = _Cell(fixed=True)
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch("src.robot.execution.record_views.RECORD_VIEWS_DIR", folder):
            run = PickRun.from_service(cell.service, runs=1, recording=Recording.off(), record_views=True).execute()
            self.assertEqual([], list(Path(folder).iterdir()))
        self.assertEqual(1, run.succeeded)
        self.assertEqual("", run.attempts[0].views_file)

    def test_the_writer_takes_a_bare_frame_and_says_what_it_did_not_have(self) -> None:
        from src.robot.execution.record_views import record_views

        depth = np.full((4, 5), 450.0)
        bare = PerceptionFrame(depth_map=depth, intrinsics=K.copy(), segmentations=())
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(record_views((), directory=folder), "nothing to keep wrote a file")
            written = record_views([bare], directory=folder, name="a pick/of mine")
            self.assertIsNotNone(written)
            assert written is not None
            self.assertEqual(Path(folder), written.parent)
            self.assertTrue(written.name.startswith("a_pick_of_mine"), written.name)
            with np.load(written, allow_pickle=False) as views:
                self.assertEqual([""], views["labels"].tolist())
                self.assertNotIn("rgb_0", views.files)
                self.assertTrue(np.all(np.isnan(views["tool_to_base_0"])), "an unstamped frame got a tool pose")
                self.assertTrue(np.all(np.isnan(views["camera_to_base_0"])))
                self.assertEqual((0, 3), views["target_cloud_base_mm"].shape)
                np.testing.assert_allclose(depth, views["depth_0"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
