"""A program names the closing axis it wants, and only the grasps that already close that way are taken.

The owner's decision of 2026-09-30, "choose, don't twist": ``closing_axis`` names the axis, by a name ``Pose.tool_down``
takes (``x``, ``-x``, ``y``, ``-y``, ``radial``, ``-radial``, ``tangential``, ``-tangential``, a leading "+" allowed) or by
the owner's own orientation, a quaternion (x, y, z, w) or a BASE ``Pose``, whose tool +X laid onto the base XY plane is
the heading. A grasp is kept when its closing axis heads within 30 degrees of it, either way round, and is turned half a
turn about its approach where that puts its closing axis the wanted way round: the same two contact faces, the jaws
swapped. That happens to the candidate list before anything reads it, so the grasp that was judged is the grasp that is
executed and ``both_faces`` keeps working. ``GraspMotion(align_closing_to_base_x=True)``, which twists a chosen grasp
afterwards, stays the simulator's aid and is refused beside a named axis. No candidate left ends the attempt
``no_valid_grasp`` with a sentence naming the axis and the tolerance.

The owner's cell closes along base -y: ``closing_axis="-y"``, or ``Pose.tool_down(x, y, z, closing_axis="-y")``.
"""

from __future__ import annotations

import json
import logging
import math
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from src.calibration.rig_calibration import RigCalibration
from src.camera.setup.image_taking.frames import RGBDFrame
from src.config import ConfigTree
from src.geometry import Frame, Pose, Transform
from src.geometry.pose import CLOSING_AXES
from src.robot.execution.autonomous_grasp import AutonomousGraspService
from src.robot.execution.cell import Cell
from src.robot.execution.pick_run import PickOutcome as CampaignOutcome
from src.robot.execution.pick_run import PickRun, Recording
from src.robot.grasping import Scene
from src.robot.grasping.generation._debug_renderer import _DebugRenderer
from src.robot.grasping.geometry.closing_axis import (
    CLOSING_AXIS_TOLERANCE_DEG,
    NO_HEADING_WITHIN_DEG_OF_VERTICAL,
    ClosingAxis,
    closing_axis_of,
    closing_axis_refusal,
    grasps_closing_along,
    result_closing_along,
)
from src.robot.grasping.loop import pick_loop
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator, PickOutcome
from src.robot.grasping.loop.progress import PickStage
from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy
from src.robot.grasping.motion.grasp_motion import GraspMotion, build_execution_policy
from src.robot.grasping.multiview.unseen_side import jaw_faces_seen
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.types.perception import PerceptionFrame
from src.robot.perception.locator import Locator
from tests._helpers import _FakeArm, _FakePerception, _perception_frame, _ScriptedCalculator
from tests._wrist_views import HEIGHT, WIDTH, camera_looking_at, render
from tests.test_a_located_part_is_looked_at_from_every_look import (
    EAST,
    NORTH,
    SOUTH,
    WEST,
    _arm,
    _cell,
    _key,
    _locator,
    _moved_to,
    _robot,
    _Wrist,
)
from tests.test_a_wrist_pick_looks_until_its_grasp_is_safe import (
    HAND_E,
    LOOK_MINUS_X,
    LOOK_PLUS_X,
    _Cell,
    _keys,
)
from tests.test_locator import _backend, _Owner

#: Where the owner's cell picks: the marker's place on the bench.
_AT = (-130.0, -700.0, 50.0)
#: The owner's orientation for a pick: straight down, closing along base -y, as a program teaches it.
_OWNERS = Pose.tool_down(*_AT, closing_axis="-y")


def _grasp(heading_deg: float, *, at: tuple[float, float, float] = _AT, score: float = 0.9,
           frame: GraspFrame = GraspFrame.BASE, lean_deg: float = 0.0) -> GraspPoint:
    """A grasp at ``at``, straight down, closing along ``heading_deg`` about base Z, its closing axis leaning
    ``lean_deg`` up out of the horizontal (its approach leaning with it, so the two stay square)."""
    heading, lean = math.radians(heading_deg), math.radians(lean_deg)
    axis = np.array([math.cos(heading) * math.cos(lean), math.sin(heading) * math.cos(lean), math.sin(lean)])
    approach = np.array([math.cos(heading) * math.sin(lean), math.sin(heading) * math.sin(lean), -math.cos(lean)])
    return GraspPoint(position=np.array(at), approach=approach, axis=axis, grip_width_mm=40.0, score=score,
                      frame=frame, label="part")


def _heading_deg(vector: Any) -> float:
    return math.degrees(math.atan2(float(vector[1]), float(vector[0])))


def _console_dummy() -> Any:
    return ConfigTree.from_directory(profile="console_dummy").load()


# ---------------------------------------------------------------------------------------------------------------------
# The axis a program names
# ---------------------------------------------------------------------------------------------------------------------


class AProgramNamesTheAxisTests(unittest.TestCase):
    def test_every_name_pose_tool_down_takes_is_taken_and_a_plus_changes_nothing(self) -> None:
        for name in CLOSING_AXES:
            with self.subTest(name=name):
                self.assertEqual(ClosingAxis(name=name), closing_axis_of(name))
        self.assertEqual(closing_axis_of("y"), closing_axis_of("+y"))
        self.assertEqual(closing_axis_of("radial"), closing_axis_of("+radial"))

    def test_a_name_it_does_not_know_is_refused_naming_the_choices(self) -> None:
        for name in ("z", "+-x", "", "-", "Y"):
            with self.subTest(name=name), self.assertRaises(ValueError) as refused:
                closing_axis_of(name)
            self.assertIn("-tangential", str(refused.exception))
        with self.assertRaisesRegex(TypeError, "quaternion"):
            closing_axis_of(3)

    def test_the_owners_orientation_names_the_heading_of_its_tool_x(self) -> None:
        """``Pose.tool_down(..., closing_axis="-y")`` closes along -y: its quaternion, or the pose, is the name."""
        for given in (_OWNERS, _OWNERS.quaternion_xyzw, tuple(_OWNERS.quaternion_xyzw), list(_OWNERS.quaternion_xyzw)):
            with self.subTest(given=type(given).__name__):
                axis = closing_axis_of(given)
                self.assertAlmostEqual(270.0, axis.heading_at(0.0, 0.0), places=9)
                self.assertAlmostEqual(closing_axis_of("-y").heading_at(*_AT[:2]) % 360.0,
                                       axis.heading_at(*_AT[:2]), places=9)
        turned = Pose.tool_down(*_AT, yaw_deg=30.0)
        self.assertAlmostEqual(30.0, closing_axis_of(turned).heading_at(0.0, 0.0), places=9)

    def test_an_orientation_whose_tool_x_points_up_or_down_names_no_heading_and_is_refused(self) -> None:
        """A tool +X within 10 degrees of the vertical names no heading about base Z: refused with a sentence."""
        for off_vertical, refused in ((0.0, True), (9.0, True), (11.0, False)):
            with self.subTest(off_vertical=off_vertical):
                # A turn about base Y takes the tool +X from base +X down toward -Z: 90 degrees less the lean is how far.
                half = math.radians(90.0 - off_vertical) / 2.0
                quaternion = (0.0, math.sin(half), 0.0, math.cos(half))
                if not refused:
                    self.assertAlmostEqual(0.0, closing_axis_of(quaternion).heading_at(0.0, 0.0), places=6)
                    continue
                with self.assertRaises(ValueError) as caught:
                    closing_axis_of(quaternion)
                self.assertIn("names no heading", str(caught.exception))
                self.assertIn(f"{NO_HEADING_WITHIN_DEG_OF_VERTICAL:g} deg", str(caught.exception))

    def test_a_closing_axis_built_by_hand_is_a_known_name_or_a_heading_never_both(self) -> None:
        for keywords in ({}, {"name": "-y", "heading_deg": 270.0}, {"name": "+y"}, {"name": "z"},
                         {"heading_deg": float("nan")}):
            with self.subTest(**keywords), self.assertRaises(ValueError):
                ClosingAxis(**keywords)  # type: ignore[arg-type]
        self.assertEqual(270.0, ClosingAxis(heading_deg=270.0).heading_at(1.0, 2.0))

    def test_an_orientation_that_is_no_quaternion_or_not_in_base_is_refused(self) -> None:
        for given in ((0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (float("nan"), 0.0, 0.0, 1.0), ("a", "b", "c", "d")):
            with self.subTest(given=given), self.assertRaisesRegex(ValueError, "quaternion"):
                closing_axis_of(given)
        with self.assertRaisesRegex(ValueError, "BASE"):
            closing_axis_of(_OWNERS.with_frame(Frame.CAMERA))


# ---------------------------------------------------------------------------------------------------------------------
# Only the grasps that close along it are taken, each turned to close that way round
# ---------------------------------------------------------------------------------------------------------------------


class OnlyTheGraspsThatCloseAlongItAreTakenTests(unittest.TestCase):
    def test_a_name_keeps_the_grasps_that_close_along_it_either_way_round(self) -> None:
        candidates = (_grasp(0.0, score=0.95), _grasp(90.0, score=0.9), _grasp(45.0, score=0.85),
                      _grasp(-90.0, score=0.8))

        kept, why = grasps_closing_along(candidates, closing_axis_of("-y"))

        self.assertEqual([0.9, 0.8], [grasp.score for grasp in kept], "best first, as they were")
        for grasp in kept:
            np.testing.assert_allclose(grasp.axis, (0.0, -1.0, 0.0), atol=1e-12)
        self.assertEqual(2, len(why), "the x grasp and the 45 degree grasp were left out, and said")

    def test_29_degrees_off_is_taken_and_31_is_not(self) -> None:
        self.assertEqual(30.0, CLOSING_AXIS_TOLERANCE_DEG)
        wanted = closing_axis_of("-y")
        for off, taken in ((29.0, True), (31.0, False), (-29.0, True), (-31.0, False), (180.0 + 29.0, True),
                           (180.0 - 31.0, False)):
            with self.subTest(off=off):
                kept, _ = grasps_closing_along((_grasp(-90.0 + off),), wanted)
                self.assertEqual(taken, bool(kept))
                if taken:
                    self.assertLessEqual(abs(_heading_deg(kept[0].axis) + 90.0), 29.0 + 1e-9, "not signed along -y")

    def test_exactly_30_degrees_off_is_within_it_either_way_round(self) -> None:
        """Within 30 degrees takes 30 degrees itself: the bound is the owner's figure, not a degree short of it."""
        wanted = closing_axis_of("-y")
        for off in (30.0, -30.0, 180.0 + 30.0, 180.0 - 30.0):
            with self.subTest(off=off):
                kept, why = grasps_closing_along((_grasp(-90.0 + off),), wanted)
                self.assertEqual((1, ()), (len(kept), why))

    def test_a_grasp_that_closes_the_other_way_round_is_turned_half_a_turn_about_its_approach(self) -> None:
        """The same two contact faces, the jaws swapped: seen from +y only, jaw 2's face (+y) is the one seen; turned
        to close along -y, jaw 1's is."""
        plus_y = _grasp(90.0, at=(_AT[0], _AT[1], 20.0))  # at the middle of the 40 mm cube below
        (minus_y,), _ = grasps_closing_along((plus_y,), closing_axis_of("-y"))

        np.testing.assert_allclose(minus_y.axis, -plus_y.axis, atol=1e-12)
        np.testing.assert_array_equal(minus_y.approach, plus_y.approach)
        np.testing.assert_array_equal(minus_y.position, plus_y.position)
        self.assertEqual((plus_y.score, plus_y.grip_width_mm, plus_y.label), (minus_y.score, minus_y.grip_width_mm,
                                                                                minus_y.label))
        before, after = plus_y.pose().to_matrix()[:3, :3], minus_y.pose().to_matrix()[:3, :3]
        turn = before.T @ after
        np.testing.assert_allclose(turn, np.diag([-1.0, -1.0, 1.0]), atol=1e-12, err_msg="not a half turn about +Z")

        cube = _cube_seen_from_plus_y()
        seen = [jaw_faces_seen(cube, centre_mm=grasp.position, approach=grasp.approach, closing_axis=grasp.axis,
                               grip_width_mm=40.0, pad_ahead_mm=HAND_E.pad_ahead_mm,
                               pad_behind_mm=HAND_E.pad_length_mm - HAND_E.pad_ahead_mm,
                               finger_width_mm=HAND_E.finger_width_mm, support_height_mm=0.0).as_pair()
                for grasp in (plus_y, minus_y)]
        self.assertEqual([(False, True), (True, False)], seen)

    def test_the_owners_orientation_takes_the_same_grasps_as_its_name(self) -> None:
        candidates = tuple(_grasp(heading, score=1.0 - index / 100.0)
                           for index, heading in enumerate((-125.0, -95.0, -65.0, 0.0, 65.0, 85.0, 118.0, 180.0)))

        by_name, _ = grasps_closing_along(candidates, closing_axis_of("-y"))
        for given in (_OWNERS, _OWNERS.quaternion_xyzw):
            by_orientation, _ = grasps_closing_along(candidates, closing_axis_of(given))
            self.assertEqual([grasp.score for grasp in by_name], [grasp.score for grasp in by_orientation])
            for mine, theirs in zip(by_name, by_orientation):
                np.testing.assert_allclose(mine.axis, theirs.axis, atol=1e-12)
        self.assertEqual([candidates[index].score for index in (1, 2, 4, 5, 6)], [grasp.score for grasp in by_name])

    def test_radial_and_tangential_are_read_at_each_grasps_own_place(self) -> None:
        """Two grasps round the base, each closing along the line from the base through it: both close radially, the
        sign pointing away from the base; neither closes tangentially. One over the base's own axis is left out."""
        east, north = (400.0, 0.0, 50.0), (0.0, 400.0, 50.0)
        candidates = (_grasp(180.0, at=east), _grasp(90.0, at=north))

        radial, _ = grasps_closing_along(candidates, closing_axis_of("radial"))
        tangential, why = grasps_closing_along(candidates, closing_axis_of("tangential"))

        self.assertEqual(2, len(radial))
        np.testing.assert_allclose(radial[0].axis, (1.0, 0.0, 0.0), atol=1e-12)
        np.testing.assert_allclose(radial[1].axis, (0.0, 1.0, 0.0), atol=1e-12)
        self.assertEqual((), tangential)
        self.assertEqual(2, len(why))
        over_the_base, why = grasps_closing_along((_grasp(0.0, at=(0.0, 0.0, 50.0)),), closing_axis_of("-radial"))
        self.assertEqual((), over_the_base)
        self.assertIn("vertical axis", why[0])

    def test_a_grasp_whose_closing_axis_names_no_heading_is_left_out_with_that_reason(self) -> None:
        wanted = closing_axis_of("-y")
        steep, _ = grasps_closing_along((_grasp(-90.0, lean_deg=85.0),), wanted)
        leaning, _ = grasps_closing_along((_grasp(-90.0, lean_deg=75.0),), wanted)

        self.assertEqual((), steep)
        self.assertEqual(1, len(leaning), "a closing axis 15 degrees off the vertical still names a heading")
        _, why = grasps_closing_along((_grasp(-90.0, lean_deg=85.0),), wanted)
        self.assertIn("names no heading", why[0])

    def test_a_grasp_that_is_not_in_base_is_left_out(self) -> None:
        kept, why = grasps_closing_along((_grasp(-90.0, frame=GraspFrame.CAMERA),), closing_axis_of("-y"))
        self.assertEqual((), kept)
        self.assertIn("BASE", why[0])

    def test_a_result_with_none_left_says_no_valid_grasp_naming_the_axis_and_the_tolerance(self) -> None:
        result = GraspResult(candidates=(_grasp(0.0, score=0.9), _grasp(20.0, score=0.8)), top_score=0.9,
                             telemetry={"final": 2})

        taken = result_closing_along(result, closing_axis_of("-y"))

        self.assertFalse(taken.is_success)
        self.assertEqual((GraspFailureReason.NO_VALID_GRASP,), taken.reasons)
        said = closing_axis_refusal(taken)
        self.assertIn("'-y'", said)
        self.assertIn("30 deg", said)
        self.assertIn("none of the 2", said)
        self.assertEqual(2, taken.telemetry["final"], "the calculator's telemetry goes with it")
        self.assertEqual("", closing_axis_refusal(result))

    def test_a_result_keeps_its_best_first_and_its_top_score_is_the_best_kept(self) -> None:
        result = GraspResult(candidates=(_grasp(0.0, score=0.9), _grasp(90.0, score=0.8)), top_score=0.9)

        taken = result_closing_along(result, closing_axis_of("-y"))

        self.assertEqual(1, len(taken.candidates))
        self.assertEqual(0.8, taken.top_score)
        np.testing.assert_allclose(taken.best.axis, (0.0, -1.0, 0.0), atol=1e-12)  # type: ignore[union-attr]
        untouched = GraspResult(candidates=(_grasp(-90.0),), top_score=0.9)
        self.assertIs(untouched, result_closing_along(untouched, closing_axis_of("-y")))


def _cube_seen_from_plus_y(centre: tuple[float, float, float] = _AT, size: float = 40.0) -> np.ndarray:
    """A cube standing on the bench under ``centre``, as a look from its +y side sees it: its top and its +y face."""
    half, step = size / 2.0, 1.0
    across = np.arange(-half, half + 1e-9, step)
    up = np.arange(0.0, size + 1e-9, step)
    top = [(centre[0] + x, centre[1] + y, size) for x in across for y in across]
    face = [(centre[0] + x, centre[1] + half, z) for x in across for z in up]
    return np.asarray(top + face, dtype=np.float64)


# ---------------------------------------------------------------------------------------------------------------------
# A motion names it; it is refused beside the simulator's twist
# ---------------------------------------------------------------------------------------------------------------------


class _Hand:
    max_width_mm = 50.0
    min_width_mm = 0.0


class AMotionNamesItTests(unittest.TestCase):
    def test_a_motion_takes_a_name_or_an_orientation_and_is_still_a_value(self) -> None:
        self.assertEqual({"closing_axis": "-y"}, GraspMotion(closing_axis="-y").to_dict())
        from_array = GraspMotion(closing_axis=_OWNERS.quaternion_xyzw)
        from_tuple = GraspMotion(closing_axis=tuple(_OWNERS.quaternion_xyzw))
        self.assertEqual(from_array, from_tuple)
        self.assertEqual(hash(from_array), hash(from_tuple))
        self.assertEqual(GraspMotion(closing_axis=_OWNERS), GraspMotion(closing_axis=_OWNERS))
        self.assertNotIn("closing_axis", GraspMotion().to_dict())

    def test_a_motion_refuses_what_names_no_axis_before_any_cell_exists(self) -> None:
        for given, error in (("z", ValueError), ((0.0, 0.7071068, 0.0, 0.7071068), ValueError),
                             (_OWNERS.with_frame(Frame.CAMERA), ValueError), (3, TypeError), (None, TypeError)):
            with self.subTest(given=given), self.assertRaises(error):
                GraspMotion(closing_axis=given)  # type: ignore[arg-type]

    def test_the_simulators_twist_beside_a_named_axis_is_refused(self) -> None:
        with self.assertRaises(ValueError) as refused:
            GraspMotion(closing_axis="-y", align_closing_to_base_x=True)
        self.assertIn("align_closing_to_base_x", str(refused.exception))
        self.assertIn("closing_axis", str(refused.exception))
        self.assertEqual({"closing_axis": "-y", "align_closing_to_base_x": False},
                         GraspMotion(closing_axis="-y", align_closing_to_base_x=False).to_dict())
        with self.assertRaisesRegex(ValueError, "align_closing_to_base_x"):
            GraspExecutionPolicy(arm=_FakeArm(), closing_axis=closing_axis_of("-y"),  # type: ignore[arg-type]
                                 align_closing_to_base_x=True)

    def test_the_one_builder_hands_the_axis_to_the_policy_the_pick_loop_reads(self) -> None:
        policy = build_execution_policy(GraspMotion(closing_axis="-y"), arm=_FakeArm(), gripper=_Hand(),  # type: ignore[arg-type]
                                        standoff_mm=80.0, retreat_mm=100.0, base_frame_required=True)
        self.assertEqual(closing_axis_of("-y"), policy.closing_axis)
        unset = build_execution_policy(GraspMotion(), arm=_FakeArm(), gripper=_Hand(),  # type: ignore[arg-type]
                                       standoff_mm=80.0, retreat_mm=100.0, base_frame_required=True)
        self.assertIsNone(unset.closing_axis)
        by_hand = GraspExecutionPolicy(arm=_FakeArm(), closing_axis="-y")  # type: ignore[arg-type]
        self.assertEqual(closing_axis_of("-y"), by_hand.closing_axis)

    def test_the_service_builds_its_policy_with_the_axis(self) -> None:
        arm = _FakeArm()
        service = AutonomousGraspService.from_components(
            arm=arm, calculator=_ScriptedCalculator([GraspResult()]), perception=_FakePerception([_perception_frame()]),  # type: ignore[arg-type]
            mode="easy", gripper=_Hand(), motion=GraspMotion(closing_axis=_OWNERS))  # type: ignore[arg-type]
        self.assertEqual(closing_axis_of(_OWNERS), service.runtime.orchestrator.policy.closing_axis)


# ---------------------------------------------------------------------------------------------------------------------
# The pick loop takes only those grasps, before anything reads them
# ---------------------------------------------------------------------------------------------------------------------


def _loop(candidates: tuple[GraspPoint, ...], closing_axis: Any = None, **wiring: Any) -> tuple[Any, Any, Any]:
    arm = _FakeArm()
    perception = _FakePerception([_perception_frame()])
    policy = GraspExecutionPolicy(arm=arm, closing_axis=closing_axis)  # type: ignore[arg-type]
    result = GraspResult(candidates=candidates, top_score=float(candidates[0].score) if candidates else 0.0)
    events: list[Any] = []
    orchestrator = BinPickingOrchestrator(
        arm=arm, calculator=_ScriptedCalculator([result]), perception=perception,  # type: ignore[arg-type]
        policy=policy, max_attempts=3, on_progress=events.append, **wiring)
    return orchestrator, arm, (perception, events)


def _closed_along(arm: Any) -> float:
    """The heading of the tool +X the arm closed with, from the last approach waypoint."""
    closing = [pose for pose in arm.move_to_calls if str(pose.label).startswith("approach")][-1]
    return _heading_deg(closing.to_matrix()[:3, 0])


class ThePickLoopTakesOnlyThoseGraspsTests(unittest.TestCase):
    def test_the_grasp_executed_is_the_best_that_closes_along_the_axis_turned_that_way_round(self) -> None:
        candidates = (_grasp(0.0, score=0.95), _grasp(90.0, score=0.9))
        for closing_axis, heading in ((None, 0.0), ("-y", -90.0), ("y", 90.0), (_OWNERS.quaternion_xyzw, -90.0)):
            with self.subTest(closing_axis=closing_axis):
                orchestrator, arm, _ = _loop(candidates, closing_axis)

                report = orchestrator.run()

                self.assertIs(PickOutcome.EXECUTED, report.outcome)
                self.assertAlmostEqual(heading, _closed_along(arm), places=6)
                executed = report.executed_grasp
                assert executed is not None
                self.assertAlmostEqual(heading, _heading_deg(executed.best.axis), places=6)  # type: ignore[union-attr]

    def test_no_grasp_along_the_axis_ends_the_attempt_no_valid_grasp_naming_the_axis_and_nothing_moves(self) -> None:
        orchestrator, arm, (perception, events) = _loop((_grasp(0.0), _grasp(45.0)), "-y")

        with self.assertLogs("src.robot.grasping.loop.pick_loop", level="WARNING") as said:
            report = orchestrator.run()

        self.assertIs(PickOutcome.RESCANNED_EXHAUSTED, report.outcome)
        (attempt,) = report.attempts
        self.assertEqual("exhausted", attempt.action)
        self.assertEqual((GraspFailureReason.NO_VALID_GRASP,), attempt.reasons)
        assert attempt.withheld is not None
        self.assertIn("'-y'", attempt.withheld)
        self.assertIn("30 deg", attempt.withheld)
        self.assertEqual([], arm.move_to_calls)
        self.assertEqual(1, perception.calls, "an axis no grasp closes along is not rescanned for")
        self.assertTrue(any("'-y'" in line and "30 deg" in line for line in said.output), said.output)
        (no_candidate,) = [event for event in events if event.stage is PickStage.NO_CANDIDATE]
        self.assertEqual(("no_valid_grasp",), tuple(no_candidate.reasons))

    def test_every_step_after_the_calculator_sees_only_the_kept_grasps_turned_the_wanted_way_round(self) -> None:
        """The reranks, the deep ranker's shadow and the approach check are handed only the grasps that close along -y,
        each already turned: the approach check falls back to the second of them, never to the x grasp."""
        candidates = (_grasp(0.0, score=0.97), _grasp(90.0, score=0.95), _grasp(0.0, score=0.93),
                      _grasp(-80.0, score=0.9))
        orchestrator, arm, _ = _loop(candidates, "-y", ranking_blend_config=object(),
                                     uncertainty_rerank_config=object(), approach_path_policy=object(),
                                     mode_label="dense_clutter", _approach_path_modes=frozenset({"dense_clutter"}))
        handed: dict[str, list[Any]] = {"blend": [], "rerank": [], "deep": [], "sweep": []}

        def blend(candidates: Any, **_: Any) -> Any:
            handed["blend"].extend(candidates)
            return candidates, None

        def rerank(candidates: Any, **_: Any) -> Any:
            handed["rerank"].extend(candidates)
            return candidates, None

        def deep(_self: Any, result: Any, *_args: Any) -> None:
            handed["deep"].extend(result.candidates)

        def sweep(candidate: Any, _cloud: Any) -> bool:
            handed["sweep"].append(candidate)
            return len(handed["sweep"]) > 1

        with mock.patch.object(pick_loop, "maybe_blend_rerank_candidates", side_effect=blend), \
                mock.patch.object(pick_loop, "maybe_uncertainty_rerank_candidates", side_effect=rerank), \
                mock.patch.object(BinPickingOrchestrator, "_stamp_deep_ranker", autospec=True, side_effect=deep), \
                mock.patch.object(BinPickingOrchestrator, "_approach_sweep_is_clear", side_effect=sweep):
            report = orchestrator.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        for step, grasps in handed.items():
            with self.subTest(step=step):
                self.assertTrue(grasps)
                self.assertTrue(all(float(grasp.axis[1]) < -0.9 for grasp in grasps), [g.axis for g in grasps])
        self.assertEqual([0.95, 0.9], [grasp.score for grasp in handed["blend"]])
        self.assertAlmostEqual(-80.0, _closed_along(arm), places=6)

    def test_the_simulators_twist_set_on_the_policy_later_is_refused_before_anything_moves(self) -> None:
        orchestrator, arm, (perception, events) = _loop((_grasp(-90.0),), "-y")
        orchestrator.policy.align_closing_to_base_x = True  # type: ignore[union-attr]

        with self.assertRaises(ValueError) as refused:
            orchestrator.run()

        self.assertIn("align_closing_to_base_x", str(refused.exception))
        self.assertEqual(([], 0), (arm.move_to_calls, perception.calls))
        self.assertEqual([], [event for event in events if event.stage is PickStage.PICK_STARTED])

    def test_without_an_axis_the_loop_is_the_one_it_always_was(self) -> None:
        result = GraspResult(candidates=(_grasp(0.0), _grasp(90.0)), top_score=0.9)
        orchestrator, _, _ = _loop(result.candidates)
        with mock.patch.object(pick_loop, "result_closing_along", side_effect=AssertionError("read")):
            self.assertIs(PickOutcome.EXECUTED, orchestrator.run().outcome)

    def test_a_calculator_that_cannot_draw_its_overlay_again_has_it_dropped_where_the_axis_changed_its_grasps(
            self) -> None:
        """A calculator with no ``redraw_debug_image`` keeps no picture that ranks first a grasp the axis left out; one
        whose grasps the axis kept as they were keeps its picture, as one with no axis does."""
        for closing_axis, candidates, kept in (("-y", (_grasp(0.0, score=0.95), _grasp(-90.0, score=0.9)), False),
                                               ("-y", (_grasp(-90.0),), True),
                                               (None, (_grasp(0.0), _grasp(-90.0)), True)):
            with self.subTest(closing_axis=closing_axis, candidates=len(candidates)):
                orchestrator, _, _ = _loop(candidates, closing_axis)
                orchestrator.calculator.last_debug_image_png = b"the calculator's picture"  # type: ignore[attr-defined]

                self.assertIs(PickOutcome.EXECUTED, orchestrator.run().outcome)

                self.assertEqual(b"the calculator's picture" if kept else None,
                                 orchestrator.calculator.last_debug_image_png)  # type: ignore[attr-defined]

    def test_the_simulators_twist_set_later_as_any_true_value_is_refused_before_anything_moves(self) -> None:
        """The policy twists wherever its switch reads true (``if self.align_closing_to_base_x``), so the refusal reads
        it the same way: 1 or numpy's True beside a named axis, or beside both_faces, is refused as True is."""
        for twist in (1, np.True_):
            for closing_axis, both_faces, said in (("-y", False, "closing_axis"), (None, True, "both_faces")):
                with self.subTest(twist=twist, both_faces=both_faces):
                    orchestrator, arm, (perception, _) = _loop((_grasp(-90.0),), closing_axis, both_faces=both_faces)
                    orchestrator.policy.align_closing_to_base_x = twist  # type: ignore[union-attr]

                    with self.assertRaises(ValueError) as refused:
                        orchestrator.run()

                    self.assertIn(said, str(refused.exception))
                    self.assertIn("align_closing_to_base_x", str(refused.exception))
                    self.assertEqual(([], 0), (arm.move_to_calls, perception.calls))
            with self.subTest(twist=twist, built=True), self.assertRaisesRegex(ValueError, "align_closing_to_base_x"):
                GraspExecutionPolicy(arm=_FakeArm(), closing_axis="-y", align_closing_to_base_x=twist)  # type: ignore[arg-type]

    def test_an_axis_set_on_the_policy_later_that_names_none_is_refused_before_the_arm_moves_to_a_look(self) -> None:
        """A program may set the axis on a policy already built; one that names no axis is refused as the pick begins,
        before the arm is sent to its first look, not once a look has been ranked."""
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X))
        cell.policy.closing_axis = "z"  # type: ignore[attr-defined]

        with self.assertRaisesRegex(ValueError, "not a name Pose.tool_down takes"):
            cell.run()
        with self.assertRaisesRegex(ValueError, "not a name Pose.tool_down takes"):
            cell.orchestrator.look_around()

        self.assertEqual([], cell.joints_moved_to())
        self.assertEqual([], cell.camera.taken)

    def test_a_part_the_axis_left_no_grasp_is_said_when_another_part_fails_for_its_own_reason(self) -> None:
        """Two parts in the frame: the first has no grasp along -y, the second no mask a grasp is found on. The frame
        fails on the second, whose reason asks for another frame, as it always did; every attempt still says that the
        first was refused for its axis, and how far, which the report's line then names."""
        first, second = np.zeros((32, 32), dtype=bool), np.zeros((32, 32), dtype=bool)
        first[4:12, 4:12], second[18:28, 18:28] = True, True
        frame = PerceptionFrame(depth_map=np.full((32, 32), 500.0), intrinsics=_perception_frame().intrinsics,
                                segmentations=(SimpleNamespace(mask=first, label="first"),
                                               SimpleNamespace(mask=second, label="second")))
        by_label = {"first": GraspResult(candidates=(_grasp(0.0),), top_score=0.9),
                    "second": GraspResult(reasons=(GraspFailureReason.EMPTY_MASK,
                                                   GraspFailureReason.RESCAN_RECOMMENDED))}
        calculator = SimpleNamespace(compute_result=lambda seg, *_args, **_kwargs: by_label[seg.label])
        arm, perception = _FakeArm(), _FakePerception([frame])
        orchestrator = BinPickingOrchestrator(
            arm=arm, calculator=calculator, perception=perception,  # type: ignore[arg-type]
            policy=GraspExecutionPolicy(arm=arm, closing_axis="-y"), max_attempts=3)  # type: ignore[arg-type]

        report = orchestrator.run()

        self.assertIs(PickOutcome.RESCANNED_EXHAUSTED, report.outcome)
        self.assertEqual(3, len(report.attempts))
        for attempt in report.attempts:
            with self.subTest(attempt=attempt.attempt_index):
                self.assertIn(GraspFailureReason.EMPTY_MASK, attempt.reasons)
                assert attempt.withheld is not None
                self.assertIn("'-y'", attempt.withheld)
                self.assertIn("30 deg", attempt.withheld)
        self.assertEqual([], arm.move_to_calls)


# ---------------------------------------------------------------------------------------------------------------------
# both_faces judges the grasp the jaws close on
# ---------------------------------------------------------------------------------------------------------------------


class BothFacesJudgesTheGraspTheJawsCloseOnTests(unittest.TestCase):
    """The wrist scene's calculator grasps the cube closing along base +x, jaw 1 on its -x face and jaw 2 on its +x face.
    Asked for -x, the same grasp is turned half a turn: jaw 1 on the +x face, jaw 2 on the -x face."""

    def test_a_grasp_whose_turned_faces_were_both_seen_is_gripped_turned(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), both_faces=True)
        cell.policy.closing_axis = "-x"  # type: ignore[attr-defined]

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual(_keys(LOOK_PLUS_X, LOOK_MINUS_X), cell.joints_moved_to())
        (executed,) = cell.policy.executed
        np.testing.assert_allclose(executed.axis, (-1.0, 0.0, 0.0), atol=1e-12)
        self.assertEqual((True, True), cell.looked.jaw_faces_seen)

    def test_a_grasp_whose_turned_faces_were_not_both_seen_is_refused_naming_the_turned_jaw(self) -> None:
        """From +x alone the cube's -x face is not seen: without the axis that is jaw 1's face, turned to close along -x
        it is jaw 2's, and the refusal names jaw 2."""
        cell = _Cell(looks=(LOOK_PLUS_X,), both_faces=True)
        cell.policy.closing_axis = "-x"  # type: ignore[attr-defined]

        report = cell.run()

        self.assertIs(PickOutcome.RESCANNED_EXHAUSTED, report.outcome)
        (attempt,) = report.attempts
        self.assertEqual("faces_unseen", attempt.action)
        self.assertEqual([], cell.policy.executed)
        self.assertIn("the contact face at jaw 2 of the chosen grasp was not seen", cell.looked.withheld)
        self.assertEqual((True, False), cell.looked.jaw_faces_seen)

    def test_a_named_axis_no_grasp_closes_along_looks_on_and_grips_nothing(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), both_faces=True)
        cell.policy.closing_axis = "y"  # type: ignore[attr-defined]

        report = cell.run()

        self.assertIs(PickOutcome.RESCANNED_EXHAUSTED, report.outcome)
        self.assertEqual(_keys(LOOK_PLUS_X, LOOK_MINUS_X), cell.joints_moved_to()[:2])
        self.assertEqual([], cell.policy.executed)
        (attempt,) = report.attempts
        self.assertEqual((GraspFailureReason.NO_VALID_GRASP,), attempt.reasons)
        assert attempt.withheld is not None
        self.assertIn("'y'", attempt.withheld)


# ---------------------------------------------------------------------------------------------------------------------
# PickRun and the service pass it through, on the rehearsal cell
# ---------------------------------------------------------------------------------------------------------------------


class APickRunPassesItThroughTests(unittest.TestCase):
    """The rehearsal cell (``console_dummy``: the real service, pick loop and calculator on a dummy arm and hand) grasps
    its box closing along base +y first and along base +x next."""

    @staticmethod
    def _run(motion: GraspMotion) -> Any:
        cell = Cell.rehearsal(_console_dummy().robot, motion=motion)
        return PickRun.from_cell(cell, runs=1, recording=Recording.off()).execute()

    def test_each_pick_closes_along_the_axis_its_motion_names(self) -> None:
        for closing_axis, heading in (("-y", -90.0), ("y", 90.0), ("-x", 180.0), ("x", 0.0),
                                      (_OWNERS.quaternion_xyzw, -90.0), (_OWNERS, -90.0)):
            with self.subTest(closing_axis=closing_axis):
                report = self._run(GraspMotion(closing_axis=closing_axis))

                (attempt,) = report.attempts
                self.assertIs(CampaignOutcome.SUCCEEDED, attempt.outcome, report.render())
                assert attempt.grasp_pose is not None
                closing = attempt.grasp_pose.to_matrix()[:3, 0]
                self.assertAlmostEqual(0.0, math.remainder(_heading_deg(closing) - heading, 360.0), places=6)

    def test_the_control_without_an_axis_closes_along_its_first_grasp(self) -> None:
        (attempt,) = self._run(GraspMotion()).attempts
        assert attempt.grasp_pose is not None
        self.assertAlmostEqual(90.0, _heading_deg(attempt.grasp_pose.to_matrix()[:3, 0]), places=6)

    def test_an_axis_no_grasp_closes_along_fails_the_pick_and_the_line_names_it(self) -> None:
        report = self._run(GraspMotion(closing_axis=Pose.tool_down(*_AT, yaw_deg=45.0)))

        (attempt,) = report.attempts
        self.assertIs(CampaignOutcome.FAILED, attempt.outcome)
        self.assertIn("no_valid_grasp", attempt.detail)
        self.assertIn("30 deg", attempt.detail)
        self.assertIn("45.0 deg", attempt.detail)
        self.assertIsNone(attempt.grasp_pose)


class TheOverlayShowsTheGraspsTheAxisTookTests(unittest.TestCase):
    """The grasp overlay (``last_debug_image_png``: what a campaign pins on the primary camera's window, what example 12
    writes as a picture of what the camera saw and gripped) shows the grasps the pick chose among, each the way round it
    closes, best first: drawn again once the axis chose, never a grasp the axis left out ranked first. On the rehearsal
    cell, whose box the calculator ranks closing along base +y first and along base +x next."""

    @staticmethod
    def _run(motion: GraspMotion, cell: Any = None) -> "tuple[Any, list[list[GraspPoint]], list[Any], Any]":
        drawn: list[list[GraspPoint]] = []
        pngs: list[Any] = []
        draw = _DebugRenderer.draw

        def spy(renderer: Any, **keywords: Any) -> Any:
            drawn.append(list(keywords["candidates"]))
            pngs.append(draw(renderer, **keywords))
            return pngs[-1]

        cell = cell if cell is not None else Cell.rehearsal(_console_dummy().robot, motion=motion)
        cell.build()
        cell.service.enable_debug_image_rendering(True)
        with mock.patch.object(_DebugRenderer, "draw", autospec=True, side_effect=spy):
            report = PickRun.from_cell(cell, runs=1, recording=Recording.off()).execute()
        return report, drawn, pngs, cell.service.last_debug_image_png

    def test_the_overlay_shows_only_the_grasps_along_the_axis_each_the_way_round_it_closes(self) -> None:
        for closing_axis, heading in (("x", 0.0), ("-y", -90.0)):
            with self.subTest(closing_axis=closing_axis):
                report, drawn, pngs, shown = self._run(GraspMotion(closing_axis=closing_axis))

                (attempt,) = report.attempts
                self.assertIs(CampaignOutcome.SUCCEEDED, attempt.outcome, report.render())
                self.assertTrue(drawn[-1])
                for grasp in drawn[-1]:
                    self.assertAlmostEqual(0.0, math.remainder(_heading_deg(grasp.axis) - heading, 360.0), places=6)
                self.assertIsNotNone(shown)
                self.assertIs(pngs[-1], shown, "the overlay the campaign pins is not the one drawn last")
                assert attempt.grasp_pose is not None
                gripped = _heading_deg(attempt.grasp_pose.to_matrix()[:3, 0])
                self.assertAlmostEqual(0.0, math.remainder(gripped - _heading_deg(drawn[-1][0].axis), 360.0), places=6)

    def test_an_axis_no_grasp_closes_along_shows_the_part_with_no_grasp(self) -> None:
        _, drawn, pngs, shown = self._run(GraspMotion(closing_axis=Pose.tool_down(*_AT, yaw_deg=45.0)))

        self.assertTrue(drawn[0], "the control: the calculator ranked grasps")
        self.assertEqual([], drawn[-1])
        self.assertIsNotNone(shown)
        self.assertIs(pngs[-1], shown)

    def test_without_an_axis_the_overlay_is_drawn_once_as_it_always_was(self) -> None:
        _, drawn, pngs, shown = self._run(GraspMotion())

        self.assertEqual(1, len(drawn))
        self.assertIs(pngs[0], shown)

    def test_a_compute_that_drew_no_overlay_leaves_none_to_draw_again(self) -> None:
        """What an overlay is drawn again from is the last compute's alone: after a compute that drew none, nothing is
        drawn again, so no picture of an earlier frame comes back under a later attempt."""
        cell = Cell.rehearsal(_console_dummy().robot, motion=GraspMotion(closing_axis="x"))
        self._run(GraspMotion(closing_axis="x"), cell)
        calculator = cell.service.runtime.orchestrator.calculator
        self.assertIsNotNone(calculator.redraw_debug_image([]), "the control: the last compute drew one")

        calculator.compute_result(SimpleNamespace(mask=np.zeros((8, 8), dtype=bool), label="none"),
                                  np.zeros((8, 8)))

        self.assertIsNone(calculator.redraw_debug_image([]))
        self.assertIsNone(calculator.last_debug_image_png)


# ---------------------------------------------------------------------------------------------------------------------
# The Locator path: Scene.grasps(closing_axis=...)
# ---------------------------------------------------------------------------------------------------------------------


def _block(points: int = 900, seed: int = 0) -> np.ndarray:
    """A 60 x 40 x 40 mm block on the bench: it closes across its 40 mm along base y best, across 60 mm along x next."""
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(400.0, 460.0, points), rng.uniform(-20.0, 20.0, points),
                            rng.uniform(0.0, 40.0, points)])


class TheSceneTakesOnlyThoseGraspsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = Scene.from_cloud(_block(), support_height_mm=0.0)

    def test_only_the_grasps_along_the_axis_are_taken_each_turned_that_way_round(self) -> None:
        every = self.scene.grasps(max_candidates=50).candidates
        self.assertEqual({0.0, 90.0}, {round(abs(math.remainder(_heading_deg(c.closing_axis), 180.0)), 0) % 180.0
                                        for c in every}, "the control: the block closes both ways")
        for closing_axis, heading in (("x", 0.0), ("-x", 180.0), ("y", 90.0), ("-y", -90.0)):
            with self.subTest(closing_axis=closing_axis):
                taken = self.scene.grasps(max_candidates=50, closing_axis=closing_axis).candidates
                self.assertTrue(taken)
                for grasp in taken:
                    self.assertLess(abs(math.remainder(_heading_deg(grasp.closing_axis) - heading, 360.0)), 1.0)
                    twin = [c for c in every if np.array_equal(c.position_mm, grasp.position_mm)
                            and np.array_equal(c.approach, grasp.approach) and c.score == grasp.score]
                    self.assertTrue(twin, "a grasp the generator did not find")
                    self.assertTrue(any(np.allclose(np.abs(c.closing_axis), np.abs(grasp.closing_axis)) for c in twin))

    def test_the_owners_orientation_takes_the_same_grasps_as_its_name(self) -> None:
        by_name = self.scene.grasps(closing_axis="-y").to_dict()
        for given in (_OWNERS, _OWNERS.quaternion_xyzw):
            by_orientation = self.scene.grasps(closing_axis=given).to_dict()
            self.assertEqual(by_name["candidates"], by_orientation["candidates"])

    def test_the_best_along_the_axis_is_found_below_the_cap(self) -> None:
        """The y grasps outrank the x grasps: capped at two, the two best x grasps come back, not none."""
        best_x = [c.score for c in self.scene.grasps(max_candidates=50, closing_axis="x").candidates][:2]

        capped = self.scene.grasps(max_candidates=2, closing_axis="x").candidates

        self.assertEqual(best_x, [c.score for c in capped])
        self.assertEqual(2, len(capped))

    def test_none_along_the_axis_is_an_answer_that_names_the_axis_and_the_tolerance(self) -> None:
        grasps = self.scene.grasps(closing_axis=Pose.tool_down(430.0, 0.0, 50.0, yaw_deg=45.0))

        self.assertEqual((), grasps.candidates)
        self.assertIsNone(grasps.best)
        text = grasps.render()
        self.assertIn("30 deg", text)
        self.assertIn("45.0 deg", text)
        self.assertNotIn("too sparse", text)
        self.assertTrue(text.isascii())
        payload = grasps.to_dict()
        self.assertEqual(json.loads(json.dumps(payload)), payload)
        self.assertIn("30 deg", payload["withheld"])

    def test_without_an_axis_the_scene_answers_as_it_always_did(self) -> None:
        unset = self.scene.grasps()
        self.assertNotIn("withheld", unset.to_dict())
        self.assertEqual(unset.to_dict(), self.scene.grasps().to_dict())
        self.assertEqual("", unset.withheld)


#: CAMERA to BASE of the owner's D415 where the east look holds it: 45 degrees up, 450 mm off the cube.
_EAST_CAMERA = camera_looking_at(bearing_deg=0.0)


class _FixedAtEast(_Wrist):
    """The owner's D415 fixed where the east look holds it, eye to hand: every frame is rendered from there, and it sees
    the cube's top and its east face."""

    rig_id = "fixed"

    def calibration(self) -> RigCalibration:
        return RigCalibration(rig_id="fixed", mounting_mode="eye_to_hand", artifact_path="eth.json",
                              transform=Transform.from_matrix(_EAST_CAMERA, from_frame=Frame.CAMERA,
                                                              to_frame=Frame.BASE))

    def grab(self) -> RGBDFrame:
        depth, self._hit = render(_EAST_CAMERA, self.boxes)
        self.grabs += 1
        return RGBDFrame(color=np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8), depth=depth,
                         captured_at_s=float(self.grabs))


def _fixed_at_east(arm: Any) -> Locator:
    camera = _FixedAtEast(arm)
    return Locator.from_parts(camera=camera, backend=camera, attempts=1)


class TheLocatorJudgesTheGraspsAlongTheAxisTests(unittest.TestCase):
    """A look around judges the grasp the program will grip: ``look_around(..., closing_axis=)`` as
    ``scene.grasps(closing_axis=)`` takes it. The 40 mm cube's x grasp closes on the faces its east and west looks show,
    its y grasp on the ones its north and south looks show."""

    @staticmethod
    def _look(*looks: Any, **keywords: Any) -> tuple[Any, Any]:
        built = _arm(*looks)
        located = _locator(_Wrist(built)).look_around("a part", [look.joints for look in looks], robot=_robot(built),
                                                      both_faces=True, **keywords)
        return located, built

    def test_the_looks_go_on_until_both_faces_of_the_grasp_along_the_axis_were_seen(self) -> None:
        control, built = self._look(EAST, WEST, NORTH, SOUTH)
        self.assertEqual([_key(EAST), _key(WEST)], _moved_to(built), "the control: the x grasp stops at the west look")

        located, built = self._look(EAST, WEST, NORTH, SOUTH, closing_axis="-y")

        self.assertEqual([_key(EAST), _key(WEST), _key(NORTH), _key(SOUTH)], _moved_to(built))
        self.assertEqual(((True, True), ""), (located.jaw_faces_seen, located.refused))
        best = located.scene(0, _cell()).grasps(closing_axis="-y").best
        assert best is not None
        self.assertAlmostEqual(-90.0, _heading_deg(best.closing_axis), delta=1.0)
        by_orientation = located.scene(0, _cell()).grasps(closing_axis=_OWNERS).best
        assert by_orientation is not None
        np.testing.assert_array_equal(best.closing_axis, by_orientation.closing_axis)

    def test_a_part_whose_grasp_along_the_axis_no_look_showed_both_faces_of_is_refused(self) -> None:
        """The east and west looks show both faces of the x grasp and neither of the y grasp's: asked for -y, the part is
        refused, where without the axis it would be gripped on faces the -y grasp does not close on."""
        control, _ = self._look(EAST, WEST)
        self.assertEqual("", control.refused)

        located, _ = self._look(EAST, WEST, closing_axis="-y")

        self.assertIn("were not seen", located.refused)
        self.assertEqual((False, False), located.jaw_faces_seen)

    def test_a_look_whose_part_has_no_grasp_along_the_axis_says_which_axis_and_how_far(self) -> None:
        across = Pose.tool_down(0.0, -700.0, 20.0, yaw_deg=45.0)  # the cube closes along x and y, 45 degrees off it

        with self.assertLogs("src.robot.perception.locator", level="INFO") as said:
            located, built = self._look(EAST, WEST, closing_axis=across)

        self.assertEqual([_key(EAST), _key(WEST)], _moved_to(built))
        self.assertTrue(any("no grasp on the part (closing_axis" in line and "30 deg" in line for line in said.output),
                        said.output)
        self.assertIsNone(located.scene(0, _cell()).grasps(closing_axis=across).best)

    def test_a_value_that_names_no_axis_raises_before_anything_moves(self) -> None:
        built = _arm(EAST)
        with self.assertRaises(ValueError):
            _locator(_Wrist(built)).look_around("a part", [EAST.joints], robot=_robot(built), closing_axis="z")
        self.assertEqual([], _moved_to(built))

    def test_the_parts_scene_takes_the_axis_the_looks_judged_and_no_other(self) -> None:
        """The grasp gripped is the grasp judged: the Located keeps the axis its looks judged the part along, and the
        part's scene takes it when asked for none, takes it named another way, and refuses any other axis before it
        chooses a grasp."""
        located, _ = self._look(EAST, WEST, NORTH, SOUTH, closing_axis="-y")
        scene = located.scene(0, _cell())

        best = scene.grasps().best

        assert best is not None
        self.assertAlmostEqual(-90.0, _heading_deg(best.closing_axis), delta=1.0)
        same = scene.grasps(closing_axis=_OWNERS).best
        assert same is not None
        np.testing.assert_array_equal(best.closing_axis, same.closing_axis)
        for other in ("y", "x", Pose.tool_down(*_AT, yaw_deg=45.0)):
            with self.subTest(other=other), self.assertRaises(ValueError) as refused:
                scene.grasps(closing_axis=other)
            self.assertIn("look_around", str(refused.exception))
        self.assertEqual(closing_axis_of("-y"), located.closing_axis)
        self.assertIn("'-y'", located.render())
        self.assertEqual({"name": "-y", "heading_deg": None}, located.to_dict()["closing_axis"])

    def test_a_part_judged_along_any_axis_refuses_a_scene_asked_for_one(self) -> None:
        """Without an axis the looks judge the best grasp of any: from east and west the x grasp, both its faces seen. A
        scene of the part asked for -y would grip a grasp no look judged, neither of whose faces those looks showed:
        refused, naming look_around. Asked for none, it takes the grasp the looks judged."""
        located, _ = self._look(EAST, WEST)
        self.assertEqual(((True, True), "", None), (located.jaw_faces_seen, located.refused, located.closing_axis))
        scene = located.scene(0, _cell())

        with self.assertRaises(ValueError) as refused:
            scene.grasps(closing_axis="-y")

        self.assertIn("look_around", str(refused.exception))
        judged = scene.grasps().best
        assert judged is not None
        self.assertLess(abs(math.remainder(_heading_deg(judged.closing_axis), 180.0)), 1.0, "not the x grasp")
        self.assertNotIn("closing_axis", located.to_dict())

    def test_one_locate_judges_no_grasp_and_its_scene_takes_any_axis(self) -> None:
        located = Locator.from_parts(camera=_Owner(), backend=_backend("red cube")).locate("a red cube")

        grasps = located.scene(0, _cell()).grasps(closing_axis="-y")

        self.assertIsNone(located.closing_axis)
        self.assertTrue(grasps.candidates)
        for grasp in grasps.candidates:
            self.assertAlmostEqual(-90.0, _heading_deg(grasp.closing_axis), delta=1.0)

    def test_a_fixed_camera_judges_the_faces_of_the_grasp_along_the_axis(self) -> None:
        """A camera fixed where the east look stands sees the cube's top and its east face, and nothing moves. The best
        grasp of any axis closes along y, on two faces it does not see; asked for x, the grasp judged closes on the east
        face it sees (jaw 2's) and the west face it does not (jaw 1's), and the refusal names jaw 1 alone. Without
        both_faces the Located still keeps the axis, which the part's scene then takes."""
        built = _arm(EAST)
        control = _fixed_at_east(built).look_around("a part", robot=_robot(built), both_faces=True)
        self.assertEqual((False, False), control.jaw_faces_seen)

        located = _fixed_at_east(built).look_around("a part", robot=_robot(built), both_faces=True, closing_axis="x")

        self.assertEqual((False, True), located.jaw_faces_seen)
        self.assertIn("the contact face at jaw 1 of the chosen grasp was not seen", located.refused)
        self.assertEqual(closing_axis_of("x"), located.closing_axis)
        self.assertEqual([], built.motions)
        unjudged = _fixed_at_east(built).look_around("a part", robot=_robot(built), closing_axis="x")
        best = unjudged.scene(0, _cell()).grasps().best
        assert best is not None
        self.assertAlmostEqual(0.0, _heading_deg(best.closing_axis), delta=1.0)
        with self.assertRaisesRegex(ValueError, "look_around"):
            unjudged.scene(0, _cell()).grasps(closing_axis="y")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    unittest.main()
