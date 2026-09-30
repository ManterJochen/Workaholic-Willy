"""Every camera grasp closes the way round the cell's hand and camera naturally stand: chosen, never dropped or tilted.

The owner's final decision of 2026-09-30. Camera grasps stay completely free: any closing direction, any tilt. The cell
names how its hand and camera naturally stand, ``robot.natural_closing_axis``: a name ``Pose.tool_down`` takes (``x``,
``-x``, ``y``, ``-y``, ``radial``, ``-radial``, ``tangential``, ``-tangential``, a leading "+" allowed) or a quaternion
(x, y, z, w) whose tool +X is the natural closing direction. Of a grasp's two equivalent wrist turns, half a turn about its
approach apart (the same two contact faces, the jaws swapped), the pick takes the one whose tool +X lies nearest the
natural direction at the grasp's own place. It does so where the closing-axis filter runs, so the judgement, both_faces,
the approach check and the execution all see the grasp that is gripped, and ``Scene.grasps`` and the Locator do the same
from the robot config. A program's own ``closing_axis`` keeps its own sign rule and takes precedence; when one is named the
calculator is asked for 36 candidates and the best 12 along the axis are kept, as ``Scene.grasps`` keeps them. Fixed poses
built through the cell take the natural orientation where the program names none: ``robot.tool_down(x, y, z)``. Unset,
nothing changes.

The owner's cell: ``robot.natural_closing_axis: "-y"``, the tool +X along base -y, where the D415's image stands upright.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import replace
from typing import Any
from unittest import mock

import numpy as np
import yaml
from pydantic import ValidationError

from src.config.schema.robot import RobotConfig
from src.geometry import Pose
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.execution.cell import Cell
from src.robot.execution.pick_run import PickOutcome as CampaignOutcome
from src.robot.execution.pick_run import PickRun, Recording
from src.robot.execution.robot import Robot
from src.robot.grasping import Scene
from src.robot.grasping import scene as scene_module
from src.robot.grasping.generation import calculator as calculator_module
from src.robot.grasping.geometry.closing_axis import (
    CANDIDATES_THE_AXIS_CHOOSES_AMONG,
    closing_axis_of,
    grasps_closing_toward,
    natural_closing_axis_of,
    result_closing_toward,
)
from src.robot.grasping.loop import pick_loop
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator, PickOutcome
from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy
from src.robot.grasping.motion.grasp_motion import GraspMotion
from src.robot.grasping.scoring.success_probability import RankingBlendTelemetry
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame
from tests._helpers import _FakeArm, _FakePerception, _perception_frame
from tests.test_a_located_part_is_looked_at_from_every_look import (
    NORTH,
    SOUTH,
    _arm,
    _cell,
    _locator,
    _robot,
    _Wrist,
)
from tests.test_a_pick_closes_along_the_axis_its_program_names import (
    _AT,
    _OWNERS,
    _block,
    _closed_along,
    _console_dummy,
    _grasp,
    _heading_deg,
    _loop,
)
from tests.test_a_scene_plans_the_trees_hand import _replace
from tests.test_a_wrist_pick_looks_until_its_grasp_is_safe import LOOK_MINUS_X, LOOK_PLUS_X, _Cell, _keys
from tests.test_robot_parts import _loaded_after

#: The owner's natural orientation, as his cell names it.
_MINUS_Y = closing_axis_of("-y")


def _off(heading_deg: float, wanted_deg: float) -> float:
    """How far ``heading_deg`` turns from ``wanted_deg`` about base Z, degrees in [-180, 180]."""
    return math.remainder(heading_deg - wanted_deg, 360.0)


# ---------------------------------------------------------------------------------------------------------------------
# Of the two ways round, the one nearer the natural direction
# ---------------------------------------------------------------------------------------------------------------------


class TheNearerWayRoundIsTakenTests(unittest.TestCase):
    def test_a_grasp_closing_along_plus_y_is_turned_and_one_along_minus_y_is_left_as_it_is(self) -> None:
        plus_y, minus_y = _grasp(90.0, score=0.9), _grasp(-90.0, score=0.8)

        turned, kept = grasps_closing_toward((plus_y, minus_y), _MINUS_Y)

        np.testing.assert_allclose(turned.axis, (0.0, -1.0, 0.0), atol=1e-12)
        np.testing.assert_array_equal(turned.approach, plus_y.approach)
        np.testing.assert_array_equal(turned.position, plus_y.position)
        self.assertEqual((plus_y.score, plus_y.grip_width_mm, plus_y.label, plus_y.frame),
                         (turned.score, turned.grip_width_mm, turned.label, turned.frame))
        self.assertIs(minus_y, kept, "a grasp already the nearer way round is the same grasp")

    def test_45_degrees_off_either_side_is_turned_to_the_nearer_way_round(self) -> None:
        for heading, nearer in ((45.0, -135.0), (135.0, -45.0), (-45.0, -45.0), (-135.0, -135.0), (100.0, -80.0)):
            with self.subTest(heading=heading):
                (grasp,) = grasps_closing_toward((_grasp(heading),), _MINUS_Y)
                self.assertAlmostEqual(0.0, _off(_heading_deg(grasp.axis), nearer), places=9)
                self.assertLess(abs(_off(_heading_deg(grasp.axis), -90.0)), 90.0)

    def test_a_grasp_closing_straight_across_it_is_left_as_it_is(self) -> None:
        """Both ways round lie equally near: nothing to choose, and the grasp is the one the calculator ranked."""
        for heading in (0.0, 180.0):
            with self.subTest(heading=heading):
                grasp = _grasp(heading)
                self.assertIs(grasp, grasps_closing_toward((grasp,), _MINUS_Y)[0])

    def test_no_grasp_is_left_out_and_none_is_tilted_whatever_its_lean(self) -> None:
        """A grasp leaning 75 degrees, one whose closing axis stands 5 degrees off the vertical, one 10 degrees off the
        other way round: each is kept, turned half a turn about its own approach, which stays where it was."""
        candidates = (_grasp(90.0, lean_deg=75.0, score=0.9), _grasp(100.0, lean_deg=85.0, score=0.8),
                      _grasp(80.0, score=0.7))

        turned = grasps_closing_toward(candidates, _MINUS_Y)

        self.assertEqual([0.9, 0.8, 0.7], [grasp.score for grasp in turned])
        for before, after in zip(candidates, turned):
            np.testing.assert_allclose(after.axis, -before.axis, atol=1e-12)
            np.testing.assert_array_equal(after.approach, before.approach)
            turn = before.pose().to_matrix()[:3, :3].T @ after.pose().to_matrix()[:3, :3]
            np.testing.assert_allclose(turn, np.diag([-1.0, -1.0, 1.0]), atol=1e-9, err_msg="not a half turn")

    def test_radial_and_tangential_are_read_at_each_grasps_own_place(self) -> None:
        east, north = (400.0, 0.0, 50.0), (0.0, 400.0, 50.0)
        toward_the_base = (_grasp(180.0, at=east), _grasp(-90.0, at=north))

        radial = grasps_closing_toward(toward_the_base, closing_axis_of("radial"))
        inward = grasps_closing_toward(toward_the_base, closing_axis_of("-radial"))
        tangential = grasps_closing_toward((_grasp(90.0, at=east), _grasp(180.0, at=north)),
                                           closing_axis_of("tangential"))

        np.testing.assert_allclose(radial[0].axis, (1.0, 0.0, 0.0), atol=1e-12)
        np.testing.assert_allclose(radial[1].axis, (0.0, 1.0, 0.0), atol=1e-12)
        self.assertEqual(list(toward_the_base), list(inward))
        np.testing.assert_allclose(tangential[0].axis, (0.0, -1.0, 0.0), atol=1e-12)
        np.testing.assert_allclose(tangential[1].axis, (1.0, 0.0, 0.0), atol=1e-12)
        over_the_base = _grasp(0.0, at=(0.0, 0.0, 50.0))
        self.assertIs(over_the_base, grasps_closing_toward((over_the_base,), closing_axis_of("-radial"))[0],
                      "over the base's own axis radial names no direction, and the grasp is left as ranked")

    def test_a_quaternion_turns_every_grasp_as_the_name_it_was_taught_from(self) -> None:
        candidates = tuple(_grasp(heading, score=1.0 - index / 100.0)
                           for index, heading in enumerate((-125.0, -95.0, -65.0, 0.0, 65.0, 85.0, 118.0, 180.0)))

        by_name = grasps_closing_toward(candidates, _MINUS_Y)

        for given in (_OWNERS.quaternion_xyzw, list(_OWNERS.quaternion_xyzw), _OWNERS):
            with self.subTest(given=type(given).__name__):
                by_orientation = grasps_closing_toward(candidates, closing_axis_of(given))
                for mine, theirs in zip(by_name, by_orientation):
                    np.testing.assert_allclose(mine.axis, theirs.axis, atol=1e-12)

    def test_a_grasp_that_is_not_in_base_is_left_as_it_is(self) -> None:
        grasp = _grasp(90.0, frame=GraspFrame.CAMERA)
        self.assertIs(grasp, grasps_closing_toward((grasp,), _MINUS_Y)[0])

    def test_a_result_keeps_its_order_its_score_and_its_telemetry(self) -> None:
        result = GraspResult(candidates=(_grasp(90.0, score=0.9), _grasp(-60.0, score=0.8)), top_score=0.9,
                             telemetry={"final": 2})

        turned = result_closing_toward(result, _MINUS_Y)

        self.assertEqual([0.9, 0.8], [grasp.score for grasp in turned.candidates])
        self.assertEqual((0.9, {"final": 2}, result.reasons), (turned.top_score, turned.telemetry, turned.reasons))
        np.testing.assert_allclose(turned.candidates[0].axis, (0.0, -1.0, 0.0), atol=1e-12)
        self.assertIs(turned.candidates[1], result.candidates[1])
        untouched = GraspResult(candidates=(_grasp(-90.0), _grasp(0.0)), top_score=0.9)
        self.assertIs(untouched, result_closing_toward(untouched, _MINUS_Y))
        empty = GraspResult()
        self.assertIs(empty, result_closing_toward(empty, _MINUS_Y))

    def test_every_grasp_of_a_result_is_turned_not_only_its_best(self) -> None:
        """Whatever comes first later (a rerank, the approach check's fallback), it was turned as well."""
        headings = (90.0, 60.0, 135.0, 100.0)
        result = GraspResult(candidates=tuple(_grasp(heading, score=0.9 - index / 100.0)
                                              for index, heading in enumerate(headings)), top_score=0.9)

        turned = result_closing_toward(result, _MINUS_Y)

        for grasp, nearer in zip(turned.candidates, (-90.0, -120.0, -45.0, -80.0)):
            self.assertAlmostEqual(0.0, _off(_heading_deg(grasp.axis), nearer), places=9)


# ---------------------------------------------------------------------------------------------------------------------
# The cell names it: robot.natural_closing_axis
# ---------------------------------------------------------------------------------------------------------------------


class TheCellNamesItsNaturalOrientationTests(unittest.TestCase):
    def test_unset_by_default_and_in_the_shipped_trees(self) -> None:
        self.assertIsNone(RobotConfig().natural_closing_axis)
        self.assertIsNone(natural_closing_axis_of(RobotConfig()))
        self.assertIsNone(natural_closing_axis_of(_console_dummy().robot))
        self.assertIsNone(natural_closing_axis_of(object()), "anything with no such key names none")

    def test_a_name_or_a_quaternion_as_a_yaml_file_writes_them(self) -> None:
        for written, name in (("-y", "-y"), ("y", "y"), ("+y", "y"), ("x", "x"), ("-radial", "-radial"),
                              ("tangential", "tangential")):
            with self.subTest(written=written):
                config = RobotConfig.model_validate(yaml.safe_load(f"natural_closing_axis: {written}"))
                self.assertEqual(closing_axis_of(name), natural_closing_axis_of(config))
        quaternion = ", ".join(repr(float(value)) for value in _OWNERS.quaternion_xyzw)
        config = RobotConfig.model_validate(yaml.safe_load(f"natural_closing_axis: [{quaternion}]"))
        taught = natural_closing_axis_of(config)
        assert taught is not None
        self.assertAlmostEqual(0.0, _off(taught.heading_at(*_AT[:2]), -90.0), places=9)

    def test_what_names_no_axis_is_refused_at_load_naming_the_key(self) -> None:
        for value in ("z", "", "-", [0.0, 0.0, 0.0, 0.0], [0.0, 0.7071068, 0.0, 0.7071068], [1.0, 0.0, 0.0], 3):
            with self.subTest(value=value), self.assertRaises(ValidationError) as refused:
                RobotConfig.model_validate({"natural_closing_axis": value})
            self.assertIn("natural_closing_axis", str(refused.exception))
        with self.assertRaises(ValidationError) as vertical:
            RobotConfig.model_validate({"natural_closing_axis": [0.0, 0.7071068, 0.0, 0.7071068]})
        self.assertIn("names no heading", str(vertical.exception))


#: What reading the key must not load: the grasping package, and OpenCV, which that package pulls in.
_READING_MUST_NOT_LOAD = ("cv2", "src.robot.grasping")


class ReadingItLoadsNoGraspingStackTests(unittest.TestCase):
    """The key is read by the one reader of a closing axis, which lives with the poses it reads
    (``src.geometry.closing_axis``) and which the grasping module hands on: reading a tree that names it pulls in no
    grasping package and no OpenCV (``src/config/loader.py``), and neither does a pose built through the cell."""

    def test_the_grasping_module_hands_on_the_one_reader(self) -> None:
        from src.geometry import closing_axis as geometry_reader  # noqa: PLC0415
        from src.robot.grasping.geometry import closing_axis as grasping_reader  # noqa: PLC0415

        for name in ("NO_HEADING_WITHIN_DEG_OF_VERTICAL", "ClosingAxis", "closing_axis_of", "natural_closing_axis_of"):
            with self.subTest(name=name):
                self.assertIs(getattr(geometry_reader, name), getattr(grasping_reader, name))

    def test_a_tree_that_names_it_is_read_without_the_grasping_package_or_opencv(self) -> None:
        quaternion = [float(value) for value in _OWNERS.quaternion_xyzw]
        script = "\n".join((
            "from src.config.schema.robot import RobotConfig",
            "RobotConfig(natural_closing_axis='-y')",
            f"RobotConfig.model_validate({{'natural_closing_axis': {quaternion!r}}})",
        ))
        self.assertEqual([], _loaded_after(script, _READING_MUST_NOT_LOAD))

    def test_a_pose_built_through_the_cell_loads_no_grasping_package(self) -> None:
        script = "\n".join((
            "from src.config.schema.robot import RobotConfig",
            "from src.robot.drivers.dummy.arm import DummyRobotArm",
            "from src.robot.execution.robot import Robot",
            "robot = Robot.from_parts(arm=DummyRobotArm(), gripper=None, lock_key=None,",
            "                         robot_config=RobotConfig(natural_closing_axis='-y'))",
            "robot.tool_down(-130.0, -700.0, 50.0)",
        ))
        self.assertEqual([], _loaded_after(script, ("src.robot.grasping",)))

    def test_the_probe_sees_the_grasping_package_where_it_is(self) -> None:
        """The self-failing control: the module that turns grasps imports the grasping package, and OpenCV with it."""
        self.assertEqual(list(_READING_MUST_NOT_LOAD),
                         _loaded_after("import src.robot.grasping.geometry.closing_axis", _READING_MUST_NOT_LOAD))


# ---------------------------------------------------------------------------------------------------------------------
# The pick loop turns every grasp there, before anything reads it
# ---------------------------------------------------------------------------------------------------------------------


class ThePickLoopTurnsEveryGraspTests(unittest.TestCase):
    def test_the_grasp_executed_is_the_best_turned_the_nearer_way_round_and_none_is_left_out(self) -> None:
        candidates = (_grasp(90.0, score=0.95), _grasp(0.0, score=0.9))
        for natural, heading in ((None, 90.0), (_MINUS_Y, -90.0), (closing_axis_of("y"), 90.0),
                                 (closing_axis_of(_OWNERS.quaternion_xyzw), -90.0), (closing_axis_of("x"), 90.0)):
            with self.subTest(natural=str(natural)):
                orchestrator, arm, _ = _loop(candidates, natural_closing_axis=natural)

                report = orchestrator.run()

                self.assertIs(PickOutcome.EXECUTED, report.outcome)
                self.assertAlmostEqual(heading, _closed_along(arm), places=6)
                executed = report.executed_grasp
                assert executed is not None
                self.assertEqual([0.95, 0.9], [grasp.score for grasp in executed.candidates])
                self.assertAlmostEqual(heading, _heading_deg(executed.candidates[0].axis), places=6)

    def test_a_programs_own_axis_wins_and_keeps_its_own_sign(self) -> None:
        orchestrator, arm, _ = _loop((_grasp(0.0, score=0.95), _grasp(-90.0, score=0.9)), "y",
                                     natural_closing_axis=_MINUS_Y)

        report = orchestrator.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertAlmostEqual(90.0, _closed_along(arm), places=6)
        executed = report.executed_grasp
        assert executed is not None
        self.assertEqual([0.9], [grasp.score for grasp in executed.candidates], "the filter still leaves the x grasp out")

    def test_without_a_natural_orientation_the_loop_is_the_one_it_always_was(self) -> None:
        orchestrator, arm, _ = _loop((_grasp(90.0), _grasp(0.0)))
        with mock.patch.object(pick_loop, "result_closing_toward", side_effect=AssertionError("read")):
            self.assertIs(PickOutcome.EXECUTED, orchestrator.run().outcome)
        self.assertAlmostEqual(90.0, _closed_along(arm), places=6)

    def test_the_overlay_is_drawn_again_over_the_turned_grasps_and_only_where_one_was_turned(self) -> None:
        for natural, redrawn in ((_MINUS_Y, True), (closing_axis_of("y"), False)):
            with self.subTest(natural=str(natural)):
                orchestrator, _, _ = _loop((_grasp(90.0), _grasp(0.0)), natural_closing_axis=natural)
                drawn: list[list[Any]] = []
                orchestrator.calculator.last_debug_image_png = b"the calculator's picture"  # type: ignore[attr-defined]
                orchestrator.calculator.redraw_debug_image = lambda grasps: drawn.append(list(grasps))  # type: ignore[attr-defined]

                self.assertIs(PickOutcome.EXECUTED, orchestrator.run().outcome)

                self.assertEqual(1 if redrawn else 0, len(drawn))
                if redrawn:
                    np.testing.assert_allclose(drawn[0][0].axis, (0.0, -1.0, 0.0), atol=1e-12)

    def test_every_grasp_is_turned_not_only_the_best(self) -> None:
        orchestrator, _, _ = _loop((_grasp(90.0, score=0.95), _grasp(60.0, score=0.9), _grasp(135.0, score=0.85)),
                                   natural_closing_axis=_MINUS_Y)

        report = orchestrator.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        executed = report.executed_grasp
        assert executed is not None
        for grasp, nearer in zip(executed.candidates, (-90.0, -120.0, -45.0), strict=True):
            self.assertAlmostEqual(0.0, _off(_heading_deg(grasp.axis), nearer), places=6)

    def test_a_rerank_that_puts_a_later_grasp_first_executes_it_turned_as_well(self) -> None:
        """The reranks run after the turn and may put any candidate first: the one they put first was turned too."""
        orchestrator, arm, _ = _loop((_grasp(90.0, score=0.95), _grasp(60.0, score=0.9)), natural_closing_axis=_MINUS_Y)

        def reversed_order(candidates: Any, **_: Any) -> Any:
            return tuple(reversed(candidates)), _blend_that_put_another_first(len(candidates))

        with mock.patch.object(pick_loop, "maybe_blend_rerank_candidates", side_effect=reversed_order):
            report = orchestrator.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual([0.9, 0.95], [grasp.score for grasp in report.executed_grasp.candidates])  # type: ignore[union-attr]
        self.assertAlmostEqual(-120.0, _closed_along(arm), places=6)

    def test_beside_the_simulators_twist_it_turns_nothing_and_the_twist_moves_as_it_always_did(self) -> None:
        """``align_closing_to_base_x`` turns every grasp about base Z to close along base X itself, whichever way round
        it came, so a turn would change only the side a leaning approach leans to. The natural orientation stands down
        there: the arm is sent exactly the waypoints it is sent without one."""
        sent: dict[str, list[Any]] = {}
        for natural in (None, _MINUS_Y):
            orchestrator, arm, _ = _loop((_grasp(90.0, lean_deg=30.0),), natural_closing_axis=natural)
            orchestrator.policy.align_closing_to_base_x = True  # type: ignore[union-attr]

            self.assertIs(PickOutcome.EXECUTED, orchestrator.run().outcome)

            sent[str(natural)] = [pose.to_matrix() for pose in arm.move_to_calls]
        self.assertTrue(sent["None"])
        np.testing.assert_allclose(np.stack(sent[str(_MINUS_Y)]), np.stack(sent["None"]), atol=1e-12)

    def test_one_set_by_hand_that_names_no_axis_is_refused_naming_it_before_anything_moves(self) -> None:
        for entry in ("run", "look_around"):
            with self.subTest(entry=entry):
                cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), natural_closing_axis="z")

                with self.assertRaises(ValueError) as refused:
                    getattr(cell.orchestrator, entry)()

                self.assertIn("natural_closing_axis", str(refused.exception))
                self.assertEqual([], cell.arm.motions)


def _blend_that_put_another_first(candidates: int) -> RankingBlendTelemetry:
    """What the probability blend says when it fired and changed the grasp that comes first."""
    return RankingBlendTelemetry(
        enabled=True, applied=True, skip_reason=None, weight=0.5, mode_label="dense_clutter", lifecycle_phase="active",
        n_candidates=candidates, top1_changed=True, winner_geometric_score=None, winner_predicted_probability=None,
        winner_final_score=None, pre_blend_top1_geometric_score=None)


class BothFacesJudgesTheGraspTurnedTheNaturalWayRoundTests(unittest.TestCase):
    """The wrist scene's calculator grasps the cube closing along base +x: jaw 1 on its -x face, jaw 2 on its +x face.
    A hand that naturally closes along -x takes the same grasp half a turn round: jaw 1 on the +x face, jaw 2 on the -x
    face. The faces judged are the faces the jaws close on."""

    def test_a_grasp_whose_faces_were_both_seen_is_gripped_the_natural_way_round(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), both_faces=True, natural_closing_axis=closing_axis_of("-x"))

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual(_keys(LOOK_PLUS_X, LOOK_MINUS_X), cell.joints_moved_to())
        (executed,) = cell.policy.executed
        np.testing.assert_allclose(executed.axis, (-1.0, 0.0, 0.0), atol=1e-12)
        self.assertEqual((True, True), cell.looked.jaw_faces_seen)

    def test_from_one_side_the_refusal_names_the_jaw_the_turned_grasp_puts_on_the_face_no_view_showed(self) -> None:
        """From +x alone the cube's -x face is not seen: as ranked that is jaw 1's face; turned to close along -x it is
        jaw 2's, and the refusal names jaw 2, as the filter's does for ``closing_axis="-x"``."""
        said: dict[str, Any] = {}
        for name, wiring in (("as ranked", {}), ("natural", {"natural_closing_axis": closing_axis_of("-x")})):
            cell = _Cell(looks=(LOOK_PLUS_X,), both_faces=True, **wiring)
            report = cell.run()
            self.assertIs(PickOutcome.RESCANNED_EXHAUSTED, report.outcome)
            self.assertEqual([], cell.policy.executed)
            said[name] = (cell.looked.jaw_faces_seen, cell.looked.withheld)
        filtered = _Cell(looks=(LOOK_PLUS_X,), both_faces=True)
        filtered.policy.closing_axis = "-x"  # type: ignore[attr-defined]
        filtered.run()

        self.assertEqual((False, True), said["as ranked"][0])
        self.assertIn("the contact face at jaw 1 of the chosen grasp was not seen", said["as ranked"][1])
        self.assertEqual((True, False), said["natural"][0])
        self.assertIn("the contact face at jaw 2 of the chosen grasp was not seen", said["natural"][1])
        self.assertEqual((filtered.looked.jaw_faces_seen, filtered.looked.withheld), said["natural"])


# ---------------------------------------------------------------------------------------------------------------------
# A program's axis asks the calculator for 36 and keeps the best 12 along it
# ---------------------------------------------------------------------------------------------------------------------


class _Capped:
    """A calculator with a cap, as ``GraspCalculator`` has one (``max_candidates``): it ranks 36 grasps, every third
    closing along base y and the others along base x, best first, returns the best ``max_candidates`` of them (``cap``,
    12 as the calculator's default) and keeps the cap it was asked with each time."""

    render_debug_images = False
    last_debug_image_png = None

    def __init__(self, headings: tuple[float, ...] | None = None, *, fails: bool = False, cap: int = 12) -> None:
        self.max_candidates = cap
        self.asked: list[int] = []
        self.fails = fails
        self.grasps = tuple(_grasp(heading, score=0.99 - index / 100.0) for index, heading in enumerate(
            headings if headings is not None else tuple(90.0 if index % 3 == 0 else 0.0 for index in range(36))))

    def compute_result(self, *_args: Any, **_kwargs: Any) -> GraspResult:
        self.asked.append(self.max_candidates)
        if self.fails:
            raise RuntimeError("the calculator fell over")
        ranked = self.grasps[:self.max_candidates]
        return GraspResult(candidates=ranked, top_score=float(ranked[0].score))


def _capped_loop(calculator: _Capped, closing_axis: Any = None, **wiring: Any) -> Any:
    arm = _FakeArm()
    return BinPickingOrchestrator(
        arm=arm, calculator=calculator, perception=_FakePerception([_perception_frame()]),  # type: ignore[arg-type]
        policy=GraspExecutionPolicy(arm=arm, closing_axis=closing_axis), max_attempts=1, **wiring)  # type: ignore[arg-type]


class AProgramsAxisChoosesAmongMoreCandidatesTests(unittest.TestCase):
    def test_a_part_with_36_candidates_keeps_every_one_along_the_axis_up_to_12(self) -> None:
        """Of the calculator's best 12 only 4 close along y; asked for 36, all 12 that do are kept, best first, each
        turned to close along -y."""
        self.assertEqual(36, CANDIDATES_THE_AXIS_CHOOSES_AMONG)
        calculator = _Capped()

        report = _capped_loop(calculator, "-y").run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual([36], calculator.asked)
        self.assertEqual(12, calculator.max_candidates, "the calculator's own cap is given back")
        executed = report.executed_grasp
        assert executed is not None
        self.assertEqual([grasp.score for grasp in calculator.grasps[::3]], [grasp.score for grasp in executed.candidates])
        for grasp in executed.candidates:
            np.testing.assert_allclose(grasp.axis, (0.0, -1.0, 0.0), atol=1e-12)

    def test_more_than_12_along_the_axis_keeps_the_best_12(self) -> None:
        calculator = _Capped(tuple(-90.0 for _ in range(36)))

        report = _capped_loop(calculator, "-y").run()

        executed = report.executed_grasp
        assert executed is not None
        self.assertEqual([grasp.score for grasp in calculator.grasps[:12]], [grasp.score for grasp in executed.candidates])

    def test_a_calculator_whose_own_cap_is_above_36_is_asked_for_its_own_and_keeps_as_many(self) -> None:
        calculator = _Capped(tuple(-90.0 for _ in range(60)), cap=50)

        report = _capped_loop(calculator, "-y").run()

        self.assertEqual(([50], 50), (calculator.asked, calculator.max_candidates))
        executed = report.executed_grasp
        assert executed is not None
        self.assertEqual([grasp.score for grasp in calculator.grasps[:50]], [grasp.score for grasp in executed.candidates])

    def test_without_an_axis_the_calculator_is_asked_for_its_own_cap(self) -> None:
        for wiring in ({}, {"natural_closing_axis": _MINUS_Y}):
            with self.subTest(natural=bool(wiring)):
                calculator = _Capped()
                report = _capped_loop(calculator, **wiring).run()
                self.assertEqual([12], calculator.asked)
                executed = report.executed_grasp
                assert executed is not None
                self.assertEqual(12, len(executed.candidates), "the natural orientation leaves no grasp out")

    def test_the_calculators_own_cap_is_given_back_when_it_raises(self) -> None:
        calculator = _Capped(fails=True)

        with self.assertRaises(RuntimeError):
            _capped_loop(calculator, "-y").run()

        self.assertEqual(([36], 12), (calculator.asked, calculator.max_candidates))

    def test_the_scene_asks_its_generator_for_as_many(self) -> None:
        scene = Scene.from_cloud(_block(), support_height_mm=0.0)
        generate = scene_module.generate_support_footprint_grasps
        for closing_axis, asked in (("-y", 36), (None, None)):
            with self.subTest(closing_axis=closing_axis), mock.patch.object(
                    scene_module, "generate_support_footprint_grasps", side_effect=generate) as generator:
                scene.grasps(closing_axis=closing_axis) if closing_axis else scene.grasps()
                self.assertEqual(asked, generator.call_args.kwargs.get("max_candidates"))

    def test_the_real_calculator_of_a_pick_run_is_asked_for_36_and_keeps_its_12(self) -> None:
        breakdowns = calculator_module.support_footprint_breakdowns
        asked: list[int] = []

        def spy(*args: Any, **keywords: Any) -> Any:
            asked.append(int(keywords["max_candidates"]))
            return breakdowns(*args, **keywords)

        cell = Cell.rehearsal(_console_dummy().robot, motion=GraspMotion(closing_axis="-y"))
        with mock.patch.object(calculator_module, "support_footprint_breakdowns", side_effect=spy):
            report = PickRun.from_cell(cell, runs=1, recording=Recording.off()).execute()

        (attempt,) = report.attempts
        self.assertIs(CampaignOutcome.SUCCEEDED, attempt.outcome, report.render())
        self.assertTrue(asked)
        self.assertEqual({36}, set(asked))
        self.assertEqual(12, cell.service.runtime.orchestrator.calculator.max_candidates)


# ---------------------------------------------------------------------------------------------------------------------
# Scene.grasps and the Locator take it from the robot config
# ---------------------------------------------------------------------------------------------------------------------


def _with(natural: Any) -> Any:
    """The owner's hand profile, naming ``natural`` as its natural orientation."""
    return _replace(_cell(), "natural_closing_axis", natural)


class TheSceneAndTheLocatorTakeItFromTheConfigTests(unittest.TestCase):
    def test_a_scene_from_the_config_turns_every_grasp_and_leaves_none_out(self) -> None:
        cloud = _block()
        ranked = Scene.from_robot_config(_cell(), cloud).grasps().candidates
        self.assertTrue(ranked)
        turned_any = False
        for natural, heading in (("-y", -90.0), ("y", 90.0)):
            with self.subTest(natural=natural):
                turned = Scene.from_robot_config(_with(natural), cloud).grasps().candidates
                self.assertEqual(len(ranked), len(turned))
                for before, after in zip(ranked, turned):
                    np.testing.assert_array_equal(before.position_mm, after.position_mm)
                    np.testing.assert_array_equal(before.approach, after.approach)
                    self.assertEqual((before.score, before.grip_width_mm), (after.score, after.grip_width_mm))
                    self.assertTrue(np.allclose(after.closing_axis, before.closing_axis)
                                    or np.allclose(after.closing_axis, -before.closing_axis))
                    self.assertLess(abs(_off(_heading_deg(after.closing_axis), heading)), 90.0)
                    turned_any = turned_any or not np.allclose(after.closing_axis, before.closing_axis)
        self.assertTrue(turned_any, "the control: the generator ranked some grasp the other way round")

    def test_a_programs_own_axis_wins_in_the_scene(self) -> None:
        taken = Scene.from_robot_config(_with("-y"), _block()).grasps(closing_axis="y").candidates
        self.assertTrue(taken)
        for grasp in taken:
            self.assertLess(abs(_off(_heading_deg(grasp.closing_axis), 90.0)), 1.0)

    def test_a_scene_from_a_bare_cloud_or_a_config_that_names_none_turns_nothing(self) -> None:
        cloud = _block()
        with mock.patch.object(scene_module, "turned_nearer", side_effect=AssertionError("read")):
            self.assertTrue(Scene.from_cloud(cloud, support_height_mm=0.0).grasps().candidates)
            self.assertTrue(Scene.from_robot_config(_cell(), cloud).grasps().candidates)

    def test_the_locator_judges_the_faces_of_the_grasp_turned_the_natural_way_round(self) -> None:
        """From the north look alone the cube's best grasp closes along +y: jaw 2 on the north face it sees, jaw 1 on
        the south face it does not. A hand that naturally closes along -y takes it half a turn round, and the refusal
        names jaw 2; the part's scene gives the program that same grasp."""
        said: dict[str, Any] = {}
        for natural in (None, "-y"):
            built = _arm(NORTH)
            config = _cell() if natural is None else _with(natural)
            located = _locator(_Wrist(built)).look_around("a part", [NORTH.joints], robot=_robot(built, config=config),
                                                          both_faces=True)
            best = replace(located, refused="").scene(0, config).grasps().best
            assert best is not None
            said[str(natural)] = (located.jaw_faces_seen, located.refused, _heading_deg(best.closing_axis))

        self.assertEqual((False, True), said["None"][0])
        self.assertIn("jaw 1", said["None"][1])
        self.assertAlmostEqual(90.0, said["None"][2], delta=1.0)
        self.assertEqual((True, False), said["-y"][0])
        self.assertIn("jaw 2", said["-y"][1])
        self.assertAlmostEqual(-90.0, said["-y"][2], delta=1.0)

    def test_a_part_whose_turned_grasp_the_looks_saw_both_faces_of_is_gripped_that_way_round(self) -> None:
        built = _arm(NORTH, SOUTH)
        config = _with("-y")

        located = _locator(_Wrist(built)).look_around("a part", [NORTH.joints, SOUTH.joints],
                                                      robot=_robot(built, config=config), both_faces=True)

        self.assertEqual(((True, True), ""), (located.jaw_faces_seen, located.refused))
        best = located.scene(0, config).grasps().best
        assert best is not None
        self.assertAlmostEqual(-90.0, _heading_deg(best.closing_axis), delta=1.0)


# ---------------------------------------------------------------------------------------------------------------------
# PickRun, on the rehearsal cell
# ---------------------------------------------------------------------------------------------------------------------


class APickRunClosesTheNaturalWayRoundTests(unittest.TestCase):
    """The rehearsal cell (``console_dummy``: the real service, pick loop and calculator on a dummy arm and hand) grasps
    its box closing along base +y first and along base +x next."""

    @staticmethod
    def _run(natural: Any, motion: GraspMotion | None = None) -> Any:
        config = _console_dummy().robot
        if natural is not None:
            config = _replace(config, "natural_closing_axis", natural)
        cell = Cell.rehearsal(config, motion=motion if motion is not None else GraspMotion())
        return PickRun.from_cell(cell, runs=1, recording=Recording.off()).execute()

    def test_each_pick_closes_the_way_round_nearer_the_cells_natural_orientation(self) -> None:
        for natural, heading in ((None, 90.0), ("-y", -90.0), ("y", 90.0), (list(_OWNERS.quaternion_xyzw), -90.0),
                                 ("x", 90.0)):
            with self.subTest(natural=natural):
                (attempt,) = self._run(natural).attempts

                self.assertIs(CampaignOutcome.SUCCEEDED, attempt.outcome)
                assert attempt.grasp_pose is not None
                self.assertAlmostEqual(0.0, _off(_heading_deg(attempt.grasp_pose.to_matrix()[:3, 0]), heading), places=6)

    def test_a_motions_own_axis_wins(self) -> None:
        (attempt,) = self._run("-y", GraspMotion(closing_axis="y")).attempts

        self.assertIs(CampaignOutcome.SUCCEEDED, attempt.outcome)
        assert attempt.grasp_pose is not None
        self.assertAlmostEqual(0.0, _off(_heading_deg(attempt.grasp_pose.to_matrix()[:3, 0]), 90.0), places=6)

    def test_beside_the_simulators_twist_each_pick_is_the_one_it_was_without_the_key(self) -> None:
        """The box's grasp leans its approach; the twist closes it along base X either way round, and a turn would lean
        it the other way. The pose gripped is the one gripped without the key."""
        twisted = GraspMotion(align_closing_to_base_x=True)
        (bare,) = self._run(None, twisted).attempts
        assert bare.grasp_pose is not None
        for natural in ("-y", "y", list(_OWNERS.quaternion_xyzw)):
            with self.subTest(natural=natural):
                (attempt,) = self._run(natural, twisted).attempts

                self.assertIs(CampaignOutcome.SUCCEEDED, attempt.outcome)
                assert attempt.grasp_pose is not None
                np.testing.assert_allclose(attempt.grasp_pose.to_matrix(), bare.grasp_pose.to_matrix(), atol=1e-9)


# ---------------------------------------------------------------------------------------------------------------------
# Fixed poses built through the cell: robot.tool_down
# ---------------------------------------------------------------------------------------------------------------------


def _desk_robot(natural: Any = None) -> Robot:
    config = RobotConfig() if natural is None else RobotConfig(natural_closing_axis=natural)
    return Robot.from_parts(arm=DummyRobotArm(), gripper=None, lock_key=None, robot_config=config)


class ARobotsPosesTakeItsNaturalOrientationTests(unittest.TestCase):
    def test_a_pose_the_program_names_no_axis_for_takes_the_cells(self) -> None:
        robot = _desk_robot("-y")
        self.assertEqual(Pose.tool_down(*_AT, closing_axis="-y"), robot.tool_down(*_AT))
        self.assertEqual(Pose.tool_down(*_AT, yaw_deg=30.0, closing_axis="-y"), robot.tool_down(*_AT, yaw_deg=30.0))
        self.assertEqual(Pose.tool_down(400.0, 300.0, 200.0, closing_axis="tangential"),
                         _desk_robot("+tangential").tool_down(400.0, 300.0, 200.0))

    def test_the_programs_own_axis_wins(self) -> None:
        robot = _desk_robot("-y")
        self.assertEqual(Pose.tool_down(*_AT), robot.tool_down(*_AT, closing_axis="x"))
        self.assertEqual(Pose.tool_down(*_AT, closing_axis="y"), robot.tool_down(*_AT, closing_axis="y"))
        turned = robot.tool_down(*_AT, closing_axis=Pose.tool_down(0.0, 0.0, 0.0, yaw_deg=45.0).quaternion_xyzw)
        np.testing.assert_allclose(turned.quaternion_xyzw, Pose.tool_down(*_AT, yaw_deg=45.0).quaternion_xyzw,
                                   atol=1e-12)

    def test_a_cell_that_names_none_is_x_exactly_as_pose_tool_down(self) -> None:
        for robot in (_desk_robot(), Robot.from_parts(arm=DummyRobotArm(), gripper=None, lock_key=None)):
            with self.subTest(config=robot.robot_config is not None):
                self.assertEqual(Pose.tool_down(*_AT), robot.tool_down(*_AT))
                self.assertEqual(Pose.tool_down(*_AT, yaw_deg=90.0), robot.tool_down(*_AT, yaw_deg=90.0))

    def test_a_quaternion_is_the_name_it_was_taught_from(self) -> None:
        taught = _desk_robot(tuple(float(value) for value in _OWNERS.quaternion_xyzw)).tool_down(*_AT, yaw_deg=10.0)
        np.testing.assert_allclose(taught.quaternion_xyzw,
                                   Pose.tool_down(*_AT, yaw_deg=10.0, closing_axis="-y").quaternion_xyzw, atol=1e-12)

    def test_a_value_that_names_no_axis_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            _desk_robot("-y").tool_down(*_AT, closing_axis="z")
        for other in (None, 3):
            with self.subTest(closing_axis=other), self.assertRaises(TypeError):
                _desk_robot("-y").tool_down(*_AT, closing_axis=other)  # type: ignore[arg-type]

    def test_a_robot_built_without_a_tree_reads_the_one_its_arm_keeps(self) -> None:
        """As ``Robot.preflight`` reads it: a robot handed no ``robot_config`` reads ``arm.config``."""
        arm = DummyRobotArm()
        arm.config = RobotConfig(natural_closing_axis="-y")  # type: ignore[attr-defined]

        robot = Robot.from_parts(arm=arm, gripper=None, lock_key=None)

        self.assertIsNone(robot.robot_config)
        self.assertEqual(Pose.tool_down(*_AT, closing_axis="-y"), robot.tool_down(*_AT))

    def test_the_tree_the_robot_was_built_from_wins_over_the_one_its_arm_keeps(self) -> None:
        arm = DummyRobotArm()
        arm.config = RobotConfig(natural_closing_axis="-y")  # type: ignore[attr-defined]

        robot = Robot.from_parts(arm=arm, gripper=None, lock_key=None,
                                 robot_config=RobotConfig(natural_closing_axis="y"))

        self.assertEqual(Pose.tool_down(*_AT, closing_axis="y"), robot.tool_down(*_AT))

    def test_a_cells_robot_reads_the_cells_tree(self) -> None:
        cell = Cell.rehearsal(_replace(_console_dummy().robot, "natural_closing_axis", "-y"))
        cell.build()
        self.assertEqual(Pose.tool_down(*_AT, closing_axis="-y"), cell.robot.tool_down(*_AT))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
