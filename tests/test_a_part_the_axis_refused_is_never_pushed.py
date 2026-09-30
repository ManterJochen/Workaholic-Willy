"""A part the closing axis refused is never pushed, its boxed-in neighbour is, and a push closes the natural way round.

The merge of the owner's closing axis and natural orientation (2026-09-30) with the push of a boxed-in part (B2), on
the push file's cell in miniature (``tests/test_a_boxed_in_part_is_pushed_inside_the_pick.py``): the UR10 judging
every line, the Hand-E as ``jaw_io`` single_toggle on tool DO0, the tilted wrist D415 and three declared looks.

Pinned:

* every object's result passes the closing-axis filter right after it is ranked, before the push reads it: a part none
  of whose grasps closes along the named axis carries only ``no_valid_grasp`` and is never pushed, boxed in or not;
* part A refused by the axis beside part B, every grasp of which collides with a post 22 mm off it: B is pushed, and
  picked closing along the axis, whichever order the camera lists the two in; A is never pushed. A part the axis
  refused is the frame's failure only where no other part failed for its own reason, and no frame-wide guard stops B's
  push because A was refused;
* where the cell names how its hand and camera naturally stand (``robot.natural_closing_axis``), every pose of a push
  closes the way round whose tool +X lies nearer that direction at the part, as every camera grasp does; without it,
  and where the push runs straight across it, the way round nearer where the tool stands, as before.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from src.geometry import Pose
from src.geometry.closing_axis import closing_axis_of
from src.geometry.quaternion import to_rotation_matrix
from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome
from src.robot.grasping.recovery.push_motion import push_tool_quaternion
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from tests._wrist_views import CUBE, Box
from tests.test_a_boxed_in_part_is_pushed_inside_the_pick import (
    LOOKS,
    POST,
    PUSH_LABELS,
    _centre_of_mask,
    _PushCell,
    _PushingArm,
)
from tests.test_a_wrist_pick_looks_until_its_grasp_is_safe import LOOK_PLUS_Y, _keys

#: Part A, of the label, 40 mm off the part's +x face: no neighbour stands within 25 mm of it.
PART_A = Box((60.0, -720.0, 0.0), (100.0, -680.0, 40.0), "part")


def _grasp(centre: np.ndarray, axis: tuple[float, float, float]) -> GraspPoint:
    return GraspPoint(position=centre, approach=np.array([0.0, 0.0, -1.0]), axis=np.asarray(axis, dtype=np.float64),
                      grip_width_mm=40.0, score=0.9, frame=GraspFrame.BASE, label="part")


class _RefusedBesideBoxedIn:
    """Part A has one grasp, closing along base x, which ``closing_axis="-y"`` refuses. Part B (the part the post stands
    22 mm off) collides with every grasp (``ALL_COLLIDED``) until it is pushed, then has one closing along base y."""

    render_debug_images = False

    def __init__(self, cell: _PushCell) -> None:
        self.cell = cell
        self.ranked: list[tuple[int, str]] = []

    def compute_result(self, seg: Any, *_args: Any, **_kwargs: Any) -> GraspResult:
        centre = _centre_of_mask(seg, self.cell)
        which = "A" if centre[0] > 40.0 else "B"
        self.ranked.append((len(self.cell.camera.taken) - 1, which))
        if which == "A":
            return GraspResult(candidates=(_grasp(centre, (1.0, 0.0, 0.0)),), top_score=0.9)
        if self.cell.camera.pushed:
            return GraspResult(candidates=(_grasp(centre, (0.0, 1.0, 0.0)),), top_score=0.9)
        return GraspResult(reasons=(GraspFailureReason.ALL_COLLIDED, GraspFailureReason.RESCAN_RECOMMENDED))


class _RecordsItsPushPoses(_PushingArm):
    """The pushing arm, keeping the tool's X in BASE of every pose the push sends (``tool_x``, by label)."""

    def move(self, pose: Any, **keywords: Any) -> Any:
        label = str(pose.label)
        if label.startswith("push_"):
            self.__dict__.setdefault("tool_x", []).append(
                (label, to_rotation_matrix(np.asarray(pose.quaternion_xyzw, dtype=np.float64))[:, 0]))
        return super().move(pose, **keywords)


def _part_b(cell: _PushCell) -> Box:
    return next(box for box in cell.camera.boxes if box.label == "part" and box.low[0] < 40.0)


class APartTheAxisRefusedIsNeverPushedTests(unittest.TestCase):
    def test_a_boxed_in_neighbour_is_pushed_and_picked_whichever_order_the_camera_lists_them(self) -> None:
        for order in ((PART_A, CUBE, POST), (CUBE, PART_A, POST)):
            with self.subTest(first=("A", "B")[order[0] is CUBE]):
                cell = _PushCell(self, boxes=order)
                cell.calculator = _RefusedBesideBoxedIn(cell)  # type: ignore[assignment]
                cell.orchestrator.calculator = cell.calculator
                cell.policy.closing_axis = "-y"  # type: ignore[attr-defined]

                with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
                    run = cell.campaign()

                self.assertTrue(run.attempts[0].passed, run.render())
                self.assertIn((0, "A"), cell.calculator.ranked, "part A was not ranked")
                # B, and only B, was pushed: once, from where it stood beside the post, in the pick that found it
                # boxed in. No next_target skipped A and drove the looks again first.
                self.assertEqual(PUSH_LABELS, cell.arm.push_motions())
                self.assertEqual(_keys(*LOOKS, LOOK_PLUS_Y), cell.joint_moves(), "the looks were driven again")
                self.assertEqual(4, len(cell.camera.taken))
                self.assertEqual([], run.last.telemetry["recovery_trail_actions"])
                (push,) = run.last.telemetry["pushes"]
                self.assertEqual(("all_collided", "pushed"), (push["trigger"], push["code"]))
                self.assertLess(float(np.hypot(push["part_centre_mm"][0], push["part_centre_mm"][1] + 700.0)), 10.0,
                                "the push was not B's")
                self.assertEqual(PART_A, next(box for box in cell.camera.boxes if box.low[0] > 40.0),
                                 "part A moved")
                # B picked from the look after the push, closing along the axis the program named.
                (grasp,) = cell.policy.executed
                b = _part_b(cell)
                np.testing.assert_allclose(grasp.position, (np.asarray(b.low) + np.asarray(b.high)) / 2.0)
                np.testing.assert_allclose(np.asarray(grasp.axis), [0.0, -1.0, 0.0], atol=1e-9)
                cell.assert_the_jaws_untouched(self)

    def test_a_boxed_in_part_the_axis_refused_is_never_pushed(self) -> None:
        """The post stands 22 mm off the part, and its one grasp closes along base x: ``closing_axis="-y"`` refuses it
        (``no_valid_grasp`` alone, never ``all_collided``), so no push is even considered, and nothing moves for one."""
        cell = _PushCell(self, allowed=("rescan", "nudge_target"))
        cell.policy.closing_axis = "-y"  # type: ignore[attr-defined]

        def along_x(seg: Any, *_args: Any, **_kwargs: Any) -> GraspResult:
            return GraspResult(candidates=(_grasp(_centre_of_mask(seg, cell), (1.0, 0.0, 0.0)),), top_score=0.9)

        cell.calculator.compute_result = along_x  # type: ignore[method-assign]

        report = cell.service.pick(look=list(LOOKS))

        self.assertEqual([], cell.arm.pushes, "a part the axis refused was pushed")
        self.assertEqual((), cell.orchestrator.pushes, "a push was considered for a part the axis refused")
        self.assertIs(AutonomousGraspOutcome.NO_VALID_GRASP, report.outcome, report.render())
        self.assertIn("'-y'", report.failure_summary())
        cell.assert_the_jaws_untouched(self)


class APushClosesTheWayRoundTheHandNaturallyStandsTests(unittest.TestCase):
    """The push runs along (0.71, -0.71) on this cell, and the arm stands at the last look with its tool X nearer the
    reversed direction."""

    def _tool_x_of_the_push(self, natural: Any) -> np.ndarray:
        cell = _PushCell(self, arm=_RecordsItsPushPoses)
        if natural is not None:
            cell.orchestrator.natural_closing_axis = natural
        with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
            run = cell.campaign()
        self.assertTrue(run.attempts[0].passed, run.render())
        poses = cell.arm.__dict__["tool_x"]
        self.assertEqual(PUSH_LABELS, [label for label, _x in poses])
        for _label, tool_x in poses:
            np.testing.assert_allclose(tool_x, poses[0][1], atol=1e-9, err_msg="the push turned its wrist between legs")
        (push,) = run.last.telemetry["pushes"]
        np.testing.assert_allclose(np.abs(poses[0][1][:2]), np.abs(np.asarray(push["direction"][:2])), atol=0.02)
        return np.asarray(poses[0][1], dtype=np.float64)

    def test_every_pose_of_the_push_closes_the_way_round_nearer_the_natural_direction(self) -> None:
        owners = Pose.tool_down(0.0, -700.0, 100.0, closing_axis="-y").quaternion_xyzw
        for natural, towards in (("-y", (0.0, -1.0)), ("y", (0.0, 1.0)), ("x", (1.0, 0.0)), ("-x", (-1.0, 0.0)),
                                 (tuple(float(v) for v in owners), (0.0, -1.0))):
            with self.subTest(natural=natural):
                tool_x = self._tool_x_of_the_push(natural)
                self.assertGreater(float(tool_x[:2] @ np.asarray(towards)), 0.5,
                                   f"the push closes {tool_x.round(3)}, away from {towards}")

    def test_without_a_natural_orientation_the_push_closes_the_way_round_nearer_where_the_tool_stands(self) -> None:
        tool_x = self._tool_x_of_the_push(None)
        np.testing.assert_allclose(tool_x[:2], [-np.sqrt(0.5), np.sqrt(0.5)], atol=0.02)

    def test_the_sign_rule(self) -> None:
        down = (0.0, 0.0, -1.0)

        def plan(direction: tuple[float, float, float]) -> Any:
            return SimpleNamespace(direction=direction, approach=down, target_centre_mm=(0.0, -700.0, 20.0))

        def x_of(quaternion: Any) -> np.ndarray:
            return to_rotation_matrix(np.asarray(quaternion, dtype=np.float64))[:, 0]

        minus_y = closing_axis_of("-y")
        # The natural direction decides, wherever the tool stands.
        for here in ((0.0, 1.0, 0.0), (0.0, -1.0, 0.0), None):
            for direction in ((0.0, 1.0, 0.0), (0.0, -1.0, 0.0)):
                np.testing.assert_allclose(
                    x_of(push_tool_quaternion(plan(direction), current_tool_x=here, natural=minus_y)),
                    [0.0, -1.0, 0.0], atol=1e-12)
        # Straight across it, where the tool stands decides, as without one.
        for here in ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)):
            for natural in (minus_y, None):
                np.testing.assert_allclose(
                    x_of(push_tool_quaternion(plan((1.0, 0.0, 0.0)), current_tool_x=here, natural=natural)),
                    here, atol=1e-12)
        # Without one, where the tool stands decides.
        np.testing.assert_allclose(x_of(push_tool_quaternion(plan((0.0, 1.0, 0.0)), current_tool_x=(0.0, -1.0, 0.0))),
                                   [0.0, -1.0, 0.0], atol=1e-12)
        # radial is read at the part: 700 mm out along -y there.
        np.testing.assert_allclose(x_of(push_tool_quaternion(plan((0.0, 1.0, 0.0)), current_tool_x=(0.0, 1.0, 0.0),
                                                              natural=closing_axis_of("radial"))),
                                   [0.0, -1.0, 0.0], atol=1e-12)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
