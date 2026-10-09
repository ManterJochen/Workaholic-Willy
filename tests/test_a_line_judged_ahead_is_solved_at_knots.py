"""A line judged ahead is solved by the controller every 12 mm, and judged at the path gate's step (the judge chain,
2026-10-08).

A grasp judged ahead asks the controller's inverse kinematics for every sample of its line down and its lift, a sample
every 3 mm: 28 round trips for an 80 mm line, 35 for a 100 mm lift, 32.5 ms each on the cell, 92 calls and 3.1 s of a
try. None of those lines is ever commanded (``start``): the line that runs is judged again from where the arm stands,
every sample solved. So a line judged ahead is solved at knots 12 mm apart at most, every fourth sample, and the joint
line between two knots is filled at the line's own step, as the step between two samples always was: the path gate still
judges configurations no further apart than its bound. On nominal UR10 DH the cell's two grasp lines filled between 12 mm
knots run 26 to 33 um off the line moveL runs, against 1.7 to 2.2 um at 3 mm. A step between two knots that leaves the
workspace box, has no solution at its knot, changes branch, or bends more than 0.1 mm off the line is solved at every
sample, seeded on its start, and judged as before.

What this file pins: the line judged ahead asks the controller at most 8 times for 80 mm, and the line judged from where
the arm stands keeps every sample; the gate's bound holds between every two configurations it is handed; a branch jump
inside a step between knots is refused in the very words a line solved at every sample refuses it with; a step judged
off the line is solved at every sample; and the DH evidence.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import JointPositions, MotionCommand, MotionStatus
from src.robot.safety._ur_ik import ur_flange_ik
from src.robot.safety._ur_kinematics import ur_link_transforms_mm
from src.robot.safety.path_samples import line_samples
from tests.test_a_grasp_judged_ahead_runs_as_it_was_judged import (
    _GRASP,
    _STANDOFF,
    _cell,
    _controller_ik,
    _nearest,
    _stand,
)
from tests.test_every_verb_meets_the_camera_world import _mesh_backend_available


def _recorded(arm: Any) -> "list[Any]":
    """Every set of samples the path gate is handed, as it is handed them."""
    handed: list[Any] = []
    gate = arm._preflight.gate_joint_path

    def recorded(samples: Any, **asked: Any) -> Any:
        handed.append(samples)
        return gate(samples, **asked)

    arm._preflight.gate_joint_path = recorded
    return handed


def _jumping(arm: Any, solved: "list[Pose]", *, below_mm: float) -> Any:
    """The controller's inverse kinematics, with the shoulder pan turned 2.5 rad off its branch for every pose below
    ``below_mm``: a branch change mid line."""
    ik = _controller_ik(arm, solved)

    def jumped(pose: Pose, *, seed: "JointPositions | None" = None) -> JointPositions:
        joints = list(ik(pose, seed=seed).tolist())
        if float(pose.position_mm[2]) < below_mm:
            joints[0] += 2.5
        return JointPositions(tuple(joints))

    return jumped


class ALineJudgedAheadIsSolvedAtKnotsTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def _ahead(self) -> "tuple[Any, list[Pose], list[Any], list[float]]":
        arm, _events, solved = _cell()
        near = [float(v) for v in arm.nearest_configuration(_STANDOFF).values]
        handed = _recorded(arm)
        solved.clear()
        return arm, solved, handed, near

    def test_an_80_mm_line_judged_ahead_asks_the_controller_at_most_8_times(self) -> None:
        """⭐ Red before: 28 round trips, one a sample."""
        arm, solved, handed, near = self._ahead()
        refused = arm._judge_linear_move(_GRASP, command=MotionCommand.MOVE_TO, commanded=False,
                                         start=(_STANDOFF, near))
        self.assertIsNone(refused, refused.message if refused is not None else "")
        self.assertLessEqual(len(solved), 8)
        self.assertGreater(len(solved), 1)
        (samples,) = handed
        radii = arm._preflight.joint_radii_mm(arm)
        step = float(arm._preflight.path_step_mm)
        self.assertLessEqual(samples.step_bound_mm, step)
        for index, (a, b) in enumerate(zip(samples.configs, samples.configs[1:])):
            with self.subTest(step=index):
                self.assertLessEqual(sum(abs(y - x) * r for x, y, r in zip(a, b, radii)), step + 1e-6)

    def test_every_configuration_judged_ahead_puts_the_flange_on_the_line(self) -> None:
        arm, _solved, handed, near = self._ahead()
        self.assertIsNone(arm._judge_linear_move(_GRASP, command=MotionCommand.MOVE_TO, commanded=False,
                                                 start=(_STANDOFF, near)))
        (samples,) = handed
        top = np.asarray(arm._pose_to_flange(_STANDOFF).position_mm, dtype=np.float64)
        bottom = np.asarray(arm._pose_to_flange(_GRASP).position_mm, dtype=np.float64)
        along = (bottom - top) / np.linalg.norm(bottom - top)
        worst = 0.0
        model = str(arm.config.ur.model)
        for config in samples.configs:
            flange = np.asarray(ur_link_transforms_mm(model, np.asarray(config))[-1])[:3, 3]
            offset = flange - top
            worst = max(worst, float(np.linalg.norm(offset - np.dot(offset, along) * along)))
        self.assertLess(worst, 0.1, "a configuration judged ahead stands off the line moveL runs")

    def test_a_line_judged_from_where_the_arm_stands_keeps_every_sample(self) -> None:
        """The control: the line that runs is solved at every sample, 28 for 80 mm."""
        arm, solved, _handed, near = self._ahead()
        _stand(arm, near)
        self.assertIsNone(arm._judge_linear_move(_GRASP, command=MotionCommand.MOVE_TO, commanded=False))
        self.assertEqual(28, len(solved))

    def test_a_branch_jump_inside_a_step_between_knots_is_refused_as_a_line_solved_at_every_sample_refuses_it(
            self) -> None:
        arm, solved, _handed, near = self._ahead()
        arm.ik = _jumping(arm, solved, below_mm=170.5)
        ahead = arm._judge_linear_move(_GRASP, command=MotionCommand.MOVE_TO, commanded=False, start=(_STANDOFF, near))
        _stand(arm, near)
        every = arm._judge_linear_move(_GRASP, command=MotionCommand.MOVE_TO, commanded=False)
        assert ahead is not None and every is not None
        self.assertIs(MotionStatus.IK_QUALITY_REJECTED, ahead.status, ahead.message)
        self.assertEqual(every.status, ahead.status)
        self.assertEqual(every.message, ahead.message)
        self.assertIn("between sample 10 and sample 11 of 28", ahead.message)
        self.assertIn("joint 1 turns 2.50 rad", ahead.message)

    def test_a_step_off_the_line_is_solved_at_every_sample(self) -> None:
        """Where no step may bend off the line at all, every step between knots is solved again at every sample, seeded
        on its start: the gate is handed exactly what the line solved at every sample hands it."""
        from src.robot.drivers.ur.arm import URRobotArm

        arm, solved, handed, near = self._ahead()

        def judged() -> None:
            self.assertIsNone(arm._judge_linear_move(_GRASP, command=MotionCommand.MOVE_TO, commanded=False,
                                                     start=(_STANDOFF, near)))

        with patch.object(URRobotArm, "_AHEAD_KNOT_MM", 0.0):
            judged()
        every_sample, asked = handed.pop(), len(solved)
        solved.clear()
        with patch.object(URRobotArm, "_AHEAD_KNOT_OFF_MM", -1.0):
            judged()
        (resolved,) = handed
        self.assertEqual(28, asked, "the line solved at every sample")
        self.assertEqual(28 + 7, len(solved), "each of the seven steps solved at its knot, then at every sample")
        self.assertEqual(every_sample.configs, resolved.configs)
        self.assertEqual(every_sample.step_bound_mm, resolved.step_bound_mm)


#: The cell's look (2026-10-08) and its tool: the flange to the TCP 157 mm along z, turned -90 degrees about z.
_LOOK_DEG = (-75.10, -83.00, -72.40, -137.31, 96.02, -76.28)
_TOOL = np.array([[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 157.0], [0.0, 0.0, 0.0, 1.0]])


def _tcp_down(x: float, y: float, z: float, yaw_deg: float) -> "np.ndarray":
    turn = np.radians(yaw_deg)
    down = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    about_z = np.array([[np.cos(turn), -np.sin(turn), 0.0], [np.sin(turn), np.cos(turn), 0.0], [0.0, 0.0, 1.0]])
    matrix = np.eye(4)
    matrix[:3, :3] = about_z @ down
    matrix[:3, 3] = (x, y, z)
    return matrix


def _worst_off_the_line_mm(x: float, y: float, z_from: float, z_to: float, *, knot_mm: float) -> float:
    """The worst of :func:`~src.robot.drivers.ur.arm._off_the_line_mm` over the steps of the line from ``z_from`` to
    ``z_to``, knots ``knot_mm`` apart at most, each solved nearest the one before, from the cell's look."""
    from src.robot.drivers.ur.arm import _off_the_line_mm

    steps = int(np.ceil(abs(z_to - z_from) / knot_mm - 1e-9))
    seed = np.radians(_LOOK_DEG)
    knots = []
    for z in np.linspace(z_from, z_to, steps + 1):
        flange = _tcp_down(x, y, float(z), 45.0) @ np.linalg.inv(_TOOL)
        seed = _nearest(ur_flange_ik("ur10", flange, q6_if_singular=float(seed[-1])), seed)
        knots.append(seed)
    fractions = [k / 32.0 for k in range(1, 32)]
    return max(_off_the_line_mm("ur10", a, b, fractions, reach_mm=1500.0) for a, b in zip(knots, knots[1:]))


class TheDhEvidenceTests(unittest.TestCase):
    """The two grasp lines of 2026-10-08 (D:/helper_cell logs, picks 1 and 2), on nominal UR10 DH."""

    _GRASPS = {"pick 1": (-178.9, -803.1, 76.6), "pick 2": (-21.8, -660.9, 79.1)}

    def test_knots_12_mm_apart_keep_the_flange_and_the_hand_within_a_tenth_of_a_millimetre_of_the_line(self) -> None:
        for name, (x, y, z) in self._GRASPS.items():
            for what, (z_from, z_to) in (("line down", (z + 80.0, z)), ("lift", (z, z + 100.0))):
                with self.subTest(grasp=name, line=what):
                    coarse = _worst_off_the_line_mm(x, y, z_from, z_to, knot_mm=12.0)
                    fine = _worst_off_the_line_mm(x, y, z_from, z_to, knot_mm=3.0)
                    self.assertLessEqual(coarse, 0.1)
                    self.assertLess(fine, 0.005)
                    self.assertGreater(coarse, fine, "the measure reads nothing")

    def test_a_model_with_no_chain_admits_no_knot(self) -> None:
        from src.robot.drivers.ur.arm import _off_the_line_mm

        self.assertEqual(float("inf"), _off_the_line_mm("not-a-ur", [0.0] * 6, [0.1] * 6, [0.5], reach_mm=100.0))

    def test_the_arm_places_its_knots_every_fourth_sample_of_an_80_mm_line(self) -> None:
        start = Pose(position_mm=np.array([0.0, -700.0, 160.0]), quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]),
                     frame=Frame.BASE)
        goal = Pose(position_mm=np.array([0.0, -700.0, 80.0]), quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]),
                    frame=Frame.BASE)
        samples = line_samples(start, goal, reach_mm=1500.0, max_step_mm=3.0)
        from src.robot.drivers.ur.arm import URRobotArm

        every = int(np.floor(URRobotArm._AHEAD_KNOT_MM / samples.step_bound_mm + 1e-9))
        self.assertEqual(4, every)
        knots = sorted({*range(0, len(samples.poses), every), len(samples.poses) - 1})
        self.assertEqual(8, len(knots))


if __name__ == "__main__":
    unittest.main()
