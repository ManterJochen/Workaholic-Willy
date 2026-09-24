"""Every configuration that reaches a UR flange pose, in closed form, on the DH tables the guards already use.

The UR driver chooses the goal configuration of a Cartesian cuRobo move itself now: the one nearest the arm, among
every inverse kinematics solution there is. That is only as good as "every solution" is, so this pins it on every
bundled model: each solution puts the flange on the goal to under a micrometre and a microradian, the
configuration the pose came from is among them, and the singular poses (wrist 2 at 0 or pi, a stretched or folded
elbow, the wrist centre on the shoulder's cylinder) return solutions rather than raising.

The turn rule is pinned beside it: a joint goes on its full turn nearest the arm inside a window, and a solution
with a joint that has no turn inside is left out rather than clipped. And the branch: which of the eight ways of
holding the arm a configuration is in, read by the closed form's own three choices, so the arm can prefer the one it
holds, and a configuration at a branch point (the UR10 CB3's candle-straight home is at all three) is on both sides.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.robot.safety._ur_ik import (
    BRANCH_POINT_TOL_RAD,
    IK_SOLUTION_TOL_MM,
    IK_SOLUTION_TOL_RAD,
    NearestGoal,
    URBranch,
    nearest_goals,
    nearest_turn,
    ur_branch,
    ur_flange_ik,
)
from src.robot.safety._ur_kinematics import UR_DH_TABLES_M, ur_link_transforms_mm

#: Random configurations per model. Each call costs about 2.4 ms, so this is about 3 s for the seven models.
_PER_MODEL = 200


def _flange(model: str, joints: "list[float] | np.ndarray") -> np.ndarray:
    frames = ur_link_transforms_mm(model, np.asarray(joints, dtype=np.float64))
    assert frames is not None
    return np.asarray(frames[-1], dtype=np.float64)


def _error(model: str, joints: "tuple[float, ...]", goal: np.ndarray) -> tuple[float, float]:
    reached = _flange(model, joints)
    error_mm = float(np.linalg.norm(reached[:3, 3] - goal[:3, 3]))
    cosine = (float(np.trace(reached[:3, :3].T @ goal[:3, :3])) - 1.0) / 2.0
    return error_mm, math.acos(max(-1.0, min(1.0, cosine)))


def _apart(a: "tuple[float, ...] | np.ndarray", b: "tuple[float, ...] | np.ndarray") -> float:
    """The largest joint difference, a full turn counting as none."""
    return max(abs((float(x) - float(y) + math.pi) % (2.0 * math.pi) - math.pi) for x, y in zip(a, b))


def _clear_of_singularities(model: str, q: np.ndarray) -> bool:
    """Wrist 2 and the elbow away from 0 and pi, and the wrist centre off the shoulder's cylinder."""
    table = UR_DH_TABLES_M[model]
    flange = _flange(model, q)
    wrist = flange @ np.array([0.0, 0.0, -table[5].d_m * 1000.0, 1.0])
    return (abs(math.sin(q[4])) > 0.05 and abs(math.sin(q[2])) > 0.05
            and math.hypot(wrist[0], wrist[1]) > abs(table[3].d_m) * 1000.0 + 5.0)


class EverySolutionReachesTheGoalTests(unittest.TestCase):
    def test_every_solution_puts_the_flange_on_the_goal_on_every_model(self) -> None:
        rng = np.random.default_rng(20260923)
        for model in UR_DH_TABLES_M:
            with self.subTest(model=model):
                worst_mm = worst_rad = 0.0
                for _ in range(_PER_MODEL):
                    q = rng.uniform(-math.pi, math.pi, 6)
                    goal = _flange(model, q)
                    solutions = ur_flange_ik(model, goal, q6_if_singular=float(q[5]))
                    assert solutions is not None
                    self.assertGreater(len(solutions), 0, f"a pose the arm reached at {q.tolist()} has no solution")
                    for solution in solutions:
                        error_mm, error_rad = _error(model, solution, goal)
                        worst_mm, worst_rad = max(worst_mm, error_mm), max(worst_rad, error_rad)
                        self.assertTrue(all(-math.pi < v <= math.pi for v in solution), solution)
                self.assertLess(worst_mm, 1e-6, f"{model}: {worst_mm:.3g} mm")
                self.assertLess(worst_rad, 1e-6, f"{model}: {worst_rad:.3g} rad")

    def test_the_configuration_the_pose_came_from_is_among_them(self) -> None:
        rng = np.random.default_rng(7)
        for model in UR_DH_TABLES_M:
            with self.subTest(model=model):
                checked = 0
                while checked < _PER_MODEL // 2:
                    q = rng.uniform(-math.pi, math.pi, 6)
                    if not _clear_of_singularities(model, q):
                        continue
                    checked += 1
                    solutions = ur_flange_ik(model, _flange(model, q), q6_if_singular=float(q[5]))
                    assert solutions is not None
                    nearest = min(_apart(solution, q) for solution in solutions)
                    self.assertLess(nearest, 1e-6, f"{model}: {q.tolist()} is {nearest:.3g} rad from every solution")

    def test_a_general_pose_has_eight_distinct_solutions(self) -> None:
        q = [0.4, -1.2, 1.3, -1.7, 1.1, 0.3]
        solutions = ur_flange_ik("ur10", _flange("ur10", q))
        assert solutions is not None
        self.assertEqual(8, len(solutions))
        for i, a in enumerate(solutions):
            for b in solutions[i + 1:]:
                self.assertGreater(_apart(a, b), 1e-3)


class SingularPosesDoNotBreakItTests(unittest.TestCase):
    """Near a singularity two branches meet or a joint is free; the answer is still a set of solutions."""

    _CASES = {
        "wrist 2 at 0": [0.3, -1.4, 1.5, -1.6, 0.0, 0.7],
        "wrist 2 at pi": [0.3, -1.4, 1.5, -1.6, math.pi, 0.7],
        "wrist 2 a hair off 0": [0.3, -1.4, 1.5, -1.6, 1e-9, 0.7],
        "elbow stretched": [0.3, -1.4, 0.0, -1.6, 1.2, 0.7],
        "elbow folded back": [0.3, -1.4, math.pi, -1.6, 1.2, 0.7],
        "wrist centre over the base": [0.0, -math.pi / 2, 0.0, -math.pi / 2, 1.2, 0.0],
        "everything at zero": [0.0] * 6,
    }

    def test_every_singular_case_returns_solutions_that_reach_the_goal(self) -> None:
        for model in UR_DH_TABLES_M:
            for name, q in self._CASES.items():
                with self.subTest(model=model, case=name):
                    goal = _flange(model, q)
                    solutions = ur_flange_ik(model, goal, q6_if_singular=q[5])
                    assert solutions is not None
                    for solution in solutions:
                        error_mm, error_rad = _error(model, solution, goal)
                        self.assertLessEqual(error_mm, IK_SOLUTION_TOL_MM)
                        self.assertLessEqual(error_rad, IK_SOLUTION_TOL_RAD)

    def test_the_wrist_centre_on_the_shoulder_cylinder_still_solves(self) -> None:
        """Measured on a ur5 before the slack was set: the wrist centre read 1.05e-9 inside d4 and nothing came back."""
        q = [-1.425096, 0.884582, 1.68635, 1.268393, 0.759764, -2.048416]
        solutions = ur_flange_ik("ur5", _flange("ur5", q), q6_if_singular=q[5])
        assert solutions is not None
        self.assertGreater(len(solutions), 0)

    def test_a_free_wrist_3_keeps_the_one_the_caller_stands_at(self) -> None:
        """At wrist 2 = 0 only wrist 1 + wrist 3 is fixed; wrist 3 is not turned for nothing."""
        q = [0.3, -1.4, 1.5, -1.6, 0.0, 0.7]
        goal = _flange("ur5e", q)
        for hint in (0.7, -2.0, 2.5):
            with self.subTest(hint=hint):
                solutions = ur_flange_ik("ur5e", goal, q6_if_singular=hint)
                assert solutions is not None
                singular = [s for s in solutions if abs(math.sin(s[4])) < 1e-9]
                self.assertTrue(singular)
                for solution in singular:
                    self.assertAlmostEqual(0.0, _apart([solution[5]], [hint]), places=6)


class WhatIsNotAnAnswerTests(unittest.TestCase):
    def test_a_pose_out_of_reach_has_no_solution(self) -> None:
        goal = np.eye(4)
        goal[:3, 3] = (5000.0, 0.0, 300.0)
        self.assertEqual((), ur_flange_ik("ur10", goal))

    def test_a_pose_inside_the_shoulder_cylinder_has_no_solution(self) -> None:
        goal = np.eye(4)
        goal[:3, 3] = (0.0, 0.0, 300.0)
        goal[:3, :3] = np.diag([1.0, -1.0, -1.0])  # tool down, so the wrist centre is on the base axis
        self.assertEqual((), ur_flange_ik("ur10", goal))

    def test_an_unknown_model_is_none(self) -> None:
        self.assertIsNone(ur_flange_ik("ur20", np.eye(4)))

    def test_a_goal_that_is_not_a_transform_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            ur_flange_ik("ur10", np.eye(3))
        bad = np.eye(4)
        bad[0, 3] = float("nan")
        with self.assertRaises(ValueError):
            ur_flange_ik("ur10", bad)


class TheNearestTurnTests(unittest.TestCase):
    def test_the_turn_nearest_the_arm_inside_the_window(self) -> None:
        self.assertAlmostEqual(-0.5 + 2 * math.pi, nearest_turn(-0.5, near=5.5, lower=-6.18, upper=6.18))
        self.assertAlmostEqual(0.5, nearest_turn(0.5, near=1.0, lower=-6.18, upper=6.18))
        self.assertAlmostEqual(0.5 - 2 * math.pi, nearest_turn(0.5, near=-5.0, lower=-6.18, upper=6.18))

    def test_a_nearer_turn_outside_the_window_is_not_taken(self) -> None:
        """The arm at 6.1 rad and the angle at 0.3: 0.3 + 2 pi is nearer and out of the window, so 0.3 it is."""
        self.assertAlmostEqual(0.3, nearest_turn(0.3, near=6.1, lower=-6.18, upper=6.18))

    def test_no_turn_inside_is_none(self) -> None:
        """An elbow at 179 degrees has no turn inside +-(pi - 0.1)."""
        self.assertIsNone(nearest_turn(math.radians(179.0), near=0.0, lower=-(math.pi - 0.1), upper=math.pi - 0.1))

    def test_the_window_ends_are_inside(self) -> None:
        self.assertAlmostEqual(1.0, nearest_turn(1.0, near=0.0, lower=1.0, upper=1.5))


class TheNearestGoalsTests(unittest.TestCase):
    _LOW = (-6.18,) * 6
    _HIGH = (6.18,) * 6

    def test_quickest_first_then_least_total_turn(self) -> None:
        here = [0.0] * 6
        solutions = [
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),   # 1 rad at most, 1 in total
            (0.5, 0.5, 0.5, 0.5, 0.5, 0.5),   # 0.5 at most, 3 in total
            (0.5, 0.0, 0.0, 0.0, 0.0, 0.0),   # 0.5 at most, 0.5 in total
        ]
        goals = nearest_goals(solutions, current=here, lower=self._LOW, upper=self._HIGH, velocity=0.5)
        self.assertEqual([2, 1, 0], [goal.branch for goal in goals])
        self.assertIsInstance(goals[0], NearestGoal)
        self.assertAlmostEqual(1.0, goals[0].time_s)
        self.assertAlmostEqual(0.5, goals[0].largest_rad)
        self.assertAlmostEqual(0.5, goals[0].total_rad)

    def test_each_joint_goes_on_its_turn_nearest_the_arm(self) -> None:
        here = [0.0, 0.0, 0.0, 0.0, 0.0, 5.5]
        goals = nearest_goals([(0.0, 0.0, 0.0, 0.0, 0.0, -0.5)], current=here, lower=self._LOW, upper=self._HIGH,
                              velocity=1.0)
        self.assertAlmostEqual(-0.5 + 2 * math.pi, goals[0].joints[5])
        self.assertAlmostEqual(2 * math.pi - 0.5 - 5.5, goals[0].largest_rad)

    def test_a_solution_with_a_joint_outside_every_turn_is_left_out(self) -> None:
        low = list(self._LOW)
        high = list(self._HIGH)
        low[2], high[2] = -(math.pi - 0.1), math.pi - 0.1
        goals = nearest_goals([(0.0, 0.0, 3.1, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)], current=[0.0] * 6,
                              lower=low, upper=high, velocity=1.0)
        self.assertEqual([1], [goal.branch for goal in goals])

    def test_two_branches_on_one_configuration_are_one_goal(self) -> None:
        goals = nearest_goals([(0.1,) * 6, (0.1 + 2 * math.pi, 0.1, 0.1, 0.1, 0.1, 0.1)], current=[0.0] * 6,
                              lower=self._LOW, upper=self._HIGH, velocity=1.0)
        self.assertEqual([0], [goal.branch for goal in goals])

    def test_the_order_is_the_same_every_time(self) -> None:
        solutions = [(0.2, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.2, 0.0, 0.0, 0.0, 0.0)]
        first = nearest_goals(solutions, current=[0.0] * 6, lower=self._LOW, upper=self._HIGH, velocity=1.0)
        again = nearest_goals(solutions, current=[0.0] * 6, lower=self._LOW, upper=self._HIGH, velocity=1.0)
        self.assertEqual(first, again)
        self.assertEqual([0, 1], [goal.branch for goal in first])

    def test_a_speed_that_is_not_a_speed_is_refused(self) -> None:
        for velocity in (0.0, -1.0, float("nan")):
            with self.subTest(velocity=velocity), self.assertRaises(ValueError):
                nearest_goals([(0.0,) * 6], current=[0.0] * 6, lower=self._LOW, upper=self._HIGH, velocity=velocity)


class TheBranchTests(unittest.TestCase):
    """``ur_branch`` reads the three choices ``ur_flange_ik`` makes, in the order it makes them."""

    #: The order the closed form returns a general pose's eight solutions in: shoulder, then wrist 2, then elbow.
    _ORDER = [(s, w, e) for s in (1, -1) for w in (1, -1) for e in (1, -1)]

    def test_each_of_the_eight_solutions_reads_the_branch_it_was_solved_on_on_every_model(self) -> None:
        rng = np.random.default_rng(29)
        for model in UR_DH_TABLES_M:
            checked = 0
            while checked < 20:
                q = rng.uniform(-math.pi, math.pi, 6)
                if not _clear_of_singularities(model, q):
                    continue
                solutions = ur_flange_ik(model, _flange(model, q))
                assert solutions is not None
                if len(solutions) != 8:
                    continue
                checked += 1
                with self.subTest(model=model, q=[round(float(v), 3) for v in q]):
                    read = [ur_branch(model, s) for s in solutions]
                    self.assertEqual([URBranch(*signs) for signs in self._ORDER], read)

    def test_a_full_turn_of_any_joint_is_the_same_branch(self) -> None:
        q = [0.3, -1.4, 1.4, -1.6, -1.5, 0.2]
        held = ur_branch("ur10", q)
        for axis in range(6):
            for turn in (-2.0 * math.pi, 2.0 * math.pi):
                turned = list(q)
                turned[axis] += turn
                with self.subTest(axis=axis, turn=turn):
                    self.assertEqual(held, ur_branch("ur10", turned))

    def test_the_candle_straight_home_is_at_every_branch_point_and_agrees_with_every_branch(self) -> None:
        candle = [0.0, -math.pi / 2.0, 0.0, -math.pi / 2.0, 0.0, 0.0]
        home = ur_branch("ur10", candle)
        self.assertEqual(URBranch(0, 0, 0), home)
        assert home is not None
        for signs in self._ORDER:
            self.assertTrue(home.agrees_with(URBranch(*signs)))
        # Encoder noise either side of it reads the same: this is what the tolerance is for.
        for noise in (-1e-5, 1e-5):
            self.assertEqual(URBranch(0, 0, 0), ur_branch("ur10", [v + noise for v in candle]))
        self.assertIn("+/-", home.render())

    def test_a_joint_just_past_the_tolerance_takes_a_side(self) -> None:
        q = [0.3, -1.4, 1.4, -1.6, 0.0, 0.2]
        for wrist, side in ((BRANCH_POINT_TOL_RAD * 1.5, 1), (-BRANCH_POINT_TOL_RAD * 1.5, -1),
                            (BRANCH_POINT_TOL_RAD * 0.5, 0), (math.pi - BRANCH_POINT_TOL_RAD * 0.5, 0)):
            with self.subTest(wrist=wrist):
                branch = ur_branch("ur10", [*q[:4], wrist, q[5]])
                assert branch is not None
                self.assertEqual(side, branch.wrist)

    def test_agreement_is_the_same_side_or_a_branch_point_on_every_one_of_the_three(self) -> None:
        self.assertTrue(URBranch(1, -1, 1).agrees_with(URBranch(1, -1, 1)))
        self.assertTrue(URBranch(1, 0, 1).agrees_with(URBranch(1, -1, 1)))
        self.assertFalse(URBranch(1, -1, 1).agrees_with(URBranch(1, 1, 1)))
        self.assertFalse(URBranch(-1, -1, 1).agrees_with(URBranch(1, -1, 1)))
        self.assertEqual("shoulder +, wrist 2 -, elbow +", URBranch(1, -1, 1).render())

    def test_no_table_is_none_and_a_bad_configuration_is_refused(self) -> None:
        self.assertIsNone(ur_branch("kuka_kr6", [0.0] * 6))
        for joints in ([0.0] * 5, [0.0, 0.0, float("nan"), 0.0, 0.0, 0.0]):
            with self.subTest(joints=joints), self.assertRaises(ValueError):
                ur_branch("ur10", joints)


if __name__ == "__main__":
    unittest.main()
