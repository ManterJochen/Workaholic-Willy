"""A motion is turned into the configurations a checker will judge, and the step is derived, not guessed.

Nothing checks the middle of a move unless the middle exists as configurations somebody can hand to a
checker. These are the pure functions that produce them, and the whole of their contract is: both ends
are always kept, no step moves any point of the arm further than the caller's bound, and a move too long
for the checker's cap is refused with a sentence rather than thinned until it fits.

The step rule is the owner's answer Q1 of 2026-09-11: a joint step keeps ``sum(|dq|)`` times the reach at
or below the bound, and a line step keeps the travel of the tool and the rotation over the reach at or
below the same bound. Both are upper bounds on how far any point of the arm can move between two
samples, so a gap the checker cannot see is bounded by a number the caller chose rather than by a
sample count somebody picked.

``slerp`` lives in the geometry package because the rotation half of a line is a rotation question, and
because that package may not import scipy (test_geometry_precision.py pins the ban).
"""

from __future__ import annotations

import unittest

import numpy as np

from src.geometry.frame import Frame
from src.geometry.pose import Pose
from src.geometry.quaternion import (
    angle_between,
    from_axis_angle,
    multiply,
    slerp,
)
from src.robot.safety.path_samples import (
    MAX_PATH_SAMPLES,
    LineSamples,
    PathSamples,
    joint_path_samples,
    line_samples,
    waypoint_path_samples,
)

_REACH_MM = 850.0  # a UR5e, near enough; every bound below is stated against it


def _pose(x: float, y: float, z: float, quat: np.ndarray | None = None) -> Pose:
    return Pose(
        position_mm=np.array([x, y, z], dtype=np.float64),
        quaternion_xyzw=(np.array([0.0, 0.0, 0.0, 1.0]) if quat is None else quat),
        frame=Frame.BASE,
    )


class SlerpTests(unittest.TestCase):
    """The rotation half of a line, on the geometry package's own terms."""

    def test_the_ends_are_the_ends(self) -> None:
        q0 = from_axis_angle(np.array([0.0, 0.0, 0.3]))
        q1 = from_axis_angle(np.array([0.2, 0.0, 1.1]))
        np.testing.assert_allclose(slerp(q0, q1, 0.0), q0, atol=1e-12)
        np.testing.assert_allclose(slerp(q0, q1, 1.0), q1, atol=1e-12)

    def test_the_midpoint_is_half_the_angle(self) -> None:
        """Half way along the arc is the rotation by half the angle, which is the whole claim."""
        axis = np.array([0.0, 0.0, 1.0])
        angle = 1.2
        q0 = from_axis_angle(np.zeros(3))
        q1 = from_axis_angle(axis * angle)
        expected = from_axis_angle(axis * (angle / 2.0))
        np.testing.assert_allclose(slerp(q0, q1, 0.5), expected, atol=1e-12)

    def test_the_arc_is_walked_at_a_constant_rate(self) -> None:
        """Constant speed along the arc is what makes a step bound mean anything."""
        q0 = from_axis_angle(np.array([0.0, 0.7, 0.0]))
        q1 = multiply(from_axis_angle(np.array([0.9, 0.0, 0.4])), q0)
        total = angle_between(q0, q1)
        for fraction in (0.25, 0.5, 0.75):
            with self.subTest(fraction=fraction):
                self.assertAlmostEqual(
                    angle_between(q0, slerp(q0, q1, fraction)), total * fraction, places=9
                )

    def test_the_short_way_round_is_taken(self) -> None:
        """q and -q are the same rotation; the arc between them is zero, not a full turn."""
        q0 = from_axis_angle(np.array([0.0, 0.0, 0.2]))
        self.assertLess(angle_between(slerp(q0, -q0, 0.5), q0), 1e-9)

    def test_a_fraction_outside_the_arc_is_refused(self) -> None:
        q0 = from_axis_angle(np.zeros(3))
        q1 = from_axis_angle(np.array([0.0, 0.0, 0.5]))
        for fraction in (-0.001, 1.001, float("nan")):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                slerp(q0, q1, fraction)


class JointSampleTests(unittest.TestCase):
    """A joint move, sampled so that no step moves the arm further than the bound."""

    def test_both_ends_are_kept(self) -> None:
        start = (0.0, -1.0, 0.9, 0.0, 1.5, 0.0)
        goal = (0.4, -1.0, 0.9, 0.0, 1.5, 0.0)
        samples = joint_path_samples(start, goal, reach_mm=_REACH_MM, max_step_mm=10.0)
        self.assertIsInstance(samples, PathSamples)
        self.assertEqual(start, samples.configs[0])
        self.assertEqual(goal, samples.configs[-1])

    def test_no_step_exceeds_the_bound(self) -> None:
        start = (0.0, -1.0, 0.9, 0.0, 1.5, 0.0)
        goal = (1.3, -0.2, 0.1, 0.7, 1.0, -0.6)
        samples = joint_path_samples(start, goal, reach_mm=_REACH_MM, max_step_mm=10.0)
        for index in range(1, len(samples.configs)):
            travel = sum(
                abs(b - a) for a, b in zip(samples.configs[index - 1], samples.configs[index])
            )
            with self.subTest(step=index):
                self.assertLessEqual(travel * _REACH_MM, 10.0 + 1e-9)
        self.assertLessEqual(samples.step_bound_mm, 10.0 + 1e-9)

    def test_the_count_is_derived_and_repeatable(self) -> None:
        """The same move gives the same samples, and the count is the rule and not a preference."""
        start = (0.0,) * 6
        goal = (0.1, 0.0, 0.0, 0.0, 0.0, 0.0)
        # sum(|dq|) * reach = 0.1 * 850 = 85 mm; at 10 mm a step that is 9 steps, so 10 configurations.
        samples = joint_path_samples(start, goal, reach_mm=_REACH_MM, max_step_mm=10.0)
        self.assertEqual(10, len(samples.configs))
        again = joint_path_samples(start, goal, reach_mm=_REACH_MM, max_step_mm=10.0)
        self.assertEqual(samples.configs, again.configs)

    def test_a_move_that_goes_nowhere_is_one_configuration(self) -> None:
        still = (0.0, -1.0, 0.9, 0.0, 1.5, 0.0)
        samples = joint_path_samples(still, still, reach_mm=_REACH_MM, max_step_mm=10.0)
        self.assertEqual((still,), samples.configs)
        self.assertEqual(0.0, samples.step_bound_mm)

    def test_a_move_past_the_backstop_is_refused_not_thinned(self) -> None:
        """Refuse and say so, rather than quietly leave gaps unchecked.

        The number moved on 2026-09-12 and the property did not. It used to be 1000, which was the
        cuRobo request size read as a rule about moves; it is now a bound on how long a caller waits,
        and a real move does not reach it. What a move that DID reach it gets is still a refusal.
        """
        start = (0.0,) * 6
        goal = (30.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # 30 rad * 850 mm = 25500 mm, 25500 steps at 1 mm
        with self.assertRaises(ValueError) as caught:
            joint_path_samples(start, goal, reach_mm=_REACH_MM, max_step_mm=1.0)
        message = str(caught.exception)
        self.assertIn(str(MAX_PATH_SAMPLES), message)
        self.assertIn("refused rather than thinned", message)

    def test_the_control_exactly_the_cap_is_produced(self) -> None:
        """Without this the refusal above could come from any rule, including an off by one."""
        step = 1.0
        total_rad = (MAX_PATH_SAMPLES - 1) * step / _REACH_MM
        samples = joint_path_samples(
            (0.0,) * 6, (total_rad, 0.0, 0.0, 0.0, 0.0, 0.0), reach_mm=_REACH_MM, max_step_mm=step
        )
        self.assertEqual(MAX_PATH_SAMPLES, len(samples.configs))

    def test_two_configurations_of_different_length_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            joint_path_samples((0.0, 0.0), (0.0, 0.0, 0.0), reach_mm=_REACH_MM, max_step_mm=10.0)

    def test_a_bound_that_is_not_positive_is_refused(self) -> None:
        for reach, step in ((0.0, 10.0), (_REACH_MM, 0.0), (-1.0, 10.0), (_REACH_MM, -10.0)):
            with self.subTest(reach=reach, step=step), self.assertRaises(ValueError):
                joint_path_samples((0.0,) * 6, (0.1,) * 6, reach_mm=reach, max_step_mm=step)


class LineSampleTests(unittest.TestCase):
    """A straight line in the base frame, position and orientation walked together."""

    def test_both_ends_are_kept_and_the_line_is_straight(self) -> None:
        start = _pose(0.0, 0.0, 0.0)
        goal = _pose(300.0, 0.0, 0.0)
        samples = line_samples(start, goal, reach_mm=_REACH_MM, max_step_mm=10.0)
        self.assertIsInstance(samples, LineSamples)
        np.testing.assert_allclose(samples.poses[0].position_mm, start.position_mm, atol=1e-12)
        np.testing.assert_allclose(samples.poses[-1].position_mm, goal.position_mm, atol=1e-12)
        direction = goal.position_mm - start.position_mm
        length = float(np.linalg.norm(direction))
        for index, pose in enumerate(samples.poses):
            offset = pose.position_mm - start.position_mm
            cross = float(np.linalg.norm(np.cross(offset, direction / length)))
            with self.subTest(sample=index):
                self.assertLess(cross, 1e-9, "the sample is off the line")

    def test_the_orientation_angle_rises_and_never_falls(self) -> None:
        start = _pose(0.0, 0.0, 0.0)
        goal = _pose(50.0, 0.0, 0.0, quat=from_axis_angle(np.array([0.0, 0.0, 0.8])))
        samples = line_samples(start, goal, reach_mm=_REACH_MM, max_step_mm=10.0)
        angles = [angle_between(start.quaternion_xyzw, p.quaternion_xyzw) for p in samples.poses]
        for index in range(1, len(angles)):
            with self.subTest(sample=index):
                self.assertGreaterEqual(angles[index] + 1e-12, angles[index - 1])
        self.assertAlmostEqual(0.8, angles[-1], places=9)

    def test_a_pure_rotation_is_sampled_by_the_rotation_alone(self) -> None:
        """A move that travels nowhere still moves the arm, and the reach is how far."""
        quat = from_axis_angle(np.array([0.0, 0.0, 0.2]))
        samples = line_samples(
            _pose(100.0, 0.0, 0.0), _pose(100.0, 0.0, 0.0, quat=quat),
            reach_mm=_REACH_MM, max_step_mm=10.0,
        )
        # 0.2 rad over 850 mm of reach is 170 mm, so 17 steps and 18 samples.
        self.assertEqual(18, len(samples.poses))
        self.assertGreater(len(samples.poses), 2, "a pure rotation is not one step")

    def test_no_step_exceeds_the_bound_in_either_half(self) -> None:
        start = _pose(0.0, 0.0, 0.0)
        goal = _pose(120.0, 40.0, -30.0, quat=from_axis_angle(np.array([0.3, 0.1, 0.5])))
        samples = line_samples(start, goal, reach_mm=_REACH_MM, max_step_mm=10.0)
        for index in range(1, len(samples.poses)):
            before, after = samples.poses[index - 1], samples.poses[index]
            travel = float(np.linalg.norm(after.position_mm - before.position_mm))
            turned = angle_between(before.quaternion_xyzw, after.quaternion_xyzw) * _REACH_MM
            with self.subTest(step=index):
                self.assertLessEqual(travel, 10.0 + 1e-9)
                self.assertLessEqual(turned, 10.0 + 1e-9)
        self.assertLessEqual(samples.step_bound_mm, 10.0 + 1e-9)

    def test_a_move_that_goes_nowhere_is_one_pose(self) -> None:
        pose = _pose(100.0, 0.0, 50.0)
        samples = line_samples(pose, pose, reach_mm=_REACH_MM, max_step_mm=10.0)
        self.assertEqual(1, len(samples.poses))
        self.assertEqual(0.0, samples.step_bound_mm)

    def test_a_line_past_the_backstop_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            line_samples(
                _pose(0.0, 0.0, 0.0), _pose(30000.0, 0.0, 0.0), reach_mm=_REACH_MM, max_step_mm=1.0
            )
        self.assertIn("refused rather than thinned", str(caught.exception))

    def test_two_frames_that_disagree_are_refused(self) -> None:
        start = _pose(0.0, 0.0, 0.0)
        goal = Pose(
            position_mm=np.array([10.0, 0.0, 0.0]),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            frame=Frame.CAMERA,
        )
        with self.assertRaises(ValueError) as caught:
            line_samples(start, goal, reach_mm=_REACH_MM, max_step_mm=10.0)
        self.assertIn("frame", str(caught.exception).lower())

    def test_the_samples_carry_the_frame_they_were_given(self) -> None:
        samples = line_samples(
            _pose(0.0, 0.0, 0.0), _pose(30.0, 0.0, 0.0), reach_mm=_REACH_MM, max_step_mm=10.0
        )
        for pose in samples.poses:
            self.assertEqual(Frame.BASE, pose.frame)


class PlannerWaypointTests(unittest.TestCase):
    """A planner hands over waypoints, and the arm runs the joint space line between each pair.

    Judging the waypoints alone judges the corners of the path and not the path: the controller moves
    from one to the next by interpolating, so the middle of every leg is executed and unexamined. The
    legs are sampled by the same rule as any other joint move, and the shared configuration between
    two legs appears once.
    """

    def test_the_waypoints_all_survive_in_order(self) -> None:
        waypoints = [(0.0,) * 6, (0.2, 0.0, 0.0, 0.0, 0.0, 0.0), (0.2, 0.3, 0.0, 0.0, 0.0, 0.0)]
        samples = waypoint_path_samples(waypoints, reach_mm=_REACH_MM, max_step_mm=10.0)
        for waypoint in waypoints:
            with self.subTest(waypoint=waypoint):
                self.assertIn(waypoint, samples.configs)
        self.assertEqual(waypoints[0], samples.configs[0])
        self.assertEqual(waypoints[-1], samples.configs[-1])

    def test_a_shared_configuration_appears_once(self) -> None:
        waypoints = [(0.0,) * 6, (0.2, 0.0, 0.0, 0.0, 0.0, 0.0), (0.4, 0.0, 0.0, 0.0, 0.0, 0.0)]
        samples = waypoint_path_samples(waypoints, reach_mm=_REACH_MM, max_step_mm=10.0)
        middle = (0.2, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.assertEqual(1, samples.configs.count(middle))

    def test_the_middle_of_a_leg_is_sampled(self) -> None:
        """The whole reason this exists: two waypoints far apart are not two configurations."""
        waypoints = [(0.0,) * 6, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
        samples = waypoint_path_samples(waypoints, reach_mm=_REACH_MM, max_step_mm=10.0)
        self.assertEqual(86, len(samples.configs))  # 1.0 rad * 850 mm / 10 mm = 85 steps

    def test_no_step_exceeds_the_bound_across_a_corner(self) -> None:
        waypoints = [(0.0,) * 6, (0.2, -0.1, 0.0, 0.0, 0.0, 0.0), (0.2, -0.1, 0.35, 0.0, 0.0, 0.0)]
        samples = waypoint_path_samples(waypoints, reach_mm=_REACH_MM, max_step_mm=10.0)
        for index in range(1, len(samples.configs)):
            travel = sum(
                abs(b - a) for a, b in zip(samples.configs[index - 1], samples.configs[index])
            )
            with self.subTest(step=index):
                self.assertLessEqual(travel * _REACH_MM, 10.0 + 1e-9)
        self.assertLessEqual(samples.step_bound_mm, 10.0 + 1e-9)

    def test_one_waypoint_is_one_configuration(self) -> None:
        samples = waypoint_path_samples([(0.1,) * 6], reach_mm=_REACH_MM, max_step_mm=10.0)
        self.assertEqual(((0.1,) * 6,), samples.configs)
        self.assertEqual(0.0, samples.step_bound_mm)

    def test_a_path_past_the_backstop_is_refused_across_all_its_legs(self) -> None:
        """The backstop is the whole path, not a leg: many legs under it can be over it together."""
        waypoints = [(float(i) * 0.8, 0.0, 0.0, 0.0, 0.0, 0.0) for i in range(40)]
        with self.assertRaises(ValueError) as caught:
            waypoint_path_samples(waypoints, reach_mm=_REACH_MM, max_step_mm=1.0)
        self.assertIn("refused rather than thinned", str(caught.exception))

    def test_an_empty_path_is_no_configurations(self) -> None:
        samples = waypoint_path_samples([], reach_mm=_REACH_MM, max_step_mm=10.0)
        self.assertEqual((), samples.configs)

    def test_waypoints_of_different_length_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            waypoint_path_samples(
                [(0.0,) * 6, (0.0,) * 5], reach_mm=_REACH_MM, max_step_mm=10.0
            )


class TheSamplesAreFrozenTests(unittest.TestCase):
    """A sample set a caller can edit is a checked path that stopped being the path that was checked."""

    def test_the_joint_samples_are_frozen(self) -> None:
        samples = joint_path_samples((0.0,) * 6, (0.1,) * 6, reach_mm=_REACH_MM, max_step_mm=10.0)
        with self.assertRaises(Exception):
            samples.configs = ()  # type: ignore[misc]

    def test_the_line_samples_are_frozen(self) -> None:
        samples = line_samples(
            _pose(0.0, 0.0, 0.0), _pose(30.0, 0.0, 0.0), reach_mm=_REACH_MM, max_step_mm=10.0
        )
        with self.assertRaises(Exception):
            samples.poses = ()  # type: ignore[misc]

    def test_the_safety_package_offers_what_a_driver_will_reach_for(self) -> None:
        """A driver builds these next to the gate that eats them; both come from the same package."""
        import src.robot.safety as safety

        for name in ("PathSamples", "LineSamples", "joint_path_samples", "line_samples",
                     "MAX_PATH_SAMPLES"):
            with self.subTest(name=name):
                self.assertIn(name, safety.__all__)
                self.assertTrue(hasattr(safety, name))

    def test_the_backstop_is_NOT_the_checkers_request_size(self) -> None:
        """⭐ THE PREMISE FLIPPED ON 2026-09-12, and the flip is the whole lesson.

        This used to assert the two numbers EQUAL, on the reasoning that a path this module builds
        should never be one the client refuses to send. That reasoning was wrong in one word: the
        client does not refuse a longer path, it splits it. Holding the sampler to the size of a
        request refused about one real cuRobo plan in ten on the M2 Isaac gate.
        """
        from src.robot.safety.planning import MAX_CHECK_CONFIGURATIONS

        self.assertGreater(
            MAX_PATH_SAMPLES, MAX_CHECK_CONFIGURATIONS,
            "the sampler is capped at the size of one request again",
        )


if __name__ == "__main__":
    unittest.main()
