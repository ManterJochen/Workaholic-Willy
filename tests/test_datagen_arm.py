"""Which configuration the arm holds, and the three ways a viewpoint gets refused.

All of it geometry, none of it Isaac. The case that matters most is the third: a viewpoint above a
pile is reachable, collision-free with itself, and puts the forearm through the objects — the arm is
the only part of the scene that can be wrong while every individual check says fine.
"""

from __future__ import annotations

import unittest

import numpy as np

from datagen.render.arm import ArmPose, closest_link_clearance_mm, resolve_arm_pose
from datagen.render.kinematics import (
    JOINT_FK_BAND_RAD,
    PARK_JOINTS_RAD,
    camera_pose_from_joints,
    sample_joint_configuration,
)
from datagen.robots import resolve_robot

_MODEL = "ur5e"


def _mount():  # noqa: ANN202 - a fixture
    mount = resolve_robot(_MODEL).wrist_camera_mount
    assert mount is not None
    return mount.camera_to_link()


class ClearanceTests(unittest.TestCase):
    def test_an_empty_scene_has_infinite_clearance(self) -> None:
        self.assertEqual(
            closest_link_clearance_mm(_MODEL, np.asarray(PARK_JOINTS_RAD), np.zeros((0, 3)), np.zeros(0)),
            float("inf"),
        )

    def test_an_object_far_away_is_clear(self) -> None:
        clearance = closest_link_clearance_mm(
            _MODEL, np.asarray(PARK_JOINTS_RAD), np.array([[2000.0, 2000.0, 0.0]]), np.array([50.0]),
        )
        self.assertGreater(clearance, 500.0)

    def test_an_object_ON_a_link_is_negative(self) -> None:
        """The signal the resolver refuses on: negative means the arm is inside the object."""
        from src.robot.safety._ur_kinematics import ur_link_origins_mm

        origins = ur_link_origins_mm(_MODEL, np.asarray(PARK_JOINTS_RAD))
        assert origins is not None
        midpoint = (origins[2] + origins[3]) / 2.0
        clearance = closest_link_clearance_mm(
            _MODEL, np.asarray(PARK_JOINTS_RAD), np.array([midpoint]), np.array([40.0]),
        )
        self.assertLess(clearance, 0.0)

    def test_a_bigger_object_reduces_clearance_by_its_radius(self) -> None:
        centre = np.array([[600.0, 300.0, 300.0]])
        small = closest_link_clearance_mm(_MODEL, np.asarray(PARK_JOINTS_RAD), centre, np.array([10.0]))
        large = closest_link_clearance_mm(_MODEL, np.asarray(PARK_JOINTS_RAD), centre, np.array([60.0]))
        self.assertAlmostEqual(small - large, 50.0, places=6)


class ResolveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_to_link = _mount()
        self.reachable = camera_pose_from_joints(
            _MODEL, np.asarray(PARK_JOINTS_RAD) + np.array([0.2, -0.2, 0.15, 0.0, 0.0, 0.0]),
            self.camera_to_link,
        )

    def _clear(self, _joints: np.ndarray) -> float:
        return 100.0

    def test_a_reachable_clear_viewpoint_is_used_on_the_first_attempt(self) -> None:
        pose, camera = resolve_arm_pose(
            _MODEL, self.camera_to_link, lambda _a: self.reachable,
            np.zeros((0, 3)), np.zeros(0), self_clearance=self._clear,
        )
        self.assertEqual(pose.origin, "posed")
        self.assertEqual(pose.attempts, 1)
        self.assertEqual(pose.rejections, ())
        self.assertIsNotNone(camera)

    def test_an_unreachable_viewpoint_parks_and_renders_no_wrist_view(self) -> None:
        far = np.eye(4)
        far[:3, 3] = [5000.0, 0.0, 3000.0]
        pose, camera = resolve_arm_pose(
            _MODEL, self.camera_to_link, lambda _a: far, np.zeros((0, 3)), np.zeros(0),
            max_attempts=3, self_clearance=self._clear,
        )
        self.assertTrue(pose.is_park)
        self.assertIsNone(camera, "no camera pose means the wrist view is skipped, not invented")
        self.assertEqual(pose.joints_rad, PARK_JOINTS_RAD)
        self.assertEqual(len(pose.rejections), 3)

    def test_self_collision_refuses_the_viewpoint(self) -> None:
        pose, _camera = resolve_arm_pose(
            _MODEL, self.camera_to_link, lambda _a: self.reachable, np.zeros((0, 3)), np.zeros(0),
            max_attempts=2, self_collision_margin_mm=8.0, self_clearance=lambda _j: 2.0,
        )
        self.assertTrue(pose.is_park)
        self.assertIn("self-collision", pose.rejections[0])

    def test_it_NEVER_returns_a_configuration_standing_inside_an_object(self) -> None:
        """The invariant, and it survived a wrong first draft of this very test.

        The draft blocked one link of one solution and asserted the arm parks. It does not — the
        solver's restarts find a different IK branch that reaches the SAME camera pose with the arm
        somewhere else, which is better than parking and was not what the test expected. So the thing
        worth asserting is not "it parks" but "whatever it returns is clear".
        """
        joints = np.asarray(PARK_JOINTS_RAD) + np.array([0.2, -0.2, 0.15, 0.0, 0.0, 0.0])
        from src.robot.safety._ur_kinematics import ur_link_origins_mm

        origins = ur_link_origins_mm(_MODEL, joints)
        assert origins is not None
        blocking = np.array([(origins[3] + origins[4]) / 2.0])
        radii = np.array([60.0])
        pose, camera = resolve_arm_pose(
            _MODEL, self.camera_to_link, lambda _a: self.reachable, blocking, radii,
            max_attempts=4, self_clearance=self._clear,
        )
        if pose.is_park:
            self.assertIsNone(camera)
            return
        self.assertGreaterEqual(
            closest_link_clearance_mm(_MODEL, np.asarray(pose.joints_rad), blocking, radii), 0.0,
            "a posed configuration must never be inside a scene object",
        )

    def test_an_object_swallowing_the_workspace_forces_a_park(self) -> None:
        """When no branch can avoid it, parking is the answer -- and the reason is recorded."""
        pose, camera = resolve_arm_pose(
            _MODEL, self.camera_to_link, lambda _a: self.reachable,
            np.array([[0.0, 0.0, 0.0]]), np.array([2000.0]),
            max_attempts=3, self_clearance=self._clear,
        )
        self.assertTrue(pose.is_park)
        self.assertIsNone(camera)
        self.assertIn("inside a scene object", pose.rejections[0])

    def test_a_later_attempt_can_succeed_and_is_counted(self) -> None:
        far = np.eye(4)
        far[:3, 3] = [5000.0, 0.0, 3000.0]

        def sample(attempt: int) -> np.ndarray:
            return self.reachable if attempt >= 2 else far

        pose, camera = resolve_arm_pose(
            _MODEL, self.camera_to_link, sample, np.zeros((0, 3)), np.zeros(0),
            max_attempts=6, self_clearance=self._clear,
        )
        self.assertEqual(pose.origin, "posed")
        self.assertEqual(pose.attempts, 3)
        self.assertEqual(len(pose.rejections), 2)
        self.assertIsNotNone(camera)

    def test_the_pose_is_frozen(self) -> None:
        pose = ArmPose((0.0,) * 6, "park", 1)
        with self.assertRaises(Exception):
            pose.origin = "posed"  # type: ignore[misc]


class JointFkTests(unittest.TestCase):
    """The second viewpoint source: draw a configuration, let FK place the camera."""

    def setUp(self) -> None:
        self.camera_to_link = _mount()

    def _clear(self, _joints: np.ndarray) -> float:
        return 100.0

    def test_the_first_draw_is_the_centre(self) -> None:
        """So joint_fk and target_ik agree exactly on the easy case and diverge only on a resample."""
        centre = np.asarray(PARK_JOINTS_RAD) + 0.3
        np.testing.assert_allclose(sample_joint_configuration(0, centre_rad=centre), centre)
        np.testing.assert_allclose(sample_joint_configuration(0), np.asarray(PARK_JOINTS_RAD))

    def test_draws_are_reproducible_and_differ_between_attempts(self) -> None:
        np.testing.assert_allclose(sample_joint_configuration(3), sample_joint_configuration(3))
        self.assertFalse(np.allclose(sample_joint_configuration(3), sample_joint_configuration(4)))

    def test_a_draw_stays_inside_the_band_around_its_centre(self) -> None:
        centre = np.asarray(PARK_JOINTS_RAD) + np.array([0.4, -0.3, 0.2, 0.1, 0.0, -0.2])
        for attempt in range(1, 40):
            offset = np.abs(sample_joint_configuration(attempt, centre_rad=centre) - centre)
            self.assertLessEqual(float(offset.max()), JOINT_FK_BAND_RAD + 1e-9)

    def test_the_centre_is_what_moves_the_draws(self) -> None:
        """The defect this parameter exists for: a band around park points the camera at nothing."""
        centre = np.asarray(PARK_JOINTS_RAD) + 1.0
        shifted = sample_joint_configuration(5, centre_rad=centre)
        np.testing.assert_allclose(shifted - sample_joint_configuration(5), np.ones(6), atol=1e-9)

    def test_no_viewpoint_is_ever_unreachable(self) -> None:
        """The whole point of the mode: there is no IK step, so there is nothing to fail at."""
        pose, camera = resolve_arm_pose(
            _MODEL, self.camera_to_link,
            lambda _a: (_ for _ in ()).throw(AssertionError("IK path must not be used")),
            np.zeros((0, 3)), np.zeros(0), self_clearance=self._clear,
            propose_joints=sample_joint_configuration,
        )
        self.assertEqual(pose.origin, "posed")
        self.assertEqual(pose.attempts, 1)
        self.assertIsNotNone(camera)

    def test_it_still_refuses_a_configuration_inside_the_objects(self) -> None:
        """Reachable by construction is not the same as usable, and the checks are shared."""
        from src.robot.safety._ur_kinematics import ur_link_origins_mm

        origins = ur_link_origins_mm(_MODEL, np.asarray(PARK_JOINTS_RAD))
        assert origins is not None
        pose, camera = resolve_arm_pose(
            _MODEL, self.camera_to_link, lambda _a: np.eye(4),
            np.array([[0.0, 0.0, 0.0]]), np.array([3000.0]),
            max_attempts=3, self_clearance=self._clear,
            propose_joints=sample_joint_configuration,
        )
        self.assertTrue(pose.is_park)
        self.assertIsNone(camera)
        self.assertIn("inside a scene object", pose.rejections[0])

    def test_a_camera_pointing_at_nothing_is_refused_and_says_so(self) -> None:
        pose, camera = resolve_arm_pose(
            _MODEL, self.camera_to_link, lambda _a: np.eye(4), np.zeros((0, 3)), np.zeros(0),
            max_attempts=2, self_clearance=self._clear,
            propose_joints=sample_joint_configuration,
            accept_camera=lambda _pose: "the camera faces away from the scene",
        )
        self.assertTrue(pose.is_park)
        self.assertIsNone(camera)
        self.assertIn("faces away", pose.rejections[0])

    def test_target_ik_is_not_given_the_camera_test(self) -> None:
        """A viewpoint the layout asked for must not be second-guessed by the joint_fk filter."""
        reachable = camera_pose_from_joints(
            _MODEL, np.asarray(PARK_JOINTS_RAD) + np.array([0.2, -0.2, 0.15, 0.0, 0.0, 0.0]),
            self.camera_to_link,
        )
        pose, camera = resolve_arm_pose(
            _MODEL, self.camera_to_link, lambda _a: reachable, np.zeros((0, 3)), np.zeros(0),
            self_clearance=self._clear,
        )
        self.assertEqual(pose.origin, "posed")
        self.assertIsNotNone(camera)


if __name__ == "__main__":
    unittest.main()
