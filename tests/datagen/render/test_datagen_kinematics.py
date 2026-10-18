"""Where the arm stands, and whether it is honest about not being able to.

Pure numpy, no Isaac. What these tests can and cannot prove is worth stating plainly:

* they CAN prove the solver is self-consistent, respects joint limits, and reports failure instead of
  returning a near-miss;
* they CANNOT prove the arm really lands there, because the IK is solved against the same DH chain the
  check would use. That circularity is broken on-box by ``python -m datagen verify-robot``, which sets
  the joints on the Isaac articulation and reads the link pose back out of USD — a number computed
  from the asset, not from our table.

The mount tests are the exception and the important ones: they assert a geometric PROPERTY (the camera
looks at the point it was aimed at), which no convention bookkeeping can fake.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from datagen.render.camera import project_to_pixels
from datagen.render.kinematics import (
    PARK_JOINTS_RAD,
    camera_pose_from_joints,
    solve_joints_for_camera,
    wrist_link_pose_mm,
)
from datagen.robots import resolve_robot

_MODEL = "ur5e"


def _reachable_target(joints: np.ndarray, camera_to_link: np.ndarray) -> np.ndarray:
    """A camera pose that IS reachable, because it was generated from a real configuration."""
    return camera_pose_from_joints(_MODEL, joints, camera_to_link)


class ForwardKinematicsTests(unittest.TestCase):
    def test_the_park_pose_is_a_real_place(self) -> None:
        pose = wrist_link_pose_mm(_MODEL, np.asarray(PARK_JOINTS_RAD))
        self.assertTrue(np.all(np.isfinite(pose)))
        np.testing.assert_allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-9)
        self.assertGreater(float(np.linalg.norm(pose[:3, 3])), 100.0)

    def test_an_unknown_model_is_refused_rather_than_defaulted(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            wrist_link_pose_mm("franka", np.zeros(6))
        self.assertIn("DH table", str(ctx.exception))

    def test_a_wrong_joint_count_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            wrist_link_pose_mm(_MODEL, np.zeros(4))


class InverseKinematicsTests(unittest.TestCase):
    def setUp(self) -> None:
        mount = resolve_robot(_MODEL).wrist_camera_mount
        assert mount is not None
        self.camera_to_link = mount.camera_to_link()

    def test_it_recovers_configurations_it_was_given(self) -> None:
        """Self-consistency over a spread of poses -- stated as self-consistency, not as accuracy."""
        rng = np.random.default_rng(0)
        solved = 0
        for _ in range(20):
            joints = np.asarray(PARK_JOINTS_RAD) + rng.uniform(-0.35, 0.35, size=6)
            target = _reachable_target(joints, self.camera_to_link)
            result = solve_joints_for_camera(_MODEL, target, self.camera_to_link, seed_joints_rad=joints)
            if result is None:
                continue
            solved += 1
            reached = camera_pose_from_joints(_MODEL, result, self.camera_to_link)
            self.assertLess(float(np.linalg.norm(reached[:3, 3] - target[:3, 3])), 1.0)
        self.assertGreaterEqual(solved, 18, "the solver should handle poses generated from real configs")

    def test_an_unreachable_target_returns_None_not_a_near_miss(self) -> None:
        """The whole point. A 40 mm miss renders a convincing image with a camera pose that is a lie."""
        far = np.eye(4)
        far[:3, 3] = [4000.0, 0.0, 2000.0]
        self.assertIsNone(solve_joints_for_camera(_MODEL, far, self.camera_to_link))

    def test_it_stays_inside_the_joint_limits(self) -> None:
        """+-360 deg, the e-series' real range.

        This assertion said +-180 deg first, matching a limit that was itself wrong: the park
        configuration sits at 3.14 rad on joint 0, so that limit clamped every descent on its first
        step. Widening it to the robot's actual range took the solver from 25/40 to 40/40 and from
        127 ms to 1 ms per solve -- the limit was the whole problem, not the algorithm.
        """
        rng = np.random.default_rng(7)
        for _ in range(10):
            joints = np.asarray(PARK_JOINTS_RAD) + rng.uniform(-0.3, 0.3, size=6)
            result = solve_joints_for_camera(
                _MODEL, _reachable_target(joints, self.camera_to_link), self.camera_to_link,
            )
            if result is not None:
                self.assertTrue(bool(np.all(np.abs(result) <= 2.0 * math.pi + 1e-9)), f"{result}")

    def test_the_limit_does_not_exclude_the_robots_own_park_pose(self) -> None:
        """The regression. A joint limit that forbids where the arm rests is not a limit."""
        from datagen.render.kinematics import _JOINT_LIMIT_RAD

        self.assertTrue(
            bool(np.all(np.abs(np.asarray(PARK_JOINTS_RAD)) <= _JOINT_LIMIT_RAD)),
            "park is outside the joint limit, so every IK descent starts clamped",
        )

    def test_it_solves_from_no_seed_at_all(self) -> None:
        """The production path has no seed. Measured 40/40 to <1 mm; asserted well short of that."""
        rng = np.random.default_rng(1)
        solved = 0
        for _ in range(20):
            joints = np.asarray(PARK_JOINTS_RAD) + rng.uniform(-0.6, 0.6, size=6)
            target = _reachable_target(joints, self.camera_to_link)
            result = solve_joints_for_camera(_MODEL, target, self.camera_to_link)
            if result is None:
                continue
            reached = camera_pose_from_joints(_MODEL, result, self.camera_to_link)
            if float(np.linalg.norm(reached[:3, 3] - target[:3, 3])) < 1.0:
                solved += 1
        self.assertGreaterEqual(solved, 17, f"only {solved}/20 unseeded solves reached the target")

    def test_orientation_is_matched_not_only_position(self) -> None:
        """A camera at the right place looking the wrong way is a wrong label, not a small error."""
        joints = np.asarray(PARK_JOINTS_RAD) + np.array([0.2, -0.15, 0.1, 0.05, -0.1, 0.3])
        target = _reachable_target(joints, self.camera_to_link)
        result = solve_joints_for_camera(_MODEL, target, self.camera_to_link, seed_joints_rad=joints)
        assert result is not None
        reached = camera_pose_from_joints(_MODEL, result, self.camera_to_link)
        relative = target[:3, :3] @ reached[:3, :3].T
        angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(relative) - 1.0) / 2.0))))
        self.assertLess(angle_deg, 0.5)


class MountTests(unittest.TestCase):
    """These assert a geometric property, which is why they cannot be faked by a convention error."""

    def setUp(self) -> None:
        mount = resolve_robot(_MODEL).wrist_camera_mount
        assert mount is not None
        self.mount = mount
        self.camera_to_link = mount.camera_to_link()

    def test_the_camera_sits_at_the_measured_offset(self) -> None:
        np.testing.assert_allclose(self.camera_to_link[:3, 3], self.mount.offset_mm, atol=1e-9)

    def test_the_camera_LOOKS_AT_the_point_it_was_aimed_at(self) -> None:
        """The property test. A 180 deg roll passes every centred check and fails nothing else."""
        aim = np.asarray(self.mount.aim_target_mm, dtype=np.float64)
        intrinsics = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        pixel = project_to_pixels(np.asarray([aim]), self.camera_to_link, intrinsics)[0]
        np.testing.assert_allclose(pixel, [320.0, 240.0], atol=1e-6)

    def test_the_aim_point_is_IN_FRONT_of_the_camera(self) -> None:
        """+Z forward. If the mount were flipped, the aim point would land behind and project to NaN."""
        aim = np.asarray(self.mount.aim_target_mm, dtype=np.float64)
        in_camera = np.linalg.inv(self.camera_to_link) @ np.append(aim, 1.0)
        self.assertGreater(float(in_camera[2]), 0.0)

    def test_composing_the_mount_moves_the_camera_with_the_arm(self) -> None:
        """Two configurations, two camera poses -- the camera is carried, not fixed to the world."""
        first = camera_pose_from_joints(_MODEL, np.asarray(PARK_JOINTS_RAD), self.camera_to_link)
        moved = np.asarray(PARK_JOINTS_RAD) + np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
        second = camera_pose_from_joints(_MODEL, moved, self.camera_to_link)
        self.assertGreater(float(np.linalg.norm(first[:3, 3] - second[:3, 3])), 10.0)


if __name__ == "__main__":
    unittest.main()
