"""Tests for the suction approach-geometry helpers (planning/suction_approach.py).

Pure NumPy + geometry, no Isaac: the free-roll tool orientation, the standoff/contact/retreat
poses, and the SuctionGrasp -> pose-recipe adapter (including the GraspFrame -> Frame bridge).
"""

from __future__ import annotations

import unittest

import numpy as np

from src.geometry import Frame, rotate_vector, to_rotation_matrix
from src.robot.grasping.types.grasp_point import GraspFrame
from src.robot.grasping.planning import (
    SuctionApproach,
    suction_approach_waypoints,
    suction_grasp_poses,
    suction_pose,
    suction_tool_orientation,
)
from src.robot.grasping.suction.synthesis import SuctionGrasp


def _unit(v: list[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    return a / np.linalg.norm(a)


class SuctionToolOrientationTests(unittest.TestCase):
    def test_tool_z_equals_approach_top_down(self) -> None:
        approach = np.array([0.0, 0.0, -1.0])
        quat = suction_tool_orientation(approach)
        self.assertAlmostEqual(float(np.linalg.norm(quat)), 1.0, places=9)
        # Tool +Z rotated into the world must be the approach direction.
        np.testing.assert_allclose(rotate_vector(quat, np.array([0.0, 0.0, 1.0])), approach, atol=1e-9)

    def test_tool_z_equals_approach_oblique(self) -> None:
        approach = _unit([1.0, 0.0, -1.0])
        quat = suction_tool_orientation(approach)
        np.testing.assert_allclose(rotate_vector(quat, np.array([0.0, 0.0, 1.0])), approach, atol=1e-9)

    def test_rotation_is_proper_orthonormal(self) -> None:
        # Even for an oblique approach the closing axis is normalised, so R stays a proper rotation.
        rot = to_rotation_matrix(suction_tool_orientation(_unit([0.3, 0.4, -0.866])))
        np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(rot)), 1.0, places=9)

    def test_zero_approach_rejected(self) -> None:
        with self.assertRaises(ValueError):
            suction_tool_orientation(np.zeros(3))


class SuctionPoseTests(unittest.TestCase):
    def test_standoff_offsets_along_negative_approach(self) -> None:
        contact = np.array([100.0, 50.0, 500.0])
        approach = np.array([0.0, 0.0, -1.0])
        pose = suction_pose(contact, approach, frame=Frame.BASE, standoff_mm=60.0)
        # -approach = +Z, so the cup hovers 60 mm above the contact.
        np.testing.assert_allclose(pose.position_mm, contact + np.array([0.0, 0.0, 60.0]))
        self.assertEqual(pose.frame, Frame.BASE)
        np.testing.assert_allclose(pose.quaternion_xyzw, suction_tool_orientation(approach))

    def test_zero_standoff_is_contact(self) -> None:
        contact = np.array([10.0, 20.0, 30.0])
        pose = suction_pose(contact, np.array([0.0, 0.0, -1.0]), standoff_mm=0.0)
        np.testing.assert_allclose(pose.position_mm, contact)

    def test_negative_standoff_rejected(self) -> None:
        with self.assertRaises(ValueError):
            suction_pose(np.zeros(3), np.array([0.0, 0.0, -1.0]), standoff_mm=-1.0)


class SuctionApproachWaypointsTests(unittest.TestCase):
    def test_inclusive_endpoints_and_linear(self) -> None:
        contact = np.array([0.0, 0.0, 400.0])
        approach = np.array([0.0, 0.0, -1.0])
        wps = suction_approach_waypoints(
            contact, approach, frame=Frame.BASE, standoff_mm=80.0, end_offset_mm=20.0, num_waypoints=4
        )
        self.assertEqual(len(wps), 4)
        # First = 80 mm back (above), last = 20 mm back, all along +Z.
        np.testing.assert_allclose(wps[0].position_mm, contact + np.array([0.0, 0.0, 80.0]))
        np.testing.assert_allclose(wps[-1].position_mm, contact + np.array([0.0, 0.0, 20.0]))
        direction = wps[-1].position_mm - wps[0].position_mm
        for wp in wps[1:-1]:
            offset = wp.position_mm - wps[0].position_mm
            self.assertLess(float(np.linalg.norm(np.cross(direction, offset))), 1e-9)
        for wp in wps:
            self.assertEqual(wp.frame, Frame.BASE)
            np.testing.assert_allclose(wp.quaternion_xyzw, suction_tool_orientation(approach))

    def test_too_few_waypoints_rejected(self) -> None:
        with self.assertRaises(ValueError):
            suction_approach_waypoints(np.zeros(3), np.array([0.0, 0.0, -1.0]), num_waypoints=1)


class SuctionGraspAdapterTests(unittest.TestCase):
    def _grasp(self, frame: GraspFrame = GraspFrame.BASE) -> SuctionGrasp:
        return SuctionGrasp(
            position_mm=np.array([120.0, -30.0, 450.0]),
            approach=np.array([0.0, 0.0, -1.0]),
            seal_score=0.9,
            quality=0.85,
            frame=frame,
        )

    def test_recipe_offsets_and_frame_bridge(self) -> None:
        grasp = self._grasp(GraspFrame.BASE)
        recipe = suction_grasp_poses(grasp, standoff_mm=60.0, contact_gap_mm=25.0, lift_mm=100.0)
        self.assertIsInstance(recipe, SuctionApproach)
        pos = grasp.position_mm
        # -approach = +Z: hover 60, contact 25, retreat 125 above the contact point.
        np.testing.assert_allclose(recipe.pre_grasp.position_mm, pos + np.array([0.0, 0.0, 60.0]))
        np.testing.assert_allclose(recipe.contact.position_mm, pos + np.array([0.0, 0.0, 25.0]))
        np.testing.assert_allclose(recipe.retreat.position_mm, pos + np.array([0.0, 0.0, 125.0]))
        # GraspFrame.BASE bridges to the geometry Frame.BASE.
        self.assertEqual(recipe.pre_grasp.frame, Frame.BASE)
        self.assertEqual(recipe.contact.frame, Frame.BASE)
        # Waypoints span standoff -> contact_gap, sharing the one tool orientation.
        np.testing.assert_allclose(recipe.waypoints[0].position_mm, pos + np.array([0.0, 0.0, 60.0]))
        np.testing.assert_allclose(recipe.waypoints[-1].position_mm, pos + np.array([0.0, 0.0, 25.0]))
        for pose in (recipe.pre_grasp, recipe.contact, recipe.retreat, *recipe.waypoints):
            np.testing.assert_allclose(pose.quaternion_xyzw, recipe.quaternion_xyzw)

    def test_camera_frame_bridges(self) -> None:
        recipe = suction_grasp_poses(self._grasp(GraspFrame.CAMERA))
        self.assertEqual(recipe.pre_grasp.frame, Frame.CAMERA)


if __name__ == "__main__":
    unittest.main()
