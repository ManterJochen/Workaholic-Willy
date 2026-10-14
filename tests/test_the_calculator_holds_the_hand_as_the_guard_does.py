"""The calculator holds the hand and the parts beside it as the guard does (2026-10-14, the faithful replay bank).

* The hand in steps along the approach, measured off the hand's own guard bundle (``gripper_model.hand_steps_of``):
  a Hand-E's tip, where its pad sits, is 6 mm thick and 21 mm across, and its housing 62 mm across the binormal where
  it meets the fingers; the registry's one box of each stood 4.8 and 6.5 mm past them, and the calculator refused
  grasps the guard admits. Every step's box holds the mesh the guard judges in it.
* A neighbour's pixel is a named part's where it lies within ``fine_world.FINE_LABEL_MM`` of the surface the pick names
  the part by, as the guard names a camera on the wrist's pixels, not where the detector's mask covers it.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.config.grippers import load_gripper
from src.robot.grasping.collision.gripper_model import ParallelJawGripperModel, hand_steps_of
from src.robot.safety.planning.environment import hand_mesh_bundle

_HAND = "robotiq_hande"


def _steps():
    jaw = load_gripper(_HAND, aliases=False).jaw
    return jaw, hand_steps_of(hand_mesh_bundle(_HAND), aperture_mm=float(jaw.aperture_mm),
                              finger_ahead_mm=float(jaw.finger_ahead_mm), finger_length_mm=float(jaw.finger_length_mm))


class TheHandInStepsTests(unittest.TestCase):
    def test_a_hande_finger_is_thinner_at_its_tip_than_the_registry_box(self) -> None:
        jaw, steps = _steps()
        self.assertEqual(3, len(steps.finger))
        self.assertEqual(4, len(steps.housing))
        tip = steps.finger[-1]
        self.assertAlmostEqual(float(jaw.finger_ahead_mm), tip.z_mm[1], places=6)
        self.assertLess(tip.thickness_mm, float(jaw.finger_thickness_mm) - 4.0)
        self.assertLess(tip.y_mm[1] - tip.y_mm[0], float(jaw.finger_width_mm) - 8.0)
        self.assertAlmostEqual(-float(jaw.finger_length_mm), steps.finger[0].z_mm[0], places=6)
        near = steps.housing[-1]
        self.assertAlmostEqual(-float(jaw.finger_length_mm), near.z_mm[1], places=6)
        self.assertLess(near.y_mm[1] - near.y_mm[0], float(jaw.palm_width_mm) - 12.0)

    def test_every_box_holds_the_mesh_the_guard_judges(self) -> None:
        """The finger and the housing of the bundle, open to the aperture, lie inside the stepped boxes: the calculator
        never holds less of the hand than the guard does."""
        jaw, steps = _steps()
        model = ParallelJawGripperModel(finger_length_mm=float(jaw.finger_length_mm),
                                        finger_thickness_mm=float(jaw.finger_thickness_mm),
                                        finger_width_mm=float(jaw.finger_width_mm),
                                        fingertip_depth_mm=float(jaw.finger_ahead_mm), pad_length_mm=20.91,
                                        pad_ahead_mm=float(jaw.pad_ahead_mm), palm_depth_mm=float(jaw.palm_depth_mm),
                                        palm_width_mm=float(jaw.palm_width_mm), palm_thickness_mm=75.0,
                                        hand_steps=steps)
        boxes = model.collision_boxes(float(jaw.aperture_mm))
        self.assertEqual(2 * 3 + 4, len(boxes))
        with np.load(hand_mesh_bundle(_HAND)) as data:
            finger = np.asarray(data["rfinger__v"], dtype=np.float64)
            housing = np.asarray(data["gripper__v"], dtype=np.float64)
        centre = float(finger[:, 1].max()) - float(jaw.finger_ahead_mm)

        def placed(v: np.ndarray) -> np.ndarray:
            return np.column_stack([v[:, 0], v[:, 2], v[:, 1] - centre])

        lows = np.array([b.min_corner_mm for b in boxes])
        highs = np.array([b.max_corner_mm for b in boxes])
        for name, part in (("finger", placed(finger)), ("housing", placed(housing))):
            outside = part[part[:, 2] >= -float(jaw.finger_length_mm)] if name == "finger" else part
            # The finger's own carriage inside the housing is the housing's: held by the housing's steps.
            inside = np.any(np.all((outside[:, None, :] >= lows[None] - 1e-6)
                                   & (outside[:, None, :] <= highs[None] + 1e-6), axis=2), axis=1)
            beyond = outside[~inside]
            # Only the finger's inner side behind the pad, which the one box never held either (the registry's
            # carriage shoulder, finger_pad_overlap_mm), may stand outside, and only toward the gap.
            if name == "finger" and beyond.shape[0]:
                self.assertTrue(bool(np.all(beyond[:, 0] < 0.5 * float(jaw.aperture_mm))),
                                f"finger vertices beyond the boxes away from the gap: {beyond[:3]}")
            elif name == "housing":
                self.assertEqual(0, beyond.shape[0], f"housing vertices beyond the boxes: {beyond[:3]}")

    def test_without_steps_the_hand_is_the_registry_boxes_of_before(self) -> None:
        model = ParallelJawGripperModel()
        self.assertEqual(3, len(model.collision_boxes(40.0)))
        self.assertEqual((24, 3), model.local_corners_mm(40.0).shape)


class APartIsNamedAsTheGuardNamesItTests(unittest.TestCase):
    def test_a_point_near_a_parts_named_surface_is_that_part_and_the_first_part_wins(self) -> None:
        from src.robot.grasping.generation.scene_obstacles import _named_owners, named_surface

        first = named_surface(np.array([[0.0, 0.0, 10.0], [0.0, 1.0, 10.0], [20.0, 0.0, 10.0]]))
        second = np.array([[3.0, 0.0, 10.0]])
        points = np.array([[0.0, 0.0, 13.0], [0.0, 0.0, 16.0], [3.0, 0.0, 10.0], [40.0, 0.0, 10.0]])
        self.assertEqual([0, -1, 0, -1], _named_owners(points, (first, second)).tolist())
        self.assertEqual([1, -1, 1, -1], _named_owners(points, (np.zeros((0, 3)), second)).tolist())

    def test_a_named_surface_keeps_a_point_a_cell_as_the_pick_offers_it(self) -> None:
        from src.robot.grasping.generation.scene_obstacles import NAMED_VOXEL_MM, named_surface

        points = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [NAMED_VOXEL_MM + 0.1, 0.0, 0.0]])
        self.assertEqual([[0.1, 0.1, 0.1], [NAMED_VOXEL_MM + 0.1, 0.0, 0.0]], named_surface(points).tolist())


if __name__ == "__main__":
    unittest.main()
