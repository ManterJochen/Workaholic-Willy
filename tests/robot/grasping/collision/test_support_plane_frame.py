"""The support plane has to be in the same frame as the poses it is checked against.

``_apply_collision_filters`` runs at ``calculator.py`` before ``_to_base_frame_only``, so the candidates
it sees are still in CAMERA frame. ``resolve_support_plane`` produces a BASE-frame plane. Wiring those
together without a conversion did not fail loudly -- it computed a camera depth and called it a height:

    measured over 730 candidates from 23 reference scenes
    clearance the check SHOULD compute (BASE)     median   -7.01 mm   ->  65.5 % below the 5 mm threshold
    clearance the check ACTUALLY computed (CAM)   median  577.04 mm   ->   0.4 % below it

That is why ``rejected_table`` was 0 in 2128 of 2128 telemetry lines while the independent datagen
reference rejected a third of the same rank-0 candidates for ``below_table``. This file is the test that
would have caught it: it checks the conversion arithmetic, that the tag defaults to the old meaning, and
that a BASE plane with no transform is REFUSED rather than quietly skipped.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.geometry import Frame
from src.robot.grasping.collision import SupportPlane, resolve_support_plane


def _camera_above_table(height_mm: float = 700.0) -> np.ndarray:
    """CAMERA -> BASE for a camera looking straight down from ``height_mm``."""
    transform = np.eye(4)
    transform[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    transform[:3, 3] = np.array([0.0, 0.0, height_mm])
    return transform


class FrameTagTests(unittest.TestCase):
    def test_the_default_is_camera_so_every_older_caller_is_unchanged(self) -> None:
        self.assertIs(SupportPlane().frame, Frame.CAMERA)

    def test_the_resolver_produces_base_because_that_is_what_it_measures(self) -> None:
        self.assertIs(resolve_support_plane(declared_height_mm=0.0).plane.frame, Frame.BASE)

    def test_a_camera_plane_is_not_rotated_twice(self) -> None:
        plane = SupportPlane(normal=np.array([0.0, 0.0, 1.0]), offset_mm=12.0)
        self.assertIs(plane.to_camera_frame(_camera_above_table()), plane)


class ConversionTests(unittest.TestCase):
    def test_a_point_on_the_table_has_exactly_zero_clearance(self) -> None:
        """The one number the whole conversion exists to get right."""
        plane = SupportPlane(np.array([0.0, 0.0, 1.0]), 0.0, Frame.BASE)
        camera = plane.to_camera_frame(_camera_above_table(700.0))
        # The table, seen by a camera 700 mm above it, is 700 mm along the camera's own view axis.
        self.assertAlmostEqual(camera.clearance_mm(np.array([[0.0, 0.0, 700.0]])), 0.0, places=9)

    def test_a_point_above_the_table_reads_as_positive_clearance(self) -> None:
        plane = SupportPlane(np.array([0.0, 0.0, 1.0]), 0.0, Frame.BASE)
        camera = plane.to_camera_frame(_camera_above_table(700.0))
        self.assertAlmostEqual(camera.clearance_mm(np.array([[0.0, 0.0, 650.0]])), 50.0, places=9)
        self.assertAlmostEqual(camera.clearance_mm(np.array([[0.0, 0.0, 730.0]])), -30.0, places=9)

    def test_a_raised_support_height_moves_the_plane_by_the_same_amount(self) -> None:
        """A part standing on another part: the resolver raises the height, the camera plane follows."""
        low = SupportPlane(np.array([0.0, 0.0, 1.0]), 0.0, Frame.BASE).to_camera_frame(
            _camera_above_table(700.0))
        high = SupportPlane(np.array([0.0, 0.0, 1.0]), 40.0, Frame.BASE).to_camera_frame(
            _camera_above_table(700.0))
        point = np.array([[0.0, 0.0, 650.0]])
        self.assertAlmostEqual(low.clearance_mm(point) - high.clearance_mm(point), 40.0, places=9)

    def test_the_conversion_agrees_with_transforming_the_points_instead(self) -> None:
        """The property that matters: converting the PLANE must equal converting the POINTS.

        Checked on a tilted, translated camera rather than a convenient axis-aligned one, because the
        defect this replaces was invisible in exactly the easy case.
        """
        rng = np.random.default_rng(20260814)
        angle = np.radians(37.0)
        rot_x = np.array([[1.0, 0.0, 0.0],
                          [0.0, np.cos(angle), -np.sin(angle)],
                          [0.0, np.sin(angle), np.cos(angle)]])
        angle_z = np.radians(-24.0)
        rot_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0.0],
                          [np.sin(angle_z), np.cos(angle_z), 0.0],
                          [0.0, 0.0, 1.0]])
        transform = np.eye(4)
        transform[:3, :3] = rot_z @ rot_x
        transform[:3, 3] = np.array([310.0, -220.0, 640.0])

        base_plane = SupportPlane(np.array([0.0, 0.1, 0.99]), 17.5, Frame.BASE)
        camera_plane = base_plane.to_camera_frame(transform)

        points_cam = rng.uniform(-300.0, 300.0, size=(64, 3))
        points_base = points_cam @ transform[:3, :3].T + transform[:3, 3]
        np.testing.assert_allclose(camera_plane.signed_distances_mm(points_cam),
                                   base_plane.signed_distances_mm(points_base), atol=1e-9)

    def test_the_rotated_normal_stays_a_unit_vector(self) -> None:
        camera = SupportPlane(np.array([0.0, 0.2, 0.98]), 5.0, Frame.BASE).to_camera_frame(
            _camera_above_table(512.0))
        self.assertAlmostEqual(float(np.linalg.norm(camera.normal)), 1.0, places=12)


class TheCheckActuallyFiresTests(unittest.TestCase):
    """The regression itself: a grasp that puts the fingertip under the table must be refused.

    Asserted at ``validate_grasp_collision`` rather than through the generator, because the point is
    what the CHECK decides, and a scene-shaped test would also be testing candidate synthesis.
    """

    @staticmethod
    def _pose_above_the_table(height_mm: float, camera_to_base: np.ndarray):  # noqa: ANN205
        """A top-down grasp whose centre sits ``height_mm`` above the table, in CAMERA frame."""
        from src.robot.grasping.planning import GraspPose

        # Columns are [closing, binormal, approach] -- the approach points straight down at the table.
        rotation_base = np.column_stack([np.array([1.0, 0.0, 0.0]),
                                         np.array([0.0, -1.0, 0.0]),
                                         np.array([0.0, 0.0, -1.0])])
        position_base = np.array([0.0, 0.0, height_mm])
        rot_cb, trans_cb = camera_to_base[:3, :3], camera_to_base[:3, 3]
        return GraspPose(
            position_mm=rot_cb.T @ (position_base - trans_cb),
            rotation_matrix=rot_cb.T @ rotation_base,
            grip_width_mm=40.0, score=0.5, confidence=0.5,
            contacts=(np.array([-20.0, 0.0, 0.0]), np.array([20.0, 0.0, 0.0])),
        )

    def test_a_grasp_whose_fingertip_is_under_the_table_is_rejected(self) -> None:
        """The 2F-85 reaches 28.72 mm PAST the grasp centre, so a 10 mm-high grasp is 18.7 mm under."""
        from src.robot.grasping.collision import validate_grasp_collision

        transform = _camera_above_table(700.0)
        pose = self._pose_above_the_table(10.0, transform)
        plane = SupportPlane(np.array([0.0, 0.0, 1.0]), 0.0, Frame.BASE).to_camera_frame(transform)
        result = validate_grasp_collision(pose, support_plane=plane)
        self.assertIn("table_clearance", result.reasons)
        self.assertLess(result.min_table_clearance_mm or 0.0, 0.0)

    def test_the_same_grasp_high_above_the_table_is_accepted(self) -> None:
        from src.robot.grasping.collision import validate_grasp_collision

        transform = _camera_above_table(700.0)
        pose = self._pose_above_the_table(120.0, transform)
        plane = SupportPlane(np.array([0.0, 0.0, 1.0]), 0.0, Frame.BASE).to_camera_frame(transform)
        result = validate_grasp_collision(pose, support_plane=plane)
        self.assertTrue(result.valid, msg=f"unexpectedly refused: {result.reasons}")

    def test_without_the_conversion_the_check_silently_passes_everything(self) -> None:
        """Pins the DEFECT, so a future refactor that drops the conversion fails here and not in a bin.

        The unconverted BASE plane measures a camera depth: hundreds of millimetres of "clearance" for
        a grasp that is in fact under the table.
        """
        from src.robot.grasping.collision import validate_grasp_collision

        transform = _camera_above_table(700.0)
        pose = self._pose_above_the_table(10.0, transform)
        unconverted = SupportPlane(np.array([0.0, 0.0, 1.0]), 0.0)   # the old, untagged meaning
        result = validate_grasp_collision(pose, support_plane=unconverted)
        self.assertTrue(result.valid)
        self.assertGreater(result.min_table_clearance_mm or 0.0, 500.0)


class FailClosedTests(unittest.TestCase):
    def test_a_base_plane_without_a_transform_is_refused_not_skipped(self) -> None:
        """Silently skipping a safety check is precisely how the original defect survived."""
        from src.robot.grasping.generation.calculator import GraspCalculator

        calculator = GraspCalculator()
        depth = np.full((32, 32), 500.0)
        mask = np.zeros((32, 32), dtype=bool)
        mask[12:20, 12:20] = True

        class _Seg:
            def __init__(self, m: np.ndarray) -> None:
                self.mask = m

        with self.assertRaises(ValueError) as caught:
            calculator.compute(
                _Seg(mask), depth, unit="mm",
                support_plane=SupportPlane(np.array([0.0, 0.0, 1.0]), 0.0, Frame.BASE))
        self.assertIn("BASE", str(caught.exception))

    def test_a_bad_transform_shape_is_refused(self) -> None:
        plane = SupportPlane(np.array([0.0, 0.0, 1.0]), 0.0, Frame.BASE)
        with self.assertRaises(ValueError):
            plane.to_camera_frame(np.eye(3))
        with self.assertRaises(ValueError):
            plane.to_camera_frame(np.full((4, 4), np.nan))


if __name__ == "__main__":
    unittest.main()
