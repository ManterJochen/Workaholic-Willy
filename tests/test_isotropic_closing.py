"""Jaw-closing direction for silhouettes that do not imply one.

A near-isotropic mask (a cube from above) has a numerically degenerate principal axis: it is decided by
which of two nearly-equal eigenvalues comes out larger and flips between frames of the same static
scene. Following it is not neutral -- MEASURED on the cuRobo planner, a UR3e cannot reach a TANGENTIAL
top-down close at any azimuth (0/4) while a RADIAL one plans 4/4; a UR5e absorbs both, which is why
this stayed invisible until a shorter arm arrived.

These tests pin the two halves that matter: that the free yaw IS spent on the reachable direction, and
-- more importantly -- that it is NOT spent when the shape actually implies an axis.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.robot.grasping.generation._isotropic_closing import (
    ISOTROPIC_ASPECT_MIN,
    snap_isotropic_closing_to_radial,
)
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint

DOWN = np.array([0.0, 0.0, -1.0])


def _grasp(position, axis, *, aspect=0.98, approach=DOWN, frame=GraspFrame.BASE, **meta):
    return GraspPoint(
        position=np.asarray(position, dtype=np.float64),
        approach=np.asarray(approach, dtype=np.float64),
        axis=np.asarray(axis, dtype=np.float64),
        grip_width_mm=30.0,
        score=0.8,
        frame=frame,
        label="centroid",
        metadata={"aspect_ratio": aspect, **meta},
    )


def _radial_alignment(grasp) -> float:
    """|cos| between the closing axis and the radial direction; 1.0 == perfectly radial."""
    r = np.array([grasp.position[0], grasp.position[1]], dtype=np.float64)
    a = np.array([grasp.axis[0], grasp.axis[1]], dtype=np.float64)
    return abs(float(np.dot(r / np.linalg.norm(r), a / np.linalg.norm(a))))


class SnapsWhenTheAxisIsMeaninglessTests(unittest.TestCase):
    def test_tangential_close_on_a_symmetric_object_becomes_radial(self) -> None:
        """The exact failure: a cube at (350,0) closing along base +Y. MEASURED unreachable on a UR3e at
        every height above ~100 mm, so the pick fails at the standoff or, if that is lowered, at the
        retreat -- always, and only sometimes, because the degenerate axis flips run to run."""
        out = snap_isotropic_closing_to_radial([_grasp([350.0, 0.0, 37.0], [0.0, 1.0, 0.0])])
        self.assertAlmostEqual(_radial_alignment(out[0]), 1.0, places=9)
        self.assertAlmostEqual(abs(out[0].axis[0]), 1.0, places=9)  # radial at (350,0) IS base X

    def test_it_works_at_every_azimuth_not_just_on_the_x_axis(self) -> None:
        """"Prefer base +X" would be a hardcode that happens to be right on the +X axis. The measured
        principle is RADIAL, and the sweep that established it covered -90..180 deg."""
        for deg in (-90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0):
            a = math.radians(deg)
            position = [350.0 * math.cos(a), 350.0 * math.sin(a), 37.0]
            tangential = [-math.sin(a), math.cos(a), 0.0]
            out = snap_isotropic_closing_to_radial([_grasp(position, tangential)])
            self.assertAlmostEqual(_radial_alignment(out[0]), 1.0, places=9, msg=f"azimuth {deg}")

    def test_the_rotation_is_rigid_so_the_grasp_frame_survives(self) -> None:
        """Only the spin about the approach may change. If the approach tilted or the axes stopped being
        orthonormal, the executed pose would no longer be the grasp that was scored."""
        out = snap_isotropic_closing_to_radial([_grasp([250.0, 250.0, 37.0], [0.0, 1.0, 0.0])])[0]
        np.testing.assert_allclose(out.approach, DOWN, atol=1e-12)
        self.assertAlmostEqual(float(np.dot(out.approach, out.axis)), 0.0, places=12)
        self.assertAlmostEqual(float(np.linalg.norm(out.axis)), 1.0, places=12)

    def test_the_correction_is_recorded(self) -> None:
        out = snap_isotropic_closing_to_radial([_grasp([350.0, 0.0, 37.0], [0.0, 1.0, 0.0])])[0]
        self.assertIn("closing_snapped_to_radial_deg", out.metadata)
        self.assertAlmostEqual(abs(out.metadata["closing_snapped_to_radial_deg"]), 90.0, places=6)

    def test_score_and_geometry_are_untouched(self) -> None:
        """The snap must not perturb ranking: position, width and score carry through unchanged."""
        src = _grasp([350.0, 0.0, 37.0], [0.0, 1.0, 0.0])
        out = snap_isotropic_closing_to_radial([src])[0]
        np.testing.assert_allclose(out.position, src.position)
        self.assertEqual(out.score, src.score)
        self.assertEqual(out.grip_width_mm, src.grip_width_mm)
        self.assertEqual(out.label, src.label)


class LeavesMeaningfulAxesAloneTests(unittest.TestCase):
    def test_an_elongated_silhouette_keeps_its_axis(self) -> None:
        """THE regression guard. An elongated object's axis carries real information -- rewriting it
        would close the jaw across the LONG dimension, which is exactly the failure _ELONGATED_ASPECT_MAX
        exists to prevent (measured elsewhere: the on-edge sugar box went 5/5 -> 0/5)."""
        src = _grasp([350.0, 0.0, 37.0], [0.0, 1.0, 0.0], aspect=0.35)
        self.assertIs(snap_isotropic_closing_to_radial([src])[0], src)

    def test_the_threshold_is_a_hard_boundary(self) -> None:
        just_under = _grasp([350.0, 0.0, 37.0], [0.0, 1.0, 0.0], aspect=ISOTROPIC_ASPECT_MIN - 1e-6)
        just_over = _grasp([350.0, 0.0, 37.0], [0.0, 1.0, 0.0], aspect=ISOTROPIC_ASPECT_MIN)
        self.assertIs(snap_isotropic_closing_to_radial([just_under])[0], just_under)
        self.assertIsNot(snap_isotropic_closing_to_radial([just_over])[0], just_over)

    def test_a_tilted_approach_is_left_alone(self) -> None:
        """For an oblique approach the yaw is coupled to the approach direction and is not free."""
        tilted = np.array([0.6, 0.0, -0.8])  # ~37 deg from vertical
        src = _grasp([350.0, 0.0, 37.0], [0.0, 1.0, 0.0], approach=tilted)
        self.assertIs(snap_isotropic_closing_to_radial([src])[0], src)

    def test_camera_frame_candidates_are_left_alone(self) -> None:
        """"Radial" is defined about the ROBOT BASE; in camera frame it would be a rotation about an
        arbitrary axis through an arbitrary origin."""
        src = _grasp([350.0, 0.0, 37.0], [0.0, 1.0, 0.0], frame=GraspFrame.CAMERA)
        self.assertIs(snap_isotropic_closing_to_radial([src])[0], src)

    def test_a_missing_aspect_ratio_is_not_treated_as_isotropic(self) -> None:
        """Absence of evidence is not evidence of symmetry -- a candidate from a generator that does not
        publish the aspect must not be rewritten."""
        src = GraspPoint(
            position=np.array([350.0, 0.0, 37.0]), approach=DOWN, axis=np.array([0.0, 1.0, 0.0]),
            grip_width_mm=30.0, score=0.8, frame=GraspFrame.BASE, metadata={},
        )
        self.assertIs(snap_isotropic_closing_to_radial([src])[0], src)

    def test_an_already_radial_grasp_is_returned_unchanged(self) -> None:
        src = _grasp([350.0, 0.0, 37.0], [1.0, 0.0, 0.0])
        self.assertIs(snap_isotropic_closing_to_radial([src])[0], src)

    def test_a_grasp_over_the_base_has_no_radial_direction(self) -> None:
        src = _grasp([0.0, 0.0, 300.0], [0.0, 1.0, 0.0])
        self.assertIs(snap_isotropic_closing_to_radial([src])[0], src)

    def test_a_vertical_closing_axis_cannot_be_yawed(self) -> None:
        src = _grasp([350.0, 0.0, 37.0], [0.0, 0.0, 1.0], approach=np.array([1.0, 0.0, 0.0]))
        self.assertIs(snap_isotropic_closing_to_radial([src])[0], src)


class MinimalRotationTests(unittest.TestCase):
    def test_the_nearer_radial_sense_is_chosen(self) -> None:
        """A parallel jaw is symmetric and BOTH senses were measured equally reachable, so the snap
        should take the smaller rotation rather than always driving to +radial."""
        # Closing axis 170 deg from +radial: the nearer target is -radial, a 10 deg correction.
        a = math.radians(170.0)
        out = snap_isotropic_closing_to_radial(
            [_grasp([350.0, 0.0, 37.0], [math.cos(a), math.sin(a), 0.0])]
        )[0]
        self.assertLess(abs(out.metadata["closing_snapped_to_radial_deg"]), 15.0)
        self.assertAlmostEqual(_radial_alignment(out), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
