"""The production envelope and the independent reference must describe the SAME gripper.

They are deliberately separate code. The reference in ``datagen/grasps/`` never imports the calculator,
because a check that shares implementation with the thing it checks can agree for the wrong reason. But
the DIMENSIONS are hardware facts, not implementation, and letting two copies of a hardware fact drift
is how this happened:

* the config schema claimed its defaults "match the shipped 2F-85-class box model" and had
  ``fingertip_depth_mm = 6.0`` against a measured 28.72 -- 23 mm of finger, pointing at the table,
  invisible to every clearance check;
* the reference put its fingertip at half a pad length ahead of the contact, 20.75 mm against the same
  measured 28.72, so it was ~8 mm too lenient and every ``below_table`` count it produced was an
  under-count.

Neither was caught by a test, because no test compared them. This one does. It does not make either
side depend on the other; it fails when they stop describing the same piece of hardware.

Every number here was measured on 2026-08-14 off the 2F-85's own collision shapes, in the grasp frame,
as the worst case over apertures 10/25/40/55/70/82 mm -- the finger swings forward as the jaw closes,
so a single-aperture reading understates its reach toward the table.
"""

from __future__ import annotations

import unittest

from src.config.schema.robot.grasping_schema import GraspingParallelJawGeometryConfig
from src.robot.grasping.collision import ParallelJawGripperModel
from datagen.grasps.verdict import JawModel

#: The measurement. Both sides are compared against THIS, not against each other, so a test failure
#: names which side drifted instead of only that they disagree.
MEASURED_2F85_MM = {
    "reach_ahead": 28.72,      # past the grasp centre, toward the support surface
    "reach_behind": 33.37,     # back toward the wrist
    "thickness": 31.35,        # along the closing axis, outward from the contact face
    "width": 27.0,             # along the binormal
    "pad_length": 38.0,        # the contact patch along the approach
    "pad_ahead": 23.61,        # how far the patch reaches in front of the grasp centre
}
TOLERANCE_MM = 0.05


class DimensionsAgreeTests(unittest.TestCase):
    def test_the_config_schema_carries_the_measurement(self) -> None:
        cfg = GraspingParallelJawGeometryConfig()
        self.assertAlmostEqual(cfg.fingertip_depth_mm, MEASURED_2F85_MM["reach_ahead"],
                               delta=TOLERANCE_MM,
                               msg="fingertip_depth_mm is the reach TOWARD the table; it was 6.0")
        self.assertAlmostEqual(cfg.finger_length_mm, MEASURED_2F85_MM["reach_behind"] + 6.61,
                               delta=7.0,
                               msg="finger_length_mm is the reach back toward the wrist")
        self.assertAlmostEqual(cfg.finger_thickness_mm, MEASURED_2F85_MM["thickness"],
                               delta=TOLERANCE_MM)
        self.assertAlmostEqual(cfg.finger_width_mm, MEASURED_2F85_MM["width"], delta=TOLERANCE_MM)
        self.assertAlmostEqual(cfg.pad_length_mm, MEASURED_2F85_MM["pad_length"], delta=TOLERANCE_MM)
        self.assertAlmostEqual(cfg.pad_ahead_mm, MEASURED_2F85_MM["pad_ahead"], delta=TOLERANCE_MM)

    def test_the_collision_model_carries_the_measurement(self) -> None:
        model = ParallelJawGripperModel()
        self.assertAlmostEqual(model.fingertip_depth_mm, MEASURED_2F85_MM["reach_ahead"],
                               delta=TOLERANCE_MM)
        self.assertAlmostEqual(model.finger_thickness_mm, MEASURED_2F85_MM["thickness"],
                               delta=TOLERANCE_MM)
        self.assertAlmostEqual(model.finger_width_mm, MEASURED_2F85_MM["width"], delta=TOLERANCE_MM)
        self.assertAlmostEqual(model.pad_length_mm, MEASURED_2F85_MM["pad_length"],
                               delta=TOLERANCE_MM)
        self.assertAlmostEqual(model.pad_ahead_mm, MEASURED_2F85_MM["pad_ahead"],
                               delta=TOLERANCE_MM)

    def test_the_independent_reference_carries_the_same_measurement(self) -> None:
        jaw = JawModel()
        self.assertAlmostEqual(jaw.finger_ahead_mm, MEASURED_2F85_MM["reach_ahead"],
                               delta=TOLERANCE_MM,
                               msg="the reference's fingertip; it used to sit at half a pad length")
        self.assertAlmostEqual(jaw.finger_behind_mm, MEASURED_2F85_MM["reach_behind"],
                               delta=TOLERANCE_MM)
        self.assertAlmostEqual(jaw.finger_thickness_mm, MEASURED_2F85_MM["thickness"],
                               delta=TOLERANCE_MM)
        self.assertAlmostEqual(jaw.finger_width_mm, MEASURED_2F85_MM["width"], delta=TOLERANCE_MM)
        self.assertAlmostEqual(jaw.pad_ahead_mm + jaw.pad_behind_mm, MEASURED_2F85_MM["pad_length"],
                               delta=TOLERANCE_MM)
        self.assertAlmostEqual(jaw.pad_ahead_mm, MEASURED_2F85_MM["pad_ahead"], delta=TOLERANCE_MM)

    def test_both_sides_reach_the_same_distance_toward_the_support(self) -> None:
        """The single number that decides a table-clearance verdict, on both sides of the fence.

        This is the assertion the whole file exists for. When these two disagree, one of them lets a
        grasp through that the other rejects, and there is no way to tell from either side alone which
        is right.
        """
        self.assertAlmostEqual(ParallelJawGripperModel().fingertip_depth_mm,
                               JawModel().finger_ahead_mm, delta=TOLERANCE_MM)
        self.assertAlmostEqual(ParallelJawGripperModel().pad_ahead_mm,
                               JawModel().pad_ahead_mm, delta=TOLERANCE_MM,
                               msg="the CONTACT patch, which decides antipodality, must also agree")

    def test_the_reference_pad_is_asymmetric_because_the_gripper_is(self) -> None:
        """A symmetric pad was an assumption, and it put a third of the patch where there is none."""
        jaw = JawModel()
        self.assertNotAlmostEqual(jaw.pad_ahead_mm, jaw.pad_behind_mm, places=1)
        self.assertGreater(jaw.pad_ahead_mm, jaw.pad_behind_mm,
                           msg="the patch extends further toward the object than back toward the wrist")


if __name__ == "__main__":
    unittest.main()
