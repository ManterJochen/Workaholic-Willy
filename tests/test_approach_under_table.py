"""The approach corridor may not pass under the table — a gap where two checks deferred to each other.

`APPROACH_BLOCKED` discarded every swept sample below the table with the comment "below-table samples
are the BELOW_TABLE check's job". `BELOW_TABLE` tests the CLOSED fingertips at the grasp point, not the
corridor. Neither looked, and the reference accepted grasps the gripper would have to reach up to from
under the table.

MEASURED on v1_proof, 25,043 jaw labels:

    32.2 %  approached from BELOW (approach z > 0), tilt 119.7 to 180 deg
    24.8 %  would have the gripper START under the table
    32.1 %  of upward labels held in physics, against 75.7 % for the rest
    96.9 %  of upward label trials REFUSED in the shake, against 1.0 % -- the whole unexplained
            28 % refusal rate of the `label` stratum was this

THE CHECK IS ON THE CORRIDOR BEHIND THE GRASP, and that narrowing is itself a measurement. Testing the
whole swept envelope refuses 1,429 grasps that hold 70.8 % of the time: for a downward approach the
part that dips under the table is the fingertip AHEAD of the grasp, and a gripper may set a fingertip
on the table. Where the start pose is under the table, the shake refused 98.4 % and 0.0 % of the twelve
scored trials held.
"""

from __future__ import annotations

import unittest

import numpy as np

from datagen.grasps.shapes import solid_from_scene
from datagen.grasps.verdict import JawGrasp, JawModel, Rejection, check_jaw_grasp


def _box(height_mm: float = 160.0, width_mm: float = 40.0):
    """A box standing on the table, centred over the origin."""
    return solid_from_scene("box", (width_mm, width_mm, height_mm),
                            (0.0, 0.0, height_mm / 2.0), (0.0, 0.0, 0.0, 1.0),
                            instance_id=0, asset_id="probe")


#: The grasp height these tests use, and it is CHOSEN rather than round. `JawModel` measures
#: `finger_behind_mm` 33.37 and `approach_clearance_mm` 80.0, so an upward approach puts the closed
#: fingertips at h - 33.37 and the start of the corridor at h - 113.37. Only inside
#: 33.4 mm < h < 113.4 mm does the corridor reach under the table while the fingertips do not -- below
#: that window `BELOW_TABLE` fires first and this check is never reached, which is exactly what the
#: first version of this test measured instead of what it meant to.
_GRASP_Z_MM = 80.0


class ApproachUnderTableTests(unittest.TestCase):
    def test_a_grasp_reached_from_underneath_is_refused(self) -> None:
        target = _box()
        verdict = check_jaw_grasp(
            JawGrasp(np.array([0.0, 0.0, _GRASP_Z_MM]), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), 40.0),
            target, (), model=JawModel(), table_z_mm=0.0)
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.reason, Rejection.APPROACH_UNDER_TABLE)
        self.assertIn("under the table", verdict.detail)

    def test_the_same_grasp_from_above_is_not_refused_for_that_reason(self) -> None:
        """The control. Only the approach sign differs, so anything else that changed is a bug."""
        target = _box()
        verdict = check_jaw_grasp(
            JawGrasp(np.array([0.0, 0.0, _GRASP_Z_MM]), np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0]), 40.0),
            target, (), model=JawModel(), table_z_mm=0.0)
        self.assertNotEqual(verdict.reason, Rejection.APPROACH_UNDER_TABLE)

    def test_a_downward_grasp_low_on_the_object_keeps_its_old_verdict(self) -> None:
        """The over-rejection this check was NARROWED to avoid.

        A downward approach near the table dips its fingertips below it, and that is `BELOW_TABLE`'s
        call with its own tolerance -- not this one's. Testing the whole envelope instead refused 1,429
        such grasps that physics says hold 70.8 % of the time.
        """
        target = _box(height_mm=30.0)
        verdict = check_jaw_grasp(
            JawGrasp(np.array([0.0, 0.0, 15.0]), np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0]), 40.0),
            target, (), model=JawModel(), table_z_mm=0.0)
        self.assertNotEqual(verdict.reason, Rejection.APPROACH_UNDER_TABLE)

    def test_the_table_height_is_respected_rather_than_assumed_zero(self) -> None:
        """Raise the table and the same upward approach is refused by more, not less."""
        target = _box()
        for table_z in (0.0, 10.0):
            with self.subTest(table_z=table_z):
                verdict = check_jaw_grasp(
                    JawGrasp(np.array([0.0, 0.0, _GRASP_Z_MM]), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), 40.0),
                    target, (), model=JawModel(), table_z_mm=table_z)
                self.assertEqual(verdict.reason, Rejection.APPROACH_UNDER_TABLE)

    def test_the_reason_is_distinct_from_below_table(self) -> None:
        """Widening BELOW_TABLE would make every count from before today incomparable with after."""
        self.assertNotEqual(Rejection.APPROACH_UNDER_TABLE, Rejection.BELOW_TABLE)
        self.assertEqual(str(Rejection.APPROACH_UNDER_TABLE), "approach_under_table")


if __name__ == "__main__":
    unittest.main()
