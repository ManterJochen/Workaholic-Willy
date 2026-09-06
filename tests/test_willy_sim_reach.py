"""Reachability audit — does the CONFIGURED scene fit the CONFIGURED robot?

Every willy_sim scene position was authored for a UR5e (850 mm). A UR3e cell (500 mm) silently puts most
of them out of reach, and the symptom is an inscrutable IK/plan failure ~60 s into an Isaac boot -- or a
run that looks like bad grasping when it is really bad geometry. These tests pin the measurement.
"""

from __future__ import annotations

import unittest

from src.willy_sim.config import load_sim_config, require_robot
from src.willy_sim.harness.reach import (
    audit_reach,
    format_reach_report,
    shoulder_height_mm,
)


def _robot(model: str):
    robot = require_robot(load_sim_config(None))
    return robot.model_copy(update={"sim": robot.sim.model_copy(update={"robot_model": model})})


class ShoulderHeightTests(unittest.TestCase):
    def test_reach_sphere_is_centred_on_the_shoulder_not_the_base(self) -> None:
        """A UR's datasheet reach is the radius about the SHOULDER (d1 above the base). Measuring from the
        base origin would over-state the envelope by up to d1 and wave through untouchable poses."""
        self.assertAlmostEqual(shoulder_height_mm("ur3e"), 151.85, places=2)
        self.assertAlmostEqual(shoulder_height_mm("ur5e"), 162.5, places=2)
        self.assertEqual(shoulder_height_mm("not-a-robot"), 0.0)


class ReachAuditTests(unittest.TestCase):
    def test_ur3e_cannot_reach_the_ur5e_tuned_safe_pose(self) -> None:
        """safe_pose (450, -300, 400) is r~595 mm about the UR3e shoulder -- ~95 mm beyond its 500 mm
        sphere. The WorkspaceGuard would happily accept it and the retreat motion would then fail."""
        bad = {f.label: f for f in audit_reach(_robot("ur3e")) if not f.reachable}
        self.assertIn("safe_pose", bad)
        self.assertGreater(bad["safe_pose"].overshoot_mm, 50.0)

    def test_ur3e_can_still_reach_the_m1_object(self) -> None:
        """Good news for the bring-up: the M1 object at (450, 0, 25) is r~468 mm -- inside the UR3e sphere,
        so the core pick scene does NOT have to move (unlike the 515-644 mm bins)."""
        obj = next(f for f in audit_reach(_robot("ur3e")) if f.label == "scene_setup.object")
        self.assertTrue(obj.reachable, f"object at r={obj.radius_mm:.1f} mm should fit a UR3e")
        self.assertLess(obj.radius_mm, 500.0)

    def test_margin_shrinks_the_allowed_sphere(self) -> None:
        """Near the very edge the arm is near-singular with almost no orientation freedom, so a 'just
        barely inside' target is not usefully reachable for a top-down grasp."""
        strict = {f.label for f in audit_reach(_robot("ur3e"), margin_mm=60.0) if not f.reachable}
        loose = {f.label for f in audit_reach(_robot("ur3e"), margin_mm=0.0) if not f.reachable}
        self.assertTrue(loose.issubset(strict))
        self.assertIn("scene_setup.marker", strict)  # r~496 mm: inside 500, outside 440

    def test_ur5e_workspace_box_claims_an_unreachable_corner(self) -> None:
        """PRE-EXISTING (not a UR3e problem): the committed workspace_limits corner (700, -350, 600) is
        r~897 mm about the UR5e shoulder -- ~47 mm beyond its 850 mm reach. The WorkspaceGuard therefore
        admits targets the arm cannot touch. Pinned so the audit keeps reporting it."""
        corner = next(f for f in audit_reach(_robot("ur5e")) if f.label.startswith("workspace_limits"))
        self.assertFalse(corner.reachable)
        self.assertGreater(corner.overshoot_mm, 20.0)

    def test_eih_viewpoint_sphere_is_audited_not_just_the_marker(self) -> None:
        """The marker being reachable does NOT mean the calibration is: the arm never goes to the marker,
        it circles a standoff sphere of camera viewpoints around it.

        MEASURED on-box 2026-07-24 -- a UR3e cell whose marker audited fine collected 0 of 24 EIH samples
        (every pose workspace_rejected or ik_failed). This is that failure in miniature: on the UR5e-authored
        scene the marker sits at r~496 mm, just INSIDE a UR3e's 500 mm sphere, while the viewpoints around it
        are ~240 mm outside it. Auditing only the marker waves the run through; auditing the sphere stops it.
        """
        findings = {f.label: f for f in audit_reach(_robot("ur3e"))}
        marker, viewpoints = findings["scene_setup.marker"], findings["scene_setup.eih_viewpoints (worst)"]
        self.assertTrue(marker.reachable, "precondition: the marker itself audits as reachable")
        self.assertFalse(viewpoints.reachable)
        self.assertGreater(viewpoints.overshoot_mm, 200.0)

    def test_eih_viewpoint_audit_does_not_false_alarm_on_the_arm_it_was_authored_for(self) -> None:
        """The same sphere on a UR5e is r~743 mm of an 850 mm reach. A check that fired here would be noise,
        and operators would learn to ignore it."""
        vp = next(f for f in audit_reach(_robot("ur5e")) if f.label.startswith("scene_setup.eih_viewpoints"))
        self.assertTrue(vp.reachable)

    def test_report_is_human_readable_and_flags_offenders(self) -> None:
        findings = audit_reach(_robot("ur3e"))
        text = format_reach_report(findings, "ur3e")
        self.assertIn("ur3e", text)
        self.assertIn("OUT", text)
        self.assertIn("beyond reach", text)


if __name__ == "__main__":
    unittest.main()
