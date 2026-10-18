"""Camera-coverage audit — "can the configured camera SEE the configured scene?"

The companion to the reach audit. Reach caught a scene the arm could not touch; this catches a scene the
camera cannot see. It exists because the config now places the optics and anchors the scene in SEPARATE,
independently-composable layers, so nothing forces the two to agree — and a scene out of frame surfaces
as a detector that finds nothing, which reads as a perception-quality problem for a long time before
anyone suspects the geometry.
"""

from __future__ import annotations

import logging
import unittest
import unittest.mock

from src.config import reload_config
from src.willy_sim.config import load_sim_config, require_robot
from src.willy_sim.harness.coverage import (
    audit_camera_coverage,
    format_coverage_report,
    warn_if_out_of_frame,
)


def _robot(model=None, extra=None):
    reload_config()
    return require_robot(load_sim_config(None, robot_model=model, extra_profiles=extra))


def _retarget(robot, cameras):
    return robot.model_copy(update={"sim": robot.sim.model_copy(update={"cameras": cameras})})


class AuditScopeTests(unittest.TestCase):
    def test_only_aimed_cameras_are_audited(self) -> None:
        """An aim point is what marks a camera as deliberately placed AND pointed. The historical nadir
        camera has none and its framing is the one that was validated on-box, so it is left alone."""
        plain = _robot()
        self.assertIsNone(plain.sim.cameras["overhead"].mount_aim_mm)
        self.assertNotIn("overhead", {f.camera for f in audit_camera_coverage(plain)})

    def test_an_unstated_lens_reports_unknown_rather_than_ok(self) -> None:
        """Without hfov_deg the audit cannot know the frame. Reporting that honestly beats a green tick
        that means "not checked"."""
        findings = [f for f in audit_camera_coverage(_robot()) if f.camera == "oblique_L"]
        self.assertTrue(findings)
        self.assertTrue(all(f.in_frame is None for f in findings))
        self.assertIn("lens unstated", format_coverage_report(findings))


class TiltedRigCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.robot = _robot("ur3e", ("tiltcam",))

    def test_the_shipped_rig_frames_the_whole_workspace(self) -> None:
        out = [f for f in audit_camera_coverage(self.robot) if f.in_frame is False]
        self.assertEqual(out, [], f"outside the frame: {[(f.camera, f.label) for f in out]}")

    def test_the_object_sits_near_the_optical_axis(self) -> None:
        """Both cameras are aimed at the workspace centre, so the graspable object should be close to
        the middle of the frame rather than merely inside it (lens distortion is worst at the edge)."""
        for f in audit_camera_coverage(self.robot):
            if f.label.startswith("object["):
                self.assertLess(f.off_axis_deg, 5.0, f"{f.camera}: object {f.off_axis_deg:.1f} deg off-axis")

    def test_workspace_corners_keep_a_real_margin(self) -> None:
        corners = [f for f in audit_camera_coverage(self.robot) if f.label.startswith("workspace corner")]
        self.assertTrue(corners)
        worst = max(corners, key=lambda f: f.off_axis_deg)
        self.assertLess(worst.off_axis_deg, worst.half_fov_deg - 5.0, "no margin left at the frame edge")


class DetectionTests(unittest.TestCase):
    def test_a_scene_outside_the_frame_is_caught_and_logged(self) -> None:
        """The failure this whole module exists for: a camera layer aimed at one workspace, chained onto
        a robot layer that anchors the scene somewhere else."""
        robot = _robot("ur3e", ("tiltcam",))
        # Same rig, aimed a metre off to the side of where the scene actually is.
        strayed = _retarget(robot, {
            name: cam.model_copy(update={"mount_aim_mm": (350.0, 1400.0, 50.0),
                                         "position_mm": (350.0, 1092.2, 895.7)})
            for name, cam in robot.sim.cameras.items() if cam.mount_aim_mm
        })
        with self.assertLogs("src.willy_sim.harness.coverage", level=logging.WARNING) as logs:
            bad = warn_if_out_of_frame(strayed)
        self.assertTrue(bad)
        self.assertIn("OUT", "\n".join(logs.output))

    def test_a_good_rig_logs_nothing(self) -> None:
        logger = logging.getLogger("src.willy_sim.harness.coverage")
        with unittest.mock.patch.object(logger, "warning") as warn:
            self.assertEqual(warn_if_out_of_frame(_robot("ur3e", ("tiltcam",))), [])
        warn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
