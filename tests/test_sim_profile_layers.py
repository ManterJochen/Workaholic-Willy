"""Composable profile layers — the sim cell, its robot, and its camera rig as INDEPENDENT dimensions.

``WILLY_PROFILE`` takes a chain (``"sim,ur3e,tiltcam"``) applied left-to-right. The alternative was a
whole profile per combination, which would have had to FORK all four ``*.sim.yaml`` overlays per robot
— including the measured detector ``torch_dtype`` reset that small-object recall depends on. These tests
pin the property that made chaining worth building: a measured value lives in exactly one file and
survives every combination, and each layer can be varied ALONE so a measured difference is attributable.
"""

from __future__ import annotations

import unittest

from src.config import reload_config
from src.willy_sim.config import load_sim_config, require_robot, sim_profile_chain
from src.willy_sim.harness.coverage import audit_camera_coverage
from src.willy_sim.harness.reach import audit_reach
from src.willy_sim.scene.cameras import camera_elevation_deg


def _load(model=None, extra=None):
    reload_config()
    return load_sim_config(None, robot_model=model, extra_profiles=extra)


class ProfileChainTests(unittest.TestCase):
    def test_default_model_adds_no_layer(self) -> None:
        """The UR5e cell is what the `sim` layer already describes, so selecting it must not append a
        second layer — otherwise every existing validated run would change chain."""
        self.assertEqual(sim_profile_chain(), "sim")
        self.assertEqual(sim_profile_chain("ur5e"), "sim")
        self.assertEqual(sim_profile_chain("ur3e"), "sim,ur3e")
        self.assertEqual(sim_profile_chain("ur3e", ("tiltcam",)), "sim,ur3e,tiltcam")


class MeasuredValuesSurviveCompositionTests(unittest.TestCase):
    def test_sim_detector_dtype_reset_survives_every_chain(self) -> None:
        """THE reason profiles compose instead of duplicating. `models/object.sim.yaml` resets the
        production fp16 `torch_dtype` to unset; running the sim detector fp16 was measured to drop the
        small overhead cube (M2 8/10 -> 0/3). A per-robot profile would need its own copy of that, and
        the copies would drift. Chained, there is one copy and it holds for every robot."""
        for model, extra in ((None, None), ("ur3e", None), ("ur3e", ("tiltcam",))):
            with self.subTest(model=model, extra=extra):
                optim = _load(model, extra).models.objectdetector.optim
                self.assertIsNone(optim.torch_dtype, "the sim layer's measured dtype reset was lost")

    def test_robot_layer_does_not_disturb_the_ur5e_cell(self) -> None:
        """Adding ur3e/tiltcam files must leave the plain `sim` load exactly as it was."""
        sim = require_robot(_load())
        self.assertEqual(sim.sim.robot_model, "ur5e")
        self.assertIsNone(sim.sim.gripper_mount)
        self.assertEqual(sim.safety.self_collision.kinematics_model, "ur5e")
        self.assertEqual(tuple(sim.sim.scene_setup.object.position_mm), (450.0, 0.0, 25.0))
        self.assertIsNone(sim.sim.cameras["overhead"].mount_aim_mm)  # still the nadir camera


class UR3eLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.robot = require_robot(_load("ur3e"))

    def test_layer_claims_the_robot_everywhere_it_matters(self) -> None:
        """Three places must agree, or the cell drives one robot with another's numbers: the driver's
        model (Lula/USD/cuRobo), the self-collision DH chain, and the payload cap."""
        self.assertEqual(self.robot.sim.robot_model, "ur3e")
        self.assertEqual(self.robot.safety.self_collision.kinematics_model, "ur3e")
        self.assertEqual(self.robot.safety.payload.max_mass_kg, 3.0)  # UR3e datasheet (UR5e: 5 kg)

    def test_bare_ur3e_asset_gets_a_mounted_gripper(self) -> None:
        """Isaac's ur3e.usd bakes in NO gripper (unlike ur5e/ur10e). Without a mount the cell would come
        up as a bare 6-DoF arm and only fail later, at gripper connect."""
        self.assertEqual(self.robot.sim.gripper_mount, "robotiq_2f85")

    def test_every_configured_position_is_within_the_ur3e_working_sphere(self) -> None:
        """The whole point of the layer. On the UR5e-tuned scene the UR3e misses `safe_pose` by 95 mm
        and the workspace corner by 402 mm; re-anchored, nothing is out of reach."""
        out = [f for f in audit_reach(self.robot) if not f.reachable]
        self.assertEqual(out, [], f"unreachable on a UR3e: {[(f.label, f.radius_mm) for f in out]}")

    def test_positions_keep_orientation_freedom_not_just_bare_reachability(self) -> None:
        """At the very edge the arm is near-straight and near-singular, with almost no orientation
        freedom left — which a top-down grasp needs. Bare reachability is not enough."""
        for finding in audit_reach(self.robot):
            if finding.label.startswith("workspace_limits"):
                continue  # a box corner is allowed to sit near the edge; a POSE is not
            self.assertLess(
                finding.radius_mm / finding.max_reach_mm, 0.90,
                f"{finding.label} sits at {100 * finding.radius_mm / finding.max_reach_mm:.0f}% of reach",
            )


class TiltCamLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.robot = require_robot(_load("ur3e", ("tiltcam",)))

    def test_rig_is_actually_tilted_to_the_stated_angle(self) -> None:
        """Pins the angle against the COORDINATES rather than against the comment beside them."""
        for name in ("overhead", "oblique_L", "oblique_R"):
            cam = self.robot.sim.cameras[name]
            elevation = camera_elevation_deg(tuple(cam.position_mm), tuple(cam.mount_aim_mm))
            self.assertAlmostEqual(elevation, 70.0, places=1, msg=f"{name} is not at 70 deg")

    def test_cameras_are_authored_with_the_real_sensor_field_of_view(self) -> None:
        """Isaac's default lens is narrower than a D435's, so the sim was framing a smaller slice of the
        table than the real cell will — a silent sim2real gap until the FOV is stated."""
        for name in ("overhead", "oblique_L", "oblique_R"):
            self.assertAlmostEqual(self.robot.sim.cameras[name].hfov_deg, 69.4, places=1)

    def test_a_tilted_camera_gets_a_close_near_clip(self) -> None:
        """Isaac's 1.0 m default near clip sits BEHIND a camera 0.9 m from its target: the frame would
        come back empty, and the extrinsic fit that depends on valid depth would raise."""
        for name in ("overhead", "oblique_L", "oblique_R"):
            near = self.robot.sim.cameras[name].near_clip_m
            self.assertIsNotNone(near, f"{name} has no near clip")
            self.assertLess(near, 0.9)

    def test_the_rig_frames_the_whole_configured_workspace(self) -> None:
        """The robot layer anchors the scene and the camera layer places the optics INDEPENDENTLY —
        nothing in the config system makes them agree. This is the check that they do."""
        findings = audit_camera_coverage(self.robot)
        self.assertTrue(findings, "tiltcam cameras should be audited (position + aim are set)")
        out = [f for f in findings if f.in_frame is False]
        self.assertEqual(out, [], f"outside the frame: {[(f.camera, f.label) for f in out]}")

    def test_swapping_only_the_camera_layer_leaves_the_robot_alone(self) -> None:
        """What separate layers buy: run the same arm with either rig and the pick-rate difference is
        attributable to the optics, because nothing else moved."""
        plain = require_robot(_load("ur3e"))
        self.assertEqual(plain.sim.robot_model, self.robot.sim.robot_model)
        self.assertEqual(
            tuple(plain.sim.scene_setup.object.position_mm),
            tuple(self.robot.sim.scene_setup.object.position_mm),
        )
        self.assertIsNone(plain.sim.cameras["overhead"].mount_aim_mm)   # control: nadir
        self.assertIsNotNone(self.robot.sim.cameras["overhead"].mount_aim_mm)  # treatment: tilted


if __name__ == "__main__":
    unittest.main()
