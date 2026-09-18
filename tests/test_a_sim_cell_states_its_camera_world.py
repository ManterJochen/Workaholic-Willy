"""An Isaac cell on cuRobo states its camera world before the arm is built, as the owner decided.

A cuRobo motion with neither a live world nor a decline is refused, so a boot that stated neither would start Isaac
for a minute and then refuse its first move. The bootstrap refuses it first. These tests stop the boot at the arm
constructor, off the box: what reaches the constructor was accepted, what raises before it was refused.

The layer half: ``sim_camera_world`` is the one tree that gives the reference cell a world, its support plane at the
top of the scene's table, and the plain ``sim`` tree every other runner loads still has none.
"""

from __future__ import annotations

import unittest
from contextlib import ExitStack
from unittest.mock import patch

from src.robot.core.camera_world import CameraWorldDecline, active_decline, without_camera_world
from src.willy_sim.harness.camera_world import CellCameraWorld, SimCameraWorld, hold_camera_world_decline


class _ArmBuilt(RuntimeError):
    """Raised by the patched arm constructor: the boot got past every refusal."""


def _boot(**kwargs):
    import src.willy_sim.harness.bootstrap as boot

    with patch.object(boot, "require_motion_stack", lambda *a, **k: ()), \
            patch.object(boot, "IsaacRobotArm", side_effect=_ArmBuilt) as arm:
        try:
            boot.bootstrap_sim_cell(None, headless=True, **kwargs)
        finally:
            _boot.built = arm.call_count  # type: ignore[attr-defined]


class ACuroboBootStatesItsCameraWorldTests(unittest.TestCase):
    def test_a_curobo_boot_that_states_nothing_is_refused_before_the_arm(self) -> None:
        with self.assertRaises(ValueError) as caught:
            _boot()
        self.assertIn("camera_world=CameraWorldDecline", str(caught.exception))
        self.assertEqual(0, _boot.built, "the arm was built before the refusal")  # type: ignore[attr-defined]

    def test_the_control_a_decline_reaches_the_arm(self) -> None:
        with self.assertRaises(_ArmBuilt):
            _boot(camera_world=CameraWorldDecline("test: a declined boot"))

    def test_an_ik_boot_needs_neither(self) -> None:
        with self.assertRaises(_ArmBuilt):
            _boot(motion_planner="ik")

    def test_something_else_is_a_type_error(self) -> None:
        with self.assertRaises(TypeError):
            _boot(camera_world="overhead")
        self.assertEqual(0, _boot.built)  # type: ignore[attr-defined]


class ASimCameraWorldIsRefusedWhereItCannotBeHonouredTests(unittest.TestCase):
    def test_the_reference_boot_with_its_layer_reaches_the_arm(self) -> None:
        with self.assertRaises(_ArmBuilt):
            _boot(camera_world=SimCameraWorld("overhead"), extra_profiles=["sim_camera_world"])

    def test_without_the_layer_it_names_the_layer(self) -> None:
        with self.assertRaises(ValueError) as caught:
            _boot(camera_world=SimCameraWorld("overhead"))
        self.assertIn("sim_camera_world", str(caught.exception))
        self.assertIn("planning_world.enabled is false", str(caught.exception))
        self.assertEqual(0, _boot.built)  # type: ignore[attr-defined]

    def test_a_camera_the_boot_does_not_author_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            _boot(camera_world=SimCameraWorld("oblique_R"), extra_profiles=["sim_camera_world"])
        self.assertIn("only the overhead camera", str(caught.exception))

    def test_a_boot_without_safety_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            _boot(camera_world=SimCameraWorld("overhead"), extra_profiles=["sim_camera_world"], safety=False)
        self.assertIn("safety=True", str(caught.exception))

    def test_an_ik_boot_is_refused_a_world(self) -> None:
        with self.assertRaises(ValueError) as caught:
            _boot(camera_world=SimCameraWorld("overhead"), extra_profiles=["sim_camera_world"], motion_planner="ik")
        self.assertIn("only cuRobo plans against a world", str(caught.exception))


class TheHeldDeclineTests(unittest.TestCase):
    """What the bootstrap holds for a declined cell: the arm's own block, open until the cell closes it."""

    class _Arm:
        def without_camera_world(self, reason: str):
            return without_camera_world(self, reason)

    def test_the_decline_is_in_scope_until_the_cell_closes_it(self) -> None:
        arm, other = self._Arm(), self._Arm()
        decline = CameraWorldDecline("test: held for the cell")
        held = hold_camera_world_decline(arm, decline, ExitStack())
        try:
            self.assertEqual(decline, active_decline(arm))
            self.assertIsNone(active_decline(other), "the decline reached another arm")
            self.assertEqual("declined(test: held for the cell)", held.render())
        finally:
            held.close()
        self.assertIsNone(active_decline(arm))
        held.close()  # idempotent

    def test_a_wired_cell_renders_its_cameras_and_the_fit_angle(self) -> None:
        class _Wiring:
            cameras = ("overhead",)

        text = CellCameraWorld(decline=None, wiring=_Wiring(), fit_angle_deg=0.4321).render()
        self.assertEqual("wired('overhead') fit_vs_scene_deg=0.43", text)

    def test_hold_refuses_what_is_not_a_decline(self) -> None:
        with self.assertRaises(TypeError):
            hold_camera_world_decline(self._Arm(), SimCameraWorld("overhead"), ExitStack())  # type: ignore[arg-type]


class TheSimCameraWorldLayerTests(unittest.TestCase):
    def test_the_support_plane_is_the_top_of_the_scene_table(self) -> None:
        from src.willy_sim.config import load_sim_config, require_robot

        robot = require_robot(load_sim_config(None, extra_profiles=["sim_camera_world"]))
        world = robot.safety.planning_world
        table = robot.sim.scene_setup.table
        self.assertTrue(world.enabled)
        plane = world.support_plane
        self.assertIsNotNone(plane)
        top = float(table.position_mm[2]) + float(table.size_mm[2]) / 2.0
        self.assertAlmostEqual(top, float(plane.height_mm))
        self.assertEqual([float(v) for v in table.size_mm[:2]], [float(v) for v in plane.extent_mm])
        self.assertEqual([float(v) for v in table.position_mm[:2]], [float(v) for v in plane.center_mm])

    def test_perceived_is_on_the_plane_is_within_its_clearance_and_no_payload_slots_are_reserved(self) -> None:
        from src.config.loader import load_config
        from src.robot.safety.planning.reservation import PlannerReservation

        robot = load_config(profile="sim,sim_camera_world").robot
        assert robot is not None
        world = robot.safety.planning_world
        self.assertTrue(world.perceived.enabled)
        table = robot.sim.scene_setup.table
        top = float(table.position_mm[2]) + float(table.size_mm[2]) / 2.0
        self.assertLessEqual(abs(top - float(world.support_plane.height_mm)), world.perceived.plane_clearance_mm)
        # Every committed ur5e evidence file was measured without attach slots, so a start with slots finds no file.
        self.assertEqual(0, PlannerReservation.from_config(robot_cfg=robot).sphere_slots)

    def test_the_layer_gives_the_overhead_camera_a_world_and_plain_sim_gives_none(self) -> None:
        from src.robot.execution.camera_world_wiring import CameraWorldPlan
        from src.willy_sim.config import load_sim_config, require_robot
        from src.willy_sim.perception.camera_owner import IsaacCameraOwner

        rig = IsaacCameraOwner.rig_for("overhead")
        layered = CameraWorldPlan.from_config(
            require_robot(load_sim_config(None, extra_profiles=["sim_camera_world"])), [rig],
            primary_rig_id="overhead")
        self.assertEqual(("overhead",), layered.rig_ids, layered.reason)
        plain = CameraWorldPlan.from_config(require_robot(load_sim_config(None)), [rig], primary_rig_id="overhead")
        self.assertEqual((), plain.rig_ids, "the plain sim tree every other runner loads now has a world")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
