"""A cell of several arms comes up whole: its world in each arm's base, a fixed camera shared, every arm or none.

Owner, 2026-10-15: one ``Cell`` with the arms inside (``Cell.from_cell``, ``cell.arms["werker"]`` a normal cell of one
arm); a camera mounted on an arm belongs to that arm, an eye-to-hand camera to every arm; bring-up all arms or none.
What a layer the arms share states of the world stands in the cell's frame, the first arm's base, and every other arm
reads it in its own base, measured where touching placed it; what an arm's own layer states stands in its own base.
"""

from __future__ import annotations

import math
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.calibration.extrinsics import Extrinsics
from src.calibration.serialization import load_extrinsics, save_extrinsics
from src.camera.orchestration.camera import Camera, CameraBusy, CameraNotOpen
from src.config import ConfigError, default_data_dir
from src.config.cell import load_cell
from src.config.schema.camera import RGBDDeviceRigConfig
from src.geometry import Frame
from src.geometry.transform import Transform
from src.robot.execution.arm_registration import ArmRegistration, registration_file
from src.robot.execution.cell import Cell
from src.robot.execution.cell_lock import peek
from src.robot.execution.cell_world import arm_tree, cameras_folder
from tests._arm_kit import WEIGHED_PAYLOAD, kit_tree, planned_arm
from tests._fake_transport import chain_of, fake_transport

#: The world both arms share, in the cell's frame: a jig the guard keeps out of, the planner's bench, and the bin the
#: parts stand in.
_ZELLE = """robot:
  safety:
    self_collision:
      fixtures:
        - {name: jig, center_mm: [600.0, 100.0, 50.0], half_extents_mm: [50.0, 20.0, 10.0]}
    planning_world:
      support_plane: {height_mm: -5.0, center_mm: [600.0, 0.0], extent_mm: [800.0, 400.0], thickness_mm: 40.0}
  grasping:
    support:
      container:
        floor_height_mm: 20.0
        interior_min_mm: [500.0, -150.0, 20.0]
        interior_max_mm: [700.0, 150.0, 220.0]
"""
#: The camera both arms have, fixed over the cell, calibrated against the first arm.
_ZELLE_CAMERA = """cameras:
  primary_rig_id: top
  rigs:
    - rig_id: top
      enabled: true
      source: rgbd
      fps: 30
      serial_number: "111"
      rgbd_backend: realsense
      align_depth_to_color: true
      calibration_paths:
        base_dir: ../calibration/top
      extrinsics:
        mounting_mode: eye_to_hand
        artifact_path: {artifact}
"""
#: The werker's own controller.
_WERKER = 'robot:\n  ur:\n    ip: "127.0.0.2"\n'
#: Where the werker stands: 1200 mm along the cell's x, facing the first arm.
_FACING = np.array([[-1.0, 0.0, 0.0, 1200.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
#: Where the camera stands in the cell's frame: 900 mm up over the bin, looking down.
_CAMERA_IN_CELL = np.array([[1.0, 0.0, 0.0, 600.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 900.0], [0, 0, 0, 1.0]])


def _turned(yaw_deg: float, *, tilt_deg: float = 0.0, shift: tuple[float, float, float] = (1200.0, 0.0, 0.0)
            ) -> np.ndarray:
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    t, u = math.cos(math.radians(tilt_deg)), math.sin(math.radians(tilt_deg))
    matrix = np.eye(4)
    matrix[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ np.array([[1, 0, 0], [0, t, -u], [0, u, t]])
    matrix[:3, 3] = shift
    return matrix


def _corners(center: Any, half: Any) -> np.ndarray:
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    return np.asarray(center, dtype=np.float64) + signs * np.asarray(half, dtype=np.float64)


class _CellTree:
    """The repository's tree, copied, with a layer both arms share and one the werker has alone."""

    def __init__(self, *, werker_layer: str = _WERKER) -> None:
        self.root = Path(tempfile.mkdtemp())
        shutil.copytree(default_data_dir(), self.root, dirs_exist_ok=True)
        self.artifact = self.root / "calibration" / "real" / "eth_top.json"
        save_extrinsics(self.artifact, Extrinsics(
            transform=Transform.from_matrix(_CAMERA_IN_CELL, from_frame=Frame.CAMERA, to_frame=Frame.BASE),
            rmse_mm=1.0, max_error_mm=2.0, num_samples=20, captured_at=datetime(2026, 10, 15, tzinfo=timezone.utc),
            rig_id="top"))
        (self.root / "robot" / "robot.zelle.yaml").write_text(_ZELLE, encoding="utf-8")
        (self.root / "camera" / "cam.zelle.yaml").write_text(
            _ZELLE_CAMERA.format(artifact=self.artifact.as_posix()), encoding="utf-8")
        (self.root / "robot" / "robot.werker.yaml").write_text(werker_layer, encoding="utf-8")
        (self.root / "cells").mkdir()
        self.path = self.root / "cells" / "clip.yaml"
        self.path.write_text("arms:\n  halter: ursim,zelle\n  werker: ursim,zelle,werker\n", encoding="utf-8")

    def cell(self) -> Any:
        cell = load_cell(self.path)
        assert cell.ok, cell.error
        return cell

    def place(self, in_cell: np.ndarray, *, rmse_mm: float = 0.8) -> None:
        ArmRegistration(arm="werker", frame_arm="halter", base_in_cell=in_cell, rmse_mm=rmse_mm,
                        residuals_mm=(0.5, 1.1, 0.7, 0.9), frame_points_mm=((0.0, 0.0, 0.0),) * 4,
                        arm_points_mm=((0.0, 0.0, 0.0),) * 4, captured_at="2026-10-15T12:00:00+00:00",
                        ).write(registration_file(self.cell(), "werker"))


class TheWorldStandsInEachArmsBaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = _CellTree()
        self.cell = self.tree.cell()

    def test_where_square_every_shared_position_is_exact_in_the_werkers_base(self) -> None:
        self.tree.place(_FACING)
        robot: Any = arm_tree(self.cell, "werker", _FACING).robot
        jig = robot.safety.self_collision.fixtures[0]
        np.testing.assert_allclose(jig.center_mm, (600.0, -100.0, 50.0), atol=1e-9)
        np.testing.assert_allclose(jig.half_extents_mm, (50.0, 20.0, 10.0), atol=1e-9)
        bench = robot.safety.planning_world.support_plane
        np.testing.assert_allclose((*bench.center_mm, bench.height_mm), (600.0, 0.0, -5.0), atol=1e-9)
        np.testing.assert_allclose(bench.extent_mm, (800.0, 400.0), atol=1e-9)
        container = robot.grasping.support.container
        np.testing.assert_allclose(container.interior_min_mm, (500.0, -150.0, 20.0), atol=1e-9)
        np.testing.assert_allclose(container.interior_max_mm, (700.0, 150.0, 220.0), atol=1e-9)
        self.assertEqual(robot.ur.ip, "127.0.0.2", "the werker's own layer stays its own")
        self.assertIs(arm_tree(self.cell, "halter", np.eye(4)), self.cell.arm("halter").tree)

    def test_off_square_a_kept_out_box_only_grows_and_a_kept_in_box_only_shrinks(self) -> None:
        turned = _turned(190.0)
        self.tree.place(turned)
        robot: Any = arm_tree(self.cell, "werker", turned).robot
        to_arm = np.linalg.inv(turned)
        jig = robot.safety.self_collision.fixtures[0]
        seen = _corners((600.0, 100.0, 50.0), (50.0, 20.0, 10.0)) @ to_arm[:3, :3].T + to_arm[:3, 3]
        low, high = np.asarray(jig.center_mm) - jig.half_extents_mm, np.asarray(jig.center_mm) + jig.half_extents_mm
        self.assertTrue(np.all((seen >= low - 1e-9) & (seen <= high + 1e-9)), "the jig grew about its turned self")
        container = robot.grasping.support.container
        inside = _corners((np.asarray(container.interior_min_mm) + container.interior_max_mm) / 2.0,
                          (np.asarray(container.interior_max_mm) - container.interior_min_mm) / 2.0)
        back = inside @ turned[:3, :3].T + turned[:3, 3]
        self.assertTrue(np.all((back >= np.array([500.0, -150.0, 20.0]) - 1e-9)
                               & (back <= np.array([700.0, 150.0, 220.0]) + 1e-9)), "the bin only shrank")

    def test_bases_that_tilt_against_each_other_are_refused(self) -> None:
        with self.assertRaises(ConfigError) as caught:
            arm_tree(self.cell, "werker", _turned(180.0, tilt_deg=1.0))
        self.assertIn("tilted", str(caught.exception))

    def test_a_block_is_the_cells_where_a_shared_layer_writes_it_and_never_half_and_half(self) -> None:
        self.assertEqual(self.cell.where_written("werker", "robot.safety.self_collision.fixtures"), "cell")
        self.assertEqual(self.cell.where_written("werker", "robot.ur.ip"), "own")
        mixed = _CellTree(werker_layer=_WERKER + "  safety:\n    planning_world:\n      support_plane:\n"
                                                "        center_mm: [0.0, 0.0]\n").cell()
        with self.assertRaises(ConfigError) as caught:
            mixed.where_written("werker", "robot.safety.planning_world.support_plane")
        self.assertIn("partly", str(caught.exception))

    def test_a_fixed_camera_both_arms_have_is_the_werkers_through_a_file_of_its_own(self) -> None:
        self.tree.place(_FACING)
        rig: Any = arm_tree(self.cell, "werker", _FACING).app_config.camera.cameras.rigs[0]
        artifact = Path(str(rig.extrinsics.artifact_path))
        self.assertEqual(artifact, cameras_folder(self.cell, "werker") / "eth_top.json")
        seen = load_extrinsics(artifact)
        np.testing.assert_allclose(seen.transform.to_matrix(), np.linalg.inv(_FACING) @ _CAMERA_IN_CELL, atol=1e-6)
        self.assertAlmostEqual(seen.rmse_mm, math.hypot(1.0, 0.8))
        self.assertAlmostEqual(seen.max_error_mm, 2.0 + 1.1)
        written = artifact.stat().st_mtime_ns
        arm_tree(self.cell, "werker", _FACING)
        self.assertEqual(artifact.stat().st_mtime_ns, written, "unchanged, not written again")
        first: Any = self.cell.arm("halter").tree.app_config.camera.cameras.rigs[0]
        self.assertEqual(Path(str(first.extrinsics.artifact_path)), self.tree.artifact, "the first arm's is the solve")

    def test_a_cell_names_the_arms_and_refuses_one_not_touched_in_yet(self) -> None:
        with self.assertRaises(ConfigError) as caught:
            Cell.from_cell(self.tree.path)
        self.assertIn("real_cell.register --cell", str(caught.exception))
        self.assertIn("--arm werker", str(caught.exception))
        self.tree.place(_FACING)
        cell = Cell.from_cell(self.tree.path)
        self.assertEqual(list(cell.arms), ["halter", "werker"])
        self.assertIs(cell.arms["halter"], cell)
        jig = cell.arms["werker"].robot_config.safety.self_collision.fixtures[0]
        np.testing.assert_allclose(jig.center_mm, (600.0, -100.0, 50.0), atol=1e-9)
        self.assertEqual(Cell.from_tree(self.cell.arm("halter").tree).arms, {}, "a cell of one arm names none")


class _Device:
    """A camera's device: what it was asked."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def open(self) -> None:
        self.calls.append("open")

    def release(self) -> None:
        self.calls.append("release")

    def grab(self) -> str:
        self.calls.append("grab")
        return "frame"


def _rig(artifact: str, *, mounting: str = "eye_to_hand") -> RGBDDeviceRigConfig:
    extrinsics: dict[str, Any] = {"mounting_mode": mounting, "artifact_path": artifact}
    if mounting == "eye_in_hand":
        extrinsics |= {"shutter_motion_tolerance_mm": 1.0, "shutter_motion_tolerance_deg": 0.5}
    return RGBDDeviceRigConfig.model_validate({
        "source": "rgbd", "rig_id": "top", "enabled": True, "fps": 30, "backend": 0, "serial_number": "111",
        "rgbd_backend": "realsense", "calibration_paths": {"base_dir": "calibration/top"}, "extrinsics": extrinsics})


class AFixedCameraIsSharedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = _Device()
        self.owner = Camera.from_rig(_rig("halter.json"), streamer=self.device)
        self.owner.open()
        self.addCleanup(self.owner.release)

    def test_another_arm_sees_the_one_device_through_its_own_rig(self) -> None:
        with Camera.sharing([self.owner]):
            view = Camera.from_rig(_rig("werker.json"))
        self.assertIsNot(view, self.owner)
        view.open()
        self.assertEqual(view.grab(), "frame")
        self.assertEqual(self.device.calls, ["open", "grab"], "one device, opened once")
        seen: Any = view.rig
        self.assertEqual(str(seen.extrinsics.artifact_path), "werker.json")
        view.release()
        self.assertTrue(self.owner.is_open, "the view never closes the device")
        self.owner.release()
        with self.assertRaises(CameraNotOpen):
            view.open()

    def test_a_wrist_camera_is_never_shared_and_outside_the_block_a_device_has_one_owner(self) -> None:
        with Camera.sharing([self.owner]), self.assertRaises(CameraBusy) as caught:
            Camera.from_rig(_rig("werker.json", mounting="eye_in_hand"))
        self.assertIn("belongs to the arm it is mounted on", str(caught.exception))
        with self.assertRaises(CameraBusy):
            Camera.from_rig(_rig("werker.json"), streamer=_Device()).open()


def _arm_cell(ip: str, *, joints: Any = (0.0, -1.2, 1.3, -0.4, 1.5, 0.2)) -> Cell:
    config = kit_tree("curobo", ip=ip, safety={"self_collision": {"backend": "fcl"}, "payload": WEIGHED_PAYLOAD},
                      gripper={"model": "robotiq_2f85", "tool_frame": {"source": "willy", "offset_mm": (0, 0, 132.0)}})
    arm = planned_arm(config, transport=fake_transport(joints=list(joints), connected=False, kinematics=chain_of("ur5e")))
    cell = Cell(robot_config=config)
    cell._service = SimpleNamespace(runtime=SimpleNamespace(orchestrator=SimpleNamespace(arm=arm, gripper=None)))
    return cell


class EveryArmComesUpOrNoneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.halter, self.werker = _arm_cell("127.0.0.11"), _arm_cell("127.0.0.12")
        self.halter._in_cell, self.werker._in_cell = np.eye(4), _turned(180.0, shift=(3000.0, 0.0, 0.0))
        self.halter._arms = {"halter": self.halter, "werker": self.werker}

    def free(self, ip: str) -> bool:
        return peek(f"ur@{ip}") is None

    def test_every_lock_every_arm_and_the_reservations_come_up_and_go_down_in_reverse(self) -> None:
        live: Any
        with self.halter.connected() as live:
            self.assertFalse(self.free("127.0.0.11") or self.free("127.0.0.12"))
            self.assertTrue(self.halter.arm._transport.connected and self.werker.arm._transport.connected)
            self.assertEqual(self.halter.arm.partners.arms, ("werker",))
            self.assertEqual(set(live.arms), {"halter", "werker"})
        self.assertFalse(self.halter.arm._transport.connected or self.werker.arm._transport.connected)
        self.assertEqual(self.halter.arm.partners.arms, ())
        self.assertTrue(self.free("127.0.0.11") and self.free("127.0.0.12"))
        self.assertEqual(set(live.teardown or {}), {"halter", "werker"})
        verbs = [call.verb for call in self.halter.arm._transport.tape]
        self.assertLess(verbs.index("connect"), verbs.index("disconnect"))

    def test_an_arm_that_does_not_come_up_takes_every_other_down(self) -> None:
        self.werker.arm._transport.script("connect", ConnectionError("the werker's controller does not answer"))
        with self.assertRaises(Exception) as caught:
            with self.halter.connected():
                self.fail("the block ran with an arm down")
        self.assertIn("does not answer", str(caught.exception))
        self.assertFalse(self.halter.arm._transport.connected, "the halter came down again")
        self.assertTrue(self.free("127.0.0.11") and self.free("127.0.0.12"))

    def test_a_halt_latches_every_arm(self) -> None:
        self.halter.halt("a person stopped the cell")
        for one in (self.halter, self.werker):
            with self.subTest(arm=one.robot_config.ur.ip):
                self.assertIsNotNone(one.arm.halt_state())


if __name__ == "__main__":
    unittest.main()
