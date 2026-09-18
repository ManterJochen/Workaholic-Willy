"""Every door that builds an arm on a cell that reads geometry refuses a wrist camera it cannot model.

Each refusal here is an owner decision. On a cell whose planner is cuRobo or whose guard reads hand geometry: an enabled
eye_in_hand rig without a body is refused, naming the key; a body nothing places is refused, and the calibration CLI
sweeps it only with a stated reason; a calibration without its flange to TCP record, or with one the cell no longer
holds, is refused; a camera the repository's registry does not stand for is refused. One resolution
(``execution.wrist_bodies``) answers for the build, ``Robot``, ``PlannerStart``, the calibration CLI and the desk, so
the five cannot disagree.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from src.calibration.serialization import FlangeToTcp, save_cam_to_tool
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.execution.robot import Robot
from src.robot.execution.wrist_bodies import WristBodies, WristBodyRequired
from tests._wrist_body import (
    WristOwner,
    camera_section,
    camera_to_tool,
    declared_frame,
    wrist_cell,
    wrist_rig,
)

_DATA = Path(__file__).resolve().parents[1] / "config"


class _BodyArm(DummyRobotArm):
    """A dummy arm that takes wrist bodies, as the UR driver does."""

    def __init__(self) -> None:
        super().__init__()
        self.bodies: tuple = ()

    def set_wrist_bodies(self, bodies: object) -> None:
        self.bodies = tuple(bodies)  # type: ignore[arg-type]


class TheResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def _artifact(self, record: object = "declared") -> str:
        kwargs: dict = {}
        if record is not None:
            matrix = declared_frame() if isinstance(record, str) else record
            kwargs["flange_to_tcp"] = FlangeToTcp.from_matrix("willy", matrix)
        return str(save_cam_to_tool(self.root / "eih_wrist.json", camera_to_tool(), rig_id="wrist", **kwargs))

    def _rig(self, artifact: str, **changes: object) -> object:
        rig = wrist_rig(**changes)
        if rig.extrinsics is not None:
            rig.extrinsics.artifact_path = artifact
        return rig

    def test_a_placed_body_resolves(self) -> None:
        wrist = WristBodies.from_config(wrist_cell(), camera_section(self._rig(self._artifact())))
        self.assertEqual([body.link_name for body in wrist.bodies], ["wrist_camera_wrist"])
        self.assertEqual(wrist.reader, "robot.ur.motion_planner is curobo")
        self.assertIn("wrist camera wrist_camera_wrist: realsense_d435", wrist.line())

    def test_an_enabled_wrist_rig_without_a_body_is_refused_naming_the_key(self) -> None:
        with self.assertRaises(WristBodyRequired) as caught:
            WristBodies.from_config(wrist_cell(), camera_section(self._rig(self._artifact(), body=False)))
        self.assertIn("camera.cameras.rigs['wrist'] is an enabled eye_in_hand camera the arm carries and declares no "
                      "body", str(caught.exception))
        self.assertIn("margin_mm and bracket", str(caught.exception))

    def test_a_disabled_wrist_rig_without_a_body_is_not_refused_and_a_disabled_one_with_a_body_is_carried(self) -> None:
        artifact = self._artifact()
        self.assertEqual(WristBodies.from_config(
            wrist_cell(), camera_section(self._rig(artifact, body=False, enabled=False))).bodies, ())
        carried = WristBodies.from_config(wrist_cell(), camera_section(self._rig(artifact, enabled=False)))
        self.assertEqual(len(carried.bodies), 1)

    def test_a_body_nothing_places_is_refused_with_the_calibration_command(self) -> None:
        with self.assertRaises(WristBodyRequired) as caught:
            WristBodies.from_config(wrist_cell(), camera_section(self._rig("", calibrated=False)))
        self.assertIn("nothing places the body on the flange", str(caught.exception))
        self.assertIn('--unmodelled-wrist-body "<why the sweep may run without it>"', str(caught.exception))

    def test_a_calibration_without_its_record_is_refused(self) -> None:
        with self.assertRaises(WristBodyRequired) as caught:
            WristBodies.from_config(wrist_cell(), camera_section(self._rig(self._artifact(record=None))))
        self.assertIn("records none: it was written before the record existed", str(caught.exception))

    def test_a_record_the_cell_no_longer_holds_is_refused(self) -> None:
        stale = declared_frame().copy()
        stale[2, 3] += 2.0
        with self.assertRaises(WristBodyRequired) as caught:
            WristBodies.from_config(wrist_cell(), camera_section(self._rig(self._artifact(record=stale))))
        self.assertIn("so its body and its pick frame are both stale", str(caught.exception))

    def test_a_camera_the_tree_describes_differently_is_refused(self) -> None:
        tree = self.root / "data"
        shutil.copytree(_DATA / "cameras", tree / "cameras")
        path = tree / "cameras" / "realsense_d435.yaml"
        path.write_text(path.read_text(encoding="utf-8").replace("[90.15, 25.15, 25.15]", "[90.0, 25.0, 25.0]"),
                        encoding="utf-8")
        with self.assertRaises(WristBodyRequired) as caught:
            WristBodies.from_config(wrist_cell(), camera_section(self._rig(self._artifact())), data_dir=tree)
        self.assertIn("differently from the repository registry", str(caught.exception))
        with self.assertRaises(WristBodyRequired) as caught:
            WristBodies.from_config(wrist_cell(), camera_section(self._rig(self._artifact(), model="acme_cam")))
        self.assertIn("no camera 'acme_cam'", str(caught.exception))
        with self.assertRaises(WristBodyRequired) as caught:
            WristBodies.from_config(wrist_cell(), camera_section(self._rig(self._artifact(), model="acme_cam")),
                                    data_dir=tree)
        self.assertIn("a rig names the camera acme_cam, and the repository registry", str(caught.exception))

    def test_a_cell_that_reads_no_geometry_resolves_nothing(self) -> None:
        """The control: the same wrist rig without a body on an ik cell whose guard is the capsule proxy."""
        cell = wrist_cell(planner="ik", backend="capsule")
        wrist = WristBodies.from_config(cell, camera_section(self._rig(self._artifact(), body=False)))
        self.assertEqual((wrist.bodies, wrist.reader), ((), None))
        self.assertIn("not read", wrist.line())

    def test_an_arm_that_cannot_hold_bodies_is_refused_only_when_there_are_some(self) -> None:
        wrist = WristBodies.from_config(wrist_cell(), camera_section(self._rig(self._artifact())))
        with self.assertRaises(WristBodyRequired) as caught:
            wrist.hand_to(DummyRobotArm())
        self.assertIn("takes no wrist bodies", str(caught.exception))
        WristBodies(bodies=(), reader="robot.ur.motion_planner is curobo").hand_to(DummyRobotArm())


class TheRobotTests(unittest.TestCase):
    def test_a_robot_handed_a_wrist_camera_without_a_body_is_refused(self) -> None:
        with self.assertRaises(WristBodyRequired):
            Robot.from_parts(arm=_BodyArm(), gripper=None, lock_key=None, cameras=[WristOwner(body=False)],
                             robot_config=wrist_cell())

    def test_a_robot_hands_its_arm_the_body_before_the_world(self) -> None:
        arm = _BodyArm()
        robot = Robot.from_parts(arm=arm, gripper=None, lock_key=None, cameras=[WristOwner()],
                                 robot_config=wrist_cell())
        self.assertEqual([body.link_name for body in arm.bodies], ["wrist_camera_wrist"])
        assert robot.wrist_bodies is not None
        self.assertIn("wrist camera wrist_camera_wrist", robot.wrist_body_line())
        self.assertIsNotNone(robot.camera_world)

    def test_a_robot_handed_no_camera_says_so(self) -> None:
        robot = Robot.from_parts(arm=_BodyArm(), gripper=None, lock_key=None)
        self.assertEqual(robot.wrist_body_line(), "wrist cameras  not handed any camera")


class ThePlannerStartTests(unittest.TestCase):
    def test_a_start_handed_a_wrist_rig_without_a_body_is_refused_before_an_arm_is_built(self) -> None:
        from src.robot.execution.planner_start import PlannerStart

        report = PlannerStart.from_robot_config(
            wrist_cell(), camera=camera_section(wrist_rig(body=False))).run()
        self.assertFalse(report.started)
        self.assertIn("margin_mm and bracket", report.refusal)
        self.assertIn("wrist_bodies", report.to_dict())


class TheDeskRowTests(unittest.TestCase):
    def _row(self, camera: object, **cell: object) -> object:
        from src.robot.execution.real_cell.preflight import run_config_preflight

        report = run_config_preflight(wrist_cell(**cell), camera=camera, curobo_available=True,  # type: ignore[arg-type]
                                      collision_engine="coal")
        return next(check for check in report.checks if check.name == "wrist camera body")

    def test_a_wrist_rig_without_a_body_blocks_a_real_cell(self) -> None:
        from src.robot.execution.real_cell.preflight import CheckStatus

        row = self._row(camera_section(wrist_rig(body=False)))
        self.assertIs(row.status, CheckStatus.BLOCK)  # type: ignore[attr-defined]
        self.assertIn("margin_mm and bracket", row.detail)  # type: ignore[attr-defined]
        self.assertTrue(row.fix)  # type: ignore[attr-defined]

    def test_no_camera_section_blocks_and_the_shipped_section_is_ok(self) -> None:
        from src.config import load_config
        from src.contracts import UNSET
        from src.robot.execution.real_cell.preflight import CheckStatus

        self.assertIs(self._row(UNSET).status, CheckStatus.BLOCK)  # type: ignore[attr-defined]
        shipped = self._row(load_config().camera)
        self.assertIs(shipped.status, CheckStatus.OK)  # type: ignore[attr-defined]
        self.assertIn("the arm carries no camera", shipped.detail)  # type: ignore[attr-defined]


class TheCalibrationSweepTests(unittest.TestCase):
    def _sweep(self, rig: object, reason: "str | None" = None, data_dir: object = None) -> tuple:
        from src.robot.execution.hand_eye import _wrist_body_for_sweep

        return _wrist_body_for_sweep(wrist_cell(), rig, data_dir=data_dir, reason=reason)

    def test_a_wrist_rig_without_a_body_is_not_swept(self) -> None:
        _, refusal = self._sweep(wrist_rig(body=False))
        self.assertIn("the camera's housing would be invisible to the planner and the guard during the sweep", refusal)

    def test_a_body_nothing_places_is_swept_only_with_a_reason(self) -> None:
        _, refusal = self._sweep(wrist_rig(calibrated=False))
        self.assertIn('--unmodelled-wrist-body "<reason>"', refusal)
        self.assertEqual(self._sweep(wrist_rig(calibrated=False), reason="first calibration of a new bracket"),
                         (None, None))
        _, blank = self._sweep(wrist_rig(calibrated=False), reason="  ")
        self.assertIsNotNone(blank)

    def test_a_reason_does_not_excuse_a_camera_the_registry_does_not_stand_for(self) -> None:
        _, refusal = self._sweep(wrist_rig(calibrated=False, model="acme_cam"), reason="new bracket")
        self.assertIn("acme_cam", refusal)

    def test_a_cell_that_reads_no_geometry_sweeps_as_before(self) -> None:
        from src.robot.execution.hand_eye import _wrist_body_for_sweep

        self.assertEqual(_wrist_body_for_sweep(wrist_cell(planner="ik", backend="capsule"), wrist_rig(body=False),
                                               data_dir=None, reason=None), (None, None))


if __name__ == "__main__":
    unittest.main()
