"""An eye in hand calibration records the flange to TCP it was solved against (owner decision).

The routine solves CAMERA to TOOL against ``arm.get_tcp_pose()``, which is the flange times a tool frame: the declared
one on a ``willy`` cell, the controller's own setting on a ``polyscope`` cell, where the driver derives it at connect.
The artifact used to store the transform and the rig id only, so nothing could say later which tool frame the camera
was placed against. A wrist camera's body is placed from that record, and a tool frame that moved since refuses the
rig. A caller that hands no record writes the old ``/1`` artifact byte for byte, and a ``/1`` artifact still loads,
with no record.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.calibration.exceptions import ExtrinsicsError
from src.calibration.rig_calibration import RigCalibration
from src.calibration.serialization import (
    CAM_TO_TOOL_SCHEMA,
    CAM_TO_TOOL_SCHEMA_V2,
    FlangeToTcp,
    load_cam_to_tool,
    load_cam_to_tool_artifact,
    save_cam_to_tool,
)
from src.contracts import UNSET, chosen
from src.geometry import Frame, Transform, transform_to_dict
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.drivers.ur.tool_frame import tool_frame_matrix
from tests.test_ur_tool_frame import _2F85_OFFSET, _2F85_QUAT, _cfg, _conn

_CAMERA_TO_TOOL = Transform(translation_mm=np.array([-32.5, 61.0, 20.0]),
                            quaternion_xyzw=np.array([0.0, 0.0, 0.7071067811865476, 0.7071067811865476]),
                            from_frame=Frame.CAMERA, to_frame=Frame.TOOL)


class _Extrinsics:
    mounting_mode = "eye_in_hand"
    shutter_motion_tolerance_mm = 2.0
    shutter_motion_tolerance_deg = 0.5
    record_tolerance_mm = 1.0
    record_tolerance_deg = 0.2

    def __init__(self, path: Path) -> None:
        self.artifact_path = str(path)


class TheArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.declared = tool_frame_matrix(_2F85_OFFSET, _2F85_QUAT)

    def test_a_record_writes_version_2_and_comes_back(self) -> None:
        record = FlangeToTcp.from_matrix("willy", self.declared)
        path = save_cam_to_tool(self.root / "eih_wrist.json", _CAMERA_TO_TOOL, rig_id="wrist", flange_to_tcp=record)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["schema"], CAM_TO_TOOL_SCHEMA_V2)
        transform, held = load_cam_to_tool_artifact(path)
        assert chosen(held)
        self.assertEqual(held, record)
        np.testing.assert_array_equal(held.matrix(), self.declared)
        np.testing.assert_array_equal(transform.translation_mm, _CAMERA_TO_TOOL.translation_mm)
        np.testing.assert_array_equal(load_cam_to_tool(path).translation_mm, _CAMERA_TO_TOOL.translation_mm)

    def test_no_record_writes_version_1_byte_for_byte(self) -> None:
        path = save_cam_to_tool(self.root / "eih_wrist.json", _CAMERA_TO_TOOL, rig_id="wrist")
        before = json.dumps({"schema": CAM_TO_TOOL_SCHEMA, "transform": transform_to_dict(_CAMERA_TO_TOOL),
                             "rig_id": "wrist"}, indent=2, sort_keys=True)
        self.assertEqual(path.read_text(encoding="utf-8"), before)

    def test_a_version_1_artifact_loads_with_no_record(self) -> None:
        path = save_cam_to_tool(self.root / "eih_wrist.json", _CAMERA_TO_TOOL, rig_id="wrist")
        _, held = load_cam_to_tool_artifact(path)
        self.assertIs(held, UNSET)

    def test_a_version_2_artifact_without_its_record_is_refused(self) -> None:
        path = self.root / "eih_wrist.json"
        path.write_text(json.dumps({"schema": CAM_TO_TOOL_SCHEMA_V2, "transform": transform_to_dict(_CAMERA_TO_TOOL),
                                    "rig_id": "wrist"}), encoding="utf-8")
        with self.assertRaises(ExtrinsicsError) as caught:
            load_cam_to_tool_artifact(path)
        self.assertIn("holds no flange_to_tcp", str(caught.exception))

    def test_a_record_that_is_not_a_rigid_transform_is_refused(self) -> None:
        sheared = self.declared.copy()
        sheared[0, 1] += 0.1
        bad_row = self.declared.copy()
        bad_row[3, 0] = 1.0
        for matrix in (sheared, bad_row, np.full((4, 4), np.nan)):
            with self.subTest(matrix=matrix.tolist()), self.assertRaises(ExtrinsicsError):
                FlangeToTcp.from_matrix("willy", matrix)
        with self.assertRaises(ExtrinsicsError):
            FlangeToTcp.from_matrix("undeclared", self.declared)
        with self.assertRaises(ExtrinsicsError):
            FlangeToTcp.from_matrix("willy", np.eye(3))

    def test_the_rig_calibration_carries_the_record_and_its_tolerances(self) -> None:
        record = FlangeToTcp.from_matrix("polyscope", self.declared)
        path = save_cam_to_tool(self.root / "eih_wrist.json", _CAMERA_TO_TOOL, rig_id="wrist", flange_to_tcp=record)
        calibration = RigCalibration.from_config("wrist", _Extrinsics(path))
        self.assertEqual(calibration.flange_to_tcp, record)
        self.assertEqual((calibration.record_tolerance_mm, calibration.record_tolerance_deg), (1.0, 0.2))
        self.assertIn("flange to TCP recorded on polyscope", calibration.render())
        self.assertEqual(calibration.to_dict()["flange_to_tcp"], record.to_dict())

    def test_a_rig_calibration_from_version_1_says_it_has_no_record(self) -> None:
        path = save_cam_to_tool(self.root / "eih_wrist.json", _CAMERA_TO_TOOL, rig_id="wrist")
        calibration = RigCalibration.from_config("wrist", _Extrinsics(path))
        self.assertIs(calibration.flange_to_tcp, UNSET)
        self.assertIn("flange to TCP not recorded", calibration.render())
        self.assertIsNone(calibration.to_dict()["flange_to_tcp"])


class TheUrArmSaysWhichFrameItAppliesTests(unittest.TestCase):
    def test_a_willy_cell_applies_the_declared_frame_connected_or_not(self) -> None:
        arm = URRobotArm(_cfg("willy"))
        declared = tool_frame_matrix(_2F85_OFFSET, _2F85_QUAT)
        frame = arm.active_tool_frame
        assert frame is not None
        np.testing.assert_array_equal(frame, declared)

    def test_a_polyscope_cell_knows_its_frame_only_after_connect(self) -> None:
        arm = URRobotArm(_cfg("polyscope"))
        self.assertIsNone(arm.active_tool_frame)
        # The controller holds a frame 0.4 mm off the declared one, inside the connect tolerance.
        held = tool_frame_matrix((0.4, 132.0, 0.0), _2F85_QUAT)
        arm._conn = _conn(held)
        arm.connect()
        frame = arm.active_tool_frame
        assert frame is not None
        np.testing.assert_allclose(frame, held, atol=1e-6)
        self.assertGreater(float(np.abs(frame - tool_frame_matrix(_2F85_OFFSET, _2F85_QUAT)).max()), 0.3)
        arm.disconnect()
        self.assertIsNone(arm.active_tool_frame)

    def test_a_refused_connect_keeps_no_frame(self) -> None:
        arm = URRobotArm(_cfg("polyscope"))
        arm._conn = _conn(None)
        with self.assertRaises(Exception):
            arm.connect()
        self.assertIsNone(arm.active_tool_frame)

    def test_the_calibrate_cli_records_what_the_arm_applies(self) -> None:
        from src.robot.execution.hand_eye import _flange_to_tcp_record

        willy = URRobotArm(_cfg("willy"))
        record = _flange_to_tcp_record(willy.config, willy)
        assert chosen(record)
        self.assertEqual(record.source, "willy")
        np.testing.assert_array_equal(record.matrix(), tool_frame_matrix(_2F85_OFFSET, _2F85_QUAT))

        polyscope = URRobotArm(_cfg("polyscope"))
        self.assertIs(_flange_to_tcp_record(polyscope.config, polyscope), UNSET)
        polyscope._conn = _conn(tool_frame_matrix(_2F85_OFFSET, _2F85_QUAT))
        polyscope.connect()
        record = _flange_to_tcp_record(polyscope.config, polyscope)
        assert chosen(record)
        self.assertEqual(record.source, "polyscope")

        class _NoFrame:
            pass

        self.assertIs(_flange_to_tcp_record(willy.config, _NoFrame()), UNSET)


if __name__ == "__main__":
    unittest.main()
