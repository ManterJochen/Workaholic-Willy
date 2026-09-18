"""A wrist camera declares its body on its rig, and the schema refuses a body that cannot be placed.

The owner decided the key (``camera.cameras.rigs[<id>].body``), its own ``margin_mm`` with no default, the bracket on
the rig, required and nullable, and ``record_tolerance_mm`` and ``record_tolerance_deg`` on the rig's extrinsics with
no default. A body is placed by the eye in hand calibration, which solves on an RGB-D rig's colour image, so a body on
a fixed camera, on a stereo rig, or on a rig whose id cannot name a link is refused at load.
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.config.schema.camera import RGBDDeviceRigConfig
from tests.test_camera_boundaries import _single_rig, _webcam_rig

_BODY = {"model": "realsense_d435", "margin_mm": 5.0, "bracket": None}
_WRIST = {"mounting_mode": "eye_in_hand", "artifact_path": "eih_wrist.json",
          "shutter_motion_tolerance_mm": 2.0, "shutter_motion_tolerance_deg": 0.5,
          "record_tolerance_mm": 1.0, "record_tolerance_deg": 0.2}


def _rgbd(*, body: object = _BODY, extrinsics: object = _WRIST, rig_id: str = "wrist") -> RGBDDeviceRigConfig:
    return RGBDDeviceRigConfig.model_validate(
        {"rig_id": rig_id, "enabled": True, "source": "rgbd", "fps": 30, "extrinsics": extrinsics, "body": body})


class ARigDeclaresItsBodyTests(unittest.TestCase):
    def test_a_wrist_rig_with_a_body_loads(self) -> None:
        rig = _rgbd()
        assert rig.body is not None
        self.assertEqual((rig.body.model, rig.body.margin_mm, rig.body.bracket), ("realsense_d435", 5.0, None))

    def test_a_rig_that_declares_no_body_is_not_carried(self) -> None:
        self.assertIsNone(_rgbd(body=None).body)

    def test_a_bracket_is_a_box_in_the_optical_frame(self) -> None:
        rig = _rgbd(body={**_BODY, "bracket": {"size_mm": [60.0, 10.0, 40.0], "centre_mm": [30.0, 18.0, -30.0]}})
        assert rig.body is not None and rig.body.bracket is not None
        self.assertEqual(rig.body.bracket.size_mm, (60.0, 10.0, 40.0))

    def test_a_body_before_its_calibration_loads(self) -> None:
        # Nothing places it yet, and that is refused where a cell is built, not in the file.
        self.assertIsNone(_rgbd(extrinsics=None).extrinsics)


class TheSchemaRefusesTests(unittest.TestCase):
    def test_the_margin_has_no_default_and_is_above_zero(self) -> None:
        body = {key: value for key, value in _BODY.items() if key != "margin_mm"}
        with self.assertRaises(ValidationError) as caught:
            _rgbd(body=body)
        self.assertIn("margin_mm", str(caught.exception))
        for margin in (0.0, -1.0, 100.5):
            with self.subTest(margin=margin), self.assertRaises(ValidationError):
                _rgbd(body={**_BODY, "margin_mm": margin})

    def test_the_bracket_is_required_and_written_out_as_null(self) -> None:
        body = {key: value for key, value in _BODY.items() if key != "bracket"}
        with self.assertRaises(ValidationError) as caught:
            _rgbd(body=body)
        self.assertIn("bracket", str(caught.exception))

    def test_a_model_that_is_not_a_registry_name_is_refused(self) -> None:
        with self.assertRaises(ValidationError):
            _rgbd(body={**_BODY, "model": "RealSense D435"})

    def test_a_body_on_a_fixed_camera_is_refused(self) -> None:
        fixed = {"mounting_mode": "eye_to_hand", "artifact_path": "eth.json"}
        with self.assertRaises(ValidationError) as caught:
            _rgbd(extrinsics=fixed)
        self.assertIn("camera.cameras.rigs['wrist'].body declares a camera the arm carries, and the rig's extrinsics "
                      "say eye_to_hand", str(caught.exception))

    def test_a_body_on_a_stereo_rig_is_refused(self) -> None:
        for rig in (_webcam_rig("pair"), _single_rig("stereo")):
            with self.subTest(source=rig.source), self.assertRaises(ValidationError) as caught:
                type(rig).model_validate({**rig.model_dump(), "body": _BODY})
            self.assertIn(f"is on a {rig.source!r} rig; a body is placed by the eye in hand calibration",
                          str(caught.exception))

    def test_a_rig_id_that_cannot_name_a_link_is_refused(self) -> None:
        for rig_id in ("wrist cam", "wrist-cam", "wrist.cam"):
            with self.subTest(rig_id=rig_id), self.assertRaises(ValidationError) as caught:
                _rgbd(rig_id=rig_id)
            self.assertIn("names the body's link in the planner and its part in the exact guard", str(caught.exception))

    def test_a_placed_body_needs_both_record_tolerances(self) -> None:
        for missing in ("record_tolerance_mm", "record_tolerance_deg"):
            with self.subTest(missing=missing), self.assertRaises(ValidationError) as caught:
                _rgbd(extrinsics={key: value for key, value in _WRIST.items() if key != missing})
            self.assertIn("need record_tolerance_mm and record_tolerance_deg, each above 0, with no default",
                          str(caught.exception))
        for bad in (0.0, -0.5):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                _rgbd(extrinsics={**_WRIST, "record_tolerance_mm": bad})

    def test_a_wrist_rig_without_a_body_needs_no_record_tolerance(self) -> None:
        plain = {key: value for key, value in _WRIST.items() if not key.startswith("record_")}
        self.assertIsNone(_rgbd(body=None, extrinsics=plain).body)

    def test_a_fixed_camera_takes_no_record_tolerance(self) -> None:
        fixed = {"mounting_mode": "eye_to_hand", "artifact_path": "eth.json", "record_tolerance_mm": 1.0}
        with self.assertRaises(ValidationError) as caught:
            _rgbd(body=None, extrinsics=fixed)
        self.assertIn("an eye_to_hand rig is not placed on the flange, so it takes neither", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
