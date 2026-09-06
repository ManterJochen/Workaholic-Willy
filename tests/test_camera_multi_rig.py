"""Two identical RGB-D cameras must be bound by SERIAL, not by enumeration order.

MEASURED 2026-08-09: the shipped `cam.yaml` carries exactly ONE RGB-D rig, disabled, with
`serial_number: null` and `device_index: 0`. The September cell is two RealSense D435s -- so the rig it
will actually run could not be expressed by the config at all.

Why the serial and not the index: `RealSenseRGBDStreamer` binds a device with
`rs_config.enable_device(self.serial)` (rgbd.py:234-235) and, with no serial, takes whatever the SDK
offers first. That enumeration order is not stable across boots or replugs, and two cameras that swap
identity do NOT fail -- they produce a complete, plausible scene with the views exchanged, every fused
position mirrored about the rig axis, and the pick goes confidently to the wrong place.
"""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

import yaml

from src.config.schema.camera.cam_schema import CameraSystemConfig

_TILTCAM = Path(__file__).resolve().parents[1] / "config/camera/cam.tiltcam.yaml"


def _rig(rig_id: str, *, enabled: bool = True, serial: str | None = None) -> dict:
    d: dict = {"rig_id": rig_id, "enabled": enabled, "source": "rgbd", "fps": 30}
    if serial is not None:
        d["serial_number"] = serial
    return d


def _build(*rigs: dict) -> CameraSystemConfig:
    return CameraSystemConfig(active_mode="auto", rigs=list(rigs))


class MultiRgbdSerialGuardTests(unittest.TestCase):
    def test_one_rgbd_rig_needs_no_serial(self) -> None:
        """A single camera is unambiguous -- the guard must not make a one-camera cell harder."""
        self.assertIsNotNone(_build(_rig("only")))

    def test_two_enabled_rgbd_rigs_without_serials_are_refused(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _build(_rig("oblique_L"), _rig("oblique_R"))
        msg = str(ctx.exception)
        self.assertIn("serial_number", msg)
        self.assertIn("oblique_L", msg)
        self.assertIn("rs-enumerate-devices", msg + _TILTCAM.read_text(encoding="utf-8"))

    def test_two_enabled_rgbd_rigs_with_distinct_serials_load(self) -> None:
        self.assertIsNotNone(
            _build(_rig("oblique_L", serial="111111111111"), _rig("oblique_R", serial="222222222222"))
        )

    def test_the_same_serial_on_two_rigs_is_refused(self) -> None:
        """Both rigs would bind the same physical camera and the second view would be a duplicate."""
        with self.assertRaises(ValueError) as ctx:
            _build(_rig("a", serial="111111111111"), _rig("b", serial="111111111111"))
        self.assertIn("same serial_number", str(ctx.exception))

    def test_a_disabled_second_rig_does_not_trigger_the_guard(self) -> None:
        """Only ENABLED rigs are opened, so only they can be confused with each other."""
        self.assertIsNotNone(_build(_rig("a"), _rig("b", enabled=False)))


class TiltcamProfileTests(unittest.TestCase):
    """The September camera half, paired with `robot.tiltcam.yaml` under the same profile name."""

    def setUp(self) -> None:
        self.rigs = yaml.safe_load(_TILTCAM.read_text(encoding="utf-8"))["cameras"]["rigs"]

    def test_it_declares_exactly_the_two_cameras_the_robot_layer_describes(self) -> None:
        self.assertEqual([r["rig_id"] for r in self.rigs], ["oblique_L", "oblique_R"])
        self.assertTrue(all(r["source"] == "rgbd" for r in self.rigs))

    def test_it_ships_disabled_so_the_tree_still_loads(self) -> None:
        """Shipping it ENABLED with null serials would make `--profile ur3e,tiltcam` refuse out of the
        box -- and that command is the runbook's first step. Disabled, the tree loads and the guard
        fires at the moment it matters."""
        self.assertTrue(all(r["enabled"] is False for r in self.rigs))
        self.assertIsNotNone(_build(*copy.deepcopy(self.rigs)))

    def test_enabling_both_without_serials_is_what_refuses(self) -> None:
        enabled = copy.deepcopy(self.rigs)
        for r in enabled:
            r["enabled"] = True
        with self.assertRaises(ValueError):
            _build(*enabled)

    def test_hole_filling_stays_off_on_a_tilted_rig(self) -> None:
        """A tilted view sees object SIDES at a steep angle, which is where a D435 drops depth. Filling
        INVENTS surface, and the grasp-depth top-reference would then sample invented geometry: an
        honest hole is a measurable signal, a filled one is a confident wrong number."""
        for r in self.rigs:
            with self.subTest(rig=r["rig_id"]):
                self.assertFalse(r["realsense"]["post_processing"]["hole_filling"])


if __name__ == "__main__":
    unittest.main()
