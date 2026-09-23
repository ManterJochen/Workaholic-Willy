"""A RealSense says what it opened: which camera, which depth mode, how near it can measure, and its depth scale.

The owner's cell, 2026-09-23: an Intel D415 on the UR10's wrist, its cable along the arm. Three things a bench can
get wrong without any message saying so, from the audit of that cell:

  * Min-Z. Intel gives the D415 a minimum depth of about 450 mm at 1280 x 720 and about 310 mm at 848 x 480, and a
    surface nearer than that reads as a hole. 1280 x 720 is the schema's default depth mode, and a wrist camera works
    at 0.3 to 0.6 m. ``open()`` now logs the device, its depth mode and that mode's Min-Z, and warns for a D415 whose
    depth runs at 1280 x 720.
  * The USB link. A D400 on a USB 2 link offers fewer modes than on USB 3, and a request it cannot serve failed with
    librealsense's own "Couldn't resolve requests", which never names the link. A request that does not start now says
    which RealSense cameras the SDK sees and the USB link each is on.
  * The depth scale. A configured ``depth_units_m`` was trusted rather than read back, and the visual preset was
    written after it. Now the preset goes first, the explicit keys over it, and the scale is the device's own reading
    after both; a device that did not take the configured units refuses to open.

These run against an SDK double; the names they rely on are held against the real SDK in
``tests/test_realsense_sdk_contract.py``.
"""

from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.camera.setup.image_taking.rgbd import RealSenseRGBDStreamer, realsense_min_depth_mm
from src.config.schema.camera import RGBDDeviceRigConfig

_LOGGER = "src.camera.setup.image_taking.rgbd"


class _Sensor:
    """A depth sensor double: records the options written, in order, and answers its depth scale."""

    def __init__(self, *, scale: float = 0.001, takes_units: bool = True,
                 supported: tuple[str, ...] = ("emitter_enabled", "laser_power", "depth_units", "visual_preset"),
                 preset_units: float | None = None) -> None:
        self.scale = scale
        self.takes_units = takes_units
        self.supported = set(supported)
        self.preset_units = preset_units
        self.written: list[tuple[str, float]] = []

    def supports(self, option: str) -> bool:
        return option in self.supported

    def set_option(self, option: str, value: float) -> None:
        self.written.append((option, value))
        if option == "depth_units" and self.takes_units:
            self.scale = value
        if option == "visual_preset" and self.preset_units is not None:
            self.scale = self.preset_units   # a preset that carries its own depth table

    def get_depth_scale(self) -> float:
        return self.scale


class _Device:
    def __init__(self, sensor: _Sensor, **info: str) -> None:
        self._sensor = sensor
        self._info = info

    def first_depth_sensor(self) -> _Sensor:
        return self._sensor

    def supports(self, key: str) -> bool:
        return key in self._info

    def get_info(self, key: str) -> str:
        return self._info[key]


class _Frameset:
    def as_frameset(self) -> _Frameset:
        return self

    def get_color_frame(self) -> Any:
        return SimpleNamespace(get_data=lambda: np.zeros((480, 848, 3), np.uint8))

    def get_depth_frame(self) -> Any:
        return SimpleNamespace(get_data=lambda: np.full((480, 848), 500, np.uint16))


class _Pipeline:
    def __init__(self, device: _Device, *, refuses: str | None = None) -> None:
        self._device = device
        self._refuses = refuses
        self.stopped = False

    def start(self, _config: Any) -> Any:
        if self._refuses is not None:
            raise RuntimeError(self._refuses)
        intr = SimpleNamespace(fx=600.0, fy=600.0, ppx=424.0, ppy=240.0, coeffs=[0.0] * 5)
        return SimpleNamespace(
            get_device=lambda: self._device,
            get_stream=lambda _s: SimpleNamespace(
                as_video_stream_profile=lambda: SimpleNamespace(get_intrinsics=lambda: intr)))

    def wait_for_frames(self) -> _Frameset:
        return _Frameset()

    def stop(self) -> None:
        self.stopped = True


class _Preset(int):
    """An enum member as pybind11 hands it out: an int with the SDK's snake_case ``name``."""

    name = ""


def _presets() -> Any:
    members = {}
    for value, name in enumerate(("custom", "default", "hand", "high_accuracy", "high_density", "medium_density")):
        member = _Preset(value)
        member.name = name
        members[name] = member
    return SimpleNamespace(__members__=members)


def _sdk(pipeline: _Pipeline, connected: tuple[_Device, ...] = ()) -> Any:
    presets = _presets()
    return SimpleNamespace(
        stream=SimpleNamespace(color="color", depth="depth"), format=SimpleNamespace(bgr8="bgr8", z16="z16"),
        option=SimpleNamespace(emitter_enabled="emitter_enabled", laser_power="laser_power",
                               depth_units="depth_units", visual_preset="visual_preset",
                               filter_magnitude="filter_magnitude", holes_fill="holes_fill"),
        rs400_visual_preset=presets,
        camera_info=SimpleNamespace(name="name", serial_number="serial_number", firmware_version="firmware_version",
                                    usb_type_descriptor="usb_type_descriptor"),
        context=lambda: SimpleNamespace(query_devices=lambda: list(connected)),
        config=lambda: SimpleNamespace(enable_device=lambda _s: None, enable_stream=lambda *_: None),
        pipeline=lambda: pipeline,
        align=lambda _to: SimpleNamespace(process=lambda fs: fs),
        decimation_filter=lambda: SimpleNamespace(set_option=lambda *_: None, process=lambda fs: fs),
        disparity_transform=lambda _to=True: SimpleNamespace(process=lambda fs: fs),
        spatial_filter=lambda: SimpleNamespace(process=lambda fs: fs),
        temporal_filter=lambda: SimpleNamespace(process=lambda fs: fs),
        hole_filling_filter=lambda: SimpleNamespace(set_option=lambda *_: None, process=lambda fs: fs),
    )


def _d415(sensor: _Sensor | None = None, *, usb: str = "3.2", serial: str = "SN-415") -> _Device:
    return _Device(sensor or _Sensor(), name="Intel RealSense D415", serial_number=serial,
                   firmware_version="5.16.0.1", usb_type_descriptor=usb)


def _rig(*, depth: tuple[int, int] = (848, 480), serial: str | None = "SN-415", **realsense: Any) -> RGBDDeviceRigConfig:
    return RGBDDeviceRigConfig(
        source="rgbd", rig_id="wrist", enabled=True, fps=30, backend=0, serial_number=serial,
        color_resolution=(1280, 720), depth_resolution=depth, rgbd_backend="realsense",
        realsense=realsense, quality={"warmup_frames": 0}, calibration_paths={"base_dir": "calibration/wrist"})


class MinZTests(unittest.TestCase):

    def test_the_datasheet_values_the_audit_used(self) -> None:
        self.assertEqual(realsense_min_depth_mm("D415", 1280), 450.0)
        self.assertEqual(realsense_min_depth_mm("D415", 848), 310.0)
        self.assertEqual(realsense_min_depth_mm("D435", 1280), 280.0)
        self.assertEqual(realsense_min_depth_mm("D435I", 1280), 280.0)

    def test_another_width_scales_with_the_width(self) -> None:
        self.assertAlmostEqual(realsense_min_depth_mm("D415", 640), 225.0)

    def test_a_model_the_table_does_not_hold_has_no_number(self) -> None:
        self.assertIsNone(realsense_min_depth_mm("D999", 1280))
        self.assertIsNone(realsense_min_depth_mm(None, 1280))


class WhatOpenSaysTests(unittest.TestCase):

    def _open(self, rig: RGBDDeviceRigConfig, device: _Device) -> tuple[RealSenseRGBDStreamer, list[logging.LogRecord]]:
        streamer = RealSenseRGBDStreamer(rig, rs_module=_sdk(_Pipeline(device)))
        with self.assertLogs(_LOGGER, level="INFO") as logs:
            streamer.open()
        self.addCleanup(streamer.release)
        return streamer, logs.records

    def test_open_names_the_camera_its_depth_mode_and_its_min_z(self) -> None:
        streamer, records = self._open(_rig(), _d415())
        opened = next(r.getMessage() for r in records if "opened" in r.getMessage())
        for part in ("Intel RealSense D415", "SN-415", "USB 3.2", "depth 848x480", "about 310 mm"):
            with self.subTest(part=part):
                self.assertIn(part, opened)
        self.assertEqual(streamer.device_name, "Intel RealSense D415")
        self.assertEqual(streamer.min_depth_mm, 310.0)

    def test_a_d415_at_1280x720_depth_is_warned_about(self) -> None:
        _, records = self._open(_rig(depth=(1280, 720)), _d415())
        warnings = [r.getMessage() for r in records if r.levelno == logging.WARNING]
        self.assertEqual(len(warnings), 1, warnings)
        for part in ("450 mm", "848, 480", "310 mm"):
            with self.subTest(part=part):
                self.assertIn(part, warnings[0])

    def test_a_d415_at_848x480_and_a_d435_at_1280x720_are_not(self) -> None:
        for rig, device in ((_rig(), _d415()),
                            (_rig(depth=(1280, 720)), _Device(_Sensor(), name="Intel RealSense D435",
                                                              usb_type_descriptor="3.2"))):
            with self.subTest(device=device.get_info("name")):
                _, records = self._open(rig, device)
                self.assertEqual([r.getMessage() for r in records if r.levelno >= logging.WARNING], [])

    def test_a_camera_the_sdk_does_not_name_opens_with_its_min_z_unknown(self) -> None:
        streamer, records = self._open(_rig(), _Device(_Sensor()))
        self.assertIsNone(streamer.min_depth_mm)
        self.assertIn("Min-Z unknown", next(r.getMessage() for r in records if "opened" in r.getMessage()))

    def test_a_camera_on_a_usb_2_link_that_opens_is_warned_about(self) -> None:
        _, records = self._open(_rig(depth=(640, 480)), _d415(usb="2.1"))
        warnings = [r.getMessage() for r in records if r.levelno == logging.WARNING]
        self.assertTrue(any("USB 2.1" in w for w in warnings), warnings)


class ARequestThatDoesNotStartTests(unittest.TestCase):

    def _refusal(self, rig: RGBDDeviceRigConfig, *connected: _Device) -> tuple[RuntimeError, RealSenseRGBDStreamer]:
        pipeline = _Pipeline(_d415(), refuses="Couldn't resolve requests")
        streamer = RealSenseRGBDStreamer(rig, rs_module=_sdk(pipeline, connected))
        with self.assertRaises(RuntimeError) as caught:
            streamer.open()
        return caught.exception, streamer

    def test_a_camera_on_usb_2_is_named_with_its_link(self) -> None:
        error, streamer = self._refusal(_rig(depth=(1280, 720)), _d415(usb="2.1"))
        text = str(error)
        for part in ("Couldn't resolve requests", "colour 1280x720", "depth 1280x720", "30 fps",
                     "Intel RealSense D415", "SN-415", "USB 2.1", "USB 3", "rs-enumerate-devices"):
            with self.subTest(part=part):
                self.assertIn(part, text)
        self.assertIsInstance(error.__cause__, RuntimeError)
        self.assertFalse(streamer.is_opened())

    def test_a_camera_on_usb_3_is_named_as_not_the_link(self) -> None:
        error, _ = self._refusal(_rig(), _d415(usb="3.2"))
        self.assertIn("USB 3.2", str(error))
        self.assertIn("the link is not the reason", str(error))

    def test_no_camera_with_that_serial_says_so(self) -> None:
        error, _ = self._refusal(_rig(serial="SN-OTHER"), _d415())
        self.assertIn("no RealSense with serial 'SN-OTHER'", str(error))
        self.assertIn("SN-415", str(error))   # the ones it does see

    def test_no_camera_at_all_says_so(self) -> None:
        error, _ = self._refusal(_rig(serial=None))
        self.assertIn("sees no RealSense", str(error))


class TheDepthScaleIsReadBackTests(unittest.TestCase):

    def _open(self, sensor: _Sensor, **realsense: Any) -> tuple[RealSenseRGBDStreamer, _Pipeline]:
        pipeline = _Pipeline(_d415(sensor))
        streamer = RealSenseRGBDStreamer(_rig(**realsense), rs_module=_sdk(pipeline))
        streamer.open()
        self.addCleanup(streamer.release)
        return streamer, pipeline

    def test_the_preset_is_written_before_the_keys_that_override_it(self) -> None:
        sensor = _Sensor()
        self._open(sensor, visual_preset="HighAccuracy", laser_power_mw=150.0, depth_units_m=0.0001)
        written = [option for option, _ in sensor.written]
        self.assertEqual(written[0], "visual_preset")
        self.assertLess(written.index("visual_preset"), written.index("depth_units"))
        self.assertLess(written.index("visual_preset"), written.index("laser_power"))

    def test_the_scale_is_the_devices_reading_after_the_units_are_written(self) -> None:
        streamer, _ = self._open(_Sensor(scale=0.001), depth_units_m=0.0001)
        self.assertAlmostEqual(streamer.depth_scale_m, 0.0001)

    def test_with_no_units_configured_the_scale_is_the_devices(self) -> None:
        streamer, _ = self._open(_Sensor(scale=0.00025))
        self.assertAlmostEqual(streamer.depth_scale_m, 0.00025)

    def test_a_device_that_did_not_take_the_units_refuses_to_open(self) -> None:
        pipeline = _Pipeline(_d415(_Sensor(scale=0.001, takes_units=False)))
        streamer = RealSenseRGBDStreamer(_rig(depth_units_m=0.0001), rs_module=_sdk(pipeline))
        with self.assertRaises(RuntimeError) as caught:
            streamer.open()
        text = str(caught.exception)
        self.assertIn("depth_units_m", text)
        self.assertIn("0.0001", text)
        self.assertIn("0.001", text)
        self.assertTrue(pipeline.stopped, "a refused open leaves the pipeline running")
        self.assertFalse(streamer.is_opened())

    def test_a_device_without_the_option_refuses_configured_units(self) -> None:
        pipeline = _Pipeline(_d415(_Sensor(scale=0.001, supported=("emitter_enabled",))))
        streamer = RealSenseRGBDStreamer(_rig(depth_units_m=0.0005), rs_module=_sdk(pipeline))
        with self.assertRaises(RuntimeError):
            streamer.open()
        self.assertTrue(pipeline.stopped)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
