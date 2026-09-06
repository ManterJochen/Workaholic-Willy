"""Hardware-free unit tests for :class:`RealSenseRGBDStreamer`.

The real librealsense round-trip needs a physical RealSense device (bucket-3,
validated on-box). These tests inject a fake ``rs`` module so the driver LOGIC --
stream configuration, device-option control, hardware align, the post-processing
filter chain (and its order), the raw-units -> uint16-millimetre conversion,
colour-size matching, intrinsics readout, and the SDK-missing error path -- is
exercised deterministically without pyrealsense2 or a camera.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from src.config.schema.camera import RGBDDeviceRigConfig
from src.camera.setup.image_taking.frames import RGBDFrame
from src.camera.setup.image_taking.rgbd import RealSenseRGBDStreamer


# ---------------------------------------------------------------------------
# Fake pyrealsense2 surface
# ---------------------------------------------------------------------------

class _FakeSensor:
    def __init__(self, depth_scale: float = 0.001) -> None:
        self._scale = depth_scale
        self.options: dict[str, float] = {}
        self.supported = {"emitter_enabled", "laser_power", "depth_units", "visual_preset"}

    def supports(self, opt: str) -> bool:
        return opt in self.supported

    def set_option(self, opt: str, val: float) -> None:
        self.options[opt] = val

    def get_depth_scale(self) -> float:
        return self._scale


class _FakeProfile:
    def __init__(self, sensor: _FakeSensor) -> None:
        self._sensor = sensor

    def get_device(self) -> Any:
        return SimpleNamespace(first_depth_sensor=lambda: self._sensor)

    def get_stream(self, _which: str) -> Any:
        intr = SimpleNamespace(
            fx=600.0, fy=600.0, ppx=320.0, ppy=240.0, width=640, height=480,
            coeffs=[0.10, -0.20, 0.001, 0.002, 0.05],  # Brown-Conrady, as the SDK reports it
        )
        return SimpleNamespace(as_video_stream_profile=lambda: SimpleNamespace(get_intrinsics=lambda: intr))


class _FakeFrame:
    def __init__(self, data: np.ndarray | None) -> None:
        self._data = data

    def __bool__(self) -> bool:
        return self._data is not None

    def get_data(self) -> np.ndarray | None:
        return self._data


class _FakeFrameset:
    def __init__(self, color: np.ndarray | None, depth: np.ndarray | None) -> None:
        self._color = _FakeFrame(color)
        self._depth = _FakeFrame(depth)

    def get_color_frame(self) -> _FakeFrame:
        return self._color

    def get_depth_frame(self) -> _FakeFrame:
        return self._depth


class _FakeAlign:
    def __init__(self, to: str) -> None:
        self.to = to
        self.calls = 0

    def process(self, fs: _FakeFrameset) -> _FakeFrameset:
        self.calls += 1
        return fs


class _FakeFilter:
    order: list[str] = []

    def __init__(self, name: str) -> None:
        self.name = name
        self.options: dict[str, float] = {}
        self.calls = 0

    def set_option(self, opt: str, val: float) -> None:
        self.options[opt] = val

    def process(self, frame: _FakeFrame) -> _FakeFrame:
        self.calls += 1
        _FakeFilter.order.append(self.name)
        return frame


class _FakePipeline:
    def __init__(self, color: np.ndarray | None, depth: np.ndarray | None, sensor: _FakeSensor) -> None:
        self._color = color
        self._depth = depth
        self._sensor = sensor
        self.started = False
        self.stopped = False
        self.waits = 0

    def start(self, _cfg: Any) -> _FakeProfile:
        self.started = True
        return _FakeProfile(self._sensor)

    def wait_for_frames(self) -> _FakeFrameset:
        self.waits += 1
        return _FakeFrameset(self._color, self._depth)

    def stop(self) -> None:
        self.stopped = True


class _FakeConfig:
    def __init__(self) -> None:
        self.device: str | None = None
        self.streams: list[tuple] = []

    def enable_device(self, serial: str) -> None:
        self.device = serial

    def enable_stream(self, *args: Any) -> None:
        self.streams.append(args)


def _make_fake_rs(pipeline: _FakePipeline) -> Any:
    _FakeFilter.order = []
    return SimpleNamespace(
        stream=SimpleNamespace(color="color", depth="depth"),
        format=SimpleNamespace(bgr8="bgr8", z16="z16"),
        option=SimpleNamespace(
            emitter_enabled="emitter_enabled",
            laser_power="laser_power",
            depth_units="depth_units",
            visual_preset="visual_preset",
            filter_magnitude="filter_magnitude",
            holes_fill="holes_fill",
        ),
        config=_FakeConfig,
        pipeline=lambda: pipeline,
        align=lambda to: _FakeAlign(to),
        decimation_filter=lambda: _FakeFilter("decimation"),
        spatial_filter=lambda: _FakeFilter("spatial"),
        temporal_filter=lambda: _FakeFilter("temporal"),
        hole_filling_filter=lambda: _FakeFilter("hole"),
    )


# ---------------------------------------------------------------------------
# Rig helper
# ---------------------------------------------------------------------------

def _rig(**realsense: Any) -> RGBDDeviceRigConfig:
    return RGBDDeviceRigConfig(
        source="rgbd",
        rig_id="rs",
        enabled=True,
        fps=30,
        backend=0,
        device_index=0,
        serial_number="SN-123",
        color_resolution=(640, 480),
        depth_resolution=(640, 480),
        align_depth_to_color=True,
        rgbd_backend="realsense",
        realsense={
            "enable_emitter": True,
            "laser_power_mw": 150.0,
            "post_processing": {
                "decimation": True,
                "spatial": True,
                "temporal": True,
                "hole_filling": True,
            },
            **realsense,
        },
        quality={"warmup_frames": 2},
        calibration_paths={"base_dir": "calibration/rs"},
    )


def _color_depth() -> tuple[np.ndarray, np.ndarray]:
    color = np.full((480, 640, 3), 7, dtype=np.uint8)
    depth = np.full((480, 640), 1000, dtype=np.uint16)  # 1000 raw units
    return color, depth


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_open_configures_streams_device_and_filters() -> None:
    color, depth = _color_depth()
    pipe = _FakePipeline(color, depth, _FakeSensor(depth_scale=0.001))
    rs = _make_fake_rs(pipe)
    streamer = RealSenseRGBDStreamer(_rig(), rs_module=rs)

    streamer.open()

    assert pipe.started is True
    assert pipe.waits == 2  # warmup_frames
    assert streamer.is_opened() is True
    # depth scale read from the device, four filters built (decimation+spatial+temporal+hole).
    assert streamer.depth_scale_m == pytest.approx(0.001)
    assert len(streamer._filters) == 4
    # colour intrinsics K assembled from the stream profile.
    k = streamer.get_intrinsics()
    assert k is not None
    assert np.allclose(k, [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    # the factory distortion is kept now, not dropped (decision D1).
    dist = streamer.get_distortion()
    assert dist is not None
    assert np.allclose(dist, [0.10, -0.20, 0.001, 0.002, 0.05])


def test_open_sets_emitter_and_laser_power() -> None:
    color, depth = _color_depth()
    pipe = _FakePipeline(color, depth, _FakeSensor())
    rs = _make_fake_rs(pipe)
    streamer = RealSenseRGBDStreamer(_rig(), rs_module=rs)

    streamer.open()
    sensor = pipe._sensor

    assert sensor.options["emitter_enabled"] == 1.0
    assert sensor.options["laser_power"] == 150.0


def test_grab_aligns_filters_in_order_and_converts_depth_to_mm() -> None:
    color, depth = _color_depth()
    pipe = _FakePipeline(color, depth, _FakeSensor(depth_scale=0.001))
    rs = _make_fake_rs(pipe)
    streamer = RealSenseRGBDStreamer(_rig(), rs_module=rs)
    streamer.open()

    frame = streamer.grab()

    assert isinstance(frame, RGBDFrame)
    # 1000 raw units * 0.001 m/unit * 1000 = 1000 mm.
    assert frame.depth.dtype == np.uint16
    assert int(frame.depth[0, 0]) == 1000
    assert frame.color.shape == frame.depth.shape[:2] + (3,)
    # align ran, and the filters ran in librealsense's recommended order.
    assert streamer._align.calls == 1
    assert _FakeFilter.order == ["decimation", "spatial", "temporal", "hole"]


def test_grab_nearest_resizes_depth_to_colour_grid() -> None:
    # Depth smaller than colour (e.g. after decimation / un-aligned native depth)
    # must be resized so the RGBDFrame size invariant holds.
    color = np.full((480, 640, 3), 7, dtype=np.uint8)
    depth = np.full((240, 320), 500, dtype=np.uint16)
    pipe = _FakePipeline(color, depth, _FakeSensor(depth_scale=0.001))
    rs = _make_fake_rs(pipe)
    streamer = RealSenseRGBDStreamer(_rig(), rs_module=rs)
    streamer.open()

    frame = streamer.grab()

    assert frame.depth.shape == (480, 640)
    assert int(frame.depth[0, 0]) == 500  # nearest keeps the value


def test_depth_units_override_wins_over_device_scale() -> None:
    color, depth = _color_depth()
    pipe = _FakePipeline(color, depth, _FakeSensor(depth_scale=0.001))
    rs = _make_fake_rs(pipe)
    streamer = RealSenseRGBDStreamer(_rig(depth_units_m=0.0005), rs_module=rs)

    streamer.open()

    assert streamer.depth_scale_m == pytest.approx(0.0005)
    assert pipe._sensor.options["depth_units"] == pytest.approx(0.0005)


def test_grab_raises_on_incomplete_frameset() -> None:
    color = np.full((480, 640, 3), 7, dtype=np.uint8)
    pipe = _FakePipeline(color, None, _FakeSensor())  # no depth frame
    rs = _make_fake_rs(pipe)
    streamer = RealSenseRGBDStreamer(_rig(), rs_module=rs)
    streamer.open()

    with pytest.raises(RuntimeError, match="incomplete frameset"):
        streamer.grab()


def test_release_stops_pipeline() -> None:
    color, depth = _color_depth()
    pipe = _FakePipeline(color, depth, _FakeSensor())
    rs = _make_fake_rs(pipe)
    streamer = RealSenseRGBDStreamer(_rig(), rs_module=rs)
    streamer.open()

    streamer.release()

    assert pipe.stopped is True
    assert streamer.is_opened() is False


def test_grab_before_open_raises() -> None:
    streamer = RealSenseRGBDStreamer(_rig(), rs_module=_make_fake_rs(_FakePipeline(None, None, _FakeSensor())))
    with pytest.raises(RuntimeError, match="not open"):
        streamer.grab()


def test_import_error_when_sdk_missing() -> None:
    # rs_module=None forces the real deferred import; block it to hit the guard.
    streamer = RealSenseRGBDStreamer(_rig())
    with patch.dict(sys.modules, {"pyrealsense2": None}):
        with pytest.raises(ImportError, match="pyrealsense2 is not installed"):
            streamer._import_rs()
