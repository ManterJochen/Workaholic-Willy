"""A stand-in for ``pyrealsense2``: the RealSense driver's open, its colour sensor and its research recording, no camera.

Richer than the doubles of ``test_realsense_streamer.py`` and ``test_a_realsense_says_what_it_opened.py``, which stay as
they are: a colour sensor with the ranges a D415 reports, whose auto exposure writes its default exposure as it goes off
(librealsense's ``uvc_pu_auto_exposure_option``) and which can hold an exposure in powers of two of a second as Windows'
Media Foundation backend does (``mf-uvc.cpp``); both infrared streams; a stream profile per stream with its lens and
where its camera sits; frame metadata; a filter that changes the depth and an alignment that resizes it, so the depth
before them can be told from the depth after. The names it answers are held against the real SDK in
``test_a_rig_recording_for_research_keeps_its_infrared_images.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.config.schema.camera import RGBDDeviceRigConfig


class Option:
    """An option as pybind11 hands one out: a distinct object with the SDK's snake_case ``name``."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"option.{self.name}"


OPTION = SimpleNamespace(**{name: Option(name) for name in (
    "emitter_enabled", "laser_power", "depth_units", "visual_preset", "filter_magnitude", "holes_fill",
    "enable_auto_exposure", "exposure", "gain", "enable_auto_white_balance", "white_balance", "filter_smooth_alpha",
)})

#: What a D415's colour sensor reports for the five options, min, max, step and default (rs-enumerate-devices -o).
D415_COLOUR_RANGES: dict[str, tuple[float, float, float, float]] = {
    "enable_auto_exposure": (0.0, 1.0, 1.0, 1.0),
    "exposure": (1.0, 10000.0, 1.0, 166.0),
    "gain": (0.0, 128.0, 1.0, 64.0),
    "enable_auto_white_balance": (0.0, 1.0, 1.0, 1.0),
    "white_balance": (2800.0, 6500.0, 10.0, 4600.0),
}


@dataclass
class ColourSensor:
    """The colour sensor: what it offers, what it holds, and every option written to it, in order."""

    ranges: dict[str, tuple[float, float, float, float]] = field(default_factory=lambda: dict(D415_COLOUR_RANGES))
    held: dict[str, float] = field(default_factory=lambda: {
        "enable_auto_exposure": 1.0, "exposure": 156.0, "gain": 64.0, "enable_auto_white_balance": 1.0,
        "white_balance": 4600.0})
    written: list[tuple[str, float]] = field(default_factory=list)
    #: Hold an exposure as the nearest power of two of a second, as librealsense's Media Foundation backend does.
    powers_of_two: bool = False
    #: The colour comes off the depth imagers: one sensor for both, as on a D405.
    is_depth: bool = False

    def supports(self, option: Option) -> bool:
        return option.name in self.ranges

    def get_option_range(self, option: Option) -> Any:
        low, high, step, default = self.ranges[option.name]
        return SimpleNamespace(min=low, max=high, step=step, default=default)

    def set_option(self, option: Option, value: float) -> None:
        self.written.append((option.name, value))
        if option.name == "enable_auto_exposure" and value == 0.0 and self.held["enable_auto_exposure"] != 0.0:
            self.held["exposure"] = self._exposure(self.ranges["exposure"][3])
        self.held[option.name] = self._exposure(value) if option.name == "exposure" else value

    def get_option(self, option: Option) -> float:
        return self.held[option.name]

    def get_supported_options(self) -> list[Option]:
        return [getattr(OPTION, name) for name in self.ranges]

    def is_depth_sensor(self) -> bool:
        return self.is_depth

    def _exposure(self, value: float) -> float:
        if not self.powers_of_two:
            return value
        return float(int(2.0 ** round(math.log2(value * 1e-4)) * 10000.0))


@dataclass
class DepthSensor:
    """The stereo module: its options, the ones written, and the depth scale it streams at."""

    scale: float = 0.001
    options: dict[str, float] = field(default_factory=lambda: {
        "emitter_enabled": 1.0, "laser_power": 150.0, "depth_units": 0.001, "visual_preset": 1.0})
    written: list[tuple[str, float]] = field(default_factory=list)

    def supports(self, option: Option) -> bool:
        return option.name in self.options

    def set_option(self, option: Option, value: float) -> None:
        self.written.append((option.name, value))
        self.options[option.name] = value
        if option.name == "depth_units":
            self.scale = value

    def get_option(self, option: Option) -> float:
        return self.options[option.name]

    def get_supported_options(self) -> list[Option]:
        return [getattr(OPTION, name) for name in self.options]

    def get_depth_scale(self) -> float:
        return self.scale

    def is_depth_sensor(self) -> bool:
        return True


class Device:
    """A D415: its two sensors and what it says of itself; ``colour=None`` is a device that names no colour sensor."""

    def __init__(self, depth: DepthSensor | None = None, colour: ColourSensor | None = None, **info: str) -> None:
        self.depth = depth or DepthSensor()
        self.colour = colour
        self.info = info or {"name": "Intel RealSense D415", "serial_number": "SN-415",
                             "firmware_version": "5.16.0.1", "usb_type_descriptor": "3.2"}

    def first_depth_sensor(self) -> DepthSensor:
        return self.depth

    def first_color_sensor(self) -> ColourSensor:
        if self.colour is None:
            raise RuntimeError("Device does not have color sensor")
        return self.colour

    def supports(self, key: str) -> bool:
        return key in self.info

    def get_info(self, key: str) -> str:
        return self.info[key]


class Stream:
    """One stream's profile: its lens, and where its camera sits along x, millimetres, for its extrinsics."""

    def __init__(self, name: str, width: int, height: int, *, x_mm: float) -> None:
        self.name, self.width, self.height, self.x_mm = name, width, height, x_mm

    def as_video_stream_profile(self) -> Stream:
        return self

    def get_intrinsics(self) -> Any:
        return SimpleNamespace(width=self.width, height=self.height, fx=float(self.width), fy=float(self.width),
                               ppx=self.width / 2.0, ppy=self.height / 2.0, model=Option("brown_conrady"),
                               coeffs=[0.01, -0.02, 0.0, 0.0, 0.003])

    def get_extrinsics_to(self, other: Stream) -> Any:
        return SimpleNamespace(rotation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                               translation=[(self.x_mm - other.x_mm) / 1000.0, 0.0, 0.0])


#: Where each stream's camera sits along x, millimetres: the left imager is the depth's origin, the right one 55 mm
#: along the baseline, the colour camera 15 mm the other way.
X_MM = {"color": -15.0, "depth": 0.0, ("infrared", 1): 0.0, ("infrared", 2): 55.0}


class Profile:
    """What a started pipeline answers: its device and the streams it resolved."""

    def __init__(self, device: Device, streams: dict[tuple[str, int], Stream]) -> None:
        self._device = device
        self.streams = streams

    def get_device(self) -> Device:
        return self._device

    def get_stream(self, kind: str, index: int = -1) -> Stream:
        for (stream_kind, stream_index), stream in self.streams.items():
            if stream_kind == kind and index in (-1, stream_index):
                return stream
        raise RuntimeError(f"Requested stream type {kind} index {index} not found")


class Frame:
    """A frame: its pixels, falsy where there are none, and the metadata it carries."""

    def __init__(self, data: np.ndarray | None, metadata: dict[str, float] | None = None) -> None:
        self.data = data
        self.metadata = metadata or {}

    def __bool__(self) -> bool:
        return self.data is not None

    def get_data(self) -> np.ndarray | None:
        return self.data

    def supports_frame_metadata(self, key: str) -> bool:
        return key in self.metadata

    def get_frame_metadata(self, key: str) -> float:
        return self.metadata[key]


class Frameset:
    """A frameset: colour, depth and the infrared images by librealsense's index; counts the infrared reads."""

    infrared_reads = 0

    def __init__(self, color: Frame, depth: Frame, infrared: dict[int, Frame]) -> None:
        self.color, self.depth, self.infrared = color, depth, infrared

    def get_color_frame(self) -> Frame:
        return self.color

    def get_depth_frame(self) -> Frame:
        return self.depth

    def get_infrared_frame(self, index: int = 0) -> Frame:
        Frameset.infrared_reads += 1
        return self.infrared.get(index, Frame(None))

    def as_frameset(self) -> Frameset:
        return self


class Filter:
    """A processing block that adds ``shift`` raw units to the depth it is handed, a new frameset each time."""

    def __init__(self, name: str, shift: int = 0) -> None:
        self.name, self.shift = name, shift

    def process(self, frameset: Frameset) -> Frameset:
        depth = frameset.depth.data
        moved = None if depth is None else (depth.astype(np.int64) + self.shift).astype(np.uint16)
        return Frameset(frameset.color, Frame(moved), frameset.infrared)

    def set_option(self, option: Option, value: float) -> None:
        pass

    def get_supported_options(self) -> list[Option]:
        return [OPTION.filter_smooth_alpha]

    def get_option(self, option: Option) -> float:
        return 0.5


class Align:
    """The alignment to colour: the depth comes back on the colour grid, its first value everywhere."""

    def __init__(self, _to: str) -> None:
        pass

    def process(self, frameset: Frameset) -> Frameset:
        height, width = frameset.color.data.shape[:2]
        value = frameset.depth.data.reshape(-1)[0]
        return Frameset(frameset.color, Frame(np.full((height, width), value, np.uint16)), {})


class Config:
    """``rs.config()``: the device and every stream asked for, in order."""

    def __init__(self) -> None:
        self.device: str | None = None
        self.streams: list[tuple[Any, ...]] = []

    def enable_device(self, serial: str) -> None:
        self.device = serial

    def enable_stream(self, *args: Any) -> None:
        self.streams.append(args)


class Pipeline:
    """A pipeline that resolves what it is asked for, infrared included unless the camera has none.

    Every frameset carries a colour image, a depth of ``raw`` units at the depth mode, and, where asked, both infrared
    images: the left one all 40, the right one all 60. ``drops`` lists infrared indices a frameset comes without.
    """

    def __init__(self, device: Device | None = None, *, infrared: bool = True, raw: int = 1000,
                 drops: tuple[int, ...] = (), metadata: dict[str, float] | None = None) -> None:
        self.device = device or Device(colour=ColourSensor())
        self.infrared = infrared
        self.raw = raw
        self.drops = drops
        self.metadata = metadata if metadata is not None else {
            "actual_exposure": 156.0, "gain_level": 64.0, "white_balance": 4600.0, "auto_exposure": 1.0}
        self.config: Config | None = None
        self.profile: Profile | None = None
        #: The last frameset handed out, to tell a copy of its pixels from the pixels themselves.
        self.last: Frameset | None = None
        self.waits = 0
        self.stopped = False

    def start(self, config: Config) -> Profile:
        self.config = config
        streams: dict[tuple[str, int], Stream] = {}
        for args in config.streams:
            kind, index = (args[0], args[1]) if len(args) == 6 else (args[0], 0)
            if kind == "infrared" and not self.infrared:
                raise RuntimeError("Couldn't resolve requests")
            width, height = args[-4], args[-3]
            streams[(kind, index)] = Stream(f"{kind}{index or ''}", width, height,
                                            x_mm=X_MM[(kind, index)] if kind == "infrared" else X_MM[kind])
        self.profile = Profile(self.device, streams)
        return self.profile

    def wait_for_frames(self) -> Frameset:
        self.waits += 1
        assert self.profile is not None
        colour = self.profile.get_stream("color")
        depth = self.profile.get_stream("depth")
        infrared: dict[int, Frame] = {}
        for index, level in ((1, 40), (2, 60)):
            if ("infrared", index) in self.profile.streams and index not in self.drops:
                infrared[index] = Frame(np.full((depth.height, depth.width), level, np.uint8))
        self.last = Frameset(Frame(np.full((colour.height, colour.width, 3), 7, np.uint8), dict(self.metadata)),
                             Frame(np.full((depth.height, depth.width), self.raw, np.uint16)), infrared)
        return self.last

    def stop(self) -> None:
        self.stopped = True


def _presets() -> Any:
    members = {}
    for value, name in enumerate(("custom", "default", "hand", "high_accuracy", "high_density", "medium_density")):
        member = _Preset(value)
        member.name = name
        members[name] = member
    return SimpleNamespace(__members__=members)


class _Preset(int):
    """An enum member as pybind11 hands it out: an int with the SDK's snake_case ``name``."""

    name = ""


def sdk(pipeline: Pipeline) -> Any:
    """The ``rs`` module the driver is handed (``rs_module``): every name it reaches, over ``pipeline``."""
    return SimpleNamespace(
        stream=SimpleNamespace(color="color", depth="depth", infrared="infrared"),
        format=SimpleNamespace(bgr8="bgr8", z16="z16", y8="y8"),
        option=OPTION,
        frame_metadata_value=SimpleNamespace(actual_exposure="actual_exposure", gain_level="gain_level",
                                             white_balance="white_balance", auto_exposure="auto_exposure"),
        camera_info=SimpleNamespace(name="name", serial_number="serial_number", firmware_version="firmware_version",
                                    usb_type_descriptor="usb_type_descriptor"),
        rs400_visual_preset=_presets(),
        context=lambda: SimpleNamespace(query_devices=lambda: [pipeline.device]),
        config=Config,
        pipeline=lambda: pipeline,
        align=Align,
        decimation_filter=lambda: Filter("decimation"),
        disparity_transform=lambda to_disparity=True: Filter("disparity"),
        spatial_filter=lambda: Filter("spatial", shift=7),
        temporal_filter=lambda: Filter("temporal"),
        hole_filling_filter=lambda: Filter("hole_filling"),
    )


def rig(*, research: bool = False, color: dict[str, Any] | None = None, warmup: int = 2) -> RGBDDeviceRigConfig:
    """A wrist D415 rig at a small mode: colour 64 x 48, depth 32 x 24, 30 fps, spatial and temporal on."""
    return RGBDDeviceRigConfig(
        source="rgbd", rig_id="wrist", enabled=True, fps=30, backend=0, serial_number="SN-415",
        color_resolution=(64, 48), depth_resolution=(32, 24), rgbd_backend="realsense",
        realsense={"record_for_research": research, "color": color or {}},
        quality={"warmup_frames": warmup}, calibration_paths={"base_dir": "calibration/wrist"})
