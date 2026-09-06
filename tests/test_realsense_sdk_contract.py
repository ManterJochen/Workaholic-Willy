"""The RealSense driver against the REAL librealsense — no camera in the room.

THE GAP THIS CLOSES. ``RealSenseRGBDStreamer`` is exercised in ``test_realsense_streamer.py`` against
an **injected fake ``rs`` module**. That proves our logic calls what we think it calls; it proves
nothing about whether librealsense HAS those names, or what its functions do to a frame. It is the
same position the UR driver was in before URSim, and the same fix applies: drive the real SDK.

librealsense installs and loads with **no device attached** (``requirements/camera-realsense.txt``,
already a declared optional dependency), and ``rs.software_device`` lets synthetic frames be pushed
through the real processing blocks. So the half of the driver that TRANSFORMS DATA can be validated
now, months before a D435 exists.

WHAT THIS COVERS (bucket ②, measured against real librealsense 2.58):

  * every ``rs.*`` symbol the driver calls exists — read out of the driver source, so the list cannot
    drift away from the code;
  * ``rs.align.process`` on a real frameset;
  * all four post-processing filters on a real depth frame, and what each one does to holes;
  * ``np.asanyarray(frame.get_data())`` — the shape/dtype the driver then reasons about;
  * the driver's own ``_to_millimetres`` and ``_match_color_size`` on that real output.

WHAT IT DOES NOT COVER, and cannot without hardware (stays bucket ③):

  * ``pipeline.start()`` / ``wait_for_frames()`` — a software device does NOT resolve through a
    pipeline (asserted below, as a tripwire: if a future librealsense makes it work, that test fails
    and tells us we can cover more);
  * device enumeration, ``first_depth_sensor()``, ``get_depth_scale()`` off a real device;
  * ``set_option`` against real sensor hardware — emitter, laser power, visual presets;
  * and the sensor itself: real noise magnitude, specular and dark dropout, auto-exposure.

MEASURED HERE, and worth knowing before September:

    decimation    48x64 -> 24x32   changes the SIZE (the driver's _match_color_size restores it)
    spatial       zeros 264 -> 264  does NOT fill holes
    temporal      zeros 264 -> 264  does NOT fill holes
    hole_filling  zeros 264 -> 4    DOES

That last row is an interaction, not a curiosity: ``RealSenseVisionPerceptionSource`` has its own
hole-skipping logic when it top-references the grasp depth. Enabling the SDK's hole filler changes
what the application-level logic sees — two layers addressing the same defect, and only one of them
is visible in a grasp record.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import numpy as np

try:  # optional dependency: declared in requirements/camera-realsense.txt, absent on CI
    import pyrealsense2 as rs

    _HAS_RS = True
    _SKIP = ""
except Exception as exc:  # pragma: no cover - exercised only where the wheel is missing
    rs = None  # type: ignore[assignment]
    _HAS_RS = False
    # ASCII only: a skip reason is printed by CI consoles that are not UTF-8, and an em-dash
    # arrives there as a replacement character.
    _SKIP = (
        f"pyrealsense2 not installed ({exc.__class__.__name__}) -- "
        "pip install -r requirements.txt"
    )

_DRIVER = Path(__file__).resolve().parents[1] / "src/camera/setup/image_taking/rgbd.py"

_W, _H, _FPS = 64, 48, 30
#: The hole patch pushed into the synthetic depth frame, so "did this filter fill holes" is measurable.
_HOLE_SLICE = (slice(10, 20), slice(10, 20))


def _driver_rs_symbols() -> list[str]:
    """Every ``rs.<name>`` the driver references, read from its source.

    A hand-maintained list would drift the first time somebody adds a filter. The negative lookbehind
    matters: ``filters.append(...)`` contains the substring ``rs.append`` and is not an rs symbol.
    """
    src = _DRIVER.read_text(encoding="utf-8")
    return sorted(set(re.findall(r"(?<![A-Za-z0-9_])rs\.([a-zA-Z_][a-zA-Z0-9_]*)", src)))


def _software_frameset():
    """A real ``rs`` frameset carrying one synthetic colour + depth pair, holes included."""
    depth = np.full((_H, _W), 400, np.uint16)
    depth[_HOLE_SLICE] = 0
    color = np.zeros((_H, _W, 3), np.uint8)
    color[..., 2] = 255

    dev = rs.software_device()
    depth_sensor = dev.add_sensor("Depth")
    color_sensor = dev.add_sensor("Color")

    def _stream(kind, uid, bpp, fmt):
        s = rs.video_stream()
        s.type, s.index, s.uid = kind, 0, uid
        s.width, s.height, s.fps, s.bpp, s.fmt = _W, _H, _FPS, bpp, fmt
        intr = rs.intrinsics()
        intr.width, intr.height = _W, _H
        intr.ppx, intr.ppy, intr.fx, intr.fy = _W / 2, _H / 2, 60.0, 60.0
        intr.model = rs.distortion.brown_conrady
        intr.coeffs = [0.0] * 5
        s.intrinsics = intr
        return s

    dprof = depth_sensor.add_video_stream(_stream(rs.stream.depth, 0, 2, rs.format.z16))
    cprof = color_sensor.add_video_stream(_stream(rs.stream.color, 1, 3, rs.format.bgr8))
    depth_sensor.add_read_only_option(rs.option.depth_units, 0.001)
    dev.create_matcher(rs.matchers.default)

    syncer = rs.syncer()
    depth_sensor.open(dprof)
    color_sensor.open(cprof)
    depth_sensor.start(syncer)
    color_sensor.start(syncer)

    def _push(sensor, prof, arr, bpp, n):
        f = rs.software_video_frame()
        f.pixels = arr.tobytes()
        f.stride = arr.shape[1] * bpp
        f.bpp = bpp
        f.frame_number = n
        f.timestamp = n * 33.0
        f.domain = rs.timestamp_domain.system_time
        f.profile = prof.as_video_stream_profile()
        sensor.on_video_frame(f)

    for n in range(1, 4):  # a few pairs so the matcher has something to sync
        _push(depth_sensor, dprof, depth, 2, n)
        _push(color_sensor, cprof, color, 3, n)

    return syncer.wait_for_frames(), int((depth == 0).sum())


@unittest.skipUnless(_HAS_RS, _SKIP)
class SymbolContractTests(unittest.TestCase):
    """The names the driver calls exist in the SDK it will be run against."""

    def test_every_rs_symbol_the_driver_calls_exists(self) -> None:
        symbols = _driver_rs_symbols()
        self.assertGreaterEqual(len(symbols), 10, "the symbol scrape found suspiciously little")
        missing = [name for name in symbols if not hasattr(rs, name)]
        self.assertEqual(missing, [], f"librealsense {rs.__version__} lacks: {missing}")

    def test_the_enum_members_the_driver_names_exist(self) -> None:
        # A missing enum member is an AttributeError at open() time, on a real cell, at a bench.
        for enum_name, members in (
            ("option", ("emitter_enabled", "laser_power", "depth_units",
                        "visual_preset", "filter_magnitude", "holes_fill")),
            ("stream", ("color", "depth")),
            ("format", ("bgr8", "z16")),
            ("rs400_visual_preset", ("high_accuracy", "high_density", "medium_density", "default")),
        ):
            enum = getattr(rs, enum_name)
            for member in members:
                with self.subTest(enum=enum_name, member=member):
                    self.assertTrue(hasattr(enum, member), f"rs.{enum_name}.{member} is gone")


@unittest.skipUnless(_HAS_RS, _SKIP)
class RealFrameProcessingTests(unittest.TestCase):
    """The driver's grab() call sequence, on frames the real SDK produced."""

    def setUp(self) -> None:
        self.frameset, self.holes_in = _software_frameset()

    def test_align_returns_a_frameset_with_both_streams(self) -> None:
        aligned = rs.align(rs.stream.color).process(self.frameset)
        self.assertTrue(aligned.get_depth_frame(), "align dropped the depth frame")
        self.assertTrue(aligned.get_color_frame(), "align dropped the colour frame")

    def test_the_depth_array_is_the_shape_and_dtype_the_driver_assumes(self) -> None:
        depth = np.asanyarray(self.frameset.get_depth_frame().get_data())
        self.assertEqual(depth.shape, (_H, _W))
        self.assertEqual(depth.dtype, np.uint16)
        self.assertEqual(int((depth == 0).sum()), self.holes_in, "the hole patch did not survive")

    def test_only_the_hole_filter_fills_holes(self) -> None:
        """Pinned because two layers address holes and only one of them is visible downstream."""
        depth_frame = self.frameset.get_depth_frame()
        for name, filt, fills in (
            ("spatial", rs.spatial_filter(), False),
            ("temporal", rs.temporal_filter(), False),
            ("hole_filling", rs.hole_filling_filter(), True),
        ):
            with self.subTest(filter=name):
                out = np.asanyarray(filt.process(depth_frame).get_data())
                holes_out = int((out == 0).sum())
                if fills:
                    self.assertLess(holes_out, self.holes_in // 2, f"{name} did not fill holes")
                else:
                    self.assertEqual(holes_out, self.holes_in, f"{name} unexpectedly changed holes")

    def test_decimation_changes_the_size_and_the_driver_restores_it(self) -> None:
        """The one filter that breaks RGBDFrame's size invariant — and the code that repairs it."""
        from src.camera.setup.image_taking.rgbd import RealSenseRGBDStreamer

        decimated = np.asanyarray(rs.decimation_filter().process(self.frameset.get_depth_frame()).get_data())
        self.assertNotEqual(decimated.shape, (_H, _W), "decimation no longer changes the size")

        color = np.zeros((_H, _W, 3), np.uint8)
        restored = RealSenseRGBDStreamer._match_color_size(decimated, color)
        self.assertEqual(restored.shape[:2], color.shape[:2])
        # Nearest, not interpolated: every restored value must be one the decimated map actually held.
        self.assertTrue(set(np.unique(restored)).issubset(set(np.unique(decimated))))

    def test_to_millimetres_uses_the_device_reported_scale(self) -> None:
        from src.camera.setup.image_taking.rgbd import RealSenseRGBDStreamer

        streamer = RealSenseRGBDStreamer.__new__(RealSenseRGBDStreamer)
        streamer._depth_scale_m = 0.001  # what the software device reports, in metres per unit
        raw = np.asanyarray(self.frameset.get_depth_frame().get_data())
        mm = streamer._to_millimetres(raw)
        self.assertEqual(mm.dtype, np.uint16)
        self.assertEqual(int(mm.max()), 400, "0.001 m/unit on a 400-unit frame must be 400 mm")


@unittest.skipUnless(_HAS_RS, _SKIP)
class WhatStillNeedsHardwareTests(unittest.TestCase):
    """A tripwire, not a limitation to be tolerated.

    A software device does not resolve through ``rs.pipeline`` on 2.58, which is why ``open()`` and
    ``wait_for_frames()`` stay bucket ③. If a future librealsense changes that, this test FAILS — and
    that failure is the notice that more of the driver became coverable without a camera.
    """

    def test_a_software_device_still_does_not_drive_a_pipeline(self) -> None:
        ctx = rs.context()
        rs.software_device().add_to(ctx)
        config = rs.config()
        config.enable_stream(rs.stream.color, _W, _H, rs.format.bgr8, _FPS)
        config.enable_stream(rs.stream.depth, _W, _H, rs.format.z16, _FPS)
        self.assertFalse(
            config.can_resolve(rs.pipeline(ctx)),
            "a software device now resolves through a pipeline -- open()/grab() can be covered "
            "without hardware; extend this suite and delete this tripwire",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
