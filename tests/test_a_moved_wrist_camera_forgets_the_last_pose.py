"""A camera the arm moved measures only where it is now: the temporal filter's history does not cross a move.

The owner's cell, 2026-09-23: an Intel D415 on the UR10's wrist. The RealSense temporal filter averages each depth
pixel over the frames the driver handed it and fills a hole with a depth it saw in those frames. The driver hands it
only the frames it grabs, and a wrist camera's grabs are taken at different arm poses: the live world takes one grab
after every move. Measured by the audit with librealsense 2.58.3 on a software device, four frames at pose A (600 mm)
and one at pose B (left half a hole, right half 612 mm) came out as ``[600, 600, 600, 600, 604, 604, 604, 604]``: the
holes filled with pose A's depth, placed along pose B's rays, and the measured pixels pulled 60 % toward pose A.

What these tests hold:

  * ``camera_moved()`` on the driver, the owner and a handle drops that history, so the next frame holds only
    depth seen from the new pose (the API the planning world calls after a move);
  * a camera the arm carries also drops it by itself when a pause between two grabs is longer than a burst of
    back-to-back grabs takes, so a program that moved the arm without saying so still gets an honest frame;
  * the chain runs in librealsense's recommended order: decimation, depth to disparity, spatial, temporal,
    disparity to depth, hole filling, and only then the alignment to colour. In the old order (align first, then
    spatial and temporal on millimetre depth) the average at pose B stopped at 610 where the scene is 612.

The real-SDK tests drive ``open()`` and ``grab()`` of the driver itself, with only the pipeline replaced by a double
that pushes synthetic frames through a librealsense software device: the filters and the alignment are the SDK's.
"""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.camera.orchestration.camera import Camera
from src.camera.orchestration.frame_provider import FrameProvider, RigHandle
from src.camera.setup.image_taking.rgbd import RealSenseRGBDStreamer
from src.config.schema.camera import RGBDDeviceRigConfig
from tests.test_camera_boundaries import _rgbd_frame

try:  # pyrealsense2 is pinned in requirements.txt, and the wheel is absent on some hosts
    import pyrealsense2 as rs

    _HAS_RS = True
    _SKIP = ""
except Exception as exc:  # pragma: no cover - exercised only where the wheel is missing
    rs = None  # type: ignore[assignment]
    _HAS_RS = False
    _SKIP = f"pyrealsense2 not installed ({exc.__class__.__name__}) -- pip install -r requirements.txt"

_W, _H = 64, 48
#: Pixels well inside each half, clear of the edge the spatial filter smooths across.
_HOLE = (slice(8, 40), slice(4, 26))
_SEEN = (slice(8, 40), slice(38, 60))


def _rig(*, carried: bool = False, temporal: bool = True, warmup: int = 0, **post: Any) -> RGBDDeviceRigConfig:
    """A RealSense rig at the software device's size; ``carried`` gives it the body of a camera on the wrist."""
    extra: dict[str, Any] = {}
    if carried:
        extra["body"] = {"model": "realsense_d415", "margin_mm": 5.0, "bracket": None}
    return RGBDDeviceRigConfig(
        source="rgbd", rig_id="wrist", enabled=True, fps=30, backend=0, serial_number="SN-415",
        color_resolution=(_W, _H), depth_resolution=(_W, _H), align_depth_to_color=True, rgbd_backend="realsense",
        realsense={"post_processing": {"spatial": True, "temporal": temporal, **post}},
        quality={"warmup_frames": warmup}, calibration_paths={"base_dir": "calibration/wrist"}, **extra)


class _Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


# ---------------------------------------------------------------------------
# The driver on the real SDK, the pipeline replaced by a software device
# ---------------------------------------------------------------------------

class _SoftwareScene:
    """A pipeline double over a librealsense software device.

    Every ``wait_for_frames`` pushes the depth the camera currently looks at (``self.depth``) and a colour image as
    one pair, and returns the frameset the SDK's own syncer builds from them.
    """

    def __init__(self) -> None:
        self.depth = np.full((_H, _W), 600, np.uint16)
        self._dev = rs.software_device()
        self._depth_sensor = self._dev.add_sensor("Depth")
        self._color_sensor = self._dev.add_sensor("Color")
        self._dprof = self._depth_sensor.add_video_stream(self._stream(rs.stream.depth, 0, 2, rs.format.z16))
        self._cprof = self._color_sensor.add_video_stream(self._stream(rs.stream.color, 1, 3, rs.format.bgr8))
        self._depth_sensor.add_read_only_option(rs.option.depth_units, 0.001)
        self._depth_sensor.add_read_only_option(rs.option.stereo_baseline, 55.0)
        identity = rs.extrinsics()
        identity.rotation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        identity.translation = [0.0, 0.0, 0.0]
        self._dprof.register_extrinsics_to(self._cprof, identity)
        self._dev.create_matcher(rs.matchers.default)
        self._syncer = rs.syncer()
        self._depth_sensor.open(self._dprof)
        self._color_sensor.open(self._cprof)
        self._depth_sensor.start(self._syncer)
        self._color_sensor.start(self._syncer)
        self._frame_number = 0
        self._buffers: list[np.ndarray] = []   # the SDK reads pushed pixels without copying them
        self.stopped = False

    @staticmethod
    def _stream(kind: Any, uid: int, bpp: int, fmt: Any) -> Any:
        stream = rs.video_stream()
        stream.type, stream.index, stream.uid = kind, 0, uid
        stream.width, stream.height, stream.fps, stream.bpp, stream.fmt = _W, _H, 30, bpp, fmt
        intr = rs.intrinsics()
        intr.width, intr.height = _W, _H
        intr.ppx, intr.ppy, intr.fx, intr.fy = _W / 2, _H / 2, 60.0, 60.0
        intr.model = rs.distortion.brown_conrady
        intr.coeffs = [0.0] * 5
        stream.intrinsics = intr
        return stream

    def start(self, _config: Any) -> Any:
        sensor = SimpleNamespace(supports=lambda _option: False, set_option=lambda *_: None,
                                 get_depth_scale=lambda: 0.001)
        device = SimpleNamespace(first_depth_sensor=lambda: sensor, supports=lambda _info: False,
                                 get_info=lambda _info: "")
        return SimpleNamespace(get_device=lambda: device, get_stream=lambda _which: self._cprof)

    def stop(self) -> None:
        self.stopped = True

    def _push(self) -> Any:
        self._frame_number += 1
        color = np.zeros((_H, _W, 3), np.uint8)
        for sensor, profile, pixels, bpp in ((self._depth_sensor, self._dprof, self.depth, 2),
                                             (self._color_sensor, self._cprof, color, 3)):
            data = np.ascontiguousarray(pixels)
            self._buffers.append(data)
            frame = rs.software_video_frame()
            frame.pixels = data.tobytes()
            frame.stride = _W * bpp
            frame.bpp = bpp
            frame.frame_number = self._frame_number
            frame.timestamp = self._frame_number * 33.0
            frame.domain = rs.timestamp_domain.system_time
            frame.profile = profile.as_video_stream_profile()
            sensor.on_video_frame(frame)
        return self._syncer.wait_for_frames()

    def wait_for_frames(self) -> Any:
        for _ in range(4):   # the syncer hands the very first depth frame out alone, before any colour has arrived
            frameset = self._push()
            if frameset.size() == 2:
                return frameset
        raise AssertionError("the software device never paired colour and depth")


class _SdkWithAScene:
    """The real ``pyrealsense2`` module, with ``pipeline()`` answering the software scene."""

    def __init__(self, scene: _SoftwareScene) -> None:
        self._scene = scene

    def pipeline(self) -> _SoftwareScene:
        return self._scene

    def __getattr__(self, name: str) -> Any:
        return getattr(rs, name)


def _assert_reads(region: np.ndarray, mm: int) -> None:
    """Every measured pixel of ``region`` reads ``mm``, and at least nine in ten are measured.

    Checked on the measured pixels because the SDK's alignment of these synthetic frames, whose colour and depth
    lenses are identical, leaves a one-pixel row without depth at some depths: a rasterisation artefact of the
    alignment, seen on the old chain order too, and not a hole in the scene.
    """
    measured = region[region > 0]
    assert measured.size >= 0.9 * region.size, f"only {measured.size} of {region.size} pixels measured"
    assert np.all(measured == mm), sorted(set(measured.tolist()))


def _pose_a() -> np.ndarray:
    return np.full((_H, _W), 600, np.uint16)


def _pose_b() -> np.ndarray:
    """12 mm nearer, and the left half a hole: a surface under the D415's minimum range, or shadowed."""
    depth = np.full((_H, _W), 612, np.uint16)
    depth[:, : _W // 2] = 0
    return depth


@unittest.skipUnless(_HAS_RS, _SKIP)
class ACameraToldItMovedTests(unittest.TestCase):
    """The driver's own open() and grab(), on the SDK's filters and alignment."""

    def _opened(self, rig: RGBDDeviceRigConfig, clock: _Clock | None = None) -> tuple[RealSenseRGBDStreamer, _SoftwareScene]:
        scene = _SoftwareScene()
        streamer = RealSenseRGBDStreamer(rig, rs_module=_SdkWithAScene(scene), clock=clock or _Clock())
        streamer.open()
        self.addCleanup(streamer.release)
        return streamer, scene

    def _four_frames_at_pose_a(self, streamer: RealSenseRGBDStreamer, scene: _SoftwareScene,
                               clock: _Clock | None = None) -> None:
        scene.depth = _pose_a()
        for _ in range(4):
            if clock is not None:
                clock.now += 0.033
            _assert_reads(streamer.grab().depth[_SEEN], 600)

    def test_a_camera_that_is_not_told_fills_its_new_holes_with_the_old_pose(self) -> None:
        """The hazard, on the SDK itself: without the notice, pose B's holes read pose A's 600 mm."""
        streamer, scene = self._opened(_rig())
        self._four_frames_at_pose_a(streamer, scene)
        scene.depth = _pose_b()
        frame = streamer.grab()
        _assert_reads(frame.depth[_HOLE], 600)

    def test_a_camera_told_it_moved_measures_only_the_new_pose(self) -> None:
        streamer, scene = self._opened(_rig())
        self._four_frames_at_pose_a(streamer, scene)
        streamer.camera_moved()
        scene.depth = _pose_b()
        frame = streamer.grab()
        np.testing.assert_array_equal(frame.depth[_HOLE], 0)
        _assert_reads(frame.depth[_SEEN], 612)

    def test_the_chain_filters_in_disparity_before_it_aligns(self) -> None:
        """Frames at pose B with the history kept: the average reaches the scene's 612 mm.

        Measured 2026-09-23 with librealsense 2.58.3, the median of the seen half over twelve frames at pose B. In
        this order: 605, 608, 609, 610, 611, 611, 612, and 612 from then on. In the old order, spatial and temporal
        on millimetre depth after the alignment: 604, 607, 609, 610, and 610 from then on, 2 mm short for good,
        because a step the average would take rounds to no step at all in whole millimetres.
        """
        streamer, scene = self._opened(_rig())
        self._four_frames_at_pose_a(streamer, scene)
        scene.depth = _pose_b()
        for _ in range(8):
            frame = streamer.grab()
        _assert_reads(frame.depth[_SEEN], 612)

    def test_a_carried_camera_drops_its_history_across_a_pause(self) -> None:
        """A program that moved the arm and said nothing: the pause the move took is enough."""
        clock = _Clock()
        streamer, scene = self._opened(_rig(carried=True), clock)
        self._four_frames_at_pose_a(streamer, scene, clock)
        clock.now += 2.0   # a move, and the arm settling
        scene.depth = _pose_b()
        frame = streamer.grab()
        np.testing.assert_array_equal(frame.depth[_HOLE], 0)
        _assert_reads(frame.depth[_SEEN], 612)

    def test_a_carried_camera_keeps_its_history_within_a_burst(self) -> None:
        """Back-to-back grabs are one still view, which is what the filter averages: the history stays."""
        clock = _Clock()
        streamer, scene = self._opened(_rig(carried=True), clock)
        self._four_frames_at_pose_a(streamer, scene, clock)
        clock.now += 0.033
        scene.depth = _pose_b()
        frame = streamer.grab()
        _assert_reads(frame.depth[_HOLE], 600)

    def test_a_fixed_camera_keeps_its_history_across_a_pause(self) -> None:
        """Scope: the pause rule is for a camera the arm carries. A fixed camera's view does not move with the arm."""
        clock = _Clock()
        streamer, scene = self._opened(_rig(), clock)
        self._four_frames_at_pose_a(streamer, scene, clock)
        clock.now += 2.0
        scene.depth = _pose_b()
        frame = streamer.grab()
        _assert_reads(frame.depth[_HOLE], 600)


# ---------------------------------------------------------------------------
# The chain's order and the notice, against an SDK double
# ---------------------------------------------------------------------------

class _Frameset:
    def __init__(self) -> None:
        self._color = SimpleNamespace(get_data=lambda: np.zeros((4, 4, 3), np.uint8))
        self._depth = SimpleNamespace(get_data=lambda: np.full((4, 4), 500, np.uint16))

    def as_frameset(self) -> _Frameset:
        return self

    def get_color_frame(self) -> Any:
        return self._color

    def get_depth_frame(self) -> Any:
        return self._depth


class _Step:
    """A filter or the alignment: records its name in the shared order when it processes a frameset."""

    def __init__(self, name: str, order: list[str]) -> None:
        self.name = name
        self._order = order

    def set_option(self, *_: Any) -> None:
        pass

    def process(self, frameset: Any) -> Any:
        self._order.append(self.name)
        return frameset


def _sdk_double(order: list[str], built: list[str]) -> Any:
    def step(name: str) -> _Step:
        built.append(name)
        return _Step(name, order)

    sensor = SimpleNamespace(supports=lambda _o: False, set_option=lambda *_: None, get_depth_scale=lambda: 0.001)
    device = SimpleNamespace(first_depth_sensor=lambda: sensor, supports=lambda _i: False, get_info=lambda _i: "")
    intr = SimpleNamespace(fx=1.0, fy=1.0, ppx=2.0, ppy=2.0, coeffs=[0.0] * 5)
    profile = SimpleNamespace(
        get_device=lambda: device,
        get_stream=lambda _s: SimpleNamespace(as_video_stream_profile=lambda: SimpleNamespace(get_intrinsics=lambda: intr)))
    pipeline = SimpleNamespace(start=lambda _c: profile, wait_for_frames=_Frameset, stop=lambda: None)
    return SimpleNamespace(
        stream=SimpleNamespace(color="color", depth="depth"), format=SimpleNamespace(bgr8="bgr8", z16="z16"),
        option=SimpleNamespace(emitter_enabled="emitter_enabled", laser_power="laser_power", depth_units="depth_units",
                               visual_preset="visual_preset", filter_magnitude="filter_magnitude",
                               holes_fill="holes_fill"),
        camera_info=SimpleNamespace(name="name", serial_number="serial_number", firmware_version="firmware_version",
                                    usb_type_descriptor="usb_type_descriptor"),
        config=lambda: SimpleNamespace(enable_device=lambda _s: None, enable_stream=lambda *_: None),
        pipeline=lambda: pipeline,
        align=lambda _to: _Step("align", order),
        decimation_filter=lambda: step("decimation"),
        disparity_transform=lambda to_disparity=True: step("to_disparity" if to_disparity else "to_depth"),
        spatial_filter=lambda: step("spatial"),
        temporal_filter=lambda: step("temporal"),
        hole_filling_filter=lambda: step("hole"),
    )


def _rig_4x4(**post: Any) -> RGBDDeviceRigConfig:
    return RGBDDeviceRigConfig(
        source="rgbd", rig_id="rs", enabled=True, fps=30, backend=0, serial_number="SN",
        color_resolution=(4, 4), depth_resolution=(4, 4), rgbd_backend="realsense",
        realsense={"post_processing": post}, quality={"warmup_frames": 0},
        calibration_paths={"base_dir": "calibration/rs"})


class TheChainsOrderTests(unittest.TestCase):

    def _grabbed_order(self, **post: Any) -> list[str]:
        order: list[str] = []
        streamer = RealSenseRGBDStreamer(_rig_4x4(**post), rs_module=_sdk_double(order, []))
        streamer.open()
        streamer.grab()
        return order

    def test_every_filter_runs_in_librealsenses_order_and_before_the_alignment(self) -> None:
        self.assertEqual(
            self._grabbed_order(decimation=True, spatial=True, temporal=True, hole_filling=True),
            ["decimation", "to_disparity", "spatial", "temporal", "to_depth", "hole", "align"])

    def test_spatial_alone_is_still_run_in_disparity(self) -> None:
        self.assertEqual(self._grabbed_order(spatial=True, temporal=False),
                         ["to_disparity", "spatial", "to_depth", "align"])

    def test_no_smoothing_means_no_disparity_round_trip(self) -> None:
        self.assertEqual(self._grabbed_order(spatial=False, temporal=False), ["align"])


class TheNoticeTests(unittest.TestCase):

    def _streamer(self, **post: Any) -> tuple[RealSenseRGBDStreamer, list[str]]:
        built: list[str] = []
        streamer = RealSenseRGBDStreamer(_rig_4x4(**post), rs_module=_sdk_double([], built))
        return streamer, built

    def test_the_notice_builds_a_fresh_temporal_filter_in_its_place(self) -> None:
        streamer, built = self._streamer(spatial=True, temporal=True)
        streamer.open()
        chain = list(streamer._filters)
        streamer.camera_moved()
        self.assertEqual(built.count("temporal"), 2)
        self.assertEqual([step.name for step in streamer._filters], [step.name for step in chain])
        self.assertIsNot(streamer._filters[2], chain[2])
        self.assertTrue(all(new is old for i, (new, old) in enumerate(zip(streamer._filters, chain)) if i != 2))

    def test_the_notice_is_a_no_op_without_a_temporal_filter(self) -> None:
        streamer, built = self._streamer(spatial=True, temporal=False)
        streamer.open()
        streamer.camera_moved()
        self.assertNotIn("temporal", built)

    def test_the_notice_before_open_and_after_release_touches_nothing(self) -> None:
        streamer, built = self._streamer(temporal=True)
        streamer.camera_moved()
        self.assertEqual(built, [])
        streamer.open()
        streamer.release()
        streamer.camera_moved()
        self.assertEqual(built.count("temporal"), 1)


class _Moving:
    """A device double that counts the notices it was given, and whether one arrived during a grab."""

    def __init__(self) -> None:
        self.moves = 0
        self.inside = False
        self.during_grab = 0

    def open(self) -> None:
        pass

    def release(self) -> None:
        pass

    def grab(self) -> Any:
        self.inside = True
        threading.Event().wait(0.05)
        self.inside = False
        return _rgbd_frame()

    def camera_moved(self) -> None:
        if self.inside:
            self.during_grab += 1
        self.moves += 1


def _rgbd(rig_id: str = "wrist") -> RGBDDeviceRigConfig:
    return RGBDDeviceRigConfig(source="rgbd", rig_id=rig_id, enabled=True, fps=30, backend=0, device_index=7,
                               calibration_paths={"base_dir": "calibration/wrist"})


class TheOwnerPassesTheNoticeOnTests(unittest.TestCase):

    def test_the_owner_hands_the_notice_to_its_device(self) -> None:
        device = _Moving()
        with Camera.from_rig(_rgbd(), streamer=device) as camera:
            camera.camera_moved()
        self.assertEqual(device.moves, 1)

    def test_the_notice_waits_for_a_grab_in_flight(self) -> None:
        """Under the rig's lock, so it never swaps the filter out from under a grab."""
        device = _Moving()
        with Camera.from_rig(_rgbd(), streamer=device) as camera:
            grabbing = threading.Thread(target=camera.grab)
            grabbing.start()
            while not device.inside and grabbing.is_alive():
                threading.Event().wait(0.001)
            camera.camera_moved()
            grabbing.join()
        self.assertEqual((device.moves, device.during_grab), (1, 0))

    def test_a_device_that_keeps_no_history_takes_the_notice_as_nothing(self) -> None:
        device = SimpleNamespace(open=lambda: None, release=lambda: None, grab=_rgbd_frame)
        with Camera.from_rig(_rgbd(), streamer=device) as camera:
            camera.camera_moved()   # no attribute on the device, and no error

    def test_a_handle_reaches_the_notice_through_the_owner_and_through_a_catalogue(self) -> None:
        device = _Moving()
        with Camera.from_rig(_rgbd(), streamer=device) as camera:
            camera.handle().camera_moved()
        self.assertEqual(device.moves, 1)

        catalogue_device = _Moving()

        class _Catalogue(FrameProvider):
            _create_streamer = staticmethod(lambda _rig: catalogue_device)

        provider = _Catalogue([_rgbd("cat")])
        provider.open_rig("cat")
        try:
            handle = provider.rig("cat")
            self.assertIsInstance(handle, RigHandle)
            handle.camera_moved()
        finally:
            provider.release()
        self.assertEqual(catalogue_device.moves, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
