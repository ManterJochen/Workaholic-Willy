"""A camera window never changes what the robot measures: looking takes the display path, and yields to every grab.

The owner, 2026-09-24: while poses are taught and while the cell picks, every camera of the cell shows live in a window
of its own, fixed cameras too. A window must not degrade perception. What a display frame could have broken, and what
holds it here:

* A RealSense's temporal filter averages each depth pixel over the frames the driver handed it and fills a hole with a
  depth it saw in them. ``camera_moved()`` drops that history after the arm moved a wrist camera, and a carried
  camera drops it by itself after a pause between two grabs (``test_a_moved_wrist_camera_forgets_the_last_pose.py``).
  A window that grabbed through ``grab()`` would hand the filter frames taken while the arm moved, after the reset, and
  would keep the pause clock fresh so the pause never came. So a window looks through ``Camera.peek()``: the driver's
  newest colour image (``poll_for_frames``), no filter, no alignment, no pause clock. On the SDK's own filters below,
  a frame looked at between the notice and the next grab leaves the new pose's holes empty, where the same frame
  taken with ``grab()`` fills them with the old pose.
* A measuring grab is never kept waiting by a display frame inside a burst: ``peek()`` never waits for the rig's lock
  (it gives nothing while a grab holds it), and hands out nothing for a quiet spell after every measuring grab and
  after ``camera_moved()``, longer than the pause the driver still counts as one burst. At most one display read, begun
  before a burst, can stand in front of its first grab.
* A frame's stamp stays the time read just before its own device read, and a display frame carries a stamp of its own.
* A device that keeps a history and offers no display path is not looked at at all, and a closed owner refuses a look
  before its device is touched.

The window's own behaviour is ``test_every_camera_of_the_cell_can_be_watched_live.py``.
"""

from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from src.calibration import preview as preview_module
from src.camera.orchestration.camera import PEEK_QUIET_S, Camera, CameraNotOpen, PeekFrame
from src.camera.setup.image_taking.frames import RGBDFrame, StereoFrame
from src.camera.setup.image_taking.rgbd import RealSenseRGBDStreamer
from src.config.schema.camera import RGBDDeviceRigConfig
from tests.test_a_moved_wrist_camera_forgets_the_last_pose import (
    _HAS_RS,
    _HOLE,
    _SEEN,
    _SKIP,
    _assert_reads,
    _Clock,
    _pose_a,
    _pose_b,
    _rig,
    _rig_4x4,
    _sdk_double,
    _SdkWithAScene,
    _SoftwareScene,
)
from tests.test_calibration_sweep_preview import _Gui, _until

_COLOUR = np.full((6, 8, 3), 90, dtype=np.uint8)


def _rgbd(rig_id: str = "wrist", *, fps: int = 30) -> RGBDDeviceRigConfig:
    return RGBDDeviceRigConfig(source="rgbd", rig_id=rig_id, enabled=True, fps=fps, backend=0, device_index=7,
                               calibration_paths={"base_dir": f"calibration/{rig_id}"})


class _Device:
    """The device behind a real ``Camera`` owner, keeping a history as a RealSense does: every call it gets, from
    which thread, and when it began and ended inside the device (so under the rig's lock)."""

    def __init__(self, *, grab_s: float = 0.0, peek_s: float = 0.0) -> None:
        self.grab_s = grab_s
        self.peek_s = peek_s
        self.calls: list[tuple[str, int, float, float]] = []
        self.inside = threading.Event()

    def _record(self, name: str, hold_s: float) -> None:
        started = time.perf_counter()
        self.inside.set()
        if hold_s:
            threading.Event().wait(hold_s)
        self.inside.clear()
        self.calls.append((name, threading.get_ident(), started, time.perf_counter()))

    def open(self) -> None:
        pass

    def release(self) -> None:
        pass

    def grab(self) -> RGBDFrame:
        self._record("grab", self.grab_s)
        return RGBDFrame(color=_COLOUR.copy(), depth=np.full(_COLOUR.shape[:2], 500, np.uint16))

    def camera_moved(self) -> None:
        self._record("camera_moved", 0.0)

    def peek(self) -> np.ndarray | None:
        self._record("peek", self.peek_s)
        return _COLOUR

    def names(self, thread: int | None = None) -> list[str]:
        return [name for name, ident, _, _ in self.calls if thread is None or ident == thread]


class _NoDisplayPath(_Device):
    """A device that keeps a history (``camera_moved``) and offers no display path."""

    peek = None  # type: ignore[assignment]


class _NoHistory:
    """A device that keeps no history: a webcam, an OpenCV RGB-D device. It is looked at with a plain grab."""

    def __init__(self, frame: Any) -> None:
        self.frame = frame
        self.grabs = 0

    def open(self) -> None:
        pass

    def release(self) -> None:
        pass

    def grab(self) -> Any:
        self.grabs += 1
        return self.frame


def _open(device: Any, *, fps: int = 30) -> Camera:
    camera = Camera.from_rig(_rgbd(fps=fps), streamer=device)
    camera.open()
    return camera


class _Tick:
    """The owner's monotonic clock, moved by hand."""

    def __init__(self) -> None:
        self.now = 50.0

    def __call__(self) -> float:
        return self.now


# ---------------------------------------------------------------------------------------------------------------------
# The owner: a display frame through the display path, and none while the rig measures
# ---------------------------------------------------------------------------------------------------------------------


class TheOwnerHandsOutDisplayFramesApartTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tick = _Tick()
        self.enterContext(mock.patch("src.camera.orchestration.camera._monotonic", self.tick))

    def test_a_display_frame_comes_through_the_devices_display_path_and_nothing_else(self) -> None:
        device = _Device()
        camera = _open(device)
        self.addCleanup(camera.release)
        before = time.time()
        shot = camera.peek()
        assert shot is not None
        self.assertIsInstance(shot, PeekFrame)
        self.assertEqual(device.names(), ["peek"], "a display frame reached the device another way")
        np.testing.assert_array_equal(shot.color, _COLOUR)
        self.assertFalse(np.shares_memory(shot.color, _COLOUR), "a kept view would hold the device's buffer")
        self.assertGreaterEqual(shot.captured_at_s, before)
        self.assertLessEqual(shot.captured_at_s, time.time())

    def test_a_device_that_keeps_no_history_is_looked_at_with_a_plain_grab(self) -> None:
        eye = np.full((4, 6, 3), 30, dtype=np.uint8)
        for frame, colour in ((RGBDFrame(color=eye, depth=np.empty(0, np.uint16)), eye),
                              (StereoFrame(left=eye, right=eye + 1), eye)):
            with self.subTest(type(frame).__name__):
                device = _NoHistory(frame)
                with _open(device) as camera:
                    shot = camera.peek()
                assert shot is not None
                np.testing.assert_array_equal(shot.color, colour)
                self.assertEqual(device.grabs, 1)

    def test_a_device_that_keeps_a_history_and_offers_no_display_path_is_never_read_for_display(self) -> None:
        device = _NoDisplayPath()
        camera = _open(device)
        self.addCleanup(camera.release)
        with self.assertRaisesRegex(RuntimeError, "keeps a history .* offers no display frame"):
            camera.peek()
        self.assertEqual(device.names(), [], "the device was read for a display frame")

    def test_no_display_frame_in_the_quiet_after_a_measuring_grab_or_a_notice_that_the_camera_moved(self) -> None:
        device = _Device()
        camera = _open(device)
        self.addCleanup(camera.release)
        for measure in (camera.grab, camera.camera_moved):
            with self.subTest(measure.__name__):
                measure()
                self.tick.now += PEEK_QUIET_S - 0.01
                self.assertIsNone(camera.peek(), "a display frame inside a burst of measuring grabs")
                self.tick.now += 0.02
                self.assertIsNotNone(camera.peek())
        self.assertEqual(device.names(), ["grab", "peek", "camera_moved", "peek"])

    def test_a_slow_mode_stays_quiet_for_three_frame_periods(self) -> None:
        camera = _open(_Device(), fps=5)
        self.addCleanup(camera.release)
        camera.grab()
        self.tick.now += 0.55
        self.assertIsNone(camera.peek(), "three frame periods at 5 fps is 0.6 s")
        self.tick.now += 0.1
        self.assertIsNotNone(camera.peek())

    def test_a_display_frame_on_a_closed_owner_is_refused_before_the_device_is_touched(self) -> None:
        device = _Device()
        camera = _open(device)
        camera.release()
        with self.assertRaises(CameraNotOpen):
            camera.peek()
        self.assertEqual(device.names(), [])

    def test_a_handle_hands_out_the_owners_display_frames(self) -> None:
        device = _Device()
        camera = _open(device)
        self.addCleanup(camera.release)
        shot = camera.handle().peek()
        assert shot is not None
        self.assertEqual(device.names(), ["peek"])


class ADisplayFrameNeverWaitsForAMeasuringGrabTests(unittest.TestCase):
    """Real time: a grab in flight on another thread, and a look taken meanwhile."""

    def test_no_display_frame_while_a_measuring_grab_holds_the_rig_and_the_look_does_not_wait(self) -> None:
        device = _Device(grab_s=0.3)
        camera = _open(device)
        self.addCleanup(camera.release)
        grabbing = threading.Thread(target=camera.grab)
        grabbing.start()
        self.assertTrue(device.inside.wait(2.0))
        started = time.perf_counter()
        self.assertIsNone(camera.peek())
        self.assertLess(time.perf_counter() - started, 0.1, "the look waited for the measuring grab")
        grabbing.join()
        self.assertEqual(device.names(), ["grab"])

    def test_a_measuring_grab_that_waited_for_a_display_frame_is_stamped_when_its_own_read_begins(self) -> None:
        device = _Device(peek_s=0.2)
        camera = _open(device)
        self.addCleanup(camera.release)
        looking = threading.Thread(target=camera.peek)
        looking.start()
        self.assertTrue(device.inside.wait(2.0))
        frame = camera.grab()
        looking.join()
        (_, _, _, peek_ended), = [call for call in device.calls if call[0] == "peek"]
        # The stamp is wall time and the device's log perf time: compare both against the same instant.
        offset = time.time() - time.perf_counter()
        self.assertGreaterEqual(frame.captured_at_s, peek_ended + offset - 0.01,
                                "the stamp was read before the rig was the grab's")


# ---------------------------------------------------------------------------------------------------------------------
# A live view beside a planning world's refreshes: the order of every call the device gets
# ---------------------------------------------------------------------------------------------------------------------


class ALiveViewBesideTheMeasuringGrabsTests(unittest.TestCase):
    """The owner's refresh, as ``RigDepthSource`` makes it on a wrist camera: the notice, five warm-ups thrown away,
    the grab kept. Four of them, a pause after each, and a live view of the same owner running all along."""

    def test_the_window_only_looks_and_never_between_a_notice_and_the_grab_that_follows_it(self) -> None:
        from src.camera.live_view import LiveView

        device = _Device(grab_s=0.01, peek_s=0.02)
        camera = _open(device)
        self.addCleanup(camera.release)
        gui = _Gui()
        waits: list[tuple[str, float, float]] = []

        def timed(name: str, call: Any) -> None:
            asked = time.perf_counter()
            call()
            waits.append((name, asked, time.perf_counter()))

        main = threading.get_ident()
        with mock.patch.object(preview_module, "preview_unavailable", lambda: None):
            with LiveView([camera], gui=gui, say=lambda _line: None) as view:
                self.assertTrue(_until(lambda: "peek" in device.names()))
                sequences: list[tuple[float, float]] = []
                for _ in range(4):
                    timed("camera_moved", camera.camera_moved)
                    for _ in range(6):
                        timed("grab", camera.grab)
                    moved = [c for c in device.calls if c[0] == "camera_moved" and c[1] == main][-1]
                    kept = [c for c in device.calls if c[0] == "grab" and c[1] == main][-1]
                    sequences.append((moved[2], kept[3]))
                    time.sleep(PEEK_QUIET_S + 0.35)  # the pause after a refresh: the window may look again
                self.assertTrue(view.running, view.off)
        peeks = [(started, ended) for name, _, started, ended in device.calls if name == "peek"]
        window_threads = {ident for name, ident, _, _ in device.calls if name == "peek"}
        self.assertNotIn(main, window_threads)
        self.assertEqual(window_threads, gui.threads(), "a look from another thread than the windows'")
        self.assertEqual(set(device.names(next(iter(window_threads)))), {"peek"},
                         "the window's thread made a measuring call")
        self.assertGreaterEqual(len(peeks), 4, "the window stopped looking between the refreshes")
        for begun, kept in sequences:
            inside = [p for p in peeks if p[1] > begun and p[0] < kept]
            self.assertEqual([], inside, "a look between the notice and the grab kept after it")
        # Inside a burst no measuring call waited for a look; before one, at most one look stood in front of it.
        for index, (name, asked, done) in enumerate(waits):
            in_front = [p for p in peeks if p[1] > asked and p[0] < done]
            self.assertLessEqual(len(in_front), 0 if name == "grab" else 1, f"call {index} ({name}) waited")

    def test_a_camera_given_back_while_its_window_is_open_is_not_read_again(self) -> None:
        """A campaign's teardown gives the cameras back before the caller closes the view: the owner refuses every
        look after that before the device is touched, and the window says why."""
        from src.camera.live_view import LiveView

        device = _Device()
        camera = _open(device)
        with mock.patch.object(preview_module, "preview_unavailable", lambda: None):
            with LiveView([camera], gui=_Gui(), say=lambda _line: None) as view:
                self.assertTrue(_until(lambda: "peek" in device.names()))
                camera.release()
                read = len(device.calls)
                self.assertTrue(_until(lambda: view._windows["wrist"].live_error.startswith("CameraNotOpen")))
                time.sleep(0.3)
        self.assertEqual(len(device.calls), read, "the device was read after it was given back")


# ---------------------------------------------------------------------------------------------------------------------
# The RealSense driver's display path
# ---------------------------------------------------------------------------------------------------------------------


class _PolledFrameset:
    """What ``poll_for_frames`` hands back: a frameset whose colour is the device's buffer, or nothing new."""

    def __init__(self, colour: np.ndarray | None) -> None:
        self._colour = colour

    def __bool__(self) -> bool:
        return self._colour is not None

    def get_color_frame(self) -> Any:
        return None if self._colour is None else SimpleNamespace(get_data=lambda: self._colour)


class TheRealSenseDriverLooksWithoutTouchingItsFiltersTests(unittest.TestCase):

    def _opened(self, order: list[str], polled: list[_PolledFrameset]) -> RealSenseRGBDStreamer:
        sdk = _sdk_double(order, [])
        pipeline = sdk.pipeline()
        pipeline.poll_for_frames = lambda: polled.pop(0)
        streamer = RealSenseRGBDStreamer(_rig_4x4(decimation=True, spatial=True, temporal=True, hole_filling=True),
                                         rs_module=sdk)
        streamer.open()
        self.addCleanup(streamer.release)
        return streamer

    def test_a_display_frame_runs_no_filter_and_no_alignment_and_is_a_copy(self) -> None:
        order: list[str] = []
        buffer = np.full((4, 4, 3), 200, dtype=np.uint8)
        streamer = self._opened(order, [_PolledFrameset(buffer)])
        colour = streamer.peek()
        assert colour is not None
        self.assertEqual(order, [], "a display frame went through the filter chain or the alignment")
        np.testing.assert_array_equal(colour, buffer)
        self.assertFalse(np.shares_memory(colour, buffer), "a kept view would hold the device's buffer")
        streamer.grab()
        self.assertEqual(order, ["decimation", "to_disparity", "spatial", "temporal", "to_depth", "hole", "align"],
                         "the control: a measuring grab runs the whole chain")

    def test_a_display_frame_leaves_the_pause_clock_alone(self) -> None:
        streamer = self._opened([], [_PolledFrameset(np.zeros((4, 4, 3), np.uint8))])
        streamer.grab()
        grabbed_at = streamer._last_grab_s
        streamer.peek()
        self.assertEqual(streamer._last_grab_s, grabbed_at)

    def test_no_new_frameset_is_no_display_frame_and_no_wait(self) -> None:
        streamer = self._opened([], [_PolledFrameset(None)])
        self.assertIsNone(streamer.peek())

    def test_a_closed_driver_gives_no_display_frame(self) -> None:
        streamer = RealSenseRGBDStreamer(_rig_4x4(), rs_module=_sdk_double([], []))
        with self.assertRaisesRegex(RuntimeError, "not open"):
            streamer.peek()


class _PolledScene(_SoftwareScene):
    """The software scene, polled as a live view polls it: the frameset the device holds now."""

    def poll_for_frames(self) -> Any:
        return self.wait_for_frames()


@unittest.skipUnless(_HAS_RS, _SKIP)
class OnTheSdksOwnFiltersTests(unittest.TestCase):
    """The driver's own ``open``, ``grab`` and ``peek`` on librealsense's filters and alignment; the control of each
    test is the same run with the window grabbing, which fills the new pose's holes with the old one."""

    def _opened(self, rig: RGBDDeviceRigConfig, clock: _Clock) -> tuple[RealSenseRGBDStreamer, _PolledScene]:
        scene = _PolledScene()
        streamer = RealSenseRGBDStreamer(rig, rs_module=_SdkWithAScene(scene), clock=clock)
        streamer.open()
        self.addCleanup(streamer.release)
        return streamer, scene

    def _four_at_pose_a(self, streamer: RealSenseRGBDStreamer, scene: _PolledScene, clock: _Clock) -> None:
        scene.depth = _pose_a()
        for _ in range(4):
            clock.now += 0.033
            _assert_reads(streamer.grab().depth[_SEEN], 600)

    def test_frames_looked_at_between_the_notice_and_the_next_grab_fill_no_hole_of_the_new_pose(self) -> None:
        """The planning world said the camera moved; frames the device still held from the old pose were looked at
        (the arm settling, the window at four a second); then the world's grab at the new pose."""
        for look in ("peek", "grab"):
            with self.subTest(window=look):
                clock = _Clock()
                streamer, scene = self._opened(_rig(), clock)
                self._four_at_pose_a(streamer, scene, clock)
                streamer.camera_moved()
                for _ in range(4):
                    getattr(streamer, look)()
                scene.depth = _pose_b()
                frame = streamer.grab()
                if look == "peek":
                    np.testing.assert_array_equal(frame.depth[_HOLE], 0)
                    _assert_reads(frame.depth[_SEEN], 612)
                else:  # the hazard a window grabbing through grab() is: the old pose in the new pose's holes
                    _assert_reads(frame.depth[_HOLE], 600)

    def test_frames_looked_at_during_a_move_leave_a_carried_camera_its_pause(self) -> None:
        """Nobody said the camera moved: a carried camera drops its history after the pause the move took, unless the
        window's frames kept its pause clock fresh."""
        for look in ("peek", "grab"):
            with self.subTest(window=look):
                clock = _Clock()
                streamer, scene = self._opened(_rig(carried=True), clock)
                self._four_at_pose_a(streamer, scene, clock)
                for _ in range(8):  # two seconds of a move, the window looking about four times a second
                    clock.now += 0.24
                    getattr(streamer, look)()
                scene.depth = _pose_b()
                clock.now += 0.1
                frame = streamer.grab()
                if look == "peek":
                    np.testing.assert_array_equal(frame.depth[_HOLE], 0)
                    _assert_reads(frame.depth[_SEEN], 612)
                else:
                    _assert_reads(frame.depth[_HOLE], 600)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
