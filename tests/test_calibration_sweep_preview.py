"""The window beside a hand-eye sweep: display only, one GUI thread, closed before the camera is given back.

Issue 4 of the calibration chain (owner, 2026-09-23): an operator running examples 09 and 10 could not see whether
the board was in view. ``src/calibration/preview.py`` shows the camera while the arm moves and pins each judged frame
with the target drawn on it. What is held here:

* the sweep's thread only enqueues and never waits, whatever the window does;
* every HighGUI call is made on the preview's own thread, and an error there switches the preview off without
  reaching the sweep;
* live frames come through the handle the sweep reads, and nothing else;
* ``annotate`` is pure and draws with no window;
* the noun builds the preview at the build, starts it only once the arm is connected, so a dry run opens nothing and
  grabs nothing, and closes it before the camera is given back on every way out of the sweep;
* ``"auto"`` stays shut when stdout is not a terminal, so the CLI's pinned transcripts do not change.

Every test drives a fake GUI module, except the last class, which opens a real window and runs only when
``WILLY_PREVIEW_SMOKE=1`` is set on a desktop that can show one.
"""

from __future__ import annotations

import ast
import datetime as dt
import io
import os
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import MagicMock, patch

import numpy as np

from src.calibration import Extrinsics, MountingMode
from src.calibration import preview as preview_module
from src.calibration.preview import MAX_LIVE_HZ, PreviewState, SweepPreview, annotate, preview_unavailable
from src.calibration.rgbd_marker_source import RGBDArucoMarkerSource
from src.calibration.stereo.sub_modules.aruco_esti import ArucoPoseEstimator
from src.calibration.targets import CharucoPoseEstimator
from src.camera.orchestration.camera import Camera
from src.config.schema.robot import RobotConfig
from src.geometry import Frame, Transform
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.events import RobotCalibrationEvent
from src.robot.execution.calibration import CalibrationResult, CalibrationRoutine
from src.robot.execution.cell_lock import CellLock, lock_path_for
from src.robot.execution.hand_eye import (
    CalibrationOutcome,
    HandEyeCalibration,
    SweepOptions,
    render_sweep_event,
)
from src.robot.execution.real_cell import calibrate
from src.robot.execution.robot import Robot
from tests.test_calibration_targets import _K, _board_view, _marker_view
from tests.test_camera_boundaries import _rgbd_rig

_IP = "10.253.253.77"
_KEY = f"ur@{_IP}"
_WIN = "General configuration for OpenCV 4.13.0\n  GUI:                           WIN32UI\n    Win32 UI:  YES\n"
_HEADLESS = "General configuration for OpenCV 4.13.0\n  GUI:                           NONE\n"
_MOVING = RobotCalibrationEvent.MOVING_TO_POSE
_ACCEPTED = RobotCalibrationEvent.POSE_ACCEPTED
_REJECTED = RobotCalibrationEvent.POSE_REJECTED


def _until(predicate: Callable[[], bool], timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class _Gui:
    """A HighGUI double: records which thread made each call, keeps what was shown, and can refuse or hang."""

    WINDOW_AUTOSIZE = 1
    WND_PROP_VISIBLE = 4

    def __init__(self, *, keys: tuple[int, ...] = (), visible: float = 1.0, fail: str = "",
                 hang: threading.Event | None = None) -> None:
        self.calls: list[tuple[str, int]] = []
        self.shown: list[np.ndarray] = []
        self.keys = list(keys)
        self.visible = visible
        self.fail = fail
        self.hang = hang
        self.log: list[str] | None = None

    def _call(self, name: str) -> None:
        self.calls.append((name, threading.get_ident()))
        if self.log is not None:
            self.log.append(f"gui.{name}")
        if name == self.fail:
            raise RuntimeError("the display went away")

    def namedWindow(self, _title: str, _flags: int) -> None:  # noqa: N802 (cv2's name)
        self._call("namedWindow")

    def imshow(self, _title: str, image: np.ndarray) -> None:
        self._call("imshow")
        self.shown.append(image)

    def waitKey(self, delay_ms: int) -> int:  # noqa: N802 (cv2's name)
        self._call("waitKey")
        if self.hang is not None:
            self.hang.wait()
        time.sleep(delay_ms / 1000.0)
        return self.keys.pop(0) if self.keys else -1

    def getWindowProperty(self, _title: str, _prop: int) -> float:  # noqa: N802 (cv2's name)
        self._call("getWindowProperty")
        return self.visible

    def destroyWindow(self, _title: str) -> None:  # noqa: N802 (cv2's name)
        self._call("destroyWindow")

    def threads(self) -> set[int]:
        return {ident for _, ident in self.calls}


class _Handle:
    """What the sweep's marker source reads: counts its grabs and the threads they came from."""

    def __init__(self, frame: np.ndarray | None = None, *, fail: bool = False) -> None:
        self.frame = _marker_view() if frame is None else frame
        self.fail = fail
        self.grabs: list[int] = []

    def grab(self) -> Any:
        self.grabs.append(threading.get_ident())
        if self.fail:
            raise RuntimeError("Frame didn't arrive within 5000")
        return SimpleNamespace(color=self.frame)

    def get_intrinsics(self) -> np.ndarray:
        return _K

    def get_distortion(self) -> None:
        return None


def _preview(handle: Any = None, gui: Any = None, **options: Any) -> tuple[SweepPreview, list[str]]:
    said: list[str] = []
    preview = SweepPreview(_Handle() if handle is None else handle, gui=_Gui() if gui is None else gui,
                           say=said.append, **options)
    return preview, said


# ---------------------------------------------------------------------------------------------------------------------
# Where a window can show
# ---------------------------------------------------------------------------------------------------------------------


class WhereAWindowCanShowTests(unittest.TestCase):

    def test_a_windows_desktop_and_a_linux_display_can_show_one(self) -> None:
        self.assertIsNone(preview_unavailable(environ={}, platform="win32", build_information=_WIN))
        for key in ("DISPLAY", "WAYLAND_DISPLAY"):
            with self.subTest(key):
                self.assertIsNone(preview_unavailable(environ={key: ":0"}, platform="linux",
                                                      build_information=_WIN.replace("WIN32UI", "QT5")))

    def test_each_reason_is_named(self) -> None:
        cases: dict[str, dict[str, Any]] = {
            "WILLY_NO_PREVIEW is set": dict(environ={"WILLY_NO_PREVIEW": "1"}, platform="win32",
                                            build_information=_WIN),
            "no GUI": dict(environ={}, platform="win32", build_information=_HEADLESS),
            "no display": dict(environ={}, platform="linux", build_information=_WIN),
            "SSH session": dict(environ={"SSH_CONNECTION": "10.0.0.2 5022 10.0.0.1 22"}, platform="win32",
                                build_information=_WIN),
            "main thread": dict(environ={}, platform="darwin", build_information=_WIN),
        }
        for said, probe in cases.items():
            with self.subTest(said):
                self.assertIn(said, preview_unavailable(**probe) or "")
        self.assertIn("SSH", preview_unavailable(environ={"SSH_CLIENT": "x"}, platform="win32",
                                                 build_information=_WIN) or "")

    def test_the_switch_reads_empty_and_zero_as_not_set(self) -> None:
        for value in ("", "0", " "):
            with self.subTest(repr(value)):
                self.assertIsNone(preview_unavailable(environ={"WILLY_NO_PREVIEW": value}, platform="win32",
                                                      build_information=_WIN))

    def test_this_opencv_names_its_gui_backend(self) -> None:
        """The probe reads the real build information: the pinned wheel has a GUI line to read."""
        import cv2

        self.assertTrue(preview_module._gui_backend(cv2.getBuildInformation()))


# ---------------------------------------------------------------------------------------------------------------------
# annotate: pure and headless
# ---------------------------------------------------------------------------------------------------------------------


def _board_observation() -> tuple[np.ndarray, Any]:
    frame = _board_view()
    estimator = CharucoPoseEstimator(squares_x=7, squares_y=5, square_length_mm=30.0, marker_length_mm=22.0)
    return frame, estimator.observe(frame, _K, np.zeros(5))


class AnnotateTests(unittest.TestCase):

    def test_it_draws_on_a_copy_and_opens_no_window(self) -> None:
        frame, observation = _board_observation()
        before = frame.copy()
        explode = MagicMock(side_effect=AssertionError("annotate touched a window"))
        with patch.object(preview_module.cv, "imshow", explode), patch.object(preview_module.cv, "namedWindow", explode):
            image = annotate(frame, observation, _K, None, lines=["JUDGED pose 3/22", "24 corners"], tone="counted",
                             history=[True, False, None], current=2, axis_mm=60.0)
        np.testing.assert_array_equal(frame, before)
        self.assertEqual((image.dtype, image.shape[1], image.shape[2]), (np.uint8, frame.shape[1], 3))
        self.assertGreater(image.shape[0], frame.shape[0])
        self.assertFalse(np.shares_memory(image, frame))

    def test_the_target_is_drawn_where_it_was_seen_and_nothing_without_one(self) -> None:
        frame, observation = _board_observation()
        drawn = annotate(frame, observation, _K, None, lines=["x"])
        plain = annotate(frame, None, lines=["x"])
        rows = frame.shape[0]
        # The frame sits under the strip: the same strip height for the same lines.
        top = plain.shape[0] - rows
        np.testing.assert_array_equal(plain[top:], frame)
        changed = np.argwhere(np.any(drawn[top:] != frame, axis=2))
        self.assertGreater(len(changed), 500)
        # Around the board, not somewhere else: the marker corners bound what was drawn, give or take the ids.
        corners = np.vstack(observation.marker_corners)
        self.assertLess(abs(np.median(changed[:, 1]) - np.median(corners[:, 0])), 60)
        self.assertLess(abs(np.median(changed[:, 0]) - np.median(corners[:, 1])), 60)

    def test_a_shrunk_frame_is_drawn_at_its_scale(self) -> None:
        frame = _marker_view(marker_id=7)
        observation = ArucoPoseEstimator(marker_length_mm=50.0, target_id=7).observe(frame, _K, np.zeros(5))
        small, factor = preview_module._shrink(frame, 640)
        self.assertEqual((small.shape[1], factor), (640, 0.5))
        drawn = annotate(small, observation, _K, None, lines=["x"], scale=factor)
        top = drawn.shape[0] - small.shape[0]
        corner = observation.marker_corners[0][0] * factor
        x, y = int(round(corner[0])), int(round(corner[1]))
        window = drawn[top + y - 3: top + y + 4, x - 3: x + 4]
        self.assertTrue(np.any(window != small[y - 3: y + 4, x - 3: x + 4]), "no outline at the scaled corner")

    def test_the_band_says_counted_or_rejected_in_its_colour(self) -> None:
        frame = np.zeros((120, 400, 3), dtype=np.uint8)
        for tone, colour in (("counted", (50, 140, 50)), ("rejected", (40, 40, 190)), ("live", (120, 80, 30))):
            with self.subTest(tone):
                image = annotate(frame, lines=["pose 1/2  REJECTED marker_not_found"], tone=tone)
                self.assertEqual(tuple(int(v) for v in image[2, 200]), colour)

    def test_the_history_row_marks_each_pose(self) -> None:
        frame = np.zeros((60, 400, 3), dtype=np.uint8)
        image = annotate(frame, lines=["x"], history=[True, False, None], current=3)
        row = image[-preview_module._HISTORY_PX:]
        step = min(24, (400 - 16) // 3)
        middle = preview_module._HISTORY_PX // 2
        self.assertEqual(tuple(int(v) for v in row[middle, 8 + step // 2]), (50, 140, 50))
        self.assertEqual(tuple(int(v) for v in row[middle, 8 + step + step // 2]), (40, 40, 190))
        self.assertEqual(tuple(int(v) for v in row[middle, 8 + 2 * step + step // 2]), (90, 90, 90))

    def test_text_that_is_not_ascii_or_too_long_is_still_drawn(self) -> None:
        frame = np.zeros((60, 320, 3), dtype=np.uint8)
        image = annotate(frame, lines=["pose → 1/2 " + "x" * 400, "Grad °"], tone="rejected")
        # Three rows at most for the long line, one for the short one, and the footer wrapped at this width.
        self.assertLessEqual(image.shape[0], 60 + 8 + preview_module._ROW_PX * 6)


# ---------------------------------------------------------------------------------------------------------------------
# PreviewState: the events, as the window says them
# ---------------------------------------------------------------------------------------------------------------------


class PreviewStateTests(unittest.TestCase):

    def test_the_event_names_are_the_routines(self) -> None:
        self.assertEqual(
            (preview_module._MOVING, preview_module._DETECTED, preview_module._ACCEPTED, preview_module._REJECTED),
            (RobotCalibrationEvent.MOVING_TO_POSE, RobotCalibrationEvent.MARKER_DETECTED,
             RobotCalibrationEvent.POSE_ACCEPTED, RobotCalibrationEvent.POSE_REJECTED))

    def test_a_sweep_is_followed_pose_by_pose_in_the_consoles_words(self) -> None:
        state = PreviewState(describe=render_sweep_event)
        state.apply(_MOVING, {"index": 1, "total": 3, "label": "look_0", "accepted": 0, "min_samples": 2})
        state.apply(_ACCEPTED, {"index": 1, "total": 3, "label": "look_0", "accepted": 1, "min_samples": 2,
                                "detail": "24 corners"})
        state.apply(_MOVING, {"index": 2, "total": 3, "label": "look_1", "accepted": 1, "min_samples": 2})
        rejected = {"index": 2, "total": 3, "label": "look_1", "reason": "marker_not_found",
                    "detail": "marker 0 not in view; DICT_5X5_100 ids in view: [7]"}
        state.apply(_REJECTED, rejected)
        self.assertEqual(state.history(), [True, False, None])
        self.assertEqual(state.judged_lines(2, None)[0], f"JUDGED  {render_sweep_event(_REJECTED, rejected)}")
        self.assertEqual((state.judged_tone(1), state.judged_tone(2), state.judged_tone(3)),
                         ("counted", "rejected", "judging"))
        self.assertEqual(state.count_line(), "1 counted, 2 needed, 3 poses in this sweep")
        state.apply(_MOVING, {"index": 3, "total": 3, "label": "look_2", "accepted": 1, "min_samples": 2})
        live = state.live_lines(None)
        self.assertTrue(live[0].startswith("LIVE, not judged  pose  3/3 'look_2'  moving"), live[0])
        self.assertIn("REJECTED marker_not_found", live[2])

    def test_a_judged_frame_before_its_verdict_says_it_is_being_judged(self) -> None:
        state = PreviewState()
        state.apply(_MOVING, {"index": 4, "total": 22, "label": "look_3", "accepted": 3})
        self.assertEqual(state.judged_lines(4, None)[0], "JUDGED  pose 4/22 'look_3'  judging")


# ---------------------------------------------------------------------------------------------------------------------
# SweepPreview: the sweep enqueues, one thread draws
# ---------------------------------------------------------------------------------------------------------------------


class TheSweepNeverWaitsTests(unittest.TestCase):

    def test_a_full_queue_drops_its_oldest_and_never_blocks(self) -> None:
        preview, _ = _preview(queue_size=3)  # never started: nothing drains the queue
        frame = _board_view()
        started = time.perf_counter()
        slowest = 0.0
        for index in range(1, 41):
            before = time.perf_counter()
            preview(_MOVING, {"index": index, "total": 40})
            preview.observe(frame, None, _K, None)
            slowest = max(slowest, time.perf_counter() - before)
        self.assertLess(slowest, 0.1)
        self.assertLess(time.perf_counter() - started, 2.0)
        items = preview._drain()
        self.assertEqual(len(items), 3)
        self.assertEqual(preview.dropped, 77)
        self.assertEqual(items[-1].image.shape[1], 960, "the newest item is kept, downscaled")
        self.assertEqual(items[-2], (_MOVING, {"index": 40, "total": 40}))

    def test_a_judged_frame_is_copied_and_downscaled(self) -> None:
        preview, _ = _preview()
        frame = _board_view()
        preview.observe(frame, None, _K, np.zeros(5))
        (queued,) = preview._drain()
        self.assertFalse(np.shares_memory(queued.image, frame), "a kept view would hold the device's buffer")
        self.assertEqual((queued.image.shape[1], queued.scale), (960, 0.75))
        frame[:] = 0
        self.assertGreater(int(queued.image.max()), 0)

    def test_a_hook_that_is_the_preview_does_not_change_the_judged_pose(self) -> None:
        """The source's pose is the estimator's, with or without the preview as its hook."""
        handle = _Handle(_board_view())
        estimator = CharucoPoseEstimator(squares_x=7, squares_y=5, square_length_mm=30.0, marker_length_mm=22.0)
        bare = RGBDArucoMarkerSource(streamer=handle, estimator=estimator, warmup_grabs=0)()
        preview, _ = _preview()
        hooked = RGBDArucoMarkerSource(streamer=handle, estimator=estimator, warmup_grabs=0,
                                       on_observation=preview.observe)()
        assert bare is not None and hooked is not None
        np.testing.assert_array_equal(bare, hooked)
        self.assertEqual(len(preview._drain()), 1)

    def test_a_closed_preview_takes_nothing(self) -> None:
        preview, _ = _preview()
        preview.close()
        preview(_MOVING, {"index": 1})
        preview.observe(_board_view(), None, _K, None)
        self.assertEqual(preview._drain(), [])


class OneThreadDrawsTests(unittest.TestCase):

    def _run(self, gui: _Gui, handle: _Handle | None = None, *, seconds: float = 0.4,
             **options: Any) -> tuple[SweepPreview, list[str], _Handle]:
        handle = _Handle() if handle is None else handle
        preview, said = _preview(handle, gui, pin_s=0.3, **options)
        self.addCleanup(preview.close)
        preview.start()
        preview(_MOVING, {"index": 1, "total": 2, "label": "look_0", "accepted": 0, "min_samples": 1})
        frame = _marker_view()
        observation = ArucoPoseEstimator(marker_length_mm=50.0, target_id=0).observe(frame, _K, np.zeros(5))
        preview.observe(frame, observation, _K, None)
        preview(_ACCEPTED, {"index": 1, "total": 2, "label": "look_0", "accepted": 1, "min_samples": 1,
                            "detail": observation.summary()})
        time.sleep(seconds)
        return preview, said, handle

    def test_every_gui_call_is_made_on_the_previews_own_thread(self) -> None:
        gui = _Gui()
        preview, said, handle = self._run(gui)
        preview.close()
        self.assertFalse(preview.running)
        names = [name for name, _ in gui.calls]
        self.assertEqual((names[0], names[-2:]), ("namedWindow", ["destroyWindow", "waitKey"]))
        self.assertEqual(len(gui.threads()), 1)
        self.assertNotIn(threading.get_ident(), gui.threads())
        self.assertGreater(len(gui.shown), 1)
        self.assertEqual((said, preview.off), ([], ""))
        # The judged frame was pinned with its verdict: a green band over the frame shown.
        self.assertTrue(any(tuple(int(v) for v in image[2, 10]) == (50, 140, 50) for image in gui.shown))

    def test_live_frames_come_through_the_handle_given_on_the_previews_thread_at_most_five_a_second(self) -> None:
        gui = _Gui()
        started = time.monotonic()
        preview, _, handle = self._run(gui, seconds=1.2, live_hz=50.0)
        preview.close()
        elapsed = time.monotonic() - started
        grabs = len(handle.grabs)
        self.assertGreaterEqual(grabs, 2)
        # Asked for 50 a second: grabs are at least 1 / MAX_LIVE_HZ apart, the first one at the start.
        self.assertLessEqual(grabs, int(elapsed * MAX_LIVE_HZ) + 1)
        self.assertEqual(set(handle.grabs), gui.threads(), "a live grab from another thread than the window's")
        self.assertTrue(any(tuple(int(v) for v in image[2, 10]) == (120, 80, 30) for image in gui.shown),
                        "no live frame was shown")
        time.sleep(0.3)
        self.assertEqual(len(handle.grabs), grabs, "a grab after close")

    def test_the_module_reaches_no_camera_and_no_robot(self) -> None:
        """Live frames come through the handle given, never through an owner the preview builds."""
        tree = ast.parse(Path(preview_module.__file__).read_text(encoding="utf-8"))
        imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertEqual({name for name in imported if name.startswith("src")}, set())

    def test_a_failing_live_grab_is_said_on_the_window_and_the_preview_stays(self) -> None:
        handle, gui = _Handle(fail=True), _Gui()
        preview, said = _preview(handle, gui)
        self.addCleanup(preview.close)
        preview.start()
        self.assertTrue(_until(lambda: len(handle.grabs) == 1 and len(gui.shown) == 1))
        time.sleep(0.5)
        self.assertTrue(preview.running)
        self.assertEqual((said, preview.off), ([], ""))
        self.assertEqual(len(handle.grabs), 1, "a failed grab is retried after a second, not hammered")
        self.assertEqual(tuple(int(v) for v in gui.shown[0][2, 10]), (120, 80, 30))

    def test_close_joins_within_its_timeout(self) -> None:
        gui = _Gui()
        preview, _, _ = self._run(gui, seconds=0.1)
        started = time.perf_counter()
        preview.close(timeout_s=2.0)
        self.assertLess(time.perf_counter() - started, 0.5)
        self.assertFalse(preview.running)

    def test_a_hung_window_does_not_hold_the_close_past_its_timeout(self) -> None:
        release = threading.Event()
        self.addCleanup(release.set)
        preview, _ = _preview(gui=_Gui(hang=release), live_hz=0)
        preview.start()
        self.assertTrue(_until(lambda: preview.running))
        started = time.perf_counter()
        preview.close(timeout_s=0.3)
        self.assertLess(time.perf_counter() - started, 1.0)
        self.assertTrue(preview.running, "the thread is left behind, as a daemon")
        release.set()
        self.assertTrue(_until(lambda: not preview.running))

    def test_a_gui_error_switches_the_preview_off_and_never_reaches_the_sweep(self) -> None:
        gui = _Gui(fail="imshow")
        preview, said = _preview(gui=gui)
        preview.start()
        self.assertTrue(_until(lambda: not preview.running))
        self.assertEqual(preview.off, "RuntimeError: the display went away")
        self.assertEqual(said, ["preview off after RuntimeError: the display went away; the sweep goes on"])
        # The sweep's side still returns at once and raises nothing.
        preview(_MOVING, {"index": 1})
        preview.observe(_board_view(), None, _K, None)
        preview.close()
        self.assertEqual(preview._drain(), [])
        self.assertEqual([name for name, _ in gui.calls][-2:], ["destroyWindow", "waitKey"])

    def test_a_window_that_cannot_open_switches_the_preview_off(self) -> None:
        preview, said = _preview(gui=_Gui(fail="namedWindow"))
        preview.start()
        self.assertTrue(_until(lambda: not preview.running))
        self.assertEqual(preview.off, "RuntimeError: the display went away")
        self.assertEqual(said, ["preview off after RuntimeError: the display went away; the sweep goes on"])

    def test_esc_and_closing_the_window_close_the_preview_only(self) -> None:
        for gui, reason in ((_Gui(keys=(27,)), "ESC"), (_Gui(visible=0.0), "its window was closed")):
            with self.subTest(reason):
                preview, said = _preview(gui=gui, live_hz=0)
                preview.start()
                self.assertTrue(_until(lambda: not preview.running))
                self.assertEqual(preview.off, reason)
                self.assertEqual(said, [f"preview closed ({reason}). The sweep goes on; the pendant stops the robot"])

    def test_ctrl_c_in_the_window_is_said_once_and_stops_nothing(self) -> None:
        preview, said = _preview(gui=_Gui(keys=(3, -1, 3)), live_hz=0)
        self.addCleanup(preview.close)
        preview.start()
        time.sleep(0.3)
        self.assertTrue(preview.running)
        self.assertEqual(len(said), 1)
        self.assertIn("Ctrl+C went to the preview window, not to the console", said[0])
        self.assertIn("The pendant stops the robot", said[0])


# ---------------------------------------------------------------------------------------------------------------------
# The noun: built at the build, started with the sweep, closed before the camera is given back
# ---------------------------------------------------------------------------------------------------------------------


class _Device:
    """The device behind a real ``Camera`` owner: writes its opens and releases to a shared log, serves frames."""

    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.grabs: list[int] = []

    def open(self) -> None:
        self.log.append("camera.open")

    def release(self) -> None:
        self.log.append("camera.release")

    def grab(self) -> Any:
        self.grabs.append(threading.get_ident())
        return SimpleNamespace(color=_marker_view())

    def get_intrinsics(self) -> np.ndarray:
        return _K

    def get_distortion(self) -> None:
        return None


class _Arm(DummyRobotArm):
    def __init__(self, *, refuse: bool = False) -> None:
        super().__init__()
        self.refuse = refuse

    def connect(self) -> None:
        if self.refuse:
            raise RuntimeError("no controller answers at this address")
        super().connect()


class _RecordingPreview:
    """Stands in for ``SweepPreview`` in the noun and writes what is done to it to the shared log."""

    log: list[str] = []
    built: list["_RecordingPreview"] = []

    def __init__(self, handle: Any, **keywords: Any) -> None:
        self.handle = handle
        self.keywords = keywords
        self.title = keywords.get("title", "")
        self.events: list[tuple[str, dict]] = []
        self.log.append("preview.built")
        self.built.append(self)

    def start(self) -> None:
        self.log.append("preview.start")

    def close(self, timeout_s: float = 2.0) -> None:
        self.log.append("preview.close")

    def observe(self, *_args: Any) -> None:
        self.log.append("preview.observe")

    def __call__(self, event_type: str, data: dict) -> None:
        self.events.append((event_type, data))


def _eth_result() -> CalibrationResult:
    transform = Transform(translation_mm=np.array([400.0, -25.0, 812.5]), quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]),
                          from_frame=Frame.CAMERA, to_frame=Frame.BASE)
    extrinsics = Extrinsics(transform=transform, rmse_mm=0.8, max_error_mm=1.5, num_samples=20,
                            captured_at=dt.datetime(2026, 9, 23, tzinfo=dt.timezone.utc), rig_id="rig_0")
    return CalibrationResult(T_cam_to_base=transform.to_matrix(), rmse_mm=0.8, max_error_mm=1.5, num_samples=20,
                             dataset_path="dataset.json", extrinsics=extrinsics, transform=transform,
                             mode=MountingMode.EYE_TO_HAND)


class _Noun(unittest.TestCase):

    def setUp(self) -> None:
        self.log: list[str] = []
        _RecordingPreview.log = self.log
        _RecordingPreview.built = []
        self.addCleanup(lambda: lock_path_for(_KEY).unlink(missing_ok=True))

    def _noun(self, *, preview: Any = True, lock_key: str | None = None, refuse: bool = False,
              on_event: Any = None) -> HandEyeCalibration:
        self.device = _Device(self.log)
        self.camera = Camera.from_rig(_rgbd_rig("overhead"), streamer=self.device)
        self.addCleanup(self.camera.release)
        self.arm = _Arm(refuse=refuse)
        out = self.enterContext(tempfile.TemporaryDirectory())
        options = SweepOptions(out_dir=out) if preview is None else SweepOptions(out_dir=out, preview=preview)
        return HandEyeCalibration.from_parts(
            robot=Robot.from_parts(arm=self.arm, gripper=None, lock_key=lock_key), camera=self.camera,
            robot_config=RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": _IP}}), mode="eye_to_hand",
            options=options, on_event=on_event)

    def _recorded(self) -> Any:
        return self.enterContext(_patched(SweepPreview=_RecordingPreview, preview_unavailable=lambda: None))


def _patched(**names: Any) -> Any:
    from contextlib import ExitStack

    stack = ExitStack()
    for name, value in names.items():
        stack.enter_context(patch.object(preview_module, name, value))
    return stack


class TheNounOpensThePreviewWithTheSweepTests(_Noun):

    def test_the_library_opens_no_window_unasked(self) -> None:
        explode = MagicMock(side_effect=AssertionError("a preview was built"))
        with _patched(SweepPreview=explode, preview_unavailable=lambda: None):
            report = self._noun(preview=None).run(dry_run=True)
        self.assertIs(report.outcome, CalibrationOutcome.DRY_RUN)
        assert report.build is not None
        self.assertEqual(report.build.preview, "")
        self.assertNotIn("preview", report.build.render())

    def test_a_dry_run_builds_it_starts_nothing_grabs_nothing_and_closes_it_before_the_release(self) -> None:
        self._recorded()
        report = self._noun().run(dry_run=True)
        self.assertIs(report.outcome, CalibrationOutcome.DRY_RUN)
        self.assertEqual(self.log, ["camera.open", "preview.built", "preview.close", "camera.release"])
        self.assertEqual(self.device.grabs, [])
        assert report.build is not None
        self.assertIn("  preview  window 'willy hand-eye sweep: overhead' opens with the sweep; closing it closes "
                      "the window only, the pendant stops the robot", report.build.render())

    def test_a_real_preview_in_a_dry_run_opens_no_window(self) -> None:
        gui = _Gui()
        with _patched(preview_unavailable=lambda: None, _highgui=lambda: gui):
            report = self._noun().run(dry_run=True)
        self.assertIs(report.outcome, CalibrationOutcome.DRY_RUN)
        self.assertEqual((gui.calls, self.device.grabs), ([], []))

    def test_every_way_out_of_the_sweep_closes_the_preview_before_the_camera_is_given_back(self) -> None:
        self._recorded()
        ways: dict[CalibrationOutcome, tuple[dict[str, Any], MagicMock]] = {
            CalibrationOutcome.WRITTEN: (dict(), MagicMock(return_value=_eth_result())),
            CalibrationOutcome.SWEEP_FAILED: (dict(), MagicMock(side_effect=RuntimeError("the board fell off"))),
            CalibrationOutcome.CELL_BUSY: (dict(lock_key=_KEY), MagicMock(side_effect=AssertionError("swept"))),
            CalibrationOutcome.CONNECT_FAILED: (dict(refuse=True), MagicMock(side_effect=AssertionError("swept"))),
        }
        for outcome, (noun, run_auto) in ways.items():
            with self.subTest(outcome.value):
                self.log.clear()
                calibration = self._noun(**noun)
                with patch.object(CalibrationRoutine, "run_auto", run_auto):
                    if outcome is CalibrationOutcome.CELL_BUSY:
                        with CellLock(_KEY, owner="operator console"):
                            report = calibration.run()
                    else:
                        report = calibration.run()
                self.assertIs(report.outcome, outcome, report.render())
                connected = outcome in (CalibrationOutcome.WRITTEN, CalibrationOutcome.SWEEP_FAILED)
                self.assertEqual(self.log, ["camera.open", "preview.built", *(["preview.start"] if connected else []),
                                            "preview.close", "camera.release"])

    def test_the_routine_hears_the_events_through_the_preview_and_the_callers_listener_alike(self) -> None:
        self._recorded()
        heard: list[tuple[str, dict]] = []
        calibration = self._noun(on_event=lambda kind, data: heard.append((kind, data)))
        build, parts = calibration._build(calibration._stage())  # noqa: SLF001
        assert parts is not None
        self.addCleanup(parts.give_back)
        self.assertTrue(build.ok, build.refusal)
        (preview,) = _RecordingPreview.built
        source: Any = parts.routine._marker_source  # noqa: SLF001
        self.assertEqual(source.on_observation, preview.observe)
        self.assertIsNot(preview.keywords["estimator"], source.estimator, "the preview poses with its own estimator")
        self.assertIs(type(preview.keywords["estimator"]), type(source.estimator))
        self.assertIs(preview.keywords["describe"], render_sweep_event)
        self.assertEqual(preview.keywords["axis_mm"], 50.0)
        parts.routine._emit(_MOVING, {"index": 1, "total": 2})  # noqa: SLF001
        self.assertEqual(heard, [(_MOVING, {"index": 1, "total": 2})])
        self.assertEqual(preview.events, heard)

    def test_a_sweep_with_the_real_preview_draws_on_one_thread_and_grabs_only_through_the_owner(self) -> None:
        gui = _Gui()
        gui.log = self.log

        def sweep(routine: CalibrationRoutine, *_args: Any, **_keywords: Any) -> CalibrationResult:
            routine._emit(_MOVING, {"index": 1, "total": 1, "label": "look_0", "accepted": 0})  # noqa: SLF001
            source: Any = routine._marker_source  # noqa: SLF001
            pose = source()  # the judged grab, as the routine makes it
            assert pose is not None
            routine._emit(_ACCEPTED, {"index": 1, "total": 1, "label": "look_0", "accepted": 1})  # noqa: SLF001
            _until(lambda: len(gui.shown) >= 3 and len(set(self.device.grabs)) == 2)
            return _eth_result()

        with _patched(preview_unavailable=lambda: None, _highgui=lambda: gui), \
                patch.object(CalibrationRoutine, "run_auto", autospec=True, side_effect=sweep):
            report = self._noun().run()
        self.assertIs(report.outcome, CalibrationOutcome.WRITTEN, report.render())
        self.assertEqual(len(gui.threads()), 1)
        self.assertNotIn(threading.get_ident(), gui.threads())
        # The judged grabs (three warm-ups and one) on the sweep's thread, the live ones on the window's.
        self.assertEqual(self.device.grabs.count(threading.get_ident()), 4)
        self.assertEqual(set(self.device.grabs) - {threading.get_ident()}, gui.threads())
        self.assertLess(self.log.index("gui.destroyWindow"), self.log.index("camera.release"))
        self.assertTrue(any(tuple(int(v) for v in image[2, 10]) == (50, 140, 50) for image in gui.shown),
                        "the judged frame was not pinned as counted")


class TheSettingTests(_Noun):

    def _line(self, setting: Any, *, terminal: bool = False, environ: dict[str, str] | None = None) -> str:
        self._recorded()
        stdout = SimpleNamespace(isatty=lambda: terminal, write=lambda _text: None, flush=lambda: None)
        with patch("sys.stdout", stdout), patch.dict(os.environ, environ or {}, clear=False), \
                _patched(preview_unavailable=lambda: preview_unavailable(
                    environ=environ or {}, platform="win32", build_information=_WIN)):
            report = self._noun(preview=setting).run(dry_run=True)
        assert report.build is not None
        return report.build.preview

    def test_auto_opens_only_on_a_terminal_and_says_nothing_otherwise(self) -> None:
        self.assertEqual(self._line("auto", terminal=False), "")
        self.assertEqual(_RecordingPreview.built, [])
        self.assertTrue(self._line("auto", terminal=True).startswith("window "))
        self.assertEqual(len(_RecordingPreview.built), 1)

    def test_asked_for_by_name_it_says_why_it_cannot_open(self) -> None:
        self.assertEqual(self._line(True, environ={"WILLY_NO_PREVIEW": "1"}), "off: WILLY_NO_PREVIEW is set")
        self.assertEqual(self._line("auto", terminal=True, environ={"WILLY_NO_PREVIEW": "1"}), "")
        self.assertTrue(self._line(True, terminal=False).startswith("window "), "a name needs no terminal")

    def test_a_preview_that_cannot_be_built_is_left_out_and_the_build_stands(self) -> None:
        broken = MagicMock(side_effect=RuntimeError("no font"))
        with _patched(SweepPreview=broken, preview_unavailable=lambda: None):
            report = self._noun(preview=True).run(dry_run=True)
        self.assertIs(report.outcome, CalibrationOutcome.DRY_RUN)
        assert report.build is not None
        self.assertEqual(report.build.preview, "off: RuntimeError: no font")

    def test_a_setting_that_is_not_one_is_refused_at_the_factory(self) -> None:
        with self.assertRaisesRegex(ValueError, "True, False or 'auto'"):
            self._noun(preview="yes")


class TheCliAsksForThePreviewTests(unittest.TestCase):

    def _options(self, *argv: str) -> SweepOptions:
        seen: list[SweepOptions] = []
        real = HandEyeCalibration.from_config

        def spy(*args: Any, **keywords: Any) -> HandEyeCalibration:
            seen.append(keywords["options"])
            return real(*args, **keywords)

        app = SimpleNamespace(robot=RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": _IP}}),
                              camera=SimpleNamespace(cameras=SimpleNamespace(rigs=[_rgbd_rig("overhead")]),
                                                     hand_eye=None))
        with patch.object(calibrate, "_load", return_value=app), \
                patch.object(calibrate.HandEyeCalibration, "from_config", side_effect=spy), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(calibrate.main(["--rig", "overhead", "--check", *argv]), calibrate._EXIT_OK)
        return seen[0]

    def test_the_flag_and_its_negation_and_the_default(self) -> None:
        self.assertEqual(self._options().preview, "auto")
        self.assertIs(self._options("--preview").preview, True)
        self.assertIs(self._options("--no-preview").preview, False)


class TheDocsAndExamplesTests(unittest.TestCase):

    def test_the_setup_page_says_what_the_window_is_and_is_not(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "docs" / "calibration-setup.md").read_text(encoding="utf-8")
        for said in ("--preview", "--no-preview", "WILLY_NO_PREVIEW", "the pendant stops the robot", "Ctrl+C"):
            with self.subTest(said):
                self.assertIn(said, text)

    def test_examples_07_to_10_ask_for_it_where_a_window_can_show(self) -> None:
        root = Path(__file__).resolve().parents[1] / "examples" / "real_robot"
        for number in ("07", "08", "09", "10"):
            (path,) = sorted(root.glob(f"{number}_*.py"))
            with self.subTest(path.name):
                self.assertIn('preview="auto"', path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------------------------------
# A real window, on a desk that can show one, when asked
# ---------------------------------------------------------------------------------------------------------------------


@unittest.skipUnless(os.environ.get("WILLY_PREVIEW_SMOKE") == "1" and preview_unavailable() is None,
                     "opens a real window: set WILLY_PREVIEW_SMOKE=1 on a desktop that can show one")
class ARealWindowTests(unittest.TestCase):

    def test_the_window_opens_draws_and_closes_from_its_own_thread(self) -> None:
        said: list[str] = []
        handle = _Handle(_board_view())
        preview = SweepPreview(handle, title="willy preview smoke test", say=said.append, pin_s=0.5,
                               estimator=CharucoPoseEstimator(squares_x=7, squares_y=5, square_length_mm=30.0,
                                                              marker_length_mm=22.0),
                               describe=render_sweep_event, axis_mm=60.0)
        self.addCleanup(preview.close)
        preview.start()
        for index in range(1, 4):
            preview(_MOVING, {"index": index, "total": 3, "label": f"look_{index}", "accepted": index - 1,
                              "min_samples": 2})
            time.sleep(0.4)
            frame = _board_view()
            estimator = CharucoPoseEstimator(squares_x=7, squares_y=5, square_length_mm=30.0, marker_length_mm=22.0)
            observation = estimator.observe(frame, _K, np.zeros(5))
            preview.observe(frame, observation, _K, None)
            preview(_ACCEPTED, {"index": index, "total": 3, "label": f"look_{index}", "accepted": index,
                                "min_samples": 2, "detail": observation.summary()})
            time.sleep(0.6)
        self.assertTrue(preview.running, preview.off)
        started = time.perf_counter()
        preview.close()
        self.assertLess(time.perf_counter() - started, 1.0)
        self.assertFalse(preview.running)
        self.assertEqual((preview.off, said), ("", []))
        self.assertGreater(preview.frames_shown, 3)
        self.assertGreater(preview.live_grabs, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
