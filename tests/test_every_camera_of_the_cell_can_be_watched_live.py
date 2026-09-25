"""Every camera of the cell, watched live: one window each, what was just found pinned on it, and nothing else touched.

The owner, 2026-09-24: while poses are taught and during picks, an optional switch shows live, one window per camera,
what every camera of the cell sees, fixed cameras too, and where the software has just detected something, that too:
the located masks and labels, and for a pick the grasp overlay. ``src/camera/live_view.py`` is that view. What is held
here, over a HighGUI double and cameras that count their looks:

* one daemon thread makes every HighGUI call for every window, and reads each camera through its own owner at about
  four frames a second; every window says its rig, whether it rides on the wrist, how old its frame is, and the last
  thing that happened on it;
* a desk that cannot show a window gets one line and a view that does nothing; asked for none, nothing at all;
* ``show_located`` pins the masks, labels and centres on that camera's window, ``show_image`` pins an overlay (a pick's
  grasp render) on the window named, for a few seconds, and then the live frame returns;
* a camera on the wrist holds the colour image its locate was made on, handed over with the ``Located``, for the pin,
  and says so, while the arm moves on and newer frames arrive, however soon after the shutter it is handed over; it
  never holds a frame the view took itself; with no image handed over the masks go on a blank and it says so; a fixed
  camera's masks go on over its live frame; a locate that found nothing holds nothing and is only said (review
  finding F3, 2026-09-25);
* closing any window, or ESC, closes every window and nothing else: a campaign runs on to its end;
* the first window is the guide of a person guiding the arm, red outside a boundary, and hands back the keys typed;
* a ``Locator`` given the view shows each ``Located`` it produces on its camera's window, a ``PickRun`` given one shows
  every camera the cell's build opened and pins each attempt's grasp overlay on the primary's, and a ``Cell`` names
  the cameras its build opened;
* the examples that pick ask for it with ``SHOW_CAMERAS``.

That a look never changes what the robot measures is ``test_a_camera_window_never_feeds_what_the_robot_measures.py``.
Teaching shows the one camera its poses are taught for, not every camera (the owner, 2026-09-24 evening: a pose belongs
to one camera): ``test_a_pose_is_taught_by_hand.py``, example 11 included.
"""

from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import cv2
import numpy as np

from src.calibration import preview as preview_module
from src.calibration.preview import MAX_LIVE_HZ
from src.camera import live_view as live_view_module
from src.camera.live_view import LiveView, draw_located
from src.robot.execution.cell import Cell, CellNotBuilt
from src.robot.execution.pick_run import PickRun, Recording
from src.robot.perception.locator import Located, LocatedObject, Locator
from tests.test_calibration_sweep_preview import _Gui, _until
from tests.test_locator import _backend, _Owner

_ROOT = Path(__file__).resolve().parents[1]
_RED = (0, 0, 230)


class _Desk(_Gui):
    """A HighGUI double that keeps what each window showed, and closes one window once it showed ``after`` images."""

    def __init__(self, *, closes: str = "", after: int = 1, **keywords: Any) -> None:
        super().__init__(**keywords)
        self.windows: dict[str, list[np.ndarray]] = {}
        self.opened: list[str] = []
        self.destroyed: list[str] = []
        self.closes = closes
        self.after = after

    def namedWindow(self, title: str, flags: int) -> None:  # noqa: N802 (cv2's name)
        super().namedWindow(title, flags)
        self.opened.append(title)
        self.windows.setdefault(title, [])

    def imshow(self, title: str, image: np.ndarray) -> None:
        super().imshow(title, image)
        self.windows.setdefault(title, []).append(image)

    def getWindowProperty(self, title: str, prop: int) -> float:  # noqa: N802 (cv2's name)
        self._call("getWindowProperty")
        if title == self.closes and len(self.windows.get(title, [])) >= self.after:
            return 0.0
        return self.visible

    def destroyWindow(self, title: str) -> None:  # noqa: N802 (cv2's name)
        super().destroyWindow(title)
        self.destroyed.append(title)

    def moveWindow(self, title: str, x: int, y: int) -> None:  # noqa: N802 (cv2's name)
        self._call("moveWindow")

    def seen(self, rig_id: str) -> list[np.ndarray]:
        return list(self.windows.get(f"willy camera: {rig_id}", []))


class _Eye:
    """An open camera as the view meets it: a rig id, the rig it was configured as, and a display path that counts
    its looks and the threads they came from. Every frame is one grey."""

    def __init__(self, rig_id: str, grey: int = 100, *, rig: Any = None, size: tuple[int, int] = (48, 64)) -> None:
        self.rig_id = rig_id
        self.rig = rig
        self.grey = grey
        self.size = size
        self.threads: list[int] = []

    @property
    def looks(self) -> int:
        return len(self.threads)

    def peek(self) -> Any:
        self.threads.append(threading.get_ident())
        return SimpleNamespace(color=np.full((*self.size, 3), self.grey, dtype=np.uint8), captured_at_s=time.time())


def _here(unavailable: str | None = None) -> Any:
    """While entered: whether a window can show on this desk (``None``: it can)."""
    return mock.patch.object(preview_module, "preview_unavailable", lambda: unavailable)


def _frame_of(image: np.ndarray, rows: int = 48) -> np.ndarray:
    """The camera's frame inside a composed window image: the rows under the status strip."""
    return image[-rows:]


def _located(camera: str, *objects: LocatedObject) -> Located:
    return Located(camera=camera, captured_at_s=time.time(), mounting="eye_to_hand", tool_to_base_mm=None,
                   objects=tuple(objects))


def _cube(label: str = "red cube", score: float = 0.91, *, rows: slice = slice(20, 60),
          cols: slice = slice(40, 100), shape: tuple[int, int] = (96, 128)) -> LocatedObject:
    mask = np.zeros(shape, dtype=bool)
    mask[rows, cols] = True
    return LocatedObject(label=label, score=score, box_px=(cols.start, rows.start, cols.stop, rows.stop), mask=mask,
                         points_base_mm=np.zeros((0, 3)), centre_mm=(412.0, -96.0, 31.0))


def _png(bgr: tuple[int, int, int], size: tuple[int, int] = (48, 64)) -> bytes:
    ok, encoded = cv2.imencode(".png", np.full((*size, 3), bgr, dtype=np.uint8))
    assert ok
    return encoded.tobytes()


#: A rig that declares the body it rides on (the wrist), and one calibrated eye to hand (fixed).
_ON_THE_WRIST = SimpleNamespace(body="wrist_camera", extrinsics=None)
_FIXED = SimpleNamespace(body=None, extrinsics=SimpleNamespace(mounting_mode="eye_to_hand"))
_TOOL = tuple(tuple(float(row == column) for column in range(4)) for row in range(4))


def _on_the_wrist(captured_at_s: float, *objects: LocatedObject) -> Located:
    """What a locator on the wrist hands over: ``eye_in_hand``, with the tool pose at its shutter."""
    return Located(camera="wrist", captured_at_s=captured_at_s, mounting="eye_in_hand", tool_to_base_mm=_TOOL,
                   objects=tuple(objects))


def _grey(image: np.ndarray, rows: int = 48) -> int:
    """The grey of the camera frame a composed window image shows, read where no mask or label is drawn."""
    return int(_frame_of(image, rows)[-1, 5, 0])


def _tinted(image: np.ndarray, rows: int = 48, at: tuple[int, int] = (15, 26)) -> bool:
    """Whether a located object's mask is drawn on the frame: its pixel ``at`` is not the frame's own grey."""
    return tuple(int(v) for v in _frame_of(image, rows)[at]) != (_grey(image, rows),) * 3


class _Case(unittest.TestCase):

    def view(self, eyes: list[_Eye], desk: _Desk | None = None, **options: Any) -> tuple[LiveView, _Desk, list[str]]:
        said: list[str] = []
        desk = _Desk() if desk is None else desk
        self.enterContext(_here())
        view = LiveView(eyes, gui=desk, say=said.append, **options)
        self.addCleanup(view.close)
        return view, desk, said


# ---------------------------------------------------------------------------------------------------------------------
# One thread, a window per camera
# ---------------------------------------------------------------------------------------------------------------------


class OneThreadDrawsEveryWindowTests(_Case):

    def test_every_camera_gets_a_window_of_its_own_and_one_thread_makes_every_gui_call(self) -> None:
        eyes = [_Eye("wrist", 40), _Eye("overhead", 120), _Eye("side", 200)]
        view, desk, said = self.view(eyes)
        with view:
            self.assertTrue(_until(lambda: all(len(desk.seen(eye.rig_id)) >= 2 for eye in eyes)))
        main = threading.get_ident()
        self.assertEqual(desk.opened, ["willy camera: wrist", "willy camera: overhead", "willy camera: side"])
        self.assertEqual(len(desk.threads()), 1, "HighGUI was called from more than one thread")
        self.assertNotIn(main, desk.threads())
        for eye in eyes:
            with self.subTest(eye.rig_id):
                self.assertEqual(set(eye.threads), desk.threads(), "a camera was read off the window's thread")
                last = _frame_of(desk.seen(eye.rig_id)[-1])
                self.assertEqual(tuple(int(v) for v in last[-1, 5]), (eye.grey,) * 3, "another camera's frame")
        self.assertEqual(sorted(desk.destroyed), sorted(desk.opened))
        self.assertEqual((said, view.off), ([], ""))
        self.assertFalse(view.running)

    def test_live_frames_come_at_about_four_a_second_per_camera(self) -> None:
        eyes = [_Eye("wrist"), _Eye("overhead")]
        view, _, _ = self.view(eyes)
        started = time.monotonic()
        with view:
            time.sleep(1.2)
        elapsed = time.monotonic() - started
        for eye in eyes:
            with self.subTest(eye.rig_id):
                self.assertGreaterEqual(eye.looks, 3)
                self.assertLessEqual(eye.looks, int(elapsed * MAX_LIVE_HZ) + 1)
        time.sleep(0.3)
        self.assertEqual([eye.looks for eye in eyes], [eye.looks for eye in eyes], "a look after the close")

    def test_a_camera_watched_later_gets_its_window_and_one_watched_twice_gets_one(self) -> None:
        wrist, side = _Eye("wrist"), _Eye("side", 180)
        view, desk, _ = self.view([wrist])
        with view:
            view.watch(side, wrist)
            self.assertTrue(_until(lambda: len(desk.seen("side")) >= 1))
        self.assertEqual(desk.opened, ["willy camera: wrist", "willy camera: side"])
        self.assertEqual(view.rig_ids, ("wrist", "side"))


class EachWindowSaysWhatItShowsTests(unittest.TestCase):

    def test_the_status_says_the_rig_where_it_rides_how_old_the_frame_is_and_what_happened_last(self) -> None:
        window = live_view_module._Window(rig_id="overhead", title="t", source=None, mounting="fixed camera")
        self.assertEqual(window.status(100.0), ["overhead, fixed camera: no live frame yet", ""])
        window.live = np.zeros((4, 4, 3), dtype=np.uint8)
        window.live_at_s = 99.6
        window.event = ("located 1: red cube 0.91", 97.0)
        self.assertEqual(window.status(100.0), ["overhead, fixed camera: live, frame 0.4 s old",
                                                "last: located 1: red cube 0.91, 3 s ago"])
        window.live_error = "CameraNotOpen: rig 'overhead' is not open"
        self.assertIn("now CameraNotOpen", window.status(100.0)[0])

    def test_where_a_camera_rides_is_read_off_its_rig(self) -> None:
        mounted = live_view_module._mounting
        eye_in_hand = SimpleNamespace(mounting_mode="eye_in_hand")
        eye_to_hand = SimpleNamespace(mounting_mode="eye_to_hand")
        self.assertEqual(mounted(SimpleNamespace(rig=SimpleNamespace(body=object(), extrinsics=None))),
                         "wrist camera")
        self.assertEqual(mounted(SimpleNamespace(rig=SimpleNamespace(body=None, extrinsics=eye_in_hand))),
                         "wrist camera")
        self.assertEqual(mounted(SimpleNamespace(rig=SimpleNamespace(body=None, extrinsics=eye_to_hand))),
                         "fixed camera")
        self.assertEqual(mounted(SimpleNamespace(rig=SimpleNamespace(body=None, extrinsics=None))),
                         "mounting not declared")
        handle = SimpleNamespace(camera=SimpleNamespace(rig=SimpleNamespace(body=None, extrinsics=eye_to_hand)))
        self.assertEqual(mounted(handle), "fixed camera", "a handle is read through its owner")


# ---------------------------------------------------------------------------------------------------------------------
# Where no window can show
# ---------------------------------------------------------------------------------------------------------------------


class WhereNoWindowCanShowTests(unittest.TestCase):

    def _everything(self, view: LiveView, eye: _Eye) -> None:
        view.watch(_Eye("side"))
        view.show_located(_located("wrist", _cube()))
        view.show_image("wrist", _png((0, 0, 255)), "run 0: succeeded", ok=True)
        view.note("run 0: succeeded")
        view.guide(["MOVE THE ARM"], "outside")
        self.assertIsNone(view.poll_key())

    def test_a_desk_with_no_display_gets_one_line_and_a_view_that_does_nothing(self) -> None:
        eye, desk, said = _Eye("wrist"), _Desk(), []
        with _here("no display: DISPLAY and WAYLAND_DISPLAY are unset"):
            with LiveView([eye], gui=desk, say=said.append) as view:
                self._everything(view, eye)
                time.sleep(0.3)
        self.assertEqual(said, ["No camera windows: no display: DISPLAY and WAYLAND_DISPLAY are unset. Everything "
                                "else runs as it would without them."])
        self.assertEqual((desk.calls, eye.looks), ([], 0))
        self.assertFalse(view.running)
        self.assertTrue(view.off.startswith("no window can show"), view.off)

    def test_asked_for_none_it_says_nothing_and_opens_nothing(self) -> None:
        eye, desk, said = _Eye("wrist"), _Desk(), []
        with _here(), LiveView([eye], show=False, gui=desk, say=said.append) as view:
            self._everything(view, eye)
            time.sleep(0.3)
        self.assertEqual((said, desk.calls, eye.looks, view.off), ([], [], 0, ""))


# ---------------------------------------------------------------------------------------------------------------------
# What was just found, pinned
# ---------------------------------------------------------------------------------------------------------------------


class WhatWasJustFoundIsPinnedTests(_Case):

    def test_the_masks_labels_and_centre_are_drawn_where_the_camera_saw_them(self) -> None:
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        drawn = draw_located(image, _located("overhead", _cube()))
        self.assertEqual(int(image.max()), 0, "the image handed in was drawn on")
        tint = tuple(int(v) for v in drawn[15, 26])
        self.assertNotEqual(tint, (0, 0, 0), "no mask where the camera saw the cube (a mask of 96x128, shown 48x64)")
        self.assertEqual(tint, tuple(int(v) for v in drawn[25, 45]), "one tint over the whole mask")
        self.assertEqual(tuple(int(v) for v in drawn[40, 5]), (0, 0, 0), "a mask where the camera saw nothing")
        centre = drawn[17:23, 31:38].reshape(-1, 3)
        self.assertTrue(any(tuple(int(v) for v in pixel) != tint for pixel in centre), "no mark at the mask's centre")

    def test_show_located_pins_the_masks_on_that_cameras_window_and_says_what_was_located(self) -> None:
        wrist, overhead = _Eye("wrist", 0), _Eye("overhead", 0)
        view, desk, _ = self.view([wrist, overhead], pin_s=5.0)
        tint = tuple(int(v) for v in draw_located(np.zeros((48, 64, 3), np.uint8), _located("x", _cube()))[15, 26])
        with view:
            self.assertTrue(_until(lambda: len(desk.seen("overhead")) >= 1))
            view.show_located(_located("overhead", _cube(), _cube("blue plate", 0.84, rows=slice(70, 90))))
            self.assertTrue(_until(lambda: any(tuple(int(v) for v in _frame_of(image)[15, 26]) == tint
                                               for image in desk.seen("overhead"))))
        self.assertFalse(any(tuple(int(v) for v in _frame_of(image)[15, 26]) == tint for image in desk.seen("wrist")),
                         "the masks of one camera on another camera's window")
        text, _ = view._windows["overhead"].event
        self.assertEqual(text, "located 2: red cube 0.91, blue plate 0.84")
        self.assertIsNone(view._windows["wrist"].event)

    def test_located_nothing_is_said_on_the_window_and_masks_of_an_unwatched_camera_go_nowhere(self) -> None:
        view, desk, _ = self.view([_Eye("wrist")])
        with view:
            view.show_located(_located("wrist"))
            view.show_located(_located("elsewhere", _cube()))
            self.assertTrue(_until(lambda: view._windows.get("wrist") is not None
                                   and view._windows["wrist"].event is not None))
            time.sleep(0.1)
        self.assertRegex(view._windows["wrist"].event[0], r"^nothing located at \d\d:\d\d:\d\d$")
        self.assertEqual(desk.opened, ["willy camera: wrist"])

    def test_an_overlay_is_pinned_on_the_window_named_for_a_few_seconds_then_the_live_frame_returns(self) -> None:
        wrist, overhead = _Eye("wrist", 90), _Eye("overhead", 90)
        view, desk, _ = self.view([wrist, overhead], pin_s=0.4)

        def red(image: np.ndarray) -> bool:
            return tuple(int(v) for v in _frame_of(image)[-1, 5]) == (0, 0, 255)

        with view:
            self.assertTrue(_until(lambda: len(desk.seen("overhead")) >= 1))
            view.show_image("overhead", _png((0, 0, 255)), "run 0: succeeded", ok=True)
            self.assertTrue(_until(lambda: any(red(image) for image in desk.seen("overhead"))))
            pinned = next(image for image in desk.seen("overhead") if red(image))
            self.assertEqual(tuple(int(v) for v in pinned[2, 10]), (50, 140, 50), "a success is not on green")
            count = len(desk.seen("overhead"))
            self.assertTrue(_until(lambda: len(desk.seen("overhead")) > count
                                   and not red(desk.seen("overhead")[-1]), timeout_s=3.0))
        self.assertFalse(any(red(image) for image in desk.seen("wrist")))
        self.assertEqual(view._windows["overhead"].event[0], "run 0: succeeded")

    def test_an_overlay_for_no_window_goes_on_the_first_and_a_note_goes_on_every_window(self) -> None:
        view, desk, _ = self.view([_Eye("wrist", 90), _Eye("overhead", 90)], pin_s=5.0)
        with view:
            view.show_image(None, _png((255, 0, 0)), "the grasp")
            view.note("run 3: succeeded")
            self.assertTrue(_until(lambda: any(tuple(int(v) for v in _frame_of(image)[-1, 5]) == (255, 0, 0)
                                               for image in desk.seen("wrist"))))
            self.assertTrue(_until(lambda: all(window.event is not None and window.event[0] == "run 3: succeeded"
                                               for window in view._windows.values())))


class AWristCamerasMasksStayOnTheImageTheyWereLocatedOnTests(_Case):
    """Review finding F3, 2026-09-25: a camera on the wrist moves with the arm, and example 13 picks and places right
    after each locate, so masks drawn over its newest live frame land where the objects no longer are. Its window holds
    the colour image the locate was made on, handed over with the ``Located``, for the pin, says so, and then goes live
    again. Never a frame the view took itself: no look falls inside a locate's grabs, so for a locate that returns
    soon after its shutter the view's nearest frame is one from the arm's way to its look. A fixed camera's scene
    stays put while the arm moves, and its masks go on over the live frame as before."""

    def test_the_masks_go_on_exactly_the_image_handed_over_while_newer_frames_arrive_then_the_live_frame_returns(
            self) -> None:
        wrist = _Eye("wrist", 50, rig=_ON_THE_WRIST)
        view, desk, _ = self.view([wrist], pin_s=1.0)
        handed = np.zeros((48, 64, 3), dtype=np.uint8)
        handed[:, :, 1] = np.arange(64, dtype=np.uint8) * 3  # the image the locate segmented: a ramp, no live frame
        located = _on_the_wrist(time.time(), _cube())
        expected = draw_located(handed.copy(), located)
        with view:
            self.assertTrue(_until(lambda: len(desk.seen("wrist")) >= 1))
            view.show_located(located, image=handed)
            handed[:] = 255  # the program's array, changed after the hand-over: the view keeps its own copy
            wrist.grey = 200  # the arm moves on to pick: every frame from now on shows somewhere else
            self.assertTrue(_until(lambda: any(_tinted(image) for image in desk.seen("wrist"))))
            moved = wrist.looks
            self.assertTrue(_until(lambda: wrist.looks >= moved + 2), "no newer live frame arrived during the pin")
            during = desk.seen("wrist")[-1]
            held = view._windows["wrist"].status(time.time())[0]
            self.assertTrue(_until(lambda: not _tinted(desk.seen("wrist")[-1])
                                   and _grey(desk.seen("wrist")[-1]) == 200, timeout_s=4.0),
                            "the live frame did not return after the pin")
        self.assertTrue(np.array_equal(_frame_of(during), expected), "the pin left the image the locate was made on")
        for image in desk.seen("wrist"):
            if _tinted(image):
                self.assertTrue(np.array_equal(_frame_of(image), expected), "masks painted on another image")
        self.assertRegex(held, r"^wrist, wrist camera: as located at \d\d:\d\d:\d\d; the arm may have moved since$")
        self.assertIsNone(view._windows["wrist"].pinned)

    def test_a_locate_handed_over_right_after_its_shutter_shows_its_own_image_never_a_frame_the_view_took(
            self) -> None:
        wrist = _Eye("wrist", 30, rig=_ON_THE_WRIST)  # the frames the view takes while the arm moves to its look
        view, desk, _ = self.view([wrist], pin_s=5.0)
        with view:
            self.assertTrue(_until(lambda: wrist.looks >= 2))
            located = _on_the_wrist(time.time(), _cube())  # the shutter, and detection back within a look's period
            view.show_located(located, image=np.full((48, 64, 3), 120, dtype=np.uint8))
            self.assertTrue(_until(lambda: any(_tinted(image) for image in desk.seen("wrist"))))
            time.sleep(0.6)
        self.assertEqual({_grey(image) for image in desk.seen("wrist") if _tinted(image)}, {120},
                         "a frame the view took before the shutter was held as the image of the locate")

    def test_with_no_image_handed_over_the_masks_go_on_a_blank_and_the_window_says_so(self) -> None:
        wrist = _Eye("wrist", 50, rig=_ON_THE_WRIST)
        view, desk, said = self.view([wrist], pin_s=5.0)
        with view:
            self.assertTrue(_until(lambda: len(desk.seen("wrist")) >= 1))
            view.show_located(_on_the_wrist(time.time(), _cube()))
            self.assertTrue(_until(lambda: any(_tinted(image) for image in desk.seen("wrist"))))
            head = view._windows["wrist"].status(time.time())[0]
            view.show_located(_on_the_wrist(float("nan"), _cube()))
            self.assertTrue(_until(lambda: view._windows["wrist"].status(time.time())[0].startswith(
                "wrist, wrist camera: located; ")))
            unstamped = view._windows["wrist"].status(time.time())[0]
            self.assertTrue(view.running)
        self.assertEqual({_grey(image) for image in desk.seen("wrist") if _tinted(image)}, {0},
                         "the masks alone go on a blank, not on a frame the view took")
        self.assertEqual((said, view.off), ([], ""))
        self.assertRegex(head, r"^wrist, wrist camera: located at \d\d:\d\d:\d\d; no image of the locate was "
                               r"handed over, so the masks alone$")
        self.assertEqual(unstamped, "wrist, wrist camera: located; no image of the locate was handed over, so the "
                                    "masks alone")

    def test_an_image_the_view_cannot_show_leaves_the_masks_alone_rather_than_losing_the_locate(self) -> None:
        """A four-channel image made the whole locate vanish from the window, event line and all (verification of
        2026-09-25): the image is only the background, so the masks go on a blank and the locate is still shown."""
        wrist = _Eye("wrist", 50, rig=_ON_THE_WRIST)
        view, desk, said = self.view([wrist], pin_s=5.0)
        with view:
            self.assertTrue(_until(lambda: len(desk.seen("wrist")) >= 1))
            view.show_located(_on_the_wrist(time.time(), _cube()), image=np.zeros((48, 64, 4), dtype=np.uint8))
            self.assertTrue(_until(lambda: any(_tinted(image) for image in desk.seen("wrist"))))
            head = view._windows["wrist"].status(time.time())[0]
        self.assertEqual({_grey(image) for image in desk.seen("wrist") if _tinted(image)}, {0})
        self.assertIn("so the masks alone", head)
        self.assertEqual((said, view.off), ([], ""))

    def test_a_pin_before_the_camera_gave_a_frame_shows_the_handed_image(self) -> None:
        wrist = _Eye("wrist", 50, rig=_ON_THE_WRIST)
        view, desk, _ = self.view([wrist], live_hz=0, pin_s=5.0)
        with view:
            view.show_located(_on_the_wrist(time.time(), _cube()), image=np.full((48, 64, 3), 120, dtype=np.uint8))
            self.assertTrue(_until(lambda: any(_tinted(image) for image in desk.seen("wrist"))))
        self.assertEqual({_grey(image) for image in desk.seen("wrist") if _tinted(image)}, {120})
        self.assertEqual(wrist.looks, 0)

    def test_a_locate_that_found_nothing_holds_nothing_and_is_only_said(self) -> None:
        wrist = _Eye("wrist", 50, rig=_ON_THE_WRIST)
        view, desk, _ = self.view([wrist], pin_s=5.0)
        with view:
            self.assertTrue(_until(lambda: len(desk.seen("wrist")) >= 1))
            view.show_located(_on_the_wrist(time.time(), _cube()), image=np.full((48, 64, 3), 120, dtype=np.uint8))
            self.assertTrue(_until(lambda: _tinted(desk.seen("wrist")[-1])))
            view.show_located(_on_the_wrist(time.time()), image=np.full((48, 64, 3), 90, dtype=np.uint8),
                              prompt="a red cube")
            wrist.grey = 200  # the arm moves on to its next look: the window follows it
            self.assertTrue(_until(lambda: _grey(desk.seen("wrist")[-1]) == 200
                                   and not _tinted(desk.seen("wrist")[-1])),
                            "a locate that found nothing froze the window")
            window = view._windows["wrist"]
            status = window.status(time.time())
            self.assertIsNone(window.pinned, "a locate that found nothing pinned something, or kept the earlier pin")
        self.assertNotIn(90, {_grey(image) for image in desk.seen("wrist")}, "the image of an empty locate was shown")
        self.assertRegex(status[0], r"^wrist, wrist camera: live, frame \d+\.\d s old$")
        self.assertRegex(status[1], r"^last: nothing located for 'a red cube' at \d\d:\d\d:\d\d, \d+ s ago$")

    def test_a_fixed_cameras_masks_go_on_over_the_live_frame(self) -> None:
        overhead = _Eye("overhead", 50, rig=_FIXED)
        view, desk, _ = self.view([overhead], pin_s=5.0)
        with view:
            self.assertTrue(_until(lambda: len(desk.seen("overhead")) >= 1))
            view.show_located(_located("overhead", _cube()), image=np.full((48, 64, 3), 120, dtype=np.uint8))
            self.assertTrue(_until(lambda: any(_tinted(image) for image in desk.seen("overhead"))))
            overhead.grey = 200  # a newer frame of a scene that stayed put
            self.assertTrue(_until(lambda: any(_tinted(image) and _grey(image) == 200
                                               for image in desk.seen("overhead"))),
                            "a fixed camera's masks stopped following its live frame")
            head = view._windows["overhead"].status(time.time())[0]
        self.assertEqual({_grey(image) for image in desk.seen("overhead") if _tinted(image)}, {50, 200})
        self.assertRegex(head, r"^overhead, fixed camera: live, frame \d+\.\d s old$")

    def test_an_overlay_pinned_on_a_wrist_cameras_window_shows_as_it_was_handed(self) -> None:
        wrist = _Eye("wrist", 90, rig=_ON_THE_WRIST)
        view, desk, _ = self.view([wrist], pin_s=5.0)
        with view:
            self.assertTrue(_until(lambda: len(desk.seen("wrist")) >= 1))
            view.show_image("wrist", _png((0, 0, 255)), "run 0: succeeded", ok=True)
            self.assertTrue(_until(lambda: any(tuple(int(v) for v in _frame_of(image)[-1, 5]) == (0, 0, 255)
                                               for image in desk.seen("wrist"))))
            head = view._windows["wrist"].status(time.time())[0]
        self.assertRegex(head, r"^wrist, wrist camera: run 0: succeeded, pinned \d+\.\d s ago$")

    def test_what_says_a_camera_moves_with_the_arm(self) -> None:
        rides = live_view_module._rides_on_the_arm
        fixed_located = _located("x")
        self.assertTrue(rides(_on_the_wrist(0.0), "mounting not declared"), "the Located says eye in hand")
        self.assertTrue(rides(SimpleNamespace(mounting="", tool_to_base_mm=_TOOL), "fixed camera"),
                        "a tool pose at the shutter")
        self.assertTrue(rides(fixed_located, "wrist camera"), "the rig declares the wrist")
        self.assertFalse(rides(fixed_located, "mounting not declared"))
        self.assertFalse(rides(SimpleNamespace(mounting="", tool_to_base_mm=None), "fixed camera"))
        self.assertTrue(rides(SimpleNamespace(), "mounting not declared"), "nothing says fixed: held, fail closed")


# ---------------------------------------------------------------------------------------------------------------------
# Closing a window ends the view and nothing else
# ---------------------------------------------------------------------------------------------------------------------


class ClosingAWindowEndsOnlyTheViewTests(_Case):

    def test_closing_any_window_closes_every_window_and_nothing_else(self) -> None:
        eyes = [_Eye("wrist"), _Eye("overhead")]
        view, desk, said = self.view(eyes, desk=_Desk(closes="willy camera: overhead"))
        view.start()
        self.assertTrue(_until(lambda: not view.running))
        self.assertEqual(sorted(desk.destroyed), sorted(desk.opened))
        self.assertEqual(said, ["camera windows closed (the window of 'overhead' was closed): the program goes on. "
                                "The pendant stops the robot"])
        self.assertEqual((view.poll_key(), view.poll_key()), ("closed", None),
                         "a person guiding the arm reads the close once, as a finish")
        looks = [eye.looks for eye in eyes]
        started = time.perf_counter()
        view.show_located(_located("wrist", _cube()))
        view.show_image("wrist", _png((0, 0, 255)), "run 1: failed")
        view.note("run 1: failed")
        view.guide(["x"], "guide")
        view.watch(_Eye("side"))
        self.assertLess(time.perf_counter() - started, 0.1, "the program's side waited on a closed view")
        time.sleep(0.3)
        self.assertEqual([eye.looks for eye in eyes], looks, "a camera was read after the view closed")

    def test_esc_in_any_window_closes_every_window(self) -> None:
        view, desk, said = self.view([_Eye("wrist"), _Eye("overhead")], desk=_Desk(keys=(-1, -1, 27)))
        view.start()
        self.assertTrue(_until(lambda: not view.running))
        self.assertEqual(view.off, "ESC")
        self.assertEqual(said, ["camera windows closed (ESC): the program goes on. The pendant stops the robot"])
        self.assertEqual(sorted(desk.destroyed), sorted(desk.opened))

    def test_a_gui_error_switches_the_view_off_with_one_line(self) -> None:
        view, _, said = self.view([_Eye("wrist")], desk=_Desk(fail="imshow"))
        view.start()
        self.assertTrue(_until(lambda: not view.running))
        self.assertEqual(said, ["camera windows off after RuntimeError: the display went away; everything else runs "
                                "as it would without them"])

    def test_a_campaign_runs_to_its_end_when_a_window_is_closed(self) -> None:
        service = _Service(["succeeded", "succeeded", "succeeded"], slow_s=0.15)
        view, _, said = self.view([], desk=_Desk(closes="willy camera: wrist"))
        with view:
            report = PickRun.from_service(service, runs=3, recording=Recording.off(), view=view).execute()
            self.assertTrue(_until(lambda: not view.running))
        self.assertEqual((report.attempted, report.succeeded, report.exit_code), (3, 3, 0), report.render())
        self.assertEqual(len(said), 1)
        self.assertIn("the window of 'wrist' was closed", said[0])


# ---------------------------------------------------------------------------------------------------------------------
# The first window guides a person moving the arm
# ---------------------------------------------------------------------------------------------------------------------


class TheFirstWindowGuidesTests(_Case):

    def test_the_guide_and_its_red_boundary_show_on_the_first_window_only(self) -> None:
        view, desk, _ = self.view([_Eye("wrist", 90), _Eye("overhead", 90)])
        view.guide(["OUTSIDE - NOT CAPTURED: wrist 3 stands at 200.0 deg"], "outside", 0.3)

        def red(image: np.ndarray) -> bool:
            # The band over the status and the border round the frame (above the closeness bar).
            return tuple(int(v) for v in image[2, 10]) == _RED and tuple(int(v) for v in image[-20, 2]) == _RED

        with view:
            self.assertTrue(_until(lambda: any(red(image) for image in desk.seen("wrist"))))
            self.assertTrue(_until(lambda: len(desk.seen("overhead")) >= 2))
        self.assertFalse(any(tuple(int(v) for v in image[2, 10]) == _RED for image in desk.seen("overhead")))

    def test_keys_typed_into_a_window_while_a_guide_shows_are_handed_back(self) -> None:
        view, _, _ = self.view([_Eye("wrist"), _Eye("overhead")], desk=_Desk(keys=(13, ord("s"), ord("q"))))
        view.guide(["MOVE THE ARM BY HAND to where LOOK_1 should be, then Enter"], "guide")
        keys: list[str] = []

        def read() -> int:
            while (key := view.poll_key()) is not None:
                keys.append(key)
            return len(keys)

        with view:
            self.assertTrue(_until(lambda: read() >= 3))
        self.assertEqual(keys, ["enter", "skip", "finish"])


# ---------------------------------------------------------------------------------------------------------------------
# A locator, a campaign and a cell, handed a view
# ---------------------------------------------------------------------------------------------------------------------


class _SpyView:
    """A view that keeps what it was handed, and the thread each call came from."""

    def __init__(self, *, raises: bool = False) -> None:
        self.raises = raises
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _keep(self, name: str, *args: Any, **keywords: Any) -> None:
        self.calls.append((name, args, keywords))
        if self.raises:
            raise RuntimeError("the display went away")

    def watch(self, *cameras: Any) -> None:
        self._keep("watch", *cameras)

    def start(self) -> None:
        self._keep("start")

    def show_located(self, located: Any, image: Any = None, **keywords: Any) -> None:
        self._keep("show_located", located, image=image, **keywords)

    def show_image(self, rig_id: Any, image: Any, caption: str = "", **keywords: Any) -> None:
        self._keep("show_image", rig_id, image, caption, **keywords)

    def note(self, text: str, rig_id: Any = None) -> None:
        self._keep("note", text, rig_id)

    def of(self, name: str) -> list[tuple[Any, ...]]:
        return [args for kind, args, _ in self.calls if kind == name]


class ALocatorShowsWhatItLocatedTests(unittest.TestCase):

    def test_a_locator_given_a_view_watches_its_camera_and_shows_each_located(self) -> None:
        owner, view = _Owner(), _SpyView()
        locator = Locator.from_parts(camera=owner, backend=_backend("red cube"), view=view)
        self.assertEqual(view.of("watch"), [(owner,)])
        located = locator.locate("a red cube")
        self.assertEqual(view.of("show_located"), [(located,)])

    def test_the_view_is_handed_the_colour_image_the_locate_segmented_and_no_camera_is_read_for_it(self) -> None:
        from tests.test_locator import _block_frame, _Handle

        def colour_frame() -> Any:
            frame = _block_frame()
            colour = np.zeros_like(frame.color)
            colour[..., 0], colour[..., 2] = np.arange(40, dtype=np.uint8)[:, None], 200  # BGR, not symmetric
            return type(frame)(color=colour, depth=frame.depth, captured_at_s=frame.captured_at_s)

        segmented: list[np.ndarray] = []
        objects = _backend("red cube").perceive(None, "")
        backend = SimpleNamespace(perceive=lambda bgr, _prompt: segmented.append(np.array(bgr, copy=True)) or objects)
        handle, view = _Handle([colour_frame()]), _SpyView()
        located = Locator.from_parts(camera=_Owner(handle=handle), backend=backend, view=view).locate("a red cube")
        ((_, args, keywords),) = [call for call in view.calls if call[0] == "show_located"]
        self.assertEqual(args, (located,))
        self.assertEqual(keywords["prompt"], "a red cube")
        (image,) = segmented
        self.assertTrue(np.array_equal(keywords["image"], image), "not the image the locate segmented, in BGR")
        self.assertTrue(np.array_equal(keywords["image"], colour_frame().color))

        alone = _Handle([colour_frame()])
        without = Locator.from_parts(camera=_Owner(handle=alone), backend=_backend("red cube")).locate("a red cube")
        self.assertEqual(handle.grabs, alone.grabs, "a view made the locate grab more")
        self.assertEqual(without.to_dict(), located.to_dict(), "a view changed what was located")

    def test_a_locate_that_found_nothing_is_handed_over_with_its_prompt(self) -> None:
        view = _SpyView()
        located = Locator.from_parts(camera=_Owner(), backend=_backend(), view=view).locate("a blue plate")
        self.assertEqual(located.objects, ())
        ((_, args, keywords),) = [call for call in view.calls if call[0] == "show_located"]
        self.assertEqual((args, keywords["prompt"]), ((located,), "a blue plate"))

    def test_the_tree_and_config_doors_hand_the_view_on(self) -> None:
        from tests.test_locator import _cell_tree

        owner, view = _Owner(), _SpyView()
        spec = mock.MagicMock()
        spec.from_config.return_value.build.return_value = _backend("red cube")
        with mock.patch("src.models.perception_spec.PerceptionSpec", spec):
            locator = Locator.from_config(_cell_tree(), object(), camera=owner, view=view)
        locator.locate("a red cube")
        self.assertEqual(len(view.of("show_located")), 1)
        tree = SimpleNamespace(robot="robot", app_config=SimpleNamespace(models="models"))
        with mock.patch.object(Locator, "from_config", return_value="locator") as from_config:
            Locator.from_tree(tree, camera=owner, view=view)
        self.assertIs(from_config.call_args.kwargs["view"], view)

    def test_a_view_that_raises_never_stops_a_locate(self) -> None:
        located = Locator.from_parts(camera=_Owner(), backend=_backend("red cube"),
                                     view=_SpyView(raises=True)).locate("a red cube")
        self.assertEqual(len(located.objects), 1)

    def test_a_locate_that_fails_says_so_on_its_window_and_raises_as_before(self) -> None:
        class _Failing:
            rig_id = "overhead"

            def grab(self) -> Any:
                raise RuntimeError("Frame didn't arrive within 5000")

            def get_intrinsics(self) -> Any:
                return None

        view = _SpyView()
        locator = Locator.from_parts(camera=_Owner(handle=_Failing()), backend=_backend("red cube"), view=view)
        with self.assertRaisesRegex(RuntimeError, "Frame didn't arrive"):
            locator.locate("a red cube")
        self.assertEqual(view.of("note"), [("locate failed: RuntimeError: Frame didn't arrive within 5000",
                                            "overhead")])


class _Report:
    def __init__(self, outcome: str) -> None:
        self.outcome = outcome

    def failure_summary(self) -> str:
        return "reason=no_valid_grasp"


def _source(camera: Any) -> Any:
    return SimpleNamespace(streamer=SimpleNamespace(camera=camera))


class _Service:
    """A built cell's service: its cameras where the build keeps them, a debug render per pick, and what was done."""

    def __init__(self, outcomes: list[str], *, renders: list[bytes | None] | None = None, slow_s: float = 0.0,
                 cameras: tuple[Any, ...] | None = None) -> None:
        self.outcomes = list(outcomes)
        self.renders = list(renders) if renders is not None else [_png((0, 0, 255)) for _ in outcomes]
        self.slow_s = slow_s
        wrist, side, overhead = cameras if cameras is not None else (_Eye("wrist"), _Eye("side"), _Eye("overhead"))
        self.cameras = (wrist, side, overhead)
        self.runtime = SimpleNamespace(orchestrator=SimpleNamespace(
            perception=_source(wrist),
            multi_camera_perception=SimpleNamespace(sources={"side": _source(side)}),
            planner_world_cameras=SimpleNamespace(cameras=(overhead,)),
        ))
        self.rendering = False
        self.rendering_while_picking: list[bool] = []
        self.last_debug_image_png: bytes | None = None
        self.pick_threads: set[int] = set()

    @property
    def debug_image_rendering_enabled(self) -> bool:
        return self.rendering

    def enable_debug_image_rendering(self, enabled: bool = True) -> None:
        self.rendering = bool(enabled)

    def pick(self) -> _Report:
        self.pick_threads.add(threading.get_ident())
        self.rendering_while_picking.append(self.rendering)
        if self.slow_s:
            time.sleep(self.slow_s)
        render = self.renders.pop(0)
        if render is not None:
            self.last_debug_image_png = render
        return _Report(self.outcomes.pop(0))


class APickRunShowsEachAttemptTests(unittest.TestCase):

    def test_the_campaign_shows_every_camera_of_the_cell_and_pins_each_overlay_on_the_primarys_window(self) -> None:
        first, second = _png((0, 0, 255)), _png((0, 255, 0))
        service, view = _Service(["succeeded", "execution_failed"], renders=[first, second]), _SpyView()
        report = PickRun.from_service(service, runs=2, recording=Recording.off(), view=view).execute()
        self.assertEqual(report.attempted, 2)
        self.assertEqual(view.of("watch"), [service.cameras], "every camera, the primary first")
        self.assertIn(("start",), [(kind,) for kind, _, _ in view.calls])
        self.assertEqual(service.rendering_while_picking, [True, True], "the grasp overlay was not rendered")
        self.assertFalse(service.rendering, "the campaign's render switch outlived it")
        images = [(args, keywords) for kind, args, keywords in view.calls if kind == "show_image"]
        self.assertEqual([(args[0], args[1], args[2], keywords.get("ok")) for args, keywords in images], [
            ("wrist", first, "run 0: succeeded", True),
            ("wrist", second, "run 1: execution_failed, reason=no_valid_grasp", False),
        ])
        self.assertEqual(view.of("note"), [("run 0: succeeded", None),
                                           ("run 1: execution_failed, reason=no_valid_grasp", None)])

    def test_an_attempt_that_rendered_nothing_new_pins_nothing_stale(self) -> None:
        service, view = _Service(["succeeded", "execution_failed"], renders=[_png((0, 0, 255)), None]), _SpyView()
        PickRun.from_service(service, runs=2, recording=Recording.off(), view=view).execute()
        self.assertEqual([args[2] for args in view.of("show_image")], ["run 0: succeeded"])
        self.assertEqual(len(view.of("note")), 2, "the attempt with no new overlay is still said")

    def test_a_render_switch_the_caller_had_on_stays_on(self) -> None:
        service = _Service(["succeeded"])
        service.rendering = True
        PickRun.from_service(service, runs=1, recording=Recording.off(), view=_SpyView()).execute()
        self.assertTrue(service.rendering)

    def test_without_a_view_nothing_is_rendered_or_shown(self) -> None:
        service = _Service(["succeeded"])
        PickRun.from_service(service, runs=1, recording=Recording.off()).execute()
        self.assertEqual(service.rendering_while_picking, [False])

    def test_a_view_that_raises_never_stops_the_campaign(self) -> None:
        service = _Service(["succeeded", "succeeded"])
        report = PickRun.from_service(service, runs=2, recording=Recording.off(), view=_SpyView(raises=True)).execute()
        self.assertEqual((report.attempted, report.succeeded, report.exit_code), (2, 2, 0))

    def test_the_view_thread_never_calls_the_service_and_the_campaign_picks_on_its_own_thread(self) -> None:
        service = _Service(["succeeded", "succeeded"])
        desk = _Desk()
        with _here(), LiveView(gui=desk, say=lambda _line: None) as view:
            PickRun.from_service(service, runs=2, recording=Recording.off(), view=view).execute()
            self.assertTrue(_until(lambda: len(desk.seen("wrist")) >= 1))
        self.assertEqual(service.pick_threads, {threading.get_ident()})
        self.assertNotIn(threading.get_ident(), desk.threads())

    def test_a_cell_campaign_shows_the_cameras_its_build_opened(self) -> None:
        service, view = _Service(["succeeded"]), _SpyView()

        class _Session:
            teardown = None

            def __init__(self) -> None:
                self.service = service

            def __enter__(self) -> Any:
                return self

            def __exit__(self, *exc: Any) -> None:
                pass

        cell = Cell(robot_config=SimpleNamespace(), _service=service)
        with mock.patch.object(Cell, "connected", lambda self, announce=None: _Session()):
            report = PickRun.from_cell(cell, runs=1, recording=Recording.off(), view=view).execute()
        self.assertEqual(report.succeeded, 1, report.render())
        self.assertEqual(view.of("watch"), [service.cameras])


class ACellNamesTheCamerasItsBuildOpenedTests(unittest.TestCase):

    def test_the_primary_first_then_the_fused_then_the_worlds_own_each_once(self) -> None:
        service = _Service([])
        wrist, side, overhead = service.cameras
        service.runtime.orchestrator.planner_world_cameras = SimpleNamespace(cameras=(overhead, wrist))
        cell = Cell(robot_config=SimpleNamespace(), _service=service)
        self.assertEqual(cell.cameras, (wrist, side, overhead))

    def test_a_cell_not_built_has_no_cameras_to_name(self) -> None:
        with self.assertRaises(CellNotBuilt):
            _ = Cell(robot_config=SimpleNamespace()).cameras

    def test_a_rehearsal_and_a_double_that_answers_everything_hold_no_camera(self) -> None:
        rehearsal = SimpleNamespace(runtime=SimpleNamespace(orchestrator=SimpleNamespace(
            perception=SimpleNamespace(), multi_camera_perception=None, planner_world_cameras=None)))
        self.assertEqual(Cell(robot_config=SimpleNamespace(), _service=rehearsal).cameras, ())
        self.assertEqual(Cell(robot_config=SimpleNamespace(), _service=mock.MagicMock()).cameras, ())


class TheLayersStayApartTests(unittest.TestCase):
    """The view sits in the camera package, below the robot: it loads no robot code, and the robot's teaching, its
    campaigns and its cell load no camera code for it (they read a view by its methods, and teaching imports it only
    where cameras are shown)."""

    _PROBE = ("import importlib, json, sys; importlib.import_module({module!r}); "
              "print(json.dumps(sorted(m for m in sys.modules if m.startswith({prefix!r}))))")

    def _loaded(self, module: str, prefix: str) -> list[str]:
        import json
        import subprocess
        import sys

        out = subprocess.run([sys.executable, "-c", self._PROBE.format(module=module, prefix=prefix)], cwd=_ROOT,
                             capture_output=True, text=True, timeout=300, check=True).stdout
        return json.loads(out.strip().splitlines()[-1])

    def test_the_view_loads_no_robot_code(self) -> None:
        self.assertEqual(self._loaded("src.camera.live_view", "src.robot"), [])

    def test_teaching_a_campaign_and_a_cell_load_no_camera_code(self) -> None:
        for module in ("src.robot.execution.teach", "src.robot.execution.pick_run", "src.robot.execution.cell"):
            with self.subTest(module):
                self.assertEqual(self._loaded(module, "src.camera"), [])

    def test_the_probe_sees_a_camera_module_where_one_is_loaded(self) -> None:
        """The control: the same probe over the view sees the camera package."""
        self.assertIn("src.camera.live_view", self._loaded("src.camera.live_view", "src.camera"))


# ---------------------------------------------------------------------------------------------------------------------
# The examples ask for it
# ---------------------------------------------------------------------------------------------------------------------


class TheExamplesShowEveryCameraTests(unittest.TestCase):
    _SWITCH = "SHOW_CAMERAS = True  # False: no camera windows"

    def _source(self, number: str) -> str:
        (path,) = sorted((_ROOT / "examples" / "real_robot").glob(f"{number}_*.py"))
        return path.read_text(encoding="utf-8")

    def test_the_switch_is_at_the_top_of_each_and_said_in_its_docstring(self) -> None:
        import ast

        for number in ("12", "13", "17"):
            with self.subTest(number):
                source = self._source(number)
                tree = ast.parse(source)
                first = next(node for node in tree.body
                             if not isinstance(node, (ast.Expr, ast.Import, ast.ImportFrom)))
                self.assertEqual(ast.get_source_segment(source, first), "SHOW_CAMERAS = True")
                self.assertIn(self._SWITCH, source)
                self.assertIn("SHOW_CAMERAS", ast.get_docstring(tree) or "")

    def test_each_hands_every_camera_to_the_view(self) -> None:
        for number in ("12", "17"):
            with self.subTest(number):
                source = self._source(number)
                self.assertIn("LiveView(show=SHOW_CAMERAS)", source)
                self.assertIn("view=view", source)
        thirteen = self._source("13")
        self.assertIn("LiveView(cameras, show=SHOW_CAMERAS)", thirteen)
        # Its locators get the view however they are built: one per camera, or all over one set of models.
        self.assertRegex(thirteen, r"Locator\.(from_tree|for_cameras)\([^\n]*view=view")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
