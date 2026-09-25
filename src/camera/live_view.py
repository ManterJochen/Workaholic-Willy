"""What every camera of a cell sees, live, one window each, beside a program that picks, locates or is taught.

    from willy import LiveView, Locator, PickRun, Recording

    with LiveView(cameras) as view:            # a window per open camera owner, each live at about four frames a second
        located = Locator.from_tree(tree, camera=cameras[0], view=view).locate("a red cube")  # its masks pinned
        report = PickRun.from_cell(cell, runs=5, recording=Recording.off(), view=view).execute()  # each overlay pinned
    taught = teach_poses(robot, tree=tree, camera=wrist)  # teaching builds its own: the one camera it teaches for

The owner, 2026-09-24: during picks, a switch shows live what every camera of the cell sees, fixed cameras too, one
window per camera, and where the software has just detected something, that as well, so the owner can follow what
each camera sees. While poses are taught it shows the one camera they are taught for, because a pose belongs to one
camera (the same evening). A window shows:

* The camera's live image, about four a second (:data:`LIVE_HZ`), shrunk to at most ``max_width`` pixels wide.
* A status line: the rig, whether it rides on the wrist or is fixed (read off its rig: a declared body or an eye in
  hand calibration is the wrist, an eye to hand calibration is fixed), how old the frame shown is, and the last thing
  that happened on the window and how long ago.
* For ``pin_s`` seconds after it happened: what a locator located (:meth:`LiveView.show_located`), each object's mask
  tinted in a colour of its own with its label, score and centre in BASE; or an image such as a pick's grasp overlay
  (the service's ``last_debug_image_png``, :meth:`LiveView.show_image`) in the live image's place, on a band green
  for a success and red for a failure. A ``Locator`` and a ``PickRun`` handed the view do both by themselves, and a
  campaign says each attempt's outcome on every window (:meth:`LiveView.note`).
* Where the masks go depends on whether the camera moves with the arm (review finding F3, 2026-09-25). A fixed
  camera's scene stays put while the arm moves, so its masks go over its live image. A camera on the wrist does not:
  example 13 picks and places right after each locate, and masks drawn over the newest frame would mark where the
  objects no longer are. So its window holds the colour image the locate was made on for the pin, masks drawn over
  it, on an amber band that says ``as located at 07:31:02; the arm may have moved since``, and goes live again after
  the pin. That image is handed over with the ``Located`` (``show_located(located, image=...)``; a ``Locator`` hands
  the frame it segmented): data the locate already holds, not a read of the camera, so ``peek()`` stays the view's
  only read. It is never a frame the view took itself: a look is quiet for ``PEEK_QUIET_S`` after every measuring
  grab, so none falls inside a locate's grabs, and for a locate whose detection returns soon after its shutter the
  view's frame nearest the shutter is one from the arm's way to its look. With no image handed over, the masks go on
  a blank and the window says so. The live frames keep coming during the pin, so the window is live again the moment
  the pin ends. What says the camera moves with the arm: the ``Located`` (``eye_in_hand``, or a tool pose at its
  shutter) or its rig (a declared body or an eye in hand calibration). Only a camera one of them calls fixed, and
  neither calls the wrist, gets its masks over the live image. A camera that nothing says anything about is held too
  (fail closed), as a held image never shows the masks anywhere the camera did not see them.
* A locate that found nothing holds nothing: its window stays live, says ``nothing located for 'a red cube' at
  07:31:02`` in its status, and a pin of an earlier locate on it ends.
* While a person guides the arm, the first window is the guide (``hand_guiding.GuideView``): the guide's lines over
  its image, red with a red border outside a boundary, a closeness bar under it, and Enter, Space, ``s`` and ``q``
  typed into any of the windows handed back through :meth:`poll_key`. ``teach_poses(robot, tree=tree, camera=...)``
  builds a view of that one camera so.

How a window looks without changing what the robot measures, the owner's second demand. Every frame comes through
the camera's own owner, never a second owner and never the device opened again, and through its display path,
``Camera.peek()``, never through ``grab()``. The display path was chosen over the two other ways because it is the
only one that keeps the window live without touching a measurement: a view that showed the last judged frame would
freeze whenever nothing measures (the arm moving, a person teaching), and one that paused only between
``camera_moved()`` and the next judged grab would still hand a RealSense's temporal filter frames taken while the
arm moved, and keep its pause clock fresh, where the pick perception relies on that pause. So:

* A RealSense is read with ``poll_for_frames``: its newest colour image, copied, with no filter, no alignment and no
  pause clock touched (``RealSenseRGBDStreamer.peek``). The temporal filter's history holds only the frames the
  measuring grabs handed it: a frame looked at after the planning world's ``camera_moved()``, still from the pose
  before the move, never fills a hole of the pose after it, and a carried camera still drops its history after a
  pause between two grabs, however many frames were looked at in it. Both are held on librealsense's own filters.
* The owner never makes a measuring grab wait for a look: a look never waits for the rig's lock and gives nothing
  while a grab holds it, and gives nothing for ``PEEK_QUIET_S`` after every measuring grab and every
  ``camera_moved()``, longer than the pause the driver still counts as one burst of grabs. A burst (the world's
  notice, warm-ups and kept grab; a pick frame's warm-ups and grab) is never entered, and at most one look, begun
  before a burst, stands in front of its first grab, for as long as a poll and a copy take. A look can take the
  frameset the device held, and the grab after it waits for the next one, which is at least as new.
* A measuring grab's stamp stays the host time read just before its own device read; a look carries its own stamp,
  which is only the age in the status line.
* A device that keeps no history (no ``camera_moved``) is read with a plain grab, which feeds nothing; one that keeps
  a history and offers no display path is not read at all, and its window says why.

Threads. One daemon thread of the view's own makes every HighGUI call for every window, as HighGUI on Windows serves
a thread's windows from that thread's message loop, and reads every camera; nothing else of the program is called
from it, the arm least of all. The program's side (:meth:`watch`, :meth:`show_located`, :meth:`show_image`,
:meth:`note`, :meth:`guide`) only hands things over, an image shrunk to a copy of the view's own, and never waits,
and :meth:`poll_key` never waits either.

Closing. Closing any window, or ESC in one, closes every window of the view and nothing else, with one printed line:
a pick goes on (the pendant stops the robot, Ctrl+C in the console stops the program), and a person guiding the arm
reads the close as a finish, so teaching finishes with the arm held. An error in a GUI call switches the view off
the same way, with one line. A camera given back while its window is open is not read again: its owner refuses the
look before the device is touched, and the window says so until it closes.

Where no window can show (``preview_unavailable``: an OpenCV with no GUI, no display, an SSH session, macOS, or
``WILLY_NO_PREVIEW``), the view says so in one line and does nothing: everything else runs as it would without it.
``show=False`` does nothing and says nothing.

This module imports nothing from ``src.robot``: a ``Located`` is read by its attributes, and a guide is two calls.
Measured with a HighGUI double and cameras of the tests' own; four frames a second across several RealSense cameras
on one USB bus, the Windows desktop, and the windows beside the CB3's teach-mode watchdog are measured at the cell.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import cv2 as cv
import numpy as np

from src.calibration import preview as _preview
from src.calibration.preview import MAX_LIVE_HZ, WindowThread, _ascii, _bgr_copy, _shrink, annotate

__all__ = ["LIVE_HZ", "PIN_S", "LiveView", "draw_located"]

logger = logging.getLogger(__name__)

#: How many live frames a second each window takes from its camera; at most :data:`MAX_LIVE_HZ`.
LIVE_HZ = 4.0
#: How long what was just found stays pinned on its window, in seconds: a located object's masks, a pick's overlay.
PIN_S = 4.0

#: How long the view's thread waits for a key per turn, in milliseconds. The windows stay responsive, and a close is
#: seen within it.
_WAIT_MS = 30
#: How soon a window asks its camera again after a look gave nothing: the rig measured, or had no new image.
_BUSY_RETRY_S = 0.1
#: How soon after a look failed.
_RETRY_S = 1.0
#: How often a window with nothing new is drawn again, so the age of its frame stays true.
_REDRAW_S = 0.5
#: The most things handed over that wait for the view's thread; the oldest goes first.
_PENDING = 64
#: How many located objects the status line names; the rest are counted.
_NAMED = 4

_FOOTER = "ESC or closing a window closes every camera window, and nothing else. The pendant stops the robot."
#: The footer on every window while a person guides the arm, where closing a window finishes the run.
_GUIDE_FOOTER = ("Enter or Space captures, s skips, q finishes; ESC or closing a window finishes, and the arm is held. "
                 "The pendant stops the robot.")

#: One colour per located object, in turn, BGR.
_PALETTE = ((0, 200, 255), (255, 140, 0), (80, 220, 80), (220, 60, 220), (0, 110, 255), (255, 255, 0))
#: How much of an object's colour tints its mask.
_TINT = 0.45
_FONT = cv.FONT_HERSHEY_SIMPLEX
_LABEL_SCALE = 0.4


# ---------------------------------------------------------------------------------------------------------------------
# The drawing: pure and headless
# ---------------------------------------------------------------------------------------------------------------------


def draw_located(image: Any, located: Any) -> np.ndarray:
    """``image`` with what ``located`` found drawn on it, as a new image.

    Each object's mask is tinted in a colour of its own and outlined, its middle is marked, and its index, label,
    score and centre (BASE millimetres) are written above it. ``located`` is a
    :class:`~src.robot.perception.locator.Located`, read by its attributes; its masks are at the size of the frame the
    locator judged and are scaled to ``image``, the same camera's image at any size. Pure and headless: ``image`` is
    not changed and no window is touched, and an object whose mask cannot be read is left out rather than raised.
    """
    out = _bgr_copy(image)
    height, width = out.shape[:2]
    for index, obj in enumerate(tuple(getattr(located, "objects", ()) or ())):
        colour = _PALETTE[index % len(_PALETTE)]
        try:
            mask = np.asarray(getattr(obj, "mask", None))
            if mask.ndim != 2 or mask.size == 0:
                continue
            scaled = cv.resize(mask.astype(np.uint8), (width, height), interpolation=cv.INTER_NEAREST) > 0
            if not scaled.any():
                continue
            out[scaled] = (out[scaled] * (1.0 - _TINT) + np.asarray(colour, dtype=np.float64) * _TINT).astype(np.uint8)
            contours, _ = cv.findContours(scaled.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(out, contours, -1, colour, 1, cv.LINE_AA)
            rows, cols = np.nonzero(scaled)
            middle = (int(round(float(cols.mean()))), int(round(float(rows.mean()))))
            cv.drawMarker(out, middle, colour, cv.MARKER_CROSS, 14, 2)
            _label(out, _object_label(index, obj), (int(cols.min()), max(10, int(rows.min()) - 4)), colour)
        except (cv.error, TypeError, ValueError) as exc:
            logger.debug("live view: object %d not drawn: %s", index, exc)
    return out


def _object_label(index: int, obj: Any) -> str:
    """``0: red cube 0.91 (412, -96, 31) mm``: what is written above a located object."""
    text = f"{index}: {_ascii(str(getattr(obj, 'label', '') or 'object'))}"
    score = getattr(obj, "score", None)
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        text += f" {float(score):.2f}"
    centre = getattr(obj, "centre_mm", None)
    if isinstance(centre, tuple) and len(centre) == 3:
        text += " ({:.0f}, {:.0f}, {:.0f}) mm".format(*(float(v) for v in centre))
    return text


def _label(image: np.ndarray, text: str, origin: tuple[int, int], colour: tuple[int, int, int]) -> None:
    """``text`` at ``origin`` in ``colour``, outlined in black so it reads on any image."""
    cv.putText(image, text, origin, _FONT, _LABEL_SCALE, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(image, text, origin, _FONT, _LABEL_SCALE, colour, 1, cv.LINE_AA)


def _located_event(located: Any, prompt: str = "") -> str:
    """What a window says a locator located: ``located 2: red cube 0.91, blue plate 0.84``, or, where it found
    nothing, ``nothing located for 'a red cube' at 07:31:02`` (the prompt and the shutter where they are known)."""
    objects = tuple(getattr(located, "objects", ()) or ())
    if not objects:
        at = _clock(getattr(located, "captured_at_s", None))
        said = f"nothing located for {_ascii(repr(str(prompt)))}" if prompt else "nothing located"
        return f"{said} at {at}" if at else said
    named = []
    for obj in objects[:_NAMED]:
        score = getattr(obj, "score", None)
        said = f" {float(score):.2f}" if isinstance(score, (int, float)) and not isinstance(score, bool) else ""
        named.append(_ascii(str(getattr(obj, "label", "") or "object")) + said)
    more = f", and {len(objects) - _NAMED} more" if len(objects) > _NAMED else ""
    return f"located {len(objects)}: {', '.join(named)}{more}"


def _mounting(camera: Any) -> str:
    """Where a camera rides, read off its rig (a handle's is its owner's): a declared body or an eye in hand
    calibration is the wrist, an eye to hand calibration is fixed, and a rig that declares neither says so."""
    rig = getattr(camera, "rig", None)
    if rig is None:
        rig = getattr(getattr(camera, "camera", None), "rig", None)
    mode = getattr(getattr(rig, "extrinsics", None), "mounting_mode", None)
    if getattr(rig, "body", None) is not None or mode == "eye_in_hand":
        return "wrist camera"
    if mode == "eye_to_hand":
        return "fixed camera"
    return "mounting not declared"


def _rides_on_the_arm(located: Any, mounting: str) -> bool:
    """Whether the camera that located ``located`` moves with the arm, so its live image leaves what it located behind
    as soon as the arm moves on, and its window holds the image of the locate for the pin.

    The ``Located`` says so (``eye_in_hand``, or a tool pose at its shutter), or the window's rig does (``mounting``
    is :func:`_mounting`'s ``wrist camera``). A camera is fixed only where the ``Located`` says ``eye_to_hand`` or the
    rig is declared fixed, and nothing says the wrist. One that nothing says anything of is held (fail closed): a held
    image never shows the masks anywhere the camera did not see them, where a live one can.
    """
    says = str(getattr(located, "mounting", "") or "")
    if says == "eye_in_hand" or getattr(located, "tool_to_base_mm", None) is not None or mounting == "wrist camera":
        return True
    return not (says == "eye_to_hand" or mounting == "fixed camera")


def _clock(stamp: Any) -> str | None:
    """``07:31:02``: the host time ``stamp`` on the local clock, or ``None`` where it is no time (``NaN``, none)."""
    if not isinstance(stamp, (int, float)) or isinstance(stamp, bool) or not math.isfinite(stamp):
        return None
    try:
        return time.strftime("%H:%M:%S", time.localtime(float(stamp)))
    except (OverflowError, OSError, ValueError):
        return None


def _take(source: Any) -> tuple[np.ndarray, float] | None:
    """One frame of ``source`` to look at and the host time it was taken, or ``None`` when it has none to give now.

    Through ``peek()`` where the source has it, as a camera owner and its handles do: the display path. A source
    without one is read with ``grab()`` only where it keeps no history a grab would feed (no ``camera_moved``); one
    that keeps a history and offers no display path is refused, because looking at it would change what it measures.
    """
    peek = getattr(source, "peek", None)
    if callable(peek):
        shot = peek()
        if shot is None:
            return None
        stamp = getattr(shot, "captured_at_s", None)
        return (np.asarray(getattr(shot, "color", shot)),
                float(stamp) if isinstance(stamp, (int, float)) else time.time())
    grab = getattr(source, "grab", None)
    if not callable(grab) or callable(getattr(source, "camera_moved", None)):
        raise RuntimeError(f"{type(source).__name__} offers no frame to look at (no peek()), and a grab() would feed "
                           "what it measures")
    before = time.time()
    frame = grab()
    colour = getattr(frame, "color", None)
    if colour is None:
        colour = getattr(frame, "left", frame)
    stamp = getattr(frame, "captured_at_s", None)
    return np.asarray(colour), float(stamp) if isinstance(stamp, (int, float)) else before


def _decoded(payload: Any, max_width: int) -> np.ndarray:
    """An image handed to :meth:`LiveView.show_image`, ready to show: PNG or JPEG bytes decoded, shrunk to fit."""
    if isinstance(payload, bytes):
        image = cv.imdecode(np.frombuffer(payload, dtype=np.uint8), cv.IMREAD_COLOR)
        if image is None:
            raise ValueError("the bytes are no PNG or JPEG")
        return _shrink(image, max_width)[0]
    return _bgr_copy(payload)


# ---------------------------------------------------------------------------------------------------------------------
# The windows
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class _Handed:
    """What the program handed the view: an image or located objects to pin (``image``, ``located``), or a line to
    say (``note``), for the window of ``rig_id`` (``None``: a note for every window, an image for the first)."""

    kind: str
    rig_id: str | None
    payload: Any = None
    text: str = ""
    ok: bool | None = None
    #: When it was handed over, host time.
    at_s: float = 0.0
    #: For located objects: the colour image the locate was made on, the view's own shrunk copy, or ``None``.
    still: np.ndarray | None = None


@dataclass(eq=False)
class _Window:
    """One camera's window, as the view's thread keeps it."""

    rig_id: str
    title: str
    source: Any
    mounting: str
    index: int = 0
    #: The latest live frame, shrunk, and the host time it was taken.
    live: np.ndarray | None = None
    live_at_s: float | None = None
    #: Why the latest look gave no frame, where it failed.
    live_error: str = ""
    #: When the camera is next asked for a frame, on the monotonic clock.
    next_look: float = 0.0
    #: What is pinned, the image it is shown as, and until when, on the monotonic clock. The image is an image handed
    #: over, or for located objects on a camera that moves with the arm, the image of the locate (or a blank) with them
    #: drawn on it; located objects on a fixed camera have none and go over each live frame.
    pinned: _Handed | None = None
    pinned_image: np.ndarray | None = None
    pinned_until: float = 0.0
    #: What the status line says of the image held for located objects (``""``: none is held).
    held: str = ""
    #: The last thing that happened on the window, and when, host time.
    event: tuple[str, float] | None = None
    drawn_at: float = float("-inf")
    dirty: bool = True
    shown: bool = False

    def status(self, now_s: float) -> list[str]:
        """The window's status at host time ``now_s``: the rig, where it rides and how old the frame shown is, or
        what the image held for located objects is; then the last thing that happened on it and how long ago, or an
        empty line."""
        where = f"{self.rig_id}, {self.mounting}"
        pinned = self.pinned
        if pinned is not None and pinned.kind == "image":
            head = f"{where}: {pinned.text or 'an image'}, pinned {max(0.0, now_s - pinned.at_s):.1f} s ago"
        elif pinned is not None and pinned.kind == "located" and self.held:
            head = f"{where}: {self.held}"
        elif self.live is None or self.live_at_s is None:
            head = f"{where}: " + (f"no live frame: {self.live_error}" if self.live_error else "no live frame yet")
        else:
            head = f"{where}: live, frame {max(0.0, now_s - self.live_at_s):.1f} s old"
            if self.live_error:
                head += f"; now {self.live_error}"
        last = "" if self.event is None else f"last: {self.event[0]}, {max(0.0, now_s - self.event[1]):.0f} s ago"
        return [head, last]


class LiveView(WindowThread):
    """One window per camera, every one drawn by one daemon thread of the view's own; a ``with`` block starts it and
    closes it.

    ``cameras`` are open camera owners (:class:`~src.camera.Camera`), or their handles: each gets a window titled
    ``<title>: <rig_id>``, in the order given, and :meth:`watch` adds more later. Each is read through its display
    path (``peek()``), never its measuring grab: the module docstring says why that leaves what the robot measures as
    it was. ``live_hz`` is capped at :data:`MAX_LIVE_HZ` (0 takes no live frames), ``pin_s`` is how long what was just
    found stays pinned, and ``max_width`` how wide a frame is shown.

    ``show=False`` is a view that does nothing and says nothing: a program's switch. ``goes_on`` finishes the one line
    a desk that cannot show a window gets, and the one an error that switched the view off gets; ``on_close`` says
    what happens once the operator closed a window (teaching: it finishes, and the arm is held). ``gui`` stands in for
    ``cv2`` and ``say`` for the printed lines.

    Building it touches no window and no camera. :meth:`start` opens the windows on the view's thread and
    :meth:`close` stops it and closes them; close it before the cameras are given back. Every other call returns at
    once and never raises.
    """

    _NAME = "live view"
    _THREAD = "willy-live-view"

    def __init__(
        self,
        cameras: Iterable[Any] = (),
        *,
        show: bool = True,
        title: str = "willy camera",
        live_hz: float = LIVE_HZ,
        pin_s: float = PIN_S,
        max_width: int = 640,
        goes_on: str = "everything else runs as it would without them",
        on_close: str = "the program goes on",
        gui: Any = None,
        say: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(gui=gui, say=say)
        self.show = bool(show)
        self.title = _ascii(title)
        rate = min(float(live_hz), MAX_LIVE_HZ)
        self._period_s: float | None = 1.0 / rate if rate > 0 else None
        self._pin_s = max(0.0, float(pin_s))
        self._max_width = max(64, int(max_width))
        self._goes_on = _ascii(goes_on)
        self._on_close = _ascii(on_close)
        #: The cameras watched, by rig id, in the order first watched: the program's side writes, the thread reads.
        self._sources: dict[str, Any] = {}
        #: What was handed over and not yet taken by the view's thread, oldest first.
        self._pending: list[_Handed] = []
        #: Each camera's window, by rig id: the view's thread's own.
        self._windows: dict[str, _Window] = {}
        #: How many frames the windows took from their cameras.
        self.looks = 0
        self.watch(*cameras)

    def __enter__(self) -> "LiveView":
        self.start()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.close()

    @property
    def rig_ids(self) -> tuple[str, ...]:
        """The rigs watched, in the order their windows open."""
        with self._lock:
            return tuple(self._sources)

    # --- the program's side: hand over and return -------------------------------------------------------------------

    def start(self) -> None:
        """Open a window per camera on the view's own thread. Idempotent, and it never raises.

        Where no window can show (``preview_unavailable``), one printed line says why and the view does nothing from
        then on; asked for none (``show=False``), it does nothing and says nothing.
        """
        if not self.show:
            return
        with self._lock:
            if self._thread is not None or self._stop.is_set():
                return
        reason = _preview.preview_unavailable()
        if reason is not None:
            with self._lock:
                if self._stop.is_set():
                    return
                self._stop.set()
            self.off = f"no window can show: {reason}"
            self._tell(f"No camera windows: {reason}. {self._goes_on[:1].upper()}{self._goes_on[1:]}.")
            return
        super().start()

    def watch(self, *cameras: Any) -> None:
        """Give each camera a window of its own, by its ``rig_id``: a rig watched already keeps its window, and a new
        owner of it takes the old one's place. Returns at once; the view's thread opens the windows."""
        with self._lock:
            for camera in cameras:
                rig_id = _ascii(str(getattr(camera, "rig_id", "") or f"camera {len(self._sources) + 1}"))
                self._sources[rig_id] = camera

    def show_located(self, located: Any, image: Any = None, *, prompt: str = "") -> None:
        """Pin what a locator located (a ``Located``) on its camera's window for ``pin_s`` seconds: every object's
        mask tinted, its label, score and centre, and a line saying what was located. On a fixed camera the masks go
        over each live frame; on one that moves with the arm, over ``image``, the BGR or grey colour image the locate
        was made on, copied here (shrunk to ``max_width``), held for the pin with a line saying when it was located,
        and with no ``image`` over a blank that says so (the module docstring says why never a frame of the view's).
        A locate that found nothing pins nothing: the window says ``nothing located for '<prompt>' at <shutter>`` and
        stays live. A camera the view has no window for shows nothing: its masks belong on no other camera's image."""
        if not self._taking():
            return
        try:
            camera = getattr(located, "camera", None)
            rig_id = None if camera is None else _ascii(str(camera))
            found = bool(tuple(getattr(located, "objects", ()) or ()))
            still = None
            if found and image is not None:
                try:
                    still = _shrink(image, self._max_width)[0]
                except Exception as exc:  # noqa: BLE001 (an image the view cannot show: the masks alone, not nothing)
                    logger.debug("live view: the image of the locate could not be shown, the masks go alone: %s", exc)
            self._hand(_Handed("located", rig_id, located, _located_event(located, prompt), at_s=time.time(),
                               still=still))
        except Exception as exc:  # noqa: BLE001 (display only: the located objects are the program's whatever happens)
            logger.debug("live view: located objects not handed over: %s", exc)

    def show_image(self, rig_id: str | None, image: Any, caption: str = "", *, ok: bool | None = None) -> None:
        """Pin ``image`` on the window of ``rig_id`` for ``pin_s`` seconds, in the live image's place, with
        ``caption`` as its line: PNG or JPEG bytes (a pick's ``last_debug_image_png``), or a BGR or grey array, copied
        here. ``ok`` puts the caption on green (``True``) or red (``False``). ``None``, or a rig the view has no window
        for, is its first window."""
        if not self._taking():
            return
        try:
            payload = bytes(image) if isinstance(image, (bytes, bytearray, memoryview)) else _shrink(
                image, self._max_width)[0]
            named = None if rig_id is None else _ascii(str(rig_id))
            self._hand(_Handed("image", named, payload, _ascii(str(caption)), ok, time.time()))
        except Exception as exc:  # noqa: BLE001 (display only)
            logger.debug("live view: image not handed over: %s", exc)

    def note(self, text: str, rig_id: str | None = None) -> None:
        """Say ``text`` as the last thing that happened, on the window of ``rig_id``, or on every window."""
        if not self._taking():
            return
        self._hand(_Handed("note", None if rig_id is None else _ascii(str(rig_id)), None, _ascii(str(text)),
                           None, time.time()))

    def _taking(self) -> bool:
        return self.show and not self._stop.is_set()

    def _hand(self, item: _Handed) -> None:
        with self._lock:
            self._pending.append(item)
            del self._pending[:-_PENDING]

    def _tell(self, line: str) -> None:
        try:
            self._say(line)
        except Exception:  # noqa: BLE001 (nothing left to tell)
            pass

    def _closed(self) -> None:
        with self._lock:
            self._pending = []

    # --- the view's thread --------------------------------------------------------------------------------------------

    def _off_line(self, reason: str) -> str:
        return f"camera windows off after {reason}; {self._goes_on}"

    def _closed_line(self, reason: str) -> str:
        return f"camera windows closed ({reason}): {self._on_close}. The pendant stops the robot"

    def _ctrl_c_line(self) -> str:
        return ("Ctrl+C went to a camera window, not to the console, and stops nothing. The pendant stops the robot; "
                "Ctrl+C in the console stops the program")

    def _run(self) -> None:
        gui = self._gui if self._gui is not None else _preview._highgui()
        windows = self._windows
        guiding: tuple[tuple[str, ...], str, float | None] = ((), "guide", None)
        try:
            while not self._stop.is_set():
                with self._lock:
                    sources = list(self._sources.items())
                    handed, self._pending = self._pending, []
                    now_guiding = self._guiding
                self._open(gui, windows, sources)
                first = next(iter(windows.values()), None)
                if now_guiding != guiding:
                    guiding = now_guiding
                    for window in windows.values():  # the footer changes on every window, the guide on the first
                        window.dirty = True
                now = time.monotonic()
                for item in handed:
                    self._apply(item, windows, first, now)
                for window in windows.values():
                    if window.pinned is not None and now >= window.pinned_until:
                        window.pinned, window.pinned_image, window.held, window.dirty = None, None, "", True
                    showing_image = window.pinned is not None and window.pinned.kind == "image"
                    if (self._period_s is not None and now >= window.next_look and not showing_image
                            and not self._stop.is_set()):
                        self._look(window, now)
                wall = time.time()
                for window in windows.values():
                    if window.dirty or now - window.drawn_at >= _REDRAW_S:
                        gui.imshow(window.title, self._compose(window, wall, guiding, window is first))
                        self.frames_shown += 1
                        window.dirty, window.shown, window.drawn_at = False, True, now
                if not windows:
                    self._stop.wait(_WAIT_MS / 1000.0)  # nothing to draw yet: no message loop to wait in
                    continue
                if self._key(int(gui.waitKey(_WAIT_MS)), bool(guiding[0])):
                    self._switch_off("ESC")
                    break
                visible = getattr(gui, "WND_PROP_VISIBLE", 4)
                closed = next((window for window in windows.values()
                               if window.shown and float(gui.getWindowProperty(window.title, visible)) < 1), None)
                if closed is not None:
                    self._switch_off(f"the window of {closed.rig_id!r} was closed")
                    break
        except Exception as exc:  # noqa: BLE001 (any GUI error switches the view off, never the program)
            self._switch_off(f"{type(exc).__name__}: {exc}", error=True)
        finally:
            self._stop.set()
            for window in list(windows.values()):
                try:
                    gui.destroyWindow(window.title)
                except Exception:  # noqa: BLE001 (the window may already be gone)
                    pass
            if windows:
                try:
                    gui.waitKey(1)
                except Exception:  # noqa: BLE001 (the display may already be gone)
                    pass

    def _open(self, gui: Any, windows: dict[str, _Window], sources: list[tuple[str, Any]]) -> None:
        """A window for every camera watched that has none yet, placed in two columns; a new owner of a rig takes the
        old one's place in its window."""
        for rig_id, source in sources:
            window = windows.get(rig_id)
            if window is None:
                window = _Window(rig_id=rig_id, title=f"{self.title}: {rig_id}", source=source,
                                 mounting=_mounting(source), index=len(windows))
                gui.namedWindow(window.title, getattr(gui, "WINDOW_AUTOSIZE", 1))
                windows[rig_id] = window
                move = getattr(gui, "moveWindow", None)
                if callable(move):
                    column, row = window.index % 2, window.index // 2
                    move(window.title, 20 + column * (self._max_width + 24),
                         20 + row * (self._max_width * 9 // 16 + 160))
            elif window.source is not source:
                window.source, window.mounting, window.dirty = source, _mounting(source), True

    def _apply(self, item: _Handed, windows: dict[str, _Window], first: _Window | None, now: float) -> None:
        """Put what was handed over on its window: a note on the one named or on every window, an image on the one
        named or the first, located objects on their camera's window only, held on the image of the locate where the
        camera moves with the arm. A locate that found nothing is only said, and ends a pin of an earlier locate."""
        if item.kind == "note" and item.rig_id is None:
            targets = list(windows.values())
        else:
            target = windows.get(item.rig_id) if item.rig_id is not None else None
            if target is None and item.kind == "image":
                target = first
            targets = [] if target is None else [target]
        for window in targets:
            window.event, window.dirty = (item.text, item.at_s), True
            if item.kind == "note":
                continue
            image, held = None, ""
            if item.kind == "image":
                try:
                    image = _decoded(item.payload, self._max_width)
                except (cv.error, TypeError, ValueError) as exc:
                    window.event = (f"{item.text or 'an image'}, which cannot be shown ({exc})", item.at_s)
                    continue
            elif not tuple(getattr(item.payload, "objects", ()) or ()):
                if window.pinned is not None and window.pinned.kind == "located":
                    window.pinned, window.pinned_image, window.held = None, None, ""
                continue
            elif _rides_on_the_arm(item.payload, window.mounting):
                image, held = self._held(window, item)
            window.pinned, window.pinned_image, window.pinned_until = item, image, now + self._pin_s
            window.held = held

    def _held(self, window: _Window, item: _Handed) -> tuple[np.ndarray, str]:
        """The image a camera on the arm made a locate on, with what it located drawn over it, and what the status
        line says of it, for the window to hold while the pin lasts: the image handed over with the ``Located``, or,
        where none was, a blank of the live frame's size and a line that says so. Never a frame of the view's own:
        taken before the shutter or after it, it would mark where the objects are not."""
        at = _clock(getattr(item.payload, "captured_at_s", None))
        when = f" at {at}" if at else ""
        if item.still is not None:
            return draw_located(item.still, item.payload), f"as located{when}; the arm may have moved since"
        blank = np.zeros_like(window.live) if window.live is not None else self._blank()
        return (draw_located(blank, item.payload),
                f"located{when}; no image of the locate was handed over, so the masks alone")

    def _blank(self) -> np.ndarray:
        """What a window shows where it has no frame: black, as wide as a frame is shown."""
        return np.zeros((self._max_width * 9 // 16, self._max_width, 3), dtype=np.uint8)

    def _look(self, window: _Window, now: float) -> None:
        """One frame of the window's camera through its display path, or why there is none."""
        try:
            taken = _take(window.source)
        except Exception as exc:  # noqa: BLE001 (said on the window: the camera is the program's)
            error = _ascii(f"{type(exc).__name__}: {exc}")
            if error != window.live_error:
                window.live_error, window.dirty = error, True
            window.next_look = now + _RETRY_S
            return
        if taken is None:  # the rig measures, or has no new image: soon again
            window.next_look = now + _BUSY_RETRY_S
            return
        colour, captured_at_s = taken
        window.live = _shrink(colour, self._max_width)[0]
        window.live_at_s, window.live_error, window.dirty = captured_at_s, "", True
        window.next_look = now + (self._period_s if self._period_s is not None else _RETRY_S)
        self.looks += 1

    def _compose(self, window: _Window, wall: float, guiding: tuple[tuple[str, ...], str, float | None],
                 first: bool) -> np.ndarray:
        """What the window shows now: the pinned image, the image held for what a camera on the arm located (on an
        amber band, a still and not the live frame), or the live frame (what a fixed camera located drawn over it); the
        guide's lines on the first window, and the window's status."""
        lines, tone, bar = guiding if first and guiding[0] else ((), "", None)
        status = [line for line in window.status(wall) if line]
        pinned = window.pinned
        if pinned is not None and pinned.kind == "image" and window.pinned_image is not None:
            base = window.pinned_image
            band = tone or ("counted" if pinned.ok else "rejected" if pinned.ok is False else "judging")
        elif pinned is not None and pinned.kind == "located" and window.pinned_image is not None:
            base, band = window.pinned_image, tone or "judging"
        else:
            base = window.live if window.live is not None else self._blank()
            if pinned is not None and pinned.kind == "located":
                base = draw_located(base, pinned.payload)
            band = tone or "live"
        return annotate(base, lines=[*lines, *status], tone=band, bar=bar if lines else None,
                        footer=_GUIDE_FOOTER if guiding[0] else _FOOTER)
