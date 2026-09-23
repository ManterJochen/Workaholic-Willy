"""A window beside a hand-eye sweep: what the camera sees, and what each judged frame showed.

A sweep that collects nothing looks, from the console, like a solver problem. This window shows the camera's view
while the arm moves, pins each frame the sweep judged with the target drawn on it (the markers with their ids, a
board's chessboard corners and the target's axes), and says for each pose whether it counted and, if not, why, in the
words the console prints for the same event.

It is display only. Nothing here commands motion, gates a sample or changes a pose:

* The judged frame is the one the marker source grabbed after the settle, handed to :meth:`SweepPreview.observe`
  after the estimator posed it (the source's ``on_observation`` hook). The preview keeps a downscaled copy: a
  RealSense colour frame is a view into the device's buffer, and a kept view would hold that buffer.
* Live frames are grabbed through the handle the marker source reads, so every grab goes through the rig's one
  ``Camera`` owner under its lock, as a viewfinder peek does (``api/viewfinder.py``), and the judged grab waits for at
  most one of them. The marker source discards its warm-up grabs before the judged one, so a live grab between them
  changes nothing it relies on. A live frame is posed by the preview's own estimator, never the sweep's, and never
  reaches the calibrator. At most :data:`MAX_LIVE_HZ` a second, and none while a judged frame is pinned.
* The sweep's thread only enqueues: :meth:`SweepPreview.observe` and the event listener put without waiting and drop
  the oldest item when the queue is full. One daemon thread of the preview's own makes every HighGUI call.
* Closing the window or pressing ESC closes the preview and nothing else. It is not a stop: the pendant stops the
  robot, and Ctrl+C in the console stops the program. On Windows, Ctrl+C typed while the window has the focus reaches
  the window, not the console, and the preview says so once.
* An error in the preview's thread switches the preview off with one printed line, and the sweep goes on.

:func:`preview_unavailable` says why no window can be shown in this process, and :func:`annotate` is the drawing,
pure and headless, so it is tested without a window.

Measured on Windows (Win32 HighGUI, OpenCV 4.13) with a synthetic frame source: the window opens, draws and closes
from the preview's thread. Not run on Linux, and never beside a physical camera or controller.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import cv2 as cv
import numpy as np

__all__ = [
    "MAX_LIVE_HZ",
    "NO_PREVIEW_ENV",
    "PreviewState",
    "SweepPreview",
    "annotate",
    "preview_unavailable",
]

logger = logging.getLogger(__name__)

#: Set to anything but empty or ``0``, it keeps the preview off whatever a caller asked for.
NO_PREVIEW_ENV = "WILLY_NO_PREVIEW"
#: The most live frames a second the preview takes from the camera.
MAX_LIVE_HZ = 5.0

# The event names of ``src/robot/events.py`` (``RobotCalibrationEvent``), spelled here so this module imports no robot
# code. A test holds the two spellings together.
_MOVING = "robot_calibration_moving_to_pose"
_DETECTED = "robot_calibration_marker_detected"
_ACCEPTED = "robot_calibration_pose_accepted"
_REJECTED = "robot_calibration_pose_rejected"

_ESC = 27
_CTRL_C = 3
#: How long the window's thread waits for a key per turn. The window stays responsive, and a close is seen within it.
_WAIT_MS = 30
#: How long the window waits before it tries the camera again after a live grab failed.
_RETRY_S = 1.0

_FOOTER = "ESC closes this window only. The pendant stops the robot."

# Colours, BGR.
_TONES = {
    "counted": (50, 140, 50),
    "rejected": (40, 40, 190),
    "judging": (30, 120, 170),
    "live": (120, 80, 30),
}
_STRIP = (32, 32, 32)
_TEXT = (240, 240, 240)
_PENDING = (90, 90, 90)
_MARKERS = (0, 255, 0)
_CORNERS = (255, 0, 255)

_FONT = cv.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_ROW_PX = 20
_PAD_PX = 8
#: Wrapped rows one status line may take; the rest is cut with ``...``.
_MAX_ROWS = 3
_HISTORY_PX = 22


# ---------------------------------------------------------------------------------------------------------------------
# Whether a window can be shown here
# ---------------------------------------------------------------------------------------------------------------------


def preview_unavailable(
    *,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
    build_information: str | None = None,
) -> str | None:
    """Why no preview window can be shown in this process, or ``None`` when one can.

    ``WILLY_NO_PREVIEW`` set to anything but empty or ``0`` keeps it off. So does an OpenCV built without a GUI (the
    headless wheel reports ``GUI: NONE``), and macOS, whose HighGUI wants the main thread, which the sweep holds. On
    Windows an SSH session has no desktop to show a window on. Anywhere else a window needs ``DISPLAY`` or
    ``WAYLAND_DISPLAY``, and asking first matters there: a Qt HighGUI with no display aborts the process rather than
    raise. The three arguments stand in for ``os.environ``, ``sys.platform`` and ``cv2.getBuildInformation()``.
    """
    env = os.environ if environ is None else environ
    if str(env.get(NO_PREVIEW_ENV, "")).strip() not in ("", "0"):
        return f"{NO_PREVIEW_ENV} is set"
    system = sys.platform if platform is None else platform
    if system == "darwin":
        return "on macOS HighGUI wants the main thread, and the sweep holds it"
    info = cv.getBuildInformation() if build_information is None else build_information
    backend = _gui_backend(info)
    if not backend or backend.upper() == "NONE":
        return "this OpenCV has no GUI (GUI: NONE, as the headless wheel is built)"
    if system == "win32":
        if env.get("SSH_CONNECTION") or env.get("SSH_CLIENT"):
            return "an SSH session has no desktop to show a window on"
        return None
    if not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")):
        return "no display: DISPLAY and WAYLAND_DISPLAY are unset"
    return None


def _gui_backend(build_information: str) -> str:
    """The ``GUI:`` line of ``cv2.getBuildInformation()``: ``WIN32UI``, ``QT5``, ``GTK3``, ``NONE`` or empty."""
    match = re.search(r"^\s*GUI:\s*(\S+)", build_information, flags=re.MULTILINE)
    return match.group(1) if match else ""


# ---------------------------------------------------------------------------------------------------------------------
# The drawing
# ---------------------------------------------------------------------------------------------------------------------


def annotate(
    bgr: Any,
    observation: Any = None,
    K: Any = None,
    dist: Any = None,
    *,
    lines: Sequence[str] = (),
    tone: str = "live",
    history: Sequence[bool | None] = (),
    current: int = 0,
    scale: float = 1.0,
    axis_mm: float | None = None,
) -> np.ndarray:
    """``bgr`` with the target drawn on it, a status strip above and a row per pose below, as a new image.

    Pure and headless: ``bgr`` is not changed and no window is touched. ``observation`` is an
    :class:`~src.calibration.targets.Observation`, or ``None``: its markers are outlined with their ids, a board's
    chessboard corners are marked, and a posed target's axes are drawn with ``K`` and ``dist``, ``axis_mm`` long (a
    tenth of the target's distance when not given). ``scale`` is the factor ``bgr`` was shrunk by from the frame the
    observation was made in, so its pixels land where they belong. ``lines`` are the status, the first on a band of
    ``tone`` (``counted``, ``rejected``, ``judging`` or ``live``), and a last line says that ESC closes the window only.
    ``history`` holds one entry per pose of the sweep (``True`` counted, ``False`` rejected, ``None`` not judged yet),
    and pose ``current`` (1-based) is outlined. Text is ASCII: anything else is replaced.

    The result is as wide as ``bgr`` and taller by the strip and the row. An overlay that OpenCV refuses to draw is
    left out rather than raised.
    """
    image = _bgr_copy(bgr)
    if observation is not None:
        _draw_observation(image, observation, K, dist, float(scale), axis_mm)
    parts = [_strip(image.shape[1], lines, tone), image]
    if history:
        parts.append(_history_row(image.shape[1], history, current))
    return np.vstack(parts)


def _bgr_copy(frame: Any) -> np.ndarray:
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return cv.cvtColor(array, cv.COLOR_GRAY2BGR)
    if array.ndim == 3 and array.shape[2] == 3:
        return np.array(array, copy=True)
    raise ValueError(f"a preview frame is grey or BGR, not an array of shape {array.shape}")


def _shrink(frame: Any, max_width: int) -> tuple[np.ndarray, float]:
    """A BGR copy of ``frame`` at most ``max_width`` wide, and the factor it was shrunk by."""
    array = np.asarray(frame)
    width = int(array.shape[1]) if array.ndim >= 2 else 0
    if max_width > 0 and width > max_width:
        factor = max_width / width
        size = (int(max_width), max(1, int(round(array.shape[0] * factor))))
        return _bgr_copy(cv.resize(array, size, interpolation=cv.INTER_AREA)), factor
    return _bgr_copy(array), 1.0


def _draw_observation(image: np.ndarray, observation: Any, K: Any, dist: Any, scale: float,
                      axis_mm: float | None) -> None:
    quads = [(np.asarray(quad, dtype=np.float64).reshape(1, 4, 2) * scale).astype(np.float32)
             for quad in _items(getattr(observation, "marker_corners", ()))]
    if quads:
        ids = _items(getattr(observation, "marker_ids", ()))
        marker_ids = np.asarray(ids, dtype=np.int32).reshape(-1, 1) if len(ids) == len(quads) else None
        try:
            cv.aruco.drawDetectedMarkers(image, quads, marker_ids, _MARKERS)
        except cv.error as exc:
            logger.debug("preview: markers not drawn: %s", exc)
    points = getattr(observation, "points_px", None)
    if getattr(observation, "kind", "") == "charuco" and points is not None and len(points):
        # The corners without their ids: beside the marker ids they would be unreadable.
        corners = (np.asarray(points, dtype=np.float64).reshape(-1, 1, 2) * scale).astype(np.float32)
        try:
            cv.aruco.drawDetectedCornersCharuco(image, corners, None, _CORNERS)
        except cv.error as exc:
            logger.debug("preview: board corners not drawn: %s", exc)
    rvec, tvec = getattr(observation, "rvec", None), getattr(observation, "tvec", None)
    if rvec is None or tvec is None or K is None:
        return
    matrix = np.diag([scale, scale, 1.0]) @ np.asarray(K, dtype=np.float64).reshape(3, 3)
    coefficients = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64).reshape(-1)
    distance = float(np.linalg.norm(np.asarray(tvec, dtype=np.float64)))
    length = float(axis_mm) if axis_mm is not None and axis_mm > 0 else max(10.0, 0.1 * distance)
    try:
        cv.drawFrameAxes(image, matrix, coefficients, np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                         np.asarray(tvec, dtype=np.float64).reshape(3, 1), length, 2)
    except cv.error as exc:
        logger.debug("preview: axes not drawn: %s", exc)


def _items(value: Any) -> tuple[Any, ...]:
    """``value`` as a tuple, ``()`` for ``None``; an array is not asked for its truth."""
    return () if value is None else tuple(value)


def _ascii(text: str) -> str:
    return str(text).encode("ascii", "replace").decode("ascii")


def _text_width(text: str) -> int:
    (width, _), _ = cv.getTextSize(text, _FONT, _FONT_SCALE, 1)
    return int(width)


def _wrap(text: str, width_px: int) -> list[str]:
    """``text`` in rows that fit ``width_px``, at most ``_MAX_ROWS`` of them."""
    rows: list[str] = []
    row = ""
    for word in _ascii(text).split():
        candidate = f"{row} {word}" if row else word
        if _text_width(candidate) <= width_px or not row:
            row = candidate
        else:
            rows.append(row)
            row = word
        while _text_width(row) > width_px and len(row) > 1:
            # A word wider than the window: cut it where it fits.
            cut = max(1, len(row) * width_px // max(1, _text_width(row)))
            rows.append(row[:cut])
            row = row[cut:]
    if row:
        rows.append(row)
    if len(rows) > _MAX_ROWS:
        rows = rows[:_MAX_ROWS]
        rows[-1] = rows[-1][: max(0, len(rows[-1]) - 3)] + "..."
    return rows


def _strip(width: int, lines: Sequence[str], tone: str) -> np.ndarray:
    """The status above the frame: the first line on a band of ``tone``, the rest and the footer on grey."""
    usable = max(40, width - 2 * _PAD_PX)
    rows: list[tuple[str, bool]] = []
    for number, line in enumerate(line for line in lines if line):
        rows += [(row, number == 0) for row in _wrap(line, usable)]
    rows += [(row, False) for row in _wrap(_FOOTER, usable)]
    height = _PAD_PX // 2 + _ROW_PX * len(rows) + _PAD_PX // 2
    strip = np.full((height, width, 3), _STRIP, dtype=np.uint8)
    banded = sum(1 for _, first in rows if first)
    if banded:
        strip[: _PAD_PX // 2 + _ROW_PX * banded + 2] = _TONES.get(tone, _TONES["live"])
    for number, (row, _) in enumerate(rows):
        baseline = _PAD_PX // 2 + _ROW_PX * (number + 1) - 6
        cv.putText(strip, row, (_PAD_PX, baseline), _FONT, _FONT_SCALE, _TEXT, 1, cv.LINE_AA)
    return strip


def _history_row(width: int, history: Sequence[bool | None], current: int) -> np.ndarray:
    """One box per pose: green counted, red rejected, grey not judged yet; the current pose outlined."""
    row = np.full((_HISTORY_PX, width, 3), _STRIP, dtype=np.uint8)
    step = max(3, min(24, (width - 2 * _PAD_PX) // max(1, len(history))))
    for number, verdict in enumerate(history):
        x0 = _PAD_PX + number * step
        if x0 + step > width:
            break
        colour = _TONES["counted"] if verdict is True else _TONES["rejected"] if verdict is False else _PENDING
        cv.rectangle(row, (x0 + 1, 5), (x0 + step - 2, _HISTORY_PX - 6), colour, -1)
        if number + 1 == current:
            cv.rectangle(row, (x0, 3), (x0 + step - 1, _HISTORY_PX - 4), _TEXT, 1)
    return row


# ---------------------------------------------------------------------------------------------------------------------
# What the window says, from the sweep's events
# ---------------------------------------------------------------------------------------------------------------------


def _line(event_type: str, data: Mapping[str, Any]) -> str:
    """A short line for one event, for a preview built without the console's wording."""
    index, total = data.get("index"), data.get("total")
    head = f"pose {index}/{total}" if isinstance(index, int) and isinstance(total, int) else "pose"
    if data.get("label"):
        head += f" {str(data['label'])!r}"
    if event_type == _MOVING:
        return f"{head}  moving, {data.get('accepted', 0)} counted so far"
    if event_type == _DETECTED:
        return f"{head}  target seen"
    if event_type == _ACCEPTED:
        return f"{head}  COUNTED {data.get('accepted', '?')}"
    if event_type == _REJECTED:
        detail = data.get("detail") or data.get("why_not") or ""
        return f"{head}  REJECTED {data.get('reason') or 'rejected'}" + (f": {detail}" if detail else "")
    return f"{head}  {event_type}"


def _seen(observation: Any) -> str:
    """What a frame showed of the target, in one line: what the pose rests on, or why there is none."""
    if observation is None:
        return ""
    if getattr(observation, "T_cam_to_target", None) is not None:
        summary = observation.summary() if callable(getattr(observation, "summary", None)) else "target posed"
        hint = str(getattr(observation, "hint", "") or "")
        return f"in view: {summary}" + (f"; !! {hint}" if hint else "")
    return str(getattr(observation, "why_not", "") or "the target was not posed")


@dataclass
class PreviewState:
    """What the window says about the sweep, built from its events alone. Pure: no window, no thread.

    ``describe`` words one event as a line; the sweep passes the console's (``render_sweep_event``), so the window and
    the console say the same thing about a pose.
    """

    describe: Callable[[str, Mapping[str, Any]], str] = _line
    total: int = 0
    #: The pose the sweep is at, 1-based; 0 before the first.
    index: int = 0
    accepted: int = 0
    need: int | None = None
    labels: dict[int, str] = field(default_factory=dict)
    #: The latest line for each pose.
    said: dict[int, str] = field(default_factory=dict)
    #: The verdict line of each judged pose, and whether it counted.
    verdicts: dict[int, str] = field(default_factory=dict)
    counted: dict[int, bool] = field(default_factory=dict)
    last_verdict: str = ""

    def apply(self, event_type: str, data: Mapping[str, Any]) -> None:
        """Take one event of the sweep, as ``CalibrationRoutine`` emits it."""
        index, total = data.get("index"), data.get("total")
        if isinstance(total, int) and not isinstance(total, bool):
            self.total = total
        if isinstance(index, int) and not isinstance(index, bool):
            self.index = index
            self.labels[index] = str(data.get("label") or "")
        if isinstance(data.get("accepted"), int):
            self.accepted = int(data["accepted"])
        if isinstance(data.get("min_samples"), int):
            self.need = int(data["min_samples"])
        line = _ascii(self.describe(event_type, data))
        self.said[self.index] = line
        if event_type in (_ACCEPTED, _REJECTED):
            self.verdicts[self.index] = line
            self.counted[self.index] = event_type == _ACCEPTED
            self.last_verdict = line

    def history(self) -> list[bool | None]:
        """One entry per pose of the sweep: ``True`` counted, ``False`` rejected, ``None`` not judged yet."""
        return [self.counted.get(number) for number in range(1, max(self.total, self.index) + 1)]

    def count_line(self) -> str:
        need = f", {self.need} needed" if self.need is not None else ""
        poses = f", {self.total} poses in this sweep" if self.total else ""
        return f"{self.accepted} counted{need}{poses}"

    def judged_lines(self, index: int, observation: Any) -> list[str]:
        """The strip over a judged frame: the pose's verdict, what the frame showed, and the count."""
        verdict = self.verdicts.get(index)
        if verdict:
            head = f"JUDGED  {verdict}"
        else:
            label = self.labels.get(index, "")
            head = f"JUDGED  pose {index}/{self.total}" + (f" {label!r}" if label else "") + "  judging"
        return [head, _seen(observation), self.count_line()]

    def judged_tone(self, index: int) -> str:
        counted = self.counted.get(index)
        return "judging" if counted is None else "counted" if counted else "rejected"

    def live_lines(self, observation: Any, error: str = "") -> list[str]:
        """The strip over a live frame: where the sweep is, what the camera sees now, the last verdict, the count."""
        doing = self.said.get(self.index, "") if self.index else "waiting for the sweep's first pose"
        seen = f"live view: {error}" if error else _seen(observation)
        last = f"last verdict: {self.last_verdict}" if self.last_verdict and self.last_verdict != doing else ""
        return [f"LIVE, not judged  {doing}", seen, last, self.count_line()]


# ---------------------------------------------------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class _Frame:
    """A frame the window shows: a downscaled copy, what was seen in it, and the lens it was seen with."""

    image: np.ndarray
    observation: Any
    K: np.ndarray | None
    dist: np.ndarray | None
    scale: float
    #: The pose it was judged for; 0 for a live frame.
    index: int = 0


def _highgui() -> Any:
    """The module whose window calls the preview makes: OpenCV's HighGUI."""
    return cv


def _say(line: str) -> None:
    print(f"  {_ascii(line)}", flush=True)


class SweepPreview:
    """A window beside a sweep, owned by one daemon thread that makes every HighGUI call.

    ``handle`` is what the sweep's marker source reads (a ``RigHandle``, anything with ``grab()``,
    ``get_intrinsics()`` and ``get_distortion()``); live frames come through it and through nothing else.
    ``estimator`` poses the target in live frames and must be the preview's own: the single-marker estimator keeps a
    grey buffer between frames, so one shared with the sweep would be read by two threads. ``None`` shows live frames
    without a detection. ``describe`` words an event as the console does. ``live_hz`` is capped at
    :data:`MAX_LIVE_HZ`, and 0 takes no live frames. A judged frame stays pinned for ``pin_s`` seconds, or until the
    next one. ``gui`` stands in for ``cv2`` and ``say`` for the one printed line a switch-off gets.

    Building it touches no window and no camera. :meth:`start` opens the window on the preview's thread, and
    :meth:`close` stops it; the sweep starts it once the arm is connected and closes it before the camera is given
    back. :meth:`observe` is the marker source's ``on_observation`` hook and the instance is an event listener; both
    return at once and never raise.
    """

    def __init__(
        self,
        handle: Any,
        *,
        title: str = "willy hand-eye sweep",
        estimator: Any = None,
        describe: Callable[[str, Mapping[str, Any]], str] | None = None,
        axis_mm: float | None = None,
        live_hz: float = MAX_LIVE_HZ,
        max_width: int = 960,
        pin_s: float = 1.5,
        queue_size: int = 16,
        gui: Any = None,
        say: Callable[[str], None] | None = None,
    ) -> None:
        self._handle = handle
        self.title = _ascii(title)
        self._estimator = estimator
        self._describe = describe if describe is not None else _line
        self._axis_mm = axis_mm
        rate = min(float(live_hz), MAX_LIVE_HZ)
        self._live_period_s: float | None = 1.0 / rate if rate > 0 else None
        self._max_width = int(max_width)
        self._pin_s = float(pin_s)
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, int(queue_size)))
        self._gui = gui
        self._say = say if say is not None else _say
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._K: np.ndarray | None = None
        self._dist: np.ndarray | None = None
        #: Why the preview switched itself off before it was closed: the operator closed it, or an error. Empty
        #: while it shows, and after a plain :meth:`close`.
        self.off = ""
        #: Items the queue dropped because it was full, oldest first.
        self.dropped = 0
        self.live_grabs = 0
        self.frames_shown = 0

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    # --- the sweep's side: enqueue and return -------------------------------------------------------------------

    def start(self) -> None:
        """Open the window on the preview's own thread. Idempotent, and it never raises: a preview that cannot start
        is off, and the sweep goes on."""
        with self._lock:
            if self._thread is not None or self._stop.is_set():
                return
            thread = threading.Thread(target=self._run, name="sweep-preview", daemon=True)
            try:
                thread.start()
            except Exception as exc:  # noqa: BLE001 (a preview is never a reason to stop a sweep)
                self._stop.set()
                self._switch_off(f"its thread did not start: {type(exc).__name__}: {exc}", error=True)
                return
            self._thread = thread

    def __call__(self, event_type: str, data: Mapping[str, Any]) -> None:
        """The sweep's event listener: queues the event for the window."""
        if self._stop.is_set():
            return
        try:
            self._offer((str(event_type), dict(data)))
        except Exception as exc:  # noqa: BLE001 (a listener must not throw)
            logger.debug("preview: event not queued: %s", exc)

    def observe(self, bgr: Any, observation: Any, K: Any, dist: Any) -> None:
        """The marker source's ``on_observation`` hook: the judged frame, copied and downscaled, queued to pin."""
        if self._stop.is_set():
            return
        try:
            image, factor = _shrink(bgr, self._max_width)
            lens = None if K is None else np.array(K, dtype=np.float64)
            coefficients = None if dist is None else np.array(dist, dtype=np.float64).reshape(-1)
            self._offer(_Frame(image, observation, lens, coefficients, factor))
        except Exception as exc:  # noqa: BLE001 (display only: the pose is the estimator's whatever happens here)
            logger.debug("preview: judged frame not queued: %s", exc)

    def close(self, timeout_s: float = 2.0) -> None:
        """Stop the thread and close the window, waiting at most ``timeout_s``. Idempotent, and it never raises.

        The sweep calls it before the camera is given back, so no live grab meets a closed camera: the thread takes
        no new grab once it is asked to stop. A thread stuck in a GUI call past the timeout is left behind as a
        daemon, and the process does not wait for it.
        """
        try:
            with self._lock:
                self._stop.set()
                thread = self._thread
            if thread is not None and thread is not threading.current_thread():
                thread.join(max(0.0, float(timeout_s)))
                if thread.is_alive():
                    logger.warning("The sweep preview did not close within %.1f s; its thread is left behind.",
                                   timeout_s)
            self._drain()
        except Exception as exc:  # noqa: BLE001 (closing a window must not stop the teardown it is part of)
            logger.warning("Closing the sweep preview raised %s: %s", type(exc).__name__, exc)

    def _offer(self, item: Any) -> None:
        """Put without waiting. When the queue is full its oldest item goes first, so the sweep never blocks here."""
        for _ in range(4):
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass
        self.dropped += 1

    def _drain(self) -> list[Any]:
        items: list[Any] = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                return items

    # --- the window's thread ------------------------------------------------------------------------------------

    def _switch_off(self, reason: str, *, error: bool = False) -> None:
        self.off = reason
        if error:
            line = f"preview off after {reason}; the sweep goes on"
            logger.warning("Sweep preview switched off: %s", reason)
        else:
            line = f"preview closed ({reason}). The sweep goes on; the pendant stops the robot"
            logger.info("Sweep preview closed: %s", reason)
        try:
            self._say(line)
        except Exception:  # noqa: BLE001 (nothing left to tell)
            pass

    def _run(self) -> None:
        gui = self._gui if self._gui is not None else _highgui()
        state = PreviewState(describe=self._describe)
        pinned: _Frame | None = None
        pinned_until = 0.0
        live: _Frame | None = None
        live_error = ""
        next_live = 0.0
        opened = shown = told_ctrl_c = False
        try:
            gui.namedWindow(self.title, getattr(gui, "WINDOW_AUTOSIZE", 1))
            opened = dirty = True
            while not self._stop.is_set():
                now = time.monotonic()
                for item in self._drain():
                    if isinstance(item, _Frame):
                        # The judged frame follows its pose's move event in the queue, so the pose is the current one.
                        pinned = _Frame(item.image, item.observation, item.K, item.dist, item.scale, state.index)
                        pinned_until = now + self._pin_s
                    else:
                        state.apply(*item)
                    dirty = True
                if pinned is not None and now >= pinned_until:
                    pinned, dirty = None, True
                if (pinned is None and self._live_period_s is not None and now >= next_live
                        and not self._stop.is_set()):
                    live, live_error = self._grab_live()
                    next_live = now + (self._live_period_s if not live_error else _RETRY_S)
                    dirty = True
                if dirty:
                    gui.imshow(self.title, self._compose(state, pinned, live, live_error))
                    self.frames_shown += 1
                    shown, dirty = True, False
                key = int(gui.waitKey(_WAIT_MS))
                if key != -1 and key & 0xFF == _ESC:
                    self._switch_off("ESC")
                    break
                if key != -1 and key & 0xFF == _CTRL_C and not told_ctrl_c:
                    told_ctrl_c = True
                    self._say("Ctrl+C went to the preview window, not to the console, and the sweep goes on. "
                              "The pendant stops the robot; Ctrl+C in the console stops the program")
                if shown and float(gui.getWindowProperty(self.title, getattr(gui, "WND_PROP_VISIBLE", 4))) < 1:
                    self._switch_off("its window was closed")
                    break
        except Exception as exc:  # noqa: BLE001 (any GUI error switches the preview off, never the sweep)
            self._switch_off(f"{type(exc).__name__}: {exc}", error=True)
        finally:
            self._stop.set()
            if opened:
                try:
                    gui.destroyWindow(self.title)
                    gui.waitKey(1)
                except Exception:  # noqa: BLE001 (the window may already be gone)
                    pass
            self._drain()

    def _grab_live(self) -> tuple[_Frame | None, str]:
        """One display-only frame through the sweep's handle, posed by the preview's own estimator, or why not."""
        try:
            frame = self._handle.grab()
            self.live_grabs += 1
            colour = np.asarray(getattr(frame, "color", frame))
            K, dist = self._lens()
            observation = None
            if self._estimator is not None and K is not None:
                try:
                    observation = self._estimator.observe(colour, K, dist)
                except Exception as exc:  # noqa: BLE001 (a live detection is display only)
                    logger.debug("preview: live detection failed: %s", exc)
            image, factor = _shrink(colour, self._max_width)
            return _Frame(image, observation, K, dist, factor), ""
        except Exception as exc:  # noqa: BLE001 (said on the window; the judged grab is the sweep's own)
            return None, f"{type(exc).__name__}: {exc}"

    def _lens(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """The camera matrix and distortion, read once through the handle; ``(None, None)`` when it gives none."""
        if self._K is None:
            try:
                matrix = self._handle.get_intrinsics()
                if matrix is not None:
                    read = getattr(self._handle, "get_distortion", None)
                    coefficients = read() if callable(read) else None
                    self._dist = (np.zeros(5) if coefficients is None
                                  else np.asarray(coefficients, dtype=np.float64).reshape(-1))
                    self._K = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
            except Exception as exc:  # noqa: BLE001 (live frames are then shown without a detection)
                logger.debug("preview: no camera matrix: %s", exc)
        return self._K, self._dist

    def _compose(self, state: PreviewState, pinned: _Frame | None, live: _Frame | None, live_error: str) -> np.ndarray:
        history = state.history()
        if pinned is not None:
            return annotate(pinned.image, pinned.observation, pinned.K, pinned.dist,
                            lines=state.judged_lines(pinned.index, pinned.observation),
                            tone=state.judged_tone(pinned.index), history=history, current=pinned.index,
                            scale=pinned.scale, axis_mm=self._axis_mm)
        if live is not None and not live_error:
            return annotate(live.image, live.observation, live.K, live.dist, lines=state.live_lines(live.observation),
                            tone="live", history=history, current=state.index, scale=live.scale,
                            axis_mm=self._axis_mm)
        width = self._max_width if self._max_width > 0 else 960
        blank = np.zeros((width * 9 // 16, width, 3), dtype=np.uint8)
        waiting = live_error or ("waiting for the first frame" if self._live_period_s is not None
                                 else "no live frames; each judged frame is shown as it is taken")
        return annotate(blank, lines=state.live_lines(None, waiting), tone="live", history=history,
                        current=state.index)
