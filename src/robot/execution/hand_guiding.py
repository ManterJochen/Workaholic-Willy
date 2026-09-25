"""Hand guiding for a calibration: a person moves the arm, the software watches and captures on Enter.

The hand-guided calibration (``CalibrationRoutine.run_freedrive``), the adjust step of fixed stations
(``run_with_poses(adjust=True)``) and poses taught by hand (:mod:`.teach`) all hand the arm to a person through the
vendor capability :class:`~src.robot.core.freedrive.SupportsFreedrive`. What they share lives here:

* :class:`HandGuide` runs the console side. ``confirm_payload`` shows the payload the controller compensates for
  and asks once whether it is right, before anything is freed. ``ask`` puts one question while the arm holds and
  returns the line typed (a taught pose's name). ``wait`` polls the session's samples at
  :data:`RATE_HZ` (which keeps the vendor's crash watchdog fed) until a person presses Enter, in the console or in
  the preview window, and then passes the stillness gate. ``hands_off`` asks for the hands off the arm and counts
  down before the arm drives by itself again; only a yes starts the countdown, and the console and the preview are
  read all through it, so a stop typed during it works.
* :class:`HandGuidingLimits` holds the cable window (the joint-limit guard's window, half a turn about home where
  the cell keeps it) and the workspace box. A sample outside either is shown in red and Enter does not capture
  there. It never holds, locks or stops the arm: an arm that locks while a person pushes it is how a hand gets
  caught, so the only answer to a boundary is a sentence saying which joint or face and which way back.
* :func:`offset_in_tool` says how far the TCP still is from a target station, along the tool's own axes, which is
  how a person holding the tool reads a direction.

Threads: everything that touches the arm runs on the caller's thread, one call after another. The terminal's
reader thread only reads lines, and the preview's thread only draws and reads keys; neither calls the arm (the UR
control interface is not documented as thread-safe). The preview and the console meet this module through
:class:`GuideView` and :class:`OperatorConsole`, both polled, never waited on.

Keys: Enter captures (the console's empty line, or Enter or Space in the preview), ``s`` skips the station or
target, ``q`` finishes. ESC closes the preview, and a preview that is closed finishes the run at the next prompt,
also when it was closed while the arm drove by itself; the session then holds the arm as it always does when it is
left. During the countdown before the arm drives by itself, ``q`` (console or preview) and ESC finish before
anything moves, and anything else typed stops the countdown and asks again.
"""

from __future__ import annotations

import math
import queue
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from src.geometry import Frame, Pose
from src.geometry.quaternion import from_axis_angle
from src.robot.core.freedrive import ControllerPayload, FreedriveSample, FreedriveSession
from src.robot.execution.pose_provider import JOINT_NAMES

__all__ = [
    "CAPTURE",
    "FINISH",
    "SKIP",
    "GuideView",
    "HandGuide",
    "HandGuidingLimits",
    "HandGuidingRefused",
    "OperatorConsole",
    "TerminalConsole",
    "ToolOffset",
    "offset_in_tool",
    "sample_pose",
]

#: How often a free arm is sampled, per second. The vendor's watchdog wants 30 or more; this leaves room.
RATE_HZ = 50.0
#: The fastest joint, in rad/s, below which a hand-held arm counts as still (about 1.1 deg/s).
STILL_RAD_S = 0.02
#: How long the arm has to stay that still before a capture.
STILL_FOR_S = 0.5
#: How long the stillness gate waits before it gives the capture back to the person.
STILL_WAIT_S = 5.0
#: The countdown before the arm drives by itself again, in whole seconds.
COUNTDOWN_S = 3

#: What :meth:`HandGuide.wait` answers.
CAPTURE, SKIP, FINISH = "capture", "skip", "finish"
#: What a console answers when its input has ended: nobody can press Enter any more.
EOF = "\x04"

_KEYS = "Enter captures, s skips, q finishes"
#: What finishes a run when it is typed: in the console, and input that has ended.
_FINISH_WORDS = ("q", "quit", "finish", EOF)
#: The question before the arm drives by itself again.
_HANDS_OFF = "Hands off the arm - Enter to continue (q finishes here)"


class HandGuidingRefused(Exception):
    """Hand guiding cannot start or go on: the arm offers none, the payload was not confirmed, or nobody answers.

    Not a ``RuntimeError``, so a sweep's per-station recovery never takes it for a station's fault.
    """


class OperatorConsole(Protocol):
    """Where the person reads and types. ``poll`` returns a line typed since the last call, or ``None``; never waits."""

    def say(self, line: str) -> None: ...

    def bell(self) -> None: ...

    def poll(self) -> str | None: ...


class GuideView(Protocol):
    """A window beside the guiding: what it shows, and the keys typed into it (``enter``, ``skip``, ``finish``,
    ``closed``). Both calls return at once; the window's own thread does the drawing."""

    def guide(self, lines: Sequence[str], tone: str, bar: float | None = None) -> None: ...

    def poll_key(self) -> str | None: ...


class TerminalConsole:
    """The operator's terminal: lines printed to stdout, lines read from stdin by a daemon thread.

    The thread starts on the first :meth:`poll` and only reads; a line is handed over through a queue, so the loop
    that samples the arm never waits on the keyboard. Input that has ended (a closed or piped stdin) answers
    :data:`EOF`, which refuses a prompt rather than taking silence for a yes.
    """

    def __init__(self, stdin: Any = None, stdout: Any = None) -> None:
        self._in = stdin
        self._out = stdout
        self._lines: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._reader: threading.Thread | None = None
        self._lock = threading.Lock()
        self._ended = False

    def say(self, line: str) -> None:
        print(f"  {line}", file=self._out or sys.stdout, flush=True)

    def bell(self) -> None:
        out = self._out or sys.stdout
        out.write("\a")
        out.flush()

    def poll(self) -> str | None:
        with self._lock:
            if self._reader is None:
                self._reader = threading.Thread(target=self._read, name="operator-console", daemon=True)
                self._reader.start()
        if self._ended:
            return EOF
        try:
            line = self._lines.get_nowait()
        except queue.Empty:
            return None
        self._ended = line == EOF
        return line

    def _read(self) -> None:
        stream = self._in or sys.stdin
        while True:
            try:
                line = stream.readline()
            except (OSError, ValueError):
                line = ""
            if not line:
                self._lines.put(EOF)
                return
            self._lines.put(line.rstrip("\r\n"))


@dataclass(frozen=True, slots=True)
class ToolOffset:
    """How far a target still is from the TCP: millimetres along, and degrees about, the TCP's own axes."""

    along_mm: tuple[float, float, float]
    about_deg: tuple[float, float, float]

    @property
    def distance_mm(self) -> float:
        return float(np.linalg.norm(self.along_mm))

    @property
    def angle_deg(self) -> float:
        return float(np.linalg.norm(self.about_deg))

    @property
    def closeness(self) -> float:
        """1 at the target, 0 at 150 mm or 30 degrees away or more: the bar the preview draws."""
        return max(0.0, 1.0 - max(self.distance_mm / 150.0, self.angle_deg / 30.0))

    def line(self) -> str:
        x, y, z = self.along_mm
        a, b, c = self.about_deg
        return (f"move {x:+.0f} / {y:+.0f} / {z:+.0f} mm along tool x / y / z, turn {a:+.0f} / {b:+.0f} / {c:+.0f} "
                f"deg about them ({self.distance_mm:.0f} mm, {self.angle_deg:.0f} deg in all)")


def sample_pose(sample: FreedriveSample) -> Pose:
    """The TCP a sample reports, as a BASE pose."""
    return Pose(position_mm=np.asarray(sample.tcp_xyz_mm, dtype=np.float64),
                quaternion_xyzw=from_axis_angle(np.asarray(sample.tcp_rotvec_rad, dtype=np.float64)),
                frame=Frame.BASE)


def offset_in_tool(sample: FreedriveSample, target: Pose) -> ToolOffset:
    """Where ``target`` stands from the TCP of ``sample``, in the TCP's own frame."""
    now = sample_pose(sample).to_matrix()
    relative = np.linalg.inv(now) @ target.to_matrix()
    about = Pose.from_matrix(relative, frame=Frame.BASE).axis_angle_rad()
    return ToolOffset(along_mm=tuple(float(v) for v in relative[:3, 3]),  # type: ignore[arg-type]
                      about_deg=tuple(float(math.degrees(v)) for v in about))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class HandGuidingLimits:
    """The cable window and the workspace box a hand-guided sample is shown against, read once from the arm.

    ``lower_deg``/``upper_deg`` are the joint-limit guard's window less its margin, the numbers a joint move to a
    captured station would be refused outside of; empty where the arm carries no joint-limit guard or it resolves
    no table, and :attr:`watched` then says so. ``box`` is the workspace box (``x_min`` ... ``z_max``, mm).
    """

    box: Any
    lower_deg: tuple[float, ...] = ()
    upper_deg: tuple[float, ...] = ()
    #: One clause on where the window came from, for the console.
    window: str = ""

    @classmethod
    def of(cls, arm: Any, box: Any) -> "HandGuidingLimits":
        """The window the arm's own joint-limit guard keeps (``safety_preflight``), and ``box``."""
        preflight = getattr(arm, "safety_preflight", None)
        guard = next((g for g in (getattr(preflight, "guards", ()) or ()) if getattr(g, "name", None) == "joint_limit"),
                     None)
        limits = None
        for_arm = getattr(guard, "limits_for_arm", None)
        if callable(for_arm):
            try:
                limits = for_arm(arm)
            except Exception:  # noqa: BLE001 (no window read is said below, never a reason to stop)
                limits = None
        if limits is None:
            return cls(box=box, window="no joint-limit window is configured on this arm, so only the workspace box "
                                      "is watched")
        margin = float(getattr(guard, "margin_deg", 0.0) or 0.0)
        half = bool(getattr(guard, "within_half_turn_of_home", False))
        return cls(box=box, lower_deg=tuple(float(v) + margin for v in limits[0]),
                   upper_deg=tuple(float(v) - margin for v in limits[1]),
                   window=("the cable window, half a turn either side of home" if half else "the joint limits")
                   + f" less the {margin:g} deg margin, and the workspace box")

    def outside(self, sample: FreedriveSample) -> str:
        """Why ``sample`` stands outside, with the way back, or ``""`` when it is inside both."""
        return self.crossed(sample)[1]

    def crossed(self, sample: FreedriveSample) -> tuple[str, str]:
        """``(which boundary, why)`` for a sample outside, ``("", "")`` inside both.

        ``which`` names the joint and end, or the box face (``wrist 3 upper``, ``box z_min``), and stays the same
        while the arm moves past it, so a console says it once; ``why`` has the numbers and the way back.
        """
        for axis, (value, low, high) in enumerate(zip(sample.joints_rad, self.lower_deg, self.upper_deg)):
            degrees = math.degrees(float(value))
            if not low <= degrees <= high:
                name = JOINT_NAMES[axis] if axis < len(JOINT_NAMES) else f"joint {axis + 1}"
                way = "back down" if degrees > high else "back up"
                return (f"{name} {'upper' if degrees > high else 'lower'}",
                        f"{name} stands at {degrees:.1f} deg, outside the cable window {low:.1f} to {high:.1f} deg: "
                        f"turn it {way} inside it")
        for name, value in zip("xyz", sample.tcp_xyz_mm):
            start = float(getattr(self.box, f"{name}_min"))
            end = float(getattr(self.box, f"{name}_max"))
            if float(value) < start:
                return (f"box {name}_min",
                        f"the TCP stands {start - float(value):.0f} mm past the workspace box's {name}_min face "
                        f"({name} {float(value):.1f} mm, the box starts at {start:.1f} mm): bring it back inside")
            if float(value) > end:
                return (f"box {name}_max",
                        f"the TCP stands {float(value) - end:.0f} mm past the workspace box's {name}_max face "
                        f"({name} {float(value):.1f} mm, the box ends at {end:.1f} mm): bring it back inside")
        return "", ""


class HandGuide:
    """The console side of hand guiding: payload, waiting for Enter, stillness, hands off and the countdown.

    ``console`` is where the person types (:class:`TerminalConsole` by default), ``view`` the preview window or
    ``None``. ``clock`` and ``sleep`` are the time it runs on, so a test runs it on a clock of its own.
    ``touched`` is set once a session was waited on, and :meth:`hands_off` clears it.
    """

    def __init__(
        self,
        console: OperatorConsole | None = None,
        view: GuideView | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        rate_hz: float = RATE_HZ,
        still_rad_s: float = STILL_RAD_S,
        still_for_s: float = STILL_FOR_S,
        still_wait_s: float = STILL_WAIT_S,
    ) -> None:
        self.console: OperatorConsole = console if console is not None else TerminalConsole()
        self.view = view
        self.clock = clock
        self.sleep = sleep
        self.period_s = 1.0 / max(1.0, float(rate_hz))
        self.still_rad_s = float(still_rad_s)
        self.still_for_s = float(still_for_s)
        self.still_wait_s = float(still_wait_s)
        self.touched = False
        self.payload_confirmed = False
        #: The view asked to finish (``q``, or it was closed) outside a prompt; the next prompt finishes.
        self._view_finished = False

    # --- before anything is freed ---------------------------------------------------------------------------

    def confirm_payload(self, arm: Any) -> None:
        """Show the payload the controller compensates for and ask whether it is right; once per guide.

        A wrong payload makes a freed arm sink or rise in a person's hands. Enter (or ``y``) confirms; anything
        else, and input that has ended, raises :class:`HandGuidingRefused` with nothing freed.
        """
        if self.payload_confirmed:
            return
        try:
            payload: ControllerPayload | None = arm.controller_payload()
        except Exception as exc:  # noqa: BLE001 (said to the person, who checks it on the pendant)
            payload = None
            self.console.say(f"The controller payload could not be read ({type(exc).__name__}: {exc}).")
        if payload is None:
            self.console.say("The payload the controller compensates for cannot be read from this arm: check it on the "
                             "pendant before you free the arm.")
        else:
            x, y, z = payload.cog_mm
            self.console.say(f"The controller compensates for a payload of {payload.mass_kg:.2f} kg, its centre of "
                             f"gravity at ({x:.0f}, {y:.0f}, {z:.0f}) mm from the flange. A wrong payload makes the "
                             "freed arm sink or rise in your hands.")
        answer = self._ask("Is this payload right (hand + camera + bracket)? Enter = yes, n = no")
        if answer.strip().lower() not in ("", "y", "yes"):
            raise HandGuidingRefused(
                "the controller payload was not confirmed, so the arm was not freed: set the payload of the hand, the "
                "camera and its bracket on the pendant (Installation, Payload) and run this again")
        self.payload_confirmed = True

    def ask(self, question: str) -> str:
        """``question`` on the console (and in the preview), then the line the person types, as typed.

        What was typed before the question is forgotten first, so an Enter meant for the last prompt does not answer
        this one. Enter in the preview answers ``""``; ``q`` there, or a preview that was closed, answers ``"q"``;
        input that has ended answers :data:`EOF` (and one that ended before the question raises
        :class:`HandGuidingRefused`). Ask only while the arm holds: nothing samples a session while a question waits,
        and a free arm is sampled by :meth:`wait` alone, which is what keeps the vendor's watchdog fed.
        """
        return self._ask(question)

    # --- while a person guides the arm ----------------------------------------------------------------------

    def wait(
        self,
        session: FreedriveSession,
        limits: HandGuidingLimits,
        *,
        banner: str,
        status: Callable[[FreedriveSample], tuple[list[str], float | None]] | None = None,
    ) -> str:
        """Sample the free arm until a person captures (:data:`CAPTURE`), skips or finishes; the arm stays free.

        Every sample is checked against ``limits``. Outside, the view turns red with the sentence and Enter does
        not capture; the arm is never held for it. ``status`` adds lines (and a closeness bar) for the view from
        the latest sample. A capture has passed the stillness gate: the fastest joint below ``still_rad_s`` for
        ``still_for_s``. The arm is still free when this returns: the caller holds it.
        """
        self.touched = True
        self._drain()
        self.console.say(f"{banner} ({_KEYS}.)")
        self.console.bell()
        said = ""
        while True:
            sample = session.sample()
            which, outside = limits.crossed(sample)
            if which != said:
                if outside:
                    self.console.say(f"OUTSIDE: {outside}. Enter does not capture here.")
                elif said:
                    self.console.say("Back inside.")
                said = which
            self._show(banner, outside, sample, status)
            key = self._key()
            if key in (FINISH, SKIP):
                self._clear()
                return key
            if key == CAPTURE:
                if outside:
                    self.console.say(f"Not captured: {outside}.")
                elif self._still(session, limits):
                    self._clear()
                    return CAPTURE
            self.sleep(self.period_s)

    def _still(self, session: FreedriveSession, limits: HandGuidingLimits) -> bool:
        """The stillness gate: ``True`` once the fastest joint stayed below the threshold long enough."""
        started = self.clock()
        since: float | None = None
        peak = 0.0
        if self.view is not None:
            self.view.guide(["HOLD THE ARM STILL: capturing once it stands still"], "guide")
        while True:
            sample = session.sample()
            now = self.clock()
            outside = limits.outside(sample)
            if outside:
                self.console.say(f"Not captured: {outside}.")
                return False
            speed = sample.peak_joint_speed_rad_s
            peak = max(peak, speed)
            if speed < self.still_rad_s:
                since = now if since is None else since
                if now - since >= self.still_for_s:
                    return True
            else:
                since = None
            if now - started >= self.still_wait_s:
                self.console.say(f"Not captured: the arm still moves ({math.degrees(peak):.1f} deg/s at most); hold it "
                                 "still and press Enter again.")
                return False
            self.sleep(self.period_s)

    # --- before the arm drives by itself again --------------------------------------------------------------

    def hands_off(self) -> bool:
        """Ask for the hands off the arm, then count down; ``False`` when the person finishes instead.

        Asked only after a session was waited on (``touched``), and cleared once a countdown ran out. Only a yes
        starts the countdown: Enter (the console's empty line, or Enter in the preview), or ``y``/``yes``. ``q``,
        ``quit``, ``finish``, input that has ended, ``q`` in the preview and a closed preview finish. Any other
        answer (``n``, ``stop``, ``wait``, ``s``) is said not to be one and the question is asked again: an arm
        must never drive because an answer was not understood.

        The countdown keeps reading the console and the preview every :attr:`period_s`, so a stop typed during it
        works: ``q`` in either, or ESC closing the window, finishes before anything moves, and anything else
        typed stops the countdown and asks again. Ctrl-C in the console raises ``KeyboardInterrupt``, also before
        anything moves; Ctrl-C typed into the window reaches no program, and the window says so.
        """
        if not self.touched:
            return True
        while True:
            answer = self._ask(_HANDS_OFF, view_tone="outside")
            word = answer.strip().lower()
            if word in _FINISH_WORDS:
                self._clear()
                return False
            if word not in ("", "y", "yes"):
                self.console.say(f"{answer.strip()!r} is not an answer here: Enter continues, q finishes.")
                continue
            stopped = self._count_down()
            if stopped == FINISH:
                self._clear()
                return False
            if stopped is None:
                break
            self.console.say(f"{stopped} stopped the countdown: Enter starts it again, q finishes.")
        self._clear()
        self.touched = False
        return True

    def _count_down(self) -> str | None:
        """The countdown before the arm drives by itself: ``None`` once it ran out, else what stopped it.

        Each second is split into steps of :attr:`period_s`, and before every step, and once more after the last,
        the console and the view are read. :data:`FINISH` for a finish from either (a finish or a close from the
        view is kept, as :meth:`_drain` keeps one, so every later prompt finishes too); for anything else typed,
        the words that say what it was.
        """
        steps = max(1, round(1.0 / self.period_s))
        for left in range(COUNTDOWN_S, 0, -1):
            self.console.say(f"Hands off: the arm moves by itself in {left} s (q finishes, Ctrl-C stops)")
            if self.view is not None:
                # Ctrl-C typed into the window reaches no program: the window names the stops that work from it.
                self.view.guide([f"HANDS OFF: THE ARM MOVES BY ITSELF IN {left} S",
                                 "q or ESC here finishes before it moves, Ctrl-C in the console stops"], "outside")
            for _ in range(steps):
                stopped = self._typed_while_counting()
                if stopped is not None:
                    return stopped
                self.sleep(1.0 / steps)
        return self._typed_while_counting()

    def _typed_while_counting(self) -> str | None:
        """What was typed since the last look during a countdown: :data:`FINISH`, what else it was, or ``None``."""
        line = self.console.poll()
        if line is not None:
            word = line.strip()
            if word.lower() in _FINISH_WORDS:
                return FINISH
            return repr(word) if word else "Enter"
        key = self.view.poll_key() if self.view is not None else None
        if key in ("finish", "closed"):
            self._view_finished = True
            return FINISH
        return None if key is None else {"enter": "Enter in the window", "skip": "s in the window"}.get(key, repr(key))

    # --- helpers --------------------------------------------------------------------------------------------

    def _ask(self, question: str, *, view_tone: str = "guide") -> str:
        """``question`` on the console (and the view), then the next line typed; the view's Enter answers ``""``."""
        self._drain()
        if self._view_finished:
            return "q"
        self.console.say(question)
        self.console.bell()
        if self.view is not None:
            self.view.guide([question], view_tone)
        while True:
            line = self.console.poll()
            if line is not None:
                return line
            key = self.view.poll_key() if self.view is not None else None
            if key == "enter":
                return ""
            if key in ("finish", "closed"):
                return "q"
            self.sleep(self.period_s)

    def _key(self) -> str | None:
        """What the person asked for since the last look, from the console first, then the view."""
        if self._view_finished:
            return FINISH
        line = self.console.poll()
        if line is not None:
            word = line.strip().lower()
            if word == "":
                return CAPTURE
            if word in ("s", "skip"):
                return SKIP
            if word in ("q", "quit", "finish", EOF):
                return FINISH
            self.console.say(f"{line!r} is not a key here: {_KEYS}.")
            return None
        key = self.view.poll_key() if self.view is not None else None
        return {"enter": CAPTURE, "skip": SKIP, "finish": FINISH, "closed": FINISH}.get(key or "")

    def _drain(self) -> None:
        """Forget what was typed before a prompt, so an Enter meant for the last one does not answer this one.

        A finish from the view (``q``, or the window closed) is kept: it answers the next prompt with a finish.
        """
        while (line := self.console.poll()) is not None:
            if line == EOF:
                raise HandGuidingRefused("the console's input has ended, so nobody can answer; hand guiding needs a "
                                         "person at a terminal")
        while self.view is not None and (key := self.view.poll_key()) is not None:
            if key in ("finish", "closed"):
                self._view_finished = True

    def _clear(self) -> None:
        """Give the view back to the sweep's own lines once nobody is being asked anything."""
        if self.view is not None:
            self.view.guide([], "guide")

    def _show(self, banner: str, outside: str, sample: FreedriveSample,
              status: Callable[[FreedriveSample], tuple[list[str], float | None]] | None) -> None:
        if self.view is None:
            return
        lines, bar = status(sample) if status is not None else ([], None)
        if outside:
            self.view.guide([f"OUTSIDE - NOT CAPTURED: {outside}", banner, *lines], "outside", bar)
        else:
            self.view.guide([banner, *lines], "guide", bar)
