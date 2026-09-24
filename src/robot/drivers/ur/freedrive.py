"""The UR hand-guiding session: teach mode behind a watchdog, and the arm held again on every way out.

:class:`URFreedriveSession` is :class:`~src.robot.core.FreedriveSession` on a UR controller, and
``URRobotArm.freedrive()`` hands one out. The order of its calls is what makes it safe, so it is written
down here once:

``free()``
    The controller must be operational and the arm standing still. Then the RTDE watchdog is armed, and
    only after the controller took it is teach mode asked for. A watchdog the controller did not take
    frees nothing.

``sample()``
    Kicks the watchdog while the arm is free, then reads the joints, their speeds and the TCP. A stop or
    a program that ended is found here and raised as a clear error once the session has marked itself
    held.

``hold()``
    ``endTeachMode``, then at once a fresh control program on a new control interface, then a wait until
    the joints stand still. The fresh program clears the watchdog, which otherwise lives as long as the
    program and would trip during the wait and during the next blocking moveJ, and stopping the old one
    ends teach mode where ``endTeachMode`` failed, because a teach mode does not outlive its program.

Measured on the CB3 URSim (PolyScope 3.15.8, ur_rtde 1.6.5, 2026-09-24): a client killed in teach mode
with no watchdog leaves the program running and the arm free (still so 3 s later). With the watchdog at
2 Hz a session of this driver whose process died was protective-stopped between 0.6 and 1.0 s later (a
bare ur_rtde client: 0.3 to 0.7 s), and a process that lives on and stops calling ``sample()`` 1.0 s after
its last call (bare: 0.5 s). The watchdog's stop action is a PROTECTIVE STOP on this controller, not a
quiet end of the program: every trip, a slow loop included, costs an unlock at the pendant and a
reconnect. ``teachMode``/``endTeachMode`` answer ``True`` in about 20 ms; a moveJ sent in teach mode past
the arm's refusal, and a stop on the pendant, each end the control program (the moveJ answers ``False``
after 20 ms and the arm does not move), which ``sample()`` reports and after which the fresh program runs
and the next move works. The fresh program runs about 2 s after the stop, after which one second with no
command sent trips nothing and a blocking moveJ runs. ``reuploadScript`` on the old interface did not do:
after a few seconds its upload never ran (``URConnection.restart_control_script`` has the measurement).

Leaving the with-block holds, on an exception and on Ctrl-C too, and every motion verb of the arm
refuses from the with-block's start to its end.

What this does not do, on purpose: an arm leaving the cable window or the workspace box while a person
moves it is not stopped or locked here. An arm that locks in a person's hands is the injury this avoids;
whoever shows the preview says so and refuses to capture there.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, NoReturn

from src.geometry.quaternion import to_axis_angle
from src.robot.constants import UR_ARM_LOG_FILE, create_robot_logger
from src.robot.core import (
    FreedriveSample,
    RobotConnectionError,
    RobotEmergencyStop,
    RobotError,
    RobotMotionRejected,
    RobotStatus,
)

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from collections.abc import Callable, Sequence

    from src.geometry import Pose

    from .connection import URConnection

__all__ = [
    "HAND_GUIDED_MESSAGE",
    "STILL_RAD_S",
    "WATCHDOG_MIN_HZ",
    "URFreedriveSession",
    "raise_unless_operational",
]

#: What every motion verb of the arm answers while a session is open.
HAND_GUIDED_MESSAGE = "the arm is hand-guided: a freedrive session is open"

#: The watchdog's minimum rate, in Hz: while the arm is free, a gap of more than 0.5 s between two
#: ``sample()`` calls may protective-stop the arm (it did after 1.0 s, see above). The contract asks
#: for 30 Hz or more, so a slow camera frame or a garbage-collector pause fits fifteen times over, which
#: matters because a trip is a protective stop, and an arm whose program died is free for under 1 s more.
WATCHDOG_MIN_HZ = 2.0

#: A joint slower than this, in rad/s, stands still: 0.3 deg/s, under 7 mm/s at the tool of a UR10.
STILL_RAD_S = 0.005

#: How long ``hold()`` waits for the joints to stand still after teach mode ended, in seconds.
SETTLE_TIMEOUT_S = 2.0

_POLL_S = 0.01


def _peak(speeds: Sequence[float]) -> float:
    """The fastest joint of ``speeds``, in rad/s."""
    return max((abs(float(v)) for v in speeds), default=0.0)


def raise_unless_operational(status: RobotStatus, what: str) -> None:
    """Raise why ``what`` is refused on a controller that is not operational; return where it is.

    Hand guiding needs the arm powered, its brakes released, no stop and the safety mode normal: a
    reduced or recovery mode is a state a person should not be handed the arm in. A stop raises
    :class:`RobotEmergencyStop`, any other state :class:`RobotMotionRejected`.
    """
    if status.is_operational:
        return
    said = f" ({status.message})" if status.message else ""
    sentence = (f"{what} is refused: the controller reports robot mode {status.robot_mode.value} and safety mode "
                f"{status.safety_mode.value}{said}, and hand guiding needs the arm running with its brakes released "
                f"and the safety mode normal")
    if status.is_stopped:
        raise RobotEmergencyStop(f"{sentence}. Clear the stop at the pendant, where the arm is visible.")
    raise RobotMotionRejected(sentence)


class URFreedriveSession:
    """A UR hand-guiding session (:class:`~src.robot.core.FreedriveSession`), built by ``URRobotArm.freedrive()``.

    ``tcp_pose`` reads the TCP the way the arm's ``get_tcp_pose`` does, the tool frame composed, so a
    sample is the pose every other caller of the arm reads. ``robot_status`` is the arm's typed controller
    state. ``on_open`` and ``on_close`` tell the arm the with-block began and ended, which is what its
    motion verbs read; ``on_open`` raises where another session is open. A session is used once: its
    with-block, then a new one.
    """

    def __init__(
        self,
        conn: URConnection,
        *,
        tcp_pose: Callable[[], Pose],
        robot_status: Callable[[], RobotStatus],
        on_open: Callable[[URFreedriveSession], None],
        on_close: Callable[[URFreedriveSession], None],
        watchdog_hz: float = WATCHDOG_MIN_HZ,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._conn = conn
        self._tcp_pose = tcp_pose
        self._robot_status = robot_status
        self._on_open = on_open
        self._on_close = on_close
        self._watchdog_hz = float(watchdog_hz)
        self._clock = clock
        self._sleep = sleep
        self._open = False
        self._left = False
        self._free = False
        self.logger = create_robot_logger("URFreedrive", UR_ARM_LOG_FILE)

    @property
    def is_free(self) -> bool:
        """Whether a person may move the arm now, as far as this session knows."""
        return self._free

    # ------------------------------------------------------------------
    # The with-block
    # ------------------------------------------------------------------

    def __enter__(self) -> URFreedriveSession:
        if self._left:
            raise RobotMotionRejected("this freedrive session was left already; open a new one with arm.freedrive()")
        if not self._open:
            self._on_open(self)
            self._open = True
        return self

    def __exit__(self, *exc: object) -> None:
        """Hold the arm and give motion back, whatever ended the block.

        A failure to hold is raised where the block ended normally. Where it ended on an exception, that
        exception is the one the caller sees and the failure is logged, because the watchdog holds an arm
        this could not reach: the program stops at the first gap in its kicks.
        """
        in_flight = bool(exc) and exc[0] is not None
        try:
            self.hold()
        except Exception as failed:
            if not in_flight:
                raise
            self.logger.error("Leaving the freedrive session could not hold the arm cleanly (%s); the error "
                              "that ended the session propagates.", failed)
        finally:
            self._open = False
            self._left = True
            self._on_close(self)

    # ------------------------------------------------------------------
    # free, hold, sample
    # ------------------------------------------------------------------

    def free(self) -> None:
        """Let a person move the arm: the watchdog armed first, then teach mode. Idempotent.

        Refused outside the with-block, on a controller that is not operational, and on an arm that still
        moves by itself. Where arming or teach mode fails, a fresh control program is started before the
        error is raised, so no watchdog and no teach mode are left behind.
        """
        if self._free:
            return
        self._require_open("free")
        if not self._conn.is_connected:
            raise RobotConnectionError("the arm is not freed: it is not connected")
        raise_unless_operational(self._robot_status(), "freeing the arm")
        moving = _peak(self._conn.get_joint_speeds())
        if moving > STILL_RAD_S:
            raise RobotMotionRejected(f"the arm is not freed: it still moves by itself ({moving:.3f} rad/s at its "
                                      f"fastest joint); let it stop first")
        try:
            if not self._conn.set_watchdog(self._watchdog_hz):
                raise RobotConnectionError("the arm is not freed: the controller did not arm the RTDE watchdog, and "
                                           "without it an arm stays free when this program dies")
            if not self._conn.teach_mode():
                raise RobotConnectionError(f"the arm is not freed: the controller refused teach mode"
                                           f"{self._state_text()}")
        except BaseException as exc:
            self._give_control_back_logged()
            if isinstance(exc, (RuntimeError, OSError)):  # ur_rtde's own faults, named as the stack names them
                raise RobotConnectionError(f"the arm is not freed: {type(exc).__name__}: {exc}") from exc
            raise
        self._free = True
        self.logger.info("The arm is free: a person may move it (UR teach mode behind a %.1f Hz watchdog).",
                         self._watchdog_hz)

    def hold(self) -> None:
        """Make the arm hold where it stands, and give motion back. Idempotent.

        ``endTeachMode``, then at once a fresh control program (about 2 s), whose first step stops the old
        one and with it the watchdog, before anything could leave a gap in its kicks, and which runs even
        where ``endTeachMode`` failed or a Ctrl-C arrived during it, then a wait until the joints stand
        still (at most ``SETTLE_TIMEOUT_S``). The session counts as held from then on: a program that
        could not be replaced is either stopped, and a stopped program holds, or still has the watchdog,
        which stops the arm within about a second now that nothing kicks it.
        """
        if not self._free:
            return
        if not self._conn.is_connected:
            # The connection was torn down: its stopScript ended the program, and teach mode with it.
            self._free = False
            self.logger.warning("The connection closed while the arm was free; its program ended, which held it.")
            return
        try:
            if not self._conn.end_teach_mode():
                self.logger.warning("endTeachMode was not confirmed; stopping the program ends teach mode instead.")
        except (RuntimeError, OSError) as exc:
            self.logger.warning("endTeachMode raised (%s); stopping the program ends teach mode instead.", exc)
        finally:
            self._free = False
            self._give_control_back()
        self._wait_still()
        self.logger.info("The arm holds where it stands, and motion is given back to the program.")

    def sample(self) -> FreedriveSample:
        """One reading of the arm, and while it is free the watchdog's kick. Raises where it stopped being guided.

        A read that fails, a kick the control program does not answer, a program that no longer runs and an
        active protective or emergency stop each end the hand guiding: the session marks itself held, gives
        control back where no stop forbids it, and raises :class:`RobotEmergencyStop` for a stop and
        :class:`RobotConnectionError` for the rest.
        """
        self._require_open("sample")
        try:
            answered = self._conn.kick_watchdog() if self._free else True
            running = self._conn.is_program_running()
            stopped = self._conn.is_protective_stopped() or self._conn.is_emergency_stopped()
            joints = self._conn.get_joint_positions()
            speeds = self._conn.get_joint_speeds()
            pose = self._tcp_pose()
        except (RuntimeError, OSError, RobotError) as exc:
            self._lost(f"the arm could not be read ({type(exc).__name__}: {exc})", exc)
        if stopped:
            self._lost(f"the controller reports a protective or emergency stop; the watchdog raises a protective "
                       f"stop when sample() is not called for {1.0 / self._watchdog_hz:.2f} s or more while the arm "
                       f"is free", stopped=True)
        if not running:
            self._lost("the control program ended, as a stop on the pendant or a motion sent in teach mode ends it")
        if not answered:
            self._lost("the control program did not answer the watchdog's kick")
        rotvec = to_axis_angle(pose.quaternion_xyzw)
        x, y, z = (float(v) for v in pose.position_mm)
        return FreedriveSample(
            joints_rad=tuple(float(v) for v in joints),
            joint_speeds_rad_s=tuple(float(v) for v in speeds),
            tcp_xyz_mm=(x, y, z),
            tcp_rotvec_rad=(float(rotvec[0]), float(rotvec[1]), float(rotvec[2])),
            t_s=self._clock(),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_open(self, verb: str) -> None:
        if not self._open:
            raise RobotMotionRejected(f"{verb}() needs the session open: use it inside "
                                      f"`with arm.freedrive() as session:`, whose end holds the arm")

    def _wait_still(self) -> None:
        """Wait until every joint stands still, at most ``SETTLE_TIMEOUT_S``; a timeout or a failed read is logged."""
        deadline = self._clock() + SETTLE_TIMEOUT_S
        while True:
            try:
                moving = _peak(self._conn.get_joint_speeds())
            except (RuntimeError, OSError) as exc:
                self.logger.warning("The joint speeds could not be read after teach mode ended (%s).", exc)
                return
            if moving <= STILL_RAD_S:
                return
            if self._clock() >= deadline:
                self.logger.warning("A joint still moves at %.3f rad/s %.1f s after teach mode ended.", moving,
                                    SETTLE_TIMEOUT_S)
                return
            self._sleep(_POLL_S)

    def _give_control_back(self) -> None:
        """A fresh control program: no watchdog, no teach mode, and moves work again. Raises where none runs.

        Stopping the old program is what ends a teach mode ``endTeachMode`` did not; the new one runs on a
        new control interface because ``reuploadScript`` on the old one uploaded programs that never ran
        (measured, see the module).
        """
        if not self._conn.is_connected:
            return
        try:
            if self._conn.is_protective_stopped() or self._conn.is_emergency_stopped():
                # A stopped controller starts no program until the stop is cleared at the pendant.
                raise RobotEmergencyStop(f"{self._no_program('the controller is stopped')}. Clear the stop at the "
                                         f"pendant, where the arm is visible, first")
            self._conn.restart_control_script()
            running = self._conn.is_program_running()
        except (RuntimeError, OSError) as exc:
            raise RobotConnectionError(self._no_program(f"{type(exc).__name__}: {exc}")) from exc
        if not running:
            raise RobotConnectionError(self._no_program("the new control interface came up with no program running"))

    def _no_program(self, why: str) -> str:
        """Why no control program runs after hand guiding, and what that leaves the arm doing."""
        return (f"the control program could not be restarted after hand guiding ({why}){self._state_text()}, so "
                f"no program runs for this arm: it holds, and it moves again only after the arm reconnects")

    def _give_control_back_logged(self) -> None:
        """:meth:`_give_control_back` where an error is already on its way, which a second one must not replace."""
        try:
            self._give_control_back()
        except Exception as exc:  # noqa: BLE001 (the first error is the one the caller needs)
            self.logger.error("Giving control back failed as well: %s", exc)

    def _lost(self, why: str, cause: BaseException | None = None, *, stopped: bool = False) -> NoReturn:
        """Mark the session held, give control back where no stop forbids it, and raise why.

        The arm holds already where the program ended or a stop is active; where only a read or a kick
        failed and the program still runs in teach mode, the restart stops it, and where even that
        cannot reach the controller, the watchdog stops it at the first gap in its kicks.
        """
        self._free = False
        status: RobotStatus | None
        try:
            status = self._robot_status()
        except Exception:  # noqa: BLE001 (a state that cannot be read adds nothing to the error)
            status = None
        stopped = stopped or (status is not None and status.is_stopped)
        if not stopped:
            # A stopped controller runs no program until the stop is cleared, so there is none to give back.
            self._give_control_back_logged()
        message = (f"the arm is no longer hand-guided: {why}{self._state_text(status)}. The controller holds it "
                   f"where it stands")
        if stopped:
            raise RobotEmergencyStop(f"{message}. Clear the stop at the pendant, where the arm is visible, and "
                                     f"reconnect before it moves again.") from cause
        raise RobotConnectionError(f"{message}.") from cause

    def _state_text(self, status: RobotStatus | None = None) -> str:
        """The controller's state as a clause for an error, or ``""`` where it cannot be read."""
        if status is None:
            try:
                status = self._robot_status()
            except Exception:  # noqa: BLE001 (best effort inside an error: no state is not a new fault)
                return ""
        said = f", {status.message!r}" if status.message else ""
        return f" (robot mode {status.robot_mode.value}, safety mode {status.safety_mode.value}{said})"
