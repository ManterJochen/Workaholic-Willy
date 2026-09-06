"""Vacuum gripper driven through the robot controller's digital I/O.

This is what lets a real cell be told it has a suction gripper. Without it the
suction stack is reachable in sim and unreachable on hardware, because
:class:`GripperVendor` would list jaw vendors only.

The driver is vendor-neutral by design rather than by omission. A suction
end-effector on a UR is an ejector or a pump wired to a controller output: assert the
pin and vacuum builds, drop it, usually with a short blow-off pulse, and the part
releases. The part-present signal comes back on a digital input from a vacuum switch.
There is no SDK to import and nothing manufacturer-specific in that loop, so a cell
can be configured for suction before anyone knows which cup will be bought.

What it needs is an object that can drive controller I/O, which is the
:class:`SupportsDigitalIO` capability the UR driver already advertises. Anything
satisfying that Protocol works.

Width semantics. The :class:`Gripper` Protocol is width-based because jaws are. A cup
has no opening, so width is reinterpreted exactly as the simulated suction gripper
reinterprets it, which keeps the two honest about each other::

    set_width_mm(w)   w <= vacuum_on_below_mm  ->  vacuum ON  (engage)
                      w >  vacuum_on_below_mm  ->  vacuum off (release)

The payoff is feedback. With a vacuum switch wired to an input this driver implements
:class:`ObjectDetectingGripper`, which opts the cell into post-close verification in
``GraspExecutionPolicy``. A jaw gripper mostly has to be trusted; a suction cup can be
asked whether it is holding something, which is a better verification signal than
anything on the jaw path.

Status is bucket 3: this has never touched a real vacuum generator. The I/O calls it
makes are the ones ``ur_rtde`` exposes and the UR I/O integration exercised. What is
unverified is the wiring, meaning the pin numbers, which port block, whether the
vacuum switch is active-high, and how long the ejector takes to build vacuum on real
hardware. Every one of those is config and is stated in :class:`VacuumGripperConfig`,
so bring-up is a matter of measuring numbers rather than writing code.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from src.robot.core import RobotConnectionError
from src.robot.core.arm_capabilities import DigitalIOPort, SupportsDigitalIO

from ..constants import VACUUM_GRIPPER_LOG_FILE, create_robot_logger

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot.robot_schema import GripperConfig

__all__ = ["VacuumGripper"]


class VacuumGripper:
    """A vacuum end-effector actuated over the controller's digital I/O.

    ``io`` is any object satisfying :class:`SupportsDigitalIO`. In a real cell that is
    the arm driver itself, because the ejector is wired to the controller. The gripper
    never opens a connection of its own: it borrows the arm's, so ``connect()`` and
    ``disconnect()`` reflect whether the I/O source is usable rather than owning a
    socket.
    """

    def __init__(
        self,
        io: SupportsDigitalIO,
        *,
        config: "GripperConfig | None" = None,
        vacuum_output_pin: int = 0,
        blow_off_output_pin: int | None = None,
        vacuum_ok_input_pin: int | None = None,
        io_port: DigitalIOPort | str = DigitalIOPort.TOOL,
        engage_timeout_s: float = 1.0,
        blow_off_s: float = 0.15,
        min_width_mm: float = 0.0,
        max_width_mm: float = 30.0,
        vacuum_on_below_mm: float = 5.0,
        sleep: Any = time.sleep,
    ) -> None:
        if not isinstance(io, SupportsDigitalIO):
            raise TypeError(
                "VacuumGripper needs a digital-I/O source (the arm driver on a real cell): "
                f"{type(io).__name__} does not satisfy SupportsDigitalIO."
            )
        self._io = io
        self._pin = int(vacuum_output_pin)
        self._blow_off_pin = None if blow_off_output_pin is None else int(blow_off_output_pin)
        self._ok_pin = None if vacuum_ok_input_pin is None else int(vacuum_ok_input_pin)
        self._port = DigitalIOPort(io_port) if not isinstance(io_port, DigitalIOPort) else io_port
        self._engage_timeout_s = float(engage_timeout_s)
        self._blow_off_s = float(blow_off_s)
        if config is not None:
            min_width_mm, max_width_mm = float(config.min_width_mm), float(config.max_width_mm)
        self._min_width_mm = float(min_width_mm)
        self._max_width_mm = float(max_width_mm)
        self._vacuum_on_below_mm = float(vacuum_on_below_mm)
        self._sleep = sleep
        self._connected = False
        self._vacuum_on = False
        self.logger = create_robot_logger("VacuumGripper", VACUUM_GRIPPER_LOG_FILE)

    # --- state ------------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def min_width_mm(self) -> float:
        return self._min_width_mm

    @property
    def max_width_mm(self) -> float:
        return self._max_width_mm

    @property
    def vacuum_on(self) -> bool:
        """Whether vacuum is currently commanded, the driver's own view and not the switch."""
        return self._vacuum_on

    # --- lifecycle --------------------------------------------------------
    def connect(self) -> None:
        """Adopt the arm's I/O and command a known state, vacuum off.

        A cell that starts with the ejector latched on from a previous run would hold
        a part it does not know about, so the safe state is asserted rather than
        assumed.
        """
        self._connected = True
        self._set_vacuum(False)
        # The pin map is the half of this driver that cannot be inferred from a later
        # failure. A report that the cup never sealed reads differently once the log
        # says which pins were being watched.
        self.logger.info(
            "connected on %s: vacuum pin=%d, ok input=%s, blow-off pin=%s; commanded OFF",
            self._port.value, self._pin,
            self._ok_pin if self._ok_pin is not None else "none",
            self._blow_off_pin if self._blow_off_pin is not None else "none",
        )

    def disconnect(self) -> None:
        """Release vacuum and detach. Never raises, so a teardown cannot strand a held part.

        This is best-effort by design. If the I/O has already gone away there is
        nothing useful to do, and raising here would mask whatever tore the cell down.
        """
        try:
            if self._connected:
                self._set_vacuum(False)
        except Exception:  # noqa: BLE001 (teardown must not raise)
            self.logger.warning("VacuumGripper.disconnect: releasing vacuum failed", exc_info=True)
        finally:
            self._connected = False
            self.logger.info("disconnected (vacuum released, I/O handed back to the arm)")

    def activate(self) -> None:
        """No calibration to run, because a cup has no jaws to home. Kept for Protocol parity."""
        self._require_connected("activate")

    # --- commands ---------------------------------------------------------
    def set_width_mm(
        self, width_mm: float, *, speed: float | None = None, force: float | None = None,
    ) -> None:
        """Reinterpret width as vacuum on or off, matching the sim cup.

        A width at or below ``vacuum_on_below_mm`` engages. ``speed`` and ``force``
        are accepted for Protocol parity and ignored, because an ejector has one
        setting. Engaging waits for the vacuum switch where one is configured:
        commanding the pin and moving away immediately is how a cell drops parts,
        since the ejector needs time to build.
        """
        self._require_connected("set_width_mm")
        engage = float(width_mm) <= self._vacuum_on_below_mm
        self._set_vacuum(engage)
        if engage and self._ok_pin is not None:
            self._await_vacuum()

    def get_width_mm(self) -> float:
        """Reported opening: the closed band while engaged, the open band while released."""
        return self._min_width_mm if self._vacuum_on else self._max_width_mm

    def is_object_detected(self) -> bool:
        """``True`` when the vacuum switch says a part is held, per :class:`ObjectDetectingGripper`.

        Without a switch wired there is nothing to read, so this reports the commanded
        state. That is the honest answer, as far as this driver knows, and it matches
        how a jaw gripper without feedback behaves. Configuring
        ``vacuum_ok_input_pin`` turns it into a real measurement.
        """
        self._require_connected("is_object_detected")
        if self._ok_pin is None:
            return self._vacuum_on
        return bool(self._io.get_digital_input(self._ok_pin, port=self._port))

    # --- internals --------------------------------------------------------
    def _require_connected(self, what: str) -> None:
        if not self._connected:
            raise RobotConnectionError(f"VacuumGripper.{what} requires connect() first.")

    def _set_vacuum(self, on: bool) -> None:
        self.logger.debug("vacuum %s (pin=%d on %s)", "ON" if on else "OFF", self._pin, self._port.value)
        self._io.set_digital_output(self._pin, bool(on), port=self._port)
        # Releasing a suction grasp is more than stopping the pull: residual vacuum
        # keeps a light part stuck to the cup and it lets go somewhere unintended. A
        # blow-off pulse pushes it off deliberately.
        if not on and self._blow_off_pin is not None:
            self._io.set_digital_output(self._blow_off_pin, True, port=self._port)
            self._sleep(self._blow_off_s)
            self._io.set_digital_output(self._blow_off_pin, False, port=self._port)
        self._vacuum_on = bool(on)

    def _await_vacuum(self) -> bool:
        """Poll the vacuum switch until it reports a seal or the timeout expires, and return the verdict.

        A timeout is not an error here. A missed seal is a normal grasp outcome and
        the verification stage decides, so raising would turn a cup that did not catch
        this one into a crash.
        """
        started = time.monotonic()
        deadline = started + self._engage_timeout_s
        while True:
            if bool(self._io.get_digital_input(self._ok_pin, port=self._port)):  # type: ignore[arg-type]
                # How long the ejector took to build is the number engage_timeout_s is
                # budgeted from, and on a real cell it drifts as the cup wears.
                self.logger.info(
                    "vacuum switch confirmed a seal after %.3f s", time.monotonic() - started,
                )
                return True
            if time.monotonic() >= deadline:
                self.logger.info(
                    "vacuum did not reach the switch threshold within %.2f s; treating as no seal "
                    "(a missed grasp, not a fault)", self._engage_timeout_s,
                )
                return False
            self._sleep(0.02)
