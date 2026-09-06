"""An OnRobot RG2 or RG6 as a `Gripper`: the width-based Protocol over a Compute Box.

This is the thin half. :mod:`.onrobot_modbus` owns the wire; this owns the
translation between the vendor-neutral `Gripper` Protocol (widths in millimetres,
`activate`, `is_object_detected`) and what an RG actually has. The split matters here
more than it does for Robotiq, because three of the Protocol's assumptions do not
hold:

  `activate()`      is a no-op, and saying so is the point. An RG has no activation
                    stroke and no register that corresponds to one. Robotiq's
                    `activate` sweeps the full finger travel; this returns without
                    commanding anything, so a caller that reads the Protocol as
                    meaning activate moves the gripper is wrong about this vendor.
  `set_width_mm`    drops the `speed` argument, and logs that it dropped it. RG2 and
                    RG6 have no speed register anywhere in the writable map, and
                    OnRobot's own library exposes only a read-only `rg_get_speed`.
                    Accepting a speed silently would be a lie, so it is accepted
                    because the Protocol requires the parameter, ignored, and said
                    out loud once.
  `force`           is newtons natively, unlike Robotiq's opaque 0-255 counts. The
                    register is tenths of a newton, so the conversion is exact rather
                    than a calibration guess.

The width also runs the other way. Here a bigger number is more open, because the
register is an opening in tenths of a millimetre, where Robotiq's counts run 0 for
open and 255 for closed. Two grippers in one package with opposite polarity is a real
hazard, and both files say so.

Honesty bucket 3: no Compute Box exists on this machine. The framing underneath was
measured against a real Modbus TCP server, the one URSim runs, but nothing here has
spoken to an RG.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from ..constants import ONROBOT_GRIPPER_LOG_FILE, create_robot_logger
from .onrobot_modbus import UNIT_QUICK_CHANGER, ModbusError, OnRobotRG, RGStatus

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import GripperConfig

__all__ = ["OnRobotGripper"]

#: What `set_width_mm` uses when the caller passes `force=None`, which the shipped
#: execution policy always does. It is newtons rather than a normalised count, because
#: an RG register is tenths of a newton, so 20 N is a real physical number and a
#: deliberately gentle default rather than a scaled midpoint.
_DEFAULT_FORCE_N = 20.0


class OnRobotGripper:
    """An RG2 or RG6 on a Compute Box, behind the vendor-neutral `Gripper` Protocol.

    The Compute Box has its own address. The Robotiq driver reuses the arm's IP
    because the URCap daemon runs on the controller; an OnRobot box is a separate
    device on the network, so `host` is config of its own and must never default to
    `robot.ur.ip`.
    """

    def __init__(
        self,
        config: "GripperConfig",
        host: str,
        *,
        port: int = 502,
        unit: int = UNIT_QUICK_CHANGER,
        default_force_n: float = _DEFAULT_FORCE_N,
        use_fingertip_offset: bool = False,
        client_factory: "Callable[[], Any] | None" = None,
    ) -> None:
        self.config = config
        self.host = host
        self.port = int(port)
        self.unit = int(unit)
        self.default_force_n = float(default_force_n)
        self.use_fingertip_offset = bool(use_fingertip_offset)
        #: The injection seam, the same shape `robotiq.py` uses. Swapping what sits
        #: behind it is what let that driver change its whole transport without
        #: touching a line above.
        self._client_factory = client_factory or (lambda: OnRobotRG(unit=self.unit))
        self._client: Any | None = None
        self.logger = create_robot_logger("OnRobotGripper", ONROBOT_GRIPPER_LOG_FILE)
        self._warned_about_speed = False

    # --- Protocol: identity ---------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    @property
    def min_width_mm(self) -> float:
        return float(self.config.min_width_mm)

    @property
    def max_width_mm(self) -> float:
        return float(self.config.max_width_mm)

    @property
    def closed_width_mm(self) -> float:
        return float(self.config.closed_width_mm)

    # --- Protocol: lifecycle --------------------------------------------------------------------

    def connect(self) -> None:
        """Open the Modbus connection and refuse a gripper that cannot be commanded.

        The refusal is the whole value of doing this at connect. A tripped safety
        switch leaves the gripper dead until tool power is cycled, and every command
        sent meanwhile is discarded silently. Finding that out at connect costs a
        message; finding it out mid-pick costs a part.

        It does not move. There is no activation stroke on an RG, so unlike a Robotiq
        connect this commands nothing at all.
        """
        client = self._client_factory()
        client.connect(self.host, self.port)
        self._client = client
        reading = client.probe(sweep_units=False)
        if not reading.reachable:
            self._client = None
            client.disconnect()
            raise ModbusError(
                f"connected to {self.host}:{self.port} but unit {self.unit} did not answer: "
                f"{reading.detail}"
            )
        if reading.blocked:
            self._client = None
            client.disconnect()
            raise ModbusError(
                f"the gripper on unit {self.unit} has a safety switch tripped "
                f"(status 0x{int(reading.status):04x}). It stays dead until TOOL POWER is cycled."
            )
        self.logger.info(
            "connected to the Compute Box at %s:%d, unit %d: %.1f mm open, status 0x%04x. "
            "this cannot tell an RG2/RG6 from an RG2-FT; the unit id is chosen by the mounting.",
            self.host, self.port, self.unit, reading.width_mm or 0.0, int(reading.status),
        )

    def disconnect(self) -> None:
        """Close the connection. It releases nothing: a gripper holding a part goes on holding it."""
        if self._client is None:
            return
        try:
            self._client.disconnect()
        finally:
            self._client = None

    def activate(self) -> None:
        """A no-op, and deliberately not an error.

        An RG has no activation and no register that means one. The Protocol has this
        method because a Robotiq needs it, where activation is a full-travel
        calibration sweep. Raising here would make every vendor-neutral caller
        special-case OnRobot, and commanding something would mean writing into
        whatever register 0 happens to be, which is the target force. Doing nothing is
        the honest implementation.
        """
        self.logger.debug("activate(): no-op, an RG has no activation routine")

    # --- Protocol: motion -----------------------------------------------------------------------

    def set_width_mm(
        self, width_mm: float, *, speed: float | None = None, force: float | None = None
    ) -> None:
        """Command an opening in millimetres.

        `speed` is accepted and ignored, which is the least-bad of three options. RG2
        and RG6 have no speed register, since the writable map is force, width and
        control. Raising would break every vendor-neutral caller on a parameter the
        Protocol requires them to be able to pass, and dropping it silently would be a
        lie. So it is dropped and said once per connection rather than per command,
        because a warning on every close is one an operator stops reading.

        `force` is newtons here, natively. The Robotiq equivalent is an opaque 0-255
        count whose physical span differs per model; an RG register is tenths of a
        newton, so this conversion is exact.
        """
        if speed is not None and not self._warned_about_speed:
            self.logger.warning(
                "set_width_mm(speed=%s) ignored: RG2/RG6 have no speed register anywhere in the "
                "writable map, and OnRobot's own library exposes only a read-only rg_get_speed. "
                "Reported once per connection.", speed,
            )
            self._warned_about_speed = True
        target = self._clamp(width_mm)
        client = self._require_connected()
        client.grip(
            width_mm=target,
            force_n=self.default_force_n if force is None else float(force),
            use_fingertip_offset=self.use_fingertip_offset,
        )

    def open(self, *, speed: float | None = None, force: float | None = None) -> None:
        self.set_width_mm(self.max_width_mm, speed=speed, force=force)

    def close(self, *, speed: float | None = None, force: float | None = None) -> None:
        self.set_width_mm(self.min_width_mm, speed=speed, force=force)

    def get_width_mm(self) -> float:
        """The actual opening, without the fingertip offset, which lives in a separate register."""
        return float(self._require_connected().width_mm())

    def is_object_detected(self) -> bool:
        """Whether the jaws stopped on something rather than reaching the commanded width.

        This is real post-grasp evidence, which the jaw-I/O path only has where reed
        switches are wired. The status word carries it directly.
        """
        return bool(self._require_connected().status() & RGStatus.GRIP_DETECTED)

    # --- internals ------------------------------------------------------------------------------

    def _clamp(self, width_mm: float) -> float:
        return max(self.min_width_mm, min(self.max_width_mm, float(width_mm)))

    def _require_connected(self) -> Any:
        if self._client is None:
            raise ModbusError("gripper is not connected; call connect() first")
        return self._client

    def __enter__(self) -> "OnRobotGripper":
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.disconnect()
