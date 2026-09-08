"""GripperController, a thin wrapper around the Robotiq HE and HE-X gripper.

It talks to the gripper over the URCap socket through :mod:`.robotiq_socket`. The
gripper speaks in raw 0-255 position counts; this class converts them to and from
millimetres using the configured opening range, so the rest of the codebase stays in
physical units.

Numerics contract
-----------------
* Public widths are millimetres.
* ``speed`` and ``force`` are normalised into ``[0.0, 1.0]``, where 1.0 is the
  maximum.
* The driver counts, 0 to 255, never leak through the public API.

Robustness
----------
* The transport is deferred-imported inside :func:`_default_driver_factory`, so this
  module imports on a host that has none of it.
* Every operation is logged to the shared robot logfile.
* The ``driver_factory`` constructor argument is a full injection seam: pass any
  callable returning an object with ``connect``, ``activate_if_needed`` or
  ``activate``, ``move``, ``get_current_position`` and ``disconnect``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.config.schema.robot import GripperConfig

from ..constants import GRIPPER_LOG_FILE, create_robot_logger

# Native Robotiq position counts.
_POS_OPEN = 0          # fully open
_POS_CLOSED = 255      # fully closed
#: Physical finger gap at _POS_CLOSED, when a config does not say. The 2F-85 fingers meet, so
#: the closed gap is 0 mm. That is a property of the hardware and not of the `min_width_mm`
#: policy floor in the config, and anchoring the count map on the physical gap rather than on
#: the floor is what keeps a commanded 40 mm landing at 40 mm.
#:
#: It is a fallback now rather than the anchor. `closed_width_mm` is a config key, is
#: documented as the physical closed width, and was read by the grasp verifier while this
#: constant answered the same question for the count map. Two anchors for one fact, agreeing
#: on exactly the gripper the constant was written for. A Robotiq Hand-E with custom
#: fingertips is where they part: setting `closed_width_mm: 8.0` fixed the verifier and left
#: every commanded width 5.6 mm out on a 50 mm tool, because this number went on saying 0.
_WIDTH_CLOSED_MM = 0.0
#: The Robotiq dashboard port on the UR controller.
_DEFAULT_PORT = 63352
#: Speed and force used when a caller passes ``None``, which is the Gripper Protocol
#: default. Every real-hardware close arrives here as ``None``:
#: ``GraspExecutionPolicy.close_speed`` and ``close_force_n`` are ``None`` and are
#: passed explicitly (execution_policy.py:283-287), and nothing in the repo sets them.
#: Without these constants that explicit ``None`` reaches ``_normalise_to_count`` and
#: raises on the first commanded close.
_DEFAULT_SPEED = 1.0
_DEFAULT_FORCE = 0.5


def _default_driver_factory() -> Any:
    """Default factory: the dependency-free client for the URCap socket.

    ``robotiq_gripper`` cannot be installed. ``pip index versions robotiq_gripper``
    answers that no matching distribution was found, and the installed `ur_rtde` ships
    nothing named ``robotiq*``. It is a single example file from SDU's repository that
    an operator drops on the path by hand, so a driver gated behind it would be gated
    behind a package that does not exist.

    :mod:`.robotiq_socket` speaks the port-63352 grammar directly in about a hundred
    lines of `socket`. The same five calls, no third-party import, and no SDK gate for
    this vendor.

    The `driver_factory` seam is what made that a one-function change: swapping what
    sits behind it touches nothing above.
    """
    from .robotiq_socket import RobotiqSocket

    return RobotiqSocket()


class GripperController:
    """High-level Robotiq gripper control.

    Parameters
    ----------
    config : GripperConfig
        Physical opening limits, ``min_width_mm`` to ``max_width_mm``.
    ip : str
        IP address of the UR controller. The gripper is daisy-chained on the robot's
        tool I/O, so it answers on the same IP.
    port : int
        Robotiq dashboard port. Defaults to 63352, the factory default.
    driver_factory : Callable returning a Robotiq-like driver, optional
        Injection seam. Defaults to :func:`_default_driver_factory`, which builds a
        :class:`.robotiq_socket.RobotiqSocket`.
    """

    def __init__(
        self,
        config: GripperConfig,
        ip: str,
        port: int = _DEFAULT_PORT,
        driver_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config
        self.ip = ip
        self.port = port
        self._driver_factory = driver_factory or _default_driver_factory
        self.logger = create_robot_logger("GripperController", GRIPPER_LOG_FILE)

        self._driver: Any | None = None
        self._activated: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._driver is not None

    @property
    def min_width_mm(self) -> float:
        """Smallest commandable jaw opening, from config. A policy floor, not the mechanism."""
        return float(self.config.min_width_mm)

    @property
    def closed_width_mm(self) -> float:
        """What :meth:`get_width_mm` reads with the jaws shut on nothing, 0 mm on a 2F-85.

        This is distinct from :attr:`min_width_mm`, which is a policy floor.
        Verification asks for this one, because whether the jaws collapsed on nothing
        is a question about the mechanism.
        """
        return float(self.config.closed_width_mm)

    @property
    def max_width_mm(self) -> float:
        """Largest commandable jaw opening, from config."""
        return float(self.config.max_width_mm)

    def connect(self) -> None:
        """Open the dashboard connection and activate. Idempotent.

        This moves the gripper. Robotiq activation is a calibration routine in which
        the fingers travel their full range so the controller can find its own limits,
        which is exactly why it happens here rather than lazily. On a real cell the
        first commanded width is the grasp: the arm has already descended and the
        fingers are around the object, and a gripper that activated then would choose
        that moment to sweep its full travel. Doing it at connect puts the sweep in
        whatever pose the cell connects in, and the bring-up order in ``real_cell``,
        which connects before any motion, makes that the arm's starting pose.

        ``set_width_mm`` still carries the lazy fallback, with a warning, as a net for
        a gripper constructed outside that order. It is not the expected route.
        """
        if self.is_connected:
            self.logger.debug("connect() called but already connected; ignored.")
            return
        self.logger.info("Connecting to gripper at %s:%d ...", self.ip, self.port)
        driver = self._driver_factory()
        driver.connect(self.ip, self.port)
        self._driver = driver
        # Fail closed: a connected but unactivated gripper accepts commands and does
        # not grip, which reads as a bad grasp rather than an unconfigured tool. Roll
        # the connection back instead.
        try:
            self.activate()
        except BaseException as exc:  # noqa: BLE001 (the rollback below is the point)
            self.logger.error("gripper activation failed; rolling back the connection: %s", exc)
            self.disconnect()
            raise

    def disconnect(self) -> None:
        """Close the dashboard connection. Always safe to call."""
        if self._driver is None:
            return
        try:
            self._driver.disconnect()
        except (RuntimeError, OSError) as exc:
            self.logger.warning("Gripper disconnect raised %s; ignoring.", exc)
        finally:
            self._driver = None
            self._activated = False

    def activate(self) -> None:
        """Activate the gripper if it is not already calibrated."""
        drv = self._require_connected()
        # A Robotiq driver exposes either the idempotent ``activate_if_needed`` or
        # only ``activate``, so both are supported.
        fn = (
            getattr(drv, "activate_if_needed", None)
            or getattr(drv, "activate", None)
        )
        if fn is None:
            raise RuntimeError("Driver exposes neither activate() nor activate_if_needed().")
        self.logger.info("Activating gripper ...")
        fn()
        self._activated = True

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def open(self, *, speed: float | None = None, force: float | None = None) -> None:
        """Fully open the gripper. Query :meth:`get_width_mm` for the result."""
        self.set_width_mm(self.config.max_width_mm, speed=speed, force=force)

    def close(self, *, speed: float | None = None, force: float | None = None) -> None:
        """Fully close the gripper. Query :meth:`get_width_mm` for the result."""
        self.set_width_mm(self.config.min_width_mm, speed=speed, force=force)

    def set_width_mm(
        self, width_mm: float, *, speed: float | None = None, force: float | None = None,
    ) -> None:
        """Command an opening of ``width_mm``, clamped to the configured range.

        Parameters
        ----------
        width_mm : float
            Target opening in millimetres.
        speed, force : float, optional
            Normalised into ``[0, 1]`` and mapped to the driver's 0-255 scale
            internally. ``None`` is the :class:`Gripper` Protocol default and what
            :class:`GraspExecutionPolicy` sends unless a caller sets ``close_speed``
            or ``close_force_n``, which nothing in the repo does. It means the
            driver's default, :data:`_DEFAULT_SPEED` and :data:`_DEFAULT_FORCE`.

        Notes
        -----
        This returns ``None`` per the :class:`Gripper` Protocol. Query
        :meth:`get_width_mm` for the achieved opening.
        """
        drv = self._require_connected()
        if not self._activated:
            self.logger.warning("set_width_mm called before activate(); activating now.")
            self.activate()

        clamped_mm = self._clamp_width_mm(width_mm)
        target_count = self._mm_to_count(clamped_mm)
        # Resolve the Protocol convention that None means the driver default here, at
        # the boundary. The vendor arithmetic below takes a real number and nothing
        # else.
        speed = _DEFAULT_SPEED if speed is None else speed
        force = _DEFAULT_FORCE if force is None else force
        speed_count = self._normalise_to_count(speed)
        force_count = self._normalise_to_count(force)

        self.logger.debug(
            "Gripper move: width=%.2f mm (cnt=%d), speed=%.2f, force=%.2f",
            clamped_mm, target_count, speed, force,
        )
        drv.move(target_count, speed_count, force_count)

    def get_width_mm(self) -> float:
        """Read the current opening width in millimetres."""
        drv = self._require_connected()
        count = int(drv.get_current_position())
        return float(self._count_to_mm(count))

    # ------------------------------------------------------------------
    # Context manager sugar
    # ------------------------------------------------------------------

    def __enter__(self) -> GripperController:
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_connected(self) -> Any:
        if self._driver is None:
            raise RuntimeError("Gripper is not connected. Call connect() first.")
        return self._driver

    def _clamp_width_mm(self, width_mm: float) -> float:
        lo = self.config.min_width_mm
        hi = self.config.max_width_mm
        if width_mm < lo:
            self.logger.debug("Clamping width %.2f -> %.2f (min).", width_mm, lo)
            return lo
        if width_mm > hi:
            self.logger.debug("Clamping width %.2f -> %.2f (max).", width_mm, hi)
            return hi
        return float(width_mm)

    @property
    def _closed_gap_mm(self) -> float:
        """The finger gap at ``_POS_CLOSED``, from the config that declares it.

        One anchor. This used to be a module constant while ``closed_width_mm`` sat in the
        config being read by the grasp verifier alone, so the two answered the same question
        and agreed only about the 2F-85. The constant remains as the schema default, which is
        one statement rather than two.
        """
        return float(getattr(self.config, "closed_width_mm", _WIDTH_CLOSED_MM))

    def _mm_to_count(self, width_mm: float) -> int:
        """Map a finger gap in mm to Robotiq driver counts, 255 closed to 0 open.

        The map is anchored on the physical travel: :attr:`_closed_gap_mm` maps to 255 and
        ``max_width_mm`` maps to 0. It deliberately does not use ``min_width_mm``, which is a
        smallest-meaningful-grip policy floor; tying the hardware mapping to it made a
        commanded 40 mm land at 37.3 mm on a 2F-85. The floor is applied separately in
        :meth:`_clamp_width_mm`.
        """
        lo, hi = self._closed_gap_mm, self.config.max_width_mm
        span = hi - lo
        frac = (width_mm - lo) / span if span > 0 else 0.0  # 0..1
        frac = min(1.0, max(0.0, frac))
        count = _POS_CLOSED + (_POS_OPEN - _POS_CLOSED) * frac
        return int(round(count))

    def _count_to_mm(self, count: int) -> float:
        """Inverse of :meth:`_mm_to_count`: driver counts to a physical finger gap in mm."""
        lo, hi = self._closed_gap_mm, self.config.max_width_mm
        count = min(_POS_CLOSED, max(_POS_OPEN, count))
        frac = (count - _POS_CLOSED) / (_POS_OPEN - _POS_CLOSED)
        return lo + (hi - lo) * frac

    @staticmethod
    def _normalise_to_count(value: float) -> int:
        v = min(1.0, max(0.0, float(value)))
        return int(round(v * 255))
