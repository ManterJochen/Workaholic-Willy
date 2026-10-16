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
from src.robot.core.gripper import HoldEvidence

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
        #: Warn once per controller about a seam that cannot confirm motion. A per-command warning
        #: is noise, and no warning at all hides that no wait happens.
        self._warned_no_wait: bool = False
        #: Whether the last command gripped from inside a bore (:meth:`grip_from_inside`); every other command clears it.
        self._from_inside: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._driver is not None

    @property
    def min_width_mm(self) -> float:
        """The thinnest part the cell plans a grasp for, from config: a policy floor about the part, not the mechanism.
        The jaws close past it to :attr:`closed_width_mm`."""
        return float(self.config.min_width_mm)

    @property
    def closed_width_mm(self) -> float:
        """What :meth:`get_width_mm` reads with the jaws shut on nothing, 0 mm on a 2F-85.

        This is distinct from :attr:`min_width_mm`, which is a policy floor. The count
        map is anchored on this one, because where the jaws stand when shut is a
        question about the mechanism.
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
        """Fully close the gripper, to its :attr:`closed_width_mm`. Query :meth:`get_width_mm` for the result."""
        self.set_width_mm(self.closed_width_mm, speed=speed, force=force)

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
        self._from_inside = False
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
        # The same line the toggle hand writes ("actuating jaws CLOSED via ..."), so a cell's log times a URCap hand's
        # close and release as it times a toggle's (scripts/cell/pick_timeline.py).
        self.logger.info("actuating jaws %s via robotiq: %.1f mm",
                         "OPEN" if clamped_mm >= self.max_width_mm - 0.5 else "CLOSED", clamped_mm)
        drv.move(target_count, speed_count, force_count)
        self._wait_for_the_fingers(drv)
        # Where the fingers stopped: on a part, its width as the hand measures it, beside the width the grasp planned.
        stopped = getattr(drv, "get_current_position", None)
        if callable(stopped):
            try:
                self.logger.info("the jaws stand at %.1f mm", self._count_to_mm(int(stopped())))
            except Exception as exc:  # noqa: BLE001 (a read for the log is no fault of the motion)
                self.logger.debug("the jaws' position could not be read: %s", exc)

    def _wait_for_the_fingers(self, drv: Any) -> None:
        """Block until the gripper says the motion is over, because ``ack`` is not motion.

        The URCap daemon answers ``ack`` once it accepts the line, and that says nothing more.
        Returning on it would let the next statement in the pick path, the retreat lift, move the
        arm while the fingers still travel.

        A driver seam without ``wait_for_motion`` stays fire-and-forget and says so once per
        connection, because a silent absence here is indistinguishable from a wait that happened.
        """
        wait = getattr(drv, "wait_for_motion", None)
        if wait is not None:
            try:
                wait()
            except Exception:
                # A wait that failed leaves the fingers travelling: `GTO` is still set, so the
                # jaws go on closing while the caller unwinds. A fault and a deadline reached while
                # a protective stop holds the cell both end here, the one place that knows the
                # motion did not finish and holds the connection that can end it.
                self.stop()
                raise
            return
        if wait is None:
            if not self._warned_no_wait:
                self._warned_no_wait = True
                self.logger.warning(
                    "the gripper driver %s cannot report motion completion, so commands are "
                    "fire-and-forget and the caller may move the arm while the fingers travel.",
                    type(drv).__name__,
                )
            return

    def stop(self) -> None:
        """Halt the jaws where they are. It does not release and does not command a width.

        On this protocol a halt clears ``GTO``; ``SPE 0`` is the minimum speed and stops nothing.
        A driver seam without a halt says so, because a caller must not come to believe the
        gripper was stopped when it was not.
        """
        drv = self._driver
        if drv is None:
            return
        halt = getattr(drv, "stop", None)
        if halt is None:
            self.logger.warning(
                "the gripper driver %s has no stop(); the jaws cannot be halted and will finish "
                "whatever motion is in flight.", type(drv).__name__,
            )
            return
        try:
            halt()
        except Exception as exc:  # noqa: BLE001 (a failed halt must not mask the original fault)
            self.logger.error("the gripper did not stop: %s", exc)

    def get_width_mm(self) -> float:
        """Read the current opening width in millimetres."""
        drv = self._require_connected()
        count = int(drv.get_current_position())
        return float(self._count_to_mm(count))

    def is_object_detected(self) -> bool:
        """Whether the fingers stalled on something rather than reaching the commanded width.

        This is the only post-grasp evidence the Robotiq protocol offers, read through
        ``robotiq_socket.object_status()``. It makes this class an ``ObjectDetectingGripper``, as
        ``jaw_io``, ``onrobot``, ``vacuum`` and the sim gripper are, so the fail-closed gate in
        ``GraspExecutionPolicy`` runs for this gripper and an empty close is not recorded as a
        successful pick.

        The register distinguishes the two cases that matter::

            STOPPED_CLOSING   the fingers stalled while closing   -> a part is held
            AT_POSITION       the fingers reached the target      -> holding nothing

        It fails closed, on purpose. A driver behind the ``driver_factory`` seam that cannot answer
        returns ``False`` here, because no evidence is not a hold, and a grasp nobody can confirm is
        not claimed. A seam double that wants the gate to pass implements ``object_status()``.
        """
        drv = self._require_connected()
        status_fn = getattr(drv, "object_status", None)
        if status_fn is None:
            self.logger.warning(
                "the gripper driver %s cannot report gOBJ, so a grasp cannot be confirmed and this "
                "reports NOT holding. Implement object_status() on the driver seam to enable the "
                "post-close check.", type(drv).__name__,
            )
            return False
        status = status_fn()
        # STOPPED_OPENING counts too: the jaws were obstructed on the way out, which is still contact
        # with something. AT_POSITION and MOVING are not evidence of a hold.
        return int(status) in (1, 2)

    def hold_evidence(self) -> HoldEvidence:
        """gOBJ as evidence: 1 or 2 is HELD, 3 is EMPTY.

        0, the fingers still moving, and a seam that cannot report gOBJ are UNMEASURED.
        """
        drv = self._require_connected()
        status_fn = getattr(drv, "object_status", None)
        if status_fn is None:
            return HoldEvidence.UNMEASURED
        status = int(status_fn())
        if status in (1, 2):
            return HoldEvidence.HELD
        if status == 3:
            return HoldEvidence.EMPTY
        return HoldEvidence.UNMEASURED

    def grip_from_inside(self, *, speed: float | None = None, force: float | None = None) -> None:
        """Open the fingers as far as they go at ``speed`` and ``force``, onto a bore's wall (``core.gripper.GripsFromInside``):
        the wall stops them, gOBJ 1, held."""
        self.set_width_mm(self.max_width_mm, speed=speed, force=force)
        self._from_inside = True

    def holds_from_inside(self) -> bool:
        """Whether the last command gripped from inside (:meth:`grip_from_inside`) and the fingers stopped on something
        while opening (gOBJ 1): a part held from inside a bore, which a release lets go by closing. A release that
        opened onto something is no such hold, and a seam that cannot report gOBJ says no."""
        if not self._from_inside:
            return False
        drv = self._require_connected()
        status_fn = getattr(drv, "object_status", None)
        return status_fn is not None and int(status_fn()) == 1

    def width_is_measured(self) -> bool:
        """``get_width_mm`` reads the fingers' position register."""
        return True

    @property
    def closes_onto_the_part(self) -> bool:
        """``True``: the fingers stop on what they close on at the force asked, and ``gOBJ`` says they did
        (``core.gripper.ClosesOntoThePart``), so a grasp closes this hand as far as it goes."""
        return True

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
        # Into the mechanism's own range: closed as far as the fingers go. Clamped at min_width_mm, a close stopped at
        # 5 mm, and a velcro band the jaws squeezed flat to 5.1 mm read as nothing held (the owner's cell, 2026-10-15).
        lo = self.closed_width_mm
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


# ---- the driver as the stack reads it (P0g slice 8, grippers/spec.py) -----------------------------------------------
def _refuse(robot_cfg: Any, facts: Any) -> Any:
    """The Robotiq lives on the UR controller's tool I/O, reached over a socket the URCap opens at that controller's
    address, so two separate facts have to hold.

    The arm in hand is asked (``facts``), not ``robot_cfg.vendor``. A supplied handle replaces the arm construction
    step, so from that point the config does not describe the arm the cell runs on: asked instead, a ``vendor: ur``
    tree beside a ``DummyRobotArm`` would yield a real Robotiq driver aimed at a real controller address, next to an
    arm holding no controller connection, and nothing on the built cell would say so. The two I/O drivers ask the
    handle too (``facts.has_digital_io``), for the same reason.

    The address comes off the arm as well: ``capabilities.controller`` is the controller this arm talks to (since P0e;
    before that it was read off the tree the UR driver keeps, ``arm.config.ur.ip``, which nothing but that driver
    declares). The tree is the wrong source for a supplied arm: it reads the UR block whatever the tree names, and
    until 2026-10-13 that block carried 192.168.1.100 as a schema default, a plausible address on a real subnet that
    nobody chose. The default is gone (a cell names its controller or is refused), and the arm still answers for
    itself.
    """
    from src.robot.core import RobotVendor

    from .null import SubstitutionReason
    from .spec import Refusal

    if facts.vendor != RobotVendor.UR.value:
        # A config requesting Robotiq on a non-UR arm (notably the Isaac sim, whose `sim` profile sets
        # gripper.vendor=robotiq + arm vendor=sim) cannot build a real gripper through from_robot_config. Warn instead
        # of silently booting gripper-less; the sim builds its IsaacGripper via
        # AutonomousGraspService.from_components (shared session).
        return Refusal(
            SubstitutionReason.ROBOTIQ_NEEDS_UR,
            f"gripper.vendor='robotiq' but the arm in hand reports vendor {facts.vendor!r}, "
            f"which is not a UR. A Robotiq lives on the UR controller's tool I/O, so an arm "
            f"that is not on such a controller cannot reach one.",
            "For the Isaac sim use AutonomousGraspService.from_components (the IsaacGripper "
            "needs the shared session). On a real cell, set robot.vendor: ur and either let "
            "this build the arm or hand in the UR arm you built yourself.",
        )
    if not facts.controller:
        # The other half of the same refusal, and it is stated separately because an operator cannot act on the two the
        # same way. Falling back to `robot_cfg.ur.ip` here is the thing that must not happen: on a tree that names no UR
        # that value is whatever the unselected block holds, and pointing a real driver at an address nobody chose is
        # exactly what this half of the refusal prevents.
        return Refusal(
            SubstitutionReason.ROBOTIQ_NEEDS_UR,
            f"gripper.vendor='robotiq' and the arm in hand does report vendor "
            f"{RobotVendor.UR.value!r}, but it names no controller address: its "
            f"capabilities.controller is None. The Robotiq is reached over a socket on "
            f"that arm's controller, and robot.ur.ip in this tree ({robot_cfg.ur.ip!r}) "
            f"describes whatever the config names, not the arm that was handed in.",
            "Hand in an arm that names its controller (URRobotArm reads robot.ur.ip of the "
            "tree it was built from), or supply no arm at all and set robot.vendor: ur so this "
            "builds the arm from this tree and both halves come from the same place.",
        )
    return None


def _build(robot_cfg: Any, arm: Any, facts: Any) -> Any:
    from src.robot.core import GripperVendor

    from .registry import create_gripper

    # When `resolve_arm` built the arm itself it built it from this tree (`create_arm(RobotVendor.UR,
    # config=robot_cfg)`), so the shipped no-handle UR cell reads the address it always read.
    return create_gripper(GripperVendor.ROBOTIQ, config=robot_cfg.gripper, ip=str(facts.controller))


def _wiring(robot_cfg: Any) -> Any:
    from .spec import Row

    return Row("bench", "the Robotiq URCap on the controller (it opens port 63352, not any pip package)",
               "confirm physically; a missing URCap looks like a gripper that never answers")


def _hazards(robot_cfg: Any, gripper: Any, asked: str) -> Any:
    from .spec import Hazard

    return (Hazard(
        what="The Robotiq activates on connect, and activation is a CALIBRATION: the fingers "
             "travel their entire range, at full speed, immediately.",
        precaution="Keep hands clear of the jaws. Make sure nothing is between them: activation "
                   "with a workpiece in the gripper is how a part gets crushed or launched.",
    ),)


def _driver() -> Any:
    from src.robot.core import GripperVendor

    from .spec import GripperDriver

    # No SDK. Naming `robotiq_gripper` would be unsatisfiable, because it is not a package: `pip index versions
    # robotiq_gripper` answers that no matching distribution was found, and the installed `ur_rtde` ships nothing by
    # that name. It is one example file from SDU's repository that an operator drops on the path by hand, so the doctor
    # would read an uninstalled SDK on every box, correctly, and no install would fix it. `robotiq_socket.py` speaks
    # the URCap port-63352 grammar directly.
    return GripperDriver(vendor=GripperVendor.ROBOTIQ, refuse=_refuse, build=_build, wiring=_wiring,
                         hazards=_hazards, sdk_modules=(), opens_as_needed=True)


DRIVER = _driver()
