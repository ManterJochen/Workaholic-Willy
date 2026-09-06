"""Isaac Sim suction :class:`Gripper` driver, the surface-gripper vacuum cup.

``IsaacSuctionGripper`` is the second end-effector modality behind the vendor-neutral
:class:`~src.robot.core.Gripper` Protocol. Instead of a parallel jaw it drives an Isaac
surface gripper, a binary proximity-joint vacuum surrogate authored on the UR5e wrist
by :func:`src.willy_sim.suction_mount.mount_suction_cup`. Width is reinterpreted as
vacuum on or off:

* ``set_width_mm(w)`` at or below ``vacuum_on_below_mm`` engages, stepping and polling
  until the surface gripper reads ``Closed`` with a body bonded;
* above that it releases.

So the same
:class:`~src.robot.grasping.motion.execution_policy.GraspExecutionPolicy` choreography
of pre-open during descent, close at contact and retreat lift drives a suction pick
unchanged: the runner sets ``pre_open_width_mm`` high for vacuum off on the way down
and ``close_width_mm`` low for vacuum on at contact. The gripper advertises
:class:`~src.robot.core.ObjectDetectingGripper`, so post-close verification polls the
real bond status.

Which cup, its contact radius, shaft and vacuum threshold, is a swappable
:class:`SuctionCupProfile`, exactly as a parallel-gripper brand is a
:class:`~src.robot.grippers.sim.gripper.GripperProfile`. A finer cup for tight gaps is
a new profile, :data:`SLIM_SUCTION_CUP`, not a new driver.

The scope is narrow. A binary proximity joint models the attach and the lift, whether a
cup at the contact holds the part through the move. It models no seal quality, air leak
or cup deformation; those live in the analytical scorer under ``grasping/suction/``,
and this driver realises in sim where the synthesis chose. ``mock_mode`` keeps a
pure-Python on or off cache and touches no Isaac. Every ``isaacsim.*`` import is lazy,
inside the connected non-mock paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core import RobotConnectionError

__all__ = [
    "SuctionCupProfile",
    "STANDARD_SUCTION_CUP",
    "SLIM_SUCTION_CUP",
    "IsaacSuctionGripper",
]


@dataclass(frozen=True)
class SuctionCupProfile:
    """A named suction end-effector: cup geometry and vacuum threshold.

    This is the single source of truth for a cup shape, read by the driver below, by
    the visible-cup author :func:`willy_sim.suction_mount.mount_visible_suction_cup`
    and by the collision envelope
    :class:`grasping.collision.SuctionCupGripperModel`, so which cup is one decision
    rather than three. A different size is a new profile, reused by the same
    :class:`IsaacSuctionGripper`.

    All dimensions are millimetres. ``max_width_mm``, the width band the grasp policy
    clamps to and the vacuum-off width, is the contact diameter from
    :attr:`cup_radius_mm`.
    """

    name: str
    cup_radius_mm: float       # contact-cup radius (the seal footprint / collision + visual radius)
    cup_height_mm: float       # cup depth along the approach axis
    shaft_radius_mm: float     # the stem behind the cup (wrist flange -> cup)
    shaft_length_mm: float     # stem length
    vacuum_on_below_mm: float  # set_width_mm at or below this engages the vacuum, above it releases
    # Optional real cup mesh, such as an Isaac ``*/grippers/short_gripper.usd``, for a
    # realistic visual. ``None`` uses primitive cylinders. It is resolved on-box and
    # ignored in mock mode.
    visual_usd_asset: str | None = None

    @property
    def max_width_mm(self) -> float:
        """Upper bound of the nominal width band, which is the cup contact diameter."""
        return 2.0 * self.cup_radius_mm


#: The default cup: a 30 mm contact diameter on a slim shaft, matching the
#: collision-envelope defaults.
STANDARD_SUCTION_CUP = SuctionCupProfile(
    name="standard",
    cup_radius_mm=15.0,
    cup_height_mm=25.0,
    shaft_radius_mm=10.0,
    shaft_length_mm=40.0,
    vacuum_on_below_mm=5.0,
)

#: A finer cup, 20 mm contact diameter on a thin shaft, for tight gaps and small flat
#: faces.
SLIM_SUCTION_CUP = SuctionCupProfile(
    name="slim",
    cup_radius_mm=10.0,
    cup_height_mm=15.0,
    shaft_radius_mm=5.0,
    shaft_length_mm=40.0,
    vacuum_on_below_mm=5.0,
)


class IsaacSuctionGripper:
    """Isaac surface-gripper :class:`Gripper`, vacuum on or off through width reinterpretation.

    Parameters
    ----------
    session
        The shared ``IsaacSimSession``, stepped to let the bond form or release.
    gripper_prim_path
        Prim path of the surface-gripper schema prim, from :func:`mount_suction_cup`.
    profile
        The cup :class:`SuctionCupProfile`, defaulting to the standard cup.
    max_width_mm, vacuum_on_below_mm
        Optional overrides. ``None`` uses the profile values.
    mock_mode
        A pure-Python on or off cache, touching no Isaac.
    settle_timeout_s
        Longest wait in sim seconds for the bond to form after commanding vacuum on.
    """

    def __init__(
        self,
        *,
        session: object,
        gripper_prim_path: str | None,
        profile: SuctionCupProfile = STANDARD_SUCTION_CUP,
        max_width_mm: float | None = None,
        vacuum_on_below_mm: float | None = None,
        mock_mode: bool = False,
        settle_timeout_s: float = 2.0,
    ) -> None:
        self._session = session
        self._prim_path = gripper_prim_path
        self._profile = profile
        self._max_width_mm = float(max_width_mm) if max_width_mm is not None else profile.max_width_mm
        self._vacuum_on_below_mm = (
            float(vacuum_on_below_mm) if vacuum_on_below_mm is not None else profile.vacuum_on_below_mm
        )
        self._mock_mode = mock_mode
        self._settle_timeout_s = settle_timeout_s
        self._connected = False
        self._view: object | None = None
        # The cache starts off, meaning open. In mock mode is_object_detected mirrors
        # the last vacuum command.
        self._vacuum_on = False

    # ---- introspection --------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def profile(self) -> SuctionCupProfile:
        return self._profile

    @property
    def min_width_mm(self) -> float:
        return 0.0

    @property
    def max_width_mm(self) -> float:
        return self._max_width_mm

    # ---- lifecycle ------------------------------------------------------

    def connect(self) -> None:
        """Wrap the surface-gripper view outside mock mode. Idempotent. The session must be started."""
        if self._connected:
            return
        if self._mock_mode:
            self._connected = True
            return
        if not self._prim_path:
            raise RobotConnectionError("IsaacSuctionGripper requires a gripper_prim_path (non-mock).")
        from isaacsim.robot.surface_gripper import GripperView  # type: ignore[import-not-found]

        self._view = GripperView(paths=self._prim_path)
        self._connected = True

    def disconnect(self) -> None:
        self._view = None
        self._connected = False

    def activate(self) -> None:
        """A no-op in sim, where a real vacuum generator would spin up."""
        return None

    # ---- commands -------------------------------------------------------

    def set_width_mm(
        self,
        width_mm: float,
        *,
        speed: float | None = None,
        force: float | None = None,
    ) -> None:
        """Reinterpret width as vacuum on or off, engaging at or below ``vacuum_on_below_mm``.

        ``speed`` and ``force`` are accepted for Protocol parity and not applied: the
        surface gripper force limits are set at mount time.
        """
        if not self._connected:
            raise RobotConnectionError("IsaacSuctionGripper is not connected.")
        vacuum_on = float(width_mm) <= self._vacuum_on_below_mm
        self._vacuum_on = vacuum_on
        if self._mock_mode or self._view is None:
            return
        # The Isaac surface-gripper action is a signed command decoded from the gantry
        # example: +0.5 engages and -0.5 releases. +1.0 did not transition the status
        # out of Open on-box.
        action = np.array([0.5 if vacuum_on else -0.5])
        self._view.apply_gripper_action(action)  # type: ignore[attr-defined]
        self._settle(expect_closed=vacuum_on)

    def get_width_mm(self) -> float:
        """Nominal width: 0 while the vacuum is on and gripping, ``max_width_mm`` while off."""
        if not self._connected:
            raise RobotConnectionError("IsaacSuctionGripper is not connected.")
        return 0.0 if self._vacuum_on else self._max_width_mm

    def open(self) -> None:
        """Release the vacuum, which the Protocol expresses as ``max_width_mm``."""
        self.set_width_mm(self._max_width_mm)

    def close(self) -> None:
        """Engage the vacuum, which the Protocol expresses as ``min_width_mm``."""
        self.set_width_mm(0.0)

    # ---- ObjectDetectingGripper -----------------------------------------

    def is_object_detected(self) -> bool:
        """``True`` only where the surface gripper has bonded a body: status Closed with an object."""
        if not self._connected:
            raise RobotConnectionError("IsaacSuctionGripper is not connected.")
        if self._mock_mode or self._view is None:
            return self._vacuum_on
        return self._is_closed()

    # ---- internals ------------------------------------------------------

    def _is_closed(self) -> bool:
        if self._view is None:
            return False
        try:
            status = list(self._view.get_surface_gripper_status())  # type: ignore[attr-defined]
            gripped = list(self._view.get_gripped_objects())  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 (status read is best-effort)
            return False
        closed = bool(status) and str(status[0]) == "Closed"
        return closed and bool(gripped) and any(bool(g) for g in gripped)

    def _settle(self, *, expect_closed: bool) -> bool:
        """Step the session until the bond forms or releases, or until the timeout expires."""
        if self._view is None:
            return True
        cfg = getattr(self._session, "config", None)
        dt = getattr(cfg, "step_dt_s", 0.0) or 1.0 / 60.0
        for _ in range(max(1, int(self._settle_timeout_s / dt))):
            self._session.step()  # type: ignore[attr-defined]
            if self._is_closed() == expect_closed:
                return True
        return False
