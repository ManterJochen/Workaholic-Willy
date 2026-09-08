"""Isaac Sim-backed parallel-jaw :class:`Gripper` driver.

``IsaacGripper`` drives a gripper articulation in Isaac behind the vendor-neutral
:class:`~src.robot.core.Gripper` Protocol. Everything brand-specific sits in a
swappable :class:`GripperProfile`: the driven joint, the nonlinear width-to-angle
curve, and the extremes. A second brand is a new profile, not a new driver.

Measured on-box against a Robotiq 2F-85:

* It is a closed-loop mimic mechanism, so commanding ``finger_joint`` alone makes the
  other five joints follow 1:1 and the profile names one ``driven_joint``.
* In this Isaac USD ``finger_joint = 0`` is closed at 0 mm and ``0.8 rad`` is open at
  about 85 mm. That is measured, not the datasheet angle convention.
* The four-bar linkage makes the curve mildly nonlinear, so the profile carries the
  measured table and interpolates both ways.

``mock_mode`` keeps a pure-Python width cache and touches no Isaac, so this runs
without a GPU. Every ``isaacsim.*`` import is lazy, inside the connected non-mock
paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...core import IsaacNotAvailableError, RobotConnectionError

if TYPE_CHECKING:  # pragma: no cover (typing only; keeps the runtime edge out of drivers.sim)
    from src.robot.drivers.sim._isaac_protocols import IsaacArticulation
    from src.robot.drivers.sim.session import IsaacSimSession

__all__ = [
    "GripperProfile",
    "ROBOTIQ_2F85_PROFILE",
    "SCHUNK_EGU50_PROFILE",
    "SCHUNK_EZU35_PROFILE",
    "IsaacGripper",
]

# Settle threshold: the gripper-joint speed in rad/s below which the jaws count as stopped.
_GRIPPER_SETTLE_VEL = 1e-2


@dataclass(frozen=True)
class GripperProfile:
    """Brand-specific sim-gripper kinematics behind the vendor-neutral Gripper Protocol.

    A new brand or model is a new ``GripperProfile``, carrying its ``driven_joint`` and
    an in-sim calibrated ``angle_width_table``, reused by the same
    :class:`IsaacGripper`.

    ``angle_width_table`` is ``((angle_rad, width_mm), ...)``, sorted by ascending angle
    and monotonic in width as measured in sim. Both lookups interpolate linearly.
    """

    name: str
    driven_joint: str
    angle_width_table: tuple[tuple[float, float], ...]
    #: Joints that must be commanded to the same value as ``driven_joint``, for a hand whose
    #: asset does not couple them itself.
    #:
    #: Measured, and it is why this exists: the Robotiq Hand-E asset carries two independent
    #: prismatic sliders, no PhysxMimicJoint and no drive API at all, so commanding one closes
    #: one jaw and leaves the other where it is. That is a grip which looks like a grip in
    #: every width reading and holds nothing. Both Schunk entries below get away with a single
    #: joint because their assets do carry the mimic, which is a property of those files
    #: rather than of parallel grippers.
    #:
    #: Empty for every profile that had no second joint, so they command exactly what they did.
    mirrored_joints: tuple[str, ...] = ()

    @property
    def _angles(self) -> np.ndarray:
        return np.asarray([a for a, _ in self.angle_width_table], dtype=np.float64)

    @property
    def _widths(self) -> np.ndarray:
        return np.asarray([w for _, w in self.angle_width_table], dtype=np.float64)

    @property
    def min_width_mm(self) -> float:
        return float(self._widths.min())

    @property
    def max_width_mm(self) -> float:
        return float(self._widths.max())

    @property
    def min_angle(self) -> float:
        return float(self._angles.min())

    @property
    def max_angle(self) -> float:
        return float(self._angles.max())

    def width_to_angle(self, width_mm: float) -> float:
        """Map a jaw width in mm to the ``driven_joint`` angle in rad, clamped to range.

        It sorts by width because the table may decrease in width, as the 2F-85 jaw gap
        does while ``finger_joint`` grows, and ``np.interp`` needs an ascending x-array.
        """
        w = float(np.clip(width_mm, self.min_width_mm, self.max_width_mm))
        order = np.argsort(self._widths)
        return float(np.interp(w, self._widths[order], self._angles[order]))

    def angle_to_width(self, angle_rad: float) -> float:
        """Map a ``driven_joint`` angle in rad to the jaw width in mm."""
        order = np.argsort(self._angles)
        return float(np.interp(float(angle_rad), self._angles[order], self._widths[order]))


# Robotiq 2F-85, calibrated in Isaac Sim 5.1 against finger_joint with the gripper
# pointing down. The jaw gap is the world aabb gap between the inner edges of the inner
# fingers and shrinks as finger_joint grows: 0 is open at about 87 mm, 0.8 is closed at
# about 2 mm. Measure that gap, not the inner-finger origin separation, which is
# inverted from the jaw opening and drives the gripper backwards.
ROBOTIQ_2F85_PROFILE = GripperProfile(
    name="robotiq_2f85",
    driven_joint="finger_joint",
    angle_width_table=(
        (0.0, 87.1),
        (0.1, 77.8),
        (0.2, 68.1),
        (0.3, 57.9),
        (0.4, 47.2),
        (0.5, 36.2),
        (0.6, 25.0),
        (0.7, 13.6),
        (0.8, 2.2),
    ),
)


# Schunk EGU-50, a different-vendor parallel gripper, mounted on the UR5e through the
# mount harness: ur5e Gripper variant "None", the standalone egu_50.usd referenced, its
# articulation root dropped, and a fixed joint wrist_3_link to base merging it into the
# UR5e articulation. Its driven joint is prismatic ("Jaw_Drive", linear, metres) rather
# than the revolute finger_joint of the 2F-85, with a PhysxMimicJoint coupling the
# second jaw 1:1, so one driven joint still parametrises the gripper. The table is
# Jaw_Drive in metres against jaw width in mm, and since the profile interpolates a
# table the unit of the driven coordinate is irrelevant. Measured in sim as the FGR
# finger-mesh facing gap against the commanded Jaw_Drive: width_mm is about
# 4.1 + 2000*Jaw_Drive, so 4 mm closed to about 84 mm open, close enough to the 87 mm of
# the 2F-85 to keep the parallel grasp synthesis compatible.
SCHUNK_EGU50_PROFILE = GripperProfile(
    name="schunk_egu50",
    driven_joint="Jaw_Drive",
    angle_width_table=(
        (0.000, 4.1),
        (0.005, 14.1),
        (0.010, 24.1),
        (0.015, 34.1),
        (0.020, 44.1),
        (0.025, 54.1),
        (0.030, 64.1),
        (0.035, 74.1),
        (0.040, 84.1),
    ),
)


# Schunk EZU-35, a three-finger centric gripper on the same harness as the EGU-50, and
# structurally identical for the driver: one prismatic driven joint ("Jaw_Drive",
# linear, metres) with PhysxMimicJoints (gba2, GBA3_TRANS) coupling the other two
# fingers 1:1, so it is 1-DOF centric and the same IsaacGripper plus a width table
# parametrise it. The width for the parallel grasp synthesis is the three-finger
# inscribed diameter, the object the fingers close around. The table is Jaw_Drive in
# metres against diameter in mm, measured in sim as the FGR finger spread against the
# commanded Jaw_Drive.
SCHUNK_EZU35_PROFILE = GripperProfile(
    name="schunk_ezu35",
    driven_joint="Jaw_Drive",
    angle_width_table=(  # measured on the mounted gripper: the 3-finger inscribed grip diameter
        (0.000, 11.0),    # Jaw_Drive lower limit -> fingers nearly touching (~11 mm inscribed)
        (0.010, 31.0),
        (0.020, 51.0),
        (0.030, 71.0),
        (0.035, 81.0),    # Jaw_Drive upper limit -> open ~81 mm
    ),
)


# Robotiq Hand-E, the cell gripper from 2026-09-08. Measured off
# `Robotiq/Hand-E/Robotiq_Hand_E_edit.usd`, in millimetres, with the stroke read from the jaw
# faces rather than from a datasheet: they stand 49.99 mm apart at the authored pose and the
# joint range is 25 mm per finger, so the stroke closes to exactly 0.00 mm and the table is
# linear, width_mm = 49.99 - 2000 * q with q the slider position in metres.
#
# The direction is bucket 3. The asset says `Slider_1` carries a 180 degree rotation about X, so
# its local +Z is the body -Z and a rising joint value would drive the left jaw away from the
# centre. The geometry forbids that: the faces are already 49.99 mm apart and the range is 25 mm
# a side, so a rising value must close them or the gripper opens to 100 mm, which no Hand-E does.
# The geometry is right and the joint frames disagree with it. This table follows the geometry;
# the sign has to be confirmed on-box before anything drives this asset, and if it turns out
# inverted the fix is to reverse this table and nothing else.
#
# Two joints, because the asset has no PhysxMimicJoint and no drive API. Commanding `Slider_1`
# alone closes one jaw and leaves the other where it is, which is a grip that looks like a grip
# in every width reading and holds nothing.
ROBOTIQ_HANDE_PROFILE = GripperProfile(
    name="robotiq_hande",
    driven_joint="Slider_1",
    mirrored_joints=("Slider_2",),
    angle_width_table=(
        (0.000, 49.99),
        (0.005, 39.99),
        (0.010, 29.99),
        (0.015, 19.99),
        (0.020, 9.99),
        (0.025, 0.0),
    ),
)


class IsaacGripper:
    """Isaac-backed parallel-jaw :class:`Gripper`, driving one articulation through a profile.

    Parameters
    ----------
    session
        The shared :class:`IsaacSimSession`, stepped here to let the jaws settle. The
        arm owns it and the gripper shares it.
    gripper_prim_path
        Prim path of the gripper articulation.
    profile
        The brand-specific :class:`GripperProfile`, defaulting to the Robotiq 2F-85.
    mock_mode
        A pure-Python width cache, touching no Isaac.
    settle_timeout_s
        Longest wait in sim seconds for the jaws to stop after a command.
    """

    def __init__(
        self,
        *,
        session: IsaacSimSession,
        gripper_prim_path: str | None,  # schema field is str | None; connect() enforces non-empty
        profile: GripperProfile = ROBOTIQ_2F85_PROFILE,
        mock_mode: bool = False,
        settle_timeout_s: float = 2.0,
    ) -> None:
        self._session = session
        self._prim_path = gripper_prim_path
        self._profile = profile
        self._joint_indices: list[int] = []
        self._mock_mode = mock_mode
        self._settle_timeout_s = settle_timeout_s
        self._connected = False
        self._articulation: IsaacArticulation | None = None  # typed Isaac handle
        self._joint_index: int | None = None
        # The mock cache, and the pre-connect default: start fully open.
        self._mock_width_mm = profile.max_width_mm

    # ---- introspection --------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def min_width_mm(self) -> float:
        return self._profile.min_width_mm

    @property
    def max_width_mm(self) -> float:
        return self._profile.max_width_mm

    @property
    def profile(self) -> GripperProfile:
        return self._profile

    # ---- lifecycle ------------------------------------------------------

    def connect(self) -> None:
        """Wrap the gripper articulation outside mock mode. Idempotent. The session must be started."""
        if self._connected:
            return
        if self._mock_mode:
            self._connected = True
            return
        if not self._prim_path:
            raise RobotConnectionError("IsaacGripper requires a gripper_prim_path (non-mock).")
        from isaacsim.core.prims import SingleArticulation  # type: ignore[import-not-found]

        articulation = SingleArticulation(prim_path=self._prim_path)
        articulation.initialize()
        names = list(articulation.dof_names)
        wanted = (self._profile.driven_joint, *self._profile.mirrored_joints)
        missing = [joint for joint in wanted if joint not in names]
        if missing:
            raise IsaacNotAvailableError(
                f"joint(s) {missing} named by profile {self._profile.name!r} are not in "
                f"gripper dof_names {names}."
            )
        self._articulation = articulation
        # Every jaw this profile drives, the driven one first. A hand whose asset couples its
        # own jaws contributes one entry, which is what every profile did before this existed.
        self._joint_indices = [names.index(joint) for joint in wanted]
        self._joint_index = self._joint_indices[0]
        self._connected = True

    def disconnect(self) -> None:
        self._articulation = None
        self._joint_index = None
        self._joint_indices = []
        self._connected = False

    def activate(self) -> None:
        """A no-op in sim, where the real Robotiq runs a calibration sweep."""
        return None

    # ---- commands -------------------------------------------------------

    def set_width_mm(
        self,
        width_mm: float,
        *,
        speed: float | None = None,
        force: float | None = None,
    ) -> None:
        """Command the jaw opening to ``width_mm``, clamped to the profile range.

        ``speed`` and ``force`` are accepted for Protocol parity and not applied,
        because the articulation drive gains govern the close.
        """
        if not self._connected:
            raise RobotConnectionError("IsaacGripper is not connected.")
        target = float(np.clip(width_mm, self.min_width_mm, self.max_width_mm))
        if self._mock_mode or self._articulation is None or self._joint_index is None:
            self._mock_width_mm = target
            return
        from isaacsim.core.utils.types import ArticulationAction  # type: ignore[import-not-found]

        angle = self._profile.width_to_angle(target)
        # One value, every jaw this profile drives. For a hand whose asset couples its own jaws
        # that is a single index and the action is byte-identical to what it was.
        indices = self._joint_indices or [self._joint_index]
        self._articulation.apply_action(
            ArticulationAction(
                joint_positions=np.array([angle] * len(indices), dtype=np.float64),
                joint_indices=np.array(indices),
            )
        )
        self._settle()

    def get_width_mm(self) -> float:
        """Current jaw opening, in millimetres."""
        if not self._connected:
            raise RobotConnectionError("IsaacGripper is not connected.")
        if self._mock_mode or self._articulation is None or self._joint_index is None:
            return self._mock_width_mm
        q = np.asarray(self._articulation.get_joint_positions(), dtype=np.float64)
        return self._profile.angle_to_width(float(q[self._joint_index]))

    def open(self) -> None:
        """Open the jaws fully, to the profile ``max_width_mm``."""
        self.set_width_mm(self.max_width_mm)

    def close(self) -> None:
        """Close the jaws fully, to the profile ``min_width_mm``."""
        self.set_width_mm(self.min_width_mm)

    # ---- internals ------------------------------------------------------

    def _settle(self) -> bool:
        if self._articulation is None:
            return True
        dt = self._session.config.step_dt_s if self._session.config.step_dt_s > 0 else 1.0 / 60.0
        for _ in range(max(1, int(self._settle_timeout_s / dt))):
            self._session.step()
            vel = np.asarray(self._articulation.get_joint_velocities(), dtype=np.float64)
            if float(np.max(np.abs(vel))) < _GRIPPER_SETTLE_VEL:
                return True
        return False
