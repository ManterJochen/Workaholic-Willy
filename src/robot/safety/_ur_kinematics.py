"""Universal Robots Denavit-Hartenberg kinematics, used by the self-collision guard to derive per-link transforms.

The rows come from the arm registry (``config/arms/<model>.yaml``) through :func:`ur_dh_rows`. They were hand-typed
here as ``UR_DH_TABLES_M`` until P0d D4 deleted the table; the numbers did not change, and
``tests/test_inlined_dh_tables.py`` holds them, exactly, to the copies the committed collision bundles were baked
through.

The DH convention is the standard Craig-style one, with alpha, a, d and theta_offset, and the joint variable added
to ``theta_offset``. Lengths are metres internally, and the consumer scales to mm for the capsule pipeline.

Sources
-------
Universal Robots' own numbers, twice over: the descriptions agree with UR's pinned ROS 2 description to float noise
(``tests/test_arm_descriptions_are_urs_numbers.py``), and with UR's support-site page "DH parameters for
calculations of kinematics and dynamics", from which the old table was typed:
https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/

What has kinematics here
------------------------
* Only a UR the arm registry describes. Anything else returns ``None``, and the
  self-collision guard then treats an arm-against-arm link check as unavailable while
  still running the tool capsule and fixture checks.
* The DH theta-offsets are zero for UR, because the controller ``q`` already
  corresponds to the canonical zero pose ``(0, 0, 0, 0, 0, 0)``.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

__all__ = [
    "URDhChain",
    "URDhRow",
    "arm_link_transforms_mm",
    "arm_link_transforms_mm_many",
    "described_ur_model",
    "ur_dh_rows",
    "ur_link_origins_mm",
    "ur_link_transforms_mm",
    "ur_link_transforms_mm_many",
    "ur_joint_radii_mm",
    "ur_series_twin",
]


@dataclass(frozen=True, slots=True)
class URDhRow:
    """One row of standard UR DH parameters, with lengths in metres."""

    a_m: float
    d_m: float
    alpha_rad: float


@functools.lru_cache(maxsize=None)
def _described_rows(key: str) -> tuple[URDhRow, ...] | None:
    from src.config import ConfigError  # noqa: PLC0415 (the config layer is below this one; lazy for import cost)
    from src.config.arms import load_arm  # noqa: PLC0415

    # A description, not an arm, and named so: the contract test reads `arm.<member>` as a member of a robot arm
    # (tests/test_robot_boundaries.py, rule III.b), and a description's `vendor` and `joints` are not one.
    try:
        description = load_arm(key)
    except ConfigError:
        return None
    if description.vendor != "ur":
        return None
    return tuple(URDhRow(joint.a_m, joint.d_m, joint.alpha_rad) for joint in description.joints)


def ur_dh_rows(model: str | None) -> tuple[URDhRow, ...] | None:
    """This UR's DH rows, from the arm registry, or ``None`` where the registry describes no such UR.

    The one place the rows come from, since P0d D3b. Every function below read ``UR_DH_TABLES_M`` directly, and so did
    the exact-mesh guard's model check, the closed-form IK, the sidecar's model guess, the sim's reach helper and
    datagen's robot resolver. They all ask here now, and this asks ``config/arms/<model>.yaml``. The table is gone (D4);
    what holds these floats now is ``tests/test_inlined_dh_tables.py``, which compares them, exactly, with the copies
    inlined in the three scripts that baked the committed collision bundles, because those are the floats the geometry
    was cut from.

    Cached per model: the continuous monitor asks on every sample of a path, and the registry is read once per process
    (``src.config.arms._repository_registry``), so the cost of asking is one dictionary hit.

    Wider than the old table by one arm, on purpose: ur20 is described from Universal Robots' pinned description and
    had no row there. So ur20 has kinematics, and the exact-mesh guard's status for it moves from ``unknown_model`` to
    ``no_bundle``, which is the true answer. No cell can reach it, because its description says ``admitted: false``.
    """
    if not model:
        return None
    return _described_rows(model.lower())


def described_ur_model(model: str | None) -> str | None:
    """``model`` where the arm registry describes a UR by that name, else ``None``: the one rule for "is this a UR".

    Since P0d D5 the only gate between an arm and UR geometry. The self-collision guard asks it of the model the arm
    reports (``SelfCollisionGuard.model_for``), and the build-time hand refusal, the wrist body reader, the desk
    checklist and the half-turn window's elbow ask it of the model the cell names, so all of them answer alike wherever
    the cell names the model its arm reports (every driver that exists does; ``planning/hand.py`` says where it could
    not). The vendor does not enter: the Isaac arm is ``vendor="sim"`` and drives
    the UR it names, and a UR driven over ROS 2 will be another vendor again. A model nobody describes, the dummy's
    ``dummy-6dof`` or another vendor's, gets ``None`` and no chain.
    """
    if not model or ur_dh_rows(model) is None:
        return None
    return str(model)


def ur_series_twin(model: str | None) -> str | None:
    """The arm of the other series with the same size, or None when this model has no twin.

    ``ur3 <-> ur3e``, ``ur5 <-> ur5e``, ``ur10 <-> ur10e``. ``ur16e`` has none, because no UR16
    CB-series row is bundled here.

    A twin is worth naming because these are the two robots a cell can confuse without any
    check noticing. A controller dashboard reports the same string for both (URSim answers
    ``UR3`` for a UR3e, measured), so the size-class comparison in the UR driver is blind to
    the difference by construction rather than by omission.

    Measured over 20000 random joint vectors, the flange separation between a twin pair::

        ur3  vs ur3e    min  8.50   median 21.26   max 28.90 mm
        ur5  vs ur5e    min 59.17   median 79.34   max 95.26 mm
        ur10 vs ur10e   min 28.43   median 59.78   max 80.31 mm

    So a ur3/ur3e swap sits under the UR driver default 10 mm tool-frame tolerance in 12.2 % of
    poses and the other two pairs never do. A cell on the wrong one of that pair plans against
    another robot link lengths and nothing downstream says so.
    """
    if not model:
        return None
    key = model.lower()
    twin = key[:-1] if key.endswith("e") else key + "e"
    return twin if ur_dh_rows(twin) is not None and twin != key else None


def _dh_transform(theta: float, row: URDhRow) -> np.ndarray:
    """The standard 4x4 DH transform, in metres."""
    ct = float(np.cos(theta))
    st = float(np.sin(theta))
    ca = float(np.cos(row.alpha_rad))
    sa = float(np.sin(row.alpha_rad))
    a = row.a_m
    d = row.d_m
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)


def ur_link_origins_mm(model: str, joints_rad: np.ndarray) -> list[np.ndarray] | None:
    """Per-joint frame origins for a UR model, in mm in the robot base frame.

    The list has length ``len(joints_rad) + 1``: index 0 is the base-frame origin
    ``[0,0,0]`` and the last entry is the flange origin, meaning the TCP without a
    tool. It returns ``None`` where ``model`` is not in the bundled DH table.
    """
    table = ur_dh_rows(model)
    if table is None:
        return None
    if joints_rad.shape[0] != len(table):
        return None
    T = np.eye(4, dtype=np.float64)
    origins_m: list[np.ndarray] = [T[:3, 3].copy()]
    for theta, row in zip(joints_rad, table, strict=True):
        T = T @ _dh_transform(float(theta), row)
        origins_m.append(T[:3, 3].copy())
    return [o * 1000.0 for o in origins_m]


def ur_joint_radii_mm(model: str) -> tuple[float, ...] | None:
    """Per joint, an upper bound in mm on how far any point of the arm beyond it is from its axis.

    The arm here ends at the flange. What the flange carries is not in the bound:
    ``SafetyPreflight.joint_radii_mm`` adds it.

    Rotating joint ``j`` by ``dq`` sweeps every point distal to it through an arc of at most
    ``dq * radius[j]``. The radius is the sum of every link length and joint offset from that joint
    outwards. Adding them all over-estimates, because the ``d`` offsets run perpendicular to the ``a``
    lengths and never stack end to end, and that is the safe direction for a bound that sets how densely a
    path is sampled.

    Per joint rather than one number for the arm, because the wrist carries a hand while the shoulder
    carries the whole arm, and holding both to the shoulder's radius samples a wrist move seventeen
    times more densely than the geometry asks for. On a ur5e a six joint move of 10 rad in total needs
    1312 samples under one reach and 743 under these, so the difference decides whether an ordinary
    move fits under the checker's cap at all. Both are upper bounds; this one keeps what the table
    already knows.
    """
    table = ur_dh_rows(model)
    if table is None:
        return None
    return tuple(
        float(sum(abs(row.a_m) + abs(row.d_m) for row in table[index:]) * 1000.0)
        for index in range(len(table))
    )


def ur_link_transforms_mm(model: str, joints_rad: np.ndarray) -> list[np.ndarray] | None:
    """Full per-link 4x4 transforms for a UR model, with the translation column in mm.

    It walks the same DH chain as :func:`ur_link_origins_mm` and returns the whole frame
    rather than the origin alone, so the exact-mesh backend can place the collision mesh
    of each link and not only its capsule end-points. The list has length
    ``len(joints_rad) + 1``: index 0 is the identity base frame and the last entry is
    the flange, the TCP without a tool. The rotation block is unitless and the
    translation column is in mm, so ``T @ [x_mm, y_mm, z_mm, 1]`` maps a mm-space mesh
    vertex straight into the mm base frame. It returns ``None`` where ``model`` is not
    in the bundled DH table or the joint count does not match.
    """
    table = ur_dh_rows(model)
    if table is None or joints_rad.shape[0] != len(table):
        return None
    T = np.eye(4, dtype=np.float64)
    out: list[np.ndarray] = [T.copy()]
    for theta, row in zip(joints_rad, table, strict=True):
        T = T @ _dh_transform(float(theta), row)
        M = T.copy()
        M[:3, 3] = M[:3, 3] * 1000.0  # metres -> mm in the translation column only
        out.append(M)
    return out


def ur_link_transforms_mm_many(model: str, joints_rad: np.ndarray) -> np.ndarray | None:
    """:func:`ur_link_transforms_mm` of every row of ``joints_rad``, ``(N, 6)``, at once: ``(N, 7, 4, 4)``.

    The same chain, multiplied in the same order with the same numbers: each frame is the one before it times the row's
    DH transform, in metres, and the translation column of the copy handed out is scaled to mm. Every frame is bit for
    bit the one :func:`ur_link_transforms_mm` gives that row (``tests/test_the_arm_measures_itself_only_where_it_could_
    refuse.py`` holds it to that on 20,000 configurations), so a whole path judged at once places every part where one
    sample judged at a time places it, about 30 times faster. It returns ``None`` where ``model`` is not in the bundled
    DH table or ``joints_rad`` is not one row of the table's joint count per configuration.
    """
    table = ur_dh_rows(model)
    rows = np.asarray(joints_rad, dtype=np.float64)
    if table is None or rows.ndim != 2 or rows.shape[1] != len(table):
        return None
    count = rows.shape[0]
    out = np.empty((count, len(table) + 1, 4, 4), dtype=np.float64)
    T = np.broadcast_to(np.eye(4, dtype=np.float64), (count, 4, 4)).copy()
    out[:, 0] = T
    for joint, row in enumerate(table):
        ct, st = np.cos(rows[:, joint]), np.sin(rows[:, joint])
        ca, sa = float(np.cos(row.alpha_rad)), float(np.sin(row.alpha_rad))
        # _dh_transform's matrix, one per configuration, entry by entry as it writes them.
        A = np.zeros((count, 4, 4), dtype=np.float64)
        A[:, 0, 0], A[:, 0, 1], A[:, 0, 2], A[:, 0, 3] = ct, -st * ca, st * sa, row.a_m * ct
        A[:, 1, 0], A[:, 1, 1], A[:, 1, 2], A[:, 1, 3] = st, ct * ca, -ct * sa, row.a_m * st
        A[:, 2, 1], A[:, 2, 2], A[:, 2, 3] = sa, ca, row.d_m
        A[:, 3, 3] = 1.0
        T = T @ A
        out[:, joint + 1] = T
        out[:, joint + 1, :3, 3] *= 1000.0  # metres -> mm in the translation column only
    return out


def arm_link_transforms_mm(model: str, joints_rad: np.ndarray, arm: object = None) -> list[np.ndarray] | None:
    """:func:`ur_link_transforms_mm` as the exact guard places ``arm``: every link by ``model``'s nominal table, and the
    flange where ``arm``'s controller puts it (:class:`~src.robot.core.arm_capabilities.KnowsItsFlange`), so what hangs
    from it, the hand, its fingers and a wrist camera, stands where the controller drives it (the owner, 2026-10-12). The
    nominal flange where ``arm`` does not say; ``None`` where :func:`ur_link_transforms_mm` answers ``None``.
    """
    from src.robot.core.arm_capabilities import controller_flanges_of  # noqa: PLC0415 (the core imports the safety)

    q = np.asarray(joints_rad, dtype=np.float64)
    transforms = ur_link_transforms_mm(model, q)
    if transforms is not None:
        flange = controller_flanges_of(arm, q.reshape(1, -1))
        if flange is not None:
            transforms[-1] = flange[0]
    return transforms


def arm_link_transforms_mm_many(model: str, joints_rad: np.ndarray, arm: object = None) -> np.ndarray | None:
    """:func:`ur_link_transforms_mm_many` as :func:`arm_link_transforms_mm` places ``arm``: every row's flange where its
    controller puts it, bit for bit the flange :func:`arm_link_transforms_mm` places that row with."""
    from src.robot.core.arm_capabilities import controller_flanges_of  # noqa: PLC0415 (the core imports the safety)

    transforms = ur_link_transforms_mm_many(model, joints_rad)
    if transforms is not None:
        flanges = controller_flanges_of(arm, np.asarray(joints_rad, dtype=np.float64))
        if flanges is not None:
            transforms[:, -1] = flanges
    return transforms


@dataclass(frozen=True, slots=True)
class URDhChain:
    """A UR's six DH rows as one controller holds them, each joint's theta offset included; lengths in metres.

    The tables above are the nominal ones of the UR support page, with no theta offset. A real arm is calibrated at the
    factory: its controller keeps a theta, a, d and alpha of its own for every joint and computes its forward and inverse
    kinematics on them, the poses ``getForwardKinematics`` reports and the joints ``getInverseKinematics`` answers. The
    owner's UR10 CB3 puts its TCP 6.0 to 6.3 mm and 0.6 degrees from where the nominal table puts it at the same joints
    (the guard map, 2026-10-08). The controller says what its rows are in the kinematics info of its robot state, which
    :meth:`~src.robot.drivers.ur.connection.URConnection.controller_kinematics` reads; URSim CB3 3.15.8 reports the
    nominal table plus whatever its ``calibration.conf`` adds, measured.

    The convention of :func:`ur_link_transforms_mm`, with the joint variable added to the row's ``theta_rad``. A calibrated
    chain may hold a frame far along a joint axis, a ``d`` of metres that the next row's cancels (nearly parallel axes meet
    their common normal far away); the product is the same flange either way.
    """

    theta_rad: tuple[float, ...]
    a_m: tuple[float, ...]
    d_m: tuple[float, ...]
    alpha_rad: tuple[float, ...]

    def __post_init__(self) -> None:
        rows = (self.theta_rad, self.a_m, self.d_m, self.alpha_rad)
        if any(len(row) != 6 for row in rows) or not np.all(np.isfinite(np.asarray(rows, dtype=np.float64))):
            raise ValueError(f"a UR chain is six finite rows of theta, a, d and alpha, got {rows!r}")

    @classmethod
    def nominal(cls, model: str) -> "URDhChain | None":
        """The rows the arm registry describes for ``model`` as a chain, or ``None`` for an arm it does not describe."""
        table = ur_dh_rows(model)
        if table is None:
            return None
        return cls(
            theta_rad=(0.0,) * len(table),
            a_m=tuple(row.a_m for row in table),
            d_m=tuple(row.d_m for row in table),
            alpha_rad=tuple(row.alpha_rad for row in table),
        )

    def frames_m(self, joints_rad: "np.ndarray | list[float] | tuple[float, ...]") -> list[np.ndarray]:
        """The base frame and the frame after each joint, the last the flange, as 4x4s in metres."""
        q = np.asarray(joints_rad, dtype=np.float64).reshape(-1)
        if q.shape != (6,):
            raise ValueError(f"a UR configuration is six joint values, got {joints_rad!r}")
        turned = q + np.asarray(self.theta_rad, dtype=np.float64)
        ct, st = np.cos(turned), np.sin(turned)
        ca, sa = np.cos(np.asarray(self.alpha_rad)), np.sin(np.asarray(self.alpha_rad))
        T = np.eye(4, dtype=np.float64)
        out = [T]
        for j in range(6):
            a, d = self.a_m[j], self.d_m[j]
            T = T @ np.array([
                [ct[j], -st[j] * ca[j], st[j] * sa[j], a * ct[j]],
                [st[j], ct[j] * ca[j], -ct[j] * sa[j], a * st[j]],
                [0.0, sa[j], ca[j], d],
                [0.0, 0.0, 0.0, 1.0],
            ], dtype=np.float64)
            out.append(T)
        return out

    def flange_m(self, joints_rad: "np.ndarray | list[float] | tuple[float, ...]") -> np.ndarray:
        """Where this chain puts the flange (tool0) at ``joints_rad``, a 4x4 in metres in the base frame."""
        return self.frames_m(joints_rad)[-1]

    def flanges_mm(self, joints_rad: np.ndarray) -> np.ndarray:
        """Where this chain puts the flange at every row of ``joints_rad``, ``(N, 6)``, at once: ``(N, 4, 4)``, the rotation
        unitless and the translation in mm, as :func:`ur_link_transforms_mm_many` hands out its last frame.

        The product :func:`ur_link_transforms_mm_many` forms, each joint turned by its row's ``theta_rad`` as well, so the
        nominal chain answers that function's flange; every row is formed alone, the same whatever else is asked with it.
        """
        rows = np.asarray(joints_rad, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != 6:
            raise ValueError(f"UR configurations are rows of six joint values, got the shape {rows.shape}")
        turned = rows + np.asarray(self.theta_rad, dtype=np.float64)
        count = rows.shape[0]
        T = np.broadcast_to(np.eye(4, dtype=np.float64), (count, 4, 4)).copy()
        for joint in range(6):
            ct, st = np.cos(turned[:, joint]), np.sin(turned[:, joint])
            ca, sa = float(np.cos(self.alpha_rad[joint])), float(np.sin(self.alpha_rad[joint]))
            a, d = self.a_m[joint], self.d_m[joint]
            A = np.zeros((count, 4, 4), dtype=np.float64)
            A[:, 0, 0], A[:, 0, 1], A[:, 0, 2], A[:, 0, 3] = ct, -st * ca, st * sa, a * ct
            A[:, 1, 0], A[:, 1, 1], A[:, 1, 2], A[:, 1, 3] = st, ct * ca, -ct * sa, a * st
            A[:, 2, 1], A[:, 2, 2], A[:, 2, 3] = sa, ca, d
            A[:, 3, 3] = 1.0
            T = T @ A
        T[:, :3, 3] *= 1000.0  # metres -> mm in the translation column only
        return T
