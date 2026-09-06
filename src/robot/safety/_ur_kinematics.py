"""Universal Robots Denavit-Hartenberg parameters, bundled for the self-collision guard.

The guard derives per-link transforms from them. The DH convention is the standard
Craig-style one, with alpha, a, d and theta_offset, and the joint variable added to
``theta_offset``. Lengths are metres internally, and the consumer scales to mm for the
capsule pipeline.

Sources
-------
The official Universal Robots kinematic data, from the support site under "DH
parameters for calculations of kinematics and dynamics":
https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/

What is bundled
---------------
* Only the standard 6-DoF UR models. Anything else returns ``None``, and the
  self-collision guard then treats an arm-against-arm link check as unavailable while
  still running the tool capsule and fixture checks.
* The DH theta-offsets are zero for UR, because the controller ``q`` already
  corresponds to the canonical zero pose ``(0, 0, 0, 0, 0, 0)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "URDhRow",
    "UR_DH_TABLES_M",
    "ur_link_origins_mm",
    "ur_link_transforms_mm",
]


@dataclass(frozen=True, slots=True)
class URDhRow:
    """One row of standard UR DH parameters, with lengths in metres."""

    a_m: float
    d_m: float
    alpha_rad: float


# Per-model six-row DH tables in metres and radians, from the UR support page linked
# above. Each row is one joint, and a link length is mostly the ``a`` of that row and
# the ``d`` of the next.

_UR3_DH: tuple[URDhRow, ...] = (
    URDhRow(0.0, 0.1519, 1.570796327),
    URDhRow(-0.24365, 0.0, 0.0),
    URDhRow(-0.21325, 0.0, 0.0),
    URDhRow(0.0, 0.11235, 1.570796327),
    URDhRow(0.0, 0.08535, -1.570796327),
    URDhRow(0.0, 0.0819, 0.0),
)

_UR3E_DH: tuple[URDhRow, ...] = (
    URDhRow(0.0, 0.15185, 1.570796327),
    URDhRow(-0.24355, 0.0, 0.0),
    URDhRow(-0.2132, 0.0, 0.0),
    URDhRow(0.0, 0.13105, 1.570796327),
    URDhRow(0.0, 0.08535, -1.570796327),
    URDhRow(0.0, 0.0921, 0.0),
)

_UR5_DH: tuple[URDhRow, ...] = (
    URDhRow(0.0, 0.089159, 1.570796327),
    URDhRow(-0.425, 0.0, 0.0),
    URDhRow(-0.39225, 0.0, 0.0),
    URDhRow(0.0, 0.10915, 1.570796327),
    URDhRow(0.0, 0.09465, -1.570796327),
    URDhRow(0.0, 0.0823, 0.0),
)

_UR5E_DH: tuple[URDhRow, ...] = (
    URDhRow(0.0, 0.1625, 1.570796327),
    URDhRow(-0.425, 0.0, 0.0),
    URDhRow(-0.3922, 0.0, 0.0),
    URDhRow(0.0, 0.1333, 1.570796327),
    URDhRow(0.0, 0.0997, -1.570796327),
    URDhRow(0.0, 0.0996, 0.0),
)

_UR10_DH: tuple[URDhRow, ...] = (
    URDhRow(0.0, 0.1273, 1.570796327),
    URDhRow(-0.612, 0.0, 0.0),
    URDhRow(-0.5723, 0.0, 0.0),
    URDhRow(0.0, 0.163941, 1.570796327),
    URDhRow(0.0, 0.1157, -1.570796327),
    URDhRow(0.0, 0.0922, 0.0),
)

_UR10E_DH: tuple[URDhRow, ...] = (
    URDhRow(0.0, 0.1807, 1.570796327),
    URDhRow(-0.6127, 0.0, 0.0),
    URDhRow(-0.57155, 0.0, 0.0),
    URDhRow(0.0, 0.17415, 1.570796327),
    URDhRow(0.0, 0.11985, -1.570796327),
    URDhRow(0.0, 0.11655, 0.0),
)

_UR16E_DH: tuple[URDhRow, ...] = (
    URDhRow(0.0, 0.1807, 1.570796327),
    URDhRow(-0.4784, 0.0, 0.0),
    URDhRow(-0.36, 0.0, 0.0),
    URDhRow(0.0, 0.17415, 1.570796327),
    URDhRow(0.0, 0.11985, -1.570796327),
    URDhRow(0.0, 0.11655, 0.0),
)


UR_DH_TABLES_M: dict[str, tuple[URDhRow, ...]] = {
    "ur3": _UR3_DH,
    "ur3e": _UR3E_DH,
    "ur5": _UR5_DH,
    "ur5e": _UR5E_DH,
    "ur10": _UR10_DH,
    "ur10e": _UR10E_DH,
    "ur16e": _UR16E_DH,
}


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
    table = UR_DH_TABLES_M.get(model.lower())
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
    table = UR_DH_TABLES_M.get(model.lower())
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
