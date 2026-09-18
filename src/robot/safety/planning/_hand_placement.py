"""Where a hand model sits on the flange, derived once from the declared tool frame.

Three frames:

* tool0, the flange: DH frame 6. wrist_3_link and URDF tool0 are that frame, on the Isaac cell and in the description
  Universal Robots publishes.
* M, every committed hand model (bundle arrays, sphere maps, mesh fits): x closing, y approach out of the mounting
  face, z = x cross y.
* G, a declared tool frame (``robot.gripper.tool_frame``): x closing, y binormal, z approach. Its axes in M are fixed,
  ``R_MG = R_x(-90)``.

A declaration gives ``R_TG``, the axes of G in tool0, and a model point goes onto the flange by
``p_tool0 = R_TM (p_model + plate along model y)`` with ``R_TM = R_TG @ R_MG^T``. The Isaac declaration of the sim is
``R_x(-90)`` itself, so ``R_TM`` is exactly the identity and the cells that work keep their bytes. The UR convention,
the identity quaternion, gives ``R_x(+90)``: approach along tool0 +Z, closing +X.

The result is snapped to the nearest signed permutation, exact 0.0 and 1.0 with no negative zero, and the angle it was
snapped by is kept. What refuses is typed, in this order: not orthonormal, a mirror, an approach more than a degree
from every flange axis, an axis approach other than +Y or +Z (-Z points into the wrist, and nobody measured +-X or -Y),
and a clocking about the approach that is not a quarter turn.

Standard library only, and no ``from __future__ import annotations``: the cuRobo interpreter loads this file by path,
and a dataclass with string annotations looks its module up in sys.modules, where a by-path load is not registered.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "ADMITTED_APPROACHES", "R_MG", "TOLERANCE_DEG", "HandPlacement", "PlacementRefused", "RefusalKind",
    "placement_quaternion_xyzw", "quaternion_xyzw_to_matrix",
]

Matrix = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]

#: The axes of G in M, as rows: closing (1,0,0), binormal (0,0,-1), approach (0,1,0) are its columns.
R_MG: Matrix = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))
#: The approaches a placement may declare: the one the committed models use (+Y) and the UR convention (+Z).
ADMITTED_APPROACHES = ("+Y", "+Z")
#: How far an axis may lean from a flange axis before it is not that axis: the tolerance hand.py already applies.
TOLERANCE_DEG = 1.0
#: How far a declared matrix may be from orthonormal. A quaternion is normalised first and never meets this.
_ORTHONORMAL = 1e-9
_IDENTITY: Matrix = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


class RefusalKind(str, Enum):
    NOT_PROPER = "not_proper"
    MIRROR = "mirror"
    OBLIQUE_APPROACH = "oblique_approach"
    APPROACH_NOT_ADMITTED = "approach_not_admitted"
    CLOCKING_NOT_QUARTER_TURN = "clocking_not_quarter_turn"


class PlacementRefused(ValueError):
    """A declared tool frame no committed hand model can be placed by."""

    def __init__(self, kind: RefusalKind, sentence: str) -> None:
        super().__init__(sentence)
        self.kind = kind


def _mul(a: Any, b: Any) -> Matrix:
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))  # type: ignore[return-value]


def _transpose(a: Any) -> Matrix:
    return tuple(tuple(a[j][i] for j in range(3)) for i in range(3))  # type: ignore[return-value]


def _det(a: Any) -> float:
    return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))


def _column(a: Any, index: int) -> tuple[float, float, float]:
    return (float(a[0][index]), float(a[1][index]), float(a[2][index]))


def quaternion_xyzw_to_matrix(q: Any) -> Matrix:
    """The rotation of a quaternion, normalised first, as rows. The same formula as ``geometry.quaternion``."""
    x, y, z, w = (float(v) for v in q)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise PlacementRefused(RefusalKind.NOT_PROPER, "the declared rotation is the zero quaternion, not a rotation")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
        (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
        (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)),
    )


#: The only magnitudes a quarter turn's quaternion components take.
_QUARTER_TURN_COMPONENTS = (0.0, 0.5, math.sqrt(0.5), 1.0)


def placement_quaternion_xyzw(approach: str, closing: str) -> tuple[float, float, float, float]:
    """The declared tool frame rotation that places a hand ``approach`` + ``closing``, as XYZW.

    The inverse of what :meth:`HandPlacement.from_quaternion_xyzw` derives, so a refusal that knows only ``+Z+X``
    can print a rotation a customer pastes into ``choose_ur_retract.py`` and ``matrix_gate.py``. ``w >= 0``, and
    every component is snapped to 0, 0.5, sqrt(0.5) or 1 with its sign and no negative zero: ``+Z+X`` is
    ``(0, 0, 0, 1)`` and ``+Y+X`` the sim's ``(-sqrt(0.5), 0, 0, sqrt(0.5))``, exactly.
    """
    words = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
    if approach not in words or closing not in words:
        raise PlacementRefused(RefusalKind.NOT_PROPER, f"{approach!r} and {closing!r} are not both flange axes")
    if approach not in ADMITTED_APPROACHES:
        raise PlacementRefused(
            RefusalKind.APPROACH_NOT_ADMITTED,
            f"a hand is placed along {' or '.join(ADMITTED_APPROACHES)}, and {approach} is not one of them",
        )
    if approach[1] == closing[1]:
        raise PlacementRefused(RefusalKind.NOT_PROPER, f"the closing {closing} lies along the approach {approach}")
    z = _unit("XYZ".index(approach[1]), 1.0 if approach[0] == "+" else -1.0)
    x = _unit("XYZ".index(closing[1]), 1.0 if closing[0] == "+" else -1.0)
    y = (z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0])
    m = ((x[0], y[0], z[0]), (x[1], y[1], z[1]), (x[2], y[2], z[2]))
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        q = ((m[2][1] - m[1][2]) * s, (m[0][2] - m[2][0]) * s, (m[1][0] - m[0][1]) * s, 0.25 / s)
    elif m[0][0] >= m[1][1] and m[0][0] >= m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
        q = (0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s)
    elif m[1][1] >= m[2][2]:
        s = 2.0 * math.sqrt(1.0 - m[0][0] + m[1][1] - m[2][2])
        q = ((m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 - m[0][0] - m[1][1] + m[2][2])
        q = ((m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s, (m[1][0] - m[0][1]) / s)
    if q[3] < 0.0 or (q[3] == 0.0 and next((c for c in q[:3] if c != 0.0), 0.0) < 0.0):
        q = (-q[0], -q[1], -q[2], -q[3])

    def snap(value: float) -> float:
        magnitude = min(_QUARTER_TURN_COMPONENTS, key=lambda candidate: abs(abs(value) - candidate))
        return (-magnitude if value < 0.0 else magnitude) + 0.0

    x_, y_, z_, w_ = (snap(c) for c in q)
    return (x_, y_, z_, w_)


def _unit(index: int, sign: float) -> tuple[float, float, float]:
    out = [0.0, 0.0, 0.0]
    out[index] = sign
    return (out[0], out[1], out[2])


def _name(index: int, sign: float) -> str:
    return f"{'+' if sign > 0 else '-'}{'XYZ'[index]}"


def _angle_deg(v: tuple[float, float, float], u: tuple[float, float, float]) -> float:
    norm = math.sqrt(sum(c * c for c in v))
    cosine = sum(a * b for a, b in zip(v, u)) / norm
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _nearest(v: tuple[float, float, float], exclude: int | None = None) -> tuple[int, float, float]:
    """(axis index, sign, degrees) of the flange axis nearest ``v``, skipping axis ``exclude``."""
    best = (0, 1.0, 360.0)
    for index in range(3):
        if index == exclude:
            continue
        for sign in (1.0, -1.0):
            angle = _angle_deg(v, _unit(index, sign))
            if angle < best[2]:
                best = (index, sign, angle)
    return best


def _clean(a: Any) -> Matrix:
    """Exact integers as floats, with no negative zero: the repr of -0.0 differs from 0.0 and would change a hash."""
    return tuple(tuple(float(round(v)) + 0.0 for v in row) for row in a)  # type: ignore[return-value]


@dataclass(frozen=True)
class HandPlacement:
    """Where the committed hand model sits on tool0: ``rotation`` is R_TM, as rows."""

    rotation: Matrix
    approach: str
    closing: str
    #: ``declared`` or ``undeclared``.
    source: str
    #: Degrees between the declared rotation and the quarter turn it was snapped to.
    residual_deg: float

    @classmethod
    def undeclared(cls) -> "HandPlacement":
        return cls(rotation=_IDENTITY, approach="+Y", closing="+X", source="undeclared", residual_deg=0.0)

    @classmethod
    def from_quaternion_xyzw(cls, q: Any, *, source: str = "declared") -> "HandPlacement":
        return cls._from_declared(quaternion_xyzw_to_matrix(q), source)

    @classmethod
    def from_matrix(cls, rows: Any, *, source: str = "declared") -> "HandPlacement":
        m = tuple(tuple(float(v) for v in row) for row in rows)
        product = _mul(_transpose(m), m)
        error = max(abs(product[i][j] - (1.0 if i == j else 0.0)) for i in range(3) for j in range(3))
        if error > _ORTHONORMAL:
            raise PlacementRefused(
                RefusalKind.NOT_PROPER,
                f"the declared tool frame matrix is not a rotation: its columns are {error:.3g} from orthonormal",
            )
        return cls._from_declared(m, source)

    @classmethod
    def _from_declared(cls, r_tg: Any, source: str) -> "HandPlacement":
        if _det(r_tg) < 0.0:
            raise PlacementRefused(
                RefusalKind.MIRROR,
                "the declared tool frame is a mirror (determinant -1): no physical hand is its own reflection, and a "
                "mirrored model belongs to the mesh writer, not to a placement",
            )
        approach_index, approach_sign, approach_angle = _nearest(_column(r_tg, 2))
        approach = _name(approach_index, approach_sign)
        if approach_angle > TOLERANCE_DEG:
            raise PlacementRefused(
                RefusalKind.OBLIQUE_APPROACH,
                f"the declared tool frame approaches {approach_angle:.0f} degrees from flange {approach}, the nearest "
                f"flange axis, and a hand is placed only along an axis (within {TOLERANCE_DEG:g} degree)",
            )
        if approach not in ADMITTED_APPROACHES:
            raise PlacementRefused(
                RefusalKind.APPROACH_NOT_ADMITTED,
                f"the declared tool frame approaches along flange {approach}; a hand is placed along "
                f"{' or '.join(ADMITTED_APPROACHES)}: -Z points into the wrist, and no other axis has been measured",
            )
        closing_index, closing_sign, closing_angle = _nearest(_column(r_tg, 0), exclude=approach_index)
        closing = _name(closing_index, closing_sign)
        if closing_angle > TOLERANCE_DEG:
            raise PlacementRefused(
                RefusalKind.CLOCKING_NOT_QUARTER_TURN,
                f"the declared tool frame closes {closing_angle:.0f} degrees from flange {closing}, the nearest quarter "
                f"turn about its approach {approach}; a hand is clocked only by quarter turns",
            )
        z = _unit(approach_index, approach_sign)
        x = _unit(closing_index, closing_sign)
        y = (z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0])
        snapped: Matrix = ((x[0], y[0], z[0]), (x[1], y[1], z[1]), (x[2], y[2], z[2]))
        trace = sum(_mul(_transpose(snapped), r_tg)[i][i] for i in range(3))
        residual = math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0))))
        return cls(rotation=_clean(_mul(snapped, _transpose(R_MG))), approach=approach, closing=closing,
                   source=source, residual_deg=residual)

    @property
    def is_identity(self) -> bool:
        return self.rotation == _IDENTITY

    @property
    def quaternion_xyzw(self) -> tuple[float, float, float, float]:
        """The declared tool frame rotation that places this hand, as :func:`placement_quaternion_xyzw` spells it."""
        return placement_quaternion_xyzw(self.approach, self.closing)

    @property
    def approach_in_tool0(self) -> tuple[float, float, float]:
        return _column(self.rotation, 1)

    @property
    def closing_in_tool0(self) -> tuple[float, float, float]:
        return _column(self.rotation, 0)

    def render(self) -> str:
        if self.source == "undeclared":
            return f"undeclared: the hand model's own axes, approach {self.approach}, closing {self.closing}"
        if self.is_identity:
            return f"approach {self.approach}, closing {self.closing}, declared on the hand models' own axes"
        return f"approach {self.approach}, closing {self.closing}, declared and not measured"

    def to_dict(self) -> dict[str, Any]:
        return {"rotation": [list(row) for row in self.rotation], "approach": self.approach, "closing": self.closing,
                "source": self.source, "residual_deg": self.residual_deg}
