"""XYZW quaternion utilities, the canonical rotation representation.

Convention:

* Order is XYZW.
* Magnitude is unit length.
* Canonical sign has ``w >= 0``. ``q`` and ``-q`` are the same rotation;
  :func:`canonicalise` enforces the sign.
* Composition ``r_total = q1 * q2`` rotates by ``q2`` first and by ``q1``
  second, the same order as homogeneous matrix multiplication.

Imports here stay limited to :mod:`numpy`, so the geometry package carries no
OpenCV, SciPy or vendor dependency.
"""

from __future__ import annotations

import numpy as np

from .exceptions import InvalidQuaternionError
from .exceptions import InvalidMatrixError
from .validation import normalise_quaternion, validate_quaternion_xyzw, validate_rotation_matrix

__all__ = [
    "IDENTITY_QUAT_XYZW",
    "angle_between",
    "canonicalise",
    "conjugate",
    "from_axis_angle",
    "from_euler",
    "from_rotation_matrix",
    "matrix_to_rotation_vector",
    "multiply",
    "rotate_vector",
    "rotation_vector_to_matrix",
    "slerp",
    "to_axis_angle",
    "to_euler",
    "to_rotation_matrix",
]


IDENTITY_QUAT_XYZW: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
IDENTITY_QUAT_XYZW.setflags(write=False)


def canonicalise(q: np.ndarray) -> np.ndarray:
    """Return a normalised, sign-canonical (``w >= 0``) copy of ``q``."""
    return normalise_quaternion(q, canonicalise=True)


def conjugate(q: np.ndarray) -> np.ndarray:
    """Return the conjugate ``[-x, -y, -z, w]`` of a unit XYZW quaternion."""
    arr = validate_quaternion_xyzw(q)
    out = np.empty(4, dtype=np.float64)
    out[0] = -arr[0]
    out[1] = -arr[1]
    out[2] = -arr[2]
    out[3] = arr[3]
    return out


def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product ``q1 * q2`` in scalar-last XYZW order.

    Applied to a vector, ``q2`` rotates first and ``q1`` second. The result is
    canonical.
    """
    a = validate_quaternion_xyzw(q1)
    b = validate_quaternion_xyzw(q2)
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    out = np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )
    return canonicalise(out)


def rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a (3,) vector by a unit XYZW quaternion."""
    qv = validate_quaternion_xyzw(q)
    arr = np.array(v, dtype=np.float64, copy=True)
    if arr.shape != (3,):
        raise InvalidQuaternionError(
            f"rotate_vector: v must have shape (3,), got {arr.shape}"
        )
    return to_rotation_matrix(qv) @ arr


def from_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a canonical XYZW quaternion."""
    try:
        arr = validate_rotation_matrix(R, name="from_rotation_matrix")
    except InvalidMatrixError as exc:
        raise InvalidQuaternionError(str(exc)) from exc

    # Take the branch with the largest divisor, so no component is recovered by
    # dividing through a value near zero.
    trace = float(np.trace(arr))
    if trace > 0.0:
        scale = float(np.sqrt(trace + 1.0) * 2.0)
        qw = 0.25 * scale
        qx = (arr[2, 1] - arr[1, 2]) / scale
        qy = (arr[0, 2] - arr[2, 0]) / scale
        qz = (arr[1, 0] - arr[0, 1]) / scale
    elif arr[0, 0] > arr[1, 1] and arr[0, 0] > arr[2, 2]:
        scale = float(np.sqrt(1.0 + arr[0, 0] - arr[1, 1] - arr[2, 2]) * 2.0)
        qw = (arr[2, 1] - arr[1, 2]) / scale
        qx = 0.25 * scale
        qy = (arr[0, 1] + arr[1, 0]) / scale
        qz = (arr[0, 2] + arr[2, 0]) / scale
    elif arr[1, 1] > arr[2, 2]:
        scale = float(np.sqrt(1.0 + arr[1, 1] - arr[0, 0] - arr[2, 2]) * 2.0)
        qw = (arr[0, 2] - arr[2, 0]) / scale
        qx = (arr[0, 1] + arr[1, 0]) / scale
        qy = 0.25 * scale
        qz = (arr[1, 2] + arr[2, 1]) / scale
    else:
        scale = float(np.sqrt(1.0 + arr[2, 2] - arr[0, 0] - arr[1, 1]) * 2.0)
        qw = (arr[1, 0] - arr[0, 1]) / scale
        qx = (arr[0, 2] + arr[2, 0]) / scale
        qy = (arr[1, 2] + arr[2, 1]) / scale
        qz = 0.25 * scale

    return canonicalise(np.array([qx, qy, qz, qw], dtype=np.float64))


def to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a unit XYZW quaternion to a 3x3 rotation matrix."""
    qv = validate_quaternion_xyzw(q)
    x, y, z, w = qv
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def rotation_vector_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """Convert a Rodrigues axis-angle rotation vector to a 3x3 matrix."""
    return to_rotation_matrix(from_axis_angle(rvec))


def matrix_to_rotation_vector(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a canonical axis-angle vector."""
    return to_axis_angle(from_rotation_matrix(R))


def from_axis_angle(rvec: np.ndarray) -> np.ndarray:
    """Convert an axis-angle vector in radians to a canonical XYZW quaternion.

    The magnitude of ``rvec`` is the angle. The zero vector maps to identity.
    """
    arr = np.asarray(rvec, dtype=np.float64).reshape(-1)
    if arr.shape != (3,):
        raise InvalidQuaternionError(
            f"from_axis_angle: expected shape (3,), got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise InvalidQuaternionError("from_axis_angle: non-finite entries")
    angle = float(np.linalg.norm(arr))
    if angle == 0.0:
        return IDENTITY_QUAT_XYZW.copy()
    axis = arr / angle
    half_angle = 0.5 * angle
    sin_half = float(np.sin(half_angle))
    return canonicalise(
        np.array(
            [axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, np.cos(half_angle)],
            dtype=np.float64,
        )
    )


def to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Convert a unit XYZW quaternion to an axis-angle vector (rad)."""
    qv = validate_quaternion_xyzw(q)
    if qv[3] < 0.0:
        qv = -qv
    vector = qv[:3]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * float(np.arctan2(vector_norm, qv[3]))
    return vector / vector_norm * angle


def from_euler(angles_rad: np.ndarray, *, order: str = "xyz") -> np.ndarray:
    """Convert XYZ Euler angles in radians to a canonical XYZW quaternion.

    The x angle applies first, then y, then z. Only ``order='xyz'`` is
    supported; any other value raises ``InvalidQuaternionError``.
    """
    arr = np.asarray(angles_rad, dtype=np.float64).reshape(-1)
    if arr.shape != (3,):
        raise InvalidQuaternionError(
            f"from_euler: expected shape (3,), got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise InvalidQuaternionError("from_euler: non-finite entries")
    if order.lower() != "xyz":
        raise InvalidQuaternionError("from_euler currently supports only order='xyz'")
    qx = from_axis_angle(np.array([arr[0], 0.0, 0.0], dtype=np.float64))
    qy = from_axis_angle(np.array([0.0, arr[1], 0.0], dtype=np.float64))
    qz = from_axis_angle(np.array([0.0, 0.0, arr[2]], dtype=np.float64))
    return multiply(qz, multiply(qy, qx))


def to_euler(q: np.ndarray, *, order: str = "xyz") -> np.ndarray:
    """Convert a unit XYZW quaternion to XYZ Euler angles in radians.

    Only ``order='xyz'`` is supported. At gimbal lock, pitch near plus or minus
    90 degrees and ``sy < 1e-12``, the decomposition is ambiguous: ``z``
    collapses to 0 and the rotation folds into ``x``. The angles still
    reconstruct the same rotation through :func:`from_euler`, but the x and z
    split is not recoverable at the singularity.
    """
    qv = validate_quaternion_xyzw(q)
    if order.lower() != "xyz":
        raise InvalidQuaternionError("to_euler currently supports only order='xyz'")
    R = to_rotation_matrix(qv)
    sy = float(np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
    singular = sy < 1e-12
    if not singular:
        x = float(np.arctan2(R[2, 1], R[2, 2]))
        y = float(np.arctan2(-R[2, 0], sy))
        z = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        x = float(np.arctan2(-R[1, 2], R[1, 1]))
        y = float(np.arctan2(-R[2, 0], sy))
        z = 0.0
    return np.array([x, y, z], dtype=np.float64)


def angle_between(q1: np.ndarray, q2: np.ndarray) -> float:
    """Return the geodesic angle in radians, from 0 to pi, between two rotations."""
    a = validate_quaternion_xyzw(q1)
    b = validate_quaternion_xyzw(q2)
    # q and -q are the same rotation, so the sign is folded out of the product.
    dot = abs(float(np.dot(a, b)))
    dot = min(1.0, max(-1.0, dot))
    return 2.0 * float(np.arccos(dot))


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation from ``q0`` to ``q1``, ``t`` in ``[0, 1]``.

    The result walks the shorter of the two arcs between the rotations at a constant
    angular rate, so ``angle_between(q0, slerp(q0, q1, t))`` is ``t`` times the total
    angle. That is the property a sampled Cartesian path needs: a caller that bounds the
    angle per step knows what the bound buys, which linear interpolation of the four
    components followed by a normalise does not give, since that walks the chord and
    speeds up in the middle.

    ``q`` and ``-q`` are the same rotation, so the sign of ``q1`` is flipped when the two
    point away from each other. Without that, a half turn of the arc between a quaternion
    and its own negation would be a half turn of the arm, for a move that does not rotate
    at all.

    Nearly parallel rotations fall back to the normalised straight line between them,
    because ``sin`` of the half angle goes to zero and the quotient stops being
    computable. At the angle where it switches, the two agree to well past float64
    precision.
    """
    a = validate_quaternion_xyzw(q0)
    b = validate_quaternion_xyzw(q1)
    fraction = float(t)
    if not np.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
        # A ValueError and not an InvalidQuaternionError: the two quaternions may be
        # perfectly good and it is the caller's fraction that is not a point on the arc
        # between them.
        raise ValueError(f"slerp takes a fraction of the arc in [0, 1], got {t!r}")
    dot = float(np.dot(a, b))
    if dot < 0.0:  # the same rotation, spelled the other way round: take the short arc
        b = -b
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 1.0 - 1e-12:
        out = a + fraction * (b - a)
        return normalise_quaternion(out, canonicalise=True)
    half_angle = float(np.arccos(dot))
    sin_half = float(np.sin(half_angle))
    weight_a = float(np.sin((1.0 - fraction) * half_angle)) / sin_half
    weight_b = float(np.sin(fraction * half_angle)) / sin_half
    return normalise_quaternion(weight_a * a + weight_b * b, canonicalise=True)
