"""4x4 homogeneous matrix helpers.

They exist for boundary code that speaks plain ndarrays, such as eye-to-hand
calibration and OpenCV, and for tight inner loops where one matrix multiply is
cheaper than allocating a new :class:`Transform`. See also
:mod:`src.geometry.conversions`.

It also holds the flange-to-TCP comparison (:func:`compare_tool_frames`), which
a planned arm's tool-frame check and the rig calibration share; it moved here
from ``drivers/ur/tool_frame.py`` in P0e because nothing in it is UR.

Everywhere else use :class:`~src.geometry.transform.Transform`, which carries
explicit frames and is type-checked at every step. Nothing here carries a
frame, so frame agreement is the caller's obligation.
"""

from __future__ import annotations

import math

import numpy as np

from .quaternion import (
    from_rotation_matrix,
    to_rotation_matrix,
)
from .validation import (
    validate_homogeneous_matrix,
    validate_position_mm,
    validate_quaternion_xyzw,
    validate_rotation_matrix,
)

__all__ = [
    "IDENTITY_MATRIX",
    "compare_tool_frames",
    "compose_homogeneous",
    "invert_homogeneous",
    "make_homogeneous",
    "matrix_to_position_quaternion",
    "position_quaternion_to_matrix",
]


IDENTITY_MATRIX: np.ndarray = np.eye(4, dtype=np.float64)
IDENTITY_MATRIX.setflags(write=False)


def make_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 transform from a 3x3 rotation and a (3,) translation in mm."""
    R = validate_rotation_matrix(R, name="R")
    t = validate_position_mm(t, name="t")
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R
    out[:3, 3] = t
    return out


def matrix_to_position_quaternion(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a validated 4x4 transform into ``(translation_mm, quat_xyzw)``."""
    arr = validate_homogeneous_matrix(T)
    t = arr[:3, 3].astype(np.float64, copy=True)
    q = from_rotation_matrix(arr[:3, :3])
    return t, q


def position_quaternion_to_matrix(t: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 transform from a (3,) translation in mm and an XYZW quaternion."""
    t_v = validate_position_mm(t, name="translation_mm")
    q_v = validate_quaternion_xyzw(q)
    R = to_rotation_matrix(q_v)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R
    out[:3, 3] = t_v
    return out


def invert_homogeneous(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid transform analytically, without a linear solver."""
    arr = validate_homogeneous_matrix(T)
    R = arr[:3, :3]
    t = arr[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    # Validation guarantees an orthonormal rotation block, so the transpose is
    # its inverse and no general inversion is needed.
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def compose_homogeneous(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return ``A @ B`` after validating both as 4x4 rigid transforms.

    The frames are unchecked here: the caller answers for ``A`` ending in the
    frame ``B`` starts from.
    """
    a = validate_homogeneous_matrix(A, name="A")
    b = validate_homogeneous_matrix(B, name="B")
    return a @ b


def compare_tool_frames(observed: np.ndarray, declared: np.ndarray) -> tuple[float, float]:
    """``(translation_mm, rotation_deg)`` between two flange-to-TCP transforms.

    Both numbers are reported because they fail differently: a wrong translation is a
    crash on the first run, and a wrong rotation is a cell that logs successes while
    closing along the wrong object axis.
    """
    d_t = float(np.linalg.norm(np.asarray(observed)[:3, 3] - np.asarray(declared)[:3, 3]))
    r = np.asarray(observed)[:3, :3] @ np.asarray(declared)[:3, :3].T
    cos = (float(np.trace(r)) - 1.0) / 2.0
    d_r = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
    return d_t, d_r
