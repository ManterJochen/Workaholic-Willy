"""Gap L0.8 — pin the from_euler/to_euler xyz-only + gimbal-lock contract.

from_euler/to_euler support only order='xyz' (else raise), and to_euler collapses the third angle to
z=0 at gimbal lock (pitch ~= +/-90 deg). The returned angles still reconstruct the same rotation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.geometry.exceptions import InvalidQuaternionError
from src.geometry.quaternion import from_euler, to_euler


def _same_rotation(q1: np.ndarray, q2: np.ndarray) -> bool:
    # Unit quaternions q and -q are the same rotation -> compare |dot|.
    return abs(float(np.dot(np.asarray(q1), np.asarray(q2)))) > 1.0 - 1e-6


def test_from_euler_rejects_non_xyz_order() -> None:
    with pytest.raises(InvalidQuaternionError):
        from_euler(np.array([0.1, 0.2, 0.3]), order="zyx")


def test_to_euler_rejects_non_xyz_order() -> None:
    q = from_euler(np.array([0.1, 0.2, 0.3]))
    with pytest.raises(InvalidQuaternionError):
        to_euler(q, order="zyx")


def test_round_trip_non_singular() -> None:
    angles = np.array([0.3, 0.4, -0.5])
    assert np.allclose(to_euler(from_euler(angles)), angles, atol=1e-9)


def test_gimbal_lock_collapses_z_and_still_reconstructs() -> None:
    angles = np.array([0.3, math.pi / 2.0, 0.5])  # pitch = +90 deg -> singular
    q = from_euler(angles)
    e = to_euler(q)
    assert abs(e[2]) < 1e-9  # z collapsed to 0 at the singularity
    assert abs(e[1] - math.pi / 2.0) < 1e-6  # pitch still recovered
    assert _same_rotation(q, from_euler(e))  # angles still reconstruct the SAME rotation
