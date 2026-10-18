"""Off-box contract for the industrial suction runner's pure helper (the pick itself is on-box, needs Isaac).

Locks the ``_orthonormalize`` fix: composing/stacking near-orthonormal axes (a Gram-Schmidt closing axis + a
cross-product binormal, or a product of measured wrist/object rotations) drifts the determinant ~1e-6 off 1.0
-- past ``ROTATION_DET_TOL`` -- which made ``from_rotation_matrix`` abort the on-box carry mid-lift
(``InvalidQuaternionError: determinant must be +1, got 1.0000010``). Snapping to the nearest proper rotation
absorbs the drift so the strict conversion always succeeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.geometry.exceptions import InvalidQuaternionError
from src.geometry.quaternion import from_rotation_matrix
from src.willy_sim.run_industrial_suction_pick import _orthonormalize


def _rot_z(theta: float) -> np.ndarray:
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_orthonormalize_absorbs_determinant_drift_that_would_abort_conversion() -> None:
    # A near-rotation scaled by (1 + 2e-6): det + orthonormality both drift past the 1e-6 validation tolerance
    # -- the exact InvalidQuaternionError that aborted the on-box carry ("determinant must be +1, got 1.0000010").
    drifted = _rot_z(0.7) * (1.0 + 2e-6)
    assert abs(float(np.linalg.det(drifted)) - 1.0) > 1e-6
    with pytest.raises(InvalidQuaternionError):
        from_rotation_matrix(drifted)  # documents the failure the fix prevents

    fixed = _orthonormalize(drifted)
    assert float(np.linalg.det(fixed)) == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(fixed.T @ fixed, np.eye(3), atol=1e-9)
    # the whole point: the strict conversion now succeeds, on the SAME rotation to within the drift.
    quat = from_rotation_matrix(fixed)
    assert np.asarray(quat).shape == (4,)
    np.testing.assert_allclose(fixed, _rot_z(0.7), atol=1e-5)


def test_orthonormalize_returns_a_proper_rotation_for_an_improper_input() -> None:
    # A reflection (det = -1) is snapped to the nearest PROPER rotation (det = +1) -- the sign-fix branch.
    fixed = _orthonormalize(np.diag([1.0, 1.0, -1.0]))
    assert float(np.linalg.det(fixed)) == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(fixed.T @ fixed, np.eye(3), atol=1e-9)


def test_orthonormalize_is_identity_on_a_clean_rotation() -> None:
    clean = _rot_z(1.1)
    np.testing.assert_allclose(_orthonormalize(clean), clean, atol=1e-12)
