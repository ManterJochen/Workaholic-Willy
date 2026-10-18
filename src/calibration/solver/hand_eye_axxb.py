"""
AX = XB hand-eye calibration solver.

Numpy only. Rotation first, in closed form, on the Kronecker-product and SVD
formulation of Park and Martin (1994), adapted.

The solver is convention-agnostic: it runs the linear algebra and nothing
else. Which transform ``X`` comes out is decided entirely by the ``A`` and
``B`` the caller builds for the chosen calibration mode:

Eye-to-hand (fixed camera, marker on gripper)
---------------------------------------------
    A_i  = T_base_to_tool_{i+1} @ inv(T_base_to_tool_i)     (relative gripper motion)
    B_i  = T_cam_to_marker_{i+1} @ inv(T_cam_to_marker_i)   (relative marker motion)
    X    = T_cam_to_base                                      (solve target)

    Both sides are plain outputs of :meth:`HandEyeAXXB.relative_motions`.

Eye-in-hand (camera on gripper, fixed marker)
----------------------------------------------
    A_i  = inv(T_base_to_tool_{i+1}) @ T_base_to_tool_i     (local relative gripper motion)
    B_i  = T_cam_to_marker_{i+1} @ inv(T_cam_to_marker_i)   (relative marker motion)
    X    = T_cam_to_tool                                      (solve target)

    :class:`EyeInHandCalibrator` in ``eye_hand/eye_in_hand/calibrator.py``
    builds this pairing. Only ``B`` is a plain output of
    :meth:`relative_motions`; ``A`` is the local motion and is built by hand.

    .. warning::
       Swapping the two sides gives the exact inverse, not an error. Marker on
       the ``A`` side and gripper on the ``B`` side, with both index orders
       reversed, is the pair ``A' = inv(B)``, ``B' = inv(A)``, which is solved
       by ``X' = inv(X)``. The result is ``T_tool_to_cam`` under a
       ``T_cam_to_tool`` label and nothing raises. On the shipped sim geometry
       that inversion is roughly 290 mm out.

Reference
---------
Park, F. C. & Martin, B. J. (1994).
"Robot sensor calibration: solving AX = XB on the Euclidean group."
IEEE Trans. Robotics and Automation, 10(5):717-721.
"""

from __future__ import annotations

import numpy as np

from src.calibration.constants import CALIBRATION_LOG_DIR, HAND_EYE_AXXB_LOG_FILE
from src.calibration.exceptions import CalibrationDataError, CalibrationSolveError
from src.utility.log_cfg import create_logger

__all__ = ["HandEyeAXXB"]

logger = create_logger("HandEyeAXXB", HAND_EYE_AXXB_LOG_FILE, log_dir=CALIBRATION_LOG_DIR)

_MIN_PAIRS = 3


def _validate_homogeneous_matrix(value: object, *, name: str) -> np.ndarray:
    """Return ``value`` as a validated 4x4 rigid transform matrix."""
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise CalibrationDataError(f"{name} must be (4, 4), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise CalibrationDataError(f"{name} must contain only finite values")
    if not np.allclose(matrix[3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
        raise CalibrationDataError(
            f"{name} bottom row must be [0, 0, 0, 1], got {matrix[3, :].tolist()}"
        )
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise CalibrationDataError(f"{name} rotation block is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > 1e-5:
        raise CalibrationDataError(
            f"{name} rotation determinant must be +1, got {determinant:.6f}"
        )
    return matrix


def _invert_homogeneous(matrix: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 transform by transposing its rotation block.

    Valid only for a rigid transform, which is what
    :func:`_validate_homogeneous_matrix` has already established.
    """
    inverse = np.eye(4, dtype=np.float64)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def _project_to_SO3(M: np.ndarray) -> np.ndarray:
    """Project an approximate rotation matrix onto SO(3) via SVD.

    The null-space vector that :meth:`HandEyeAXXB._solve_rotation` feeds in
    carries a sign ambiguity (+/-vec(R_X)), so ``M`` may be the negative of the
    wanted rotation, which shows as ``det ~= -1``. Negating ``M`` first recovers
    ``R_X_true``; the second determinant check catches a reflection introduced
    by the SVD factors themselves.
    """
    if np.linalg.det(M) < 0:
        M = -M
    U, _S, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


class HandEyeAXXB:
    """Closed-form AX = XB hand-eye calibration solver.

    Solves for the unknown rigid transform X (4 x 4) from N pairs of relative
    transforms (A_i, B_i) that satisfy:

        A_i  X  ~=  X  B_i      for i = 1 ... N

    At least :data:`_MIN_PAIRS` (3) linearly independent pairs are required.
    The rotation is solved first and the translation on top of it, so a bad
    rotation carries into the translation.

    Usage::

        A_list = HandEyeAXXB.relative_motions(T_base_to_tool_list)
        B_list = HandEyeAXXB.relative_motions(T_cam_to_marker_list)
        solver = HandEyeAXXB()
        X, rmse = solver.solve(A_list, B_list)  # X = T_cam_to_base (eye-to-hand)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        A_mats: list[np.ndarray],
        B_mats: list[np.ndarray],
    ) -> tuple[np.ndarray, float]:
        """Solve A X = X B for all pairs; return X and residual RMSE.

        Parameters
        ----------
        A_mats:
            List of N (4, 4) homogeneous transforms (the "left" side).
        B_mats:
            List of N (4, 4) homogeneous transforms (the "right" side).
            Must have the same length as *A_mats*, and ``B_mats[i]`` must be
            the motion recorded at the same step as ``A_mats[i]``.

        Returns
        -------
        X:
            (4, 4) best-fit homogeneous transform. Which transform that is
            depends on the pairing the caller passed in; see the module
            docstring.
        rmse:
            Frobenius-norm residual RMSE averaged over all pairs:
            ``mean||A_i X - X B_i||_F``. The norm runs over the whole 4x4, so
            it mixes the rotation and translation residuals.

        Raises
        ------
        CalibrationDataError
            If fewer than 3 pairs are provided, the two lists differ in
            length, or an array is not a valid rigid homogeneous transform.
        CalibrationSolveError
            If the solve produces non-finite values.
        """
        A_mats = [
            _validate_homogeneous_matrix(A, name=f"A_mats[{index}]")
            for index, A in enumerate(A_mats)
        ]
        B_mats = [
            _validate_homogeneous_matrix(B, name=f"B_mats[{index}]")
            for index, B in enumerate(B_mats)
        ]

        if len(A_mats) != len(B_mats):
            raise CalibrationDataError(
                f"A_mats and B_mats must have the same length; "
                f"got {len(A_mats)} vs {len(B_mats)}"
            )
        if len(A_mats) < _MIN_PAIRS:
            raise CalibrationDataError(
                f"At least {_MIN_PAIRS} linearly independent pairs required; "
                f"got {len(A_mats)}"
            )

        R_X = self._solve_rotation(A_mats, B_mats)
        t_X = self._solve_translation(A_mats, B_mats, R_X)
        if not np.all(np.isfinite(R_X)) or not np.all(np.isfinite(t_X)):
            raise CalibrationSolveError("AX=XB solve produced non-finite values")

        X = np.eye(4, dtype=np.float64)
        X[:3, :3] = R_X
        X[:3, 3] = t_X

        rmse = self._residual_rmse(A_mats, B_mats, X)
        # The solver is convention-agnostic, so this line is the only record of what was
        # actually solved. An inverted A and B pairing reaches it as a plausible rmse
        # next to a translation of the wrong magnitude.
        logger.info(
            "Solved X from %d pairs: residual rmse=%.4f, |t_X|=%.2f mm.",
            len(A_mats),
            rmse,
            float(np.linalg.norm(t_X)),
        )
        return X, rmse

    @staticmethod
    def relative_motions(T_mats: list[np.ndarray]) -> list[np.ndarray]:
        """Compute consecutive relative transforms from absolute poses.

        Given [T_0, T_1, ..., T_{N-1}] returns
        [T_1 @ inv(T_0),  T_2 @ inv(T_1),  ...,  T_{N-1} @ inv(T_{N-2})]
        (length N-1). This is the global relative motion, which both sides of
        the eye-to-hand pairing want. Eye-in-hand needs the local motion on
        the ``A`` side and builds it without this method.

        Parameters
        ----------
        T_mats:
            List of N (4, 4) absolute homogeneous transforms, in the order
            they were recorded.

        Returns
        -------
        list[np.ndarray]
            N-1 relative transforms, each (4, 4).

        Raises
        ------
        CalibrationDataError
            If fewer than 2 poses are given, or one is not a valid rigid
            homogeneous transform.
        """
        if len(T_mats) < 2:
            raise CalibrationDataError("Need at least 2 poses to compute relative motions")
        mats = [
            _validate_homogeneous_matrix(T, name=f"T_mats[{index}]")
            for index, T in enumerate(T_mats)
        ]
        motions = [
            mats[i + 1] @ _invert_homogeneous(mats[i])
            for i in range(len(mats) - 1)
        ]
        logger.debug("Built %d relative motions from %d absolute poses.", len(motions), len(mats))
        return motions

    # ------------------------------------------------------------------
    # Internal: the rotation step
    # ------------------------------------------------------------------

    @staticmethod
    def _solve_rotation(
        A_mats: list[np.ndarray],
        B_mats: list[np.ndarray],
    ) -> np.ndarray:
        """Find R_X via the Kronecker-product null-space approach.

        The rotation blocks alone give R_A R_X = R_X R_B, and the column-major
        vec identity turns that into a homogeneous linear system:

            vec(AXB) = kron(B^T, A) vec(X)
            =>  (kron(I, R_A) - kron(R_B^T, I)) vec(R_X) = 0

        Stack all pairs into K (9N x 9), find the null-space vector via SVD
        (last right singular vector), reshape with Fortran ('F') order to
        match the column-major convention, then project to SO(3), which the
        null vector only approximates.
        """
        n = len(A_mats)
        K = np.zeros((9 * n, 9), dtype=np.float64)
        for i, (A, B) in enumerate(zip(A_mats, B_mats)):
            R_A = A[:3, :3]
            R_B = B[:3, :3]
            # In the column-major convention: kron(I, R_A) vec(X) = vec(R_A X)
            # and kron(R_B^T, I) vec(X) = vec(X R_B).
            K[9 * i : 9 * (i + 1)] = (
                np.kron(np.eye(3), R_A) - np.kron(R_B.T, np.eye(3))
            )

        _, _, Vt = np.linalg.svd(K)
        v = Vt[-1]  # last right singular vector: the null space, vec(R_X)
        R_approx = v.reshape(3, 3, order="F")
        return _project_to_SO3(R_approx)

    # ------------------------------------------------------------------
    # Internal: the translation step
    # ------------------------------------------------------------------

    @staticmethod
    def _solve_translation(
        A_mats: list[np.ndarray],
        B_mats: list[np.ndarray],
        R_X: np.ndarray,
    ) -> np.ndarray:
        """Find t_X via linear least squares, given the solved R_X.

        The translation rows of AX = XB give:

            R_A t_X + t_A = R_X t_B + t_X
            (R_A - I) t_X = R_X t_B - t_A

        Stack all N equations into 3N rows over three unknowns and solve in
        the least-squares sense.
        """
        n = len(A_mats)
        C = np.zeros((3 * n, 3), dtype=np.float64)
        d = np.zeros(3 * n, dtype=np.float64)
        for i, (A, B) in enumerate(zip(A_mats, B_mats)):
            R_A = A[:3, :3]
            t_A = A[:3, 3]
            t_B = B[:3, 3]
            C[3 * i : 3 * (i + 1)] = R_A - np.eye(3)
            d[3 * i : 3 * (i + 1)] = R_X @ t_B - t_A

        t_X, _, _, _ = np.linalg.lstsq(C, d, rcond=None)
        return t_X

    # ------------------------------------------------------------------
    # Internal: the residual
    # ------------------------------------------------------------------

    @staticmethod
    def _residual_rmse(
        A_mats: list[np.ndarray],
        B_mats: list[np.ndarray],
        X: np.ndarray,
    ) -> float:
        """Mean Frobenius-norm residual ||A_i X - X B_i||_F over all pairs."""
        errs = [
            np.linalg.norm(A @ X - X @ B, "fro")
            for A, B in zip(A_mats, B_mats)
        ]
        return float(np.mean(errs))
