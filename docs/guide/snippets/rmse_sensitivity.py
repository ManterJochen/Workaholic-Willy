"""Why ``rmse_mm`` is not translational millimetres: a reproducible measurement.

Companion to ``docs/calibration-setup.md`` section 6 and ``docs/guide/03-calibration.md``
section 3.1, and the instrument behind the ratio quoted in ``src/calibration/quality.py``.
It builds an exact synthetic eye-to-hand problem with no noise, so the true solution has a
residual of about zero. It then perturbs the known ``X`` by exactly 1 mm of translation and
by exactly 1 degree of rotation, one axis at a time, and reports the residual each
perturbation produces.

The point: ``HandEyeAXXB._residual_rmse`` is the mean Frobenius norm
``mean ||A_i X - X B_i||_F``. That norm mixes the dimensionless rotation block with
millimetre translation terms and scales with ``|t_X|``, so the number it prints is a
self-consistency score, not a distance.

Run it from the repository root, off-box, with no GPU and no Isaac:

    python docs/guide/snippets/rmse_sensitivity.py

Deterministic: fixed seed, fixed camera pose, fixed pose set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root on sys.path

from src.calibration.solver.hand_eye_axxb import HandEyeAXXB  # noqa: E402

SEED = 0
N_POSES = 8
# Camera 450 mm out, 1000 mm up, looking down: |t_X| = 1096.59 mm, the same order of
# magnitude as the sim overhead camera and a realistic fixed-camera cell.
CAM_T_BASE_MM = np.array([450.0, 0.0, 1000.0])


def _rot(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix about a unit axis."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle_rad) * K + (1.0 - np.cos(angle_rad)) * (K @ K)


def _homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def build_problem() -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Return (A_mats, B_mats, X_true) for an exact, noise-free ETH problem."""
    rng = np.random.default_rng(SEED)

    # X = base_T_cam  (what EyeToHandCalibrator returns, frame-tagged CAMERA to BASE).
    # Optical axis pointing down at the table, plus a small yaw so the rotation is generic.
    R_X = _rot(np.array([1.0, 0.0, 0.0]), np.pi) @ _rot(np.array([0.0, 0.0, 1.0]), 0.3)
    X_true = _homogeneous(R_X, CAM_T_BASE_MM)

    # tool_T_marker: an arbitrary rigid offset. It cancels in AX=XB, which this script also
    # demonstrates: changing it does not move any number below.
    T_tool_marker = _homogeneous(_rot(np.array([0.0, 1.0, 0.0]), 0.2),
                                 np.array([12.0, -8.0, 65.0]))

    base_T_tool: list[np.ndarray] = []
    for _ in range(N_POSES):
        axis = rng.normal(size=3)
        angle = rng.uniform(-0.6, 0.6)          # generous, multi-axis tilts
        t = np.array([450.0, 0.0, 400.0]) + rng.uniform(-80.0, 80.0, size=3)
        base_T_tool.append(_homogeneous(_rot(axis, angle), t))

    # cam_T_marker_i = inv(X) @ base_T_tool_i @ tool_T_marker   (exact, no noise)
    X_inv = np.linalg.inv(X_true)
    cam_T_marker = [X_inv @ M @ T_tool_marker for M in base_T_tool]

    A = HandEyeAXXB.relative_motions(base_T_tool)
    B = HandEyeAXXB.relative_motions(cam_T_marker)
    return A, B, X_true


def main() -> None:
    A, B, X = build_problem()
    residual = HandEyeAXXB._residual_rmse  # noqa: SLF001 - the whole point is to measure it

    print(f"|t_X|             {np.linalg.norm(X[:3, 3]):.4f} mm")
    print(f"exact rmse        {residual(A, B, X):.4e}")

    print("\n+1.0 mm translation error, one axis at a time:")
    t_scores = []
    for i, name in enumerate("xyz"):
        X_bad = X.copy()
        X_bad[i, 3] += 1.0
        score = residual(A, B, X_bad)
        t_scores.append(score)
        print(f"  t {name} +1.0 mm    rmse {score:.6f}")

    print("\n+1.0 deg rotation error, one axis at a time:")
    r_scores = []
    for i, name in enumerate("xyz"):
        axis = np.zeros(3)
        axis[i] = 1.0
        X_bad = X.copy()
        X_bad[:3, :3] = _rot(axis, np.deg2rad(1.0)) @ X[:3, :3]
        score = residual(A, B, X_bad)
        r_scores.append(score)
        print(f"  R {name} +1.0 deg   rmse {score:.6f}")

    ratios = [r / t for r in r_scores for t in t_scores]
    print(f"\nrotation-vs-translation penalty ratio: {min(ratios):.1f}x .. {max(ratios):.1f}x")
    print("=> the residual is dominated by rotation consistency; it is not a distance.")


if __name__ == "__main__":
    main()
