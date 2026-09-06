"""Vendor-neutral singularity analysis helpers.

The implementation is robot-agnostic by construction: it uses only `fk` and `ik` from
the :class:`RobotArm` Protocol surface. It estimates a numerical Jacobian by central
differences around a joint vector, then classifies risk from the singular values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry import Pose, conjugate, multiply, to_axis_angle
from src.robot.core import JointPositions, RobotArm, RobotSingularityRisk

__all__ = [
    "SingularityGuard",
    "SingularityReport",
    "SingularityThresholds",
    "analyze_joint_singularity",
    "analyze_pose_singularity",
    "assert_pose_not_singular",
]


@dataclass(frozen=True, slots=True)
class SingularityThresholds:
    """Thresholds that classify singularity risk.

    Parameters
    ----------
    min_singular_value : float
        Smallest acceptable singular value of the geometric Jacobian.
    max_condition_number : float
        Largest acceptable Jacobian condition number.
    rank_tol : float
        A singular value at or below this counts as rank-deficient.
    """

    min_singular_value: float = 0.005
    max_condition_number: float = 250.0
    rank_tol: float = 1e-6


DEFAULT_THRESHOLDS = SingularityThresholds()


@dataclass(frozen=True, slots=True)
class SingularityReport:
    """The structured result of a singularity analysis for one joint state."""

    joints: JointPositions
    jacobian: np.ndarray
    singular_values: np.ndarray
    min_singular_value: float
    condition_number: float
    rank: int
    expected_rank: int
    is_near_singularity: bool
    reasons: tuple[str, ...]


def _readonly(a: np.ndarray) -> np.ndarray:
    out = np.asarray(a, dtype=np.float64).copy()
    out.setflags(write=False)
    return out


def _orientation_delta_rad(q_plus: np.ndarray, q_minus: np.ndarray) -> np.ndarray:
    # The rotation that takes q_minus to q_plus.
    q_rel = multiply(q_plus, conjugate(q_minus))
    return to_axis_angle(q_rel)


def _pose_twist_delta(p_plus: Pose, p_minus: Pose, delta_rad: float) -> np.ndarray:
    dpos_m = (p_plus.position_mm - p_minus.position_mm) / (2.0 * delta_rad * 1000.0)
    drot = _orientation_delta_rad(p_plus.quaternion_xyzw, p_minus.quaternion_xyzw) / (2.0 * delta_rad)
    return np.concatenate([dpos_m, drot]).astype(np.float64)


def _estimate_geometric_jacobian(
    arm: RobotArm,
    joints: JointPositions,
    *,
    delta_rad: float,
) -> np.ndarray:
    q = joints.values
    n = joints.dof
    J = np.zeros((6, n), dtype=np.float64)

    for i in range(n):
        qp = q.copy()
        qm = q.copy()
        qp[i] += delta_rad
        qm[i] -= delta_rad
        pose_p = arm.fk(JointPositions(qp))
        pose_m = arm.fk(JointPositions(qm))
        J[:, i] = _pose_twist_delta(pose_p, pose_m, delta_rad)

    return J


def analyze_joint_singularity(
    arm: RobotArm,
    joints: JointPositions,
    *,
    thresholds: SingularityThresholds = DEFAULT_THRESHOLDS,
    delta_rad: float = 1e-4,
) -> SingularityReport:
    """Analyse the singularity risk at one joint configuration."""
    if delta_rad <= 0.0:
        raise ValueError(f"delta_rad must be > 0, got {delta_rad}")

    J = _estimate_geometric_jacobian(arm, joints, delta_rad=delta_rad)
    _u, s, _vh = np.linalg.svd(J, full_matrices=False)

    expected_rank = int(min(joints.dof, 6))
    rank = int(np.sum(s > thresholds.rank_tol))
    min_sigma = float(s[-1]) if s.size else 0.0
    max_sigma = float(s[0]) if s.size else 0.0
    cond = float(np.inf) if min_sigma <= thresholds.rank_tol else float(max_sigma / min_sigma)

    reasons: list[str] = []
    if rank < expected_rank:
        reasons.append(f"rank_deficient({rank}<{expected_rank})")
    if min_sigma < thresholds.min_singular_value:
        reasons.append(
            f"min_sigma_below_threshold({min_sigma:.6g}<{thresholds.min_singular_value:.6g})"
        )
    if cond > thresholds.max_condition_number:
        reasons.append(
            f"condition_number_above_threshold({cond:.3f}>{thresholds.max_condition_number:.3f})"
        )

    return SingularityReport(
        joints=joints,
        jacobian=_readonly(J),
        singular_values=_readonly(s),
        min_singular_value=min_sigma,
        condition_number=cond,
        rank=rank,
        expected_rank=expected_rank,
        is_near_singularity=bool(reasons),
        reasons=tuple(reasons),
    )


def analyze_pose_singularity(
    arm: RobotArm,
    pose: Pose,
    *,
    seed: JointPositions | None = None,
    thresholds: SingularityThresholds = DEFAULT_THRESHOLDS,
    delta_rad: float = 1e-4,
) -> SingularityReport:
    """Solve IK for ``pose`` and analyse the resulting joint state."""
    joints = arm.ik(pose, seed=seed)
    return analyze_joint_singularity(
        arm,
        joints,
        thresholds=thresholds,
        delta_rad=delta_rad,
    )


def assert_pose_not_singular(
    arm: RobotArm,
    pose: Pose,
    *,
    seed: JointPositions | None = None,
    thresholds: SingularityThresholds = DEFAULT_THRESHOLDS,
    delta_rad: float = 1e-4,
) -> JointPositions:
    """Raise :class:`RobotSingularityRisk` where ``pose`` is too close to a singularity."""
    report = analyze_pose_singularity(
        arm,
        pose,
        seed=seed,
        thresholds=thresholds,
        delta_rad=delta_rad,
    )
    if report.is_near_singularity:
        joined = "; ".join(report.reasons)
        raise RobotSingularityRisk(
            "Target pose rejected due to singularity risk: "
            f"{joined}. cond={report.condition_number:.3f}, "
            f"min_sigma={report.min_singular_value:.6g}"
        )
    return report.joints


class SingularityGuard:
    """A reusable guard object for singularity checks at pipeline level."""

    def __init__(
        self,
        arm: RobotArm,
        *,
        thresholds: SingularityThresholds = DEFAULT_THRESHOLDS,
        delta_rad: float = 1e-4,
    ) -> None:
        self.arm = arm
        self.thresholds = thresholds
        self.delta_rad = float(delta_rad)

    def analyze_pose(self, pose: Pose, *, seed: JointPositions | None = None) -> SingularityReport:
        return analyze_pose_singularity(
            self.arm,
            pose,
            seed=seed,
            thresholds=self.thresholds,
            delta_rad=self.delta_rad,
        )

    def require_safe_pose(self, pose: Pose, *, seed: JointPositions | None = None) -> JointPositions:
        return assert_pose_not_singular(
            self.arm,
            pose,
            seed=seed,
            thresholds=self.thresholds,
            delta_rad=self.delta_rad,
        )
