"""An arm-backed, singularity-aware IK service for the grasp calculator.

The grasp calculator's ``IKService`` seam (``src/robot/grasping/planning/reachability.py``) lets the
execution layer veto grasp candidates that are not executable. This implementation backs that seam
with the real arm kinematics and the vendor-neutral singularity analysis:

* ``arm.ik(pose)`` resolves the grasp, and a failure rejects the candidate as unreachable;
* ``analyze_joint_singularity(arm, joints)`` in the safety layer takes an SVD of the
  finite-difference FK Jacobian and flags a near-singular configuration, which is rejected as well.
  A near-singular top-down close, such as the UR5e base-Y closing axis, solves IK but the controller
  executes it tens of mm off the target, so it is unreachable in practice. Dropping it leaves the
  calculator with the well-conditioned base-X grasp.

The Jacobian ``condition_number`` and ``min_singular_value`` are published as ``IKQualityMetrics``,
so the feasibility scorer can demote a marginal pose rather than only drop it, where that path is
enabled.

Vendor-neutral: it needs only ``arm.fk`` and ``arm.ik``, so any :class:`RobotArm` with native FK
works. It lives in willy_sim, the composition layer, so the grasping layer sees only the injected
Protocol.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from src.robot.grasping.planning.reachability import IKQualityMetrics, IKResult


class ArmBackedIKService:
    """``IKService`` that rejects unreachable and near-singular grasp poses via the arm kinematics."""

    def __init__(
        self,
        arm: Any,
        *,
        thresholds: Any | None = None,
        joint_limits_config: Any | None = None,
        debug: bool = False,
    ) -> None:
        self._arm = arm
        self._thresholds = thresholds  # safety.SingularityThresholds, or None for the safe default
        # JointLimitSafetyConfig, or None. When set, query() fills
        # IKQualityMetrics.joint_margin_deg, the degrees from the closest hard joint limit, so the
        # feasibility scorer and the success-probability feature see a value. None leaves it None.
        self._joint_limits_config = joint_limits_config
        self._debug = bool(debug)      # print per-query cond, min_sv and reachability

    def query(self, pose: Any) -> IKResult:
        # Local imports keep this module free of any geometry or safety import at load time.
        from src.geometry import Frame, Pose
        from src.geometry.quaternion import from_rotation_matrix
        from src.robot.safety.singularity import analyze_joint_singularity

        geom = Pose(
            position_mm=np.asarray(pose.position_mm, dtype=np.float64),
            quaternion_xyzw=from_rotation_matrix(np.asarray(pose.rotation_matrix, dtype=np.float64)),
            frame=Frame.BASE,
            label="grasp_ik_query",
        )
        try:
            joints = self._arm.ik(geom)
        except Exception as exc:  # noqa: BLE001 (any IK failure means the pose is unreachable)
            if self._debug:
                print(f"[ik-probe] pos={np.round(geom.position_mm, 1)} ik_failed:{type(exc).__name__}", flush=True)
            return IKResult(reachable=False, reason=f"ik_failed:{type(exc).__name__}")

        try:
            kwargs = {"thresholds": self._thresholds} if self._thresholds is not None else {}
            report = analyze_joint_singularity(self._arm, joints, **kwargs)
        except Exception as exc:  # noqa: BLE001 (IK solved but the probe failed, so do not over-reject)
            if self._debug:
                print(f"[ik-probe] pos={np.round(geom.position_mm, 1)} probe_unavailable:{type(exc).__name__}", flush=True)
            return IKResult(reachable=True, reason="singularity_probe_unavailable")

        quality = IKQualityMetrics(
            condition_number=_finite_or_none(report.condition_number),
            min_singular_value=_finite_or_none(report.min_singular_value),
            joint_margin_deg=self._joint_margin_deg(joints),  # real margin (None if no limits)
        )
        if self._debug:
            print(f"[ik-probe] pos={np.round(geom.position_mm, 1)} cond={report.condition_number:.1f} "
                  f"min_sv={report.min_singular_value:.4f} joint_margin_deg={quality.joint_margin_deg} "
                  f"near_singular={report.is_near_singularity}", flush=True)
        if report.is_near_singularity:
            return IKResult(reachable=False, reason="near_singular", quality=quality)
        return IKResult(reachable=True, quality=quality)

    def _joint_margin_deg(self, joints: Any) -> Optional[float]:
        """Degrees from the closest hard joint limit.

        The minimum over axes of ``min(q - lower, upper - q)``, which is the per-axis proximity
        :class:`IKQualityGuard` computes in ``src/robot/safety/ik_quality.py``, so the carrier value
        and the guard agree on one envelope.

        Returns ``None`` when no joint-limit table resolves: no limits means no margin. A solution
        already past a limit clamps to ``0.0``, because the carrier requires a value of at least
        zero and the feasibility scorer then gives it a sub-score of 0, fully demoted. The limits
        come from :func:`resolve_joint_limits_deg`, either the config static table or the UR vendor
        table, keyed by the arm's vendor and model: the same resolver the joint-limit and IK-quality
        guards use.
        """
        if self._joint_limits_config is None:
            return None
        import math

        from src.robot.safety.joint_limits import resolve_joint_limits_deg

        caps = getattr(self._arm, "capabilities", None)
        limits = resolve_joint_limits_deg(
            self._joint_limits_config,
            vendor=getattr(caps, "vendor", None),
            model=getattr(caps, "model", None),
        )
        if limits is None:
            return None
        lo_deg, hi_deg = limits
        values = np.asarray(joints.values, dtype=np.float64)
        if len(lo_deg) != values.size:
            return None  # table length disagrees with the DoF, so abstain
        margins = [
            min(math.degrees(float(values[a])) - lo_deg[a], hi_deg[a] - math.degrees(float(values[a])))
            for a in range(values.size)
        ]
        return max(0.0, float(min(margins)))


def _finite_or_none(value: float) -> Optional[float]:
    v = float(value)
    return v if np.isfinite(v) else None


__all__ = ["ArmBackedIKService"]
