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

Vendor-neutral: it needs only ``arm.fk`` and ``arm.ik``, so any :class:`RobotArm` with native FK
works. It lives in willy_sim, the composition layer, so the grasping layer sees only the injected
Protocol.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.robot.grasping.planning.reachability import IKResult


class ArmBackedIKService:
    """``IKService`` that rejects unreachable and near-singular grasp poses via the arm kinematics."""

    def __init__(
        self,
        arm: Any,
        *,
        thresholds: Any | None = None,
        debug: bool = False,
    ) -> None:
        self._arm = arm
        self._thresholds = thresholds  # safety.SingularityThresholds, or None for the safe default
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

        if self._debug:
            print(f"[ik-probe] pos={np.round(geom.position_mm, 1)} cond={report.condition_number:.1f} "
                  f"min_sv={report.min_singular_value:.4f} "
                  f"near_singular={report.is_near_singularity}", flush=True)
        if report.is_near_singularity:
            return IKResult(reachable=False, reason="near_singular")
        return IKResult(reachable=True)


__all__ = ["ArmBackedIKService"]
