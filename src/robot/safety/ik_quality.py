"""IKQualityGuard, the IK-solution quality safety guard.

It runs after the driver has produced a joint solution in ``ctx.target_joints`` for
the commanded TCP pose, and it rejects:

* a NaN or infinite value, or a wrong-DoF vector;
* a joint solution that differs from ``ctx.current_joints`` by more than
  ``max_jump_rad`` on any axis, because a large IK jump signals a wrap-around
  ambiguity that would slam the arm 180 deg around an axis to land on the same TCP
  pose;
* a near-singular configuration, through :func:`analyze_joint_singularity` and its
  Jacobian condition number and smallest singular value;
* a configuration within ``limit_proximity_deg`` of the configured joint limits, which
  come from :func:`src.robot.safety.joint_limits.resolve_joint_limits_deg` so that
  this guard and :class:`JointLimitGuard` agree on the envelope.

The guard calls :func:`analyze_joint_singularity` rather than duplicating the Jacobian
maths, so its own work is the finiteness, jump and proximity checks.

Operating modes
---------------
* With ``ctx.target_joints`` as ``None`` the guard returns
  :attr:`SafetyReason.UNAVAILABLE`, and the preflight fails closed where
  ``enforce=True``. Pre-resolving IK on a Cartesian command is the driver task, in
  ``URRobotArm.move``.
* With ``ctx.arm`` as ``None``, or ``arm.capabilities.has_native_fk`` false, the
  singularity estimate is skipped and the other checks still run.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from src.robot.core.errors import RobotError

from .decision import SafetyDecision, SafetyReason
from .guard import SafetyContext
from .joint_limits import resolve_joint_limits_deg
from .singularity import SingularityThresholds, analyze_joint_singularity

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import (
        IkQualitySafetyConfig,
        JointLimitSafetyConfig,
    )

__all__ = ["IKQualityGuard"]


class IKQualityGuard:
    """IK-solution quality guard.

    Stateless across calls. :meth:`SafetyPreflight.from_safety_config` constructs it
    when ``safety.ik_quality.enforce`` is ``True``.

    Parameters
    ----------
    config
        :class:`IkQualitySafetyConfig`, carrying the per-axis jump cap, the Jacobian
        thresholds and the limit-proximity buffer.
    joint_limits_config
        :class:`JointLimitSafetyConfig`, used by the limit-proximity check. Set
        alongside the joint-limit guard the two agree on the envelope, and set alone
        the proximity check still works.
    """

    name = "ik_quality"

    def __init__(
        self,
        config: "IkQualitySafetyConfig",
        joint_limits_config: "JointLimitSafetyConfig | None" = None,
    ) -> None:
        self._config = config
        self._joint_limits_config = joint_limits_config
        self._thresholds = SingularityThresholds(
            min_singular_value=float(config.min_singular_value),
            max_condition_number=float(config.max_condition_number),
        )
        self._max_jump_rad = float(config.max_jump_rad)
        self._limit_proximity_deg = float(config.limit_proximity_deg)

    def evaluate(self, ctx: SafetyContext) -> SafetyDecision:
        joints = ctx.target_joints
        if joints is None:
            return SafetyDecision.unavailable(
                self.name,
                message="no target_joints available for IK-quality check",
                detail={"reason": "missing_target_joints"},
            )

        # ---- NaN / inf -----------------------------------------------
        # JointPositions rejects a non-finite value at construction, but a buggy
        # driver adapter pushing raw arrays bypasses the type, so this re-checks.
        values = joints.values
        if not np.isfinite(values).all():
            return SafetyDecision.reject(
                self.name,
                SafetyReason.IK_QUALITY,
                message="IK solution contains non-finite values",
                detail={"reason": "non_finite"},
            )

        # ---- Wrong DoF ------------------------------------------------
        if ctx.arm is not None:
            expected = ctx.arm.capabilities.dof
            if joints.dof != expected:
                return SafetyDecision.reject(
                    self.name,
                    SafetyReason.IK_QUALITY,
                    message=(
                        f"IK solution has {joints.dof} axes; arm "
                        f"expects {expected}"
                    ),
                    detail={
                        "reason": "wrong_dof",
                        "got": str(joints.dof),
                        "expected": str(expected),
                    },
                )

        # ---- Joint jump from current ---------------------------------
        current = ctx.current_joints
        if current is not None and current.dof == joints.dof:
            delta = np.abs(values - current.values)
            worst = int(np.argmax(delta))
            worst_delta = float(delta[worst])
            if worst_delta > self._max_jump_rad:
                return SafetyDecision.reject(
                    self.name,
                    SafetyReason.IK_QUALITY,
                    message=(
                        f"IK jump on axis {worst}: "
                        f"{worst_delta:.4f} rad > "
                        f"{self._max_jump_rad:.4f} rad"
                    ),
                    detail={
                        "reason": "joint_jump",
                        "axis": str(worst),
                        "delta_rad": f"{worst_delta:.6f}",
                        "max_jump_rad": f"{self._max_jump_rad:.6f}",
                    },
                )

        # ---- Limit-proximity -----------------------------------------
        # Reusing the joint-limit resolver keeps this guard and the joint-limit guard
        # on one envelope. The check is soft: with no static or vendor table available
        # it skips rather than fails closed, which the joint-limit guard handles.
        if self._joint_limits_config is not None:
            vendor = ctx.arm.capabilities.vendor if ctx.arm is not None else None
            model = ctx.arm.capabilities.model if ctx.arm is not None else None
            limits = resolve_joint_limits_deg(
                self._joint_limits_config, vendor=vendor, model=model,
            )
            if limits is not None and len(limits[0]) == joints.dof:
                lo_deg, hi_deg = limits
                buffer_deg = self._limit_proximity_deg
                for axis_idx in range(joints.dof):
                    q_deg = math.degrees(float(values[axis_idx]))
                    near_lower = q_deg - lo_deg[axis_idx]
                    near_upper = hi_deg[axis_idx] - q_deg
                    closest = min(near_lower, near_upper)
                    if closest < buffer_deg:
                        return SafetyDecision.reject(
                            self.name,
                            SafetyReason.IK_QUALITY,
                            message=(
                                f"axis {axis_idx} at {q_deg:.3f} deg is "
                                f"only {closest:.3f} deg from a joint "
                                f"limit (buffer={buffer_deg:.3f})"
                            ),
                            detail={
                                "reason": "limit_proximity",
                                "axis": str(axis_idx),
                                "q_deg": f"{q_deg:.6f}",
                                "min_deg": f"{lo_deg[axis_idx]:.6f}",
                                "max_deg": f"{hi_deg[axis_idx]:.6f}",
                                "proximity_deg": f"{closest:.6f}",
                                "buffer_deg": f"{buffer_deg:.6f}",
                            },
                        )

        # ---- Singularity (Jacobian) ----------------------------------
        if ctx.arm is not None and ctx.arm.capabilities.has_native_fk:
            try:
                report = analyze_joint_singularity(
                    ctx.arm, joints, thresholds=self._thresholds,
                )
            except (RobotError, RuntimeError, OSError, ValueError) as exc:
                # The FK probe failed mid-flight, for instance with the controller
                # dropped. Reporting unavailable rather than passing silently is what
                # tells an operator the check did not run.
                #
                # ``RobotError`` has to stay in the tuple for that to hold on real
                # hardware. ``IsaacNotAvailableError`` is also a RuntimeError, so the
                # sim reaches this branch either way, but ``RobotKinematicsError`` and
                # ``RobotConnectionError`` subclass ``RobotError`` alone, and those are
                # what ``URRobotArm.fk()`` raises (arm.py:463-471) when the connection
                # is closed or the controller refuses the solve. Without
                # ``RobotError`` they escape as an unhandled traceback mid-pick
                # instead of the typed unavailable this handler exists to produce.
                return SafetyDecision.unavailable(
                    self.name,
                    message=f"Jacobian probe failed: {exc}",
                    detail={
                        "reason": "jacobian_probe_failure",
                        "exception": type(exc).__name__,
                    },
                )
            if report.is_near_singularity:
                return SafetyDecision.reject(
                    self.name,
                    SafetyReason.IK_QUALITY,
                    message=(
                        "near-singular IK solution: "
                        + "; ".join(report.reasons)
                    ),
                    detail={
                        "reason": "singularity",
                        "min_singular_value": (
                            f"{report.min_singular_value:.6g}"
                        ),
                        "condition_number": (
                            f"{report.condition_number:.3f}"
                        ),
                        "rank": str(report.rank),
                        "expected_rank": str(report.expected_rank),
                    },
                )

        return SafetyDecision.accept(self.name)
