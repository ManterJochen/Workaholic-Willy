"""MotionContinuityGuard, the step-size guard between consecutive commanded targets.

It catches an accidental ``move(other_pose)`` that would slam the arm across the
workspace, which is the operator mistake of clicking the wrong waypoint or sending a
stale target after a workspace re-frame.

Memo
----
The previous accepted target lives on :class:`SafetyPreflight`, as
``_last_target_pose`` and ``_last_target_joints``, and is folded into
:class:`SafetyContext` by :meth:`SafetyPreflight.context_for_pose` and
:meth:`SafetyPreflight.context_for_joints`. This guard always accepts the first
command after a :meth:`SafetyPreflight.reset`, because there is no previous target to
compare against.

Three checks, each applied where its data is available:

* Joint step, the largest absolute per-axis joint delta in degrees.
* TCP step, the Cartesian distance in mm between the previous and the current
  commanded TCP positions.
* Orientation step, the geodesic angle in degrees between the previous and the current
  commanded TCP orientations.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .decision import SafetyDecision, SafetyReason
from .guard import SafetyContext

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import MotionContinuitySafetyConfig

__all__ = ["MotionContinuityGuard"]


class MotionContinuityGuard:
    """Continuity-of-motion guard."""

    name = "motion_continuity"

    def __init__(self, config: "MotionContinuitySafetyConfig") -> None:
        self._max_joint_step_deg = float(config.max_joint_step_deg)
        self._max_orientation_step_deg = float(config.max_orientation_step_deg)
        self._max_tcp_step_mm = float(config.max_tcp_step_mm)

    def evaluate(self, ctx: SafetyContext) -> SafetyDecision:
        # ---- Joint step ----------------------------------------------
        if ctx.target_joints is not None and ctx.last_target_joints is not None:
            prev = ctx.last_target_joints
            now = ctx.target_joints
            if prev.dof == now.dof:
                delta_rad = np.abs(now.values - prev.values)
                worst = int(np.argmax(delta_rad))
                worst_deg = math.degrees(float(delta_rad[worst]))
                if worst_deg > self._max_joint_step_deg:
                    return SafetyDecision.reject(
                        self.name,
                        SafetyReason.CONTINUITY,
                        message=(
                            f"joint step axis {worst}: "
                            f"{worst_deg:.3f} deg > "
                            f"{self._max_joint_step_deg:.3f} deg"
                        ),
                        detail={
                            "reason": "joint_step",
                            "axis": str(worst),
                            "step_deg": f"{worst_deg:.6f}",
                            "max_joint_step_deg": (
                                f"{self._max_joint_step_deg:.6f}"
                            ),
                        },
                    )

        # ---- TCP step + orientation step -----------------------------
        if ctx.target_pose is not None and ctx.last_target_pose is not None:
            prev_pose = ctx.last_target_pose
            now_pose = ctx.target_pose
            tcp_step = float(
                np.linalg.norm(now_pose.position_mm - prev_pose.position_mm)
            )
            if tcp_step > self._max_tcp_step_mm:
                return SafetyDecision.reject(
                    self.name,
                    SafetyReason.CONTINUITY,
                    message=(
                        f"TCP step {tcp_step:.3f} mm > "
                        f"{self._max_tcp_step_mm:.3f} mm"
                    ),
                    detail={
                        "reason": "tcp_step",
                        "step_mm": f"{tcp_step:.6f}",
                        "max_tcp_step_mm": (
                            f"{self._max_tcp_step_mm:.6f}"
                        ),
                    },
                )
            angle_deg = math.degrees(float(prev_pose.angle_to(now_pose)))
            if angle_deg > self._max_orientation_step_deg:
                return SafetyDecision.reject(
                    self.name,
                    SafetyReason.CONTINUITY,
                    message=(
                        f"orientation step {angle_deg:.3f} deg > "
                        f"{self._max_orientation_step_deg:.3f} deg"
                    ),
                    detail={
                        "reason": "orientation_step",
                        "step_deg": f"{angle_deg:.6f}",
                        "max_orientation_step_deg": (
                            f"{self._max_orientation_step_deg:.6f}"
                        ),
                    },
                )

        return SafetyDecision.accept(self.name)
