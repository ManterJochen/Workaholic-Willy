"""JointLimitGuard, the per-axis joint hard-limit safety guard.

The per-axis limits the guard enforces come from the first of these that answers:

1. The static ``min_deg`` and ``max_deg`` lists set in
   :class:`src.config.schema.robot.JointLimitSafetyConfig`. They come first because an
   operator-supplied value is the closest thing to ground truth and overrides any
   built-in default.
2. The built-in vendor table, keyed on ``arm.capabilities.vendor`` and
   ``arm.capabilities.model``. Only Universal Robots have a built-in table here, so
   another vendor supplies static ``min_deg`` and ``max_deg`` in YAML.
3. Neither, which is :attr:`SafetyReason.UNAVAILABLE`. With ``enforce=True``, the
   default, the preflight refuses the motion and the operator sees a fail-closed
   refusal.

Units
-----
* ``JointPositions`` carries radians, and the guard converts to degrees at the
  boundary, because both the static config and the vendor table are in degrees, which
  is how an operator reads a pendant display.
* ``margin_deg`` is subtracted from the allowed range of each axis at both ends.

What each vendor exposes
------------------------
* ``ur_rtde`` does not expose joint limits as runtime telemetry; they live in the
  URCap installation file. The vendor table below carries the factory ranges from the
  Universal Robots user manuals, and a tighter pendant-configured range is enforced by
  the controller as a protective stop. This guard therefore catches the
  manufacturer-specified envelope, and the controller-side envelope is the more
  restrictive backstop.
* KUKA EKI does not expose joint limits either. The shipped ``robot.yaml`` populates
  ``min_deg`` and ``max_deg`` per model, so the guard never fails closed on a clean
  install.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .decision import SafetyDecision, SafetyReason
from .guard import SafetyContext

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import JointLimitSafetyConfig

__all__ = [
    "JointLimitGuard",
    "UR_JOINT_LIMITS_DEG",
    "resolve_joint_limits_deg",
]


# The Universal Robots factory joint-position envelope, from the user manual under
# "Joint position limits" in the "Safety configuration" chapter. Every e-Series and
# CB-Series arm ships with the same +/-360 deg per axis, and a tighter
# operator-configured limit lives in the URCap installation file and is enforced by
# the controller as a protective stop. This guard catches the manufacturer-specified
# outer envelope, so commanded joints the controller would refuse never leave Python.
_UR_PLUSMINUS_360: tuple[tuple[float, ...], tuple[float, ...]] = (
    (-360.0, -360.0, -360.0, -360.0, -360.0, -360.0),
    (360.0, 360.0, 360.0, 360.0, 360.0, 360.0),
)

UR_JOINT_LIMITS_DEG: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "ur3": _UR_PLUSMINUS_360,
    "ur3e": _UR_PLUSMINUS_360,
    "ur5": _UR_PLUSMINUS_360,
    "ur5e": _UR_PLUSMINUS_360,
    "ur10": _UR_PLUSMINUS_360,
    "ur10e": _UR_PLUSMINUS_360,
    "ur16e": _UR_PLUSMINUS_360,
    "ur20": _UR_PLUSMINUS_360,
}


def resolve_joint_limits_deg(
    config: "JointLimitSafetyConfig",
    *,
    vendor: str | None,
    model: str | None,
) -> tuple[list[float], list[float]] | None:
    """Resolve per-axis joint limits in degrees, from config and capabilities.

    Returns ``None`` where neither the config nor the built-in vendor table gives a
    usable range. A caller treats ``None`` as the guard being unavailable, never as an
    absence of limits.
    """
    if config.min_deg is not None and config.max_deg is not None:
        # The Pydantic validator already enforces equal length and ordering.
        return (list(config.min_deg), list(config.max_deg))
    if vendor == "ur" and model is not None:
        entry = UR_JOINT_LIMITS_DEG.get(model.lower())
        if entry is not None:
            lo, hi = entry
            return (list(lo), list(hi))
    return None


class JointLimitGuard:
    """Per-axis joint hard-limit guard.

    :meth:`SafetyPreflight.from_safety_config` constructs it when
    ``safety.joint_limits.enforce`` is ``True``. Stateless across calls.
    """

    name = "joint_limit"

    def __init__(self, config: "JointLimitSafetyConfig") -> None:
        self._config = config
        self._margin_deg = float(config.margin_deg)

    def evaluate(self, ctx: SafetyContext) -> SafetyDecision:
        joints = ctx.target_joints
        if joints is None:
            # There is no joint solution to check: either the driver has not
            # pre-resolved IK, or the command is a pure Cartesian path on a vendor
            # with no native IK, such as the sim mock.
            return SafetyDecision.unavailable(
                self.name,
                message="no target_joints available for joint-limit check",
                detail={"reason": "missing_target_joints"},
            )

        vendor: str | None = None
        model: str | None = None
        if ctx.arm is not None:
            caps = ctx.arm.capabilities
            vendor = caps.vendor
            model = caps.model
        limits = resolve_joint_limits_deg(
            self._config, vendor=vendor, model=model,
        )
        if limits is None:
            return SafetyDecision.unavailable(
                self.name,
                message=(
                    "no static or vendor joint-limit table configured "
                    f"(vendor={vendor!r}, model={model!r})"
                ),
                detail={
                    "vendor": str(vendor),
                    "model": str(model),
                },
            )
        lo_deg, hi_deg = limits
        if len(lo_deg) != joints.dof:
            return SafetyDecision.unavailable(
                self.name,
                message=(
                    f"joint-limit table length {len(lo_deg)} != arm DoF "
                    f"{joints.dof}"
                ),
                detail={
                    "table_len": str(len(lo_deg)),
                    "dof": str(joints.dof),
                },
            )

        margin = self._margin_deg
        for axis_idx in range(joints.dof):
            q_rad = float(joints.values[axis_idx])
            q_deg = math.degrees(q_rad)
            lower = lo_deg[axis_idx] + margin
            upper = hi_deg[axis_idx] - margin
            if lower >= upper:
                # A margin this large eats the whole range, so the config is broken.
                return SafetyDecision.unavailable(
                    self.name,
                    message=(
                        f"axis {axis_idx}: margin {margin} deg inverts "
                        f"range [{lo_deg[axis_idx]}, {hi_deg[axis_idx]}]"
                    ),
                    detail={
                        "axis": str(axis_idx),
                        "margin_deg": f"{margin:.6f}",
                        "min_deg": f"{lo_deg[axis_idx]:.6f}",
                        "max_deg": f"{hi_deg[axis_idx]:.6f}",
                    },
                )
            if q_deg < lower or q_deg > upper:
                return SafetyDecision.reject(
                    self.name,
                    SafetyReason.JOINT_LIMIT,
                    message=(
                        f"axis {axis_idx}: {q_deg:.3f} deg outside "
                        f"[{lower:.3f}, {upper:.3f}] (incl. margin "
                        f"{margin:.3f})"
                    ),
                    detail={
                        "axis": str(axis_idx),
                        "q_deg": f"{q_deg:.6f}",
                        "min_deg": f"{lo_deg[axis_idx]:.6f}",
                        "max_deg": f"{hi_deg[axis_idx]:.6f}",
                        "margin_deg": f"{margin:.6f}",
                        "lower_with_margin_deg": f"{lower:.6f}",
                        "upper_with_margin_deg": f"{upper:.6f}",
                    },
                )

        return SafetyDecision.accept(self.name)
