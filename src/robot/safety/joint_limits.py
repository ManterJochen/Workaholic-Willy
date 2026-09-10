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
* KUKA EKI does not expose joint limits either, and the shipped ``robot.yaml`` does not
  fill the gap: ``safety.joint_limits.min_deg`` and ``max_deg`` both ship as ``null`` in
  ``config/robot/robot.yaml``. Measured 2026-09-10 on the ``web`` profile, the only
  ``vendor: kuka`` config in the tree, ``resolve_joint_limits_deg`` returned ``None``, the
  guard answered UNAVAILABLE for an all-zeros joint target, and the preflight turned that
  into ``controller_rejected``. A KUKA cell refused every motion and named the controller
  for a missing config key. These three lines said the opposite until then, which is why
  nobody looked. That refusal now happens at build instead, see
  :func:`assert_joint_limit_table_available`, called from
  :func:`src.robot.drivers.create_arm`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from src.robot.core import RobotError

from .decision import SafetyDecision, SafetyReason
from .guard import SafetyContext

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import JointLimitSafetyConfig

__all__ = [
    "JointLimitGuard",
    "JointLimitTableMissing",
    "UR_JOINT_LIMITS_DEG",
    "assert_joint_limit_table_available",
    "resolve_joint_limits_deg",
]


class JointLimitTableMissing(RobotError):
    """This cell wired an enforcing joint-limit guard that has no table to enforce.

    It is raised at build rather than at the first move. The runtime alternative was
    measured on the shipped ``web`` profile and it is worse in the one way that matters:
    the guard answers UNAVAILABLE, the preflight fails closed as ``controller_rejected``,
    and the operator reads a controller fault for a missing key in their own YAML. Same
    refusal, wrong address, once per commanded move.
    """


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

    def limits_for(
        self, *, vendor: str | None, model: str | None,
    ) -> tuple[list[float], list[float]] | None:
        """The table this guard would enforce for that vendor and model, or ``None`` for none.

        It is public so a caller can ask the question before a move rather than discovering
        the answer as a refusal. :meth:`evaluate` asks it here too, so what the build check
        reads and what the runtime enforces cannot drift apart.
        """
        return resolve_joint_limits_deg(self._config, vendor=vendor, model=model)

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
        limits = self.limits_for(vendor=vendor, model=model)
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


def assert_joint_limit_table_available(arm: object) -> None:
    """Refuse an arm whose enforcing joint-limit guard has no table, at build.

    A cell that refuses every motion is indistinguishable from a cell with a broken
    controller. Measured 2026-09-10 on the shipped ``web`` profile:
    ``resolve_joint_limits_deg`` returned ``None`` for ``vendor: kuka``, so
    ``JointLimitGuard`` answered UNAVAILABLE for an all-zeros joint target, legal and
    absurd and it made no difference, and ``SafetyPreflight`` mapped that to
    ``controller_rejected``. Fail-closed, therefore safe, and unusable, and pointing at
    the wrong box. The cost of leaving it at runtime is one wrong diagnosis per operator.

    It is asked of the arm rather than of the config, so it costs a new vendor nothing:
    the driver states its own ``capabilities`` and its own pipeline, and both already
    exist. A vendor with no built-in table here and no static lists in YAML cannot boot.
    UR resolves from :data:`UR_JOINT_LIMITS_DEG`, and ``sim`` resolves from the explicit
    lists in ``robot.sim.yaml``.

    It is silent for an arm that gates nothing, where ``safety_preflight`` is ``None``, as
    the dummy has it, and for one whose operator set ``joint_limits.enforce: false``,
    because ``from_safety_config`` then leaves the guard out of the pipeline entirely and
    there is no guard here to starve. Those are stated decisions, and
    ``SafetyAttestation`` is what reports them.

    Raises
    ------
    JointLimitTableMissing
        The arm carries a :class:`JointLimitGuard` that resolves no table for the arm's
        own vendor and model. The message names ``min_deg`` and ``max_deg``, which is the
        fix.
    """
    preflight = getattr(arm, "safety_preflight", None)
    if preflight is None:
        return
    caps = getattr(arm, "capabilities", None)
    vendor = getattr(caps, "vendor", None)
    model = getattr(caps, "model", None)
    for guard in getattr(preflight, "guards", ()):
        if not isinstance(guard, JointLimitGuard):
            continue
        if guard.limits_for(vendor=vendor, model=model) is not None:
            continue
        raise JointLimitTableMissing(
            f"vendor {vendor!r} (model {model!r}) enforces the joint-limit guard but no table "
            f"resolves for it: set robot.safety.joint_limits.min_deg and "
            f"robot.safety.joint_limits.max_deg (one entry per axis, degrees) in your robot "
            f"config, or set robot.safety.joint_limits.enforce to false and accept an ungated "
            f"axis envelope. Only these models carry a built-in table: "
            f"{', '.join(sorted(UR_JOINT_LIMITS_DEG))}. Refused here rather than at the first "
            f"move, where the guard would have answered UNAVAILABLE and every motion would have "
            f"come back as controller_rejected."
        )
