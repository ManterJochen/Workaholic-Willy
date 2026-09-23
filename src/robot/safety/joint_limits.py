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

Half a turn either side of home
-------------------------------
``safety.joint_limits.within_half_turn_of_home`` narrows the table on every axis whose range is a
full turn or more to ``[home - 180, home + 180]`` degrees, home being the configuration the arm
itself states (:func:`stated_home_rad`), and the margin comes off that. It is how a cell whose
cables run along the arm keeps every joint within half a turn of where the cables were laid,
without anybody typing a joint range. It costs no reach where the window lies inside the table:
every UR joint is 2*pi-periodic, so an angle has exactly one twin in any full-turn window, and a
configuration the window refuses is one whose twin inside it is the same pose. What it refuses is
the move across the seam half a turn from home, the long way round, and a band of
``2 * margin_deg`` at that seam, which the margin costs and the window does not. Measured
2026-09-23 with the closed-form inverse kinematics on 20000 random UR10 poses, homes within 174
degrees of zero: none lost at a margin of zero; 73 lost at 5 degrees, each needing a joint within 5
degrees of the seam in every configuration the window holds. The UR driver chooses a Cartesian goal's
configuration inside the same window (``_goal_joint_window``), so the guard and the choice cannot
disagree.

Reach is not the same as a plan. Where the only path cuRobo finds from where the arm stands crosses
the seam, the window refuses the move, and the configuration inside it may have no collision-free
path at all. Measured 2026-09-23 on the real sidecar (UR10, 2F-85, home facing the work): example
10's three rings and a ring round the owner's marker ran exactly as without the window; example 09's
automatic sweep, 22 tool-down poses all round the base, ran 10 legs where it ran 18.

A UR elbow is left as the table has it (:data:`UR_ELBOW_AXIS`). Universal Robots plans it within
+-180 degrees and the planner within +-(pi - 0.1), so it never turns a full turn and has no twin to
fall back on: narrowed about home, it refuses the elbow bent the other way outright. Measured the
same day with the home at START (elbow at -90 degrees): narrowed, none of example 10's six radial
stations ran; left alone, three did. The forearm stops it short of half a turn either way, so there
is no cable round it to wind.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from src.robot.core import RobotError

from .decision import SafetyDecision, SafetyReason
from .guard import SafetyContext

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from collections.abc import Sequence

    from src.config.schema.robot import JointLimitSafetyConfig

__all__ = [
    "HALF_TURN_DEG",
    "JointLimitGuard",
    "JointLimitTableMissing",
    "UR_ELBOW_AXIS",
    "UR_JOINT_LIMITS_DEG",
    "assert_joint_limit_table_available",
    "half_turn_about_home_deg",
    "half_turn_axes",
    "resolve_joint_limits_deg",
    "stated_home_rad",
]

#: How far either side of home ``within_half_turn_of_home`` lets a joint turn, in degrees.
HALF_TURN_DEG = 180.0
#: An axis whose range spans at least this many degrees is narrowed to the window about home.
_FULL_TURN_DEG = 360.0


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

#: The one UR joint ``within_half_turn_of_home`` leaves alone: the elbow, index 2. Its table says +-360, but Universal
#: Robots plans it within +-180 (``scripts/curobo/_planner_limits.py``) and cuRobo within +-(pi - 0.1), so it never
#: turns the full turn a window about home is for, and narrowing it would refuse the elbow bent the other way, which
#: has no twin inside.
UR_ELBOW_AXIS = 2


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


def stated_home_rad(arm: object) -> tuple[float, ...] | None:
    """The home configuration ``arm`` states, in radians, or ``None`` where it states none.

    It is read off ``arm.home_joint_positions``. The UR driver answers with the joints its
    ``move_home`` drives to: its constructor argument, else ``robot.home_joint_positions``, else
    :data:`~src.robot.constants.HOME_JOINTS_DEFAULT`. Anything that is not a non-empty sequence of
    finite numbers is no home, and a guard that needs one refuses rather than guessing.
    """
    value = getattr(arm, "home_joint_positions", None) if arm is not None else None
    if value is None:
        return None
    try:
        home = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if not home or not all(math.isfinite(v) for v in home):
        return None
    return home


def half_turn_axes(
    lower_deg: "Sequence[float]", upper_deg: "Sequence[float]", *, vendor: str | None = None,
) -> tuple[bool, ...]:
    """Which axes the window about home narrows: a range of a full turn or more, less a UR's elbow.

    An axis whose range is less than a full turn cannot wind a cable round, and narrowing it would
    refuse angles that have no twin; a UR elbow is one whatever its table says (:data:`UR_ELBOW_AXIS`).
    """
    return tuple(
        float(high) - float(low) >= _FULL_TURN_DEG and not (vendor == "ur" and axis == UR_ELBOW_AXIS)
        for axis, (low, high) in enumerate(zip(lower_deg, upper_deg))
    )


def half_turn_about_home_deg(
    lower_deg: "Sequence[float]",
    upper_deg: "Sequence[float]",
    home_rad: "Sequence[float]",
    *,
    vendor: str | None = None,
) -> tuple[list[float], list[float]]:
    """The table ``[lower_deg, upper_deg]`` narrowed to half a turn either side of home on every full-turn axis.

    Limits are in degrees, as the table is, and ``home_rad`` in radians, as the config and the driver
    hold it. :func:`half_turn_axes` says which axes are narrowed; the rest come back as they are. No
    margin is taken off here; the guard takes it off the result.

    An axis comes back empty, its lower bound above its upper, where home lies more than half a turn
    outside the table, which is how a home written in degrees reads. The caller reports that.

    Raises
    ------
    ValueError
        The three sequences are not one entry per axis.
    """
    if not len(lower_deg) == len(upper_deg) == len(home_rad):
        raise ValueError(
            f"the table has {len(lower_deg)} and {len(upper_deg)} bounds and home {len(home_rad)} joints: "
            "one of each per axis"
        )
    lower: list[float] = []
    upper: list[float] = []
    narrowed = half_turn_axes(lower_deg, upper_deg, vendor=vendor)
    for low, high, home, narrow in zip(lower_deg, upper_deg, home_rad, narrowed):
        low, high = float(low), float(high)
        if narrow:
            centre = math.degrees(float(home))
            low, high = max(low, centre - HALF_TURN_DEG), min(high, centre + HALF_TURN_DEG)
        lower.append(low)
        upper.append(high)
    return lower, upper


class JointLimitGuard:
    """Per-axis joint hard-limit guard.

    :meth:`SafetyPreflight.from_safety_config` constructs it when
    ``safety.joint_limits.enforce`` is ``True``. Stateless across calls: with
    ``within_half_turn_of_home`` set, the home it centres the window on is read off the arm
    each context carries, so the window is always the one about the home that arm drives to.
    """

    name = "joint_limit"

    def __init__(self, config: "JointLimitSafetyConfig") -> None:
        self._config = config
        self._margin_deg = float(config.margin_deg)
        self._half_turn = bool(getattr(config, "within_half_turn_of_home", False))

    @property
    def within_half_turn_of_home(self) -> bool:
        """Whether this guard keeps every full-turn axis within half a turn of the arm's home."""
        return self._half_turn

    def limits_for(
        self, *, vendor: str | None, model: str | None, home_rad: "Sequence[float] | None" = None,
    ) -> tuple[list[float], list[float]] | None:
        """The table this guard would enforce for that vendor and model, or ``None`` for none.

        With ``within_half_turn_of_home`` set, it is the table narrowed to half a turn either side
        of ``home_rad``, and ``None`` without a home, because this guard then enforces nothing but a
        refusal. :meth:`limits_for_arm` reads the home off an arm.

        It is public so a caller can ask the question before a move rather than discovering
        the answer as a refusal. :meth:`evaluate` asks it here too, so what the build check
        reads and what the runtime enforces cannot drift apart.
        """
        return self._resolve(vendor=vendor, model=model, home_rad=home_rad)[0]

    def table_for(self, *, vendor: str | None, model: str | None) -> tuple[list[float], list[float]] | None:
        """The static or vendor table under any window, or ``None`` where neither resolves: what the window narrows."""
        return resolve_joint_limits_deg(self._config, vendor=vendor, model=model)

    def limits_for_arm(self, arm: object) -> tuple[list[float], list[float]] | None:
        """:meth:`limits_for` the arm's own capabilities and, where the window is on, its own home."""
        caps = getattr(arm, "capabilities", None)
        return self.limits_for(
            vendor=getattr(caps, "vendor", None), model=getattr(caps, "model", None),
            home_rad=stated_home_rad(arm) if self._half_turn else None,
        )

    def _resolve(
        self, *, vendor: str | None, model: str | None, home_rad: "Sequence[float] | None",
    ) -> "tuple[tuple[list[float], list[float]] | None, str]":
        """The limits and ``""``, or ``None`` and the sentence saying which piece is missing."""
        table = self.table_for(vendor=vendor, model=model)
        if table is None:
            return None, f"no static or vendor joint-limit table configured (vendor={vendor!r}, model={model!r})"
        if not self._half_turn:
            return table, ""
        if home_rad is None:
            return None, (
                "safety.joint_limits.within_half_turn_of_home is set and the arm states no home "
                "configuration (home_joint_positions) to centre the window on"
            )
        if len(home_rad) != len(table[0]):
            return None, (
                f"safety.joint_limits.within_half_turn_of_home is set and the arm's home has "
                f"{len(home_rad)} joints where the joint-limit table has {len(table[0])}"
            )
        return half_turn_about_home_deg(table[0], table[1], home_rad, vendor=vendor), ""

    @property
    def margin_deg(self) -> float:
        """How many degrees off each end of a range this guard keeps.

        A caller reads it to land a value inside the range the guard admits.
        """
        return self._margin_deg

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
        home = stated_home_rad(ctx.arm) if self._half_turn else None
        limits, missing = self._resolve(vendor=vendor, model=model, home_rad=home)
        if limits is None:
            return SafetyDecision.unavailable(
                self.name,
                message=missing,
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
        table = self.table_for(vendor=vendor, model=model)
        narrowed = (half_turn_axes(table[0], table[1], vendor=vendor)
                    if home is not None and table is not None else (False,) * len(lo_deg))
        for axis_idx in range(joints.dof):
            q_rad = float(joints.values[axis_idx])
            q_deg = math.degrees(q_rad)
            lower = lo_deg[axis_idx] + margin
            upper = hi_deg[axis_idx] - margin
            if lower >= upper:
                # A margin this large eats the whole range, so the config is broken.
                message = (
                    f"axis {axis_idx}: margin {margin} deg inverts "
                    f"range [{lo_deg[axis_idx]}, {hi_deg[axis_idx]}]"
                )
                if narrowed[axis_idx] and home is not None:
                    message += (
                        f", which is what is left of the table within half a turn of home "
                        f"({math.degrees(home[axis_idx]):.1f} deg on this axis); "
                        "robot.home_joint_positions is read in radians"
                    )
                return SafetyDecision.unavailable(
                    self.name,
                    message=message,
                    detail={
                        "axis": str(axis_idx),
                        "margin_deg": f"{margin:.6f}",
                        "min_deg": f"{lo_deg[axis_idx]:.6f}",
                        "max_deg": f"{hi_deg[axis_idx]:.6f}",
                    },
                )
            if q_deg < lower or q_deg > upper:
                detail = {
                    "axis": str(axis_idx),
                    "q_deg": f"{q_deg:.6f}",
                    "min_deg": f"{lo_deg[axis_idx]:.6f}",
                    "max_deg": f"{hi_deg[axis_idx]:.6f}",
                    "margin_deg": f"{margin:.6f}",
                    "lower_with_margin_deg": f"{lower:.6f}",
                    "upper_with_margin_deg": f"{upper:.6f}",
                }
                message = (
                    f"axis {axis_idx}: {q_deg:.3f} deg outside "
                    f"[{lower:.3f}, {upper:.3f}] (incl. margin "
                    f"{margin:.3f})"
                )
                if home is not None and narrowed[axis_idx]:
                    home_deg = math.degrees(home[axis_idx])
                    detail["home_deg"] = f"{home_deg:.6f}"
                    message += _half_turn_sentence(q_deg, home_deg, lower, upper, margin)
                return SafetyDecision.reject(
                    self.name,
                    SafetyReason.JOINT_LIMIT,
                    message=message,
                    detail=detail,
                )

        return SafetyDecision.accept(self.name)


def _half_turn_sentence(q_deg: float, home_deg: float, lower: float, upper: float, margin: float) -> str:
    """What a refusal adds on a guard that keeps half a turn about home: the window, and where the same angle lies."""
    said = (
        f"; that is half a turn either side of home ({home_deg:.1f} deg on this axis), which "
        "safety.joint_limits.within_half_turn_of_home keeps so a cable along the arm is never wound further"
    )
    turns = [q_deg + k * _FULL_TURN_DEG for k in (-2, -1, 1, 2)]
    inside = [t for t in turns if lower <= t <= upper]
    if inside:
        twin = min(inside, key=lambda t: abs(t - q_deg))
        return said + (
            f". The same angle a full turn away, {twin:.3f} deg, is inside it: this configuration is that one "
            "wound a full turn further, across the seam behind home"
        )
    return said + (
        f". No full turn of this angle is inside it: it lies within the {margin:.1f} deg margin of the seam "
        "half a turn from home, which no configuration inside the window reaches"
    )


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
        fix. It is raised too where the guard keeps half a turn about home and the arm states
        no home, or one that leaves an axis no range, which is how a home written in degrees
        reads; the message names ``robot.home_joint_positions``.
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
        if guard.table_for(vendor=vendor, model=model) is not None:
            _refuse_a_window_with_no_room(guard, arm)
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


def _refuse_a_window_with_no_room(guard: JointLimitGuard, arm: Any) -> None:
    """Refuse, at build, a guard keeping half a turn about home that the arm gives no home, or no room about it.

    Either would be refused at every move instead, as UNAVAILABLE, which reads as the controller's fault.
    """
    if not guard.within_half_turn_of_home:
        return
    home = stated_home_rad(arm)
    limits = guard.limits_for_arm(arm)
    if home is None or limits is None:
        raise JointLimitTableMissing(
            f"robot.safety.joint_limits.within_half_turn_of_home is set, and {type(arm).__name__} states "
            f"{'no home configuration' if home is None else f'a home of {len(home)} joints'} to centre the window "
            "on: set robot.home_joint_positions (radians, one entry per axis) on a driver that reads it, or turn "
            "the window off. Refused here rather than at the first move, where every motion would have come back "
            "as controller_rejected."
        )
    margin = guard.margin_deg
    for axis, (low, high) in enumerate(zip(*limits)):
        if low + margin >= high - margin:
            raise JointLimitTableMissing(
                f"robot.safety.joint_limits.within_half_turn_of_home leaves axis {axis} no range: its home is "
                f"{math.degrees(home[axis]):.1f} deg, more than half a turn outside the joint-limit table less "
                f"its {margin:.1f} deg margin. robot.home_joint_positions is read in radians, so a home written in "
                "degrees reads this way."
            )
