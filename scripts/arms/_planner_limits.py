"""The planner's envelope and its clip, as a descriptor and the retract rule hand them to cuRobo (UM6). Stdlib only.

``scripts/arms/descriptor.py`` runs under the cuRobo interpreter, where this repository cannot be imported, so this lives
beside it and a test loads the same file by path. Since P0f F7 the numbers are the arm description's,
``planner_limit_deg`` per joint and ``planner.position_limit_clip_rad`` (read through ``_arm_file``); this module only
computes with what it is handed.

Two limits, and only together do they mean anything:

* The narrowed joints. Universal Robots plans ``elbow_joint`` within +-180 degrees (its ``joint_limits.yaml``),
  while the arm itself turns +-360. A descriptor inherits whichever number its URDF carries, so the builder writes the
  planner's limit into the description instead of trusting it (:func:`clamp_to_planner_limits`).
* Every joint's clip. cuRobo narrows every joint by ``position_limit_clip`` when it loads the descriptor. 0.1 rad is
  5.73 degrees, which is wider than the guard's default 5 degree margin, and that is the whole reason the planner
  cannot propose a configuration the guard then refuses. :func:`violations` is that comparison, so a cell with a wider
  margin is caught by a test rather than on a bench.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

__all__ = ["clamp_to_planner_limits", "narrowed_rad", "planner_envelope_rad", "violations"]

#: Every joint element, not the first one that carries the name. A UR description declares ``elbow_joint`` twice:
#: once as a ``ros2_control`` block with command interfaces and no ``<limit>``, and once as the revolute joint that
#: has one. Taking the first match rewrites nothing and reports "already inside the limit", which is the same answer
#: a correct run gives on a correct file.
_JOINT_BLOCK = re.compile(r"<joint\b[^>]*>.*?</joint>", re.S)
_NAME = re.compile(r'name="([^"]+)"')
#: Only the joint's own bounds: ``soft_lower_limit`` and friends do not match, because ``\b`` fails after an underscore.
_BOUND = re.compile(r'\b(lower|upper)="(-?[0-9.eE+-]+)"')


def planner_envelope_rad(
    limits_deg: Sequence[Sequence[float]], clip: float,
) -> "tuple[tuple[float, ...], tuple[float, ...]]":
    """``(lower, upper)`` per joint in radians, as cuRobo holds them after its ``position_limit_clip`` of ``clip``."""
    return (tuple(math.radians(low) + clip for low, _high in limits_deg),
            tuple(math.radians(high) - clip for _low, high in limits_deg))


def narrowed_rad(
    names: Sequence[str], planner_deg: Sequence[Sequence[float]], guard_deg: Sequence[Sequence[float]],
) -> dict[str, list[float]]:
    """The joints the planner is held narrower than the guard, ``{name: [low, high]}`` in radians, base to flange.

    What a descriptor clamps its URDF to and records as ``planner_joint_limits``: the elbow alone on every UR.
    """
    return {name: [math.radians(planner[0]), math.radians(planner[1])]
            for name, planner, guard in zip(names, planner_deg, guard_deg, strict=True)
            if tuple(planner) != tuple(guard)}


def violations(
    envelope_deg: "tuple[tuple[float, ...], tuple[float, ...]]",
    guard_lo: "tuple[float, ...]",
    guard_hi: "tuple[float, ...]",
    margin_deg: float,
) -> "tuple[str, ...]":
    """Where the planner may propose what the guard refuses, one sentence per axis and bound; empty when it cannot."""
    lower, upper = envelope_deg
    out: list[str] = []
    for axis, (plan_lo, plan_hi, low, high) in enumerate(zip(lower, upper, guard_lo, guard_hi)):
        allowed_lo = float(low) + float(margin_deg)
        allowed_hi = float(high) - float(margin_deg)
        if plan_lo < allowed_lo - 1e-9:
            out.append(f"joint {axis}: the planner may reach {plan_lo:.3f} deg and the guard refuses below "
                       f"{allowed_lo:.3f} deg")
        if plan_hi > allowed_hi + 1e-9:
            out.append(f"joint {axis}: the planner may reach {plan_hi:.3f} deg and the guard refuses above "
                       f"{allowed_hi:.3f} deg")
    return tuple(out)


def clamp_to_planner_limits(urdf_text: str, narrowed: Mapping[str, Sequence[float]]) -> "tuple[str, bool]":
    """The description with each narrowed joint's ``<limit>`` held to its planner limit, and whether anything changed.

    A description already inside every limit comes back byte for byte: every description is rewritten on every build,
    and an implementation that re-serialised the XML would change all of them for nothing. A narrowed joint the
    description does not declare with a ``<limit>`` is refused rather than passed through, because then nothing wrote
    the planner's limit anywhere.
    """
    changed = False
    out: list[str] = []
    end = 0
    bounded: set[str] = set()
    for block in _JOINT_BLOCK.finditer(urdf_text):
        text = block.group(0)
        name = _NAME.search(text)
        if name is None or name.group(1) not in narrowed or "<limit" not in text:
            continue
        low, high = (float(value) for value in narrowed[name.group(1)])
        bounded.add(name.group(1))

        def bound(match: "re.Match[str]", low: float = low, high: float = high) -> str:
            nonlocal changed
            which, value = match.group(1), float(match.group(2))
            if (which == "lower" and value >= low - 1e-9) or (which == "upper" and value <= high + 1e-9):
                return match.group(0)
            changed = True
            return f'{which}="{low if which == "lower" else high:.8f}"'

        out.append(urdf_text[end:block.start()])
        out.append(_BOUND.sub(bound, text))
        end = block.end()
    missing = sorted(set(narrowed) - bounded)
    if missing:
        raise ValueError(
            f"this description declares no {', '.join(missing)} with a <limit>, so the planner's limit cannot be "
            "written into it"
        )
    if not changed:
        return urdf_text, False
    out.append(urdf_text[end:])
    return "".join(out), True
