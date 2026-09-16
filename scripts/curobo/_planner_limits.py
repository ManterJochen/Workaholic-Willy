"""UR's planning envelope, declared rather than inherited: the elbow at +-180 degrees. Standard library only.

``build_ur_config.py`` runs under the cuRobo interpreter, where this repository cannot be imported, so this lives
beside it and is loaded by path.

Two limits, and only together do they mean anything:

* The elbow. Universal Robots plans ``elbow_joint`` within +-180 degrees (its ``joint_limits.yaml``), while the arm
  itself turns +-360. A descriptor inherits whichever number the Isaac description it was copied from happens to
  carry, so the builder writes UR's limit into the description instead of trusting it.
* Every other joint. They keep the factory +-360, and cuRobo narrows every joint by ``position_limit_clip`` when it
  loads the descriptor. 0.1 rad is 5.73 degrees, which is wider than the guard's default 5 degree margin, and that is
  the whole reason the planner cannot propose a configuration the guard then refuses. :func:`violations` is that
  comparison, so a cell with a wider margin is caught before it reaches a bench.
"""

from __future__ import annotations

import math
import re

__all__ = [
    "ELBOW_JOINT",
    "ELBOW_INDEX",
    "ELBOW_LIMIT_RAD",
    "FULL_TURN_RAD",
    "POSITION_LIMIT_CLIP_RAD",
    "clamp_elbow_limit",
    "planner_envelope_rad",
    "violations",
]

#: UR's own planning limit for the elbow (Universal Robots ROS 2 description, joint_limits.yaml).
ELBOW_LIMIT_RAD = math.pi
#: What every other UR joint turns, and what the guard's factory table holds.
FULL_TURN_RAD = 2.0 * math.pi
#: cuRobo narrows every joint by this when it loads a descriptor (the UR template ships 0.1).
POSITION_LIMIT_CLIP_RAD = 0.1

ELBOW_JOINT = "elbow_joint"
ELBOW_INDEX = 2

#: Every joint element, not the first one that carries the name. A UR description declares ``elbow_joint`` twice:
#: once as a ``ros2_control`` block with command interfaces and no ``<limit>``, and once as the revolute joint that
#: has one. Taking the first match rewrites nothing and reports "already inside the limit", which is the same answer
#: a correct run gives on a correct file.
_JOINT_BLOCK = re.compile(r"<joint\b[^>]*>.*?</joint>", re.S)
_NAME = re.compile(r'name="([^"]+)"')
#: Only the joint's own bounds: ``soft_lower_limit`` and friends do not match, because ``\b`` fails after an underscore.
_BOUND = re.compile(r'\b(lower|upper)="(-?[0-9.eE+-]+)"')


def planner_envelope_rad(
    clip: float = POSITION_LIMIT_CLIP_RAD,
) -> "tuple[tuple[float, ...], tuple[float, ...]]":
    """``(lower, upper)`` per joint in radians, as cuRobo holds them after its ``position_limit_clip``."""
    limits = [FULL_TURN_RAD] * 6
    limits[ELBOW_INDEX] = ELBOW_LIMIT_RAD
    return tuple(-(limit - clip) for limit in limits), tuple(limit - clip for limit in limits)


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


def clamp_elbow_limit(urdf_text: str) -> "tuple[str, bool]":
    """The description with ``elbow_joint`` held to UR's +-pi, and whether anything changed.

    A description already inside the limit comes back byte for byte: six descriptions are rewritten on every build, and
    an implementation that re-serialised the XML would change all of them for nothing. A description without an elbow
    is refused rather than passed through, because then nothing wrote UR's limit anywhere.
    """
    changed = False

    def bound(match: "re.Match[str]") -> str:
        nonlocal changed
        name, value = match.group(1), float(match.group(2))
        if abs(value) <= ELBOW_LIMIT_RAD + 1e-9:
            return match.group(0)
        changed = True
        return f'{name}="{math.copysign(ELBOW_LIMIT_RAD, value):.8f}"'

    out: list[str] = []
    end = 0
    bounded = 0
    for block in _JOINT_BLOCK.finditer(urdf_text):
        text = block.group(0)
        name = _NAME.search(text)
        if name is None or name.group(1) != ELBOW_JOINT or "<limit" not in text:
            continue
        bounded += 1
        out.append(urdf_text[end:block.start()])
        out.append(_BOUND.sub(bound, text))
        end = block.end()
    if not bounded:
        raise ValueError(
            f"this description declares no {ELBOW_JOINT} with a <limit>, so UR's planning limit cannot be written "
            "into it"
        )
    if not changed:
        return urdf_text, False
    out.append(urdf_text[end:])
    return "".join(out), True
