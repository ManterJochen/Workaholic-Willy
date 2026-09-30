"""Which way a grasp closes about base Z, and the grasps that close along the axis a program names.

The owner's rule (2026-09-30), "choose, don't twist": a program may name the closing axis it wants, and only the grasps
that already close that way are taken. Each is turned half a turn about its approach where that puts its closing axis
the wanted way round: the same two contact faces, the jaws swapped. So the grasp that was judged is the grasp that is
executed, and ``both_faces`` judges the faces the jaws close on. ``GraspMotion(align_closing_to_base_x=True)`` turns a
chosen grasp about base Z after the fact, onto faces nobody chose; it is the simulator's aid for symmetric parts and is
refused beside a named axis.

The wanted axis is either a name ``Pose.tool_down`` takes (:data:`~src.geometry.pose.CLOSING_AXES`, read at each grasp's
position through the geometry's own rule, :func:`~src.geometry.pose.closing_axis_heading_deg`), or an orientation: a
quaternion (x, y, z, w) or a BASE ``Pose``, whose tool +X laid onto the base XY plane is the wanted heading. So
``Pose.tool_down(x, y, z, closing_axis="-y").quaternion_xyzw`` and ``"-y"`` ask for the same grasps. The one reader of
both, :func:`closing_axis_of` and :class:`ClosingAxis`, lives with the poses it reads, in :mod:`src.geometry.closing_axis`,
so that a config naming the axis is read without this package; it is handed on here.

A grasp is kept when its closing axis heads within :data:`CLOSING_AXIS_TOLERANCE_DEG` of the wanted heading, either way
round: a parallel jaw closes on the same two faces both ways. A closing axis within
:data:`NO_HEADING_WITHIN_DEG_OF_VERTICAL` of the vertical names no heading, and its grasp is left out. A ranking that
chooses along a named axis asks its generator for :data:`CANDIDATES_THE_AXIS_CHOOSES_AMONG` and keeps as many of those
that close along it as it keeps without one, so the pick loop and ``Scene.grasps`` choose among the same candidates.

A cell also names how its hand and camera naturally stand, ``robot.natural_closing_axis`` (the owner's decision,
2026-09-30), read the same way (:func:`natural_closing_axis_of`). Its camera grasps stay free, any closing direction and
any tilt: of a grasp's two equivalent wrist turns, half a turn about its approach apart, the one whose tool +X lies nearer
the natural direction at the grasp's place is taken (:func:`grasps_closing_toward`). None is left out and none tilted,
and a named axis, which keeps its own sign rule, takes precedence.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from typing import Final

import numpy as np

from src.geometry.closing_axis import (
    NO_HEADING_WITHIN_DEG_OF_VERTICAL,
    ClosingAxis,
    ClosingAxisLike,
    closing_axis_of,
    natural_closing_axis_of,
)
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint

__all__ = [
    "CANDIDATES_THE_AXIS_CHOOSES_AMONG",
    "CLOSING_AXIS_REFUSED",
    "CLOSING_AXIS_TOLERANCE_DEG",
    "NO_HEADING_WITHIN_DEG_OF_VERTICAL",
    "ClosingAxis",
    "ClosingAxisLike",
    "closing_along",
    "closing_axis_of",
    "closing_axis_refusal",
    "closing_axis_said",
    "grasps_closing_along",
    "grasps_closing_toward",
    "natural_closing_axis_of",
    "result_closing_along",
    "result_closing_toward",
    "same_closing_axis",
    "turned_nearer",
]

#: How far a grasp's closing axis may head off the wanted one and still count as closing along it, degrees, either way
#: round. The owner's figure (2026-09-30).
CLOSING_AXIS_TOLERANCE_DEG = 30.0

#: The key of ``GraspResult.telemetry`` under which a result whose every candidate closed off the wanted axis carries the
#: sentence that says so (:func:`closing_axis_refusal`).
CLOSING_AXIS_REFUSED = "closing_axis_refused"

#: How many candidates a ranking asks its generator for where a program names the closing axis, before it keeps the
#: best of them that close along it, as many as it keeps without an axis (the calculator's own cap, 12 by default): the
#: owner's figures (2026-09-30), so a pick and a ``Scene.grasps`` of the same part choose among the same candidates.
CANDIDATES_THE_AXIS_CHOOSES_AMONG: Final[int] = 36

#: How far apart two headings may lie and still be one, degrees: rounding, not a tolerance (:func:`same_closing_axis`).
_SAME_HEADING_DEG = 1e-6

#: How far a closing axis must reach toward the far side of the natural direction, as a cosine, before the grasp turned
#: half a turn counts as nearer it: rounding, not a tolerance (:func:`turned_nearer`). A closing axis straight across the
#: natural direction lies as near it either way round, and its grasp is left as it was ranked.
_ACROSS = 1e-9

#: Why a grasp was left out, as the sentence of an attempt counts them.
_OFF_THE_AXIS = "close further off it"
_NO_HEADING = f"close about an axis within {NO_HEADING_WITHIN_DEG_OF_VERTICAL:g} deg of the vertical, which names no heading"
_NOT_IN_BASE = "are not in BASE, where the axis is named"
_NO_DIRECTION = "stand on the base's vertical axis, where the name points nowhere"


def _gap_deg(heading_deg: float, wanted_deg: float) -> float:
    """How far ``heading_deg`` turns from ``wanted_deg`` about base Z, degrees in (-180, 180]."""
    gap = (heading_deg - wanted_deg) % 360.0
    return gap - 360.0 if gap > 180.0 else gap


def same_closing_axis(first: ClosingAxis, second: ClosingAxis, x_mm: float, y_mm: float) -> bool:
    """Whether ``first`` and ``second`` take the same grasps, turned the same way round, at (``x_mm``, ``y_mm``) in BASE:
    the same heading there. ``"-y"`` and an orientation whose tool +X heads along -y are one axis; ``"y"`` is another,
    whose grasps close the other way round. ``radial`` or ``tangential`` on the base's vertical axis names no heading,
    and is the same only as itself."""
    if first == second:
        return True
    try:
        return abs(_gap_deg(first.heading_at(x_mm, y_mm), second.heading_at(x_mm, y_mm))) <= _SAME_HEADING_DEG
    except ValueError:
        return False


def closing_along(
    wanted: ClosingAxis, position_mm: Sequence[float] | np.ndarray, closing_axis: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray | None, str]:
    """``closing_axis`` of a grasp at ``position_mm`` (BASE), signed so it heads along ``wanted``, or ``None`` and why the
    grasp is left out.

    Kept where its heading about base Z lies within :data:`CLOSING_AXIS_TOLERANCE_DEG` of the wanted heading either way
    round; the sign is the one whose heading lies within a quarter turn of it, so a grasp that closes the other way
    round comes back turned half a turn about its approach: the same two contact faces, the jaws swapped.
    """
    axis = np.asarray(closing_axis, dtype=np.float64).reshape(3)
    across = math.hypot(float(axis[0]), float(axis[1]))
    if math.degrees(math.atan2(across, abs(float(axis[2])))) < NO_HEADING_WITHIN_DEG_OF_VERTICAL:
        return None, _NO_HEADING
    position = np.asarray(position_mm, dtype=np.float64).reshape(3)
    try:
        wanted_deg = wanted.heading_at(float(position[0]), float(position[1]))
    except ValueError:
        return None, _NO_DIRECTION
    gap = _gap_deg(math.degrees(math.atan2(float(axis[1]), float(axis[0]))), wanted_deg)
    if abs(gap) > 90.0:
        axis, gap = -axis, _gap_deg(gap + 180.0, 0.0)
    if abs(gap) > CLOSING_AXIS_TOLERANCE_DEG:
        return None, _OFF_THE_AXIS
    return axis, ""


def grasps_closing_along(
    candidates: Sequence[GraspPoint], wanted: ClosingAxis,
) -> tuple[tuple[GraspPoint, ...], tuple[str, ...]]:
    """The candidates that close along ``wanted``, in their order, each signed along it (:func:`closing_along`), and
    why each of the others was left out. A candidate that is not in BASE is left out: the axis is named in BASE."""
    kept: list[GraspPoint] = []
    whys: list[str] = []
    for grasp in candidates:
        if grasp.frame is not GraspFrame.BASE:
            whys.append(_NOT_IN_BASE)
            continue
        signed, why = closing_along(wanted, grasp.position, grasp.axis)
        if signed is None:
            whys.append(why)
        elif float(np.dot(signed, grasp.axis)) < 0.0:
            kept.append(replace(grasp, axis=signed))
        else:
            kept.append(grasp)
    return tuple(kept), tuple(whys)


def closing_axis_said(wanted: ClosingAxis, whys: Sequence[str], total: int) -> str:
    """What the axis cost, in one sentence: how many of ``total`` candidates were left out and why, counted; ``""`` when
    none was."""
    if not whys:
        return ""
    counted = ", ".join(f"{count} {why}" for why, count in Counter(whys).items())
    if len(whys) >= total:
        return (f"closing_axis {wanted}: none of the {total} grasp candidate(s) closes within "
                f"{CLOSING_AXIS_TOLERANCE_DEG:g} deg of it, either way round ({counted})")
    return (f"closing_axis {wanted}: {len(whys)} of the {total} grasp candidate(s) left out, not closing within "
            f"{CLOSING_AXIS_TOLERANCE_DEG:g} deg of it ({counted})")


def result_closing_along(result: GraspResult, wanted: ClosingAxis) -> GraspResult:
    """``result`` with only the candidates that close along ``wanted``, each signed along it, best first as they were.

    Where none does, a result with no candidate and ``NO_VALID_GRASP``, its telemetry carrying the sentence that names
    the axis and the tolerance (:data:`CLOSING_AXIS_REFUSED`). A result with no candidate is returned as it is.
    """
    if not result.candidates:
        return result
    kept, whys = grasps_closing_along(result.candidates, wanted)
    if kept:
        unchanged = not whys and all(mine is theirs for mine, theirs in zip(kept, result.candidates))
        return result if unchanged else replace(result, candidates=kept, top_score=float(kept[0].score))
    return GraspResult(
        candidates=(),
        reasons=(GraspFailureReason.NO_VALID_GRASP,),
        telemetry={**result.telemetry, CLOSING_AXIS_REFUSED: closing_axis_said(wanted, whys, len(result.candidates))},
        depth_confidence=result.depth_confidence,
        mask_confidence=result.mask_confidence,
        top_score=0.0,
        metadata=result.metadata,
    )


def turned_nearer(
    natural: ClosingAxis, position_mm: Sequence[float] | np.ndarray, closing_axis: Sequence[float] | np.ndarray,
) -> bool:
    """Whether a grasp at ``position_mm`` (BASE) that closes along ``closing_axis`` lies nearer the natural direction
    there turned half a turn about its approach: whether its tool +X, reversed, points nearer it. ``False`` where both
    ways round lie as near (a closing axis straight across it), and where the natural direction names none there
    (``radial`` or ``tangential`` on the base's vertical axis)."""
    axis = np.asarray(closing_axis, dtype=np.float64).reshape(3)
    position = np.asarray(position_mm, dtype=np.float64).reshape(3)
    try:
        heading = math.radians(natural.heading_at(float(position[0]), float(position[1])))
    except ValueError:
        return False
    return float(axis[0]) * math.cos(heading) + float(axis[1]) * math.sin(heading) < -_ACROSS


def grasps_closing_toward(candidates: Sequence[GraspPoint], natural: ClosingAxis) -> tuple[GraspPoint, ...]:
    """Every candidate, in its order, turned half a turn about its approach where that puts its tool +X nearer the
    natural direction at its place (:func:`turned_nearer`): the same two contact faces, the jaws swapped. None is left
    out and none tilted; one that is not in BASE, where the natural orientation is named, is left as it is."""
    return tuple(replace(grasp, axis=-grasp.axis)
                 if grasp.frame is GraspFrame.BASE and turned_nearer(natural, grasp.position, grasp.axis) else grasp
                 for grasp in candidates)


def result_closing_toward(result: GraspResult, natural: ClosingAxis) -> GraspResult:
    """``result`` with every candidate the way round nearer ``natural`` (:func:`grasps_closing_toward`), in its order,
    with its scores, its reasons and its telemetry; ``result`` itself where none needed turning."""
    turned = grasps_closing_toward(result.candidates, natural)
    if all(mine is theirs for mine, theirs in zip(turned, result.candidates)):
        return result
    return replace(result, candidates=turned)


def closing_axis_refusal(result: object) -> str:
    """Why ``result`` has no candidate on account of the axis a program named, or ``""``: the sentence
    :func:`result_closing_along` left in its telemetry."""
    telemetry = getattr(result, "telemetry", None)
    said = telemetry.get(CLOSING_AXIS_REFUSED) if isinstance(telemetry, dict) else None
    return said if isinstance(said, str) else ""
