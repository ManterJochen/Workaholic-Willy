"""The planner's cushion band: poses the exact mesh guard accepts and the planner's padded spheres refuse.

Two authorities judge the arm against itself, with different geometry. The exact guard measures the meshes and demands
``safety.self_collision.min_distance_mm`` (10 mm) on every pair more than one DH frame apart, the wrist camera's parts
also against wrist_2. cuRobo judges a cover fit of spheres, which reaches past the meshes by design (on a UR10 the
forearm's 15 mm and wrist_2's 8 mm), and pads every link by half of ``planner_margin_mm`` on top, so a pair carries
the whole margin. On a tilted look the two disagree in one direction only: at the owner's LOOK[0] the meshes keep
19.0 mm on forearm|wrist_2 and the planner's spheres overlap by 1.2 mm, and over the whole wrist_1 x wrist_2 torus a
third of the cells are refused by the planner at 4 mm while the guard accepts them, and no cell the guard refuses is
cleared by the planner (research, 2026-09-30).

The owner, 2026-09-30: for the self pairs the exact guard judges, the exact guard decides. A configuration the exact
guard has accepted is not vetoed by the planner's self term for a pair of the planner's links that maps onto parts the
guard checks against one another (:meth:`ExactPairs.decides`). Everything else stays the planner's, unchanged: a pair
the guard does not check, the carried part (``attached_object``, which only the planner holds), the planner's world and
its joint bounds (:func:`admission_refusal`). The composed config, its hash and the 4 mm margin are untouched: cuRobo
still plans with them, which is what keeps its own paths away from the guard's 10 mm.

What is lifted, exactly: the ``planner_margin_mm`` padding and the spheres' own reach past the meshes, on pairs the
guard judges, and nothing more. The shoulder_link is not one of the guard's parts here (:data:`_ARM_PARTS`): the
guard's UR bundles hold no part for the robot's own base (a UR10's shoulder mesh starts 38.3 mm above the base plate),
and the planner's shoulder_link, whose spheres reach down to 18 mm and which carries cuRobo's stock 70 mm cushion, is
the only model of that base anywhere. Left to the guard, hand|shoulder_link admitted the Hand-E touching the base
(review of F1, 2026-09-30). And a pair padded by more than the margin carries a cushion the descriptor put there for
something else, so it stays the planner's whatever its links are. The owner's decision rested on forearm|wrist_2 at
LOOK[0] and LOOK[1]: the pairs it now leaves to the guard are the arm's links from the upper arm to wrist_3, the hand
and the wrist cameras, where the guard judges them, and the owner confirmed that scope after the fix round
(2026-09-30).

A planned move out of such a pose, or into one, takes a straight leg of at most :data:`BAND_LEG_MAX_DEG` per joint to
the nearest configuration both authorities clear (:func:`band_neighbours`), and cuRobo plans from there; a move that is
only ever a straight line (the generated view, its move back) never does. Before anything is sent, both authorities
judge the whole route, as every route. Where no such configuration lies within the cap, straight lines still run into
the pose and out of it, and a planned move out of it or into it is refused, saying so.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from src.contracts import chosen

from ._curobo_attach import ATTACHED_LINK_NAME
from ._curobo_body_links import HAND_LINK, WRIST_BODY_PREFIX
from ._hand_bundle import HAND_PARTS

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from ._curobo_pairs import PairOverlap
    from .curobo_client import RefusedSample

__all__ = [
    "BAND_LEG_MAX_DEG",
    "ExactPairs",
    "PoseScreen",
    "PoseVerdict",
    "admission_refusal",
    "band_neighbours",
    "band_sentence",
    "degrees_line",
    "guard_parts",
    "joints_between",
]

#: How far one escape or approach leg may turn any joint, in degrees (the owner, 2026-09-30). LOOK[0]'s nearest pose
#: both authorities clear is 4 to 6 degrees of wrist_1 away and LOOK[1]'s 19.5: the cap was 10 at first, which left
#: LOOK[1] to straight lines alone, and the owner raised it to 20 after the fix round.
BAND_LEG_MAX_DEG = 20.0

#: How many configurations one search for a clear neighbour asks the planner about: one check request
#: (``curobo_client.MAX_CHECK_CONFIGURATIONS``).
NEIGHBOUR_BUDGET = 1000

#: The finest the neighbour grid gets, in degrees, where few joints leave the budget room for more.
_FINEST_STEP_DEG = 0.5

#: How far the padding a pair carries may lie past ``planner_margin_mm`` and still be the margin's, in millimetres: the
#: report's two depths of a pair are one float64 subtraction apart, which rounds in its last bits and never by this much.
_PADDING_TOLERANCE_MM = 1e-3

#: The planner's arm links and the exact guard's parts of the same name (``{model}_collision_meshes.npz``). Not the
#: shoulder_link: its spheres and cuRobo's stock 70 mm cushion on it cover the robot's own base as well as the shoulder,
#: and no guard bundle holds a part for the base, so every pair with it stays the planner's (review of F1, 2026-09-30).
_ARM_PARTS: Mapping[str, tuple[str, ...]] = {
    "upper_arm_link": ("upper_arm",),
    "forearm_link": ("forearm",),
    "wrist_1_link": ("wrist_1",),
    "wrist_2_link": ("wrist_2",),
    "wrist_3_link": ("wrist_3",),
}


def guard_parts(link: str) -> tuple[str, ...]:
    """The exact guard's parts a planner link is, or ``()`` where the guard holds nothing that is that link.

    The arm links from the upper arm to wrist_3 are the parts of the same name; the hand link is the hand's three parts
    (``HAND_PARTS``); a wrist camera's link is its own part, named alike (``body_link.WristBody.guard_parts``). The
    shoulder_link, which is the planner's model of the base too, the carried part (``attached_object``), the coupling
    link and any name this does not know are none of the guard's, so their pairs stay the planner's.
    """
    if link in _ARM_PARTS:
        return _ARM_PARTS[link]
    if link == HAND_LINK:
        return tuple(HAND_PARTS)
    if link.startswith(WRIST_BODY_PREFIX) and len(link) > len(WRIST_BODY_PREFIX):
        return (link,)
    return ()


@dataclass(frozen=True)
class ExactPairs:
    """The exact mesh guard's own pair rule and distances, asked in the planner's link names.

    Built by the guard over the backend it runs (``SelfCollisionGuard.exact_pairs``), so the rule is that backend's
    and not a copy of it. ``checks(part_a, part_b)`` is whether the guard judges the two parts against one another,
    ``frames`` the DH frame of every part it holds, ``distance(joints, part_a, part_b)`` the exact distance of two parts
    at a configuration in millimetres (``None`` where it cannot be placed), and ``min_distance_mm`` what it demands.
    """

    checks: Callable[[str, str], bool]
    frames: Mapping[str, int]
    distance: "Callable[[Sequence[float], str, str], float | None]"
    min_distance_mm: float

    def held(self, link: str) -> tuple[str, ...]:
        """The guard's parts ``link`` is, where the guard holds every one of them; else ``()``."""
        parts = guard_parts(link)
        return parts if parts and all(part in self.frames for part in parts) else ()

    def decides(self, link_a: str, link_b: str) -> bool:
        """Whether the exact guard judges every part of ``link_a`` against every part of ``link_b``."""
        parts_a, parts_b = self.held(link_a), self.held(link_b)
        return bool(parts_a) and bool(parts_b) and all(
            self.checks(part_a, part_b) for part_a in parts_a for part_b in parts_b)

    def frame(self, link: str) -> "int | None":
        """The DH frame ``link``'s parts hang from, or ``None`` where the guard holds none or they hang from two."""
        frames = {int(self.frames[part]) for part in self.held(link)}
        return frames.pop() if len(frames) == 1 else None

    def distance_mm(self, joints: Sequence[float], link_a: str, link_b: str) -> "float | None":
        """The exact distance between two planner links at ``joints``: the nearest of their parts, or ``None``."""
        parts_a, parts_b = self.held(link_a), self.held(link_b)
        found = [self.distance(joints, part_a, part_b) for part_a in parts_a for part_b in parts_b]
        known = [value for value in found if value is not None]
        return min(known) if known and len(known) == len(found) else None


def admission_refusal(sample: "RefusedSample", exact: ExactPairs, *, margin_mm: float) -> "str | None":
    """Why the planner's refusal of a configuration stands, or ``None`` where the exact guard decides every pair of it.

    Only the self term is ever left to the exact guard, and only where every pair of links the planner found there is a
    pair the guard judges and carries no padding beyond ``margin_mm``, the cell's ``planner_margin_mm``: the joint
    bounds, the world and the carried part are the planner's alone and refuse as they always did, and so does a self
    collision the planner could not name, and a pair padded for something else. The padding of a pair is read off the
    report: its depth with the padding less its depth without. The caller has to have had the exact guard accept this
    very configuration; this reads the planner's report and nothing else.
    """
    if not sample.bound_ok:
        return "a joint is outside the planner's bounds, which the planner alone judges"
    if not sample.world_ok:
        return "it reaches into the planner's world, or nearer it than the clearance asked, which the planner judges"
    if sample.self_ok:
        return "the planner refused it with no self collision"
    if not chosen(sample.pairs):
        return "the planner did not name the pairs of links it found, so none can be left to the exact guard"
    if not sample.pairs:
        return "the planner found a self collision and named no pair of links for it"
    for pair in sample.pairs:
        if ATTACHED_LINK_NAME in (pair.link_a, pair.link_b):
            return f"{pair.link_a} and {pair.link_b} overlap, and the carried part is judged by the planner alone"
        if not exact.decides(pair.link_a, pair.link_b):
            return f"{pair.link_a} and {pair.link_b} overlap, a pair the exact guard does not judge"
        padding = float(pair.depth_mm) - float(pair.unpadded_depth_mm)
        if not padding <= float(margin_mm) + _PADDING_TOLERANCE_MM:
            return (f"{pair.link_a} and {pair.link_b} overlap with {padding:g} mm of the planner's padding, more than "
                    f"the {float(margin_mm):g} mm margin, which only the planner judges")
    return None


def joints_between(pairs: "Sequence[PairOverlap]", exact: ExactPairs, *, joints: int = 6) -> tuple[int, ...]:
    """The joints that turn one link of each pair against the other, 0-based from the base; every joint where unknown.

    Frame ``f`` hangs after joint ``f`` of the DH chain, so two parts on frames ``a < b`` move against one another by
    joints ``a + 1`` to ``b``, which are indices ``a`` to ``b - 1``: forearm (3) and wrist_2 (5) by wrist_1 and wrist_2.
    """
    moved: set[int] = set()
    for pair in pairs:
        frame_a, frame_b = exact.frame(pair.link_a), exact.frame(pair.link_b)
        if frame_a is None or frame_b is None:
            return tuple(range(joints))
        low, high = sorted((frame_a, frame_b))
        moved.update(range(max(0, low), min(joints, high)))
    return tuple(sorted(moved)) if moved else tuple(range(joints))


def band_neighbours(
    config: Sequence[float],
    joints: Sequence[int],
    radii_mm: Sequence[float],
    *,
    max_deg: float = BAND_LEG_MAX_DEG,
    budget: int = NEIGHBOUR_BUDGET,
) -> list[tuple[float, ...]]:
    """Configurations near ``config``, turned on ``joints`` alone and at most ``max_deg`` per joint, nearest first.

    A grid as fine as ``budget`` configurations allow, one planner check request, never finer than half a degree; a
    wider reach widens the step rather than the request. At the 20 degree cap: 0.5 degrees on one joint (80
    configurations), 1.33 on two (960), 5 on three (728), 10 on four (624), 20 on five or six (242, 728). Nearest by the
    path gate's own measure of motion, the sum of ``|dq_j| * radii_mm[j]``. ``config`` itself is not among them.
    """
    base = tuple(float(value) for value in config)
    turned = tuple(sorted({int(j) for j in joints}))
    if not turned or len(radii_mm) != len(base):
        return []
    per_side = max(1, min(int(round(max_deg / _FINEST_STEP_DEG)),
                          int((budget ** (1.0 / len(turned)) - 1.0) / 2.0 + 1e-9)))
    step = math.radians(max_deg) / per_side
    offsets = [index * step for index in range(-per_side, per_side + 1)]
    out: list[tuple[float, ...]] = []
    for combo in itertools.product(offsets, repeat=len(turned)):
        if all(value == 0.0 for value in combo):
            continue
        values = list(base)
        for joint, offset in zip(turned, combo):
            values[joint] += offset
        out.append(tuple(values))

    def travel(values: tuple[float, ...]) -> float:
        return sum(abs(a - b) * float(radius) for a, b, radius in zip(values, base, radii_mm))

    out.sort(key=lambda values: (travel(values), values))
    return out


def band_sentence(sample: "RefusedSample", exact: ExactPairs, joints: Sequence[float]) -> str:
    """What the two authorities found at ``joints`` on each pair the planner named: its depth and the exact distance."""
    said: list[str] = []
    pairs = sample.pairs if chosen(sample.pairs) else ()
    for pair in pairs[:3]:
        exact_mm = exact.distance_mm(joints, pair.link_a, pair.link_b)
        meshes = (f"the exact meshes keep {exact_mm:.1f} mm" if exact_mm is not None
                  else "the exact guard holds no distance for it")
        said.append(f"{pair.render()}, and {meshes}")
    if len(pairs) > 3:
        said.append(f"and {len(pairs) - 3} more pair(s)")
    return "; ".join(said) + f" (the guard demands {exact.min_distance_mm:g} mm)"


class PoseVerdict(StrEnum):
    """What the two authorities say about one configuration an arm is sent to, before it is sent there."""

    #: Both clear it.
    CLEAR = "clear"
    #: The exact guard accepts it and the planner's padded spheres refuse it on pairs the guard judges: straight lines
    #: run into it and out of it, and a planned move out of it or into it takes a straight leg of at most
    #: :data:`BAND_LEG_MAX_DEG` per joint to the nearest pose both clear where there is one, and is refused where not.
    BAND = "band"
    #: The exact guard refuses it: nothing is ever sent there.
    GUARD_REFUSED = "guard_refused"
    #: The planner refuses it for something only the planner judges (its world, its bounds, the carried part, a pair
    #: the guard does not check): nothing is sent there while it does.
    PLANNER_REFUSED = "planner_refused"
    #: It could not be asked, and the reason says why.
    UNSCREENED = "unscreened"


_HEADS: Mapping[PoseVerdict, str] = {
    PoseVerdict.CLEAR: "clear",
    PoseVerdict.BAND: "in the planner's cushion band",
    PoseVerdict.GUARD_REFUSED: "refused by the exact guard",
    PoseVerdict.PLANNER_REFUSED: "refused by the planner",
    PoseVerdict.UNSCREENED: "not screened",
}


@dataclass(frozen=True)
class PoseScreen:
    """One configuration, screened by both authorities before any move goes there: the verdict and why.

    ``nearby`` is the nearest configuration both clear with every joint at most :data:`BAND_LEG_MAX_DEG` from it,
    radians, where one was looked for and found. For a band pose it is where a planned move's leg goes on its own, so
    nothing is re-taught; with none, only straight lines run there, and a pose planned moves have to reach or leave is
    re-taught by hand where both clear it. For a refused pose, an ``ERROR`` line, it is where to re-teach it; with none,
    it is re-taught by hand elsewhere. A re-taught pose is screened again.
    """

    verdict: PoseVerdict
    detail: str
    nearby: "tuple[float, ...] | None" = None
    #: True where the planner could not be asked at all (it did not start): a caller screening many poses asks the exact
    #: guard alone for the rest rather than waiting on a planner that will not come.
    planner_unavailable: bool = False

    @property
    def is_error(self) -> bool:
        """Whether no move will go there: the exact guard refuses it, or the planner does for what only it judges."""
        return self.verdict in (PoseVerdict.GUARD_REFUSED, PoseVerdict.PLANNER_REFUSED)

    def render(self) -> str:
        """The verdict, why, and a pose nearby both clear: ``in the planner's cushion band: ... Nearby, both clear: ...``."""
        text = f"{_HEADS[self.verdict]}: {self.detail}"
        if self.nearby is not None:
            text += f" Nearby, both clear: {degrees_line(self.nearby)}."
        return text

    def line(self, label: str) -> str:
        """One line for the pose called ``label``: ``ERROR`` first where no move goes there, then :meth:`render`."""
        return f"{'ERROR ' if self.is_error else ''}{label}: {self.render()}"


def degrees_line(joints: Sequence[float]) -> str:
    """Joints in radians as a person reads them: ``(-42.5, -67.9, ...) deg``, a tenth of a degree, never ``-0.0``."""
    said = [f"{math.degrees(float(value)):.1f}" for value in joints]
    return "(" + ", ".join("0.0" if value == "-0.0" else value for value in said) + ") deg"
