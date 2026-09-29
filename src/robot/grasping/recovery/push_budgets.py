"""How many pushes are left: 1 per part, 2 per pick, 5 per campaign (owner, 2026-09-29).

The service holds one :class:`PushBudgets` for a campaign. It calls :meth:`PushBudgets.start_pick` at the start
of every pick. The pick attempt calls :meth:`PushBudgets.check` before it plans a push, and
:meth:`PushBudgets.record` as soon as a push has commanded any motion, whether or not the push finished. A spent
budget means no more pushes, never a stop: the pick falls through to its next recovery action and the campaign
goes on.

No part identity survives a rescan, but a position does. So a part is recognised by where it stands in BASE:
a part whose XY centre lies within :data:`SAME_PART_RADIUS_MM` plus the push's length of where a pushed part
stood is that part. The push's length is the distance from that centre to its predicted landing. A push that
glanced off, turned or tipped the part moved it about that far, in whatever direction, so the disc covers the
predicted landing and every askew one. Height is ignored, since a part is pushed over the table. A part that
rolled farther is taken for a new one; the pick and campaign budgets still bound how often that can happen.

Key :meth:`PushBudgets.check` and :meth:`PushBudgets.record` on the same centre estimate of the part (the
pick attempt's own), and pass ``landing_mm`` as that centre plus the plan's ``push_distance_mm`` along its
``direction``. Two estimates of one part can differ by a few mm, and the 30 mm radius absorbs that, but only
if both calls use one of them.

One campaign drives one instance from one thread. Nothing here moves the arm, asks a person or imports robot
code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

__all__ = [
    "BUDGET_ALLOWED",
    "BUDGET_CAMPAIGN_SPENT",
    "BUDGET_PART_SPENT",
    "BUDGET_PICK_SPENT",
    "PUSHES_PER_CAMPAIGN",
    "PUSHES_PER_PART",
    "PUSHES_PER_PICK",
    "SAME_PART_RADIUS_MM",
    "PushBudgetVerdict",
    "PushBudgets",
    "PushRecord",
]

#: Pushes one part may take in a campaign.
PUSHES_PER_PART = 1
#: Pushes one pick may take, over all its recovery attempts.
PUSHES_PER_PICK = 2
#: Pushes one campaign may take.
PUSHES_PER_CAMPAIGN = 5
#: A part whose BASE XY centre lies within this plus the push's length of a pushed part's centre is the same
#: part. It matches the floor of the next_target exclusion radius.
SAME_PART_RADIUS_MM = 30.0

BUDGET_ALLOWED = "allowed"
BUDGET_PART_SPENT = "part_already_pushed"
BUDGET_PICK_SPENT = "pick_budget_spent"
BUDGET_CAMPAIGN_SPENT = "campaign_budget_spent"

Vec3 = tuple[float, float, float]


def _centre(value: Sequence[float], name: str) -> Vec3:
    out = tuple(float(c) for c in value)
    if len(out) != 3 or not all(math.isfinite(c) for c in out):
        raise ValueError(f"{name} must be three finite numbers in BASE mm, got {value!r}")
    return (out[0], out[1], out[2])


def _count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative whole number, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class PushRecord:
    """One push that commanded motion: the part's centre before it, where it was predicted to land, and the pick."""

    centre_mm: Vec3
    landing_mm: Optional[Vec3]
    pick_index: int

    @property
    def push_length_mm(self) -> float:
        """How far the part was predicted to move, in BASE XY; 0 when no landing was given."""

        if self.landing_mm is None:
            return 0.0
        return math.hypot(self.landing_mm[0] - self.centre_mm[0], self.landing_mm[1] - self.centre_mm[1])


@dataclass(frozen=True, slots=True)
class PushBudgetVerdict:
    """``allowed`` with the code that decided it (``BUDGET_*``) and one sentence for the report."""

    allowed: bool
    code: str
    sentence: str


class PushBudgets:
    """The push budgets of one campaign."""

    def __init__(
        self,
        *,
        per_part: int = PUSHES_PER_PART,
        per_pick: int = PUSHES_PER_PICK,
        per_campaign: int = PUSHES_PER_CAMPAIGN,
        same_part_radius_mm: float = SAME_PART_RADIUS_MM,
    ) -> None:
        self._per_part = _count(per_part, "per_part")
        self._per_pick = _count(per_pick, "per_pick")
        self._per_campaign = _count(per_campaign, "per_campaign")
        radius = float(same_part_radius_mm)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError(f"same_part_radius_mm must be a positive number of mm, got {same_part_radius_mm!r}")
        self._radius = radius
        self._pick_index = 0
        self._records: list[PushRecord] = []

    @property
    def pick_index(self) -> int:
        """How many picks have started; 0 before the first."""

        return self._pick_index

    @property
    def records(self) -> tuple[PushRecord, ...]:
        return tuple(self._records)

    @property
    def pushes_this_pick(self) -> int:
        return sum(1 for r in self._records if r.pick_index == self._pick_index)

    @property
    def pushes_this_campaign(self) -> int:
        return len(self._records)

    def start_pick(self) -> int:
        """A new pick starts: its own two pushes are available again. Returns the new pick's index."""

        self._pick_index += 1
        return self._pick_index

    def pushes_of(self, centre_mm: Sequence[float]) -> int:
        """How many recorded pushes were of the part standing at ``centre_mm``: within the match radius plus the
        push's length of where a pushed part stood."""

        x, y, _ = _centre(centre_mm, "centre_mm")
        return sum(
            1 for record in self._records
            if math.hypot(x - record.centre_mm[0], y - record.centre_mm[1]) <= self._radius + record.push_length_mm
        )

    def check(self, centre_mm: Sequence[float]) -> PushBudgetVerdict:
        """Whether the part standing at ``centre_mm`` (BASE mm) may be pushed now."""

        centre = _centre(centre_mm, "centre_mm")
        if self.pushes_this_campaign >= self._per_campaign:
            return PushBudgetVerdict(False, BUDGET_CAMPAIGN_SPENT,
                                     f"This campaign has used its {self._per_campaign} pushes, so no more parts are "
                                     "pushed; the campaign goes on without them.")
        if self.pushes_this_pick >= self._per_pick:
            return PushBudgetVerdict(False, BUDGET_PICK_SPENT,
                                     f"This pick has used its {self._per_pick} pushes, so this part is not pushed.")
        if self.pushes_of(centre) >= self._per_part:
            return PushBudgetVerdict(False, BUDGET_PART_SPENT,
                                     f"This part was pushed {self.pushes_of(centre)} time(s) already and may be "
                                     f"pushed {self._per_part}, so it is not pushed again.")
        return PushBudgetVerdict(True, BUDGET_ALLOWED, "A push is within budget.")

    def record(self, *, centre_mm: Sequence[float], landing_mm: Optional[Sequence[float]] = None) -> PushRecord:
        """Count one push that commanded motion, whether or not it finished. ``centre_mm`` is the same estimate
        :meth:`check` was given; ``landing_mm`` is where the push was to take it."""

        record = PushRecord(
            centre_mm=_centre(centre_mm, "centre_mm"),
            landing_mm=None if landing_mm is None else _centre(landing_mm, "landing_mm"),
            pick_index=self._pick_index,
        )
        self._records.append(record)
        return record
