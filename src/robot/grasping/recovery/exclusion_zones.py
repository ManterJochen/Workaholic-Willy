"""Which parts a rescan skips: the memory behind ``next_target`` (owner, 2026-09-29).

``next_target`` means "rescan, skipping the part that just failed". It never switches to another object: a zone
applies only to parts of the label it was made for, which is the label the operator asked for. A zone is a
column over the table:

* its centre is the failed part's BASE XY centre;
* its radius is max(:data:`EXCLUSION_RADIUS_FLOOR_MM`, half the part's footprint diagonal);
* it lasts for the pick it was made in plus the next :data:`EXCLUSION_EXTRA_PICKS` picks.

A position, not an index or a detector's label, is what survives a rescan, a re-look and a second camera.

The service holds one :class:`ExclusionZones` for a campaign and calls :meth:`ExclusionZones.start_pick` at the
start of every pick. The ``NEXT_TARGET`` step of the recovery loop calls :meth:`ExclusionZones.exclude`. The
pick loop drops every segmentation for which :meth:`ExclusionZones.excludes` is true, next to its label gate.
When :meth:`ExclusionZones.only_excluded_remain` is true, the pick stops with
:meth:`ExclusionZones.only_excluded_sentence` rather than picking a part it was told to skip.

One campaign drives one instance from one thread. Nothing here moves the arm or imports robot code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "EXCLUSION_EXTRA_PICKS",
    "EXCLUSION_RADIUS_FLOOR_MM",
    "ExclusionZone",
    "ExclusionZones",
    "exclusion_radius_mm",
    "footprint_diagonal_mm",
]

#: The smallest exclusion radius, in mm: a small part's zone still covers the jitter of its centre between scans.
EXCLUSION_RADIUS_FLOOR_MM = 30.0
#: A zone lasts for the pick it was made in plus this many picks.
EXCLUSION_EXTRA_PICKS = 2


def _xy(value: Sequence[float], name: str) -> tuple[float, float]:
    out = tuple(float(c) for c in value)
    if len(out) not in (2, 3) or not all(math.isfinite(c) for c in out):
        raise ValueError(f"{name} must be two or three finite numbers in BASE mm, got {value!r}")
    return (out[0], out[1])


def _label(value: object) -> str:
    label = str(value).strip().casefold()
    if not label:
        raise ValueError("an exclusion zone needs the label it is for; got an empty one")
    return label


def exclusion_radius_mm(footprint_diagonal_mm: Optional[float]) -> float:
    """max(:data:`EXCLUSION_RADIUS_FLOOR_MM`, half the footprint diagonal); the floor when the diagonal is unknown."""

    if footprint_diagonal_mm is None or not math.isfinite(float(footprint_diagonal_mm)):
        return EXCLUSION_RADIUS_FLOOR_MM
    return max(EXCLUSION_RADIUS_FLOOR_MM, 0.5 * float(footprint_diagonal_mm))


def footprint_diagonal_mm(points_mm: ArrayLike) -> float:
    """The diagonal of the part's footprint in BASE XY.

    It is the smaller of two boxes' diagonals: the box along the part's own principal axes, and the box along
    BASE X and Y. A box at any angle has a diagonal at least as long as the part's own, so the smaller one is the
    closer estimate. A long part lying at an angle gets its own diagonal, not the larger box around it. A square
    part, whose principal axes are arbitrary, still gets its side-by-side diagonal. Returns 0.0 for no points.
    """

    points = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 2:
        return 0.0
    xy = points[:, :2] - points[:, :2].mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov(xy.T))
    diagonals = []
    for axes in (vectors, np.eye(2)):
        along = xy @ axes
        extent = along.max(axis=0) - along.min(axis=0)
        diagonals.append(math.hypot(float(extent[0]), float(extent[1])))
    return float(min(diagonals))


@dataclass(frozen=True, slots=True)
class ExclusionZone:
    """One skipped part: a column over the table for one label, alive from ``added_pick`` through ``last_pick``."""

    label: str
    centre_xy_mm: tuple[float, float]
    radius_mm: float
    added_pick: int
    last_pick: int

    def contains(self, centre_mm: Sequence[float]) -> bool:
        x, y = _xy(centre_mm, "centre_mm")
        return math.hypot(x - self.centre_xy_mm[0], y - self.centre_xy_mm[1]) <= self.radius_mm


class ExclusionZones:
    """The exclusion zones of one campaign."""

    def __init__(self, *, extra_picks: int = EXCLUSION_EXTRA_PICKS) -> None:
        if isinstance(extra_picks, bool) or not isinstance(extra_picks, int) or extra_picks < 0:
            raise ValueError(f"extra_picks must be a non-negative whole number, got {extra_picks!r}")
        self._extra_picks = extra_picks
        self._pick_index = 0
        self._zones: list[ExclusionZone] = []

    @property
    def pick_index(self) -> int:
        """How many picks have started; 0 before the first."""

        return self._pick_index

    def start_pick(self) -> int:
        """A new pick starts; zones whose lifetime ended are dropped. Returns the new pick's index."""

        self._pick_index += 1
        self._zones = [z for z in self._zones if z.last_pick >= self._pick_index]
        return self._pick_index

    def exclude(
        self,
        *,
        label: str,
        centre_mm: Sequence[float],
        footprint_diagonal_mm: Optional[float] = None,
    ) -> ExclusionZone:
        """Skip the part of ``label`` standing at ``centre_mm`` for this pick and the next ``extra_picks``."""

        zone = ExclusionZone(
            label=_label(label),
            centre_xy_mm=_xy(centre_mm, "centre_mm"),
            radius_mm=exclusion_radius_mm(footprint_diagonal_mm),
            added_pick=self._pick_index,
            last_pick=self._pick_index + self._extra_picks,
        )
        self._zones.append(zone)
        return zone

    def zones(self, label: Optional[str] = None) -> tuple[ExclusionZone, ...]:
        if label is None:
            return tuple(self._zones)
        wanted = _label(label)
        return tuple(z for z in self._zones if z.label == wanted)

    def excludes(self, *, label: str, centre_mm: Sequence[float]) -> bool:
        """Whether a part of ``label`` standing at ``centre_mm`` is to be skipped now."""

        return any(zone.contains(centre_mm) for zone in self.zones(label))

    def remaining(self, *, label: str, centres_mm: Sequence[Sequence[float]]) -> tuple[int, ...]:
        """The indices of the parts of ``label`` (by centre) that are not skipped."""

        zones = self.zones(label)
        return tuple(i for i, centre in enumerate(centres_mm) if not any(z.contains(centre) for z in zones))

    def only_excluded_remain(self, *, label: str, centres_mm: Sequence[Sequence[float]]) -> bool:
        """True when parts of ``label`` were seen and every one of them is skipped: the stop condition."""

        return len(centres_mm) > 0 and not self.remaining(label=label, centres_mm=centres_mm)

    def only_excluded_sentence(self, *, label: str, count: int) -> str:
        """The stop message for :meth:`only_excluded_remain`."""

        return (f"Every {str(label).strip()!r} the camera sees ({count}) failed in one of the last "
                f"{self._extra_picks + 1} picks and is being skipped, so the pick stops here rather than trying "
                "one of them again or another object.")
