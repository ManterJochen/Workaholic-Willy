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

A task keeps out of its picks the area it places into (the owner's Q4, 2026-09-30): the bin its camera found, for
both scopes, and a circle about a taught place's drop for "until empty", so it never picks the bin, a part it already
put in it, or a part it set down. That is an :class:`ExclusionRegion`, laid with
:meth:`ExclusionZones.keep_out_region`. Unlike a zone, a region:

* is a turned rectangle or a circle about BASE Z, a column over the table;
* applies to every label where it names none, the empty label of an unlabelled detector included;
* lasts every pick until :meth:`ExclusionZones.forget_regions`, which the task calls as it ends, or until
  :meth:`ExclusionZones.forget_region` forgets it alone (one place of a sort checked again replaces only its own).

The pick loop asks :meth:`ExclusionZones.applies` whether anything keeps a segmentation's label out before it reads
where that segmentation stands, so a region keeps out an unlabelled part as a zone keeps out its own label's. It also
asks :meth:`ExclusionZones.kept_out_by_a_region` of every part it skips, so a pick that saw only skipped parts says
whether its task kept every one of them out (a look with nothing left for the task to pick) or a zone skipped one a
pick failed on (a part still to be picked).

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
    "ExclusionRegion",
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


def _label_read(value: object) -> str:
    """A segmentation's label as the zones compare labels: trimmed and case-folded; ``""`` for none."""
    return str(value or "").strip().casefold()


def _positive(value: object, name: str) -> float:
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number of mm, got {value!r}")
    return number


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


@dataclass(frozen=True, slots=True)
class ExclusionRegion:
    """An area of the table a task keeps out of its picks for as long as it runs: a column over a turned rectangle or a
    circle about BASE Z.

    ``centre_xy_mm`` is its middle in BASE XY. A circle has a ``radius_mm``; a turned rectangle has ``size_mm``, its
    full sizes along its own axes, and ``yaw_rad``, how far those axes turn from BASE X about Z. ``label`` is the label
    it keeps out, ``None`` for every label, the empty one of an unlabelled detector included. ``reason`` says what it
    is, in the words a pick's stop sentence uses ("the blue bin it places into"). Build one with :meth:`circle` or
    :meth:`rectangle`; a region that names no shape, both, or a size that is not a positive finite number is refused
    with ``ValueError``.
    """

    centre_xy_mm: tuple[float, float]
    radius_mm: float | None = None
    size_mm: tuple[float, float] | None = None
    yaw_rad: float = 0.0
    label: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "centre_xy_mm", _xy(self.centre_xy_mm, "centre_xy_mm"))
        if (self.radius_mm is None) == (self.size_mm is None):
            raise ValueError("an exclusion region is a circle (radius_mm) or a turned rectangle (size_mm), one of the two")
        if self.radius_mm is not None:
            object.__setattr__(self, "radius_mm", _positive(self.radius_mm, "radius_mm"))
        if self.size_mm is not None:
            sizes = tuple(self.size_mm)
            if len(sizes) != 2:
                raise ValueError(f"size_mm is the rectangle's two full sizes, got {self.size_mm!r}")
            object.__setattr__(self, "size_mm", (_positive(sizes[0], "size_mm"), _positive(sizes[1], "size_mm")))
        yaw = float(self.yaw_rad)
        if not math.isfinite(yaw):
            raise ValueError(f"yaw_rad must be finite, got {self.yaw_rad!r}")
        object.__setattr__(self, "yaw_rad", yaw)
        if self.label is not None:
            object.__setattr__(self, "label", _label(self.label))

    @classmethod
    def circle(cls, centre_mm: Sequence[float], radius_mm: float, *, label: Optional[str] = None,
               reason: str = "") -> "ExclusionRegion":
        """A circle of ``radius_mm`` about ``centre_mm`` (BASE, its Z ignored)."""
        return cls(centre_xy_mm=_xy(centre_mm, "centre_mm"), radius_mm=radius_mm, label=label, reason=reason)

    @classmethod
    def rectangle(cls, centre_mm: Sequence[float], size_mm: Sequence[float], *, yaw_rad: float = 0.0,
                  label: Optional[str] = None, reason: str = "") -> "ExclusionRegion":
        """A rectangle of full sizes ``size_mm`` about ``centre_mm``, its first side turned ``yaw_rad`` from BASE X."""
        sizes = tuple(float(v) for v in size_mm)
        if len(sizes) != 2:
            raise ValueError(f"size_mm is the rectangle's two full sizes, got {size_mm!r}")
        return cls(centre_xy_mm=_xy(centre_mm, "centre_mm"), size_mm=(sizes[0], sizes[1]), yaw_rad=yaw_rad,
                   label=label, reason=reason)

    def applies(self, label: object) -> bool:
        """Whether this region keeps out a part called ``label``: every label where it names none."""
        return self.label is None or _label_read(label) == self.label

    def contains(self, centre_mm: Sequence[float]) -> bool:
        """Whether a part whose centre stands at ``centre_mm`` stands in this region; its height does not matter."""
        x, y = _xy(centre_mm, "centre_mm")
        dx, dy = x - self.centre_xy_mm[0], y - self.centre_xy_mm[1]
        if self.radius_mm is not None:
            return math.hypot(dx, dy) <= self.radius_mm
        assert self.size_mm is not None  # one of the two, checked at construction
        c, s = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        along, across = c * dx + s * dy, -s * dx + c * dy
        return abs(along) <= self.size_mm[0] / 2.0 and abs(across) <= self.size_mm[1] / 2.0

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """Describe this to a person, as one line of ASCII, no trailing newline."""
        if self.radius_mm is not None:
            shape = f"a circle of {self.radius_mm:.0f} mm"
        else:
            assert self.size_mm is not None
            shape = (f"a {self.size_mm[0]:.0f} x {self.size_mm[1]:.0f} mm rectangle turned "
                     f"{math.degrees(self.yaw_rad):.0f} deg")
        whom = "every label" if self.label is None else repr(self.label)
        what = f"{self.reason}: " if self.reason else ""
        line = (f"{what}{shape} about ({self.centre_xy_mm[0]:.0f}, {self.centre_xy_mm[1]:.0f}) mm, kept out for "
                f"{whom}")
        return line.encode("ascii", "backslashreplace").decode("ascii")

    def to_dict(self) -> dict[str, object]:
        """Plain data, ``json.dumps`` safe."""
        return {
            "shape": "circle" if self.radius_mm is not None else "rectangle",
            "centre_xy_mm": [self.centre_xy_mm[0], self.centre_xy_mm[1]],
            "radius_mm": self.radius_mm,
            "size_mm": None if self.size_mm is None else [self.size_mm[0], self.size_mm[1]],
            "yaw_deg": math.degrees(self.yaw_rad),
            "label": self.label,
            "reason": self.reason,
        }


class ExclusionZones:
    """The exclusion zones of one campaign, and the regions its task keeps out (:class:`ExclusionRegion`)."""

    def __init__(self, *, extra_picks: int = EXCLUSION_EXTRA_PICKS) -> None:
        if isinstance(extra_picks, bool) or not isinstance(extra_picks, int) or extra_picks < 0:
            raise ValueError(f"extra_picks must be a non-negative whole number, got {extra_picks!r}")
        self._extra_picks = extra_picks
        self._pick_index = 0
        self._zones: list[ExclusionZone] = []
        self._regions: list[ExclusionRegion] = []

    @property
    def pick_index(self) -> int:
        """How many picks have started; 0 before the first."""

        return self._pick_index

    def start_pick(self) -> int:
        """A new pick starts; zones whose lifetime ended are dropped, regions are kept. Returns the new pick's index."""

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

    def keep_out_region(self, region: ExclusionRegion) -> ExclusionRegion:
        """Keep ``region`` out of every pick from the next one on, until :meth:`forget_regions` (or until
        :meth:`forget_region` forgets it alone). Returns ``region``, the object to hand :meth:`forget_region`."""

        if not isinstance(region, ExclusionRegion):
            raise TypeError(f"keep_out_region takes an ExclusionRegion, not {type(region).__name__}")
        self._regions.append(region)
        return region

    def forget_regions(self) -> int:
        """Forget every region kept out (a task that ends); the zones stay. Returns how many were forgotten."""

        forgotten = len(self._regions)
        self._regions = []
        return forgotten

    def forget_region(self, region: ExclusionRegion) -> bool:
        """Forget ``region`` alone: the one kept out (the very object :meth:`keep_out_region` returned), else the first
        one equal to it; every other region and every zone stays. Returns whether one was forgotten.

        What a check of one place of a sort calls (the sorting map's F4, 2026-10-09): its bin, followed or found again,
        replaces its own region, and every other place's bin and every circle about a taught drop stays kept out, so no
        pick takes a sorted part back out of another rule's bin. :meth:`forget_regions` forgets them all, as a task ends.
        """

        if not isinstance(region, ExclusionRegion):
            raise TypeError(f"forget_region takes an ExclusionRegion, not {type(region).__name__}")
        for index, kept in enumerate(self._regions):
            if kept is region:
                del self._regions[index]
                return True
        for index, kept in enumerate(self._regions):
            if kept == region:
                del self._regions[index]
                return True
        return False

    def regions(self, label: Optional[str] = None) -> tuple[ExclusionRegion, ...]:
        """The regions kept out: every one, or those that apply to a part called ``label`` (``""`` included)."""

        if label is None:
            return tuple(self._regions)
        return tuple(region for region in self._regions if region.applies(label))

    def _zones_for(self, label: object) -> tuple[ExclusionZone, ...]:
        """The zones of ``label``: none for the empty label, which no ``next_target`` zone is made for."""

        wanted = _label_read(label)
        return tuple(z for z in self._zones if z.label == wanted) if wanted else ()

    def applies(self, label: object) -> bool:
        """Whether anything keeps a part called ``label`` out: a zone of its label, or a region that applies to it. The
        pick loop asks this before it reads where a part stands; an empty label is asked about too."""

        return bool(self._zones_for(label)) or any(region.applies(label) for region in self._regions)

    def _keeps_out(self, label: object, centre_mm: Sequence[float]) -> bool:
        return (any(zone.contains(centre_mm) for zone in self._zones_for(label))
                or any(region.contains(centre_mm) for region in self._regions if region.applies(label)))

    def excludes(self, *, label: str, centre_mm: Sequence[float]) -> bool:
        """Whether a part of ``label`` standing at ``centre_mm`` is to be skipped now: a zone of its label holds it, or
        a region that applies to it does."""

        return self._keeps_out(label, centre_mm)

    def kept_out_by_a_region(self, *, label: str, centre_mm: Sequence[float]) -> bool:
        """Whether a part of ``label`` standing at ``centre_mm`` stands in a region its task keeps out, whatever zone
        holds it too: a part the task placed (or the bin it places into), never one ``next_target`` skips only because a
        pick failed on it, which still stands there to be picked."""

        return any(region.contains(centre_mm) for region in self._regions if region.applies(label))

    def remaining(self, *, label: str, centres_mm: Sequence[Sequence[float]]) -> tuple[int, ...]:
        """The indices of the parts of ``label`` (by centre) that are not skipped."""

        return tuple(i for i, centre in enumerate(centres_mm) if not self._keeps_out(label, centre))

    def only_excluded_remain(self, *, label: str, centres_mm: Sequence[Sequence[float]]) -> bool:
        """True when parts of ``label`` were seen and every one of them is skipped: the stop condition."""

        return len(centres_mm) > 0 and not self.remaining(label=label, centres_mm=centres_mm)

    def only_excluded_sentence(self, *, label: str, count: int) -> str:
        """The stop message for :meth:`only_excluded_remain`.

        Where a region applies to ``label`` it says what the task keeps out (each region's reason); otherwise it is the
        zones' own sentence, as it always was. An empty label is a part the detector named nothing.
        """

        regions = self.regions(label)
        if not regions:
            return (f"Every {str(label).strip()!r} the camera sees ({count}) failed in one of the last "
                    f"{self._extra_picks + 1} picks and is being skipped, so the pick stops here rather than trying "
                    "one of them again or another object.")
        what = repr(str(label).strip()) if str(label).strip() else "part"
        kept = "; ".join(dict.fromkeys(region.reason or "an area the task keeps out" for region in regions))
        skipped = (f", or failed in one of the last {self._extra_picks + 1} picks and is being skipped"
                   if self._zones_for(label) else "")
        return (f"Every {what} the camera sees ({count}) stands where this task keeps out of its picks ({kept})"
                f"{skipped}, so the pick stops here rather than picking one of them or another object.")
