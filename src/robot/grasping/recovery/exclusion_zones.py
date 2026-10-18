"""Which parts a pick keeps out: the areas a task places into (the owner's Q4, 2026-09-30).

A task keeps out of its picks the area it places into: the bin its camera found, for both scopes, and a circle about
a taught place's drop for "until empty", so it never picks the bin, a part it already put in it, or a part it set
down. That is an :class:`ExclusionRegion`, laid with :meth:`ExclusionZones.keep_out_region`. A region:

* is a turned rectangle or a circle about BASE Z, a column over the table;
* applies to every label where it names none, the empty label of an unlabelled detector included;
* lasts every pick until :meth:`ExclusionZones.forget_regions`, which the task calls as it ends, or until
  :meth:`ExclusionZones.forget_region` forgets it alone (one place of a sort checked again replaces only its own).

The service holds one :class:`ExclusionZones` for a campaign. The pick loop asks :meth:`ExclusionZones.applies`
whether anything keeps a segmentation's label out before it reads where that segmentation stands, and drops every
segmentation for which :meth:`ExclusionZones.excludes` is true, next to its label gate. When
:meth:`ExclusionZones.only_excluded_remain` is true, the pick stops with :meth:`ExclusionZones.only_excluded_sentence`:
a look with nothing left for the task to pick.

The zones beside the regions, the parts ``next_target`` skipped for a few picks after a pick failed on them, left on
2026-10-16 with ``next_target`` (the owner's cleanup).

One campaign drives one instance from one thread. Nothing here moves the arm or imports robot code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

__all__ = [
    "ExclusionRegion",
    "ExclusionZones",
]


def _xy(value: Sequence[float], name: str) -> tuple[float, float]:
    out = tuple(float(c) for c in value)
    if len(out) not in (2, 3) or not all(math.isfinite(c) for c in out):
        raise ValueError(f"{name} must be two or three finite numbers in BASE mm, got {value!r}")
    return (out[0], out[1])


def _label(value: object) -> str:
    label = str(value).strip().casefold()
    if not label:
        raise ValueError("an exclusion region names the label it is for, or none; got an empty one")
    return label


def _label_read(value: object) -> str:
    """A segmentation's label as the regions compare labels: trimmed and case-folded; ``""`` for none."""
    return str(value or "").strip().casefold()


def _positive(value: object, name: str) -> float:
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number of mm, got {value!r}")
    return number


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
    """The regions one campaign's task keeps out of its picks (:class:`ExclusionRegion`)."""

    def __init__(self) -> None:
        self._regions: list[ExclusionRegion] = []

    def keep_out_region(self, region: ExclusionRegion) -> ExclusionRegion:
        """Keep ``region`` out of every pick from the next one on, until :meth:`forget_regions` (or until
        :meth:`forget_region` forgets it alone). Returns ``region``, the object to hand :meth:`forget_region`."""

        if not isinstance(region, ExclusionRegion):
            raise TypeError(f"keep_out_region takes an ExclusionRegion, not {type(region).__name__}")
        self._regions.append(region)
        return region

    def forget_regions(self) -> int:
        """Forget every region kept out (a task that ends). Returns how many were forgotten."""

        forgotten = len(self._regions)
        self._regions = []
        return forgotten

    def forget_region(self, region: ExclusionRegion) -> bool:
        """Forget ``region`` alone: the one kept out (the very object :meth:`keep_out_region` returned), else the first
        one equal to it; every other region stays. Returns whether one was forgotten.

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

    def applies(self, label: object) -> bool:
        """Whether anything keeps a part called ``label`` out: a region that applies to it. The pick loop asks this
        before it reads where a part stands; an empty label is asked about too."""

        return any(region.applies(label) for region in self._regions)

    def excludes(self, *, label: str, centre_mm: Sequence[float]) -> bool:
        """Whether a part of ``label`` standing at ``centre_mm`` is kept out now: a region that applies to it holds
        it."""

        return any(region.contains(centre_mm) for region in self._regions if region.applies(label))

    def remaining(self, *, label: str, centres_mm: Sequence[Sequence[float]]) -> tuple[int, ...]:
        """The indices of the parts of ``label`` (by centre) that are not kept out."""

        return tuple(i for i, centre in enumerate(centres_mm) if not self.excludes(label=label, centre_mm=centre))

    def only_excluded_remain(self, *, label: str, centres_mm: Sequence[Sequence[float]]) -> bool:
        """True when parts of ``label`` were seen and every one of them is kept out: the stop condition."""

        return len(centres_mm) > 0 and not self.remaining(label=label, centres_mm=centres_mm)

    def only_excluded_sentence(self, *, label: str, count: int) -> str:
        """The stop message for :meth:`only_excluded_remain`: what the task keeps out (each region's reason). An empty
        label is a part the detector named nothing."""

        regions = self.regions(label)
        what = repr(str(label).strip()) if str(label).strip() else "part"
        kept = "; ".join(dict.fromkeys(region.reason or "an area the task keeps out" for region in regions))
        return (f"Every {what} the camera sees ({count}) stands where this task keeps out of its picks ({kept}), so "
                "the pick stops here rather than picking one of them or another object.")
