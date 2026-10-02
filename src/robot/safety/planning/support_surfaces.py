"""What the parts stand on, found in the camera's own pixels on every world build.

The owner's decision of 2026-10-01: no fixture boxes for the mat. A 55 mm foam mat on the bench used to come back as
strips of obstacles 15 mm taller than the mat, which refused every low finger over it and filled the planner's slots.
"Der Stack muss selbst raffen wo was ist": the mat, the bare bench, a bin's floor, a slightly tilted reading, found from
the depth itself on every build, so a mat that is gone tomorrow is gone from the world tomorrow.

A support is a large, nearly level surface: at least 100 x 100 mm and 75 mm wide, at most :data:`MAX_SUPPORT_TILT_DEG`
off level, flat to the band in every view that sees it, and standing above the declared bench's band somewhere. Each
one becomes at most four tilted solids, each from the declared bench up to the surface's local reading plus the p90 of
what the reading stands over it plus the band: the *erasure top*. What lies within a solid up to that top is the
surface's (``perceived.DropReason.SUPPORT``), and the solid the guard and the planner hold stands
``support_allowance_mm`` over it, so a part the band swallowed keeps the guard's distance and the allowance while the
camera reads up to that much low (fix plan F3). The declared bench is held as an upright solid too, over the workspace
box, up to its band plus the allowance.

The rule every consumer leans on, and a property test pins: a point leaves the world only where such a solid holds
every pixel it stands for (``perceived.build_perceived_boxes``). Nothing here makes free space out of what was seen.

Pure: the same views, limits and tuning give the same model. No camera, no clock, no config. Millimetres in BASE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from src.robot.safety.planning.height_map import turn_of

if TYPE_CHECKING:  # pragma: no cover (typing only: perceived imports this module)
    from src.robot.core.keep_out import KeepOutBox

    from .perceived import DepthView, WorldBuildLimits, WorldBuildTuning
    from .self_envelope import BaseShape

__all__ = [
    "DETECTION_STRIDE",
    "MAX_SUPPORT_TILT_DEG",
    "BenchReading",
    "SupportModel",
    "SupportSolid",
    "SupportSurface",
    "detection_step",
    "find_supports",
]

#: The most a support may tilt from level, degrees. One bound for the camera world and the push
#: (``push_planner.MAX_SUPPORT_TILT_DEG`` is this). The camera reads the owner's mat 1.0-1.2 deg and the cell's table
#: 0.8-2.2 deg; a surface read steeper is no support and stays obstacles, as before.
MAX_SUPPORT_TILT_DEG = 5.0
#: The least a support covers, square millimetres: 100 x 100 mm, twice the Hand-E's 50 mm stroke each way, so no part a
#: hand can grip is ever its own support. A person's palm is 8 500 mm2.
MIN_SUPPORT_AREA_MM2 = 10_000.0
#: The least width of a support's smallest rectangle, millimetres: three 25 mm cells, wider than a ruler, a slat, a rim
#: or the top of a wall.
MIN_SUPPORT_WIDTH_MM = 75.0
#: The share of a support's own pixels that may lie inside a target the pick holds: a part's own top is not its support.
MAX_TARGET_SHARE = 0.5
#: How far past the workspace box the surfaces are looked for, millimetres: a mat whose edge lies past the box still
#: bounds what the hand comes down beside.
REGION_GROWTH_MM = 200.0
#: How sparsely the frame is read to find the surfaces: every 4th pixel. The surfaces come out the same as from every
#: 2nd, their tops within 0.8 mm, in a third of the time (2026-10-01, the owner's looks). What leaves the world is
#: still judged at every pixel the world reads.
DETECTION_STRIDE = 4
#: How many of a surface's own pixels a cell of its footprint holds at least.
_OWN_CELL_PIXELS = 3
#: A piece is split while its cells stand more than this over its local plane (p90), millimetres.
_PIECE_FLAT_MM = 1.5
#: Which percentile of a piece's cells over its local plane its solid takes as the reading's excess.
_EXCESS_PERCENTILE = 90.0
#: A piece is split while less than this share of it is the surface's own footprint.
_MIN_FILL = 0.75
#: At most this many solids per surface, one per this many square millimetres of footprint.
_SOLIDS_PER_SURFACE = 4
_AREA_PER_SOLID_MM2 = 50_000.0
#: At most this many support solids in a world, the largest surfaces first; the bench solid comes on top.
MAX_SUPPORT_SOLIDS = 12
#: How deep the bench solid reaches under the declared plane, millimetres.
_BENCH_DEPTH_MM = 50.0
#: The bench check reads cells within the band plus this of the declared plane, millimetres.
_BENCH_CHECK_EXTRA_MM = 10.0
#: The bench check says so where at least this many of its cells read lower than the allowance under the plane.
BENCH_LOW_LEAST_CELLS = 16


def detection_step(pixel_stride: int) -> int:
    """How many of the world's read pixels the detection takes one of, along each axis: every 4th pixel of the frame."""
    stride = max(1, int(pixel_stride))
    return max(1, -(-DETECTION_STRIDE // stride))


@dataclass(frozen=True, slots=True)
class SupportSolid:
    """One solid the camera world, the guard and the planner hold: a piece of a support surface, or the declared bench.

    ``centre_mm``, ``half_extents_mm`` and ``rotation`` are the box the guard and the planner hold, its columns the box's
    axes in BASE; its top stands ``allowance_mm`` over the *erasure top*, ``local_plane`` + ``excess_mm`` +
    ``band_mm``, which is how high the world takes what lies within it as the surface's.
    """

    name: str
    #: ``support`` or ``bench``.
    kind: str
    #: Which surface it is a piece of (:attr:`SupportSurface.index`); -1 for the bench.
    surface: int
    centre_mm: np.ndarray
    half_extents_mm: np.ndarray
    rotation: np.ndarray
    #: ``(a, b, c)``: the surface's reading under this solid, ``z = a + b x + c y``.
    local_plane: np.ndarray
    #: How far the reading stands over its local plane, p90 of its cells, millimetres.
    excess_mm: float
    #: The band over the reading that is still the surface, millimetres.
    band_mm: float
    #: How far the held solid stands over the erasure top, millimetres.
    allowance_mm: float
    #: How many of the surface's cells it holds.
    cells: int = 0

    def reading_at(self, xy: Any) -> np.ndarray:
        """The surface's local reading under each of ``(N, 2)`` BASE points, millimetres."""
        points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        a, b, c = (float(v) for v in self.local_plane)
        return a + b * points[:, 0] + c * points[:, 1]

    def erasure_top_at(self, xy: Any) -> np.ndarray:
        """How high the world takes what stands over each of ``xy`` as the surface's, millimetres."""
        return self.reading_at(xy) + float(self.excess_mm) + float(self.band_mm)

    def top_at(self, xy: Any) -> np.ndarray:
        """The top the guard and the planner hold over each of ``xy``, millimetres."""
        return self.erasure_top_at(xy) + float(self.allowance_mm)

    def _local(self, points: Any) -> np.ndarray:
        return (np.asarray(points, dtype=np.float64).reshape(-1, 3) - self.centre_mm) @ self.rotation

    def covers_xy(self, xy: Any) -> np.ndarray:
        """Whether the solid's top face lies over each of ``(N, 2)`` BASE points."""
        points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        local = self._local(np.column_stack((points, self.reading_at(points))))
        return np.all(np.abs(local[:, :2]) <= self.half_extents_mm[:2], axis=1)

    def contains(self, points: Any, grow_mm: float = 0.0) -> np.ndarray:
        """Whether each of ``(N, 3)`` BASE points lies inside the held solid grown by ``grow_mm``."""
        return np.all(np.abs(self._local(points)) <= self.half_extents_mm + float(grow_mm), axis=1)

    def erases(self, points: Any, grow_mm: float = 0.0, above_mm: "float | None" = None) -> np.ndarray:
        """Whether each of ``(N, 3)`` BASE points lies inside the solid up to its erasure top, grown by ``grow_mm``, and
        over the top by ``above_mm`` where that is given."""
        local = self._local(points)
        half = self.half_extents_mm + float(grow_mm)
        inside = np.all(np.abs(local[:, :2]) <= half[:2], axis=1)
        top = self.half_extents_mm[2] - float(self.allowance_mm) + float(grow_mm if above_mm is None else above_mm)
        return inside & (local[:, 2] >= -half[2]) & (local[:, 2] <= top)

    @property
    def enclosing_half_extents_mm(self) -> np.ndarray:
        """Half extents of the axis-aligned box that holds it, never smaller."""
        return np.abs(self.rotation) @ self.half_extents_mm

    @property
    def tilt_deg(self) -> float:
        """How far its up axis leans from BASE +Z, degrees."""
        return math.degrees(math.acos(max(-1.0, min(1.0, float(self.rotation[2, 2])))))

    @property
    def held_top_mm(self) -> float:
        """The highest point of the held solid, millimetres."""
        return float(self.centre_mm[2] + self.enclosing_half_extents_mm[2])

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind, "surface": int(self.surface),
            "centre_mm": [round(float(v), 3) for v in self.centre_mm],
            "dims_mm": [round(2.0 * float(v), 3) for v in self.half_extents_mm],
            "rotation": [round(float(v), 9) for v in self.rotation.reshape(-1)],
            "local_plane": [float(v) for v in self.local_plane], "excess_mm": round(float(self.excess_mm), 3),
            "band_mm": float(self.band_mm), "allowance_mm": float(self.allowance_mm), "cells": int(self.cells),
            "tilt_deg": round(self.tilt_deg, 3), "held_top_mm": round(self.held_top_mm, 3),
        }


@dataclass(frozen=True, slots=True)
class SupportSurface:
    """One surface the parts stand on, as the camera read it."""

    index: int
    #: ``support<index>``: what its solids' names end with.
    name: str
    #: ``(a, b, c)``: its plane, ``z = a + b x + c y``, fitted over its cells.
    coef: np.ndarray
    tilt_deg: float
    #: How far its cells' readings lie from that plane, root mean square, millimetres.
    rms_mm: float
    area_mm2: float
    width_mm: float
    #: Its plane over its own pixels, lowest and highest, millimetres.
    z_range_mm: tuple[float, float]
    #: The band over its reading that is still it, millimetres.
    band_mm: float
    #: Its solids' names.
    solids: tuple[str, ...] = ()
    #: How many pixels the detection read of it.
    own_pixels: int = 0
    #: Set where the surface is asked for under a part (:meth:`SupportModel.surface_under`): the plane through its local
    #: reading there, ``normal . p = offset_mm``, and that reading.
    offset_mm: float | None = None
    at_mm: tuple[float, float, float] | None = None

    @property
    def normal(self) -> np.ndarray:
        """Its plane's unit normal, pointing up."""
        n = np.array([-float(self.coef[1]), -float(self.coef[2]), 1.0])
        return n / float(np.linalg.norm(n))

    def height_at(self, xy: Any) -> np.ndarray:
        """Its plane under each of ``(N, 2)`` BASE points, millimetres."""
        points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        return float(self.coef[0]) + float(self.coef[1]) * points[:, 0] + float(self.coef[2]) * points[:, 1]

    def render(self) -> str:
        """One clause: where it is, how it tilts, how large it is and how many solids hold it."""
        return (f"{self.name} at z {self.z_range_mm[0]:.1f} to {self.z_range_mm[1]:.1f} mm, tilted {self.tilt_deg:.1f} "
                f"deg, {self.area_mm2 / 1e6:.3f} m2, {len(self.solids)} solid(s)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index), "name": self.name, "coef": [float(v) for v in self.coef],
            "tilt_deg": round(float(self.tilt_deg), 3), "rms_mm": round(float(self.rms_mm), 3),
            "area_mm2": round(float(self.area_mm2), 1), "width_mm": round(float(self.width_mm), 1),
            "z_range_mm": [round(float(v), 2) for v in self.z_range_mm], "band_mm": float(self.band_mm),
            "solids": list(self.solids), "own_pixels": int(self.own_pixels),
        }


@dataclass(frozen=True, slots=True)
class BenchReading:
    """Where the bare declared bench was seen, against where it is declared.

    The cells of the workspace box that lie in no support and read within the band plus 10 mm of the declared plane.
    It decides nothing: a sagging table and a camera that reads low look the same here, and only a ruler and a touch-off
    tell them apart. Nothing is raised by it (the owner, 2026-10-02).
    """

    #: How many cells it read.
    cells: int
    #: Their reading against the declared plane, median and 10th percentile, millimetres; below 0 reads low.
    median_mm: float
    p10_mm: float
    #: The cells that read more than the allowance under the plane, and where they lie: ``(x_min, x_max, y_min, y_max)``
    #: BASE millimetres, and their median reading.
    low_cells: int = 0
    low_where_mm: tuple[float, float, float, float] | None = None
    low_median_mm: float | None = None
    #: The allowance the low cells were judged against, millimetres.
    allowance_mm: float = 0.0

    @property
    def reads_low(self) -> bool:
        """Whether enough cells read more than the allowance under the declared plane to say so."""
        return self.low_cells >= BENCH_LOW_LEAST_CELLS

    def render(self) -> str:
        line = f"bare bench read {self.median_mm:+.1f} mm (median of {self.cells} cells, p10 {self.p10_mm:+.1f})"
        if self.low_cells and self.low_where_mm is not None and self.low_median_mm is not None:
            x0, x1, y0, y1 = self.low_where_mm
            line += (f", {self.low_cells} cells {self.low_median_mm:+.1f} mm at x {x0:.0f} to {x1:.0f}, "
                     f"y {y0:.0f} to {y1:.0f}")
        return line

    def warning(self) -> str:
        """What a WARNING says where the bench :attr:`reads_low`."""
        assert self.low_where_mm is not None and self.low_median_mm is not None  # noqa: S101 (reads_low holds)
        x0, x1, y0, y1 = self.low_where_mm
        return (
            f"the bare bench reads lower than declared: {self.low_cells} cells at x {x0:.0f} to {x1:.0f} mm, y {y0:.0f} "
            f"to {y1:.0f} mm read {self.low_median_mm:+.1f} mm (median), more than the "
            f"{self.allowance_mm:g} mm the support solids stand over what they take. The solids follow the camera, so "
            "a thin part there keeps less than the guard's distance plus the allowance if the camera reads low. Nothing "
            "is raised: a sagging bench and a camera that reads low look the same here. Measure the bench with a ruler "
            "and a TCP touch-off; a recalibration is a person's decision"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cells": int(self.cells), "median_mm": round(float(self.median_mm), 2), "p10_mm": round(float(self.p10_mm), 2),
            "low_cells": int(self.low_cells),
            "low_where_mm": None if self.low_where_mm is None else [round(float(v), 1) for v in self.low_where_mm],
            "low_median_mm": None if self.low_median_mm is None else round(float(self.low_median_mm), 2),
            "allowance_mm": float(self.allowance_mm), "reads_low": self.reads_low,
        }


@dataclass(frozen=True, slots=True)
class SupportModel:
    """The surfaces a set of views stands on, their solids and the bench's, and how the bare bench reads.

    One model per world build, and per judged look in the pick: the same frames give the same model. The camera world,
    the guard and the planner hold :attr:`solids`; the calculator and the push ask the rest.
    """

    surfaces: tuple[SupportSurface, ...]
    #: The support solids, then the bench solid where one is held.
    solids: tuple[SupportSolid, ...]
    #: The declared plane's top, millimetres.
    bench_top_mm: float
    #: The largest bench band of the views, millimetres.
    bench_band_mm: float
    allowance_mm: float
    bench_reading: BenchReading | None = None
    #: Surfaces found and not held, each with why: over the solid budget, or a window it could not leave open.
    left_out: tuple[str, ...] = ()
    #: The pixel stride the views were read at, for :meth:`table_points`.
    pixel_stride: int = 2
    #: How the detection went, for a reader: cells, flat cells, regions and their verdicts.
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def support_solids(self) -> tuple[SupportSolid, ...]:
        return tuple(solid for solid in self.solids if solid.kind == "support")

    @property
    def bench_solid(self) -> SupportSolid | None:
        return next((solid for solid in self.solids if solid.kind == "bench"), None)

    # ---- what the camera world asks --------------------------------------------------------------------------------

    def erased_by_support(self, points: Any) -> np.ndarray:
        """Whether each of ``(N, 3)`` points lies within a support solid up to its erasure top."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        out = np.zeros(pts.shape[0], dtype=bool)
        for solid in self.support_solids:
            # Only what lies within the box that encloses the solid is asked the solid itself.
            reach = solid.enclosing_half_extents_mm
            near = np.nonzero(np.all(np.abs(pts - solid.centre_mm) <= reach, axis=1) & ~out)[0]
            if near.size:
                out[near] = solid.erases(pts[near])
        return out

    def erasure_top_under(self, points: Any) -> np.ndarray:
        """Per point, the erasure top of the highest solid under it: a support's where one covers it, else the bench's
        band where the bench solid covers it, else ``-inf``."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        top = np.full(pts.shape[0], -np.inf)
        for solid in self.solids:
            covered = solid.covers_xy(pts[:, :2])
            if covered.any():
                top[covered] = np.maximum(top[covered], solid.erasure_top_at(pts[covered, :2]))
        return top

    def stands_on_a_solid(self, point: Any, grow_mm: float, above_mm: "float | None" = None) -> bool:
        """Whether a point lies within ``grow_mm`` of a solid up to its erasure top, a support's or the bench's, or at
        most ``above_mm`` over that top where that is given."""
        pts = np.asarray(point, dtype=np.float64).reshape(-1, 3)
        return any(bool(solid.erases(pts, grow_mm=grow_mm, above_mm=above_mm).any()) for solid in self.solids)

    # ---- what the calculator and the push ask (contract 7) ---------------------------------------------------------

    def _covering(self, xy: np.ndarray) -> "list[tuple[SupportSolid, np.ndarray]]":
        return [(solid, solid.covers_xy(xy)) for solid in self.support_solids]

    def local_plane_under(self, xy: Any) -> np.ndarray | None:
        """Per point of ``(N, 2)``, the local plane of the highest support reading over it, ``(nx, ny, nz, d)`` with
        ``n . p = d``; rows of NaN where no support covers a point, ``None`` where none covers any."""
        points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        best = np.full(points.shape[0], -np.inf)
        planes = np.full((points.shape[0], 4), np.nan)
        for solid, covered in self._covering(points):
            if not covered.any():
                continue
            reading = solid.reading_at(points)
            higher = covered & (reading > best)
            if higher.any():
                a, b, c = (float(v) for v in solid.local_plane)
                norm = math.sqrt(b * b + c * c + 1.0)
                planes[higher] = (-b / norm, -c / norm, 1.0 / norm, a / norm)
                best[higher] = reading[higher]
        return planes if np.isfinite(best).any() else None

    def height_under(self, foot_xy: Any) -> float | None:
        """The highest local reading of a support under a part's foot, millimetres; ``None`` where none lies under it.

        What the calculator takes as the support under a part, beside the part's own foot (the higher wins)."""
        points = np.asarray(foot_xy, dtype=np.float64).reshape(-1, 2)
        best: float | None = None
        for solid, covered in self._covering(points):
            if covered.any():
                high = float(solid.reading_at(points[covered]).max())
                best = high if best is None else max(best, high)
        return best

    def surface_under(self, foot_xy: Any) -> SupportSurface | None:
        """The surface whose solids cover the most of a part's foot, the higher on a tie, with :attr:`SupportSurface.
        offset_mm` set to the plane through its local reading there; ``None`` where no support lies under it."""
        points = np.asarray(foot_xy, dtype=np.float64).reshape(-1, 2)
        best: "tuple[tuple[int, float], SupportSurface, np.ndarray] | None" = None
        for surface in self.surfaces:
            covered = np.zeros(points.shape[0], dtype=bool)
            reading = np.full(points.shape[0], -np.inf)
            for solid in self.support_solids:
                if solid.surface != surface.index:
                    continue
                here = solid.covers_xy(points)
                covered |= here
                reading = np.where(here, np.maximum(reading, solid.reading_at(points)), reading)
            count = int(covered.sum())
            if count:
                key = (count, float(reading[covered].max()))
                if best is None or key > best[0]:
                    best = (key, surface, np.column_stack((points[covered], reading[covered])))
        if best is None:
            return None
        _, surface, at = best
        centre = at.mean(axis=0)
        offset = float(surface.normal @ centre)
        return replace(surface, offset_mm=offset, at_mm=(float(centre[0]), float(centre[1]), float(centre[2])))

    def is_support_point(self, points: Any) -> np.ndarray:
        """Whether each of ``(N, 3)`` points lies within the band of the local reading of a support under it (F5).

        Not "inside a solid": a solid also holds what its reading stands over its local plane, and a thing that high is
        no surface for a neighbour rule or a push to land on."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        out = np.zeros(pts.shape[0], dtype=bool)
        for solid, covered in self._covering(pts[:, :2]):
            if covered.any():
                near = np.abs(pts[covered, 2] - solid.reading_at(pts[covered, :2])) <= float(solid.band_mm)
                out[np.nonzero(covered)[0][near]] = True
        return out

    def table_points(self, views: "Sequence[DepthView]", foot_xy: Any) -> np.ndarray:
        """The pixels of the surface under a part's foot that lie within the band of their local reading (F5), ``(N, 3)``
        BASE millimetres, read at this model's pixel stride; empty where no support lies under the foot."""
        surface = self.surface_under(foot_xy)
        if surface is None:
            return np.zeros((0, 3))
        mine = [solid for solid in self.support_solids if solid.surface == surface.index]
        out = []
        for view in views:
            points, _, _ = _view_pixels(view, int(self.pixel_stride))
            if points.shape[0] == 0:
                continue
            keep = np.zeros(points.shape[0], dtype=bool)
            for solid in mine:
                covered = solid.covers_xy(points[:, :2])
                near = np.abs(points[covered, 2] - solid.reading_at(points[covered, :2])) <= float(solid.band_mm)
                keep[np.nonzero(covered)[0][near]] = True
            out.append(points[keep])
        return np.concatenate(out) if out else np.zeros((0, 3))

    def solid_top_over(self, xy: Any) -> float | None:
        """The highest top any held solid stands at over ``(N, 2)`` BASE points, allowance included, millimetres; ``None``
        where none covers any. What a push's finger keeps its distance from along its path."""
        points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        best: float | None = None
        for solid in self.solids:
            covered = solid.covers_xy(points)
            if covered.any():
                high = float(solid.top_at(points[covered]).max())
                best = high if best is None else max(best, high)
        return best

    # ---- what a person reads --------------------------------------------------------------------------------------

    def render(self) -> str:
        """One line: each support, and how the bare bench reads. ASCII, no trailing newline."""
        if self.surfaces:
            parts = ["supports: " + "; ".join(surface.render() for surface in self.surfaces)]
        else:
            parts = ["supports: none found"]
        bench = self.bench_solid
        if bench is not None:
            parts.append(f"bench held to {bench.held_top_mm:.1f} mm")
        if self.bench_reading is not None:
            parts.append(self.bench_reading.render())
        if self.left_out:
            parts.append("not held: " + "; ".join(self.left_out))
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "surfaces": [surface.to_dict() for surface in self.surfaces],
            "solids": [solid.to_dict() for solid in self.solids],
            "bench_top_mm": float(self.bench_top_mm), "bench_band_mm": float(self.bench_band_mm),
            "allowance_mm": float(self.allowance_mm),
            "bench_reading": None if self.bench_reading is None else self.bench_reading.to_dict(),
            "left_out": list(self.left_out),
        }


# ---------------------------------------------------------------------------------------------------------------------
# Finding them
# ---------------------------------------------------------------------------------------------------------------------


def find_supports(
    views: "Sequence[DepthView]",
    *,
    limits: "WorldBuildLimits",
    tuning: "WorldBuildTuning",
    base: "BaseShape | None" = None,
    keep_out: "Sequence[KeepOutBox]" = (),
) -> SupportModel:
    """The supports ``views`` stand on, as the camera world finds them on a world build of the same views.

    Each view is read as the world reads it at ``tuning.pixel_stride``, every :func:`detection_step` of those pixels,
    without the pixels its masks leave out and without what its own body (``DepthView.self_body``) holds. ``base`` is
    the robot's base about the BASE axis: no cell within its radius, the margin and one cell of the axis is a support,
    or its top would put the shoulder in a solid. ``keep_out`` are the targets a pick holds: a surface more than half of
    whose own pixels lie in them is a part's own top and no support.
    """
    if limits.support_plane_top_mm is None:
        raise ValueError("no support plane is declared, so no surface can stand over it")
    step = int(tuning.pixel_stride) * detection_step(int(tuning.pixel_stride))
    clearance = float(limits.plane_clearance_mm)
    pixels, view_of, bands = [], [], []
    for index, view in enumerate(views):
        points, ranges, _ = _view_pixels(view, step)
        if view.self_body is not None and points.shape[0]:
            mine = ~view.self_body.contains(points)
            points, ranges = points[mine], ranges[mine]
        band = clearance + np.minimum(float(view.placement_error_mm) + float(view.placement_error_rad) * ranges,
                                      _max_declared_band())
        pixels.append(points)
        view_of.append(np.full(points.shape[0], index, dtype=np.int64))
        bands.append(float(band.max()) if band.size else clearance)
    return detect(
        np.concatenate(pixels) if pixels else np.zeros((0, 3)),
        np.concatenate(view_of) if view_of else np.zeros(0, dtype=np.int64),
        tuple(bands), limits=limits, tuning=tuning, base=base, keep_out=keep_out,
    )


def detect(
    pixels: np.ndarray,
    view_of: np.ndarray,
    band_of_view: Sequence[float],
    *,
    limits: "WorldBuildLimits",
    tuning: "WorldBuildTuning",
    base: "BaseShape | None" = None,
    keep_out: "Sequence[KeepOutBox]" = (),
) -> SupportModel:
    """The supports of ``(N, 3)`` BASE pixels, ``view_of`` naming each pixel's view and ``band_of_view`` each view's
    largest bench band: the steps :func:`find_supports` takes after reading the views, which the camera world takes on
    the pixels its self filters left."""
    assert limits.support_plane_top_mm is not None  # noqa: S101 (both callers refuse a world without a plane first)
    plane = float(limits.support_plane_top_mm)
    clearance = float(limits.plane_clearance_mm)
    bench_band = float(max(band_of_view, default=clearance))
    allowance = float(getattr(tuning, "support_allowance_mm", 0.0))
    cell = float(tuning.cluster_voxel_mm)
    margin = float(tuning.margin_mm)
    prefix = str(tuning.name_prefix)
    budget_boxes = int(tuning.max_boxes)
    bench = _bench_solid(limits, bench_band, margin, allowance, prefix) if budget_boxes >= 1 else None

    def model(surfaces: "Sequence[SupportSurface]", solids: "Sequence[SupportSolid]", **extra: Any) -> SupportModel:
        return SupportModel(surfaces=tuple(surfaces), solids=tuple(solids), bench_top_mm=plane,
                            bench_band_mm=bench_band, allowance_mm=allowance, pixel_stride=int(tuning.pixel_stride),
                            **extra)

    points = np.asarray(pixels, dtype=np.float64).reshape(-1, 3)
    views = np.asarray(view_of, dtype=np.int64).reshape(-1)
    x0, x1 = float(limits.x_mm[0]) - REGION_GROWTH_MM, float(limits.x_mm[1]) + REGION_GROWTH_MM
    y0, y1 = float(limits.y_mm[0]) - REGION_GROWTH_MM, float(limits.y_mm[1]) + REGION_GROWTH_MM
    keep = (np.all(np.isfinite(points), axis=1) & (points[:, 0] >= x0) & (points[:, 0] <= x1)
            & (points[:, 1] >= y0) & (points[:, 1] <= y1) & (points[:, 2] > plane - 3.0 * clearance))
    if base is not None and points.shape[0]:
        # Never the robot's base: its top is a flat disc at the shoulder's foot, and as a support it would put the
        # shoulder in a solid and refuse every motion (attack S4). No cell within its radius, the margin and one cell.
        reach = float(base.radius_mm) + margin + cell
        cx = (np.floor(points[:, 0] / cell) + 0.5) * cell
        cy = (np.floor(points[:, 1] / cell) + 0.5) * cell
        keep &= np.hypot(cx, cy) > reach
    points, views = points[keep], views[keep]
    if points.shape[0] < int(tuning.min_points):
        return model((), () if bench is None else (bench,))

    grid = _CellGrid.of(points, views, cell=cell, min_points=int(tuning.min_points), band=clearance)
    # The bench as the camera reads it, before any view is raised: the check says how the camera reads, not how the
    # solids are built.
    bench_cells = _bench_cells(grid, limits, plane, clearance)
    offsets = grid.view_offsets()
    if np.any(offsets > 0.0):
        # F6 (attack H6): a view reads its own cells by itself as low as it reads the cells it shares with the others.
        # Its pixels are raised by that, so a region only it sees is not held lower than the views that see more agree.
        points = points.copy()
        points[:, 2] += offsets[views]
        grid = _CellGrid.of(points, views, cell=cell, min_points=int(tuning.min_points), band=clearance)
    stats: dict[str, Any] = {"cells": int(grid.cells.size), "flat_cells": int(grid.flat.sum()), "views": grid.views,
                             "cells_refused_by_a_view": grid.refused_by_a_view,
                             "view_offsets_mm": [round(float(v), 2) for v in offsets]}
    regions = grid.regions(clearance)
    candidates = []
    verdicts = []
    for member in regions:
        region = _fit_region(grid, member, clearance, plane, bench_band, cell)
        if region is None:
            continue
        verdicts.append({k: region[k] for k in ("verdict", "area", "tilt", "rms", "width", "z50")})
        if region["verdict"] == "ok":
            candidates.append(region)
    stats["regions"] = verdicts
    candidates.sort(key=lambda region: -float(region["area"]))

    max_solids = min(MAX_SUPPORT_SOLIDS, max(0, budget_boxes - 2))
    surfaces: list[SupportSurface] = []
    solids: list[SupportSolid] = []
    left_out: list[str] = []
    for region in candidates:
        built = _surface(points, views, grid, region, clearance, band_of_view, plane, margin, cell,
                         int(tuning.min_points), keep_out)
        if built is None:
            stats["refused_own_pixels"] = int(stats.get("refused_own_pixels", 0)) + 1
            continue
        pieces, coef, extra, why = built
        if why:
            left_out.append(why)
        if not pieces:
            continue
        if len(solids) + len(pieces) > max_solids:
            left_out.append(f"a surface of {region['area'] / 1e6:.3f} m2 at {region['z50']:.0f} mm: its "
                            f"{len(pieces)} solid(s) did not fit the {max_solids} a world holds, so it takes nothing")
            continue
        index = len(surfaces)
        names = []
        for piece in pieces:
            name = f"{prefix}s{len(solids):02d}_support{index}"
            names.append(name)
            solids.append(replace(piece, name=name, surface=index, allowance_mm=allowance,
                                  centre_mm=piece.centre_mm + piece.rotation[:, 2] * (allowance / 2.0),
                                  half_extents_mm=piece.half_extents_mm + np.array([0.0, 0.0, allowance / 2.0])))
        surfaces.append(SupportSurface(
            index=index, name=f"support{index}", coef=coef, tilt_deg=float(region["tilt"]), rms_mm=float(region["rms"]),
            area_mm2=float(region["area"]), width_mm=float(region["width"]), z_range_mm=extra["z_range"],
            band_mm=float(extra["band"]), solids=tuple(names), own_pixels=int(extra["own"]),
        ))
    supports = list(solids)
    if bench is not None:
        solids.append(replace(bench, name=f"{prefix}s{len(solids):02d}_bench"))
    return model(surfaces, solids, bench_reading=_bench_reading(bench_cells, supports, allowance),
                 left_out=tuple(left_out), stats=stats)


# ---------------------------------------------------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------------------------------------------------


def _max_declared_band() -> float:
    from .perceived import MAX_DECLARED_BAND_MM  # noqa: PLC0415 (perceived imports this module)

    return float(MAX_DECLARED_BAND_MM)


def _view_pixels(view: "DepthView", stride: int) -> "tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]":
    """Every ``stride``-th pixel of a view with a depth and not left out by its masks: BASE points, their ranges from
    the camera, and their rows and columns in the frame."""
    depth = np.asarray(view.surface_depth_mm, dtype=np.float64)
    keep = np.ones(depth.shape, dtype=bool)
    for mask in view.exclude_masks:
        keep &= ~np.asarray(mask).astype(bool)
    step = max(1, int(stride))
    sampled = depth[::step, ::step]
    with np.errstate(invalid="ignore"):
        valid = keep[::step, ::step] & np.isfinite(sampled) & (sampled > 0.0)
    rows, cols = np.nonzero(valid)
    z = sampled[rows, cols]
    rows, cols = rows * step, cols * step
    matrix = np.asarray(view.intrinsics, dtype=np.float64)
    camera = np.column_stack(((cols - matrix[0, 2]) * z / matrix[0, 0], (rows - matrix[1, 2]) * z / matrix[1, 1], z))
    transform = np.asarray(view.camera_to_base, dtype=np.float64)
    return camera @ transform[:3, :3].T + transform[:3, 3], np.linalg.norm(camera, axis=1), (rows, cols)


def _plane_fit(x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``(a, b, c)`` of the weighted least-squares plane ``z = a + b x + c y``."""
    root = np.sqrt(np.maximum(w, 0.0))
    design = np.column_stack((np.ones_like(x), x, y)) * root[:, None]
    coef, *_ = np.linalg.lstsq(design, z * root, rcond=None)
    return np.asarray(coef, dtype=np.float64)


def _tilt_deg(coef: np.ndarray) -> float:
    return math.degrees(math.atan(math.hypot(float(coef[1]), float(coef[2]))))


@dataclass
class _CellGrid:
    """The detection's cells: per cell of the clustering grid, whether every view that saw it fully saw it flat, and
    its level, the highest such view's median."""

    origin: np.ndarray       # the grid index of cell (0, 0)
    span: np.ndarray         # cells along x and y
    cell: float
    cells: np.ndarray        # the occupied cells' keys, sorted
    cell_of_pixel: np.ndarray
    flat: np.ndarray         # per cell
    level: np.ndarray        # per cell, the highest flat view's median (-inf where none)
    weight: np.ndarray       # per cell, the pixels of the views that saw it fully
    views: int
    refused_by_a_view: int
    # per (cell, view) group
    group_cell: np.ndarray
    group_view: np.ndarray
    group_median: np.ndarray
    group_flat: np.ndarray
    group_full: np.ndarray
    n_full: np.ndarray       # per cell

    @classmethod
    def of(cls, points: np.ndarray, views: np.ndarray, *, cell: float, min_points: int, band: float) -> "_CellGrid":
        ij = np.floor(points[:, :2] / cell).astype(np.int64)
        origin = ij.min(axis=0)
        ij -= origin
        span = ij.max(axis=0) + 1
        key = ij[:, 0] * span[1] + ij[:, 1]
        count_views = int(views.max()) + 1 if views.size else 1
        g_key, g_n, g_median, g_spread = _group_stats(points, key * count_views + views)
        g_cell, g_view = g_key // count_views, g_key % count_views
        full = g_n >= int(min_points)
        g_flat = full & (g_spread <= float(band))
        cells, inverse = np.unique(g_cell, return_inverse=True)
        n_full = np.bincount(inverse, full.astype(np.int64), minlength=cells.size)
        n_flat = np.bincount(inverse, g_flat.astype(np.int64), minlength=cells.size)
        level = np.full(cells.size, -np.inf)
        np.maximum.at(level, inverse[g_flat], g_median[g_flat])
        weight = np.bincount(inverse, np.where(full, g_n, 0).astype(np.float64), minlength=cells.size)
        flat = (n_full > 0) & (n_flat == n_full)
        return cls(origin=origin, span=span, cell=float(cell), cells=cells, cell_of_pixel=key, flat=flat, level=level,
                   weight=weight, views=count_views, refused_by_a_view=int(((n_flat > 0) & (n_flat < n_full)).sum()),
                   group_cell=inverse, group_view=g_view, group_median=g_median, group_flat=g_flat, group_full=full,
                   n_full=n_full)

    def centres(self, which: np.ndarray) -> np.ndarray:
        """``(N, 2)`` BASE centres of the cells ``which`` (indices into :attr:`cells`)."""
        keys = self.cells[which]
        i, j = keys // self.span[1], keys % self.span[1]
        return np.column_stack(((i + self.origin[0] + 0.5) * self.cell, (j + self.origin[1] + 0.5) * self.cell))

    def view_offsets(self) -> np.ndarray:
        """Per view, how much lower it reads the flat cells it shares with another view than their level (F6)."""
        offsets = np.zeros(self.views)
        if self.views < 2:
            return offsets
        shared = self.flat[self.group_cell] & (self.n_full[self.group_cell] >= 2) & self.group_flat
        for view in range(self.views):
            mine = shared & (self.group_view == view)
            if mine.any():
                offsets[view] = max(0.0, float(np.median(self.level[self.group_cell[mine]] - self.group_median[mine])))
        return offsets

    def regions(self, band: float) -> "list[np.ndarray]":
        """The flat cells joined to their 8 neighbours where their levels differ by at most half the band, as arrays of
        cell indices, each region once."""
        from scipy.sparse import coo_matrix  # noqa: PLC0415 (kept out of import time)
        from scipy.sparse.csgraph import connected_components  # noqa: PLC0415

        flat = np.nonzero(self.flat)[0]
        if flat.size == 0:
            return []
        keys = self.cells[flat]
        i, j = keys // self.span[1], keys % self.span[1]
        lookup = np.full((int(self.span[0]) + 2, int(self.span[1]) + 2), -1, dtype=np.int64)
        lookup[i + 1, j + 1] = np.arange(flat.size)
        level = self.level[flat]
        rows, cols = [], []
        for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
            here = lookup[i + 1, j + 1]
            there = lookup[i + 1 + di, j + 1 + dj]
            both = there >= 0
            a, b = here[both], there[both]
            close = np.abs(level[a] - level[b]) <= band / 2.0
            rows.append(a[close])
            cols.append(b[close])
        r, c = np.concatenate(rows), np.concatenate(cols)
        graph = coo_matrix((np.ones(r.size), (r, c)), shape=(flat.size, flat.size))
        count, label = connected_components(graph, directed=False)
        order = np.argsort(label, kind="stable")
        bounds = np.searchsorted(label[order], np.arange(count + 1))
        return [flat[order[bounds[k]:bounds[k + 1]]] for k in range(count)]


def _group_stats(points: np.ndarray, key: np.ndarray) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Per group of ``key``: the keys, the counts, the median z, and the spread q90-q10 of z about the group's own
    least-squares plane (a cell's own slope is the D415's ripple, 7 deg at the median, and says nothing of a tilt)."""
    keys, inverse, count = np.unique(key, return_inverse=True, return_counts=True)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    mean_x = np.bincount(inverse, x) / count
    mean_y = np.bincount(inverse, y) / count
    mean_z = np.bincount(inverse, z) / count
    dx, dy, dz = x - mean_x[inverse], y - mean_y[inverse], z - mean_z[inverse]
    sxx, sxy, syy = np.bincount(inverse, dx * dx), np.bincount(inverse, dx * dy), np.bincount(inverse, dy * dy)
    sxz, syz = np.bincount(inverse, dx * dz), np.bincount(inverse, dy * dz)
    det = sxx * syy - sxy * sxy
    solvable = (det > 1e-6) & (count >= 3)
    safe = np.where(solvable, det, 1.0)
    b = np.where(solvable, (sxz * syy - syz * sxy) / safe, 0.0)
    c = np.where(solvable, (syz * sxx - sxz * sxy) / safe, 0.0)
    residual = dz - b[inverse] * dx - c[inverse] * dy
    start = np.concatenate(([0], np.cumsum(count)[:-1]))
    by_residual = residual[np.lexsort((residual, inverse))]
    spread = (by_residual[start + np.floor(0.9 * (count - 1)).astype(np.int64)]
              - by_residual[start + np.floor(0.1 * (count - 1)).astype(np.int64)])
    by_z = z[np.lexsort((z, inverse))]
    median = by_z[start + (count - 1) // 2]
    return keys, count, median, spread


def _fit_region(grid: _CellGrid, member: np.ndarray, band: float, plane: float, bench_band: float,
                cell: float) -> "dict[str, Any] | None":
    """A region's weighted plane, trimmed three times at half the band, and its verdict; ``None`` for a region too
    small to be one."""
    least = int(math.ceil(MIN_SUPPORT_AREA_MM2 / (cell * cell)))
    if member.size < least:
        return None
    xy = grid.centres(member)
    x, y, z, w = xy[:, 0], xy[:, 1], grid.level[member], grid.weight[member]
    keep = np.ones(member.size, dtype=bool)
    coef = None
    for _ in range(3):
        if keep.sum() < 3:
            break
        coef = _plane_fit(x[keep], y[keep], z[keep], w[keep])
        keep = np.abs(z - (coef[0] + coef[1] * x + coef[2] * y)) <= band / 2.0
    if coef is None or keep.sum() < least:
        return None
    residual = z - (coef[0] + coef[1] * x + coef[2] * y)
    tilt = _tilt_deg(coef)
    rms = float(np.sqrt(np.average(residual[keep] ** 2, weights=np.maximum(w[keep], 1e-9))))
    inliers = xy[keep]
    yaw = turn_of(inliers)
    u = math.cos(yaw) * inliers[:, 0] + math.sin(yaw) * inliers[:, 1]
    v = -math.sin(yaw) * inliers[:, 0] + math.cos(yaw) * inliers[:, 1]
    width = float(min(np.ptp(u), np.ptp(v))) + cell
    verdict = "ok"
    if tilt > MAX_SUPPORT_TILT_DEG:
        verdict = "tilted"
    elif width < MIN_SUPPORT_WIDTH_MM:
        verdict = "narrow"
    elif rms > band / 2.0:
        verdict = "rough"
    elif float((coef[0] + coef[1] * x[keep] + coef[2] * y[keep]).max()) + band <= plane + bench_band:
        # Never above the declared band: the declared bench holds it.
        verdict = "bench"
    return {"coef": coef, "cells": member[keep], "area": float(keep.sum()) * cell * cell, "tilt": tilt, "rms": rms,
            "width": width, "verdict": verdict, "z50": float(np.median(z[keep]))}


def _surface(
    points: np.ndarray, views: np.ndarray, grid: _CellGrid, region: dict[str, Any], band: float,
    band_of_view: Sequence[float], plane: float, margin: float, cell: float, min_points: int,
    keep_out: "Sequence[KeepOutBox]",
) -> "tuple[list[SupportSolid], np.ndarray, dict[str, Any], str] | None":
    """A region's own pixels, its footprint, its pieces and their erasure solids (without the allowance);
    ``None`` where it is a held target's own top or too thin to hold. The last item says why a piece was not built."""
    from scipy import ndimage  # noqa: PLC0415 (kept out of import time)
    from scipy.spatial import Delaunay  # noqa: PLC0415

    coef = np.asarray(region["coef"], dtype=np.float64)
    span = grid.span
    mask = np.zeros((int(span[0]) + 2, int(span[1]) + 2), dtype=bool)
    keys = grid.cells[region["cells"]]
    ci, cj = keys // span[1], keys % span[1]
    mask[ci + 1, cj + 1] = True
    if keys.size >= 3:
        # The region's hull: holes and bays inside it are its, where nothing lower was seen (below).
        try:
            hull = Delaunay(np.column_stack((ci, cj)).astype(np.float64))
            gi, gj = np.mgrid[0:int(span[0]), 0:int(span[1])]
            inside = hull.find_simplex(np.column_stack((gi.ravel(), gj.ravel())).astype(np.float64)) >= 0
            mask[1:-1, 1:-1] |= inside.reshape(gi.shape)
        except Exception:  # noqa: BLE001 (cells in one line: the cells alone)
            pass
    mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool))
    key = grid.cell_of_pixel
    near = mask[key // span[1] + 1, key % span[1] + 1]
    near_points, near_views = points[near], views[near]
    residual = near_points[:, 2] - (coef[0] + coef[1] * near_points[:, 0] + coef[2] * near_points[:, 1])
    own = np.abs(residual) <= band
    lower = residual < -band
    if int(own.sum()) < min_points:
        return None
    if keep_out:
        inside_target = np.zeros(int(own.sum()), dtype=bool)
        for box in keep_out:
            inside_target |= box.contains(near_points[own])
        if float(inside_target.mean()) > MAX_TARGET_SHARE:
            return None
    seen_views = np.unique(near_views[own])
    surface_band = max(band, max((float(band_of_view[v]) for v in seen_views if v < len(band_of_view)), default=band))
    mine = near_points[own]
    yaw = turn_of(mine[:: max(1, mine.shape[0] // 4000), :2])
    cs, sn = math.cos(yaw), math.sin(yaw)

    def turn(xy: np.ndarray) -> np.ndarray:
        return np.column_stack((cs * xy[:, 0] + sn * xy[:, 1], -sn * xy[:, 0] + cs * xy[:, 1]))

    uv = turn(mine[:, :2])
    # The size rules once more on the surface's own pixels: whole cells round a small top up, a 90 mm block straight
    # under the camera fills 16 of them. Its pixels' rectangle, one pixel's spacing added, does not.
    extent = np.ptp(uv, axis=0)
    spacing = math.sqrt(float(np.prod(np.maximum(extent, 1e-9))) / float(max(1, mine.shape[0])))
    sides = extent + spacing
    if float(sides.min()) < MIN_SUPPORT_WIDTH_MM or float(np.prod(sides)) < MIN_SUPPORT_AREA_MM2:
        return None
    u0, v0 = uv.min(axis=0) - cell
    nu = int(math.ceil((uv[:, 0].max() + cell - u0) / cell)) + 1
    nv = int(math.ceil((uv[:, 1].max() + cell - v0) / cell)) + 1
    ou = np.clip(np.floor((uv[:, 0] - u0) / cell).astype(np.int64), 0, nu - 1)
    ov = np.clip(np.floor((uv[:, 1] - v0) / cell).astype(np.int64), 0, nv - 1)
    flat_key = ov * nu + ou
    count = np.bincount(flat_key, minlength=nu * nv).reshape(nv, nu)
    seen = count >= _OWN_CELL_PIXELS
    own_residual = residual[own]
    order = np.lexsort((own_residual, flat_key))
    sorted_keys = flat_key[order]
    unique, first, per = np.unique(sorted_keys, return_index=True, return_counts=True)
    median_flat = np.full(nu * nv, np.nan)
    median_flat[unique] = own_residual[order][first + (per - 1) // 2]
    median_residual = median_flat.reshape(nv, nu)
    low = np.zeros((nv, nu), dtype=bool)
    if lower.any():
        low_uv = turn(near_points[lower, :2])
        lu = np.floor((low_uv[:, 0] - u0) / cell).astype(np.int64)
        lv = np.floor((low_uv[:, 1] - v0) / cell).astype(np.int64)
        inside = (lu >= 0) & (lu < nu) & (lv >= 0) & (lv < nv)
        low = np.bincount(lv[inside] * nu + lu[inside], minlength=nu * nv).reshape(nv, nu) >= _OWN_CELL_PIXELS
    footprint = ndimage.binary_fill_holes(seen) & (seen | ~low)
    lower_cells = low & ~footprint
    # What was seen lower inside the surface's own outline: a frame's window, a hole. Lower cells beyond its edge, the
    # bench beside a mat, are no window.
    windows = ndimage.binary_fill_holes(footprint) & ~footprint
    centre_u = u0 + (np.arange(nu) + 0.5) * cell
    centre_v = v0 + (np.arange(nv) + 0.5) * cell
    grid_u, grid_v = np.meshgrid(centre_u, centre_v)
    grid_x, grid_y = cs * grid_u - sn * grid_v, sn * grid_u + cs * grid_v
    reading = coef[0] + coef[1] * grid_x + coef[2] * grid_y + median_residual
    good = seen & np.isfinite(reading)

    def shrink(rect: "tuple[int, int, int, int]") -> "tuple[int, int, int, int] | None":
        r0, r1, c0, c1 = rect
        sub = footprint[r0:r1 + 1, c0:c1 + 1]
        if not sub.any():
            return None
        rr, cc = np.nonzero(sub)
        return (r0 + int(rr.min()), r0 + int(rr.max()), c0 + int(cc.min()), c0 + int(cc.max()))

    def fit(rect: "tuple[int, int, int, int]") -> "tuple[np.ndarray, float]":
        r0, r1, c0, c1 = rect
        sel = good[r0:r1 + 1, c0:c1 + 1]
        xs = grid_x[r0:r1 + 1, c0:c1 + 1][sel]
        ys = grid_y[r0:r1 + 1, c0:c1 + 1][sel]
        zs = reading[r0:r1 + 1, c0:c1 + 1][sel]
        local = coef
        if xs.size >= 3 and (np.ptp(xs) > 0.0 or np.ptp(ys) > 0.0):
            ws = count[r0:r1 + 1, c0:c1 + 1][sel].astype(np.float64)
            trial = _plane_fit(xs, ys, zs, np.maximum(ws, 1.0))
            if _tilt_deg(trial) <= MAX_SUPPORT_TILT_DEG:
                local = trial
        excess = (float(np.percentile(zs - (local[0] + local[1] * xs + local[2] * ys), _EXCESS_PERCENTILE))
                  if xs.size else 0.0)
        return local, excess

    budget = int(min(_SOLIDS_PER_SURFACE, max(1, math.ceil(float(footprint.sum()) * cell * cell
                                                             / _AREA_PER_SOLID_MM2))))
    first_rect = shrink((0, nv - 1, 0, nu - 1))
    queue = [first_rect] if first_rect is not None else []
    done: list[tuple[tuple[int, int, int, int], np.ndarray, float]] = []
    dropped = 0
    while queue:
        rect = queue.pop(0)
        local, excess = fit(rect)
        r0, r1, c0, c1 = rect
        long_rows = (r1 - r0) >= (c1 - c0)
        can_split = (r1 - r0 >= 1) if long_rows else (c1 - c0 >= 1)
        sub = footprint[r0:r1 + 1, c0:c1 + 1]
        lower_inside = bool(lower_cells[r0:r1 + 1, c0:c1 + 1].any())
        must = excess > _PIECE_FLAT_MM or lower_inside or float(sub.mean()) < _MIN_FILL
        if must and can_split and len(done) + len(queue) + 2 <= budget:
            if long_rows:
                middle = (r0 + r1) // 2
                halves = [(r0, middle, c0, c1), (middle + 1, r1, c0, c1)]
            else:
                middle = (c0 + c1) // 2
                halves = [(r0, r1, c0, middle), (r0, r1, middle + 1, c1)]
            for half in halves:
                kept = shrink(half)
                if kept is not None:
                    queue.append(kept)
            continue
        if bool(windows[r0:r1 + 1, c0:c1 + 1].any()):
            # H8: a piece that would cover what was seen lower inside the surface, a frame's window, is not built once
            # it can no longer be split. Its pixels stay obstacles, as they were before supports were found.
            dropped += 1
            continue
        done.append((rect, local, excess))

    padded = np.zeros((nv + 2, nu + 2), dtype=bool)
    padded[1:-1, 1:-1] = footprint
    pieces: list[SupportSolid] = []
    e_u = np.array([cs, sn, 0.0])
    for (r0, r1, c0, c1), local, excess in done:
        cells_mask = np.zeros((nv, nu), dtype=bool)
        cells_mask[r0:r1 + 1, c0:c1 + 1] = True
        cells_mask &= footprint
        sel = cells_mask[ov, ou]
        lo_u, hi_u = u0 + c0 * cell, u0 + (c1 + 1) * cell
        lo_v, hi_v = v0 + r0 * cell, v0 + (r1 + 1) * cell
        if sel.any():
            # An outer side, with no footprint beyond it, ends at the surface's own pixels grown by the margin.
            pu, pv = uv[sel, 0], uv[sel, 1]
            if not padded[r0 + 1:r1 + 2, c0].any():
                lo_u = max(lo_u, float(pu.min())) - margin
            if not padded[r0 + 1:r1 + 2, c1 + 2].any():
                hi_u = min(hi_u, float(pu.max())) + margin
            if not padded[r0, c0 + 1:c1 + 2].any():
                lo_v = max(lo_v, float(pv.min())) - margin
            if not padded[r1 + 2, c0 + 1:c1 + 2].any():
                hi_v = min(hi_v, float(pv.max())) + margin
        corners_uv = np.array([[lo_u, lo_v], [hi_u, lo_v], [hi_u, hi_v], [lo_u, hi_v]])
        corners = np.column_stack((cs * corners_uv[:, 0] - sn * corners_uv[:, 1],
                                   sn * corners_uv[:, 0] + cs * corners_uv[:, 1]))
        top = local[0] + local[1] * corners[:, 0] + local[2] * corners[:, 1] + excess + surface_band
        normal = np.array([-local[1], -local[2], 1.0])
        normal /= np.linalg.norm(normal)
        axis_u = e_u - (e_u @ normal) * normal
        axis_u /= np.linalg.norm(axis_u)
        rotation = np.column_stack((axis_u, np.cross(normal, axis_u), normal))
        # Down to the declared bench where the surface stands over it; where its top lies under the bench, the bench
        # solid holds that corner and this one is not lifted to reach it.
        ends = np.vstack((np.column_stack((corners, top)), np.column_stack((corners, np.minimum(plane, top - 1.0)))))
        in_box = ends @ rotation
        low_corner, high_corner = in_box.min(axis=0), in_box.max(axis=0)
        pieces.append(SupportSolid(
            name="", kind="support", surface=-2, centre_mm=rotation @ ((low_corner + high_corner) / 2.0),
            half_extents_mm=(high_corner - low_corner) / 2.0, rotation=rotation,
            local_plane=np.asarray(local, dtype=np.float64), excess_mm=float(excess), band_mm=float(surface_band),
            allowance_mm=0.0, cells=int(cells_mask.sum()),
        ))
    heights = coef[0] + coef[1] * mine[:, 0] + coef[2] * mine[:, 1]
    why = (f"{dropped} piece(s) of a surface at {region['z50']:.0f} mm would have covered a window where something "
           "lower was seen inside it, and are not held" if dropped else "")
    return pieces, coef, {"z_range": (float(heights.min()), float(heights.max())), "band": surface_band,
                          "own": int(own.sum())}, why


def _bench_solid(limits: "WorldBuildLimits", band: float, margin: float, allowance: float, prefix: str) -> SupportSolid:
    """The declared bench as a solid: upright over the workspace box grown by the margin, from 50 mm under the plane up
    to the plane, the largest band and the allowance. The guard never held a bench before (design finding 1)."""
    assert limits.support_plane_top_mm is not None  # noqa: S101
    plane = float(limits.support_plane_top_mm)
    top = plane + band + allowance
    bottom = plane - _BENCH_DEPTH_MM
    x0, x1 = float(limits.x_mm[0]) - margin, float(limits.x_mm[1]) + margin
    y0, y1 = float(limits.y_mm[0]) - margin, float(limits.y_mm[1]) + margin
    return SupportSolid(
        name=f"{prefix}bench", kind="bench", surface=-1,
        centre_mm=np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0, (top + bottom) / 2.0]),
        half_extents_mm=np.array([(x1 - x0) / 2.0, (y1 - y0) / 2.0, (top - bottom) / 2.0]),
        rotation=np.eye(3), local_plane=np.array([plane, 0.0, 0.0]), excess_mm=0.0, band_mm=float(band),
        allowance_mm=float(allowance),
    )


def _bench_cells(
    grid: _CellGrid, limits: "WorldBuildLimits", plane: float, band: float,
) -> "tuple[np.ndarray, np.ndarray] | None":
    """The flat cells of the workspace box that read within the band plus 10 mm of the declared plane: their BASE
    centres and their readings against the plane."""
    flat = np.nonzero(grid.flat)[0]
    if flat.size == 0:
        return None
    centres = grid.centres(flat)
    offsets = grid.level[flat] - plane
    inside = ((centres[:, 0] >= float(limits.x_mm[0])) & (centres[:, 0] <= float(limits.x_mm[1]))
              & (centres[:, 1] >= float(limits.y_mm[0])) & (centres[:, 1] <= float(limits.y_mm[1]))
              & (np.abs(offsets) <= band + _BENCH_CHECK_EXTRA_MM))
    return centres[inside], offsets[inside]


def _bench_reading(
    cells: "tuple[np.ndarray, np.ndarray] | None", supports: "Sequence[SupportSolid]", allowance: float,
) -> BenchReading | None:
    """How the bare bench reads: the bench check's cells that lie in no support, against the declared plane."""
    if cells is None:
        return None
    centres, offsets = cells
    free = np.ones(centres.shape[0], dtype=bool)
    for solid in supports:
        free &= ~solid.covers_xy(centres)
    centres, offsets = centres[free], offsets[free]
    if offsets.size == 0:
        return None
    low = offsets < -float(allowance)
    where = None
    low_median = None
    if low.any():
        at = centres[low]
        where = (float(at[:, 0].min()), float(at[:, 0].max()), float(at[:, 1].min()), float(at[:, 1].max()))
        low_median = float(np.median(offsets[low]))
    return BenchReading(cells=int(offsets.size), median_mm=float(np.median(offsets)),
                        p10_mm=float(np.percentile(offsets, 10.0)), low_cells=int(low.sum()), low_where_mm=where,
                        low_median_mm=low_median, allowance_mm=float(allowance))
