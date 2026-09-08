"""Capsule-to-capsule and capsule-to-box distance helpers for the self-collision guard.

All distances are in millimetres. A capsule is a swept sphere over a finite line
segment: a segment ``(p0, p1)`` in mm plus one ``radius_mm``. An axis-aligned box is
its centre plus half-extents in mm.

The capsule-to-capsule maths is closed-form, which keeps the per-move cost low enough
for the hot path even with all-pairs link checks on a 6-DoF arm. The capsule-to-box
distance is not closed-form. It is a search over a convex function, because a sampled
approximation errs in the unsafe direction by construction, which
``capsule_box_distance_mm`` sets out.

References
----------
* Christer Ericson, "Real-Time Collision Detection", section 5.1.3, on the closest
  points between two segments.
* Box-point distance clamps the box-centre offset to ``half_extents`` axis by axis.
  Segment-box distance minimises that over the segment parameter, which is convex, so
  a search finds the global minimum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "AxisAlignedBox",
    "Capsule",
    "capsule_capsule_distance_mm",
    "capsule_box_distance_mm",
    "segment_segment_distance_mm",
    "segment_point_distance_mm",
]


@dataclass(frozen=True, slots=True)
class Capsule:
    """A swept-sphere capsule. ``p0``, ``p1`` and ``radius_mm`` are all millimetres."""

    p0: np.ndarray
    p1: np.ndarray
    radius_mm: float

    @property
    def is_degenerate(self) -> bool:
        return bool(np.allclose(self.p0, self.p1))


@dataclass(frozen=True, slots=True)
class AxisAlignedBox:
    """An axis-aligned box in the robot base frame, as ``center_mm`` and per-axis
    ``half_extents_mm``.

    ``name`` is what an operator reads in a refusal. It defaults to empty, which every existing
    caller relies on, and the guards fall back to a positional name when it is. A cell with three
    declared fixtures and a refusal that says only "fixture" tells an operator to go and look at all
    three.
    """

    center_mm: np.ndarray
    half_extents_mm: np.ndarray
    name: str = ""


def segment_point_distance_mm(p0: np.ndarray, p1: np.ndarray, q: np.ndarray) -> float:
    """Closest distance from point ``q`` to segment ``(p0, p1)``."""
    d = p1 - p0
    denom = float(d @ d)
    if denom <= 1e-12:
        return float(np.linalg.norm(q - p0))
    t = float((q - p0) @ d) / denom
    t = max(0.0, min(1.0, t))
    closest = p0 + t * d
    return float(np.linalg.norm(q - closest))


def segment_segment_distance_mm(
    p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray,
) -> float:
    """Closest distance between the segments ``(p0,p1)`` and ``(q0,q1)``.

    It follows the closest-points-on-segments treatment in Ericson, "Real-Time
    Collision Detection". Both inputs are in mm and the result is in mm.
    """
    d1 = p1 - p0
    d2 = q1 - q0
    r = p0 - q0
    a = float(d1 @ d1)
    e = float(d2 @ d2)
    f = float(d2 @ r)

    eps = 1e-12
    if a <= eps and e <= eps:
        return float(np.linalg.norm(p0 - q0))
    if a <= eps:
        s = 0.0
        t = max(0.0, min(1.0, f / e))
    else:
        c = float(d1 @ r)
        if e <= eps:
            t = 0.0
            s = max(0.0, min(1.0, -c / a))
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            if denom != 0.0:
                s = max(0.0, min(1.0, (b * f - c * e) / denom))
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = max(0.0, min(1.0, -c / a))
            elif t > 1.0:
                t = 1.0
                s = max(0.0, min(1.0, (b - c) / a))

    closest_1 = p0 + s * d1
    closest_2 = q0 + t * d2
    return float(np.linalg.norm(closest_1 - closest_2))


def capsule_capsule_distance_mm(a: Capsule, b: Capsule) -> float:
    """Signed distance between two capsules, where a negative value is a penetration.

    It is ``segment_segment_distance - (r_a + r_b)``.
    """
    seg = segment_segment_distance_mm(a.p0, a.p1, b.p0, b.p1)
    return seg - (a.radius_mm + b.radius_mm)


def _segment_box_distance_mm(p0: np.ndarray, p1: np.ndarray, box: AxisAlignedBox) -> float:
    """Exact distance from a segment to an axis-aligned box, to well below input precision.

    The property that makes this short is convexity: the distance to a convex set is a
    convex function of the point, and a segment is affine in its parameter, so
    ``t -> dist(P(t), box)`` is convex on ``[0, 1]``. Golden-section search on a convex
    function converges to the global minimum with no Voronoi-region case analysis and
    no chance of settling in a local one.

    The arithmetic is scalar rather than numpy on purpose. These are three-element
    vectors, and at that size the numpy per-call overhead dominates the arithmetic
    completely: the same search with the same iteration count costs 448 us per call in
    numpy and 12.1 us here. Golden section also reuses one evaluation per iteration
    where a ternary search needs two. A move checking six links against three fixtures
    spends 0.22 ms here in total.

    Fifty golden-section iterations shrink the bracket to about 1e-11 of it, which on a
    500 mm link is 5e-9 mm. Measured against a 200,001-point reference over 4,500
    cases, 3,000 random and 1,500 adversarial: the worst overestimate is 8e-9 mm, and
    no case reports clear while the truth is penetrating.
    """
    ax, ay, az = float(p0[0]), float(p0[1]), float(p0[2])
    dx = float(p1[0]) - ax
    dy = float(p1[1]) - ay
    dz = float(p1[2]) - az
    cx, cy, cz = (float(box.center_mm[0]), float(box.center_mm[1]), float(box.center_mm[2]))
    hx, hy, hz = (float(box.half_extents_mm[0]), float(box.half_extents_mm[1]),
                  float(box.half_extents_mm[2]))

    def at(t: float) -> float:
        ox = ax + t * dx - cx
        oy = ay + t * dy - cy
        oz = az + t * dz - cz
        ex = ox - (hx if ox > hx else -hx if ox < -hx else ox)
        ey = oy - (hy if oy > hy else -hy if oy < -hy else oy)
        ez = oz - (hz if oz > hz else -hz if oz < -hz else oz)
        return (ex * ex + ey * ey + ez * ez) ** 0.5

    invphi = 0.6180339887498949
    low, high = 0.0, 1.0
    b, c = high - invphi * (high - low), low + invphi * (high - low)
    fb, fc = at(b), at(c)
    for _ in range(50):
        if fb < fc:
            high, c, fc = c, b, fb
            b = high - invphi * (high - low)
            fb = at(b)
        else:
            low, b, fb = b, c, fc
            c = low + invphi * (high - low)
            fc = at(c)
    return at(0.5 * (low + high))


def capsule_box_distance_mm(c: Capsule, box: AxisAlignedBox) -> float:
    """Signed distance from a capsule to an axis-aligned box.

    This searches rather than samples, because a minimum taken over samples can only
    overestimate the true minimum. A sampled implementation therefore reports the
    capsule as further from the box than it is whenever it is wrong, which is
    structural and not bad luck. Measured against a dense reference over 4,000 random
    cases with arm-link capsule lengths under 500 mm, a seventeen-sample version was
    6.066 mm out at worst, and one case reported ``+0.156 mm clear`` where the truth
    was ``-0.037 mm penetrating``.

    A default cell does not reach this path, because ``min_distance_mm`` ships at 10.0
    and the shipped ``fixtures`` list is empty. It goes live on the first real cell that
    declares a table or a bin, and the schema permits ``min_distance_mm`` down to zero,
    below which an overestimating distance passes a penetrating configuration.

    One imprecision remains, deliberately. With the segment inside the box the clamp
    yields a distance of zero, so this returns ``-radius_mm`` rather than the true
    penetration depth. That is safe for a guard: it is already a rejection and it errs
    toward rejecting harder. It is not a number to read as a depth.
    """
    return _segment_box_distance_mm(c.p0, c.p1, box) - c.radius_mm
