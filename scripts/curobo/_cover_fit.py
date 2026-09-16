"""Cover a body with spheres that never leave a hole, and never reach further past it than they were told.

A collision sphere model makes two kinds of error, and only one of them is safe.

A hole is a false clear. Where the spheres do not cover the body, the planner believes that part of the arm is not
there. It clears configurations the exact mesh guard then refuses, and on a cell without that guard it clears
configurations where the arm actually is. There is no upper bound on what that costs.

Reach is a false collide. Where a sphere sticks out past the body, the planner refuses configurations that are fine.
It costs picks and reachable workspace, and nothing else.

The vendor's map optimises a compromise between the two and measures the result afterwards: the family it ships
reaches 0 to 61 mm past its own links and leaves gaps up to 45 mm once it is clamped. This fitter makes the safe half
a property of the construction. Every surface sample ends up inside some sphere by at least the sample spacing, or
there is no fit at all. What the caller then trades is count against reach, which is a planning cost: cuRobo's plan
time doubles up to about 600 spheres and grows roughly quadratically past it.

How it works, per body:

1. Seeds are surface samples. Each is marched inward along its own normal. A step is admissible when the
   sphere ``r = u(c(t)) + reach`` still passes the seed with the spacing to spare, which is
   ``t - u <= reach - spacing``; among those, the step with the largest ``u`` is kept, because that is the
   biggest sphere and the deepest step is not it: ``u`` rises to the medial axis and falls away again, so the
   deepest admissible step sits ``(reach - spacing) / 2`` past the best one. Both are equally safe: ``u`` is the
   unsigned distance to the body, so ``B(c, u)`` lies inside it and a sphere of ``u + reach`` about the same
   centre lies at most ``reach`` outside, whatever the body's shape.
2. A march must not walk through the body: past the far wall the distance grows again and the test above starts
   passing for the wrong reason. So the march stops at the first return to the surface after it started, and a centre
   the inside test does not confirm falls back to a sphere of ``reach`` sitting on the surface.
3. Lazy greedy set cover over the candidates, counting a sample covered only when it is inside by the spacing.
   Samples nothing covers seed another round of candidates of their own.

The body is an oracle, not a mesh: ``distance(points)`` and ``inside(points)``. On the box those come from the
planner's own mesh query; where exact arithmetic on a box, a plate and a sphere answers instead, what is measured is
this algorithm rather than somebody's mesh library.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

__all__ = ["CoverFit", "cover_body", "unit_directions"]

#: How many candidate seeds one round draws, at most. A round that cannot cover everything seeds another from what is
#: left, so this bounds memory rather than quality.
_SEEDS_PER_ROUND = 4096

#: Rounds before the fitter gives up on a body. Reaching it means the seeds cannot cover something, which is a body
#: to look at rather than a number to raise.
_MAX_ROUNDS = 50


@dataclass(frozen=True)
class CoverFit:
    """Spheres covering one body, and what the cover cost."""

    centres_m: np.ndarray
    radii_m: np.ndarray
    reach_m: float
    #: The largest distance from a surface sample to the union, as a signed number: negative means every sample is
    #: inside a sphere by that much, which is the only outcome this fitter returns.
    uncovered_max_m: float
    rounds: int
    #: Seeds whose march never found an inside, so they became a sphere of ``reach`` sitting on the surface. A body
    #: with many of them is thin somewhere, which the record carries rather than hides.
    flat: int

    def __len__(self) -> int:
        return len(self.radii_m)

    def render(self) -> str:
        radii_mm = self.radii_m * 1000.0
        return (f"{len(self)} spheres at reach {self.reach_m * 1000.0:.1f} mm, radii "
                f"{radii_mm.min():.1f}/{np.median(radii_mm):.1f}/{radii_mm.max():.1f} mm, "
                f"every sample inside by {-self.uncovered_max_m * 1000.0:.2f} mm, {self.flat} flat, "
                f"{self.rounds} round(s)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "spheres": len(self),
            "reach_mm": self.reach_m * 1000.0,
            "radius_min_mm": float(self.radii_m.min() * 1000.0),
            "radius_median_mm": float(np.median(self.radii_m) * 1000.0),
            "radius_max_mm": float(self.radii_m.max() * 1000.0),
            "uncovered_max_mm": self.uncovered_max_m * 1000.0,
            "rounds": self.rounds,
            "flat": self.flat,
        }


def unit_directions(count: int = 128) -> np.ndarray:
    """``count`` directions spread over the unit sphere, for measuring how far a sphere reaches past a body."""
    index = np.arange(count, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * index / count)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * index
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def _march(
    seeds: np.ndarray,
    normals: np.ndarray,
    *,
    distance: "Callable[[np.ndarray], np.ndarray]",
    inside: "Callable[[np.ndarray], np.ndarray]",
    reach_m: float,
    spacing_m: float,
    span_m: float,
) -> "tuple[np.ndarray, np.ndarray, int]":
    """One candidate sphere per seed: centre, radius, and how many seeds found no inside at all."""
    inward = -np.asarray(normals, dtype=np.float64)
    step = min(2.0 * spacing_m, reach_m)
    goes_in = inside(seeds + inward * step)
    turned = ~goes_in
    if turned.any():  # a face wound the other way: the body is on the other side of it
        other = inside(seeds[turned] - inward[turned] * step)
        index = np.nonzero(turned)[0][other]
        inward[index] *= -1.0
        goes_in[index] = True

    centres = np.array(seeds, dtype=np.float64, copy=True)
    radii = np.full(len(seeds), reach_m, dtype=np.float64)
    steps = np.arange(0.0, span_m, spacing_m / 2.0)
    live = np.nonzero(goes_in)[0]
    for start in range(0, len(live), 256):
        block = live[start:start + 256]
        walk = seeds[block, None, :] + steps[None, :, None] * inward[block, None, :]
        shape = walk.shape[:2]
        u = distance(walk.reshape(-1, 3)).reshape(shape)
        # Two rules bound which steps of the walk are admissible. The sphere has to cover its own seed with the
        # spacing to spare, which is `t <= radius - spacing`, that is `t - u <= reach - spacing`. And the walk
        # must not pass through the far wall: past it the distance grows again and the first rule starts
        # passing for the wrong reason, so the first return to the surface ends it.
        returned = (u < 0.5 * spacing_m) & (steps[None, :] > 2.0 * spacing_m)
        left_at = np.where(returned.any(axis=1), returned.argmax(axis=1), len(steps))
        too_far = ((steps[None, :] - u) > (reach_m - spacing_m)) | (np.arange(len(steps))[None, :] >= left_at[:, None])
        first = np.where(too_far.any(axis=1), too_far.argmax(axis=1), len(steps))
        # Among the admissible steps, keep the one with the largest sphere, not the deepest one.
        # The reach bound does not care which is chosen: a centre inside the body at unsigned distance u has
        # the whole ball B(c, u) inside the body, so a sphere of radius u + reach around it lies at most
        # `reach` outside, whatever t is. What the depth changes is the radius, and u rises to the medial
        # axis and falls again, so the deepest admissible step is past the best one by (reach - spacing) / 2.
        # Taking the deepest step instead, on the ur5e upper arm at 12 mm reach, gives a median radius of
        # 30.8 mm on a body whose axis is about 60 mm from the surface, and needs 477 spheres to cover it.
        admissible = np.arange(len(steps))[None, :] < np.maximum(first, 1)[:, None]
        best = np.where(admissible, u, -np.inf).argmax(axis=1)
        rows = np.arange(len(block))
        centres[block] = walk[rows, best]
        radii[block] = u[rows, best] + reach_m

    flat = int((~goes_in).sum())
    if len(live):  # the reach bound holds only for a centre that is inside; anything else sits back on the surface
        stray = live[~inside(centres[live])]
        centres[stray] = seeds[stray]
        radii[stray] = reach_m
        flat += len(stray)
    return centres, radii, flat


def _greedy(
    samples: np.ndarray,
    covered: np.ndarray,
    centres: np.ndarray,
    radii: np.ndarray,
    *,
    spacing_m: float,
    cap: int,
) -> list[int]:
    """Lazy greedy set cover: take the sphere covering the most uncovered samples, until nothing is left or cap is."""
    reach = radii - spacing_m
    gains = np.zeros(len(centres), dtype=np.int64)
    pending = samples[~covered]
    for start in range(0, len(centres), 64):
        block = slice(start, start + 64)
        distances = np.linalg.norm(pending[None, :, :] - centres[block][:, None, :], axis=2)
        gains[block] = (distances <= reach[block][:, None]).sum(axis=1)

    heap = [(-int(gain), index) for index, gain in enumerate(gains) if gain > 0]
    heapq.heapify(heap)
    picked: list[int] = []
    while heap and len(picked) < cap:
        _, index = heapq.heappop(heap)
        mask = (~covered) & (np.linalg.norm(samples - centres[index][None, :], axis=1) <= reach[index])
        gain = int(mask.sum())
        if gain == 0:
            continue
        if heap and gain < -heap[0][0]:  # stale: another candidate now covers more
            heapq.heappush(heap, (-gain, index))
            continue
        covered |= mask
        picked.append(index)
    return picked


def cover_body(
    samples: np.ndarray,
    normals: np.ndarray,
    *,
    distance: "Callable[[np.ndarray], np.ndarray]",
    inside: "Callable[[np.ndarray], np.ndarray]",
    reach_m: float,
    spacing_m: float,
    cap: int,
    seed: int = 0,
    seeds_per_round: int = _SEEDS_PER_ROUND,
) -> "CoverFit | None":
    """Spheres covering every sample by ``spacing_m``, reaching at most ``reach_m`` past the body, or ``None``.

    ``None`` means it could not be done within ``cap`` spheres at this reach, and is the answer a caller walks a
    ladder of reaches against. It is never a partial cover: half a cover is the dangerous answer, because it looks
    like a fit and leaves holes.
    """
    points = np.asarray(samples, dtype=np.float64)
    directions = np.asarray(normals, dtype=np.float64)
    if len(points) != len(directions) or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"samples and normals must be matching (n, 3) arrays, got {points.shape} and {directions.shape}")
    if reach_m < 2.0 * spacing_m:
        # Below this the margin a candidate has to work with is thinner than the grid it is judged on. A corner
        # sphere then covers its own seed by exactly zero and rounding decides whether the body has a hole in it.
        raise ValueError(
            f"a reach of {reach_m * 1000.0:.1f} mm against a sample spacing of {spacing_m * 1000.0:.1f} mm leaves "
            f"a candidate no room: sample the body finer, or accept more reach"
        )
    rng = np.random.default_rng(seed)
    span = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) or (4.0 * reach_m)

    covered = np.zeros(len(points), dtype=bool)
    kept_centres: list[np.ndarray] = []
    kept_radii: list[float] = []
    flat = 0
    pool = np.arange(len(points))
    if len(pool) > seeds_per_round:
        pool = rng.choice(pool, size=seeds_per_round, replace=False)
    for round_index in range(1, _MAX_ROUNDS + 1):
        centres, radii, round_flat = _march(
            points[pool], directions[pool], distance=distance, inside=inside,
            reach_m=reach_m, spacing_m=spacing_m, span_m=0.5 * span,
        )
        flat += round_flat
        picked = _greedy(points, covered, centres, radii, spacing_m=spacing_m, cap=cap - len(kept_centres) + 1)
        kept_centres.extend(centres[index] for index in picked)
        kept_radii.extend(float(radii[index]) for index in picked)
        if len(kept_centres) > cap:
            return None
        left = np.nonzero(~covered)[0]
        if len(left) == 0:
            fit_centres = np.asarray(kept_centres, dtype=np.float64)
            fit_radii = np.asarray(kept_radii, dtype=np.float64)
            gaps = np.min(
                np.linalg.norm(points[:, None, :] - fit_centres[None, :, :], axis=2) - fit_radii[None, :], axis=1,
            )
            return CoverFit(
                centres_m=fit_centres, radii_m=fit_radii, reach_m=float(reach_m),
                uncovered_max_m=float(gaps.max()), rounds=round_index, flat=flat,
            )
        pool = left if len(left) <= seeds_per_round else rng.choice(left, size=seeds_per_round, replace=False)
    raise RuntimeError(
        f"{len(np.nonzero(~covered)[0])} sample(s) of this body are still uncovered after {_MAX_ROUNDS} rounds at "
        f"reach {reach_m * 1000.0:.1f} mm: look at the body rather than at the round count"
    )
