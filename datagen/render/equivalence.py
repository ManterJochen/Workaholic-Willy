"""Tier 0 of the engine equivalence gate: can a backend lie about geometry?

Why this exists before any second engine does. A backend that produces a subtly different corpus is
worse than no second backend, because the difference is invisible in the `.npz`: the file carries
points and labels, not the transform that made them. It would then confound engine with scene family
in every number drawn from it, silently and permanently.

The checks grade on an oblique CAMERA, and that is the whole design. A principal-point offset of one
pixel moves reconstructed geometry by roughly 0.5 to 2 mm at working distance, and:

* an overhead view is structurally blind to it: a principal-point shift cannot change recovered Z on
  a plane normal to the optical axis, so the error is exactly zero there for any offset;
* the repo's existing `_DEPTH_MATCH_TOLERANCE_MM = 2.0` compares two renders of the same geometry,
  so a constant offset cancels and cannot be seen;
* and solo-vs-scene depth differencing compares a backend against itself.

So all three of the checks datagen already has would pass a backend whose camera is wrong by a pixel.
This module is the one that would not, and running it on an overhead view turns it into a check that
cannot fail.

A gate nobody has seen reject anything is decoration. Exercise these checks against deliberately
broken backends (a shifted principal point, a scaled depth, a mask that disagrees with its own depth)
and confirm each one is caught.

Tier 0 is what one frame can settle without a GPU box. Judging that two engines settle the same
population, against bars fixed in advance, and re-earning the physics labeller's four controls from
scratch both need a second backend and a GPU box respectively, and neither lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from datagen.render.camera import intrinsics_matrix, look_at_camera_to_base, unproject_to_base

__all__ = ["EquivalenceFinding", "check_depth_reconstructs_geometry", "check_mask_matches_depth",
           "LANDMARK_TOLERANCE_MM", "check_landmark_reconstructs",
           "check_points_match_analytic_rays", "oblique_camera",
           "run_tier0"]

#: The bar for Tier 0's geometry check. Tight on purpose and not comparable to the repo's 2.0 mm
#: depth-match tolerance, which answers a different and much easier question (two renders of the same
#: geometry). 0.1 mm is far below the smallest thing the corpus resolves and far above float noise.
GEOMETRY_TOLERANCE_MM = 0.1

#: The landmark check's own bar, deliberately looser and not a weakening. It compares the centre
#: of a reconstructed pixel bounding box against a position the layout chose, so it carries the
#: quantisation of the mask edge: at 320x240 an exact backend reads 0.1241 mm purely from which
#: pixels the edge fell into. The defect it exists to catch is an order of magnitude larger:
#: 2.09 mm for a one-pixel principal-point error and 8.37 mm for four.
LANDMARK_TOLERANCE_MM = 0.5


@dataclass(frozen=True, slots=True)
class EquivalenceFinding:
    """One check's verdict. ``passed`` is the gate; ``detail`` is what an operator acts on."""

    check: str
    passed: bool
    measured: float
    tolerance: float
    detail: str = ""

    def __str__(self) -> str:                                    # pragma: no cover (display only)
        mark = "ok  " if self.passed else "FAIL"
        return f"  [{mark}] {self.check:34s} {self.measured:10.4f} (bar {self.tolerance:g})  {self.detail}"


def oblique_camera(resolution: tuple[int, int] = (640, 480), horizontal_fov_deg: float = 60.0,
                   eye_mm: tuple[float, float, float] = (420.0, -380.0, 610.0),
                   look_at_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
                   ) -> tuple[np.ndarray, np.ndarray]:
    """``(camera_to_base, K)`` for a view that is not down the world Z axis.

    The obliquity is the instrument, not a detail. With the camera looking straight down, a
    principal-point offset produces exactly 0.0 mm of reconstruction error at every offset, because
    the depth of a plane normal to the axis does not depend on where the axis meets it. Tilt it and
    the same offset shows up immediately. The default eye is off-axis in x, y and z for that reason.
    """
    return (look_at_camera_to_base(np.asarray(eye_mm, dtype=np.float64),
                                   np.asarray(look_at_mm, dtype=np.float64)),
            intrinsics_matrix(resolution, horizontal_fov_deg))


def check_depth_reconstructs_geometry(depth_mm: np.ndarray, mask: np.ndarray,
                                      camera_to_base: np.ndarray, intrinsics: np.ndarray,
                                      *, plane_z_mm: float,
                                      tolerance_mm: float = GEOMETRY_TOLERANCE_MM,
                                      ) -> EquivalenceFinding:
    """Does this backend's depth, read through the repo's own camera model, land on the known plane?

    The reference is a fact about the WORLD (the table is at ``plane_z_mm``), not about another
    render, which is what makes it able to catch a constant error. Everything else datagen checks
    compares a backend against itself.

    It sees only the component along the plane normal, which is a real limit. `look_at` with
    ``up = (0, 0, 1)`` gives the camera an exactly horizontal x-axis, so a principal-point offset in
    the column direction displaces the reconstruction purely sideways: 0.0000 mm of z-error at 4 px
    of offset, while a row offset of one pixel shows up as 1.714 mm. Use
    :func:`check_points_match_analytic_rays` for the full three-dimensional error; this one is the
    fallback for when only the plane is known.
    """
    points = unproject_to_base(depth_mm, mask, camera_to_base, intrinsics)
    if not len(points):
        return EquivalenceFinding("depth reconstructs geometry", False, float("inf"), tolerance_mm,
                                  "no masked depth pixels; the backend rendered nothing")
    error = float(np.max(np.abs(points[:, 2] - plane_z_mm)))
    return EquivalenceFinding(
        "depth reconstructs geometry", error <= tolerance_mm, error, tolerance_mm,
        f"{len(points)} point(s); worst |z - {plane_z_mm:.1f}| on an OBLIQUE view")


def check_points_match_analytic_rays(depth_mm: np.ndarray, mask: np.ndarray,
                                     camera_to_base: np.ndarray, intrinsics: np.ndarray,
                                     *, plane_z_mm: float,
                                     tolerance_mm: float = GEOMETRY_TOLERANCE_MM,
                                     ) -> EquivalenceFinding:
    """The full three-dimensional error: where each pixel's ray truly meets the plane, versus where
    the backend's depth puts it.

    This is the check that sees everything, and it exists because the plane check does not. A
    one-pixel principal-point offset in the column direction produces exactly 0.0000 mm of z-error
    against the plane, because the repo's `look_at` gives the camera a horizontal x-axis and that
    defect is a pure sideways slide: the cloud is 1.4 mm out of place and every z is perfect.
    Comparing against the analytically known intersection catches both components at once.

    What it cannot see is fundamental rather than a gap to close here: a camera error that is
    consistent with itself. Both the reconstruction and the expectation are computed from the K the
    backend reports, so a backend that renders through one principal point and reports that same
    wrong principal point agrees with itself perfectly, at 0.0000 mm for a 4 px offset in cx. A
    camera cannot be caught with its own parameters. That needs a WORLD reference:
    :func:`check_landmark_reconstructs`.
    """
    depth = np.asarray(depth_mm, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool) & (depth > 0.0)
    rows, columns = np.nonzero(selected)
    if rows.size == 0:
        return EquivalenceFinding("points match analytic rays", False, float("inf"), tolerance_mm,
                                  "no masked depth pixels; the backend rendered nothing")
    reconstructed = unproject_to_base(depth, selected, camera_to_base, intrinsics)

    k = np.asarray(intrinsics, dtype=np.float64)
    transform = np.asarray(camera_to_base, dtype=np.float64)
    directions = np.stack([(columns - k[0, 2]) / k[0, 0], (rows - k[1, 2]) / k[1, 1],
                           np.ones_like(rows, dtype=np.float64)], axis=1)
    world = directions @ transform[:3, :3].T
    origin = transform[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (plane_z_mm - origin[2]) / world[:, 2]
    usable = np.isfinite(t) & (t > 0.0)
    if not usable.any():
        return EquivalenceFinding("points match analytic rays", False, float("inf"), tolerance_mm,
                                  "no pixel's ray meets the plane in front of the camera")
    expected = origin + t[usable, None] * world[usable]
    error = float(np.max(np.linalg.norm(reconstructed[usable] - expected, axis=1)))
    return EquivalenceFinding(
        "points match analytic rays", error <= tolerance_mm, error, tolerance_mm,
        f"{int(usable.sum())} ray(s); worst 3-D displacement, both components")


def check_landmark_reconstructs(depth_mm: np.ndarray, mask: np.ndarray,
                                camera_to_base: np.ndarray, intrinsics: np.ndarray,
                                *, landmark_centre_mm: np.ndarray,
                                tolerance_mm: float = LANDMARK_TOLERANCE_MM,
                                ) -> EquivalenceFinding:
    """Does a body whose true position the WORLD knows reconstruct to where it actually is?

    The only check here that can catch a self-consistent CAMERA, and the reason the other two are not
    enough: a backend that renders through a shifted principal point and reports that same shifted
    point passes both of them at 0.0000 mm, because each computes its expectation from the K the
    backend handed over. The reference has to come from outside the camera.

    ``mask`` selects the landmark's pixels and ``landmark_centre_mm`` is where the scene spec says
    its centre is, a number the layout chose rather than one the renderer produced. The two disagree
    exactly when the backend's camera is wrong, however consistently.

    The centroid of the visible pixels is not the centroid of the body, because a camera sees one
    side. This compares against the centre of the reconstructed points' bounding box in the two
    directions the view resolves, which is what a plane-mounted landmark permits. For a real backend
    the landmark should be a body whose full footprint the view sees.
    """
    points = unproject_to_base(depth_mm, mask, camera_to_base, intrinsics)
    if not len(points):
        return EquivalenceFinding("landmark lands where the world says", False, float("inf"),
                                  tolerance_mm, "the landmark has no pixels; nothing was rendered")
    centre = 0.5 * (points.min(axis=0) + points.max(axis=0))
    error = float(np.max(np.abs(centre - np.asarray(landmark_centre_mm, dtype=np.float64))))
    return EquivalenceFinding(
        "landmark lands where the world says", error <= tolerance_mm, error, tolerance_mm,
        f"{len(points)} point(s); worst axis error against the SPEC's position")


def check_mask_matches_depth(instance_map: np.ndarray, depth_mm: np.ndarray,
                             background_depth_mm: np.ndarray) -> EquivalenceFinding:
    """Does the backend's native instance mask agree with what its own depth says is in front?

    These are two independent claims a backend makes about the same frame, and a mismatch is the
    signature of an id space that belongs to a different frame. That is why datagen derives masks by
    depth differencing rather than trusting an engine's own segmentation annotator. IoU 1.0 or it is
    not the same scene twice.
    """
    occupied = np.asarray(instance_map) > 0
    nearer = np.asarray(depth_mm, dtype=np.float64) < (
        np.asarray(background_depth_mm, dtype=np.float64) - 1e-6)
    union = int(np.logical_or(occupied, nearer).sum())
    if union == 0:
        return EquivalenceFinding("mask matches its own depth", False, 0.0, 1.0,
                                  "nothing occupied in either; an empty frame proves nothing")
    iou = float(np.logical_and(occupied, nearer).sum()) / union
    return EquivalenceFinding("mask matches its own depth", iou >= 1.0, iou, 1.0,
                              f"{union} pixel(s) in the union")


def check_repeatable(render_once: Any, *, runs: int = 2) -> EquivalenceFinding:
    """Does calling the backend again give the same answer?

    ``render_once`` is a zero-argument callable returning a numpy array. In-process only. A resume is
    a new process, so an engine that seeds from process state passes this check and still produces a
    different corpus on restart; that case has to be checked across processes.
    """
    first = np.asarray(render_once(), dtype=np.float64)
    worst = 0.0
    for _ in range(runs - 1):
        again = np.asarray(render_once(), dtype=np.float64)
        if again.shape != first.shape:
            return EquivalenceFinding("repeatable in one process", False, float("inf"), 0.0,
                                      f"shape moved {first.shape} -> {again.shape}")
        worst = max(worst, float(np.max(np.abs(again - first))))
    return EquivalenceFinding("repeatable in one process", worst == 0.0, worst, 0.0,
                              f"{runs} call(s)")


def run_tier0(findings: list[EquivalenceFinding]) -> tuple[bool, str]:
    """``(admissible, report)``. A backend is admissible only when every check passes.

    No partial credit and no weighting: each of these answers a question with a right answer, and a
    backend that fails one of them is producing a corpus that reads as fine.
    """
    lines = [str(f) for f in findings]
    failed = [f.check for f in findings if not f.passed]
    if failed:
        lines.append(f"  NOT ADMISSIBLE; failed: {', '.join(failed)}")
    else:
        lines.append(f"  admissible on Tier 0 ({len(findings)} check(s))")
    return not failed, "\n".join(lines)
