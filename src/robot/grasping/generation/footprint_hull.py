"""Where the mask pixels whose depth the support-footprint stage does not read stand: in the visual hull of the part.

The support-footprint stage builds its footprint from the depth under a part's mask, and three kinds of mask pixel give
it no depth it can use. The rim (:mod:`footprint_rim`), where the D415 smears a far edge into a ramp and the cut takes
it off; the depth-step pixels, which measure the background behind the edge (:mod:`depth_steps`); and the holes, where
the D415 measured nothing: dark or shiny surfaces, the 20 % of an aluminium profile's pixels the review found empty
beside the owner's mat (2026-10-09), and bare metal more still. The colour mask is sharp at all three; the depth is
not. So each such pixel is given a point from the mask instead of from the depth.

The stage reconstructs a vertical prism: the footprint, extruded from the support to the part's top. A point of that
prism stands on a vertical from the support to the top, and the whole vertical is part of it, so the whole vertical
projects into the part's mask: the visual hull of a vertical prism in one view. A pixel's ray is met at the part's
top (the prism's own top, the 98th percentile of the points it read), and where that point's vertical, from just over
the floor to just under the top, projects into the mask, the point stands there. Where it does not, the ray is met
just over the floor, under the part, and kept on the same test. A pixel neither point passes for gives none: a
footprint is never grown past what its mask shows.

MEASURED offline (2026-10-09, desk, ``GraspCalculator`` on the 23 grey 40 mm cubes the owner's cell recorded on
2026-10-07, the cell's reproduced SAM2 masks, the colour top face as the reference, as the rim was measured): with the
2.0 mm rim cut and every mask pixel the stage does not read placed here, ``inflate_mm`` 0.0, the centre the grasp
closes on 0.58 mm off at the median and 1.79 at the 90th percentile, the footprint's short side 0.63 mm short and its
long side 0.07 mm short at the median, the long side's 90th percentile 4.9 mm long (the rim with ``inflate_mm`` 1.25:
0.77 and 2.26 mm, 0.53 short, 0.75 long, 6.8 at the 90th; neither: 2.53 and 4.44 mm). With blob holes knocked into the
cubes' depth, 3 to 8 px across, the same holes for both: 20 % of the mask's pixels, 1.02 and 1.91 mm, the short side
0.49 short (the rim alone: 0.77 and 2.24, 0.53 short); 35 %, 0.59 and 1.87 mm, 0.49 short (the rim alone: 0.87 and
2.28, 0.74 short). Better at the 90th percentile every time, at the median twice of three. The owner, 2026-10-09: what
the desk proves is on, without a switch of its own; it comes with the rim (``robot.grasping.geometry.footprint_rim_mm``),
and a calculator that cuts no rim places nothing.

Where the mask and the depth disagree, the hull follows the mask. The cell's hand-eye calibration poses its marker in
the colour image alone (``src/calibration/rgbd_marker_source.py``), so the colour image is the frame the arm is
calibrated in, and the depth aligned to it sits about 1.9 px off on the owner's D415 (the review, 2026-10-09). On a
ray-cast cube whose mask is drawn 2 px off a depth taken for the truth, the grasp centre follows the mask 2.4 mm.

The points join the stage's input and nothing else: the jaw faces, the planner world, the neighbours and the exact
guard read the measured surface alone, so a point placed here can shape a proposal and never clears a motion.

Pure and deterministic: numpy, image-shaped arrays, BASE millimetres.
"""

from __future__ import annotations

from typing import Final

import numpy as np

__all__ = ["HULL_LIFT_MM", "hull_points", "unread_pixels"]

#: How far inside its two ends a placed point's vertical is tested: over the floor the stage cuts and under the top.
#: One pixel at the part spans 0.7 mm at the owner's 650 mm; the millimetre keeps a rounding off the mask's own edge.
HULL_LIFT_MM: Final[float] = 1.0
#: How far over the support the vertical's foot is tested, millimetres (negative: under it).
_FOOT_OVER_SUPPORT_MM = HULL_LIFT_MM


def unread_pixels(mask: np.ndarray, read: np.ndarray) -> np.ndarray:
    """The pixels of ``mask`` whose depth the support-footprint stage does not read.

    Args:
        mask (np.ndarray): The part's whole mask, ``(H, W)``, anything truthy for the part.
        read (np.ndarray): The pixels the stage builds its footprint from with a depth it keeps, ``(H, W)``: the mask
            less its depth steps and its rim, where the depth is valid.

    Returns:
        np.ndarray: ``mask`` less ``read``, boolean ``(H, W)``.

    Raises:
        ValueError: Where the two images differ in shape.
    """
    whole = np.asarray(mask).astype(bool)
    kept = np.asarray(read).astype(bool)
    if whole.shape != kept.shape:
        raise ValueError(f"the mask is {whole.shape} and the pixels read {kept.shape}: one image each, the same size")
    return whole & ~kept


def _in_mask(points: np.ndarray, mask: np.ndarray, intrinsics: np.ndarray, base_to_camera: np.ndarray) -> np.ndarray:
    """Which of ``points`` (BASE, ``(N, 3)``) project, at the nearest pixel, into ``mask``."""
    camera = points @ base_to_camera[:3, :3].T + base_to_camera[:3, 3]
    ahead = camera[:, 2] > 1.0
    z = np.where(ahead, camera[:, 2], 1.0)
    u = np.rint(intrinsics[0, 0] * camera[:, 0] / z + intrinsics[0, 2]).astype(np.int64)
    v = np.rint(intrinsics[1, 1] * camera[:, 1] / z + intrinsics[1, 2]).astype(np.int64)
    height, width = mask.shape
    inside = ahead & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    out = np.zeros(points.shape[0], dtype=bool)
    out[inside] = mask[v[inside], u[inside]]
    return out


def hull_points(
    pixels: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_base: np.ndarray,
    support_height_mm: float,
    top_mm: float,
    *,
    floor_margin_mm: float,
    lift_mm: float = HULL_LIFT_MM,
) -> np.ndarray:
    """Each of ``pixels`` placed in the visual hull of the part's prism, where it has a place there.

    A pixel's ray is met at ``top_mm``, and the point stands where its vertical, from ``lift_mm`` over the support to
    ``lift_mm`` under ``top_mm``, projects into ``mask`` at both ends; else the ray is met ``lift_mm`` over the floor the
    stage cuts (``support_height_mm + floor_margin_mm``), so that the stage keeps the point, and kept on the same test;
    else the pixel gives none. The vertical is tested from the support itself and not from that floor: a point a
    camera tilted 45 degrees sees in front of a part's near face still projects onto the face from as high as it stands
    there, 3 mm out at 3 mm (measured on the ray-cast cube); from 1 mm over the support it is 1 mm.

    Args:
        pixels (np.ndarray): The pixels to place, ``(H, W)``, anything truthy; usually :func:`unread_pixels`.
        mask (np.ndarray): The part's whole mask, ``(H, W)``, the image the hull is read from.
        intrinsics (np.ndarray): The camera matrix ``K``, ``(3, 3)``, pixels.
        camera_to_base (np.ndarray): CAMERA to BASE, ``(4, 4)``, millimetres.
        support_height_mm (float): The support's height under the part, BASE z, millimetres.
        top_mm (float): The part's top, BASE z, millimetres: the prism's own top.
        floor_margin_mm (float): How far over the support a point still counts as the support, millimetres: the
            stage's own, which cuts every point under it.
        lift_mm (float): How far inside its ends a vertical is tested, millimetres; default :data:`HULL_LIFT_MM`.

    Returns:
        np.ndarray: The placed points, BASE millimetres, ``(N, 3)``, ``N`` at most the pixels given; ``(0, 3)`` where
        none has a place, or where the part stands no taller than the test needs (``top_mm`` within ``2 * lift_mm`` of
        the floor).

    Raises:
        ValueError: Where ``pixels`` and ``mask`` differ in shape, or ``camera_to_base`` is not ``(4, 4)``.
    """
    chosen = np.asarray(pixels).astype(bool)
    whole = np.asarray(mask).astype(bool)
    if chosen.shape != whole.shape:
        raise ValueError(f"the pixels are {chosen.shape} and the mask {whole.shape}: one image each, the same size")
    matrix = np.asarray(camera_to_base, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"camera_to_base must be a 4x4 transform, got shape {matrix.shape}")
    foot = float(support_height_mm) + _FOOT_OVER_SUPPORT_MM
    low = float(support_height_mm) + float(floor_margin_mm) + float(lift_mm)
    high = float(top_mm) - float(lift_mm)
    rows, cols = np.nonzero(chosen)
    if rows.size == 0 or high <= low:
        return np.zeros((0, 3), dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    rays = np.stack([(cols - k[0, 2]) / k[0, 0], (rows - k[1, 2]) / k[1, 1], np.ones(cols.size)], axis=-1)
    directions = rays @ matrix[:3, :3].T
    origin = matrix[:3, 3]
    base_to_camera = np.linalg.inv(matrix)
    placed = np.zeros(cols.size, dtype=bool)
    found: list[np.ndarray] = []
    for height in (float(top_mm), low):
        with np.errstate(divide="ignore", invalid="ignore"):
            reach = (height - origin[2]) / directions[:, 2]
        open_ = np.isfinite(reach) & (reach > 0.0) & ~placed
        if not open_.any():
            continue
        points = origin + reach[open_, None] * directions[open_]
        footing = points[:, :2]
        holds = (_in_mask(np.column_stack([footing, np.full(footing.shape[0], foot)]), whole, k, base_to_camera)
                 & _in_mask(np.column_stack([footing, np.full(footing.shape[0], high)]), whole, k, base_to_camera))
        if holds.any():
            index = np.flatnonzero(open_)[holds]
            placed[index] = True
            found.append(points[holds])
    if not found:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(found).astype(np.float64)
