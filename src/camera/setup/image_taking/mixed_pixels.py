"""Depth pixels that mix a part's top with the floor behind it, found with the colour image and taken out.

A D400 does not measure a step from a part's top down to the floor as a step. Its correlation window straddles the
edge, and the pixels just outside the part read depths between the two: 15 to 25 pixels wide on the owner's cell, a
ramp from the top of the part down to the floor, in the raw depth as in the filtered one (2026-10-12, the cell's
recordings). Placed in BASE, the ramp is a skirt 10 to 20 mm wide about every part, between its top and the floor. The
camera world takes the skirt for an obstacle and grows its box by the margin; on the owner's crate (2026-10-12 13:45) the
box of a skirt reached back to the part's own face, and every grasp of a gray cube 30 mm from its neighbour was refused
0.1 to 0.6 mm short of the guard's distance.

A mixed pixel is told apart from a real surface by what lies on either side of it and by what the colour camera sees
there, never by its depth alone, which a steep flank or a wall seen at a slant shares:

* its depth lies between a nearer and a farther surface within :data:`RADIUS_PX`, with a jump of more than
  :data:`JUMP_MM` between them;
* along the depth's own slope there is, within :data:`REACH_PX` either way, a flat stretch nearer than it and a flat
  stretch farther than it (:data:`FLAT_MM` over 5 x 5 pixels): a top and a floor. A wall seen at a slant, a cylinder's
  flank, a sphere keep sloping past the reach, and keep their depth;
* the nearer flat stretch is no longer than a part (:data:`PART_MM`): a part's top. A crate's rim, a mat, the bench and a
  wall keep every pixel, a skirt of theirs included;
* the colour camera sees there what it sees on the farther side and not what it sees on the nearer side: the pixel looks
  at the floor (:data:`SAME_LAB`, :data:`SIDES_DIFFER_LAB`). A part whose colour cannot be told from what it stands on
  keeps every pixel.

Such a pixel loses its depth and reads as measured nothing. Nothing is put in its place: the floor it looks at is what
the parts stand on, and the space in front of it is free. Nothing gains a depth, and no pixel is moved.

Pure: numpy, SciPy's image filters and OpenCV's colour conversion. Depth in millimetres, aligned to the colour image.
"""

from __future__ import annotations

from typing import Final

import cv2 as cv
import numpy as np

__all__ = [
    "FLAT_MM",
    "JUMP_MM",
    "PART_MM",
    "RADIUS_PX",
    "REACH_PX",
    "SAME_LAB",
    "SIDES_DIFFER_LAB",
    "mixed_pixels",
    "without_mixed_pixels",
]

#: How far about a pixel the nearer and the farther surface are looked for, pixels of the full frame: half the widest
#: ramp the cell's D415 left at 1280 x 720.
RADIUS_PX: Final[int] = 12
#: The least jump between the two surfaces, millimetres: a step a finger meets. Less is a part's own relief.
JUMP_MM: Final[float] = 15.0
#: How far from both surfaces a mixed pixel's depth lies at least, millimetres: a pixel within it is the surface's own.
SIDE_MM: Final[float] = 3.0
#: How far along the depth's slope the flat stretches either side are looked for, pixels of the full frame: a ramp's
#: width and a few pixels onto either stretch.
REACH_PX: Final[int] = 26
#: A stretch is flat where 5 x 5 pixels about a point read within this of each other, millimetres: a part's top or the
#: floor under the D415's noise, never a wall or a flank seen at a slant.
FLAT_MM: Final[float] = 3.0
#: The longest a part's top is, millimetres: ``perceived.PART_SIZED_MM``'s width. A longer flat stretch is a rim, a mat
#: or the bench, whose skirts stay.
PART_MM: Final[float] = 250.0
#: How near in colour a mixed pixel is to the farther side, CIE L*a*b* with the lightness halved (a shadow on the floor
#: changes the lightness more than the hue), in OpenCV's 8-bit units.
SAME_LAB: Final[float] = 20.0
#: How far apart in colour the two sides are at least, in the same units: sides that cannot be told apart keep every
#: pixel.
SIDES_DIFFER_LAB: Final[float] = 25.0
#: Every how many pixels of the full frame the search reads along each axis: the camera world reads every second.
_STRIDE: Final[int] = 2


def mixed_pixels(depth_mm: np.ndarray, color_bgr: np.ndarray, focal_px: float) -> np.ndarray:
    """Which pixels of ``depth_mm`` mix a part's top with the floor behind it (see the module docstring).

    Args:
        depth_mm (np.ndarray): The depth, ``(H, W)`` millimetres, aligned to the colour image; 0 or not finite where
            nothing was measured.
        color_bgr (np.ndarray): The colour image, ``(H, W, 3)`` 8-bit BGR, the same pixels.
        focal_px (float): The colour camera's focal length, pixels: how long a part's top is.

    Returns:
        np.ndarray: ``(H, W)`` boolean, true where a pixel mixes the two; all false where the shapes differ.
    """
    from scipy import ndimage  # noqa: PLC0415 (kept out of import time)

    depth_full = np.asarray(depth_mm, dtype=np.float64)
    colour_full = np.asarray(color_bgr)
    out = np.zeros(depth_full.shape, dtype=bool)
    if depth_full.ndim != 2 or colour_full.shape[:2] != depth_full.shape or colour_full.ndim != 3:
        return out
    depth = depth_full[::_STRIDE, ::_STRIDE]
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(depth) & (depth > 0.0)
    if not valid.any():
        return out
    radius = max(1, RADIUS_PX // _STRIDE)
    size = 2 * radius + 1
    nearest = ndimage.minimum_filter(np.where(valid, depth, np.inf), size=size, mode="constant", cval=np.inf)
    farthest = ndimage.maximum_filter(np.where(valid, depth, -np.inf), size=size, mode="constant", cval=-np.inf)
    with np.errstate(invalid="ignore"):
        between = (valid & (farthest - nearest > JUMP_MM) & (depth - nearest > SIDE_MM)
                   & (farthest - depth > SIDE_MM))
    rows, cols = np.nonzero(between)
    if rows.size == 0:
        return out

    low = ndimage.minimum_filter(np.where(valid, depth, np.inf), size=3, mode="constant", cval=np.inf)
    high = ndimage.maximum_filter(np.where(valid, depth, -np.inf), size=3, mode="constant", cval=-np.inf)
    with np.errstate(invalid="ignore"):
        flat = valid & (high - low <= FLAT_MM)
    # The slope of the depth, holes left out: a normalised blur, then Sobel.
    weight = ndimage.gaussian_filter(valid.astype(np.float64), 1.0)
    smooth = ndimage.gaussian_filter(np.where(valid, depth, 0.0), 1.0) / np.maximum(weight, 1e-6)
    slope_rows = ndimage.sobel(smooth, axis=0)[rows, cols]
    slope_cols = ndimage.sobel(smooth, axis=1)[rows, cols]
    length = np.hypot(slope_rows, slope_cols)
    sloped = length > 1e-9
    rows, cols = rows[sloped], cols[sloped]
    down_rows, down_cols = slope_rows[sloped] / length[sloped], slope_cols[sloped] / length[sloped]
    height, width = depth.shape
    here = depth[rows, cols]
    count = rows.size
    near_at = np.full((count, 2), -1, dtype=np.int64)
    far_at = np.full((count, 2), -1, dtype=np.int64)
    # Along the slope either way, the first flat stretch nearer (up) and farther (down) than the pixel.
    for step in range(2, REACH_PX // _STRIDE + 1):
        for sign, found in ((-1.0, near_at), (1.0, far_at)):
            r = np.rint(rows + sign * step * down_rows).astype(np.int64)
            c = np.rint(cols + sign * step * down_cols).astype(np.int64)
            inside = (r >= 0) & (r < height) & (c >= 0) & (c < width)
            r, c = np.clip(r, 0, height - 1), np.clip(c, 0, width - 1)
            hit = inside & (found[:, 0] < 0) & flat[r, c]
            hit &= (depth[r, c] < here - SIDE_MM) if sign < 0.0 else (depth[r, c] > here + SIDE_MM)
            found[hit, 0], found[hit, 1] = r[hit], c[hit]
    both = np.nonzero((near_at[:, 0] >= 0) & (far_at[:, 0] >= 0))[0]
    if both.size == 0:
        return out
    near_depth = depth[near_at[both, 0], near_at[both, 1]]
    far_depth = depth[far_at[both, 0], far_at[both, 1]]
    both = both[far_depth - near_depth > JUMP_MM]
    if both.size == 0:
        return out

    # The nearer stretch has to be a part's top: no longer than a part at its own depth.
    labels, pieces = ndimage.label(flat)
    spans = np.zeros(pieces + 1, dtype=np.float64)
    for index, piece in enumerate(ndimage.find_objects(labels), start=1):
        if piece is not None:
            spans[index] = max(piece[0].stop - piece[0].start, piece[1].stop - piece[1].start) * _STRIDE
    depths = np.zeros(pieces + 1, dtype=np.float64)
    if pieces:
        depths[1:] = np.asarray(ndimage.mean(depth, labels, index=np.arange(1, pieces + 1)), dtype=np.float64)
    part_top = spans * depths / max(float(focal_px), 1e-6) <= PART_MM
    both = both[part_top[labels[near_at[both, 0], near_at[both, 1]]]]
    if both.size == 0:
        return out

    small = colour_full[::_STRIDE, ::_STRIDE]
    lab = cv.cvtColor(cv.blur(np.ascontiguousarray(small, dtype=np.uint8), (3, 3)), cv.COLOR_BGR2LAB).astype(np.float64)
    lab[..., 0] *= 0.5
    at = lab[rows[both], cols[both]]
    near_colour = lab[near_at[both, 0], near_at[both, 1]]
    far_colour = lab[far_at[both, 0], far_at[both, 1]]
    to_far = np.linalg.norm(at - far_colour, axis=1)
    to_near = np.linalg.norm(at - near_colour, axis=1)
    sides = np.linalg.norm(near_colour - far_colour, axis=1)
    taken = both[(sides > SIDES_DIFFER_LAB) & (to_far < SAME_LAB) & (to_far < 0.5 * to_near)]
    small_out = np.zeros(depth.shape, dtype=bool)
    small_out[rows[taken], cols[taken]] = True
    # Each pixel read stands for the block of the full frame it was read for.
    full_rows, full_cols = int(depth_full.shape[0]), int(depth_full.shape[1])
    grown = np.repeat(np.repeat(small_out, _STRIDE, axis=0), _STRIDE, axis=1)
    out = np.asarray(grown[:full_rows, :full_cols], dtype=bool)
    with np.errstate(invalid="ignore"):
        out &= np.isfinite(depth_full) & (depth_full > 0.0)
    return out


def without_mixed_pixels(depth_mm: np.ndarray, color_bgr: np.ndarray, focal_px: float) -> "tuple[np.ndarray, int]":
    """``depth_mm`` with its mixed pixels (:func:`mixed_pixels`) read as measured nothing, and how many there were.

    Args:
        depth_mm (np.ndarray): The depth, ``(H, W)`` millimetres, aligned to the colour image.
        color_bgr (np.ndarray): The colour image, ``(H, W, 3)`` 8-bit BGR, the same pixels.
        focal_px (float): The colour camera's focal length, pixels.

    Returns:
        tuple[np.ndarray, int]: A copy of the depth with every mixed pixel 0, in the depth's own dtype, and their count.
    """
    depth = np.asarray(depth_mm)
    taken = mixed_pixels(depth, color_bgr, focal_px)
    count = int(np.count_nonzero(taken))
    if count == 0:
        return depth.copy(), 0
    out = depth.copy()
    out[taken] = 0
    return out, count
