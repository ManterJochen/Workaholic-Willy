"""Mask pixels that see past the object: the background behind a depth step.

A segmentation mask is drawn on the colour image and the depth is aligned to it, so the rim of a mask
carries whatever depth the alignment put there. One or two pixels of misregistration at a part's far
edge put the surface behind the part inside the mask. Looking straight down, that surface is the table
right under the edge, inside the part's own footprint, and it does no harm. Looking at 45 degrees it
is the table 40 to 46 mm behind a 40 mm part, and the support-footprint hull, which is convex and keeps
every point above its floor, reaches back to it. Measured 2026-09-23 on a ray-cast of a D415 colour
camera (fx 925, 1280x720) 500 mm from a 40 mm cube, with the far edge misregistered by 1 to 2 px, the
table read 3 mm high and 1 mm depth noise: the best grasp line lay 36.6 to 37.0 mm from the part centre
in 9 of 10 noise seeds, so the jaws closed 16.7 mm behind a part whose half-size is 20 mm. The same
scene seen straight down put it 0.3 to 7.9 mm from the centre.

Such a pixel can be recognised in the image without knowing anything about the part: some pixel of the
same mask a few pixels away is much nearer. A continuous surface cannot do that. At 500 mm one D415
colour pixel is 0.54 mm across, so across ``DEPTH_STEP_RADIUS_PX`` pixels a surface changes its depth
by more than ``DEPTH_STEP_MM`` only when it is seen within about 9 degrees of edge-on (about 15 degrees
at 800 mm), where the sensor returns little depth anyway. The background behind a part's edge sits
further back than that by roughly the part's height divided by the cosine of the view tilt, 56 mm for
the 40 mm cube at 45 degrees.

What this removes is a band of at most ``DEPTH_STEP_RADIUS_PX`` pixels on the far side of every depth
step inside the mask. The nearest pixel of any neighbourhood is never removed, so a mask with depth
under it keeps depth under it. Pixels with no depth are left alone; back-projection drops them anyway.

Pure and deterministic: numpy only, image-shaped arrays, depth in millimetres.
"""

from __future__ import annotations

from typing import Final

import numpy as np

__all__ = ["DEPTH_STEP_MM", "DEPTH_STEP_RADIUS_PX", "pixels_behind_depth_steps"]

#: How much further back than a nearby pixel of the same mask a pixel has to measure before it is
#: taken for the background behind an edge, millimetres. See the module docstring for what a
#: continuous surface can do across ``DEPTH_STEP_RADIUS_PX`` pixels.
DEPTH_STEP_MM: Final[float] = 10.0

#: How far, in pixels, a pixel looks for a nearer pixel of its own mask. Three covers the one to two
#: pixels of depth-to-colour misregistration a D400 shows at an edge, with one to spare.
DEPTH_STEP_RADIUS_PX: Final[int] = 3


def pixels_behind_depth_steps(
    mask: np.ndarray,
    depth_mm: np.ndarray,
    *,
    step_mm: float = DEPTH_STEP_MM,
    radius_px: int = DEPTH_STEP_RADIUS_PX,
) -> np.ndarray:
    """The mask pixels that measure more than ``step_mm`` behind a pixel of the same mask nearby.

    Returns a boolean image of the mask's shape, true where a pixel of ``mask`` has a valid depth and
    some other valid pixel of ``mask`` within ``radius_px`` (a square window) is nearer by more than
    ``step_mm``. ``mask & ~pixels_behind_depth_steps(mask, depth_mm)`` is the part of the mask that
    sees the object rather than what lies behind its edge. Only pixels of the mask are compared, so an
    occluder in front of the object, which is not in its mask, removes nothing.
    """
    selected = np.asarray(mask).astype(bool)
    depth = np.asarray(depth_mm, dtype=np.float64)
    if selected.shape != depth.shape or selected.ndim != 2:
        raise ValueError(f"mask and depth must be the same 2-D shape, got {selected.shape} and {depth.shape}")
    if not np.isfinite(step_mm) or step_mm <= 0.0:
        raise ValueError(f"step_mm must be finite and > 0, got {step_mm!r}")
    radius = int(radius_px)
    if radius < 1:
        raise ValueError(f"radius_px must be >= 1, got {radius_px!r}")
    behind = np.zeros(selected.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        valid = selected & np.isfinite(depth) & (depth > 0.0)
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return behind
    # Only the mask's bounding box is worked on: a part is a few thousand pixels of a 1280x720 frame.
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    crop_valid = valid[r0:r1, c0:c1]
    near = np.where(crop_valid, depth[r0:r1, c0:c1], np.inf)
    height, width = near.shape
    padded = np.pad(near, radius, mode="constant", constant_values=np.inf)
    # The nearest depth in a (2r+1)-square window, as two one-dimensional passes: a minimum over a
    # square is a minimum over its rows of a minimum over its columns.
    across = padded[:, 0:width].copy()
    for shift in range(1, 2 * radius + 1):
        np.minimum(across, padded[:, shift:shift + width], out=across)
    nearest = across[0:height, :].copy()
    for shift in range(1, 2 * radius + 1):
        np.minimum(nearest, across[shift:shift + height, :], out=nearest)
    with np.errstate(invalid="ignore"):
        behind[r0:r1, c0:c1] = crop_valid & (near - nearest > float(step_mm))
    return behind
