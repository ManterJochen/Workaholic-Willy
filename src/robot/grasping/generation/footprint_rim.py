"""The rim a part's mask loses for the cloud the support-footprint stage builds its footprint from, and for nothing else.

The D415 on the owner's wrist smears every far edge of a part, its top face against the mat behind it, into a ramp of
depths between the two, and the outer 3 px or so of the colour mask lie on that ramp. The depth-step rule
(``depth_steps``, 10 mm within 3 px) lets the ramp through: its points are about 1 % of the stage's input and hang 5
to 35 mm high outside the part, and the footprint, a convex hull of every point above the floor, follows them.
MEASURED offline (2026-10-09) on 23 grey cubes the owner's cell recorded from 5 look poses on 2026-10-07, the cell's
masks reproduced from its recorded looks and each cube's top face read off the colour image for the reference: the
footprint 1.4 and 5.1 mm too large (short and long side, median), and the prism's centre, which the stage's closing
line follows, 2.53 mm off at the median and 4.44 at the 90th percentile, 2.5 mm of it away from the camera, more the
more the view is tilted (2.0 mm at 5 degrees, 4.3 at 36). With 3 px (about 2 mm at the part) eroded off the mask for
the stage's input, and ``robot.grasping.geometry.inflate_mm`` at 1.25 mm giving the faces back: the centre 0.98 mm
off at the median and 2.26 at the 90th percentile, 0.4 mm of it away from the camera, the footprint 0.0 and 1.4 mm
too large; better in 20 of the 23 views, worse in 2, the single views at 27 and 32 degrees. The owner, 2026-10-09:
"Ja, für Montag". MEASURED through this code on the same 23 cubes (2026-10-09, desk, the recorded frames through
``GraspCalculator`` with ``support_footprint_rim_mm`` 2.0 and ``support_footprint_inflate_mm`` 1.25): the rim 3 px on
15 of them and 4 px on the 8 seen from nearer than 609 mm, 694 points off at the median; the centre 0.77 mm off at the
median and 2.26 at the 90th percentile, 0.18 mm of it away from the camera, the footprint's sides 0.5 mm short and
0.8 mm long (1.8 and 3.0 mm short without the inflate); better by more than 0.25 mm in 19 of the 23, worse in the same
2. The depth-step rule read on the mask less its rim instead of on the whole mask gives the same centres.

:func:`footprint_rim` cuts it: the mask eroded by ``ceil(rim_mm * fx / z)`` px, ``z`` the median depth under the mask
and ``fx`` the colour lens's focal length in pixels (the depth is aligned to the colour image), with the elliptical
kernel the rim was measured with, never more than :data:`FOOTPRINT_RIM_MAX_SHARE` of the mask. Two inputs lose it: a
look's own input in the calculator (``GraspCalculator(support_footprint_rim_mm=...)``, from
``robot.grasping.geometry.footprint_rim_mm``) and the footprint cloud the pick loop fuses beside the looks' surfaces
(``ObservedView.footprints``, ``FusedSceneGeometry.footprint_clouds_base_mm``). The full mask stays for everything
else: the jaw-face check, the association of looks, the kept scene, the colour check, the planner world's hold-out and
the neighbours' growth. A rim taken off adds no point anywhere: it can only make a footprint smaller, which
``inflate_mm`` gives back as a margin, and the exact guard judges every motion as before.

Pure and deterministic: numpy and OpenCV's erosion, image-shaped arrays, depth in millimetres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

__all__ = ["FOOTPRINT_RIM_MAX_SHARE", "FootprintRim", "footprint_rim", "rim_px"]

#: The most of a mask's pixels the rim may take. Where it would take more, the part is small for the rim, and a rim one
#: pixel narrower is cut, and so on down to none. A choice, not a measurement: the review that measured the rim on
#: 40 mm cubes believed about 30 % right. A 3 px rim takes about a fifth of a 40 mm cube's mask seen from 700 mm; a
#: square part less than 37 px across, 28 mm there, is cut a narrower one.
FOOTPRINT_RIM_MAX_SHARE: Final[float] = 0.3


@dataclass(frozen=True, slots=True, eq=False)
class FootprintRim:
    """A mask less its rim (:func:`footprint_rim`), and what the cut took.

    ``mask`` is the mask less its rim: the mask itself where nothing was cut. ``px`` is the rim cut, in pixels, and
    ``asked_px`` the rim ``rim_mm`` asked for at ``depth_mm``, the median depth under the mask (``None`` where it held
    no depth); ``px`` is the narrower where :data:`FOOTPRINT_RIM_MAX_SHARE` held the rim back. ``pixels`` is how many
    pixels the mask held, ``taken`` how many of them the rim took, and ``points`` how many of those had a depth: the
    points it took off a cloud built from the mask.
    """

    mask: np.ndarray
    rim_mm: float
    px: int = 0
    asked_px: int = 0
    depth_mm: float | None = None
    pixels: int = 0
    taken: int = 0
    points: int = 0

    @property
    def limited(self) -> bool:
        """Whether the small-part guard cut a narrower rim than ``rim_mm`` asked for."""
        return self.px < self.asked_px

    def said(self) -> str:
        """One line for the log: the rim cut, what it was asked at which depth, and what it took."""
        if self.depth_mm is None:
            return f"no rim cut: the {self.pixels} px mask holds no depth"
        asked = f"{self.rim_mm:g} mm at {self.depth_mm:.0f} mm"
        if self.px == 0:
            return (f"no rim cut: {self.asked_px} px asked ({asked}), and 1 px would take more than "
                    f"{FOOTPRINT_RIM_MAX_SHARE:.0%} of the {self.pixels} px mask")
        took = f"{self.taken} of {self.pixels} px, {self.points} of them with a depth"
        if self.limited:
            return (f"a {self.px} px rim, {self.asked_px} px asked ({asked}) and narrowed to take at most "
                    f"{FOOTPRINT_RIM_MAX_SHARE:.0%} of the mask: {took}")
        return f"a {self.px} px rim ({asked}): {took}"


def rim_px(rim_mm: float, fx: float, depth_mm: float) -> int:
    """How many whole pixels ``rim_mm`` spans at ``depth_mm`` through a lens of focal length ``fx`` pixels, rounded up:
    3 for 2 mm at 700 mm through the D415's colour lens (fx 914), 2 at 961 mm. 0 for no rim."""
    if rim_mm <= 0.0:
        return 0
    # A rim of exactly a whole number of pixels is that number, whatever the last bit of the product says.
    return max(1, math.ceil(float(rim_mm) * float(fx) / float(depth_mm) - 1e-9))


def footprint_rim(mask: np.ndarray, depth_mm: np.ndarray, fx: float, rim_mm: float) -> FootprintRim:
    """``mask`` less a rim of ``rim_mm`` at the part, for the cloud the support-footprint stage builds its footprint
    from (the module docstring says why, and what keeps the full mask).

    The mask is eroded by :func:`rim_px` pixels at the median of the depth under it (finite and above zero), with an
    elliptical kernel ``2 * px + 1`` across and the image's border taken for the outside, as the rim was measured.
    Where that would take more than :data:`FOOTPRINT_RIM_MAX_SHARE` of the mask's pixels, a rim one pixel narrower is
    cut, and so on down to none. ``rim_mm`` 0 cuts nothing, and neither does a mask with no depth under it.
    """
    selected = np.asarray(mask).astype(bool)
    depth = np.asarray(depth_mm, dtype=np.float64)
    if selected.shape != depth.shape or selected.ndim != 2:
        raise ValueError(f"mask and depth must be the same 2-D shape, got {selected.shape} and {depth.shape}")
    if not (np.isfinite(rim_mm) and rim_mm >= 0.0):
        raise ValueError(f"rim_mm must be finite and >= 0, got {rim_mm!r}")
    if not (np.isfinite(fx) and fx > 0.0):
        raise ValueError(f"fx must be finite and > 0, got {fx!r}")
    pixels = int(np.count_nonzero(selected))
    if rim_mm == 0.0 or pixels == 0:
        return FootprintRim(mask=selected, rim_mm=float(rim_mm), pixels=pixels)
    # Only the mask's bounding box is read: a part is a few thousand pixels of a 1280x720 frame.
    rows, cols = np.nonzero(selected)
    top, bottom, left, right = int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1
    inside = selected[top:bottom, left:right]
    depth_inside = depth[top:bottom, left:right]
    with np.errstate(invalid="ignore"):
        measured = inside & np.isfinite(depth_inside) & (depth_inside > 0.0)
    if not measured.any():
        return FootprintRim(mask=selected, rim_mm=float(rim_mm), pixels=pixels)
    import cv2  # noqa: PLC0415 - deferred, as in colour_check: OpenCV is no small import

    middle = float(np.median(depth_inside[measured]))
    asked = rim_px(rim_mm, fx, middle)
    # Eroded on the box grown by more than the rim, so the crop's edge never erodes the mask; the image's own border
    # does, as it did where the rim was measured.
    pad = asked + 1
    r0, r1 = max(0, top - pad), min(selected.shape[0], bottom + pad)
    c0, c1 = max(0, left - pad), min(selected.shape[1], right + pad)
    crop = selected[r0:r1, c0:c1].astype(np.uint8)
    for px in range(asked, 0, -1):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
        kept_crop = cv2.erode(crop, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        taken = pixels - int(np.count_nonzero(kept_crop))
        if taken <= FOOTPRINT_RIM_MAX_SHARE * pixels:
            kept = np.zeros(selected.shape, dtype=bool)
            kept[r0:r1, c0:c1] = kept_crop
            points = int(np.count_nonzero(measured & ~kept[top:bottom, left:right]))
            return FootprintRim(mask=kept, rim_mm=float(rim_mm), px=px, asked_px=asked, depth_mm=middle,
                                pixels=pixels, taken=taken, points=points)
    return FootprintRim(mask=selected, rim_mm=float(rim_mm), asked_px=asked, depth_mm=middle, pixels=pixels)
