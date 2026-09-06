"""Topology-risk signals for grasp candidate evaluation.

The grasp pipeline uses ``1 - solidity`` (the non-solid fraction of a mask) as
a heuristic for clutter and for the auto mode when it samples densely. This module makes that signal
a risk channel of its own, which:

* exposes a per-call, per-mask risk score in calculator telemetry,
* lets an operator set a threshold that rejects candidates whose source
  mask has a structurally risky silhouette, such as a U-shape or a handle,
* never penalises silently, since every threshold breach surfaces an
  explicit reason code in :class:`GraspResult`.

The helpers stay lightweight and reuse the OpenCV contour and hull
primitives rather than adding a heavy dependency, so the scoring stays
inspectable and deterministic for a given mask.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np

__all__ = [
    "depth_continuity_risk",
    "topology_risk_from_mask",
]


def topology_risk_from_mask(mask: np.ndarray) -> float | None:
    """Return ``1 - solidity`` (contour area over convex-hull area) of the largest contour in ``mask``.

    A convex blob scores near ``0.0`` and a concave outline scores higher: a
    disc measures 0.012 against 0.306 for a U-shape of the same extent. A hole
    enclosed by the silhouette does not register, because the area is taken over
    the largest contour and that contour is the outer boundary, so a ring scores
    exactly what the disc around it scores. Returns :data:`None` when the ratio
    cannot be computed, from an empty mask, a degenerate hull or a zero-area
    contour. Read that :data:`None` as no signal rather than as low risk, so a
    missing score never silently passes a threshold gate.
    """
    arr = np.asarray(mask)
    if arr.ndim != 2 or arr.size == 0:
        return None
    binary = (arr > 0).astype(np.uint8) * 255
    if not np.any(binary):
        return None
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv.contourArea)
    area = float(cv.contourArea(contour))
    if area <= 0.0:
        return None
    hull = cv.convexHull(contour)
    hull_area = float(cv.contourArea(hull))
    if hull_area <= 0.0:
        return None
    solidity = float(np.clip(area / hull_area, 0.0, 1.0))
    return 1.0 - solidity


def depth_continuity_risk(
    depth_map: np.ndarray, mask: np.ndarray, *, jump_mm: float = 25.0
) -> float | None:
    """Return the fraction of mask pixels whose neighbours show a depth jump.

    A jump is any 4-neighbour absolute depth difference greater than
    ``jump_mm`` between finite, positive depth samples. The output is a
    bounded fraction in ``[0.0, 1.0]``, so it composes with the
    silhouette-based :func:`topology_risk_from_mask` signal.

    Returns :data:`None` when not enough finite depth pixels lie under
    the mask to estimate the statistic reliably.
    """
    depth_arr = np.asarray(depth_map, dtype=np.float64)
    mask_arr = np.asarray(mask)
    if depth_arr.ndim != 2 or mask_arr.shape != depth_arr.shape:
        return None
    mask_bool = mask_arr.astype(bool)
    if not np.any(mask_bool):
        return None
    finite = np.isfinite(depth_arr) & (depth_arr > 0.0)
    valid = mask_bool & finite
    total = int(np.count_nonzero(valid))
    if total < 8:  # too few pixels to make the ratio meaningful
        return None
    jumps = np.zeros_like(valid, dtype=bool)
    for axis in (0, 1):
        d = np.abs(np.diff(depth_arr, axis=axis))
        finite_pair = (
            finite[:-1, :] & finite[1:, :] if axis == 0 else finite[:, :-1] & finite[:, 1:]
        )
        big = (d > float(jump_mm)) & finite_pair
        if axis == 0:
            jumps[:-1, :] |= big & mask_bool[:-1, :]
            jumps[1:, :] |= big & mask_bool[1:, :]
        else:
            jumps[:, :-1] |= big & mask_bool[:, :-1]
            jumps[:, 1:] |= big & mask_bool[:, 1:]
    return float(np.count_nonzero(jumps & valid)) / float(total)
