"""Should a segmentation mask be replaced by a box? One policy, one implementation, three answers.

Both perception sources, the real-camera one and the Isaac one, apply the same rule: a SAM2 mask
covering less than 80 % of the detector's box area (``DEFAULT_MIN_FILL_RATIO``) is thrown away and
the box is used instead. The rule lives here once, behind a named policy, so a change to it cannot
fix one cell and miss the other.

The rule assumes the detector box is a good centred silhouette of the object. That holds for an
axis-aligned, roughly rectangular object seen top-down and for nothing else, and the rule never
checks it. Measured with ``python -m datagen camera-probe`` over 809 views of the ``v1_proof``
reference corpus (3288 masks), it fires on 78.7 % of predicted masks and on 78.8 % of ground-truth
masks, at a median mask/box area of 0.696 and 0.691. The ground-truth figure is the control: the
rule detects a mask that is not a rectangle, not a mask that is incomplete. A circle fills
pi/4 = 0.785 of its bounding box and is already under 0.8.

A filled mask is an axis-aligned rectangle whose principal axis lies on an image axis by
construction, and the grasp's closing axis is read off that principal axis. When the rule fires it
rotates the closing axis by a median of 16.9 deg, p95 47.9 deg, max 89.6 deg.

The rescue it exists for is rare: SAM2 silhouettes genuinely truncate in 10.3 % of detections, so
the rule fires 7.6x more often than the condition it was written for.

Measured in grasp outcome by ``datagen eval-grasps --mask-completion`` over 270 reference scenes,
paired over the 1365 object-views evaluated under every policy (the populations differ, so raw
rates are a denominator trap):

    policy                top-1     coverage
    none                  26.7 %     30.9 %
    axis_aligned_box      19.2 %     25.6 %     -7.5 pp / -5.3 pp
    oriented_box          26.2 %     30.0 %     -0.5 pp / -1.0 pp

    axis_aligned_box vs none, per object-view:  129 lost, 26 won, net -103 of 1365

That population carries no truncated masks, so it measures the harm and cannot see the rescue. The
rescue is bounded by arithmetic rather than measured: it applies to the 10.3 % of masks that
truncate, so at the 26.7 % baseline it can add at most about 2.7 pp. The harm is two to three times
larger even with the rescue at its theoretical maximum.

Filling also moves objects across the visibility floor. ``MIN_VISIBLE_PX`` in the datagen eval
ladder is 100; an instance with 31 ground-truth pixels is skipped and has 192 pixels after filling,
so the transform promotes barely-visible objects into grasp targets whose silhouette is a rectangle
covering mostly their neighbours.

So the default is ``NONE``. ``ORIENTED_BOX`` is statistically indistinguishable from it, 0.5 pp over
1365 object-views being about seven of them, and ``NONE`` never invents geometry: filling an
L-shaped silhouette to any box adds material that is not there.

On the sim side only ``MultiObjectVisionPerceptionSource`` in ``willy_sim/perception/vision.py``
calls ``complete_mask``; ``IsaacVisionPerceptionSource`` does not fill masks at all.

The rule tests one of the two conditions it rests on: that the mask covers much less than the box,
and not that the box is a good centred silhouette. A rule that also checked whether the mask sits
off-centre within the detector box, which is what truncation looks like and what non-rectangularity
does not, would rescue the 10.3 % without touching the 78.7 %.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import numpy as np

__all__ = [
    "MaskCompletion",
    "DEFAULT_MASK_COMPLETION",
    "DEFAULT_MIN_FILL_RATIO",
    "complete_mask",
]

#: A mask covering less than this share of its reference box is treated as incomplete. At 0.8 the
#: test selects for a mask that is not a rectangle; the module docstring carries the measurement.
DEFAULT_MIN_FILL_RATIO = 0.8


class MaskCompletion(StrEnum):
    """What to do with a mask that does not fill its reference box."""

    #: Never replace the mask. SAM2's silhouette is used exactly as segmented.
    NONE = "none"

    #: Replace it with the detector's axis-aligned box. It can restore extent the segmenter
    #: dropped, and it destroys the object's orientation whenever it fires.
    AXIS_ALIGNED_BOX = "axis_aligned_box"

    #: Replace it with the mask's own minimum-area oriented box. Fills concavity and interior holes
    #: while preserving the principal axis.
    #:
    #: It gives up the extent rescue: the oriented box is derived from the mask, so a mask missing
    #: one end of an object yields a box missing that end too. It removes the harm and does not keep
    #: the cure.
    ORIENTED_BOX = "oriented_box"


#: The shipped policy, and the one place it is decided. Both perception sources take it, so a real
#: cell and the Isaac cell cannot drift apart on the transform that decides a grasp's closing axis.
#:
#: ``NONE`` is selected by the measurement in this module's docstring.
DEFAULT_MASK_COMPLETION = MaskCompletion.NONE


def _detector_box(mask: np.ndarray, det: Any) -> tuple[int, int, int, int] | None:
    """The detector's box, clipped to the image. ``None`` when there is nothing usable."""
    raw = getattr(det, "box", None)
    if raw is None:
        return None
    height, width = mask.shape[:2]
    x0, y0, x1, y1 = (int(round(float(c))) for c in raw)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    return None if (x1 <= x0 or y1 <= y0) else (x0, y0, x1, y1)


def _fill_axis_aligned(mask: np.ndarray, box: tuple[int, int, int, int], ratio: float) -> np.ndarray:
    x0, y0, x1, y1 = box
    area = (x1 - x0) * (y1 - y0)
    if int(mask.sum()) >= ratio * area:
        return mask                       # already a near-complete silhouette: keep SAM2's precision
    filled = np.zeros(mask.shape[:2], dtype=bool)
    filled[y0:y1, x0:x1] = True
    return filled


def _fill_oriented(mask: np.ndarray, ratio: float) -> np.ndarray:
    """Fill to ``cv.minAreaRect`` of the mask itself, when the mask underfills it."""
    import cv2 as cv                      # deferred: this module is otherwise numpy-only

    ys, xs = np.nonzero(mask)
    if xs.size < 3:
        return mask
    points = np.stack([xs, ys], axis=1).astype(np.int32)
    rect = cv.minAreaRect(points)
    (_, (w, h), _) = rect
    area = float(w) * float(h)
    if area <= 0.0 or int(mask.sum()) >= ratio * area:
        return mask
    filled = np.zeros(mask.shape[:2], dtype=np.uint8)
    cv.fillPoly(filled, [np.round(cv.boxPoints(rect)).astype(np.int32)], 1)
    return filled.astype(bool)


def complete_mask(
    mask: np.ndarray,
    det: Any = None,
    *,
    policy: MaskCompletion = DEFAULT_MASK_COMPLETION,
    min_fill_ratio: float = DEFAULT_MIN_FILL_RATIO,
) -> np.ndarray:
    """Apply the configured mask-completion policy. Returns a boolean mask.

    Non-2-D input, a missing detector box, or a degenerate box return the mask untouched: the rule
    refuses rather than guessing, which is why a source can call it unconditionally.
    """
    # copy=False so an already-boolean mask comes back as the same object when nothing is replaced.
    # Identity means "untouched" to a caller, and the copy would also be pure waste here.
    mask = np.asarray(mask).astype(bool, copy=False)
    if mask.ndim != 2 or policy is MaskCompletion.NONE:
        return mask
    if policy is MaskCompletion.ORIENTED_BOX:
        return _fill_oriented(mask, float(min_fill_ratio))
    box = _detector_box(mask, det)
    return mask if box is None else _fill_axis_aligned(mask, box, float(min_fill_ratio))
