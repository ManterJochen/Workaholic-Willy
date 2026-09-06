"""From a rendered instance map to labels, including the one that makes evaluation honest.

Boxes and masks are mechanical. Visibility is not, and it is the label that decides whether a number
about a dataset means anything: in a pile, half the objects are mostly hidden, and a model scored on
targets that are four visible pixels is being scored on a task nobody could do.

Visibility is measured, not estimated. The visible mask comes from the rendered instance map of the
full scene, and the unoccluded mask from rendering the object alone, so their ratio is exact.

The solo passes do not need the expensive renderer. A silhouette does not depend on light transport,
so the object-alone passes run on the raster path while the beauty pass is path-traced: same
geometry, same camera, orders of magnitude cheaper. Without that, this one label would multiply the
cost of the whole dataset by the object count.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

__all__ = [
    "BACKGROUND_ID",
    "ObjectLabel",
    "bbox_from_mask",
    "build_object_labels",
    "instance_map_from_masks",
    "mask_for_instance",
    "pixel_id",
    "visibility",
]

#: Pixel value for "no object". Objects therefore start at 1, the usual instance-map convention.
BACKGROUND_ID = 0


def pixel_id(index: int) -> int:
    """Object index to its value in an instance map. The one place this offset exists.

    A painter that writes object k as k+1 while a reader looks up k gives every label its neighbour's
    pixels and gives object 0 the background's, which is most of the image. The offset itself is not
    the problem; two functions each applying their own is. Both sides call this, so they cannot
    disagree.
    """
    return index + BACKGROUND_ID + 1


def instance_map_from_masks(
    visible_masks: "dict[int, np.ndarray] | Mapping[int, np.ndarray]", shape: tuple[int, ...],
) -> np.ndarray:
    """Paint per-object visible masks into one instance map: the inverse of ``mask_for_instance``.

    Later objects win overlapping pixels. Overlap should be empty when the masks come from matching
    scene depth against solo depth, because a pixel has one nearest surface, so this is a tie-break
    for antialiased edges rather than a policy decision about occlusion.
    """
    out = np.zeros(shape, dtype=np.int32)
    for index in sorted(visible_masks):
        out[np.asarray(visible_masks[index], dtype=bool)] = pixel_id(index)
    return out


@dataclass(frozen=True, slots=True)
class ObjectLabel:
    """One object as seen from one view. Pixel coordinates; poses in mm / XYZW, base frame."""

    asset_id: str
    #: The object's index in the scene, the same key as ``settled_poses`` and ``materials``. Its value
    #: in ``*_instances.png`` is ``pixel_id(instance_id)``, because 0 there means background.
    instance_id: int
    #: ``(x1, y1, x2, y2)`` inclusive-exclusive pixels of the visible part, or ``None`` when nothing of
    #: this object reached the image. ``None`` is a real answer and is kept rather than dropped.
    bbox_xyxy: tuple[int, int, int, int] | None
    visible_px: int
    unoccluded_px: int
    #: ``visible / unoccluded`` in [0, 1]. 1.0 means nothing is in front of it; 0.0 means fully
    #: hidden or off-frame.
    visibility: float
    position_mm: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]

    @property
    def in_frame(self) -> bool:
        return self.bbox_xyxy is not None


def mask_for_instance(instance_map: np.ndarray, instance_id: int) -> np.ndarray:
    """Boolean mask of one instance in a rendered instance-id image."""
    return np.asarray(instance_map) == instance_id


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Tight ``(x1, y1, x2, y2)`` around the True pixels, or ``None`` for an empty mask.

    x2/y2 are exclusive, matching numpy slicing, so ``mask[y1:y2, x1:x2]`` is the box. COCO wants
    ``[x, y, w, h]``; the export converts, rather than this carrying two conventions.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return int(x1), int(y1), int(x2) + 1, int(y2) + 1


def visibility(visible_px: int, unoccluded_px: int) -> float:
    """``visible / unoccluded``, clamped to [0, 1]; 0.0 when the object is not in frame at all.

    The zero case is guarded rather than left to divide: an object entirely outside the frustum has
    an empty solo mask too, and a NaN visibility would silently poison every aggregate computed over
    the dataset.
    """
    if unoccluded_px <= 0:
        return 0.0
    return float(min(1.0, max(0.0, visible_px / unoccluded_px)))


def build_object_labels(
    instance_map: np.ndarray,
    unoccluded_masks: dict[int, np.ndarray],
    placements: dict[int, tuple[str, tuple[float, float, float], tuple[float, float, float, float]]],
) -> tuple[ObjectLabel, ...]:
    """Assemble one label per object from the scene instance map plus the per-object solo masks.

    ``placements`` maps instance id to (asset_id, settled position, settled orientation). The settled
    pose, because that is what the image shows: the spawn pose describes a moment that no longer
    exists by the time anything is rendered.

    Objects with no solo mask are refused rather than assumed fully visible. That combination means
    the solo pass did not run for them, and inventing a visibility of 1.0 there would be a fabricated
    label.
    """
    labels: list[ObjectLabel] = []
    for instance_id in sorted(placements):
        asset_id, position, orientation = placements[instance_id]
        if not isinstance(asset_id, str):
            # Checked rather than annotated: the caller's dict is typed `Any`, so a type checker
            # cannot see a whole `SceneAsset` passed where an id belongs. Unchecked, that mistake
            # writes a record into `views[i].objects[j].asset_id` and nothing downstream notices,
            # because every consumer that needs an id reads `spec.objects[i].asset_id` instead.
            raise TypeError(
                f"instance {instance_id}: asset_id must be a str, got {type(asset_id).__name__}. "
                f"Pass `record.asset_id`, not the record."
            )
        if instance_id not in unoccluded_masks:
            raise KeyError(
                f"instance {instance_id} ({asset_id}) has no unoccluded mask; its solo pass did not "
                f"run. Refusing to guess its visibility: an invented 1.0 here would be indistinguishable "
                f"from a genuinely unoccluded object."
            )
        visible = mask_for_instance(instance_map, pixel_id(instance_id))
        solo = np.asarray(unoccluded_masks[instance_id], dtype=bool)
        visible_px, unoccluded_px = int(visible.sum()), int(solo.sum())
        labels.append(ObjectLabel(
            asset_id=asset_id,
            instance_id=instance_id,
            bbox_xyxy=bbox_from_mask(visible),
            visible_px=visible_px,
            unoccluded_px=unoccluded_px,
            visibility=visibility(visible_px, unoccluded_px),
            position_mm=position,
            orientation_xyzw=orientation,
        ))
    return tuple(labels)
