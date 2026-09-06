"""Re-identify a known target across perception frames.

Between the initial scan and the refined (standoff) scan the target can shift in
the image, dramatically so for an eye-in-hand wrist camera that moves with the
arm. A :class:`TargetTracker` answers "which segmentation in the new frame is the
same object?". Two default trackers ship here:

* :class:`IoUCentroidTargetTracker`: image-space, highest mask-IoU with a
  centroid-distance tie-break. Cheap, but loses the target when the viewpoint
  moves.
* :class:`WorldSpacePoseTracker`: viewpoint-invariant, matching by 3D BASE-frame
  centroid distance. A strict superset that falls back to the image-space
  tracker when the 3D signal is missing.

These are injected into :class:`refinement.DefaultPreGraspRefiner` and live in
their own module, so the refiner and the trackers stay independent.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

import numpy as np

from src.geometry import Transform
from src.robot.grasping.constants import (
    TARGET_TRACKING_LOG_FILE,
    create_grasping_logger,
)
from src.robot.grasping.types.perception import SegmentationLike

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.grasping.types.perception import PerceptionFrame


#: A lost target ends the refine as TARGET_LOST, and the two trackers lose it for
#: different reasons (IoU collapse vs. no 3D signal). One line per match call;
#: the per-segmentation scan inside stays silent.
logger = create_grasping_logger("TargetTracking", TARGET_TRACKING_LOG_FILE)

__all__ = [
    "IoUCentroidTargetTracker",
    "TargetIdentity",
    "TargetTracker",
    "WorldSpacePoseTracker",
    "target_identity_from_segmentation",
]


@dataclass(frozen=True, slots=True)
class TargetIdentity:
    """Pixel-level fingerprint of the target the initial grasp was on.

    The default :class:`IoUCentroidTargetTracker` only consumes
    :attr:`mask` and :attr:`centroid_xy`; the other fields exist so
    learned matchers can be plugged in via the
    :class:`TargetTracker` Protocol without widening this dataclass.

    Attributes
    ----------
    mask
        Boolean (or 0/1 uint8) array of shape ``(H, W)``. Same shape
        as the initial frame's segmentation masks. The refiner does
        not attempt to handle resolution mismatches; the perception
        source is expected to produce same-resolution frames.
    centroid_xy
        ``(x, y)`` pixel centroid of :attr:`mask`. Pre-computed so the
        tracker does not have to recompute it on every match call.
    area_px
        Pixel count of :attr:`mask`. Used by the default tracker as a
        sanity check (a refined mask that is wildly different in area
        is unlikely to be the same object).
    label
        Optional semantic label carried through from the segmentation,
        for label-aware tracking. The default tracker ignores it.
    """

    mask: np.ndarray
    centroid_xy: tuple[float, float]
    area_px: int
    label: Optional[str] = None
    # The target's 3D centroid in the BASE frame (mm), back-projected from the initial frame's depth +
    # intrinsics + camera_to_base. ``None`` for image-space-only identities (the default factory call). The
    # :class:`WorldSpacePoseTracker` consumes it to match by 3D pose (viewpoint-invariant) instead of pixels.
    centroid_xyz_mm: Optional[tuple[float, float, float]] = None

    def __post_init__(self) -> None:
        if self.mask.ndim != 2:
            raise ValueError(
                f"TargetIdentity mask must be 2-D; got shape {self.mask.shape}"
            )
        if self.area_px < 0:
            raise ValueError(
                f"area_px must be non-negative; got {self.area_px}"
            )


def _centroid_xyz_base_mm(
    mask: Optional[np.ndarray],
    depth_map: Optional[np.ndarray],
    intrinsics: Optional[np.ndarray],
    camera_to_base: Optional[Transform],
) -> Optional[tuple[float, float, float]]:
    """Back-project a mask's centroid to a 3D point (mm): BASE frame when ``camera_to_base`` is supplied, else
    CAMERA frame. Depth is the median of the mask's positive foreground samples; ``None`` on any empty/degenerate
    input so the caller falls back to image-space matching."""

    if mask is None or depth_map is None or intrinsics is None:
        return None
    m = np.asarray(mask)
    if m.ndim != 2:
        return None
    m = m.astype(bool, copy=False)
    if not m.any():
        return None
    d_all = np.asarray(depth_map, dtype=np.float64)
    if d_all.shape != m.shape:
        return None
    d_vals = d_all[m]
    d_vals = d_vals[np.isfinite(d_vals) & (d_vals > 0.0)]
    if d_vals.size == 0:
        return None
    depth = float(np.median(d_vals))
    k = np.asarray(intrinsics, dtype=np.float64)
    if k.shape != (3, 3):
        return None
    fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    if fx == 0.0 or fy == 0.0:
        return None
    ys, xs = np.where(m)
    px, py = float(xs.mean()), float(ys.mean())
    pt_cam = np.array([(px - cx) * depth / fx, (py - cy) * depth / fy, depth], dtype=np.float64)
    pt = pt_cam if camera_to_base is None else camera_to_base.apply_point(pt_cam)
    return (float(pt[0]), float(pt[1]), float(pt[2]))


def target_identity_from_segmentation(
    segmentation: SegmentationLike,
    *,
    label: Optional[str] = None,
    depth_map: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    camera_to_base: Optional[Transform] = None,
) -> TargetIdentity:
    """Build a :class:`TargetIdentity` from a segmentation.

    Centroid is computed in pixel coordinates as the mean of the
    foreground pixel indices. An empty mask raises :class:`ValueError`
    because there is no meaningful target to track.

    When both ``depth_map`` and ``intrinsics`` are supplied, the
    target's 3D centroid is also back-projected, in the BASE frame when
    ``camera_to_base`` is given, and stored on
    :attr:`TargetIdentity.centroid_xyz_mm` so the
    :class:`WorldSpacePoseTracker` can match by 3D pose. Omitting them
    keeps the image-space-only identity.
    """

    mask_raw = np.asarray(segmentation.mask)
    if mask_raw.ndim != 2:
        raise ValueError(
            "target_identity_from_segmentation requires a 2-D mask; "
            f"got shape {mask_raw.shape}"
        )
    mask_bool = mask_raw.astype(bool, copy=False)
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        raise ValueError(
            "Cannot build TargetIdentity from an empty segmentation mask."
        )
    centroid = (float(xs.mean()), float(ys.mean()))
    seg_label = label
    if seg_label is None:
        seg_label = getattr(segmentation, "label", None)
    centroid_xyz = _centroid_xyz_base_mm(mask_bool, depth_map, intrinsics, camera_to_base)
    return TargetIdentity(
        mask=mask_bool,
        centroid_xy=centroid,
        area_px=int(ys.size),
        label=seg_label,
        centroid_xyz_mm=centroid_xyz,
    )


@runtime_checkable
class TargetTracker(Protocol):
    """Re-identify a known target inside a new frame.

    Implementations return the index of the matching segmentation in
    ``frame.segmentations`` together with the achieved IoU score, or
    :data:`None` when no candidate clears the threshold.
    """

    def match(
        self,
        identity: TargetIdentity,
        frame: "PerceptionFrame",
        *,
        iou_threshold: float,
        camera_to_base: Optional[Transform] = None,
    ) -> Optional[tuple[int, float]]: ...


def _binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two boolean masks.

    Returns ``0.0`` when either mask is empty or the shapes differ. A
    shape mismatch counts as "no overlap" so a perception-side
    resolution glitch never raises.
    """

    if a.shape != b.shape:
        return 0.0
    a_bool = a.astype(bool, copy=False)
    b_bool = b.astype(bool, copy=False)
    intersection = int(np.logical_and(a_bool, b_bool).sum())
    if intersection == 0:
        return 0.0
    union = int(np.logical_or(a_bool, b_bool).sum())
    if union == 0:
        return 0.0
    return intersection / union


@dataclass(frozen=True, slots=True)
class IoUCentroidTargetTracker:
    """Default :class:`TargetTracker`: pick the highest-IoU mask.

    Ties on IoU are broken by smaller centroid distance. Learned
    trackers slot in by implementing the :class:`TargetTracker` Protocol.

    Attributes
    ----------
    max_centroid_distance_px
        Hard cap on the pixel distance between the identity's centroid
        and the candidate's centroid. Candidates beyond this distance
        are rejected even if their IoU clears ``iou_threshold``. This
        prevents a coincidental high-IoU match from an overlapping
        object on the other side of the bin.
    """

    max_centroid_distance_px: float = 200.0

    def __post_init__(self) -> None:
        if self.max_centroid_distance_px < 0.0:
            raise ValueError(
                "max_centroid_distance_px must be non-negative; "
                f"got {self.max_centroid_distance_px}"
            )

    def match(
        self,
        identity: TargetIdentity,
        frame: "PerceptionFrame",
        *,
        iou_threshold: float,
        camera_to_base: Optional[Transform] = None,
    ) -> Optional[tuple[int, float]]:
        del camera_to_base  # image-space tracker; the 3D transform is for the WorldSpacePoseTracker
        best_idx: Optional[int] = None
        best_iou: float = -1.0
        best_dist: float = math.inf
        for idx, seg in enumerate(frame.segmentations):
            cand_mask = np.asarray(seg.mask)
            iou = _binary_iou(identity.mask, cand_mask)
            if iou < iou_threshold:
                continue
            ys, xs = np.where(cand_mask.astype(bool, copy=False))
            if ys.size == 0:
                continue
            cand_centroid = (float(xs.mean()), float(ys.mean()))
            dx = cand_centroid[0] - identity.centroid_xy[0]
            dy = cand_centroid[1] - identity.centroid_xy[1]
            dist = math.hypot(dx, dy)
            if dist > self.max_centroid_distance_px:
                continue
            if iou > best_iou or (iou == best_iou and dist < best_dist):
                best_iou = iou
                best_dist = dist
                best_idx = idx
        if best_idx is None:
            logger.warning(
                "Target lost in image space: none of %d segmentation(s) cleared "
                "iou>=%.2f within %.0f px of the tracked centroid",
                len(frame.segmentations),
                iou_threshold,
                self.max_centroid_distance_px,
            )
            return None
        logger.debug(
            "Target re-identified in image space: segmentation %d, iou=%.3f",
            best_idx,
            best_iou,
        )
        return best_idx, best_iou


@dataclass(frozen=True, slots=True)
class WorldSpacePoseTracker:
    """Viewpoint-invariant :class:`TargetTracker`: matches by 3D pose in the BASE frame, not pixels.

    The image-space :class:`IoUCentroidTargetTracker` returns ``TARGET_LOST`` whenever the standoff
    re-perceive shifts the target's image location (eye-in-hand: the wrist camera moves with the arm), because
    the mask IoU collapses. This tracker back-projects each candidate mask's centroid to a 3D base-frame point
    (``frame.depth_map`` + ``frame.intrinsics`` + ``camera_to_base``) and matches the candidate whose 3D
    centroid is closest to the identity's 3D centroid within ``max_pose_distance_mm``. Tolerances are in mm,
    not pixels, so a viewpoint change does not lose the target.

    A re-perceive/refine loop that survives camera motion is what multi-view perception (views_seen /
    reobserve) and a reobserve recovery rest on.

    Strict superset of the default: when the 3D signal is unavailable (no identity ``centroid_xyz_mm``, the
    frame lacks usable depth/intrinsics, or nothing matches within tolerance) it falls back to the embedded
    :class:`IoUCentroidTargetTracker`, so it is never worse than image-space matching.
    """

    max_pose_distance_mm: float = 50.0
    iou_fallback: IoUCentroidTargetTracker = field(default_factory=IoUCentroidTargetTracker)

    def __post_init__(self) -> None:
        if self.max_pose_distance_mm <= 0.0:
            raise ValueError(
                f"max_pose_distance_mm must be positive; got {self.max_pose_distance_mm}"
            )

    def match(
        self,
        identity: TargetIdentity,
        frame: "PerceptionFrame",
        *,
        iou_threshold: float,
        camera_to_base: Optional[Transform] = None,
    ) -> Optional[tuple[int, float]]:
        debug = bool(os.environ.get("WILLY_POSE_TRACKER_DEBUG"))
        ref_xyz = identity.centroid_xyz_mm
        depth = getattr(frame, "depth_map", None)
        intrinsics = getattr(frame, "intrinsics", None)
        if ref_xyz is None or depth is None or intrinsics is None:
            # No 3D signal: never worse than the default image-space tracker.
            if debug:
                print(
                    f"[pose-tracker] no-3D-signal -> IoU fallback (ref_xyz={ref_xyz}, "
                    f"depth={'set' if depth is not None else None}, "
                    f"intr={'set' if intrinsics is not None else None})",
                    flush=True,
                )
            logger.debug(
                "No 3D signal for pose tracking (centroid=%s, depth=%s, intrinsics=%s); "
                "falling back to image-space matching",
                ref_xyz is not None,
                depth is not None,
                intrinsics is not None,
            )
            return self.iou_fallback.match(identity, frame, iou_threshold=iou_threshold)
        ref = np.asarray(ref_xyz, dtype=np.float64)
        best_idx: Optional[int] = None
        best_dist: float = math.inf
        for idx, seg in enumerate(frame.segmentations):
            cand_xyz = _centroid_xyz_base_mm(
                getattr(seg, "mask", None), depth, intrinsics, camera_to_base
            )
            if debug:
                print(f"[pose-tracker] cand[{idx}] xyz={cand_xyz} ref={ref_xyz}", flush=True)
            if cand_xyz is None:
                continue
            dist = float(np.linalg.norm(np.asarray(cand_xyz, dtype=np.float64) - ref))
            if dist <= self.max_pose_distance_mm and dist < best_dist:
                best_dist = dist
                best_idx = idx
        if debug:
            print(
                f"[pose-tracker] best_idx={best_idx} best_dist={best_dist:.1f} "
                f"tol={self.max_pose_distance_mm} cam2base={'yes' if camera_to_base else 'no'}",
                flush=True,
            )
        if best_idx is None:
            # Nothing within the 3D tolerance: the camera may not have moved, and image-space can
            # still re-identify. Falling back keeps this a strict superset of the default tracker.
            logger.info(
                "No candidate within %.0f mm of the tracked 3D centroid (%d segmentation(s)); "
                "falling back to image-space matching",
                self.max_pose_distance_mm,
                len(frame.segmentations),
            )
            return self.iou_fallback.match(identity, frame, iou_threshold=iou_threshold)
        # Confidence in [0, 1]: 1.0 at zero distance, 0.0 at the tolerance boundary.
        confidence = max(0.0, 1.0 - best_dist / self.max_pose_distance_mm)
        logger.debug(
            "Target re-identified in 3D: segmentation %d at %.1f mm (confidence %.2f)",
            best_idx,
            best_dist,
            confidence,
        )
        return best_idx, confidence
