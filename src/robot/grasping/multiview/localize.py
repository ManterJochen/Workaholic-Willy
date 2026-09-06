"""Multi-view target localization: a visibility-weighted BASE-frame centroid fusion.

The reusable core of the switchable multi-view pick. Given each active camera's BASE-frame estimate of
the prompted target's centroid, already back-projected through that camera's own ``CAMERA`` to ``BASE``
transform, together with how many target pixels it saw, the views fuse into one BASE-frame target
point. A camera that did not see the target, because the robot arm or a stacked neighbour occluded it
or it fell out of frame, reports ``centroid_base_mm=None`` with ``visible_px=0`` and contributes
nothing, so the fused point is carried by whichever cameras did see it. That redundancy is what lets a
fixed oblique resolve a target the overhead loses, measured in ``run_occlusion_probe`` and
``run_multiview_pick``.

Pure, deterministic and numpy-only: no Isaac, no camera objects and no perception models. Those live in
the runner, which builds the :class:`ViewLocalization` list and calls :func:`fuse_view_localizations`,
so the fusion logic is reusable by any caller, a runner today and the config-driven orchestrator later.

Determinism: the fusion is a visibility-weighted mean over the views in input order, a stable reduction
with no set iteration and no reordering. For a fixed input the result is bit-stable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["ViewLocalization", "fuse_scene_points_base", "fuse_view_localizations"]


@dataclass(frozen=True, slots=True)
class ViewLocalization:
    """One camera's BASE-frame estimate of the target centroid plus its visibility weight.

    ``centroid_base_mm`` is the target's BASE-frame XYZ in millimetres, or ``None`` when the camera did
    not see it. ``visible_px`` is the visible mask pixel count, which serves as both the fusion weight
    and the signal for whether the camera saw the target at all.
    """

    name: str
    centroid_base_mm: "np.ndarray | None"
    visible_px: int

    @property
    def saw_target(self) -> bool:
        """True exactly when this view contributes to the fusion: a centroid that is not None, and more than zero visible pixels."""
        return self.centroid_base_mm is not None and self.visible_px > 0


def fuse_view_localizations(views: Sequence[ViewLocalization]) -> "np.ndarray | None":
    """Visibility-weighted mean of the per-view BASE-frame target centroids.

    Each contributing view, meaning each one where ``saw_target`` holds, is weighted by its
    ``visible_px``; an occluded view at 0 px or ``None`` is ignored. Returns the fused BASE-frame XYZ
    in millimetres as a length-3 ``float64`` array, or ``None`` when no view saw the target, which the
    caller treats as being unable to localize it.
    """
    pts: list[np.ndarray] = []
    weights: list[float] = []
    for v in views:
        if v.saw_target:
            pts.append(np.asarray(v.centroid_base_mm, dtype=np.float64).reshape(3))
            weights.append(float(v.visible_px))
    if not pts:
        return None
    w = np.asarray(weights, dtype=np.float64)
    return (np.stack(pts) * w[:, None]).sum(axis=0) / w.sum()


def fuse_scene_points_base(
    per_camera_clouds: Sequence["np.ndarray | None"],
) -> "np.ndarray | None":
    """Concatenate the per-camera BASE-frame scene clouds into one fused cloud, in millimetres.

    Each fixed camera, the overhead one and the obliques, back-projects the neighbour object masks,
    meaning every object except the grasp target, to BASE-frame points. Stacking them gives the
    full-scene geometry the single grasp-synthesis view cannot see. Returns an ``(N, 3)`` float64
    array, or ``None`` when no camera contributed any points, in which case the orchestrator feeds
    nothing extra to the collision filter. Empty and ``None`` per-camera entries are skipped, and the
    order is preserved so the concatenation is deterministic.
    """
    clouds: list[np.ndarray] = []
    for c in per_camera_clouds:
        if c is None:
            continue
        arr = np.asarray(c, dtype=np.float64).reshape(-1, 3)
        if arr.size > 0:
            clouds.append(arr)
    if not clouds:
        return None
    return np.vstack(clouds)
