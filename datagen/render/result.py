"""What a render returns: the engine-independent half of the rendering contract.

These two dataclasses are the boundary, and they carry no vendor type: numpy arrays, a `ViewOutcome`
from `views.py`, `ObjectLabel`s from `labels.py` and `MaterialParams` from `materials.py`.

What a second engine owes is less than it looks. The corpus the learned generator reads is built
from `scene.json`, `<view>_depth.png`, `<view>_instances.png`, an optional `<view>_arm.png`,
`grasps.jsonl` and `provenance.json`; `label-grasps` never opens an image at all and its only engine
input is `scene.json.settled_poses_mm_xyzw`. So a backend owes four things: depth in mm, per-object
silhouettes, settled poses, and the camera pose and intrinsics it used. Everything else datagen asks
of Isaac is colour, so `rgb` may stay `None` for every view without any consumer noticing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from datagen.render.labels import ObjectLabel
from datagen.render.materials import MaterialParams
from datagen.render.views import ViewOutcome

__all__ = ["SceneRenderResult", "ViewRender"]


@dataclass(frozen=True, slots=True)
class ViewRender:
    """One rendered view: the images plus how the view came to exist."""

    name: str
    outcome: ViewOutcome
    rgb: np.ndarray | None = None
    depth_mm: np.ndarray | None = None
    depth_noisy_mm: np.ndarray | None = None
    instance_map: np.ndarray | None = None
    #: Which pixels the arm occupies, from depth alone. Separate from `instance_map` because the arm
    #: is not an object to be grasped and must not take an instance id, and because it is the one
    #: thing in a scene whose geometry changes completely between scenes, so a corpus that files it
    #: under "background" hands a net a class with no consistent shape.
    arm_mask: np.ndarray | None = None
    labels: tuple[ObjectLabel, ...] = ()
    camera_position_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_look_at_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: CAMERA(CV optical) to BASE and the pinhole K, in the convention of :mod:`datagen.render.camera`.
    #: Carried explicitly so that nothing downstream has to infer them from the two fields above.
    #:
    #: This pair is what :mod:`datagen.render.equivalence` grades, and it grades it on an oblique
    #: view. A backend whose depth is right but whose principal point is off by one pixel
    #: reconstructs the world wrongly by a constant; an overhead view is structurally blind to that,
    #: because a principal-point shift cannot change recovered Z on a plane normal to the axis.
    camera_to_base: np.ndarray = field(default_factory=lambda: np.eye(4))
    intrinsics: np.ndarray = field(default_factory=lambda: np.eye(3))


@dataclass(frozen=True, slots=True)
class SceneRenderResult:
    """What happened to one scene. ``status`` is 'ok' or the reason it was rejected."""

    scene_id: str
    status: str
    views: tuple[ViewRender, ...] = ()
    settled_poses: dict[int, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = (
        field(default_factory=dict)
    )
    materials: dict[int, MaterialParams] = field(default_factory=dict)
    seconds: float = 0.0
    note: str = ""
    #: How the arm came to stand where it stands, or ``None`` when no arm was placed. Recorded so a
    #: consumer can tell a viewpoint's occlusion from a parked arm's, which look identical in an image.
    arm: Any = None
    #: Indices the layout placed that left the table and were removed before rendering. Recorded
    #: because it is the one way ``spec.objects`` and the labels can legitimately disagree, and an
    #: unexplained disagreement is indistinguishable from a labelling bug.
    dropped: tuple[int, ...] = ()
