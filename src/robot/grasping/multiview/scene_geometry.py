"""Every object's surface, as every fixed camera that can identify it sees it.

This is the consumer side of :mod:`~src.robot.grasping.multiview.association`. That module decides
which blob in the other camera is the same physical object, and this one turns that decision into
what the grasp generator needs: one BASE-frame cloud per candidate object, carrying the sides a
single depth view cannot see.

It matters more than any scoring knob. A candidate grasp is accepted or rejected on its closing
axis, the closing axis comes from the object's silhouette, and one depth view only ever observes one
side of an object while an antipodal grasp needs two. Measured on the datagen reference through the
real calculator over 354 objects, top-1 is 43.50 % single-view against 55.93 % fused, and coverage
53.67 % against 64.69 %. That is roughly three times the best single scoring change measured in the
same pass, and it is structural rather than incidental: no ranker can recover a contact face that
was never proposed.

The same observation has a second half. Fusing the target surface alone makes approach blockages
fall by half and finger collisions rise by 69 %, and the fused gap analysis over 345 objects then
attributes 60.0 % of every remaining mis-rank to ``finger_collision``. The mechanism is not subtle:
fusion hands the generator the object's full surface, so it proposes the side-face grasps a top-down
view never could, and those are exactly the grasps whose fingers swing into a neighbour, while the
fused target cloud adds no neighbour geometry to the collision filter. The control that settles it
is the ``sparse`` family, whose gap falls to exactly 0.0 %: with nothing standing nearby,
post-fusion ranking is already perfect. So the residual is occluded neighbours, and
:func:`fuse_scene_geometry` optionally returns those too.

Returning them is only half a fix. Feeding those neighbour clouds to the collision filter over 1073
reference objects rescues 5 grasps and destroys 4, a net gain of 3, while ``finger_collision`` stays
at 59.0 % of the gap. Replaying the reference verdict on the survivors says why: 60.9 % of them hit
a bin wall rather than an object, and inside the ``bin`` family it is 14 of 14. A wall is not an
instance any camera segments, so no cross-camera fusion can supply it, and nothing in this stack
carries wall geometry to supply it from. Where the blocker really is an object the lever works as
designed and the ``pile`` gap halves, from 8 to 4. That is why ``neighbour_scene_enabled`` is a
separate default-off switch rather than part of the fusion everyone should run.

The scope is deliberately narrow: pure functions over already-resolved inputs, meaning masks, depth,
intrinsics and one ``CAMERA`` to ``BASE`` matrix per camera. Resolving those, with frame resolvers
and a live TCP for a wrist camera, and deciding what to do when a camera is missing, are policy, and
policy lives with the config in the orchestrator. Frames are BASE millimetres throughout, per repo
convention.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.robot.grasping.constants import (
    SCENE_GEOMETRY_LOG_FILE,
    create_grasping_logger,
)
from src.robot.grasping.geometry.pointcloud import masked_points
from src.robot.grasping.multiview.association import (
    DEFAULT_MIN_SCORE,
    DEFAULT_NEIGHBOUR_MM,
    AssociationMetric,
    SceneAssociation,
    SceneCluster,
    ViewCandidates,
    fuse_scene_clouds,
)

#: Which cameras contributed, and how many objects gained a second view, is the
#: fusion's honest record, and the difference between the 43.5 % and the 55.9 %
#: top-1 measured above. One line per fusion call, never per object.
logger = create_grasping_logger("SceneGeometry", SCENE_GEOMETRY_LOG_FILE)

__all__ = [
    "FusedSceneGeometry",
    "ObservedView",
    "SceneObject",
    "build_scene_objects",
    "fuse_scene_geometry",
    "to_base_mm",
]


@dataclass(frozen=True, slots=True)
class ObservedView:
    """One camera's segmented objects plus everything needed to place them in BASE."""

    name: str
    masks: tuple[np.ndarray, ...]
    depth_map: np.ndarray
    intrinsics: np.ndarray
    camera_to_base: np.ndarray
    #: The segmentation objects the masks came from, in the same order, when the caller has them.
    #:
    #: The mask alone is enough to fuse a surface and not enough to grasp from. Every other camera
    #: used to be reduced to `.mask` here, so its labels and detection scores were discarded and an
    #: object only it could see had nothing to become: the calculator is called with a segmentation,
    #: this camera's depth and this camera's intrinsics, and a bare mask supplies one of the three.
    #:
    #: Empty is the old shape and stays valid: fusing a surface onto an object the primary already
    #: found needs no segmentation at all, so a caller that only fuses passes masks and nothing else.
    segmentations: tuple[Any, ...] = ()

    def segmentation(self, index: int) -> "Any | None":
        """The segmentation behind mask ``index``, or ``None`` when this view carries only masks."""
        return self.segmentations[index] if index < len(self.segmentations) else None


@dataclass(frozen=True, slots=True)
class FusedSceneGeometry:
    """One fused cloud per primary object, plus what it was built from.

    ``clouds_base_mm[i]`` is :data:`None` when object ``i`` gained nothing, meaning no other camera
    could identify it. :data:`None` rather than the object's own cloud is the point: the caller then
    omits the generator kwarg entirely and the single-view path stays unchanged instead of being
    re-fed a copy of what it already had.

    ``neighbour_clouds_base_mm`` is the same idea for the other side of the pick, what object ``i``
    would collide with. It is empty unless the caller asked for it.
    """

    clouds_base_mm: tuple[np.ndarray | None, ...]
    views_used: tuple[str, ...]
    associations: tuple[SceneAssociation, ...]
    neighbour_clouds_base_mm: tuple[np.ndarray | None, ...] = ()

    @property
    def objects_fused(self) -> int:
        return sum(1 for cloud in self.clouds_base_mm if cloud is not None)

    @property
    def neighbour_points(self) -> int:
        return sum(len(cloud) for cloud in self.neighbour_clouds_base_mm if cloud is not None)

    def cloud_for(self, index: int) -> np.ndarray | None:
        """The fused surface of object ``index``, or ``None``. Bounds-safe by design.

        The caller indexes by segmentation and the two counts can disagree, because a segmentation
        with no mask never becomes a primary, so an out-of-range index means nothing was fused for
        that one rather than a crash in the pick path.
        """

        if 0 <= index < len(self.clouds_base_mm):
            return self.clouds_base_mm[index]
        return None

    def neighbour_for(self, index: int) -> np.ndarray | None:
        if 0 <= index < len(self.neighbour_clouds_base_mm):
            return self.neighbour_clouds_base_mm[index]
        return None


def to_base_mm(
    mask: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_base: np.ndarray,
) -> np.ndarray:
    """One segmented object's surface in BASE millimetres, ``(N, 3)``.

    Empty in, empty out: a mask that selects no valid depth yields ``(0, 3)`` rather than raising,
    because a camera seeing nothing is an ordinary outcome of a real cell, not an error.
    """

    points_cam = np.asarray(masked_points(mask, depth_map, intrinsics), dtype=np.float64)
    if points_cam.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    matrix = np.asarray(camera_to_base, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"camera_to_base must be shape (4, 4), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("camera_to_base must contain only finite values")
    return points_cam @ matrix[:3, :3].T + matrix[:3, 3]


def _neighbour_clouds(
    views: Sequence[ViewCandidates],
    associations: Sequence[SceneAssociation],
    primary_count: int,
    voxel_mm: float,
) -> tuple[np.ndarray | None, ...]:
    """For each primary object: everything the other cameras saw, minus that object itself.

    Two rules make this safe to feed a collision filter, and they are deliberately asymmetric,
    because the three ways to be wrong here cost wildly different amounts.

    What is excluded is decided more generously than what is fused. Fusing the wrong blob corrupts
    the target's surface, and dropping one real obstacle loses one rejection, but treating the target
    as its own obstacle puts points exactly where the fingers have to close and kills nearly every
    candidate for that object. The last is by far the worst, and it is reachable: the association
    abstains on a measured 8.5 % of object and view pairs, and a target the assignment could not
    confirm would otherwise walk straight into its own obstacle cloud. So a candidate is excluded
    when it was assigned to this object, and also when it is merely this object's best match in that
    view with any overlap at all. A candidate scoring exactly zero shares no surface with the target,
    so it stays an obstacle.

    Objects nobody matched stay in. A blob some other camera saw and the primary view never did is
    not an error to be filtered out, it is precisely the hidden collider this seam exists to surface.

    Nothing from the primary view is included: the calculator already builds those from
    ``other_object_masks`` and vstacks them itself, so adding them here would only duplicate points.
    """

    from src.robot.grasping.geometry.sampling import voxel_downsample_indices

    # One flat list over (view, candidate) so the per-object exclusion below is a boolean mask on a
    # single array rather than a vstack per object. Downsampled per candidate, not over the union,
    # so a voxel shared by two objects cannot let one of them delete the other's evidence.
    clouds: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    flat_id: dict[tuple[int, int], int] = {}
    for view_index, view in enumerate(views):
        for candidate_index, cloud in enumerate(view.clouds_base_mm):
            points = np.asarray(cloud, dtype=np.float64).reshape(-1, 3)
            if not points.size:
                continue
            if voxel_mm > 0.0:
                points = points[voxel_downsample_indices(points, voxel_mm)]
            flat_id[(view_index, candidate_index)] = len(clouds)
            owners.append(np.full(len(points), len(clouds), dtype=np.int64))
            clouds.append(points)

    if not clouds:
        return tuple(None for _ in range(primary_count))

    all_points = np.vstack(clouds)
    owner = np.concatenate(owners)

    result: list[np.ndarray | None] = []
    for index in range(primary_count):
        excluded: set[int] = set()
        for view_index, association in enumerate(associations):
            row = association.scores[index] if index < len(association.scores) else ()
            assigned = association.assignment[index] if index < len(association.assignment) else None
            if assigned is not None:
                excluded.add(flat_id.get((view_index, assigned), -1))
            if row:
                best = int(np.argmax(np.asarray(row, dtype=np.float64)))
                if row[best] > 0.0:
                    excluded.add(flat_id.get((view_index, best), -1))
        excluded.discard(-1)
        keep = ~np.isin(owner, sorted(excluded)) if excluded else np.ones(len(owner), dtype=bool)
        points = all_points[keep]
        result.append(points if points.size else None)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class SceneObject:
    """One physical object, and the camera whose reading of it a grasp is synthesised from.

    The point of the type: the pick loop used to iterate the primary camera's segmentation list, and
    every consumer downstream was an integer index into it. So an object existed only if the primary
    camera had a segmentation for it, and "which object" and "which position in one camera's list"
    were the same fact. They are separated here. The index becomes a position, which is all it was
    ever entitled to be.

    ``camera_id``, ``segmentation``, ``depth_map`` and ``intrinsics`` belong together and come from
    one camera: the primary whenever the primary saw this object, which is every object in a
    single-camera cell and most of them in any cell. For an object the primary could not see, they
    are the camera that did see it, and the grasp is synthesised in that camera's frame with that
    camera's calculator. That is why they travel as a group rather than as four arguments a caller
    could accidentally mix.
    """

    camera_id: str
    segmentation: Any
    depth_map: np.ndarray
    intrinsics: np.ndarray
    camera_to_base: np.ndarray
    #: Every camera that agreed this is the same object, including ``camera_id``, in view order.
    views_used: tuple[str, ...]
    #: True when no primary segmentation corresponds to this object: it is in the list because a
    #: secondary camera saw it and the primary did not. Logged per candidate rather than acted on,
    #: so the question "is a promoted object worth ranking normally" is answered from data later.
    promoted: bool
    #: The surface fused across ``views_used``, or ``None`` when only one camera saw this object.
    #: ``None`` rather than the object's own cloud, for the same reason as in `FusedSceneGeometry`:
    #: the caller then omits the generator kwarg entirely instead of re-feeding it what it already
    #: derives from the mask.
    fused_cloud_base_mm: "np.ndarray | None" = None


def build_scene_objects(
    views: Sequence[ObservedView],
    clusters: Sequence["SceneCluster"],
    *,
    primary_index: int = 0,
    fused_clouds: "Sequence[np.ndarray | None] | None" = None,
    view_clouds: "Sequence[Sequence[np.ndarray]] | None" = None,
) -> tuple[SceneObject, ...]:
    """Turn clusters into the list a pick iterates, primary objects first and in their own order.

    Order is a contract here, not a convenience. Every object the primary camera saw comes first, in
    the primary's own segmentation order, so a cell whose secondary cameras add nothing produces
    exactly the list it produced before this existed, position for position. Promoted objects follow,
    ordered by the camera that saw them and then by that camera's detection order, which is
    deterministic for the same reason the clustering is.

    Which camera owns a cluster: the primary when it is a member, otherwise the first member's view.
    So the grasp is synthesised from the primary's reading wherever the primary has one, which is the
    behaviour that every measured number in this repository was taken against.

    ``view_clouds`` is each view's blobs already in BASE millimetres, indexed as the clusters index
    them. Given, a promoted object seen by more than one camera gets those views fused, exactly as a
    primary object does. Without it a promoted object would be grasped from one view while two were
    available, which is the very deficit fusion exists to close, left open for the objects that need
    it most: an object the primary cannot see is by construction one that something is in front of.
    """

    by_primary_blob: dict[int, SceneCluster] = {}
    promoted: list[SceneCluster] = []
    for cluster in clusters:
        owner = next((blob for view, blob in cluster.members if view == primary_index), None)
        if owner is None:
            promoted.append(cluster)
        else:
            by_primary_blob[owner] = cluster

    objects: list[SceneObject] = []
    primary = views[primary_index] if primary_index < len(views) else None
    if primary is not None:
        for blob in range(len(primary.masks)):
            grouped = by_primary_blob.get(blob)
            objects.append(SceneObject(
                camera_id=primary.name,
                segmentation=primary.segmentation(blob),
                depth_map=primary.depth_map,
                intrinsics=primary.intrinsics,
                camera_to_base=primary.camera_to_base,
                views_used=(tuple(views[v].name for v in grouped.views) if grouped is not None
                            else (primary.name,)),
                promoted=False,
                fused_cloud_base_mm=(
                    fused_clouds[blob] if fused_clouds is not None and blob < len(fused_clouds)
                    else None),
            ))

    for cluster in sorted(promoted, key=lambda c: c.members):
        view_index, blob = cluster.members[0]
        view = views[view_index]
        objects.append(SceneObject(
            camera_id=view.name,
            segmentation=view.segmentation(blob),
            depth_map=view.depth_map,
            intrinsics=view.intrinsics,
            camera_to_base=view.camera_to_base,
            views_used=tuple(views[v].name for v in cluster.views),
            promoted=True,
            fused_cloud_base_mm=_fused_member_cloud(cluster, view_clouds),
        ))
    return tuple(objects)


def _fused_member_cloud(
    cluster: "SceneCluster", view_clouds: "Sequence[Sequence[np.ndarray]] | None"
) -> "np.ndarray | None":
    """Every camera's points for one cluster, stacked, or ``None`` when only one camera saw it.

    ``None`` for a single member rather than that member's own points, which is the same rule the
    primary objects follow: the caller then omits the generator kwarg and the calculator derives the
    cloud from the mask and the depth it was given, instead of being handed back the very points it
    would have derived. Handing them over would not be wrong, it would be a second array and a
    silently different code path for an identical result.

    The owning view's own points come first, so the fused cloud starts where the grasp is being
    synthesised. `fuse_scene_clouds` orders the primary's points the same way for the same reason.
    """
    if view_clouds is None or len(cluster.members) < 2:
        return None
    owner = cluster.members[0]
    ordered = [owner, *(m for m in cluster.members if m != owner)]
    parts = [
        np.asarray(view_clouds[view][blob], dtype=np.float64).reshape(-1, 3)
        for view, blob in ordered
        if view < len(view_clouds) and blob < len(view_clouds[view])
    ]
    parts = [cloud for cloud in parts if cloud.size]
    return np.vstack(parts) if len(parts) > 1 else None


def fuse_scene_geometry(
    primary_masks: Sequence[np.ndarray],
    primary_depth_map: np.ndarray,
    primary_intrinsics: np.ndarray,
    primary_camera_to_base: np.ndarray,
    other_views: Sequence[ObservedView],
    *,
    metric: AssociationMetric = AssociationMetric.OVERLAP,
    min_score: float = DEFAULT_MIN_SCORE,
    neighbour_mm: float = DEFAULT_NEIGHBOUR_MM,
    max_centroid_mm: float = 150.0,
    with_neighbours: bool = False,
    neighbour_voxel_mm: float = 8.0,
    score_voxel_mm: float = 0.0,
) -> FusedSceneGeometry:
    """Fuse every primary object with whatever the other cameras can confirm about it.

    ``primary_masks`` are the segmentations of the camera the grasp is being synthesised from, in
    that camera's own order; the returned clouds are in the same order, so a caller can index them
    by segmentation without a lookup table.

    A view whose ``CAMERA`` to ``BASE`` transform is unusable, or which segmented nothing,
    contributes nothing and is absent from :attr:`FusedSceneGeometry.views_used`, the honest record
    of what the fusion was built from and what the telemetry stamps.

    ``with_neighbours`` additionally returns, per object, the other cameras' view of everything that
    is not that object, as :func:`_neighbour_clouds` builds it, which is the obstacle side of the
    same observation. It is off by default: it costs a second pass over every candidate cloud, and a
    caller that only feeds the generator has no use for it.
    """

    primary_clouds = [
        to_base_mm(mask, primary_depth_map, primary_intrinsics, primary_camera_to_base)
        for mask in primary_masks
    ]
    if not primary_clouds:
        logger.debug("Fusion skipped: the primary view segmented nothing")
        return FusedSceneGeometry((), (), ())

    candidates: list[ViewCandidates] = []
    used: list[str] = []
    for view in other_views:
        clouds = tuple(
            cloud
            for cloud in (
                to_base_mm(mask, view.depth_map, view.intrinsics, view.camera_to_base)
                for mask in view.masks
            )
            if cloud.size
        )
        if not clouds:
            continue
        candidates.append(ViewCandidates(view.name, clouds))
        used.append(view.name)

    if not candidates:
        # Not an error, since a one-camera cell is a legitimate deployment, but it means
        # every grasp in this pick is synthesised from one side of the object only.
        logger.warning(
            "No other view contributed to fusion (%d offered, %d primary object(s)); "
            "single-view geometry stands",
            len(other_views),
            len(primary_clouds),
        )
        return FusedSceneGeometry(tuple(None for _ in primary_clouds), (), ())

    fused, associations = fuse_scene_clouds(
        primary_clouds,
        candidates,
        metric=metric,
        min_score=min_score,
        neighbour_mm=neighbour_mm,
        max_centroid_mm=max_centroid_mm,
        score_voxel_mm=score_voxel_mm,
    )
    # An object nothing confirmed comes back from fuse_scene_clouds as its own cloud unchanged.
    # Report it as None so the caller can leave the generator kwarg off entirely rather than hand
    # back the very points the generator already derives from the mask.
    gained = [
        any(association.assignment[index] is not None for association in associations)
        for index in range(len(primary_clouds))
    ]
    fused_or_none: tuple[np.ndarray | None, ...] = tuple(
        cloud if gained[index] else None for index, cloud in enumerate(fused)
    )
    neighbours = (
        _neighbour_clouds(candidates, associations, len(primary_clouds), neighbour_voxel_mm)
        if with_neighbours
        else ()
    )
    result = FusedSceneGeometry(fused_or_none, tuple(used), associations, neighbours)
    logger.info(
        "Fused %d/%d object(s) from view(s) %s (metric=%s, neighbours=%s, %d neighbour point(s))",
        result.objects_fused,
        len(primary_clouds),
        ", ".join(used),
        metric.value if hasattr(metric, "value") else metric,
        with_neighbours,
        result.neighbour_points,
    )
    return result
