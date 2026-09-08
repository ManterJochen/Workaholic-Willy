"""What the cameras see, in the shape the trajectory planner accepts.

The planner's world is axis-aligned-in-its-own-frame boxes and nothing else, and until now every box
it ever received was written by hand in a YAML file. This module is the other source: a depth frame
in, a list of boxes in BASE millimetres out, so a planner can route around an obstacle nobody
declared.

It is pure. Depth, intrinsics and a transform in, a frozen report out. No camera, no client, no
process, no clock. That matters because everything it decides is a safety decision made from data
nobody checked, and a pure function is the only kind that can be exercised against an adversarial
frame in a test.

Three things it must never do quietly, because each one hands a planner a world that is wrong in the
direction that hurts:

  1. Lose a box. Every cluster that does not survive is counted and named with the reason. An
     obstacle the planner never received is one it drives straight through, and the count of those
     belongs in the same report as the boxes that did survive.
  2. Trust a grasp depth. A producer may replace the depth inside each detected mask with one
     number, the object's top surface plus a penetration, because that is where a jaw is driven.
     Building a box from that yields a sheet at the top face and empty space where the body is. The
     views this module takes come off the camera handle rather than out of a `PerceptionFrame`, so
     the overwrite was never on this path at all: while every other stage received sheets, the
     planner's world had real geometry, for a reason nothing here said out loud. It is worth saying,
     because the fix that removed the overwrite did not change this module and could easily be read
     as having done so.
  3. Model the bench as an obstacle. Everything the camera sees includes the table the arm works
     over, and a table registered as a hundred small boxes both fills the planner's slots and
     duplicates the support plane the operator already declared. Points at or below the declared
     plane are dropped, by name, and the plane keeps its one clean box.

Units and frames at this boundary: input depth is millimetres in the CAMERA frame, the transform is
the usual 4x4 in millimetres, output is millimetres in BASE. The conversion to the planner's metres
and WXYZ happens in `world.py`, which owns that wire format for both sources.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Sequence

import numpy as np

__all__ = [
    "DepthView",
    "DropReason",
    "PerceivedBox",
    "PerceivedWorld",
    "PerceptionGeometryError",
    "SelfBody",
    "VoxelField",
    "WorldBuildLimits",
    "WorldBuildTuning",
    "build_perceived_boxes",
    "build_voxel_field",
    "voxel_grid_extent",
]


class PerceptionGeometryError(ValueError):
    """The frame cannot be turned into geometry, so no world can be built from it."""


class DropReason(StrEnum):
    """Why a point or a cluster did not become a box.

    Every one of these is a hole in what the planner is told, so they are named rather than counted
    into one number: "eleven points fell outside the workspace" and "one obstacle did not fit the
    slot budget" are the same subtraction and completely different news.
    """

    #: Inside a mask the caller asked to leave out, which is normally the object being grasped.
    EXCLUDED = "excluded"
    #: At or below the declared support plane: the bench, which is already one clean box.
    BELOW_PLANE = "below_plane"
    #: Outside the cell's declared reach, so the arm cannot meet it.
    OUTSIDE_LIMITS = "outside_limits"
    #: Too few points to be anything but sensor noise.
    TOO_FEW_POINTS = "too_few_points"
    #: On the robot's own body, or on what it is carrying.
    SELF = "self"
    #: Real, in reach, and there was no collision slot left for it.
    NO_SLOT = "no_slot"


@dataclass(frozen=True, slots=True)
class WorldBuildLimits:
    """Where in the cell a perceived obstacle is allowed to be.

    These are not tuning. They are the declaration of which part of the room the planner is
    responsible for, and they come from the same config the arm's workspace guard reads, so a box
    can never appear somewhere the arm was already forbidden to go.
    """

    #: The workspace box in BASE millimetres, as the guard states it.
    x_mm: tuple[float, float]
    y_mm: tuple[float, float]
    z_mm: tuple[float, float]
    #: Top surface of the declared support plane, in BASE millimetres, or `None` for no plane.
    #:
    #: Points within `plane_clearance_mm` above it are dropped with the bench. Without a plane every
    #: table point becomes an obstacle, which is why registering a perceived world on a cell that
    #: declared no plane is refused rather than attempted.
    support_plane_top_mm: float | None
    #: How far above the plane a point still counts as the plane.
    plane_clearance_mm: float = 5.0

    def __post_init__(self) -> None:
        for name, span in (("x_mm", self.x_mm), ("y_mm", self.y_mm), ("z_mm", self.z_mm)):
            lo, hi = (float(v) for v in span)
            if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
                raise PerceptionGeometryError(
                    f"{name} must be a finite (min, max) with max > min, got {span!r}"
                )
        if float(self.plane_clearance_mm) < 0.0:
            raise PerceptionGeometryError("plane_clearance_mm cannot be negative")


@dataclass(frozen=True, slots=True)
class WorldBuildTuning:
    """How coarse the geometry is, and how much of it fits.

    The defaults are chosen so that one pass over a full-resolution depth frame is cheap enough to
    sit in front of a plan: the frame is voxelised twice, once to thin the cloud and once to cluster
    it, and neither pass ever holds more than the voxel count.
    """

    #: Read every nth pixel of every frame. 1 reads all of them.
    #:
    #: Not an approximation as long as it stays under the voxel size in image terms. At a metre a
    #: D435 pixel is about 1.6 mm, so a 10 mm voxel spans six pixels and every second pixel fills the
    #: same voxels the full frame would. It is the cheapest lever there is: the cost of this whole
    #: conversion is dominated by how many pixels are back-projected, not by how many survive.
    pixel_stride: int = 1
    #: Cloud thinning before anything else. Smaller keeps more shape and costs more time.
    voxel_size_mm: float = 10.0
    #: The grid the connected-component pass runs on. Two points in touching cells are one object.
    cluster_voxel_mm: float = 25.0
    #: Below this a cluster is sensor noise rather than a thing.
    min_points: int = 12
    #: Grown on every side of every box. A box that is exactly the measured hull is a box the
    #: planner will graze, and depth noise is one-sided at an edge.
    margin_mm: float = 15.0
    #: Boxes the caller has slots for. The nearest survive; the rest are reported, never dropped in
    #: silence.
    max_boxes: int = 8
    #: Prefix every emitted name carries, so a perceived box can never collide with a declared
    #: fixture and silently replace it during the merge.
    name_prefix: str = "seen_"
    #: Edge length of a voxel in the distance field, millimetres. Zero builds no field at all,
    #: which is the byte-identical path and the one a cell without the planner storage takes.
    voxel_field_mm: float = 0.0
    #: Carry every box down to the support plane instead of stopping at the surface that was seen.
    #:
    #: A depth camera measures surfaces, not bodies. Looking down at a part on a bench it returns the
    #: part's top face and nothing else, so the honest box around those points is a sheet with the
    #: part's footprint and no height, and a planner routing around that sheet will happily drive a
    #: link through the part underneath it.
    #:
    #: Carrying the box down to the plane is the conservative reading: the space under a surface is
    #: either the object or somewhere the arm has no business being. It has one cost, and it is the
    #: cost worth knowing before switching this on. A container whose rim the camera sees becomes a
    #: solid block from the rim to the bench, and the cell can then never reach into it. A declared
    #: fixture does not have this problem, because the bin walls are already boxes with a hollow
    #: between them: this is about a container nobody declared.
    floor_to_plane: bool = True

    def __post_init__(self) -> None:
        if float(self.voxel_size_mm) <= 0.0 or float(self.cluster_voxel_mm) <= 0.0:
            raise PerceptionGeometryError("voxel sizes must be positive")
        if float(self.cluster_voxel_mm) < float(self.voxel_size_mm):
            raise PerceptionGeometryError(
                "cluster_voxel_mm must not be finer than voxel_size_mm: clustering on a grid finer "
                "than the cloud splits one object into one cluster per point"
            )
        if int(self.pixel_stride) < 1:
            raise PerceptionGeometryError("pixel_stride must be at least 1")
        if int(self.min_points) < 1:
            raise PerceptionGeometryError("min_points must be at least 1")
        if int(self.max_boxes) < 0:
            raise PerceptionGeometryError("max_boxes cannot be negative")
        if float(self.margin_mm) < 0.0:
            raise PerceptionGeometryError("margin_mm cannot be negative")
        if float(self.voxel_field_mm) < 0.0:
            raise PerceptionGeometryError("voxel_field_mm cannot be negative")


@dataclass(frozen=True, slots=True)
class DepthView:
    """One camera's reading of the cell, with what it takes to place it in the base frame.

    A cell with two cameras is not two worlds. The second camera exists because the first one cannot
    see behind the arm, into the far side of a tote, or under an overhang, and a box built from one
    view alone carries that blind spot into the planner. So views are combined before anything is
    clustered: a part half seen by each camera is one obstacle, not two overlapping ones.
    """

    #: The depth the camera measured, CAMERA frame, millimetres.
    surface_depth_mm: np.ndarray
    #: The 3x3 that depth was captured with.
    intrinsics: np.ndarray
    #: CAMERA to BASE, 4x4, millimetres.
    camera_to_base: np.ndarray
    #: Pixels to leave out of this view, normally the object being grasped.
    exclude_masks: tuple[np.ndarray, ...] = ()
    #: Named segmentations in this view, so a cluster can take an object's name.
    labelled_masks: tuple[tuple[str, np.ndarray], ...] = ()
    #: Which camera this is, for a message an operator has to act on.
    name: str = ""
    #: Shutter time, on the same clock the caller ages frames against.
    timestamp: float | None = None


@dataclass(frozen=True, slots=True)
class SelfBody:
    """Where the robot's own body is, so the camera's view of it can be taken back out.

    A fixed camera watching a cell sees the arm. Every one of those points is real geometry and
    registering it makes the planner refuse to move, because the arm is then standing inside an
    obstacle. After a successful close the same is true of the part in the gripper, which travels
    with the hand and is already modelled as an attached body on the planner side.

    The body arrives here as capsules, a segment and a radius each, rather than as a model name and
    a joint vector. This module then needs no kinematics, no vendor and no config, and a driver that
    cannot describe its own links is a refusal for its caller to make rather than a guess for this
    one.
    """

    #: `(K, 2, 3)` segment endpoints in BASE millimetres.
    segments_mm: np.ndarray
    #: `(K,)` radius per segment, millimetres, generous rather than exact.
    radii_mm: np.ndarray

    def __post_init__(self) -> None:
        segments = np.asarray(self.segments_mm, dtype=np.float64)
        radii = np.asarray(self.radii_mm, dtype=np.float64)
        if segments.ndim != 3 or segments.shape[1:] != (2, 3):
            raise PerceptionGeometryError(
                f"segments_mm must be (K, 2, 3), got {segments.shape}"
            )
        if radii.shape != (segments.shape[0],):
            raise PerceptionGeometryError(
                f"radii_mm must be ({segments.shape[0]},), got {radii.shape}"
            )
        if segments.size and not np.all(np.isfinite(segments)):
            raise PerceptionGeometryError("segments_mm must be finite")
        if radii.size and np.any(radii < 0.0):
            raise PerceptionGeometryError("radii_mm cannot be negative")
        object.__setattr__(self, "segments_mm", segments)
        object.__setattr__(self, "radii_mm", radii)

    @classmethod
    def from_polyline(
        cls,
        points_mm: Sequence[Sequence[float]],
        *,
        radius_mm: float,
        tool_radius_mm: float | None = None,
    ) -> "SelfBody":
        """Capsules along a chain of link origins, which is what a forward kinematic returns.

        `tool_radius_mm` widens the last capsule alone. That is where the gripper is, and after a
        close it is also where the part is: both are wider than the wrist they hang off, and neither
        is worth a second geometry when a wider radius covers them.
        """
        chain = np.asarray(points_mm, dtype=np.float64)
        if chain.ndim != 2 or chain.shape[1] != 3 or chain.shape[0] < 2:
            raise PerceptionGeometryError(
                f"a self body needs at least two 3D points, got shape {chain.shape}"
            )
        segments = np.stack((chain[:-1], chain[1:]), axis=1)
        radii = np.full(segments.shape[0], float(radius_mm), dtype=np.float64)
        if tool_radius_mm is not None:
            radii[-1] = float(tool_radius_mm)
        return cls(segments_mm=segments, radii_mm=radii)

    def contains(self, points_mm: np.ndarray) -> np.ndarray:
        """Which of `points_mm` lie inside the body. `(N,)` boolean.

        Point to segment, capsule by capsule. Six or seven capsules against a thinned cloud is one
        small matrix per capsule, which keeps the cost flat and predictable in front of a plan.
        """
        points = np.asarray(points_mm, dtype=np.float64)
        if points.size == 0:
            return np.zeros(points.shape[0], dtype=bool)
        inside = np.zeros(points.shape[0], dtype=bool)
        for (start, end), radius in zip(self.segments_mm, self.radii_mm):
            axis = end - start
            length_squared = float(axis @ axis)
            if length_squared <= 0.0:
                closest = np.broadcast_to(start, points.shape)
            else:
                travel = np.clip(((points - start) @ axis) / length_squared, 0.0, 1.0)
                closest = start + travel[:, None] * axis
            inside |= np.linalg.norm(points - closest, axis=1) <= radius
        return inside


@dataclass(frozen=True, slots=True)
class PerceivedBox:
    """One obstacle, upright, turned about BASE Z to sit closest around its cluster.

    The rotation is yaw only. A cell's obstacles stand on a floor, so the two axes worth fitting are
    the ones in the plane of the bench; a full three-axis fit on a noisy cluster tumbles between one
    frame and the next, and a planner that is handed a differently tumbled world every 50 ms plans a
    different path for a scene that did not move.
    """

    name: str
    #: Box centre in BASE millimetres.
    center_mm: tuple[float, float, float]
    #: Full side lengths in the box's own frame, millimetres, margin already included.
    dims_mm: tuple[float, float, float]
    #: Rotation about BASE Z, radians.
    yaw_rad: float
    #: How many cloud points it was fitted to, so a reader can tell a wall from three stray pixels.
    points: int
    #: Distance from the ranking reference to the box centre, millimetres.
    distance_mm: float
    #: The segmentation label that overlaps this cluster, when one does.
    label: str | None = None

    @property
    def enclosing_half_extents_mm(self) -> tuple[float, float, float]:
        """Half extents of the axis-aligned box that contains this turned one.

        For a consumer whose geometry has no rotation, which is what the capsule guards use. It is
        bigger than the turned box, never smaller, so a guard fed this can refuse a path the planner
        allowed but can never allow one the planner refused. That is the only direction of
        disagreement worth having between the thing that plans and the thing that decides.
        """
        cos_yaw, sin_yaw = abs(math.cos(self.yaw_rad)), abs(math.sin(self.yaw_rad))
        half_x, half_y, half_z = (d / 2.0 for d in self.dims_mm)
        return (
            half_x * cos_yaw + half_y * sin_yaw,
            half_x * sin_yaw + half_y * cos_yaw,
            half_z,
        )

    def to_dict(self) -> dict[str, Any]:
        """The wire view. A view of what was computed, never a second computation."""
        return {
            "name": self.name,
            "center_mm": list(self.center_mm),
            "dims_mm": list(self.dims_mm),
            "yaw_rad": float(self.yaw_rad),
            "points": int(self.points),
            "distance_mm": float(self.distance_mm),
            "label": self.label,
        }


@dataclass(frozen=True, slots=True)
class VoxelField:
    """The cell as a signed distance field over a grid, in the planner's own sign and order.

    The box channel carries as many obstacles as there are collision slots, and the caller then has
    to choose which ones matter. This carries everything the cameras saw at the resolution it was cut
    to, and nothing is left out. It costs one array.

    ⛔ The sign is positive INSIDE an obstacle and negative in free space, which is the opposite of
    the distance-to-obstacle a person would write, and the wrong sign fails silently. Measured on
    this repository's planner against one wall: written positive-inside the planner refused the path;
    written the intuitive way it registered without an error, reported success, and planned straight
    through the wall.

    The order is the planner's: x slowest, z fastest, over voxel centres that start half a voxel
    inside the low corner. Anything else puts the geometry somewhere the cell is not.
    """

    #: Flat float32 field, one value per voxel, in the order described above.
    field: np.ndarray
    #: Grid size in BASE millimetres.
    dims_mm: tuple[float, float, float]
    #: Edge length of one voxel, millimetres.
    voxel_size_mm: float
    #: Grid centre in BASE millimetres.
    center_mm: tuple[float, float, float]
    #: Voxels the cameras put something in, before any filling or margin.
    occupied: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(round(d / self.voxel_size_mm)) for d in self.dims_mm)  # type: ignore[return-value]

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
        nx, ny, nz = self.shape
        return (
            f"live scene: {nx} x {ny} x {nz} voxels at {self.voxel_size_mm:.0f} mm, "
            f"{self.occupied} occupied"
        )

    def to_dict(self) -> dict[str, Any]:
        """The wire view. The field itself is not in it: it travels as a file, not as a message."""
        return {
            "shape": list(self.shape),
            "dims_mm": list(self.dims_mm),
            "voxel_size_mm": float(self.voxel_size_mm),
            "center_mm": list(self.center_mm),
            "occupied": int(self.occupied),
        }


@dataclass(frozen=True, slots=True)
class PerceivedWorld:
    """The boxes a frame yielded, and everything it did not.

    `dropped` is the half that matters when something goes wrong: a planner that received six boxes
    out of nine is planning through three obstacles, and nothing else in the stack can notice.
    """

    boxes: tuple[PerceivedBox, ...]
    #: Points removed before clustering, per reason.
    dropped_points: dict[str, int]
    #: Clusters that did not become boxes, per reason.
    dropped_clusters: dict[str, int]
    #: Points that reached the clustering pass.
    considered_points: int
    #: Capture time of the frame this was built from, as the producer stamped it, or `None`.
    source_timestamp: float | None = None
    #: The same scene as a distance field, when the caller asked for one. The boxes are what a guard
    #: and an operator can read; this is what lets the planner see everything rather than the nearest
    #: few. Both come from one pass over one cloud, so they cannot describe different cells.
    voxels: "VoxelField | None" = None

    @property
    def is_empty(self) -> bool:
        return not self.boxes

    @property
    def dropped_obstacle_count(self) -> int:
        """Clusters that were real, in reach, and did not fit. The number that is a hole."""
        return int(self.dropped_clusters.get(DropReason.NO_SLOT, 0))

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
        if not self.boxes:
            head = "no perceived obstacle: nothing in reach above the support plane"
        else:
            head = f"{len(self.boxes)} perceived obstacle(s), nearest first:"
        rows = [
            f"  {b.name:<20} centre ({b.center_mm[0]:8.1f}, {b.center_mm[1]:8.1f}, "
            f"{b.center_mm[2]:8.1f}) mm  size ({b.dims_mm[0]:7.1f} x {b.dims_mm[1]:7.1f} x "
            f"{b.dims_mm[2]:7.1f}) mm  yaw {math.degrees(b.yaw_rad):6.1f} deg  {b.points:6d} pt"
            for b in self.boxes
        ]
        tail = []
        if self.dropped_obstacle_count:
            tail.append(
                f"  {self.dropped_obstacle_count} obstacle(s) did NOT fit the slot budget and are "
                "not registered: the planner will route through them"
            )
        for reason, count in sorted(self.dropped_clusters.items()):
            if reason != DropReason.NO_SLOT and count:
                tail.append(f"  {count} cluster(s) dropped: {reason}")
        return "\n".join([head, *rows, *tail])

    def to_dict(self) -> dict[str, Any]:
        """The wire view."""
        return {
            "boxes": [b.to_dict() for b in self.boxes],
            "dropped_points": dict(self.dropped_points),
            "dropped_clusters": dict(self.dropped_clusters),
            "considered_points": int(self.considered_points),
            "dropped_obstacles": self.dropped_obstacle_count,
            "source_timestamp": self.source_timestamp,
            "voxels": None if self.voxels is None else self.voxels.to_dict(),
        }


def build_perceived_boxes(
    *,
    views: Sequence[DepthView],
    limits: WorldBuildLimits,
    tuning: WorldBuildTuning | None = None,
    self_body: "SelfBody | None" = None,
    near_point_mm: Sequence[float] | None = None,
) -> PerceivedWorld:
    """Turn what the cameras see into the obstacles a planner should route around.

    Every view is back-projected and carried into BASE, and the points are then treated as one
    cloud. That is the whole of the fusion: a part one camera sees the front of and another sees the
    side of becomes one cluster and therefore one box, rather than two overlapping boxes that each
    cover part of it. It also means a second camera can only ever make an obstacle bigger and better
    placed, never smaller, which is the direction it is safe to be wrong in.

    Parameters
    ----------
    views
        One entry per camera, each with its own depth, intrinsics and transform. The depth is the
        surface the camera measured, not the grasp-referenced map: see this module header for why
        that one produces sheets. A cell that cannot resolve a camera transform has no business
        registering that camera view, so a missing transform is a refusal for the caller to make.
    limits
        Where an obstacle may be, and where the bench is.
    self_body
        The robot own links, and the part it carries, as capsules in BASE millimetres. A camera
        watching a cell sees the arm, and an arm registered as an obstacle is an arm that cannot
        move. Leaving this out is only right where no camera can see the robot at all.
    near_point_mm
        What "nearest" is measured from when the slot budget bites. Normally the goal of the motion
        about to be planned. Defaults to the base origin.

    Raises
    ------
    PerceptionGeometryError
        If a frame, an intrinsic matrix or a transform cannot be used, or if no support plane is
        declared. Every one of those is a state where guessing would produce a world that looks
        right and is not.
    """
    tuning = tuning or WorldBuildTuning()
    if not views:
        raise PerceptionGeometryError("a perceived world needs at least one camera view")
    if limits.support_plane_top_mm is None:
        raise PerceptionGeometryError(
            "no support plane is declared, so every point of the bench would become an obstacle. "
            "Declare safety.planning_world.support_plane before registering a perceived world."
        )

    dropped_points: dict[str, int] = {}
    per_view_points: list[np.ndarray] = []
    per_view_pixels: list[np.ndarray] = []
    per_view_index: list[np.ndarray] = []
    lookups: list[list[tuple[str, np.ndarray]]] = []

    for index, view in enumerate(views):
        depth = np.asarray(view.surface_depth_mm, dtype=np.float64)
        where = view.name or f"view {index}"
        if depth.ndim != 2 or depth.size == 0:
            raise PerceptionGeometryError(
                f"{where}: surface depth must be a non-empty 2D map, got {depth.shape}"
            )
        transform = np.asarray(view.camera_to_base, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise PerceptionGeometryError(
                f"{where}: camera_to_base must be a finite 4x4 in millimetres, got shape "
                f"{transform.shape}"
            )

        keep_mask = np.ones(depth.shape, dtype=bool)
        for mask in view.exclude_masks:
            excluded = np.asarray(mask).astype(bool)
            if excluded.shape != depth.shape:
                raise PerceptionGeometryError(
                    f"{where}: an exclude mask is {excluded.shape} and the depth map is "
                    f"{depth.shape}"
                )
            keep_mask &= ~excluded
        excluded_pixels = int(np.count_nonzero(~keep_mask))
        if excluded_pixels:
            dropped_points[DropReason.EXCLUDED] = (
                dropped_points.get(DropReason.EXCLUDED, 0) + excluded_pixels
            )

        points, pixels = _base_points(
            depth, view.intrinsics, transform, keep_mask, tuning.voxel_size_mm,
            stride=int(tuning.pixel_stride),
        )
        lookups.append(_label_lookup(view.labelled_masks, depth.shape, where))
        per_view_points.append(points)
        per_view_pixels.append(pixels)
        per_view_index.append(np.full(points.shape[0], index, dtype=np.int64))

    # The age of a fused world is the age of its oldest half. One unstamped view makes the whole
    # thing unstamped, because a world is only as current as the part of it nobody can date.
    stamps = [view.timestamp for view in views]
    oldest = (
        None if any(stamp is None for stamp in stamps)
        else min(float(stamp) for stamp in stamps if stamp is not None)
    )

    points_base = np.concatenate(per_view_points)
    pixels = np.concatenate(per_view_pixels)
    view_of = np.concatenate(per_view_index)

    def _empty() -> PerceivedWorld:
        return PerceivedWorld(
            boxes=(), dropped_points=dropped_points, dropped_clusters={},
            considered_points=0, source_timestamp=oldest,
        )

    if points_base.shape[0] == 0:
        return _empty()

    if self_body is not None:
        on_self = self_body.contains(points_base)
        dropped_points[DropReason.SELF] = int(np.count_nonzero(on_self))
        points_base, pixels, view_of = points_base[~on_self], pixels[~on_self], view_of[~on_self]
        if points_base.shape[0] == 0:
            return _empty()

    floor = float(limits.support_plane_top_mm) + float(limits.plane_clearance_mm)
    above = points_base[:, 2] > floor
    dropped_points[DropReason.BELOW_PLANE] = int(np.count_nonzero(~above))

    inside = (
        (points_base[:, 0] >= limits.x_mm[0]) & (points_base[:, 0] <= limits.x_mm[1])
        & (points_base[:, 1] >= limits.y_mm[0]) & (points_base[:, 1] <= limits.y_mm[1])
        & (points_base[:, 2] >= limits.z_mm[0]) & (points_base[:, 2] <= limits.z_mm[1])
    )
    dropped_points[DropReason.OUTSIDE_LIMITS] = int(np.count_nonzero(above & ~inside))

    keep = above & inside
    points, kept_pixels, kept_view = points_base[keep], pixels[keep], view_of[keep]
    if points.shape[0] == 0:
        return _empty()

    labels = _cluster(points, float(tuning.cluster_voxel_mm))
    reference = (
        np.asarray(near_point_mm, dtype=np.float64) if near_point_mm is not None
        else np.zeros(3, dtype=np.float64)
    )
    if reference.shape != (3,):
        raise PerceptionGeometryError(f"near_point_mm must be three numbers, got {near_point_mm!r}")

    candidates: list[PerceivedBox] = []
    dropped_clusters: dict[str, int] = {}
    for cluster_id in np.unique(labels):
        member = labels == cluster_id
        count = int(np.count_nonzero(member))
        if count < int(tuning.min_points):
            dropped_clusters[DropReason.TOO_FEW_POINTS] = (
                dropped_clusters.get(DropReason.TOO_FEW_POINTS, 0) + 1
            )
            continue
        centre, dims, yaw = _oriented_box(
            points[member],
            float(tuning.margin_mm),
            floor_mm=float(limits.support_plane_top_mm) if tuning.floor_to_plane else None,
        )
        candidates.append(
            PerceivedBox(
                name="",  # named once the survivors are known, so the numbering has no holes
                center_mm=centre,
                dims_mm=dims,
                yaw_rad=yaw,
                points=count,
                distance_mm=float(np.linalg.norm(np.asarray(centre) - reference)),
                label=_dominant_label(lookups, kept_pixels[member], kept_view[member]),
            )
        )

    candidates.sort(key=lambda box: box.distance_mm)
    survivors = candidates[: int(tuning.max_boxes)]
    if len(candidates) > len(survivors):
        dropped_clusters[DropReason.NO_SLOT] = len(candidates) - len(survivors)

    boxes = tuple(
        PerceivedBox(
            name=_box_name(tuning.name_prefix, index, box.label),
            center_mm=box.center_mm,
            dims_mm=box.dims_mm,
            yaw_rad=box.yaw_rad,
            points=box.points,
            distance_mm=box.distance_mm,
            label=box.label,
        )
        for index, box in enumerate(survivors)
    )
    return PerceivedWorld(
        boxes=boxes,
        dropped_points=dropped_points,
        dropped_clusters=dropped_clusters,
        considered_points=int(points.shape[0]),
        source_timestamp=oldest,
        voxels=(
            build_voxel_field(points, limits=limits, tuning=tuning)
            if tuning.voxel_field_mm > 0.0 else None
        ),
    )


def voxel_grid_extent(
    limits: WorldBuildLimits, voxel_size_mm: float
) -> "tuple[tuple[float, float, float], float] | None":
    """The grid a live scene will occupy, as ``(dims_mm, voxel_mm)``, or `None` for no grid.

    The planner allocates its voxel storage when it starts and never again, so the size and the
    resolution have to be decided before it is spawned and then not change. This is that decision, in
    one place, so the reservation and the field cannot disagree. They must not: a field that does not
    match the grid it is written into is geometry in the wrong place, and it registers without error.
    """
    if voxel_size_mm <= 0.0:
        return None
    floor_mm = float(limits.support_plane_top_mm or 0.0)
    high_z = max(limits.z_mm[1], floor_mm + voxel_size_mm)
    counts = [
        max(1, int(round((limits.x_mm[1] - limits.x_mm[0]) / voxel_size_mm))),
        max(1, int(round((limits.y_mm[1] - limits.y_mm[0]) / voxel_size_mm))),
        max(1, int(round((high_z - floor_mm) / voxel_size_mm))),
    ]
    dims = (counts[0] * voxel_size_mm, counts[1] * voxel_size_mm, counts[2] * voxel_size_mm)
    return dims, float(voxel_size_mm)


def build_voxel_field(
    points_mm: np.ndarray, *, limits: WorldBuildLimits, tuning: WorldBuildTuning
) -> "VoxelField | None":
    """The kept points as a distance field over the cell, or `None` when there is nothing to say.

    The grid spans the declared workspace and sits on the support plane, because that is the volume
    the arm is allowed into and the volume the planner reserved storage for. A point outside it was
    already dropped before this is called.

    Two things happen to the occupancy before the distance transform, and both are the conservative
    reading of what a depth camera can know:

      * Every occupied column is filled down to the plane, for the same reason a box is: a camera
        measures surfaces, so what it returns for a part standing on a bench is the top face, and the
        body underneath is unmeasured rather than empty. `floor_to_plane` turns this off.
      * The surface is grown by `margin_mm`, by subtracting it from the field. Depth noise at an edge
        is one-sided and a planner that grazes is a planner that touches.

    The transform is a Euclidean distance transform in each direction, which is the honest way to get
    a field rather than a shell: without the inside half, a thin surface has no depth and an
    optimiser stepping over it lands on the far side having never seen it.
    """
    if points_mm.shape[0] == 0 or tuning.voxel_field_mm <= 0.0:
        return None
    from scipy import ndimage  # noqa: PLC0415 - kept out of import time, this is the only user

    voxel = float(tuning.voxel_field_mm)
    floor_mm = float(limits.support_plane_top_mm or 0.0)
    low = np.array([limits.x_mm[0], limits.y_mm[0], floor_mm], dtype=np.float64)
    high = np.array([limits.x_mm[1], limits.y_mm[1], max(limits.z_mm[1], floor_mm + voxel)],
                    dtype=np.float64)
    shape = np.maximum(np.round((high - low) / voxel).astype(int), 1)

    index = np.floor((points_mm - low) / voxel).astype(np.int64)
    inside = np.all((index >= 0) & (index < shape), axis=1)
    index = index[inside]
    if index.shape[0] == 0:
        return None

    occupancy = np.zeros(tuple(int(n) for n in shape), dtype=bool)
    occupancy[index[:, 0], index[:, 1], index[:, 2]] = True
    occupied = int(occupancy.sum())

    if tuning.floor_to_plane:
        # Fill each column from its highest occupied voxel down. `maximum.accumulate` over a
        # reversed axis is the whole operation, and it is the voxel form of carrying a box to the
        # bench.
        occupancy = np.flip(np.maximum.accumulate(np.flip(occupancy, axis=2), axis=2), axis=2)

    outside = ndimage.distance_transform_edt(~occupancy) * voxel
    within = ndimage.distance_transform_edt(occupancy) * voxel
    # Positive inside, negative outside, shifted out by the margin. See the note on `VoxelField`:
    # the other sign registers cleanly and stops nothing.
    field = (within - outside + float(tuning.margin_mm)).astype(np.float32)

    centre = (low + high) / 2.0
    return VoxelField(
        field=field.reshape(-1),
        dims_mm=(float(shape[0] * voxel), float(shape[1] * voxel), float(shape[2] * voxel)),
        voxel_size_mm=voxel,
        center_mm=(float(centre[0]), float(centre[1]), float(centre[2])),
        occupied=occupied,
    )


# ---------------------------------------------------------------------------------------------
# The steps, each one small enough to read against a frame
# ---------------------------------------------------------------------------------------------


def _base_points(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_base: np.ndarray,
    keep_mask: np.ndarray,
    voxel_size_mm: float,
    *,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project the kept pixels, thin them, and carry them into BASE with their pixel coordinates.

    The pixels travel with the points because a cluster's label comes from which segmentation its
    pixels fell in, and after a thinning there is no way back from a point to a pixel.

    The back-projection and the voxel thinning are written out here rather than taken from the
    grasping package's point-cloud helpers, which do the same two things. The dependency stack runs
    downward and grasping sits above safety, so a safety module that reached up into it would invert
    the one edge the whole layering rests on. Eight lines of arithmetic is the cheaper price.
    """
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise PerceptionGeometryError(f"intrinsics must be a finite 3x3, got shape {matrix.shape}")
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    if fx == 0.0 or fy == 0.0:
        raise PerceptionGeometryError("intrinsics carry a zero focal length, so nothing projects")
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])

    # Sampled before anything is computed, so a stride costs nothing rather than costing a filter
    # over the full frame. The pixel coordinates are scaled back up, because a label lookup happens
    # in the original image and a mask knows nothing about this.
    step = max(1, int(stride))
    sampled_depth = depth[::step, ::step]
    sampled_keep = keep_mask[::step, ::step]
    valid = sampled_keep & np.isfinite(sampled_depth) & (sampled_depth > 0.0)
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int32)

    z = sampled_depth[rows, cols]
    rows, cols = rows * step, cols * step
    camera = np.column_stack((
        (cols.astype(np.float64) - cx) * z / fx,
        (rows.astype(np.float64) - cy) * z / fy,
        z,
    ))
    pixels = np.column_stack((rows, cols)).astype(np.int32)

    # Voxel thinning: one point per occupied cell, the first one seen. Deterministic, because the
    # same frame has to yield the same world twice.
    #
    # The cell indices are folded into ONE integer per point rather than compared as triples.
    # `np.unique` over an (N, 3) array sorts rows through a structured view, and on a full sensor
    # frame that one call was the whole cost of this function: measured 103 ms for an 848 x 480
    # frame, against 8 ms for the same work keyed by integer. Nothing about the result changes.
    cells = np.floor(camera / float(voxel_size_mm)).astype(np.int64)
    cells -= cells.min(axis=0)
    span = cells.max(axis=0) + 1
    keys = (cells[:, 0] * span[1] + cells[:, 1]) * span[2] + cells[:, 2]
    _, keep = np.unique(keys, return_index=True)
    keep.sort()
    camera, pixels = camera[keep], pixels[keep]

    homogeneous = np.column_stack((camera, np.ones(camera.shape[0], dtype=np.float64)))
    base = (np.asarray(camera_to_base, dtype=np.float64) @ homogeneous.T).T[:, :3]
    return base, pixels


def _cluster(points_mm: np.ndarray, cell_mm: float) -> np.ndarray:
    """Label points by connectivity on a voxel grid, 26-neighbour, iteratively.

    A grid rather than a distance graph because the cost has to be predictable in front of a plan:
    this is one pass to bin the points, then a walk over the occupied cells, and the walk cannot
    visit more cells than there are points. No recursion, so a wall spanning the frame cannot end
    the run with a stack overflow.
    """
    if points_mm.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    cells = np.floor(points_mm / float(cell_mm)).astype(np.int64)
    occupied: dict[tuple[int, int, int], list[int]] = {}
    for index, cell in enumerate(map(tuple, cells)):
        occupied.setdefault(cell, []).append(index)  # type: ignore[arg-type]

    neighbours = [
        (dx, dy, dz)
        for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]
    labels = np.full(points_mm.shape[0], -1, dtype=np.int64)
    current = 0
    for seed in occupied:
        if labels[occupied[seed][0]] != -1:
            continue
        stack = [seed]
        while stack:
            cell = stack.pop()
            members = occupied.get(cell)
            if members is None or labels[members[0]] != -1:
                continue
            labels[members] = current
            cx, cy, cz = cell
            for dx, dy, dz in neighbours:
                nxt = (cx + dx, cy + dy, cz + dz)
                if nxt in occupied and labels[occupied[nxt][0]] == -1:
                    stack.append(nxt)
        current += 1
    return labels


def _oriented_box(
    points_mm: np.ndarray, margin_mm: float, *, floor_mm: float | None = None
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Fit an upright box turned about Z to sit closest around a cluster.

    The yaw comes from the principal direction of the points in the bench plane, which is the axis a
    long part lies along. Z is left alone: an obstacle in a cell stands on something, and a box that
    leans is both harder to reason about and unstable frame to frame.

    A cluster of one point, or a cluster on a line, yields a zero-variance direction. That is not an
    error and is not worth a special case in the caller, so the fit falls back to no rotation, which
    for such a cluster is the same box.
    """
    centred_xy = points_mm[:, :2] - points_mm[:, :2].mean(axis=0)
    yaw = 0.0
    if points_mm.shape[0] >= 2:
        covariance = np.cov(centred_xy, rowvar=False)
        if np.all(np.isfinite(covariance)) and float(np.abs(covariance).max()) > 0.0:
            values, vectors = np.linalg.eigh(covariance)
            principal = vectors[:, int(np.argmax(values))]
            yaw = float(math.atan2(float(principal[1]), float(principal[0])))

    cos_yaw, sin_yaw = math.cos(-yaw), math.sin(-yaw)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)
    local_xy = points_mm[:, :2] @ rotation.T
    lo_xy, hi_xy = local_xy.min(axis=0), local_xy.max(axis=0)
    lo_z, hi_z = float(points_mm[:, 2].min()), float(points_mm[:, 2].max())

    if floor_mm is not None:
        # The camera saw a surface. What is under it was not measured and is not free.
        lo_z = min(lo_z, float(floor_mm))

    local_centre = np.array([(lo_xy[0] + hi_xy[0]) / 2.0, (lo_xy[1] + hi_xy[1]) / 2.0])
    # Back out of the box frame, so the centre is where the box actually sits in BASE.
    back = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    centre_xy = back @ local_centre
    centre = (float(centre_xy[0]), float(centre_xy[1]), (lo_z + hi_z) / 2.0)
    dims = (
        float(hi_xy[0] - lo_xy[0]) + 2.0 * margin_mm,
        float(hi_xy[1] - lo_xy[1]) + 2.0 * margin_mm,
        float(hi_z - lo_z) + 2.0 * margin_mm,
    )
    return centre, dims, yaw


def _label_lookup(
    labelled_masks: Sequence[tuple[str, np.ndarray]], shape: tuple[int, ...], where: str
) -> list[tuple[str, np.ndarray]]:
    """Validate one view named masks once, rather than per cluster."""
    lookup: list[tuple[str, np.ndarray]] = []
    for name, mask in labelled_masks:
        array = np.asarray(mask).astype(bool)
        if array.shape != tuple(shape):
            raise PerceptionGeometryError(
                f"{where}: labelled mask {name!r} is {array.shape} and the depth map is "
                f"{tuple(shape)}"
            )
        lookup.append((str(name), array))
    return lookup


def _dominant_label(
    lookups: Sequence[Sequence[tuple[str, np.ndarray]]],
    pixels_yx: np.ndarray,
    view_of: np.ndarray,
) -> str | None:
    """The segmentation most of a cluster pixels fell in, or `None` when nothing named it.

    Most rather than any, because a cluster straddling two objects belongs to neither cleanly and
    the larger share is the one a reader would recognise. An unnamed cluster is the normal case and
    the point of the whole module: the things nobody detected are exactly the things worth avoiding.

    A pixel only means something in the view it came from, so the count runs per view and the shares
    are added together. Two cameras that both name the same object agree rather than compete, and
    one that names nothing costs the other nothing.
    """
    if pixels_yx.size == 0:
        return None
    totals: dict[str, int] = {}
    for index, lookup in enumerate(lookups):
        if not lookup:
            continue
        here = view_of == index
        if not bool(here.any()):
            continue
        rows, cols = pixels_yx[here][:, 0], pixels_yx[here][:, 1]
        for name, mask in lookup:
            count = int(np.count_nonzero(mask[rows, cols]))
            if count:
                totals[name] = totals.get(name, 0) + count
    if not totals:
        return None
    return max(totals.items(), key=lambda item: item[1])[0]


def _box_name(prefix: str, index: int, label: str | None) -> str:
    """A name a person can find in a refusal, and that no declared fixture can collide with.

    The index is always there because two totes carry the same label and the planner keys its world
    by name: two boxes called the same name are one box, and the other is gone with no message.
    """
    clean = "".join(c if c.isalnum() else "_" for c in (label or "")).strip("_").lower()
    return f"{prefix}{index:02d}" + (f"_{clean}" if clean else "")
