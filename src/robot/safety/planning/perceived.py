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

Two more things it says out loud, because each one is space the planner receives as free:

  * A pixel with no depth. A depth camera answers nothing closer than its minimum range, nothing
    off a surface it cannot read, and nothing in its shadowed band, and the space along that ray
    is unknown rather than empty. It reaches the planner as free space, so it is counted by name
    (``DropReason.NO_DEPTH``) and each view's share of it is reported (``depth_coverage``). Whether
    a world with that many holes may be planned against at all is the live world's call.
  * The bench band. How far above the declared plane a point still counts as the bench is the
    configured clearance plus the placement error the view's camera declares, grown with each
    point's range, and capped (``bench_band_mm``). Anything lower than that band above the bench
    is not an obstacle to the planner.

One kind of unseen space it does not receive as free: what the robot's own body hid from every
camera that could have shown it free, where it lies inside an object the cameras saw, runs on
from a row of one where the self filter took the rest for the robot, or lies between two objects
the robot's shadow cut apart (``_RobotShadow``, ``height_map``). A camera over a bin beside the
base cannot see the rim under the shoulder housing, and that rim is the one the housing meets. It
stands as high as what was seen beside it, and each box says how many of its cells the robot hid
(``hidden_cells``).

And two things it leaves to the geometry somebody declared, because registering them again would
make a hollow solid:

  * Declared geometry. A camera that sees a declared fixture or a declared mesh returns its
    surface, and that surface fitted as a perceived box is solid where the declared shape is
    hollow: a tote's rim becomes a block over its inside. A point within a declared body's own band
    of it, the view's placement error on :data:`DECLARED_SURFACE_MM`, is that body
    (``DropReason.DECLARED``), as a point within the band above the bench is the bench. Not the
    bench band itself, which a sunk slab raises by its sink.
  * The space kept out. A keep-out box takes its points out of the world, and a cluster of
    neighbours around it would still be fitted with one box over the hole. Such a cluster is cut
    around the box before it is fitted, so no perceived box covers what a keep-out left out beyond
    the margin every box is grown by (``keep_out_cuts``).

Units and frames at this boundary: input depth is millimetres in the CAMERA frame, the transform is
the usual 4x4 in millimetres, output is millimetres in BASE. The conversion to the planner's metres
and WXYZ happens in `world.py`, which owns that wire format for both sources.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from src.robot.core.keep_out import KeepOutBox
from src.robot.safety.planning._curobo_protocol import PERCEIVED_PREFIX
from src.robot.safety.planning.height_map import Column, bridge_columns, coarsen, height_map_columns, turn_of
from src.robot.safety.planning.support_surfaces import SupportModel, SupportSolid, detect, detection_step

if TYPE_CHECKING:  # pragma: no cover (typing only: self_envelope imports this module)
    from src.robot.safety.planning.self_envelope import BaseShape

__all__ = [
    "DECLARED_SURFACE_MM",
    "MAX_DECLARED_BAND_MM",
    "DeclaredBody",
    "DepthView",
    "DropReason",
    "LinkCapsule",
    "PerceivedBox",
    "PerceivedWorld",
    "PerceptionGeometryError",
    "ReachSphere",
    "SeenBox",
    "SelfBody",
    "SelfEnvelope",
    "VoxelField",
    "WorldBuildLimits",
    "WorldBuildTuning",
    "build_perceived_boxes",
    "build_voxel_field",
    "has_depth",
    "seen_part_boxes",
    "target_keep_out_box",
    "voxel_grid_extent",
]


class PerceptionGeometryError(ValueError):
    """The frame cannot be turned into geometry, so no world can be built from it."""


#: The most a view's declared placement error may add to the bench band, millimetres.
#:
#: A choice, not a measurement. Everything lower than the band above the bench is dropped as the
#: bench, so a band that followed any declared error without limit would take a real part out of
#: the world in silence. A camera declared to be off by more than this at some range stops growing
#: its band there, and a bench that then comes back above the band comes back as an obstacle that a
#: refusal names, which is the direction it is safe to be wrong in.
MAX_DECLARED_BAND_MM = 25.0

#: How far a surface a camera sees may lie from a declared fixture or mesh and still be that body,
#: millimetres, before the view's own placement error is added.
#:
#: It holds what nothing declares about a body: how well it was measured, the depth error of the
#: camera and the error of its calibration. Five, the figure the shipped ``plane_clearance_mm``
#: gives the bench for the same three errors, and a choice rather than a measurement. It is its own
#: number and not that key because a cell that sinks its slab below the bench raises the key by the
#: sink (robot.yaml): reused as the band round a fixture, a sink of 50 mm took every part within
#: about 60 mm of a declared tote out of the world (review of 2026-09-23). It never exceeds the
#: bench band, so the slab, which is declared too, takes nothing the plane test did not.
DECLARED_SURFACE_MM = 5.0

#: How far past the self filter's padding every point of a box may lie from the robot's own links for the box to be
#: named as maybe the robot itself, seen off where its model stands (``PerceivedBox.robot_gap_mm``), millimetres.
#:
#: A choice, for a refusal to say something an operator can act on. The filter takes a point within the padding
#: (``margin_mm``, 15) of a link's surface for the link, and a camera placed a degree off sees a link 15 mm off at
#: 0.85 m (review of 2026-09-30): past the padding, a box of it is refused where the arm stands, and nothing in the
#: refusal said it might be the arm. Nothing is dropped or kept by it.
_ROBOT_GAP_MM = 15.0

#: How much nearer than a point the robot's own body has to be seen along a camera's ray before the point counts as
#: hidden by it, and how much further the cell's surface has to be seen before the point counts as seen free, as a
#: share of the thinning voxel (``_RobotShadow``): half of it, the resolution the robot's pixels were judged at.
_SHADOW_TOLERANCE_SHARE = 0.5

#: How far past what the cameras saw of an object the robot's shadow is filled, millimetres: a row of it running on
#: into what the robot hides (``height_map.height_map_columns``, ``reach_mm``), and the shadow between two objects
#: (``height_map.bridge_columns``). About the diameter of the UR10's shoulder housing, 150 mm, the widest of its links,
#: whose shadow cut the owner's bin beside the base in two for a camera over the base (review of 2026-09-30), and
#: more than the Hand-E's fingers hide of a wall beside them. A choice; only space the robot hides and no camera saw is
#: filled.
_HIDDEN_REACH_MM = 150.0


class DropReason(StrEnum):
    """Why a point or a cluster did not become a box.

    Every one of these is a hole in what the planner is told, so they are named rather than counted
    into one number: "eleven points fell outside the workspace" and "one obstacle did not fit the
    slot budget" are the same subtraction and completely different news.
    """

    #: Inside a mask the caller asked to leave out, which is normally the object being grasped.
    EXCLUDED = "excluded"
    #: At or below the bench band over the declared support plane: the bench, which is already one
    #: clean box.
    BELOW_PLANE = "below_plane"
    #: Outside the workspace box and, where the robot's reach is given, outside that reach as well,
    #: so no part of the robot can meet it. The workspace box alone bounds the TCP and not the links,
    #: the hand or a wrist camera, which swing past it, so a world built for a robot keeps what its
    #: body can reach.
    OUTSIDE_LIMITS = "outside_limits"
    #: Too few points to be anything but sensor noise.
    TOO_FEW_POINTS = "too_few_points"
    #: On the robot's own body, or on what it is carrying.
    SELF = "self"
    #: Real, in reach, and there was no collision slot left for it.
    NO_SLOT = "no_slot"
    #: Inside a keep-out box a caller handed over: the target of the motion, left out of every
    #: view.
    KEEP_OUT = "keep_out"
    #: A pixel with no depth: zero, not a number, or behind the camera. Counted in pixels of the
    #: full frame, like ``EXCLUDED``. Nothing was measured along that ray, closer than the sensor's
    #: minimum range, off a surface it cannot read, or in its shadowed band, so the space there is
    #: unknown, and the planner receives it as free.
    NO_DEPTH = "no_depth"
    #: Within a declared fixture's or mesh's own band of it (:class:`DeclaredBody`): the camera seeing
    #: geometry the planner already holds as declared, which it would otherwise hold a second time
    #: as a solid box.
    DECLARED = "declared"
    #: Within a support surface's solid up to its band (``support_surfaces``): the mat, the bench or a bin's floor
    #: the parts stand on, which the guard and the planner hold as that solid. Only where the solid holds every pixel
    #: the point stands for.
    SUPPORT = "support"


@dataclass(frozen=True, slots=True)
class WorldBuildLimits:
    """Where in the cell a perceived obstacle is allowed to be, and where the bench is.

    The box is the workspace box the arm's guard reads, and it bounds the TCP. It is not where the
    robot's body can be: the links, the hand and a wrist camera swing past it. So a caller that knows
    the robot's reach hands it to :func:`build_perceived_boxes` as well (:class:`ReachSphere`), and a
    point is kept when either holds it. The box still sets the grid a distance field is cut to,
    because the planner reserves that grid when it starts.
    """

    #: The workspace box in BASE millimetres, as the guard states it.
    x_mm: tuple[float, float]
    y_mm: tuple[float, float]
    z_mm: tuple[float, float]
    #: Top surface of the declared support plane, in BASE millimetres, or `None` for no plane.
    #:
    #: Points within the bench band above it are dropped with the bench. Without a plane every
    #: table point becomes an obstacle, which is why registering a perceived world on a cell that
    #: declared no plane is refused rather than attempted.
    support_plane_top_mm: float | None
    #: How far above the plane a point still counts as the plane, millimetres: the floor of the
    #: bench band.
    #:
    #: It has to hold what nothing declares: how well the bench height was measured, the depth
    #: error of the sensor, and the error of the camera's own calibration. A view whose camera
    #: declares a placement error (:attr:`DepthView.placement_error_mm`) adds that error on top,
    #: grown with each point's range and capped at :data:`MAX_DECLARED_BAND_MM`. Anything lower
    #: than the band above the bench is not an obstacle to the planner.
    #:
    #: It is the slab's alone. A slab sunk below the bench raises it by the sink, so the band round a
    #: declared fixture takes :data:`DECLARED_SURFACE_MM` instead, and the grasp planner's floor over
    #: the bench its own (``grasping.scene.SUPPORT_READ_ERROR_MM``).
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
    #: How far into a box of a named part a finger may come (``PerceivedBox.soft_mm``): the margin and the cell's
    #: finger contact where the fingers may touch parts (``perceived.fingers_touch_parts``), 0 where they may not.
    part_soft_mm: float = 0.0
    #: Whether a finger keeps the guard's distance from a support's reading and its excess rather than from the solid
    #: held over it (``PerceivedBox.finger_top_mm``, ``perceived.fingers_to_the_support_reading``). Off for a library
    #: caller, as before.
    fingers_to_the_reading: bool = False
    #: How much nearer than the guard's distance a finger may come to what a support reads, millimetres: the guard's
    #: ``perceived_min_distance_mm`` less ``perceived.finger_floor_mm`` (the owner's 1 mm, 2026-10-06), 0 for none. A
    #: support's solid lowers its top for the fingers by this as well (``PerceivedBox.finger_top_mm``).
    finger_floor_drop_mm: float = 0.0
    #: Boxes the caller has slots for. Every cluster is a height map of several boxes
    #: (``height_map``), so this is room for a few bins and their parts. Past it boxes merge into
    #: the boxes that hold them first; only what still does not fit, one box per object, is left
    #: out, the nearest surviving, and the rest are reported, never dropped in silence.
    max_boxes: int = 64
    #: Prefix every emitted name carries, so a perceived box can never collide with a declared
    #: fixture and silently replace it during the merge. The planner's sidecar tells the camera's
    #: boxes from the rest of its world by it (``_curobo_protocol.PERCEIVED_PREFIX``).
    name_prefix: str = PERCEIVED_PREFIX
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
    #: either the object or somewhere the arm has no business being. Each box is one column of the
    #: cluster's height map (``height_map``), carried down from the highest point seen over it, so
    #: a container whose rim and floor the camera sees is its walls, carried down, with its inside
    #: free; a floor above the bench band comes back as a low slab. What the camera did not see
    #: beside what it saw, the floor behind a wall it looked over, is free, as unseen space is
    #: everywhere else in the world, unless the robot's own body is what hid it (``_RobotShadow``).
    floor_to_plane: bool = True
    #: Find what the parts stand on and hold it as solid (``support_surfaces.detect``): the large, nearly level
    #: surfaces in the views' own pixels, each a few tilted solids from the declared bench up to its reading plus the
    #: band, and the declared bench as an upright solid up to its band. A point leaves the world only where such a
    #: solid holds every pixel it stands for, and a small cluster standing on one is no noise. Off is the world of
    #: before, byte for byte: the library's default, so a caller that builds a tuning by hand gets what it always got.
    #: A cell turns it on with ``planning_world.perceived.support_surfaces``.
    support_surfaces: bool = False
    #: How far the solids the guard and the planner hold stand over what they take out, millimetres (fix plan F3).
    support_allowance_mm: float = 0.0

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
        allowance = float(self.support_allowance_mm)
        if not (math.isfinite(allowance) and allowance >= 0.0):
            raise PerceptionGeometryError(f"support_allowance_mm is finite and not negative, got {allowance!r}")


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
    #: How far ``camera_to_base`` may be off in translation, as the camera declares it,
    #: millimetres. Added to the bench band of every point of this view.
    placement_error_mm: float = 0.0
    #: How far ``camera_to_base`` may be turned, as the camera declares it, radians. A turn moves a
    #: point by at most its range times the angle, so this times each point's range from the camera
    #: is added to that point's bench band.
    placement_error_rad: float = 0.0
    #: The robot's own body where it stood when this view was taken, or `None` for a view taken
    #: where the robot stands now.
    #:
    #: A frame shows the robot where it stood at the shutter. The body handed to
    #: :func:`build_perceived_boxes` is where it stands now, which takes the robot out of a frame
    #: taken now and out of nothing older: a wrist frame a pick holds from its first look, filtered
    #: by the arm at the approach alone, keeps the links it showed as obstacles where the arm no
    #: longer is, and the planner routes round a robot that left. So a view taken at another pose
    #: carries the body from its own shutter, and its points inside that body are the robot's
    #: (``DropReason.SELF``) as well as its points inside the body now. Only this view's points: where
    #: the robot stood for one frame says nothing about what another frame saw there.
    self_body: "SelfBody | None" = None


@dataclass(frozen=True, slots=True, eq=False)
class DeclaredBody:
    """A piece of the cell somebody declared, as a camera sees it again: a box, or points over a mesh's surface.

    The planner already holds it as it was declared. What a camera returns of it is its surface, and a point within
    its band of it, the view's placement error on :data:`DECLARED_SURFACE_MM`, is taken as it
    (``DropReason.DECLARED``) rather than as an obstacle nobody declared. Build one with :meth:`box` or :meth:`mesh`.

    The distance it answers errs long, never short: inside a box it is zero, and to a mesh it is the distance to the
    nearest point sampled on its surface, which is never nearer than the surface itself. So a point is taken as
    declared only where it truly lies within the band, and a sparse sampling keeps more points as obstacles rather
    than dropping one that is not on the body.
    """

    #: The name the declared world gives it, which the report counts its points under.
    name: str
    #: A box: its own frame in BASE, a rigid 4x4 in millimetres, and its half extents along that frame's axes.
    box_to_base_mm: "np.ndarray | None" = None
    half_extents_mm: "tuple[float, float, float] | None" = None
    #: A mesh: ``(N, 3)`` points on its surface in BASE millimetres, no surface point further than
    #: ``spacing_mm`` from one of them.
    surface_mm: "np.ndarray | None" = None
    #: How far apart the surface points were laid, millimetres. Zero for a box.
    spacing_mm: float = 0.0
    _tree: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise PerceptionGeometryError("a declared body needs the name the declared world gives it")
        is_box = self.box_to_base_mm is not None
        if is_box == (self.surface_mm is not None):
            raise PerceptionGeometryError(f"declared body {self.name!r} is a box or a surface, and exactly one of them")
        if is_box:
            matrix = np.asarray(self.box_to_base_mm, dtype=np.float64)
            half = np.asarray(self.half_extents_mm if self.half_extents_mm is not None else (), dtype=np.float64)
            if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                raise PerceptionGeometryError(f"declared box {self.name!r}: its frame must be a finite 4x4")
            if half.shape != (3,) or not np.all(np.isfinite(half)) or np.any(half < 0.0):
                raise PerceptionGeometryError(f"declared box {self.name!r}: half extents are three finite numbers >= 0")
            object.__setattr__(self, "box_to_base_mm", matrix)
            object.__setattr__(self, "half_extents_mm", tuple(float(v) for v in half))
            return
        points = np.asarray(self.surface_mm, dtype=np.float64).reshape(-1, 3)
        if points.shape[0] == 0 or not np.all(np.isfinite(points)):
            raise PerceptionGeometryError(f"declared mesh {self.name!r}: its surface needs finite points")
        from scipy.spatial import cKDTree  # noqa: PLC0415 (kept out of import time, only a mesh needs it)

        object.__setattr__(self, "surface_mm", points)
        object.__setattr__(self, "_tree", cKDTree(points))

    @classmethod
    def box(cls, name: str, box_to_base_mm: Any, half_extents_mm: Sequence[float]) -> "DeclaredBody":
        """A declared box, placed by its own frame in BASE (millimetres) with its half extents."""
        return cls(name=name, box_to_base_mm=np.asarray(box_to_base_mm, dtype=np.float64),
                   half_extents_mm=tuple(float(v) for v in half_extents_mm))  # type: ignore[arg-type]

    @classmethod
    def mesh(
        cls, name: str, vertices_mm: Any, faces: Any, *, spacing_mm: float = 2.0, max_points: int = 400_000,
    ) -> "DeclaredBody":
        """A declared mesh, its triangles in BASE millimetres, as points laid over its surface.

        Each triangle is cut into a regular grid fine enough that no point of it is further than ``spacing_mm``
        from a grid point, so the answer does not depend on how the file happened to be triangulated. A mesh whose
        surface would take more than ``max_points`` at that spacing is laid coarser, which only keeps more points
        as obstacles. Deterministic: the same mesh lays the same points.
        """
        vertices = np.asarray(vertices_mm, dtype=np.float64).reshape(-1, 3)
        triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        if triangles.shape[0] == 0 or vertices.shape[0] == 0:
            raise PerceptionGeometryError(f"declared mesh {name!r} has no triangle")
        if not np.all(np.isfinite(vertices)) or triangles.min() < 0 or triangles.max() >= vertices.shape[0]:
            raise PerceptionGeometryError(f"declared mesh {name!r}: its vertices or its faces cannot be read")
        corners = vertices[triangles]
        a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
        area = 0.5 * float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
        spacing = max(float(spacing_mm), math.sqrt(2.0 * area / max(int(max_points), 1)))
        longest = np.max(np.stack((np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                                   np.linalg.norm(a - c, axis=1))), axis=0)
        # A grid step of s along each edge leaves no point of the triangle further than s from a grid point. A long
        # thin triangle lays far more points than its area asks for, so the spacing grows until the budget holds.
        cuts = np.maximum(1, np.ceil(longest / spacing)).astype(np.int64)
        for _ in range(32):
            count = int(np.sum((cuts + 1) * (cuts + 2) // 2))
            if count <= int(max_points):
                break
            spacing *= 1.01 * math.sqrt(count / float(max_points))
            cuts = np.maximum(1, np.ceil(longest / spacing)).astype(np.int64)
        laid: list[np.ndarray] = []
        for n in np.unique(cuts):
            rows = np.nonzero(cuts == n)[0]
            i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
            keep = (i + j) <= n
            u, v = (i[keep] / float(n)), (j[keep] / float(n))
            base = a[rows][:, None, :]
            laid.append((base + u[None, :, None] * (b[rows] - a[rows])[:, None, :]
                         + v[None, :, None] * (c[rows] - a[rows])[:, None, :]).reshape(-1, 3))
        points = np.unique(np.round(np.concatenate(laid), 6), axis=0)
        return cls(name=name, surface_mm=points, spacing_mm=spacing)

    def distance_mm(self, points_mm: Any) -> np.ndarray:
        """How far each of ``(N, 3)`` BASE points lies from this body, millimetres, never short of the true distance."""
        points = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
        if points.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        if self.box_to_base_mm is not None:
            matrix = self.box_to_base_mm
            local = (points - matrix[:3, 3]) @ matrix[:3, :3]
            outside = np.maximum(np.abs(local) - np.asarray(self.half_extents_mm), 0.0)
            return np.linalg.norm(outside, axis=1)
        distance, _ = self._tree.query(points)
        return np.asarray(distance, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ReachSphere:
    """Everywhere the robot's own body can be, in any configuration: a sphere about its base.

    Built by :meth:`SelfEnvelope.reach`. A point outside it is somewhere no link, no hand, no wrist
    camera and no carried part can ever meet, which is the one honest reason to leave an obstacle
    out of the planner's world.
    """

    #: Centre in BASE millimetres: the origin of the chain's base frame.
    center_mm: tuple[float, float, float]
    #: Radius, millimetres.
    radius_mm: float

    def __post_init__(self) -> None:
        centre = tuple(float(v) for v in self.center_mm)
        if len(centre) != 3 or not all(math.isfinite(v) for v in centre):
            raise PerceptionGeometryError(f"a reach sphere's centre is three finite numbers, got {self.center_mm!r}")
        if not math.isfinite(float(self.radius_mm)) or float(self.radius_mm) <= 0.0:
            raise PerceptionGeometryError(f"a reach sphere's radius is finite and above 0, got {self.radius_mm!r}")
        object.__setattr__(self, "center_mm", centre)
        object.__setattr__(self, "radius_mm", float(self.radius_mm))

    def contains(self, points_mm: np.ndarray) -> np.ndarray:
        """Which of ``points_mm`` lie inside the sphere. ``(N,)`` boolean."""
        points = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
        return np.linalg.norm(points - np.asarray(self.center_mm), axis=1) <= self.radius_mm


@dataclass(frozen=True, slots=True)
class LinkCapsule:
    """One capsule of the robot's own body, in the frame of the link that carries it.

    A zero-length capsule is a sphere, which is how a hand's sphere map arrives. It stays in its
    link's frame, so it is fitted once and placed wherever the kinematics say that link is now.
    """

    #: Which frame of the chain carries it: 0 is the base, 6 the flange of a six-joint arm.
    frame: int
    #: One end of the segment, in that frame, millimetres.
    start_mm: tuple[float, float, float]
    #: The other end, in that frame, millimetres.
    end_mm: tuple[float, float, float]
    #: Radius before any padding, millimetres.
    radius_mm: float
    #: The link's own surface, in that frame, where the capsule is fitted round a link's mesh: a point
    #: inside the padded capsule is the link only within the padding of this surface
    #: (:meth:`SelfBody.contains`). ``None`` takes the capsule as it is: a hand sphere, a carried part.
    surface: "DeclaredBody | None" = None


@dataclass(frozen=True, slots=True)
class SelfEnvelope:
    """The robot's own body at one instant: where every frame of its chain is, and the capsules they carry."""

    #: One 4x4 per frame, BASE millimetres, index 0 the base.
    frames_mm: tuple[np.ndarray, ...]
    #: The capsules, each on the frame it names.
    capsules: tuple[LinkCapsule, ...]
    #: The robot's base about the BASE axis, where its shape is known (``self_envelope.ROBOT_BASES``): no support
    #: surface is found over it, or its top would put the shoulder in a solid. ``None`` keeps none out.
    base: "BaseShape | None" = None

    def reach(self, *, padding_mm: float = 0.0) -> ReachSphere:
        """The sphere about the base frame's origin that holds this body in every configuration.

        The frames are a Denavit-Hartenberg chain, as every envelope in this repository is: joint
        ``k`` turns frame ``k`` about an axis through the origin of frame ``k - 1``. So a point that
        frame ``k`` carries keeps its distance from that origin whatever the joints do, and so does
        each frame's origin from the one before it. A point of a capsule on frame ``k`` then lies,
        in any configuration, at most the chain of origin distances up to frame ``k - 1``, plus the
        distance of the capsule's farther end from that origin, plus its radius, from the base
        origin. The largest of those over every capsule, grown by ``padding_mm``, is the radius.

        It is a bound, not the reach itself, and it errs outward, which keeps an obstacle rather than
        dropping one. The body is what the envelope carries now: every link, the hand, a wrist camera
        and a part while it is held. Measured 2026-09-23 on the committed UR10 bundle with the Robotiq
        Hand-E on a 20 mm plate (``tests/test_the_camera_world_sees_what_it_can.py``): a radius of
        1793 mm, where the farthest point of the body over 300 random configurations stood 1558 mm
        from the base origin.
        """
        if float(padding_mm) < 0.0:
            raise PerceptionGeometryError(f"padding_mm cannot be negative, got {padding_mm}")
        if not self.frames_mm:
            raise PerceptionGeometryError("a self envelope needs at least its base frame")
        frames = [np.asarray(frame, dtype=np.float64) for frame in self.frames_mm]
        origins = [frame[:3, 3] for frame in frames]
        chain = [0.0]
        for before, after in zip(origins[:-1], origins[1:]):
            chain.append(chain[-1] + float(np.linalg.norm(after - before)))
        radius = chain[-1]
        for capsule in self.capsules:
            frame = int(capsule.frame)
            if not 0 <= frame < len(frames):
                raise PerceptionGeometryError(
                    f"a capsule sits on frame {capsule.frame}, and the chain has {len(frames)} frames"
                )
            pivot = max(frame - 1, 0)
            far_end = max(
                float(np.linalg.norm(frames[frame][:3, :3] @ np.asarray(end, dtype=np.float64)
                                     + origins[frame] - origins[pivot]))
                for end in (capsule.start_mm, capsule.end_mm)
            )
            radius = max(radius, chain[pivot] + far_end + float(capsule.radius_mm))
        centre = origins[0]
        return ReachSphere(
            center_mm=(float(centre[0]), float(centre[1]), float(centre[2])),
            radius_mm=max(radius + float(padding_mm), 1e-9),
        )


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

    A capsule fitted round an arm link holds all of it and a good deal more: the UR10 upper arm's
    reaches 118 mm about its shoulder end at the height of a bin's rim, where the housing hangs 52 mm
    over the base plate, so a rim 6 mm under the housing was the robot. Such a capsule carries the
    link's own surface (``LinkCapsule.surface``), and a point inside it is the link only within the
    padding of that surface (``surfaces``). The padding is then all that is added around the link, as
    everywhere else in the body: what the camera sees past it is an obstacle, and the box grown round
    it by the same margin reaches back to the link (2026-09-30). What the filter took of an object it
    also saw past the padding stands again as high as what was seen of it, a stretch of it or a row
    run on (``height_map``); an object the cameras saw only within the padding of the robot is taken
    for the robot, as it always was, and a camera placed so far off that it sees a link past the
    padding sees it as an obstacle (``PerceivedBox.robot_gap_mm`` says so).
    """

    #: `(K, 2, 3)` segment endpoints in BASE millimetres.
    segments_mm: np.ndarray
    #: `(K,)` radius per segment, millimetres, generous rather than exact.
    radii_mm: np.ndarray
    #: Per capsule, BASE to the frame of the link it holds and that link's surface, or ``None`` for a
    #: capsule taken as it is. Empty takes every capsule as it is.
    surfaces: "tuple[tuple[np.ndarray, DeclaredBody] | None, ...]" = ()
    #: How far from a link's surface a point is still that link, millimetres: the padding.
    surface_padding_mm: float = 0.0
    #: Per capsule, the frame of the chain that carries it (0 the base, 6 the flange of a six-joint arm); empty where the
    #: body was not built from a chain.
    frames: tuple[int, ...] = ()

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
        if self.surfaces and len(self.surfaces) != segments.shape[0]:
            raise PerceptionGeometryError(
                f"surfaces must name one surface or None per capsule, {segments.shape[0]}, got {len(self.surfaces)}"
            )
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

    @classmethod
    def from_frames(
        cls,
        frames_mm: Sequence[np.ndarray],
        capsules: Sequence[LinkCapsule],
        *,
        padding_mm: float = 0.0,
    ) -> "SelfBody":
        """Capsules carried by the frames of a chain, each placed by its frame and grown by ``padding_mm``.

        This is what a driver hands over: the pose of every frame from its kinematics, and capsules
        fitted once in those frames. The padding is the one number added around the body, so a point
        just outside the robot is kept. A capsule that carries its link's surface is taken to that
        surface wherever the padding is at least as wide as the surface's own spacing; below that the
        link's own surface between two of its points could be left in, and the capsule stands as it is.
        """
        if float(padding_mm) < 0.0:
            raise PerceptionGeometryError(f"padding_mm cannot be negative, got {padding_mm}")
        if not capsules:
            raise PerceptionGeometryError("a self body needs at least one capsule")
        segments = np.empty((len(capsules), 2, 3), dtype=np.float64)
        radii = np.empty(len(capsules), dtype=np.float64)
        surfaces: list[tuple[np.ndarray, DeclaredBody] | None] = []
        for index, capsule in enumerate(capsules):
            if not 0 <= int(capsule.frame) < len(frames_mm):
                raise PerceptionGeometryError(
                    f"a capsule sits on frame {capsule.frame}, and the chain has {len(frames_mm)} frames"
                )
            frame = np.asarray(frames_mm[int(capsule.frame)], dtype=np.float64)
            if frame.shape != (4, 4):
                raise PerceptionGeometryError(f"frame {capsule.frame} must be a 4x4, got shape {frame.shape}")
            ends = np.asarray((capsule.start_mm, capsule.end_mm), dtype=np.float64)
            segments[index] = (frame[:3, :3] @ ends.T).T + frame[:3, 3]
            radii[index] = float(capsule.radius_mm) + float(padding_mm)
            surface = capsule.surface
            refined = surface is not None and float(padding_mm) >= float(surface.spacing_mm)
            surfaces.append((np.linalg.inv(frame), surface) if refined and surface is not None else None)
        frames = tuple(int(capsule.frame) for capsule in capsules)
        if not any(entry is not None for entry in surfaces):
            return cls(segments_mm=segments, radii_mm=radii, frames=frames)
        return cls(segments_mm=segments, radii_mm=radii, surfaces=tuple(surfaces),
                   surface_padding_mm=float(padding_mm), frames=frames)

    def contains(self, points_mm: np.ndarray) -> np.ndarray:
        """Which of `points_mm` lie inside the body. `(N,)` boolean.

        Point to segment, capsule by capsule. Six or seven capsules against a thinned cloud is one
        small matrix per capsule, which keeps the cost flat and predictable in front of a plan. A
        point inside a capsule that carries its link's surface is asked that surface too, and is the
        link only within the padding of it: a nearest neighbour among the points laid over the link,
        never nearer than the surface, so no point further than the padding is ever taken out.
        """
        points = np.asarray(points_mm, dtype=np.float64)
        if points.size == 0:
            return np.zeros(points.shape[0], dtype=bool)
        inside = np.zeros(points.shape[0], dtype=bool)
        for index, ((start, end), radius) in enumerate(zip(self.segments_mm, self.radii_mm)):
            axis = end - start
            length_squared = float(axis @ axis)
            if length_squared <= 0.0:
                closest = np.broadcast_to(start, points.shape)
            else:
                travel = np.clip(((points - start) @ axis) / length_squared, 0.0, 1.0)
                closest = start + travel[:, None] * axis
            near = np.linalg.norm(points - closest, axis=1) <= radius
            refined = self.surfaces[index] if self.surfaces else None
            if refined is not None:
                asked = np.nonzero(near & ~inside)[0]
                near[:] = False
                if asked.size:
                    to_link, surface = refined
                    local = points[asked] @ to_link[:3, :3].T + to_link[:3, 3]
                    near[asked] = surface.distance_mm(local) <= self.surface_padding_mm
            inside |= near
        return inside

    def on_itself(self, points_mm: np.ndarray, *, within_mm: float, from_frame: int = 0) -> np.ndarray:
        """Which of ``points_mm`` lie on the robot's own body, without the padding: within ``within_mm`` of a link's
        surface where its capsule carries one, inside the capsule as fitted (its radius less the padding) where not, a
        hand's or a camera's sphere. ``(N,)`` boolean. A column sampled more finely than twice ``within_mm`` meets every
        link it passes through, the surface being closed. ``from_frame`` asks only the capsules carried from that frame
        of the chain on (all of them where the body knows no frames)."""
        points = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
        on = np.zeros(points.shape[0], dtype=bool)
        if points.shape[0] == 0:
            return on
        padding = float(self.surface_padding_mm) if self.surfaces else 0.0
        for index, ((start, end), radius) in enumerate(zip(self.segments_mm, self.radii_mm)):
            if self.frames and int(self.frames[index]) < int(from_frame):
                continue
            axis = end - start
            length_squared = float(axis @ axis)
            travel = (np.clip(((points - start) @ axis) / length_squared, 0.0, 1.0) if length_squared > 0.0
                      else np.zeros(points.shape[0]))
            distance = np.linalg.norm(points - (start + travel[:, None] * axis), axis=1)
            refined = self.surfaces[index] if self.surfaces else None
            if refined is None:
                on |= distance <= max(float(radius) - padding, 0.0) + float(within_mm)
                continue
            asked = np.nonzero(~on & (distance <= float(radius)))[0]
            if asked.size:
                to_link, surface = refined
                local = points[asked] @ to_link[:3, :3].T + to_link[:3, 3]
                on[asked] = surface.distance_mm(local) <= float(within_mm)
        return on

    def surface_gap_mm(self, points_mm: np.ndarray, *, within_mm: float) -> np.ndarray:
        """How far each of ``points_mm`` lies from the nearest link surface this body carries, millimetres, where it lies
        within ``within_mm`` of one; ``inf`` elsewhere, and everywhere for a body that carries no link's surface.

        Never short of the true distance (``DeclaredBody.distance_mm``). It decides nothing; a refusal says with it
        that a box may be the robot itself (``PerceivedBox.robot_gap_mm``).
        """
        points = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
        gap = np.full(points.shape[0], np.inf)
        for index, refined in enumerate(self.surfaces):
            if refined is None or points.shape[0] == 0:
                continue
            start, end = self.segments_mm[index]
            axis = end - start
            length_squared = float(axis @ axis)
            travel = (np.clip(((points - start) @ axis) / length_squared, 0.0, 1.0) if length_squared > 0.0
                      else np.zeros(points.shape[0]))
            # The link lies inside its capsule, so a point within ``within_mm`` of it lies within that of the capsule.
            asked = np.nonzero(np.linalg.norm(points - (start + travel[:, None] * axis), axis=1)
                               <= float(self.radii_mm[index]) + float(within_mm))[0]
            if asked.size:
                to_link, surface = refined
                local = points[asked] @ to_link[:3, :3].T + to_link[:3, 3]
                gap[asked] = np.minimum(gap[asked], surface.distance_mm(local))
        gap[gap > float(within_mm)] = np.inf
        return gap


@dataclass(frozen=True, slots=True)
class PerceivedBox:
    """One obstacle, upright, turned about BASE Z the way its cluster lies: one column of the cluster's height map.

    The rotation is yaw only. A cell's obstacles stand on a floor, so the two axes worth fitting are
    the ones in the plane of the bench; a full three-axis fit on a noisy cluster tumbles between one
    frame and the next, and a planner that is handed a differently tumbled world every 50 ms plans a
    different path for a scene that did not move. The planner and the exact mesh guard hold this
    box as it is, turned (``live_world._guard_boxes``).

    A support surface's solid is the one box that tilts (``kind`` ``support``): it follows the surface's own reading,
    which a camera reads a degree off level, and an upright box round a tilted reading stood up to 14 mm over it at its
    low corner. Its ``rotation`` is the whole turn, and ``yaw_rad`` its heading alone. The bench's solid (``bench``)
    stands upright.
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
    #: How many of its height map's cells stand where the robot's own body hid them from the cameras, filled to the
    #: height seen beside them (``height_map``); 0 for a box of what the cameras saw alone.
    hidden_cells: int = 0
    #: The farthest any of its points lies from the robot's own links, millimetres, where every one of them lies
    #: within the padding and :data:`_ROBOT_GAP_MM` of a link: it may be the robot itself, seen off where its model
    #: stands. ``None`` otherwise, or for a box with no point.
    robot_gap_mm: float | None = None
    #: The box's whole turn, row-major 3x3 whose columns are its axes in BASE, for a box that tilts; ``None`` for one
    #: turned about BASE Z alone by ``yaw_rad``.
    rotation: tuple[float, ...] | None = None
    #: ``seen`` for a box of what the cameras saw; ``support`` for a solid of a surface the parts stand on; ``bench``
    #: for the declared bench's solid.
    kind: str = "seen"
    #: What a refusal against a solid adds: which surface it holds and how high.
    detail: str = ""
    #: How far into this box a finger link may come, millimetres: a box of a part the detector named, no larger than a
    #: part (:data:`PART_SIZED_MM`), holds the fingers to its measured surface (``WorldBuildTuning.part_soft_mm``, the
    #: owner's "Finger dürfen streifen" of 2026-10-05). 0 for every other box, and for every link but the fingers.
    soft_mm: float = 0.0
    #: How far under its top a finger link keeps the guard's distance from, millimetres: a support's solid, its band and
    #: its allowance, so the fingers keep the distance from the reading and its excess
    #: (``WorldBuildTuning.fingers_to_the_reading``, the owner's "wir müssen tiefer gehen" of 2026-10-05). 0 for every
    #: other box, and for every link but the fingers.
    finger_top_mm: float = 0.0

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Its turn as a 3x3 whose columns are its axes in BASE."""
        if self.rotation is not None:
            return np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)
        c, s = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    @property
    def enclosing_half_extents_mm(self) -> tuple[float, float, float]:
        """Half extents of the axis-aligned box that contains this turned one.

        For a consumer whose geometry has no rotation, which is what the capsule guard falls back to;
        the exact mesh guard holds the turned box itself. It is bigger than the turned box, never
        smaller, so a guard fed this can refuse a path the planner allowed but can never allow one
        the planner refused. That is the only direction of disagreement worth having between the
        thing that plans and the thing that decides.
        """
        if self.rotation is not None:
            enclosing = np.abs(self.rotation_matrix) @ (np.asarray(self.dims_mm, dtype=np.float64) / 2.0)
            return (float(enclosing[0]), float(enclosing[1]), float(enclosing[2]))
        cos_yaw, sin_yaw = abs(math.cos(self.yaw_rad)), abs(math.sin(self.yaw_rad))
        half_x, half_y, half_z = (d / 2.0 for d in self.dims_mm)
        return (
            half_x * cos_yaw + half_y * sin_yaw,
            half_x * sin_yaw + half_y * cos_yaw,
            half_z,
        )

    def to_dict(self) -> dict[str, Any]:
        """The wire view. A view of what was computed, never a second computation."""
        out = {
            "name": self.name,
            "center_mm": list(self.center_mm),
            "dims_mm": list(self.dims_mm),
            "yaw_rad": float(self.yaw_rad),
            "points": int(self.points),
            "distance_mm": float(self.distance_mm),
            "label": self.label,
            "hidden_cells": int(self.hidden_cells),
            "robot_gap_mm": None if self.robot_gap_mm is None else float(self.robot_gap_mm),
        }
        if self.kind != "seen" or self.rotation is not None:
            out.update(kind=self.kind, rotation=None if self.rotation is None else [float(v) for v in self.rotation],
                       detail=self.detail)
        return out

    def note(self) -> str:
        """What a refusal against this box should add, as text, ASCII; empty for a box of what was seen and nothing
        more."""
        if self.kind != "seen":
            return self.detail
        said = []
        if self.hidden_cells:
            said.append(f"{self.hidden_cells} of its cells the robot's own body hid from the cameras, filled to the "
                        "height seen beside them")
        if self.robot_gap_mm is not None:
            said.append(f"all {self.points} of its points lie within {self.robot_gap_mm:.0f} mm of the robot's own "
                        "links: it may be the robot itself, seen that far off where its model stands (check the "
                        "hand-eye calibration and the DH table), or something standing that close to the arm")
        return "; ".join(said)


@dataclass(frozen=True, slots=True)
class VoxelField:
    """The cell as a signed distance field over a grid, in the planner's own sign and order.

    The box channel carries as many obstacles as there are collision slots, and the caller then has
    to choose which ones matter. This carries everything the cameras saw at the resolution it was cut
    to, and nothing is left out. It costs one array.

    The sign is negative inside an obstacle and positive in free space, and the values here are
    millimetres; the wire divides by 1000, because the planner reads metres. Measured with
    ``scripts/curobo/probe_live_world.py``: a uniform field over the arm collides at -0.5 and is clear
    at +0.5, and a plane written this way touches the arm exactly where the same plane as a box does.
    Written the other way, every voxel that is not an obstacle reads as inside one, so the whole grid
    blocks and a wall looks seen when the cell is.

    The order is the planner's: x slowest, z fastest, over voxel centres that start half a voxel
    inside the low corner. The same sweep confirms that placement to the millimetre.
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
    #: Points each keep-out box took out, by box name. Empty when no box was handed over.
    keep_out_points: dict[str, int] = field(default_factory=dict)
    #: The share of each view's pixels that held a depth, by view name, of the pixels no mask left
    #: out. The rest is ``DropReason.NO_DEPTH``: space the planner receives as free because nothing
    #: was measured there.
    depth_coverage: dict[str, float] = field(default_factory=dict)
    #: The bench band each view's points were judged against, ``(lowest, highest)`` millimetres above
    #: the declared plane, by view name. A view that declares no placement error has the configured
    #: clearance at both ends; empty for a view with no point.
    bench_band_mm: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: Points each declared body took as its own (``DropReason.DECLARED``), by its name. Empty when no
    #: body was handed over or none was seen.
    declared_points: dict[str, int] = field(default_factory=dict)
    #: How many clusters were cut around each keep-out box so that no box covers it, by the box's name.
    #: Empty when nothing had to be cut.
    keep_out_cuts: dict[str, int] = field(default_factory=dict)
    #: How many boxes the slot budget merged into the boxes that hold them (``height_map.coarsen``), 0
    #: when every column of every height map fitted. Nothing is lost by a merge; the world is only
    #: coarser than the cameras saw it, and more room is ``max_boxes``.
    merged_to_fit: int = 0
    #: How many cells of the boxes stand where the robot's own body hid them from every camera that could have shown
    #: them free, filled to the height seen beside them (``PerceivedBox.hidden_cells``); 0 when the robot hid nothing
    #: of what the cameras saw.
    hidden_cells: int = 0
    #: The surfaces the parts stand on, their solids and the bench's, and how the bare bench reads, where the world
    #: looked for them (``WorldBuildTuning.support_surfaces``); ``None`` where it did not. The solids are among
    #: :attr:`boxes`, last.
    supports: SupportModel | None = None

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
        for name, coverage in self.depth_coverage.items():
            if coverage < 1.0:
                tail.append(
                    f"  {name}: {100.0 * (1.0 - coverage):.0f}% of the image held no depth, and that "
                    "space reaches the planner as free"
                )
        for name, (low, high) in self.bench_band_mm.items():
            band = f"{low:.1f} mm" if abs(high - low) < 0.05 else f"{low:.1f} to {high:.1f} mm"
            tail.append(f"  {name}: anything up to {band} above the declared plane is taken as the bench")
        for name, count in self.declared_points.items():
            if count:
                tail.append(f"  {count} point(s) on the declared {name!r}, left to what was declared")
        for name, count in self.keep_out_cuts.items():
            tail.append(f"  {count} cluster(s) cut around {name!r}, so no box covers it")
        if self.merged_to_fit:
            tail.append(f"  {self.merged_to_fit} box(es) merged into the boxes holding them to fit the slot budget")
        if self.hidden_cells:
            tail.append(f"  {self.hidden_cells} cell(s) the robot hid from the cameras stand as high as what was "
                        "seen beside them")
        for box in self.boxes:
            if box.robot_gap_mm is not None:
                tail.append(f"  {box.name}: every point within {box.robot_gap_mm:.0f} mm of the robot's own links, "
                            "maybe the robot seen off where its model stands")
        if self.supports is not None:
            tail.append(f"  {self.supports.render()}")
        return "\n".join([head, *rows, *tail])

    def to_dict(self) -> dict[str, Any]:
        """The wire view."""
        out = {
            "boxes": [b.to_dict() for b in self.boxes],
            "dropped_points": dict(self.dropped_points),
            "dropped_clusters": dict(self.dropped_clusters),
            "considered_points": int(self.considered_points),
            "dropped_obstacles": self.dropped_obstacle_count,
            "source_timestamp": self.source_timestamp,
            "voxels": None if self.voxels is None else self.voxels.to_dict(),
            "keep_out_points": dict(self.keep_out_points),
            "depth_coverage": {name: float(value) for name, value in self.depth_coverage.items()},
            "bench_band_mm": {name: [float(low), float(high)] for name, (low, high) in self.bench_band_mm.items()},
            "declared_points": dict(self.declared_points),
            "keep_out_cuts": dict(self.keep_out_cuts),
            "merged_to_fit": int(self.merged_to_fit),
            "hidden_cells": int(self.hidden_cells),
        }
        if self.supports is not None:
            out["supports"] = self.supports.to_dict()
        return out


def build_perceived_boxes(
    *,
    views: Sequence[DepthView],
    limits: WorldBuildLimits,
    tuning: WorldBuildTuning | None = None,
    self_body: "SelfBody | None" = None,
    near_point_mm: Sequence[float] | None = None,
    keep_out: Sequence[KeepOutBox] = (),
    reach: "ReachSphere | None" = None,
    declared: Sequence[DeclaredBody] = (),
    cut_around: "Sequence[KeepOutBox] | None" = None,
    base: "BaseShape | None" = None,
    named_points_base_mm: Sequence[tuple[str, np.ndarray]] = (),
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
        A pixel with no depth is counted (``DropReason.NO_DEPTH``) and each view's share of depth
        is reported (``depth_coverage``).
    limits
        Where an obstacle may be, and where the bench is. A point counts as the bench up to the
        configured clearance above the plane, plus the view's declared placement error grown with
        the point's range, capped at :data:`MAX_DECLARED_BAND_MM` (``bench_band_mm``).
    self_body
        The robot own links, and the part it carries, as capsules in BASE millimetres. A camera
        watching a cell sees the arm, and an arm registered as an obstacle is an arm that cannot
        move. Leaving this out is only right where no camera can see the robot at all. It is the
        body now and leaves every view; a view taken at another pose also loses the body from its
        own shutter (:attr:`DepthView.self_body`), from its own points only.
    near_point_mm
        What "nearest" is measured from when the slot budget bites. Normally the goal of the motion
        about to be planned. Defaults to the base origin.
    keep_out
        Boxes in BASE whose points are no obstacle for this motion, normally the target a pick is
        closing on (``target_keep_out_box``). Their points leave every view right after the self
        filter, so they reach neither a box nor the voxel field, and the count each box took is
        reported by name. A cluster whose box would still reach into such a box, the neighbours
        around a part in a pile or the floor of a tote around it, is cut around the box before it is
        fitted: into what lies beyond each side of it, above it and below it, square with BASE, and
        what lies beside a turned box again in its own turn. So a cut box reaches into the keep-out
        by no more than ``margin_mm``, which is the margin a target's keep-out is grown by, and never
        covers the target (``keep_out_cuts``).
    cut_around
        The keep-out boxes a cluster is cut around; ``None`` is every box in ``keep_out``. The live
        world cuts around the targets it holds and not around the space between a jaw's pads at a
        goal, which is not padded: a part wider than that space stays one box over it, and a bare
        motion to a goal inside a part meets the part.
    reach
        Everywhere the robot's body can be (:meth:`SelfEnvelope.reach`). A point is kept when the
        workspace box or this sphere holds it, because the box bounds the TCP and the links, the
        hand and a wrist camera swing past it. ``None`` keeps the box alone.
    declared
        The declared fixtures and meshes (:class:`DeclaredBody`). A point above the bench band that
        lies within :data:`DECLARED_SURFACE_MM` of one of them, grown by the view's placement error
        as the bench band is and never wider than it, is that body, which the planner already holds,
        and is dropped by name (``DropReason.DECLARED``), counted per body (``declared_points``).
    base
        The robot's base about the BASE axis (``SelfEnvelope.base``), where ``tuning.support_surfaces`` looks for
        supports: none is found over it.

    Where ``tuning.support_surfaces`` is on, the surfaces the parts stand on are found in the pixels the self filters
    left (``support_surfaces.detect``) and held as solids, the declared bench among them. Each pixel the world reads
    is then judged on its own: under the bench band, or within a support's solid up to its band, it is the surface's.
    A thinned point leaves the world only when every pixel it stands for is (``DropReason.SUPPORT``, or
    ``BELOW_PLANE`` for the band alone), and one with pixels on both sides stands at its first pixel outside, so every
    pixel either lies in a solid the guard holds or stands for a point that stays. A cluster under ``min_points``
    standing on a solid is no noise. The solids come first in the slot budget, last in the boxes.

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
    keep_out_points: dict[str, int] = {}
    depth_coverage: dict[str, float] = {}
    bench_band_mm: dict[str, tuple[float, float]] = {}
    per_view_points: list[np.ndarray] = []
    per_view_pixels: list[np.ndarray] = []
    per_view_index: list[np.ndarray] = []
    per_view_band: list[np.ndarray] = []
    per_view_read: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    lookups: list[list[tuple[str, np.ndarray]]] = []
    clearance = float(limits.plane_clearance_mm)

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
        error_mm, error_rad = float(view.placement_error_mm), float(view.placement_error_rad)
        if not (math.isfinite(error_mm) and math.isfinite(error_rad)) or error_mm < 0.0 or error_rad < 0.0:
            raise PerceptionGeometryError(
                f"{where}: a placement error is finite and not negative, got {error_mm!r} mm and "
                f"{error_rad!r} rad"
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

        # Holes are counted on the full frame, whatever the stride, so the count and the share
        # describe the image the camera took rather than the pixels this pass happened to read.
        asked = int(np.count_nonzero(keep_mask))
        measured = int(np.count_nonzero(keep_mask & has_depth(depth)))
        if asked - measured:
            dropped_points[DropReason.NO_DEPTH] = (
                dropped_points.get(DropReason.NO_DEPTH, 0) + asked - measured
            )
        depth_coverage[where] = measured / asked if asked else 0.0

        points, pixels, ranges, read = _base_points(
            depth, view.intrinsics, transform, keep_mask, tuning.voxel_size_mm,
            stride=int(tuning.pixel_stride),
        )
        band = clearance + np.minimum(error_mm + error_rad * ranges, MAX_DECLARED_BAND_MM)
        if band.size:
            bench_band_mm[where] = (float(band.min()), float(band.max()))
        lookups.append(_label_lookup(view.labelled_masks, depth.shape, where))
        per_view_points.append(points)
        per_view_pixels.append(pixels)
        per_view_index.append(np.full(points.shape[0], index, dtype=np.int64))
        per_view_band.append(band)
        per_view_read.append(read)

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
    band_of = np.concatenate(per_view_band)
    reference = (
        np.asarray(near_point_mm, dtype=np.float64) if near_point_mm is not None
        else np.zeros(3, dtype=np.float64)
    )
    # The support stage's model and solids, once it ran.
    supports: SupportModel | None = None
    solids: tuple[PerceivedBox, ...] = ()

    def _empty(declared_points: "dict[str, int] | None" = None) -> PerceivedWorld:
        return PerceivedWorld(
            boxes=solids, dropped_points=dropped_points, dropped_clusters={},
            considered_points=0, source_timestamp=oldest, keep_out_points=keep_out_points,
            voxels=(
                build_voxel_field(np.empty((0, 3)), limits=limits, tuning=tuning)
                if tuning.voxel_field_mm > 0.0 else None
            ),
            depth_coverage=depth_coverage, bench_band_mm=bench_band_mm,
            declared_points=dict(declared_points or {}), supports=supports,
        )

    if points_base.shape[0] == 0:
        return _empty()

    # Which of the points read were the robot's own body, by where each stood before any was taken out: what the
    # robot hides from a camera is not free (``_RobotShadow``).
    robot_read = np.zeros(points_base.shape[0], dtype=bool)
    read_as = np.arange(points_base.shape[0])
    read_points = points_base
    if self_body is not None:
        on_self = self_body.contains(points_base)
        dropped_points[DropReason.SELF] = int(np.count_nonzero(on_self))
        robot_read[read_as[on_self]] = True
        points_base, pixels, view_of, band_of, read_as = (
            points_base[~on_self], pixels[~on_self], view_of[~on_self], band_of[~on_self], read_as[~on_self]
        )
        if points_base.shape[0] == 0:
            return _empty()

    # A view taken at another pose shows the robot where it stood then, which the body now does not
    # cover. Its own body takes that out of its own points and nobody else's, counted with the robot.
    # Skipped whole where no view carries one, so a world with no held frame is the one it always was.
    own_bodies = [(index, view.self_body) for index, view in enumerate(views) if view.self_body is not None]
    if own_bodies:
        on_own = np.zeros(points_base.shape[0], dtype=bool)
        for index, own in own_bodies:
            member = np.nonzero(view_of == index)[0]
            if member.size:
                on_own[member[own.contains(points_base[member])]] = True
        dropped_points[DropReason.SELF] = dropped_points.get(DropReason.SELF, 0) + int(np.count_nonzero(on_own))
        robot_read[read_as[on_own]] = True
        points_base, pixels, view_of, band_of, read_as = (
            points_base[~on_own], pixels[~on_own], view_of[~on_own], band_of[~on_own], read_as[~on_own]
        )
        if points_base.shape[0] == 0:
            return _empty()
    # What a keep-out left out stays out: nothing is filled within the margin of one, as no box reaches further in.
    shadow = (
        _RobotShadow.of(views, per_view_read, [points.shape[0] for points in per_view_points], robot_read, tuning,
                        clear_of=_cutters(keep_out), taken_mm=read_points[robot_read])
        if bool(robot_read.any()) else None
    )

    # What the parts stand on, found in what the self filters left, the targets still in; each pixel then judged on
    # its own against the solids and the bench band (fix plan Track S, F4).
    erasure: _Erasure | None = None
    if tuning.support_surfaces:
        alive = np.zeros(read_points.shape[0], dtype=bool)
        alive[read_as] = True
        erasure = _support_stage(
            views, per_view_read, [points.shape[0] for points in per_view_points], per_view_band, alive, read_as,
            points_base, pixels, band_of, limits=limits, tuning=tuning, base=base,
            keep_out=tuple(keep_out if cut_around is None else cut_around),
        )
        supports = erasure.model
        solids = _solid_boxes(erasure.model, reference,
                              fingers_to_the_reading=bool(getattr(tuning, "fingers_to_the_reading", False)),
                              finger_floor_drop_mm=float(getattr(tuning, "finger_floor_drop_mm", 0.0) or 0.0))
        points_base, pixels, band_of = erasure.points_mm, erasure.pixels, erasure.band_mm

    if keep_out:
        kept_out = np.zeros(points_base.shape[0], dtype=bool)
        for box in keep_out:
            inside_box = box.contains(points_base)
            keep_out_points[box.name] = keep_out_points.get(box.name, 0) + int(np.count_nonzero(inside_box))
            kept_out |= inside_box
        dropped_points[DropReason.KEEP_OUT] = int(np.count_nonzero(kept_out))
        points_base, pixels, view_of, band_of = (
            points_base[~kept_out], pixels[~kept_out], view_of[~kept_out], band_of[~kept_out]
        )
        if erasure is not None:
            erasure = erasure.taking(~kept_out)
        if points_base.shape[0] == 0:
            return _empty()

    if erasure is None:
        above = points_base[:, 2] > float(limits.support_plane_top_mm) + band_of
        dropped_points[DropReason.BELOW_PLANE] = int(np.count_nonzero(~above))
    else:
        # A point is the bench's or a support's only where every pixel it stands for is.
        above = ~erasure.erased
        dropped_points[DropReason.BELOW_PLANE] = int(np.count_nonzero(erasure.erased & erasure.banded))
        dropped_points[DropReason.SUPPORT] = int(np.count_nonzero(erasure.erased & ~erasure.banded))

    # A declared body's band is the view's placement error on its own surface margin, never the slab's
    # clearance: a sink of the slab raises that by 50 mm, and a part beside a declared tote is not the
    # tote. After the plane, and never wider than the bench band, so the support slab, which is declared
    # too, takes nothing the bench band did not: a point above the band is further than it from the slab.
    declared_points: dict[str, int] = {}
    if declared:
        on_declared = np.zeros(points_base.shape[0], dtype=bool)
        over_band = np.nonzero(above)[0]
        body_band = band_of[over_band] - clearance + min(clearance, DECLARED_SURFACE_MM)
        for body in declared:
            near = body.distance_mm(points_base[over_band]) <= body_band
            declared_points[body.name] = declared_points.get(body.name, 0) + int(
                np.count_nonzero(near & ~on_declared[over_band]))
            on_declared[over_band[near]] = True
        dropped_points[DropReason.DECLARED] = int(np.count_nonzero(on_declared))
        above &= ~on_declared

    inside = (
        (points_base[:, 0] >= limits.x_mm[0]) & (points_base[:, 0] <= limits.x_mm[1])
        & (points_base[:, 1] >= limits.y_mm[0]) & (points_base[:, 1] <= limits.y_mm[1])
        & (points_base[:, 2] >= limits.z_mm[0]) & (points_base[:, 2] <= limits.z_mm[1])
    )
    if reach is not None:
        inside |= reach.contains(points_base)
    dropped_points[DropReason.OUTSIDE_LIMITS] = int(np.count_nonzero(above & ~inside))

    keep = above & inside
    points, kept_pixels, kept_view = points_base[keep], pixels[keep], view_of[keep]
    kept_free = None if erasure is None else erasure.free_pixels[keep]
    if points.shape[0] == 0:
        return _empty(declared_points)

    labels = _cluster(points, float(tuning.cluster_voxel_mm))
    if reference.shape != (3,):
        raise PerceptionGeometryError(f"near_point_mm must be three numbers, got {near_point_mm!r}")

    dropped_clusters: dict[str, int] = {}
    cutters = _cutters(keep_out if cut_around is None else cut_around)
    keep_out_cuts: dict[str, int] = {}
    floor = float(limits.support_plane_top_mm) if tuning.floor_to_plane else None
    # Every part of every cluster is a height map of columns in the part's own turn (``height_map``): a bin is its
    # walls, a low part beside a tall one keeps its height, and every point stays inside a column by the margin.
    mapped: list[tuple[np.ndarray, float, list[Column]]] = []
    for cluster_id in np.unique(labels):
        member = np.nonzero(labels == cluster_id)[0]
        if member.size < int(tuning.min_points) and not (
            supports is not None and kept_free is not None
            and _stands_on_a_solid(points[member], kept_free[member], supports, tuning)
        ):
            dropped_clusters[DropReason.TOO_FEW_POINTS] = (
                dropped_clusters.get(DropReason.TOO_FEW_POINTS, 0) + 1
            )
            continue
        # A cluster whose box would reach into a keep-out box is cut around it first; any other is one part.
        cut: set[str] = set()
        parts = _cut_around(points[member], cutters, floor, cut)
        for name in cut:
            keep_out_cuts[name] = keep_out_cuts.get(name, 0) + 1
        for part, part_floor, part_yaw in parts:
            chosen_points = member[part]
            mapped.append((chosen_points, part_yaw, height_map_columns(
                points[chosen_points], yaw_rad=part_yaw, floor_mm=part_floor, margin_mm=float(tuning.margin_mm),
                cell_mm=float(tuning.cluster_voxel_mm), step_mm=float(tuning.voxel_size_mm),
                hidden=None if shadow is None else shadow.hides, reach_mm=_HIDDEN_REACH_MM,
                taken_mm=None if shadow is None else shadow.taken_mm,
                robot_on=None if self_body is None else _the_moving_robot(self_body),
            )))
    if shadow is not None and len(mapped) > 1:
        # What the robot hid between two parts, a bin its shadow cut in two, is in neither's height map: boxes of its own.
        bridged = bridge_columns(
            [points[chosen_points] for chosen_points, _, _ in mapped], floor_mm=floor,
            margin_mm=float(tuning.margin_mm), cell_mm=float(tuning.cluster_voxel_mm),
            step_mm=float(tuning.voxel_size_mm), hidden=shadow.hides, reach_mm=_HIDDEN_REACH_MM,
            robot_on=None if self_body is None else _the_moving_robot(self_body),
        )
        if bridged:
            mapped.append((np.zeros(0, dtype=np.int64), 0.0, bridged))
    # Past the slot budget the columns merge into the boxes that hold them, never into less. The solids of what the
    # parts stand on come first and are never merged: the obstacles get what is left.
    budget = max(0, int(tuning.max_boxes) - len(solids))
    # The boxes far from the motion's goal merge first: those the hand passes between stay as they were seen.
    merged_to_fit = coarsen([columns for _, _, columns in mapped], budget,
                            near_mm=None if near_point_mm is None else reference,
                            yaws=[yaw for _, yaw, _ in mapped])
    # How far each point lies from the robot's own links, where it lies close: a box of nothing else may be the robot,
    # seen off where its model stands, which a refusal then says (``PerceivedBox.robot_gap_mm``).
    gaps = (
        self_body.surface_gap_mm(points, within_mm=float(tuning.margin_mm) + _ROBOT_GAP_MM)
        if self_body is not None else np.full(points.shape[0], np.inf)
    )

    # The parts a detector named in BASE (``named_points_base_mm``): a box no mask names is named by its own points.
    named_trees = _named_trees(named_points_base_mm)
    candidates: list[PerceivedBox] = []
    for chosen_points, part_yaw, columns in mapped:
        for column in columns:
            held = chosen_points[column.members]
            centre, dims = column.placed(part_yaw)
            label = (_dominant_label(lookups, kept_pixels[held], kept_view[held])
                     or _named_by_points(named_trees, points[held]))
            candidates.append(
                PerceivedBox(
                    name="",  # named once the survivors are known, so the numbering has no holes
                    center_mm=centre,
                    dims_mm=dims,
                    yaw_rad=part_yaw,
                    points=int(held.size),
                    distance_mm=float(np.linalg.norm(np.asarray(centre) - reference)),
                    label=label,
                    soft_mm=_soft_mm(label, dims, tuning),
                    hidden_cells=int(column.hidden_cells),
                    robot_gap_mm=(float(gaps[held].max()) if held.size and bool(np.isfinite(gaps[held]).all())
                                  else None),
                )
            )

    candidates.sort(key=lambda box: box.distance_mm)
    survivors = candidates[:budget]
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
            hidden_cells=box.hidden_cells,
            robot_gap_mm=box.robot_gap_mm,
            soft_mm=box.soft_mm,
        )
        for index, box in enumerate(survivors)
    ) + solids
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
        keep_out_points=keep_out_points,
        depth_coverage=depth_coverage,
        bench_band_mm=bench_band_mm,
        declared_points=declared_points,
        keep_out_cuts=keep_out_cuts,
        merged_to_fit=merged_to_fit,
        hidden_cells=sum(box.hidden_cells for box in boxes),
        supports=supports,
    )


def has_depth(depth_mm: Any) -> np.ndarray:
    """Which pixels hold a depth the converter uses: finite and in front of the camera. Boolean, same shape.

    One test, used here for every pixel and by the live world to judge a frame, so a frame judged to
    hold too little depth is exactly a frame whose points would have been too few.
    """
    depth = np.asarray(depth_mm, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.isfinite(depth) & (depth > 0.0)


def target_keep_out_box(
    points_base_mm: Any,
    *,
    name: str,
    limits: WorldBuildLimits,
    tuning: WorldBuildTuning | None = None,
) -> KeepOutBox | None:
    """The box a perceived world would fit around a target, as the space that world leaves out for it.

    The target's BASE points pass the filters every obstacle passes: the plane clearance, the limits,
    and the largest cluster on the ``cluster_voxel_mm`` grid, which leaves a flying pixel and a strip
    of bench that bled into a mask behind. The box is then the one ``_oriented_box`` fits, grown by
    ``margin_mm`` and carried down to the plane when ``floor_to_plane`` is on, so the target leaves
    exactly the space it would otherwise fill. ``None`` when no point survives.

    Two filters differ from what an obstacle meets, because the points arrive in BASE with no view.
    The plane clearance is the configured floor alone, without the band a view's declared error
    adds, so a part as flat as that band still leaves its space. The limits are the workspace box
    alone, without the robot's reach, which holds every target the TCP may be sent to.
    """
    tuning = tuning or WorldBuildTuning()
    points = np.asarray(points_base_mm, dtype=np.float64).reshape(-1, 3)
    points = points[np.all(np.isfinite(points), axis=1)]
    plane = limits.support_plane_top_mm
    if plane is not None:
        points = points[points[:, 2] > float(plane) + float(limits.plane_clearance_mm)]
    inside = (
        (points[:, 0] >= limits.x_mm[0]) & (points[:, 0] <= limits.x_mm[1])
        & (points[:, 1] >= limits.y_mm[0]) & (points[:, 1] <= limits.y_mm[1])
        & (points[:, 2] >= limits.z_mm[0]) & (points[:, 2] <= limits.z_mm[1])
    )
    points = points[inside]
    if points.shape[0] == 0:
        return None
    labels = _cluster(points, float(tuning.cluster_voxel_mm))
    ids, counts = np.unique(labels, return_counts=True)
    member = labels == ids[int(np.argmax(counts))]
    centre, dims, yaw = _oriented_box(
        points[member], float(tuning.margin_mm),
        floor_mm=float(plane) if (tuning.floor_to_plane and plane is not None) else None,
    )
    box_to_base = np.eye(4)
    box_to_base[:3, :3] = [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    box_to_base[:3, 3] = centre
    return KeepOutBox.from_matrix(name, box_to_base, tuple(d / 2.0 for d in dims))


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
    """The kept points as a distance field over the cell, or `None` when the cell configured no field.

    A cell with a field and nothing in it gets a field that is free everywhere, never `None`. The
    planner keeps the last field it was sent until another replaces it, so a refresh that sent
    nothing would leave behind whatever was seen before: a part that has since been taken away, or
    the target a pick has just taken out of the world.

    The grid spans the declared workspace and sits on the support plane, because that is the volume
    the TCP is allowed into and the volume the planner reserved storage for. A kept point outside
    it, within the robot's reach but past the workspace box, is not in the field: the boxes carry
    it, and they are registered beside the field.

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
    if tuning.voxel_field_mm <= 0.0:
        return None
    from scipy import ndimage  # noqa: PLC0415 (kept out of import time, this is the only user)

    voxel = float(tuning.voxel_field_mm)
    floor_mm = float(limits.support_plane_top_mm or 0.0)
    low = np.array([limits.x_mm[0], limits.y_mm[0], floor_mm], dtype=np.float64)
    high = np.array([limits.x_mm[1], limits.y_mm[1], max(limits.z_mm[1], floor_mm + voxel)],
                    dtype=np.float64)
    shape = np.maximum(np.round((high - low) / voxel).astype(int), 1)

    index = np.floor((np.asarray(points_mm, dtype=np.float64).reshape(-1, 3) - low) / voxel).astype(np.int64)
    inside = np.all((index >= 0) & (index < shape), axis=1)
    index = index[inside]
    # The centre of the grid the voxels were indexed into, not of the span. The grid is rounded to
    # whole voxels, and the planner puts a voxel centre half a voxel inside the low corner of the
    # pose, so a centre taken from the span would move every obstacle by half the rounding.
    centre = low + shape * voxel / 2.0
    dims = (float(shape[0] * voxel), float(shape[1] * voxel), float(shape[2] * voxel))
    if index.shape[0] == 0:
        # Nothing to measure a distance to: every voxel is free by more than the grid is long.
        free = float(np.linalg.norm(shape * voxel)) - float(tuning.margin_mm)
        return VoxelField(
            field=np.full(int(np.prod(shape)), free, dtype=np.float32), dims_mm=dims, voxel_size_mm=voxel,
            center_mm=(float(centre[0]), float(centre[1]), float(centre[2])), occupied=0,
        )

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
    # Negative inside, positive outside, and the margin moves the surface outward by making every
    # value smaller. See the note on `VoxelField`: the other sign turns the whole grid into an
    # obstacle.
    field = (outside - within - float(tuning.margin_mm)).astype(np.float32)

    return VoxelField(
        field=field.reshape(-1),
        dims_mm=dims,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, "tuple[np.ndarray, np.ndarray, np.ndarray]"]:
    """Back-project the kept pixels, thin them, and carry them into BASE with their pixel coordinates.

    The pixels travel with the points because a cluster's label comes from which segmentation its
    pixels fell in, and after a thinning there is no way back from a point to a pixel. Each point's
    range from the camera, millimetres, travels too, because the bench band grows with it. And every
    pixel read, as its row and column in the strided image with the point that stands for it after the
    thinning, so what the self filter says of that point can be said of every pixel it stands for
    (``_RobotShadow``).

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
    valid = sampled_keep & has_depth(sampled_depth)
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        none = np.empty((0,), dtype=np.int64)
        return (np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float64),
                (none, none, none))

    z = sampled_depth[rows, cols]
    read = (rows, cols)
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
    _, first, voxel = np.unique(keys, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    keep = first[order]
    # Which kept point stands for each pixel read: the kept points are in the order their first pixel was read.
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    camera, pixels = camera[keep], pixels[keep]

    homogeneous = np.column_stack((camera, np.ones(camera.shape[0], dtype=np.float64)))
    base = (np.asarray(camera_to_base, dtype=np.float64) @ homogeneous.T).T[:, :3]
    return base, pixels, np.linalg.norm(camera, axis=1), (read[0], read[1], rank[np.asarray(voxel).reshape(-1)])


def _pixels_in_base(view: DepthView, stride: int, rows: np.ndarray, cols: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Every pixel :func:`_base_points` read of a view, ``rows`` and ``cols`` in the strided frame, as BASE points with
    their ranges from the camera, computed as that function computes the points it keeps."""
    step = max(1, int(stride))
    if rows.size == 0:
        return np.zeros((0, 3)), np.zeros(0)
    z = np.asarray(view.surface_depth_mm, dtype=np.float64)[::step, ::step][rows, cols]
    matrix = np.asarray(view.intrinsics, dtype=np.float64)
    camera = np.column_stack((
        ((cols * step).astype(np.float64) - float(matrix[0, 2])) * z / float(matrix[0, 0]),
        ((rows * step).astype(np.float64) - float(matrix[1, 2])) * z / float(matrix[1, 1]),
        z,
    ))
    transform = np.asarray(view.camera_to_base, dtype=np.float64)
    return camera @ transform[:3, :3].T + transform[:3, 3], np.linalg.norm(camera, axis=1)


def _first_pixel(owner: np.ndarray, which: np.ndarray, count: int) -> np.ndarray:
    """Per point, the first pixel in read order among ``which`` that it stands for; -1 where it stands for none."""
    out = np.full(count, -1, dtype=np.int64)
    index = np.nonzero(which)[0]
    if index.size:
        owners = owner[index]
        order = np.argsort(owners, kind="stable")
        sorted_owners = owners[order]
        starts = np.concatenate(([True], sorted_owners[1:] != sorted_owners[:-1]))
        out[sorted_owners[starts]] = index[order][starts]
    return out


def _first_reads(owner: np.ndarray) -> np.ndarray:
    """Per point of one view, the first pixel read that it stands for: the pixel it was taken from.

    The points are numbered in the order their first pixel was read (:func:`_base_points`), so a point's first pixel is
    where the running highest number read so far rises to it."""
    if owner.size == 0:
        return np.zeros(0, dtype=np.int64)
    rises = np.concatenate(([True], owner[1:] > np.maximum.accumulate(owner)[:-1]))
    return np.nonzero(rises)[0]


@dataclass(frozen=True, slots=True)
class _Erasure:
    """What the support stage decided of each point still in the world, in the world's own order."""

    model: SupportModel
    #: Each point where it stands now: at its first pixel no solid and no band holds, where its first pixel was one.
    points_mm: np.ndarray
    pixels: np.ndarray
    band_mm: np.ndarray
    #: Every pixel it stands for lies within a support's solid up to its band, or under the bench band.
    erased: np.ndarray
    #: Every pixel it stands for lies under the bench band.
    banded: np.ndarray
    #: How many of the pixels it stands for neither holds.
    free_pixels: np.ndarray

    def taking(self, keep: np.ndarray) -> "_Erasure":
        """The same for the points ``keep`` holds."""
        return _Erasure(model=self.model, points_mm=self.points_mm[keep], pixels=self.pixels[keep],
                        band_mm=self.band_mm[keep], erased=self.erased[keep], banded=self.banded[keep],
                        free_pixels=self.free_pixels[keep])


def _support_stage(
    views: Sequence[DepthView],
    read: Sequence["tuple[np.ndarray, np.ndarray, np.ndarray]"],
    counts: Sequence[int],
    view_bands: Sequence[np.ndarray],
    alive: np.ndarray,
    read_as: np.ndarray,
    points_mm: np.ndarray,
    pixels: np.ndarray,
    band_of: np.ndarray,
    *,
    limits: WorldBuildLimits,
    tuning: WorldBuildTuning,
    base: "BaseShape | None",
    keep_out: Sequence[KeepOutBox],
) -> _Erasure:
    """The supports of the pixels the self filters left, and what each point still in the world becomes.

    ``read`` is each view's pixels read and the point each stands for, as :func:`_base_points` gives them, ``counts``
    how many points each view kept, ``view_bands`` each view's points' bench bands, ``alive`` which of all the points
    read the self filters left and ``read_as`` which of those each point in ``points_mm`` is.

    Each pixel is judged on its own: under its bench band, or within a support's solid up to the band, it is the
    surface's. A point all of whose pixels are leaves; one with a pixel neither holds stands at its first such pixel.
    """
    stride = max(1, int(tuning.pixel_stride))
    step = detection_step(stride)
    clearance = float(limits.plane_clearance_mm)
    plane = float(limits.support_plane_top_mm)  # type: ignore[arg-type]
    offsets = np.cumsum([0, *[int(count) for count in counts]])[:-1]
    xyz_parts, owner_parts, band_parts, sampled_parts, view_parts, pixel_parts = [], [], [], [], [], []
    first_parts: list[np.ndarray] = []
    band_of_view: list[float] = []
    read_before = 0
    for index, (view, (rows, cols, owner), band) in enumerate(zip(views, read, view_bands)):
        xyz, ranges = _pixels_in_base(view, stride, rows, cols)
        error = float(view.placement_error_mm) + float(view.placement_error_rad) * ranges
        xyz_parts.append(xyz)
        first_parts.append(_first_reads(np.asarray(owner, dtype=np.int64)) + read_before)
        read_before += xyz.shape[0]
        owner_parts.append(np.asarray(owner, dtype=np.int64) + int(offsets[index]))
        band_parts.append(clearance + np.minimum(error, MAX_DECLARED_BAND_MM))
        sampled_parts.append((rows % step == 0) & (cols % step == 0))
        view_parts.append(np.full(xyz.shape[0], index, dtype=np.int64))
        pixel_parts.append(np.column_stack((rows * stride, cols * stride)).astype(np.int32))
        band_of_view.append(float(np.max(band)) if np.size(band) else clearance)
    xyz = np.concatenate(xyz_parts)
    owner = np.concatenate(owner_parts)
    pixel_band = np.concatenate(band_parts)
    alive_pixel = alive[owner]
    sampled = alive_pixel & np.concatenate(sampled_parts)
    model = detect(xyz[sampled], np.concatenate(view_parts)[sampled], band_of_view, limits=limits, tuning=tuning,
                   base=base, keep_out=keep_out)

    banded = xyz[:, 2] <= plane + pixel_band
    erasable = banded.copy()
    asked = np.nonzero(alive_pixel & ~banded)[0]
    if asked.size and model.support_solids:
        erasable[asked] = model.erased_by_support(xyz[asked])
    total = alive.size
    free = alive_pixel & ~erasable
    free_count = np.bincount(owner[free], minlength=total)
    every_banded = np.bincount(owner[alive_pixel & ~banded], minlength=total) == 0
    first_read = np.concatenate(first_parts) if first_parts else np.zeros(0, dtype=np.int64)

    erased = free_count[read_as] == 0
    # A point that stays though the pixel it was taken from is the surface's stands at its first pixel that is not.
    mixed = ~erased & erasable[first_read[read_as]]
    moved = np.nonzero(mixed)[0]
    points_now, pixels_now, band_now = points_mm, pixels, band_of
    if moved.size:
        asked_points = np.zeros(total, dtype=bool)
        asked_points[read_as[moved]] = True
        to = _first_pixel(owner, free & asked_points[owner], total)[read_as[moved]]
        points_now, pixels_now, band_now = points_mm.copy(), pixels.copy(), band_of.copy()
        points_now[moved] = xyz[to]
        pixels_now[moved] = np.concatenate(pixel_parts)[to]
        band_now[moved] = pixel_band[to]
    return _Erasure(model=model, points_mm=points_now, pixels=pixels_now, band_mm=band_now, erased=erased,
                    banded=every_banded[read_as], free_pixels=free_count[read_as])


def _solid_boxes(model: SupportModel, reference: np.ndarray, *, fingers_to_the_reading: bool = False,
                 finger_floor_drop_mm: float = 0.0) -> tuple[PerceivedBox, ...]:
    """The model's solids as the boxes the planner and the guard hold, a support's tilted, the bench's upright. With
    ``fingers_to_the_reading`` each holds the fingers its band, its allowance and ``finger_floor_drop_mm`` under its
    top (``PerceivedBox.finger_top_mm``): at the guard's distance from that, a finger keeps the finger floor from the
    reading and its excess."""
    where = reference if reference.shape == (3,) else np.zeros(3)
    surfaces = {surface.index: surface for surface in model.surfaces}
    out = []
    for solid in model.solids:
        rotation = solid.rotation
        tilted = solid.kind == "support"
        out.append(PerceivedBox(
            name=solid.name, center_mm=tuple(float(v) for v in solid.centre_mm),  # type: ignore[arg-type]
            dims_mm=tuple(2.0 * float(v) for v in solid.half_extents_mm),  # type: ignore[arg-type]
            yaw_rad=float(math.atan2(rotation[1, 0], rotation[0, 0])), points=0,
            distance_mm=float(np.linalg.norm(solid.centre_mm - where)),
            rotation=tuple(float(v) for v in rotation.reshape(-1)) if tilted else None,
            kind=solid.kind, detail=_solid_detail(solid, surfaces.get(solid.surface)),
            finger_top_mm=(float(solid.band_mm) + float(solid.allowance_mm) + max(0.0, float(finger_floor_drop_mm))
                           if fingers_to_the_reading else 0.0),
        ))
    return tuple(out)


def _solid_detail(solid: SupportSolid, surface: Any) -> str:
    """What a refusal against a solid adds: the surface it holds, and how high it is held."""
    held = float(solid.excess_mm) + float(solid.band_mm) + float(solid.allowance_mm)
    if solid.kind == "bench":
        return (f"the declared bench, held as a solid up to {held:.1f} mm over its plane (its band "
                f"{solid.band_mm:.1f} mm and the {solid.allowance_mm:g} mm allowance)")
    where = ""
    if surface is not None:
        where = (f" (z {surface.z_range_mm[0]:.1f} to {surface.z_range_mm[1]:.1f} mm, tilted "
                 f"{surface.tilt_deg:.1f} deg)")
    name = surface.name if surface is not None else "a surface"
    return (f"a solid of {name}, a surface the camera world found the parts standing on{where}, held up to its "
            f"reading plus {held:.1f} mm (the reading's excess {solid.excess_mm:.1f}, the band {solid.band_mm:.1f} "
            f"and the {solid.allowance_mm:g} mm allowance)")


def _stands_on_a_solid(points: np.ndarray, free_pixels: np.ndarray, model: SupportModel,
                       tuning: WorldBuildTuning) -> bool:
    """Whether a cluster under ``min_points`` is a thing and not noise: its lowest point stands on a solid, a support's
    or the bench's, within a cluster cell and a voxel over its erasure top (fix plan F1). A pin 3 mm across standing on
    the mat shows fewer points than that.

    What stays of the world stands outside every solid's erasure box, or it would have left with the solid, so such a
    cluster stands out of the solid it stands on: never speckle under its top, which leaves with the solid. A solid's
    sides lean with its surface, and a point beside a side is under no top, which is why the test is the box and not the
    top over the point (a slab's side face 2 degrees off level, test_nothing_leaves_the_world_that_a_solid_does_not_hold).
    ``free_pixels``, how many pixels the points stand for that no solid holds, is kept for the record.
    """
    low = points[int(np.argmin(points[:, 2]))]
    # Over a solid, as high as one cluster cell and a voxel: a thin thing seen from straight above shows its top alone,
    # and before the surface was a solid that top joined the surface's own cluster from that high (26 neighbours on
    # cluster_voxel_mm). A 3 mm pin 21 mm tall standing on a slab, its top 14 mm over the slab's erasure top, was
    # dropped at the voxel's 10 mm (test_nothing_leaves_the_world_that_a_solid_does_not_hold, 2026-10-02).
    reach = float(tuning.cluster_voxel_mm) + float(tuning.voxel_size_mm)
    if not model.stands_on_a_solid(low, float(tuning.voxel_size_mm), above_mm=reach):
        return False
    held = model.erased_by_support(points)
    bench = model.bench_solid
    if bench is not None:
        held |= bench.erases(points)
    del free_pixels
    return bool(np.any(~held))


#: The first frame of the chain whose capsules a fill keeps clear of (``height_map._under_the_robot``): wrist 1 on, with
#: the hand, its camera and a carried part on the flange. What stands under the base, the shoulder and the arm keeps
#: its fill: a rim the shoulder housing hides was the reason to fill (review of 2026-09-30).
_MOVING_FROM_FRAME = 4
#: How near a link's surface a sample lies and is on the link, millimetres: more than half the column's sample step.
_ON_THE_LINK_MM = 3.0


def _the_moving_robot(body: SelfBody) -> "Callable[[np.ndarray], np.ndarray]":
    """Which BASE points lie on the robot's own body from the wrist on, as it stands now: the hand down in a bin, which
    a fill must not run through (``height_map._under_the_robot``)."""
    def on(points_mm: np.ndarray) -> np.ndarray:
        return body.on_itself(points_mm, within_mm=_ON_THE_LINK_MM, from_frame=_MOVING_FROM_FRAME)
    return on


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
    points_mm: np.ndarray, margin_mm: float, *, floor_mm: float | None = None, yaw_rad: float | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Fit an upright box turned about Z to sit closest around a cluster, or turned by ``yaw_rad`` where given.

    The yaw is the turn of the smallest rectangle about the points in the bench plane, along its
    longer side (``height_map.turn_of``): a long part is boxed along its length, a square part
    turned on the bench in its own turn, and two walls of a bin seen at a corner along the walls
    rather than between them, where their principal direction runs. Z is left alone: an obstacle in a
    cell stands on something, and a box that leans is both harder to reason about and unstable frame
    to frame.

    A cluster that is about as small square with BASE as turned, a round part or a noisy square one,
    is fitted square with BASE: which way it would turn is decided by noise, and a box turned by noise
    is a different world for a scene that did not move. So is a cluster of one point.
    """
    yaw = turn_of(points_mm[:, :2]) if yaw_rad is None else float(yaw_rad)

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


@dataclass(frozen=True, slots=True)
class _Upright:
    """An upright box turned about Z that encloses a keep-out box, which a cluster is cut around.

    Two of them stand for each keep-out (:func:`_cutters`): the one square with BASE, whose cut pieces are fitted
    square with BASE, and the one turned with the keep-out. A cut square with BASE is the one a guard that cannot turn
    a box reads exactly, because it holds every perceived box as the axis-aligned box that encloses it: the capsule
    fallback does. A keep-out turned against BASE is a long part's, turned along its length, or the space between a
    jaw's pads at a turned grasp.
    """

    name: str
    #: Centre in BASE millimetres.
    centre: np.ndarray
    #: Turn about BASE Z, radians.
    yaw: float
    #: Half extents along the turned X and Y and along BASE Z, millimetres.
    half: np.ndarray

    @classmethod
    def enclosing(cls, box: KeepOutBox, *, yaw: float | None = None) -> "_Upright":
        """The upright box that encloses ``box``, turned by ``yaw``, or by the yaw of its flattest axis."""
        matrix = box.matrix()
        rotation = matrix[:3, :3]
        if yaw is None:
            flattest = int(np.argmax(np.linalg.norm(rotation[:2, :], axis=0)))
            yaw = float(math.atan2(rotation[1, flattest], rotation[0, flattest]))
        signs = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])
        corners = matrix[:3, 3] + (signs * np.asarray(box.half_extents_mm)) @ rotation.T
        local = _turned(corners, yaw)
        low, high = local.min(axis=0), local.max(axis=0)
        mid = (low + high) / 2.0
        back = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        centre = np.array([*(back @ mid[:2]), mid[2]])
        return cls(name=box.name, centre=centre, yaw=float(yaw), half=(high - low) / 2.0)

    def local(self, points_mm: np.ndarray) -> np.ndarray:
        """``(N, 3)`` BASE points in this box's own upright frame, its centre the origin."""
        return _turned(np.asarray(points_mm, dtype=np.float64) - self.centre, self.yaw)


def _cutters(keep_out: Sequence[KeepOutBox]) -> tuple[_Upright, ...]:
    """What a cluster is cut around, in order: per keep-out box, its enclosure square with BASE, then its own turn.

    The second is left out where the box is square with BASE already, a quarter turn included.
    """
    cutters: list[_Upright] = []
    for box in keep_out:
        square = _Upright.enclosing(box, yaw=0.0)
        own = _Upright.enclosing(box)
        cutters.append(square)
        if abs(math.sin(2.0 * own.yaw)) > 1e-9:
            cutters.append(own)
    return tuple(cutters)


def _turned(points_mm: np.ndarray, yaw: float) -> np.ndarray:
    """Points with their X and Y turned by ``-yaw`` about the origin, Z as it was."""
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    out = np.array(points_mm, dtype=np.float64, copy=True).reshape(-1, 3)
    x, y = out[:, 0].copy(), out[:, 1].copy()
    out[:, 0] = cos_yaw * x + sin_yaw * y
    out[:, 1] = -sin_yaw * x + cos_yaw * y
    return out


def _overlap(
    centre_a: np.ndarray, half_a: np.ndarray, yaw_a: float, centre_b: np.ndarray, half_b: np.ndarray, yaw_b: float,
) -> bool:
    """Whether two upright boxes turned about Z share an inside. Boxes that only touch do not.

    Separating axes in the bench plane, the four sides' normals, and the one vertical interval.
    """
    if abs(float(centre_a[2]) - float(centre_b[2])) >= float(half_a[2]) + float(half_b[2]) - 1e-9:
        return False
    sides_a = ((math.cos(yaw_a), math.sin(yaw_a)), (-math.sin(yaw_a), math.cos(yaw_a)))
    sides_b = ((math.cos(yaw_b), math.sin(yaw_b)), (-math.sin(yaw_b), math.cos(yaw_b)))
    apart = (float(centre_b[0]) - float(centre_a[0]), float(centre_b[1]) - float(centre_a[1]))
    for axis in (*sides_a, *sides_b):
        reach_a = sum(abs(axis[0] * u[0] + axis[1] * u[1]) * float(h) for u, h in zip(sides_a, half_a[:2]))
        reach_b = sum(abs(axis[0] * u[0] + axis[1] * u[1]) * float(h) for u, h in zip(sides_b, half_b[:2]))
        if abs(axis[0] * apart[0] + axis[1] * apart[1]) >= reach_a + reach_b - 1e-9:
            return False
    return True


def _cut_around(
    points_mm: np.ndarray, cutters: Sequence[_Upright], floor_mm: float | None, cut: set[str],
) -> list[tuple[np.ndarray, float | None, float]]:
    """The parts of one cluster to fit a box each, as ``(indices, floor, yaw)``; the whole cluster when none is cut.

    A part is cut around a keep-out box when the box it would be fitted with, before its margin and carried down
    to its floor, shares an inside with the keep-out's enclosure: it wraps around the keep-out, bridges over it, or
    reaches into it at a corner, and one box around it would cover space the keep-out left out. It is then cut into
    what lies beyond each of the enclosure's four sides, what lies above it, which stands on the enclosure's top
    rather than on the bench, what lies below it, and what lies beside it within its footprint, and each part is
    fitted in the enclosure's turn. A part beyond one face is fitted wholly beyond it, so its box reaches back into
    the keep-out by no more than its margin. What lies beside a keep-out turned against BASE is cut again around the
    keep-out's own turn. A part is cut around each enclosure at most once, so the cut ends. ``cut`` collects the
    names of the keep-out boxes this cluster was cut around. A part's yaw is the one its box was tested in, so every
    box later fitted to its points in that turn lies within that box and the cut holds for them too.
    """
    work: list[tuple[np.ndarray, float | None, float | None, frozenset[int]]] = [
        (np.arange(points_mm.shape[0]), floor_mm, None, frozenset())
    ]
    parts: list[tuple[np.ndarray, float | None, float]] = []
    while work:
        index, floor, yaw, done = work.pop()
        centre, dims, fitted_yaw = _oriented_box(points_mm[index], 0.0, floor_mm=floor, yaw_rad=yaw)
        for number, cutter in enumerate(cutters):
            if number in done or not _overlap(np.asarray(centre), np.asarray(dims) / 2.0, fitted_yaw,
                                              cutter.centre, cutter.half, cutter.yaw):
                continue
            cut.add(cutter.name)
            local = cutter.local(points_mm[index])
            half = cutter.half
            x, y, z = local[:, 0], local[:, 1], local[:, 2]
            left, right = x < -half[0], x > half[0]
            across = ~(left | right)
            front, back = across & (y < -half[1]), across & (y > half[1])
            over = across & ~(front | back)
            above, below = over & (z > half[2]), over & (z < -half[2])
            beside = over & ~(above | below)
            top = float(cutter.centre[2] + half[2])
            raised = None if floor is None else max(floor, top)
            for piece, piece_floor in ((left, floor), (right, floor), (front, floor), (back, floor),
                                       (above, raised), (below, floor), (beside, floor)):
                if bool(piece.any()):
                    work.append((index[piece], piece_floor, cutter.yaw, done | {number}))
            break
        else:
            parts.append((index, floor, fitted_yaw))
    parts.sort(key=lambda part: int(part[0].min()))
    return parts


@dataclass(frozen=True, slots=True)
class SeenBox:
    """A box the camera world holds for what a camera saw: its centre, its axes and its half extents, BASE millimetres,
    the margin already in them. ``rotation``'s columns are its axes in BASE; it turns about z alone."""

    centre_mm: tuple[float, float, float]
    rotation: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    half_extents_mm: tuple[float, float, float]
    #: How far into it a finger may come: a box of a named neighbour part (``PerceivedBox.soft_mm``), 0 otherwise.
    soft_mm: float = 0.0


def seen_part_boxes(
    points_base_mm: np.ndarray,
    *,
    margin_mm: float,
    cluster_voxel_mm: float,
    voxel_size_mm: float,
    floor_mm: float | None,
    min_points: int = 1,
) -> tuple[SeenBox, ...]:
    """The boxes :func:`build_perceived_boxes` holds for ``points_base_mm``, built as it builds them: clustered on
    ``cluster_voxel_mm``, each cluster a height map of columns in its own turn (:func:`height_map_columns`), grown by
    ``margin_mm`` and carried down to ``floor_mm``; a cluster of fewer than ``min_points`` points is dropped, as the world
    drops it.

    Without what the world adds only from a whole scene and a robot: the cuts around keep-out boxes, the cells the
    robot's body hides, and the merging past the slot budget, which only ever makes a box bigger. So whatever keeps a
    distance from these boxes keeps at least that distance from what the world builds of the same points alone.
    """
    points = np.asarray(points_base_mm, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] == 0:
        return ()
    labels = _cluster(points, float(cluster_voxel_mm))
    out: list[SeenBox] = []
    for cluster_id in np.unique(labels):
        member = points[labels == cluster_id]
        if member.shape[0] < int(min_points):
            continue
        for part, part_floor, part_yaw in _cut_around(member, (), floor_mm, set()):
            cos, sin = math.cos(part_yaw), math.sin(part_yaw)
            rotation = ((cos, -sin, 0.0), (sin, cos, 0.0), (0.0, 0.0, 1.0))
            for column in height_map_columns(member[part], yaw_rad=part_yaw, floor_mm=part_floor,
                                             margin_mm=float(margin_mm), cell_mm=float(cluster_voxel_mm),
                                             step_mm=float(voxel_size_mm)):
                centre, size = column.placed(part_yaw)
                out.append(SeenBox(centre_mm=centre, rotation=rotation,
                                   half_extents_mm=(size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)))
    return tuple(out)


@dataclass(frozen=True, slots=True, eq=False)
class _ShadowView:
    """One camera's frame as :class:`_RobotShadow` reads it: the depth of every pixel read, and which were the robot."""

    #: BASE to CAMERA, 4x4, millimetres.
    to_camera: np.ndarray
    #: The 3x3 the depth was captured with.
    intrinsics: np.ndarray
    #: Every how many pixels the frame was read (``WorldBuildTuning.pixel_stride``).
    stride: int
    #: The depth each pixel read measured, the frame strided, CAMERA millimetres; no depth where none was measured.
    depth: np.ndarray
    #: Which of those pixels the self filter took for the robot's own body, the strided frame's shape.
    robot: np.ndarray

    def judge(self, points_mm: np.ndarray, tolerance_mm: float) -> "tuple[np.ndarray, np.ndarray]":
        """Per point, whether this camera saw as far as it, and whether it saw the robot's own body in front of it.

        Each point is read at the pixel its ray passes nearest, in the strided frame. As far as it: a depth measured
        no nearer than the point less ``tolerance_mm``, so the ray reached where the point is, the bench a point just
        over it stands on included. The robot in front: a pixel the self filter took for the robot, measured nearer
        than the point by more than that. A point outside the frame, behind the camera or at a pixel with no depth is
        neither.
        """
        local = points_mm @ self.to_camera[:3, :3].T + self.to_camera[:3, 3]
        free = np.zeros(points_mm.shape[0], dtype=bool)
        robot = np.zeros(points_mm.shape[0], dtype=bool)
        ahead = np.nonzero(local[:, 2] > 1.0)[0]
        if ahead.size == 0:
            return free, robot
        z = local[ahead, 2]
        matrix = np.asarray(self.intrinsics, dtype=np.float64)
        col = np.rint((matrix[0, 0] * local[ahead, 0] / z + matrix[0, 2]) / self.stride)
        row = np.rint((matrix[1, 1] * local[ahead, 1] / z + matrix[1, 2]) / self.stride)
        rows, cols = self.depth.shape
        inside = (col >= 0.0) & (col < cols) & (row >= 0.0) & (row < rows)
        ahead, z = ahead[inside], z[inside]
        row, col = row[inside].astype(np.int64), col[inside].astype(np.int64)
        measured = self.depth[row, col]
        read = has_depth(measured)
        reached = measured >= z - tolerance_mm
        free[ahead] = read & reached
        robot[ahead] = read & self.robot[row, col] & ~reached
        return free, robot


@dataclass(frozen=True, slots=True, eq=False)
class _RobotShadow:
    """What the robot's own body hid from the cameras, asked point by point (``height_map.height_map_columns``).

    A camera sees the cell past the robot, and the robot hides what stands behind it: a camera over a bin beside the
    UR10's base cannot see the rim under the shoulder housing. A point is hidden where some camera saw the robot's
    own body in front of it and no camera saw as far as it. What the scene hides from itself, the floor behind a
    wall, is not hidden here, nor is what a camera measured nothing at: both are unseen space, which the planner
    receives as free everywhere else (``DropReason.NO_DEPTH``).

    The robot is what the self filter took out, the body now and each held view's own (``DropReason.SELF``), read
    back to every pixel through the point that stood for it after the thinning, so a pixel is the robot to the
    thinning's resolution. What the filter took out beside a link as the link, the padding, counts as the robot too:
    what stands behind it was not seen either.

    Nothing within the margin of a keep-out box is hidden: the target a pick closes on and the space between the jaws
    at a goal are left out of the world on purpose, and the hand that closes on them is what hides them.
    """

    views: tuple[_ShadowView, ...]
    #: :data:`_SHADOW_TOLERANCE_SHARE` of the thinning voxel.
    tolerance_mm: float
    #: The keep-out boxes' enclosures (``_cutters``), and how far about them nothing is hidden: the margin.
    clear_of: tuple["_Upright", ...] = ()
    clearance_mm: float = 0.0
    #: The points the self filter took for the robot, ``(N, 3)`` BASE millimetres: where it may have taken the rest of
    #: a wall for the hand beside it (``height_map.height_map_columns``, ``taken_mm``).
    taken_mm: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    @classmethod
    def of(
        cls,
        views: Sequence[DepthView],
        read: Sequence["tuple[np.ndarray, np.ndarray, np.ndarray]"],
        counts: Sequence[int],
        robot_read: np.ndarray,
        tuning: WorldBuildTuning,
        *,
        clear_of: Sequence["_Upright"] = (),
        taken_mm: "np.ndarray | None" = None,
    ) -> "_RobotShadow":
        """The shadow of ``views``: ``read`` is each view's pixels read and the point each stands for
        (``_base_points``), ``counts`` how many points each view kept, ``robot_read`` which of all those points, the
        views' in order, were the robot, ``clear_of`` the keep-out boxes' enclosures, ``taken_mm`` those points
        themselves."""
        stride = max(1, int(tuning.pixel_stride))
        shadow_views: list[_ShadowView] = []
        offset = 0
        for view, (rows, cols, owner), count in zip(views, read, counts):
            strided = np.asarray(view.surface_depth_mm, dtype=np.float64)[::stride, ::stride]
            robot = np.zeros(strided.shape, dtype=bool)
            robot[rows, cols] = robot_read[offset:offset + int(count)][owner]
            offset += int(count)
            shadow_views.append(_ShadowView(
                to_camera=np.linalg.inv(np.asarray(view.camera_to_base, dtype=np.float64)),
                intrinsics=np.asarray(view.intrinsics, dtype=np.float64), stride=stride, depth=strided, robot=robot,
            ))
        return cls(views=tuple(shadow_views), tolerance_mm=_SHADOW_TOLERANCE_SHARE * float(tuning.voxel_size_mm),
                   clear_of=tuple(clear_of), clearance_mm=float(tuning.margin_mm),
                   taken_mm=np.zeros((0, 3)) if taken_mm is None else np.asarray(taken_mm, dtype=np.float64))

    def hides(self, points_mm: np.ndarray) -> np.ndarray:
        """Which of ``(N, 3)`` BASE points some camera saw the robot in front of and no camera saw as far as, outside
        the margin of every keep-out box."""
        points = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
        free = np.zeros(points.shape[0], dtype=bool)
        robot = np.zeros(points.shape[0], dtype=bool)
        for view in self.views:
            seen_past, behind_robot = view.judge(points, self.tolerance_mm)
            free |= seen_past
            robot |= behind_robot
        for enclosure in self.clear_of:
            local = enclosure.local(points)
            free |= np.all(np.abs(local) <= enclosure.half + self.clearance_mm, axis=1)
        return robot & ~free


#: The largest a box may be and hold a finger to a named part's surface, millimetres: its longer side across and its
#: height. A bin's wall is longer or taller and stays whole whatever the detector named it.
PART_SIZED_MM: tuple[float, float] = (250.0, 150.0)


def _soft_mm(label: "str | None", dims_mm: Sequence[float], tuning: WorldBuildTuning) -> float:
    """How far into a box of ``dims_mm`` a finger may come: ``tuning.part_soft_mm`` where a segmentation named it and
    it is no larger than a part (:data:`PART_SIZED_MM`), else 0."""
    soft = float(getattr(tuning, "part_soft_mm", 0.0) or 0.0)
    if soft <= 0.0 or not label:
        return 0.0
    across, tall = PART_SIZED_MM
    if max(float(dims_mm[0]), float(dims_mm[1])) > across or float(dims_mm[2]) > tall:
        return 0.0
    return soft


#: A box's point lies on a named part's where it lies this near one of that part's points, millimetres: a little over
#: the cloud's own voxel, so a part seen from another pose still meets its own points.
_ON_A_NAMED_PART_MM = 12.0
#: The share of a box's points that has to lie on one named part for the box to be that part's.
_NAMED_SHARE = 0.5


def _named_trees(named: Sequence[tuple[str, np.ndarray]]) -> "list[tuple[str, Any]]":
    """A KD-tree over each named part's BASE points, with its label; parts with no point are left out."""
    if not named:
        return []
    from scipy.spatial import cKDTree  # noqa: PLC0415 (only a world handed named parts pays for it)

    trees: list[tuple[str, Any]] = []
    for label, points in named:
        cloud = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        cloud = cloud[np.all(np.isfinite(cloud), axis=1)]
        if cloud.shape[0]:
            trees.append((str(label), cKDTree(cloud)))
    return trees


def _named_by_points(trees: "Sequence[tuple[str, Any]]", points_mm: np.ndarray) -> str | None:
    """The named part most of ``points_mm`` lie on (:data:`_ON_A_NAMED_PART_MM`, :data:`_NAMED_SHARE`), or ``None``."""
    if not trees or points_mm.shape[0] == 0:
        return None
    best, share = None, 0.0
    for label, tree in trees:
        near, _ = tree.query(points_mm, k=1, distance_upper_bound=_ON_A_NAMED_PART_MM)
        on = float(np.mean(np.isfinite(near)))
        if on > share:
            best, share = label, on
    return best if share >= _NAMED_SHARE else None


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
