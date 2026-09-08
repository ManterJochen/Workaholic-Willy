"""The cell as it is now, in the form a planner can be handed, on demand.

The pieces either side of this module already exist. `perceived.py` turns a depth frame into boxes,
`world.py` turns declared geometry into boxes and owns the wire format, and a planner client will
take a list of them. What was missing is the thing that holds a camera, a transform and the declared
world together and can answer one question at any moment, including moments that have nothing to do
with a pick: what does the cell look like right now.

It is the answer to a specific shape of failure. A planner that is told about the cell once plans
every later move against a photograph, and everything that has moved since is invisible to it: the
tote somebody slid across the bench, the part that fell out, the operator's hand. Nothing about that
is visible in a log, because a plan through an obstacle the planner never received looks exactly
like a plan through empty space.

Two rules run through everything here:

  * A world nobody can vouch for is not a world. When the camera answers nothing, when the frame is
    older than the caller allows, or when the transform is missing, this hands back a snapshot that
    says so, and the caller refuses the motion. Falling back to the last good world is how a cell
    ends up driving through something that left the frame ten seconds ago.
  * The declared world is never lost. Registration replaces rather than extends, so every snapshot
    carries the bench, the fixtures and the payload underneath the perceived boxes, and a perceived
    box can never take a declared box's name.

The depth source is a protocol rather than the perception stack, and the self body arrives as
capsules rather than as a model name. Both keep this module inside the safety layer instead of
reaching up into grasping, which is where the perception frame lives.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, Sequence

import numpy as np

from src.robot.safety._capsule import AxisAlignedBox
from src.robot.safety.planning.perceived import (
    DepthView,
    PerceivedWorld,
    PerceptionGeometryError,
    SelfBody,
    WorldBuildLimits,
    WorldBuildTuning,
    build_perceived_boxes,
)
from src.robot.safety.planning.world import merge_planner_worlds, planner_cuboid

__all__ = [
    "CameraView",
    "DepthSnapshot",
    "LivePlannerWorld",
    "PlannerWorldSnapshot",
    "SurfaceDepthSource",
    "WorldRefresh",
    "WorldVerdict",
    "refresh_planner_world",
]


class WorldVerdict(StrEnum):
    """Whether the world in a snapshot may be planned against, and if not, why not.

    A caller branches on this and on nothing else. The reason strings beside it are for an operator
    reading a refusal, not for control flow.
    """

    #: Built from a frame inside the age the caller allows.
    FRESH = "fresh"
    #: A frame was available and is older than the caller allows.
    STALE = "stale"
    #: The camera answered nothing, or answered something that cannot be used.
    NO_FRAME = "no_frame"
    #: The frame arrived and the geometry could not be built from it.
    UNUSABLE = "unusable"


@dataclass(frozen=True, slots=True)
class DepthSnapshot:
    """One depth reading, with what it takes to place it in the cell.

    `timestamp` is when the shutter opened, not when this object was made. Age is the whole point of
    the field, and a stamp applied after a slow segmentation pass would report a world as fresher
    than it is. A producer that cannot stamp its frames leaves it `None`, which counts as unknown
    age and therefore as too old: an unstamped frame passing as fresh is the failure that costs a
    collision, and a producer being made to stamp is the one that costs an afternoon.
    """

    #: The depth the camera measured, CAMERA frame, millimetres.
    depth_mm: np.ndarray
    #: The 3x3 that depth was captured with.
    intrinsics: np.ndarray
    #: Shutter time on the same clock as `time.time`, or `None` when the producer did not stamp it.
    timestamp: float | None = None


class SurfaceDepthSource(Protocol):
    """Anything that can answer with a depth reading of the cell on demand.

    Deliberately narrower than the perception source the pick loop uses. That one runs a detector and
    a segmenter to produce named objects, which costs hundreds of milliseconds and is exactly what a
    world refresh does not need: a planner wants to know where the geometry is, not what it is
    called. A cell that wants named obstacles hands its labels over separately.
    """

    def grab_surface_depth(self) -> DepthSnapshot | None:
        """The current reading, or `None` when the camera cannot answer."""
        ...


@dataclass(frozen=True, slots=True)
class CameraView:
    """One camera this cell can ask, and the transform that places what it sees.

    A cell with two fixed cameras registers one world, not two. The second camera is there because
    the first cannot see behind the arm or into the far side of a tote, so the points are combined
    before anything is clustered and a part both cameras see half of comes out as one obstacle.
    """

    #: How this camera is named in a refusal an operator has to act on.
    name: str
    #: Where its depth comes from.
    depth_source: "SurfaceDepthSource"
    #: CAMERA to BASE for this camera, 4x4, millimetres.
    camera_to_base: np.ndarray


@dataclass(frozen=True, slots=True)
class PlannerWorldSnapshot:
    """The world as of one moment, ready to register, with its own provenance attached.

    `cuboids` is the whole world, declared and perceived together, in the planner's wire format.
    Sending only part of it deletes the rest, so there is deliberately no way to get the perceived
    boxes out of here in a form that could be registered on their own.
    """

    verdict: WorldVerdict
    cuboids: tuple[dict[str, Any], ...]
    perceived: PerceivedWorld | None
    #: Age of the frame at the moment the snapshot was built, milliseconds, or `None` when there was
    #: no frame or no stamp on it.
    age_ms: float | None
    #: Declared meshes, which travel beside the boxes rather than among them: the sidecar reads them
    #: from a path and the guards cannot read them at all.
    meshes: tuple[dict[str, Any], ...] = ()
    #: What an operator needs to read when the verdict is not `FRESH`.
    reason: str = ""
    #: Declared boxes in this snapshot, so a reader can tell the two halves apart in a report.
    declared_count: int = 0

    @property
    def usable(self) -> bool:
        """Whether a motion may be planned against this."""
        return self.verdict is WorldVerdict.FRESH

    @property
    def perceived_count(self) -> int:
        return len(self.cuboids) - int(self.declared_count)

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
        if not self.usable:
            return f"world not usable ({self.verdict}): {self.reason}"
        age = "unstamped" if self.age_ms is None else f"{self.age_ms:.0f} ms old"
        head = (
            f"world: {self.declared_count} declared + {self.perceived_count} perceived "
            f"obstacle(s), {age}"
        )
        return head if self.perceived is None else head + "\n" + self.perceived.render()

    def to_dict(self) -> dict[str, Any]:
        """The wire view."""
        return {
            "verdict": str(self.verdict),
            "usable": self.usable,
            "age_ms": self.age_ms,
            "reason": self.reason,
            "declared": int(self.declared_count),
            "perceived": self.perceived_count,
            "dropped_obstacles": (
                self.perceived.dropped_obstacle_count if self.perceived is not None else 0
            ),
        }


@dataclass
class LivePlannerWorld:
    """Holds the cameras, their transforms and the declared world, and answers with the cell as it is.

    Built once by whatever composes the cell and handed to the arm, which asks it immediately before
    every plan. The arm never learns where the boxes came from, so the dependency arrow still runs
    downward, and a motion that no pick loop ever sees is covered exactly like one that does.

    Not frozen, because it holds a cache. Everything it hands out is.
    """

    #: The cameras, in the order a report lists them. One is normal; two is better.
    cameras: tuple[CameraView, ...]
    #: The declared world in wire format, from `build_planner_cuboids`.
    declared: tuple[dict[str, Any], ...]
    #: Where a perceived obstacle may be, and where the bench is.
    limits: WorldBuildLimits
    #: The declared meshes, from `build_planner_meshes`. A shape the planner routes around as it is
    #: rather than as a box, which for a container is the difference between reaching into it and
    #: never reaching into it.
    declared_meshes: tuple[dict[str, Any], ...] = ()
    #: How coarse the geometry is and how many boxes there is room for.
    tuning: WorldBuildTuning = field(default_factory=WorldBuildTuning)
    #: A frame older than this is not planned against. An unstamped frame is older than any value.
    max_age_ms: float = 500.0
    #: Radius of the capsules that stand for the arm own links, millimetres.
    self_radius_mm: float = 90.0
    #: Radius of the last capsule, which covers the gripper and anything in it.
    tool_radius_mm: float = 150.0

    _frames: dict[str, DepthSnapshot] = field(default_factory=dict, init=False, repr=False)
    _labels: dict[str, tuple[tuple[str, np.ndarray], ...]] = field(
        default_factory=dict, init=False, repr=False
    )
    _exclude: dict[str, tuple[np.ndarray, ...]] = field(
        default_factory=dict, init=False, repr=False
    )
    _labels_stamped: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.tuning.voxel_field_mm > 0.0:
            # Pay the distance-transform import while a cell is being built, where a tenth of a
            # second costs nothing. Measured against the real planner: the first field a cell ever
            # built took 126 ms and every one after it 6 ms, and the whole difference was this
            # import landing on the first motion instead of here.
            import scipy.ndimage  # noqa: F401, PLC0415 - imported for its cost, not for its names

        if not self.cameras:
            raise PerceptionGeometryError(
                "a live planner world needs at least one camera, and a cell that has none should "
                "not be building one"
            )
        names = [camera.name for camera in self.cameras]
        if len(set(names)) != len(names):
            raise PerceptionGeometryError(
                f"two cameras share a name in {names}: a per-camera cache and a per-camera refusal "
                "both key on it"
            )

    # -----------------------------------------------------------------------------------------
    # What the pick loop can add, and nothing else has to
    # -----------------------------------------------------------------------------------------

    def offer_segmentation(
        self,
        *,
        camera: str | None = None,
        labelled_masks: Sequence[tuple[str, np.ndarray]] = (),
        exclude_masks: Sequence[np.ndarray] = (),
        timestamp: float | None = None,
    ) -> None:
        """Hand over the masks from a perception frame, for as long as they stay fresh.

        Two things come from here and neither is required. Names, so a refusal can say which object
        refused rather than printing a number. And the object being grasped, which has to be left out
        of the world: it is the one obstacle the arm is deliberately driving into, and a goal inside
        an obstacle has no plan by construction.

        Masks belong to the camera that produced them, because a pixel means nothing in another
        camera image. `camera` names it, and defaults to the first, which is the only one a
        single-camera cell has.

        A cell that never calls this still gets a world. It gets unnamed boxes, and during a pick it
        gets one box where the target is, which is why the pick loop calls it.
        """
        name = self.cameras[0].name if camera is None else str(camera)
        if name not in {view.name for view in self.cameras}:
            raise PerceptionGeometryError(
                f"no camera named {name!r} in this cell: "
                f"{[view.name for view in self.cameras]}"
            )
        self._labels[name] = tuple(
            (str(label), np.asarray(mask).astype(bool)) for label, mask in labelled_masks
        )
        self._exclude[name] = tuple(np.asarray(mask).astype(bool) for mask in exclude_masks)
        self._labels_stamped = time.time() if timestamp is None else float(timestamp)

    def drop_cached_frames(self) -> None:
        """Forget the cached readings, so the next question reaches the cameras.

        The cache exists so two plans inside one perception cycle share one reading. A caller that
        knows the scene changed, which in practice means a probe or a test rather than a cell, says
        so here rather than by waiting out the age limit.
        """
        self._frames.clear()

    def forget_segmentation(self) -> None:
        """Drop the masks. Called when a pick ends, so the next motion excludes nothing."""
        self._labels.clear()
        self._exclude.clear()
        self._labels_stamped = None

    # -----------------------------------------------------------------------------------------
    # The one question
    # -----------------------------------------------------------------------------------------

    def world_for(
        self,
        *,
        link_origins_mm: Sequence[Sequence[float]] | None = None,
        near_point_mm: Sequence[float] | None = None,
        now: float | None = None,
    ) -> PlannerWorldSnapshot:
        """The cell as it is, declared and perceived together, ready to register.

        Parameters
        ----------
        link_origins_mm
            The arm own link origins in BASE millimetres, as a forward kinematic returns them. They
            become the capsules that take the robot back out of what the cameras saw. `None` means
            the caller cannot describe its own links, and then the arm itself would be registered as
            an obstacle, so no perceived world is built at all.
        near_point_mm
            Where the motion is going. It decides which obstacles survive the slot budget, because
            the ones near the path are the ones that matter.
        now
            The clock, injectable so a test can age a frame without sleeping.
        """
        clock = time.time() if now is None else float(now)
        declared = tuple(dict(box) for box in self.declared)
        meshes = tuple(dict(mesh) for mesh in self.declared_meshes)
        frames, verdict, reason = self._frames_for(clock)
        # The clock is read again after the cameras answered, because a grab takes time and the
        # first reading is older than the frames it went out to fetch. Measured before this line
        # existed: a fresh frame reported as two milliseconds from the future.
        if now is None:
            clock = time.time()

        if verdict is not WorldVerdict.FRESH:
            # A refusal still reports an age where one is knowable, because the first thing an
            # operator asks a stale-world refusal is how stale.
            known = [
                age for age in (self._age_ms(frame, clock) for frame in frames.values())
                if age is not None
            ]
            return PlannerWorldSnapshot(
                verdict=verdict, cuboids=declared, perceived=None,
                age_ms=max(known) if known else None, reason=reason,
                meshes=meshes, declared_count=len(declared),
            )

        oldest = self._oldest_age_ms(frames, clock)
        if link_origins_mm is None:
            return PlannerWorldSnapshot(
                verdict=WorldVerdict.UNUSABLE, cuboids=declared, perceived=None, age_ms=oldest,
                reason=(
                    "the arm cannot describe where its own links are, so every point of the robot "
                    "in a camera view would be registered as an obstacle and the arm would be "
                    "standing inside one"
                ),
                meshes=meshes, declared_count=len(declared),
            )

        fresh_masks = self._labels_fresh(clock)
        views = [
            DepthView(
                surface_depth_mm=frames[camera.name].depth_mm,
                intrinsics=frames[camera.name].intrinsics,
                camera_to_base=camera.camera_to_base,
                exclude_masks=self._exclude.get(camera.name, ()) if fresh_masks else (),
                labelled_masks=self._labels.get(camera.name, ()) if fresh_masks else (),
                name=camera.name,
                timestamp=frames[camera.name].timestamp,
            )
            for camera in self.cameras
        ]

        try:
            perceived = build_perceived_boxes(
                views=views,
                limits=self.limits,
                tuning=self.tuning,
                self_body=SelfBody.from_polyline(
                    link_origins_mm,
                    radius_mm=self.self_radius_mm,
                    tool_radius_mm=self.tool_radius_mm,
                ),
                near_point_mm=near_point_mm,
            )
        except PerceptionGeometryError as exc:
            return PlannerWorldSnapshot(
                verdict=WorldVerdict.UNUSABLE, cuboids=declared, perceived=None, age_ms=oldest,
                reason=str(exc), meshes=meshes, declared_count=len(declared),
            )

        perceived_boxes = [
            planner_cuboid(box.name, box.center_mm, box.dims_mm, yaw_rad=box.yaw_rad)
            for box in perceived.boxes
        ]
        return PlannerWorldSnapshot(
            verdict=WorldVerdict.FRESH,
            cuboids=tuple(merge_planner_worlds(declared, perceived_boxes)),
            perceived=perceived,
            age_ms=oldest,
            meshes=meshes, declared_count=len(declared),
        )

    # -----------------------------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------------------------

    def _frames_for(
        self, clock: float
    ) -> tuple[dict[str, DepthSnapshot], WorldVerdict, str]:
        """A reading from every camera: the cached one while it is young enough, else a new grab.

        Every camera has to answer. A cell with two cameras has two because one of them cannot see
        the whole cell, so carrying on without one means planning against a world with a hole in it
        exactly where nobody is looking. The verdict names the camera that failed, because that is
        the thing an operator has to go and fix.
        """
        frames: dict[str, DepthSnapshot] = {}
        for camera in self.cameras:
            cached = self._frames.get(camera.name)
            cached_age = None if cached is None else self._age_ms(cached, clock)
            if cached is not None and cached_age is not None and cached_age <= self.max_age_ms:
                frames[camera.name] = cached
                continue

            grabbed = camera.depth_source.grab_surface_depth()
            if grabbed is None:
                return (
                    frames,
                    WorldVerdict.NO_FRAME,
                    f"camera {camera.name!r} returned no depth, so nothing can vouch for what is "
                    "in the part of the cell it watches",
                )
            self._frames[camera.name] = grabbed
            frames[camera.name] = grabbed

            age = self._age_ms(grabbed, clock)
            if age is None:
                return (
                    frames,
                    WorldVerdict.STALE,
                    f"the depth frame from camera {camera.name!r} carries no capture time, so its "
                    "age is unknown and it cannot be shown to be inside the limit this cell allows",
                )
            if age > self.max_age_ms:
                return (
                    frames,
                    WorldVerdict.STALE,
                    f"the depth frame from camera {camera.name!r} is {age:.0f} ms old and this "
                    f"cell allows {self.max_age_ms:.0f} ms",
                )
        return frames, WorldVerdict.FRESH, ""

    @staticmethod
    def _age_ms(frame: DepthSnapshot, clock: float) -> float | None:
        if frame.timestamp is None:
            return None
        return (clock - float(frame.timestamp)) * 1000.0

    def _oldest_age_ms(self, frames: dict[str, DepthSnapshot], clock: float) -> float | None:
        """A fused world is exactly as current as its stalest half."""
        ages = [self._age_ms(frame, clock) for frame in frames.values()]
        if not ages or any(age is None for age in ages):
            return None
        return max(age for age in ages if age is not None)

    def _labels_fresh(self, clock: float) -> bool:
        """Masks age exactly like the frame they came from.

        An excluded target from half a second ago is a hole in the world where the object no longer
        is, and the object may well be somewhere else in it now.
        """
        if self._labels_stamped is None:
            return False
        return (clock - self._labels_stamped) * 1000.0 <= self.max_age_ms


# ---------------------------------------------------------------------------------------------
# The one call every planning driver makes, immediately before it plans
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WorldRefresh:
    """What one registration did, in numbers a report can carry and an operator can read.

    It exists because none of this was measurable before. The plan call has been timed and logged
    since it was written; the world call has not, so nobody could say what a registration costs, and
    every argument about how often to refresh had nothing to stand on. Now every refresh states its
    own price.
    """

    verdict: WorldVerdict
    #: Boxes handed to the planner.
    sent: int
    #: Boxes the planner confirmed. Fewer than sent means part of the cell is not in its world.
    registered: int
    #: Building the geometry from the frame.
    build_ms: float
    #: The round trip that registers it.
    register_ms: float
    #: Age of the frame the world was built from, or `None` when there was none or it was unstamped.
    age_ms: float | None
    #: Obstacles that were real, in reach, and had no collision slot left.
    dropped_obstacles: int
    #: Empty while the world is usable.
    reason: str = ""
    #: The perceived obstacles, axis-aligned, for the guards. Empty when the world was not usable,
    #: which is the state a guard has to be put back into rather than left holding stale boxes.
    guard_boxes: tuple["AxisAlignedBox", ...] = ()
    #: Values in the distance field the planner accepted, or `None` when no field was sent. The
    #: boxes go to the guard and the field goes to the planner, and both come from one cloud.
    voxels_registered: int | None = None

    @property
    def ok(self) -> bool:
        """Whether the caller may plan. Anything else is a refusal, never a warning."""
        return self.verdict is WorldVerdict.FRESH and not self.reason

    @property
    def total_ms(self) -> float:
        return self.build_ms + self.register_ms

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
        if not self.ok:
            return f"planner world NOT refreshed ({self.verdict}): {self.reason}"
        age = "unstamped" if self.age_ms is None else f"{self.age_ms:.0f} ms old"
        line = (
            f"planner world refreshed: {self.registered} box(es), frame {age}, "
            f"{self.build_ms:.1f} ms to build, {self.register_ms:.1f} ms to register"
        )
        if self.voxels_registered:
            line += f", live scene {self.voxels_registered} voxel(s)"
        if self.dropped_obstacles:
            line += (
                f"; {self.dropped_obstacles} obstacle(s) had no slot and are NOT in the world"
            )
        return line

    def to_dict(self) -> dict[str, Any]:
        """The wire view."""
        return {
            "verdict": str(self.verdict),
            "ok": self.ok,
            "sent": int(self.sent),
            "registered": int(self.registered),
            "build_ms": round(float(self.build_ms), 3),
            "register_ms": round(float(self.register_ms), 3),
            "total_ms": round(float(self.total_ms), 3),
            "age_ms": self.age_ms,
            "dropped_obstacles": int(self.dropped_obstacles),
            "voxels": self.voxels_registered,
            "reason": self.reason,
        }


def refresh_planner_world(
    *,
    source: LivePlannerWorld,
    client: Any,
    link_origins_mm: Sequence[Sequence[float]] | None,
    near_point_mm: Sequence[float] | None = None,
    require_registration: bool = True,
    now: float | None = None,
) -> WorldRefresh:
    """Build the current world and hand it to the planner, or say why the caller must not plan.

    This is the whole feature in one call, and it is deliberately the only way in. A driver that
    planned against a world it did not refresh would be the exact defect this exists to remove, and
    a driver that refreshed in three places would have three chances to get the merge wrong.

    `client` is anything with `set_world(list) -> int`, which is what both planner clients in this
    repository are. The count it returns is the count REGISTERED rather than the count sent, and the
    difference matters more than any other number here: an obstacle the planner never received is one
    it will route straight through, and every layer above reads that as a successful plan.

    Refuses rather than degrades, which is the safety layer's rule everywhere else. Nothing here
    falls back to the previous world: the previous world is a photograph of a cell that has since
    had time to change, and planning against it is the failure, not the recovery.
    """
    started = time.perf_counter()
    snapshot = source.world_for(
        link_origins_mm=link_origins_mm, near_point_mm=near_point_mm, now=now
    )
    build_ms = (time.perf_counter() - started) * 1000.0
    dropped = snapshot.perceived.dropped_obstacle_count if snapshot.perceived is not None else 0

    if not snapshot.usable:
        return WorldRefresh(
            verdict=snapshot.verdict, sent=0, registered=0, build_ms=build_ms, register_ms=0.0,
            age_ms=snapshot.age_ms, dropped_obstacles=dropped, reason=snapshot.reason,
        )

    started = time.perf_counter()
    boxes = [dict(box) for box in snapshot.cuboids]
    # Named only when there is something to name: a planner client from before meshes existed takes
    # one argument, and a cell that declares no mesh should not need a newer one.
    registered = int(
        client.set_world(boxes, [dict(m) for m in snapshot.meshes])
        if snapshot.meshes else client.set_world(boxes)
    )
    wanted, voxels, why = _send_voxel_field(client, snapshot)
    register_ms = (time.perf_counter() - started) * 1000.0

    reason = ""
    if wanted and voxels is None:
        # A cell that asked for a live scene and did not get one is planning against less than it
        # believes, and believing it is the whole danger. Refuse, and say which of the two happened.
        reason = (
            f"the live scene did not reach the planner ({why}), so it would be planning against "
            "the declared world alone while the cameras can see more than that"
        )
    expected = len(snapshot.cuboids) + len(snapshot.meshes)
    if registered != expected and require_registration:
        reason = (
            f"the planner confirmed {registered} of {expected} obstacle(s), so it is "
            "planning against a world that is missing part of this cell. Set "
            "safety.planning_world.require_registration false to plan anyway, deliberately."
        )
    return WorldRefresh(
        verdict=snapshot.verdict, sent=expected, registered=registered,
        build_ms=build_ms, register_ms=register_ms, age_ms=snapshot.age_ms,
        dropped_obstacles=dropped, reason=reason,
        guard_boxes=() if reason else _guard_boxes(snapshot.perceived),
        voxels_registered=voxels,
    )


def _guard_boxes(perceived: PerceivedWorld | None) -> tuple[AxisAlignedBox, ...]:
    """The perceived obstacles in the shape the capsule guards hold, which has no rotation.

    Each box is the axis-aligned one that encloses the turned box the planner got, so the guard is
    never more permissive than the planner and can be stricter. Being stricter is a refusal an
    operator can see and argue with; being more permissive is a collision nobody predicted.
    """
    if perceived is None:
        return ()
    return tuple(
        AxisAlignedBox(
            center_mm=np.asarray(box.center_mm, dtype=np.float64),
            half_extents_mm=np.asarray(box.enclosing_half_extents_mm, dtype=np.float64),
            name=box.name,
        )
        for box in perceived.boxes
    )


def _send_voxel_field(
    client: Any, snapshot: PlannerWorldSnapshot
) -> "tuple[bool, int | None, str]":
    """Hand the planner the distance field, through a file rather than through the pipe.

    Returns `(wanted, registered, why)`. `wanted` says whether this cell asked for a live scene at
    all, which is the difference between a cell that never configured one and a cell whose planner
    would not take it. Both come back with `registered` as `None` and only one of them is a problem.

    A 30 mm grid over a two metre cell is 179,560 values. That is not a message, and this protocol is
    one JSON object per line, so the field is written where the sidecar can read it and the request
    carries the path. Writing costs about a millisecond and the sidecar reads it in about two.

    The same file is reused every refresh. A new one per motion would fill a temp directory at the
    rate the arm moves, and there is never more than one live scene.

    """
    if snapshot.perceived is None or snapshot.perceived.voxels is None:
        return False, None, ""
    setter = getattr(client, "set_voxels", None)
    if not callable(setter):
        return (
            True,
            None,
            "this planner has no live-scene channel at all, which on the real sidecar means it was "
            "started without a voxel reservation",
        )

    field = snapshot.perceived.voxels
    path = os.path.join(tempfile.gettempdir(), f"willy_live_scene_{os.getpid()}.npy")
    np.save(path, field.field.astype(np.float16))
    registered = setter(
        path,
        dims_m=[d / 1000.0 for d in field.dims_mm],
        voxel_size_m=field.voxel_size_mm / 1000.0,
        pose=[c / 1000.0 for c in field.center_mm] + [1.0, 0.0, 0.0, 0.0],
    )
    if registered is None:
        return True, None, "the planner refused it"
    return True, int(registered), ""
