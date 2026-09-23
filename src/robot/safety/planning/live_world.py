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
    box can never take a declared box's name. Nor is it registered twice: what the cameras see of a
    declared fixture or mesh is left to it (``perceived.DeclaredBody``), so a declared tote keeps its
    hollow instead of coming back as a solid perceived block over it.

A frame that answers is not yet a frame that saw. A pixel with no depth is space the planner would
receive as free, so a frame that holds no depth at all is refused as blind, a goal whose surroundings
are mostly holes in every camera that looks at it is refused as unseen, and what stays below those
limits is said on the stamp the motion carries (``UnseenSpace``) rather than vouched for. How much of
the whole frame has to hold a depth depends on where the camera stands. A fixed camera is never carried
toward what it sees, so a fixed frame that is mostly holes is a camera fault and is refused as blind. A
camera on the wrist is carried inside its own minimum range by every grasp, so its frame is judged
where the motion goes, and by the whole frame only when no goal is named.

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

from src.contracts import UNSET, Maybe, chosen
from src.robot.core.camera_world import CameraWorldStamp, UnseenSpace
from src.robot.core.errors import CameraWorldUnavailable
from src.robot.core.keep_out import GoalKeepOut, KeepOutBox, KeepOutSummary
from src.robot.safety._capsule import AxisAlignedBox
from src.robot.safety.planning.perceived import (
    DeclaredBody,
    DepthView,
    PerceivedWorld,
    PerceptionGeometryError,
    SelfBody,
    SelfEnvelope,
    VoxelField,
    WorldBuildLimits,
    WorldBuildTuning,
    build_perceived_boxes,
    has_depth,
    target_keep_out_box,
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
    #: The camera answered with a frame that holds no valid depth at all, a covered lens, a dead
    #: emitter, a frame of zeros, or so little that most of what it watches went unmeasured: less
    #: than ``min_depth_coverage`` of its pixels in a fixed camera, as when a cable hangs in front
    #: of the lens, and in a camera on the wrist asked about no goal. Not an empty cell, which is
    #: valid depth that shows nothing on the bench.
    BLIND = "blind"
    #: The frame arrived and the geometry could not be built from it.
    UNUSABLE = "unusable"
    #: The cameras that look at the motion's goal hold mostly holes about it, none of them enough:
    #: the region is closer than they measure, or surfaces they cannot read. Planned against, that
    #: region would be free because nothing was measured there. A fact about where the cameras stand
    #: rather than a fault of a camera, so the motion is refused and no camera is asked again.
    UNSEEN = "unseen"


#: What a camera answers when it cannot vouch for the cell. These are asked again, and after the
#: attempts they raise; every other verdict is decided on the first answer.
_CAMERA_FAILURES = frozenset({WorldVerdict.NO_FRAME, WorldVerdict.STALE, WorldVerdict.BLIND})


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
    #: Where the tool stood when the shutter opened: the TCP in BASE, 4x4, millimetres. Only a
    #: camera on the wrist needs it, and `None` means the producer did not stamp it. A wrist view
    #: refuses such a frame rather than place it by where the tool is now.
    tool_to_base_mm: np.ndarray | None = None
    #: How far the place this frame is put may lie from where it was taken, as its producer
    #: declares it, millimetres and degrees. A camera on the wrist declares how far the tool moved
    #: between the poses read either side of its grab (``depth_source.RigDepthSource``), which its
    #: rig's shutter motion tolerance bounds: the frame is placed by the pose read before, and the
    #: tool moved that far before the shutter closed. It widens the bench band of every point of
    #: the frame (``perceived.DepthView.placement_error_mm``). Zero is a producer that declares
    #: nothing, or an arm that stood still, and then the configured plane clearance holds every
    #: error alone.
    placement_error_mm: float = 0.0
    placement_error_deg: float = 0.0


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

    A view is either fixed or on the wrist, and says which by the one transform it carries. A fixed
    camera is placed by its CAMERA to BASE. A camera on the wrist is placed frame by frame, by its
    CAMERA to TOOL composed with the tool pose its frame was stamped with, because it moves with the
    arm between one frame and the next.
    """

    #: How this camera is named in a refusal an operator has to act on.
    name: str
    #: Where its depth comes from.
    depth_source: "SurfaceDepthSource"
    #: CAMERA to BASE for a fixed camera, 4x4, millimetres. `None` for a camera on the wrist.
    camera_to_base: np.ndarray | None = None
    #: CAMERA to TOOL for a camera on the wrist, 4x4, millimetres. `None` for a fixed camera.
    camera_to_tool: np.ndarray | None = None

    def __post_init__(self) -> None:
        if (self.camera_to_base is None) == (self.camera_to_tool is None):
            raise ValueError(
                f"camera {self.name!r} needs exactly one of camera_to_base, for a fixed camera, and "
                "camera_to_tool, for a camera on the wrist"
            )


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
    #: The camera behind a verdict that is not `FRESH`, empty otherwise. It is what an operator
    #: has to go and look at, and what a refusal raised later names.
    camera: str = ""
    #: The cameras a `FRESH` world was built from, empty otherwise.
    cameras: tuple[str, ...] = ()
    #: When the oldest image in a `FRESH` world was captured, `time.time()` seconds, `None`
    #: otherwise. A fused world is exactly as current as its stalest image, so this is the one
    #: moment it can name.
    captured_at_s: float | None = None
    #: What a `FRESH` world left out: the goal region or why there was none, and the held boxes;
    #: `None` when no goal was asked about and no box was held.
    keep_out: KeepOutSummary | None = None
    #: The cameras that looked at the motion's goal and held a depth over most of the region about
    #: it, in camera order; `None` when no goal was asked about or the world is not `FRESH`. Empty is
    #: a goal no camera looked at, whose surroundings reach the planner as whatever was declared
    #: there.
    goal_seen_by: tuple[str, ...] | None = None
    #: The declared fixtures or meshes the world could not read for its cameras, each with why. What the
    #: cameras see of them comes back as obstacles, which is the direction it is safe to be wrong in.
    unread_declared: tuple[str, ...] = ()

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
        if self.goal_seen_by == ():
            head += (
                "\n  the goal is in no camera's view: around it the planner knows only what is "
                "declared"
            )
        for unread in self.unread_declared:
            head += f"\n  {unread}: what the cameras see of it comes back as obstacles"
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
            "camera": self.camera,
            "dropped_obstacles": (
                self.perceived.dropped_obstacle_count if self.perceived is not None else 0
            ),
            "goal_seen_by": None if self.goal_seen_by is None else list(self.goal_seen_by),
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
    #: How many more times a camera that cannot vouch for the cell is asked before the refresh
    #: raises `CameraWorldUnavailable` instead of refusing. From `perceived.fresh_frame_attempts`.
    fresh_frame_attempts: int = 3
    #: Whether a registration the planner confirms only in part refuses the motion. From
    #: `planning_world.require_registration`, so every driver reads the cell's own answer.
    require_registration: bool = True
    #: The share of the region about a goal a camera looks at, and of a frame's pixels where the
    #: frame is judged whole, that has to hold a depth before it may vouch for anything. Below it a
    #: goal is `UNSEEN` and a frame is `BLIND`. One half is "mostly holes" and nothing finer:
    #: measured 2026-09-23 in a simulation of the owner's wrist D415 at its calibration view (45
    #: degrees down, about 0.5 m from the marker, 1280 x 720 with its 450 mm minimum range), 64 % of
    #: the frame held depth.
    #:
    #: A camera on the wrist is judged by the whole frame only when no goal is named. Measured the
    #: same day by the review's ray cast of the owner's pick (the D415 60 mm beside the Hand-E and
    #: 130 mm behind its TCP, tilted 45 degrees outward, a 40 mm cube): at the grasp the camera
    #: stands 150 mm over the bench and 27 % of the frame holds a depth at 848 x 480, 12 % at
    #: 1280 x 720, all of it bench beyond the minimum range, while the hand's line up is outside
    #: the view at every pose of the pick. Judged whole, that frame raised on every retreat and left
    #: the part clamped at the bench; judged where the retreat goes, it is planned against what is
    #: declared there and the stamp says the goal was not seen.
    min_depth_coverage: float = 0.5
    #: Half the side of the region about a goal whose depth a camera that looks at the goal must
    #: hold, millimetres at the goal's range. About a hand with its fingers open: the space the hand
    #: arrives in, which is what the planner is about to route through.
    goal_region_mm: float = 100.0

    _frames: dict[str, DepthSnapshot] = field(default_factory=dict, init=False, repr=False)
    #: Where the robot's body stood when each cached frame was taken, as its capsule end points
    #: in BASE. A frame is served again only while the body still stands there.
    _frame_bodies: dict[str, "np.ndarray | None"] = field(default_factory=dict, init=False, repr=False)
    _labels: dict[str, tuple[tuple[str, np.ndarray], ...]] = field(
        default_factory=dict, init=False, repr=False
    )
    _exclude: dict[str, tuple[np.ndarray, ...]] = field(
        default_factory=dict, init=False, repr=False
    )
    #: When each offer's frame was taken, by the key the offer was stored under. Each offer ages on
    #: its own stamp.
    _offer_stamps: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    #: Offers held until forgotten, whatever their age: a pick's target while the pick runs.
    _held: set[str] = field(default_factory=set, init=False, repr=False)
    #: The target box each offer's points were fitted to, by the same key.
    _targets: dict[str, KeepOutBox] = field(default_factory=dict, init=False, repr=False)
    #: The declared fixtures and meshes as the cameras see them again, read once when the world is built.
    _declared_bodies: tuple[DeclaredBody, ...] = field(default=(), init=False, repr=False)
    #: The declared shapes that could not be read for that, each with why.
    _unread_declared: tuple[str, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        if self.tuning.voxel_field_mm > 0.0:
            # Pay the distance-transform import while a cell is being built, where a tenth of a
            # second costs nothing. Measured against the real planner: the first field a cell ever
            # built took 126 ms and every one after it 6 ms, and the whole difference was this
            # import landing on the first motion instead of here.
            import scipy.ndimage  # noqa: F401, PLC0415 (imported for its cost, not for its names)

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
        if not 0.0 < float(self.min_depth_coverage) <= 1.0:
            raise PerceptionGeometryError(
                f"min_depth_coverage is a share above 0 and at most 1, got {self.min_depth_coverage!r}: "
                "at 0 a frame of holes would vouch for the cell"
            )
        if not (np.isfinite(float(self.goal_region_mm)) and float(self.goal_region_mm) > 0.0):
            raise PerceptionGeometryError(f"goal_region_mm is finite and above 0, got {self.goal_region_mm!r}")
        self._declared_bodies, self._unread_declared = _declared_bodies(self.declared, self.declared_meshes)

    # -----------------------------------------------------------------------------------------
    # What the pick loop can add, and nothing else has to
    # -----------------------------------------------------------------------------------------

    def offer_segmentation(
        self,
        *,
        camera: Maybe[str] = UNSET,
        labelled_masks: Sequence[tuple[str, np.ndarray]] = (),
        exclude_masks: Sequence[np.ndarray] = (),
        timestamp: float | None = None,
        target_points_base_mm: np.ndarray | None = None,
        target_label: str = "",
        hold: bool = False,
    ) -> None:
        """Hand over the masks from a perception frame, for as long as they stay fresh.

        Two things come from here and neither is required. Names, so a refusal can say which object
        refused rather than printing a number. And the object being grasped, which has to be left out
        of the world: it is the one obstacle the arm is deliberately driving into, and a goal inside
        an obstacle has no plan by construction.

        Masks belong to the camera that produced them, because a pixel means nothing in another
        camera image, so an offer with masks or labels names its camera and one that does not is
        refused. An offer with only target points may leave `camera` unset: a box in BASE belongs
        to no camera, and it is kept under a key of its own.

        Masks only ever leave a fixed camera's view. A camera on the wrist has moved with the arm
        since its frame, so its pixels no longer mark the target; ``target_points_base_mm`` does,
        in BASE. The world fits the box it would fit around those points
        (``target_keep_out_box``) once, here, and leaves its inside out of every view, fixed or on
        the wrist, and out of the voxel field.

        Each offer ages on its own ``timestamp``, so an offer for one camera does not make another's
        masks fresh again. ``hold`` keeps the offer, masks and box, whatever its age until
        :meth:`forget_segmentation`: a pick's target stays out of the world for every motion of the
        pick.

        A cell that never calls this still gets a world. It gets unnamed boxes, and during a pick it
        gets one box where the target is, which is why the pick loop calls it.
        """
        if not chosen(camera) or camera is None:
            if len(labelled_masks) or len(exclude_masks):
                raise PerceptionGeometryError(
                    "a segmentation offer with masks names the camera that took them, one of "
                    f"{[view.name for view in self.cameras]}: a pixel means nothing in another camera's image"
                )
            if target_points_base_mm is None:
                return
            name = f"{_BOX_ONLY_KEY}{target_label or 'target'}"
        else:
            name = str(camera)
            if name not in {view.name for view in self.cameras}:
                raise PerceptionGeometryError(
                    f"no camera named {name!r} in this cell: "
                    f"{[view.name for view in self.cameras]}"
                )
            self._labels[name] = tuple(
                (str(label), np.asarray(mask).astype(bool)) for label, mask in labelled_masks
            )
            self._exclude[name] = tuple(np.asarray(mask).astype(bool) for mask in exclude_masks)
        self._offer_stamps[name] = time.time() if timestamp is None else float(timestamp)
        if hold:
            self._held.add(name)
        else:
            self._held.discard(name)
        self._targets.pop(name, None)
        if target_points_base_mm is not None:
            box = target_keep_out_box(
                target_points_base_mm, name=f"target_{target_label or name}", limits=self.limits, tuning=self.tuning,
            )
            if box is not None:
                self._targets[name] = box

    def drop_cached_frames(self) -> None:
        """Forget the cached readings, so the next question reaches the cameras.

        The cache exists so two plans inside one perception cycle share one reading. A caller that
        knows the scene changed, which in practice means a probe or a test rather than a cell, says
        so here rather than by waiting out the age limit.
        """
        self._frames.clear()
        self._frame_bodies.clear()

    def drop_cached_frame(self, camera: str) -> None:
        """Forget one camera's cached reading, so the next question reaches that camera and no other."""
        self._frames.pop(str(camera), None)
        self._frame_bodies.pop(str(camera), None)

    def forget_segmentation(self) -> None:
        """Drop every offer and every cached frame. Called when a pick ends.

        The next motion then excludes nothing. The frames go too, because a pick that ends has
        changed the cell: a part was taken, released or placed, and a frame cached while its target
        was held out shows the part where the hand held it. The cache keys on the arm alone, and a
        jaw that opens without the arm moving would otherwise hand the next motion that frame.
        """
        self._labels.clear()
        self._exclude.clear()
        self._offer_stamps.clear()
        self._held.clear()
        self._targets.clear()
        self.drop_cached_frames()

    # -----------------------------------------------------------------------------------------
    # The one question
    # -----------------------------------------------------------------------------------------

    def world_for(
        self,
        *,
        self_envelope: SelfEnvelope | None = None,
        near_point_mm: Sequence[float] | None = None,
        now: float | None = None,
        goal_keep_out: Maybe[GoalKeepOut] = UNSET,
    ) -> PlannerWorldSnapshot:
        """The cell as it is, declared and perceived together, ready to register.

        Parameters
        ----------
        self_envelope
            The robot's own body now: where every frame of its chain is and the capsules those
            frames carry, from ``self_envelope.self_envelope``. Padded by ``tuning.margin_mm``, it
            takes the robot back out of what the cameras saw. `None` means the caller cannot
            describe its own body, and then the arm itself would be registered as an obstacle, so
            no perceived world is built at all.
        near_point_mm
            Where the motion is going. It decides which obstacles survive the slot budget, because
            the ones near the path are the ones that matter. When cameras look at it, one of them
            has to hold a depth over most of the region about it (``goal_region_mm``), or the
            world is `UNSEEN`; the cameras that did are named in ``goal_seen_by``.
        now
            The clock, injectable so a test can age a frame without sleeping.
        goal_keep_out
            The space between the jaws at the motion's goal (``self_envelope.goal_keep_out``), left
            out of every view and the voxel field beside the held boxes, or the reason there is
            none, which the snapshot keeps.
        """
        clock = time.time() if now is None else float(now)
        declared = tuple(dict(box) for box in self.declared)
        meshes = tuple(dict(mesh) for mesh in self.declared_meshes)
        goal = None if near_point_mm is None else np.asarray(near_point_mm, dtype=np.float64).reshape(-1)
        if goal is not None and not (goal.shape == (3,) and bool(np.all(np.isfinite(goal)))):
            goal = None
        frames, verdict, reason, failed_camera = self._frames_for(
            clock, _placed_body(self_envelope), goal_named=goal is not None,
        )
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
                meshes=meshes, declared_count=len(declared), camera=failed_camera,
            )

        oldest = self._oldest_age_ms(frames, clock)
        placed: dict[str, np.ndarray] = {}
        for camera in self.cameras:
            if camera.camera_to_base is not None:
                placed[camera.name] = camera.camera_to_base
                continue
            tool_to_base = frames[camera.name].tool_to_base_mm
            if tool_to_base is None or camera.camera_to_tool is None:
                # Decided on the first reading and not asked again: where the tool stood is not
                # something another reading of the camera can supply.
                return PlannerWorldSnapshot(
                    verdict=WorldVerdict.UNUSABLE, cuboids=declared, perceived=None, age_ms=oldest,
                    reason=(
                        f"camera {camera.name!r} is on the wrist and its depth frame carries no tool pose, so "
                        "where it stood when the shutter opened is unknown and nothing it saw can be placed"
                    ),
                    meshes=meshes, declared_count=len(declared), camera=camera.name,
                )
            placed[camera.name] = (
                np.asarray(tool_to_base, dtype=np.float64) @ np.asarray(camera.camera_to_tool, dtype=np.float64)
            )
        if self_envelope is None:
            return PlannerWorldSnapshot(
                verdict=WorldVerdict.UNUSABLE, cuboids=declared, perceived=None, age_ms=oldest,
                reason=(
                    "the arm cannot describe where its own links are, so every point of the robot "
                    "in a camera view would be registered as an obstacle and the arm would be "
                    "standing inside one"
                ),
                meshes=meshes, declared_count=len(declared),
            )

        # Where the motion goes has to have been measured by a camera that looks at it. The views are
        # fused, so one camera that measured the region is enough and a second one's holes take
        # nothing away; a camera that does not look at it vouches for nothing there, and the stamp
        # says so.
        goal_seen_by: tuple[str, ...] | None = None
        if goal is not None:
            seers: list[str] = []
            blind_to_it: list[tuple[str, float]] = []
            for camera in self.cameras:
                share = _goal_coverage(frames[camera.name], placed[camera.name], goal, float(self.goal_region_mm))
                if share is None:
                    continue
                if share < float(self.min_depth_coverage):
                    blind_to_it.append((camera.name, share))
                else:
                    seers.append(camera.name)
            if blind_to_it and not seers:
                shares = " and ".join(f"{name!r} {100.0 * share:.0f}%" for name, share in blind_to_it)
                return PlannerWorldSnapshot(
                    verdict=WorldVerdict.UNSEEN, cuboids=declared, perceived=None, age_ms=oldest,
                    reason=(
                        f"the cameras that look at where this motion goes hold a depth on {shares} of the "
                        f"{2.0 * float(self.goal_region_mm):.0f} mm about it, and at least "
                        f"{100.0 * float(self.min_depth_coverage):.0f}% must: the rest is closer than a camera "
                        "measures or a surface it cannot read, and planned against it would be free space "
                        "nobody saw. Look from further away, or at a depth mode with a shorter minimum range"
                    ),
                    meshes=meshes, declared_count=len(declared), camera=blind_to_it[0][0],
                )
            goal_seen_by = tuple(seers)

        live = {key for key in self._offer_stamps if self._labels_fresh(key, clock)}
        views = [
            DepthView(
                surface_depth_mm=frames[camera.name].depth_mm,
                intrinsics=frames[camera.name].intrinsics,
                camera_to_base=placed[camera.name],
                exclude_masks=self._masks_for(camera, live, self._exclude),
                labelled_masks=self._masks_for(camera, live, self._labels),
                name=camera.name,
                timestamp=frames[camera.name].timestamp,
                **_placement_error(camera, frames[camera.name]),
            )
            for camera in self.cameras
        ]
        held_keys = [key for key in sorted(live) if key in self._targets]
        keep_out = tuple(self._targets[key] for key in held_keys)
        region = goal_keep_out.region if chosen(goal_keep_out) else None
        if region is not None:
            keep_out = keep_out + (region,)

        try:
            perceived = build_perceived_boxes(
                views=views,
                limits=self.limits,
                tuning=self.tuning,
                self_body=SelfBody.from_frames(
                    self_envelope.frames_mm, self_envelope.capsules,
                    padding_mm=float(self.tuning.margin_mm),
                ),
                near_point_mm=near_point_mm,
                keep_out=keep_out,
                # Where the robot's body can be, not where its TCP may go: the links, the hand and
                # a wrist camera swing past the workspace box, and an obstacle there is one they meet.
                reach=self_envelope.reach(padding_mm=float(self.tuning.margin_mm)),
                declared=self._declared_bodies,
                # Around the targets held out, and not around the goal's jaw region, which is not padded.
                cut_around=tuple(self._targets[key] for key in held_keys),
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
        summary = None
        if chosen(goal_keep_out) or held_keys:
            counts = perceived.keep_out_points
            summary = KeepOutSummary(
                goal_points=None if region is None else int(counts.get(region.name, 0)),
                goal_reason="" if region is not None or not chosen(goal_keep_out) else goal_keep_out.reason,
                held=tuple(
                    ("box only" if key.startswith(_BOX_ONLY_KEY) else key, float(self._offer_stamps[key]),
                     int(counts.get(self._targets[key].name, 0)))
                    for key in held_keys
                ),
            )
        return PlannerWorldSnapshot(
            verdict=WorldVerdict.FRESH,
            cuboids=tuple(merge_planner_worlds(declared, perceived_boxes)),
            perceived=perceived,
            age_ms=oldest,
            meshes=meshes, declared_count=len(declared),
            cameras=tuple(camera.name for camera in self.cameras),
            captured_at_s=self._oldest_capture_s(frames),
            keep_out=summary,
            goal_seen_by=goal_seen_by,
            unread_declared=self._unread_declared,
        )

    # -----------------------------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------------------------

    def _frames_for(
        self, clock: float, body: "np.ndarray | None" = None, *, goal_named: bool = False,
    ) -> tuple[dict[str, DepthSnapshot], WorldVerdict, str, str]:
        """A reading from every camera: the cached one while it is young and the robot has not moved.

        Otherwise a new grab. ``body`` is where the robot stands now, as :func:`_placed_body` gives
        it. A frame shows the robot where it stood when it was taken, and the self filter takes out
        the robot where it stands now, so a frame taken before a motion would keep the arm it shows
        as an obstacle beside the arm. ``None`` is a body nobody can describe: the world built from
        it is refused whatever frame it uses, so the age alone decides.

        Every camera has to answer. A cell with two cameras has two because one of them cannot see
        the whole cell, so carrying on without one means planning against a world with a hole in it
        exactly where nobody is looking. The verdict names the camera that failed, because that is
        the thing an operator has to go and fix.

        A blind frame is judged per camera and never cached. Fused, a sighted camera's points would
        hide a blind one, and cached, a young blind frame would be served again instead of asking
        the camera. A cached frame is judged again, because whether it is blind depends on what it
        is asked about (:meth:`_blind`): ``goal_named`` says the motion names where it goes.
        """
        frames: dict[str, DepthSnapshot] = {}
        for camera in self.cameras:
            cached = self._frames.get(camera.name)
            cached_age = None if cached is None else self._age_ms(cached, clock)
            reused = (
                cached is not None and cached_age is not None and cached_age <= self.max_age_ms
                and self._body_still(camera.name, body)
            )
            frame = cached if reused else camera.depth_source.grab_surface_depth()
            if frame is None:
                return (
                    frames,
                    WorldVerdict.NO_FRAME,
                    f"camera {camera.name!r} returned no depth, so nothing can vouch for what is "
                    "in the part of the cell it watches",
                    camera.name,
                )
            blind = self._blind(camera, frame, goal_named=goal_named)
            if blind:
                self.drop_cached_frame(camera.name)
                return frames, WorldVerdict.BLIND, blind, camera.name
            frames[camera.name] = frame
            if reused:
                continue
            self._frames[camera.name] = frame
            self._frame_bodies[camera.name] = body

            age = self._age_ms(frame, clock)
            if age is None:
                return (
                    frames,
                    WorldVerdict.STALE,
                    f"the depth frame from camera {camera.name!r} carries no capture time, so its "
                    "age is unknown and it cannot be shown to be inside the limit this cell allows",
                    camera.name,
                )
            if age > self.max_age_ms:
                return (
                    frames,
                    WorldVerdict.STALE,
                    f"the depth frame from camera {camera.name!r} is {age:.0f} ms old and this "
                    f"cell allows {self.max_age_ms:.0f} ms",
                    camera.name,
                )
        return frames, WorldVerdict.FRESH, "", ""

    def _blind(self, camera: CameraView, frame: DepthSnapshot, *, goal_named: bool) -> str:
        """Why ``frame`` cannot vouch for the part of the cell ``camera`` watches, or empty when it can.

        No depth at all is blind in every camera. Less than ``min_depth_coverage`` of the frame
        holding a depth is blind in a fixed camera, which is never carried toward what it sees, so
        a frame of it mostly holes is the camera or something in front of it. A camera on the wrist
        is carried inside its own minimum range by every grasp, so its frame is blind by that share
        only when no goal is named; asked about a goal it is judged about the goal instead
        (``world_for``: the region about it, and the stamp saying what the rest of the frame did not
        see).
        """
        coverage = _depth_coverage(frame)
        if coverage <= 0.0:
            return (
                f"camera {camera.name!r} returned a depth frame with no valid pixel, so it is "
                "blind and cannot vouch for the part of the cell it watches"
            )
        on_wrist = camera.camera_to_tool is not None
        if coverage >= float(self.min_depth_coverage) or (on_wrist and goal_named):
            return ""
        return (
            f"camera {camera.name!r} returned a depth frame in which {100.0 * coverage:.0f}% of the "
            f"pixels hold a depth, and at least {100.0 * float(self.min_depth_coverage):.0f}% must, so "
            "it is mostly blind and cannot vouch for the part of the cell it watches: it stands "
            "closer to what it sees than its minimum range, something covers part of the lens, or "
            "it looks at surfaces it cannot read"
            + (". It is on the wrist and the motion names no goal, so the whole frame is what it is asked "
               "about" if on_wrist else "")
        )

    def _body_still(self, camera: str, body: "np.ndarray | None") -> bool:
        """Whether the robot stands where it stood when ``camera``'s cached frame was taken, within 1 mm."""
        if body is None:
            return True
        then = self._frame_bodies.get(camera)
        if then is None or then.shape != body.shape:
            return False
        return body.size == 0 or float(np.max(np.linalg.norm(then - body, axis=1))) <= _BODY_STILL_MM

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

    @staticmethod
    def _oldest_capture_s(frames: dict[str, DepthSnapshot]) -> float | None:
        """When the stalest image was captured, or `None` when any image carries no capture time."""
        stamps = [frame.timestamp for frame in frames.values()]
        if not stamps or any(stamp is None for stamp in stamps):
            return None
        return min(float(stamp) for stamp in stamps if stamp is not None)

    def _labels_fresh(self, key: str, clock: float) -> bool:
        """Whether the offer stored under ``key`` still applies: held, or no older than a frame may be.

        An offer ages exactly like the frame it came from, on its own stamp. An excluded target from
        half a second ago is a hole in the world where the object no longer is, and the object may
        well be somewhere else in it now. A held offer is the exception a pick asks for, and it ends
        when the pick forgets it.
        """
        if key in self._held:
            return True
        stamp = self._offer_stamps.get(key)
        if stamp is None:
            return False
        return (clock - stamp) * 1000.0 <= self.max_age_ms

    @staticmethod
    def _masks_for(camera: CameraView, live: set[str], store: dict[str, tuple[Any, ...]]) -> tuple[Any, ...]:
        """The live masks stored for ``camera``, and none for a camera on the wrist.

        A wrist camera's pixels moved with the arm since its frame.
        """
        if camera.camera_to_tool is not None or camera.name not in live:
            return ()
        return store.get(camera.name, ())


#: The key a box-only offer is kept under, followed by its target label. No camera view carries
#: masks under it.
_BOX_ONLY_KEY = "box only: "

#: How far the robot's body may move, at any capsule end, before a cached frame of it is taken
#: again, millimetres. Well above joint encoder noise at arm's length, well below the padding the
#: self filter adds.
_BODY_STILL_MM = 1.0


def _declared_bodies(
    cuboids: Sequence[dict[str, Any]], meshes: Sequence[dict[str, Any]],
) -> tuple[tuple[DeclaredBody, ...], tuple[str, ...]]:
    """The declared world in the planner's wire format, as the cameras see it again, and what could not be read.

    A box is placed by its wire pose, metres and WXYZ, and sized by its wire dimensions. A mesh is read from the
    path the sidecar reads, from the working directory it inherits from this process, scaled and placed as the
    sidecar places it. One that cannot be read here is said, and left out of this filter only: what the cameras see
    of it stays an obstacle.
    """
    bodies: list[DeclaredBody] = []
    unread: list[str] = []
    for cuboid in cuboids:
        name = str(cuboid.get("name", "") or "unnamed box")
        try:
            pose = [float(v) for v in cuboid["pose"]]
            dims = [float(v) for v in cuboid["dims_m"]]
            bodies.append(DeclaredBody.box(name, _wire_matrix_mm(pose), [500.0 * d for d in dims]))
        except (KeyError, TypeError, ValueError, IndexError, PerceptionGeometryError) as exc:
            unread.append(f"declared box {name!r} could not be read ({type(exc).__name__}: {exc})")
    for mesh in meshes:
        name = str(mesh.get("name", "") or "unnamed mesh")
        try:
            bodies.append(_mesh_body(name, mesh))
        except Exception as exc:  # noqa: BLE001 (a mesh file can fail in any way; it only keeps obstacles)
            unread.append(f"declared mesh {name!r} could not be read ({type(exc).__name__}: {exc})")
    return tuple(bodies), tuple(unread)


def _wire_matrix_mm(pose: Sequence[float]) -> np.ndarray:
    """A wire pose, metres and WXYZ, as a 4x4 in millimetres."""
    if len(pose) != 7:
        raise ValueError(f"a wire pose is three metres and a WXYZ quaternion, got {len(pose)} numbers")
    w, x, y, z = pose[3:]
    norm = float(np.sqrt(w * w + x * x + y * y + z * z))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("a wire pose's quaternion has no length")
    w, x, y, z = (v / norm for v in (w, x, y, z))
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    matrix[:3, 3] = [1000.0 * float(v) for v in pose[:3]]
    return matrix


def _mesh_body(name: str, mesh: dict[str, Any]) -> DeclaredBody:
    """One declared mesh, read from its file and placed in BASE millimetres as the sidecar places it."""
    import trimesh  # noqa: PLC0415 (an optional dependency, and only a declared mesh needs it)

    loaded: Any = trimesh.load(str(mesh["file_path"]), force="mesh")
    scale = np.asarray(mesh.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64).reshape(3)
    matrix = _wire_matrix_mm([float(v) for v in mesh["pose"]])
    vertices_m = np.asarray(loaded.vertices, dtype=np.float64) * scale
    vertices_mm = (1000.0 * vertices_m) @ matrix[:3, :3].T + matrix[:3, 3]
    return DeclaredBody.mesh(name, vertices_mm, np.asarray(loaded.faces, dtype=np.int64))


def _placed_body(envelope: SelfEnvelope | None) -> "np.ndarray | None":
    """The robot's body as the end points of its capsules in BASE, or ``None`` for a body nobody can describe."""
    if envelope is None:
        return None
    points = []
    for capsule in envelope.capsules:
        frame = np.asarray(envelope.frames_mm[capsule.frame], dtype=np.float64)
        for end in (capsule.start_mm, capsule.end_mm):
            points.append(frame[:3, :3] @ np.asarray(end, dtype=np.float64) + frame[:3, 3])
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def _depth_coverage(frame: DepthSnapshot) -> float:
    """The share of the frame's pixels that hold a depth the converter would use, 0 to 1.

    It is the same test the converter applies to each pixel (``perceived.has_depth``), so a frame
    judged blind here is exactly a frame whose points would have been too few there.
    """
    valid = has_depth(frame.depth_mm)
    return float(np.count_nonzero(valid)) / float(valid.size) if valid.size else 0.0


def _has_valid_depth(frame: DepthSnapshot) -> bool:
    """Whether any pixel holds a depth the converter would use: finite and in front of the camera."""
    return _depth_coverage(frame) > 0.0


def _goal_coverage(
    frame: DepthSnapshot, camera_to_base: np.ndarray, goal_mm: np.ndarray, region_mm: float
) -> float | None:
    """The share of the pixels about ``goal_mm`` in ``frame`` that hold a depth, or ``None`` when the camera does not look at it.

    The goal is carried into the camera and projected with the frame's own intrinsics, the inverse
    of how a pixel is back-projected. It is looked at when it stands in front of the camera and
    projects inside the image. The region is the square of ``region_mm`` on each side of it at its
    range, clipped to the image: the pixels whose rays pass about the goal. A ray with a depth
    measured the space along it up to the surface it met, the goal's included; a ray with none
    measured nothing.
    """
    matrix = np.asarray(frame.intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return None
    fx, fy, cx, cy = float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[0, 2]), float(matrix[1, 2])
    if fx == 0.0 or fy == 0.0:
        return None
    try:
        base_to_camera = np.linalg.inv(np.asarray(camera_to_base, dtype=np.float64))
    except np.linalg.LinAlgError:
        return None
    x, y, z = base_to_camera[:3, :3] @ goal_mm + base_to_camera[:3, 3]
    depth = np.asarray(frame.depth_mm)
    if depth.ndim != 2 or not np.isfinite(z) or z <= 0.0:
        return None
    rows, cols = depth.shape
    u, v = fx * x / z + cx, fy * y / z + cy
    if not (0.0 <= u < cols and 0.0 <= v < rows):
        return None
    half_u, half_v = abs(fx) * region_mm / z, abs(fy) * region_mm / z
    col_lo, col_hi = max(0, int(np.floor(u - half_u))), min(cols, int(np.ceil(u + half_u)) + 1)
    row_lo, row_hi = max(0, int(np.floor(v - half_v))), min(rows, int(np.ceil(v + half_v)) + 1)
    window = has_depth(depth[row_lo:row_hi, col_lo:col_hi])
    return float(np.count_nonzero(window)) / float(window.size) if window.size else None


def _placement_error(camera: CameraView, frame: DepthSnapshot) -> dict[str, float]:
    """How far ``frame`` may be placed off, as ``DepthView`` takes it: millimetres and radians.

    The producer declares the tool's motion measured across the grab. For a camera on the wrist a turn of
    the tool about the TCP also moves the camera, by at most the angle times its distance from the
    TCP, which is added to the translation.
    """
    error_mm = float(frame.placement_error_mm)
    error_rad = float(np.radians(float(frame.placement_error_deg)))
    if camera.camera_to_tool is not None:
        error_mm += error_rad * float(np.linalg.norm(np.asarray(camera.camera_to_tool, dtype=np.float64)[:3, 3]))
    return {"placement_error_mm": error_mm, "placement_error_rad": error_rad}


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
    #: The cameras the world was built from, empty when none answered.
    cameras: tuple[str, ...] = ()
    #: When the oldest image in it was captured, `time.time()` seconds, or `None`.
    captured_at_s: float | None = None
    #: What the world left out for this motion: its goal region or why there was none, and the
    #: held boxes.
    keep_out: KeepOutSummary | None = None
    #: The share of each camera's image that held a depth, in camera order, as the world was built
    #: (``PerceivedWorld.depth_coverage``). Empty when no world was built.
    depth_coverage: tuple[tuple[str, float], ...] = ()
    #: The cameras that looked at the motion's goal and measured the region about it; `None` when
    #: no goal was asked about or no world was built, empty when no camera looked at it.
    goal_seen_by: tuple[str, ...] | None = None

    @property
    def ok(self) -> bool:
        """Whether the caller may plan. Anything else is a refusal, never a warning."""
        return self.verdict is WorldVerdict.FRESH and not self.reason

    @property
    def total_ms(self) -> float:
        return self.build_ms + self.register_ms

    def unseen(self) -> UnseenSpace | None:
        """What this refresh's world did not see: pixels with no depth and a goal no camera looked at, or `None`."""
        no_depth = tuple((name, 1.0 - share) for name, share in self.depth_coverage if share < 1.0)
        goal_out_of_view = self.goal_seen_by == ()
        if not no_depth and not goal_out_of_view:
            return None
        return UnseenSpace(no_depth=no_depth, goal_out_of_view=goal_out_of_view)

    def camera_world(self) -> CameraWorldStamp | None:
        """The stamp this refresh vouches for, or `None` when it vouches for nothing.

        PLANNED, on its cameras and the capture time of its oldest image, for a refresh the caller
        may plan against. A refused refresh vouches for nothing, and neither does one that cannot
        name a camera or a capture time, because a stamp that vouches has to say for what. What the
        world did not see rides on the stamp (:meth:`unseen`), so it is not vouched for.
        """
        if not self.ok or not self.cameras or self.captured_at_s is None:
            return None
        # The summary only when something was kept out: a stamp says what stood behind the motion.
        kept = self.keep_out if self.keep_out is not None and self.keep_out.in_force else None
        return CameraWorldStamp.planned(
            cameras=self.cameras, captured_at_s=self.captured_at_s, keep_out=kept, unseen=self.unseen(),
        )

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
        if self.keep_out is not None:
            line += f"; kept out: {self.keep_out.render()}"
        unseen = self.unseen()
        if unseen is not None:
            line += f"; not seen, planned as free: {unseen.render()}"
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
            "cameras": list(self.cameras),
            "captured_at_s": self.captured_at_s,
            "reason": self.reason,
            "keep_out": None if self.keep_out is None else self.keep_out.to_dict(),
            "depth_coverage": {name: float(share) for name, share in self.depth_coverage},
            "goal_seen_by": None if self.goal_seen_by is None else list(self.goal_seen_by),
        }


def refresh_planner_world(
    *,
    source: LivePlannerWorld,
    client: Any,
    self_envelope: SelfEnvelope | None,
    near_point_mm: Sequence[float] | None = None,
    require_registration: bool = True,
    now: float | None = None,
    goal_keep_out: Maybe[GoalKeepOut] = UNSET,
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
    # A camera that cannot vouch for the cell is asked again before anything is decided, and after
    # its attempts the motion raises rather than being refused. Only the camera's own failures are
    # asked again: an arm that cannot place its links, or geometry that cannot be built, gives the
    # same answer however often it is asked.
    snapshot = source.world_for(
        self_envelope=self_envelope, near_point_mm=near_point_mm, now=now, goal_keep_out=goal_keep_out
    )
    readings = 1
    while snapshot.verdict in _CAMERA_FAILURES and readings <= max(0, int(source.fresh_frame_attempts)):
        source.drop_cached_frame(snapshot.camera)
        snapshot = source.world_for(
            self_envelope=self_envelope, near_point_mm=near_point_mm, now=now, goal_keep_out=goal_keep_out
        )
        readings += 1
    build_ms = (time.perf_counter() - started) * 1000.0
    dropped = snapshot.perceived.dropped_obstacle_count if snapshot.perceived is not None else 0

    if snapshot.verdict in _CAMERA_FAILURES:
        refused = WorldRefresh(
            verdict=snapshot.verdict, sent=0, registered=0, build_ms=build_ms, register_ms=0.0,
            age_ms=snapshot.age_ms, dropped_obstacles=dropped, reason=snapshot.reason,
        )
        raise CameraWorldUnavailable(
            camera=snapshot.camera, verdict=snapshot.verdict, attempts=readings,
            reason=snapshot.reason, refresh=refused,
        )

    if not snapshot.usable:
        return WorldRefresh(
            verdict=snapshot.verdict, sent=0, registered=0, build_ms=build_ms, register_ms=0.0,
            age_ms=snapshot.age_ms, dropped_obstacles=dropped, reason=snapshot.reason,
        )

    started = time.perf_counter()
    boxes = [dict(box) for box in snapshot.cuboids]
    meshes = [dict(m) for m in snapshot.meshes]
    voxel_field = snapshot.perceived.voxels if snapshot.perceived is not None else None
    scene_setter = getattr(client, "set_scene", None)
    if voxel_field is not None and callable(scene_setter):
        # One request for the boxes, the meshes and the field. Two are not atomic: the field request
        # rebuilds the planner's world from the boxes it remembered, and a declared mesh is lost in
        # between with both requests reporting success.
        registration = scene_setter(boxes, meshes, _write_voxel_field(client, voxel_field))
        registered = int(registration.world_set)
        wanted, voxels = True, registration.voxels_set
        why = str(registration.reason or "") or "the planner did not say why"
    else:
        # Named only when there is something to name: a planner client from before meshes existed
        # takes one argument, and a cell that declares no mesh should not need a newer one.
        registered = int(client.set_world(boxes, meshes) if meshes else client.set_world(boxes))
        wanted, voxels, why = _send_voxel_field(client, snapshot)
    register_ms = (time.perf_counter() - started) * 1000.0

    reasons: list[str] = []
    if wanted and voxels is None:
        # A cell that asked for a live scene and did not get one is planning against less than it
        # believes, and believing it is the whole danger. Refuse, and say which of the two happened.
        reasons.append(
            f"the live scene did not reach the planner ({why}), so it would be planning against "
            "the declared world alone while the cameras can see more than that"
        )
    expected = len(boxes) + len(meshes)
    if registered != expected and require_registration:
        # Kept beside a field reason rather than written over it, so an operator is told all of
        # what is wrong.
        reasons.append(
            f"the planner confirmed {registered} of {expected} obstacle(s), so it is "
            "planning against a world that is missing part of this cell. Set "
            "safety.planning_world.require_registration false to plan anyway, deliberately."
        )
    if dropped:
        # An obstacle the cameras saw and the planner has no slot for is one it will route through,
        # so it refuses the motion rather than standing as a footnote in the render.
        reasons.append(
            f"{dropped} perceived obstacle(s) did not fit the {source.tuning.max_boxes} slot(s) this "
            "cell allows, so the planner would route through them"
        )
    reason = "; and ".join(reasons)
    return WorldRefresh(
        verdict=snapshot.verdict, sent=expected, registered=registered,
        build_ms=build_ms, register_ms=register_ms, age_ms=snapshot.age_ms,
        dropped_obstacles=dropped, reason=reason,
        guard_boxes=() if reason else _guard_boxes(snapshot.perceived),
        voxels_registered=voxels,
        cameras=snapshot.cameras, captured_at_s=snapshot.captured_at_s,
        keep_out=snapshot.keep_out,
        depth_coverage=(
            () if snapshot.perceived is None
            else tuple((name, float(share)) for name, share in snapshot.perceived.depth_coverage.items())
        ),
        goal_seen_by=snapshot.goal_seen_by,
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


def _write_voxel_field(client: Any, voxels: VoxelField) -> dict[str, Any]:
    """Write the field where the planner reads it, in its sign and unit, and say where and on which grid.

    The file is the client's own when it names one, so two arms in one process never write the
    same file; a client without one falls back to one file per process. `VoxelField` is in
    millimetres like the rest of this package and the planner reads metres, so the values are
    divided by 1000 here and nowhere else. The same file is reused every refresh, because a new one
    per motion would fill a temp directory at the rate the arm moves.
    """
    path = getattr(client, "live_scene_path", None) or os.path.join(
        tempfile.gettempdir(), f"willy_live_scene_{os.getpid()}.npy"
    )
    np.save(path, (voxels.field / 1000.0).astype(np.float16))
    return {
        "path": str(path),
        "dims_m": [d / 1000.0 for d in voxels.dims_mm],
        "voxel_size_m": voxels.voxel_size_mm / 1000.0,
        "pose": [c / 1000.0 for c in voxels.center_mm] + [1.0, 0.0, 0.0, 0.0],
    }


def _send_voxel_field(
    client: Any, snapshot: PlannerWorldSnapshot
) -> "tuple[bool, int | None, str]":
    """Hand the planner the field on its own, for a client that has no one-request scene call.

    Returns `(wanted, registered, why)`. `wanted` says whether this cell asked for a live scene at
    all, which is the difference between a cell that never configured one and a cell whose planner
    would not take it. Both come back with `registered` as `None` and only one of them is a problem.

    A 30 mm grid over a two metre cell is 179,560 values. That is not a message, and this protocol is
    one JSON object per line, so the field is written where the sidecar can read it and the request
    carries the path. Writing costs about a millisecond and the sidecar reads it in about two.
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
    payload = _write_voxel_field(client, snapshot.perceived.voxels)
    registered = setter(
        payload["path"],
        dims_m=payload["dims_m"],
        voxel_size_m=payload["voxel_size_m"],
        pose=payload["pose"],
    )
    if registered is None:
        return True, None, "the planner refused it"
    return True, int(registered), ""
