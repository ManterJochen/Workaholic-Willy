"""What a task keeps of the parts it grounded, to find them again with no detector: following (the owner, 2026-10-09).

A task's picks used to ask the detector at every first look, Qwen boxing every part on the mat in 7 to 17 s (13.1 s
at the median, 15 groundings on the cell, 2026-10-07/08) to find the same parts where the last pick left them. The
owner's rule: Qwen on the first pick and on every trigger, SAM2 on the known parts between, "das muss wirklich richtig
gehen und passen ... wir können keine Sicherheit verlieren": all or nothing per frame, every check strict, and any
doubt grounds the frame with the detector as before.

:class:`KeptScene` is that memory: the parts one look grounded (or followed), each as its surface in BASE, its colour,
its top and its size in pixels, with the look's own depth, lens and camera placement. It is built from a pick's first
look (:meth:`KeptScene.of_look`), less the part the pick gripped (:meth:`KeptScene.without`), and handed to the next
pick, whose first look reads it on its own frame, with no extra grab (:meth:`KeptScene.following`):

1. **Nothing new in depth.** The kept frame's points go into the new camera and the new frame's into the kept one,
   inside the cell's workspace: a point is changed where it is off by more than :data:`CHANGED_MM`, or where it is new,
   stands over the support and the kept frame read a hole or something farther there (a part a person set down on the
   shiny profiles, where the last frame had holes, is seen only this second way). Left out: the task's keep-out regions
   (its bin, the circle about its drop: an expected change), each kept part's footprint grown by ``max_shift_mm`` and
   :data:`FOOTPRINT_GROWTH_MM` (the parts are judged one by one), and the taken part's, which must read the support
   (:data:`TAKEN_TOP_MM`): a part under it would stand out. A changed cluster of :data:`CHANGED_CLUSTER_PX` pixels, or
   half the smallest kept part, grounds the frame.
2. **Each part where it stood.** Its surface, z-buffered onto the new frame by the locator's rule
   (``locator._measured``): :data:`SEEN_SHARE` of it seen with depth, :data:`AGREE_SHARE` of that within
   :data:`AGREE_MM` and its colour within :data:`DELTA_E` stands where it stood, its box its pixels padded
   :data:`BOX_PAD_PX`; :data:`GONE_SHARE` farther by :data:`GONE_MM` is gone, and grounds the frame; anything between
   is boxed wider by ``max_shift_mm``, for its new mask to measure the move.
3. **SAM2 on the boxes** (the backend's ``segment_boxes``), no detector asked.
4. **Each mask accepted** (:meth:`Following.accept`) only where its pixels are :data:`AREA_RATIO` of the kept part's,
   its surface moved at most ``max_shift_mm`` and at most ``max_creep_mm`` since its grounding, at most
   :data:`LEAK_SHARE` of it lies :data:`LEAK_MM` past the kept footprint shifted by that move, its top stands within
   :data:`TOP_MM` and its colour within :data:`DELTA_E`; and every mask is clipped to that shifted footprint and
   :data:`LEAK_MM`, so the target a frame leaves out of the planner world is at most the kept part and 25 mm, its own
   pixels of this frame.

Only where every part passes is the frame the followed masks under their kept labels; any failure, and the frame is
grounded by the detector as before, the reason said. The camera source runs the colour check on the followed masks as
on every mask, and a refusal grounds the frame too. Following never adds or removes a depth point, and the kept frame
is never handed to the planner world: the world is built from fresh depth before every motion, and the exact guard
judges every sample of every path, as before.

The later looks of one pick (:meth:`KeptScene.projected`, map2's F): the parts the pick's first look saw, each its
surface projected into the later look at its stamped pose, padded :data:`BOX_PAD_PX`, and SAM2 on the boxes; each mask
must stay within the part's footprint and :data:`LEAK_MM` (the extent guard), cover :data:`AREA_RATIO`'s lower bound of
the projected top, stand at its top and keep its colour, else the look is grounded. The parts have not moved within a
pick: the camera has.

Numbers (CPU, the cell's recorded frames of 2026-10-07/08, ``speedmap4/reuse``): the parts left on the mat read their
depth back from pick to pick at a median 0.9 to 2.0 mm, 92 to 99.6 % within 8 mm, colour dE 0.0 to 2.9; a taken part's
spot 0 % within, dE 43 to 46; the whole scene checked both ways at every third pixel found no changed cluster in 7
transitions (noise 0.6 to 0.8 % of the points, all in clusters under 300 px) and found a pasted part on the mat (3034
px) and on the profiles (3742 px); a kept cloud shifted 3 mm still reads 0.875 within 8 mm, 5 mm 0.815; SAM2 on the
projected box: masks 0.73 to 0.80 IoU of the depth blob, 0 % outside the footprint and 15 mm.

What remains: a part of the same size, height and colour set down within ``max_shift_mm`` of a kept one, nothing else
changed, is followed as that part. In everything the camera measures it is then a part of that kind, and the guard still
judges every motion against depth. ``refresh_every_picks`` bounds how long such a swap goes unseen.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from src.robot.constants import create_robot_logger

__all__ = ["Accepted", "Following", "KeptPart", "KeptScene"]

#: The memory's own lines, to the robot log and ``kept_scene.log``: every part it followed or did not, with its numbers.
_LOG: logging.Logger = create_robot_logger(__name__, "kept_scene.log")

#: Every how many pixels, each way, the scene check reads both frames: 10 to 16 ms on the desk for both ways.
SCENE_STRIDE_PX = 3
#: A point is changed where the other frame reads it off by more than this, millimetres: no cluster on 7 transitions.
CHANGED_MM = 10.0
#: The size of a changed cluster, pixels, that grounds the frame where half the smallest kept part is not smaller: noise
#: came in clusters under 300 px, a part set down in 3034 and 3742.
CHANGED_CLUSTER_PX = 1000
#: How far over the support a new point stands to be a part that came, millimetres.
OVER_SUPPORT_MM = 10.0
#: How far past ``max_shift_mm`` each kept part's footprint is left to the part checks, millimetres; the taken part's
#: footprint is grown by this alone.
FOOTPRINT_GROWTH_MM = 15.0
#: How near a point of the new frame stands to the taken part's surface, in BASE xy, to read its footprint, millimetres.
FOOTPRINT_MM = 3.0
#: The taken part's footprint reads at most this over the support (its 98th percentile), millimetres: measured -0.0 and
#: -0.9 mm over the mat after the grip, so a part that was under it stands out.
TAKEN_TOP_MM = 10.0
#: The share of a kept part's surface the new frame must see with depth.
SEEN_SHARE = 0.80
#: A kept part's point reads where it stood within this, millimetres, and the share of them that does stands it where it
#: stood: 0.977 at 0 mm, 0.875 at 3, 0.815 at 5 mm (a cube's surface shifted in x, follow_rules.py).
AGREE_MM = 8.0
AGREE_SHARE = 0.85
#: A kept part reads gone where this share of it reads farther by more than :data:`GONE_MM`: a taken part's spot read
#: 100 % more than 10 mm farther.
GONE_MM = 10.0
GONE_SHARE = 0.60
#: The colour a part keeps, CIE76 dE in L*a*b*: measured 0.0 to 2.9 for the parts left, 43 to 46 for a taken one's spot.
DELTA_E = 10.0
#: How far past its pixels a part's box reaches, pixels: SAM2 gave the same mask with 12 px and with none.
BOX_PAD_PX = 12
#: A followed mask's pixels against the kept part's: SAM2's masks were 3232 to 3932 px against blobs of 3257 to 4133.
AREA_RATIO = (0.6, 1.5)
#: How far past the (shifted) kept footprint a followed mask's surface may reach, in BASE xy, and the share of it that
#: may: measured 0 % past 15 mm.
LEAK_MM = 15.0
LEAK_SHARE = 0.02
#: How far a followed mask's top (its 90th percentile) may stand from the kept part's, millimetres: measured +0.2, +2.2.
TOP_MM = 6.0
TOP_PERCENTILE = 90.0
#: The gripped part is the kept part whose surface's middle stands nearest its cloud's, within this, in BASE xy.
NEAREST_TAKEN_MM = 15.0
#: A kept part's surface is thinned to this voxel, millimetres: about 1000 points a cube.
VOXEL_MM = 2.0
#: Fewer surface points than this say where a few pixels are, not where a part is: none is kept, or followed.
MIN_POINTS = 20
#: The largest a kept part may be, millimetres across and tall: the size the planner world holds a named part to
#: (``perceived.PART_SIZED_MM``). A larger segmentation is a mat or a bin, and a memory that kept it would leave its
#: whole footprint out of the scene check.
PART_SIZED_MM = (250.0, 150.0)
#: The 98th percentile of the taken footprint's heights is read against :data:`TAKEN_TOP_MM`.
_TAKEN_PERCENTILE = 98.0


@dataclass(frozen=True, slots=True, eq=False)
class KeptPart:
    """One part a look grounded or followed: its label as the camera source mapped it, its surface in BASE (thinned to
    :data:`VOXEL_MM`), the median CIE L*a*b* under its mask, its top (90th percentile of the surface's z), its mask's
    pixel count, the middle of its surface (BASE xy, the median of every point, as a new mask's is measured), and where
    its grounding found it (BASE xy), which bounds how far it may creep."""

    label: str
    cloud_base_mm: np.ndarray
    lab: np.ndarray
    top_mm: float
    pixels: int
    centre: tuple[float, float]
    grounded_xy_mm: tuple[float, float]

    @property
    def centre_xy_mm(self) -> np.ndarray:
        """The middle of its surface, BASE xy, as an array."""
        return np.asarray(self.centre, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Accepted:
    """What :meth:`Following.accept` made of the masks: each clipped to its part's footprint and :data:`LEAK_MM`, in the
    boxes' order, where every one passed; ``why`` the first that did not, ``""`` where all did."""

    masks: tuple[np.ndarray, ...] = ()
    why: str = ""


@dataclass(frozen=True, slots=True, eq=False)
class Following:
    """What a kept scene made of a new frame before any mask was cut: the boxes to cut, each with its kept label, in the
    parts' order; or ``why`` the frame is grounded instead. ``said`` is what the frame's route says when every mask is
    accepted. The rest is the frame :meth:`accept` judges the masks on."""

    why: str = ""
    boxes: tuple[tuple[tuple[float, float, float, float], str], ...] = ()
    said: str = ""
    kept: "KeptScene | None" = None
    depth_mm: "np.ndarray | None" = None
    intrinsics: "np.ndarray | None" = None
    camera_to_base: "np.ndarray | None" = None
    #: A later look of the pick that kept the parts (:meth:`KeptScene.projected`): no move and no pixel count of its own
    #: to compare with, the projected top's pixels instead (``expected_px``).
    later: bool = False
    expected_px: tuple[int, ...] = ()

    def accept(self, image_bgr: Any, masks: Sequence[Any]) -> Accepted:
        """Judge the masks SAM2 cut for :attr:`boxes`, in their order, on this frame (``image_bgr``, BGR uint8): every
        check of step 4 (the module's), each mask clipped to its part's footprint, shifted by its move, and
        :data:`LEAK_MM`. All or nothing: the first mask that fails says why, and none is handed on."""
        kept = self.kept
        if self.why or kept is None or self.depth_mm is None or self.intrinsics is None or self.camera_to_base is None:
            return Accepted(why=self.why or "there is no frame to judge the masks on")
        if len(masks) != len(kept.parts):
            return Accepted(why=f"{len(masks)} mask(s) came back for {len(kept.parts)} kept part(s)")
        depth = np.asarray(self.depth_mm, dtype=np.float64)
        clipped: list[np.ndarray] = []
        for index, (part, mask) in enumerate(zip(kept.parts, masks)):
            expected = self.expected_px[index] if self.later and index < len(self.expected_px) else part.pixels
            judged = _judge_mask(part, np.asarray(mask).astype(bool), image_bgr, depth, self.intrinsics,
                                 self.camera_to_base, expected_px=expected, later=self.later,
                                 max_shift_mm=kept.max_shift_mm, max_creep_mm=kept.max_creep_mm)
            if isinstance(judged, str):
                _LOG.info("kept part %d (%r): its mask is not followed: %s", index, part.label, judged)
                return Accepted(why=f"part {index} ({part.label!r}): {judged}")
            clipped.append(judged)
        return Accepted(masks=tuple(clipped))


@dataclass(frozen=True, slots=True, eq=False)
class KeptScene:
    """The parts one look of a task's pick grounded or followed, and the look itself, to find them again.

    ``parts`` are the look's segmentations of the target label (every label where the task names none) that stood
    outside the task's keep-out regions; ``look`` the look's view identity (``rig@look``); ``depth_mm``,
    ``intrinsics`` and ``camera_to_base`` its measured depth, lens and CAMERA to BASE at the shutter, never handed to
    the planner world; ``support_mm`` the support the judged part stood on (the mat); ``prompt`` the phrase grounded
    (another one never follows); ``grounded_at_pick`` the task's pick whose detector grounded them; ``workspace`` the
    cell's ``robot.workspace_limits`` the scene check reads within (``x_min, x_max, y_min, y_max, z_min, z_max``),
    ``None`` for all of the frame; ``max_shift_mm`` and ``max_creep_mm`` the cell's ``follow_parts`` bounds; ``taken``
    the surfaces of the parts taken since the look, whose footprints must read the support.
    """

    parts: tuple[KeptPart, ...]
    look: str
    depth_mm: np.ndarray
    intrinsics: np.ndarray
    camera_to_base: np.ndarray
    support_mm: "float | None" = None
    prompt: str = ""
    grounded_at_pick: int = 0
    workspace: "tuple[float, float, float, float, float, float] | None" = None
    max_shift_mm: float = 10.0
    max_creep_mm: float = 20.0
    taken: tuple[np.ndarray, ...] = ()

    # ---- what a look keeps ----------------------------------------------------------------------------------------

    @classmethod
    def of_view(
        cls, view: Any, *, label: "str | None", regions: Sequence[Any] = (), prompt: str = "", pick: int = 0,
        support_mm: "float | None" = None, workspace: Any = None, max_shift_mm: float = 10.0,
        max_creep_mm: float = 20.0, before: "KeptScene | None" = None,
    ) -> "KeptScene | None":
        """The parts ``view`` (a pick loop's look: its frame, measured depth, CAMERA to BASE, surfaces in BASE and view
        identity) holds, or ``None`` where it keeps none to follow.

        A part is a segmentation of ``label`` (every label where ``label`` is ``None``) whose surface's middle stands
        outside every keep-out region that applies to it: a part placed into the bin or onto the drop is no part to
        pick; nor is a surface the parts lie on, by the pick loop's own rule (``pick_loop.SURFACE_SPAN_MM`` across and
        flatter than ``SURFACE_FLAT_MM``: a mat the detector boxed). None at all, a frame with no colour or no
        placement, and a part that cannot be kept (fewer than :data:`MIN_POINTS` surface points, or larger than
        :data:`PART_SIZED_MM`) keep nothing: a part left out of the memory would leave the task unseen. ``before`` is the
        memory ``view`` followed: each part keeps its grounding (its pick and where it was found), matched in order; a
        part that does not match keeps nothing.
        """
        from src.robot.grasping.loop.pick_loop import SURFACE_FLAT_MM, SURFACE_SPAN_MM  # noqa: PLC0415 - one rule

        frame = getattr(view, "frame", None)
        matrix = getattr(view, "camera_to_base", None)
        depth = getattr(view, "depth", None)
        rgb = getattr(frame, "rgb", None)
        if frame is None or matrix is None or depth is None or rgb is None:
            return None
        matrix = np.asarray(matrix, dtype=np.float64)
        depth = np.asarray(depth, dtype=np.float64)
        segmentations = tuple(getattr(frame, "segmentations", ()) or ())
        clouds = tuple(getattr(view, "clouds", ()) or ())
        parts: list[KeptPart] = []
        for index, seg in enumerate(segmentations):
            said = str(getattr(seg, "label", "") or "")
            if label is not None and said != label:
                continue
            mask = getattr(seg, "mask", None)
            cloud = np.asarray(clouds[index], dtype=np.float64).reshape(-1, 3) if index < len(clouds) else None
            if mask is None or cloud is None:
                return _nothing_kept(f"segmentation {index} ({said!r}) has no surface to keep")
            cloud = cloud[np.all(np.isfinite(cloud), axis=1)]
            if cloud.shape[0] < MIN_POINTS:
                return _nothing_kept(f"segmentation {index} ({said!r}) has {cloud.shape[0]} surface point(s), fewer "
                                     f"than {MIN_POINTS}")
            centre = np.median(cloud[:, :2], axis=0)
            if any(_applies(region, said) and region.contains((float(centre[0]), float(centre[1])))
                   for region in regions):
                continue
            # Across its 5th to 95th percentile, as the pick loop measures a surface that is not a part.
            low, high = np.percentile(cloud, 5.0, axis=0), np.percentile(cloud, 95.0, axis=0)
            span = high[:2] - low[:2]
            flat = float(np.percentile(cloud[:, 2], 90.0) - np.percentile(cloud[:, 2], 10.0))
            if float(np.linalg.norm(span)) > SURFACE_SPAN_MM and flat < SURFACE_FLAT_MM:
                _LOG.info("segmentation %d (%r) is a surface the parts lie on, %.0f mm across: no part to keep", index,
                          said, float(np.linalg.norm(span)))
                continue
            if float(span.max()) > PART_SIZED_MM[0] or float(high[2] - low[2]) > PART_SIZED_MM[1]:
                return _nothing_kept(f"segmentation {index} ({said!r}) spans {float(span.max()):.0f} mm, larger than a "
                                     "part")
            selected = np.asarray(mask).astype(bool)
            rows, cols = np.nonzero(selected)
            lab = _lab_at(np.asarray(rgb), rows, cols, order="rgb")
            if lab.shape[0] < MIN_POINTS:
                return _nothing_kept(f"segmentation {index} ({said!r}) has no colour to keep")
            middle = (float(centre[0]), float(centre[1]))
            parts.append(KeptPart(
                label=said, cloud_base_mm=_voxel(cloud, VOXEL_MM), lab=np.median(lab, axis=0),
                top_mm=float(np.percentile(cloud[:, 2], TOP_PERCENTILE)), pixels=int(rows.size), centre=middle,
                grounded_xy_mm=middle))
        if not parts:
            return None
        grounded_at = int(pick)
        if before is not None:
            carried = _carried_over(parts, before)
            if carried is None:
                return _nothing_kept("the followed parts do not match the parts they were followed from")
            parts, grounded_at = carried, before.grounded_at_pick
        box = None
        if workspace is not None:
            box = tuple(float(getattr(workspace, name)) for name in ("x_min", "x_max", "y_min", "y_max", "z_min",
                                                                       "z_max"))
        return cls(parts=tuple(parts), look=str(getattr(view, "name", "") or ""),
                   depth_mm=np.asarray(depth, dtype=np.float32),
                   intrinsics=np.asarray(frame.intrinsics, dtype=np.float64),
                   camera_to_base=matrix, support_mm=None if support_mm is None else float(support_mm),
                   prompt=str(prompt), grounded_at_pick=grounded_at, workspace=box,  # type: ignore[arg-type]
                   max_shift_mm=float(max_shift_mm), max_creep_mm=float(max_creep_mm))

    @classmethod
    def of_look(cls, looked: Any, **keywords: Any) -> "KeptScene | None":
        """:meth:`of_view` of the first look of ``looked`` (a pick's ``LookedAround``), the look a next pick's first
        look stands at; ``before`` counts only where that look was followed from it (``looked.followed``)."""
        views = tuple(getattr(looked, "views", ()) or ())
        if not views:
            return None
        if not getattr(looked, "followed", False):
            keywords["before"] = None
        return cls.of_view(views[0], **keywords)

    def without(self, taken_cloud_base_mm: Any) -> "KeptScene | None":
        """This memory less the part a pick gripped, found by ``taken_cloud_base_mm`` (its cloud, BASE): the kept part
        whose middle stands nearest that cloud's, within :data:`NEAREST_TAKEN_MM`; its surface is kept as taken, whose
        footprint the next look reads. ``None`` where no kept part stands so near: the next pick grounds again."""
        cloud = np.asarray(taken_cloud_base_mm if taken_cloud_base_mm is not None else np.zeros((0, 3)),
                           dtype=np.float64).reshape(-1, 3)
        cloud = cloud[np.all(np.isfinite(cloud), axis=1)]
        if not cloud.shape[0] or not self.parts:
            return None
        centre = np.median(cloud[:, :2], axis=0)
        apart = [float(np.hypot(*(part.centre_xy_mm - centre))) for part in self.parts]
        nearest = int(np.argmin(apart))
        if apart[nearest] > NEAREST_TAKEN_MM:
            _LOG.info("the gripped part stood %.1f mm from the nearest kept part (at most %g): it is not among them",
                      apart[nearest], NEAREST_TAKEN_MM)
            return None
        part = self.parts[nearest]
        return replace(self, parts=self.parts[:nearest] + self.parts[nearest + 1:],
                       taken=(*self.taken, part.cloud_base_mm))

    # ---- the next pick's first look -------------------------------------------------------------------------------

    def following(
        self, depth_mm: Any, image_bgr: Any, intrinsics: Any, camera_to_base: Any, *, look: str, prompt: str,
        regions: Sequence[Any] = (),
    ) -> Following:
        """Steps 1 and 2 of the module's rule on a new frame of the look ``look``, grounded for ``prompt``: the boxes
        SAM2 cuts, or why the frame is grounded instead. ``regions`` are the task's keep-out regions now."""
        if not self.parts:
            return _grounded("no part is kept: the end of a task is always asked of the detector")
        if prompt != self.prompt:
            return _grounded(f"the phrase is {prompt!r}, and the parts were grounded for {self.prompt!r}")
        if look != self.look:
            return _grounded(f"the first look is {look!r}, and the parts were kept from {self.look!r}")
        if self.support_mm is None:
            return _grounded("the support the parts stand on is not known")
        depth = np.asarray(depth_mm, dtype=np.float64)
        lens = np.asarray(intrinsics, dtype=np.float64)
        matrix = np.asarray(camera_to_base, dtype=np.float64)
        if depth.shape != self.depth_mm.shape or lens.shape != (3, 3) or not np.allclose(lens, self.intrinsics):
            return _grounded("the frame is not of the camera the parts were kept with (another size or lens)")
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            return _grounded("the frame cannot be placed in BASE")
        changed = self._scene_changed(depth, lens, matrix, regions)
        if changed:
            return _grounded(changed)
        boxes: list[tuple[tuple[float, float, float, float], str]] = []
        for index, part in enumerate(self.parts):
            box = self._where_it_stands(index, part, depth, image_bgr, lens, matrix)
            if isinstance(box, str):
                return _grounded(f"part {index} ({part.label!r}): {box}")
            boxes.append((box, part.label))
        return Following(boxes=tuple(boxes), said=f"{len(boxes)} part(s) followed, nothing new in depth", kept=self,
                         depth_mm=depth, intrinsics=lens, camera_to_base=matrix)

    def _scene_changed(self, depth: np.ndarray, lens: np.ndarray, matrix: np.ndarray, regions: Sequence[Any]) -> str:
        """Step 1: ``""`` where nothing changed in depth between the kept frame and this one, both ways, else why."""
        import cv2  # noqa: PLC0415 - deferred: OpenCV is no small import, and only a followed look needs it here

        kept_depth = np.asarray(self.depth_mm, dtype=np.float64)
        old, _, _ = _frame_points(kept_depth, self.intrinsics, self.camera_to_base, SCENE_STRIDE_PX)
        new, rows, cols = _frame_points(depth, lens, matrix, SCENE_STRIDE_PX)
        within = self._within(new)
        taken = _cloud_xy(self.taken)
        if taken is not None:
            # The taken part's spot reads the support: a part that stood under it would stand out.
            spot = within & _near(taken, new[:, :2], FOOTPRINT_MM)
            if int(spot.sum()) < MIN_POINTS:
                return (f"the taken part's spot reads {int(spot.sum())} point(s) of depth, fewer than {MIN_POINTS}, so "
                        "nothing says it is empty")
            assert self.support_mm is not None  # checked by following()
            high = float(np.percentile(new[spot, 2], _TAKEN_PERCENTILE)) - self.support_mm
            if high > TAKEN_TOP_MM:
                return (f"the taken part's spot reads {high:.1f} mm over the support (at most {TAKEN_TOP_MM:g}): "
                        "something stands where it stood")
        old = old[self._within(old) & self._judged_by_depth(old[:, :2], regions)]
        judged = within & self._judged_by_depth(new[:, :2], regions)
        new, rows, cols = new[judged], rows[judged], cols[judged]
        changed = np.zeros(depth.shape, dtype=np.uint8)
        # What the kept frame saw, read in this one: off by more than CHANGED_MM where this frame measured a depth.
        r, c, z, inside = _projected(old, lens, matrix, depth.shape)
        r, c, z = r[inside], c[inside], z[inside]
        read = depth[r, c]
        with np.errstate(invalid="ignore"):
            off = np.isfinite(read) & (read > 0.0) & (np.abs(read - z) > CHANGED_MM)
        changed[r[off], c[off]] = 1
        # What this frame sees, read in the kept one: new over the support where the kept frame read a hole or farther.
        r, c, z, inside = _projected(new, self.intrinsics, self.camera_to_base, kept_depth.shape)
        read = kept_depth[r[inside], c[inside]]
        with np.errstate(invalid="ignore"):
            hole = ~(np.isfinite(read) & (read > 0.0))
            farther = hole | (read - z[inside] > CHANGED_MM)
        assert self.support_mm is not None  # checked by following()
        came = farther & (new[inside, 2] > self.support_mm + OVER_SUPPORT_MM)
        changed[rows[inside][came], cols[inside][came]] = 1
        # The points sit on every SCENE_STRIDE_PX-th pixel: close the grid, then open away single stray readings.
        grown = cv2.dilate(changed, np.ones((SCENE_STRIDE_PX + 1, SCENE_STRIDE_PX + 1), np.uint8))
        opened = cv2.morphologyEx(grown, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        count, _, stats, centres = cv2.connectedComponentsWithStats(opened, connectivity=8)
        limit = min(CHANGED_CLUSTER_PX, 0.5 * min(part.pixels for part in self.parts))
        areas = [(int(stats[label, cv2.CC_STAT_AREA]), centres[label]) for label in range(1, count)]
        largest = max(areas, key=lambda item: item[0]) if areas else (0, (0.0, 0.0))
        _LOG.info("scene check against look %s: %d and %d point(s) read both ways, the largest changed cluster %d px "
                  "(at most %.0f)", self.look, old.shape[0], new.shape[0], largest[0], limit)
        if largest[0] >= limit:
            return (f"something changed in depth: {largest[0]} px about pixel ({largest[1][0]:.0f}, "
                    f"{largest[1][1]:.0f}) (from {limit:.0f} px)")
        return ""

    def _within(self, points: np.ndarray) -> np.ndarray:
        """Which of ``points`` (BASE) stand inside the cell's workspace, all where the memory names none."""
        if self.workspace is None or not points.shape[0]:
            return np.ones(points.shape[0], dtype=bool)
        x0, x1, y0, y1, z0, z1 = self.workspace
        return ((points[:, 0] >= x0) & (points[:, 0] <= x1) & (points[:, 1] >= y0) & (points[:, 1] <= y1)
                & (points[:, 2] >= z0) & (points[:, 2] <= z1))

    def _judged_by_depth(self, xy: np.ndarray, regions: Sequence[Any]) -> np.ndarray:
        """Which of ``xy`` (BASE) the scene check judges: none in a keep-out region (an expected change), near a kept
        part's footprint (its own check judges it) or near a taken part's (it reads the support)."""
        judged = ~_in_regions(xy, regions)
        parts = _cloud_xy(tuple(part.cloud_base_mm for part in self.parts))
        if parts is not None:
            judged &= ~_near(parts, xy, self.max_shift_mm + FOOTPRINT_GROWTH_MM)
        taken = _cloud_xy(self.taken)
        if taken is not None:
            judged &= ~_near(taken, xy, FOOTPRINT_GROWTH_MM)
        return judged

    def _where_it_stands(
        self, index: int, part: KeptPart, depth: np.ndarray, image_bgr: Any, lens: np.ndarray, matrix: np.ndarray,
    ) -> "tuple[float, float, float, float] | str":
        """Step 2 for one part: its box, or why the frame is grounded."""
        rows, cols, z, inside = _projected(part.cloud_base_mm, lens, matrix, depth.shape)
        total = int(part.cloud_base_mm.shape[0])
        nearest = _nearest_per_pixel(rows[inside], cols[inside], z[inside], depth.shape[1])
        r, c, e = rows[inside][nearest], cols[inside][nearest], z[inside][nearest]
        read = depth[r, c]
        with np.errstate(invalid="ignore"):
            has = np.isfinite(read) & (read > 0.0)
        seen = (int(inside.sum()) / total) * (float(has.mean()) if has.size else 0.0)
        gap = read[has] - e[has]
        agree = float(np.mean(np.abs(gap) <= AGREE_MM)) if gap.size else 0.0
        gone = float(np.mean(gap > GONE_MM)) if gap.size else 0.0
        delta_e = math.inf
        agreeing = has.copy()
        agreeing[has] = np.abs(gap) <= AGREE_MM
        if int(agreeing.sum()) >= MIN_POINTS:
            lab = _lab_at(np.asarray(image_bgr), r[agreeing], c[agreeing], order="bgr")
            delta_e = float(np.linalg.norm(np.median(lab, axis=0) - part.lab))
        if seen < SEEN_SHARE:
            why = f"only {seen:.0%} of it was seen with depth (at least {SEEN_SHARE:.0%})"
        elif gone > GONE_SHARE:
            why = f"{gone:.0%} of it reads more than {GONE_MM:g} mm farther: it is gone"
        elif agree >= AGREE_SHARE and delta_e > DELTA_E:
            why = (f"it reads where it stood and its colour stands dE {delta_e:.1f} from the one kept (at most "
                   f"{DELTA_E:g})")
        else:
            why = ""
        stands = not why and agree >= AGREE_SHARE
        _LOG.info("kept part %d (%r): %.0f %% seen, %.0f %% within %g mm, %.0f %% farther by %g mm, dE %.1f: %s", index,
                  part.label, 100.0 * seen, 100.0 * agree, AGREE_MM, 100.0 * gone, GONE_MM, delta_e,
                  why or ("it stands where it stood" if stands else "it may have moved: its mask measures how far"))
        if why:
            return why
        pad = float(BOX_PAD_PX)
        if not stands:
            # Boxed wider by the move it may have made, at its depth: about 13 px for 10 mm at 720 mm.
            pad += self.max_shift_mm * float(lens[0, 0]) / max(float(np.median(e)), 1.0)
        return _box(r, c, pad, depth.shape)

    # ---- a later look of the same pick (map2's F) -----------------------------------------------------------------

    def projected(self, depth_mm: Any, image_bgr: Any, intrinsics: Any, camera_to_base: Any) -> Following:
        """Every kept part's box in a later look of the pick that kept them, ``image_bgr`` unread: its surface
        projected at this look's stamped pose, padded :data:`BOX_PAD_PX`; or why the look is grounded. A part less than
        :data:`SEEN_SHARE` of whose surface falls in this frame grounds it: the detector would say what this look
        sees."""
        if not self.parts:
            return _grounded("the first look kept no part")
        depth = np.asarray(depth_mm, dtype=np.float64)
        lens = np.asarray(intrinsics, dtype=np.float64)
        matrix = np.asarray(camera_to_base, dtype=np.float64)
        if lens.shape != (3, 3) or matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            return _grounded("the frame cannot be placed in BASE")
        boxes: list[tuple[tuple[float, float, float, float], str]] = []
        expected: list[int] = []
        for index, part in enumerate(self.parts):
            rows, cols, z, inside = _projected(part.cloud_base_mm, lens, matrix, depth.shape)
            share = float(inside.mean()) if inside.size else 0.0
            if share < SEEN_SHARE:
                return _grounded(f"part {index} ({part.label!r}): only {share:.0%} of it falls in this look's frame")
            nearest = _nearest_per_pixel(rows[inside], cols[inside], z[inside], depth.shape[1])
            r, c = rows[inside][nearest], cols[inside][nearest]
            boxes.append((_box(r, c, float(BOX_PAD_PX), depth.shape), part.label))
            # The pixels its kept surface covers in this look: the outline of its thinned points, which land a few
            # pixels apart, filled.
            expected.append(_outline_px(r, c))
        return Following(boxes=tuple(boxes), said=f"{len(boxes)} part(s) of the first look followed by their projected "
                         "boxes", kept=self, depth_mm=depth, intrinsics=lens, camera_to_base=matrix, later=True,
                         expected_px=tuple(expected))


# ---------------------------------------------------------------------------------------------------------------------
# One mask
# ---------------------------------------------------------------------------------------------------------------------


def _judge_mask(
    part: KeptPart, mask: np.ndarray, image_bgr: Any, depth: np.ndarray, lens: Any, matrix: Any, *, expected_px: int,
    later: bool, max_shift_mm: float, max_creep_mm: float,
) -> "np.ndarray | str":
    """Step 4 for one mask: the mask clipped to the part's footprint, shifted by its move, and :data:`LEAK_MM`; or why
    it is not followed. On a later look of the same pick the part has not moved, and the mask is held to the projected
    top's pixels from below only (its sides may show too)."""
    from src.robot.grasping.generation.depth_steps import pixels_behind_depth_steps  # noqa: PLC0415
    from src.robot.grasping.multiview.scene_geometry import grazing_pixels, to_base_mm  # noqa: PLC0415

    pixels = int(mask.sum())
    ratio = pixels / max(int(expected_px), 1)
    if ratio < AREA_RATIO[0] or (not later and ratio > AREA_RATIO[1]):
        bounds = f"{AREA_RATIO[0]:g} to {AREA_RATIO[1]:g}" if not later else f"at least {AREA_RATIO[0]:g}"
        return f"its mask holds {pixels} px, {ratio:.2f} of the {expected_px} kept ({bounds})"
    lens = np.asarray(lens, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    # Its surface as a look keeps one (the pick loop's ``_look_view``): less the pixels behind a depth step, then less
    # the grazing ones of what is left, so a mask that did not change measures where its part stood.
    surface = mask & ~pixels_behind_depth_steps(mask, depth)
    surface &= ~grazing_pixels(surface, depth, lens)
    cloud = to_base_mm(surface, depth, lens, matrix)
    if cloud.shape[0] < MIN_POINTS:
        return f"its mask holds {cloud.shape[0]} surface point(s), fewer than {MIN_POINTS}"
    centre = np.median(cloud[:, :2], axis=0)
    move = np.zeros(2) if later else centre - part.centre_xy_mm
    shift = float(np.hypot(*move))
    creep = float(np.hypot(centre[0] - part.grounded_xy_mm[0], centre[1] - part.grounded_xy_mm[1]))
    footprint = part.cloud_base_mm[:, :2] + move
    outside = ~_near(footprint, cloud[:, :2], LEAK_MM)
    leak = float(outside.mean())
    top = float(np.percentile(cloud[:, 2], TOP_PERCENTILE)) - part.top_mm
    rows, cols = np.nonzero(mask)
    lab = _lab_at(np.asarray(image_bgr), rows, cols, order="bgr")
    delta_e = float(np.linalg.norm(np.median(lab, axis=0) - part.lab)) if lab.shape[0] else math.inf
    if not later and shift > max_shift_mm:
        why = f"it moved {shift:.1f} mm (at most {max_shift_mm:g})"
    elif not later and creep > max_creep_mm:
        why = f"it crept {creep:.1f} mm since its grounding (at most {max_creep_mm:g})"
    elif leak > LEAK_SHARE:
        why = (f"{leak:.1%} of its surface lies more than {LEAK_MM:g} mm past the part's footprint (at most "
               f"{LEAK_SHARE:.0%})")
    elif abs(top) > TOP_MM:
        why = f"its top stands {top:+.1f} mm from the part's (within {TOP_MM:g})"
    elif delta_e > DELTA_E:
        why = f"its colour stands dE {delta_e:.1f} from the part's (at most {DELTA_E:g})"
    else:
        why = ""
    _LOG.info("kept part %r: mask %d px (%.2f), moved %.1f mm, crept %.1f mm, %.1f %% past %g mm, top %+.1f mm, "
              "dE %.1f: %s", part.label, pixels, ratio, shift, creep, 100.0 * leak, LEAK_MM, top, delta_e,
              why or "followed")
    if why:
        return why
    # Clipped to the footprint: no pixel the planner world may leave out stands past the part by more than LEAK_MM.
    with np.errstate(invalid="ignore"):
        has = np.isfinite(depth[rows, cols]) & (depth[rows, cols] > 0.0)
    clipped = mask.copy()
    if has.any():
        points = _pixels_in_base(rows[has], cols[has], depth[rows[has], cols[has]], lens, matrix)
        far = ~_near(footprint, points[:, :2], LEAK_MM)
        clipped[rows[has][far], cols[has][far]] = False
    return clipped


# ---------------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------------


def _grounded(why: str) -> Following:
    """A frame the detector grounds, and why."""
    return Following(why=why)


def _nothing_kept(why: str) -> "KeptScene | None":
    """No memory, and why, said."""
    _LOG.info("no part is kept to follow: %s", why)
    return None


def _applies(region: Any, label: str) -> bool:
    applies = getattr(region, "applies", None)
    return bool(applies(label)) if callable(applies) else True


def _carried_over(parts: list[KeptPart], before: KeptScene) -> "list[KeptPart] | None":
    """``parts`` with the grounding of the parts of ``before`` they were followed from, in order, or ``None`` where they
    do not match: another count, another label, or a middle further than ``max_shift_mm`` from the one followed."""
    if len(parts) != len(before.parts):
        return None
    carried: list[KeptPart] = []
    for part, earlier in zip(parts, before.parts):
        apart = float(np.hypot(*(part.centre_xy_mm - earlier.centre_xy_mm)))
        if part.label != earlier.label or apart > before.max_shift_mm + VOXEL_MM:
            return None
        carried.append(replace(part, grounded_xy_mm=earlier.grounded_xy_mm))
    return carried


def _voxel(points: np.ndarray, size: float) -> np.ndarray:
    """``points`` thinned to one per ``size`` voxel, the first of each, in their order."""
    if not points.shape[0]:
        return points
    _, first = np.unique(np.floor(points / size).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(first)]


def _frame_points(
    depth: np.ndarray, lens: np.ndarray, matrix: np.ndarray, stride: int,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Every ``stride``-th pixel's point of a frame each way, BASE, where it measured a depth, with its row and
    column."""
    height, width = depth.shape[:2]
    rows, cols = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[::stride, ::stride]
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(z) & (z > 0.0)
    rows, cols, z = rows[valid], cols[valid], z[valid]
    return _pixels_in_base(rows, cols, z, lens, matrix), rows, cols


def _pixels_in_base(
    rows: np.ndarray, cols: np.ndarray, z: np.ndarray, lens: np.ndarray, matrix: np.ndarray,
) -> np.ndarray:
    """Pixels with their depth in BASE, a pixel's centre at its whole index, as ``masked_points`` places them."""
    x = (cols.astype(np.float64) - lens[0, 2]) * z / lens[0, 0]
    y = (rows.astype(np.float64) - lens[1, 2]) * z / lens[1, 1]
    return np.column_stack([x, y, z]) @ matrix[:3, :3].T + matrix[:3, 3]


def _projected(
    points: np.ndarray, lens: np.ndarray, matrix: np.ndarray, shape: tuple[int, ...],
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """BASE points onto a frame's pixels, by ``locator._measured``'s rule (the inverse of how a pixel is placed): rows
    and columns (0 where a point lands outside), each point's CAMERA z, and which land in the frame in front of the
    camera."""
    in_camera = (np.asarray(points, dtype=np.float64).reshape(-1, 3) - matrix[:3, 3]) @ matrix[:3, :3]
    z = in_camera[:, 2]
    ahead = np.isfinite(z) & (z > 0.0)
    safe = np.where(ahead, z, 1.0)
    cols = np.rint(lens[0, 0] * in_camera[:, 0] / safe + lens[0, 2])
    rows = np.rint(lens[1, 1] * in_camera[:, 1] / safe + lens[1, 2])
    height, width = shape[:2]
    inside = (ahead & np.isfinite(cols) & np.isfinite(rows) & (cols >= 0) & (cols < width) & (rows >= 0)
              & (rows < height))
    return np.where(inside, rows, 0).astype(np.int64), np.where(inside, cols, 0).astype(np.int64), z, inside


def _nearest_per_pixel(rows: np.ndarray, cols: np.ndarray, z: np.ndarray, width: int) -> np.ndarray:
    """Of the points landing on one pixel, the nearest: their indices, one per pixel (the z-buffer)."""
    if not rows.size:
        return np.zeros(0, dtype=np.int64)
    pixel = rows.astype(np.int64) * int(width) + cols.astype(np.int64)
    order = np.lexsort((z, pixel))
    first = np.ones(order.size, dtype=bool)
    first[1:] = pixel[order][1:] != pixel[order][:-1]
    return order[first]


def _outline_px(rows: np.ndarray, cols: np.ndarray) -> int:
    """How many pixels the outline of the pixels ``rows``, ``cols`` holds: their convex hull's area, the pixels
    themselves where they are too few to have one."""
    import cv2  # noqa: PLC0415 - deferred, as in colour_check

    if rows.size < 3:
        return int(rows.size)
    hull = cv2.convexHull(np.column_stack([cols, rows]).astype(np.int32))
    return max(int(round(cv2.contourArea(hull))), int(rows.size))


def _box(rows: np.ndarray, cols: np.ndarray, pad: float, shape: tuple[int, ...]) -> tuple[float, float, float, float]:
    """The pixels' bounding box padded by ``pad``, inside the frame: ``(x0, y0, x1, y1)``."""
    height, width = shape[:2]
    x0 = max(0.0, float(cols.min()) - pad)
    y0 = max(0.0, float(rows.min()) - pad)
    x1 = min(float(width), float(cols.max()) + 1.0 + pad)
    y1 = min(float(height), float(rows.max()) + 1.0 + pad)
    return (x0, y0, x1, y1)


def _lab_at(image: np.ndarray, rows: np.ndarray, cols: np.ndarray, *, order: str) -> np.ndarray:
    """CIE L*a*b* of ``image``'s pixels at ``rows``, ``cols`` (uint8, ``order`` ``"bgr"`` or ``"rgb"``), as the colour
    check reads a part's pixels: OpenCV's 8-bit conversion, L* 0 to 100, a* and b* about 0."""
    import cv2  # noqa: PLC0415 - deferred, as in colour_check

    if not rows.size or image.ndim != 3:
        return np.zeros((0, 3), dtype=np.float64)
    pixels = np.ascontiguousarray(image[rows, cols, :3], dtype=np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(pixels, cv2.COLOR_BGR2LAB if order == "bgr" else cv2.COLOR_RGB2LAB)
    lab = lab.reshape(-1, 3).astype(np.float64)
    return np.column_stack([lab[:, 0] * (100.0 / 255.0), lab[:, 1] - 128.0, lab[:, 2] - 128.0])


def _cloud_xy(clouds: Sequence[np.ndarray]) -> "np.ndarray | None":
    """The clouds' points in BASE xy, together, or ``None`` for none."""
    stacked = [np.asarray(cloud, dtype=np.float64).reshape(-1, 3)[:, :2] for cloud in clouds if np.asarray(cloud).size]
    return np.vstack(stacked) if stacked else None


def _near(reference_xy: np.ndarray, xy: np.ndarray, reach_mm: float) -> np.ndarray:
    """Which of ``xy`` stand within ``reach_mm`` of a point of ``reference_xy`` (both BASE xy).

    Only the points inside the reference's bounding box grown by the reach are asked the tree: a point outside it stands
    further than the reach from every reference point, and the tree would only have said so (a frame's 100 000 points
    against a few parts' footprints: 13 ms a query on the desk, most of it for points nowhere near)."""
    from scipy.spatial import cKDTree  # noqa: PLC0415 - deferred: only a followed look pays for it

    near = np.zeros(xy.shape[0], dtype=bool)
    if not xy.shape[0] or not reference_xy.shape[0]:
        return near
    reach = float(reach_mm)
    low, high = reference_xy.min(axis=0) - reach, reference_xy.max(axis=0) + reach
    asked = np.nonzero(np.all((xy >= low) & (xy <= high), axis=1))[0]
    if asked.size:
        distance, _ = cKDTree(reference_xy).query(xy[asked], k=1, distance_upper_bound=reach)
        near[asked] = np.isfinite(distance)
    return near


def _in_regions(xy: np.ndarray, regions: Sequence[Any]) -> np.ndarray:
    """Which of ``xy`` stand in one of ``regions`` (``ExclusionRegion``: a circle or a turned rectangle about BASE Z, by
    its own rule), whatever label they keep out."""
    inside = np.zeros(xy.shape[0], dtype=bool)
    for region in regions:
        cx, cy = (float(value) for value in region.centre_xy_mm)
        dx, dy = xy[:, 0] - cx, xy[:, 1] - cy
        radius = getattr(region, "radius_mm", None)
        if radius is not None:
            inside |= np.hypot(dx, dy) <= float(radius)
            continue
        size = region.size_mm
        c, s = math.cos(float(region.yaw_rad)), math.sin(float(region.yaw_rad))
        along, across = c * dx + s * dy, -s * dx + c * dy
        inside |= (np.abs(along) <= float(size[0]) / 2.0) & (np.abs(across) <= float(size[1]) / 2.0)
    return inside
