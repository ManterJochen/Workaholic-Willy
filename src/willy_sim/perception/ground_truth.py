"""Ground-truth perception source for the Isaac validation platform.

``GroundTruthPerceptionSource`` satisfies the grasping stack's ``PerceptionSource`` Protocol
(``acquire() -> PerceptionFrame``) by reading the Isaac camera's ground-truth instance
segmentation for a known object prim, plus the rendered depth and intrinsics. The vision models,
GroundingDINO and SAM2, never run, so a failure here is a motion failure and not a perception
one; the geometry pipeline from depth and mask to a grasp still runs end to end.

This module imports no ``isaacsim``. The camera, an ``isaacsim.sensors.camera.Camera``, is
injected by the runner or the scene, so the module imports on a machine without Isaac.

The contract:
* depth is emitted in millimetres, because the orchestrator calls ``GraspCalculator.compute``
  with the default ``unit="mm"``; the camera's ``distance_to_image_plane`` reports metres and is
  converted here.
* the segmentation ``mask`` has the same HxW shape as the depth map; both come from the camera.
* the object mask comes from ``instance_id_segmentation``: a per-pixel prim-path id plus the
  ``info.idToLabels`` table mapping each id to a prim path.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from src.robot.drivers.sim.adapter import metres_to_millimetres
from src.robot.grasping.types.perception import PerceptionFrame
from src.utility.log_cfg import create_logger
from src.willy_sim.constants import PERCEPTION_GT_LOG_FILE, WILLY_SIM_LOG_DIR

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.grasping.generation.calculator import SegmentationLike

__all__ = [
    "GroundTruthSegmentation",
    "GroundTruthPerceptionSource",
    "object_mask_from_frame",
]


@dataclass(frozen=True)
class GroundTruthSegmentation:
    """A boolean object mask satisfying the grasping stack's ``SegmentationLike`` (``.mask``).

    ``label``: a human/prompt-facing name for the object (e.g. ``"red_cube"``) used by the
    label-aware target selection so a prompt can pick this object among several. Empty by default.
    """

    mask: np.ndarray
    prim_path: str = ""
    label: str = ""


#: Extra render pumps allowed when the annotators come back with nothing. Bounded on purpose: an
#: annotator that is genuinely unattached never fills, and an unbounded wait turns a crash into a
#: hang that reads as a slow run.
_BUFFER_RETRIES = 8


def _buffers_usable(depth_mm: Any, mask: Any) -> bool:
    """Report whether the annotators produced a frame on this read.

    An unticked annotator returns an empty array, which numpy makes 1-D, so the tell is the shape
    and not the content. A blank but 2-D depth or an all-False mask is a legitimate answer,
    meaning nothing in view, and must pass. ``mask is None`` likewise means the target was not
    segmented this frame, which the caller already handles.
    """
    depth = np.asarray(depth_mm)
    if depth.ndim != 2 or depth.size == 0:
        return False
    return mask is None or np.asarray(mask).ndim == 2


def object_mask_from_frame(
    frame: Mapping[str, Any], target_prim_path: str
) -> np.ndarray | None:
    """Extract the boolean mask for ``target_prim_path`` from an ``instance_id_segmentation`` frame.

    Returns ``None`` if the frame has no instance-id segmentation or the prim is not visible
    (the caller substitutes an empty mask).
    """
    iid = frame.get("instance_id_segmentation")
    if not isinstance(iid, Mapping):
        return None
    data = iid.get("data")
    info = iid.get("info") or {}
    labels = info.get("idToLabels") or {}
    if data is None:
        return None
    # Match the prim and its subtree: a DynamicCuboid carries the instance-id on the prim itself,
    # an exact match, but a referenced YCB USD carries it on child mesh prims, for example
    # ``/World/Object_0/_05_tomato_soup_can/.../mesh``. So any id whose label is at or under
    # ``target_prim_path`` matches, and their pixels are unioned. A cube still matches on one id.
    prefix = target_prim_path.rstrip("/") + "/"
    matching_ids = [
        int(k) for k, v in labels.items()
        if str(v) == target_prim_path or str(v).startswith(prefix)
    ]
    if not matching_ids:
        return None
    return np.isin(np.asarray(data), matching_ids)


class GroundTruthPerceptionSource:
    """``PerceptionSource`` that emits a :class:`PerceptionFrame` from Isaac ground truth.

    Parameters
    ----------
    camera
        A live ``isaacsim.sensors.camera.Camera`` (injected; this module never imports Isaac).
    target_prim_path
        Prim path of the object to grasp (its instance-id mask becomes the segmentation).
    session
        Optional :class:`IsaacSimSession`; when given, ``acquire`` warms up the render by
        stepping it (so the camera has a fresh frame).
    warmup_steps
        Render steps to take before each read (the first frames after a change are stale).
    """

    def __init__(
        self,
        *,
        camera: Any,
        target_prim_path: str,
        session: Any | None = None,
        warmup_steps: int = 2,
        ground_truth_depth: bool = False,
        grasp_lift_mm: float = 0.0,
    ) -> None:
        self._camera = camera
        self._target = target_prim_path
        self._session = session
        self._warmup = max(0, warmup_steps)
        # When True, override the rendered depth over the object mask with the true distance from
        # the camera to the object's centre, read from its known prim pose. The rendered
        # distance_to_image_plane on a short object can read the support plane, which would place
        # the grasp at the object base; the centre depth gives a solid mid-body grip.
        self._ground_truth_depth = ground_truth_depth
        # Lift the ground-truth grasp point this many mm above the object centre (shrinks the
        # masked depth). A long parallel jaw must grip the upper body so its fingertips clear the
        # table: pad centre = grasp point, fingertips ~28 mm below it, so an object whose centre is
        # below the fingertip standoff needs the grasp raised to keep the tips off the tabletop.
        self._grasp_lift_mm = float(grasp_lift_mm)
        # Both ground-truth sources log to one file under their own logger names: a dense run
        # swaps the single-object source for the multi-object one, and the swap then reads as one
        # sequence. The log format carries the logger name.
        self._log = create_logger(
            "GroundTruthPerceptionSource", log_file=PERCEPTION_GT_LOG_FILE, log_dir=WILLY_SIM_LOG_DIR,
        )
        self._log.info(
            "ground-truth source ready: target=%s warmup=%d ground_truth_depth=%s lift=%.1f mm",
            target_prim_path, self._warmup, self._ground_truth_depth, self._grasp_lift_mm,
        )
        # Attach the annotators this source reads; repeated calls on one camera are harmless.
        camera.add_distance_to_image_plane_to_frame()
        camera.add_instance_id_segmentation_to_frame()

    def _read_buffers(self):  # noqa: ANN202 ((depth_mm, frame, mask) straight off the annotators)
        depth_mm = metres_to_millimetres(np.asarray(self._camera.get_depth(), dtype=np.float64))
        # distance_to_image_plane sets background pixels to far-clip / +inf; zero them so the
        # calculator (which is mask-restricted anyway) never back-projects a non-finite depth.
        depth_mm = np.where(np.isfinite(depth_mm), depth_mm, 0.0)
        frame = self._camera.get_current_frame()
        return depth_mm, frame, object_mask_from_frame(frame, self._target)

    def acquire(self) -> PerceptionFrame:
        started = time.perf_counter()
        # Pump both world.step(render=True) and app.update() so the depth and instance-id
        # segmentation annotators flush on every acquire. render=True on its own leaves the
        # annotators stale on a second consecutive acquire, which yields degenerate depth, no
        # graspable points and no candidates. A mode that acquires more than once per pick hits
        # that; a single acquire per pick does not. IsaacVisionPerceptionSource pumps the same way.
        app = getattr(self._session, "app", None)
        for _ in range(self._warmup):
            if self._session is not None:
                self._session.step(render=True)
                if app is not None:
                    app.update()

        # Check the buffers, do not assume them. An annotator that has not produced a frame returns
        # an empty array rather than a blank image, and the failure surfaces three frames away as a
        # shape error about something else: "Expected 2-D mask, got shape (0,)" out of
        # mask_analyzer, or "<3 valid depth pixels ... ndim=1" out of the ground-truth camera fit.
        # The process then hangs rather than exiting, so the run looks slow instead of broken.
        #
        # Raising the warmup does not fix it: render_warmup_steps is already 20 on this path and
        # the buffer still comes back empty. So this re-reads only while the buffers are unusable
        # and reports how many extra pumps it needed. A fixed extra count is what fails
        # intermittently.
        depth_mm, frame, mask = self._read_buffers()
        for extra in range(1, _BUFFER_RETRIES + 1):
            if _buffers_usable(depth_mm, mask):
                break
            if self._session is None:
                break                       # no session to pump; report what the camera gave us
            self._session.step(render=True)
            if app is not None:
                app.update()
            depth_mm, frame, mask = self._read_buffers()
            if _buffers_usable(depth_mm, mask):
                print(f"[perception] {self._target}: annotator buffers needed {extra} extra "
                      f"pump(s) after {self._warmup} warmup steps", flush=True)
                # On file as well as on stdout: this is the rare event the retry loop exists for,
                # and a run whose stdout was not kept still has to show whether the annotators
                # were stale.
                self._log.warning(
                    "%s: annotator buffers needed %d extra pump(s) after %d warmup steps",
                    self._target, extra, self._warmup,
                )
                break
        else:
            raise RuntimeError(
                f"perception buffers never became usable for {self._target} after "
                f"{self._warmup} warmup steps + {_BUFFER_RETRIES} extra pumps: "
                f"depth shape={getattr(depth_mm, 'shape', None)}, "
                f"mask shape={getattr(mask, 'shape', None)}. An empty buffer means the annotator "
                f"produced no frame (attach/tick), not that the scene is empty."
            )
        intrinsics = np.asarray(self._camera.get_intrinsics_matrix(), dtype=np.float64)
        if mask is None:
            # An empty mask is a legitimate frame, meaning not visible, so it is substituted rather
            # than raised, and it travels on as an ordinary no_valid_grasp. Whether the target was
            # occluded or the prim path is wrong is only decidable here.
            self._log.warning(
                "%s not present in the instance-id segmentation; emitting an EMPTY mask", self._target,
            )
            mask = np.zeros(depth_mm.shape, dtype=bool)

        # Kept before the overwrite below: that value is where a jaw is driven, and a
        # consumer building obstacle geometry needs the surface the camera rendered.
        rendered_depth_mm = depth_mm.copy()
        if self._ground_truth_depth and bool(np.asarray(mask).any()):
            from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

            obj_z_m = float(np.asarray(SingleRigidPrim(self._target).get_world_pose()[0])[2])
            cam_z_m = float(np.asarray(self._camera.get_world_pose()[0])[2])
            grasp_depth_mm = (cam_z_m - obj_z_m) * 1000.0 - self._grasp_lift_mm
            depth_mm = np.where(np.asarray(mask), grasp_depth_mm, depth_mm)

        # Read RGB from the rendered frame dict, as MultiObjectVisionPerceptionSource does:
        # camera.get_rgb() returns None on Isaac 5.1, so the grasp overlay that --debug-frames
        # dumps has nothing to draw on without frame['rgb']. rgb stays None when unavailable.
        _rgba = frame.get("rgb") if isinstance(frame, Mapping) else None
        rgb = np.asarray(_rgba)[..., :3] if _rgba is not None else None
        self._log.info(
            "acquired %s in %.1f ms: mask %d px, depth %s, rgb=%s",
            self._target, (time.perf_counter() - started) * 1000.0,
            int(np.asarray(mask).astype(bool).sum()), depth_mm.shape, rgb is not None,
        )
        # GroundTruthSegmentation is frozen, so its read-only .mask does not match
        # SegmentationLike's settable `mask` member. The calculator only reads .mask, so the
        # runtime contract holds and the cast states that.
        seg = cast("SegmentationLike", GroundTruthSegmentation(mask=mask, prim_path=self._target))
        return PerceptionFrame(
            depth_map=depth_mm,
            intrinsics=intrinsics,
            segmentations=(seg,),
            rgb=rgb,
            timestamp=time.time(),
            surface_depth_map=rendered_depth_mm,
        )


class MultiObjectGroundTruthPerceptionSource:
    """Dense-clutter substrate: one labelled segmentation per object prim, from ground truth.

    Where :class:`GroundTruthPerceptionSource` emits a single mask for one target, this emits one
    mask per object in a single :class:`PerceptionFrame`. The orchestrator then sees every object
    at once, so its per-target ``compute_result`` receives the remaining objects as
    ``other_object_masks``: the clutter signal the ``DENSE_CLUTTER`` sampler enables on, and the
    input the corridor-risk rerank and the approach-validation feature need. Each segmentation
    carries the object's ``label``, so label-aware target selection can pick the prompted one.

    Same injected-camera pattern, with no ``isaacsim`` at module top. ``ground_truth_depth``
    overlays each object's true centre depth over its own mask, and the masks are disjoint. The
    depth is biased up by ``grasp_lift_mm`` so the fingertips clear the table, the same semantics
    as the single-object source applied per object.
    """

    def __init__(
        self,
        *,
        camera: Any,
        targets: "list[tuple[str, str]]",
        session: Any | None = None,
        warmup_steps: int = 2,
        ground_truth_depth: bool = False,
        grasp_lift_mm: float = 0.0,
        grasp_top_penetration_mm: float | None = None,
    ) -> None:
        # targets: a list of (prim_path, label) pairs, one per object in the clutter scene.
        self._camera = camera
        self._targets = list(targets)
        self._session = session
        self._warmup = max(0, warmup_steps)
        self._ground_truth_depth = ground_truth_depth
        self._grasp_lift_mm = float(grasp_lift_mm)
        # Tall-object fix, opt-in: None leaves the centre+lift path untouched. When set, the grasp
        # is also referenced to the object top, the nearest rendered depth over its mask, biased
        # down by this penetration, and the higher of the two candidate points wins. A short object
        # keeps centre+lift because that is already the higher point; a tall object is raised to
        # near its top, so the gripper body clears the top instead of driving into it and shoving
        # the object while the descent times out.
        self._grasp_top_penetration_mm = (
            None if grasp_top_penetration_mm is None else float(grasp_top_penetration_mm)
        )
        self._log = create_logger(
            "MultiObjectGroundTruthPerceptionSource",
            log_file=PERCEPTION_GT_LOG_FILE, log_dir=WILLY_SIM_LOG_DIR,
        )
        self._log.info(
            "multi-object ground-truth source ready: %d target(s) %s, warmup=%d "
            "ground_truth_depth=%s lift=%.1f mm top_penetration=%s",
            len(self._targets), [label for _, label in self._targets], self._warmup,
            self._ground_truth_depth, self._grasp_lift_mm, self._grasp_top_penetration_mm,
        )
        camera.add_distance_to_image_plane_to_frame()
        camera.add_instance_id_segmentation_to_frame()

    def _read_buffers(self):  # noqa: ANN202 ((depth_mm, frame) straight off the annotators)
        depth_mm = metres_to_millimetres(np.asarray(self._camera.get_depth(), dtype=np.float64))
        # distance_to_image_plane sets background pixels to far-clip / +inf; zero them so the
        # calculator (mask-restricted anyway) never back-projects a non-finite depth.
        depth_mm = np.where(np.isfinite(depth_mm), depth_mm, 0.0)
        return depth_mm, self._camera.get_current_frame()

    def acquire(self) -> PerceptionFrame:
        started = time.perf_counter()
        app = getattr(self._session, "app", None)
        for _ in range(self._warmup):
            if self._session is not None:
                self._session.step(render=True)
                if app is not None:
                    app.update()

        # The same buffer check the single-object source runs, for the same reason. An annotator
        # that has not produced a frame returns an empty array rather than a blank image, and the
        # failure surfaces three frames away as a shape error about something else, or, in a mode
        # that acquires more than once per pick, as a silent `no_valid_grasp` on the second acquire
        # while the first acquire found candidates at the same pose.
        depth_mm, frame = self._read_buffers()
        for extra in range(1, _BUFFER_RETRIES + 1):
            if _buffers_usable(depth_mm, None) or self._session is None:
                break
            self._session.step(render=True)
            if app is not None:
                app.update()
            depth_mm, frame = self._read_buffers()
            if _buffers_usable(depth_mm, None):
                print(f"[perception] multi-object: annotator buffers needed {extra} extra pump(s) "
                      f"after {self._warmup} warmup steps", flush=True)
                # On file as well as on stdout; the single-object source carries the reason this
                # one rare line is worth the duplicate.
                self._log.warning(
                    "multi-object: annotator buffers needed %d extra pump(s) after %d warmup steps",
                    extra, self._warmup,
                )
                break
        else:
            raise RuntimeError(
                f"perception buffers never became usable for {len(self._targets)} targets after "
                f"{self._warmup} warmup steps + {_BUFFER_RETRIES} extra pumps: depth "
                f"shape={getattr(depth_mm, 'shape', None)}. An empty buffer means the annotator "
                f"produced no frame (attach/tick), not that the scene is empty."
            )
        # Capture the rendered depth before any per-object override: the top-referencing below reads the
        # object's true top surface (nearest depth over its mask) from here, not from the overridden map.
        rendered_depth_mm = depth_mm.copy()
        intrinsics = np.asarray(self._camera.get_intrinsics_matrix(), dtype=np.float64)

        cam_z_m = None
        if self._ground_truth_depth:
            cam_z_m = float(np.asarray(self._camera.get_world_pose()[0])[2])

        # list[Any]: GroundTruthSegmentation satisfies the calculator's SegmentationLike at runtime
        # because it carries .mask, but mypy will not pair a frozen-dataclass field with the
        # Protocol's mutable attribute. That is the same gap the single-object source accepts, and
        # Any keeps the tuple assignable.
        segmentations: list[Any] = []
        # Collected over the loop and logged once after it: a dense scene is a dozen-plus objects and
        # a line each would drown the file at one acquire per pick.
        unsegmented: list[str] = []
        for prim, label in self._targets:
            mask = object_mask_from_frame(frame, prim)
            if mask is None:
                unsegmented.append(label)
                mask = np.zeros(depth_mm.shape, dtype=bool)
            if self._ground_truth_depth and cam_z_m is not None and bool(np.asarray(mask).any()):
                from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

                obj_z_m = float(np.asarray(SingleRigidPrim(prim).get_world_pose()[0])[2])
                grasp_depth_mm = (cam_z_m - obj_z_m) * 1000.0 - self._grasp_lift_mm
                grasp_depth_mm = self._maybe_top_reference(grasp_depth_mm, rendered_depth_mm, mask, label)
                depth_mm = np.where(np.asarray(mask), grasp_depth_mm, depth_mm)
            segmentations.append(GroundTruthSegmentation(mask=mask, prim_path=prim, label=label))

        # Read RGB from the rendered frame dict so the grasp overlay renders on the ground-truth
        # dense path too: camera.get_rgb() returns None on Isaac 5.1. rgb stays None when
        # unavailable, and the overlay renders only when rgb is present.
        _rgba = frame.get("rgb") if isinstance(frame, Mapping) else None
        rgb = np.asarray(_rgba)[..., :3] if _rgba is not None else None
        if unsegmented:
            # Not an error: an object can be fully occluded. But an empty mask is also what a wrong
            # prim path produces, and downstream both are just "this object had no candidates".
            self._log.warning(
                "%d of %d object(s) had no instance-id mask -> empty masks for %s",
                len(unsegmented), len(self._targets), unsegmented,
            )
        self._log.info(
            "acquired %d segmentation(s) in %.1f ms (depth %s, rgb=%s, ground_truth_depth=%s)",
            len(segmentations), (time.perf_counter() - started) * 1000.0, depth_mm.shape,
            rgb is not None, self._ground_truth_depth,
        )
        return PerceptionFrame(
            depth_map=depth_mm,
            intrinsics=intrinsics,
            segmentations=tuple(segmentations),
            rgb=rgb,
            timestamp=time.time(),
            surface_depth_map=rendered_depth_mm,
        )

    def _maybe_top_reference(
        self, center_depth_mm: float, rendered_depth_mm: np.ndarray, mask: np.ndarray, label: str
    ) -> float:
        """Return the grasp depth, optionally raised to reference the object top.

        With ``grasp_top_penetration_mm`` at ``None`` this returns ``center_depth_mm`` unchanged.
        When it is set, ``top_depth`` is ``min(rendered_depth over the mask)``, the object's
        nearest and therefore topmost surface; the grasp is referenced to
        ``top_depth + penetration``, and the smaller of that and ``center_depth_mm`` wins. Depth
        and world z are inverse, so the smaller depth is the higher grasp point: a short object
        keeps centre plus lift, a tall object rises to near its top.
        """
        if self._grasp_top_penetration_mm is None:
            return center_depth_mm
        m = np.asarray(mask, dtype=bool)
        vals = np.asarray(rendered_depth_mm, dtype=np.float64)[m]
        vals = vals[vals > 0.0]  # ignore zeroed background/non-finite pixels that leaked into the mask edge
        if vals.size == 0:
            return center_depth_mm
        top_depth_mm = float(np.min(vals)) + self._grasp_top_penetration_mm
        chosen = min(center_depth_mm, top_depth_mm)
        if os.environ.get("WILLY_TRACE_DEPTH"):
            raised = "TOP" if top_depth_mm < center_depth_mm else "centre"
            print(f"[depth] {label!r} center_depth={center_depth_mm:.1f} top_depth(+pen)={top_depth_mm:.1f} "
                  f"-> chosen={chosen:.1f} ({raised}-referenced; grasp_z = cam_z - chosen)", flush=True)
        return chosen
