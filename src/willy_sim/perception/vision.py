"""Real-vision perception sources for the Isaac cell: overhead RGB, GroundingDINO, then SAM2.

Two sources. The single-object source detects the prompted object, segments it and emits one
``PerceptionFrame``. The multi-object dense-clutter twin detects and segments every object, so the
orchestrator sees the neighbour clutter. Both take an injected camera, detector and segmenter, and
neither ``isaacsim`` nor torch is imported at module top, so this module imports on a machine
without Isaac.
"""
from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from src.robot.drivers.sim.adapter import metres_to_millimetres
from src.robot.grasping.types.perception import PerceptionFrame
from src.robot.perception.mask_completion import (
    DEFAULT_MASK_COMPLETION,
    MaskCompletion,
    complete_mask,
)
from src.utility.log_cfg import create_logger
from src.willy_sim.constants import PERCEPTION_VISION_LOG_FILE, WILLY_SIM_LOG_DIR


def _last_route(backend: Any) -> tuple[str, str] | None:
    """``(route, reason)`` for the most recent grounding, or ``None`` when nothing routed it.

    Duck-typed on purpose. Only a routed backend has a decision to report; a plain two-stage
    backend, a ground-truth source or no backend at all returns ``None``, and the pick loop then
    stamps no route on its progress event. Nothing fabricates a route here: a stamped "simple"
    would read as a router's choice when no router ran.
    """
    decision = getattr(backend, "last_decision", None)
    if decision is None:
        return None
    return str(decision.route), str(decision.reason)

__all__ = ["MultiObjectVisionPerceptionSource", "IsaacVisionPerceptionSource"]


class MultiObjectVisionPerceptionSource:
    """Dense-clutter real vision, the perception-in-the-loop twin of the ground-truth source.

    Overhead RGB goes to GroundingDINO ``detect_all`` under a multi-phrase prompt, so it grounds
    every object and not just the target; that is the clutter the dense sampler, the corridor-risk
    rerank and the approach-validation feature need. Each box then goes to SAM2
    ``segment_detection``, and the result is N labelled :class:`SegmentationResult` in one
    :class:`PerceptionFrame`. Neither ``isaacsim`` nor torch is imported at module top: the camera,
    detector and segmenter are injected, and the runner builds them after the scene.

    Two behaviours the dense pick needs:

    * Canonical labels. The orchestrator's label-target match is exact,
      ``seg.label == target_label``, but GroundingDINO returns free-form phrases. Given the
      canonical object labels, the spawn names, this maps each detection's label to the nearest
      canonical one, so ``set_target_label("sugar box")`` selects the right segmentation.
    * Top-referenced depth. As in :class:`MultiObjectGroundTruthPerceptionSource`, the grasp depth
      is referenced to each object's top, the nearest rendered depth over its mask, plus a small
      penetration, and overlaid over the mask. That clears the gripper body on a tall object and
      survives a SAM2 mask edge leaking onto the table, because ``min`` over the mask ignores the
      far pixels.

    The same cell constraints as :class:`IsaacVisionPerceptionSource`: read RGB from
    ``frame["rgb"][..., :3]``, pump ``world.step(render=True)`` and ``app.update()`` on every
    acquire, set a small near clip so the close workspace is not clipped to black, and park the arm
    out of the overhead view before acquiring, which the runner does. The detector must run fp32
    here: fp16 misses small overhead objects.
    """

    def __init__(
        self,
        *,
        camera: Any,
        detector: Any = None,
        segmenter: Any = None,
        backend: Any = None,
        prompt: str,
        object_labels: "list[str] | None" = None,
        session: Any | None = None,
        warmup_steps: int = 20,
        grasp_top_penetration_mm: float = 12.0,
        mask_completion: MaskCompletion = DEFAULT_MASK_COMPLETION,
    ) -> None:
        # Either a ready-made perception backend, or the detector and segmenter a runner passes.
        # The pair is composed into the same backend here rather than chained inside acquire().
        # A runner keeps handing in models it built because the dense path grounds with a larger
        # GroundingDINO than the config default, kept separate from the single-object runners.
        if backend is None:
            if detector is None or segmenter is None:
                raise ValueError(
                    "MultiObjectVisionPerceptionSource needs either backend=..., or both detector=... "
                    "and segmenter=..."
                )
            from src.models.perception_backend import TwoStageBackend

            backend = TwoStageBackend(detector=detector, segmenter=segmenter)
        self._camera = camera
        self._backend = backend
        self._prompt = prompt
        # Canonical object labels, the spawn names. They build the multi-phrase detect prompt so
        # GroundingDINO grounds every object, and normalise each detection's label for the exact
        # target match.
        self._object_labels = list(object_labels) if object_labels else None
        self._session = session
        self._warmup = max(1, warmup_steps)
        self._grasp_top_penetration_mm = float(grasp_top_penetration_mm)
        #: Mask-completion policy. The default is ``DEFAULT_MASK_COMPLETION``;
        #: `robot/perception/mask_completion.py` carries what each policy does and why.
        self._mask_completion = MaskCompletion(mask_completion)
        # Both vision sources log to one file under their own logger names, so a run that swaps
        # single-object for multi-object reads as one sequence. The log format carries the name.
        self._log = create_logger(
            "MultiObjectVisionPerceptionSource",
            log_file=PERCEPTION_VISION_LOG_FILE, log_dir=WILLY_SIM_LOG_DIR,
        )
        self._log.info(
            "multi-object vision source ready: prompt=%r canonical_labels=%d warmup=%d "
            "top_penetration=%.1f mm mask_completion=%s",
            prompt, len(self._object_labels or ()), self._warmup,
            self._grasp_top_penetration_mm, self._mask_completion,
        )
        camera.add_distance_to_image_plane_to_frame()
        try:
            camera.set_clipping_range(0.05, 1.0e6)
        except Exception as exc:  # noqa: BLE001 (near-clip best-effort (the close workspace renders RGB))
            # Swallowed on purpose, but not free: with Isaac's 1.0 m default near clip a camera ~1 m
            # over its own workspace clips the colour pass to black, and the symptom surfaces three
            # frames away as "GroundingDINO detects nothing".
            self._log.warning("set_clipping_range(0.05, 1e6) failed (%r); RGB may be clipped", exc)

    def _detect_prompt(self) -> str:
        """A multi-phrase GroundingDINO prompt ('a. b. c.') so detect_all grounds every object."""
        if self._object_labels:
            return ". ".join(self._object_labels) + "."
        return self._prompt

    def _canonical_label(self, gdino_label: str) -> str:
        """Map a free-form GroundingDINO phrase to the nearest canonical object label.

        The orchestrator's target match is exact, so a phrase has to be normalised first.
        Case-insensitive substring wins first, because GroundingDINO usually echoes the prompt
        phrase, then the largest word overlap. A phrase matching nothing falls back to the raw
        label: that segmentation is then not the target, and it still feeds the neighbour clutter.
        """
        if not self._object_labels:
            return gdino_label
        gl = gdino_label.lower().strip()
        for c in self._object_labels:
            cl = c.lower()
            if cl and (cl in gl or gl in cl):
                return c
        gw = set(gl.split())
        best, best_n = gdino_label, 0
        for c in self._object_labels:
            n = len(gw & set(c.lower().split()))
            if n > best_n:
                best, best_n = c, n
        return best

    def _maybe_fill_to_detection_box(self, mask: np.ndarray, det: Any) -> np.ndarray:
        """Apply the configured mask-completion policy.

        Delegates to ``complete_mask``, the one implementation the real-camera source also calls,
        so the sim and a real cell cannot drift apart on the transform that decides the closing
        axis. The rule replaces a mask that does not fill its detector box, and it never checks
        the precondition it rests on: that the box is a good centred silhouette, which holds for
        an axis-aligned, roughly rectangular object seen top-down and for little else. The default
        policy is ``NONE``, so nothing is filled unless a caller asks for it.

        :class:`IsaacVisionPerceptionSource` does no mask completion at all, so the single-object
        runners never reach this rule, and the ground-truth runners have no detection box to fill.
        This class is the only sim caller. ``mask_completion.py`` carries the measurements behind
        the default.
        """
        return complete_mask(mask, det, policy=self._mask_completion)

    @property
    def last_route(self) -> tuple[str, str] | None:
        """Which route grounded the last frame, for the pick loop to stamp onto its progress event."""
        return _last_route(self._backend)

    def acquire(self) -> PerceptionFrame:
        started = time.perf_counter()
        app = getattr(self._session, "app", None) if self._session is not None else None
        for _ in range(self._warmup):
            if self._session is not None:
                self._session.step(render=True)
            if app is not None:
                app.update()

        depth_mm = metres_to_millimetres(np.asarray(self._camera.get_depth(), dtype=np.float64))
        depth_mm = np.where(np.isfinite(depth_mm), depth_mm, 0.0)
        rendered_depth_mm = depth_mm.copy()  # top-referencing reads the true surface from here
        intrinsics = np.asarray(self._camera.get_intrinsics_matrix(), dtype=np.float64)

        frame = self._camera.get_current_frame()
        rgba = frame.get("rgb") if isinstance(frame, Mapping) else None
        rgb = np.asarray(rgba)[..., :3] if rgba is not None else None

        segmentations: list[Any] = []
        # Counted over the loop and logged once after it: a cluttered multi-phrase prompt grounds
        # a dozen-plus boxes, and one line each would drown the file. WILLY_TRACE_VISION is the
        # per-detection trace.
        empty_masks = 0
        if rgb is not None:
            bgr = np.ascontiguousarray(rgb[..., ::-1])  # detector/segmenter take OpenCV BGR
            # The detector and segmenter chain, and its two failure rules, live in the backend: a
            # detector error emits no segmentation, so a model failure reaches the pick loop as an
            # empty frame rather than a traceback, and a single segmentation error skips that
            # object and keeps the others.
            #
            # One segmentation per detection, carrying the canonical label and the top-referenced
            # depth. GroundingDINO is noisy at a high camera, emitting merged, duplicate and
            # low-score boxes, so several segmentations can share the target label. They are not
            # deduplicated: the orchestrator evaluates every target-labelled segmentation and
            # executes the best-scoring grasp, which is more reliable than picking by detection
            # score, because the highest-scored box can be a fragment of the real one.
            for _obj in self._backend.perceive(bgr, self._detect_prompt()):
                det, seg = _obj.detection, _obj.segmentation
                seg_label = self._canonical_label(getattr(seg, "label", "") or "")
                # SAM2's mask of a long box can truncate at one end, which moves the centroid off
                # centre and with it the grasp. The configured policy decides whether such a mask
                # is replaced by the detection box; see _maybe_fill_to_detection_box.
                mask = self._maybe_fill_to_detection_box(np.asarray(seg.mask).astype(bool), det)
                # SegmentationResult is frozen, so the relabelled, mask-completed segmentation is
                # built with replace: the same values in a new instance, never mutated in place.
                seg = replace(seg, label=seg_label, mask=mask.astype(np.uint8))
                if mask.any():
                    vals = rendered_depth_mm[mask]
                    vals = vals[vals > 0.0]
                    if vals.size:
                        grasp_depth_mm = float(np.min(vals)) + self._grasp_top_penetration_mm
                        depth_mm = np.where(mask, grasp_depth_mm, depth_mm)
                segmentations.append(seg)
                if not mask.any():
                    empty_masks += 1
                if os.environ.get("WILLY_TRACE_VISION"):
                    _ys, _xs = np.where(mask)
                    _ext = (f"y[{int(_ys.min())}-{int(_ys.max())}] x[{int(_xs.min())}-{int(_xs.max())}]"
                            if _ys.size else "empty")
                    _gb = [int(c) for c in getattr(det, "box", [0, 0, 0, 0])]
                    print(f"[vision] det label={getattr(det, 'label', '?')!r}->{seg.label!r} "
                          f"score={getattr(det, 'score', 0.0):.2f} mask_px={int(mask.sum())} "
                          f"mask_ext={_ext} gdino_box={_gb}", flush=True)

        if rgb is None:
            # No colour pass means the detector never ran, and the frame goes downstream with zero
            # segmentations, which the pipeline reports as an ordinary empty perception. From the
            # outside that is indistinguishable from an empty bin; here it is not.
            self._log.warning(
                "camera returned no RGB (near clip? annotator not ticked?); perceiving nothing"
            )
        elif not segmentations:
            self._log.warning(
                "nothing grounded for prompt %r in %.1f ms -> no_valid_grasp downstream",
                self._detect_prompt(), (time.perf_counter() - started) * 1000.0,
            )
        else:
            self._log.info(
                "acquired %d segmentation(s) for prompt %r in %.1f ms (%d empty mask(s), depth %s)",
                len(segmentations), self._detect_prompt(),
                (time.perf_counter() - started) * 1000.0, empty_masks, depth_mm.shape,
            )
        # `rendered_depth_mm` is what the camera rendered, before the grasp-referenced
        # overwrite above. A consumer building obstacle geometry needs the body of an
        # object, and `depth_map` holds a sheet at its top face.
        return PerceptionFrame(
            depth_map=depth_mm, intrinsics=intrinsics, segmentations=tuple(segmentations), rgb=rgb,
            timestamp=time.time(), surface_depth_map=rendered_depth_mm,
        )


class IsaacVisionPerceptionSource:
    """Real-vision perception for a single object, the perception-in-the-loop path.

    Renders RGB and depth from an overhead Isaac ``Camera``, detects the prompted object with
    GroundingDINO, segments it with SAM2, and emits a grasping :class:`PerceptionFrame`. It stands
    in for the ground-truth source wherever a runner wants the vision models in the loop.

    Neither ``isaacsim`` nor torch is imported at module level. The ``camera``, ``detector`` and
    ``segmenter`` are injected, and the runner constructs them after the scene is built, so this
    module imports on a machine without Isaac.

    Two constraints of the Isaac camera this encodes:

    * RGB is read from ``camera.get_current_frame()["rgb"][..., :3]``. Isaac 5.1 stores rgba under
      the ``"rgb"`` key, and ``get_rgb()`` reads an absent ``"rgba"`` key and returns ``None``.
    * each ``acquire()`` pumps ``app.update()`` and ``world.step(render=True)`` to drive the Kit
      render loop reliably; this class sets the small near clip itself in ``__init__`` so the close
      workspace is not clipped to black. ``build_combined_scene`` sets one only for a tilted
      camera: the nadir overhead camera keeps Isaac's 1.0 m default on purpose, which hides the
      home-pose arm link and keeps the ground-truth instance mask clean.

    The caller must park the arm out of the camera's view before ``acquire()``, because an overhead
    camera sees the arm whenever it is over the workspace. The vision runner does this.
    """

    def __init__(
        self,
        *,
        camera: Any,
        detector: Any = None,
        segmenter: Any = None,
        backend: Any = None,
        prompt: str,
        session: Any | None = None,
        warmup_steps: int = 20,
        grasp_depth_offset_mm: float = 0.0,
    ) -> None:
        # Two construction paths, kept apart because they ground differently. This source is
        # single-object: with a detector and a segmenter it calls ``detect``, the best match for
        # the prompt, while a backend returns every grounded object under ``detect_all``. With a
        # backend, ``acquire`` takes the highest-scoring perceived object, which is what ``detect``
        # returns. A backend, or both a detector and a segmenter: anything else is refused below.
        if backend is None and (detector is None or segmenter is None):
            raise ValueError(
                "IsaacVisionPerceptionSource needs either backend=..., or both detector=... and "
                "segmenter=..."
            )
        self._camera = camera
        self._backend = backend
        self._detector = detector
        self._segmenter = segmenter
        self._prompt = prompt
        self._session = session
        self._warmup = max(1, warmup_steps)
        # Push the grasp this many mm below the detected top surface (a single top-down view only
        # sees the object's top; gripping the top edge slips, so bias into the body for a solid grip).
        self._grasp_depth_offset_mm = float(grasp_depth_offset_mm)
        self._log = create_logger(
            "IsaacVisionPerceptionSource",
            log_file=PERCEPTION_VISION_LOG_FILE, log_dir=WILLY_SIM_LOG_DIR,
        )
        self._log.info(
            "single-object vision source ready: prompt=%r backend=%s warmup=%d depth_offset=%.1f mm",
            prompt, type(backend).__name__ if backend is not None else "two-stage", self._warmup,
            self._grasp_depth_offset_mm,
        )
        camera.add_distance_to_image_plane_to_frame()
        # The Isaac Camera defaults to a 1.0 m near clip; the overhead cam sits ~1 m above the
        # table, so the workspace lands at/inside that plane and the color pass clips it to black
        # (depth still reports geometry). A small near clip is essential for close-range RGB.
        try:
            camera.set_clipping_range(0.05, 1.0e6)
        except Exception as exc:  # noqa: BLE001 (near-clip is best-effort (the clipping API varies by Isaac version))
            # See the sibling source: a clip that did not take is why the RGB pass comes back black,
            # and nothing downstream can tell that from an empty scene.
            self._log.warning("set_clipping_range(0.05, 1e6) failed (%r); RGB may be clipped", exc)

    @property
    def last_route(self) -> tuple[str, str] | None:
        """Which route grounded the last frame, for the pick loop to stamp onto its progress event."""
        return _last_route(self._backend)

    def acquire(self) -> PerceptionFrame:
        started = time.perf_counter()
        app = getattr(self._session, "app", None) if self._session is not None else None
        for _ in range(self._warmup):
            if self._session is not None:
                self._session.step(render=True)
            if app is not None:
                app.update()

        depth_mm = metres_to_millimetres(np.asarray(self._camera.get_depth(), dtype=np.float64))
        depth_mm = np.where(np.isfinite(depth_mm), depth_mm, 0.0)
        intrinsics = np.asarray(self._camera.get_intrinsics_matrix(), dtype=np.float64)

        frame = self._camera.get_current_frame()
        rgba = frame.get("rgb") if isinstance(frame, Mapping) else None
        rgb = np.asarray(rgba)[..., :3] if rgba is not None else None

        segmentations: tuple = ()
        if rgb is not None:
            bgr = np.ascontiguousarray(rgb[..., ::-1])  # detector/segmenter take OpenCV BGR
            try:
                if self._backend is not None:
                    perceived = self._backend.perceive(bgr, self._prompt)
                    # Singular by contract: this source answers "where is the prompted object", so
                    # it takes the highest-scoring one, not the first. `detect()` selects
                    # `scores.argmax()`, while `detect_all()` yields the model's own emission
                    # order, which is not sorted. The two coincide on a single-object scene and
                    # diverge as soon as the scene holds several similar objects, where taking the
                    # first would grasp the wrong one.
                    #
                    # This holds for the VLM route too: every VLM detection carries the same
                    # nominal score, so max() is stable and keeps the model's own ordering.
                    best = max(perceived, key=lambda o: o.detection.score, default=None)
                    segmentations = (best.segmentation,) if best is not None else ()
                else:
                    detection = self._detector.detect(bgr, self._prompt)
                    segmentations = (self._segmenter.segment_detection(bgr, detection),)
            except Exception:  # noqa: BLE001 (no detection / model error -> honest no_valid_grasp (below))
                # Nothing cleared the detection threshold, or a model failed: emit no segmentation
                # rather than grasp at nothing, and let the pipeline report an empty perception.
                #
                # `()` is returned and nothing is raised, so this line is the only record the
                # exception leaves; hence a traceback rather than a one-line message. "The model
                # crashed" and "the scene is empty" reach the caller as the very same outcome.
                self._log.exception(
                    "grounding failed for prompt %r; perceiving nothing", self._prompt
                )
                segmentations = ()

        # The rendered surface is kept first, because the bias below is a grasp target and
        # not a measurement.
        rendered_depth_mm = depth_mm.copy()
        # Bias the grasp into the object body (larger depth = farther from cam = lower) over the
        # object mask, so the grip is mid-body rather than on the slippery top edge.
        if segmentations and self._grasp_depth_offset_mm:
            obj_mask = np.asarray(segmentations[0].mask).astype(bool)
            depth_mm = np.where(obj_mask, depth_mm + self._grasp_depth_offset_mm, depth_mm)

        if rgb is None:
            self._log.warning(
                "camera returned no RGB (near clip? annotator not ticked?); perceiving nothing"
            )
        elif not segmentations:
            self._log.warning(
                "nothing grounded for prompt %r in %.1f ms -> no_valid_grasp downstream",
                self._prompt, (time.perf_counter() - started) * 1000.0,
            )
        else:
            self._log.info(
                "acquired 1 segmentation for prompt %r in %.1f ms (mask %d px, depth %s)",
                self._prompt, (time.perf_counter() - started) * 1000.0,
                int(np.asarray(segmentations[0].mask).astype(bool).sum()), depth_mm.shape,
            )
        return PerceptionFrame(
            depth_map=depth_mm, intrinsics=intrinsics, segmentations=segmentations, rgb=rgb,
            timestamp=time.time(), surface_depth_map=rendered_depth_mm,
        )
