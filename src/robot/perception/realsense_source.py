"""Live-camera perception: an RGB-D frame, GroundingDINO, SAM2, one :class:`PerceptionFrame`.

This is the real-hardware twin of ``willy_sim/perception/vision.py``'s
:class:`MultiObjectVisionPerceptionSource`, with the Isaac session-stepping removed and an RGB-D
streamer in place of an Isaac ``Camera``. The whole vision-driven pick path waits on it: the sim
feeds this stack, and a physical camera never has.

It lives under ``robot/`` and not ``camera/`` because the dependency stack only ever crosses
``robot -> camera`` (measured: ``grep "from src.robot" src/camera`` is empty). This adapter needs
both a camera streamer and ``robot.grasping``'s :class:`PerceptionFrame`, so it cannot sit in the
camera package without inverting that edge. It takes its streamer, detector and segmenter as
injected dependencies (duck-typed, exactly like the sim source), so this module imports with no
``pyrealsense2`` and no torch; the ``__main__`` exerciser is what builds the heavy pieces.

Bucket (3): the behaviour on real D435 depth, which returns holes (0) inside a mask and leaks at
mask edges, is unmeasured until a camera exists. The two robustness fixes carried over from the sim
source, top-referenced depth over holes and mask completion, are the right starting point and not a
proven answer.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from src.robot.grasping.types.perception import PerceptionFrame
from src.robot.perception.mask_completion import (
    DEFAULT_MASK_COMPLETION,
    MaskCompletion,
    complete_mask,
)

__all__ = ["RealSenseVisionPerceptionSource"]


class RealSenseVisionPerceptionSource:
    """Grab one RGB-D frame, ground + segment the prompted object(s), emit a :class:`PerceptionFrame`.

    Parameters
    ----------
    streamer
        Anything with ``grab() -> RGBDFrame`` (``.color`` BGR uint8, ``.depth`` uint16 millimetres)
        and ``get_intrinsics() -> 3x3 K | None``. In production this is
        ``camera.setup.image_taking.rgbd.RealSenseRGBDStreamer``; any object carrying those two
        methods satisfies it.
    backend
        A ready-made perception backend (``models.perception_spec.PerceptionSpec.build()``). When
        given, ``detector`` and ``segmenter`` are ignored; when omitted, both of them are required
        or ``__init__`` raises ``ValueError``. Both real construction sites in
        ``robot/execution/autonomous_grasp/cells.py`` pass ``backend``.
    detector, segmenter
        The alternative to ``backend``, composed here into a ``TwoStageBackend``:
        ``detector.detect_all(bgr, prompt) -> [Detection]`` and
        ``segmenter.segment_detection(bgr, det) -> SegmentationResult`` (``.mask`` HxW, ``.label`` str).
    prompt
        The GroundingDINO phrase(s). A multi-phrase prompt grounds every object (the neighbour clutter
        the dense sampler needs), not only the target.
    object_labels
        Optional canonical labels (the known object names). When given, each detection's free-form
        GDINO label is mapped to the nearest canonical one, so an exact ``seg.label == target``
        match works downstream. When omitted, GDINO labels pass through unchanged.
    intrinsics
        Optional 3x3 K override. The D1 seam: leave ``None`` to use the streamer's factory K (the
        D435 ships calibrated), or pass a bench-calibrated K to override it. The source records
        which was used on the frame's provenance via ``intrinsics_source``.
    grasp_top_penetration_mm
        Bias the grasp depth this far below each object's nearest (top) surface. A single top-down view
        sees only the top; gripping the very edge slips, so the grasp is referenced to the top + this.
    warmup_grabs
        Throwaway grabs before the real one, so a physical camera's auto-exposure / auto-white-balance
        has settled. A real RealSense needs a handful; a fake ignores it.
    mask_completion
        Which mask-completion policy to apply; the default is ``DEFAULT_MASK_COMPLETION``, which is
        ``NONE``. :mod:`mask_completion` carries the measurement behind that default.
    """

    def __init__(
        self,
        *,
        streamer: Any,
        detector: Any = None,
        segmenter: Any = None,
        backend: Any = None,
        prompt: str,
        object_labels: tuple[str, ...] = (),
        intrinsics: np.ndarray | None = None,
        grasp_top_penetration_mm: float = 3.0,
        warmup_grabs: int = 5,
        mask_completion: MaskCompletion = DEFAULT_MASK_COMPLETION,
    ) -> None:
        # Either a ready-made perception backend, or the detector plus segmenter pair, which is
        # composed into the same backend here rather than run as a two-stage chain by this source.
        # The pair stays a supported path because callers that hand-assemble models need it.
        if backend is None:
            if detector is None or segmenter is None:
                raise ValueError(
                    "RealSenseVisionPerceptionSource needs either backend=..., or both detector=... "
                    "and segmenter=..."
                )
            from src.models.perception_backend import TwoStageBackend

            backend = TwoStageBackend(detector=detector, segmenter=segmenter)
        self._streamer = streamer
        self._backend = backend
        self._prompt = prompt
        self._object_labels = tuple(object_labels)
        self._intrinsics_override = None if intrinsics is None else np.asarray(intrinsics, dtype=np.float64)
        self._grasp_top_penetration_mm = float(grasp_top_penetration_mm)
        self._warmup_grabs = max(0, int(warmup_grabs))
        #: What to do with a mask that underfills its detection box. `mask_completion.py` carries
        #: the measurement behind the default and the reason this is a lever.
        self._mask_completion = MaskCompletion(mask_completion)
        #: "override" if a calibrated K was supplied, else "factory". D1 provenance.
        self.intrinsics_source = "override" if intrinsics is not None else "factory"

    # ------------------------------------------------------------------ label canonicalisation
    def _canonical_label(self, gdino_label: str) -> str:
        """Map a free-form GDINO phrase to the nearest known object label (identity if none given)."""
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

    # ------------------------------------------------------------------ mask robustness
    def _maybe_fill_to_detection_box(self, mask: np.ndarray, det: Any) -> np.ndarray:
        """Apply the configured mask-completion policy.

        The rule lives in :mod:`mask_completion` so a real cell and the sim source cannot drift
        apart on a transform that decides the grasp's closing axis. The default is ``NONE``: over
        270 reference scenes the axis-aligned rule costs 7.5 percentage points of top-1 and loses
        five grasps for every one it wins. That module's docstring carries the numbers and the
        bound on the rescue it gives up.
        """
        return complete_mask(mask, det, policy=self._mask_completion)

    # ------------------------------------------------------------------ the frame
    #: These pixels come off a physical device, so the console may say "camera". See
    #: `perception/viewfinder.py` for why this is declared rather than inferred.
    colour_source_kind = "camera"

    def peek_color(self) -> np.ndarray | None:
        """One colour frame, with no detector, no segmenter and no depth work. BGR uint8.

        The :mod:`~src.robot.perception.viewfinder` capability, implemented here because this
        source owns an open streamer. It takes one frameset and no warm-up prefix: a viewer that
        discarded five frames per tick would pull the device at six times the rate it displays.

        Not safe to call concurrently with :meth:`acquire`: both reach the same unsynchronised
        pipeline. The caller enforces that; see the module docstring in ``viewfinder.py``.
        """
        grabbed = self._streamer.grab()
        colour = getattr(grabbed, "color", None)
        if colour is None:
            return None
        return np.ascontiguousarray(np.asarray(colour))

    def close(self) -> None:
        """Release the camera. Idempotent, and it never raises.

        The counterpart to the ``provider.open_rig(...)`` that ``build_real_components`` performs;
        it gives the rig back through the handle's ``release()``, so the rig lifecycle stays with
        the ``FrameProvider`` and not with the streamer. A long-running process builds a cell more
        than once: a CLI runner opens a device and lets process exit close it, while a console
        builds, gets refused, has its config fixed and builds again. On real hardware a second
        ``pipeline.start()`` on a device the first build is still streaming fails, so without this
        the console cannot rebuild until it is restarted.

        Never raises, because this is teardown: a camera that cannot be closed must not stop the
        thing that was closing it.
        """
        release = getattr(self._streamer, "release", None)
        if callable(release):
            try:
                release()
            except Exception:  # noqa: BLE001 (teardown reports nothing it can fix)
                pass

    def acquire(self) -> PerceptionFrame:
        for _ in range(self._warmup_grabs):
            self._streamer.grab()  # discard: let auto-exposure / white-balance settle on real hardware

        rgbd = self._streamer.grab()
        bgr = np.ascontiguousarray(np.asarray(rgbd.color))          # detector/segmenter take OpenCV BGR
        depth_mm = np.asarray(rgbd.depth, dtype=np.float64)         # uint16 mm as float; 0 == hole
        rendered_depth_mm = depth_mm.copy()                         # true surface, read for top-referencing

        if self._intrinsics_override is not None:
            intrinsics = self._intrinsics_override.copy()
        else:
            k = self._streamer.get_intrinsics()
            if k is None:
                raise RuntimeError(
                    "streamer.get_intrinsics() returned None: open() the streamer before acquire(), or "
                    "pass a calibrated `intrinsics=` (the D1 override)."
                )
            intrinsics = np.asarray(k, dtype=np.float64)

        segmentations: list[Any] = []
        # The detector/segmenter chain and its two failure rules live in the backend: a detector
        # error yields no objects (an honest no_valid_grasp downstream), and a single segmentation
        # error skips that object and keeps the rest.
        perceived = self._backend.perceive(bgr, self._prompt)

        for obj in perceived:
            det, seg = obj.detection, obj.segmentation
            label = self._canonical_label(getattr(seg, "label", "") or "")
            mask = self._maybe_fill_to_detection_box(np.asarray(seg.mask).astype(bool), det)
            seg = replace(seg, label=label, mask=mask.astype(np.uint8))
            # Top-referenced depth: read the nearest real surface over the mask, skipping D435
            # holes (0), and overlay grasp_depth = that top + penetration. Holes inside the mask
            # are not sampled; if the whole mask is holes (all 0), the raw depth stays, so the
            # failure is visible downstream rather than silently grasping at a fabricated plane.
            if mask.any():
                vals = rendered_depth_mm[mask]
                vals = vals[vals > 0.0]
                if vals.size:
                    grasp_depth_mm = float(np.min(vals)) + self._grasp_top_penetration_mm
                    depth_mm = np.where(mask, grasp_depth_mm, depth_mm)
            segmentations.append(seg)

        rgb = bgr[..., ::-1]  # BGR -> RGB for any debugging consumer (no reader in robot/ today)
        return PerceptionFrame(
            depth_map=depth_mm, intrinsics=intrinsics, segmentations=tuple(segmentations), rgb=rgb
        )
