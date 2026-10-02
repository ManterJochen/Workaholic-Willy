"""One seam between "a prompt" and "masks", so a pipeline is a choice rather than an assembly.

The two-stage chain, ``detect_all(bgr, prompt)`` followed by ``segment_detection(bgr, det)`` per
detection, lives here instead of in each caller. A :class:`PerceptionBackend` is that loop, named
and owned:

    backend = build_perception(models_cfg)
    objects = backend.perceive(image_bgr, "a green cube")

``perceive`` returns pairs, ``(detection, segmentation)``, because callers consume both: the real
cell fills an incomplete mask back out to the detection box when SAM2 drops one end of a long object
(a measured ~22 mm centroid shift, an off-centre grasp, no lift). A backend whose model produces
masks natively, with no box stage, synthesises the detection from the mask's own bounding box.

This module does not import torch. It is the contract, not the models; the wrappers stay behind the
lazy imports in :mod:`src.models.factory`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from src.models.constants import MODELS_LOG_DIR, PERCEPTION_BACKEND_LOG_FILE
from src.utility.log_cfg import create_logger

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.models.detection.types import Detection
    from src.models.segmentation.types import SegmentationResult

__all__ = ["PerceivedObject", "PerceptionBackend", "TwoStageBackend", "failures_of", "last_failure_of"]

#: How much of an exception's text a failure keeps: enough to name the cause, short enough for an event field.
_FAILURE_TEXT_LIMIT = 300


def failures_of(backend: Any) -> int:
    """How many perceives of ``backend`` returned nothing because a model raised; 0 for a backend that counts none.

    Duck-typed, so a caller sums the count over whatever backends a cell holds (a hand-built one included) without
    knowing their classes. Only a real integer counts: a stand-in whose attribute is a mock reads 0, never a number
    that happens to convert.
    """
    value = getattr(backend, "failures", 0)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def last_failure_of(backend: Any) -> str:
    """The latest counted failure of ``backend``, one line, or ``""``."""
    value = getattr(backend, "last_failure", "")
    return value if isinstance(value, str) else ""


@dataclass(frozen=True, slots=True)
class PerceivedObject:
    """One grounded object: the detection box and the mask cut for it.

    Both halves travel together because both are consumed. The grasp calculator reads the mask; a
    caller compares the mask against the box to decide whether the mask is trustworthy.
    """

    detection: "Detection"
    segmentation: "SegmentationResult"


@runtime_checkable
class PerceptionBackend(Protocol):
    """Prompt in, grounded objects out. The whole perception contract.

    :func:`src.models.factory.build_perception` constructs an implementation from config, and the
    implementation holds whatever models it needs. Loaded weights are its only state; ``perceive``
    is a pure function of its arguments.
    """

    def perceive(self, image_bgr: Any, prompt: str) -> tuple[PerceivedObject, ...]:
        """Ground ``prompt`` in ``image_bgr``. Returns ``()`` when nothing matched.

        An empty result is an answer, not an error: the pick loop reports ``no_perception`` and
        moves on, the honest outcome when the scene does not contain what was asked for.
        """
        ...


class TwoStageBackend:
    """Detector then segmenter, the two-stage chain.

    Two failures are swallowed rather than raised:

    * a detector failure yields no objects, so a model error surfaces downstream as
      ``no_valid_grasp`` instead of a traceback in the middle of a pick;
    * a single segmentation failure skips that object and keeps the rest, so one bad mask in a
      cluttered bin does not discard the objects that segmented fine.

    Swallowed is not hidden. Every perceive that returned nothing because a model raised (the
    detector, or the segmenter on every detection) is counted in :attr:`failures` and named in
    :attr:`last_failure`, so a caller that must tell "the scene is empty" from "the model could not
    look" reads the count before and after; the console's task ends ``detector_failed`` on it instead
    of ``nothing_left``. A VLM that cannot load raises inside the detector, so it is counted too.

    Detections are not de-duplicated. A multi-phrase prompt grounds neighbour clutter on purpose:
    the dense sampler needs to know what else is in the bin.
    """

    def __init__(self, detector: Any, segmenter: Any) -> None:
        self.detector = detector
        self.segmenter = segmenter
        # The seam's own log. Both swallow-and-continue paths below are silent downstream: a pick
        # that dropped every mask and a pick over an empty bin both arrive as `no_perception`, and
        # these lines are the only place that difference is recorded.
        self.logger = create_logger(
            "TwoStageBackend", log_file=PERCEPTION_BACKEND_LOG_FILE, log_dir=MODELS_LOG_DIR,
        )
        self._failures = 0
        self._last_failure = ""
        # A count read on one thread while a perceive on another adds to it: one lock keeps the
        # number and its sentence a pair.
        self._failure_lock = threading.Lock()

    @property
    def failures(self) -> int:
        """Perceives that returned nothing because a model raised. Only grows; read it before and after."""
        return self._failures

    @property
    def last_failure(self) -> str:
        """The latest of those failures, one line naming the model and the exception, or ``""``."""
        return self._last_failure

    def _count_failure(self, what: str) -> None:
        with self._failure_lock:
            self._failures += 1
            self._last_failure = what[:_FAILURE_TEXT_LIMIT]

    def perceive(self, image_bgr: Any, prompt: str) -> tuple[PerceivedObject, ...]:
        started = time.perf_counter()
        try:
            detections = self.detector.detect_all(image_bgr, prompt)
        except Exception as exc:  # noqa: BLE001 (a model error emits no object (honest no_valid_grasp))
            # The failure returns `()` and never raises, so this line is the only record of the
            # exception; hence a full traceback rather than a one-line message.
            self.logger.exception("detector failed for prompt %r, perceiving nothing", prompt)
            self._count_failure(f"the detector raised {type(exc).__name__}: {exc}")
            return ()

        objects: list[PerceivedObject] = []
        # Counted, not logged per detection: a cluttered multi-phrase prompt grounds a dozen-plus
        # boxes, and one line each would drown the file. One aggregate line follows the loop.
        seg_failures = 0
        first_seg_error = ""
        # Counted rather than `len(detections)`: the detector is duck-typed, and a backend that
        # yields its boxes has no length to take on a path that must not crash.
        n_detections = 0
        for detection in detections:
            n_detections += 1
            try:
                segmentation = self.segmenter.segment_detection(image_bgr, detection)
            except Exception as exc:  # noqa: BLE001 (skip one segmentation failure, keep the rest)
                seg_failures += 1
                if not first_seg_error:
                    first_seg_error = repr(exc)
                continue
            objects.append(PerceivedObject(detection=detection, segmentation=segmentation))

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if seg_failures:
            self.logger.warning(
                "segmentation dropped %d of %d detection(s) for prompt %r (first error: %s)",
                seg_failures, n_detections, prompt, first_seg_error,
            )
            if seg_failures == n_detections:
                # The detector saw something and every mask raised: nothing came back because the
                # model failed, not because the scene is empty.
                self._count_failure(
                    f"the segmenter raised on all {n_detections} detection(s), first: {first_seg_error}"
                )
        self.logger.info(
            "perceived %d object(s) from %d detection(s) for prompt %r in %.1f ms",
            len(objects), n_detections, prompt, elapsed_ms,
        )
        return tuple(objects)
