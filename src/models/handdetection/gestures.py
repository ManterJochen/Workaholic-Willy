"""Thumbs up / thumbs down, from MediaPipe's canned gesture classifier.

The canned model stands in for hand-written geometry because "thumb extended along the hand's
up-axis, other fingers curled" also describes a hand holding a pen, a fist seen at an angle, and a
thumb pointing at the camera.

What this package promises is narrower than what the classifier knows. Of the seven shapes it
reports, two are mapped and the rest become `OTHER` with the raw label kept. Narrowing here rather
than at each call site means a caller cannot start depending on `Pointing_Up` without editing this
file.

The label spelling is not assumed. MediaPipe documents the categories as `Thumb_Up` / `Thumb_Down`,
but the mapping runs on a normalised form (case-folded, separators stripped), so `Thumb_Up`,
`thumb up` and `ThumbUp` all land on the same value. The `.task` bundles are an operator download
and are not present here, so the spelling a given bundle uses is unverified and the normalisation
is what makes the mapping hold.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any, Final, Literal, Optional

import numpy as np

from src.models.handdetection.constants import (
    DETECTION_THRESHOLD,
    GESTURE_MODEL_URL,
    GESTURE_RECOGNIZER_LOG_FILE,
    MAX_HANDS,
    MIN_GESTURE_CONFIDENCE,
    MODELS_LOG_DIR,
    PRESENCE_THRESHOLD,
    TRACKING_THRESHOLD,
)
from src.models.handdetection.model_files import require_mediapipe, resolve_model_file
from src.models.handdetection.palm_detector import PalmDetector
from src.models.handdetection.types import (
    GestureReading,
    HandGesture,
    HandObservation,
)
from src.utility.log_cfg import create_logger
from src.utility.vision import bgr_to_rgb

__all__ = ["ThumbGestureRecognizer", "normalise_gesture_label", "to_hand_gesture"]

_SEPARATORS: Final[re.Pattern[str]] = re.compile(r"[\s_\-]+")

#: Normalised MediaPipe label -> the value this package promises. A label that is absent here
#: becomes `OTHER`, a shape recognised and not acted on, as against `NONE`, nothing recognised.
_GESTURE_BY_LABEL: Final[dict[str, HandGesture]] = {
    "thumbup": HandGesture.THUMB_UP,
    "thumbsup": HandGesture.THUMB_UP,
    "thumbdown": HandGesture.THUMB_DOWN,
    "thumbsdown": HandGesture.THUMB_DOWN,
}

#: MediaPipe's own "no gesture" categories. They arrive as ordinary classifications carrying a
#: score, so without this set they would read as a confidently recognised `OTHER`.
_NONE_LABELS: Final[frozenset[str]] = frozenset({"none", "unknown", ""})


def normalise_gesture_label(label: str) -> str:
    """Case-fold a classifier label and strip separators: `Thumb_Up` -> `thumbup`."""
    return _SEPARATORS.sub("", str(label or "").strip()).lower()


def to_hand_gesture(label: str) -> HandGesture:
    """Map a raw classifier label onto the four values this package commits to."""
    normalised = normalise_gesture_label(label)
    if normalised in _NONE_LABELS:
        return HandGesture.NONE
    return _GESTURE_BY_LABEL.get(normalised, HandGesture.OTHER)


class ThumbGestureRecognizer:
    """Recognise thumbs-up / thumbs-down, and return the palm centre from the same pass.

    The canned bundle embeds a hand landmarker, so its result already carries `hand_landmarks`; a
    separate `PalmDetector` over the same frame would load a second model and repeat that
    inference. Each returned `HandObservation` pairs a gesture with the palm it came from, aligned
    by construction rather than by the caller zipping two lists.
    """

    def __init__(
        self,
        model_path: str,
        *,
        max_hands: int = MAX_HANDS,
        threshold: float = DETECTION_THRESHOLD,
        tracking_threshold: float = TRACKING_THRESHOLD,
        presence_threshold: float = PRESENCE_THRESHOLD,
        min_gesture_confidence: float = MIN_GESTURE_CONFIDENCE,
        config_key: str = "the gesture bundle (mediapipe-gesture) under models.weights_root",
    ) -> None:
        require_mediapipe("gesture recognition")
        resolved = resolve_model_file(
            model_path, config_key=config_key, download_url=GESTURE_MODEL_URL
        )

        from mediapipe.tasks import python
        from mediapipe.tasks.python.components.processors import ClassifierOptions
        from mediapipe.tasks.python.vision import (
            GestureRecognizer,
            GestureRecognizerOptions,
            RunningMode,
        )

        self.logger = create_logger(
            "GestureRecognizer", GESTURE_RECOGNIZER_LOG_FILE, log_dir=MODELS_LOG_DIR
        )
        self._min_gesture_confidence = float(min_gesture_confidence)
        self._recognizer = GestureRecognizer.create_from_options(
            GestureRecognizerOptions(
                base_options=python.BaseOptions(model_asset_path=resolved),
                running_mode=RunningMode.VIDEO,
                num_hands=int(max_hands),
                min_hand_detection_confidence=float(threshold),
                min_hand_presence_confidence=float(presence_threshold),
                min_tracking_confidence=float(tracking_threshold),
                # The classifier's own floor, set alongside the `_read_gesture` check rather than
                # instead of it: this one keeps low-scoring categories out of the result entirely,
                # while that check decides `NONE` versus `OTHER` among the ones that do come back.
                canned_gesture_classifier_options=ClassifierOptions(
                    score_threshold=float(min_gesture_confidence),
                ),
            )
        )
        self._lock = threading.Lock()
        self._last_timestamp_ms = -1
        self.logger.info(
            "ThumbGestureRecognizer ready: model=%s max_hands=%d detect>=%.2f gesture>=%.2f",
            resolved, max_hands, threshold, min_gesture_confidence,
        )

    # --- Timestamps ------------------------------------------------------------------------------

    def _next_timestamp_ms(self, timestamp_ms: Optional[int]) -> int:
        with self._lock:
            proposed = (
                int(timestamp_ms)
                if timestamp_ms is not None
                else int(time.monotonic_ns() // 1_000_000)
            )
            if proposed <= self._last_timestamp_ms:
                proposed = self._last_timestamp_ms + 1
            self._last_timestamp_ms = proposed
            return proposed

    # --- Recognition -----------------------------------------------------------------------------

    def recognise(
        self, frame_bgr: np.ndarray, *, timestamp_ms: Optional[int] = None
    ) -> list[HandObservation]:
        """One observation per detected hand: where its palm is, and what it was doing."""
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            raise ValueError("ThumbGestureRecognizer.recognise needs a non-empty BGR image")

        import mediapipe as mp

        rgb = bgr_to_rgb(frame_bgr)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._recognizer.recognize_for_video(
            image, self._next_timestamp_ms(timestamp_ms)
        )
        return self._to_observations(result, frame_bgr.shape[:2])

    def observe(self, frame_bgr: np.ndarray) -> list[HandObservation]:
        """`HandObserver`: the same call as `recognise`, under the name `HandFinder` expects."""
        return self.recognise(frame_bgr)

    def _to_observations(self, result: Any, shape_hw: tuple[int, int]) -> list[HandObservation]:
        palms = PalmDetector._to_detections(result, shape_hw)
        gestures = getattr(result, "gestures", None) or []

        observations: list[HandObservation] = []
        for index, palm in enumerate(palms):
            categories = gestures[index] if index < len(gestures) else None
            observations.append(
                HandObservation(palm=palm, gesture=self._read_gesture(categories, index))
            )
        return observations

    def _read_gesture(self, categories: Any, hand_index: int) -> GestureReading:
        """Turn one hand's category list into a single reading.

        A thumb label scoring below `min_gesture_confidence` is reported as `NONE` rather than as a
        low-confidence `THUMB_UP`, so a confirmation gesture cannot fire on a weak score.
        """
        if not categories:
            return GestureReading(
                gesture=HandGesture.NONE, confidence=0.0, raw_label="", hand_index=hand_index
            )
        top = categories[0]
        raw_label = str(getattr(top, "category_name", "") or "")
        confidence = float(getattr(top, "score", 0.0) or 0.0)
        confidence = min(max(confidence, 0.0), 1.0)

        gesture = to_hand_gesture(raw_label)
        if gesture in (HandGesture.THUMB_UP, HandGesture.THUMB_DOWN) and (
            confidence < self._min_gesture_confidence
        ):
            gesture = HandGesture.NONE
        return GestureReading(
            gesture=gesture,
            confidence=confidence,
            raw_label=raw_label,
            hand_index=hand_index,
        )

    # --- Lifetime --------------------------------------------------------------------------------

    def close(self) -> None:
        closer = getattr(self._recognizer, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> ThumbGestureRecognizer:
        return self

    def __exit__(
        self, exc_type: object, exc_val: object, exc_tb: object
    ) -> Literal[False]:
        self.close()
        return False
