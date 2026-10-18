"""Shared constants for the hand-detection package.

Three kinds of constant here, each written once rather than re-typed at every call site where they
would drift apart silently.

* Log files, following the `models/constants.py` pattern of one directory and one file per
  subsystem rather than the robot package's aggregate file. Hand detection runs independently of
  the grasp pipeline, so its lines stay on their own.
* The model-download URLs, which appear in an error message. Nothing here downloads anything:
  `scripts/model_weights/fetch.py --mediapipe` writes the `.task` bundles under the weights folder
  (`models.weights_root`), from these same URLs (the weights catalogue's, `src/models/weights.py`).
  The URL is in the message so that "file not found" does not send the reader to a search engine.
* How the detectors run: the numbers the owner's cell ran (its `models.handdetect` and
  `models.gesturedetect` of 2026-10-14, the schema's defaults), fixed since 2026-10-17, when the two
  blocks left the config.
"""

from __future__ import annotations

from typing import Final

from src.models.constants import MODELS_LOG_DIR
from src.models.weights import BUNDLES

__all__ = [
    "DETECTION_THRESHOLD",
    "GESTURE_BUNDLE",
    "GESTURE_MODEL_URL",
    "GESTURE_RECOGNIZER_LOG_FILE",
    "HAND_BUNDLE",
    "HAND_FINDER_LOG_FILE",
    "HAND_LANDMARK_MODEL_URL",
    "MAX_HANDS",
    "MIN_DEPTH_SAMPLES",
    "MIN_GESTURE_CONFIDENCE",
    "MODELS_LOG_DIR",
    "PALM_DETECTOR_LOG_FILE",
    "PALM_PATCH_RADIUS_PX",
    "PRESENCE_THRESHOLD",
    "TRACKING_THRESHOLD",
]

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
#: The 3-D search across camera rigs. Its own file because its lines are about geometry: which rig
#: saw the hand, what depth came back, why a detection was discarded.
HAND_FINDER_LOG_FILE: Final[str] = "hand_finder.log"

#: The 2-D landmark detector. Separate from the finder above because "MediaPipe saw no hand" and
#: "the hand was seen but had no usable depth" are different failures with different fixes, and
#: mixing them sends an operator to re-aim a camera that was already aimed correctly.
PALM_DETECTOR_LOG_FILE: Final[str] = "palm_detector.log"

#: The canned gesture classifier. Its own file because gesture lines are a confirmation trail, and
#: "did the cell see the thumbs-up?" should be one grep rather than a scroll through landmarks.
GESTURE_RECOGNIZER_LOG_FILE: Final[str] = "gesture_recognizer.log"

# ----------------------------------------------------------------------
# Where the models come from
# ----------------------------------------------------------------------
_BY_KEY: Final = {bundle.key: bundle for bundle in BUNDLES}

#: The weights catalogue's key of the hand-landmark bundle: 21 landmarks per hand, no gesture head. The
#: smaller and faster of the two, and enough when only the palm centre is wanted.
HAND_BUNDLE: Final[str] = "mediapipe-hand"
HAND_LANDMARK_MODEL_URL: Final[str] = _BY_KEY[HAND_BUNDLE].url

#: The weights catalogue's key of the canned gesture bundle. It embeds a hand landmarker, so its results
#: carry `hand_landmarks` too, which is why `ThumbGestureRecognizer` returns palm detections alongside the
#: gesture instead of running a second model over the same frame.
GESTURE_BUNDLE: Final[str] = "mediapipe-gesture"
GESTURE_MODEL_URL: Final[str] = _BY_KEY[GESTURE_BUNDLE].url

# ----------------------------------------------------------------------
# How the detectors run: the owner's cell's numbers
# ----------------------------------------------------------------------
#: Upper bound on hands tracked at once (MediaPipe's ``num_hands``). The 3-D HandFinder needs exactly one
#: hand and skips a rig that shows more, so this is deliberately not 1: the second hand has to be seen to
#: be refused.
MAX_HANDS: Final[int] = 2
#: MediaPipe's confidence floors, each 0 to 1: whether a hand is in the frame at all
#: (``min_hand_detection_confidence``), whether the hand already tracked still is
#: (``min_hand_presence_confidence``), and the frame-to-frame association (``min_tracking_confidence``,
#: meaningful because the detectors run in VIDEO mode).
DETECTION_THRESHOLD: Final[float] = 0.5
PRESENCE_THRESHOLD: Final[float] = 0.5
TRACKING_THRESHOLD: Final[float] = 0.5
#: The score a thumbs-up must reach to be reported as one; below it the answer is no gesture, not a
#: low-confidence thumbs-up. Passed to the canned classifier as its ``score_threshold`` and checked again.
MIN_GESTURE_CONFIDENCE: Final[float] = 0.5
#: Not a MediaPipe number. The disc around the palm centre whose median depth becomes the hand's distance,
#: in pixels: too small and one dropout on skin loses the hand, too large and it starts averaging in
#: whatever is behind the hand.
PALM_PATCH_RADIUS_PX: Final[int] = 15
#: How many valid depth pixels that disc must hold before its median is believed. Below this the rig
#: contributes no distance: a median over three surviving pixels is a number, not a measurement.
MIN_DEPTH_SAMPLES: Final[int] = 20
