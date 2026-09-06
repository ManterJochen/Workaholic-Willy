"""Hand and gesture detection (MediaPipe): palm centre in 2-D and 3-D, thumbs up / thumbs down.

Optional and standalone: nothing in the grasp pipeline imports this package, and mediapipe is
pinned in `requirements.txt`. Importing this module works without mediapipe installed.
Constructing a detector is what refuses, with a message naming the config key and the model
download.

Layout
------
`landmarks`      pure geometry: the 21-point layout and the palm centre. No MediaPipe, no camera.
`types`          the value objects that cross the boundary out of inference.
`model_files`    the optional-extra guard and the fail-closed `.task` resolver.
`palm_detector`  MediaPipe hand landmarker -> palm centres, in pixels.
`gestures`       MediaPipe canned classifier -> thumbs-up / thumbs-down, with the palm centre.
`hand_finder`    pixels + depth + calibration -> millimetres in the robot base frame.
`factory`        the config readers: `models.handdetect` / `models.gesturedetect`.
"""

from src.models.handdetection.factory import (
    build_gesture_recognizer,
    build_hand_finder,
    build_palm_detector,
)
from src.models.handdetection.gestures import (
    ThumbGestureRecognizer,
    normalise_gesture_label,
    to_hand_gesture,
)
from src.models.handdetection.hand_finder import HandFinder, HandObserver
from src.models.handdetection.landmarks import (
    PALM_LANDMARKS,
    HandLandmark,
    calculate_palm_center,
    draw_hand_landmarks,
)
from src.models.handdetection.model_files import (
    MEDIAPIPE_AVAILABLE,
    require_mediapipe,
    resolve_model_file,
)
from src.models.handdetection.palm_detector import PalmDetector
from src.models.handdetection.types import (
    GestureReading,
    Handedness,
    HandGesture,
    HandObservation,
    HandPosition3D,
    LocatedHand,
    PalmDetection,
)

__all__ = [
    "MEDIAPIPE_AVAILABLE",
    "PALM_LANDMARKS",
    "GestureReading",
    "HandFinder",
    "HandGesture",
    "HandLandmark",
    "HandObservation",
    "HandObserver",
    "HandPosition3D",
    "Handedness",
    "LocatedHand",
    "PalmDetection",
    "PalmDetector",
    "ThumbGestureRecognizer",
    "build_gesture_recognizer",
    "build_hand_finder",
    "build_palm_detector",
    "calculate_palm_center",
    "draw_hand_landmarks",
    "normalise_gesture_label",
    "require_mediapipe",
    "resolve_model_file",
    "to_hand_gesture",
]
