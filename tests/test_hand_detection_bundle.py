"""Checks that need the real MediaPipe `.task` bundles, and skip cleanly without them.

`test_hand_detection.py` covers the geometry and `test_hand_detection_wiring.py` covers the config
wiring; both run everywhere because neither needs a model file. This one is the other half: the
facts that live inside the downloaded bundle, which no amount of unit testing can assert.

THE ONE THAT MATTERS is the label vocabulary. `to_hand_gesture` maps `Thumb_Up` / `Thumb_Down` and
sends everything else to `OTHER` -- so if a future revision of the canned classifier renames a
category, thumbs-up silently stops being recognised and starts reporting as "some other gesture".
Nothing else in the suite would notice: the mapping is total, so it has no failure mode to observe.
Reading the labels out of the bundle is what turns that silent degradation into a red test.

The bundles are an operator download (`assets/models/mediapipe/`, gitignored). Absent, everything
here skips -- CI has no model files and must not fail for that reason.

Measured 2026-08-23 against the `float16/latest` bundles, on five of MediaPipe's own test images:
thumbs_up -> thumb_up (0.73), thumbs_down -> thumb_down (0.77), victory -> other (raw=Victory),
pointing_up -> other (raw=Pointing_Up), and a two-hand frame -> both hands found, both gesture NONE.
The palm centre landed inside the palm in every one. Those images are NOT vendored: they are third
party with unclear licensing, and this repo's licensing boundary is fail-closed.
"""

from __future__ import annotations

import io
import unittest
import zipfile
from pathlib import Path

import numpy as np

from src.config.schema.models import GestureDetectConfig, HandDetectConfig
from src.models.handdetection.gestures import to_hand_gesture
from src.models.handdetection.model_files import MEDIAPIPE_AVAILABLE
from src.models.handdetection.types import HandGesture

_BUNDLES = Path("assets/models/mediapipe")
_LANDMARKER = _BUNDLES / "hand_landmarker.task"
_GESTURE = _BUNDLES / "gesture_recognizer.task"

_HAVE_BUNDLES = _LANDMARKER.is_file() and _GESTURE.is_file()
_REASON = (
    "the MediaPipe .task bundles are an operator download; run "
    "`python -m src.models.handdetection --check`"
)

#: What the canned classifier actually emits, read out of the bundle on 2026-08-23. Two of these are
#: promised by this package and five are deliberately not.
_EXPECTED_LABELS = (
    "None", "Closed_Fist", "Open_Palm", "Pointing_Up",
    "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou",
)


def _classifier_bytes() -> bytes:
    """The canned classifier's tflite, two zip layers down. Its labels ride along as plain ASCII."""
    with zipfile.ZipFile(_GESTURE) as outer:
        inner = outer.read("hand_gesture_recognizer.task")
    with zipfile.ZipFile(io.BytesIO(inner)) as bundle:
        return bundle.read("canned_gesture_classifier.tflite")


@unittest.skipUnless(_HAVE_BUNDLES and MEDIAPIPE_AVAILABLE, _REASON)
class TheBundleStillSpellsTheGesturesTheWayWeMapThemTests(unittest.TestCase):
    def test_both_gestures_this_package_promises_are_in_the_model(self) -> None:
        """If this reds, the classifier was revised and `_GESTURE_BY_LABEL` has to follow."""
        blob = _classifier_bytes()
        for label in ("Thumb_Up", "Thumb_Down"):
            with self.subTest(label=label):
                self.assertIn(
                    label.encode(), blob,
                    f"{label!r} is no longer a category in the canned classifier -- "
                    f"thumbs-up detection has silently become HandGesture.OTHER",
                )

    def test_the_whole_vocabulary_is_still_the_one_the_mapping_was_written_against(self) -> None:
        blob = _classifier_bytes()
        missing = [label for label in _EXPECTED_LABELS if label.encode() not in blob]
        self.assertEqual(
            missing, [],
            "the canned classifier's categories changed; re-read them and revisit which ones this "
            "package promises",
        )

    def test_every_category_maps_to_something_deliberate(self) -> None:
        """Two mapped, one NONE, the rest OTHER -- asserted per label so a drift names itself."""
        expected = {
            "Thumb_Up": HandGesture.THUMB_UP,
            "Thumb_Down": HandGesture.THUMB_DOWN,
            "None": HandGesture.NONE,
        }
        for label in _EXPECTED_LABELS:
            with self.subTest(label=label):
                self.assertIs(to_hand_gesture(label), expected.get(label, HandGesture.OTHER))


@unittest.skipUnless(_HAVE_BUNDLES and MEDIAPIPE_AVAILABLE, _REASON)
class TheModelsActuallyLoadAndRunTests(unittest.TestCase):
    """The construction path end to end: config -> factory -> MediaPipe graph -> a result.

    A blank frame is enough. What is under test is that every option this package passes is accepted
    by the real graph and that a result comes back in the shape the converters expect -- not whether
    MediaPipe can find a hand, which is MediaPipe's business and was measured separately.
    """

    def _blank(self) -> np.ndarray:
        return np.zeros((240, 320, 3), dtype=np.uint8)

    def test_the_landmark_model_builds_from_config_and_returns_no_hands_on_a_blank_frame(self) -> None:
        from src.models.handdetection.factory import build_palm_detector

        with build_palm_detector(HandDetectConfig()) as detector:
            self.assertEqual(detector.observe(self._blank()), [])

    def test_the_gesture_model_builds_from_config_and_returns_no_hands_on_a_blank_frame(self) -> None:
        from src.models.handdetection.factory import build_gesture_recognizer

        with build_gesture_recognizer(GestureDetectConfig()) as recognizer:
            self.assertEqual(recognizer.observe(self._blank()), [])

    def test_consecutive_frames_do_not_trip_the_monotonic_timestamp_requirement(self) -> None:
        """VIDEO mode rejects a repeated timestamp, and two frames can land in one millisecond.

        This is the assertion behind `_next_timestamp_ms`: ten calls in a tight loop is exactly the
        case the previous implementation's invented `+= 33` was hiding.
        """
        from src.models.handdetection.factory import build_palm_detector

        with build_palm_detector(HandDetectConfig()) as detector:
            for _ in range(10):
                detector.observe(self._blank())  # must not raise


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
