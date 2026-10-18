"""Every number the hand detectors run at reaches something, at the owner's cell's value.

WHY THIS FILE EXISTS. The `models.handdetect` and `models.gesturedetect` blocks were deleted once (2026-08-22) because
nothing read them, restored with this guard, and left the config for good on 2026-10-17 (the owner's cleanup): every
key they held was the bundle's path or a number no cell changed. The numbers are fixed in
`src/models/handdetection/constants.py` at the cell's values, and the bundles are the weights catalogue's, read under
`models.weights_root`. A fixed number that reaches nothing is the same dead weight a key was, so this asserts, per
number, that it arrives at MediaPipe's options (or, for the two depth-sampling numbers, at `HandFinder`), and that the
values are the cell's.

No `.task` bundle and no inference is involved: MediaPipe's option classes are real, its task constructors are
captured, and the bundles are empty files where the weights folder keeps them.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

from src.models.handdetection import constants
from src.models.handdetection.model_files import MEDIAPIPE_AVAILABLE
from src.models.weights import fixed_file

#: Each fixed number of the palm detector -> how to read it back off the captured MediaPipe options.
_HAND_READERS: dict[str, Callable[[Any], Any]] = {
    "MAX_HANDS": lambda options: options.num_hands,
    "DETECTION_THRESHOLD": lambda options: options.min_hand_detection_confidence,
    "TRACKING_THRESHOLD": lambda options: options.min_tracking_confidence,
    "PRESENCE_THRESHOLD": lambda options: options.min_hand_presence_confidence,
}
#: The gesture recogniser's: the same four, and the floor a gesture must clear.
_GESTURE_READERS: dict[str, Callable[[Any], Any]] = {
    **_HAND_READERS,
    "MIN_GESTURE_CONFIDENCE": lambda options: options.canned_gesture_classifier_options.score_threshold,
}
#: What the owner's cell's `models.handdetect` and `models.gesturedetect` held on 2026-10-14, the schema's defaults.
_THE_CELLS = {"MAX_HANDS": 2, "DETECTION_THRESHOLD": 0.5, "TRACKING_THRESHOLD": 0.5, "PRESENCE_THRESHOLD": 0.5,
              "MIN_GESTURE_CONFIDENCE": 0.5, "PALM_PATCH_RADIUS_PX": 15, "MIN_DEPTH_SAMPLES": 20}


class _Captured:
    """Stands in for a MediaPipe task and remembers the options it was built with."""

    def __init__(self, options: Any) -> None:
        self.options = options

    def close(self) -> None:
        return None


class TheCellsNumbersTests(unittest.TestCase):

    def test_every_number_is_the_owners_cells(self) -> None:
        for name, value in _THE_CELLS.items():
            with self.subTest(name):
                self.assertEqual(value, getattr(constants, name))

    def test_the_bundles_are_the_weights_catalogues_under_the_weights_root(self) -> None:
        self.assertEqual(Path("R/mediapipe/hand_landmarker.task"), fixed_file(constants.HAND_BUNDLE, root="R"))
        self.assertEqual(Path("R/mediapipe/gesture_recognizer.task"), fixed_file(constants.GESTURE_BUNDLE, root="R"))


@unittest.skipUnless(MEDIAPIPE_AVAILABLE, "mediapipe is an optional extra")
class _WiringCase(unittest.TestCase):
    """Shared machinery: an empty bundle where the weights folder keeps it, and one task constructor captured."""

    task_class_name = ""
    bundle = ""

    def setUp(self) -> None:
        from mediapipe.tasks.python import vision

        self.vision = vision
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.file = fixed_file(self.bundle, root=self.root)
        self.file.parent.mkdir(parents=True)
        self.file.write_bytes(b"")
        self.seen: list[Any] = []
        task_class = getattr(vision, self.task_class_name)
        original = task_class.__dict__["create_from_options"]
        self.addCleanup(setattr, task_class, "create_from_options", original)

        def _create_from_options(options: Any) -> _Captured:
            self.seen.append(options)
            return _Captured(options)

        task_class.create_from_options = staticmethod(_create_from_options)

    def options(self) -> Any:
        raise NotImplementedError

    def check(self, readers: dict[str, Callable[[Any], Any]]) -> None:
        options = self.options()
        self.assertEqual(self.file.resolve(), Path(options.base_options.model_asset_path).resolve(),
                         "the bundle under models.weights_root reaches MediaPipe's base options")
        for name, reader in readers.items():
            with self.subTest(name):
                self.assertEqual(getattr(constants, name), reader(options))
        self.assertEqual(self.vision.RunningMode.VIDEO, options.running_mode,
                         "IMAGE mode would leave the tracking threshold inert")


class ThePalmDetectorRunsAtTheCellsNumbersTests(_WiringCase):
    task_class_name = "HandLandmarker"
    bundle = constants.HAND_BUNDLE

    def options(self) -> Any:
        from src.models.handdetection.factory import build_palm_detector

        build_palm_detector(weights_root=self.root.as_posix())
        return self.seen[-1]

    def test_each_number_and_the_bundle_arrive(self) -> None:
        self.check(_HAND_READERS)


class TheGestureRecogniserRunsAtTheCellsNumbersTests(_WiringCase):
    task_class_name = "GestureRecognizer"
    bundle = constants.GESTURE_BUNDLE

    def options(self) -> Any:
        from src.models.handdetection.factory import build_gesture_recognizer

        build_gesture_recognizer(weights_root=self.root.as_posix())
        return self.seen[-1]

    def test_each_number_and_the_bundle_arrive(self) -> None:
        self.check(_GESTURE_READERS)

    def test_the_gesture_floor_is_enforced_on_our_side_too(self) -> None:
        """MediaPipe's `score_threshold` filters categories; this checks the NONE-vs-THUMB_UP call.

        Both are needed. The classifier's floor keeps weak categories out of the result; ours decides what a
        surviving-but-weak thumbs-up means, and a confirmation gesture firing below the floor is worse than one that
        does not fire.
        """
        from src.models.handdetection.factory import build_gesture_recognizer
        from src.models.handdetection.types import HandGesture

        recognizer = build_gesture_recognizer(weights_root=self.root.as_posix())

        class _Category:
            category_name = "Thumb_Up"
            score = 0.4

        reading = recognizer._read_gesture([_Category()], 0)
        self.assertIs(reading.gesture, HandGesture.NONE)
        self.assertEqual(reading.raw_label, "Thumb_Up", "the raw label must survive the demotion")
        self.assertAlmostEqual(reading.confidence, 0.4)


@unittest.skipUnless(MEDIAPIPE_AVAILABLE, "mediapipe is an optional extra")
class AMissingBundleTests(unittest.TestCase):

    def test_a_bundle_the_weights_folder_lacks_is_refused_naming_the_fetch(self) -> None:
        from src.models.handdetection.factory import build_gesture_recognizer, build_palm_detector

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            for build in (build_palm_detector, build_gesture_recognizer):
                with self.subTest(build.__name__), self.assertRaises(FileNotFoundError) as caught:
                    build(weights_root=tmp)
                self.assertIn("fetch.py --mediapipe", str(caught.exception))
                self.assertIn("models.weights_root", str(caught.exception))


class TheDepthNumbersReachTheFinderTests(unittest.TestCase):
    """The two numbers that are ours, not MediaPipe's. They travel through `build_hand_finder`."""

    class _Provider:
        @property
        def rig_ids(self) -> list[str]:
            return []

    def test_the_patch_radius_and_sample_floor_arrive_at_the_cells_values(self) -> None:
        from src.models.handdetection.factory import build_hand_finder

        finder = build_hand_finder(
            provider=self._Provider(),  # type: ignore[arg-type]
            transforms={},
            observer=object(),  # type: ignore[arg-type]
        )
        self.assertEqual(constants.PALM_PATCH_RADIUS_PX, finder.palm_patch_radius_px)
        self.assertEqual(constants.MIN_DEPTH_SAMPLES, finder.min_depth_samples)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
