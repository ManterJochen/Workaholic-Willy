"""Every field of `models.handdetect` / `models.gesturedetect` must reach something.

WHY THIS FILE EXISTS. Both blocks were DELETED from the schema on 2026-08-22 because a trace found
that nothing read them: the config classes survived only as type annotations on a constructor no
code ever called. Restoring them without this test would restore that situation exactly, and the
next sweep would delete them again.

So this does not hand-check the fields it happens to remember. It ENUMERATES them out of the pydantic
model and asserts, per field, that changing the value changes what reaches MediaPipe (or, for the two
depth-sampling knobs, what reaches `HandFinder`). A field added tomorrow is covered the day its
schema line lands: it will fail here until someone wires it, and the failure names it.

No `.task` bundle and no inference is involved -- MediaPipe's option classes are real, its task
constructors are captured. The point is the WIRING, and a model file would only slow it down.
"""

from __future__ import annotations

import unittest
from typing import Any, Callable

from src.config.schema.models import GestureDetectConfig, HandDetectConfig
from src.models.handdetection.model_files import MEDIAPIPE_AVAILABLE

#: Schema field -> how to read the value back off the captured MediaPipe options object.
#: `None` means "not a MediaPipe option" and is checked against `HandFinder` instead, below.
_HANDDETECT_READERS: dict[str, Callable[[Any], Any] | None] = {
    "model_path": lambda options: options.base_options.model_asset_path,
    "max_hands": lambda options: options.num_hands,
    "threshold": lambda options: options.min_hand_detection_confidence,
    "tracking_threshold": lambda options: options.min_tracking_confidence,
    "presence_threshold": lambda options: options.min_hand_presence_confidence,
    "palm_patch_radius_px": None,
    "min_depth_samples": None,
}

_GESTUREDETECT_READERS: dict[str, Callable[[Any], Any] | None] = {
    "model_path": lambda options: options.base_options.model_asset_path,
    "max_hands": lambda options: options.num_hands,
    "threshold": lambda options: options.min_hand_detection_confidence,
    "tracking_threshold": lambda options: options.min_tracking_confidence,
    "presence_threshold": lambda options: options.min_hand_presence_confidence,
    "min_gesture_confidence": (
        lambda options: options.canned_gesture_classifier_options.score_threshold
    ),
}

#: A value that differs from the schema default for each field type, so the differential has an edge
#: to detect. `model_path` is special-cased: it must be a file that EXISTS, because the resolver is
#: fail-closed by design and a nonexistent path would never reach MediaPipe at all.
_PROBES: dict[str, Any] = {
    "max_hands": 4,
    "threshold": 0.77,
    "tracking_threshold": 0.66,
    "presence_threshold": 0.44,
    "min_gesture_confidence": 0.91,
    "palm_patch_radius_px": 33,
    "min_depth_samples": 7,
}


class _Captured:
    """Stands in for a MediaPipe task and remembers the options it was built with."""

    def __init__(self, options: Any) -> None:
        self.options = options

    def close(self) -> None:
        return None


def _capture(monkey: Any, class_name: str) -> list[Any]:
    """Replace one MediaPipe task's `create_from_options` and collect what it is given."""
    from mediapipe.tasks.python import vision

    seen: list[Any] = []
    task_class = getattr(vision, class_name)

    def _create_from_options(options: Any) -> _Captured:
        seen.append(options)
        return _Captured(options)

    monkey(task_class, "create_from_options", staticmethod(_create_from_options))
    return seen


@unittest.skipUnless(MEDIAPIPE_AVAILABLE, "mediapipe is an optional extra")
class _WiringCase(unittest.TestCase):
    """Shared machinery: patch a task constructor for the duration of one test."""

    task_class_name = ""

    def setUp(self) -> None:
        from mediapipe.tasks.python import vision

        self._originals: list[tuple[Any, str, Any]] = []
        self.seen = _capture(self._patch, self.task_class_name)
        self.vision = vision

    def _patch(self, target: Any, name: str, value: Any) -> None:
        self._originals.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def tearDown(self) -> None:
        for target, name, original in reversed(self._originals):
            setattr(target, name, original)

    def _options_for(self, **overrides: Any) -> Any:
        raise NotImplementedError


class EveryHandDetectFieldReachesMediaPipeTests(_WiringCase):
    task_class_name = "HandLandmarker"

    def _options_for(self, **overrides: Any) -> Any:
        from src.models.handdetection.factory import build_palm_detector

        config = HandDetectConfig(model_path=__file__, **overrides)
        build_palm_detector(config)
        return self.seen[-1]

    def test_the_reader_table_covers_the_schema(self) -> None:
        """The guard's own completeness check: a new schema field must be classified here.

        Without this, adding a field and forgetting to wire it would pass silently -- the loop below
        only iterates fields the table already knows about.
        """
        self.assertEqual(
            set(HandDetectConfig.model_fields),
            set(_HANDDETECT_READERS),
            "a `models.handdetect` field is missing from this guard's reader table -- classify it "
            "as a MediaPipe option or as a HandFinder knob, and wire it",
        )

    def test_each_mediapipe_field_arrives_with_the_value_it_was_given(self) -> None:
        for field, reader in _HANDDETECT_READERS.items():
            if reader is None:
                continue
            with self.subTest(field=field):
                if field == "model_path":
                    options = self._options_for()
                    self.assertTrue(
                        str(reader(options)).endswith("test_hand_detection_wiring.py"),
                        "model_path did not reach MediaPipe's base options",
                    )
                    continue
                probe = _PROBES[field]
                options = self._options_for(**{field: probe})
                self.assertEqual(
                    reader(options), probe,
                    f"models.handdetect.{field} was set to {probe!r} and did not arrive",
                )

    def test_the_running_mode_is_video_so_tracking_threshold_can_matter(self) -> None:
        """IMAGE mode would make `tracking_threshold` inert -- a wired key that still does nothing."""
        options = self._options_for()
        self.assertEqual(options.running_mode, self.vision.RunningMode.VIDEO)


class EveryGestureDetectFieldReachesMediaPipeTests(_WiringCase):
    task_class_name = "GestureRecognizer"

    def _options_for(self, **overrides: Any) -> Any:
        from src.models.handdetection.factory import build_gesture_recognizer

        config = GestureDetectConfig(model_path=__file__, **overrides)
        build_gesture_recognizer(config)
        return self.seen[-1]

    def test_the_reader_table_covers_the_schema(self) -> None:
        self.assertEqual(
            set(GestureDetectConfig.model_fields),
            set(_GESTUREDETECT_READERS),
            "a `models.gesturedetect` field is missing from this guard's reader table",
        )

    def test_each_field_arrives_with_the_value_it_was_given(self) -> None:
        for field, reader in _GESTUREDETECT_READERS.items():
            if reader is None:
                continue
            with self.subTest(field=field):
                if field == "model_path":
                    options = self._options_for()
                    self.assertTrue(
                        str(reader(options)).endswith("test_hand_detection_wiring.py")
                    )
                    continue
                probe = _PROBES[field]
                options = self._options_for(**{field: probe})
                self.assertEqual(
                    reader(options), probe,
                    f"models.gesturedetect.{field} was set to {probe!r} and did not arrive",
                )

    def test_the_gesture_floor_is_enforced_on_our_side_too(self) -> None:
        """MediaPipe's `score_threshold` filters categories; this checks the NONE-vs-THUMB_UP call.

        Both are needed. The classifier's floor keeps weak categories out of the result; ours decides
        what a surviving-but-weak thumbs-up means -- and a confirmation gesture firing at 0.2 is
        worse than one that does not fire.
        """
        from src.models.handdetection.factory import build_gesture_recognizer
        from src.models.handdetection.types import HandGesture

        recognizer = build_gesture_recognizer(
            GestureDetectConfig(model_path=__file__, min_gesture_confidence=0.8)
        )

        class _Category:
            category_name = "Thumb_Up"
            score = 0.5

        reading = recognizer._read_gesture([_Category()], 0)
        self.assertIs(reading.gesture, HandGesture.NONE)
        self.assertEqual(reading.raw_label, "Thumb_Up", "the raw label must survive the demotion")
        self.assertAlmostEqual(reading.confidence, 0.5)


class TheDepthKnobsReachTheFinderTests(unittest.TestCase):
    """The two fields that are ours, not MediaPipe's. They travel through `build_hand_finder`."""

    class _Provider:
        @property
        def rig_ids(self) -> list[str]:
            return []

    def test_the_patch_radius_and_sample_floor_arrive(self) -> None:
        from src.models.handdetection.factory import build_hand_finder

        config = HandDetectConfig(
            model_path=__file__, palm_patch_radius_px=33, min_depth_samples=7
        )
        finder = build_hand_finder(
            config,
            provider=self._Provider(),  # type: ignore[arg-type]
            transforms={},
            observer=object(),  # type: ignore[arg-type]
        )
        self.assertEqual(finder.palm_patch_radius_px, 33)
        self.assertEqual(finder.min_depth_samples, 7)

    def test_the_defaults_match_the_schema(self) -> None:
        """A default that drifts between schema and code is a silent behaviour change."""
        from src.models.handdetection.factory import build_hand_finder

        config = HandDetectConfig(model_path=__file__)
        finder = build_hand_finder(
            config,
            provider=self._Provider(),  # type: ignore[arg-type]
            transforms={},
            observer=object(),  # type: ignore[arg-type]
        )
        self.assertEqual(finder.palm_patch_radius_px, config.palm_patch_radius_px)
        self.assertEqual(finder.min_depth_samples, config.min_depth_samples)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
