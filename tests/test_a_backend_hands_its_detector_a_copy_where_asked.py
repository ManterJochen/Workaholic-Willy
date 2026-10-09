"""A perception backend hands its detector a copy of the image where asked, and its segmenter the image itself.

The task's own places are hidden from the detector where the cell says so (``robot.grasping.hide_own_places``, the
owner, 2026-10-08 night): the camera source paints the regions a task keeps out over a copy of the frame and hands it
as ``perceive(image, prompt, detect_on=copy)`` (``src/robot/perception/realsense_source.py``). A backend that takes it
says so (``detects_on_a_copy``). The two-stage backend's detector reads the copy, and its segmenter, and so every mask,
the real image; the guarded VLM backend and the routed backend hand it on to the route that grounds, where that route
takes it, and a route that does not reads the image as it always did. Without ``detect_on`` every backend is asked as
before.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.models.perception_backend import TwoStageBackend

_IMAGE = np.full((40, 60, 3), 200, dtype=np.uint8)
_COPY = np.zeros((40, 60, 3), dtype=np.uint8)


class _Detector:
    """Boxes one part, and writes down every image it was handed."""

    def __init__(self) -> None:
        self.seen: list[np.ndarray] = []

    def detect_all(self, bgr: Any, prompt: str) -> list[Any]:
        self.seen.append(np.asarray(bgr))
        return [type("Det", (), {"box": [2.0, 3.0, 12.0, 15.0], "label": prompt, "score": 0.9})()]


class _Segmenter:
    """Cuts every box, and writes down every image it was handed."""

    def __init__(self) -> None:
        self.seen: list[np.ndarray] = []

    def segment_detection(self, bgr: Any, det: Any) -> Any:
        self.seen.append(np.asarray(bgr))
        return type("Seg", (), {"mask": np.ones(np.asarray(bgr).shape[:2], dtype=np.uint8), "label": det.label})()


def _two_stage() -> tuple[TwoStageBackend, _Detector, _Segmenter]:
    detector, segmenter = _Detector(), _Segmenter()
    return TwoStageBackend(detector=detector, segmenter=segmenter), detector, segmenter


class _Plain:
    """A backend as every backend was: one image for its detector and its segmenter."""

    def __init__(self) -> None:
        self.asked: list[tuple[Any, ...]] = []

    def perceive(self, *handed: Any) -> tuple[Any, ...]:
        self.asked.append(handed)
        return ()


class TheTwoStageBackendTests(unittest.TestCase):
    def test_its_detector_reads_the_copy_and_its_segmenter_the_image(self) -> None:
        backend, detector, segmenter = _two_stage()

        objects = backend.perceive(_IMAGE, "grey cube", detect_on=_COPY)

        self.assertIs(True, backend.detects_on_a_copy)
        self.assertEqual(1, len(objects))
        self.assertIs(_COPY, detector.seen[0])
        self.assertIs(_IMAGE, segmenter.seen[0])

    def test_without_a_copy_both_read_the_image_as_before(self) -> None:
        backend, detector, segmenter = _two_stage()

        backend.perceive(_IMAGE, "grey cube")

        self.assertIs(_IMAGE, detector.seen[0])
        self.assertIs(_IMAGE, segmenter.seen[0])


class TheGuardedVlmBackendTests(unittest.TestCase):
    def test_it_hands_the_copy_to_the_vlm_route(self) -> None:
        from src.models.vlm import GuardedVlmBackend

        vlm, detector, segmenter = _two_stage()
        guarded = GuardedVlmBackend(vlm=vlm, model_id="Qwen/Qwen3-VL-8B-Instruct")

        guarded.perceive(_IMAGE, "grey cube", detect_on=_COPY)

        self.assertIs(True, guarded.detects_on_a_copy)
        self.assertIs(_COPY, detector.seen[0])
        self.assertIs(_IMAGE, segmenter.seen[0])

    def test_a_route_that_takes_no_copy_is_asked_as_it_always_was(self) -> None:
        from src.models.vlm import GuardedVlmBackend

        plain = _Plain()
        guarded = GuardedVlmBackend(vlm=plain, model_id="Qwen/Qwen3-VL-8B-Instruct")

        guarded.perceive(_IMAGE, "grey cube", detect_on=_COPY)

        self.assertIs(False, guarded.detects_on_a_copy)
        self.assertEqual([(_IMAGE, "grey cube")], plain.asked)


class TheRoutedBackendTests(unittest.TestCase):
    def test_it_hands_the_copy_to_the_route_it_chose_where_that_route_takes_it(self) -> None:
        from src.models.routed_backend import RoutedPerceptionBackend

        simple, detector, segmenter = _two_stage()
        plain = _Plain()
        routed = RoutedPerceptionBackend(simple_factory=lambda: simple, vlm_factory=lambda: plain)

        routed.perceive(_IMAGE, "grey cube", detect_on=_COPY)
        routed.perceive(_IMAGE, "der graue Würfel", detect_on=_COPY)  # German: the VLM route

        self.assertIs(True, routed.detects_on_a_copy)
        self.assertIs(_COPY, detector.seen[0], "the simple route's detector read the image, not the copy")
        self.assertIs(_IMAGE, segmenter.seen[0])
        self.assertEqual(1, len(plain.asked))
        self.assertEqual(2, len(plain.asked[0]), "a route that takes no copy was handed one")


if __name__ == "__main__":
    unittest.main()
