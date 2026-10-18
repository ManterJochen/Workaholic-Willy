"""A perception backend hands its detector a copy of the image where asked, and its segmenter the image itself.

The task's own places are hidden from the detector in every cell built from a tree (the owner, 2026-10-08 night): the
camera source paints the regions a task keeps out over a copy of the frame and hands it
as ``perceive(image, prompt, detect_on=copy)`` (``src/robot/perception/realsense_source.py``). A backend that takes it
says so (``detects_on_a_copy``). The two-stage backend's detector reads the copy, and its segmenter, and so every mask,
the real image; it is every stack a cell builds since the guarded VLM backend and the routed backend left with
GroundingDINO (2026-10-17). Without ``detect_on`` it is asked as before.
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


if __name__ == "__main__":
    unittest.main()
