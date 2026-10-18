"""Cutting the boxes of parts a task kept asks no detector, and never loads the VLM (``segment_boxes``).

Following the parts a task kept is SAM2 on their boxes alone (``src/robot/perception/kept_scene.py``): the two-stage
backend cuts each box as a detection of score 1.0 under its label, and says a box it could not cut rather than counting
it, so the source grounds the frame instead and only that grounding counts its own failures. Every stack a cell builds
is that backend since the guarded VLM backend and the router left with GroundingDINO (2026-10-17), so the VLM is never
loaded to cut a box.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.models.perception_backend import PerceivedObject, SegmentedBoxes, TwoStageBackend

_IMAGE = np.zeros((40, 60, 3), dtype=np.uint8)
_BOXES = (((2.0, 3.0, 12.0, 15.0), "grey cube"), ((30.0, 4.0, 44.0, 20.0), "grey cube"))


class _Detector:
    """A detector that must not be asked: every call is written down and raises, as a VLM that cannot load does."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def detect_all(self, bgr: Any, prompt: str) -> list[Any]:
        self.calls.append(prompt)
        raise RuntimeError("the VLM was asked to load")

    def ensure_loaded(self) -> None:  # pragma: no cover - the test fails if it is reached
        self.calls.append("load")
        raise AssertionError("the VLM was loaded to cut boxes")


class _Segmenter:
    """Cuts a box as its own mask; ``fail_on`` are the box numbers it raises on."""

    def __init__(self, fail_on: tuple[int, ...] = ()) -> None:
        self.fail_on = fail_on
        self.seen: list[Any] = []

    def segment_detection(self, bgr: Any, det: Any) -> Any:
        number = len(self.seen)
        self.seen.append(det)
        if number in self.fail_on:
            raise RuntimeError("SAM2 out of memory")
        mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        x0, y0, x1, y1 = (int(value) for value in det.box)
        mask[y0:y1, x0:x1] = 1
        return type("Seg", (), {"mask": mask, "label": det.label})()


class TheTwoStageBackendCutsBoxesAloneTests(unittest.TestCase):
    def test_each_box_is_cut_under_its_label_and_the_detector_is_never_asked(self) -> None:
        detector, segmenter = _Detector(), _Segmenter()
        backend = TwoStageBackend(detector=detector, segmenter=segmenter)

        cut = backend.segment_boxes(_IMAGE, _BOXES)

        self.assertEqual([], detector.calls)
        self.assertEqual((), cut.failed)
        self.assertEqual(2, len(cut.objects))
        self.assertTrue(all(isinstance(obj, PerceivedObject) for obj in cut.objects))
        self.assertEqual([[2.0, 3.0, 12.0, 15.0], [30.0, 4.0, 44.0, 20.0]], [det.box for det in segmenter.seen])
        self.assertEqual(["grey cube", "grey cube"], [det.label for det in segmenter.seen])
        self.assertEqual([1.0, 1.0], [det.score for det in segmenter.seen])

    def test_a_box_that_cannot_be_cut_is_said_and_never_counted(self) -> None:
        backend = TwoStageBackend(detector=_Detector(), segmenter=_Segmenter(fail_on=(1,)))

        cut = backend.segment_boxes(_IMAGE, _BOXES)

        self.assertIsNotNone(cut.objects[0])
        self.assertIsNone(cut.objects[1])
        self.assertEqual(1, len(cut.failed))
        self.assertIn("SAM2 out of memory", cut.failed[0])
        self.assertEqual(0, backend.failures, "a box not cut is no model failure: the frame is grounded instead")

    def test_a_box_that_is_no_box_is_said(self) -> None:
        backend = TwoStageBackend(detector=_Detector(), segmenter=_Segmenter())

        cut = backend.segment_boxes(_IMAGE, (((10.0, 10.0, 5.0, 20.0), "grey cube"),))

        self.assertEqual((None,), cut.objects)
        self.assertEqual(1, len(cut.failed))


class SegmentedBoxesTests(unittest.TestCase):
    def test_segmented_boxes_refused_says_one_reason_for_every_box(self) -> None:
        refused = SegmentedBoxes.refused(_BOXES, "no route")

        self.assertEqual((None, None), refused.objects)
        self.assertEqual(("no route",), refused.failed)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
