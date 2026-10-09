"""``segment=True`` cuts every box of an image in one SAM2 pass, and each mask stays with its own box.

The owner, 2026-10-09: the masks come from the same call as the boxes, switched on by one flag. The cell's perception
cuts one box per SAM2 call, which encodes the image once per box; ``ObjectDetector`` hands SAM2's processor and model
every box of the image at once, so the image is encoded once and the decoder answers each box on its own. What is held
here: one processor call and one model call for N boxes; N masks of the image's size, each lying in its own box and
carrying its own predicted IoU; the first of SAM2's three proposals kept, as ``Sam2Segmenter.segment`` keeps it, and
cleaned as it cleans its one; SAM2 built once, on the first call with a box to cut, and kept; and nothing asked of SAM2
when there is no box. Both models are stand-ins (``tests/_object_detector_fakes.py``); ``Sam2Segmenter`` itself runs.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import torch

from src.config.schema.models.models_schema import ObjectDetectorConfig
from src.models.detection.closed_set.detector import RtDetrObjectDetector
from src.models.detection.object_detector import Detections, ObjectDetector, _cleaned
from tests._object_detector_fakes import CHESS, detector_folder, pin_cpu, rtdetr, sam2, sam2_folder

#: What undoes the CPU pin of this module, once ``setUpModule`` has set it.
_UNPIN: list = []


def setUpModule() -> None:  # noqa: N802 (unittest's name)
    _UNPIN.append(pin_cpu())


def tearDownModule() -> None:  # noqa: N802 (unittest's name)
    while _UNPIN:
        _UNPIN.pop()()


HEIGHT, WIDTH = 120, 160
#: Four pieces in the image, the black pawn reaching past its corner, and a white knight below the threshold.
SCENE = [
    ((10.0, 10.0, 30.0, 40.0), CHESS.index("white_pawn"), 0.95),
    ((50.0, 20.0, 80.0, 60.0), CHESS.index("black_king"), 0.8),
    ((100.0, 50.0, 140.0, 100.0), CHESS.index("white_queen"), 0.6),
    ((150.0, 90.0, 175.0, 130.0), CHESS.index("black_pawn"), 0.7),
    ((60.0, 70.0, 90.0, 100.0), CHESS.index("white_knight"), 0.3),
]


def _image() -> np.ndarray:
    image = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)
    image[:, :, 0] = 30  # blue low, so the image handed to SAM2 shows whether it was turned to RGB
    return image


class _Tmp(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def built(self, scene: list, **kwargs: Any) -> ObjectDetector:
        with rtdetr(scene):
            return ObjectDetector.from_weights(detector_folder(self.root / "chess"),
                                               segmenter=sam2_folder(self.root / "sam2"), **kwargs)


def _inside_its_box(test: unittest.TestCase, found: Detections) -> None:
    for obj in found:
        x0, y0, x1, y1 = (int(value) for value in obj.box)
        outside = obj.mask.copy()
        outside[y0:y1, x0:x1] = False
        test.assertFalse(outside.any(), f"the mask of {obj.label} reaches past its box")
        cx, cy = obj.centre
        test.assertTrue(obj.mask[int(cy), int(cx)], f"the mask of {obj.label} misses the middle of its box")


class OnePassTests(_Tmp):

    def test_n_boxes_are_one_sam2_call_with_n_boxes_on_one_image(self) -> None:
        detector = self.built(SCENE)
        with sam2() as fake:
            found = detector.detect(_image(), segment=True)
        self.assertEqual(4, len(found))
        self.assertEqual(1, len(fake.processor.calls))
        self.assertEqual(1, len(fake.model.calls))
        (call,) = fake.processor.calls
        self.assertEqual([list(obj.box) for obj in found], call["input_boxes"][0])
        np.testing.assert_array_equal(_image()[:, :, ::-1], call["images"])

    def test_each_mask_is_the_image_size_and_lies_in_its_own_box(self) -> None:
        with sam2():
            found = self.built(SCENE).detect(_image(), segment=True)
        self.assertTrue(found.segmented)
        for obj in found:
            with self.subTest(obj.label):
                self.assertEqual((HEIGHT, WIDTH), obj.mask.shape)
                self.assertEqual(np.bool_, obj.mask.dtype)
                self.assertFalse(obj.mask.flags.writeable)
        _inside_its_box(self, found)
        (pawn,) = found.of("black_pawn")
        self.assertEqual((150.0, 90.0, 160.0, 120.0), pawn.box, "the box SAM2 was given is the clipped one")

    def test_the_first_of_sam2s_proposals_is_kept_with_its_own_predicted_iou(self) -> None:
        with sam2():
            found = self.built(SCENE).detect(_image(), segment=True)
        for index, obj in enumerate(found):
            with self.subTest(obj.label):
                self.assertAlmostEqual(0.9 - 0.01 * index, obj.mask_score, places=6)
                self.assertLess(int(obj.mask.sum()), HEIGHT * WIDTH // 4, "a whole-image proposal was kept")

    def test_more_boxes_than_one_scaling_round_keep_their_order(self) -> None:
        scene = [((4.0 + 15 * i, 5.0, 14.0 + 15 * i, 25.0), i % len(CHESS), 0.99 - 0.01 * i) for i in range(10)]
        with sam2() as fake:
            found = self.built(scene).detect(_image(), segment=True)
        self.assertEqual(10, len(found))
        self.assertEqual(1, len(fake.model.calls))
        self.assertEqual(10, len(fake.processor.calls[0]["input_boxes"][0]))
        for index, obj in enumerate(found):
            self.assertAlmostEqual(0.9 - 0.01 * index, obj.mask_score, places=6)
        _inside_its_box(self, found)

    def test_a_filtered_call_cuts_only_the_classes_it_looked_for(self) -> None:
        with sam2() as fake:
            found = self.built(SCENE).detect(_image(), classes=["white_queen", "black_king"], segment=True)
        self.assertEqual(("black_king", "white_queen"), found.labels)
        self.assertEqual(2, len(fake.processor.calls[0]["input_boxes"][0]))


class LoadedOnceTests(_Tmp):

    def test_sam2_is_built_on_the_first_call_with_a_box_to_cut_and_kept(self) -> None:
        detector = self.built(SCENE)
        with sam2() as fake:
            self.assertIn("loaded by the first segment=True with a box to cut", detector.render())
            detector.detect(_image())
            self.assertEqual(0, fake.loads.call_count)
            detector.detect(_image(), segment=True)
            detector.detect(_image(), segment=True, classes=["white_pawn"])
        self.assertEqual(1, fake.loads.call_count)
        self.assertEqual(2, len(fake.model.calls))
        self.assertIn("loaded on cpu", detector.render())

    def test_an_image_with_no_box_asks_sam2_nothing(self) -> None:
        with sam2() as fake:
            found = self.built([]).detect(_image(), segment=True)
        self.assertEqual((0, True), (len(found), found.segmented))
        self.assertEqual(0, fake.loads.call_count)

    def test_a_detector_built_without_sam2_refuses_segment_true_before_it_detects(self) -> None:
        with rtdetr(SCENE) as fake:
            bare = ObjectDetector(RtDetrObjectDetector(ObjectDetectorConfig(
                model_path=str(detector_folder(self.root / "bare")), threshold=0.5, local=True)))
        with self.assertRaisesRegex(ValueError, "segment=True needs SAM2"):
            bare.detect(_image(), segment=True)
        self.assertEqual([], fake.processor.thresholds, "it refused before RT-DETR ran")
        self.assertIn("none: segment=True is refused", bare.render())
        self.assertEqual(4, len(bare.detect(_image())))

    def test_sam2_is_moved_to_the_device_from_weights_names(self) -> None:
        from src.models.segmentation.realtime import segmenter as module

        detector = self.built(SCENE, device="cpu")
        self.assertEqual("cpu", detector.device)
        # SAM2 places itself where every model here runs; this one lands elsewhere and has to be brought to the CPU.
        with sam2() as fake, mock.patch.object(module, "get_device", return_value=torch.device("meta")):
            detector.detect(_image(), segment=True)
        self.assertEqual([torch.device("meta"), torch.device("cpu")], fake.model.moved_to)
        self.assertEqual(torch.device("cpu"), detector._segmenter.device)  # noqa: SLF001 (where it was moved)


class TheCleaningTests(unittest.TestCase):

    def test_a_mask_is_cleaned_as_the_segmenter_cleans_its_one_mask(self) -> None:
        from src.models.segmentation.realtime.segmenter import Sam2Segmenter

        segmenter = Sam2Segmenter.__new__(Sam2Segmenter)
        rng = np.random.default_rng(7)
        for kernel, largest in ((5, True), (3, False), (7, True), (1, True)):
            segmenter.keep_largest_component = largest
            segmenter.morph_kernel_size = kernel
            for _ in range(4):
                raw = rng.random((60, 80)) > 0.55
                raw[10:40, 20:50] = True
                with self.subTest(kernel=kernel, largest=largest):
                    expected = segmenter._postprocess_mask(raw.astype(np.float32)).astype(bool)  # noqa: SLF001
                    np.testing.assert_array_equal(expected, _cleaned(raw, kernel=kernel, keep_largest=largest))


if __name__ == "__main__":
    unittest.main()
