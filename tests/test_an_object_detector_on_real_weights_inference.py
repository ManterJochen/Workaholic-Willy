"""Real weights, real GPU: RT-DETR's classes in one call, and SAM2-large cutting a batch of boxes (``inference``).

    python -m pytest tests/test_an_object_detector_on_real_weights_inference.py -m inference -q

Skips unless CUDA and both checkpoints are on this machine, the folders ``scripts/model_weights/fetch.py rtdetr sam2``
writes or the Hugging Face cache, so a normal run pays nothing, and it never downloads. NOT part of the CI gate.

What only real weights can show: the COCO checkpoint's 80 classes in id order, and SAM2-large, on the GPU under the
autocast the cell's segmenter runs under, cutting each box of one batch as it cuts that box alone. The toy SAM2 of
``tests/test_one_sam2_pass_cuts_what_one_pass_per_box_cuts.py`` holds the same on a CPU in full precision, where the two
agree pixel for pixel; here half precision may move a pixel at a mask's edge, so the bar is an IoU. Written 2026-10-09
and not run yet: how long a batch of N boxes takes against N passes of one is printed, not asserted.
"""

from __future__ import annotations

import time
import unittest

import numpy as np
import pytest

pytestmark = pytest.mark.inference

DETECTOR = "PekingU/rtdetr_r50vd"
SEGMENTER = "facebook/sam2-hiera-large"
HEIGHT, WIDTH = 480, 640
#: Four squares on a grey field, ground truth by construction: (x0, y0, x1, y1) pixels and the BGR colour.
SQUARES = (((80, 80, 200, 200), (40, 180, 40)), ((260, 60, 380, 180), (30, 30, 220)),
           ((420, 240, 560, 380), (220, 60, 30)), ((120, 300, 240, 420), (40, 200, 220)))
#: How far a box reaches past its square, as a detector's box does.
_MARGIN = 6
#: The least IoU between a mask cut in the batch and the same box cut alone.
MIN_IOU = 0.98


def _ready() -> bool:
    try:
        import torch
        from huggingface_hub import snapshot_download

        from src.utility.paths import weights_root
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    for repo_id, category in ((DETECTOR, "detection"), (SEGMENTER, "segmentation")):
        if (weights_root() / category / repo_id.replace("/", "--") / "config.json").is_file():
            continue
        try:
            snapshot_download(repo_id=repo_id, local_files_only=True)
        except Exception:  # noqa: BLE001 - any failure means "not usable here"
            return False
    return True


def _scene() -> np.ndarray:
    image = np.full((HEIGHT, WIDTH, 3), 150, dtype=np.uint8)
    for (x0, y0, x1, y1), colour in SQUARES:
        image[y0:y1, x0:x1] = colour
    return image


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b)) / union if union else 1.0


@unittest.skipUnless(_ready(), f"needs CUDA and the {DETECTOR} and {SEGMENTER} weights "
                               "(scripts/model_weights/fetch.py rtdetr sam2)")
class RealWeightsTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from src.models.detection.object_detector import ObjectDetector

        cls.detector = ObjectDetector.from_weights(DETECTOR, local=True)

    def test_the_coco_checkpoint_names_its_80_classes_in_id_order(self) -> None:
        classes = self.detector.classes
        self.assertEqual(80, len(classes))
        self.assertEqual("person", classes[0])
        found = self.detector.detect(_scene(), threshold=0.3)
        self.assertTrue(set(found.labels) <= set(classes), found.labels)
        self.assertEqual(classes, found.classes)

    def test_a_batch_cuts_every_box_as_its_own_pass_cuts_it(self) -> None:
        image = _scene()
        boxes = [(float(x0 - _MARGIN), float(y0 - _MARGIN), float(x1 + _MARGIN), float(y1 + _MARGIN))
                 for (x0, y0, x1, y1), _ in SQUARES]
        started = time.perf_counter()
        batch = self.detector._cut(image, boxes)  # noqa: SLF001 (the batch alone, on boxes known by construction)
        batch_s = time.perf_counter() - started
        segmenter = self.detector._segmenter_built()  # noqa: SLF001
        alone_s = 0.0
        for (mask, score), box, ((x0, y0, x1, y1), _) in zip(batch, boxes, SQUARES):
            with self.subTest(box=box):
                started = time.perf_counter()
                alone = segmenter.segment(image_rgb=image[:, :, ::-1].copy(),
                                          bbox_xyxy=tuple(int(value) for value in box), label="square")
                alone_s += time.perf_counter() - started
                self.assertGreaterEqual(_iou(mask, alone.mask.astype(bool)), MIN_IOU)
                self.assertAlmostEqual(alone.metadata["predicted_iou"][0], score, delta=0.01)
                square = np.zeros_like(mask)
                square[y0:y1, x0:x1] = True
                self.assertGreaterEqual(_iou(mask, square), 0.9, "SAM2 cut the square its box holds")
        print(f"\n  {len(boxes)} boxes: one batch {batch_s * 1000:.0f} ms, one pass each {alone_s * 1000:.0f} ms "
              "(the first batch includes loading SAM2)")

    def test_detect_with_masks_gives_each_object_the_image_size_and_its_own_box(self) -> None:
        found = self.detector.detect(_scene(), threshold=0.2, segment=True)
        for obj in found:
            with self.subTest(obj.label):
                self.assertEqual((HEIGHT, WIDTH), obj.mask.shape)
                x0, y0, x1, y1 = (int(round(value)) for value in obj.box)
                inside = int(obj.mask[y0:y1, x0:x1].sum())
                self.assertGreaterEqual(inside, 0.9 * int(obj.mask.sum()))


if __name__ == "__main__":
    unittest.main()
