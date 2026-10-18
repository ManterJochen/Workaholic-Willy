"""A grounded mask that splits into bodies is a box drawn round several parts, and the frame drops it (the owner,
2026-10-15: Qwen's group boxes on the cell's Thursday).

A pile boxed whole, or two parts in one box, gives a mask of two or more pieces, each a fair share of its pixels
(``realsense_source.BODY_SHARE``, a fifth): no part, whatever the detector called it. What it covered stays in the depth
as what nobody named. One part's mask with a speck beside it, and a part whose mask is one piece, stay.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.perception import RealSenseVisionPerceptionSource
from src.robot.perception.realsense_source import BODY_SHARE, bodies_of
from tests.camera.setup.image_taking.test_a_label_of_another_colour_is_never_mapped_onto_the_object import _Det, _Seg, _Streamer


def _mask(*squares: tuple[int, int, int]) -> np.ndarray:
    """A 64 x 64 mask of the squares ``(row, column, side)``."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    for row, column, side in squares:
        mask[row:row + side, column:column + side] = 1
    return mask


class _Detector:
    def detect_all(self, bgr: np.ndarray, prompt: str) -> list[_Det]:
        return [_Det(box=(2, 2, 62, 62), label="part"), _Det(box=(40, 40, 60, 60), label="part")]


class _Segmenter:
    """The first box's mask is two parts, the second's one."""

    def segment_detection(self, bgr: np.ndarray, det: _Det) -> _Seg:
        if det.box[0] == 2:
            return _Seg(mask=_mask((4, 4, 14), (30, 30, 14)), label=det.label)
        return _Seg(mask=_mask((40, 40, 18)), label=det.label)


class AMaskRoundSeveralPartsTests(unittest.TestCase):
    def test_it_splits_into_bodies(self) -> None:
        self.assertEqual(0.2, BODY_SHARE)
        self.assertEqual(2, bodies_of(_mask((4, 4, 14), (30, 30, 14))))
        self.assertEqual(1, bodies_of(_mask((4, 4, 14))))
        self.assertEqual(1, bodies_of(_mask((4, 4, 14), (30, 30, 2))), "a speck beside one part is no body")
        self.assertEqual(0, bodies_of(np.zeros((8, 8))))

    def test_a_grounded_frame_drops_it_and_keeps_the_part(self) -> None:
        source = RealSenseVisionPerceptionSource(streamer=_Streamer(), detector=_Detector(), segmenter=_Segmenter(),
                                                 prompt="part", warmup_grabs=0, colour_check="off")
        frame = source.acquire()
        self.assertEqual(1, len(frame.segmentations))
        self.assertEqual(18 * 18, int(np.asarray(frame.segmentations[0].mask).sum()))


if __name__ == "__main__":
    unittest.main()
