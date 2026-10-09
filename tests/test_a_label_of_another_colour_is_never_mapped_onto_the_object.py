"""A label of another colour is never mapped onto the object: the camera source's label fix (S0, 2026-10-08).

The camera source maps a detector's free words onto the object labels a task names, so the pick loop's label gate can
compare exactly. It used to map a label onto the object it shared one word with: "red cube" became "grey cube" by
"cube", "grey cylinder" became "grey cube" by "grey", and every box was a target. A label now maps onto an object label
only where it holds all of that label's words (grey and gray one word, a plural its singular), the most specific
wins, a tie maps nothing, and a bare shortened label ("cube") maps only where there is one object label to shorten.

What the detector called each box is logged too (``src/models/vlm/qwen.py``), so a run can be read afterwards: before,
only the count was.

The colour check is switched off here: these are the words alone (its pixels are ``tests/test_realsense_perception_
source.py``'s).
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest import mock

import numpy as np

from src.camera.setup.image_taking.frames import RGBDFrame
from src.robot.perception import RealSenseVisionPerceptionSource

_K = np.array([[600.0, 0.0, 32.0], [0.0, 600.0, 32.0], [0.0, 0.0, 1.0]])


@dataclass
class _Det:
    box: tuple[float, float, float, float]
    label: str
    score: float = 0.9


@dataclass(frozen=True)
class _Seg:
    mask: np.ndarray
    label: str


class _Streamer:
    def grab(self) -> RGBDFrame:
        return RGBDFrame(color=np.zeros((64, 64, 3), dtype=np.uint8), depth=np.full((64, 64), 500, np.uint16))

    def get_intrinsics(self) -> np.ndarray:
        return _K.copy()


class _Detector:
    def __init__(self, label: str) -> None:
        self.label = label

    def detect_all(self, bgr: np.ndarray, prompt: str) -> list[_Det]:
        return [_Det(box=(10, 10, 30, 30), label=self.label)]


class _Segmenter:
    def segment_detection(self, bgr: np.ndarray, det: _Det) -> _Seg:
        mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        mask[10:30, 10:30] = 1
        return _Seg(mask=mask, label=det.label)


def _mapped(said: str, *labels: str) -> str:
    """What the source calls a box the detector called ``said``, with ``labels`` the task's object labels."""
    source = RealSenseVisionPerceptionSource(
        streamer=_Streamer(), detector=_Detector(said), segmenter=_Segmenter(), prompt="x", object_labels=labels,
        warmup_grabs=0, colour_check="off")
    return source.acquire().segmentations[0].label


class AnotherColourIsNeverMappedTests(unittest.TestCase):
    def test_a_label_of_another_colour_stays_what_the_detector_called_it(self) -> None:
        for said, labels in (("red cube", ("grey cube",)), ("green cube", ("grey cube",)),
                             ("grey cylinder", ("grey cube",)), ("red part", ("green part",)),
                             ("orange part", ("green part", "red part")), ("red cube", ("green part", "red part"))):
            with self.subTest(said=said, labels=labels):
                self.assertEqual(said, _mapped(said, *labels))

    def test_a_label_that_holds_every_word_of_the_object_maps_onto_it(self) -> None:
        for said in ("grey cube", "each separate grey cube", "a grey cube on the black mat", "cube, grey"):
            with self.subTest(said=said):
                self.assertEqual("grey cube", _mapped(said, "grey cube"))

    def test_plural_and_gray_or_grey_still_map(self) -> None:
        for said in ("gray cube", "grey cubes", "each separate gray cubes", "Gray Cube"):
            with self.subTest(said=said):
                self.assertEqual("grey cube", _mapped(said, "grey cube"))
        self.assertEqual("sugar box", _mapped("two sugar boxes", "sugar box", "mug"))

    def test_two_equally_specific_labels_map_nothing(self) -> None:
        self.assertEqual("green and red part", _mapped("green and red part", "green part", "red part"))

    def test_the_most_specific_label_wins(self) -> None:
        self.assertEqual("green part", _mapped("green part", "part", "green part"))
        self.assertEqual("green part", _mapped("each separate green part", "part", "green part"))

    def test_a_bare_shortened_label_maps_only_onto_the_one_object_label(self) -> None:
        self.assertEqual("grey cube", _mapped("cube", "grey cube"))
        self.assertEqual("a mug", _mapped("mug", "a mug"))
        self.assertEqual("cube", _mapped("cube", "grey cube", "red cube"))

    def test_with_no_object_labels_every_label_passes_through(self) -> None:
        self.assertEqual("red cube", _mapped("red cube"))


class TheGrounderLogsEveryBoxTests(unittest.TestCase):
    def test_the_grounding_line_names_what_the_model_called_every_box(self) -> None:
        from src.models.vlm import Qwen3VLGrounder

        answer = ('[{"bbox_2d": [100, 100, 200, 200], "label": "grey cube"}, '
                  '{"bbox_2d": [300, 300, 400, 400], "label": "red cube"}]')
        grounder = Qwen3VLGrounder(model_id="Qwen/Qwen3-VL-4B-Instruct")
        with mock.patch.object(grounder, "_ensure_loaded", return_value=False), \
                mock.patch.object(grounder, "_generate", return_value=(answer, 60)), \
                self.assertLogs("src.models.vlm.qwen", level="INFO") as said:
            detections = grounder.detect_all(np.zeros((72, 128, 3), np.uint8), "each separate grey cube")

        self.assertEqual(["grey cube", "red cube"], [detection.label for detection in detections])
        [line] = [line for line in said.output if "grounded 2 box(es)" in line]
        self.assertIn("'grey cube' at [", line)
        self.assertIn("'red cube' at [", line)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
