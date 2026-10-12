"""A grounding answer that loops ends there, and an object written twice is one box (the cell, 2026-10-12).

The owner: "Qwen ... hat unfassbar viele Objekte teilweise gefunden, also über 50 Stück auf einfachen Szenen". Qwen
wrote three or four boxes of the crate and then the same two boxes of one cube in turn until its 2048 tokens ran out:
52 boxes of 4 objects in 12 to 28 s, in 10 of the day's 157 answers, and SAM2 and the pick loop took 52 parts.

* The answer ends once three boxes in a row each repeat a box written before it (``qwen.answer_loops``), whether or
  not ``stop_at_answer_end`` is on;
* the parser drops a box that repeats one kept before it under the same description (``parsing.REPEAT_IOU``), for
  every grounding VLM;
* two parts pressed together stay two, and one box under two descriptions stays ambiguous, as before.
"""

from __future__ import annotations

import json
import unittest

from src.models.vlm import parse_grounding_response
from src.models.vlm.parsing import AMBIGUOUS_LABEL, REPEAT_IOU, box_iou
from src.models.vlm.qwen import REPEATED_BOXES_TO_STOP, _closed_or_looping, _EndOfAnswer, answer_loops
from tests.test_vlm_answer_speed import END, _CharProcessor, _ids, _torch

LABEL = "each separate gray cube in the blue bin"
#: The cell's loop of 2026-10-12 13:25, on Qwen's grid: three cubes, then the third and a box 7 px lower in turn.
A, B, C = [494, 558, 542, 626], [517, 612, 566, 685], [536, 517, 583, 592]
C_LOWER = [536, 526, 583, 592]


def _answer(boxes: list[list[int]], *, label: str = LABEL, closed: bool = True) -> str:
    items = ", ".join(json.dumps({"bbox_2d": box, "label": label}) for box in boxes)
    return "[" + items + ("]" if closed else "")


def _parsed(text: str, **kwargs: object) -> list[tuple[list[float], str]]:
    return [(det.box, det.label) for det in parse_grounding_response(
        text, image_width=1280, image_height=720, fallback_label=LABEL, **kwargs)]


class ALoopingAnswerEndsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.torch = _torch()
        self.prompt = _ids('Return [{"bbox_2d": [x0, y0, x1, y1]}]. Description: cube') + [END]

    def _where_it_stops(self, answer: str) -> str:
        """The answer as the stop leaves it: up to the first character after which the criterion ends it."""
        criterion = _EndOfAnswer(_CharProcessor(), self.torch, len(self.prompt), closed=_closed_or_looping)
        for end in range(1, len(answer) + 1):
            if bool(criterion(self.torch.tensor([self.prompt + _ids(answer[:end])]), None)[0]):
                return answer[:end]
        return answer

    def test_the_cells_loop_ends_after_three_repeats_and_reads_as_its_three_cubes(self) -> None:
        self.assertGreaterEqual(box_iou(C, C_LOWER), REPEAT_IOU)
        looping = _answer([A, B, C] + [C_LOWER, C] * 24, closed=False)
        cut = self._where_it_stops(looping)
        self.assertLess(len(cut), len(looping) // 5)
        self.assertEqual(3 + REPEATED_BOXES_TO_STOP, cut.count("bbox_2d"))
        self.assertTrue(answer_loops(cut))
        self.assertEqual(3, len(_parsed(cut)))
        self.assertEqual(3, len(_parsed(_answer([A, B, C] + [C_LOWER, C] * 24))), "the whole loop reads the same")

    def test_a_list_of_distinct_parts_never_loops(self) -> None:
        boxes = [[40 * i, 40 * j, 40 * i + 30, 40 * j + 30] for i in range(8) for j in range(5)]
        answer = _answer(boxes)
        self.assertFalse(any(answer_loops(answer[:end]) for end in range(len(answer) + 1)))
        self.assertEqual(answer, self._where_it_stops(answer)[:len(answer)])
        self.assertEqual(40, len(_parsed(answer)))

    def test_one_part_written_twice_is_kept_once_and_does_not_end_the_answer(self) -> None:
        answer = _answer([A, B, C_LOWER, C, [100, 100, 150, 150]])
        self.assertFalse(answer_loops(answer))
        self.assertEqual(answer, self._where_it_stops(answer))
        self.assertEqual(4, len(_parsed(answer)))


class ThePartsTheParserKeepsTests(unittest.TestCase):
    def test_two_parts_pressed_together_stay_two(self) -> None:
        left, right = [400, 400, 460, 460], [430, 400, 490, 460]
        self.assertLess(box_iou(left, right), REPEAT_IOU)
        self.assertEqual(2, len(_parsed(_answer([left, right]))))

    def test_one_box_under_two_descriptions_stays_ambiguous(self) -> None:
        text = ("[" + json.dumps({"bbox_2d": C, "label": "cube"}) + ", "
                + json.dumps({"bbox_2d": C_LOWER, "label": "cylinder"}) + "]")
        parsed = _parsed(text, classes=("cube", "cylinder"))
        self.assertEqual([AMBIGUOUS_LABEL, AMBIGUOUS_LABEL], [label for _box, label in parsed])

    def test_one_description_written_twice_in_a_class_list_is_one_part(self) -> None:
        text = ("[" + json.dumps({"bbox_2d": C, "label": "cube"}) + ", "
                + json.dumps({"bbox_2d": C_LOWER, "label": "cube"}) + ", "
                + json.dumps({"bbox_2d": A, "label": "cylinder"}) + "]")
        self.assertEqual(["cube", "cylinder"], [label for _box, label in _parsed(text, classes=("cube", "cylinder"))])


if __name__ == "__main__":
    unittest.main()
