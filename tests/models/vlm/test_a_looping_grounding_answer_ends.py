"""A grounding answer that loops ends there, and an object written twice is one box (the cell, 2026-10-12).

The owner: "Qwen ... hat unfassbar viele Objekte teilweise gefunden, also über 50 Stück auf einfachen Szenen". Qwen
wrote three or four boxes of the crate and then the same two boxes of one cube in turn until its 2048 tokens ran out:
52 boxes of 4 objects in 12 to 28 s, in 10 of the day's 157 answers, and SAM2 and the pick loop took 52 parts.

* The answer ends once three boxes in a row each repeat a box written before it (``qwen.answer_loops``), whether or
  not ``stop_at_answer_end`` is on;
* the parser drops a box that repeats one kept before it under the same description (``parsing.REPEAT_IOU``), for
  every grounding VLM;
* two parts pressed together stay two, and parts set out in rows with room between them stay every one.

The cell's Thursday (2026-10-15) met a loop that repeats nothing: Qwen walked across the picture box by box, each the
size of the one before, a fixed step further (a ladder of 51 boxes up a diagonal), down a column (each box spanning the
columns of the one before) or edge to edge over a pile (a tiling of 32 boxes), until its 2048 tokens ran out after 17 s.
Those boxes go and end the answer (``parsing.walked``), and the parts written before the walk stay; a box round two
parts goes too (``parsing.without_groups``). Every answer below is one the cell's log holds, on the picture's pixels.
"""

from __future__ import annotations

import json
import unittest

from src.models.detection.types import Detection
from src.models.vlm import parse_grounding_response
from src.models.vlm.parsing import REPEAT_IOU, CoordinateSpace, box_iou, walked, without_groups
from src.models.vlm.qwen import REPEATED_BOXES_TO_STOP, _closed_or_looping, _EndOfAnswer, answer_loops
from tests.models.vlm.test_vlm_answer_speed import END, _CharProcessor, _ids, _torch

LABEL = "each separate gray cube in the blue bin"
#: The cell's loop of 2026-10-12 13:25, on Qwen's grid: three cubes, then the third and a box 7 px lower in turn.
A, B, C = [494, 558, 542, 626], [517, 612, 566, 685], [536, 517, 583, 592]
C_LOWER = [536, 526, 583, 592]

#: 08:39:34, "each separate square part": two parts, then a ladder of 49 boxes 64 x 58 px, each 13 px left and 7 up.
SQUARES = [[707, 477, 771, 541], [764, 469, 842, 549]]
LADDER = [[707 - 13 * step, 459 - 7 * step, 771 - 13 * step, 523 - 7 * step] for step in range(49)]
#: 14:02:52, "each separate square object": the tiling of a pile, row after row edge to edge, 51 x 43 px boxes.
TILING = [[456 + 51 * col, 405 + 43 * row, 507 + 51 * col, 448 + 43 * row] for row in range(6) for col in range(4)]
#: 10-14 11:42:26, "each separate cylinder and cube": seven parts, then down a column of 50 px from y 293 to the
#: picture's foot.
PARTS = [[239, 383, 298, 444], [399, 383, 457, 425], [419, 462, 484, 522], [547, 423, 631, 462], [625, 390, 681, 441],
         [649, 351, 698, 394], [570, 336, 621, 390]]
COLUMN = [[484, 293, 534, 336], [484, 336, 534, 383], [484, 383, 534, 425], [484, 425, 534, 468],
          [484, 468, 534, 511], [484, 511, 534, 554], [484, 554, 534, 598], [484, 598, 534, 641]]


def _answer(boxes: list[list[int]], *, label: str = LABEL, closed: bool = True) -> str:
    items = ", ".join(json.dumps({"bbox_2d": box, "label": label}) for box in boxes)
    return "[" + items + ("]" if closed else "")


def _parsed(text: str, **kwargs: object) -> list[tuple[list[float], str]]:
    return [(det.box, det.label) for det in parse_grounding_response(
        text, image_width=1280, image_height=720, fallback_label=LABEL, **kwargs)]


def _in_pixels(boxes: list[list[int]]) -> list[list[float]]:
    """The boxes the parser keeps of an answer of ``boxes`` written on the picture's pixels."""
    return [det.box for det in parse_grounding_response(
        _answer(boxes), image_width=1280, image_height=720, fallback_label=LABEL, space=CoordinateSpace.ABSOLUTE)]


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

    def test_parts_set_out_in_rows_with_room_between_them_stay(self) -> None:
        rows = [[40 * i, 40 * j, 40 * i + 30, 40 * j + 30] for j in range(5) for i in range(8)]
        self.assertEqual(40, len(_in_pixels(rows)))
        self.assertFalse(any(walked(rows)))


class AWalkAcrossThePictureTests(unittest.TestCase):
    """The cell's Thursday (2026-10-15): loops that repeat nothing."""

    def test_the_ladder_goes_and_the_parts_before_it_stay(self) -> None:
        self.assertEqual([[float(v) for v in box] for box in SQUARES], _in_pixels(SQUARES + LADDER))

    def test_the_tiling_of_a_pile_goes(self) -> None:
        self.assertEqual([], _in_pixels(TILING))

    def test_a_walk_down_a_column_goes_and_the_parts_before_it_stay(self) -> None:
        self.assertEqual([[float(v) for v in box] for box in PARTS], _in_pixels(PARTS + COLUMN))

    def test_a_walk_ends_the_answer_as_it_starts(self) -> None:
        for walk in (SQUARES + LADDER, TILING, PARTS + COLUMN):
            with self.subTest(first=walk[0]):
                written = [walk[:end] for end in range(1, len(walk) + 1)]
                ends = next(index for index, boxes in enumerate(written) if answer_loops(_answer(boxes, closed=False)))
                self.assertLess(ends, len(walk) - 1, "the answer ran on to its end")
        self.assertFalse(answer_loops(_answer(PARTS, closed=False)))


class ABoxRoundTwoPartsTests(unittest.TestCase):
    def _boxes(self, *boxes: list[float]) -> list[list[float]]:
        found = [Detection(box=list(box), x_center=0.0, y_center=0.0, label=LABEL, score=1.0) for box in boxes]
        return [det.box for det in without_groups(found)[0]]

    def test_it_goes_and_the_two_parts_stay(self) -> None:
        self.assertEqual([[60.0, 60.0, 160.0, 160.0], [300.0, 300.0, 400.0, 400.0]],
                         self._boxes([50.0, 50.0, 450.0, 450.0], [60.0, 60.0, 160.0, 160.0], [300.0, 300.0, 400.0, 400.0]))

    def test_a_box_round_one_part_stays(self) -> None:
        self.assertEqual(2, len(self._boxes([50.0, 50.0, 450.0, 450.0], [60.0, 60.0, 160.0, 160.0])))

    def test_one_object_boxed_three_times_holds_nothing(self) -> None:
        box = [60.0, 60.0, 160.0, 160.0]
        self.assertEqual(3, len(self._boxes(box, [61.0, 60.0, 160.0, 160.0], [60.0, 61.0, 160.0, 160.0])))


if __name__ == "__main__":
    unittest.main()
