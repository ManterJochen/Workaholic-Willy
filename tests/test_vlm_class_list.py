"""One Qwen call grounds every rule's kind of part: the class-list prompt, its instruction, and the boxes no rule may take.

No weights and no GPU: the model is a double that answers a fixed string, and its processor keeps what it was asked.
What the real model writes for a class list is ``tests/test_vlm_class_list_inference.py`` (GPU, outside the gate).

* ``class_list_prompt(["green part", "red part"])`` is ``"green part | red part"``, and ``classes_of`` reads it back. A
  prompt of one description reads ``()`` and is asked with today's instruction, byte for byte.
* A class list is asked with the class-list instruction: each object once, labelled with the one description it
  matches best, copied exactly, what matches none left out, ``[]`` for none. The answer's end, the lookup and the decode
  graphs are the same for both.
* The answer fails closed, since sorting places no part by guess (the owner, 2026-10-09). Boxes over one object
  (IoU >= 0.7) under two descriptions are all ``ambiguous``, and so is a box named by none. A label outside the list is
  kept as the model wrote it, and the camera source maps it onto no rule
  (``tests/test_a_locate_of_a_class_list_finds_every_bin_in_one_call.py``).
"""

from __future__ import annotations

import contextlib
import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.models.vlm import parse_grounding_response
from src.models.vlm.parsing import AMBIGUOUS_IOU, AMBIGUOUS_LABEL, CoordinateSpace
from src.models.vlm.qwen import (
    CLASS_LIST_SEPARATOR,
    GROUNDING_INSTRUCTION,
    Qwen3VLGrounder,
    class_list_prompt,
    classes_of,
    grounding_instruction,
)

#: Today's instruction around one description, spelled out here rather than read from the module: a change to the
#: words the model is asked is a change to this test too.
TODAY = ("Locate every object matching this description and return ONLY a JSON array, no prose:\n"
         '[{"bbox_2d": [x0, y0, x1, y1], "label": "<short name>"}]\n'
         "Coordinates are absolute pixels in this image. "
         "If no object matches, return an empty array []. Do not guess.\n\n"
         "Description: ")
#: The class-list instruction around two descriptions, spelled out as well.
CLASS_LIST = ("Locate every object matching one of these descriptions and return ONLY a JSON array, no prose:\n"
              '[{"bbox_2d": [x0, y0, x1, y1], "label": "<the description it matches, copied exactly>"}]\n'
              "Coordinates are absolute pixels in this image. "
              "Give each object once, in a box of its own, labelled with the one description it matches best; "
              "leave out every object that matches none. "
              "If no object matches, return an empty array []. Do not guess.\n\n"
              'Descriptions: "green part"; "red part"')

W, H = 640, 480
#: The end token of the model double below.
END = 1_000_001


def _boxes(answer: str, *, classes: tuple[str, ...] = (),
           prompt: str = "green part | red part") -> list[tuple[str, list[float]]]:
    """``(label, box)`` of every box read from ``answer``, its numbers taken as pixels."""
    return [(det.label, det.box) for det in parse_grounding_response(
        answer, image_width=W, image_height=H, fallback_label=prompt, space=CoordinateSpace.ABSOLUTE, classes=classes)]


def _answer(*boxes: tuple[list[int], str | None]) -> str:
    """A grounding answer of ``(box, label)`` pairs, a ``None`` label written as no label at all."""
    items = []
    for box, label in boxes:
        items.append(f'{{"bbox_2d": {box}}}' if label is None else f'{{"bbox_2d": {box}, "label": "{label}"}}')
    return "[" + ", ".join(items) + "]"


class AClassListPromptTests(unittest.TestCase):
    def test_two_descriptions_make_one_prompt_that_reads_back(self) -> None:
        prompt = class_list_prompt(["green part", "red part"])
        self.assertEqual("green part | red part", prompt)
        self.assertEqual(" | ", CLASS_LIST_SEPARATOR)
        self.assertEqual(("green part", "red part"), classes_of(prompt))
        three = class_list_prompt(("yellow bin", "blue bin", "red bin"))
        self.assertEqual(("yellow bin", "blue bin", "red bin"), classes_of(three))

    def test_one_description_is_the_prompt_it_always_was(self) -> None:
        for phrase in ("grey cube", "each separate gray cube", "the cup left of the bin"):
            with self.subTest(phrase=phrase):
                self.assertEqual(phrase, class_list_prompt([phrase]))
                self.assertEqual((), classes_of(phrase))

    def test_a_description_given_twice_is_one(self) -> None:
        self.assertEqual("yellow bin | blue bin", class_list_prompt(["yellow bin", "blue bin", "yellow bin"]))
        self.assertEqual("green part", class_list_prompt([" green part ", "green part"]),
                         "one description left is no list")
        self.assertEqual("Yellow bin | blue bin", class_list_prompt(["Yellow bin", "blue bin", "yellow  BIN"]),
                         "case and spacing aside, kept as first written")

    def test_what_cannot_be_a_description_is_refused(self) -> None:
        for phrases in ([], [""], ["  "], ["green part", ""], ["green | red part"], ["ambiguous part"],
                        ["red part", "the Ambiguous one"]):
            with self.subTest(phrases=phrases), self.assertRaises(ValueError):
                class_list_prompt(phrases)

    def test_a_list_written_by_hand_reads_too(self) -> None:
        self.assertEqual(("green part", "red part"), classes_of("green part|red part"))
        self.assertEqual(("green part", "red part"), classes_of("green part |  | red part"))
        self.assertEqual((), classes_of("green part | green part"), "one description twice is no list")
        self.assertEqual((), classes_of("green part | Green  Part"))
        self.assertEqual((), classes_of(""))


class TheInstructionTests(unittest.TestCase):
    def test_one_description_is_asked_today_s_instruction_byte_for_byte(self) -> None:
        for prompt in ("each separate grey cube", "yellow bin", 'a "quoted" {braced} part'):
            with self.subTest(prompt=prompt):
                self.assertEqual(TODAY + prompt, grounding_instruction(prompt))
                self.assertEqual(GROUNDING_INSTRUCTION.format(prompt=prompt), grounding_instruction(prompt))

    def test_a_class_list_is_asked_the_class_list_instruction(self) -> None:
        self.assertEqual(CLASS_LIST, grounding_instruction("green part | red part"))
        asked = grounding_instruction(class_list_prompt(["yellow bin", "blue bin", "red bin"]))
        self.assertTrue(asked.endswith('Descriptions: "yellow bin"; "blue bin"; "red bin"'), asked)
        for said in ("copied exactly", "Give each object once", "the one description it matches best",
                     "leave out every object that matches none", "return an empty array []", "Do not guess"):
            self.assertIn(said, asked)

    def test_a_description_is_written_as_a_json_string_writes_it(self) -> None:
        """The label the model copies into its JSON answer reads back as the description."""
        asked = grounding_instruction(class_list_prompt(['5" pipe', "red part"]))
        self.assertTrue(asked.endswith('Descriptions: "5\\" pipe"; "red part"'), asked)


# --- the grounder, through doubles of the processor and the model ---------------------------------------------------


class _Batch(dict):  # type: ignore[type-arg]
    def to(self, device: Any) -> "_Batch":
        return self


class _Processor:
    """Keeps every conversation it was handed, and decodes every answer as ``answer``."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.asked: list[Any] = []
        self.tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: None, eos_token_id=END)

    def apply_chat_template(self, messages: Any, **_: Any) -> _Batch:
        self.asked.append(messages)
        return _Batch(input_ids=np.arange(7).reshape(1, -1), attention_mask=np.ones((1, 7)))

    def batch_decode(self, ids: Any, skip_special_tokens: bool = False) -> list[str]:
        return [self.answer]


class _Model:
    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.config = SimpleNamespace(image_token_id=None, video_token_id=None, vision_start_token_id=None,
                                      vision_end_token_id=None)
        self.generation_config = SimpleNamespace(eos_token_id=[END])

    def generate(self, **kwargs: Any) -> np.ndarray:
        self.calls.append(kwargs)
        if "prompt_lookup_num_tokens" in kwargs:
            self._get_candidate_generator(generation_config=SimpleNamespace(max_length=2055),
                                          input_ids=kwargs["input_ids"])
        return np.concatenate([kwargs["input_ids"], np.full((1, 4), 99)], axis=1)


class _Torch:
    @staticmethod
    def inference_mode() -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    cuda = SimpleNamespace(empty_cache=lambda: None)


def _grounder(answer: str, **settings: Any) -> tuple[Qwen3VLGrounder, _Processor, _Model]:
    processor, model = _Processor(answer), _Model()
    grounder = Qwen3VLGrounder(model_id="Qwen/Qwen3-VL-4B-Instruct", local=True, **settings)
    grounder._load = lambda: (processor, model, _Torch())  # type: ignore[method-assign]
    return grounder, processor, model


def _image() -> np.ndarray:
    return np.zeros((H, W, 3), np.uint8)


def _asked(processor: _Processor, call: int = -1) -> str:
    [message] = processor.asked[call]
    image, text = message["content"]
    assert image["type"] == "image", image
    return str(text["text"])


class TheGrounderAsksTests(unittest.TestCase):
    def test_one_description_is_asked_as_ever(self) -> None:
        grounder, processor, _ = _grounder("[]")
        grounder.detect_all(_image(), "each separate grey cube")
        self.assertEqual(TODAY + "each separate grey cube", _asked(processor))

    def test_a_class_list_is_asked_in_one_call(self) -> None:
        grounder, processor, model = _grounder("[]")
        grounder.detect_all(_image(), class_list_prompt(["green part", "red part"]))
        self.assertEqual(CLASS_LIST, _asked(processor))
        self.assertEqual(1, len(model.calls), "every description in one call")

    def test_both_end_at_the_array_and_take_the_same_lookup_and_graphs(self) -> None:
        grounder, _, model = _grounder("[]", prompt_lookup_tokens=10)
        taken: list[dict[str, Any]] = []
        greedy = grounder._greedy

        def spy(*args: Any, **kwargs: Any) -> Any:
            taken.append({"max_new_tokens": kwargs["max_new_tokens"], "lookup_tokens": kwargs["lookup_tokens"],
                          "graphs_mode": kwargs["graphs_mode"],
                          "stops": [type(stop).__name__ for stop in kwargs.get("stops") or ()]})
            return greedy(*args, **kwargs)

        grounder._greedy = spy  # type: ignore[method-assign]
        grounder.detect_all(_image(), "green part")
        grounder.detect_all(_image(), "green part | red part")
        self.assertEqual(taken[0], taken[1])
        self.assertEqual(["_EndOfAnswer"], taken[0]["stops"])
        self.assertEqual(["auto", "auto"], [each["graphs_mode"] for each in taken])
        self.assertEqual([10, 10], [call["prompt_lookup_num_tokens"] for call in model.calls])


class TheAnswerFailsClosedTests(unittest.TestCase):
    """What a class list's answer may not hand a rule: a part two descriptions claim, or one named by none."""

    CLASSES = ("green part", "red part")

    def test_one_object_boxed_under_two_descriptions_is_ambiguous(self) -> None:
        answer = _answer(([100, 100, 200, 200], "green part"), ([104, 100, 200, 200], "red part"),
                         ([400, 100, 500, 200], "green part"))
        self.assertEqual([AMBIGUOUS_LABEL, AMBIGUOUS_LABEL, "green part"],
                         [label for label, _ in _boxes(answer, classes=self.CLASSES)])

    def test_the_boxes_stay_where_they_were(self) -> None:
        """Relabelled, never dropped: an ambiguous part is an obstacle every grasp is planned around."""
        answer = _answer(([100, 100, 200, 200], "green part"), ([104, 100, 200, 200], "red part"))
        self.assertEqual([[100.0, 100.0, 200.0, 200.0], [104.0, 100.0, 200.0, 200.0]],
                         [box for _, box in _boxes(answer, classes=self.CLASSES)])

    def test_two_parts_side_by_side_keep_their_descriptions(self) -> None:
        answer = _answer(([100, 100, 200, 200], "green part"), ([150, 100, 250, 200], "red part"))
        self.assertEqual(["green part", "red part"], [label for label, _ in _boxes(answer, classes=self.CLASSES)])

    def test_the_overlap_that_makes_one_object_is_inclusive(self) -> None:
        self.assertEqual(0.7, AMBIGUOUS_IOU)
        at = _answer(([0, 0, 100, 100], "green part"), ([0, 0, 70, 100], "red part"))      # IoU 0.70
        below = _answer(([0, 0, 100, 100], "green part"), ([0, 0, 69, 100], "red part"))   # IoU 0.69
        self.assertEqual([AMBIGUOUS_LABEL] * 2, [label for label, _ in _boxes(at, classes=self.CLASSES)])
        self.assertEqual(["green part", "red part"], [label for label, _ in _boxes(below, classes=self.CLASSES)])

    def test_a_chain_of_overlaps_is_one_object(self) -> None:
        """A and B overlap, B and C overlap, A and C less: all three are the one object, and C names another
        description, so not even A keeps its own."""
        answer = _answer(([0, 0, 100, 100], "green part"), ([0, 0, 100, 80], "green part"),
                         ([0, 0, 100, 60], "red part"))
        self.assertEqual([AMBIGUOUS_LABEL] * 3, [label for label, _ in _boxes(answer, classes=self.CLASSES)])

    def test_one_description_boxed_twice_keeps_it(self) -> None:
        answer = _answer(([100, 100, 200, 200], "green part"), ([102, 100, 200, 200], "Green  Part"))
        self.assertEqual(["green part", "Green  Part"], [label for label, _ in _boxes(answer, classes=self.CLASSES)])

    def test_a_box_named_by_none_is_ambiguous(self) -> None:
        answer = _answer(([100, 100, 200, 200], None), ([400, 100, 500, 200], "red part"))
        self.assertEqual([AMBIGUOUS_LABEL, "red part"], [label for label, _ in _boxes(answer, classes=self.CLASSES)])

    def test_an_object_one_box_names_and_another_does_not_is_ambiguous(self) -> None:
        answer = _answer(([100, 100, 200, 200], "green part"), ([101, 101, 200, 200], None))
        self.assertEqual([AMBIGUOUS_LABEL] * 2, [label for label, _ in _boxes(answer, classes=self.CLASSES)])

    def test_a_label_outside_the_list_is_kept_as_written(self) -> None:
        """It names no description; the camera source maps it onto none and it stays an obstacle."""
        answer = _answer(([100, 100, 200, 200], "orange part"), ([400, 100, 500, 200], "green part"))
        self.assertEqual(["orange part", "green part"], [label for label, _ in _boxes(answer, classes=self.CLASSES)])

    def test_one_description_reads_as_it_always_did(self) -> None:
        answer = _answer(([100, 100, 200, 200], "cube"), ([104, 100, 200, 200], "grey cube"),
                         ([400, 100, 500, 200], None))
        self.assertEqual(["cube", "grey cube", "grey cube"], [label for label, _ in _boxes(answer, prompt="grey cube")])
        self.assertEqual(
            parse_grounding_response(answer, image_width=W, image_height=H, fallback_label="grey cube"),
            parse_grounding_response(answer, image_width=W, image_height=H, fallback_label="grey cube", classes=()),
        )


class TheGrounderLabelsTests(unittest.TestCase):
    """Through ``detect_all``: the prompt says whether the answer is a class list's."""

    ANSWER = _answer(([100, 100, 300, 400], "green part"), ([110, 100, 300, 400], "red part"),
                     ([600, 100, 800, 400], "red part"), ([600, 500, 800, 900], None))

    def test_a_class_list_s_answer_fails_closed(self) -> None:
        grounder, _, _ = _grounder(self.ANSWER)
        found = grounder.detect_all(_image(), "green part | red part")
        self.assertEqual([AMBIGUOUS_LABEL, AMBIGUOUS_LABEL, "red part", AMBIGUOUS_LABEL], [det.label for det in found])

    def test_the_same_answer_to_one_description_reads_as_ever(self) -> None:
        grounder, _, _ = _grounder(self.ANSWER)
        found = grounder.detect_all(_image(), "each separate part")
        self.assertEqual(["green part", "red part", "red part", "each separate part"], [det.label for det in found])


if __name__ == "__main__":
    unittest.main()
