"""A class list is grounded one description per Qwen call, each box under the description it was asked for, and what no
rule may take stays an obstacle or goes (the owner, 2026-10-15: every loop, group box and box written twice of the
cell's Thursday came from a class list asked in one call, none from a description asked alone).

No weights and no GPU: the model is a double that answers fixed strings, one per call, and its processor keeps what it
was asked. What the real model writes is ``tests/models/vlm/test_vlm_class_list_inference.py`` (GPU, outside the gate).

* ``class_list_prompt(["green part", "red part"])`` is ``"green part | red part"``, and ``classes_of`` reads it back.
* Each description is asked with today's instruction, byte for byte, in a call of its own; the answer's end, the lookup
  and the decode graphs are the same for every call.
* Each box comes back under the description it was asked for, whatever the model called it. Boxes over one object
  (IoU >= 0.7) under two descriptions are all ``ambiguous``, never a target (the owner, 2026-10-09: sorting places no
  part by guess), and a box that holds two others at most half its size is a box round a pile or a bin, and goes.
"""

from __future__ import annotations

import contextlib
import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.models.detection.types import Detection
from src.models.vlm import parse_grounding_response
from src.models.vlm.parsing import AMBIGUOUS_IOU, AMBIGUOUS_LABEL, CoordinateSpace, claimed_once
from src.models.vlm.qwen import (
    CLASS_LIST_SEPARATOR,
    GROUNDING_INSTRUCTION,
    Qwen3VLGrounder,
    class_list_prompt,
    classes_of,
)

#: Today's instruction around one description, spelled out here rather than read from the module: a change to the
#: words the model is asked is a change to this test too.
TODAY = ("Locate every object matching this description and return ONLY a JSON array, no prose:\n"
         '[{"bbox_2d": [x0, y0, x1, y1], "label": "<short name>"}]\n'
         "Coordinates are absolute pixels in this image. "
         "If no object matches, return an empty array []. Do not guess.\n\n"
         "Description: ")
W, H = 640, 480
#: The end token of the model double below.
END = 1_000_001


def _boxes(answer: str, *, prompt: str = "green part") -> list[tuple[str, list[float]]]:
    """``(label, box)`` of every box read from ``answer``, its numbers taken as pixels."""
    return [(det.label, det.box) for det in parse_grounding_response(
        answer, image_width=W, image_height=H, fallback_label=prompt, space=CoordinateSpace.ABSOLUTE)]


def _labels(*boxes: tuple[list[float], str]) -> list[str]:
    """The labels :func:`claimed_once` leaves on ``(box, label)`` pairs, the class list being green and red parts."""
    return [det.label for det in claimed_once(
        [Detection(box=box, x_center=0.0, y_center=0.0, label=label, score=1.0) for box, label in boxes],
        "green part | red part")]


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


# --- the grounder, through doubles of the processor and the model ---------------------------------------------------


class _Batch(dict):  # type: ignore[type-arg]
    def to(self, device: Any) -> "_Batch":
        return self


class _Processor:
    """Keeps every conversation it was handed, and decodes the answers in turn, the last one again once they run out."""

    def __init__(self, answer: "str | list[str]") -> None:
        self.answers = [answer] if isinstance(answer, str) else list(answer)
        self.decoded = 0
        self.asked: list[Any] = []
        self.tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: None, eos_token_id=END)

    def apply_chat_template(self, messages: Any, **_: Any) -> _Batch:
        self.asked.append(messages)
        return _Batch(input_ids=np.arange(7).reshape(1, -1), attention_mask=np.ones((1, 7)))

    def batch_decode(self, ids: Any, skip_special_tokens: bool = False) -> list[str]:
        answer = self.answers[min(self.decoded, len(self.answers) - 1)]
        self.decoded += 1
        return [answer]


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


def _grounder(answer: "str | list[str]", **settings: Any) -> tuple[Qwen3VLGrounder, _Processor, _Model]:
    processor, model = _Processor(answer), _Model()
    grounder = Qwen3VLGrounder(model_id="Qwen/Qwen3-VL-4B-Instruct", **settings)
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

    def test_a_class_list_is_asked_one_description_per_call(self) -> None:
        grounder, processor, model = _grounder("[]")
        grounder.detect_all(_image(), class_list_prompt(["green part", "red part"]))
        self.assertEqual([TODAY + "green part", TODAY + "red part"], [_asked(processor, call) for call in (0, 1)])
        self.assertEqual(2, len(model.calls), "one call per description")
        self.assertEqual(GROUNDING_INSTRUCTION.format(prompt="green part"), _asked(processor, 0))

    def test_every_call_ends_at_the_array_and_takes_the_same_lookup_and_graphs(self) -> None:
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
        self.assertEqual([taken[0]] * 3, taken)
        self.assertEqual(["_EndOfAnswer"], taken[0]["stops"])
        self.assertEqual("auto", taken[0]["graphs_mode"])
        self.assertEqual([10, 10, 10], [call["prompt_lookup_num_tokens"] for call in model.calls])


class TheAnswerFailsClosedTests(unittest.TestCase):
    """What a class list's boxes may not hand a rule: a part two descriptions claim (``claimed_once``)."""

    def test_one_object_boxed_under_two_descriptions_is_ambiguous(self) -> None:
        self.assertEqual([AMBIGUOUS_LABEL, AMBIGUOUS_LABEL, "green part"],
                         _labels(([100, 100, 200, 200], "green part"), ([104, 100, 200, 200], "red part"),
                                 ([400, 100, 500, 200], "green part")))

    def test_two_parts_side_by_side_keep_their_descriptions(self) -> None:
        self.assertEqual(["green part", "red part"],
                         _labels(([100, 100, 200, 200], "green part"), ([150, 100, 250, 200], "red part")))

    def test_the_overlap_that_makes_one_object_is_inclusive(self) -> None:
        self.assertEqual(0.7, AMBIGUOUS_IOU)
        self.assertEqual([AMBIGUOUS_LABEL] * 2,
                         _labels(([0, 0, 100, 100], "green part"), ([0, 0, 70, 100], "red part")))   # IoU 0.70
        self.assertEqual(["green part", "red part"],
                         _labels(([0, 0, 100, 100], "green part"), ([0, 0, 69, 100], "red part")))   # IoU 0.69

    def test_a_chain_of_overlaps_is_one_object(self) -> None:
        """A and B overlap, B and C overlap, A and C less: all three are the one object, and C names another
        description, so not even A keeps its own."""
        self.assertEqual([AMBIGUOUS_LABEL] * 3,
                         _labels(([0, 0, 100, 100], "green part"), ([0, 0, 100, 80], "green part"),
                                 ([0, 0, 100, 60], "red part")))

    def test_one_description_boxed_twice_keeps_one(self) -> None:
        """The second box, the same object written again, goes (2026-10-12)."""
        answer = _answer(([100, 100, 200, 200], "green part"), ([102, 100, 200, 200], "Green  Part"))
        self.assertEqual(["green part"], [label for label, _ in _boxes(answer)])

    def test_one_description_reads_as_it_always_did(self) -> None:
        answer = _answer(([100, 100, 200, 200], "cube"), ([300, 100, 400, 200], "grey cube"),
                         ([400, 300, 500, 400], None))
        self.assertEqual(["cube", "grey cube", "grey cube"], [label for label, _ in _boxes(answer, prompt="grey cube")])


class TheGrounderLabelsTests(unittest.TestCase):
    """Through ``detect_all``: a class list's boxes, each description asked on its own."""

    def test_each_box_comes_under_the_description_it_was_asked_for(self) -> None:
        grounder, _, _ = _grounder([_answer(([100, 100, 300, 400], "cube")), _answer(([600, 100, 800, 400], None))])
        found = grounder.detect_all(_image(), "green part | red part")
        self.assertEqual(["green part", "red part"], [det.label for det in found])

    def test_one_object_both_descriptions_boxed_is_ambiguous_and_stays(self) -> None:
        grounder, _, _ = _grounder([_answer(([100, 100, 300, 400], "green part"), ([600, 100, 800, 400], None)),
                                    _answer(([110, 100, 300, 400], "red part"))])
        found = grounder.detect_all(_image(), "green part | red part")
        self.assertEqual([AMBIGUOUS_LABEL, "green part", AMBIGUOUS_LABEL], [det.label for det in found])

    def test_a_box_round_two_parts_goes(self) -> None:
        pile = ([50, 50, 450, 450], "green part")
        grounder, _, _ = _grounder([_answer(pile, ([60, 60, 160, 160], "green part")),
                                    _answer(([300, 300, 400, 400], "red part"))])
        found = grounder.detect_all(_image(), "green part | red part")
        # On Qwen's grid of 1000: the 640 x 480 picture's boxes, the pile's gone.
        self.assertEqual([("green part", [38.4, 28.8, 102.4, 76.8]), ("red part", [192.0, 144.0, 256.0, 192.0])],
                         [(det.label, [round(value, 1) for value in det.box]) for det in found])

    def test_one_description_reads_as_ever(self) -> None:
        grounder, _, _ = _grounder(_answer(([100, 100, 300, 400], "green part"), ([600, 100, 800, 400], None)))
        found = grounder.detect_all(_image(), "each separate part")
        self.assertEqual(["green part", "each separate part"], [det.label for det in found])


if __name__ == "__main__":
    unittest.main()
