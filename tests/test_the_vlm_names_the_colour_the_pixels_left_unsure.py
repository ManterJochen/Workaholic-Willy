"""The VLM names the colour a part's pixels left unsure: one colour word for a crop of its box (F1, 2026-10-08).

Where the camera source's colour check cannot tell from the pixels (a lightness between grey and white, a share between
the rules, a part of fewer than 50 pixels), it asks the backend's ``name_colour`` for one colour word: never a yes or a
no, which an instruct model gives too easily. ``Qwen3VLGrounder.name_colour`` crops the box grown by 15 %, scales it
to a long side of 224 px (about 50 image tokens) and asks greedily for at most three tokens, under the one generate
lock. Every backend a cell composes forwards the question: ``TwoStageBackend`` to its detector, counting a question
that raises as a failure of the model (so a task reads ``detector_failed``, never "nothing left");
``GuardedVlmBackend`` while it is not degraded; ``RoutedPerceptionBackend`` to its VLM route.

No weights: the model, its processor and torch are doubles.
"""

from __future__ import annotations

import contextlib
import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.models.perception_backend import TwoStageBackend, failures_of


class _Batch(dict):  # type: ignore[type-arg]
    def to(self, device: Any) -> "_Batch":
        return self


class _Processor:
    """Takes the messages it is handed, and decodes every answer as the model's word."""

    def __init__(self, word: str) -> None:
        self.word = word
        self.messages: list[Any] = []

    def apply_chat_template(self, messages: Any, **_: Any) -> _Batch:
        self.messages.append(messages)
        return _Batch(input_ids=np.arange(9).reshape(1, -1))

    def batch_decode(self, ids: Any, skip_special_tokens: bool = False) -> list[str]:
        return [self.word]


class _Model:
    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> np.ndarray:
        self.calls.append(kwargs)
        return np.concatenate([kwargs["input_ids"], np.full((1, 2), 7)], axis=1)


class _Torch:
    @staticmethod
    def inference_mode() -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()


def _loaded(word: str) -> tuple[Any, _Processor, _Model]:
    """A grounder whose weights are in hand: the doubles."""
    from src.models.vlm import Qwen3VLGrounder

    grounder = Qwen3VLGrounder(model_id="Qwen/Qwen3-VL-4B-Instruct")
    processor, model = _Processor(word), _Model()
    grounder._processor, grounder._model, grounder._torch = processor, model, _Torch()  # noqa: SLF001
    return grounder, processor, model


class TheQuestionTests(unittest.TestCase):
    def test_one_colour_word_is_asked_for_a_crop_of_the_box_greedily_and_short(self) -> None:
        grounder, processor, model = _loaded(" Grey\n")
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        image[50:100, 100:200] = (0, 0, 255)

        answer = grounder.name_colour(image, (100.0, 50.0, 200.0, 100.0))

        self.assertEqual("Grey", answer)
        [messages] = processor.messages
        [image_part, text_part] = messages[0]["content"]
        crop = image_part["image"]
        self.assertEqual(224, max(crop.shape[:2]), "the crop's long side is 224 px")
        self.assertEqual((112, 224, 3), crop.shape, "the box grown by 15 %, scaled whole")
        self.assertEqual((255, 0, 0), tuple(int(v) for v in crop[56, 112]), "the crop is RGB, as the processor reads")
        self.assertEqual(grounder.COLOUR_QUESTION, text_part["text"])
        self.assertIn("one colour word", text_part["text"])
        [call] = model.calls
        self.assertEqual(3, call["max_new_tokens"])
        self.assertIs(False, call["do_sample"])

    def test_the_question_takes_its_turn_on_the_generate_lock(self) -> None:
        grounder, _, model = _loaded("white")
        seen: list[bool] = []
        generate = model.generate

        def generate_under_the_lock(**kwargs: Any) -> np.ndarray:
            seen.append(grounder._generate_lock.locked())  # noqa: SLF001
            return generate(**kwargs)

        model.generate = generate_under_the_lock  # type: ignore[method-assign]
        grounder.name_colour(np.zeros((40, 40, 3), np.uint8), (5, 5, 30, 30))
        self.assertEqual([True], seen)

    def test_a_box_at_the_image_s_edge_is_cut_there(self) -> None:
        grounder, processor, _ = _loaded("red")
        grounder.name_colour(np.zeros((100, 100, 3), np.uint8), (0.0, 0.0, 100.0, 50.0))
        crop = processor.messages[0][0]["content"][0]["image"]
        self.assertEqual((121, 224), crop.shape[:2], "the crop stops at the image: 100 x 54 px, scaled by 2.24")

    def test_the_answer_is_logged(self) -> None:
        grounder, _, _ = _loaded("green")
        with self.assertLogs("src.models.vlm.qwen", level="INFO") as said:
            grounder.name_colour(np.zeros((40, 40, 3), np.uint8), (5, 5, 30, 30))
        self.assertIn("named the colour", said.output[-1])
        self.assertIn("'green'", said.output[-1])


class _Naming:
    def __init__(self, answer: str = "grey", *, raises: bool = False) -> None:
        self.answer, self.raises = answer, raises
        self.asked = 0

    def detect_all(self, bgr: Any, prompt: str) -> list[Any]:
        return []

    def name_colour(self, image_bgr: Any, box: Any) -> str:
        self.asked += 1
        if self.raises:
            raise RuntimeError("CUDA error: out of memory")
        return self.answer


class _Phrases:
    """A phrase grounder: it names no colour."""

    def detect_all(self, bgr: Any, prompt: str) -> list[Any]:
        return []


_IMAGE = np.zeros((40, 40, 3), np.uint8)
_BOX = (5.0, 5.0, 30.0, 30.0)


class EveryBackendForwardsTheQuestionTests(unittest.TestCase):
    def test_the_two_stage_backend_asks_its_detector(self) -> None:
        self.assertEqual("grey", TwoStageBackend(_Naming("grey"), segmenter=None).name_colour(_IMAGE, _BOX))
        self.assertEqual("", TwoStageBackend(_Phrases(), segmenter=None).name_colour(_IMAGE, _BOX))

    def test_a_question_that_raises_answers_nothing_and_counts_as_the_model_s_failure(self) -> None:
        backend = TwoStageBackend(_Naming(raises=True), segmenter=None)
        self.assertEqual("", backend.name_colour(_IMAGE, _BOX))
        self.assertEqual(1, failures_of(backend))
        self.assertIn("RuntimeError naming a colour", backend.last_failure)

    def test_the_guarded_backend_asks_the_vlm_until_it_has_degraded(self) -> None:
        from src.models.vlm import GuardedVlmBackend

        naming = _Naming("white")
        guarded = GuardedVlmBackend(vlm=TwoStageBackend(naming, segmenter=None), model_id="qwen")
        self.assertEqual("white", guarded.name_colour(_IMAGE, _BOX))

        raising = TwoStageBackend(_Naming(raises=True), segmenter=None)
        counting = GuardedVlmBackend(vlm=raising, model_id="qwen")
        self.assertEqual("", counting.name_colour(_IMAGE, _BOX))
        self.assertEqual(1, failures_of(counting))
        self.assertIn("naming a colour", counting.last_failure)

        class _Unloadable:
            def perceive(self, image_bgr: Any, prompt: str) -> tuple[Any, ...]:
                raise OSError("no weights on disk")

            def name_colour(self, image_bgr: Any, box: Any) -> str:  # pragma: no cover - never asked once degraded
                raise AssertionError("a degraded backend asked the VLM")

        degraded = GuardedVlmBackend(vlm=_Unloadable(), model_id="qwen", degrade=True,
                                     fallback_factory=lambda: TwoStageBackend(_Phrases(), segmenter=None))
        with self.assertLogs("src.models.vlm.availability", level="WARNING"):
            degraded.perceive(_IMAGE, "grey cube")
        self.assertTrue(degraded.degraded)
        self.assertEqual("", degraded.name_colour(_IMAGE, _BOX))

    def test_the_routed_backend_asks_its_vlm_route(self) -> None:
        from src.models.routed_backend import RoutedPerceptionBackend
        from src.models.routing import Route

        naming = _Naming("green")
        routed = RoutedPerceptionBackend(
            simple_factory=lambda: TwoStageBackend(_Phrases(), segmenter=None),
            vlm_factory=lambda: TwoStageBackend(naming, segmenter=None),
            router=SimpleNamespace(route=lambda prompt: None),  # type: ignore[arg-type]
        )
        self.assertEqual("green", routed.name_colour(_IMAGE, _BOX))
        self.assertEqual((Route.VLM,), routed.built_routes())
        self.assertEqual(1, naming.asked)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
