"""How fast the VLM writes its answers: the answer's end, the answer lookup, and the switches that choose them.

No weights and no GPU. A grounding call costs one pass of the model per token of its answer (measured 2026-10-08:
about 32 ms on the dev box's 4B, about 140 ms on the cell's 8B, the image and the instruction nearly free), so
everything here is about writing fewer passes without moving a box:

* **The answer ends where its JSON array closes** (``stop_at_answer_end``, on). The model's own end comes one pass
  later, three after a code fence. The parser reads the same boxes from the answer cut there, which is pinned here
  on answers the model really wrote.
* **The answer lookup** (``prompt_lookup_tokens`` for grounding, ``text_prompt_lookup_tokens`` for the command
  reader's text answers, both off). The next tokens are proposed from the answer itself, after the image only, never
  an image placeholder (one without features fails the pass) and never an end token; the model checks them in one
  pass. A lookup that fails falls back to the plain answer and stays off until a reload.
* **The switches** live in the ``vlm`` block and reach the process's one copy at every build.

The real model's side of the same claims is ``tests/test_vlm_answer_speed_inference.py``.
"""

from __future__ import annotations

import contextlib
import threading
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
from pydantic import ValidationError

from src.config.schema.models.models_schema import PipelineConfig, VlmConfig
from src.models.vlm import Qwen3VLGrounder, parse_grounding_response, shared_vlm
from src.models.vlm.command import extract_command_object
from src.models.vlm.qwen import _AnswerLookup, _EndOfAnswer, answer_array_closed, answer_object_closed

#: Answers Qwen3-VL-4B wrote on the cell's wrist frames (2026-10-08), and the code-fenced shape the cell's 8B writes.
THREE_CUBES = ('[{"bbox_2d": [165, 400, 233, 502], "label": "each separate gray cube"}, '
               '{"bbox_2d": [513, 454, 570, 556], "label": "each separate gray cube"}, '
               '{"bbox_2d": [349, 680, 401, 780], "label": "each separate gray cube"}]')
YELLOW_BIN = '[{"bbox_2d": [697, 223, 985, 526], "label": "yellow bin"}]'
FENCED = '```json\n[\n\t{"bbox_2d": [697, 223, 985, 526], "label": "yellow bin"}\n]\n```'
ABSENT = "[]"

#: Special ids of the character-level doubles below: no character has them.
END, VISION_START, IMAGE, VISION_END = 1_000_001, 1_000_002, 1_000_003, 1_000_004


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - the CI image installs the CPU wheel
        raise unittest.SkipTest(f"torch is not installed: {exc}") from exc
    return torch


def _ids(text: str) -> list[int]:
    """One token per character: what the doubles here read and write."""
    return [ord(char) for char in text]


class _CharProcessor:
    """Decodes the character-level ids; special ids are skipped as ``skip_special_tokens`` skips them."""

    def batch_decode(self, sequences: Any, skip_special_tokens: bool = False) -> list[str]:
        out = []
        for sequence in sequences:
            ids = [int(token) for token in (sequence.tolist() if hasattr(sequence, "tolist") else sequence)]
            out.append("".join(chr(token) for token in ids if token < 0x110000 or not skip_special_tokens))
        return out


def _cut_where_it_closes(answer: str) -> str:
    """The answer as the stop leaves it: up to the first character after which it reads as closed."""
    for end in range(1, len(answer) + 1):
        if answer_array_closed(answer[:end]):
            return answer[:end]
    return answer


def _parsed(text: str) -> list[tuple[list[float], str]]:
    return [(det.box, det.label) for det in parse_grounding_response(
        text, image_width=1280, image_height=720, fallback_label="each separate gray cube")]


class TheAnswerEndsWhereItsArrayClosesTests(unittest.TestCase):
    def test_a_closed_array_ends_the_answer(self) -> None:
        for answer in (THREE_CUBES, YELLOW_BIN, ABSENT):
            with self.subTest(answer=answer[:30]):
                self.assertTrue(answer_array_closed(answer))
                self.assertFalse(answer_array_closed(answer[:-1]), "a list still open is not ended")

    def test_a_code_fence_is_read_through(self) -> None:
        self.assertTrue(answer_array_closed(FENCED[:FENCED.index("]\n```") + 1]))
        self.assertFalse(answer_array_closed("```json"), "the fence's own line is no array yet")

    def test_prose_first_never_ends_early(self) -> None:
        """The parser reads prose answers from the first ``[`` to the last ``]``: cutting one would change that."""
        self.assertFalse(answer_array_closed('Here they are [1]: [{"bbox_2d": [1, 2, 3, 4]}]'))
        self.assertFalse(answer_array_closed('{"bbox_2d": [1, 2, 3, 4]}'), "an object first is not a list")

    def test_brackets_inside_a_label_are_text(self) -> None:
        answer = '[{"bbox_2d": [1, 2, 3, 4], "label": "cube ] [ \\" ]"}]'
        self.assertEqual(_cut_where_it_closes(answer), answer)

    def test_unbalanced_braces_never_end_early(self) -> None:
        """An extra ``}`` mid-list: the whole answer is read object by object, so it must be written whole."""
        answer = '[{"bbox_2d": [1, 2, 3, 4]}}, {"bbox_2d": [5, 6, 7, 8]}]'
        self.assertFalse(answer_array_closed(answer[:answer.index("}}") + 2]))

    def test_the_cut_answer_reads_as_the_whole_one(self) -> None:
        """The claim the default rests on: the same boxes, the same labels, from every answer shape measured."""
        for whole in (THREE_CUBES + "\n", YELLOW_BIN, FENCED, ABSENT, THREE_CUBES.replace("}, {", "},{")):
            with self.subTest(answer=whole[:40]):
                cut = _cut_where_it_closes(whole)
                self.assertLessEqual(len(cut), len(whole))
                self.assertEqual(_parsed(cut), _parsed(whole))
        self.assertEqual(len(_parsed(THREE_CUBES)), 3)


class TheStoppingCriterionTests(unittest.TestCase):
    """``_EndOfAnswer`` reads only what the model wrote, never the prompt's own example."""

    def setUp(self) -> None:
        self.torch = _torch()
        self.prompt = _ids('Return [{"bbox_2d": [x0, y0, x1, y1]}]. Description: cube') + [END]

    def _done(self, written: str, *, extra: list[int] | None = None) -> bool:
        ids = self.prompt + _ids(written) + (extra or [])
        criterion = _EndOfAnswer(_CharProcessor(), self.torch, len(self.prompt))
        verdict = criterion(self.torch.tensor([ids]), None)
        self.assertEqual(tuple(verdict.shape), (1,))
        return bool(verdict[0])

    def test_open_until_the_array_closes(self) -> None:
        self.assertFalse(self._done(""), "nothing written yet; the prompt's example is not the answer")
        self.assertFalse(self._done(YELLOW_BIN[:-1]))
        self.assertTrue(self._done(YELLOW_BIN))
        self.assertTrue(self._done(YELLOW_BIN, extra=[END]))

    def test_a_long_answer_is_read_whole_when_its_newest_tokens_close_it(self) -> None:
        many = "[" + ", ".join(f'{{"bbox_2d": [{i}, 2, 3, 4], "label": "part"}}' for i in range(40)) + "]"
        self.assertGreater(len(many), 10 * _EndOfAnswer.NEWEST)
        self.assertFalse(self._done(many[:-1]))
        self.assertTrue(self._done(many))


#: Command readings the 4B wrote: the earlier format with every key (2026-10-08), and the compact one the reader asks for
#: since; each ended with the end token right after its object.
FULL_COMMAND = ('{ "intent": "none", "object": "", "object_said": null, "place": null, "place_said": null, '
                '"place_pose": null, "scope": "once", "count": null, "return_to": null }')
COMPACT_COMMAND = ('{"intent":"task","object":"gray cube","object_said":"grauen Würfel","place":"yellow bin",'
                   '"place_said":"in die Gelbe Kiste","scope":"until_empty"}')


def _cut_where_the_object_closes(answer: str) -> str:
    for end in range(1, len(answer) + 1):
        if answer_object_closed(answer[:end]):
            return answer[:end]
    return answer


class TheCommandReadingEndsWhereItsObjectClosesTests(unittest.TestCase):
    """A text answer the prefill opens with ``{`` ends where its object closes: one pass before the end token, about
    140 ms a reading on the cell. The reader takes an answer's first whole object, and the cut answer reads the same."""

    def test_a_whole_object_ends_the_answer(self) -> None:
        for answer in (FULL_COMMAND, COMPACT_COMMAND, '{"intent":"stop"}'):
            with self.subTest(answer=answer[:30]):
                self.assertTrue(answer_object_closed(answer))
                self.assertFalse(answer_object_closed(answer[:-1]), "an object still open is not ended")

    def test_braces_inside_a_string_are_text(self) -> None:
        answer = '{"intent":"task","object_said":"den Würfel {oben} \\" }"}'
        self.assertEqual(answer, _cut_where_the_object_closes(answer))

    def test_an_object_that_does_not_read_where_it_closes_never_ends_early(self) -> None:
        """The model ends it, and the reader looks further, as it always did."""
        for answer in ('{{"intent":"none"}}', '{"intent":"task" "object":"cube"}'):
            with self.subTest(answer=answer):
                self.assertFalse(any(answer_object_closed(answer[:end]) for end in range(1, len(answer) + 1)))

    def test_prose_or_an_array_first_never_ends_early(self) -> None:
        self.assertFalse(answer_object_closed('Here: {"intent":"stop"}'))
        self.assertFalse(answer_object_closed('[{"intent":"none"}]'))

    def test_the_cut_answer_reads_as_the_whole_one(self) -> None:
        for whole in (FULL_COMMAND + "\n", COMPACT_COMMAND, '{"intent":"stop"}\n{"intent":"none"}',
                      '{"intent":"none"} Hope it helps!', '{{"intent":"none"}'):
            with self.subTest(answer=whole[:40]):
                cut = _cut_where_the_object_closes(whole)
                self.assertTrue(whole.startswith(cut))
                self.assertEqual(extract_command_object(whole), extract_command_object(cut))

    def test_the_criterion_reads_the_answer_with_its_prefill_in_front(self) -> None:
        torch = _torch()
        prompt = _ids("Answer with one JSON object.") + [END] + _ids("{")
        end = _EndOfAnswer(_CharProcessor(), torch, len(prompt), bracket="}", closed=answer_object_closed, prefix="{")
        for written, done in (('"intent":"none"', False), ('"intent":"none"}', True), ("", False)):
            with self.subTest(written=written):
                self.assertEqual(done, bool(end(torch.tensor([prompt + _ids(written)]), None)[0]))


def _lookup(torch: Any, *, tokens: int = 10, max_length: int = 10_000, after: int | None = VISION_END) -> _AnswerLookup:
    return _AnswerLookup(
        torch, tokens=tokens, digits=frozenset(_ids("0123456789")), never=frozenset({VISION_START, IMAGE, VISION_END}),
        ends=frozenset({END}), after=after, max_length=max_length,
    )


class TheAnswerLookupTests(unittest.TestCase):
    """What ``_AnswerLookup`` proposes: the previous box's tail, never an image token, never an end."""

    def setUp(self) -> None:
        self.torch = _torch()
        self.prompt = ([VISION_START] + [IMAGE] * 6 + [VISION_END]
                       + _ids('Return [{"bbox_2d": [x0, y0, x1, y1], "label": "<short name>"}]\nDescription: cube')
                       + [END] + _ids("\nassistant\n"))

    def _proposal(self, written: str, **kwargs: Any) -> str:
        ids = self.torch.tensor([self.prompt + _ids(written)])
        candidates, logits = _lookup(self.torch, **kwargs).get_candidates(ids)
        self.assertIsNone(logits)
        self.assertTrue(self.torch.equal(candidates[:, : ids.shape[1]], ids), "the answer so far is kept as it is")
        return "".join(chr(int(token)) for token in candidates[0, ids.shape[1]:].tolist())

    def test_a_box_s_tail_comes_from_the_previous_box_whatever_its_digits(self) -> None:
        first = '[{"bbox_2d": [165, 400, 233, 502], "label": "cube"}, '
        self.assertEqual(self._proposal(first + '{"bbox_2d": [513, 454, 570, 556'), '], "label"')
        self.assertEqual(self._proposal(first + '{"bbox_2d": [513, 454, 570, 556', tokens=20),
                         '], "label": "cube"},')

    def test_a_separator_is_proposed_after_a_number(self) -> None:
        proposal = self._proposal('[{"bbox_2d": [165, 400, 233, 502], "label": "cube"}, {"bbox_2d": [513')
        self.assertTrue(proposal.startswith(", "), proposal)

    def test_nothing_before_the_image_is_searched(self) -> None:
        """Before the image lie the chat's own turn markers, and right after them the image's placeholders."""
        ids = self.torch.tensor([_ids("xyz") + [VISION_START, IMAGE, VISION_END] + _ids("q") + _ids("xy")])
        candidates, _ = _lookup(self.torch).get_candidates(ids)
        self.assertIs(candidates, ids, "'xy' matched only before the image: nothing is proposed")

    def test_an_image_placeholder_is_never_proposed(self) -> None:
        """An image token with no features behind it fails the pass ("Image features and image tokens do not
        match"): transformers' own prompt lookup proposed exactly that on every grounding call."""
        ids = self.torch.tensor([_ids("ab") + [IMAGE, VISION_END] + _ids("ab")])
        candidates, _ = _lookup(self.torch, after=None).get_candidates(ids)
        self.assertIs(candidates, ids)

    def test_an_end_token_is_never_proposed(self) -> None:
        """An end accepted in the middle of a pass would not stop the answer."""
        proposal = self._proposal("Description: cu")
        self.assertEqual(proposal, "be", "the proposal stops before the prompt's end token")

    def test_no_match_proposes_nothing(self) -> None:
        ids = self.torch.tensor([self.prompt + _ids("€")])
        candidates, _ = _lookup(self.torch).get_candidates(ids)
        self.assertIs(candidates, ids)

    def test_the_answer_s_length_bound_is_kept(self) -> None:
        written = '[{"bbox_2d": [165, 400, 233, 502], "label": "cube"}, {"bbox_2d": [513, 454, 570, 556'
        length = len(self.prompt) + len(written)
        self.assertEqual(self._proposal(written, max_length=length + 4), "], ")
        ids = self.torch.tensor([self.prompt + _ids(written)])
        candidates, _ = _lookup(self.torch, max_length=length + 1).get_candidates(ids)
        self.assertIs(candidates, ids, "room for the model's own token only")

    def test_a_text_question_is_searched_whole(self) -> None:
        """No image: the instruction's own keys come back in the answer, which is what is proposed."""
        system = _ids('Answer {"intent": "pick", "object": null}. ')
        ids = self.torch.tensor([system + _ids('{"intent": "pick", "obj')])
        candidates, _ = _lookup(self.torch, after=None).get_candidates(ids)
        self.assertEqual("".join(chr(int(t)) for t in candidates[0, ids.shape[1]:].tolist()), 'ect": null')


# --- the grounder's decoding, through doubles of the processor and the model ----------------------------------------


class _Batch(dict):  # type: ignore[type-arg]
    def to(self, device: Any) -> "_Batch":
        return self


class _Processor:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: None, eos_token_id=END)

    def apply_chat_template(self, messages: Any, **_: Any) -> _Batch:
        return _Batch(input_ids=np.arange(7).reshape(1, -1), attention_mask=np.ones((1, 7)))

    def batch_decode(self, ids: Any, skip_special_tokens: bool = False) -> list[str]:
        return [self.answer]


class _Model:
    device = "cpu"

    def __init__(self, *, refuse_lookup: BaseException | None = None, hold: threading.Event | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.generators: list[Any] = []
        self.refuse_lookup = refuse_lookup
        self.hold = hold
        self.entered = threading.Event()
        self.config = SimpleNamespace(image_token_id=IMAGE, video_token_id=None, vision_start_token_id=VISION_START,
                                      vision_end_token_id=VISION_END)
        self.generation_config = SimpleNamespace(eos_token_id=[END])

    def generate(self, **kwargs: Any) -> np.ndarray:
        self.calls.append(kwargs)
        self.entered.set()
        if self.hold is not None:
            self.hold.wait(5.0)
        if "prompt_lookup_num_tokens" in kwargs:
            # What transformers' assisted decoding does first: ask the model for its candidate generator.
            self.generators.append(self._get_candidate_generator(
                generation_config=SimpleNamespace(max_length=2055), input_ids=kwargs["input_ids"]))
            if self.refuse_lookup is not None:
                raise self.refuse_lookup
        return np.concatenate([kwargs["input_ids"], np.full((1, 4), 99)], axis=1)


class _Torch:
    @staticmethod
    def inference_mode() -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    cuda = SimpleNamespace(empty_cache=lambda: None)


def _grounder(model: _Model, answer: str = YELLOW_BIN, **settings: Any) -> Qwen3VLGrounder:
    grounder = Qwen3VLGrounder(model_id="Qwen/Qwen3-VL-4B-Instruct", local=True, **settings)
    grounder._load = lambda: (_Processor(answer), model, _Torch())  # type: ignore[method-assign]
    return grounder


def _look(grounder: Qwen3VLGrounder) -> list[Any]:
    return grounder.detect_all(np.zeros((72, 128, 3), np.uint8), "yellow bin")


class TheGroundersDecodingTests(unittest.TestCase):
    def test_the_defaults_are_today_s_answer_ended_at_its_array(self) -> None:
        grounder = Qwen3VLGrounder(model_id="Qwen/Qwen3-VL-4B-Instruct")
        self.assertEqual(grounder.prompt_lookup_tokens, 0)
        self.assertEqual(grounder.text_prompt_lookup_tokens, 0)
        self.assertTrue(grounder.stop_at_answer_end)
        self.assertEqual("auto", grounder.decode_graphs, "the graphs write the plain answers, byte for byte")
        self.assertIsNone(grounder.lookup_failure)

    def test_a_look_ends_at_the_array_and_looks_nothing_up_by_default(self) -> None:
        model = _Model()
        self.assertEqual(len(_look(_grounder(model))), 1)
        [call] = model.calls
        self.assertIs(call["do_sample"], False)
        self.assertNotIn("prompt_lookup_num_tokens", call)
        self.assertEqual([type(c).__name__ for c in call["stopping_criteria"]][-1], "_EndOfAnswer")

    def test_switched_off_the_answer_runs_to_the_model_s_own_end(self) -> None:
        model = _Model()
        _look(_grounder(model, stop_at_answer_end=False))
        self.assertNotIn("stopping_criteria", model.calls[0])

    def test_the_lookup_hands_assisted_decoding_this_copy_s_candidates(self) -> None:
        model = _Model()
        _look(_grounder(model, prompt_lookup_tokens=10))
        [call] = model.calls
        self.assertEqual(call["prompt_lookup_num_tokens"], 10)
        [generator] = model.generators
        self.assertIsInstance(generator, _AnswerLookup)
        self.assertFalse(hasattr(model, "_get_candidate_generator"), "the class's own generator is back after the call")

    def test_a_text_answer_uses_its_own_lookup_and_ends_where_its_object_closes(self) -> None:
        model = _Model()
        grounder = _grounder(model, answer='"intent": "none"}', text_prompt_lookup_tokens=8)
        self.assertEqual(grounder.answer_text("S", "U", max_new_tokens=160), '{"intent": "none"}')
        [call] = model.calls
        self.assertEqual(call["prompt_lookup_num_tokens"], 8)
        self.assertEqual(call["max_new_tokens"], 160)
        [end] = call["stopping_criteria"]
        self.assertIsInstance(end, _EndOfAnswer)
        self.assertIs(end._closed, answer_object_closed, "the command reader's answer is an object")  # noqa: SLF001
        self.assertEqual("{", end._prefix, "read with the prefill in front, which the prompt holds")  # noqa: SLF001

    def test_a_text_answer_runs_to_the_model_s_end_without_an_object_or_the_switch(self) -> None:
        model = _Model()
        grounder = _grounder(model, answer='{"intent": "none"}')
        grounder.answer_text("S", "U", prefill="")
        self.assertNotIn("stopping_criteria", model.calls[-1], "no prefill: nothing says the answer is an object")
        grounder.configure(stop_at_answer_end=False)
        grounder.answer_text("S", "U")
        self.assertNotIn("stopping_criteria", model.calls[-1])

    def test_each_lookup_switch_is_its_own_path_s(self) -> None:
        """The grounding lookup is not byte-identical, the command reader's was on every measured command: the
        two are switched apart."""
        model = _Model()
        grounder = _grounder(model, answer='"intent": "none"}', prompt_lookup_tokens=10)
        grounder.answer_text("S", "U")
        self.assertNotIn("prompt_lookup_num_tokens", model.calls[-1])
        grounder.configure(prompt_lookup_tokens=0, text_prompt_lookup_tokens=10)
        _look(grounder)
        self.assertNotIn("prompt_lookup_num_tokens", model.calls[-1])

    def test_a_failed_lookup_answers_without_it_until_a_reload(self) -> None:
        model = _Model(refuse_lookup=ValueError("Image features and image tokens do not match"))
        grounder = _grounder(model, prompt_lookup_tokens=10)
        self.assertEqual(len(_look(grounder)), 1, "the look still grounds")
        self.assertIn("ValueError", grounder.lookup_failure or "")
        self.assertEqual(["prompt_lookup_num_tokens" in call for call in model.calls], [True, False])
        _look(grounder)
        self.assertNotIn("prompt_lookup_num_tokens", model.calls[-1], "not tried again on this copy")
        grounder.unload()
        self.assertIsNone(grounder.lookup_failure)
        _look(grounder)
        self.assertIn("prompt_lookup_num_tokens", model.calls[-2], "a reload tries it again")

    def test_running_out_of_memory_skips_the_lookup_for_that_answer_only(self) -> None:
        class OutOfMemoryError(RuntimeError):
            pass

        model = _Model(refuse_lookup=OutOfMemoryError("CUDA out of memory"))
        grounder = _grounder(model, prompt_lookup_tokens=10)
        self.assertEqual(len(_look(grounder)), 1)
        self.assertIsNone(grounder.lookup_failure)

    def test_a_build_s_settings_never_wait_for_a_look_in_flight(self) -> None:
        """A grounding on the cell takes 15 s: the build that sets the switches meanwhile does not wait for it, and
        the look keeps what it started with."""
        hold = threading.Event()
        model = _Model(hold=hold)
        grounder = _grounder(model)
        grounder.ensure_loaded()
        look = threading.Thread(target=lambda: _look(grounder))
        look.start()
        self.assertTrue(model.entered.wait(5.0))
        try:
            setter = threading.Thread(target=lambda: grounder.configure(prompt_lookup_tokens=10))
            setter.start()
            setter.join(2.0)
            self.assertFalse(setter.is_alive(), "configure waited for the look in flight")
        finally:
            hold.set()
            look.join(5.0)
        self.assertNotIn("prompt_lookup_num_tokens", model.calls[0], "the look in flight kept its settings")
        _look(grounder)
        self.assertEqual(model.calls[-1]["prompt_lookup_num_tokens"], 10, "the next look takes the new ones")

    def test_configure_keeps_what_it_is_not_given(self) -> None:
        grounder = Qwen3VLGrounder(model_id="Qwen/Qwen3-VL-4B-Instruct", prompt_lookup_tokens=5)
        grounder.configure(stop_at_answer_end=False)
        self.assertEqual(grounder.prompt_lookup_tokens, 5)
        self.assertFalse(grounder.stop_at_answer_end)
        grounder.configure(prompt_lookup_tokens=0)
        self.assertEqual(grounder.prompt_lookup_tokens, 0)
        for wrong in ({"prompt_lookup_tokens": -1}, {"text_prompt_lookup_tokens": -1}):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                grounder.configure(**wrong)
        self.assertEqual((grounder.prompt_lookup_tokens, grounder.text_prompt_lookup_tokens), (0, 0),
                         "a refused setting changes nothing")


# --- the switches ---------------------------------------------------------------------------------------------------


class TheSwitchesTests(unittest.TestCase):
    def setUp(self) -> None:
        shared_vlm().forget()
        self.addCleanup(shared_vlm().forget)

    def test_the_schema_defaults(self) -> None:
        vlm = VlmConfig()
        self.assertEqual(vlm.prompt_lookup_tokens, 0)
        self.assertEqual(vlm.text_prompt_lookup_tokens, 0)
        self.assertTrue(vlm.stop_at_answer_end)
        self.assertEqual(vlm.decode_graphs, "auto")
        for wrong in ({"prompt_lookup_tokens": -1}, {"text_prompt_lookup_tokens": -1}):
            with self.subTest(wrong=wrong), self.assertRaises(ValidationError):
                VlmConfig(**wrong)

    def test_a_build_hands_the_block_s_settings_to_the_held_copy(self) -> None:
        from src.config.loader import load_config
        from src.models import factory

        def build(**vlm: Any) -> Any:
            pipeline = PipelineConfig(kind="zero_shot", zero_shot={"backend": "vlm", "vlm": {
                "model_id": "Qwen/Qwen3-VL-4B-Instruct", "local": True, **vlm}}, router={"enabled": False})
            models = load_config().models.model_copy(update={"pipeline": pipeline})
            with mock.patch.object(factory, "_build_named_segmenter", return_value="SEG"):
                factory.build_perception(models)
            assert models.pipeline is not None
            return shared_vlm().grounder_for(models.pipeline.zero_shot.vlm)

        grounder = build(prompt_lookup_tokens=10, text_prompt_lookup_tokens=6, stop_at_answer_end=False,
                         decode_graphs="off")
        self.assertEqual(grounder.prompt_lookup_tokens, 10)
        self.assertEqual(grounder.text_prompt_lookup_tokens, 6)
        self.assertFalse(grounder.stop_at_answer_end)
        self.assertEqual("off", grounder.decode_graphs)
        self.assertFalse(grounder.loaded, "a build sets how the copy answers and loads nothing")
        again = build()
        self.assertIs(again, grounder, "the same weights keep the same copy")
        self.assertEqual((again.prompt_lookup_tokens, again.text_prompt_lookup_tokens), (0, 0),
                         "the next build's block decides, not the last one's")
        self.assertTrue(again.stop_at_answer_end)
        self.assertEqual("auto", again.decode_graphs)


if __name__ == "__main__":
    unittest.main()
