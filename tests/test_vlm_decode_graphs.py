"""The decode passes as CUDA graphs (``vlm.decode_graphs``): the plain pass's arithmetic, or the plain pass itself.

No weights and no card. A pass of the model over one new token is about 2400 kernel launches, and on the cell each
costs the CPU more than the card needs to run it (about 140 ms a token for the 8B, where reading its weights takes about
20). ``_DecodeGraphs`` hands the stretches between the layers' attentions to the card as CUDA graphs and runs each
attention as the plain pass runs it, through the model's own cache and attention function. What a box without a card
can pin:

* **The split is the model's pass.** A tiny Qwen3-VL built from its config, run through the stretches the graphs
  capture and the attentions they leave out, eagerly, writes the logits and the cache the model's own forward writes,
  bit for bit: for one new token and for a chunk the answer lookup checks.
* **Which calls the graphs take.** A decode pass over the model's own cache, and nothing else: never the prefill, an
  image, a batch, another kind of cache, or outputs the graphs do not keep.
* **The decision.** Read off the copy: CUDA graphs in torch, the model on a CUDA card (``auto``: compute capability 8.0
  or newer), the decoder laid out as Qwen3-VL's; logged once per loaded copy and mode.
* **The fallback.** A capture that fails (here: no card to capture on) or a pass that raises halfway turns graphs off for
  the copy, the cache is cut back and the pass runs plainly: the answer is the plain answer.
* **The switch.** ``decode_graphs`` in the ``vlm`` block, ``auto`` by default, refused when it names no mode.

The real model's side of the same claims: ``tests/test_vlm_decode_graphs_inference.py``.
"""

from __future__ import annotations

import contextlib
import functools
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
from pydantic import ValidationError

from src.config.schema.models.models_schema import VlmConfig
from src.models.vlm import Qwen3VLGrounder
from src.models.vlm import qwen as qwen_module
from src.models.vlm.qwen import DECODE_GRAPHS, GRAPHS_MAX_TOKENS, _DecodeGraphs, _graphs_unsupported, _GraphSet

PROMPT = [1, 5, 9, 17, 33, 65, 129, 7, 3]


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - the CI image installs the CPU wheel
        raise unittest.SkipTest(f"torch is not installed: {exc}") from exc
    return torch


@functools.lru_cache(maxsize=1)
def _tiny() -> Any:
    """A Qwen3-VL with three small decoder layers and random weights, on the CPU: the real modules, the real layout."""
    torch = _torch()
    try:
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    except ImportError as exc:  # pragma: no cover
        raise unittest.SkipTest(f"this transformers has no Qwen3-VL: {exc}") from exc
    config = Qwen3VLConfig(
        text_config=dict(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                         rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [4, 2, 2],
                                          "mrope_interleaved": True}),
        vision_config=dict(depth=1, hidden_size=32, num_heads=2, intermediate_size=64, out_hidden_size=64,
                           deepstack_visual_indexes=[0]),
        image_token_id=250, video_token_id=251, vision_start_token_id=252, vision_end_token_id=253)
    torch.manual_seed(0)
    return Qwen3VLForConditionalGeneration(config).eval()


def _prefilled(model: Any, prompt: list[int] = PROMPT) -> tuple[Any, Any]:
    """The prompt read into a fresh cache, as generate's prefill reads it: ``(cache, the first answer token)``."""
    torch = _torch()
    from transformers import DynamicCache

    ids = torch.tensor([prompt])
    cache = DynamicCache(config=model.config)
    positions = torch.arange(len(prompt)).view(1, 1, -1).expand(4, 1, -1)
    with torch.inference_mode():
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=positions,
                       past_key_values=cache, use_cache=True, logits_to_keep=1)
    return cache, int(output.logits[0, -1].argmax())


def _decode_kwargs(cache: Any, tokens: list[int], *, keep: int = 1) -> dict[str, Any]:
    """What generate hands the model for a decode pass over ``tokens`` after what ``cache`` holds."""
    torch = _torch()
    past = cache.get_seq_length()
    ids = torch.tensor([tokens])
    positions = (torch.arange(len(tokens)) + past).view(1, 1, -1).expand(4, 1, -1).contiguous()
    return {"input_ids": ids, "past_key_values": cache, "position_ids": positions,
            "attention_mask": torch.ones((1, past + len(tokens)), dtype=torch.long), "pixel_values": None,
            "pixel_values_videos": None, "image_grid_thw": None, "video_grid_thw": None, "use_cache": True,
            "mm_token_type_ids": torch.zeros((1, past + len(tokens)), dtype=torch.long), "logits_to_keep": keep,
            "return_dict": True}


def _eager_split_pass(graphs: _DecodeGraphs, kwargs: dict[str, Any], keep: int) -> Any:
    """A pass through the stretches the graphs capture and the attentions they leave out, run eagerly in their order:
    the arithmetic a replayed pass runs. The logits."""
    torch = _torch()
    model = graphs.model
    text = model.model.language_model
    steps = int(kwargs["input_ids"].shape[1])
    pieces = _GraphSet(
        tokens=kwargs["input_ids"].clone(), positions=kwargs["position_ids"].clone(),
        attended=[torch.zeros((1, steps, text.config.num_attention_heads, layer.self_attn.head_dim),
                              dtype=model.lm_head.weight.dtype) for layer in text.layers])
    inputs: list[Any] = []

    def attend(index: int) -> None:
        if not inputs:  # made before the first attention updates the cache, as a replayed pass makes them
            inputs.append(graphs._attention_inputs(pieces, kwargs))  # noqa: SLF001
        graphs._attend(pieces, index, kwargs["past_key_values"], inputs[0])  # noqa: SLF001

    graphs._run_pieces(pieces, keep, lambda index: contextlib.nullcontext(), attend)  # noqa: SLF001
    return pieces.logits


class TheSplitPassIsTheModelsPassTests(unittest.TestCase):
    """What the graphs replay is the plain pass, cut where each layer's attention reads the cache."""

    def setUp(self) -> None:
        self.torch = _torch()
        self.model = _tiny()

    def _compare(self, tokens: list[int], keep: int) -> None:
        cache, _ = _prefilled(self.model)
        past = cache.get_seq_length()
        kwargs = _decode_kwargs(cache, tokens, keep=keep)
        with self.torch.inference_mode():
            plain = self.model(**kwargs).logits
            written = [(layer.keys[..., past:, :].clone(), layer.values[..., past:, :].clone()) for layer in cache.layers]
            cache.crop(past)
            split = _eager_split_pass(_DecodeGraphs(self.model, self.torch, model_id="tiny"), kwargs, keep)
        self.assertTrue(self.torch.equal(plain, split), "the logits differ")
        for layer, (keys, values) in zip(cache.layers, written, strict=True):
            self.assertTrue(self.torch.equal(layer.keys[..., past:, :], keys), "the cache's keys differ")
            self.assertTrue(self.torch.equal(layer.values[..., past:, :], values), "the cache's values differ")

    def test_one_new_token_writes_the_plain_pass_s_bits(self) -> None:
        self._compare([42], keep=1)

    def test_a_chunk_the_lookup_checks_writes_the_plain_pass_s_bits(self) -> None:
        """Several tokens a pass, every one's logits kept, and a causal mask inside the chunk."""
        self._compare([42, 7, 99, 3], keep=4)


class TheCallsTheGraphsTakeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.torch = _torch()
        self.model = _tiny()
        self.graphs = _DecodeGraphs(self.model, self.torch, model_id="tiny")
        self.cache, self.token = _prefilled(self.model)

    def _shape(self, **changes: Any) -> Any:
        kwargs = _decode_kwargs(self.cache, [self.token])
        kwargs.update(changes)
        return self.graphs._shape_of(kwargs)  # noqa: SLF001

    def test_a_decode_pass_over_the_model_s_own_cache(self) -> None:
        self.assertEqual((1, 1), self._shape())
        kwargs = _decode_kwargs(self.cache, [1, 2, 3], keep=3)
        self.assertEqual((3, 3), self.graphs._shape_of(kwargs))  # noqa: SLF001

    def test_never_the_prefill_or_an_image(self) -> None:
        from transformers import DynamicCache

        self.assertIsNone(self._shape(past_key_values=DynamicCache(config=self.model.config)), "an empty cache")
        self.assertIsNone(self._shape(pixel_values=self.torch.zeros(4, 8)))
        self.assertIsNone(self._shape(inputs_embeds=self.torch.zeros(1, 1, 64)))

    def test_never_a_batch_another_cache_or_outputs_the_graphs_do_not_keep(self) -> None:
        from transformers import DynamicCache

        class _Offloading(DynamicCache):
            pass

        self.assertIsNone(self._shape(input_ids=self.torch.tensor([[self.token], [self.token]])))
        self.assertIsNone(self._shape(past_key_values=_Offloading.__new__(_Offloading)))
        self.assertIsNone(self._shape(output_attentions=True))
        self.assertIsNone(self._shape(output_hidden_states=True))
        self.assertIsNone(self._shape(use_cache=False))
        self.assertIsNone(self._shape(labels=self.torch.tensor([[1]])))

    def test_never_a_pass_of_another_shape(self) -> None:
        self.assertIsNone(self._shape(position_ids=self.torch.zeros((3, 1, 1), dtype=self.torch.long)))
        self.assertIsNone(self._shape(attention_mask=self.torch.ones((1, 3), dtype=self.torch.long)))
        self.assertIsNone(self._shape(logits_to_keep=True))
        self.assertIsNone(self._shape(logits_to_keep=2), "more logits kept than the pass has tokens")
        tokens = list(range(GRAPHS_MAX_TOKENS + 1))
        self.assertIsNone(self.graphs._shape_of(_decode_kwargs(self.cache, tokens)))  # noqa: SLF001


def _cuda(capability: tuple[int, int] = (8, 9), *, graphs: bool = True, available: bool = True) -> Any:
    """A torch double that says what a card can do."""
    return SimpleNamespace(cuda=SimpleNamespace(
        CUDAGraph=object if graphs else None, is_available=lambda: available,
        get_device_capability=lambda device: capability, get_device_name=lambda device: "a card"))


def _on_a_card(model: Any) -> Any:
    """``model``'s parts, reading as if they sat on a CUDA card."""
    return SimpleNamespace(device=SimpleNamespace(type="cuda"), model=model.model, lm_head=model.lm_head)


class TheDecisionTests(unittest.TestCase):
    """Read off the copy, never off a list of cards."""

    def setUp(self) -> None:
        self.model = _on_a_card(_tiny())

    def test_a_qwen3_vl_decoder_on_a_newer_card_takes_graphs(self) -> None:
        self.assertEqual("", _graphs_unsupported(self.model, _cuda((8, 9)), auto=True))
        self.assertEqual("", _graphs_unsupported(self.model, _cuda((12, 0)), auto=True))

    def test_a_card_below_compute_capability_8_takes_them_only_when_asked(self) -> None:
        reason = _graphs_unsupported(self.model, _cuda((7, 5)), auto=True)
        self.assertIn("7.5", reason)
        self.assertIn("decode_graphs: on", reason)
        self.assertEqual("", _graphs_unsupported(self.model, _cuda((7, 5)), auto=False))

    def test_no_cuda_graphs_no_card_or_another_decoder_decode_plainly(self) -> None:
        self.assertIn("no CUDA graphs", _graphs_unsupported(self.model, _cuda(graphs=False), auto=False))
        self.assertIn("no CUDA graphs", _graphs_unsupported(self.model, _cuda(available=False), auto=False))
        self.assertIn("no CUDA graphs", _graphs_unsupported(self.model, SimpleNamespace(), auto=False))
        self.assertIn("not on a CUDA device", _graphs_unsupported(_tiny(), _cuda(), auto=False))
        other = SimpleNamespace(device=SimpleNamespace(type="cuda"), model=SimpleNamespace(), lm_head=object())
        self.assertIn("not laid out", _graphs_unsupported(other, _cuda(), auto=False))


# --- the grounder, through doubles of the processor and the model -------------------------------------------------


class _Batch(dict):  # type: ignore[type-arg]
    def to(self, device: Any) -> "_Batch":
        return self


class _Processor:
    tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: None, eos_token_id=None)

    def apply_chat_template(self, messages: Any, **_: Any) -> _Batch:
        return _Batch(input_ids=np.arange(7).reshape(1, -1), attention_mask=np.ones((1, 7)))

    def batch_decode(self, ids: Any, skip_special_tokens: bool = False) -> list[str]:
        return ['[{"bbox_2d": [100, 100, 200, 200], "label": "red cube"}]']


class _Model:
    """Records whether its ``forward`` was swapped while it generated."""

    device = "cpu"

    def __init__(self) -> None:
        self.swapped: list[bool] = []

    def generate(self, **kwargs: Any) -> np.ndarray:
        self.swapped.append("forward" in self.__dict__)
        return np.concatenate([kwargs["input_ids"], np.full((1, 3), 9)], axis=1)


class _NoGraphsTorch:
    cuda = SimpleNamespace(empty_cache=lambda: None)

    @staticmethod
    def inference_mode() -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()


def _grounder(model: Any, **settings: Any) -> Qwen3VLGrounder:
    grounder = Qwen3VLGrounder(model_id="Qwen/Qwen3-VL-4B-Instruct", local=True, **settings)
    grounder._load = lambda: (_Processor(), model, _NoGraphsTorch())  # type: ignore[method-assign]
    return grounder


def _look(grounder: Qwen3VLGrounder) -> list[Any]:
    return grounder.detect_all(np.zeros((72, 128, 3), np.uint8), "red cube")


class TheGroundersSwitchTests(unittest.TestCase):
    def test_off_leaves_every_pass_as_it_was(self) -> None:
        model = _Model()
        grounder = _grounder(model, decode_graphs="off")
        self.assertEqual("off", grounder.decode_graphs)
        _look(grounder)
        grounder.answer_text("S", "U")
        self.assertEqual([False, False], model.swapped)
        self.assertIsNone(grounder.graphs_failure)

    def test_a_copy_that_cannot_take_graphs_decodes_plainly_and_says_so_once(self) -> None:
        model = _Model()
        grounder = _grounder(model, decode_graphs="auto")
        with self.assertLogs("src.models.vlm.qwen", level="INFO") as said:
            _look(grounder)
            _look(grounder)
        decided = [line for line in said.output if "decodes plainly" in line]
        self.assertEqual(1, len(decided), said.output)
        self.assertIn("decode_graphs=auto", decided[0])
        self.assertIn("no CUDA graphs", decided[0])
        self.assertEqual([False, False], model.swapped)
        with self.assertLogs("src.models.vlm.qwen", level="INFO") as said:
            grounder.configure(decode_graphs="on")
            _look(grounder)
        self.assertEqual(1, len([line for line in said.output if "decode_graphs=on)" in line]), said.output)

    def test_a_card_that_cannot_be_read_decodes_plainly(self) -> None:
        """A speed-up never costs an answer, not even its decision."""

        def unreadable(device: Any) -> Any:
            raise RuntimeError("no such card")

        model = _Model()
        model.device = SimpleNamespace(type="cuda")  # type: ignore[assignment]
        torch = _NoGraphsTorch()
        torch.cuda = SimpleNamespace(CUDAGraph=object, is_available=lambda: True, empty_cache=lambda: None,  # type: ignore[misc]
                                     get_device_capability=unreadable, get_device_name=unreadable)
        grounder = Qwen3VLGrounder(model_id="Qwen/Qwen3-VL-4B-Instruct", local=True, decode_graphs="auto")
        grounder._load = lambda: (_Processor(), model, torch)  # type: ignore[method-assign]
        with self.assertLogs("src.models.vlm.qwen", level="INFO") as said:
            self.assertEqual(1, len(_look(grounder)))
        self.assertTrue(any("the card could not be read: RuntimeError: no such card" in line for line in said.output),
                        said.output)
        self.assertEqual([False], model.swapped)

    def test_configure_takes_the_three_modes_and_nothing_else(self) -> None:
        grounder = Qwen3VLGrounder(model_id="Qwen/Qwen3-VL-4B-Instruct")
        self.assertEqual(("auto", "on", "off"), DECODE_GRAPHS)
        for mode in DECODE_GRAPHS:
            grounder.configure(decode_graphs=mode)
            self.assertEqual(mode, grounder.decode_graphs)
        grounder.configure(prompt_lookup_tokens=3)
        self.assertEqual("off", grounder.decode_graphs, "a setting not given is kept")
        with self.assertRaises(ValueError):
            grounder.configure(decode_graphs="fast")
        self.assertEqual("off", grounder.decode_graphs, "a refused setting changes nothing")

    def test_the_schema_s_key(self) -> None:
        """``auto`` by default, as the grounder's own: byte-identical answers on the 90 calls and the 67 sentences."""
        self.assertEqual("auto", VlmConfig().decode_graphs)
        self.assertEqual("auto", Qwen3VLGrounder(model_id="Qwen/Qwen3-VL-4B-Instruct").decode_graphs)
        self.assertEqual("off", VlmConfig(decode_graphs="off").decode_graphs)
        with self.assertRaises(ValidationError):
            VlmConfig(decode_graphs="fast")  # type: ignore[arg-type]

    def test_a_bare_on_or_off_in_yaml_is_the_word(self) -> None:
        """YAML reads a bare ``on`` as true and ``off`` as false."""
        import yaml

        for text, mode in (("decode_graphs: on", "on"), ("decode_graphs: off", "off"), ('decode_graphs: "off"', "off")):
            with self.subTest(text=text):
                self.assertEqual(mode, VlmConfig(**yaml.safe_load(text)).decode_graphs)

    def test_an_unload_takes_the_graphs_and_the_decision_with_the_weights(self) -> None:
        grounder = _grounder(_Model(), decode_graphs="auto")
        _look(grounder)
        self.assertIsNotNone(grounder._graphs_decided)  # noqa: SLF001
        grounder.unload()
        self.assertIsNone(grounder._graphs_decided)  # noqa: SLF001
        self.assertIsNone(grounder._graphs)  # noqa: SLF001


class TheFallbackTests(unittest.TestCase):
    """A speed-up never costs an answer: whatever the graphs cannot do, the plain pass does, from the same cache."""

    def setUp(self) -> None:
        self.torch = _torch()
        self.model = _tiny()

    def _answer(self, graphs: _DecodeGraphs | None = None) -> list[int]:
        ids = self.torch.tensor([PROMPT])
        if graphs is not None:
            self.model.forward = graphs.forward_for(self.model.forward)
        try:
            with self.torch.inference_mode():
                out = self.model.generate(input_ids=ids, attention_mask=self.torch.ones_like(ids), max_new_tokens=10,
                                          do_sample=False, eos_token_id=None, pad_token_id=0)
        finally:
            if graphs is not None:
                del self.model.forward
        return [int(token) for token in out[0, len(PROMPT):]]

    def test_a_capture_that_fails_turns_graphs_off_and_the_answer_is_the_plain_one(self) -> None:
        plain = self._answer()
        graphs = _DecodeGraphs(self.model, self.torch, model_id="tiny")
        with self.assertLogs("src.models.vlm.qwen", level="WARNING") as said:
            self.assertEqual(plain, self._answer(graphs))
        self.assertIsNotNone(graphs.failure, "no card to capture on")
        self.assertEqual(1, len(said.output), "turned off once; every later pass goes straight to the plain one")
        self.assertIn("decoding plainly until the model is loaded again", said.output[0])

    def test_a_pass_that_raises_halfway_is_cut_back_and_run_plainly(self) -> None:
        """Layer 0 has written its token into the cache when layer 1's attention raises: the cache is cut back to
        before the pass, which then runs plainly, and every later pass too."""
        plain = self._answer()
        graphs = _DecodeGraphs(self.model, self.torch, model_id="tiny")
        graphs._checked.add((1, 1))  # noqa: SLF001 - checked already: the pass goes straight through the graphs

        def capture(shape: tuple[int, int]) -> _GraphSet:
            steps, keep = shape
            text = self.model.model.language_model
            pieces = _GraphSet(
                tokens=self.torch.zeros((1, steps), dtype=self.torch.long),
                positions=self.torch.zeros((4, 1, steps), dtype=self.torch.long),
                attended=[self.torch.zeros((1, steps, 4, 16)) for _ in text.layers])
            graphs._run_pieces(pieces, keep, lambda index: contextlib.nullcontext())  # noqa: SLF001
            pieces.pieces = [SimpleNamespace(replay=lambda: None) for _ in range(len(text.layers) + 1)]
            graphs._sets[shape] = pieces  # noqa: SLF001
            return pieces

        attend = graphs._attend  # noqa: SLF001

        def attend_until_layer_1(pieces: _GraphSet, index: int, cache: Any, inputs: Any) -> None:
            if index == 1:
                raise RuntimeError("a replay went wrong")
            attend(pieces, index, cache, inputs)

        with mock.patch.object(graphs, "_capture", capture), \
                mock.patch.object(graphs, "_attend", attend_until_layer_1), \
                self.assertLogs("src.models.vlm.qwen", level="WARNING"):
            self.assertEqual(plain, self._answer(graphs))
        self.assertIn("a replay went wrong", graphs.failure or "")


class TheModuleStaysLightTests(unittest.TestCase):
    def test_nothing_heavy_is_imported_for_the_graphs_at_module_import(self) -> None:
        """The graphs import transformers' parts only when a loaded copy makes them: this module imports on CI and on a
        box with no GPU."""
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(qwen_module))
        top = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        names = {alias.name.split(".")[0] for node in top for alias in node.names}
        names |= {node.module.split(".")[0] for node in top if isinstance(node, ast.ImportFrom) and node.module}
        self.assertFalse({"torch", "transformers"} & names, names)


if __name__ == "__main__":
    unittest.main()
