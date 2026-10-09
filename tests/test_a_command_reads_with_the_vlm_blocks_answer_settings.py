"""A command and Laden hand the VLM block's answer settings to the copy, as a cell build does (2026-10-09).

The answer lookups and the stop at the answer's end are the ``vlm`` block's (``models.pipeline.zero_shot.vlm``), not
the weights'. A build hands them to the process's one copy (``src/models/factory.py``); the command reader and Laden
reach the copy through the holder, so they hand them over too: on a cell whose detector is not the VLM, or a routed
cell before its first VLM prompt, the reader would otherwise answer with the defaults, and the owner's
``text_prompt_lookup_tokens: 10`` would do nothing. Nothing here loads weights.
"""

from __future__ import annotations

import unittest
from typing import Any

from src.config.schema.models.models_schema import VlmConfig
from src.models.vlm import Qwen3VLGrounder, VlmHolder, shared_vlm

MODEL = "Qwen/Qwen3-VL-4B-Instruct"


def _vlm(**overrides: Any) -> VlmConfig:
    return VlmConfig(**{"model_id": MODEL, "local": True, **overrides})


class _NoLoad:
    """The grounder's ``_load`` seam: a load here would be a bug, so it says so."""

    def __call__(self) -> Any:
        raise AssertionError("nothing in these tests may load weights")


class ACommandAnswersWithTheBlocksSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        shared_vlm().forget()
        self.addCleanup(shared_vlm().forget)
        self.built: list[Qwen3VLGrounder] = []
        self.holder = VlmHolder.from_parts(build=self._build)
        self.addCleanup(self.holder.forget)

    def _build(self, weights: Any) -> Qwen3VLGrounder:
        grounder = Qwen3VLGrounder(model_id=weights.model_id, local=True)
        grounder._load = _NoLoad()  # type: ignore[method-assign]
        self.built.append(grounder)
        return grounder

    def test_the_reader_takes_the_text_lookup_its_block_names(self) -> None:
        self.holder.asker_for(_vlm(text_prompt_lookup_tokens=10, prompt_lookup_tokens=10), may_load=True)

        (grounder,) = self.built
        self.assertEqual((10, 10), (grounder.text_prompt_lookup_tokens, grounder.prompt_lookup_tokens))
        self.assertFalse(grounder.loaded)

    def test_the_reader_takes_the_decode_graphs_its_block_names(self) -> None:
        self.holder.asker_for(_vlm(decode_graphs="off"), may_load=True)
        self.holder.asker_for(_vlm(decode_graphs="on"), may_load=True)

        (grounder,) = self.built
        self.assertEqual("on", grounder.decode_graphs)

    def test_a_block_that_turns_the_lookup_off_turns_it_off_on_the_held_copy(self) -> None:
        self.holder.asker_for(_vlm(text_prompt_lookup_tokens=10), may_load=True)
        self.holder.asker_for(_vlm(text_prompt_lookup_tokens=0), may_load=True)

        (grounder,) = self.built
        self.assertEqual(0, grounder.text_prompt_lookup_tokens)

    def test_a_block_without_the_settings_keeps_what_the_copy_has(self) -> None:
        self.holder.asker_for(_vlm(text_prompt_lookup_tokens=10), may_load=True)

        class _Bare:
            model_id = MODEL
            model_path = None
            local = True

        self.holder.asker_for(_Bare(), may_load=True)

        (grounder,) = self.built
        self.assertEqual(10, grounder.text_prompt_lookup_tokens)

    def test_laden_hands_the_block_over_before_it_loads(self) -> None:
        loads: list[tuple[int, int, bool]] = []

        def _build(weights: Any) -> Qwen3VLGrounder:
            grounder = Qwen3VLGrounder(model_id=weights.model_id, local=True)

            def _load() -> Any:
                loads.append((grounder.prompt_lookup_tokens, grounder.text_prompt_lookup_tokens,
                              grounder.stop_at_answer_end))
                raise OSError("no weights on this desk")

            grounder._load = _load  # type: ignore[method-assign]
            return grounder

        holder = VlmHolder.from_parts(build=_build)
        self.addCleanup(holder.forget)

        status = holder.load_for(_vlm(prompt_lookup_tokens=10, text_prompt_lookup_tokens=10, stop_at_answer_end=True))

        self.assertEqual([(10, 10, True)], loads)
        self.assertEqual("failed", status.state)


if __name__ == "__main__":
    unittest.main()
