"""A command and Laden answer at the owner's cell's settings, as a cell's look does, whatever built the copy.

The answer lookups and the stop at the answer's end were keys of the ``vlm`` block until 2026-10-17, handed to the
process's one copy by every build, every command and Laden, so a reader on a cell whose detector is not the VLM would
not answer with the defaults while the cell's block said 10. They are fixed now at what the cell ran (its grounder's log
of 2026-10-12: lookups 10 and 10, the stop on, the decode graphs ``auto``), and every copy is built with them, so there
is nothing left to hand over: a copy a command or Laden builds answers as one a build made. Nothing here loads weights.
"""

from __future__ import annotations

import unittest
from typing import Any

from src.config.schema.models.models_schema import VlmConfig
from src.models.vlm import Qwen3VLGrounder, VlmHolder, shared_vlm
from src.models.vlm.qwen import (
    DECODE_GRAPHS_MODE,
    PROMPT_LOOKUP_TOKENS,
    STOP_AT_ANSWER_END,
    TEXT_PROMPT_LOOKUP_TOKENS,
)

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
#: What the owner's cell's grounder answered with (its log of 2026-10-12).
THE_CELLS = (10, 10, True, "auto")


def _settings(grounder: Qwen3VLGrounder) -> tuple[int, int, bool, str]:
    return (grounder.prompt_lookup_tokens, grounder.text_prompt_lookup_tokens, grounder.stop_at_answer_end,
            grounder.decode_graphs)


class _NoLoad:
    """The grounder's ``_load`` seam: a load here would be a bug, so it says so."""

    def __call__(self) -> Any:
        raise AssertionError("nothing in these tests may load weights")


class TheCellsSettingsTests(unittest.TestCase):
    def test_the_fixed_settings_are_the_cells(self) -> None:
        self.assertEqual(THE_CELLS, (PROMPT_LOOKUP_TOKENS, TEXT_PROMPT_LOOKUP_TOKENS, STOP_AT_ANSWER_END,
                                     DECODE_GRAPHS_MODE))
        self.assertEqual(THE_CELLS, _settings(Qwen3VLGrounder(model_id=MODEL)))


class ACommandAndLadenAnswerAtTheCellsSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        shared_vlm().forget()
        self.addCleanup(shared_vlm().forget)
        self.built: list[Qwen3VLGrounder] = []
        self.holder = VlmHolder.from_parts(build=self._build)
        self.addCleanup(self.holder.forget)

    def _build(self, weights: Any) -> Qwen3VLGrounder:
        grounder = Qwen3VLGrounder(model_id=weights.model_id)
        grounder._load = _NoLoad()  # type: ignore[method-assign]
        self.built.append(grounder)
        return grounder

    def test_the_readers_copy_answers_at_the_cells_settings(self) -> None:
        self.holder.asker_for(VlmConfig(model_id=MODEL), may_load=True)

        (grounder,) = self.built
        self.assertEqual(THE_CELLS, _settings(grounder))
        self.assertFalse(grounder.loaded)

    def test_laden_loads_a_copy_at_the_cells_settings(self) -> None:
        loads: list[tuple[int, int, bool, str]] = []

        def _build(weights: Any) -> Qwen3VLGrounder:
            grounder = Qwen3VLGrounder(model_id=weights.model_id)

            def _load() -> Any:
                loads.append(_settings(grounder))
                raise OSError("no weights on this desk")

            grounder._load = _load  # type: ignore[method-assign]
            return grounder

        holder = VlmHolder.from_parts(build=_build)
        self.addCleanup(holder.forget)

        status = holder.load_for(VlmConfig(model_id=MODEL))

        self.assertEqual([THE_CELLS], loads)
        self.assertEqual("failed", status.state)

    def test_a_block_that_still_names_a_setting_is_refused(self) -> None:
        """The key left the block; a tree that still writes one is refused at load with its sentence."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            VlmConfig(model_id=MODEL, text_prompt_lookup_tokens=0)  # type: ignore[call-arg]


if __name__ == "__main__":
    unittest.main()
