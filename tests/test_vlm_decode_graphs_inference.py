"""Real weights, real GPU: the decode passes as CUDA graphs on the real model, the `inference` tier.

    .venv/Scripts/python.exe -m pytest tests/test_vlm_decode_graphs_inference.py -m inference -q

Auto-skips unless CUDA, a card of compute capability 8.0 or newer (where ``decode_graphs: auto`` takes graphs) and the
weights are all present, as `tests/test_vlm_inference.py` does; not part of the CI gate. The CPU half of the same claims
is `tests/test_vlm_decode_graphs.py`.

WHAT THIS EXISTS TO CATCH. The graphs replay the plain pass's kernels and leave each layer's attention to the model's
own code, so every answer must come out byte for byte as the plain passes write it: grounding and text, with and
without the answer lookup; and a pass must cost the CPU less, which is the point. Measured on the dev box's 4B and RTX
5080 (2026-10-09): the 90 calls on the cell's frames and every command sentence byte-identical, about 19 ms a pass
instead of about 40.
"""

from __future__ import annotations

import time
import unittest
from typing import Any
from unittest import mock

import numpy as np
import pytest

pytestmark = pytest.mark.inference

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
W, H = 640, 480
GREEN = ((60, 300, 140, 380), (260, 60, 340, 140), (500, 80, 580, 160))
#: Measured on the RTX 5080: 8.93 GB after the load, about 9.5 GB at the peak of a grounding answer.
_VRAM_NEEDED_BYTES = 10.5e9


def _card_takes_graphs() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available()) and tuple(torch.cuda.get_device_capability()) >= (8, 0)


def _weights_present() -> bool:
    """True if the snapshot is already in the cache. Never downloads."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=MODEL_ID, local_files_only=True)
    except Exception:  # noqa: BLE001 - any failure means "not usable here"
        return False
    return True


def _scene() -> np.ndarray:
    """A red, a blue and three green squares on near-white, in BGR."""
    image = np.full((H, W, 3), 245, dtype=np.uint8)
    image[100:200, 100:200] = (0, 0, 255)
    image[300:420, 400:520] = (255, 0, 0)
    for x0, y0, x1, y1 in GREEN:
        image[y0:y1, x0:x1] = (0, 200, 0)
    return image


@unittest.skipUnless(_card_takes_graphs(), "no CUDA card of compute capability 8.0 or newer in this interpreter")
@unittest.skipUnless(_weights_present(), f"{MODEL_ID} not in the HF cache (scripts/model_weights/fetch.py)")
class DecodeGraphsInferenceTests(unittest.TestCase):
    """One model load for the class, about 7 s and 9 GB of VRAM; a test that unloads loads it again."""

    grounder: Any
    rgb: np.ndarray

    @classmethod
    def setUpClass(cls) -> None:
        import torch

        from src.models.vlm import Qwen3VLGrounder

        free, _ = torch.cuda.mem_get_info()
        if free < _VRAM_NEEDED_BYTES:
            raise unittest.SkipTest(f"{free / 1e9:.1f} GB of VRAM free, the model needs about "
                                    f"{_VRAM_NEEDED_BYTES / 1e9:.1f} GB")
        cls.grounder = Qwen3VLGrounder(model_id=MODEL_ID, local=True, preload=True)
        cls.rgb = np.ascontiguousarray(_scene()[..., ::-1])

    @classmethod
    def tearDownClass(cls) -> None:
        cls.grounder.unload()

    def tearDown(self) -> None:
        self.grounder.configure(prompt_lookup_tokens=0, text_prompt_lookup_tokens=0, stop_at_answer_end=True,
                                decode_graphs="off")

    def _ground(self, prompt: str, **settings: Any) -> tuple[str, int]:
        """``(answer, tokens)`` for one grounding question, as ``detect_all`` asks it."""
        self.grounder.configure(**settings)
        return self.grounder._generate(self.rgb, prompt)  # noqa: SLF001

    def test_a_grounding_answer_is_byte_identical_with_and_without_graphs(self) -> None:
        for lookup in (0, 10):
            for prompt in ("each separate green square", "the red square", "a yellow triangle"):
                with self.subTest(prompt=prompt, lookup=lookup):
                    plain = self._ground(prompt, prompt_lookup_tokens=lookup, decode_graphs="off")
                    graphed = self._ground(prompt, prompt_lookup_tokens=lookup, decode_graphs="auto")
                    self.assertEqual(plain, graphed)
        self.assertIsNone(self.grounder.graphs_failure)
        self.assertIn((1, 1), self.grounder._graphs._checked, "the graphs ran")  # noqa: SLF001

    def test_a_command_reading_is_byte_identical_with_and_without_graphs(self) -> None:
        from src.models.vlm.command import COMMAND_INSTRUCTION, COMMAND_MAX_NEW_TOKENS

        question = 'POSES: "Ablage links" -> "ablage_links", "Home" -> "home"\nCOMMAND: Alle grauen Würfel von der ' \
                   'schwarzen Matte in die gelbe Kiste'
        answers = []
        for lookup, mode in ((0, "off"), (0, "auto"), (10, "off"), (10, "auto")):
            self.grounder.configure(text_prompt_lookup_tokens=lookup, decode_graphs=mode)
            answers.append(self.grounder.answer_text(COMMAND_INSTRUCTION, question, max_new_tokens=COMMAND_MAX_NEW_TOKENS))
        self.assertEqual(answers[0], answers[1], "graphs, no lookup")
        self.assertEqual(answers[2], answers[3], "graphs, with the lookup")
        self.assertTrue(answers[0].endswith("}"), f"the reading ends where its object closes: {answers[0]!r}")

    def test_a_pass_costs_less_through_the_graphs(self) -> None:
        import torch

        times = {}
        for mode in ("off", "auto", "off", "auto"):
            self.grounder.configure(decode_graphs=mode)
            torch.cuda.synchronize()
            started = time.perf_counter()
            _, tokens = self.grounder._generate(self.rgb, "each separate green square")  # noqa: SLF001
            torch.cuda.synchronize()
            times[mode] = (time.perf_counter() - started) / tokens
        self.assertLess(times["auto"], 0.8 * times["off"], times)

    def test_the_first_pass_of_each_shape_is_checked_bit_for_bit_after_a_load(self) -> None:
        self.grounder.unload()
        self.grounder.configure(prompt_lookup_tokens=10, decode_graphs="auto")
        with self.assertLogs("src.models.vlm.qwen", level="INFO") as said:
            self.grounder.ensure_loaded()
            self.grounder._generate(self.rgb, "each separate green square")  # noqa: SLF001
        self.assertTrue(any("decodes with CUDA graphs (decode_graphs=auto" in line for line in said.output),
                        said.output)
        checked = [line for line in said.output if "bit for bit" in line]
        self.assertGreaterEqual(len(checked), 2, "one token a pass, and the lookup's chunks")
        self.assertIsNone(self.grounder.graphs_failure)

    def test_a_pass_that_fails_mid_answer_answers_plainly_and_a_reload_tries_graphs_again(self) -> None:
        from src.models.vlm.qwen import _DecodeGraphs

        plain = self._ground("each separate green square", decode_graphs="off")
        self._ground("the red square", decode_graphs="auto")  # captured and checked
        attend = _DecodeGraphs._attend  # noqa: SLF001
        passes = [0]

        def fails_at_the_tenth_pass(self_: Any, graphs: Any, index: int, cache: Any, inputs: Any) -> None:
            if index == 0:
                passes[0] += 1
            if passes[0] == 10 and index == 17:
                raise RuntimeError("a replay went wrong")
            attend(self_, graphs, index, cache, inputs)

        with mock.patch.object(_DecodeGraphs, "_attend", fails_at_the_tenth_pass):
            self.assertEqual(plain, self._ground("each separate green square", decode_graphs="auto"))
        self.assertIn("a replay went wrong", self.grounder.graphs_failure or "")
        self.grounder.unload()
        self.grounder.ensure_loaded()
        self.assertEqual(plain, self._ground("each separate green square", decode_graphs="auto"))
        self.assertIsNone(self.grounder.graphs_failure, "a reload tries the graphs again")


if __name__ == "__main__":
    unittest.main()
