"""Real weights, real GPU: the answer's end and the answer lookup on the real model, the `inference` tier.

    .venv/Scripts/python.exe -m pytest tests/test_vlm_answer_speed_inference.py -m inference -q

Auto-skips unless CUDA and the weights are both present, as `tests/test_vlm_inference.py` does; not part of the CI
gate. The CPU half of the same claims is `tests/test_vlm_answer_speed.py`.

WHAT THIS EXISTS TO CATCH. Each pass of the model is the cost of a grounding call (about 32 ms on the dev box's 4B,
about 140 ms on the cell's 8B), and both switches cut passes. Only the real model shows whether they cost a box:

* ``stop_at_answer_end`` must leave every box and label exactly as the model's own end leaves them, one pass sooner.
* ``prompt_lookup_tokens`` must still ground what plain decoding grounds (the same count, IoU 0.9 or more, the
  absent object still absent) in clearly fewer passes. It is not byte-identical, which is why it is off by default.
* ``text_prompt_lookup_tokens`` must read a command as plain decoding reads it, in fewer passes, and touch no
  grounding answer.

Measured on the cell's own wrist frames on 2026-10-08 (two frames, four prompts): the lookup wrote three cubes in 47
passes instead of 108, and the answer's end saved one pass a call with byte-identical answers.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.inference

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
W, H = 640, 480
RED_GT = (100.0, 100.0, 200.0, 200.0)
BLUE_GT = (400.0, 300.0, 520.0, 420.0)
#: Three green squares: a list of several boxes is where the lookup saves most and where a box could go missing.
GREEN_GT = ((60.0, 300.0, 140.0, 380.0), (260.0, 60.0, 340.0, 140.0), (500.0, 80.0, 580.0, 160.0))
#: The bar a grounding result must clear against the scene's truth, as in `tests/test_vlm_inference.py`.
MIN_IOU = 0.5
#: How close the lookup's boxes stay to plain decoding's: measured 0.98 or more on the cell's frames.
SAME_BOX_IOU = 0.9


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _weights_present() -> bool:
    """True if the snapshot is already in the cache. Never downloads."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=MODEL_ID, local_files_only=True)
    except Exception:  # noqa: BLE001 - any failure means "not usable here"
        return False
    return True


def _iou(a: Any, b: Any) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _scene() -> np.ndarray:
    """A red, a blue and three green squares on near-white, in BGR. Ground truth by construction."""
    image = np.full((H, W, 3), 245, dtype=np.uint8)
    image[100:200, 100:200] = (0, 0, 255)
    image[300:420, 400:520] = (255, 0, 0)
    for x0, y0, x1, y1 in GREEN_GT:
        image[int(y0):int(y1), int(x0):int(x1)] = (0, 200, 0)
    return image


@unittest.skipUnless(_cuda_available(), "no CUDA in this interpreter")
@unittest.skipUnless(_weights_present(), f"{MODEL_ID} not in the HF cache (scripts/model_weights/fetch.py)")
class AnswerSpeedInferenceTests(unittest.TestCase):
    """One model load for the whole class: about 7 s and 9 GB of VRAM."""

    grounder: Any
    passes: list[int]

    @classmethod
    def setUpClass(cls) -> None:
        from src.models.vlm import Qwen3VLGrounder

        cls.grounder = Qwen3VLGrounder(model_id=MODEL_ID, local=True, preload=True)
        cls.image = _scene()
        cls.passes = [0]

        def count(*_: Any) -> None:
            cls.passes[0] += 1

        cls.hook = cls.grounder._model.register_forward_hook(count)  # noqa: SLF001

    @classmethod
    def tearDownClass(cls) -> None:
        cls.hook.remove()
        cls.grounder.unload()

    def tearDown(self) -> None:
        self.grounder.configure(prompt_lookup_tokens=0, text_prompt_lookup_tokens=0, stop_at_answer_end=True)

    def _ground(self, prompt: str, **settings: Any) -> tuple[list[Any], int]:
        self.grounder.configure(**settings)
        self.passes[0] = 0
        detections = self.grounder.detect_all(self.image, prompt)
        return detections, self.passes[0]

    def test_the_answer_s_end_changes_no_box_and_saves_a_pass(self) -> None:
        for prompt in ("the red square", "each separate green square", "a yellow triangle"):
            with self.subTest(prompt=prompt):
                whole, whole_passes = self._ground(prompt, prompt_lookup_tokens=0, stop_at_answer_end=False)
                ended, ended_passes = self._ground(prompt, prompt_lookup_tokens=0, stop_at_answer_end=True)
                self.assertEqual([(d.box, d.label) for d in ended], [(d.box, d.label) for d in whole])
                self.assertEqual(ended_passes, whole_passes - 1, "the end token's pass is the one saved")

    def test_the_lookup_grounds_what_plain_decoding_grounds_in_fewer_passes(self) -> None:
        plain, plain_passes = self._ground("each separate green square", prompt_lookup_tokens=0)
        looked, looked_passes = self._ground("each separate green square", prompt_lookup_tokens=10)
        self.assertIsNone(self.grounder.lookup_failure)
        self.assertEqual(len(plain), len(GREEN_GT), f"plain decoding found {[d.box for d in plain]}")
        self.assertEqual(len(looked), len(plain), f"the lookup found {[d.box for d in looked]}")
        for det in plain:
            best = max(_iou(det.box, other.box) for other in looked)
            self.assertGreaterEqual(best, SAME_BOX_IOU, f"{det.box} has no twin in {[d.box for d in looked]}")
        self.assertLessEqual(looked_passes, 0.8 * plain_passes, f"{looked_passes} passes against {plain_passes}")

    def test_the_lookup_still_meets_the_scene_s_truth(self) -> None:
        for prompt, truth in (("the red square", RED_GT), ("the blue square", BLUE_GT)):
            with self.subTest(prompt=prompt):
                detections, _ = self._ground(prompt, prompt_lookup_tokens=10)
                self.assertTrue(detections, "no box for an object that is plainly there")
                best = max(_iou(det.box, truth) for det in detections)
                self.assertGreaterEqual(best, MIN_IOU, f"best IoU {best:.3f}; boxes={[d.box for d in detections]}")

    def test_an_absent_object_stays_absent_with_the_lookup(self) -> None:
        detections, _ = self._ground("a yellow triangle", prompt_lookup_tokens=10)
        self.assertEqual(detections, [], f"invented {len(detections)} box(es) for an absent object")

    def test_a_text_answer_reads_the_same_with_its_lookup_in_fewer_passes(self) -> None:
        """The command reader's shape: a JSON object whose keys the instruction names (measured on its real
        instruction: ten commands in 128 passes instead of 617, every answer byte-identical)."""
        import json

        system = ('Read the operator\'s command. Answer with ONE JSON object and nothing else: '
                  '{"intent": "pick" | "stop" | "none", "object": <English noun phrase or null>, '
                  '"place": <English noun phrase or null>, "count": <number or null>}')
        answers, passes = [], []
        for tokens in (0, 10):
            self.grounder.configure(text_prompt_lookup_tokens=tokens)
            self.passes[0] = 0
            answers.append(self.grounder.answer_text(system, "COMMAND: Leg den roten Würfel in die blaue Kiste",
                                                     max_new_tokens=160))
            passes.append(self.passes[0])
        self.assertEqual(json.loads(answers[1]), json.loads(answers[0]), answers)
        self.assertLess(passes[1], passes[0])

    def test_the_text_lookup_leaves_grounding_alone(self) -> None:
        plain, plain_passes = self._ground("each separate green square")
        self.grounder.configure(text_prompt_lookup_tokens=10)
        again, again_passes = self._ground("each separate green square")
        self.assertEqual([d.box for d in again], [d.box for d in plain])
        self.assertEqual(again_passes, plain_passes)


if __name__ == "__main__":
    unittest.main()
