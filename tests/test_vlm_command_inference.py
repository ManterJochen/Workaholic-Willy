"""Real weights, real GPU: the command reader on the owner's sentences (the `inference` tier).

    python -m pytest tests/test_vlm_command_inference.py -m inference -q

Skips unless CUDA and the weights are both present, so it costs a normal run nothing, and it is NOT part of the
CI gate: a 9 GB model load and seconds of GPU time per sentence.

WHAT THIS EXISTS TO CATCH. `tests/test_vlm_command.py` pins what the reader does with a given answer. Only this
file can tell whether the model still gives that answer to the instruction as written: that it answers JSON at
all through the text-only path (no image, the prefill continued), that it translates the German object into
English, and that it gives a taught pose back by its NAME when it hears the label. One model load for one test;
each sentence is a subtest, so a miss names the sentence.

The checkpoint is the dev box's 4B unless the environment names another: on the cell its 8B, by id and by the path
its weights lie at (`WILLY_VLM_MODEL_ID`, `WILLY_VLM_MODEL_PATH`). `TheOwnersDayOnTheRealModelTests` reads the
sentences of 2026-10-08 the way the instruction asks for them since: compact, only what was said, the one part a
sentence singles out (`which`) and where the parts lie (`from`); it checks that the morning's sentence answers in 45
tokens or fewer (70 before), and that the grounding of a which finds the one part it names.
"""

from __future__ import annotations

import os
import unittest
from typing import Any

import pytest

from src.config.schema.models.models_schema import VlmConfig
from src.models.vlm import VlmHolder, shared_vlm, understand
from src.models.vlm.command import COMMAND_MAX_NEW_TOKENS

pytestmark = pytest.mark.inference

MODEL_ID = os.environ.get("WILLY_VLM_MODEL_ID") or "Qwen/Qwen3-VL-4B-Instruct"
#: Where the weights lie, on a box that keeps them outside the Hugging Face cache (the cell's 8B); none reads the cache.
MODEL_PATH = os.environ.get("WILLY_VLM_MODEL_PATH") or None
POSES = {"Ablage links": "ablage_links", "Wartepose": "wartepose"}
#: Measured on the RTX 5080 (2026-10-01): 8.93 GB after the load, 9.94 GB at the peak of an answer, plus the CUDA
#: context. With less free than this the weights spill into shared system memory on Windows and one sentence takes
#: about a minute instead of two seconds; on Linux the load fails. `tests/test_vlm_inference.py` keeps its own copy
#: in a class attribute for the rest of the process, so this test checks before it loads a second one. The 8B's bf16
#: weights alone are 17.5 GB (not measured loaded).
_VRAM_NEEDED_BYTES = 10.5e9 if "8B" not in MODEL_ID else 20.0e9


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _weights_present() -> bool:
    """True if the weights are on this box: at ``MODEL_PATH``, or the snapshot already in the cache. Never downloads;
    that is the fetch script's job."""
    if MODEL_PATH is not None:
        return os.path.isdir(MODEL_PATH)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=MODEL_ID, local_files_only=True)
    except Exception:  # noqa: BLE001 - any failure means "not usable here"
        return False
    return True


class _OnTheRealModel(unittest.TestCase):
    """One copy of the weights per test, loaded through a fresh holder and unloaded after it."""

    def setUp(self) -> None:
        import torch

        free, _ = torch.cuda.mem_get_info()
        if free < _VRAM_NEEDED_BYTES:
            self.skipTest(f"{free / 1e9:.1f} GB of VRAM free, the model needs about {_VRAM_NEEDED_BYTES / 1e9:.1f} GB: "
                          f"another copy or process holds the card")
        shared_vlm().forget()
        self.holder = VlmHolder.from_parts()
        self.addCleanup(self._release)
        self.vlm = VlmConfig(model_id=MODEL_ID, model_path=MODEL_PATH, local=True)
        status = self.holder.load_for(self.vlm)
        self.assertEqual(status.state, "ready", status.cause)

    def _release(self) -> None:
        self.holder.release_unless_requested(None)  # unloads: the weights leave the card here, not at a later GC
        self.holder.forget()
        shared_vlm().forget()
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()


@unittest.skipUnless(_cuda_available(), "no CUDA in this interpreter")
@unittest.skipUnless(_weights_present(), f"{MODEL_ID} not in the HF cache (scripts/model_weights/fetch.py)")
class TheOwnersSentencesOnTheRealModelTests(_OnTheRealModel):
    def test_the_owners_sentences_read_as_the_card_needs_them(self) -> None:
        ask = self.holder.asker_for(self.vlm, may_load=False)
        expected = [
            ("Leg den grünen Würfel in die blaue Kiste",
             dict(intent="task", object=("green", "cube"), place=("blue",), place_pose=None, scope="once")),
            ("Räum die Kiste aus",
             dict(intent="task", object=None, place=None, place_pose=None, scope="until_empty")),
            ("Nimm alle Schrauben und leg sie auf Ablage links",
             dict(intent="task", object=("screw",), place=None, place_pose="ablage_links", scope="until_empty")),
            ("Put the red cylinder on Ablage links and go back to the Wartepose",
             dict(intent="task", object=("red", "cylinder"), place=None, place_pose="ablage_links", scope="once",
                  return_to="wartepose")),
            ("Nimm den roten Zylinder und fahr danach in die Wartepose",
             dict(intent="task", object=("red", "cylinder"), place=None, place_pose=None, scope="once",
                  return_to="wartepose")),
            ("Stopp!", dict(intent="stop")),
            ("Hallo Willy, wie geht es dir?", dict(intent="none")),
            # A German noun the router takes for English: the card must get it in English.
            ("Nimm die Mutter",
             dict(intent="task", object=("nut",), place=None, place_pose=None, scope="once")),
            # The model answers until_empty with the count (measured); the card reads one part.
            ("Nimm alle drei Würfel",
             dict(intent="task", object=("cube",), place=None, place_pose=None, scope="once")),
        ]
        for sentence, want in expected:
            with self.subTest(sentence=sentence):
                reading = understand(sentence, ask=ask, poses=POSES)
                self.assertTrue(reading.understood, f"{reading.reason} | raw: {reading.raw}")
                self.assertEqual(reading.intent, want["intent"], reading.raw)
                if want["intent"] != "task":
                    continue
                words = want["object"]
                if words is None:
                    self.assertIsNone(reading.object, reading.raw)
                else:
                    self.assertIsNotNone(reading.object, reading.raw)
                    assert reading.object is not None
                    for word in words:
                        self.assertIn(word, reading.object.phrase, reading.raw)
                if want["place"] is None:
                    self.assertIsNone(reading.place, reading.raw)
                else:
                    self.assertIsNotNone(reading.place, reading.raw)
                    assert reading.place is not None
                    for word in want["place"]:
                        self.assertIn(word, reading.place.phrase, reading.raw)
                self.assertEqual(reading.place_pose, want["place_pose"], reading.raw)
                self.assertEqual(reading.scope, want["scope"], reading.raw)
                if "return_to" in want:
                    self.assertEqual(reading.return_to, want["return_to"], reading.raw)

    def test_the_mat_the_parts_lie_on_is_never_the_object(self) -> None:
        """The demo's sentence and the ways the owner may say it on the day (2026-10-07): everything off the foam mat
        into the yellow bin. The mat is where the parts lie, never what is picked; the yellow bin is the place, and the
        task runs until the mat is clear."""
        ask = self.holder.asker_for(self.vlm, may_load=False)
        for sentence, place in (
            ("alle Objekte auf der Schaumstoffmatte in die gelbe Kiste", ("yellow",)),
            ("Leg alle Teile von der Matte in die gelbe Kiste", ("yellow",)),
            ("Nimm alles von der Schaumstoffmatte und pack es in die gelbe Box", ("yellow",)),
            ("Räum die Schaumstoffmatte ab", None),
        ):
            with self.subTest(sentence=sentence):
                reading = understand(sentence, ask=ask, poses=POSES)
                self.assertTrue(reading.understood, f"{reading.reason} | raw: {reading.raw}")
                self.assertEqual("task", reading.intent, reading.raw)
                if reading.object is not None:
                    for word in ("mat", "foam", "table"):
                        self.assertNotIn(word, reading.object.phrase, reading.raw)
                if place is None:
                    self.assertIsNone(reading.place, reading.raw)
                else:
                    self.assertIsNotNone(reading.place, reading.raw)
                    assert reading.place is not None
                    for word in place:
                        self.assertIn(word, reading.place.phrase, reading.raw)
                self.assertEqual("until_empty", reading.scope, reading.raw)


#: The owner's long sentence of 2026-10-08 (348 characters), as ``tests/test_vlm_command.py`` reads it from a canned
#: answer.
_LONG_SENTENCE = (
    "Willy, sei bitte so gut und nimm von den Teilen, die da vorne auf der schwarzen Schaumstoffmatte liegen, den "
    "kleinen grauen Würfel, der oben auf dem größeren grauen Würfel liegt, und leg ihn vorsichtig in die gelbe Kiste "
    "rechts daneben, und wenn du fertig bist, fahr bitte wieder in die Wartepose zurück, damit ich die nächsten Teile "
    "hinlegen kann."
)


def _stacked_squares() -> tuple[Any, tuple[float, float, float, float]]:
    """Two gray squares of one colour on near-white, in BGR, the smaller lying across the upper corner of the larger:
    the one on top is whole, outlined dark, and higher in the picture, so "on top" means it either way; and its box,
    by construction. (A first scene with the small square inside the large one read as a frame: the 4B boxed the large
    one, 2026-10-08.)"""
    import numpy as np

    image = np.full((480, 640, 3), 245, dtype=np.uint8)
    image[200:400, 180:380] = (128, 128, 128)  # the one below
    image[116:264, 296:444] = (20, 20, 20)  # the outline of the one on top
    image[120:260, 300:440] = (128, 128, 128)  # the one on top
    return image, (296.0, 116.0, 444.0, 264.0)


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@unittest.skipUnless(_cuda_available(), "no CUDA in this interpreter")
@unittest.skipUnless(_weights_present(), f"{MODEL_ID} not on this box (scripts/model_weights/fetch.py)")
class TheOwnersDayOnTheRealModelTests(_OnTheRealModel):
    """The sentences of 2026-10-08, read through the instruction as it asks since: compact, only what was said, the one
    part singled out in ``which`` and where the parts lie in ``from``. Each check names the words a card needs, not a
    whole answer, so a model that words them its own way still passes where the card would be right."""

    def _ask(self) -> Any:
        return self.holder.asker_for(self.vlm, may_load=False, max_new_tokens=COMMAND_MAX_NEW_TOKENS)

    def test_the_days_sentences_read_as_the_card_needs_them(self) -> None:
        ask = self._ask()
        cases = [
            ("Alle grauen Würfel in die Gelbe Kiste.",
             dict(object=("gray", "cube"), place=("yellow",), scope="until_empty", which=None)),
            ("Alle grauen Würfel von der schwarzen Matte in die gelbe Kiste",
             dict(object=("gray", "cube"), place=("yellow",), scope="until_empty", source=("mat",), which=None)),
            ("Nimm den grauen Würfel, der oben auf dem anderen liegt, und leg ihn in die gelbe Kiste",
             dict(object=("cube",), place=("yellow",), scope="once", which=("cube",))),
            ("Nimm den oberen grauen Würfel", dict(object=("cube",), scope="once", which=("cube",))),
            ("Nimm den großen grauen Würfel", dict(object=("cube",), scope="once", which=None)),
            ("Nimm den kleinsten Würfel", dict(object=("cube",), scope="once", which=("small",))),
            ("Nimm den Becher links von der Kiste und leg ihn auf Ablage links",
             dict(object=("cup",), place_pose="ablage_links", scope="once", which=("left",))),
            ("Nimm die Colaflasche und leg sie in die gelbe Kiste", dict(object=("bottle",), place=("yellow",))),
            (_LONG_SENTENCE, dict(object=("cube",), place=("yellow",), scope="once", which=("cube",),
                                  return_to="wartepose")),
        ]
        for sentence, want in cases:
            with self.subTest(sentence=sentence[:60]):
                reading = understand(sentence, ask=ask, poses=POSES)
                self.assertTrue(reading.understood, f"{reading.reason} | raw: {reading.raw}")
                self.assertEqual("task", reading.intent, reading.raw)
                self.assertIsNotNone(reading.object, reading.raw)
                assert reading.object is not None
                for word in want["object"]:
                    self.assertIn(word, reading.object.phrase, reading.raw)
                for word in ("mat", "foam", "table"):
                    self.assertNotIn(word, reading.object.phrase, reading.raw)
                for word in want.get("place", ()):
                    self.assertIn(word, reading.place.phrase if reading.place else "", reading.raw)
                if want.get("which") is None:
                    self.assertIsNone(reading.which, reading.raw)
                else:
                    for word in want["which"]:
                        self.assertIn(word, reading.which or "", reading.raw)
                for word in want.get("source", ()):
                    self.assertIn(word, reading.source or "", reading.raw)
                for key in ("scope", "place_pose", "return_to"):
                    if key in want:
                        self.assertEqual(want[key], getattr(reading, key), reading.raw)
                if reading.object.said is not None:
                    self.assertLessEqual(len(reading.object.said.split()), 15, reading.raw)

    def test_the_mornings_sentence_answers_in_60_tokens_or_fewer(self) -> None:
        """70 tokens before 2026-10-08, every key written; about 140 ms each on the cell's 8B."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH or MODEL_ID, local_files_only=True)
        reading = understand("Alle grauen Würfel in die Gelbe Kiste.", ask=self._ask(), poses=POSES)
        self.assertTrue(reading.understood, f"{reading.reason} | raw: {reading.raw}")
        # The answer opens with the reader's "{", which is the prompt's, not the model's, and ends where its object
        # closes (`vlm.stop_at_answer_end`, on): no end token follows it any more.
        written = len(tokenizer(reading.raw[1:], add_special_tokens=False).input_ids)
        # 70 tokens before the sparse answer (2026-10-08). The 4B spaces its JSON (": " and ", "), which the
        # compact spelling of the examples would save: 54 measured on 2026-10-09 against about 43 without the spaces.
        self.assertLessEqual(written, 60, reading.raw)

    def test_the_gray_square_on_top_is_grounded_alone(self) -> None:
        """A which goes to the detector alone (``task.pick_phrase``): the stacked pair is one box, over the upper."""
        image, top = _stacked_squares()
        grounder = self.holder.asker_for(self.vlm, may_load=False).grounder
        found = grounder.detect_all(image, "the gray square on top of the other one")
        self.assertEqual(1, len(found), [detection.box for detection in found])
        self.assertGreaterEqual(_iou(tuple(found[0].box), top), 0.5, found[0].box)


if __name__ == "__main__":
    unittest.main()
