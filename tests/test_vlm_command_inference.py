"""Real weights, real GPU: the command reader on the owner's sentences (the `inference` tier).

    python -m pytest tests/test_vlm_command_inference.py -m inference -q

Skips unless CUDA and the weights are both present, so it costs a normal run nothing, and it is NOT part of the
CI gate: a 9 GB model load and seconds of GPU time per sentence.

WHAT THIS EXISTS TO CATCH. `tests/test_vlm_command.py` pins what the reader does with a given answer. Only this
file can tell whether the model still gives that answer to the instruction as written: that it answers JSON at
all through the text-only path (no image, the prefill continued), that it translates the German object into
English, and that it gives a taught pose back by its NAME when it hears the label. One model load for one test;
each sentence is a subtest, so a miss names the sentence.
"""

from __future__ import annotations

import unittest

import pytest

from src.config.schema.models.models_schema import VlmConfig
from src.models.vlm import VlmHolder, shared_vlm, understand

pytestmark = pytest.mark.inference

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
POSES = {"Ablage links": "ablage_links", "Wartepose": "wartepose"}
#: Measured on the RTX 5080 (2026-10-01): 8.93 GB after the load, 9.94 GB at the peak of an answer, plus the CUDA
#: context. With less free than this the weights spill into shared system memory on Windows and one sentence takes
#: about a minute instead of two seconds; on Linux the load fails. `tests/test_vlm_inference.py` keeps its own copy
#: in a class attribute for the rest of the process, so this test checks before it loads a second one.
_VRAM_NEEDED_BYTES = 10.5e9


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _weights_present() -> bool:
    """True if the snapshot is already in the cache. Never downloads; that is the fetch script's job."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=MODEL_ID, local_files_only=True)
    except Exception:  # noqa: BLE001 - any failure means "not usable here"
        return False
    return True


@unittest.skipUnless(_cuda_available(), "no CUDA in this interpreter")
@unittest.skipUnless(_weights_present(), f"{MODEL_ID} not in the HF cache (scripts/model_weights/fetch.py)")
class TheOwnersSentencesOnTheRealModelTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        free, _ = torch.cuda.mem_get_info()
        if free < _VRAM_NEEDED_BYTES:
            self.skipTest(f"{free / 1e9:.1f} GB of VRAM free, the model needs about {_VRAM_NEEDED_BYTES / 1e9:.1f} GB: "
                          f"another copy or process holds the card")
        shared_vlm().forget()
        self.holder = VlmHolder.from_parts()
        self.addCleanup(self._release)
        self.vlm = VlmConfig(model_id=MODEL_ID, local=True)
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


if __name__ == "__main__":
    unittest.main()
