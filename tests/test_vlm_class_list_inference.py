"""Real weights, real GPU: one call grounds two colours of part (the `inference` tier).

    python -m pytest tests/test_vlm_class_list_inference.py -m inference -q

Skips unless CUDA and the weights are both present, so it costs a normal run nothing, and it is NOT part of the CI gate:
a model load of 9 GB (the 4B) or more and seconds of GPU time per call. The checkpoint is the dev box's 4B unless the
environment names another: on the cell its 8B, by id and by the path its weights lie at (``WILLY_VLM_MODEL_ID``,
``WILLY_VLM_MODEL_PATH``).

WHAT THIS EXISTS TO CATCH. ``tests/test_vlm_class_list.py`` pins what the grounder asks for a class list and what the
parser makes of a given answer. Only this file can tell whether the model answers the class-list instruction as it is
asked: each square once, under the one description it matches, copied exactly; a square of a third colour under neither
description; no box for a description the scene does not hold; and no square boxed under two descriptions, which the
parser would make ``ambiguous`` and no rule could take. Not measured on any checkpoint yet (2026-10-09): how well the
cell's 8B keeps two colours apart in one call. Until it passes there, a task of one rule keeps today's prompt.
"""

from __future__ import annotations

import os
import unittest
from typing import Any

import numpy as np
import pytest

from src.models.vlm.parsing import AMBIGUOUS_LABEL
from src.models.vlm.qwen import class_list_prompt

pytestmark = pytest.mark.inference

MODEL_ID = os.environ.get("WILLY_VLM_MODEL_ID") or "Qwen/Qwen3-VL-4B-Instruct"
#: Where the weights lie, on a box that keeps them outside the Hugging Face cache (the cell's 8B); none reads the cache.
MODEL_PATH = os.environ.get("WILLY_VLM_MODEL_PATH") or None
#: What a copy needs free on the card: the 4B's load and an answer's peak, measured (`test_vlm_command_inference.py`),
#: and the 8B's bf16 weights alone (17.5 GB) with room for an answer, not measured loaded.
_VRAM_NEEDED_BYTES = 10.5e9 if "8B" not in MODEL_ID else 20.0e9

W, H = 640, 480
#: Three squares on near-white, ground truth by construction: (x0, y0, x1, y1) pixels and the BGR colour.
GREEN_GT, GREEN_BGR = (100.0, 100.0, 200.0, 200.0), (40, 180, 40)
RED_GT, RED_BGR = (400.0, 100.0, 520.0, 220.0), (30, 30, 220)
BLUE_GT, BLUE_BGR = (250.0, 300.0, 350.0, 400.0), (220, 60, 30)
#: The bar a box must clear to be the square it names: the standard referring-expression threshold, which the 4B
#: clears at about 0.87 on one square of this kind (`test_vlm_inference.py`).
MIN_IOU = 0.5


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


def _iou(a: Any, b: Any) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _scene() -> np.ndarray:
    """A green, a red and a blue square on near-white, in BGR."""
    image = np.full((H, W, 3), 245, dtype=np.uint8)
    for (x0, y0, x1, y1), colour in ((GREEN_GT, GREEN_BGR), (RED_GT, RED_BGR), (BLUE_GT, BLUE_BGR)):
        image[int(y0):int(y1), int(x0):int(x1)] = colour
    return image


def _said(label: str) -> str:
    return " ".join(str(label).casefold().split())


@unittest.skipUnless(_cuda_available(), "no CUDA in this interpreter")
@unittest.skipUnless(_weights_present(), f"{MODEL_ID} not on this box (scripts/model_weights/fetch.py)")
class OneCallForTwoColoursTests(unittest.TestCase):
    """One model load for the whole class, unloaded after it."""

    grounder: Any
    image: np.ndarray

    @classmethod
    def setUpClass(cls) -> None:
        import torch

        free, _ = torch.cuda.mem_get_info()
        if free < _VRAM_NEEDED_BYTES:
            raise unittest.SkipTest(f"{free / 1e9:.1f} GB of VRAM free, the model needs about "
                                    f"{_VRAM_NEEDED_BYTES / 1e9:.1f} GB: another copy or process holds the card")
        from src.models.vlm import Qwen3VLGrounder

        cls.grounder = Qwen3VLGrounder(model_id=MODEL_ID, model_path=MODEL_PATH, local=True, preload=True)
        cls.image = _scene()

    @classmethod
    def tearDownClass(cls) -> None:
        grounder = getattr(cls, "grounder", None)
        if grounder is not None:
            grounder.unload()

    def _found(self, descriptions: list[str]) -> list[Any]:
        return self.grounder.detect_all(self.image, class_list_prompt(descriptions))

    def test_each_square_is_found_under_its_own_description(self) -> None:
        found = self._found(["green square", "red square"])
        boxes = [(det.label, [round(v) for v in det.box]) for det in found]
        for description, other, truth in (("green square", "red square", GREEN_GT),
                                          ("red square", "green square", RED_GT)):
            with self.subTest(description=description):
                own = [det for det in found if _said(det.label) == description]
                self.assertTrue(own, f"no box under {description!r}: {boxes}")
                best = max(_iou(det.box, truth) for det in own)
                self.assertGreaterEqual(best, MIN_IOU, f"best IoU {best:.3f} under {description!r}: {boxes}")
                wrong = [det for det in found if _said(det.label) == other and _iou(det.box, truth) >= MIN_IOU]
                self.assertFalse(wrong, f"the {description} came back under {other!r} too: {boxes}")

    def test_a_square_of_a_third_colour_is_neither_description(self) -> None:
        found = self._found(["green square", "red square"])
        over_blue = [det.label for det in found if _iou(det.box, BLUE_GT) >= MIN_IOU]
        self.assertFalse([label for label in over_blue if _said(label) in ("green square", "red square")],
                         f"the blue square came back as {over_blue}")

    def test_a_clean_scene_has_nothing_ambiguous(self) -> None:
        """Each square once, under one description: nothing the parser had to take away from a rule."""
        found = self._found(["green square", "red square"])
        self.assertNotIn(AMBIGUOUS_LABEL, [det.label for det in found],
                         [(det.label, [round(v) for v in det.box]) for det in found])

    def test_a_description_the_scene_does_not_hold_gets_no_box(self) -> None:
        found = self._found(["green square", "yellow triangle"])
        boxes = [(det.label, [round(v) for v in det.box]) for det in found]
        self.assertFalse([det for det in found if _said(det.label) == "yellow triangle"],
                         f"invented a yellow triangle: {boxes}")
        self.assertGreaterEqual(max((_iou(det.box, GREEN_GT) for det in found if _said(det.label) == "green square"),
                                    default=0.0), MIN_IOU, boxes)

    def test_three_colours_in_one_call(self) -> None:
        found = self._found(["green square", "red square", "blue square"])
        boxes = [(det.label, [round(v) for v in det.box]) for det in found]
        for description, truth in (("green square", GREEN_GT), ("red square", RED_GT), ("blue square", BLUE_GT)):
            with self.subTest(description=description):
                best = max((_iou(det.box, truth) for det in found if _said(det.label) == description), default=0.0)
                self.assertGreaterEqual(best, MIN_IOU, f"best IoU {best:.3f} under {description!r}: {boxes}")

    def test_one_description_still_grounds_as_before(self) -> None:
        found = self.grounder.detect_all(self.image, "the red square")
        self.assertTrue(found, "no box for a square that is plainly there")
        self.assertGreaterEqual(max(_iou(det.box, RED_GT) for det in found), MIN_IOU,
                                [(det.label, [round(v) for v in det.box]) for det in found])


if __name__ == "__main__":
    unittest.main()
