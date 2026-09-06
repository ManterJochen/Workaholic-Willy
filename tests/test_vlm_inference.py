"""Real weights, real GPU, real grounding — the `inference` tier.

    D:\\isaacsim\\...\\python.bat -m pytest tests/test_vlm_inference.py -m inference -q

Auto-skips unless CUDA and the weights are both present, so it costs a normal run nothing. It is NOT
part of the CI gate: multi-gigabyte weights and minutes of GPU time.

**Run it with an interpreter that has CUDA.** On this workstation the project `.venv` carries
``torch+cpu``, so the GPU interpreter is Isaac's bundled Python — the same one that runs GroundingDINO
and SAM2 for M2. The Hugging Face cache is shared between them, which is what makes
``scripts/model_weights/fetch.py`` (run from the `.venv`, which can reach the hub) work at all.

WHAT THIS EXISTS TO CATCH. The unit tests assert what the parser does with a *given* string. Only this
file can tell us the model still emits the string we think it does. It was written after the model
turned out to ground on a **0-1000 grid** rather than in pixels — a mistake that produced well-formed,
in-range, sanely-sized boxes roughly 50-200 px away from the object, with nothing raised and nothing
logged. A synthetic scene with known ground truth is the cheapest possible detector for that entire
class of error.
"""

from __future__ import annotations

import unittest

import numpy as np
import pytest

pytestmark = pytest.mark.inference

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
W, H = 640, 480
RED_GT = (100.0, 100.0, 200.0, 200.0)
BLUE_GT = (400.0, 300.0, 520.0, 420.0)
#: The bar a grounding result must clear to count as "found it". 0.5 is the standard
#: referring-expression threshold; measured on this scene the model reaches ~0.87, so this is not a
#: threshold tuned to make the test pass.
MIN_IOU = 0.5


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _weights_present() -> bool:
    """True if the snapshot is already in the cache. Never downloads -- that is the script's job."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=MODEL_ID, local_files_only=True)
    except Exception:  # noqa: BLE001 - any failure means "not usable here"
        return False
    return True


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _scene() -> np.ndarray:
    """A red square and a blue square on near-white, in BGR. Ground truth by construction."""
    image = np.full((H, W, 3), 245, dtype=np.uint8)
    image[100:200, 100:200] = (0, 0, 255)
    image[300:420, 400:520] = (255, 0, 0)
    return image


@unittest.skipUnless(_cuda_available(), "no CUDA in this interpreter")
@unittest.skipUnless(_weights_present(), f"{MODEL_ID} not in the HF cache (scripts/model_weights/fetch.py)")
class QwenGroundingInferenceTests(unittest.TestCase):
    """One model load for the whole class -- it costs ~7 s and ~9 GB of VRAM."""

    grounder: object

    @classmethod
    def setUpClass(cls) -> None:
        from src.models.vlm import Qwen3VLGrounder

        cls.grounder = Qwen3VLGrounder(model_id=MODEL_ID, local=True, preload=True)
        cls.image = _scene()

    def test_grounds_the_red_square(self) -> None:
        dets = self.grounder.detect_all(self.image, "the red square")  # type: ignore[attr-defined]
        self.assertTrue(dets, "the model returned no box for an object that is plainly there")
        best = max(_iou(d.box, RED_GT) for d in dets)
        self.assertGreaterEqual(best, MIN_IOU, f"best IoU {best:.3f}; boxes={[d.box for d in dets]}")

    def test_grounds_the_blue_square(self) -> None:
        """The second colour matters: a coordinate-space bug can still look plausible on one object."""
        dets = self.grounder.detect_all(self.image, "the blue square")  # type: ignore[attr-defined]
        self.assertTrue(dets)
        best = max(_iou(d.box, BLUE_GT) for d in dets)
        self.assertGreaterEqual(best, MIN_IOU, f"best IoU {best:.3f}; boxes={[d.box for d in dets]}")

    def test_an_absent_object_yields_nothing(self) -> None:
        """The safety-critical one.

        Instruct-tuned models are agreeable by default and will invent a plausible box rather than
        return nothing. If this regresses, every prompt for something that is not in the bin becomes a
        confident grasp at a fabricated location.
        """
        dets = self.grounder.detect_all(self.image, "a green triangle")  # type: ignore[attr-defined]
        self.assertEqual(dets, [], f"invented {len(dets)} box(es) for an absent object")

    def test_boxes_land_inside_the_image(self) -> None:
        """The 0-1000-grid regression, stated as an invariant rather than as a number."""
        dets = self.grounder.detect_all(self.image, "the red square")  # type: ignore[attr-defined]
        for det in dets:
            x0, y0, x1, y1 = det.box
            self.assertGreaterEqual(x0, 0.0)
            self.assertGreaterEqual(y0, 0.0)
            self.assertLessEqual(x1, float(W))
            self.assertLessEqual(y1, float(H))


if __name__ == "__main__":
    unittest.main()
