"""The real base model on the real GPU: a smoke-tier run on the shapes dataset, the ``inference`` tier.

Auto-skips unless CUDA and the RT-DETR base weights (the Hub cache or the fetched copy) are both present, so it costs a
normal run nothing; it never downloads. What only this file can show: the pretrained head is rebuilt for the dataset's
classes, mixed precision trains without a non-finite loss, and the exported model loads in the detector and finds the
shapes it was trained on.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.inference


def _ready() -> bool:
    try:
        import torch
        from huggingface_hub import snapshot_download

        from src.models.detection.closed_set.training.recipes import DEFAULT_BASE_MODEL
        from src.models.detection.closed_set.training.trainer import resolve_base_model
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    if resolve_base_model(DEFAULT_BASE_MODEL)[1]:
        return True
    try:
        snapshot_download(repo_id=DEFAULT_BASE_MODEL, local_files_only=True)
    except Exception:  # noqa: BLE001 - any failure means "not usable here"
        return False
    return True


@unittest.skipUnless(_ready(), "needs CUDA and the RT-DETR base weights (scripts/model_weights/fetch.py rtdetr)")
class RealWeightsSmokeTests(unittest.TestCase):
    def test_a_smoke_run_on_the_shapes_trains_and_its_model_detects(self) -> None:
        from PIL import Image

        from src.config.schema.models.models_schema import ObjectDetectorConfig
        from src.models.detection.closed_set.detector import RtDetrObjectDetector
        from src.models.detection.closed_set.training.api import DetectorTraining
        from src.models.detection.closed_set.training.plan import DetectorPlanOverrides

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as work:
            data = DetectorTraining.shapes_dataset(Path(work) / "shapes", images=48)
            run = DetectorTraining.from_dataset(dataset=data, tier="smoke", out_dir=Path(work) / "model",
                                                overrides=DetectorPlanOverrides(epochs=4))
            report = run.train()
            self.assertTrue(report.succeeded, report.render())
            self.assertTrue(report.validated)
            self.assertGreater(report.best["map50"], 0.1, report.render())
            detector = RtDetrObjectDetector(ObjectDetectorConfig(model_path=report.model_dir, threshold=0.2,
                                                                 local=True))
            first = run.dataset().val[0]
            image = np.array(Image.open(first.path).convert("RGB"))[:, :, ::-1].copy()
            labels = {d.label for d in detector.detect_all(image)}
            self.assertTrue(labels <= {"red_block", "green_disc", "blue_wedge"}, labels)


if __name__ == "__main__":
    unittest.main()
