"""Building a cell's perception through a clean VLM holder loads no weights.

The VLM is held once per process, so a build does not own its copy: it asks the holder. That must not turn a build
into a load. Nothing reaches the GPU until a prompt or a command needs the model, as on the owner's cell (whose
``preload: false`` is the only behaviour since the key left on 2026-10-17); the holder itself is created empty, and
asking it how things stand builds nothing.

The load seam is a spy that fails the test if it is ever called, so "nothing loaded" is observed rather than
inferred from a flag.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

from src.config.schema.models.models_schema import PipelineConfig
from src.models.vlm import Qwen3VLGrounder, reader_availability, shared_vlm


def _never_load(*_: Any) -> Any:
    raise AssertionError("the VLM weights were loaded while the cell was only being built")


class ACleanHolderLoadsNothingAtBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        shared_vlm().forget()
        self.addCleanup(shared_vlm().forget)
        spy = mock.patch.object(Qwen3VLGrounder, "_load", side_effect=_never_load)
        self.load = spy.start()
        self.addCleanup(spy.stop)

    def _models(self, **pipeline: Any) -> Any:
        from src.config.loader import load_config

        return load_config().models.model_copy(update={"pipeline": PipelineConfig(**pipeline)})

    def _build(self, models: Any) -> Any:
        from src.models import factory
        from src.models.detection.closed_set import detector
        from src.models.segmentation.realtime import segmenter

        with mock.patch.object(segmenter, "Sam2Segmenter", return_value="SEG"), \
             mock.patch.object(detector, "RtDetrObjectDetector", return_value="RTDETR"):
            return factory.build_perception(models)

    def test_the_vlm_stack_builds_on_a_clean_holder_without_loading(self) -> None:
        models = self._models()
        backend = self._build(models)
        self.assertFalse(backend.detector.loaded)
        self.load.assert_not_called()
        self.assertEqual(shared_vlm().status_for(models.pipeline.zero_shot.vlm).state, "idle")

    def test_a_closed_set_cell_never_touches_the_holder(self) -> None:
        models = self._models(kind="closed_set")
        backend = self._build(models)
        self.assertEqual(backend.detector, "RTDETR")
        self.assertIsNone(shared_vlm().held_for(models.pipeline.zero_shot.vlm))
        self.load.assert_not_called()

    def test_the_shared_holder_is_created_empty(self) -> None:
        models = self._models()
        vlm = models.pipeline.zero_shot.vlm
        self.assertIsNone(shared_vlm().held_for(vlm))
        self.assertFalse(shared_vlm().loaded_for(vlm))
        self.assertEqual(shared_vlm().status_for(vlm).state, "idle")
        self.assertIsNone(shared_vlm().held_for(vlm), "asking how things stand built nothing")
        self.load.assert_not_called()

    def test_asking_whether_commands_can_be_read_loads_nothing(self) -> None:
        for kind in ("zero_shot", "closed_set"):
            with self.subTest(kind=kind):
                rule = reader_availability(self._models(kind=kind), weights_present=True)
                self.assertNotEqual(rule.state, "ready")
        self.load.assert_not_called()

    def test_the_first_look_loads_once_and_a_rebuild_takes_the_loaded_copy(self) -> None:
        """The owner's cell: one copy, loaded by the first prompt that needs it, never by a build."""
        self.load.side_effect = None
        self.load.return_value = ("PROCESSOR", mock.Mock(), mock.Mock())
        models = self._models()
        first = self._build(models)
        first.detector.ensure_loaded()  # the first look
        second = self._build(models)
        self.assertEqual(self.load.call_count, 1)
        self.assertIs(first.detector, second.detector)
        self.assertTrue(shared_vlm().loaded_for(models.pipeline.zero_shot.vlm))


if __name__ == "__main__":
    unittest.main()
