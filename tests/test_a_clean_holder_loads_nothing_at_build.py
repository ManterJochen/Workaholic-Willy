"""Building a cell's perception through a clean VLM holder loads no weights.

The VLM is now held once per process, so a build no longer owns its copy: it asks the holder. That must not turn
a build into a load. With ``preload: false`` (the schema default) nothing reaches the GPU until a prompt or a
command needs the model, exactly as before the holder existed; the holder itself is created empty, and asking it
how things stand builds nothing.

The load seam is a spy that fails the test if it is ever called, so "nothing loaded" is observed rather than
inferred from a flag. ``preload: true`` is the one documented exception (the owner's profile), and it loads once.
"""

from __future__ import annotations

import sys
import types
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

        with mock.patch.object(factory, "_build_named_segmenter", return_value="SEG"):
            return factory.build_perception(models)

    def test_the_vlm_stack_builds_on_a_clean_holder_without_loading(self) -> None:
        models = self._models(zero_shot={"backend": "vlm"}, router={"enabled": False})
        backend = self._build(models)
        grounder = backend._vlm.detector  # noqa: SLF001 - asserting the wiring, not the behaviour
        self.assertFalse(grounder.loaded)
        self.load.assert_not_called()
        self.assertEqual(shared_vlm().status_for(models.pipeline.zero_shot.vlm).state, "idle")

    def test_the_routed_stack_builds_on_a_clean_holder_without_building_a_route(self) -> None:
        models = self._models(zero_shot={"backend": "vlm"}, router={"enabled": True})
        backend = self._build(models)
        self.assertEqual(backend.built_routes(), ())
        self.load.assert_not_called()
        self.assertIsNone(shared_vlm().held_for(models.pipeline.zero_shot.vlm),
                          "the lazy route did not even take a grounder from the holder")

    def test_a_phrase_grounder_cell_never_touches_the_holder(self) -> None:
        from src.models import factory

        models = self._models(zero_shot={"backend": "grounded_sam"})
        # A stand-in module: the real one imports torch at module scope, and this test is about the holder.
        dino = types.SimpleNamespace(GroundingDinoObjectDetector=lambda *args, **kwargs: "DINO")
        with mock.patch.dict(sys.modules, {"src.models.detection.zero_shot.detector": dino}), \
             mock.patch.object(factory, "_build_named_segmenter", return_value="SEG"):
            backend = factory.build_perception(models)
        self.assertEqual(backend.detector, "DINO")
        self.assertIsNone(shared_vlm().held_for(models.pipeline.zero_shot.vlm))
        self.load.assert_not_called()

    def test_the_shared_holder_is_created_empty(self) -> None:
        models = self._models(zero_shot={"backend": "vlm"})
        vlm = models.pipeline.zero_shot.vlm
        self.assertIsNone(shared_vlm().held_for(vlm))
        self.assertFalse(shared_vlm().loaded_for(vlm))
        self.assertEqual(shared_vlm().status_for(vlm).state, "idle")
        self.assertIsNone(shared_vlm().held_for(vlm), "asking how things stand built nothing")
        self.load.assert_not_called()

    def test_asking_whether_commands_can_be_read_loads_nothing(self) -> None:
        for backend in ("vlm", "grounded_sam"):
            with self.subTest(backend=backend):
                rule = reader_availability(self._models(zero_shot={"backend": backend}), weights_present=True)
                self.assertNotEqual(rule.state, "ready")
        self.load.assert_not_called()

    def test_preload_is_the_one_build_that_loads_and_it_loads_once(self) -> None:
        """The owner's profile (backend vlm, router off, preload on): VRAM held from the build, one copy."""
        self.load.side_effect = None
        self.load.return_value = ("PROCESSOR", mock.Mock(), mock.Mock())
        models = self._models(zero_shot={"backend": "vlm", "vlm": {"preload": True}}, router={"enabled": False})
        first = self._build(models)
        second = self._build(models)
        self.assertEqual(self.load.call_count, 1)
        self.assertIs(first._vlm.detector, second._vlm.detector)  # noqa: SLF001
        self.assertTrue(shared_vlm().loaded_for(models.pipeline.zero_shot.vlm))


if __name__ == "__main__":
    unittest.main()
