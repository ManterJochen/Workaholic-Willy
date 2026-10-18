"""Choosing a perception stack instead of assembling one -- and refusing the stacks that cannot mean anything.

Three things are pinned here, and the second is the one that matters most:

1. **The facade builds the cell's stack.** ``models.pipeline`` names it: the VLM grounds free text
   (``zero_shot``, the default) or RT-DETR answers with its trained classes (``closed_set``), and SAM2 cuts
   every box either way. GroundingDINO, OneFormer, the prompt router and the fallback to GroundingDINO left
   on 2026-10-17 (the owner's cleanup), so a cell whose VLM cannot load refuses the pick.
2. **The two-stage chain moved, it was not rewritten.** Its two failure rules are behaviour, not
   accident: a detector error yields no objects at all (downstream reports an honest ``no_valid_grasp``
   rather than raising mid-pick), while a *single* segmentation error skips that object and keeps the
   rest (one bad mask in a cluttered bin must not discard the objects that segmented fine).
3. **A key that chose what left is refused at load, by name.** A stack nobody can build must not load
   quietly and run a different one.

No model weights and no torch inference: these exercise the wiring, which is exactly the part that used
to be duplicated in six places.
"""

from __future__ import annotations

import unittest
from unittest import mock

import pydantic

from src.config.schema.models.models_schema import PipelineConfig
from src.models.perception_backend import (
    PerceivedObject,
    PerceptionBackend,
    TwoStageBackend,
)


class _Detection:
    """Stand-in for the frozen Detection; only ``box`` is read by the code under test."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.box = [0.0, 0.0, 10.0, 10.0]


class _Segmentation:
    def __init__(self, label: str) -> None:
        self.label = label


class _Detector:
    def __init__(self, labels: list[str] | None = None, *, raises: bool = False) -> None:
        self._labels = labels or []
        self._raises = raises
        self.calls: list[tuple[object, str]] = []

    def detect_all(self, image: object, prompt: str) -> list[_Detection]:
        self.calls.append((image, prompt))
        if self._raises:
            raise RuntimeError("the detector fell over")
        return [_Detection(label) for label in self._labels]


class _Segmenter:
    def __init__(self, *, fail_on: set[str] | None = None) -> None:
        self._fail_on = fail_on or set()
        self.calls: list[str] = []

    def segment_detection(self, image: object, detection: _Detection) -> _Segmentation:
        self.calls.append(detection.label)
        if detection.label in self._fail_on:
            raise RuntimeError(f"no mask for {detection.label}")
        return _Segmentation(detection.label)


class TwoStageBackendTests(unittest.TestCase):
    """The chain that used to live inside every perception source."""

    def test_it_pairs_each_detection_with_its_mask(self) -> None:
        backend = TwoStageBackend(_Detector(["cube", "mug"]), _Segmenter())
        objects = backend.perceive(object(), "cube | mug")

        self.assertEqual(len(objects), 2)
        self.assertTrue(all(isinstance(o, PerceivedObject) for o in objects))
        self.assertEqual([o.detection.label for o in objects], ["cube", "mug"])
        self.assertEqual([o.segmentation.label for o in objects], ["cube", "mug"])

    def test_it_satisfies_the_backend_protocol(self) -> None:
        self.assertIsInstance(TwoStageBackend(_Detector(), _Segmenter()), PerceptionBackend)

    def test_the_prompt_reaches_the_detector_verbatim(self) -> None:
        """No normalisation here. Whatever a caller grounds with is what the detector receives."""
        detector = _Detector(["cube"])
        TwoStageBackend(detector, _Segmenter()).perceive("IMG", "a small green cube.")
        self.assertEqual(detector.calls, [("IMG", "a small green cube.")])

    def test_a_detector_failure_yields_nothing_rather_than_raising(self) -> None:
        """Downstream this reads as an honest ``no_valid_grasp``; raising would be a traceback
        in the middle of a pick, on a cell that is holding something."""
        backend = TwoStageBackend(_Detector(raises=True), _Segmenter())
        self.assertEqual(backend.perceive(object(), "anything"), ())

    def test_one_bad_mask_does_not_discard_the_objects_that_worked(self) -> None:
        """A cluttered bin routinely has one object SAM2 cannot cut. Dropping the whole frame for it
        would throw away the neighbours the dense sampler needs."""
        segmenter = _Segmenter(fail_on={"mug"})
        backend = TwoStageBackend(_Detector(["cube", "mug", "box"]), segmenter)
        objects = backend.perceive(object(), "everything")

        self.assertEqual([o.detection.label for o in objects], ["cube", "box"])
        self.assertEqual(segmenter.calls, ["cube", "mug", "box"], "every detection was still attempted")

    def test_detections_are_not_deduplicated(self) -> None:
        """Deliberate: a class list grounds neighbour clutter, and the sampler wants it."""
        backend = TwoStageBackend(_Detector(["cube", "cube"]), _Segmenter())
        self.assertEqual(len(backend.perceive(object(), "a cube")), 2)


class PipelineSchemaTests(unittest.TestCase):
    """What the schema holds, and what it refuses."""

    def test_the_default_is_the_owners_stack(self) -> None:
        pipeline = PipelineConfig()
        self.assertEqual(pipeline.kind, "zero_shot")
        self.assertEqual("Qwen/Qwen3-VL-4B-Instruct-FP8", pipeline.zero_shot.vlm.model_id)

    def test_the_vlm_block_holds_its_model_id_alone(self) -> None:
        self.assertEqual({"model_id"}, set(type(PipelineConfig().zero_shot.vlm).model_fields))
        self.assertEqual({"vlm"}, set(type(PipelineConfig().zero_shot).model_fields))
        self.assertEqual({"kind", "zero_shot"}, set(PipelineConfig.model_fields))

    def test_a_key_that_chose_what_left_is_refused(self) -> None:
        """The schema forbids them; the loader names each with its sentence (`tests/test_config_schema.py`)."""
        for values in ({"router": {"enabled": True}}, {"zero_shot": {"backend": "grounded_sam"}},
                       {"zero_shot": {"segmenter": "oneformer"}}, {"zero_shot": {"vlm": {"on_unavailable": "degrade"}}},
                       {"zero_shot": {"vlm": {"preload": True}}}, {"closed_set": {"segmenter": "sam2"}}):
            with self.subTest(values=values), self.assertRaises(pydantic.ValidationError):
                PipelineConfig(**values)


class FactoryTests(unittest.TestCase):
    """``build_perception`` is the one-line API."""

    def test_the_vlm_stack_builds_the_holders_grounder_without_loading_the_vlm(self) -> None:
        """Every prompt goes to the VLM. The grounder must still not load at build time: the owner's cell loads it at
        its first prompt, and a cell that never picks must never pay."""
        from src.models import factory
        from src.models.segmentation.realtime import segmenter
        from src.models.vlm import Qwen3VLGrounder

        models = _models_config()
        with mock.patch.object(segmenter, "Sam2Segmenter", return_value="SEG") as seg:
            backend = factory.build_perception(models)

        seg.assert_called_once()
        self.assertIsInstance(backend, TwoStageBackend)
        self.assertIsInstance(backend.detector, Qwen3VLGrounder)
        self.assertFalse(backend.detector.loaded, "the VLM must not be loaded while merely building the cell")
        self.assertEqual(backend.segmenter, "SEG")
        self.assertEqual(models.weights_root, seg.call_args.kwargs["weights_root"])

    def test_the_closed_set_stack_builds_rtdetr_and_sam2_from_the_weights_root(self) -> None:
        from src.models import factory
        from src.models.detection.closed_set import detector
        from src.models.segmentation.realtime import segmenter

        models = _models_config(pipeline=PipelineConfig(kind="closed_set"))
        with mock.patch.object(detector, "RtDetrObjectDetector", return_value="DET") as det, \
             mock.patch.object(segmenter, "Sam2Segmenter", return_value="SEG"):
            backend = factory.build_perception(models)

        self.assertEqual((backend.detector, backend.segmenter), ("DET", "SEG"))
        self.assertIs(models.rtdetr, det.call_args.args[0])
        self.assertEqual(models.weights_root, det.call_args.kwargs["weights_root"])

    def test_a_closed_set_pipeline_without_its_config_block_says_what_is_missing(self) -> None:
        from src.models import factory

        models = _models_config(pipeline=PipelineConfig(kind="closed_set"), rtdetr=None)
        with self.assertRaises(ValueError) as ctx:
            factory.build_perception(models)
        self.assertIn("models.rtdetr", str(ctx.exception))

    def test_there_is_no_other_builder(self) -> None:
        from src.models import factory

        self.assertEqual(["build_perception"], factory.__all__)
        for gone in ("build_object_detector", "build_segmenter", "_build_named_segmenter"):
            self.assertFalse(hasattr(factory, gone), gone)


def _models_config(**overrides: object) -> object:
    """The loaded ModelsConfig with fields overridden -- avoids hand-building every required block."""
    from src.config.loader import load_config

    return load_config().models.model_copy(update=overrides)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
