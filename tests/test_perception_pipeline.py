"""Choosing a perception stack instead of assembling one -- and refusing the stacks that cannot mean anything.

Three things are pinned here, and the second is the one that matters most:

1. **The facade builds what the legacy keys always built.** With ``models.pipeline`` unset, the stack is
   the same two objects wrapped in a backend -- so every cell that predates the block is unaffected.
2. **The two-stage chain moved, it was not rewritten.** Its two failure rules are behaviour, not
   accident: a detector error yields no objects at all (downstream reports an honest ``no_valid_grasp``
   rather than raising mid-pick), while a *single* segmentation error skips that object and keeps the
   rest (one bad mask in a cluttered bin must not discard the objects that segmented fine).
3. **Illegal combinations are refused at load, by name.** Every detector x segmenter pairing used to
   build silently, including ones where the prompt means something different to each half. The failure
   this whole arc exists to remove is a stack that runs confidently and grounds the wrong thing, and a
   warning in a log has never stopped a grasp.

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
        objects = backend.perceive(object(), "a cube. a mug.")

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
        """Deliberate: a multi-phrase prompt grounds neighbour clutter, and the sampler wants it."""
        backend = TwoStageBackend(_Detector(["cube", "cube"]), _Segmenter())
        self.assertEqual(len(backend.perceive(object(), "a cube. a cube.")), 2)


class PipelineMatrixTests(unittest.TestCase):
    """What the schema refuses, and what it quietly gets right."""

    def test_the_default_is_the_stack_this_repo_has_always_run(self) -> None:
        pipeline = PipelineConfig()
        self.assertEqual(pipeline.kind, "zero_shot")
        self.assertEqual(pipeline.zero_shot.backend, "grounded_sam")
        self.assertEqual(pipeline.zero_shot.segmenter, "sam2")

    def test_a_vlm_stack_may_choose_its_segmenter(self) -> None:
        """Reversed 2026-08-11. This used to be refused, on the belief that the VLM route always cuts
        with SAM2 -- but that was POLICY, not a technical limit: OneFormer implements the same
        box-prompted `segment_detection`, so it works with any box source, VLM included. Refusing it
        was a knob withheld for no reason."""
        pipeline = PipelineConfig(zero_shot={"backend": "vlm", "segmenter": "oneformer"})
        self.assertEqual(pipeline.zero_shot.segmenter, "oneformer")

    def test_a_vlm_stack_with_the_real_segmenter_is_legal(self) -> None:
        self.assertEqual(PipelineConfig(zero_shot={"backend": "vlm"}).zero_shot.backend, "vlm")

    def test_a_closed_set_stack_cannot_be_told_to_route(self) -> None:
        """Routing decides between open-vocabulary backends; a closed-set detector answers only with
        its trained class list, so the decision would have nothing to act on."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            PipelineConfig(kind="closed_set", router={"enabled": True})
        self.assertIn("closed-set detector answers only", str(ctx.exception))

    def test_a_closed_set_stack_that_never_mentioned_the_router_is_simply_resolved(self) -> None:
        """Failing someone for a default they never wrote -- when the correct value is unambiguous --
        is a validator being pedantic rather than protective."""
        pipeline = PipelineConfig(kind="closed_set")
        self.assertFalse(pipeline.router.enabled)

    def test_the_vlm_refuses_by_default_when_it_cannot_be_loaded(self) -> None:
        """Degrading would do exactly what this route exists to prevent: hand a complex prompt to the
        model that grounds it confidently and wrongly."""
        self.assertEqual(PipelineConfig().zero_shot.vlm.on_unavailable, "refuse")
        self.assertFalse(PipelineConfig().zero_shot.vlm.preload)

    def test_oneformer_no_longer_advertises_modes_it_does_not_implement(self) -> None:
        """The wrapper always calls ``post_process_instance_segmentation``, so ``semantic`` and
        ``panoptic`` validated cleanly and then ran instance segmentation anyway."""
        from src.config.schema.models.models_schema import OneFormerConfig

        self.assertEqual(
            OneFormerConfig(model_path="x", local=True, task="instance").task, "instance"
        )
        with self.assertRaises(pydantic.ValidationError):
            OneFormerConfig(model_path="x", local=True, task="panoptic")


class FactoryTests(unittest.TestCase):
    """``build_perception`` is the one-line API. It must resolve the legacy stack unchanged."""

    def test_an_absent_pipeline_block_builds_the_legacy_pair(self) -> None:
        from unittest.mock import patch

        from src.models import factory

        models = _models_config(pipeline=None)
        with patch.object(factory, "build_object_detector", return_value="DET") as det, \
             patch.object(factory, "build_segmenter", return_value="SEG") as seg:
            backend = factory.build_perception(models)

        det.assert_called_once()
        seg.assert_called_once()
        self.assertIsInstance(backend, TwoStageBackend)
        self.assertEqual((backend.detector, backend.segmenter), ("DET", "SEG"))

    def test_the_vlm_route_builds_a_guarded_backend_without_loading_the_vlm(self) -> None:
        """P3, router off: every prompt goes to the VLM. The grounder must still not load at build
        time -- `preload: false` is the default, and a cell that never picks must never pay."""
        from src.models import factory
        from src.models.vlm import GuardedVlmBackend

        models = _models_config(
            pipeline=PipelineConfig(zero_shot={"backend": "vlm"}, router={"enabled": False}),
        )
        with mock.patch.object(factory, "_build_named_segmenter", return_value="SEG") as seg:
            backend = factory.build_perception(models)

        seg.assert_called_once()
        self.assertIsInstance(backend, GuardedVlmBackend)
        grounder = backend._vlm.detector  # noqa: SLF001 - asserting the wiring, not the behaviour
        self.assertFalse(grounder.loaded, "the VLM must not be loaded while merely building the cell")
        self.assertEqual(backend._vlm.segmenter, "SEG")  # noqa: SLF001

    def test_the_routed_stack_builds_neither_model_until_a_prompt_arrives(self) -> None:
        """Router on: the join. Eager construction would load GroundingDINO AND Qwen at cell build."""
        from src.models import factory
        from src.models.routed_backend import RoutedPerceptionBackend

        models = _models_config(
            pipeline=PipelineConfig(zero_shot={"backend": "vlm"}, router={"enabled": True}),
        )
        with mock.patch.object(factory, "_build_named_segmenter", return_value="SEG") as seg:
            backend = factory.build_perception(models)

        self.assertIsInstance(backend, RoutedPerceptionBackend)
        self.assertEqual(backend.built_routes(), (), "no route may be built before a prompt selects one")
        seg.assert_called_once()  # ...but SAM2 is built ONCE and shared by both routes

    def test_an_explicit_router_without_a_vlm_is_refused_at_load(self) -> None:
        """Routing with nowhere better to send a hard prompt is a decision nothing can act on."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            PipelineConfig(zero_shot={"backend": "grounded_sam"}, router={"enabled": True})
        self.assertIn("needs zero_shot.backend='vlm'", str(ctx.exception))

    def test_the_default_pipeline_leaves_routing_off_rather_than_failing(self) -> None:
        """`pipeline: {}` must be valid: the router defaults on, the backend defaults to the phrase
        grounder, and rejecting a combination nobody wrote would be pedantry, not protection."""
        self.assertFalse(PipelineConfig().router.enabled)

    def test_the_vlm_route_pins_sam2_rather_than_reading_the_segmenter_knob(self) -> None:
        """There is no second mask source to choose between until a native-mask checkpoint exists, so
        offering the choice here would be a knob that cannot do anything."""
        from src.models import factory

        models = _models_config(
            pipeline=PipelineConfig(zero_shot={"backend": "vlm", "segmenter": "sam2"}),
        )
        with mock.patch.object(factory, "_build_named_segmenter", return_value="SEG") as seg:
            factory.build_perception(models)
        self.assertEqual(seg.call_args.args[1], "sam2")

    def test_degrade_without_a_phrase_detector_is_refused_at_build_time(self) -> None:
        """Caught here, not at the moment of degradation -- which would be mid-pick, far too late."""
        from src.models import factory

        models = _models_config(
            pipeline=PipelineConfig(zero_shot={"backend": "vlm", "vlm": {"on_unavailable": "degrade"}}),
            objectdetector=None,
        )
        with self.assertRaises(ValueError) as ctx:
            factory.build_perception(models)
        self.assertIn("models.objectdetector", str(ctx.exception))

    def test_a_closed_set_pipeline_without_its_config_block_says_what_is_missing(self) -> None:
        from src.models import factory

        models = _models_config(pipeline=PipelineConfig(kind="closed_set"), rtdetr=None)
        with self.assertRaises(ValueError) as ctx:
            factory.build_perception(models)
        self.assertIn("models.rtdetr", str(ctx.exception))


def _models_config(**overrides: object) -> object:
    """The loaded ModelsConfig with fields overridden -- avoids hand-building every required block."""
    from src.config.loader import load_config

    return load_config().models.model_copy(update=overrides)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
