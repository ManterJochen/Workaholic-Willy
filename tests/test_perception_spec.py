"""The four fields that decide a perception stack, and the refusals the builder and its prediction share.

Measured 2026-09-05 against the shipped tree and the code as it was the hour before, and cut to four fields on
2026-10-17, when GroundingDINO, OneFormer and the router left (the owner's cleanup): the VLM or RT-DETR finds the
boxes, SAM2 cuts them, and every model reads its weights under ``models.weights_root``.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from src.config.schema.app import ModelsConfig
from src.config.schema.models.models_schema import PipelineConfig, VlmConfig
from src.models import factory
from src.models.perception_spec import PerceptionSpec

_REPO = Path(__file__).resolve().parents[1]
_CELLS = _REPO / "src" / "robot" / "execution" / "autonomous_grasp" / "cells.py"


def _models() -> ModelsConfig:
    from src.config import load_config

    return load_config().models


class FourFieldsTests(unittest.TestCase):
    def test_the_spec_carries_exactly_what_the_builder_reads(self) -> None:
        """⭐ THE CLAIM, CHECKED AGAINST THE BUILDER'S SOURCE RATHER THAN AGAINST A LIST. Every
        `models.<x>` attribute read anywhere in `factory.py` must be a field of the spec, and every
        field of the spec must be read there. A hand-kept list of names would be a second declaration
        of the builder's appetite and would rot the first time it grew another."""
        tree = ast.parse((_REPO / "src" / "models" / "factory.py").read_text("utf-8"))
        read: set[str] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "models"
            ):
                read.add(node.attr)
        fields = {f.name for f in PerceptionSpec.__dataclass_fields__.values()}
        self.assertEqual(read, fields, "the spec and the builder disagree about what is read")
        self.assertEqual({"segmenter", "rtdetr", "pipeline", "weights_root"}, fields)

    def test_the_one_field_perception_never_reads_is_speech(self) -> None:
        """⛔ THE REASON THIS IS NOT JUST `ModelsConfig`. `stt` is REQUIRED on `ModelsConfig` and is no perception
        block, so a Python caller that builds a stack must not need a Whisper block to do it."""
        fields = {f.name for f in PerceptionSpec.__dataclass_fields__.values()}
        self.assertEqual({"stt"}, set(ModelsConfig.model_fields) - fields)
        self.assertTrue(ModelsConfig.model_fields["stt"].is_required())

    def test_a_spec_can_be_built_with_no_stt_block_at_all(self) -> None:
        models = _models()
        spec = PerceptionSpec.zero_shot(segmenter=models.segmenter)
        resolved = spec.resolve()
        self.assertEqual(("zero_shot", "vlm", "sam2"), (resolved.kind, resolved.detector, resolved.segmenter))
        self.assertEqual(VlmConfig().model_id, resolved.vlm_model_id, "no block given is the schema's checkpoint")
        self.assertTrue(resolved.buildable)


class TheShippedTreeTests(unittest.TestCase):
    def test_the_shipped_config_resolves_to_the_vlm_and_sam2(self) -> None:
        """⭐ The owner's cell's stack (2026-10-17: only the VLM grounds free text)."""
        models = _models()
        resolved = PerceptionSpec.from_config(models).resolve()
        self.assertEqual(("zero_shot", "vlm", "sam2"), (resolved.kind, resolved.detector, resolved.segmenter))
        self.assertEqual(models.pipeline.zero_shot.vlm.model_id, resolved.vlm_model_id)
        self.assertEqual(models.weights_root, PerceptionSpec.from_config(models).weights_root)

    def test_one_word_changes_the_stack_the_real_cell_builds(self) -> None:
        models = _models()
        closed = models.model_copy(update={"pipeline": PipelineConfig(kind="closed_set")})
        resolved = PerceptionSpec.from_config(closed).resolve()
        self.assertEqual(("closed_set", "rtdetr"), (resolved.kind, resolved.detector))
        self.assertIsNone(resolved.vlm_model_id)

    def test_the_cell_builds_through_the_spec(self) -> None:
        """⚠ Parsed, not grepped: a comment in `cells.py` explains what it used to call, and a
        substring check would find the name in the prose describing its removal."""
        tree = ast.parse(_CELLS.read_text(encoding="utf-8"))
        fn = next(n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "build_real_components")
        code = ast.dump(fn)
        self.assertIn("PerceptionSpec", code)
        self.assertNotIn("build_object_detector", code)
        self.assertNotIn("build_segmenter", code)


class RefusalsAreTheBuildersOwnWordsTests(unittest.TestCase):
    def test_a_predicted_refusal_is_the_string_the_builder_raises(self) -> None:
        """⛔ NOT A PARAPHRASE. The console answers 'which stack would this cell build' before any
        weight loads. Both halves read the same constants, so a predicted refusal and a raised one
        cannot be different sentences -- asserted by RAISING one and comparing."""
        models = _models()
        broken = models.model_copy(update={"pipeline": PipelineConfig(kind="closed_set"), "rtdetr": None})
        spec = PerceptionSpec.from_config(broken)
        predicted = spec.resolve()
        self.assertFalse(predicted.buildable)
        with self.assertRaises(ValueError) as caught:
            spec.build()
        self.assertEqual(str(caught.exception), predicted.refusal)
        self.assertEqual(predicted.refusal, factory.REFUSE_CLOSED_SET_WITHOUT_RTDETR)

    def test_a_python_spec_without_sam2_is_predicted_and_refused_alike(self) -> None:
        spec = PerceptionSpec(segmenter=None)
        predicted = spec.resolve()
        self.assertEqual(factory.REFUSE_SAM2_WITHOUT_BLOCK, predicted.refusal)
        with self.assertRaises(ValueError) as caught:
            spec.build()
        self.assertEqual(str(caught.exception), predicted.refusal)

    def test_a_closed_set_python_spec_is_buildable_from_its_two_blocks(self) -> None:
        models = _models()
        spec = PerceptionSpec.closed_set(rtdetr=models.rtdetr, segmenter=models.segmenter)
        self.assertEqual("rtdetr", spec.resolve().detector)
        self.assertTrue(spec.resolve().buildable)


class ResolutionRenderTests(unittest.TestCase):
    def test_render_is_ascii_whole_and_unterminated(self) -> None:
        models = _models()
        for spec in (PerceptionSpec.from_config(models),
                     PerceptionSpec.from_config(models.model_copy(update={"pipeline": PipelineConfig(
                         kind="closed_set")}))):
            text = spec.resolve().render()
            text.encode("ascii")
            self.assertFalse(text.endswith("\n"))
            self.assertIn("perception stack:", text)

    def test_to_dict_and_the_properties_agree(self) -> None:
        resolution = PerceptionSpec.from_config(_models()).resolve()
        payload = resolution.to_dict()
        self.assertEqual(payload["buildable"], resolution.buildable)
        self.assertEqual(payload["detector"], resolution.detector)


class ConsoleReadsTheSameResolutionTests(unittest.TestCase):
    def test_the_endpoint_does_not_re_derive_the_branch_logic(self) -> None:
        """⛔ It read `models.pipeline` and described the stack in prose, for a cell built otherwise. Both halves
        were individually correct, which is why no test could catch it."""
        source = (_REPO / "api" / "routers" / "diagnostics.py").read_text(encoding="utf-8")
        fn = next(n for n in ast.parse(source).body
                  if isinstance(n, ast.FunctionDef) and n.name == "_perception")
        self.assertIn("PerceptionSpec", ast.dump(fn))
        read_off_pipeline = {node.attr for node in ast.walk(fn)
                             if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute)
                             and node.value.attr == "pipeline"}
        self.assertEqual(set(), read_off_pipeline, "the stack is the resolution's to decide, not the endpoint's")

    def test_the_endpoint_reports_a_refusal_in_the_builders_words(self) -> None:
        from api.routers.diagnostics import _perception

        models = _models().model_copy(update={"pipeline": PipelineConfig(kind="closed_set"), "rtdetr": None})

        class _Cell:
            def config(self):  # noqa: ANN202
                from src.config import load_config

                return load_config().model_copy(update={"models": models})

        out = _perception(_Cell())  # type: ignore[arg-type]
        self.assertEqual(("closed_set", "rtdetr", "sam2"), (out.kind, out.backend, out.segmenter))
        self.assertIn(factory.REFUSE_CLOSED_SET_WITHOUT_RTDETR, out.detail)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
