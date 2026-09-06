"""The seven fields that decide a perception stack, and the block that reached nothing.

Measured 2026-09-05 against the shipped tree and the code as it was the hour before.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from src.config.schema.app import ModelsConfig
from src.config.schema.models.models_schema import (
    ClosedSetPipelineConfig,
    PipelineConfig,
    PromptRouterConfig,
    VlmConfig,
    ZeroShotPipelineConfig,
)
from src.models import factory
from src.models.perception_spec import PerceptionSpec

_REPO = Path(__file__).resolve().parents[1]
_CELLS = _REPO / "backend" / "src" / "robot" / "execution" / "autonomous_grasp" / "cells.py"


def _models() -> ModelsConfig:
    from src.config import load_config

    return load_config().models


class SevenNotTenTests(unittest.TestCase):
    def test_the_spec_carries_exactly_what_the_builder_reads(self) -> None:
        """⭐ THE CLAIM, CHECKED AGAINST THE BUILDER'S SOURCE RATHER THAN AGAINST A LIST. Every
        `models.<x>` attribute read anywhere in `factory.py` must be a field of the spec, and every
        field of the spec must be read there. A hand-kept list of seven names would be a second
        declaration of the builder's appetite and would rot the first time it grew an eighth."""
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
        self.assertEqual(len(fields), 7)

    def test_the_three_fields_perception_never_reads_are_the_expensive_ones(self) -> None:
        """⛔ THE REASON THIS IS NOT JUST `ModelsConfig`. `stt` is REQUIRED on `ModelsConfig` and has
        eleven fields, ten of them mandatory, none of them read by perception. The repository already
        measured that cost: `tests/test_perception_pipeline.py` gives up on constructing the input and
        loads the config from disk, with the docstring 'avoids hand-building every required block'."""
        from src.config.schema.models.models_schema import SpeechToTextConfig

        fields = {f.name for f in PerceptionSpec.__dataclass_fields__.values()}
        unread = set(ModelsConfig.model_fields) - fields
        self.assertEqual(unread, {"stt", "handdetect", "gesturedetect"})
        self.assertTrue(ModelsConfig.model_fields["stt"].is_required())
        mandatory = sum(f.is_required() for f in SpeechToTextConfig.model_fields.values())
        self.assertEqual((len(SpeechToTextConfig.model_fields), mandatory), (11, 10))

    def test_a_spec_can_be_built_with_no_stt_block_at_all(self) -> None:
        """The whole point of (b) for a Python caller: a perception stack without ten Whisper values."""
        models = _models()
        spec = PerceptionSpec.zero_shot(
            objectdetector=models.objectdetector, segmenter=models.segmenter,
        )
        self.assertEqual(spec.resolve().detector, "groundingdino")
        self.assertEqual(spec.resolve().source, "legacy keys")


class TheShippedTreeTests(unittest.TestCase):
    def test_the_shipped_config_resolves_to_the_pair_the_legacy_keys_gave(self) -> None:
        """⭐ WHY THE CONVERGENCE IS BYTE-IDENTICAL TODAY, provable at a desk with no GPU. The tree
        sets `pipeline: zero_shot / grounded_sam / sam2`, which resolves to exactly the GroundingDINO
        + SAM2 pair `build_object_detector` + `build_segmenter` resolved from the legacy keys. That
        is why nobody noticed the block was inert, and why converting `cells.py` moves nothing until
        somebody edits one word."""
        models = _models()
        through_pipeline = PerceptionSpec.from_config(models).resolve()
        self.assertEqual(through_pipeline.source, "models.pipeline")
        self.assertEqual((through_pipeline.detector, through_pipeline.segmenter),
                         ("groundingdino", "sam2"))

        legacy = PerceptionSpec.from_config(models.model_copy(update={"pipeline": None})).resolve()
        self.assertEqual(legacy.source, "legacy keys")
        self.assertEqual((legacy.detector, legacy.segmenter), ("groundingdino", "sam2"))

    def test_one_word_now_changes_the_stack_the_real_cell_builds(self) -> None:
        """⛔⛔ THE DEFECT, AS A DIFF. Before 2026-09-05 the real cell read only `models.detector`
        and `models.segmenter_backend`, so this edit changed nothing it built -- no error, no log
        line. The resolution is what `cells.py` now acts on."""
        models = _models()
        vlm = models.model_copy(update={"pipeline": PipelineConfig(
            kind="zero_shot",
            zero_shot=ZeroShotPipelineConfig(backend="vlm", segmenter="sam2"),
            router=PromptRouterConfig(enabled=False),
        )})
        resolved = PerceptionSpec.from_config(vlm).resolve()
        self.assertEqual(resolved.detector, "vlm")
        self.assertTrue(resolved.vlm_model_id)

        # And the LEGACY keys are untouched by that edit, which is precisely why the old path could
        # not see it: they still say groundingdino.
        self.assertEqual(vlm.detector, "groundingdino")

    def test_the_cell_no_longer_reads_the_legacy_doors(self) -> None:
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
        weight loads, and it used to answer by re-deriving the branch logic in prose. Both halves now
        read the same constants, so a predicted refusal and a raised one cannot be different
        sentences -- asserted by RAISING one and comparing."""
        models = _models()
        broken = models.model_copy(update={
            "pipeline": PipelineConfig(kind="closed_set", closed_set=ClosedSetPipelineConfig()),
            "rtdetr": None,
        })
        spec = PerceptionSpec.from_config(broken)
        predicted = spec.resolve()
        self.assertFalse(predicted.buildable)
        with self.assertRaises(ValueError) as caught:
            spec.build()
        self.assertEqual(str(caught.exception), predicted.refusal)
        self.assertEqual(predicted.refusal, factory.REFUSE_CLOSED_SET_WITHOUT_RTDETR)

    def test_the_router_refusal_is_predicted_and_raised_alike(self) -> None:
        models = _models()
        routed = models.model_copy(update={
            "objectdetector": None,
            "pipeline": PipelineConfig(
                kind="zero_shot",
                zero_shot=ZeroShotPipelineConfig(backend="vlm", vlm=VlmConfig()),
                router=PromptRouterConfig(enabled=True),
            ),
        })
        spec = PerceptionSpec.from_config(routed)
        predicted = spec.resolve()
        self.assertEqual(predicted.refusal, factory.REFUSE_ROUTER_WITHOUT_OBJECTDETECTOR)
        with self.assertRaises(ValueError) as caught:
            spec.build()
        self.assertEqual(str(caught.exception), predicted.refusal)

    def test_a_closed_set_python_spec_carries_no_phrase_grounder(self) -> None:
        """⚠ AND THAT IS WHY THE ROUTER REFUSAL IS REACHABLE FROM PYTHON AT ALL. `ModelsConfig` makes
        `objectdetector` REQUIRED, so a YAML closed-set cell always has one it never uses. A spec
        built in Python leaves it empty, which is the honest shape and makes the guard fire."""
        models = _models()
        spec = PerceptionSpec.closed_set(rtdetr=models.objectdetector, segmenter=models.segmenter)
        self.assertIsNone(spec.objectdetector)
        self.assertEqual(spec.resolve().detector, "rtdetr")
        self.assertTrue(spec.resolve().buildable)


class ResolutionRenderTests(unittest.TestCase):
    def test_render_is_ascii_whole_and_unterminated(self) -> None:
        for models in (_models(), _models().model_copy(update={"pipeline": None})):
            text = PerceptionSpec.from_config(models).resolve().render()
            text.encode("ascii")
            self.assertFalse(text.endswith("\n"))
            self.assertIn("decided by", text)

    def test_to_dict_and_the_properties_agree(self) -> None:
        resolution = PerceptionSpec.from_config(_models()).resolve()
        payload = resolution.to_dict()
        self.assertEqual(payload["buildable"], resolution.buildable)
        self.assertEqual(payload["detector"], resolution.detector)


class ConsoleReadsTheSameResolutionTests(unittest.TestCase):
    def test_the_endpoint_no_longer_re_derives_the_branch_logic(self) -> None:
        """⛔ It read `models.pipeline` and described the stack in prose, for a cell built from the
        legacy keys. Both halves were individually correct, which is why no test could catch it."""
        source = (_REPO / "api" / "routers" / "diagnostics.py").read_text(encoding="utf-8")
        fn = next(n for n in ast.parse(source).body
                  if isinstance(n, ast.FunctionDef) and n.name == "_perception")
        self.assertIn("PerceptionSpec", ast.dump(fn))

        # ⚠ SCOPED TO WHAT ACTUALLY MOVED, and my first version was not: it forbade the string
        # "closed_set" anywhere in the function and failed, correctly. The endpoint still branches on
        # `resolution.kind` to pick which SENTENCE an operator reads, and that is its job -- prose is
        # the one thing a wire schema cannot derive. What it must no longer do is read the pipeline
        # block and decide the stack from it a second time.
        read_off_pipeline = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
                if node.value.attr == "pipeline":
                    read_off_pipeline.add(node.attr)
                # `models.pipeline.zero_shot.<x>` -- one level deeper.
                if isinstance(node.value.value, ast.Attribute) and node.value.value.attr == "pipeline":
                    read_off_pipeline.add(f"{node.value.attr}.{node.attr}")
        self.assertEqual(
            read_off_pipeline, {"zero_shot", "zero_shot.vlm"},
            "the only remaining read is the VLM block, and only for `model_path` -- a filesystem "
            "check the resolution deliberately does not carry, because it is not part of deciding "
            "the stack. Anything else here is the branch logic growing back.",
        )

    def test_the_vlm_branch_reports_the_segmenter_instead_of_pinning_sam2(self) -> None:
        """⛔ MEASURED DEFECT. That branch hard-coded `segmenter="sam2"` while `_build_vlm_backend`
        reads `pipeline.zero_shot.segmenter`. The pin was removed from the schema on 2026-08-11 and
        the console did not follow, so a VLM cell cutting masks with OneFormer was reported as SAM2.
        """
        from api.routers.diagnostics import _perception

        models = _models().model_copy(update={
            "pipeline": PipelineConfig(
                kind="zero_shot",
                zero_shot=ZeroShotPipelineConfig(
                    backend="vlm", segmenter="oneformer", vlm=VlmConfig(),
                ),
                router=PromptRouterConfig(enabled=False),
            ),
            "oneformer": None,
        })

        class _Cell:
            def config(self):  # noqa: ANN202
                from src.config import load_config

                return load_config().model_copy(update={"models": models})

        out = _perception(_Cell())  # type: ignore[arg-type]
        # oneformer without a config block is refused, and the console says so in the builder's own
        # words -- but the SEGMENTER it reports is the configured one either way, which is the fix.
        self.assertEqual(out.segmenter, "oneformer")
        self.assertIn(factory.REFUSE_NAMED_ONEFORMER_WITHOUT_BLOCK, out.detail)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
