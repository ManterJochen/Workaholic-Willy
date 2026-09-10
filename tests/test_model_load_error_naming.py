"""Unusable weights say WHICH model, in every way weights are actually unusable.

The five model wrappers each wrap their ``from_pretrained`` pair in a handler whose only job is to
name the model before re-raising: it is the one log line that connects a transformers traceback to
the config key that pointed there. It caught ``RuntimeError``.

MEASURED 2026-09-10 on this box (transformers 5.5.4, torch
2.7.1+cu128, safetensors 0.8.0), driving the REAL ``from_pretrained`` against three throwaway
directories built from the local HF cache:

    a directory holding only a README   -> ValueError  (GroundingDINO), OSError (SAM2)
    a half-downloaded snapshot          -> OSError, both
    a corrupted model.safetensors       -> safetensors.SafetensorError
                                           "Error while deserializing header: header too large"

Not one of the three is a RuntimeError, so the handler was dead in exactly the situations it was
written for: the operator got a traceback naming a cache path and never the config key.

These tests do not need the weights. They raise the three measured types out of a patched
``from_pretrained`` and check the handler still names the source. The AttributeError case is the
other half of the contract: widening to ``except Exception`` would label a defect in our own code
"failed to load weights", which is a worse lie than the silence it replaces, so a defect must pass
through UNLABELLED.
"""

from __future__ import annotations

import importlib
import unittest
from unittest.mock import MagicMock, patch


def _vision_config() -> MagicMock:
    cfg = MagicMock()
    cfg.local = False
    cfg.model_path = ""
    cfg.model_id = "org/model"
    cfg.threshold = 0.4
    cfg.task = "panoptic"
    cfg.optim = MagicMock(
        torch_dtype="auto", attn_implementation="eager", channels_last=True,
        compile=False, compile_mode="reduce-overhead",
    )
    return cfg


def _speech_config() -> MagicMock:
    cfg = _vision_config()
    cfg.samplerate = 16000
    cfg.blocksize = 1024
    cfg.channels = 1
    cfg.dtype = "float32"
    cfg.chunk_duration = 1.0
    cfg.language = "en"
    cfg.task = "transcribe"
    return cfg


#: (module, wrapper class, the two module-level from_pretrained holders, config builder).
_LOADERS = (
    (
        "src.models.detection.zero_shot.detector",
        "GroundingDinoObjectDetector",
        ("AutoProcessor", "AutoModelForZeroShotObjectDetection"),
        _vision_config,
    ),
    (
        "src.models.detection.closed_set.detector",
        "RtDetrObjectDetector",
        ("AutoImageProcessor", "AutoModelForObjectDetection"),
        _vision_config,
    ),
    (
        "src.models.segmentation.realtime.segmenter",
        "Sam2Segmenter",
        ("Sam2Processor", "Sam2Model"),
        _vision_config,
    ),
    (
        "src.models.segmentation.research.segmenter",
        "OneFormerSegmenter",
        ("OneFormerProcessor", "OneFormerForUniversalSegmentation"),
        _vision_config,
    ),
    (
        "src.models.speech.speech_to_text",
        "WhisperSpeechToText",
        ("WhisperProcessor", "WhisperForConditionalGeneration"),
        _speech_config,
    ),
)


def _corrupt_safetensors_error() -> BaseException:
    """The real class from the real library. A stand-in subclass would prove nothing."""
    import safetensors

    return safetensors.SafetensorError("Error while deserializing header: header too large")


def _build_with_load_failure(
    module_name: str, class_name: str, holders: tuple[str, str], config: MagicMock,
    failure: BaseException,
) -> MagicMock:
    """Construct the wrapper with ``from_pretrained`` raising ``failure``; return its logger mock."""
    module = importlib.import_module(module_name)
    logger = MagicMock()
    processor, model = holders
    with patch.object(module, "create_logger", return_value=logger), \
            patch.object(module, processor) as processor_mock, \
            patch.object(module, model) as model_mock:
        processor_mock.from_pretrained.side_effect = failure
        model_mock.from_pretrained.side_effect = failure
        getattr(module, class_name)(config)
    return logger


class UnusableWeightsAreNamedTests(unittest.TestCase):

    def _failures(self) -> tuple[tuple[str, BaseException], ...]:
        return (
            # README-only directory, GroundingDINO half of the pair.
            ("readme-only", ValueError("Unrecognized model in ...")),
            # README-only directory (SAM2) and half-downloaded snapshot (both), same type.
            ("no weights file", OSError("Error no file named model.safetensors, ... found in directory")),
            ("corrupt safetensors", _corrupt_safetensors_error()),
        )

    def test_every_loader_names_the_model_for_every_measured_failure(self) -> None:
        for module_name, class_name, holders, build_config in _LOADERS:
            for tag, failure in self._failures():
                with self.subTest(module=module_name, failure=tag):
                    with self.assertRaises(type(failure)) as caught:
                        _build_with_load_failure(
                            module_name, class_name, holders, build_config(), failure,
                        )
                    self.assertIs(caught.exception, failure, "the handler must NAME, never swallow")

    def test_the_log_line_carries_the_source(self) -> None:
        # Without this the operator gets a cache path and not the config key that pointed there.
        for module_name, class_name, holders, build_config in _LOADERS:
            for tag, failure in self._failures():
                with self.subTest(module=module_name, failure=tag):
                    config = build_config()
                    logger = MagicMock()
                    module = importlib.import_module(module_name)
                    processor, model = holders
                    with patch.object(module, "create_logger", return_value=logger), \
                            patch.object(module, processor) as processor_mock, \
                            patch.object(module, model) as model_mock:
                        processor_mock.from_pretrained.side_effect = failure
                        model_mock.from_pretrained.side_effect = failure
                        with self.assertRaises(type(failure)):
                            getattr(module, class_name)(config)
                    self.assertTrue(logger.error.called, "no line named the failing model")
                    spoken = " ".join(str(arg) for call in logger.error.call_args_list for arg in call.args)
                    self.assertIn("org/model", spoken)

    def test_a_defect_in_our_own_code_is_not_labelled_a_load_failure(self) -> None:
        # `except Exception` would pass this file's other tests and still be wrong: an AttributeError
        # from our own code would be reported to the operator as unusable weights.
        defect = AttributeError("MagicMock object has no attribute 'not_a_real_field'")
        for module_name, class_name, holders, build_config in _LOADERS:
            with self.subTest(module=module_name):
                config = build_config()
                logger = MagicMock()
                module = importlib.import_module(module_name)
                processor, model = holders
                with patch.object(module, "create_logger", return_value=logger), \
                        patch.object(module, processor) as processor_mock, \
                        patch.object(module, model) as model_mock:
                    processor_mock.from_pretrained.side_effect = defect
                    model_mock.from_pretrained.side_effect = defect
                    with self.assertRaises(AttributeError):
                        getattr(module, class_name)(config)
                self.assertFalse(
                    logger.error.called,
                    "a defect in our own code was reported as a weights-loading failure",
                )


class WeightLoadErrorsTests(unittest.TestCase):

    def test_it_resolves_the_real_safetensors_class(self) -> None:
        import safetensors

        from src.models._inference import weight_load_errors

        # The real class from the real library: a stand-in subclass would prove nothing.
        self.assertIn(safetensors.SafetensorError, weight_load_errors())

    def test_safetensors_is_not_imported_at_module_level(self) -> None:
        # SafetensorError lives in a compiled extension, and this repository lazy-imports heavy
        # vendor libraries inside the function that needs them. Checked structurally rather than via
        # sys.modules, because torch pulls safetensors in anyway and would hide a top-level import.
        import ast
        import inspect

        from src.models import _inference

        tree = ast.parse(inspect.getsource(_inference))
        top_level = [node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)]
        named = {
            alias.name.split(".")[0] for node in top_level if isinstance(node, ast.Import)
            for alias in node.names
        }
        named |= {
            (node.module or "").split(".")[0] for node in top_level if isinstance(node, ast.ImportFrom)
        }
        self.assertNotIn("safetensors", named)

    def test_it_keeps_runtimeerror_and_refuses_to_be_a_catch_all(self) -> None:
        from src.models._inference import weight_load_errors

        errors = weight_load_errors()
        # A CUDA OOM on the trailing `.to(device)` is still a load failure worth naming.
        self.assertIn(RuntimeError, errors)
        self.assertIn(OSError, errors)
        self.assertIn(ValueError, errors)
        self.assertNotIn(Exception, errors)
        self.assertNotIn(BaseException, errors)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
