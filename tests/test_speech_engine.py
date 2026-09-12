"""One speech engine behind a protocol: loaded once, never at import, and answering with a report.

The console loaded Whisper from disk on every transcription and handed back a bare string, so nobody
could see which language was heard, how long the recording was, or what the decode cost against the
2 s budget from the end of speaking. These tests pin the engine that replaces it,
`WhisperTransformersEngine`, the main engine behind `SpeechEngine` since the owner's decision of
2026-09-11, running openai/whisper-large-v3-turbo. faster-whisper on CTranslate2 is a later bake-off.

Under `language: auto` the language is German or English and nothing else (owner decision, the same
day): the encoder runs once, one decoder step scores the language tokens, and whichever of `<|de|>` and
`<|en|>` scores higher is handed to `generate` with the same encoder pass.

No test here needs weights. `TheShippedWeightsTests` runs the real checkpoint when it is on disk and
says why it skipped when it is not.
"""

from __future__ import annotations

import builtins
import dataclasses
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import yaml

from src.config.schema.models.models_schema import SpeechToTextConfig
from src.contracts import Rendered, Structured
from src.utility.paths import weights_root
from tests._speech_fakes import (
    END,
    ENGLISH,
    FRENCH,
    GERMAN,
    NO_TIMESTAMPS,
    START,
    TRANSCRIBE,
    whisper_parts,
)

_REPO = Path(__file__).resolve().parents[1]
_SHIPPED_STT = _REPO / "config" / "models" / "stt.yaml"
sys.path.insert(0, str(_REPO))


def _config(**overrides: Any) -> SpeechToTextConfig:
    block = yaml.safe_load(_SHIPPED_STT.read_text(encoding="utf-8"))["stt"]
    return SpeechToTextConfig.model_validate({**block, **overrides})


def _transcript(**overrides: Any) -> Any:
    from src.models.speech.transcript import LanguageSource, Transcript

    fields: dict[str, Any] = dict(
        text="Nimm den roten Würfel",
        language="de",
        language_source=LanguageSource.DETECTED,
        duration_s=1.84,
        engine="whisper-transformers",
        model="assets/models/hf/speech/openai--whisper-large-v3-turbo",
        device="cuda",
        latency_ms=412.0,
    )
    fields.update(overrides)
    return Transcript(**fields)


def _engine(
    *,
    language: str = "auto",
    text: str = "  Pick the red cube  ",
    token: int = GERMAN,
    scores: dict[int, float] | None = None,
) -> tuple[Any, MagicMock, MagicMock]:
    from src.models.speech.whisper_transformers import WhisperTransformersEngine

    processor, model = whisper_parts(text=text, language_token=token, language_scores=scores)
    engine = WhisperTransformersEngine.from_parts(
        source="org/whisper", language=language, device="cpu", processor=processor, model=model,
    )
    return engine, processor, model


def _silence(seconds: float = 1.0, samplerate: int = 16000) -> np.ndarray:
    return np.zeros(int(seconds * samplerate), dtype=np.float32)


class TranscriptReportTests(unittest.TestCase):
    def test_it_is_frozen(self) -> None:
        transcript = _transcript()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            transcript.text = "something else"

    def test_to_dict_is_plain_data_that_survives_json(self) -> None:
        data = _transcript().to_dict()
        self.assertEqual(json.loads(json.dumps(data)), data)
        self.assertEqual(
            set(data),
            {"text", "language", "language_source", "duration_s", "engine", "model", "device",
             "latency_ms"},
        )
        self.assertIs(type(data["language_source"]), str, "a StrEnum member is not plain data")
        self.assertEqual(data["text"], "Nimm den roten Würfel", "the wire half keeps the words")

    def test_render_is_ascii_whole_and_ends_without_a_newline(self) -> None:
        text = _transcript().render()
        self.assertTrue(text.isascii(), text)
        self.assertFalse(text.endswith("\n"))
        for fragment in ("W\\xfcrfel", "(de, detected)", "1.84 s", "412 ms", "whisper-transformers",
                         "cuda", "openai--whisper-large-v3-turbo"):
            self.assertIn(fragment, text)

    def test_a_transcript_whose_output_carried_no_language_says_so(self) -> None:
        self.assertIn("language unknown", _transcript(language=None).render())

    def test_it_is_both_halves_of_the_report_contract(self) -> None:
        transcript = _transcript()
        self.assertIsInstance(transcript, Rendered)
        self.assertIsInstance(transcript, Structured)


class TheShippedSectionTests(unittest.TestCase):
    def test_the_shipped_tree_runs_whisper_large_v3_turbo_at_its_catalogue_pin(self) -> None:
        """Owner decision 2026-09-11: the transformers path is the main engine, with large-v3-turbo."""
        from scripts.model_weights.fetch import CATALOGUE

        stt = _config()
        self.assertEqual(stt.model_id, "openai/whisper-large-v3-turbo")
        entry = next(w for w in CATALOGUE if w.repo_id == stt.model_id)
        self.assertEqual(entry.revision, "41f01f3fe87f28c78e2fbf8b568835947dd65ed9")


class TheEngineIsLoadedOnceTests(unittest.TestCase):
    def _weights_dir(self) -> str:
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        return str(folder)

    def test_importing_the_speech_package_imports_no_heavy_library(self) -> None:
        """Weights and their libraries load on first use, never at import: a console without the speech
        stack must still start, and say what is missing when someone asks it to transcribe."""
        probe = (
            "import sys\n"
            "import src.models.speech\n"
            "import src.models.speech.capture\n"
            "import src.models.speech.endpointing\n"
            "import src.models.speech.engine\n"
            "import src.models.speech.gate\n"
            "import src.models.speech.holder\n"
            "import src.models.speech.listener\n"
            "import src.models.speech.silero\n"
            "import src.models.speech.transcript\n"
            "import src.models.speech.whisper_transformers\n"
            "heavy = ('torch', 'transformers', 'tokenizers', 'safetensors', 'sounddevice', 'scipy')\n"
            "print(sorted(name for name in heavy if name in sys.modules))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], cwd=_REPO, capture_output=True, text=True, timeout=300,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "[]")

    def test_from_config_is_keyword_only_and_loads_nothing(self) -> None:
        import transformers

        from src.models.speech.whisper_transformers import WhisperTransformersEngine

        with patch.object(transformers.WhisperProcessor, "from_pretrained") as processors, \
                patch.object(transformers.WhisperForConditionalGeneration, "from_pretrained") as models:
            WhisperTransformersEngine.from_config(config=_config())
            with self.assertRaises(TypeError):
                WhisperTransformersEngine.from_config(_config())  # type: ignore[misc]
        processors.assert_not_called()
        models.assert_not_called()

    def test_the_weights_load_once_across_an_explicit_load_and_two_transcriptions(self) -> None:
        import os

        import transformers

        from src.models.speech.whisper_transformers import WhisperTransformersEngine

        config = _config(model_path=self._weights_dir())
        processor, model = whisper_parts()
        with patch.dict(os.environ, {"WILLY_DEVICE": "cpu"}), \
                patch.object(transformers.WhisperProcessor, "from_pretrained",
                             return_value=processor) as processors, \
                patch.object(transformers.WhisperForConditionalGeneration, "from_pretrained",
                             return_value=model) as models:
            engine = WhisperTransformersEngine.from_config(config=config)
            engine.load()
            engine.load()
            for _ in range(2):
                engine.transcribe(_silence(), samplerate=16000)
        self.assertEqual((processors.call_count, models.call_count), (1, 1))
        self.assertEqual(processors.call_args.args[0], config.model_path, "local: true reads model_path")
        self.assertTrue(models.call_args.kwargs.get("local_files_only"), "and never downloads")

    def test_a_local_directory_that_is_not_there_is_refused_naming_it(self) -> None:
        """Left to `from_pretrained`, a missing directory is read as a Hub id and refused as one, which
        sends an operator looking for a Hub problem instead of a fetch that never ran."""
        import transformers

        from src.models.speech.engine import SpeechModelMissing
        from src.models.speech.whisper_transformers import WhisperTransformersEngine

        absent = Path(tempfile.gettempdir()) / "no-such-whisper-for-willy"
        engine = WhisperTransformersEngine.from_parts(source=str(absent), device="cpu")
        with patch.object(transformers.WhisperProcessor, "from_pretrained") as processors:
            with self.assertRaises(SpeechModelMissing) as caught:
                engine.load()
        processors.assert_not_called()
        self.assertIsInstance(caught.exception, FileNotFoundError)
        for fragment in ("no-such-whisper-for-willy", "models.stt.model_path", "fetch.py"):
            self.assertIn(fragment, str(caught.exception))


class TheEngineTranscribesTests(unittest.TestCase):
    def test_under_auto_generate_is_given_german_or_english_whichever_scores_higher(self) -> None:
        cases = (({GERMAN: 3.0, ENGLISH: 1.0}, "de"), ({GERMAN: 1.0, ENGLISH: 3.0, FRENCH: 9.0}, "en"))
        for scores, expected in cases:
            with self.subTest(expected=expected):
                engine, _, model = _engine(language="auto", scores=scores)
                engine.transcribe(_silence(), samplerate=16000)
                kwargs = model.generate.call_args.kwargs
                self.assertEqual(kwargs.get("language"), expected)
                self.assertEqual(kwargs.get("task"), "transcribe")
                self.assertNotIn("forced_decoder_ids", kwargs, "deprecated in transformers 5.5.4")
                self.assertTrue(kwargs.get("return_dict_in_generate"))
                self.assertIs(kwargs.get("encoder_outputs"), model.get_encoder.return_value.return_value,
                              "generate reuses the encoder pass the language was scored on")
                self.assertEqual(model.get_encoder.return_value.call_count, 1)

    def test_the_detected_language_is_read_from_the_decoder_prompt(self) -> None:
        for token, code in ((GERMAN, "de"), (ENGLISH, "en")):
            with self.subTest(language=code):
                engine, _, _ = _engine(language="auto", token=token)
                transcript = engine.transcribe(_silence(), samplerate=16000)
                self.assertEqual(
                    (transcript.language, transcript.language_source.value), (code, "detected")
                )

    def test_a_named_language_is_forced_and_reported_as_configured(self) -> None:
        engine, _, model = _engine(language="english", token=ENGLISH)
        transcript = engine.transcribe(_silence(), samplerate=16000)
        self.assertEqual(model.generate.call_args.kwargs.get("language"), "english")
        self.assertEqual((transcript.language, transcript.language_source.value), ("en", "configured"))
        model.assert_not_called()

    def test_the_text_is_stripped_and_never_lower_cased(self) -> None:
        engine, _, _ = _engine(text="  Nimm den ROTEN Würfel  ")
        transcript = engine.transcribe(_silence(), samplerate=16000)
        self.assertEqual(transcript.text, "Nimm den ROTEN Würfel")

    def test_the_report_carries_duration_engine_model_device_and_latency(self) -> None:
        engine, _, _ = _engine()
        transcript = engine.transcribe(_silence(2.0), samplerate=16000)
        self.assertEqual(transcript.duration_s, 2.0)
        self.assertEqual(
            (transcript.engine, transcript.model, transcript.device),
            ("whisper-transformers", "org/whisper", "cpu"),
        )
        self.assertGreaterEqual(transcript.latency_ms, 0.0)
        self.assertEqual(engine.name, transcript.engine)

    def test_stereo_is_mixed_and_another_rate_is_resampled_before_the_features(self) -> None:
        engine, processor, _ = _engine()
        stereo = np.zeros((48000, 2), dtype=np.float32)
        stereo[:, 0] = 0.5
        transcript = engine.transcribe(stereo, samplerate=48000)
        audio = processor.call_args.args[0]
        self.assertEqual(audio.ndim, 1, "Whisper takes mono")
        self.assertEqual(len(audio), 16000, "one second at the model's rate")
        self.assertEqual(processor.call_args.kwargs.get("sampling_rate"), 16000)
        self.assertAlmostEqual(float(np.median(audio)), 0.25, places=3, msg="mixed, not one side")
        self.assertEqual(transcript.duration_s, 1.0)

    def test_a_checkpoints_own_forced_tokens_cannot_decide_the_language_under_auto(self) -> None:
        engine, _, model = _engine(language="auto")
        model.generation_config.language = "<|en|>"
        engine.load()
        self.assertIsNone(model.generation_config.language)
        self.assertIsNone(model.generation_config.forced_decoder_ids)

    def test_a_checkpoint_without_a_language_table_is_refused_with_a_sentence(self) -> None:
        engine, _, model = _engine()
        model.generation_config.lang_to_id = None
        with self.assertRaises(ValueError) as caught:
            engine.load()
        self.assertIn("org/whisper", str(caught.exception))
        self.assertIn("lang_to_id", str(caught.exception))

    def test_a_checkpoint_without_german_and_english_is_refused_at_load(self) -> None:
        engine, _, model = _engine()
        model.generation_config.lang_to_id = {"<|fr|>": FRENCH}
        with self.assertRaises(ValueError) as caught:
            engine.load()
        for fragment in ("org/whisper", "<|de|>", "<|en|>"):
            self.assertIn(fragment, str(caught.exception))

    def test_a_recording_longer_than_the_30_s_window_is_refused_with_a_sentence(self) -> None:
        from src.models.speech.engine import RecordingTooLong

        engine, _, model = _engine()
        with self.assertRaises(RecordingTooLong) as caught:
            engine.transcribe(_silence(30.5), samplerate=16000)
        self.assertIn("30 s", str(caught.exception))
        model.generate.assert_not_called()

    def test_exactly_30_s_is_decoded(self) -> None:
        engine, _, model = _engine()
        engine.transcribe(_silence(30.0), samplerate=16000)
        self.assertEqual(model.generate.call_count, 1)


class TheEngineNamesUnusableWeightsTests(unittest.TestCase):
    """Unusable weights arrive as ValueError, OSError or safetensors.SafetensorError (measured on the
    dev tree on 2026-09-10), and a CUDA out-of-memory on the trailing `.to(device)` as RuntimeError. The
    handler names the model and re-raises; a defect in our own code must pass through unlabelled."""

    def _load(self, failure: BaseException, logger: MagicMock) -> None:
        import os

        import transformers

        from src.models.speech import whisper_transformers as module

        with patch.dict(os.environ, {"WILLY_DEVICE": "cpu"}), \
                patch.object(module, "create_logger", return_value=logger), \
                patch.object(transformers.WhisperProcessor, "from_pretrained", side_effect=failure), \
                patch.object(transformers.WhisperForConditionalGeneration, "from_pretrained",
                             side_effect=failure):
            module.WhisperTransformersEngine.from_parts(source="org/model", local_files_only=False).load()

    def test_every_measured_failure_is_re_raised_and_names_the_model(self) -> None:
        import safetensors

        failures: tuple[BaseException, ...] = (
            ValueError("Unrecognized model in ..."),
            OSError("Error no file named model.safetensors found in directory"),
            safetensors.SafetensorError("Error while deserializing header: header too large"),
            RuntimeError("CUDA out of memory"),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                logger = MagicMock()
                with self.assertRaises(type(failure)) as caught:
                    self._load(failure, logger)
                self.assertIs(caught.exception, failure, "the handler must name, never swallow")
                spoken = " ".join(str(arg) for call in logger.error.call_args_list for arg in call.args)
                self.assertIn("org/model", spoken)

    def test_a_defect_in_our_own_code_is_not_labelled_a_load_failure(self) -> None:
        logger = MagicMock()
        with self.assertRaises(AttributeError):
            self._load(AttributeError("object has no attribute 'not_a_real_field'"), logger)
        self.assertFalse(logger.error.called, "a defect was reported as unusable weights")


class AStackThatCannotLoadIsNamedTests(unittest.TestCase):
    """A missing package is a ModuleNotFoundError; a DLL Windows refuses is an OSError, or an ImportError
    saying the DLL load failed. Each is a machine that cannot run the stack, and each is answered with
    what to do about it."""

    def _refused(self, package: str, failure: BaseException) -> Any:
        from src.models.speech.engine import SpeechStackUnavailable
        from src.models.speech.whisper_transformers import WhisperTransformersEngine

        real_import = builtins.__import__

        def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == package or name.startswith(package + "."):
                raise failure
            return real_import(name, *args, **kwargs)

        engine = WhisperTransformersEngine.from_parts(source="org/whisper", device="cpu")
        with patch.object(builtins, "__import__", blocked):
            with self.assertRaises(SpeechStackUnavailable) as caught:
                engine.load()
        self.assertIsInstance(caught.exception, ImportError, "an `except ImportError` catches it")
        self.assertEqual(caught.exception.package, package)
        self.assertIs(caught.exception.__cause__, failure)
        return caught.exception

    def test_a_missing_package_names_the_requirements_file_that_holds_it(self) -> None:
        cases = (
            ("torch", "requirements.txt"),
            ("tokenizers", "requirements.txt"),
            ("safetensors", "requirements.txt"),
            ("transformers", "requirements.txt"),
        )
        for package, requirements in cases:
            with self.subTest(package=package):
                missing = ModuleNotFoundError(f"No module named '{package}'", name=package)
                self.assertIn(requirements, str(self._refused(package, missing)))

    def test_a_dll_windows_refuses_says_the_package_is_installed_and_names_the_file(self) -> None:
        loading = 'Error loading "D:\\venv\\Lib\\site-packages\\torch\\lib\\{}" or one of its dependencies.'
        cases: tuple[tuple[str, BaseException, str], ...] = (
            ("torch", OSError("[WinError 4551] An Application Control policy has blocked this file. "
                              + loading.format("c10.dll")), "c10.dll"),
            ("torch", OSError("[WinError 4551] Eine Anwendungssteuerungsrichtlinie hat diese Datei "
                              "blockiert. " + loading.format("torch_cpu.dll")), "torch_cpu.dll"),
            ("tokenizers", ImportError("DLL load failed while importing tokenizers: Eine "
                                       "Anwendungssteuerungsrichtlinie hat diese Datei blockiert."),
             "tokenizers"),
            ("safetensors", OSError("An Application Control policy has blocked this file: "
                                    "_safetensors_rust.pyd"), "_safetensors_rust.pyd"),
        )
        for package, failure, named in cases:
            with self.subTest(package=package, named=named):
                message = str(self._refused(package, failure))
                self.assertIn(f"{package} is installed", message)
                self.assertIn("Windows refused", message)
                self.assertIn(named, message)

    def test_an_ordinary_os_error_is_not_called_a_windows_refusal(self) -> None:
        message = str(self._refused("torch", OSError("libgomp.so.1: cannot open shared object file")))
        self.assertNotIn("Windows refused", message)
        self.assertIn("requirements.txt", message)


class _FeaturesWithAStubTokenizer:
    """The real feature extractor, and a tokenizer that records what it was asked to decode."""

    def __init__(self) -> None:
        from transformers import WhisperFeatureExtractor

        self._features = WhisperFeatureExtractor()
        self.decoded: Any = None

    def __call__(self, audio: Any, **kwargs: Any) -> Any:
        return self._features(audio, **kwargs)

    def batch_decode(self, sequences: Any, skip_special_tokens: bool = False) -> list[str]:
        self.decoded = sequences
        return [" tiny "]


def _tiny_whisper(bias: dict[int, float] | None = None) -> Any:
    """A randomly initialised two-layer Whisper with the real prompt vocabulary. No weights, no download.

    ``bias`` is added to the logits of the named tokens at every decoder step, so a test decides which
    language the model itself prefers."""
    import torch
    from transformers import GenerationConfig, WhisperConfig, WhisperForConditionalGeneration

    torch.manual_seed(0)
    config = WhisperConfig(
        vocab_size=51865, num_mel_bins=80, d_model=16, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32,
        decoder_ffn_dim=32, max_source_positions=1500, max_target_positions=448,
        decoder_start_token_id=START, pad_token_id=END, bos_token_id=END, eos_token_id=END,
    )
    model = WhisperForConditionalGeneration(config).eval()
    model.generation_config = GenerationConfig(
        decoder_start_token_id=START, eos_token_id=END, pad_token_id=END, max_new_tokens=3,
        lang_to_id={"<|en|>": ENGLISH, "<|de|>": GERMAN, "<|fr|>": FRENCH},
        task_to_id={"translate": 50358, "transcribe": TRANSCRIBE},
        no_timestamps_token_id=NO_TIMESTAMPS, is_multilingual=True,
        forced_decoder_ids=[[1, None], [2, TRANSCRIBE]],
    )
    if bias:
        shift = torch.zeros(config.vocab_size)
        for token, value in bias.items():
            shift[token] = value
        model.proj_out.register_forward_hook(lambda _module, _inputs, output: output + shift)
    return model


class TheRealGenerateTests(unittest.TestCase):
    """transformers' own `generate`, on a randomly initialised Whisper.

    The fakes above prove what the engine ASKS for. This proves transformers 5.5.4 answers the way the
    engine reads it: the returned sequence starts with the decoder prompt, language token at index 1,
    and a French score that beats both German and English does not make the language French.
    """

    def _transcribe(self, language: str, model: Any) -> tuple[Any, _FeaturesWithAStubTokenizer]:
        from src.models.speech.whisper_transformers import WhisperTransformersEngine

        processor = _FeaturesWithAStubTokenizer()
        engine = WhisperTransformersEngine.from_parts(
            source="tiny-random-whisper", language=language, device="cpu",
            processor=processor, model=model,
        )
        return engine.transcribe(_silence(0.5), samplerate=16000), processor

    def test_auto_picks_german_or_english_even_when_another_language_scores_higher(self) -> None:
        cases = (({FRENCH: 100.0, GERMAN: 50.0}, GERMAN, "de"), ({FRENCH: 100.0, ENGLISH: 50.0}, ENGLISH, "en"))
        for bias, token, code in cases:
            with self.subTest(expected=code):
                transcript, processor = self._transcribe("auto", _tiny_whisper(bias))
                self.assertEqual(int(processor.decoded[0][1]), token)
                self.assertEqual((transcript.language, transcript.language_source.value), (code, "detected"))
                self.assertEqual(int(processor.decoded[0][2]), TRANSCRIBE, "transcribe, never translate")
                self.assertEqual(transcript.text, "tiny")

    def test_the_encoder_runs_once_per_transcription(self) -> None:
        model = _tiny_whisper({FRENCH: 100.0, GERMAN: 50.0})
        runs: list[int] = []
        model.model.encoder.register_forward_hook(lambda *_: runs.append(1))
        self._transcribe("auto", model)
        self.assertEqual(len(runs), 1, "the language and the words come from one encoder pass")

    def test_a_named_language_puts_exactly_that_token_in_the_prompt(self) -> None:
        for language, token, code in (("german", GERMAN, "de"), ("english", ENGLISH, "en")):
            with self.subTest(language=language):
                transcript, processor = self._transcribe(language, _tiny_whisper({FRENCH: 100.0}))
                self.assertEqual(int(processor.decoded[0][1]), token)
                self.assertEqual(int(processor.decoded[0][2]), TRANSCRIBE)
                self.assertEqual(
                    (transcript.language, transcript.language_source.value), (code, "configured")
                )


def _shipped_weights() -> Path | None:
    """The directory the catalogue puts `models.stt.model_id` in, under this box's weights root."""
    try:
        from scripts.model_weights.fetch import CATALOGUE
    except ImportError:
        return None
    model_id = yaml.safe_load(_SHIPPED_STT.read_text(encoding="utf-8"))["stt"]["model_id"]
    entry = next((w for w in CATALOGUE if w.repo_id == model_id), None)
    if entry is None:
        return None
    path = entry.local_dir(weights_root())
    return path if (path / "model.safetensors").is_file() else None


_SHIPPED_WEIGHTS = _shipped_weights()


@unittest.skipUnless(
    _SHIPPED_WEIGHTS is not None,
    "the weights models.stt names are not under this checkout's weights root; fetch them with "
    "`python scripts/model_weights/fetch.py whisper-turbo` to run this test",
)
class TheShippedWeightsTests(unittest.TestCase):
    def test_the_shipped_checkpoint_transcribes_a_second_of_silence(self) -> None:
        from src.models.speech.whisper_transformers import WhisperTransformersEngine

        engine = WhisperTransformersEngine.from_config(config=_config(model_path=str(_SHIPPED_WEIGHTS)))
        transcript = engine.transcribe(_silence(), samplerate=16000)
        self.assertIn(transcript.language, ("de", "en"))
        self.assertEqual(transcript.language_source.value, "detected")
        self.assertGreater(transcript.latency_ms, 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
