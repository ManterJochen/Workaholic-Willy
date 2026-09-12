"""Whisper transcribes and never translates, and the spoken language is German or English, detected.

The owner decided on 2026-09-11 that text stays in the language it was spoken in: German stays German,
English stays English, and under `auto` the engine picks between exactly those two from Whisper's
language scores. The shipped tree said `task: translate` and `language: german`, so a German command
reached the prompt box as English text and an English command was decoded as if it were German.

transformers 5.5.4 ignores `forced_decoder_ids` whenever `task` or `language` is passed to `generate`,
and warns that the old flag is deprecated in favour of those two
(`transformers/models/whisper/generation_whisper.py:1491-1505`). So the decoder is asked through the
two supported arguments. Left without a language, `generate` would detect among every language the
checkpoint knows, so under `auto` it is always told German or English.

No weights are needed: the two `from_pretrained` holders are faked and the test reads what `generate`
was asked for.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml
from pydantic import ValidationError

from src.config.loader import ConfigError, load_config, reload_config
from src.config.schema.models.models_schema import SpeechToTextConfig

_SHIPPED = Path(__file__).resolve().parents[1] / "config"


def _shipped_stt() -> dict[str, Any]:
    """The `stt` block exactly as the shipped tree writes it."""
    text = (_SHIPPED / "models" / "stt.yaml").read_text(encoding="utf-8")
    block = yaml.safe_load(text)["stt"]
    assert isinstance(block, dict)
    return block


def _generate_kwargs(language: str) -> dict[str, Any]:
    """The keyword arguments `generate` receives for one decode, through the engine's YAML door.

    Written first against `WhisperSpeechToText`, where it was red; the wrapper was retired for
    `WhisperTransformersEngine` and only this helper moved, the assertions below did not."""
    import numpy as np
    import transformers

    from src.models.speech.whisper_transformers import WhisperTransformersEngine
    from tests._speech_fakes import whisper_parts

    processor, model = whisper_parts()
    with tempfile.TemporaryDirectory() as weights:
        config = SpeechToTextConfig.model_validate(
            {**_shipped_stt(), "language": language, "task": "transcribe", "model_path": weights}
        )
        with patch.object(transformers.WhisperProcessor, "from_pretrained", return_value=processor), \
                patch.object(transformers.WhisperForConditionalGeneration, "from_pretrained",
                             return_value=model):
            engine = WhisperTransformersEngine.from_config(config=config, device="cpu")
            engine.transcribe(np.zeros(1600, dtype=np.float32), samplerate=16000)
    assert model.generate.call_count == 1
    return dict(model.generate.call_args.kwargs)


class TheTaskIsTranscribeTests(unittest.TestCase):
    def test_a_block_that_says_translate_is_refused_with_a_sentence_naming_the_key(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            SpeechToTextConfig.model_validate({**_shipped_stt(), "task": "translate"})
        message = str(caught.exception)
        self.assertIn("models.stt.task", message, "the refusal must name the key")
        self.assertIn("transcribe", message, "and say what to write instead")

    def test_a_tree_that_says_translate_is_refused_by_the_loader(self) -> None:
        """The operator meets this through the loader, so the sentence has to survive it."""
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        root = tmp / "data"
        shutil.copytree(_SHIPPED, root)
        stt = root / "models" / "stt.yaml"
        text, hits = re.subn(
            r'(?m)^(\s*)task:.*$', r'\g<1>task: "translate"', stt.read_text(encoding="utf-8")
        )
        self.assertEqual(hits, 1, "the shipped stt.yaml stopped writing a task")
        stt.write_text(text, encoding="utf-8")
        reload_config()
        self.addCleanup(reload_config)
        with self.assertRaises(ConfigError) as caught:
            load_config(root, profile=None)
        message = str(caught.exception)
        self.assertIn("models.stt.task", message)
        self.assertIn("stt.yaml", message, "and point at the file the key was written in")

    def test_the_shipped_tree_transcribes_and_detects_the_language(self) -> None:
        reload_config()
        self.addCleanup(reload_config)
        stt = load_config(_SHIPPED, profile=None).models.stt
        self.assertEqual((stt.task, stt.language), ("transcribe", "auto"))


class TheLanguageIsTypedTests(unittest.TestCase):
    def test_auto_german_and_english_are_accepted(self) -> None:
        for language in ("auto", "german", "english"):
            with self.subTest(language=language):
                config = SpeechToTextConfig.model_validate({**_shipped_stt(), "language": language})
                self.assertEqual(config.language, language)

    def test_any_other_value_is_refused_with_a_sentence_naming_the_key(self) -> None:
        """A free string let a typo through to Whisper, which raised at the first transcription."""
        for language in ("klingon", "de", "German", ""):
            with self.subTest(language=language):
                with self.assertRaises(ValidationError) as caught:
                    SpeechToTextConfig.model_validate({**_shipped_stt(), "language": language})
                message = str(caught.exception)
                self.assertIn("models.stt.language", message)
                self.assertIn("auto", message)


class TheDecoderIsAskedToTranscribeTests(unittest.TestCase):
    def test_auto_hands_the_decoder_german_or_english_and_never_leaves_it_open(self) -> None:
        kwargs = _generate_kwargs("auto")
        self.assertNotIn("forced_decoder_ids", kwargs, "deprecated in transformers 5.5.4")
        self.assertEqual(kwargs.get("task"), "transcribe")
        self.assertIn(kwargs.get("language"), ("de", "en"), "generate was left to detect any language")

    def test_a_named_language_forces_exactly_that_language(self) -> None:
        for language in ("german", "english"):
            with self.subTest(language=language):
                kwargs = _generate_kwargs(language)
                self.assertNotIn("forced_decoder_ids", kwargs)
                self.assertEqual(kwargs.get("task"), "transcribe")
                self.assertEqual(kwargs.get("language"), language)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
