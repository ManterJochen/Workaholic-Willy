"""Silero VAD as TorchScript on the torch this interpreter already has, behind `VoiceActivityDetector`.

The owner decided on 2026-09-11 that Silero runs as TorchScript through the shared venv's torch, with no
new package. The model file is `silero_vad.jit` from the silero-vad 6.2.1 wheel (MIT), pinned by hash in
`scripts/model_weights/fetch.py`, and `models.stt.vad_model_path` names where it lies. Every test but the
last runs against a stand-in for what `torch.jit.load` returns; `TheModelOnThisBoxTests` loads the real
file when it is on disk and says why it skipped when it is not.
"""

from __future__ import annotations

import builtins
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import yaml

from src.config.schema.models.models_schema import SpeechToTextConfig
from src.utility.paths import weights_root
from tests._speech_fakes import FakeSileroModel, silence, tone

_REPO = Path(__file__).resolve().parents[1]
_SHIPPED_STT = _REPO / "config" / "models" / "stt.yaml"
sys.path.insert(0, str(_REPO))


def _config(**overrides: Any) -> SpeechToTextConfig:
    block = yaml.safe_load(_SHIPPED_STT.read_text(encoding="utf-8"))["stt"]
    return SpeechToTextConfig.model_validate({**block, **overrides})


def _on_box_model() -> Path | None:
    """The catalogue's Silero file under this box's weights root, when it is there."""
    try:
        from scripts.model_weights.fetch import PACKAGED_FILES
    except ImportError:
        return None
    entry = next((f for f in PACKAGED_FILES if f.key == "silero-vad"), None)
    if entry is None:
        return None
    path = entry.local_path(weights_root())
    return path if path.is_file() else None


class _WithAModelFile(unittest.TestCase):
    def _file(self) -> Path:
        """A file where the detector looks, so the missing-file refusal stays out of the way."""
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        path = folder / "silero_vad.jit"
        path.write_bytes(b"stand-in")
        return path


class TheDetectorTests(_WithAModelFile):
    def test_it_is_a_voice_activity_detector_at_16_khz_in_windows_of_512(self) -> None:
        from src.models.speech.endpointing import VoiceActivityDetector
        from src.models.speech.silero import SileroVoiceActivityDetector

        detector = SileroVoiceActivityDetector.from_parts(model_path=self._file())
        self.assertIsInstance(detector, VoiceActivityDetector)
        self.assertEqual((detector.samplerate, detector.window_samples), (16000, 512))

    def test_from_config_reads_vad_model_path_is_keyword_only_and_loads_nothing(self) -> None:
        import torch

        from src.models.speech.silero import SileroVoiceActivityDetector

        path = self._file()
        with patch.object(torch.jit, "load") as load:
            detector = SileroVoiceActivityDetector.from_config(config=_config(vad_model_path=str(path)))
            with self.assertRaises(TypeError):
                SileroVoiceActivityDetector.from_config(_config())  # type: ignore[misc]
        load.assert_not_called()
        self.assertEqual(detector.model_path, path)

    def test_the_model_loads_once_on_first_use_and_on_the_cpu(self) -> None:
        import torch

        from src.models.speech.silero import SileroVoiceActivityDetector

        path = self._file()
        with patch.object(torch.jit, "load", return_value=FakeSileroModel(speech=True)) as load:
            detector = SileroVoiceActivityDetector.from_parts(model_path=path)
            detector.reset()
            for _ in range(3):
                detector.speech_probability(np.zeros(512, dtype=np.float32))
        self.assertEqual(load.call_count, 1)
        self.assertEqual(Path(load.call_args.args[0]), path)
        self.assertEqual(str(load.call_args.kwargs.get("map_location")), "cpu")

    def test_a_window_is_scored_at_16_khz_as_a_plain_float(self) -> None:
        from src.models.speech.silero import SileroVoiceActivityDetector

        model = FakeSileroModel()
        detector = SileroVoiceActivityDetector.from_parts(model_path=self._file(), model=model)
        loud = detector.speech_probability(tone(512 / 16000))
        quiet = detector.speech_probability(np.zeros(512, dtype=np.float32))
        self.assertEqual((loud, quiet), (1.0, 0.0))
        self.assertIs(type(loud), float)
        self.assertEqual(model.calls[0], ((512,), 16000))

    def test_reset_forgets_the_state_carried_between_windows(self) -> None:
        from src.models.speech.silero import SileroVoiceActivityDetector

        model = FakeSileroModel()
        detector = SileroVoiceActivityDetector.from_parts(model_path=self._file(), model=model)
        detector.reset()
        detector.reset()
        self.assertEqual(model.resets, 2)

    def test_a_window_of_another_shape_is_refused_with_a_sentence(self) -> None:
        from src.models.speech.silero import SileroVoiceActivityDetector

        detector = SileroVoiceActivityDetector.from_parts(model_path=self._file(), model=FakeSileroModel())
        for window in (np.zeros(400, dtype=np.float32), np.zeros((512, 2), dtype=np.float32)):
            with self.subTest(shape=window.shape):
                with self.assertRaises(ValueError) as caught:
                    detector.speech_probability(window)
                self.assertIn("512", str(caught.exception))

    def test_a_missing_model_file_is_refused_naming_the_path_and_the_fetch(self) -> None:
        import torch

        from src.models.speech.engine import SpeechModelMissing
        from src.models.speech.silero import SileroVoiceActivityDetector

        absent = Path(tempfile.gettempdir()) / "no-such-folder-for-willy" / "silero_vad.jit"
        detector = SileroVoiceActivityDetector.from_parts(model_path=absent)
        with patch.object(torch.jit, "load") as load:
            with self.assertRaises(SpeechModelMissing) as caught:
                detector.speech_probability(np.zeros(512, dtype=np.float32))
        load.assert_not_called()
        self.assertIsInstance(caught.exception, FileNotFoundError)
        message = str(caught.exception)
        for fragment in ("no-such-folder-for-willy", "models.stt.vad_model_path", "fetch.py silero-vad"):
            self.assertIn(fragment, message)

    def test_a_machine_that_cannot_import_torch_gets_the_named_refusal(self) -> None:
        from src.models.speech.engine import SpeechStackUnavailable
        from src.models.speech.silero import SileroVoiceActivityDetector

        refused = OSError("[WinError 4551] An Application Control policy has blocked this file.")
        real_import = builtins.__import__

        def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "torch" or name.startswith("torch."):
                raise refused
            return real_import(name, *args, **kwargs)

        detector = SileroVoiceActivityDetector.from_parts(model_path=self._file())
        with patch.object(builtins, "__import__", blocked):
            with self.assertRaises(SpeechStackUnavailable) as caught:
                detector.reset()
        self.assertEqual(caught.exception.package, "torch")


_ON_BOX = _on_box_model()


@unittest.skipUnless(
    _ON_BOX is not None,
    "silero_vad.jit is not under this checkout's weights root; fetch it with "
    "`python scripts/model_weights/fetch.py silero-vad` to run this test",
)
class TheModelOnThisBoxTests(unittest.TestCase):
    def test_the_real_model_scores_silence_and_a_tone_below_the_onset(self) -> None:
        """Neither is speech. The probabilities it measured are in speech_README.md."""
        from src.models.speech.silero import SileroVoiceActivityDetector

        assert _ON_BOX is not None
        detector = SileroVoiceActivityDetector.from_parts(model_path=_ON_BOX)
        for name, audio in (("3 s of silence", silence(3.0)),
                            ("3 s of a 440 Hz tone", tone(3.0, frequency=440.0))):
            with self.subTest(audio=name):
                detector.reset()
                scores = [detector.speech_probability(audio[i:i + 512])
                          for i in range(0, audio.size - 511, 512)]
                self.assertTrue(all(0.0 <= score <= 1.0 for score in scores), name)
                self.assertLess(max(scores), 0.5, f"{name}: peak probability {max(scores):.4f}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
