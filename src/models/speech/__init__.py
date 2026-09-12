"""Speech to text: one engine behind a protocol, a voice gate, the microphone stream, and `listen()`.

The names below resolve on first use, so importing this package imports neither torch, transformers,
scipy nor sounddevice. A console without the speech stack must still start.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

#: Public name -> the module that defines it.
_EXPORTS: dict[str, str] = {
    "AudioBlock": "src.models.speech.capture",
    "AudioSource": "src.models.speech.capture",
    "CutUtterance": "src.models.speech.endpointing",
    "Endpointing": "src.models.speech.endpointing",
    "HeldSpeech": "src.models.speech.holder",
    "LanguageSource": "src.models.speech.transcript",
    "ListenOutcome": "src.models.speech.listener",
    "Listener": "src.models.speech.listener",
    "MicrophoneSource": "src.models.speech.capture",
    "Proposal": "src.models.speech.transcript",
    "RecordingTooLong": "src.models.speech.engine",
    "SileroVoiceActivityDetector": "src.models.speech.silero",
    "SpeechCheck": "src.models.speech.transcript",
    "SpeechEngine": "src.models.speech.engine",
    "SpeechGate": "src.models.speech.gate",
    "SpeechHolder": "src.models.speech.holder",
    "SpeechModelMissing": "src.models.speech.engine",
    "SpeechStackUnavailable": "src.models.speech.engine",
    "Transcript": "src.models.speech.transcript",
    "Utterance": "src.models.speech.listener",
    "UtteranceCutter": "src.models.speech.endpointing",
    "UtteranceEnd": "src.models.speech.endpointing",
    "VoiceActivityDetector": "src.models.speech.endpointing",
    "WHISPER_WINDOW_S": "src.models.speech.engine",
    "WhisperTransformersEngine": "src.models.speech.whisper_transformers",
    "code_integrity_refusal": "src.models.speech.engine",
    "requirements_for": "src.models.speech.engine",
    "shared_speech": "src.models.speech.holder",
    "to_mono_at_rate": "src.models.speech.capture",
    "to_mono_float32": "src.models.speech.capture",
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        value = getattr(import_module(_EXPORTS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_EXPORTS)
