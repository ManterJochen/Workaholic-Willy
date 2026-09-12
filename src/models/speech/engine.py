"""The one seam every speech engine sits behind, and the refusals of a machine that cannot run one.

`SpeechEngine` is what the upload path and `Listener.listen()` both hold, so a recording uploaded from a
browser and an utterance cut from the cell's microphone go through the same weights and come back as the
same `Transcript`. `WhisperTransformersEngine` (`whisper_transformers.py`) is the main engine: Whisper
large-v3-turbo through transformers, on the torch this interpreter already has. faster-whisper on
CTranslate2 stays a later bake-off, because Smart App Control refuses CTranslate2's DLL on this
workstation and its wheel carries no kernels for the RTX 5080.

Three refusals live here because the engine, the voice detector and the upload path all raise them:

* `SpeechStackUnavailable`: a package this machine cannot import, answered with the requirements file
  that installs it, or, for a DLL Windows refused, with the file Windows refused.
* `SpeechModelMissing`: a model file or directory the config names that this machine has not fetched.
* `RecordingTooLong`: a recording longer than Whisper's 30 s window.

stdlib only at import.
"""

from __future__ import annotations

import re
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover, typing only
    import numpy as np

    from src.models.speech.transcript import Transcript

__all__ = [
    "WHISPER_WINDOW_S",
    "RecordingTooLong",
    "SpeechEngine",
    "SpeechModelMissing",
    "SpeechStackUnavailable",
    "code_integrity_refusal",
    "requirements_for",
]

#: The longest recording one Whisper window holds, in seconds. The engine decodes one window and reads
#: the language from its prompt; it does not stitch several, so a longer recording is refused.
WHISPER_WINDOW_S: Final[float] = 30.0


@runtime_checkable
class SpeechEngine(Protocol):
    """Turns a recording into a `Transcript`. Weights load on first use or in `load()`, never at import."""

    @property
    def name(self) -> str:
        """The engine's name as a `Transcript` reports it."""
        ...

    def load(self) -> None:
        """Load the weights now rather than on the first transcription. Calling it twice loads once."""
        ...

    def transcribe(self, samples: "np.ndarray", *, samplerate: int) -> "Transcript":
        """The most likely text for ``samples`` (frames, or frames by channels) recorded at ``samplerate``."""
        ...


#: Which requirements file installs each package the speech stack imports. One install covers all of
#: them, so what the entry adds is the CPU alternative and what each package brings with it.
_REQUIREMENTS: Final[dict[str, str]] = {
    "torch": "requirements.txt (requirements-cpu.txt on a machine without a GPU)",
    "transformers": "requirements.txt",
    "tokenizers": "requirements.txt, which installs it with transformers",
    "safetensors": "requirements.txt, which installs it with transformers",
    "scipy": "requirements.txt",
    "numpy": "requirements.txt",
    "sounddevice": "requirements.txt, and it loads the system PortAudio library",
}


def requirements_for(package: str | None) -> str:
    """The requirements file that installs ``package``, or the one that installs the whole stack."""
    if package:
        top = package.split(".")[0]
        if top in _REQUIREMENTS:
            return _REQUIREMENTS[top]
    return "requirements.txt (requirements-cpu.txt on a machine without a GPU)"


#: How a refused DLL reads. Windows code integrity refuses a load with WinError 4551, which torch raises as
#: an OSError, and Python's extension loader reports it as "DLL load failed while importing <module>:
#: <the policy sentence>". The sentence is localised, so the German wording is matched beside the English.
_REFUSAL_MARKERS: Final[tuple[str, ...]] = (
    "WinError 4551",
    "Application Control",
    "Anwendungssteuerungsrichtlinie",
)
_LOADING: Final[re.Pattern[str]] = re.compile(r'Error loading "([^"]+)"')
_BINARY: Final[re.Pattern[str]] = re.compile(r"([\w.+-]+\.(?:dll|pyd))", re.IGNORECASE)
_EXTENSION: Final[re.Pattern[str]] = re.compile(r"while importing ([\w.]+)")


def code_integrity_refusal(cause: BaseException) -> str | None:
    """The file Windows refused to load, when ``cause`` is a code integrity refusal; otherwise None.

    The file comes from the message where it names one (torch writes `Error loading "...\\c10.dll"`),
    then from the exception's own file name, then from the extension module Python was importing. When
    none of them is there, the answer points at the event log that names the file.
    """
    if not isinstance(cause, (OSError, ImportError)):
        return None
    text = str(cause)
    if getattr(cause, "winerror", None) != 4551 and not any(marker in text for marker in _REFUSAL_MARKERS):
        return None
    if (loading := _LOADING.search(text)) is not None:
        return PureWindowsPath(loading.group(1)).name
    if (binary := _BINARY.search(text)) is not None:
        return binary.group(1)
    filename = getattr(cause, "filename", None)
    if filename:
        return PureWindowsPath(str(filename)).name
    if (extension := _EXTENSION.search(text)) is not None:
        return f"the extension module {extension.group(1)}"
    return "the file named in Microsoft-Windows-CodeIntegrity/Operational"


class SpeechStackUnavailable(ImportError):
    """This machine cannot import a package the speech stack needs.

    An `ImportError`, so an ``except ImportError`` still catches it, and a type of its own because the
    cause is often not an ImportError at all: Smart App Control refuses a DLL with an `OSError`, and a
    missing PortAudio makes ``import sounddevice`` raise `OSError`. Each is a capability this host lacks,
    never a bad request. A missing package is answered with the requirements file that installs it; a
    DLL Windows refused with the sentence that the package is installed and which file was refused,
    because reinstalling does not help while the policy refuses it.
    """

    def __init__(self, *, package: str, cause: BaseException) -> None:
        self.package = package
        self.requirements = requirements_for(package)
        #: The file Windows refused, when that is why the import failed; None otherwise.
        self.refused_file = code_integrity_refusal(cause)
        if self.refused_file is not None:
            message = (
                f"the speech stack cannot load {package} on this machine: {package} is installed, and "
                f"Windows refused to load {self.refused_file} (Application Control, WinError 4551; "
                f"{type(cause).__name__}: {cause}). Reinstalling does not help while the policy refuses "
                f"that file; Microsoft-Windows-CodeIntegrity/Operational lists the refusal."
            )
        else:
            message = (
                f"the speech stack cannot import {package} on this machine "
                f"({type(cause).__name__}: {cause}). It is installed by {self.requirements}."
            )
        super().__init__(message, name=package)


class SpeechModelMissing(FileNotFoundError):
    """A model file or directory the config names is not on this machine.

    A capability this host lacks, never a bad request. Nothing downloads on its own (`models.stt.local:
    true` is the point of the flag), so the refusal names the key, the path and the fetch that fixes it.
    """

    def __init__(self, *, key: str, path: str, kind: str, fetch: str) -> None:
        self.key = key
        self.path = path
        super().__init__(
            f"{key} is {path!r}, but there is no such {kind} (resolved: {Path(path).resolve()}). "
            f"Nothing is downloaded on its own; {fetch}."
        )


class RecordingTooLong(ValueError):
    """A recording longer than one Whisper window, refused before any model runs.

    The engine decodes one 30 s window and reads the language from its decoder prompt. A longer recording
    would be cut into windows whose words and languages this stack does not stitch together.
    """

    def __init__(self, *, duration_s: float, limit_s: float = WHISPER_WINDOW_S) -> None:
        self.duration_s = duration_s
        self.limit_s = limit_s
        super().__init__(
            f"the recording is {duration_s:.1f} s long, and Whisper decodes one window of {limit_s:.0f} s; "
            f"record a command of at most {limit_s:.0f} s."
        )
