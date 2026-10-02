"""The VLM route: a vision-language model used as the detector, then SAM2 for masks; and the command reader.

    from src.models.vlm import Qwen3VLGrounder
    from src.models.perception_backend import TwoStageBackend

    backend = TwoStageBackend(detector=Qwen3VLGrounder(model_id=...), segmenter=sam2)

One copy of the model per process (``shared_vlm()``) serves detection and the console's command reader
(``understand``, ``read_command``), which reads an operator's sentence into a task card and moves nothing.

Nothing here imports torch or transformers at module scope. See [README.md](README.md).
"""

from __future__ import annotations

from .availability import (
    CommandRefused,
    GuardedVlmBackend,
    ReaderAvailability,
    VlmAnswerFailedError,
    VlmCopyConflictError,
    VlmNotLoadedError,
    VlmUnavailableError,
    reader_availability,
    vlm_detects,
)
from .command import (
    COMMAND_INSTRUCTION,
    CommandAnswer,
    CommandReading,
    PhraseReading,
    read_command,
    understand,
)
from .holder import TextAsker, VlmHolder, VlmStatus, VlmWeights, shared_vlm
from .parsing import VLM_NOMINAL_SCORE, extract_json_payload, parse_grounding_response
from .qwen import GROUNDING_INSTRUCTION, Qwen3VLGrounder

__all__ = [
    "Qwen3VLGrounder",
    "GROUNDING_INSTRUCTION",
    "GuardedVlmBackend",
    "VlmUnavailableError",
    "parse_grounding_response",
    "extract_json_payload",
    "VLM_NOMINAL_SCORE",
    "VlmHolder",
    "VlmStatus",
    "VlmWeights",
    "TextAsker",
    "shared_vlm",
    "ReaderAvailability",
    "reader_availability",
    "VlmNotLoadedError",
    "VlmCopyConflictError",
    "VlmAnswerFailedError",
    "vlm_detects",
    "CommandRefused",
    "COMMAND_INSTRUCTION",
    "CommandAnswer",
    "CommandReading",
    "PhraseReading",
    "understand",
    "read_command",
]
