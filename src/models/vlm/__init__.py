"""The VLM grounding route: a vision-language model used as the detector, then SAM2 for masks.

    from src.models.vlm import Qwen3VLGrounder
    from src.models.perception_backend import TwoStageBackend

    backend = TwoStageBackend(detector=Qwen3VLGrounder(model_id=...), segmenter=sam2)

Nothing here imports torch or transformers at module scope. See [README.md](README.md).
"""

from __future__ import annotations

from .availability import GuardedVlmBackend, VlmUnavailableError
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
]
