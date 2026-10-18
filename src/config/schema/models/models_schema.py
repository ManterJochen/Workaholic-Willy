"""Schemas for the ML and CV models: which model detects, cuts and transcribes.

A block names its model and nothing else (the owner, 2026-10-17): a ``model_id``, whose weights are read from the
one weights folder (``models.weights_root``, :mod:`src.models.weights`), and a choice only where a cell has one to
make: the closed-set detector's threshold and Whisper's language. How each model loads and runs (its dtype, its
attention, its memory format, the VLM's answer speed, the microphone's stream, the hand detectors' thresholds) is
fixed per model in code at the values the owner's cell ran, and a tree that still writes one of those keys is refused
at load with the sentence that says what became of it (``src/config/schema/_removed.py``).
"""

from __future__ import annotations

from typing import Annotated, Any, Final, Literal, get_args

from pydantic import AfterValidator, Field, StringConstraints, ValidationInfo, field_validator

from ...paths import CONFIG_FOLDER, resolve_config_path
from .._base import StrictModel

#: Where every model's weights are read from unless a tree says otherwise: the repository's own folder, which
#: ``scripts/model_weights/fetch.py`` writes and the repository's ``.gitignore`` keeps out of the index.
DEFAULT_WEIGHTS_ROOT: Final[str] = "${WILLY_PROJECT_ROOT}/assets/models/hf"

#: How a model id that names a folder relative to the tree begins. A Hugging Face id never does.
_RELATIVE: Final = ("./", "../", ".\\", "..\\")


def _model_id_in_the_tree(value: Any, info: ValidationInfo) -> Any:
    """A model id written as a folder relative to the tree (``./chess``) is read against the tree's folder, as every
    relative path in a tree is (:mod:`src.config.paths`); a Hugging Face id and an absolute folder stay as written."""
    if not isinstance(value, str) or not value.startswith(_RELATIVE):
        return value
    context = info.context if isinstance(info.context, dict) else {}
    return resolve_config_path(value, context.get(CONFIG_FOLDER))


#: A model's name: a Hugging Face id (``org/name``), whose weights are read from
#: ``<models.weights_root>/<kind>/<org>--<name>``, or a folder's path (a ``DetectorTraining`` out_dir): absolute,
#: from home (``~/...``), or relative to the tree's folder (``./...``).
ModelId = Annotated[str, StringConstraints(min_length=1), AfterValidator(_model_id_in_the_tree)]


class ObjectDetectorConfig(StrictModel):
    """The closed-set detector, RT-DETR: its weights and the score a box must clear.

    ``model_id`` is a Hugging Face id, read from the weights folder (``scripts/model_weights/fetch.py rtdetr`` writes
    the shipped one), or the folder a ``DetectorTraining`` run wrote, its ``out_dir``: the best epoch at the top, the
    last in ``last/``. ``threshold`` is the confidence a box must clear.
    """

    model_id: ModelId
    threshold: float


class SegmenterConfig(StrictModel):
    """SAM2, the segmenter of every stack: one box in, one mask out. ``model_id`` names its weights."""

    model_id: ModelId


#: Which language the decoder runs with. `auto` picks German or English per recording from Whisper's
#: language scores for exactly those two, which is how a cell with German and English operators runs; a
#: recording in any other language still comes out as one of the two. A named language forces that
#: language token instead. Only the two languages this stack is decided on are offered; widen this the
#: day another one is measured. Whisper transcribes and never translates, so the text stays in the
#: language it was spoken in and a German command reaches the prompt box as German.
SpeechLanguage = Literal["auto", "german", "english"]


class SpeechToTextConfig(StrictModel):
    """Speech to text: which Whisper checkpoint, and which language it decodes.

    The Silero voice detector is the weights catalogue's file under the weights folder, and the cell PC's microphone
    opens at the owner's cell's numbers (16 kHz, one channel of int16, 8000 frames a block): both fixed in
    ``src/models/speech``.
    """

    model_id: ModelId
    language: SpeechLanguage

    @field_validator("language", mode="before")
    @classmethod
    def _a_language_this_stack_is_decided_on(cls, value: object) -> object:
        """Refuse a free string with a sentence naming the three values that are offered."""
        if value not in get_args(SpeechLanguage):
            raise ValueError(
                f"models.stt.language: {value!r} is not offered. Write auto (German or English, picked "
                f"per recording), german or english."
            )
        return value


# ---------------------------------------------------------------------------
# The perception stack: the VLM for any prompt, or RT-DETR for fixed classes
# ---------------------------------------------------------------------------


class VlmConfig(StrictModel):
    """The vision-language model that grounds every free-text prompt and reads the console's commands.

    VRAM is the constraint on this block: the model shares a card with SAM2 and, in simulation, with Isaac's own
    renderer, so the checkpoint has to fit in what those two leave. Record the VRAM figure for a changed checkpoint in
    the README next to the choice. How it answers is fixed at the owner's cell's values (``src/models/vlm/qwen.py``):
    it loads at the first prompt that reaches it, ends an answer where its JSON closes, proposes ten tokens a pass
    from its own answer, and decodes with CUDA graphs on a card that takes them.
    """

    #: A released Qwen3-VL-Instruct checkpoint. The paper's Qwen3-VL-Seg (arXiv 2605.07141) publishes
    #: neither weights nor code, so this stack grounds with the released instruct model and cuts masks
    #: with SAM2 from the same bbox_2d JSON the paper's own decoder consumes.
    model_id: ModelId = "Qwen/Qwen3-VL-4B-Instruct-FP8"


class ZeroShotPipelineConfig(StrictModel):
    """Open-vocabulary perception: the VLM grounds any prompt, and SAM2 cuts each box."""

    vlm: VlmConfig = Field(default_factory=VlmConfig)


class PipelineConfig(StrictModel):
    """Which perception stack a cell builds: the VLM for free text, or RT-DETR's trained classes.

    ``zero_shot`` grounds the prompt with the VLM (``zero_shot.vlm``); ``closed_set`` answers with RT-DETR's class list
    (``models.rtdetr``, refused at build without it) and reads a prompt only as a class-name filter. SAM2 cuts the
    masks of both. The VLM block is read on a closed-set cell too: the console's command reader asks it.
    """

    kind: Literal["zero_shot", "closed_set"] = "zero_shot"
    zero_shot: ZeroShotPipelineConfig = Field(default_factory=ZeroShotPipelineConfig)
