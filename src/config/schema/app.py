"""Root :class:`AppConfig` plus the camera/models composition.

Hierarchy::

    AppConfig
    +-- camera   : CameraConfig         (cameras + eye-to-hand)
    +-- models   : ModelsConfig         (ML/CV model configs)
    +-- robot    : RobotConfig | None   (optional robot tree)
    `-- runtime  : RuntimeConfig        (app-service tuning, see runtime.py)

All schemas inherit from :class:`StrictModel` (immutable, ``extra="forbid"``).
"""

from __future__ import annotations

from pydantic import Field

from ._base import ConfigPath, StrictModel
from .camera import (
    CameraSystemConfig,
    HandEyeConfig,
)
from .models import (
    DEFAULT_WEIGHTS_ROOT,
    ObjectDetectorConfig,
    PipelineConfig,
    SegmenterConfig,
    SpeechToTextConfig,
)
from .robot import RobotConfig
from .runtime import RuntimeConfig


class CameraConfig(StrictModel):
    """Camera section: rigs and hand-eye calibration.

    A stereo rig's matcher has no block of its own since 2026-10-17: its search is derived from its calibration and
    the robot's workspace, and the rest is fixed at measured values (``src/calibration/stereo/sub_modules``).
    """

    cameras: CameraSystemConfig
    hand_eye: HandEyeConfig = Field(default_factory=HandEyeConfig)


class ModelsConfig(StrictModel):
    """The ML and CV model blocks: the perception stack, the segmenter, speech, and where their weights are."""

    #: The folder every model's weights are read from, set once (the owner, 2026-10-17): a Hugging Face id is read
    #: from ``<weights_root>/<kind>/<org>--<name>``, the layout ``scripts/model_weights/fetch.py`` writes, and
    #: Silero and the two MediaPipe bundles from their fixed places under it (:mod:`src.models.weights`).
    weights_root: ConfigPath = Field(default=DEFAULT_WEIGHTS_ROOT, validate_default=True)
    segmenter: SegmenterConfig
    stt: SpeechToTextConfig
    #: The closed-set detector block, needed when ``pipeline.kind`` is ``closed_set``. Asking for that stack without
    #: this block refuses the build; there is no fallback to the VLM.
    rtdetr: ObjectDetectorConfig | None = None
    #: Which perception stack a cell builds: the VLM for free text (the default), or RT-DETR's trained classes.
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)


class PerceptionModelsConfig(StrictModel):
    """The four ``models`` keys the perception stack reads, loadable without the rest of ``models``.

    ``src.config.loader.load_perception_section`` returns this, so perception loads without the Whisper block. Field
    for field it carries :class:`ModelsConfig`'s annotations and defaults for the fields ``PerceptionSpec`` reads.
    """

    segmenter: SegmenterConfig
    rtdetr: ObjectDetectorConfig | None = None
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    weights_root: ConfigPath = Field(default=DEFAULT_WEIGHTS_ROOT, validate_default=True)


class WeightsRootSection(StrictModel):
    """``models.weights_root`` alone, as ``src.config.loader.load_weights_root`` reads it beside another section."""

    weights_root: ConfigPath = Field(default=DEFAULT_WEIGHTS_ROOT, validate_default=True)


class AppConfig(StrictModel):
    """Root configuration object returned by :func:`src.config.load_config`."""

    camera: CameraConfig
    models: ModelsConfig
    robot: RobotConfig | None = None
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

