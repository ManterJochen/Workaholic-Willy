"""Root :class:`AppConfig` plus the camera/models composition.

Hierarchy::

    AppConfig
    +-- camera   : CameraConfig         (cameras + matcher + eye-to-hand)
    +-- models   : ModelsConfig         (ML/CV model configs)
    +-- robot    : RobotConfig | None   (optional robot tree)
    `-- runtime  : RuntimeConfig        (app-service tuning, see runtime.py)

All schemas inherit from :class:`StrictModel` (immutable, ``extra="forbid"``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from ._base import StrictModel
from .camera import (
    CameraSystemConfig,
    HandEyeConfig,
    StereoMatcherConfig,
)
from .models import (
    GestureDetectConfig,
    HandDetectConfig,
    ObjectDetectorConfig,
    OneFormerConfig,
    PipelineConfig,
    SegmenterConfig,
    SpeechToTextConfig,
)
from .robot import RobotConfig
from .runtime import RuntimeConfig


class CameraConfig(StrictModel):
    """Camera section: rigs, stereo matcher and hand-eye calibration."""

    cameras: CameraSystemConfig
    stereomatcher: StereoMatcherConfig
    hand_eye: HandEyeConfig = Field(default_factory=HandEyeConfig)


class ModelsConfig(StrictModel):
    """The ML and CV model blocks, and the two ways of choosing a perception stack."""

    objectdetector: ObjectDetectorConfig
    segmenter: SegmenterConfig
    stt: SpeechToTextConfig
    #: Optional MediaPipe hand and gesture surface. Standalone: nothing in the grasp pipeline
    #: reads these. Their readers are `src.models.handdetection.factory` and that package's
    #: `python -m` CLI.
    handdetect: HandDetectConfig = Field(default_factory=HandDetectConfig)
    gesturedetect: GestureDetectConfig = Field(default_factory=GestureDetectConfig)
    # Perception backends chosen by hand: two independently settable keys with no cross-check, so
    # every detector x segmenter combination builds, including ones where the prompt means
    # something different to each half. Kept because assembling a stack by hand is legitimate;
    # `pipeline` below is the cross-checked way to choose for everyday use.
    detector: Literal["groundingdino", "rtdetr"] = "groundingdino"
    segmenter_backend: Literal["sam2", "oneformer"] = "sam2"
    #: The closed-set detector block, needed when ``detector`` is ``rtdetr``. Asking for that
    #: backend without this block refuses the build; there is no fallback to the default backend.
    rtdetr: ObjectDetectorConfig | None = None
    #: The OneFormer block, needed when ``segmenter_backend`` is ``oneformer``. Asking for that
    #: backend without this block refuses the build; there is no fallback to the default backend.
    oneformer: OneFormerConfig | None = None
    #: A perception stack chosen in one line, with the combinations validated fail-closed. Set,
    #: ``build_perception`` decides the stack from it and the two keys above go unread; a build
    #: through ``build_object_detector`` or ``build_segmenter`` reads those two keys whatever this
    #: block says. ``None``, the default, leaves them in force and behaviour is byte-identical to
    #: a build without this block.
    pipeline: PipelineConfig | None = None


class PerceptionModelsConfig(StrictModel):
    """The seven ``models`` keys the perception stack reads, loadable without the rest of ``models``.

    ``src.config.loader.load_perception_section`` returns this, so perception loads without the
    Whisper block or the hand detectors. Field for field it carries :class:`ModelsConfig`'s
    annotations and defaults for the fields ``PerceptionSpec`` reads. ``ModelsConfig`` itself is
    unchanged, so a loaded tree keeps its field order.
    """

    objectdetector: ObjectDetectorConfig
    segmenter: SegmenterConfig
    detector: Literal["groundingdino", "rtdetr"] = "groundingdino"
    segmenter_backend: Literal["sam2", "oneformer"] = "sam2"
    rtdetr: ObjectDetectorConfig | None = None
    oneformer: OneFormerConfig | None = None
    pipeline: PipelineConfig | None = None


def camera_calibration_conflict(camera: CameraConfig, robot: RobotConfig | None) -> str | None:
    """The refusal when ``robot.grasping.fusion.cameras`` names a camera that is not a rig, else None.

    A fused camera's calibration is declared on its rig, ``camera.cameras.rigs[<id>].extrinsics``, so
    an id in the map that names no rig is a camera whose calibration has nowhere to be. Whether that
    rig declares ``extrinsics`` is decided when the cell is built and in the preflight, not at load: a
    profile that is not ready to run is not a malformed file.

    The rule needs both sections, the rigs under ``camera`` and the map under ``robot``, so
    :class:`AppConfig` runs it for the whole tree. A section loader sees one half and cannot run it, so
    a door that combines a camera section with a robot section loaded apart calls this function.
    """
    fusion = getattr(getattr(robot, "grasping", None), "fusion", None)
    cameras = getattr(fusion, "cameras", None) or {}
    rigs = sorted(rig.rig_id for rig in camera.cameras.rigs)
    for cam_id in sorted(cameras):
        if cam_id not in rigs:
            return (
                f"robot.grasping.fusion.cameras names {cam_id!r}, which is not a rig in camera.cameras.rigs "
                f"({rigs}). A fused camera's calibration is declared on its rig, so this camera's calibration "
                "has nowhere to be."
            )
    return None


class AppConfig(StrictModel):
    """Root configuration object returned by :func:`src.config.load_config`."""

    camera: CameraConfig
    models: ModelsConfig
    robot: RobotConfig | None = None
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @model_validator(mode="after")
    def _every_fused_camera_is_a_rig(self) -> AppConfig:
        """Every camera the fusion map names is a rig, where its calibration is declared.

        The rule is :func:`camera_calibration_conflict`; this validator runs it for the whole tree,
        the one place both halves are visible at load.
        """
        conflict = camera_calibration_conflict(self.camera, self.robot)
        if conflict is not None:
            raise ValueError(conflict)
        return self
