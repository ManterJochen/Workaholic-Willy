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


class AppConfig(StrictModel):
    """Root configuration object returned by :func:`src.config.load_config`."""

    camera: CameraConfig
    models: ModelsConfig
    robot: RobotConfig | None = None
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @model_validator(mode="after")
    def _the_primary_camera_is_calibrated_in_one_place(self) -> AppConfig:
        """The primary camera's calibration may be stated twice, and then it must say one thing.

        ``robot.grasping.fusion.cameras`` lists every camera that takes part in fusion, the primary
        included, each with its own artifact. ``fusion.extrinsics_artifact_path`` beside it is the
        primary's, and it stays: it is the key ``from_robot_config`` names when it refuses a cell
        with no CAMERA to BASE transform, and the key ``real_cell --check`` reports on. So a cell
        that lists its primary in the map has written the same fact down twice, and two artifacts
        for one camera is a cell that is calibrated differently depending on which loader ran.

        This lives on the root because it is the only place both halves are visible: which rig is
        primary is ``camera.cameras.primary_rig_id`` and the map is under ``robot``. A validator on
        the fusion block cannot see the camera section at all.
        """
        fusion = getattr(getattr(self.robot, "grasping", None), "fusion", None)
        if fusion is None:
            return self
        entry = (getattr(fusion, "cameras", None) or {}).get(self.camera.cameras.primary_rig_id)
        scalar = getattr(fusion, "extrinsics_artifact_path", None)
        if entry is None or scalar is None:
            return self
        mapped = getattr(entry, "extrinsics_artifact_path", None)
        if mapped is not None and str(mapped) != str(scalar):
            raise ValueError(
                f"the primary camera {self.camera.cameras.primary_rig_id!r} is calibrated twice "
                f"and the two disagree: robot.grasping.fusion.extrinsics_artifact_path is "
                f"{scalar!r} and its entry in robot.grasping.fusion.cameras is {mapped!r}. Both "
                "must name the same artifact. Which of the two a cell ends up using depends on "
                "which loader ran, so a cell configured this way is calibrated differently on two "
                "code paths."
            )
        return self
