"""Camera schemas shared between rig types and calibration routines."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AfterValidator, Field, model_validator

from .._base import StrictModel, validate_aruco_dict_name

# ArUco dictionary name with strict validation against the OpenCV catalogue.
ArucoDictName = Annotated[str, AfterValidator(validate_aruco_dict_name)]


# ---------------------------------------------------------------------------
# Calibration paths (auto-derived from ``base_dir``)
# ---------------------------------------------------------------------------

class StereoCalibPaths(StrictModel):
    """Filesystem layout for stereo rigs (``webcam_pair`` / ``single_device``).

    Only ``base_dir`` is required; the sub-paths derive from it, and each one can be overridden by
    writing it out in YAML.
    """

    base_dir: str
    stereo_map_file: str = ""
    left_images_dir: str = ""
    right_images_dir: str = ""
    left_images_glob: str = ""
    right_images_glob: str = ""

    @model_validator(mode="before")
    @classmethod
    def _derive_from_base(cls, data: Any) -> Any:
        if isinstance(data, dict):
            base = data.get("base_dir", "")
            if base:
                data.setdefault("stereo_map_file", f"{base}/stereoMap.xml")
                data.setdefault("left_images_dir", f"{base}/left")
                data.setdefault("right_images_dir", f"{base}/right")
                data.setdefault("left_images_glob", f"{base}/left/*.png")
                data.setdefault("right_images_glob", f"{base}/right/*.png")
        return data


class RGBDCalibPaths(StrictModel):
    """Filesystem layout for RGB-D rigs: where the intrinsics artefact lives.

    Only ``base_dir`` is required; ``intrinsics_file`` derives from it and can be overridden.

    There are deliberately no ``color_images_dir`` / ``depth_images_dir`` fields, real though those
    are on the stereo rig above. Nothing here would read them: ``StereoCapturePipeline`` skips every
    RGB-D rig and reaches images through ``left/right_images_glob``, which this class has no
    counterpart for. The RGB-D artefact path uses no folders at all, only
    ``export_intrinsics_on_open`` writing ``intrinsics_file`` and ``load_intrinsics`` reading it
    back. If in-repo RGB-D intrinsics capture is ever built, add the directory and its glob
    together: one without the other is dead.
    """

    base_dir: str
    intrinsics_file: str = ""

    @model_validator(mode="before")
    @classmethod
    def _derive_from_base(cls, data: Any) -> Any:
        if isinstance(data, dict):
            base = data.get("base_dir", "")
            if base:
                data.setdefault("intrinsics_file", f"{base}/intrinsics.json")
        return data


# ---------------------------------------------------------------------------
# Shared rig + calibration settings
# ---------------------------------------------------------------------------

class QualityConfig(StrictModel):
    """Camera quality / capture-mode tuning shared by all rig types."""

    prefer_uncompressed: bool = False
    manual_exposure: float | None = None
    manual_gain: float | None = None
    manual_wb: float | None = None
    disable_auto_features: bool = True
    warmup_frames: int = Field(default=30, ge=0)
    fps_tolerance: float = Field(default=3.0, ge=0.0)


class CalibrationConfig(StrictModel):
    """Stereo-calibration ChArUco board parameters + this block's ArUco marker settings.

    These describe the stereo board only. The standalone hand-eye marker is a separate physical
    artefact configured under ``camera.hand_eye``, and it may use a different dictionary: under
    ``WILLY_PROFILE=sim`` the board is ``DICT_5X5_100`` at 50 mm and the hand-eye marker
    ``DICT_4X4_50`` at 48 mm. No cross-block validator ties the two, because OpenCV allows
    different dictionaries and a board is not a single marker.
    """

    charuco_squares_x: int = Field(gt=1)
    charuco_squares_y: int = Field(gt=1)
    charuco_square_length_mm: float = Field(gt=0.0)
    charuco_marker_length_mm: float = Field(gt=0.0)
    frame_size: tuple[int, int]
    rectify_alpha: float = Field(ge=0.0, le=1.0)
    marker_length_mm: float = Field(gt=0.0)
    aruco_dict_name: ArucoDictName

    @model_validator(mode="after")
    def _check(self) -> CalibrationConfig:
        if len(self.frame_size) != 2 or self.frame_size[0] <= 0 or self.frame_size[1] <= 0:
            raise ValueError("frame_size must contain two positive values")
        if self.charuco_marker_length_mm >= self.charuco_square_length_mm:
            raise ValueError(
                "charuco_marker_length_mm must be < charuco_square_length_mm (the marker fits inside a square)"
            )
        return self


class RigExtrinsicsConfig(StrictModel):
    """Where a rig's calibration is: its mounting, its artifact and, for a wrist camera, how far the
    arm may move while a frame is taken.

    Declared on the rig in the camera section, the one key every reader of CAMERA to BASE takes it
    from. A rig without the block is not calibrated, which is a stated state and not a default
    transform.
    """

    mounting_mode: Literal["eye_to_hand", "eye_in_hand"]
    artifact_path: str = Field(min_length=1)
    shutter_motion_tolerance_mm: float | None = Field(default=None, gt=0.0)
    shutter_motion_tolerance_deg: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _tolerances_follow_the_mounting(self) -> RigExtrinsicsConfig:
        tolerances = (self.shutter_motion_tolerance_mm, self.shutter_motion_tolerance_deg)
        if self.mounting_mode == "eye_in_hand" and None in tolerances:
            raise ValueError(
                "an eye_in_hand rig needs shutter_motion_tolerance_mm and shutter_motion_tolerance_deg, each "
                "above 0: a depth frame the planner world takes from it while the arm moved further than they "
                "allow is no frame, and "
                "there is no default, because how still the arm must be is a fact about the cell")
        if self.mounting_mode == "eye_to_hand" and any(t is not None for t in tolerances):
            raise ValueError(
                "shutter_motion_tolerance_mm and shutter_motion_tolerance_deg bound a wrist camera's motion "
                "at the shutter, and an eye_to_hand rig does not move with the arm, so it takes neither")
        return self


class BaseRigConfig(StrictModel):
    """Fields common to every camera rig variant."""

    rig_id: str = Field(min_length=1)
    enabled: bool
    source: Literal["webcam_pair", "single_device", "rgbd"]

    fps: int = Field(gt=0)
    backend: int = Field(default=0, ge=0)

    quality: QualityConfig = Field(default_factory=QualityConfig)

    #: The rig's calibration, declared on the rig. None is a rig that is not calibrated.
    extrinsics: RigExtrinsicsConfig | None = None
