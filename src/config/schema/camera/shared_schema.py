"""Camera schemas shared between rig types and calibration routines."""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, ConfigDict, Field, model_validator

from .._base import StrictModel, validate_aruco_dict_name
from ..cameras import OpticalBox
from ..grippers import MODEL_NAME_PATTERN

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
    #: How far, in millimetres, the flange to TCP a wrist camera's calibration recorded may lie from the
    #: one the cell has now before the rig is refused, because its body and its pick frame were placed
    #: from a tool frame the arm no longer holds. Required on a rig that declares a ``body``, with no
    #: default: on a ``polyscope`` cell the frame is derived from the controller at connect, and how much
    #: that derivation moves is measured on the cell. On a ``willy`` cell the record is written from the
    #: declared numbers and compared exactly.
    record_tolerance_mm: float | None = Field(default=None, gt=0.0)
    #: The same bound on rotation, in degrees.
    record_tolerance_deg: float | None = Field(default=None, gt=0.0)

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
        if self.mounting_mode == "eye_to_hand" and (self.record_tolerance_mm, self.record_tolerance_deg) != (None, None):
            raise ValueError(
                "record_tolerance_mm and record_tolerance_deg bound how far a wrist camera's recorded flange to TCP "
                "may lie from the cell's, and an eye_to_hand rig is not placed on the flange, so it takes neither")
        return self


class RigBodyConfig(StrictModel):
    """The body a wrist camera adds to the arm: the registry housing of its model and the cell's bracket.

    Declared on the rig, placed on the flange by the rig's eye in hand calibration, and carried by the
    cuRobo planner, the exact mesh guard and the self filter. A rig that declares one is carried by the
    arm.
    """

    model_config = ConfigDict(allow_inf_nan=False)

    #: The camera model, the stem of a file under ``config/cameras/``, for example ``realsense_d435``.
    #: The repository's registry is the one authority for its housing.
    model: str = Field(pattern=MODEL_NAME_PATTERN)
    #: How far every box of the body is grown before a collision model reads it, in millimetres, above 0
    #: and at most 100. Required with no default: it covers the cable, the drawing's tolerance and how
    #: well the bracket was measured, which are facts about the cell.
    margin_mm: float = Field(gt=0.0, le=100.0)
    #: The bracket that holds the camera on the arm, as a box in the colour camera's optical frame, or
    #: ``null`` for a camera held by nothing the planner has to model. Required, written out as ``null``
    #: when there is none.
    bracket: OpticalBox | None


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
    #: The body the arm carries when this camera is mounted on it: a housing from the camera registry, a
    #: bracket and a margin. None is a camera the arm does not carry. A cell whose planner or guard reads
    #: geometry refuses an enabled eye_in_hand rig without one.
    body: RigBodyConfig | None = None

    @model_validator(mode="after")
    def _a_body_is_carried_by_the_arm(self) -> "BaseRigConfig":
        if self.body is None:
            return self
        where = f"camera.cameras.rigs[{self.rig_id!r}].body"
        if self.extrinsics is not None and self.extrinsics.mounting_mode == "eye_to_hand":
            raise ValueError(
                f"{where} declares a camera the arm carries, and the rig's extrinsics say eye_to_hand, a camera that "
                "does not move with the arm: a wrist camera is eye_in_hand, a fixed camera declares no body")
        if self.source != "rgbd":
            raise ValueError(
                f"{where} is on a {self.source!r} rig; a body is placed by the eye in hand calibration, which solves on "
                "an RGB-D rig's colour image, so only an rgbd rig declares one")
        if not re.fullmatch(r"[A-Za-z0-9_]+", self.rig_id):
            raise ValueError(
                f"{where}: the rig id {self.rig_id!r} names the body's link in the planner and its part in the exact "
                "guard, so a rig that declares a body has an id of letters, digits and underscores")
        if self.extrinsics is not None and None in (self.extrinsics.record_tolerance_mm,
                                                    self.extrinsics.record_tolerance_deg):
            raise ValueError(
                f"{where} is placed from the flange to TCP its calibration recorded, so the rig's extrinsics need "
                "record_tolerance_mm and record_tolerance_deg, each above 0, with no default: how far the recorded "
                "frame may lie from the cell's is measured on the cell")
        return self
