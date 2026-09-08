"""Camera-system, rig-variant, stereo-matcher and eye-to-hand schemas."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AliasChoices, Field, model_validator

from .._base import StrictModel
from .shared_schema import (
    ArucoDictName,
    BaseRigConfig,
    CalibrationConfig,
    RGBDCalibPaths,
    StereoCalibPaths,
)


def _ensure_calib_paths(data: Any, rig_id_fallback: str = "unknown") -> Any:
    """Auto-fill ``calibration_paths.base_dir`` from ``rig_id`` if missing."""
    if not isinstance(data, dict):
        return data
    rig_id = data.get("rig_id", rig_id_fallback)
    cp = data.get("calibration_paths")
    if cp is None:
        data["calibration_paths"] = {"base_dir": f"calibration/{rig_id}"}
    elif isinstance(cp, dict):
        cp.setdefault("base_dir", f"calibration/{rig_id}")
    return data


def _check_positive_pair(pair: tuple[int, int], name: str) -> None:
    if len(pair) != 2 or pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"{name} must contain two positive values")


# ---------------------------------------------------------------------------
# Webcam pair
# ---------------------------------------------------------------------------

class WebcamPairRigConfig(BaseRigConfig):
    """Two independent USB cameras configured as a stereo pair."""

    source: Literal["webcam_pair"]

    frame_size: tuple[int, int]
    max_cam_scan: int = Field(ge=1)
    cam_left_id: int | None = Field(default=None, ge=0)
    cam_right_id: int | None = Field(default=None, ge=0)

    min_pairs: int = Field(default=10, ge=1)
    max_pairs: int = Field(default=40, ge=1)

    calibration_paths: StereoCalibPaths

    @model_validator(mode="before")
    @classmethod
    def _default_calib_paths(cls, data: Any) -> Any:
        return _ensure_calib_paths(data)

    @model_validator(mode="after")
    def _validate_capture_settings(self) -> WebcamPairRigConfig:
        _check_positive_pair(self.frame_size, "frame_size")
        if self.min_pairs > self.max_pairs:
            raise ValueError("min_pairs must be <= max_pairs")
        if (
            self.cam_left_id is not None
            and self.cam_right_id is not None
            and self.cam_left_id == self.cam_right_id
        ):
            raise ValueError("cam_left_id and cam_right_id must be different")
        return self


# ---------------------------------------------------------------------------
# Single stereo device (side-by-side frames from one capture device)
# ---------------------------------------------------------------------------

class SingleDeviceRigConfig(BaseRigConfig):
    """Single capture device that delivers a side-by-side stereo image."""

    source: Literal["single_device"]

    device_index: int = Field(ge=0)
    device_frame_size: tuple[int, int]
    per_eye_frame_size: tuple[int, int]

    layout: Literal["horizontal", "vertical"]

    crop_left: int = Field(default=0, ge=0)
    crop_right: int = Field(default=0, ge=0)
    crop_top: int = Field(default=0, ge=0)
    crop_bottom: int = Field(default=0, ge=0)

    allow_resize: bool = True
    min_sharpness: float | None = Field(default=None, ge=0.0)

    min_pairs: int = Field(default=10, ge=1)
    max_pairs: int = Field(default=40, ge=1)

    calibration_paths: StereoCalibPaths

    @model_validator(mode="before")
    @classmethod
    def _default_calib_paths(cls, data: Any) -> Any:
        return _ensure_calib_paths(data)

    @model_validator(mode="after")
    def _validate_capture_settings(self) -> SingleDeviceRigConfig:
        _check_positive_pair(self.device_frame_size, "device_frame_size")
        _check_positive_pair(self.per_eye_frame_size, "per_eye_frame_size")
        if self.min_pairs > self.max_pairs:
            raise ValueError("min_pairs must be <= max_pairs")
        return self


# ---------------------------------------------------------------------------
# RGB-D device  (e.g. Intel RealSense, Azure Kinect)
# ---------------------------------------------------------------------------

class RealSensePostProcessingConfig(StrictModel):
    """Depth post-processing filter chain for the RealSense backend.

    Each depth frame passes the enabled filters in librealsense's recommended order: decimation,
    spatial, temporal, hole-filling. A disabled filter is a no-op, so the defaults deliver raw
    device depth apart from the spatial and temporal smoothing RealSense recommends.
    """

    decimation: bool = False
    decimation_magnitude: int = Field(default=2, ge=2, le=8)
    spatial: bool = True
    temporal: bool = True
    hole_filling: bool = False
    hole_filling_mode: int = Field(default=1, ge=0, le=2)


class RealSenseConfig(StrictModel):
    """Intel RealSense (pyrealsense2) device tuning.

    Read only when ``rgbd_backend == "realsense"``; the generic OpenCV backend ignores it. Every
    field carries a working default, so the driver streams from a connected device without further
    configuration, and ``None`` leaves the device or SDK default in place.
    """

    enable_emitter: bool = True
    laser_power_mw: float | None = Field(default=None, ge=0.0)
    visual_preset: str | None = None
    depth_units_m: float | None = Field(default=None, gt=0.0)
    export_intrinsics: bool = False
    post_processing: RealSensePostProcessingConfig = Field(
        default_factory=RealSensePostProcessingConfig
    )


class RGBDDeviceRigConfig(BaseRigConfig):
    """RGB-D camera with native depth (e.g. Intel RealSense, Azure Kinect)."""

    source: Literal["rgbd"]

    device_index: int = Field(default=0, ge=0)
    serial_number: str | None = None

    color_resolution: tuple[int, int] = (1280, 720)
    depth_resolution: tuple[int, int] = (1280, 720)
    align_depth_to_color: bool = True

    # Driver backend: "opencv" is generic VideoCapture/OpenNI and needs no vendor SDK, "realsense"
    # is pyrealsense2 with native aligned depth. The default is the generic path, and the
    # ``realsense`` block below is read only by the RealSense backend.
    rgbd_backend: Literal["opencv", "realsense"] = "opencv"
    realsense: RealSenseConfig = Field(default_factory=RealSenseConfig)

    calibration_paths: RGBDCalibPaths

    @model_validator(mode="before")
    @classmethod
    def _default_calib_paths(cls, data: Any) -> Any:
        return _ensure_calib_paths(data)

    @model_validator(mode="after")
    def _validate_resolutions(self) -> RGBDDeviceRigConfig:
        _check_positive_pair(self.color_resolution, "color_resolution")
        _check_positive_pair(self.depth_resolution, "depth_resolution")
        return self


# ---------------------------------------------------------------------------
# Discriminated union: Pydantic v2 selects the variant via ``source``.
# ---------------------------------------------------------------------------

CameraRigUnion = Annotated[
    WebcamPairRigConfig | SingleDeviceRigConfig | RGBDDeviceRigConfig,
    Field(discriminator="source"),
]


class CameraSystemConfig(StrictModel):
    """Top-level camera section: the rig catalogue, and which of them the cell runs on."""

    primary_rig_id: str = Field(
        description=(
            "Which rig the cell opens. Required, and it must name a rig in the list below. It was "
            "`active_rig_id`, optional, and read only when `active_mode` was 'rig': nothing on the "
            "grasp path consulted either, and three pieces of code chose the camera three "
            "different ways. `build_real_components` took the first rig with `source: rgbd` by list "
            "position and did not look at `enabled`, so the shipped base profile, whose only RGB-D "
            "rig is switched off, would still open it. The order of a YAML list is not a place to "
            "keep a decision this size."
        ),
    )

    stereo_calibration: CalibrationConfig | None = None
    rigs: list[CameraRigUnion]

    @model_validator(mode="after")
    def _validate_rigs(self) -> CameraSystemConfig:
        if not self.rigs:
            raise ValueError("at least one camera rig must be configured")
        ids = [r.rig_id for r in self.rigs]
        dupes = {x for x in ids if ids.count(x) > 1}
        if dupes:
            raise ValueError(f"duplicate rig_id(s): {sorted(dupes)}")
        primary = next((rig for rig in self.rigs if rig.rig_id == self.primary_rig_id), None)
        if primary is None:
            raise ValueError(
                f"primary_rig_id {self.primary_rig_id!r} is not configured; the rigs are "
                f"{sorted(ids)}"
            )
        # `enabled` is not checked here, deliberately. `cam.tiltcam.yaml` ships both of its rigs off
        # until an operator fills in their serial numbers, and a profile that is not ready to run is
        # not the same thing as a file that is malformed: refusing it at load would make
        # `python -m src.config` fail on a tree nobody has finished writing. The cell refuses to open
        # a disabled primary when it is built, where the message can say what to switch on.

        # Two or more enabled RGB-D rigs are two or more physical cameras on one bus, and the serial
        # number is the only stable way to tell them apart. `RealSenseRGBDStreamer` binds a device
        # with `rs_config.enable_device(self.serial)`, and with no serial it takes whatever the SDK
        # offers first, an order that survives neither a reboot nor a replug. Two identical D435s
        # that swap identity do not fail: they produce a complete, plausible scene with left and
        # right exchanged, so every fused position is mirrored about the rig axis. No shipped
        # profile enables two RGB-D rigs at once: `cam.tiltcam.yaml` carries the two-rig case
        # disabled with `serial_number: null`, so this refusal does not fire on the shipped tree.
        rgbd = [r for r in self.rigs if getattr(r, "source", None) == "rgbd" and r.enabled]
        if len(rgbd) > 1:
            missing = [r.rig_id for r in rgbd if not getattr(r, "serial_number", None)]
            if missing:
                raise ValueError(
                    f"{len(rgbd)} RGB-D rigs are enabled but these have no serial_number: {missing}. "
                    "Two identical cameras on one bus can only be told apart by serial; device_index is "
                    "the SDK's enumeration order and can swap between boots, which silently exchanges "
                    "the views instead of failing. Read the serials with `rs-enumerate-devices` (or "
                    "pyrealsense2) and set one per rig."
                )
            serials = [str(getattr(r, "serial_number", "")) for r in rgbd]
            shared = sorted({v for v in serials if serials.count(v) > 1})
            if shared:
                raise ValueError(
                    f"the same serial_number is configured on more than one RGB-D rig: {shared}. "
                    "Both rigs would bind the same physical camera."
                )
        return self


# ---------------------------------------------------------------------------
# Stereo matcher (SGBM + post-filters)
# ---------------------------------------------------------------------------

class WlsFilterConfig(StrictModel):
    """Optional WLS (Weighted Least Squares) post-filter for SGBM disparity.

    Smooths disparity in low-texture regions while preserving edges. Needs ``cv2.ximgproc``, which
    ships with ``opencv-contrib-python``.
    """

    enabled: bool = False
    lambda_: float = Field(8000.0, alias="lambda")
    sigma_color: float = 1.5
    lr_check: bool = True


class StereoMatcherConfig(StrictModel):
    """SGBM disparity parameters plus optional post-processing knobs.

    YAML keeps the OpenCV-native spelling through ``camelCase`` aliases (``numDisparities``,
    ``blockSize``); Python reaches the same fields under their snake_case attribute names
    (``num_disparities``, ``block_size``).
    """

    min_disparity: int = Field(alias="minDisparity")
    num_disparities: int = Field(alias="numDisparities")
    block_size: int = Field(alias="blockSize")
    p1: int | None = Field(default=None, ge=0)
    p2: int | None = Field(default=None, ge=0)
    #: Percent by which the best match must beat the runner-up before it is trusted. Raising it
    #: trades wrong depths on textureless surfaces for holes where nothing wins clearly.
    uniqueness_ratio: int = Field(alias="uniquenessRatio", ge=0)
    #: Largest blob of similar disparity still treated as noise and cleared. A speck at the wrong
    #: depth is worse than a hole, because a grasp can be planned onto it. ``0`` disables it.
    speckle_window_size: int = Field(alias="speckleWindowSize", ge=0)
    #: Disparity spread allowed inside one blob before it reads as a separate surface rather than
    #: one speckle. It acts only through ``speckle_window_size``; on its own it changes nothing.
    speckle_range: int = Field(alias="speckleRange", ge=0)
    #: Tolerance (px) of the left-right consistency check: a pixel survives only if matching
    #: left-to-right and right-to-left agree within it. Guards against occlusion-edge depths, which
    #: are exactly the depths a grasp planner would otherwise aim at. ``-1`` disables the check.
    disp12_max_diff: int = Field(alias="disp12MaxDiff")

    # --- Quality knobs, all optional ---
    mode: Literal["sgbm", "sgbm_3way"] = "sgbm"
    subpixel: bool = True
    temporal_alpha: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Exponential moving average for disparity in realtime mode. "
            "0 = disabled (default), 1 = no smoothing (always latest), "
            "0<α<1 weights the previous frame by (1-α)."
        ),
    )
    wls: WlsFilterConfig = Field(default_factory=WlsFilterConfig)  # type: ignore[arg-type]

    @model_validator(mode="after")
    def _validate_sgbm_params(self) -> StereoMatcherConfig:
        if self.num_disparities <= 0 or self.num_disparities % 16 != 0:
            raise ValueError(
                f"num_disparities must be a positive multiple of 16 "
                f"(got {self.num_disparities})"
            )
        if self.block_size < 1 or self.block_size % 2 == 0:
            raise ValueError(
                f"block_size must be a positive odd integer (got {self.block_size})"
            )
        if self.p1 is not None and self.p2 is not None and self.p2 < self.p1:
            raise ValueError("p2 must be >= p1 when both are configured")
        return self


# ---------------------------------------------------------------------------
# Eye-hand calibration settings
# ---------------------------------------------------------------------------

class EyeHandRoutineConfig(StrictModel):
    """Shared sample-collection tuning for a hand-eye calibration workflow."""

    enabled: bool = True
    min_samples: int = Field(default=6, ge=4)
    min_distance_mm: float = Field(default=40.0, ge=0.0)
    min_angle: float = Field(
        default=10.0,
        ge=0.0,
        validation_alias=AliasChoices("min_angle", "min_angle_deg"),
    )
    marker_length_mm: float = Field(default=50.0, gt=0.0)
    aruco_dict_name: ArucoDictName = "DICT_5X5_100"

    @property
    def min_angle_deg(self) -> float:
        return self.min_angle


class EyeToHandWorkflowConfig(EyeHandRoutineConfig):
    """Config for fixed-camera calibration returning CAMERA -> BASE."""

    mode: Literal["eye_to_hand"] = "eye_to_hand"


class EyeInHandWorkflowConfig(EyeHandRoutineConfig):
    """Config for tool-mounted-camera calibration returning CAMERA -> TOOL."""

    mode: Literal["eye_in_hand"] = "eye_in_hand"


class HandEyeConfig(StrictModel):
    """Independent settings for both supported hand-eye workflows."""

    eye_to_hand: EyeToHandWorkflowConfig = Field(
        default_factory=EyeToHandWorkflowConfig
    )
    eye_in_hand: EyeInHandWorkflowConfig = Field(
        default_factory=EyeInHandWorkflowConfig
    )


