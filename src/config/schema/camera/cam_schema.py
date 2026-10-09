"""Camera-system, rig-variant, stereo-matcher and eye-to-hand schemas."""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, Field, model_validator

from ...paths import PROJECT_ROOT_ANCHOR
from .._base import StrictModel
from .shared_schema import (
    ArucoDictName,
    BaseRigConfig,
    CalibrationConfig,
    RGBDCalibPaths,
    StereoCalibPaths,
)


def _ensure_calib_paths(data: Any, rig_id_fallback: str = "unknown") -> Any:
    """Auto-fill ``calibration_paths.base_dir`` from ``rig_id`` if missing.

    A default nobody wrote, so it names the repository's ``calibration/<rig_id>`` wherever the tree
    lives, as every path default does (:mod:`src.config.paths`).
    """
    if not isinstance(data, dict):
        return data
    rig_id = data.get("rig_id", rig_id_fallback)
    cp = data.get("calibration_paths")
    if cp is None:
        data["calibration_paths"] = {"base_dir": f"{PROJECT_ROOT_ANCHOR}/calibration/{rig_id}"}
    elif isinstance(cp, dict):
        cp.setdefault("base_dir", f"{PROJECT_ROOT_ANCHOR}/calibration/{rig_id}")
    return data


def _check_positive_pair(pair: tuple[int, int], name: str) -> None:
    if len(pair) != 2 or pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"{name} must contain two positive values")


def _refuse_stereo_extrinsics(rig: BaseRigConfig) -> None:
    """Refuse a calibration on a stereo rig. The planner world and the pick path take depth from
    RGB-D rigs only, so a calibrated stereo rig would be a camera every reader refuses later."""
    if rig.extrinsics is not None:
        raise ValueError(
            f"camera.cameras.rigs[{rig.rig_id!r}] is a {rig.source!r} rig; stereo rigs carry no extrinsics "
            "until their own step, because the planner world and the pick path take depth from RGB-D rigs only")


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
        _refuse_stereo_extrinsics(self)
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
        _refuse_stereo_extrinsics(self)
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

    Each depth frame passes the enabled filters in librealsense's recommended order, before it is
    aligned to colour: decimation, depth to disparity, spatial, temporal, disparity to depth,
    hole-filling. Spatial and temporal smooth disparity, which is what the sensor measures. A
    disabled filter is a no-op, so the defaults deliver raw device depth apart from the spatial and
    temporal smoothing RealSense recommends.
    """

    decimation: bool = False
    decimation_magnitude: int = Field(default=2, ge=2, le=8)
    spatial: bool = True
    #: Averages each depth pixel over the frames the driver grabbed and fills a hole with a depth seen
    #: in them. The driver drops that history when its owner is told the camera moved
    #: (``Camera.camera_moved()``), and on a camera the arm carries also after a pause between two
    #: grabs longer than a burst of back-to-back grabs takes, so a wrist camera's frame after a move
    #: holds no depth from the pose before it.
    temporal: bool = True
    hole_filling: bool = False
    hole_filling_mode: int = Field(default=1, ge=0, le=2)


class RealSenseColorConfig(StrictModel):
    """The colour sensor's exposure, gain and white balance, written when the stream opens (the owner, 2026-10-09:
    "Config-Block im Code").

    Every field ``null`` as shipped, which is today's full auto: nothing is written, and the camera keeps what it holds,
    auto exposure and auto white balance as it powers up. A value is written to the sensor that streams colour, never to
    the depth sensor, in the order of the fields: an auto mode goes off before its manual value is written, because
    librealsense writes the exposure's default when auto exposure goes off. Every value is checked against the range the
    sensor offers before the first is written, and read back once the warm-up frames have run; the driver logs what the
    sensor holds, also with every field null. A camera whose colour sensor does not offer an option set here refuses to
    open, naming the option and the camera: a fixed value was asked for, and a camera left on auto is not it.

    Measured 2026-10-09 on the owner's views, 31 looks: on full auto the same parts come out 1 to 1.8 stops brighter or
    darker with what else is in view at each look, orange parts clip in red and read as yellow, and grey cubes drift over
    the colour check's grey/white limit. Fixed at the values auto settles on at look 0 with the work lamp on, look 0 looks
    as it does today and the other looks stop drifting.

    The values stay in the camera until it is unplugged or power-cycled (RealSense support), so a block put back to null
    leaves the last values written in place: write ``auto_exposure: true`` and ``auto_white_balance: true`` to go back
    to auto.
    """

    #: Whether the colour sensor sets its own exposure and gain. ``false`` is written first, and ``exposure`` and
    #: ``gain`` need it: librealsense writes its default exposure as it goes off, and turns it off itself when an
    #: exposure or a gain is written.
    auto_exposure: bool | None = None
    #: The exposure time in the D400 colour sensor's own unit, 100 microseconds (UVC's exposure time unit, as
    #: librealsense reads and writes it: 156 is 15.6 ms; a D415 offers up to 10000). Needs ``auto_exposure: false``. On
    #: Windows, librealsense's Media Foundation backend sets the colour exposure in powers of two of a second, 39, 78,
    #: 156, 312, 625 and so on, so another value is held as the nearest of those: the driver logs what the sensor holds.
    exposure: float | None = Field(default=None, gt=0.0)
    #: The colour sensor's gain, in its own unit (UVC gain: a D415 offers 0 to 128 and starts at 64). Needs
    #: ``auto_exposure: false``, which sets the gain too while it is on.
    gain: float | None = Field(default=None, ge=0.0)
    #: Whether the colour sensor balances white itself. ``false`` is written before ``white_balance``, which needs it.
    auto_white_balance: bool | None = None
    #: The colour temperature white is balanced for, in Kelvin (a D415 offers 2800 to 6500 in steps of 10). Needs
    #: ``auto_white_balance: false``.
    white_balance: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _a_fixed_value_needs_its_auto_off(self) -> RealSenseColorConfig:
        for key, auto in (("exposure", "auto_exposure"), ("gain", "auto_exposure"),
                          ("white_balance", "auto_white_balance")):
            if getattr(self, key) is not None and getattr(self, auto) is not False:
                stated = "null" if getattr(self, auto) is None else "true"
                raise ValueError(
                    f"realsense.color.{key} is a value the colour sensor sets itself while {auto} is on, so it needs "
                    f"`{auto}: false` beside it, and {auto} is {stated}: librealsense would switch the auto mode off "
                    "unasked as the value is written, and the config is where what the camera does is said")
        return self


class RealSenseConfig(StrictModel):
    """Intel RealSense (pyrealsense2) device tuning.

    Read only when ``rgbd_backend == "realsense"``; the generic OpenCV backend ignores it. Every
    field carries a working default, so the driver streams from a connected device without further
    configuration, and ``None`` leaves the device or SDK default in place.
    """

    enable_emitter: bool = True
    laser_power_mw: float | None = Field(default=None, ge=0.0)
    #: A D400 preset by name, ``HighAccuracy`` or the SDK's own ``high_accuracy``, matched ignoring
    #: case and underscores. Written first, so the emitter, laser power and depth units written out
    #: here are applied over it.
    visual_preset: str | None = None
    #: Metres per raw depth unit. Written after the preset and read back: the driver scales depth by
    #: the device's reading, and refuses to open when that reading is not this value.
    depth_units_m: float | None = Field(default=None, gt=0.0)
    export_intrinsics: bool = False
    post_processing: RealSensePostProcessingConfig = Field(
        default_factory=RealSensePostProcessingConfig
    )
    #: The colour sensor's exposure, gain and white balance (:class:`RealSenseColorConfig`); every field null, the
    #: default, writes nothing and leaves the camera on what it holds.
    color: RealSenseColorConfig = Field(default_factory=RealSenseColorConfig)
    #: Record for research (the owner, 2026-10-09: "alles in einem Schalter"), off by default. On, the camera streams
    #: both infrared images as well (infrared 1 and 2, Y8, at the depth resolution and frame rate, with the projector as
    #: ``enable_emitter`` sets it and never toggled), and every frame carries them, the depth as the sensor streamed it,
    #: before the filters and the alignment, and the camera's facts. The views file of every pick a program keeps
    #: (``record_views``; the console keeps every task's) then holds them per look, with their lenses and extrinsics,
    #: every segmentation's mask, box, label and score, SAM2's predicted IoUs and which one was the target, and once the
    #: camera's name, serial, firmware, depth units, filters, preset, colour settings, the colour and depth lenses and the
    #: depth to colour extrinsics (``src/robot/execution/record_views.py``). Both infrared streams add 442 Mbit/s on the
    #: USB link at 1280 x 720 and 30 fps, half again what colour and depth take; a camera that cannot stream them refuses
    #: to open. Off, the streams, the frames and the views file are what they were.
    record_for_research: bool = False


class RGBDDeviceRigConfig(BaseRigConfig):
    """RGB-D camera with native depth (e.g. Intel RealSense, Azure Kinect)."""

    source: Literal["rgbd"]

    device_index: int = Field(default=0, ge=0)
    serial_number: str | None = None

    color_resolution: tuple[int, int] = (1280, 720)
    #: The depth mode sets how near the camera measures. Intel gives a D415 a minimum depth of about
    #: 450 mm at 1280 x 720 and about 310 mm at 848 x 480, and a D435 about 280 mm at 1280 x 720
    #: (datasheet figures, not measured here); anything nearer reads as a hole. A D415 on the wrist
    #: streams depth at (848, 480), with colour kept at (1280, 720) and depth aligned to it. The
    #: RealSense driver logs the camera, this mode and its Min-Z when it opens.
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
# Hand-eye calibration target
# ---------------------------------------------------------------------------

#: How many ids the predefined ArUco dictionaries whose name does not say it hold. An ``NxN_M`` name holds ``M``.
#: Written out rather than read from ``cv2``, so loading a config imports no OpenCV;
#: ``tests/test_calibration_targets.py`` holds every name against ``cv2.aruco``.
_ARUCO_DICT_IDS = {
    "DICT_ARUCO_ORIGINAL": 1024,
    "DICT_APRILTAG_16h5": 30,
    "DICT_APRILTAG_25h9": 35,
    "DICT_APRILTAG_36h10": 2320,
    "DICT_APRILTAG_36h11": 587,
    "DICT_ARUCO_MIP_36h12": 250,
}


def ids_in_aruco_dictionary(name: str) -> int | None:
    """How many marker ids a predefined dictionary holds (ids run from 0), ``None`` for a name this does not know."""
    if name in _ARUCO_DICT_IDS:
        return _ARUCO_DICT_IDS[name]
    match = re.fullmatch(r"DICT_\dX\d_(\d+)", name)
    return int(match.group(1)) if match else None


def _refuse_id_outside(marker_id: int, dictionary: str, key: str = "marker_id") -> None:
    size = ids_in_aruco_dictionary(dictionary)
    if size is not None and marker_id >= size:
        raise ValueError(f"{key} {marker_id} is not in {dictionary}, whose ids run from 0 to {size - 1}")


class ArucoTargetConfig(StrictModel):
    """One printed ArUco marker, posed from its four corners (``SOLVEPNP_IPPE_SQUARE``).

    ``marker_length_mm`` is the edge of the black square, measured on the print. It sets the scale of every solved
    translation, so a wrong value scales the whole calibration and no residual shows it.
    """

    kind: Literal["aruco"] = "aruco"
    marker_id: int = Field(default=0, ge=0)
    marker_length_mm: float = Field(gt=0.0)
    aruco_dict_name: ArucoDictName = "DICT_5X5_100"

    @model_validator(mode="after")
    def _id_in_dictionary(self) -> ArucoTargetConfig:
        _refuse_id_outside(self.marker_id, self.aruco_dict_name)
        return self


class CharucoTargetConfig(StrictModel):
    """A ChArUco board, posed from its interpolated chessboard corners.

    ``squares_x`` by ``squares_y`` chessboard squares of ``square_length_mm``, with ArUco markers of
    ``marker_length_mm`` in the white ones. ``legacy_pattern`` is the layout OpenCV generated before 4.6, which
    differs for an even ``squares_y``: a board read in the other layout shows its markers and no corners. A pose
    needs ``min_corners`` corners. The board poses from many points at once, so it has none of a single marker's
    flip ambiguity near a frontal view.
    """

    kind: Literal["charuco"] = "charuco"
    squares_x: int = Field(ge=2)
    squares_y: int = Field(ge=2)
    square_length_mm: float = Field(gt=0.0)
    marker_length_mm: float = Field(gt=0.0)
    aruco_dict_name: ArucoDictName = "DICT_5X5_100"
    legacy_pattern: bool = False
    min_corners: int = Field(default=6, ge=4)

    @model_validator(mode="after")
    def _a_board_opencv_can_hold(self) -> CharucoTargetConfig:
        if self.marker_length_mm >= self.square_length_mm:
            raise ValueError(
                f"marker_length_mm {self.marker_length_mm} must be smaller than square_length_mm "
                f"{self.square_length_mm}: the marker sits inside a white square")
        markers = (self.squares_x * self.squares_y) // 2
        size = ids_in_aruco_dictionary(self.aruco_dict_name)
        if size is not None and markers > size:
            raise ValueError(
                f"a {self.squares_x}x{self.squares_y} board carries {markers} markers and {self.aruco_dict_name} "
                f"holds {size}")
        corners = (self.squares_x - 1) * (self.squares_y - 1)
        if self.min_corners > corners:
            raise ValueError(
                f"min_corners {self.min_corners} is more than the {corners} inner corners a "
                f"{self.squares_x}x{self.squares_y} board has")
        return self


#: What a hand-eye sweep poses, told apart by ``kind``.
CalibrationTargetConfig = Annotated[
    ArucoTargetConfig | CharucoTargetConfig,
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Eye-hand calibration settings
# ---------------------------------------------------------------------------

class EyeHandRoutineConfig(StrictModel):
    """Shared sample-collection tuning for a hand-eye calibration workflow.

    The target is one ArUco marker described by ``marker_length_mm``, ``aruco_dict_name`` and ``marker_id``, unless
    ``target`` names it in full (a marker, or a ChArUco board). ``target.aruco_dict_name`` left out is the block's.
    Beside a marker ``target``, each of the three keys the block writes out must agree with it, so one marker is
    described once. Beside a board, ``aruco_dict_name`` must agree, and ``marker_length_mm`` and ``marker_id``, which
    describe one marker, are not read.
    """

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
    marker_id: int = Field(default=0, ge=0)
    target: CalibrationTargetConfig | None = None

    @property
    def min_angle_deg(self) -> float:
        return self.min_angle

    @model_validator(mode="before")
    @classmethod
    def _the_target_dictionary_defaults_to_the_block(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        target = data.get("target")
        if isinstance(target, dict) and "aruco_dict_name" not in target and "aruco_dict_name" in data:
            data = {**data, "target": {**target, "aruco_dict_name": data["aruco_dict_name"]}}
        return data

    @model_validator(mode="after")
    def _the_target_is_stated_once(self) -> EyeHandRoutineConfig:
        target = self.target
        if target is None:
            _refuse_id_outside(self.marker_id, self.aruco_dict_name)
            return self
        stated = self.model_fields_set
        keys = ("marker_length_mm", "aruco_dict_name", "marker_id") if isinstance(target, ArucoTargetConfig) else (
            "aruco_dict_name",)
        disagree = [f"{key} {getattr(self, key)!r} and target.{key} {getattr(target, key)!r}"
                    for key in keys if key in stated and getattr(self, key) != getattr(target, key)]
        if disagree:
            raise ValueError(
                "the block and its target disagree: " + "; ".join(disagree)
                + ". State the target once: remove the block's key, or make it agree")
        return self

    def resolved_target(self) -> ArucoTargetConfig | CharucoTargetConfig:
        """The target this block describes: ``target`` when it is set, else the one marker its own keys name."""
        if self.target is not None:
            return self.target
        return ArucoTargetConfig(kind="aruco", marker_id=self.marker_id, marker_length_mm=self.marker_length_mm,
                                 aruco_dict_name=self.aruco_dict_name)


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


