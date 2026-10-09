"""RGB-D runtime streamers: a generic OpenCV backend and an Intel RealSense backend.

Both return the canonical :class:`RGBDFrame`, BGR colour plus uint16 millimetre depth,
so :class:`~src.camera.orchestration.frame_provider.FrameProvider` selects between them
on ``RGBDDeviceRigConfig.rgbd_backend`` and the rest of the pipeline sees one frame type.

* :class:`OpenCvRGBDStreamer` reads colour and depth through ``cv2.VideoCapture`` and the
  OpenNI retrieve flags. No vendor SDK, so depth exists only where the build exposes it.
* :class:`RealSenseRGBDStreamer` drives Intel RealSense through ``pyrealsense2``:
  hardware-aligned depth, device depth-scale, emitter and laser and preset control, a
  post-processing filter chain, intrinsics. The SDK import is deferred, so this module
  imports without it; the librealsense round-trip itself needs a physical device, and is
  validated on one.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections.abc import Callable
from typing import Any, Literal, Protocol, runtime_checkable

import cv2 as cv
import numpy as np

from src.config.schema.camera import RGBDDeviceRigConfig
from src.camera.setup.image_taking.frames import CameraFacts, ResearchCapture, RGBDFrame
from src.camera.setup.quality import configure_camera_for_quality
from src.utility.log_cfg import create_logger


#: Where the RGB-D drivers' own lines go: ``logs/camera/rgbd.log`` and the console. Each driver logs as
#: ``<this module>.<rig id>``, and its lines reach this module's logger and its file: what a RealSense opened, what its
#: colour sensor holds, every warning it gives. Nothing in this repository configures the root logger, so before this
#: sink a driver's INFO lines reached no file and its warnings only the console's window (read off the code,
#: 2026-10-09, when the colour sensor's read-back was added and had to be read at the cell).
CAMERA_LOG_DIR = "logs/camera"
create_logger(__name__, "rgbd.log", log_dir=CAMERA_LOG_DIR)


#: The colour sensor's options a rig's ``realsense.color`` block writes, by the block's key and librealsense's option
#: name, in the order they are written: each auto mode before its manual value, because librealsense writes the
#: exposure's default as auto exposure goes off, and switches an auto mode off itself when its value is written.
_COLOR_OPTIONS: tuple[tuple[str, str], ...] = (
    ("auto_exposure", "enable_auto_exposure"),
    ("exposure", "exposure"),
    ("gain", "gain"),
    ("auto_white_balance", "enable_auto_white_balance"),
    ("white_balance", "white_balance"),
)
#: What a colour frame's metadata says of how it was exposed, kept with every frame a camera records for research.
_COLOR_METADATA: tuple[str, ...] = ("actual_exposure", "gain_level", "white_balance", "auto_exposure")
#: The infrared streams a recording for research adds, by the name the recording keeps each under, with librealsense's
#: index: infrared 1 is the left imager, the one the depth is measured in, and infrared 2 the right one.
_INFRARED: tuple[tuple[str, int], ...] = (("ir_left", 1), ("ir_right", 2))


#: Intel's minimum depth (Min-Z) of a D400 camera, in millimetres, by model and depth-stream width: a surface nearer
#: than this is not measured and reads as 0, a hole. The 1280-pixel values are Intel's datasheet figures for the full
#: 1280 x 720 mode, and the D415's 848 x 480 value is the one the 2026-09-23 audit of the owner's cell took from
#: Intel's material. None of them is measured on a device here. A D435i has the D435's depth module.
_MIN_Z_MM: dict[str, dict[int, float]] = {
    "D415": {1280: 450.0, 848: 310.0},
    "D435": {1280: 280.0},
    "D435I": {1280: 280.0},
}


def realsense_min_depth_mm(model: str | None, depth_width: int) -> float | None:
    """How near a D400 camera measures at a depth-stream width, in millimetres, or ``None`` for a model not in the table.

    A width the table does not list scales the model's 1280-pixel value with the width, as Intel's tuning guide
    describes Min-Z: it falls in proportion to the horizontal depth resolution. The number is Intel's, not a
    measurement, and a disparity shift, which this driver never sets, would change it.
    """
    table = _MIN_Z_MM.get((model or "").upper())
    if table is None:
        return None
    if depth_width in table:
        return table[depth_width]
    return table[1280] * float(depth_width) / 1280.0


def _depth_camera_model(device_name: str | None) -> str | None:
    """The model in a device name the SDK reports, ``"D415"`` out of ``"Intel RealSense D415"``, or ``None``."""
    match = re.search(r"\b(D\d{3}[A-Za-z]?)\b", device_name or "")
    return match.group(1).upper() if match else None


def _preset_key(name: str) -> str:
    """A preset name reduced to letters and digits, so ``HighAccuracy`` names the SDK's ``high_accuracy``."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _camera_info(rs: Any, device: Any, key: str) -> str | None:
    """One ``rs.camera_info`` field of a device, or ``None`` where the device does not report it.

    Only ever a diagnostic, so a device or an SDK that cannot answer gives ``None`` rather than an error.
    """
    try:
        field = getattr(rs.camera_info, key)
        if not device.supports(field):
            return None
        value = str(device.get_info(field)).strip()
    except Exception:  # noqa: BLE001 (a diagnostic never takes the stream down)
        return None
    return value or None


def _connected_cameras(rs: Any) -> list[dict[str, str | None]]:
    """The name, serial and USB link of every RealSense the SDK enumerates; empty where it cannot enumerate."""
    try:
        devices = list(rs.context().query_devices())
    except Exception:  # noqa: BLE001 (a diagnostic never replaces the refusal it explains)
        return []
    return [{key: _camera_info(rs, device, key) for key in ("name", "serial_number", "usb_type_descriptor")}
            for device in devices]


def _options_of(holder: Any) -> dict[str, float]:
    """Every option a librealsense sensor or processing block offers, by name, with the value it holds now.

    Only ever a record, so an option that does not answer is left out, and a holder that cannot list its options gives
    an empty map rather than an error.
    """
    try:
        offered = list(holder.get_supported_options())
    except Exception:  # noqa: BLE001 (a record never takes the stream down)
        return {}
    held: dict[str, float] = {}
    for option in offered:
        try:
            held[str(getattr(option, "name", option))] = float(holder.get_option(option))
        except Exception:  # noqa: BLE001 (an option that does not answer is left out of the record)
            continue
    return held


def _extrinsics_mm(extrinsics: Any) -> np.ndarray:
    """An SDK extrinsics as a 4x4 in millimetres that maps a point of the first stream's frame into the second's.

    librealsense keeps the rotation column-major and the translation in metres: a point ``p`` maps to
    ``R @ p + t`` with ``R`` the rotation read row by row and transposed (``rs2_transform_point_to_point``).
    """
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(extrinsics.rotation, dtype=np.float64).reshape(3, 3).T
    matrix[:3, 3] = np.asarray(extrinsics.translation, dtype=np.float64).reshape(3) * 1000.0
    return matrix


def _copied(frame: Any) -> np.ndarray | None:
    """A frame's pixels as an array the device's buffer no longer backs, or None where there is no frame."""
    if not frame:
        return None
    return np.array(np.asanyarray(frame.get_data()), copy=True)


def _said_colour(key: str, value: float | None) -> str:
    """One colour option as the open line says it: ``exposure 156 (15.6 ms)``, ``auto white balance on``."""
    words = key.replace("_", " ")
    if value is None:
        return f"{words} ?"
    if key.startswith("auto_"):
        return f"{words} {'on' if value else 'off'}"
    if key == "exposure":
        return f"exposure {value:g} ({value / 10.0:.1f} ms)"
    if key == "white_balance":
        return f"white balance {value:g} K"
    return f"{words} {value:g}"


@runtime_checkable
class RGBDStreamerProtocol(Protocol):
    """Runtime contract every RGB-D streamer backend fulfils."""

    def open(self) -> None: ...
    def release(self) -> None: ...
    def is_opened(self) -> bool: ...
    def grab(self) -> RGBDFrame: ...


# ------------------------------------------------------------------
# Generic OpenCV / OpenNI RGB-D streamer
# ------------------------------------------------------------------

class OpenCvRGBDStreamer:
    """Generic RGB-D streamer over ``cv2.VideoCapture`` and the OpenNI depth channels.

    Depth arrives only when the OpenCV build exposes it through the ``CAP_OPENNI_*``
    retrieve flags. No vendor SDK is involved, so a device whose depth is reachable
    only through its native SDK, a RealSense over librealsense for one, hands back an
    empty depth channel here; :class:`RealSenseRGBDStreamer` covers that device.
    Colour stream settings come from ``quality.configure_camera_for_quality``.
    """

    def __init__(self, config: RGBDDeviceRigConfig):
        self.cfg = config
        self.logger = logging.getLogger(f"{__name__}.{config.rig_id}")

        self.device_index = int(config.device_index)
        self.color_res = tuple(config.color_resolution)
        self.depth_res = tuple(config.depth_resolution)
        self.align = config.align_depth_to_color
        self.fps = config.fps
        self.backend = config.backend

        self._cap: cv.VideoCapture | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the RGB-D device and apply quality settings to the colour stream."""
        self._cap = cv.VideoCapture(self.device_index, self.backend)

        if not self._cap.isOpened():
            raise OSError(f"Could not open RGB-D device index {self.device_index}")

        q = self.cfg.quality
        settings = configure_camera_for_quality(
            cap=self._cap,
            width=self.color_res[0],
            height=self.color_res[1],
            fps=self.fps,
            prefer_uncompressed=q.prefer_uncompressed,
            manual_exposure=q.manual_exposure,
            manual_gain=q.manual_gain,
            manual_wb=q.manual_wb,
            disable_auto_features=q.disable_auto_features,
            warmup_frames=q.warmup_frames,
        )
        # Fail closed when the device did not honour the requested colour resolution,
        # as SingleDeviceStreamer._validate_settings does. Depth resolution cannot be
        # read back through this single VideoCapture, so it stays unchecked.
        if (settings["width"], settings["height"]) != self.color_res:
            raise ValueError(
                f"RGB-D colour resolution mismatch: "
                f"{settings['width']}x{settings['height']} != "
                f"{self.color_res[0]}x{self.color_res[1]}"
            )
        if abs(settings["fps"] - self.fps) > q.fps_tolerance:
            self.logger.warning(
                "RGB-D FPS differs: actual=%.1f, requested=%d, tol=%.1f",
                settings["fps"], self.fps, q.fps_tolerance,
            )
        self.logger.info(
            "RGB-D device %d opened: FOURCC=%s %dx%d @ %.1f fps",
            self.device_index,
            settings["fourcc_str"],
            settings["width"],
            settings["height"],
            settings["fps"],
        )

    def release(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
        self._cap = None

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def grab(self) -> RGBDFrame:
        """Grab a colour frame and its depth map.

        Colour is retrieved with ``CAP_OPENNI_BGR_IMAGE`` and depth with
        ``CAP_OPENNI_DEPTH_MAP``. A backend or device without depth yields an empty
        depth array of shape ``(0,)`` rather than an error.

        Returns:
            RGBDFrame with .color (BGR uint8) and .depth (uint16 mm).
        """
        if not self.is_opened():
            raise RuntimeError("Device is not open. Call open() first.")

        assert self._cap is not None  # guaranteed by is_opened() above
        ok = self._cap.grab()
        if not ok:
            raise RuntimeError("Failed to grab frame from RGB-D device.")

        _, color = self._cap.retrieve(flag=cv.CAP_OPENNI_BGR_IMAGE)
        if color is None:
            raise RuntimeError("Failed to retrieve colour frame from RGB-D device.")

        # Not every backend or device carries a depth channel, so a missing one is an
        # expected result rather than a failure.
        _, depth = self._cap.retrieve(flag=cv.CAP_OPENNI_DEPTH_MAP)
        if depth is None:
            self.logger.debug("Depth channel not available, returning empty array.")
            depth = np.empty(0, dtype=np.uint16)

        return RGBDFrame(color=color, depth=depth)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> OpenCvRGBDStreamer:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> Literal[False]:
        self.release()
        return False


# ------------------------------------------------------------------
# Intel RealSense RGB-D streamer (pyrealsense2)
# ------------------------------------------------------------------

class RealSenseRGBDStreamer:
    """Intel RealSense RGB-D streamer over ``pyrealsense2``.

    Streams hardware time-synced BGR colour and Z16 depth, runs the configured post-processing
    filter chain on the depth, aligns it to the colour frame if configured, and converts it to
    uint16 millimetres with the device's depth scale. The SDK import is deferred to
    :meth:`_import_rs`, so importing this module never requires ``pyrealsense2``, and
    ``rs_module`` substitutes a stand-in for it, which runs the driver logic with no hardware
    attached. ``clock`` is the monotonic clock the pause rule of :meth:`grab` reads.

    The temporal filter averages each depth pixel over the frames this driver handed it and
    fills a hole with a depth it saw in them, and the driver hands it only the frames it grabs.
    On a camera the arm carries those frames can come from the pose before the last move, so
    the history is dropped when :meth:`camera_moved` says the camera moved, and, on a camera the
    arm carries, when a pause between two grabs is longer than a burst of back-to-back grabs
    takes.

    The colour sensor's exposure, gain and white balance are the rig's ``realsense.color`` block,
    written as the stream opens and read back once the warm-up frames have run; every key null
    writes nothing, and the open still says what the sensor holds. A rig that records for research
    (``realsense.record_for_research``) streams both infrared images as well, and every frame then
    carries them, the depth as the sensor sent it and the camera's facts (:class:`ResearchCapture`).
    """

    #: The shortest pause between two grabs, in seconds, that drops a carried camera's temporal
    #: history. A burst of back-to-back grabs, the warm-ups a pick frame takes, is one frame
    #: period apart plus the filtering; any arm move and settle takes longer than this.
    _PAUSE_S = 0.25
    #: The same bound in frame periods, for a slow mode whose frame period nears the pause.
    _PAUSE_FRAMES = 3.0
    #: How far the depth scale the device reads back may lie from a configured ``depth_units_m``,
    #: relative. The SDK holds the option as a 32-bit float, so a value written reads back within
    #: about 1e-7 of itself; a unit the device did not take is off by a factor.
    _UNITS_REL_TOL = 1e-3

    def __init__(self, config: RGBDDeviceRigConfig, *, rs_module: Any | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg = config
        self.rs_cfg = config.realsense
        self.logger = logging.getLogger(f"{__name__}.{config.rig_id}")

        self.color_res = tuple(config.color_resolution)
        self.depth_res = tuple(config.depth_resolution)
        self.fps = int(config.fps)
        self.serial = config.serial_number
        self.align_to_color = config.align_depth_to_color
        #: A camera the arm carries: it declares a body, or its calibration says eye_in_hand.
        self.carried = config.body is not None or (
            config.extrinsics is not None and config.extrinsics.mounting_mode == "eye_in_hand")

        self._rs = rs_module
        self._clock = clock
        self._pipeline: Any | None = None
        self._align: Any | None = None
        self._filters: list[Any] = []
        #: Where the temporal filter sits in ``_filters``, or None when the chain has none.
        self._temporal_at: int | None = None
        self._last_grab_s: float | None = None
        self._depth_scale_m: float | None = None
        self._intrinsics: np.ndarray | None = None
        self._distortion: np.ndarray | None = None
        self._device_name: str | None = None
        self._usb: str | None = None
        self._min_depth_mm: float | None = None
        #: The sensor that streams colour, found as the stream opens; None where the device has none of its own.
        self._color_sensor: Any | None = None
        #: What the rig's ``realsense.color`` block wrote, by its key: the number written and the range the sensor
        #: offered for it.
        self._color_written: dict[str, tuple[float, Any]] = {}
        #: What the colour sensor held once the warm-up frames had run, by the block's key; None where it did not say.
        self._color_held: dict[str, float | None] = {}
        #: The camera's facts while the rig records for research (``realsense.record_for_research``), else None.
        self._facts: CameraFacts | None = None
        #: Whether a frameset without an infrared image was said during this open, so it is said once.
        self._research_gap_said = False

    # ------------------------------------------------------------------
    # SDK import seam
    # ------------------------------------------------------------------

    def _import_rs(self) -> Any:
        """Return the injected ``rs_module`` if there is one, else import ``pyrealsense2``."""
        if self._rs is not None:
            return self._rs
        try:
            import pyrealsense2 as rs  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover (exercised only on bare envs)
            raise ImportError(
                "pyrealsense2 is not installed. It is pinned in requirements.txt, so "
                "`pip install -r requirements.txt` supplies it, or pass a custom rs_module."
            ) from exc
        self._rs = rs
        return rs

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Start the pipeline, configure the depth sensor, the colour sensor and the filters, then read intrinsics.

        A request the SDK cannot start is raised with the RealSense cameras it sees and the USB link
        each is on. A depth sensor that did not take the configured ``depth_units_m`` refuses the open,
        and so does a ``realsense.color`` block the colour sensor cannot hold. Either way the pipeline
        is not left running. A rig that records for research asks for both infrared images beside
        colour and depth, at the depth's resolution and frame rate, in Y8.
        """
        rs = self._import_rs()

        rs_config = rs.config()
        if self.serial:
            rs_config.enable_device(self.serial)
        rs_config.enable_stream(
            rs.stream.color, self.color_res[0], self.color_res[1], rs.format.bgr8, self.fps
        )
        rs_config.enable_stream(
            rs.stream.depth, self.depth_res[0], self.depth_res[1], rs.format.z16, self.fps
        )
        if self.rs_cfg.record_for_research:
            for _, index in _INFRARED:
                rs_config.enable_stream(
                    rs.stream.infrared, index, self.depth_res[0], self.depth_res[1], rs.format.y8, self.fps
                )

        pipeline = rs.pipeline()
        try:
            profile = pipeline.start(rs_config)
        except RuntimeError as exc:
            raise RuntimeError(self._start_refusal(rs, exc)) from exc
        self._pipeline = pipeline
        try:
            self._configure(rs, profile)
        except BaseException:
            self.release()
            raise

    def _configure(self, rs: Any, profile: Any) -> None:
        """Everything ``open()`` does once the pipeline runs."""
        device = profile.get_device()
        self._device_name = _camera_info(rs, device, "name")
        self._usb = _camera_info(rs, device, "usb_type_descriptor")
        self._min_depth_mm = realsense_min_depth_mm(_depth_camera_model(self._device_name), int(self.depth_res[0]))

        depth_sensor = device.first_depth_sensor()
        self._configure_depth_sensor(rs, depth_sensor)
        self._depth_scale_m = self._read_back_depth_scale(depth_sensor)
        # Before the warm-up frames, so they run at the exposure the frames after them are taken with.
        self._configure_color_sensor(rs, device)

        self._align = rs.align(rs.stream.color) if self.align_to_color else None
        self._filters = self._build_filters(rs)
        self._last_grab_s = None

        assert self._pipeline is not None  # set by open() before it calls this
        for _ in range(self.cfg.quality.warmup_frames):
            self._pipeline.wait_for_frames()

        self._intrinsics = self._read_intrinsics(rs, profile)
        if self.rs_cfg.export_intrinsics and self._intrinsics is not None:
            self._export_intrinsics(self._intrinsics)

        # Once the warm-up frames have run: on auto, what auto settled on.
        self._color_held = self._read_back_color(rs)
        self._facts = self._camera_facts(rs, profile, device, depth_sensor) if self.rs_cfg.record_for_research else None
        self._research_gap_said = False

        self._say_what_opened(device, rs)
        self._say_what_the_colour_holds()

    def release(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception as exc:  # pragma: no cover (defensive logging)
                self.logger.warning("RealSense pipeline stop failed: %s", exc)
        self._pipeline = None
        self._align = None
        self._filters = []
        self._temporal_at = None
        self._last_grab_s = None
        self._color_sensor = None
        self._facts = None

    def is_opened(self) -> bool:
        return self._pipeline is not None

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def grab(self) -> RGBDFrame:
        """Grab a colour and depth pair, filtered and aligned as configured, as an RGBDFrame.

        On a camera the arm carries, a grab that follows the previous one by more than a burst
        takes starts the temporal filter afresh (see the class docstring). On a rig that records for
        research the frame also carries the infrared images and the depth of the frameset as the sensor
        sent it, copied before any filter runs (:class:`ResearchCapture`); on every other rig it carries
        none, and the frame is what it was.
        """
        if self._pipeline is None:
            raise RuntimeError("Device is not open. Call open() first.")

        now = self._clock()
        if self.carried and self._last_grab_s is not None and now - self._last_grab_s > self._pause_s():
            self.camera_moved()
        self._last_grab_s = now

        frameset = self._pipeline.wait_for_frames()
        research = None if self._facts is None else self._research_capture(frameset)
        # librealsense's recommended order: every filter on the depth as it left the sensor, and the
        # alignment to colour last. Spatial and temporal run between the two disparity transforms.
        for filt in self._filters:
            frameset = filt.process(frameset).as_frameset()
        if self._align is not None:
            frameset = self._align.process(frameset)

        depth_frame = frameset.get_depth_frame()
        color_frame = frameset.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("RealSense returned an incomplete frameset (missing colour or depth).")

        color = np.ascontiguousarray(np.asanyarray(color_frame.get_data()))
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_mm = self._match_color_size(self._to_millimetres(depth_raw), color)
        return RGBDFrame(color=color, depth=depth_mm, research=research)

    def camera_moved(self) -> None:
        """Say that the camera moved since its last grab, so the next frame holds only depth seen from where it is now.

        Builds a new temporal filter in the old one's place, which drops the frames it averaged at
        the pose before. The other filters keep no history. A no-op on a chain without a temporal
        filter, and on a driver that is not open.
        """
        if self._temporal_at is None:
            return
        self._filters[self._temporal_at] = self._import_rs().temporal_filter()

    def peek(self) -> np.ndarray | None:
        """The newest colour image the pipeline holds, copied, for a person to look at; ``None`` when it holds none.

        The display path a camera window reads (``Camera.peek``), which changes nothing a grab measures:

        * The frameset is taken with ``poll_for_frames``, which never waits, and only its colour is read. The colour
          stream is what the alignment aligns depth to, so it is the image a grab would have shown.
        * No filter runs on it and no alignment, so the temporal filter's history holds the frames :meth:`grab`
          handed it and nothing else: a frame looked at after :meth:`camera_moved`, still from the pose before the
          move, never fills a hole of the pose after it.
        * The pause clock is not touched, so a carried camera still drops its history when the pause between two
          grabs says it moved, however many frames were looked at in that pause.
        * Taking the frameset means the next grab waits for the next one, at most a frame period, and gets a frame
          at least as new as it would have.

        Copied, because the colour is a view of the device's buffer and a view kept by a window would hold it.
        """
        if self._pipeline is None:
            raise RuntimeError("Device is not open. Call open() first.")
        frameset = self._pipeline.poll_for_frames()
        if not frameset:
            return None
        color_frame = frameset.get_color_frame()
        if not color_frame:
            return None
        return np.array(np.asanyarray(color_frame.get_data()), copy=True)

    def _pause_s(self) -> float:
        return max(self._PAUSE_S, self._PAUSE_FRAMES / float(self.fps))

    def _research_capture(self, frameset: Any) -> ResearchCapture:
        """What a rig recording for research keeps of ``frameset`` as the sensor sent it: both infrared images and the
        raw depth, copied, the colour frame's metadata and the camera's facts. A frameset missing an infrared image
        keeps None in its place, said once per open: the recording is for research, and the frame is the pick's."""
        assert self._facts is not None  # grab() asks only while the rig records for research
        images = {name: _copied(frameset.get_infrared_frame(index)) for name, index in _INFRARED}
        missing = [name for name, image in images.items() if image is None]
        if missing and not self._research_gap_said:
            self._research_gap_said = True
            self.logger.warning(
                "RealSense rig %r records for research, and a frameset came without %s: that frame is kept without "
                "it (said once per open)", self.cfg.rig_id, " and ".join(missing))
        return ResearchCapture(
            camera=self._facts, ir_left=images["ir_left"], ir_right=images["ir_right"],
            raw_depth=_copied(frameset.get_depth_frame()), color_metadata=self._color_metadata(frameset.get_color_frame()),
        )

    def _color_metadata(self, frame: Any) -> dict[str, float]:
        """What a colour frame's metadata says of how it was exposed, the values the device reports; empty where it
        reports none (on Windows a RealSense reports metadata once its driver is told to, which the RealSense Viewer
        offers to do)."""
        keys = getattr(self._import_rs(), "frame_metadata_value", None)
        said: dict[str, float] = {}
        if not frame or keys is None:
            return said
        for name in _COLOR_METADATA:
            key = getattr(keys, name, None)
            try:
                if key is not None and frame.supports_frame_metadata(key):
                    said[name] = float(frame.get_frame_metadata(key))
            except Exception:  # noqa: BLE001 (metadata is a record, never a reason to lose the frame)
                continue
        return said

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_intrinsics(self) -> np.ndarray | None:
        """Return the 3x3 colour-stream camera matrix K (available after ``open()``)."""
        return None if self._intrinsics is None else self._intrinsics.copy()

    def get_distortion(self) -> np.ndarray | None:
        """Return the colour-stream Brown-Conrady coefficients (available after ``open()``).

        ``None`` when the device reported none, and a PnP solve should then pass
        ``np.zeros(5)``. The aligned colour stream typically reports near-zero coefficients,
        read off the device rather than assumed.
        """
        return None if self._distortion is None else self._distortion.copy()

    @property
    def depth_scale_m(self) -> float | None:
        """Metres per raw depth unit, as the device reads it back after configuration (after ``open()``)."""
        return self._depth_scale_m

    @property
    def device_name(self) -> str | None:
        """The name the SDK reports for the opened camera, ``"Intel RealSense D415"``, or None."""
        return self._device_name

    @property
    def min_depth_mm(self) -> float | None:
        """Intel's Min-Z for the opened model at this depth mode, in millimetres, or None where it is not known.

        Nothing nearer than this is measured. A datasheet figure, not a measurement: see
        :func:`realsense_min_depth_mm`.
        """
        return self._min_depth_mm

    @property
    def color_held(self) -> dict[str, float | None]:
        """What the colour sensor held once the warm-up frames had run, by the key of the rig's ``realsense.color``
        block (after ``open()``): ``auto_exposure`` and ``auto_white_balance`` as 1 or 0, the exposure in 100
        microseconds, the gain, the white balance in Kelvin. None for an option it does not offer or did not answer."""
        return dict(self._color_held)

    # ------------------------------------------------------------------
    # Internal: what open() says
    # ------------------------------------------------------------------

    def _say_what_opened(self, device: Any, rs: Any) -> None:
        serial = _camera_info(rs, device, "serial_number")
        firmware = _camera_info(rs, device, "firmware_version")
        name = self._device_name or "RealSense (the SDK reports no name)"
        dw, dh = self.depth_res
        min_z = ("Min-Z unknown for this model" if self._min_depth_mm is None
                 else f"nothing nearer than about {self._min_depth_mm:.0f} mm is measured at this depth mode "
                      "(Intel's figure, not measured here)")
        infrared = (f" + infrared 1 and 2 {dw}x{dh} Y8 (record_for_research)" if self._facts is not None else "")
        self.logger.info(
            "RealSense opened: %s (serial %s, firmware %s, USB %s) | colour %dx%d + depth %dx%d%s @ %d fps | %s | "
            "depth_scale=%.6g m/unit read back from the device | align=%s | filters=%s",
            name, serial or "?", firmware or "?", self._usb or "?", self.color_res[0], self.color_res[1],
            dw, dh, infrared, self.fps, min_z, self._depth_scale_m, self.align_to_color,
            [type(f).__name__ for f in self._filters],
        )
        if _depth_camera_model(self._device_name) == "D415" and (dw, dh) == (1280, 720):
            self.logger.warning(
                "%s streams depth at 1280x720, where a D415 measures nothing nearer than about %.0f mm: a surface "
                "closer than that is a hole. A D415 on the wrist works nearer than that; depth_resolution "
                "[848, 480] brings Min-Z to about %.0f mm, with color_resolution kept at [1280, 720] and depth "
                "aligned to it (Intel's figures, not measured here).",
                name, realsense_min_depth_mm("D415", 1280), realsense_min_depth_mm("D415", 848))
        if self._usb is not None and self._usb.startswith("2"):
            self.logger.warning(
                "%s is on a USB %s link. A D400 camera offers fewer modes and lower frame rates over USB 2 than "
                "over USB 3; on an arm this is usually the cable or an extension along it.", name, self._usb)

    def _say_what_the_colour_holds(self) -> None:
        """One line of what the colour sensor holds once the warm-up frames have run and what the rig's
        ``realsense.color`` block wrote, and a warning for each value written that the sensor holds otherwise: it took
        the nearest value it offers, and the frames are taken with that one."""
        name = self._device_name or "the RealSense"
        if self._color_sensor is None:
            self.logger.info(
                "RealSense rig %r: %s has no colour sensor of its own to read (realsense.color is null, so nothing "
                "was asked of one)", self.cfg.rig_id, name)
            return
        held = ", ".join(_said_colour(key, self._color_held.get(key)) for key, _ in _COLOR_OPTIONS)
        written = ", ".join(_said_colour(key, number) for key, (number, _) in self._color_written.items())
        self.logger.info(
            "RealSense rig %r colour sensor holds: %s | %s", self.cfg.rig_id, held,
            f"realsense.color wrote {written}" if written else
            "realsense.color wrote nothing (every key null): the sensor keeps what it holds, auto as it powers up")
        for key, (number, span) in self._color_written.items():
            value = self._color_held.get(key)
            if key.startswith("auto_"):
                took = value is not None and bool(value) == bool(number)
            else:
                took = value is not None and abs(value - number) <= max(float(getattr(span, "step", 0.0) or 0.0), 1e-6)
            if took:
                continue
            windows = (" On Windows librealsense sets the colour exposure in powers of two of a second (39, 78, 156, "
                       "312, 625, in 100 microseconds): write the value it holds to have the config say it."
                       if key == "exposure" else "")
            self.logger.warning(
                "RealSense rig %r: the colour sensor of %s holds %s after %s was written (realsense.color.%s); the "
                "frames are taken with what it holds.%s", self.cfg.rig_id, name, _said_colour(key, value),
                _said_colour(key, number), key, windows)

    def _start_refusal(self, rs: Any, exc: BaseException) -> str:
        """Why the pipeline did not start, with every RealSense the SDK sees and the USB link each is on.

        On a rig that records for research the request held both infrared images too, and the refusal says so: a
        camera without them at this mode, or a link that cannot carry four streams, cannot record for research.
        """
        cw, ch = self.color_res
        dw, dh = self.depth_res
        asked = f"colour {cw}x{ch} and depth {dw}x{dh} at {self.fps} fps"
        if self.rs_cfg.record_for_research:
            asked = (f"colour {cw}x{ch}, depth {dw}x{dh} and both infrared images {dw}x{dh} Y8 at {self.fps} fps "
                     "(record_for_research)")
        head = f"RealSense rig {self.cfg.rig_id!r} could not start {asked}: {exc}."
        if self.rs_cfg.record_for_research:
            head += (f" camera.cameras.rigs[{self.cfg.rig_id!r}].realsense.record_for_research asks for both infrared "
                     "images beside colour and depth; a camera that does not stream them at this mode, or a link that "
                     "cannot carry four streams, cannot record for research: switch record_for_research off to stream "
                     "colour and depth alone.")
        seen = _connected_cameras(rs)
        if not seen:
            return f"{head} librealsense sees no RealSense camera: check the cable and run rs-enumerate-devices."
        listed = "; ".join(
            f"{c['name'] or 'a RealSense'} serial {c['serial_number'] or '?'} on USB {c['usb_type_descriptor'] or '?'}"
            for c in seen)
        mine = [c for c in seen if not self.serial or c["serial_number"] == self.serial]
        if not mine:
            return f"{head} librealsense sees no RealSense with serial {self.serial!r}; it sees {listed}."
        usb2 = [c for c in mine if (c["usb_type_descriptor"] or "").startswith("2")]
        if usb2:
            return (
                f"{head} {listed}. A D400 camera on a USB 2 link offers fewer modes and lower frame rates than on "
                "USB 3, so a request valid on USB 3 can fail there; on an arm this is usually the cable or an "
                "extension along it. Connect it over USB 3 (rs-enumerate-devices shows the link as 'Usb Type "
                "Descriptor'), or ask for a mode rs-enumerate-devices lists for this link.")
        known = [c for c in mine if c["usb_type_descriptor"]]
        if known:
            return (f"{head} {listed}: the camera is on USB {known[0]['usb_type_descriptor']}, so the link is not "
                    "the reason. rs-enumerate-devices lists the modes this model offers.")
        return f"{head} {listed}. rs-enumerate-devices lists the modes and the USB link of each camera."

    # ------------------------------------------------------------------
    # Internal: depth conversion
    # ------------------------------------------------------------------

    def _to_millimetres(self, depth_raw: np.ndarray) -> np.ndarray:
        """Convert raw device depth units to a uint16 millimetre map."""
        scale_m = float(self._depth_scale_m or 0.0)
        depth_mm = depth_raw.astype(np.float64) * scale_m * 1000.0
        return np.clip(depth_mm, 0.0, 65535.0).astype(np.uint16)

    @staticmethod
    def _match_color_size(depth: np.ndarray, color: np.ndarray) -> np.ndarray:
        """Resize depth onto the colour grid so the RGBDFrame size invariant holds.

        Only bites when depth and colour differ: decimated depth that was not aligned, or an
        un-aligned native depth resolution. A no-op when the sizes already match, which they do
        whenever depth is aligned, because the alignment runs after the filters. Nearest
        interpolation keeps the depth values intact.
        """
        if depth.shape[:2] != color.shape[:2]:
            h, w = color.shape[:2]
            depth = cv.resize(depth, (w, h), interpolation=cv.INTER_NEAREST)
        return depth

    # ------------------------------------------------------------------
    # Internal: device and filter configuration
    # ------------------------------------------------------------------

    def _configure_depth_sensor(self, rs: Any, sensor: Any) -> None:
        """Apply the visual preset, then the emitter, the laser power and the depth units over it.

        The preset goes first because a D400 preset carries settings of its own, the laser among
        them and, for ``Default`` on a D415, the depth units: a key the rig writes out is written
        over the preset, never under it.
        """
        if self.rs_cfg.visual_preset:
            self._apply_visual_preset(rs, sensor, self.rs_cfg.visual_preset)
        if sensor.supports(rs.option.emitter_enabled):
            sensor.set_option(rs.option.emitter_enabled, 1.0 if self.rs_cfg.enable_emitter else 0.0)
        if self.rs_cfg.laser_power_mw is not None and sensor.supports(rs.option.laser_power):
            sensor.set_option(rs.option.laser_power, float(self.rs_cfg.laser_power_mw))
        if self.rs_cfg.depth_units_m is not None and sensor.supports(rs.option.depth_units):
            sensor.set_option(rs.option.depth_units, float(self.rs_cfg.depth_units_m))

    def _read_back_depth_scale(self, sensor: Any) -> float:
        """The depth scale the device reports once the preset and the units are written.

        Always the device's reading, never the configured value: a scale the device does not stream
        at would scale every depth by a constant factor. A configured ``depth_units_m`` the device did
        not take refuses the open.
        """
        scale = float(sensor.get_depth_scale())
        if not (math.isfinite(scale) and scale > 0.0):
            raise RuntimeError(
                f"RealSense rig {self.cfg.rig_id!r} reports a depth scale of {scale!r} m per unit, which no depth "
                "can be read with.")
        wanted = self.rs_cfg.depth_units_m
        if wanted is not None and not math.isclose(scale, wanted, rel_tol=self._UNITS_REL_TOL):
            raise RuntimeError(
                f"camera.cameras.rigs[{self.cfg.rig_id!r}].realsense.depth_units_m is {wanted:g} m per unit, and the "
                f"device reads back {scale:g} after the visual preset and the units were written, so it streams at "
                f"{scale:g}: it does not offer the option, or did not take this value. Remove depth_units_m to keep "
                "the device's own units, or write a value it takes (rs-enumerate-devices -o lists its range).")
        return scale

    def _configure_color_sensor(self, rs: Any, device: Any) -> None:
        """Write the rig's ``realsense.color`` block to the sensor that streams colour (the owner, 2026-10-09).

        The block's keys in their order, each auto mode before its manual value; a null key writes nothing, and a block
        of nulls asks nothing of the sensor. Every value is checked against the range the sensor offers before the
        first is written, so a refused block leaves the camera as it found it, which matters because the camera keeps
        what is written until it is power-cycled. The sensor is the device's own colour sensor, never its depth sensor:
        a camera whose colour comes off its depth imagers, a D405, has one sensor for both, and an exposure written
        there would change the depth's. That camera, one with no colour sensor librealsense names, and one whose colour
        sensor does not offer an option the block sets, or offers a range the value lies outside, refuse to open with
        a sentence naming the option and the camera: a fixed value was asked for.
        """
        self._color_sensor = self._sensor_streaming_color(device)
        self._color_written = {}
        block = self.rs_cfg.color
        wanted = [(key, name, getattr(block, key)) for key, name in _COLOR_OPTIONS if getattr(block, key) is not None]
        if not wanted:
            return
        where = f"camera.cameras.rigs[{self.cfg.rig_id!r}].realsense.color"
        camera = self._device_name or "the RealSense"
        sensor = self._color_sensor
        if sensor is None:
            raise RuntimeError(
                f"{where} sets {', '.join(key for key, _, _ in wanted)}, and {camera} has no colour sensor of its own "
                "that librealsense names: its colour comes off the depth imagers, or it has none, and an exposure "
                "written there would change the depth's. Leave every key of realsense.color null on this camera.")
        writes: list[tuple[str, Any, float, Any]] = []
        for key, name, value in wanted:
            option = getattr(rs.option, name, None)
            if option is None or not sensor.supports(option):
                raise RuntimeError(
                    f"{where}.{key} is {value}, and the colour sensor of {camera} does not offer {name}, so it cannot "
                    f"hold the value asked for. Remove {key} to leave the sensor as it is (rs-enumerate-devices -o "
                    "lists the options it offers).")
            number = (1.0 if value else 0.0) if isinstance(value, bool) else float(value)
            span = sensor.get_option_range(option)
            if not float(span.min) <= number <= float(span.max):
                raise RuntimeError(
                    f"{where}.{key} is {number:g}, outside the {float(span.min):g} to {float(span.max):g} the colour "
                    f"sensor of {camera} offers for {name}. Write a value in that range, or remove {key} to leave the "
                    "sensor as it is.")
            writes.append((key, option, number, span))
        for key, option, number, span in writes:
            sensor.set_option(option, number)
            self._color_written[key] = (number, span)

    @staticmethod
    def _sensor_streaming_color(device: Any) -> Any | None:
        """The device's own colour sensor, or None where librealsense names none or the one it names is the depth
        sensor (a camera whose colour comes off its depth imagers)."""
        try:
            sensor = device.first_color_sensor()
        except Exception:  # noqa: BLE001 (a device with no colour sensor: a block that asks one refuses on that)
            return None
        is_depth = getattr(sensor, "is_depth_sensor", None)
        try:
            if sensor is None or (callable(is_depth) and bool(is_depth())):
                return None
        except Exception:  # noqa: BLE001 (a sensor that cannot say what it is is not written to)
            return None
        return sensor

    def _read_back_color(self, rs: Any) -> dict[str, float | None]:
        """What the colour sensor holds now, by the key of the rig's ``realsense.color`` block; None for an option it
        does not offer or does not answer. Only ever said and recorded, so a sensor that cannot be read gives None
        rather than an error."""
        held: dict[str, float | None] = {key: None for key, _ in _COLOR_OPTIONS}
        sensor = self._color_sensor
        if sensor is None:
            return held
        for key, name in _COLOR_OPTIONS:
            option = getattr(rs.option, name, None)
            try:
                if option is not None and sensor.supports(option):
                    held[key] = float(sensor.get_option(option))
            except Exception:  # noqa: BLE001 (a read-back is a record, never a reason to lose the stream)
                held[key] = None
        return held

    def _camera_facts(self, rs: Any, profile: Any, device: Any, depth_sensor: Any) -> CameraFacts:
        """What the camera is while it records for research, read once as it opens (:class:`CameraFacts`).

        A camera whose profile holds no infrared stream after all, though both were asked for, refuses to open: the
        recording was asked for, and frames without the infrared images are not it.
        """
        where = f"camera.cameras.rigs[{self.cfg.rig_id!r}].realsense.record_for_research"
        camera = self._device_name or "the RealSense"
        asked = (("color", rs.stream.color, None), ("depth", rs.stream.depth, None),
                 *((name, rs.stream.infrared, index) for name, index in _INFRARED))
        said_as = {"color": "colour image", "depth": "depth", "ir_left": "left infrared image (infrared 1)",
                   "ir_right": "right infrared image (infrared 2)"}
        streams: dict[str, Any] = {}
        for name, kind, index in asked:
            try:
                stream = profile.get_stream(kind) if index is None else profile.get_stream(kind, index)
                streams[name] = stream.as_video_stream_profile()
            except Exception as exc:  # noqa: BLE001 (the SDK's own words are kept in the refusal)
                raise RuntimeError(
                    f"{where} asks for both infrared images, and {camera} streams no {said_as[name]} at this "
                    f"mode ({exc}): it cannot record for research. Switch record_for_research off to stream colour "
                    "and depth alone.") from exc
        intrinsics: dict[str, np.ndarray] = {}
        distortion: dict[str, np.ndarray] = {}
        modes: dict[str, Any] = {}
        for name, stream in streams.items():
            intr = stream.get_intrinsics()
            intrinsics[name] = np.array(
                [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]], dtype=np.float64)
            distortion[name] = np.asarray(getattr(intr, "coeffs", None) or (), dtype=np.float64).reshape(-1)
            model = getattr(intr, "model", None)
            modes[name] = {"width": int(intr.width), "height": int(intr.height),
                           "distortion_model": None if model is None else str(getattr(model, "name", model))}
        modes["color"]["format"], modes["depth"]["format"] = "bgr8", "z16"
        for name, _ in _INFRARED:
            modes[name]["format"] = "y8"
        extrinsics = {
            "depth_to_color": _extrinsics_mm(streams["depth"].get_extrinsics_to(streams["color"])),
            "ir_left_to_color": _extrinsics_mm(streams["ir_left"].get_extrinsics_to(streams["color"])),
            "ir_left_to_ir_right": _extrinsics_mm(streams["ir_left"].get_extrinsics_to(streams["ir_right"])),
        }
        said = {
            "rig_id": self.cfg.rig_id,
            "device": {"name": self._device_name, "serial": _camera_info(rs, device, "serial_number"),
                       "firmware": _camera_info(rs, device, "firmware_version"), "usb": self._usb},
            "fps": self.fps,
            "streams": modes,
            "depth_units_m": self._depth_scale_m,
            "visual_preset": self.rs_cfg.visual_preset,
            "align_depth_to_color": bool(self.align_to_color),
            "post_processing": self.rs_cfg.post_processing.model_dump(),
            "filters": [{"name": type(filt).__name__, "options": _options_of(filt)} for filt in self._filters],
            "depth_sensor": _options_of(depth_sensor),
            "color_sensor": {
                "asked": self.rs_cfg.color.model_dump(),
                "held": dict(self._color_held),
                "options": {} if self._color_sensor is None else _options_of(self._color_sensor),
            },
        }
        return CameraFacts(said=said, depth_units_m=float(self._depth_scale_m or 0.0), intrinsics=intrinsics,
                           distortion=distortion, extrinsics_mm=extrinsics)

    def _apply_visual_preset(self, rs: Any, sensor: Any, preset: str) -> None:
        """Select a depth visual preset by name, matched ignoring case, spaces and underscores.

        The SDK names its presets in snake case, ``high_accuracy``, and the config documents
        ``HighAccuracy``; both name the same preset. The preset enum is device-family specific,
        so an unknown name is logged and skipped rather than raised: a bad preset name must not
        take the stream down.
        """
        if not sensor.supports(rs.option.visual_preset):
            self.logger.warning("Device does not support visual_preset; ignoring %r", preset)
            return
        try:
            enum = rs.rs400_visual_preset
            match = next(
                (m for m in enum.__members__.values() if _preset_key(m.name) == _preset_key(preset)),
                None,
            )
            if match is None:
                self.logger.warning(
                    "Unknown visual_preset %r; leaving device default. The SDK offers %s",
                    preset, sorted(enum.__members__))
                return
            sensor.set_option(rs.option.visual_preset, float(int(match)))
        except Exception as exc:  # pragma: no cover (defensive, on-box only)
            self.logger.warning("Failed to apply visual_preset %r: %s", preset, exc)

    def _build_filters(self, rs: Any) -> list[Any]:
        """Build the depth post-processing chain in librealsense's recommended order.

        Decimation, depth to disparity, spatial, temporal, disparity to depth, hole filling. Spatial
        and temporal smooth disparity, which is what the sensor measures, rather than millimetres
        rounded to the unit. The chain runs before the alignment to colour (see :meth:`grab`).
        """
        pp = self.rs_cfg.post_processing
        filters: list[Any] = []
        self._temporal_at = None
        if pp.decimation:
            dec = rs.decimation_filter()
            dec.set_option(rs.option.filter_magnitude, float(pp.decimation_magnitude))
            filters.append(dec)
        smoothing = pp.spatial or pp.temporal
        if smoothing:
            filters.append(rs.disparity_transform(True))
        if pp.spatial:
            filters.append(rs.spatial_filter())
        if pp.temporal:
            self._temporal_at = len(filters)
            filters.append(rs.temporal_filter())
        if smoothing:
            filters.append(rs.disparity_transform(False))
        if pp.hole_filling:
            hole = rs.hole_filling_filter()
            hole.set_option(rs.option.holes_fill, float(pp.hole_filling_mode))
            filters.append(hole)
        return filters

    def _read_intrinsics(self, rs: Any, profile: Any) -> np.ndarray | None:
        """Read the colour-stream 3x3 K matrix from the active profile and store its distortion.

        A D400 camera ships factory-calibrated and reports ``intr.coeffs``, the 5-term Brown-Conrady
        distortion, right next to fx, fy, ppx and ppy. Storing it is what lets
        :meth:`get_distortion` hand a downstream ArUco or PnP solve the device's own
        coefficients instead of a zero vector, as decision D1 requires.
        """
        try:
            stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = stream.get_intrinsics()
        except Exception as exc:  # pragma: no cover (defensive, on-box only)
            self.logger.warning("Could not read RealSense intrinsics: %s", exc)
            return None
        coeffs = getattr(intr, "coeffs", None)
        self._distortion = (
            np.asarray(coeffs, dtype=np.float64).reshape(-1) if coeffs is not None else None
        )
        return np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def _export_intrinsics(self, k: np.ndarray) -> None:
        """Persist the colour intrinsics to the rig's ``intrinsics.json``."""
        from src.utility import dump_json, ensure_dir

        path = self.cfg.calibration_paths.intrinsics_file
        ensure_dir(self.cfg.calibration_paths.base_dir)
        dump_json(
            {
                "fx": float(k[0, 0]),
                "fy": float(k[1, 1]),
                "cx": float(k[0, 2]),
                "cy": float(k[1, 2]),
                "width": int(self.color_res[0]),
                "height": int(self.color_res[1]),
                # The factory distortion, persisted so `load_intrinsics` hands a PnP solve the
                # device's own coefficients. An empty list where the device reported none.
                "dist": [] if self._distortion is None else [float(c) for c in self._distortion],
            },
            path,
        )
        self.logger.info("Exported RealSense colour intrinsics (+distortion) to %s", path)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> RealSenseRGBDStreamer:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> Literal[False]:
        self.release()
        return False


AnyRGBDStreamer = OpenCvRGBDStreamer | RealSenseRGBDStreamer
