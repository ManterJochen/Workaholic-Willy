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
from src.camera.setup.image_taking.frames import RGBDFrame
from src.camera.setup.quality import configure_camera_for_quality


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
        """Start the pipeline, configure the depth sensor and filters, then read intrinsics.

        A request the SDK cannot start is raised with the RealSense cameras it sees and the USB link
        each is on. A depth sensor that did not take the configured ``depth_units_m`` refuses the open.
        Either way the pipeline is not left running.
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

        self._align = rs.align(rs.stream.color) if self.align_to_color else None
        self._filters = self._build_filters(rs)
        self._last_grab_s = None

        assert self._pipeline is not None  # set by open() before it calls this
        for _ in range(self.cfg.quality.warmup_frames):
            self._pipeline.wait_for_frames()

        self._intrinsics = self._read_intrinsics(rs, profile)
        if self.rs_cfg.export_intrinsics and self._intrinsics is not None:
            self._export_intrinsics(self._intrinsics)

        self._say_what_opened(device, rs)

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

    def is_opened(self) -> bool:
        return self._pipeline is not None

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def grab(self) -> RGBDFrame:
        """Grab a colour and depth pair, filtered and aligned as configured, as an RGBDFrame.

        On a camera the arm carries, a grab that follows the previous one by more than a burst
        takes starts the temporal filter afresh (see the class docstring).
        """
        if self._pipeline is None:
            raise RuntimeError("Device is not open. Call open() first.")

        now = self._clock()
        if self.carried and self._last_grab_s is not None and now - self._last_grab_s > self._pause_s():
            self.camera_moved()
        self._last_grab_s = now

        frameset = self._pipeline.wait_for_frames()
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
        return RGBDFrame(color=color, depth=depth_mm)

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
        self.logger.info(
            "RealSense opened: %s (serial %s, firmware %s, USB %s) | colour %dx%d + depth %dx%d @ %d fps | %s | "
            "depth_scale=%.6g m/unit read back from the device | align=%s | filters=%s",
            name, serial or "?", firmware or "?", self._usb or "?", self.color_res[0], self.color_res[1],
            dw, dh, self.fps, min_z, self._depth_scale_m, self.align_to_color,
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

    def _start_refusal(self, rs: Any, exc: BaseException) -> str:
        """Why the pipeline did not start, with every RealSense the SDK sees and the USB link each is on."""
        cw, ch = self.color_res
        dw, dh = self.depth_res
        asked = f"colour {cw}x{ch} and depth {dw}x{dh} at {self.fps} fps"
        head = f"RealSense rig {self.cfg.rig_id!r} could not start {asked}: {exc}."
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
