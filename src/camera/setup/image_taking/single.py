from __future__ import annotations

import cv2 as cv
import logging
import numpy as np

from src.config.schema.camera import SingleDeviceRigConfig
from src.camera.setup.capture_backend import refuse_a_backend_this_platform_lacks
from src.camera.setup.image_taking.frames import StereoFrame
from src.camera.setup.quality import configure_camera_for_quality


class SingleDeviceStreamer:
    """Streams both eyes from one device that delivers them in a single frame.

    ``layout`` says whether the two eyes sit side by side or one above the other.
    Quality settings come from ``configure_camera_for_quality``. The configured crop is
    applied before the split, and each eye is resized to ``per_eye_frame_size`` when
    ``allow_resize`` permits it.
    """

    def __init__(self, config: SingleDeviceRigConfig):
        refuse_a_backend_this_platform_lacks(config)
        self.cfg = config
        self.logger = logging.getLogger(f"{__name__}.{config.rig_id}")

        self.device_index = int(config.device_index)
        self.device_frame_size = tuple(config.device_frame_size)
        self.per_eye_frame_size = tuple(config.per_eye_frame_size)
        self.layout = config.layout
        self.fps = config.fps
        self.backend = config.backend

        self.crop_left = int(config.crop_left)
        self.crop_right = int(config.crop_right)
        self.crop_top = int(config.crop_top)
        self.crop_bottom = int(config.crop_bottom)
        self.allow_resize = config.allow_resize

        self._cap: cv.VideoCapture | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the stereo device and apply quality settings."""
        self._cap = cv.VideoCapture(self.device_index, self.backend)

        if not self._cap.isOpened():
            raise OSError(f"Could not open stereo device index {self.device_index}")

        q = self.cfg.quality
        settings = configure_camera_for_quality(
            cap=self._cap,
            width=self.device_frame_size[0],
            height=self.device_frame_size[1],
            fps=self.fps,
            prefer_uncompressed=q.prefer_uncompressed,
            manual_exposure=q.manual_exposure,
            manual_gain=q.manual_gain,
            manual_wb=q.manual_wb,
            disable_auto_features=q.disable_auto_features,
            warmup_frames=q.warmup_frames,
        )
        self._validate_settings(settings)
        self.logger.info(
            "Stereo device %d opened: FOURCC=%s %dx%d @ %.1f fps",
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

    def grab(self) -> StereoFrame:
        """Grab one combined frame, split it, and return both eyes as a StereoFrame.

        Raises RuntimeError if the device is not open, or if the grab or the retrieve
        fails.
        """
        if not self.is_opened():
            raise RuntimeError("Device is not open. Call open() first.")

        assert self._cap is not None  # guaranteed by is_opened() above
        ok = self._cap.grab()
        if not ok:
            raise RuntimeError("Failed to grab frame from stereo device.")

        _, raw = self._cap.retrieve()
        if raw is None:
            raise RuntimeError("Failed to retrieve frame from stereo device.")

        left, right = self._split_frame(raw)
        left, right = self._ensure_eye_size(left, right)
        return StereoFrame(left=left, right=right)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate_settings(self, settings: dict) -> None:
        w, h = self.device_frame_size
        if (settings["width"], settings["height"]) != (w, h):
            raise ValueError(
                f"Device resolution mismatch: "
                f"{settings['width']}x{settings['height']} != {w}x{h}"
            )
        tol = self.cfg.quality.fps_tolerance
        if abs(settings["fps"] - self.fps) > tol:
            self.logger.warning(
                "FPS differs: actual=%.1f, requested=%d, tol=%.1f",
                settings["fps"], self.fps, tol,
            )

    def _apply_crop(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        x0 = max(0, self.crop_left)
        x1 = max(0, w - self.crop_right)
        y0 = max(0, self.crop_top)
        y1 = max(0, h - self.crop_bottom)

        if x0 >= x1 or y0 >= y1:
            self.logger.warning("Invalid crop values, skipping crop.")
            return frame

        return frame[y0:y1, x0:x1]

    def _split_frame(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        frame = self._apply_crop(frame)
        h, w = frame.shape[:2]

        if self.layout == "horizontal":
            mid = w // 2
            return frame[:, :mid], frame[:, mid:]
        else:
            mid = h // 2
            return frame[:mid, :], frame[mid:, :]

    def _ensure_eye_size(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # per_eye_frame_size is (width, height), which is the order cv.resize wants.
        # Do not swap it against the (height, width) that shape[:2] returns.
        tw, th = self.per_eye_frame_size

        def _resize_if_needed(img: np.ndarray, side: str) -> np.ndarray:
            ih, iw = img.shape[:2]
            if (iw, ih) == (tw, th):
                return img
            if not self.allow_resize:
                raise ValueError(
                    f"{side} eye size {iw}x{ih} != {tw}x{th} and allow_resize=False"
                )
            self.logger.debug("Resizing %s eye %dx%d -> %dx%d", side, iw, ih, tw, th)
            return cv.resize(img, (tw, th), interpolation=cv.INTER_AREA)

        return _resize_if_needed(left, "Left"), _resize_if_needed(right, "Right")
