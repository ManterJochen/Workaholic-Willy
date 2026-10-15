from __future__ import annotations

import cv2 as cv
import logging

from src.config.schema.camera import WebcamPairRigConfig
from src.camera.setup.capture_backend import refuse_a_backend_this_platform_lacks
from src.camera.setup.image_taking.frames import StereoFrame
from src.camera.setup.quality import configure_camera_for_quality


class WebcamPairStreamer:
    """Streams a stereo pair from two separate USB cameras.

    Both cameras get the same settings from ``configure_camera_for_quality``, and a pair
    whose resolution or frame rate disagrees is refused. Capture grabs from both cameras
    before either is retrieved, which keeps the offset between the two exposures small.
    """

    def __init__(self, config: WebcamPairRigConfig):
        refuse_a_backend_this_platform_lacks(config)
        self.cfg = config
        self.logger = logging.getLogger(f"{__name__}.{config.rig_id}")

        self.frame_size = tuple(config.frame_size)
        self.fps = config.fps
        self.backend = config.backend

        self._cap_left: cv.VideoCapture | None = None
        self._cap_right: cv.VideoCapture | None = None

        # Configured ids win. Otherwise scan, and take the last two indices that
        # returned a frame as left and right.
        if config.cam_left_id is not None and config.cam_right_id is not None:
            self._left_id = int(config.cam_left_id)
            self._right_id = int(config.cam_right_id)
        else:
            ids = self._scan_cameras(config.max_cam_scan)
            if len(ids) < 2:
                raise OSError("Not enough cameras detected for stereo pair.")
            self._left_id = ids[-2]
            self._right_id = ids[-1]
            self.logger.info("Auto-selected cameras: L=%d, R=%d", self._left_id, self._right_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open both cameras and apply quality settings.

        Either camera failing to open releases both and raises OSError.
        """
        self._cap_left = cv.VideoCapture(self._left_id, self.backend)
        self._cap_right = cv.VideoCapture(self._right_id, self.backend)

        if not self._cap_left.isOpened() or not self._cap_right.isOpened():
            self.release()
            raise OSError(
                f"Could not open webcam pair L={self._left_id}, R={self._right_id}"
            )

        left_settings = self._apply_quality(self._cap_left, "Left")
        right_settings = self._apply_quality(self._cap_right, "Right")
        self._validate_pair(left_settings, right_settings)
        self.logger.info("Webcam pair opened and configured.")

    def release(self) -> None:
        """Release both camera resources."""
        for cap in (self._cap_left, self._cap_right):
            if cap is not None and cap.isOpened():
                cap.release()
        self._cap_left = None
        self._cap_right = None

    def is_opened(self) -> bool:
        return (
            self._cap_left is not None
            and self._cap_right is not None
            and self._cap_left.isOpened()
            and self._cap_right.isOpened()
        )

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def grab(self) -> StereoFrame:
        """Grab one frame from each camera and return the two as a StereoFrame.

        Raises RuntimeError if the cameras are not open, or if a grab or a retrieve fails.
        """
        if not self.is_opened():
            raise RuntimeError("Cameras are not open. Call open() first.")

        # guaranteed non-None by is_opened() above
        assert self._cap_left is not None
        assert self._cap_right is not None
        ok_l = self._cap_left.grab()
        ok_r = self._cap_right.grab()

        if not ok_l or not ok_r:
            raise RuntimeError("Failed to grab frames from webcam pair.")

        _, left = self._cap_left.retrieve()
        _, right = self._cap_right.retrieve()

        if left is None or right is None:
            raise RuntimeError("Failed to retrieve frames from webcam pair.")

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

    def _scan_cameras(self, max_id: int) -> list[int]:
        ids: list[int] = []
        for i in range(max_id):
            cap = cv.VideoCapture(i, self.backend)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    ids.append(i)
            cap.release()
        return ids

    def _apply_quality(self, cap: cv.VideoCapture, side: str) -> dict:
        q = self.cfg.quality
        settings = configure_camera_for_quality(
            cap=cap,
            width=self.frame_size[0],
            height=self.frame_size[1],
            fps=self.fps,
            prefer_uncompressed=q.prefer_uncompressed,
            manual_exposure=q.manual_exposure,
            manual_gain=q.manual_gain,
            manual_wb=q.manual_wb,
            disable_auto_features=q.disable_auto_features,
            warmup_frames=q.warmup_frames,
        )
        self.logger.info(
            "%s camera: FOURCC=%s %dx%d @ %.1f fps",
            side,
            settings["fourcc_str"],
            settings["width"],
            settings["height"],
            settings["fps"],
        )
        return settings

    def _validate_pair(self, left: dict, right: dict) -> None:
        w, h = self.frame_size
        tol = self.cfg.quality.fps_tolerance

        if (left["width"], left["height"]) != (w, h):
            raise ValueError(
                f"Left camera resolution mismatch: "
                f"{left['width']}x{left['height']} != {w}x{h}"
            )
        if (right["width"], right["height"]) != (w, h):
            raise ValueError(
                f"Right camera resolution mismatch: "
                f"{right['width']}x{right['height']} != {w}x{h}"
            )
        if abs(left["fps"] - right["fps"]) > tol:
            raise ValueError(
                f"FPS mismatch: L={left['fps']:.1f}, R={right['fps']:.1f}, tol={tol}"
            )
