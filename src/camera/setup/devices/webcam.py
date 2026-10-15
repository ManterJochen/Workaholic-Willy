import os
import logging

import cv2 as cv

from src.config.schema.camera import WebcamPairRigConfig
from src.camera.setup.capture_backend import refuse_a_backend_this_platform_lacks
from src.camera.setup.quality import configure_camera_for_quality


class StereoVisionCalibrationWebcams:
    """Records stereo calibration pairs from two separate webcams.

    Opens both cameras and configures each through ``configure_camera_for_quality``. A pair
    whose resolution or frame rate disagrees is refused; a pixel format that differs is only
    logged. Capture grabs from both cameras before either is retrieved, which keeps the offset
    between the two exposures small. It is interactive: ``s`` saves the pair on screen and ESC
    ends the session. The saved images are the raw frames, so the operator overlay is drawn on
    preview copies only.
    """

    def __init__(self, wc_cfg: WebcamPairRigConfig):
        refuse_a_backend_this_platform_lacks(wc_cfg)
        self.cfg = wc_cfg
        self.logger = logging.getLogger(f"{__name__}.{wc_cfg.rig_id}")

        self.img_left = wc_cfg.calibration_paths.left_images_dir
        self.img_right = wc_cfg.calibration_paths.right_images_dir

        self.frame_size = tuple(wc_cfg.frame_size)
        self.fps = int(wc_cfg.fps)

        self.min_pairs = int(wc_cfg.min_pairs)
        self.max_pairs = int(wc_cfg.max_pairs)

        self.backend = wc_cfg.backend
        self.max_cam_scan = int(wc_cfg.max_cam_scan)
        self.fixed_left_id: int | None = wc_cfg.cam_left_id
        self.fixed_right_id: int | None = wc_cfg.cam_right_id

        self.prefer_uncompressed = wc_cfg.quality.prefer_uncompressed
        self.manual_exposure = wc_cfg.quality.manual_exposure
        self.manual_gain = wc_cfg.quality.manual_gain
        self.manual_wb = wc_cfg.quality.manual_wb
        self.disable_auto_features = wc_cfg.quality.disable_auto_features
        self.warmup_frames = wc_cfg.quality.warmup_frames
        self.fps_tolerance = wc_cfg.quality.fps_tolerance

        if self.max_pairs < 1:
            raise ValueError("max_pairs must be >= 1")
        if self.min_pairs < 0:
            raise ValueError("min_pairs must be >= 0")
        if self.min_pairs > self.max_pairs:
            raise ValueError("min_pairs cannot be greater than max_pairs")

        if (
            self.fixed_left_id is not None and self.fixed_left_id >= 0 and
            self.fixed_right_id is not None and self.fixed_right_id >= 0
        ):
            self.cam_left = int(self.fixed_left_id)
            self.cam_right = int(self.fixed_right_id)
            self.logger.info(
                f"Using fixed camera IDs: L={self.cam_left}, R={self.cam_right}"
            )
        else:
            cam_ids = self.find_cams(self.max_cam_scan)
            if len(cam_ids) < 2:
                raise OSError("Not enough cameras detected for stereo capture.")
            cam_ids = cam_ids[-2:]
            self.cam_left = cam_ids[0]
            self.cam_right = cam_ids[1]
            self.logger.info(
                f"Auto-selected cameras: L={self.cam_left}, R={self.cam_right}"
            )

    def find_cams(self, max_ID: int = 10) -> list[int]:
        cam_ids: list[int] = []

        for i in range(max_ID):
            cap = cv.VideoCapture(i, self.backend)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cam_ids.append(i)
            cap.release()

        return cam_ids

    def _configure_camera(self, cap: cv.VideoCapture, side: str) -> dict:
        settings = configure_camera_for_quality(
            cap=cap,
            width=self.frame_size[0],
            height=self.frame_size[1],
            fps=self.fps,
            prefer_uncompressed=self.prefer_uncompressed,
            manual_exposure=self.manual_exposure,
            manual_gain=self.manual_gain,
            manual_wb=self.manual_wb,
            disable_auto_features=self.disable_auto_features,
            warmup_frames=self.warmup_frames
        )

        self.logger.info(
            f"{side} camera settings: FOURCC={settings['fourcc_str']} "
            f"size={settings['width']}x{settings['height']} "
            f"fps={settings['fps']:.2f} "
            f"exp={settings.get('exposure')} gain={settings.get('gain')} "
            f"wb={settings.get('wb_blue_u')}"
        )
        return settings

    def _validate_stereo_settings(self, left: dict, right: dict) -> bool:
        expected_w, expected_h = self.frame_size

        if (left["width"], left["height"]) != (expected_w, expected_h):
            raise ValueError(
                f"Left camera resolution mismatch: {left['width']}x{left['height']} "
                f"!= {expected_w}x{expected_h}"
            )

        if (right["width"], right["height"]) != (expected_w, expected_h):
            raise ValueError(
                f"Right camera resolution mismatch: {right['width']}x{right['height']} "
                f"!= {expected_w}x{expected_h}"
            )

        if left.get("fourcc_str") != right.get("fourcc_str"):
            self.logger.warning(
                f"FOURCC mismatch: L={left.get('fourcc_str')} vs R={right.get('fourcc_str')}"
            )

        if abs(left["fps"] - right["fps"]) > self.fps_tolerance:
            raise ValueError(
                f"FPS mismatch too high: L={left['fps']:.2f}, R={right['fps']:.2f}, "
                f"tolerance={self.fps_tolerance}"
            )

        return True

    def _take_pictures(self, cam_left: cv.VideoCapture, cam_right: cv.VideoCapture):
        os.makedirs(self.img_left, exist_ok=True)
        os.makedirs(self.img_right, exist_ok=True)

        num = 0
        font = cv.FONT_HERSHEY_SIMPLEX

        while cam_left.isOpened() and cam_right.isOpened():
            okL_grab = cam_left.grab()
            okR_grab = cam_right.grab()

            if not okL_grab or not okR_grab:
                raise InterruptedError("Failed to grab frames from both cameras.")

            successL, rawL = cam_left.retrieve()
            successR, rawR = cam_right.retrieve()

            if not successL or not successR or rawL is None or rawR is None:
                raise InterruptedError("Failed to retrieve frames from cameras.")

            previewL = rawL.copy()
            previewR = rawR.copy()

            overlay_text = (
                f"Stereo pairs: {num}/{self.max_pairs} (min={self.min_pairs}) | "
                f"'s' save | ESC exit"
            )

            cv.putText(previewL, f"L | {overlay_text}", (12, 28), font, 0.6, (0, 255, 0), 2, cv.LINE_AA)
            cv.putText(previewR, f"R | {overlay_text}", (12, 28), font, 0.6, (0, 255, 0), 2, cv.LINE_AA)

            cv.imshow(f"{self.cfg.rig_id} - Img left", previewL)
            cv.imshow(f"{self.cfg.rig_id} - Img right", previewR)

            if num >= self.max_pairs:
                self.logger.info(f"Reached max_pairs={self.max_pairs}. Stopping capture.")
                break

            key = cv.waitKey(5) & 0xFF

            if key == 27:
                if num < self.min_pairs:
                    self.logger.warning(
                        f"Capture aborted before min_pairs: {num}/{self.min_pairs}"
                    )
                break

            elif key == ord('s'):
                left_path = os.path.join(self.img_left, f"imageL{num}.png")
                right_path = os.path.join(self.img_right, f"imageR{num}.png")

                retL = cv.imwrite(left_path, rawL)
                retR = cv.imwrite(right_path, rawR)

                if not retL or not retR:
                    raise OSError("Failed to save stereo image pair.")

                self.logger.info(f"Saved stereo pair {num}: {left_path} | {right_path}")
                print(f"Saved stereo pair {num}")
                num += 1

        cam_left.release()
        cam_right.release()
        cv.destroyAllWindows()

        if num < self.min_pairs:
            self.logger.warning(
                f"Session ended with {num} images < min_pairs={self.min_pairs}"
            )

    def forward(self):
        """Open both cameras, check the pair agrees, and run the capture session."""
        os.makedirs(self.img_left, exist_ok=True)
        os.makedirs(self.img_right, exist_ok=True)

        capL = cv.VideoCapture(self.cam_left, self.backend)
        capR = cv.VideoCapture(self.cam_right, self.backend)

        if not capL.isOpened() or not capR.isOpened():
            raise OSError("Could not open webcams.")

        left_settings = self._configure_camera(capL, "Left")
        right_settings = self._configure_camera(capR, "Right")
        self._validate_stereo_settings(left_settings, right_settings)
        self._take_pictures(cam_left=capL, cam_right=capR)
