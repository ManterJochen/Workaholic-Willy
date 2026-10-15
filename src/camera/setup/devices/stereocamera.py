import os
import logging

import cv2 as cv

from src.config.schema.camera import SingleDeviceRigConfig
from src.camera.setup.capture_backend import refuse_a_backend_this_platform_lacks
from src.camera.setup.quality import configure_camera_for_quality


class StereoVisionCalibrationSingleDevice:
    """Records stereo calibration pairs from one device that delivers both eyes in a frame.

    Opens the device, configures its stream through ``configure_camera_for_quality`` and
    checks the resolution and frame rate it reports back. Each frame is cropped, split into a
    left and a right eye, and checked against the configured per-eye size. Capture is
    interactive: ``s`` saves the pair on screen and ESC ends the session. Where
    ``min_sharpness`` is set, a pair whose Laplacian variance falls below it is not saved.
    The saved images are the raw split, so the operator overlay goes on preview copies only.
    """

    def __init__(self, dev_cfg: SingleDeviceRigConfig):
        refuse_a_backend_this_platform_lacks(dev_cfg)
        self.cfg = dev_cfg

        self.device_index = int(dev_cfg.device_index)
        self.img_left_dir = dev_cfg.calibration_paths.left_images_dir
        self.img_right_dir = dev_cfg.calibration_paths.right_images_dir

        self.device_frame_size = tuple(dev_cfg.device_frame_size)
        self.per_eye_frame_size = tuple(dev_cfg.per_eye_frame_size)

        self.fps = int(dev_cfg.fps)
        self.min_pairs = int(dev_cfg.min_pairs)
        self.max_pairs = int(dev_cfg.max_pairs)

        self.layout = dev_cfg.layout
        self.crop_left = int(dev_cfg.crop_left)
        self.crop_right = int(dev_cfg.crop_right)
        self.crop_top = int(dev_cfg.crop_top)
        self.crop_bottom = int(dev_cfg.crop_bottom)

        self.backend = dev_cfg.backend

        self.prefer_uncompressed = dev_cfg.quality.prefer_uncompressed
        self.manual_exposure = dev_cfg.quality.manual_exposure
        self.manual_gain = dev_cfg.quality.manual_gain
        self.manual_wb = dev_cfg.quality.manual_wb
        self.disable_auto_features = dev_cfg.quality.disable_auto_features
        self.warmup_frames = dev_cfg.quality.warmup_frames
        self.fps_tolerance = dev_cfg.quality.fps_tolerance

        self.allow_resize = dev_cfg.allow_resize
        self.min_sharpness = dev_cfg.min_sharpness

        self.logger = logging.getLogger(f"{__name__}.{self.cfg.rig_id}")

        if self.max_pairs < 1:
            raise ValueError("max_pairs must be >= 1")
        if self.min_pairs < 0:
            raise ValueError("min_pairs must be >= 0")
        if self.min_pairs > self.max_pairs:
            raise ValueError("min_pairs cannot be greater than max_pairs")

        if self.layout not in ("horizontal", "vertical"):
            raise ValueError("layout must be 'horizontal' or 'vertical'")

    def _open_and_configure_device(self):
        cap = cv.VideoCapture(self.device_index, self.backend)

        if not cap.isOpened():
            raise OSError(f"Could not open stereo device index {self.device_index}.")

        settings = configure_camera_for_quality(
            cap=cap,
            width=self.device_frame_size[0],
            height=self.device_frame_size[1],
            fps=self.fps,
            prefer_uncompressed=self.prefer_uncompressed,
            manual_exposure=self.manual_exposure,
            manual_gain=self.manual_gain,
            manual_wb=self.manual_wb,
            disable_auto_features=self.disable_auto_features,
            warmup_frames=self.warmup_frames
        )

        self.logger.info(
            f"Configured device {self.device_index}: FOURCC={settings.get('fourcc_str')} "
            f"size={settings.get('width')}x{settings.get('height')} "
            f"fps={settings.get('fps', 0.0):.2f}"
        )
        return cap, settings

    def _validate_device_settings(self, settings: dict) -> bool:
        expected_w, expected_h = self.device_frame_size

        if (settings["width"], settings["height"]) != (expected_w, expected_h):
            raise ValueError(
                f"Stereo device frame mismatch: "
                f"{settings['width']}x{settings['height']} != {expected_w}x{expected_h}"
            )

        if abs(settings["fps"] - self.fps) > self.fps_tolerance:
            self.logger.warning(
                f"FPS differs from requested value: actual={settings['fps']:.2f}, "
                f"requested={self.fps}, tolerance={self.fps_tolerance}"
            )

        return True

    def _apply_crop(self, frame):
        h, w = frame.shape[:2]
        x0 = max(0, self.crop_left)
        x1 = max(0, w - self.crop_right)
        y0 = max(0, self.crop_top)
        y1 = max(0, h - self.crop_bottom)

        if x0 >= x1 or y0 >= y1:
            self.logger.warning("Invalid crop values detected. Ignoring crop.")
            return frame

        return frame[y0:y1, x0:x1]

    def _split_frame(self, frame):
        frame = self._apply_crop(frame)
        H, W = frame.shape[:2]

        if self.layout == "horizontal":
            mid = W // 2
            imgL = frame[:, :mid]
            imgR = frame[:, mid:]
        else:
            mid = H // 2
            imgL = frame[:mid, :]
            imgR = frame[mid:, :]

        return imgL, imgR

    def _check_per_eye_size(self, imgL, imgR):
        target_w, target_h = self.per_eye_frame_size

        def _ensure_size(img, side_name: str):
            h, w = img.shape[:2]
            if (w, h) == (target_w, target_h):
                return img

            if not self.allow_resize:
                raise ValueError(
                    f"{side_name} eye image has wrong size: {w}x{h}, "
                    f"expected {target_w}x{target_h}, allow_resize=False"
                )

            self.logger.warning(
                f"{side_name} eye image size mismatch: {w}x{h} -> resizing to {target_w}x{target_h}"
            )
            return cv.resize(img, (target_w, target_h), interpolation=cv.INTER_AREA)

        return _ensure_size(imgL, "Left"), _ensure_size(imgR, "Right")

    def _sharpness_score(self, img) -> float:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return float(cv.Laplacian(gray, cv.CV_64F).var())

    def _is_sharp_enough(self, imgL, imgR) -> bool:
        if self.min_sharpness is None:
            return True

        sharpL = self._sharpness_score(imgL)
        sharpR = self._sharpness_score(imgR)

        self.logger.info(
            f"Sharpness scores: left={sharpL:.2f}, right={sharpR:.2f}, "
            f"threshold={self.min_sharpness}"
        )

        return sharpL >= self.min_sharpness and sharpR >= self.min_sharpness

    def _capture_pairs(self, cap):
        os.makedirs(self.img_left_dir, exist_ok=True)
        os.makedirs(self.img_right_dir, exist_ok=True)

        num = 0
        font = cv.FONT_HERSHEY_SIMPLEX

        while cap.isOpened():
            ok_grab = cap.grab()
            if not ok_grab:
                raise InterruptedError("Failed to grab frame from stereo device.")

            ok, raw_frame = cap.retrieve()
            if not ok or raw_frame is None:
                raise InterruptedError("Failed to retrieve frame from stereo device.")

            imgL_raw, imgR_raw = self._split_frame(raw_frame)
            imgL, imgR = self._check_per_eye_size(imgL_raw, imgR_raw)

            full_preview = raw_frame.copy()
            previewL = imgL.copy()
            previewR = imgR.copy()

            overlay_text = (
                f"Stereo pairs: {num}/{self.max_pairs} (min={self.min_pairs}) | s=save, ESC=exit"
            )

            cv.putText(full_preview, overlay_text, (12, 28), font, 0.6, (0, 255, 0), 2, cv.LINE_AA)
            cv.putText(previewL, "LEFT", (10, 26), font, 0.7, (255, 200, 0), 2, cv.LINE_AA)
            cv.putText(previewR, "RIGHT", (10, 26), font, 0.7, (255, 200, 0), 2, cv.LINE_AA)

            cv.imshow(f"{self.cfg.rig_id} - Full", full_preview)
            cv.imshow(f"{self.cfg.rig_id} - Left", previewL)
            cv.imshow(f"{self.cfg.rig_id} - Right", previewR)

            if num >= self.max_pairs:
                self.logger.info("Reached max_pairs. Ending session.")
                break

            k = cv.waitKey(5) & 0xFF

            if k == 27:
                if num < self.min_pairs:
                    self.logger.warning(
                        f"Capture aborted before min_pairs: {num}/{self.min_pairs}"
                    )
                break

            elif k == ord('s'):
                if not self._is_sharp_enough(imgL, imgR):
                    self.logger.warning("Skipped pair because images are too blurry.")
                    continue

                lp = os.path.join(self.img_left_dir, f"imageL{num}.png")
                rp = os.path.join(self.img_right_dir, f"imageR{num}.png")

                okL = cv.imwrite(lp, imgL)
                okR = cv.imwrite(rp, imgR)

                if not okL or not okR:
                    raise OSError("Failed to save split stereo images.")

                print(f"Saved stereo pair {num} -> {lp} | {rp}")
                self.logger.info(f"Saved stereo pair #{num}: {lp} | {rp}")
                num += 1

        cap.release()
        cv.destroyAllWindows()

        if num < self.min_pairs:
            self.logger.warning(
                f"Session finished with {num}/{self.min_pairs} pairs."
            )

    def forward(self):
        """Open the device, check its reported settings, and run the capture session."""
        os.makedirs(self.img_left_dir, exist_ok=True)
        os.makedirs(self.img_right_dir, exist_ok=True)

        cap, settings = self._open_and_configure_device()
        self._validate_device_settings(settings)
        self._capture_pairs(cap)
