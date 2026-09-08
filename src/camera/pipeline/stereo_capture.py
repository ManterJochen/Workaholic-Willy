from __future__ import annotations

import logging
import shutil
from pathlib import Path

import cv2 as cv

from src.config.schema.camera import (
    CalibrationConfig,
    CameraSystemConfig,
    RGBDDeviceRigConfig,
    SingleDeviceRigConfig,
    StereoMatcherConfig,
    WebcamPairRigConfig,
)
from src.calibration.stereo.config import StereoRigConfig
from src.calibration.stereo.manager import StereoCam3D
from src.camera.orchestration.frame_provider import FrameProvider
from src.camera.setup.devices.stereocamera import StereoVisionCalibrationSingleDevice
from src.camera.setup.devices.webcam import StereoVisionCalibrationWebcams

CameraRig = WebcamPairRigConfig | SingleDeviceRigConfig | RGBDDeviceRigConfig

__all__ = ["StereoCapturePipeline"]


class StereoCapturePipeline:
    """Turns a camera configuration into the runtime objects a caller streams from.

    Resolves which rigs to use, records the stereo calibration images that are missing, and
    builds the ``FrameProvider`` together with the ``StereoCam3D`` that rectifies its stereo
    frames. RGB-D rigs are carried through to the provider but never calibrated here.
    """

    def __init__(
        self,
        camera_config: CameraSystemConfig,
        calibration: CalibrationConfig,
        stereo_matcher: StereoMatcherConfig,
    ) -> None:
        self.camera_config = camera_config
        self.calibration = calibration
        self.stereo_matcher = stereo_matcher
        self.logger = logging.getLogger(__name__)

    def run(
        self,
        rig_id: str | None = None,
        force_record: bool = False,
        clear_existing: bool = False,
    ) -> tuple[FrameProvider, StereoCam3D | None]:
        """Resolve rigs, ensure stereo calibration images exist, and build runtime objects.

        Recording is interactive and blocks until the operator ends the session.

        Args:
            rig_id: Explicit rig to target. ``None`` resolves from the configuration, either
                every enabled rig that probes as available.
            force_record: Re-record calibration images even when enough exist.
            clear_existing: Delete the existing calibration images first.

        Returns:
            ``(FrameProvider, StereoCam3D | None)``. The ``StereoCam3D`` is ``None`` only
            when every target rig is RGB-D.
        """
        target_rigs = self._resolve_target_rigs(rig_id)
        stereo_rig_cfgs: list[StereoRigConfig] = []

        for rig in target_rigs:
            if isinstance(rig, RGBDDeviceRigConfig):
                self.logger.info("%s is RGB-D; skipping stereo calibration", rig.rig_id)
                continue

            if clear_existing:
                self._clear_images(rig)

            enough, count = self._check_images(rig)
            if force_record or not enough:
                self.logger.info("Recording calibration images for %s", rig.rig_id)
                self._record(rig)
            else:
                self.logger.info("Using %d existing calibration pairs for %s", count, rig.rig_id)

            stereo_rig_cfgs.append(
                StereoRigConfig(
                    stereomap_file=rig.calibration_paths.stereo_map_file,
                    left_glob=rig.calibration_paths.left_images_glob,
                    right_glob=rig.calibration_paths.right_images_glob,
                )
            )

        stereo: StereoCam3D | None = None
        if stereo_rig_cfgs:
            stereo = StereoCam3D(
                rigs=stereo_rig_cfgs,
                calibration=self.calibration,
                stereo_matcher=self.stereo_matcher,
                marker_length_mm=self.calibration.marker_length_mm,
                aruco_dict_name=self.calibration.aruco_dict_name,
            )

        return FrameProvider(rigs=target_rigs, stereo=stereo), stereo

    def _resolve_target_rigs(self, rig_id: str | None = None) -> list[CameraRig]:
        enabled = [rig for rig in self.camera_config.rigs if rig.enabled]
        if not enabled:
            raise RuntimeError("No enabled rigs in camera configuration")

        if rig_id is not None:
            return [self._select_enabled_rig(enabled, rig_id)]

        # Every enabled rig that answers. There used to be a `camera.cameras.active_mode` key whose
        # "rig" setting made this open one named camera instead, and this class was its only reader:
        # the grasp path chose its camera three other ways and consulted neither that key nor the id
        # beside it. Two places naming "the" camera and disagreeing is worse than one, so the mode is
        # gone and the id it pointed at became `primary_rig_id`, which is what a cell opens. This is a
        # calibration tool over the whole catalogue, so it keeps working over the catalogue, and a
        # caller that wants one rig passes `rig_id`.
        available = [rig for rig in enabled if self._is_available(rig)]
        if not available:
            raise RuntimeError("No enabled camera rigs are currently available")
        return available

    @staticmethod
    def _select_enabled_rig(enabled: list[CameraRig], rig_id: str) -> CameraRig:
        for rig in enabled:
            if rig.rig_id == rig_id:
                return rig
        raise RuntimeError(f"Rig {rig_id!r} was not found or is disabled")

    def _is_available(self, rig: CameraRig) -> bool:
        if isinstance(rig, WebcamPairRigConfig):
            return self._probe_webcam_pair(rig)
        if isinstance(rig, (SingleDeviceRigConfig, RGBDDeviceRigConfig)):
            return self._probe_device(rig.device_index, rig.backend)
        return False

    def _probe_webcam_pair(self, rig: WebcamPairRigConfig) -> bool:
        if rig.cam_left_id is not None and rig.cam_right_id is not None:
            return self._probe_device(rig.cam_left_id, rig.backend) and self._probe_device(
                rig.cam_right_id, rig.backend
            )
        return len(self._scan_cameras(rig.max_cam_scan, rig.backend)) >= 2

    @staticmethod
    def _probe_device(idx: int, backend: int) -> bool:
        cap = cv.VideoCapture(int(idx), int(backend))
        try:
            if not cap.isOpened():
                return False
            ok, _ = cap.read()
            return bool(ok)
        finally:
            cap.release()

    @staticmethod
    def _scan_cameras(max_id: int, backend: int) -> list[int]:
        ids: list[int] = []
        for idx in range(int(max_id)):
            cap = cv.VideoCapture(idx, int(backend))
            try:
                if cap.isOpened():
                    ok, _ = cap.read()
                    if ok:
                        ids.append(idx)
            finally:
                cap.release()
        return ids

    def _check_images(self, rig: WebcamPairRigConfig | SingleDeviceRigConfig) -> tuple[bool, int]:
        paths = rig.calibration_paths
        left = list(Path(paths.left_images_dir).glob("*.png"))
        right = list(Path(paths.right_images_dir).glob("*.png"))
        count = min(len(left), len(right))
        return count >= rig.min_pairs, count

    def _clear_images(self, rig: WebcamPairRigConfig | SingleDeviceRigConfig) -> None:
        paths = rig.calibration_paths
        for folder in (Path(paths.left_images_dir), Path(paths.right_images_dir)):
            if not folder.is_dir():
                continue
            for child in folder.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()

    def _record(self, rig: WebcamPairRigConfig | SingleDeviceRigConfig) -> None:
        if isinstance(rig, WebcamPairRigConfig):
            StereoVisionCalibrationWebcams(wc_cfg=rig).forward()
            return
        if isinstance(rig, SingleDeviceRigConfig):
            StereoVisionCalibrationSingleDevice(dev_cfg=rig).forward()
            return
        raise ValueError(f"Cannot record calibration images for rig type: {type(rig)!r}")