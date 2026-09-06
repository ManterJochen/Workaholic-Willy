from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.config.schema.camera import RGBDDeviceRigConfig, SingleDeviceRigConfig, WebcamPairRigConfig
from src.camera import RGBDFrame as RootRGBDFrame
from src.camera import StereoFrame as RootStereoFrame
from src.camera.orchestration import FrameProvider, FrameProviderStateError, UnknownCameraRigError
from src.camera.pipeline import StereoCapturePipeline
from src.camera.setup.image_taking import RGBDFrame, StereoFrame
from src.camera.setup.image_taking.rgbd import OpenCvRGBDStreamer
from src.camera.setup.image_taking.single import SingleDeviceStreamer
from src.camera.setup.image_taking.frames import RGBDFrame as CanonicalRGBDFrame
from src.camera.setup.image_taking.frames import StereoFrame as CanonicalStereoFrame


class FakeStreamer:
    def __init__(self, frame: StereoFrame | RGBDFrame) -> None:
        self.frame = frame
        self.opened = False
        self.released = False

    def open(self) -> None:
        self.opened = True

    def release(self) -> None:
        self.released = True
        self.opened = False

    def grab(self) -> StereoFrame | RGBDFrame:
        if not self.opened:
            raise RuntimeError("streamer is closed")
        return self.frame


class FakeStereo:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def rectify(self, left: np.ndarray, right: np.ndarray, *, rig: int) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append(rig)
        return left + 1, right + 1


def _stereo_frame(value: int = 1) -> StereoFrame:
    left = np.full((3, 4, 3), value, dtype=np.uint8)
    right = np.full((3, 4, 3), value + 1, dtype=np.uint8)
    return StereoFrame(left=left, right=right)


def _rgbd_frame() -> RGBDFrame:
    color = np.zeros((3, 4, 3), dtype=np.uint8)
    depth = np.zeros((3, 4), dtype=np.uint16)
    return RGBDFrame(color=color, depth=depth)


def _webcam_rig(rig_id: str, base_dir: str = "calibration/webcam") -> WebcamPairRigConfig:
    return WebcamPairRigConfig(
        source="webcam_pair",
        rig_id=rig_id,
        enabled=True,
        fps=30,
        backend=0,
        frame_size=(4, 3),
        max_cam_scan=4,
        cam_left_id=0,
        cam_right_id=1,
        min_pairs=2,
        max_pairs=4,
        calibration_paths={"base_dir": base_dir},
    )


def _single_rig(rig_id: str, base_dir: str = "calibration/single") -> SingleDeviceRigConfig:
    return SingleDeviceRigConfig(
        source="single_device",
        rig_id=rig_id,
        enabled=True,
        fps=30,
        backend=0,
        device_index=2,
        device_frame_size=(8, 3),
        per_eye_frame_size=(4, 3),
        layout="horizontal",
        min_pairs=2,
        max_pairs=4,
        calibration_paths={"base_dir": base_dir},
    )


def _rgbd_rig(rig_id: str, base_dir: str = "calibration/rgbd") -> RGBDDeviceRigConfig:
    return RGBDDeviceRigConfig(
        source="rgbd",
        rig_id=rig_id,
        enabled=True,
        fps=30,
        backend=0,
        device_index=3,
        calibration_paths={"base_dir": base_dir},
    )


def _fake_streamer_factory(rig) -> FakeStreamer:
    if isinstance(rig, RGBDDeviceRigConfig):
        return FakeStreamer(_rgbd_frame())
    return FakeStreamer(_stereo_frame())


class CameraFrameProviderTests(unittest.TestCase):
    def test_frame_dataclasses_have_one_canonical_import_surface(self) -> None:
        self.assertIs(StereoFrame, CanonicalStereoFrame)
        self.assertIs(RGBDFrame, CanonicalRGBDFrame)
        self.assertIs(RootStereoFrame, CanonicalStereoFrame)
        self.assertIs(RootRGBDFrame, CanonicalRGBDFrame)

    def test_frame_validation_rejects_mismatched_or_empty_frames(self) -> None:
        with self.assertRaises(ValueError):
            StereoFrame(left=np.zeros((3, 4), dtype=np.uint8), right=np.zeros((4, 4), dtype=np.uint8))
        with self.assertRaises(ValueError):
            StereoFrame(left=np.empty((0,), dtype=np.uint8), right=np.empty((0,), dtype=np.uint8))
        frame = RGBDFrame(color=np.zeros((3, 4, 3), dtype=np.uint8), depth=np.empty(0, dtype=np.uint16))
        self.assertEqual(frame.depth.shape, (0,))

    def test_rig_registration_and_stereo_index_mapping(self) -> None:
        rigs = [_webcam_rig("webcam"), _rgbd_rig("rgbd"), _single_rig("single")]
        with patch.object(FrameProvider, "_create_streamer", side_effect=_fake_streamer_factory):
            provider = FrameProvider(rigs=rigs)

        self.assertEqual(provider.rig_ids, ["webcam", "rgbd", "single"])
        self.assertTrue(provider.is_stereo("webcam"))
        self.assertTrue(provider.is_rgbd("rgbd"))
        self.assertEqual(provider.get_stereo_rig_index("webcam"), 0)
        self.assertEqual(provider.get_stereo_rig_index("single"), 1)
        with self.assertRaises(KeyError):
            provider.get_stereo_rig_index("rgbd")
        with self.assertRaises(UnknownCameraRigError):
            provider.get_rig_config("missing")

    def test_grab_requires_open_provider(self) -> None:
        with patch.object(FrameProvider, "_create_streamer", side_effect=_fake_streamer_factory):
            provider = FrameProvider(rigs=[_webcam_rig("webcam")])

        with self.assertRaises(FrameProviderStateError):
            provider.grab("webcam")

    def test_grab_and_release_delegate_to_streamers(self) -> None:
        with patch.object(FrameProvider, "_create_streamer", side_effect=_fake_streamer_factory):
            provider = FrameProvider(rigs=[_webcam_rig("webcam")])
            provider.open()
            frame = provider.grab("webcam")
            provider.release()

        self.assertIsInstance(frame, StereoFrame)
        self.assertFalse(provider.is_open)

    def test_grab_rectified_fails_without_stereo(self) -> None:
        with patch.object(FrameProvider, "_create_streamer", side_effect=_fake_streamer_factory):
            provider = FrameProvider(rigs=[_webcam_rig("webcam")])
            provider.open()

        with self.assertRaises(RuntimeError):
            provider.grab_rectified("webcam")

    def test_grab_rectified_uses_stereo_rig_index(self) -> None:
        stereo = FakeStereo()
        with patch.object(FrameProvider, "_create_streamer", side_effect=_fake_streamer_factory):
            provider = FrameProvider(rigs=[_rgbd_rig("rgbd"), _single_rig("single")], stereo=stereo)
            provider.open()
            frame = provider.grab_rectified("single")

        self.assertEqual(stereo.calls, [0])
        self.assertTrue(np.array_equal(frame.left, _stereo_frame().left + 1))


class StereoCapturePipelineTests(unittest.TestCase):
    def _pipeline(self, rigs, active_mode: str = "auto", active_rig_id: str | None = None) -> StereoCapturePipeline:
        camera_config = SimpleNamespace(active_mode=active_mode, active_rig_id=active_rig_id, rigs=rigs)
        calibration = SimpleNamespace(marker_length_mm=50.0, aruco_dict_name="DICT_5X5_100")
        return StereoCapturePipeline(camera_config, calibration, stereo_matcher=SimpleNamespace())

    def test_resolves_explicit_rig_id(self) -> None:
        rigs = [_webcam_rig("a"), _single_rig("b")]
        pipeline = self._pipeline(rigs)

        self.assertEqual([rig.rig_id for rig in pipeline._resolve_target_rigs("b")], ["b"])

    def test_resolves_active_rig_mode(self) -> None:
        rigs = [_webcam_rig("a"), _single_rig("b")]
        pipeline = self._pipeline(rigs, active_mode="rig", active_rig_id="a")

        self.assertEqual([rig.rig_id for rig in pipeline._resolve_target_rigs()], ["a"])

    def test_auto_mode_probes_enabled_rigs(self) -> None:
        rigs = [_webcam_rig("a"), _single_rig("b"), _rgbd_rig("c")]
        pipeline = self._pipeline(rigs, active_mode="auto", active_rig_id="a")
        with patch.object(pipeline, "_is_available", side_effect=lambda rig: rig.rig_id != "b"):
            resolved = pipeline._resolve_target_rigs()

        self.assertEqual([rig.rig_id for rig in resolved], ["a", "c"])

    def test_run_skips_rgbd_when_building_stereo_cam(self) -> None:
        rigs = [_webcam_rig("a"), _rgbd_rig("rgbd"), _single_rig("b")]
        pipeline = self._pipeline(rigs)

        with patch.object(pipeline, "_is_available", return_value=True), \
            patch.object(pipeline, "_check_images", return_value=(True, 4)), \
            patch("src.camera.pipeline.stereo_capture.StereoCam3D") as stereo_cls, \
            patch("src.camera.pipeline.stereo_capture.FrameProvider") as provider_cls:
            stereo_obj = Mock(name="stereo")
            stereo_cls.return_value = stereo_obj
            provider_cls.return_value = Mock(name="provider")
            _, stereo = pipeline.run()

        self.assertIs(stereo, stereo_obj)
        stereo_rigs = stereo_cls.call_args.kwargs["rigs"]
        self.assertEqual(len(stereo_rigs), 2)
        self.assertEqual([cfg.stereomap_file for cfg in stereo_rigs], [
            "calibration/webcam/stereoMap.xml",
            "calibration/single/stereoMap.xml",
        ])
        provider_cls.assert_called_once()
        self.assertEqual([rig.rig_id for rig in provider_cls.call_args.kwargs["rigs"]], ["a", "rgbd", "b"])

    def test_force_record_records_even_when_images_exist(self) -> None:
        pipeline = self._pipeline([_webcam_rig("a")])
        with patch.object(pipeline, "_is_available", return_value=True), \
            patch.object(pipeline, "_check_images", return_value=(True, 4)), \
            patch.object(pipeline, "_record") as record, \
            patch("src.camera.pipeline.stereo_capture.StereoCam3D"), \
            patch("src.camera.pipeline.stereo_capture.FrameProvider"):
            pipeline.run(force_record=True)

        record.assert_called_once()

    def test_check_and_clear_images_are_per_rig(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            rig = _webcam_rig("a", base_dir=str(Path(tmp_dir) / "a"))
            left_dir = Path(rig.calibration_paths.left_images_dir)
            right_dir = Path(rig.calibration_paths.right_images_dir)
            left_dir.mkdir(parents=True)
            right_dir.mkdir(parents=True)
            for index in range(3):
                (left_dir / f"left_{index}.png").write_bytes(b"x")
            for index in range(2):
                (right_dir / f"right_{index}.png").write_bytes(b"x")

            pipeline = self._pipeline([rig])
            enough, count = pipeline._check_images(rig)
            pipeline._clear_images(rig)

            self.assertTrue(enough)
            self.assertEqual(count, 2)
            self.assertEqual(list(left_dir.iterdir()), [])
            self.assertEqual(list(right_dir.iterdir()), [])


class OpenCvRGBDStreamerValidationTests(unittest.TestCase):
    """Iteration-2: OpenCvRGBDStreamer.open() fails closed on a colour-resolution
    mismatch, matching SingleDeviceStreamer (previously it only logged)."""

    @staticmethod
    def _open_with(settings: dict) -> OpenCvRGBDStreamer:
        streamer = OpenCvRGBDStreamer(_rgbd_rig("rgbd"))
        fake_cap = Mock()
        fake_cap.isOpened.return_value = True
        with patch(
            "src.camera.setup.image_taking.rgbd.cv.VideoCapture",
            return_value=fake_cap,
        ), patch(
            "src.camera.setup.image_taking.rgbd.configure_camera_for_quality",
            return_value=settings,
        ):
            streamer.open()
        return streamer

    def test_open_rejects_colour_resolution_mismatch(self) -> None:
        streamer = OpenCvRGBDStreamer(_rgbd_rig("rgbd"))
        w, h = streamer.color_res
        bad = {"width": w + 13, "height": h + 7, "fps": 30, "fourcc_str": "MJPG"}
        with self.assertRaises(ValueError):
            self._open_with(bad)

    def test_open_accepts_matching_colour_resolution(self) -> None:
        streamer = OpenCvRGBDStreamer(_rgbd_rig("rgbd"))
        w, h = streamer.color_res
        ok = {"width": w, "height": h, "fps": 30, "fourcc_str": "MJPG"}
        # Must not raise.
        self._open_with(ok)


class SingleDeviceResizeOrderTests(unittest.TestCase):
    """Iteration-2: _ensure_eye_size resizes to (width, height) order — the
    transpose concern from the audit is refuted, this locks correct behaviour."""

    def test_ensure_eye_size_uses_width_height_order(self) -> None:
        cfg = _single_rig("single").model_copy(update={"allow_resize": True})
        streamer = SingleDeviceStreamer(cfg)
        # per_eye_frame_size=(4, 3) == (width=4, height=3). Feed mis-sized eyes
        # (5 wide x 6 tall); output must be (h=3, w=4, c=3), NOT transposed.
        left = np.zeros((6, 5, 3), dtype=np.uint8)
        right = np.zeros((6, 5, 3), dtype=np.uint8)
        out_left, out_right = streamer._ensure_eye_size(left, right)
        self.assertEqual(out_left.shape, (3, 4, 3))
        self.assertEqual(out_right.shape, (3, 4, 3))


if __name__ == "__main__":
    unittest.main()