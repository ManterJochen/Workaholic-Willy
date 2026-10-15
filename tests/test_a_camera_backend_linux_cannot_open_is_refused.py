"""A camera backend Linux cannot open is refused by name where a camera is built (P0h h7, owner decision 5a).

``camera.cameras.rigs[*].backend`` is OpenCV's capture API, an integer. The owner's tree names 700, DirectShow, on its
primary webcam rig. 700 and 1400, Media Foundation, exist only in OpenCV's Windows builds, so on Linux
``VideoCapture(i, 700)`` opens nothing, and the operator read "No enabled camera rigs are currently available" with no
word about the backend. The owner chose, on 2026-09-26, a refusal that names the key and the Linux values 0 and 200,
over a silent translation, and his Windows tree stays as it is. So the tree still loads on Linux and the refusal sits
where a camera is built: the cell's camera, the calibration tools, the stereo pipeline's probe. A RealSense rig opens
through its SDK and never reads the key, so it is never refused for it.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from unittest import mock

from src.config.loader import load_camera_section
from src.camera.orchestration.camera import Camera, CameraRefused
from src.camera.pipeline.stereo_capture import StereoCapturePipeline
from src.camera.setup.devices.stereocamera import StereoVisionCalibrationSingleDevice
from src.camera.setup.devices.webcam import StereoVisionCalibrationWebcams
from src.camera.setup.image_taking.rgbd import OpenCvRGBDStreamer, RealSenseRGBDStreamer
from src.camera.setup.image_taking.single import SingleDeviceStreamer
from src.camera.setup.image_taking.webcam import WebcamPairStreamer
from tests.test_camera_boundaries import _rgbd_rig, _single_rig, _webcam_rig
from tests.test_camera_noun import _section


def _on(platform: str) -> mock._patch:  # type: ignore[type-arg]
    return mock.patch.object(sys, "platform", platform)


def _refusal(rig_id: str, backend: int, name: str) -> str:
    return (f"camera.cameras.rigs[{rig_id!r}].backend is {backend}, {name}, which OpenCV has on Windows only, so this "
            "rig opens no camera here. On Linux set it to 0 (CAP_ANY, OpenCV chooses) or 200 (CAP_V4L2).")


class TheRuleTests(unittest.TestCase):
    def test_directshow_and_media_foundation_are_refused_off_windows_by_name(self) -> None:
        from src.camera.setup.capture_backend import capture_backend_refusal

        for backend, name in ((700, "CAP_DSHOW (DirectShow)"), (1400, "CAP_MSMF (Media Foundation)")):
            with self.subTest(backend):
                rig = _rgbd_rig("overhead").model_copy(update={"backend": backend})
                self.assertEqual(capture_backend_refusal(rig, platform="linux"), _refusal("overhead", backend, name))
                self.assertIsNone(capture_backend_refusal(rig, platform="win32"))

    def test_the_linux_values_pass_everywhere(self) -> None:
        from src.camera.setup.capture_backend import capture_backend_refusal

        for backend in (0, 200):
            for platform in ("linux", "win32"):
                with self.subTest(backend=backend, platform=platform):
                    rig = _webcam_rig("pair").model_copy(update={"backend": backend})
                    self.assertIsNone(capture_backend_refusal(rig, platform=platform))

    def test_a_realsense_rig_never_reads_the_key(self) -> None:
        from src.camera.setup.capture_backend import capture_backend_refusal

        rig = _rgbd_rig("wrist").model_copy(update={"backend": 700, "rgbd_backend": "realsense"})
        self.assertIsNone(capture_backend_refusal(rig, platform="linux"))


class TheCellsCameraIsRefusedTests(unittest.TestCase):
    """The grasp cell builds its cameras through ``Camera.from_config`` and turns a refusal into its own."""

    def test_on_linux_the_primary_is_refused_with_its_own_reason(self) -> None:
        rig = _rgbd_rig("overhead").model_copy(update={"backend": 700})
        with _on("linux"), self.assertRaises(CameraRefused) as caught:
            Camera.from_config(_section("overhead", rig))
        self.assertEqual(caught.exception.reason, "backend")
        self.assertEqual(str(caught.exception), _refusal("overhead", 700, "CAP_DSHOW (DirectShow)"))

    def test_on_windows_the_same_rig_is_built_as_before(self) -> None:
        rig = _rgbd_rig("overhead").model_copy(update={"backend": 700})
        with _on("win32"):
            camera = Camera.from_config(_section("overhead", rig))
        self.assertEqual(camera.rig_id, "overhead")
        self.assertFalse(camera.is_open)

    def test_a_realsense_primary_is_built_on_linux_whatever_the_key_says(self) -> None:
        rig = _rgbd_rig("wrist").model_copy(update={"backend": 700, "rgbd_backend": "realsense"})
        with _on("linux"):
            self.assertEqual(Camera.from_config(_section("wrist", rig)).rig_id, "wrist")


class TheToolsRefuseBeforeTouchingADeviceTests(unittest.TestCase):
    """The calibration tools and the stereo pipeline build their cameras without the cell's noun."""

    def _refused(self, build) -> ValueError:  # noqa: ANN001
        with _on("linux"), mock.patch("cv2.VideoCapture", side_effect=AssertionError("a device was touched")), \
                self.assertRaises(ValueError) as caught:
            build()
        return caught.exception

    def test_every_opencv_camera_refuses_on_construction(self) -> None:
        webcam = _webcam_rig("pair").model_copy(update={"backend": 700, "cam_left_id": None, "cam_right_id": None})
        single = _single_rig("stereo").model_copy(update={"backend": 1400})
        rgbd = _rgbd_rig("overhead").model_copy(update={"backend": 700})
        builds = {
            "WebcamPairStreamer": (lambda: WebcamPairStreamer(webcam), "pair", 700),
            "SingleDeviceStreamer": (lambda: SingleDeviceStreamer(single), "stereo", 1400),
            "OpenCvRGBDStreamer": (lambda: OpenCvRGBDStreamer(rgbd), "overhead", 700),
            "StereoVisionCalibrationWebcams": (lambda: StereoVisionCalibrationWebcams(webcam), "pair", 700),
            "StereoVisionCalibrationSingleDevice": (lambda: StereoVisionCalibrationSingleDevice(single), "stereo", 1400),
        }
        for name, (build, rig_id, backend) in builds.items():
            with self.subTest(name):
                refused = self._refused(build)
                self.assertIn(f"camera.cameras.rigs[{rig_id!r}].backend is {backend}", str(refused))

    def test_the_stereo_pipelines_probe_says_the_backend_and_not_no_camera(self) -> None:
        rig = _webcam_rig("pair").model_copy(update={"backend": 700})
        pipeline = object.__new__(StereoCapturePipeline)
        refused = self._refused(lambda: pipeline._is_available(rig))  # noqa: SLF001
        self.assertIn("camera.cameras.rigs['pair'].backend is 700", str(refused))

    def test_the_realsense_streamer_is_built_on_linux(self) -> None:
        rig = _rgbd_rig("wrist").model_copy(update={"backend": 700, "rgbd_backend": "realsense"})
        with _on("linux"):
            RealSenseRGBDStreamer(rig)


class NoCameraOpensPastTheRuleTests(unittest.TestCase):
    """The way this would decay: a new module opens ``cv.VideoCapture(index, backend)`` and never asks the rule."""

    _CAMERA = Path(__file__).resolve().parents[1] / "src" / "camera"

    def _opens_with_a_backend(self, tree: ast.AST) -> bool:
        return any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == "VideoCapture" and len(node.args) >= 2 for node in ast.walk(tree))

    def _asks_the_rule(self, tree: ast.AST) -> bool:
        return any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == "refuse_a_backend_this_platform_lacks" for node in ast.walk(tree))

    def test_every_module_that_opens_a_camera_with_a_backend_asks_the_rule(self) -> None:
        opening = []
        for path in sorted(self._CAMERA.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if self._opens_with_a_backend(tree):
                opening.append(path.relative_to(self._CAMERA).as_posix())
                with self.subTest(path.name):
                    self.assertTrue(self._asks_the_rule(tree), f"{path.name} opens a camera past the rule")
        self.assertEqual(len(opening), 6, opening)

    def test_the_guard_can_fail(self) -> None:
        tree = ast.parse("cap = cv.VideoCapture(0, self.backend)\n")
        self.assertTrue(self._opens_with_a_backend(tree))
        self.assertFalse(self._asks_the_rule(tree))


class TheOwnersTreeStillLoadsOnLinuxTests(unittest.TestCase):
    def test_the_shipped_camera_section_loads_with_its_directshow_rig(self) -> None:
        with _on("linux"):
            section = load_camera_section()
        backends = {rig.rig_id: rig.backend for rig in section.cameras.rigs}
        self.assertEqual(backends.get("webcam_main"), 700, "the owner's tree is unchanged, and it loads")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
