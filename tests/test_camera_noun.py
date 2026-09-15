"""One owner per camera rig: grabs are serialised, a second opener is refused, and a frame says when it was taken.

Without one owner, two `FrameProvider` instances in one process can each open the same device, two threads can
interleave inside one grab, and no frame carries its capture time. The owner decisions behind the camera noun are
recorded in `.commits/robot/52-one-owner-per-camera.md`.

Identity is the device, not the rig name: a RealSense by its serial, where no serial is one shared key because
the SDK then binds whichever camera it offers first; an OpenCV RGB-D rig and a single-device stereo rig by
`device_index`; a webcam pair by both of its ids, where an unset id collides with every video device.
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from src.camera.orchestration.camera import Camera, CameraBusy, CameraRefused
from src.camera.orchestration.frame_provider import FrameProvider
from src.camera.setup.image_taking.frames import StereoFrame
from tests.test_camera_boundaries import _rgbd_frame, _rgbd_rig, _single_rig, _webcam_rig

_ROOT = Path(__file__).resolve().parents[1]


class _Device:
    """A streamer double: counts opens and releases, and notices a grab that entered while another ran."""

    def __init__(self, *, hold_s: float = 0.0) -> None:
        self.hold_s = hold_s
        self.opens = 0
        self.releases = 0
        self.inside = False
        self.overlaps = 0
        self.events: list[str] = []

    def open(self) -> None:
        self.opens += 1

    def release(self) -> None:
        self.releases += 1

    def grab(self):  # noqa: ANN201
        self.events.append("grab")
        if self.inside:
            self.overlaps += 1
        self.inside = True
        time.sleep(self.hold_s)
        self.inside = False
        return _rgbd_frame()

    def get_intrinsics(self) -> np.ndarray:
        return np.eye(3)

    def get_distortion(self) -> np.ndarray:
        return np.zeros(5)


def _owner(rig, device: _Device | None = None) -> tuple[Camera, _Device]:  # noqa: ANN001
    device = _Device() if device is None else device
    return Camera.from_rig(rig, streamer=device), device


def _opencv(rig_id: str, index: int):  # noqa: ANN202
    return _rgbd_rig(rig_id).model_copy(update={"device_index": index})


def _realsense(rig_id: str, *, serial: str | None, index: int = 0):  # noqa: ANN202
    return _rgbd_rig(rig_id).model_copy(
        update={"rgbd_backend": "realsense", "serial_number": serial, "device_index": index})


def _stereo_device(rig_id: str, index: int):  # noqa: ANN202
    return _single_rig(rig_id).model_copy(update={"device_index": index})


def _pair(rig_id: str, left: int | None, right: int | None):  # noqa: ANN202
    return _webcam_rig(rig_id).model_copy(update={"cam_left_id": left, "cam_right_id": right})


def _section(primary: str, *rigs) -> SimpleNamespace:  # noqa: ANN002
    """The camera section a caller hands `from_config`: `app_cfg.camera`."""
    return SimpleNamespace(cameras=SimpleNamespace(primary_rig_id=primary, rigs=list(rigs)))


class OneOwnerPerDeviceTests(unittest.TestCase):

    def test_two_threads_never_grab_one_rig_at_once(self) -> None:
        device = _Device(hold_s=0.02)
        camera, _ = _owner(_opencv("overhead", 3), device)
        camera.open()
        self.addCleanup(camera.release)
        handles = [camera.handle(), camera.handle()]
        threads = [threading.Thread(target=lambda h=h: [h.grab() for _ in range(5)]) for h in handles]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(device.events.count("grab"), 10)
        self.assertEqual(device.overlaps, 0, "two grabs reached one device at the same time")

    def test_a_second_owner_of_an_open_rig_is_refused_naming_the_holder(self) -> None:
        first, _ = _owner(_opencv("overhead", 3))
        second, second_device = _owner(_opencv("bench", 3))
        first.open()
        self.addCleanup(first.release)
        with self.assertRaises(CameraBusy) as caught:
            second.open()
        self.assertEqual((caught.exception.rig_id, caught.exception.holder), ("bench", "overhead"))
        self.assertIn("'overhead'", str(caught.exception))
        self.assertEqual(second_device.opens, 0, "the refusal came after the device was touched")
        self.assertFalse(second.is_open)

    def test_a_catalogue_and_a_camera_cannot_both_open_one_rig(self) -> None:
        rig = _opencv("overhead", 3)
        camera, _ = _owner(rig)
        camera.open()
        provider = FrameProvider([rig])
        provider._streamers = {"overhead": _Device()}  # noqa: SLF001 (a device double behind the catalogue)
        with self.assertRaises(CameraBusy):
            provider.open_rig("overhead")
        self.assertNotIn("overhead", provider.open_rig_ids())
        camera.release()
        provider.open_rig("overhead")
        self.addCleanup(provider.release)
        with self.assertRaises(CameraBusy):
            _owner(rig)[0].open()

    def test_two_realsense_rigs_without_serials_are_one_device(self) -> None:
        """With no serial the SDK binds whichever RealSense it offers first, so a null serial collides
        with every open RealSense, named or not."""
        first, _ = _owner(_realsense("left", serial=None, index=0))
        first.open()
        self.addCleanup(first.release)
        for other in (_realsense("right", serial=None, index=1), _realsense("named", serial="123", index=1)):
            with self.subTest(other=other.rig_id), self.assertRaises(CameraBusy):
                _owner(other)[0].open()

    def test_identity_decides_so_distinct_devices_both_open(self) -> None:
        """The control: two OpenCV rigs on two indices, and two RealSense rigs with two serials."""
        pairs = ((_opencv("a", 0), _opencv("b", 1)),
                 (_realsense("a", serial="111"), _realsense("b", serial="222")))
        for a, b in pairs:
            with self.subTest(backends=(a.rgbd_backend, b.rgbd_backend)):
                first, _ = _owner(a)
                second, _ = _owner(b)
                first.open()
                second.open()
                self.assertTrue(first.is_open and second.is_open)
                first.release()
                second.release()

    def test_a_stereo_device_and_an_opencv_rgbd_rig_on_one_index_are_one_device(self) -> None:
        stereo, _ = _owner(_stereo_device("stereo", 0))
        stereo.open()
        self.addCleanup(stereo.release)
        with self.assertRaises(CameraBusy):
            _owner(_opencv("rgbd", 0))[0].open()
        other, _ = _owner(_opencv("rgbd", 3))
        other.open()
        other.release()

    def test_a_webcam_pair_with_an_unset_id_collides_with_every_video_device(self) -> None:
        rgbd, _ = _owner(_opencv("rgbd", 5))
        rgbd.open()
        self.addCleanup(rgbd.release)
        with self.assertRaises(CameraBusy):
            _owner(_pair("pair", None, 1))[0].open()
        paired, _ = _owner(_pair("pair", 0, 1))
        paired.open()
        paired.release()

    def test_release_gives_the_device_back_and_the_next_owner_opens(self) -> None:
        first, first_device = _owner(_opencv("overhead", 3))
        first.open()
        first.release()
        first.release()
        second, _ = _owner(_opencv("bench", 3))
        second.open()
        self.addCleanup(second.release)
        self.assertEqual((first_device.opens, first_device.releases), (1, 1))
        self.assertFalse(first.is_open)
        self.assertTrue(second.is_open)

    def test_a_release_that_raises_is_swallowed_and_gives_the_device_back(self) -> None:
        """Teardown: a camera that cannot be closed must not stop the thing that was closing it."""
        first, device = _owner(_opencv("overhead", 3))
        first.open()
        device.release = mock.Mock(side_effect=RuntimeError("device is wedged"))  # type: ignore[method-assign]
        first.release()
        self.assertFalse(first.is_open)
        second, _ = _owner(_opencv("bench", 3))
        second.open()
        second.release()

    def test_an_owner_that_no_longer_exists_holds_no_device(self) -> None:
        """The registry names live owners. A build that raised past its release and was dropped must not
        leave the next build meeting CameraBusy for an object nobody can reach any more."""
        first, _ = _owner(_opencv("overhead", 3))
        first.open()
        del first
        gc.collect()
        second, _ = _owner(_opencv("bench", 3))
        second.open()
        second.release()


class TheHandleTests(unittest.TestCase):

    def test_the_handle_keeps_the_calibration_surface(self) -> None:
        from src.calibration.rgbd_marker_source import RGBDArucoMarkerSource

        camera, _ = _owner(_opencv("overhead", 3))
        camera.open()
        self.addCleanup(camera.release)
        handle = camera.handle()
        for name in ("grab", "get_intrinsics", "get_distortion", "release"):
            self.assertTrue(callable(getattr(handle, name, None)), name)
        self.assertTrue(handle.is_open)
        self.assertEqual(handle.rig_id, "overhead")
        self.assertIs(handle.camera, camera)
        self.assertIsNotNone(RGBDArucoMarkerSource(streamer=handle, marker_length_mm=50.0))

    def test_a_catalogue_handle_names_its_owner_too(self) -> None:
        rig = _opencv("overhead", 3)
        provider = FrameProvider([rig])
        provider._streamers = {"overhead": _Device()}  # noqa: SLF001 (a device double behind the catalogue)
        provider.open_rig("overhead")
        self.addCleanup(provider.release)
        owner = provider.rig("overhead").camera
        self.assertIsInstance(owner, Camera)
        self.assertEqual(owner.rig_id, "overhead")

    def test_releasing_a_handle_releases_its_owner(self) -> None:
        camera, device = _owner(_opencv("overhead", 3))
        camera.open()
        camera.handle().release()
        self.assertFalse(camera.is_open)
        self.assertEqual(device.releases, 1)


class SelectionTests(unittest.TestCase):
    """The cell's three refusals, which the noun owns: an unknown, a disabled and a non RGB-D rig."""

    def test_an_unknown_primary_is_refused_naming_the_rigs(self) -> None:
        with self.assertRaises(CameraRefused) as caught:
            Camera.from_config(_section("ghost", _opencv("overhead", 3)))
        self.assertEqual(caught.exception.reason, "unknown")
        self.assertEqual(
            str(caught.exception),
            "camera.cameras.primary_rig_id names 'ghost', which is not in camera.cameras.rigs "
            "(['overhead']).")

    def test_an_unknown_named_rig_is_refused_naming_the_rigs(self) -> None:
        with self.assertRaises(CameraRefused) as caught:
            Camera.from_config(_section("overhead", _opencv("overhead", 3), _opencv("wrist", 4)),
                               rig_id="front")
        self.assertEqual(caught.exception.reason, "unknown")
        self.assertEqual(str(caught.exception),
                         "no rig 'front' in camera.cameras.rigs; configured: ['overhead', 'wrist'].")

    def test_a_disabled_primary_is_refused(self) -> None:
        rig = _opencv("overhead", 3).model_copy(update={"enabled": False})
        with self.assertRaises(CameraRefused) as caught:
            Camera.from_config(_section("overhead", rig))
        self.assertEqual(caught.exception.reason, "disabled")
        self.assertEqual(
            str(caught.exception),
            "camera.cameras.primary_rig_id names 'overhead' and that rig has `enabled: false`. A cell "
            "cannot run on a camera its own config declares off. Switch it on, or name a rig that is on.")

    def test_a_non_rgbd_primary_is_refused_naming_the_rgbd_rigs(self) -> None:
        with self.assertRaises(CameraRefused) as caught:
            Camera.from_config(_section("pair", _pair("pair", 0, 1), _opencv("overhead", 3)))
        self.assertEqual(caught.exception.reason, "not_rgbd")
        self.assertEqual(
            str(caught.exception),
            "camera.cameras.primary_rig_id names 'pair', a 'webcam_pair' rig, not an RGB-D device. A "
            "grasp cell needs an RGB-D primary: the grasp is synthesised from its depth. The RGB-D rigs "
            "here are ['overhead']. Name one of those instead, and prove it standalone with "
            "`python -m src.robot.perception`.")

    def test_open_disabled_lifts_the_disabled_refusal_and_nothing_else(self) -> None:
        off = _opencv("overhead", 3).model_copy(update={"enabled": False})
        camera = Camera.from_config(_section("overhead", off), open_disabled=True)
        self.assertEqual(camera.rig_id, "overhead")
        self.assertFalse(camera.enabled)
        self.assertFalse(camera.is_open, "building the owner must not open the device")
        stereo = _pair("pair", 0, 1)
        with self.assertRaises(CameraRefused) as caught:
            Camera.from_config(_section("pair", stereo), open_disabled=True)
        self.assertEqual(caught.exception.reason, "not_rgbd")
        with self.assertRaises(CameraRefused) as unknown:
            Camera.from_config(_section("pair", stereo), rig_id="ghost", open_disabled=True)
        self.assertEqual(unknown.exception.reason, "unknown")


class CaptureTimeTests(unittest.TestCase):

    def test_a_frame_from_the_owner_carries_its_capture_time(self) -> None:
        device = _Device()
        camera, _ = _owner(_opencv("overhead", 3), device)
        camera.open()
        self.addCleanup(camera.release)

        def clock() -> float:
            device.events.append("clock")
            return 1234.5

        with mock.patch("src.camera.orchestration.camera.time", SimpleNamespace(time=clock)):
            frame = camera.handle().grab()
        self.assertEqual(frame.captured_at_s, 1234.5)
        self.assertEqual(device.events, ["clock", "grab"], "the host time was not read just before the grab")
        self.assertIsNone(device.grab().captured_at_s, "a streamer used directly has no owner to stamp it")

    def test_both_frame_kinds_carry_the_field_and_default_to_none(self) -> None:
        self.assertIsNone(_rgbd_frame().captured_at_s)
        eye = np.zeros((2, 2, 3), np.uint8)
        self.assertEqual(StereoFrame(left=eye, right=eye, captured_at_s=7.0).captured_at_s, 7.0)


class ImportBoundaryTests(unittest.TestCase):
    """`Robot` may not load the camera package, and the camera package may not load the robot one."""

    _PROBE = ("import importlib, json, sys; importlib.import_module({module!r}); "
              "print(json.dumps(sorted(m for m in sys.modules if m.startswith('src.robot'))))")

    def _robot_modules_after(self, module: str) -> list[str]:
        out = subprocess.run([sys.executable, "-c", self._PROBE.format(module=module)], cwd=_ROOT,
                             capture_output=True, text=True, check=True).stdout
        return json.loads(out.strip().splitlines()[-1])

    def test_the_camera_module_loads_nothing_from_robot(self) -> None:
        self.assertEqual(self._robot_modules_after("src.camera.orchestration.camera"), [])

    def test_the_probe_sees_a_module_that_does_load_robot(self) -> None:
        """The control: the same probe over the perception package sees the grasping package arrive."""
        self.assertIn("src.robot.grasping", self._robot_modules_after("src.robot.perception"))


if __name__ == "__main__":
    unittest.main()
