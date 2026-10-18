"""The pick service lends its own detector and its own cameras to the locators a task finds its target with (OD 19).

A task that places into a bin the camera finds locates that bin with the cell's pick service's own perception backend,
on the service's own camera owners: no second copy of the detector loads (a VLM of nine gigabytes on the owner's cell,
``src/models/vlm/README.md``), and no camera is opened twice. ``locators_for_service(service)`` builds one single-phrase
``Locator`` per camera owner the service holds (``lifecycle.service_cameras``: the primary first), each over the backend
the service's camera source was built with, which that source now hands out read-only (``backend``, beside
``streamer``). ``PerceptionSpec.build`` is never called on that path: a spy proves it.

A cell with no camera owner (the rehearsal scene, a service built around doubles) or whose camera source carries no
backend is refused with ``LocatorRefused``, before anything is located; a task's camera place is refused there before
anything moves. A wrist camera's locator reads the arm's TCP at each shutter, as the service's own frames are stamped.
Each locator keeps the colour image of what it located for a prompt, so the task can show the target it kept
(``draw_located`` over the sighting, as PNG).
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from src.camera.setup.image_taking.frames import RGBDFrame
from src.geometry import Frame, Pose, Transform
from src.robot.perception import RealSenseVisionPerceptionSource
from src.robot.perception.locator import Locator, LocatorRefused

_K = np.array([[600.0, 0.0, 32.0], [0.0, 600.0, 32.0], [0.0, 0.0, 1.0]])


@dataclass
class _Det:
    box: tuple[float, float, float, float]
    label: str = "blue bin"
    score: float = 0.8


@dataclass(frozen=True)
class _Seg:
    mask: np.ndarray
    label: str = "blue bin"


class _Detector:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def detect_all(self, bgr: np.ndarray, prompt: str) -> list[_Det]:
        self.prompts.append(prompt)
        return [_Det((16.0, 16.0, 48.0, 48.0), label=prompt)]


class _Segmenter:
    def segment_detection(self, bgr: np.ndarray, det: _Det) -> _Seg:
        mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        x0, y0, x1, y1 = (int(c) for c in det.box)
        mask[y0:y1, x0:x1] = 1
        return _Seg(mask=mask, label=det.label)


class _Handle:
    """What a camera owner hands its consumers: grabs under the owner, and the owner itself."""

    def __init__(self, camera: "_Owner") -> None:
        self.camera = camera
        self.rig_id = camera.rig_id

    def grab(self) -> RGBDFrame:
        self.camera.grabs += 1
        colour = np.full((64, 64, 3), 90, dtype=np.uint8)
        depth = np.full((64, 64), 600, dtype=np.uint16)
        return RGBDFrame(color=colour, depth=depth)

    def get_intrinsics(self) -> np.ndarray:
        return _K.copy()


class _Owner:
    """A camera owner as ``service_cameras`` finds one: a rig id, a peek, a handle and a calibration."""

    source = "rgbd"

    def __init__(self, rig_id: str, *, wrist: bool = False) -> None:
        self.rig_id = rig_id
        self.wrist = wrist
        self.grabs = 0

    def peek(self) -> None:
        return None

    def handle(self) -> _Handle:
        return _Handle(self)

    def calibration(self) -> Any:
        looking_down = np.eye(4)
        looking_down[:3, :3] = np.diag([1.0, -1.0, -1.0])
        looking_down[:3, 3] = (0.0, -500.0, 700.0)
        transform = Transform.from_matrix(looking_down, from_frame=Frame.CAMERA, to_frame=Frame.BASE)
        return SimpleNamespace(
            rig_id=self.rig_id, mounting_mode="eye_in_hand" if self.wrist else "eye_to_hand",
            camera_to_base=lambda: transform, camera_to_tool=lambda: transform,
            shutter_motion_tolerance_mm=1.0, shutter_motion_tolerance_deg=0.5)


def _source(owner: _Owner) -> RealSenseVisionPerceptionSource:
    return RealSenseVisionPerceptionSource(streamer=owner.handle(), detector=_Detector(), segmenter=_Segmenter(),
                                           prompt="object", warmup_grabs=0)


class _Arm:
    def __init__(self) -> None:
        self.read = 0

    def get_tcp_pose(self) -> Pose:
        self.read += 1
        return Pose.tool_down(0.0, -500.0, 400.0)


def _service(*owners: _Owner, sources: "list[Any] | None" = None) -> Any:
    primary = sources[0] if sources else _source(owners[0])
    extra = {owner.rig_id: _source(owner) for owner in owners[1:]}
    orchestrator = SimpleNamespace(
        perception=primary, arm=_Arm(),
        multi_camera_perception=SimpleNamespace(sources=extra) if extra else None,
    )
    return SimpleNamespace(runtime=SimpleNamespace(orchestrator=orchestrator))


class TheSourceHandsOutItsBackendTests(unittest.TestCase):
    def test_the_backend_a_source_was_built_with_is_read_back_and_cannot_be_replaced(self) -> None:
        backend = SimpleNamespace(perceive=lambda bgr, prompt: [])
        source = RealSenseVisionPerceptionSource(streamer=_Owner("wrist").handle(), backend=backend, prompt="object")

        self.assertIs(backend, source.backend)
        with self.assertRaises(AttributeError):
            source.backend = object()  # type: ignore[misc]

    def test_a_source_built_from_a_detector_and_a_segmenter_hands_out_the_backend_it_composed(self) -> None:
        source = _source(_Owner("wrist"))
        self.assertIs(source._backend, source.backend)  # noqa: SLF001


class LocatorsForTheServiceTests(unittest.TestCase):
    def test_one_locator_per_camera_owner_all_over_the_services_backend_and_nothing_built(self) -> None:
        """Red before: there was no way to reach the service's backend, and a task building its locators from the
        config would have loaded a second copy of every model (``PerceptionSpec.build``)."""
        from src.robot.execution.place_target import locators_for_service

        primary, side = _Owner("overhead"), _Owner("side")
        service = _service(primary, side)
        backend = service.runtime.orchestrator.perception.backend
        with mock.patch("src.models.perception_spec.PerceptionSpec.build",
                        side_effect=AssertionError("a second model copy was built")) as build, \
                mock.patch("src.models.perception_spec.PerceptionSpec.from_config",
                           side_effect=AssertionError("the models were read from the config")) as from_config:
            locators = locators_for_service(service)

        build.assert_not_called()
        from_config.assert_not_called()
        self.assertEqual(2, len(locators))
        self.assertTrue(all(isinstance(locator, Locator) for locator in locators))
        self.assertEqual([primary, side], [locator._camera for locator in locators])  # noqa: SLF001
        self.assertTrue(all(locator._backend is backend for locator in locators), "a locator has a backend of its own")  # noqa: SLF001
        self.assertFalse(any(locator.on_the_wrist for locator in locators))

    def test_a_locator_of_the_service_locates_through_the_services_detector(self) -> None:
        from src.robot.execution.place_target import locators_for_service

        owner = _Owner("overhead")
        service = _service(owner)
        detector = service.runtime.orchestrator.perception.backend.detector

        located = locators_for_service(service)[0].locate("blue bin")

        self.assertEqual(["blue bin"], detector.prompts, "the service's detector was not the one asked")
        self.assertEqual(1, len(located.objects))
        self.assertEqual("blue bin", located.objects[0].label)
        self.assertGreater(located.objects[0].points_base_mm.shape[0], 500)
        self.assertEqual("object", service.runtime.orchestrator.perception.prompt,
                         "locating changed the phrase the service's own picks ground")

    def test_a_wrist_cameras_locator_reads_the_arms_tcp_at_its_shutter(self) -> None:
        from src.robot.execution.place_target import locators_for_service

        owner = _Owner("wrist", wrist=True)
        service = _service(owner)
        arm = service.runtime.orchestrator.arm
        with mock.patch("src.calibration.rig_calibration.flange_to_tcp_refusal", return_value=None):
            (locator,) = locators_for_service(service, tool_frame=SimpleNamespace())
        locator.locate("blue bin")

        self.assertTrue(locator.on_the_wrist)
        self.assertGreaterEqual(arm.read, 2, "the wrist frame was not stamped with the tool pose at its shutter")

    def test_a_wrist_camera_with_no_tool_frame_to_hold_its_calibration_to_is_refused(self) -> None:
        from src.robot.execution.place_target import locators_for_service

        with self.assertRaises(LocatorRefused):
            locators_for_service(_service(_Owner("wrist", wrist=True)))

    def test_a_cell_with_no_camera_owner_is_refused_and_says_why(self) -> None:
        from src.robot.execution.place_target import locators_for_service

        rehearsal = SimpleNamespace(acquire=lambda: None, set_prompt=lambda *_a, **_k: None)
        service = SimpleNamespace(runtime=SimpleNamespace(orchestrator=SimpleNamespace(
            perception=rehearsal, arm=_Arm(), multi_camera_perception=None)))

        with self.assertRaisesRegex(LocatorRefused, "no camera"):
            locators_for_service(service)

    def test_a_camera_source_that_carries_no_backend_is_refused(self) -> None:
        from src.robot.execution.place_target import locators_for_service

        owner = _Owner("overhead")
        handle = owner.handle()
        no_backend = SimpleNamespace(streamer=handle, set_prompt=lambda *_a, **_k: None)
        service = _service(owner, sources=[no_backend])

        with self.assertRaisesRegex(LocatorRefused, "backend"):
            locators_for_service(service)

    def test_each_locator_keeps_the_image_of_what_it_located_for_the_target_overlay(self) -> None:
        from src.robot.execution.place_target import located_image, locators_for_service

        (locator,) = locators_for_service(_service(_Owner("overhead")))
        self.assertIsNone(located_image(locator, "blue bin"))

        located = locator.locate("blue bin")

        image = located_image(locator, "blue bin")
        assert image is not None
        self.assertEqual((64, 64, 3), image.shape)
        self.assertIsNone(located_image(locator, "red cube"))
        self.assertEqual(1, len(located.objects))
        (bare,) = locators_for_service(_service(_Owner("overhead")), keep_image=False)
        bare.locate("blue bin")
        self.assertIsNone(located_image(bare, "blue bin"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
