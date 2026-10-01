"""Asking where a hand is through a camera the program already holds open.

⚠ WHY THIS SEAM EXISTS, AND WHAT IT PREVENTS. `HandFinder` reached its frames through
`FrameProvider`, the multi-rig catalogue, and `FrameProvider.open()` claims every configured
streamer. A program that has already opened one `Camera` -- anything that hands the arm a live
camera world, which is every camera-checked motion -- therefore could not ask where a hand is
without either taking the cell's other devices away from it or opening one device twice, which is
exactly what the camera owner exists to prevent. The capability shipped and no such program could
reach it; `examples/real_robot/15` is the program.

The transform is the other half. A fixed rig's CAMERA to BASE is `RigCalibration.camera_to_base()`,
one method with one answer. A wrist rig has none: its calibration is CAMERA to TOOL, so each of its
frames is taken through the pick frame's own stamp (the tool pose read before and after the grab,
the grab taken again while the tool moved beyond the rig's shutter tolerance) and placed by the one
composition the `Locator` places its frames by (`shutter_motion.camera_to_base_at_shutter`). A second
spelling of that matrix would be a second source of truth for the number that decides where an arm
moves next to a person's hand. A frame the tool moved across is placed by nothing and gives no hand,
and a wrist rig without the arm's TCP reader, or held to another flange to TCP, is refused by name.

No mediapipe, no `.task` bundle and no device: the observer is a double and the camera is a stub,
because what is under test is the wiring, the placement and the refusals.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from src.calibration.rig_calibration import RigCalibration, RigCalibrationError, RigNotCalibrated
from src.calibration.serialization import FlangeToTcp
from src.camera.setup.image_taking.frames import RGBDFrame
from src.config.schema.models import HandDetectConfig
from src.contracts import UNSET
from src.geometry import Frame, Pose
from src.geometry.transform import Transform
from src.models.handdetection.factory import build_hand_finder_on_camera
from src.models.handdetection.hand_finder import OneCamera, OneWristCamera
from src.models.handdetection.landmarks import LANDMARK_COUNT, PALM_LANDMARKS
from src.models.handdetection.types import (
    GestureReading,
    Handedness,
    HandGesture,
    HandObservation,
    PalmDetection,
    as_landmark_tuple,
)

_K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])


def _camera_to_base() -> Transform:
    return Transform(translation_mm=np.array([100.0, 0.0, 800.0]),
                     quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                     from_frame=Frame.CAMERA, to_frame=Frame.BASE)


class _Calibration:
    """A rig calibration that answers for one mounting and refuses the other, as the real one does."""

    def __init__(self, *, fixed: bool) -> None:
        self._fixed = fixed

    def camera_to_base(self) -> Transform:
        if not self._fixed:
            raise RigCalibrationError("rig 'wrist' is eye_in_hand: a wrist camera has no fixed "
                                      "CAMERA to BASE; ask its frames for the tool pose.")
        return _camera_to_base()


class _Rig:
    """Stands in for the rig config. `OneCamera.is_rgbd` asks its type, as `FrameProvider` does."""


class _Camera:
    def __init__(self, *, rig_id: str = "overhead", intrinsics: Any = _K,
                 calibration: Any = None, rig: Any = None, frame: Any = None) -> None:
        self.rig_id = rig_id
        self.rig = rig if rig is not None else _Rig()
        self._intrinsics = intrinsics
        self._calibration = calibration if calibration is not None else _Calibration(fixed=True)
        self._frame = frame
        self.grabs = 0

    def get_intrinsics(self) -> Any:
        return self._intrinsics

    def calibration(self) -> Any:
        if self._calibration is None:                      # pragma: no cover - set in every case
            raise RigNotCalibrated("rig declares no calibration")
        return self._calibration

    def grab(self) -> Any:
        self.grabs += 1
        return self._frame


def _rgbd_camera(**kwargs: Any) -> _Camera:
    from src.config.schema.camera import RGBDDeviceRigConfig

    rig = RGBDDeviceRigConfig.model_construct()
    return _Camera(rig=rig, **kwargs)


# -- A camera on the wrist -------------------------------------------------------------------------


def _rotation(axis: str, deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return {"x": np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]),
            "z": np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])}[axis]


def _matrix(rotation: np.ndarray, translation_mm: tuple[float, float, float]) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation_mm
    return matrix


#: The camera 60 mm out from the tool and 40 mm back along it, tilted 45 degrees: a wrist camera, roughly.
_CAMERA_TO_TOOL = _matrix(_rotation("x", 45.0), (60.0, 0.0, -40.0))
#: Where the tool stood at the shutter: pointing down, turned 30 degrees, well inside a cell.
_TOOL_AT_SHUTTER = _matrix(_rotation("z", 30.0) @ _rotation("x", 180.0), (400.0, -150.0, 600.0))
#: The cell's `robot.gripper.tool_frame`: a willy cell with no offset, the frame the wrist record holds.
_TOOL_FRAME = SimpleNamespace(source="willy", offset_mm=(0.0, 0.0, 0.0), rotation_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                              verify_tolerance_mm=1.0)
#: Where the palm is in every frame, in pixels, and how far away the depth map says it is.
_PALM_PX, _PALM_DEPTH_MM = (400.0, 200.0), 500


def _wrist(*, record: Any = "declared") -> RigCalibration:
    flange = FlangeToTcp.from_matrix("willy", np.eye(4)) if record == "declared" else record
    return RigCalibration(rig_id="wrist", mounting_mode="eye_in_hand", artifact_path="eih.json",
                          transform=Transform.from_matrix(_CAMERA_TO_TOOL, from_frame=Frame.CAMERA,
                                                          to_frame=Frame.TOOL),
                          shutter_motion_tolerance_mm=1.0, shutter_motion_tolerance_deg=0.5, flange_to_tcp=flange)


def _tool(*, dx_mm: float = 0.0, turn_deg: float = 0.0, matrix: np.ndarray = _TOOL_AT_SHUTTER) -> Pose:
    """`matrix` shifted `dx_mm` along BASE X, and turned `turn_deg` about BASE Z where the tool stands."""
    moved = matrix.copy()
    moved[0, 3] += dx_mm
    moved[:3, :3] = _rotation("z", turn_deg) @ moved[:3, :3]
    return Pose.from_matrix(moved, frame=Frame.BASE)


class _Reader:
    """The arm's `get_tcp_pose`: the poses given, in turn, then the last one forever."""

    def __init__(self, *poses: Pose, events: "list[str] | None" = None) -> None:
        self._poses = list(poses)
        self._events = events
        self.reads = 0

    def __call__(self) -> Pose:
        if self._events is not None:
            self._events.append("pose")
        pose = self._poses[min(self.reads, len(self._poses) - 1)]
        self.reads += 1
        return pose


class _WristCamera(_Camera):
    """An open RGB-D camera the arm carries: a new frame object on every grab, each one kept.

    `depth_mm` gives grab n (from 1) its depth, so a palm's depth can say which grab it was read from.
    """

    def __init__(self, *, calibration: Any = None, events: "list[str] | None" = None,
                 depth_mm: Callable[[int], int] = lambda _grab: _PALM_DEPTH_MM) -> None:
        from src.config.schema.camera import RGBDDeviceRigConfig

        super().__init__(rig_id="wrist", rig=RGBDDeviceRigConfig.model_construct(),
                         calibration=calibration if calibration is not None else _wrist())
        self.events = events if events is not None else []
        self.frames: list[RGBDFrame] = []
        self._depth_mm = depth_mm

    def grab(self) -> RGBDFrame:
        self.grabs += 1
        self.events.append("grab")
        frame = RGBDFrame(color=np.zeros((480, 640, 3), dtype=np.uint8),
                          depth=np.full((480, 640), self._depth_mm(self.grabs), dtype=np.uint16))
        self.frames.append(frame)
        return frame


def _observation(centre: tuple[float, float] = _PALM_PX) -> HandObservation:
    points = [(0, 0)] * LANDMARK_COUNT
    for index in PALM_LANDMARKS:
        points[int(index)] = (int(centre[0]), int(centre[1]))
    return HandObservation(
        palm=PalmDetection(palm_center_xy=centre, landmarks=as_landmark_tuple(points), handedness=Handedness.RIGHT),
        gesture=GestureReading(gesture=HandGesture.NONE, confidence=0.0),
    )


class _OneOpenHand:
    """The 2-D model: one open hand, its palm at `_PALM_PX`, in every frame it is shown."""

    def __init__(self) -> None:
        self.frames_seen = 0

    def observe(self, frame_bgr: np.ndarray) -> list[HandObservation]:
        self.frames_seen += 1
        return [_observation()]


def _wrist_finder(camera: _Camera, *, tool_pose: Any = "still", tool_frame: Any = _TOOL_FRAME,
                  **kwargs: Any) -> Any:
    reader = _Reader(_tool()) if tool_pose == "still" else tool_pose
    return build_hand_finder_on_camera(HandDetectConfig(model_path=__file__), camera, observer=_OneOpenHand(),
                                       tool_pose=reader, tool_frame=tool_frame, **kwargs)


def _palm_in_base(tool: Pose, depth_mm: float = _PALM_DEPTH_MM) -> np.ndarray:
    """The palm at `_PALM_PX` back-projected by hand, then carried to BASE by `tool` and `_CAMERA_TO_TOOL`."""
    u, v = _PALM_PX
    z = float(depth_mm)
    palm_cam = np.array([(u - _K[0, 2]) * z / _K[0, 0], (v - _K[1, 2]) * z / _K[1, 1], z, 1.0])
    return (tool.to_matrix() @ _CAMERA_TO_TOOL @ palm_cam)[:3]


class OneCameraServesTheRigSurfaceTests(unittest.TestCase):

    def test_it_holds_exactly_the_one_rig(self) -> None:
        self.assertEqual(["overhead"], OneCamera(_Camera()).rig_ids)

    def test_a_grab_goes_to_that_camera_and_opens_nothing(self) -> None:
        camera = _Camera(frame="a frame")
        frames = OneCamera(camera)
        self.assertEqual("a frame", frames.grab("overhead"))
        self.assertEqual(1, camera.grabs)

    def test_another_rig_id_is_refused_rather_than_answered_from_this_one(self) -> None:
        """One camera's frames under another rig's name is how a confident wrong answer is made."""
        with self.assertRaises(KeyError) as caught:
            OneCamera(_Camera()).grab("wrist")
        self.assertIn("overhead", str(caught.exception))

    def test_it_reports_rgbd_from_the_rigs_own_type(self) -> None:
        self.assertTrue(OneCamera(_rgbd_camera()).is_rgbd("overhead"))
        self.assertFalse(OneCamera(_Camera()).is_rgbd("overhead"))

    def test_a_stereo_index_is_refused_rather_than_guessed_as_zero(self) -> None:
        """Returning 0 would triangulate against whichever calibration sat first in the file."""
        with self.assertRaises(ValueError) as caught:
            OneCamera(_Camera()).get_stereo_rig_index("overhead")
        self.assertIn("StereoCam3D", str(caught.exception))


class TheCameraDoorReadsTheCamerasOwnFactsTests(unittest.TestCase):

    def _finder(self, camera: Any) -> Any:
        return build_hand_finder_on_camera(
            HandDetectConfig(model_path=__file__), camera, observer=object())

    def test_the_transform_is_the_rigs_declared_camera_to_base(self) -> None:
        finder = self._finder(_rgbd_camera())
        np.testing.assert_allclose(_camera_to_base().to_matrix(), finder.transforms["overhead"])

    def test_the_intrinsics_are_the_devices_own(self) -> None:
        np.testing.assert_allclose(_K, self._finder(_rgbd_camera()).camera_matrices["overhead"])

    def test_the_search_reaches_that_rig_and_no_other(self) -> None:
        finder = self._finder(_rgbd_camera())
        self.assertEqual(["overhead"], finder.rig_ids)
        self.assertIsInstance(finder.provider, OneCamera)

    def test_the_depth_knobs_still_travel(self) -> None:
        config = HandDetectConfig(model_path=__file__, palm_patch_radius_px=33, min_depth_samples=7)
        finder = build_hand_finder_on_camera(config, _rgbd_camera(), observer=object())
        self.assertEqual(33, finder.palm_patch_radius_px)
        self.assertEqual(7, finder.min_depth_samples)


class WhatTheCameraDoorRefusesTests(unittest.TestCase):

    def test_a_wrist_rig_without_the_arms_tcp_reader_is_refused_naming_the_fix(self) -> None:
        """Where a wrist camera stood is where the tool stood at the shutter; without a reader nobody knows."""
        camera = _WristCamera()
        with self.assertRaises(RigCalibrationError) as caught:
            _wrist_finder(camera, tool_pose=None)
        self.assertIn("tool_pose=robot.arm.get_tcp_pose", str(caught.exception))
        self.assertEqual(0, camera.grabs, "refused before anything was grabbed")

    def test_a_wrist_rig_without_the_cells_tool_frame_is_refused_naming_the_fix(self) -> None:
        """As the Locator holds it: CAMERA to TOOL is only true for the flange to TCP it was solved against."""
        camera, reader = _WristCamera(), _Reader(_tool())
        with self.assertRaises(RigCalibrationError) as caught:
            _wrist_finder(camera, tool_pose=reader, tool_frame=None)
        self.assertIn("tool_frame=tree.robot.gripper.tool_frame", str(caught.exception))
        self.assertEqual((0, 0), (camera.grabs, reader.reads))

    def test_a_wrist_rig_solved_against_another_flange_to_tcp_is_refused(self) -> None:
        """A stale record puts every palm off by the difference: the Locator's refusal, word for word."""
        moved = np.eye(4)
        moved[:3, 3] = (0.0, 0.0, 12.0)
        for label, record, says in (("no record", UNSET, "records none"),
                                    ("a stale record", FlangeToTcp.from_matrix("willy", moved), "stale")):
            with self.subTest(label):
                camera = _WristCamera(calibration=_wrist(record=record))
                with self.assertRaises(RigCalibrationError) as caught:
                    _wrist_finder(camera)
                self.assertIn(says, str(caught.exception))
                self.assertEqual(0, camera.grabs)

    def test_an_rgbd_rig_with_no_intrinsics_refuses_instead_of_inventing_a_lens(self) -> None:
        """A back-projection through an invented K completes and returns an unmeasured position."""
        camera = _rgbd_camera(intrinsics=None)
        with self.assertRaises(ValueError) as caught:
            build_hand_finder_on_camera(HandDetectConfig(model_path=__file__), camera,
                                        observer=object())
        self.assertIn("no intrinsics", str(caught.exception))

    def test_an_uncalibrated_rig_refuses_through_its_own_error(self) -> None:
        class _Uncalibrated:
            def camera_to_base(self) -> Transform:
                raise RigNotCalibrated("rig 'overhead' declares no calibration.")

        camera = _rgbd_camera(calibration=_Uncalibrated())
        with self.assertRaises(RigNotCalibrated):
            build_hand_finder_on_camera(HandDetectConfig(model_path=__file__), camera,
                                        observer=object())


class AWristCameraPlacesEachFrameByItsShutterTests(unittest.TestCase):
    """The owner's cell, 2026-10-01: the hand-over over a camera on the wrist, the arm holding still while it looks."""

    def test_the_palm_is_placed_by_the_tool_pose_at_the_shutter_and_the_camera_to_tool(self) -> None:
        tool = _tool()
        finder = _wrist_finder(_WristCamera(), tool_pose=_Reader(tool))

        seen, _annotated = finder.find_hand()

        assert seen is not None
        np.testing.assert_allclose(seen.position.position_base, _palm_in_base(tool), atol=1e-6)
        self.assertEqual("wrist", seen.position.rig_id)

    def test_a_frame_the_arm_moved_across_on_every_attempt_gives_no_hand(self) -> None:
        """Placed by no single pose, so placed by none: no stale pose, no guessed one."""
        camera = _WristCamera()
        moving = _Reader(*[_tool(dx_mm=5.0 * n) for n in range(40)])
        finder = _wrist_finder(camera, tool_pose=moving, attempts=2)

        self.assertEqual((None, None), finder.find_hand())

        from src.robot.core.shutter_motion import PICK_FRAME_WARMUP_GRABS

        self.assertEqual(PICK_FRAME_WARMUP_GRABS + 3, camera.grabs, "the warm-ups, then the first grab and two more")

    def test_a_grab_the_tool_moved_across_is_taken_again_and_placed_by_the_still_one(self) -> None:
        """The palm is read from the still grab's own frame, never from the moved one with the still pose."""
        from src.robot.core.shutter_motion import PICK_FRAME_WARMUP_GRABS

        still = _tool()
        reader = _Reader(_tool(dx_mm=-10.0), _tool(dx_mm=-5.0), still, still)
        finder = _wrist_finder(_WristCamera(depth_mm=lambda grab: 400 + 10 * grab), tool_pose=reader)

        seen, _annotated = finder.find_hand()

        assert seen is not None
        kept_mm = 400 + 10 * (PICK_FRAME_WARMUP_GRABS + 2)   # the warm-ups, the moved grab, then the still one
        self.assertEqual(float(kept_mm), seen.position.depth_mm)
        np.testing.assert_allclose(seen.position.position_base, _palm_in_base(still, kept_mm), atol=1e-6)
        self.assertEqual(4, reader.reads, "two grabs, the pose read before and after each")

    def test_a_look_the_arm_moved_across_after_a_good_one_gives_no_hand(self) -> None:
        """Never a stale placement: the good look before it places nothing of a look the arm moved across."""
        camera, here = _WristCamera(), _tool()
        finder = _wrist_finder(camera, tool_pose=_Reader(here, here, *[_tool(dx_mm=5.0 * n) for n in range(1, 40)]),
                               attempts=2)

        first, _annotated = finder.find_hand()
        assert first is not None
        np.testing.assert_allclose(first.position.position_base, _palm_in_base(here), atol=1e-6)

        self.assertEqual((None, None), finder.find_hand())
        moved = camera.frames[-1]
        self.assertIsNone(finder.provider.camera_to_base_of(moved))
        self.assertIsNone(finder.locate(_observation(), moved, "wrist"))

    def test_the_rigs_tolerance_holds_in_millimetres_and_in_degrees(self) -> None:
        """The rig's (1.0 mm, 0.5 deg), in that order, and a turn counts: 1 deg at 0.5 m moves a palm about 9 mm."""
        rows = (("0.9 mm per read, inside 1.0 mm", lambda n: _tool(dx_mm=0.9 * n), True),
                ("1.1 mm per read, beyond 1.0 mm", lambda n: _tool(dx_mm=1.1 * n), False),
                ("0.4 deg per read, inside 0.5 deg", lambda n: _tool(turn_deg=0.4 * n), True),
                ("0.75 deg per read, beyond 0.5 deg", lambda n: _tool(turn_deg=0.75 * n), False))
        for label, read_n, placed in rows:
            with self.subTest(label):
                reader = _Reader(*[read_n(n) for n in range(40)])
                seen, _annotated = _wrist_finder(_WristCamera(), tool_pose=reader).find_hand()
                if not placed:
                    self.assertIsNone(seen)
                    continue
                assert seen is not None
                np.testing.assert_allclose(seen.position.position_base, _palm_in_base(_tool()), atol=1e-6,
                                           err_msg="placed by the pose read just before the kept grab")

    def test_an_earlier_frame_is_never_placed_by_a_later_shutter(self) -> None:
        """The arm moved between two looks: the first frame keeps no placement, least of all the second's."""
        camera = _WristCamera()
        here, there = _tool(), _tool(dx_mm=80.0)
        finder = _wrist_finder(camera, tool_pose=_Reader(here, here, there))
        first, _ = finder.find_hand()
        earlier = camera.frames[-1]
        second, _ = finder.find_hand()
        later = camera.frames[-1]

        assert first is not None and second is not None
        np.testing.assert_allclose(second.position.position_base, _palm_in_base(there), atol=1e-6)
        self.assertIsNone(finder.locate(_observation(), earlier, "wrist"))
        later_seen = finder.locate(_observation(), later, "wrist")
        assert later_seen is not None
        np.testing.assert_allclose(later_seen.position_base, _palm_in_base(there), atol=1e-6)

    def test_the_frame_is_the_pick_frames_warm_ups_then_its_stamp(self) -> None:
        """The frames a device queued before the pose was read are thrown away first, as the pick frame's are."""
        import inspect

        from src.robot.core.shutter_motion import PICK_FRAME_WARMUP_GRABS
        from src.robot.perception.realsense_source import RealSenseVisionPerceptionSource

        events: list[str] = []
        finder = _wrist_finder(_WristCamera(events=events), tool_pose=_Reader(_tool(), events=events))
        finder.find_hand()

        self.assertEqual(["grab"] * PICK_FRAME_WARMUP_GRABS + ["pose", "grab", "pose"], events)
        pick_frame = inspect.signature(RealSenseVisionPerceptionSource.__init__).parameters["warmup_grabs"].default
        self.assertEqual(pick_frame, PICK_FRAME_WARMUP_GRABS)
        self.assertEqual(5, PICK_FRAME_WARMUP_GRABS, "the READMEs' five; with none, a frame queued in motion is kept")

    def test_the_attempts_are_the_locators_unless_given(self) -> None:
        from src.config.schema.robot.safety_schema import PerceivedWorldConfig
        from src.robot.core.shutter_motion import PICK_FRAME_WARMUP_GRABS

        default = int(PerceivedWorldConfig.model_fields["fresh_frame_attempts"].default)
        for label, kwargs, grabs in (("unset", {}, 1 + default), ("one", {"attempts": 1}, 2)):
            with self.subTest(label):
                camera = _WristCamera()
                finder = _wrist_finder(camera, tool_pose=_Reader(*[_tool(dx_mm=5.0 * n) for n in range(40)]),
                                       **kwargs)
                finder.find_hand()
                self.assertEqual(PICK_FRAME_WARMUP_GRABS + grabs, camera.grabs)

    def test_the_search_holds_one_wrist_camera_and_no_fixed_transform(self) -> None:
        finder = _wrist_finder(_WristCamera())
        self.assertIsInstance(finder.provider, OneWristCamera)
        self.assertEqual(["wrist"], finder.rig_ids)
        self.assertEqual({}, finder.transforms, "a camera on the wrist has no CAMERA to BASE of its own")

    def test_a_fixed_rig_reads_no_tool_pose_and_keeps_its_one_transform(self) -> None:
        """The control: a fixed camera's path is the one it was, whatever a caller hands in."""
        reader = _Reader(_tool())
        camera = _rgbd_camera(frame=RGBDFrame(color=np.zeros((480, 640, 3), dtype=np.uint8),
                                              depth=np.full((480, 640), _PALM_DEPTH_MM, dtype=np.uint16)))
        finder = build_hand_finder_on_camera(HandDetectConfig(model_path=__file__), camera, observer=_OneOpenHand(),
                                             tool_pose=reader, tool_frame=_TOOL_FRAME)
        seen, _annotated = finder.find_hand()

        self.assertIs(OneCamera, type(finder.provider))
        self.assertEqual({}, finder.frame_transforms)
        np.testing.assert_allclose(_camera_to_base().to_matrix(), finder.transforms["overhead"])
        assert seen is not None
        u, v = _PALM_PX
        palm_cam = np.array([(u - 320.0) * 500.0 / 600.0, (v - 240.0) * 500.0 / 600.0, 500.0])
        np.testing.assert_allclose(seen.position.position_base, palm_cam + [100.0, 0.0, 800.0], atol=1e-9)
        self.assertEqual((0, 1), (reader.reads, camera.grabs))


class OnePlacementForEveryWristFrameTests(unittest.TestCase):
    """The hand finder and the Locator place a wrist frame by one composition, so they cannot disagree."""

    def test_a_palm_and_a_part_seen_at_one_pixel_of_one_wrist_frame_are_placed_alike(self) -> None:
        from dataclasses import dataclass

        from src.robot.perception.locator import Locator

        u, v = (int(c) for c in _PALM_PX)
        mask = np.zeros((480, 640), dtype=bool)
        mask[v - 1:v + 2, u - 1:u + 2] = True   # three by three around the palm's pixel, so its median is that pixel

        @dataclass(frozen=True)
        class _Segmentation:
            label: str
            score: float
            bbox_xyxy: tuple
            mask: np.ndarray

        seg = _Segmentation(label="part", score=0.9, bbox_xyxy=(u - 1.0, v - 1.0, u + 1.0, v + 1.0), mask=mask)
        backend = SimpleNamespace(perceive=lambda _bgr, _prompt: (
            SimpleNamespace(detection=SimpleNamespace(score=0.9, box=list(seg.bbox_xyxy)), segmentation=seg),))
        camera = _WristCamera()
        owner = SimpleNamespace(rig_id="wrist", source="rgbd", handle=lambda: camera, calibration=lambda: _wrist())
        tool = _tool()

        located = Locator.from_parts(camera=owner, backend=backend, tool_pose=_Reader(tool),
                                     tool_frame=_TOOL_FRAME).locate("a part")
        seen, _annotated = _wrist_finder(_WristCamera(), tool_pose=_Reader(tool)).find_hand()

        assert seen is not None and located.objects[0].centre_mm is not None
        np.testing.assert_allclose(seen.position.position_base, located.objects[0].centre_mm, atol=1e-3)
        np.testing.assert_allclose(seen.position.position_base, _palm_in_base(tool), atol=1e-6)

    def test_every_wrist_frame_of_the_hand_finder_is_placed_by_the_locators_composition(self) -> None:
        """Not a copy of it: the one helper, handed the frame's own stamped pose."""
        from unittest import mock

        from src.robot.core import shutter_motion
        from src.robot.perception import locator

        self.assertIs(shutter_motion.camera_to_base_at_shutter, locator.camera_to_base_at_shutter)
        camera, tool = _WristCamera(), _tool()
        with mock.patch.object(shutter_motion, "camera_to_base_at_shutter",
                               wraps=shutter_motion.camera_to_base_at_shutter) as placed:
            seen, _annotated = _wrist_finder(camera, tool_pose=_Reader(tool)).find_hand()
        assert seen is not None
        placed.assert_called_once_with(camera.calibration(), tool)


class TheDoorIsExportedTests(unittest.TestCase):

    def test_willy_exports_it(self) -> None:
        import willy

        self.assertIs(build_hand_finder_on_camera, willy.build_hand_finder_on_camera)

    def test_the_package_re_exports_it(self) -> None:
        import src.models.handdetection as package

        self.assertIn("build_hand_finder_on_camera", package.__all__)
        self.assertIs(build_hand_finder_on_camera, package.build_hand_finder_on_camera)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
