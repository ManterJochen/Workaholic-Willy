"""A pick frame carries the time its shutter opened, and a wrist camera's frame is taken with the tool held still.

The RealSense source stamps the owner's capture time, or a clock read just before the grab; the four Isaac sources
stamp the clock read just before the last render step whose buffers they read. A wrist source reads the TCP before and after its grab, grabs again when the tool moved
beyond the rig's shutter tolerance, and raises ``PerceptionFrameMoved`` after its attempts. The pick frame and the
live world judge that motion by one rule, and a moved fused wrist camera stops the pick rather than going absent.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from src.calibration.rig_calibration import RigCalibration, RigCalibrationError
from src.calibration.serialization import FlangeToTcp
from src.camera.setup.image_taking.frames import RGBDFrame
from src.geometry import Frame, Pose, Transform
from src.robot.core.errors import PerceptionFrameMoved
from src.robot.core.shutter_motion import ShutterMotion, ShutterStamp, camera_to_base_at_shutter
from src.robot.perception import RealSenseVisionPerceptionSource

_K = np.array([[600.0, 0.0, 4.0], [0.0, 600.0, 4.0], [0.0, 0.0, 1.0]])


def _pose(x: float = 400.0, *, frame: Frame = Frame.BASE) -> Pose:
    return Pose(position_mm=np.array([x, 0.0, 300.0]), quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), frame=frame)


def _turned_by(deg: float) -> Pose:
    """The tool of ``_pose()`` turned ``deg`` about BASE Z where it stands: no millimetre of travel."""
    half = np.radians(deg) / 2.0
    return Pose(position_mm=np.array([400.0, 0.0, 300.0]),
                quaternion_xyzw=np.array([0.0, 0.0, np.sin(half), np.cos(half)]), frame=Frame.BASE)


class _Streamer:
    """RGB-D frames on every grab, each one counted and kept, optionally carrying the owner's capture time."""

    def __init__(self, *, captured_at_s: float | None = None, events: list | None = None) -> None:
        self.grabs = 0
        self.rig_id = "wrist"
        self._captured = captured_at_s
        self._events = events
        self.frames: list[RGBDFrame] = []

    def grab(self) -> RGBDFrame:
        self.grabs += 1
        if self._events is not None:
            self._events.append("grab")
        frame = RGBDFrame(color=np.zeros((8, 8, 3), dtype=np.uint8), depth=np.full((8, 8), 500, dtype=np.uint16),
                          captured_at_s=self._captured)
        self.frames.append(frame)
        return frame

    def get_intrinsics(self) -> np.ndarray:
        return _K.copy()


def _source(streamer: _Streamer, *, warmup_grabs: int = 0) -> RealSenseVisionPerceptionSource:
    return RealSenseVisionPerceptionSource(streamer=streamer, backend=SimpleNamespace(perceive=lambda _b, _p: []),
                                           prompt="a box", warmup_grabs=warmup_grabs)


class _Reader:
    """A TCP reader answering a fixed sequence of poses, then the last one forever."""

    def __init__(self, poses: list, events: list | None = None) -> None:
        self._poses = list(poses)
        self._events = events
        self.reads = 0

    def __call__(self) -> Pose:
        if self._events is not None:
            self._events.append("pose")
        pose = self._poses[min(self.reads, len(self._poses) - 1)]
        self.reads += 1
        return pose


class TheRealSenseStampTests(unittest.TestCase):
    def test_the_realsense_frame_carries_the_owners_capture_time(self) -> None:
        frame = _source(_Streamer(captured_at_s=1234.5)).acquire()
        self.assertEqual(1234.5, frame.timestamp)

    def test_a_direct_streamer_frame_is_stamped_before_its_grab(self) -> None:
        readings = iter(float(n) for n in range(100, 200))
        streamer = _Streamer()
        seen_at_grab: list[float] = []
        clock_now = [0.0]

        def clock() -> float:
            clock_now[0] = next(readings)
            return clock_now[0]

        grab = streamer.grab

        def recording_grab() -> RGBDFrame:
            seen_at_grab.append(clock_now[0])
            return grab()

        streamer.grab = recording_grab  # type: ignore[method-assign]
        with mock.patch("src.robot.perception.realsense_source.time.time", side_effect=clock):
            frame = _source(streamer).acquire()
        self.assertEqual(seen_at_grab[-1], frame.timestamp, "the stamp was read after the grab")


class AWristFrameIsTakenWithTheToolStillTests(unittest.TestCase):
    def test_a_wrist_frame_taken_while_the_arm_moved_is_grabbed_again(self) -> None:
        streamer = _Streamer(captured_at_s=10.0)
        source = _source(streamer)
        still = _pose(400.0)
        source.stamp_tool_pose_with(_Reader([_pose(400.0), _pose(405.0), still, still]),
                                    motion_tolerance=(1.0, 0.5), attempts=2)
        frame = source.acquire()
        self.assertEqual(2, streamer.grabs)
        self.assertEqual(still, frame.tool_pose)

    def test_a_wrist_frame_that_keeps_moving_raises_after_its_attempts(self) -> None:
        streamer = _Streamer(captured_at_s=10.0)
        source = _source(streamer)
        poses = [_pose(400.0 + 5.0 * n) for n in range(10)]
        source.stamp_tool_pose_with(_Reader(poses), motion_tolerance=(1.0, 0.5), attempts=2)
        with self.assertRaises(PerceptionFrameMoved) as caught:
            source.acquire()
        self.assertEqual(3, streamer.grabs)
        self.assertEqual("wrist", caught.exception.camera)
        self.assertEqual(3, caught.exception.attempts)
        self.assertAlmostEqual(5.0, caught.exception.moved_mm)

    def test_a_regrab_does_not_repeat_the_warm_ups(self) -> None:
        streamer = _Streamer(captured_at_s=10.0)
        source = _source(streamer, warmup_grabs=3)
        still = _pose(400.0)
        source.stamp_tool_pose_with(_Reader([_pose(400.0), _pose(405.0), still, still]),
                                    motion_tolerance=(1.0, 0.5), attempts=2)
        source.acquire()
        self.assertEqual(3 + 2, streamer.grabs)

    def test_a_dummy_arm_reader_stamps_a_resting_pose(self) -> None:
        from src.robot.drivers.dummy.arm import DummyRobotArm

        arm = DummyRobotArm(initial_pose=_pose(420.0))
        arm.connect()
        source = _source(_Streamer(captured_at_s=10.0))
        source.stamp_tool_pose_with(arm.get_tcp_pose, motion_tolerance=(1.0, 0.5), attempts=1)
        self.assertEqual(arm.get_tcp_pose(), source.acquire().tool_pose)

    def test_the_binding_is_refused_without_a_tolerance_or_with_negative_attempts(self) -> None:
        source = _source(_Streamer())
        with self.assertRaises(TypeError):
            source.stamp_tool_pose_with(_Reader([_pose()]))  # type: ignore[call-arg]
        with self.assertRaises(ValueError):
            source.stamp_tool_pose_with(_Reader([_pose()]), motion_tolerance=(1.0, 0.5), attempts=-1)
        with self.assertRaises(ValueError):
            source.stamp_tool_pose_with(_Reader([_pose()]), motion_tolerance=(-1.0, 0.5), attempts=1)


class OneMotionRuleTests(unittest.TestCase):
    def test_the_pick_frame_and_the_world_judge_motion_by_one_rule(self) -> None:
        from src.robot.safety.planning.depth_source import RigDepthSource

        source = RigDepthSource(_Streamer(), tool_pose=lambda: _pose(), motion_tolerance=(1.0, 0.5))
        turned = Pose(position_mm=np.array([400.0, 0.0, 300.0]),
                      quaternion_xyzw=np.array([0.0, 0.0, 0.00872654, 0.99996192]), frame=Frame.BASE)
        rows = (
            ("still", _pose(), _pose(), True),
            ("moved 0.9 mm", _pose(), _pose(400.9), True),
            ("moved 1.1 mm", _pose(), _pose(401.1), False),
            ("turned 1 deg", _pose(), turned, False),
            ("a pose in another frame", _pose(), _pose(frame=Frame.TOOL), False),
        )
        for label, before, after, within in rows:
            with self.subTest(label):
                motion = ShutterMotion.between(before, after, tolerance_mm=1.0, tolerance_deg=0.5)
                self.assertIs(within, motion.within)
                self.assertIs(within, source._held_still(before, after))  # noqa: SLF001


class TheStampOnItsOwnTests(unittest.TestCase):
    """The pick frame's stamp as one object, so a hand finder over a wrist camera takes its frames through it too."""

    def test_a_still_grab_carries_the_pose_read_before_it(self) -> None:
        streamer = _Streamer()
        before = _pose(400.0)
        grabbed = ShutterStamp.of(_Reader([before, _pose(400.5)]), motion_tolerance=(1.0, 0.5),
                                  attempts=2).grab(streamer.grab)
        self.assertEqual(before, grabbed.tool_pose)
        self.assertEqual((1, 1), (grabbed.grabs, streamer.grabs))
        self.assertAlmostEqual(0.5, grabbed.motion.moved_mm)

    def test_after_its_attempts_the_stamp_hands_back_the_last_frame_with_no_pose(self) -> None:
        """No pose places a frame the tool moved across, and the caller decides what that means."""
        streamer = _Streamer()
        grabbed = ShutterStamp.of(_Reader([_pose(400.0 + 5.0 * n) for n in range(10)]), motion_tolerance=(1.0, 0.5),
                                  attempts=2).grab(streamer.grab)
        self.assertIsNone(grabbed.tool_pose)
        self.assertEqual((3, 3), (grabbed.grabs, streamer.grabs))
        self.assertIsInstance(grabbed.frame, RGBDFrame)
        self.assertFalse(grabbed.motion.within)
        self.assertAlmostEqual(5.0, grabbed.motion.moved_mm)

    def test_a_retaken_grab_keeps_the_frame_of_the_attempt_its_pose_places(self) -> None:
        """The still pose places the still grab's frame, never the frame the tool moved across."""
        streamer, still = _Streamer(), _pose(400.0)
        grabbed = ShutterStamp.of(_Reader([_pose(395.0), _pose(400.0), still, still]), motion_tolerance=(1.0, 0.5),
                                  attempts=2).grab(streamer.grab)
        self.assertEqual((2, 2), (grabbed.grabs, len(streamer.frames)))
        self.assertIs(streamer.frames[-1], grabbed.frame)
        self.assertEqual(still, grabbed.tool_pose)

    def test_a_turn_is_judged_against_the_degrees_though_the_tool_point_stood_still(self) -> None:
        """No millimetre of travel, so only the rig's degrees can see it: 0.4 deg is kept, 0.75 deg is taken again."""
        for label, turn_deg, grabs in (("inside 0.5 deg", 0.4, 1), ("beyond 0.5 deg", 0.75, 2)):
            with self.subTest(label):
                streamer, turned = _Streamer(), _turned_by(turn_deg)
                grabbed = ShutterStamp.of(_Reader([_pose(), turned, turned, turned]), motion_tolerance=(1.0, 0.5),
                                          attempts=2).grab(streamer.grab)
                self.assertEqual(grabs, grabbed.grabs)
                self.assertIs(streamer.frames[-1], grabbed.frame)
                self.assertEqual(_pose() if grabs == 1 else turned, grabbed.tool_pose)
                self.assertAlmostEqual(0.0, grabbed.motion.moved_mm)

    def test_a_stamp_is_refused_without_a_reader_a_finite_tolerance_or_with_negative_attempts(self) -> None:
        with self.assertRaises(TypeError):
            ShutterStamp.of(None, motion_tolerance=(1.0, 0.5), attempts=1)  # type: ignore[arg-type]
        for tolerance, attempts in (((-1.0, 0.5), 1), ((1.0, float("nan")), 1), ((1.0, 0.5), -1)):
            with self.subTest(tolerance=tolerance, attempts=attempts), self.assertRaises(ValueError):
                ShutterStamp.of(_Reader([_pose()]), motion_tolerance=tolerance, attempts=attempts)


#: A camera on the tool, tilted 45 degrees about the tool's X and set 60 mm out and 40 mm back.
_CAMERA_TO_TOOL = Transform(translation_mm=np.array([60.0, 0.0, -40.0]),
                            quaternion_xyzw=np.array([np.sin(np.radians(22.5)), 0.0, 0.0, np.cos(np.radians(22.5))]),
                            from_frame=Frame.CAMERA, to_frame=Frame.TOOL)
#: A tool turned 30 degrees about BASE Z.
_TURNED = Pose(position_mm=np.array([400.0, -150.0, 600.0]),
               quaternion_xyzw=np.array([0.0, 0.0, np.sin(np.radians(15.0)), np.cos(np.radians(15.0))]), frame=Frame.BASE)


def _wrist_calibration() -> RigCalibration:
    return RigCalibration(rig_id="wrist", mounting_mode="eye_in_hand", artifact_path="eih.json", transform=_CAMERA_TO_TOOL,
                          shutter_motion_tolerance_mm=1.0, shutter_motion_tolerance_deg=0.5,
                          flange_to_tcp=FlangeToTcp.from_matrix("willy", np.eye(4)))


class OneCompositionPlacesEveryFrameTests(unittest.TestCase):
    """``camera_to_base_at_shutter`` is where a frame is put in BASE, for the Locator and the hand finder alike."""

    def test_a_wrist_frame_is_its_tool_pose_composed_with_its_camera_to_tool_bit_for_bit(self) -> None:
        """The Locator's own arithmetic, so what it locates did not move by a bit."""
        placed = camera_to_base_at_shutter(_wrist_calibration(), _TURNED)
        expected = (np.asarray(_TURNED.to_matrix(), dtype=np.float64)
                    @ np.asarray(_CAMERA_TO_TOOL.to_matrix(), dtype=np.float64))
        self.assertTrue(np.array_equal(expected, placed))
        self.assertEqual(np.float64, placed.dtype)

    def test_a_fixed_frame_is_its_calibrations_camera_to_base(self) -> None:
        camera_to_base = Transform(translation_mm=np.array([100.0, 0.0, 800.0]),
                                   quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]), from_frame=Frame.CAMERA,
                                   to_frame=Frame.BASE)
        fixed = RigCalibration(rig_id="overhead", mounting_mode="eye_to_hand", artifact_path="eth.json",
                               transform=camera_to_base)
        self.assertTrue(np.array_equal(camera_to_base.to_matrix(), camera_to_base_at_shutter(fixed, None)))

    def test_a_wrist_frame_nobody_stamped_is_never_placed(self) -> None:
        with self.assertRaises(RigCalibrationError):
            camera_to_base_at_shutter(_wrist_calibration(), None)

    def test_a_tool_pose_outside_base_places_nothing(self) -> None:
        for label, pose in (("a TOOL frame pose", _pose(frame=Frame.TOOL)), ("no pose at all", "a pose")):
            with self.subTest(label), self.assertRaises(ValueError):
                camera_to_base_at_shutter(_wrist_calibration(), pose)  # type: ignore[arg-type]

    def test_the_locator_places_each_wrist_frame_by_it(self) -> None:
        from src.robot.perception import locator

        calibration = _wrist_calibration()
        owner = SimpleNamespace(rig_id="wrist", source="rgbd", handle=lambda: _Streamer(captured_at_s=10.0),
                                calibration=lambda: calibration)
        tool_frame = SimpleNamespace(source="willy", offset_mm=(0.0, 0.0, 0.0),
                                     rotation_quat_xyzw=(0.0, 0.0, 0.0, 1.0), verify_tolerance_mm=1.0)
        found = locator.Locator.from_parts(camera=owner, backend=SimpleNamespace(perceive=lambda _b, _p: []),
                                           tool_pose=_Reader([_TURNED]), tool_frame=tool_frame)
        with mock.patch.object(locator, "camera_to_base_at_shutter", wraps=camera_to_base_at_shutter) as placed:
            located = found.locate("a box")
        placed.assert_called_once_with(calibration, _TURNED)
        self.assertEqual(tuple(tuple(float(v) for v in row) for row in _TURNED.to_matrix()), located.tool_to_base_mm)


class AMovedFusedWristCameraTests(unittest.TestCase):
    def test_a_moved_fused_wrist_camera_stops_the_pick(self) -> None:
        from src.robot.grasping.types.perception import MappedCameraRig

        class _Moved:
            def acquire(self):  # noqa: ANN202
                raise PerceptionFrameMoved(camera="side_wrist", attempts=3, moved_mm=4.0, turned_deg=0.0,
                                           tolerance_mm=1.0, tolerance_deg=0.5)

        class _Absent:
            def acquire(self):  # noqa: ANN202
                raise RuntimeError("the pipeline never started")

        with self.assertRaises(PerceptionFrameMoved):
            MappedCameraRig({"side_wrist": _Moved()}).acquire_all()
        rig = MappedCameraRig({"side": _Absent()})
        self.assertEqual((), rig.acquire_all(), "the control: any other failure is still an absent camera")
        self.assertIn("side", rig.last_failures)


class EverySimSourceIsStampedBeforeItsLastRenderTests(unittest.TestCase):
    """Each Isaac source's stamp is the clock read just before the render step whose buffers it reads."""

    class _Session:
        def __init__(self, clock: "list[float]") -> None:
            self.clock = clock
            self.at_step: list[float] = []
            self.app = SimpleNamespace(update=lambda: None)

        def step(self, *, render: bool = False) -> None:
            self.at_step.append(self.clock[0])

    @staticmethod
    def _clock() -> "tuple[list[float], object]":
        now = [0.0]
        counter = iter(float(n) for n in range(1000, 5000))

        def tick() -> float:
            now[0] = next(counter)
            return now[0]

        return now, SimpleNamespace(time=tick, perf_counter=lambda: 0.0)

    def _check(self, module: str, build) -> None:  # noqa: ANN001
        now, clock = self._clock()
        session = self._Session(now)
        source = build(session)
        with mock.patch(f"{module}.time", clock):
            frame = source.acquire()
        self.assertTrue(session.at_step, "the source pumped no render step")
        self.assertEqual(session.at_step[-1], frame.timestamp)

    def test_every_sim_source_is_stamped_before_its_last_render(self) -> None:
        from src.willy_sim import GroundTruthPerceptionSource
        from src.willy_sim.perception import (
            IsaacVisionPerceptionSource,
            MultiObjectGroundTruthPerceptionSource,
            MultiObjectVisionPerceptionSource,
        )
        from tests.test_willy_sim_perception import _FakeCamera, _FakeDetector, _FakeSegmenter, _VisCam, _frame

        data = np.zeros((4, 4), dtype=np.uint32)
        data[1:3, 1:3] = 3
        rows = (
            ("src.willy_sim.perception.ground_truth", lambda s: GroundTruthPerceptionSource(
                camera=_FakeCamera(np.full((4, 4), 0.5), np.eye(3), _frame(data)), target_prim_path="/World/Cube",
                session=s, warmup_steps=2)),
            ("src.willy_sim.perception.ground_truth", lambda s: MultiObjectGroundTruthPerceptionSource(
                camera=_FakeCamera(np.full((4, 4), 0.5), np.eye(3), _frame(data)), targets=[("/World/Cube", "cube")],
                session=s, warmup_steps=2)),
            ("src.willy_sim.perception.vision", lambda s: MultiObjectVisionPerceptionSource(
                camera=_VisCam(np.full((4, 4), 5.0), np.zeros((4, 4, 3), dtype=np.uint8)),
                detector=_FakeDetector([]), segmenter=_FakeSegmenter({}), prompt="a cube", session=s, warmup_steps=2)),
            ("src.willy_sim.perception.vision", lambda s: IsaacVisionPerceptionSource(
                camera=_VisCam(np.full((4, 4), 5.0), np.zeros((4, 4, 3), dtype=np.uint8)),
                backend=SimpleNamespace(perceive=lambda _b, _p: ()), prompt="a cube", session=s, warmup_steps=2)),
        )
        for module, build in rows:
            with self.subTest(source=build(self._Session([0.0])).__class__.__name__):
                self._check(module, build)

    def test_every_sim_source_takes_the_name_of_its_camera(self) -> None:
        from src.contracts import UNSET
        from src.willy_sim.perception import IsaacVisionPerceptionSource
        from tests.test_willy_sim_perception import _VisCam

        cam = _VisCam(np.full((4, 4), 5.0), np.zeros((4, 4, 3), dtype=np.uint8))
        named = IsaacVisionPerceptionSource(camera=cam, backend=SimpleNamespace(perceive=lambda _b, _p: ()),
                                            prompt="x", camera_name="overhead")
        unnamed = IsaacVisionPerceptionSource(camera=cam, backend=SimpleNamespace(perceive=lambda _b, _p: ()),
                                              prompt="x")
        self.assertEqual("overhead", named.camera_name)
        self.assertIs(UNSET, unnamed.camera_name)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
