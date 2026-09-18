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

from src.camera.setup.image_taking.frames import RGBDFrame
from src.geometry import Frame, Pose
from src.robot.core.errors import PerceptionFrameMoved
from src.robot.core.shutter_motion import ShutterMotion
from src.robot.perception import RealSenseVisionPerceptionSource

_K = np.array([[600.0, 0.0, 4.0], [0.0, 600.0, 4.0], [0.0, 0.0, 1.0]])


def _pose(x: float = 400.0, *, frame: Frame = Frame.BASE) -> Pose:
    return Pose(position_mm=np.array([x, 0.0, 300.0]), quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), frame=frame)


class _Streamer:
    """RGB-D frames on every grab, counting grabs, optionally carrying the owner's capture time."""

    def __init__(self, *, captured_at_s: float | None = None, events: list | None = None) -> None:
        self.grabs = 0
        self.rig_id = "wrist"
        self._captured = captured_at_s
        self._events = events

    def grab(self) -> RGBDFrame:
        self.grabs += 1
        if self._events is not None:
            self._events.append("grab")
        return RGBDFrame(color=np.zeros((8, 8, 3), dtype=np.uint8), depth=np.full((8, 8), 500, dtype=np.uint16),
                         captured_at_s=self._captured)

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
