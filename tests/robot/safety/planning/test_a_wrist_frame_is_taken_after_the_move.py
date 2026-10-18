"""A wrist camera's world frame is one taken after the move, not the first one its stream hands back.

The audit of the owner's cell, 2026-09-23, found that the world refresh after every move took a single grab from the
wrist D415 with no warm-up, and stamped it around ``wait_for_frames``. The pick frame takes five warm-ups first; the
world took none. Two things came back in that one frame:

  * the RealSense temporal filter's history from the pose before: a pixel that is a hole at the new pose was filled
    with the depth it had there, and a frame that should have been BLIND was vouched FRESH with stale geometry.
    Measured with librealsense 2.58.3's own filter on a software device: the first frame at the new pose held the old
    depth in its holes, and from the fourth frame on the holes were holes again;
  * a frame the pipeline had queued while the arm was still coming to rest, placed at the pose read after it.

The source now reads the pose, tells a handle that can drop its filter history that the camera moved
(``camera_moved()``, the RealSense handle's), throws away ``WRIST_WARMUP_GRABS`` frames, keeps the next and reads the
pose again. The stream below models the filter's persistency rule ("valid in 2 of the last 4"), which is what the world
meets on the owner's cell; the warm-ups alone clear it, for a handle that cannot be told. Names that are new with this
change are imported inside the tests.
"""

from __future__ import annotations

import unittest
from collections import deque
from types import SimpleNamespace
from unittest import mock

import numpy as np

from src.geometry import Frame, Pose
from src.robot.safety.planning.depth_source import RigDepthSource
from src.robot.safety.planning.live_world import CameraView, LivePlannerWorld, WorldVerdict
from src.robot.safety.planning.perceived import LinkCapsule, SelfEnvelope, WorldBuildLimits, WorldBuildTuning

_K = np.array([[300.0, 0.0, 40.0], [0.0, 300.0, 30.0], [0.0, 0.0, 1.0]])
_SHAPE = (60, 80)
_MODULE = "src.robot.safety.planning.depth_source"


def _pose(x_mm: float = 400.0) -> Pose:
    return Pose(position_mm=np.array([x_mm, 0.0, 600.0]), quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]),
                frame=Frame.BASE)


class _FilteredStream:
    """A depth stream with a temporal filter's memory: a hole is filled from the frames before while the pixel was
    valid in at least two of the last four frames, this one included, which is librealsense's default persistency.

    ``scene`` is what the camera sees now; the test changes it to move the arm.
    """

    def __init__(self, scene: np.ndarray, *, stamps: bool = False) -> None:
        self.scene = scene
        self.rig_id = "wrist"
        self.grabs = 0
        self._raw: deque[np.ndarray] = deque(maxlen=4)
        self._last_valid = np.zeros(_SHAPE)
        self._stamps = stamps
        self.events: list[str] = []

    def grab(self) -> SimpleNamespace:
        self.grabs += 1
        self.events.append("grab")
        raw = np.asarray(self.scene, dtype=np.float64).copy()
        self._raw.append(raw)
        valid = raw > 0.0
        count = np.sum([frame > 0.0 for frame in self._raw], axis=0)
        out = np.where(valid, raw, np.where(count >= 2, self._last_valid, 0.0))
        self._last_valid = np.where(valid, raw, self._last_valid)
        frame = SimpleNamespace(depth=out, index=self.grabs)
        if self._stamps:
            frame.captured_at_s = 1000.0 + self.grabs
        return frame

    def get_intrinsics(self) -> np.ndarray:
        return _K


class _ResettableStream(_FilteredStream):
    """The same stream with the notice a RealSense handle takes: ``camera_moved()`` drops the filter's history."""

    def camera_moved(self) -> None:
        self.events.append("moved")
        self._raw.clear()
        self._last_valid = np.zeros(_SHAPE)


def _moved(stream: _FilteredStream) -> None:
    """Four frames at the pose before, then the arm moves: 80 % of the new view is closer than the camera measures."""
    stream.scene = np.full(_SHAPE, 600.0)
    for _ in range(4):
        stream.grab()
    near = np.zeros(_SHAPE)
    near[:, : _SHAPE[1] // 5] = 450.0
    stream.scene = near


class _Reader:
    def __init__(self, stream: _FilteredStream) -> None:
        self._stream = stream

    def __call__(self) -> Pose:
        self._stream.events.append("pose")
        return _pose()


def _wrist_source(stream: _FilteredStream, **kwargs: object) -> RigDepthSource:
    return RigDepthSource(stream, tool_pose=_Reader(stream), motion_tolerance=(1.0, 0.5), **kwargs)  # type: ignore[arg-type]


class TheFrameKeptIsTakenAfterTheMoveTests(unittest.TestCase):

    def test_the_holes_at_the_new_pose_are_holes(self) -> None:
        """⛔ The finding: the one frame grabbed after the move held the old pose's depth in 80 % of its pixels."""
        stream = _FilteredStream(np.zeros(_SHAPE))
        _moved(stream)
        reading = _wrist_source(stream).grab_surface_depth()
        assert reading is not None
        self.assertAlmostEqual(float(np.mean(reading.depth_mm > 0.0)), 0.2, places=6)

    def test_the_world_calls_that_frame_blind_instead_of_fresh(self) -> None:
        """⛔ The same frame through the live world: before, FRESH with stale geometry; now BLIND, and asked again."""
        stream = _FilteredStream(np.zeros(_SHAPE))
        _moved(stream)
        tool = np.diag([1.0, -1.0, -1.0, 1.0])
        world = LivePlannerWorld(
            cameras=(CameraView(name="wrist", depth_source=_wrist_source(stream), camera_to_tool=tool),),
            declared=(), limits=WorldBuildLimits(x_mm=(-800.0, 800.0), y_mm=(-800.0, 800.0), z_mm=(-100.0, 900.0),
                                                 support_plane_top_mm=0.0),
            tuning=WorldBuildTuning(max_boxes=4), max_age_ms=1e9,
        )
        body = SelfEnvelope(frames_mm=(np.eye(4),), capsules=(
            LinkCapsule(frame=0, start_mm=(0.0, 0.0, 0.0), end_mm=(0.0, 0.0, 100.0), radius_mm=50.0),))
        self.assertIs(world.world_for(self_envelope=body).verdict, WorldVerdict.BLIND)

    def test_the_pose_is_read_before_the_warm_ups_and_after_the_frame_kept(self) -> None:
        from src.robot.safety.planning.depth_source import WRIST_WARMUP_GRABS

        stream = _FilteredStream(np.full(_SHAPE, 600.0))
        reading = _wrist_source(stream).grab_surface_depth()
        assert reading is not None
        self.assertEqual(stream.events, ["pose", *["grab"] * (WRIST_WARMUP_GRABS + 1), "pose"])
        self.assertEqual(WRIST_WARMUP_GRABS, 5, "five, as the pick frame takes")

    def test_a_handle_that_can_drop_its_history_is_told_before_the_warm_ups(self) -> None:
        from src.robot.safety.planning.depth_source import WRIST_WARMUP_GRABS

        stream = _ResettableStream(np.zeros(_SHAPE))
        _moved(stream)
        stream.events.clear()
        reading = _wrist_source(stream).grab_surface_depth()
        assert reading is not None
        self.assertEqual(stream.events, ["pose", "moved", *["grab"] * (WRIST_WARMUP_GRABS + 1), "pose"])
        self.assertAlmostEqual(float(np.mean(reading.depth_mm > 0.0)), 0.2, places=6)

    def test_a_fixed_camera_is_not_told_it_moved(self) -> None:
        """The control: a fixed camera did not move, and its history is the scene's."""
        stream = _ResettableStream(np.full(_SHAPE, 600.0))
        RigDepthSource(stream).grab_surface_depth()
        self.assertEqual(stream.events, ["grab"])

    def test_the_stamp_is_the_frame_kept(self) -> None:
        """The owner stamps each frame before its grab; the reading carries the stamp of the one it keeps."""
        stream = _FilteredStream(np.full(_SHAPE, 600.0), stamps=True)
        reading = _wrist_source(stream).grab_surface_depth()
        assert reading is not None
        self.assertEqual(reading.timestamp, 1000.0 + stream.grabs)

    def test_an_unstamped_frame_is_stamped_after_the_warm_ups(self) -> None:
        stream = _FilteredStream(np.full(_SHAPE, 600.0))
        clock = SimpleNamespace(now=50.0)

        def grab() -> SimpleNamespace:
            clock.now += 0.033
            return _FilteredStream.grab(stream)

        stream.grab = grab  # type: ignore[method-assign]
        with mock.patch(f"{_MODULE}.time", SimpleNamespace(time=lambda: clock.now)):
            reading = _wrist_source(stream).grab_surface_depth()
        assert reading is not None
        self.assertAlmostEqual(reading.timestamp, 50.0 + 5 * 0.033, places=9)

    def test_a_frame_taken_while_the_tool_moved_is_still_none(self) -> None:
        """The pose read before the warm-ups against the one after the frame kept: the whole window is judged."""
        stream = _FilteredStream(np.full(_SHAPE, 600.0))
        poses = iter([_pose(400.0), _pose(405.0)])
        source = RigDepthSource(stream, tool_pose=lambda: next(poses), motion_tolerance=(1.0, 0.5))
        self.assertIsNone(source.grab_surface_depth())


class HowManyFramesAreThrownAwayTests(unittest.TestCase):

    def test_a_fixed_camera_takes_one_frame_unless_told_otherwise(self) -> None:
        """The control: a fixed camera did not move, and its world grabs once, as before."""
        stream = _FilteredStream(np.full(_SHAPE, 600.0))
        self.assertIsNotNone(RigDepthSource(stream).grab_surface_depth())
        self.assertEqual(stream.grabs, 1)
        stream = _FilteredStream(np.full(_SHAPE, 600.0))
        source = RigDepthSource(stream, warmup_grabs=2)
        self.assertEqual(source.warmup_grabs, 2)
        source.grab_surface_depth()
        self.assertEqual(stream.grabs, 3)

    def test_a_wrist_source_may_be_told_another_count(self) -> None:
        stream = _FilteredStream(np.full(_SHAPE, 600.0))
        _wrist_source(stream, warmup_grabs=0).grab_surface_depth()
        self.assertEqual(stream.grabs, 1)

    def test_a_count_that_is_not_a_whole_number_of_frames_is_refused(self) -> None:
        for bad in (-1, 1.5, True, "5"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                RigDepthSource(_FilteredStream(np.zeros(_SHAPE)), warmup_grabs=bad)  # type: ignore[arg-type]

    def test_a_warm_up_that_fails_is_no_frame(self) -> None:
        stream = _FilteredStream(np.full(_SHAPE, 600.0))
        calls = {"n": 0}

        def grab() -> SimpleNamespace:
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("Frame didn't arrive within 5000")
            return _FilteredStream.grab(stream)

        stream.grab = grab  # type: ignore[method-assign]
        self.assertIsNone(_wrist_source(stream).grab_surface_depth())


if __name__ == "__main__":
    unittest.main()
