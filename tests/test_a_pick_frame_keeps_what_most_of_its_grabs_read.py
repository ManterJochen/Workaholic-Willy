"""A pick frame's depth is the median of a burst of grabs, pixel by pixel, where more than half of them read the pixel.

The owner, 2026-10-13: up to 0.5 s more a look for what the cell can tell. A RealSense's temporal filter takes a value
that jumps past its delta as it comes, so a speck on the mat, or an edge flickering between a part and the floor behind
it, stood in the one frame a pick kept whenever the last grab read it. Seven grabs after the warm-ups: what fewer than
half of them read is out, a hole most of them read is filled, and on a wrist the tool holds still across the whole burst
or the whole burst is taken again.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.camera.setup.image_taking.frames import RGBDFrame
from src.geometry import Frame, Pose
from src.robot.core.shutter_motion import PICK_FRAME_BURST_GRABS
from src.robot.perception import RealSenseVisionPerceptionSource
from src.robot.perception.realsense_source import burst_depth

_K = np.array([[600.0, 0.0, 4.0], [0.0, 600.0, 4.0], [0.0, 0.0, 1.0]])


def _pose(x: float) -> Pose:
    return Pose(position_mm=np.array([x, 0.0, 300.0]), quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), frame=Frame.BASE)


class _Burst:
    """A camera over a mat 500 mm away: every grab a little noise, a speck at (1, 1) in the last of every seven, and at
    (2, 2) a hole in the last of every seven only; each grab's colour its number, its capture time 10 s on."""

    def __init__(self, events: "list[str] | None" = None) -> None:
        self.grabs = 0
        self.rig_id = "wrist"
        self._events = events

    def grab(self) -> RGBDFrame:
        self.grabs += 1
        if self._events is not None:
            self._events.append("grab")
        depth = np.full((8, 8), 500, dtype=np.uint16) + np.uint16(self.grabs % 3)
        if self.grabs % PICK_FRAME_BURST_GRABS == 0:
            depth[1, 1] = 120
            depth[2, 2] = 0
        return RGBDFrame(color=np.full((8, 8, 3), self.grabs, dtype=np.uint8), depth=depth,
                         captured_at_s=10.0 * self.grabs)

    def get_intrinsics(self) -> np.ndarray:
        return _K.copy()


def _source(camera: _Burst) -> RealSenseVisionPerceptionSource:
    return RealSenseVisionPerceptionSource(streamer=camera, backend=SimpleNamespace(perceive=lambda _b, _p: []),
                                           prompt="a box", warmup_grabs=0)


class TheMedianOfABurstTests(unittest.TestCase):
    def test_what_fewer_than_half_of_the_grabs_read_is_out_and_a_hole_most_of_them_read_is_filled(self) -> None:
        depths = [np.array([[500, 0, 700, 300]], dtype=np.uint16) for _ in range(7)]
        depths[3][0, 0] = 90                      # a speck in one grab of seven
        for n in range(4):                        # the second pixel read by three of seven
            depths[n][0, 1] = 0
        depths[6][0, 1] = 0
        for n in range(3):                        # the third a hole in three of seven
            depths[n][0, 2] = 0
        for n, value in enumerate((300, 301, 302, 310, 290, 305, 299)):
            depths[n][0, 3] = value

        out = burst_depth(depths)

        self.assertEqual(np.uint16, out.dtype)
        self.assertEqual([500, 0, 700, 301], out[0].tolist())

    def test_an_even_count_takes_the_nearer_of_the_middle_two(self) -> None:
        depths = [np.array([[value]], dtype=np.float64) for value in (np.nan, 0.0, 0.0, 500.0, 504.0, 502.0, 506.0)]

        self.assertEqual(502.0, float(burst_depth(depths)[0, 0]), "the four that read it: 500, 502, 504, 506")


class APickFrameIsABurstTests(unittest.TestCase):
    def test_the_frame_keeps_the_last_grabs_colour_the_bursts_depth_and_the_first_grabs_time(self) -> None:
        camera = _Burst()

        frame = _source(camera).acquire()

        self.assertEqual(PICK_FRAME_BURST_GRABS, camera.grabs)
        self.assertEqual(PICK_FRAME_BURST_GRABS, int(frame.rgb[0, 0, 0]), "the last grab's colour")
        self.assertEqual(10.0, frame.timestamp, "the first grab's capture time: the frame holds depth from then on")
        depth = np.asarray(frame.surface_depth_map)
        self.assertTrue(500.0 <= depth[1, 1] <= 502.0, f"the speck of the last grab stayed: {depth[1, 1]}")
        self.assertTrue(500.0 <= depth[2, 2] <= 502.0, f"the hole of the last grab stayed: {depth[2, 2]}")

    def test_a_camera_that_hands_the_same_depth_every_grab_keeps_its_last_frame(self) -> None:
        frames: list[RGBDFrame] = []

        def grab() -> RGBDFrame:
            frames.append(RGBDFrame(color=np.zeros((8, 8, 3), dtype=np.uint8),
                                    depth=np.full((8, 8), 500, dtype=np.uint16), captured_at_s=float(len(frames))))
            return frames[-1]

        camera = SimpleNamespace(grab=grab, get_intrinsics=_K.copy, rig_id="overhead")
        frame = _source(camera).acquire()  # type: ignore[arg-type]

        self.assertEqual(PICK_FRAME_BURST_GRABS, len(frames))
        self.assertEqual(frames[-1].captured_at_s, frame.timestamp)


class OnAWristTheToolHoldsStillAcrossTheBurstTests(unittest.TestCase):
    def test_the_pose_is_read_before_the_first_grab_and_after_the_last(self) -> None:
        events: list[str] = []
        source = _source(_Burst(events))

        def reader() -> Pose:
            events.append("pose")
            return _pose(400.0)

        source.stamp_tool_pose_with(reader, motion_tolerance=(1.0, 0.5), attempts=1)
        source.acquire()

        self.assertEqual(["pose", *["grab"] * PICK_FRAME_BURST_GRABS, "pose"], events)

    def test_a_burst_the_tool_moved_across_is_taken_again_whole(self) -> None:
        camera = _Burst()
        source = _source(camera)
        poses = iter([_pose(400.0), _pose(405.0), _pose(405.0), _pose(405.0)])
        source.stamp_tool_pose_with(lambda: next(poses), motion_tolerance=(1.0, 0.5), attempts=2)

        frame = source.acquire()

        self.assertEqual(2 * PICK_FRAME_BURST_GRABS, camera.grabs)
        self.assertEqual(_pose(405.0), frame.tool_pose)
        self.assertEqual(10.0 * (PICK_FRAME_BURST_GRABS + 1), frame.timestamp, "the second burst's first grab")


if __name__ == "__main__":
    unittest.main()
