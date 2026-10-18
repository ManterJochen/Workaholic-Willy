"""The depth pixels that mix a part's top with the floor behind it are taken out, and nothing a camera really saw.

A D400 reads the step at a part's edge as a ramp 15 to 25 pixels wide (the owner's cell, 2026-10-12): placed in BASE, a
skirt about every part, which the camera world took for an obstacle and grew by its margin back to the part's face. On
the owner's crate every grasp of a cube 30 mm from its neighbour was refused 0.1 to 0.6 mm short of the guard's
distance. ``mixed_pixels`` finds such a pixel by the flat stretches either side of it and by the colour camera, which
sees the floor there (see its module docstring). The scenes here are rendered: a floor, parts with a ramp outside their
silhouette as the D415 reads it, and the surfaces a ramp must never be taken for.

* the ramp outside a part is taken out; the part, the floor and every hole keep what they read;
* a part the colour of the floor keeps every pixel, and so does a lying cylinder's flank, a wall seen at a slant, a long
  rim, and a lower part's own top beside the part;
* ``Camera.kept`` takes them out of a frame of a rig whose switch is on and leaves every other frame as it was, and the
  camera world's source keeps the frame only after the pose after the grab was read.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

import numpy as np

from src.camera.orchestration.camera import Camera
from src.camera.setup.image_taking.frames import RGBDFrame
from src.camera.setup.image_taking.mixed_pixels import mixed_pixels, without_mixed_pixels
from src.config.schema.camera import RGBDDeviceRigConfig

H, W = 480, 640
FX = 600.0
FLOOR_MM = 700.0
#: BGR colours: a blue crate floor, a green part, a grey part.
FLOOR = (150, 80, 20)
GREEN = (60, 160, 60)
GREY = (128, 128, 128)
#: The ramp the D415 leaves outside a part, pixels.
RAMP_PX = 20


def _scene() -> "tuple[np.ndarray, np.ndarray]":
    depth = np.full((H, W), FLOOR_MM)
    colour = np.empty((H, W, 3), dtype=np.uint8)
    colour[:] = FLOOR
    return depth, colour


def _distance_to_box(rows: "tuple[int, int]", cols: "tuple[int, int]") -> np.ndarray:
    """Per pixel, its distance in pixels to the box ``rows`` x ``cols`` (0 inside)."""
    r = np.arange(H)[:, None]
    c = np.arange(W)[None, :]
    dr = np.maximum(np.maximum(rows[0] - r, r - (rows[1] - 1)), 0)
    dc = np.maximum(np.maximum(cols[0] - c, c - (cols[1] - 1)), 0)
    return np.hypot(dr, dc)


def _cube(depth: np.ndarray, colour: np.ndarray, rows: "tuple[int, int]", cols: "tuple[int, int]", top_mm: float,
          bgr: "tuple[int, int, int]", *, ramp: bool = True) -> np.ndarray:
    """A box part: its top flat at ``top_mm``, its colour over its silhouette, and outside it the ramp down to what was
    read there before. Returns the ramp's pixels."""
    distance = _distance_to_box(rows, cols)
    inside = distance == 0
    below = depth.copy()
    depth[inside] = top_mm
    colour[inside] = bgr
    ring = (distance > 0) & (distance <= RAMP_PX)
    if ramp:
        share = distance[ring] / (RAMP_PX + 1)
        depth[ring] = top_mm + share * (below[ring] - top_mm)
    return ring


class TheSkirtIsTakenOutTests(unittest.TestCase):
    def test_the_ramp_outside_a_part_goes_and_the_part_and_the_floor_stay(self) -> None:
        depth, colour = _scene()
        ring = _cube(depth, colour, (200, 240), (300, 340), 660.0, GREEN)
        taken = mixed_pixels(depth, colour, FX)
        # The pixels strictly within the ramp, a few clear of either end, go (the two ends lie within a few mm of the
        # top and the floor, which they belong to).
        distance = _distance_to_box((200, 240), (300, 340))
        middle = ring & (distance >= 4) & (distance <= RAMP_PX - 4)
        self.assertGreater(np.count_nonzero(taken & middle) / np.count_nonzero(middle), 0.8)
        self.assertFalse(taken[distance == 0].any(), "a pixel of the part's own top")
        self.assertFalse(taken[distance > RAMP_PX + 2].any(), "a pixel of the floor")

    def test_nothing_gains_a_depth_and_nothing_kept_changes(self) -> None:
        depth, colour = _scene()
        _cube(depth, colour, (200, 240), (300, 340), 660.0, GREEN)
        depth[100:130, 100:160] = 0.0
        out, count = without_mixed_pixels(depth.astype(np.uint16), colour, FX)
        self.assertGreater(count, 0)
        self.assertEqual(np.uint16, out.dtype)
        kept = out > 0
        self.assertTrue(np.array_equal(out[kept], depth.astype(np.uint16)[kept]))
        self.assertFalse(kept[100:130, 100:160].any())
        self.assertEqual(count, int(np.count_nonzero((depth > 0) & ~kept)))

    def test_a_part_the_colour_of_the_floor_keeps_every_pixel(self) -> None:
        depth, colour = _scene()
        _cube(depth, colour, (200, 240), (300, 340), 660.0, FLOOR)
        self.assertFalse(mixed_pixels(depth, colour, FX).any())

    def test_a_lying_cylinder_keeps_its_flank_and_loses_its_ramp(self) -> None:
        depth, colour = _scene()
        radius_px, axis_col = 30, 320
        cols = np.arange(W)
        across = np.abs(cols - axis_col)
        on = across < radius_px
        # Its surface as the camera sees it: the crest nearest, the flank falling away to the floor.
        flank = FLOOR_MM - 40.0 * np.sqrt(np.clip(1.0 - (across / radius_px) ** 2, 0.0, 1.0))
        rows = slice(150, 330)
        depth[rows, on] = flank[on]
        colour[rows, on] = GREY
        outside = (across >= radius_px) & (across < radius_px + RAMP_PX)
        ramp = FLOOR_MM - 40.0 * np.sqrt(np.clip(1.0 - (radius_px / (radius_px + 1.0)) ** 2, 0.0, 1.0)) * (
            1.0 - (across - radius_px) / RAMP_PX)
        depth[rows, outside] = np.minimum(FLOOR_MM, ramp[outside])
        taken = mixed_pixels(depth, colour, FX)
        self.assertFalse(taken[rows, on].any(), "a pixel of the cylinder's own surface")

    def test_a_wall_seen_at_a_slant_and_its_long_rim_keep_every_pixel(self) -> None:
        depth, colour = _scene()
        # A rim 400 pixels long, flat at 580 mm, a lighter blue; under it the wall's inner face, darker, sloping over
        # 100 pixels to the floor.
        depth[100:110, 100:500] = 580.0
        colour[100:110, 100:500] = (190, 120, 60)
        slope = np.linspace(580.0, FLOOR_MM, 100)
        depth[110:210, 100:500] = slope[:, None]
        colour[110:210, 100:500] = (120, 60, 15)
        self.assertFalse(mixed_pixels(depth, colour, FX).any())

    def test_a_lower_part_beside_keeps_its_own_top(self) -> None:
        depth, colour = _scene()
        lower = _distance_to_box((205, 235), (355, 395)) == 0
        _cube(depth, colour, (205, 235), (355, 395), 680.0, GREY)
        _cube(depth, colour, (200, 240), (300, 340), 660.0, GREEN)
        # The grey part's own top stands 15 pixels from the green one, partly in its ramp's reach: it reads 680 again.
        depth[lower] = 680.0
        colour[lower] = GREY
        taken = mixed_pixels(depth, colour, FX)
        self.assertFalse(taken[lower].any())

    def test_frames_that_do_not_match_lose_nothing(self) -> None:
        depth, colour = _scene()
        _cube(depth, colour, (200, 240), (300, 340), 660.0, GREEN)
        self.assertFalse(mixed_pixels(depth, colour[:, :-1], FX).any())
        self.assertFalse(mixed_pixels(np.zeros((H, W)), colour, FX).any())


def _rig(**post: Any) -> RGBDDeviceRigConfig:
    return RGBDDeviceRigConfig(
        source="rgbd", rig_id="wrist", enabled=True, fps=30, backend=0, serial_number="SN-415",
        color_resolution=(W, H), depth_resolution=(W, H), align_depth_to_color=True, rgbd_backend="realsense",
        realsense={"post_processing": {"spatial": True, "temporal": True, **post}},
        quality={"warmup_frames": 0}, calibration_paths={"base_dir": "calibration/wrist"})


class _Streamer:
    def __init__(self, frame: Any) -> None:
        self.frame = frame

    def open(self) -> None:
        pass

    def release(self) -> None:
        pass

    def grab(self) -> Any:
        return self.frame

    def get_intrinsics(self) -> np.ndarray:
        return np.array([[FX, 0.0, W / 2.0], [0.0, FX, H / 2.0], [0.0, 0.0, 1.0]])


def _frame() -> RGBDFrame:
    depth, colour = _scene()
    _cube(depth, colour, (200, 240), (300, 340), 660.0, GREEN)
    return RGBDFrame(color=colour, depth=depth.astype(np.uint16))


class TheCameraKeepsItsFramesWithoutThemTests(unittest.TestCase):
    def _kept(self, rig: RGBDDeviceRigConfig, frame: Any) -> Any:
        camera = Camera(rig, _Streamer(frame))
        camera.open()
        try:
            return camera.kept(camera.grab())
        finally:
            camera.release()

    def test_a_rig_that_takes_them_out_keeps_its_frame_without_them(self) -> None:
        frame = _frame()
        kept = self._kept(_rig(), frame)
        self.assertIsNot(kept, frame)
        taken = (frame.depth > 0) & (kept.depth == 0)
        self.assertTrue(taken.any())
        self.assertTrue(np.array_equal(kept.color, frame.color))

    def test_the_switch_off_keeps_the_frame_as_it_was(self) -> None:
        frame = _frame()
        kept = self._kept(_rig(mixed_pixels=False), frame)
        self.assertTrue(np.array_equal(kept.depth, frame.depth))

    def test_a_depth_not_aligned_to_colour_keeps_the_frame_as_it_was(self) -> None:
        frame = _frame()
        rig = _rig().model_copy(update={"align_depth_to_color": False})
        kept = self._kept(rig, frame)
        self.assertTrue(np.array_equal(kept.depth, frame.depth))

    def test_the_world_keeps_its_frame_after_the_pose_after_the_grab(self) -> None:
        from src.geometry import Pose
        from src.robot.safety.planning.depth_source import RigDepthSource

        order: list[str] = []
        frame = _frame()

        class _Handle:
            rig_id = "wrist"

            def grab(self) -> Any:
                order.append("grab")
                return frame

            def kept(self, grabbed: Any) -> Any:
                order.append("kept")
                return grabbed

            def get_intrinsics(self) -> np.ndarray:
                return _Streamer(frame).get_intrinsics()

        def pose() -> Pose:
            order.append("pose")
            return Pose.tool_down(0.0, -500.0, 600.0)

        source = RigDepthSource(_Handle(), tool_pose=pose, motion_tolerance=(5.0, 10.0), warmup_grabs=0)
        with mock.patch("time.time", return_value=1000.0):
            snapshot = source.grab_surface_depth()
        self.assertIsNotNone(snapshot)
        self.assertEqual(["pose", "grab", "pose", "kept"], order)


if __name__ == "__main__":
    unittest.main()
