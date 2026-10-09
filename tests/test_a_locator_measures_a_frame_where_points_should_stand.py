"""A locator reads one frame where given points of BASE should stand, with no detector (2026-10-08: speed first).

``Locator.measure(points)`` takes its frame as a locate takes one, through the pick frame's source: the warm-up grabs,
the tool pose at the shutter on the wrist, where the camera stood at that shutter. Only its backend grounds nothing, so
a measure costs a frame and never a detector's call. Per point it says the depth the camera should read there, the
depth the frame measured at its pixel and that pixel's CIE L*a*b* colour; a point outside the frame, or behind the
camera, reads nothing. ``Located.measure(points, image)`` reads the same off the frame a locate placed, so a point
placed from a frame lands on the pixel it came from. A task reads a kept bin's rim with it before it asks the detector
again (``place_target.recheck``).
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.camera.setup.image_taking.frames import RGBDFrame
from src.robot.core.errors import PerceptionFrameMoved
from src.robot.core.shutter_motion import PICK_FRAME_WARMUP_GRABS
from src.robot.perception.locator import Located, Locator, Measured
from tests.test_locator import _SHAPE, _TOOL_FRAME, _backend, _block_frame, _Handle, _Owner, _pose, _wrist


class _CountingBackend:
    """A backend that says how often it was asked."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def perceive(self, _image: Any, prompt: str) -> tuple[()]:
        self.asked.append(prompt)
        return ()


def _coloured_frame(bgr: tuple[int, int, int]) -> RGBDFrame:
    """The bench a metre down and the block 80 mm up, the block's pixels coloured ``bgr``."""
    frame = _block_frame()
    colour = np.zeros((*_SHAPE, 3), dtype=np.uint8)
    colour[8:32, 8:32] = bgr
    return RGBDFrame(color=colour, depth=frame.depth, captured_at_s=frame.captured_at_s)


def _top_points() -> np.ndarray:
    """Where the block's top stands, BASE mm, read off a locate of it."""
    located = Locator.from_parts(camera=_Owner(), backend=_backend("block")).locate("block")
    return located.objects[0].points_base_mm


class AMeasureTests(unittest.TestCase):
    def test_a_measure_takes_one_frame_through_the_source_and_asks_no_detector(self) -> None:
        backend = _CountingBackend()
        handle = _Handle()
        locator = Locator.from_parts(camera=_Owner(handle=handle), backend=backend)

        read = locator.measure(np.array([[0.0, 0.0, 80.0]]))

        self.assertIsInstance(read, Measured)
        self.assertEqual([], backend.asked, "a measure asked the detector")
        self.assertEqual(PICK_FRAME_WARMUP_GRABS + 1, handle.grabs, "not the warm-ups and one frame, as a locate takes")
        self.assertEqual("overhead", read.camera)
        self.assertEqual(50.0, read.captured_at_s)

    def test_a_point_placed_from_a_frame_reads_the_depth_it_was_placed_from(self) -> None:
        points = _top_points()

        read = Locator.from_parts(camera=_Owner(), backend=_backend()).measure(points)

        self.assertTrue(bool(np.all(read.in_frame)))
        np.testing.assert_allclose(920.0, read.expected_mm, atol=1e-3)
        np.testing.assert_allclose(read.expected_mm, read.measured_mm, atol=1e-3)

    def test_a_point_outside_the_frame_or_behind_the_camera_reads_nothing(self) -> None:
        points = np.array([[5000.0, 0.0, 0.0], [0.0, 0.0, 1500.0], [0.0, -32.0, 0.0]])

        read = Locator.from_parts(camera=_Owner(), backend=_backend()).measure(points)

        self.assertEqual([False, False, True], read.in_frame.tolist())
        self.assertTrue(np.isnan(read.measured_mm[:2]).all())
        self.assertTrue(np.isnan(read.lab[:2]).all())
        self.assertTrue(np.isnan(read.expected_mm[1]), "a point behind the camera has a depth it should read")
        self.assertAlmostEqual(1000.0, float(read.measured_mm[2]), delta=1e-3, msg="the bench beside the block")

    def test_what_stands_in_front_of_a_point_is_what_its_pixel_reads(self) -> None:
        read = Locator.from_parts(camera=_Owner(), backend=_backend()).measure(np.array([[0.0, 0.0, 0.0]]))

        self.assertAlmostEqual(1000.0, float(read.expected_mm[0]), delta=1e-6)
        self.assertAlmostEqual(920.0, float(read.measured_mm[0]), delta=1e-3, msg="the block's top hides the bench")

    def test_a_pixel_with_no_depth_measured_reads_nothing_there(self) -> None:
        frame = _block_frame()
        depth = frame.depth.copy()
        depth[20, 20] = 0
        handle = _Handle([RGBDFrame(color=frame.color, depth=depth, captured_at_s=50.0)])

        read = Locator.from_parts(camera=_Owner(handle=handle), backend=_backend()).measure(np.array([[0.0, 0.0, 80.0]]))

        self.assertEqual([True], read.in_frame.tolist())
        self.assertTrue(np.isnan(read.measured_mm[0]))
        self.assertAlmostEqual(920.0, float(read.expected_mm[0]), delta=1e-6)

    def test_the_colour_is_the_pixels_cie_lab(self) -> None:
        for bgr, lab in (((255, 255, 255), (100.0, 0.0, 0.0)), ((0, 255, 255), (97.14, -21.55, 94.48)),
                         ((255, 0, 0), (32.30, 79.19, -107.86))):
            with self.subTest(bgr=bgr):
                handle = _Handle([_coloured_frame(bgr)])
                read = Locator.from_parts(camera=_Owner(handle=handle), backend=_backend()).measure(_top_points())
                np.testing.assert_allclose(np.tile(lab, (read.points, 1)), read.lab, atol=0.05)

    def test_a_wrist_measure_is_placed_by_the_tool_pose_at_its_shutter(self) -> None:
        reads: list[int] = []

        def tool_pose() -> Any:
            reads.append(1)
            return _pose(0.0)

        locator = Locator.from_parts(camera=_Owner(calibration=_wrist(), rig_id="wrist"), backend=_backend(),
                                     tool_pose=tool_pose, tool_frame=_TOOL_FRAME)
        read = locator.measure(np.array([[0.0, -32.0, 0.0]]))

        self.assertGreaterEqual(len(reads), 2, "the frame was not stamped with the tool pose at its shutter")
        # The tool stands a metre up, turned to look down: the bench beside the block reads a metre away.
        self.assertAlmostEqual(1000.0, float(read.expected_mm[0]), delta=1e-6)
        self.assertAlmostEqual(1000.0, float(read.measured_mm[0]), delta=1e-3)

    def test_a_wrist_frame_the_tool_moved_through_raises_as_a_locate_does(self) -> None:
        poses = iter(_pose(5.0 * n) for n in range(100))
        locator = Locator.from_parts(camera=_Owner(calibration=_wrist(), rig_id="wrist"), backend=_backend(),
                                     tool_pose=lambda: next(poses), attempts=2, tool_frame=_TOOL_FRAME)

        with self.assertRaises(PerceptionFrameMoved):
            locator.measure(np.array([[0.0, 0.0, 0.0]]))


class ALocatedFrameTests(unittest.TestCase):
    def test_a_located_frame_reads_as_a_new_frame_of_the_same_scene_does(self) -> None:
        image = _coloured_frame((0, 200, 230)).color
        handle = _Handle([_coloured_frame((0, 200, 230))])
        located = Locator.from_parts(camera=_Owner(handle=handle), backend=_backend("block")).locate("block")
        points = located.objects[0].points_base_mm

        off_the_locate = located.measure(points, image)
        new = Locator.from_parts(camera=_Owner(handle=handle), backend=_backend()).measure(points)

        assert off_the_locate is not None
        for name in ("in_frame", "expected_mm", "measured_mm", "lab"):
            np.testing.assert_allclose(getattr(new, name), getattr(off_the_locate, name), err_msg=name)

    def test_a_located_frame_without_its_image_reads_no_colour(self) -> None:
        located = Locator.from_parts(camera=_Owner(), backend=_backend("block")).locate("block")

        read = located.measure(located.objects[0].points_base_mm)

        assert read is not None
        self.assertTrue(np.isnan(read.lab).all())
        self.assertFalse(np.isnan(read.measured_mm).any())

    def test_a_located_that_kept_no_frame_reads_nothing(self) -> None:
        bare = Located(camera="wrist", captured_at_s=1.0, mounting="eye_in_hand", tool_to_base_mm=None, objects=())

        self.assertIsNone(bare.measure(np.zeros((3, 3))))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
