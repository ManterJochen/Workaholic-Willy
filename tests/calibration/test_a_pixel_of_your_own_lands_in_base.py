"""A pixel of your own lands in BASE: one frame kept whole, a plane fit about the pixel, and a TCP pose over it.

The owner, 2026-10-14: an external program finds a point in the camera image itself (a pen stroke on paper), and willy
says where it lies in the robot's base frame and proposes a TCP pose over it; nothing moves. ``capture(camera, robot)``
takes the frame a pick takes, the arm at rest by its steady gate; ``pixel_to_base`` and ``pixels_to_base`` meet each
pixel's viewing ray with the plane the depth about it measured, never the one pixel's depth, or with a plane named; and
``tcp_above`` proposes the TCP perpendicular to the surface. What cannot be found raises ``PixelRefused`` with its
reason.

The scenes are the wrist-look tests' (``tests/_wrist_views.py``): a wrist D415 at half resolution over a bench at BASE z
0, ray cast, the depth exact unless a test rounds it to whole millimetres as a RealSense reads it.
"""

from __future__ import annotations

import json
import math
import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.calibration.rig_calibration import RigCalibration
from src.calibration.serialization import FlangeToTcp
from src.camera.setup.image_taking.frames import RGBDFrame
from src.config import load_tree
from src.geometry import Frame, Pose
from src.robot.core.errors import PerceptionFrameMoved
from src.robot.core.shutter_motion import PICK_FRAME_BURST_GRABS, PICK_FRAME_WARMUP_GRABS
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.perception.locator import LocatorRefused
from src.robot.perception.pixel_to_base import (
    CapturedFrame,
    PixelPoint,
    PixelRefused,
    capture,
    pixel_to_base,
    pixels_to_base,
    tcp_above,
)
from tests._wrist_views import CUBE, HEIGHT, K, WIDTH, camera_looking_at, camera_to_tool, mount, render, tool_for
from tests.robot.perception.test_locator import _TOOL_FRAME, _fixed, _Handle, _Owner, _pose

#: The camera over the bench in most tests: 450 mm from a point of the bench, 50 degrees over it.
_OVER_THE_BENCH = camera_looking_at((0.0, -700.0, 0.0), bearing_deg=30.0, elevation_deg=50.0)


def _frame(camera_to_base: np.ndarray, *, boxes: tuple = (), rounded: bool = False) -> CapturedFrame:
    """The wrist camera's frame of the bench (and ``boxes``) from ``camera_to_base``, as a capture keeps it."""
    depth, _ = render(camera_to_base, boxes)
    return CapturedFrame(camera="wrist", captured_at_s=1.0, mounting="eye_in_hand",
                         color=np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8),
                         depth_mm=np.rint(depth) if rounded else depth, intrinsics=K,
                         camera_to_base_mm=camera_to_base, tcp_pose=tool_for(camera_to_base),
                         camera_to_tool_mm=mount())


def _on_the_bench(camera_to_base: np.ndarray, u: float, v: float) -> np.ndarray:
    """Where the ray through pixel (u, v) meets the bench, BASE z 0: the answer, worked out apart."""
    ray = camera_to_base[:3, :3] @ np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    origin = camera_to_base[:3, 3]
    return origin - origin[2] / ray[2] * ray


def _pixel_of(camera_to_base: np.ndarray, point: Any) -> tuple[float, float]:
    """The pixel a BASE point stands at, a fraction where it falls between pixels."""
    seen = np.linalg.inv(camera_to_base) @ np.append(np.asarray(point, dtype=np.float64), 1.0)
    return (float(K[0, 0] * seen[0] / seen[2] + K[0, 2]), float(K[1, 1] * seen[1] / seen[2] + K[1, 2]))


def _grid() -> list[tuple[float, float]]:
    return [(float(u), float(v)) for u in np.linspace(8.0, WIDTH - 9.0, 9) for v in np.linspace(8.0, HEIGHT - 9.0, 7)]


class APixelOfARenderedPlaneTests(unittest.TestCase):

    def test_a_pixel_of_the_bench_comes_back_on_it_within_a_tenth_of_a_millimetre(self) -> None:
        for bearing, elevation in ((30.0, 50.0), (200.0, 70.0), (90.0, 35.0)):
            camera = camera_looking_at((0.0, -700.0, 0.0), bearing_deg=bearing, elevation_deg=elevation)
            frame = _frame(camera)
            for u, v in _grid():
                with self.subTest(bearing=bearing, elevation=elevation, pixel=(u, v)):
                    point = pixel_to_base(frame, u, v)
                    self.assertLess(float(np.linalg.norm(point.base_mm - _on_the_bench(camera, u, v))), 0.1)
                    np.testing.assert_allclose([0.0, 0.0, 1.0], point.normal, atol=1e-6)
                    self.assertEqual("depth", point.source)
                    self.assertEqual(1.0, point.valid_share)
                    assert point.spread_mm is not None
                    self.assertLess(point.spread_mm, 1e-6)

    def test_a_point_of_the_bench_goes_to_its_pixel_and_back(self) -> None:
        frame = _frame(_OVER_THE_BENCH)
        for point in ((0.0, -700.0, 0.0), (35.5, -742.25, 0.0), (-61.0, -655.0, 0.0), (80.0, -690.0, 0.0)):
            with self.subTest(point=point):
                u, v = _pixel_of(_OVER_THE_BENCH, point)
                found = pixel_to_base(frame, u, v)
                self.assertLess(float(np.linalg.norm(found.base_mm - np.asarray(point))), 0.1)
                np.testing.assert_allclose((u, v), _pixel_of(_OVER_THE_BENCH, found.base_mm), atol=1e-6)

    def test_whole_millimetres_of_depth_hold_it_to_half_a_millimetre(self) -> None:
        """A RealSense reads its depth in whole millimetres: a surface square to the camera is rounded as one, so the
        plane is as far off as the rounding, and a tilted one averages it down."""
        for elevation in (40.0, 60.0, 89.0):
            camera = camera_looking_at((0.0, -700.0, 0.0), bearing_deg=30.0, elevation_deg=elevation)
            frame = _frame(camera, rounded=True)
            worst = max(float(np.linalg.norm(pixel_to_base(frame, u, v).base_mm - _on_the_bench(camera, u, v)))
                        for u, v in _grid())
            with self.subTest(elevation=elevation):
                self.assertLess(worst, 0.5)

    def test_a_stroke_with_no_depth_of_its_own_stands_on_the_paper(self) -> None:
        frame = _frame(_OVER_THE_BENCH)
        depth = frame.depth_mm.copy()
        depth[178:181, :] = 0.0  # a thin stroke the camera measured nothing on
        stroke = CapturedFrame(camera=frame.camera, captured_at_s=frame.captured_at_s, mounting=frame.mounting,
                               color=frame.color, depth_mm=depth, intrinsics=frame.intrinsics,
                               camera_to_base_mm=frame.camera_to_base_mm, tcp_pose=frame.tcp_pose,
                               camera_to_tool_mm=frame.camera_to_tool_mm)
        for u in range(20, WIDTH - 20, 41):
            with self.subTest(u=u):
                point = pixel_to_base(stroke, float(u), 179.0)
                self.assertLess(float(np.linalg.norm(point.base_mm - _on_the_bench(_OVER_THE_BENCH, u, 179.0))), 0.1)
                assert point.valid_share is not None
                self.assertAlmostEqual(12.0 / 15.0, point.valid_share)

    def test_a_pixel_on_a_part_stands_on_the_face_it_shows_however_near_an_edge(self) -> None:
        """The window about a pixel 1 mm from the cube's edge holds two faces, or a face and the bench beyond it: the
        plane is the one the pixel's own depth lies on, not the bevel across the edge nor the bench."""
        camera = camera_looking_at((0.0, -700.0, 20.0), bearing_deg=0.0, elevation_deg=60.0)
        frame = _frame(camera, boxes=(CUBE,))
        for point in ((19.0, -700.0, 40.0), (-19.0, -700.0, 40.0), (0.0, -681.0, 40.0), (20.0, -700.0, 30.0),
                      (60.0, -700.0, 0.0)):
            with self.subTest(point=point):
                found = pixel_to_base(frame, *_pixel_of(camera, point))
                self.assertLess(float(np.linalg.norm(found.base_mm - np.asarray(point))), 0.1)

    def test_the_ray_meets_a_plane_handed_in_and_reads_no_depth(self) -> None:
        blind = _frame(_OVER_THE_BENCH)
        blind = CapturedFrame(camera=blind.camera, captured_at_s=blind.captured_at_s, mounting=blind.mounting,
                              color=blind.color, depth_mm=np.zeros_like(blind.depth_mm), intrinsics=blind.intrinsics,
                              camera_to_base_mm=blind.camera_to_base_mm, tcp_pose=blind.tcp_pose,
                              camera_to_tool_mm=blind.camera_to_tool_mm)
        for plane in (0.0, ((5.0, -3.0, 0.0), (0.0, 0.0, 1.0)), ((0.0, 0.0, 0.0), (0.0, 0.0, -2.0))):
            with self.subTest(plane=plane):
                point = pixel_to_base(blind, 300.5, 150.0, plane=plane)
                self.assertLess(float(np.linalg.norm(point.base_mm - _on_the_bench(_OVER_THE_BENCH, 300.5, 150.0))),
                                1e-6)
                np.testing.assert_allclose([0.0, 0.0, 1.0], point.normal, atol=1e-12, err_msg="toward the camera")
                self.assertEqual(("plane", None, None), (point.source, point.valid_share, point.spread_mm))
        raised = pixel_to_base(blind, 300.5, 150.0, plane=50.0)
        self.assertAlmostEqual(50.0, float(raised.base_mm[2]), places=9)

    def test_the_point_in_the_tool_frame_is_the_base_point_seen_from_the_tool(self) -> None:
        frame = _frame(_OVER_THE_BENCH)
        point = pixel_to_base(frame, 200.0, 100.0)
        assert point.tool_mm is not None and frame.tool_to_base_mm is not None
        np.testing.assert_allclose(point.base_mm, (frame.tool_to_base_mm @ np.append(point.tool_mm, 1.0))[:3],
                                   atol=1e-9)
        np.testing.assert_allclose(frame.camera_to_base_mm, frame.tool_to_base_mm @ frame.camera_to_tool_mm, atol=1e-9)
        np.testing.assert_allclose(frame.camera_to_base_mm, point.camera_to_base_mm)


class WhatCannotBeFoundSaysWhyTests(unittest.TestCase):

    def _refused(self, frame: CapturedFrame, u: float, v: float, **keywords: Any) -> PixelRefused:
        with self.assertRaises(PixelRefused) as caught:
            pixel_to_base(frame, u, v, **keywords)
        self.assertEqual((u, v), caught.exception.pixel)
        self.assertIsNone(caught.exception.index)
        return caught.exception

    def test_a_hole_in_the_depth_is_no_depth_and_says_to_name_a_plane(self) -> None:
        frame = _frame(_OVER_THE_BENCH)
        depth = frame.depth_mm.copy()
        depth[90:130, 180:240] = 0.0
        holed = CapturedFrame(camera="wrist", captured_at_s=1.0, mounting="eye_in_hand", color=frame.color,
                              depth_mm=depth, intrinsics=K, camera_to_base_mm=_OVER_THE_BENCH)
        refused = self._refused(holed, 210.0, 110.0)
        self.assertEqual("no_depth", refused.reason)
        self.assertIn("plane=", str(refused))
        self.assertIsNone(pixel_to_base(holed, 210.0, 110.0, plane=0.0).tool_mm,
                          "a frame with no TCP has no TOOL frame")

    def test_a_pixel_outside_the_image_is_outside(self) -> None:
        frame = _frame(_OVER_THE_BENCH)
        for u, v in ((-0.6, 10.0), (float(WIDTH) - 0.5, 10.0), (10.0, float(HEIGHT)), (10.0, -3.0)):
            with self.subTest(pixel=(u, v)):
                self.assertEqual("outside", self._refused(frame, u, v).reason)
        self.assertEqual("depth", pixel_to_base(frame, -0.4, 0.0).source, "pixel 0 reaches half a pixel out")

    def test_a_ray_along_the_plane_is_parallel(self) -> None:
        frame = _frame(_OVER_THE_BENCH)
        ray = _OVER_THE_BENCH[:3, :3] @ np.array([(100.0 - K[0, 2]) / K[0, 0], (80.0 - K[1, 2]) / K[1, 1], 1.0])
        along = (_OVER_THE_BENCH[:3, 3] + 200.0 * ray, np.cross(ray, (0.0, 0.0, 1.0)))
        refused = self._refused(frame, 100.0, 80.0, plane=along)
        self.assertEqual("parallel", refused.reason)

    def test_a_plane_behind_the_camera_is_behind(self) -> None:
        frame = _frame(_OVER_THE_BENCH)
        above_the_camera = float(_OVER_THE_BENCH[2, 3]) + 100.0
        self.assertEqual("behind", self._refused(frame, 320.0, 180.0, plane=above_the_camera).reason)

    def test_a_window_of_no_one_surface_is_no_surface(self) -> None:
        """A camera square to three steps: the middle pixel with no depth of its own, in a window a third each; and a
        pixel whose own depth stands on a strip narrower than a quarter of its window."""
        lens = np.array([[500.0, 0.0, 20.0], [0.0, 500.0, 20.0], [0.0, 0.0, 1.0]])
        depth = np.full((40, 40), 450.0)
        depth[:, 13:18], depth[:, 18:23], depth[:, 23:28] = 400.0, 450.0, 500.0
        depth[20, 20] = 0.0
        steps = CapturedFrame(camera="square", captured_at_s=1.0, mounting="eye_to_hand",
                              color=np.zeros((40, 40, 3), dtype=np.uint8), depth_mm=depth, intrinsics=lens,
                              camera_to_base_mm=np.eye(4))
        self.assertIn("no depth of its own", str(self._refused(steps, 20.0, 20.0)))
        strip = np.full((40, 40), 450.0)
        strip[:, 19:22] = 400.0
        narrow = CapturedFrame(camera="square", captured_at_s=1.0, mounting="eye_to_hand",
                               color=np.zeros((40, 40, 3), dtype=np.uint8), depth_mm=strip, intrinsics=lens,
                               camera_to_base_mm=np.eye(4))
        refused = self._refused(narrow, 20.0, 20.0)
        self.assertEqual("no_surface", refused.reason)
        self.assertIn("a quarter", str(refused))
        self.assertAlmostEqual(450.0, float(pixel_to_base(narrow, 10.0, 20.0).base_mm[2]), places=6)

    def test_what_is_no_pixel_no_window_and_no_plane_is_a_value_error(self) -> None:
        frame = _frame(_OVER_THE_BENCH)
        for keywords in ({"window_px": 14}, {"window_px": 1}, {"window_px": True}, {"plane": "mat"},
                         {"plane": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))}, {"plane": math.inf}):
            with self.subTest(keywords=keywords), self.assertRaises(ValueError) as caught:
                pixel_to_base(frame, 10.0, 10.0, **keywords)
            self.assertNotIsInstance(caught.exception, PixelRefused)
        with self.assertRaises(ValueError):
            pixel_to_base(frame, math.nan, 10.0)


class ManyPixelsAtOnceTests(unittest.TestCase):

    def test_a_stroke_is_every_pixel_in_its_order_as_one_call_finds_each(self) -> None:
        frame = _frame(_OVER_THE_BENCH)
        stroke = [(100.0, 50.0), (101.5, 52.0), (400.25, 300.0)]
        one_by_one = [pixel_to_base(frame, u, v) for u, v in stroke]
        for pixels in (stroke, np.array(stroke), np.array(stroke).reshape(-1, 1, 2)):
            with self.subTest(shape=np.shape(pixels)):
                found = pixels_to_base(frame, pixels)
                self.assertEqual(len(stroke), len(found))
                for each, alone in zip(found, one_by_one):
                    self.assertEqual(alone.pixel, each.pixel)
                    np.testing.assert_allclose(alone.base_mm, each.base_mm)
        self.assertEqual((), pixels_to_base(frame, []))

    def test_the_first_pixel_that_cannot_be_found_names_its_place_in_the_list(self) -> None:
        frame = _frame(_OVER_THE_BENCH)
        with self.assertRaises(PixelRefused) as caught:
            pixels_to_base(frame, [(10.0, 10.0), (20.0, 10.0), (float(WIDTH) + 5.0, 10.0), (-9.0, 0.0)])
        self.assertEqual((2, "outside", (float(WIDTH) + 5.0, 10.0)),
                         (caught.exception.index, caught.exception.reason, caught.exception.pixel))
        self.assertIn("pixel 2 of 4", str(caught.exception))
        with self.assertRaises(ValueError):
            pixels_to_base(frame, [(1.0, 2.0, 3.0)])

    def test_a_point_prints_as_it_renders_and_its_plain_data_dumps(self) -> None:
        point = pixel_to_base(_frame(_OVER_THE_BENCH), 320.0, 180.0)
        self.assertIsInstance(point, PixelPoint)
        self.assertEqual(point.render(), str(point))
        self.assertIn("mm in BASE", point.render())
        self.assertEqual("depth", json.loads(json.dumps(point.to_dict()))["source"])
        frame = _frame(_OVER_THE_BENCH)
        self.assertEqual(frame.render(), str(frame))
        self.assertEqual(WIDTH, json.loads(json.dumps(frame.to_dict()))["width"])


class _Arm(DummyRobotArm):
    """A desk arm whose steady gate answers what it is told, and which counts how often it was asked."""

    def __init__(self, tool: Pose, *, steady: bool = True) -> None:
        super().__init__(initial_pose=tool)
        self.steady = steady
        self.asked: list[float] = []

    def wait_until_steady(self, timeout_s: float = 5.0, poll_interval_s: float = 0.02) -> bool:
        self.asked.append(float(timeout_s))
        return self.steady


def _robot(arm: Any) -> SimpleNamespace:
    """A connected robot as a capture reads it: its arm, and the tool frame of its robot section."""
    section = SimpleNamespace(gripper=SimpleNamespace(tool_frame=_TOOL_FRAME))
    return SimpleNamespace(arm=arm, robot_config=section)


class _WristHandle(_Handle):
    """The wrist D415's handle: the frames it is given, through the renderer's lens."""

    def get_intrinsics(self) -> np.ndarray:
        return K.copy()


def _wrist_camera(tool: Pose) -> _Owner:
    """The wrist camera on its mount, calibrated against the cell's flange to TCP, seeing the bench from ``tool``."""
    depth, _ = render(tool.to_matrix() @ mount(), ())
    colour = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    colour[..., 0], colour[..., 2] = 30, 200  # blue 30, red 200: BGR, as the camera hands it
    handle = _WristHandle([RGBDFrame(color=colour, depth=depth, captured_at_s=12.5)])
    calibration = RigCalibration(rig_id="wrist", mounting_mode="eye_in_hand", artifact_path="eih.json",
                                 transform=camera_to_tool(), shutter_motion_tolerance_mm=1.0,
                                 shutter_motion_tolerance_deg=0.5,
                                 flange_to_tcp=FlangeToTcp.from_matrix("willy", np.eye(4)))
    return _Owner(calibration=calibration, handle=handle, rig_id="wrist")


class ACaptureTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tool = tool_for(_OVER_THE_BENCH)

    def test_a_wrist_frame_is_kept_whole_with_where_it_stood_and_its_pixels_land_on_the_bench(self) -> None:
        camera, arm = _wrist_camera(self.tool), _Arm(self.tool)
        arm.connect()

        frame = capture(camera, _robot(arm))

        self.assertEqual(("wrist", "eye_in_hand", 12.5), (frame.camera, frame.mounting, frame.captured_at_s))
        self.assertEqual([0.0], arm.asked, "the steady gate, asked once to wait for nothing")
        self.assertEqual(PICK_FRAME_WARMUP_GRABS + PICK_FRAME_BURST_GRABS, camera.handle().grabs,
                         "not the warm-ups and one burst, as a pick frame takes")
        self.assertEqual((30, 200), (int(frame.color[0, 0, 0]), int(frame.color[0, 0, 2])), "not BGR")
        np.testing.assert_allclose(K, frame.intrinsics)
        assert frame.tcp_pose is not None and frame.camera_to_tool_mm is not None
        np.testing.assert_allclose(self.tool.to_matrix(), frame.tcp_pose.to_matrix(), atol=1e-12)
        np.testing.assert_allclose(mount(), frame.camera_to_tool_mm, atol=1e-12)
        np.testing.assert_allclose(_OVER_THE_BENCH, frame.camera_to_base_mm, atol=1e-9)
        for u, v in ((320.0, 180.0), (37.0, 300.5), (600.0, 20.0)):
            with self.subTest(pixel=(u, v)):
                point = pixel_to_base(frame, u, v)
                self.assertLess(float(np.linalg.norm(point.base_mm - _on_the_bench(_OVER_THE_BENCH, u, v))), 0.1)

    def test_an_arm_that_is_not_at_rest_is_refused_before_any_grab(self) -> None:
        camera, arm = _wrist_camera(self.tool), _Arm(self.tool, steady=False)
        arm.connect()
        with self.assertRaises(LocatorRefused) as caught:
            capture(camera, _robot(arm))
        self.assertIn("not at rest", str(caught.exception))
        self.assertEqual(0, camera.handle().grabs)

    def test_a_wrist_camera_needs_the_connected_robot_that_carries_it(self) -> None:
        arm = _Arm(self.tool)
        for robot, said in ((None, "capture(camera, robot)"), (_robot(arm), "robot.connected()"),
                            (SimpleNamespace(arm=arm, robot_config=None), "Robot.from_tree")):
            camera = _wrist_camera(self.tool)
            with self.subTest(said=said), self.assertRaises(LocatorRefused) as caught:
                capture(camera, robot)
            self.assertIn(said, str(caught.exception))
            self.assertEqual(0, camera.handle().grabs)

    def test_a_tool_that_moves_across_every_grab_raises_as_a_pick_frame_does(self) -> None:
        poses = iter(_pose(5.0 * n) for n in range(100))

        class _Moving(_Arm):
            def get_tcp_pose(self) -> Pose:
                return next(poses)

        arm = _Moving(self.tool)
        arm.connect()
        with self.assertRaises(PerceptionFrameMoved):
            capture(_wrist_camera(self.tool), _robot(arm))

    def test_a_fixed_camera_is_placed_by_its_own_calibration_and_asks_no_arm(self) -> None:
        frame = capture(_Owner(calibration=_fixed()))

        self.assertEqual(("overhead", "eye_to_hand"), (frame.camera, frame.mounting))
        self.assertIsNone(frame.tcp_pose)
        self.assertIsNone(frame.camera_to_tool_mm)
        self.assertIsNone(frame.tool_to_base_mm)
        np.testing.assert_allclose(_fixed().transform.to_matrix(), frame.camera_to_base_mm)
        self.assertAlmostEqual(80.0, float(pixel_to_base(frame, 20.0, 20.0).base_mm[2]), places=6,
                               msg="the block's top, a metre under the camera")

    def test_a_rig_that_does_not_align_its_depth_to_its_colour_is_refused(self) -> None:
        camera = _wrist_camera(self.tool)
        camera.rig = SimpleNamespace(align_depth_to_color=False)  # type: ignore[attr-defined]
        arm = _Arm(self.tool)
        arm.connect()
        with self.assertRaises(LocatorRefused) as caught:
            capture(camera, _robot(arm))
        self.assertIn("align_depth_to_color", str(caught.exception))

    def test_a_frame_of_your_own_is_held_to_its_shapes(self) -> None:
        good: dict[str, Any] = {"camera": "own", "captured_at_s": 0.0, "mounting": "eye_to_hand",
                                "color": np.zeros((4, 5, 3), dtype=np.uint8), "depth_mm": np.zeros((4, 5)),
                                "intrinsics": K, "camera_to_base_mm": np.eye(4)}
        self.assertEqual((4, 5), CapturedFrame(**good).depth_mm.shape)
        for name, value in (("depth_mm", np.zeros((4, 6))), ("intrinsics", np.eye(4)), ("mounting", "on the wall"),
                            ("camera_to_base_mm", np.full((4, 4), np.nan)),
                            ("tcp_pose", Pose.tool_down(0.0, 0.0, 0.0, frame=Frame.CAMERA))):
            with self.subTest(name=name), self.assertRaises(ValueError):
                CapturedFrame(**{**good, name: value})


class ATcpPoseOverAPointTests(unittest.TestCase):

    def test_over_a_level_surface_it_is_the_cells_tool_down_pose(self) -> None:
        above = tcp_above((-130.0, -700.0, 50.0), (0.0, 0.0, 3.0), 20.0)
        expected = Pose.tool_down(-130.0, -700.0, 70.0, closing_axis="-y")
        np.testing.assert_allclose(expected.to_matrix(), above.to_matrix(), atol=1e-12)
        self.assertIs(Frame.BASE, above.frame)
        along_x = tcp_above((0.0, -500.0, 0.0), (0.0, 0.0, 1.0), 0.0, closing_axis="x")
        np.testing.assert_allclose(Pose.tool_down(0.0, -500.0, 0.0).to_matrix(), along_x.to_matrix(), atol=1e-12)

    def test_over_a_tilted_surface_the_tool_stands_square_to_it(self) -> None:
        tilt = math.radians(25.0)
        normal = np.array([math.sin(tilt), 0.0, math.cos(tilt)])
        matrix = tcp_above((10.0, -600.0, 30.0), normal, 15.0).to_matrix()
        np.testing.assert_allclose(np.array([10.0, -600.0, 30.0]) + 15.0 * normal, matrix[:3, 3], atol=1e-12)
        np.testing.assert_allclose(-normal, matrix[:3, 2], atol=1e-12, err_msg="the tool's +Z onto the surface")
        np.testing.assert_allclose([0.0, -1.0, 0.0], matrix[:3, 0], atol=1e-12, err_msg="its +X along base -Y")

    def test_the_fingertips_stand_on_the_offset_where_asked(self) -> None:
        from src.robot.safety.planning.hand import hand_past_the_tcp_mm, planner_hand

        robot = load_tree(None).with_values({"robot.gripper.model": "robotiq_2f85"}).robot
        hand = planner_hand(robot)
        reach = float(hand.jaw.finger_ahead_mm) + float(hand_past_the_tcp_mm(robot))  # type: ignore[union-attr]
        self.assertGreater(reach, 0.0)
        tips = tcp_above((0.0, -600.0, 10.0), (0.0, 0.0, 1.0), 5.0, fingertips_of=robot)
        self.assertAlmostEqual(10.0 + 5.0 + reach, float(tips.position_mm[2]), places=9)
        with self.assertRaisesRegex(ValueError, "robot.gripper.model"):
            tcp_above((0.0, -600.0, 10.0), (0.0, 0.0, 1.0), 5.0, fingertips_of=load_tree(None).robot)

    def test_what_proposes_no_pose_is_a_value_error(self) -> None:
        for point, normal, offset, axis in (((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), -1.0, "-y"),
                                            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 10.0, "-y"),
                                            ((0.0, 0.0), (0.0, 0.0, 1.0), 10.0, "-y"),
                                            ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 10.0, "-y"),
                                            ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 10.0, "sideways")):
            with self.subTest(point=point, normal=normal, offset=offset, axis=axis), self.assertRaises(ValueError):
                tcp_above(point, normal, offset, closing_axis=axis)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
