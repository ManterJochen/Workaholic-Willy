"""The task's own places are painted out of the image its detector reads, where the cell says so (the owner, 2026-10-08).

Every look that saw the task's bin had the detector box every part already placed in it, 2 to 4 s a box on the owner's
cell, each then left out as standing in the bin. Where the cell turns ``robot.grasping.hide_own_places`` on, a pick hands
its camera source what paints the regions its task keeps out (the bin's footprint, the circles about its drops) over the
copy of the frame the detector reads (``realsense_source.kept_out_pixels``, ``painted_copy``). What these pin:

* a pixel is painted where its own depth places it in such a region, and only there: a part outside every region keeps
  every pixel, and so does a part that reaches into one from outside; flying pixels at a part's edge never make it reach
  out; a hole in the depth is painted only where painted pixels close round it;
* the detector reads the painted copy; the segmenter, the colour check, the depth and the frame handed on are the real
  ones, and nothing the source was handed is written to;
* a backend that hands its detector no copy, a ``hide`` that raises or answers no mask of the frame's size, and no
  ``hide`` at all leave the detector the real frame, the backend asked as it always was;
* the pick loop hands ``hide`` only with the switch on, a region that keeps out every label, a source that says it can
  paint (``hides_places``), a CAMERA to BASE and a declared support; otherwise its acquire is the one it always was,
  said once where something is missing; the switch is off by default and the service sets it where the cell is built.

The scenes are ray cast through the wrist D415 of ``tests/_wrist_views.py``.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.camera.setup.image_taking.frames import RGBDFrame
from src.config.schema.robot import RobotConfig
from src.config.schema.robot.grasping_schema import GraspingSupportConfig, RobotGraspingConfig
from src.geometry import Frame, Transform
from src.models.perception_backend import PerceivedObject
from src.robot.core import JointPositions
from src.robot.execution.autonomous_grasp import AutonomousGraspService
from src.robot.grasping.collision.gripper_model import ParallelJawGripperModel
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator, PickOutcome
from src.robot.grasping.motion.execution_policy import PolicyOutcome, PolicyReport
from src.robot.grasping.motion.frame_resolver import EyeInHandFrameResolver, StaticCameraToBaseResolver
from src.robot.grasping.recovery.exclusion_zones import ExclusionRegion, ExclusionZones
from src.robot.grasping.types.perception import PerceptionFrame
from src.robot.perception import RealSenseVisionPerceptionSource
from src.robot.perception.realsense_source import LEAST_OUTSIDE_PX, _inside_region, kept_out_pixels, painted_copy
from tests._helpers import _FakePerception, _perception_frame, _ScriptedCalculator
from tests._wrist_views import (
    CUBE,
    HEIGHT,
    WIDTH,
    Box,
    K,
    LookCalculator,
    LookingArm,
    WristCamera,
    camera_looking_at,
    camera_to_tool,
    mount,
    render,
    tool_for,
)

#: A camera 600 mm from the drop, 60 degrees up, round from the drop's +x side.
CAMERA = camera_looking_at((0.0, -700.0, 0.0), bearing_deg=30.0, elevation_deg=60.0, range_mm=600.0)
#: A part laid at the drop, one reaching 15 mm into the drop's circle from outside it, and one well away.
PLACED = Box((-20.0, -720.0, 0.0), (20.0, -680.0, 40.0), "placed")
REACHING = Box((65.0, -720.0, 0.0), (105.0, -680.0, 40.0), "reaching")
AWAY = Box((-20.0, -560.0, 0.0), (20.0, -520.0, 40.0), "away")
#: The circle a task keeps out about its drop, 80 mm round it.
DROP = ExclusionRegion.circle((0.0, -700.0), 80.0, reason="the drop at pose 'ablage'")
#: The Hand-E's contact patch (config/grippers/robotiq_hande.yaml), the owner's hand.
HAND_E = ParallelJawGripperModel(finger_width_mm=29.24, pad_length_mm=20.91, pad_ahead_mm=10.45)


def _scene(boxes: tuple[Box, ...] = (PLACED, REACHING, AWAY)) -> tuple[np.ndarray, np.ndarray]:
    return render(CAMERA, boxes)


def _bench_inside(depth: np.ndarray, hit: np.ndarray, region: Any) -> np.ndarray:
    """The bench's pixels whose own depth places them in ``region``."""
    rows, cols = np.nonzero(hit == -1)
    z = depth[rows, cols]
    camera = np.column_stack([(cols - K[0, 2]) * z / K[0, 0], (rows - K[1, 2]) * z / K[1, 1], z])
    base = camera @ CAMERA[:3, :3].T + CAMERA[:3, 3]
    inside = np.zeros(depth.shape, dtype=bool)
    inside[rows, cols] = _inside_region(region, base[:, 0], base[:, 1])
    return inside


class WhatIsPaintedTests(unittest.TestCase):
    def test_a_part_laid_in_the_region_is_painted_and_a_part_away_from_it_keeps_every_pixel(self) -> None:
        depth, hit = _scene()

        painted = kept_out_pixels(depth, K, CAMERA, [DROP], support_mm=0.0)

        self.assertGreater(float(painted[hit == 0].mean()), 0.9, "the part laid at the drop is not hidden")
        self.assertFalse(painted[hit == 2].any(), "a part away from the region lost pixels")
        bench = _bench_inside(depth, hit, DROP)
        self.assertGreater(float(painted[bench].mean()), 0.9, "the drop's circle is not painted")
        self.assertFalse(painted[(hit == -1) & ~bench].any(), "the bench outside the circle was painted")

    def test_a_part_that_reaches_into_the_region_from_outside_keeps_every_pixel(self) -> None:
        depth, hit = _scene()
        reaching = np.argwhere(hit == 1)
        self.assertGreater(len(reaching), LEAST_OUTSIDE_PX, "the scene's reaching part is too small to say")

        painted = kept_out_pixels(depth, K, CAMERA, [DROP], support_mm=0.0)

        self.assertFalse(painted[hit == 1].any(), "the part reaching into the circle lost pixels to the paint")

    def test_flying_pixels_at_a_parts_edge_never_make_it_reach_out_of_its_region(self) -> None:
        import cv2

        depth, hit = _scene((PLACED,))
        tight = ExclusionRegion.rectangle((0.0, -700.0), (60.0, 60.0), reason="a bin's footprint and 10 mm")
        edge = cv2.dilate((hit == 0).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & (hit == -1)
        flying = depth.copy()
        flying[edge] *= 0.8  # placed somewhere between the part and the camera, as a flying pixel is

        painted = kept_out_pixels(flying, K, CAMERA, [tight], support_mm=0.0)

        self.assertGreater(float(painted[hit == 0].mean()), 0.9, "a part's flying pixels kept it from being hidden")

    def test_a_hole_closed_round_by_painted_pixels_is_painted_and_one_in_a_part_away_is_not(self) -> None:
        depth, hit = _scene()
        holed = depth.copy()
        placed_rows, placed_cols = np.nonzero(hit == 0)
        away_rows, away_cols = np.nonzero(hit == 2)
        mid_placed = (int(np.median(placed_rows)), int(np.median(placed_cols)))
        mid_away = (int(np.median(away_rows)), int(np.median(away_cols)))
        for row, col in (mid_placed, mid_away):
            holed[row - 1:row + 2, col - 1:col + 2] = 0.0

        painted = kept_out_pixels(holed, K, CAMERA, [DROP], support_mm=0.0)

        self.assertTrue(painted[mid_placed], "a hole inside the hidden part was left showing")
        self.assertFalse(painted[mid_away], "a hole inside a part away from the region was painted")

    def test_no_region_and_no_depth_paint_nothing(self) -> None:
        depth, _ = _scene()
        self.assertFalse(kept_out_pixels(depth, K, CAMERA, [], support_mm=0.0).any())
        self.assertFalse(kept_out_pixels(np.zeros_like(depth), K, CAMERA, [DROP], support_mm=0.0).any())
        elsewhere = ExclusionRegion.circle((900.0, 900.0), 50.0)
        self.assertFalse(kept_out_pixels(depth, K, CAMERA, [elsewhere], support_mm=0.0).any())

    def test_a_region_holds_a_point_as_the_task_keeps_it_out(self) -> None:
        rng = np.random.default_rng(9)
        points = rng.uniform(-150.0, 150.0, (400, 2)) + np.array([10.0, -700.0])
        for region in (DROP, ExclusionRegion.rectangle((10.0, -700.0), (120.0, 60.0), yaw_rad=0.6)):
            with self.subTest(region=region.render()):
                held = _inside_region(region, points[:, 0], points[:, 1])
                self.assertEqual([region.contains((x, y)) for x, y in points], held.tolist())


class ThePaintedCopyTests(unittest.TestCase):
    def test_the_copy_is_painted_in_the_colour_round_the_region_and_the_image_is_left_as_it_was(self) -> None:
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        image[:, :] = (20, 22, 24)               # the mat
        image[10:30, 20:40] = (40, 200, 240)     # a bright bin on it
        original = image.copy()
        mask = np.zeros((40, 60), dtype=bool)
        mask[10:30, 20:40] = True

        painted = painted_copy(image, mask)

        np.testing.assert_array_equal(original, image, "the image handed in was written to")
        np.testing.assert_array_equal(np.array([20, 22, 24]), painted[15, 25], "not the mat's colour round it")
        self.assertEqual(1, len({tuple(pixel) for pixel in painted[mask]}), "the region is not one colour")
        np.testing.assert_array_equal(image[~mask], painted[~mask])

    def test_nothing_to_paint_is_a_copy(self) -> None:
        image = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
        painted = painted_copy(image, np.zeros((4, 5), dtype=bool))
        np.testing.assert_array_equal(image, painted)
        self.assertIsNot(image, painted)


# ---------------------------------------------------------------------------------------------------------------------
# The camera source: the detector reads the copy, everything else the real frame
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class _Det:
    box: tuple[float, float, float, float]
    label: str = "grey cube"
    score: float = 0.9


@dataclass(frozen=True)
class _Seg:
    mask: np.ndarray
    label: str = "grey cube"


class _Streamer:
    def __init__(self, color: np.ndarray, depth: np.ndarray) -> None:
        self.frame = RGBDFrame(color=color, depth=depth)

    def grab(self) -> RGBDFrame:
        return self.frame

    def get_intrinsics(self) -> np.ndarray:
        return np.array([[600.0, 0.0, 32.0], [0.0, 600.0, 32.0], [0.0, 0.0, 1.0]])


class _Backend:
    """A backend that writes down what each perceive was handed: the image the segmenter reads, and the copy the
    detector reads where one was handed (``detects_on_a_copy``)."""

    detects_on_a_copy = True

    def __init__(self) -> None:
        self.seen: list[tuple[np.ndarray, np.ndarray | None]] = []

    def perceive(self, image_bgr: Any, prompt: str, *, detect_on: Any = None) -> tuple[Any, ...]:
        self.seen.append((np.array(image_bgr, copy=True), None if detect_on is None else np.array(detect_on, copy=True)))
        mask = np.zeros(np.asarray(image_bgr).shape[:2], dtype=np.uint8)
        mask[40:50, 40:50] = 1
        return (PerceivedObject(detection=_Det((40.0, 40.0, 50.0, 50.0)), segmentation=_Seg(mask=mask)),)  # type: ignore[arg-type]


class _PlainBackend:
    """A backend as every backend was: one image, read by its detector and its segmenter alike."""

    def __init__(self) -> None:
        self.asked: list[tuple[Any, ...]] = []

    def perceive(self, *handed: Any) -> tuple[Any, ...]:
        self.asked.append(handed)
        return ()


def _frame_parts() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    color = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    depth = np.full((64, 64), 500, dtype=np.uint16)
    mask = np.zeros((64, 64), dtype=bool)
    mask[5:25, 5:25] = True
    return color, depth, mask


def _source(backend: Any, color: np.ndarray, depth: np.ndarray) -> RealSenseVisionPerceptionSource:
    return RealSenseVisionPerceptionSource(streamer=_Streamer(color, depth), backend=backend,
                                           prompt="each separate grey cube", object_labels=("grey cube",),
                                           warmup_grabs=0, colour_check="off")


class TheSourceTests(unittest.TestCase):
    def test_the_detector_reads_a_painted_copy_and_everything_else_the_real_frame(self) -> None:
        color, depth, mask = _frame_parts()
        backend = _Backend()
        source = _source(backend, color, depth)
        handed: list[tuple[np.ndarray, np.ndarray, Any]] = []

        def hide(depth_mm: np.ndarray, intrinsics: np.ndarray, tool_pose: Any) -> np.ndarray:
            handed.append((depth_mm, intrinsics, tool_pose))
            return mask

        frame = source.acquire(hide=hide)

        [(segmented_on, detected_on)] = backend.seen
        np.testing.assert_array_equal(color, segmented_on, "the segmenter did not read the real frame")
        assert detected_on is not None
        np.testing.assert_array_equal(color[~mask], detected_on[~mask], "the copy changed outside the region")
        self.assertEqual(1, len({tuple(pixel) for pixel in detected_on[mask]}), "the region is not painted over")
        self.assertFalse(np.array_equal(color[mask], detected_on[mask]))
        np.testing.assert_array_equal(color[..., ::-1], frame.rgb, "the frame handed on is not the real image")
        np.testing.assert_array_equal(depth.astype(np.float64), frame.depth_map)
        np.testing.assert_array_equal(depth.astype(np.float64), frame.surface_depth_map)
        np.testing.assert_array_equal(depth.astype(np.float64), handed[0][0], "hide was handed another depth")
        self.assertEqual(int(mask.sum()), source.last_hidden_px)
        self.assertEqual(1, len(frame.segmentations))

    def test_a_hide_that_writes_into_the_depth_is_refused_and_the_depth_kept(self) -> None:
        color, depth, mask = _frame_parts()
        backend = _Backend()
        source = _source(backend, color, depth)

        def hide(depth_mm: np.ndarray, intrinsics: np.ndarray, tool_pose: Any) -> np.ndarray:
            depth_mm[:] = 0.0
            return mask

        with self.assertLogs("src.robot.perception.realsense_source", level="ERROR"):
            frame = source.acquire(hide=hide)

        np.testing.assert_array_equal(depth.astype(np.float64), frame.depth_map)
        np.testing.assert_array_equal(depth.astype(np.float64), frame.surface_depth_map)
        self.assertIsNone(backend.seen[0][1], "the detector read a copy of a frame whose regions could not be placed")
        self.assertEqual(0, source.last_hidden_px)

    def test_a_hide_that_raises_or_answers_no_mask_of_the_frames_size_paints_nothing(self) -> None:
        color, depth, _ = _frame_parts()

        def raising(*_: Any) -> np.ndarray:
            raise RuntimeError("the frame could not be placed")

        for hide in (raising, lambda *_: np.ones((10, 10), dtype=bool), lambda *_: None,
                     lambda *_: np.zeros((64, 64), dtype=bool)):
            with self.subTest(hide=hide):
                backend = _Backend()
                source = _source(backend, color, depth)
                source.acquire(hide=hide)
                self.assertIsNone(backend.seen[0][1])
                self.assertEqual(0, source.last_hidden_px)

    def test_a_backend_that_hands_its_detector_no_copy_is_asked_as_it_always_was_and_says_so_once(self) -> None:
        color, depth, mask = _frame_parts()
        backend = _PlainBackend()
        source = _source(backend, color, depth)

        with self.assertLogs("src.robot.perception.realsense_source", level="WARNING") as said:
            source.acquire(hide=lambda *_: mask)
            source.acquire(hide=lambda *_: mask)

        self.assertEqual([2, 2], [len(handed) for handed in backend.asked], "the backend was handed more than before")
        np.testing.assert_array_equal(color, backend.asked[0][0])
        self.assertEqual(1, sum("hands its detector no copy" in line for line in said.output), said.output)

    def test_without_a_hide_the_backend_is_asked_as_it_always_was(self) -> None:
        color, depth, _ = _frame_parts()
        backend = _PlainBackend()

        _source(backend, color, depth).acquire()

        self.assertEqual(2, len(backend.asked[0]))


# ---------------------------------------------------------------------------------------------------------------------
# The pick loop: what it hands its camera, and when
# ---------------------------------------------------------------------------------------------------------------------

#: The cube from its +x side, 45 degrees up; a part laid at a drop 150 mm along -y of it, in the drop's circle.
LOOK_PLUS_X = JointPositions.deg(0.0, -90.0, -110.0, -60.0, 90.0, 0.0)
LAID = Box((-20.0, -870.0, 0.0), (20.0, -830.0, 40.0), "laid")
LAID_DROP = ExclusionRegion.circle((0.0, -850.0), 60.0, reason="the drop at pose 'ablage'")


@dataclass(eq=False)
class _HidingCamera(WristCamera):
    """The wrist D415, as a source that paints the task's places for its detector: it calls the ``hide`` it is handed
    on the frame it renders and keeps the mask, or writes down that it was handed none."""

    hides_places = True
    handed: list[Any] = field(default_factory=list)
    hidden: list[np.ndarray] = field(default_factory=list)

    def acquire(self, hide: Any = None) -> PerceptionFrame:  # type: ignore[override]
        frame = super().acquire()
        self.handed.append(hide)
        if hide is not None:
            self.hidden.append(np.asarray(hide(frame.depth_map, frame.intrinsics, frame.tool_pose), dtype=bool))
        return frame


class _Policy:
    def execute(self, grasp: Any) -> PolicyReport:
        return PolicyReport(outcome=PolicyOutcome.EXECUTED)


def _loop(*, camera_type: type = _HidingCamera, region: Any = LAID_DROP, fixed: bool = False,
          **wiring: Any) -> tuple[BinPickingOrchestrator, Any]:
    arm = LookingArm({LOOK_PLUS_X: tool_for(camera_looking_at(bearing_deg=0.0))})
    camera = camera_type(arm, boxes=(CUBE, LAID))
    zones = ExclusionZones()
    if region is not None:
        zones.keep_out_region(region)
    resolver: Any = (StaticCameraToBaseResolver(transform=Transform.from_matrix(
        camera_looking_at(bearing_deg=0.0), from_frame=Frame.CAMERA, to_frame=Frame.BASE)) if fixed
        else EyeInHandFrameResolver(t_cam_to_tool=camera_to_tool()))
    wiring.setdefault("support_config", GraspingSupportConfig())
    wiring.setdefault("hide_own_places", True)
    orchestrator = BinPickingOrchestrator(
        arm=arm, calculator=LookCalculator(camera), perception=camera,  # type: ignore[arg-type]
        frame_resolver=resolver, policy=_Policy(),  # type: ignore[arg-type]
        max_attempts=1, primary_camera_id="wrist", gripper_model=HAND_E, looks=(LOOK_PLUS_X,),
        exclusion_zones=zones, **wiring,
    )
    return orchestrator, camera


def _hits(camera: Any) -> np.ndarray:
    """Which box each pixel of the camera's first frame met, as it rendered it."""
    pose = camera.taken[0]
    return render(pose.to_matrix() @ mount(), camera.boxes)[1]


class ThePickLoopHandsItsCameraWhatPaintsTests(unittest.TestCase):
    def test_a_pick_hands_its_camera_what_paints_the_tasks_region_and_never_the_part_it_picks(self) -> None:
        orchestrator, camera = _loop()

        report = orchestrator.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        [mask] = camera.hidden
        self.assertEqual((HEIGHT, WIDTH), mask.shape)
        hit = _hits(camera)
        self.assertGreater(float(mask[hit == 1].mean()), 0.9, "the part laid at the drop is not hidden")
        self.assertFalse(mask[hit == 0].any(), "the part the pick takes lost pixels to the paint")

    def test_a_fixed_camera_is_handed_it_too(self) -> None:
        orchestrator, camera = _loop(fixed=True)

        orchestrator.run()

        self.assertEqual(1, len(camera.hidden))

    def test_the_switch_off_hands_nothing_and_the_acquire_is_the_one_it_always_was(self) -> None:
        for wiring in ({"hide_own_places": False}, {"region": None},
                       {"region": ExclusionRegion.circle((0.0, -850.0), 60.0, label="red cube")}):
            with self.subTest(wiring=wiring):
                orchestrator, camera = _loop(**wiring)
                orchestrator.run()
                self.assertEqual([None], camera.handed)

    def test_a_camera_that_cannot_paint_or_no_declared_support_is_handed_nothing_and_said_once(self) -> None:
        for wiring, said in (({"camera_type": WristCamera}, "paints nothing for its detector"),
                             ({"support_config": None}, "no support is declared")):
            with self.subTest(said=said):
                orchestrator, camera = _loop(**wiring)
                with self.assertLogs("src.robot.grasping.loop.pick_loop", level="WARNING") as logged:
                    orchestrator.run()
                    orchestrator.run()
                self.assertEqual(1, sum(said in line for line in logged.output), logged.output)
                self.assertEqual([], getattr(camera, "hidden", []))


class TheCellsKeyTests(unittest.TestCase):
    def test_the_key_is_off_by_default_and_can_be_switched_on(self) -> None:
        self.assertIs(False, RobotGraspingConfig().hide_own_places)
        self.assertIs(True, RobotGraspingConfig.model_validate({"hide_own_places": True}).hide_own_places)
        self.assertIs(False, BinPickingOrchestrator.__dataclass_fields__["hide_own_places"].default)

    def test_a_cell_built_from_its_tree_hides_its_places_as_its_key_says(self) -> None:
        def built(**grasping: object) -> Any:
            config = RobotConfig(vendor="dummy", gripper={"vendor": "none"}, grasping=grasping)
            return AutonomousGraspService.from_robot_config(
                config, calculator=_ScriptedCalculator([]),  # type: ignore[arg-type]
                perception=_FakePerception([_perception_frame()])).runtime.orchestrator

        self.assertIs(False, built().hide_own_places)
        self.assertIs(True, built(hide_own_places=True).hide_own_places)


if __name__ == "__main__":
    unittest.main()
