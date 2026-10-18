"""The calculator sees what the prompt did not name, by the planner world's rules, and refuses nothing for it where there
is room.

The scene (the cell fixes' Track A, every cell's) hands the calculator the depth around the part: what the
camera world keeps there is an obstacle, the support the part stands on is not. The pins here are the other side of
``test_a_boxed_in_part_gets_no_grasp_and_says_so.py``:

* an isolated part, a part with cubes 40-70 mm off (E3's photo layout) and a part drawn onto an empty patch of the owner's
  real, tilted, noisy mat keep every grasp SFE offered without the scene;
* the rim of a mask cut a little short is the part's own edge, not a neighbour;
* off is the calculator of before, byte for byte; every cell built from a tree runs both, a deep cell builds with them;
* a tilted approach is offered only through space this frame's depth saw (contract 5), and the push gets the world rule's
  points (contract 2).
"""

from __future__ import annotations

import hashlib
import logging
import unittest
from typing import Any

import numpy as np

from src.robot.grasping.collision.support_resolver import TABLE_CLEARANCE_MM
from scipy import ndimage

from tests.robot.grasping.collision.test_a_boxed_in_part_gets_no_grasp_and_says_so import (
    BENCH,
    MAT,
    MAT_MM,
    PART,
    Frame_,
    calculator,
    compute,
    owner_rules,
)
from tests.robot.grasping.generation.test_a_side_grasp_never_goes_through_unseen_space import Box, Cylinder

def digest(result: Any) -> str:
    """The candidates rounded, with the sign of a rounded zero dropped, the telemetry's keys and the reasons.

    OpenBLAS picks its kernels by CPU: the owner's AVX-512 machine runs SkylakeX's, CI's runner Haswell's or Zen's. Their
    last bits differ (1e-13 mm, 5e-17 in an axis), and a coordinate that is zero lands on either side of it; rounding
    keeps that sign, ``repr`` writes ``-0.0``, and the digest of the very same grasps was another one on CI.
    """
    h = hashlib.sha256()
    for g in result.candidates:
        for v in (*np.round(g.position, 6), *np.round(g.approach, 9), *np.round(g.axis, 9),
                  round(float(g.grip_width_mm), 6), round(float(g.score), 9)):
            h.update(repr(float(v) + 0.0).encode())
    h.update(repr(sorted(result.telemetry)).encode())
    h.update(repr([r.value for r in result.reasons]).encode())
    return h.hexdigest()


def isolated() -> Frame_:
    return Frame_((BENCH, MAT), PART)


#: E3's photo layout: the part between cubes 40 to 70 mm off it.
PHOTO_NEIGHBOURS = (Box(lo=(60.0, -670.0, MAT_MM), hi=(100.0, -630.0, MAT_MM + 40.0)),        # 40 mm off on +x
                    Box(lo=(-130.0, -680.0, MAT_MM), hi=(-90.0, -640.0, MAT_MM + 40.0)),      # 70 mm off on -x
                    Cylinder((0.0, -740.0), 20.0, MAT_MM, MAT_MM + 60.0))                     # 50 mm off on -y


def photo_layout() -> Frame_:
    """E3's photo layout: the part between cubes 40 to 70 mm off it."""
    return Frame_((BENCH, MAT, *PHOTO_NEIGHBOURS), PART)


def scene_off_calculator() -> Any:
    """The calculator a caller builds without the new keyword at all."""
    from src.robot.grasping.calculator_factory import build_calculator
    from tests.robot.grasping.collision.test_a_boxed_in_part_gets_no_grasp_and_says_so import FOCAL, HEIGHT, WIDTH, _hande

    cfg = _hande()
    return build_calculator(cfg, camera_matrix=np.array([[FOCAL, 0.0, WIDTH / 2.0], [0.0, FOCAL, HEIGHT / 2.0],
                                                         [0.0, 0.0, 1.0]]),
                            max_grip_width_mm=cfg.gripper.max_width_mm, min_grip_width_mm=cfg.gripper.min_width_mm)


def same_grasps(a: Any, b: Any) -> bool:
    if len(a.candidates) != len(b.candidates):
        return False
    return all(np.allclose(x.position, y.position) and np.allclose(x.approach, y.approach)
               and np.allclose(x.axis, y.axis) for x, y in zip(a.candidates, b.candidates))


class NoFalseRefusalWhereThereIsRoomTests(unittest.TestCase):
    """Track A test 2, on ray-cast scenes."""

    def test_an_isolated_part_keeps_every_grasp(self) -> None:
        frame = isolated()
        off, on = compute(calculator(scene=False), frame), compute(calculator(scene=True), frame)
        self.assertGreater(len(off.candidates), 0)
        self.assertTrue(same_grasps(off, on))
        self.assertEqual(0, on.telemetry["scene_obstacle_points"])

    def test_cubes_40_to_70_mm_off_take_away_only_what_the_guard_refuses(self) -> None:
        """The cubes are seen, and of the grasps offered without them those whose open hand comes nearer the boxes the
        camera world builds of them than the guard's 5 mm are taken away: a tilted approach toward the cube 40 mm off,
        and a vertical one closing toward it, came within 3 mm of its box (the owner, 2026-10-03). None is offered that
        the guard refuses; the body's seen grid of 2 mm cells holds a hand up to a cell further off than the guard's
        boxes do, so a grasp a little over 5 mm from a box may go too (2026-10-13)."""
        from tests.robot.grasping.generation.test_the_calculator_keeps_the_guards_distance_from_a_neighbour import GUARD_MM, least_mm, world_boxes

        frame = photo_layout()
        # 36 grasps a side, as the pick loop asks where a closing axis is named: the round part's axes tie, and since
        # its outline is read in its visual hull (2026-10-09) its first 12 grasps share one axis, every one of them too
        # near the cube 40 mm off; the scene's calculator offers 9 that all keep the guard's distance even at 12.
        unseen, seen = calculator(scene=False), calculator(scene=True)
        unseen.max_candidates = seen.max_candidates = 36
        off, on = compute(unseen, frame), compute(seen, frame)
        self.assertGreater(on.telemetry["scene_obstacle_points"], 0)
        boxes = world_boxes(PHOTO_NEIGHBOURS, frame)
        refused = [g for g in off.candidates if least_mm(g, boxes) < GUARD_MM]
        self.assertTrue(refused and on.candidates, (len(refused), len(on.candidates)))
        key = lambda g: tuple(np.round(np.concatenate([g.position, g.axis, g.approach]), 2))  # noqa: E731
        offered = set(map(key, on.candidates))
        self.assertFalse(set(map(key, refused)) & offered, "a grasp the guard refuses beside the cubes is offered")
        self.assertTrue(all(least_mm(g, boxes) >= GUARD_MM for g in on.candidates))

    def test_a_mask_cut_short_at_its_rim_is_still_the_part(self) -> None:
        """Track A test 5: the 5 mm the mask is grown by holds a rim the segmenter left out (2 px here, 2 mm)."""
        frame = isolated()
        short = ndimage.binary_erosion(frame.mask, iterations=2)
        off = compute(calculator(scene=False), frame, mask=short)
        on = compute(calculator(scene=True), frame, mask=short)
        self.assertGreater(len(on.candidates), 0)
        self.assertTrue(same_grasps(off, on))


class NoFalseRefusalOnTheRealMatTests(unittest.TestCase):
    """Track A test 2 on the owner's recorded looks: a part drawn onto an empty patch of that tilted, noisy mat keeps the
    grasps SFE offered without the scene (``R_SCOPE/a_rule_mat_noise_v2.out``: 10 to 12 for 30 and 48 mm parts)."""

    PATCHES = ((-341.0, -782.0), (9.0, -572.0))

    def _run(self, name: str, x: float, y: float, height_mm: float) -> tuple[Any, Any, Any]:
        from src.robot.grasping.collision.gripper_model import jaw_envelope
        from src.robot.grasping.collision import SupportPlane
        from src.robot.safety.planning.self_envelope import ROBOT_BASES
        from src.robot.safety.planning.support_surfaces import find_supports
        from src.geometry import Frame, Transform
        from tests._cell_2026_10_01 import OWNER_LIMITS, depth_view, local_reading, look, owner_tuning, with_solids
        from tests.robot.grasping.collision.test_a_boxed_in_part_gets_no_grasp_and_says_so import _hande
        from src.robot.grasping.calculator_factory import build_calculator
        from types import SimpleNamespace

        recorded = look(name)
        ground = local_reading(recorded, x, y)
        drawn = with_solids(recorded, cylinders=[(x, y, 20.0, ground - 5.0, ground + height_mm)])
        mask = np.abs(drawn - recorded.surface_depth_mm) > 0.0
        model = find_supports([depth_view(recorded, surface_depth_mm=drawn)], limits=OWNER_LIMITS,
                              tuning=owner_tuning(support_surfaces=True, support_allowance_mm=2.0),
                              base=ROBOT_BASES.get("ur10"))
        rows, cols = np.nonzero(mask)
        k, t = recorded.intrinsics, recorded.camera_to_base
        z = drawn[rows, cols]
        foot = np.column_stack(((cols - k[0, 2]) * z / k[0, 0], (rows - k[1, 2]) * z / k[1, 1], z)) @ t[:3, :3].T
        foot = foot + t[:3, 3]
        support = model.height_under(foot[:, :2])
        assert support is not None, "the patch lies on the mat"
        cfg = _hande()

        def run(scene: bool, with_model: bool) -> Any:
            extra: dict[str, Any] = {"side_approaches": True}
            if scene:
                extra["scene_obstacles"] = owner_rules()
            calc = build_calculator(cfg, camera_matrix=k, max_grip_width_mm=cfg.gripper.max_width_mm,
                                    min_grip_width_mm=cfg.gripper.min_width_mm, **extra)
            return calc.compute_result(
                SimpleNamespace(mask=mask, label="part", score=1.0), drawn, other_object_masks=[],
                camera_to_base=Transform.from_matrix(t, from_frame=Frame.CAMERA, to_frame=Frame.BASE),
                support_plane=SupportPlane(normal=np.array([0.0, 0.0, 1.0]), offset_mm=float(support),
                                           frame=Frame.BASE),
                min_table_clearance_mm=TABLE_CLEARANCE_MM,
                gripper_model=jaw_envelope(cfg),
                **({"support_model": model} if with_model else {}))

        return run(False, False), run(True, True), (x, y)

    def test_an_empty_patch_of_the_tilted_mat_is_no_neighbour(self) -> None:
        for name in ("P1", "P5"):
            for x, y in self.PATCHES:
                for height in (30.0, 48.0):
                    with self.subTest(look=name, patch=(x, y), height=height):
                        off, on, centre = self._run(name, x, y, height)
                        # The body's grasps, the same with the scene as without: the mat is no neighbour.
                        offered = on.telemetry["body_kept"]
                        self.assertEqual(off.telemetry["body_kept"], offered)
                        self.assertGreaterEqual(offered, 10)
                        # Nothing of the mat within the open hand's reach of the part (the Hand-E open stands 36 mm
                        # off its axis); what tilted corridors meet further off is the pile, refused builds, no grasp.
                        kept = on.metadata["scene_obstacle_points_base_mm"]
                        near = np.hypot(kept[:, 0] - centre[0], kept[:, 1] - centre[1]) < 60.0
                        self.assertEqual(0, int(near.sum()))
                        if height >= 48.0:
                            self.assertGreater(len(on.candidates), 0)


class TheSwitchesTests(unittest.TestCase):
    """Track A test 7: off is the calculator without the keyword, byte for byte; every cell built from a tree runs both;
    a deep cell builds with them."""

    def test_off_is_the_calculator_of_before_byte_for_byte(self) -> None:
        from tests.robot.grasping.collision.test_a_boxed_in_part_gets_no_grasp_and_says_so import boxed_in_bar

        for name, frame in (("isolated", isolated()), ("photo_layout", photo_layout()),
                            ("boxed_in_bar", boxed_in_bar())):
            with self.subTest(name):
                plain = compute(scene_off_calculator(), frame)
                passed_none = compute(calculator(scene=False, side=False), frame)
                self.assertEqual(digest(plain), digest(passed_none))
                for key in ("scene_obstacle_points", "candidates_final_source", "rejected_support"):
                    self.assertNotIn(key, plain.telemetry)
                self.assertNotIn("scene_obstacle_points_base_mm", plain.metadata)

    def test_a_cell_built_from_its_tree_reads_the_scene_and_approaches_from_the_side(self) -> None:
        from src.config.loader import load_robot_section
        from src.robot.execution.autonomous_grasp.cells import build_rehearsal_components

        cfg = load_robot_section(profile="hande")
        calc = build_rehearsal_components(cfg)[0]
        self.assertIsNotNone(calc._scene_obstacles)  # noqa: SLF001
        self.assertTrue(calc._side_approaches)  # noqa: SLF001

    def test_the_cell_hands_SFE_side_approaches_and_a_corridor_test(self) -> None:
        result = compute(calculator(scene=True, side=True), isolated())
        self.assertTrue(result.telemetry["support_footprint_side_approaches"])
        self.assertTrue(result.telemetry["support_footprint_corridor_seen"])

    def test_a_deep_cell_builds_with_both_keys_and_says_it_ignores_them(self) -> None:
        import tempfile
        from pathlib import Path
        from unittest import mock

        from src.robot.grasping.calculator_factory import build_calculator
        from src.robot.grasping.generation.scene_obstacles import SceneObstacleRules
        from tests.robot.grasping.test_calculator_factory_drops_nothing_silently import _artifact, _config

        with tempfile.TemporaryDirectory() as name:
            cfg = _config("deep", _artifact(Path(name)))
            with mock.patch("src.robot.grasping.deep.calculator.DeepGraspCalculator"):
                with self.assertLogs("CalculatorFactory", level=logging.WARNING) as said:
                    build_calculator(cfg, camera_matrix=None, scene_obstacles=SceneObstacleRules(),
                                     side_approaches=True)
        self.assertTrue(any("scene_obstacles" in line and "side_approaches" in line for line in said.output))

    def test_scene_obstacles_takes_rules_or_nothing(self) -> None:
        from src.robot.grasping.generation.calculator import GraspCalculator

        with self.assertRaises(TypeError):
            GraspCalculator(scene_obstacles=True)  # type: ignore[arg-type]


class TheRulesAreTheWorldsTests(unittest.TestCase):
    """``SceneObstacleRules.from_robot_config`` reads what the camera world reads."""

    def test_the_owners_cell(self) -> None:
        from src.config.schema.robot import RobotConfig
        from src.robot.grasping.generation.scene_obstacles import SceneObstacleRules
        from tests._cell_2026_10_01 import owner_robot

        rules = SceneObstacleRules.from_robot_config(RobotConfig.model_validate(owner_robot()))
        self.assertEqual(0.0, rules.bench_top_mm)
        self.assertEqual(5.0, rules.band_mm)
        self.assertEqual((12, 25.0, 2), (rules.min_points, rules.cluster_voxel_mm, rules.pixel_stride))
        self.assertEqual((5.0, 5.0), (rules.support_distance_mm, rules.declared_distance_mm))
        self.assertEqual(["support_plane"], [body.name for body in rules.declared])
        self.assertEqual((), rules.declared_boxes)

    def test_a_declared_fixture_is_a_box_the_hand_keeps_its_distance_from(self) -> None:
        from src.config.schema.robot import RobotConfig
        from src.robot.grasping.generation.scene_obstacles import SceneObstacleRules
        from tests._cell_2026_10_01 import owner_robot

        robot = owner_robot()
        robot["safety"]["self_collision"].update(
            min_distance_mm=10.0,
            fixtures=[{"name": "tray", "center_mm": [0.0, -600.0, 20.0], "half_extents_mm": [50.0, 50.0, 20.0]}])
        rules = SceneObstacleRules.from_robot_config(RobotConfig.model_validate(robot))
        self.assertEqual(["tray"], [box.name for box in rules.declared_boxes])
        self.assertEqual(10.0, rules.declared_distance_mm)
        # A fixture stays the guard's (2026-10-17): the camera world drops nothing it sees about one as the fixture's.
        self.assertEqual(["support_plane"], [body.name for body in rules.declared])

    def test_a_tree_with_no_planning_world_has_no_bench(self) -> None:
        from src.config.loader import load_robot_section
        from src.robot.grasping.generation.scene_obstacles import SceneObstacleRules

        rules = SceneObstacleRules.from_robot_config(load_robot_section(profile="hande"))
        self.assertIsNone(rules.bench_top_mm)
        self.assertEqual((), rules.declared)


class OnlySeenSpaceIsClearTests(unittest.TestCase):
    """Contract 5: a BASE point is seen where its pixel's measured depth reaches at least to it, 5 mm tolerance."""

    def test_in_front_on_and_behind_the_measured_surface(self) -> None:
        from src.robot.grasping.generation.scene_obstacles import corridor_seen_in

        frame = isolated()
        seen = corridor_seen_in(frame.depth, frame.intrinsics, frame.camera_to_base)
        eye = np.asarray(frame.camera_to_base[:3, 3])
        top = np.array([0.0, -650.0, MAT_MM + 50.0])                    # the part's lid
        ray = (top - eye) / np.linalg.norm(top - eye)
        points = np.array([top - 30.0 * ray,                             # in front of the lid: seen
                           top + 4.0 * ray,                              # within the tolerance behind it: seen
                           top + 12.0 * ray,                             # inside the part, behind its lid: not seen
                           [0.0, -650.0, 2000.0],                        # behind the camera: not seen
                           [3000.0, -650.0, 60.0]])                      # outside the image: not seen
        self.assertEqual([True, True, False, False, False], seen(points).tolist())

    def test_a_pixel_with_no_depth_says_nothing_is_seen(self) -> None:
        from src.robot.grasping.generation.scene_obstacles import corridor_seen_in

        frame = isolated()
        seen = corridor_seen_in(np.zeros_like(frame.depth), frame.intrinsics, frame.camera_to_base)
        self.assertFalse(seen(np.array([[0.0, -650.0, 150.0]])).any())
        self.assertEqual((0,), seen(np.zeros((0, 3))).shape)


class WhatAnotherLookSawTests(unittest.TestCase):
    """The guard holds the boxes of every look of a pick; the calculator, handed the other looks
    (``scene_obstacles.SceneView``), holds them too: a neighbour only another look saw keeps the open hand off it
    (2026-10-13, the part at the wall: every grasp straight down, the housing at a wall the last look did not show)."""

    def test_a_neighbour_only_another_look_saw_keeps_the_open_fingers_off_it(self) -> None:
        from src.robot.grasping.generation.scene_obstacles import SceneView
        from tests.robot.grasping.collision.test_a_boxed_in_part_gets_no_grasp_and_says_so import (
            BENCH,
            MAT,
            MAT_MM,
            PART,
            Frame_,
            calculator,
            compute,
            open_fingers_hold,
            solid_points,
        )

        cube = Box(lo=(35.0, -670.0, MAT_MM), hi=(75.0, -630.0, MAT_MM + 40.0))   # 15 mm off the part along +x
        here, other = Frame_((BENCH, MAT), PART), Frame_((BENCH, MAT, cube), PART)
        inside = solid_points((cube,))
        blind = compute(calculator(scene=True), here)
        self.assertTrue(any(open_fingers_hold(g, inside) for g in blind.candidates),
                        "the look that does not show the cube offers grasps whose open fingers stand in it")
        view = SceneView(depth_mm=other.depth, intrinsics=other.intrinsics, camera_to_base=other.camera_to_base)
        told = compute(calculator(scene=True), here, scene_views=(view,))
        self.assertTrue(told.candidates)
        self.assertEqual(1, told.telemetry["scene_views"])
        self.assertFalse(any(open_fingers_hold(g, inside) for g in told.candidates))

    def test_another_look_leaves_the_part_out_as_the_guard_does(self) -> None:
        """The part's pixels in another look are the ones whose points stand in its keep-out box, as the guard leaves
        them out (``perceived.target_keep_out_box``): its faces only that look saw among them, and nothing that stands
        clear of the box."""
        from src.robot.grasping.generation.scene_obstacles import SceneObstacleRules, SceneView, _the_parts_keep_out
        from tests.robot.grasping.collision.test_a_boxed_in_part_gets_no_grasp_and_says_so import BENCH, MAT, MAT_MM, PART, Frame_

        frame = Frame_((BENCH, MAT), PART)
        view = SceneView(depth_mm=frame.depth, intrinsics=frame.intrinsics, camera_to_base=frame.camera_to_base)
        box = _the_parts_keep_out(frame.target_points(), SceneObstacleRules(), MAT_MM)
        assert box is not None
        held = view.held_out(box)
        self.assertGreater(float((held & frame.mask).sum()) / float(frame.mask.sum()), 0.95)
        # A part 100 mm beside this one holds out none of this one's pixels, though they are its shape and its height.
        beside = _the_parts_keep_out(frame.target_points() + np.array([100.0, 0.0, 0.0]), SceneObstacleRules(), MAT_MM)
        assert beside is not None
        self.assertFalse((view.held_out(beside) & frame.mask).any())


if __name__ == "__main__":
    unittest.main()
