"""A part's rim is cut off the cloud its footprint is built from, and off nothing else (the owner, 2026-10-09).

``robot.grasping.geometry.footprint_rim_mm`` ("Ja, für Montag"): the D415 smears a part's far edge into a ramp of
depths, the outer ~3 px of the colour mask lie on it, the depth-step rule lets it through, and the support-footprint
stage's hull follows it. Offline on 23 of the cell's recorded grey cubes the centre the grasp closes on lay 2.53 mm off
at the median, 2.5 mm of it away from the camera; with a 3 px rim (about 2 mm) and ``inflate_mm`` 1.25, 0.98 mm. The
rim is cut, ``ceil(rim * fx / z)`` px and never more than 30 % of the mask, off a look's own SFE input in the
calculator and off the footprint cloud the pick loop fuses beside the looks' surfaces; the jaw faces, the association,
the planner world's hold-out and the neighbours keep the whole mask. 2.0 is the default since the owner's evening of
2026-10-09, and with it every mask pixel SFE does not read is placed in the part's visual hull
(``tests/test_a_parts_unread_pixels_stand_in_its_visual_hull.py``); the tests of the rim's own rule here hold that
hull back (:func:`_no_hull`). 0.0 is the input of before.

The part is ray-cast as ``tests/test_a_tilted_view_grasps_on_the_part.py`` casts it: a 40 mm cube on a table at
z = 0, the owner's D415 colour lens (fx 913.7, 1280 x 720) 680 mm away, tilted 36 or 45 degrees, the far edge at the
top of the image. The ramp is what the review measured at the cell: the colour mask reaches 2 px past the depth's
far edge (the depth sits about 2 px lower than the colour), and its outer three rows read the top face's plane pushed
back 7, 5 and 2 mm, which the depth-step rule (10 mm within 3 px) keeps.
"""

from __future__ import annotations

import functools
import logging
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from src.geometry import Frame
from src.robot.grasping.collision import SupportPlane
from src.robot.grasping.generation import calculator as calculator_module
from src.robot.grasping.generation.footprint_rim import FOOTPRINT_RIM_MAX_SHARE, footprint_rim, rim_px
from src.robot.grasping.generation.support_footprint import reconstruct_support_prism

_W, _H, _F = 1280, 720, 913.7
_K = np.array([[_F, 0.0, 640.0], [0.0, _F, 360.0], [0.0, 0.0, 1.0]])
_CENTRE = np.array([500.0, -300.0])
_SIZE = 40.0
#: The ramp the mask's outer three rows at the far edge read, millimetres behind the top face's plane, outermost first.
_RAMP = (7.0, 5.0, 2.0)


@functools.lru_cache(maxsize=None)
def _hande() -> Any:
    from src.config.loader import load_robot_section

    return load_robot_section(profile="hande")


def _replace(model: Any, path: str, value: Any) -> Any:
    """``model`` with the dotted field ``path`` set to ``value``, every level copied and nothing else touched."""
    head, _, rest = path.partition(".")
    if not rest:
        return model.model_copy(update={head: value})
    return model.model_copy(update={head: _replace(getattr(model, head), rest, value)})


def _camera_to_base(tilt_deg: float, distance_mm: float = 680.0) -> np.ndarray:
    """A camera ``distance_mm`` from the cube's centre, ``tilt_deg`` off straight down, looking from -y toward +y."""
    tilt = np.radians(tilt_deg)
    optical = np.array([0.0, np.sin(tilt), -np.cos(tilt)])
    right = np.array([1.0, 0.0, 0.0])
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack([right, np.cross(optical, right), optical])
    matrix[:3, 3] = np.array([_CENTRE[0], _CENTRE[1], _SIZE / 2.0]) - distance_mm * optical
    return matrix


def _rays(camera_to_base: np.ndarray) -> np.ndarray:
    u, v = np.meshgrid(np.arange(_W), np.arange(_H))
    return np.stack([(u - _K[0, 2]) / _F, (v - _K[1, 2]) / _F, np.ones(u.shape)], axis=-1) @ camera_to_base[:3, :3].T


def _ray_cast(camera_to_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The cube's exact mask and the depth of every pixel, cube or table, millimetres along the optical axis."""
    rotation, origin = camera_to_base[:3, :3], camera_to_base[:3, 3]
    rays = _rays(camera_to_base)
    low = np.array([_CENTRE[0] - _SIZE / 2, _CENTRE[1] - _SIZE / 2, 0.0])
    high = np.array([_CENTRE[0] + _SIZE / 2, _CENTRE[1] + _SIZE / 2, _SIZE])
    with np.errstate(divide="ignore", invalid="ignore"):
        t_low, t_high = (low - origin) / rays, (high - origin) / rays
    enter = np.nanmax(np.minimum(t_low, t_high), axis=-1)
    leave = np.nanmin(np.maximum(t_low, t_high), axis=-1)
    cube = (leave >= enter) & (enter > 0.0)
    t = np.where(cube, enter, -origin[2] / rays[..., 2])
    return cube, t * (rays @ rotation[:, 2])


def _ramped(tilt_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(mask, depth in whole mm, camera_to_base)``: the cube with the far-edge ramp the module docstring describes."""
    camera_to_base = _camera_to_base(tilt_deg)
    cube, depth = _ray_cast(camera_to_base)
    rays = _rays(camera_to_base)
    with np.errstate(divide="ignore", invalid="ignore"):
        on_the_top_plane = (_SIZE - camera_to_base[2, 3]) / rays[..., 2] * (rays @ camera_to_base[:3, 2])
    mask = cube.copy()
    ramped = depth.copy()
    for column in np.nonzero(cube.any(axis=0))[0]:
        top = int(np.nonzero(cube[:, column])[0].min())
        for row, back in zip(range(top - 2, top + 1), _RAMP):
            mask[row, column] = True
            ramped[row, column] = on_the_top_plane[row, column] + back
    return mask, np.round(ramped), camera_to_base


def _calculator(**keywords: Any) -> Any:
    cfg = _hande()
    return calculator_module.GraspCalculator(
        camera_matrix=_K, max_grip_width_mm=cfg.gripper.max_width_mm, min_grip_width_mm=cfg.gripper.min_width_mm,
        **keywords)


def _compute(calculator: Any, mask: np.ndarray, depth: np.ndarray, camera_to_base: np.ndarray,
             **keywords: Any) -> tuple[list, dict, np.ndarray]:
    """``(candidates, telemetry, the cloud SFE was handed)`` of one compute on the bench at z = 0."""
    handed: list[np.ndarray] = []
    real = calculator_module.support_footprint_breakdowns

    def spy(cloud: Any, **arguments: Any) -> Any:
        handed.append(np.array(cloud, dtype=np.float64))
        return real(cloud, **arguments)

    with mock.patch.object(calculator_module, "support_footprint_breakdowns", spy):
        candidates = calculator.compute(
            SimpleNamespace(mask=mask), depth, camera_to_base=camera_to_base,
            support_plane=SupportPlane(np.array([0.0, 0.0, 1.0]), 0.0, Frame.BASE), min_table_clearance_mm=1.0,
            **keywords)
    (cloud,) = handed
    return candidates, dict(calculator.last_telemetry), cloud


def _no_hull() -> Any:
    """The calculator with the visual hull its rim comes with held back: the rim's own rule, as it was measured."""
    return mock.patch.object(calculator_module.GraspCalculator, "_with_the_hull",
                             lambda self, target_base, *arguments, **keywords: target_base)


def _footprint(cloud: np.ndarray, inflate_mm: float) -> tuple[float, float, float]:
    """``(centre off the cube's along the view, far face off the cube's, near face off the cube's)``, millimetres, of
    the prism SFE builds from ``cloud``: positive away from the camera, outside the part."""
    prism = reconstruct_support_prism(cloud, 0.0, inflate_mm=inflate_mm)
    assert prism is not None
    far = float(prism.hull[:, 1].max()) + inflate_mm - (_CENTRE[1] + _SIZE / 2)
    near = (_CENTRE[1] - _SIZE / 2) - float(prism.hull[:, 1].min()) + inflate_mm
    return float(prism.centre[1] - _CENTRE[1]), far, near


# --------------------------------------------------------------------------- the rim itself


class TheRimTests(unittest.TestCase):
    def test_the_rim_is_whole_pixels_at_the_parts_depth_rounded_up(self) -> None:
        self.assertEqual(3, rim_px(2.0, _F, 700.0))
        self.assertEqual(2, rim_px(2.0, _F, 961.0))
        self.assertEqual(3, rim_px(2.0, 900.0, 600.0), "exactly three pixels is three, whatever the last bit says")
        self.assertEqual(0, rim_px(0.0, _F, 700.0))

    def test_a_rim_of_nothing_cuts_nothing(self) -> None:
        mask = np.zeros((60, 60), dtype=bool)
        mask[10:50, 10:50] = True
        rim = footprint_rim(mask, np.full(mask.shape, 700.0), _F, 0.0)
        np.testing.assert_array_equal(mask, rim.mask)
        self.assertEqual((0, 0, 0), (rim.px, rim.taken, rim.points))

    def test_the_mask_loses_its_outer_pixels_and_nothing_inside(self) -> None:
        mask = np.zeros((60, 60), dtype=bool)
        mask[10:50, 10:50] = True
        rim = footprint_rim(mask, np.full(mask.shape, 700.0), _F, 2.0)
        expected = np.zeros_like(mask)
        expected[13:47, 13:47] = True
        np.testing.assert_array_equal(expected, rim.mask)
        self.assertEqual((3, 3, 1600 - 34 * 34), (rim.px, rim.asked_px, rim.taken))
        self.assertAlmostEqual(700.0, rim.depth_mm)
        self.assertFalse(rim.limited)
        self.assertIn("a 3 px rim (2 mm at 700 mm)", rim.said())

    def test_the_points_it_took_are_the_rim_pixels_with_a_depth(self) -> None:
        mask = np.zeros((60, 60), dtype=bool)
        mask[10:50, 10:50] = True
        depth = np.full(mask.shape, 700.0)
        depth[10, 10:30] = 0.0  # a hole in the rim: no point to take
        depth[30, 30] = np.nan  # and one inside: no point to keep
        rim = footprint_rim(mask, depth, _F, 2.0)
        self.assertEqual(rim.taken - 20, rim.points)

    def test_a_small_part_loses_a_narrower_rim_never_more_than_its_share(self) -> None:
        """A 20 px square: 3 px would take 51 % of it, 2 px 36 %, 1 px 19 %."""
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:30, 10:30] = True
        rim = footprint_rim(mask, np.full(mask.shape, 700.0), _F, 2.0)
        self.assertEqual((1, 3), (rim.px, rim.asked_px))
        self.assertTrue(rim.limited)
        self.assertLessEqual(rim.taken, FOOTPRINT_RIM_MAX_SHARE * rim.pixels)
        self.assertIn("narrowed", rim.said())

    def test_a_part_too_small_for_a_pixel_keeps_its_mask(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[7:13, 7:13] = True
        rim = footprint_rim(mask, np.full(mask.shape, 700.0), _F, 2.0)
        np.testing.assert_array_equal(mask, rim.mask)
        self.assertEqual(0, rim.px)
        self.assertIn("no rim cut", rim.said())

    def test_a_mask_with_no_depth_under_it_is_not_cut(self) -> None:
        mask = np.zeros((60, 60), dtype=bool)
        mask[10:50, 10:50] = True
        rim = footprint_rim(mask, np.zeros(mask.shape), _F, 2.0)
        np.testing.assert_array_equal(mask, rim.mask)
        self.assertIsNone(rim.depth_mm)

    def test_a_rim_that_is_no_length_or_a_lens_that_is_none_is_refused(self) -> None:
        mask = np.ones((8, 8), dtype=bool)
        for rim_mm, fx in ((-1.0, _F), (float("nan"), _F), (2.0, 0.0)):
            with self.subTest(rim_mm=rim_mm, fx=fx), self.assertRaises(ValueError):
                footprint_rim(mask, np.full(mask.shape, 700.0), fx, rim_mm)


# --------------------------------------------------------------------------- a look's own input, in the calculator


class TheCalculatorCutsTheRimOffSfesInputTests(unittest.TestCase):
    def test_without_a_rim_the_ramp_pulls_the_footprint_out_behind_the_part(self) -> None:
        for tilt in (36.0, 45.0):
            mask, depth, camera_to_base = _ramped(tilt)
            _candidates, telemetry, cloud = _compute(_calculator(), mask, depth, camera_to_base)
            centre, far, _near = _footprint(cloud, 0.0)
            with self.subTest(tilt=tilt):
                self.assertEqual(0, telemetry["support_footprint_depth_step_pixels"], "the ramp passes the step rule")
                self.assertGreater(far, 4.0)
                self.assertGreater(centre, 2.0, "away from the camera, as on the cell")

    def test_a_2_mm_rim_brings_the_footprint_and_its_centre_back_to_the_part(self) -> None:
        """The rim alone, its hull held back: this scene's colour mask reaches 2 px past the cube, which the hull would
        follow, where the cell's recorded cubes showed the colour right and the depth off."""
        for tilt in (36.0, 45.0):
            mask, depth, camera_to_base = _ramped(tilt)
            _c, _t, before = _compute(_calculator(support_footprint_inflate_mm=1.25), mask, depth, camera_to_base)
            with _no_hull():
                _c, telemetry, after = _compute(
                    _calculator(support_footprint_inflate_mm=1.25, support_footprint_rim_mm=2.0), mask, depth,
                    camera_to_base)
            centre_before, far_before, _n = _footprint(before, 1.25)
            centre, far, near = _footprint(after, 1.25)
            with self.subTest(tilt=tilt):
                self.assertEqual(3, telemetry["support_footprint_rim_px"])
                self.assertLess(abs(centre), abs(centre_before))
                self.assertLess(abs(centre), 1.0)
                self.assertLess(abs(far), abs(far_before))
                self.assertLess(abs(far), 1.5)
                self.assertLess(abs(near), 2.0, "the near face keeps its place with the faces given back")

    def test_the_rim_and_the_points_it_took_are_stamped_and_said_every_compute(self) -> None:
        mask, depth, camera_to_base = _ramped(45.0)
        calculator = _calculator(support_footprint_rim_mm=2.0)
        with self.assertLogs(calculator.logger, level=logging.INFO) as said, _no_hull():
            _c, telemetry, cloud = _compute(calculator, mask, depth, camera_to_base)
        _c, _t, whole = _compute(_calculator(), mask, depth, camera_to_base)
        self.assertEqual(whole.shape[0] - cloud.shape[0], telemetry["support_footprint_rim_points"])
        self.assertGreater(telemetry["support_footprint_rim_points"], 0)
        self.assertTrue(any("footprint rim: a 3 px rim (2 mm at" in line and "point(s) off SFE's input" in line
                            for line in said.output), said.output)

    def test_no_rim_is_the_input_of_before_byte_for_byte(self) -> None:
        mask, depth, camera_to_base = _ramped(45.0)
        before, telemetry_before, cloud_before = _compute(_calculator(), mask, depth, camera_to_base)
        after, telemetry_after, cloud_after = _compute(_calculator(support_footprint_rim_mm=0.0), mask, depth,
                                                       camera_to_base)
        np.testing.assert_array_equal(cloud_before, cloud_after)
        self.assertEqual(len(before), len(after))
        for left, right in zip(before, after):
            np.testing.assert_array_equal(left.position, right.position)
            np.testing.assert_array_equal(left.axis, right.axis)
            self.assertEqual(left.score, right.score)
        self.assertNotIn("support_footprint_rim_px", telemetry_after)
        self.assertEqual(telemetry_before, telemetry_after)

    def test_the_full_mask_still_reaches_the_neighbours_and_the_target_cloud(self) -> None:
        """The scene the open hand keeps off is read beside the whole mask, and the target's own cloud is the whole
        mask's: only SFE's input loses the rim."""
        from src.robot.grasping.generation.scene_obstacles import SceneObstacleRules

        rules = SceneObstacleRules(bench_top_mm=0.0, band_mm=5.0, min_points=12, cluster_voxel_mm=25.0, pixel_stride=2,
                                   support_distance_mm=5.0, declared_distance_mm=5.0)
        mask, depth, camera_to_base = _ramped(45.0)
        seen: dict[str, list[np.ndarray]] = {"off": [], "on": []}
        real = calculator_module.scene_obstacle_points
        telemetry: dict[str, dict] = {}
        for name, rim in (("off", 0.0), ("on", 2.0)):
            def spy(depth_mm: Any, target_mask: Any, *args: Any, name: str = name, **keywords: Any) -> Any:
                seen[name].append(np.array(target_mask, dtype=bool))
                return real(depth_mm, target_mask, *args, **keywords)

            with mock.patch.object(calculator_module, "scene_obstacle_points", spy):
                _c, telemetry[name], _cloud = _compute(
                    _calculator(scene_obstacles=rules, support_footprint_rim_mm=rim), mask, depth, camera_to_base)
        np.testing.assert_array_equal(mask, seen["on"][0])
        np.testing.assert_array_equal(seen["off"][0], seen["on"][0])
        for key in ("target_cloud_z_min_mm", "target_cloud_z_max_mm", "scene_obstacle_points", "scene_support_mm"):
            with self.subTest(key=key):
                self.assertEqual(telemetry["off"][key], telemetry["on"][key])

    def test_the_rim_never_leaves_sfe_nothing(self) -> None:
        """A part whose only depth lies on its outermost pixel keeps its rim: a footprint is never built from nothing."""
        mask, depth, camera_to_base = _ramped(45.0)
        holed = depth.copy()
        holed[footprint_rim(mask, depth, _F, 0.5).mask] = 0.0  # a 1 px rim keeps its depth, nothing inside it does
        with _no_hull():
            _c, telemetry, cloud = _compute(_calculator(support_footprint_rim_mm=2.0), mask, holed, camera_to_base)
        _c, _t, whole = _compute(_calculator(), mask, holed, camera_to_base)
        self.assertEqual(0, telemetry["support_footprint_rim_px"])
        np.testing.assert_array_equal(whole, cloud)


class AFusedCloudComesWithItsRimCutTests(unittest.TestCase):
    """The pick loop hands the looks' surfaces less their rim beside the fused cloud; SFE reads that alone."""

    def _fused(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from src.robot.grasping.multiview.scene_geometry import to_base_mm

        mask, depth, camera_to_base = _ramped(45.0)
        whole = to_base_mm(mask, depth, _K, camera_to_base)
        foot = to_base_mm(footprint_rim(mask, depth, _F, 2.0).mask, depth, _K, camera_to_base)
        return mask, depth, camera_to_base, whole, foot

    def test_sfe_reads_the_footprint_and_says_how_many_points_it_has_fewer(self) -> None:
        mask, depth, camera_to_base, whole, foot = self._fused()
        calculator = _calculator(support_footprint_rim_mm=2.0)
        with self.assertLogs(calculator.logger, level=logging.INFO) as said:
            _c, telemetry, cloud = _compute(calculator, mask, depth, camera_to_base, geometry_points_base_mm=whole,
                                            footprint_points_base_mm=foot)
        np.testing.assert_array_equal(foot, cloud[:foot.shape[0]])
        self.assertEqual(cloud.shape[0] - foot.shape[0], telemetry["support_footprint_hull_points"],
                         "this look's unread pixels placed in the part's visual hull beside the footprint")
        self.assertGreater(telemetry["support_footprint_hull_points"], 0)
        self.assertEqual(whole.shape[0] - foot.shape[0], telemetry["support_footprint_rim_points"])
        self.assertNotIn("support_footprint_rim_px", telemetry, "the looks' rims were cut where they were seen")
        self.assertTrue(any("reads the looks' surfaces less their rim" in line for line in said.output), said.output)

    def test_without_a_footprint_the_fused_cloud_is_read_whole_as_before(self) -> None:
        mask, depth, camera_to_base, whole, _foot = self._fused()
        _c, telemetry, cloud = _compute(_calculator(support_footprint_rim_mm=2.0), mask, depth, camera_to_base,
                                        geometry_points_base_mm=whole)
        np.testing.assert_array_equal(whole, cloud)
        self.assertNotIn("support_footprint_rim_points", telemetry)

    def test_a_footprint_with_no_point_leaves_the_fused_cloud_whole(self) -> None:
        mask, depth, camera_to_base, whole, _foot = self._fused()
        _c, telemetry, cloud = _compute(_calculator(support_footprint_rim_mm=2.0), mask, depth, camera_to_base,
                                        geometry_points_base_mm=whole, footprint_points_base_mm=np.zeros((0, 3)))
        np.testing.assert_array_equal(whole, cloud)
        self.assertEqual(0, telemetry["support_footprint_rim_points"])

    def test_a_footprint_with_no_fused_cloud_beside_it_is_not_read(self) -> None:
        """What a blocker's call would carry if the part's footprint stayed in it: the blocker's own mask decides."""
        mask, depth, camera_to_base, _whole, foot = self._fused()
        _c, _t, alone = _compute(_calculator(), mask, depth, camera_to_base)
        _c, _t, cloud = _compute(_calculator(), mask, depth, camera_to_base, footprint_points_base_mm=foot)
        np.testing.assert_array_equal(alone, cloud)


# --------------------------------------------------------------------------- the key


class TheKeyTests(unittest.TestCase):
    def test_two_millimetres_is_the_default_of_the_schema_and_of_the_repositorys_tree(self) -> None:
        """On since the owner's evening of 2026-10-09, the visual hull giving the faces back: inflate_mm stays 0."""
        from src.config.loader import load_robot_section
        from src.config.schema.robot.grasping_schema import GraspingGeometryStageConfig

        self.assertEqual(2.0, GraspingGeometryStageConfig().footprint_rim_mm)
        self.assertEqual(2.0, load_robot_section().grasping.geometry.footprint_rim_mm)
        self.assertEqual(0.0, load_robot_section().grasping.geometry.inflate_mm, "inflate_mm's default stays")

    def test_a_negative_rim_or_one_past_ten_millimetres_is_refused(self) -> None:
        from pydantic import ValidationError

        from src.config.schema.robot.grasping_schema import GraspingGeometryStageConfig

        self.assertEqual(2.0, GraspingGeometryStageConfig(footprint_rim_mm=2.0).footprint_rim_mm)
        for wrong in (-0.5, 10.5):
            with self.subTest(wrong=wrong), self.assertRaises(ValidationError):
                GraspingGeometryStageConfig(footprint_rim_mm=wrong)

    def test_the_factory_hands_it_to_the_calculator_only_where_it_is_set(self) -> None:
        from src.config.schema.robot.grasping_schema import GraspingGeometryStageConfig
        from src.robot.grasping.calculator_factory import _depth_kwargs, build_calculator

        def tree(**geometry: Any) -> Any:
            return SimpleNamespace(grasping=SimpleNamespace(geometry=GraspingGeometryStageConfig(**geometry)))

        self.assertEqual({"support_footprint_rim_mm": 2.0}, _depth_kwargs(tree(), {}), "the default reaches it")
        self.assertEqual({}, _depth_kwargs(tree(footprint_rim_mm=0.0), {}))
        self.assertEqual({"support_footprint_rim_mm": 3.0}, _depth_kwargs(tree(footprint_rim_mm=3.0), {}))
        self.assertEqual({}, _depth_kwargs(tree(footprint_rim_mm=2.0), {"support_footprint_rim_mm": 0.0}),
                         "a construction site's own value wins")
        cfg = _replace(_hande(), "grasping.geometry.footprint_rim_mm", 0.0)
        self.assertEqual(0.0, build_calculator(cfg, camera_matrix=_K).support_footprint_rim_mm)
        self.assertEqual(2.0, build_calculator(_hande(), camera_matrix=_K).support_footprint_rim_mm)

    def test_a_deep_cell_says_it_ignores_the_rim_and_does_not_refuse_over_it(self) -> None:
        from src.robot.grasping.calculator_factory import _IGNORED_BY_DEEP

        self.assertIn("support_footprint_rim_mm", _IGNORED_BY_DEEP)

    def test_a_calculator_refuses_a_rim_that_is_no_length(self) -> None:
        for wrong in (-1.0, float("inf"), float("nan")):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                _calculator(support_footprint_rim_mm=wrong)

    def test_the_reference_tree_writes_it_down_with_its_default(self) -> None:
        from pathlib import Path

        text = (Path(__file__).resolve().parents[1] / "config" / "all_keys" / "robot" / "robot.yaml").read_text(
            encoding="utf-8")
        before, _, after = text.partition("      footprint_rim_mm: 2.0\n")
        self.assertTrue(after, "robot.grasping.geometry.footprint_rim_mm is not in config/all_keys/robot/robot.yaml")
        self.assertIn("# [default: 2.0]", before.splitlines()[-1])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
