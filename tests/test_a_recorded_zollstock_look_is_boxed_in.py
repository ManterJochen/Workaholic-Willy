"""The owner's recorded Zollstock looks: the calculator of today offers three grasps that meet a neighbour, the cell's
calculator with the scene offers none and says why.

P1 (``pick-63e1b3dac29f``) and P5 (``pick-44c3e7e060ac``), 2026-10-01, as K ships them (depth only). The Zollstock lay in
a pile of parts on the mat; the prompt named it alone, the calculator saw it alone, and every grasp it offered put an
open finger inside a part beside it (E2/E6: 289-433 one-mm voxels in a finger box), so the exact guard refused each one
and the pick ended with no typed reason. Replayed here the way the pick loop called the calculator (E2's faithful
replay, ``pick_loop.py:4264-4384``): the owner's Hand-E, the dense mode of that profile, the support the single view
resolved.

With ``robot.grasping.scene_obstacles`` the same calculator offers nothing on both looks, with the support model of
the look and without it: ``ALL_COLLIDED`` first, the reason the push answers, and the sentence that a neighbour the
camera saw is in the way. On all five of the owner's recorded looks (P1-P5, ``cellfix/ae/work/replay_zollstock.out``):
today 3 of 3 grasps with an open finger in a neighbour, after 0 grasps.
"""

from __future__ import annotations

import unittest
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.geometry import Frame, Transform


@lru_cache(maxsize=None)
def _cell() -> Any:
    """The owner's grasping as the shipped ``hande`` profile carries it (the same Hand-E: 49.99 mm stroke, fingertip
    10.45 mm), in that profile's mode."""
    from src.config.loader import load_robot_section

    cfg = load_robot_section(profile="hande")
    return cfg.model_copy(update={"grasping": cfg.grasping.model_copy(update={"default_mode": "dense_clutter"})})


@lru_cache(maxsize=None)
def _rules() -> Any:
    """The planner world's rules on the owner's cell (its bench, band, voxel rule and 5 mm distances)."""
    from src.config.schema.robot import RobotConfig
    from src.robot.grasping.generation.scene_obstacles import SceneObstacleRules
    from tests._cell_2026_10_01 import owner_robot

    return SceneObstacleRules.from_robot_config(RobotConfig.model_validate(owner_robot()))


def _support(recorded: Any) -> Any:
    """``pick_loop._resolve_support``, single view: the declared support raised to the target's lowest point when its
    cloud reaches 25 mm along the support normal."""
    from src.robot.grasping.collision import resolve_support_plane
    from src.robot.grasping.geometry.pointcloud import masked_point_cloud

    support = _cell().grasping.support
    cloud = masked_point_cloud(recorded.target_mask, recorded.depth_mm, recorded.intrinsics)
    base = (np.asarray(cloud.points_mm, dtype=float).reshape(-1, 3) @ recorded.camera_to_base[:3, :3].T
            + recorded.camera_to_base[:3, 3])
    along = base @ np.asarray(support.normal, dtype=float)
    clouds = [base] if support.refine_from_target and float(along.max() - along.min()) >= 25.0 else None
    return resolve_support_plane(declared_height_mm=support.height_mm,
                                 container_floor_mm=support.container.floor_height_mm, normal=support.normal,
                                 target_clouds_base_mm=clouds, refine_from_target=support.refine_from_target)


def _model(recorded: Any) -> Any:
    from src.robot.safety.planning.self_envelope import ROBOT_BASES
    from src.robot.safety.planning.support_surfaces import find_supports
    from tests._cell_2026_10_01 import OWNER_LIMITS, depth_view, owner_tuning

    return find_supports([depth_view(recorded)], limits=OWNER_LIMITS,
                         tuning=owner_tuning(support_surfaces=True, support_allowance_mm=2.0),
                         base=ROBOT_BASES.get("ur10"))


def _compute(recorded: Any, *, scene: bool, model: Any = None) -> Any:
    from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry
    from src.robot.execution.autonomous_grasp.config import _profile_for, resolve_grasp_mode
    from src.robot.grasping.calculator_factory import build_calculator

    cfg = _cell()
    extra: dict[str, Any] = {"side_approaches": True}
    if scene:
        extra["scene_obstacles"] = _rules()
    calc = build_calculator(cfg, camera_matrix=recorded.intrinsics, max_grip_width_mm=cfg.gripper.max_width_mm,
                            min_grip_width_mm=cfg.gripper.min_width_mm, support_footprint_geometry=True,
                            support_footprint_inflate_mm=0.0, **extra)
    support = _support(recorded)
    return calc.compute_result(
        SimpleNamespace(mask=recorded.target_mask, label="Einen Zollstock", score=1.0), recorded.depth_mm,
        pixel_to_mm=None, dense_sampling=True,
        grasp_sampling_mode=_profile_for(resolve_grasp_mode(cfg.grasping.default_mode)).sampling_mode,
        other_object_masks=[],
        camera_to_base=Transform.from_matrix(recorded.camera_to_base, from_frame=Frame.CAMERA, to_frame=Frame.BASE),
        support_plane=support.plane, min_table_clearance_mm=cfg.grasping.support.min_clearance_mm,
        gripper_model=build_gripper_geometry(cfg.grasping.gripper_geometry),
        **({"support_model": model} if model is not None else {}))


def _in_open_fingers(grasp: Any, points: np.ndarray) -> int:
    """How many ``points`` stand inside the open Hand-E fingers of a BASE grasp."""
    from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry

    cfg = _cell()
    turn = np.column_stack([grasp.axis, np.cross(grasp.approach, grasp.axis), grasp.approach])
    local = (np.asarray(points, dtype=np.float64) - grasp.position) @ turn
    return int(sum(int(box.contains_local_points(local).sum())
                   for box in build_gripper_geometry(cfg.grasping.gripper_geometry).collision_boxes(
                       float(cfg.gripper.max_width_mm)) if box.label.startswith("finger")))


class ARecordedZollstockLookIsBoxedInTests(unittest.TestCase):
    """Track A test 8."""

    def test_today_three_grasps_each_with_an_open_finger_in_a_neighbour(self) -> None:
        from tests._cell_2026_10_01 import look

        for name in ("P1", "P5"):
            with self.subTest(look=name):
                recorded = look(name)
                today = _compute(recorded, scene=False)
                # Three until 2026-10-05, six since SFE stands its fingers at the anchor and tries the part's middle.
                self.assertGreaterEqual(len(today.candidates), 3)
                # What the camera world keeps beside the part: every one of today's grasps has an open finger in it.
                world = _compute(recorded, scene=True, model=_model(recorded)).metadata[
                    "scene_points_world_rule_base_mm"]
                held = [_in_open_fingers(grasp, world) for grasp in today.candidates]
                self.assertTrue(all(n > 0 for n in held), held)

    def test_with_the_scene_none_and_it_says_a_neighbour_is_in_the_way(self) -> None:
        from src.robot.grasping.generation.scene_obstacles import no_grasp_said
        from src.robot.grasping.types.feedback import GraspFailureReason
        from tests._cell_2026_10_01 import look

        for name in ("P1", "P5"):
            recorded = look(name)
            for label, model in (("without the support model", None), ("with it", _model(recorded))):
                with self.subTest(look=name, model=label):
                    after = _compute(recorded, scene=True, model=model)
                    self.assertEqual((), after.candidates)
                    self.assertEqual((GraspFailureReason.ALL_COLLIDED, GraspFailureReason.RESCAN_RECOMMENDED),
                                     after.reasons)
                    self.assertGreater(after.telemetry["support_footprint_refused"]["seen_corridor"], 0)
                    self.assertIn("meets a neighbour the camera saw beside the part", no_grasp_said(after))

    def test_with_the_model_the_mat_is_no_neighbour(self) -> None:
        """The support model takes the mat out of what the push is handed: the world rule keeps the pile, not the mat
        (51 000-52 000 points without it, 7 000-8 000 with it on the full-resolution looks)."""
        from tests._cell_2026_10_01 import look

        for name in ("P1", "P5"):
            with self.subTest(look=name):
                recorded = look(name)
                bare = _compute(recorded, scene=True).telemetry["scene_obstacle_world_rule_points"]
                held = _compute(recorded, scene=True, model=_model(recorded))
                world = held.metadata["scene_points_world_rule_base_mm"]
                self.assertLess(world.shape[0], bare / 3)
                self.assertGreater(float(np.median(world[:, 2])), 70.0)


if __name__ == "__main__":
    unittest.main()
