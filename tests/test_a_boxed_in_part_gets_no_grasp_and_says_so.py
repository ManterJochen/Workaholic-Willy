"""A part its neighbours box in gets no grasp from the calculator, and the calculator says so.

The owner's cell, 2026-10-01 (fix plan RC2): every prompt grounded one box, so the calculator saw the Zollstock alone and
offered grasps whose open fingers stood inside the parts beside it; the exact guard refused every one, and the pick
failed with no reason a person could read. With ``robot.grasping.scene_obstacles`` the calculator reads the depth around
the part by the planner world's rules (``generation/scene_obstacles.py``): a part boxed in gets no candidate, the reason
is ``ALL_COLLIDED`` (the one a push answers) and a sentence says a neighbour the camera saw is in the way. A neighbour on
one side leaves the closing axis that keeps clear of it. The locator's door (``Located.scene``, example 13) says the same.

The scenes are ray-cast by a pinhole looking down at about 20 degrees, as the owner's D415 looks at the mat: a 55 mm
slab on the bench, the part on it, neighbours beside it. Only the part is segmented, as on the owner's cell.
"""

from __future__ import annotations

import unittest
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.geometry import Frame, Transform
from tests.test_a_side_grasp_never_goes_through_unseen_space import Box, Cylinder, DepthCamera, Plane

#: The slab the parts stand on, as the owner's mat: 55 mm over the bench.
MAT_MM = 55.0
MAT = Box(lo=(-300.0, -950.0, 0.0), hi=(250.0, -450.0, MAT_MM))
BENCH = Plane(0.0)
#: The camera: about 20 degrees off vertical, 600 mm from the parts, a D415's lens at 640 x 480.
EYE = (0.0, -450.0, 650.0)
AIM = (0.0, -650.0, 70.0)
WIDTH, HEIGHT, FOCAL = 640, 480, 560.0


def hande_cell() -> Any:
    """The shipped ``hande`` profile: the owner's Hand-E, built the way that tree builds it."""
    from src.config.loader import load_robot_section

    return load_robot_section(profile="hande")


@lru_cache(maxsize=None)
def _hande() -> Any:
    return hande_cell()


def owner_rules(**changes: Any) -> Any:
    """The planner world's rules on the owner's cell: the bench declared at 0 with its 5 mm band, the world read at every
    second pixel, 25 mm voxels of 12 points, the guard's 5 mm from a camera box and from a declared one."""
    from src.robot.grasping.generation.scene_obstacles import SceneObstacleRules

    values: dict[str, Any] = dict(bench_top_mm=0.0, band_mm=5.0, min_points=12, cluster_voxel_mm=25.0, pixel_stride=2,
                                  support_distance_mm=5.0, declared_distance_mm=5.0)
    values.update(changes)
    return SceneObstacleRules(**values)


class Frame_:
    """One ray-cast frame of a scene, with the part's mask: what the pick loop hands the calculator."""

    def __init__(self, solids: tuple[Any, ...], target: Any) -> None:
        self.camera = DepthCamera(eye=EYE, target=AIM, solids=(*solids, target), width=WIDTH, height=HEIGHT,
                                  focal_px=FOCAL)
        without = DepthCamera(eye=EYE, target=AIM, solids=solids, width=WIDTH, height=HEIGHT, focal_px=FOCAL).depth
        depth = self.camera.depth
        self.mask = np.isfinite(depth) & (~np.isfinite(without) | (depth < without - 0.5))
        self.depth = np.where(np.isfinite(depth), depth, 0.0)
        self.intrinsics = np.array([[FOCAL, 0.0, WIDTH / 2.0], [0.0, FOCAL, HEIGHT / 2.0], [0.0, 0.0, 1.0]])
        self.camera_to_base = np.eye(4)
        self.camera_to_base[:3, :3] = self.camera.rotation
        self.camera_to_base[:3, 3] = EYE

    def target_points(self, mask: "np.ndarray | None" = None) -> np.ndarray:
        """The part's measured surface, BASE millimetres."""
        rows, cols = np.nonzero(self.mask if mask is None else mask)
        z = self.depth[rows, cols]
        camera = np.column_stack(((cols - WIDTH / 2.0) * z / FOCAL, (rows - HEIGHT / 2.0) * z / FOCAL, z))
        return camera @ self.camera_to_base[:3, :3].T + self.camera_to_base[:3, 3]


def calculator(*, scene: bool, rules: Any = None, side: bool = True, cell: Any = None) -> Any:
    """The cell's calculator over the Hand-E, as ``cells.py`` builds it, with the scene's obstacles on or off."""
    from src.robot.grasping.calculator_factory import build_calculator

    cfg = cell if cell is not None else _hande()
    extra: dict[str, Any] = {"side_approaches": side}
    if scene:
        extra["scene_obstacles"] = rules if rules is not None else owner_rules()
    return build_calculator(cfg, camera_matrix=np.array([[FOCAL, 0.0, WIDTH / 2.0], [0.0, FOCAL, HEIGHT / 2.0],
                                                         [0.0, 0.0, 1.0]]),
                            max_grip_width_mm=cfg.gripper.max_width_mm, min_grip_width_mm=cfg.gripper.min_width_mm,
                            support_footprint_geometry=True, support_footprint_inflate_mm=0.0, **extra)


def compute(calc: Any, frame: Frame_, *, support_mm: float = MAT_MM, mask: "np.ndarray | None" = None,
            **kwargs: Any) -> Any:
    """``compute_result`` as the pick loop calls it: the part's mask, the frame's depth, the BASE support it resolved."""
    from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry
    from src.robot.grasping.collision import SupportPlane

    cfg = _hande()
    seg = SimpleNamespace(mask=frame.mask if mask is None else mask, label="part", score=1.0)
    return calc.compute_result(
        seg, frame.depth, pixel_to_mm=None, dense_sampling=False, other_object_masks=[],
        camera_to_base=Transform.from_matrix(frame.camera_to_base, from_frame=Frame.CAMERA, to_frame=Frame.BASE),
        support_plane=SupportPlane(normal=np.array([0.0, 0.0, 1.0]), offset_mm=float(support_mm), frame=Frame.BASE),
        min_table_clearance_mm=float(cfg.grasping.support.min_clearance_mm),
        gripper_model=build_gripper_geometry(cfg.grasping.gripper_geometry), **kwargs)


# --- the scenes -------------------------------------------------------------------------------------------------------

#: A 20 mm wide, 40 mm tall bar, 160 mm long, on the slab.
BAR = Box(lo=(-80.0, -660.0, MAT_MM), hi=(80.0, -640.0, MAT_MM + 40.0))


def boxed_in_bar(gap_mm: float = 3.0) -> Frame_:
    """The bar between 100 mm tall cylinders, 50 mm across, ``gap_mm`` off its two long sides along its length."""
    radius = 25.0
    pillars = tuple(Cylinder((x, y), radius, MAT_MM, MAT_MM + 100.0)
                    for x in (-60.0, 0.0, 60.0)
                    for y in (-640.0 + gap_mm + radius, -660.0 - gap_mm - radius))
    return Frame_((BENCH, MAT, *pillars), BAR)


#: A 40 mm cylinder, 50 mm tall, on the slab.
PART = Cylinder((0.0, -650.0), 20.0, MAT_MM, MAT_MM + 50.0)


def cubes_on_x(gap_mm: float = 20.0, side_mm: float = 40.0) -> Frame_:
    """The cylinder with a cube ``gap_mm`` off it on each side along x."""
    off = 20.0 + gap_mm + side_mm / 2.0
    cubes = tuple(Box(lo=(sx * off - side_mm / 2.0, -650.0 - side_mm / 2.0, MAT_MM),
                      hi=(sx * off + side_mm / 2.0, -650.0 + side_mm / 2.0, MAT_MM + side_mm))
                  for sx in (-1.0, 1.0))
    return Frame_((BENCH, MAT, *cubes), PART)


def open_fingers_hold(grasp: Any, points: np.ndarray) -> int:
    """How many of ``points`` stand inside the open Hand-E fingers of a BASE grasp."""
    from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry

    cfg = _hande()
    hand = build_gripper_geometry(cfg.grasping.gripper_geometry)
    turn = np.column_stack([grasp.axis, np.cross(grasp.approach, grasp.axis), grasp.approach])
    local = (np.asarray(points, dtype=np.float64) - grasp.position) @ turn
    return int(sum(int(box.contains_local_points(local).sum())
                   for box in hand.collision_boxes(float(cfg.gripper.max_width_mm)) if box.label.startswith("finger")))


def solid_points(solids: tuple[Any, ...], step_mm: float = 3.0) -> np.ndarray:
    """Points filling the neighbours, BASE millimetres: what an open finger inside one would hold."""
    out = []
    for solid in solids:
        if isinstance(solid, Cylinder):
            for r in np.arange(0.0, solid.radius_mm + 0.1, step_mm):
                for a in np.arange(0.0, 2.0 * np.pi, step_mm / max(r, step_mm)):
                    for z in np.arange(solid.z0_mm, solid.z1_mm + 0.1, step_mm):
                        out.append((solid.centre_xy[0] + r * np.cos(a), solid.centre_xy[1] + r * np.sin(a), z))
        elif isinstance(solid, Box):
            axes = [np.arange(lo, hi + 0.1, step_mm) for lo, hi in zip(solid.lo, solid.hi)]
            out.extend((x, y, z) for x in axes[0] for y in axes[1] for z in axes[2])
    return np.asarray(out, dtype=np.float64)


class ABoxedInPartGetsNoGraspTests(unittest.TestCase):
    """The plan's Track A test 1: a bar boxed in by its neighbours, only the bar segmented."""

    frame: Frame_
    today: Any
    after: Any

    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = boxed_in_bar()
        cls.today = compute(calculator(scene=False), cls.frame)
        cls.after = compute(calculator(scene=True), cls.frame)

    def test_today_it_offers_grasps_whose_open_fingers_stand_in_a_neighbour(self) -> None:
        """What the owner's cell did: candidates, and an open finger inside a pillar."""
        self.assertGreater(len(self.today.candidates), 0)
        pillars = tuple(s for s in self.frame.camera.solids if isinstance(s, Cylinder))
        inside = [open_fingers_hold(g, solid_points(pillars)) for g in self.today.candidates]
        self.assertTrue(any(n > 0 for n in inside), inside)

    def test_with_the_scene_it_offers_none_and_says_a_neighbour_is_in_the_way(self) -> None:
        from src.robot.grasping.generation.scene_obstacles import no_grasp_said
        from src.robot.grasping.types.feedback import GraspFailureReason

        self.assertEqual(0, len(self.after.candidates))
        self.assertEqual(GraspFailureReason.ALL_COLLIDED, self.after.reasons[0])
        self.assertIn(GraspFailureReason.RESCAN_RECOMMENDED, self.after.reasons)
        self.assertNotIn(GraspFailureReason.ALL_TABLE_CONFLICT, self.after.reasons)
        self.assertGreater(self.after.telemetry["scene_obstacle_points"], 0)
        self.assertGreater(self.after.telemetry["support_footprint_refused"]["seen_corridor"], 0)
        self.assertEqual("scene", self.after.telemetry["candidates_final_source"])
        self.assertIn("meets a neighbour the camera saw beside the part", no_grasp_said(self.after))

    def test_the_push_gets_the_evidence_both_ways(self) -> None:
        """Contract 2: A's points and the world rule's, BASE, on the result's metadata."""
        kept = self.after.metadata["scene_obstacle_points_base_mm"]
        world = self.after.metadata["scene_points_world_rule_base_mm"]
        self.assertEqual((self.after.telemetry["scene_obstacle_points"], 3), kept.shape)
        self.assertGreaterEqual(world.shape[0], kept.shape[0])
        # Everything kept stands over the slab's band, beside the bar, within the reach.
        self.assertTrue(np.all(kept[:, 2] > MAT_MM + 5.0))
        self.assertTrue(np.all(np.hypot(kept[:, 0], kept[:, 1] + 650.0) < 250.0))

    def test_a_wider_gap_lets_the_bar_be_gripped_again(self) -> None:
        """The control: the same pillars 40 mm off, and the scene leaves the bar its grasps."""
        roomy = compute(calculator(scene=True), boxed_in_bar(gap_mm=40.0))
        self.assertGreater(len(roomy.candidates), 0)


class ANeighbourOnOneSideTests(unittest.TestCase):
    """Track A test 3: cubes 20 mm off on both sides along x leave the grasps that close along y."""

    def test_only_the_free_closing_axis_survives(self) -> None:
        frame = cubes_on_x()
        today = compute(calculator(scene=False), frame)
        after = compute(calculator(scene=True), frame)
        along_x = [abs(float(g.axis[0])) for g in today.candidates]
        self.assertTrue(any(v > 0.9 for v in along_x), along_x)          # today one closes towards the cubes
        self.assertGreater(len(after.candidates), 0)
        self.assertTrue(all(abs(float(g.axis[0])) < 0.5 for g in after.candidates),
                        [np.round(g.axis, 2).tolist() for g in after.candidates])


class TheLocatorSaysItTooTests(unittest.TestCase):
    """Track A test 6: ``Located.scene(0, cfg).grasps()`` on the boxed-in bar: no grasp, and why."""

    def _located(self, frame: Frame_) -> Any:
        from src.robot.perception.locator import Located, LocatedObject, _Frame

        part = LocatedObject(label="bar", score=1.0, box_px=None, mask=frame.mask,
                             points_base_mm=frame.target_points(), centre_mm=None)
        return Located(camera="cam", captured_at_s=0.0, mounting="eye_to_hand", tool_to_base_mm=None, objects=(part,),
                       _frame=_Frame(depth_mm=frame.depth, intrinsics=frame.intrinsics,
                                     camera_to_base=frame.camera_to_base))

    def _cell(self) -> Any:
        """The Hand-E profile with the owner's bench and planning world on, so the scene reads the support it stands on."""
        from src.config.schema.robot.safety_schema import SupportPlaneConfig

        cfg = _hande()
        world = cfg.safety.planning_world.model_copy(update={
            "enabled": True,
            "support_plane": SupportPlaneConfig(height_mm=0.0, extent_mm=(900.0, 700.0), thickness_mm=50.0,
                                                center_mm=(-10.0, -585.0))})
        safety = cfg.safety.model_copy(update={"planning_world": world})
        limits = cfg.workspace_limits.model_copy(update={"x_min": -420.0, "x_max": 400.0, "y_min": -900.0,
                                                         "y_max": -270.0})
        return cfg.model_copy(update={"safety": safety, "workspace_limits": limits})

    def test_the_scene_has_no_grasp_and_says_a_neighbour_is_in_the_way(self) -> None:
        cell = self._cell()
        grasps = self._located(boxed_in_bar()).scene(0, cell).grasps()
        self.assertEqual((), grasps.candidates)
        self.assertGreater(grasps.refused_by_obstacles, 0)
        self.assertIn("meets a neighbour the camera saw beside the part", grasps.said)
        self.assertIn("meets a neighbour", grasps.render())
        self.assertEqual(grasps.said, grasps.to_dict()["said"])

    def test_with_the_scene_switched_off_the_bar_gets_its_grasps_again(self) -> None:
        cell = self._cell()
        off = cell.model_copy(update={"grasping": cell.grasping.model_copy(update={"scene_obstacles": False})})
        grasps = self._located(boxed_in_bar()).scene(0, off).grasps()
        self.assertGreater(len(grasps.candidates), 0)
        self.assertEqual(0, grasps.refused_by_obstacles)

    def test_room_beside_the_bar_leaves_its_grasps(self) -> None:
        grasps = self._located(boxed_in_bar(gap_mm=40.0)).scene(0, self._cell()).grasps()
        self.assertGreater(len(grasps.candidates), 0)


if __name__ == "__main__":
    unittest.main()
