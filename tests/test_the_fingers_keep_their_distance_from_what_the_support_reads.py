"""The fingers keep the guard's distance from what a support surface reads, not from the solid held over it (the owner,
2026-10-05: "wir müssen tiefer gehen").

A support's solid stands over the surface's reading by what the reading stands over its plane, the band and the
allowance; the band and the allowance keep the guard's distance from a thin thing the band took for the surface. With
``perceived.fingers_to_the_support_reading`` the Hand-E's fingers keep the distance from the solid's top less its band
and its allowance (``PerceivedBox.finger_top_mm``), and every other link from the whole solid. On the grasp bench the
solid stood about 7 mm over the mat's reading and the fingertips stayed 10 mm over the mat; a D415 on a real cell reads a
wider band, and its grasps stood at the parts' tops. Pinned:

* the exact guard lets a finger come the band and the allowance nearer the solid's top, and no nearer;
* the camera world hands the guard its support solids with the band and the allowance, only where the switch is on;
* the calculator's floor and its filter hold the fingers so, the palm to the whole solid;
* the switch is on in the shipped config and reaches the world and the calculator.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.robot.safety._capsule import AxisAlignedBox, TurnedBox
from src.robot.safety._ur_kinematics import ur_link_transforms_mm
from tests.test_a_bin_the_camera_saw_beside_the_base_is_the_exact_guards import AWAY, _arm, _backend

GUARD_MM = 3.0
#: The band and the allowance a support's solid stands over its reading by, here.
DROP_MM = 5.0


class TheExactGuardHoldsAFingerToTheReadingTests(unittest.TestCase):
    """A support's solid square to the Hand-E's approach, marched up under the fingertips on the owner-like UR10."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = _backend(_arm())
        cls.transforms = ur_link_transforms_mm("ur10", np.asarray(AWAY, dtype=np.float64))
        flange = np.asarray(cls.transforms[-1], dtype=np.float64)
        cls.approach = flange[:3, 2]
        cls.tcp = flange[:3, :3] @ np.array([0.0, 0.0, 155.75]) + flange[:3, 3]
        # The solid's own axes: its up against the approach, so its top faces the fingertips.
        up = -cls.approach
        side = np.cross(up, flange[:3, 0])
        side /= np.linalg.norm(side)
        cls.turn = np.column_stack([np.cross(side, up), side, up])

    @classmethod
    def _solid(cls, top_at_mm: float, drop_mm: float) -> AxisAlignedBox:
        """A 300 x 300 x 40 mm solid whose top stands ``top_at_mm`` past the TCP along the approach."""
        half = np.array([150.0, 150.0, 20.0])
        centre = cls.tcp + cls.approach * (top_at_mm + half[2])
        enclosing = np.abs(cls.turn) @ half
        return AxisAlignedBox(center_mm=centre, half_extents_mm=enclosing, name="seen_s00_support0",
                              turned=TurnedBox(half_extents_mm=half, yaw_rad=0.0, rotation=cls.turn),
                              finger_top_mm=drop_mm)

    @classmethod
    def _first(cls, drop_mm: float) -> "tuple[float, str]":
        """Marching the solid's top in toward the fingertips: where it is first refused, and the pair that refuses it."""
        for at in np.arange(60.0, -30.0, -0.25):
            hit = cls.backend.evaluate(cls.transforms, 0.0, (cls._solid(float(at), drop_mm),), GUARD_MM,
                                       arm_pairs=False)
            if hit is not None:
                return float(at), str(hit[0])
        raise AssertionError("the solid never met the hand")

    def test_a_finger_comes_the_band_and_the_allowance_nearer_and_no_nearer(self) -> None:
        whole, by = self._first(0.0)
        self.assertIn("finger", by, "the control: the fingertips meet a solid under them first")
        read, by_read = self._first(DROP_MM)
        self.assertIn("finger", by_read)
        self.assertAlmostEqual(whole - read, DROP_MM, delta=0.3)

    def test_the_housing_keeps_the_whole_solid(self) -> None:
        # Raised past the fingertips the solid meets the rest of the hand, which keeps the whole solid: a lowered top
        # for the fingers moves no other link's refusal.
        for at in np.arange(-40.0, -160.0, -0.5):
            whole = self.backend.evaluate(self.transforms, 0.0, (self._solid(float(at), 0.0),), GUARD_MM,
                                          arm_pairs=False)
            read = self.backend.evaluate(self.transforms, 0.0, (self._solid(float(at), DROP_MM),), GUARD_MM,
                                         arm_pairs=False)
            if whole is not None and "finger" not in whole[0]:
                self.assertIsNotNone(read)
                return
        self.skipTest("no height met the housing before the fingers at this pose")


class TheCameraWorldHandsTheGuardTheReadingTests(unittest.TestCase):
    def _boxes(self, on: bool) -> list[Any]:
        from src.robot.safety.planning.live_world import _guard_boxes
        from src.robot.safety.planning.perceived import _solid_boxes

        solid = SimpleNamespace(
            name="seen_s00_support0", kind="support", surface=0, centre_mm=np.array([0.0, -650.0, 40.0]),
            half_extents_mm=np.array([200.0, 150.0, 20.0]), rotation=np.eye(3), excess_mm=1.5, band_mm=3.0,
            allowance_mm=2.0)
        model = SimpleNamespace(solids=(solid,), surfaces=())
        boxes = _solid_boxes(model, np.zeros(3), fingers_to_the_reading=on)  # type: ignore[arg-type]
        return list(_guard_boxes(SimpleNamespace(boxes=boxes)))  # type: ignore[arg-type]

    def test_on_a_support_solid_holds_the_fingers_its_band_and_allowance_under_its_top(self) -> None:
        (box,) = self._boxes(True)
        self.assertAlmostEqual(5.0, box.finger_top_mm)

    def test_off_it_holds_them_to_the_whole_solid(self) -> None:
        (box,) = self._boxes(False)
        self.assertEqual(0.0, box.finger_top_mm)


class TheCalculatorPlansTheFingersOverTheReadingTests(unittest.TestCase):
    def _solid(self) -> Any:
        from src.robot.safety.planning.support_surfaces import SupportSolid

        return SupportSolid(name="seen_s00_support0", kind="support", surface=0,
                            centre_mm=np.array([0.0, -650.0, 63.0 - 20.0]), half_extents_mm=np.array([300.0, 300.0, 20.0]),
                            rotation=np.eye(3), local_plane=np.array([56.0, 0.0, 0.0]), excess_mm=2.0, band_mm=3.0,
                            allowance_mm=2.0)

    def test_the_floor_is_the_reading_for_a_finger_and_the_solid_for_the_palm(self) -> None:
        from src.robot.grasping.generation.support_footprint import HandFloor

        floor = HandFloor(solids=(self._solid(),), distance_mm=GUARD_MM, fingers_to_the_reading=True)
        xy = np.array([[0.0, -650.0]])
        self.assertAlmostEqual(float(floor.at(xy)[0]), 63.0 + GUARD_MM)
        self.assertAlmostEqual(float(floor.at(xy, fingers=True)[0]), 63.0 - 5.0 + GUARD_MM)
        off = HandFloor(solids=(self._solid(),), distance_mm=GUARD_MM)
        self.assertAlmostEqual(float(off.at(xy, fingers=True)[0]), 63.0 + GUARD_MM)

    def test_sfe_plans_a_grasp_lower_where_the_fingers_go_to_the_reading(self) -> None:
        from src.robot.grasping.generation.support_footprint import HandFloor, generate_support_footprint_grasps
        from tests.test_a_grasp_keeps_the_guards_distance_from_the_support_it_holds import _box_cloud, _jaw

        cloud = _box_cloud(30.0)
        held = generate_support_footprint_grasps(cloud, support_height_mm=56.0, jaw=_jaw(), max_candidates=40,
                                                 hand_floor=HandFloor((self._solid(),), GUARD_MM))
        read = generate_support_footprint_grasps(cloud, support_height_mm=56.0, jaw=_jaw(), max_candidates=40,
                                                 hand_floor=HandFloor((self._solid(),), GUARD_MM,
                                                                      fingers_to_the_reading=True))
        self.assertTrue(held and read)
        lowest = min(float(g.position_mm[2]) for g in read)
        self.assertLess(lowest, min(float(g.position_mm[2]) for g in held) - 4.0)
        for grasp in read:
            self.assertGreaterEqual(56.0 + grasp.clearance_mm, 58.0 + GUARD_MM - 1e-6)

    def test_the_filter_holds_a_finger_to_the_reading_and_the_palm_to_the_solid(self) -> None:
        from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry
        from src.robot.grasping.generation.scene_obstacles import envelope_verdicts
        from tests.test_a_boxed_in_part_gets_no_grasp_and_says_so import hande_cell

        cfg = hande_cell()
        hand = build_gripper_geometry(cfg.grasping.gripper_geometry)
        tip = float(cfg.grasping.gripper_geometry.parallel_jaw.fingertip_depth_mm)
        down = np.column_stack([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        # The fingertips 4 mm over the reading and its excess, 1 mm into the solid's band.
        pose = (np.array([0.0, -650.0, 58.0 + 4.0 + tip]), down)
        held = envelope_verdicts([pose], gripper_model=hand, open_width_mm=float(cfg.gripper.max_width_mm),
                                 solids=(self._solid(),), support_distance_mm=GUARD_MM)[0]
        read = envelope_verdicts([pose], gripper_model=hand, open_width_mm=float(cfg.gripper.max_width_mm),
                                 solids=(self._solid(),), support_distance_mm=GUARD_MM, fingers_to_the_reading=True)[0]
        self.assertEqual("support", held.refused)
        self.assertEqual("", read.refused)
        self.assertAlmostEqual(read.distance_mm, 4.0, delta=1e-6)


class TheSwitchIsOnInTheShippedConfigTests(unittest.TestCase):
    def test_the_schema_turns_it_on_and_the_calculator_reads_it(self) -> None:
        from src.config.schema.robot.safety_schema import PerceivedWorldConfig
        from src.robot.grasping.generation.scene_obstacles import SceneObstacleRules
        from tests.test_a_boxed_in_part_gets_no_grasp_and_says_so import hande_cell

        self.assertTrue(PerceivedWorldConfig().fingers_to_the_support_reading)
        self.assertTrue(SceneObstacleRules.from_robot_config(hande_cell()).fingers_to_the_reading)



class TheFingersComeTo1mmOfTheReadingTests(unittest.TestCase):
    """The owner, 2026-10-06: "man kann ja richtig nah an die Matte bzw den Boden gehen, gerne bis auf 1mm solange er
    diesen nicht berührt". The guard keeps its 3 mm from a support's solid lowered for the fingers by its band, its
    allowance and the 2 mm the finger floor lies under the guard's distance: 1 mm over the reading and its excess."""

    def test_the_shipped_config_lowers_the_fingers_floor_by_the_guards_distance_less_1_mm(self) -> None:
        from src.config.schema.robot.safety_schema import PerceivedWorldConfig
        from src.robot.grasping.generation.scene_obstacles import SceneObstacleRules, finger_floor_drop_mm
        from tests.test_a_boxed_in_part_gets_no_grasp_and_says_so import hande_cell

        self.assertEqual(1.0, PerceivedWorldConfig().finger_floor_mm)
        cfg = hande_cell()
        guard = float(cfg.safety.self_collision.perceived_min_distance_mm)
        self.assertAlmostEqual(max(0.0, guard - 1.0), finger_floor_drop_mm(cfg))
        self.assertAlmostEqual(finger_floor_drop_mm(cfg), SceneObstacleRules.from_robot_config(cfg).finger_floor_drop_mm)

    def test_no_drop_where_the_fingers_keep_to_the_whole_solid_or_the_floor_is_no_nearer(self) -> None:
        from src.robot.grasping.generation.scene_obstacles import finger_floor_drop_mm

        def cell(on: bool, floor: float) -> SimpleNamespace:
            return SimpleNamespace(safety=SimpleNamespace(
                self_collision=SimpleNamespace(perceived_min_distance_mm=3.0),
                planning_world=SimpleNamespace(perceived=SimpleNamespace(fingers_to_the_support_reading=on,
                                                                         finger_floor_mm=floor))))

        self.assertAlmostEqual(2.0, finger_floor_drop_mm(cell(True, 1.0)))
        self.assertEqual(0.0, finger_floor_drop_mm(cell(False, 1.0)))
        self.assertEqual(0.0, finger_floor_drop_mm(cell(True, 5.0)))

    def test_the_world_and_the_calculator_hold_a_finger_1_mm_over_the_reading(self) -> None:
        from src.robot.grasping.generation.support_footprint import HandFloor
        from src.robot.safety.planning.perceived import _solid_boxes

        solid = TheCalculatorPlansTheFingersOverTheReadingTests()._solid()
        floor = HandFloor(solids=(solid,), distance_mm=GUARD_MM, fingers_to_the_reading=True, finger_floor_drop_mm=2.0)
        xy = np.array([[0.0, -650.0]])
        # The solid's top 63 mm, its band and allowance 5: a finger 1 mm over 58, the palm the guard's 3 over 63.
        self.assertAlmostEqual(float(floor.at(xy, fingers=True)[0]), 58.0 + 1.0)
        self.assertAlmostEqual(float(floor.at(xy)[0]), 63.0 + GUARD_MM)
        model = SimpleNamespace(solids=(solid,), surfaces=())
        (box,) = _solid_boxes(model, np.zeros(3), fingers_to_the_reading=True,  # type: ignore[arg-type]
                              finger_floor_drop_mm=2.0)
        self.assertAlmostEqual(5.0 + 2.0, box.finger_top_mm)

    def test_the_exact_guard_lets_a_finger_2_mm_nearer_the_solid(self) -> None:
        case = TheExactGuardHoldsAFingerToTheReadingTests
        case.setUpClass()
        reading, _ = case._first(DROP_MM)
        floor, by = case._first(DROP_MM + 2.0)
        self.assertIn("finger", by)
        self.assertAlmostEqual(reading - floor, 2.0, delta=0.3)

if __name__ == "__main__":
    unittest.main()
