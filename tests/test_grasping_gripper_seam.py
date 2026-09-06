"""Tests for the gripper-geometry seam: the strategy contract, the two shipped models, and the config factory."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from src.config.schema.robot.grasping_schema import GraspingGripperGeometryConfig
from src.geometry import Frame
from src.robot.grasping import (
    CollisionBox,
    GripperGeometryStrategy,
    ParallelJawGripperModel,
    SuctionCupGripperModel,
)
from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry
from src.robot.grasping.collision import validate_grasp_collision
from src.robot.grasping.generation.calculator import GraspCalculator
from src.robot.grasping.planning import GraspPose

_IDENTITY_ROT = np.eye(3, dtype=np.float64)


def _grasp(position_mm=(0.0, 0.0, 0.0), grip_width_mm: float = 40.0) -> GraspPose:
    return GraspPose(
        position_mm=np.asarray(position_mm, dtype=np.float64),
        rotation_matrix=_IDENTITY_ROT.copy(),
        grip_width_mm=grip_width_mm,
        score=0.5,
        confidence=0.5,
        contacts=(
            np.array([-grip_width_mm / 2.0, 0.0, 0.0]),
            np.array([+grip_width_mm / 2.0, 0.0, 0.0]),
        ),
        frame=Frame.BASE,
        metadata={},
    )


@dataclass(frozen=True, slots=True)
class _ThreeFingerStubModel:
    """Minimal alternate strategy: three thin finger boxes around the closing axis. Proves the seam accepts any conforming object."""

    finger_length_mm: float = 50.0
    finger_thickness_mm: float = 8.0

    def collision_boxes(self, grip_width_mm: float) -> tuple[CollisionBox, ...]:
        half_gap = 0.5 * float(grip_width_mm)
        boxes = []
        for angle_deg in (0.0, 120.0, 240.0):
            theta = np.deg2rad(angle_deg)
            cx = float(half_gap * np.cos(theta))
            cy = float(half_gap * np.sin(theta))
            t = self.finger_thickness_mm
            boxes.append(
                CollisionBox(
                    f"finger_{int(angle_deg):03d}",
                    [cx - t, cy - t, -self.finger_length_mm],
                    [cx + t, cy + t, 5.0],
                )
            )
        return tuple(boxes)

    def local_corners_mm(self, grip_width_mm: float) -> np.ndarray:
        return np.vstack([box.corners_local_mm() for box in self.collision_boxes(grip_width_mm)])


class GripperGeometryStrategyProtocolTests(unittest.TestCase):
    def test_shipped_models_conform_to_protocol(self) -> None:
        self.assertIsInstance(ParallelJawGripperModel(), GripperGeometryStrategy)
        self.assertIsInstance(SuctionCupGripperModel(), GripperGeometryStrategy)
        self.assertIsInstance(_ThreeFingerStubModel(), GripperGeometryStrategy)

    def test_alternate_strategy_produces_boxes_and_corners(self) -> None:
        model = _ThreeFingerStubModel()
        self.assertEqual(len(model.collision_boxes(40.0)), 3)
        self.assertEqual(model.local_corners_mm(40.0).shape, (24, 3))

    def test_protocol_rejects_incomplete_object(self) -> None:
        class _NoBoxes:
            def local_corners_mm(self, _w: float) -> np.ndarray:
                return np.zeros((0, 3))

        class _NoCorners:
            def collision_boxes(self, _w: float) -> tuple[CollisionBox, ...]:
                return ()

        self.assertNotIsInstance(_NoBoxes(), GripperGeometryStrategy)
        self.assertNotIsInstance(_NoCorners(), GripperGeometryStrategy)

    def test_parallel_jaw_baseline_unchanged(self) -> None:
        # Snapshot: box count + labels for the default jaw must stay stable (the seam is additive only).
        model = ParallelJawGripperModel()
        labels = tuple(b.label for b in model.collision_boxes(80.0))
        self.assertEqual(labels, ("finger_negative_x", "finger_positive_x", "palm"))


class SuctionCupGripperModelTests(unittest.TestCase):
    def test_boxes_labels_and_corner_shape(self) -> None:
        model = SuctionCupGripperModel()
        labels = tuple(b.label for b in model.collision_boxes(0.0))
        self.assertEqual(labels, ("suction_cup", "suction_shaft", "suction_mount"))
        self.assertEqual(model.local_corners_mm(0.0).shape, (24, 3))

    def test_grip_width_is_ignored(self) -> None:
        model = SuctionCupGripperModel()
        wide = model.local_corners_mm(120.0)
        narrow = model.local_corners_mm(0.0)
        self.assertTrue(np.array_equal(wide, narrow))

    def test_boxes_tile_back_along_negative_approach(self) -> None:
        # cup contacts near z=0; shaft then mount extend to more-negative z (away from the object).
        cup, shaft, mount = SuctionCupGripperModel().collision_boxes(0.0)
        self.assertGreater(cup.max_corner_mm[2], shaft.max_corner_mm[2])
        self.assertGreater(shaft.max_corner_mm[2], mount.max_corner_mm[2])
        for box in (cup, shaft, mount):
            self.assertTrue(np.all(box.min_corner_mm < box.max_corner_mm))

    def test_outer_margin_inflates_every_box(self) -> None:
        base = SuctionCupGripperModel(outer_margin_mm=0.0).collision_boxes(0.0)
        inflated = SuctionCupGripperModel(outer_margin_mm=5.0).collision_boxes(0.0)
        for b0, b1 in zip(base, inflated):
            self.assertTrue(np.all(b1.min_corner_mm <= b0.min_corner_mm))
            self.assertTrue(np.all(b1.max_corner_mm >= b0.max_corner_mm))

    def test_invalid_dimensions_raise(self) -> None:
        for bad in (
            {"cup_radius_mm": 0.0},
            {"shaft_length_mm": -1.0},
            {"mount_radius_mm": float("nan")},
        ):
            with self.assertRaises(ValueError):
                SuctionCupGripperModel(**bad)

    def test_flows_through_the_checker_and_differs_from_the_jaw(self) -> None:
        # A point off to the side sits inside the wide 2F-85 palm but outside the slim suction cup:
        # the two envelopes must give different verdicts through the shared checker.
        pose = _grasp()
        side_point = np.array([[28.0, 0.0, -30.0]])  # +X of the closing axis, behind the tip
        jaw = validate_grasp_collision(pose, scene_points_mm=side_point, gripper_model=ParallelJawGripperModel())
        cup = validate_grasp_collision(pose, scene_points_mm=side_point, gripper_model=SuctionCupGripperModel())
        self.assertFalse(jaw.valid)  # inside the jaw palm/finger envelope
        self.assertTrue(cup.valid)   # outside the slim cup/shaft envelope


class BuildGripperGeometryFactoryTests(unittest.TestCase):
    def test_default_config_reproduces_hardcoded_jaw_byte_identical(self) -> None:
        model = build_gripper_geometry(GraspingGripperGeometryConfig())
        self.assertIsInstance(model, ParallelJawGripperModel)
        self.assertEqual(model, ParallelJawGripperModel())

    def test_suction_kind_builds_suction_model_with_config_dims(self) -> None:
        cfg = GraspingGripperGeometryConfig(kind="suction", outer_margin_mm=1.5)
        cfg = cfg.model_copy(update={"suction": cfg.suction.model_copy(update={"cup_radius_mm": 12.0})})
        model = build_gripper_geometry(cfg)
        self.assertIsInstance(model, SuctionCupGripperModel)
        assert isinstance(model, SuctionCupGripperModel)  # narrow for the checker
        self.assertEqual(model.cup_radius_mm, 12.0)
        self.assertEqual(model.outer_margin_mm, 1.5)


class GraspCalculatorGripperModelSeamTests(unittest.TestCase):
    def test_default_is_none_byte_identical(self) -> None:
        self.assertIsNone(GraspCalculator()._gripper_model)

    def test_constructor_stores_the_model(self) -> None:
        model = SuctionCupGripperModel()
        self.assertIs(GraspCalculator(gripper_model=model)._gripper_model, model)


if __name__ == "__main__":
    unittest.main()
