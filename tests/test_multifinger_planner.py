"""Multi-finger / N-contact planner tests."""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.robot.grasping import (
    FingerKinematicSpec,
    GraspCalculator,
    GripperKind,
    MultiContactGrasp,
    MultiContactGraspPlanner,
    MultiContactPlanRequest,
    ParallelJawContactPlanner,
    RadialMultiFingerPlanner,
)
from src.robot.grasping.geometry import CameraIntrinsics


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _disk_mask(H: int, W: int, cx: float, cy: float, radius_px: float) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_px**2


def _flat_depth(H: int, W: int, depth_mm: float) -> np.ndarray:
    return np.full((H, W), float(depth_mm), dtype=np.float32)


def _default_intrinsics(W: int = 320, H: int = 240) -> CameraIntrinsics:
    # Choose fx so a 1 mm offset at 500 mm depth maps to a reasonable pixel
    # offset; the exact value is not important for the geometric tests.
    return CameraIntrinsics(fx=600.0, fy=600.0, cx=W / 2.0, cy=H / 2.0)


class _Segmentation:
    """Minimal SegmentationLike stub for plan_multifinger."""

    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask


# --------------------------------------------------------------------------- #
# FingerKinematicSpec validation                                              #
# --------------------------------------------------------------------------- #


class FingerKinematicSpecTests(unittest.TestCase):
    def test_basic_three_finger_spec(self) -> None:
        spec = FingerKinematicSpec(
            finger_count=3, min_radius_mm=10.0, max_radius_mm=60.0
        )
        self.assertEqual(spec.finger_count, 3)

    def test_finger_count_must_be_at_least_two(self) -> None:
        with self.assertRaises(ValueError):
            FingerKinematicSpec(finger_count=1, min_radius_mm=5, max_radius_mm=30)

    def test_radius_ordering_enforced(self) -> None:
        with self.assertRaises(ValueError):
            FingerKinematicSpec(finger_count=2, min_radius_mm=20, max_radius_mm=10)

    def test_negative_palm_offset_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FingerKinematicSpec(
                finger_count=2,
                min_radius_mm=5,
                max_radius_mm=30,
                palm_offset_mm=-1.0,
            )

    def test_excessive_angular_separation_rejected(self) -> None:
        # 4 fingers equal-spaced give 90 deg; require 120 deg -> reject.
        with self.assertRaises(ValueError):
            FingerKinematicSpec(
                finger_count=4,
                min_radius_mm=5,
                max_radius_mm=30,
                min_angular_separation_deg=120.0,
            )


# --------------------------------------------------------------------------- #
# MultiContactGrasp validation                                                #
# --------------------------------------------------------------------------- #


class MultiContactGraspTests(unittest.TestCase):
    def _make(self, n: int) -> MultiContactGrasp:
        points = tuple(
            np.array(
                [
                    30.0 * math.cos(2 * math.pi * i / n),
                    30.0 * math.sin(2 * math.pi * i / n),
                    500.0,
                ]
            )
            for i in range(n)
        )
        normals = tuple(
            np.array(
                [math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n), 0.0]
            )
            for i in range(n)
        )
        return MultiContactGrasp(
            palm_center_mm=np.array([0.0, 0.0, 480.0]),
            approach_axis=np.array([0.0, 0.0, 1.0]),
            contact_points_mm=points,
            contact_normals=normals,
            finger_count=n,
            gripper_kind=GripperKind.THREE_FINGER if n == 3 else GripperKind.FOUR_FINGER,
            score=0.7,
        )

    def test_three_finger_round_trip(self) -> None:
        g = self._make(3)
        d = g.to_dict()
        self.assertEqual(d["finger_count"], 3)
        self.assertEqual(d["gripper_kind"], "three_finger")
        self.assertEqual(len(d["contact_points_mm"]), 3)

    def test_contact_count_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MultiContactGrasp(
                palm_center_mm=np.zeros(3),
                approach_axis=np.array([0.0, 0.0, 1.0]),
                contact_points_mm=(np.zeros(3), np.zeros(3)),
                contact_normals=(np.array([1.0, 0.0, 0.0]),),
                finger_count=2,
                gripper_kind=GripperKind.PARALLEL_JAW,
                score=0.5,
            )

    def test_non_unit_normal_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MultiContactGrasp(
                palm_center_mm=np.zeros(3),
                approach_axis=np.array([0.0, 0.0, 1.0]),
                contact_points_mm=(np.zeros(3), np.zeros(3)),
                contact_normals=(np.array([2.0, 0.0, 0.0]), np.array([-2.0, 0.0, 0.0])),
                finger_count=2,
                gripper_kind=GripperKind.PARALLEL_JAW,
                score=0.5,
            )

    def test_score_must_be_in_unit_interval(self) -> None:
        with self.assertRaises(ValueError):
            MultiContactGrasp(
                palm_center_mm=np.zeros(3),
                approach_axis=np.array([0.0, 0.0, 1.0]),
                contact_points_mm=(np.zeros(3), np.zeros(3)),
                contact_normals=(np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])),
                finger_count=2,
                gripper_kind=GripperKind.PARALLEL_JAW,
                score=1.5,
            )


# --------------------------------------------------------------------------- #
# Radial planner                                                              #
# --------------------------------------------------------------------------- #


class RadialMultiFingerPlannerTests(unittest.TestCase):
    def test_protocol_conformance(self) -> None:
        self.assertIsInstance(RadialMultiFingerPlanner(), MultiContactGraspPlanner)

    def test_plans_three_fingers_on_disk(self) -> None:
        H, W = 240, 320
        cx, cy = 160.0, 120.0
        radius_px = 50.0
        mask = _disk_mask(H, W, cx, cy, radius_px)
        depth = _flat_depth(H, W, 600.0)
        intrinsics = _default_intrinsics(W, H)
        # At fx=600, depth=600 mm: 1 px ~= 1 mm radial. Disk radius
        # 50 px -> ~50 mm contact radius -> sits inside [20, 80].
        spec = FingerKinematicSpec(
            finger_count=3,
            min_radius_mm=20.0,
            max_radius_mm=80.0,
            min_angular_separation_deg=60.0,
        )
        planner = RadialMultiFingerPlanner(rotation_samples=4)
        result = planner.plan(
            MultiContactPlanRequest(
                mask=mask,
                depth_map=depth,
                intrinsics=intrinsics,
                kinematics=spec,
                max_results=4,
            )
        )
        self.assertGreaterEqual(len(result), 1)
        best = result[0]
        self.assertEqual(best.finger_count, 3)
        self.assertEqual(best.gripper_kind, GripperKind.THREE_FINGER)
        for contact, normal in zip(best.contact_points_mm, best.contact_normals):
            self.assertEqual(contact.shape, (3,))
            self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0, places=4)
        # Uniformity should be near 1 for a perfect disk.
        self.assertGreater(best.metadata["mean_radius_mm"], 20.0)
        self.assertLess(best.metadata["radius_std_mm"], 5.0)
        self.assertEqual(best.metadata["confidence_kind"], "heuristic")

    def test_radius_out_of_envelope_returns_empty(self) -> None:
        H, W = 240, 320
        mask = _disk_mask(H, W, 160.0, 120.0, 50.0)
        depth = _flat_depth(H, W, 600.0)
        intrinsics = _default_intrinsics(W, H)
        # Envelope demands [100, 200] mm radii but the disk only gives ~50.
        spec = FingerKinematicSpec(
            finger_count=3,
            min_radius_mm=100.0,
            max_radius_mm=200.0,
            min_angular_separation_deg=60.0,
        )
        planner = RadialMultiFingerPlanner(rotation_samples=4)
        result = planner.plan(
            MultiContactPlanRequest(
                mask=mask,
                depth_map=depth,
                intrinsics=intrinsics,
                kinematics=spec,
            )
        )
        self.assertEqual(result, [])

    def test_empty_mask_returns_empty(self) -> None:
        H, W = 240, 320
        mask = np.zeros((H, W), dtype=bool)
        depth = _flat_depth(H, W, 600.0)
        spec = FingerKinematicSpec(
            finger_count=3, min_radius_mm=10, max_radius_mm=50
        )
        result = RadialMultiFingerPlanner().plan(
            MultiContactPlanRequest(
                mask=mask,
                depth_map=depth,
                intrinsics=_default_intrinsics(W, H),
                kinematics=spec,
            )
        )
        self.assertEqual(result, [])

    def test_four_finger_kind_assignment(self) -> None:
        H, W = 240, 320
        mask = _disk_mask(H, W, 160.0, 120.0, 50.0)
        depth = _flat_depth(H, W, 600.0)
        spec = FingerKinematicSpec(
            finger_count=4,
            min_radius_mm=20.0,
            max_radius_mm=80.0,
            min_angular_separation_deg=60.0,
        )
        result = RadialMultiFingerPlanner(rotation_samples=3).plan(
            MultiContactPlanRequest(
                mask=mask,
                depth_map=depth,
                intrinsics=_default_intrinsics(W, H),
                kinematics=spec,
            )
        )
        self.assertGreaterEqual(len(result), 1)
        self.assertEqual(result[0].gripper_kind, GripperKind.FOUR_FINGER)
        self.assertEqual(result[0].finger_count, 4)


# --------------------------------------------------------------------------- #
# Parallel-jaw planner (wraps antipodal pipeline)                             #
# --------------------------------------------------------------------------- #


class ParallelJawContactPlannerTests(unittest.TestCase):
    def test_protocol_conformance(self) -> None:
        self.assertIsInstance(ParallelJawContactPlanner(), MultiContactGraspPlanner)

    def test_requires_finger_count_two(self) -> None:
        spec = FingerKinematicSpec(
            finger_count=3, min_radius_mm=10, max_radius_mm=50
        )
        H, W = 240, 320
        request = MultiContactPlanRequest(
            mask=_disk_mask(H, W, 160.0, 120.0, 50.0),
            depth_map=_flat_depth(H, W, 600.0),
            intrinsics=_default_intrinsics(W, H),
            kinematics=spec,
        )
        with self.assertRaises(ValueError):
            ParallelJawContactPlanner().plan(request)


# --------------------------------------------------------------------------- #
# GraspCalculator integration                                                 #
# --------------------------------------------------------------------------- #


class CalculatorMultifingerIntegrationTests(unittest.TestCase):
    def test_default_planner_selected_by_finger_count(self) -> None:
        H, W = 240, 320
        mask = _disk_mask(H, W, 160.0, 120.0, 50.0)
        depth = _flat_depth(H, W, 600.0)
        calc = GraspCalculator()
        spec = FingerKinematicSpec(
            finger_count=3,
            min_radius_mm=20.0,
            max_radius_mm=80.0,
            min_angular_separation_deg=60.0,
        )
        result = calc.plan_multifinger(
            segmentation=_Segmentation(mask),
            depth_map=depth,
            intrinsics=_default_intrinsics(W, H),
            kinematics=spec,
        )
        self.assertGreaterEqual(len(result), 1)
        # plan_multifinger must not poison the parallel-jaw compute() pipeline state.
        self.assertEqual(calc.last_telemetry, {})
        self.assertEqual(calc.last_failure_reasons, ())

    def test_missing_segmentation_mask_returns_empty(self) -> None:
        class _Noop:
            mask = None

        calc = GraspCalculator()
        spec = FingerKinematicSpec(
            finger_count=3, min_radius_mm=10, max_radius_mm=50
        )
        result = calc.plan_multifinger(
            segmentation=_Noop(),
            depth_map=_flat_depth(240, 320, 600.0),
            intrinsics=_default_intrinsics(320, 240),
            kinematics=spec,
        )
        self.assertEqual(result, [])

    def test_invalid_planner_rejected(self) -> None:
        calc = GraspCalculator()
        spec = FingerKinematicSpec(
            finger_count=3, min_radius_mm=10, max_radius_mm=50
        )

        class _Bogus:
            pass

        with self.assertRaises(TypeError):
            calc.plan_multifinger(
                segmentation=_Segmentation(_disk_mask(240, 320, 160, 120, 50)),
                depth_map=_flat_depth(240, 320, 600.0),
                intrinsics=_default_intrinsics(320, 240),
                kinematics=spec,
                planner=_Bogus(),  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
