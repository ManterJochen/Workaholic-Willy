"""Boundary tests for ``backend.src.robot.grasping``.

These tests focus on contracts that are *not* covered by
``tests/test_robot_boundaries.py``:

* segmentation-metadata propagation onto every returned ``GraspPoint``;
* mask variants (``uint8 {0,1}``, ``uint8 {0,255}``, ``bool``);
* empty / unusable masks yielding an empty result instead of crashing;
* unit conversion (mm / cm / m) producing equivalent geometry;
* eye-in-hand dynamic-composition path (``T_tool_to_base @ T_cam_to_tool``);
* ambiguous-transform rejection (both ``camera_to_base`` and
  ``T_cam_to_base`` passed simultaneously).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.geometry import Frame, Transform
from src.robot.grasping import (
    GraspCalculator,
    GraspFailureReason,
    GraspFrame,
    GraspResult,
)
from src.robot.grasping.planning import (
    IKResult,
    WorkspaceBoxIKService,
)


def _camera_matrix() -> np.ndarray:
    return np.array(
        [[200.0, 0.0, 12.0], [0.0, 200.0, 12.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _calculator() -> GraspCalculator:
    return GraspCalculator(
        min_grip_width_mm=1.0,
        max_grip_width_mm=200.0,
        max_candidates=3,
        camera_matrix=_camera_matrix(),
    )


def _make_mask(shape: tuple[int, int] = (24, 24)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[7:17, 6:18] = 1
    return mask


def _full_segmentation_result(mask: np.ndarray) -> SimpleNamespace:
    """A ``SegmentationResult``-shaped object with every optional field."""
    return SimpleNamespace(
        label="screw",
        score=0.87,
        bbox_xyxy=(6, 7, 18, 17),
        mask=mask,
        mask_area_px=int(mask.sum()),
        centroid_xy=(11.5, 11.5),
        derived_bbox_xyxy=(6, 7, 18, 17),
        inference_time_s=0.042,
        timestamp_utc="2026-05-11T15:00:00+00:00",
        frame_id="left:001423",
        metadata={"device": "cuda:0", "keep_largest_component": True},
    )


class SegmentationMetadataPropagationTests(unittest.TestCase):
    """SAM2 metadata must reach ``GraspPoint.metadata['segmentation']``."""

    def test_full_segmentation_result_metadata_is_preserved(self) -> None:
        mask = _make_mask()
        seg = _full_segmentation_result(mask)
        depth = np.full(mask.shape, 1000.0, dtype=np.float64)

        grasps = _calculator().compute(seg, depth, pixel_to_mm=5.0, unit="mm")

        self.assertTrue(grasps)
        for grasp in grasps:
            seg_meta = grasp.metadata["segmentation"]
            self.assertEqual(seg_meta["label"], "screw")
            self.assertAlmostEqual(seg_meta["score"], 0.87)
            self.assertEqual(seg_meta["bbox_xyxy"], (6, 7, 18, 17))
            self.assertEqual(seg_meta["mask_area_px"], int(mask.sum()))
            self.assertEqual(seg_meta["frame_id"], "left:001423")
            self.assertEqual(
                seg_meta["timestamp_utc"], "2026-05-11T15:00:00+00:00"
            )
            self.assertEqual(seg_meta["sam2_metadata"]["device"], "cuda:0")

    def test_minimal_segmentation_object_works_without_metadata_key(self) -> None:
        mask = _make_mask()
        seg = SimpleNamespace(mask=mask)
        depth = np.full(mask.shape, 1000.0, dtype=np.float64)

        grasps = _calculator().compute(seg, depth, pixel_to_mm=5.0, unit="mm")

        self.assertTrue(grasps)
        for grasp in grasps:
            self.assertNotIn("segmentation", grasp.metadata)

    def test_metadata_snapshot_is_isolated_from_segmentation_instance(self) -> None:
        """Mutating GraspPoint metadata must not leak back to the segmenter."""
        mask = _make_mask()
        seg = _full_segmentation_result(mask)
        depth = np.full(mask.shape, 1000.0, dtype=np.float64)

        grasps = _calculator().compute(seg, depth, pixel_to_mm=5.0, unit="mm")
        snapshot = grasps[0].metadata["segmentation"]["sam2_metadata"]
        snapshot["device"] = "mutated"

        self.assertEqual(seg.metadata["device"], "cuda:0")


class MaskVariantTests(unittest.TestCase):
    """Mask boundary must accept uint8 {0,1}, uint8 {0,255}, and bool."""

    def _run_with_mask(self, mask: np.ndarray) -> list:
        seg = SimpleNamespace(mask=mask, label="box")
        depth = np.full(mask.shape, 1000.0, dtype=np.float64)
        return _calculator().compute(seg, depth, pixel_to_mm=5.0, unit="mm")

    def test_uint8_zero_one_mask(self) -> None:
        self.assertTrue(self._run_with_mask(_make_mask().astype(np.uint8)))

    def test_uint8_zero_255_mask(self) -> None:
        mask = (_make_mask() * 255).astype(np.uint8)
        self.assertTrue(self._run_with_mask(mask))

    def test_bool_mask(self) -> None:
        self.assertTrue(self._run_with_mask(_make_mask().astype(bool)))

    def test_empty_mask_yields_empty_result(self) -> None:
        mask = np.zeros((24, 24), dtype=np.uint8)
        seg = SimpleNamespace(mask=mask, label="missing")
        depth = np.full(mask.shape, 1000.0, dtype=np.float64)

        grasps = _calculator().compute(seg, depth, pixel_to_mm=5.0, unit="mm")

        self.assertEqual(grasps, [])

    def test_mask_under_min_area_yields_empty_result(self) -> None:
        mask = np.zeros((24, 24), dtype=np.uint8)
        mask[0:2, 0:2] = 1  # 4 px, below default min_area_px (50)
        seg = SimpleNamespace(mask=mask, label="tiny")
        depth = np.full(mask.shape, 1000.0, dtype=np.float64)

        grasps = _calculator().compute(seg, depth, pixel_to_mm=5.0, unit="mm")

        self.assertEqual(grasps, [])


class UnitConversionTests(unittest.TestCase):
    """mm / cm / m depth inputs must produce equivalent results."""

    def _grasps_for_unit(self, unit: str, depth_value: float) -> list:
        mask = _make_mask()
        seg = SimpleNamespace(mask=mask, label="box")
        depth = np.full(mask.shape, depth_value, dtype=np.float64)
        return _calculator().compute(seg, depth, pixel_to_mm=5.0, unit=unit)

    def test_units_mm_cm_m_yield_equivalent_geometry(self) -> None:
        grasps_mm = self._grasps_for_unit("mm", 1000.0)
        grasps_cm = self._grasps_for_unit("cm", 100.0)
        grasps_m = self._grasps_for_unit("m", 1.0)

        self.assertEqual(len(grasps_mm), len(grasps_cm))
        self.assertEqual(len(grasps_mm), len(grasps_m))
        # Positions are in mm regardless of the input unit; same depth in
        # different units must give the same world position.
        np.testing.assert_allclose(
            grasps_mm[0].position, grasps_cm[0].position, atol=1e-6
        )
        np.testing.assert_allclose(
            grasps_mm[0].position, grasps_m[0].position, atol=1e-6
        )

    def test_invalid_unit_is_rejected(self) -> None:
        mask = _make_mask()
        seg = SimpleNamespace(mask=mask)
        depth = np.full(mask.shape, 1000.0, dtype=np.float64)

        with self.assertRaises(ValueError):
            _calculator().compute(seg, depth, pixel_to_mm=5.0, unit="inches")


class EyeInHandCompositionTests(unittest.TestCase):
    """``T_cam_to_base_now = T_tool_to_base @ T_cam_to_tool`` integration."""

    @staticmethod
    def _make_rigid(translation_mm: np.ndarray, axis_angle: np.ndarray) -> np.ndarray:
        """Return a 4x4 rigid transform from a Rodrigues vector and translation."""
        theta = float(np.linalg.norm(axis_angle))
        if theta < 1e-12:
            rotation = np.eye(3, dtype=np.float64)
        else:
            k = axis_angle / theta
            kx, ky, kz = float(k[0]), float(k[1]), float(k[2])
            K = np.array(
                [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=np.float64
            )
            rotation = (
                np.eye(3, dtype=np.float64)
                + np.sin(theta) * K
                + (1.0 - np.cos(theta)) * (K @ K)
            )
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation_mm
        return matrix

    def test_eye_in_hand_dynamic_composition_returns_base_frame(self) -> None:
        T_cam_to_tool = self._make_rigid(
            translation_mm=np.array([40.0, -10.0, 80.0], dtype=np.float64),
            axis_angle=np.array([0.05, 0.10, 0.02], dtype=np.float64),
        )
        T_tool_to_base_now = self._make_rigid(
            translation_mm=np.array([500.0, -200.0, 350.0], dtype=np.float64),
            axis_angle=np.array([0.20, -0.15, 0.08], dtype=np.float64),
        )
        T_cam_to_base_now = T_tool_to_base_now @ T_cam_to_tool

        mask = _make_mask()
        seg = _full_segmentation_result(mask)
        depth = np.full(mask.shape, 1000.0, dtype=np.float64)

        # Raw-matrix boundary (legacy).
        grasps_raw = _calculator().compute(
            seg, depth, T_cam_to_base=T_cam_to_base_now, pixel_to_mm=5.0, unit="mm"
        )
        # Typed-Transform boundary.
        transform = Transform.from_matrix(
            T_cam_to_base_now, from_frame=Frame.CAMERA, to_frame=Frame.BASE
        )
        grasps_typed = _calculator().compute(
            seg, depth, camera_to_base=transform, pixel_to_mm=5.0, unit="mm"
        )

        self.assertTrue(grasps_raw)
        self.assertTrue(grasps_typed)
        self.assertTrue(all(g.frame is GraspFrame.BASE for g in grasps_raw))
        self.assertTrue(all(g.frame is GraspFrame.BASE for g in grasps_typed))
        # Both boundary paths must yield the same world position for the
        # top-ranked candidate.
        np.testing.assert_allclose(
            grasps_raw[0].position, grasps_typed[0].position, atol=1e-6
        )

    def test_eye_in_hand_stale_pose_changes_world_position(self) -> None:
        """Different TCP poses must produce different world positions.

        This protects against silently re-using a stale TCP pose: if the
        production path forgot to compose with the live pose, the test
        would not detect a shift in the world-frame position.
        """
        T_cam_to_tool = self._make_rigid(
            translation_mm=np.array([40.0, -10.0, 80.0], dtype=np.float64),
            axis_angle=np.array([0.05, 0.10, 0.02], dtype=np.float64),
        )
        T_tool_to_base_stale = self._make_rigid(
            translation_mm=np.array([500.0, -200.0, 350.0], dtype=np.float64),
            axis_angle=np.array([0.20, -0.15, 0.08], dtype=np.float64),
        )
        T_tool_to_base_now = self._make_rigid(
            translation_mm=np.array([520.0, -180.0, 360.0], dtype=np.float64),
            axis_angle=np.array([0.22, -0.13, 0.09], dtype=np.float64),
        )

        mask = _make_mask()
        seg = SimpleNamespace(mask=mask, label="box")
        depth = np.full(mask.shape, 1000.0, dtype=np.float64)

        grasps_stale = _calculator().compute(
            seg,
            depth,
            T_cam_to_base=T_tool_to_base_stale @ T_cam_to_tool,
            pixel_to_mm=5.0,
            unit="mm",
        )
        grasps_now = _calculator().compute(
            seg,
            depth,
            T_cam_to_base=T_tool_to_base_now @ T_cam_to_tool,
            pixel_to_mm=5.0,
            unit="mm",
        )

        self.assertTrue(grasps_stale)
        self.assertTrue(grasps_now)
        # Different live TCP -> different world position.
        self.assertFalse(
            np.allclose(grasps_stale[0].position, grasps_now[0].position, atol=1.0)
        )


class GraspResultFeedbackTests(unittest.TestCase):
    """``compute_result`` exposes failure reasons and depth/mask confidence."""

    def test_success_path_returns_no_reasons_and_top_score(self) -> None:
        mask = _make_mask()
        seg = _full_segmentation_result(mask)
        depth = np.full(mask.shape, 1000.0, dtype=np.float64)

        result = _calculator().compute_result(seg, depth, pixel_to_mm=5.0, unit="mm")

        self.assertIsInstance(result, GraspResult)
        self.assertTrue(result.is_success)
        self.assertEqual(result.reasons, ())
        self.assertGreater(result.top_score, 0.0)
        self.assertEqual(result.mask_confidence, 0.87)
        self.assertIsNotNone(result.depth_confidence)
        self.assertIs(result.best, result.candidates[0])

    def test_empty_mask_reports_empty_mask_reason(self) -> None:
        mask = np.zeros((24, 24), dtype=np.uint8)
        seg = SimpleNamespace(mask=mask)
        depth = np.full(mask.shape, 1000.0, dtype=np.float64)

        result = _calculator().compute_result(seg, depth, pixel_to_mm=5.0, unit="mm")

        self.assertFalse(result.is_success)
        self.assertEqual(result.candidates, ())
        self.assertIn(GraspFailureReason.EMPTY_MASK, result.reasons)
        self.assertIn(GraspFailureReason.RESCAN_RECOMMENDED, result.reasons)

    def test_no_valid_depth_reports_no_valid_depth_reason(self) -> None:
        mask = _make_mask()
        seg = SimpleNamespace(mask=mask)
        # All depth pixels NaN under the mask.
        depth = np.full(mask.shape, np.nan, dtype=np.float64)

        result = _calculator().compute_result(seg, depth, pixel_to_mm=5.0, unit="mm")

        self.assertFalse(result.is_success)
        self.assertIn(GraspFailureReason.NO_VALID_DEPTH, result.reasons)

    def test_to_dict_is_json_friendly(self) -> None:
        import json

        mask = _make_mask()
        seg = _full_segmentation_result(mask)
        depth = np.full(mask.shape, 1000.0, dtype=np.float64)

        result = _calculator().compute_result(seg, depth, pixel_to_mm=5.0, unit="mm")
        payload = result.to_dict()

        # Round-trip through json.dumps with default=str catches any
        # numpy / Enum values that would block serialisation.
        json.dumps(payload, default=str)
        self.assertIn("candidates", payload)
        self.assertIn("reasons", payload)
        self.assertIn("telemetry", payload)


class DeterminismTests(unittest.TestCase):
    """The geometric pipeline must be reproducible for a given seed."""

    def _grasps(self, seed: int | None) -> list:
        mask = _make_mask()
        seg = SimpleNamespace(mask=mask, label="box")
        depth = np.full(mask.shape, 1000.0, dtype=np.float64)
        return _calculator().compute(
            seg, depth, pixel_to_mm=5.0, unit="mm", seed=seed
        )

    def test_same_seed_yields_identical_candidates(self) -> None:
        a = self._grasps(seed=42)
        b = self._grasps(seed=42)
        self.assertEqual(len(a), len(b))
        for ga, gb in zip(a, b):
            np.testing.assert_array_equal(ga.position, gb.position)
            np.testing.assert_array_equal(ga.approach, gb.approach)
            np.testing.assert_array_equal(ga.axis, gb.axis)
            self.assertEqual(ga.score, gb.score)

    def test_none_seed_still_returns_candidates(self) -> None:
        self.assertTrue(self._grasps(seed=None))

    def test_negative_seed_is_rejected(self) -> None:
        mask = _make_mask()
        seg = SimpleNamespace(mask=mask)
        depth = np.full(mask.shape, 1000.0, dtype=np.float64)
        with self.assertRaises(ValueError):
            _calculator().compute(seg, depth, pixel_to_mm=5.0, seed=-1)


class _RejectAllIKService:
    """Vendor-neutral fake IK that declares every pose unreachable."""

    def __init__(self) -> None:
        self.queries = 0

    def query(self, pose):  # noqa: ANN001 - fake duck-typed service
        self.queries += 1
        return IKResult(reachable=False, reason="ik_failed")


class _AcceptAllIKService:
    def query(self, pose):  # noqa: ANN001
        return IKResult(reachable=True, joints=(0.0,) * 6)


class _BadReturnIKService:
    def query(self, pose):  # noqa: ANN001
        return "not-an-ik-result"


class IKServiceFilterTests(unittest.TestCase):
    """Phase D — vendor-neutral reachability filtering."""

    def test_reject_all_emits_ik_failed_and_empty_result(self) -> None:
        mask = _make_mask()
        seg = SimpleNamespace(mask=mask)
        depth = np.full(mask.shape, 1000.0, dtype=np.float64)
        service = _RejectAllIKService()
        result = _calculator().compute_result(
            seg, depth, pixel_to_mm=5.0, ik_service=service
        )
        self.assertEqual(result.candidates, ())
        self.assertIn(GraspFailureReason.IK_FAILED, result.reasons)
        self.assertGreater(service.queries, 0)
        self.assertGreaterEqual(result.telemetry.get("rejected_ik", 0), 1)

    def test_accept_all_preserves_candidates(self) -> None:
        mask = _make_mask()
        seg = SimpleNamespace(mask=mask)
        depth = np.full(mask.shape, 1000.0, dtype=np.float64)
        baseline = _calculator().compute(seg, depth, pixel_to_mm=5.0)
        result = _calculator().compute_result(
            seg, depth, pixel_to_mm=5.0, ik_service=_AcceptAllIKService()
        )
        self.assertEqual(len(result.candidates), len(baseline))
        self.assertNotIn(GraspFailureReason.IK_FAILED, result.reasons)

    def test_bad_return_raises_typeerror(self) -> None:
        mask = _make_mask()
        seg = SimpleNamespace(mask=mask)
        depth = np.full(mask.shape, 1000.0, dtype=np.float64)
        with self.assertRaises(TypeError):
            _calculator().compute(
                seg, depth, pixel_to_mm=5.0, ik_service=_BadReturnIKService()
            )

    def test_workspace_box_service_filters_by_position(self) -> None:
        from src.robot.grasping.planning import (
            GraspPose,
            filter_reachable_poses,
        )

        def _pose(x: float) -> GraspPose:
            return GraspPose(
                position_mm=np.array([x, 0.0, 0.0], dtype=np.float64),
                rotation_matrix=np.eye(3, dtype=np.float64),
                grip_width_mm=40.0,
                score=0.5,
                confidence=0.5,
                contacts=(
                    np.array([x, -20.0, 0.0], dtype=np.float64),
                    np.array([x, 20.0, 0.0], dtype=np.float64),
                ),
                frame=Frame.BASE,
                metadata={},
            )

        service = WorkspaceBoxIKService(
            min_corner_mm=np.array([-100.0, -100.0, -100.0]),
            max_corner_mm=np.array([100.0, 100.0, 100.0]),
        )
        poses = [_pose(0.0), _pose(500.0), _pose(50.0)]
        kept, diagnostics = filter_reachable_poses(poses, service)
        self.assertEqual([float(p.position_mm[0]) for p in kept], [0.0, 50.0])
        self.assertEqual(len(diagnostics), 3)
        self.assertFalse(diagnostics[1].reachable)


class MotionPlanWaypointTests(unittest.TestCase):
    """Phase E — pre-grasp / approach / retreat helpers.

    The helpers in :mod:`src.robot.grasping.planning.approach_planner`
    are vendor-neutral and must produce poses the motion layer can hand
    to ``RobotArm.move_to`` without grasping-specific imports.
    """

    @staticmethod
    def _grasp() -> "object":
        from src.robot.grasping.planning import GraspPose

        return GraspPose(
            position_mm=np.array([100.0, 50.0, 200.0], dtype=np.float64),
            rotation_matrix=np.eye(3, dtype=np.float64),
            grip_width_mm=40.0,
            score=0.5,
            confidence=0.5,
            contacts=(
                np.array([100.0, 30.0, 200.0], dtype=np.float64),
                np.array([100.0, 70.0, 200.0], dtype=np.float64),
            ),
            frame=Frame.BASE,
            metadata={},
        )

    def test_pre_grasp_offset_along_negative_approach_axis(self) -> None:
        from src.robot.grasping.planning import pre_grasp_pose

        grasp = self._grasp()
        pre = pre_grasp_pose(grasp, standoff_mm=80.0)
        delta = pre.position_mm - grasp.position_mm
        # approach axis is +Z for identity rotation -> pre should be at -Z
        np.testing.assert_allclose(delta, np.array([0.0, 0.0, -80.0]))
        self.assertEqual(pre.frame, Frame.BASE)
        self.assertEqual(pre.metadata.get("approach_role"), "pre_grasp")

    def test_retreat_pose_lifts_along_world_up(self) -> None:
        from src.robot.grasping.planning import retreat_pose

        grasp = self._grasp()
        retreat = retreat_pose(grasp, lift_mm=120.0)
        delta = retreat.position_mm - grasp.position_mm
        np.testing.assert_allclose(delta, np.array([0.0, 0.0, 120.0]))
        self.assertEqual(retreat.metadata.get("approach_role"), "retreat")

    def test_approach_waypoints_inclusive_endpoints_and_linear(self) -> None:
        from src.robot.grasping.planning import approach_waypoints

        grasp = self._grasp()
        waypoints = approach_waypoints(grasp, standoff_mm=60.0, num_waypoints=4)
        self.assertEqual(len(waypoints), 4)
        for wp in waypoints:
            self.assertEqual(wp.frame, Frame.BASE)
        # First waypoint == pre-grasp position; last == grasp position
        np.testing.assert_allclose(
            waypoints[0].position_mm,
            grasp.position_mm + np.array([0.0, 0.0, -60.0]),
        )
        np.testing.assert_allclose(waypoints[-1].position_mm, grasp.position_mm)
        # Mid-segment is colinear
        midpoint = (waypoints[0].position_mm + waypoints[-1].position_mm) / 2.0
        # With 4 waypoints, indices 0,1,2,3 -> fractions 0, 1/3, 2/3, 1
        # so no exact midpoint; check linearity via collinearity tolerance.
        direction = waypoints[-1].position_mm - waypoints[0].position_mm
        for wp in waypoints[1:-1]:
            offset = wp.position_mm - waypoints[0].position_mm
            cross = np.cross(direction, offset)
            self.assertLess(float(np.linalg.norm(cross)), 1e-9)
        del midpoint  # silence unused

    def test_approach_waypoints_rejects_too_few(self) -> None:
        from src.robot.grasping.planning import approach_waypoints

        with self.assertRaises(ValueError):
            approach_waypoints(self._grasp(), num_waypoints=1)


if __name__ == "__main__":
    unittest.main()
