"""Boundary tests for ``backend.src.robot.grasping``.

These tests focus on contracts that are *not* covered by
``tests/robot/drivers/test_robot_boundaries.py``:

* segmentation-metadata propagation onto every returned ``GraspPoint``;
* mask variants (``uint8 {0,1}``, ``uint8 {0,255}``, ``bool``);
* empty / unusable masks yielding an empty result instead of crashing;
* unit conversion (mm / cm / m) producing equivalent geometry;
* eye-in-hand dynamic-composition path (``T_tool_to_base @ T_cam_to_tool``);
* ambiguous-transform rejection (both ``camera_to_base`` and
  ``T_cam_to_base`` passed simultaneously).

Every call grasps a 40 mm box on the bench, seen straight down (``tests/_calculator_scenes.py``): the calculator plans
in BASE over the support the part stands on, and with no BASE plane it offers nothing and says why.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.geometry import Frame, Transform
from src.robot.grasping import (
    GraspCalculator,
    GraspFailureReason,
    GraspFrame,
    GraspResult,
)
from src.robot.grasping.planning import IKResult
from tests._calculator_scenes import box_on_the_bench

#: The box the calls grasp.
SCENE = box_on_the_bench()


def _calculator() -> GraspCalculator:
    return GraspCalculator(
        min_grip_width_mm=5.0,
        max_grip_width_mm=85.0,
        max_candidates=3,
        camera_matrix=SCENE.camera_matrix,
    )


def _make_mask() -> np.ndarray:
    return SCENE.mask.astype(np.uint8)


def _compute(seg: Any, depth: "np.ndarray | None" = None, *, result: bool = False, **keywords: Any) -> Any:
    """One call on the bench scene: its depth, where the camera stood and the bench, unless given."""
    keywords.setdefault("unit", "mm")
    if "T_cam_to_base" not in keywords:
        keywords.setdefault("camera_to_base", SCENE.camera_to_base)
    keywords.setdefault("support_plane", SCENE.support_plane)
    calculator = _calculator()
    run = calculator.compute_result if result else calculator.compute
    return run(seg, SCENE.depth_mm if depth is None else depth, **keywords)


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

        grasps = _compute(seg)

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
        grasps = _compute(SimpleNamespace(mask=_make_mask()))

        self.assertTrue(grasps)
        for grasp in grasps:
            self.assertNotIn("segmentation", grasp.metadata)

    def test_metadata_snapshot_is_isolated_from_segmentation_instance(self) -> None:
        """Mutating GraspPoint metadata must not leak back to the segmenter."""
        seg = _full_segmentation_result(_make_mask())

        grasps = _compute(seg)
        snapshot = grasps[0].metadata["segmentation"]["sam2_metadata"]
        snapshot["device"] = "mutated"

        self.assertEqual(seg.metadata["device"], "cuda:0")


class MaskVariantTests(unittest.TestCase):
    """Mask boundary must accept uint8 {0,1}, uint8 {0,255}, and bool."""

    def _run_with_mask(self, mask: np.ndarray) -> list:
        return _compute(SimpleNamespace(mask=mask, label="box"))

    def test_uint8_zero_one_mask(self) -> None:
        self.assertTrue(self._run_with_mask(_make_mask().astype(np.uint8)))

    def test_uint8_zero_255_mask(self) -> None:
        mask = (_make_mask() * 255).astype(np.uint8)
        self.assertTrue(self._run_with_mask(mask))

    def test_bool_mask(self) -> None:
        self.assertTrue(self._run_with_mask(_make_mask().astype(bool)))

    def test_empty_mask_yields_empty_result(self) -> None:
        mask = np.zeros(SCENE.mask.shape, dtype=np.uint8)

        self.assertEqual(_compute(SimpleNamespace(mask=mask, label="missing")), [])

    def test_mask_under_min_area_yields_empty_result(self) -> None:
        mask = np.zeros(SCENE.mask.shape, dtype=np.uint8)
        mask[0:2, 0:2] = 1  # 4 px, below default min_area_px (50)

        self.assertEqual(_compute(SimpleNamespace(mask=mask, label="tiny")), [])


class UnitConversionTests(unittest.TestCase):
    """mm / cm / m depth inputs must produce equivalent results."""

    def _grasps_for_unit(self, unit: str, per_mm: float) -> list:
        return _compute(SimpleNamespace(mask=_make_mask(), label="box"), SCENE.depth_mm * per_mm, unit=unit)

    def test_units_mm_cm_m_yield_equivalent_geometry(self) -> None:
        grasps_mm = self._grasps_for_unit("mm", 1.0)
        grasps_cm = self._grasps_for_unit("cm", 0.1)
        grasps_m = self._grasps_for_unit("m", 0.001)

        self.assertTrue(grasps_mm)
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
        with self.assertRaises(ValueError):
            _compute(SimpleNamespace(mask=_make_mask()), unit="inches")


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

    @staticmethod
    def _tool_to_base_now(cam_to_tool: np.ndarray) -> np.ndarray:
        """The tool pose that puts the camera where the scene's camera stood."""
        return SCENE.camera_to_base @ np.linalg.inv(cam_to_tool)

    def test_eye_in_hand_dynamic_composition_returns_base_frame(self) -> None:
        T_cam_to_tool = self._make_rigid(
            translation_mm=np.array([40.0, -10.0, 80.0], dtype=np.float64),
            axis_angle=np.array([0.05, 0.10, 0.02], dtype=np.float64),
        )
        T_cam_to_base_now = self._tool_to_base_now(T_cam_to_tool) @ T_cam_to_tool

        seg = _full_segmentation_result(_make_mask())

        # Raw-matrix boundary (legacy).
        grasps_raw = _compute(seg, T_cam_to_base=T_cam_to_base_now)
        # Typed-Transform boundary.
        transform = Transform.from_matrix(
            T_cam_to_base_now, from_frame=Frame.CAMERA, to_frame=Frame.BASE
        )
        grasps_typed = _compute(seg, camera_to_base=transform)

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
        T_tool_to_base_now = self._tool_to_base_now(T_cam_to_tool)
        shift = np.eye(4)
        shift[:3, 3] = (20.0, -15.0, 0.0)
        T_tool_to_base_stale = shift @ T_tool_to_base_now

        seg = SimpleNamespace(mask=_make_mask(), label="box")

        grasps_stale = _compute(seg, T_cam_to_base=T_tool_to_base_stale @ T_cam_to_tool)
        grasps_now = _compute(seg, T_cam_to_base=T_tool_to_base_now @ T_cam_to_tool)

        self.assertTrue(grasps_stale)
        self.assertTrue(grasps_now)
        # Different live TCP -> different world position.
        self.assertFalse(
            np.allclose(grasps_stale[0].position, grasps_now[0].position, atol=1.0)
        )


class GraspResultFeedbackTests(unittest.TestCase):
    """``compute_result`` exposes failure reasons and depth/mask confidence."""

    def test_success_path_returns_no_reasons_and_top_score(self) -> None:
        result = _compute(_full_segmentation_result(_make_mask()), result=True)

        self.assertIsInstance(result, GraspResult)
        self.assertTrue(result.is_success)
        self.assertEqual(result.reasons, ())
        self.assertGreater(result.top_score, 0.0)
        self.assertEqual(result.mask_confidence, 0.87)
        self.assertIsNotNone(result.depth_confidence)
        self.assertIs(result.best, result.candidates[0])

    def test_empty_mask_reports_empty_mask_reason(self) -> None:
        mask = np.zeros(SCENE.mask.shape, dtype=np.uint8)

        result = _compute(SimpleNamespace(mask=mask), result=True)

        self.assertFalse(result.is_success)
        self.assertEqual(result.candidates, ())
        self.assertIn(GraspFailureReason.EMPTY_MASK, result.reasons)
        self.assertIn(GraspFailureReason.RESCAN_RECOMMENDED, result.reasons)

    def test_no_valid_depth_reports_no_valid_depth_reason(self) -> None:
        # All depth pixels NaN under the mask.
        depth = np.where(SCENE.mask, np.nan, SCENE.depth_mm)

        result = _compute(SimpleNamespace(mask=_make_mask()), depth, result=True)

        self.assertFalse(result.is_success)
        self.assertIn(GraspFailureReason.NO_VALID_DEPTH, result.reasons)

    def test_without_a_base_support_plane_nothing_is_offered_and_it_is_said(self) -> None:
        """The grasps are planned over the support in BASE: with no plane, or no transform to place one, none."""
        for keywords in ({"support_plane": None}, {"support_plane": None, "camera_to_base": None}):
            with self.subTest(keywords=sorted(keywords)):
                result = _compute(SimpleNamespace(mask=_make_mask()), result=True, **keywords)
                self.assertEqual(result.candidates, ())
                self.assertEqual((GraspFailureReason.NO_CANDIDATES_GENERATED,), result.reasons)
                self.assertFalse(result.telemetry["support_footprint_ready"])
                self.assertIn("plans in BASE over the support", result.telemetry["no_grasp_said"])

    def test_to_dict_is_json_friendly(self) -> None:
        import json

        result = _compute(_full_segmentation_result(_make_mask()), result=True)
        payload = result.to_dict()

        # Round-trip through json.dumps with default=str catches any
        # numpy / Enum values that would block serialisation.
        json.dumps(payload, default=str)
        self.assertIn("candidates", payload)
        self.assertIn("reasons", payload)
        self.assertIn("telemetry", payload)


class DeterminismTests(unittest.TestCase):
    """The geometric pipeline is reproducible: the same frame gives the same grasps."""

    def test_the_same_frame_yields_identical_candidates(self) -> None:
        a = _compute(SimpleNamespace(mask=_make_mask(), label="box"))
        b = _compute(SimpleNamespace(mask=_make_mask(), label="box"))
        self.assertTrue(a)
        self.assertEqual(len(a), len(b))
        for ga, gb in zip(a, b):
            np.testing.assert_array_equal(ga.position, gb.position)
            np.testing.assert_array_equal(ga.approach, gb.approach)
            np.testing.assert_array_equal(ga.axis, gb.axis)
            self.assertEqual(ga.score, gb.score)


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
        service = _RejectAllIKService()
        result = _compute(SimpleNamespace(mask=_make_mask()), result=True, ik_service=service)
        self.assertEqual(result.candidates, ())
        self.assertIn(GraspFailureReason.IK_FAILED, result.reasons)
        self.assertGreater(service.queries, 0)
        self.assertGreaterEqual(result.telemetry.get("rejected_ik", 0), 1)

    def test_accept_all_preserves_candidates(self) -> None:
        seg = SimpleNamespace(mask=_make_mask())
        baseline = _compute(seg)
        result = _compute(seg, result=True, ik_service=_AcceptAllIKService())
        self.assertTrue(baseline)
        self.assertEqual(len(result.candidates), len(baseline))
        self.assertNotIn(GraspFailureReason.IK_FAILED, result.reasons)

    def test_bad_return_raises_typeerror(self) -> None:
        with self.assertRaises(TypeError):
            _compute(SimpleNamespace(mask=_make_mask()), ik_service=_BadReturnIKService())


if __name__ == "__main__":
    unittest.main()
