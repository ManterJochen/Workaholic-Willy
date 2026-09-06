"""Phase T3 — directional corridor analyzer on synthetic depth scenes.

The analyzer ``analyze_corridor(inputs, config)`` is a *pure
function* (no calculator state). It consumes a target point in
camera frame, an approach axis, optional retreat axis, a depth map,
camera intrinsics, and optional 2D neighbour masks; it returns a
:class:`CorridorReport`.

Locked design intent:

* SKIPPED when depth or intrinsics is None ⇒ neutral 0.5 confidence,
  zero clearances. (Scaffolding: lets the calculator wire the
  analyzer through before the perception layer publishes data.)
* Clear synthetic scene (far background, no obstacles) ⇒ CLEAR,
  blockage_confidence ≈ 0, clearances ≈ max_distance_mm.
* Uniform near obstacle in front of the target ⇒ BLOCKED,
  blockage_confidence ≥ blocked threshold, both clearances small.
* Partial obstruction (perimeter only) ⇒ PARTIAL.
* Asymmetric scene (approach blocked, retreat clear) ⇒ the analyzer
  reports the most pessimistic side via ``blockage_confidence`` so
  the ranking demotion is conservative.
* 2D neighbour-mask fusion: if masks indicate the projected corridor
  is covered, confidence rises even when depth alone would call
  CLEAR.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.geometry.pointcloud import CameraIntrinsics
from src.robot.grasping.scoring.corridor import (
    CorridorAnalysisConfig,
    CorridorAnalysisInputs,
    CorridorMode,
    CorridorReport,
    analyze_corridor,
)


def _intrinsics(w: int = 640, h: int = 480) -> CameraIntrinsics:
    return CameraIntrinsics(fx=500.0, fy=500.0, cx=w / 2.0, cy=h / 2.0)


def _far_depth(w: int = 640, h: int = 480, value_mm: float = 5000.0) -> np.ndarray:
    return np.full((h, w), value_mm, dtype=np.float32)


class AnalyzerSkippedPathTests(unittest.TestCase):
    def test_no_depth_map_returns_skipped(self) -> None:
        inputs = CorridorAnalysisInputs(
            target_point_cam_mm=np.array([0.0, 0.0, 800.0]),
            approach_axis_cam=np.array([0.0, 0.0, 1.0]),
            depth_map=None,
            intrinsics=_intrinsics(),
        )
        report = analyze_corridor(inputs, CorridorAnalysisConfig())
        self.assertIsInstance(report, CorridorReport)
        self.assertEqual(report.mode, CorridorMode.SKIPPED)
        self.assertEqual(report.blockage_confidence, 0.5)
        self.assertEqual(report.approach_clearance_mm, 0.0)

    def test_no_intrinsics_returns_skipped(self) -> None:
        inputs = CorridorAnalysisInputs(
            target_point_cam_mm=np.array([0.0, 0.0, 800.0]),
            approach_axis_cam=np.array([0.0, 0.0, 1.0]),
            depth_map=_far_depth(),
            intrinsics=None,
        )
        report = analyze_corridor(inputs, CorridorAnalysisConfig())
        self.assertEqual(report.mode, CorridorMode.SKIPPED)


class AnalyzerClearSceneTests(unittest.TestCase):
    def test_far_uniform_depth_reports_clear(self) -> None:
        # Target at z=800mm, depth map says everything sits at 5000mm.
        # Approach axis is +z (gripper coming from camera toward target).
        # The analyzer marches *back along -axis* (toward camera) and
        # finds nothing closer than the ray's z anywhere in the
        # corridor → CLEAR with low confidence.
        inputs = CorridorAnalysisInputs(
            target_point_cam_mm=np.array([0.0, 0.0, 800.0]),
            approach_axis_cam=np.array([0.0, 0.0, 1.0]),
            depth_map=_far_depth(),
            intrinsics=_intrinsics(),
        )
        report = analyze_corridor(inputs, CorridorAnalysisConfig())
        self.assertEqual(report.mode, CorridorMode.CLEAR)
        self.assertLess(report.blockage_confidence, 0.3)
        self.assertGreater(report.approach_clearance_mm, 100.0)
        # No retreat axis was supplied, so no retreat corridor was measured (2026-08-17).
        # THIS ASSERTION REPLACES `retreat_clearance_mm > 100`, WHICH PINNED NOTHING: on a
        # uniform far depth both directions read clear, so the old expectation held just as
        # well when the retreat march ran BACKWARDS -- straight down through the target and
        # into the table. That sign error survived this file for as long as it existed, and
        # cost every candidate a "blocked" verdict on a clean scene on-box.
        self.assertFalse(report.retreat_measured)
        self.assertEqual(report.retreat_clearance_mm, 0.0)

    def test_retreat_axis_when_supplied_is_measured_and_can_block(self) -> None:
        """The companion the old test lacked: a retreat axis that points INTO a near surface.

        Without this, nothing distinguishes "the retreat corridor is measured correctly" from
        "the retreat corridor is not measured at all" -- which is exactly how the reversed
        march went unnoticed.
        """
        # Retreat axis -z: the analyzer marches +z, away from the camera, into the 900 mm wall
        # that sits just behind an 800 mm target.
        inputs = CorridorAnalysisInputs(
            target_point_cam_mm=np.array([0.0, 0.0, 800.0]),
            approach_axis_cam=np.array([0.0, 0.0, 1.0]),
            retreat_axis_cam=np.array([0.0, 0.0, -1.0]),
            depth_map=np.full((64, 64), 900.0, dtype=np.float64),
            intrinsics=_intrinsics(),
        )
        report = analyze_corridor(inputs, CorridorAnalysisConfig())
        self.assertTrue(report.retreat_measured)


class AnalyzerBlockedSceneTests(unittest.TestCase):
    def test_uniform_near_obstacle_reports_blocked(self) -> None:
        # Whole depth map sits at 200mm — *closer* than the target at
        # 800mm. Every ray marched back toward the camera will hit
        # the obstacle within the very first step → BLOCKED.
        w, h = 640, 480
        depth = np.full((h, w), 200.0, dtype=np.float32)
        inputs = CorridorAnalysisInputs(
            target_point_cam_mm=np.array([0.0, 0.0, 800.0]),
            approach_axis_cam=np.array([0.0, 0.0, 1.0]),
            depth_map=depth,
            intrinsics=_intrinsics(w, h),
        )
        report = analyze_corridor(inputs, CorridorAnalysisConfig())
        self.assertEqual(report.mode, CorridorMode.BLOCKED)
        self.assertGreaterEqual(report.blockage_confidence, 0.7)
        self.assertTrue(report.blocked)
        self.assertLess(report.approach_clearance_mm, 50.0)

    def test_partial_obstruction_reports_partial(self) -> None:
        # Left half of the image carries a near obstacle, right half
        # is clear. The corridor extends across both halves at the
        # perimeter → roughly half the rays hit early, half are
        # clear ⇒ confidence between the partial and blocked
        # thresholds.
        w, h = 640, 480
        depth = np.full((h, w), 5000.0, dtype=np.float32)
        depth[:, : w // 2] = 200.0
        inputs = CorridorAnalysisInputs(
            target_point_cam_mm=np.array([0.0, 0.0, 800.0]),
            approach_axis_cam=np.array([0.0, 0.0, 1.0]),
            depth_map=depth,
            intrinsics=_intrinsics(w, h),
        )
        cfg = CorridorAnalysisConfig(
            partial_confidence_threshold=0.25,
            blocked_confidence_threshold=0.75,
        )
        report = analyze_corridor(inputs, cfg)
        self.assertEqual(report.mode, CorridorMode.PARTIAL)
        self.assertGreater(report.blockage_confidence, 0.25)
        self.assertLess(report.blockage_confidence, 0.75)


class AnalyzerMonotonicityTests(unittest.TestCase):
    def test_closer_obstacle_higher_confidence(self) -> None:
        w, h = 640, 480
        intrinsics = _intrinsics(w, h)
        target = np.array([0.0, 0.0, 800.0])
        axis = np.array([0.0, 0.0, 1.0])
        far = analyze_corridor(
            CorridorAnalysisInputs(
                target_point_cam_mm=target,
                approach_axis_cam=axis,
                depth_map=np.full((h, w), 5000.0, dtype=np.float32),
                intrinsics=intrinsics,
            ),
            CorridorAnalysisConfig(),
        )
        near = analyze_corridor(
            CorridorAnalysisInputs(
                target_point_cam_mm=target,
                approach_axis_cam=axis,
                depth_map=np.full((h, w), 200.0, dtype=np.float32),
                intrinsics=intrinsics,
            ),
            CorridorAnalysisConfig(),
        )
        self.assertGreater(near.blockage_confidence, far.blockage_confidence)


class AnalyzerMaskFusionTests(unittest.TestCase):
    def test_mask_overlap_raises_confidence(self) -> None:
        # Depth says clear, but the projected corridor passes
        # through a neighbour mask → fusion raises confidence above
        # depth-only.
        w, h = 640, 480
        depth = _far_depth(w, h)
        mask = np.zeros((h, w), dtype=bool)
        # Block the central column the corridor will project into.
        mask[:, w // 2 - 30 : w // 2 + 30] = True
        intrinsics = _intrinsics(w, h)
        inputs_with_mask = CorridorAnalysisInputs(
            target_point_cam_mm=np.array([0.0, 0.0, 800.0]),
            approach_axis_cam=np.array([0.0, 0.0, 1.0]),
            depth_map=depth,
            intrinsics=intrinsics,
            other_object_masks=(mask,),
        )
        inputs_no_mask = CorridorAnalysisInputs(
            target_point_cam_mm=np.array([0.0, 0.0, 800.0]),
            approach_axis_cam=np.array([0.0, 0.0, 1.0]),
            depth_map=depth,
            intrinsics=intrinsics,
        )
        cfg = CorridorAnalysisConfig(mask_fusion_weight=0.5)
        with_mask = analyze_corridor(inputs_with_mask, cfg)
        no_mask = analyze_corridor(inputs_no_mask, cfg)
        self.assertGreater(
            with_mask.blockage_confidence, no_mask.blockage_confidence
        )

    def test_zero_fusion_weight_ignores_masks(self) -> None:
        # ``mask_fusion_weight=0`` ⇒ analyzer ignores masks entirely
        # so depth-only path is preserved.
        w, h = 640, 480
        depth = _far_depth(w, h)
        mask = np.ones((h, w), dtype=bool)  # everywhere
        intrinsics = _intrinsics(w, h)
        cfg = CorridorAnalysisConfig(mask_fusion_weight=0.0)
        report = analyze_corridor(
            CorridorAnalysisInputs(
                target_point_cam_mm=np.array([0.0, 0.0, 800.0]),
                approach_axis_cam=np.array([0.0, 0.0, 1.0]),
                depth_map=depth,
                intrinsics=intrinsics,
                other_object_masks=(mask,),
            ),
            cfg,
        )
        self.assertEqual(report.mode, CorridorMode.CLEAR)


class AnalysisConfigValidationTests(unittest.TestCase):
    def test_defaults_in_safe_envelope(self) -> None:
        c = CorridorAnalysisConfig()
        self.assertGreater(c.radius_mm, 0.0)
        self.assertGreater(c.step_mm, 0.0)
        self.assertGreater(c.max_distance_mm, 0.0)
        self.assertGreaterEqual(c.mask_fusion_weight, 0.0)
        self.assertLessEqual(c.mask_fusion_weight, 1.0)
        # Partial threshold must sit strictly below blocked threshold.
        self.assertLess(
            c.partial_confidence_threshold, c.blocked_confidence_threshold
        )

    def test_non_positive_radius_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CorridorAnalysisConfig(radius_mm=0.0)
        with self.assertRaises(ValueError):
            CorridorAnalysisConfig(radius_mm=-1.0)

    def test_non_positive_step_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CorridorAnalysisConfig(step_mm=0.0)

    def test_mask_fusion_weight_outside_unit_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CorridorAnalysisConfig(mask_fusion_weight=-0.1)
        with self.assertRaises(ValueError):
            CorridorAnalysisConfig(mask_fusion_weight=1.1)

    def test_partial_threshold_must_be_below_blocked(self) -> None:
        with self.assertRaises(ValueError):
            CorridorAnalysisConfig(
                partial_confidence_threshold=0.8,
                blocked_confidence_threshold=0.7,
            )
        with self.assertRaises(ValueError):
            CorridorAnalysisConfig(
                partial_confidence_threshold=0.5,
                blocked_confidence_threshold=0.5,
            )


if __name__ == "__main__":
    unittest.main()
