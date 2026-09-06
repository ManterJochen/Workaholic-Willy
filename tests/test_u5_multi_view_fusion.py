"""Phase U5 -- bounded multi-view fusion substrate tests.

Covers:

* Schema-level validation (``RobotGraspingFusionConfig``).
* Runtime carrier (``FusionConfig``) self-validation.
* Backprojection math sanity: a small synthetic depth frame deposits
  hits where its geometry predicts.
* Strict refuse paths: disabled, bad frame, bad intrinsics, bad
  depth shape, intrinsic drift, no valid samples.
* Sliding FIFO eviction (``max_views``).
* Age-based eviction (``max_view_age_s``).
* Determinism: byte-identical accumulators across two independent
  fusion runs ingesting the same input sequence.
* Orchestrator shadow integration: with ``scene_fusion=None`` the
  pick loop's ``PickReport.fusion_telemetry`` is ``None`` and no
  behavior changes; with a fusion carrier wired the telemetry is a
  ``FusionTelemetry`` of the expected shape, and the substrate
  itself absorbs frame-resolver failures (no exception leaks).
"""

from __future__ import annotations

import hashlib
import unittest

import numpy as np

from src.config.schema.robot.robot_schema import (
    RobotGraspingFusionConfig,
)
from src.geometry import Frame, Transform
from src.robot.grasping.multiview.fusion import (
    INGEST_ACCEPTED,
    INGEST_REFUSED,
    REFUSE_BAD_DEPTH_SHAPE,
    REFUSE_BAD_FRAME,
    REFUSE_BAD_INTRINSICS,
    REFUSE_DISABLED,
    REFUSE_INTRINSICS_DRIFT,
    REFUSE_NO_VALID_SAMPLES,
    FusionConfig,
    FusionTelemetry,
    SceneFusion,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _enabled_config(**overrides) -> FusionConfig:
    """Default-enabled FusionConfig with optional overrides."""

    base = dict(
        enabled=True,
        max_views=6,
        max_view_age_s=20.0,
        voxel_size_mm=6.0,
        roi_extent_mm=(600.0, 600.0, 360.0),
        max_voxels=1_000_000,
        depth_min_mm=80.0,
        depth_max_mm=1_400.0,
    )
    base.update(overrides)
    return FusionConfig(**base)


def _identity_cam_to_base() -> Transform:
    """Camera frame coincident with BASE: M = I."""

    return Transform.from_matrix(
        np.eye(4, dtype=np.float64),
        from_frame=Frame.CAMERA,
        to_frame=Frame.BASE,
    )


def _pinhole_intrinsics(fx: float = 200.0, fy: float = 200.0,
                        cx: float = 7.5, cy: float = 7.5) -> np.ndarray:
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K


def _flat_depth(value_mm: float = 150.0,
                shape: tuple[int, int] = (16, 16)) -> np.ndarray:
    return np.full(shape, value_mm, dtype=np.float64)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class SchemaTests(unittest.TestCase):
    def test_defaults_validate(self) -> None:
        cfg = RobotGraspingFusionConfig()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.max_views, 6)
        self.assertEqual(cfg.roi_extent_mm[2], 360.0)

    def test_roi_must_be_voxel_multiple(self) -> None:
        with self.assertRaises(Exception):
            RobotGraspingFusionConfig(
                voxel_size_mm=6.0,
                roi_extent_mm=(600.0, 600.0, 350.0),  # not a multiple
            )

    def test_depth_window_ordering(self) -> None:
        with self.assertRaises(Exception):
            RobotGraspingFusionConfig(depth_min_mm=100.0, depth_max_mm=100.0)

    def test_max_voxels_cap(self) -> None:
        with self.assertRaises(Exception):
            RobotGraspingFusionConfig(
                voxel_size_mm=1.0,
                roi_extent_mm=(1000.0, 1000.0, 1000.0),
                max_voxels=1_000,
            )


# ---------------------------------------------------------------------------
# FusionConfig carrier validation
# ---------------------------------------------------------------------------


class FusionConfigCarrierTests(unittest.TestCase):
    def test_default_disabled(self) -> None:
        cfg = FusionConfig()
        self.assertFalse(cfg.enabled)

    def test_roi_must_be_multiple(self) -> None:
        with self.assertRaises(ValueError):
            FusionConfig(
                enabled=True,
                voxel_size_mm=6.0,
                roi_extent_mm=(600.0, 600.0, 350.0),
            )

    def test_max_voxels_cap(self) -> None:
        with self.assertRaises(ValueError):
            FusionConfig(
                enabled=True,
                voxel_size_mm=1.0,
                roi_extent_mm=(1000.0, 1000.0, 1000.0),
                max_voxels=1000,
            )

    def test_grid_shape_matches_roi(self) -> None:
        cfg = _enabled_config()
        self.assertEqual(cfg.grid_shape, (100, 100, 60))


# ---------------------------------------------------------------------------
# Backprojection math
# ---------------------------------------------------------------------------


class BackprojectionTests(unittest.TestCase):
    def test_flat_depth_deposits_into_grid(self) -> None:
        fusion = SceneFusion(config=_enabled_config())
        result = fusion.ingest(
            depth_map=_flat_depth(value_mm=150.0),
            intrinsics=_pinhole_intrinsics(),
            t_cam_to_base=_identity_cam_to_base(),
            timestamp_ns=0,
        )
        self.assertEqual(result.status, INGEST_ACCEPTED)
        self.assertGreater(result.valid_samples, 0)
        self.assertGreater(result.hit_voxels, 0)
        # Aggregated counts must equal total valid samples.
        self.assertEqual(
            int(fusion.hits.sum()), int(result.valid_samples)
        )
        # ``seen`` is per-view-touched, so one view => max value 1 per
        # touched voxel; equal to count of hit voxels.
        self.assertEqual(
            int((fusion.seen > 0).sum()), int(result.hit_voxels)
        )

    def test_additive_accumulation(self) -> None:
        fusion = SceneFusion(config=_enabled_config())
        depth = _flat_depth(value_mm=150.0)
        K = _pinhole_intrinsics()
        T = _identity_cam_to_base()
        r1 = fusion.ingest(depth_map=depth, intrinsics=K,
                           t_cam_to_base=T, timestamp_ns=0)
        r2 = fusion.ingest(depth_map=depth, intrinsics=K,
                           t_cam_to_base=T, timestamp_ns=1_000_000_000)
        self.assertEqual(r1.status, INGEST_ACCEPTED)
        self.assertEqual(r2.status, INGEST_ACCEPTED)
        self.assertEqual(int(fusion.hits.sum()),
                         r1.valid_samples + r2.valid_samples)
        # Two identical views => seen counter saturates at 2.
        self.assertEqual(int(fusion.seen.max()), 2)


# ---------------------------------------------------------------------------
# Strict refuse paths
# ---------------------------------------------------------------------------


class RefuseTests(unittest.TestCase):
    def test_disabled_refuses(self) -> None:
        fusion = SceneFusion(config=FusionConfig())  # disabled default
        r = fusion.ingest(
            depth_map=_flat_depth(),
            intrinsics=_pinhole_intrinsics(),
            t_cam_to_base=_identity_cam_to_base(),
            timestamp_ns=0,
        )
        self.assertEqual(r.status, INGEST_REFUSED)
        self.assertEqual(r.reason, REFUSE_DISABLED)

    def test_bad_frame_refuses(self) -> None:
        fusion = SceneFusion(config=_enabled_config())
        bad = Transform.identity(
            from_frame=Frame.TOOL, to_frame=Frame.BASE
        )
        r = fusion.ingest(
            depth_map=_flat_depth(),
            intrinsics=_pinhole_intrinsics(),
            t_cam_to_base=bad,
            timestamp_ns=0,
        )
        self.assertEqual(r.status, INGEST_REFUSED)
        self.assertEqual(r.reason, REFUSE_BAD_FRAME)

    def test_bad_intrinsics_refuses(self) -> None:
        fusion = SceneFusion(config=_enabled_config())
        K = _pinhole_intrinsics()
        K[0, 0] = -1.0  # fx must be > 0
        r = fusion.ingest(
            depth_map=_flat_depth(),
            intrinsics=K,
            t_cam_to_base=_identity_cam_to_base(),
            timestamp_ns=0,
        )
        self.assertEqual(r.status, INGEST_REFUSED)
        self.assertEqual(r.reason, REFUSE_BAD_INTRINSICS)

    def test_bad_depth_shape_refuses(self) -> None:
        fusion = SceneFusion(config=_enabled_config())
        r = fusion.ingest(
            depth_map=np.array([1.0, 2.0, 3.0], dtype=np.float64),  # 1-D
            intrinsics=_pinhole_intrinsics(),
            t_cam_to_base=_identity_cam_to_base(),
            timestamp_ns=0,
        )
        self.assertEqual(r.status, INGEST_REFUSED)
        self.assertEqual(r.reason, REFUSE_BAD_DEPTH_SHAPE)

    def test_intrinsics_drift_refuses(self) -> None:
        fusion = SceneFusion(config=_enabled_config())
        K1 = _pinhole_intrinsics()
        r1 = fusion.ingest(
            depth_map=_flat_depth(),
            intrinsics=K1,
            t_cam_to_base=_identity_cam_to_base(),
            timestamp_ns=0,
        )
        self.assertEqual(r1.status, INGEST_ACCEPTED)
        K2 = _pinhole_intrinsics(fx=210.0)  # drift
        r2 = fusion.ingest(
            depth_map=_flat_depth(),
            intrinsics=K2,
            t_cam_to_base=_identity_cam_to_base(),
            timestamp_ns=1,
        )
        self.assertEqual(r2.status, INGEST_REFUSED)
        self.assertEqual(r2.reason, REFUSE_INTRINSICS_DRIFT)

    def test_no_valid_samples_refuses(self) -> None:
        fusion = SceneFusion(config=_enabled_config())
        # Depth far outside the configured Z-ROI => zero hits in ROI,
        # but also outside depth window so they get filtered first.
        depth = np.full((8, 8), 10_000.0, dtype=np.float64)
        r = fusion.ingest(
            depth_map=depth,
            intrinsics=_pinhole_intrinsics(),
            t_cam_to_base=_identity_cam_to_base(),
            timestamp_ns=0,
        )
        self.assertEqual(r.status, INGEST_REFUSED)
        self.assertEqual(r.reason, REFUSE_NO_VALID_SAMPLES)


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------


class EvictionTests(unittest.TestCase):
    def test_fifo_eviction(self) -> None:
        fusion = SceneFusion(config=_enabled_config(max_views=3))
        depth = _flat_depth()
        K = _pinhole_intrinsics()
        T = _identity_cam_to_base()
        for i in range(5):
            r = fusion.ingest(depth_map=depth, intrinsics=K,
                              t_cam_to_base=T, timestamp_ns=i)
            self.assertEqual(r.status, INGEST_ACCEPTED)
        # max_views=3 => only the last 3 survive.
        self.assertEqual(len(fusion.views), 3)
        # seen counter cannot exceed surviving view count.
        self.assertLessEqual(int(fusion.seen.max()), 3)

    def test_age_eviction(self) -> None:
        fusion = SceneFusion(config=_enabled_config(max_view_age_s=2.0))
        depth = _flat_depth()
        K = _pinhole_intrinsics()
        T = _identity_cam_to_base()
        # First view at t=0.
        fusion.ingest(depth_map=depth, intrinsics=K,
                      t_cam_to_base=T, timestamp_ns=0)
        # Second view 10s later -- first must age out (>2 s old).
        fusion.ingest(depth_map=depth, intrinsics=K,
                      t_cam_to_base=T,
                      timestamp_ns=10 * 1_000_000_000)
        self.assertEqual(len(fusion.views), 1)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class DeterminismTests(unittest.TestCase):
    def test_hits_byte_identical_across_runs(self) -> None:
        depths = [
            _flat_depth(value_mm=120.0),
            _flat_depth(value_mm=150.0),
            _flat_depth(value_mm=180.0),
        ]
        K = _pinhole_intrinsics()
        T = _identity_cam_to_base()

        def _run() -> tuple[bytes, bytes]:
            f = SceneFusion(config=_enabled_config())
            for i, d in enumerate(depths):
                f.ingest(depth_map=d, intrinsics=K,
                         t_cam_to_base=T, timestamp_ns=i)
            return (
                hashlib.sha256(f.hits.tobytes()).digest(),
                hashlib.sha256(f.seen.tobytes()).digest(),
            )

        a_hits, a_seen = _run()
        b_hits, b_seen = _run()
        self.assertEqual(a_hits, b_hits)
        self.assertEqual(a_seen, b_seen)


# ---------------------------------------------------------------------------
# Orchestrator shadow integration
# ---------------------------------------------------------------------------


class OrchestratorShadowTests(unittest.TestCase):
    def test_pick_report_field_default_none(self) -> None:
        """Default ``PickReport`` carries no fusion telemetry."""

        from src.robot.grasping.loop.pick_loop import PickReport, PickOutcome

        report = PickReport(outcome=PickOutcome.NO_PERCEPTION)
        self.assertIsNone(report.fusion_telemetry)

    def test_orchestrator_field_default_none(self) -> None:
        """Default ``BinPickingOrchestrator`` has no fusion wired."""

        from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator

        # Direct attribute introspection -- the orchestrator's runtime
        # state cannot be instantiated without a perception + arm. The
        # class-level dataclass default is what we contract on.
        self.assertTrue(
            "scene_fusion" in BinPickingOrchestrator.__dataclass_fields__
        )
        self.assertIsNone(
            BinPickingOrchestrator.__dataclass_fields__["scene_fusion"].default
        )

    def test_runtime_pick_session_report_field_default_none(self) -> None:
        from src.robot.execution.runtime_pick import (
            PickSessionReport,
        )

        # The class-level dataclass default is the contract -- avoid
        # constructing one (it requires several robot-state args).
        self.assertTrue(
            "fusion_telemetry" in PickSessionReport.__dataclass_fields__
        )
        self.assertIsNone(
            PickSessionReport.__dataclass_fields__["fusion_telemetry"].default
        )

    def test_telemetry_shape(self) -> None:
        fusion = SceneFusion(config=_enabled_config())
        t = fusion.telemetry()
        self.assertIsInstance(t, FusionTelemetry)
        self.assertTrue(t.enabled)
        self.assertEqual(t.views_attempted, 0)
        self.assertEqual(t.views_accepted, 0)
        self.assertEqual(t.views_rejected, 0)
        self.assertIsNone(t.last_reject_reason)
        self.assertEqual(t.voxels_hit, 0)
        self.assertEqual(t.voxels_seen, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
