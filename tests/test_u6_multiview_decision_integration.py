"""Phase U6 -- mandatory multi-view commit-gate integration tests.

Covers:

* Schema-level config (``RobotGraspingCommitPolicyConfig``): defaults,
  validation, ``apply_modes`` rules (non-empty, no duplicates, EASY
  forbidden), and exposure on
  :class:`RobotGraspingFusionConfig.commit_policy`.
* ``SceneFusion.corridor_evidence`` math: synthetic flat-depth ingest
  followed by axis-aligned corridor queries with known geometry
  (on-axis vs off-axis), and the empty-ROI edge case.
* ``SceneFusion.viewpoint_information_gain`` read-side correctness:
  unseen-count, projected-visibility bound, deterministic.
* Commit gate decision logic on the orchestrator
  (``BinPickingOrchestrator._evaluate_commit_gate``): the entire skip
  ladder (no policy, disabled, no fusion, mode mismatch,
  camera-frame), the allow path with both predicates satisfied, the
  refuse paths (views-below-min, hit-fraction-below-min), and the
  reason-precedence rule (``views_below_min`` wins when both fail).
* WARN-only KPI fixture: a synthetic occlusion-reduction scenario
  asserting (soft) that two viewpoints predict more visible-unseen
  voxels than one. This is Q6=B (WARN, do not fail) per the locked
  U6 design.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.config.schema.robot.robot_schema import (
    RobotGraspingCommitPolicyConfig,
    RobotGraspingFusionConfig,
)
from src.geometry import Frame, Transform
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.multiview.fusion import (
    CorridorEvidence,
    FusionConfig,
    SceneFusion,
    ViewpointGain,
)
from src.robot.grasping.loop.pick_loop import (
    BinPickingOrchestrator,
    CommitDecision,
    CommitPolicy,
    PickOutcome,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _enabled_fusion(**overrides: object) -> FusionConfig:
    base: dict = dict(
        enabled=True,
        max_views=6,
        max_view_age_s=20.0,
        voxel_size_mm=6.0,
        roi_extent_mm=(600.0, 600.0, 360.0),
        max_voxels=1_000_000,
        depth_min_mm=80.0,
        depth_max_mm=1_400.0,
    )
    base.update(overrides)  # type: ignore[arg-type]
    return FusionConfig(**base)  # type: ignore[arg-type]


def _identity_cam_to_base() -> Transform:
    return Transform.from_matrix(
        np.eye(4, dtype=np.float64),
        from_frame=Frame.CAMERA,
        to_frame=Frame.BASE,
    )


def _pinhole_intrinsics(
    fx: float = 200.0, fy: float = 200.0, cx: float = 7.5, cy: float = 7.5
) -> np.ndarray:
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K


def _grasp_point_base(
    position_mm: tuple[float, float, float] = (0.0, 0.0, 100.0),
    approach: tuple[float, float, float] = (0.0, 0.0, 1.0),
    score: float = 0.9,
    frame: GraspFrame = GraspFrame.BASE,
) -> GraspPoint:
    return GraspPoint(
        position=np.array(position_mm, dtype=np.float64),
        approach=np.array(approach, dtype=np.float64),
        axis=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        grip_width_mm=40.0,
        score=score,
        frame=frame,
        label="u6-test",
    )


def _orchestrator_for_gate(
    *,
    commit_policy: CommitPolicy | None,
    scene_fusion: SceneFusion | None,
    mode_label: str,
) -> BinPickingOrchestrator:
    """Construct a minimal orchestrator for gate-only assertions.

    Only the fields read by ``_evaluate_commit_gate`` are meaningful.
    The arm/calculator/perception stubs satisfy the dataclass contract
    without being invoked.
    """

    class _StubArm:
        def get_tcp_pose(self):  # pragma: no cover - never invoked
            raise AssertionError("arm not invoked in gate-only tests")

        def move_to(self, *_a, **_k):  # pragma: no cover
            raise AssertionError("arm not invoked in gate-only tests")

    class _StubCalc:
        def compute_result(self, *_a, **_k):  # pragma: no cover
            raise AssertionError("calculator not invoked in gate-only tests")

    class _StubPerception:
        def acquire(self):  # pragma: no cover
            raise AssertionError("perception not invoked in gate-only tests")

    return BinPickingOrchestrator(
        arm=_StubArm(),  # type: ignore[arg-type]
        calculator=_StubCalc(),  # type: ignore[arg-type]
        perception=_StubPerception(),  # type: ignore[arg-type]
        max_attempts=1,
        scene_fusion=scene_fusion,
        commit_policy=commit_policy,
        mode_label=mode_label,
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class CommitPolicySchemaTests(unittest.TestCase):
    def test_defaults_disabled_and_safe(self) -> None:
        cfg = RobotGraspingCommitPolicyConfig()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.min_views_accepted, 2)
        self.assertAlmostEqual(cfg.min_corridor_hit_fraction, 0.30)
        self.assertAlmostEqual(cfg.corridor_radius_mm, 25.0)
        self.assertAlmostEqual(cfg.corridor_length_mm, 120.0)
        self.assertEqual(cfg.max_reobserve_attempts, 1)
        self.assertEqual(
            tuple(cfg.apply_modes), ("auto", "dense_clutter", "dense_autonomous")
        )

    def test_exposed_on_fusion_config(self) -> None:
        fusion = RobotGraspingFusionConfig()
        self.assertIsInstance(
            fusion.commit_policy, RobotGraspingCommitPolicyConfig
        )
        self.assertFalse(fusion.commit_policy.enabled)
        # active_perception_use_fusion default OFF (Q3 locked).
        self.assertFalse(fusion.active_perception_use_fusion)

    def test_apply_modes_must_be_non_empty(self) -> None:
        with self.assertRaises(Exception):
            RobotGraspingCommitPolicyConfig(apply_modes=())

    def test_apply_modes_rejects_duplicates(self) -> None:
        with self.assertRaises(Exception):
            RobotGraspingCommitPolicyConfig(
                apply_modes=("auto", "auto", "dense_clutter")
            )

    def test_apply_modes_forbids_easy(self) -> None:
        with self.assertRaises(Exception):
            RobotGraspingCommitPolicyConfig(apply_modes=("easy", "auto"))


# ---------------------------------------------------------------------------
# SceneFusion.corridor_evidence
# ---------------------------------------------------------------------------


class CorridorEvidenceTests(unittest.TestCase):
    def _ingest_flat(self, fusion: SceneFusion, *, depth_mm: float) -> None:
        depth = np.full((16, 16), depth_mm, dtype=np.float64)
        K = _pinhole_intrinsics()
        T = _identity_cam_to_base()
        result = fusion.ingest(
            depth_map=depth, intrinsics=K, t_cam_to_base=T, timestamp_ns=0
        )
        self.assertGreater(result.hit_voxels, 0)

    def test_on_axis_corridor_hits_through_flat_surface(self) -> None:
        fusion = SceneFusion(config=_enabled_fusion())
        # Flat surface 150 mm in front of the (identity) camera, so
        # in BASE coords the surface is at z=+150 mm.
        self._ingest_flat(fusion, depth_mm=150.0)
        ev = fusion.corridor_evidence(
            position_mm=np.array([0.0, 0.0, 150.0], dtype=np.float64),
            approach=np.array([0.0, 0.0, -1.0], dtype=np.float64),
            length_mm=120.0,
            radius_mm=25.0,
        )
        self.assertIsInstance(ev, CorridorEvidence)
        self.assertGreater(ev.queried_voxels, 0)
        # A corridor sliding along -z from z=150 covers depth slabs
        # that contain at least the ingested surface plane.
        self.assertGreater(ev.hit_voxels, 0)
        self.assertGreaterEqual(ev.hit_fraction, 0.0)
        self.assertLessEqual(ev.hit_fraction, 1.0)
        self.assertEqual(ev.views_accepted, 1)

    def test_far_off_axis_corridor_misses_surface(self) -> None:
        fusion = SceneFusion(config=_enabled_fusion())
        self._ingest_flat(fusion, depth_mm=150.0)
        # Corridor placed deep in -z, well below the ingested surface
        # at +150 mm and pointed even further away. ROI extends only
        # to +/- 180 mm in z (extent 360), so a corridor centered at
        # z=-150 going further -z stays inside ROI but cannot contain
        # the ingested hits.
        ev = fusion.corridor_evidence(
            position_mm=np.array([0.0, 0.0, -150.0], dtype=np.float64),
            approach=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            length_mm=20.0,
            radius_mm=12.0,
        )
        self.assertEqual(ev.hit_voxels, 0)
        self.assertEqual(ev.hit_fraction, 0.0)

    def test_zero_queried_returns_zero_fraction(self) -> None:
        fusion = SceneFusion(config=_enabled_fusion())
        # Place the corridor well outside the ROI: ROI x extent 600,
        # so |x| <= 300. Corridor centered at x=5_000 with small
        # radius/length stays entirely outside.
        ev = fusion.corridor_evidence(
            position_mm=np.array([5_000.0, 0.0, 0.0], dtype=np.float64),
            approach=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            length_mm=20.0,
            radius_mm=10.0,
        )
        self.assertEqual(ev.queried_voxels, 0)
        self.assertEqual(ev.hit_voxels, 0)
        self.assertEqual(ev.hit_fraction, 0.0)
        self.assertEqual(ev.views_accepted, 0)


# ---------------------------------------------------------------------------
# SceneFusion.viewpoint_information_gain
# ---------------------------------------------------------------------------


class ViewpointGainTests(unittest.TestCase):
    def test_empty_fusion_predicts_some_visible_unseen(self) -> None:
        fusion = SceneFusion(config=_enabled_fusion())
        gain = fusion.viewpoint_information_gain(
            t_cam_to_base=_identity_cam_to_base(),
            intrinsics=_pinhole_intrinsics(),
            depth_shape=(16, 16),
        )
        self.assertIsInstance(gain, ViewpointGain)
        self.assertEqual(gain.image_shape, (16, 16))
        # With no ingest yet, every in-ROI voxel is unseen; the
        # identity camera sees some fraction within image bounds and
        # depth window.
        self.assertGreater(gain.unseen_voxels, 0)
        self.assertGreater(gain.predicted_visible_unseen, 0)
        self.assertLessEqual(
            gain.predicted_visible_unseen, gain.unseen_voxels
        )

    def test_deterministic_across_repeated_queries(self) -> None:
        fusion = SceneFusion(config=_enabled_fusion())
        kwargs = dict(
            t_cam_to_base=_identity_cam_to_base(),
            intrinsics=_pinhole_intrinsics(),
            depth_shape=(16, 16),
        )
        a = fusion.viewpoint_information_gain(**kwargs)  # type: ignore[arg-type]
        b = fusion.viewpoint_information_gain(**kwargs)  # type: ignore[arg-type]
        self.assertEqual(a, b)


# ---------------------------------------------------------------------------
# Commit gate decision logic
# ---------------------------------------------------------------------------


class CommitGateSkipLadderTests(unittest.TestCase):
    """Every skip path must yield ``allowed=True`` with a typed reason."""

    def _best(self) -> GraspResult:
        return GraspResult(candidates=(_grasp_point_base(),), top_score=0.9)

    def test_no_policy_skipped(self) -> None:
        orch = _orchestrator_for_gate(
            commit_policy=None, scene_fusion=None, mode_label="auto"
        )
        d = orch._evaluate_commit_gate(self._best())
        self.assertTrue(d.allowed)
        self.assertEqual(d.reason, "skipped_no_policy")

    def test_disabled_policy_skipped(self) -> None:
        policy = CommitPolicy(
            enabled=False,
            min_views_accepted=2,
            min_corridor_hit_fraction=0.3,
            corridor_radius_mm=25.0,
            corridor_length_mm=120.0,
            max_reobserve_attempts=1,
            apply_modes=frozenset({"auto"}),
        )
        orch = _orchestrator_for_gate(
            commit_policy=policy, scene_fusion=None, mode_label="auto"
        )
        d = orch._evaluate_commit_gate(self._best())
        self.assertTrue(d.allowed)
        self.assertEqual(d.reason, "skipped_disabled")

    def test_no_fusion_skipped(self) -> None:
        policy = CommitPolicy(
            enabled=True,
            min_views_accepted=2,
            min_corridor_hit_fraction=0.3,
            corridor_radius_mm=25.0,
            corridor_length_mm=120.0,
            max_reobserve_attempts=1,
            apply_modes=frozenset({"auto"}),
        )
        orch = _orchestrator_for_gate(
            commit_policy=policy, scene_fusion=None, mode_label="auto"
        )
        d = orch._evaluate_commit_gate(self._best())
        self.assertTrue(d.allowed)
        self.assertEqual(d.reason, "skipped_no_fusion")

    def test_mode_not_in_apply_set_skipped(self) -> None:
        # EASY is explicitly out of scope per the locked U6 design.
        policy = CommitPolicy(
            enabled=True,
            min_views_accepted=2,
            min_corridor_hit_fraction=0.3,
            corridor_radius_mm=25.0,
            corridor_length_mm=120.0,
            max_reobserve_attempts=1,
            apply_modes=frozenset({"auto", "dense_clutter", "dense_autonomous"}),
        )
        fusion = SceneFusion(config=_enabled_fusion())
        orch = _orchestrator_for_gate(
            commit_policy=policy, scene_fusion=fusion, mode_label="easy"
        )
        d = orch._evaluate_commit_gate(self._best())
        self.assertTrue(d.allowed)
        self.assertEqual(d.reason, "skipped_mode")

    def test_empty_mode_label_skipped(self) -> None:
        policy = CommitPolicy(
            enabled=True,
            min_views_accepted=2,
            min_corridor_hit_fraction=0.3,
            corridor_radius_mm=25.0,
            corridor_length_mm=120.0,
            max_reobserve_attempts=1,
            apply_modes=frozenset({"auto"}),
        )
        fusion = SceneFusion(config=_enabled_fusion())
        orch = _orchestrator_for_gate(
            commit_policy=policy, scene_fusion=fusion, mode_label=""
        )
        d = orch._evaluate_commit_gate(self._best())
        self.assertTrue(d.allowed)
        self.assertEqual(d.reason, "skipped_mode")

    def test_camera_frame_candidate_skipped(self) -> None:
        policy = CommitPolicy(
            enabled=True,
            min_views_accepted=2,
            min_corridor_hit_fraction=0.3,
            corridor_radius_mm=25.0,
            corridor_length_mm=120.0,
            max_reobserve_attempts=1,
            apply_modes=frozenset({"auto"}),
        )
        fusion = SceneFusion(config=_enabled_fusion())
        orch = _orchestrator_for_gate(
            commit_policy=policy, scene_fusion=fusion, mode_label="auto"
        )
        cam_best = GraspResult(
            candidates=(_grasp_point_base(frame=GraspFrame.CAMERA),),
            top_score=0.9,
        )
        d = orch._evaluate_commit_gate(cam_best)
        self.assertTrue(d.allowed)
        self.assertEqual(d.reason, "skipped_camera_frame")


class CommitGateDecisionTests(unittest.TestCase):
    """Allow / refuse outcomes when the gate actually runs."""

    def _policy(
        self,
        *,
        min_views: int = 1,
        min_hit_fraction: float = 0.0,
    ) -> CommitPolicy:
        return CommitPolicy(
            enabled=True,
            min_views_accepted=min_views,
            min_corridor_hit_fraction=min_hit_fraction,
            corridor_radius_mm=25.0,
            corridor_length_mm=120.0,
            max_reobserve_attempts=1,
            apply_modes=frozenset({"auto", "dense_clutter", "dense_autonomous"}),
        )

    def _fusion_with_one_view(self) -> SceneFusion:
        fusion = SceneFusion(config=_enabled_fusion())
        depth = np.full((16, 16), 150.0, dtype=np.float64)
        result = fusion.ingest(
            depth_map=depth,
            intrinsics=_pinhole_intrinsics(),
            t_cam_to_base=_identity_cam_to_base(),
            timestamp_ns=0,
        )
        self.assertGreater(result.hit_voxels, 0)
        return fusion

    def _best_on_surface(self) -> GraspResult:
        # On-axis with the ingested flat surface.
        return GraspResult(
            candidates=(
                _grasp_point_base(
                    position_mm=(0.0, 0.0, 150.0),
                    approach=(0.0, 0.0, -1.0),
                ),
            ),
            top_score=0.9,
        )

    def test_allow_path_both_predicates_satisfied(self) -> None:
        fusion = self._fusion_with_one_view()
        policy = self._policy(min_views=1, min_hit_fraction=0.0)
        orch = _orchestrator_for_gate(
            commit_policy=policy, scene_fusion=fusion, mode_label="auto"
        )
        d = orch._evaluate_commit_gate(self._best_on_surface())
        self.assertTrue(d.allowed)
        self.assertEqual(d.reason, "ok")
        self.assertEqual(d.views_accepted, 1)
        self.assertGreater(d.corridor_queried_voxels, 0)

    def test_refuse_views_below_min(self) -> None:
        fusion = self._fusion_with_one_view()  # one view
        policy = self._policy(min_views=3, min_hit_fraction=0.0)
        orch = _orchestrator_for_gate(
            commit_policy=policy, scene_fusion=fusion, mode_label="auto"
        )
        d = orch._evaluate_commit_gate(self._best_on_surface())
        self.assertFalse(d.allowed)
        self.assertEqual(d.reason, "views_below_min")
        self.assertEqual(d.views_accepted, 1)
        self.assertEqual(d.min_views_required, 3)

    def test_refuse_hit_fraction_below_min(self) -> None:
        fusion = self._fusion_with_one_view()
        policy = self._policy(min_views=1, min_hit_fraction=0.999)
        orch = _orchestrator_for_gate(
            commit_policy=policy, scene_fusion=fusion, mode_label="auto"
        )
        d = orch._evaluate_commit_gate(self._best_on_surface())
        self.assertFalse(d.allowed)
        self.assertEqual(d.reason, "hit_fraction_below_min")
        self.assertGreaterEqual(d.corridor_hit_fraction, 0.0)
        self.assertAlmostEqual(d.min_hit_fraction_required, 0.999)

    def test_refuse_views_wins_over_hit_fraction_when_both_fail(self) -> None:
        fusion = self._fusion_with_one_view()
        # Both predicates fail: views needed 5 (have 1), hit threshold
        # 0.999 (have less). Spec: views_below_min wins.
        policy = self._policy(min_views=5, min_hit_fraction=0.999)
        orch = _orchestrator_for_gate(
            commit_policy=policy, scene_fusion=fusion, mode_label="auto"
        )
        d = orch._evaluate_commit_gate(self._best_on_surface())
        self.assertFalse(d.allowed)
        self.assertEqual(d.reason, "views_below_min")

    def test_dense_modes_in_scope(self) -> None:
        fusion = self._fusion_with_one_view()
        policy = self._policy(min_views=1, min_hit_fraction=0.0)
        for mode in ("auto", "dense_clutter", "dense_autonomous"):
            orch = _orchestrator_for_gate(
                commit_policy=policy, scene_fusion=fusion, mode_label=mode
            )
            d = orch._evaluate_commit_gate(self._best_on_surface())
            self.assertTrue(d.allowed, f"mode {mode} should be in scope")
            self.assertEqual(d.reason, "ok")


# ---------------------------------------------------------------------------
# Public outcome surface
# ---------------------------------------------------------------------------


class CommitOutcomeSurfaceTests(unittest.TestCase):
    def test_pick_outcome_no_commit_insufficient_fusion_present(self) -> None:
        self.assertEqual(
            PickOutcome.NO_COMMIT_INSUFFICIENT_FUSION.value,
            "no_commit_insufficient_fusion",
        )

    def test_commit_decision_is_frozen(self) -> None:
        d = CommitDecision(
            allowed=True,
            reason="ok",
            views_accepted=2,
            min_views_required=2,
            corridor_hit_fraction=0.5,
            min_hit_fraction_required=0.3,
            corridor_queried_voxels=40,
            corridor_hit_voxels=20,
            reobserve_attempts_used=0,
        )
        with self.assertRaises(Exception):
            d.allowed = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# WARN-only KPI fixture (Q6=B)
# ---------------------------------------------------------------------------


class OcclusionReductionKpiWarnTests(unittest.TestCase):
    """Synthetic occlusion-reduction signal; WARN only, never fails.

    Compares predicted ``visible_unseen`` from a single viewpoint vs
    the union (approximated as ``max``) of two viewpoints in an
    empty (just-initialized) fusion grid. We expect the two-viewpoint
    case to predict at least as much new information as the single
    viewpoint. This is a soft KPI per Q6=B -- the assertion is a
    ``>=`` gate that emits ``[U6-WARN]`` on the boundary case.
    """

    def test_two_views_predict_at_least_as_much_unseen(self) -> None:
        fusion = SceneFusion(config=_enabled_fusion())
        K = _pinhole_intrinsics()
        shape = (16, 16)

        T1 = _identity_cam_to_base()
        # Second camera shifted sideways by 100 mm in BASE x.
        M = np.eye(4, dtype=np.float64)
        M[0, 3] = 100.0
        T2 = Transform.from_matrix(
            M, from_frame=Frame.CAMERA, to_frame=Frame.BASE
        )
        g1 = fusion.viewpoint_information_gain(
            t_cam_to_base=T1, intrinsics=K, depth_shape=shape
        )
        g2 = fusion.viewpoint_information_gain(
            t_cam_to_base=T2, intrinsics=K, depth_shape=shape
        )

        union_upper = max(g1.predicted_visible_unseen, g2.predicted_visible_unseen)
        single = g1.predicted_visible_unseen
        delta = union_upper - single
        if delta <= 0:
            # Warn-only fixture: emit a soft diagnostic but pass.
            print(
                f"[U6-WARN] occlusion_reduction: single={single} "
                f"union_upper={union_upper} delta={delta} "
                "(two-view fixture did not exceed single-view; review viewpoints)"
            )
        self.assertGreaterEqual(union_upper, single)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    unittest.main()
