"""Tests for the active-perception bin-picking orchestrator."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.geometry import Frame, Pose, Transform
from src.robot.grasping import GraspCalculator
from src.robot.grasping.motion.execution_policy import PolicyOutcome, PolicyReport
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.multiview.fusion import FusionConfig, SceneFusion
from src.robot.grasping.loop.pick_loop import (
    BinPickingOrchestrator,
    CommitPolicy,
    LateralOffsetViewpointPlanner,
    PerceptionFrame,
    PickOutcome,
)
from src.robot.grasping.motion.trajectory_safety import ApproachPathPolicy
from tests._helpers import (
    _FakeArm,
    _FakePerception,
    _ScriptedCalculator,
    _perception_frame,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _grasp_point(score: float = 0.9) -> GraspPoint:
    return GraspPoint(
        position=np.array([100.0, 50.0, 400.0]),
        approach=np.array([0.0, 0.0, 1.0]),
        axis=np.array([1.0, 0.0, 0.0]),
        grip_width_mm=40.0,
        score=score,
        frame=GraspFrame.BASE,
        label="test",
    )


def _success_result() -> GraspResult:
    return GraspResult(
        candidates=(_grasp_point(),),
        reasons=(),
        top_score=0.9,
    )


def _rescan_result() -> GraspResult:
    return GraspResult(
        candidates=(),
        reasons=(GraspFailureReason.RESCAN_RECOMMENDED,),
        top_score=0.0,
    )


def _active_perception_result() -> GraspResult:
    return GraspResult(
        candidates=(),
        reasons=(GraspFailureReason.ACTIVE_PERCEPTION_RECOMMENDED,),
        top_score=0.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class BinPickingOrchestratorTests(unittest.TestCase):
    def test_execute_on_first_success(self) -> None:
        arm = _FakeArm()
        calc = _ScriptedCalculator([_success_result()])
        perception = _FakePerception([_perception_frame()])
        orch = BinPickingOrchestrator(
            arm=arm,  # type: ignore[arg-type]
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
            max_attempts=3,
        )
        report = orch.run()
        self.assertEqual(report.outcome, PickOutcome.EXECUTED)
        self.assertEqual(len(report.attempts), 1)
        self.assertEqual(report.attempts[0].action, "executed")
        # 4 approach waypoints + 1 retreat = 5 motion commands.
        self.assertEqual(len(arm.move_to_calls), 5)

    def test_rescan_then_success(self) -> None:
        arm = _FakeArm()
        calc = _ScriptedCalculator([_rescan_result(), _success_result()])
        perception = _FakePerception([_perception_frame()])
        orch = BinPickingOrchestrator(
            arm=arm,  # type: ignore[arg-type]
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
            max_attempts=3,
        )
        report = orch.run()
        self.assertEqual(report.outcome, PickOutcome.EXECUTED)
        self.assertEqual(len(report.attempts), 2)
        self.assertEqual(report.attempts[0].action, "rescan")
        self.assertEqual(report.attempts[1].action, "executed")
        self.assertEqual(perception.calls, 2)

    def test_active_perception_drives_viewpoint(self) -> None:
        arm = _FakeArm()
        calc = _ScriptedCalculator([_active_perception_result(), _success_result()])
        perception = _FakePerception([_perception_frame()])
        orch = BinPickingOrchestrator(
            arm=arm,  # type: ignore[arg-type]
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
            viewpoint_planner=LateralOffsetViewpointPlanner(offset_mm=50.0),
            max_attempts=3,
        )
        report = orch.run()
        self.assertEqual(report.outcome, PickOutcome.EXECUTED)
        self.assertEqual(len(report.viewpoints_visited), 1)
        # Viewpoint move + 4 approach + 1 retreat = 6 commands.
        self.assertEqual(len(arm.move_to_calls), 6)
        self.assertEqual(arm.move_to_calls[0].label, "viewpoint_00")

    def test_exhausts_when_viewpoint_planner_returns_none(self) -> None:
        arm = _FakeArm()
        calc = _ScriptedCalculator([_active_perception_result()])
        perception = _FakePerception([_perception_frame()])
        planner = LateralOffsetViewpointPlanner(offset_mm=50.0, max_viewpoints=0)
        orch = BinPickingOrchestrator(
            arm=arm,  # type: ignore[arg-type]
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
            viewpoint_planner=planner,
            max_attempts=2,
        )
        report = orch.run()
        self.assertEqual(report.outcome, PickOutcome.RELOCATED_EXHAUSTED)
        self.assertEqual(len(arm.move_to_calls), 0)

    def test_no_segmentations_yields_no_perception(self) -> None:
        arm = _FakeArm()
        calc = _ScriptedCalculator([_success_result()])
        empty_frame = PerceptionFrame(
            depth_map=np.full((16, 16), 500.0, dtype=np.float64),
            intrinsics=np.eye(3),
            segmentations=(),
        )
        perception = _FakePerception([empty_frame])
        orch = BinPickingOrchestrator(
            arm=arm,  # type: ignore[arg-type]
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
        )
        report = orch.run()
        self.assertEqual(report.outcome, PickOutcome.NO_PERCEPTION)
        self.assertEqual(len(arm.move_to_calls), 0)

    def test_real_calculator_smoke(self) -> None:
        """A real GraspCalculator + the orchestrator must not raise."""
        arm = _FakeArm()
        perception = _FakePerception([_perception_frame()])
        orch = BinPickingOrchestrator(
            arm=arm,  # type: ignore[arg-type]
            calculator=GraspCalculator(
                min_grip_width_mm=1.0,
                max_grip_width_mm=200.0,
                max_candidates=3,
            ),
            perception=perception,
            max_attempts=2,
            dense_sampling=False,
        )
        report = orch.run()
        # We don't care about the outcome here; we care that the
        # plumbing (calculator <-> orchestrator <-> arm) is wired.
        self.assertIn(
            report.outcome,
            {
                PickOutcome.EXECUTED,
                PickOutcome.RESCANNED_EXHAUSTED,
                PickOutcome.RELOCATED_EXHAUSTED,
                PickOutcome.NO_PERCEPTION,
            },
        )


class LateralOffsetViewpointPlannerTests(unittest.TestCase):
    def test_alternates_cardinal_offsets(self) -> None:
        planner = LateralOffsetViewpointPlanner(offset_mm=50.0, max_viewpoints=4)
        start = Pose(
            position_mm=np.array([100.0, 200.0, 500.0]),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            frame=Frame.BASE,
        )
        history: list[Pose] = []
        for _ in range(4):
            next_pose = planner.next_viewpoint(current_tcp=start, history=history)
            self.assertIsNotNone(next_pose)
            history.append(next_pose)  # type: ignore[arg-type]
        self.assertIsNone(planner.next_viewpoint(current_tcp=start, history=history))
        # First two viewpoints differ by 100 mm along x (±50 each from start).
        self.assertAlmostEqual(history[0].position_mm[0] - 100.0, +50.0, places=6)
        self.assertAlmostEqual(history[1].position_mm[0] - 100.0, -50.0, places=6)


class CommitGateReobserveRelocateTests(unittest.TestCase):
    """G5 Stufe B: the U6 commit-gate reobserve loop relocates the camera to a DIVERSE viewpoint when a
    viewpoint_planner is wired, instead of re-acquiring the same static frame (which would only inflate
    views_accepted without growing the corridor evidence)."""

    @staticmethod
    def _fusion() -> SceneFusion:
        return SceneFusion(config=FusionConfig(
            enabled=True, max_views=6, max_view_age_s=20.0, voxel_size_mm=6.0,
            roi_extent_mm=(600.0, 600.0, 360.0), max_voxels=1_000_000,
            depth_min_mm=80.0, depth_max_mm=1_400.0,
        ))

    @staticmethod
    def _refusing_policy() -> CommitPolicy:
        # min_views_accepted high -> the gate always refuses (views_below_min) -> reobserve fires.
        return CommitPolicy(
            enabled=True, min_views_accepted=99, min_corridor_hit_fraction=0.0,
            corridor_radius_mm=25.0, corridor_length_mm=120.0, max_reobserve_attempts=2,
            apply_modes=frozenset({"auto"}),
        )

    def _orch(self, *, planner: LateralOffsetViewpointPlanner | None) -> BinPickingOrchestrator:
        return BinPickingOrchestrator(
            arm=_FakeArm(),  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_success_result()]),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            viewpoint_planner=planner,
            max_attempts=3,
            scene_fusion=self._fusion(),
            commit_policy=self._refusing_policy(),
            mode_label="auto",
        )

    def test_reobserve_relocates_camera_when_planner_wired(self) -> None:
        orch = self._orch(planner=LateralOffsetViewpointPlanner(offset_mm=50.0))
        arm = orch.arm
        report = orch.run()
        # The gate refuses (views_below_min) -> reobserve -> the camera was relocated to a NEW viewpoint
        # each time (diverse views), proven by the viewpoint_NN move commands + the visited history.
        self.assertEqual(report.outcome, PickOutcome.NO_COMMIT_INSUFFICIENT_FUSION)
        self.assertGreaterEqual(len(report.viewpoints_visited), 1)
        self.assertTrue(any(c.label.startswith("viewpoint_") for c in arm.move_to_calls))  # type: ignore[attr-defined]

    def test_reobserve_no_relocate_without_planner_is_byte_identical(self) -> None:
        # Default-off invariant: no viewpoint_planner -> the reobserve re-acquires the same pose, the
        # camera is NOT relocated (byte-identical to pre-Stufe-B).
        orch = self._orch(planner=None)
        arm = orch.arm
        report = orch.run()
        self.assertEqual(report.outcome, PickOutcome.NO_COMMIT_INSUFFICIENT_FUSION)
        self.assertEqual(len(report.viewpoints_visited), 0)
        self.assertFalse(any(c.label.startswith("viewpoint_") for c in arm.move_to_calls))  # type: ignore[attr-defined]


class _SpyPolicy:
    """Records which candidate was executed (so the test can assert the fall-back)."""

    def __init__(self) -> None:
        self.calls: list[GraspPoint] = []

    def execute(self, grasp: GraspPoint) -> PolicyReport:
        self.calls.append(grasp)
        return PolicyReport(outcome=PolicyOutcome.EXECUTED)


class V3CandidateLogCaptureTests(unittest.TestCase):
    """P4.1: the V3 ranking-shadow seam captures the per-candidate log + behavior id (mock, no Isaac).

    Proves the WIRING (the pure assembler is unit-tested in test_ranking_shadow_and_pairwise_logistic):
    when a ranking shadow is wired, ``_maybe_run_ranking_shadow`` populates ``_candidate_log`` +
    ``_behavior_candidate_id`` (which shadow.py then splices into ``record.extra``); default-off otherwise.
    """

    @staticmethod
    def _gp(score: float, feats: dict) -> GraspPoint:
        return GraspPoint(
            position=np.array([100.0, 50.0, 400.0]),
            approach=np.array([0.0, 0.0, 1.0]),
            axis=np.array([1.0, 0.0, 0.0]),
            grip_width_mm=40.0,
            score=score,
            frame=GraspFrame.CAMERA,
            label="t",
            metadata={"rl_state_features": feats},
        )

    @staticmethod
    def _ranking_router() -> object:
        from src.robot.grasping.rl.ranking_policy import (
            load_pairwise_logistic_ranking_policy,
        )
        from src.robot.grasping.rl.router import ShadowRouter

        committed = (
            Path(__file__).resolve().parents[1]
            / "docs" / "baselines" / "rl_policies" / "v3_ranking_baseline_v1.json"
        )
        return ShadowRouter(
            candidate_policy=object(),  # never invoked on the ranking path
            ranking_policy=load_pairwise_logistic_ranking_policy(committed),
        )

    def _orch(self, *, with_router: bool) -> BinPickingOrchestrator:
        return BinPickingOrchestrator(
            arm=_FakeArm(),  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_success_result()]),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            shadow_router=self._ranking_router() if with_router else None,  # type: ignore[arg-type]
        )

    def test_candidate_log_and_behavior_captured(self) -> None:
        orch = self._orch(with_router=True)
        cands = (
            self._gp(0.9, {"geometric_score": 0.9}),
            self._gp(0.7, {"geometric_score": 0.7}),
            self._gp(0.5, {"geometric_score": 0.5}),
        )
        orch._maybe_run_ranking_shadow(
            pre_blend_candidates=cands, post_blend_candidates=cands, attempt_index=0
        )
        log = orch._candidate_log
        self.assertIsNotNone(log)
        assert log is not None  # narrow for type-checkers
        self.assertEqual([r["candidate_id"] for r in log], ["a0_c0", "a0_c1", "a0_c2"])
        self.assertEqual([r["rank"] for r in log], [0, 1, 2])
        self.assertEqual([r["executed"] for r in log], [True, False, False])
        self.assertEqual(orch._behavior_candidate_id, "a0_c0")
        # the 15-key feature row was projected (geometric_score carried through the metadata block)
        self.assertEqual(log[0]["features"]["geometric_score"], 0.9)

    def test_no_router_is_byte_identical_noop(self) -> None:
        orch = self._orch(with_router=False)
        gp = (self._gp(0.9, {}),)
        orch._maybe_run_ranking_shadow(
            pre_blend_candidates=gp, post_blend_candidates=gp, attempt_index=0
        )
        self.assertIsNone(orch._candidate_log)
        self.assertIsNone(orch._behavior_candidate_id)


class ApproachPathValidationTests(unittest.TestCase):
    """G12-slice: the dense-mode approach/retreat SWEEP validator + the fall-back to the next candidate.

    Geometry mirrors tests/test_grasp_trajectory_safety.py: an identity-axis grasp at the origin
    (axis=+x closing, approach=+z) whose pre-grasp sweep is blocked by an obstacle at (-25, 0, -80).
    """

    @staticmethod
    def _gp(pos: tuple[float, float, float], *, frame: GraspFrame = GraspFrame.CAMERA) -> GraspPoint:
        return GraspPoint(
            position=np.asarray(pos, dtype=np.float64),
            approach=np.array([0.0, 0.0, 1.0]),
            axis=np.array([1.0, 0.0, 0.0]),
            grip_width_mm=40.0,
            score=0.9,
            frame=frame,
            label="t",
        )

    @staticmethod
    def _result(candidates: list[GraspPoint], obstacle_mm: list) -> GraspResult:
        return GraspResult(
            candidates=tuple(candidates),
            reasons=(),
            top_score=float(candidates[0].score) if candidates else 0.0,
            metadata={
                "scene_points_mm": np.asarray(obstacle_mm, dtype=np.float64),
                "scene_points_frame": "camera",
            },
        )

    def _orch(self, spy: _SpyPolicy, *, enabled: bool = True) -> BinPickingOrchestrator:
        return BinPickingOrchestrator(
            arm=_FakeArm(),  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_success_result()]),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            policy=spy,  # type: ignore[arg-type]
            approach_path_policy=(
                ApproachPathPolicy(standoff_mm=80.0, num_approach_samples=6) if enabled else None
            ),
            _approach_path_modes=(frozenset({"dense_clutter"}) if enabled else frozenset()),
            mode_label="dense_clutter",
        )

    def test_blocked_candidate_falls_back_to_next(self) -> None:
        # candidates[0] sweep is blocked; candidates[1] is far from the obstacle -> execute candidates[1].
        spy = _SpyPolicy()
        orch = self._orch(spy)
        report = orch._execute(
            self._result([self._gp((0.0, 0.0, 0.0)), self._gp((200.0, 0.0, 0.0))], [[-25.0, 0.0, -80.0]])
        )
        self.assertIs(report.outcome, PolicyOutcome.EXECUTED)
        self.assertEqual(len(spy.calls), 1)
        np.testing.assert_allclose(spy.calls[0].position, [200.0, 0.0, 0.0])

    def test_all_blocked_refuses_no_motion(self) -> None:
        spy = _SpyPolicy()
        orch = self._orch(spy)
        report = orch._execute(self._result([self._gp((0.0, 0.0, 0.0))], [[-25.0, 0.0, -80.0]]))
        self.assertIs(report.outcome, PolicyOutcome.APPROACH_PATH_BLOCKED)
        self.assertEqual(spy.calls, [])  # motion never commanded

    def test_clear_executes_first_candidate(self) -> None:
        spy = _SpyPolicy()
        orch = self._orch(spy)
        report = orch._execute(self._result([self._gp((0.0, 0.0, 0.0))], [[500.0, 0.0, 0.0]]))
        self.assertIs(report.outcome, PolicyOutcome.EXECUTED)
        self.assertEqual(len(spy.calls), 1)

    def test_disabled_is_byte_identical(self) -> None:
        # approach_path_policy off -> the obstacle is ignored, candidates[0] executes (legacy path).
        spy = _SpyPolicy()
        orch = self._orch(spy, enabled=False)
        report = orch._execute(self._result([self._gp((0.0, 0.0, 0.0))], [[-25.0, 0.0, -80.0]]))
        self.assertIs(report.outcome, PolicyOutcome.EXECUTED)
        self.assertEqual(len(spy.calls), 1)

    def test_base_frame_transforms_cloud(self) -> None:
        # A BASE candidate + an identity CAMERA->BASE transform -> the cloud is mapped to base and still
        # blocks (proves the frame transform fires on the production base-frame path).
        spy = _SpyPolicy()
        orch = self._orch(spy)
        orch._pending_camera_to_base = Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE)
        report = orch._execute(
            self._result([self._gp((0.0, 0.0, 0.0), frame=GraspFrame.BASE)], [[-25.0, 0.0, -80.0]])
        )
        self.assertIs(report.outcome, PolicyOutcome.APPROACH_PATH_BLOCKED)

    def test_base_frame_no_transform_skips(self) -> None:
        # A BASE candidate with NO transform -> fail-safe skip (proceed), never validate base vs camera.
        spy = _SpyPolicy()
        orch = self._orch(spy)
        orch._pending_camera_to_base = None
        report = orch._execute(
            self._result([self._gp((0.0, 0.0, 0.0), frame=GraspFrame.BASE)], [[-25.0, 0.0, -80.0]])
        )
        self.assertIs(report.outcome, PolicyOutcome.EXECUTED)
        self.assertEqual(len(spy.calls), 1)


if __name__ == "__main__":
    unittest.main()
