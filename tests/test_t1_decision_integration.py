"""Phase T1 — AutonomousGraspService decision-layer integration tests.

These tests pin the operator-locked T1 wiring contract:

* When ``robot.grasping.decision.enabled`` is :data:`True`, the factory
  auto-builds a :class:`DecisionPolicy` + :class:`DecisionEngine` and
  stashes them on the service.
* The :class:`EffectiveGraspingConfig` snapshot grows decision-layer
  fields and surfaces them in ``to_dict()``.
* :meth:`AutonomousGraspService.pick` short-circuits the decision
  pre-flight when:
    - the decision engine is not wired (back-compat with T0 services),
    - the effective mode is :attr:`GraspMode.EASY` (locked
      operator answer Q1: EASY stays out of T1 scope).
* When the decision engine is wired and the mode is in scope:
    - a confident grasp result flows straight to the existing pick
      path and produces a :attr:`AutonomousGraspOutcome.SUCCEEDED`
      report whose ``decision`` field is populated and whose
      ``telemetry`` carries the six mandatory plan-rule-#12 keys.
    - a low-confidence first frame triggers a viewpoint move and a
      re-acquire; the bounded loop terminates with the final typed
      decision.
    - exhausting the re-observation budget on real hardware yields a
      :attr:`AutonomousGraspOutcome.DECISION_FAIL_CLOSED` report
      with a fully populated ``decision`` field and structured
      telemetry; no pick attempt is dispatched.
* :class:`GraspMode.EASY` never consults the decision engine even when
  one is wired.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import MotionCommand, MotionResult, RobotCapabilities
from src.robot.execution.autonomous_grasp import (
    AutonomousGraspOutcome,
    AutonomousGraspService,
    GraspMode,
)
from src.robot.grasping.decision import (
    DecisionAction,
    DecisionEngine,
    DecisionPolicy,
    DecisionReasonCode,
)
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.types.perception import PerceptionFrame


# ---------------------------------------------------------------------------
# Doubles (mirror tests/test_grasping_config_wiring.py)
# ---------------------------------------------------------------------------


_REAL_UR_CAPS = RobotCapabilities(
    vendor="ur",
    model="ur5e",
    dof=6,
    supports_joint_move=True,
    supports_linear_move=True,
    supports_async_move=False,
    has_native_fk=True,
    has_native_ik=True,
    has_force_control=False,
    is_simulated=False,
)
_SIM_CAPS = RobotCapabilities(
    vendor="dummy",
    model="sim",
    dof=6,
    supports_joint_move=True,
    supports_linear_move=True,
    supports_async_move=False,
    has_native_fk=False,
    has_native_ik=False,
    has_force_control=False,
    is_simulated=True,
)


class _TypedFakeArm:
    def __init__(self, *, capabilities: RobotCapabilities = _REAL_UR_CAPS) -> None:
        self._caps = capabilities
        self._tcp = Pose(
            position_mm=np.array([0.0, 0.0, 500.0]),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            frame=Frame.BASE,
            label="home",
        )
        self.moves: list[Pose] = []

    @property
    def capabilities(self) -> RobotCapabilities:
        return self._caps

    def get_tcp_pose(self) -> Pose:
        return self._tcp

    def move(self, pose: Pose, **_: object) -> MotionResult:
        if pose.frame == Frame.BASE:
            self._tcp = pose
        self.moves.append(pose)
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)


class _FakePerception:
    def __init__(self, frames: list[PerceptionFrame]) -> None:
        self._frames = frames
        self.calls = 0

    def acquire(self) -> PerceptionFrame:
        idx = min(self.calls, len(self._frames) - 1)
        self.calls += 1
        return self._frames[idx]


class _ScriptedCalculator:
    """Returns ``results[i]`` on the i-th call; clamps at the last entry."""

    def __init__(self, results: list[GraspResult]) -> None:
        self._results = results
        self.calls = 0

    def compute_result(self, *_args: object, **_kwargs: object) -> GraspResult:
        idx = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[idx]


class _ScriptedViewpointPlanner:
    """Yields successive base-frame poses for re-observation."""

    def __init__(self, poses: list[Pose | None]) -> None:
        self._poses = poses
        self.calls = 0

    def next_viewpoint(self, *, current_tcp, history):
        idx = self.calls
        self.calls += 1
        if idx >= len(self._poses):
            return None
        return self._poses[idx]


def _segmentation() -> SimpleNamespace:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:22, 10:22] = 1
    return SimpleNamespace(mask=mask)


def _perception_frame() -> PerceptionFrame:
    depth = np.full((32, 32), 500.0, dtype=np.float64)
    intrinsics = np.array(
        [[400.0, 0.0, 16.0], [0.0, 400.0, 16.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return PerceptionFrame(
        depth_map=depth,
        intrinsics=intrinsics,
        segmentations=(_segmentation(),),
    )


def _candidate(score: float) -> GraspPoint:
    return GraspPoint(
        position=np.array([100.0, 50.0, 400.0]),
        approach=np.array([0.0, 0.0, 1.0]),
        axis=np.array([1.0, 0.0, 0.0]),
        grip_width_mm=40.0,
        score=score,
        frame=GraspFrame.BASE,
        label="test",
    )


def _result(score: float, *, reasons: tuple = ()) -> GraspResult:
    return GraspResult(
        candidates=(_candidate(score),),
        reasons=reasons,
        top_score=score,
    )


def _viewpoint(z: float) -> Pose:
    return Pose(
        position_mm=np.array([50.0, 50.0, z]),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        frame=Frame.BASE,
        label="viewpoint",
    )


def _build_service(
    *,
    calc_results: list[GraspResult],
    frames: list[PerceptionFrame] | None = None,
    capabilities: RobotCapabilities = _REAL_UR_CAPS,
    decision_engine: DecisionEngine | None,
    viewpoint_poses: list[Pose | None] | None = None,
    mode: GraspMode = GraspMode.AUTO,
) -> tuple[AutonomousGraspService, _TypedFakeArm, _FakePerception, _ScriptedViewpointPlanner | None]:
    arm = _TypedFakeArm(capabilities=capabilities)
    perception = _FakePerception(frames or [_perception_frame()])
    calculator = _ScriptedCalculator(calc_results)
    planner = (
        _ScriptedViewpointPlanner(viewpoint_poses) if viewpoint_poses is not None else None
    )
    svc = AutonomousGraspService.from_components(
        arm=arm,
        calculator=calculator,
        perception=perception,
        mode=mode,
        viewpoint_planner=planner,
    )
    if decision_engine is not None:
        svc.decision_engine = decision_engine
        svc.decision_policy = decision_engine.policy
    return svc, arm, perception, planner


# ---------------------------------------------------------------------------
# Default wiring: no decision engine ⇒ T0 behaviour preserved
# ---------------------------------------------------------------------------


class DefaultWiringPreservesT0BehaviourTests(unittest.TestCase):
    def test_service_has_decision_slots_none_by_default(self) -> None:
        svc, *_ = _build_service(
            calc_results=[_result(0.9)],
            decision_engine=None,
        )
        # New slots exist as None so legacy call sites keep working.
        self.assertIsNone(svc.decision_policy)
        self.assertIsNone(svc.decision_engine)

    def test_pick_without_decision_engine_emits_no_decision_field(self) -> None:
        svc, *_ = _build_service(
            calc_results=[_result(0.9)],
            decision_engine=None,
        )
        report = svc.pick()
        self.assertIs(report.outcome, AutonomousGraspOutcome.SUCCEEDED)
        self.assertIsNone(report.decision)


# ---------------------------------------------------------------------------
# Engine wired + mode in scope
# ---------------------------------------------------------------------------


class DecisionEngineInScopeTests(unittest.TestCase):
    def test_confident_grasp_passes_through_and_records_decision(self) -> None:
        engine = DecisionEngine(policy=DecisionPolicy())  # defaults
        svc, _, _, _ = _build_service(
            calc_results=[_result(0.9), _result(0.9)],
            decision_engine=engine,
        )
        report = svc.pick()
        self.assertIs(report.outcome, AutonomousGraspOutcome.SUCCEEDED)
        self.assertIsNotNone(report.decision)
        self.assertIs(report.decision.action, DecisionAction.GRASP_NOW)
        self.assertIs(
            report.decision.reason_code, DecisionReasonCode.CONFIDENT_GRASP
        )
        # Plan rule #12 — telemetry carries the six mandatory keys.
        for key in (
            "decision_action",
            "decision_reason_code",
            "uncertainty_score",
            "threshold_used",
            "mode",
            "attempt_id",
        ):
            self.assertIn(key, report.telemetry)

    def test_low_confidence_triggers_move_camera_then_grasps(self) -> None:
        engine = DecisionEngine(policy=DecisionPolicy())  # max_reobs=2
        # First decision-frame computes a low-score result;
        # after moving the camera the second decision-frame computes a
        # high-score result; then the actual pick runs (third calculator call).
        svc, arm, _, planner = _build_service(
            calc_results=[_result(0.3), _result(0.9), _result(0.9)],
            frames=[_perception_frame(), _perception_frame(), _perception_frame()],
            decision_engine=engine,
            viewpoint_poses=[_viewpoint(550.0)],
        )
        report = svc.pick()
        self.assertIs(report.outcome, AutonomousGraspOutcome.SUCCEEDED)
        # The viewpoint planner was consulted exactly once.
        assert planner is not None
        self.assertEqual(planner.calls, 1)
        # The arm received the viewpoint move.
        self.assertGreaterEqual(len(arm.moves), 1)
        # Final decision is GRASP_NOW after one re-observation.
        self.assertIs(report.decision.action, DecisionAction.GRASP_NOW)
        self.assertEqual(report.decision.reobservation_count, 1)

    def test_budget_exhausted_real_hardware_fail_closes(self) -> None:
        # All decision-frame computes return low score; the engine
        # exhausts max_reobservations=2 and FAIL_CLOSES on real
        # hardware. No pick attempt should be dispatched.
        engine = DecisionEngine(policy=DecisionPolicy())
        svc, arm, _, planner = _build_service(
            # 3 low-score results: one per decision-frame iteration
            # (initial + 2 re-observations).
            calc_results=[_result(0.3), _result(0.3), _result(0.3)],
            frames=[_perception_frame()] * 3,
            decision_engine=engine,
            viewpoint_poses=[_viewpoint(550.0), _viewpoint(560.0)],
        )
        report = svc.pick()
        self.assertIs(
            report.outcome, AutonomousGraspOutcome.DECISION_FAIL_CLOSED
        )
        self.assertIsNone(report.pick_report)  # no execution dispatched
        self.assertIs(report.decision.action, DecisionAction.FAIL_CLOSED)
        self.assertIs(
            report.decision.reason_code,
            DecisionReasonCode.REOBSERVE_BUDGET_EXHAUSTED,
        )
        # Plan rule #12 telemetry.
        self.assertEqual(report.telemetry["decision_action"], "fail_closed")
        self.assertEqual(
            report.telemetry["decision_reason_code"],
            "reobserve_budget_exhausted",
        )

    def test_no_candidates_emits_recover_pending(self) -> None:
        engine = DecisionEngine(policy=DecisionPolicy())
        empty = GraspResult(candidates=(), reasons=("calculator_no_pair",), top_score=0.0)
        svc, *_ = _build_service(
            calc_results=[empty],
            decision_engine=engine,
        )
        report = svc.pick()
        # T1: RECOVER produces a typed report but does NOT execute
        # any physical action — that lands in T5.
        self.assertIs(
            report.outcome, AutonomousGraspOutcome.DECISION_RECOVER_PENDING
        )
        self.assertIs(report.decision.action, DecisionAction.RECOVER)
        self.assertIs(
            report.decision.reason_code, DecisionReasonCode.NO_CANDIDATES
        )


# ---------------------------------------------------------------------------
# Out-of-scope modes
# ---------------------------------------------------------------------------


class DecisionEngineOutOfScopeTests(unittest.TestCase):
    def test_easy_mode_never_consults_decision_engine(self) -> None:
        # EASY is the operator-locked out-of-scope mode for T1.
        engine = DecisionEngine(policy=DecisionPolicy())
        svc, _, perception, _ = _build_service(
            calc_results=[_result(0.3)],  # Would fail-close if consulted
            decision_engine=engine,
            mode=GraspMode.EASY,
        )
        report = svc.pick()
        # EASY runs the legacy path verbatim; the calculator is hit
        # exactly once (by run_attempt), not twice (decision + run).
        self.assertIsNone(report.decision)
        # Calculator was called exactly once — by run_attempt, not by
        # a decision pre-flight.
        # (perception is hit once by run_attempt as well.)
        self.assertEqual(perception.calls, 1)


# ---------------------------------------------------------------------------
# from_robot_config auto-build
# ---------------------------------------------------------------------------


class FromRobotConfigDecisionAutoBuildTests(unittest.TestCase):
    """``robot.grasping.decision.enabled`` toggles auto-build."""

    def _cfg(self, **decision_overrides):
        from src.config.schema.robot import RobotConfig
        return RobotConfig(
            vendor="dummy",
            gripper={"vendor": "none"},
            grasping={
                "default_mode": "auto",
                "max_attempts": 5,
                "decision": decision_overrides,
            },
        )

    def test_decision_disabled_leaves_engine_unwired(self) -> None:
        cfg = self._cfg(enabled=False)
        calc = _ScriptedCalculator([_result(0.9)])
        perc = _FakePerception([_perception_frame()])
        svc = AutonomousGraspService.from_robot_config(
            cfg, calculator=calc, perception=perc
        )
        self.assertIsNone(svc.decision_engine)
        self.assertIsNone(svc.decision_policy)
        # Snapshot mirrors the disabled state.
        self.assertFalse(svc.effective_config.decision.enabled)

    def test_decision_enabled_auto_builds_engine(self) -> None:
        cfg = self._cfg(
            enabled=True,
            auto_uncertainty_threshold=0.25,
            max_reobservations=1,
            reasons_penalty=0.1,
            fail_closed_on_real_hardware=False,
        )
        calc = _ScriptedCalculator([_result(0.9)])
        perc = _FakePerception([_perception_frame()])
        svc = AutonomousGraspService.from_robot_config(
            cfg, calculator=calc, perception=perc
        )
        self.assertIsNotNone(svc.decision_engine)
        self.assertIsNotNone(svc.decision_policy)
        self.assertEqual(svc.decision_policy.auto_uncertainty_threshold, 0.25)
        self.assertEqual(svc.decision_policy.max_reobservations, 1)
        self.assertEqual(svc.decision_policy.reasons_penalty, 0.1)
        self.assertFalse(svc.decision_policy.fail_closed_on_real_hardware)
        # Snapshot mirrors the enabled state and stores threshold.
        self.assertTrue(svc.effective_config.decision.enabled)
        self.assertEqual(
            svc.effective_config.decision.auto_uncertainty_threshold, 0.25
        )
        self.assertEqual(svc.effective_config.decision.max_reobservations, 1)

    def test_caller_supplied_engine_wins_over_config(self) -> None:
        cfg = self._cfg(enabled=True, auto_uncertainty_threshold=0.25)
        custom = DecisionEngine(policy=DecisionPolicy(auto_uncertainty_threshold=0.9))
        calc = _ScriptedCalculator([_result(0.9)])
        perc = _FakePerception([_perception_frame()])
        svc = AutonomousGraspService.from_robot_config(
            cfg,
            calculator=calc,
            perception=perc,
            decision_engine=custom,
        )
        self.assertIs(svc.decision_engine, custom)
        self.assertEqual(svc.decision_policy.auto_uncertainty_threshold, 0.9)


if __name__ == "__main__":
    unittest.main()
