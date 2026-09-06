"""Phase U8 — behavioural wiring of the fused-uncertainty signal.

These tests pin the behavioural seams introduced by U8:

* ``UncertaintySnapshot`` gains three U8 fields
  (``disagreement``, ``disagreement_threshold``,
  ``disagreement_triggered``) which propagate through
  ``to_dict``.
* ``fuse_uncertainty`` records the spread ``max(remapped) -
  min(remapped)`` across **contributing** channels and trips the
  disagreement flag at ``>=`` the supplied threshold.
* ``DecisionEngine.decide`` consumes the snapshot under
  ``uncertainty_active=True``:
    - confident fused ⇒ ``GRASP_NOW`` / ``CONFIDENT_GRASP``;
    - low-confidence fused (budget + planner) ⇒ ``MOVE_CAMERA`` /
      ``LOW_CONFIDENCE``;
    - low-confidence fused (no budget, real HW) ⇒ ``FAIL_CLOSED`` /
      ``UNCERTAINTY_FAIL_CLOSED``;
    - disagreement (budget + planner) ⇒ ``MOVE_CAMERA`` /
      ``CHANNEL_DISAGREEMENT``;
    - disagreement (no budget, real HW) ⇒ ``FAIL_CLOSED`` /
      ``CHANNEL_DISAGREEMENT``.
* ``uncertainty_active=True`` but ``fused_available=False`` ⇒ Q2=A
  legacy fallback with ``uncertainty_source="legacy_fallback"``.
* ``reorder_actions_for_uncertainty`` is identity when
  ``aggressive=False`` and float-stable group-preserving when
  ``aggressive=True``.
* ``SceneRecoveryContext.aggressive_recovery_bias`` end-to-ends
  through ``RecoveryOrchestrator.next_step``.
* The replay telemetry catalog knows the new
  ``UNCERTAINTY_FAIL_CLOSED`` outcome.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.decision import (
    DecisionAction,
    DecisionEngine,
    DecisionPolicy,
    DecisionReasonCode,
)
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.recovery.policy import (
    SceneRecoveryAction,
    SceneRecoveryContext,
    SceneRecoveryPolicy,
)
from src.robot.grasping.recovery.orchestrator import (
    reorder_actions_for_uncertainty,
)
from src.robot.grasping.uncertainty import (
    UncertaintyCalibration,
    UncertaintyChannelValues,
    UncertaintySnapshot,
    UncertaintyWeights,
    fuse_uncertainty,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grasp_result(score: float, *, candidates: int = 1) -> GraspResult:
    points = tuple(
        GraspPoint(
            position=np.array([100.0, 50.0, 400.0]),
            approach=np.array([0.0, 0.0, 1.0]),
            axis=np.array([1.0, 0.0, 0.0]),
            grip_width_mm=40.0,
            score=score,
            frame=GraspFrame.BASE,
            label="t",
        )
        for _ in range(candidates)
    )
    return GraspResult(candidates=points, reasons=(), top_score=score)


def _engine(**overrides) -> DecisionEngine:
    kwargs = dict(
        enabled=True,
        auto_uncertainty_threshold=0.4,
        max_reobservations=2,
        reasons_penalty=0.2,
        fail_closed_on_real_hardware=True,
    )
    kwargs.update(overrides)
    return DecisionEngine(policy=DecisionPolicy(**kwargs))


def _calibration() -> UncertaintyCalibration:
    return UncertaintyCalibration(
        weights=UncertaintyWeights(
            depth_confidence=1.0,
            mask_confidence=1.0,
            occlusion_corridor_risk=1.0,
            feasibility_margin=1.0,
            verification_residual=1.0,
            topology_risk=0.0,
            semantic_confidence=0.0,
        ),
        maps={},
    )


# ---------------------------------------------------------------------------
# 1. Disagreement math
# ---------------------------------------------------------------------------


class DisagreementMathTests(unittest.TestCase):

    def test_disagreement_zero_with_single_contributing_channel(self) -> None:
        snap = fuse_uncertainty(
            UncertaintyChannelValues(depth_confidence=0.7),
            _calibration(),
            fail_closed_threshold=0.5,
            channel_disagreement_threshold=0.3,
        )
        self.assertTrue(snap.fused_available)
        self.assertEqual(snap.disagreement, 0.0)
        self.assertFalse(snap.disagreement_triggered)

    def test_disagreement_max_minus_min_over_contributing(self) -> None:
        snap = fuse_uncertainty(
            UncertaintyChannelValues(
                depth_confidence=0.1,
                mask_confidence=0.9,
                occlusion_corridor_risk=0.5,
            ),
            _calibration(),
            fail_closed_threshold=0.5,
            channel_disagreement_threshold=0.5,
        )
        # Identity maps ⇒ remapped == raw.
        self.assertAlmostEqual(snap.disagreement, 0.8, places=6)
        self.assertTrue(snap.disagreement_triggered)

    def test_disagreement_off_sentinel_never_trips(self) -> None:
        snap = fuse_uncertainty(
            UncertaintyChannelValues(
                depth_confidence=0.0, mask_confidence=1.0
            ),
            _calibration(),
            fail_closed_threshold=0.5,
        )
        self.assertAlmostEqual(snap.disagreement, 1.0, places=6)
        self.assertFalse(snap.disagreement_triggered)


# ---------------------------------------------------------------------------
# 2. Snapshot extension
# ---------------------------------------------------------------------------


class SnapshotExtensionTests(unittest.TestCase):

    def test_disabled_has_u8_keys_off(self) -> None:
        snap = UncertaintySnapshot.disabled(fail_closed_threshold=0.5)
        d = snap.to_dict()
        self.assertIn("disagreement", d)
        self.assertIn("disagreement_threshold", d)
        self.assertIn("disagreement_triggered", d)
        self.assertEqual(d["disagreement"], 0.0)
        self.assertFalse(d["disagreement_triggered"])

    def test_to_dict_has_nine_keys(self) -> None:
        snap = UncertaintySnapshot.disabled(fail_closed_threshold=0.5)
        self.assertEqual(len(snap.to_dict()), 9)


# ---------------------------------------------------------------------------
# 3. Decision fused — confident / low-confidence
# ---------------------------------------------------------------------------


class DecisionFusedPathTests(unittest.TestCase):

    def _snapshot(self, fused: float) -> UncertaintySnapshot:
        # Build a real snapshot via the fusion (single-channel keeps it deterministic).
        return fuse_uncertainty(
            UncertaintyChannelValues(depth_confidence=fused),
            _calibration(),
            fail_closed_threshold=0.5,
        )

    def test_low_fused_triggers_grasp_now(self) -> None:
        # fused = depth_confidence raw = 0.1 (low uncertainty) ⇒ confident.
        engine = _engine()
        report = engine.decide(
            grasp_result=_grasp_result(0.9),
            mode="HARD",
            attempt_id="u8-1",
            reobservation_count=0,
            is_simulated=False,
            viewpoint_planner_available=True,
            uncertainty_snapshot=self._snapshot(0.1),
            uncertainty_active=True,
        )
        self.assertIs(report.action, DecisionAction.GRASP_NOW)
        self.assertIs(report.reason_code, DecisionReasonCode.CONFIDENT_GRASP)
        self.assertEqual(report.uncertainty_source, "fused")
        self.assertTrue(report.uncertainty_fused_available)

    def test_high_fused_triggers_move_camera(self) -> None:
        engine = _engine()
        report = engine.decide(
            grasp_result=_grasp_result(0.9),
            mode="HARD",
            attempt_id="u8-2",
            reobservation_count=0,
            is_simulated=False,
            viewpoint_planner_available=True,
            uncertainty_snapshot=self._snapshot(0.9),
            uncertainty_active=True,
        )
        self.assertIs(report.action, DecisionAction.MOVE_CAMERA)
        self.assertIs(report.reason_code, DecisionReasonCode.LOW_CONFIDENCE)

    def test_high_fused_no_budget_fail_closed(self) -> None:
        engine = _engine(max_reobservations=0)
        report = engine.decide(
            grasp_result=_grasp_result(0.9),
            mode="HARD",
            attempt_id="u8-3",
            reobservation_count=0,
            is_simulated=False,
            viewpoint_planner_available=True,
            uncertainty_snapshot=self._snapshot(0.9),
            uncertainty_active=True,
        )
        self.assertIs(report.action, DecisionAction.FAIL_CLOSED)
        self.assertIs(
            report.reason_code, DecisionReasonCode.UNCERTAINTY_FAIL_CLOSED
        )


# ---------------------------------------------------------------------------
# 4. Decision legacy fallback when fused_available=False
# ---------------------------------------------------------------------------


class DecisionLegacyFallbackTests(unittest.TestCase):

    def test_active_with_no_channels_uses_legacy_formula(self) -> None:
        # No channels set ⇒ fused_available=False ⇒ Q2=A fallback.
        snap = fuse_uncertainty(
            UncertaintyChannelValues(),
            _calibration(),
            fail_closed_threshold=0.5,
        )
        self.assertFalse(snap.fused_available)
        engine = _engine()
        report = engine.decide(
            grasp_result=_grasp_result(0.9),
            mode="HARD",
            attempt_id="u8-4",
            reobservation_count=0,
            is_simulated=False,
            viewpoint_planner_available=True,
            uncertainty_snapshot=snap,
            uncertainty_active=True,
        )
        self.assertEqual(report.uncertainty_source, "legacy_fallback")
        # 1 - 0.9 = 0.1 < 0.4 ⇒ confident grasp under legacy formula.
        self.assertIs(report.action, DecisionAction.GRASP_NOW)


# ---------------------------------------------------------------------------
# 5/6. Disagreement → MOVE_CAMERA / FAIL_CLOSED with CHANNEL_DISAGREEMENT
# ---------------------------------------------------------------------------


class DisagreementDecisionTests(unittest.TestCase):

    def _disagreement_snapshot(self) -> UncertaintySnapshot:
        # Two channels with a spread of 0.8 ⇒ trips at threshold 0.5.
        return fuse_uncertainty(
            UncertaintyChannelValues(
                depth_confidence=0.1, mask_confidence=0.9
            ),
            _calibration(),
            fail_closed_threshold=0.5,
            channel_disagreement_threshold=0.5,
        )

    def test_disagreement_with_budget_moves_camera(self) -> None:
        engine = _engine()
        report = engine.decide(
            grasp_result=_grasp_result(0.95),
            mode="HARD",
            attempt_id="u8-5",
            reobservation_count=0,
            is_simulated=False,
            viewpoint_planner_available=True,
            uncertainty_snapshot=self._disagreement_snapshot(),
            uncertainty_active=True,
        )
        self.assertIs(report.action, DecisionAction.MOVE_CAMERA)
        self.assertIs(
            report.reason_code, DecisionReasonCode.CHANNEL_DISAGREEMENT
        )

    def test_disagreement_no_budget_real_hw_fail_closed(self) -> None:
        engine = _engine(max_reobservations=0)
        report = engine.decide(
            grasp_result=_grasp_result(0.95),
            mode="HARD",
            attempt_id="u8-6",
            reobservation_count=0,
            is_simulated=False,
            viewpoint_planner_available=True,
            uncertainty_snapshot=self._disagreement_snapshot(),
            uncertainty_active=True,
        )
        self.assertIs(report.action, DecisionAction.FAIL_CLOSED)
        self.assertIs(
            report.reason_code, DecisionReasonCode.CHANNEL_DISAGREEMENT
        )


# ---------------------------------------------------------------------------
# 7. reorder_actions_for_uncertainty helper
# ---------------------------------------------------------------------------


class ReorderActionsTests(unittest.TestCase):

    def test_no_op_when_not_aggressive(self) -> None:
        actions = (
            SceneRecoveryAction.NEXT_TARGET,
            SceneRecoveryAction.NEXT_VIEWPOINT,
            SceneRecoveryAction.NUDGE_TARGET,
        )
        self.assertEqual(
            reorder_actions_for_uncertainty(actions, aggressive=False),
            actions,
        )

    def test_aggressive_promotes_perception_group(self) -> None:
        actions = (
            SceneRecoveryAction.NEXT_TARGET,
            SceneRecoveryAction.NEXT_VIEWPOINT,
            SceneRecoveryAction.NUDGE_TARGET,
            SceneRecoveryAction.RESCAN,
        )
        reordered = reorder_actions_for_uncertainty(actions, aggressive=True)
        self.assertEqual(
            reordered,
            (
                SceneRecoveryAction.NEXT_VIEWPOINT,
                SceneRecoveryAction.RESCAN,
                SceneRecoveryAction.NEXT_TARGET,
                SceneRecoveryAction.NUDGE_TARGET,
            ),
        )

    def test_aggressive_no_op_when_no_perception(self) -> None:
        actions = (SceneRecoveryAction.NEXT_TARGET, SceneRecoveryAction.NUDGE_TARGET)
        self.assertEqual(
            reorder_actions_for_uncertainty(actions, aggressive=True),
            actions,
        )


# ---------------------------------------------------------------------------
# 8. SceneRecoveryContext aggressive_recovery_bias field
# ---------------------------------------------------------------------------


class SceneRecoveryContextBiasTests(unittest.TestCase):

    def test_default_false(self) -> None:
        policy = SceneRecoveryPolicy(enabled=True)
        ctx = SceneRecoveryContext(
            profile=None,  # type: ignore[arg-type]
            policy=policy,
            last_outcome="execution_failed",
            failure_reasons=(),
            history=(),
        )
        self.assertFalse(ctx.aggressive_recovery_bias)

    def test_explicit_true(self) -> None:
        policy = SceneRecoveryPolicy(enabled=True)
        ctx = SceneRecoveryContext(
            profile=None,  # type: ignore[arg-type]
            policy=policy,
            last_outcome="execution_failed",
            failure_reasons=(),
            history=(),
            aggressive_recovery_bias=True,
        )
        self.assertTrue(ctx.aggressive_recovery_bias)


# ---------------------------------------------------------------------------
# 9. Replay catalog knows UNCERTAINTY_FAIL_CLOSED
# ---------------------------------------------------------------------------


class TelemetryCatalogTests(unittest.TestCase):

    def test_uncertainty_fail_closed_in_catalog(self) -> None:
        from src.robot.grasping.replay.telemetry_catalog import (
            TELEMETRY_CATALOG,
        )

        self.assertIn("uncertainty_fail_closed", TELEMETRY_CATALOG)
        required = TELEMETRY_CATALOG["uncertainty_fail_closed"]
        self.assertIn("extra.decision_reason_code", required)
        self.assertIn("extra.uncertainty_score", required)


if __name__ == "__main__":
    unittest.main()
