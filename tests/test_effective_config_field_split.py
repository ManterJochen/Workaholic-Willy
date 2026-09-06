"""Byte-identity guard for the ``EffectiveGraspingConfig`` flat→nested split.

The config was split into per-phase nested sub-configs (``config.watchdog.mode``
instead of ``config.watchdog_mode``), but ``to_dict()`` must keep emitting the
historical *flat* 79-key dict in the exact same order with the exact same value
transforms — it feeds ``outcome_logging.json_safe`` (serialized with
``sort_keys=False``), so key order is load-bearing for replay/KPI diffs.

The rest of the suite only does subset (``assertIn``) checks on ``to_dict()``,
so a silent key reorder or drop would pass everywhere else. This file is the
explicit lock on key order, count, and the non-trivial value transforms.
"""

from __future__ import annotations

import json
import unittest

from src.robot.execution.autonomous_grasp import (
    EffectiveGraspingConfig,
    EffectiveRecoveryOrchestratorConfig,
    EffectiveUncertaintyConfig,
    EffectiveWatchdogConfig,
    GraspMode,
)

# The canonical flat key order emitted by ``EffectiveGraspingConfig.to_dict()``.
# Note ``feasibility_corridor_risk_*`` deliberately sit in the *corridor* region
# (positions 24-25) even though they now live on the feasibility sub-config — the
# split must NOT reorder them.
EXPECTED_KEYS: tuple[str, ...] = (
    "default_mode",
    "max_attempts",
    "closed_loop_enabled",
    "verification_enabled",
    "dense_recovery_enabled",
    "dense_recovery_allowed_actions",
    "decision_enabled",
    "decision_auto_uncertainty_threshold",
    "decision_max_reobservations",
    "feasibility_enabled",
    "feasibility_score_weight",
    "feasibility_ik_quality_enabled",
    "feasibility_joint_margin_enabled",
    "feasibility_swept_approach_enabled",
    "feasibility_swept_approach_top_k",
    "corridor_directional_enabled",
    "corridor_hard_reject_enabled",
    "corridor_top_k",
    "corridor_radius_mm",
    "corridor_step_mm",
    "corridor_max_distance_mm",
    "corridor_mask_fusion_weight",
    "corridor_hard_reject_confidence_threshold",
    "feasibility_corridor_risk_enabled",
    "feasibility_corridor_risk_weight",
    "ordering_enabled",
    "ordering_unlock_weight",
    "ordering_max_local_score_drop",
    "ordering_mask_adjacency_enabled",
    "ordering_depth_only_enabled",
    "ordering_corridor_overlap_enabled",
    "ordering_adjacency_radius_px",
    "ordering_depth_tolerance_mm",
    "recovery_orchestrator_enabled",
    "recovery_orchestrator_max_actions",
    "recovery_orchestrator_apply_modes",
    "recovery_orchestrator_allowed_actions",
    "recovery_orchestrator_per_action_budget",
    "uncertainty_enabled",
    "uncertainty_fail_closed_threshold",
    "uncertainty_apply_modes",
    "uncertainty_weights",
    "uncertainty_ranking_penalty_weight",
    "uncertainty_recovery_aggressive_threshold",
    "uncertainty_channel_disagreement_threshold",
    "uncertainty_calibration_artifact_path",
    "watchdog_mode",
    "watchdog_window_size",
    "watchdog_min_samples_for_trend",
    "watchdog_calibration_delta_moderate_mm",
    "watchdog_calibration_delta_high_mm",
    "watchdog_calibration_delta_severe_mm",
    "watchdog_depth_confidence_shift_moderate",
    "watchdog_depth_confidence_shift_high",
    "watchdog_depth_confidence_shift_severe",
    "watchdog_fail_closed_rate_moderate",
    "watchdog_fail_closed_rate_high",
    "watchdog_fail_closed_rate_severe",
    "watchdog_hand_eye_residual_trend_moderate_mm",
    "watchdog_hand_eye_residual_trend_high_mm",
    "watchdog_hand_eye_residual_trend_severe_mm",
    "watchdog_verification_residual_trend_moderate_mm",
    "watchdog_verification_residual_trend_high_mm",
    "watchdog_verification_residual_trend_severe_mm",
    "watchdog_ood_score_moderate",
    "watchdog_ood_score_high",
    "watchdog_ood_score_severe",
    "watchdog_block_modes",
    "performance_enabled",
    "performance_decision_latency_slo_ms",
    "performance_ranking_latency_slo_ms",
    "performance_fusion_latency_slo_ms",
    "performance_decision_timeout_ms",
    "performance_ranking_timeout_ms",
    "performance_fusion_timeout_ms",
    "performance_model_fallback_policy",
    "performance_fusion_fallback_policy",
    "performance_emit_breach_events",
    "performance_breach_window_size",
    # APPENDED 2026-08-17. The contract was silent about three of the thirteen blocks, so a
    # GraspAttemptRecord could not say whether the learned success model, the multi-view fusion
    # substrate, or the swept-volume approach validator was active for that attempt -- and the
    # last of those is the one that can refuse a motion. Appended rather than inserted, so every
    # key above keeps its index for any consumer that reads positionally.
    "success_model_enabled",
    "fusion_enabled",
    "approach_validation_enabled",
)


def _bare() -> EffectiveGraspingConfig:
    """A bare snapshot: 6 core fields set, all 8 phase sub-configs defaulted."""

    return EffectiveGraspingConfig(
        default_mode=GraspMode.AUTO,
        max_attempts=5,
        closed_loop_enabled=False,
        verification_enabled=False,
        dense_recovery_enabled=False,
        dense_recovery_allowed_actions=(),
    )


class ToDictByteIdentityTests(unittest.TestCase):
    def test_key_order_and_count_is_canonical(self) -> None:
        keys = tuple(_bare().to_dict().keys())
        # 79 historical + 3 appended 2026-08-17 (success_model / fusion / approach_validation).
        self.assertEqual(len(keys), 82)
        self.assertEqual(keys, EXPECTED_KEYS)
        # The extension must be an APPEND. A consumer that reads this contract positionally --
        # and the KPI / soak / RL tail is exactly that kind of consumer -- breaks silently on an
        # insertion and not at all on an append, so the shape of the change is itself the contract.
        self.assertEqual(keys[:79], EXPECTED_KEYS[:79])

    def test_corridor_risk_keys_stay_in_corridor_region(self) -> None:
        # Regression guard: these belong to the feasibility sub-config now but
        # must still serialize between the corridor block and ordering block.
        keys = list(_bare().to_dict().keys())
        self.assertEqual(
            keys[keys.index("corridor_hard_reject_confidence_threshold") + 1],
            "feasibility_corridor_risk_enabled",
        )
        self.assertEqual(
            keys[keys.index("feasibility_corridor_risk_weight") + 1],
            "ordering_enabled",
        )

    def test_value_transforms_preserved(self) -> None:
        d = _bare().to_dict()
        # enum -> stable string
        self.assertEqual(d["default_mode"], "auto")
        self.assertIsInstance(d["default_mode"], str)
        # tuple -> list
        self.assertEqual(d["dense_recovery_allowed_actions"], [])
        # off-sentinels survive
        self.assertEqual(d["uncertainty_recovery_aggressive_threshold"], 1.01)
        self.assertEqual(d["uncertainty_channel_disagreement_threshold"], 1.01)
        # None stays None
        self.assertIsNone(d["uncertainty_calibration_artifact_path"])
        # tuple-of-tuples -> list-of-lists
        self.assertEqual(
            d["uncertainty_weights"],
            [
                ["depth_confidence", 1.0],
                ["mask_confidence", 1.0],
                ["occlusion_corridor_risk", 1.0],
                ["feasibility_margin", 1.0],
                ["verification_residual", 1.0],
                ["topology_risk", 0.0],
                ["semantic_confidence", 0.0],
            ],
        )
        self.assertEqual(d["recovery_orchestrator_per_action_budget"], [])
        self.assertEqual(d["watchdog_mode"], "shadow")
        # the whole thing must be JSON-serializable (json_safe contract)
        json.dumps(d)

    def test_per_action_budget_list_of_pairs_transform(self) -> None:
        eff = EffectiveGraspingConfig(
            default_mode=GraspMode.DENSE_CLUTTER,
            max_attempts=5,
            closed_loop_enabled=False,
            verification_enabled=False,
            dense_recovery_enabled=False,
            dense_recovery_allowed_actions=(),
            recovery_orchestrator=EffectiveRecoveryOrchestratorConfig(
                per_action_budget=(("rescan", 1), ("next_viewpoint", 2)),
            ),
        )
        self.assertEqual(
            eff.to_dict()["recovery_orchestrator_per_action_budget"],
            [["rescan", 1], ["next_viewpoint", 2]],
        )


class NestedStructureTests(unittest.TestCase):
    def test_nested_attrs_agree_with_flat_to_dict(self) -> None:
        eff = EffectiveGraspingConfig(
            default_mode=GraspMode.AUTO,
            max_attempts=5,
            closed_loop_enabled=False,
            verification_enabled=False,
            dense_recovery_enabled=False,
            dense_recovery_allowed_actions=(),
            watchdog=EffectiveWatchdogConfig(mode="active", window_size=30),
            uncertainty=EffectiveUncertaintyConfig(
                enabled=True, fail_closed_threshold=0.7
            ),
        )
        d = eff.to_dict()
        self.assertEqual(eff.watchdog.mode, "active")
        self.assertEqual(d["watchdog_mode"], "active")
        self.assertEqual(d["watchdog_window_size"], 30)
        self.assertEqual(eff.uncertainty.fail_closed_threshold, 0.7)
        self.assertEqual(d["uncertainty_fail_closed_threshold"], 0.7)
        self.assertTrue(d["uncertainty_enabled"])

    def test_snapshot_is_frozen_and_hashable(self) -> None:
        eff = _bare()
        self.assertIsInstance(hash(eff), int)
        with self.assertRaises(Exception):
            eff.watchdog = EffectiveWatchdogConfig()  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
