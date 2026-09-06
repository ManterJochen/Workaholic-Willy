"""Gap G3 — the recovery-uncertainty bias gate is now the canonical single source of truth.

should_bias_recovery_for_uncertainty was dead (0 callers) while AutonomousGraspService reimplemented it
inline WRONG: it read a 0.0-default telemetry scalar, IGNORED fused_available, and used a `>` boundary.
The service now calls this canonical fn on the report's real UncertaintySnapshot. These tests pin the
canonical logic, including the two cases the inline copy got wrong (unavailable, and the >= boundary).
"""

from __future__ import annotations

from src.robot.grasping.uncertainty import (
    UncertaintySnapshot,
    should_bias_recovery_for_uncertainty,
)


def _snap(fused: float, available: bool) -> UncertaintySnapshot:
    return UncertaintySnapshot(
        fused=fused, fused_available=available, fail_closed=False, fail_closed_threshold=1.01,
    )


class TestRecoveryBiasGate:
    def test_none_snapshot_does_not_bias(self) -> None:
        assert should_bias_recovery_for_uncertainty(None, recovery_aggressive_threshold=0.5) is False

    def test_unavailable_does_not_bias(self) -> None:
        # The bug the inline copy had: a high fused with fused_available=False must NOT bias
        # (telemetry default 0.0 is indistinguishable from a real 0.0 without the available flag).
        assert should_bias_recovery_for_uncertainty(_snap(0.9, False), recovery_aggressive_threshold=0.5) is False

    def test_above_threshold_biases(self) -> None:
        assert should_bias_recovery_for_uncertainty(_snap(0.9, True), recovery_aggressive_threshold=0.5) is True

    def test_below_threshold_does_not_bias(self) -> None:
        assert should_bias_recovery_for_uncertainty(_snap(0.3, True), recovery_aggressive_threshold=0.5) is False

    def test_at_threshold_biases_inclusive(self) -> None:
        # The >= boundary the inline copy got wrong with `>`.
        assert should_bias_recovery_for_uncertainty(_snap(0.5, True), recovery_aggressive_threshold=0.5) is True
