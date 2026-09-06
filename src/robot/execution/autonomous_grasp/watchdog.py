"""Drift/OOD watchdog coordinator.

Stateless with respect to the service: the rolling sample history and the event
listener live on the service and are passed into each method, with
``service.watchdog_history`` as the injection point for the history. This module
owns the policy construction, evaluation, severity/reason logic, and rising-edge
transition emission.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Optional

from src.robot.events import RobotWatchdogEvent
from src.robot.execution.calibration_watchdog import (
    DriftSeverity,
    WatchdogAction,
    WatchdogHistory,
    WatchdogMode,
    WatchdogPolicy,
    WatchdogReport,
    WatchdogSample,
    evaluate_watchdog,
)
from src.robot.grasping.decision import (
    DecisionAction,
    DecisionReasonCode,
    DecisionReport,
)

if TYPE_CHECKING:
    from src.robot.events import RobotWatchdogEventListener

    from .config import EffectiveGraspingConfig, GraspMode


class WatchdogCoordinator:
    """Drift/OOD watchdog logic.

    The sample history and event listener are owned by the service and passed in
    per call, so ``service.watchdog_history`` is the public injection point for
    crafted drift trends.
    """

    HISTORY_CEILING: ClassVar[int] = 4096

    # Severity order shared by event-edge detection and the drift/ood reason picker.
    _SEVERITY_ORDER: ClassVar[dict[DriftSeverity, int]] = {
        DriftSeverity.NONE: 0,
        DriftSeverity.LOW: 1,
        DriftSeverity.MODERATE: 2,
        DriftSeverity.HIGH: 3,
        DriftSeverity.SEVERE: 4,
    }

    def build_policy(
        self, config: Optional["EffectiveGraspingConfig"]
    ) -> Optional[WatchdogPolicy]:
        """Construct a :class:`WatchdogPolicy` from the effective config snapshot.

        Returns :data:`None` when no effective config is wired, which is the
        ``from_components`` path. Otherwise the policy is built unconditionally,
        ``mode == "disabled"`` included, so the evaluator's typed shape stays
        stable; the caller owns the disabled-mode short-circuit on telemetry.
        """

        cfg = config
        if cfg is None:
            return None
        return WatchdogPolicy(
            mode=WatchdogMode(cfg.watchdog.mode),
            window_size=int(cfg.watchdog.window_size),
            min_samples_for_trend=int(cfg.watchdog.min_samples_for_trend),
            calibration_delta_moderate_mm=float(
                cfg.watchdog.calibration_delta_moderate_mm
            ),
            calibration_delta_high_mm=float(cfg.watchdog.calibration_delta_high_mm),
            calibration_delta_severe_mm=float(
                cfg.watchdog.calibration_delta_severe_mm
            ),
            depth_confidence_shift_moderate=float(
                cfg.watchdog.depth_confidence_shift_moderate
            ),
            depth_confidence_shift_high=float(
                cfg.watchdog.depth_confidence_shift_high
            ),
            depth_confidence_shift_severe=float(
                cfg.watchdog.depth_confidence_shift_severe
            ),
            fail_closed_rate_moderate=float(cfg.watchdog.fail_closed_rate_moderate),
            fail_closed_rate_high=float(cfg.watchdog.fail_closed_rate_high),
            fail_closed_rate_severe=float(cfg.watchdog.fail_closed_rate_severe),
            hand_eye_residual_trend_moderate_mm=float(
                cfg.watchdog.hand_eye_residual_trend_moderate_mm
            ),
            hand_eye_residual_trend_high_mm=float(
                cfg.watchdog.hand_eye_residual_trend_high_mm
            ),
            hand_eye_residual_trend_severe_mm=float(
                cfg.watchdog.hand_eye_residual_trend_severe_mm
            ),
            verification_residual_trend_moderate_mm=float(
                cfg.watchdog.verification_residual_trend_moderate_mm
            ),
            verification_residual_trend_high_mm=float(
                cfg.watchdog.verification_residual_trend_high_mm
            ),
            verification_residual_trend_severe_mm=float(
                cfg.watchdog.verification_residual_trend_severe_mm
            ),
            ood_score_moderate=float(cfg.watchdog.ood_score_moderate),
            ood_score_high=float(cfg.watchdog.ood_score_high),
            ood_score_severe=float(cfg.watchdog.ood_score_severe),
            block_modes=tuple(cfg.watchdog.block_modes),
        )

    def evaluate_pretick(
        self,
        *,
        history: list[WatchdogSample],
        policy: Optional[WatchdogPolicy],
        effective_mode: "GraspMode",
        is_simulated: bool,
    ) -> Optional[WatchdogReport]:
        """Evaluate the watchdog over the rolling history.

        Returns :data:`None` when no policy is wired. Otherwise a typed
        :class:`WatchdogReport` even when mode is ``disabled``; enforcement is
        encoded in ``report.enforced``.
        """

        if policy is None:
            return None
        return evaluate_watchdog(
            WatchdogHistory(tuple(history)),
            policy,
            grasp_mode=effective_mode.value,
            is_simulated=is_simulated,
        )

    def telemetry_dict(self, report: WatchdogReport) -> dict[str, Any]:
        """Return the telemetry triple (+ aggregates) for ``record.extra``.

        Telemetry only when the watchdog is not disabled. In disabled mode this
        returns an empty dict so callers can ``telemetry.update(...)``
        unconditionally.
        """

        if report.mode is WatchdogMode.DISABLED:
            return {}
        return {
            "drift_severity": report.drift_severity.value,
            "ood_flagged": bool(report.ood_flagged),
            "degraded_mode_active": bool(report.degraded_mode_active),
            "watchdog_aggregate_severity": report.aggregate_severity.value,
            "watchdog_recommended_action": report.recommended_action.value,
            "watchdog_enforced": bool(report.enforced),
        }

    @staticmethod
    def block_reason_code(report: WatchdogReport) -> DecisionReasonCode:
        """Pick drift vs OOD reason code for an enforced BLOCK_AUTO.

        OOD wins when its severity strictly dominates drift; ties resolve to drift
        (the five drift monitors are the primary calibration-quality signal).
        """

        order = WatchdogCoordinator._SEVERITY_ORDER
        if order[report.ood_severity] > order[report.drift_severity]:
            return DecisionReasonCode.OOD_BLOCKED_AUTO
        return DecisionReasonCode.DRIFT_BLOCKED_AUTO

    def synthesize_block_decision(
        self,
        *,
        report: WatchdogReport,
        effective_mode: "GraspMode",
        attempt_id: str,
        reobservation_count: int,
    ) -> DecisionReport:
        """Synthesize a typed FAIL_CLOSED decision for an enforced block."""

        return DecisionReport(
            action=DecisionAction.FAIL_CLOSED,
            reason_code=self.block_reason_code(report),
            uncertainty_score=None,
            threshold_used=0.0,
            mode=effective_mode.value,
            attempt_id=attempt_id,
            reobservation_count=reobservation_count,
            top_score=None,
        )

    def record_sample(
        self, history: list[WatchdogSample], sample: WatchdogSample
    ) -> None:
        """Append one sample to the rolling history, trimming on overflow."""

        history.append(sample)
        if len(history) > self.HISTORY_CEILING:
            # Drop oldest. List slicing is O(n) but n is bounded.
            del history[: len(history) - self.HISTORY_CEILING]

    @staticmethod
    def build_sample_from_decision(decision: DecisionReport) -> WatchdogSample:
        """Default per-tick sample built from the just-emitted decision.

        Captures only the fail-closed flag, the one signal derivable without new
        producer wiring. Every other drift signal reaches the watchdog only when a
        caller appends its own sample to ``service.watchdog_history``.
        """

        return WatchdogSample(
            fail_closed=bool(decision.action is DecisionAction.FAIL_CLOSED),
        )

    def _emit_event(
        self,
        event_listener: Optional["RobotWatchdogEventListener"],
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Fire a watchdog event, swallowing listener exceptions. Caller gates on
        the watchdog being not disabled."""

        if event_listener is None:
            return
        try:
            event_listener(event_type, dict(data))
        except Exception:  # pragma: no cover: best-effort emission
            # A buggy subscriber must not abort a pick.
            pass

    def _detect_transitions(
        self, *, prev: Optional[WatchdogReport], curr: WatchdogReport
    ) -> tuple[str, ...]:
        """Return canonical rising-edge event types fired by curr vs prev."""

        order = self._SEVERITY_ORDER
        prev_drift = order[prev.drift_severity] if prev is not None else 0
        prev_ood_flagged = bool(prev.ood_flagged) if prev is not None else False
        prev_degraded = (
            bool(prev.degraded_mode_active) if prev is not None else False
        )
        prev_block_enforced = bool(
            prev is not None
            and prev.enforced
            and prev.recommended_action is WatchdogAction.BLOCK_AUTO
        )

        events: list[str] = []
        if (
            prev_drift < order[DriftSeverity.MODERATE]
            and order[curr.drift_severity] >= order[DriftSeverity.MODERATE]
        ):
            events.append(RobotWatchdogEvent.DRIFT_DETECTED)
        if (not prev_ood_flagged) and bool(curr.ood_flagged):
            events.append(RobotWatchdogEvent.OOD_DETECTED)
        if (not prev_degraded) and bool(curr.degraded_mode_active):
            events.append(RobotWatchdogEvent.DEGRADED_MODE_ENGAGED)
        curr_block_enforced = bool(
            curr.enforced and curr.recommended_action is WatchdogAction.BLOCK_AUTO
        )
        if (not prev_block_enforced) and curr_block_enforced:
            events.append(RobotWatchdogEvent.BLOCK_AUTO_TRIGGERED)
        return tuple(events)

    def fire_transitions(
        self,
        *,
        event_listener: Optional["RobotWatchdogEventListener"],
        prev: Optional[WatchdogReport],
        curr: WatchdogReport,
        attempt_id: str,
        reobservation_count: int,
        effective_mode: "GraspMode",
    ) -> None:
        """Emit any rising-edge transitions on the listener.

        No-op when the watchdog mode is ``disabled`` or no listener is wired.
        """

        if curr.mode is WatchdogMode.DISABLED:
            return
        if event_listener is None:
            return
        transitions = self._detect_transitions(prev=prev, curr=curr)
        if not transitions:
            return
        payload: dict[str, Any] = {
            "attempt_id": attempt_id,
            "reobservation_count": reobservation_count,
            "grasp_mode": effective_mode.value,
            **curr.to_dict(),
        }
        for ev in transitions:
            self._emit_event(event_listener, ev, payload)
