"""Latency / SLO telemetry coordinator.

Owns the rolling per-stage latency window and breach state and emits ``SLO_BREACH``
on rising-edge breaches. Decoupled from the service: the effective config and event
listener are passed per call, so the coordinator holds no service references, only
its own rolling state. The runtime never fail-closes on its own SLO; the listener
decides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from src.robot.constants import GRASP_LATENCY_LOG_FILE, create_robot_logger
from src.robot.events import RobotWatchdogEvent
from src.robot.grasping.telemetry.latency_tracker import LatencyStage, LatencyTracker

if TYPE_CHECKING:
    from src.robot.events import RobotWatchdogEventListener

    from .config import EffectiveGraspingConfig, GraspMode


# Rising-edge only, so this file stays quiet while a cell is healthy: one line the moment a stage's
# p95 crosses its SLO, and nothing per attempt. The runtime never fail-closes on its own SLO, so
# the log is where a breach is written down.
logger = create_robot_logger("LatencyTelemetry", GRASP_LATENCY_LOG_FILE)


class LatencyTelemetryCoordinator:
    """Rolling-window per-stage latency tracking + SLO-breach emission."""

    def __init__(self) -> None:
        self._history: dict[str, list[float]] = {
            "decision": [],
            "ranking": [],
            "fusion": [],
        }
        self._breach_state: dict[str, bool] = {
            "decision": False,
            "ranking": False,
            "fusion": False,
        }

    @staticmethod
    def _p95(samples: list[float]) -> float:
        """Tiny p95 estimator over a non-empty list (no numpy import here)."""

        if not samples:
            return 0.0
        ordered = sorted(samples)
        # Nearest-rank p95.
        rank = max(int(round(0.95 * len(ordered))) - 1, 0)
        return float(ordered[rank])

    def update_and_emit(
        self,
        *,
        config: Optional["EffectiveGraspingConfig"],
        event_listener: Optional["RobotWatchdogEventListener"],
        tracker: Optional[LatencyTracker],
        attempt_id: str,
        effective_mode: "GraspMode",
    ) -> None:
        """Append the latest spans to a rolling window and fire breaches.

        The window length is ``performance.breach_window_size`` (default 32).
        Emission is gated on ``performance_enabled`` and
        ``performance_emit_breach_events`` and a wired ``event_listener``. A
        missing stage span never counts toward the rolling p95.
        """

        if tracker is None:
            return
        cfg = config
        if cfg is None:
            return
        history = self._history
        window = max(int(cfg.performance.breach_window_size), 2)

        # Three stage tuples: (stage, history_key, slo_ms).
        stages: tuple[tuple[LatencyStage, str, float], ...] = (
            (
                LatencyStage.DECISION,
                "decision",
                float(cfg.performance.decision_latency_slo_ms),
            ),
            (
                LatencyStage.RANKING,
                "ranking",
                float(cfg.performance.ranking_latency_slo_ms),
            ),
            (
                LatencyStage.FUSION,
                "fusion",
                float(cfg.performance.fusion_latency_slo_ms),
            ),
        )
        breach_state = self._breach_state
        breached_now: list[dict[str, Any]] = []
        for stage, key, slo_ms in stages:
            sample = tracker.get(stage)
            if sample is None:
                # Missing spans never count toward the rolling p95.
                continue
            buf = history[key]
            buf.append(float(sample))
            if len(buf) > window:
                del buf[: len(buf) - window]
            # Only evaluate once the window is full so a single cold-start
            # outlier cannot fire a breach.
            if len(buf) < window:
                continue
            p95 = self._p95(buf)
            breached = p95 > slo_ms
            was_breached = breach_state.get(key, False)
            breach_state[key] = breached
            if breached and not was_breached:
                breached_now.append(
                    {
                        "stage": key,
                        "p95_ms": float(p95),
                        "slo_ms": float(slo_ms),
                        "window_size": int(window),
                    }
                )

        if not breached_now:
            return
        if not bool(cfg.performance.enabled):
            return
        # Logged before the emission gates: a breach with no listener wired (or with event emission
        # turned off) is still a breach, and the log is then the only place it exists.
        for breach in breached_now:
            logger.warning(
                "SLO breach on %s during attempt %s (mode=%s): p95 %.1f ms > %.1f ms over %d samples",
                breach["stage"], attempt_id, effective_mode.value,
                breach["p95_ms"], breach["slo_ms"], breach["window_size"],
            )
        if not bool(cfg.performance.emit_breach_events):
            return
        if event_listener is None:
            return
        for breach in breached_now:
            payload: dict[str, Any] = {
                "attempt_id": attempt_id,
                "grasp_mode": effective_mode.value,
                **breach,
            }
            try:
                event_listener(RobotWatchdogEvent.SLO_BREACH, payload)
            except Exception:  # pragma: no cover (best-effort emission)
                pass
