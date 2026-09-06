"""Pre-execution decision policy.

This module is the single owner of the auto fail-closed decision
logic. It runs before the existing grasp/refinement/verification
pipeline and either:

* approves the current grasp candidates (``GRASP_NOW``),
* requests one bounded camera re-observation
  (``MOVE_CAMERA``, executed by
  :class:`AutonomousGraspService` inside its bounded loop),
* requests scene recovery (``RECOVER``: the engine only emits the
  typed report so the service can surface it to the operator), or
* refuses to execute (``FAIL_CLOSED``: only on real hardware, by
  operator-locked default).

Operator-locked defaults:

* ``auto_uncertainty_threshold = 0.4``: requires ``top_score >=
  0.6`` to proceed (1 - threshold).
* ``max_reobservations = 2``: bounded retry budget.
* ``reasons_penalty = 0.2``: added to the base uncertainty when the
  grasp result carries any failure reasons.
* ``fail_closed_on_real_hardware = True``: simulated runs stay
  permissive even when budget is exhausted (the engine reports
  ``GRASP_NOW`` and surfaces the would-have-been fail-closed reason in
  the structured log fields).

This module is pure: no perception, no hardware, no file I/O. It maps a typed
input bag onto a typed :class:`DecisionReport`. The one side effect is a single
log line per decision (see :func:`decide`); this gate is the last thing that can
stop a motion, so its verdict has to stay reconstructable per attempt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Optional

from src.robot.grasping.constants import (
    DECISION_LOG_FILE,
    create_grasping_logger,
)
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.uncertainty import (
    UncertaintySnapshot,
    apply_uncertainty_ranking_penalty,
)

logger = create_grasping_logger("DecisionEngine", DECISION_LOG_FILE)

#: Actions that mean "this attempt did not get permission to grasp"; logged at
#: warning level so a refusal never hides between two confident ticks.
_REFUSAL_ACTIONS: frozenset[str] = frozenset(
    {"fail_closed", "recover"}
)

__all__ = [
    "DecisionAction",
    "DecisionEngine",
    "DecisionPolicy",
    "DecisionReasonCode",
    "DecisionReport",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DecisionAction(StrEnum):
    """Terminal action the decision engine selects for a single tick.

    Values are stable strings so they round-trip through JSON
    telemetry without translation.
    """

    GRASP_NOW = "grasp_now"
    MOVE_CAMERA = "move_camera"
    RECOVER = "recover"
    FAIL_CLOSED = "fail_closed"


class DecisionReasonCode(StrEnum):
    """Machine-readable reason accompanying a :class:`DecisionAction`."""

    CONFIDENT_GRASP = "confident_grasp"
    LOW_CONFIDENCE = "low_confidence"
    NO_CANDIDATES = "no_candidates"
    REOBSERVE_BUDGET_EXHAUSTED = "reobserve_budget_exhausted"
    REOBSERVE_PLANNER_UNAVAILABLE = "reobserve_planner_unavailable"
    DECISION_DISABLED = "decision_disabled"
    # Fused-uncertainty driven reasons.
    UNCERTAINTY_FAIL_CLOSED = "uncertainty_fail_closed"
    CHANNEL_DISAGREEMENT = "channel_disagreement"
    # Drift + OOD watchdog blocks auto on real hardware at
    # ``DriftSeverity.HIGH`` and ``DriftSeverity.SEVERE``. Distinct codes so
    # replay can attribute the fail-closure cleanly.
    DRIFT_BLOCKED_AUTO = "drift_blocked_auto"
    OOD_BLOCKED_AUTO = "ood_blocked_auto"


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    """Bounded configuration for the auto fail-closed decision layer.

    All numeric fields are validated at construction so a misconfigured
    YAML can never let a non-finite threshold slip through.
    """

    enabled: bool = True
    auto_uncertainty_threshold: float = 0.4
    max_reobservations: int = 2
    reasons_penalty: float = 0.2
    fail_closed_on_real_hardware: bool = True

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.auto_uncertainty_threshold) <= 1.0):
            raise ValueError(
                "DecisionPolicy.auto_uncertainty_threshold must be in "
                f"[0.0, 1.0], got {self.auto_uncertainty_threshold!r}"
            )
        if int(self.max_reobservations) < 0:
            raise ValueError(
                "DecisionPolicy.max_reobservations must be >= 0, got "
                f"{self.max_reobservations!r}"
            )
        if not (0.0 <= float(self.reasons_penalty) <= 1.0):
            raise ValueError(
                "DecisionPolicy.reasons_penalty must be in [0.0, 1.0], "
                f"got {self.reasons_penalty!r}"
            )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DecisionReport:
    """Typed decision outcome carrying the machine-readable log contract.

    The six mandatory fields surface verbatim in :meth:`to_dict`:
    ``decision_action``, ``decision_reason_code``, ``uncertainty_score``,
    ``threshold_used``, ``mode``, ``attempt_id``. ``reobservation_count``
    (camera re-observations before this decision) and ``top_score`` (the
    top GraspResult score, when known) are serialised beside them for
    downstream debugging, ahead of the fused-uncertainty fields below.
    """

    action: DecisionAction
    reason_code: DecisionReasonCode
    uncertainty_score: Optional[float]
    threshold_used: float
    mode: str
    attempt_id: str
    reobservation_count: int
    top_score: Optional[float] = None
    # Fused-uncertainty plumbing. ``uncertainty_source`` is ``"fused"`` when
    # ``snapshot.fused`` drove the threshold check, ``"legacy"`` when the
    # engine used ``(1 - top_score) + reasons_penalty``, and
    # ``"legacy_fallback"`` when that formula ran with the master gate on and
    # a snapshot whose ``fused_available`` was false. ``uncertainty_fused``
    # carries ``snapshot.fused`` (or ``None`` when the snapshot was
    # unavailable). ``ranking_penalty_applied`` records whether the
    # scene-level penalty shifted ``penalised_top_score``.
    uncertainty_source: str = "legacy"
    uncertainty_fused: Optional[float] = None
    uncertainty_fused_available: bool = False
    ranking_penalty_applied: bool = False
    penalised_top_score: Optional[float] = None
    channel_disagreement: Optional[float] = None
    channel_disagreement_triggered: bool = False

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict mirroring the log contract."""

        return {
            "decision_action": self.action.value,
            "decision_reason_code": self.reason_code.value,
            "uncertainty_score": (
                float(self.uncertainty_score)
                if self.uncertainty_score is not None
                else None
            ),
            "threshold_used": float(self.threshold_used),
            "mode": str(self.mode),
            "attempt_id": str(self.attempt_id),
            "reobservation_count": int(self.reobservation_count),
            "top_score": (
                float(self.top_score) if self.top_score is not None else None
            ),
            # Fused-uncertainty telemetry surfaces.
            "uncertainty_source": str(self.uncertainty_source),
            "uncertainty_fused": (
                float(self.uncertainty_fused)
                if self.uncertainty_fused is not None
                else None
            ),
            "uncertainty_fused_available": bool(self.uncertainty_fused_available),
            "ranking_penalty_applied": bool(self.ranking_penalty_applied),
            "penalised_top_score": (
                float(self.penalised_top_score)
                if self.penalised_top_score is not None
                else None
            ),
            "channel_disagreement": (
                float(self.channel_disagreement)
                if self.channel_disagreement is not None
                else None
            ),
            "channel_disagreement_triggered": bool(
                self.channel_disagreement_triggered
            ),
        }


# ---------------------------------------------------------------------------
# Terminal selectors (pure, shared by the fused + legacy paths)
# ---------------------------------------------------------------------------


def _permissive_reason(viewpoint_planner_available: bool) -> DecisionReasonCode:
    """Reason code when auto cannot re-observe and downgrades permissively.

    ``REOBSERVE_PLANNER_UNAVAILABLE`` iff there is no viewpoint planner;
    otherwise ``REOBSERVE_BUDGET_EXHAUSTED``. Keyed on
    ``viewpoint_planner_available`` alone, never on ``budget_left``: the
    caller has already established that re-observation is impossible.
    """

    return (
        DecisionReasonCode.REOBSERVE_PLANNER_UNAVAILABLE
        if not viewpoint_planner_available
        else DecisionReasonCode.REOBSERVE_BUDGET_EXHAUSTED
    )


def _fail_closed_action(fail_closed_active: bool) -> DecisionAction:
    """Terminal action when re-observation is impossible.

    ``FAIL_CLOSED`` on real hardware, which is the operator-locked default.
    Otherwise the permissive ``GRASP_NOW`` downgrade, which covers simulation
    and a disabled ``fail_closed_on_real_hardware``.
    """

    return (
        DecisionAction.FAIL_CLOSED
        if fail_closed_active
        else DecisionAction.GRASP_NOW
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DecisionEngine:
    """Pure, stateless wrapper around :class:`DecisionPolicy`.

    The caller threads the re-observation count and ``attempt_id`` through
    each :meth:`decide` call, so one instance drives many concurrent picks
    safely.
    """

    policy: DecisionPolicy

    def decide(
        self,
        *,
        grasp_result: Optional[GraspResult],
        mode: str,
        attempt_id: str,
        reobservation_count: int,
        is_simulated: bool,
        viewpoint_planner_available: bool,
        uncertainty_snapshot: Optional[UncertaintySnapshot] = None,
        uncertainty_active: bool = False,
        ranking_penalty_weight: float = 0.0,
    ) -> DecisionReport:
        """Return a typed decision for the current perception tick.

        ``grasp_result`` may be :data:`None` when the calculator emits
        no result at all (e.g. an empty perception frame). When the
        result has zero candidates the engine treats it identically to
        :data:`None` and routes to ``RECOVER`` with
        :attr:`DecisionReasonCode.NO_CANDIDATES`.

        ``is_simulated`` mirrors
        ``arm.capabilities.is_simulated``. The engine only fail-closes
        on real hardware (operator-locked default).

        Fused-uncertainty parameters (all default to legacy behaviour):

        * ``uncertainty_snapshot``: fused-channel snapshot from
          :func:`fuse_uncertainty`. When :data:`None` (or
          ``fused_available`` is :data:`False`) the engine falls back
          to the legacy ``(1 - top_score) + reasons_penalty`` formula.
        * ``uncertainty_active``: master gate. When :data:`False`
          the snapshot is ignored entirely and the engine emits the
          baseline behaviour byte-identically.
        * ``ranking_penalty_weight``: scene-level penalty subtracted
          from ``top_score`` to produce ``penalised_top_score``. The
          penalty does not participate in the threshold check.
        """

        threshold = float(self.policy.auto_uncertainty_threshold)
        mode_str = str(mode)
        attempt_str = str(attempt_id)

        def _report(
            *,
            action: DecisionAction,
            reason_code: DecisionReasonCode,
            uncertainty_score: Optional[float],
            top_score: Optional[float],
            carrier: dict[str, Any],
        ) -> DecisionReport:
            # Shared terminal builder. The immutable per-tick scalars
            # (threshold / mode / attempt_id / reobservation_count) are closed
            # over; the per-path carrier block (``legacy_common`` vs
            # ``fused_common_kwargs``, or ``{}`` for no-candidates) is passed
            # verbatim by the caller and spread, never merged across paths.
            report = DecisionReport(
                action=action,
                reason_code=reason_code,
                uncertainty_score=uncertainty_score,
                threshold_used=threshold,
                mode=mode_str,
                attempt_id=attempt_str,
                reobservation_count=int(reobservation_count),
                top_score=top_score,
                **carrier,
            )
            # Every branch returns through this builder, so this one call logs
            # the whole decision surface.
            logger.log(
                logging.WARNING if action in _REFUSAL_ACTIONS else logging.INFO,
                "attempt=%s mode=%s -> %s (%s): uncertainty=%s threshold=%.3f "
                "top_score=%s reobservations=%d source=%s",
                attempt_str,
                mode_str,
                action.value,
                reason_code.value,
                "n/a" if uncertainty_score is None else f"{uncertainty_score:.3f}",
                threshold,
                "n/a" if top_score is None else f"{top_score:.3f}",
                int(reobservation_count),
                carrier.get("uncertainty_source", "none"),
            )
            return report

        # ---- No candidates branch -----------------------------------------
        has_candidates = (
            grasp_result is not None and len(grasp_result.candidates) > 0
        )
        if not has_candidates:
            return _report(
                action=DecisionAction.RECOVER,
                reason_code=DecisionReasonCode.NO_CANDIDATES,
                uncertainty_score=None,
                top_score=None,
                carrier={},
            )

        # `grasp_result` is narrowed above.
        assert grasp_result is not None
        top_score = float(grasp_result.top_score)

        # ---- Fused-uncertainty path ----
        use_fused = bool(
            uncertainty_active
            and uncertainty_snapshot is not None
            and uncertainty_snapshot.fused_available
        )
        fail_closed_active = (
            (not is_simulated) and self.policy.fail_closed_on_real_hardware
        )

        if use_fused:
            # Mypy: narrowed above.
            assert uncertainty_snapshot is not None
            snapshot = uncertainty_snapshot
            decision_uncertainty = float(snapshot.fused)
            penalised_top_score, ranking_applied = (
                apply_uncertainty_ranking_penalty(
                    top_score, snapshot, float(ranking_penalty_weight)
                )
            )
            fused_common_kwargs: dict[str, Any] = dict(
                uncertainty_source="fused",
                uncertainty_fused=float(snapshot.fused),
                uncertainty_fused_available=True,
                ranking_penalty_applied=bool(ranking_applied),
                penalised_top_score=penalised_top_score,
                channel_disagreement=float(snapshot.disagreement),
                channel_disagreement_triggered=bool(
                    snapshot.disagreement_triggered
                ),
            )
            budget_left = (
                int(reobservation_count) < int(self.policy.max_reobservations)
            )

            # Disagreement gate fires first: it represents inconsistent
            # perception channels and must not be masked by a confident
            # top score.
            if snapshot.disagreement_triggered:
                if budget_left and viewpoint_planner_available:
                    return _report(
                        action=DecisionAction.MOVE_CAMERA,
                        reason_code=DecisionReasonCode.CHANNEL_DISAGREEMENT,
                        uncertainty_score=decision_uncertainty,
                        top_score=top_score,
                        carrier=fused_common_kwargs,
                    )
                action = _fail_closed_action(fail_closed_active)
                return _report(
                    action=action,
                    reason_code=DecisionReasonCode.CHANNEL_DISAGREEMENT,
                    uncertainty_score=decision_uncertainty,
                    top_score=top_score,
                    carrier=fused_common_kwargs,
                )

            # Confident grasp under fused signal.
            if decision_uncertainty <= threshold:
                return _report(
                    action=DecisionAction.GRASP_NOW,
                    reason_code=DecisionReasonCode.CONFIDENT_GRASP,
                    uncertainty_score=decision_uncertainty,
                    top_score=top_score,
                    carrier=fused_common_kwargs,
                )

            # Low confidence: re-observe if possible, else fail-close
            # with the fused-path reason code.
            if budget_left and viewpoint_planner_available:
                return _report(
                    action=DecisionAction.MOVE_CAMERA,
                    reason_code=DecisionReasonCode.LOW_CONFIDENCE,
                    uncertainty_score=decision_uncertainty,
                    top_score=top_score,
                    carrier=fused_common_kwargs,
                )
            if fail_closed_active:
                return _report(
                    action=DecisionAction.FAIL_CLOSED,
                    reason_code=DecisionReasonCode.UNCERTAINTY_FAIL_CLOSED,
                    uncertainty_score=decision_uncertainty,
                    top_score=top_score,
                    carrier=fused_common_kwargs,
                )
            # Permissive downgrade (sim or fail_closed disabled).
            permissive_reason = _permissive_reason(viewpoint_planner_available)
            return _report(
                action=DecisionAction.GRASP_NOW,
                reason_code=permissive_reason,
                uncertainty_score=decision_uncertainty,
                top_score=top_score,
                carrier=fused_common_kwargs,
            )

        # ---- Legacy path ----
        # Either ``uncertainty_active`` is False or the snapshot is
        # unavailable; fall back to ``(1 - top_score) + reasons_penalty``.
        legacy_source = (
            "legacy_fallback"
            if uncertainty_active and uncertainty_snapshot is not None
            else "legacy"
        )
        legacy_fused = (
            float(uncertainty_snapshot.fused)
            if (uncertainty_snapshot is not None and uncertainty_snapshot.fused_available)
            else None
        )
        legacy_fused_available = bool(
            uncertainty_snapshot is not None and uncertainty_snapshot.fused_available
        )
        legacy_disagreement = (
            float(uncertainty_snapshot.disagreement)
            if uncertainty_snapshot is not None
            else None
        )
        legacy_disagreement_triggered = bool(
            uncertainty_snapshot is not None
            and uncertainty_snapshot.disagreement_triggered
        )

        base = max(0.0, 1.0 - top_score)
        penalty = (
            float(self.policy.reasons_penalty)
            if grasp_result.reasons
            else 0.0
        )
        uncertainty = min(1.0, base + penalty)

        legacy_common: dict[str, Any] = dict(
            uncertainty_source=legacy_source,
            uncertainty_fused=legacy_fused,
            uncertainty_fused_available=legacy_fused_available,
            ranking_penalty_applied=False,
            penalised_top_score=None,
            channel_disagreement=legacy_disagreement,
            channel_disagreement_triggered=legacy_disagreement_triggered,
        )

        # ---- Confident grasp path -----------------------------------------
        if uncertainty <= threshold:
            return _report(
                action=DecisionAction.GRASP_NOW,
                reason_code=DecisionReasonCode.CONFIDENT_GRASP,
                uncertainty_score=uncertainty,
                top_score=top_score,
                carrier=legacy_common,
            )

        # ---- Low confidence: try to re-observe ----------------------------
        budget_left = (
            int(reobservation_count) < int(self.policy.max_reobservations)
        )

        if budget_left and viewpoint_planner_available:
            return _report(
                action=DecisionAction.MOVE_CAMERA,
                reason_code=DecisionReasonCode.LOW_CONFIDENCE,
                uncertainty_score=uncertainty,
                top_score=top_score,
                carrier=legacy_common,
            )

        # ---- Cannot re-observe: fail-closed or permissive downgrade -------
        reason = _permissive_reason(viewpoint_planner_available)
        action = _fail_closed_action(fail_closed_active)
        return _report(
            action=action,
            reason_code=reason,
            uncertainty_score=uncertainty,
            top_score=top_score,
            carrier=legacy_common,
        )
