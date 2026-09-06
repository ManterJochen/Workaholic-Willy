"""Drift + OOD watchdog (pure-function evaluators).

This module owns the typed surface for the production calibration drift and
out-of-distribution watchdog.

Design locks:

* Pure-function evaluators. The watchdog itself holds no state. The caller
  (``AutonomousGraspService``) owns the rolling window of
  :class:`WatchdogSample` records and hands an immutable
  :class:`WatchdogHistory` slice to :func:`evaluate_watchdog` on every tick.
  Replay tooling re-runs the same code over the same record stream.
* The default mode is ``shadow``. The watchdog computes severity and emits
  telemetry but never alters runtime behaviour until the operator escalates to
  ``canary`` or ``active``. A fourth state ``disabled`` short-circuits
  evaluation entirely and omits telemetry.
* Typed outcomes. ``DRIFT_BLOCKED_AUTO`` and ``OOD_BLOCKED_AUTO`` live in
  :class:`AutonomousGraspOutcome` so replay can attribute the fail-closed
  cause cleanly.
* Pre-emptive gate. The recommended action is computed before
  ``DecisionEngine.decide`` is called. On HIGH/SEVERE the service
  short-circuits to a synthetic FAIL_CLOSED decision report carrying that
  reason code.
* Moderate severity recommends REOBSERVE. The service honours this only when
  a viewpoint planner and budget are available; otherwise it proceeds with
  telemetry.
* OOD has its own severity ladder mirroring drift.
* Hardware lock: HIGH/SEVERE block the auto grasp mode on real hardware. In
  simulated runs the block is downgraded to a degraded-mode telemetry flag.

All types are frozen + slotted, JSON-safe via :meth:`to_dict()`, and strictly
validated at construction so no non-finite knob or out-of-range threshold can
slip through from a YAML.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Optional, Sequence

__all__ = [
    "DriftSeverity",
    "WatchdogMode",
    "WatchdogAction",
    "WatchdogPolicy",
    "WatchdogSample",
    "WatchdogHistory",
    "DriftMonitorBreakdown",
    "WatchdogReport",
    "evaluate_drift",
    "evaluate_ood",
    "evaluate_watchdog",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DriftSeverity(StrEnum):
    """Five-level severity ladder shared by drift and OOD evaluators.

    Values match the strings reserved by ``telemetry_catalog`` at
    ``_DRIFT_SEVERITY_VALUES`` so the audit pipeline accepts them
    without extension.
    """

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


_SEVERITY_ORDER: tuple[DriftSeverity, ...] = (
    DriftSeverity.NONE,
    DriftSeverity.LOW,
    DriftSeverity.MODERATE,
    DriftSeverity.HIGH,
    DriftSeverity.SEVERE,
)


def _severity_rank(s: DriftSeverity) -> int:
    return _SEVERITY_ORDER.index(s)


def _max_severity(*values: DriftSeverity) -> DriftSeverity:
    best = DriftSeverity.NONE
    for v in values:
        if _severity_rank(v) > _severity_rank(best):
            best = v
    return best


class WatchdogMode(StrEnum):
    """Lifecycle of the watchdog subsystem.

    * ``disabled``: short-circuit; do not evaluate, do not emit
      watchdog telemetry.
    * ``shadow``: evaluate and emit telemetry only; never alter
      runtime behaviour. The default.
    * ``canary``: evaluate, emit, and enforce on real hardware in the
      auto grasp mode only. Sim and other modes get telemetry without
      behavioural change.
    * ``active``: evaluate, emit, and enforce on every configured
      ``block_mode`` on real hardware.
    """

    DISABLED = "disabled"
    SHADOW = "shadow"
    CANARY = "canary"
    ACTIVE = "active"


class WatchdogAction(StrEnum):
    """Recommended action surfaced on the :class:`WatchdogReport`."""

    NONE = "none"
    REOBSERVE = "reobserve"
    BLOCK_AUTO = "block_auto"


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def _validate_finite_non_negative(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a real number; got {value!r}")
    fv = float(value)
    if math.isnan(fv) or math.isinf(fv):
        raise ValueError(f"{name} must be finite; got {value!r}")
    if fv < 0.0:
        raise ValueError(f"{name} must be non-negative; got {value!r}")
    return fv


def _validate_unit_strict(name: str, value: float) -> float:
    fv = _validate_finite_non_negative(name, value)
    if fv > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]; got {value!r}")
    return fv


def _validate_monotone_thresholds(name: str, *values: float) -> None:
    """Assert that thresholds form a non-decreasing ladder."""

    prev = values[0]
    for cur in values[1:]:
        if cur < prev:
            raise ValueError(
                f"{name} thresholds must be non-decreasing; got "
                f"{values!r}"
            )
        prev = cur


@dataclass(frozen=True, slots=True)
class WatchdogPolicy:
    """Operator-locked thresholds for the drift + OOD watchdog.

    Defaults are conservative: they fire MODERATE/HIGH only on
    clearly degraded inputs so a healthy production scene stays in
    severity ``NONE``. Thresholds form non-decreasing ladders;
    a misconfigured YAML is rejected at construction.
    """

    mode: WatchdogMode = WatchdogMode.SHADOW
    window_size: int = 20
    min_samples_for_trend: int = 6

    # Predicted-vs-observed calibration delta (mm, latest sample).
    calibration_delta_moderate_mm: float = 2.0
    calibration_delta_high_mm: float = 5.0
    calibration_delta_severe_mm: float = 10.0

    # Depth-confidence distribution shift (|recent-half mean - older-half mean|).
    depth_confidence_shift_moderate: float = 0.05
    depth_confidence_shift_high: float = 0.10
    depth_confidence_shift_severe: float = 0.20

    # Fail-closed rate in the rolling window.
    fail_closed_rate_moderate: float = 0.30
    fail_closed_rate_high: float = 0.50
    fail_closed_rate_severe: float = 0.75

    # Hand-eye residual trend (recent-half mean - older-half mean, mm).
    hand_eye_residual_trend_moderate_mm: float = 1.0
    hand_eye_residual_trend_high_mm: float = 3.0
    hand_eye_residual_trend_severe_mm: float = 7.0

    # Verification residual trend (recent-half mean - older-half mean, mm).
    verification_residual_trend_moderate_mm: float = 0.5
    verification_residual_trend_high_mm: float = 2.0
    verification_residual_trend_severe_mm: float = 5.0

    # OOD score (1.0 = strongly in-distribution, 0.0 = far OOD).
    # The ladder is inverted: smaller scores mean higher severity.
    ood_score_moderate: float = 0.5
    ood_score_high: float = 0.3
    ood_score_severe: float = 0.1

    # Block the auto grasp mode on real hardware at these severities.
    block_severities: tuple[DriftSeverity, ...] = (
        DriftSeverity.HIGH,
        DriftSeverity.SEVERE,
    )
    # Modes that fall under the block on real hardware. ``auto`` is the
    # only mode blocked by default; the tuple shape lets operators widen
    # the block from config, without a code change. Strings match
    # :class:`src.robot.grasping.types.modes.GraspMode` values
    # (lowercase: ``"auto"``, ``"dense_clutter"``, ...).
    block_modes: tuple[str, ...] = ("auto",)

    def __post_init__(self) -> None:
        # window
        if int(self.window_size) < 1:
            raise ValueError(
                f"WatchdogPolicy.window_size must be >= 1; got "
                f"{self.window_size!r}"
            )
        if int(self.min_samples_for_trend) < 2:
            raise ValueError(
                "WatchdogPolicy.min_samples_for_trend must be >= 2; "
                f"got {self.min_samples_for_trend!r}"
            )
        # ladders
        _validate_monotone_thresholds(
            "calibration_delta",
            _validate_finite_non_negative(
                "calibration_delta_moderate_mm",
                self.calibration_delta_moderate_mm,
            ),
            _validate_finite_non_negative(
                "calibration_delta_high_mm",
                self.calibration_delta_high_mm,
            ),
            _validate_finite_non_negative(
                "calibration_delta_severe_mm",
                self.calibration_delta_severe_mm,
            ),
        )
        _validate_monotone_thresholds(
            "depth_confidence_shift",
            _validate_unit_strict(
                "depth_confidence_shift_moderate",
                self.depth_confidence_shift_moderate,
            ),
            _validate_unit_strict(
                "depth_confidence_shift_high",
                self.depth_confidence_shift_high,
            ),
            _validate_unit_strict(
                "depth_confidence_shift_severe",
                self.depth_confidence_shift_severe,
            ),
        )
        _validate_monotone_thresholds(
            "fail_closed_rate",
            _validate_unit_strict(
                "fail_closed_rate_moderate",
                self.fail_closed_rate_moderate,
            ),
            _validate_unit_strict(
                "fail_closed_rate_high",
                self.fail_closed_rate_high,
            ),
            _validate_unit_strict(
                "fail_closed_rate_severe",
                self.fail_closed_rate_severe,
            ),
        )
        _validate_monotone_thresholds(
            "hand_eye_residual_trend",
            _validate_finite_non_negative(
                "hand_eye_residual_trend_moderate_mm",
                self.hand_eye_residual_trend_moderate_mm,
            ),
            _validate_finite_non_negative(
                "hand_eye_residual_trend_high_mm",
                self.hand_eye_residual_trend_high_mm,
            ),
            _validate_finite_non_negative(
                "hand_eye_residual_trend_severe_mm",
                self.hand_eye_residual_trend_severe_mm,
            ),
        )
        _validate_monotone_thresholds(
            "verification_residual_trend",
            _validate_finite_non_negative(
                "verification_residual_trend_moderate_mm",
                self.verification_residual_trend_moderate_mm,
            ),
            _validate_finite_non_negative(
                "verification_residual_trend_high_mm",
                self.verification_residual_trend_high_mm,
            ),
            _validate_finite_non_negative(
                "verification_residual_trend_severe_mm",
                self.verification_residual_trend_severe_mm,
            ),
        )
        # OOD ladder is inverted: smaller score means higher severity, so the
        # same unit helper runs per threshold and the order check below is
        # reversed.
        for nm, v in (
            ("ood_score_moderate", self.ood_score_moderate),
            ("ood_score_high", self.ood_score_high),
            ("ood_score_severe", self.ood_score_severe),
        ):
            _validate_unit_strict(nm, v)
        if not (
            self.ood_score_moderate
            >= self.ood_score_high
            >= self.ood_score_severe
        ):
            raise ValueError(
                "OOD thresholds must satisfy moderate >= high >= "
                f"severe; got "
                f"({self.ood_score_moderate}, "
                f"{self.ood_score_high}, "
                f"{self.ood_score_severe})"
            )
        for s in self.block_severities:
            if not isinstance(s, DriftSeverity):
                raise ValueError(
                    "WatchdogPolicy.block_severities entries must be "
                    f"DriftSeverity; got {s!r}"
                )
        for m in self.block_modes:
            if not isinstance(m, str) or not m:
                raise ValueError(
                    "WatchdogPolicy.block_modes entries must be "
                    f"non-empty strings; got {m!r}"
                )


# ---------------------------------------------------------------------------
# Samples and history
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WatchdogSample:
    """One attempt's worth of raw drift+OOD signals.

    All fields are ``Optional``: a sample may come from a pipeline that
    does not produce every signal. A missing signal is skipped by the
    corresponding monitor.

    Coordinate / unit conventions:

    * ``calibration_residual_mm``: hand-eye residual reported by
      the most recent calibration quality probe, in millimetres.
    * ``verification_residual_mm``: post-grasp verification pose
      error, in millimetres.
    * ``predicted_observed_calibration_delta_mm``: |predicted
      target pose - observed target pose|, in millimetres.
    * ``depth_confidence_mean``: mean per-pixel depth confidence
      in ``[0, 1]``.
    * ``fail_closed``: whether the decision engine fail-closed on
      this attempt (drives the fail-closed-spike monitor).
    * ``ood_score``: scene-level confidence validity in ``[0, 1]``;
      ``1.0`` is strongly in-distribution, ``0.0`` is far OOD.
    """

    calibration_residual_mm: Optional[float] = None
    verification_residual_mm: Optional[float] = None
    predicted_observed_calibration_delta_mm: Optional[float] = None
    depth_confidence_mean: Optional[float] = None
    fail_closed: Optional[bool] = None
    ood_score: Optional[float] = None

    def __post_init__(self) -> None:
        for nm, v in (
            ("calibration_residual_mm", self.calibration_residual_mm),
            ("verification_residual_mm", self.verification_residual_mm),
            (
                "predicted_observed_calibration_delta_mm",
                self.predicted_observed_calibration_delta_mm,
            ),
        ):
            if v is None:
                continue
            _validate_finite_non_negative(nm, v)
        for nm, v in (
            ("depth_confidence_mean", self.depth_confidence_mean),
            ("ood_score", self.ood_score),
        ):
            if v is None:
                continue
            _validate_unit_strict(nm, v)
        if self.fail_closed is not None and not isinstance(
            self.fail_closed, bool
        ):
            raise ValueError(
                "WatchdogSample.fail_closed must be bool or None; "
                f"got {self.fail_closed!r}"
            )


@dataclass(frozen=True, slots=True)
class WatchdogHistory:
    """Immutable ordered window of recent :class:`WatchdogSample`.

    The caller (``AutonomousGraspService``) appends one sample per
    attempt and constructs a fresh :class:`WatchdogHistory` snapshot
    before each evaluation. Oldest sample first; newest sample last.
    """

    samples: tuple[WatchdogSample, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.samples, tuple):
            raise ValueError(
                "WatchdogHistory.samples must be a tuple; got "
                f"{type(self.samples).__name__}"
            )
        for i, s in enumerate(self.samples):
            if not isinstance(s, WatchdogSample):
                raise ValueError(
                    f"WatchdogHistory.samples[{i}] must be "
                    f"WatchdogSample; got {type(s).__name__}"
                )

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def latest(self) -> Optional[WatchdogSample]:
        return self.samples[-1] if self.samples else None

    def trimmed(self, window_size: int) -> "WatchdogHistory":
        """Return a new history limited to the last ``window_size`` samples."""

        if window_size <= 0 or len(self.samples) <= window_size:
            return self
        return WatchdogHistory(self.samples[-window_size:])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect(
    samples: Sequence[WatchdogSample], field_name: str
) -> list[float]:
    out: list[float] = []
    for s in samples:
        v = getattr(s, field_name)
        if v is None:
            continue
        out.append(float(v))
    return out


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _half_split(values: Sequence[float]) -> tuple[list[float], list[float]]:
    """Split ``values`` into older-half and recent-half.

    For odd lengths the middle sample lands in the recent half so
    the trend monitors react slightly faster to the newest data.
    """

    n = len(values)
    mid = n // 2
    return list(values[:mid]), list(values[mid:])


def _severity_from_ascending(
    value: float, moderate: float, high: float, severe: float
) -> DriftSeverity:
    """Pick severity for monitors where larger values are worse."""

    if value >= severe:
        return DriftSeverity.SEVERE
    if value >= high:
        return DriftSeverity.HIGH
    if value >= moderate:
        return DriftSeverity.MODERATE
    # No explicit LOW band: LOW is reserved for monitors that want a
    # quieter early-warning step, and no monitor uses one.
    return DriftSeverity.NONE


def _severity_from_descending(
    value: float, moderate: float, high: float, severe: float
) -> DriftSeverity:
    """Pick severity for monitors where smaller values are worse (OOD)."""

    if value <= severe:
        return DriftSeverity.SEVERE
    if value <= high:
        return DriftSeverity.HIGH
    if value <= moderate:
        return DriftSeverity.MODERATE
    return DriftSeverity.NONE


# ---------------------------------------------------------------------------
# Monitors
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DriftMonitorBreakdown:
    """Per-monitor severity breakdown surfaced on :class:`WatchdogReport`.

    All five monitors emit a :class:`DriftSeverity`; the aggregate
    drift severity is the maximum of the five.
    """

    calibration_delta: DriftSeverity = DriftSeverity.NONE
    depth_confidence_shift: DriftSeverity = DriftSeverity.NONE
    fail_closed_spike: DriftSeverity = DriftSeverity.NONE
    hand_eye_residual_trend: DriftSeverity = DriftSeverity.NONE
    verification_residual_trend: DriftSeverity = DriftSeverity.NONE

    def to_dict(self) -> dict[str, str]:
        return {
            "calibration_delta": self.calibration_delta.value,
            "depth_confidence_shift": self.depth_confidence_shift.value,
            "fail_closed_spike": self.fail_closed_spike.value,
            "hand_eye_residual_trend": (
                self.hand_eye_residual_trend.value
            ),
            "verification_residual_trend": (
                self.verification_residual_trend.value
            ),
        }

    def aggregate(self) -> DriftSeverity:
        return _max_severity(
            self.calibration_delta,
            self.depth_confidence_shift,
            self.fail_closed_spike,
            self.hand_eye_residual_trend,
            self.verification_residual_trend,
        )


def _monitor_calibration_delta(
    history: WatchdogHistory, policy: WatchdogPolicy
) -> DriftSeverity:
    latest = history.latest
    if latest is None or latest.predicted_observed_calibration_delta_mm is None:
        return DriftSeverity.NONE
    return _severity_from_ascending(
        latest.predicted_observed_calibration_delta_mm,
        policy.calibration_delta_moderate_mm,
        policy.calibration_delta_high_mm,
        policy.calibration_delta_severe_mm,
    )


def _monitor_depth_confidence_shift(
    history: WatchdogHistory, policy: WatchdogPolicy
) -> DriftSeverity:
    values = _collect(history.samples, "depth_confidence_mean")
    if len(values) < policy.min_samples_for_trend:
        return DriftSeverity.NONE
    older, recent = _half_split(values)
    old_mean = _mean(older)
    new_mean = _mean(recent)
    if old_mean is None or new_mean is None:
        return DriftSeverity.NONE
    # Drift = drop in depth confidence (older > recent means degradation).
    drop = max(0.0, old_mean - new_mean)
    return _severity_from_ascending(
        drop,
        policy.depth_confidence_shift_moderate,
        policy.depth_confidence_shift_high,
        policy.depth_confidence_shift_severe,
    )


def _monitor_fail_closed_spike(
    history: WatchdogHistory, policy: WatchdogPolicy
) -> DriftSeverity:
    flags = [
        bool(s.fail_closed)
        for s in history.samples
        if s.fail_closed is not None
    ]
    if len(flags) < policy.min_samples_for_trend:
        return DriftSeverity.NONE
    rate = sum(1 for f in flags if f) / len(flags)
    return _severity_from_ascending(
        rate,
        policy.fail_closed_rate_moderate,
        policy.fail_closed_rate_high,
        policy.fail_closed_rate_severe,
    )


def _monitor_residual_trend(
    history: WatchdogHistory,
    field_name: str,
    moderate: float,
    high: float,
    severe: float,
    min_samples: int,
) -> DriftSeverity:
    values = _collect(history.samples, field_name)
    if len(values) < min_samples:
        return DriftSeverity.NONE
    older, recent = _half_split(values)
    old_mean = _mean(older)
    new_mean = _mean(recent)
    if old_mean is None or new_mean is None:
        return DriftSeverity.NONE
    # Drift = growing residual (recent > older).
    delta = max(0.0, new_mean - old_mean)
    return _severity_from_ascending(delta, moderate, high, severe)


def evaluate_drift(
    history: WatchdogHistory, policy: WatchdogPolicy
) -> DriftMonitorBreakdown:
    """Run all five drift monitors over ``history``.

    Pure function: depends only on its inputs. Replay tooling and the
    runtime service call this exact code.
    """

    trimmed = history.trimmed(policy.window_size)
    return DriftMonitorBreakdown(
        calibration_delta=_monitor_calibration_delta(trimmed, policy),
        depth_confidence_shift=_monitor_depth_confidence_shift(
            trimmed, policy
        ),
        fail_closed_spike=_monitor_fail_closed_spike(trimmed, policy),
        hand_eye_residual_trend=_monitor_residual_trend(
            trimmed,
            "calibration_residual_mm",
            policy.hand_eye_residual_trend_moderate_mm,
            policy.hand_eye_residual_trend_high_mm,
            policy.hand_eye_residual_trend_severe_mm,
            policy.min_samples_for_trend,
        ),
        verification_residual_trend=_monitor_residual_trend(
            trimmed,
            "verification_residual_mm",
            policy.verification_residual_trend_moderate_mm,
            policy.verification_residual_trend_high_mm,
            policy.verification_residual_trend_severe_mm,
            policy.min_samples_for_trend,
        ),
    )


def evaluate_ood(
    history: WatchdogHistory, policy: WatchdogPolicy
) -> DriftSeverity:
    """OOD severity from the most recent sample's ``ood_score``.

    The severity ladder mirrors drift; ``ood_flagged`` on the final
    report is True iff severity >= MODERATE.
    """

    latest = history.latest
    if latest is None or latest.ood_score is None:
        return DriftSeverity.NONE
    return _severity_from_descending(
        latest.ood_score,
        policy.ood_score_moderate,
        policy.ood_score_high,
        policy.ood_score_severe,
    )


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WatchdogReport:
    """Typed aggregate output of one watchdog tick.

    JSON-safe via :meth:`to_dict`. The three telemetry fields that
    land on ``GraspAttemptRecord.extra`` are surfaced as:

    * ``drift_severity``: max of the five drift monitors.
    * ``ood_flagged``: True iff ``ood_severity >= MODERATE``.
    * ``degraded_mode_active``: True iff ``aggregate_severity >=
      MODERATE`` and the watchdog mode is not ``disabled``. In
      ``shadow`` mode this is reported truthfully (telemetry only,
      no behavioural effect).

    ``recommended_action`` translates the severity ladder into a
    runtime instruction: NONE / REOBSERVE / BLOCK_AUTO.

    ``enforced`` records whether the recommended action is enforced
    under the current mode + hardware + grasp mode. ``False`` in
    shadow mode and on simulated runs even when severity is
    HIGH/SEVERE.
    """

    mode: WatchdogMode
    drift_severity: DriftSeverity
    ood_severity: DriftSeverity
    ood_flagged: bool
    aggregate_severity: DriftSeverity
    degraded_mode_active: bool
    recommended_action: WatchdogAction
    enforced: bool
    monitors: DriftMonitorBreakdown
    grasp_mode: Optional[str] = None
    is_simulated: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "watchdog_mode": self.mode.value,
            "drift_severity": self.drift_severity.value,
            "ood_severity": self.ood_severity.value,
            "ood_flagged": self.ood_flagged,
            "aggregate_severity": self.aggregate_severity.value,
            "degraded_mode_active": self.degraded_mode_active,
            "recommended_action": self.recommended_action.value,
            "enforced": self.enforced,
            "drift_monitors": self.monitors.to_dict(),
            "grasp_mode": self.grasp_mode,
            "is_simulated": self.is_simulated,
        }


def _action_for_severity(
    severity: DriftSeverity, block_severities: Sequence[DriftSeverity]
) -> WatchdogAction:
    if severity in block_severities:
        return WatchdogAction.BLOCK_AUTO
    if severity is DriftSeverity.MODERATE:
        return WatchdogAction.REOBSERVE
    return WatchdogAction.NONE


def _should_enforce(
    *,
    mode: WatchdogMode,
    action: WatchdogAction,
    grasp_mode: Optional[str],
    is_simulated: Optional[bool],
    policy: WatchdogPolicy,
) -> bool:
    if mode in (WatchdogMode.DISABLED, WatchdogMode.SHADOW):
        return False
    if action is WatchdogAction.NONE:
        return False
    if action is WatchdogAction.BLOCK_AUTO:
        # Hardware lock: only enforce on real hardware, and only in
        # the configured block_modes.
        if is_simulated is True:
            return False
        if grasp_mode is None:
            # Caller did not pass the mode: refuse to enforce
            # silently. Watchdog stays in advisory state.
            return False
        if grasp_mode not in policy.block_modes:
            return False
        return True
    if action is WatchdogAction.REOBSERVE:
        # REOBSERVE is honoured in both canary and active modes; the
        # service decides whether budget+planner allow it. The
        # watchdog only reports that the action is eligible to be
        # enforced.
        return True
    return False


def evaluate_watchdog(
    history: WatchdogHistory,
    policy: WatchdogPolicy,
    *,
    grasp_mode: Optional[str] = None,
    is_simulated: Optional[bool] = None,
) -> WatchdogReport:
    """Run all monitors and produce a typed :class:`WatchdogReport`.

    When ``policy.mode`` is :attr:`WatchdogMode.DISABLED` this
    function still returns a typed report; the caller is responsible
    for skipping telemetry emission in disabled mode. Returning a
    report unconditionally keeps the call shape the same in every
    mode.
    """

    if policy.mode is WatchdogMode.DISABLED:
        return WatchdogReport(
            mode=policy.mode,
            drift_severity=DriftSeverity.NONE,
            ood_severity=DriftSeverity.NONE,
            ood_flagged=False,
            aggregate_severity=DriftSeverity.NONE,
            degraded_mode_active=False,
            recommended_action=WatchdogAction.NONE,
            enforced=False,
            monitors=DriftMonitorBreakdown(),
            grasp_mode=grasp_mode,
            is_simulated=is_simulated,
        )

    monitors = evaluate_drift(history, policy)
    drift_severity = monitors.aggregate()
    ood_severity = evaluate_ood(history, policy)
    aggregate = _max_severity(drift_severity, ood_severity)
    ood_flagged = _severity_rank(ood_severity) >= _severity_rank(
        DriftSeverity.MODERATE
    )
    degraded = _severity_rank(aggregate) >= _severity_rank(
        DriftSeverity.MODERATE
    )
    action = _action_for_severity(aggregate, policy.block_severities)
    enforced = _should_enforce(
        mode=policy.mode,
        action=action,
        grasp_mode=grasp_mode,
        is_simulated=is_simulated,
        policy=policy,
    )
    return WatchdogReport(
        mode=policy.mode,
        drift_severity=drift_severity,
        ood_severity=ood_severity,
        ood_flagged=ood_flagged,
        aggregate_severity=aggregate,
        degraded_mode_active=degraded,
        recommended_action=action,
        enforced=enforced,
        monitors=monitors,
        grasp_mode=grasp_mode,
        is_simulated=is_simulated,
    )
