"""The typed per-stage latency tracker.

:class:`LatencyTracker` is the small context-manager surface
:class:`AutonomousGraspService` uses to record the per-stage wall-clock spans that feed
the runtime SLO gate. ``decision`` is the ``DecisionEngine.decide`` call, ``ranking`` is
the grasp calculator with the scoring blend, and ``fusion`` is the multi-view fusion
update.

The tracker is deliberately passive: it never raises, never blocks the caller and never
performs IO. It uses :func:`time.monotonic_ns`, so an NTP adjustment does not affect it.
The snapshot dict it returns is JSON-safe and keyed by the canonical stage names that
match the extra-bag field names locked in the telemetry catalog,
``decision_latency_ms``, ``ranking_latency_ms`` and ``fusion_latency_ms``.

Five design notes:

1. It is pure Python, with no async and no threads.
2. A stage is named by a small enum, so a typo at a callsite is a name error rather than
   a silently missed metric.
3. A stage that is never entered reports ``None``, and the SLO gate skips a null rather
   than counting it as a pass.
4. Re-entering a stage replaces the previous span, so inside a retry loop the SLO
   contract measures the final, decision-relevant span.
5. ``snapshot()`` returns a fresh dict, so a caller mutates it without corrupting the
   tracker.

A stage whose span context is never entered is omitted from the snapshot entirely, which
leaves the pipelines unchanged where the performance config is disabled.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Optional


__all__ = (
    "LatencyStage",
    "LatencySpan",
    "LatencyTracker",
    "STAGE_FIELD_NAMES",
)


class LatencyStage(str, Enum):
    """The canonical per-attempt stages the SLO gate tracks."""

    DECISION = "decision"
    RANKING = "ranking"
    FUSION = "fusion"


#: Maps each stage onto its locked ``extra`` field name in the telemetry catalog. It is
#: a separate constant so the catalog contract stays the single source of truth, and a
#: caller looks a field name up through this map rather than concatenating strings.
STAGE_FIELD_NAMES: dict[LatencyStage, str] = {
    LatencyStage.DECISION: "decision_latency_ms",
    LatencyStage.RANKING: "ranking_latency_ms",
    LatencyStage.FUSION: "fusion_latency_ms",
}


@dataclass(frozen=True, slots=True)
class LatencySpan:
    """One closed span the tracker recorded."""

    stage: LatencyStage
    elapsed_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.stage, LatencyStage):
            raise TypeError(
                f"LatencySpan.stage must be a LatencyStage; got "
                f"{type(self.stage).__name__}"
            )
        if not isinstance(self.elapsed_ms, (int, float)):
            raise TypeError(
                "LatencySpan.elapsed_ms must be a finite number"
            )
        value = float(self.elapsed_ms)
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("LatencySpan.elapsed_ms must be finite")
        if value < 0.0:
            raise ValueError("LatencySpan.elapsed_ms must be non-negative")
        # Normalise an int to a float, so the downstream JSON encoding is stable.
        object.__setattr__(self, "elapsed_ms", value)


@dataclass(slots=True)
class LatencyTracker:
    """A mutable accumulator of per-stage spans for one attempt.

    It is cheap: create one per pick and discard it.
    """

    spans: dict[LatencyStage, LatencySpan] = field(default_factory=dict)

    @contextmanager
    def span(self, stage: LatencyStage) -> Iterator[None]:
        """Time a code block and record it under ``stage``.

        The span is always committed, including where the body raises, because the
        tracker measures real cost rather than successful paths alone. The exception is
        re-raised unchanged.
        """

        if not isinstance(stage, LatencyStage):
            raise TypeError(
                "LatencyTracker.span requires a LatencyStage; "
                f"got {type(stage).__name__}"
            )
        start_ns = time.monotonic_ns()
        try:
            yield
        finally:
            elapsed_ns = time.monotonic_ns() - start_ns
            elapsed_ms = elapsed_ns / 1_000_000.0
            # Clamp to non-negative, against a monotonic_ns quirk on a very fast span:
            # some platforms report 0.
            if elapsed_ms < 0.0:
                elapsed_ms = 0.0
            self.spans[stage] = LatencySpan(
                stage=stage, elapsed_ms=elapsed_ms
            )

    def record(self, stage: LatencyStage, elapsed_ms: float) -> None:
        """Record a span the caller computed.

        It covers a stage timed by an upstream component, such as a perception pipeline
        that already measures itself, where the service only threads the value into
        telemetry.
        """

        self.spans[stage] = LatencySpan(stage=stage, elapsed_ms=elapsed_ms)

    def get(self, stage: LatencyStage) -> Optional[float]:
        """The recorded elapsed milliseconds for ``stage``."""

        span = self.spans.get(stage)
        return None if span is None else span.elapsed_ms

    def snapshot(self) -> dict[str, float]:
        """Return a fresh ``{field_name: elapsed_ms}`` dict.

        A stage that was never entered is omitted entirely, so the caller merges the
        snapshot into the ``extra`` bag of a record without introducing a spurious
        ``0.0``. A stage that did not run reports as absent and never as a misleading
        zero.
        """

        return {
            STAGE_FIELD_NAMES[stage]: span.elapsed_ms
            for stage, span in self.spans.items()
        }
