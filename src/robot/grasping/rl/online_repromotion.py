"""Re-promotion / snapshot cadence for the Tier-0 online estimator.

A promoted policy's ``pass`` verdict is earned against the frozen artifact. Tier-0 online updates
accumulate deltas onto that artifact (see :mod:`online_state`), so the effective policy drifts away
from the thing the promotion gate actually evaluated, and the ``pass`` verdict silently goes stale.

This module is the cadence that keeps the verdict honest. It is a pure function of update counts:

* snapshot cadence: every ``snapshot_every_n`` online updates an immutable :class:`OnlineState`
  snapshot should be taken (provenance + a rollback point).
* staleness budget: once ``stale_after_n`` updates have accumulated since the last promotion
  baseline, the promoted verdict is stale: the effective policy must be frozen as a fresh immutable
  artifact and re-run through the promotion gate before it could ever leave shadow.

The re-promotion the cadence asks for cannot complete: freezing produces a promotable
(no-``online_state``-stamp) candidate, but the promotion gate needs real evaluation episodes and
honestly abstains without them (same wall as the committed policies). So this module flags staleness
and refuses to call a stale verdict fresh; it never fabricates a pass. ``still_shadow`` stays ``True``
on every decision: the cadence is a freshness check, not an activation path.

Pure-stdlib leaf: integer arithmetic + frozen dataclasses only, no numpy/torch, no policy imports.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Cadence status codes (typed, machine-parseable).
REPROMOTION_STATUS_CURRENT: str = "current"
REPROMOTION_STATUS_SNAPSHOT_DUE: str = "snapshot_due"
REPROMOTION_STATUS_STALE: str = "repromotion_required"
REPROMOTION_STATUSES: tuple[str, ...] = (
    REPROMOTION_STATUS_CURRENT,
    REPROMOTION_STATUS_SNAPSHOT_DUE,
    REPROMOTION_STATUS_STALE,
)


class RepromotionCadenceError(ValueError):
    """Raised on an invalid cadence configuration."""


@dataclass(frozen=True)
class RepromotionCadence:
    """Update-count thresholds that keep a promoted verdict fresh (all conservative defaults).

    ``snapshot_every_n`` must be ``<= stale_after_n`` so a provenance snapshot is always recommended
    before the verdict is declared stale: the snapshot comes first, the re-promotion only if drift
    continues."""

    snapshot_every_n: int = 100
    stale_after_n: int = 250

    def __post_init__(self) -> None:
        if self.snapshot_every_n <= 0:
            raise RepromotionCadenceError(
                f"snapshot_every_n must be > 0; got {self.snapshot_every_n!r}"
            )
        if self.stale_after_n <= 0:
            raise RepromotionCadenceError(
                f"stale_after_n must be > 0; got {self.stale_after_n!r}"
            )
        if self.snapshot_every_n > self.stale_after_n:
            raise RepromotionCadenceError(
                "snapshot_every_n must be <= stale_after_n "
                f"(snapshot before stale); got {self.snapshot_every_n} > {self.stale_after_n}"
            )


@dataclass(frozen=True)
class RepromotionDecision:
    """One cadence decision for one online family.

    ``still_shadow`` is always ``True``: a stale verdict never activates the policy; it only asks for
    a fresh promotion, which abstains without real evaluation data."""

    family: str
    n_updates: int
    updates_since_promotion: int
    updates_since_snapshot: int
    status: str
    snapshot_due: bool
    repromotion_required: bool
    reason: str
    still_shadow: bool = True


def evaluate_repromotion_cadence(
    *,
    family: str,
    n_updates: int,
    last_promoted_n_updates: int = 0,
    last_snapshot_n_updates: int = 0,
    cadence: RepromotionCadence | None = None,
) -> RepromotionDecision:
    """Decide whether a snapshot and/or a re-promotion is due for one online family.

    ``n_updates`` is the accumulator's current update count; ``last_promoted_n_updates`` /
    ``last_snapshot_n_updates`` are the counts at the last promotion baseline / last snapshot (0 for a
    fresh, never-promoted, never-snapshotted state). Staleness dominates: once the drift budget is
    exceeded the status is ``repromotion_required`` regardless of the snapshot cadence."""

    cad = cadence or RepromotionCadence()
    if n_updates < 0:
        raise RepromotionCadenceError(f"n_updates must be >= 0; got {n_updates!r}")
    since_promotion = max(0, n_updates - int(last_promoted_n_updates))
    since_snapshot = max(0, n_updates - int(last_snapshot_n_updates))

    snapshot_due = since_snapshot >= cad.snapshot_every_n
    repromotion_required = since_promotion >= cad.stale_after_n

    if repromotion_required:
        status = REPROMOTION_STATUS_STALE
        reason = (
            f"{since_promotion} online updates since the promotion baseline "
            f">= stale_after_n={cad.stale_after_n}: the promoted verdict is stale; freeze the "
            "effective policy and re-promote on real evaluation episodes before it may leave shadow"
        )
    elif snapshot_due:
        status = REPROMOTION_STATUS_SNAPSHOT_DUE
        reason = (
            f"{since_snapshot} online updates since the last snapshot "
            f">= snapshot_every_n={cad.snapshot_every_n}: take an immutable snapshot for provenance"
        )
    else:
        status = REPROMOTION_STATUS_CURRENT
        reason = (
            f"{since_promotion} updates since promotion (< {cad.stale_after_n}) and "
            f"{since_snapshot} since snapshot (< {cad.snapshot_every_n}): verdict still fresh"
        )

    return RepromotionDecision(
        family=str(family),
        n_updates=int(n_updates),
        updates_since_promotion=since_promotion,
        updates_since_snapshot=since_snapshot,
        status=status,
        snapshot_due=snapshot_due,
        repromotion_required=repromotion_required,
        reason=reason,
    )


__all__ = (
    "REPROMOTION_STATUS_CURRENT",
    "REPROMOTION_STATUS_SNAPSHOT_DUE",
    "REPROMOTION_STATUS_STALE",
    "REPROMOTION_STATUSES",
    "RepromotionCadence",
    "RepromotionCadenceError",
    "RepromotionDecision",
    "evaluate_repromotion_cadence",
)
