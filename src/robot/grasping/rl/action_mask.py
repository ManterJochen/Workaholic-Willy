"""Inviolable action mask for candidate-selection policies.

The action mask is the first filter applied before any RL/learned
policy sees a candidate. It consumes deterministic rejection channels
and produces, for every candidate, the tuple of channels that masked
it. A non-empty ``masked_by`` tuple means no policy may elevate this
candidate to top-1, ever, regardless of mode (``rl_shadow`` /
``rl_active`` / ``rl_experimental``).

This module has zero runtime side-effects: it transforms an immutable
candidate list + a typed :class:`ActionMaskContext` into a tuple of
:class:`MaskedCandidate` rows. It is the safety boundary the shadow
router relies on today, and the one any future active rollout
inherits unchanged.

Authority hierarchy (do not weaken):

1. ``robot.safety``                    gives ``"safety"``
2. feasibility deterministic gate      gives ``"feasibility"``
3. corridor / occlusion deterministic  gives ``"corridor"``
4. uncertainty fail-closed             gives ``"uncertainty"``
5. drift / OOD fail-closed             gives ``"drift_ood"``
6. degraded mode active                gives ``"degraded_mode"``

Order matters: ``masked_by`` is reported in this precedence so the
operator can attribute a mask to the highest-authority channel that
fired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


#: Frozen channel order. ``apply_action_mask`` walks the channels in
#: this order; ``masked_by`` preserves the order of channels that
#: actually fired. New channels may only be appended (additive contract).
MASK_CHANNELS: tuple[str, ...] = (
    "safety",
    "feasibility",
    "corridor",
    "uncertainty",
    "drift_ood",
    "degraded_mode",
)


@dataclass(frozen=True)
class ActionMaskContext:
    """Immutable per-attempt mask inputs; reflects only what the
    deterministic stack already decided (the mask never invents
    reasons). ``degraded_mode_active`` masks every candidate at the
    scene level via the ``"degraded_mode"`` channel."""

    safety_rejected_ids: frozenset[str] = frozenset()
    feasibility_failed_ids: frozenset[str] = frozenset()
    corridor_failed_ids: frozenset[str] = frozenset()
    uncertainty_failed_ids: frozenset[str] = frozenset()
    drift_ood_failed_ids: frozenset[str] = frozenset()
    degraded_mode_active: bool = False


@dataclass(frozen=True)
class MaskedCandidate:
    """One row of action-mask evaluation: ``masked_by`` lists the
    channels (in :data:`MASK_CHANNELS` order) that rejected this
    candidate; an empty tuple means it is eligible for elevation."""

    candidate_id: str
    masked_by: tuple[str, ...] = field(default_factory=tuple)

    @property
    def keep(self) -> bool:
        """``True`` iff no channel masked this candidate."""

        return len(self.masked_by) == 0


def apply_action_mask(
    candidate_ids: Sequence[str],
    context: ActionMaskContext,
) -> tuple[MaskedCandidate, ...]:
    """Evaluate the action mask over an ordered candidate list.

    Deterministic and order-preserving. Each ``masked_by`` lists
    channels in :data:`MASK_CHANNELS` precedence.
    """

    safety = context.safety_rejected_ids
    feas = context.feasibility_failed_ids
    corr = context.corridor_failed_ids
    unc = context.uncertainty_failed_ids
    drift = context.drift_ood_failed_ids
    degraded = context.degraded_mode_active

    rows: list[MaskedCandidate] = []
    for cid in candidate_ids:
        channels: list[str] = []
        if cid in safety:
            channels.append("safety")
        if cid in feas:
            channels.append("feasibility")
        if cid in corr:
            channels.append("corridor")
        if cid in unc:
            channels.append("uncertainty")
        if cid in drift:
            channels.append("drift_ood")
        if degraded:
            channels.append("degraded_mode")
        rows.append(MaskedCandidate(candidate_id=cid, masked_by=tuple(channels)))
    return tuple(rows)


def mask_summary(masked: Sequence[MaskedCandidate]) -> Mapping[str, int]:
    """Return per-channel mask counts + a ``total_masked`` aggregate."""

    counts: dict[str, int] = {ch: 0 for ch in MASK_CHANNELS}
    total_masked = 0
    for row in masked:
        if row.masked_by:
            total_masked += 1
        for ch in row.masked_by:
            counts[ch] += 1
    counts["total_masked"] = total_masked
    return counts


__all__ = (
    "MASK_CHANNELS",
    "ActionMaskContext",
    "MaskedCandidate",
    "apply_action_mask",
    "mask_summary",
)
