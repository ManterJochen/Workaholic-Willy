"""Honest per-candidate action-mask evaluator (execution tier).

The RL action mask (``grasping.rl.action_mask``) is a pure-stdlib function over an
:class:`ActionMaskContext` of rejection-id sets: it never invents a reason, it only reflects what
the deterministic stack already decided. This module produces that context. It lives in the
execution tier, where the heavy deterministic services (SafetyPreflight, the IK solver) are wired,
so ``rl/`` stays numpy-free, and it runs the real per-candidate checks to fill the sets.

The evaluator composes per-channel predicates, each a ``Callable[[MaskCandidate], bool]`` reading
"this channel rejects this candidate". Only the channels for which a predicate is supplied are
populated; an unsupplied channel stays empty, so the mask reflects the checks that actually ran and
nothing more. Ready-made adapters turn a real ``SafetyPreflight`` and IK solver into the safety and
feasibility predicates.

Per-candidate because the open-loop deterministic path preflights only the executed candidate, so
filling the mask honestly means re-running the deterministic checks for every shadow candidate. The
strongest check on that is the ``executed_candidate_masked`` metric: the deterministically-executed
candidate passed the stack, so the evaluator must never mask it, and a nonzero rate means the
evaluator's per-candidate verdict diverged from the pick path. See
:func:`executed_candidate_masked`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from src.robot.grasping.rl.action_mask import (
    ActionMaskContext,
    apply_action_mask,
)


@dataclass(frozen=True)
class MaskCandidate:
    """One shadow candidate to evaluate: a stable id + an opaque payload the predicates read
    (typically the grasp pose / a prebuilt SafetyContext)."""

    candidate_id: str
    payload: Any = None


Predicate = Callable[[MaskCandidate], bool]


def build_action_mask_context(
    candidates: Sequence[MaskCandidate],
    *,
    safety_reject: Predicate | None = None,
    feasibility_reject: Predicate | None = None,
    corridor_reject: Predicate | None = None,
    uncertainty_reject: Predicate | None = None,
    drift_ood_reject: Predicate | None = None,
    degraded_mode_active: bool = False,
) -> ActionMaskContext:
    """Run each supplied per-channel predicate over the candidate set and assemble the context.

    A predicate that raises is treated as fail-closed: the candidate is masked on that channel,
    because an errored deterministic check must never silently let a candidate through."""

    def _ids(pred: Predicate | None) -> frozenset[str]:
        if pred is None:
            return frozenset()
        rejected: set[str] = set()
        for c in candidates:
            try:
                hit = bool(pred(c))
            except Exception:  # noqa: BLE001 (a failed check is a rejection, never a pass)
                hit = True
            if hit:
                rejected.add(c.candidate_id)
        return frozenset(rejected)

    return ActionMaskContext(
        safety_rejected_ids=_ids(safety_reject),
        feasibility_failed_ids=_ids(feasibility_reject),
        corridor_failed_ids=_ids(corridor_reject),
        uncertainty_failed_ids=_ids(uncertainty_reject),
        drift_ood_failed_ids=_ids(drift_ood_reject),
        degraded_mode_active=degraded_mode_active,
    )


def safety_reject_from_preflight(
    preflight: Any, context_of: Callable[[MaskCandidate], Any]
) -> Predicate:
    """Adapt a real ``SafetyPreflight`` into a safety-channel predicate.

    ``context_of`` builds the per-candidate ``SafetyContext`` (target pose etc.). The candidate is
    safety-rejected iff ``preflight.evaluate(ctx).rejected``."""

    def _pred(c: MaskCandidate) -> bool:
        return bool(preflight.evaluate(context_of(c)).rejected)

    return _pred


def feasibility_reject_from_ik(ik_solve: Callable[[Any], Any]) -> Predicate:
    """Adapt an IK solver into a feasibility-channel predicate: no solution means feasibility-failed.

    ``ik_solve(payload)`` returns the solution (any truthy value) or ``None``/raises on no solution."""

    def _pred(c: MaskCandidate) -> bool:
        return ik_solve(c.payload) is None

    return _pred


def executed_candidate_masked(
    context: ActionMaskContext, executed_candidate_id: str | None
) -> bool:
    """Consistency metric: was the deterministically-executed candidate masked?

    Must be ``False``: the executed candidate passed the deterministic stack, so a faithful mask
    keeps it. A ``True`` (a nonzero rate over a soak) means the evaluator's per-candidate verdict
    diverged from the pick path, and this is the on-box gate for un-deferring Tier-1."""

    if executed_candidate_id is None:
        return False
    rows = apply_action_mask((executed_candidate_id,), context)
    return bool(rows[0].masked_by)


__all__ = (
    "MaskCandidate",
    "Predicate",
    "build_action_mask_context",
    "executed_candidate_masked",
    "feasibility_reject_from_ik",
    "safety_reject_from_preflight",
)
