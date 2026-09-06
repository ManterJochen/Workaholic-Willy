"""The router's pure telemetry/extras projection helpers: build the ``rl_*`` telemetry-extras dicts + the per-candidate breakdown from the ShadowRouter carriers.

The carriers are TYPE_CHECKING-only here (fields are read off the arguments), so there is no runtime import of
router and no circular import in any order. ``BREAKDOWN_TOP_N`` + ``KENDALL_TAU_BOUNDS`` are re-exported from
router.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional

if TYPE_CHECKING:
    from .router import (
        PerceptionShadowTelemetry,
        RankingShadowTelemetry,
        RecoveryShadowTelemetry,
        SequencingShadowTelemetry,
        ShadowRouterTelemetry,
    )


#: Breakdown emit cap. The top-N proposals (by baseline rank, not by
#: policy rank; the deterministic baseline is the cheap, stable anchor)
#: get full breakdown rows; the remainder is compacted into a single
#: aggregate row keyed by ``"tail_count"`` / ``"tail_mean_score"``.
BREAKDOWN_TOP_N: int = 10

#: Lower / upper bounds on Kendall's tau. Used to clamp the computed
#: value defensively before emit (the telemetry catalog only verifies
#: type, not range).
KENDALL_TAU_BOUNDS: tuple[float, float] = (-1.0, 1.0)


def _clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def build_candidate_breakdown(
    *,
    telemetry: ShadowRouterTelemetry,
    baseline_ranks: Mapping[str, int] | None = None,
    top_n: int = BREAKDOWN_TOP_N,
) -> list[dict[str, object]]:
    """Build the per-candidate breakdown rows.

    The ``top_n`` proposals with the best baseline rank get a full row each; a candidate
    with no baseline rank falls back to its policy rank. The remainder is compacted into
    one tail summary row, so the JSONL payload stays bounded on dense scenes.
    """

    proposals = telemetry.selection.proposals
    if not proposals:
        return []

    def _baseline_rank(cid: str) -> int:
        if baseline_ranks is not None and cid in baseline_ranks:
            return int(baseline_ranks[cid])
        # No baseline rank: use policy rank as a stable fallback.
        return next(
            (p.rank for p in proposals if p.candidate_id == cid),
            10_000,
        )

    by_baseline = sorted(
        proposals, key=lambda p: (_baseline_rank(p.candidate_id), p.rank)
    )
    head = by_baseline[:top_n]
    tail = by_baseline[top_n:]

    rows: list[dict[str, object]] = []
    for p in head:
        rows.append(
            {
                "candidate_id": p.candidate_id,
                "baseline_rank": _baseline_rank(p.candidate_id),
                "shadow_rank": int(p.rank),
                "shadow_score": float(p.score),
                "pruned": bool(p.pruned),
            }
        )
    if tail:
        tail_count = len(tail)
        tail_mean_score = sum(float(p.score) for p in tail) / float(tail_count)
        rows.append(
            {
                "tail_count": int(tail_count),
                "tail_mean_score": float(tail_mean_score),
            }
        )
    return rows


def emit_candidate_shadow_extras(
    *,
    telemetry: ShadowRouterTelemetry,
    baseline_action: str,
    deterministic_top1_id: Optional[str],
    baseline_ranks: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Build the flat RL-required extras dict for a shadow attempt: every required ``RL_REQUIRED_TELEMETRY_FIELDS`` entry is populated non-null so the audit conditional gate is satisfied."""

    top1 = telemetry.selection.top1_id
    rl_action_proposed = "accept" if top1 is not None else "skip"
    # Shadow never applies, so the applied action is whatever the
    # deterministic baseline picked. Both fields carry the kind of
    # action, not the candidate id.
    rl_action_applied = baseline_action
    rl_override = rl_action_proposed != baseline_action

    reason_features: dict[str, float] = {
        "top1_score": float(
            telemetry.selection.proposals[0].score
            if telemetry.selection.proposals
            else 0.0
        ),
        "mask_total": float(telemetry.mask_summary.get("total_masked", 0)),
        "pruned_count": float(telemetry.agreement.pruned_count),
        "kendall_tau": float(telemetry.agreement.kendall_tau),
    }
    rl_confidence = _clamp(reason_features["top1_score"], 0.0, 1.0)

    breakdown = build_candidate_breakdown(
        telemetry=telemetry,
        baseline_ranks=baseline_ranks,
    )

    return {
        # RL-required (conditional gate consumes these).
        "rl_mode": "rl_shadow",
        "rl_policy_id": telemetry.policy_id,
        "rl_artifact_version": f"v{telemetry.policy_version}",
        "rl_action_proposed": rl_action_proposed,
        "rl_action_applied": rl_action_applied,
        "rl_action_blocked_by_mask": _top1_was_masked(
            telemetry=telemetry,
            deterministic_top1_id=deterministic_top1_id,
        ),
        "rl_reason_features": reason_features,
        "rl_confidence": rl_confidence,
        "rl_baseline_action": baseline_action,
        "rl_override": bool(rl_override),
        "rl_fallback_triggered": bool(telemetry.fallback_triggered),
        "rl_router_path": telemetry.router_path,
        "rl_fallback_reason_code": telemetry.fallback_reason_code,
        # Additive (type-only catalog entries).
        "rl_candidate_breakdown": breakdown,
        "rl_candidate_agreement_top1": bool(telemetry.agreement.top1_agree),
        "rl_candidate_agreement_kendall_tau": float(
            telemetry.agreement.kendall_tau
        ),
        "rl_candidate_mask_total": int(
            telemetry.mask_summary.get("total_masked", 0)
        ),
        "rl_candidate_pruned_count": int(telemetry.agreement.pruned_count),
    }


def _top1_was_masked(
    *,
    telemetry: ShadowRouterTelemetry,
    deterministic_top1_id: Optional[str],
) -> bool:
    """``True`` iff the deterministic top-1 candidate was masked: only possible when the mask is fed advisory (non-binding) flags; when the mask and the deterministic stack share a source this is ``False`` in steady state."""

    if deterministic_top1_id is None:
        return False
    total_masked = int(telemetry.mask_summary.get("total_masked", 0))
    if total_masked == 0:
        return False
    # The mask is opaque at this point (only the summary is on the
    # telemetry carrier), so the result is ``False`` unless the mask
    # was scene-wide (degraded_mode_active), in which case every
    # candidate including the deterministic top-1 was masked.
    return bool(telemetry.mask_summary.get("degraded_mode", 0) > 0)


def emit_ranking_extras(
    *,
    telemetry: "RankingShadowTelemetry",
) -> dict[str, object]:
    """Build the flat RL ranker telemetry extras dict: distinct keys from the candidate extras so consumers can grep for either independently."""

    return {
        "rl_ranking_policy_id": telemetry.policy_id,
        "rl_ranking_artifact_version": f"v{telemetry.policy_version}",
        "rl_ranking_regret_top1": bool(telemetry.regret_top1),
        "rl_ranking_kendall_tau": _clamp(
            float(telemetry.kendall_tau), *KENDALL_TAU_BOUNDS
        ),
    }


def emit_sequencing_extras(
    *,
    telemetry: "SequencingShadowTelemetry",
) -> dict[str, object]:
    """Build the flat RL sequencing telemetry extras dict: summarises the last failure attempt; on an empty decision tuple the ``*_action_*`` keys carry ``"none"`` and the agreement flag is ``False``."""

    policy_id = telemetry.policy_id
    policy_version = int(telemetry.policy_version)
    decisions = telemetry.decisions
    if decisions:
        last = decisions[-1]
        proposed = last.gated_action
        baseline = last.baseline_action
        agree = bool(last.agree_with_baseline)
    else:
        proposed = "none"
        baseline = "none"
        agree = False
    return {
        "rl_sequencing_policy_id": policy_id,
        "rl_sequencing_artifact_version": f"v{policy_version}",
        "rl_sequencing_action_proposed": proposed,
        "rl_sequencing_action_baseline": baseline,
        "rl_sequencing_action_agree": agree,
    }


def emit_perception_extras(
    *,
    telemetry: "PerceptionShadowTelemetry",
) -> dict[str, object]:
    """Build the perception-budget shadow telemetry summary: a flat ``{field_name: value}`` dict for internal router-side / shadow-artifact logging only; it is not wired into the replay-record ``extra.*`` schema or the telemetry catalog (the replay JSONL contract stays byte-frozen)."""

    selection = telemetry.selection
    return {
        "rl_perception_policy_id": telemetry.policy_id,
        "rl_perception_artifact_version": f"v{telemetry.policy_version}",
        "rl_perception_action_proposed": selection.action,
        "rl_perception_action_baseline": telemetry.baseline_action,
        "rl_perception_action_agree": bool(telemetry.agree_with_baseline),
        "rl_perception_ucb_stop": float(selection.ucb_stop),
        "rl_perception_ucb_continue": float(selection.ucb_continue),
        "rl_perception_support_stop": int(selection.support_stop),
        "rl_perception_support_continue": int(selection.support_continue),
        "rl_perception_used_fallback": bool(selection.used_fallback),
        "rl_perception_state_key": telemetry.state_key.as_string_key(),
        "rl_perception_fallback_triggered": bool(
            telemetry.fallback_triggered
        ),
        "rl_perception_fallback_reason_code": telemetry.fallback_reason_code,
    }


def emit_recovery_extras(
    *,
    telemetry: "RecoveryShadowTelemetry",
) -> dict[str, object]:
    """Build the recovery shadow telemetry summary: a flat ``{field_name: value}`` dict for internal router-side / shadow-artifact logging only; it is not wired into the replay-record ``extra.*`` schema or the telemetry catalog (the replay JSONL contract stays byte-frozen)."""

    selection = telemetry.selection
    return {
        "rl_recovery_policy_id": telemetry.policy_id,
        "rl_recovery_artifact_version": f"v{telemetry.policy_version}",
        "rl_recovery_action_proposed": selection.action,
        "rl_recovery_action_gated": telemetry.gated_action,
        "rl_recovery_action_baseline": telemetry.baseline_action,
        "rl_recovery_action_agree": bool(telemetry.agree_with_baseline),
        "rl_recovery_ranked_actions": list(selection.ranked_actions),
        "rl_recovery_used_fallback": bool(selection.used_fallback),
        "rl_recovery_state_key": telemetry.state_key.as_string_key(),
        "rl_recovery_attempt_index": int(telemetry.attempt_index),
        "rl_recovery_max_attempts": int(telemetry.max_recovery_attempts),
        "rl_recovery_gate_clipped": bool(telemetry.gate_clipped),
        "rl_recovery_gate_clip_reason": telemetry.gate_clip_reason,
        "rl_recovery_fallback_triggered": bool(telemetry.fallback_triggered),
        "rl_recovery_fallback_reason_code": telemetry.fallback_reason_code,
    }
