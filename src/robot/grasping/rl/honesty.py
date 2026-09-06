"""Single source of truth for RL-artifact honesty + provenance stamps.

Every RL artifact (dataset manifest; candidate-selection / ranking / sequencing / perception-budget /
recovery policy artifacts; OPE report; promotion report;
paired-soak / rollback-drill reports) should carry consistent stamps along three dimensions so a reader can never
mistake a synthetic/off-policy/degenerate artifact for a real-hardware-validated one:

  (1) ``reward_model`` + ``reward_interpretation``: where the reward/score originates (synthetic vs
      real-records-derived) and what it is and is not (mirrors :data:`paired_soak.SOAK_REWARD_MODEL`).
  (2) ``dataset_provenance``: sim-vs-real origin, dataset id/hash, leakage-proven, fit_for/not_fit_for
      (mirrors the ``replay/soak.py`` provenance banner shape).
  (3) gate/verdict interpretation: what a pass/abstain/verdict proves and does not prove (free-form per
      producer, since the honest caveat is genuinely artifact-specific) + a conditional
      :func:`single_action_degeneracy_note` so a vacuous policy cannot pass for a trained one.

The reward-model strings + dataset-provenance key schema live here (shared, validated); the interpretation
sentences stay inline in each producer. These stamps live at the artifact root (never inside
``GraspAttemptRecord.extra`` and never in ``replay/telemetry_catalog.py``) so the locked soak baseline
is untouched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# reward_model enum
# "logged_*": the reward is read from the records' logged outcome/metric fields, not from a synthetic
# outcome model like the paired soak's synthetic_hardcoded. This describes the reward method only;
# whether the records themselves are real-hardware or synthetic-canonical is a separate dimension
# carried in dataset_provenance. The method names carry no "real": the baselines train on synthetic
# canonical packs, so "real" in a method name would mislead.
REWARD_LOGGED_LABELS: str = "logged_success_labels"  # candidate-selection / ranking logistic over logged success labels
REWARD_OFFLINE_EMPIRICAL_FREQUENCY: str = "offline_empirical_frequency"  # sequencing lookup-table frequency
REWARD_LOGGED_COEFFICIENTS: str = "logged_metric_coefficients"  # perception-budget LinUCB over OPE V1 coeffs on logged metrics
REWARD_LOGGED_OUTCOMES: str = "logged_recovery_outcomes"  # recovery-policy binary outcome
REWARD_LOGGED_RECORDS_DERIVED: str = "logged_records_derived"  # OPE: locked V1 coeffs applied to record.extra
# Synthetic / authored rewards (not observed from a real or high-fidelity run):
REWARD_SYNTHETIC_HARDCODED: str = "synthetic_hardcoded"  # paired soak / rollback drill (matches paired_soak.py)
REWARD_SYNTHETIC_CANONICAL: str = "synthetic_canonical_soak_deterministic"  # promotion gate over canonical packs
# Not applicable (the artifact carries no reward; raw records only):
REWARD_NOT_APPLICABLE_RAW: str = "not_applicable_raw_records_only"  # dataset manifest

REWARD_MODELS: frozenset[str] = frozenset(
    {
        REWARD_LOGGED_LABELS,
        REWARD_OFFLINE_EMPIRICAL_FREQUENCY,
        REWARD_LOGGED_COEFFICIENTS,
        REWARD_LOGGED_OUTCOMES,
        REWARD_LOGGED_RECORDS_DERIVED,
        REWARD_SYNTHETIC_HARDCODED,
        REWARD_SYNTHETIC_CANONICAL,
        REWARD_NOT_APPLICABLE_RAW,
    }
)

# dataset_provenance: input_source enum
INPUT_REAL_HARDWARE: str = "real_hardware"
INPUT_SIM: str = "sim"
INPUT_SYNTHETIC_DETERMINISTIC: str = "synthetic_deterministic"
INPUT_CANONICAL_BOOTSTRAP: str = "canonical_bootstrap_replay_packs"

INPUT_SOURCES: frozenset[str] = frozenset(
    {INPUT_REAL_HARDWARE, INPUT_SIM, INPUT_SYNTHETIC_DETERMINISTIC, INPUT_CANONICAL_BOOTSTRAP}
)


def make_dataset_provenance(
    *,
    input_source: str,
    fit_for: str,
    not_fit_for: Sequence[str],
    dataset_id: str | None = None,
    dataset_hash: str | None = None,
    source_pack_paths: Sequence[str] | None = None,
    leakage_proven: bool | None = None,
) -> dict[str, Any]:
    """Build the canonical ``dataset_provenance`` block (deterministic, json-safe)."""

    # leakage_proven is forwarded from the dataset manifest's leakage audit: None where the audit
    # never ran, False where it ran but did not prove separation. It is the cross-artifact spine
    # that lets a downstream policy artifact state whether its training data was leakage-proven.
    if input_source not in INPUT_SOURCES:
        raise ValueError(f"unknown input_source {input_source!r}; expected one of {sorted(INPUT_SOURCES)}")
    return {
        "input_source": input_source,
        "dataset_id": dataset_id,
        "dataset_hash": dataset_hash,
        "source_pack_paths": list(source_pack_paths) if source_pack_paths is not None else None,
        "leakage_proven": leakage_proven,
        "fit_for": fit_for,
        "not_fit_for": list(not_fit_for),
    }


def single_action_degeneracy_note(
    num_per_action: Mapping[str, int],
    *,
    fallback_rate: float | None = None,
    dominance_threshold: float = 0.95,
) -> str | None:
    """Return an honesty note iff the policy collapsed to ~one action, else ``None``."""

    # A degenerate recovery policy (one action carries every training tuple, the rest have zero
    # support and fall back to the hand-authored table) still writes an artifact body that looks
    # like a normal trained LinUCB policy; this note makes the degeneracy visible so a reader
    # cannot mistake a fallback-table wrapper for a meaningfully trained policy.
    counts = {a: int(c) for a, c in num_per_action.items()}
    total = sum(counts.values())
    fb = "" if fallback_rate is None else f" Effective fallback_rate={fallback_rate:.3f}."
    if total == 0:
        return (
            "DEGENERATE: zero training tuples; this artifact is the hand-authored fallback table only, "
            "NOT a trained policy." + fb
        )
    nonzero = {a: c for a, c in counts.items() if c > 0}
    top_action, top_count = max(counts.items(), key=lambda kv: kv[1])
    share = top_count / total
    if len(nonzero) == 1 or share >= dominance_threshold:
        return (
            f"DEGENERATE: a single action dominates training ({top_action}: {top_count}/{total} tuples, "
            f"{share:.0%}); the remaining actions have little/no support and fall back to the hand-authored "
            f"table. This artifact behaves as a fallback-table wrapper, NOT a meaningfully trained policy." + fb
        )
    return None


# per-artifact stamp builders (the single review surface for the honest wording)
# Shared "not_fit_for" for every RL policy artifact: none is hardware-validated, and none may drive
# active runtime control without first passing the promotion gate.
_POLICY_NOT_FIT_FOR: tuple[str, ...] = (
    "real-hardware lift (no captured real-robot replay was used)",
    "active runtime control without a passing offline promotion gate",
)

CANDIDATE_REWARD_INTERPRETATION: str = (
    "off-policy L2 logistic over LOGGED grasp-success labels; it reorders/prunes candidates. The committed "
    "baseline trains on the synthetic canonical v1_bootstrap dataset (see dataset_provenance + the dataset "
    "manifest's leakage audit). Off-policy / shadow-only; NOT validated for real live-loop lift."
)
RANKING_REWARD_INTERPRETATION: str = (
    "off-policy pairwise L2 logistic over LOGGED success labels; it ranks candidates within a group. Trains "
    "on the synthetic canonical v1_bootstrap dataset. Off-policy / shadow-only; NOT validated for real "
    "live-loop ranking lift."
)
SEQUENCING_REWARD_INTERPRETATION: str = (
    "deterministic empirical action-FREQUENCY lookup over LOGGED sequencing outcomes; a cell's action is the "
    "most-frequent observed action, NOT a live success probability. Cells below min_support fall back to the "
    "hand-authored DEFAULT table (see fallback_honesty). Synthetic canonical data; off-policy; NOT real lift."
)
PERCEPTION_REWARD_INTERPRETATION: str = (
    "LinUCB over the OPE V1 reward coefficients applied to LOGGED perception/cycle-time signals; proposes "
    "STOP/CONTINUE per perception budget. Synthetic canonical data. Off-policy / offline estimate; NOT "
    "real-hardware lift."
)


def _policy_stamps(
    *,
    reward_model: str,
    reward_interpretation: str,
    input_source: str,
    dataset_id: str | None,
    dataset_hash: str | None,
    fit_for: str,
    degeneracy_note: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the standard 3-stamp honesty block for a policy artifact (+ optional notes)."""

    block: dict[str, Any] = {
        "reward_model": reward_model,
        "reward_interpretation": reward_interpretation,
        "dataset_provenance": make_dataset_provenance(
            input_source=input_source,
            dataset_id=dataset_id,
            dataset_hash=dataset_hash,
            leakage_proven=None,  # the dataset manifest is the leakage authority; policies do not re-assert it
            fit_for=fit_for,
            not_fit_for=_POLICY_NOT_FIT_FOR,
        ),
    }
    if extra:
        block.update(extra)
    if degeneracy_note is not None:
        block["degeneracy_note"] = degeneracy_note
    return block


def build_candidate_honesty(*, dataset_id: str | None, dataset_hash: str | None) -> dict[str, Any]:
    return _policy_stamps(
        reward_model=REWARD_LOGGED_LABELS,
        reward_interpretation=CANDIDATE_REWARD_INTERPRETATION,
        input_source=INPUT_CANONICAL_BOOTSTRAP,
        dataset_id=dataset_id,
        dataset_hash=dataset_hash,
        fit_for="candidate reordering/pruning in SHADOW (off-policy)",
    )


def build_ranking_honesty(
    *,
    dataset_id: str | None,
    dataset_hash: str | None,
    pairwise_accuracy: float | None = None,
    pairwise_accuracy_baseline: float | None = None,
) -> dict[str, Any]:
    # Surface a performance_note when a ranker does not beat its own pairwise baseline, so a reader
    # cannot mistake a weak ranker for a good one.
    extra: dict[str, Any] | None = None
    if (
        pairwise_accuracy is not None
        and pairwise_accuracy_baseline is not None
        and pairwise_accuracy < pairwise_accuracy_baseline
    ):
        extra = {
            "performance_note": (
                f"BELOW BASELINE: pairwise_accuracy={pairwise_accuracy:.4f} is under its own baseline "
                f"({pairwise_accuracy_baseline:.4f}); this committed ranker does NOT outrank the deterministic "
                f"baseline on the eval set and would NOT pass a promotion gate. It is a weak shadow artifact."
            )
        }
    return _policy_stamps(
        reward_model=REWARD_LOGGED_LABELS,
        reward_interpretation=RANKING_REWARD_INTERPRETATION,
        input_source=INPUT_CANONICAL_BOOTSTRAP,
        dataset_id=dataset_id,
        dataset_hash=dataset_hash,
        fit_for="candidate ranking in SHADOW (off-policy pairwise)",
        extra=extra,
    )


def build_sequencing_honesty(
    *,
    dataset_id: str | None,
    dataset_hash: str | None,
    num_cells_committed: int,
    num_cells_below_threshold: int,
) -> dict[str, Any]:
    fallback_honesty = {
        "num_cells_committed": int(num_cells_committed),
        "num_cells_below_threshold": int(num_cells_below_threshold),
        "fallback_table_source": "hand-authored DEFAULT_FALLBACK_TABLE",
        "meaning": (
            "cells below min_support_threshold use the hand-authored fallback, NOT learned data; "
            "the larger this share, the less the artifact learned."
        ),
    }
    degeneracy = None
    if int(num_cells_committed) == 0:
        degeneracy = (
            "DEGENERATE: zero cells met min_support; every decision uses the hand-authored fallback table, "
            "NOT learned data."
        )
    return _policy_stamps(
        reward_model=REWARD_OFFLINE_EMPIRICAL_FREQUENCY,
        reward_interpretation=SEQUENCING_REWARD_INTERPRETATION,
        input_source=INPUT_CANONICAL_BOOTSTRAP,
        dataset_id=dataset_id,
        dataset_hash=dataset_hash,
        fit_for="sequencing-action lookup in SHADOW (off-policy empirical frequency)",
        degeneracy_note=degeneracy,
        extra={"fallback_honesty": fallback_honesty},
    )


def build_perception_honesty(
    *,
    dataset_id: str | None,
    dataset_hash: str | None,
    num_records_kept: int,
    num_per_action: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    # Degeneracy precedence: zero in-scope records (nothing trained) first, else, when the per-action
    # counts are supplied, a single-action collapse (every tuple in one cell and one action, while the
    # artifact body still looks like a trained LinUCB policy). ``num_per_action`` is optional; a caller
    # that passes only ``num_records_kept`` gets the same behaviour.
    degeneracy = None
    if int(num_records_kept) == 0:
        degeneracy = (
            "DEGENERATE: zero in-scope records; the policy is the hand-authored fallback only (no perception-"
            "budget signal was present in the data), NOT a trained LinUCB policy."
        )
    elif num_per_action is not None:
        degeneracy = single_action_degeneracy_note(num_per_action)
    return _policy_stamps(
        reward_model=REWARD_LOGGED_COEFFICIENTS,
        reward_interpretation=PERCEPTION_REWARD_INTERPRETATION,
        input_source=INPUT_SYNTHETIC_DETERMINISTIC,
        dataset_id=dataset_id,
        dataset_hash=dataset_hash,
        fit_for="perception-budget STOP/CONTINUE in SHADOW (off-policy)",
        degeneracy_note=degeneracy,
    )


# report stamp builders (OPE estimate + promotion gate)
OPE_REWARD_INTERPRETATION: str = (
    "reward = the LOCKED OPE V1 coefficients applied to LOGGED record outcomes/metrics (real-records-derived, "
    "NOT a synthetic outcome model). The WIS/DM section values are OFF-POLICY ESTIMATES with documented bias "
    "(importance-weight clip + epsilon-smoothing); NOT a guaranteed or real live-loop lift; the candidate "
    "and ranking sections may be non-estimable."
)
PROMOTION_REWARD_INTERPRETATION: str = (
    "the WIS/DM lift is estimated over the SYNTHETIC canonical soak-generated packs (authored outcome "
    "distribution, see dataset_provenance), NOT real hardware. A verdict characterises the gate over "
    "synthetic data; it is NOT a real-hardware lift."
)
PROMOTION_VERDICT_MEANING: dict[str, str] = {
    "pass": (
        "the off-policy WIS lift over the synthetic packs cleared the thresholds (positive floor + CI lower "
        "bound + min effective sample size). NOT a real-hardware lift."
    ),
    "abstain": (
        "the policy could NOT be graded (too few effective samples, or a degenerate single-action policy); "
        "no improvement claim either way."
    ),
    "fail": "the off-policy WIS lift did not clear the thresholds (regression or below the positive floor).",
}


def build_ope_honesty(*, dataset_id: str | None, dataset_hash: str | None) -> dict[str, Any]:
    return {
        "reward_model": REWARD_LOGGED_RECORDS_DERIVED,
        "reward_interpretation": OPE_REWARD_INTERPRETATION,
        "dataset_provenance": make_dataset_provenance(
            input_source=INPUT_SYNTHETIC_DETERMINISTIC,
            dataset_id=dataset_id,
            dataset_hash=dataset_hash,
            leakage_proven=None,
            fit_for="off-policy candidate/ranking/sequencing shadow-telemetry evaluation",
            not_fit_for=["real-hardware lift measurement", "a promotion decision on its own"],
        ),
    }


def build_paired_soak_honesty() -> dict[str, Any]:
    """Dims (2)+(3) for the paired soak; reward_model/interpretation already live in paired_soak.py."""

    return {
        "dataset_provenance": make_dataset_provenance(
            input_source=INPUT_SYNTHETIC_DETERMINISTIC,
            fit_for="router/rollback MECHANICS characterization (paired RL-on/off under a synthetic outcome model)",
            not_fit_for=["real-hardware lift", "a promotion decision"],
        ),
        "gate_verdict_interpretation": {
            "pass_means": (
                "the RL arm improved cycle/retry with NO regression in pick_success/safety, over the SYNTHETIC "
                "outcome model; MECHANICS only, NOT a real-hardware lift."
            ),
            "warn_means": "no regression, but insufficient positive delta to claim improvement.",
            "fail_means": "the RL arm regressed pick_success / dead_loop / safety, or broke baseline parity.",
            "verdict_applies_to": "router_mechanics_only",
        },
    }


def build_rollback_drill_honesty(
    *, policy_artifact_path: str, promotion_report_path: str
) -> dict[str, Any]:
    return {
        "reward_model": REWARD_SYNTHETIC_HARDCODED,
        "reward_interpretation": (
            "the drills exercise the kill-switch / auto-fallback MECHANICS (operator engage, regret/override "
            "auto-fallback, policy-error fallback, post-fallback baseline parity). A PASS proves the SAFETY "
            "machinery works; NOT that the policy is good or that there is any real-hardware lift."
        ),
        "policy_provenance": {
            "policy_artifact_path": policy_artifact_path,
            "promotion_report_path": promotion_report_path,
        },
    }


def build_promotion_honesty(
    *,
    dataset_id: str | None,
    dataset_hash: str | None,
    verdict: str = "",
    reasons: Sequence[str] = (),
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "reward_model": REWARD_SYNTHETIC_CANONICAL,
        "reward_interpretation": PROMOTION_REWARD_INTERPRETATION,
        "dataset_provenance": make_dataset_provenance(
            input_source=INPUT_SYNTHETIC_DETERMINISTIC,
            dataset_id=dataset_id,
            dataset_hash=dataset_hash,
            leakage_proven=None,
            fit_for="offline promotion-gate decision over canonical packs",
            not_fit_for=["real-hardware lift", "a guarantee of live-loop improvement"],
        ),
        "verdict_meaning": dict(PROMOTION_VERDICT_MEANING),
    }
    # When the gate abstains on a degenerate policy, surface a degeneracy_note in the report; the
    # bare reasons=["degenerate_single_action_policy"] is easy to miss.
    if verdict == "abstain" and any("degenerate" in str(r) for r in reasons):
        block["degeneracy_note"] = (
            "DEGENERATE candidate policy: the gate ABSTAINED because the policy collapsed to a single action "
            "(reasons include degenerate_single_action_policy), so the off-policy WIS importance weights "
            "collapse and NO lift can be estimated. The policy is NOT promotable."
        )
    return block
