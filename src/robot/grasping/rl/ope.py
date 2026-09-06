"""Off-Policy Evaluation (OPE) harness for the candidate/ranking/
sequencing shadow policies.

This module is purely offline and has zero runtime touch points. It
consumes JSONL replay packs and a target policy artifact, and emits
a deterministic JSON document with three sections:

* ``candidate``: WIS-only, attempt-level rewards. A coverage warning
  is emitted because logs do not store the per-candidate-id of the
  candidate policy's choice.
* ``ranking``: WIS-only, attempt-level rewards. Same coverage
  warning.
* ``sequencing``: WIS + Direct-Method (DM) side-by-side using a
  tabular bandit Q (``gamma=0``) over the sequencing 4-tuple state
  key.

The ``v2_``, ``v3_`` and ``v4_`` prefixes belong to policy names and
policy-artifact filenames, not to these section keys.

Estimator notes:

* Behaviour policy = epsilon-smoothed deterministic, ``epsilon=0.05``.
  WIS is computable but a ``behaviour_policy_assumption`` metadata
  field is always emitted so the bias is visible in the artifact.
* Importance weights clipped at ``IMPORTANCE_WEIGHT_CLIP=100``.
* Confidence intervals: bootstrap (1000 resamples, default) and
  closed-form normal-approximation are both computed.
* Reward function = ``OPE_REWARD_COEFFICIENTS_V1``.
* Malformed input raises :class:`MalformedOPEInputError` (strict).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.robot.grasping.constants import (
    RL_OPE_LOG_FILE,
    create_grasping_logger,
)

from .dataset import OUTCOME_CLASS_SUCCESS, derive_outcome_class, load_jsonl
from .honesty import build_ope_honesty
from .sequencing_policy import (
    SEQUENCING_ACTIONS,
    SequencingRequest,
    SequencingStateKey,
    load_lookup_table_sequencing_policy,
)
from .train_sequencing import (
    _derive_action_from_successor,
    _extract_outcome_label,
    _record_state_key,
)
from .train_ranking import _group_key as _ranking_group_key

#: The estimate this module produces is what the promotion gate votes on, so the
#: inputs it was computed over (record count, dataset hash, seed) belong next to
#: it; a report alone cannot say which corpus it saw.
logger = create_grasping_logger("RLOPE", RL_OPE_LOG_FILE)


# ---------------------------------------------------------------------------
# Locked constants.
# ---------------------------------------------------------------------------


#: Schema version of the OPE report artifact. Bump on any
#: backward-incompatible change to the report shape.
OPE_REPORT_SCHEMA_VERSION: int = 2

#: Epsilon used to smooth the deterministic behaviour policy.
#: Surfaced in the report metadata.
BEHAVIOUR_POLICY_EPSILON: float = 0.05

#: Per-record importance-weight cap.
IMPORTANCE_WEIGHT_CLIP: float = 100.0

#: Bootstrap resample count.
BOOTSTRAP_N: int = 1000

#: Normal-approximation z-score for 95% CI.
NORMAL_CI_Z: float = 1.96

#: Locked reward coefficients.
#:
#: Locked means a process, not just a constant. Changing any coefficient here needs explicit
#: operator sign-off and a replay re-baseline: reports scored under different weights are not
#: comparable with each other, and both still look like reports.
OPE_REWARD_COEFFICIENTS_V1: Mapping[str, float] = {
    "success": 1.0,
    "failed_attempt": -0.5,
    "cycle_time_seconds": -0.3,
    "collision_event": -1.5,
    "dead_loop": -2.0,
    "perception_cost": -0.2,
}

#: Sequencing OPE uses gamma=0 because the policy makes one-shot
#: per-attempt decisions (no multi-step return). This collapses FQE
#: to a tabular bandit Q.
FQE_GAMMA: float = 0.0


# ---------------------------------------------------------------------------
# Exceptions.
# ---------------------------------------------------------------------------


class MalformedOPEInputError(ValueError):
    """Raised when a record lacks fields required for OPE."""


# ---------------------------------------------------------------------------
# Reward.
# ---------------------------------------------------------------------------


def compute_record_reward(
    record: Mapping[str, Any],
    *,
    coefficients: Mapping[str, float] = OPE_REWARD_COEFFICIENTS_V1,
) -> float:
    """Compute the per-attempt reward for ``record`` under V1 coefficients.

    Required fields: a derivable outcome class and ``extra.cycle_time_s``.
    Optional flags ``extra.collision_event`` / ``extra.dead_loop`` /
    ``extra.perception_cost`` modulate the reward when present.
    """

    if not isinstance(record, Mapping):
        raise MalformedOPEInputError(
            f"record must be a mapping, got {type(record).__name__}"
        )
    extra = record.get("extra")
    if not isinstance(extra, Mapping):
        raise MalformedOPEInputError(
            f"record {record.get('attempt_id')!r} missing extra block"
        )
    if "cycle_time_s" not in extra:
        raise MalformedOPEInputError(
            f"record {record.get('attempt_id')!r} missing extra.cycle_time_s"
        )

    success = derive_outcome_class(record) == OUTCOME_CLASS_SUCCESS
    r = coefficients["success"] if success else coefficients["failed_attempt"]
    r += coefficients["cycle_time_seconds"] * float(extra["cycle_time_s"])
    if bool(extra.get("collision_event", False)):
        r += coefficients["collision_event"]
    if bool(extra.get("dead_loop", False)):
        r += coefficients["dead_loop"]
    perception_cost = extra.get("perception_cost", 0.0)
    try:
        r += coefficients["perception_cost"] * float(perception_cost)
    except (TypeError, ValueError) as exc:  # pragma: no cover (defensive)
        raise MalformedOPEInputError(
            f"record {record.get('attempt_id')!r} has non-numeric "
            f"extra.perception_cost: {perception_cost!r}"
        ) from exc
    return float(r)


# ---------------------------------------------------------------------------
# Behaviour-policy probability under epsilon-smoothing.
# ---------------------------------------------------------------------------


def behaviour_action_probability(
    *,
    num_actions: int,
    chosen: bool,
    epsilon: float = BEHAVIOUR_POLICY_EPSILON,
) -> float:
    """Probability assigned by the epsilon-smoothed behaviour policy.

    The deterministic baseline selects one action; the smoothing
    assigns ``epsilon/K`` mass to every action (including the chosen
    one).
    """

    if num_actions < 1:
        raise MalformedOPEInputError("num_actions must be >= 1")
    if not 0.0 <= epsilon < 1.0:
        raise MalformedOPEInputError(
            f"epsilon must be in [0, 1), got {epsilon!r}"
        )
    base = epsilon / float(num_actions)
    if chosen:
        return (1.0 - epsilon) + base
    return base


# ---------------------------------------------------------------------------
# Confidence intervals.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceInterval:
    """A 95% CI plus the method that produced it."""

    method: str           # "bootstrap" | "normal_approx"
    lower: float
    upper: float
    n: int


def _bootstrap_mean_ci(
    samples: Sequence[float],
    *,
    n_resamples: int = BOOTSTRAP_N,
    rng_seed: int = 0,
) -> ConfidenceInterval:
    """95% bootstrap CI on the sample mean (percentile method)."""

    n = len(samples)
    if n == 0:
        return ConfidenceInterval("bootstrap", 0.0, 0.0, 0)
    rng = random.Random(rng_seed)
    means: list[float] = []
    for _ in range(n_resamples):
        s = 0.0
        for _i in range(n):
            s += samples[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo_idx = max(0, int(round(0.025 * (n_resamples - 1))))
    hi_idx = min(n_resamples - 1, int(round(0.975 * (n_resamples - 1))))
    return ConfidenceInterval("bootstrap", means[lo_idx], means[hi_idx], n)


def _normal_approx_mean_ci(samples: Sequence[float]) -> ConfidenceInterval:
    """95% closed-form normal-approximation CI on the sample mean."""

    n = len(samples)
    if n == 0:
        return ConfidenceInterval("normal_approx", 0.0, 0.0, 0)
    mean = sum(samples) / n
    if n == 1:
        return ConfidenceInterval("normal_approx", mean, mean, 1)
    var = sum((x - mean) ** 2 for x in samples) / (n - 1)
    se = math.sqrt(var / n)
    half = NORMAL_CI_Z * se
    return ConfidenceInterval("normal_approx", mean - half, mean + half, n)


# ---------------------------------------------------------------------------
# Attempt-level WIS (used by the candidate / ranking / sequencing sections).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WISResult:
    """Output of the attempt-level WIS estimator."""

    value: float
    num_records: int
    num_clipped: int
    weight_concentration_index: float    # max(w) / sum(w); 1 = total concentration
    effective_sample_size: float          # (sum w)^2 / sum w^2
    bootstrap_ci: ConfidenceInterval
    normal_approx_ci: ConfidenceInterval


def _wis_from_weighted_rewards(
    weights: Sequence[float],
    rewards: Sequence[float],
    *,
    num_clipped: int,
    rng_seed: int,
) -> WISResult:
    n = len(weights)
    if n == 0:
        empty = ConfidenceInterval("bootstrap", 0.0, 0.0, 0)
        empty_n = ConfidenceInterval("normal_approx", 0.0, 0.0, 0)
        return WISResult(
            value=0.0,
            num_records=0,
            num_clipped=0,
            weight_concentration_index=1.0,
            effective_sample_size=0.0,
            bootstrap_ci=empty,
            normal_approx_ci=empty_n,
        )
    sum_w = sum(weights)
    if sum_w <= 0.0:
        empty = ConfidenceInterval("bootstrap", 0.0, 0.0, n)
        empty_n = ConfidenceInterval("normal_approx", 0.0, 0.0, n)
        return WISResult(
            value=0.0,
            num_records=n,
            num_clipped=num_clipped,
            weight_concentration_index=1.0,
            effective_sample_size=0.0,
            bootstrap_ci=empty,
            normal_approx_ci=empty_n,
        )
    weighted_rewards = [w * r for w, r in zip(weights, rewards)]
    value = sum(weighted_rewards) / sum_w
    wci = max(weights) / sum_w
    ess = (sum_w * sum_w) / sum(w * w for w in weights)
    # CIs are computed over the per-record contributions w_i * r_i,
    # treating the WIS estimator as approximately a sample mean (with
    # the normaliser absorbed). A full-rigour SE needs the delta
    # method; the report exposes both bootstrap and normal_approx.
    per_record_contrib = [w * r * (n / sum_w) for w, r in zip(weights, rewards)]
    return WISResult(
        value=value,
        num_records=n,
        num_clipped=num_clipped,
        weight_concentration_index=wci,
        effective_sample_size=ess,
        bootstrap_ci=_bootstrap_mean_ci(per_record_contrib, rng_seed=rng_seed),
        normal_approx_ci=_normal_approx_mean_ci(per_record_contrib),
    )


def attempt_level_wis(
    records: Sequence[Mapping[str, Any]],
    *,
    target_probs: Sequence[float] | None = None,
    behaviour_probs: Sequence[float] | None = None,
    rng_seed: int = 0,
) -> WISResult:
    """Per-record WIS where each record is its own one-step episode.

    The canonical packs carry no per-candidate-id logging, so for the
    candidate/ranking sections the caller passes ``target_probs is
    None`` and ``behaviour_probs is None``; both default to 1.0, the
    importance ratios are 1.0 and WIS collapses to the mean reward.
    The report flags this with a coverage warning.
    """

    rewards = [compute_record_reward(rec) for rec in records]
    n = len(records)
    if target_probs is None:
        target_probs = [1.0] * n
    if behaviour_probs is None:
        behaviour_probs = [1.0] * n
    if not (len(target_probs) == len(behaviour_probs) == n):
        raise MalformedOPEInputError(
            "target_probs / behaviour_probs length mismatch with records"
        )
    weights: list[float] = []
    num_clipped = 0
    for tp, bp in zip(target_probs, behaviour_probs):
        if bp <= 0.0:
            weights.append(0.0)
            continue
        raw = tp / bp
        if raw > IMPORTANCE_WEIGHT_CLIP:
            num_clipped += 1
            weights.append(IMPORTANCE_WEIGHT_CLIP)
        else:
            weights.append(raw)
    return _wis_from_weighted_rewards(
        weights, rewards, num_clipped=num_clipped, rng_seed=rng_seed
    )


# ---------------------------------------------------------------------------
# Sequencing tabular Q (gamma=0 bandit) + Direct-Method estimator.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TabularQ:
    """Tabular Q mapping (state_key_string, action) to an empirical mean.

    With ``gamma=0`` (one-shot sequencing semantics) this is the
    Bellman fixed point in closed form: Q(s, a) = mean of immediate
    next-record rewards for all (s, a) pairs in the training data.
    Cells with no support fall back to ``0.0``.
    """

    table: Mapping[str, Mapping[str, float]]
    support: Mapping[str, Mapping[str, int]]

    def value(self, state_key: str, action: str) -> float:
        cell = self.table.get(state_key)
        if cell is None:
            return 0.0
        return float(cell.get(action, 0.0))

    def has_support(self, state_key: str, action: str) -> bool:
        cell = self.support.get(state_key)
        if cell is None:
            return False
        return int(cell.get(action, 0)) > 0


def _build_sequencing_pairs(
    records: Sequence[Mapping[str, Any]],
) -> list[tuple[SequencingStateKey, str, float, Mapping[str, Any]]]:
    """Walk the (state, action, reward, successor) sequencing pair set.

    The state is taken from the current record (failure-only, mirroring
    the sequencing trainer), the action is derived from the successor,
    and the reward is the successor's per-attempt reward.
    """

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for rec in records:
        groups.setdefault(_ranking_group_key(rec), []).append(rec)

    pairs: list[tuple[SequencingStateKey, str, float, Mapping[str, Any]]] = []
    for group_recs in groups.values():
        if len(group_recs) < 2:
            continue
        for i in range(len(group_recs) - 1):
            cur = group_recs[i]
            nxt = group_recs[i + 1]
            if _extract_outcome_label(cur) != "failure":
                # Success records never invoke the sequencing seam.
                continue
            state = _record_state_key(cur)
            action = _derive_action_from_successor(nxt)
            reward = compute_record_reward(nxt)
            pairs.append((state, action, reward, nxt))
    return pairs


def fit_sequencing_tabular_q(records: Sequence[Mapping[str, Any]]) -> TabularQ:
    """Empirical mean-reward Q over (state, action) cells."""

    sums: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for state, action, reward, _nxt in _build_sequencing_pairs(records):
        sk = state.as_string_key()
        sums.setdefault(sk, {}).setdefault(action, 0.0)
        counts.setdefault(sk, {}).setdefault(action, 0)
        sums[sk][action] += reward
        counts[sk][action] += 1
    table: dict[str, dict[str, float]] = {}
    for sk, action_sums in sums.items():
        table[sk] = {
            a: (action_sums[a] / counts[sk][a]) for a in action_sums
        }
    return TabularQ(table=table, support=counts)


@dataclass(frozen=True)
class DirectMethodResult:
    """Direct-Method value estimate over a fitted tabular Q."""

    value: float
    num_states_evaluated: int
    num_states_covered: int          # state cells with at least one Q sample for the target action
    fraction_covered: float
    bootstrap_ci: ConfidenceInterval
    normal_approx_ci: ConfidenceInterval


def direct_method_sequencing(
    *,
    fitted_q: TabularQ,
    eval_records: Sequence[Mapping[str, Any]],
    target_action_for_state: Callable[[SequencingStateKey], str],
    rng_seed: int = 0,
) -> DirectMethodResult:
    """DM estimate: mean of ``Q(s, pi_target(s))`` over the failure-side
    state distribution induced by ``eval_records``.
    """

    pairs = _build_sequencing_pairs(eval_records)
    if not pairs:
        empty = ConfidenceInterval("bootstrap", 0.0, 0.0, 0)
        empty_n = ConfidenceInterval("normal_approx", 0.0, 0.0, 0)
        return DirectMethodResult(0.0, 0, 0, 0.0, empty, empty_n)
    per_state_values: list[float] = []
    covered = 0
    for state, _action, _reward, _nxt in pairs:
        sk = state.as_string_key()
        tgt = target_action_for_state(state)
        q = fitted_q.value(sk, tgt)
        per_state_values.append(q)
        if fitted_q.has_support(sk, tgt):
            covered += 1
    n = len(per_state_values)
    mean = sum(per_state_values) / n
    return DirectMethodResult(
        value=mean,
        num_states_evaluated=n,
        num_states_covered=covered,
        fraction_covered=covered / n,
        bootstrap_ci=_bootstrap_mean_ci(per_state_values, rng_seed=rng_seed),
        normal_approx_ci=_normal_approx_mean_ci(per_state_values),
    )


def wis_sequencing(
    *,
    records: Sequence[Mapping[str, Any]],
    target_action_for_state: Callable[[SequencingStateKey], str],
    rng_seed: int = 0,
) -> WISResult:
    """WIS over the sequencing pairs with epsilon-smoothed behaviour."""

    pairs = _build_sequencing_pairs(records)
    n = len(pairs)
    if n == 0:
        empty = ConfidenceInterval("bootstrap", 0.0, 0.0, 0)
        empty_n = ConfidenceInterval("normal_approx", 0.0, 0.0, 0)
        return WISResult(0.0, 0, 0, 1.0, 0.0, empty, empty_n)
    k = len(SEQUENCING_ACTIONS)
    weights: list[float] = []
    rewards: list[float] = []
    num_clipped = 0
    for state, action, reward, _nxt in pairs:
        rewards.append(reward)
        target_action = target_action_for_state(state)
        # epsilon-smoothed behaviour: the behaviour deterministically
        # chose ``action`` (from the next record). The target policy is
        # deterministic too, so target_prob is 1.0 on the target action
        # and 0.0 elsewhere. The same smoothing keeps ratios
        # well-defined.
        b_prob = behaviour_action_probability(num_actions=k, chosen=True)
        if target_action == action:
            t_prob = behaviour_action_probability(num_actions=k, chosen=True)
        else:
            t_prob = behaviour_action_probability(num_actions=k, chosen=False)
        raw = t_prob / b_prob
        if raw > IMPORTANCE_WEIGHT_CLIP:
            num_clipped += 1
            weights.append(IMPORTANCE_WEIGHT_CLIP)
        else:
            weights.append(raw)
    return _wis_from_weighted_rewards(
        weights, rewards, num_clipped=num_clipped, rng_seed=rng_seed
    )


# ---------------------------------------------------------------------------
# Report builder.
# ---------------------------------------------------------------------------


def _hash_records(records: Sequence[Mapping[str, Any]]) -> str:
    """Stable sha256 over the canonical JSON serialisation of records."""

    hasher = hashlib.sha256()
    for rec in records:
        hasher.update(json.dumps(rec, sort_keys=True, default=str).encode())
        hasher.update(b"\n")
    return hasher.hexdigest()


def _ci_dict(ci: ConfidenceInterval) -> dict[str, Any]:
    return {
        "method": ci.method,
        "lower": ci.lower,
        "upper": ci.upper,
        "n": ci.n,
    }


def _wis_dict(
    wis: WISResult,
    *,
    coverage_warning: str | None = None,
    estimable: bool = True,
) -> dict[str, Any]:
    blob: dict[str, Any] = {
        "estimator": "WIS",
        # An explicit, machine-readable abstain marker. When the behaviour action is not recoverable
        # from the logs (candidate and ranking log no per-candidate id or rank), the importance
        # ratios all collapse to 1.0 and WIS degenerates to the behaviour mean reward, which is not
        # an off-policy estimate. ``estimable=false`` says so plainly, so no consumer mistakes
        # ``value`` for a real lift, and ``coverage_warning`` carries the reason. Recovering the
        # behaviour action needs a GraspAttemptRecord contract change; this abstains rather than
        # fabricates.
        "estimable": estimable,
        "value": wis.value,
        "num_records": wis.num_records,
        "num_clipped": wis.num_clipped,
        "weight_concentration_index": wis.weight_concentration_index,
        "effective_sample_size": wis.effective_sample_size,
        "ci_bootstrap": _ci_dict(wis.bootstrap_ci),
        "ci_normal_approx": _ci_dict(wis.normal_approx_ci),
    }
    if coverage_warning is not None:
        blob["coverage_warning"] = coverage_warning
    return blob


def _dm_dict(dm: DirectMethodResult) -> dict[str, Any]:
    return {
        "estimator": "DirectMethod",
        "value": dm.value,
        "num_states_evaluated": dm.num_states_evaluated,
        "num_states_covered": dm.num_states_covered,
        "fraction_covered": dm.fraction_covered,
        "ci_bootstrap": _ci_dict(dm.bootstrap_ci),
        "ci_normal_approx": _ci_dict(dm.normal_approx_ci),
    }


def build_sequencing_section(
    *,
    records: Sequence[Mapping[str, Any]],
    target_action_for_state: Callable[[SequencingStateKey], str],
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Build the ``sequencing`` report section (WIS + FQE + DM)."""

    fitted_q = fit_sequencing_tabular_q(records)
    dm = direct_method_sequencing(
        fitted_q=fitted_q,
        eval_records=records,
        target_action_for_state=target_action_for_state,
        rng_seed=rng_seed,
    )
    wis = wis_sequencing(
        records=records,
        target_action_for_state=target_action_for_state,
        rng_seed=rng_seed,
    )
    return {
        "policy_kind": "sequencing",
        "estimators": {
            "wis": _wis_dict(wis),
            "direct_method": _dm_dict(dm),
            "fqe": {
                "algorithm": "tabular_bandit_gamma_0",
                "num_state_cells": len(fitted_q.table),
                "num_state_action_cells": sum(
                    len(c) for c in fitted_q.support.values()
                ),
            },
        },
    }


def build_candidate_section(
    records: Sequence[Mapping[str, Any]], *, rng_seed: int = 0
) -> dict[str, Any]:
    """Candidate-policy section: WIS-only over attempt-level rewards.

    The canonical replay packs do not log the per-candidate-id chosen
    by the behaviour, so per-candidate importance ratios cannot be
    reconstructed. WIS is emitted at the attempt level (it collapses
    to mean reward) with a coverage warning, so the limitation is
    auditable.
    """

    wis = attempt_level_wis(records, rng_seed=rng_seed)
    return {
        "policy_kind": "candidate_selection",
        "estimators": {
            "wis": _wis_dict(
                wis,
                coverage_warning=(
                    "behaviour_action_unknown_in_logs: "
                    "replay packs do not store per-candidate-id; "
                    "WIS collapses to mean per-attempt reward."
                ),
                estimable=False,  # abstain: not a real off-policy estimate
            ),
        },
    }


def build_ranking_section(
    records: Sequence[Mapping[str, Any]], *, rng_seed: int = 0
) -> dict[str, Any]:
    """Ranking-policy section: WIS-only (mirror of the candidate section)."""

    wis = attempt_level_wis(records, rng_seed=rng_seed)
    return {
        "policy_kind": "ranking",
        "estimators": {
            "wis": _wis_dict(
                wis,
                coverage_warning=(
                    "behaviour_action_unknown_in_logs: "
                    "replay packs do not store per-candidate ranks; "
                    "WIS collapses to mean per-attempt reward."
                ),
                estimable=False,  # abstain: not a real off-policy estimate
            ),
        },
    }


def build_ope_report(
    *,
    records: Sequence[Mapping[str, Any]],
    sequencing_target_action_for_state: Callable[[SequencingStateKey], str],
    dataset_id: str,
    dataset_paths: Sequence[str],
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Top-level OPE report builder."""

    if not records:
        raise MalformedOPEInputError("records sequence is empty")

    dataset_hash = _hash_records(records)
    # Two of the three sections abstain by construction (the packs do not store the
    # behaviour's per-candidate choice), so a promotion decision that leans on them
    # is leaning on a mean reward.
    logger.info(
        "OPE report over %d record(s) of dataset %r (hash %s, seed %d): sequencing "
        "estimated; candidate and ranking sections ABSTAIN "
        "(behaviour_action_unknown_in_logs)",
        len(records),
        dataset_id,
        dataset_hash[:16],
        rng_seed,
    )
    return {
        "schema_version": OPE_REPORT_SCHEMA_VERSION,
        **build_ope_honesty(dataset_id=dataset_id, dataset_hash=dataset_hash),
        "report_kind": "ope",
        "dataset_id": dataset_id,
        "dataset_paths": list(dataset_paths),
        "dataset_hash": dataset_hash,
        "num_records": len(records),
        "behaviour_policy_assumption": {
            "kind": "epsilon_smoothed_deterministic",
            "epsilon": BEHAVIOUR_POLICY_EPSILON,
            "note": (
                "Logs are deterministic; epsilon-smoothing applied to "
                "make importance weights well-defined. WIS estimates "
                "carry documented bias under this assumption."
            ),
        },
        "reward_coefficients": dict(OPE_REWARD_COEFFICIENTS_V1),
        "importance_weight_clip": IMPORTANCE_WEIGHT_CLIP,
        "bootstrap_n": BOOTSTRAP_N,
        "rng_seed": rng_seed,
        "sections": {
            "candidate": build_candidate_section(records, rng_seed=rng_seed),
            "ranking": build_ranking_section(records, rng_seed=rng_seed),
            "sequencing": build_sequencing_section(
                records=records,
                target_action_for_state=sequencing_target_action_for_state,
                rng_seed=rng_seed,
            ),
        },
    }


# ---------------------------------------------------------------------------
# Target-action adapter using a loaded LookupTableSequencingPolicy.
# ---------------------------------------------------------------------------


def sequencing_target_action_from_policy_path(
    policy_path: Path,
) -> Callable[[SequencingStateKey], str]:
    """Return a ``(state) -> action`` function backed by a sequencing artifact."""

    policy = load_lookup_table_sequencing_policy(policy_path)

    def _target(state: SequencingStateKey) -> str:
        sel = policy.propose_sequencing(
            SequencingRequest(attempt_id="ope", state_key=state)
        )
        return sel.action

    return _target


# ---------------------------------------------------------------------------
# Dataset loader.
# ---------------------------------------------------------------------------


def load_records_for_ope(
    paths: Sequence[Path],
) -> list[Mapping[str, Any]]:
    """Load and concatenate JSONL records from ``paths``."""

    out: list[Mapping[str, Any]] = []
    for p in paths:
        if not p.exists():
            raise MalformedOPEInputError(
                f"OPE dataset path does not exist: {p}"
            )
        for rec in load_jsonl(p):
            out.append(rec)
    logger.debug(
        "Loaded %d OPE record(s) from %d path(s): %s",
        len(out),
        len(paths),
        ", ".join(str(p) for p in paths),
    )
    if not out:
        raise MalformedOPEInputError(
            "OPE dataset is empty after loading all paths"
        )
    return out


__all__ = (
    "OPE_REPORT_SCHEMA_VERSION",
    "BEHAVIOUR_POLICY_EPSILON",
    "IMPORTANCE_WEIGHT_CLIP",
    "BOOTSTRAP_N",
    "NORMAL_CI_Z",
    "OPE_REWARD_COEFFICIENTS_V1",
    "FQE_GAMMA",
    "MalformedOPEInputError",
    "ConfidenceInterval",
    "WISResult",
    "TabularQ",
    "DirectMethodResult",
    "compute_record_reward",
    "behaviour_action_probability",
    "attempt_level_wis",
    "fit_sequencing_tabular_q",
    "direct_method_sequencing",
    "wis_sequencing",
    "build_candidate_section",
    "build_ranking_section",
    "build_sequencing_section",
    "build_ope_report",
    "sequencing_target_action_from_policy_path",
    "load_records_for_ope",
)
