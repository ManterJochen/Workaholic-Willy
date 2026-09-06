"""Promotion OPE estimators (WIS + DM) over ``EvaluationTriple`` sequences: the numbers a promotion report carries."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from .ope import (
    BEHAVIOUR_POLICY_EPSILON,
    BOOTSTRAP_N,
    IMPORTANCE_WEIGHT_CLIP,
    ConfidenceInterval,
    behaviour_action_probability,
)

from ._promotion_extractors import EvaluationTriple


@dataclass(frozen=True)
class WISLiftEstimate:
    """WIS value of the target policy minus the baseline mean-reward.

    ``lift_ci`` and ``target_ci`` are bootstrap percentile intervals
    drawn under the promotion report's RNG seed.
    """

    target_value: float
    baseline_value: float
    lift: float
    lift_ci: ConfidenceInterval
    target_ci: ConfidenceInterval
    num_records: int
    num_records_with_weight: int
    num_clipped: int
    effective_sample_size: float
    weight_concentration_index: float


@dataclass(frozen=True)
class DMEstimate:
    """Direct-Method value of the target policy."""

    value: float
    num_records: int


def _per_record_weight(
    behavior_action: str,
    target_action: str,
    *,
    num_actions: int,
    epsilon: float,
    clip: float,
) -> tuple[float, bool]:
    """Importance weight ``(weight, was_clipped)`` for one (behavior, target) pair: deterministic target, epsilon-smoothed behavior (per :func:`ope.behaviour_action_probability`)."""

    if behavior_action == target_action:
        target_prob = 1.0
    else:
        target_prob = 0.0
    if target_prob == 0.0:
        return (0.0, False)
    beh = behaviour_action_probability(
        num_actions=num_actions, chosen=True, epsilon=epsilon
    )
    w = target_prob / beh
    if w > clip:
        return (clip, True)
    return (w, False)


def compute_wis_lift(
    triples: Sequence[EvaluationTriple],
    *,
    num_actions: int,
    epsilon: float = BEHAVIOUR_POLICY_EPSILON,
    clip: float = IMPORTANCE_WEIGHT_CLIP,
    bootstrap_n: int = BOOTSTRAP_N,
    seed: int = 0,
) -> WISLiftEstimate:
    """WIS estimator and bootstrap CI for the lift over the baseline.

    The baseline value is the mean reward across all triples, the
    target value is the WIS estimate, and the lift is the target minus
    the baseline.
    """

    n = len(triples)
    if n == 0:
        empty = ConfidenceInterval("bootstrap", 0.0, 0.0, 0)
        return WISLiftEstimate(
            target_value=0.0,
            baseline_value=0.0,
            lift=0.0,
            lift_ci=empty,
            target_ci=empty,
            num_records=0,
            num_records_with_weight=0,
            num_clipped=0,
            effective_sample_size=0.0,
            weight_concentration_index=1.0,
        )
    weights: list[float] = []
    rewards: list[float] = []
    num_clipped = 0
    for t in triples:
        w, clipped = _per_record_weight(
            t.behavior_action,
            t.target_action,
            num_actions=num_actions,
            epsilon=epsilon,
            clip=clip,
        )
        weights.append(w)
        rewards.append(t.reward)
        if clipped:
            num_clipped += 1
    sum_w = sum(weights)
    baseline_value = sum(rewards) / n
    num_with_weight = sum(1 for w in weights if w > 0.0)
    if sum_w <= 0.0:
        empty = ConfidenceInterval("bootstrap", 0.0, 0.0, n)
        return WISLiftEstimate(
            target_value=0.0,
            baseline_value=baseline_value,
            lift=0.0 - baseline_value,
            lift_ci=empty,
            target_ci=empty,
            num_records=n,
            num_records_with_weight=0,
            num_clipped=num_clipped,
            effective_sample_size=0.0,
            weight_concentration_index=1.0,
        )
    target_value = sum(w * r for w, r in zip(weights, rewards)) / sum_w
    wci = max(weights) / sum_w
    ess = (sum_w * sum_w) / sum(w * w for w in weights)
    rng = random.Random(seed)
    target_resamples: list[float] = []
    lift_resamples: list[float] = []
    for _ in range(bootstrap_n):
        sw = 0.0
        swr = 0.0
        sr = 0.0
        for _i in range(n):
            j = rng.randrange(n)
            sw += weights[j]
            swr += weights[j] * rewards[j]
            sr += rewards[j]
        if sw <= 0.0:
            tv = 0.0
        else:
            tv = swr / sw
        bv = sr / n
        target_resamples.append(tv)
        lift_resamples.append(tv - bv)
    target_ci = _percentile_ci(target_resamples, n)
    lift_ci = _percentile_ci(lift_resamples, n)
    return WISLiftEstimate(
        target_value=target_value,
        baseline_value=baseline_value,
        lift=target_value - baseline_value,
        lift_ci=lift_ci,
        target_ci=target_ci,
        num_records=n,
        num_records_with_weight=num_with_weight,
        num_clipped=num_clipped,
        effective_sample_size=ess,
        weight_concentration_index=wci,
    )


def _percentile_ci(samples: Sequence[float], n_records: int) -> ConfidenceInterval:
    if not samples:
        return ConfidenceInterval("bootstrap", 0.0, 0.0, n_records)
    s = sorted(samples)
    m = len(s)
    lo_idx = max(0, int(round(0.025 * (m - 1))))
    hi_idx = min(m - 1, int(round(0.975 * (m - 1))))
    return ConfidenceInterval("bootstrap", s[lo_idx], s[hi_idx], n_records)


def compute_dm(triples: Sequence[EvaluationTriple]) -> DMEstimate:
    """Direct Method: mean of per-record target expected reward."""

    n = len(triples)
    if n == 0:
        return DMEstimate(value=0.0, num_records=0)
    return DMEstimate(
        value=sum(float(t.target_expected_reward) for t in triples) / n,
        num_records=n,
    )


@dataclass(frozen=True)
class IndependentDMEstimate:
    """Independent (data-fit) Direct Method: fits ``Q(s, a) = mean logged reward`` per ``(state_key, behaviour action)`` cell and averages ``Q(s, target_action)`` over the eval distribution. It never reads the policy's circular self-estimate; it is report-only and does not feed the verdict."""

    value: float
    fraction_covered: float
    num_state_action_cells: int
    num_covered_records: int
    num_records: int


def fit_tabular_dm(triples: Sequence[EvaluationTriple]) -> IndependentDMEstimate:
    """Fit the independent tabular DM (see :class:`IndependentDMEstimate`): deterministic means over insertion-ordered ``(state_key, action)`` cells, no RNG (so no platform-locked bootstrap CI)."""

    cells: dict[tuple[str, str], list[float]] = {}
    for t in triples:
        cells.setdefault((t.state_key, t.behavior_action), []).append(float(t.reward))
    q: dict[tuple[str, str], float] = {
        key: sum(rewards) / len(rewards) for key, rewards in cells.items()
    }
    covered: list[float] = []
    for t in triples:
        target_cell = (t.state_key, t.target_action)
        if target_cell in q:
            covered.append(q[target_cell])
    n = len(triples)
    value = sum(covered) / len(covered) if covered else 0.0
    fraction = (len(covered) / n) if n else 0.0
    return IndependentDMEstimate(
        value=value,
        fraction_covered=fraction,
        num_state_action_cells=len(q),
        num_covered_records=len(covered),
        num_records=n,
    )
