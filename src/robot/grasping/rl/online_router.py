"""Tier-0 online-update coordinator: the mutable seam outside the frozen ShadowRouter.

The :class:`ShadowRouter` is a frozen dataclass that only proposes; it carries no reward channel
and never mutates. Online learning therefore lives here, outside it: this coordinator holds the
frozen base router plus one mutable accumulator per LinUCB family (recovery, perception-budget),
folds in observed ``(state, action, reward)`` outcomes, and on demand rebuilds an effective
frozen router (``frozen (+) online_state`` via :func:`dataclasses.replace`) for the next proposals.

Honesty / safety contract:

* Shadow-only, zero grasp influence: the effective router still only emits telemetry.
* Pure-stdlib default; the accumulator tier follows ``robot.rl.runtime_tier``.
* An effective router's LinUCB policies are online-mutated; :func:`online_state_stamps` produces the
  artifact-root ``online_state`` stamp so the promotion gate refuses them (they must be frozen +
  re-evaluated as a fresh artifact first).

Only the LinUCB families are Tier-0 online here; the candidate/ranking logistic online update is a
different algorithm and is not built. ``observe_from_record`` reuses the same behaviour-tuple
extraction the offline trainers and the promotion OPE use, so online accumulation and offline batch
training consume identical ``(state, action, reward)`` signals: no divergent second code path.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from src.robot.grasping.constants import (
    RL_ONLINE_ROUTER_LOG_FILE,
    create_grasping_logger,
)

from .online_repromotion import (
    RepromotionCadence,
    RepromotionDecision,
    evaluate_repromotion_cadence,
)
from .online_state import (
    ONLINE_TIER_STDLIB,
    OnlineAccumulator,
    apply_online_state,
    build_online_state_stamp,
    make_online_accumulator,
)
from .perception_budget_policy import (
    PERCEPTION_ACTIONS,
    PERCEPTION_ACTIONS_SET,
    LinUCBPerceptionBudgetPolicy,
    PerceptionBudgetStateKey,
)
from .perception_budget_policy import onehot_dim as perception_onehot_dim
from .perception_budget_policy import state_key_to_onehot as perception_onehot
from .recovery_policy import (
    RECOVERY_ACTIONS,
    LinUCBRecoveryPolicy,
    RecoveryStateKey,
)
from .recovery_policy import onehot_dim as recovery_onehot_dim
from .recovery_policy import state_key_to_onehot as recovery_onehot
from .router import ShadowRouter
from .train_perception_budget import (
    _is_in_training_scope as _perception_in_scope,
)
from .train_perception_budget import (
    extract_action_label as _extract_perception_action,
)
from .train_perception_budget import (
    extract_perception_state as _extract_perception_state,
)
from .train_recovery import _record_in_training_scope as _recovery_in_scope
from .train_recovery import extract_recovery_tuples as _extract_recovery_tuples

FAMILY_RECOVERY: str = "recovery"
FAMILY_PERCEPTION: str = "perception"
ONLINE_FAMILIES: tuple[str, ...] = (FAMILY_RECOVERY, FAMILY_PERCEPTION)


#: This coordinator is the only place a policy mutates at runtime, and the mutation
#: is invisible from the artifact on disk. The log therefore records which families
#: went online and how many observations had been folded in when an effective router
#: was handed out; without that number a later "why did it propose that" cannot be
#: reconstructed. ``observe_from_record`` runs per record over a whole log, so it
#: never logs; its fail-soft extraction drops are debug.
logger = create_grasping_logger("RLOnlineRouter", RL_ONLINE_ROUTER_LOG_FILE)


def _reward_from_record(record: Mapping[str, Any]) -> float:
    """Binary attempt reward (1.0 on success); matches the trainers' / OPE labelling."""

    return 1.0 if record.get("final_outcome") == "succeeded" else 0.0


class OnlineUpdateCoordinator:
    """Mutable Tier-0 online-update side-state over a frozen :class:`ShadowRouter`.

    An accumulator is created only for a family whose policy slot is wired and is a diagonal-LinUCB
    policy; other slots are left untouched. Construction never mutates the base router.
    """

    def __init__(
        self,
        base_router: ShadowRouter,
        *,
        tier: str = ONLINE_TIER_STDLIB,
        training_scope: str = "dense",
    ) -> None:
        self._base = base_router
        self._tier = tier
        self._training_scope = training_scope
        self._acc: dict[str, OnlineAccumulator] = {}
        rec = base_router.recovery_policy
        if isinstance(rec, LinUCBRecoveryPolicy):
            self._acc[FAMILY_RECOVERY] = make_online_accumulator(
                tier, dim=recovery_onehot_dim(), actions=RECOVERY_ACTIONS
            )
        perc = base_router.perception_policy
        if isinstance(perc, LinUCBPerceptionBudgetPolicy):
            self._acc[FAMILY_PERCEPTION] = make_online_accumulator(
                tier, dim=perception_onehot_dim(), actions=PERCEPTION_ACTIONS
            )
        log = logger.info if self._acc else logger.warning
        log(
            "Online coordinator built on tier %s, scope %s: %s",
            tier,
            training_scope,
            "families " + ", ".join(sorted(self._acc))
            if self._acc
            else "no LinUCB family attached; every observation will be dropped",
        )

    @property
    def tier(self) -> str:
        return self._tier

    @property
    def active_families(self) -> tuple[str, ...]:
        return tuple(f for f in ONLINE_FAMILIES if f in self._acc)

    def n_updates(self, family: str) -> int:
        acc = self._acc.get(family)
        return acc.n_updates if acc is not None else 0

    # --- direct observation -------------------------------------------------

    def observe_recovery(
        self, *, state_key: RecoveryStateKey, action: str, reward: float
    ) -> None:
        acc = self._acc.get(FAMILY_RECOVERY)
        if acc is None or action not in RECOVERY_ACTIONS:
            return
        acc.observe(onehot=recovery_onehot(state_key), action=action, reward=float(reward))

    def observe_perception(
        self, *, state_key: PerceptionBudgetStateKey, action: str, reward: float
    ) -> None:
        acc = self._acc.get(FAMILY_PERCEPTION)
        if acc is None or action not in PERCEPTION_ACTIONS_SET:
            return
        acc.observe(onehot=perception_onehot(state_key), action=action, reward=float(reward))

    # --- record-driven observation (same extraction as the offline trainers) --

    def observe_from_record(self, record: Mapping[str, Any]) -> int:
        """Fold every behaviour tuple derivable from one finalized record into the accumulators.

        Returns the number of observations applied. Fail-soft: a malformed record contributes
        what it can and never raises onto the caller's path."""

        applied = 0
        if FAMILY_RECOVERY in self._acc and _recovery_in_scope(
            record, self._training_scope
        ):
            try:
                tuples, _drop_tok, _drop_none = _extract_recovery_tuples(record)
            except Exception:  # noqa: BLE001 (trainer is the source of truth)
                # Debug, not warning: this runs once per record over a whole log,
                # and fail-soft dropping is the documented contract.
                logger.debug(
                    "Recovery extraction failed for one record; contributing nothing",
                    exc_info=True,
                )
                tuples = ()
            for t in tuples:
                self.observe_recovery(
                    state_key=t.state_key, action=t.action, reward=t.reward
                )
                applied += 1
        if FAMILY_PERCEPTION in self._acc and _perception_in_scope(
            record, self._training_scope
        ):
            try:
                state = _extract_perception_state(record)
                action = _extract_perception_action(record)
            except Exception:  # noqa: BLE001
                logger.debug(
                    "Perception extraction failed for one record; contributing nothing",
                    exc_info=True,
                )
                state, action = None, None
            if state is not None and action in PERCEPTION_ACTIONS_SET:
                self.observe_perception(
                    state_key=state,
                    action=action,
                    reward=_reward_from_record(record),
                )
                applied += 1
        return applied

    # --- effective router + stamps -----------------------------------------

    def effective_router(self) -> ShadowRouter:
        """Return ``frozen (+) online_state``; identical to the base until an observation lands."""

        overrides: dict[str, Any] = {}
        rec_acc = self._acc.get(FAMILY_RECOVERY)
        base_rec = self._base.recovery_policy
        if (
            rec_acc is not None
            and rec_acc.n_updates > 0
            and isinstance(base_rec, LinUCBRecoveryPolicy)
        ):
            new_A, new_b, new_sup = apply_online_state(
                A_diag_per_action=base_rec.A_diag_per_action,
                b_per_action=base_rec.b_per_action,
                feature_support_per_action=base_rec.feature_support_per_action,
                state=rec_acc.snapshot(),
            )
            overrides["recovery_policy"] = replace(
                base_rec,
                A_diag_per_action=new_A,
                b_per_action=new_b,
                feature_support_per_action=new_sup,
            )
        perc_acc = self._acc.get(FAMILY_PERCEPTION)
        base_perc = self._base.perception_policy
        if (
            perc_acc is not None
            and perc_acc.n_updates > 0
            and isinstance(base_perc, LinUCBPerceptionBudgetPolicy)
        ):
            new_A, new_b, new_sup = apply_online_state(
                A_diag_per_action=base_perc.A_diag_per_action,
                b_per_action=base_perc.b_per_action,
                feature_support_per_action=base_perc.feature_support_per_action,
                state=perc_acc.snapshot(),
            )
            overrides["perception_policy"] = replace(
                base_perc,
                A_diag_per_action=new_A,
                b_per_action=new_b,
                feature_support_per_action=new_sup,
            )
        if not overrides:
            logger.debug(
                "Effective router == base router: no online observation folded in yet"
            )
            return self._base
        logger.info(
            "Effective router carries online-mutated %s (%s); promotion refuses "
            "these until they are frozen and re-evaluated",
            ", ".join(sorted(overrides)),
            ", ".join(
                f"{family}={self.n_updates(family)} update(s)"
                for family in sorted(self._acc)
            ),
        )
        return replace(self._base, **overrides)

    def online_state_stamps(self) -> dict[str, dict[str, object]]:
        """Per-family artifact-root ``online_state`` stamps (for the effective policies)."""

        return {
            family: build_online_state_stamp(acc.snapshot())
            for family, acc in self._acc.items()
        }

    def repromotion_status(
        self,
        *,
        cadence: RepromotionCadence | None = None,
        last_promoted: Mapping[str, int] | None = None,
        last_snapshot: Mapping[str, int] | None = None,
    ) -> dict[str, RepromotionDecision]:
        """Per-family snapshot / re-promotion cadence decision (is the promoted verdict stale?).

        Keeps a promoted ``pass`` from silently rotting as online deltas drift the effective policy:
        once ``stale_after_n`` updates accumulate since the promotion baseline the family is flagged
        ``repromotion_required``. ``last_promoted`` / ``last_snapshot`` map each family to the update
        count at the last promotion / snapshot (default 0 for a fresh coordinator). Every decision
        keeps ``still_shadow=True``: the cadence asks for a fresh promotion, which abstains without
        real episodes, and never activates the effective policy."""

        cad = cadence or RepromotionCadence()
        promoted = last_promoted or {}
        snapped = last_snapshot or {}
        return {
            family: evaluate_repromotion_cadence(
                family=family,
                n_updates=acc.n_updates,
                last_promoted_n_updates=int(promoted.get(family, 0)),
                last_snapshot_n_updates=int(snapped.get(family, 0)),
                cadence=cad,
            )
            for family, acc in self._acc.items()
        }


__all__ = (
    "FAMILY_PERCEPTION",
    "FAMILY_RECOVERY",
    "ONLINE_FAMILIES",
    "OnlineUpdateCoordinator",
)
