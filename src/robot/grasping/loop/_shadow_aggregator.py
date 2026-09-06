"""Shadow-telemetry aggregator: a stateless, logic-only collaborator of ``BinPickingOrchestrator``.

Holds the shadow producers and finalizers, the RL-observability logic interleaved with the
deterministic perceive, score, decide, commit, execute loop. It owns no per-pick state: the
orchestrator keeps every per-pick slot as ``field(init=False)``. The ``AutonomousGraspService``
shadow seam reads several of those slots via ``getattr`` and ``runtime_pick`` reads another one
directly, so moving a slot off the instance drops telemetry silently, with no error anywhere.
The aggregator reads orchestrator loop state via getter-closures-over-self and returns telemetry;
the orchestrator's thin same-named delegators assign their own slots (return-and-assign), so a
slot-write typo is a mypy error and never a silent new attribute.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import numpy as np

from src.robot.grasping.constants import (
    SHADOW_AGGREGATOR_LOG_FILE,
    create_grasping_logger,
)

#: Every shadow producer below swallows its exceptions by contract ("never break
#: the pick path"). Without a line here a permanently broken shadow looks exactly
#: like a shadow that is switched off: telemetry silently absent, pick unaffected.
logger = create_grasping_logger("ShadowAggregator", SHADOW_AGGREGATOR_LOG_FILE)

if TYPE_CHECKING:  # pragma: no cover (import only for typing)
    from src.robot.grasping.rl.router import (
        PerceptionShadowTelemetry,
        RankingShadowTelemetry,
        RecoveryShadowTelemetry,
        SequencingAttemptState,
        SequencingShadowTelemetry,
        ShadowRouter,
    )

    from src.robot.grasping.loop.pick_loop import CommitPolicy, PickAttempt

#: Per-candidate log cap: full feature rows for the top-N candidates, a tail aggregate beyond.
_CANDIDATE_LOG_TOP_N: int = 16


def _candidate_geometry(candidate: object) -> dict[str, float]:
    """The grasp's own shape, per candidate, as numbers a learner can use.

    Kept alongside ``features`` rather than inside it. Of the fifteen RANKING_FEATURE_KEYS only three
    vary per candidate at all, measured over 84 candidate rows from generated scenes; the pairwise
    ranker differences features within a scene, so a key that is identical for every candidate of a
    scene contributes exactly zero however well it is populated. The grasp's own geometry is the
    opposite: every quantity below moved across the 24 candidates it was measured on::

        grip_width_mm          11 distinct   32.29 .. 38.65 mm
        grasp_z_mm              5 distinct   68.70 .. 104.50 mm
        approach_tilt_deg       4 distinct    0.00 .. 75.00 deg
        approach_clearance_mm   3 distinct    0.00 .. 70.00 mm

    A 75-degree tilted approach and a top-down one are not the same grasp, and nothing in the current
    feature contract can tell them apart.

    The geometry stays a separate block: ``load_pairwise_logistic_ranking_policy`` requires an
    artifact's ``feature_keys`` to equal RANKING_FEATURE_KEYS exactly and raises otherwise, so
    extending that tuple makes every already-committed ranking artifact unloadable. Logging geometry
    beside the policy vector leaves that contract untouched; whether a future policy trains on these
    keys is a separate, versioned decision.

    Pure, and never raises: a candidate missing any of this yields the keys it can supply.
    """
    out: dict[str, float] = {}
    width = getattr(candidate, "grip_width_mm", None)
    if isinstance(width, (int, float)):
        out["grip_width_mm"] = float(width)
    position = getattr(candidate, "position", None)
    if position is not None:
        try:
            out["grasp_z_mm"] = float(np.asarray(position, dtype=np.float64).reshape(-1)[2])
        except Exception:  # noqa: BLE001 (a malformed pose logs no height rather than failing a pick)
            pass
    approach = getattr(candidate, "approach", None)
    if approach is not None:
        try:
            vec = np.array(approach, dtype=np.float64).reshape(-1)[:3]
            norm = float(np.linalg.norm(vec))
            if norm > 0.0:
                # Angle from straight down. 0 deg is a top-down grasp; larger means the wrist is
                # tilted, which changes both reachability and how the jaw meets the surface.
                cosine = float(np.clip(-vec[2] / norm, -1.0, 1.0))
                out["approach_tilt_deg"] = float(np.degrees(np.arccos(cosine)))
        except Exception:  # noqa: BLE001
            pass
    metadata = getattr(candidate, "metadata", None) or {}
    clearance = metadata.get("approach_clearance_mm") if isinstance(metadata, dict) else None
    if isinstance(clearance, (int, float)):
        out["approach_clearance_mm"] = float(clearance)
    return out


def _build_candidate_log(
    deterministic_ranking: "tuple[str, ...]",
    features_by_id: "dict[str, dict[str, object]]",
    *,
    geometry_by_id: "dict[str, dict[str, float]] | None" = None,
    top_n: int = _CANDIDATE_LOG_TOP_N,
) -> "tuple[list[dict[str, object]], str | None]":
    """Assemble the per-candidate log (top-N full rows + a tail aggregate) and the behavior (executed top-1) id.

    Each of the first ``top_n`` rows is ``{candidate_id, rank, executed, features}``; a trailing
    ``{tail_count, tail_mean_geometric_score}`` summary row folds the remainder (mirroring the
    rl_candidate_breakdown shape so consumers join on ``candidate_id``). Pure and side-effect free."""
    ranked = list(deterministic_ranking)
    geometry_by_id = geometry_by_id or {}
    rows: list[dict[str, object]] = []
    for rank, cid in enumerate(ranked[:top_n]):
        row: dict[str, object] = {
            "candidate_id": cid,
            "rank": rank,
            "executed": rank == 0,
            "features": dict(features_by_id.get(cid, {})),
        }
        # Absent rather than empty when there is nothing to say: a candidate that can supply none
        # of this logs a row carrying no ``geometry`` key at all.
        geometry = geometry_by_id.get(cid)
        if geometry:
            row["geometry"] = dict(geometry)
        rows.append(row)
    tail = ranked[top_n:]
    if tail:
        scores: list[float] = []
        for cid in tail:
            raw = features_by_id.get(cid, {}).get("geometric_score", 0.0)
            scores.append(float(raw) if isinstance(raw, (int, float)) else 0.0)
        rows.append(
            {
                "tail_count": len(tail),
                "tail_mean_geometric_score": (sum(scores) / len(scores)) if scores else 0.0,
            }
        )
    behavior_candidate_id = ranked[0] if ranked else None
    return rows, behavior_candidate_id


class _ShadowTelemetryAggregator:
    """Stateless shadow logic: reads orchestrator loop state via getter-closures; the orchestrator owns
    the slots and assigns them from this aggregator's returns (return-and-assign)."""

    def __init__(
        self,
        *,
        max_attempts_getter: Callable[[], int],
        shadow_router_getter: "Callable[[], ShadowRouter | None]",
        viewpoints_getter: Callable[[], list],
        commit_reobserve_getter: Callable[[], int],
        resolved_mode_getter: Callable[[], object],
        commit_policy_getter: "Callable[[], CommitPolicy | None]",
    ) -> None:
        # Getter-closures-over-self (not construction-time copies) so the logic always reads end-of-pick values
        # (e.g. _commit_reobserve_count is incremented mid-loop; _sequencing_current_attempts is rebound per pick).
        self._max_attempts_getter = max_attempts_getter
        self._shadow_router_getter = shadow_router_getter
        self._viewpoints_getter = viewpoints_getter
        self._commit_reobserve_getter = commit_reobserve_getter
        self._resolved_mode_getter = resolved_mode_getter
        self._commit_policy_getter = commit_policy_getter

    def derive_post_attempt_baseline_action(self, *, last_attempt: "PickAttempt") -> str:
        """Deterministic baseline action the loop would take next.

        Pure mapping from the just-recorded attempt's ``action`` label into the sequencing 4-action
        enum, mirroring what :meth:`BinPickingOrchestrator._execute_pick` does on the following
        iteration (or at terminal return). ``"grasp"`` means "continue with the next pick attempt after
        the deterministic perception refresh".
        """
        # Lazy import to avoid load-time coupling with the rl package.
        from src.robot.grasping.rl.sequencing_policy import (
            SEQUENCING_ACTION_ABORT,
            SEQUENCING_ACTION_GRASP,
            SEQUENCING_ACTION_RECOVER,
            SEQUENCING_ACTION_REOBSERVE,
        )

        action = last_attempt.action
        idx = last_attempt.attempt_index
        terminal_idx = idx >= self._max_attempts_getter() - 1
        # Terminal-attempt actions map to abort in every branch: the
        # loop exits to ``RESCANNED_EXHAUSTED``, ``EXECUTION_FAILED``
        # or ``NO_PERCEPTION``.
        if terminal_idx and action != "executed":
            return SEQUENCING_ACTION_ABORT
        if action == "rescan":
            return SEQUENCING_ACTION_GRASP
        if action == "relocate":
            return SEQUENCING_ACTION_RECOVER
        if action == "commit_refused_reobserve":
            return SEQUENCING_ACTION_REOBSERVE
        if action == "commit_refused_exhausted":
            return SEQUENCING_ACTION_ABORT
        if action == "exhausted":
            return SEQUENCING_ACTION_ABORT
        if action == "execution_failed":
            return SEQUENCING_ACTION_GRASP
        if action == "object_not_detected":
            return SEQUENCING_ACTION_GRASP
        if action == "camera_frame_rejected":
            return SEQUENCING_ACTION_ABORT
        # ``"executed"`` is the success path: the seam is skipped.
        # Default safety: anything not enumerated above maps to abort.
        return SEQUENCING_ACTION_ABORT

    def finalize_perception(self, *, attempts: "list[PickAttempt]") -> "PerceptionShadowTelemetry | None":
        """Perception-budget shadow (logging-only). Returns the telemetry, or None on unwired and on
        exception; the orchestrator wrapper unconditionally assigns ``_perception_telemetry`` from
        this return (return-and-assign). Reads the live attempts list + end-of-pick loop state via
        getters."""
        router = self._shadow_router_getter()
        if router is None or getattr(router, "perception_policy", None) is None:
            return None
        try:
            from src.robot.grasping.rl.perception_budget_policy import (
                PERCEPTION_ACTION_CONTINUE,
                PERCEPTION_ACTION_STOP,
                PerceptionBudgetStateKey,
                bucket_candidate_count,
                bucket_mode,
                bucket_occlusion,
                bucket_views_seen,
            )

            views_seen = int(len(self._viewpoints_getter())) + int(self._commit_reobserve_getter())
            attempts = list(attempts)
            last = attempts[-1] if attempts else None
            # PickAttempt does not surface a per-attempt candidate count; use an honest floor: 1 if any
            # attempt produced a ranked winner (score>0), else 0.
            last_cand = 0
            for att in reversed(attempts):
                if att.score and att.score > 0.0:
                    last_cand = 1
                    break
            state_key = PerceptionBudgetStateKey(
                views_seen_bucket=bucket_views_seen(views_seen),
                last_candidate_count_bucket=bucket_candidate_count(last_cand),
                # The deterministic loop carries no occlusion ratio, so 'unknown'.
                last_occlusion_bucket=bucket_occlusion(None),
                mode_bucket=bucket_mode(str(self._resolved_mode_getter())),
            )
            last_action = last.action if last is not None else ""
            kept_capturing = last_action in (
                "rescan",
                "relocate",
                "commit_refused_reobserve",
            )
            baseline_action = (
                PERCEPTION_ACTION_CONTINUE if kept_capturing else PERCEPTION_ACTION_STOP
            )
            attempt_id = (
                f"pick:{len(attempts)}:{last_action}" if attempts else "pick:empty"
            )
            return router.run_perception_shadow(
                attempt_id=attempt_id,
                state_key=state_key,
                baseline_action=baseline_action,
            )
        except Exception as exc:  # noqa: BLE001 (never break the pick path)
            logger.warning("Perception shadow failed, telemetry dropped: %s", exc)
            return None

    def finalize_recovery(self, *, attempts: "list[PickAttempt]") -> "RecoveryShadowTelemetry | None":
        """Recovery shadow (logging-only). Returns the telemetry, or None on unwired, on the success
        path (terminal attempt executed) and on exception; the orchestrator wrapper unconditionally
        assigns ``_recovery_telemetry`` from this return."""
        router = self._shadow_router_getter()
        if router is None or getattr(router, "recovery_policy", None) is None:
            return None
        try:
            attempts = list(attempts)
            last = attempts[-1] if attempts else None
            if last is None or last.action == "executed":
                return None
            from src.robot.grasping.rl.recovery_policy import (
                RECOVERY_ACTION_ABORT_RECOVERY,
                RECOVERY_ACTION_REOBSERVE,
                RECOVERY_ACTION_REPLAN_GRASP,
                RECOVERY_ACTION_RE_SEGMENT,
                RecoveryStateKey,
                bucket_attempt_index,
                bucket_dense,
                bucket_last_outcome,
                bucket_reobserve_count,
            )
            from src.robot.grasping.rl.sequencing_policy import (
                FAILURE_CLASS_UNKNOWN,
            )

            state_key = RecoveryStateKey(
                # The live runtime computes no failure taxonomy (same as the sequencing seam).
                failure_class_bucket=FAILURE_CLASS_UNKNOWN,
                attempt_index_bucket=bucket_attempt_index(int(last.attempt_index)),
                reobserve_count_bucket=bucket_reobserve_count(
                    int(self._commit_reobserve_getter())
                ),
                dense_bucket=bucket_dense(str(self._resolved_mode_getter())),
                last_outcome_bucket=bucket_last_outcome("failed"),
            )
            action = last.action
            if action in ("relocate", "commit_refused_reobserve"):
                baseline_action = RECOVERY_ACTION_REOBSERVE
            elif action == "rescan":
                baseline_action = RECOVERY_ACTION_RE_SEGMENT
            elif action in ("execution_failed", "object_not_detected"):
                baseline_action = RECOVERY_ACTION_REPLAN_GRASP
            else:
                baseline_action = RECOVERY_ACTION_ABORT_RECOVERY
            attempt_id = f"pick:{len(attempts)}:{action}"
            return router.run_recovery_shadow(
                attempt_id=attempt_id,
                state_key=state_key,
                baseline_action=baseline_action,
                attempt_index=int(last.attempt_index),
                max_recovery_attempts=int(self._max_attempts_getter()),
            )
        except Exception as exc:  # noqa: BLE001 (never break the pick path)
            logger.warning("Recovery shadow failed, telemetry dropped: %s", exc)
            return None

    def maybe_capture_sequencing_failure_state(
        self, *, last_attempt: "PickAttempt"
    ) -> "SequencingAttemptState | None":
        """Build a :class:`SequencingAttemptState` for a single failure attempt (success rows never reach
        here). Returns the state, or None on unwired / exception; :meth:`finalize_sequencing` collects the
        non-None returns into the states list."""
        router = self._shadow_router_getter()
        if router is None or getattr(router, "sequencing_policy", None) is None:
            return None
        try:
            from src.robot.grasping.rl.router import SequencingAttemptState
            from src.robot.grasping.rl.sequencing_policy import (
                FAILURE_CLASS_UNKNOWN,
                OUTCOME_LABEL_FAILURE,
                clip_attempt_index,
                clip_commit_reobserve_count,
            )

            commit_count = int(self._commit_reobserve_getter())
            max_reobserve = 0
            commit_policy = self._commit_policy_getter()
            if commit_policy is not None:
                max_reobserve = int(commit_policy.max_reobserve_attempts)
            baseline_action = self.derive_post_attempt_baseline_action(last_attempt=last_attempt)
            return SequencingAttemptState(
                attempt_index=int(last_attempt.attempt_index),
                last_outcome_label=OUTCOME_LABEL_FAILURE,
                attempt_index_clipped=clip_attempt_index(int(last_attempt.attempt_index)),
                commit_reobserve_count_clipped=clip_commit_reobserve_count(commit_count),
                commit_reobserve_count=commit_count,
                # Live runtime does not compute failure taxonomy; the sequencing lookup-table policy serves
                # these from the hand-authored fallback by default.
                last_failure_class=FAILURE_CLASS_UNKNOWN,
                max_attempts=int(self._max_attempts_getter()),
                max_reobserve_attempts=max_reobserve,
                baseline_action=baseline_action,
            )
        except Exception as exc:  # noqa: BLE001 (never break the pick path)
            logger.warning(
                "Sequencing failure-state capture failed, attempt dropped from states: %s",
                exc,
            )
            return None

    def finalize_sequencing(
        self, *, attempts: "list[PickAttempt]"
    ) -> "tuple[SequencingShadowTelemetry | None, list[SequencingAttemptState] | None]":
        """Sequencing shadow. Re-derives the per-failure ``SequencingAttemptState`` list from the final
        attempts (success rows skipped), runs the sequencing shadow, and returns ``(telemetry, states)``. On
        unwired returns ``(None, None)`` (the wrapper leaves ``_sequencing_states`` at its per-pick reset
        value); on a run-shadow exception returns ``(None, <states built so far>)``. The wrapper assigns both
        slots from the return."""
        router = self._shadow_router_getter()
        if router is None or getattr(router, "sequencing_policy", None) is None:
            return None, None
        states: "list[SequencingAttemptState]" = []
        try:
            # Failure states are re-derived from the final attempts list, so no capture hook is
            # needed at the terminal return and continue sites inside the loop.
            attempts = list(attempts)
            for att in attempts:
                if att.action == "executed":
                    continue
                captured = self.maybe_capture_sequencing_failure_state(last_attempt=att)
                if captured is not None:
                    states.append(captured)
            attempt_id = (
                f"pick:{len(attempts)}:{attempts[-1].action}" if attempts else "pick:empty"
            )
            telemetry = router.run_sequencing_shadow(
                attempt_id=attempt_id,
                states=tuple(states),
            )
            return telemetry, states
        except Exception as exc:  # noqa: BLE001 (never break the pick path)
            logger.warning(
                "Sequencing shadow failed after %d state(s), telemetry dropped: %s",
                len(states),
                exc,
            )
            return None, states

    def run_ranking_shadow(
        self,
        *,
        pre_blend_candidates: tuple,
        post_blend_candidates: tuple,
        attempt_index: int,
    ) -> "tuple[RankingShadowTelemetry | None, list[dict[str, object]] | None, str | None]":
        """Ranking-shadow producer. The caller (the orchestrator delegator) guarantees a wired router +
        ranking_policy + non-empty pre-blend. Builds the 15-key feature rows per candidate (precedence:
        rl_state_features > shadow > top-level metadata > 0.0), runs the ranking shadow, and returns
        ``(telemetry, candidate_log, behavior_id)``, or ``(None, None, None)`` on the BLE001 except. The
        candidate log lets offline RL/OPE recover which candidate the deterministic ranking executed."""
        router = self._shadow_router_getter()
        if router is None:  # defensive: the orchestrator delegator already guards this; keeps mypy + runtime safe.
            return None, None, None
        try:
            # Lazy import to avoid module-load coupling.
            from src.robot.grasping.rl.ranking_policy import (
                RANKING_FEATURE_KEYS,
            )

            candidate_ids: list[str] = []
            features_by_id: dict[str, dict[str, object]] = {}
            geometry_by_id: dict[str, dict[str, float]] = {}
            pre_obj_to_id: dict[int, str] = {}
            for idx, cand in enumerate(pre_blend_candidates):
                cid = f"a{attempt_index}_c{idx}"
                pre_obj_to_id[id(cand)] = cid
                candidate_ids.append(cid)
                md = dict(getattr(cand, "metadata", None) or {})
                shadow_md: dict = cast(
                    dict,
                    md.get("shadow") if isinstance(md.get("shadow"), dict) else {},
                )
                rl_md: dict = cast(
                    dict,
                    md.get("rl_state_features")
                    if isinstance(md.get("rl_state_features"), dict)
                    else {},
                )
                # The upstream producers guarantee numeric leaves for every documented key;
                # unwired keys fall through to 0.0.
                row: dict[str, object] = {}
                for key in RANKING_FEATURE_KEYS:
                    if key in rl_md:
                        row[key] = rl_md[key]
                    elif key in shadow_md:
                        row[key] = shadow_md[key]
                    elif key in md:
                        row[key] = md[key]
                    else:
                        row[key] = 0.0
                features_by_id[cid] = row
                geometry_by_id[cid] = _candidate_geometry(cand)
            deterministic_ranking = tuple(
                pre_obj_to_id.get(id(cand), f"unknown_{i}")
                for i, cand in enumerate(post_blend_candidates)
            )
            attempt_id = f"a{attempt_index}"
            telemetry = router.run_ranking_shadow(
                attempt_id=attempt_id,
                candidate_ids=tuple(candidate_ids),
                per_candidate_features=features_by_id,
                deterministic_ranking=deterministic_ranking,
            )
            # Per-candidate features and the behavior action, for offline RL and OPE.
            candidate_log, behavior_candidate_id = _build_candidate_log(
                deterministic_ranking, features_by_id, geometry_by_id=geometry_by_id
            )
            logger.debug(
                "Ranking shadow: %d candidate(s) logged for attempt a%d (behavior=%s)",
                len(candidate_log),
                attempt_index,
                behavior_candidate_id,
            )
            return telemetry, candidate_log, behavior_candidate_id
        except Exception as exc:  # noqa: BLE001 (never break the pick path)
            # Same as the outer guards: the report loses nothing, the slots stay None.
            logger.warning(
                "Ranking shadow failed for attempt %d, candidate log dropped: %s",
                attempt_index,
                exc,
            )
            return None, None, None
