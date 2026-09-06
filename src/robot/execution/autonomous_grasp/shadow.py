"""Candidate-selection shadow-router wiring + post-processing.

The shadow router is the only RL-package surface that lands in the runtime; the
operator opts in via ``robot.rl.mode == "rl_shadow"``. Everything here is
fail-safe: a missing field, missing artifact, or any load/run error degrades to
"no shadow" rather than aborting startup or a pick. Shadow routing has no
influence on the executed grasp; it only annotates
``report.shadow_router_telemetry`` and the RL extras on ``report.telemetry``.

Artifact paths resolve from :func:`project_root` rather than ``__file__``-relative
``parents[...]``, so they survive the module being moved within the package.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

from src.robot.constants import RL_SHADOW_LOG_FILE, create_robot_logger
from src.robot.grasping.rl.action_mask import ActionMaskContext
from src.robot.grasping.rl.candidate_policy import CandidatePolicy
from src.robot.grasping.rl.perception_budget_policy import (
    PerceptionBudgetPolicy,
)
from src.robot.grasping.rl.ranking_policy import RankingPolicy
from src.robot.grasping.rl.recovery_policy import RecoveryPolicy
from src.robot.grasping.rl.router import (
    ShadowRouter,
    emit_candidate_shadow_extras,
)
from src.robot.grasping.rl.sequencing_policy import SequencingPolicy
from src.utility.paths import project_root

from .report import AutonomousGraspOutcome

if TYPE_CHECKING:
    from .report import AutonomousGraspReport


# Every failure path in this module degrades to "no shadow" and returns, because RL must never be
# able to abort a pick. An operator who set rl.mode=rl_shadow therefore gets a cell that looks
# configured and routes nothing; this logger is the only evidence either way. Nothing here logs per
# candidate: the router's own telemetry carries the per-pick data.
logger = create_robot_logger("RLShadowWiring", RL_SHADOW_LOG_FILE)


def _first_number(*values: Any) -> float:
    """The first value that is a real number, else ``0.0``. Booleans count; ``None`` does not."""
    for value in values:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _drift_severity_to_unit(value: Any) -> float | None:
    """Map a ``DriftSeverity`` (or its string) onto [0, 1] using the watchdog's own order.

    ``None`` when the value matches no known severity. The result is normalised rather
    than raw, so it carries the same unit-interval shape as the neighbouring features and
    an added severity level rescales the ladder instead of changing what a trained weight
    means.
    """
    if value is None:
        return None
    from src.robot.execution.autonomous_grasp.watchdog import WatchdogCoordinator

    order = WatchdogCoordinator._SEVERITY_ORDER
    text = str(getattr(value, "value", value)).lower()
    for severity, rank in order.items():
        if str(getattr(severity, "value", severity)).lower() == text:
            return rank / max(1, max(order.values()))
    return None


def project_candidate_feature_row(report: "AutonomousGraspReport") -> dict[str, Any]:
    """Project an :class:`AutonomousGraspReport` into the candidate-policy SAR row.

    The 12 keys in
    :data:`src.robot.grasping.rl.candidate_policy.CANDIDATE_FEATURE_KEYS`
    are pulled from the report's telemetry bag, the uncertainty snapshot, and the
    shadow-success snapshot. Every key is defaulted here rather than downstream:
    absent telemetry becomes ``0``, ``0.0`` or ``False`` in the key's own type, so
    a policy never sees a hole and a missing signal is indistinguishable from a
    zero one. :func:`_first_number` picks the first value that is genuinely
    numeric.
    """

    tel = report.telemetry or {}
    unc = report.uncertainty
    shadow = report.shadow_success_telemetry
    row: dict[str, Any] = {
        "uncertainty_score": getattr(unc, "fused", 0.0),
        # `uncertainty_disagreement` is a declared catalog field that no producer writes. The
        # quantity itself is emitted under the decision engine's own name `channel_disagreement`,
        # and the uncertainty snapshot carries it as `.disagreement`. All three are read, catalog
        # name first, so a producer that starts emitting the catalog name wins and the record stops
        # reading zero.
        "uncertainty_disagreement": _first_number(
            tel.get("uncertainty_disagreement"),
            tel.get("channel_disagreement"),
            getattr(unc, "disagreement", None),
        ),
        "fused_view_count": tel.get("fused_view_count", 0),
        "fusion_evidence_quality": tel.get("fusion_evidence_quality", 0.0),
        "predicted_success_probability": (
            getattr(shadow, "predicted_success_probability", 0.0)
            if shadow is not None
            else 0.0
        ),
        "decision_latency_ms": tel.get("decision_latency_ms", 0.0),
        "ranking_latency_ms": tel.get("ranking_latency_ms", 0.0),
        "fusion_latency_ms": tel.get("fusion_latency_ms", 0.0),
        # `drift_severity_numeric` is not a catalog field and nothing writes it; the catalog declares
        # `drift_severity` as a string enum, which is what the watchdog emits. Mapping it here, with
        # the watchdog's own severity order normalised to [0, 1], keeps the numeric form an RL
        # projection concern instead of a second telemetry field.
        "drift_severity": _first_number(
            tel.get("drift_severity_numeric"),
            _drift_severity_to_unit(tel.get("drift_severity")),
        ),
        "ood_flagged": bool(tel.get("ood_flagged", False)),
        "degraded_mode_active": bool(tel.get("degraded_mode_active", False)),
        "multi_view_occlusion_reduced": bool(
            tel.get("multi_view_occlusion_reduced", False)
        ),
    }
    return row


def maybe_build_shadow_router(robot_cfg: Any) -> Optional[ShadowRouter]:
    """Wire a :class:`ShadowRouter` iff ``robot.rl.mode == "rl_shadow"``.

    The configured mode is first passed through the typed admission gate
    :func:`assert_rl_mode_supported`: a mode whose producer has not shipped
    (``rl_active`` / ``rl_experimental``) raises here rather than degrading to a
    silent no-op. ``geometry_only`` / ``hybrid_ml`` are admitted and return
    :data:`None` (no shadow); only ``rl_shadow`` builds a router.

    Fail-safe: once past admission, any missing field, missing artifact, or load
    error returns :data:`None` so the service degrades to "no shadow" rather than
    aborting startup. A deployment that needs a hard wire must assert
    ``service.shadow_router is not None`` in its own boot path.
    """

    rl_cfg = getattr(robot_cfg, "rl", None)
    if rl_cfg is None:
        return None
    mode = getattr(rl_cfg, "mode", None)
    if mode is None:
        return None
    # The typed admission gate runs before the fail-safe try, so the trailing
    # ``except Exception`` cannot swallow this raise.
    from src.robot.grasping.rl import assert_rl_mode_supported

    assert_rl_mode_supported(mode)
    if mode != "rl_shadow":
        return None

    try:
        from src.robot.grasping.rl.candidate_policy import (
            load_logistic_candidate_policy,
        )

        # Operator may override the artifact path on the RL config; otherwise
        # the default path under ``docs/baselines/rl_policies`` is used.
        artifact_path = getattr(rl_cfg, "artifact_path", None)
        if artifact_path is None:
            artifact_path = (
                project_root()
                / "docs"
                / "baselines"
                / "rl_policies"
                / "v2_candidate_baseline_v1.json"
            )
        else:
            artifact_path = Path(artifact_path)
        if not Path(artifact_path).is_file():
            logger.warning(
                "rl.mode=rl_shadow but the candidate policy artifact %s does not exist: running "
                "with no shadow router; nothing will be shadow-routed or annotated.",
                artifact_path,
            )
            return None
        policy = load_logistic_candidate_policy(artifact_path)

        # The artifact says which policy it is; the config says which policy the operator meant.
        # `artifact_path` only selects a file, so without this check a renamed or swapped artifact
        # runs under the name of the one it replaced.
        #
        # Refuse rather than warn: shadow routing writes annotations that later become training
        # data, and data labelled with the wrong policy is worse than no data. An empty declaration
        # is skipped here; `RobotRLConfig._validate_rl_active_requirements` is what makes
        # `policy_id` mandatory for every RL-active mode.
        #
        # The two sides carry different kinds of identifier. The artifact stores `policy_name`
        # and `policy_version` separately and `LogisticCandidatePolicy.policy_id` composes them
        # as `name@version`; `config/robot/robot.rl_datagen.yaml` declares the bare name, which
        # is also the file's own name. A raw string comparison refuses a correct config.
        #
        # So a bare name is read as "any version of this policy" and `name@version` as exact. A
        # swapped artifact still names a different policy and is still refused, while a version
        # bump is caught by anyone who writes `@1`. The same guard in `ActiveCanaryRouter` compares
        # raw because both sides there are machine-written: `build_promotion_report_artifact`
        # writes `report.policy_id` straight out of `policy.policy_id`, so both are `name@version`.
        # This is the one place where one side is typed by a human.
        #
        # The log records which reading applied: "accepted" and "accepted loosely" are different
        # facts about a run that later becomes training data.
        declared = str(getattr(rl_cfg, "policy_id", "") or "")
        actual = str(getattr(policy, "policy_id", "") or "")
        if declared and actual:
            exact = "@" in declared
            # `split("@", 1)[0]` and not `startswith`: a declaration of `v2_candidate` must not
            # accept an artifact named `v2_candidate_baseline_v1`, which is a different policy whose
            # name happens to begin the same way.
            matches = declared == actual if exact else declared == actual.split("@", 1)[0]
            if not matches:
                logger.warning(
                    "rl.policy_id mismatch: config declares %r but the artifact at %s is %r; "
                    "running with no shadow router rather than annotating attempts with the wrong "
                    "policy name.%s",
                    declared, artifact_path, actual,
                    "" if exact else " (the declaration names no version, so only the name before "
                                     "'@' was compared)",
                )
                return None
            if not exact:
                # Logged rather than silent: the run is admissible and the declaration pins no
                # version, so a version bump under that bare name stays visible in the log.
                logger.info(
                    "rl.policy_id %r matches the artifact %r by name; the version was not declared, "
                    "so any version of this policy is accepted. Write %r to pin it.",
                    declared, actual, actual,
                )
        # Optional ranking policy. Operator may override via
        # ``rl.ranking_artifact_path``; otherwise the default path under
        # ``docs/baselines/rl_policies`` is used. A missing artifact falls
        # back to candidate-only, without a warning.
        ranking_policy = None
        try:
            from src.robot.grasping.rl.ranking_policy import (
                load_pairwise_logistic_ranking_policy,
            )

            ranking_artifact_path = getattr(
                rl_cfg, "ranking_artifact_path", None
            )
            if ranking_artifact_path is None:
                ranking_artifact_path = (
                    project_root()
                    / "docs"
                    / "baselines"
                    / "rl_policies"
                    / "v3_ranking_baseline_v1.json"
                )
            else:
                ranking_artifact_path = Path(ranking_artifact_path)
            if Path(ranking_artifact_path).is_file():
                ranking_policy = load_pairwise_logistic_ranking_policy(
                    ranking_artifact_path
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("optional ranking policy unavailable (%s); shadow continues without it", exc)
            ranking_policy = None
        # Optional sequencing policy. Operator may override via
        # ``rl.sequencing_artifact_path``; otherwise the default path under
        # ``docs/baselines/rl_policies`` is used. A missing artifact falls
        # back to the prior policy set, without a warning.
        sequencing_policy = None
        try:
            from src.robot.grasping.rl.sequencing_policy import (
                load_lookup_table_sequencing_policy,
            )

            sequencing_artifact_path = getattr(
                rl_cfg, "sequencing_artifact_path", None
            )
            if sequencing_artifact_path is None:
                sequencing_artifact_path = (
                    project_root()
                    / "docs"
                    / "baselines"
                    / "rl_policies"
                    / "v4_sequencing_baseline_v1.json"
                )
            else:
                sequencing_artifact_path = Path(sequencing_artifact_path)
            if Path(sequencing_artifact_path).is_file():
                sequencing_policy = load_lookup_table_sequencing_policy(
                    sequencing_artifact_path
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("optional sequencing policy unavailable (%s); shadow continues without it", exc)
            sequencing_policy = None
        # Optional perception-budget policy. Operator may override via
        # ``rl.perception_artifact_path``; otherwise the default path under
        # ``docs/baselines/rl_policies`` is used. A missing or invalid
        # artifact leaves the policy :data:`None` and the shadow does not run.
        perception_policy = None
        try:
            from src.robot.grasping.rl.perception_budget_policy import (
                load_linucb_perception_budget_policy,
            )

            perception_artifact_path = getattr(
                rl_cfg, "perception_artifact_path", None
            )
            if perception_artifact_path is None:
                perception_artifact_path = (
                    project_root()
                    / "docs"
                    / "baselines"
                    / "rl_policies"
                    / "v5_perception_budget_baseline_v1.json"
                )
            else:
                perception_artifact_path = Path(perception_artifact_path)
            if Path(perception_artifact_path).is_file():
                perception_policy = load_linucb_perception_budget_policy(
                    perception_artifact_path
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("optional perception policy unavailable (%s); shadow continues without it", exc)
            perception_policy = None
        # Optional recovery policy.
        recovery_policy = None
        try:
            from src.robot.grasping.rl.recovery_policy import (
                load_linucb_recovery_policy,
            )

            recovery_artifact_path = getattr(
                rl_cfg, "recovery_artifact_path", None
            )
            if recovery_artifact_path is None:
                recovery_artifact_path = (
                    project_root()
                    / "docs"
                    / "baselines"
                    / "rl_policies"
                    / "v6_recovery_baseline_v1.json"
                )
            else:
                recovery_artifact_path = Path(recovery_artifact_path)
            if Path(recovery_artifact_path).is_file():
                recovery_policy = load_linucb_recovery_policy(
                    recovery_artifact_path
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("optional recovery policy unavailable (%s); shadow continues without it", exc)
            recovery_policy = None
        logger.info(
            "shadow router wired from %s; optional policies: ranking=%s sequencing=%s "
            "perception=%s recovery=%s",
            artifact_path,
            ranking_policy is not None, sequencing_policy is not None,
            perception_policy is not None, recovery_policy is not None,
        )
        return ShadowRouter(
            candidate_policy=cast(CandidatePolicy, policy),
            ranking_policy=cast(Optional[RankingPolicy], ranking_policy),
            sequencing_policy=cast(Optional[SequencingPolicy], sequencing_policy),
            perception_policy=cast(
                Optional[PerceptionBudgetPolicy], perception_policy
            ),
            recovery_policy=cast(Optional[RecoveryPolicy], recovery_policy),
        )
    except Exception as exc:  # noqa: BLE001
        # A load failure is returned as "no shadow" rather than raised, so the pick path is
        # unaffected and this line is the only record of it.
        logger.error(
            "shadow router could not be built (%s: %s): degrading to no shadow",
            type(exc).__name__, exc,
        )
        return None


def attach_shadow_router(
    *,
    shadow_router: Optional[ShadowRouter],
    runtime: Any,
    report: "AutonomousGraspReport",
) -> "AutonomousGraspReport":
    """Shadow-router post-processor. Fail-safe; never raises.

    Invoked by the public ``pick`` once the deterministic pipeline, including the
    latency tracker tear-down, has produced the final report. Shadow routing has
    no influence on the executed grasp, outcome, or any other field of the
    report; it only annotates
    :attr:`AutonomousGraspReport.shadow_router_telemetry` and the RL extras dict on
    ``report.telemetry``.
    """

    if shadow_router is None:
        return report

    try:
        router = shadow_router
        pick_report = report.pick_report
        attempt_count = (
            len(pick_report.attempts) if pick_report is not None else 0
        )
        if attempt_count == 0:
            # With no attempts, a single placeholder candidate keeps
            # the router emitting a structured "nothing-to-rank"
            # telemetry row. The downstream agreement metric then
            # reports ``top1_agree=False`` under the one-side-missing
            # rule.
            candidate_ids: tuple[str, ...] = ("a0",)
        else:
            candidate_ids = tuple(
                f"a{i}" for i in range(attempt_count)
            )
        deterministic_top1 = (
            "a0"
            if pick_report is not None
            and report.outcome is AutonomousGraspOutcome.SUCCEEDED
            else None
        )
        feature_row = project_candidate_feature_row(report)
        per_candidate_features = {
            cid: feature_row for cid in candidate_ids
        }
        # The mask context is built with all six channels unset, so nothing
        # here masks a candidate: no per-candidate failure-id frozenset is
        # populated from the deterministic stack's fail-closed signals.
        mask_context = ActionMaskContext()
        attempt_id = (
            f"candidate:{report.outcome.value}:{attempt_count}:"
            f"{report.telemetry.get('attempt_wall_time_s', '')}"
        )
        telemetry = router.run_candidate_selection(
            attempt_id=attempt_id,
            candidate_ids=candidate_ids,
            per_candidate_features=per_candidate_features,
            mask_context=mask_context,
            deterministic_top1_id=deterministic_top1,
        )
        extras = emit_candidate_shadow_extras(
            telemetry=telemetry,
            baseline_action=(
                "accept" if deterministic_top1 is not None else "skip"
            ),
            deterministic_top1_id=deterministic_top1,
        )
        # Fold ranking-shadow telemetry, if produced. The orchestrator
        # stashes typed telemetry on ``_ranking_telemetry`` after the
        # blend block; it is read post-pick, folded into the shadow
        # telemetry block, and turned into the four ``rl_ranking_*``
        # extras. Never raises: a wiring failure leaves the shadow
        # telemetry intact.
        ranking_tel = None
        try:
            from src.robot.grasping.rl.router import (
                emit_ranking_extras,
            )

            orchestrator = getattr(runtime, "orchestrator", None)
            ranking_tel = getattr(
                orchestrator, "_ranking_telemetry", None
            )
            if ranking_tel is not None:
                extras.update(
                    emit_ranking_extras(telemetry=ranking_tel)
                )
                telemetry = replace(
                    telemetry, ranking_telemetry=ranking_tel
                )
                # Splice the per-candidate log + behavior action into the record as free-form extra
                # keys. They are present only when the ranking shadow ran, and let the offline RL
                # and OPE tooling recover the per-candidate features and which candidate the
                # deterministic ranking executed.
                candidate_log = getattr(orchestrator, "_candidate_log", None)
                if candidate_log:
                    extras["rl_candidate_features"] = candidate_log
                    behavior_candidate_id = getattr(
                        orchestrator, "_behavior_candidate_id", None
                    )
                    if behavior_candidate_id is not None:
                        extras["rl_behavior_candidate_id"] = behavior_candidate_id
        except Exception:  # noqa: BLE001
            ranking_tel = None
        # Fold sequencing-shadow telemetry, if produced. The orchestrator's
        # :meth:`run` finaliser stashes a typed
        # :class:`SequencingShadowTelemetry` on ``_sequencing_telemetry``; it
        # is spliced into the shadow telemetry block and turned into the five
        # ``rl_sequencing_*`` extras. Never raises.
        try:
            from src.robot.grasping.rl.router import (
                emit_sequencing_extras,
            )

            orchestrator = getattr(runtime, "orchestrator", None)
            sequencing_tel = getattr(
                orchestrator, "_sequencing_telemetry", None
            )
            if sequencing_tel is not None:
                extras.update(
                    emit_sequencing_extras(telemetry=sequencing_tel)
                )
                telemetry = replace(
                    telemetry, sequencing_telemetry=sequencing_tel
                )
        except Exception:  # noqa: BLE001
            pass
        # Fold perception-budget + recovery shadow telemetry (carrier only).
        # These sub-blocks live only on the ShadowRouterTelemetry carrier;
        # ``emit_*_extras`` are not spliced into ``extras``, which feeds
        # ``report.telemetry`` and ``record.extra``, so the replay JSONL and
        # the telemetry catalog stay byte-frozen. Do not add an
        # ``extras.update(...)`` line here pending the catalog decision.
        try:
            orchestrator = getattr(runtime, "orchestrator", None)
            perception_tel = getattr(
                orchestrator, "_perception_telemetry", None
            )
            if perception_tel is not None:
                telemetry = replace(
                    telemetry, perception_telemetry=perception_tel
                )
        except Exception:  # noqa: BLE001
            pass
        try:
            orchestrator = getattr(runtime, "orchestrator", None)
            recovery_tel = getattr(
                orchestrator, "_recovery_telemetry", None
            )
            if recovery_tel is not None:
                telemetry = replace(
                    telemetry, recovery_telemetry=recovery_tel
                )
        except Exception:  # noqa: BLE001
            pass
        new_telemetry = dict(report.telemetry)
        new_telemetry.update(extras)
        return replace(
            report,
            telemetry=new_telemetry,
            shadow_router_telemetry=telemetry,
        )
    except Exception:  # noqa: BLE001
        # Fail-safe: shadow telemetry must never abort a pick.
        return report
