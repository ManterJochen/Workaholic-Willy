"""Replay adaptation CLI handlers.

The adaptation plan / verify / apply / rollback mode handlers plus
their helpers. ``__main__`` imports these at module level and
dispatches to them; argparse + dispatch stay in ``__main__``.
"""

from __future__ import annotations

import json
from pathlib import Path


from src.robot.grasping.constants import (
    REPLAY_ADAPTATION_CLI_LOG_FILE,
    create_grasping_logger,
)
from src.robot.grasping.replay.baseline_report import (
    build_baseline_report,
)
from src.robot.grasping.replay.canonical_datasets import (
    repo_root_from_module,
)

#: What an operator asked for, and every refusal. The verdicts themselves are
#: logged one level down (``adaptation``/``adaptation_io``); what only exists here
#: is the invocation: which plan file, which inputs, and the missing-input
#: refusals that never reach the planner at all.
logger = create_grasping_logger("AdaptationCLI", REPLAY_ADAPTATION_CLI_LOG_FILE)


_DEFAULT_BASELINE_RELATIVE = "docs/baselines/u_plus_baseline_v1.json"


def _default_overlay_dir(repo_root: Path) -> Path:
    return repo_root / "configs" / "overlays"


def _default_audit_path(repo_root: Path) -> Path:
    return repo_root / "logs" / "adaptation" / "adaptation_audit.jsonl"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _sha256_of(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _adaptation_plan_mode(
    *,
    baseline_in: Path | None,
    taxonomy_in: Path | None,
    mode: str,
) -> int:
    from src.config.schema.robot.robot_schema import RobotConfig
    from src.robot.grasping.replay.adaptation import (
        compute_plan,
        discover_runtime_mutable_fields,
        validate_plan,
    )

    repo_root = repo_root_from_module()
    baseline_path = baseline_in or (repo_root / _DEFAULT_BASELINE_RELATIVE)
    if not baseline_path.exists():
        logger.error("adaptation-plan refused: baseline report not found: %s", baseline_path)
        print(
            json.dumps(
                {
                    "mode": "adaptation-plan",
                    "error": f"baseline report not found: {baseline_path}",
                },
                sort_keys=True,
            )
        )
        return 2
    baseline = _load_json(baseline_path)
    baseline_sha = _sha256_of(baseline_path)
    taxonomy: dict | None = None
    taxonomy_sha: str | None = None
    if taxonomy_in is not None:
        if not taxonomy_in.exists():
            logger.error(
                "adaptation-plan refused: taxonomy report not found: %s", taxonomy_in
            )
            print(
                json.dumps(
                    {
                        "mode": "adaptation-plan",
                        "error": f"taxonomy report not found: {taxonomy_in}",
                    },
                    sort_keys=True,
                )
            )
            return 2
        taxonomy = _load_json(taxonomy_in)
        taxonomy_sha = _sha256_of(taxonomy_in)

    logger.info(
        "adaptation-plan in mode %s from baseline %s (sha %s)%s",
        mode,
        baseline_path,
        baseline_sha[:12],
        "" if taxonomy_in is None else f" + taxonomy {taxonomy_in}",
    )
    robot_cfg = RobotConfig()
    plan = compute_plan(
        robot_config=robot_cfg,
        baseline=baseline,
        taxonomy=taxonomy,
        mode=mode,
        source_baseline_sha=baseline_sha,
        source_taxonomy_sha=taxonomy_sha,
    )
    specs = discover_runtime_mutable_fields(robot_cfg)
    result = validate_plan(plan, specs)
    payload = {
        "mode": "adaptation-plan",
        "plan": plan.to_dict(),
        "validation": result.to_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.ok else 2


def _adaptation_verify_mode(plan_path: Path) -> int:
    from src.config.schema.robot.robot_schema import RobotConfig
    from src.robot.grasping.replay.adaptation import (
        AdaptationPlan,
        discover_runtime_mutable_fields,
        validate_plan,
    )

    if not plan_path.exists():
        logger.error("adaptation-verify refused: plan file not found: %s", plan_path)
        print(
            json.dumps(
                {
                    "mode": "adaptation-verify",
                    "error": f"plan file not found: {plan_path}",
                },
                sort_keys=True,
            )
        )
        return 2
    plan = AdaptationPlan.from_dict(_load_json(plan_path))
    specs = discover_runtime_mutable_fields(RobotConfig())
    result = validate_plan(plan, specs)
    payload = {
        "mode": "adaptation-verify",
        "plan_id": plan.plan_id,
        "validation": result.to_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.ok else 2


def _adaptation_apply_mode(
    *,
    plan_path: Path,
    overlay_dir: Path | None,
    audit_path: Path | None,
    auto_rollback_on_regression: bool = True,
    regression_tolerance_ms: float = 0.0,
    post_apply_report_path: Path | None = None,
) -> int:
    from src.config.schema.robot.robot_schema import RobotConfig
    from src.robot.grasping.replay.adaptation import (
        AdaptationPlan,
        discover_runtime_mutable_fields,
        invert_plan,
        validate_plan,
    )
    from src.robot.grasping.replay.adaptation_io import (
        AUDIT_SCHEMA_VERSION,
        AuditEntry,
        active_overlay_path,
        append_audit_entry,
        build_audit_entry,
        clear_active_overlay,
        write_overlay_sidecar,
    )
    from src.robot.grasping.replay.baseline_report import (
        compare_kpi_deltas,
    )

    repo_root = repo_root_from_module()
    overlay_dir = overlay_dir or _default_overlay_dir(repo_root)
    audit_path = audit_path or _default_audit_path(repo_root)

    if not plan_path.exists():
        logger.error("adaptation-apply refused: plan file not found: %s", plan_path)
        print(
            json.dumps(
                {
                    "mode": "adaptation-apply",
                    "error": f"plan file not found: {plan_path}",
                },
                sort_keys=True,
            )
        )
        return 2
    plan = AdaptationPlan.from_dict(_load_json(plan_path))
    if plan.mode != "apply_with_guardrails":
        logger.error(
            "adaptation-apply refused: plan %s has mode %r, not 'apply_with_guardrails'",
            plan.plan_id,
            plan.mode,
        )
        print(
            json.dumps(
                {
                    "mode": "adaptation-apply",
                    "error": (
                        "plan.mode must be 'apply_with_guardrails' to apply "
                        f"(got {plan.mode!r})"
                    ),
                    "plan_id": plan.plan_id,
                },
                sort_keys=True,
            )
        )
        return 2

    specs = discover_runtime_mutable_fields(RobotConfig())
    result = validate_plan(plan, specs)
    if not result.ok:
        append_audit_entry(
            build_audit_entry(plan, action="verify", applied=False),
            audit_path,
        )
        print(
            json.dumps(
                {
                    "mode": "adaptation-apply",
                    "error": "plan failed validation",
                    "plan_id": plan.plan_id,
                    "validation": result.to_dict(),
                },
                sort_keys=True,
            )
        )
        return 2

    sidecar = write_overlay_sidecar(plan, overlay_dir)
    append_audit_entry(
        build_audit_entry(plan, action="apply", applied=True),
        audit_path,
    )

    # Auto-rollback guardrail: compare the runtime-SLO KPI state before the apply against a
    # post-apply measurement; if the verdict reports a regression (and the guardrail is enabled) revert the
    # apply in-place (active overlay cleared + an inverse-plan audit entry).
    #
    # The "before" is the current static baseline report. The "after" carries a real signal only when the
    # caller supplies ``--post-apply-report PATH``, a baseline-report JSON re-measured under the applied
    # overlay by a hardware or sim run. With no such path the after equals the before by construction, so
    # the deltas are zero and the guardrail cannot trigger: it is a structural no-op, not a live regression
    # trap. The detect and revert mechanism (compare_kpi_deltas, invert_plan, clear_active_overlay, the
    # audit entry) is wired; the post-apply re-measurement is the data-bound half.
    pre_apply_report = build_baseline_report(repo_root)
    post_apply_signal_available = post_apply_report_path is not None
    post_apply_report: dict[str, object]
    if post_apply_report_path is not None:
        post_apply_report = json.loads(
            Path(post_apply_report_path).read_text(encoding="utf-8")
        )
    else:
        post_apply_report = pre_apply_report
    verdict = compare_kpi_deltas(
        pre_apply_report,
        post_apply_report,
        tolerance_ms=float(regression_tolerance_ms),
    )
    auto_rollback_triggered = False
    auto_rollback_active_existed: bool | None = None
    auto_rollback_plan_id: str | None = None
    if not post_apply_signal_available:
        # The guardrail cannot fire without a post-apply re-measurement, so a
        # clean apply is not a passed gate.
        logger.warning(
            "Auto-rollback guardrail is INERT for plan %s: no --post-apply-report, "
            "so the after-measurement equals the before by construction",
            plan.plan_id,
        )
    if (
        auto_rollback_on_regression
        and len(verdict.regressions) > 0
    ):
        rollback_plan = invert_plan(plan)
        active = active_overlay_path(overlay_dir)
        auto_rollback_active_existed = bool(active.exists())
        clear_active_overlay(overlay_dir)
        notes = (
            "auto-rollback: regression detected: "
            + "; ".join(verdict.regressions)
        )
        entry = AuditEntry(
            schema_version=AUDIT_SCHEMA_VERSION,
            plan_id=rollback_plan.plan_id,
            timestamp_ns=rollback_plan.created_at_ns,
            mode=rollback_plan.mode,
            strategy=rollback_plan.strategy,
            action="rollback",
            applied=True,
            source_baseline_sha=rollback_plan.source_baseline_sha,
            source_taxonomy_sha=rollback_plan.source_taxonomy_sha,
            rollback_of=plan.plan_id,
            changes=tuple(c.to_dict() for c in rollback_plan.changes),
            notes=notes,
        )
        append_audit_entry(entry, audit_path)
        auto_rollback_triggered = True
        auto_rollback_plan_id = rollback_plan.plan_id
        logger.warning(
            "AUTO-ROLLBACK of plan %s (rollback plan %s): %s",
            plan.plan_id,
            rollback_plan.plan_id,
            "; ".join(verdict.regressions),
        )

    payload = {
        "mode": "adaptation-apply",
        "plan_id": plan.plan_id,
        "overlay_path": str(sidecar),
        "active_overlay_path": str(overlay_dir / "adaptation_active.yaml"),
        "audit_path": str(audit_path),
        "changes": [c.to_dict() for c in plan.changes],
        "auto_rollback": {
            "capability_group": "soak_gate",
            "enabled": bool(auto_rollback_on_regression),
            # False unless a real post-apply re-measurement was supplied via --post-apply-report. When
            # false the after equals the before by construction, so the guardrail cannot trigger; it is
            # surfaced here so no operator mistakes it for an active gate.
            "post_apply_signal_available": bool(post_apply_signal_available),
            "tolerance_ms": float(regression_tolerance_ms),
            "regressions": list(verdict.regressions),
            "triggered": bool(auto_rollback_triggered),
            "rollback_plan_id": auto_rollback_plan_id,
            "active_overlay_cleared": auto_rollback_active_existed,
            "deltas": {
                "decision_p95_delta_ms": verdict.decision_p95_delta_ms,
                "ranking_p95_delta_ms": verdict.ranking_p95_delta_ms,
                "fusion_p95_delta_ms": verdict.fusion_p95_delta_ms,
            },
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _adaptation_rollback_mode(
    *,
    plan_id: str,
    overlay_dir: Path | None,
    audit_path: Path | None,
) -> int:
    """Clear the active overlay + append a rollback audit entry (per-plan sidecar archive is kept)."""

    from src.robot.grasping.replay.adaptation import (
        AdaptationPlan,
        ProposedChange,
        invert_plan,
    )
    from src.robot.grasping.replay.adaptation_io import (
        AUDIT_SCHEMA_VERSION,
        AuditEntry,
        active_overlay_path,
        append_audit_entry,
        clear_active_overlay,
        find_plan_in_audit,
    )

    repo_root = repo_root_from_module()
    overlay_dir = overlay_dir or _default_overlay_dir(repo_root)
    audit_path = audit_path or _default_audit_path(repo_root)

    prior = find_plan_in_audit(plan_id, audit_path)
    if prior is None:
        logger.error(
            "adaptation-rollback refused: no prior 'apply' entry for plan %s in %s",
            plan_id,
            audit_path,
        )
        print(
            json.dumps(
                {
                    "mode": "adaptation-rollback",
                    "error": (
                        f"no prior 'apply' entry found for plan_id "
                        f"{plan_id!r} in {audit_path}"
                    ),
                },
                sort_keys=True,
            )
        )
        return 2

    # Reconstruct the prior plan from its audit changes so it can be
    # inverted deterministically.
    prior_changes = tuple(
        ProposedChange(
            key_path=c["key_path"],
            current_value=c["current_value"],
            proposed_value=c["proposed_value"],
            rationale=c["rationale"],
            source=c["source"],
        )
        for c in prior["changes"]
    )
    prior_plan = AdaptationPlan(
        plan_id=plan_id,
        created_at_ns=int(prior["timestamp_ns"]),
        mode=str(prior["mode"]),
        strategy=str(prior["strategy"]),
        source_baseline_sha=prior.get("source_baseline_sha"),
        source_taxonomy_sha=prior.get("source_taxonomy_sha"),
        changes=prior_changes,
        rate_limit_max=len(prior_changes) or 1,
    )
    rollback_plan = invert_plan(prior_plan)

    active = active_overlay_path(overlay_dir)
    active_existed = active.exists()
    clear_active_overlay(overlay_dir)

    entry = AuditEntry(
        schema_version=AUDIT_SCHEMA_VERSION,
        plan_id=rollback_plan.plan_id,
        timestamp_ns=rollback_plan.created_at_ns,
        mode=rollback_plan.mode,
        strategy=rollback_plan.strategy,
        action="rollback",
        applied=True,
        source_baseline_sha=rollback_plan.source_baseline_sha,
        source_taxonomy_sha=rollback_plan.source_taxonomy_sha,
        rollback_of=plan_id,
        changes=tuple(c.to_dict() for c in rollback_plan.changes),
        notes=(
            f"cleared active overlay (existed={active_existed})"
        ),
    )
    append_audit_entry(entry, audit_path)

    payload = {
        "mode": "adaptation-rollback",
        "rolled_back_plan_id": plan_id,
        "rollback_plan_id": rollback_plan.plan_id,
        "active_overlay_cleared": active_existed,
        "audit_path": str(audit_path),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
