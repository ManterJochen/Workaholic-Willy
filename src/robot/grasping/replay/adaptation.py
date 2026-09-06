"""Guarded adaptation: planning, validation, and audit.

This module is replay-only: plan generation and plan validation never
touch the runtime. Adaptation plans are JSON artefacts derived from a
baseline KPI/SLO report (`docs/baselines/u_plus_baseline_v1.json` or
equivalent) and an optional failure-taxonomy report. The change set is
deterministic, because strategies use no clock, no RNG and no I/O,
while ``plan_id`` and ``created_at_ns`` vary per run unless the caller
pins ``now_ns``.

Apply-mode writes a YAML overlay sidecar plus an append-only JSONL
audit entry under ``logs/adaptation/adaptation_audit.jsonl``. The
overlay is opt-in at load time: the config loader merges it into
``robot.*`` at startup (read once, never mid-run) only when
``WILLY_ADAPTATION_OVERLAY`` names the overlay file; unset means no
overlay is merged. The runtime stays silent on adaptation telemetry:
overlay-merged values are indistinguishable from operator-edited
``robot.yaml`` values once loaded, so provenance lives in the audit
JSONL alone.

* Pluggable :class:`AdaptationStrategy` Protocol; :class:`RuleBasedStrategy`
  is the locked default.
* Per-key ``min_value``/``max_value``/``max_abs_step``/``max_rel_step``
  bounds plus a global ``rate_limit_max`` (default 3) per plan.
* ``compute_plan`` and ``--adaptation-mode`` both default to
  ``recommend_only``; mode ``off`` composes a plan with no changes.
  There is no ``adaptation_mode`` config key; the name is a field in
  :mod:`src.robot.grasping.replay.telemetry_catalog`.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import (
    Any,
    Final,
    Iterable,
    Mapping,
    Protocol,
    Sequence,
    cast,
    runtime_checkable,
)

from pydantic.fields import FieldInfo

from src.robot.grasping.constants import (
    REPLAY_ADAPTATION_LOG_FILE,
    create_grasping_logger,
)

#: Planning is pure, but the plan it emits is a proposal to change ``robot.*``.
#: Two things here are decided rather than raised: the rate-limit clip and the
#: validation verdict. Neither leaves a trace anywhere else.
logger = create_grasping_logger("ReplayAdaptation", REPLAY_ADAPTATION_LOG_FILE)


#: Bump on any contract-level change to plan schema, audit entry
#: layout, or overlay semantics.
ADAPTATION_MODULE_VERSION: Final[int] = 1

#: Default global rate limit: a single plan may mutate at most this
#: many distinct ``key_path``s. Per-plan override is allowed within
#: ``[1, MAX_RATE_LIMIT]``.
DEFAULT_RATE_LIMIT_MAX: Final[int] = 3
MAX_RATE_LIMIT: Final[int] = 8

#: Allowed adaptation modes; mirrors the catalog enum in
#: :mod:`src.robot.grasping.replay.telemetry_catalog`.
ADAPTATION_MODES: Final[tuple[str, ...]] = (
    "off",
    "recommend_only",
    "apply_with_guardrails",
)


@dataclass(frozen=True, slots=True)
class MutableFieldSpec:
    """A discovered ``runtime_mutable=True`` field with its mutation bounds (``dotted_key`` is rooted at ``robot``, the overlay YAML root)."""

    dotted_key: str
    runtime_type: str  # "int" or "float"
    min_value: float
    max_value: float
    max_abs_step: float
    max_rel_step: float
    current_value: float | int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dotted_key": self.dotted_key,
            "runtime_type": self.runtime_type,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "max_abs_step": self.max_abs_step,
            "max_rel_step": self.max_rel_step,
            "current_value": self.current_value,
        }


def _field_runtime_type(info: FieldInfo) -> str | None:
    """Return ``"int"``/``"float"`` for an int/float field, else ``None`` (bool/str/other not mutable)."""

    ann = info.annotation
    # Optional[int]/Optional[float] not currently used by mutable fields.
    if ann is int:
        return "int"
    if ann is float:
        return "float"
    return None


def _extract_adaptation_meta(info: FieldInfo) -> Mapping[str, Any] | None:
    """Return the ``json_schema_extra`` dict if it marks ``runtime_mutable``."""

    extra = info.json_schema_extra
    if not isinstance(extra, Mapping):
        return None
    if not extra.get("runtime_mutable", False):
        return None
    return extra


def discover_runtime_mutable_fields(
    robot_config: Any,
    *,
    root_prefix: str = "robot",
) -> tuple[MutableFieldSpec, ...]:
    """Walk a ``RobotConfig`` ``StrictModel`` and emit every runtime-mutable field (purely introspective; never imports the runtime grasping stack)."""

    found: list[MutableFieldSpec] = []
    _walk_mutable(robot_config, root_prefix, found)
    # Sort lexicographically for byte-stable output.
    return tuple(sorted(found, key=lambda spec: spec.dotted_key))


def _walk_mutable(
    instance: Any,
    prefix: str,
    out: list[MutableFieldSpec],
) -> None:
    """Recursive helper for :func:`discover_runtime_mutable_fields`."""

    model_fields = getattr(type(instance), "model_fields", None)
    if model_fields is None:
        return
    for name, info in model_fields.items():
        value = getattr(instance, name, None)
        dotted = f"{prefix}.{name}"
        if hasattr(type(value), "model_fields"):
            _walk_mutable(value, dotted, out)
            continue
        meta = _extract_adaptation_meta(info)
        if meta is None:
            continue
        rtype = _field_runtime_type(info)
        if rtype is None:
            # A non-numeric field marked runtime_mutable is ignored;
            # only ints and floats are mutated.
            continue
        try:
            spec = MutableFieldSpec(
                dotted_key=dotted,
                runtime_type=rtype,
                min_value=float(meta["min_value"]),
                max_value=float(meta["max_value"]),
                max_abs_step=float(meta["max_abs_step"]),
                max_rel_step=float(meta["max_rel_step"]),
                current_value=cast("float | int", value),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"runtime_mutable field {dotted!r} has malformed adaptation "
                f"metadata: {meta!r}"
            ) from exc
        out.append(spec)


@dataclass(frozen=True, slots=True)
class ProposedChange:
    """A single ``(key_path, current_value -> proposed_value)`` proposal."""

    key_path: str
    current_value: float | int
    proposed_value: float | int
    rationale: str
    source: str  # strategy name

    def to_dict(self) -> dict[str, Any]:
        return {
            "key_path": self.key_path,
            "current_value": self.current_value,
            "proposed_value": self.proposed_value,
            "rationale": self.rationale,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class AdaptationPlan:
    """A deterministic, auditable adaptation plan."""

    plan_id: str
    created_at_ns: int
    mode: str
    strategy: str
    source_baseline_sha: str | None
    source_taxonomy_sha: str | None
    changes: tuple[ProposedChange, ...]
    rate_limit_max: int
    module_version: int = ADAPTATION_MODULE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "created_at_ns": self.created_at_ns,
            "mode": self.mode,
            "strategy": self.strategy,
            "source_baseline_sha": self.source_baseline_sha,
            "source_taxonomy_sha": self.source_taxonomy_sha,
            "changes": [c.to_dict() for c in self.changes],
            "rate_limit_max": self.rate_limit_max,
            "module_version": self.module_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AdaptationPlan":
        return cls(
            plan_id=str(payload["plan_id"]),
            created_at_ns=int(payload["created_at_ns"]),
            mode=str(payload["mode"]),
            strategy=str(payload["strategy"]),
            source_baseline_sha=_opt_str(payload.get("source_baseline_sha")),
            source_taxonomy_sha=_opt_str(payload.get("source_taxonomy_sha")),
            changes=tuple(
                ProposedChange(
                    key_path=str(c["key_path"]),
                    current_value=c["current_value"],
                    proposed_value=c["proposed_value"],
                    rationale=str(c.get("rationale", "")),
                    source=str(c.get("source", "")),
                )
                for c in payload.get("changes", ())
            ),
            rate_limit_max=int(payload.get("rate_limit_max", DEFAULT_RATE_LIMIT_MAX)),
            module_version=int(payload.get("module_version", ADAPTATION_MODULE_VERSION)),
        )


def _opt_str(v: Any) -> str | None:
    return None if v is None else str(v)


def _stable_plan_id(
    changes: Sequence[ProposedChange],
    *,
    created_at_ns: int,
    strategy: str,
) -> str:
    """Deterministic 16-hex plan id over the change tuple, the strategy name and the timestamp."""

    payload = {
        "strategy": strategy,
        "created_at_ns": created_at_ns,
        "changes": [c.to_dict() for c in changes],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


@runtime_checkable
class AdaptationStrategy(Protocol):
    """Pluggable strategy contract; implementations must be pure functions of their inputs (no clock, no RNG, no I/O): identical inputs must yield identical changes."""

    name: str

    def propose(
        self,
        *,
        baseline: Mapping[str, Any] | None,
        taxonomy: Mapping[str, Any] | None,
        mutable_fields: Mapping[str, MutableFieldSpec],
    ) -> tuple[ProposedChange, ...]:
        ...


@dataclass(frozen=True, slots=True)
class RuleBasedStrategy:
    """Default locked rule table: each rule inspects one signal and emits at most one :class:`ProposedChange`; evaluation is deterministic top-to-bottom and ``rate_limit_max`` clips the result."""

    name: str = "rule_based_v1"

    def propose(
        self,
        *,
        baseline: Mapping[str, Any] | None,
        taxonomy: Mapping[str, Any] | None,
        mutable_fields: Mapping[str, MutableFieldSpec],
    ) -> tuple[ProposedChange, ...]:
        out: list[ProposedChange] = []

        # Rule 1: SLO p95 ranking near budget shrinks breach_window
        # so drift is detected sooner.
        out.extend(self._rule_ranking_p95_near_budget(baseline, mutable_fields))

        # Rule 2: a high occlusion_misread count lowers
        # recovery_aggressive_threshold, making recovery more sensitive.
        out.extend(self._rule_occlusion_misread(taxonomy, mutable_fields))

        # Rule 3: a high empty_air_grasp count reduces
        # ranking_blend_weight, which trusts the probability model less.
        out.extend(self._rule_empty_air_grasp(taxonomy, mutable_fields))

        # Rule 4: a high slip_after_grasp count raises
        # ranking_penalty_weight, penalising uncertain candidates.
        out.extend(self._rule_slip_after_grasp(taxonomy, mutable_fields))

        return tuple(out)

    def _rule_ranking_p95_near_budget(
        self,
        baseline: Mapping[str, Any] | None,
        fields: Mapping[str, MutableFieldSpec],
    ) -> tuple[ProposedChange, ...]:
        if not baseline:
            return ()
        slo = baseline.get("runtime_slo")
        if not isinstance(slo, Mapping):
            return ()
        p95 = slo.get("p95_ranking_latency_ms")
        gate = slo.get("p95_ranking_latency_ms_gate")
        if not isinstance(p95, (int, float)) or not isinstance(gate, (int, float)):
            return ()
        if gate <= 0:
            return ()
        # Trigger when p95 is within 10 % of the budget.
        if p95 < 0.90 * float(gate):
            return ()
        key = "robot.grasping.performance.breach_window_size"
        spec = fields.get(key)
        if spec is None:
            return ()
        # Shrink by max_abs_step but never below min_value.
        proposed = max(int(spec.min_value), int(spec.current_value) - int(spec.max_abs_step))
        if proposed == spec.current_value:
            return ()
        return (
            ProposedChange(
                key_path=key,
                current_value=spec.current_value,
                proposed_value=proposed,
                rationale=(
                    f"ranking p95 {p95:.2f} ms >= 90 %% of gate {gate:.2f} ms; "
                    "shrink breach window to detect SLO regressions sooner."
                ),
                source=self.name,
            ),
        )

    def _taxonomy_count(self, taxonomy: Mapping[str, Any] | None, cause: str) -> int:
        if not taxonomy:
            return 0
        per_cause = taxonomy.get("per_root_cause")
        if not isinstance(per_cause, Mapping):
            return 0
        entry = per_cause.get(cause)
        if not isinstance(entry, Mapping):
            return 0
        count = entry.get("count")
        return int(count) if isinstance(count, (int, float)) else 0

    def _rule_occlusion_misread(
        self,
        taxonomy: Mapping[str, Any] | None,
        fields: Mapping[str, MutableFieldSpec],
    ) -> tuple[ProposedChange, ...]:
        if self._taxonomy_count(taxonomy, "occlusion_misread") < 5:
            return ()
        key = "robot.grasping.uncertainty.recovery_aggressive_threshold"
        spec = fields.get(key)
        if spec is None or spec.current_value <= spec.min_value:
            return ()
        # Lower the threshold (make recovery more sensitive) by
        # max_abs_step, clamped to min_value.
        proposed = max(
            spec.min_value,
            float(spec.current_value) - spec.max_abs_step,
        )
        # Round to 4 decimals to keep YAML readable.
        proposed = round(proposed, 4)
        if proposed >= spec.current_value:
            return ()
        return (
            ProposedChange(
                key_path=key,
                current_value=spec.current_value,
                proposed_value=proposed,
                rationale=(
                    "occlusion_misread count >= 5 in failure taxonomy; "
                    "lower recovery_aggressive_threshold to trigger "
                    "reobserve sooner."
                ),
                source=self.name,
            ),
        )

    def _rule_empty_air_grasp(
        self,
        taxonomy: Mapping[str, Any] | None,
        fields: Mapping[str, MutableFieldSpec],
    ) -> tuple[ProposedChange, ...]:
        if self._taxonomy_count(taxonomy, "empty_air_grasp") < 5:
            return ()
        key = "robot.grasping.success_model.ranking_blend_weight"
        spec = fields.get(key)
        if spec is None or spec.current_value <= spec.min_value:
            return ()
        proposed = max(
            spec.min_value,
            float(spec.current_value) - spec.max_abs_step,
        )
        proposed = round(proposed, 4)
        if proposed >= spec.current_value:
            return ()
        return (
            ProposedChange(
                key_path=key,
                current_value=spec.current_value,
                proposed_value=proposed,
                rationale=(
                    "empty_air_grasp count >= 5 in failure taxonomy; "
                    "reduce probability-blend weight (model may be "
                    "over-trusting low-evidence candidates)."
                ),
                source=self.name,
            ),
        )

    def _rule_slip_after_grasp(
        self,
        taxonomy: Mapping[str, Any] | None,
        fields: Mapping[str, MutableFieldSpec],
    ) -> tuple[ProposedChange, ...]:
        if self._taxonomy_count(taxonomy, "slip_after_grasp") < 5:
            return ()
        key = "robot.grasping.uncertainty.ranking_penalty_weight"
        spec = fields.get(key)
        if spec is None or spec.current_value >= spec.max_value:
            return ()
        # Raise the penalty by max_abs_step, clamped to max_value.
        proposed = min(
            spec.max_value,
            float(spec.current_value) + spec.max_abs_step,
        )
        proposed = round(proposed, 4)
        if proposed <= spec.current_value:
            return ()
        return (
            ProposedChange(
                key_path=key,
                current_value=spec.current_value,
                proposed_value=proposed,
                rationale=(
                    "slip_after_grasp count >= 5 in failure taxonomy; "
                    "raise uncertainty ranking_penalty_weight."
                ),
                source=self.name,
            ),
        )


def compute_plan(
    *,
    robot_config: Any,
    baseline: Mapping[str, Any] | None,
    taxonomy: Mapping[str, Any] | None,
    mode: str = "recommend_only",
    strategy: AdaptationStrategy | None = None,
    rate_limit_max: int = DEFAULT_RATE_LIMIT_MAX,
    now_ns: int | None = None,
    source_baseline_sha: str | None = None,
    source_taxonomy_sha: str | None = None,
) -> AdaptationPlan:
    """Compose an adaptation plan from a strategy + signals (always structurally valid; per-change bound enforcement is :func:`validate_plan`'s job)."""

    if mode not in ADAPTATION_MODES:
        raise ValueError(
            f"adaptation mode {mode!r} not in {ADAPTATION_MODES!r}"
        )
    if mode == "off":
        # 'off' never composes a plan with changes.
        changes: tuple[ProposedChange, ...] = ()
        logger.debug("Adaptation mode 'off': no plan composed")
    else:
        strat = strategy or RuleBasedStrategy()
        specs = discover_runtime_mutable_fields(robot_config)
        spec_map = {s.dotted_key: s for s in specs}
        raw = strat.propose(
            baseline=baseline, taxonomy=taxonomy, mutable_fields=spec_map
        )
        # Apply the global rate limit deterministically: a strategy may
        # propose more, but the plan is clipped to the first
        # ``rate_limit_max`` changes in strategy order.
        changes = tuple(raw[:rate_limit_max])
        if len(raw) > len(changes):
            # The clip is logged because a dropped proposal is otherwise
            # indistinguishable downstream from one the strategy never made.
            logger.warning(
                "Strategy %s proposed %d change(s); clipped to the first %d by "
                "rate_limit_max (dropped: %s)",
                strat.name,
                len(raw),
                len(changes),
                ", ".join(c.key_path for c in raw[rate_limit_max:]),
            )
        logger.info(
            "Planned %d change(s) in mode %s via %s over %d mutable field(s): %s",
            len(changes),
            mode,
            strat.name,
            len(spec_map),
            ", ".join(c.key_path for c in changes) or "none",
        )
    if not (1 <= rate_limit_max <= MAX_RATE_LIMIT):
        raise ValueError(
            f"rate_limit_max must be in [1, {MAX_RATE_LIMIT}]; got {rate_limit_max}"
        )
    ts = int(now_ns) if now_ns is not None else time.time_ns()
    strat_name = (strategy or RuleBasedStrategy()).name
    plan_id = _stable_plan_id(
        changes, created_at_ns=ts, strategy=strat_name
    )
    return AdaptationPlan(
        plan_id=plan_id,
        created_at_ns=ts,
        mode=mode,
        strategy=strat_name,
        source_baseline_sha=source_baseline_sha,
        source_taxonomy_sha=source_taxonomy_sha,
        changes=changes,
        rate_limit_max=rate_limit_max,
    )


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    key_path: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"key_path": self.key_path, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class ValidationResult:
    ok: bool
    issues: tuple[ValidationIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": [i.to_dict() for i in self.issues],
        }


def validate_plan(
    plan: AdaptationPlan,
    mutable_fields: Sequence[MutableFieldSpec],
) -> ValidationResult:
    """Validate every change against the allow-list, per-key type/bounds/step limits, and the rate limit."""

    issues: list[ValidationIssue] = []
    spec_map = {s.dotted_key: s for s in mutable_fields}

    if len(plan.changes) > plan.rate_limit_max:
        issues.append(
            ValidationIssue(
                "<plan>",
                f"plan exceeds rate_limit_max: "
                f"{len(plan.changes)} > {plan.rate_limit_max}",
            )
        )
    if len(plan.changes) > MAX_RATE_LIMIT:
        issues.append(
            ValidationIssue(
                "<plan>",
                f"plan exceeds hard MAX_RATE_LIMIT={MAX_RATE_LIMIT}",
            )
        )

    seen: set[str] = set()
    for change in plan.changes:
        if change.key_path in seen:
            issues.append(
                ValidationIssue(change.key_path, "duplicate key_path in plan")
            )
            continue
        seen.add(change.key_path)

        spec = spec_map.get(change.key_path)
        if spec is None:
            issues.append(
                ValidationIssue(
                    change.key_path,
                    "key_path is not in the runtime_mutable allow-list",
                )
            )
            continue

        if spec.runtime_type == "int":
            if not isinstance(change.proposed_value, int) or isinstance(
                change.proposed_value, bool
            ):
                issues.append(
                    ValidationIssue(
                        change.key_path,
                        f"proposed_value must be int; got {type(change.proposed_value).__name__}",
                    )
                )
                continue
        elif spec.runtime_type == "float":
            if isinstance(change.proposed_value, bool) or not isinstance(
                change.proposed_value, (int, float)
            ):
                issues.append(
                    ValidationIssue(
                        change.key_path,
                        f"proposed_value must be number; got {type(change.proposed_value).__name__}",
                    )
                )
                continue

        proposed = float(change.proposed_value)
        current = float(change.current_value)

        if proposed < spec.min_value or proposed > spec.max_value:
            issues.append(
                ValidationIssue(
                    change.key_path,
                    f"proposed_value {proposed} out of bounds "
                    f"[{spec.min_value}, {spec.max_value}]",
                )
            )
            continue

        if abs(proposed - current) > spec.max_abs_step + 1e-12:
            issues.append(
                ValidationIssue(
                    change.key_path,
                    f"abs step {abs(proposed - current):.6g} exceeds "
                    f"max_abs_step {spec.max_abs_step}",
                )
            )
            continue

        if current != 0.0:
            rel = abs(proposed / current - 1.0)
            if rel > spec.max_rel_step + 1e-12:
                issues.append(
                    ValidationIssue(
                        change.key_path,
                        f"rel step {rel:.6g} exceeds "
                        f"max_rel_step {spec.max_rel_step}",
                    )
                )
                continue

    if issues:
        # Returned, not raised: the caller turns the verdict into an exit code,
        # so outside this log the per-key reasons reach only stdout.
        logger.error(
            "Plan %s REJECTED with %d issue(s): %s",
            plan.plan_id,
            len(issues),
            "; ".join(f"{i.key_path}: {i.reason}" for i in issues),
        )
    else:
        logger.info(
            "Plan %s validated: %d change(s) within allow-list, bounds and step limits",
            plan.plan_id,
            len(plan.changes),
        )
    return ValidationResult(ok=not issues, issues=tuple(issues))


def plan_to_overlay_mapping(plan: AdaptationPlan) -> dict[str, Any]:
    """Convert an :class:`AdaptationPlan` to a nested overlay mapping for ``yaml.safe_dump`` (dotted ``key_path``s become nested dicts; no I/O)."""

    out: dict[str, Any] = {}
    for change in plan.changes:
        parts = change.key_path.split(".")
        cursor = out
        for key in parts[:-1]:
            cursor = cursor.setdefault(key, {})
            if not isinstance(cursor, dict):
                raise ValueError(
                    f"overlay key collision at {key!r} while applying "
                    f"{change.key_path!r}"
                )
        cursor[parts[-1]] = change.proposed_value
    return out


class OverlayPathError(ValueError):
    """Typed error for an overlay that touches a non-allow-listed key path.

    No code raises it: ``validate_overlay_against_allowlist`` returns the
    forbidden paths and the config loader turns them into ``ConfigError``.
    """


def flatten_overlay_paths(
    overlay: Mapping[str, Any],
    *,
    prefix: str = "",
) -> tuple[tuple[str, Any], ...]:
    """Return every leaf ``(dotted_key, value)`` in a nested overlay mapping (lists/tuples and ``None`` are emitted as leaves)."""

    out: list[tuple[str, Any]] = []
    for key, value in overlay.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.extend(flatten_overlay_paths(value, prefix=dotted))
        else:
            out.append((dotted, value))
    return tuple(out)


def validate_overlay_against_allowlist(
    overlay: Mapping[str, Any],
    allowed_paths: Iterable[str],
) -> tuple[str, ...]:
    """Return the forbidden ``dotted_key``s in ``overlay``; an empty tuple means the overlay is safe to merge."""

    allowed_set = frozenset(allowed_paths)
    bad: list[str] = []
    for dotted, _value in flatten_overlay_paths(overlay):
        if dotted not in allowed_set:
            bad.append(dotted)
    if bad:
        logger.error(
            "Overlay touches %d non-allow-listed key path(s): %s",
            len(bad),
            ", ".join(bad),
        )
    return tuple(bad)


def invert_plan(plan: AdaptationPlan, *, now_ns: int | None = None) -> AdaptationPlan:
    """Return the rollback plan: every change with current/proposed swapped."""

    inverted = tuple(
        ProposedChange(
            key_path=c.key_path,
            current_value=c.proposed_value,
            proposed_value=c.current_value,
            rationale=f"rollback of {plan.plan_id}",
            source=f"rollback_of:{plan.plan_id}",
        )
        for c in plan.changes
    )
    ts = int(now_ns) if now_ns is not None else time.time_ns()
    new_id = _stable_plan_id(inverted, created_at_ns=ts, strategy=plan.strategy)
    return AdaptationPlan(
        plan_id=new_id,
        created_at_ns=ts,
        mode=plan.mode,
        strategy=plan.strategy,
        source_baseline_sha=plan.source_baseline_sha,
        source_taxonomy_sha=plan.source_taxonomy_sha,
        changes=inverted,
        rate_limit_max=plan.rate_limit_max,
    )


__all__ = [
    "ADAPTATION_MODES",
    "ADAPTATION_MODULE_VERSION",
    "AdaptationPlan",
    "AdaptationStrategy",
    "DEFAULT_RATE_LIMIT_MAX",
    "MAX_RATE_LIMIT",
    "MutableFieldSpec",
    "OverlayPathError",
    "ProposedChange",
    "RuleBasedStrategy",
    "ValidationIssue",
    "ValidationResult",
    "compute_plan",
    "discover_runtime_mutable_fields",
    "flatten_overlay_paths",
    "invert_plan",
    "plan_to_overlay_mapping",
    "validate_overlay_against_allowlist",
    "validate_plan",
]
