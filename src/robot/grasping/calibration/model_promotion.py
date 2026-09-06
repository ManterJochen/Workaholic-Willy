"""Model promotion gate for the success-probability artifact.

This module is the offline gatekeeper between a trained artifact and the
runtime lifecycle phases that grant the model behavioural influence,
``canary`` and ``active``. It does three things:

1. It evaluates an artifact against a frozen, deterministic synthetic
   validation slice, seeded with :data:`PROMOTION_VALIDATION_SEED`, which is
   distinct from the training and CV seeds.
2. It emits a structured :class:`PromotionReport` carrying the bound
   thresholds, the eval metrics, a SHA-256 attestation chain over the artifact
   bytes, and the tooling version of the evaluator. The report is written to
   ``promotion.json`` next to ``model.json`` and ``manifest.json``.
3. It verifies, at runtime load time, that an artifact carries a valid
   promotion report. There are five clauses:

   * ``verdict == "pass"``;
   * the artifact bytes match the SHA attestation chain;
   * the recorded thresholds are not weaker than the plan-locked limits;
   * the recorded validation slice is the plan-locked one, by strict equality
     on kind, seed and ``n_attempts``. Without that clause,
     ``promote --validation-seed 20260401`` turns a refused regression into a
     verified promotion, and ``--n-attempts 200`` does the same by shrinking
     the exam;
   * the recorded metrics do not contradict the recorded thresholds. Without
     that clause, a report asserting ``"pass"`` beside ``brier: 0.5`` verifies
     clean.

   Verification runs only when the loader is asked for a ``lifecycle_phase`` in
   ``{"canary", "active"}``. ``"shadow"`` loads freely, so development and CI
   paths are unaffected.

   Every field the verifier compares is written by whoever wrote the file, so
   this refuses an accident, a flag or a script or a default, and not a forgery.
   Refusing a forgery would need the signature to cover the bytes of the report
   itself, and it covers the chain SHA, meaning ``model.json`` with
   ``manifest.json``, alone.

Signing
-------
The :class:`PromotionReport` carries a ``signature`` field, ``"none"`` by default. The SHA-256
attestation chain is the trust root and is always checked, and the signature is an optional layer
above it: :func:`build_promotion_report` accepts a ``signer`` and :func:`verify_promotion` a
``verifier``, both described in :mod:`signing`. With no signer injected the signature stays ``"none"``
and no trust root is implied that does not exist, since faking PKI without a key-management story
would be worse than having none. A deployment with one, a cloud KMS or an HSM or a CI signing
identity, drops in a :class:`~signing.Signer`, either by implementing the Protocol or by using the
reference Ed25519 implementation with the optional ``cryptography`` dependency. The runtime verdict
never depends on the signature: it gates on ``verdict == "pass"`` and the SHA chain, and the
signature exists for external attestation anchored in a published trust root.

This module is offline-only. Importing it imports neither sklearn nor any
other heavy ML dependency at the top level, and the dataset builder is
imported lazily inside :func:`evaluate_for_promotion`, because building the
synthetic dataset is the only call site that needs it.
"""

from __future__ import annotations

import getpass
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np

from src.robot.grasping.scoring.success_probability import (
    SuccessProbabilityModel,
    load_success_probability_model,
    predict_proba,
)
from src.robot.grasping.calibration.signing import (
    SIGNATURE_NONE,
    Signer,
    Verifier,
)
from src.robot.grasping.constants import (
    MODEL_PROMOTION_LOG_FILE,
    create_grasping_logger,
)

#: Promotion is an audit path: every verdict, every written attestation and every
#: runtime refusal has to be reconstructable from the log alone, months later.
logger = create_grasping_logger("ModelPromotion", MODEL_PROMOTION_LOG_FILE)


__all__ = [
    "PROMOTION_FILENAME",
    "PROMOTION_SCHEMA_VERSION",
    "PROMOTION_TOOLING_VERSION",
    "PROMOTION_VALIDATION_ATTEMPTS",
    "PROMOTION_VALIDATION_SEED",
    "PromotionLoadError",
    "PromotionReport",
    "PromotionThresholds",
    "PromotionVerdict",
    "ValidationSlice",
    "build_promotion_report",
    "evaluate_for_promotion",
    "load_promotion_report",
    "promote_artifact",
    "verify_promotion",
    "write_promotion_report",
]


# Module-level constants, locked at ship. Bumping one requires a new
# promotion-report schema version and a retrained artifact.

#: Filename of the promotion attestation, written next to ``model.json``.
PROMOTION_FILENAME: str = "promotion.json"

#: Schema version for the on-disk attestation. Bump only with a migration
#: path; runtime verification refuses unknown schema versions.
PROMOTION_SCHEMA_VERSION: int = 1

#: Tooling identifier embedded in every report so future evaluator versions stay distinguishable.
PROMOTION_TOOLING_VERSION: str = "promotion-v1"

#: An independent seed for the validation slice, picked distinct from
#: ``DEFAULT_DATASET_SEED`` at 20260517, ``DEFAULT_TRAIN_SEED`` at 20260518, and
#: the CLI ``eval`` default of DATASET_SEED + 1, which is also 20260518, so that
#: the gate sees data the trainer, the CV and ``eval`` never touched.
PROMOTION_VALIDATION_SEED: int = 20260519

#: The only slice kind the gate attests. Named so the producer and the verifier cannot spell it
#: differently.
_LOCKED_SLICE_KIND: str = "synthetic_split"

#: Sample budget for the validation slice. 2,000 attempts keeps the eval cost under 100 ms on a
#: laptop.
#:
#: It is a thin exam. At this n an artifact near the bound passes or fails depending on which
#: validation seed it is drawn against, so the verdict carries more sampling noise than the
#: thresholds suggest. Raising n is the right repair and it is not a free one: it changes the
#: metrics of every ``promotion.json`` already written, which is a re-promotion rather than a gate
#: fix.
PROMOTION_VALIDATION_ATTEMPTS: int = 2000

#: Lifecycle phases that require a valid promotion. ``"shadow"`` is deliberately
#: excluded: it carries no behavioural influence, and a development or CI path
#: should not depend on a report having been generated.
_PROMOTED_LIFECYCLE_PHASES: frozenset[str] = frozenset({"canary", "active"})


@dataclass(frozen=True, slots=True)
class PromotionThresholds:
    """Plan-honest maxima for the promotion gate."""

    # The bounds the plan locked, and the gate is a real wall: the runtime gate
    # refuses any stored promotion whose thresholds are weaker than these, so a
    # regressed artifact cannot be smuggled past by loosening the recorded bound.
    brier_max: float = 0.09
    log_loss_max: float = 0.31

    def to_dict(self) -> dict[str, float]:
        return {"brier_max": float(self.brier_max), "log_loss_max": float(self.log_loss_max)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromotionThresholds":
        return cls(
            brier_max=float(payload["brier_max"]),
            log_loss_max=float(payload["log_loss_max"]),
        )


#: The bar and the exam this repository actually locked. Named as objects so nothing has to restate
#: the numbers, and so :func:`verify_promotion` can compare a report against them rather than against
#: whatever its own caller happened to ask for.
PLAN_LOCKED_THRESHOLDS: PromotionThresholds = PromotionThresholds()


def bounded_metrics(
    metrics: Mapping[str, float], thresholds: PromotionThresholds,
) -> list[tuple[str, float, str, float]]:
    """Every metric that has a bound, as ``(name, value, sense, limit)``.

    One implementation of which metric is bounded by which threshold, derived from the threshold
    names. The gate names its bounds ``<metric>_max`` and ``<metric>_min``, so the pairing is already
    written down in :class:`PromotionThresholds` and needs no second list beside it. A hand-kept
    ``(("brier", "brier_max"), ("log_loss", "log_loss_max"))`` rots on first contact: adding
    ``ece_max`` leaves the hand list silently checking two metrics while
    :meth:`PromotionReport.render`, which derives the pairing this way, displays three.

    A metric with no matching bound is simply absent: ``ece`` and ``auroc`` are recorded and ungated,
    and that is a fact about the gate rather than an omission here.
    """
    out: list[tuple[str, float, str, float]] = []
    limits = thresholds.to_dict()
    for name in sorted(metrics):
        for suffix, sense in (("_max", "<="), ("_min", ">=")):
            if name + suffix in limits:
                out.append((name, float(metrics[name]), sense, float(limits[name + suffix])))
                break
    return out


def gate_reasons(
    metrics: Mapping[str, float], thresholds: PromotionThresholds,
) -> list[str]:
    """Why these metrics fail these thresholds. Empty means they pass.

    Written as ``not (value <= limit)`` rather than ``value > limit``, which is not a style choice.
    ``json.loads`` accepts a bare ``NaN`` literal and every comparison against NaN is False, so
    ``value > limit`` hands a NaN metric a free pass: a ``promotion.json`` carrying ``brier: NaN``
    verifies clean under the ``>`` form.

    A required metric that is absent is a refusal, not a pass. A report that omits ``log_loss``
    cannot be shown to satisfy the ``log_loss`` bound, and silence is the one answer a gate may never
    give.
    """
    reasons: list[str] = []
    limits = thresholds.to_dict()
    for key, limit in sorted(limits.items()):
        metric = key.rsplit("_", 1)[0]
        if metric not in metrics:
            reasons.append(f"{metric} is missing, so the {key}={limit:.4f} bound cannot be met")
            continue
        value = float(metrics[metric])
        ok = value <= limit if key.endswith("_max") else value >= limit
        if not ok:
            sense = "exceeds max" if key.endswith("_max") else "is below min"
            reasons.append(f"{metric}={value:.4f} {sense} {limit:.4f}")
    return reasons


@dataclass(frozen=True, slots=True)
class ValidationSlice:
    """Describes the deterministic validation slice the gate used."""

    kind: str
    seed: int
    n_attempts: int
    dataset_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "seed": int(self.seed),
            "n_attempts": int(self.n_attempts),
            "dataset_sha256": self.dataset_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValidationSlice":
        return cls(
            kind=str(payload["kind"]),
            seed=int(payload["seed"]),
            n_attempts=int(payload["n_attempts"]),
            dataset_sha256=str(payload["dataset_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class PromotionVerdict:
    """In-memory pass/fail verdict + the reasons that produced it."""

    verdict: str  # "pass" | "fail"
    metrics: Mapping[str, float]
    thresholds: PromotionThresholds
    reasons: tuple[str, ...] = ()

    def passed(self) -> bool:
        return self.verdict == "pass"


@dataclass(frozen=True, slots=True)
class PromotionReport:
    """On-disk attestation payload (mirrors ``promotion.json``)."""

    schema_version: int
    artifact_basename: str
    model_json_sha256: str
    manifest_json_sha256: str
    artifact_sha256: str
    validation: ValidationSlice
    metrics: Mapping[str, float]
    gate_thresholds: PromotionThresholds
    verdict: str
    promoted_at: str
    promoted_by: str
    tooling_version: str
    signature: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_basename": self.artifact_basename,
            "model_json_sha256": self.model_json_sha256,
            "manifest_json_sha256": self.manifest_json_sha256,
            "artifact_sha256": self.artifact_sha256,
            "validation": self.validation.to_dict(),
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "gate_thresholds": self.gate_thresholds.to_dict(),
            "verdict": self.verdict,
            "promoted_at": self.promoted_at,
            "promoted_by": self.promoted_by,
            "tooling_version": self.tooling_version,
            "signature": self.signature,
        }

    @property
    def exit_code(self) -> int:
        """0 promoted, 1 refused. It is a property of the report rather than a rule a handler restates.

        It is deliberately not on the same scale as the one ``verify`` returns, which is 2 for a
        failed verification. Those answer different questions, whether this should ship and whether
        what shipped is still the thing that was signed, and collapsing them would leave a caller
        that only checks for non-zero unable to tell a gate refusal from a tampered artifact.
        """
        return 0 if self.verdict == "pass" else 1

    def render(self) -> str:
        """The attestation, for a person. ``to_dict`` remains the byte contract on disk.

        Every line is a view of a field that is already there. Nothing here recomputes a metric
        or re-decides the verdict: ``promotion.json`` is SHA-chained through
        ``_artifact_sha256(model_sha, manifest_sha)``, so a second computation that disagreed by a
        float would produce a report describing an artifact that does not exist.
        """
        lines = [
            f"promotion {self.verdict.upper()}  {self.artifact_basename}",
            f"  chain sha256    : {self.artifact_sha256}",
            f"    model.json    : {self.model_json_sha256}",
            f"    manifest.json : {self.manifest_json_sha256}",
            f"  validation      : {self.validation.kind} "
            f"seed={self.validation.seed} n={self.validation.n_attempts}",
            f"    dataset sha   : {self.validation.dataset_sha256}",
        ]
        if self.metrics:
            lines.append("  metrics (threshold in brackets where the gate has one):")
            width = max(len(k) for k in self.metrics)
            # The gate names its thresholds `<metric>_max` and `<metric>_min`. That pairing is
            # derived once, in `bounded_metrics`, and this line and `gate_reasons` are its two
            # readers, so a metric this render shows with a bracket is one the gate decides on.
            bounds = {name: (sense, limit) for name, _v, sense, limit in
                      bounded_metrics(self.metrics, self.gate_thresholds)}
            for key in sorted(self.metrics):
                sense_limit = bounds.get(key)
                bound = "" if sense_limit is None else f"  [{sense_limit[0]} {sense_limit[1]:.4f}]"
                lines.append(f"    {key.ljust(width)}  {float(self.metrics[key]):.4f}{bound}")
        lines.extend([
            f"  promoted at     : {self.promoted_at}",
            f"  promoted by     : {self.promoted_by}",
            f"  tooling         : {self.tooling_version} (schema {self.schema_version})",
            f"  signature       : {self.signature}",
            f"  exit code       : {self.exit_code}",
        ])
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromotionReport":
        return cls(
            schema_version=int(payload["schema_version"]),
            artifact_basename=str(payload["artifact_basename"]),
            model_json_sha256=str(payload["model_json_sha256"]),
            manifest_json_sha256=str(payload["manifest_json_sha256"]),
            artifact_sha256=str(payload["artifact_sha256"]),
            validation=ValidationSlice.from_dict(payload["validation"]),
            metrics={k: float(v) for k, v in payload["metrics"].items()},
            gate_thresholds=PromotionThresholds.from_dict(payload["gate_thresholds"]),
            verdict=str(payload["verdict"]),
            promoted_at=str(payload["promoted_at"]),
            promoted_by=str(payload["promoted_by"]),
            tooling_version=str(payload["tooling_version"]),
            signature=str(payload.get("signature", "none")),
        )


def runtime_gate_reasons(
    report: "PromotionReport", *, thresholds: PromotionThresholds = PLAN_LOCKED_THRESHOLDS,
) -> list[str]:
    """Everything :func:`verify_promotion` can decide from the report alone. Empty means acceptable.

    Split out so `promote` can tell an operator, at the moment it writes the file, that the runtime
    will refuse it, rather than leaving them to discover that one command later.

    The slice is checked here because nothing else checks it. ``evaluate_for_promotion`` takes
    ``validation_seed`` and ``n_attempts`` as ordinary keyword arguments and stamps whatever it was
    handed into the report, so without this clause an artifact the locked slice refuses passes on
    another seed, or on a smaller exam, and ``verify`` then accepts the result. One flag would turn
    a refused evaluation into a verified promotion, and the reasoning at
    :data:`PROMOTION_VALIDATION_SEED` about picking a seed the trainer never touched would be
    defended by nothing but a default value.

    Strict equality on ``n_attempts``, not a lower bound. A bigger slice looks like a superset and is
    not one: ``build_synthetic_dataset(seed, 4000)[:2000]`` does not match
    ``build_synthetic_dataset(seed, 2000)`` in even its first row. A different n is a different draw.

    The recorded metrics are checked against the recorded thresholds, because a report may assert
    ``verdict: "pass"`` beside metrics that violate the very bounds it carries. A hand-edited
    ``promotion.json`` with ``brier: 0.5``, ``log_loss: 2.0`` and ``verdict: "pass"`` verifies clean
    at exit 0 without that check.

    What this does not do, stated plainly: it does not stop a forgery. Every field compared here is
    written by whoever wrote the file, so a hand-written ``promotion.json`` carrying the locked
    slice, plausible metrics and a chain SHA recomputed over degraded model bytes passes this
    function. What it closes is the accident, a flag or a script or a default or an operator in a
    hurry, which is the failure this repository meets. Refusing a forgery needs the signature to
    cover the bytes of the report itself, and ``build_promotion_report`` signs the chain SHA alone,
    meaning ``model.json`` with ``manifest.json``, so the slice, the metrics, the verdict and the
    thresholds all sit outside the signed payload.
    """
    reasons: list[str] = []
    if report.verdict != "pass":
        reasons.append(f"verdict={report.verdict!r} != 'pass'")

    # The bar is the stricter of the two. The `thresholds` parameter of `verify_promotion` is the
    # bar its caller demands, and comparing the report against that alone lets a caller demand less
    # than the plan: on the degraded copy, `verify --brier-max 0.5 --log-loss-max 2.0` answers
    # ok=true at exit 0. The parameter means an additional demand, which is its only sound reading,
    # so a caller may ask for more than the plan and never for less.
    effective = PromotionThresholds(
        brier_max=min(thresholds.brier_max, PLAN_LOCKED_THRESHOLDS.brier_max),
        log_loss_max=min(thresholds.log_loss_max, PLAN_LOCKED_THRESHOLDS.log_loss_max),
    )
    if report.gate_thresholds.brier_max > effective.brier_max:
        reasons.append(
            f"recorded brier_max={report.gate_thresholds.brier_max} weaker "
            f"than plan-locked {effective.brier_max}"
        )
    if report.gate_thresholds.log_loss_max > effective.log_loss_max:
        reasons.append(
            f"recorded log_loss_max={report.gate_thresholds.log_loss_max} weaker "
            f"than plan-locked {effective.log_loss_max}"
        )

    v = report.validation
    if v.kind != _LOCKED_SLICE_KIND:
        reasons.append(f"validation kind={v.kind!r} != {_LOCKED_SLICE_KIND!r}")
    if v.seed != PROMOTION_VALIDATION_SEED:
        reasons.append(
            f"validation seed={v.seed} != the plan-locked {PROMOTION_VALIDATION_SEED}; the gate's "
            f"independence from the training and CV seeds is a property of THAT seed only"
        )
    if v.n_attempts != PROMOTION_VALIDATION_ATTEMPTS:
        reasons.append(
            f"validation n_attempts={v.n_attempts} != the plan-locked "
            f"{PROMOTION_VALIDATION_ATTEMPTS}; a different n is a different draw, not a subset"
        )

    if report.verdict == "pass":
        for reason in gate_reasons(report.metrics, report.gate_thresholds):
            reasons.append(f"verdict='pass' contradicts its own recorded metrics: {reason}")
    return reasons


class PromotionLoadError(ValueError):
    """Raised when ``promotion.json`` is missing, malformed, or schema-incompatible."""

    # A runtime caller catches this and ``OSError`` together to stay fail-safe.


def _sha256_of_file(path: Path) -> str:
    """Return the lowercase hex SHA-256 of a file's raw bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_sha256(model_sha: str, manifest_sha: str) -> str:
    """Combine the two per-file SHAs into one deterministic chain SHA (order fixed: model, then manifest)."""
    return hashlib.sha256(
        (model_sha + ":" + manifest_sha).encode("ascii")
    ).hexdigest()


def _dataset_sha256(x: np.ndarray, y: np.ndarray) -> str:
    """Hash of the canonical (X, y) bytes used for the validation eval."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(y, dtype=np.float64).tobytes())
    return h.hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Stable JSON serialisation (sorted keys, trailing newline) matching the trainer."""
    return (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _log_loss(y: np.ndarray, p: np.ndarray, *, eps: float = 1e-15) -> float:
    """Binary cross-entropy log-loss (pure numpy, clipped to avoid log(0))."""
    # Kept here rather than in the trainer so this gate does not change the
    # locked ``evaluate_metrics`` key set {brier, ece, auroc}.
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    if y.shape != p.shape:
        raise ValueError("shape mismatch between y and p")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def evaluate_for_promotion(
    artifact_dir: str | Path,
    *,
    validation_seed: int = PROMOTION_VALIDATION_SEED,
    n_attempts: int = PROMOTION_VALIDATION_ATTEMPTS,
    thresholds: PromotionThresholds = PromotionThresholds(),
) -> tuple[PromotionVerdict, dict[str, float], ValidationSlice]:
    """Load an artifact, score it on the validation slice, and return the verdict + metrics."""
    # A local import, which keeps this module free of sklearn at the top level
    # and rules out an import cycle with the trainer.
    from src.robot.grasping.calibration.success_model_calibration import (
        DatasetSpec,
        build_synthetic_dataset,
        evaluate_metrics,
    )

    spec = DatasetSpec(seed=validation_seed, n_attempts=n_attempts)
    x, y, _modes = build_synthetic_dataset(spec)
    dataset_sha = _dataset_sha256(x, y)

    model: SuccessProbabilityModel = load_success_probability_model(artifact_dir)
    p = predict_proba(model, x)

    base = evaluate_metrics(y, p)  # {brier, ece, auroc}
    metrics: dict[str, float] = {
        "brier": float(base["brier"]),
        "log_loss": _log_loss(y, p),
        "ece": float(base["ece"]),
        "auroc": float(base["auroc"]),
    }

    # The rule lives in `gate_reasons` so that verify applies the same one to a report it did not
    # produce. A reason reads `brier=<value> exceeds max <limit>`, with brier before log_loss.
    reasons = gate_reasons(metrics, thresholds)
    verdict = "pass" if not reasons else "fail"

    logger.info(
        "Promotion eval: verdict=%s brier=%.4f log_loss=%.4f ece=%.4f auroc=%.4f "
        "(seed=%d n=%d, artifact=%s)",
        verdict,
        metrics["brier"],
        metrics["log_loss"],
        metrics["ece"],
        metrics["auroc"],
        validation_seed,
        n_attempts,
        Path(artifact_dir).name,
    )
    if reasons:
        logger.warning("Promotion gate refused: %s", "; ".join(reasons))

    slice_ = ValidationSlice(
        kind=_LOCKED_SLICE_KIND,
        seed=validation_seed,
        n_attempts=n_attempts,
        dataset_sha256=dataset_sha,
    )
    return (
        PromotionVerdict(
            verdict=verdict,
            metrics=metrics,
            thresholds=thresholds,
            reasons=tuple(reasons),
        ),
        metrics,
        slice_,
    )


def _default_promoted_by() -> str:
    # ``getpass.getuser`` honours ``USER`` and ``LOGNAME`` and falls back to
    # the pwd database. Any failure is swallowed, so the API stays total.
    try:
        return getpass.getuser() or "unknown"
    except Exception:  # pragma: no cover (environment-dependent)
        return "unknown"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_promotion_report(
    artifact_dir: str | Path,
    verdict: PromotionVerdict,
    validation: ValidationSlice,
    *,
    promoted_at: str | None = None,
    promoted_by: str | None = None,
    signature: str = SIGNATURE_NONE,
    signer: Signer | None = None,
) -> PromotionReport:
    """Assemble a :class:`PromotionReport` from an evaluated artifact. ``promoted_at`` and ``promoted_by`` override the current UTC time and the OS user, so the report regenerates deterministically. An injected ``signer`` signs the chain SHA; without one the signature stays ``"none"`` and the SHA-256 chain is the sole trust root."""
    a = Path(artifact_dir)
    model_sha = _sha256_of_file(a / "model.json")
    manifest_sha = _sha256_of_file(a / "manifest.json")
    chain_sha = _artifact_sha256(model_sha, manifest_sha)
    effective_signature = (
        signer.sign(chain_sha.encode("utf-8")) if signer is not None else signature
    )

    return PromotionReport(
        schema_version=PROMOTION_SCHEMA_VERSION,
        artifact_basename=a.name,
        model_json_sha256=model_sha,
        manifest_json_sha256=manifest_sha,
        artifact_sha256=chain_sha,
        validation=validation,
        metrics=dict(verdict.metrics),
        gate_thresholds=verdict.thresholds,
        verdict=verdict.verdict,
        promoted_at=promoted_at if promoted_at is not None else _utc_now_iso(),
        promoted_by=promoted_by if promoted_by is not None else _default_promoted_by(),
        tooling_version=PROMOTION_TOOLING_VERSION,
        signature=effective_signature,
    )


def write_promotion_report(
    report: PromotionReport,
    artifact_dir: str | Path,
) -> Path:
    """Write ``promotion.json`` next to ``model.json`` (canonical bytes)."""
    target = Path(artifact_dir) / PROMOTION_FILENAME
    payload = _canonical_json_bytes(report.to_dict())
    target.write_bytes(payload)
    logger.info(
        "Wrote %s (%d bytes, verdict=%s, chain_sha=%s...)",
        target,
        len(payload),
        report.verdict,
        report.artifact_sha256[:12],
    )
    return target


def load_promotion_report(artifact_dir: str | Path) -> PromotionReport:
    """Read ``promotion.json`` from disk and validate the schema header (raises :class:`PromotionLoadError`)."""
    target = Path(artifact_dir) / PROMOTION_FILENAME
    if not target.is_file():
        raise PromotionLoadError(
            f"missing {PROMOTION_FILENAME} under {artifact_dir}"
        )
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionLoadError(f"could not parse {target}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PromotionLoadError(f"{target} is not a JSON object")
    sv = payload.get("schema_version")
    if int(cast(Any, sv)) != PROMOTION_SCHEMA_VERSION:
        raise PromotionLoadError(
            f"unsupported promotion schema_version={sv!r}, "
            f"runtime expects {PROMOTION_SCHEMA_VERSION}"
        )
    try:
        return PromotionReport.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise PromotionLoadError(f"malformed promotion payload: {exc}") from exc


def verify_promotion(
    artifact_dir: str | Path,
    *,
    thresholds: PromotionThresholds = PromotionThresholds(),
    verifier: Verifier | None = None,
    require_signature: bool = False,
) -> tuple[bool, list[str]]:
    """Verify that the on-disk artifact is promoted and untampered. Returns ``(ok, reasons)`` and never raises."""
    a = Path(artifact_dir)
    reasons: list[str] = []
    try:
        report = load_promotion_report(a)
    except (PromotionLoadError, OSError) as exc:
        # Returned rather than raised, so this is the only place it is visible.
        logger.error("Promotion verify failed to load %s: %s", a, exc)
        return False, [f"promotion_load_failed: {exc}"]

    # Everything decidable from the report alone: the verdict, the thresholds, the validation slice,
    # and whether the recorded metrics agree with the recorded bounds. `runtime_gate_reasons` states
    # what each clause is worth and the limit all four share.
    reasons.extend(runtime_gate_reasons(report, thresholds=thresholds))

    # Tamper detection: per-file SHA + chain SHA.
    try:
        live_model = _sha256_of_file(a / "model.json")
        live_manifest = _sha256_of_file(a / "manifest.json")
    except OSError as exc:
        logger.error("Promotion verify could not read artifact files under %s: %s", a, exc)
        return False, [f"artifact_files_unreadable: {exc}"]
    if live_model != report.model_json_sha256:
        reasons.append("model.json SHA mismatch (artifact tampered or stale)")
    if live_manifest != report.manifest_json_sha256:
        reasons.append("manifest.json SHA mismatch (artifact tampered or stale)")
    if _artifact_sha256(live_model, live_manifest) != report.artifact_sha256:
        reasons.append("chained artifact_sha256 mismatch")

    # The optional signature layer above the SHA chain. With no verifier the chain is the sole
    # trust, which is the default. When a deployment configures a Verifier, meaning its published
    # trust root, the signature over the chain SHA is checked, and ``require_signature`` additionally
    # rejects an unsigned report. A malformed or forged signature fails verify() and surfaces here.
    if verifier is not None:
        if report.signature == SIGNATURE_NONE:
            if require_signature:
                reasons.append(
                    "signature_required_but_absent: report is unsigned (signature='none')"
                )
        elif not verifier.verify(
            report.artifact_sha256.encode("utf-8"), report.signature
        ):
            reasons.append("signature_invalid: chain-SHA signature failed verification")

    if reasons:
        logger.warning("Promotion verify REFUSED %s: %s", a.name, "; ".join(reasons))
    else:
        logger.info("Promotion verify passed for %s", a.name)
    return (len(reasons) == 0), reasons


def promote_artifact(
    artifact_dir: str | Path,
    *,
    promoted_at: str | None = None,
    promoted_by: str | None = None,
    thresholds: PromotionThresholds = PromotionThresholds(),
    validation_seed: int = PROMOTION_VALIDATION_SEED,
    n_attempts: int = PROMOTION_VALIDATION_ATTEMPTS,
) -> PromotionReport:
    """Evaluate, build, write and return in one call. ``promotion.json`` is always written, including on a failure, so the audit trail is complete."""
    verdict, _metrics, validation = evaluate_for_promotion(
        artifact_dir,
        validation_seed=validation_seed,
        n_attempts=n_attempts,
        thresholds=thresholds,
    )
    report = build_promotion_report(
        artifact_dir,
        verdict,
        validation,
        promoted_at=promoted_at,
        promoted_by=promoted_by,
    )
    write_promotion_report(report, artifact_dir)
    # Said here rather than one command later. `promote` keeps its own verdict and exit codes, and
    # `verdict` means the metrics cleared the thresholds, which stays true on a 200-sample slice.
    # Without this an operator reads `pass`, exits 0, and ships a file the runtime refuses. It is a
    # pure function of the report: no files, and no second evaluation.
    refusals = runtime_gate_reasons(report)
    if refusals:
        logger.warning(
            "Wrote %s, and the runtime gate WILL REFUSE it: %s. The verdict above is about the "
            "metrics only; `verify` is the wall.",
            PROMOTION_FILENAME, "; ".join(refusals),
        )
    return report


def lifecycle_phase_requires_promotion(lifecycle_phase: str) -> bool:
    """Return ``True`` exactly when this lifecycle phase requires a valid promotion. ``"shadow"`` does not, because it has no behavioural influence."""
    return lifecycle_phase in _PROMOTED_LIFECYCLE_PHASES
