"""Canonical replay datasets and manifest contract.

Declares the immutable synthetic replay packs that carry production
sign-off authority. The packs are synthetic; real-hardware capture is
promoted later.

Every pack is generated deterministically from a frozen tuple of
:class:`SoakScenarioSpec` instances. Identical specs produce
byte-identical record sequences via :meth:`GraspAttemptRecord.to_dict`,
so the JSONL payloads can be checked into source control and hashed for
the manifest contract.

The manifest is the binding contract between the canonical packs and
any downstream replay/KPI/SLO/baseline-report tooling. It records:

* ``manifest_version``: bumped on any contract change (record shape,
  generator change, audit policy change).
* ``extra_telemetry_version``: snapshot of
  :data:`src.robot.grasping.replay.telemetry_catalog.EXTRA_TELEMETRY_VERSION`
  at generation time.
* ``label_policy``: how each record's success/failure label is
  derived (here: directly from ``final_outcome``).
* ``packs``: per-pack entry with name, file path (repo-relative),
  capability group, label policy, record count, sha256 of the JSONL
  bytes, and the deterministic generator spec digest.

The packs cover four scenes:

1. ``replay_easy_canonical_v1.jsonl``: easy-mode signoff pack.
2. ``replay_dense_canonical_v1.jsonl``: dense-mode signoff pack.
3. ``replay_drift_synthetic_v1.jsonl``: drift-injection pack; the
   telemetry shape is locked and meaningful ``drift_*`` values are
   added by the enrichment pass.
4. ``replay_ood_synthetic_v1.jsonl``: OOD-injection pack; the
   telemetry shape is locked and ``ood_flagged`` is set by the
   enrichment pass.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from src.robot.grasping.constants import (
    REPLAY_CANONICAL_DATASETS_LOG_FILE,
    create_grasping_logger,
)
from src.robot.grasping.telemetry.outcome_logging import (
    GraspAttemptRecord,
    iter_jsonl,
)
from src.robot.grasping.replay.soak import (
    SoakScenarioSpec,
    generate_soak_records,
)
from src.robot.grasping.replay.telemetry_catalog import (
    EXTRA_TELEMETRY_VERSION,
)


#: Regenerating a pack rewrites a file whose sha is a signed-off contract, so every
#: write is recorded with what it wrote. Four packs per run, not a hot loop.
logger = create_grasping_logger(
    "CanonicalDatasets", REPLAY_CANONICAL_DATASETS_LOG_FILE
)


#: Bump only when the canonical-pack contract changes (record shape,
#: generator semantics, or audit policy). Do not bump for pure
#: dataset re-generations that produce identical bytes.
MANIFEST_VERSION: int = 1


@dataclass(frozen=True, slots=True)
class CanonicalPackSpec:
    """Immutable declaration of a single canonical pack."""

    name: str
    relative_path: str  # relative to the repo root
    capability_group: str
    label_policy: str
    scenarios: tuple[SoakScenarioSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.relative_path, str) or not self.relative_path:
            raise ValueError("relative_path must be a non-empty string")
        if not self.scenarios:
            raise ValueError("scenarios must not be empty")

    def generate_records(self) -> tuple[GraspAttemptRecord, ...]:
        """Materialise the deterministic record tuple for this pack."""

        out: list[GraspAttemptRecord] = []
        for scenario in self.scenarios:
            out.extend(generate_soak_records(scenario))
        records = tuple(out)
        # Drift/OOD packs get an additive post-process pass that
        # enriches each record's ``extra`` dict with watchdog labels
        # and signals (drift_label, drift_calibration_residual_mm, ...,
        # ood_label, ood_score). The base outcome+telemetry shape
        # produced by the soak generator is left untouched so the
        # telemetry-catalog audit still passes.
        if self.name == "replay_drift_synthetic_v1":
            records = _enrich_drift_pack(records)
        elif self.name == "replay_ood_synthetic_v1":
            records = _enrich_ood_pack(records)
        # Every canonical pack carries per-stage latency telemetry
        # (``decision_latency_ms`` / ``ranking_latency_ms`` and, when
        # applicable, ``fusion_latency_ms``) plus an
        # ``attempt_wall_time_s`` summary. Easy packs omit
        # ``fusion_latency_ms``; dense packs always emit it.
        records = _enrich_latency_pack(records, pack_name=self.name)
        return records


#: Deterministic seeds for the enrichment passes. Distinct from the
#: scenario seeds so the labels/signals do not collide with the base
#: soak outcome RNG stream.
_DRIFT_ENRICH_SEED: int = 3_001_009
_OOD_ENRICH_SEED: int = 4_001_009

#: Drift-severity values matching :class:`DriftSeverity` (lowercase).
#: Mirrored here as a frozenset for readers: this module never imports
#: the enum. Nothing reads this constant; :func:`_enrich_drift_pack`
#: assigns the severity strings directly.
_DRIFT_SEVERITY_VALUES: frozenset[str] = frozenset(
    {"none", "low", "moderate", "high", "severe"}
)


def _enrich_drift_pack(
    records: tuple[GraspAttemptRecord, ...],
) -> tuple[GraspAttemptRecord, ...]:
    """Add drift labels + signals: a stable first half and a degraded second half whose residuals ramp moderate to high (ground truth for the watchdog evaluator)."""

    n = len(records)
    half = n // 2
    rng = random.Random(_DRIFT_ENRICH_SEED)
    enriched: list[GraspAttemptRecord] = []
    for i, rec in enumerate(records):
        in_degraded = i >= half
        if in_degraded:
            # Ramp ``t`` from 0.0 to ~1.0 across the degraded segment.
            t = (i - half) / max(1, n - half - 1)
            cal_residual_mm = 1.5 + 6.5 * t          # 1.5 -> 8.0 mm
            poo_delta_mm = 3.0 + 9.0 * t              # 3.0 -> 12.0 mm
            hand_eye_residual_mm = 2.0 + 5.0 * t      # 2.0 -> 7.0 mm
            verification_residual_mm = 1.0 + 4.0 * t  # 1.0 -> 5.0 mm
            depth_conf = 0.70 - 0.15 * t              # 0.70 -> 0.55
            fail_closed = rng.random() < 0.5
            drift_label = True
            drift_severity = "high" if t > 0.5 else "moderate"
            degraded_mode_active = True
        else:
            cal_residual_mm = rng.uniform(0.20, 0.40)
            poo_delta_mm = rng.uniform(0.30, 0.70)
            hand_eye_residual_mm = rng.uniform(0.40, 0.80)
            verification_residual_mm = rng.uniform(0.20, 0.50)
            depth_conf = rng.uniform(0.85, 0.95)
            fail_closed = rng.random() < 0.05
            drift_label = False
            drift_severity = "none"
            degraded_mode_active = False
        new_extra: dict[str, object] = dict(rec.extra)
        new_extra.update(
            {
                # Ground-truth label for KPI gating.
                "drift_label": bool(drift_label),
                # Per-attempt signal fields consumed by the watchdog
                # evaluator's monitors.
                "drift_calibration_residual_mm": float(cal_residual_mm),
                "drift_predicted_observed_delta_mm": float(poo_delta_mm),
                "drift_hand_eye_residual_mm": float(hand_eye_residual_mm),
                "drift_verification_residual_mm": float(
                    verification_residual_mm
                ),
                "drift_depth_confidence_mean": float(depth_conf),
                "drift_fail_closed": bool(fail_closed),
                # Telemetry catalog triple. ``drift_severity`` is one
                # of the locked ladder values; ``ood_flagged`` is always
                # False on the drift pack.
                "drift_severity": drift_severity,
                "ood_flagged": False,
                "degraded_mode_active": bool(degraded_mode_active),
            }
        )
        enriched.append(replace(rec, extra=new_extra))
    return tuple(enriched)


def _enrich_ood_pack(
    records: tuple[GraspAttemptRecord, ...],
) -> tuple[GraspAttemptRecord, ...]:
    """Add OOD labels + signals: an in-distribution first half (ood_score~=0.875) and an OOD second half (ood_score~=0.10); ground truth for the watchdog evaluator."""

    n = len(records)
    half = n // 2
    rng = random.Random(_OOD_ENRICH_SEED)
    enriched: list[GraspAttemptRecord] = []
    for i, rec in enumerate(records):
        in_ood = i >= half
        if in_ood:
            ood_score = rng.uniform(0.05, 0.15)
            ood_label = True
            ood_flagged = True
            drift_severity = "none"
            degraded_mode_active = True
        else:
            ood_score = rng.uniform(0.80, 0.95)
            ood_label = False
            ood_flagged = False
            drift_severity = "none"
            degraded_mode_active = False
        new_extra: dict[str, object] = dict(rec.extra)
        new_extra.update(
            {
                "ood_label": bool(ood_label),
                "ood_score": float(ood_score),
                # OOD pack does not vary drift signals; keep them
                # nominal so the watchdog evaluator's drift monitors
                # report NONE for every record in this pack.
                "drift_severity": drift_severity,
                "ood_flagged": bool(ood_flagged),
                "degraded_mode_active": bool(degraded_mode_active),
            }
        )
        enriched.append(replace(rec, extra=new_extra))
    return tuple(enriched)


#: Deterministic seed for the latency enrichment pass. Distinct
#: from the drift/OOD seeds so the latency stream cannot collide
#: with the watchdog labels.
_LATENCY_ENRICH_SEED: int = 5_001_009

#: SLO budgets in milliseconds. Mirrored from
#: :class:`GraspingPerformanceConfig` defaults; this module never
#: imports the schema. Nothing reads these constants: the gauss
#: parameters in :func:`_enrich_latency_pack` are hand-sized to stay
#: under them, so changing a budget here does not resize the draws.
_DECISION_SLO_MS: float = 60.0
_RANKING_SLO_MS: float = 80.0
_FUSION_SLO_MS: float = 220.0


def _enrich_latency_pack(
    records: tuple[GraspAttemptRecord, ...],
    *,
    pack_name: str,
) -> tuple[GraspAttemptRecord, ...]:
    """Add synthetic-but-seeded per-stage latencies (byte-stable JSONL) sized under the SLO budgets; easy packs omit ``fusion_latency_ms``, dense always emit it."""

    is_easy = "easy" in pack_name
    rng = random.Random(_LATENCY_ENRICH_SEED)
    enriched: list[GraspAttemptRecord] = []
    for rec in records:
        # Decision: normal(20 ms, sigma 6 ms), floored at 1 ms; p95 well under 60.
        decision_ms = max(1.0, rng.gauss(20.0, 6.0))
        # Ranking: ~30 ms mean, slightly heavier tail.
        ranking_ms = max(1.0, rng.gauss(30.0, 10.0))
        if is_easy:
            fusion_ms: float | None = None
        else:
            # Fusion only meaningful when dense/multi-view runs.
            fusion_ms = max(1.0, rng.gauss(80.0, 25.0))
        # Wall time = sum of stage times + small overhead (ms to s).
        overhead_ms = rng.uniform(5.0, 15.0)
        total_ms = decision_ms + ranking_ms + overhead_ms
        if fusion_ms is not None:
            total_ms += fusion_ms
        wall_time_s = total_ms / 1000.0
        new_extra: dict[str, object] = dict(rec.extra)
        new_extra["decision_latency_ms"] = float(decision_ms)
        new_extra["ranking_latency_ms"] = float(ranking_ms)
        if fusion_ms is not None:
            new_extra["fusion_latency_ms"] = float(fusion_ms)
        new_extra["attempt_wall_time_s"] = float(wall_time_s)
        enriched.append(replace(rec, extra=new_extra))
    return tuple(enriched)


def _scenario_digest(scenario: SoakScenarioSpec) -> dict[str, object]:
    """Return a stable, JSON-serialisable digest of a scenario spec."""

    return {
        "name": scenario.name,
        "mode": scenario.mode,
        "attempts": int(scenario.attempts),
        "failure_class_weights": {
            k: float(v) for k, v in scenario.failure_class_weights.items()
        },
        "recovery_success_rate": float(scenario.recovery_success_rate),
        "cycle_time_mean_s": float(scenario.cycle_time_mean_s),
        "cycle_time_jitter_s": float(scenario.cycle_time_jitter_s),
        "seed": int(scenario.seed),
    }


CANONICAL_PACKS: tuple[CanonicalPackSpec, ...] = (
    CanonicalPackSpec(
        name="replay_easy_canonical_v1",
        relative_path="tests/data/replay/replay_easy_canonical_v1.jsonl",
        capability_group="baseline",
        label_policy=(
            "success := final_outcome == 'succeeded'; "
            "no auxiliary verification overrides at the baseline."
        ),
        scenarios=(
            SoakScenarioSpec(
                name="easy_canonical",
                mode="easy",
                attempts=400,
                failure_class_weights={
                    "succeeded": 197.0,
                    "no_valid_grasp": 2.0,
                    "no_target": 1.0,
                },
                recovery_success_rate=0.0,
                cycle_time_mean_s=1.5,
                cycle_time_jitter_s=0.2,
                seed=1001,
            ),
        ),
    ),
    CanonicalPackSpec(
        name="replay_dense_canonical_v1",
        relative_path="tests/data/replay/replay_dense_canonical_v1.jsonl",
        capability_group="baseline",
        label_policy=(
            "success := final_outcome == 'succeeded'; "
            "dense-recovery denominator follows KPI contract."
        ),
        scenarios=(
            SoakScenarioSpec(
                name="dense_canonical_clutter",
                mode="dense_clutter",
                attempts=400,
                failure_class_weights={
                    "succeeded": 360.0,
                    "execution_failed": 18.0,
                    "no_valid_grasp": 14.0,
                    "recovery_exhausted": 4.0,
                    "decision_fail_closed": 4.0,
                },
                recovery_success_rate=0.6,
                cycle_time_mean_s=3.5,
                cycle_time_jitter_s=0.7,
                seed=2001,
            ),
            SoakScenarioSpec(
                name="dense_canonical_autonomous",
                mode="dense_autonomous",
                attempts=200,
                failure_class_weights={
                    "succeeded": 175.0,
                    "execution_failed": 10.0,
                    "no_valid_grasp": 8.0,
                    "recovery_exhausted": 4.0,
                    "decision_fail_closed": 3.0,
                },
                recovery_success_rate=0.55,
                cycle_time_mean_s=3.8,
                cycle_time_jitter_s=0.8,
                seed=2002,
            ),
        ),
    ),
    CanonicalPackSpec(
        name="replay_drift_synthetic_v1",
        relative_path="tests/data/replay/replay_drift_synthetic_v1.jsonl",
        capability_group="baseline",
        label_policy=(
            "drift_label := extra.drift_label (bool); first half stable "
            "(drift_label=False, low residuals), second half degraded "
            "(drift_label=True, ramping calibration / predicted-observed "
            "/ hand-eye / verification residuals + lower depth confidence). "
            "Watchdog KPI gate: precision >= 0.90, recall >= 0.85."
        ),
        scenarios=(
            SoakScenarioSpec(
                name="drift_baseline",
                mode="auto",
                attempts=120,
                failure_class_weights={
                    "succeeded": 90.0,
                    "execution_failed": 15.0,
                    "decision_fail_closed": 10.0,
                    "verification_failed": 5.0,
                },
                recovery_success_rate=0.4,
                cycle_time_mean_s=2.6,
                cycle_time_jitter_s=0.5,
                seed=3001,
            ),
        ),
    ),
    CanonicalPackSpec(
        name="replay_ood_synthetic_v1",
        relative_path="tests/data/replay/replay_ood_synthetic_v1.jsonl",
        capability_group="baseline",
        label_policy=(
            "ood_label := extra.ood_label (bool); first half in-distribution "
            "(ood_label=False, ood_score ~= 0.85), second half perturbed "
            "(ood_label=True, ood_score ~= 0.10). Watchdog KPI gate "
            "precision >= 0.90, recall >= 0.80."
        ),
        scenarios=(
            SoakScenarioSpec(
                name="ood_baseline",
                mode="auto",
                attempts=120,
                failure_class_weights={
                    "succeeded": 60.0,
                    "no_valid_grasp": 30.0,
                    "decision_fail_closed": 20.0,
                    "execution_failed": 10.0,
                },
                recovery_success_rate=0.35,
                cycle_time_mean_s=2.7,
                cycle_time_jitter_s=0.5,
                seed=4001,
            ),
        ),
    ),
)


def find_pack(name: str) -> CanonicalPackSpec:
    """Return the canonical pack spec with the given name."""

    for pack in CANONICAL_PACKS:
        if pack.name == name:
            return pack
    raise KeyError(f"unknown canonical pack: {name!r}")


def render_pack_jsonl(pack: CanonicalPackSpec) -> str:
    """Return the byte-stable JSONL payload for ``pack`` (compact records, trailing newline)."""

    lines = [record.to_json() for record in pack.generate_records()]
    return "\n".join(lines) + ("\n" if lines else "")


def sha256_hex(data: str | bytes) -> str:
    """Return the lowercase hex sha256 of ``data``."""

    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def write_pack(pack: CanonicalPackSpec, repo_root: Path) -> Path:
    """Render and write ``pack`` to disk, returning the absolute path."""

    payload = render_pack_jsonl(pack)
    out_path = (repo_root / pack.relative_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload, encoding="utf-8")
    # These packs carry production sign-off authority and are checked in, so the
    # sha of what was written is logged at the moment of writing; that is the
    # number the manifest contract is later compared against.
    logger.info(
        "Wrote canonical pack %s to %s (%d bytes, sha256 %s)",
        pack.name,
        out_path,
        len(payload.encode("utf-8")),
        sha256_hex(payload)[:16],
    )
    return out_path


def build_manifest(
    repo_root: Path,
    packs: Sequence[CanonicalPackSpec] = CANONICAL_PACKS,
) -> dict[str, object]:
    """Return a JSON-safe manifest describing the on-disk packs (sha256 read back from disk so a stale file surfaces immediately)."""

    entries: list[dict[str, object]] = []
    for pack in packs:
        abs_path = (repo_root / pack.relative_path).resolve()
        if not abs_path.exists():
            raise FileNotFoundError(
                f"canonical pack missing on disk: {abs_path}"
            )
        raw = abs_path.read_bytes()
        records = tuple(iter_jsonl(abs_path))
        entries.append(
            {
                "name": pack.name,
                "path": pack.relative_path,
                "capability_group": pack.capability_group,
                "label_policy": pack.label_policy,
                "record_count": len(records),
                "sha256": sha256_hex(raw),
                "generator": {
                    "module": "src.robot.grasping.replay.soak",
                    "callable": "generate_soak_records",
                    "scenarios": [
                        _scenario_digest(s) for s in pack.scenarios
                    ],
                },
            }
        )
    return {
        "manifest_version": int(MANIFEST_VERSION),
        "extra_telemetry_version": int(EXTRA_TELEMETRY_VERSION),
        "label_policy_global": (
            "label := (final_outcome == 'succeeded'); "
            "Extra overrides may add audit-time corrections via "
            "extra.verification_failed_after_success, mirroring the "
            "locked KPI semantics."
        ),
        "packs": entries,
    }


def write_manifest(
    repo_root: Path,
    packs: Sequence[CanonicalPackSpec] = CANONICAL_PACKS,
    relative_path: str = "tests/data/replay/MANIFEST.json",
) -> Path:
    """Write the manifest JSON next to the canonical packs."""

    manifest = build_manifest(repo_root, packs=packs)
    out_path = (repo_root / relative_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    out_path.write_text(body, encoding="utf-8")
    logger.info(
        "Wrote pack manifest v%d for %d pack(s) to %s (%d bytes)",
        int(MANIFEST_VERSION),
        len(packs),
        out_path,
        len(body.encode("utf-8")),
    )
    return out_path


def regenerate_all(
    repo_root: Path,
    packs: Sequence[CanonicalPackSpec] = CANONICAL_PACKS,
) -> dict[str, Path]:
    """Regenerate every canonical pack plus the manifest on disk; return ``pack.name -> path`` (manifest under ``"__manifest__"``)."""

    written: dict[str, Path] = {}
    for pack in packs:
        written[pack.name] = write_pack(pack, repo_root)
    written["__manifest__"] = write_manifest(repo_root, packs=packs)
    return written


def repo_root_from_module() -> Path:
    """Return the repository root inferred from this module's location."""

    return Path(__file__).resolve().parents[4]
