"""Offline dataset builder.

What a build emits:

* JSONL always, one record per line.
* Parquet only when ``pyarrow`` imports successfully; otherwise the
  manifest records ``parquet=null`` and the build prints a hint,
  which keeps ``requirements.txt`` lean.
* Splits of 60/20/20 train/val/test, stratified by failure-taxonomy
  outcome class.
* Sources combining canonical bootstrap packs with caller-supplied
  extra paths; every input file is sha256-hashed for the manifest.
* The manifest goes to
  ``docs/baselines/rl_datasets/<dataset_id>.json`` and the split files
  to ``logs/rl/datasets/<dataset_id>/splits/``; the ``logs/`` tree is
  gitignored, so only the manifest pins truth.

This module is build-time only. It imports no runtime robot/grasping
code path and never mutates configuration. The one import it does make
into the grasping tree is ``replay.failure_taxonomy`` (offline, never
imported back into the pick path), so the dataset failure axis and the
taxonomy report cannot disagree about the same record.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

from src.robot.grasping.constants import (
    RL_DATASET_LOG_FILE,
    create_grasping_logger,
)

from ._io import load_jsonl
from .honesty import INPUT_CANONICAL_BOOTSTRAP, REWARD_NOT_APPLICABLE_RAW
from .leakage import run_leakage_audits

#: The manifest records only what the build believed, so the log carries what it
#: cannot: which source files were read, the class balance that came out, and a
#: parquet emit skipped because pyarrow was absent.
logger = create_grasping_logger("RLDataset", RL_DATASET_LOG_FILE)

#: Manifest schema version. v2 carries dataset_origin / fitness_note /
#: reward stamps and embeds the leakage audit in the build.
RL_DATASET_SCHEMA_VERSION: int = 2

#: Honesty stamp for the dataset manifest (raw records carry no reward).
MANIFEST_REWARD_INTERPRETATION: str = (
    "this manifest indexes RAW GraspAttemptRecord entries; it carries NO reward. Downstream policy "
    "artifacts define their own reward models + carry their own reward_model stamps. See "
    "dataset_origin + the embedded leakage.interpretation for provenance + what the split separation proves."
)

#: Train / val / test ratios. Sum must equal 1.0.
SPLIT_RATIOS: tuple[float, float, float] = (0.60, 0.20, 0.20)

#: 7-class outcome taxonomy. Stratification operates over this set plus
#: the explicit ``unclassified`` bucket, so a record that matches no
#: class still partitions deterministically.
OUTCOME_CLASS_SUCCESS: str = "success"
OUTCOME_CLASS_SLIP: str = "slip_after_grasp"
OUTCOME_CLASS_EMPTY_AIR: str = "empty_air_grasp"
OUTCOME_CLASS_COLLISION: str = "collision_rejection"
OUTCOME_CLASS_OCCLUSION: str = "occlusion_misread"
OUTCOME_CLASS_DEFORMABLE: str = "deformable_misclassification"
OUTCOME_CLASS_DRIFT: str = "calibration_drift_suspected"
OUTCOME_CLASS_UNCLASSIFIED: str = "unclassified"

#: Insertion order is part of the manifest contract: class
#: histograms in the manifest are emitted in this order.
OUTCOME_CLASSES: tuple[str, ...] = (
    OUTCOME_CLASS_SUCCESS,
    OUTCOME_CLASS_SLIP,
    OUTCOME_CLASS_EMPTY_AIR,
    OUTCOME_CLASS_COLLISION,
    OUTCOME_CLASS_OCCLUSION,
    OUTCOME_CLASS_DEFORMABLE,
    OUTCOME_CLASS_DRIFT,
    OUTCOME_CLASS_UNCLASSIFIED,
)

#: Failure-taxonomy tokens. They validate ``extra.expected_root_cause``
#: and ``extra.failure_taxonomy_class``, and they keep a cause the
#: classifier names but this histogram has no column for out of the
#: class balance.
_FAILURE_TOKENS: frozenset[str] = frozenset(
    {
        OUTCOME_CLASS_SLIP,
        OUTCOME_CLASS_EMPTY_AIR,
        OUTCOME_CLASS_COLLISION,
        OUTCOME_CLASS_OCCLUSION,
        OUTCOME_CLASS_DEFORMABLE,
        OUTCOME_CLASS_DRIFT,
    }
)

SPLIT_TRAIN: str = "train"
SPLIT_VAL: str = "val"
SPLIT_TEST: str = "test"
SPLIT_NAMES: tuple[str, str, str] = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)

#: Canonical bootstrap sources, relative to the repo root, used when a
#: build passes no ``canonical_paths``. These packs are not present in
#: this tree: ``collect_sources`` raises ``FileNotFoundError`` naming
#: the first absent source, which the library door reports as
#: ``DatasetVerdict.SOURCE_MISSING``.
CANONICAL_BOOTSTRAP_SOURCES: tuple[str, ...] = (
    "tests/data/replay/replay_easy_canonical_v1.jsonl",
    "tests/data/replay/replay_dense_canonical_v1.jsonl",
    "tests/data/replay/replay_failure_taxonomy_v1.jsonl",
)


# Outcome-class derivation


def derive_outcome_class(record: Mapping[str, Any]) -> str:
    """Classify one record into an outcome class.

    Precedence: ``extra.failure_taxonomy_class`` when it holds a known failure token, else
    ``success`` when ``final_outcome`` is ``succeeded``, else ``extra.expected_root_cause`` when it
    holds a known failure token, else whatever the replay taxonomy classifier makes of the record
    symptom flags, else ``unclassified``. The order is fixed, so the class is deterministic.

    The last step was missing and the two readers keyed on disjoint fields. The first three rules
    read ``extra.failure_taxonomy_class`` (stamped by the sim runners from ground truth) and
    ``extra.expected_root_cause`` (a label on the committed test pack). ``replay.failure_taxonomy``
    reads ``extra.*_evidence`` and ``final_outcome``, which is what the production serializer writes
    (``record_logging._stamp_taxonomy_evidence``). Neither set overlapped the other, so measured
    2026-09-10 the same record was ``collision_rejection`` in the taxonomy report and
    ``unclassified`` here: the failure axis of an RL dataset built from a real cell log collapsed
    into one bucket, and the stratified split had nothing to stratify.

    Borrowed, not reimplemented. A second copy of the rule table is exactly what produced this
    divergence, so the classifier is called. It is a pure function of the two fields, so nothing
    about this module build-time-only promise changes.

    The other direction stays open, deliberately. The classifier does not read
    ``failure_taxonomy_class``, so a sim-runner record is still ``unclassified`` to the taxonomy
    report while being classified here. That is not an oversight: that field is the producer own
    label, and a classifier scored on rows where the producer handed it the answer would be
    measuring the producer. The dataset may believe a label; the classifier may not.

    Order matters. ``succeeded`` is decided before the classifier is consulted. The classifier would
    also answer ``unclassified`` for a success, but only because of its own gating set; a dataset
    that let a stray flag relabel a success would be reading a symptom as an outcome.
    """

    from src.robot.grasping.replay.failure_taxonomy import classify_symptoms

    extra = record.get("extra") or {}
    if not isinstance(extra, Mapping):
        extra = {}
    tax = extra.get("failure_taxonomy_class")
    if isinstance(tax, str) and tax in _FAILURE_TOKENS:
        return tax
    if record.get("final_outcome") == "succeeded":
        return OUTCOME_CLASS_SUCCESS
    expected = extra.get("expected_root_cause")
    if isinstance(expected, str) and expected in _FAILURE_TOKENS:
        return expected
    classified = classify_symptoms(str(record.get("final_outcome") or ""), extra).primary.value
    if classified in _FAILURE_TOKENS:
        return classified
    return OUTCOME_CLASS_UNCLASSIFIED


# Hashing helpers (deterministic; sort_keys=True everywhere)


def canonical_record_hash(record: Mapping[str, Any]) -> str:
    """Stable sha256 hex of a record (sorted-key JSON)."""

    payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_file_sha256(path: Path) -> str:
    """sha256 hex of a file's bytes."""

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# Source loading


@dataclass(frozen=True)
class SourceFile:
    """Description of one input pack contributing to a dataset."""

    path: str  # repo-relative
    sha256: str
    record_count: int
    role: str  # "canonical" or "extra"


def collect_sources(
    repo_root: Path,
    canonical_paths: Sequence[str],
    extra_paths: Sequence[str],
) -> tuple[list[SourceFile], list[dict[str, Any]]]:
    """Resolve the source paths and load every record behind them.

    The returned ``SourceFile`` list holds the canonical entries first and the extras second,
    each group in caller order; that ordering is part of the determinism contract.
    """

    sources: list[SourceFile] = []
    records: list[dict[str, Any]] = []
    for role, paths in (("canonical", canonical_paths), ("extra", extra_paths)):
        for rel in paths:
            resolved = (repo_root / rel).resolve()
            if not resolved.is_file():
                raise FileNotFoundError(
                    f"dataset source missing: {rel} (resolved {resolved})"
                )
            file_records = load_jsonl(resolved)
            sources.append(
                SourceFile(
                    path=rel,
                    sha256=hash_file_sha256(resolved),
                    record_count=len(file_records),
                    role=role,
                )
            )
            records.extend(file_records)
    logger.info(
        "Collected %d record(s) from %d source file(s): %s",
        len(records),
        len(sources),
        ", ".join(f"{s.path} [{s.role}] x{s.record_count}" for s in sources),
    )
    return sources, records


# Stratified split


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    if len(ratios) != 3:
        raise ValueError(f"split ratios must be length 3, got {ratios!r}")
    total = sum(ratios)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {total!r}")
    for r in ratios:
        if r < 0.0:
            raise ValueError(f"split ratios must be non-negative, got {ratios!r}")


def stratified_split_by_outcome(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
) -> dict[str, list[dict[str, Any]]]:
    """Split records deterministically into train, val and test, stratified by outcome class.

    A class with fewer than 3 samples cannot be split three ways, so all of its records go to
    train; ``build_dataset`` records a note naming that class.
    """

    _validate_ratios(ratios)
    by_class: dict[str, list[dict[str, Any]]] = {c: [] for c in OUTCOME_CLASSES}
    for record in records:
        cls = derive_outcome_class(record)
        by_class.setdefault(cls, []).append(dict(record))

    splits: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLIT_NAMES}

    for cls in OUTCOME_CLASSES:
        items = by_class.get(cls, [])
        if not items:
            continue
        items.sort(
            key=lambda r: (
                r.get("attempt_id") or canonical_record_hash(r),
            )
        )
        if len(items) < 3:
            # Too few samples for a 3-way split; keep them in train.
            splits[SPLIT_TRAIN].extend(items)
            continue
        # Class-scoped RNG seeded deterministically from (seed, cls).
        # The hash must be process-stable: ``hash(str)`` is randomised
        # per interpreter via ``PYTHONHASHSEED`` and breaks
        # cross-process determinism. sha256 of the canonical bytes is
        # stable across machines and Python versions.
        cls_seed_bytes = hashlib.sha256(
            f"{seed}:{cls}".encode("utf-8")
        ).digest()
        cls_seed = int.from_bytes(cls_seed_bytes[:8], "big", signed=False)
        cls_rng = random.Random(cls_seed)
        cls_rng.shuffle(items)
        n = len(items)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        n_test = n - n_train - n_val
        # Floor rounding can leave val or test empty, so the loops below
        # borrow from train, the largest split at the default ratios,
        # until each holds at least 1 record.
        while n_test < 1 and n_train > 1:
            n_train -= 1
            n_test += 1
        while n_val < 1 and n_train > 1:
            n_train -= 1
            n_val += 1
        splits[SPLIT_TRAIN].extend(items[:n_train])
        splits[SPLIT_VAL].extend(items[n_train : n_train + n_val])
        splits[SPLIT_TEST].extend(items[n_train + n_val :])
        # ``cls_rng`` is scoped to this class's shuffle; the ``del`` ends that scope and the
        # name is rebound on the next iteration.
        del cls_rng
    # Stable order inside each split for downstream determinism.
    for split_name in SPLIT_NAMES:
        splits[split_name].sort(
            key=lambda r: (
                r.get("attempt_id") or canonical_record_hash(r),
            )
        )
    return splits


def group_aware_split_by_family(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    group_field: str = "scene_family_id",
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
) -> dict[str, list[dict[str, Any]]]:
    """Leakage-safe group-aware split: every record sharing a ``group_field`` value (e.g. ``scene_family_id``)
    is kept entirely in one split, so no object identity leaks between train and test and success/fail pairs
    stay within a split; records lacking the field degrade to a per-record singleton group."""

    _validate_ratios(ratios)
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        extra = record.get("extra") or {}
        candidate = extra.get(group_field) or record.get(group_field)
        gid = candidate if isinstance(candidate, str) and candidate else (
            record.get("attempt_id") or canonical_record_hash(record)
        )
        groups.setdefault(str(gid), []).append(dict(record))

    total = max(1, len(records))
    targets = {
        SPLIT_TRAIN: total * ratios[0],
        SPLIT_VAL: total * ratios[1],
        SPLIT_TEST: total * ratios[2],
    }
    counts: dict[str, int] = {s: 0 for s in SPLIT_NAMES}
    splits: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLIT_NAMES}

    def _order_key(g: str) -> tuple[int, int]:
        h = int.from_bytes(
            hashlib.sha256(f"{seed}:{g}".encode("utf-8")).digest()[:8], "big"
        )
        return (-len(groups[g]), h)

    # Largest groups first: the deficit-greedy assignment balances toward the ratios deterministically.
    for gid in sorted(groups.keys(), key=_order_key):
        members = groups[gid]
        target_split = max(
            SPLIT_NAMES, key=lambda s: (targets[s] - counts[s], -SPLIT_NAMES.index(s))
        )
        splits[target_split].extend(members)
        counts[target_split] += len(members)

    for split_name in SPLIT_NAMES:
        splits[split_name].sort(
            key=lambda r: (r.get("attempt_id") or canonical_record_hash(r),)
        )
    return splits


# Manifest


@dataclass
class DatasetManifest:
    """Self-describing manifest for one dataset build."""

    dataset_id: str
    schema_version: int
    rl_tooling_version: str
    seed: int
    ratios: tuple[float, float, float]
    sources: list[SourceFile]
    record_count: int
    split_counts: dict[str, int]
    class_counts: dict[str, int]
    class_counts_by_split: dict[str, dict[str, int]]
    split_files: dict[str, dict[str, str | None]]  # per split: {jsonl, parquet}
    split_hashes: dict[str, str]  # sha256 of canonical JSONL bytes
    parquet_emitted: bool
    parquet_skip_reason: str | None
    notes: list[str] = field(default_factory=list)
    # provenance + the embedded leakage audit
    dataset_origin: str = INPUT_CANONICAL_BOOTSTRAP
    fitness_note: str | None = None
    leakage: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "schema_version": self.schema_version,
            "reward_model": REWARD_NOT_APPLICABLE_RAW,
            "reward_interpretation": MANIFEST_REWARD_INTERPRETATION,
            "dataset_origin": self.dataset_origin,
            "fitness_note": self.fitness_note,
            "leakage": self.leakage,
            "rl_tooling_version": self.rl_tooling_version,
            "seed": self.seed,
            "ratios": list(self.ratios),
            "sources": [
                {
                    "path": s.path,
                    "sha256": s.sha256,
                    "record_count": s.record_count,
                    "role": s.role,
                }
                for s in self.sources
            ],
            "record_count": self.record_count,
            "split_counts": dict(self.split_counts),
            "class_counts": {c: int(self.class_counts.get(c, 0)) for c in OUTCOME_CLASSES},
            "class_counts_by_split": {
                split: {c: int(self.class_counts_by_split.get(split, {}).get(c, 0)) for c in OUTCOME_CLASSES}
                for split in SPLIT_NAMES
            },
            "split_files": {
                split: dict(self.split_files.get(split, {})) for split in SPLIT_NAMES
            },
            "split_hashes": dict(self.split_hashes),
            "parquet_emitted": bool(self.parquet_emitted),
            "parquet_skip_reason": self.parquet_skip_reason,
            "notes": list(self.notes),
        }


# Emission


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> str:
    """Write records to ``path`` and return sha256 of the bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            line = json.dumps(record, sort_keys=True, separators=(",", ":"))
            fh.write(line)
            fh.write("\n")
            h.update(line.encode("utf-8"))
            h.update(b"\n")
    return h.hexdigest()


def _try_write_parquet(
    path: Path, records: Sequence[Mapping[str, Any]]
) -> tuple[bool, str | None]:
    """Best-effort parquet emit. Returns ``(emitted, skip_reason)``."""

    try:  # pragma: no cover (runs only where pyarrow is installed)
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        # Degraded, not fatal: the JSONL split is still authoritative, but anyone
        # expecting a parquet mirror gets nothing and no error.
        logger.warning("Parquet emit skipped: pyarrow unavailable (%s)", exc)
        return False, f"pyarrow unavailable: {type(exc).__name__}: {exc}"
    # Records are heterogeneous nested dicts; serialise each as a
    # canonical JSON string to keep the parquet schema flat and
    # forward-compatible.
    payload = [
        {
            "attempt_id": r.get("attempt_id"),
            "mode": r.get("mode"),
            "final_outcome": r.get("final_outcome"),
            "outcome_class": derive_outcome_class(r),
            "record_json": json.dumps(r, sort_keys=True, separators=(",", ":")),
        }
        for r in records
    ]
    try:
        table = pa.Table.from_pylist(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, str(path))
    except Exception as exc:  # pragma: no cover
        logger.warning("Parquet write to %s failed: %s", path, exc)
        return False, f"parquet write failed: {type(exc).__name__}: {exc}"
    return True, None


# Public build entry point


def build_dataset(
    repo_root: Path,
    *,
    dataset_id: str,
    seed: int = 1,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
    canonical_paths: Sequence[str] = CANONICAL_BOOTSTRAP_SOURCES,
    extra_paths: Sequence[str] = (),
    splits_root: Path | None = None,
    emit_parquet: bool = True,
    dataset_origin: str = INPUT_CANONICAL_BOOTSTRAP,
    fitness_note: str | None = None,
    strict_leakage: bool = False,
    group_split_field: str | None = None,
) -> DatasetManifest:
    """Build a dataset.

    Args:
        repo_root: repo root used for resolving relative source paths.
        dataset_id: stable identifier; becomes the directory name and
            manifest file stem.
        seed: deterministic RNG seed; recorded in the manifest.
        ratios: train/val/test split ratios (must sum to 1.0).
        canonical_paths: bootstrap pack paths (default = canonical
            replay packs).
        extra_paths: caller-supplied additional sources.
        splits_root: directory where split files are written. Default
            is ``logs/rl/datasets/<dataset_id>/splits``.
        emit_parquet: when ``True`` and ``pyarrow`` is importable, a
            ``.parquet`` mirror is emitted alongside each ``.jsonl``.
        dataset_origin: provenance stamp recorded in the manifest.
        fitness_note: free-text fitness caveat recorded in the
            manifest.
        strict_leakage: when ``True``, a warned or skipped audit
            counts as a failure in the embedded leakage report, which
            is what the caller's exit code reads.
        group_split_field: when set, split by this field via
            ``group_aware_split_by_family`` instead of the per-record
            stratified split; this changes the split files.

    Returns:
        DatasetManifest. The caller is responsible for committing the
        manifest JSON to ``docs/baselines/rl_datasets/``.
    """

    # The manifest records the code version that built it. The import is local to avoid
    # a cycle at module load.
    from . import RL_TOOLING_VERSION

    _validate_ratios(ratios)
    sources, records = collect_sources(repo_root, canonical_paths, extra_paths)
    # A group-aware split (grouped by scene_family_id) keeps each object identity in one split so the
    # leakage audit passes rigidly and success/fail pairs stay within a split. ``None`` selects the
    # per-record stratified split instead.
    if group_split_field is not None:
        splits = group_aware_split_by_family(
            records, seed=seed, group_field=group_split_field, ratios=ratios
        )
    else:
        splits = stratified_split_by_outcome(records, seed=seed, ratios=ratios)

    if splits_root is None:
        splits_root = repo_root / "logs" / "rl" / "datasets" / dataset_id / "splits"

    split_files: dict[str, dict[str, str | None]] = {}
    split_hashes: dict[str, str] = {}
    class_counts: dict[str, int] = {c: 0 for c in OUTCOME_CLASSES}
    class_counts_by_split: dict[str, dict[str, int]] = {
        s: {c: 0 for c in OUTCOME_CLASSES} for s in SPLIT_NAMES
    }
    notes: list[str] = []
    parquet_any_emitted = False
    parquet_skip_reason: str | None = None

    for split_name in SPLIT_NAMES:
        split_records = splits[split_name]
        for r in split_records:
            cls = derive_outcome_class(r)
            class_counts[cls] = class_counts.get(cls, 0) + 1
            class_counts_by_split[split_name][cls] = (
                class_counts_by_split[split_name].get(cls, 0) + 1
            )
        jsonl_path = splits_root / f"{split_name}.jsonl"
        rel_jsonl = jsonl_path.relative_to(repo_root).as_posix() if jsonl_path.is_relative_to(repo_root) else str(jsonl_path)
        split_hashes[split_name] = _write_jsonl(jsonl_path, split_records)
        parquet_rel: str | None = None
        if emit_parquet:
            parquet_path = splits_root / f"{split_name}.parquet"
            ok, reason = _try_write_parquet(parquet_path, split_records)
            if ok:
                parquet_any_emitted = True
                parquet_rel = parquet_path.relative_to(repo_root).as_posix() if parquet_path.is_relative_to(repo_root) else str(parquet_path)
            else:
                if parquet_skip_reason is None:
                    parquet_skip_reason = reason
        else:
            parquet_skip_reason = "emit_parquet=False"
        split_files[split_name] = {"jsonl": rel_jsonl, "parquet": parquet_rel}

    # Sanity: classes that fell entirely into train.
    for cls in OUTCOME_CLASSES:
        in_train = class_counts_by_split[SPLIT_TRAIN].get(cls, 0)
        in_val = class_counts_by_split[SPLIT_VAL].get(cls, 0)
        in_test = class_counts_by_split[SPLIT_TEST].get(cls, 0)
        if in_train and not in_val and not in_test:
            notes.append(
                f"class {cls!r} has {in_train} sample(s); kept in train only "
                f"(< 3 samples cannot stratify)"
            )

    manifest = DatasetManifest(
        dataset_id=dataset_id,
        schema_version=RL_DATASET_SCHEMA_VERSION,
        rl_tooling_version=RL_TOOLING_VERSION,
        seed=seed,
        ratios=cast("tuple[float, float, float]", tuple(ratios)),
        sources=sources,
        record_count=len(records),
        split_counts={s: len(splits[s]) for s in SPLIT_NAMES},
        class_counts=class_counts,
        class_counts_by_split=class_counts_by_split,
        split_files=split_files,
        split_hashes=split_hashes,
        parquet_emitted=parquet_any_emitted,
        parquet_skip_reason=parquet_skip_reason,
        notes=notes,
        dataset_origin=dataset_origin,
        fitness_note=fitness_note,
    )
    # Embed the leakage audit in the manifest so a programmatic build ships with leakage data. The
    # leakage module is a stdlib-only leaf, so its top-level import creates no cycle.
    manifest.leakage = run_leakage_audits(
        {name: splits[name] for name in SPLIT_NAMES}, strict=strict_leakage
    ).to_json()
    logger.info(
        "Built dataset %r (schema v%d, seed %d): %d record(s) split %s; classes %s; "
        "origin %s%s",
        dataset_id,
        RL_DATASET_SCHEMA_VERSION,
        seed,
        len(records),
        "/".join(f"{name}={len(splits[name])}" for name in SPLIT_NAMES),
        ", ".join(f"{k}={v}" for k, v in sorted(class_counts.items()) if v),
        dataset_origin,
        "" if parquet_any_emitted else " (JSONL only)",
    )
    return manifest


def write_manifest(manifest: DatasetManifest, path: Path) -> str:
    """Write the manifest as deterministic sort-keyed JSON. Returns sha256."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest.to_json(), sort_keys=True, indent=2)
    path.write_text(payload + "\n", encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    logger.info(
        "Wrote dataset manifest %r to %s (%d bytes, sha256 %s)",
        manifest.dataset_id,
        path,
        len(payload.encode("utf-8")) + 1,
        digest[:16],
    )
    return digest


__all__ = (
    "CANONICAL_BOOTSTRAP_SOURCES",
    "DatasetManifest",
    "OUTCOME_CLASSES",
    "OUTCOME_CLASS_DEFORMABLE",
    "OUTCOME_CLASS_DRIFT",
    "OUTCOME_CLASS_EMPTY_AIR",
    "OUTCOME_CLASS_OCCLUSION",
    "OUTCOME_CLASS_SLIP",
    "OUTCOME_CLASS_SUCCESS",
    "OUTCOME_CLASS_UNCLASSIFIED",
    "OUTCOME_CLASS_COLLISION",
    "RL_DATASET_SCHEMA_VERSION",
    "SPLIT_NAMES",
    "SPLIT_RATIOS",
    "SPLIT_TEST",
    "SPLIT_TRAIN",
    "SPLIT_VAL",
    "SourceFile",
    "build_dataset",
    "canonical_record_hash",
    "collect_sources",
    "derive_outcome_class",
    "hash_file_sha256",
    "load_jsonl",
    "stratified_split_by_outcome",
    "write_manifest",
)
