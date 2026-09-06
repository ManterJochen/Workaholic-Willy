"""Assembling a policy training set from records, and asking whether it leaked.

`build_dataset` returns a manifest and says nothing about whether there is anything to learn. A
corpus with no success/fail pair, or with every feature reading 0.0, builds a perfectly valid
manifest and then trains a converged model over nothing, so :class:`PolicyDataset` reads the
records a second time and reports readiness beside the manifest.

Exit code 3 is not a refusal. It means the split files were written and the leakage audit then
failed, so those splits exist on disk; the manifest is still in memory, because
:meth:`PolicyDatasetReport.write` is what puts it there. A caller treating 3 as "it did not
happen" leaves a corpus lying around that they believe was never created.

:class:`LeakageAudit` stands alone because it is the only lever that carries `--strict-leakage`.
Strict counts a skipped audit as a failure, so it is the only thing here that can say a corpus's
leakage check was partly vacuous; without the lever the same manifest passes.

`build()` is not side-effect free and does not pretend to be. `build_dataset` writes the split
JSONLs because it hashes them. Only the manifest write is deferred, to
:meth:`PolicyDatasetReport.write`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from src.contracts import UNSET, Maybe, chosen

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from .leakage import LeakageReport
    from .runs import RecordLogReport

__all__ = [
    "PolicyDatasetReport",
    "DatasetVerdict",
    "LeakageAudit",
    "LeakageAuditReport",
    "LeakageVerdict",
    "PolicyDataset",
]


class DatasetVerdict(StrEnum):
    """How an assembly ended.

    Ratios that do not sum to 1.0 and a missing source pack are things an operator typed, not
    programming errors, so each is a verdict here rather than an exception.
    """

    BUILT = "built"
    #: The split files were written and the leakage audit then failed. Not a refusal: the splits
    #: exist on disk. The manifest write is deferred to :meth:`PolicyDatasetReport.write`; the CLI
    #: writes it before returning exit 3.
    LEAKAGE_FAILED = "leakage_failed"
    #: The split ratios do not sum to 1.0. Nothing was written.
    RATIOS_INVALID = "ratios_invalid"
    #: A named source pack does not exist. Nothing was written.
    SOURCE_MISSING = "source_missing"


class LeakageVerdict(StrEnum):
    """What a leakage audit concluded."""

    PASSED = "passed"
    #: At least one audit rejected, or, under `strict`, warned or was skipped. The skip case is why
    #: strict exists: an audit that could not run is not an audit that passed.
    FAILED = "failed"
    #: The manifest names split files that are not on disk. They are gitignored, so this is the
    #: ordinary state of a fresh clone rather than a fault.
    SPLITS_MISSING = "splits_missing"
    #: No manifest at that path.
    MANIFEST_MISSING = "manifest_missing"


@dataclass(frozen=True, slots=True)
class LeakageAuditReport:
    """The audit's own findings, plus which lever produced them."""

    verdict: LeakageVerdict
    strict: bool
    source: str
    #: The `leakage.LeakageReport`, wrapped rather than restated. It already is the findings
    #: object; a second `passed` computed here would be a second answer to one question.
    findings: "LeakageReport | None" = None
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict is LeakageVerdict.PASSED

    @property
    def exit_code(self) -> int:
        """0 passed, 2 no manifest, 3 anything else. 1 is deliberately absent: missing split files
        are a verdict here, not an uncaught `FileNotFoundError`."""
        if self.verdict is LeakageVerdict.MANIFEST_MISSING:
            return 2
        return 0 if self.passed else 3

    def render(self) -> str:
        """ASCII, no trailing newline, no arguments."""
        lever = "strict" if self.strict else "default"
        head = f"  {self.verdict.value.upper()} ({lever}): {self.source}"
        if self.findings is None:
            return f"{head}\n    {self.detail}" if self.detail else head
        # Counted from the findings themselves. `counts_by_severity` is a key in `to_json()`, not
        # an attribute on the report; the list is the object and the counts are a view of it.
        counts: dict[str, int] = {}
        for finding in getattr(self.findings, "findings", ()) or ():
            severity = str(getattr(finding, "severity", "?"))
            counts[severity] = counts.get(severity, 0) + 1
        lines = [head]
        lines.extend(f"    {name:<10}{count}" for name, count in sorted(counts.items()))
        if getattr(self.findings, "has_notes", False):
            # Every NOTE in `leakage` is a skipped audit, meaning a required field was absent. A
            # note count is the number of checks that did not happen, and it reads as a pass
            # everywhere except under `strict`.
            lines.append("    a NOTE here means an audit was SKIPPED, not that it passed")
        if not self.strict and not self.passed:
            lines.append("    NOTE: this FAILED without --strict; the findings above are hard rejects")
        elif self.strict and not self.passed:
            # A skipped audit reads as a pass everywhere else in this repository, so the strict
            # path states the difference.
            lines.append(
                "    strict counts a WARNED or SKIPPED audit as a failure; an audit that could not"
            )
            lines.append("    run is not an audit that passed.")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """A view over the findings object's own serialisation, never a second one."""
        return {
            "verdict": self.verdict.value,
            "strict": self.strict,
            "source": self.source,
            "passed": self.passed,
            "exit_code": self.exit_code,
            "detail": self.detail,
            "findings": self.findings.to_json() if self.findings is not None else None,
        }


@dataclass(frozen=True, slots=True)
class LeakageAudit:
    """Did this dataset's splits share anything they should not?

        from src.robot.grasping.rl.datasets import LeakageAudit

        report = LeakageAudit.from_manifest(
            "docs/baselines/rl_datasets/<dataset_id>.json", repo_root=root, strict=True
        ).audit()
        print(report.render())

    Separate from `PolicyDataset`: they share `run_leakage_audits` and exit code 3, but they take
    different inputs (splits in memory against a manifest on disk) and only this one carries
    `strict`.
    """

    splits: "Mapping[str, Sequence[Mapping[str, Any]]] | None" = None
    manifest_path: Path | None = None
    repo_root: Path | None = None
    strict: bool = False

    @classmethod
    def from_splits(
        cls,
        splits: "Mapping[str, Sequence[Mapping[str, Any]]]",
        *,
        strict: bool = False,
    ) -> "LeakageAudit":
        """Splits already in memory, from a build that has just run. The Python door."""
        return cls(splits=dict(splits), strict=strict)

    @classmethod
    def from_manifest(
        cls, manifest_path: "str | Path", *, repo_root: "str | Path", strict: bool = False
    ) -> "LeakageAudit":
        """A written manifest. Stores the path and the repo root; :meth:`audit` resolves the
        manifest into splits through ``_load_splits``, so the two doors meet at ``audit()`` rather
        than at :meth:`from_splits`.

        `repo_root` is required, because the manifest stores split paths relative to it. A default
        here would resolve a customer's manifest against this repository's root.
        """
        return cls(
            manifest_path=Path(manifest_path), repo_root=Path(repo_root), strict=strict
        )

    def audit(self) -> LeakageAuditReport:
        """Run it. Reads; writes nothing."""
        from .leakage import run_leakage_audits  # noqa: PLC0415

        splits = self.splits
        source = str(self.manifest_path) if self.manifest_path is not None else "in-memory splits"
        if splits is None:
            resolved = self._load_splits()
            if isinstance(resolved, LeakageAuditReport):
                return resolved
            splits = resolved
        findings = run_leakage_audits(dict(splits), strict=self.strict)
        return LeakageAuditReport(
            verdict=LeakageVerdict.PASSED if findings.passed else LeakageVerdict.FAILED,
            strict=self.strict,
            source=source,
            findings=findings,
        )

    def _load_splits(self) -> "dict[str, Any] | LeakageAuditReport":
        import json  # noqa: PLC0415

        from .dataset import SPLIT_NAMES  # noqa: PLC0415
        from ._io import load_jsonl  # noqa: PLC0415

        assert self.manifest_path is not None and self.repo_root is not None
        source = str(self.manifest_path)
        path = (self.repo_root / self.manifest_path).resolve()
        if not path.is_file():
            return LeakageAuditReport(
                verdict=LeakageVerdict.MANIFEST_MISSING, strict=self.strict, source=source,
                detail=f"no manifest at {path}",
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        splits: dict[str, Any] = {}
        for name in SPLIT_NAMES:
            rel = payload["split_files"][name]["jsonl"]
            split_path = self.repo_root / rel
            if not split_path.is_file():
                # A verdict, not a traceback. The split files are gitignored, so a fresh clone has
                # the manifest and not the data, which is the ordinary state.
                return LeakageAuditReport(
                    verdict=LeakageVerdict.SPLITS_MISSING, strict=self.strict, source=source,
                    detail=(
                        f"the manifest names {rel}, which is not on disk. Split files are "
                        f"gitignored; rebuild the dataset or fetch them."
                    ),
                )
            splits[name] = load_jsonl(split_path)
        return splits


@dataclass(frozen=True, slots=True)
class PolicyDatasetReport:
    """What an assembly produced, and whether it is worth training on."""

    verdict: DatasetVerdict
    dataset_id: str
    #: The `DatasetManifest`, verbatim. `None` when nothing was built.
    manifest: Any = None
    leakage: LeakageAuditReport | None = None
    #: A second read over the same records. `build_dataset` indexes and splits; it never looks at
    #: whether there is anything to learn, and a corpus with no success/fail pair trains a converged
    #: model over nothing.
    readiness: "RecordLogReport | None" = None
    detail: str = ""
    #: Where the manifest went, once :meth:`write` has been called.
    manifest_path: str = ""

    @property
    def built(self) -> bool:
        return self.verdict in (DatasetVerdict.BUILT, DatasetVerdict.LEAKAGE_FAILED)

    @property
    def artifacts_exist(self) -> bool:
        """True for `LEAKAGE_FAILED` too, and that is the point of having it. Exit 3 means the
        splits were written and the audit then failed; a caller who reads 3 as "nothing happened"
        leaves a corpus on disk they believe was never created."""
        return self.built

    @property
    def exit_code(self) -> int:
        """0 built, 2 bad arguments, 3 written-then-leaked, 1 a refusal before anything was written."""
        return {
            DatasetVerdict.BUILT: 0,
            DatasetVerdict.LEAKAGE_FAILED: 3,
            DatasetVerdict.RATIOS_INVALID: 2,
            DatasetVerdict.SOURCE_MISSING: 1,
        }[self.verdict]

    def render(self) -> str:
        """ASCII, no trailing newline, no arguments."""
        if not self.built:
            return f"  {self.verdict.value.upper()}: {self.detail}"
        m = self.manifest
        lines = [
            f"  {self.verdict.value.upper()}: {self.dataset_id}, "
            f"{getattr(m, 'record_count', 0)} record(s)"
        ]
        for name, count in sorted(dict(getattr(m, "split_counts", {}) or {}).items()):
            lines.append(f"    {name:<8}{count}")
        if self.leakage is not None:
            lines.append(f"    leakage: {self.leakage.verdict.value}")
        if self.readiness is not None:
            # Reported, never gating. A dataset that is not trainable is still a dataset, and the
            # build is not the place to decide what someone may do with it, so it has to be loud.
            lines.append(f"    {self.readiness.summary()}")
        if self.verdict is DatasetVerdict.LEAKAGE_FAILED:
            lines.append("    NOTE: THE SPLITS AND MANIFEST WERE WRITTEN. This is not a refusal.")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        m = self.manifest
        return {
            "verdict": self.verdict.value,
            "dataset_id": self.dataset_id,
            "exit_code": self.exit_code,
            "artifacts_exist": self.artifacts_exist,
            "manifest_path": self.manifest_path,
            "record_count": getattr(m, "record_count", None),
            "split_counts": dict(getattr(m, "split_counts", {}) or {}),
            "leakage": self.leakage.to_dict() if self.leakage is not None else None,
            "trainable": getattr(self.readiness, "trainable", None),
            "detail": self.detail,
        }

    def write(self, path: "str | Path") -> Path:
        """Write the manifest. Separate from `build()`, because writing where the committed
        manifests live is a decision rather than a step."""
        from .dataset import write_manifest  # noqa: PLC0415

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_manifest(self.manifest, out)
        return out


@dataclass(frozen=True, slots=True)
class PolicyDataset:
    """A training set assembled from record packs.

        from src.robot.grasping.rl.datasets import PolicyDataset

        report = PolicyDataset.from_sources(repo_root=root, dataset_id="v2").build()
        print(report.render())
        if report.readiness and not report.readiness.trainable:
            ...   # a valid manifest over a corpus with nothing to learn

    `DatasetBuild` in `datagen/api.py` is a different noun: it renders and labels scenes, while
    this assembles rows that already exist.
    """

    repo_root: Path
    dataset_id: str
    canonical_paths: "Maybe[Sequence[str]]" = UNSET
    extra_paths: "Maybe[Sequence[str]]" = UNSET
    seed: "Maybe[int]" = UNSET
    ratios: "Maybe[Sequence[float]]" = UNSET
    emit_parquet: "Maybe[bool]" = UNSET
    dataset_origin: "Maybe[str]" = UNSET
    fitness_note: "Maybe[str]" = UNSET
    strict_leakage: "Maybe[bool]" = UNSET
    group_split_field: "Maybe[str]" = UNSET
    _extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_sources(
        cls,
        *,
        repo_root: "str | Path",
        dataset_id: str,
        canonical_paths: "Maybe[Sequence[str]]" = UNSET,
        extra_paths: "Maybe[Sequence[str]]" = UNSET,
        seed: "Maybe[int]" = UNSET,
        ratios: "Maybe[Sequence[float]]" = UNSET,
        emit_parquet: "Maybe[bool]" = UNSET,
        dataset_origin: "Maybe[str]" = UNSET,
        fitness_note: "Maybe[str]" = UNSET,
        strict_leakage: "Maybe[bool]" = UNSET,
        group_split_field: "Maybe[str]" = UNSET,
    ) -> "PolicyDataset":
        """Named record packs. The plain-Python door.

        Nothing is defaulted here. `build_dataset` declares its own seed, ratios and canonical
        source list, and a copy of any of them in this signature would be a second declaration of
        one fact, which is how two builds come to write a different artifact sha256.
        """
        return cls(
            repo_root=Path(repo_root),
            dataset_id=dataset_id,
            canonical_paths=canonical_paths,
            extra_paths=extra_paths,
            seed=seed,
            ratios=ratios,
            emit_parquet=emit_parquet,
            dataset_origin=dataset_origin,
            fitness_note=fitness_note,
            strict_leakage=strict_leakage,
            group_split_field=group_split_field,
        )

    def build(self) -> PolicyDatasetReport:
        """Index, split, hash and audit.

        This writes. `build_dataset` emits the split JSONLs because it hashes them, so a refusal
        after that point leaves them on disk. Only the manifest write is deferred, to
        :meth:`PolicyDatasetReport.write`.
        """
        from .dataset import build_dataset  # noqa: PLC0415

        options: dict[str, Any] = {}
        for name, value in (
            ("canonical_paths", self.canonical_paths),
            ("extra_paths", self.extra_paths),
            ("seed", self.seed),
            ("ratios", self.ratios),
            ("emit_parquet", self.emit_parquet),
            ("dataset_origin", self.dataset_origin),
            ("fitness_note", self.fitness_note),
            ("strict_leakage", self.strict_leakage),
            ("group_split_field", self.group_split_field),
        ):
            if chosen(value):
                options[name] = tuple(value) if isinstance(value, (list, tuple)) else value

        if chosen(self.ratios) and abs(sum(self.ratios) - 1.0) > 1e-9:
            # A verdict before anything is written. Ratios that do not sum to 1.0 are a number an
            # operator typed, not a programming error, so they never reach `build_dataset`.
            return PolicyDatasetReport(
                verdict=DatasetVerdict.RATIOS_INVALID, dataset_id=self.dataset_id,
                detail=f"ratios {tuple(self.ratios)} sum to {sum(self.ratios)}, not 1.0",
            )

        try:
            manifest = build_dataset(
                repo_root=self.repo_root, dataset_id=self.dataset_id, **options
            )
        except FileNotFoundError as exc:
            return PolicyDatasetReport(
                verdict=DatasetVerdict.SOURCE_MISSING, dataset_id=self.dataset_id,
                detail=f"a named source pack does not exist: {exc}",
            )

        payload = manifest.leakage or {}
        passed = bool(payload.get("passed", False))
        leakage = LeakageAuditReport(
            verdict=LeakageVerdict.PASSED if passed else LeakageVerdict.FAILED,
            strict=bool(chosen(self.strict_leakage) and self.strict_leakage),
            source=f"embedded in {self.dataset_id}",
        )
        return PolicyDatasetReport(
            verdict=DatasetVerdict.BUILT if passed else DatasetVerdict.LEAKAGE_FAILED,
            dataset_id=self.dataset_id,
            manifest=manifest,
            leakage=leakage,
            readiness=self._readiness(options),
        )

    def _readiness(self, options: "Mapping[str, Any]") -> "RecordLogReport | None":
        """The second read, delegating to `RecordLogCheck` rather than re-deriving its verdict.

        One implementation of "is there anything to learn here", not two: `RecordLogCheck` answers
        it and renders it, so calling `assess_records` and `format_readiness` here would be the
        same computation with a second presentation.
        """
        from ._io import load_jsonl  # noqa: PLC0415
        from .runs import RecordLogCheck  # noqa: PLC0415

        sources = list(options.get("canonical_paths") or ()) + list(options.get("extra_paths") or ())
        rows: list[Any] = []
        for source in sources:
            path = self.repo_root / source
            if path.is_file():
                rows.extend(load_jsonl(path))
        if not rows:
            return None
        try:
            return RecordLogCheck.from_records(rows).assess()
        except Exception:  # noqa: BLE001 (a diagnosis must never fail the build it is diagnosing)
            return None
