r"""Judging a policy: the promotion gate, and the off-policy estimate underneath it.

`repo_root` is required and carries no default. Omitting it silently changes `dataset_hash` and
turns `dataset_paths` into backslash strings, with the same verdict and no warning. A promotion
report's whole job is to say which dataset a policy was judged against, so the question cannot be
skipped.

The exit code does not carry the verdict. `promote-policy` returns 0 on pass, on fail and on
abstain alike; only a `PromotionInputError` gives 2. That is kept deliberately, so no caller
gating on the code changes meaning; the verdict lives on the report, where a caller can branch on
it. Branching on the exit code is branching on "did it run".

The artifacts are byte-locked and these reports are views over them. `to_dict()` returns
`build_promotion_report_artifact(report)` and `ope.build_ope_report(...)` unchanged. Nothing here
re-derives a number, reorders a key or normalises a path: `_relpath` writes platform-native
separators into the report, and tidying them to posix changes the report's bytes.

Editing a string that flows into a hash-locked artifact regenerates that artifact. That is a
deliberate act, not a side effect of another change.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.contracts import UNSET, Maybe, chosen

__all__ = [
    "OPERun",
    "OPERunReport",
    "OPEVerdict",
    "PromotionGate",
    "PromotionGateReport",
    "PromotionVerdict",
]


class PromotionVerdict(StrEnum):
    """What the gate concluded. No verdict here reaches the exit code."""

    PASS = "pass"
    FAIL = "fail"
    #: The estimator could not form an opinion: too few triples, no behaviour action logged, or a
    #: degenerate policy. A degenerate policy earns this verdict, never a pass.
    ABSTAIN = "abstain"
    #: The inputs were refused before anything was evaluated. Exit 2.
    REFUSED = "refused"


class OPEVerdict(StrEnum):
    """Whether an off-policy estimate could be produced at all."""

    ESTIMATED = "estimated"
    #: No pack could be resolved. This includes the manifest fallback, which reads a key the
    #: manifest has never had and therefore always yields nothing.
    NO_PACKS = "no_packs"
    #: The packs resolved and held no usable transition.
    EMPTY = "empty"


@dataclass(frozen=True, slots=True)
class PromotionGateReport:
    """A promotion decision, and the artifact it is a view of."""

    verdict: PromotionVerdict
    #: `build_promotion_report_artifact`'s dict, verbatim. The one field; `to_dict` returns it
    #: unchanged, because a second serialisation of a byte-locked artifact is a second answer.
    artifact: Mapping[str, Any] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    detail: str = ""

    @property
    def promoted(self) -> bool:
        return self.verdict is PromotionVerdict.PASS

    @property
    def abstained(self) -> bool:
        return self.verdict is PromotionVerdict.ABSTAIN

    @property
    def dataset_hash(self) -> str:
        """Which dataset this policy was judged against. The field `repo_root` decides."""
        return str(self.artifact.get("dataset_hash", ""))

    @property
    def exit_code(self) -> int:
        """0 for any completed evaluation, 2 for refused inputs.

        PASS, FAIL and ABSTAIN all exit 0, deliberately. A caller branching on this is branching
        on "did it run", not on whether the policy is good; read :attr:`verdict` instead. Making
        the code carry the verdict changes the meaning of every caller that already gates on it.
        """
        return 2 if self.verdict is PromotionVerdict.REFUSED else 0

    def render(self) -> str:
        """ASCII, no trailing newline, no arguments."""
        if self.verdict is PromotionVerdict.REFUSED:
            return f"  REFUSED: {self.detail}"
        a = self.artifact
        lines = [
            f"  {self.verdict.value.upper()}: {a.get('policy_id', '?')} "
            f"({a.get('policy_family', '?')})",
            f"    dataset {a.get('dataset_id', '?')}  hash {self.dataset_hash[:16]}",
        ]
        wis = dict((a.get("estimators") or {}).get("wis") or {})
        if wis:
            lines.append(
                f"    wis lift {wis.get('lift')}  target {wis.get('target_value')}  "
                f"baseline {wis.get('baseline_value')}"
            )
        for reason in self.reasons:
            lines.append(f"    reason: {reason}")
        if self.abstained:
            # The exit code is 0 for all three verdicts, so this line is the only place a reader
            # is told that an ABSTAIN is not a PASS.
            lines.append("    ABSTAIN is not a pass. The exit code is 0 for pass, fail and abstain.")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """The artifact, unchanged. Not reshaped: it is compared byte for byte elsewhere."""
        return dict(self.artifact)

    def write(self, path: "str | Path") -> str:
        """Write the artifact and return the sha256 of the file.

        Writing bytes makes the digest true and makes the file match what every other machine
        produces. `write_promotion_report` used to build the blob with LF, write it with
        `write_text` (which translates to CRLF on Windows), and hash the pre-translation bytes, so
        a verifier comparing an artifact against that digest saw a mismatch on every Windows write.
        It was repaired the same way on 2026-09-10; this docstring described that defect in the
        present tense for as long as it stood.
        """
        import json  # noqa: PLC0415

        blob = (json.dumps(dict(self.artifact), sort_keys=True, indent=2) + "\n").encode("utf-8")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(blob)
        return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True, slots=True)
class PromotionGate:
    """Would this policy be promoted, and against which dataset.

        from src.robot.grasping.rl.evaluation import PromotionGate

        report = PromotionGate.from_policy_artifact(
            policy_family="v6_recovery",
            policy_artifact_path="docs/baselines/rl_policies/v6_recovery_baseline_v1.json",
            repo_root=root,
            pack_paths=packs,
            dataset_id="v7_promotion_canonical",
            training_scope="dense",
            seed=1,
        ).evaluate()
        print(report.render())
        if report.abstained:
            ...   # not a pass, and the exit code does not say so
    """

    policy_family: str
    policy_artifact_path: Path
    #: Required. Keyword-only on `from_policy_artifact`; the dataclass constructor still takes
    #: it positionally. Omitting it silently changes `dataset_hash` and turns `dataset_paths`
    #: into backslash strings, with the same verdict and no warning.
    repo_root: Path
    #: Required, because the callee requires them. `evaluate_policy_for_promotion` declares no
    #: default for pack_paths, dataset_id, training_scope or seed, so there is nothing to inherit
    #: and an `UNSET` here would only postpone a `TypeError`.
    pack_paths: "Sequence[Path]" = ()
    dataset_id: str = ""
    training_scope: str = ""
    seed: int = 1
    #: The two the callee does default (`thresholds=None`, `repo_root=None`); `repo_root` being
    #: defaultable is the defect, so it is required above.
    thresholds: "Maybe[Any]" = UNSET

    @classmethod
    def from_policy_artifact(
        cls,
        *,
        policy_family: str,
        policy_artifact_path: "str | Path",
        repo_root: "str | Path",
        pack_paths: "Sequence[str | Path]",
        dataset_id: str,
        training_scope: str,
        seed: int,
        thresholds: "Maybe[Any]" = UNSET,
    ) -> "PromotionGate":
        """A trained artifact on disk, judged against replay packs.

        Nothing is defaulted here. `evaluate_policy_for_promotion` declares its own seed and
        thresholds; a copy in this signature would be a second declaration of one fact, and two
        declarations of one default produce two different artifacts.
        """
        return cls(
            policy_family=policy_family,
            policy_artifact_path=Path(policy_artifact_path),
            repo_root=Path(repo_root),
            pack_paths=tuple(Path(p) for p in pack_paths),
            dataset_id=dataset_id,
            training_scope=training_scope,
            seed=seed,
            thresholds=thresholds,
        )

    def evaluate(self) -> PromotionGateReport:
        """Run the gate. Reads; writes nothing until :meth:`PromotionGateReport.write`."""
        from .promotion import (  # noqa: PLC0415
            PromotionInputError,
            build_promotion_report_artifact,
            evaluate_policy_for_promotion,
        )

        options: dict[str, Any] = {}
        if chosen(self.thresholds):
            options["thresholds"] = self.thresholds

        try:
            report = evaluate_policy_for_promotion(
                policy_family=self.policy_family,
                policy_artifact_path=self.policy_artifact_path,
                pack_paths=list(self.pack_paths),
                dataset_id=self.dataset_id,
                training_scope=self.training_scope,
                seed=self.seed,
                repo_root=self.repo_root,
                **options,
            )
        except PromotionInputError as exc:
            return PromotionGateReport(verdict=PromotionVerdict.REFUSED, detail=str(exc))

        artifact = build_promotion_report_artifact(report)
        return PromotionGateReport(
            verdict=PromotionVerdict(str(report.verdict)),
            artifact=artifact,
            reasons=tuple(report.reasons),
        )


@dataclass(frozen=True, slots=True)
class OPERunReport:
    """An off-policy evaluation, as a view over its byte-locked dict."""

    verdict: OPEVerdict
    #: `ope.build_ope_report`'s dict, verbatim and in one field. `to_dict` returns it unchanged.
    report: Mapping[str, Any] = field(default_factory=dict)
    detail: str = ""

    @property
    def estimated(self) -> bool:
        return self.verdict is OPEVerdict.ESTIMATED

    @property
    def exit_code(self) -> int:
        """0 estimated, 3 anything else. The CLI has four distinct exit-3 refusals and a caller
        cannot tell them apart from the code; :attr:`verdict` and :attr:`detail` can."""
        return 0 if self.estimated else 3

    def render(self) -> str:
        """ASCII, no trailing newline, no arguments."""
        if not self.estimated:
            return f"  {self.verdict.value.upper()}: {self.detail}"
        sections = dict(self.report.get("sections") or {})
        lines = [f"  ESTIMATED: {len(sections)} section(s)"]
        for name, body in sorted(sections.items()):
            wis = dict((dict(body).get("estimators") or {}).get("wis") or {})
            # `value` and `effective_sample_size`, not `lift`. An OPE section carries a WIS
            # value; `lift` is the promotion gate's word, one layer up, and reading it here
            # yields `None` for every section.
            estimable = bool(wis.get("estimable"))
            ess = wis.get("effective_sample_size")
            lines.append(
                f"    {name:<12}wis {wis.get('value')}"
                + (f"  ess {float(ess):.1f}" if isinstance(ess, (int, float)) else "")
                + ("" if estimable else "   (NOT estimable: abstain)")
            )
        # Candidate and ranking carry no behaviour action in the logs, so their WIS is not a real
        # off-policy estimate; only sequencing has real importance weights. A reader comparing the
        # three numbers is comparing two abstains against one answer.
        lines.append("    candidate/ranking WIS is NOT a real off-policy estimate: the replay packs")
        lines.append("    store no behaviour action, so those sections abstain by construction.")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """The report, unchanged. Byte-locked: `_relpath` writes platform-native separators,
        so normalising a path here changes the bytes of a hash-locked artifact."""
        return dict(self.report)

    def write(self, path: "str | Path") -> str:
        """Write and return the sha256 of the file, in bytes, for the same reason as the gate's."""
        import json  # noqa: PLC0415

        blob = (json.dumps(dict(self.report), sort_keys=True, indent=2) + "\n").encode("utf-8")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(blob)
        return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True, slots=True)
class OPERun:
    """An off-policy evaluation over replay packs.

    The noun is the run; the dict it produces is carried, not recomputed. A second computation
    over a byte-locked dict is a second answer.
    """

    repo_root: Path
    pack_paths: "Maybe[Sequence[Path]]" = UNSET
    #: Required: the CLI declares these as argparse literals, so there is no callee default to
    #: inherit and an UNSET here would only postpone the failure.
    dataset_id: str = ""
    sequencing_artifact: str = ""
    #: UNSET here, unlike the fields above. `build_ope_report` declares `rng_seed = 0` while the
    #: CLI's argparse declares `1`: two live declarations that disagree. Forwarding only when
    #: chosen means a library caller gets the builder's 0 and the CLI keeps its 1, which is the
    #: honest reading of "the caller did not choose".
    seed: "Maybe[int]" = UNSET
    #: Which manifest the fallback would read. It cannot work; see :meth:`evaluate`.
    fallback_dataset_id: str = "v1_bootstrap"

    @classmethod
    def over_packs(
        cls,
        *,
        repo_root: "str | Path",
        dataset_id: str,
        sequencing_artifact: str,
        pack_paths: "Maybe[Sequence[str | Path]]" = UNSET,
        seed: "Maybe[int]" = UNSET,
    ) -> "OPERun":
        """Named replay packs, or the canonical pair when none are given.

        Nothing is defaulted here. The CLI declares the canonical pack list, the dataset id,
        the sequencing artifact and the seed; copies in this signature would be second declarations
        of four facts that decide what a byte-locked report is computed over.
        """
        return cls(
            repo_root=Path(repo_root),
            pack_paths=tuple(Path(p) for p in pack_paths) if chosen(pack_paths) else UNSET,
            dataset_id=dataset_id,
            sequencing_artifact=sequencing_artifact,
            seed=seed,
        )

    def evaluate(self) -> OPERunReport:
        """Resolve the packs, load the records, and estimate. Reads; writes nothing.

        `build_ope_report` takes records, a target-action callable, a dataset id and a list of
        path strings; it does no resolution. The resolution lives here: the canonical pack pair,
        the manifest fallback, the sequencing artifact, and the `_relpath` convention whose
        separators end up in the report's bytes.

        The refusals share one exit code, 3, and are separate verdicts here, each with its own
        `detail`: canonical packs missing with no fallback manifest, a fallback manifest that
        yields no packs, malformed input while loading, a missing sequencing artifact, and
        malformed input during the build.
        """
        import json  # noqa: PLC0415

        from ._cli_common import COMMITTED_MANIFEST_DIR_REL  # noqa: PLC0415
        # The dataset id, the sequencing artifact and the fallback id are argparse literals in
        # the CLI's parser, so there is no constant to inherit, which is why `dataset_id` and
        # `sequencing_artifact` are required here rather than defaulted.
        # `DEFAULT_OPE_REPLAY_PACKS` and `_relpath` do exist as names and are imported below.
        from ._cli_common import DEFAULT_OPE_REPLAY_PACKS  # noqa: PLC0415
        from ._commands_eval import _relpath  # noqa: PLC0415
        from .ope import (  # noqa: PLC0415
            MalformedOPEInputError,
            build_ope_report,
            load_records_for_ope,
            sequencing_target_action_from_policy_path,
        )

        if chosen(self.pack_paths):
            packs = [self.repo_root / p if not p.is_absolute() else p for p in self.pack_paths]
        else:
            packs = [self.repo_root / rel for rel in DEFAULT_OPE_REPLAY_PACKS]
            if not all(p.exists() for p in packs):
                manifest = (
                    self.repo_root / COMMITTED_MANIFEST_DIR_REL / f"{self.fallback_dataset_id}.json"
                )
                if not manifest.exists():
                    return OPERunReport(
                        verdict=OPEVerdict.NO_PACKS,
                        detail=(
                            f"the canonical replay packs are not all on disk and there is no "
                            f"fallback manifest at {manifest}"
                        ),
                    )
                # This fallback cannot work, and it is kept unchanged on purpose. It reads a
                # `"splits"` key; a dataset manifest carries `split_files`, `split_counts`,
                # `split_hashes` and `class_counts_by_split`, and no `splits`. The list is
                # therefore always empty and the run dies a step later with "empty after loading
                # all paths". Repairing it changes which records an OPE report is computed over,
                # which is a behaviour change to a byte-locked artifact. The verdict names it
                # instead.
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                packs = [self.repo_root / rel for rel in (payload.get("splits") or {}).values() if rel]
                if not packs:
                    return OPERunReport(
                        verdict=OPEVerdict.NO_PACKS,
                        detail=(
                            f"the manifest fallback yielded no packs. It reads a 'splits' key that "
                            f"{manifest.name} does not have (it has split_files), so this path has "
                            f"never been able to resolve anything."
                        ),
                    )

        try:
            records = load_records_for_ope(packs)
        except MalformedOPEInputError as exc:
            return OPERunReport(verdict=OPEVerdict.EMPTY, detail=f"malformed input: {exc}")

        sequencing = self.repo_root / self.sequencing_artifact
        if not sequencing.exists():
            return OPERunReport(
                verdict=OPEVerdict.NO_PACKS,
                detail=f"the sequencing artifact is not at {sequencing}",
            )

        try:
            report = build_ope_report(
                records=records,
                sequencing_target_action_for_state=sequencing_target_action_from_policy_path(
                    sequencing
                ),
                dataset_id=self.dataset_id,
                # `_relpath`, not `as_posix()`. Its separators are part of the report's bytes,
                # so tidying them here changes a hash-locked artifact.
                dataset_paths=[_relpath(p, self.repo_root) for p in packs],
                **({"rng_seed": self.seed} if chosen(self.seed) else {}),
            )
        except MalformedOPEInputError as exc:
            return OPERunReport(
                verdict=OPEVerdict.EMPTY, detail=f"malformed input during the build: {exc}"
            )
        return OPERunReport(verdict=OPEVerdict.ESTIMATED, report=report)
