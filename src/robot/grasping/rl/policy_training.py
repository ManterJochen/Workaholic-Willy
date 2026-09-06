"""Training one offline policy from Python, with the context that otherwise lives in the CLI.

`train_from_manifest`, `train_ranking_from_manifest` and `train_sequencing_from_manifest` are
public, pure and take a path. This module supplies what surrounds them: turning a dataset id into a
manifest path, the typed refusal when that file is absent, the exit code that refusal carries, the
default output location, and the summary that names the artifact and its hash.

The five training commands do not share one input contract, which is why this module covers three:

    candidate, ranking, sequencing   --dataset-id           exit 0 / 2
    perception-budget                --dataset-id or packs  exit 0 / 2 / 3
    recovery                         --replay-pack          exit 0 / 2 / 3

The last two take a different input contract, and perception-budget's summary additionally quotes
flag values and an artifact key, so both keep their handlers in `_commands_train.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Mapping

from src.contracts import UNSET, Maybe, chosen, resolve

__all__ = [
    "PolicyTraining",
    "PolicyTrainingReport",
    "TrainingFamily",
    "TrainingVerdict",
]


class TrainingFamily(StrEnum):
    """The policies this module can train from a dataset manifest.

    `perception-budget` and `recovery` take replay packs rather than, or as well as, a manifest,
    which is a different input contract; they keep their CLI handlers.
    """

    CANDIDATE = "candidate"
    RANKING = "ranking"
    SEQUENCING = "sequencing"


class TrainingVerdict(StrEnum):
    """How the run ended. Carries the exit code each outcome maps to.

    The exit code is what a shell branches on; the verdict is what a person reads. The two
    verdicts map one-to-one onto exit 0 and exit 2; a second refusal would have to reuse exit 2,
    and its verdict is what would keep it distinct from `NO_MANIFEST`.
    """

    #: Trained and the artifact was written. Exit 0.
    TRAINED = "trained"
    #: No manifest for that dataset id. Exit 2.
    NO_MANIFEST = "no_manifest"


@dataclass(frozen=True, slots=True)
class _Recipe:
    """What differs between the three families. Everything else is shared by construction."""

    command: str
    filename: str
    #: ``(manifest_path, seed, **extra) -> (policy, train_result, artifact)``
    trainer: Callable[..., Any]
    #: ``(artifact, output_path) -> sha256``
    writer: Callable[..., str]
    #: Fields copied from the trainer's result onto the summary.
    result_fields: tuple[str, ...]
    #: Fields copied from the caller's trainer options onto the summary.
    #:
    #: :data:`TrainingFamily.SEQUENCING` needs this. Its summary carries
    #: ``min_support_threshold``, which the handler takes from ``int(args.min_support)`` and not
    #: from the training result. A recipe that could only copy result fields drops that key
    #: silently, and `sort_keys=True` means a missing key changes the JSON without moving anything
    #: else.
    option_fields: tuple[str, ...] = ()


def _recipes() -> Mapping[TrainingFamily, _Recipe]:
    """Built on demand so importing this module costs no trainer import.

    The trainers pull numpy and the dataset layer between them. A caller who wants one should not
    pay for the others, and a module-level table would charge them at import.
    """
    from src.robot.grasping.rl._cli_common import (  # noqa: PLC0415
        CANDIDATE_POLICY_FILENAME,
        RANKING_POLICY_FILENAME,
        SEQUENCING_POLICY_FILENAME,
    )
    from src.robot.grasping.rl.train_candidate import (  # noqa: PLC0415
        train_from_manifest,
        write_artifact,
    )
    from src.robot.grasping.rl.train_ranking import (  # noqa: PLC0415
        train_ranking_from_manifest,
        write_ranking_artifact,
    )
    from src.robot.grasping.rl.train_sequencing import (  # noqa: PLC0415
        train_sequencing_from_manifest,
        write_sequencing_artifact,
    )

    return {
        TrainingFamily.CANDIDATE: _Recipe(
            command="train-candidate-policy",
            filename=CANDIDATE_POLICY_FILENAME,
            trainer=train_from_manifest,
            writer=write_artifact,
            result_fields=("iterations", "converged", "num_samples", "num_positive",
                           "final_log_loss"),
        ),
        TrainingFamily.RANKING: _Recipe(
            command="train-ranking-policy",
            filename=RANKING_POLICY_FILENAME,
            trainer=train_ranking_from_manifest,
            writer=write_ranking_artifact,
            result_fields=("iterations", "converged", "num_pairs", "num_groups_with_pairs",
                           "final_log_loss", "ndcg_at_1", "pairwise_accuracy",
                           "ndcg_at_1_baseline", "pairwise_accuracy_baseline"),
        ),
        TrainingFamily.SEQUENCING: _Recipe(
            command="train-sequencing-policy",
            filename=SEQUENCING_POLICY_FILENAME,
            trainer=train_sequencing_from_manifest,
            writer=write_sequencing_artifact,
            result_fields=("num_records", "num_groups", "num_pairs", "num_cells_observed",
                           "num_cells_committed", "num_cells_below_threshold"),
            option_fields=("min_support_threshold",),
        ),
    }


@dataclass(frozen=True, slots=True)
class PolicyTrainingReport:
    """What the run produced, or why it did not. Satisfies both halves of the report contract."""

    family: TrainingFamily
    verdict: TrainingVerdict
    #: The CLI's summary mapping, verbatim. Empty when nothing was trained.
    summary: Mapping[str, Any] = ()  # type: ignore[assignment]
    #: Why it was refused. Empty otherwise.
    detail: str = ""

    @property
    def trained(self) -> bool:
        return self.verdict is TrainingVerdict.TRAINED

    @property
    def exit_code(self) -> int:
        """The exit code, carried on the report rather than decided in the CLI."""
        return 0 if self.trained else 2

    def render(self) -> str:
        """The whole result, for a person. ASCII, no trailing newline, no arguments."""
        if not self.trained:
            return f"  {self.family.value}: REFUSED ({self.verdict.value}) {self.detail}".rstrip()
        # Lines are built and joined so an empty summary cannot leave a trailing newline, which
        # embedding the separator in the f-string would.
        lines = [f"  {self.family.value}: trained"]
        lines.extend(f"    {key:<28}{value}" for key, value in sorted(dict(self.summary).items()))
        return "\n".join(lines)

    def as_cli_json(self) -> str:
        """The summary exactly as the CLI prints it.

        The bytes are the contract here, which is why this is not `render()`. The separator, the
        key order and the indent are part of the answer rather than presentation, so this returns
        `json.dumps(..., sort_keys=True, indent=2)` and has to keep returning exactly that.
        """
        return json.dumps(dict(self.summary), sort_keys=True, indent=2)

    def to_dict(self) -> dict[str, Any]:
        """Plain data, `json.dumps`-safe with no custom encoder."""
        return {
            "family": self.family.value,
            "verdict": self.verdict.value,
            "exit_code": self.exit_code,
            "detail": self.detail,
            "summary": dict(self.summary),
        }


@dataclass(frozen=True, slots=True)
class PolicyTraining:
    """Train one offline policy from a committed dataset manifest.

        from src.robot.grasping.rl.policy_training import PolicyTraining, TrainingFamily

        report = PolicyTraining.from_dataset_id(
            TrainingFamily.RANKING, "v1_bootstrap", repo_root=Path.cwd()).train()
        print(report.render())

    `from_dataset_id` resolves against the committed manifest directory named by
    `COMMITTED_MANIFEST_DIR_REL`, which this repository does not populate; a caller holding a
    manifest of their own passes its path to :meth:`from_manifest` instead.
    """

    family: TrainingFamily
    repo_root: Path
    manifest_path: Path
    #: Where the artifact goes. ``None`` means the committed location for this family.
    output_path: Path | None = None
    #: Not defaulted here. Every trainer declares `seed: int = 1` and the CLI's argparse declares 1
    #: as well; a third literal in this class silently changes the artifact hash for the same
    #: nominal call. `UNSET` means the trainer's own default applies.
    seed: "Maybe[int]" = UNSET
    #: Extra keyword arguments for this family's trainer, e.g. ``prune_threshold`` for candidate.
    trainer_options: Mapping[str, Any] = ()  # type: ignore[assignment]

    @classmethod
    def from_dataset_id(
        cls,
        family: TrainingFamily,
        dataset_id: str,
        *,
        repo_root: Path,
        output_path: "Maybe[Path | None]" = UNSET,
        seed: "Maybe[int]" = UNSET,
        **trainer_options: Any,
    ) -> "PolicyTraining":
        """Resolve a dataset id to its committed manifest. Nothing is read until :meth:`train`.

        The path is resolved here and checked in the verb. Resolution is arithmetic on a string and
        cannot fail; reading is what fails, and :meth:`train` has a report to carry that failure.
        """
        from src.robot.grasping.rl._cli_common import (  # noqa: PLC0415
            COMMITTED_MANIFEST_DIR_REL,
        )

        return cls(
            family=family,
            repo_root=repo_root,
            manifest_path=(repo_root / f"{COMMITTED_MANIFEST_DIR_REL}/{dataset_id}.json").resolve(),
            output_path=resolve("output_path", output_path, None),
            seed=seed,
            trainer_options=dict(trainer_options),
        )

    @classmethod
    def from_manifest(
        cls,
        family: TrainingFamily,
        manifest_path: Path,
        *,
        repo_root: Path,
        output_path: "Maybe[Path | None]" = UNSET,
        seed: "Maybe[int]" = UNSET,
        **trainer_options: Any,
    ) -> "PolicyTraining":
        """A manifest the caller already has, anywhere on disk.

        The door a customer training on their own dataset uses: `from_dataset_id` assumes the
        repository's committed layout, and a customer's manifest is not in it.
        """
        return cls(
            family=family,
            repo_root=repo_root,
            manifest_path=Path(manifest_path).resolve(),
            output_path=resolve("output_path", output_path, None),
            seed=seed,
            trainer_options=dict(trainer_options),
        )

    def train(self) -> PolicyTrainingReport:
        """Fit the policy and write its artifact. Refuses rather than raising on a missing manifest.

        A manifest path that is not a file yields :data:`TrainingVerdict.NO_MANIFEST` and exit 2
        instead of whatever the trainer would raise on it.
        """
        recipe = _recipes()[self.family]
        if not self.manifest_path.is_file():
            return PolicyTrainingReport(
                family=self.family,
                verdict=TrainingVerdict.NO_MANIFEST,
                detail=f"manifest not found at {self.manifest_path}",
            )

        options = dict(self.trainer_options)
        if chosen(self.seed):
            options["seed"] = int(self.seed)
        policy, result, artifact = recipe.trainer(manifest_path=self.manifest_path, **options)
        output = self.output_path or self._committed_output(recipe.filename)
        artifact_hash = recipe.writer(artifact, output)

        summary: dict[str, Any] = {
            "artifact_path": self._reported_path(output),
            "artifact_sha256": artifact_hash,
            "policy_id": policy.policy_id,
        }
        for field in recipe.result_fields:
            summary[field] = getattr(result, field)
        for field in recipe.option_fields:
            summary[field] = self._effective_option(recipe.trainer, field)
        return PolicyTrainingReport(
            family=self.family, verdict=TrainingVerdict.TRAINED, summary=summary,
        )

    def _effective_option(self, trainer: Callable[..., Any], field: str) -> Any:
        """What the trainer will actually use for ``field``: the caller's value, or its default.

        The default is read off the trainer's own signature, so there is one source and drift is
        impossible. The summary has to report the effective value, and a caller who supplied none
        still gets a number from the trainer. Writing that number here as a literal, or importing
        the constant separately, is a second declaration of the same fact and drifts into a
        different artifact hash for the same nominal call.
        """
        options = dict(self.trainer_options)
        if field in options:
            return options[field]
        import inspect  # noqa: PLC0415 (only reached when a summary field was not supplied)

        return inspect.signature(trainer).parameters[field].default

    def _committed_output(self, filename: str) -> Path:
        from src.robot.grasping.rl._cli_common import (  # noqa: PLC0415
            COMMITTED_RL_POLICY_DIR_REL,
        )

        return self.repo_root / COMMITTED_RL_POLICY_DIR_REL / filename

    def _reported_path(self, output: Path) -> str:
        """Repo-relative when it can be, absolute when it cannot.

        `--output` may point outside the repository, a tempdir during CI being the normal case, so
        the path is reported absolute there rather than crashing on `relative_to`.
        """
        try:
            return output.relative_to(self.repo_root).as_posix()
        except ValueError:
            return output.as_posix()
