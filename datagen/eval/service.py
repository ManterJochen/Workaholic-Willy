"""Grading a calculator against the labels, as one noun with four verbs and no command line.

The pieces underneath are typed already: `evaluate_dataset`, `summarise`, `run_gate` and
`side_approach_report`. What this file adds is the part a caller cannot guess, which is which output
file a run may write to and where the rows a report reads come from.

`grasp_eval.jsonl` is appended across runs, so a run that graded only some rungs, or that changed
what the calculator saw, must write elsewhere: its rows are otherwise indistinguishable from a full
run's and a later summary reports them together. :func:`evaluation_output_name` is that rule.

`last_report` runs nothing. It prints the last report, so it returns in seconds and can name rungs
that no longer exist, which reads exactly like a fast run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

from src.contracts.options import UNSET, Maybe, chosen

if TYPE_CHECKING:  # pragma: no cover (imported for annotations only)
    from datagen.config import DatagenConfig
    from datagen.grasps.verdict import JawModel

__all__ = ["GraspEvaluation", "EvaluationReport", "GateReport", "TiltReport",
           "evaluation_output_name", "REFERENCE_OUTPUT_NAME"]

MaskSource = Literal["gt", "pred"]

#: The one file every reported number is quoted from. Only a full, ground-truth, unmodified run may
#: write to it.
REFERENCE_OUTPUT_NAME = "grasp_eval.jsonl"

_NEWLINE = chr(10)


def evaluation_output_name(*, mask_source: str = "gt", mask_completion: str = "none",
                           selective: bool = False) -> str:
    """Which file this run is allowed to write, given what it changed.

    A partial ladder never writes into the reference file. `grasp_eval.jsonl` is appended across
    runs, so a run that evaluated a subset would leave rows indistinguishable from a full one's, and
    a later summary would report them together.

    And a different mask policy is a different measurement. Two runs that differ in what the
    calculator saw must not share a file, or the comparison becomes a run against itself.
    """
    if mask_source == "gt" and mask_completion == "none" and not selective:
        return REFERENCE_OUTPUT_NAME
    suffix = "" if mask_completion == "none" else f"_{mask_completion}"
    rungs = "_rungs" if selective else ""
    if mask_source != "gt":
        return f"grasp_eval_{mask_source}{suffix}{rungs}.jsonl"
    return f"grasp_eval{suffix}{rungs}.jsonl"


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """One ladder run, per configuration."""

    raw: dict[str, Any]
    out_name: str

    @property
    def rows(self) -> int:
        return int(self.raw.get("rows", 0))

    @property
    def by_config(self) -> dict[str, Any]:
        return dict(self.raw.get("by_config", {}))

    def render(self) -> str:
        lines = []
        for name, entry in sorted(self.by_config.items()):
            lines.append(
                f"  {name:<9} precision {entry['precision'] * 100:5.1f}% "
                f"({entry['valid']}/{entry['candidates']})   coverage "
                f"{entry['coverage'] * 100:5.1f}% of {entry['objects_with_labels']} "
                f"graspable objects")
        lines.append(f"  rows {self.rows} -> {self.out_name}")
        return _NEWLINE.join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {"out_name": self.out_name, "rows": self.rows, "by_config": self.by_config}


@dataclass(frozen=True, slots=True)
class GateReport:
    """The gate's deltas against the committed baseline."""

    deltas: dict[str, Any]

    @property
    def passed(self) -> bool:
        """Every key, not the mean. A gate that averages its own checks can be carried by one."""
        return all(bool(entry.get("pass", False)) for entry in self.deltas.values()
                   if isinstance(entry, dict) and "pass" in entry)

    def render(self) -> str:
        lines = [f"  {key:<28} {value}" for key, value in sorted(self.deltas.items())]
        lines.append("PASS" if self.passed else "FAIL")
        return _NEWLINE.join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "deltas": self.deltas}


@dataclass(frozen=True, slots=True)
class TiltReport:
    """Coverage by the object's most top-down admissible grasp.

    The number the ladder's `top1` cannot give. One rate cannot distinguish "found a side grasp for
    a short object" from "found a top grasp for a tall one", so a generator could solve the whole
    off-vertical population and the headline would barely move.
    """

    text: str
    rows: int
    files: tuple[str, ...]

    def render(self) -> str:
        return _NEWLINE.join([f"rows: {self.rows} from {', '.join(self.files)}", self.text])

    def as_dict(self) -> dict[str, Any]:
        return {"rows": self.rows, "files": list(self.files), "text": self.text}


@dataclass
class GraspEvaluation:
    """A rendered, labelled dataset, and the instruments that grade a calculator on it."""

    dataset: Path

    @classmethod
    def from_config(cls, config: "DatagenConfig", *, name: str,
                    out_root: str | Path | None = None) -> "GraspEvaluation":
        root = Path(out_root) if out_root is not None else Path(config.output.root)
        return cls(dataset=root / name)

    @classmethod
    def from_dataset(cls, dataset: str | Path) -> "GraspEvaluation":
        return cls(dataset=Path(dataset))

    def describe(self) -> str:
        """What is on disk to grade, before anything runs. Zero arguments, ASCII."""
        evals = sorted(p.name for p in self.dataset.glob("grasp_eval*.jsonl"))
        return _NEWLINE.join([
            f"grasp evaluation over {self.dataset}",
            f"  labels         {'present' if (self.dataset / 'grasps.jsonl').is_file() else 'MISSING'}",
            f"  eval files     {', '.join(evals) if evals else 'none yet'}"])

    def evaluate(self, *, mask_source: MaskSource = "gt", mask_completion: str = "none",
                 rungs: str | None = None, jobs: Maybe[int] = UNSET,
                 limit: int | None = None, model: "Maybe[JawModel | None]" = UNSET,
                 files: Maybe[Sequence[str]] = UNSET, scene_step: Maybe[int] = UNSET,
                 with_jaw: Maybe[bool] = UNSET, with_suction: Maybe[bool] = UNSET,
                 suction_every: Maybe[int] = UNSET, progress_every: Maybe[int] = UNSET,
                 support_cfg: Maybe[Any] = UNSET) -> EvaluationReport:
        """Run the ladder. CPU only, but minutes to hours: it re-runs the generator on every object
        in every view, in every configuration.

        `jobs` is the whole difference: a scene reads only its own files, so the work parallelises
        over scenes.

        The rest decide what is graded, and none of them is cosmetic: `with_jaw` and `with_suction`
        choose the modality, `model` picks the jaw the verdict is computed against, `scene_step`
        subsamples. A wrapper that omits one is quieter than one that fights over its default:
        nothing raises, the caller gets the callee's value, and the run is not the one they asked
        for.

        Every one is `UNSET` and forwarded only when chosen, so `evaluate_dataset` stays the single
        declaration of what each default is.
        """
        from datagen.eval.ladder import evaluate_dataset, select_configurations  # noqa: PLC0415

        configs = select_configurations(rungs)
        out_name = evaluation_output_name(mask_source=mask_source,
                                          mask_completion=mask_completion,
                                          selective=rungs is not None)
        chosen_only = {name: value for name, value in (
            ("jobs", jobs), ("model", model), ("files", files), ("scene_step", scene_step),
            ("with_jaw", with_jaw), ("with_suction", with_suction),
            ("suction_every", suction_every), ("progress_every", progress_every),
            ("support_cfg", support_cfg)) if chosen(value)}
        raw = evaluate_dataset(self.dataset, mask_source=mask_source, out_name=out_name,
                               mask_completion=mask_completion, configs=configs, limit=limit,
                               **chosen_only)
        return EvaluationReport(raw=raw, out_name=out_name)

    def last_report(self) -> EvaluationReport:
        """The last run's rows, re-summarised. Nothing is evaluated.

        This is the `--summary-only` path and it is easy to misread: it returns in seconds and can
        name rungs that no longer exist, which reads exactly like a fast run.
        """
        from datagen.eval.ladder import summarise  # noqa: PLC0415

        return EvaluationReport(raw=summarise(self.dataset), out_name="(not written)")

    def gate(self, *, write: bool = False, model: "JawModel | None" = None) -> GateReport:
        """The committed gate: its own pinned configurations, scene step, file name and baseline."""
        from datagen.eval.ladder import run_gate  # noqa: PLC0415

        return GateReport(deltas=run_gate(self.dataset, write=write, model=model))

    def approach_tilt(self, *, files: Sequence[str] | None = None) -> TiltReport:
        """Coverage by approach tilt, read off rows a previous :meth:`evaluate` wrote.

        The three functions in `approach_tilt` take in-memory row sequences, so locating
        `grasps.jsonl`, globbing `grasp_eval*.jsonl` and parsing both happens here rather than in
        every caller.
        """
        from datagen.eval.approach_tilt import (  # noqa: PLC0415
            format_report, object_reach_tilt, side_approach_report,
        )

        labels_path = self.dataset / "grasps.jsonl"
        if not labels_path.is_file():
            raise FileNotFoundError(
                f"no grasp labels at {labels_path}; run `label-grasps` first")
        names = (list(files) if files
                 else sorted(p.name for p in self.dataset.glob("grasp_eval*.jsonl")))
        rows = [row for name in names for row in _jsonl(self.dataset / name)]
        if not rows:
            raise FileNotFoundError(
                f"no grasp_eval*.jsonl rows under {self.dataset}; run `eval-grasps` first")
        report = side_approach_report(rows, object_reach_tilt(_jsonl(labels_path)))
        return TiltReport(text=format_report(report), rows=len(rows), files=tuple(names))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    """Every non-blank line of a JSONL file, parsed. Missing file means no rows, not an error."""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
