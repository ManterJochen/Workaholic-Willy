"""The ranker path as two nouns and two verbs, callable without building a command line.

The walk, the physics join and the write are one noun, :class:`RankerCorpus`; the fit, the export,
the artifact, the card and the baseline comparison are another, :class:`RankerFit`. Both are callable
from Python rather than only through `python -m datagen`, because the ranker is the part of the stack
a customer is most likely to retrain: it is the part that learns their objects.

The self-reference is real and is not fixed here. Every feature is computed from the point cloud and
the label is the analytic verdict already in `grasp_eval*.jsonl`, so a model fitted on it learns to
predict the referee and the `eval-grasps` ladder is self-referential for it. The ladder stays a
development signal and promotion is measured against the Isaac shake only. Joining physics labels
with :meth:`RankerCorpus.with_physics` is the way out of that loop.

The fit lives in `src/` and this file lives in `datagen/`, and the direction is the point.
`deep.ranker.training.fit_ranker` takes a table, never a path, so the robot library never imports
this package: a generator that became a runtime dependency of a robot would be a generator nobody
can delete or decline to install.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping

from src.contracts.options import UNSET, Maybe, chosen

if TYPE_CHECKING:  # pragma: no cover (imported for annotations only)
    import numpy as np

    from datagen.config import DatagenConfig

__all__ = [
    "CorpusCheck",
    "CorpusCheckReport",
    "RankerCorpus",
    "RankerCorpusReport",
    "RankerFit",
    "RankerFitReport",
    "MaskSource",
]

MaskSource = Literal["gt", "pred"]

#: Where the CLI writes the corpus, declared here as the one place that says so.
DEFAULT_CORPUS_PATH = Path("logs/dl/corpus/ranker_v1.npz")
#: Where the CLI writes the artifact and its card.
DEFAULT_MODEL_DIR = Path("logs/dl/models/ranker")
#: These two are declared here and read by the argparse parser, not the other way round, so the
#: library twin and the CLI cannot hold two different defaults for one thing. Both values have to be
#: admissible: `SOURCES` holds `grasp_labels`, `grasp_eval` and `ranker_npz`, and `spec_named`
#: refuses an unknown name outright. A default outside either list makes
#: `RankerFit.from_corpus(path).fit()` raise on its own default path, which is the worst kind of
#: twin: it exists, it is documented, and it cannot run.
DEFAULT_SOURCE = "grasp_labels"
DEFAULT_SPEC = "bootstrap_jaw_v1"

_NEWLINE = chr(10)
_NO_SERVING = "none given, so a train/serve mismatch cannot be seen"


@dataclass(frozen=True, slots=True)
class RankerCorpusReport:
    """What one corpus build produced, and what the physics join could not resolve."""

    path: Path
    rows: int
    objects: int
    valid: int
    #: The `attach_physics_labels` report, or None when no physics file was joined. Kept whole
    #: rather than flattened: it carries the per-source strata, and a pooled rate over a stratified
    #: draw describes the sampler rather than the data.
    physics: Mapping[str, Any] | None = None

    @property
    def valid_share(self) -> float:
        return self.valid / self.rows if self.rows else 0.0

    def render(self) -> str:
        lines = [f"{self.rows} candidate(s) over {self.objects} object(s): "
                 f"{self.valid} valid ({100 * self.valid_share:.1f} %)",
                 f"wrote {self.path}"]
        if self.physics is not None:
            report = self.physics
            lines.append(f"joined physics onto {report['rows_out']} of {report['rows_in']} row(s) "
                         f"from {report['physics_verdicts']} verdict(s)")
            lines.append(f"  {'stratum':26s} {'scored':>6s} {'held':>6s} {'rate':>7s} "
                         f"{'refused':>8s} {'unmatched':>10s}")
            for name, row in report["by_source"].items():
                rate = f"{100 * row['held'] / row['trials']:.1f} %" if row["trials"] else "-"
                lines.append(f"  {name:26s} {row['trials']:6d} {row['held']:6d} {rate:>7s} "
                             f"{row['refused']:8d} {row['unmatched']:10d}")
            if report["conflicting_keys"]:
                lines.append(f"  WARNING {len(report['conflicting_keys'])} row(s) shaken twice with "
                             f"different verdicts; the first stands: {report['conflicting_keys'][:3]}")
            lines.append(f"  WARNING {report['warning']}")
        return _NEWLINE.join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "rows": self.rows, "objects": self.objects,
                "valid": self.valid, "valid_share": self.valid_share,
                "physics": dict(self.physics) if self.physics is not None else None}


@dataclass
class RankerCorpus:
    """A rendered dataset, turned into the table a ranker trains on. No GPU, no cell, no model.

    The mask source is a real fork and it is not cosmetic. `gt` reads the simulator's instance masks;
    `pred` reads the predicted ones written by `predict-masks`. A corpus built on `gt` measures the
    ranker against perfect segmentation, which no camera delivers.
    """

    dataset: Path
    scenes: int | None = None
    mask_source: MaskSource = "gt"
    physics_path: Path | None = None
    _table: "dict[str, np.ndarray] | None" = field(default=None, repr=False, compare=False)

    @classmethod
    def from_config(cls, config: "DatagenConfig", *, name: str,
                    out_root: str | Path | None = None,
                    scenes: int | None = None, mask_source: MaskSource = "gt",
                    physics_path: str | Path | None = None) -> "RankerCorpus":
        """Locate the dataset the way every sibling command does, from `config.output.root`.

        The root is never hard-coded here. A resolver that ignored the config would look for the
        dataset somewhere other than where `datagen build` put it, and with a same-named dataset in
        that other directory it would build a corpus from the wrong scenes and stamp it with the
        right name.
        """
        root = Path(out_root) if out_root is not None else Path(config.output.root)
        return cls.from_dataset(root / name, scenes=scenes, mask_source=mask_source,
                                physics_path=physics_path)

    @classmethod
    def from_dataset(cls, dataset: str | Path, *, scenes: int | None = None,
                     mask_source: MaskSource = "gt",
                     physics_path: str | Path | None = None) -> "RankerCorpus":
        """Point straight at a dataset directory, for a caller who already knows where it is."""
        return cls(dataset=Path(dataset), scenes=scenes, mask_source=mask_source,
                   physics_path=Path(physics_path) if physics_path is not None else None)

    def with_physics(self, physics_path: str | Path) -> "RankerCorpus":
        """The same corpus, joined against an Isaac shake result. Returns a new object."""
        return RankerCorpus(dataset=self.dataset, scenes=self.scenes,
                            mask_source=self.mask_source, physics_path=Path(physics_path))

    def describe(self) -> str:
        """What this would read, before anything walks a dataset. Zero arguments, ASCII."""
        lines = [f"ranker corpus from {self.dataset}",
                 f"  scenes         {self.scenes if self.scenes is not None else 'every'}",
                 f"  masks          {self.mask_source}"]
        if self.physics_path is not None:
            lines.append(f"  physics        {self.physics_path}")
        else:
            lines.append("  physics        none, so the label is this repository's own referee")
        return _NEWLINE.join(lines)

    def build(self, out: str | Path | None = None) -> RankerCorpusReport:
        """Walk the dataset, optionally join physics, and write the `.npz`.

        Raises `FileNotFoundError` or `ValueError` the way the underlying functions do; the CLI turns
        those into exit codes and a caller may want the exception.
        """
        from datagen.corpus.build import (  # noqa: PLC0415
            attach_physics_labels, build_ranker_corpus, write_corpus,
        )
        from datagen.grasps.masks import PRED_SUFFIX  # noqa: PLC0415

        suffix = "instances" if self.mask_source == "gt" else PRED_SUFFIX
        table = build_ranker_corpus(self.dataset, scenes=self.scenes, mask_suffix=suffix)

        physics: Mapping[str, Any] | None = None
        if self.physics_path is not None:
            table, physics = attach_physics_labels(table, self.physics_path)
            if not physics["rows_out"]:
                raise ValueError(
                    f"{self.physics_path} resolved none of these rows, so there is nothing to train "
                    f"on; the join key is (file, line), never the pose")

        self._table = table
        written = write_corpus(Path(out) if out is not None else DEFAULT_CORPUS_PATH, table)
        labels = table["valid"]
        return RankerCorpusReport(
            path=written, rows=int(len(labels)),
            objects=int(len(set(table["object_key"].tolist()))),
            valid=int(labels.sum()), physics=physics)

    @property
    def table(self) -> "dict[str, np.ndarray]":
        """The table the last :meth:`build` produced, so a caller can fit without a round trip."""
        if self._table is None:
            raise RuntimeError("build() has not run yet, so there is no table to hand out")
        return self._table


@dataclass(frozen=True, slots=True)
class RankerFitReport:
    """One fit, its artifact, and whether it beat the straw man.

    The verdict is a comparison because an absolute is not one: an AUROC means nothing until it is
    read beside what `width_mm` alone scores on the same folds.
    """

    spec: str
    artifact_path: Path
    card_path: Path
    sha256: str
    rows: int
    groups: int
    positives: int
    auroc: float
    baseline_auroc: float
    precision_at_250: float
    baseline_precision_at_250: float
    summary: str = ""

    @property
    def beats_baseline(self) -> bool:
        """Both, not either. A fit that wins on one and loses on the other has not won."""
        return (self.auroc > self.baseline_auroc
                and self.precision_at_250 > self.baseline_precision_at_250)

    def render(self) -> str:
        lines = [self.summary] if self.summary else []
        lines += [f"wrote {self.artifact_path} (NOT committed)",
                  f"      {self.card_path} (committed)",
                  f"  sha256 {self.sha256}"]
        if not self.beats_baseline:
            lines.append("REFUSED: this fit does not beat width_mm alone on both AUROC and P@250")
        return _NEWLINE.join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {"spec": self.spec, "artifact_path": str(self.artifact_path),
                "card_path": str(self.card_path), "sha256": self.sha256, "rows": self.rows,
                "groups": self.groups, "positives": self.positives, "auroc": self.auroc,
                "baseline_auroc": self.baseline_auroc,
                "precision_at_250": self.precision_at_250,
                "baseline_precision_at_250": self.baseline_precision_at_250,
                "beats_baseline": self.beats_baseline}


@dataclass
class RankerFit:
    """Fit the learned grasp ranker on a corpus and write its artifact. Offline: no cell, no policy.

    The hyperparameters are declared in one place, and it is not this one. Every knob below is
    `UNSET` and forwarded only when chosen, so `fit_ranker`'s own defaults stay the single
    declaration. A wrapper that restated a default would agree today and drift silently later, and
    the same nominal call would then produce a different artifact sha256 with nothing to say so.
    """

    corpus: Path
    source: str = DEFAULT_SOURCE
    spec_name: str = DEFAULT_SPEC
    model_dir: Path = DEFAULT_MODEL_DIR
    folds: Maybe[int] = UNSET
    trees: Maybe[int] = UNSET
    learning_rate: Maybe[float] = UNSET
    tree_depth: Maybe[int] = UNSET

    @classmethod
    def from_corpus(cls, corpus: str | Path, *, source: str = DEFAULT_SOURCE,
                    spec: str = DEFAULT_SPEC, model_dir: str | Path = DEFAULT_MODEL_DIR,
                    folds: Maybe[int] = UNSET, trees: Maybe[int] = UNSET,
                    learning_rate: Maybe[float] = UNSET,
                    tree_depth: Maybe[int] = UNSET) -> "RankerFit":
        return cls(corpus=Path(corpus), source=source, spec_name=spec, model_dir=Path(model_dir),
                   folds=folds, trees=trees, learning_rate=learning_rate, tree_depth=tree_depth)

    @classmethod
    def from_report(cls, report: RankerCorpusReport, **kwargs: Any) -> "RankerFit":
        """Fit the corpus a :class:`RankerCorpus` just wrote, without restating its path."""
        return cls.from_corpus(report.path, **kwargs)

    def describe(self) -> str:
        """What this would fit, before sklearn is imported. Zero arguments, ASCII."""
        chosen_knobs = {name: value for name, value in (
            ("folds", self.folds), ("trees", self.trees),
            ("learning_rate", self.learning_rate), ("tree_depth", self.tree_depth))
            if chosen(value)}
        return _NEWLINE.join([
            f"ranker fit {self.spec_name} on {self.corpus} (source {self.source})",
            f"  artifact       {self.model_dir / (self.spec_name + '.json')}",
            f"  card           {self.model_dir / (self.spec_name + '.card.json')}",
            f"  overrides      {chosen_knobs or 'none, the trainer decides every one'}"])

    def fit(self) -> RankerFitReport:
        """Fit, export, write the artifact and its card, and report against the baseline."""
        from src.robot.grasping.deep.ranker.features import spec_named  # noqa: PLC0415
        from src.robot.grasping.deep.ranker.training import (  # noqa: PLC0415
            export_ranker, fit_ranker, write_artifact,
        )

        from datagen.corpus.sources import load_corpus  # noqa: PLC0415

        spec = spec_named(self.spec_name)
        table = load_corpus(self.corpus, self.source)
        knobs: dict[str, Any] = {}
        if chosen(self.folds):
            knobs["folds"] = self.folds
        if chosen(self.trees):
            knobs["n_estimators"] = self.trees
        if chosen(self.learning_rate):
            knobs["learning_rate"] = self.learning_rate
        if chosen(self.tree_depth):
            knobs["max_depth"] = self.tree_depth
        model, report = fit_ranker(table, spec, **knobs)

        payload = export_ranker(model, spec, report)
        artifact = write_artifact(self.model_dir / f"{spec.name}.json", payload)
        # The card is committed and the trees are not: a model that refits from its corpus is better
        # regenerated than carried in every clone forever. What has to survive in git is the
        # measurement, because without the card an AUROC is a number in a chat log.
        card = self.model_dir / f"{spec.name}.card.json"
        card.write_text(json.dumps({
            "kind": payload["kind"], "artifact_version": payload["artifact_version"],
            "spec": payload["spec"], "frame": payload["frame"], "features": payload["features"],
            "trees": len(payload["trees"]), "base_rate": payload["base_rate"],
            "model_sha256": payload["sha256"], "card": payload["card"],
            "regenerate": "python -m datagen train-ranker --corpus <corpus.npz>",
        }, indent=2, sort_keys=True), encoding="utf-8")

        return RankerFitReport(
            spec=spec.name, artifact_path=artifact, card_path=card, sha256=payload["sha256"],
            rows=report.rows, groups=report.groups, positives=report.positives,
            auroc=report.auroc, baseline_auroc=report.baseline_auroc,
            precision_at_250=report.precision_at_k[250],
            baseline_precision_at_250=report.baseline_precision_at_k[250],
            summary=report.summary())


@dataclass(frozen=True, slots=True)
class CorpusCheckReport:
    """Is this corpus usable, and what would refuse it."""

    verdict: Any

    @property
    def usable(self) -> bool:
        return bool(self.verdict.usable)

    def render(self, *, verbose: bool = True) -> str:
        from datagen.corpus.gate import format_verdict  # noqa: PLC0415

        return format_verdict(self.verdict, verbose=verbose)

    def as_dict(self) -> dict[str, Any]:
        import dataclasses  # noqa: PLC0415

        return {"usable": self.usable, "verdict": dataclasses.asdict(self.verdict)}


@dataclass
class CorpusCheck:
    """Judge a corpus before anyone trains on it. Opens no cell, loads no model, trains nothing.

    It is meant to run before the training code it protects exists, which is why it lives in
    `datagen` and not beside a model. Whether a corpus is trainable is decidable from the columns
    alone, so the answer costs no GPU hour.

    A serving corpus is the second half and it is optional for a reason. Training data that is
    perfectly learnable against a serving distribution that does not match it is the failure this
    catches, and it can only catch it when both are given.
    """

    corpus: Path
    source: str = DEFAULT_SOURCE
    spec_name: str = DEFAULT_SPEC
    serving: Path | None = None
    serving_source: str = DEFAULT_SOURCE
    serving_spec_name: str | None = None

    @classmethod
    def from_corpus(cls, corpus: str | Path, *, source: str = DEFAULT_SOURCE,
                    spec: str = DEFAULT_SPEC, serving: str | Path | None = None,
                    serving_source: str = DEFAULT_SOURCE,
                    serving_spec: str | None = None) -> "CorpusCheck":
        return cls(corpus=Path(corpus), source=source, spec_name=spec,
                   serving=Path(serving) if serving is not None else None,
                   serving_source=serving_source, serving_spec_name=serving_spec)

    def describe(self) -> str:
        """What would be judged against what, before a file is read. ASCII, zero arguments."""
        return _NEWLINE.join([
            f"corpus check on {self.corpus} (source {self.source})",
            f"  spec           {self.spec_name}",
            f"  serving        {self.serving if self.serving is not None else _NO_SERVING}"])

    def assess(self) -> CorpusCheckReport:
        """Read both corpora, derive what the spec needs, and judge.

        The derived-spec branch is load-bearing. A derived spec's features are computed from the raw
        columns, so they have to exist before anything judges them: a gate that judged the raw
        columns would be judging what the model does not read, and would pass a corpus whose derived
        columns are constant.
        """
        from src.robot.grasping.deep.ranker.features import (  # noqa: PLC0415
            derive_grasp_frame_columns, spec_named,
        )

        from datagen.corpus.gate import assess_corpus  # noqa: PLC0415
        from datagen.corpus.sources import load_corpus  # noqa: PLC0415

        spec = spec_named(self.spec_name)
        train = load_corpus(self.corpus, self.source)
        serving = (load_corpus(self.serving, self.serving_source)
                   if self.serving is not None else None)
        serving_spec = spec_named(self.serving_spec_name) if self.serving_spec_name else None
        if spec.derived or (serving_spec is not None and serving_spec.derived):
            train = derive_grasp_frame_columns(train)
            serving = derive_grasp_frame_columns(serving) if serving is not None else None
        return CorpusCheckReport(
            verdict=assess_corpus(train, spec, serving=serving, serving_spec=serving_spec))
