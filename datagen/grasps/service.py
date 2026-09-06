"""Asking the simulator whether a grasp holds, as one noun with two verbs and no command line.

`run_physics_sample` and `paired_trials` are both typed callables. What lived only in argparse
handlers was the composition and, more importantly, the output-file rule that keeps three different
questions from sharing one answer file.

The paired comparison is the promotion gate. `paired_trials` is the only instrument here that can
compare two generators without object difficulty sitting inside the comparison: `sample_trials`
gives each arm its own strata, so the two arms land on different objects, and in a corpus where some
objects fail for reasons that have nothing to do with which generator proposed the grasp, that
variance is not small. A comparison that is not paired is not a comparison of the generators.

An arm that proposed nothing is a refusal, not a tie. Objects where any arm proposed nothing are
dropped from the draw and counted, because "arm B held 60 % of what it proposed" says nothing
without "and it proposed on 40 % fewer objects". :class:`PhysicsReport` carries both halves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

if TYPE_CHECKING:  # pragma: no cover (imported for annotations only)
    from datagen.config import DatagenConfig

__all__ = ["PhysicsSampling", "PhysicsReport", "physics_output_name", "PhysicsEngine"]

PhysicsEngine = Literal["isaac", "mujoco"]

_NEWLINE = chr(10)


def physics_output_name(*, proposals: bool = False,
                        paired_arms: Sequence[str] | None = None) -> str:
    """Which file this run writes, given which question it asks.

    Three questions, three files, and the separation is not tidiness. A proposal verdict and a label
    verdict answer different questions: the first asks whether the model's own proposal held, the
    second whether a grasp the geometry calls valid held. A shared file lets one be quoted as the
    other. A paired comparison is a third question again, and its file names the arms so that two
    comparisons of different arms cannot merge.
    """
    if paired_arms:
        return f"grasp_physics_paired_{'_vs_'.join(paired_arms)}.jsonl"
    return "grasp_physics_proposals.jsonl" if proposals else "grasp_physics.jsonl"


@dataclass(frozen=True, slots=True)
class PhysicsReport:
    """What the simulator said, per stratum.

    Per source, never pooled: the draw is stratified, so a pooled hold rate describes the sampler
    rather than the data.
    """

    raw: dict[str, Any]
    out_name: str
    paired_arms: tuple[str, ...] = ()

    @property
    def trials(self) -> int:
        return int(self.raw.get("trials", 0))

    @property
    def drawn(self) -> int:
        return int(self.raw.get("drawn", 0))

    @property
    def refused(self) -> int:
        return int(self.raw.get("refused", 0))

    @property
    def held(self) -> int:
        return sum(int(b["held"]) for b in self.raw.get("by_source", {}).values())

    @property
    def hold_rate(self) -> float:
        """Pooled, and only meaningful for the proposal run, where the draw is the model's own
        proposals rather than a stratified sample. Read `by_source` for anything else."""
        return self.held / self.trials if self.trials else 0.0

    def render(self) -> str:
        lines = []
        for source, bucket in sorted(self.raw.get("by_source", {}).items()):
            lines.append(f"  {source:<28} {bucket['held']:>4}/{bucket['trials']:<4} held "
                         f"({bucket['hold_rate'] * 100:5.1f}%)   {bucket['refused']:>3} refused")
        lines.append(f"  {self.drawn} drawn, {self.trials} scored, {self.refused} refused "
                     "(a refused trial is a scene that was not the labelled scene, not a grasp "
                     "that failed)")
        if self.paired_arms:
            lines.append("  PAIRED: every object above contributed one trial to every arm, so the "
                         "arms are the only thing that varies. An unpaired rate is not comparable "
                         "to these.")
        lines.append(f"  -> {self.out_name}")
        return _NEWLINE.join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {"out_name": self.out_name, "trials": self.trials, "drawn": self.drawn,
                "refused": self.refused, "held": self.held, "hold_rate": self.hold_rate,
                "paired_arms": list(self.paired_arms),
                "by_source": dict(self.raw.get("by_source", {}))}


@dataclass
class PhysicsSampling:
    """A labelled dataset, and the simulator that grades grasps on it. Needs a running cell."""

    dataset: Path
    engine: PhysicsEngine = "isaac"
    headless: bool = True

    @classmethod
    def from_config(cls, config: "DatagenConfig", *, name: str,
                    out_root: str | Path | None = None, engine: PhysicsEngine = "isaac",
                    headless: bool = True) -> "PhysicsSampling":
        root = Path(out_root) if out_root is not None else Path(config.output.root)
        return cls(dataset=root / name, engine=engine, headless=headless)

    @classmethod
    def from_dataset(cls, dataset: str | Path, *, engine: PhysicsEngine = "isaac",
                     headless: bool = True) -> "PhysicsSampling":
        return cls(dataset=Path(dataset), engine=engine, headless=headless)

    def describe(self) -> str:
        """What would be graded and by what, before a simulator starts. ASCII, zero arguments."""
        return _NEWLINE.join([f"physics sampling over {self.dataset}",
                              f"  referee        {self.engine}",
                              f"  display        {'headless' if self.headless else 'gui'}"])

    def sample(self, *, per_class: int = 40, proposals: str | Path | None = None) -> PhysicsReport:
        """Does a grasp the geometry calls valid actually hold?

        Both the accepted and the rejected are sampled, because only the second half can show that
        the reference discriminates: a referee that holds everything it is handed has measured
        nothing.

        `proposals` asks a different question and writes a different file. It grades a model's own
        proposals, and mixing those rows with label verdicts would let one be quoted as the other.
        """
        from datagen.grasps.physics import (  # noqa: PLC0415
            run_physics_sample, trials_from_proposals,
        )

        trials = trials_from_proposals(proposals) if proposals is not None else None
        out_name = physics_output_name(proposals=proposals is not None)
        raw = run_physics_sample(self.dataset, trials=trials, headless=self.headless,
                                 per_class=per_class, engine=self.engine, out_name=out_name)
        return PhysicsReport(raw=raw, out_name=out_name)

    def compare(self, arms: Sequence[str], *, per_class: int = 20) -> PhysicsReport:
        """Which of two or more generators actually holds more, on the same objects.

        The promotion gate. Raises `ValueError` when fewer than two arms are named, and when no
        object carried a candidate from every arm, because an empty pairing is not a tie.
        """
        from datagen.grasps.physics import paired_trials, run_physics_sample  # noqa: PLC0415

        named = [a.strip() for a in arms if a.strip()]
        if len(named) < 2:
            raise ValueError(f"a comparison needs at least two arms, got {named}")
        trials = paired_trials(self.dataset, configs=named, per_class=per_class)
        if not trials:
            raise ValueError("no object carried a candidate from every arm, so there is nothing to "
                             "compare; an empty pairing is not a tie")
        out_name = physics_output_name(paired_arms=named)
        raw = run_physics_sample(self.dataset, trials=trials, headless=self.headless,
                                 engine=self.engine, out_name=out_name)
        return PhysicsReport(raw=raw, out_name=out_name, paired_arms=tuple(named))
