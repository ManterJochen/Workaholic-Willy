"""The RL data arc as one noun with three verbs, callable without building a command line.

`RecordCollection` joins the four modules of `datagen/rl/`: `perception` replays a rendered scene
as a camera rig, `occupancy` reports whether the RL feature keys are populated and vary, `collect`
grades top-k candidates in physics and joins the outcomes onto them, and `proof` trains on the
records that come out. The collect verb owns the whole sequence of driving the stack, running the
referee and joining, so a library caller reaches the sequence rather than only the pieces.

Occupancy is the verb to run first: it needs no physics, and a feature that never varies cannot be
learned from, so finding that out after a physics pass costs the physics pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover (imported for annotations only)
    from datagen.rl.occupancy import SweepVerdict

__all__ = ["RecordCollection", "CollectionReport", "OccupancyReport", "ProofReport",
           "MaskSource", "DepthSource", "PhysicsEngine"]

MaskSource = Literal["pred", "gt"]
DepthSource = Literal["noisy", "clean"]
PhysicsEngine = Literal["isaac", "mujoco"]

_NEWLINE = chr(10)


@dataclass(frozen=True, slots=True)
class OccupancyReport:
    """What the feature sweep saw, and whether it is usable.

    Here `occupancy` means feature-key coverage, not space. In a package that voxelises point
    clouds and reasons about free volume the word reads as an occupancy grid; it is not one. It
    measures how often each of the fifteen RL feature keys is present, non-null and varying.
    """

    raw: dict[str, Any]
    verdict: "SweepVerdict"

    @property
    def ok(self) -> bool:
        return self.verdict.ok

    def render(self) -> str:
        v = self.verdict
        lines = [f"feature occupancy over {v.rows} candidate row(s)",
                 f"  record-level    {v.live_record}/{v.features} live",
                 f"  per-candidate   {v.live_per_candidate}/{v.per_candidate_features} live "
                 f"(the only ones a pairwise ranker can use)"]
        if not v.ok:
            lines.append(f"  REFUSED: {v.reason}")
        return _NEWLINE.join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "reason": self.verdict.reason, "rows": self.verdict.rows,
                "live_record": self.verdict.live_record, "features": self.verdict.features,
                "live_per_candidate": self.verdict.live_per_candidate,
                "per_candidate_features": self.verdict.per_candidate_features}


@dataclass(frozen=True, slots=True)
class CollectionReport:
    """One pass of collect, then physics, then join: counts plus where the records landed."""

    records_path: Path
    trials: int
    records: int
    held: int
    refused: int
    harness_controls: Any = None
    outcomes: dict[str, int] | None = None

    @property
    def hold_rate(self) -> float:
        return self.held / self.records if self.records else 0.0

    def render(self) -> str:
        lines = [f"{self.trials} trial(s) -> {self.records} record(s), {self.held} held "
                 f"({100 * self.hold_rate:.1f} %), {self.refused} refused",
                 f"wrote {self.records_path}"]
        if self.outcomes:
            lines.append(f"  outcomes {dict(sorted(self.outcomes.items()))}")
        if self.harness_controls is not None:
            lines.append(f"  harness controls: {self.harness_controls}")
        return _NEWLINE.join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {"records_path": str(self.records_path), "trials": self.trials,
                "records": self.records, "held": self.held, "refused": self.refused,
                "hold_rate": self.hold_rate, "outcomes": self.outcomes}


@dataclass(frozen=True, slots=True)
class ProofReport:
    """Does the per-candidate geometry buy anything over the stock 15-key contract?"""

    raw: dict[str, Any]

    def render(self) -> str:
        return _NEWLINE.join(f"  {key:<28} {value}" for key, value in sorted(self.raw.items())
                             if not isinstance(value, (dict, list)))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.raw)


@dataclass
class RecordCollection:
    """A rendered dataset, driven through the real grasping stack into RL training records.

    The features come from the actual stack, which is the whole premise. This builds the service
    through `AutonomousGraspService.from_robot_config`, the same entry point a real cell uses,
    rather than reimplementing the feature computation. A reimplementation would measure itself.

    The mask and depth sources are not cosmetic: `pred` plus `noisy` is what a camera delivers,
    while `gt` plus `clean` measures the stack against information no sensor has.
    """

    dataset: Path
    scene_limit: int = 12
    top_k: int = 5
    mask_source: MaskSource = "pred"
    depth_source: DepthSource = "noisy"
    engine: PhysicsEngine = "isaac"

    @classmethod
    def from_dataset(cls, dataset: str | Path, *, scenes: int = 12, top_k: int = 5,
                     mask_source: MaskSource = "pred", depth_source: DepthSource = "noisy",
                     engine: PhysicsEngine = "isaac") -> "RecordCollection":
        return cls(dataset=Path(dataset), scene_limit=scenes, top_k=top_k,
                   mask_source=mask_source, depth_source=depth_source, engine=engine)

    @classmethod
    def from_config(cls, config: Any, *, name: str, out_root: str | Path | None = None,
                    **kwargs: Any) -> "RecordCollection":
        """Locate the dataset from `config.output.root`, the way every sibling command does."""
        root = Path(out_root) if out_root is not None else Path(config.output.root)
        return cls.from_dataset(root / name, **kwargs)

    def describe(self) -> str:
        """What this would drive, before a simulator starts. Zero arguments, ASCII."""
        return _NEWLINE.join([
            f"rl collection over {self.dataset}",
            f"  scenes         {self.scene_limit}",
            f"  candidates     top {self.top_k} per scene",
            f"  masks / depth  {self.mask_source} / {self.depth_source}",
            f"  referee        {self.engine}"])

    def measure_occupancy(self, *, record_path: str | Path | None = None) -> OccupancyReport:
        """Drive the stack and report what the fifteen feature keys did. No physics, dummy arm.

        Run this before collecting: a feature that never varies cannot be learned from, and finding
        that out after a physics pass costs the physics pass.
        """
        from datagen.rl.occupancy import measure_occupancy, sweep_verdict  # noqa: PLC0415

        raw = measure_occupancy(
            self.dataset, scene_limit=self.scene_limit, mask_source=self.mask_source,
            depth_source=self.depth_source,
            record_path=Path(record_path) if record_path is not None else None)
        return OccupancyReport(raw=raw, verdict=sweep_verdict(raw))

    def collect(self, *, physics: bool = True) -> CollectionReport:
        """Collect candidates, grade them, and join the outcomes into one record per candidate.

        Three steps in one call: drive the stack, run the referee, join.

        `physics=False` is the join-only path, and it needs `rl_physics.jsonl` to already exist.
        It re-joins a previous grading rather than skipping the grading.
        """
        from datagen.rl.collect import collect_trials, write_records  # noqa: PLC0415

        trials, index, outcomes = collect_trials(
            self.dataset, scene_limit=self.scene_limit, top_k=self.top_k,
            mask_source=self.mask_source, depth_source=self.depth_source)
        physics_path = self.dataset / "rl_physics.jsonl"
        if physics:
            from datagen.grasps.physics import run_physics_sample  # noqa: PLC0415

            run_physics_sample(self.dataset, trials=trials, out_name="rl_physics.jsonl",
                               engine=self.engine)
        records_path = self.dataset / "rl_records.jsonl"
        summary = write_records(index, physics_path, records_path)
        return CollectionReport(
            records_path=records_path, trials=len(trials), records=summary["records"],
            held=summary["held"], refused=summary["refused"],
            harness_controls=summary.get("harness_controls"), outcomes=dict(outcomes))

    def prove(self, *, records_path: str | Path | None = None,
              holdout: float = 0.3) -> ProofReport:
        """Compare the stock 15-key contract against the per-candidate geometry keys."""
        from datagen.rl.proof import run_proof  # noqa: PLC0415

        path = Path(records_path) if records_path is not None else self.dataset / "rl_records.jsonl"
        return ProofReport(raw=run_proof(path, holdout=holdout))
