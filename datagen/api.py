"""Generating a dataset from code, in the idiom of `AutonomousGraspService`.

A customer writes their own script: they describe their cell, they run the chain, they read a typed
report.

    from datagen.api import DatasetBuild
    from datagen.config import CameraSpec

    build = DatasetBuild.from_file(
        "my_cell.json", name="run_01", scenes=2000, engine="mujoco",
        overrides={"camera_rig": {"cameras": (
            CameraSpec(name="left",  position_mm=(100.0, -400.0, 700.0)),
            CameraSpec(name="right", position_mm=(100.0,  400.0, 700.0)),
            CameraSpec(name="eih",   mount="wrist"),
        )}})

    print(build.describe())          # what it will do, before it costs anything
    report = build.run()             # render, then label, then clouds
    print(report.render())           # what it did

No YAML is required and none is invented. `DatagenConfig` is a frozen Pydantic tree and constructing
one by hand stays first-class; `from_file` exists because a lot of settings live in a file already.
This is the same arrangement the sibling training package settled on.

This is not a second pipeline. `build_dataset`, `label_dataset` and `build_cloud_corpus` keep their
signatures and their modules. This wraps them, owns the glue a CLI handler would otherwise own
privately, and returns something typed. Every step is separately callable, because a customer
re-labelling an existing dataset should not have to re-render it: labelling is minutes and rendering
is hours.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from src.contracts.options import UNSET, chosen
from datagen.config import DatagenConfig

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from collections.abc import Callable

    from datagen.cost import Estimate
    from datagen.verify import VerifyReport

__all__ = ["DatasetBuild", "DatasetReport", "Stage", "StageResult", "merged_settings"]


def merged_settings(path: str | Path | None = None, *, scenes: int | None = None,
                    seed: int | None = None, domain: str | None = None,
                    engine: str | None = None,
                    overrides: Mapping[str, Any] | None = None,
                    ) -> tuple[dict[str, Any], dict[str, Any]]:
    """The settings a config is built from, and a record of what was layered on top.

    One implementation, two doors: `__main__._config(args)` calls this, and so does
    `DatasetBuild.from_file`. Two copies of a merge is two behaviours waiting to disagree, and the
    one place that knows `--engine` becomes the nested `render.engine` has to be reachable from both.
    """
    base: dict[str, Any] = {}
    if path is not None:
        base = json.loads(Path(path).read_text(encoding="utf-8"))
    applied: dict[str, Any] = {}
    for key, value in (("scenes", scenes), ("seed", seed), ("domain", domain)):
        if value is not None:
            base[key] = value
            applied[key] = value
    if engine:
        # Nested, and this is the whole reason the function is shared. `--engine` names a key that
        # lives under `render`, so a caller who sets it at the top level gets a config that keeps
        # the engine it already had, without a word, and the only trace is a `provenance.json`
        # naming a renderer nobody asked for. Nothing downstream compares those two.
        render = dict(base.get("render") or {})
        render["engine"] = engine
        base["render"] = render
        applied["render.engine"] = engine
    for key, value in (overrides or {}).items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
        applied[key] = "overridden"
    return base, applied

NEWLINE = chr(10)


# The sentinel comes from `contracts/`, and importing it rather than declaring one is the point:
# `UNSET` is compared by identity, so two declarations are two different answers to "did the caller
# choose this". `contracts/options.py` is the one declaration, and it is stdlib-only.
#
# `chosen()` there is a `TypeGuard`, so it narrows for a type checker where `x is UNSET` narrows
# nothing. The function below is a different thing with almost the same name: it takes many
# candidates and returns the ones to forward. It is named for what it decides.


def _forwarded(**candidates: Any) -> dict[str, Any]:
    """Only the arguments the caller actually chose, as kwargs for the wrapped function."""
    return {name: value for name, value in candidates.items() if chosen(value)}

class Stage(StrEnum):
    """The three steps that turn a described cell into something a model can train on."""

    #: Scenes to images and geometry. Hours, and the only step that needs a GPU.
    RENDER = "render"
    #: Images and geometry to grasp labels. Minutes, pure geometry, re-runnable.
    LABEL = "label"
    #: Labels to the point-cloud corpus a generator eats.
    CLOUDS = "clouds"


@dataclass(frozen=True, slots=True)
class StageResult:
    """What one step did, and whether the next one may run."""

    stage: Stage
    ok: bool
    summary: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True, slots=True)
class DatasetReport:
    """What a build did, in the shape a program can branch on."""

    root: Path
    stages: tuple[StageResult, ...]

    @property
    def succeeded(self) -> bool:
        return bool(self.stages) and all(step.ok for step in self.stages)

    def result(self, stage: Stage) -> StageResult | None:
        """One step's result, or None when it never ran."""
        return next((step for step in self.stages if step.stage is stage), None)

    def failure_summary(self) -> str:
        """One line naming the first step that stopped, and why. Empty when everything ran."""
        for step in self.stages:
            if not step.ok:
                return f"{step.stage} did not complete: {step.reason}"
        return ""

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """What happened. What was going to happen is `DatasetBuild.describe`.

        The two do not overlap. A configuration block printed before the run and again after it
        makes a reader check whether the two differ, which is work the report should have done.

        ASCII only: this reaches a Windows console under cp1252, where a non-ASCII character in a
        printed line raises `UnicodeEncodeError`.
        """
        lines = [f"  dataset    {self.root}"]
        for step in self.stages:
            mark = "ok " if step.ok else "NOT"
            detail = step.reason or ", ".join(f"{k} {v}" for k, v in list(step.summary.items())[:3])
            lines.append(f"  {str(step.stage):10s} {mark}  {detail}")
        summary = self.failure_summary()
        if summary:
            lines.append(f"  outcome    {summary}")
        return NEWLINE.join(lines)


@dataclass
class DatasetBuild:
    """Describe a cell's dataset, then generate it: scenes, images and depth, grasp labels, and the point-cloud corpus a
    grasp generator trains on.

    ```python
    build = DatasetBuild.from_file(name="my_parts", scenes=200, seed=0, engine="mujoco")
    print(build.describe())        # what it will do, before it costs anything
    print(build.cost())            # hours and gigabytes
    report = build.run("corpora/my_parts")
    print(report)
    ```

    Build it with :meth:`from_file` (the usual way) or :meth:`from_config`, then call a verb: :meth:`run` does render,
    label and clouds in one go and stops at the first step that fails.

    Attributes:
        config (DatagenConfig): The dataset's settings: scenes, seed, domain, engine, cameras, output.
        name (str): The dataset's name; its folder under ``out_root``.
        out_root (Path | None): Where datasets go; ``None`` is the config's ``output.root`` (default: None).
        notes (Mapping[str, Any]): What a recipe or a file override supplied, for a caller that logs it (default: {}).
    """

    config: DatagenConfig
    name: str
    out_root: Path | None = None
    #: What a recipe or a file override supplied, for a caller that wants to log it.
    notes: Mapping[str, Any] = field(default_factory=dict)
    _on_stage: "Callable[[StageResult], None] | None" = field(default=None, repr=False)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, config: DatagenConfig, *, name: str,
                    out_root: str | Path | None = None) -> "DatasetBuild":
        """A build from a config you already have.

        Args:
            config (DatagenConfig): The dataset's settings.
            name (str): The dataset's name.
            out_root (str | Path | None): Where datasets go; ``None`` is ``config.output.root`` (default: None).

        Returns:
            DatasetBuild: The build; nothing has run.
        """
        return cls(config=config, name=name,
                   out_root=Path(out_root) if out_root is not None else None)

    @classmethod
    def from_file(cls, path: str | Path | None = None, *, name: str,
                  scenes: int | None = None,
                  seed: int | None = None,
                  domain: str | None = None,
                  engine: str | None = None,
                  overrides: Mapping[str, Any] | None = None,
                  out_root: str | Path | None = None) -> "DatasetBuild":
        """A build from a JSON config with overrides on top, merged as ``datagen`` on the command line merges them, so
        this door and the command line cannot drift.

        Args:
            path (str | Path | None): The JSON config; ``None`` is the shipped default (default: None).
            name (str): The dataset's name.
            scenes (int | None): How many scenes to request; ``None`` keeps the config's (default: None).
            seed (int | None): The seed every scene is laid out from; ``None`` keeps the config's (default: None).
            domain (str | None): The scene domain: ``"cell"`` (the cell's own robot and cameras) or ``"tabletop"``
                (general-purpose data, no robot); ``None`` keeps the config's (default: None).
            engine (str | None): What settles and renders the scenes: ``"isaac"``, ``"mujoco"`` or ``"none"`` (the
                engine-free backend); set as ``render.engine``. ``None`` keeps the config's (default: None).
            overrides (Mapping[str, Any] | None): A nested mapping applied last, merged one level deep, so a
                ``camera_rig`` block needs no restating of the rest (default: None).
            out_root (str | Path | None): Where datasets go (default: None).

        Returns:
            DatasetBuild: The build; nothing has run.

        Raises:
            FileNotFoundError: ``path`` names no file.
            ValueError: The merged settings do not validate (pydantic's ``ValidationError`` is one).
        """
        base, applied = merged_settings(path, scenes=scenes, seed=seed, domain=domain,
                                        engine=engine, overrides=overrides)
        built = cls.from_config(DatagenConfig(**base), name=name, out_root=out_root)
        built.notes = applied
        return built

    # ------------------------------------------------------------------ opt-in side channel
    def attach_stage_listener(
            self, listener: "Callable[[StageResult], None] | None") -> None:
        """Be told as each step finishes. Called inline on the thread running the build, so a slow listener slows it:
        hand the result to a queue.

        Args:
            listener (Callable[[StageResult], None] | None): Called once per finished step; ``None`` detaches.
        """
        self._on_stage = listener

    # ------------------------------------------------------------------ where things land
    @property
    def root(self) -> Path:
        """The dataset directory: `out_root` when given, otherwise `output.root`, plus the name."""
        base = self.out_root if self.out_root is not None else Path(self.config.output.root)
        return base / self.name

    def describe(self) -> str:
        """What this build will do, before it costs anything: a mistyped engine, cameras that are not the ones meant, or
        a scene count off by a decimal place show in the first second.

        Returns:
            str: The settings that matter, one per line.
        """
        rig = self.config.camera_rig
        if rig.cameras is not None:
            views = ", ".join(f"{c.name}({c.mount})" for c in rig.cameras)
        else:
            views = ", ".join(rig.views)
        lines = [
            f"  dataset    {self.root}",
            f"  scenes     {self.config.scenes}, seed {self.config.seed}, domain {self.config.domain}",
            f"  engine     {self.config.render.engine}",
            f"  arm        {self.config.render.arm.mode} ({self.config.render.arm.robot_model})",
            f"  cameras    {views}",
        ]
        if self.notes:
            lines.append("  overrides  "
                         + " ".join(f"{k}={v}" for k, v in sorted(self.notes.items())))
        if self.config.render.arm.wrist_camera_mount is not None:
            lines.append("  wrist mount declared by the caller: "
                         f"{self.config.render.arm.wrist_camera_mount.source}")
        return NEWLINE.join(lines)

    def cost(self, *, scenes: int | None = None, engine: str | None = None, jobs: int = 1,
             label: bool = True, corpus: bool = True, epochs: int = 0, folds: int = 1,
             refit: bool = True, dense: bool = False) -> "Estimate":
        """Hours, gigabytes and the request a yield needs, for this dataset; reads and writes nothing.

        Args:
            scenes (int | None): The usable scenes to price; ``None`` is the build's own count (default: None).
            engine (str | None): The engine to price (``"isaac"``, ``"mujoco"``, ``"none"``); ``None`` the build's own
                (default: None).
            jobs (int): Render jobs in parallel (default: 1).
            label (bool): Include labelling (default: True).
            corpus (bool): Include extracting the corpus (default: True).
            epochs (int): Training epochs to include; 0 prices no training (default: 0).
            folds (int): Cross-validation folds of that training (default: 1).
            refit (bool): Include the refit on all data after the folds (default: True).
            dense (bool): Price dense labels instead of the default density (default: False).

        Returns:
            Estimate: Hours, gigabytes, the scenes to request for the usable ones, and warnings; ``print`` shows the
                report ``datagen cost`` prints.

        Raises:
            ValueError: A count that is not positive, an unknown engine, or ``jobs`` under 1.
        """
        from datagen.cost import estimate_for_config  # noqa: PLC0415 (a leaf, only for this verb)

        return estimate_for_config(self.config, scenes=scenes, engine=engine, jobs=jobs,
                                   label=label, corpus=corpus, epochs=epochs, folds=folds,
                                   refit=refit, dense=dense)

    # ------------------------------------------------------------------ the verbs
    def render(self, *, headless: Any = UNSET, preview: Any = UNSET,
               limit: Any = UNSET) -> StageResult:
        """Scenes to images and geometry: the expensive step, and the only one that wants a GPU.

        Args:
            headless (Any): Run the engine without a window; unset is ``True`` (default: UNSET).
            preview (Any): Write preview images beside the scenes; unset is ``False`` (default: UNSET).
            limit (Any): Render only this many scenes; unset renders the config's count (default: UNSET).

        Returns:
            StageResult: ``ok``, the step's ``summary`` (what was rendered) and, on a failure, its ``reason``.
        """
        from datagen.build import build_dataset  # noqa: PLC0415

        summary = build_dataset(self.config, name=self.name,
                                out_root=str(self.out_root) if self.out_root else None,
                                **_forwarded(headless=headless, preview=preview, limit=limit))
        rendered = int(summary.get("by_status", {}).get("ok", 0))
        return self._record(StageResult(
            stage=Stage.RENDER, ok=rendered > 0, summary=summary,
            reason="" if rendered else "no scene rendered; every one was refused or failed"))

    def label(self, *, limit: Any = UNSET, label_budget: Any = UNSET,
              density: str | None = None, jaw: Any = UNSET) -> StageResult:
        """Images and geometry to grasp labels: minutes, no GPU, re-runnable on its own.

        Args:
            limit (Any): Label only this many scenes; unset labels all (default: UNSET).
            label_budget (Any): Stop after this many label rows, scenes taken in a balanced order; unset is no budget
                (default: UNSET).
            density (str | None): ``"default"``, ``"dense"`` or ``"grid"``; the counts they produce are an order of
                magnitude apart on the same object. ``None`` is ``"default"`` (default: None).
            jaw (Any): Label for this hand instead of the 2F-85, a procedural jaw or a registry hand by model name, into
                its own ``grasps_jaw_<name>.jsonl``; unset labels for the default (default: UNSET).

        Returns:
            StageResult: ``ok``, the step's ``summary`` (rows, scenes) and, on a failure, its ``reason``.
        """
        from datagen.grasps.labels import DENSITIES, label_dataset  # noqa: PLC0415

        summary = label_dataset(
            self.root,
            **_forwarded(limit=limit, label_budget=label_budget, jaw=jaw,
                      density=DENSITIES[density] if density is not None else UNSET))
        found = int(summary.get("jaw", 0)) + int(summary.get("suction", 0))
        return self._record(StageResult(
            stage=Stage.LABEL, ok=found > 0, summary=summary,
            reason="" if found else "no grasp label was produced for any scene"))

    def clouds(self, out_dir: str | Path, *, scenes: Any = UNSET, physics: Any = UNSET,
               kinds: Sequence[str] | None = None, labels: Any = UNSET,
               mask_suffix: Any = UNSET, voxel_mm: Any = UNSET,
               with_environment: Any = UNSET, environment_voxel_mm: Any = UNSET,
               environment_margin_mm: Any = UNSET) -> StageResult:
        """Labels to the point-cloud corpus a generator trains on: one ``.npz`` per scene, the fused cloud and its grasp
        table.

        Args:
            out_dir (str | Path): Where the corpus goes.
            scenes (Any): Extract only this many scenes; unset all (default: UNSET).
            physics (Any): A physics verdicts file to join into the grasp tables; unset none (default: UNSET).
            kinds (Sequence[str] | None): Which grasps go in, ``("jaw",)`` or ``("jaw", "suction")``; ``None`` is
                ``("jaw",)`` (default: None).
            labels (Any): The label file inside the dataset; unset is ``"grasps.jsonl"`` (``label(jaw=)`` writes
                ``grasps_jaw_<name>.jsonl``) (default: UNSET).
            mask_suffix (Any): Which masks cut the objects: ``"instances"`` (ground truth) or the ones a real camera
                would produce; unset is ``"instances"`` (default: UNSET).
            voxel_mm (Any): The object cloud's voxel size, millimetres; unset is 3.0 (default: UNSET).
            with_environment (Any): Keep what surrounds the object; unset is ``True`` (default: UNSET).
            environment_voxel_mm (Any): The surroundings' voxel size, millimetres; unset is 6.0 (default: UNSET).
            environment_margin_mm (Any): How far around the object the surroundings reach, millimetres; unset is 150.0
                (default: UNSET).

        Returns:
            StageResult: ``ok``, the step's ``summary`` and, on a failure, its ``reason``. Nothing raises for a missing
                setting: it takes the default, so set what defines your corpus.
        """
        from datagen.corpus.clouds import build_cloud_corpus  # noqa: PLC0415

        summary = build_cloud_corpus(
            self.root, out_dir,
            **_forwarded(scenes=scenes, physics=physics, labels=labels,
                         mask_suffix=mask_suffix, voxel_mm=voxel_mm,
                         with_environment=with_environment,
                         environment_voxel_mm=environment_voxel_mm,
                         environment_margin_mm=environment_margin_mm,
                         kinds=tuple(kinds) if kinds is not None else UNSET))
        # The key is `scenes_written`. A missing key returns the fallback rather than raising, so a
        # name invented here reports a perfect corpus as ok=False with "no cloud was written", and
        # nothing anywhere says why.
        written = int(summary.get("scenes_written", 0))
        return self._record(StageResult(
            stage=Stage.CLOUDS, ok=written > 0, summary=summary,
            reason="" if written else "no cloud was written"))

    def run(self, corpus_out: str | Path | None = None, *,
            headless: Any = UNSET, density: str | None = None) -> DatasetReport:
        """Render, label and, given an out dir, extract the corpus, stopping at the first step that fails.

        Args:
            corpus_out (str | Path | None): Where the corpus goes; ``None`` renders and labels only (default: None).
            headless (Any): Run the engine without a window; unset is ``True`` (default: UNSET).
            density (str | None): The label density, as :meth:`label` takes it (default: None).

        Returns:
            DatasetReport: Each step's ``StageResult``, ``succeeded``, and ``failure_summary()`` for the first failure.
        """
        stages: list[StageResult] = [self.render(headless=headless)]
        if stages[-1].ok:
            stages.append(self.label(density=density))
        if stages[-1].ok and corpus_out is not None:
            stages.append(self.clouds(corpus_out))
        return DatasetReport(root=self.root, stages=tuple(stages))

    def verify(self) -> "VerifyReport":
        """Check what was written against itself: labels, masks, poses and pictures agreeing. Opens no engine, so it
        also answers for a dataset built months ago (``verify_dataset(root)`` is the same check by path).

        Returns:
            VerifyReport: Each check and ``ok``; a failed check is returned, never raised.
        """
        from datagen.verify import verify_dataset  # noqa: PLC0415 (a leaf, only for this verb)

        return verify_dataset(self.root)

    def _record(self, result: StageResult) -> StageResult:
        if self._on_stage is not None:
            self._on_stage(result)
        return result
