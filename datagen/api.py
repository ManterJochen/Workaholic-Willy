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
    """Describe a cell, then generate data from it. Construct with a factory, then call a verb."""

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
        """Raw handles. The `from_components` analogue: the caller already has a config."""
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
        """A JSON config with overrides on top. The `from_robot_config` analogue.

        The merge is `merged_settings`, shared with `__main__._config(args)`, so this door and the
        command line cannot drift. `--engine` names a key nested under `render`: a caller who writes
        `{"engine": "mujoco"}` at the top level gets a config that keeps the engine it already had,
        without a word, and the only trace is a `provenance.json` naming a renderer nobody asked for.

        `overrides` is applied last and is a plain nested mapping, merged one level deep, so a
        caller can hand in a `camera_rig` block without restating the rest of it.
        """
        base, applied = merged_settings(path, scenes=scenes, seed=seed, domain=domain,
                                        engine=engine, overrides=overrides)
        built = cls.from_config(DatagenConfig(**base), name=name, out_root=out_root)
        built.notes = applied
        return built

    # ------------------------------------------------------------------ opt-in side channel
    def attach_stage_listener(
            self, listener: "Callable[[StageResult], None] | None") -> None:
        """Called once per finished step. `None` detaches.

        Off by default, so a build started without one behaves exactly as it does without this seam.
        Called inline on the thread running the build, so a slow listener slows the build: hand the
        result to a queue.
        """
        self._on_stage = listener

    # ------------------------------------------------------------------ where things land
    @property
    def root(self) -> Path:
        """The dataset directory: `out_root` when given, otherwise `output.root`, plus the name."""
        base = self.out_root if self.out_root is not None else Path(self.config.output.root)
        return base / self.name

    def describe(self) -> str:
        """What this build will do, before it costs anything.

        A render is hours. A mistyped engine, a cell whose cameras are not the ones the caller
        meant, or a scene count off by a decimal place should be visible in the first second.
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

    # ------------------------------------------------------------------ the verbs
    def render(self, *, headless: Any = UNSET, preview: Any = UNSET,
               limit: Any = UNSET) -> StageResult:
        """Scenes to images and geometry. The expensive step, and the only one that wants a GPU.

        No defaults here: every parameter starts at `UNSET`. `build_dataset` declares the defaults,
        and repeating them would make this layer a second, competing source for the same facts.
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
        """Images and geometry to grasp labels. Minutes, no GPU, re-runnable on its own.

        `density` defaults to `default` here and to `grid` for a screen, and the two produce label
        counts an order of magnitude apart on the same object. Whether the caller chose is decided
        by `None`, never by comparing the value to the flag's own default string: typing the default
        explicitly has to stay distinguishable from not typing it.
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
        """Labels to the point-cloud corpus a generator eats.

        Every keyword `build_cloud_corpus` takes is forwarded, because those keywords are what
        defines the corpus: `mask_suffix` chooses between ground-truth masks and the ones a real
        camera would produce, `voxel_mm` decides how much geometry survives at all, and the
        environment settings decide what surrounds the object.

        A missing parameter is quieter than a wrong one. Nothing raises: a caller simply gets the
        default and a corpus that is not the one they meant, stamped with a name that says nothing
        about which.
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
        """Render, label and optionally extract the corpus, stopping at the first step that fails.

        It stops rather than continuing. Labelling a directory nothing rendered into produces a
        report saying zero labels, which reads like a labelling problem and is not one. The first
        failure is the one worth reporting.
        """
        stages: list[StageResult] = [self.render(headless=headless)]
        if stages[-1].ok:
            stages.append(self.label(density=density))
        if stages[-1].ok and corpus_out is not None:
            stages.append(self.clouds(corpus_out))
        return DatasetReport(root=self.root, stages=tuple(stages))

    def _record(self, result: StageResult) -> StageResult:
        if self._on_stage is not None:
            self._on_stage(result)
        return result
