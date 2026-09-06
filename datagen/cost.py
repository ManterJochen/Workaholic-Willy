"""What a dataset will cost, before anybody starts it: hours, gigabytes and refused scenes.

Scene generation is the operator's job, so "how big a corpus should I build?" is the first question
asked, and on the Isaac path a guess is only settled by the run itself.

Every coefficient below is measured and carries what it was measured over. Where a number
extrapolates past what was measured, the report says so, because an estimate that cannot state how
far it is reaching is indistinguishable from a guess.

The three traps this module exists to keep an operator out of:

  1. Per-shard time is not throughput. Concurrent shards share one GPU, so the seconds per scene
     one shard sees and the seconds per scene the whole run achieves are different numbers. Quoting
     the first over-estimates a parallel run and quoting the second under-estimates a
     single-process one, so both are reported and both are labelled.
  2. Scenes requested are not scenes usable. The engine-free backend refuses `pile` by name, and
     even Isaac drops scenes whose objects never settle. The calculator answers in usable scenes
     and prints the request that reaches them.
  3. Rendering is not the whole bill. Labelling, corpus extraction and training are each a real
     fraction of it, and on the cheap engines they dominate, so an estimate covering the render
     stage alone is worst on exactly the configuration a customer without a GPU will pick.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = ["ENGINE_COSTS", "EngineCost", "Estimate", "StageCost", "estimate"]


@dataclass(frozen=True, slots=True)
class EngineCost:
    """What one scene costs on one engine, and the evidence for it."""

    #: Seconds per scene for one process. For Isaac this is per-shard time, which is what a
    #: single-process run actually experiences.
    seconds_per_scene: float
    #: Seconds per scene in aggregate at `measured_jobs` parallel processes. Not
    #: `seconds_per_scene / jobs`: concurrent shards contend for one GPU, so n of them do not run n
    #: times as fast, and dividing by `jobs` under-states the hours.
    seconds_per_scene_parallel: float
    #: How many parallel processes the parallel figure was measured at. Beyond this, scaling is
    #: extrapolation and the report says so.
    measured_jobs: int
    #: Megabytes one written scene occupies on disk.
    megabytes_per_scene: float
    #: Fraction of requested scenes that come out usable. Below 1.0 because of refused families and
    #: bodies that never settle.
    yield_fraction: float
    #: Where the numbers come from, printed verbatim in the report.
    evidence: str
    #: What this engine cannot do, or "" when there is nothing to warn about.
    caveat: str = ""


#: Measured on an RTX 5080, Windows 11, python 3.11.
#:
#: `isaac`: measured across concurrent shards. The per-shard figure is what one shard sees and the
#:   aggregate is what the whole run achieves; the aggregate is the number to plan with. The spread
#:   between shards under identical contention is not explained, which is why this module reports a
#:   range for Isaac rather than a single figure.
#:
#: `none` / `mujoco`: measured on a timed build of procedural objects, nothing else running. `none`
#:   refuses `pile` by name, which is where its yield fraction comes from.
#:
#: Those two figures are procedural figures and are wrong by an order of magnitude for a mesh
#: corpus. The cause is the raster, not the physics: `none` runs no settle at all and is still the
#: slower of the two, because `render/raster.py` loops over triangles in Python and a scanned mesh
#: carries thousands of faces against a procedural box's twelve. So the cost scales with the
#: triangle count of the assets a config draws, which this module cannot see from a scene count.
#: `_MESH_SECONDS_PER_SCENE` prices the mesh case; the constants here stay for the procedural one,
#: and the warning `estimate` emits says which is being quoted.
#:   Both cheap engines write far less per scene than Isaac because they write depth, masks and
#:   labels but no path-traced RGB. Neither has a measured parallel figure: they are CPU-bound and
#:   the probe was too short to separate scaling from noise, so both report `measured_jobs=1` and
#:   the calculator refuses to pretend parallel helps.

ENGINE_COSTS: Final[dict[str, EngineCost]] = {
    "isaac": EngineCost(
        seconds_per_scene=101.2,
        seconds_per_scene_parallel=34.0,
        measured_jobs=3,
        megabytes_per_scene=0.461,
        yield_fraction=0.990,
        evidence="the v1 corpus itself: 1,981 scenes, 18.73 h wall clock, 913 MB, 3 concurrent shards",
        caveat="needs a local Isaac Sim install and an RTX-class GPU",
    ),
    "mujoco": EngineCost(
        seconds_per_scene=1.46,
        seconds_per_scene_parallel=1.46,
        measured_jobs=1,
        megabytes_per_scene=0.089,
        yield_fraction=1.000,
        evidence="timed 40-scene build, 58.3 s, 40 usable",
        caveat="settles with a solver but rasterises depth only; no photoreal RGB. Mesh contact is "
               "the CONVEX HULL, so concave meshes are split first: run `datagen decompose` before "
               "the build or the render pays seconds per mesh in its own loop, one at a time",
    ),
    "none": EngineCost(
        seconds_per_scene=1.39,
        seconds_per_scene_parallel=1.39,
        measured_jobs=1,
        megabytes_per_scene=0.089,
        yield_fraction=0.750,
        evidence="timed 40-scene build, 41.7 s, 30 usable of 40 (10 refused: `pile`)",
        caveat="no physics at all: objects are seated analytically on their support, and the `pile` "
               "family is refused by name because a pile IS the physics",
    ),
}

#: Seconds per scene the `label-grasps` stage costs, with `_CORPUS_SECONDS_PER_SCENE` and
#: `_CORPUS_MEGABYTES_PER_SCENE` below measured over the same build.
#:
#: All three are pure CPU geometry and none depends on the engine that wrote the scenes, which is
#: why they are one set of constants rather than one set per engine.
_LABEL_SECONDS_PER_SCENE: Final[float] = 0.148

#: What `--density dense` multiplies the label stage by. Priced separately because the density is
#: chosen at label time and this module cannot see the choice from a config.

_DENSE_LABEL_MULTIPLIER: Final[float] = 2.63

#: Seconds to convex-decompose one scanned mesh.
#:
#: Re-measure this over the whole bank, never over a warm-up: mesh sizes there span thousands of
#: faces to tens of thousands, so a handful of assets can be dominated by whichever few are slowest
#: and read several times the true per-asset cost.
#:
#: Paid once per (mesh, settings) pair and cached to disk, so it scales with the number of distinct
#: meshes a config places, not with scenes. Procedural and composite assets cost nothing: their
#: parts are already convex.
_DECOMPOSE_SECONDS_PER_ASSET: Final[float] = 2.85

#: What a scene of scanned meshes costs to render, as opposed to a scene of procedural primitives.
#: Measured on `mujoco` with the decomposition cache warm. Used whenever the config weights a mesh
#: source, because quoting the procedural figure for a mesh corpus promises a night nobody agreed
#: to.
_MESH_SECONDS_PER_SCENE: Final[float] = 16.7
_CORPUS_SECONDS_PER_SCENE: Final[float] = 0.665
_CORPUS_MEGABYTES_PER_SCENE: Final[float] = 0.298

#: Seconds one training scene costs per epoch, measured on an RTX 5080 over the steady-state epochs
#: of a training run.
#:
#: Linear in scenes is an assumption, not a measurement. It holds for the data-loading half and
#: roughly for the forward pass; it ignores that a bigger corpus may want a bigger model. Treat the
#: training line as the right order of magnitude and not as a promise.
_TRAIN_SECONDS_PER_SCENE_EPOCH: Final[float] = 0.1846
_TRAIN_FRACTION: Final[float] = 0.8

#: How many full passes over the corpus a training run with the shipped recipe makes.
#:
#: A fold pass and a refit on everything are two passes and twice the hours, so pricing one
#: under-states the training line by that factor. `--train-folds N` prices N fold passes; the refit
#: is added on top unless `--no-refit`. `--folds` is train-ranker's and is ignored by this command.

_DEFAULT_TRAIN_PASSES: Final[int] = 2


@dataclass(frozen=True, slots=True)
class StageCost:
    """One stage of the pipeline: what it costs in time and disk."""

    name: str
    hours: float
    gigabytes: float
    note: str = ""


@dataclass(frozen=True, slots=True)
class Estimate:
    """The whole bill, plus what the operator has to ask for to get it."""

    usable_scenes: int
    requested_scenes: int
    engine: str
    jobs: int
    stages: tuple[StageCost, ...]
    warnings: tuple[str, ...]

    @property
    def hours(self) -> float:
        return sum(stage.hours for stage in self.stages)

    @property
    def gigabytes(self) -> float:
        return sum(stage.gigabytes for stage in self.stages)


def _render_seconds(cost: EngineCost, scenes: int, jobs: int) -> tuple[float, list[str]]:
    """Render seconds for `scenes` at `jobs` processes, and every caveat that entails."""
    warnings: list[str] = []
    if jobs <= 1:
        return (scenes * cost.seconds_per_scene, warnings)
    if jobs <= cost.measured_jobs:
        # Interpolating inside what was measured. The parallel figure already is the aggregate.
        return (scenes * cost.seconds_per_scene_parallel * cost.measured_jobs / jobs, warnings)
    if cost.measured_jobs <= 1:
        warnings.append(
            f"`{cost.evidence}` was measured single-process; this engine has NO measured parallel "
            f"figure, so --jobs {jobs} is reported as if it were 1. It will be faster than this "
            f"says, by an amount nobody here has measured.")
        return (scenes * cost.seconds_per_scene, warnings)
    warnings.append(
        f"--jobs {jobs} is past the {cost.measured_jobs} this engine was measured at. Beyond that "
        f"point the scaling below is EXTRAPOLATION: contention for one GPU is exactly what stops "
        f"being linear, and the measured 3-shard run already gave up 1 % of perfect scaling.")
    return (scenes * cost.seconds_per_scene_parallel * cost.measured_jobs / jobs, warnings)


def estimate(
    usable_scenes: int,
    *,
    engine: str = "isaac",
    jobs: int = 1,
    label: bool = True,
    corpus: bool = True,
    epochs: int = 0,
    folds: int = 1,
    refit: bool = True,
    dense: bool = False,
    meshes: int = 0,
) -> Estimate:
    """What it costs to end up with `usable_scenes` usable scenes.

    The answer is in usable scenes, which is the question an operator has. A request is not a
    yield: the engine-free backend refuses whole families by name. So the request is computed
    backwards from the engine's yield fraction, and both numbers are reported.
    """
    if usable_scenes <= 0:
        raise ValueError(f"usable_scenes must be positive, got {usable_scenes}")
    if engine not in ENGINE_COSTS:
        raise ValueError(f"unknown engine {engine!r}; expected one of {sorted(ENGINE_COSTS)}")
    if jobs < 1:
        raise ValueError(f"jobs must be at least 1, got {jobs}")

    cost = ENGINE_COSTS[engine]
    requested = int(round(usable_scenes / cost.yield_fraction))
    warnings: list[str] = []
    if cost.caveat:
        warnings.append(f"{engine}: {cost.caveat}")
    if cost.yield_fraction < 0.999:
        warnings.append(
            f"{engine} yields {cost.yield_fraction:.0%}: you must REQUEST {requested} scenes to end "
            f"up with {usable_scenes}. The estimate below already asks for {requested}.")

    seconds, render_warnings = _render_seconds(cost, requested, jobs)
    warnings.extend(render_warnings)

    per_scene = cost.seconds_per_scene_parallel if jobs > 1 else cost.seconds_per_scene
    if meshes > 0 and engine != "isaac":
        # The procedural figure is not the mesh figure, and quoting it for a mesh corpus understates
        # the job by an order of magnitude: the rasteriser loops over triangles in Python, so the
        # cost belongs to the assets rather than to the engine. `meshes > 0` is the only signal this
        # module has that the draw is scanned.
        seconds = requested * _MESH_SECONDS_PER_SCENE / max(1, jobs)
        per_scene = _MESH_SECONDS_PER_SCENE / max(1, jobs)
        warnings.append(
            f"this config draws SCANNED meshes, so the render is priced at "
            f"{_MESH_SECONDS_PER_SCENE:.1f} s/scene (measured), not the {cost.seconds_per_scene:.2f} "
            f"the engine costs on procedural objects. Shard the build across processes to divide it: "
            f"the box has no parallel mode inside one run.")
    stages = [StageCost(
        "render", seconds / 3600.0, requested * cost.megabytes_per_scene / 1000.0,
        f"{per_scene:.2f} s/scene at {jobs} job(s)")]

    if engine == "mujoco" and meshes > 0:
        # Paid once per mesh rather than per scene, and on a full bank that is hours, so leaving it
        # out would understate the job an operator is about to start. Zero when the config places no
        # scanned meshes.
        stages.append(StageCost(
            "decompose", meshes * _DECOMPOSE_SECONDS_PER_ASSET / 3600.0, 0.0,
            f"{meshes} scanned mesh/meshes at {_DECOMPOSE_SECONDS_PER_ASSET:.1f} s each, ONCE, "
            f"cached to disk; run `datagen decompose --jobs N` before the build"))
    if label:
        seconds_per_scene = _LABEL_SECONDS_PER_SCENE * (_DENSE_LABEL_MULTIPLIER if dense else 1.0)
        stages.append(StageCost(
            "label-grasps", usable_scenes * seconds_per_scene / 3600.0, 0.0,
            "closed-form geometry, CPU, writes into the dataset"
            + (f"; `--density dense`, {_DENSE_LABEL_MULTIPLIER:.1f}x the shipped density" if dense
               else "")))
    if corpus:
        stages.append(StageCost(
            "build-cloud-corpus", usable_scenes * _CORPUS_SECONDS_PER_SCENE / 3600.0,
            usable_scenes * _CORPUS_MEGABYTES_PER_SCENE / 1000.0,
            "the .npz the trainer reads; kept alongside the dataset"))
    if epochs > 0:
        train_scenes = usable_scenes * _TRAIN_FRACTION
        passes = max(1, int(folds)) + (1 if refit else 0)
        stages.append(StageCost(
            "train", passes * train_scenes * epochs * _TRAIN_SECONDS_PER_SCENE_EPOCH / 3600.0, 0.0,
            f"{passes} pass(es) x {epochs} epochs over {train_scenes:.0f} training scenes "
            f"({int(folds)} fold(s)" + (" + refit)" if refit else ")")))
        if passes != 1:
            warnings.append(
                f"the trainer makes {passes} full passes at these settings, not one; "
                f"{int(folds)} fold(s)" + (" plus a refit on everything. " if refit else ". ")
                + "Pricing a single pass understated this line by that factor, which is what an "
                  "earlier version of this calculator did.")
        warnings.append(
            "the training line assumes cost is LINEAR in corpus size. That is an assumption, not a "
            "measurement; it was read off one run at one corpus size, and a bigger corpus may want "
            "a bigger model.")

    return Estimate(usable_scenes=usable_scenes, requested_scenes=requested, engine=engine,
                    jobs=jobs, stages=tuple(stages), warnings=tuple(warnings))


def format_estimate(result: Estimate) -> str:
    """The report, as the CLI prints it."""
    lines = [
        f"{result.usable_scenes} usable scene(s) on `{result.engine}` at {result.jobs} job(s)",
        f"  request {result.requested_scenes} to get them"
        + ("" if result.requested_scenes == result.usable_scenes else "   <- not the same number"),
        "",
        f"  {'stage':22} {'hours':>8} {'GB':>8}  note",
    ]
    for stage in result.stages:
        lines.append(f"  {stage.name:22} {stage.hours:8.2f} {stage.gigabytes:8.2f}  {stage.note}")
    lines.append(f"  {'TOTAL':22} {result.hours:8.2f} {result.gigabytes:8.2f}")

    if result.hours > 8.0:
        lines.append(f"\n  That is {result.hours / 24.0:.1f} day(s). `datagen build` RESUMES: a run "
                     f"that is interrupted continues from the scenes already written.")
    if result.warnings:
        lines.append("")
        for warning in result.warnings:
            lines.append(f"  ! {warning}")
    lines.append(f"\n  evidence: {ENGINE_COSTS[result.engine].evidence}")
    return "\n".join(lines)
