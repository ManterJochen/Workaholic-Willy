# Synthetic scenes, grasp labels and training corpora (`datagen`)

`datagen` renders physically settled scenes of parts, labels every grasp their geometry admits, and turns the
result into the point-cloud corpus a grasp generator trains on. It is a tool beside the robot: it imports from
`src`, `src` never imports from it, and a test holds that direction.

```python
from willy import DatasetBuild

build = DatasetBuild.from_file(name="first", scenes=8, seed=0, engine="none")
print(build.describe())                  # the dataset, the engine and the cameras, before anything costs
report = build.run(corpus_out="data/clouds/first")
print(report)                            # render, label, clouds; it stops at the first step that fails
```

This runs on a laptop in seconds. `engine="none"` needs nothing beyond this repository, the dataset lands in
`data/datagen/first` (`output.root`, or pass `out_root=`), and a quarter of the scenes come back refused,
because `none` does not build the `pile` family. The same steps from a shell:

```bash
python -m datagen cost --scenes 8 --engine none                  # hours, gigabytes and refused scenes, first
python -m datagen build --name first --scenes 8 --engine none    # render
python -m datagen label-grasps --name first                      # grasp labels from the settled geometry
python -m datagen build-cloud-corpus --name first --corpus-out data/clouds/first
```

`python -m datagen` exits 0 when done, 1 on a problem it names, and 2 on a bad request: a wrong flag, a
config that does not validate, or an input that is not on disk yet. Every subcommand is in
[docs/cli.md](../docs/cli.md) and in `python -m datagen --help`. The files in
[examples/offline/datagen/](../examples/offline/datagen/) take the chain one step at a time, and
[train_your_own_generator.md](../docs/runbooks/train_your_own_generator.md) is the whole route from your own
CAD files to a trained generator.

## The nouns

| Noun | Built by | Verbs | Returns |
|---|---|---|---|
| `DatasetBuild` | `from_file(path, name=..., scenes=..., engine=..., overrides=...)`, `from_config(config, name=...)` | `describe`, `cost`, `render`, `label`, `clouds`, `run`, `verify` | a `StageResult` per step; `run` a `DatasetReport`, `cost` an `Estimate`, `verify` a `VerifyReport` |
| `MeshPreparation` | `from_sources(["gso"])`; every public collection when none is named | `fetch`, `normalise`, `screen`, `decompose`, `why_no_jaw` | one report per verb |
| `PhysicsSampling` | `from_dataset(root, engine="mujoco")` | `sample`, `compare` | `PhysicsReport`: held of shaken, per stratum |
| `GraspEvaluation` | `from_dataset(root)`, in `datagen.eval.service` | `evaluate`, `gate`, `approach_tilt` | the ladder, the regression gate, coverage by tilt |
| `RankerCorpus`, `RankerFit`, `CorpusCheck` | `from_dataset`, `from_corpus`, in `datagen.corpus.service` | `build`, `fit`, `assess` | see [corpus/](corpus/README.md) |
| `RecordCollection` | `from_dataset(root)`, in `datagen.rl.service` | `measure_occupancy`, `collect`, `prove` | RL training records from the real grasping stack |

Six functions go with them. `layout_scene(config, index)` returns one scene's `SceneSpec` without rendering
it, `engine_is_available(name)` returns `(ok, why not)`, `available_sources()` lists the public mesh
collections, `import_from_directory("custom", folder, license="own")` adds your parts to the mesh library,
`verify_dataset(root)` checks a written dataset against itself, and `held_out_assets(corpus)` says which
placeable assets a corpus never trained on. `willy` exports the first three nouns and all six functions; the
others import from the module the table names. `DatasetReport`, `PhysicsReport`, `Estimate`, `VerifyReport`
and `HeldOutReport` print as themselves, and the other reports have `render()` for a person and `as_dict()`
for a program.

A scene is reproducible from the seed, the scene index, the code version and the physics version. Every random
draw happens at layout time, never in the renderer, and layout is numpy only. Units are millimetres and XYZW
quaternions throughout; the conversion to Isaac's metres and WXYZ happens at the render boundary.

## What it refuses

| Refusal | When | What to do |
|---|---|---|
| a `pile` scene on `engine="none"` | always: a pile is the physics, and `none` runs none | ask for a third more scenes (`datagen cost` counts usable ones), or use `mujoco` |
| an engine this machine cannot run | at `build`, before any scene renders | `engine_is_available(name)` names what is missing |
| a concave mesh on `mujoco` | `coacd` is missing, or the operating system blocks it | install it, or see [code-integrity.md](../docs/code-integrity.md) |
| a non-commercial, ShareAlike or NoDerivatives licence | at the licence audit, before a render | draw CC0, CC-BY or your own parts; `python -m datagen audit` checks first |
| your own parts with no licence | `--source custom` without `--license` | declare `own` for parts you designed, or the SPDX id |
| a mesh that cannot be closed into a solid | when it would be drawn or labelled | nothing to set: it is refused by name rather than drawn as a box |
| a fixed camera with no `position_mm` | when the config is built | give its position in millimetres |
| a wrist camera on a config with no arm | when the config is built | pose an arm, or set `camera_rig.allow_unmounted_wrist: true` |
| a declared bin wider than `workspace.half_extents_mm` | when the config is built; an inherited bin only warns | widen the workspace, or shrink `workspace.bin.inner_mm` |
| filling in generated shapes | `assets.refuse_procedural_fallback: true` and none of your parts is placeable | fix the parts the screen refused |

## The three engines

| `render.engine` | physics | images | needs |
|---|---|---|---|
| `isaac`, the default | PhysX | path-traced RGB, depth and masks | Isaac Sim and an RTX-class GPU; run under Isaac's own `python.bat` |
| `mujoco` | the MuJoCo solver | depth and masks, no RGB | `mujoco`, and `coacd` for concave meshes, both in `requirements.txt` |
| `none` | none: objects are seated analytically | depth and masks, no RGB | nothing beyond this repository |

What the two cheaper engines give up:

- **No RGB.** Training on geometry works; anything that needs an image does not.
- **MuJoCo collides a mesh against its convex hull**, so a concave mesh is split into convex parts first.
  `python -m datagen decompose --jobs 8` does it once and caches it; cold, the split runs inside the render
  loop at seconds per mesh. The split recovers part of a concavity, not all of it.
- **On scanned meshes the rasteriser is the cost**, not the solver, so `none` can be slower than `mujoco`
  there. `datagen cost` prices this when the config restricts the mesh draw and warns when it cannot tell.

The engine is stamped into `provenance.json` and into every cloud file, because two engines never settle a
scene identically. Nothing refuses a corpus that mixes engines; the stamp is how you tell them apart. There is
no parallel mode inside one run: shard it and give each shard its own `--corpus-out` directory, because scene
ids count within one dataset and two shards in one directory overwrite each other silently.

Grading a grasp in physics is a separate choice: `physics-sample` and `physics-compare` take
`--physics-engine isaac|mujoco`, and `python -m datagen.rl.collect` takes `--engine isaac|mujoco`. MuJoCo
needs no NVIDIA install. [grasps/](grasps/README.md) says what the referee measures.

## Getting the parts

A fresh checkout has no meshes: `assets/meshes/` is gitignored, and the collections below are
gigabytes under somebody else's licence. A build still runs without them, on generated shapes, and a
grasp number over those is a number about convex primitives. Downloading is one call:

```python
from willy import MeshPreparation, available_sources

for source in available_sources():          # what exists, its size, its licence, opens no connection
    print(source)
print(MeshPreparation.from_sources(["gso"]).fetch(limit=20, report=print))   # a trial slice, ~70 MB
```

An already-present mesh is skipped, so an interrupted fetch resumes, and a mesh whose licence is not
CC0 or CC-BY is refused before it is downloaded rather than after. The same from a shell is
`python -m datagen.assets.fetch --list` and `python -m datagen.assets.fetch gso --limit 20`;
[06_fetch_public_parts.py](../examples/offline/datagen/06_fetch_public_parts.py) is the whole step,
including what to set so a build draws from the meshes instead of from generated shapes.

**Normalise a fetched collection before you raise its weight.** Every mesh is read as metres, and a
collection exported in millimetres arrives a thousand times too large.

## Your own parts

```bash
python -m datagen.assets --fetch --source custom --from ./my_parts --license own --attribution-text "ACME GmbH"
python -m datagen prepare-assets --collection custom --out assets/screens/mine.json
python -m datagen why-no-jaw --from-screen assets/screens/mine.json   # why a part earns no jaw label
```

From Python the import is `import_from_directory("custom", "./my_parts", license="own", attribution="ACME
GmbH")`, and [05_bring_your_own_parts.py](../examples/offline/datagen/05_bring_your_own_parts.py) builds a
dataset from the result.

- **The licence is never defaulted.** It is written to `LICENSE.txt` beside the meshes, where the audit
  reads it back; a mesh dropped into the folder by hand is reported as missing a licence.
- **Six formats are read**, `CUSTOM_SUFFIXES` in [assets/library.py](assets/library.py): `.obj`, `.stl`,
  `.ply`, `.off`, `.glb`, `.gltf`. Not `.dae`.
- **Every mesh is read as metres.** A CAD export in millimetres arrives a thousand times too large, the
  normaliser names it, and `python -m datagen normalise-meshes --collection custom --scale 0.001` converts it.
- **Restricting the meshes does not restrict the scenes.** Set `assets.procedural_weight: 0`,
  `assets.custom_weight: 1` and `assets.refuse_procedural_fallback: true`, or generated shapes fill the
  dataset beside your parts.

| `assets.<source>_weight` | geometry | grasp labels |
|---|---|---|
| `procedural`, the default | parametric kinds, each drawn as a box, a cylinder or a sphere | analytic, whole object |
| `composite` | mug, jug, pan, hammer, bucket: primitives at poses | analytic, per part |
| `gso`, `ycb`, `thingi10k`, `asos`, `objaverse` | public scanned or authored meshes, fetched and never committed | analytic on the closed surface |
| `custom` | your own parts | analytic on the closed surface |

The weights are relative: `procedural 1.0, composite 2.0` draws twice as many composites.
`available_sources()` and `python -m datagen.assets.fetch --list` print each public collection's size and
licence, and whether that licence was verified per model or taken from the collection's terms.
`python -m datagen.assets --check` says what this machine holds.

## Looking at a dataset

`*_depth.png` and `*_instances.png` look black in an image viewer, and that is correct: they are uint16
millimetres and uint16 object ids. `python -m datagen preview --name first` writes coloured views into
`preview/` at the dataset root, each stamped with the range it was scaled to, plus a `HOW_TO_READ.txt`.
`build --preview` writes them as the run goes, so a bad run shows at the third scene rather than the
three hundredth. `python -m datagen verify --name first` checks the labels, masks and images.

## Limits

- **Procedural objects are convex primitives.** `primitive_for_kind` in
  [assets/procedural.py](assets/procedural.py), which the renderer and the labeller both read, draws a
  `flange` or a `bowl` as a box. A grasp number over a procedural-only corpus is a number about convex
  primitives; composite and mesh sources are what extend it.
- **Oversized objects are labelled, not filtered.** An object wider than the jaw is still a suction target,
  so `jaw_graspable` records the fact.
- **With Isaac, `assets.mesh_collision` decides whether PhysX collides the surface the labeller reads.**
  `convexHull` fills every concavity; the default is `convexDecomposition`.

## Status

| Capability | Evidence |
|---|---|
| Layout, the `none` engine, labels, clouds, the corpus gate | measured in simulation: CI runs every file in `examples/offline/datagen/` with nothing attached |
| The whole chain at corpus scale | measured in simulation: a proof run completed and `verify` came back clean |
| The MuJoCo and Isaac physics referees | measured in simulation: each passes the four harness controls; see [grasps/](grasps/README.md) |
| A model trained on this data, at a physical cell | never touched hardware: every image here is rendered |

## Files

| File | Holds |
|---|---|
| [api.py](api.py) | `DatasetBuild`: render, label and clouds from a described cell |
| [config.py](config.py) | `DatagenConfig`, a Pydantic tree of its own, apart from the robot's config |
| [recipes.py](recipes.py) | `CORPUS_RECIPES`, the named configurations `init-config --recipe` writes |
| [cost.py](cost.py) | hours, gigabytes and refused scenes before a run, each coefficient with its evidence |
| [scenes/](scenes/) | `SceneSpec` and the seed-to-scene layout: numpy only, spawn poses only |
| [assets/](assets/) | procedural and composite parts, the mesh library, licensing, fetching, `MeshPreparation` |
| [render/](render/) | the three engines behind `SceneEngine`, and the equivalence gate a new engine has to pass |
| [grasps/](grasps/README.md) | closed-form grasp labels and the physics referee |
| [eval/](eval/) | the instruments that grade a generator: the ladder, its floors, the gate, the sweeps |
| [corpus/](corpus/README.md) | the point clouds and ranker table a model trains on, and the gate that judges them |
| [rl/](rl/) | scenes driven through the real grasping stack into RL records |
| [prompts/](prompts/) | referring expressions in German and English, from `scene.json` alone |
| [heldout.py](heldout.py) | which assets a model has never seen |
| [verify.py](verify.py), [preview.py](preview.py), [verify_robot.py](verify_robot.py) | check a dataset, draw it for a person, check the posed arm in Isaac |
| [provenance.py](provenance.py), [constants.py](constants.py) | the stamp every dataset carries; the log files under `logs/datagen/` |

## Details

- [grasps/](grasps/README.md), the grasp reference and the physics referee; [corpus/](corpus/README.md), whether a
  corpus is worth training on; [src/robot/grasping/deep/](../src/robot/grasping/deep/README.md), the generator it trains.
- [docs/cli.md](../docs/cli.md), [train_your_own_generator.md](../docs/runbooks/train_your_own_generator.md) and
  [corpus_v5_build.md](../docs/runbooks/corpus_v5_build.md) for the commands and the procedures.
- Tests: `tests/test_datagen_layering.py`, `tests/test_datagen_api.py`, `tests/test_license_boundary.py`.
