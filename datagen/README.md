# `datagen`: the scene generator

Physically settled, licence-clean scenes with labels: 2D boxes and masks, depth, camera and 6-DoF
poses, visibility, and grasp labels. Optionally path-traced RGB.

It exists to serve four consumers: a grounding benchmark, the RL experiment, this repository's own
learned models, and an independent reference for the analytic grasp calculator.

**A tool, not part of the robot.** `datagen` imports from `src`; `src` never imports from `datagen`,
and that direction is enforced rather than merely intended. A generator that becomes a runtime
dependency of a robot is a generator nobody can delete.

## The split that makes it testable

The pipeline is `layout -> SceneSpec -> render -> dataset`, and the seam in the middle is the point.

**Layout** takes a seed and a scene index and is numpy only, so it runs on a laptop. It produces a
**`SceneSpec`**: frozen, millimetres and XYZW, spawn poses only. **Render** takes that spec and one of
three engines, and spawns, settles, verifies, shades or rasterises, and labels. What lands on disk is
the dataset.

Every layout bug caught before the render is a bug that would otherwise be found after a night of path
tracing. Every Isaac import is lazy and lives on the render side of that boundary.

A scene is reproducible from the dataset seed, the scene index, the code version and the physics
version, and the spec says so rather than pretending a seed alone is enough. Domain randomisation is
resolved at layout time, not at render time, because a renderer that rolls its own dice makes the seed
a half-truth.

```python
from datagen.config import DatagenConfig
from datagen.scenes import layout_scene

config = DatagenConfig(scenes=300, seed=0)
scene = layout_scene(config, 0)     # same inputs, byte-identical scene, forever
```

## What is here

| File | What it is |
|---|---|
| [`config.py`](config.py) | `DatagenConfig`: its own Pydantic tree, deliberately not part of the robot's config, so adding a scene knob cannot re-bless the robot's goldens. |
| [`scenes/spec.py`](scenes/spec.py) | `SceneSpec` and friends. Frozen, millimetres and XYZW (this repository's units, not Isaac's), spawn poses only. |
| [`scenes/layout.py`](scenes/layout.py) | Seed to scene: the four family samplers, the camera rig, the randomisation draw. |
| [`assets/procedural.py`](assets/procedural.py) | Four parametric families (primitive, industrial, packaging, vessel), licensed as your own. |
| [`assets/composite.py`](assets/composite.py) | Objects with parts: mug, jug, pan, hammer, bucket. Each part is a box, cylinder or sphere at a pose, never a new primitive, so the labeller intersects exactly what the renderer authors. Gives per-part grasp labels. |
| [`assets/library.py`](assets/library.py) | The mesh library: five public collections, and `custom`, a customer's own parts under a licence they declare. Meshes are fetched, never vendored. |
| [`assets/meshes.py`](assets/meshes.py) | A mesh's size is a fact, so it is measured rather than drawn from a band. Also the placeability limit. |
| [`assets/manifest.py`](assets/manifest.py) | What may be placed, with the licence that permits it, and the attribution file. |
| [`assets/licensing.py`](assets/licensing.py) | The licence rules, owned here because this is the code that must obey them. |
| [`assets/service.py`](assets/service.py) | `MeshPreparation`: fetch, normalise, screen, decompose, diagnose. The whole customer path on one noun. |
| [`render/engine.py`](render/engine.py) | The engine selector and the `SceneEngine` contract. A backend is not admissible until it passes the equivalence gate in [`render/equivalence.py`](render/equivalence.py). |
| [`cost.py`](cost.py) | What a corpus will cost before anybody starts it: hours, gigabytes, refused scenes. Every coefficient carries its evidence in the printed report. |
| [`provenance.py`](provenance.py) | The stamp: config, code revision and dirty flag, asset hash, renderer, platform. |
| [`grasps/`](grasps/README.md) | Closed-form grasp labels, and the physics referee that says whether a grasp holds. |
| [`eval/`](eval/) | The instruments that grade something, as against producing it: the ladder, its floors, the approach-tilt view, cross-view association, and the sweeps. |
| [`corpus/`](corpus/README.md) | Point clouds and the ranker table, plus the gate that says whether a corpus is worth training on. |
| [`rl/`](rl/) | Scene to rig replay, the occupancy gate, and physics-outcome collection for the RL chain. |
| [`prompts/`](prompts/) | Referring expressions, German and English, from `scene.json` alone. |
| [`api.py`](api.py) | `DatasetBuild`: render, label, clouds, or all three, from a described cell. |
| [`recipes.py`](recipes.py) | `CORPUS_RECIPES`, the named corpus shapes `init-config --recipe` writes out. |
| [`heldout.py`](heldout.py) | Which assets a model has never seen. A held-out set drawn from a contaminated pool is not one. |
| [`verify.py`](verify.py), [`preview.py`](preview.py) | Check a written dataset; human-viewable strips. |
| [`verify_robot.py`](verify_robot.py) | On box: does the posed arm land where the forward kinematics says, and can cuRobo plan to it. |
| [`constants.py`](constants.py) | The log paths and the mesh library root, and why there is one log file per module rather than an aggregate. |

Every module that does real work writes a rotating file under `logs/datagen/`, and `cli.log` carries
the run header that makes the rest attributable to one invocation. One file per module and not an
aggregate, because a datagen run is one long pass owning the process: what you want hours later is that
pass's file, not five passes that never overlapped.

## Usage

Everything below has a library twin. The CLI is one door and the Python API is the other, and they
share one implementation rather than agreeing by accident.

```python
from datagen.api import DatasetBuild

build = DatasetBuild.from_file("my_cell.json", name="v1", scenes=300, engine="mujoco")
print(build.describe())                   # the cell, before anything costs
report = build.run(corpus_out="logs/corpus/v1")
print(report.render())                    # render, label, clouds, stopping at the first failure
```

| noun | module | covers |
|---|---|---|
| `DatasetBuild` | [`api.py`](api.py) | `build`, `label-grasps`, `build-cloud-corpus` |
| `MeshPreparation` | [`assets/service.py`](assets/service.py) | fetch, `normalise-meshes`, `screen-meshes`, `decompose`, `why-no-jaw` |
| `GraspEvaluation` | [`eval/service.py`](eval/service.py) | `eval-grasps`, `grasp-gate`, `side-approach` |
| `PhysicsSampling` | [`grasps/service.py`](grasps/service.py) | `physics-sample`, `physics-compare` |
| `RankerCorpus`, `RankerFit`, `CorpusCheck` | [`corpus/service.py`](corpus/service.py) | `build-ranker-corpus`, `train-ranker`, `check-dataset` |
| `RecordCollection` | [`rl/service.py`](rl/service.py) | the `datagen.rl` commands |

Each noun takes keyword arguments, has one verb per question, and returns a frozen report with
`render()` for a person and `as_dict()` for a program.

```bash
python -m datagen plan                    # family mix, view count, objects per scene
python -m datagen describe --index 0      # one scene in full, as JSON
python -m datagen audit                   # the licence gate on the manifest this config draws

python -m datagen verify  --name v1       # check a written dataset: labels, masks, images
python -m datagen preview --name v1       # human-viewable strips
python -m datagen prompts --name v1       # referring expressions from scene.json alone

python -m datagen label-grasps --name v1  # closed-form grasp labels from the settled geometry
python -m datagen label-grasps --name v1 --density dense   # densities: default | dense | grid
python -m datagen eval-grasps  --name v1  # grade the calculator against them

python -m datagen decompose --jobs 8      # BEFORE a mujoco build: split concave meshes, cached
python -m datagen cost --scenes 2000 --engine mujoco    # BEFORE you start: hours and gigabytes
python -m datagen build --name v1 [--preview]           # --engine isaac needs Isaac and a GPU
python -m datagen build --name v1 --engine none         # no Isaac, no GPU, no physics engine
python -m datagen verify-robot [--require-curobo]       # on box: does the posed arm land where the kinematics says

python -m datagen.assets.fetch --list     # what can be downloaded, how big, on what licence
python -m datagen.assets.fetch gso --limit 200
python -m datagen normalise-meshes        # scale into metres, decimate into the face budget
python -m datagen screen-meshes --out screen.json       # which meshes earn a jaw label, in every pose
python -m datagen prepare-assets --out screen.json      # normalise then screen, in that order
python -m datagen why-no-jaw --from-screen screen.json  # and why the rest earn none

python -m datagen physics-sample  --name v1 [--physics-engine mujoco]   # on box: does it hold
python -m datagen physics-compare --name v1 --configs sfe_fused,deep    # the promotion gate, paired
python -m datagen build-ranker-corpus --name v1     # point cloud to the table a ranker trains on
python -m datagen check-dataset --corpus c.npz      # is it trainable, before a GPU hour is spent
python -m datagen train-ranker --corpus c.npz       # fit, and refuse if it loses to width alone
python -m datagen split-dataset --name v1 --parts 4 # views for a held-out split
python -m datagen heldout --corpus <dir>            # which assets a model has never seen
python -m datagen predict-masks --name v1           # segmentation a real camera would produce
python -m datagen camera-probe --name v1            # what each view actually reconstructs
python -m datagen init-config --recipe v1           # a config file to edit, from a named recipe
```

`python -m datagen --help` lists all twenty-nine subcommands with their flags. The three not shown
above are `build-cloud-corpus`, `grasp-gate` and `side-approach`.

## Three render engines, and what each one gives up

Isaac is not required. `render.engine`, or `--engine`, selects the backend, and the choice is stamped
into `provenance.json` and into every point-cloud file the corpus writes, because two engines never
settle a scene identically and a corpus that cannot say which one made it is a corpus nobody can
compare.

**Nothing refuses a mixed corpus.** The engine is recorded and recoverable; no consumer reads it back.
Stating that as a refusal would be describing a guard this package does not have.

| engine | physics | image | needs |
|---|---|---|---|
| `isaac` | PhysX | path-traced RGB, depth, masks | a local Isaac Sim install and an RTX-class GPU |
| `mujoco` | the MuJoCo solver | depth and masks, no RGB | `pip install mujoco` |
| `none` | none; objects are seated analytically | depth and masks, no RGB | nothing beyond this repository's own requirements |

What the cheap engines cost you, stated plainly:

- **No photoreal RGB.** They rasterise depth, instance masks and the arm silhouette. Everything that
  trains on geometry works; anything that needs an image does not.
- **`none` refuses the `pile` family by name**, because a pile is the physics. With the default equal
  family weights that costs a quarter of every request, so a request for 2,000 usable scenes needs
  2,667 asked for. `datagen cost` does that arithmetic for you and answers in usable scenes.
- **MuJoCo resolves mesh contact against the convex hull**, so a concave mesh has to be split before it
  can be settled. That is what `python -m datagen decompose` does, and the engine will also do it on
  demand. Run the warm-up first: cold, the decomposition happens inside the render loop and costs
  seconds per mesh; warm, a cache hit is milliseconds. The split recovers part of the concavity a hull
  fills in and not all of it, which is a real limit rather than a hidden one. Without the decomposition
  library installed the engine refuses concave meshes by name, because falling back to the hull would
  be invisible for the life of the corpus.
- **The cheap engines' cost is set by the assets, not by the engine.** The rasteriser walks triangles
  in Python, and a scanned mesh carries several orders of magnitude more faces than a procedural box.
  The engine that runs no physics at all can be the slower of the two on scanned meshes, which is how
  you know the raster and not the solver is the cost. `datagen cost` prices this correctly when the
  config restricts the mesh draw, and warns when it cannot tell.

There is no parallel mode inside one run: shard it, and give each shard its own `--corpus-out`
subdirectory. Scene ids count within one dataset, so flattening two shards into one directory
overwrites scenes silently.

`physics-sample`, `physics-compare` and the RL collection path all take `--physics-engine isaac|mujoco`.
The MuJoCo referee exists so that "generate your own data" is a claim a customer without a large
NVIDIA-only install can act on.

## Looking at it

**`*_depth.png` and `*_instances.png` are black in every image viewer, and that is correct.** They are
uint16 millimetres and uint16 object ids. A 300 mm surface is a small fraction of the 16-bit range and
object 3 is a smaller one. Rescaling them to look nice would throw away the precision they exist to
carry.

To actually see a scene, use `preview`. Everything it writes lands in one flat `preview/` folder at the
dataset root, named `<scene>_<view>_<kind>.png`, because the point of these files is to be skimmed and
skimming hundreds of them should be one scroll rather than hundreds of folders.

| in `preview/` | what it shows |
|---|---|
| `..._preview.png` | RGB with labelled boxes, depth, noisy depth, instances |
| `..._depth_view.png` | depth alone, colour-mapped, near is bright |
| `..._depth_noisy_view.png` | drawn on the clean depth's scale, so the two are comparable |
| `..._instances_view.png` | one distinct colour per object id |

Every coloured panel is stamped with the range it was scaled to, so per-image normalisation never
misleads. `preview` also drops a `HOW_TO_READ.txt` at the dataset root, because a folder of black PNGs
reads as a broken run and the explanation belongs where the confusion happens. `build --preview` writes
all of it as the run goes, so an overnight run producing rubbish is visible at scene 3 rather than at
scene 300.

## Notes

**Four families, four physics regimes.** `sparse` (nothing touches), `packed` (touching, single layer,
where segmentation merges masks), `pile` (dropped, stacked, occluding) and `bin` (with KLT walls). The
mix is assigned up front, so 300 scenes with equal weights is exactly 75 each, and it is interleaved,
so an interrupted run has still seen every family.

**The camera rig is a default, not the only one.** It is a wrist camera plus two raised, tilted cameras
flanking the workspace. Two opposed views mean the arm, which is this cell's dominant occluder, can
never hide the scene from both. The wrist view is marked as wrist-mounted because it is the one the
renderer may fail to reach, and that failure is recorded rather than silently relocated.

**Describe your own cell instead.** `camera_rig.cameras` takes a tuple of camera specs: a name, a mount
(fixed or wrist), a position and look-at in millimetres, and optionally a per-camera resolution and
field of view. A fixed camera with no position is refused rather than defaulted. An eye-in-hand camera
belongs on the arm, so a declared wrist camera on a config with no arm is refused, and
`camera_rig.allow_unmounted_wrist` is the explicit opt-in for the case where a stationary wrist-height
view is genuinely what you want.

**And the scene is yours too.** `workspace.table_size_mm`, `workspace.bin` (inner footprint, wall
height, wall thickness, and the jaw clearance objects keep from the walls) and `families.placement`
(the margins between objects, the pile clearances, the try budgets) are configuration, and each default
carries the measurement it came from as field documentation. A bin larger than
`workspace.half_extents_mm` places objects the arm cannot reach: declaring one is refused, and
inheriting the default one warns.

**A rendered image is a derivative work of every mesh in it.** That is why the licence gate runs on the
manifest before a run, and why an asset under a licence that obliges attribution is refused without
one. Only CC0, CC-BY, or your own. Never non-commercial, never ShareAlike, never NoDerivatives. Those
last two are not decoration: a `cc-by-sa` and a `cc-by-nd` string both pass a `startswith("cc-by")`
test, which is why `assets/licensing.py` exports `is_sharealike` and `is_noderivatives` beside
`is_noncommercial`. Normalising a licence string alone opens that hole.

**Oversized objects are labelled, not filtered.** An object too big for the jaw aperture is a real
object and a real suction target, so `jaw_graspable` records the fact instead of the library pretending
the world is jaw-sized.

**Units are this repository's**, millimetres and XYZW, and convert to Isaac's metres and WXYZ at the
render boundary, exactly like a vendor driver.

## Asset sources, and what each one can and cannot do

`assets.<source>_weight` values are relative weights, not probabilities: `procedural 1.0,
composite 2.0` draws twice as many composites. The default is procedural-only, so an existing run
generates exactly what it always did.

| source | geometry | grasp labels | objects published |
|---|---|---|---|
| `procedural` | authored from parameters: sixteen kind words, three actual solids | analytic, whole object | unlimited |
| `composite` | several primitives at poses: a handle, a grip, a neck | analytic, per part | unlimited |
| `gso` | Google Scanned Objects, real scans | analytic on the closed surface: sound, not complete | 1,033 |
| `ycb` | the YCB object set | the same | 103 |
| `thingi10k` | a CC0 slice of Thingi10K | the same | 30 |
| `asos` | scanned retail objects | the same | 50 |
| `objaverse` | individually authored models | the same | 724,500 |
| `custom` | your own parts: any mesh trimesh reads | the same | yours |

`custom` accepts six suffixes, listed as `CUSTOM_SUFFIXES` in [`assets/library.py`](assets/library.py):
`.obj`, `.stl`, `.ply`, `.off`, `.glb`, `.gltf`. `.dae` is not among them.

**A fetched collection is not yet a placeable one.** Meshes arrive in whatever units their author used,
and the bank refuses one whose implied mass is absurd rather than scaling it silently. Run
`normalise-meshes` on a collection before raising its weight.

**Restricting the meshes does not restrict the scenes.** `assets.mesh_asset_ids` says which meshes may
be drawn, not which scenes are built. Procedural objects carry their own weight, so a corpus that names
your parts and leaves `assets.procedural_weight` at its default comes out part full of objects nobody
asked for, and nothing says so. Zero both weights when the point of the dataset is which objects are in
it, and set `assets.refuse_procedural_fallback: true` to turn that surprise into a refusal.

### Bringing your own parts

A bin-picking cell runs on the parts that cell handles, and no public dataset contains them:

```bash
python -m datagen.assets --fetch --source custom --from ./my_parts \
    --license own --attribution-text "ACME GmbH"
```

**The licence is refused rather than defaulted.** Nobody but you knows what your parts are licensed as,
and this repository's audit reads that string, so a default would put an unchecked licence into the
audit, which is the one failure mode a licence gate must not have. `own` is the ordinary answer for
parts you designed. The declaration is written to `LICENSE.txt` beside the meshes, where the manifest
and the audit read it back later; a licence that lived only in the argument list of the command that
imported them would be gone by the next session. Meshes dropped into the directory by hand carry no
declaration, and the audit reports them as missing a licence rather than passing them.

`python -m datagen.assets.fetch --list` prints every public collection with its object count,
approximate size and licence, and says per source whether the licence was verified per model or taken
from the collection's published terms. That column is not cosmetic: YCB states no per-object licence,
so a dataset built on it inherits a claim rather than a check.

The same thing from Python, with counts you can act on rather than text to scrape:

```python
from datagen.assets.service import MeshPreparation, available_sources

for row in available_sources():
    print(row["key"], row["objects"], row["licence"], row["licence_verified"])

report = MeshPreparation.from_sources(["gso"]).fetch(limit=200)
print(report.render())      # fetched, already present, skipped on licence, failed, per source
```

### What it takes for a scanned mesh to be usable

The meshes are rendered, collided and labelled from one closed surface, so pixels, contacts and labels
are the same triangles. Three properties follow, and each is a limit worth knowing before you trust a
number computed over them.

**Most scanned meshes are not watertight as shipped, and a generic repair pass fixes almost none of
them.** The overwhelming majority fail on open boundaries alone, which are clean simple loops where a
scanner never saw the underside. Capping those loops recovers most of the library, and shells with no
interior are then refused.

**A mesh label is sound but not complete.** A box admits exactly three closing axes and they can be
listed; a scanned surface admits a continuum, and the labeller samples it. Every label is a real grasp,
so a missing label no longer proves a missing grasp, and recall over meshes is a lower bound.
Soundness is asserted rather than assumed: a box put through the mesh path produces grasps the
closed-form box verdict independently accepts.

**`assets.mesh_collision` is not a fidelity preference.** It decides whether the physics and the labels
describe the same object. PhysX cannot collide a dynamic body against raw triangles, and setting the
signed-distance-field token through the USD collision API alone is ignored: PhysX falls back to a
convex hull, which fills every concavity. The PhysX-specific SDF API must be applied alongside the
token.

An asset the renderer cannot draw refuses rather than becoming a box. The kind-to-primitive map answers
`box` for an unknown kind, so an unguarded scanned object would be rendered as a cuboid and labelled as
the same cuboid: two components agreeing with each other and both wrong about the object, which is
worse than a crash because the dataset looks fine.

### What a composite buys

A parallel jaw can only take a handle whose free span exceeds the finger geometry, and for a
2F-85-sized gripper that threshold is around 100 mm, which is wider than a typical mug handle. So the
composite families are a deliberate mix: hammer and pan offer feature grasps in almost every draw,
bucket and jug in some, and the mug in none. The positives teach "grip the feature, not the body", and
the mug is the negative, because a library of positives only would teach a model that handles are
always graspable.

## The procedural library's own limit

Sixteen kind words, three solids, in the procedural library, which is still the default. The renderer
authors a sphere, a cylinder or a scaled cube, so a `flange` is a cuboid and a `bowl` is a cuboid. The
map lives in [`assets/procedural.py`](assets/procedural.py) as `PRIMITIVE_FOR_KIND`, and both the
renderer and the labeller import it from there, because if the two ever disagreed the labels would
describe a shape that was never rendered.

For the prompt layer this costs nothing, since colour and size carry the grounding. For grasping it
costs a great deal: the concave, awkward shapes where a top-down jaw grasp really fails are not in a
procedural-only corpus, so any grasping number measured over one is a number about convex primitives
and says so. Composite objects carry handles and grips, and the mesh collections place real scanned
objects. Both are default-off, so an existing run is unaffected, and turning either on is what makes a
grasping measurement generalise.

## See also

- [`grasps/README.md`](grasps/README.md), the analytic grasp reference and the physics referee
- [`corpus/README.md`](corpus/README.md), is this corpus worth training on
- [`src/robot/grasping/`](../src/robot/grasping/README.md), the stack this data grades
