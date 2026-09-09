# Training the grasp generator on your own parts

Copy and paste, in order. Every step says what it costs, what it produces, and what breaks if it is
skipped.

**There is no download.** No trained weights ship in this repository, and there is no fetch script
for them, because a generator trained on somebody else's objects is a generator that has never seen
yours. This runbook is the whole route from a CAD export to a model a cell can select. Each stage is
a separate command, so a question about one is settled on a finished output of the one before it
rather than guessed at.

**What this does not promise.** The pipeline below runs end to end. The quality of the model that
comes out of it is the open question the pipeline exists to answer, and it is answered on your
corpus, with the instruments in Step 5 and in `src/robot/grasping/deep/eval/probes.py`, not by any
claim made here. Nothing in the learned generator has ever been measured on a physical cell.

Read [`src/robot/grasping/deep/README.md`](../../src/robot/grasping/deep/README.md) alongside this:
it describes the architecture, the artifact and the serving seam, and it carries no result numbers
for the same reason this runbook carries none.

---

## What you need

| | |
|---|---|
| your parts | `.obj`, `.stl`, `.ply`, `.off`, `.glb` or `.gltf`. Not `.dae`: nothing installed here can open one |
| a GPU | the training defaults are `--batch 4` at `--points 8192`; `deep train-set --help` records what that measured on one card |
| disk | the corpus dominates. Price it with `python -m datagen cost` before you start |
| time | the render is the long pole, hours per shard |

---

## Step 0. The configuration, from the recipe

```bash
python -m datagen init-config --recipe v1 --out my_config.json
```

Do this first, and do not hand-write the configuration. The command prints its reasons and the
ordered commands that follow it, and it wires two things a hand-written file leaves out.

`families.flat_orientation: random` dumps objects instead of standing them up. An object placed
upright on its own base sits in a stable equilibrium and the solver leaves it there, while a tipped
object is several times as likely to admit a jaw grasp, so this buys several times the jaw labels
per scene. It compounds with the engine: MuJoCo is the engine that leaves objects upright and is
what a customer without an NVIDIA box uses, so upright plus MuJoCo is the worst cell of the matrix
and is exactly what an unspecified configuration produces. A corpus of standing objects also does
not look like a bin-picking cell, where parts are dumped rather than placed.

`assets.jaw_screen_path` points at the screen Step 1 produces. Without it the bank is filtered by a
box proxy on the mesh hull, and a jaw closes on a line through the object rather than on its hull,
so the proxy calls meshes graspable that the labeller finds nothing on and refuses meshes that do
have grasps.

The defaults are the other values on purpose, so every corpus already on disk rebuilds unchanged. A
recipe is the named bundle that says which of them a new corpus should not keep, and it is frozen
once it ships, so `v1` will always mean what it means today. The command refuses to overwrite a
configuration file you have already edited.

## Step 1. Your parts into the mesh library

The parts have to be imported with a licence before anything can screen them:

```bash
python -m datagen.assets --fetch --source custom --from ./my_parts \
    --license own --attribution-text "ACME GmbH"
```

The licence is required and the import refuses without it. Nobody but you knows what your parts are
licensed as, and this repository's audit reads that string, so a default would put an unchecked
licence into the audit, which is the one failure mode a licence gate must not have. `own` is the
ordinary answer for parts you designed; otherwise give the real identifier, such as `CC0-1.0`. The
declaration is written to `LICENSE.txt` beside the meshes, where the manifest and the audit read it
back later. Meshes dropped into the directory by hand carry no declaration and the audit reports
them as missing one rather than passing them.

The library twin is `datagen.assets.library.import_from_directory`, with the same refusal.

Then screen them:

```bash
python -m datagen prepare-assets --collection custom --out assets/screens/mine.json
```

**Units: every mesh is read as metres.** A CAD package emits millimetres by default, so a 70 mm
bracket arrives as a 70-metre object, and every check downstream still passes. The normaliser names
it rather than silently rescaling, because guessing would replace a visible thousandfold error with
an invisible one:

```
IMPLAUSIBLE SIZE 70000 mm: this mesh is probably authored in millimetres and every mesh here is
read as METRES. Pass --scale 0.001 to convert it
```

Re-run with `--scale 0.001` when you see it. A part that lands a thousand times too large still
screens as `ok` and earns no jaw label at all, because nothing that size can be gripped.

What you get is a screen file saying which of your parts a jaw can grasp and which a suction cup
can. Expect fewer than you think, and budget in graspable objects rather than in files: the yield
differs sharply by collection, and `--want-graspable N` stops once N of them have been found and
reports how many had to be looked at. A screen stopped that way is stamped partial, because a
stopped screen and a complete one are different statements about a library.

## Step 2. Scenes, and their grasp labels

```bash
python -m datagen build --config my_config.json --name mine_s0
python -m datagen label-grasps --name mine_s0 --density grid
```

`--kinds` does nothing at this step. The labeller always writes both jaw and suction labels, and
passing the flag here prints a note saying so. Where it decides something is Step 3.

`--density grid` guarantees the top-down approach per closing axis, which uniform azimuth sampling
reaches only by coincidence. The density is stamped into `grasp_label_report.json`, because a label
count means nothing without it.

`--label-budget N` stops early and walks the scene families round robin, so a budgeted corpus is a
cross-section rather than the first family. Scene directories sort by family, so a plain prefix
would hand you every bin scene and no sparse one.

## Step 3. The point clouds the model trains on

```bash
python -m datagen build-cloud-corpus --name mine_s0 --corpus-out logs/dl/clouds/mine/s0 --kinds both
```

One `.npz` per scene. This is the input to training and nothing else reads it.

`--kinds both` is load-bearing here, because the flag defaults to `jaw` and would otherwise drop
every suction label Step 2 wrote. Suction labels do not train the pose heads, since a cup has no
closing axis and no opening, but they do train the stage that decides where to grasp, and they
sharply cut the share of training units that have nothing to learn from.

## Step 4. Train

```bash
# prove the chain closes on YOUR corpus and YOUR box first. Minutes, not hours.
python -m src.robot.grasping.deep train-set --recipe v1 --tier smoke \
  --clouds logs/dl/clouds/mine --out logs/dl/models/mine_smoke \
  --slots 4 --artifact-gripper 2f85

# then the model you deploy
python -m src.robot.grasping.deep train-set --recipe v1 --tier full \
  --clouds logs/dl/clouds/mine --out logs/dl/models/mine \
  --slots 4 --artifact-gripper 2f85
```

The smoke tier proves the chain closes and nothing else. It is 2 epochs on 400 training units with
the refit off, it says nothing whatever about grasp quality, and what comes out of it is not a model
to deploy. Its value is that a corpus with a problem in it surfaces in minutes rather than after a
night. The tier is recorded in the plan, the run report and the model card, and `deep inspect`
prints a loud line for a smoke artifact, so a two-epoch file is never indistinguishable at load
time from one that took hours.

What `--recipe v1` sets, so you can see it rather than trust it:

* `refit`, which trains one more net on every unit and ships that. Without it the artifact is one
  fold's net and never sees the units of the folds that were not run, which on your corpus is a
  share of your own catalogue.
* `collapse_floor_deg: 1.0`, which stops a fold whose head answers every seed identically, from
  epoch three. A collapsed head is invisible in every quality metric until the poses are read hours
  downstream.

The banner prints what the recipe applied and names anything your own flags overrode. A recipe is
frozen once it ships.

`--tier full` is 36 epochs, and that is a floor rather than a ceiling: it is the length at which the
reference arm had not stopped improving, not a round number and not a measured optimum. Watch your
own curve instead of trusting it. Read it in windows of several epochs, because a single epoch is
noisy and a metric that rises, falls and rises again reads as a peak followed by overfitting at any
one point. If yours is still climbing at 36, pass `--epochs` and run longer.

| flag | why |
|---|---|
| `--artifact-gripper` | the artifact stamps which hand it plans for, and the loader refuses to guess. No recipe can set it, because no bundle knows which hand you own |
| `--slots` | grasps predicted per seed. At K=1 a seed binds one grasp and everything else that seed knows is discarded |
| `--crop-mm` | optional: gives the head a ball of points around its seed. Default off, and whether it helps is an open question |
| `--generative` | optional: replaces the K-slot head with a denoising one trained on every label. An arm being measured, not a replacement |
| `--freeze-backbone` | train only the heads. The cheap form of fine-tuning and the one to reach for on a small new corpus, since almost all of this model's parameters are in the backbone |

Do not add `--slot-mixing film` to this command. Two separate reasons, either enough on its own.
First, the argument for it is RETRACTED: it rested on a slot spread of 9.6e-09 measured under
`affine`, and the same metric on an actual trained artifact gives 34.87 deg, more than film's 25.81.
The 9.6e-09 came from seeds that were themselves feature-identical, which was a serve-time seeding
defect with its own separate repair. What `affine` really freezes is the seed conditioning, and
whether `film` repairs that in a way which reaches grasp quality is still open. Second, a recipe
deliberately contains no unsettled arm, so prescribing one as though it were settled is how a
withdrawn claim reaches a production cell.

An epoch is not a pass over the corpus. `--train-units` defaults to 4,000, so an epoch is 4,000
units however big the corpus is.

Watch it while it runs:

```bash
python -m src.robot.grasping.deep report --run logs/dl/models/mine --no-curve
```

`epochs.json` is rewritten after every epoch, so this works on a live run. It prints the lift over
the run's own held-out floor and declines to give a verdict it does not have the epochs for.

## Step 5. Judge it with physics, not with its own loss

```bash
python -m src.robot.grasping.deep propose \
  --artifact logs/dl/models/mine/set_grasp_generator_v1.pt \
  --clouds logs/dl/clouds/mine/s1 --out logs/dl/mine_proposals.jsonl \
  --scenes 40 --spread-mm 20

python -m datagen physics-sample --name mine_s1 --out logs/dl \
  --proposals logs/dl/mine_proposals.jsonl --physics-engine mujoco

python -m src.robot.grasping.deep score-proposals \
  --proposals logs/dl/mine_proposals.jsonl \
  --verdicts logs/dl/mine_s1/grasp_physics_proposals.jsonl \
  --control logs/dl/mine_s1/grasp_physics.jsonl
```

**Run a label control beside it, always.** That is the same referee on the same shard, judging the
corpus labels instead of the model:

```bash
python -m datagen physics-sample --name mine_s1 --out logs/dl --per-class 300 --physics-engine mujoco
```

Without it the model's hold rate has no scale. The referee's own ceiling is nowhere near 100 %, so a
model that reproduced the labels perfectly would not score 100 either, and a customer reading a bare
rate out of 100 will conclude the wrong thing. A zero from the model means nothing until the control
says the harness grips at all.

`--physics-engine mujoco` needs no NVIDIA box. Isaac is the other option and cannot share a machine
with a training run.

The join between a proposal and its verdict is by scene and rank, never by pose, because a pose join
is ambiguous wherever a label and the row describing it carry identical poses.

## Step 6. Point a cell at it

```yaml
robot:
  grasping:
    calculator: deep
    deep_generator:
      artifact_path: logs/dl/models/mine/set_grasp_generator_v1.pt
```

It fails closed. `calculator: deep` with no readable artifact of the right kind and version refuses
to build the cell rather than falling back to the analytic stack, because a cell that asked for the
learned generator and quietly got the other one would file the analytic stack's numbers under the
learned one's name. `robot.grasping.calculator` defaults to `geometric`, and `build_calculator` is
its only reader.

**Which device it runs on.** Leave `device` unset and it resolves the way every other model in this
stack resolves one: the `WILLY_DEVICE` environment variable first, then cuda, then mps, then a CPU
landing that warns, because a silent order-of-magnitude slowdown in the pick loop is worth a line in
the log. Naming a device that is not on the machine refuses rather than falling back to the CPU. An
indexed device such as `cuda:1` passes straight through for a multi-GPU box.

**Some analytic levers do not apply**, and the factory says which. `ik_service` and
`corridor_risk_per_candidate` are refused by name under `deep`: without them nothing filters for
reachability and the uncertainty re-rank reorders nothing, so a run would look like a bad generator
rather than a wrong wiring. Analytic-only knobs such as `isotropic_radial_closing` are logged as
ignored, because the learned generator decodes its own closing axis.

---

## Continuing a run, and adapting to your own parts

```bash
# continue THIS experiment: refuses on any change to the plan, field by field
python -m src.robot.grasping.deep train-set --resume --out logs/dl/models/mine ...

# start a NEW run from those weights: this is what training on your own objects uses
python -m src.robot.grasping.deep train-set --init-from logs/dl/models/mine/fold0.pt ...
```

The difference matters. `--resume` restores the optimiser moments, the schedule position and both
seed streams, and refuses if anything about the plan changed, because a resume continues one
experiment. `--init-from` deliberately does not, because a warm start on a different corpus is a
different experiment and should not wear the old run's card. A checkpoint that does not fit the
architecture is refused rather than loaded partly, and the report records where the weights came
from.

## Training from Python

The command line is a formatter over an API that reaches strictly more of the stack, and there is no
YAML here by design:

```python
from src.robot.grasping.deep.train.api import GeneratorTraining
from src.robot.grasping.deep.train.plan import PlanOverrides

run = GeneratorTraining.from_recipe(
    corpus="logs/dl/clouds/mine",
    recipe="v1", tier="full",
    overrides=PlanOverrides(target="approach", slots=4),
    out_dir="logs/dl/models/mine",
    artifact_gripper="2f85")

print(run.describe())          # what is about to run, before it costs anything
report = run.train()
print(report.render())
```

## See also

- [`src/robot/grasping/deep/`](../../src/robot/grasping/deep/README.md), the model, the artifact and
  the serving seam.
- [`corpus_v5_build.md`](corpus_v5_build.md), the same pipeline run at corpus scale, with the
  sharding rules a long build needs.
- [`datagen/`](../../datagen/README.md), the scene generator, its engines and its asset sources.
- `scripts/examples/api/06_datagen/bring_your_own_parts.py` and `scripts/examples/api/07_training/train_on_your_own_meshes.py`, both steps as runnable files.
