# The learned grasp generator (`src/robot/grasping/deep`)

A 6-DoF grasp generator you train on your own parts, and the seam a cell loads its weights through.
One config key selects it in place of the analytic generator. No trained weights ship in this
repository, and this page carries no result numbers: a number measured on another corpus says nothing
about yours.

```python
from willy import GeneratorTraining, PlanOverrides

run = GeneratorTraining.from_recipe(
    corpus="logs/dl/clouds/my_parts",        # a directory of scene files, walked recursively
    recipe="v1", tier="full",
    overrides=PlanOverrides(slots=4),        # optional; a field left out keeps the recipe's value
    out_dir="logs/dl/models/my_arm",
)
print(run.describe())                        # what is about to run, before it costs anything
report = run.train()                         # the folds, the probes, the artifact
print(report)                                # what happened, and where the weights went
if report.succeeded:
    run.write_report(report)                 # report.json, which `deep report` reads
```

A cell then selects the weights in its profile:

```yaml
robot:
  gripper:
    model: robotiq_2f85        # the hand the artifact was trained across
  grasping:
    calculator: deep
    deep_generator:
      # A relative path is read against the config folder; the anchor names the repository, whose
      # logs/ the trainer writes to when run from the repository root (src/config/paths.py).
      artifact_path: "${WILLY_PROJECT_ROOT}/logs/dl/models/my_arm/set_grasp_generator_v1.pt"
```

The corpus comes from [datagen/](../../../../datagen/README.md), and the whole route from a CAD export
to a model a cell selects is the runbook
[train_your_own_generator.md](../../../../docs/runbooks/train_your_own_generator.md).
[02_train_on_your_own_meshes.py](../../../../examples/offline/training/02_train_on_your_own_meshes.py)
runs it at the smoke tier. With no simulator and no cell, `PublicCorpus` fetches a published corpus
into the same scene files and the run above is unchanged:
[04_train_on_a_public_corpus.py](../../../../examples/offline/training/04_train_on_a_public_corpus.py).

The same run from a shell, which exits 0 when done, 2 on bad arguments and 3 on a problem:

```bash
python -m src.robot.grasping.deep train-set \
    --clouds logs/dl/clouds/my_parts --out logs/dl/models/my_arm --recipe v1 --tier full
python -m src.robot.grasping.deep report --run logs/dl/models/my_arm
python -m src.robot.grasping.deep inspect --artifact logs/dl/models/my_arm/set_grasp_generator_v1.pt
```

The command formats the same API and validates nothing the API does not. `train-set --help` lists
every knob; `SetTrainingPlan` reaches more than the flags do, and building one by hand and passing it
to `GeneratorTraining.from_plan` is as supported as a recipe.

## The nouns

| Noun | Built by | Verb | Returns |
| --- | --- | --- | --- |
| `GeneratorTraining` | `from_recipe(corpus=..., recipe=..., tier=...)`, `from_plan(corpus=..., plan=...)` | `probe()`, `train()` | `CorpusProbe`, `TrainingRunReport` |
| `PublicCorpus` | `from_source(out_dir=...)`, in [`foreign/service.py`](foreign/service.py) | `describe()`, `fetch()` | `ImportReport`, and scene files to train on |
| the runtime generator | `build_calculator(tree.robot, data_dir=tree.root, ...)` with `calculator: deep` | `compute(...)` | ranked `GraspPoint`s, as the analytic one |

`describe()` says what a run will do, `probe()` measures the floor and the ceiling of the corpus with
no weights at all (see [What to measure](#what-to-measure)), `attach_progress_listener(fn)` calls `fn`
once per finished epoch, and `write_report(report)` writes `report.json` beside the weights.

## How it learns

One scene file in, one optimiser step out. Every box names the module that holds it.

```text
  A CORPUS OF SCENES                                      two producers, one file format
  ------------------------------------------------------------------------------------------
    datagen/     scenes rendered and labelled here     -->  <scene>.npz
    foreign/     a corpus somebody else published      -->  <scene>.npz
                                  |
    corpus/discovery.scene_files  |  walked recursively: a scene id counts within one shard,
                                  |  so a flat listing silently overwrites the collisions
    corpus/index.training_units   |  one TRAINING UNIT per object, not per scene
                                  v
  ONE SAMPLE                    corpus/sample.build_sample(scene, rng, SampleSpec)
  ------------------------------------------------------------------------------------------
    points_m   (N,3)  the cloud in METRES, in its own frame: xy mean removed, support height
                      subtracted from z, so the net never sees where the table was
    features   (N,F)  per-point channels; 0..2 are the unit surface normal
    supervise  (N,)   which points carry a label and may therefore be scored
    the labels (G,)   grasp centre, approach (signed), closing axis (antipodal), width
                                  |
                                  v
  THE NET                       net/set_generator.SetGenerator
  ------------------------------------------------------------------------------------------
    net/serialized_backbone   order the cloud along space-filling curves and attend inside
          |                   fixed windows of that order: width 384, depth 12, 6 heads
          |
          |   net/set_targets.sample_seeds    draw S seeds from the supervised points
          v
    net/local_crop            the neighbourhood of each seed, radius 40 to 120 mm
          |
          |   net/gripper.gripper_vector      the HAND, as 14 numbers, as an INPUT
          v
    net/slot_head             K = 4 slots per seed; each slot is one GraspSetPrediction:
                              approach, closing-axis parameters, offset, width, confidence
                                  |
                                  v
  THE LEARNING SIGNAL           train/step.set_training_step
  ------------------------------------------------------------------------------------------
    net/set_targets.gather_set_targets   the labels of THAT seed, in the sample's frame
    net/set_loss.grasp_set_cost          a K x L cost: angle, offset, width
    net/assignment.optimal_one_to_one    Hungarian match, one slot to at most one label
    net/set_loss.grasp_set_loss          the matched pairs, plus a confidence target that
                                  |      teaches an unmatched slot to say "no grasp here"
                                  v
                            one optimiser step
```

The run around that step, and how the weights reach a cell:

```text
  A RUN                         train/trainer.train_set_generator, via GeneratorTraining
  ------------------------------------------------------------------------------------------
    train/plan.build_plan(recipe, tier, overrides) --> SetTrainingPlan, frozen, and stamped
          |                                            into the artifact that comes out
          |   corpus/index.grouped_folds over ASSET GROUPS
          v
    for each fold:  train on the rest, measure on the held-out one, run the probes
          |
          v
    refit on every unit (optional)  -->  set_artifact.py: the weights plus a model card that
          |                              says recipe, tier, hands and which pass wrote them
          v
    eval/run_report --> report.json --> `deep report`;  eval/probes for the floor and ceiling
          |
          v
    calculator_factory.build_calculator(tree.robot)  under `calculator: deep`
          --> calculator.py, the runtime generator
          --> compute(cloud) --> ranked GraspPoints, the seam the analytic generator also fills
```

Four things the picture exists to make visible, each of which is a decision rather than a detail:

1. **The loss is over a set, so a slot index means nothing.** Nothing forces slot 0 to be any
   particular grasp. `optimal_one_to_one` decides which slot is compared with which label, per seed
   and per step, and that is what lets K slots learn K genuinely different answers instead of K
   copies of the average one. A regression head with fixed outputs cannot: it averages the grasps a
   seed admits, and the average of two good grasps is usually not one.
2. **The hand is an input, not a constant.** One net serves several hands, trained with `hands=`,
   and [eval/gripper_differential.py](eval/gripper_differential.py) measures whether the proposals
   actually move when the hand changes. Whether they move *correctly* is a separate question.
3. **Folds are cut over asset groups, not over scenes or samples.** Every variant of one object
   lands in one fold, so a held-out number is about objects the net never saw rather than about
   other views of objects it did. `memorisation_probe` is the check on that.
4. **A corpus this repository generated and a corpus somebody else published arrive in the same
   file format.** The loop cannot tell them apart, so no metric ever has to ask which path a sample
   came through. [04_train_on_a_public_corpus.py](../../../../examples/offline/training/04_train_on_a_public_corpus.py)
   is that route end to end.

[The model](#the-model) below says what is inside the backbone and the head; this says how a label
becomes a gradient.

## Recipes and tiers

A **recipe** is a named, frozen bundle of settings, stamped into the artifact so a model can say which
recipe produced it. `v1` always means what it means today; a better bundle becomes `v2` beside it. A
**tier** narrows a recipe to a compute budget. `smoke` is 2 epochs on 400 training units with the refit
off: it proves the chain closes on your corpus and your machine, and says nothing about grasp quality.
`full` is 36 epochs and is the model you deploy. Thirty-six is a floor: if your curve still climbs
there, raise `epochs` (`--epochs`) and read `deep report` for the verdict. The tier is recorded in the
plan, the run report and the model card, and `inspect` prints a loud line for a smoke artifact.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `ValueError` from `from_recipe` | an unknown recipe or tier | use a name the error lists |
| `ValueError` at construction | an unknown target, axis mode, slot mixing or control level | use a value the error lists |
| `ValueError` at construction | a backbone width the head count does not divide, a crop radius outside 40 to 120 mm | the error names the counts that work |
| `ValueError` from `train()` | folds cannot be cut asset-disjoint, a `resume` whose plan differs, several hands and no `artifact_gripper` | fix the corpus, or name the hand |
| `FileNotFoundError` from `build_calculator` | `calculator: deep` and no file at `artifact_path` | train one, or set `calculator: geometric` |
| `ValueError` from `build_calculator` | not a generator artifact of this version (a fold checkpoint, a retired model) | name the artifact a finished run wrote |
| `ValueError` from `build_calculator` | `robot.gripper.model` unset, not in the registry, or a hand the artifact never saw | name the cell's hand, or train across it |
| `ValueError` from `build_calculator` | a construction argument it cannot honour is active, such as `ik_service` | switch it off, or use `geometric` there |

Every refusal fails closed. The factory never falls back to the analytic generator, because a cell that
silently ran it would report every KPI under the learned generator's name.

## What to measure

Read the instruments, not a headline number. [eval/probes.py](eval/probes.py) ships three:

- `oracle_ceiling` scores a perfect model on your corpus. It is below 100 percent, because a seed that
  admits several grasps caps what any single answer scores.
- `baseline_floor` scores an untrained copy of the same architecture: what learning nothing looks like
  on your data, with your metric and your unit count.
- `memorisation_probe` compares the trained model on matched seen and unseen samples against the same
  gap on an untrained net. The fold key is `asset_group`, so the variants of one object stay in one
  fold. No pass threshold is built in.

`GeneratorTraining.probe()` runs all three before a run costs anything and prints them as one table;
[05_floor_and_ceiling_before_you_train.py](../../../../examples/offline/training/05_floor_and_ceiling_before_you_train.py)
is that call. `top_down` is the arm to beat, not `random`: every slot straight down at the seed is the
grasp a cell with no model at all would try.

A hit is a top-1 slot within 15 degrees (`hit_angle_deg`) and 20 mm (`hit_offset_mm`) of a real label,
with 30 mm and 40 mm reported beside it. `held_top1_hit` is not comparable between runs with different
targets, because the hit gate reads a different angle per target; compare `held_axis_error_deg` and
`held_approach_error_deg`, which both targets report.

## The model

The backbone ([net/serialized_backbone.py](net/serialized_backbone.py)) orders the cloud along
space-filling curves and runs attention inside fixed windows of that order, width 384, depth 12 and 6
heads over 8,192 points by default. It simplifies a published design (Z-order curves under permuted axes
in place of Hilbert curves), and no result from it may be read as a result about the published encoder.
It needs plain `torch` and no compiled extension. The head ([net/slot_head.py](net/slot_head.py))
proposes K grasps per seed, K = 4 by default, matched one-to-one against the seed's labels. Per slot it
emits a `GraspSetPrediction` ([net/set_loss.py](net/set_loss.py)): an approach, the closing axis
parameters, an offset and a width in metres, a confidence logit, and optional part logits. The model has
21,506,799 parameters at the defaults; `SetGenerator(SetTrainingPlan().model).parameter_count` measures
it.

The hand is an input, a 14-number vector ([net/gripper.py](net/gripper.py), resolved by
[hands.py](hands.py)), so one net can serve several hands: train with `hands=` (`--hands`), and
[eval/gripper_differential.py](eval/gripper_differential.py) measures whether the proposals change with
the hand. That a net's grasps are good for each hand is a separate question it does not answer.

## Rules for a corpus and a run

1. Sample scenes with `corpus.discovery.stratified_scenes`, never `sorted()[:N]`: a scene id is
   `<family>_<index>`, so a prefix is one family.
2. Walk a corpus recursively (`corpus.discovery.scene_files`): shards reuse scene names, and a flat
   listing overwrites the collisions.
3. `SystemExit` passes through `except Exception`, so a helper that exits ends a training loop. Catch
   `(Exception, SystemExit)`, and never bare `BaseException`, so an interrupt still stops the run.
4. Assemble a plan through `train.plan.build_plan`: `plan_with_slots` rebuilds the head and discards any
   head setting applied before it.

## Status

| Capability | Evidence |
| --- | --- |
| From a simulated corpus to trained weights a cell loads through `build_calculator` | measured in simulation |
| From a published corpus to the same weights, through `PublicCorpus` | measured: the import writes the format the loop reads, and a smoke run closes on it. Its folds are scene-disjoint, not asset-disjoint, because the source publishes no asset identity |
| Grasp quality of a trained generator | measured in simulation, on your corpus, with the probes above |
| The learned generator on a physical cell | never touched hardware |

The score a slot carries is not a calibrated probability of a hold. The generator proposes; the
[ranker](ranker/) ranks, and each is promoted on its own evidence.

## Files

| Path | Holds |
| --- | --- |
| `protocol.py`, `calculator.py` | the seam a cell sees; the runtime generator, which never raises from `compute` |
| `set_artifact.py`, `set_decode.py`, `hands.py` | weights plus a model card; one slot to a pose; a hand's name to its vector |
| `__main__.py` | the command line |
| `net/` | backbone, slot head, losses, targets, assignment, rotation, gripper vector, local crop, generative head |
| `corpus/` | one scene to one training sample; units, folds and asset groups; finding and sampling scenes |
| `train/` | the loop and `SetTrainingPlan`, one batch, plan assembly, `api` for training from code, recipes |
| `eval/` | the probes, `propose`, charts, run reports and the other measuring instruments |
| `ranker/` | a separate product: a gradient-boosted tree that ranks analytic candidates (`grasp_gbt_ranker`) |
| `foreign/` | a public corpus read into the `.npz` scene format this loop expects; `PublicCorpus` is its door |

`net/`, `corpus/`, `train/` and `ranker/` re-export nothing, so an import of one module does not load
its siblings; import the submodule you mean. `eval/` and `foreign/` re-export their `__all__`.

## Details

- [grasping/](../README.md): the stack, and `build_calculator`
- [generation/](../generation/README.md): the analytic generator this one is selected against
- [docs/cli.md](../../../../docs/cli.md): the data generation and training commands in order
- [datagen/](../../../../datagen/README.md): the scene generator that writes the corpus
