# The learned grasp generator (`src.robot.grasping.deep`)

A 6-DoF grasp generator you train on your own cell's data, selectable against the analytic default
with one config key. This package holds the model, the data conventions, the training path, the
measurement instruments and the runtime seam a cell loads weights through.

**No trained weights ship in this repository.** A customer trains on their own cell's data. Every
number in this document is a default or a shape you can read off the code; there are no results
here, because a number measured on another corpus describes nothing you will see.

## Two products live here

`deep/` holds two unrelated learned components.

| | What it is | Where |
| --- | --- | --- |
| the generator | a network that proposes grasps from a point cloud | everything except `ranker/` |
| the ranker | a gradient-boosted tree that scores candidates the analytic stack already produced | `ranker/` |

They share nothing but a directory. The ranker's artifact kind is `grasp_gbt_ranker`, the
generator's is `set_grasp_generator`. The rest of this document is about the generator.

## Layout

```
deep/
  protocol.py        the seam a cell sees: GraspCandidateGenerator, PreloadableGenerator
  calculator.py      the runtime: artifact in, GraspPoint candidates out, never raises
  set_artifact.py    reading and writing weights plus a model card
  set_decode.py      one slot's raw output turned into a pose
  __main__.py        the operator command line, ten subcommands

  net/       serialized_backbone (the encoder), set_generator (the whole net)
             slot_head (K grasps per seed), set_loss, set_targets, assignment
             rotation, gripper, local_crop, generative_head, generative_loss
  corpus/    sample (one scene to one training sample), index (units, folds, asset groups)
             grasp_encoding, discovery (finding and sampling scenes)
  train/     trainer (the loop and SetTrainingPlan), step (one batch), plan (assembling a run)
             api (training from code), report, recipes, progress, synthetic_control
  eval/      probes (ceiling, floor, memorisation), propose, proposal_scorer_training
             charts, run_report, asset_consistency, geometric_floor, label_coverage
             gripper_differential
  ranker/    the other product: runtime, training, features, context, shadow
  foreign/   reading a public corpus into the .npz format this loop expects
```

`net/`, `corpus/`, `train/` and `ranker/` re-export nothing, and that is load-bearing: a package
`__init__` executes on every import of anything beneath it, so a re-exported name drags its module
into every run that touches any sibling and makes it undeletable. Import the submodule you mean.
`eval/` and `foreign/` do re-export, from `__all__` lists in their `__init__`.

## The architecture

**Parameter count: 21,506,799.** Measure it yourself with
`SetGenerator(SetTrainingPlan().model).parameter_count`.

**The backbone, `net/serialized_backbone.py`.** Attention over an unstructured cloud is quadratic
and hopeless at the default 8,192 points. The published trick is to give the cloud an order first:
sort the points along a space-filling curve so that neighbours in the sequence are neighbours in
space, then run attention inside fixed windows of that sequence. Cost drops to `O(N x window)`, and
the window shape changes from block to block because the curve does. The defaults are width 384,
depth 12 and 6 heads, from `SetTrainingPlan().model.backbone`.

This is a simplification of the published design, and the file says so at its top. That work
alternates Hilbert and transposed-Hilbert curves; this alternates Z-order curves under permuted
axes. Whether the difference costs anything is not measured, and no result from this stack may be
read as a result about the published encoder.

Plain `torch` and nothing else. The strong published encoders lean on a sparse-convolution engine or
a scatter extension, both of which need a CUDA compiler on the machine that installs them. Users
train this on their own cell, so a dependency that needs a build toolchain there is a product defect
rather than an inconvenience.

**The head, `net/slot_head.py`.** Seed features plus a 14-number gripper vector produce K grasps per
seed, K defaulting to 4, matched one-to-one against that seed's label set by a Hungarian assignment.
Per slot the net emits (`net/set_loss.py`, `SlotOutputs`):

| Field | Shape | Meaning |
| --- | --- | --- |
| `approach` | `(B,K,3)` | unnormalised approach direction |
| `axis_params` | `(B,K,6)` | director parameters for the closing axis |
| `offset_m` | `(B,K,3)` | seed-to-centre offset in metres, seed-local frame |
| `width_m` | `(B,K)` | gripper opening in metres |
| `confidence` | `(B,K)` | logit: "this slot answers a real grasp" |
| `part_logits` | `(B,K,R)` | optional affordance logits, `None` by default |

**The gripper is a model input, not a constant.** `GRIPPER_VECTOR_DIM` is 14 numbers describing the
hand (`net/gripper.py`, with the named geometries in `JAW_GEOMETRY`), so one trained net can serve
more than one gripper. See the honest limit below.

## Training

### From code

```python
from src.robot.grasping.deep.train.api import GeneratorTraining
from src.robot.grasping.deep.train.plan import PlanOverrides

run = GeneratorTraining.from_recipe(
    corpus="logs/dl/clouds/v5",              # a directory is walked recursively
    recipe="v1", tier="full",
    overrides=PlanOverrides(target="approach", slots=4),
    out_dir="logs/dl/models/my_arm",
    artifact_gripper="2f85")

print(run.describe())                          # what is about to run, before it costs anything
run.attach_progress_listener(lambda row: ...)  # optional, one call per finished epoch
report = run.train()
print(report.render())                         # what happened
if report.succeeded:
    run.write_report(report)
```

**There is no YAML here, and none is planned.** `SetTrainingPlan` is a frozen dataclass and already
reaches strictly more of the stack than the command line does: `learning_rate` and `weight_decay`,
most of `SlotHeadConfig`, and every `SampleSpec` field but `points` have no flag at all. `build_plan(recipe=..., overrides=...)` is a convenience layer; constructing a
`SetTrainingPlan(...)` by hand stays first-class. That is the PyTorch arrangement and it is
deliberate.

`from_plan` takes raw handles, `from_recipe` resolves a named bundle. Both refuse at construction
rather than falling back: an unknown target, axis mode, slot mixing or control level, a backbone
width the head count does not divide (naming the head counts that would work), a crop radius outside
the band it was measured over, and an unknown recipe or tier.

### From the shell

```bash
python -m src.robot.grasping.deep train-set \
    --clouds logs/dl/clouds/v5 --out logs/dl/models/my_arm \
    --recipe v1 --tier full --artifact-gripper 2f85

python -m src.robot.grasping.deep report --run logs/dl/models/my_arm
python -m src.robot.grasping.deep inspect --artifact logs/dl/models/my_arm/set_grasp_generator_v1.pt
```

The command is a formatter over the same API. It validates nothing the API does not.

### Recipes and tiers

A **recipe** is a named, frozen bundle of settings, written into the artifact so a model can say
which recipe produced it. `v1` will always mean what it means today; a better bundle becomes `v2`
beside it. Individual defaults do not move, because moving them would make every earlier run
irreproducible without explicit flags.

A **tier** narrows a recipe to a compute budget. `smoke` is 2 epochs on 400 training units with the
refit off: it proves the chain closes on your corpus and your box, and says nothing about grasp
quality. `full` is 36 epochs and is the model you deploy. Thirty-six is a floor, not a ceiling; if
your own curve is still climbing there, pass `--epochs` and run longer, and read `deep report` for
the verdict on your own data.

The tier is recorded in the plan, the run report and the model card, and `inspect` prints a loud
line for `smoke`, so a two-epoch artifact is never indistinguishable at load time from one that took
hours.

### Serving it

```yaml
robot:
  grasping:
    calculator: deep
    deep_generator:
      artifact_path: logs/dl/models/my_arm/set_grasp_generator_v1.pt
```

`robot.grasping.calculator` defaults to `geometric`. `build_calculator` in
[`calculator_factory.py`](../calculator_factory.py) is the only reader of that key and is wired at
both construction sites in `execution/autonomous_grasp/cells.py`. It refuses a file that is not a
`set_grasp_generator` artifact of this version, naming the retired binned family explicitly if it
sees one, and it **fails closed**: it does not fall back to the analytic stack, because a cell that
silently ran the analytic generator would report every KPI under the wrong name.

The factory also refuses to drop a construction kwarg that would change what a pick does. `ik_service`
and `corridor_risk_per_candidate` are refused by name when passed with an active value, because
without them nothing filters for reachability and the uncertainty re-rank reorders nothing, and both
would look like a bad generator rather than a wrong wiring.

## What is and is not measured here

Read the instruments, not a headline number. `eval/probes.py` ships three:

- `oracle_ceiling` scores a perfect model on your corpus, which is what a top-1 number has to be read
  against. It is not 100 percent, because a seed that admits several grasps caps what any single
  answer can score.
- `baseline_floor` scores an untrained copy of the same architecture, which is what "learned
  nothing" looks like on your data with your metric and your unit count.
- `memorisation_probe` measures the trained model on matched seen and unseen samples, against the
  same gap measured on an untrained net. The fold key is `asset_group`, not the asset id, so the
  procedural variants of one object stay in one fold: the honest question is whether it can grasp a
  jug it has never seen, not a seventh jug having seen six. No pass or fail threshold is hard-coded,
  because a threshold chosen after seeing the number it judges is not a threshold.

A hit means the top-1 slot is within `hit_angle_deg` (15 degrees) and `hit_offset_mm` (20 mm) of a
real label, with 30 mm and 40 mm reported beside it so a curve can be read while it is still
climbing.

`held_top1_hit` is not comparable between two arms with different targets. `train/step.py` switches
which angle the hit gate reads on `config.target`, so the floors differ by metric definition alone.
Compare `held_axis_error_deg` and `held_approach_error_deg`, which are reported under both targets
for exactly this reason.

## The traps this package has paid for

1. **A prefix is not a sample.** A scene id is `<family>_<index>`, so `sorted()[:N]` returns one
   family and any conclusion drawn from it is about that family. Use
   `corpus.discovery.stratified_scenes`.
2. **The corpus walk must be recursive.** Scene ids count within one dataset, so two shards hold
   different scenes under identical names and a flat listing silently overwrites the collisions.
   `corpus.discovery.scene_files` walks with `rglob` for this reason.
3. **A comment that quotes a defect breaks a substring test.** A test searching source text for the
   wrong line passes on the defect and fails on the repair, because the fixing comment quotes the
   old text. Match on the AST instead.
4. **Membership, never substring, on artifact kinds.** `grasp_generator` is a substring of
   `set_grasp_generator`, so an `in` test refuses the family that ships.
5. **`SystemExit` is not an `Exception`.** It derives from `BaseException`, so a helper that exits
   kills a training loop through `except Exception`. Catch `(Exception, SystemExit)` explicitly, and
   never bare `BaseException`, because an interrupt must still work.
6. **The plan's assembly order is load-bearing.** `plan_with_slots` rebuilds `model` and
   `model.head`, so any head setting applied before it is silently discarded and the run reports the
   setting it did not run. `train.plan.build_plan` owns the order so no caller has to know it.
7. **A missing module can pass both gates.** `ignore_missing_imports = true` hides a deleted module
   from the type checker, and an import inside a function body is invisible to test collection, so a
   live path can be broken with both gates green. Deep imports are walked by AST for this reason.

## What is not here

- **No trained weights.** A customer trains on their own cell's data.
- **No multi-gripper evidence.** The gripper vector reaches the net, and every run so far used one
  hand, so the conditioning is wiring rather than a demonstrated capability. `eval/gripper_differential.py`
  is the instrument for that measurement when a multi-gripper corpus exists.
- **No calibrated probability.** The score the generator attaches is not `P(hold)`. There is no
  calibrated hold head; the generator proposes and the ranker ranks, and each is promoted on its own
  evidence.
- **No result from a physical cell.** Everything in this package is measured in simulation or
  derived from an analytical model.

## See also

- [`grasping/`](../README.md) for the tier and for `calculator_factory.build_calculator`
- [`generation/`](../generation/README.md) for the analytic generator this one replaces
- [`datagen/`](../../../../datagen/README.md) for the scene generator that writes the corpus
