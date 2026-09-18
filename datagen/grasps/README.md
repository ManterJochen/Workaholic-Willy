# Grasp labels from geometry, and the physics that checks them (`datagen/grasps`)

This package lists every jaw and suction grasp the true geometry of a rendered scene admits, and runs the
physics referee that shakes a grasp to see whether it holds. It never reads a depth image and never imports the
grasping stack it grades, so when the two disagree the disagreement is about the world. You reach it through
`DatasetBuild.label` and `PhysicsSampling`.

```python
from willy import DatasetBuild, PhysicsSampling

build = DatasetBuild.from_file(name="first", scenes=8, seed=0, engine="none")   # a dataset already rendered
labels = build.label(density="default")      # every grasp the settled geometry admits, into grasps.jsonl
print(labels.summary["jaw"], "jaw and", labels.summary["suction"], "suction labels")

shake = PhysicsSampling.from_dataset(build.root, engine="mujoco").sample(per_class=4)
print(shake)                                 # held of shaken, per stratum of the verdict
```

Labelling needs no GPU and re-runs on a dataset without rendering it again.
[03_label_and_shake.py](../../examples/offline/datagen/03_label_and_shake.py) runs the same two steps. From a
shell, with the exit codes of `python -m datagen` (0 done, 1 a named problem, 2 a bad request):

```bash
python -m datagen label-grasps --name first                  # grasps.jsonl; --density dense or grid for more
python -m datagen label-grasps --name first --jaw narrow_55  # another gripper, into grasps_jaw_narrow_55.jsonl
python -m datagen physics-sample --name first --physics-engine mujoco --per-class 4
python -m datagen eval-grasps --name first                   # grade the calculator against the labels
python -m datagen physics-compare --name first --configs sfe_fused,fused   # two rungs, on the same objects
python -m datagen grasp-gate --name first                    # exit 1 on a drop against the committed baseline
```

## The nouns

| Noun | Built by | Verb | Returns |
|---|---|---|---|
| `DatasetBuild` | `from_file(name=..., ...)`, see [datagen](../README.md) | `label(density=..., jaw=...)` | a `StageResult` whose `summary` counts labels per kind and family |
| `PhysicsSampling` | `from_dataset(root, engine="mujoco")` | `sample(per_class=...)`, `compare(arms)` | `PhysicsReport`: held of shaken, per stratum |
| `GraspEvaluation` | `from_dataset(root)`, in `datagen.eval.service` | `evaluate()`, `gate()` | the ladder's `EvaluationReport`, the gate's `GateReport` |

`--jaw` takes a jaw from `PROCEDURAL_JAWS` in [verdict.py](verdict.py) or a hand from the gripper registry.
Each label run and each physics run writes its own file, so a proposal verdict, a label verdict and a paired
comparison can never be quoted as one another.

## What it refuses

| Refusal | When | What to do |
|---|---|---|
| a physics pass whose controls fail | before any trial: the harness is wired wrong | nothing is scored; fix the cell the controls name |
| a trial in a full bin | the teleported gripper overlaps something the solver cannot resolve | it is refused, not scored; shake lone objects |
| `compare` with one arm, or no object every arm proposed on | always: an empty pairing is not a tie | name two rungs that `eval-grasps` wrote |
| `--jaw` with `--physics-engine isaac` | the Isaac cell carries the 2F-85 only | shake another jaw's labels on `mujoco` |
| `grasp-gate` against a baseline of other rungs or another scene step | always | rewrite it with `--write-baseline`, as a separate step |
| a mesh that cannot be closed into a solid | at labelling | refused by name, never labelled as a box |

## How far a label can be trusted

Every label is computed from the primitive or closed mesh the renderer authored, at the pose physics left it
in. The table from kind to solid is the one the renderer reads, and `tests/test_datagen_grasps.py` pins that.

| Object | Closing axes | So |
|---|---|---|
| box, cylinder, sphere | all of them, in closed form | a missing label proves a missing grasp |
| scanned or custom mesh | a sample, plus the three principal axes | every label is real; recall over meshes is a lower bound |

The verdict is one predicate with typed reasons: does the jaw open wide enough, do both pads land inside the
friction cone, is the approach clear, does a fingertip go under the table (`BELOW_TABLE`). A fingertip may rest
on the table, never pass through it. The pad is a rectangle measured off the 2F-85's collision geometry, the
fingertip leads the contact, and approach directions are not pre-filtered, which is why `below_table` leads the
rejection histogram. The gripper envelope is written here from the hardware dimensions rather than imported
from the grasping stack, because shared code would make the two agree for the wrong reason.

## Grading a generator: the ladder

`eval-grasps` runs the calculator on every object in every view and grades each candidate against these labels.
Precision needs no tolerance: a candidate would close or it would not. Recall does, so it is a curve over
position in millimetres by angle in degrees, with `WORKING_POINT` in [../eval/ladder.py](../eval/ladder.py),
10 mm and 20 degrees, marked on it. Both are reported over the objects visible in the view and over all
labelled objects, because the gap is what one camera cannot see. Every number is a pure function of the
per-candidate file, so `--summary-only` asks a new question without a re-run.

The rungs in `CONFIGURATIONS` each add one setting to the rung above. Three floors come first in that list and
propose without ranking, as the scale the rungs are read against: `floor_random` (luck inside the admissible
cone), `floor_topdown` (straight down, the honest bar for a claim of "learned") and `floor_normal` (into the
surface along its oriented normal). A learned generator is graded in `DEEP_CONFIGURATIONS`, apart from the
ladder.

`predict-masks` runs a real detector and segmenter over every rendered view, and `eval-grasps --masks pred`
re-runs the ladder on those masks. Read both denominators there: over visible objects a predicted-mask pass can
beat ground truth, because the objects it missed leave the denominator. `grasp-gate` runs `GATE_CONFIGS` on
every fourth scene against the committed `gate_baseline.json`, reports a gain as loudly as a drop, and needs the
dataset on disk, so it runs on a workstation rather than in CI.

## The physics referee

`physics-sample` draws trials from both sides of the verdict: labels, accepted candidates, and candidates the
verdict rejected, per reason. A referee that holds everything it is handed has measured nothing. Four controls
run first, and the pass stops if one fails: the shut jaw is solid, a top-down grasp on a lone block holds, the
same grasp on air does not, and the block grasp still holds after an unrelated scene was built and retired.

Held is measured by taking the table away, not by lifting: the gripper is fixed in the world and teleported
to the grasp, so no verdict depends on whether an arm could reach it. That is also the limit. In a full bin the
teleport drives the gripper into an overlap the solver cannot resolve, and the trial is refused rather than
scored as "did not hold". `physics-compare` is the paired comparison: every object contributes one trial to
every arm, so the arms are the only thing that varies.

## What this reference does not know

| | |
|---|---|
| Reachability | no inverse kinematics: a real grasp may still be out of the arm's reach |
| Dynamics | a label closes inside the friction cone; whether it survives a lift is the referee's question |
| The sensor | the geometry is the settled truth; what a camera sees of it is the evaluation's problem |
| Concave procedural shapes | a procedural `bowl` is drawn and labelled as a box; composite and mesh sources extend it |

## Status

| Capability | Evidence |
|---|---|
| Closed-form spans, widths and normals | measured in simulation: checked against a brute-force ray march in `tests/test_datagen_grasps.py` |
| The MuJoCo and Isaac referees | measured in simulation: each passes the four controls; the two have not been compared on the same grasps |
| The referee in a full bin | measured in simulation: it refuses rather than scores, because the gripper is teleported |
| A label holding on a physical gripper | never touched hardware |

## Files

| File | Holds |
|---|---|
| [shapes.py](shapes.py) | box, cylinder, sphere and closed mesh: ray spans, normals, support widths |
| [verdict.py](verdict.py) | the jaw and suction verdicts, their typed reasons, and the gripper envelopes |
| [labels.py](labels.py) | every grasp the geometry admits, per object, and the label densities |
| [identity.py](identity.py) | what makes two candidates the same grasp, for the corpus dedupe and the physics join |
| [masks.py](masks.py) | predicted masks in the renderer's encoding |
| [physics.py](physics.py) | the sample, the controls and the paired draw |
| [physics_isaac.py](physics_isaac.py), [physics_mujoco.py](physics_mujoco.py) | the two referees behind one interface |
| [service.py](service.py) | `PhysicsSampling` and the rule that names each run's output file |

The grader is in [../eval/](../eval/): [ladder.py](../eval/ladder.py), [floors.py](../eval/floors.py) and
[cross_view_association.py](../eval/cross_view_association.py), which scores whether a second camera's mask
belongs to the same object before a cloud is fused across views.

## Details

- [datagen](../README.md), the scenes these labels come from; [corpus/](../corpus/README.md), whether the
  corpus built on them is worth training on.
- [src/robot/grasping/](../../src/robot/grasping/README.md), the stack under test, and
  [docs/grasping-math.md](../../docs/grasping-math.md), the formulas it evaluates.
- Tests: `tests/test_datagen_grasps.py`, `tests/test_physics_mujoco.py`, `tests/test_ladder_gate_baseline.py`,
  `tests/test_grasp_floors.py`.
