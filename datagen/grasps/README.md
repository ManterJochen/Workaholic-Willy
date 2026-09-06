# `datagen/grasps/`: the grasp reference, and the physics that judges it

Closed-form grasp labels computed from the true scene geometry, and the referee that asks whether a
grasp actually holds. Needs no GPU and no Isaac for the label path, reads no rendered depth in the
reference path, and never imports the grasping stack it exists to grade.

## Why this can be called a reference

Everything here is computed from the primitive the renderer authored and the pose physics left it in.
It never reads a depth image, never builds a point cloud, and never imports a line of the code under
test. That is what separates a reference from a second opinion: when the two disagree, the
disagreement is about the world and not about two heuristics with different taste.

The property that whole claim rests on is that the solids this package builds sit at zero distance
from the object poses the renderer itself wrote into `scene.json`. It is asserted rather than assumed.
If it ever drifts, nothing else here means anything.

| file | what it is |
|---|---|
| [`shapes.py`](shapes.py) | Box, cylinder, sphere: ray spans, surface normals, support widths, all closed form and all checked against a brute-force march. Plus a mesh branch, exact but not closed form. |
| [`verdict.py`](verdict.py) | Is this grasp real? One predicate, no tolerance, no labels, typed rejection reasons. Also the gripper envelopes. |
| [`labels.py`](labels.py) | Every grasp the geometry admits, per object, enumerated from the antipodal structure rather than searched for. |
| [`identity.py`](identity.py) | What makes two candidates the same grasp, in one place, so the corpus builder's dedupe and the physics join cannot disagree. |
| [`masks.py`](masks.py) | The second mask source: real segmentation instead of the renderer's ground truth. |
| [`physics.py`](physics.py) | The sample: draw trials from both sides of the verdict, run the controls, and answer with a number. |
| [`physics_isaac.py`](physics_isaac.py) / [`physics_mujoco.py`](physics_mujoco.py) | The two referees behind the same interface. |
| [`service.py`](service.py) | `PhysicsSampling`: the library twin of `physics-sample` and `physics-compare`. |

The grader that runs the calculator against these labels is not in this package. It is
[`datagen/eval/ladder.py`](../eval/ladder.py), with the floors in [`floors.py`](../eval/floors.py) and
the cross-view study in [`cross_view_association.py`](../eval/cross_view_association.py), because an
instrument that grades something belongs apart from the thing that produces it.

```bash
python -m datagen label-grasps  --name v1   # grasps.jsonl and a label report
python -m datagen eval-grasps   --name v1   # grade the calculator against them
python -m datagen eval-grasps   --name v1 --summary-only   # re-derive every curve, no re-run
python -m datagen predict-masks --name v1   # GPU: real segmentation into the same encoding
python -m datagen eval-grasps   --name v1 --masks pred     # the same ladder on those masks
```

The expensive pass writes one row per candidate, and every rate, curve and breakdown in the report is
a pure function of that file. So asking a different question, a tighter tolerance, a per-family split
or a different working point, costs nothing.

## The mesh branch: sound, but not complete

A scanned object answers the same two questions exactly, because a ray/triangle intersection and a
nearest-triangle normal are exact operations, but only once the surface is closed, and never in closed
form. [`assets/mesh_geometry.py`](../assets/mesh_geometry.py) does the closing.

What changes is the enumeration:

| | closing axes | therefore |
|---|---|---|
| box, cylinder, sphere | all of them, in closed form | a missing label proves a missing grasp |
| mesh | a sample, plus the three principal axes | a missing label proves nothing |

So for scanned objects the label set is sound but not complete: every label is a real grasp, and recall
computed over them is a lower bound. Soundness is asserted rather than assumed, by putting a box
through the mesh path and requiring the closed-form box verdict to accept the grasps it produces.

## The two measurements, and why they are separate

| | needs a tolerance | therefore |
|---|---|---|
| Precision | No. Does the jaw open wide enough for what it must span, do both pads land inside the friction cone, is the path clear, does a fingertip go under the table? A candidate is physically possible or it is not. | one number |
| Recall | Yes. Whether the calculator found a graspable spot depends on how close it has to be to count. | a curve: the sweep over position in millimetres by angle in degrees |

`WORKING_POINT` in [`../eval/ladder.py`](../eval/ladder.py) is `(10.0 mm, 20.0 deg)`. It is marked on
the recall curve, not used to compute it.

Both are reported against two denominators, the objects visible in that view and all labelled objects,
because the gap between them is exactly what a single camera cannot see.

### Seven modelling decisions, each of which changed a number

**The pad is a rectangle, not a line.** Treating the jaw as one line through the anchor rejects
candidates whose miss is a fraction of a millimetre. That is an idealisation failing, not a grasp
failing. Contacts are the extremes over the whole patch.

**And the patch is measured, not guessed.** The finger and pad dimensions in `verdict.py` are read off
the gripper asset's own collision geometry. Guessing them wrong moves precision by a factor, and the
error was caught by the physics harness rather than by inspection.

**The table clearance is not half the finger thickness.** The pad's thickness runs along the closing
axis, which for a top-down grasp is horizontal, so subtracting it from a vertical clearance puts every
short object out of reach. A gripper may set its fingertip on the table; it may only not go through it.

**The fingertip leads the contact.** Sampling the finger backwards from the contact point lets a pose
whose tip is under the table pass the table check, because the finger ahead of the contact is never
looked at.

**The approach sweep is one long finger.** A finger translating along its own length axis sweeps
exactly a longer finger, so stepping along the path re-checks the same volume many times over.

**Approach directions are not pre-filtered.** An approach from below is measured as `BELOW_TABLE`
rather than filtered out of the sampling, which is why that reason leads the label histogram and why it
is worth reading rather than hiding.

**The gripper envelope is re-implemented here**, from the measured hardware dimensions, rather than
imported from the grasping stack's collision code. Sharing code with the thing under test would
produce agreement for the wrong reason. The dimensions are hardware facts; the code that uses them is
not.

## The ladder, and the floors under it

The evaluation configurations are a ladder: each rung adds exactly one thing to the rung above it, so
every difference means one thing rather than three. `CONFIGURATIONS` in
[`../eval/ladder.py`](../eval/ladder.py) is the list, and `DEEP_CONFIGURATIONS` holds the learned
generator separately, deliberately out of the ladder, because every ladder rung is the analytic
calculator with one setting changed and a learned model is not that.

The headline quantity is coverage: of the visible objects that admit a jaw grasp at all, how many got
at least one candidate that would physically close. It needs no tolerance.

Three floors sit at the top of the list and are not part of the one-change-per-rung progression. They
are its axis, because a rate read against zero is a rate without a scale, and the analytic verdict
admits a band of approach directions around vertical, so a generator that knows nothing except "point
down at the object" already collects some coverage.

| floor | what it proposes | what it answers |
|---|---|---|
| `floor_random` | seed on the target, approach uniform on the area of the generator's own admissible cone, random rotation and width | did the generator learn anything at all |
| `floor_topdown` | seed on the target, straight down, random in-plane rotation, mid-range width | the bar a claim of "learned" has to clear |
| `floor_normal` | seed on the target, into the surface along its camera-oriented normal, mid-range width | the dumbest thing that uses visible geometry |

None of them ranks. Every candidate scores zero, because a floor that ordered its own proposals would
be a generator, and a top-1 measurement would then be grading that order instead of the scale.

`floor_random` is deliberately the weakest of the three: area-uniform over a wide cone puts the mean
approach close to horizontal, and those rarely close. Quote it as beating coin-flipping, never as
beating a baseline. `floor_topdown` is the honest bar.

**A surface normal has no sign, and a floor that uses one must orient it.** Local principal-axis
analysis returns an eigenvector, the corpus builder orients separately against the observing cameras,
and a floor that skips that step approaches from underneath the table. The orientation is pinned from
both directions.

## The second mask source

Every ladder number over the renderer's own instance masks isolates the grasp generator from the
perception ahead of it, which is the right place to start and an advantage no real cell has. So
`predict-masks` runs a real detector and segmenter over every view and writes their masks in the same
encoding, and `eval-grasps --masks pred` re-runs the whole ladder on those.

The two are reported separately and never merged: the mask source is part of the key in every summary.

The prompt deliberately does not test grounding. It names every object kind in the scene at once, and
every detection is matched to the ground-truth instance it overlaps most, so a detector that finds the
right objects but attaches the wrong words to them still scores. That leaves the narrower question the
grasp generator actually depends on: given that the object was found, how good is its silhouette.

Unmatched objects are not dropped. An object the detector missed keeps its row with an empty mask, so
it lands in the evaluation as an object that produced no candidate. Silence is a miss, which is the
same rule the ground-truth pass uses.

**Read both denominators, because one of them lies.** Over only the visible objects, a predicted-mask
pass can score better than ground truth. That is not progress, it is selection: the undetected objects
leave the denominator and they are the occluded ones. A single number there reads as worse perception
producing better grasps.

## The physics sample: honest on a lone object, and it refuses itself in clutter

`python -m datagen physics-sample` answers whether a grasp the geometry calls valid actually holds.

**Trials are drawn from both sides of the verdict.** Testing only grasps the verdict accepted would
measure its false-positive rate and nothing else, because a predicate that says yes to everything would
score perfectly. So the draw covers the labels the verdict produced, candidates it accepted, and
candidates it rejected, stratified by the reason it gave. If the rejected ones fail at the same rate as
the accepted ones hold, the predicate does not discriminate and every number the analytic verdict
produces is decoration. That result is equally publishable either way.

**Four controls run first and the pass refuses to continue without them**, because a physics harness
that silently reports that nothing holds is indistinguishable from one that is wired wrong. Each
referee asks the same four questions its own way:

- the shut jaw must be solid, so the gripper has collision geometry at all;
- a textbook top-down grasp through the centre of a lone block must hold;
- the same grasp made to grip air must not;
- and the first control, repeated after an unrelated scene has been built and retired in between, must
  still hold, so the scene lifecycle does not leak.

**Held is measured by taking the table away, not by lifting the gripper.** The gripper is world-fixed
and teleported, and a teleported body transmits no tangential force, so a lift would measure the
teleport rather than the grip. Removing the support asks the same question, whether the jaw can carry
this object's weight, with no gripper motion at all.

**The gripper is teleported rather than carried there by an arm**, so no verdict here depends on
whether a particular robot could have reached the pose. That is deliberate, and it is also the limit:
in a full bin the teleport drives the gripper into an overlap the solver cannot resolve, the driven
joint leaves its range, and the trial is refused rather than scored. Refusing is the correct behaviour,
because a dead simulation answers "did not hold" to every grasp, which is the most convincing wrong
answer available. It also means **a teleported gripper is the wrong instrument for a cluttered scene**,
and carrying the gripper along a collision-free path is what the sample needs next.

Both referees are available: `--physics-engine isaac|mujoco`, on `physics-sample`, `physics-compare`
and the RL collection path. The MuJoCo referee needs neither a large NVIDIA-only install nor a GPU,
which is what makes the physics half of "generate your own data" a claim a customer can act on.

## The gate: the ladder as a regression check rather than a study

```bash
python -m datagen grasp-gate --name v1                   # exit 1 on a drop
python -m datagen grasp-gate --name v1 --write-baseline  # a deliberate second step
```

It runs the rungs named by `GATE_CONFIGS`, currently `sfe` and `sfe_fused`, on every fourth scene,
against the `gate_baseline.json` committed beside the ladder. The baseline records the scene step, the
rungs and the tolerance its rates were taken with, and it refuses to compare against a run that used
different ones.

Two properties are deliberate. An improvement is reported as loudly as a regression, and the baseline
is then meant to be rewritten on purpose, so a gain cannot be lost again unnoticed. And
`--write-baseline` is a separate command, because a gate that regenerates its own reference when it
fails is not a gate.

It needs the dataset on disk, so it is an on-box gate rather than a continuous-integration one.

## Cross-view association: the piece multi-view fusion was missing

Fusing a target cloud across cameras rests on something no camera can do alone, which is deciding that
the blob camera B segmented is the same physical object camera A is looking at. The one place in this
repository that fuses a target cloud today does it with ground-truth perception in every camera, which
is not a method, it is the answer.

[`../eval/cross_view_association.py`](../eval/cross_view_association.py) runs the real associator on
real segmented masks and scores every decision it would make against the dataset's own instance ids:

| verdict | meaning |
|---|---|
| correct | it chose a candidate belonging to the same instance |
| wrong | it chose a different object. The expensive one: it fabricates a contact face on the far side of a neighbour, and the generator cannot tell that cloud from a real one |
| abstained | nothing cleared the threshold, which costs one view and no more |
| unavailable | the other camera never detected the target, so no method could have won it |

Those last two are why the threshold is swept and three metrics are reported over the same decisions
rather than one being argued for. A metric measured only on the answerable cases looks best when it
never abstains, which is exactly the behaviour that welds a neighbour's surface into the target's
cloud. The useful quantity is not an accuracy, it is how much wrongness has to be accepted to buy each
extra view.

## What this reference does not know

| | |
|---|---|
| Reachability | No inverse kinematics, so no verdict depends on which arm is bolted to the table. A grasp that is geometrically real may still be unreachable. |
| Dynamics | A valid grasp here is one that closes on the object inside the friction cone, not one that provably survives a lift and a shake. Turning that from an argument into a measurement is what the physics sample is for. |
| The sensor | The geometry is the settled truth. What a camera could actually see of it is the evaluation's problem, not the reference's. |
| Concave shape | On a procedural-only corpus the renderer authors sixteen kind words as three solids, so a bowl is a cuboid. Any number measured there is a number about convex primitives. Composite objects and the scanned mesh collections are what extend it. |

## See also

- [`datagen/README.md`](../README.md), the scene generator these labels are computed from
- [`datagen/corpus/README.md`](../corpus/README.md), whether the resulting corpus is worth training on
- [`src/robot/grasping/`](../../src/robot/grasping/README.md), the stack under test
