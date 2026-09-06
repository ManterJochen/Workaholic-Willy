# `scripts/examples/`: twenty-nine executable examples, one decision each

Each one is a file you run, not a list of commands to copy. They share a spine
([`_common.py`](_common.py)) so twenty-nine scripts read as one tool: the same step and verdict
shape, the same exit codes, the same rule about hardware. Every example drives the real library
nouns rather than re-implementing a check, so a pipeline an example demonstrates is a pipeline that
exists.

```bash
python scripts/examples/cell/01_robot_setup.py       # start here
python scripts/examples/pipeline/80_full_pipeline.py # all of it, in the order a cell needs
```

## One example, one decision

An example covers exactly one choice you have to make, shows the alternatives, and says what each
costs. Not one capability and not one API call: a decision. Where an answer fails, the example runs
the failure, because a refusal is what you will actually meet and it is the most useful thing a file
here can print.

The numbering runs across the folders, so reading them in numeric order is a path from an unopened
box to a trained model. The folders are there so you can go straight to the question you have.

## Nothing moves unless you pass `--live`

Every example that can drive an arm defaults to a rehearsal: it loads the config, builds the real
components, runs every check that needs no motion, and commands nothing. A file that moves a robot
the moment somebody types `python run.py` is a hazard however good its docstring is, because the
reader who most needs the example is the least likely to have read it to the end first.

The banner says which mode you are in before the run, not only after.

These examples otherwise assume you have hardware. They reach for the real driver, the real camera
and the real config and let them refuse, rather than mocking a cell you do not have. `sim/` is the
exception: there the machine, not the cell, is the requirement.

## The cell in front of you

| | example | what it settles |
|---|---|---|
| 01 | [`robot_setup`](cell/01_robot_setup.py) | point the stack at a real cell, and find out what is wrong with it before anything moves |
| 02 | [`which_gripper`](cell/02_which_gripper.py) | which end-effector this cell builds, what each needs wired, and what stands in for it |
| 03 | [`first_pick`](cell/03_first_pick.py) | one grasp, end to end, and a measured account of which layers ran |
| 04 | [`planner_or_ik`](cell/04_planner_or_ik.py) | `robot.ur.motion_planner`: a collision-free plan, or the controller's straight line |

## Where the camera is

| | example | what it settles |
|---|---|---|
| 10 | [`eye_to_hand`](calibration/10_eye_to_hand.py) | a camera bolted in the cell, and the one transform that never changes |
| 11 | [`eye_in_hand`](calibration/11_eye_in_hand.py) | a camera on the flange, and half a transform the arm completes every frame |
| 12 | [`two_cameras`](calibration/12_two_cameras.py) | a second camera, and the two places a two-camera cell is wired wrong |

## What the cell sees

| | example | what it settles |
|---|---|---|
| 20 | [`detector_choice`](perception/20_detector_choice.py) | an open-vocabulary detector against a closed-set one, resolved before a weight loads |
| 21 | [`when_the_detector_is_wrong`](perception/21_when_the_detector_is_wrong.py) | the prompts a phrase grounder answers confidently and wrongly, and where they should go |
| 22 | [`depth_source`](perception/22_depth_source.py) | a stereo pair against an RGB-D rig, and which of them a pick can be planned from |

## What refuses

| | example | what it settles |
|---|---|---|
| 30 | [`the_six_guards`](safety/30_the_six_guards.py) | the six guards, each one made to refuse on purpose |
| 31 | [`capsule_or_mesh`](safety/31_capsule_or_mesh.py) | `safety.self_collision.backend`, and the failure that does not refuse |
| 32 | [`declare_your_bench`](safety/32_declare_your_bench.py) | the path between two checked endpoints, and the three keys that make it visible |

## How a grasp is chosen

| | example | what it settles |
|---|---|---|
| 40 | [`jaw_or_suction`](grasping/40_jaw_or_suction.py) | a jaw or a cup, and the two different questions they ask |
| 41 | [`geometric_or_deep`](grasping/41_geometric_or_deep.py) | `robot.grasping.calculator`, and the refusal when the learned generator has no weights |
| 42 | [`the_thirteen_switches`](grasping/42_the_thirteen_switches.py) | the `grasping.*` blocks that ship off, and the mode gate that keeps most of them off |

## Making your own data

| | example | what it settles |
|---|---|---|
| 50 | [`fetch_or_bring_your_own`](datagen/50_fetch_or_bring_your_own.py) | which objects your corpus is made of, public collections or your own parts |
| 51 | [`author_a_scene`](datagen/51_author_a_scene.py) | what a scene is, and what the layout has decided before anything renders |
| 52 | [`which_engine`](datagen/52_which_engine.py) | which engine settles and renders your scenes, and what each one costs |
| 53 | [`label_and_screen`](datagen/53_label_and_screen.py) | the labels are geometry and the screen is physics; which of the two you ship |
| 54 | [`build_a_corpus`](datagen/54_build_a_corpus.py) | extracting the corpus a generator trains on, and what `--kinds both` changes |

## Learning from it

| | example | what it settles |
|---|---|---|
| 60 | [`your_own_corpus`](train/60_your_own_corpus.py) | the whole chain on your own parts, from a directory of meshes to a report you can read |
| 61 | [`the_foreign_corpus`](train/61_the_foreign_corpus.py) | the other road: import a public corpus, and the licence boundary that gates it |
| 62 | [`recipe_and_tier`](train/62_recipe_and_tier.py) | `--recipe` and `--tier`, and exactly what each one resolves to |
| 63 | [`read_the_report`](train/63_read_the_report.py) | read what a run produced, and the one number that means anything |
| 64 | [`what_the_generator_does_not_cover`](train/64_what_the_generator_does_not_cover.py) | the limits of the learned generator, read off the code rather than promised |

## Where a wrong answer is free

| | example | what it settles |
|---|---|---|
| 70 | [`sim_pick`](sim/70_sim_pick.py) | the same pick in simulation |
| 71 | [`record_a_demo`](sim/71_record_a_demo.py) | a watchable recording of a run |

Both need an Isaac Sim install and its own interpreter, because `import isaacsim` cannot succeed in
this repository's virtual environment however the config is set. They also need the motion engines,
which live outside this repository: point `WILLY_CUROBO_PYTHON` and `WILLY_COAL_PREFIX` at a pair
built by `scripts/ext_deps/install.ps1`, or the cell refuses and names what is missing. Each example
checks that first and refuses with the fix, rather than failing eight imports deep.

## All of it

| | example | what it settles |
|---|---|---|
| 80 | [`full_pipeline`](pipeline/80_full_pipeline.py) | all of it, in the order a cell needs, stopping at the first blocking stage |

It stops at the first blocking stage because continuing past a failed calibration produces a pick
attempt whose result means nothing.

## Three rules the spine enforces

**A finding is not a failure.** "The calibration error is marginal" is a result you need. Treating it
as a crash would hide the rest of the run, so it is reported as a finding and the run continues.

**Not ready is neither.** Exit `2` means the example could not start, because there is no cell, no
corpus or no config, and it names the command that fixes it. Exit `1` means something it was testing
came out wrong. Exit `0` is success. A script chaining examples has to tell those apart.

**A capability you did not exercise is not one you demonstrated.** `03_first_pick` prints which
layers ran, read off report fields that stay `None` when a layer produced nothing, and it says
plainly when the safety layer was not among them. `SafetyPreflight` is constructed inside the vendor
driver, so a rehearsal on a dummy arm reaches no guard at all; `cell.safety()` asks the built arm
what it will refuse, so the printed claim and the check are one thing.

## Two things worth knowing before you edit one

**The calculator comes from the config, never from a class name.** `robot.grasping.calculator`
selects `geometric` or `deep` and `Cell.build()` reaches it through `build_calculator`. No example
names a calculator class, so a cell configured for `deep` cannot quietly run the analytic stack.

**These files are linted and type-checked with the library.** `scripts` is in the same `ruff` and
`mypy` invocations as `src`, `api` and `datagen`, and a new folder is gated on arrival. An example
that nothing checks is documentation that compiles, and documentation that compiles is documentation
nobody verifies.
