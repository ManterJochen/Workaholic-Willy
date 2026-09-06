# `scripts/examples/`: seven executable examples, in dependency order

Each one is a file you run, not a list of commands to copy. They share a spine
([`_common.py`](_common.py)) so seven scripts read as one tool: the same step and verdict shape, the
same exit codes, the same rule about hardware. Every example drives the real library nouns rather than
re-implementing a check, so a pipeline the example demonstrates is a pipeline that exists.

```bash
python scripts/examples/01_robot_setup.py            # start here, with or without a robot
python scripts/examples/06_full_pipeline.py          # all of it, in the order a cell needs
```

## Nothing moves unless you pass `--live`

Every example that can drive an arm defaults to a rehearsal: it loads the config, builds the real
components, runs every check that needs no motion, and commands nothing. A file that moves a robot the
moment somebody types `python run.py` is a hazard however good its docstring is, because the reader who
most needs the example is the least likely to have read it to the end first.

The banner says which mode you are in before the run, not only after.

## The seven

| | example | robot | what it settles |
|---|---|---|---|
| 01 | [`robot_setup`](01_robot_setup.py) | optional, `--live` connects | is this cell described coherently, and does the controller agree |
| 02 | [`calibration`](02_calibration.py) | `--live` to move | where is the camera, in the robot's frame |
| 03 | [`pick`](03_pick.py) | optional, `--live` picks | one grasp, and which layers actually ran |
| 04 | [`datagen`](04_datagen.py) | no | build a corpus, and price it first |
| 05 | [`train`](05_train.py) | no | train a model, and read the number that means something |
| 06 | [`full_pipeline`](06_full_pipeline.py) | optional | all of it, stopping at the first blocking stage |
| 07 | [`sim`](07_sim.py) | no, needs an Isaac Sim install | the same pick where a wrong answer is free |

Run them in order the first time. The order is a dependency chain and each link is a failure a cell
can produce: without 02 the driver refuses every motion as `INVALID_TARGET`, which reads as a broken
cell rather than an uncalibrated one, and a corpus from 04 with no grasp labels trains successfully in
05 and teaches nothing.

`06_full_pipeline` calls five of the siblings as the files you can also run alone, in the order
`01, 02, 04, 05, 03`. Two of the five are conditional, so a plain invocation runs four: 02 needs
`--rig`, and `--skip-corpus` drops 04 and 05. It stops at the first blocking stage, because continuing
past a failed calibration produces a pick attempt whose result means nothing.

07 is not in that chain. Isaac Sim ships its own interpreter, so `import isaacsim` fails in this
repository's virtual environment however the config is set, and 07 must be run with that install's
`python.bat`. The example checks that first and refuses with the fix rather than failing eight imports
deep.

## Three rules the spine enforces

**A finding is not a failure.** "The calibration RMSE is marginal" is a result the reader needs.
Treating it as a crash would hide the rest of the run, so it is reported as a finding and the run
continues.

**Not ready is neither.** Exit `2` means the example could not start (no cell, no corpus, no config)
and it names the command that fixes it. Exit `1` means something it was testing came out wrong. Exit
`0` is success. A script chaining examples has to tell those apart.

**A capability you did not exercise is not one you demonstrated.** `03_pick` prints which layers ran,
read off report fields that stay `None` when a layer produced nothing, and it says plainly when the
safety layer was not among them. `SafetyPreflight` is constructed inside the vendor driver, so a
rehearsal on a dummy arm reaches no guard at all; `cell.safety()` asks the built arm what it will
refuse, so the printed claim and the check are one thing.

## Two things worth knowing before you edit one

**The calculator comes from the config, never from a class name.** `robot.grasping.calculator` selects
`geometric` or `deep` and `Cell.build()` reaches it through `build_calculator`. No example names a
calculator class, so a cell configured for `deep` cannot quietly run the analytic stack.

**These files are linted and type-checked with the library.** `scripts/examples` is in the same `ruff`
and `mypy` invocations as `src`, `api` and `datagen`. An example that nothing checks is documentation
that compiles, and documentation that compiles is documentation nobody verifies.
