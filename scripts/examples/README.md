# `scripts/examples/`: one subject per file, through both doors

Every capability in this repository reaches an operator twice: through Python, and through
`python -m <pkg>`. That is not a convention this directory invented, it is
[the calling convention](../../src/contracts/README.md) the library is built on,
and its first sentence is that the two callers "have to be the same answer in two costumes".

So the examples come in pairs.

```
api/01_first_cell/one_pick_end_to_end.py     the Python door
cli/01_first_cell/one_pick_end_to_end.ps1    the same subject, from a shell
cli/01_first_cell/one_pick_end_to_end.sh     the same again, on a machine without PowerShell
```

Run one:

```bash
python scripts/examples/api/01_first_cell/one_pick_end_to_end.py
```

## What an example is, and what it is not

An example **teaches**. It is straight-line code with numbered steps, it calls the real library
nouns, it prints what came back, and it has no exit code worth reading. It fits on a screen.

A check **answers**. It holds this cell against its own configuration and refuses when the two
disagree, which is what lets an operator put it in a bring-up list. Those live next door in
[`../checks/`](../checks/) and they keep their exit codes.

The line between them is the verdict, not the length. A file that can only say "here is what
happened" is an example; a file that can say "this is wrong" is a check.

| | example | check |
|---|---|---|
| lives in | `scripts/examples/api/` | `scripts/checks/` |
| answers | what the call returns | whether this cell is coherent |
| exit code | always 0 | 0 ok, 1 wrong, 2 nothing to check |
| flags | none | at most one, and only where connecting needs consent |

## Three rules, and each one has a scar behind it

**Nothing here writes to the process environment.** Select a profile the way every other tool
selects one, by setting it around the command: `WILLY_PROFILE=ur3e python scripts/checks/...`. The
previous generation took a `--profile` flag and set `WILLY_PROFILE` itself without putting it back.
One run leaked it, and a safety-guard test three files away then failed inside the suite while
passing on its own, because it was reading a different cell's workspace box. The symptom appeared
three files from the cause. A flag that mutates the environment is a second way to say what the
environment already says, and two ways to say one thing is the defect rather than the leak.

**Every example runs on a machine with nothing attached.** No camera, no robot, no GPU, no
downloaded model, no generated corpus. Where an example needs one it checks and prints what is
missing, because a reader who cannot run it learns nothing from a stack trace and everything from a
sentence naming the file that is absent. That is what lets
[`tests/test_examples_run.py`](../../tests/test_examples_run.py) execute all of them in CI, which is
the only place a rot guard is any use.

That test runs each api example four ways: on this checkout, on a config tree with no `robot`
block in it, on a tree whose YAML does not parse, and in a working directory it cannot write.
It also runs the cli halves, both costumes of each, minus the subjects that need a corpus, the
network, an optional engine or a simulator, which are named one by one with the measurement
behind each reason. Where a cli step calls a check, the step reports what the check said rather
than inheriting its exit code: an example has no verdict to give, so a refusal it triggers is a
thing to print and not a thing to become.

**Nothing here shares a spine.** Each file stands alone. The previous generation shared a 232-line
`_common.py` that all twenty-nine imported, and the cost was that reading one example meant reading
a framework first. Twenty-nine files that read as one tool is a good property for a tool and a bad
one for twenty-nine examples.

## Why this was rewritten

The generation before this one was measured at 8,492 lines across thirty files, an average of 283
per example, 1,563 of them lines of pure output. One file was 375 lines wrapped around six library
calls. They were good tools and they were not examples.

⛔ **And nothing ran them.** Their own shared spine said it was kept honest by "the tests that keep
them from decaying into prose", and no test imported an example. The replacement inherited half
of that hole and kept it until 2026-09-10: the api files were executed from the day they were
written, while the forty-eight cli files were only checked for existing, which is how three of them
came to exit 1 on a bare box with nobody noticing. Four files mentioned the directory
and every mention was prose in a docstring. Two defects had been sitting in one of them unnoticed:
it caught `SystemExit` where the library had started raising `CellBuildRefused`, and its headline
lesson had been reversed in the library and never in the file. Both were found by reading, which is
not a method that scales.

The replacement is an executor rather than a better search. A dead reference assembled at runtime,
`"scripts" / "examples" / f"{name}.py"`, exists nowhere in the source as text, so a grep cannot see
it even in principle. Only running the file can.

## The folders

Numbered by the order a cell needs them, not by importance. Read them in order to go from an
unopened box to a trained model, or go straight to the question you have.

| | |
|---|---|
| `01_first_cell` | which gripper gets built, one pick end to end, planner or IK |
| `02_calibration` | the fixed camera, the wrist camera |
| `03_perception` | which detector, what to do when it is wrong, where depth comes from |
| `04_safety` | capsule or mesh, and gating the whole path |
| `05_grasping` | jaw or suction, geometric or deep |
| `06_datagen` | parts, scenes, engines, labels, and a corpus |
| `07_training` | your own corpus, a public one, the recipe, and what the generator does not cover |
| `08_sim` | the pick rate, and recording one |

File names say what the file does. The numbers are on the folders, because the reading order is a
property of the sequence and not of any one subject.
