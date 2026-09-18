# Customer chain trial (`scripts/trial/`)

This folder checks that [`docs/runbooks/your_own_gripper.md`](../../docs/runbooks/your_own_gripper.md)
works as written: that a customer with a UR arm and a gripper this repository never shipped gets from
a registry file to a planner that starts by running the runbook's own commands. You do not need it to
use the library; it is the instrument that keeps that runbook honest.

It runs the runbook in a fresh copy of the tree, with two hands whose answers are known before the run.
Nothing here is product code, and a trial commits nothing except its record.

## Running it

Run it with nothing else on the GPU, no test suite running, and no edit to the checkout while it runs:
phase P4 compares the checkout with the snapshot phase P0 took. From the repository root:

```bash
.venv/Scripts/python.exe scripts/trial/run_runbook.py --runbook docs/runbooks/your_own_gripper.md \
    --plan scripts/trial/customer_hand_trial.json --root D:/willy_trial/customer_hand \
    --logs D:/dev/Workaholic-Willy/logs/customer_hand_trial
```

- Both paths are absolute. `--root` is the copy, and it must not exist yet; delete it afterwards.
- The plan's `bindings` name this checkout's paths (`ORIG`, `PY`, `CONTENT`). On another machine, edit
  them first.
- Exit 0: every step went as expected. Exit 1: the run stopped at a step, and the line printed says
  which and why. Exit 2: the plan or the runbook is malformed, and nothing ran.
- A stop means a defect: fix it in the tree with a test first, delete the copy, and start again.
  `--from ID` resumes inside the same copy, and only after an interruption with no source change.

## The phases

| Phase | What it does |
|---|---|
| P0 | snapshots the checkout, read only |
| P1 | makes the copy and proves it is isolated; a control gate must write the committed bytes |
| P2 | `acme_dims`: a hand built from its numbers |
| P3 | `acme_mesh`: a hand built from vendor STL files of the Hand-E |
| P4 | proves the checkout did not change |
| P5 | runs the test suite in the copy |

## Files

| File | Holds |
|---|---|
| [`run_runbook.py`](run_runbook.py) | the runner: runs the runbook's marked blocks against the plan, and records the run |
| [`customer_hand_trial.json`](customer_hand_trial.json) | the plan: the environment, the bindings and the phases with their expected results |
| [`tree_copy.py`](tree_copy.py) | the copy, its isolation, the snapshot, and which tests the chain touches |
| [`customer_hands.py`](customer_hands.py) | the two hands and their known answers |
| [`retract_checks.py`](retract_checks.py) | `table-diff` and `guard-agree`: a new hand adds its retract rows and moves nothing else |
| [`cell_probes.py`](cell_probes.py) | `derived` and `build-arm`: a cell naming the hand, and the UR driver built without connecting |

## What the runner guarantees

- The runbook is the commands that ran. The runner executes only fenced blocks that carry a
  `<!-- step: ID -->` marker, and the record holds a SHA-256 over every step's id, kind and text.
- A name the plan does not bind refuses the plan before anything runs.
- The first unexpected result stops the run.
- Every file a step hides is restored at the end, and its bytes are verified.
- Each command runs without a shell, with its own output files and a timeout that ends its whole
  process tree.

## The two hands

- `acme_dims` is invented. Its collision boxes were computed by hand, so the chain's result has an
  answer to meet.
- `acme_mesh` is the committed Hand-E, exported as STL files in a vendor frame nobody uses. The mesh
  writer has to give the Hand-E back. The export is Isaac-derived geometry, so it is written only under
  the trial's log folder, never into the tree.

The planner start at the end of the chain is the product's own,
`python -m src.robot.execution.real_cell --start-planner`.

## Details

- The runbook this trial runs: [`docs/runbooks/your_own_gripper.md`](../../docs/runbooks/your_own_gripper.md).
- The tests that hold each instrument to its known answers: `tests/test_trial_*.py`.
