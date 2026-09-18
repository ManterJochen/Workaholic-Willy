# `scripts/trial/`: the customer chain trial

Whether a customer with a UR arm and a gripper this repository never shipped can build their combination with the
scripts, from a registry file to a planner that starts. This folder answers it by running
[`docs/runbooks/your_own_gripper.md`](../../docs/runbooks/your_own_gripper.md) itself, in a copy of the tree, with two
hands whose answers are known before the chain runs. Nothing here is product code, and nothing a trial makes is
committed except its record.

| File | What it proves |
|---|---|
| [`run_runbook.py`](run_runbook.py) | Runs the runbook's own marked blocks against a plan, so the runbook is the commands that ran. Unbound names refuse the plan before anything runs; the first unexpected result stops it; hidden files come back with their bytes verified; the record binds the run to the runbook text by hash. |
| [`customer_hand_trial.json`](customer_hand_trial.json) | The plan: the environment the copy reaches cuRobo and Coal through, and phases P0 (the original, read only), P1 (the copy and its isolation, with a control gate that must write the committed bytes), P2 (`acme_dims`, a hand from its numbers), P3 (`acme_mesh`, vendor STL files of the Hand-E), P4 (the original did not move) and P5 (the suite in the copy). |
| [`tree_copy.py`](tree_copy.py) | The copy holds every file git sees, byte for byte; the chain's artefact directories resolve inside the copy; the original is unchanged against a snapshot; and which tests the chain touches, derived. |
| [`customer_hands.py`](customer_hands.py) | The two hands. `acme_dims` is invented, and its boxes were computed by hand. `acme_mesh` is the committed Hand-E exported as STL files in a vendor frame nobody uses, so the mesh writer must give the Hand-E back. Isaac-derived exports stay under the trial's log folder. |
| [`retract_checks.py`](retract_checks.py) | Adding a hand adds its retract rows and moves nothing (`table-diff`, keyed as the merge keys rows), and the exact guard of the STL hand agrees with the Hand-E's over the gate's pose set while another hand does not (`guard-agree`). |
| [`cell_probes.py`](cell_probes.py) | A cell naming the hand runs its registry numbers (`derived`), and the UR driver builds from the chain without connecting (`build-arm`). The planner start is the product's own, `real_cell --start-planner`. |

## Running it

On the box, with nothing else on the GPU and no suite running, and with no edit to the original while it runs (P4
compares it with P0):

```bash
.venv/Scripts/python.exe scripts/trial/run_runbook.py --runbook docs/runbooks/your_own_gripper.md \
    --plan scripts/trial/customer_hand_trial.json --root D:/willy_trial/customer_hand \
    --logs D:/dev/Workaholic-Willy/logs/box_<date>/customer_hand_trial
```

The copy must not exist before the run; delete it afterwards. A break stops the run: fix it in the tree with a test first,
delete the copy, and start again. `--from ID` resumes inside one copy only after an interruption with no source change.
