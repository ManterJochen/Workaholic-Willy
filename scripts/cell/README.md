# A cell's logs, read back (`scripts/cell/`)

**Where a pick's seconds go, stage by stage, from the logs a cell already writes.** `pick_timeline.py` reads a
cell's logs folder as the console writes it and splits every task into its survey, its picks and its places, and
each of those into stages: the command read, the look and its grounding (Qwen, SAM2), the grasp search, the judging
before the arm leaves the look (world refresh, IK, guard, cuRobo), every motion, the waits at the standoff and in
the jaws' settles, the carry, the drop and the way home. It prints each stage's best, median and worst over the
folder, and with two folders it sets A against B, stage by stage, with what the speed cost beside it: picks gripped,
parts placed, tries refused.

```bash
python scripts/cell/pick_timeline.py D:/cell/logs                    # every stage: n, best, median, worst
python scripts/cell/pick_timeline.py logs_block3_A logs_block3_B     # A against B, stage by stage
python scripts/cell/pick_timeline.py D:/cell/logs#last               # only the runs of the last connect
python scripts/cell/pick_timeline.py D:/cell/logs#4                  # only the runs of connect 4, as listed
python scripts/cell/pick_timeline.py D:/cell/logs@09:30-10:15        # only the runs that start in that window
python scripts/cell/pick_timeline.py D:/cell/logs --picks            # one line per pick and per place besides
python scripts/cell/pick_timeline.py D:/cell/logs --json day.json    # everything, every pick's stages included
python scripts/cell/pick_timeline.py --stages                        # what each stage holds
```

It writes nothing but the `--json` file and needs nothing but Python: it runs on the cell PC beside the console, or
on any copy of the folder. A copy taken while the console runs holds every run since the logs began, so
a block of runs is picked out of it by its connect (`#4`, `#last`) or its time window: the listing at the top names
every connect with its config fingerprint and the selector that keeps it. A switch changed in the config is a new
build and a new connect, so `<folder>#last` after each half of an A/B block is that half alone.

## What it reads

| In the folder | What it gives | Without it |
|---|---|---|
| `robot/robot.log` (rotated `robot.log.1` ... included), or the per-module files under `robot/` | every motion, refresh, try and jaw command | nothing to time |
| `models/*.log` | each grounding's length, Qwen's and SAM2's share, a colour Qwen named | a grounding is timed inside its look |
| `api/*.log` | the runs, the sentences read, each pick's end and outcome, the connects | one run over the whole folder, the picks told apart by their grasp records |

A line the aggregate `robot.log` and a module file both hold is read once. A line no pattern knows is passed over,
so a log with lines this script never saw still splits.

## How a stage is timed

A stage runs from the line that starts it to the line that starts the next. Most lines are written where what they
name begins (`try 1 of 12`, `move to approach_00: ...`, `Moving to 'retreat'`, `actuating jaws CLOSED`). A few are
written at the end with their length in them, and begin that long before: a grounding (`perceived 3 object(s) ... in
13165.6 ms`), a planner world refresh (`806.2 ms to build, 245.7 ms to register`), a command read. A motion whose end
no line names runs until the first line of what comes after it, so it holds the steady gate and whatever the
controller is asked before that line: the line down holds the lift's IK at the part, the line out the start of the
way home's judgement.

| Stage | Part | What it holds |
|---|---|---|
| `command` | task | the sentence read: by Qwen, or a known sentence, or an answer from memory |
| `to_start` | task | from the reading to the task's start: the card and the person's click, or Enter |
| `survey` | task | from the task's start to its first pick: the countdown, a restart's way home, the bin found |
| `look` | pick | a look judged and driven to (or skipped where the arm stands there), its frame taken |
| `grounding` | pick | the detector on the look's frame: Qwen and SAM2, and a colour Qwen is asked to name |
| `grasp_search` | pick | the grasp calculator over the parts the look saw |
| `judging` | pick | the tries judged before the arm leaves the look: world, IK, guard, cuRobo; refused tries |
| `push`, `blocker` | pick | a push of the part, or a blocker searched for and taken away, their own motions included |
| `approach` | pick | the route to the standoff, from its send to `judged path executed` |
| `standoff` | pick | at the standoff: the steady gate, the line down solved and judged, the move's checks |
| `line_down` | pick | the moveL down, then the lift solved at the part, to the first line of its judgement |
| `at_part` | pick | the lift judged at the part as if the jaws held it, to the close |
| `close` | pick | the jaws closed and settled, the line up judged, to its send |
| `lift` | pick | the moveL up, to the pick's end (a record written before the pick ends included) |
| `place_judging` | place | from the pick's end to the place's first motion |
| `carry` | place | the carry to the bin's look, and the bin checked by depth there |
| `bin_check` | place | the bin checked again by the detector |
| `drop_judging` | place | the route to the drop's standoff judged |
| `to_drop` | place | the route to the drop's standoff, from its send to `judged path executed` |
| `drop_standoff` | place | at the drop's standoff: the line in solved and judged |
| `line_in` | place | the moveL in, to the jaws opening |
| `release` | place | the jaws opened and settled, the line out judged, to its send |
| `line_out` | place | the moveL out, to the first line of the way home's judgement |
| `return_judging` | place | the way home judged |
| `return` | place | the move home, to the next pick's first line or the task's end |

Beside the stages it sums what said its own length inside a pick (`in qwen`, `in sam2`, `in world` with its camera
read and its registration, `in curobo`), counts what a pick did (tries, refusals, looks, refreshes, held worlds,
mesh-first judgements, routes run as judged ahead), and times `pick`, `place` and `cycle` whole. A pick that tried no
grasp (the until-empty task's check look) is an `empty pick`; one a disconnect cut short is counted, not timed.

## What it cannot see

- **Lines that reach no file.** A module whose logger is a bare `logging.getLogger(__name__)` writes nothing: the
  console configures no root logger. Today that is the task's own lines (`task: ...`: a pick that follows its parts,
  a carry over the rim, the spot a part is laid at), the drop under the rim, the survey and the bin's recheck by
  depth, the look the arm already stands at, and the grasp workers' start. So a skipped look is timed inside `look`
  and a recheck by depth inside `carry`. The look's, the survey's, the recheck's and the workers' lines are read
  already, the day their modules write them to a file.
- **A hand that logs no jaw command.** The Robotiq socket driver logs its moves at debug, so on a URCap hand `close`
  is timed inside `at_part` and `release` inside `line_in`.
- **What a motion took on the controller.** A motion's end is the next line, not the controller's own report.
