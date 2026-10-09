# The cell and the robot (`src/robot/execution`)

A `Robot` is an arm and the hand on it: connect, move, grasp, pick and place, each answered by a
report. A `Cell` is the whole pick (the robot, its cameras, perception and the grasp stack), a
`PickRun` is a campaign of picks against a cell, judged by a rule you state, and a **task**
(`run_task`) picks a part, sets it down, returns and looks again, once or until nothing is left: the
operator console's unit of work.

```python
from willy import Cell, PickRun, Recording, load_tree

cell = Cell.from_tree(load_tree(), prompt="a red cube")   # the cell WILLY_PROFILE names
print(cell.preflight())    # every stop-the-cell condition a desk can decide, each with its fix
cell.build()               # drivers, cameras, models and the grasp stack; nothing moves
print(cell.safety())       # what this arm refuses, asked of the arm that was built

report = PickRun.from_cell(cell, runs=3, recording=Recording.off()).execute()
print(report)              # one connect, three picks, the teardown and one verdict
raise SystemExit(report.exit_code)   # 0 passed, 1 refused, 2 did not pass, 3 a fault stopped it
```

The arm and its hand alone, with no pick service:

```python
from willy import Robot, load_tree

robot = Robot.from_tree(load_tree())
with robot.connected(), robot.without_camera_world("bench run, the table is clear"):
    print(robot.home())
    print(robot.pick(robot.tool_down(450.0, 100.0, 120.0), 40.0))   # standoff, line in, close, line out
```

Every motion of a real arm goes through its safety pipeline. A robot handed no camera has no camera
world, so each motion says why it needs none: `without_camera_world(reason)` for a block, `decline=`
on one verb. The command line over `Cell` and `PickRun` is `python -m src.robot.execution.real_cell`
([real_cell/](real_cell/README.md)). The examples bring a cell up in order, from
[03_connect_and_move.py](../../../examples/real_robot/03_connect_and_move.py) to
[12_pick_with_the_camera.py](../../../examples/real_robot/12_pick_with_the_camera.py), and
[01_rehearse_a_pick.py](../../../examples/simulation/01_rehearse_a_pick.py) runs a campaign at a
desk on a dummy arm.

## The nouns

| Noun | Built by | Verbs | Returns |
| --- | --- | --- | --- |
| `Robot` | `from_tree`, `from_config`, `from_parts` | `connected()`, `move`, `move_joints`, `home`, `grasp`, `release`, `pick`, `place`; `tool_down(x, y, z, *, yaw_deg, closing_axis)`, a BASE pose through the cell that moves nothing: straight down, the fingers along `robot.natural_closing_axis`, else along x, `yaw_deg` counted from that axis (on a `"-y"` cell `yaw_deg=90.0` closes along base x) | `MotionReport`, `HandReport`, `HandlingReport`; `tool_down` a `Pose`, `ValueError` for a value that names no axis, `TypeError` for another type |
| `Cell` | `from_tree`, `from_robot_config`, `rehearsal` | `preflight()`, `start_planner()`, `build()`, `safety()`, `connected()` | `PreflightReport`, `PlannerStartReport`, `SafetyAttestation` |
| `PickRun` | `from_cell` (it owns the connect), `from_service` (you do); `look=` where each pick looks from, `put_back=True` to put each lifted part back, `both_faces=True` to grip only on both jaw contact faces seen, `record_views=True` to keep each pick's looks, `push_mm=` how far a push moves a part | `execute()` | `PickRunReport`, with one `PickAttempt` per pick: its looks and the ones fused, the jaw faces seen, the generated view, the hand-eye check, `object_mm`, `grasp_pose`, put back and where its views were kept |
| `PassRule` | `PassRule(fraction=1.0, confirm=None)` | `accepts(attempts)` | the verdict rule; the default is every pick |
| `TaskPlan` | `TaskPlan(object, place=PlaceAt(pose=...) or PlaceAt(camera=...), return_to="home", scope="once" or "until_empty", options=TaskOptions(...), which="", source="", more_rules=())`; `which` the one part singled out, grounded alone, `source` where the parts lie (`pick_phrase`: "the gray cube on top of the other one", "each separate gray cube on the black mat"); `more_rules` up to `MAX_FURTHER_RULES` (3) `SortRule(object, place, which="", source="")` make it a sort, each kind to its rule's place (`plan.rules`, its own first) | `run_task(service, plan, hooks=, poses=, known_target=, known_targets=)` | `TaskReport`: why it ended (`TaskStop`), the parts placed, the picks, whether the arm is back, the bin each camera place followed last (`kept_targets`, by phrase), the parts a sort left unsorted; `TaskRefused` before anything moved |
| `Recording` | `Recording.off()`, `Recording.to_file(path)` | | where a campaign appends one record per attempt |
| `HandEyeCalibration` | `from_tree(tree, rig_id=, mode=)`, `from_config`, `from_parts` | `check()`, `run(dry_run=False)` | `CalibrationCheck`, `CalibrationRunReport` |
| `PlannerStart` | `from_robot_config` | `run()` | `PlannerStartReport` |
| `AutonomousGraspService` | `Cell.build()` | `pick()` | `AutonomousGraspReport` ([autonomous_grasp/](autonomous_grasp/README.md)) |

Every report prints as itself, and `to_dict()` gives the same readings as plain data.
`robot.is_holding()` commands nothing and returns the hand's `HoldEvidence`.

A `Cell` runs its steps in the order that makes them safe: the preflight before the build, the safety
attestation before any motion, the cell lock before the connect, the arm before the hand on the way
up and the hand before the arm on the way down. Inside `with cell.connected():`, `cell.robot` is the
built arm and hand as a `Robot`, so its verbs drive the same handles with no second build and no
second lock.

`move`, `move_joints` and `home` end as `EXECUTED`, `MOTION_REFUSED`, `CAMERA_WORLD_UNAVAILABLE` or
`REFUSED`. `pick` and `place` add `NOTHING_HELD` (the close measured nothing, the hand opened and
backed out), `RELEASE_NOT_CONFIRMED` and `GRIPPER_FAULT`, and `pick` adds `CARRIED_RETREAT_REFUSED`
(its line out, judged at the part as if the jaws held the part, would be refused: the jaws stayed open
and the arm went back up the line it came down). A desk arm runs its motions and its report
says `UNPLANNED`.

`PassRule` defaults to unanimity. Without `confirm=`, the service's own word is the evidence of a
success, so a campaign that must not take that word passes a check of its own per attempt.

A wrist camera sees what the arm points it at, so each pick of a wrist cell first moves to its
**looks**: the joint positions the program declares (`JointPositions.deg(...)` takes the pendant's
degrees), else the ones the cell profile configures (`robot.look_joint_positions_deg`), else home. The
pick loop fuses each look with the ones before and stops at the first whose grasp is valid with no
rescan reason; once every look is used up it may generate one more view, on the straight joint line only
([grasping/loop](../grasping/loop/README.md)). A fixed camera moves only to looks the program or the
profile names, and stops at the first look that finds something. Nothing is said to the hand before a
look. With `put_back=True` a campaign places each lifted part back at the pose the tool closed at, so one
part serves every pick, and a part that does not go back stops the campaign ([looks.py](looks.py)).
A look the arm already stands at is not driven to: on a controller's arm at rest and not halted, every joint
within 0.05 degrees of the look and its tool where the look puts it, nothing is sent and the report says
`already there; nothing sent` (the owner's cell paid 1.0 to 1.4 s for that move at every part, its first look
being its home).

`both_faces=True` asks every pick to see both jaw contact faces of its chosen grasp before it grips, for
safety-critical parts: a pick no view showed both of ends `no_valid_grasp` with nothing gripped.
`record_views=True` keeps each pick's looks for training: the RGB and depth of every look, the tool pose
stamped at each, the intrinsics and the fused target cloud, one `.npz` per wrist pick under
`logs/robot/views`, named after the pick's record (`attempt_id`) and the time; a fixed camera has no
looks to keep, and a file that cannot be written is said while the campaign goes on. Both are off by
default. Where the camera's rig records for research (`realsense.record_for_research`, the owner, 2026-10-09), the
file is format 2: per look also both infrared images, the depth as the sensor sent it, the colour frame's exposure
metadata and every segmentation's packed mask, box, label, score, SAM2's predicted IoUs and which one the pick went
for, and once the camera's facts; a file without them is format 1, byte for byte as before. A task on a real cell
takes its picks' looks when each pick ends and has the process's background writer keep them while it goes on
(`keep_pick_views(..., writer=)`): the file it reports is where they are, once the writer reaches them.

Each `PickRun` is **one campaign** of the service (`start_campaign`): fresh push budgets, no part skipped,
and a person's go-ahead after a push that stopped. `push_mm=` is how far a `dense_clutter` push moves a part
([grasping/recovery](../grasping/recovery/README.md)): unset, the cell's
`recovery.fixture.push_distance_mm`, 30 mm where no fixture is declared; above 50 or under 10 mm it is
refused as the run is built, and above the cell's `max_nudge_mm` (50 mm without a fixture) as its campaign
starts, before any pick (`PickRunReport.error`). A campaign stops on a fault of the cell, a controller that
cannot move, a hand that needs a person (`gripper_fault`) and a recovery that stopped where the arm stands
(`needs_person`).

**Judged while the arm waits** (the owner, 2026-10-09, each switch off as shipped). On an arm that gates its own
sends (`safety.dwell.gate_at: send`) the verbs ask with no steady gate of their own: the arm judges first, while it
settles, and waits right before it sends (`motion.gates_its_own_sends`). `pick` and `place` judge their line down
or in in the world their route to the standoff was judged in (`safety.planning_world.hold.standoff`), and `place`
holds the world its line in was judged in through the release and the line out (`hold.drop`). Where the arm judges
the next leg (`robot.motion.judge_next_leg`), a place's release leaves the jaws' stroke to the verb: the one change
goes out, and while the jaws open the line out is judged where the arm stands and the joint move a task declared
after the place (`arm.expecting_next(joints)`, the return) from where the line out ends; nothing is sent before the
stroke is over, and each runs as judged only where nothing it was judged on changed
([drivers/ur](../drivers/ur/README.md#judged-while-the-arm-waits)).

`Robot.pick` lets go of the frames a wrist camera's `Locator.look_around` held in the planner world when
it ends, however it ends, a pick refused before any command included; so does the exit of
`Robot.connected()` and `Cell.connected()` (`lifecycle.let_go_of_held_views`). `place` does not touch
them. A program that tries again after a failed pick looks around again first. A pick and a look around
hold their frames from the first look the arm stands at, never from the pose they started at.

**Every pose a pick meets is screened before it goes there**, by the exact mesh guard and the planner,
with nothing moved (`URRobotArm.screen_configuration`): `clear`, `in the planner's cushion band` (it runs,
and a planned move out of it or into it takes a straight leg of at most 20 degrees per joint), `beside the
boxes the camera saw` (straight lines and moveL run there with a hand known empty and open, and a planned
move into it or out of it is refused), or an `ERROR`
line, with the nearest pose within 20 degrees per joint both clear where one exists:

- **Teaching** (`teach_poses`, example 11) says one line per pose once it is held. The planner starts with
  the first pose, about a minute, and one that cannot start is not asked again that session. It screens
  only on an arm that carries every wrist camera housing the tree hangs on it (`Robot.from_tree`); on any
  other arm it screens nothing and says why in one line.
- **A campaign's start** (`PickRun`) logs one line per look it will visit, the program's or the profile's,
  home where it is one; an `ERROR` line logs at ERROR. It only logs: each pick still judges its looks as it
  moves.
- **The desk** (`PlannerStart`, `Cell.start_planner`, `real_cell --start-planner`) screens
  `robot.look_joint_positions_deg` on the planner it just started, into `PlannerStartReport.looks`; handed
  no camera section, it says the looks were not screened, because which wrist cameras hang on the arm is
  not known.

**What to re-teach.** A `clear` pose needs nothing, and neither does a band pose while its line names a
`Nearby, both clear` pose: a planned move takes its leg by itself. A band pose whose line names none runs on
straight lines only; where a planned move has to reach it or leave it, re-teach it by hand where both clear
and screen it again. An `ERROR` pose goes nowhere: re-teach it at the nearby pose its line names, or by hand
where both clear when it names none, then screen it again.

## A task: pick, place, return

```python
from willy import Cell, JointPositions, PlaceAt, TaskPlan, load_tree, run_task


class Said:
    """What a program hears from a task: each event's sentence, and nothing that stops it."""

    def event(self, name, /, **data):
        print(name, "-", data.get("said", ""))

    def pick_done(self, part, pick, report):
        print(f"part {part}, pick {pick}: {report.outcome}")

    def stop_after_part(self):
        return False

    def halted(self):
        return False

    def abandoned(self):
        return ""


cell = Cell.rehearsal(load_tree("console_dummy").robot)   # a dummy arm and a synthetic scene
cell.build()
with cell.connected():
    report = run_task(cell.service, TaskPlan(object="", place=PlaceAt(pose="drop_left")), hooks=Said(),
                      poses={"drop_left": JointPositions.deg(-60, -95, -120, -55, 90, 0)})
print(report)   # task FINISHED, 1 part placed, the arm back home
```

The console's Start runs exactly this ([`api/README.md`](../../../api/README.md)); a program runs it the same
way, with hooks of its own.

- **The place.** `PlaceAt(pose=name)` is a taught pose from `poses`, and it says where the part's **bottom** is
  let go: the tool goes there raised by the part's hang (the grasp height over the declared support; with no
  grasp pose, `payload.length_mm`), keeping the grasp's tilt at the taught heading. `PlaceAt(camera="blue
  bin")` is a bin the camera finds ([`place_target.py`](place_target.py)): before the first pick the task
  visits its looks and keeps the bin (`survey`), then drops each part at rim + hang + air (`air_mm`, 10 to 50,
  20 unset); a grasp within 5 degrees of vertical is turned about the vertical along
  `robot.natural_closing_axis`. It checks the
  bin again before every drop (`recheck`): moved more than min(100 mm, half its diagonal), another footprint
  (20 %) or another rim (20 mm), and it is lost. A lost bin is looked for again (`robot.place.relocate`, on; the
  owner, 2026-10-09: "wenn sie dies nicht mehr tut, dann kann er seine Ablage nochmal neu errechnen"): the hold of
  the look ends, the task drives its looks once more with the part in the jaws (`place_target.relocate`), and a bin
  of the size the survey found and the colour it followed, standing in no other place, is kept in the old one's
  stead (`task.target_relocated`); the part is carried to the look it was found from, the bin checked there, the
  drop planned anew over it. Once per part: lost again, found nowhere, or with the switch off, the part goes back
  where it was gripped, the arm returns, and the task asks. A fixed camera's bin is looked for where the camera
  stands, before the pick, and nothing moves. A wrist drop holds every frame of the bin's look across the drop and
  the return.
- **A sort** (the owner, 2026-10-09: "Gruene Teile in die gelbe Kiste, rote in die blaue"). `TaskPlan(...,
  more_rules=(SortRule("red part", PlaceAt(camera="blue bin")),))` puts each kind of part where its rule says, up
  to four rules, two of which may share a place. Every place a camera finds is found before the first pick, one
  locate of a class list per look (`place_target.survey_places`), or the task stops `target_not_found` before its
  first pick, a `task.target_missing` for each place it is missing; each bin is kept out of the picks, and a check of
  one replaces its own region alone. Where the detector looks at one bin of several, at a check or a search, it
  grounds them all (`together`), so a second bin in view is never taken for it. Each pick grounds every rule's kind
  in one call ("each separate green part | each separate red part", `PickPrompt.target_labels`), and the part it
  gripped goes by the rule of the kind it went for (`PickReport.target_label`, said as `task.rule`); a gripped part
  no rule names goes back, and the task asks (`target_not_found`). A part no rule clearly claims stays where it lies:
  a sort that ends `nothing_left` names what its last look saw of them (`task.unsorted`, `TaskReport.unsorted`). A
  sort sets every blocker aside, follows no parts from pick to pick, judges no carry ahead of its pick, and runs only
  on a cell whose perception grounds a phrase (else `bad_request`). `TaskReport.kept_targets` keeps the bin of every
  camera place by its phrase, which the next task's `known_targets` looks at first.
- **Set down, not dropped, never piled** (`robot.place`, each switch off as shipped; the owner, 2026-10-08 night).
  `release_in_a_box: below_the_rim` sets a part into a box `below_the_rim_mm` (30, the owner's 10 to 50) under
  its rim, never lower than 10 mm over what the check before the drop reads inside it (its floor, or the parts
  already there, read on the same frame as its rim: `recheck(..., inside=True)`); a box that reads full to within
  10 mm of its rim, a part or an open hand that would not stand `opening_margin_mm` inside its opening, a grasp off
  vertical, or a line in the guard refuses before anything is sent let the part go over the rim, as before
  (`handling.place(..., instead=)`). `part_bottom: measured` hangs the part from what the pick's looks read under
  it, not the declared support (`place_target.part_bottom`: on the cell the drop stood 85 to 92 mm over the rim, a
  55 mm mat under the part), and keeps 5 mm of air over a taught pose. `side_by_side` lays the parts of a flat
  place (a taught pose, a target with no inside) at spots of their own, their reach and `spacing_margin_mm` apart:
  a spot the camera reads free first, else the next of `grid` (rows x columns about the taught pose) this task
  has not filled; no spot left puts the part back and asks (`part_does_not_fit`). `carry: over_the_rim` (or
  `TaskOptions.carry`) carries the part straight over to the bin where one of the pick's looks was the bin's: it is
  checked on that look's frame, the arm's world holds the pick's frames again (`hold_pick_views_again`), and the
  part goes up and over on judged lines, its bottom the rim air and `rim_floor_margin_mm` over the rim and over
  what the looks saw on the way; anything else carries it via the look.
- **The scope.** `once` ends when the part is placed and the arm is back; `until_empty` after
  `empty_looks_to_end` (2) empty looks in a row; `max_failed_in_a_row` (3) failed picks end it where the arm
  stands, and `max_parts` (100) end it `part_limit`. The two rows count apart, and only a placed part starts
  both again. A failed pick whose move back to its look did not run (its report's `stands_at`, `"standoff of
  grasp 3"`, 2026-10-08) ends it `recovery_needs_person` there, naming it, instead of picking again from where
  the arm was left. Its own drop area stays out of its picks (`ExclusionRegion`, [recovery](../grasping/recovery/README.md)):
  the bin's footprint for both scopes, and `pose_keep_out_mm` (150 mm) about a pose drop for "until empty",
  where the arm's own kinematics say where the pose puts the tool. A look that sees only those parts is empty.
- **The end of the search** (the owner, 2026-10-08). An empty pick counts every look it perceived from (its report's
  `looks`): one empty pass over a wrist camera's four looks ends the task, where it took two passes, eight look
  moves and 90 to 115 s on the cell; a pick of one look or none still counts one. And where the part placed was
  the only target its pick's first look counted (`telemetry['targets_by_look']`), the next pick is the check look
  (`TaskOptions.check_look`, on): its first look alone, home, where the return left the arm, with no generated view.
  Nothing there ends the task `nothing_left` at once; a part there is picked; a part there it fails on counts no
  failure, and the pick after it looks from every look. A part only the other looks could see stays behind then:
  the cell's home view shows the whole mat. `task.nothing_found` carries `looks` and `check_look`.
- **Following its parts** (`robot.grasping.follow_parts`, off unless the cell turns it on; the owner, 2026-10-09:
  "Qwen on the first pick and on every trigger, SAM2 on the known parts between"). After each pick the task keeps
  what that pick's first look grounded or followed, less the part it gripped (`KeptScene.of_look`, `without`,
  [perception](../perception/README.md)), and hands it to the next pick (`pick(follow=..., follow_looks=True)`):
  its first look finds the parts again with SAM2 on their boxes and no detector, where nothing changed in depth and
  every part passes every check, and is grounded as before where anything does not. It grounds again after a push,
  a blocker cleared, a recovery, a try that sent motion and failed (a grip that closed on nothing among them), the
  arm left where a try left it, or a gripped part that was not kept, and `refresh_every_picks` after the last
  grounding (0, the default: only on a trigger). A pick that failed with nothing sent keeps what its first look
  saw. The check look is always grounded, and so is the pick after the last part kept: the end of a task is always
  asked of the detector. The memory is the run's alone: a Restart starts with nothing kept, and a halt, a stop or a
  disconnect ends it with the task. Each pick's report says whether it followed (`telemetry['followed']`).
- **The order.** The service's needs-a-person latch first: a task that meets it ends `recovery_needs_person` with
  nothing commanded and the latch kept, since it never starts the campaign that would clear it. Then every taught
  pose it will use is screened before any motion (`pose_refused`), then the setup (push distance, prompt, closing
  axis, overlay, cancel check, regions), put back as found whatever ended it. A Restart's plan
  (`first_motion="return"`) moves to its return pose first.
- **The hand and the carried part.** Before its first motion and before every pick it reads the hand: a toggle the
  program believes closed or nobody can vouch for, a measured hold, or a planner that still models a part ends it
  with nothing moved. After every pick the planner must carry the part (`payload_model` planner and filter), or it
  ends `part_still_held` where the arm stands. A toggle changes DO0 exactly twice per part, and the place leaves the
  count open. Nobody is asked on its thread (`asking_nobody`).
- **The ends.** `TaskStop` has 19 codes in four classes: done (`finished`, `nothing_left`, `part_limit`), the
  operator's (`stopped_after_part`) and the asks (`target_not_found`, `target_lost`, `target_unreachable`,
  `part_does_not_fit`, `pose_refused`) return to `return_to` first; a problem (`halted`, `controller_stopped`,
  `hand_needs_person`, `recovery_needs_person`, `part_still_held`, `return_failed`, `failed_in_a_row`,
  `detector_failed`, `cell_fault`, `disconnected`) leaves the arm where it stands and commands nothing more, an
  output included. A detector that failed is `detector_failed`, never "nothing left".
- **The hooks.** `event` gets each `TaskEvent` with its data and `said`, the library's sentence; `pick_done`
  each pick's report; `stop_after_part` is read at the top of every part and before the survey's looks;
  `halted` and `abandoned` before every motion and in the pick's cancel check.
- **What it refuses** before anything moves raises `TaskRefused` with the console's code: `unknown_pose`,
  `route_refused`, `carried_part_not_modelled`, `object_required`, `closing_axis_refused`, `bad_request`,
  `push_distance_refused`, `camera_target_unavailable`. Every end after the start is the report's.

`AutonomousGraspService.pick(multi_view=False)` looks from the first look only, with no generated view, and
`service.set_closing_axis(axis)` sets a task's axis (`closing_axis_refusal(axis)` says why one would be
refused, with nothing changed). `TaskOptions(every_look=True)` is the console's "Alle Posen" (the owner, 2026-10-08
night): each pick but the check look is handed `pick(every_look=True)` and visits every look, the early stop off;
with `multi_view` it is the console's one choice of looks (first look only, when needed, every look), and every look
with multi-view off is refused (`ValueError`).

## Teaching one pose

The console teaches a pose by hand with `teach_one` ([`teach.py`](teach.py)): one session on a connected arm
that offers freedrive, the arm freed, captured once it stood still for half a second, held, screened at once by
the exact guard and the planner, and written only where both clear it or it lies in the planner's band. An `ERROR`
and an unscreened verdict are never written. The chain ends in the cell's own layer, whose file must exist (one
comment line in `config/robot/robot.cell.yaml` is enough, and git ignores it; the loader refuses a layer with no
file). The store names the file before anything is freed:

```python
from willy import Robot, load_tree
from src.robot.execution.teach import ProfilePoseStore, teach_refusal

tree = load_tree("console_dummy,cell")                               # the chain ends in the cell's own layer
store = ProfilePoseStore(tree.tree, "drop_right", "Ablage rechts")   # refuses here a pose that could not be written
print(store.target)                                                  # the file it goes to: robot/robot.cell.yaml
robot = Robot.from_tree(tree)
with robot.connected():
    print(teach_refusal(robot, tree=tree, name="drop_right", store=store))   # no_hand_guiding on the desk's dummy
```

- **`ProfilePoseStore`** writes through the pose door into the chain's last layer, which git must keep out, and
  refuses as it is made: `invalid_name`, `invalid_label`, `no_layer`, `name_taken` (unless `replace=True`), and a
  word another pose already answers to.
- **`teach_refusal`** says why no pose could be taught now, as `teach_one` would refuse it before anything is freed
  (`TeachRefused` and its `code`: `invalid_name`, `no_hand_guiding`, `not_connected`, `screen_unavailable`,
  `planner_not_ready`, `no_layer`), and the console's `GET /v1/poses` says the same. `ask_planner` is `True` only:
  only a pose both authorities clear can be written.
- **`StillnessGate`** is the hold the console asks for on its own (a lapsed heartbeat, the time limit): the arm is
  held only once its fastest joint stayed below about 1.1 deg/s for half a second, never while it moves in a
  person's hands. Save, Hold, Cancel, a halt and a Disconnect hold at once, even while a Save waits for the arm to
  stand still.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `ConfigError` | `from_tree` is handed a tree that did not load | print the tree; it names the file and the line |
| `CellNotBuilt` | `safety()`, `connected()`, `robot` or `service` before `build()` | call `cell.build()` first |
| `CellBuildRefused`, `ValueError` | the build cannot make this cell, such as a real cell with no CAMERA to BASE | fix what the message names; `preflight()` shows it first |
| `NoRealGripper` | connecting a cell whose hand had to be replaced by a `NullGripper` | name a hand this arm can carry; at a desk use `console_dummy` |
| `CellBusy` | another process holds this controller's lock | the message names the holder |
| `LockKeyRequired` | `Robot.from_parts` with an arm that drives a controller and no lock key | pass `lock_key="ur@<ip>"`, or `lock_key=None` on purpose |
| `CameraWorldRequired` | a cuRobo arm is handed a calibrated camera that gives it no live world | enable `safety.planning_world`, as the message says |
| `WristBodyRequired` | a wrist camera whose body the cell cannot place | declare and calibrate the rig the message names |
| `REFUSED` on a report | a closed link, a pose not in BASE, a camera world the arm refuses, an arm no planner guards | the report's message says which; nothing was commanded |

`PickRun.execute()` never raises for a refused build or connect: the refusal is on the report, and
its exit code is 1. A UR on the `ik` planner and a KUKA are refused by the robot verbs, because
nothing plans their motions against the camera world or judges their paths.

## Status

| Capability | Evidence |
| --- | --- |
| The pick service on a UR5e with a 2F-85 in Isaac Sim | measured in simulation ([willy_sim](../../willy_sim/README.md)) |
| Hand-eye calibration, eye to hand and eye in hand, through `CalibrationRoutine` | measured in simulation |
| Connect, live telemetry and a refused motion, driven through the operator console | measured against real controller software |
| `Robot`, `HandEyeCalibration`, `Cell` and `PickRun` on a physical arm | run on a physical cell: a UR10 (CB3) with a wrist D415 and a `jaw_io` Hand-E, through examples 03, 09 and 10, 12 and 13; not the looks, the generated view or the push |
| `run_task` with a pose place, a camera place, the stops and the put-back | the offline suite on doubles and the real service with fake perception (`tests/test_a_task_*.py`); the console's task on `console_dummy` |
| A task on a controller: three cycles with 2 DO0 edges each, halt in the approach and the carry, the way back | measured against real controller software: URSim CB3 (`scripts/ursim/probe_console_task.py`) |
| A task, a camera place or `teach_one` on a physical arm | never touched hardware |
| The KUKA driver behind the same nouns | never touched hardware |

The default pick is open-loop: perceive, rank, gate, move, log. The decision gate,
recovery, fusion and the learned layers ship `enabled: false`, and a pick report's
`layers` line reads `(none)` on the shipped tree. A rehearsal on the dummy arm proves the wiring, not
a pick.

## Files

| File | Holds |
| --- | --- |
| `robot.py` | `Robot`, `LockKeyRequired` |
| `motion.py`, `handling.py` | the motion verbs and the hand verbs `Robot` delegates to, and their reports; `move_joints_on_the_line`, the straight joint line and nothing else, refused under a `without_camera_world` decline |
| `cell.py` | `Cell`, `CellNotBuilt` |
| `pick_run.py` | `PickRun`, `PickRunReport`, `PassRule`, `Recording`, `PickAttempt`, `PickOutcome`, `configured_looks_of` |
| `looks.py` | where a camera looks from before a pick: the joints a program declares, the cell profile's, or home, and the move there, none where the arm already stands there |
| `generated_view.py` | the one view a wrist pick generates and its move back (`go_to_generated_view`, `move_back_to_look`): straight joint lines only, run only against the planner world that holds every frame of the pick; with no live world, no wrist camera in it, a `without_camera_world` decline, or a camera world that reads `DECLINED`, `MISSING` or `UNPLANNED`, no view is generated and the move back is refused |
| `record_views.py` | the `.npz` layout `PickRun(record_views=True)` and `Locator.look_around(record_views=True)` write under `logs/robot/views`: format 1, and format 2 where the camera records for research (`realsense.record_for_research`: infrared images, raw depth, every mask, box, label, score, SAM2's IoUs and target per look) |
| `lifecycle.py` | connecting and taking down a cell as one transaction, `NoRealGripper`, `TeardownReport` |
| `cell_lock.py` | `CellLock`, `CellBusy`: one owner per controller, shared with the operator console |
| `robot_parts.py` | the arm and the hand a robot section describes, with the readiness gate and every substitution |
| `camera_world_wiring.py` | which cameras feed a cell's live planner world, `CameraWorldRequired` |
| `camera_fusion.py` | `CameraFusionPlan`: which cameras a cell fuses into each object's cloud before it grasps, or what it is missing to |
| `wrist_bodies.py` | the wrist cameras an arm carries, `WristBodyRequired` |
| `planner_start.py` | `PlannerStart`: a cuRobo planner started and stopped at a desk, the configured looks screened on it (`PlannerStartReport.looks`) |
| `hand_eye.py`, `calibration.py` | `HandEyeCalibration`, and the `CalibrationRoutine` that sweeps and solves `AX=XB` |
| `hand_guiding.py` | the console, the stillness gate, the payload question and the red boundaries of a hand-guided arm, `HandGuidingRefused` |
| `teach.py` | `teach_poses`: joint poses taught by guiding the arm by hand, each screened once held, printed to paste and kept in `logs/taught_poses.json`; `teach_one`, `ProfilePoseStore`, `StillnessGate`, `teach_refusal`: the console's one pose, written into the cell's own layer |
| `task.py` | `run_task`, `TaskPlan`, `SortRule`, `PlaceAt`, `TaskOptions`, `TaskHooks`, `TaskReport`, `TaskStop`, `TaskRefused`: pick, place, return, once or until empty; a sort of up to four rules |
| `place_target.py` | the bin a camera finds (`survey`, `recheck`, `KeptTarget`), every bin of a sort in one locate per look (`survey_places`, `PlacesSurvey`), a bin the check lost found again (`relocate`, `Relocated`), each locate of one bin of several grounding them all (`together`), the drop over its rim or below it (`drop_plan`, `InsideRead`) and over a taught pose (`pose_drop`), the screen of each, the hang from what the part stood on (`part_bottom`), the spots of a flat place (`grid_spots`, `top_spots`, `choose_spot`), and what the pick's own looks saw (`recheck_on_look`, `highest_on_the_way`) |
| `pose_provider.py` | workspace-checked and diversity-checked TCP poses for a sweep |
| `ik_service.py` | reachability through the live controller; `URAnalyticIKService` needs `ur_ikfast`, which is not on PyPI |
| `runtime_pick.py` | `RuntimePickService`, one open-loop attempt and its `PickSessionReport` |
| `calibration_watchdog.py` | pure drift and out-of-distribution evaluators |
| [`autonomous_grasp/`](autonomous_grasp/README.md) | the pick service a cell builds, its modes and its report |
| [`real_cell/`](real_cell/README.md) | the command line, the desk checklist and the calibration command |

Import the nouns through `willy`; `src.robot.execution` resolves the same names lazily and loads no
vendor SDK on import.

## Details

- Guide: [robot and safety](../../../docs/guide/04-robot-and-safety.md), [the pick loop](../../../docs/guide/05-pick-loop.md)
- Runbooks: [bringing up a cell](../../../docs/runbooks/cell_bringup.md), [the first pick on a physical arm](../../../docs/runbooks/real_cell_first_pick.md)
- The guards every motion passes: [robot/safety](../safety/README.md); the contract the drivers keep: [robot/core](../core/README.md)
- The operator console drives this layer over HTTP ([api/](../../../api/README.md)); nothing under `src/` imports it
- Tests: `tests/test_robot.py`, `tests/test_robot_moves.py`, `tests/test_robot_pick_and_place.py`, `tests/test_cell.py`, `tests/test_pick_run.py`, `tests/test_cell_lifecycle.py`, `tests/test_hand_eye_calibration.py`; the task in `tests/test_a_task_*.py`, `tests/test_a_bin_is_dropped_into_just_above_its_rim.py`, `tests/test_a_moved_bin_is_followed_only_close_by.py` and `tests/test_a_bin_that_moved_is_found_again_before_the_drop.py`; a sort in `tests/test_a_sort_*.py` and `tests/test_a_moved_place_is_found_again.py`; teaching one pose in `tests/test_one_pose_is_taught_screened_and_held.py`
