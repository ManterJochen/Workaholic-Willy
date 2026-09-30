# The cell and the robot (`src/robot/execution`)

A `Robot` is an arm and the hand on it: connect, move, grasp, pick and place, each answered by a
report. A `Cell` is the whole pick (the robot, its cameras, perception and the grasp stack), and a
`PickRun` is a campaign of picks against a cell, judged by a rule you state.

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
backed out), `RELEASE_NOT_CONFIRMED` and `GRIPPER_FAULT`. A desk arm runs its motions and its report
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

`both_faces=True` asks every pick to see both jaw contact faces of its chosen grasp before it grips, for
safety-critical parts: a pick no view showed both of ends `no_valid_grasp` with nothing gripped.
`record_views=True` keeps each pick's looks for training: the RGB and depth of every look, the tool pose
stamped at each, the intrinsics and the fused target cloud, one `.npz` per wrist pick under
`logs/robot/views`, named after the pick's record (`attempt_id`) and the time; a fixed camera has no
looks to keep, and a file that cannot be written is said while the campaign goes on. Both are off by
default.

Each `PickRun` is **one campaign** of the service (`start_campaign`): fresh push budgets, no part skipped,
and a person's go-ahead after a push that stopped. `push_mm=` is how far a `dense_clutter` push moves a part
([grasping/recovery](../grasping/recovery/README.md)): unset, the cell's
`recovery.fixture.push_distance_mm`; above 50 or under 10 mm it is refused as the run is built, and above
the cell's `max_nudge_mm` as its campaign starts, before any pick (`PickRunReport.error`). A campaign
stops on a fault of the cell, a controller that cannot move, a hand that needs a person (`gripper_fault`)
and a recovery that stopped where the arm stands (`needs_person`).

`Robot.pick` lets go of the frames a wrist camera's `Locator.look_around` held in the planner world when
it ends, however it ends, a pick refused before any command included; so does the exit of
`Robot.connected()` and `Cell.connected()` (`lifecycle.let_go_of_held_views`). `place` does not touch
them. A program that tries again after a failed pick looks around again first. A pick and a look around
hold their frames from the first look the arm stands at, never from the pose they started at.

**Every pose a pick meets is screened before it goes there**, by the exact mesh guard and the planner,
with nothing moved (`URRobotArm.screen_configuration`): `clear`, `in the planner's cushion band` (it runs,
and a planned move out of it or into it takes a straight leg of at most 20 degrees per joint), or an `ERROR`
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
| `looks.py` | where a camera looks from before a pick: the joints a program declares, the cell profile's, or home, and the move there |
| `generated_view.py` | the one view a wrist pick generates and its move back (`go_to_generated_view`, `move_back_to_look`): straight joint lines only, run only against the planner world that holds every frame of the pick; with no live world, no wrist camera in it, a `without_camera_world` decline, or a camera world that reads `DECLINED`, `MISSING` or `UNPLANNED`, no view is generated and the move back is refused |
| `record_views.py` | the `.npz` layout `PickRun(record_views=True)` and `Locator.look_around(record_views=True)` write under `logs/robot/views` |
| `lifecycle.py` | connecting and taking down a cell as one transaction, `NoRealGripper`, `TeardownReport` |
| `cell_lock.py` | `CellLock`, `CellBusy`: one owner per controller, shared with the operator console |
| `robot_parts.py` | the arm and the hand a robot section describes, with the readiness gate and every substitution |
| `camera_world_wiring.py` | which cameras feed a cell's live planner world, `CameraWorldRequired` |
| `camera_fusion.py` | `CameraFusionPlan`: which cameras a cell fuses into each object's cloud before it grasps, or what it is missing to |
| `wrist_bodies.py` | the wrist cameras an arm carries, `WristBodyRequired` |
| `planner_start.py` | `PlannerStart`: a cuRobo planner started and stopped at a desk, the configured looks screened on it (`PlannerStartReport.looks`) |
| `hand_eye.py`, `calibration.py` | `HandEyeCalibration`, and the `CalibrationRoutine` that sweeps and solves `AX=XB` |
| `hand_guiding.py` | the console, the stillness gate, the payload question and the red boundaries of a hand-guided arm, `HandGuidingRefused` |
| `teach.py` | `teach_poses`: joint poses taught by guiding the arm by hand, each screened once held, printed to paste and kept in `logs/taught_poses.json` |
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
- Tests: `tests/test_robot.py`, `tests/test_robot_moves.py`, `tests/test_robot_pick_and_place.py`, `tests/test_cell.py`, `tests/test_pick_run.py`, `tests/test_cell_lifecycle.py`, `tests/test_hand_eye_calibration.py`
