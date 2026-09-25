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
from willy import Pose, Robot, load_tree

robot = Robot.from_tree(load_tree())
with robot.connected(), robot.without_camera_world("bench run, the table is clear"):
    print(robot.home())
    print(robot.pick(Pose.tool_down(450.0, 100.0, 120.0), 40.0))   # standoff, line in, close, line out
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
| `Robot` | `from_tree`, `from_config`, `from_parts` | `connected()`, `move`, `move_joints`, `home`, `grasp`, `release`, `pick`, `place` | `MotionReport`, `HandReport`, `HandlingReport` |
| `Cell` | `from_tree`, `from_robot_config`, `rehearsal` | `preflight()`, `start_planner()`, `build()`, `safety()`, `connected()` | `PreflightReport`, `PlannerStartReport`, `SafetyAttestation` |
| `PickRun` | `from_cell` (it owns the connect), `from_service` (you do); `look=` where each pick looks from, `put_back=True` to put each lifted part back | `execute()` | `PickRunReport`, with one `PickAttempt` per pick: its looks, `object_mm`, `grasp_pose` and put back |
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

A wrist camera sees what the arm points it at, so each pick of a wrist cell first moves to a look:
joint positions the program declares (`JointPositions.deg(...)` takes the pendant's degrees), tried in
order until one finds something, or home when the program declares none. A fixed camera does not move
to look. Nothing is said to the hand before a look. With `put_back=True` a campaign places each lifted
part back at the pose the tool closed at, so one part serves every pick, and a part that does not go
back stops the campaign ([looks.py](looks.py)).

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
| `Robot`, `Cell`, `PickRun` and `HandEyeCalibration` on a physical arm | never touched hardware |
| The KUKA driver behind the same nouns | never touched hardware |

The default pick is open-loop: perceive, rank, gate, move, log. The decision gate, closed-loop refine
and verify, recovery, fusion and the learned layers ship `enabled: false`, and a pick report's
`layers` line reads `(none)` on the shipped tree. A rehearsal on the dummy arm proves the wiring, not
a pick.

## Files

| File | Holds |
| --- | --- |
| `robot.py` | `Robot`, `LockKeyRequired` |
| `motion.py`, `handling.py` | the motion verbs and the hand verbs `Robot` delegates to, and their reports |
| `cell.py` | `Cell`, `CellNotBuilt` |
| `pick_run.py` | `PickRun`, `PickRunReport`, `PassRule`, `Recording`, `PickAttempt`, `PickOutcome` |
| `looks.py` | where a camera looks from before a pick: the joints a program declares, or home, and the move there |
| `lifecycle.py` | connecting and taking down a cell as one transaction, `NoRealGripper`, `TeardownReport` |
| `cell_lock.py` | `CellLock`, `CellBusy`: one owner per controller, shared with the operator console |
| `robot_parts.py` | the arm and the hand a robot section describes, with the readiness gate and every substitution |
| `camera_world_wiring.py` | which cameras feed a cell's live planner world, `CameraWorldRequired` |
| `camera_fusion.py` | `CameraFusionPlan`: which cameras a cell fuses into each object's cloud before it grasps, or what it is missing to |
| `wrist_bodies.py` | the wrist cameras an arm carries, `WristBodyRequired` |
| `planner_start.py` | `PlannerStart`: a cuRobo planner started and stopped at a desk |
| `hand_eye.py`, `calibration.py` | `HandEyeCalibration`, and the `CalibrationRoutine` that sweeps and solves `AX=XB` |
| `hand_guiding.py` | the console, the stillness gate, the payload question and the red boundaries of a hand-guided arm, `HandGuidingRefused` |
| `teach.py` | `teach_poses`: joint poses taught by guiding the arm by hand, printed to paste and kept in `logs/taught_poses.json` |
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
