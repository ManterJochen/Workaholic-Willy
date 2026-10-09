# UR driver (`src/robot/drivers/ur`)

The driver for Universal Robots arms. It speaks RTDE to the controller through `ur_rtde` (pinned in
`requirements.txt`) and hands the rest of the library millimetres, XYZW quaternions and base-frame
poses. `robot.ur.model` names the arm: `ur3`, `ur3e`, `ur5`, `ur5e`, `ur10` or `ur10e`.

```python
from willy import Robot, load_tree

robot = Robot.from_tree(load_tree())    # a cell with robot.vendor: ur
print(robot.route())                    # cuRobo with the mesh guard, or the controller's own line
with robot.connected(), robot.without_camera_world("bench run, the table is clear"):
    print(robot.home())                 # connect has already run the refusals below
    print(robot.move(robot.tool_down(450.0, 100.0, 300.0)))   # the cell's natural closing axis
```

The whole walk at a cell is [examples/real_robot/03_connect_and_move.py](../../../../examples/real_robot/03_connect_and_move.py);
`create_arm("ur", config=tree.robot)` builds the arm alone. Two command lines belong to this driver:

```bash
python -m src.robot.drivers.doctor --require ur   # exit 0 when ur_rtde imports on this machine
python -m src.robot.drivers.ur --read             # every pin on the tool I/O bank; moves nothing
```

## The nouns

| Noun | Built by | Verb | Returns |
|---|---|---|---|
| `URRobotArm` | `Robot.from_tree(tree)`, or `create_arm("ur", config=tree.robot)` | `move(pose)`, `move_to_joints(joints)`, `get_tcp_pose()` | `MotionResult` with a `MotionStatus`; a base-frame `Pose` |
| `Bench` | `Bench.from_robot_config(tree.robot, arm, confirm=...)` | `run(action)` with `Read`, `Watch`, `Set`, `Pulse` or `Measure` | `BenchReading` |
| `URConnection` | the arm, at `connect()` | the RTDE reads, moves, I/O and payload push | the one place `ur_rtde` is imported |

Without connecting, `line_motion()` says what `move(pose, linear=True)` keeps of the line. Wrench,
digital I/O, robot status and hand guiding are optional capabilities in [robot/core](../../core/README.md).
Two more serve a wrist pick's one generated view and its move back. `nearest_configuration(pose)` names
the joints a pose goes to, as the endpoint gate judges them, and moves nothing; it answers on the `ik`
planner too. `move_to_joints_on_the_line(joints)` drives the straight joint line from where the arm
stands and nothing else: a line that is not clear is refused with nothing sent, never planned around,
and on the `ik` planner, where nobody judges a path, every such motion is refused `UNSUPPORTED`.

## What `connect()` refuses

Each refusal is a `RobotConnectionError`, and the controller is left disconnected. The payload and
the tool frame the base tree ships are refused on purpose: a cell nobody has measured does not move.

| Refusal | When | What to do |
|---|---|---|
| unweighed payload | `safety.payload.enforce: true` with `mass_kg: 0.0`, the shipped value | Weigh the whole wrist assembly and set `mass_kg` and `cog_mm`, or `enforce: false` for a bare flange |
| payload as a point mass | `mass_kg` above 0 with `cog_mm: [0, 0, 0]` | Measure the centre of gravity from the flange, in millimetres |
| undeclared tool frame | `gripper.tool_frame.source: undeclared`, the shipped value | Set `offset_mm` and `rotation_quat_xyzw`, then `source: willy` or `polyscope` |
| wrong model | the dashboard reports another size of UR than `robot.ur.model` | Set `robot.ur.model` to the arm that is plugged in |
| wrong series | a CB-series arm configured as its e-Series namesake or the reverse, and the other DH table fits better | The same key; check the pendant TCP too |
| wrong tool frame | the controller's active tool frame is more than `verify_tolerance_mm` or 5 degrees from the declared one | Match the pendant TCP to the config, or switch `source` |
| arm moving | the tool-frame check pairs joints with a pose, so the arm must be at rest | Let it stop and connect again |
| payload push failed | the controller refused `setPayload` | The connection is rolled back; read the message |

`source: willy` means this driver composes the tool frame and the controller runs a bare flange;
`source: polyscope` means an operator set the TCP on the pendant and the driver only verifies it. The
tool frame is derived, never read: `inv(flange from the DH table) @ getForwardKinematics(q)`. The model
check needs the dashboard: a dashboard that does not answer is logged, not refused.

## What a motion refuses

| Refusal | When | What to do |
|---|---|---|
| `CONTROLLER_REJECTED` | `robot.ur.motion_planner: curobo` and the planner is not available | `python -m src.robot.safety.planning --doctor`; nothing falls back to blind IK |
| `TIMEOUT` | no straight line to the goal is clear and cuRobo found no collision-free plan to any configuration of it | Move the goal or clear the cell; `ur_arm.log` names every configuration tried and why it failed |
| `IK_FAILED` | on `curobo`, the flange goal is out of the arm's reach | Move the goal |
| `JOINT_LIMIT_REJECTED`, no configuration in the window | on `curobo`, no configuration of the goal lies inside the planner's and the joint-limit guard's window | Move the goal; with `within_half_turn_of_home`, it lies behind home |
| `JOINT_LIMIT_REJECTED`, a detour | on `curobo`, every plan cuRobo found swings a joint more than `safety.planned_motion.max_detour_deg` past its span | Clear the way, or raise the bound; the message names the joint |
| `JOINT_LIMIT_REJECTED`, a joint target past a full turn | a joint target holds a value beyond 2 pi, which reads as degrees | Joint targets are radians |
| `SELF_COLLISION_REJECTED` or `JOINT_LIMIT_REJECTED`, the start | the planner will not start from where the arm stands, for more than its cushion band or with no escape leg out of it | Jog the arm out of that configuration; the message names the pair, the planner's depth, the distance the exact meshes keep and why no leg was taken |
| `SELF_COLLISION_REJECTED`, the band's goal | a goal in the planner's cushion band that no approach leg of at most 20 degrees per joint reaches: `no plan reaches it` | Reach it on a straight line, or re-teach it by hand where both authorities clear it and screen it again |
| `SELF_COLLISION_REJECTED` or `JOINT_LIMIT_REJECTED`, a leg | the path gate or the planner refuses a leg of the plan that would run | Read the message; nothing has moved |
| `SELF_COLLISION_REJECTED`, `it stands: a part is carried` | on `curobo`, only the camera's boxes refuse the planner's world while a part is carried: any attach since the last detach, modelled or not | Open the jaws on it (`Robot.release`, which forgets the part); the arm leaves empty-handed |
| `SELF_COLLISION_REJECTED`, `the hand is not known to be empty and open` | on `curobo`, only the camera's boxes refuse the planner's world, and the hand reads closed, unvouched or short of open, or none was handed over (`set_hand`) | Open the hand, or answer where the toggle's jaws stand; the sentence says what it read |
| `INVALID_TARGET`, a waypoint | `ur_rtde` refused a speed or acceleration before sending that waypoint's `moveJ` | The message says which waypoint and what ran before it |
| `UNSUPPORTED`, camera world MISSING | on `curobo`, a motion with neither a live camera world nor a stated decline | Hand the robot its cameras, or `without_camera_world(reason)` |
| `CONTROLLER_REJECTED`, hand-guided | any motion verb while a `freedrive()` session is open, free or held | Leave the session; its end holds the arm and gives motion back |
| `CONTROLLER_REJECTED`, plan off its goal | a plan ending more than 5 mm or 6 degrees from its goal on the DH chain | Read the message; nothing has moved |
| `CANCELLED`, halted | the halt latch is set: nothing is sent; or a halt braked the move in flight, and a judged path sends no later waypoint | A person says the cell is clear (`clear_halt()`), then the way back |
| a refused straight line | `linear=True` and a joint turns over 0.35 rad between samples, or the flange leaves the line by 1 mm | Plan the move in legs, or drop `linear` |

`move_joint` and `move_linear` raise `RobotMotionRejected` carrying the result, and `move_home`
returns `False`. The `Robot` verbs return a report instead. `ik` as the planner is the controller's
calibrated IK and a straight `moveJ` or `moveL`, which knows nothing about the cell and drives through
anything in it.

## How a planned move runs

On `curobo`, the arm chooses where every move goes and cuRobo never does: it is never handed a pose.
`move(pose)` takes every closed-form inverse kinematics solution of the flange goal, each joint on its
full turn nearest the arm inside the planner's and the joint-limit guard's window (the cable window
about home with `within_half_turn_of_home`), and ranks them: the branch the arm holds first (shoulder,
wrist 2 and elbow on the same side; a joint within 2 degrees of its branch point, as every one is in a
candle-straight home, is on both), then the least time a `moveJ` there takes. The planner and the
endpoint gate screen each. Then, on the arm's branch first:

1. the straight joint line to each goal, judged by the path gate and by the planner against its world,
   the camera's included, at `safety.planned_motion.line_clearance_mm` (10 mm): the first that passes
   runs as one leg, and cuRobo plans nothing;
2. only where no line passes, cuRobo plans to the goals in the same order, up to three, from the same
   seed every time and never through its retract. A plan is taken where it ends on the configuration
   asked for and no joint swings more than `safety.planned_motion.max_detour_deg` (45) past the span
   between its start and its goal; else the next goal is planned to.

A goal on another branch is tried only once every goal on the arm's own failed, and taking one is a
warning naming both branches. A goal that nothing reaches costs up to three failed joint plans, 7 to
9 s each measured on the UR10 descriptor.

**A change of branch is the last resort** (the owner's R2, 2026-10-08). Inside `keeping_its_branch(15.0)`
every Cartesian move keeps the branch the arm holds: the other branches are not screened, lined or
planned to, a plan swinging a joint more than the block's bound past its span is a detour, and
`nearest_configuration` answers among the same goals. A move nothing on the branch runs, while another
branch or a plan within `max_detour_deg` was left untried, is refused before anything is sent,
`JOINT_LIMIT_REJECTED` ("kept to the branch the arm holds"), and counted on the record the block yields;
the pick loop leaves such a grasp for after the grasps that keep the branch. A route a grasp judged ahead
on the arm's branch is judged again by the move to its standoff on that branch alone: a change of branch
nobody judged ahead is refused the same way. `has_a_goal_on_its_branch(pose, here)` answers on the closed
form alone, with no screen, planner or controller call, whether a pose has a configuration on the
branch `here` holds inside the window; the pick loop takes a grasp's half-turned twin where only the twin
has one (the window gap of 2026-10-07).

**The planner's cushion band.** cuRobo's padded spheres refuse some poses the exact meshes keep well clear,
such as the owner's LOOK[0], forearm|wrist_2 1.2 mm deep for the spheres and 19.0 mm apart for the meshes.
On the arm's own pairs the exact guard decides (the owner, 2026-09-30,
[`planning/band.py`](../../safety/planning/band.py)): where the planner refuses a sample of a line, a leg or
a moveL as a self collision, the arm asks it for every refused sample (`judge_joint_path`, which refreshes
nothing), leaves to the exact guard only the pairs that guard judges, padded by no more than
`planner_margin_mm`, and has the exact guard judge each of those samples again. The bounds, the carried
part and the `shoulder_link` stay the planner's, and a report it cannot read whole leaves the refusal
standing. Such a refusal names the sample and the term it stands on; a bound reads `JOINT_LIMIT_REJECTED`.

**The camera's boxes** (the owner's Option 1). The same spheres reach 25 to 29 mm past the shoulder housing
into the world. Where the planner refuses samples on its world, the arm asks it again about exactly those
samples with every box the camera saw set aside (`judge_joint_path(..., ignore_perceived=)`), and admits them
only where its world and bounds then clear and the robot itself reads as before, no part is carried (any
`attach_payload` on the arm since the last `detach_payload`, modelled or not, and
`CuroboUrPlanner.carries_part`), the hand it carries reads empty and open (`set_hand`, which `connect_cell`
calls once both are up: a toggle's count open, a gripper measured fully open; no hand, or any other, keeps
the boxes in), the glue's last confirmed refresh handed the planner exactly the boxes the exact guard holds
(`perceived_in_world`), and the exact guard accepts every refused sample with them at
`perceived_min_distance_mm`. The bench and the declared fixtures and meshes stay the planner's, and so does
cuRobo's own plan, which routes around the camera's boxes: a planned move into or out of a pose its spheres
meet them in is refused, and straight lines and moveL run there with a hand known empty and open.

**A grasp judges its lift before it closes** (the owner, 2026-10-01). `carried_line_refusal(pose,
grip_width_mm=, camera_world=)` answers what `move(pose, linear=True, camera_world=)` would answer with a part
in the jaws: the part handed to the planner for the judgement and taken back after, nothing of the camera's
world set aside, nothing sent; on `ik`, the end's gate, screened, so nothing is remembered.
`GraspExecutionPolicy` and `Robot.pick` ask it at the part with the jaws still open, at the width the attach
after the close carries (a hand that does not measure: the width it is told); where the lift would be
refused, the jaws stay open, the arm goes back up the line it came down, and the grasp ends
`carried_retreat_refused` (`PolicyOutcome`, `HandlingOutcome`; a pick loop attempt reads `execution_failed`).
Where it would run, the controller is asked last, and the jaws close. The lift is judged again as it starts,
so a new camera frame, the part's spheres fitted anew by the attach, a part measured wider or a lift in steps
can still refuse it. An arm that holds a part
beside the camera's boxes, after that or after a plain close, is held there: every way out starts at that
pose, and the refusal names the way out, a person releasing the part (`Robot.release`).

A **planned** move out of a band pose, or into one, takes a straight **escape** or **approach** leg of at
most 20 degrees per joint, only the joints between the colliding links, to the nearest configuration both
authorities clear; both judge the leg as a straight line, and cuRobo plans between clear configurations
(`CuroboUrPlanner.plan_joint(goal, start_ur=...)`, whose explicit start is never read from the controller).
The route is judged whole again, and the shortening keeps each leg as the one `moveJ` it was judged as; the
route's log line says the leg. Once a leg left the start, a goal that still does not plan is that goal's
failure, and the next goal is planned from the leg's end; only a start no leg leaves is refused as the
start's. The generated view and its move back never take a leg. The glue logs a refused start as `cuRobo
refused the start of this joint move`, a refused goal as `this joint goal`.

`screen_configuration(joints)` asks both authorities about a pose and moves nothing: `clear`, `band`,
`seen_boxes`, `guard_refused`, `planner_refused` or `unscreened`, with the nearest pose within 20 degrees
per joint both clear where one exists (`planning.band.PoseScreen`). It asks what a move asks, the camera's
boxes set aside included, for a hand known empty and open (it reads no hand), so `seen_boxes` is a pose a line
and moveL reach with such a hand, and only the two refusals read `ERROR`. Teaching, the desk start and every campaign's start ask it.

A joint target (`move_to_joints`, `move_home`, a joint station) goes the same way: a joint outside the
window runs as its full turn inside it, the same pose, and the log says so; a joint inside is left as
written, and a value past a full turn, a target in degrees, is refused. Its straight line runs as one
`moveJ` where both authorities pass it at the same clearance, and where it collides cuRobo plans
around it to the same target under the same detour bound.

A plan then meets the plan-end check and the endpoint gate, is shortened to the fewest of its own
waypoints whose legs stay within the path gate's step, and runs as one `moveJ` per kept waypoint once
the path gate and the planner at `line_clearance_mm` have both passed those legs, which are straight
lines nobody planned. Where either refuses, the plan as cuRobo returned it is judged by both at no
contact, as cuRobo validated it, and runs instead. A transport that fails while cuRobo plans, which
reads the joints again, is `CONNECTION_ERROR` and nothing is sent. Every move logs how its route was chosen (a direct
line, a cuRobo plan to which goal, a branch change, a turned joint), how many waypoints cuRobo returned
and how many ran, and each joint's total and largest turn in degrees on what ran, and warns when a
joint turns more than half a turn past what its end needs.

## Judged while the arm waits

Three switches move a judgement to where the arm waits anyway (the owner, 2026-10-09, "solange wir keine Qualität
verlieren"). Each is off as shipped, and none changes what is judged: the exact guard judges every sample of every
leg before it is sent.

- **The steady gate at the send** (`safety.dwell.gate_at: send`). The motion is judged first, while the arm settles
  from the one before (the controller's `is_steady()` took 0.54 to 0.61 s after every motion on the cell), and the
  gate waits right before the `moveJ` or `moveL`, in `_drive_curobo`, `_drive_judged_joints`, `_drive_joint_path` and
  `_drive_checked_line`. Where the arm then stands more than 0.5 mm from where the motion was judged from, it is
  judged again from there (three times at most, then `TIMEOUT`); a timeout sends nothing, and a halt is refused by
  the send itself. A route's first waypoint, where the gate found the arm, is not sent as a `moveJ` of its own.
  `gates_its_own_sends` tells the verbs (`GraspExecutionPolicy`, `Robot.move`, `Robot.pick`) to ask with no gate
  of their own.
- **The world held at the standoff** (`safety.planning_world.hold.standoff`, the map's O2). `grasp_refusal_ahead`
  judges the route in the world the line down was judged in, and `holding_the_approach(standoff, reason)` holds that
  world for the move to the standoff and the line down from it, where the route judged ahead stands ready
  (`_why_judged_again`): no frame is taken at the standoff, where a bin frame often held no depth at all. The line
  down is still judged sample by sample, in that world.
- **The next leg judged ahead** (`robot.motion.judge_next_leg`, with `hold.carry` and `hold.drop`). A task declares
  the joint move after a verb (`expecting_next(joints)`: the carry to the bin's look, the return). While the jaws'
  stroke is waited out, a verb asks `judge_the_next_leg(after, junction=)`: the straight joint line from where the
  line to `after` will end (the controller's solution of `after`, seeded where the arm stands) to the declared joints,
  judged in the world held there; `judge_line_ahead(pose)` judges the line out of a place where the arm stands, as it
  will run. `move_to_joints` and `move_to_home` run the move as judged, and `_drive_checked_line` the line, only where
  nothing it was judged on changed: the same planner and refresh, the arm within 0.5 mm of where it was judged from,
  the carried part, the hand, the camera's boxes and the halt count as they were, at most 10 s since (0.5 s for a
  route judged ahead), one move. Anything else, and it is judged as it runs. With `in_settles_and_motion` on an arm
  whose halt brakes the line in flight (`robot.ur.brake_on_halt`), a move no stroke left time for is judged on a
  second thread while the line before it runs; that thread asks the controller nothing (its start and what it reads
  are read before the line is sent), and the line's send waits for it once it ended. Without the brake it falls back
  to the strokes alone and says so once in the log; on a cell that models a carried part while none is carried,
  whose judgement reads the hand off the controller, the move is judged as it runs.

`scripts/ursim/probe_next_leg.py` measured the judgement during motion against URSim CB3 (2026-10-09, both gate modes,
every check passed): the joint move judged on the second thread before the line's `moveL` returned, the arm 0.0005 mm
from where it was judged from at the junction, the move run as judged; a halt 0.4 s into the line braked it with
`stopL` while the thread still judged, and the move was then judged again (the arm 574 to 581 mm from there). The
judgement gives up and takes the GIL around every exact-guard query, and at Python's 5 ms switch interval the thread
that watches the line waited for it up to 451 ms on Windows and 521 ms in WSL. While a judgement runs during a line the
interval is 0.5 ms (`_WATCHED_SWITCH_S`, put back after it): the stop then went out 0.2 to 15 ms after the halt
against 0.1 to 7 ms with no thread, and the arm ran at most 3.2 mm further past its brake, five halts each.

## Solved here, not asked

Three switches answer here what the controller was asked, round trip by round trip (the owner, 2026-10-09). Each is off
as shipped, and each falls back to asking the controller wherever its answer here cannot be vouched for.

- **The line's samples** (`safety.ik_quality.line_ik: local`). A line judged before a `moveL` had every sample solved by
  `getInverseKinematics`, 32.5 ms each: 28 for the line down, 35 for the lift. The controller's own DH rows, the table
  plus the arm's factory calibration, are read once per connection from the kinematics info of its primary interface
  (`URConnection.controller_kinematics`: port 30011, the read-only one, then 30001; nothing is ever sent there), and
  its active TCP from `getTCPOffset` (`active_tcp_offset`). `_judge_line_samples` solves every sample and knot on them
  (`_ur_ik.ur_chain_ik_nearest`: the closed form nearest the seed, then Newton on the rows) and asks the controller at
  the first and the last sample solved; both have to agree within 1e-6 rad. Anything else, and the line is walked again
  from its start with the controller solving it, as before: an answer that parts, a sample with another configuration
  about as near the seed, near a singularity or out of the table's reach, rows or a TCP that cannot be read.
- **The singularity check before a move** (`safety.ik_quality.singularity_fk: dh`). `MotionController` differentiated
  the controller's FK, 12 round trips before every `moveL`; with `dh` it differentiates the same rows here, times the
  active TCP, with the same steps and thresholds. Where the rows cannot be read the nominal table stands in only if two
  FK probes, once per connection, show it is the controller's chain; else the controller is asked, as always.
- **The steady gate** (`safety.dwell.steady_signal: joint_speeds`). `wait_until_steady` reads the actual joint speeds
  every 8 ms, locally, and the arm is steady once every joint is under 0.01 rad/s on three reads in a row, each a sample
  the controller sent since the one before (its timestamp moved on, so a receive stream that stopped never reads as an
  arm that stands); `isSteady`, a script command of 33 ms polled every 20 ms, decides only where the speeds cannot be
  read. The timeout and its refusal are the same. The zero-move skip at a look asks the same gate (a wait of 0: three
  reads, 16 ms). On URSim CB3 the gate answered 16 to 24 ms after a `moveJ` returned, where `isSteady` took 562 to
  564 ms, and during an asynchronous `moveJ` it said steady first 60 ms after the move's own end, `isSteady` 0.57 s
  after it: `isSteady` holds about half a second after the arm stopped, and the joint speeds do not.

Measured on URSim CB3 3.15.8 (a UR10 with the owner's TCP set on it), against the controller and against the same
controller given a `calibration.conf` whose rows put the flange 3.2 mm from the table's: the controller's FK and the rows
read here agree to 6e-15 m; through `_judge_linear_move`, 168 lines (judged ahead and from where the arm stood) handed
the gate the controller's configurations to 1.7e-10 rad with 368 round trips instead of 2070, and with rows 1 mm off
all 46 lines fell back and judged as before; the singularity check gave the controller's verdict on all 177 line ends
of each controller, the cell's 21 of 2026-10-07/08 and 156 around where the verdict flips toward the wrist, elbow and
shoulder singularities, down to 1e-7 rad of it, with no round trip (0.4 s to 1.4 ms a check).

## Halt now

`URRobotArm` carries the halt latch (`SupportsHalt`): `halt(reason)` latches the arm and sends nothing from the
calling thread, `halt_state()` reads it, and `clear_halt()` ends it, which the operator console calls only once
a person said the cell is clear. While it is set, every motion verb answers `CANCELLED` with nothing sent,
`set_digital_output` raises `ArmHalted` (the toggle's DO0 included), `freedrive()` refuses, and
`RobotStatus.is_operational` is false while `controller_operational` keeps the controller's own answer: a halt
is never a stopped controller. The latch lives on the `URConnection`, so it outlives a disconnect and a connect,
and `stop()` is a halt too: the stop it used to send from the calling thread was never read by a synchronous move
in flight.

**`robot.ur.brake_on_halt`** decides what happens to a move in flight.

- **Off, as shipped**: every `moveJ` and `moveL` is the synchronous `ur_rtde` call it always was, byte for byte.
  A halt lets the move in flight run to its end, and nothing after it is sent.
- **On**: every move is sent asynchronously and watched by the thread that sent it, every 8 ms (one CB3 cycle),
  and answers true only once the arm stands at its target: joints within 2e-3 rad, a line's TCP within 1 mm and
  2e-3 rad. A halt makes that thread brake it, `stopJ` or `stopL` at max(2.0, the move's own acceleration), and
  the move answers false. A move the controller never shows running within 1 s is stopped and refused.
  `last_move_end` (`MoveEnd`) says how the last move ended: `arrived`, `braked`, `brake_unconfirmed`,
  `refused_halted`, `ended_short`, `stopped` (a protective or emergency stop, or the program ended) or
  `not_started`.

`HaltState.brake` says what became of the move in flight: `none`, `pending`, `braked` (with `brake_s`),
`ran_out`, or `unconfirmed`, where the arm was never seen to stand still. A judged cuRobo path counts the halt
requests (`halt_requests`), so a halt that came and was cleared mid-leg still ends the waypoints after it. The one
output a halted arm still writes is `end_output_pulse(pin)`, which only ever drives low: a double solenoid's coil
and a vacuum blow-off end their pulse through it, never a toggle hand.

Two reads serve the console's ready bar without the dashboard: `quick_robot_status()`, the four receive-stream
fields and the latch, and `planner_state` (`not_used`, `off`, `starting`, `ready`), attribute reads that never
wait for the planner's lock. A disconnect retires the planner: it waits for a start in progress (at most 120 s) or
a planner call a move is inside (at most 30 s), then closes the sidecar, so no sidecar is orphaned. Measured
against URSim CB3 with `scripts/ursim/probe_halt.py`
([console_at_the_cell.md](../../../../docs/runbooks/console_at_the_cell.md)); switch the brake on in the cell's
own profile only after those measurements and a supervised halt at the cell.

## Traps

- **Forward kinematics after a motion.** `getForwardKinematics(q)` shares controller registers with
  `moveJ` and `moveL`, so after a move it can return a pose hundreds of millimetres off.
  `URConnection.fk_current()` is immune and is what to use for the joints the arm stands at. `fk(q)`
  for an arbitrary `q` has no immune form yet.
- **Remote control.** A UR in Local mode never runs an external program. `connect()` diagnoses Local
  mode or an uncleared protective stop through the dashboard, and claims a cause only where the
  dashboard proves it. Remote mode locks the pendant for motion, and a second `RTDEControlInterface`
  stops the first. What a physical emergency stop does in Remote mode is not verified here; read the
  robot manual before the first powered run.
- **`robot.ur.model` is a safety key.** It selects the DH chain, the collision meshes and the cuRobo
  descriptor. The built-in joint limits are factory-wide; site limits go in `robot.safety.joint_limits`.
- **`rtde_frequency: 0.0`** means the controller chooses its rate. The driver translates it for
  `ur_rtde`, which would read 0.0 as zero hertz.
- **Force and torque are read-only.** No force control, no compliant or blended motion, no tool changer.
- **An inverse kinematics the controller cannot solve ends its control program.** On URSim CB3 3.15.8 with ur_rtde
  1.6.5, `getInverseKinematics` of a pose out of reach answered `[]` and the control script stopped ("RTDE control
  script is not running!"): every later call failed until a reconnect. A line with a sample out of reach meets it,
  `line_ik: local` or not, because such a line is solved by the controller.

## The I/O bench

`python -m src.robot.drivers.ur` measures the digital I/O a gripper is wired to. It opens the
connection, so the refusals above apply, and it has no path to a motion. Driving a pin is still
physical, so every write needs `--yes`, and a non-interactive shell refuses rather than prompts.

```bash
python -m src.robot.drivers.ur --watch 0 --for 15              # trip a sensor by hand and watch the pin
python -m src.robot.drivers.ur --set 4=1 --yes                 # drive one output, then read it back
python -m src.robot.drivers.ur --measure 4=1 --watch 0 --yes   # the time close_settle_s wants
python -m src.robot.drivers.ur --where                         # JointPositions.deg(...) to paste, and the TCP
```

One action per call. `--port` picks the bank (`tool` by default) and `--profile` the cell; every
refusal prints the profile chain it loaded, and says so where neither `--profile` nor `WILLY_PROFILE`
was given and the base tree, which names a Robotiq, was read. `--where` is read-only. Run
`--measure` a few times and configure the worst reading. It exits 0 when the action ran, 1 when the
config or the connection refused, 2 when the input never answered, and 3 on an unexpected error.
`Bench.from_robot_config` is the same from Python and reads the bank from `gripper.jaw_io.io_port`
or `gripper.vacuum.io_port`; the `io_bench.py` functions under it confirm nothing.

## Status

| Capability | Evidence |
|---|---|
| Power-on and brake release through the dashboard, `moveJ`, the RTDE reads of pose, joints, modes and wrench | measured against real controller software (URSim) |
| Digital outputs that read back changed; the tool-frame check, including a disagreement that refuses | measured against real controller software (URSim) |
| `setPayload` with its rollback; a protective stop, with the commanded motions refused | measured against real controller software (URSim) |
| Connect, moves on the straight joint line and cuRobo plans, home, and a tool output switching a Hand-E, on a UR10 (CB3) | run on a physical cell |
| The halt latch and the brake: latency, stop point, 200 watched moves and 50 watched lines with no early return, a braked path, no DO0 change after a halt, the latch across a reconnect | measured against real controller software (URSim CB3, `probe_halt.py`) |
| The controller's rows read from its primary interface, lines solved on them and checked at two samples, the singularity check on them, nominal and calibrated | measured against real controller software (URSim CB3, see "Solved here, not asked") |
| Torque, payload dynamics, a physical emergency stop, a halt on a physical arm | never touched hardware |

The container and the probes behind those measurements are in
[scripts/ursim/](../../../../scripts/ursim/README.md). Their profile is
[config/robot/robot.ursim.yaml](../../../../config/robot/robot.ursim.yaml) (`WILLY_PROFILE=ursim`, or
`ursim,ursim_ur3` for a UR3e container).

## Bringing up a real UR

Nothing in this list moves the arm.

1. `python -m src.robot.drivers.doctor --require ur`: the SDK imports, else install `requirements.txt`.
2. `python -m src.config explain robot.ur.model`: the config names the arm in front of you.
3. Measure the payload and the tool frame; `connect()` refuses without either.
4. Put the pendant in Remote control.
5. `python scripts/checks/cell_bringup.py --live` under your profile: it connects, reads the pose and
   the robot and safety modes, and disconnects. Without `--live` it connects nothing and exits 2.

A green run means the declaration is coherent and the controller accepts it, not that it matches what
is bolted to the robot. The full procedure is [docs/runbooks/cell_bringup.md](../../../../docs/runbooks/cell_bringup.md),
another UR model is [ur_family_bringup.md](../../../../docs/runbooks/ur_family_bringup.md), and the
first pick is [real_cell_first_pick.md](../../../../docs/runbooks/real_cell_first_pick.md).

## Files

| File | Holds |
|---|---|
| `arm.py` | `URRobotArm`, `UR_CAPABILITIES`, `ur_capabilities(model)`: `move()`, the connect refusals, the planner switch, the halt, `quick_robot_status()` |
| `connection.py` | `URConnection`, the RTDE boundary: the halt latch, and with `brake_on_halt` the watched move a halt brakes (`MoveEnd`); the controller's own rows (`controller_kinematics`) and TCP; the steady gate on `isSteady` or the joint speeds |
| `freedrive.py` | `URFreedriveSession`: hand guiding on teach mode behind an RTDE watchdog, held again on every way out |
| `motion.py` | `MotionController`: clamped, workspace-checked point-to-point moves for `move_to`, and the singularity check before each, on the controller's FK or the arm's own rows (`singularity_fk`) |
| `curobo_motion.py` | `CuroboUrPlanner`: a collision-free plan to a pose or a joint goal, from where the arm stands or an explicit `start_ur`, run as one `moveJ` per waypoint of the list the arm judged; `judge_joint_path`, every refused sample of a path with its terms and pairs |
| `planner_frame.py` | `PlannerFrameClient`: the planner's base is the controller's turned half a turn about Z |
| `tool_frame.py` | the derived tool frame and its comparison with the declared one |
| `pose.py`, `pose_adapter.py` | `URPose` in millimetres and axis-angle, and its bridge to `Pose` |
| `io_bench.py`, `bench.py`, `__main__.py` | the I/O bench: primitives, the `Bench` noun and the command line |

## Details

- [drivers](../README.md): the registry, the doctor and the other vendors
- [safety](../../safety/README.md): the preflight every `move()` passes and the cuRobo planning;
  [docs/safety-math.md](../../../../docs/safety-math.md) for the formulas
- [grippers](../../grippers/README.md): the Robotiq, jaw and suction drivers that ride on this controller
