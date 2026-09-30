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
| `INVALID_TARGET`, a waypoint | `ur_rtde` refused a speed or acceleration before sending that waypoint's `moveJ` | The message says which waypoint and what ran before it |
| `UNSUPPORTED`, camera world MISSING | on `curobo`, a motion with neither a live camera world nor a stated decline | Hand the robot its cameras, or `without_camera_world(reason)` |
| `CONTROLLER_REJECTED`, hand-guided | any motion verb while a `freedrive()` session is open, free or held | Leave the session; its end holds the arm and gives motion back |
| `CONTROLLER_REJECTED`, plan off its goal | a plan ending more than 5 mm or 6 degrees from its goal on the DH chain | Read the message; nothing has moved |
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

**The planner's cushion band.** cuRobo's padded spheres refuse some poses the exact meshes keep well clear,
such as the owner's LOOK[0], forearm|wrist_2 1.2 mm deep for the spheres and 19.0 mm apart for the meshes.
On the arm's own pairs the exact guard decides (the owner, 2026-09-30,
[`planning/band.py`](../../safety/planning/band.py)): where the planner refuses a sample of a line, a leg or
a moveL as a self collision, the arm asks it for every refused sample (`judge_joint_path`, which refreshes
nothing), leaves to the exact guard only the pairs that guard judges, padded by no more than
`planner_margin_mm`, and has the exact guard judge each of those samples again. The world, the bounds, the
carried part and the `shoulder_link` stay the planner's, and a report it cannot read whole leaves the
refusal standing. Such a refusal names the sample and the term it stands on; a bound reads
`JOINT_LIMIT_REJECTED`.

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
`guard_refused`, `planner_refused` or `unscreened`, with the nearest pose within 20 degrees per joint
both clear where one exists (`planning.band.PoseScreen`). Teaching, the desk start and every campaign's
start ask it.

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
| Torque, payload dynamics, a physical emergency stop | never touched hardware |

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
| `arm.py` | `URRobotArm`, `UR_CAPABILITIES`, `ur_capabilities(model)`: `move()`, the connect refusals, the planner switch |
| `connection.py` | `URConnection`, the RTDE boundary |
| `freedrive.py` | `URFreedriveSession`: hand guiding on teach mode behind an RTDE watchdog, held again on every way out |
| `motion.py` | `MotionController`: clamped, workspace-checked point-to-point moves for `move_to` |
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
