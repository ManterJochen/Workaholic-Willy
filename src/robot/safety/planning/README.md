# The motion planner and the collision engines (`src/robot/safety/planning`)

This package holds the two external engines the motion stack builds on, cuRobo for collision-free
trajectories and Coal (or python-fcl) for exact mesh distance, and one reading that says which of them
is wired on this machine for this robot. A `Robot` or a `Cell` reaches the planner on every move; call
`MotionStack` yourself to check a machine before the cell comes up.

```python
from willy import MotionStack, load_tree

report = MotionStack.from_robot_config(load_tree().robot).probe()   # the cell WILLY_PROFILE names
print(report)                       # both engines, the arm, and the config key that named it
raise SystemExit(report.exit_code)
```

```bash
python -m src.robot.safety.planning --check                 # reads paths; spawns nothing
python -m src.robot.safety.planning --doctor                # loads every engine; takes seconds
python -m src.robot.safety.planning --check --model ur3e --hand robotiq_hande --json
```

Exit `0` means fully anchored: the planner's interpreter is present, and an exact mesh engine and the
arm's mesh bundle import. Exit `1` means a degraded fallback would run (blind IK instead of the planner,
the capsule proxy instead of meshes), or no hand is named, or the config did not load. Exit `2` comes
only from `--doctor`: an operating-system policy blocked a binary that is present, which needs the
opposite fix to a missing one ([code-integrity.md](../../../../docs/code-integrity.md)). A development
machine without the GPU environment answers `1` by design. The same reading runs in
[planner_or_ik.py](../../../../examples/offline/config/planner_or_ik.py).

## Install

`scripts/ext_deps/install.ps1` installs both engines under `ext_deps/`, where the defaults look, so a
standard machine sets no environment variable ([ext_deps/README.md](../../../../ext_deps/README.md)).
The planner runs in its own interpreter, a sidecar, because it and the simulator need incompatible
versions of the same GPU runtime. `CuroboPlanClient` spawns it once, keeps it warm, and talks
newline-delimited JSON with it. This stack is millimetres and XYZW; the planner is metres and WXYZ, and
the conversion happens at that boundary.

## The nouns

| Noun | Built by | Verb | Returns |
| --- | --- | --- | --- |
| `MotionStack` | `from_robot_config(robot)`, `for_this_box()`, `from_model(model=...)` | `probe()` | `MotionStackReport` with `exit_code` |
| `CuroboPlanClient` | the driver, sized by the cell's reservation | `plan(goal)`, `check_joints(configs)` | joint waypoints; a verdict per sample |
| `LivePlannerWorld` | the driver, from the cell's cameras and declared world | `world_for(...)` | a `PlannerWorldSnapshot` with its verdict |
| `PlannerHand` | `planner_hand(robot, data_dir=...)` | read | the hand the planner and the guard model |

`MotionStack.from_model` is the one builder; the other two resolve a model and call it. The reading
names the arm and the key that decided it (`model_source`), because a green reading with no arm named
is a green light for a robot nobody configured.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| exit `1`, no hand | the tree names no `robot.gripper.model` | name the hand, or pass `--hand` |
| a move fails `CONTROLLER_REJECTED` | the planner is unavailable on a UR with `robot.ur.motion_planner: curobo`, the default | run `--doctor`; the arm never moved |
| a move fails `TIMEOUT` | the planner found no collision-free plan | move the goal, or clear the cell |
| `CuroboUnavailableError` from `check_joints` | no reply, a sidecar that exited, or a reply that is not a verdict for every sample | restart the sidecar from this tree |
| `ValueError` from `check_joints` | no configuration, a value that is not finite, or a wrong joint count | send whole configurations |
| `CameraWorldUnavailable` from a verb | a camera stays silent, blind or stale after `perceived.fresh_frame_attempts` more readings | fix the camera; a pick stops |
| a planner that does not start | no committed evidence file for this arm, hand, coupling, placement and margin | measure one, see [`robot/evidence/`](robot/evidence/README.md) |

The simulator refuses the same way as a real cell: there is no fallback to blind IK. `WILLY_CUROBO_ROBOT`
loses to the arm and hand the cell declares, and the driver refuses a descriptor built for another arm,
one that carries a hand, or one that records nothing about itself.

## What the planner is told about the cell

`robot.safety.planning_world` turns on a world rebuilt immediately before every plan, for any motion.
A world nobody can vouch for refuses the motion instead of being planned against. Three kinds of
geometry reach the planner:

| Channel | What it is for | Cost, measured on the workstation |
| --- | --- | --- |
| Boxes | what an operator declares; the only kind a refusal can name. Bounded by the planner's slots | 21.6 ms to register 41 |
| Meshes | a container that keeps its hollow, under `planning_world.meshes`; the sidecar reads the file | 6.4 ms to register one |
| A distance field | the whole scene at the resolution it is cut to, with nothing dropped for want of a slot | 13.7 ms to build 179,560 cells, 1.9 ms to register |

The field is a signed distance in metres, negative inside an obstacle. The guards read boxes only, so a
declared mesh is known to the planner alone, and the path guard gets the perceived boxes. That points
the safe way: a box that encloses a turned box is bigger, so a guard is never more permissive than the
planner. The path guard hears the same refresh, so the guard and the planner judge the same cell, and a
move planned on a vouched world is stamped `PLANNED` with its cameras and the capture time of its oldest
image. A pick's goal keeps the space between the open jaws out of every view (`goal_keep_out`), so a
world that registered the part still has a plan to it; `WorldRefresh.keep_out` records what was left
out, or why nothing was.

`check_js` judges a whole joint path in one request: up to 1000 configurations, each against the joint
limits, the robot itself and the world the planner holds, with an attached payload counted. A sample
passes when no sphere penetrates; that is not a kept clearance. A longer path is split across requests
and never thinned.

```bash
python scripts/curobo/probe_live_world.py
```

That probe runs the real sidecar with a wall no config file declares: the empty cell plans, the wall
stops the plan, removing it plans again. The middle result alone would prove nothing, because a planner
that refuses everything looks the same.

## Environment variables

| Variable | Default | Read by | Meaning |
| --- | --- | --- | --- |
| `WILLY_CUROBO_PYTHON` | `ext_deps/curobo_env/python.exe` | client | the planner environment's interpreter |
| `WILLY_CUROBO_ROBOT` | `ur5e.yml` | client | a descriptor for a client built without one; a cell names `willy_{arm}.yml` |
| `WILLY_CUROBO_CUBOID_CACHE` | `16` | client | box slots |
| `WILLY_CUROBO_MESH_CACHE` | `0` | client | mesh slots |
| `WILLY_CUROBO_VOXEL_GRID` | unset, no grid | client | the live-scene grid, `x,y,z,voxel` in metres |
| `WILLY_CUROBO_STDERR` | unset, discarded | client | a file for the sidecar's stderr |
| `WILLY_CUROBO_MAX_ATTEMPTS` | `16` | sidecar | plan attempts, each a fresh seed batch |
| `WILLY_CUROBO_GRAPH_FROM_ATTEMPT` | `1` | sidecar | the first graph-seeded attempt |
| `WILLY_CUROBO_ATTACH_SPHERES` | unset, as `0` | sidecar | spheres for a carried payload; the client sets it from the cell's reservation |
| `WILLY_CUROBO_SELF_COLLISION_MARGIN_MM` | unset, as `0` | sidecar | the guard's clearance, raised into the descriptor's link buffers |
| `WILLY_COAL_PREFIX` | `ext_deps/coal_env` | collision engine | the environment that provides Coal |

The names live in `environment.py`, except the attach and margin variables, which `_curobo_attach.py`
and `_curobo_margin.py` name. A cell's reservation (`reservation.py`) wins over the three slot
variables, and the client warns when one disagrees with it.

## Status

| Capability | Evidence |
| --- | --- |
| Planned moves on a UR arm | measured in simulation (Isaac) and measured against real controller software (URSim) |
| The live world and the batch check | measured in simulation: the probe runs the real sidecar on a synthetic camera |
| Exact mesh collision through Coal or python-fcl | measured in simulation |
| A planned move on a physical arm | never touched hardware |

`--check` reports that the sidecar's interpreter exists; it does not report that the descriptor beside
it was built, because that lives in an environment this process does not spawn. `--doctor` closes that
gap. Planner collision awareness is not a certified functional-safety stop.

## Files

| File | Holds |
| --- | --- |
| [`environment.py`](environment.py) | every variable name, default path and availability probe; `import_collision_engine` |
| [`stack.py`](stack.py), [`__main__.py`](__main__.py) | `MotionStack` and the `--check` command |
| [`doctor.py`](doctor.py) | `--doctor`: loads every engine and classifies an OS policy block |
| [`curobo_client.py`](curobo_client.py), [`curobo_planner_server.py`](curobo_planner_server.py) | the client, and the sidecar it runs in the planner's interpreter |
| [`world.py`](world.py), [`perceived.py`](perceived.py), [`live_world.py`](live_world.py) | declared geometry, camera geometry, and the world rebuilt before every plan |
| [`depth_source.py`](depth_source.py) | a camera rig asked for depth alone |
| [`reservation.py`](reservation.py), [`margin.py`](margin.py) | the slots the sidecar allocates, and the clearance the planner keeps |
| [`hand.py`](hand.py), `_hand_placement.py`, `_hand_bundle.py` | the hand from `robot.gripper.model`, where it sits, and its bundle checked |
| [`body_link.py`](body_link.py) | a wrist camera as one body for the planner, the guard and the self filter |
| [`self_envelope.py`](self_envelope.py) | the robot's own body, filtered out of the camera view |
| [`evidence.py`](evidence.py), [`bundle_index.py`](bundle_index.py) | the measured combination files, and the index of committed bundles |
| `_declared_body.py` | a declared box as bundle arrays, and the proof that a sphere fill covers it |
| `_curobo_*.py` | the sidecar's descriptor, attachment, margin, pair and protocol helpers |
| [`robot/`](robot/PROVENANCE.md) | sphere maps per arm and hand, the retract table, the hand writers, the evidence |

## Details

- [`../README.md`](../README.md): the guards this package sits under, and what the capsule fallback costs
- [`../../drivers/ur/`](../../drivers/ur/README.md): the UR path that binds the planner fail-closed
- [`../../grasping/collision/`](../../grasping/collision/README.md): another user of the exact mesh engine
- [Robot and safety guide](../../../../docs/guide/04-robot-and-safety.md), section 6; [safety-math.md](../../../../docs/safety-math.md)
- New hand: `scripts/grippers/write_hand_from_mesh.py`, then [your_own_gripper.md](../../../../docs/runbooks/your_own_gripper.md)
- Descriptors: `scripts/curobo/build_ur_config.py` builds `willy_{arm}.yml` inside `ext_deps/`
- Tests: `tests/test_motion_stack.py`, `tests/test_planning_doctor.py`, `tests/test_curobo_batch_check.py`
