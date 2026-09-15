# robot.safety.planning

The single named home for the two external engines the motion stack builds on, with one command that
says which of them is actually wired on this box, for this robot.

## What this package guarantees

Two engines cannot be plain lines in `requirements.txt`: they need a different Python, they clash on
a GPU runtime dependency, or they ship no wheel for the platform. Rather than scatter optional-import
guards through the code, this package makes that dependency explicit, named, centrally probed and
documented in one place.

| Engine | Role | Lives in | Verify with |
| --- | --- | --- | --- |
| cuRobo | GPU collision-aware trajectory planner | this package: `curobo_client.py` plus a separate-interpreter sidecar | `--check`, `--doctor` |
| Coal, with python-fcl as the fallback | exact mesh-against-mesh collision distance | [`../_fcl_self_collision.py`](../_fcl_self_collision.py) | `--check`, `--doctor` |

Install both with `scripts/ext_deps/install.ps1`. They land under `ext_deps/`, which is where the
defaults look, so a standard box needs no environment variables at all. Both are used by the real
pick paths, the simulator driver and the real UR execution path alike.

## Contents

| File or directory | Role |
| --- | --- |
| `environment.py` | The anchor. Every environment-variable name, default path and availability probe for both engines, plus the typed snapshots `CuroboStatus`, `CollisionEngineStatus` and `PlanningEnvironment`. |
| `stack.py` | `MotionStack` and `MotionStackReport`: which robot the reading is about, and where that robot name came from. |
| `__main__.py` | The `--check` and `--doctor` verify command. |
| `doctor.py` | The deep check. Where `--check` reads paths, this loads every engine: it imports the collision engine and runs a real distance query, spawns the sidecar's own interpreter, verifies the robot descriptor inside it, and reports how many kernel backends resolve. An operating-system application-control refusal is classified as its own outcome. |
| `curobo_client.py` | `CuroboPlanClient`, in-process and standard-library only. It spawns and drives the sidecar and imports no planner code, so it type-checks in the main environment. |
| `curobo_planner_server.py` | The sidecar itself, run by the planner's own interpreter and never imported here. |
| `world.py` | Converts the cell's declared geometry into the shape the planner accepts: boxes and meshes, metres and WXYZ, plus the merge rule that stops a caller deleting the bench. Pure: config in, wire dictionaries out. |
| `perceived.py` | What the cameras see, as geometry: several camera views fused into one cloud, the robot filtered out of it, and out of that either turned boxes or a distance field over a grid. Pure, so an adversarial frame is a unit test. |
| `live_world.py` | What an arm asks before every plan: it holds the cameras, the transforms and the declared world, answers with the cell as it is, and refuses when nobody can vouch for it. |
| `reservation.py` | What the planner sidecar allocates when it starts: box slots, mesh slots, the live scene grid and payload spheres, derived once from the robot config and handed to every client that starts one, so a declared tote or a live scene always has somewhere to go. |
| `hand.py` | The hand the planner and the guard model, derived from `robot.gripper.model` alone: its sphere map and origin, the bundle the exact-mesh guard loads, and the coupling from `robot.gripper.coupling_plates_mm`. An unset name is `UNSET`, and a cell whose guard reads hand geometry refuses to build on it. |
| `self_envelope.py` | The robot's own body as the self filter takes it out of the camera view: one capsule per arm link fitted to the committed bundle, the hand's sphere map grown to hold its mesh, and a carried part while it is attached, all placed with the guard's model and base yaw. |
| `_curobo_attach.py` | Tells the planner the gripper is carrying something, by deriving the attachment link the shipped UR configs do not declare. |
| `_curobo_margin.py` | Tells the planner the clearance the self-collision guard will demand, by deriving it into a temporary config. |
| [`robot/`](robot/) | The committed reference geometry and its label: [`PROVENANCE.md`](robot/PROVENANCE.md), the one sphere fit in `gripper_spheres.py`, its command line `build_gripper_spheres.py`, and one committed map per hand, named by its registry name: `robotiq_2f85_gripper_spheres.yml`, `robotiq_hande_gripper_spheres.yml` and `schunk_egu50_gripper_spheres.yml`. |

## Why the planner runs in another process

The planner and the simulator need different, incompatible versions of the same GPU kernel runtime,
and one process holds exactly one of them. So `CuroboPlanClient` spawns the sidecar once, keeps it
warm, and issues many calls over newline-delimited JSON on its standard input and output.

The units and conventions flip at that boundary. This stack is millimetres and XYZW quaternions; the
planner is metres and WXYZ. Goal poses cross as `tool0` in the base frame, in metres, with a WXYZ
quaternion, and the trajectory comes back as joint waypoints in the planner's own joint order.

`check_js` judges a whole joint path in one request: up to 1000 configurations in the planner's joint
order, each held against the joint limits, the robot itself and the world the planner holds. The
spheres come from the planner's own kinematics, so a payload attached to the planner counts, where a
checker built on its own would be blind to one. A sample passes when the three terms sum to exactly
0, which means no sphere penetrates, not that any clearance is kept. The reply names the first
refused sample, counted from 0.

## What the planner is told about the cell

Three kinds of geometry reach it, and the difference between them is not academic.

| Channel | What it is for | Cost, measured on this box |
| --- | --- | --- |
| Boxes | What an operator writes down, and the only kind a refusal can name. Bounded by the planner's collision slots, so something always has to be left out and the report has to say what. | 21.6 ms to register 41 |
| Meshes | How a container keeps its hollow. As a box a tote is solid and a cell can never reach into it. Declared under `planning_world.meshes`; the sidecar reads the file itself, because a tote is tens of thousands of triangles and this is a line-based protocol. | 6.4 ms to register one |
| A distance field | The whole scene at once, at the resolution it is cut to, with nothing dropped for want of a slot. Built from the same cloud as the boxes. | 13.7 ms to build 179,560 cells, 1.9 ms to register |

The field is a signed distance in metres, negative inside an obstacle. Written positive inside,
every voxel that is not an obstacle reads as inside one, so the whole grid blocks and a wall looks seen
when the cell is. That is also why the probe below carries a decoy: a wall stopping the plan proves
nothing on its own.

⛔ **The guards cannot read a mesh or a field.** They work on boxes, so geometry declared as a mesh is
known to the planner and to nothing else, and the perceived boxes are what the path guard is given.
That asymmetry is deliberate and it points the safe way: the box that encloses a turned box is
bigger, so a guard is never more permissive than the planner.

### The world is rebuilt before every plan

A world registered once is a photograph. Everything that arrived afterwards is invisible to the
planner, and a plan through an obstacle it never received looks exactly like a plan through empty
space at every layer above. So `planning_world.enabled` also turns on a refresh that runs immediately
before every plan, for any motion rather than only for a pick, and a world nobody can vouch for
refuses the motion instead of being planned against.

A camera that cannot vouch for the cell is not refused, it raises. After `perceived.fresh_frame_attempts`
more readings, a camera that stays silent, blind or stale raises `CameraWorldUnavailable` out of every
verb, and a pick stops rather than trying its next candidate. The path guard hears the same refresh
before it judges a joint move or a line, so the guard and the planner judge the same cell, and a
`move` on a world its refresh vouched for is stamped `PLANNED` with its cameras and the capture time of
its oldest image.

Measured end to end through the real sidecar, with a wall that exists in no config file:

```bash
python scripts/curobo/probe_live_world.py
```

An empty cell plans, the wall stops the plan, taking the wall away lets it plan again. The middle one
alone would prove nothing: a planner that refuses everything looks identical.

## Usage

```console
$ python -m src.robot.safety.planning --check --hand robotiq_2f85
Workaholic-Willy motion-stack external engines
  cuRobo planner: MISSING  (python=.../ext_deps/curobo_env/python.exe, robot=ur5e_robotiq_2f85.yml not verified: ...)
  exact-mesh collision engine: fcl  (ur5e mesh bundle present)
  => partially anchored (degraded fallbacks active)
  (model: ur5e, from robot.ur.model)
```

Exit `0` means fully anchored: the planner environment is present, and an exact-mesh engine plus the
reference mesh bundle are importable. Exit `1` means partially anchored, so a degraded fallback is in
force: blind IK instead of the planner, the capsule proxy instead of exact meshes, or both. It also
means no hand is named. Exit `2`
comes only from `--doctor` and means an operating-system policy blocked a binary that is otherwise
present and intact, which needs the opposite response to a missing environment.

A development box without the GPU environment reports `1` by design. The check is meant to run on the
target cell, not to gate a build.

Both readings are per robot. The mesh bundle ships as `{model}_collision_meshes.npz` and the planner
descriptor as `{model}_{hand}.yml`, so a present `ur5e` bundle says nothing about a UR3e cell. The entry
point reads the model from the config the cell will load; `--profile` selects that config layer and
`--model` overrides the model outright for a box with no config tree. `--json` prints the reading as
data.

The hand is `robot.gripper.model`, or `--hand`. A tree that names no hand names no descriptor, and the
reading says which key to set and exits `1`, because a reading about a hand nobody named would be a
reading about an implied 2F-85.

From code:

```python
from src.robot.safety.planning import MotionStack

report = MotionStack.for_this_box().probe()
print(report.render())
raise SystemExit(report.exit_code)
```

`MotionStack.from_model(model=...)` is the one builder; `from_robot_config(robot_config)` and
`for_this_box()` resolve a model and call it. The model provenance travels on the reading as
`model_source`, which names the config key that decided it, because a `fully_anchored` verdict with
no robot attached is a green light for a robot nobody configured.
`probe_planning_environment()` is the lighter call underneath: it spawns nothing and warms nothing.

## Environment variables

All of them are read through `environment.py`, so every knob is discoverable in one place.

| Variable | Default | Read by | Meaning |
| --- | --- | --- | --- |
| `WILLY_CUROBO_PYTHON` | `ext_deps/curobo_env/python.exe` | client | interpreter of the planner environment |
| `WILLY_CUROBO_ROBOT` | `ur5e.yml` | client | descriptor of a client built without one; every cell names `{arm}_{hand}.yml` itself |
| `WILLY_CUROBO_CUBOID_CACHE` | `16` | client | reserved collision-world cuboid slots |
| `WILLY_CUROBO_MESH_CACHE` | `0` | client | reserved collision-world mesh slots |
| `WILLY_CUROBO_VOXEL_GRID` | unset, meaning no grid | client | live-scene grid, `x,y,z,voxel` in metres |
| `WILLY_CUROBO_STDERR` | unset, meaning discarded | client | optional sidecar stderr log file |
| `WILLY_CUROBO_MAX_ATTEMPTS` | `16` | sidecar | plan attempts, each a fresh seed batch |
| `WILLY_CUROBO_GRAPH_FROM_ATTEMPT` | `1` | sidecar | the first graph-seeded attempt |
| `WILLY_CUROBO_ATTACH_SPHERES` | unset, and `0` means the same | sidecar | collision spheres reserved for a carried payload. Unset leaves the descriptor untouched and the sidecar unable to attach anything, which is the unchanged path. The client passes `16` when it does set it. |
| `WILLY_CUROBO_SELF_COLLISION_MARGIN_MM` | unset, and `0` means the same | sidecar | the pairwise clearance the self-collision guard will demand, raised into the descriptor's per-link buffers. Unset leaves the descriptor untouched. |
| `WILLY_COAL_PREFIX` | `ext_deps/coal_env` | collision engine | the environment prefix that provides Coal |

A cell's own reservation (`reservation.py`) wins over the three slot variables, and the client warns
when one of them disagrees with it.

The sidecar-only variables are read inside the separate interpreter, which cannot import this
package. They are named here so the full set stays in one table.

`WILLY_CUROBO_ROBOT` loses to the arm and hand the cell config declares when the two disagree. The
environment variable is ignored loudly rather than obeyed, because planning one cell against another
robot's geometry has no visible symptom. The descriptor the sidecar loaded then has to say so itself:
its `_provenance` (arm, hand, plate) arrives with the ready line, and the driver refuses one that names
another arm, hand or plate, or none.

## The fail-closed contract

| Path | When the planner is unavailable |
| --- | --- |
| simulator and mock | refuses, as the real cell does. There is no fall back to blind IK, because a planner that never started then hides behind a low pick rate |
| real UR with `robot.ur.motion_planner: curobo`, which is the default | fails closed. No blind motion. `CuroboUnavailableError` becomes `CONTROLLER_REJECTED` or `TIMEOUT`, and the message says the arm never moved |
| batch check, `CuroboPlanClient.check_joints` sending the `check_js` request | never a pass. No reply, a sidecar that exited, a failed call (`planner_error`) or a reply that is not a whole verdict for every sample sent raises `CuroboUnavailableError`. A sidecar older than `check_js` answers through its plan branch, and the error says to restart it from this tree. No configuration at all, a value that is not finite or a wrong joint count raise `ValueError` before anything is sent. A path longer than one request is split across requests and never thinned |

Coal and python-fcl resolve through one seam. Both the exact-mesh self-collision backend and the
continuous monitor call `environment.import_collision_engine()`, which prefers Coal and falls back to
python-fcl. The doctor resolves through the same seam's `resolve_collision_engine()`, which
additionally keeps what each engine said when it refused. Without that, an OS policy refusing
`coal.dll` was invisible behind the substitution: `--doctor` reported `[ok] fcl` and exited 0,
measured 2026-09-10. The swap is behaviour-identical: same meshes, same pairs, same thresholds, same distance
query. If neither engine nor the mesh bundle is importable, the guard falls back to the capsule
proxy and logs the reason. Selecting the exact-mesh backend never silently disables the guard, but it
does not refuse the motion either; see [`../README.md`](../README.md) for what that fallback costs.

Planner collision-awareness is simulation-grade and is not a certified functional-safety stop. A real
cell still needs the vendor safety-rated stop, an independent emergency-stop circuit, and compliance
with ISO 10218, ISO/TS 15066 and ISO 13849.

## The reference robot

[`robot/`](robot/) is the version-controlled geometry authority for the robot the planner plans for.
The `tool0` gripper spheres there are grid-fit from the vertex-exact mesh bundle in
[`../data/`](../data/) by a generator that needs no simulator, so they are repo-derived and
regenerable. The complete `{arm}_{hand}.yml` descriptors the planner loads are assembled on the target box by
`scripts/curobo/build_ur_config.py` and written into the ignored content directory under `ext_deps/`.

What `available` does not mean: the planner half of the reading reports that the sidecar interpreter
exists on disk. It does not report that the descriptor named beside it was built, because that
descriptor lives inside a separate environment this process deliberately does not spawn. The caveat
travels in the report where a consumer can read it, and `--doctor` is what closes the gap.

The client and the real UR execution logic are exercised with fakes. The real round trip and the real
planning quality can only be measured on a box that has the engines.

## See also

- [`../README.md`](../README.md) for the fail-closed motion gate this package sits under
- [`../../drivers/ur/`](../../drivers/ur/README.md) for the execution path that binds the planner fail-closed
- [`../../grasping/collision/`](../../grasping/collision/README.md) for a downstream consumer of the exact-mesh engine
- [`ext_deps/README.md`](../../../../ext_deps/README.md), sections on Coal and on cuRobo, for the install
- `docs/code-integrity.md` for the blocked-binary outcome, and `docs/safety-math.md` for the geometry
