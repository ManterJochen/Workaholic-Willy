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
| `world.py` | Converts the cell's static geometry into the shape the planner accepts. Pure: config in, wire dictionaries out. |
| `_curobo_attach.py` | Tells the planner the gripper is carrying something, by deriving the attachment link the shipped UR configs do not declare. |
| `_curobo_margin.py` | Tells the planner the clearance the self-collision guard will demand, by deriving it into a temporary config. |
| [`robot/`](robot/) | The committed reference geometry and its label: [`PROVENANCE.md`](robot/PROVENANCE.md), the sphere-map generator `build_gripper_spheres.py`, and `ur5e_gripper_spheres.yml`. |

## Why the planner runs in another process

The planner and the simulator need different, incompatible versions of the same GPU kernel runtime,
and one process holds exactly one of them. So `CuroboPlanClient` spawns the sidecar once, keeps it
warm, and issues many calls over newline-delimited JSON on its standard input and output.

The units and conventions flip at that boundary. This stack is millimetres and XYZW quaternions; the
planner is metres and WXYZ. Goal poses cross as `tool0` in the base frame, in metres, with a WXYZ
quaternion, and the trajectory comes back as joint waypoints in the planner's own joint order.

The planner's collision world is axis-aligned boxes and nothing else. It has no mesh, point-cloud or
voxel channel, so perceived geometry cannot reach it and anything that is not box-shaped has to be
enclosed in one.

## Usage

```console
$ python -m src.robot.safety.planning --check
Workaholic-Willy motion-stack external engines
  cuRobo planner: MISSING  (python=.../ext_deps/curobo_env/python.exe, robot=ur5e.yml not verified: ...)
  exact-mesh collision engine: fcl  (ur5e mesh bundle present)
  => partially anchored (degraded fallbacks active)
  (model: ur5e, from robot.ur.model)
```

Exit `0` means fully anchored: the planner environment is present, and an exact-mesh engine plus the
reference mesh bundle are importable. Exit `1` means partially anchored, so a degraded fallback is in
force: blind IK instead of the planner, the capsule proxy instead of exact meshes, or both. Exit `2`
comes only from `--doctor` and means an operating-system policy blocked a binary that is otherwise
present and intact, which needs the opposite response to a missing environment.

A development box without the GPU environment reports `1` by design. The check is meant to run on the
target cell, not to gate a build.

Both readings are per robot. The mesh bundle ships as `{model}_collision_meshes.npz` and the planner
descriptor as `{model}.yml`, so a present `ur5e` bundle says nothing about a UR3e cell. The entry
point reads the model from the config the cell will load; `--profile` selects that config layer and
`--model` overrides the model outright for a box with no config tree. `--json` prints the reading as
data.

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
| `WILLY_CUROBO_ROBOT` | `ur5e.yml` | client | planner robot descriptor |
| `WILLY_CUROBO_CUBOID_CACHE` | `16` | client | reserved collision-world cuboid slots |
| `WILLY_CUROBO_STDERR` | unset, meaning discarded | client | optional sidecar stderr log file |
| `WILLY_CUROBO_MAX_ATTEMPTS` | `16` | sidecar | plan attempts, each a fresh seed batch |
| `WILLY_CUROBO_GRAPH_FROM_ATTEMPT` | `1` | sidecar | the first graph-seeded attempt |
| `WILLY_CUROBO_ATTACH_SPHERES` | unset, and `0` means the same | sidecar | collision spheres reserved for a carried payload. Unset leaves the descriptor untouched and the sidecar unable to attach anything, which is the unchanged path. The client passes `16` when it does set it. |
| `WILLY_CUROBO_SELF_COLLISION_MARGIN_MM` | unset, and `0` means the same | sidecar | the pairwise clearance the self-collision guard will demand, raised into the descriptor's per-link buffers. Unset leaves the descriptor untouched. |
| `WILLY_COAL_PREFIX` | `ext_deps/coal_env` | collision engine | the environment prefix that provides Coal |

The sidecar-only variables are read inside the separate interpreter, which cannot import this
package. They are named here so the full set stays in one table.

`WILLY_CUROBO_ROBOT` loses to the robot model the cell config declares when the two disagree. The
environment variable is ignored loudly rather than obeyed, because planning one cell against another
robot's geometry has no visible symptom.

## The fail-closed contract

| Path | When the planner is unavailable |
| --- | --- |
| simulator and mock | may fall back to blind IK, leaving behaviour unchanged |
| real UR with `robot.ur.motion_planner: curobo`, which is the default | fails closed. No blind motion. `CuroboUnavailableError` becomes `CONTROLLER_REJECTED` or `TIMEOUT`, and the message says the arm never moved |

Coal and python-fcl resolve through one seam. Both the exact-mesh self-collision backend and the
continuous monitor call `environment.import_collision_engine()`, which prefers Coal and falls back to
python-fcl. The swap is behaviour-identical: same meshes, same pairs, same thresholds, same distance
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
regenerable. The complete descriptor the planner loads is assembled on the target box by
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
