# `ext_deps/`: the local install root for external dependencies

The one place where the heavy, machine-local dependencies of Willy get installed, and the one page
that says how to install them, how to verify them, and what to do when they refuse.

> Isaac Sim is the deliberate exception. It is a multi-gigabyte standalone installer with its own
> bundled Python, and it stays where NVIDIA puts it. Everything else Willy bridges to lives here,
> and the Isaac section below says what that bridge is.

---

## The three external systems

Everything else Willy needs installs from the requirements files of the repository.

| System | What it is | Installs at | Bridged by | Used by |
|---|---|---|---|---|
| **Isaac Sim 5.1** | the validation simulator, a UR5e with a 2F-85 gripper | wherever its standalone installer put it | its own bundled `python.bat` | every `src/willy_sim/run_*.py` |
| **cuRobo** | collision-aware motion planner | `ext_deps/curobo_env` (Python 3.10) plus the source at `ext_deps/curobo` | `WILLY_CUROBO_PYTHON`, over a stdio sidecar | `safety/planning/curobo_client.py` talking to `safety/planning/curobo_planner_server.py` |
| **Coal** | exact mesh self-collision | `ext_deps/coal_env` | `WILLY_COAL_PREFIX` | `src/robot/safety/_fcl_self_collision.py` |

### Isaac Sim

Isaac drives the cell: the arm, the gripper, the cameras, the physics. Its bundled `python.bat` is
a complete interpreter with Python 3.11, `isaacsim`, a CUDA torch and warp 1.8.2. Every on-box run
uses it rather than the repository venv:

```
<isaac-sim-standalone>/python.bat -m src.willy_sim.run_m1_pick
```

## Layout, and what satisfies which default

| Subfolder | What it is | Wired via (default when unset) |
|---|---|---|
| `micromamba/` | the private package manager this folder bootstraps for itself | not configurable |
| `locks/` | the committed lockfiles: every conda package pinned by URL and hash | consumed by the install script |
| `curobo_env/` | micromamba env, Python 3.10 plus the cuRobo deps (torch cu128, cuda-toolkit 12.8) | `WILLY_CUROBO_PYTHON`, defaulting to `ext_deps/curobo_env/python.exe` |
| `curobo/` | the NVlabs/curobo source clone, at a pinned revision, with both kernel backends | found through `curobo_env` (`pip -e`) |
| `coal_env/` | micromamba env carrying Coal, the exact-mesh collision engine | `WILLY_COAL_PREFIX`, defaulting to `ext_deps/coal_env` |

Install elsewhere if you prefer: set the matching `WILLY_*` variable and the default is ignored.

---

## Install: one command for both

```powershell
powershell -ExecutionPolicy Bypass -File scripts\ext_deps\install.ps1
```

From nothing to a verified stack.

`-Clean` deletes the targets first, which is how to test that it really does rebuild from nothing.
`-Component coal|curobo` does one of them.

```bash
python -m src.robot.safety.planning --doctor   # 0 healthy | 1 degraded | 2 blocked by policy
```

Nothing outside `ext_deps/` is touched: no system conda, no PATH or registry changes, no admin
rights. Deleting this folder undoes all of it.

---

## Coal: the exact-mesh self-collision engine

The mesh-against-mesh self-collision backend (`self_collision.backend='fcl'`) and the continuous
collision-avoidance guard both run on **Coal**.
`python-fcl` is the guarded fallback.

> [!WARNING]
> **Honesty, and it is safety-critical.** This is a software collision-avoidance layer, not "the
> safety system". It reduces collision risk in simulation. The real-hardware
> safety guarantee is independent certified functional safety: hardware E-stop circuits and
> ISO 10218, ISO/TS 15066 and ISO 13849, running independently of this software.


## cuRobo: the collision-aware motion planner

The `motion_planner="curobo"` path plans global, collision-free trajectories with NVIDIA
[cuRobo](https://curobo.org).

### 0. Prerequisites

| | why |
|---|---|
| a conda-family package manager (`micromamba`, `mamba` or `conda`) | creates the isolated env |
| an NVIDIA GPU and a driver for your CUDA version | cuRobo is CUDA-only |
| **MSVC build tools** (Windows) or `g++` (Linux) | needed for the compiled backend in step 2b |

On Windows, `winget install --id Microsoft.VisualStudio.2022.BuildTools` (workload: *Desktop
development with C++*) provides MSVC. Step 2b must run from a shell where `cl.exe` is on `PATH`:
a *Developer Command Prompt*, or after sourcing `vcvars64.bat`.

Everything below is run from the repository root, and every path is relative to it.

### 1. The environment

```bash
# Adjust MM to your package manager; MAMBA_ROOT_PREFIX only matters for micromamba.
MM=micromamba

"$MM" create -y -p ext_deps/curobo_env -c conda-forge -c nvidia \
    python=3.10 cuda-toolkit=12.8 git-lfs

# torch matching the CUDA of your GPU (cu128 is the Blackwell / sm_120 line; use the wheel index
# for your card)
ext_deps/curobo_env/python.exe -m pip install "torch==2.7.0" \
    --index-url https://download.pytorch.org/whl/cu128
```

Python 3.10 is the supported version of cuRobo, and `cuda-toolkit` supplies the `nvcc` the compiled
backend needs.

### 2. cuRobo, with both kernel backends

cuRobo has two ways to run its CUDA kernels, and this repository installs both on purpose. The box
below says why that is not optional.

```bash
# No `git lfs install`: the pinned revision ships an empty .gitattributes, so it declares no LFS
# filters, and the call would write filter.lfs.* into your global .gitconfig for nothing.
git clone https://github.com/NVlabs/curobo.git ext_deps/curobo
cd ext_deps/curobo
git checkout 8e734f3          # see "Which revision" below
cd ../..
```

**2a. The JIT backend (`cuda.core`).**

```bash
ext_deps/curobo_env/python.exe -m pip install -e "ext_deps/curobo[cu12]" --no-build-isolation
```

The `[cu12]` extra installs the `cuda.core` runtime (NVRTC) that compiles kernels at run time. It
compiles nothing at install time and needs no MSVC.

**2b. The compiled backend (`pybind`), which is the fallback.** Run this from a shell where
`cl.exe` is available:

```bash
CUROBO_USE_PYBIND=1 ext_deps/curobo_env/python.exe -m pip install -e "ext_deps/curobo" \
    --no-build-isolation
```

This compiles the CUDA kernels ahead of time, which takes several minutes. cuRobo then prefers
`cuda.core` and falls back to the compiled extensions where it is unavailable.

> [!WARNING]
> ### Why both, and not just the JIT backend
>
> Either backend can be taken out on its own, and the JIT one is the more exposed: `cuda.core` is
> unsigned native code, so a Windows application-control policy can refuse to load it, and a broken
> wheel or a CUDA upgrade can do the same to either.
>
> You are free to install just one backend. We recommend both. However, if you choose to install just
> one backend, make sure it is the one you intend to use and that it is compatible with your system.

**Smoke test**

```bash
# UTF-8 so a cp1252 console can print the tick mark of cuRobo
$env:PYTHONIOENCODING=utf-8; ext_deps/curobo_env/python.exe \
    -m curobo.examples.getting_started.motion_planning

# and check that both backends resolve
ext_deps/curobo_env/python.exe -c "
import curobo._src.runtime as rt
from curobo._src.curobolib.backends import get_backend_name
print('auto:', get_backend_name())
rt.kernel_backend = 'pybind'
"
```

### 3. Robot configs

cuRobo ships `ur10e` and `franka`, but not `ur5e` or `ur3e`. The builder generates each UR arm descriptor from the vendored Universal Robots description and the arm's committed collision-sphere fit, producing `<model>.urdf` and `willy_<model>.yml`. Grippers are added separately by the cell at planner startup, so `--gripper` and `--coupling-mm` are not supported here.

```bash
ext_deps/curobo_env/python.exe scripts/curobo/fetch_ur_meshes.py
ext_deps/curobo_env/python.exe scripts/curobo/build_ur_config.py ur5e
ext_deps/curobo_env/python.exe scripts/curobo/build_ur_config.py ur3e
ext_deps/curobo_env/python.exe scripts/curobo/check_ur_descriptors.py --hand robotiq_2f85
```

Each descriptor records its arm and `carries_hand: false` in `_provenance`. The planner rejects descriptors for the wrong arm or descriptors containing a hand, rebuilding stale descriptors when necessary.

The builder also fixes an upstream `forarm_link` → `forearm_link` typo in `self_collision_ignore`. Without this fix, the dense Lula `ur5e` spheres cause persistent `forearm`/`wrist_1` collisions and `plan_pose` returns `None`.

Committed sphere fits make builds deterministic and require no simulator. Without a committed fit, the builder falls back to the Lula spheres and adds collision spheres fitted to the arm's own meshes. This fallback is randomized; use `--seed` for reproducible descriptors. The seed is recorded in `_provenance`.

Keep descriptors when measured planner results depend on a specific build. Full provenance is documented in [`src/robot/safety/planning/robot/PROVENANCE.md`](../src/robot/safety/planning/robot/PROVENANCE.md).

> Robot descriptors live inside the clone and must be rebuilt after a fresh clone. The boot banner only verifies that the sidecar Python environment exists, not the descriptor. `python -m src.robot.safety.planning --check --hand <hand>` reports the expected descriptor.


### 4. Wiring

The driver reaches this path only when `motion_planner="curobo"`, which is the default for both the
simulation driver (`SimRobotConfig.motion_planner`) and a real UR arm (`URConfig.motion_planner`).
What differs by deployment is what happens when the planner is not there, and the difference is
deliberate:

* **In simulation it degrades.** The arm probes once, warns, latches the result and plans with
  blind IK, so a host without the cuRobo environment still runs. A runner that reports a pass rate
  reports `curobo_degraded` next to it, because a rate measured on the fallback is not a
  measurement of the configured cell.
* **On a real UR arm it fails closed.** The move returns `CONTROLLER_REJECTED` where the planner is
  unavailable and `TIMEOUT` where no collision-free plan exists. It never degrades to blind IK, so
  a cell without a cuRobo environment does not move at all. Run the doctor before commissioning.

The other value is `"ik"`: the calibrated IK of the controller and a straight `moveJ` or `moveL`,
which knows nothing about the cell and will drive through anything in it.

Every location is an environment variable with an in-repo default, so a standard install needs no
configuration:

| variable | default |
|---|---|
| `WILLY_CUROBO_PYTHON` | `ext_deps/curobo_env/python.exe` |
| `WILLY_CUROBO_ROBOT` | `ur5e.yml`, read only by a client built without a descriptor: every cell names `willy_{arm}.yml` itself, and both drivers refuse a descriptor whose `_provenance` names another arm or records a hand |
| `WILLY_CUROBO_CUBOID_CACHE` | `16` collision-world cuboid slots reserved at boot: a real table plus far-away placeholders that `set_world` later fills |
| `WILLY_CUROBO_MAX_ATTEMPTS` | `16` plan attempts, each a fresh IK and trajopt seed batch, which is what finds a plan reliably on a tight query |
| `WILLY_CUROBO_GRAPH_FROM_ATTEMPT` | `1`, which is the cuRobo default: the first attempt stays trajopt-only, because the graph seeder can return no seed for a tight final approach and would otherwise skip every attempt |
| `WILLY_CUROBO_STDERR` | unset, so the stderr of the server is discarded; set a path to capture cuRobo diagnostics |

On the first `move()` the driver spawns the server and pays a one-off JIT warm-up of seconds; each
plan after that costs tens of milliseconds. The driver executes the full trajectory of cuRobo,
meaning the planner owns the final motion with no blind IK snap at the end, and closes the server
on `disconnect()`.

### 5. When it does not work

**Run the doctor first.** It loads every engine and names whatever refused, instead of leaving the
reader to infer it from a pick rate:

```bash
python -m src.robot.safety.planning --doctor    # 0 healthy | 1 degraded | 2 blocked by policy
```

Then, for anything it cannot see into, set `WILLY_CUROBO_STDERR`. Without it the stderr of the
sidecar is discarded and every failure looks the same from the outside.

| symptom | cause | fix |
|---|---|---|
| `cuRobo planner unavailable ... did not become ready` | the sidecar died at import | read the stderr log; the real error is there |
| `Error loading cuda.core backend: DLL load failed ... blocked` | an OS code-integrity policy refused the unsigned extension | the compiled backend (2b) takes over; where it is missing, build it. See the lockfile rule at the end of this page |
| `No curobo kernel backend available!`, the sidecar never ready; Code Integrity events 3077 and 3118 name `cuda\bindings\cyruntime.cp310-win_amd64.pyd` | Windows **Smart App Control** refused cuda.core's bindings, and no compiled backend (2b) was built, so no kernel loads and no sidecar starts | build 2b from a Developer Command Prompt. Until then, `python scripts/curobo/probe_band_admission.py --cpu-replica ext_deps/curobo/curobo/content` and `probe_turned_boxes.py --cpu-replica ...` judge the driver's decisions on a CPU replica of the sidecar, labelled `[replica]` on every line; the replica plans nothing (`probe_turned_boxes.py` skips its plan and says so), and only a GPU run shows the kernel and the pair naming agree. Another route: run the sidecar on a cuRobo environment whose compiled backend loads, with this repository's descriptors |
| `PyBind backend not available` | 2b was never run, or ran without `cl.exe` on `PATH` | re-run 2b from a Developer Command Prompt |
| `plan_pose` returns `None` everywhere | the self-collision spheres of the robot config overlap in every configuration | rebuild it (section 3); check the `self_collision_ignore` names |
| the server starts, then `ModuleNotFoundError` for a repository module | the sidecar is Python 3.10 and cannot import this package (3.11+) | the server is launched by file path, never `-m`; see `curobo_client.start()` |

**Honesty.** cuRobo here is a bucket 1 simulation capability, validated on-box in Isaac. It is not
certified motion safety, and it has never planned for a physical arm.

---

### One geometry, two consumers

[`src/robot/safety/data/ur5e_collision_meshes.npz`](../src/robot/safety/data/ur5e_collision_meshes.npz)
holds the DH-baked per-link collision meshes, read from the collision STL files of Universal Robots
and pinned to one upstream commit. It lives in the repository rather than being generated per box,
and it is the single source of truth for both the Coal self-collision backend and the sphere fit
that `scripts/curobo/fit_cover_spheres.py` writes for `scripts/curobo/build_ur_config.py` to read.

That coupling is the point. The collision model of the planner and the safety gate that checks the
answer of the planner derive from the same geometry, so the gate cannot disagree with the planner
about what the shape of the robot is. Regenerating the bundle for a new model means both change
together.

## Bringing this up on a fresh box

1. **Isaac Sim 5.1** standalone. Its `python.bat` becomes the on-box runner. Machine-local, and not
   something that can be vendored.
2. **Coal**, through the install script above. Optional: without it the guard falls back to the
   capsule approximation and the repository stays runnable.
3. **The cuRobo env and source**, also through the install script. Optional in simulation, where
   the driver falls back to blind IK at both build time and move time; not optional on a real UR
   arm, which refuses to move without it.
4. **Generate the robot configs**, which cuRobo does not ship. The install script does this, and by
   hand it is `ext_deps/curobo_env/python.exe scripts/curobo/build_ur_config.py ur5e` and the same
   for every arm the box runs, as in section 3. They are written inside the cuRobo clone, so a fresh
   clone always needs this step again.
5. **If your layout differs**, every path above is an environment variable and none of them is
   baked into the package. The `ISAAC_MODEL_DIR` and `CUROBO` constants of the generator are the
   last two.


## See also

- [`src/robot/safety/planning/`](../src/robot/safety/planning/) the anchor that reads these paths,
  and the `--check` and `--doctor` entry points
- [`src/robot/grasping/suction/`](../src/robot/grasping/suction/) the analytical scorer, which is
  the only one now
- [`locks/`](locks/) the two lockfiles the install script consumes
