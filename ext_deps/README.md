# `ext_deps/`: the local install root for external dependencies

The one place where the heavy, machine-local dependencies of Willy get installed, and the one page
that says how to install them, how to verify them, and what to do when they refuse.

`gitignored except this README, the lockfiles and .gitignore` `no admin rights`
`no PATH or registry changes` `delete the folder to undo it`

Nothing here is committed, but the code defaults look here, so a fresh box has one obvious install
target instead of scattered absolute paths.

> Isaac Sim is the deliberate exception. It is a multi-gigabyte standalone installer with its own
> bundled Python, and it stays where NVIDIA puts it. Everything else Willy bridges to lives here,
> and the Isaac section below says what that bridge is.

> [!NOTE]
> **None of these is a hard dependency of the repository.** Mock mode and CI never touch them, and
> in simulation the arm falls back on its own: cuRobo to blind IK, Coal to the capsule
> self-collision proxy. On a real UR arm the cuRobo path is fail-closed instead and refuses to
> move, which section 4 explains. They are measured enhancements to the on-box validation cell.

---

## The three external systems

Everything else Willy needs installs from the requirements files of the repository, into whatever
Python runs Willy. These three are different: they are heavy, they are machine-local, and the
repository bridges to them rather than containing them.

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

There is no bridge beyond the import boundary. The sim driver lazy-imports `isaacsim.*` only after
mock mode has been ruled out, so macOS and CI never touch it, and `src/robot/drivers/sim/` holds no
Isaac SDK import until `connect()` runs. The cell config and the scene authoring live in the
repository; only Isaac itself is machine-local.

One caveat, because it looks like a crash and is not. The headless `SimulationApp.close()` of Isaac
can segfault on shutdown. It is a known upstream teardown issue and it fires after the run has
already produced its result, so a caller that needs a reliable result captures it before stopping.
[`src/robot/drivers/sim/session.py`](../src/robot/drivers/sim/session.py) says so at the method
where it happens.

### Three environments, three different torch versions

This looks like a conflict and is not. There are three separate Pythons here:

| Environment | Python | torch | Who uses it |
|---|---|---|---|
| the repository venv | 3.11 | `2.7.1+cu128` | everything except sim runs and cuRobo planning |
| `ext_deps/curobo_env` | 3.10 | `2.7.0` | the cuRobo sidecar, and nothing else |
| the bundled Python of Isaac | 3.11 | its own, cu128 | every `willy_sim` run |

cuRobo needs its own environment for a reason unrelated to torch: it wants warp 1.14 while Isaac
ships 1.8.2, and one Python process holds exactly one warp. That is what the sidecar is for.

---

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

From nothing to a verified stack. It bootstraps micromamba into this folder, builds both
environments from `locks/`, clones and pins cuRobo, installs both of its kernel backends, generates
the ur5e and ur3e descriptors, and ends by running the doctor, so its exit code means "this box can
plan" rather than "the downloads finished".

`-Clean` deletes the targets first, which is how to test that it really does rebuild from nothing.
`-Component coal|curobo` does one of them.

```bash
python -m src.robot.safety.planning --doctor   # 0 healthy | 1 degraded | 2 blocked by policy
```

Nothing outside `ext_deps/` is touched: no system conda, no PATH or registry changes, no admin
rights. Deleting this folder undoes all of it.

### Without administrator rights, in full

Measured 2026-09-08, because "no admin rights" was a claim and not yet a check. Three things
contradicted it, and all three are closed now.

* **The package caches.** `MAMBA_ROOT_PREFIX` was already inside `ext_deps/`; pip's was not, and held
  607 MB in `AppData\Local\pip\Cache`. `PIP_CACHE_DIR` points into `ext_deps/` too now, so deleting
  the folder really does undo everything.
* **`git lfs install`** wrote `filter.lfs.*` into the user's global `.gitconfig`, which outlives
  `ext_deps/` and belongs to every other repository on the machine. It bought nothing: the pinned
  cuRobo revision ships an empty `.gitattributes`, so it declares no LFS filters and a clone of it
  produces no `.git/lfs` directory. The call is gone from the script and from the manual recipe below.
* **MAX_PATH, the one nobody predicts.** The deepest installed path was 281 characters against a 260
  limit, and it worked here only because `HKLM\...\FileSystem\LongPathsEnabled` is 1, which needs
  administrator rights to set. All 40 over-length paths belonged to `cuda-nvvp` and `nsight-compute`,
  a visual profiler and a profiler GUI the planner never loads; they and their metapackages are out
  of the lock. That moves the gate rather than removing it: the deepest survivor is 219 characters,
  so a checkout root longer than 39 characters still needs the key.

Two prerequisites this script cannot bootstrap, and it now says so in seconds instead of failing
minutes in: `git` (Git for Windows installs per-user, choose "Only for me", or use the portable
build) and `tar` (ships with Windows 10 1803 and later).

The compiled backend is the normal case and the fallback is an emergency. A C++ toolchain is the one
thing here whose usual installer wants administrator rights. Where none is present the build script
exits 2, the install continues with a loud warning, and the planner runs on `cuda.core` alone with no
spare, which is exactly the configuration that failed on 2026-08-30, when an application-control
policy refused `cuda.core` on this machine and the compiled backend was the only reason the planner
kept working. Build it as soon as a toolchain exists. `build_compiled_backend.bat` accepts an
already-active x64 environment now, so a per-user toolchain is enough and vswhere is not required.

The two sections below are what that script does, by hand. Read them when it fails, when installing
on Linux or macOS, or when a version has to change.

---

## Coal: the exact-mesh self-collision engine

The mesh-against-mesh self-collision backend (`self_collision.backend='fcl'`) and the continuous
collision-avoidance guard both run on **Coal**, the maintained successor to `hpp-fcl`. It powers
the closest-distance queries; `python-fcl` is the guarded fallback.

> [!WARNING]
> **Honesty, and it is safety-critical.** This is a software collision-avoidance layer, not "the
> safety system". It reduces collision risk in simulation, which is bucket 1. The real-hardware
> safety guarantee is independent certified functional safety: hardware E-stop circuits and
> ISO 10218, ISO/TS 15066 and ISO 13849, running independently of this software, which is bucket 3.
> A green simulation result is not "commercially safe". Coal makes the check better; it does not
> certify the robot.

### Why Coal and not pip `hpp-fcl`

`hpp-fcl` on PyPI is unmaintained. Coal (`coal` on conda-forge, 3.x) is the maintained project:
faster GJK, and a usable distance lower bound. On this codebase the swap was proven
behaviour-identical to `python-fcl` at the distance level, on the values the guard actually
triggers on, and Coal is the faster of the two.
[`src/robot/safety/_fcl_self_collision.py`](../src/robot/safety/_fcl_self_collision.py) records
both facts with their values.

### Install (Windows, no system conda)

Coal has no Windows pip wheel, so it comes from conda-forge through the self-contained micromamba
this folder bootstraps:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\ext_deps\install.ps1 -Component coal
```

That builds `ext_deps/coal_env` from the committed lockfile `ext_deps/locks/coal_env.win-64.lock`
and verifies it with the doctor. No environment variable is needed: `WILLY_COAL_PREFIX` defaults to
`ext_deps/coal_env`, and it is set only in order to install elsewhere.

The backend reaches Coal by injecting the env into whichever interpreter is running. It adds
`<prefix>/Library/bin` as a DLL directory and appends `<prefix>/Lib/site-packages`, so the host's
own numpy still wins and Coal works against numpy 1.x and 2.x either way. The `python.exe` of the
coal env is never invoked, so whatever that interpreter can or cannot import is irrelevant.

```bash
python -m src.robot.safety.planning --doctor      # loads Coal and runs a real distance query
python -m pytest tests/test_safety_self_collision.py      # the suite of the guard itself
```

### Changing the Coal version

The distances the self-collision guard reports are a safety-relevant output, so a version change is
proven rather than assumed. Dump every pairwise link distance over a fixed set of poses on the old
and the new engine and compare the full-precision values; the bump is admissible only where they
agree exactly.

### Linux, macOS, and the fallback

`conda install coal -c conda-forge`, or a fresh micromamba env, and `import coal` then works
directly, so `WILLY_COAL_PREFIX` is usually unnecessary. The cmeel pip wheels also exist there.

Where Coal is absent the backend imports `python-fcl` instead, which is pinned in
`requirements.txt` and ships Windows wheels. Where neither engine nor the mesh bundle is present,
`make_backend` returns `None` and the one-shot self-collision check falls back to its capsule path
rather than crashing. Only the `capsule` backend needs neither dependency, and it is not the
default: `self_collision.backend` ships as `fcl`, so a standard install does want one of the two.

The continuous collision-avoidance guard is opt-in (`continuous_guard=True`, default off) and needs
a mesh backend on this box. Without one it logs a warning and stays off; it does not run unguarded
in silence.

---

## cuRobo: the collision-aware motion planner

The `motion_planner="curobo"` path plans global, collision-free trajectories with NVIDIA
[cuRobo](https://curobo.org).

cuRobo cannot run inside the Isaac process: it wants `warp 1.14` while Isaac 5.1 ships `warp 1.8.2`,
and one Python process holds exactly one `warp` (first on `sys.path` wins). So cuRobo runs in its
own environment and the driver talks to it over stdio, from
[`src/robot/safety/planning/curobo_client.py`](../src/robot/safety/planning/curobo_client.py) to
[`src/robot/safety/planning/curobo_planner_server.py`](../src/robot/safety/planning/curobo_planner_server.py).
Process isolation solves the warp clash and keeps the validated Isaac env untouched.

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
`cuda.core` and falls back to the compiled extensions where it is unavailable. On Windows,
[`scripts/curobo/build_compiled_backend.bat`](../scripts/curobo/build_compiled_backend.bat) does it
with the MSVC environment already sourced.

> [!WARNING]
> ### Why both, and not just the JIT backend
>
> Either backend can be taken out on its own, and the JIT one is the more exposed: `cuda.core` is
> unsigned native code, so a Windows application-control policy can refuse to load it, and a broken
> wheel or a CUDA upgrade can do the same to either.
>
> cuRobo has a documented fallback from `cuda.core` to `pybind`, but a fallback needs something to
> fall back to. With only the JIT backend installed there is nothing, and the failure is quiet: the
> planner dies, the driver degrades to blind IK, the self-collision guard correctly refuses the
> poses that path proposes, and a validation run collapses to a pass rate with no message naming
> any of it.
>
> Two independent backends mean one policy verdict, one broken wheel or one CUDA upgrade cannot
> take the planner down. It costs a few minutes of compile time, once.

**Which revision.** Pinned to `8e734f3`. Two commits in it matter here:

* `82ea96e`, *Fix CUDA 13 and pybind backend source build*. It also repairs the fallback itself:
  `_try_load_cuda_core_backend` used to call `log_and_raise` instead of returning `None`, so a
  missing `cuda.core` raised rather than falling through to pybind. Without this commit the
  fallback in 2b cannot work no matter what is installed.
* `8e734f3`, *Fix mesh SDF gradient sign for query points outside the surface*. A correctness fix
  in collision distance, which a collision-avoiding planner is entitled to care about.

The newest tag, `v0.8.0`, is older than this commit, so the pin tracks the branch rather than a tag.

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

### 3. Robot configs (assembled, not shipped)

cuRobo ships `ur10e` and `franka` but no `ur5e` or `ur3e`. One script assembles either from on-box
ingredients, the canonical URDF of Isaac plus its Lula collision spheres, writing `<model>.urdf`
and `<model>.yml` into the content directory of the clone:

```bash
ext_deps/curobo_env/python.exe scripts/curobo/build_ur_config.py ur5e
ext_deps/curobo_env/python.exe scripts/curobo/build_ur_config.py ur3e
```

The builder fixes an upstream `self_collision_ignore` typo, `forarm_link` for `forearm_link`. It is
harmless with the sparse ur10e spheres, but with the dense Lula ur5e spheres it leaves
`forearm` against `wrist_1` self-colliding in every configuration, so `plan_pose` returns `None`
everywhere until the key is renamed.

The arm-link surface augmentation is UR5e-only and the script says so: the fitted meshes and the DH
chain are ur5e-specific, so any other model keeps its own model-tuned Lula arm spheres.

The sphere fit is not deterministic, so two runs do not produce the same `ur5e.yml`. Everything
that is not a sphere centre is stable. Where a measured planner result depends on one specific
descriptor, keep that file: a rebuild will not reproduce it. Full record:
[`src/robot/safety/planning/robot/PROVENANCE.md`](../src/robot/safety/planning/robot/PROVENANCE.md).

> A robot config lives inside the clone, so a fresh clone needs this step again. The boot banner
> cannot verify it: it reports that the Python of the sidecar exists, and explicitly not that the
> descriptor inside that separate environment is present.
> `python -m src.robot.safety.planning --check` names the descriptor it expects.

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
| `WILLY_CUROBO_ROBOT` | `ur5e.yml` |
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
| `PyBind backend not available` | 2b was never run, or ran without `cl.exe` on `PATH` | re-run 2b from a Developer Command Prompt |
| `plan_pose` returns `None` everywhere | the self-collision spheres of the robot config overlap in every configuration | rebuild it (section 3); check the `self_collision_ignore` names |
| the server starts, then `ModuleNotFoundError` for a repository module | the sidecar is Python 3.10 and cannot import this package (3.11+) | the server is launched by file path, never `-m`; see `curobo_client.start()` |

**Honesty.** cuRobo here is a bucket 1 simulation capability, validated on-box in Isaac. It is not
certified motion safety, and it has never planned for a physical arm.

---

### One geometry, two consumers

[`src/robot/safety/data/ur5e_collision_meshes.npz`](../src/robot/safety/data/ur5e_collision_meshes.npz)
holds the DH-baked per-link collision meshes, vertex-exact to under 0.15 mm against the Isaac USD.
It lives in the repository rather than being generated per box, and it is the single source of
truth for both the Coal self-collision backend and the cuRobo sphere fit that
`scripts/curobo/build_ur_config.py` performs.

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
   for `ur3e`. They are written inside the cuRobo clone, so a fresh clone always needs this step
   again.
5. **If your layout differs**, every path above is an environment variable and none of them is
   baked into the package. The `ISAAC_MODEL_DIR` and `CUROBO` constants of the generator are the
   last two.

**What this is and is not.** Everything here is simulation evidence: software collision-aware
planning and mesh self-collision, in a simulator. It is not certified functional safety. The
transfer to real hardware, meaning a real UR base yaw, a real gripper mesh and real planner timing,
is still open. The repository runs fully without all three of these systems, through mock mode and
the automatic fallbacks. They are measured enhancements to the on-box validation cell, not
requirements.

## Two rules this folder is built around

> [!WARNING]
> ### The lockfiles are not a nicety
>
> On Windows, an application-control policy such as Smart App Control refuses unsigned native code
> that the reputation service of Microsoft does not vouch for, and a conda build published days ago
> usually has no reputation yet. So an unpinned `micromamba create` can produce an environment that
> installs perfectly and then cannot load.
>
> The verdict is a property of the contents of the file, so it belongs to the specific package
> build rather than to the version, and it changes over time as the reputation service catches up.
> Re-downloading, copying or moving an install changes nothing; choosing a build with reputation is
> the fix, and that is what `locks/` is for. The lock pins `bzip2` to build `_9` for exactly this
> reason, an earlier build of the same version than the one conda-forge would otherwise resolve.
>
> The rule that follows: reach decides the verdict, not recency, so when a package is refused, step
> back to the build before it rather than reaching for the policy switch.
>
> To read the verdicts themselves (Windows):
>
> ```powershell
> Get-WinEvent -LogName 'Microsoft-Windows-CodeIntegrity/Operational' -MaxEvents 200 |
>   Where-Object { $_.Id -in 3033,3077 } | Select-Object TimeCreated, Message
> ```
>
> Re-lock with `micromamba env export --explicit` and re-run the doctor.

> [!CAUTION]
> ### Nothing non-commercially licensed is installed here
>
> An RGB-D suction network was once installed into this folder and wrapped as a second suction
> scorer. It was removed because its training data is released under a non-commercial licence, and
> it is not coming back under a different name:
> [`tests/test_license_boundary.py`](../tests/test_license_boundary.py) fails CI on the dependency,
> on the import and on the prose, because an install guide is how a forbidden dependency actually
> arrives.
>
> This file is the reason the prose half of that check has an exception list. `ext_deps/` is in the
> skip list of the scanner, correctly, since it holds third-party payloads that are not ours to
> police. This README is the one thing in here that is ours, so it is named explicitly and scanned
> anyway.
>
> The suction scorer seam is unchanged: implement the `SuctionScorer` Protocol and drop it in. What a
> replacement needs is licence-clean training data, which is what [`datagen/`](../datagen/) is for,
> not another vendor.

---

## See also

- [`src/robot/safety/planning/`](../src/robot/safety/planning/) the anchor that reads these paths,
  and the `--check` and `--doctor` entry points
- [`src/robot/grasping/suction/`](../src/robot/grasping/suction/) the analytical scorer, which is
  the only one now
- [`locks/`](locks/) the two lockfiles the install script consumes
