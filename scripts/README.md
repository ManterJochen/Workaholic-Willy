# `scripts/`: what you run from a shell, beside the library

The tools here check a cell against its own configuration, fetch weights, install the motion engines,
probe real controller software, and build the artifacts the planner and the collision guard read.
The examples that call the library as your code would are not here: they live in
[`examples/`](../examples/README.md), and the library's own command lines are in
[`docs/cli.md`](../docs/cli.md).

Run every script from the repository root with the project environment active, unless its folder says
it needs another interpreter.

| Folder | What it does | Start here |
|---|---|---|
| [`checks/`](checks/) | holds this cell against its own configuration, and exits non-zero when the two disagree | [the checks](#the-checks) below |
| [`model_weights/`](model_weights/) | fetches the detector, segmenter, VLM and speech weights into `assets/models/hf` in the repository | `python scripts/model_weights/fetch.py --list` |
| [`ext_deps/`](ext_deps/) | installs Coal and cuRobo, builds a planner descriptor for every UR arm the stack models, and runs the doctor | [`ext_deps/README.md`](../ext_deps/README.md) |
| [`curobo/`](curobo/) | builds the planner's arm descriptors, fits its collision spheres, chooses retract poses, and measures an arm with a hand | `build_ur_config.py <model>` |
| [`grippers/`](grippers/) | writes a hand's collision bundle from its numbers, its vendor meshes or a USD, and measures its jaw | [`your_own_gripper.md`](../docs/runbooks/your_own_gripper.md) |
| [`isaac/`](isaac/) | bakes an arm's collision meshes into `src/robot/safety/data/`, from UR's own description or from Isaac's USD | `bake_ur_meshes_from_urdf.py` |
| [`ursim/`](ursim/) | real UR controller software in Docker: the SDK, the driver, both I/O grippers and a protective stop driven against it | [`ursim/README.md`](ursim/README.md) |
| [`trial/`](trial/) | runs [`your_own_gripper.md`](../docs/runbooks/your_own_gripper.md) in a copy of the tree with two hands whose answers are known | [`trial/README.md`](trial/README.md) |

```bash
python scripts/model_weights/fetch.py --list            # what is fetchable, and what is already here
python -m datagen.assets.fetch --list                   # the object meshes, and what they cost
python -m datagen cost --scenes 2000 --engine mujoco    # what a corpus costs before you start it
```

On Windows the engine install is
`powershell -ExecutionPolicy Bypass -File scripts/ext_deps/install.ps1`; its exit code says whether this
box can plan.

## The checks

Each check reads the tree `WILLY_PROFILE` names and takes one flag at most; a profile is set around
the command, never inside it. They answer questions the test suite cannot, because the suite runs
against fixtures and a check runs against the tree this box loads.

| Check | What it answers | Exit codes |
|---|---|---|
| [`cell_bringup.py`](checks/cell_bringup.py) `--live` | Does the arm answer, and does it stand inside the workspace box it will be held to? Connects, moves nothing | 0 inside, 1 disagrees, 2 nothing to connect to |
| [`safety_guards.py`](checks/safety_guards.py) | Does every wired guard of this cell refuse a violation of its own family, for its own reason? | 0 yes, 1 one did not, 2 no guard pipeline |
| [`camera_artifacts.py`](checks/camera_artifacts.py) | Does every calibration the tree declares open and resolve? Opens no device | 0 yes, 1 one does not, 2 none declared |
| [`grasping_switches.py`](checks/grasping_switches.py) | Which `robot.grasping` block is reachable in which grasp mode, and is every unwired switch refused? | 0 yes, 1 one is not, 2 no grasping block |

`cell_bringup.py` refuses to connect without `--live`, because reading a controller is not something a
check does because someone typed its name. `safety_guards.py` builds a real cell's driver, so it needs a
profile whose tree names a hand, such as your own cell's.

## Which interpreter

- The project environment runs `checks/`, `model_weights/`, `grippers/`, `trial/`, and in `curobo/` the
  sphere fit, the retract choice and the matrix gate, which spawns the cuRobo environment itself.
- The cuRobo environment's interpreter, `ext_deps/curobo_env/python.exe`, runs
  `curobo/build_ur_config.py`; `curobo/build_compiled_backend.bat` builds the kernel backend on Windows.
- `isaac/bake_ur_collision_meshes.py` and the Isaac probes need an Isaac install;
  `isaac/bake_ur_meshes_from_urdf.py` needs no Isaac and no GPU.

## The object meshes are fetched by the generator

The mesh fetcher belongs to the thing that consumes meshes, at
[`datagen/assets/fetch.py`](../datagen/assets/fetch.py), and the probes that grade a mesh library are
in [`datagen/eval/`](../datagen/eval/). See [`datagen/README.md`](../datagen/README.md).

## Two properties both fetchers hold

**TLS verification stays on, and the trusted roots can come from the OS.** With `truststore`
installed (it is in `requirements.txt`), both `scripts/model_weights/fetch.py` and
`datagen/assets/fetch.py` inject the operating system trust store. That answers a corporate proxy
whose CA lives in the Windows certificate store and not in `certifi`'s bundle. Only the set of trusted
roots changes, never whether the certificate is checked, which matters because these weights run in a
robot's perception path.

**The mesh fetcher checks the licence at the source, per model.** Anything that is not CC0, CC-BY or
your own is skipped before it is downloaded. Non-commercial, ShareAlike and NoDerivatives are all
refused. Where a collection publishes no per-object licence, the meshes carry the collection's
published terms and the fetcher says so rather than implying it verified something it did not.

To bring your own parts instead:

```bash
python -m datagen.assets --fetch --source custom --from <dir> --license own --attribution-text "<you>"
```

The licence is refused rather than defaulted, because nobody but you knows what your parts are licensed
as. It is written to `LICENSE.txt` beside the meshes, where the manifest and the audit read it back
later. From Python: [`examples/offline/datagen/05_bring_your_own_parts.py`](../examples/offline/datagen/05_bring_your_own_parts.py).
