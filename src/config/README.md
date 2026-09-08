# `src/config`: the configuration system

Type-safe, immutable YAML configuration, validated by Pydantic v2, loaded once at startup and never
mutated by anything downstream. This package is the loader and the schemas; the YAML you edit is the
[`config/`](../../config/) tree at the repository root.

This is the bottom of the dependency stack. Every other subsystem reads a frozen `AppConfig` and
nobody mutates it.

**Typos are rejected at load time, not silently dropped.** Every model is a `StrictModel` with
`extra="forbid"`, so a misspelled key is a `ConfigError` naming the file and the line, the profile
layer that file belongs to, and a near-miss suggestion. It is never a setting that quietly never
applied.

## How a config is built

The YAML tree is read, `${VAR}` references are substituted before parsing, profile overlays are
deep-merged left to right, each section is validated by its `StrictModel`, and the result is a frozen
`AppConfig` cached by data directory and profile chain.

```python
from src.config import load_config

cfg = load_config()                            # validated, immutable AppConfig

cfg.camera.cameras.primary_rig_id
cfg.camera.stereomatcher.num_disparities
cfg.models.stt.model_id
cfg.runtime.image_encoding.frame_quality
```

`cfg` is frozen: assigning to `cfg.camera.cameras.primary_rig_id` raises. Configs are values, not state.
Schema classes are also safe to import on their own, without the loader:

```python
from src.config.schema.camera import CameraSystemConfig
```

## Asking the tree about itself

```bash
python -m src.config                     # validate the shipped tree      (exit 0 = OK)
python -m src.config --print             # also dump the parsed AppConfig as JSON
python -m src.config --data DIR          # validate a different tree

python -m src.config decisions           # only what differs from the schema default
python -m src.config where fusion --tier all
python -m src.config explain robot.grasping.fusion.enabled --profile ur5e,eth2
```

Exit codes: `0` succeeded, `1` `ConfigError` (file, parse or schema), `2` bad CLI arguments. A query
that finds nothing still exits `0`: `explain` reports the key as unknown and offers near matches, and
`where` says no key matches.

`explain` reports a key's type, constraints, default, tier, which file and layer set the winning value,
the whole override chain, and the comment written above that line, which `yaml.safe_load` discards.
`where` searches the schema rather than the files, so it finds the fields no YAML mentions.

These three exist because the tree has to be asked rather than read. A YAML file states what this cell
decided; everything it stays silent about is the schema default, in force and unchanged, and a reader
who greps the files sees only half the configuration.

## What is here

| Path | Role |
|---|---|
| [`__init__.py`](__init__.py) | Public surface: `load_config`, `load_robot_config`, `reload_config`, `default_data_dir`, `ConfigError`, `ConfigTree`, `LoadedTree`, and the section models `AppConfig`, `CameraConfig`, `ModelsConfig`, `RobotConfig`, `RuntimeConfig`. Schema classes import eagerly; loader helpers load lazily, so a schema-only import needs no YAML dependency. |
| [`loader.py`](loader.py) | The pipeline above. Result cached by data directory and profile chain. |
| [`tree.py`](tree.py) | `ConfigTree` and `default_data_dir()`: the one place that answers "which directory does `load_config()` read", so no caller rebuilds that walk from its own location and gets a silently wrong answer. |
| [`_merge.py`](_merge.py) | The recursive dict merge profile overlays are built on. |
| [`explain.py`](explain.py) | Value, type, default, tier, which layer set it, and the YAML comment above that line. Backs `explain`, `where` and `decisions`, and the console's provenance view. |
| [`edit.py`](edit.py) | Writing a bench measurement back in: allowlisted keys only, one line rewritten in place so comments survive, the group validated as one transaction, files restored if the loader rejects the result. |
| [`schema/`](schema/) | The `StrictModel` schemas: [`app.py`](schema/app.py), [`runtime.py`](schema/runtime.py), [`camera/`](schema/camera/), [`models/`](schema/models/), and [`robot/`](schema/robot/) split per vendor and subsystem (`ur`, `kuka`, `sim`, `dummy`, `safety`, `grasping`, `calibration`, `kpi`, `rl`, `tool_frame`). |
| `__main__.py` | The `python -m src.config` validator and query CLI. |

## The tree on disk

```
AppConfig
  camera  : CameraConfig
    cameras       (primary_rig_id, rigs, stereo_calibration: ChArUco/ArUco board)
    stereomatcher (SGBM + WLS + temporal; the YAML keeps OpenCV's camelCase)
    hand_eye      (independent eye-to-hand and eye-in-hand workflows)
  models  : ModelsConfig
    objectdetector, segmenter, oneformer, rtdetr  (the model blocks)
    detector, segmenter_backend                   (the do-it-yourself choice)
    pipeline      (the one-block way of choosing a perception stack)
    handdetect, gesturedetect                     (standalone, not in the grasp path)
    stt           (Whisper)
  robot   : RobotConfig or None   (absent means camera-only)
  runtime : RuntimeConfig
```

Layout under [`config/`](../../config/):

```
camera/cam.yaml            (required)   camera/stereomatcher.yaml   (required)
camera/hand_eye.yaml       (optional)
models/*.yaml              (auto-discovered; every top-level key merged, duplicates rejected)
robot/robot.yaml           (optional)   robot/kpi_thresholds.yaml   (the KPI gate)
app/runtime.yaml           (optional; schema defaults otherwise)
grasping_presets/{easy,dense_clutter,verification_heavy}.yaml
all_keys/                  (a reference tree, see below)
```

`load_config("/path/to/tree")` accepts a different directory, which must follow the same
`camera/ models/ robot/ app/` layout.

### `config/all_keys/` is the reference tree, not the one that loads

[`config/all_keys/`](../../config/all_keys/) is a complete tree in which every key the schema accepts
is written out with what it does, its default, and its legal values. Nothing loads it by default. It
validates on its own, so it is checkable rather than merely documentation:

```bash
python -m src.config --data config/all_keys
```

Read it to find out what exists; edit the shipped tree beside it to change what your cell does. Do not
copy it over the shipped tree, because a file that restates every default turns each of those lines
into something a reader must check against the schema.

### Two kinds of line stay written even when they equal their default

Safety bounds, and the site facts a bring-up has to find (`ur.ip`, `kuka.controller_ip`,
`sim.robot_model`, `grasping.default_mode`, `rl.mode`). A leaner file that hides them is not a better
file. Rig catalogues in `config/camera/cam.yaml` also keep their full per-rig fields, because ragged entries
where one rig lists `fps` and the next does not are harder to read, not easier.

When you add a field, document it on the schema field (a `#:` or a plain `#` block above it; `explain`
harvests both). Only write it into a YAML if the cell is actually choosing something.

## Profiles and overlays

Set `WILLY_PROFILE`, or pass `--profile`, to layer per-file overlays on top of the base YAML. For a
base `foo.yaml` the loader deep-merges `foo.<profile>.yaml` from the same directory.

Profiles compose. `WILLY_PROFILE` takes a comma-separated chain, applied left to right:

```bash
WILLY_PROFILE=ur5e                # a real UR5e on a bench
WILLY_PROFILE=ur5e,eth2           # the same bench with two fixed RGB-D cameras
WILLY_PROFILE=sim                 # the Isaac cell (a UR5e)
WILLY_PROFILE=sim,ur3e            # the same sim cell driving a UR3e
WILLY_PROFILE=ursim,ursim_ur3     # real UR controller software, UR3e kinematics
```

Layers are independent dimensions, not alternative whole configurations, which is why they chain
rather than fork. A profile per combination would have to copy the same measured values per robot and
the copies would drift; chained, each value is stated once. It also keeps an experiment interpretable:
change the robot layer alone, or the camera layer alone, and a measured difference is attributable to
it.

### The shipped layers

| Layer | What it changes |
|---|---|
| `ur5e` | a real UR5e cell: workspace box, home pose, safe pose, bench fixture, planning world, and the four bench measurements written out as commented blocks |
| `ur3e` | reach-anchored geometry and kinematics for the shorter arm |
| `eth2` | the fusion half of a cell with two fixed RGB-D cameras; chain it after a robot layer |
| `sim` | the Isaac cell; also overlays `config/camera/hand_eye.sim.yaml`, `config/models/object.sim.yaml`, `config/models/segmenting.sim.yaml` |
| `ursim` | points the UR driver at a URSim container, and declares the two things `connect()` fails closed on (payload, tool frame) |
| `ursim_ur3` | chained onto `ursim`; changes one thing, `ur.model`, which keys the DH chain, the collision bundle and the cuRobo config |
| `tiltcam` | two tilted eye-to-hand D435s instead of one nadir camera |
| `console_dummy` | hardware-free: dummy arm and gripper with a declared tool frame. The cell the operator console is developed against |
| `web` | the webcam-pair rig and its runtime |
| `decision` | turns the default-off AUTO decision gate on, alone, so a measurement of it means one thing |
| `rl_datagen` | the advanced grasping blocks on plus `rl.mode: rl_shadow` with a bootstrap policy: the layer that makes a record log trainable |

Four of these are worked cell examples rather than single-dimension switches: `ur5e`, `ur3e`, `sim`
and `eth2`. Read [`config/robot/robot.ur5e.yaml`](../../config/robot/robot.ur5e.yaml) before you
describe your own bench. It states which values it ships and which it deliberately refuses to guess:
a wrong value that fails closed, such as a workspace box that is too small, ships as a worked example
with its assumption stated, because it refuses a motion visibly and an operator widens it. A wrong
value that fails open does not ship at all. A tool frame or a payload that is merely plausible drives
the arm into the bench and logs a success, so those are written out as commented blocks saying what to
measure, and the shipped refusals in `run_config_preflight` and `URRobotArm.connect` stay armed.

Prefer a new profile layer over an edit to `robot.yaml` when you are measuring something. A block that
has not been measured on your cell does not belong in the default, and a block being measured should be
the only thing that changed.

### Two traps in the two-camera cell

[`config/robot/robot.eth2.yaml`](../../config/robot/robot.eth2.yaml) documents both in place, and both
are the kind that produce a working-looking cell.

**The primary camera does not go in `fusion.cameras`.** That map is every camera except the primary,
because the primary already streams through the main perception source. Four places read the map and
two of them read it differently: the extra-camera rig builder filters the primary out, while the
orchestrator's configured-camera list does not, so a primary listed there is permanently among the
cameras that delivered no frame. That is a warning on every pick under
`on_camera_unavailable: degrade` and a raised error on every pick under `refuse`. Note that
`robot.sim.yaml` does list all of its cameras including the primary, which is correct there and wrong
here; do not copy the shape of that block.

**The primary camera's own artifact is a separate key, and it is the one that satisfies the refusal.**
`fusion.extrinsics_artifact_path` is what `AutonomousGraspService.from_robot_config` checks for a
CAMERA to BASE transform. The calibration runner prints a `fusion.cameras` block to paste after each
camera, and that block alone leaves this key null, so a cell can be fully calibrated, load both
artifacts, and still be refused at build.

### Merge rules, the reset sentinel, and environment substitution

Dictionaries merge recursively per key; scalars and lists replace the base value; a later layer wins
over an earlier one, and the base YAML is the earliest layer of all. A `null` overlay leaf keeps the
base value, so a partial overlay never accidentally wipes a field. The sentinel string `"__null__"` is
the only way an overlay can unset a base value to `None`.

If `WILLY_PROFILE` names a layer with no matching `*.<layer>.yaml` anywhere in the tree, loading fails
with `ConfigError`, checked per layer, so a typo in the middle of a chain cannot merge as a silent
no-op and leave the cell running another robot's geometry.

Any string in any YAML may reference an environment variable, substituted before parsing, so the result
must stay valid YAML:

```yaml
robot:  { ur: { ip: ${ROBOT_IP:-192.168.1.100} } }    # default supplied
models: { stt: { model_path: ${MODEL_DIR}/whisper } }  # required: ConfigError names MODEL_DIR if unset
```

Sim is a profile of this one tree, not a second tree. `src.willy_sim.config.load_sim_config` builds the
chain for you from a robot model and any extra layers.

## Robot and fail-closed safety

`config/robot/robot.yaml` is vendor-block shaped: the top-level `vendor` key selects which sibling block the
runtime consults. The `sim` block carries `mock_mode`, a pure-Python kinematic mock that needs no
Isaac, plus the Isaac scene-authoring extras read by `src.willy_sim` rather than by the bare driver.
KUKA is schema-validated and its EthernetKRL driver has never been validated on real hardware;
controller-side templates ship under
[`config/robot/templates/kuka/`](../../config/robot/templates/kuka/).

`robot.safety` is a vendor-neutral block applied on top of controller-side limits (UR safety planes,
KUKA SafeOperation and so on). The more restrictive value always wins, so this block can only ever
tighten a cell, never loosen it. Each sub-block carries its own gate, so one guard can be dropped
without touching the others.

| Guard block | Checks | Gate |
|---|---|---|
| `limits` | workspace-face margins | `enforce` |
| `joint_limits` | per-axis angle margins | `enforce` |
| `ik_quality` | joint jump, singular value, condition number, limit proximity | `enforce` |
| `motion_continuity` | max joint, orientation and TCP step per move | `enforce` |
| `payload` | mass, centre of gravity and inertia envelope | `enforce` |
| `self_collision` | link-link and link-fixture, capsule or mesh backend | `enforce` |
| `dwell` | post-stop dwell and steady-before-motion | `require_steady_before_motion` |
| `planning_world` | the boxes the trajectory planner routes around | `enabled` |
| `trajectory_check` | every configuration of a planned path, not only its end | `enabled` |

The single entry point is `SafetyPreflight` in [`src/robot/safety/`](../robot/safety/). Each driver
builds one with `SafetyPreflight.from_safety_config(...)` and evaluates it before commanding motion.
Rejections surface as typed `MotionStatus` reasons: `WORKSPACE_REJECTED`, `JOINT_LIMIT_REJECTED`,
`IK_QUALITY_REJECTED`, `SELF_COLLISION_REJECTED`, `PAYLOAD_REJECTED`, `CONTINUITY_REJECTED`. No data
means no motion, by design.

The emergency stop is deliberately not here. It is a hardware and controller function, and nothing in
this package can enable, disable, route or observe it.

### A switch that would lie is refused at load

`robot.grasping` configures optional grasp behaviour layered on top of the locked mode profiles. Its
values cannot relax safety or unlock a recovery action the active mode forbids, and the defaults
reproduce the shipped open-loop jaw pick byte for byte.

`RobotGraspingConfig.UNWIRED_SWITCHES` lists any switch that would land in the cell's telemetry while
nothing reads it back. Setting one is a load-time refusal, not a warning, because enabling it would
give you a cell that reports a capability and does not have it. Today the list holds one entry,
`occlusion.hard_reject_enabled`, and it stays there until the occlusion score itself is trusted.

## Choosing a perception stack

`models.detector` and `models.segmenter_backend` are two independently settable keys with no
cross-check, so every detector and segmenter combination builds, including ones where the prompt means
something different to each half. They are kept as the do-it-yourself path.

The everyday way is one block:

```yaml
models:
  pipeline:                     # absent means the two keys above are in force, byte-identically
    kind: zero_shot             # zero_shot (any prompt) | closed_set (a trained class list)
    zero_shot:
      backend: grounded_sam     # grounded_sam (a phrase grounder + a segmenter) | vlm
      segmenter: sam2           # sam2 | oneformer, on either backend
```

`build_perception(cfg.models)` resolves it, and so does the physical cell, through
`PerceptionSpec.from_config(cfg.models).build()`.

Two combinations are refused at load, each naming the legal one:

- `kind: closed_set` with `router.enabled: true`. Routing decides between open-vocabulary backends, and
  a closed-set detector answers only from its class list.
- `router.enabled: true` with any `zero_shot.backend` other than `vlm`. The router chooses between the
  phrase grounder and the VLM, and no other backend configures a VLM for it to choose.

So the router is opt-in and it needs `backend: vlm`:

```yaml
models:
  pipeline:
    kind: zero_shot
    zero_shot: { backend: vlm }
    router:    { enabled: true }
```

Left unwritten, `router.enabled` is corrected to `false` rather than refused, so a bare `pipeline: {}`
is legal. Only an explicit `true` on an illegal stack is an error.

Nothing pins the segmenter to a backend. Both mask sources implement the same box-prompted contract, so
either works with either grounding model, and which one segments better is unmeasured. It is a knob,
not a recommendation.

This is fail-closed rather than warn-and-continue, because the failure it exists to remove is a stack
that runs confidently and grounds the wrong thing, and a warning in a log has never stopped a grasp.

## Notes

**Grasping presets bypass schema validation.** The three overlays under
[`config/grasping_presets/`](../../config/grasping_presets/), `easy` (min-risk single pick),
`dense_clutter` (bin picking with bounded next-viewpoint recovery and an uncertainty fail-closed) and
`verification_heavy` (closed-loop with mandatory post-grasp verification), are merged onto
`robot.grasping` by `apply_preset` in the replay package. They are operator overlays, not validated
`AppConfig` fields.

**Writing config from a tool goes through `edit.py`, and it is deliberately narrow.** Eight keys are
writable: the payload mass and centre of gravity, the three tool-frame keys, a camera rig's serial
number, and the two controller addresses `robot.ur.ip` and `robot.kuka.controller_ip`. They are there
because they are measurements and site facts rather than policy. Limits, thresholds and safety toggles
are refused by name, because in this tree the evidence for such a value lives in the comment above it,
and a tool that writes the number without showing the comment invites changing a value whose reason
nobody remembers. A group is written as one transaction (the three tool-frame keys only validate
together), the result is re-validated by the real loader, and every touched file is restored byte for
byte if it is rejected.

**`kpi_thresholds.yaml` is the KPI gate** consumed by `python -m src.robot.grasping.replay`. That gate
is a synthetic contract self-check: it proves the telemetry and KPI plumbing agree with themselves, not
that any grasp is good, and it says so in its own provenance block.

**Config owns validated data only.** It never opens a camera and never runs calibration; that is
`src.camera` and `src.robot.execution.real_cell.calibrate`.

**There is an optional HTTP surface**, the operator console in [`api/`](../../api/README.md). Nothing
under `src/` imports it, and the web framework it needs is an optional extra.

**Caching.** `load_config()` is cached by absolute data directory and active profile chain. Call
`reload_config()` to invalidate it after editing files.

**Validation you can rely on.** `numDisparities` must be a positive multiple of 16 and `blockSize`
odd; `temporal_alpha` sits in `[0, 1]`; a duplicate `rig_id` is rejected; `aruco_dict_name` is checked
against the OpenCV catalogue; duplicate model keys across `models/*.yaml` are rejected; and camera rigs
are a discriminated union on `source` (`webcam_pair`, `single_device`, `rgbd`).

**Changing config in anger has runbooks.** This page describes the tree; the ordered procedures
that edit it on a live cell live under [`docs/runbooks/`](../../docs/runbooks/):
[`real_cell_first_pick.md`](../../docs/runbooks/real_cell_first_pick.md) for the first pick on a
physical arm and [`ur3e_cell_bringup.md`](../../docs/runbooks/ur3e_cell_bringup.md) for a robot the
stack has not run before. Measuring the extrinsics those procedures write is
[`docs/calibration-setup.md`](../../docs/calibration-setup.md).
