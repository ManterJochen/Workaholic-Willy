# The config tree of a cell (`src/config`)

A cell is described once, as a tree of YAML files under [`config/`](../../config/) that Pydantic
validates; this package loads that tree, answers questions about it and refuses what does not
validate. A misspelled key, a value out of bounds or a profile layer that does not exist is a
`ConfigError` naming the file, the line and the layer, never a setting that quietly never applied.
It opens no device and runs no calibration.

```python
from willy import load_tree

tree = load_tree()                   # the chain WILLY_PROFILE names, validated as one tree
print(tree)                          # OK with its layers, or the refusal with its file and line
if not tree.ok:
    raise SystemExit(tree.exit_code)

print(tree.explain("robot.safety.payload.mass_kg"))       # value, bounds, default, where it was set
weighed = tree.with_values({"robot.safety.payload.mass_kg": 1.2})   # validated in memory; nothing written
print(weighed.robot.safety.payload.mass_kg)               # the validated robot section
```

`load_tree("console_dummy")` loads the desk profile, `load_tree(None)` the base tree alone, and
`root=` another directory. `Robot`, `Cell`, `Camera`, `Locator`, `HandEyeCalibration` and
`SafetyPreflight` each take the loaded tree (`from_tree`), so values given in memory reach them.
[01_load_your_cell.py](../../examples/real_robot/01_load_your_cell.py) shows the same calls.

```bash
python -m src.config                                      # validate the tree WILLY_PROFILE names
python -m src.config --profile ur5e,eth2 --print          # another chain, and the validated tree as JSON
python -m src.config explain robot.gripper.max_width_mm   # one key: value, type, default, who set it
python -m src.config decisions --section robot.safety     # only what this cell decided
python -m src.config where fusion --tier all              # searches the schema, so it finds unwritten keys
```

Exit codes: `0` the tree validates or the question was answered (an unknown key is answered too), `1`
the tree does not load, `2` bad arguments. More in [docs/cli.md](../../docs/cli.md#the-config-tree).

## The nouns

| Noun | Built by | Verbs | Returns |
| --- | --- | --- | --- |
| `LoadedTree` | `load_tree(profile, root=)`, `ConfigTree.load()` | `explain(key)`, `decisions()`, `with_values(values)` | `KeyExplanation`, text, a `LoadedTree` |
| `ConfigTree` | `ConfigTree.from_directory(root=, profile=)` | `load()`, `write(items, connected=)` | `LoadedTree`, `WriteResult` |
| `AppConfig` | `load_config(data_dir, profile=)`, `tree.app_config` | | the frozen tree: `camera`, `models`, `robot`, `runtime` |
| one section | `load_robot_config()`, `load_robot_section`, `load_camera_section`, `load_speech_section`, `load_perception_section` | | one validated section |

A `LoadedTree` never raises on `load()`: it is the verdict, `ok` and `exit_code`, with the refusal
when it did not load. Nouns read it under `app_config`, `robot`, `root`, `profile`, `layers` and
`values`; `app_config` and `robot` raise the tree's `ConfigError` when it did not load.

`with_values` reads the files again under the same root and chain, sets the dotted keys on top of
every layer and runs the whole load: the named hand fills the numbers it supplies, the schema
validates and the registries are checked. `[n]` sets a list item (`camera.cameras.rigs[1].enabled`).
`model_copy` on a loaded section runs none of this, so naming another hand that way keeps the old
hand's widths without a word. A config is frozen: assigning to it raises.

## Profiles

A layer is the files `<name>.<layer>.yaml` beside each base `<name>.yaml`, deep-merged onto it.
`WILLY_PROFILE` (or `--profile`) takes a chain, applied left to right, so each dimension is stated
once: `ur5e,eth2` is a UR5e bench with two fixed cameras, `ur3e,hande` a UR3e with a Hand-E.

| Layer | What it changes |
| --- | --- |
| `ur5e` | a real UR5e bench: workspace box, home pose, bench fixture, planning world |
| `ur3e` | a UR3e, in simulation or on a bench, every position re-anchored for its shorter reach |
| `ur3`, `ur5`, `ur10`, `ur10e` | the arm model, its workspace box, its safe pose and its payload cap |
| `hande` | the Robotiq Hand-E as the cell's hand; chains after any arm layer |
| `eth2`, `tiltcam` | two fixed RGB-D cameras fused; two tilted eye to hand D435s |
| `sim`, `sim_camera_world` | the Isaac UR5e cell; chained on it, a planner that plans against the overhead camera |
| `ursim`, `ursim_ur3`, `ursim_curobo` | the UR driver aimed at a URSim container; chained on it, UR3e kinematics, or the planner on |
| `console_dummy` | a dummy arm and a dummy hand with a declared tool frame: the desk profile |
| `web` | a development rig of two USB webcams with a KUKA KR6 R900 robot section |
| `decision`, `rl_datagen` | the AUTO decision gate alone; the advanced grasping blocks and `rl.mode: rl_shadow` |

Read [`config/robot/robot.ur5e.yaml`](../../config/robot/robot.ur5e.yaml) before you describe your own
bench. A wrong value that fails closed, such as a workspace box that is too small, ships there as a
worked example, because it refuses a motion visibly. A wrong value that fails open does not ship at
all: a plausible tool frame or payload drives the arm into the bench and logs a success, so those are
commented blocks saying what to measure. Prefer a new layer over an edit to `robot.yaml` when you are
measuring something, so the measured block is the only thing that changed.

Merging: mappings merge key by key, scalars and lists replace, and a later layer wins. A `null` leaf
keeps the base value; the string `"__null__"` sets `None`. Any string may name an environment
variable, `${ROBOT_IP:-192.168.1.100}` with a default or `${MODEL_DIR}` without one, substituted
before parsing. `load_config()` is cached by directory and chain; call `reload_config()` after
editing a file.

## The tree on disk

| Path under `config/` | Holds |
| --- | --- |
| `camera/cam.yaml`, `camera/stereomatcher.yaml`, `camera/hand_eye.yaml` | the rigs, stereo matching, and the eye to hand and eye in hand workflows |
| `models/*.yaml` | detection, segmentation, the perception `pipeline`, speech; every top-level key merged |
| `robot/robot.yaml`, `robot/kpi_thresholds.yaml` | the robot section, and the KPI gate |
| `app/runtime.yaml` | runtime settings; schema defaults otherwise |
| `grippers/<model>.yaml`, `cameras/<model>.yaml` | the hand registry and the camera body registry |
| `grasping_presets/` | operator overlays the loader never reads |
| `all_keys/` | a reference tree with every key written out; `python -m src.config --data config/all_keys` validates it |

**Paths.** A relative path in any file of a tree is read against the tree's folder (the `root` it was
loaded from), wherever that folder is: not against the working directory, and not against the
subfolder of the file that writes it. `${WILLY_PROJECT_ROOT}/...` names the repository (or the folder
the `WILLY_PROJECT_ROOT` variable names), in path keys only (any other key refuses it, and it may be
written unquoted inside `{...}` or `[...]`), which is how the shipped tree names model weights and
committed artifacts so a copy of `config/` elsewhere still finds them; the shipped tree reaches its
cell data at the repository root with `../`. An absolute path is read as written and `~` is the home
folder. A schema default is written nowhere and names the repository's file. Every path field is typed
`ConfigPath` and the loader validates with the tree's folder as context, so the loaded config holds
absolute paths and no reader needs the rule; a model built in code with no tree keeps a relative path
as written. [`paths.py`](paths.py) states the rule and why it is applied at load.

`robot.gripper.model` names the hand (the registry holds `robotiq_2f85`, `robotiq_hande` and
`schunk_egu50`); the loader fills its widths and collision envelope for every key the chain leaves
unset and refuses a stated one that differs, while `robot.gripper.vendor` picks the driver.
`camera.cameras.rigs[<id>].body` declares a camera the arm carries, from the RealSense D435i, D435,
D405 and D415 bodies. A tree whose copy of a registry file differs from the repository's is refused,
naming both files.

`robot.safety` tightens the controller's own limits and never loosens them. Each block has its own
gate: `limits`, `joint_limits`, `ik_quality`, `motion_continuity`, `payload` and `self_collision`
(`enforce`), `dwell` (`require_steady_before_motion`) and `planning_world` (`enabled`). The drivers
build `SafetyPreflight` from it. The emergency stop is a hardware and controller function, and
nothing in this package can enable, disable or observe it.

`robot.look_joint_positions_deg` is where the camera looks from before each pick whose program names no
look: one list per look, in **degrees** as the pendant shows the joints, visited in order. It has no
radians twin, because a look is where the arm goes next and its unit must be said, and a program's own
looks override it. A wrist camera fuses the looks until its grasp is safe; a fixed camera's arm is moved
to them too, and its pick stops at the first look that finds something. The loader cannot tell radians
from small degrees, so the desk check (`real_cell --check`) flags a look that reads as radians.

`robot.natural_closing_axis` says how the hand and its camera naturally stand: a name (`"-y"`, `radial`,
...) or a quaternion `[x, y, z, w]`, of which only the heading of the tool +X counts. Every camera grasp
and push closes the way round nearer it and none is left out; `robot.tool_down` builds fixed poses along
it. It is read by `src.geometry.closing_axis`, so validating a tree that names it loads no grasping
package. Unset by default and in every shipped profile.

`robot.named_poses` holds the poses taught by hand in the operator console, each under its **name** (an ASCII
identifier of at most 32 characters, no keyword, never `home`, no word YAML reads as true, false or null, no
`__dunder__`) with its joints in degrees under the looks' rules, a **label** (free text of at most 40 characters,
no newline, no `" #"`), when it was taught, its screen (`clear` or `band`) and a note. A pose answers to its name
and its label, and two poses sharing a word, whatever the case, are refused at load. `robot.default_place_pose`
names the one a task places at when its command names no target, or `null`; a name that is no pose of the tree is
refused at load. Layers merge a pose key by key, so a task screens every pose again before it moves.

`robot.ur.brake_on_halt` decides what the console's "halt now" does to a move already in flight: off, as shipped,
every move is sent as it always was and the move in flight runs to its end; on, it is braked under control. Either
way a halt latches the arm and is not an emergency stop.

`config/grasping_presets/` holds `easy` (a single pick at the least risk) and `dense_clutter` (bin
picking with a bounded rescan recovery). `verification_heavy` set the `closed_loop` mode; it was
deleted on 2026-09-29 with that mode, whose two-scan refinement was the only path that ran the check
after the grasp. `apply_preset(robot_dict, name)` from `src.robot.grasping.replay` merges one onto a
robot section as a plain dict without validation; `validate_preset` in its `presets` module checks
that the merge validates, and refuses a preset that names a removed mode, a removed recovery action
(`next_viewpoint`, merged into `rescan` on 2026-09-29) or a removed block (`verification`,
`dense_recovery`).

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| an unknown key | a key the schema does not have | the message names the file, the line, the layer and the nearest key |
| a key, a grasp mode or a recovery action removed on purpose | a key in `REMOVED_KEYS` (the `verification` and `dense_recovery` blocks among them), `closed_loop` / `dense_autonomous` (`REMOVED_GRASP_MODES`), or `next_viewpoint` in `recovery.allowed_actions` or `recovery.per_action_budget` (`REMOVED_RECOVERY_ACTIONS`, all in `schema/_removed.py`) | the message says `removed on purpose` and what to write instead; a preset's validation and `explain` say the same sentence |
| an unknown layer | a chain names a layer no `*.<layer>.yaml` carries | fix the name; it is checked per layer |
| an unset variable | `${VAR}` with no default and no such environment variable | set it, or give a default |
| a hand or camera the registry does not hold | `robot.gripper.model`, or a rig body, names no registry file | name one the registry holds |
| a switch that would do nothing | `occlusion.hard_reject_enabled: true` (`UNWIRED_SWITCHES`) | leave it off |
| a perception stack that would ground the wrong thing | `kind: closed_set` with the router on, or the router on a backend other than `vlm` | use the combination the message names |
| a fused camera with no rig | `grasping.fusion.cameras` names an id `camera.cameras.rigs` does not have | fix the id |
| a look that is no look | `robot.look_joint_positions_deg` names no look, a look with no joint or a value that is not finite, looks of different lengths, a length unlike the home's, or a joint past a full turn (360) | one list of joint degrees per look, one value per joint |
| a closing direction that names none | `robot.natural_closing_axis` an unknown name, `""`, a quaternion that is none, a 3-vector, a number, or a quaternion whose tool +X stands within 10 degrees of the vertical | a name `Pose.tool_down` takes, or a taught pose's `[x, y, z, w]` |
| a recovery that could not act | `container_agitate` with no `recovery.fixture` (the push, `nudge_target`, needs none); `container_agitate` while `grasping.support.container` declares no interior box; `recovery.fixture.max_nudge_mm` or `push_distance_mm` under 10 or over 50 mm, or the distance above the ceiling | declare what the sentence names; an old `max_nudge_mm: 5` is refused on purpose: delete it to take 50 mm, or write 10 to 50 with a `push_distance_mm` no longer than it |
| a write outside the allowlist | `ConfigTree.write` of a limit, a threshold or a safety toggle | edit the profile by hand |
| a taught pose written as a value (`NOT_WRITABLE`) | any key under `robot.named_poses`, or `robot.default_place_pose`, through `ConfigTree.write` or the console's `PATCH` | teach it in the console; edit the YAML to rename or delete one |
| a pose with nowhere of its own to go (`NO_LAYER`) | the pose door under a chain with no layer, whose pose would land in the shared `robot.yaml`, or whose last layer git tracks or does not ignore, or a git that cannot be asked | end the chain in the cell's own layer: create `robot/robot.cell.yaml` (one comment line), which `config/**/*.cell.yaml` ignores, or list it in `.git/info/exclude` |
| a pose name or label that is none (`INVALID_NAME`, `INVALID_LABEL`) | the pose door, with nothing written: a name that is no identifier, a keyword, `home`, a YAML true/false/null word or a dunder; a label that is empty, too long, or holds a newline or `" #"` | another name or label |
| a pose the loader refuses (`INVALID_VALUE`) | the written tree does not validate, a default place that names no pose among them | every file is restored byte for byte; the message is the loader's |

`ConfigTree.write` takes the bench measurements and site facts only: the payload mass and centre of
gravity, the tool frame's `source`, `offset_mm` and `rotation_quat_xyzw`, a rig's `serial_number` and
`enabled`, `camera.cameras.primary_rig_id`, and the controller addresses `robot.ur.ip` and
`robot.kuka.controller_ip`. It rewrites one line in place so comments survive, validates the group as
one transaction and restores every file if the loader rejects the result. The primary rig and the
controller addresses are refused while anything is connected.

**A taught pose has its own door.** `POSE_KEYS` (the five keys of a pose under `robot.named_poses.<name>`, and
`robot.default_place_pose`) are never in `WRITABLE`. `set_named_pose(name, joints_deg=, label=, screen=,
taught_at=, ...)` and `set_default_place_pose(name)` in [`edit.py`](edit.py), as `ConfigTree.write_named_pose`
and `write_default_place_pose`, write them all or nothing into the robot overlay of the chain's **last** layer
(`pose_target_file`), and `pose_layer_refusal` (also `ConfigTree.pose_layer_refusal`) says beforehand why that
layer may not take one. A pose screened anything but `clear` or `band` raises `ValueError`: an unscreened or
refused pose is never written. Clearing the default place writes `"__null__"`, the loader's reset: a `null` in an
overlay would keep a lower layer's default. A pose taught again is rewritten in place, every comment kept; a
joint list written by hand as a block is not rewritten, and the loader's refusal comes back with the file
untouched.

## Files

| File | Holds |
| --- | --- |
| `tree.py` | `ConfigTree`, `LoadedTree`, `load_tree`, `default_data_dir` |
| `loader.py` | `load_config`, `load_robot_config`, the section loaders, `reload_config`, `ConfigError` |
| `paths.py` | the path rule: `ConfigPath`, `${WILLY_PROJECT_ROOT}`, and `path_note` for a refusal that cannot find a file |
| `explain.py`, `_provenance.py`, `_schema_index.py`, `_tiers.py` | `explain`, `where` and `decisions`: value, type, default, tier and the layer that set it |
| `edit.py` | the allowlisted bench writes, and the pose door: `set_named_pose`, `set_default_place_pose`, `pose_layer_refusal` |
| `grippers.py`, `cameras.py`, `_registry.py`, `hand_numbers.py` | the hand and camera registries, and the robot keys a named hand fills |
| `_merge.py` | the deep merge the layers are built on |
| `schema/` | the `StrictModel` schemas, `robot/` split per vendor and subsystem |
| `__main__.py` | `python -m src.config` |

## Details

- Guide: [configuration](../../docs/guide/01-configuration.md); perception stacks: [models](../../docs/guide/02-models.md)
- Every `robot.grasping` block and the mode it fires in: [grasping-config-reference.md](../../docs/grasping-config-reference.md)
- Changing config on a live cell: the runbooks under [`docs/runbooks/`](../../docs/runbooks/), first
  [real_cell_first_pick.md](../../docs/runbooks/real_cell_first_pick.md) and [cell_bringup.md](../../docs/runbooks/cell_bringup.md);
  measuring extrinsics: [calibration-setup.md](../../docs/calibration-setup.md)
- The KPI gate `kpi_thresholds.yaml` feeds is a synthetic contract self-check, not a grasp quality measure
- Tests: `tests/test_config_tree.py`, `tests/test_config_tree_with_values.py`, `tests/test_config_cli.py`, `tests/test_config_explain.py`, `tests/test_config_edit.py`, and `tests/test_a_removed_key_is_answered_wherever_it_is_asked.py` for every `REMOVED_KEYS` entry through `explain` and a preset; the poses in `tests/test_named_poses_load_and_refuse.py`, `tests/test_a_pose_is_written_only_through_its_own_door.py` and `tests/test_the_cells_own_layer_is_kept_out_of_git.py`
