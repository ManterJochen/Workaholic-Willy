# Building the configuration

This guide gets you a validated config tree for your own cell: a different arm, gripper, camera rig
or workspace box than the one in the repository. It needs the repository and Python, and no
hardware: the config layer imports no camera, no robot SDK and no simulator.

```bash
python -m src.config                                  # validate: exit 0 means the tree loads
python -m src.config explain robot.gripper.model      # one key: value, default, file and line
python -m src.config decisions --section robot        # only what this cell decided
```

```python
from willy import load_tree

tree = load_tree()                          # the chain WILLY_PROFILE names; load_tree(None) is the base tree
print(tree)                                 # the verdict line, or the refusal with its file and line
print(tree.explain("robot.gripper.model"))  # value, type, default, and the file and line that set it
```

At the end, your tree or profile layer passes `python -m src.config` with exit code 0, and you can
justify every value in it that differs from the default. A green tree is not a running pick:
[section 9](#9-what-a-config-actually-reaches) says what a config reaches.

This is the walkthrough. The reference is [`src/config/README.md`](../../src/config/README.md): the
shipped layers, the merge rules, the safety block, the guided write path and the rules for adding a
field. Where the two disagree, that one wins. Next: [02-models.md](02-models.md), then
[03](03-calibration.md), [04](04-robot-and-safety.md), [05](05-pick-loop.md) and
[06](06-grippers.md).

## Prerequisites

Python 3.11, the repository root as your working directory, and the repository installed:
`pip install -r requirements.txt`, then `pip install -e . --no-deps` ([02 section 2](02-models.md)).
Of that set this guide needs only `pydantic` and `PyYAML`.

---

## 1. The mental model

**One object.** The loader reads a short, closed list of YAML files. It substitutes `${ENV}` in the
raw text, deep-merges each file's profile overlays on top, resolves `"__null__"` to `None`, fills a
named hand's numbers from the gripper registry, and validates the result as one `AppConfig`
([`loader.py`](../../src/config/loader.py)).

**Strict.** Every schema class is a `StrictModel` with `extra="forbid"` and `frozen=True`. An unknown
key is a load error, never a line that is quietly ignored, and assigning to the config raises. A
misspelled threshold cannot become a setting that never applied.

**A file states what this cell decided.** Everything a file leaves out is the schema default, in
force. That is why the shipped `robot.yaml` leaves most `grasping:` and `rl:` blocks out: the
features exist, and `extra="forbid"` still accepts every one of their keys.

**So ask the tree rather than read it.** Most schema fields are written in no YAML at all. Count them
on your checkout:

```bash
python -c "from pathlib import Path; from src.config._schema_index import schema_index; from src.config._provenance import index_origins; i=schema_index(); o=index_origins(Path('config'), []); print(len(i),'schema fields;',len([k for k in i if k in o]),'written in some YAML')"
```

Reading the files can never show you the rest. [Section 4](#4-how-to-ask-the-config-questions) is how
you ask.

**Loads are cached** by data directory and profile chain, through `load_tree()` and `load_config()`
alike. Editing a YAML in a running process changes nothing until you call `reload_config()` from
`src.config`.

---

## 2. What the loader reads

The list is closed. Nothing is discovered except `models/*.yaml`. The shipped tree is
[`config/`](../../config/) at the repository root; `--data <dir>` or `load_tree(root=<dir>)` reads
another one.

| File, relative to the tree | Required | Top-level key | Lands at |
|---|---|---|---|
| `camera/cam.yaml` | yes | `cameras:` | `camera.cameras` |
| `camera/stereomatcher.yaml` | yes | `stereomatcher:` | `camera.stereomatcher` |
| `camera/hand_eye.yaml` | no | `hand_eye:` | `camera.hand_eye` |
| `models/*.yaml` | the directory, non-empty | each file's top-level keys | `models.*` |
| `robot/robot.yaml` | no | `robot:` | `robot` |
| `app/runtime.yaml` | no | `runtime:` | `runtime` |

Without `robot:`, `cfg.robot` is `None`, which is a legal camera-only cell. The required fields sit
under `camera` and `models`; in the robot section they sit only inside optional blocks. Ask the schema
rather than this page:

```bash
python -c "from src.config._schema_index import schema_index; print(sorted(k for k,v in schema_index().items() if v.required))"
```

Every `models/*.yaml` that is not an overlay is read, and the files' top-level keys merge into one
mapping. The same top-level key in two base files is a `ConfigError`: override a model with a profile
overlay, not a second base file. Deleting the file that carries a required key is a load failure, not
a leaner config.

**The other directories under `config/`:**

| Directory | Read by | What to know |
|---|---|---|
| `grippers/<model>.yaml` | the loader, for the hand `robot.gripper.model` names | a tree that names a hand needs this directory |
| `cameras/<model>.yaml` | the tree check, and a cell that reads geometry | holds the housing a rig's `body` names; a tree without it reads the repository's |
| `grasping_presets/*.yaml` | `apply_preset`, outside Pydantic | [section 8](#8-the-grasping-presets-a-schema-bypass) |
| `robot/kpi_thresholds.yaml` | the soak and KPI gate | not part of `AppConfig` |
| `robot/templates/kuka/` | nothing here | controller-side files you copy to a KUKA by hand |
| `all_keys/` | nothing by default | the reference tree, below |

The gripper registry gives a named hand its widths and its collision envelope, and the
self-collision guard, the planner and the deep calculator take the hand from it. Both registries are
the repository's. `python -m src.config` refuses a malformed hand or camera file, and it refuses a
tree whose copy of a named hand or camera differs from the repository's, naming both files. So a tree
of your own carries an unchanged copy of `grippers/`. A hand this repository does not ship joins it
through [your_own_gripper.md](../runbooks/your_own_gripper.md).

**`config/all_keys/` is a reference tree.** It writes out every key the schema accepts, with what it
does, its default and its legal values. Nothing loads it by default, and it validates on its own:

```bash
python -m src.config --data config/all_keys
```

Read it to find out what exists. Do not copy it over your tree: a file that restates every default
turns each of those lines into something a reader has to check against the schema.

**One input is not a file in the tree.** If `WILLY_ADAPTATION_OVERLAY` is set, the loader deep-merges
the `robot:` section of the YAML at that path before validation. It is unset by default. With it set,
`explain` and `decisions` attribute an overlaid value to the base file, so unset it before you trust
an explanation.

**A relative path in the tree is read against the tree's folder.** That is the directory the tree was
loaded from, `config/` for the shipped tree and whatever `--data` or `load_tree(root=...)` names for
yours. It is not the directory the program was started in, and not the subfolder of the file that
writes the path: `camera/cam.yaml` writing `calibration/eih_wrist.json` names
`<tree folder>/calibration/eih_wrist.json`. So a cell kept in its own folder, outside the repository,
keeps its calibration beside its config and names it relatively:

```yaml
# D:/cells/line3/camera/cam.yaml, loaded with load_tree(root="D:/cells/line3")
        extrinsics:
          mounting_mode: eye_in_hand
          artifact_path: calibration/eih_wrist.json    # D:/cells/line3/calibration/eih_wrist.json
```

| Written in a file | Read as |
|---|---|
| `calibration/eih_wrist.json` | `<tree folder>/calibration/eih_wrist.json` |
| `../calibration/eih_wrist.json` | beside the tree folder; from the shipped `config/` that is the repository root |
| `${WILLY_PROJECT_ROOT}/assets/models/hf/...` | in the repository, or in the folder the `WILLY_PROJECT_ROOT` variable names |
| `D:/cells/shared/eih_wrist.json`, `~/cal/eih_wrist.json` | as written; `~` is your home folder |
| nothing, the schema default | the repository's file (every path default is `${WILLY_PROJECT_ROOT}/...`) or no file |

`${WILLY_PROJECT_ROOT}` is for what the repository holds rather than the cell: the model weights
`scripts/model_weights/fetch.py` writes, the committed success model and ranker, the RL baselines. The
shipped tree names those with it, so a copy of `config/` moved out of the repository still finds them,
and names its cell data (`calibration/...`) with `../`, which from `config/` is the repository root.
When you copy the tree out, rewrite those `../` paths to where the cell's files are now, relative to
the new folder or absolute.

The loaded config holds the absolute path, so `explain <key>` prints the file that will be opened
above the line that wrote it. `${VAR}` substitution runs first, so a relative path a variable supplies
is read against the tree's folder too: give such a variable an absolute path. A value given to
`with_values` is read against the same folder as the same value in a file. A file that is not there is
refused with the path as written, the path it was read as and this rule. The rule in full, and why it
is applied once at load rather than by each reader: [`src/config/paths.py`](../../src/config/paths.py).

---

## 3. Profiles, overlays and the four worked cells

`WILLY_PROFILE` takes a comma-separated chain of layer names, applied left to right. For a base file
`foo.yaml`, layer `L` contributes `foo.L.yaml` from the same directory. A layer is one dimension of a
cell, not a whole alternative configuration, and layers compose.

```bash
python -m src.config --profile sim,ur3e,tiltcam                # the flag, which is not sticky
python -c "from src.config.loader import available_profiles; print(available_profiles())"
```

From Python, `load_tree("sim,ur3e,tiltcam")` names a chain, `load_tree()` takes the one
`WILLY_PROFILE` names, and `load_tree(None)` is the base tree whatever the variable says.

**Four shipped layers are worked cell examples.** Reading them is the fastest way to learn what a real
cell has to state.

| Chain | The cell it describes |
|---|---|
| `ur5e` | a real UR5e on a bench: workspace box, home and safe pose, bench fixture, planning world, four measurements to take |
| `ur3e` | the shorter arm's reach-anchored geometry and kinematics, under `sim` or on its own against a real UR |
| `sim` | the simulated cell, with the camera inventory and the model overlays the simulator needs |
| `ur5e,eth2` | the same bench with two fixed RGB-D cameras, fused; `eth2` alone is the fusion half and names no arm |

`console_dummy` is the desk profile: a dummy arm and a dummy hand that command nothing. The two desk
programs in [`examples/simulation/`](../../examples/README.md) run on it.

[`config/robot/robot.ur5e.yaml`](../../config/robot/robot.ur5e.yaml) draws a line worth adopting for
your own bench. A wrong value that fails closed ships as a worked example with its assumption stated:
a workspace box that is too small refuses a motion visibly, and an operator widens it. A wrong value
that fails open does not ship at all: a tool frame or a payload that is merely plausible drives the arm
into the bench and logs a success. So those are commented blocks, and the refusals in
`run_config_preflight` and `URRobotArm.connect` stay armed.

**Two things decide the two-camera cell.** Both are documented in place in
[`config/robot/robot.eth2.yaml`](../../config/robot/robot.eth2.yaml) and
[`config/camera/cam.eth2.yaml`](../../config/camera/cam.eth2.yaml).

- **`fusion.cameras` names every camera the cell fuses, the primary included**, keyed by rig id, and
  an entry holds `enabled` only. The code leaves the primary, `camera.cameras.primary_rig_id`, out of
  the cameras a pick waits for, so listing it costs nothing. A fused id that names no rig in
  `camera.cameras.rigs` is refused when the config loads.
- **Each camera's calibration is declared on its rig**, as `camera.cameras.rigs[<id>].extrinsics`. The
  pick service reads the primary's CAMERA to BASE transform there, and the second camera's resolver
  reads its own. The calibration command prints that block after each camera, and a fused camera whose
  rig declares none is refused when the cell is built.

**Prefer the flag, because the variable is sticky and silent.** `WILLY_PROFILE` lives for the whole
shell session and `load_tree()` honours it everywhere. A forgotten `sim` changes safety thresholds and
calibration board parameters under you, and nothing in the files warns you. Clear it with
`Remove-Item Env:\WILLY_PROFILE`, or `unset WILLY_PROFILE` on POSIX. The validate banner names the
chain it loaded, which is the cheapest check.

**Every layer must exist.** Each layer in the chain must contribute at least one `*.<layer>.yaml`
under the tree, or the load fails naming the layer. A typo mid-chain would otherwise merge as nothing
and leave the cell running another robot's geometry.

**A chain that composes is not always meaningful.** The cross-field validators in section 5 refuse
chains that would evaluate one robot against another robot's geometry. `sim,web` is one: it lands
`kinematics_model: ur5e` on a `vendor: kuka` cell, and that key selects a Universal Robots DH table.
Validate any chain in one command rather than trusting a document about it.

**Merge rules.** The full treatment is in [`src/config/README.md`](../../src/config/README.md). Three
rules to remember:

- A `null` overlay leaf keeps the base value, so a partial overlay cannot wipe a field by accident.
- Scalars and lists replace. There is no list-append merge.
- `"__null__"` is the only way to unset, resolved to `None` after the merge, because of the first
  rule. `config/models/object.sim.yaml` uses it, and [02 section 6](02-models.md) says why it matters.

**`${ENV}`.** `${VAR}` and `${VAR:-default}` are substituted in the raw text before the YAML is parsed,
so the result must still be valid YAML. Quote anything that could contain a `:` or a `#`. An unset
variable with no default is an error that names the file. `${WILLY_PROJECT_ROOT}` is the one name
left in place: it is the path anchor of [section 2](#2-what-the-loader-reads), expanded in path keys
only. Written in any other key it is refused with the key's name rather than carried along as text.
It works quoted or unquoted, in block style and inside `{...}` and `[...]` alike.

---

## 4. How to ask the config questions

Learn this section before you edit anything. On `explain` and `decisions`, the flags `--data` and
`--profile` work on either side of the subcommand, so `config --profile sim explain KEY` and
`config explain KEY --profile sim` agree. `where` refuses both, because it reads the schema and no
tree.

### 4.1 Validate

```bash
python -m src.config
python -m src.config --profile sim,ur3e
python -m src.config --data /path/to/mycell --profile night
```

Exit codes: `0` valid; `1` a `ConfigError` (a missing file, a parse error, an unresolved environment
variable, a schema failure, a registry that refuses); `2` bad command-line arguments. It boots nothing,
so put it in CI and in your shell history. The banner names the chain that was loaded, so a chain that
came from the environment shows up with no flag given.

**Only the bare command is a usable gate.** `explain` and `decisions` exit `0` once the tree loads,
and an unknown key is reported in the text, not in the exit code. `where` exits `0` whatever it
finds. `--print` dumps the validated `AppConfig` as JSON: good for diffing two chains, useless for
understanding a cell, because it cannot tell a decision from a restated default. `--print` on a
subcommand is refused, naming the command that accepts it.

### 4.2 `explain <dotted.key>`: what is this, and where did it come from

The most useful command here.

```bash
python -m src.config explain robot.safety.self_collision.planner_margin_mm --profile sim,ur3e
```

For one key it prints the value, tier, type, constraints and schema default; the documentation from
the schema; the winning file, line and layer, with the full override chain; and the YAML comment above
the winning line. That comment is the reason a value is what it is, and a YAML parser throws it away,
which is why this beats grep.

On a key no YAML mentions, it says so and still gives the type, the default and the meaning. That is
most of the schema: someone reading `robot.yaml` cannot otherwise discover `robot.ur.model`, the field
that decides which robot they are configuring. On an unknown key it says so and offers near misses
from the schema, which is the reliable way to find a typo. Ask with the snake_case attribute name and
an alias resolves: `camera.stereomatcher.num_disparities` finds the YAML's `numDisparities`.

### 4.3 `where <substring>`: what am I allowed to set

`where` searches the schema, not the YAML. Grepping the files for "gripper" misses the whole suction
end-effector, because no shipped YAML writes it; `where` finds it.

```bash
python -m src.config where gripper
python -m src.config where grasping --limit 500
```

`where` lists at most 40 keys per tier unless you pass `--limit`. The header count is always complete,
and the footer says how many keys it left out, so raise `--limit` or narrow the substring. `where`
loads no tree, so it still works when the config is broken. `--tier T` filters the listing and
`--tier all` shows every tier. A tier is a display filter and nothing else: every field stays
settable.

### 4.4 `decisions`: what did this cell actually decide

`decisions` prints only the values that differ from the schema default, each with the file, line and
layer that won, grouped by tier.

```bash
python -m src.config decisions
python -m src.config decisions --section robot.safety --profile sim,ur3e
```

Scoped, it is short enough to read line by line, and that is how to use it. Read which way each entry
goes. On the simulator chain some entries loosen a guard a real arm keeps tight, because the simulator
has no controller-side safety to lean on; others tighten one. `kinematics_model` and
`kinematics_base_yaw_deg` do neither: they state which robot and which base frame the guard checks
against.

### 4.5 From Python

The same questions, through the one import a program uses:

```python
from willy import load_tree

tree = load_tree("sim,ur3e")                # load_tree(None, root="C:/cells/mycell") for a tree of your own
print(tree)
if not tree.ok:
    raise SystemExit(tree.exit_code)

print(tree.explain("robot.safety.self_collision.planner_margin_mm"))
print(tree.decisions(section="robot.safety"))
cfg = tree.app_config                       # the frozen AppConfig; assigning to it raises

# A change tried in memory: loaded and validated as a file would be, and nothing is written.
heavier = tree.with_values({"robot.safety.payload.mass_kg": 1.2})
print(heavier.explain("robot.safety.payload.mass_kg"))
```

A tree that does not load comes back with `ok` false and its refusal; it does not raise. `app_config`
and `robot` on such a tree raise `ConfigError` with the same text. `Cell`, `Robot`, `Camera`,
`Locator`, `HandEyeCalibration` and `SafetyPreflight` each take the loaded tree through `from_tree`.
[`examples/real_robot/01_load_your_cell.py`](../../examples/real_robot/01_load_your_cell.py) is this
section as a program.

`src.config` keeps the lower-level loaders. `load_config()` returns the cached `AppConfig` alone.
`load_robot_section`, `load_camera_section`, `load_speech_section` and `load_perception_section` each
validate one section through the same chain, uncached, so a broken camera file does not refuse an arm.
A caller that combines the camera and robot sections runs the one rule that spans them,
`src.config.schema.camera_calibration_conflict`.

For a simulated cell, `load_sim_config` in [`src/willy_sim/config.py`](../../src/willy_sim/README.md)
builds the `sim` chain, forces it for the load, and restores the previous `WILLY_PROFILE` afterwards.

**Writing a value from a program** goes through `ConfigTree.write(items, connected=...)` (the loaded
tree's `tree`), backed by [`src/config/edit.py`](../../src/config/edit.py). It writes a short list of
keys only: measurements and site facts such as the payload, the tool frame, a rig's serial number and
the controller addresses, never policy. A group is written as one transaction, comments survive, and
every touched file is restored byte for byte if the loader rejects the result. The two controller
addresses refuse a write while the cell is connected. Ask it what it accepts:

```bash
python -c "from src.config.edit import WRITABLE; print([w.path for w in WRITABLE])"
```

---

## 5. Worked example: a config for a new cell

Target: a UR3e on a bench, a Robotiq 2F-85, a small workspace box, one stereo webcam pair.

### Step 0: new profile layer, or new tree

If the new cell shares most of an existing one (the same optics and models, a different arm), add a
**profile layer**: one `<base>.<layer>.yaml` next to each base file it changes. Each measured value
stays stated once, and this is the shipped pattern. If it shares nothing, make a **new tree** and point
`--data` or `load_tree(root=...)` at it.

A tree given to a simulator runner's `--data-dir` loads under the `sim` profile, so it must carry
`*.sim.yaml` overlays or the load fails. Without them the simulated cell would run the production fp16
detectors, which lose small-object recall.

### Step 1: the required half, and a first green run

Copy and edit rather than write from nothing: the required fields live in these files. The copy of
`grippers/` stays unchanged, because the hand you name in step 2 is read from it.

```powershell
$Cell = "C:/cells/mycell"
New-Item -ItemType Directory -Force "$Cell/camera", "$Cell/models", "$Cell/robot" | Out-Null
Copy-Item config/camera/cam.yaml, config/camera/stereomatcher.yaml "$Cell/camera/"
Get-ChildItem config/models/*.yaml | Where-Object Name -notlike "*.sim.yaml" | Copy-Item -Destination "$Cell/models/"
Copy-Item -Recurse config/grippers "$Cell/"
python -m src.config --data $Cell
```

```bash
CELL=~/cells/mycell
mkdir -p $CELL/camera $CELL/models $CELL/robot
cp config/camera/cam.yaml config/camera/stereomatcher.yaml $CELL/camera/
cp $(ls config/models/*.yaml | grep -v '\.sim\.yaml$') $CELL/models/
cp -r config/grippers $CELL/
python -m src.config --data $CELL
```

Then edit `camera/cam.yaml` for the hardware you have. Its `base_dir: ../calibration/<rig>` lines were
written for `config/`, where `..` is the repository; in the copy they point beside `$Cell`, so write
them `calibration/<rig>` to keep the cell's calibration in its own folder ([section 2](#2-what-the-loader-reads)).
The models' `${WILLY_PROJECT_ROOT}/...` paths stay as they are. Rigs belong to
[03-calibration.md](03-calibration.md), models to [02-models.md](02-models.md). For a simulated cell,
copy the `*.sim.yaml` overlays too. `robot:` is optional, so a green run here proves the perception
half is sound before the robot half can confuse the diagnosis.

### Step 2: the robot half

Write only the site facts, and use `explain` for every number you are unsure of. A minimal
`robot/robot.yaml` states the vendor, the model and the address. Give the address a fallback, because
a config that loads only when the environment is right fails at the worst moment.

```yaml
robot:
  vendor: ur
  ur:
    model: "ur3e"
    ip: "${WILLY_UR_IP:-192.168.1.100}"
  gripper:
    model: robotiq_2f85
```

**The workspace box.** This tree writes no `workspace_limits`, so the schema default box is in force.
It is a symmetric placeholder with corners a UR3e cannot reach; see it with
`python -m src.config explain robot.workspace_limits.z_max --data $CELL`. Write your own box, and put
`safe_pose` inside it with margin: the workspace guard also subtracts
`safety.limits.workspace_margin_mm` from every face.

**The hand.** `robot.gripper.model` names a file in the gripper registry, and the hand brings its
widths and its collision envelope from it:
`python -m src.config explain robot.gripper.max_width_mm --data $CELL` names that file and field
instead of a layer. A cell that names no hand keeps the default of 85 mm,
the Robotiq 2F-85. The Robotiq driver anchors its count map on that value, so it must be the real
physical open width, not a policy ceiling. `python -m src.config where gripper` lists the rest.

**Suction.** A vacuum end-effector uses `robot.gripper.vacuum.*`, read only when `robot.gripper.vendor`
is `"vacuum"`. Under any other vendor that block validates green and is ignored, and every field in it
is a number somebody has to measure.

**Safety bounds stay written even at their default**, so an auditor can read the guard envelope out of
the file. The guards themselves belong to [04-robot-and-safety.md](04-robot-and-safety.md).

Validators on `RobotConfig` stop you when one robot would be evaluated against another robot's
geometry. When `safety.self_collision.kinematics_model` is set, it must equal `sim.robot_model` on a
simulated cell and `ur.model` on a UR cell, and on any other vendor it must stay unset: the key picks a
UR DH table, and only UR tables are bundled. A UR5e's upper-arm and forearm links are roughly 425 mm and 392 mm against
a UR3e's 244 mm and 213 mm. A further validator keeps `safe_pose` inside the workspace box, because the
workspace guard would otherwise refuse the retreat itself.

### Step 3: audit the decisions, which is the acceptance check

Run `python -m src.config decisions --data $CELL --section robot` and read it line by line. An entry
you did not intend is a bug in the YAML. An entry you intended that is missing means you wrote a value
equal to the default, which is correct and not listed: `ip: 192.168.1.100` does not appear, because
that is the schema default. Lines from a profile layer carry their layer name and file.

---

## 6. Enabling a default-off feature

**Find the gate first.** Most `robot.grasping.*` blocks carry `enabled: false`, but not all, and
writing `enabled: true` under a block that has no such field fails the load. `watchdog` is gated by
`mode`, `occlusion` by two separate flags, `gripper_geometry` by nothing, and `robot.rl` by `mode`. So
ask first: `explain robot.grasping.watchdog.enabled` answers that the key is not known, and
`where watchdog` shows you the real gate.

**One gate is a string.** `robot.grasping.calculator` is `geometric` or `deep`, and it selects the grasp
generator the cell runs. Every calculator the cell builds goes through `build_calculator`
([`autonomous_grasp/`](../../src/robot/execution/autonomous_grasp/README.md)), so the key is live. It
fails closed: `deep` with no readable artifact raises rather than falling back, because a cell that
asked for the learned generator and got the analytic one would report the analytic one's results under
the learned one's name. **No trained weights ship in this repository**; you train on your own cell's
data ([`src/robot/grasping/deep/README.md`](../../src/robot/grasping/deep/README.md)).

The workflow is three steps, and the file edit is the last one:

1. **Find** the key: `python -m src.config where fusion`.
2. **Read** it: `python -m src.config explain robot.grasping.fusion.enabled`.
3. **Write** the block with only the keys you change, and put the reason in a comment directly above
   the line. `explain` shows that comment to the next person who asks.

Then run the bare command, `explain`, and `decisions --section robot.grasping` again. If the block lands
under the wrong parent, `extra="forbid"` catches it, and the dotted path in the message says where it
went: `robot.rl.fusion` means you wrote it under `rl:`, not under `grasping:`.

**Two shapes are refused before they can mislead you.** A switch in
`RobotGraspingConfig.UNWIRED_SWITCHES` would land in the cell's telemetry while nothing reads it, so
setting one is refused at load; it holds `occlusion.hard_reject_enabled` until the occlusion score is
trusted. And `robot.grasping.support.container.wall_collision_enabled` without both `interior_min_mm`
and `interior_max_mm` is refused at load, rather than running a cell that believes a bin protects it
when it cannot locate the walls.

Several blocks can fire only in some grasp modes, and several ship their operative weight at `0.0`,
so `enabled: true` alone does nothing. Read
[`docs/grasping-config-reference.md`](../grasping-config-reference.md) before you enable or measure
any of them.

---

## 7. Reading a validation failure

Every load failure is a `ConfigError`. For each offending key it prints the dotted path, the file and
line with the layer where there is one, the pydantic message, and sometimes a suggestion. The location
line tells you which of three shapes you have:

- **A file and line** is a field error: go to that line.
- **`(not written in any YAML: a default or a cross-field rule)`** is a cross-field validator. It fires
  on the whole model, so there is no single line, and the message names both keys and the values it
  compared.
- **A file path with no dotted key** failed before validation ran: a missing file, a YAML syntax error,
  or an unresolved `${VAR}`.

**The suggestion is built from keys written in some YAML** that share the typo's parent, not from the
schema. So the ordinary typo, a misspelling of the only spelling of a key, usually gets no suggestion.
Ask the schema instead, with `python -m src.config explain <the key you meant>`.

The failures that need a word:

- **A duplicate model key**: two base files declare the same top-level key. Use a profile overlay
  instead of a second base file.
- **A profile-layer error**: the tree has no `*.<layer>.yaml` for that layer.
- **`Extra inputs are not permitted`**: a typo, a wrong indentation level, or a renamed key.
- **You edited a YAML and nothing changed**: the load is cached. Call `reload_config()`.

---

## 8. The grasping presets, a schema bypass

The files in [`config/grasping_presets/`](../../config/grasping_presets/) are not part of `AppConfig`,
not read by the loader, and not covered when `python -m src.config` validates green. Each is a partial
`grasping:` block that `apply_preset` in
[`src/robot/grasping/replay/`](../../src/robot/grasping/replay/README.md) merges onto an already-loaded
`robot` mapping. They are not part of your tree either: the presets directory is the repository's, not
`--data`'s, so a YAML you drop into your own `grasping_presets/` is ignored. You can still apply a
shipped preset to your own cell, because `apply_preset(base, name)` takes any base mapping.

The merge runs outside Pydantic, so a typo in a preset survives and does nothing. The check is opt-in.
Run it whenever you add or edit a preset:

```bash
python -c "from src.robot.grasping.replay.presets import validate_all_presets; print(validate_all_presets())"
```

What each preset holds, and what its `default_mode` does to a `pick()`, is in
[05-pick-loop.md](05-pick-loop.md).

---

## 9. What a config actually reaches

**The real-cell path reads the whole tree.** `Cell.from_tree(load_tree())` builds a cell from it and
runs the steps in a fixed order: preflight, build, safety attestation, then connect (lock, arm, then
gripper) and pick. [`python -m src.robot.execution.real_cell`](../../src/robot/execution/real_cell/README.md)
and the [examples](../../examples/README.md) go through `Cell`, and the operator console calls the same
builders, `build_real_cell` and `build_rehearsal_cell`. `Cell.rehearsal(...)` and `--rehearse` run the
whole path on a dummy arm and a synthetic scene. Under it, `AutonomousGraspService.from_robot_config`
builds the pick service from `robot:`. No part of this path has run against a physical controller:
past the rehearsal it has never touched hardware. The procedure is
[`docs/runbooks/real_cell_first_pick.md`](../runbooks/real_cell_first_pick.md).

**The default pick is open-loop.** Perceive, generate and score candidates on a deterministic
geometric rank, run the safety preflight and IK, move and close, log. The decision gate, the
closed-loop refine, verify and recover path, fusion with its commit gate, the rerank stage, the learned
success model and the reinforcement-learning layer are all built and all default to `enabled: false`.
Check that against the tree rather than this page:

```bash
python -c "from willy import load_tree; g = load_tree(None).robot.grasping; print({n: getattr(g, n).enabled for n in ('fusion', 'decision', 'closed_loop', 'verification', 'recovery', 'success_model')})"
```

**Most simulator runners take a separate path.** They build the service through `from_components`
and turn the config-driven overlays back on in runner code, taking the mode from a flag and fixing the
attempt count. So `grasping.default_mode` and `grasping.max_attempts` are ignored there, while every
`grasping:` sub-block the runner does not override reaches a simulated pick. `run_multiview_pick`
builds through `from_robot_config` by default (`--boot config`).

**Record logging is opt-in.** `from_robot_config` reads `grasping.record_log_path` and stamps the
robot's vendor and model, so setting it makes every `pick()` append one record, which the soak, KPI
and learning tools read. Features per candidate are written only while `rl.mode` is `rl_shadow`. Left
null, nothing is recorded. The shipped KPI gate is a synthetic contract self-check: it proves the
telemetry and KPI plumbing agree with each other, not that any grasp is good, and its own provenance
block says so.

**Two required sections have few live readers.** `camera.cameras` and `camera.stereomatcher` are
required at load and hold most of the schema's required fields, but the simulated cell authors its
cameras from `robot.sim.cameras`. A green `cam.yaml` is not evidence that a camera works.

---

## Appendix: reminders, in the order they will bite you

- `where` lists only 40 keys per tier until you pass `--limit`.
- Only the bare `python -m src.config` is a gate: `explain` and `decisions` exit `0` on any tree
  that loads, and `where` on anything it finds.
- `--print` belongs to the bare command and is refused on a subcommand.
- Unset an overlay value with `"__null__"`, never `null`.
- Call `reload_config()` after editing a file in a running process.
- Unset `WILLY_ADAPTATION_OVERLAY` before trusting an explanation.
- A tree of your own that names a hand carries an unchanged copy of `grippers/`.
- A relative path in the tree is read against the tree's folder, never the working directory; a
  copied tree's `../` paths need rewriting, its `${WILLY_PROJECT_ROOT}` paths do not.
- A green `python -m src.config` says nothing about the presets.

**Adding a new field?** The rules are in [`src/config/README.md`](../../src/config/README.md): a
schema field with a default that changes no behaviour, its documentation on the schema field rather
than in a YAML comment, a YAML line only where a cell actually chooses something, and the package
README updated. New behaviour ships off by default, and a pick with it off is byte-identical.

---

*Next: [02-models.md](02-models.md), the perception models this config points at.*
