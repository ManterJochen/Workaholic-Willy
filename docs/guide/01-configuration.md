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
[section 8](#8-what-a-config-actually-reaches) says what a config reaches.

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
`scripts/model_weights/fetch.py` writes and the committed ranker (the success model and the RL baselines left
on 2026-10-16). The
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
| `ur5e` | a real UR5e on a bench: workspace box, home, bench fixture, planning world, four measurements to take |
| `ur3e` | the shorter arm's reach-anchored geometry and kinematics, under `sim` or on its own against a real UR |
| `sim` | the simulated cell, with the camera inventory and the model overlays the simulator needs |
| `ur5e,eth2` | the same bench with two fixed RGB-D cameras, fused; `eth2` alone is the camera half and names no arm |

`console_dummy` is the desk profile: a dummy arm and a dummy hand that command nothing, and two poses written by
hand, "Ablage links" (its default place) and "Parkposition", so the console's desk runs a task as shipped. The two
desk programs in [`examples/simulation/`](../../examples/README.md) run on it.

[`config/robot/robot.ur5e.yaml`](../../config/robot/robot.ur5e.yaml) draws a line worth adopting for
your own bench. A wrong value that fails closed ships as a worked example with its assumption stated:
a workspace box that is too small refuses a motion visibly, and an operator widens it. A wrong value
that fails open does not ship at all: a tool frame or a payload that is merely plausible drives the arm
into the bench and logs a success. So those are commented blocks, and the refusals in
`run_config_preflight` and `URRobotArm.connect` stay armed.

**One thing decides the two-camera cell**, documented in place in
[`config/camera/cam.eth2.yaml`](../../config/camera/cam.eth2.yaml): **each camera's calibration is
declared on its rig**, as `camera.cameras.rigs[<id>].extrinsics`. A cell fuses every enabled RGB-D rig
that declares one, where two or more do and the primary, `camera.cameras.primary_rig_id`, is among
them; there is no switch. The pick service reads the primary's CAMERA to BASE transform there, and the
second camera's resolver reads its own. The calibration command prints that block after each camera,
and a fused camera whose file does not load is refused when the cell is built.

**Prefer the flag, because the variable is sticky and silent.** `WILLY_PROFILE` lives for the whole
shell session and `load_tree()` honours it everywhere. A forgotten `sim` changes safety thresholds and
calibration board parameters under you, and nothing in the files warns you. Clear it with
`Remove-Item Env:\WILLY_PROFILE`, or `unset WILLY_PROFILE` on POSIX. The validate banner names the
chain it loaded, which is the cheapest check.

**Every layer must exist.** Each layer in the chain must contribute at least one `*.<layer>.yaml`
under the tree, or the load fails naming the layer. A typo mid-chain would otherwise merge as nothing
and leave the cell running another robot's geometry.

**A chain that composes is not always meaningful.** The loader validates the merged tree, not the intent
behind the chain, so a chain that loads can still be a cell nobody meant. `sim,web` was refused while the
`web` layer carried a KUKA arm, because the sim layer then landed a second name for a UR arm on a
`vendor: kuka` cell. The `web` layer writes no robot file since the KUKA driver went, and the arm has one
name only (`ur.model`, or `sim.robot_model` on a simulated cell), so `sim,web` loads a sim cell on the webcam
rig. Validate any chain in one command rather than trusting a document about it.

**Merge rules.** The full treatment is in [`src/config/README.md`](../../src/config/README.md). Three
rules to remember:

- A `null` overlay leaf keeps the base value, so a partial overlay cannot wipe a field by accident.
- Scalars and lists replace. There is no list-append merge.
- `"__null__"` is the only way to unset, resolved to `None` after the merge, because of the first
  rule. `src/config/loader.py` defines it; no shipped overlay needs it since the model overlays left on
  2026-10-17.

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
from the schema, which is the reliable way to find a typo. Every key is asked as the file writes it:
no field has a YAML spelling of its own since the stereo matcher block, whose keys kept OpenCV's
camelCase, left the tree on 2026-10-17.

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
goes. The numbers the guards hold an arm to are no keys since 2026-10-17: every real arm runs the owner's
cell's, and the simulator's arm, which loosens some of them because the simulator has no controller-side
safety to lean on and turns its base half a turn, states its own in code
([`src/willy_sim/guard_numbers.py`](../../src/willy_sim/guard_numbers.py)), so `decisions` lists none of them.
Which robot it checks is not a safety key either: the arm reports the model its vendor block names,
`sim.robot_model` on the simulator chain.

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

**A taught pose has a door of its own.** `robot.motion.named_poses` and `robot.motion.default_place_pose` are never
in `WRITABLE` and never a `PATCH`: `set_named_pose` and `set_default_place_pose` (`ConfigTree.write_named_pose`,
`write_default_place_pose`) write them, all or nothing, into the **last layer** of the chain only, and refuse a
chain whose last layer git does not keep out (`no_layer`). Clearing the default place writes `"__null__"`.

---

## 5. Worked example: a config for a new cell

Target: a UR3e on a bench, a Robotiq 2F-85, a small workspace box, one stereo webcam pair.

### Step 0: new profile layer, or new tree

If the new cell shares most of an existing one (the same optics and models, a different arm), add a
**profile layer**: one `<base>.<layer>.yaml` next to each base file it changes. Each measured value
stays stated once, and this is the shipped pattern. If it shares nothing, make a **new tree** and point
`--data` or `load_tree(root=...)` at it.

A tree given to a simulator runner's `--data-dir` loads under the `sim` profile, so it must carry
`*.sim.yaml` overlays or the load fails. Without them the simulated cell would run the production
values, the real vendor among them.

### Step 1: the required half, and a first green run

Copy and edit rather than write from nothing: the required fields live in these files. The copy of
`grippers/` stays unchanged, because the hand you name in step 2 is read from it.

```powershell
$Cell = "C:/cells/mycell"
New-Item -ItemType Directory -Force "$Cell/camera", "$Cell/models", "$Cell/robot" | Out-Null
Copy-Item config/camera/cam.yaml "$Cell/camera/"
Get-ChildItem config/models/*.yaml | Where-Object Name -notlike "*.sim.yaml" | Copy-Item -Destination "$Cell/models/"
Copy-Item -Recurse config/grippers "$Cell/"
python -m src.config --data $Cell
```

```bash
CELL=~/cells/mycell
mkdir -p $CELL/camera $CELL/models $CELL/robot
cp config/camera/cam.yaml $CELL/camera/
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

**The workspace box.** This tree writes no `robot.motion.workspace`, so the schema default box is in force.
It is a symmetric placeholder with corners a UR3e cannot reach; see it with
`python -m src.config explain robot.motion.workspace.z_max --data $CELL`. Write your own box, with margin about
every pose the cell drives to: the workspace guard also subtracts 20 mm from every face, every cell's
since 2026-10-17 (the key `safety.limits.workspace_margin_mm` left that day).

**How the cell moves is one block.** Every key of it stands under `robot.motion` since 2026-10-17: the planner,
`max_velocity` and `max_acceleration`, the `workspace` box, home (`home_deg`, in degrees as the pendant shows the
joints; there is no radians twin), the looks (`look_deg`), the `natural_closing_axis`, the taught poses
(`named_poses`, `default_place_pose`) and how a task sets its parts down (`place`). A tree that still writes one of
them straight under `robot` is refused at load, naming where it went, and
`python -m src.config migrate PATH --write` moves it there, comments and all. `robot.safe_pose` left the same day:
nothing moved to it.

**The hand.** `robot.gripper.model` names a file in the gripper registry, and the hand brings its
widths and its collision envelope from it:
`python -m src.config explain robot.gripper.max_width_mm --data $CELL` names that file and field
instead of a layer. A cell that names no hand keeps the default of 85 mm,
the Robotiq 2F-85. The Robotiq driver anchors its count map on that value, so it must be the real
physical open width, not a policy ceiling. `python -m src.config where gripper` lists the rest.

**Where the camera looks from.** A wrist camera sees what the arm points it at, so a wrist cell names its
**looks**: `robot.motion.look_deg`, one list of joint degrees per look, as the pendant shows them
and `python -m src.robot.execution.io_bench --where` prints them, visited in this order.

```yaml
robot:
  motion:
    look_deg:
      - [-90.0, -100.0, -110.0, -60.0, 90.0, 0.0]    # the part's open side first
      - [-70.0, -100.0, -110.0, -60.0, 90.0, 0.0]
```

**Degrees only**, with no radians twin: a look is where the arm goes next, so its unit is said. The load
refuses a list that names no look, a look that names no joint or holds a value that is not a finite number,
looks of different lengths, a length unlike the home's, and a joint past a full turn either way. Radians
written here read as small degrees, which no loader can tell apart: the desk check's `looks` row flags a
look whose every joint sits within 6.3 of zero. A program's own looks override the key. `PickRun` and the
console send every pick whose program names no look to these, a fixed camera's arm too: a wrist camera fuses
them until its grasp is safe, and a fixed camera stops at the first look that finds something
([05](05-pick-loop.md)). Unset, a wrist camera looks from home and a fixed camera does not move to look.

**How the hand naturally stands.** `robot.motion.natural_closing_axis` names the way round the jaws close where
nothing else says: a name (`x`, `-x`, `y`, `-y`, `radial`, `-radial`, `tangential`, `-tangential`, a
leading `+` allowed; `radial` and `tangential` are read at each place), the names `Pose.tool_down` takes,
or a taught pose's quaternion `[x, y, z, w]`, of which only the heading of its tool +X on the base XY plane
counts, not its pitch.

```yaml
robot:
  motion:
    natural_closing_axis: "-y"    # the tool +X along base -y: the owner's cell, where its wrist D415's image stands upright
```

**Camera grasps stay free**, any closing direction and any tilt: of a grasp's two equivalent wrist turns,
half a turn about its approach apart, the pick, the push, the `Locator` and `Scene.grasps` take the one
whose tool +X lies nearer this direction, and none is left out ([05](05-pick-loop.md), 5.2). A program's own
`closing_axis` wins, and beside the simulator's `align_closing_to_base_x` no grasp is turned.
`robot.tool_down(x, y, z)` builds the cell's fixed poses along it ([04](04-robot-and-safety.md)). The load
refuses an unknown name, a quaternion that is none, and one whose tool +X stands within 10 degrees of the
vertical, naming the key; reading it loads no grasping package. Unset, the default, nothing is turned and
`robot.tool_down` closes along x, exactly as `Pose.tool_down`. No shipped profile sets it: the owner's cell
sets `"-y"` in its own profile on the cell PC.

**Poses taught at the cell.** The operator console teaches a pose by hand (Setup, "Pose einlernen"): a person
guides the arm there, the exact guard and the planner screen it while it holds, and only a clear pose, or one in
the planner's band, is written, under `robot.motion.named_poses`:

```yaml
robot:
  motion:
    named_poses:
      drop_left:                              # the NAME: what the YAML and the command reader answer with
        joints_deg: [-60.0, -95.0, -120.0, -55.0, 90.0, 0.0]
        label: "Ablage links"                 # what the chat, the cards and a spoken command say
        taught_at: "2026-10-01T09:12:00+02:00"
        screen: clear                         # clear | band; an unscreened or refused pose is never written
        note: ""
    default_place_pose: drop_left             # where a task sets its part down when its command names no target
```

- **The name** is an ASCII identifier of at most 32 characters: no Python keyword, never `home` (Home stays
  `robot.motion.home_deg`), no word YAML reads as true, false or null, no `__dunder__`.
- **The label** is free text, Unicode and spaces allowed, at most 40 characters, with no newline and no `" #"`.
  A pose answers to its name and its label, and the load refuses two poses that share a word, whatever the case.
- **The joints** are degrees, under the looks' rules. Layers merge a pose key by key, so a pose a later layer
  names with its joints alone keeps a lower layer's screen: a task screens every pose again before it moves.
- **A place pose says where the part's bottom is let go**: a task raises the tool over it by the part's hang.
- **`default_place_pose`** names one of them, or `null` for none, and a task that names no place is then refused
  (`no_place_declared`). A name that is no pose of the tree is refused at load.
- **Where they are written.** Into the **last layer of the chain** and nowhere else, which must be the cell's own
  and kept out of git: end the chain in `cell` (`--profile ur10,hande,cell`) with `robot/robot.cell.yaml`, which
  `.gitignore` already covers (`config/**/*.cell.yaml`). A chain with no layer, or a last layer git tracks or does
  not ignore, is refused before the arm is freed (`no_layer`). The console never renames or deletes a pose: edit
  the YAML. A joint list written by hand as a block cannot be rewritten in place; write it as a flow list.

**How far the parts hang.** `robot.motion.place.part_hang_mm` is how far the longest part hangs past the fingertips:
a place sets the part's bottom down where the taught fingertips stood, raised by it, and keeps it over a box's rim.
It ships `null`, which hangs the part from the TCP; the owner's cell sets 1. It was
`robot.safety.planning_world.payload.length_mm` until 2026-10-17, when the carried part left the planner (no cell
modelled it: a closed hand is judged as an empty one, by the exact guard at full stroke); the load refuses the old
key and names this one, the migrate command moves it, and no task waits on it.

**The place's numbers are the owner's cell's**, fixed in
[`src/robot/execution/place_target.py`](../../src/robot/execution/place_target.py) since 2026-10-17: a part set into
a box goes 30 mm under the rim and keeps 10 mm inside its opening, the jaws open 20 mm over a box's rim (a task may
give its own, 0 to 50), parts side by side lie 20 mm apart, and a carry over the rim keeps the part's bottom 15 mm
over the rim beyond the rim air. A bin the check before a drop lost is always looked for again. What stays under
`robot.motion.place` is what a cell chooses or measures: `release_in_a_box`, `side_by_side` with its `grid`, `carry`,
and `part_hang_mm`.

**Halt now.** The console's "Sofort anhalten" latches the arm, so nothing after the move in flight is sent, and
on every UR that move is braked under control (since 2026-10-17, as the owner's cell ran it; the key
`robot.ur.brake_on_halt` left then). It is not an emergency stop ([04](04-robot-and-safety.md), section 8).

**Suction.** A vacuum end-effector uses `robot.gripper.vacuum.*`, read only when `robot.gripper.vendor`
is `"vacuum"`. Under any other vendor that block validates green and is ignored, and every field in it
is a number somebody has to measure.

**Safety bounds stay written even at their default**, so an auditor can read the guard envelope out of
the file. The guards themselves belong to [04-robot-and-safety.md](04-robot-and-safety.md).

A validator on `RobotConfig` kept `safe_pose` inside the workspace box until 2026-10-17, when the safe pose left:
nothing moved to it, and its one reader, the simulator's reach doctor, only asked whether it lay in reach. Three
more stood beside it until 2026-10-13 (P0d D5) and coupled
`safety.self_collision.kinematics_model` to `sim.robot_model`, to `ur.model`, and to nothing at all on any
other vendor, because that key picked the DH chain the self-collision guard evaluates, and a UR5e's
upper-arm and forearm links are roughly 425 mm and 392 mm against a UR3e's 244 mm and 213 mm. They went
with the key, because the arm now names itself: the guard takes the model the arm reports, and `ur.model`
and `sim.robot_model` each refuse an arm the registry does not admit. A tree that still writes the key,
`null` included, is refused at load with a sentence naming those two keys instead.

### Step 3: audit the decisions, which is the acceptance check

Run `python -m src.config decisions --data $CELL --section robot` and read it line by line. An entry
you did not intend is a bug in the YAML. An entry you intended that is missing means you wrote a value
equal to the default, which is correct and not listed: `planner: curobo` does not appear, because
that is the schema default. Lines from a profile layer carry their layer name and file.

---

## 6. Enabling a default-off feature

**Find the gate first.** Few `robot.grasping.*` blocks carry `enabled: false` since the cleanup of
2026-10-16 and 2026-10-17 (`deep_ranker` does), and writing `enabled: true` under a block that has no such
field fails the load: `support` has none,
and a declared container's walls are switched by `support.container.wall_collision_enabled`. So ask
first: `explain robot.grasping.support.enabled` answers that the key is not known, and
`where support` shows you what the block holds.

**One gate is a string.** `robot.grasping.calculator` is `geometric` or `deep`, and it selects the grasp
generator the cell runs. Every calculator the cell builds goes through `build_calculator`
([`autonomous_grasp/`](../../src/robot/execution/autonomous_grasp/README.md)), so the key is live. It
fails closed: `deep` with no readable artifact raises rather than falling back, because a cell that
asked for the learned generator and got the analytic one would report the analytic one's results under
the learned one's name. **No trained weights ship in this repository**; you train on your own cell's
data ([`src/robot/grasping/deep/README.md`](../../src/robot/grasping/deep/README.md)), and a cell
builds from your weights only once their proof has passed, which nothing can issue yet (2026-10-09).

The workflow is three steps, and the file edit is the last one:

1. **Find** the key: `python -m src.config where container`.
2. **Read** it: `python -m src.config explain robot.grasping.support.container.wall_collision_enabled`.
3. **Write** the block with only the keys you change, and put the reason in a comment directly above
   the line. `explain` shows that comment to the next person who asks.

Then run the bare command, `explain`, and `decisions --section robot.grasping` again. If the block lands
under the wrong parent, `extra="forbid"` catches it, and the dotted path in the message says where it
went: `robot.safety.support` means you wrote it under `safety:`, not under `grasping:`.

**Some shapes are refused before they can mislead you.** A block removed on purpose, such as
`verification` or `dense_recovery` (both 2026-09-29) or `decision` and the eight blocks that left with it
on 2026-10-16, and a switch fixed at the owner's cell's value (2026-10-17), is refused at load with the
sentence that says what to do instead. And
`robot.grasping.support.container.wall_collision_enabled` without both `interior_min_mm` and
`interior_max_mm` is refused at load, rather than running a cell that believes a bin protects it when
it cannot locate the walls.

**Recovery has no block to write.** Every cell runs the recovery the owner's cell ran, fixed since
2026-10-16: a rescan once a pick in `auto` and `dense_clutter`, and in `dense_clutter` the push and the
clearing of a blocker inside the pick attempt. A tree that still writes `robot.grasping.recovery` is refused
at load with the sentence that says to delete the block. **The push** needs no fixture: it lands inside the
automatic push box. It moves the part 30 mm unless a run asks otherwise (`push_mm` on a `PickRun` or a
console run, 10 to 50 mm).

**Whether a push may change the scene** is one key, `runtime.task.critical_parts` in `app/runtime.yaml`
(the owner, 2026-10-03; under `robot.grasping.recovery` until 2026-10-16). `false`, the default: a boxed-in
part is pushed first, and the push may shove and brush its neighbours; a blocker is set aside where no push
plans. `true`: nothing is pushed, and a blocker is set aside instead. A console run can set it for itself
(*Kritische Teile*). A clearing tries three of a blocker's grasps, best first. The cell reads it when it is
built.

**How the console starts a task is the cell's too**, beside it in `runtime.task` (the owner, 2026-10-16; until
then each browser kept its own): `start` (`enter`, Enter starts a clean reading at once, or `card`), `scope`
(`once` or `until_empty`), `place` (`default`, `pose:<name>` or `camera:<phrase>`, where a sentence names no
place), `anything` (a card whose sentence names no part comes ticked for anything the camera sees) and `looks`
(`first`, `when_needed`, `every`). The console's Settings page (*Auftrag*) writes all six into the cell's own
layer, `app/runtime.cell.yaml` under `--profile <your cell>,cell`, as it writes a taught pose
([`src/config/README.md`](../../src/config/README.md)); none of them starts a task.

The recovery acts in `auto` and `dense_clutter` alone, and `deep_ranker`, switched on, changes no grasp: a
shadow by design. Read [`docs/grasping-config-reference.md`](../grasping-config-reference.md) before you
enable or measure any of them.

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
- **`removed on purpose`**: a key that left the schema on purpose, with the sentence that names what replaced
  it or the value it is fixed at. `python -m src.config migrate PATH --write` brings a whole old tree to the keys
  this Willy reads.
- **You edited a YAML and nothing changed**: the load is cached. Call `reload_config()`.

---

## 8. What a config actually reaches

**The real-cell path reads the whole tree.** `Cell.from_tree(load_tree())` builds a cell from it and
runs the steps in a fixed order: preflight, build, safety attestation, then connect (lock, arm, then
gripper) and pick. [`python -m src.robot.execution.real_cell`](../../src/robot/execution/real_cell/README.md)
and the [examples](../../examples/README.md) go through `Cell`, and the operator console calls the same
builders, `build_real_cell` and `build_rehearsal_cell`. `Cell.rehearsal(...)` and `--rehearse` run the
whole path on a dummy arm and a synthetic scene. Under it, `AutonomousGraspService.from_robot_config`
builds the pick service from `robot:`. **Its arm, hand and camera have run on a physical cell**: a UR10
(CB3) with a D415 on the wrist and a Hand-E on one tool output connected, moved, calibrated its wrist
camera and picked by camera, through the examples; the console has not run there. The wrist looks of
2026-09-29, their generated view and the push have not run there yet. The procedure is
[`docs/runbooks/real_cell_first_pick.md`](../runbooks/real_cell_first_pick.md).

**The default pick is open-loop.** Perceive, generate and score candidates on a deterministic
geometric rank, run the safety preflight and IK, move and close, log. The recover path runs on every cell
and multi-camera fusion on every cell of two calibrated RGB-D rigs, with no switch since 2026-10-16; the
learned ranker's shadow is built and defaults to `enabled: false`; the decision gate, the rerank stage, the
learned success model and the reinforcement-learning layer left on 2026-10-16 with their code. The hold is
checked after every close by the execution policy itself, wherever the gripper can report one; the
separate post-grasp verification stage, which no pick path ran, was removed on 2026-09-29 with its
`verification` block. **The looks of a wrist camera need no switch**: a pick handed looks goes through
them and fuses what it saw, and `PickRun` and the console hand every wrist pick its looks, the
program's, else `robot.motion.look_deg`, else home ([05](05-pick-loop.md)). Check the
switches against the tree rather than this page:

```bash
python -c "from willy import load_tree; g = load_tree(None).robot.grasping; print({n: getattr(g, n).enabled for n in ('deep_ranker',)})"
```

**Most simulator runners take a separate path.** They build the service through `from_components`
and turn the config-driven overlays back on in runner code, taking the mode from a flag and fixing the
attempt count. So `grasping.default_mode` is ignored there, while every
`grasping:` sub-block the runner does not override reaches a simulated pick. `run_multiview_pick`
builds through `from_robot_config` by default (`--boot config`).

**Record logging is on.** `from_robot_config` reads `grasping.record_log_path` and stamps the
robot's vendor and model, and every `pick()` appends one record, which the KPI tools read: to that file,
or, left null, to `grasp_records.jsonl` in the log folder (since 2026-10-17). `--records-gate` judges such a
log and can fail; the synthetic soak
that passed by construction left on 2026-10-16.

**A required section with few live readers in simulation.** `camera.cameras` is required at load
and holds most of the schema's required fields, but the simulated cell authors its cameras from
`robot.sim.cameras`. A green `cam.yaml` is not evidence that a camera works. The stereo matcher block
that stood beside it, `camera/stereomatcher.yaml`, left on 2026-10-17: a stereo rig's matcher is derived
from its calibration and fixed in code, and a tree that still carries the file is refused at load,
saying so.

---

## Appendix: reminders, in the order they will bite you

- `where` lists only 40 keys per tier until you pass `--limit`.
- Only the bare `python -m src.config` is a gate: `explain` and `decisions` exit `0` on any tree
  that loads, and `where` on anything it finds.
- `--print` belongs to the bare command and is refused on a subcommand.
- Unset an overlay value with `"__null__"`, never `null`.
- Call `reload_config()` after editing a file in a running process.
- A tree of your own that names a hand carries an unchanged copy of `grippers/`.
- A relative path in the tree is read against the tree's folder, never the working directory; a
  copied tree's `../` paths need rewriting, its `${WILLY_PROJECT_ROOT}` paths do not.
- Every key of how a cell moves is under `robot.motion` since 2026-10-17, and home is `robot.motion.home_deg`, in
  degrees alone; `migrate` (below) writes a radians home given on one line in degrees, and one written over several
  lines is converted by hand.
- `robot.motion.look_deg` is degrees; radians there read as small degrees, and only the desk
  check's `looks` row says so.
- A taught pose, and a task setting the console's Settings page chooses, are written into the chain's last
  layer only, and that layer must be git-ignored: end the console's chain in `cell`.
- The parts' hang past the fingertips is `robot.motion.place.part_hang_mm` since 2026-10-17; a tree that still writes
  `robot.safety.planning_world.payload.length_mm` is refused at load, naming it.
- A tree that still writes a key removed on purpose is refused at load, naming what replaced it;
  `python -m src.config migrate PATH --write` brings an old tree to the keys this Willy reads, every comment kept.

**Adding a new field?** The rules are in [`src/config/README.md`](../../src/config/README.md): a
schema field with a default that changes no behaviour, its documentation on the schema field rather
than in a YAML comment, a YAML line only where a cell actually chooses something, and the package
README updated. New behaviour ships off by default, and a pick with it off is byte-identical.

---

*Next: [02-models.md](02-models.md), the perception models this config points at.*
