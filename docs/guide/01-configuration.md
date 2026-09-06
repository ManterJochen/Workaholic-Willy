# Building the configuration

**Who this is for.** Anyone who has to make Workaholic-Willy run on a cell that is not the one in
the repository: a different arm, a different gripper, a different camera rig, a different workspace
box.

**What you get.** A validated `AppConfig` for your own cell: a tree, or a profile layer, that passes
`python -m src.config` with exit code 0, whose every non-default value you can justify line by line,
and which you can interrogate with three commands instead of grepping five files.

**What you do not get.** A running pick. Read [section 9](#9-what-a-config-actually-reaches) before
you assume a full `robot.grasping:` block changes what the arm does.

**This is the walkthrough, not the reference.** The reference is
[`src/config/README.md`](../../src/config/README.md): the shipped profile layers, the merge rules,
the safety block, the guided-write path, and the rules for adding a field. Where the two overlap,
that one wins. Siblings: [02-models.md](02-models.md), [03-calibration.md](03-calibration.md),
[04-robot-and-safety.md](04-robot-and-safety.md) and [05-pick-loop.md](05-pick-loop.md).

## Prerequisites

Python 3.11, the repository root as your working directory, and `pydantic` plus `PyYAML`. Check with
`python -c "import yaml, pydantic; print(pydantic.VERSION)"`; if it fails, install the whole
dependency set with `pip install -r requirements.txt`, or `requirements-cpu.txt` on a host that
cannot take the CUDA wheels. Nothing else is needed for this guide: the config layer imports no
camera, no robot SDK and no simulator, and `python -m src.config` boots nothing. The CLI prints
through an encoding-safe emitter, so a cp1252 Windows console needs no codepage change.

---

## 1. The mental model

**One object.** `load_config()` reads a short, closed list of YAML files, substitutes `${ENV}` in the
raw text, deep-merges each file's profile overlays on top of it, resolves `"__null__"` sentinels to
`None`, and hands the merged dict to `AppConfig.model_validate`. There is exactly one `AppConfig` and
it is assembled, never scattered. The pipeline is `_load_cached` in
[`loader.py`](../../src/config/loader.py).

**Strict.** Every schema class inherits `StrictModel` with `extra="forbid"` and `frozen=True`. An
unknown key is a load error, never a silently ignored line, and `cfg.x = y` raises. That is the whole
safety argument for this layer: a misspelled threshold cannot become a setting that quietly never
applied.

**A file states what this cell decided.** Everything a file stays silent about is the schema default,
in force and unchanged. That is why the shipped `robot.yaml` leaves most `grasping:` and `rl:`
sub-blocks out. The features are untouched, and `extra="forbid"` still accepts every one of those
keys.

**Therefore the tree must be asked, not read.** Most schema fields are written in no YAML at all. Ask
your own tree how lopsided it is:

```bash
python -c "from pathlib import Path; from src.config._schema_index import schema_index; from src.config._provenance import index_origins; i=schema_index(); o=index_origins(Path('config'), []); print(len(i),'schema fields;',len([k for k in i if k in o]),'written in some YAML')"
```

Reading the files can never show you the rest. [Section 4](#4-how-to-ask-the-config-questions) exists
because of that gap.

`load_config` is memoised by absolute data directory and resolved profile chain, so editing a YAML at
runtime changes nothing until you call `reload_config()`. A simulator config and a production config
therefore coexist in one process.

---

## 2. What the loader reads

The list is closed and lives in `_load_cached`. Nothing is auto-discovered except `models/*.yaml`.
The tree is [`config/`](../../config/) at the repository root.

| Path, relative to the data root | Required | Top-level key | Lands at |
|---|---|---|---|
| `config/camera/cam.yaml` | yes | `cameras:` | `camera.cameras` |
| `config/camera/stereomatcher.yaml` | yes | `stereomatcher:` | `camera.stereomatcher` |
| `config/camera/hand_eye.yaml` | no | `hand_eye:` | `camera.hand_eye` |
| `models/*.yaml` | directory required, non-empty | each file's top-level keys | `models.*` |
| `config/robot/robot.yaml` | no | `robot:` | `robot` |
| `config/app/runtime.yaml` | no | `runtime:` | `runtime` |

`robot:` absent means `cfg.robot is None`, a legal camera-only cell. The asymmetry is real: the
required fields cluster under `camera` and `models`, while every field on `RobotConfig` has a
default. Get the current list rather than trusting one written down:

```bash
python -c "from src.config._schema_index import schema_index; print(sorted(k for k,v in schema_index().items() if v.required))"
```

In `models/`, every non-overlay `*.yaml` is globbed and each file's top-level keys merge into one
mapping. A duplicate top-level key across two base files is a hard `ConfigError`, so you cannot
override a model by adding a second base file; that is what profile overlays are for. Deleting the
file that carries a required key is a load failure, not a leaner config.

Three things under `config/` are not read by `load_config` at all: `config/robot/kpi_thresholds.yaml`, which
belongs to the soak and KPI gate; `robot/templates/kuka/`, controller-side files copied to a KUKA by
hand; and `grasping_presets/*.yaml`, merged downstream and outside Pydantic
([section 8](#8-the-grasping-presets-a-schema-bypass)).

**`config/all_keys/` is a reference tree, not one that loads.** It writes out every key the schema
accepts with what it does, its default and its legal values. Nothing loads it by default, and it
validates on its own, so it is checkable rather than merely documentation:

```bash
python -m src.config --data config/all_keys
```

Read it to find out what exists. Do not copy it over the shipped tree: a file that restates every
default turns each of those lines into something a reader has to check against the schema.

**The one input that is not a file in the tree.** If `WILLY_ADAPTATION_OVERLAY` is non-empty, the
loader deep-merges the `robot:` section of the YAML at that external path before validation. It is
narrow and unset by default. It matters here because the provenance side-car does not index it: with
the variable set, `explain` and `decisions` attribute an overlaid value to the base file. Unset it
first.

---

## 3. Profiles, overlays and the four worked cells

`WILLY_PROFILE` takes a comma-separated chain of layer names, applied left to right. For a base file
`foo.yaml`, layer `L` contributes `foo.L.yaml` from the same directory. A layer is not a whole
alternative configuration. It is one dimension of one, and dimensions compose.

```bash
python -m src.config --profile sim,ur3e,tiltcam                # preferred: not sticky
python -c "from src.config.loader import available_profiles; print(available_profiles())"
```

**Four of the shipped layers are worked cell examples**, and reading them is the fastest way to
learn what a real cell has to state.

| Chain | The cell it describes |
|---|---|
| `ur5e` | a real UR5e on a bench: workspace box, home pose, safe pose, bench fixture, planning world, and the four bench measurements written out as commented blocks with what to measure |
| `ur3e` | reach-anchored geometry and kinematics for the shorter arm; it works both as a robot layer under `sim` and on its own against a real UR |
| `sim` | the simulated cell, including the camera inventory and the model overlays the simulator needs |
| `ur5e,eth2` | that same bench with two fixed RGB-D cameras, fused. `eth2` is the fusion half only and says nothing about the arm, so the pair is the cell and `eth2` alone is not |

[`config/robot/robot.ur5e.yaml`](../../config/robot/robot.ur5e.yaml) states the line it draws, and it
is worth adopting for your own bench. A wrong value that fails closed ships as a worked example with
its assumption stated, because a workspace box that is too small refuses a motion visibly and an
operator widens it. A wrong value that fails open does not ship at all: a tool frame or a payload
that is merely plausible drives the arm into the bench and logs a success, so those are written out
as commented blocks and the refusals in `run_config_preflight` and `URRobotArm.connect` stay armed.

**Two traps live in the two-camera cell**, both documented in place in
[`config/robot/robot.eth2.yaml`](../../config/robot/robot.eth2.yaml), and both produce a cell that
looks like it works.

- **The primary camera does not go in `fusion.cameras`.** That map is every camera except the
  primary, which already streams through the main perception source. Four places read the map and
  two read it differently, so a primary listed there is permanently among the cameras that delivered
  no frame: a warning on every pick under `on_camera_unavailable: degrade`, and a raised error under
  `refuse`. `robot.sim.yaml` does list all of its cameras including the primary, which is right there
  and wrong here.
- **The primary camera's own artifact is a separate key.** `fusion.extrinsics_artifact_path` is what
  `AutonomousGraspService.from_robot_config` checks for a CAMERA to BASE transform. The calibration
  runner prints a `fusion.cameras` block to paste after each camera, and that block alone leaves this
  key null, so a cell can be fully calibrated, load both artifacts, and still be refused at build.

**Prefer the flag, because the variable is sticky and silent.** `WILLY_PROFILE` lives for the whole
shell session and `load_config()` honours it everywhere. A forgotten `sim` changes safety thresholds
and calibration board parameters out from under you, and nothing in the files warns you. Clear it
with `Remove-Item Env:\WILLY_PROFILE`, or POSIX `unset WILLY_PROFILE`. The cheapest check is the
validate banner, which names the chain it loaded.

**Why chain instead of fork.** The layers are independent dimensions. Expressing "the simulated cell,
but a UR3e" as its own profile would fork every `sim` file and the copies would drift. Chained, each
value is stated once, and changing the robot layer alone keeps a measured difference attributable to
it.

**Fail-closed per layer.** Every layer must contribute at least one `*.<layer>.yaml` under the data
root, or the load fails naming the layer. A typo mid-chain would otherwise merge as a silent no-op
and leave the cell running another robot's geometry.

**Composability is not meaningfulness.** Some chains are refused by the cross-field validators in
section 5, because they would evaluate one robot against another robot's geometry. `sim,web` is one:
it lands `kinematics_model: ur5e` on a `vendor: kuka` cell, and that key selects a Universal Robots
DH table. Check any chain in one command rather than assuming a document is right about it.

**Merge rules.** Full treatment in [`src/config/README.md`](../../src/config/README.md). Three things
to internalise. A `null` overlay leaf keeps the base value, so a partial overlay can never
accidentally wipe a field. Scalars and lists replace, with no list-append merge. And `"__null__"` is
the only way to unset, resolved to `None` after the merge, precisely because of the first rule. The
canonical use is in `config/models/object.sim.yaml`, and [02 section 6](02-models.md) is why it
matters.

**`${ENV}`.** `${VAR}` and `${VAR:-default}` are substituted in the raw text before `yaml.safe_load`,
so the result must still be valid YAML. Quote anything that could contain a `:` or a `#`. An unset
variable with no default is a hard error that names the file.

---

## 4. How to ask the config questions

This is the section that makes the system usable. Learn it before you edit anything. The shared flags
`--data` and `--profile` work on both sides of the subcommand, so
`config --profile sim explain KEY` and `config explain KEY --profile sim` agree.

### 4.1 Validate

```bash
python -m src.config
python -m src.config --profile sim,ur3e
python -m src.config --data /path/to/mycell --profile night
```

Exit codes: `0` valid, `1` `ConfigError` (missing file, parse error, unresolved environment
variable, schema failure), `2` bad CLI arguments. It boots nothing, so put it in CI and in your own
shell history. The banner names the chain that was actually loaded, so a chain that came from the
environment shows up with no flag given.

**Only the bare command is a usable gate.** `explain`, `where` and `decisions` return `0` once the
tree loads; an unknown key is reported in the text, not in the exit code. `--print` dumps the
validated `AppConfig` as JSON, which is good for diffing two chains and useless for understanding a
cell, because it cannot tell a decision from a restated default. Passing `--print` to a subcommand is
refused with a message naming the command that does accept it.

### 4.2 `explain <dotted.key>`: what is this, and where did it come from

The single most valuable command here.

```bash
python -m src.config explain robot.safety.self_collision.planner_margin_mm --profile sim,ur3e
```

For one key it prints the value, tier, type, constraints and schema default; the documentation
harvested from the schema source; the winning file, line and layer plus the full override chain; and
the YAML comment block above the winning line. That last one is the reason a value is what it is, and
`yaml.safe_load` throws it away, which is why this beats grep.

On a key no YAML mentions it says so and still gives the type, default and meaning. That is most of
the schema: someone configuring real hardware by reading `robot.yaml` cannot otherwise discover
`robot.ur.model`, the field that decides which robot they are configuring. On an unknown key it says
the key is not known and offers near misses from the schema index, which is the reliable way to find
a typo. Ask with the snake_case attribute name and an alias resolves, so
`camera.stereomatcher.num_disparities` finds the YAML's `numDisparities`.

### 4.3 `where <substring>`: what am I allowed to set

Searches the schema index, not the YAML. That is the entire point: grepping the files for "gripper"
misses the whole suction end-effector, because no shipped YAML writes it.

```bash
python -m src.config where gripper
python -m src.config where grasping --limit 500
```

`where` lists at most 40 keys per tier by default, so on a broad needle the key you came for may be
in the part it elided. The header count is always honest and the footer says how many were hidden.
Raise `--limit`, or narrow the substring. `where` loads no tree on purpose, so it still works when
the config is broken. `--tier T` filters the listing and `--tier all` shows every tier, but a tier is
a display filter and nothing else: every field stays settable, and the same field can print under one
tier in `where` and another in `decisions`.

### 4.4 `decisions`: what did this cell actually decide

Prints only the leaves whose loaded value differs from the schema default, each with the file, line
and layer that won, grouped by tier.

```bash
python -m src.config decisions
python -m src.config decisions --section robot.safety --profile sim,ur3e
```

Scoped, it is short enough to read line by line, and that is how to use it. Read which way each entry
goes, because they do not all go the same way. On the simulator chain some entries loosen a guard a
real arm would keep tight, because the simulator has no controller-side safety to lean on. Others
tighten it. A couple are neither: `kinematics_model` and `kinematics_base_yaw_deg` state which robot
and which base frame the guard checks against, which is a correctness fact rather than a relaxation.

### 4.5 From Python

```python
from src.config import load_config, reload_config, ConfigError, AppConfig

cfg = load_config("/path/to/mycell")     # frozen AppConfig; assignment raises
reload_config()                          # drop the (directory, profile) cache
```

For a simulated cell use `load_sim_config` in
[`src/willy_sim/config.py`](../../src/willy_sim/README.md), which builds the chain, forces it for the
load, and restores the previous `WILLY_PROFILE` afterwards. Writing values from a tool goes through
[`src/config/edit.py`](../../src/config/edit.py), which is deliberately narrow: eight keys are
writable, being the payload mass and centre of gravity, the three tool-frame keys, a camera rig's
serial number, and the two controller addresses. They are there because they are measurements and
site facts rather than policy. A group is written as one transaction, comments survive, and every
touched file is restored byte for byte if the loader rejects the result. Ask it what it accepts:

```bash
python -c "from src.config.edit import WRITABLE; print([w.path for w in WRITABLE])"
```

---

## 5. Worked example: a config for a new cell

Target: a UR3e on a bench, a Robotiq 2F-85, a small workspace box, one stereo webcam pair.

### Step 0: new profile layer, or new tree

If the new cell shares most of an existing one (same optics, same models, different arm), add a
**profile layer**: one `<base>.<layer>.yaml` next to the base file. Each measured value stays stated
once, and this is the shipped pattern. If it shares nothing, add a **new tree** and point `--data` or
`load_config(dir)` at it. One constraint either way: a simulator runner's `--data-dir` routes to
`load_sim_config`, which forces the `sim` profile, so such a tree must carry `*.sim.yaml` overlays or
the load fails closed. That is intentional. A tree without them would run the production fp16
detectors through the simulated cell, which is the configuration that costs small-object recall.

### Step 1: the required half, and a first green run

```powershell
$Cell = "C:\cells\mycell"
New-Item -ItemType Directory -Force "$Cell\camera", "$Cell\models", "$Cell\robot", "$Cell\app" | Out-Null
Copy-Item config\camera\cam.yaml, config\camera\stereomatcher.yaml "$Cell\camera\"
Get-ChildItem config\models\*.yaml | Where-Object { $_.Name -notlike "*.sim.yaml" } |
    Copy-Item -Destination "$Cell\models\"
python -m src.config --data $Cell
```

Copy and edit rather than writing from nothing: the required fields live in those files. Then edit
`$Cell\camera\cam.yaml` for the hardware you have. Rigs belong to
[03-calibration.md](03-calibration.md), models to [02-models.md](02-models.md). For a simulated cell,
drop the `Where-Object` filter and copy the overlays too. `robot:` is optional, so a green run here
proves the perception half is sound before the robot half can confuse the diagnosis.

### Step 2: the robot half

Write only the site facts, and use `explain` for every number you are unsure of. A minimal
`robot.yaml` is `vendor`, `ur.model` and `ur.ip`; give the address a `${WILLY_UR_IP:-...}` fallback,
because a config that only loads when the environment is right is a config that fails at 2 a.m.

```yaml
robot:
  vendor: ur
  ur:
    model: "ur3e"
    ip: "${WILLY_UR_IP:-192.168.1.100}"
```

Your tree writes no `workspace_limits`, so the schema default box is in force, and it is a symmetric
placeholder with corners a UR3e cannot reach: check with
`python -m src.config explain robot.workspace_limits.z_max --data $Cell`, then write your own and put `safe_pose` inside
it with margin, because the workspace guard also subtracts `safety.limits.workspace_margin_mm` from
every face. For the gripper, run `python -m src.config where gripper` and then
`python -m src.config explain robot.gripper.max_width_mm`; the default is already 85 mm, the Robotiq 2F-85 this project
ships, and the Robotiq driver anchors its count map on that value, so it must be the real physical
open width rather than a policy ceiling. A vacuum end-effector uses `robot.gripper.vacuum.*` and is
consulted only when `robot.gripper.vendor` is `"vacuum"`; under any other vendor that block validates
green and is ignored, and every field in it is a number somebody has to measure. Safety bounds stay
written even at their default, so an auditor can read the guard envelope out of the file; the guards
themselves belong to [04-robot-and-safety.md](04-robot-and-safety.md).

Four validators on `RobotConfig` will stop you, and all four guard one failure class: evaluating one
robot against another robot's geometry. Three couple `safety.self_collision.kinematics_model` to
`sim.robot_model`, to `ur.model`, and to nothing at all on a vendor that is neither, because only UR
DH tables are bundled and profiles compose. The fourth keeps the retreat pose inside the workspace
box. The mechanism is that the key picks the DH table, and a UR5e's upper-arm and forearm links are
roughly 425 mm and 392 mm against a UR3e's 244 mm and 213 mm.

### Step 3: audit the decisions, which is the acceptance check

Run `decisions --data $Cell --section robot` and read it line by line. Anything you did not intend is
a bug, so go fix the YAML. Anything you intended that is missing means you wrote a value equal to the
default, which is correct and not listed: writing `ip: 192.168.1.100` does not appear, because that
is the schema default. Profile-layer lines carry their layer name and file.

---

## 6. Enabling a default-off feature

**Find the gate first.** Most `robot.grasping.*` sub-blocks carry `enabled: false`, but not all, and
writing `enabled: true` under a block that has no such field is an `extra="forbid"` load failure.
`watchdog` is gated by `mode`, `occlusion` by two separate flags, `gripper_geometry` by nothing at
all, and `robot.rl` by `mode`. So ask first: `explain robot.grasping.watchdog.enabled` answers that
the key is not known, and `where watchdog` then shows you the real gate.

**One gate is not a boolean at all.** `robot.grasping.calculator` is a string, `geometric` or `deep`,
and it selects which grasp generator the cell runs. Both construction sites in
[`src/robot/execution/autonomous_grasp/cells.py`](../../src/robot/execution/autonomous_grasp/README.md) go through
`build_calculator`, so the seam is live. It fails closed: `deep` with no readable artifact raises
rather than falling back, because a cell that asked for the learned generator and quietly got the
analytic one would file the analytic one's results under the learned one's name. **No trained weights
ship in this repository**; you train on your own cell's data. Background:
[`src/robot/grasping/deep/README.md`](../../src/robot/grasping/deep/README.md).

The workflow is three commands and the file edit is last. **Find** the key with `where fusion`.
**Read** it with `explain robot.grasping.fusion.enabled`. **Write** the block back with only the keys
you are changing, and put the reason in a comment directly above the line, because `explain` harvests
that comment and shows it to the next person who asks. Then re-run the bare command, `explain`, and
`decisions --section robot.grasping`. If the block lands under the wrong parent, `extra="forbid"`
catches it, but the message reads oddly until you notice the dotted path: `robot.rl.fusion` means you
appended under `rl:`, not under `grasping:`.

**Two shapes refuse before they can mislead you.** `RobotGraspingConfig.UNWIRED_SWITCHES` lists any
switch that would land in the cell's telemetry while nothing reads it back; setting one is a
load-time refusal, because it would give you a cell that reports a capability it does not have. It
holds one entry today, `occlusion.hard_reject_enabled`, and it stays there until the occlusion score
itself is trusted. Separately, `robot.grasping.support.container.wall_collision_enabled` without both
`interior_min_mm` and `interior_max_mm` raises at load rather than running a cell that believes it is
protected by a bin whose walls it cannot locate.

Which block can fire in which grasp mode is its own subject, and several ship their operative weight
at `0.0`, so `enabled: true` alone is inert by construction. Read
[`docs/grasping-config-reference.md`](../grasping-config-reference.md) before enabling or measuring
any of them.

---

## 7. Reading a validation failure

Every load-time failure is a `ConfigError`, rendered by `_describe_validation_error` in `loader.py`.
Per offending key it prints the dotted path, the file and line with the layer where there is one, the
pydantic message, and for an unknown key a suggestion. The location line tells you which of three
shapes you have.

- A real file and line means a field-level error, so go to that line.
- `(not written in any YAML: a default or a cross-field rule)` means a cross-field validator. It
  fires on the whole model, so there is no single line, and the message names both keys and the
  values it compared.
- A bare file path with no dotted key means a pre-schema failure raised before validation ran: a
  missing file, YAML syntax, or an unresolved `${VAR}`.

**The suggestion has a caveat.** It is built from keys actually written in some YAML that share the
typo's parent prefix, not from a schema search. So the ordinary typo, the one that renames the only
spelling of a key, usually gets no suggestion at all. What always works is asking the schema instead,
with `python -m src.config explain <the key you meant>`.

The file and line come from a read-only side-car, `_provenance.py`, which re-parses the same files
with `yaml.compose()`, so a bug there can only fail to explain a value, never change one. Its
`section_sources` is a hand-kept mirror of the loader's file list, so teaching the loader a new file
and forgetting that table makes the new file's keys unexplainable.

The failures that are not self-explanatory: a duplicate model key means two base files declare the
same top-level key, and the fix is a profile overlay rather than a second base file; a profile-layer
error means the tree has no `*.<layer>.yaml` for that layer; `Extra inputs are not permitted` is a
typo, a wrong indentation level, or a renamed key. And if you edited a YAML and nothing changed, that
is the cache: call `reload_config()`.

---

## 8. The grasping presets, a schema bypass

Three files under [`config/grasping_presets/`](../../config/grasping_presets/) are not part of
`AppConfig`, not read by `load_config`, and not covered by `python -m src.config` validating green.
They are partial `grasping:` blocks merged onto an already-loaded `robot` dict, downstream, by
`apply_preset` in [`src/robot/grasping/replay/`](../../src/robot/grasping/replay/README.md). They are
also not part of your tree: the presets directory is resolved from the repository, not from `--data`,
so dropping a YAML into your own `grasping_presets/` is silently ignored. You can still apply a
shipped preset to your own cell, because `apply_preset(base, name)` accepts any base mapping.

Because the merge is outside Pydantic, a typo survives silently and simply does nothing. The check is
opt-in and you have to call it. Run it whenever you add or edit a preset:

```bash
python -c "from src.robot.grasping.replay.presets import validate_all_presets; print(validate_all_presets())"
```

The preset contents, and what each `default_mode` does to a `pick()`, belong to
[05-pick-loop.md](05-pick-loop.md), so the two pages cannot drift.

---

## 9. What a config actually reaches

Stated plainly, because it is easy to get wrong in both directions.

**`from_robot_config` has live callers.** Verify it yourself, and re-verify before repeating anything
in this section:

```bash
git grep "from_robot_config(" src datagen scripts api
```

Several classes carry a factory of that name, so read the results rather than counting them.
`AutonomousGraspService.from_robot_config` is called in `src/robot/execution/autonomous_grasp/cells.py` twice, by
`build_real_cell` and `build_rehearsal_cell`, and directly by `src/willy_sim/run_eih_pick.py` and
`datagen/rl/occupancy.py`. `Cell.from_robot_config` wraps those builders as the four ordered steps a
bench needs, and it is what
[`python -m src.robot.execution.real_cell`](../../src/robot/execution/real_cell/README.md),
`src/robot/execution/pick_run.py` and `scripts/examples/03_pick.py` drive: config, preflight, build, connect
(arm before gripper), then pick. `--rehearse` runs the whole path on a dummy arm. **None of it has
run against a physical controller**, so everything past the rehearsal is unvalidated on hardware.
Procedure: [`docs/runbooks/real_cell_first_pick.md`](../runbooks/real_cell_first_pick.md).

**The default pick is open-loop.** Perceive, generate and score candidates on a deterministic
geometric rank, safety preflight and IK, driver motion and gripper close, log. The decision gate, the
closed-loop refine, verify and recover path, fusion with its commit gate, the rerank stage, the
learned success model and the reinforcement-learning layer are all built and all default to
`enabled: false`. The simulator runners turn them on per flag in runner code rather than through
config. Check any claim of that shape against [`config/robot/robot.yaml`](../../config/robot/robot.yaml)
before repeating it:

```bash
python -c "from src.config import load_config; g=load_config().robot.grasping; print({n: getattr(getattr(g,n),'enabled',None) for n in ('fusion','decision','closed_loop','verification','recovery','success_model')})"
```

**Most simulator runners are a separate path.** They build the service via `from_components`, which
leaves `effective_config=None` and so silences the config-driven overlays unless the runner re-enables
them. Several do exactly that, re-reading `robot.grasping.*` from the loaded tree and rebuilding the
effective config themselves, while taking the mode from a CLI flag and hardcoding the attempt count.
So `grasping.default_mode` and `grasping.max_attempts` are ignored there while every sub-block is
honoured, and a YAML edit under `grasping:` does reach a live simulated pick for everything the runner
does not override.

**Record logging is opt-in and wired.** `from_robot_config` reads `grasping.record_log_path` and
stamps robot provenance, so setting it makes every `pick()` append one record, which is what the
soak, KPI and learning tools read. Per-candidate features are written only while `rl.mode` is
`rl_shadow`. Leave it null and nothing is recorded, byte-identically. Note that the shipped KPI gate
is a synthetic contract self-check: it proves the telemetry and KPI plumbing agree with themselves,
not that any grasp is good, and it says so in its own provenance block.

**Two required sections have thin live consumers.** `camera.cameras` and `camera.stereomatcher` are
required at load, and most of the schema's required fields live there, but the simulated cell authors
its cameras from `robot.sim.cameras`. A green `cam.yaml` is not evidence that a camera works.

---

## Appendix: reminders, in the order they will bite you

`where` lists only 40 keys per tier until you pass `--limit`. `explain`, `where` and `decisions`
always exit `0`, so only the bare `python -m src.config` is a gate. `--print` belongs to the bare
command and is refused on a subcommand. Use `"__null__"`, never `null`, to unset an overlay value.
Call `reload_config()` after editing a file in a live process. Unset `WILLY_ADAPTATION_OVERLAY`
before trusting an explanation. And a green `python -m src.config` says nothing about the presets.

**Adding a new field?** The rules live in [`src/config/README.md`](../../src/config/README.md), not
here: a schema field with a behaviour-preserving default, documentation on the schema field rather
than in a YAML comment, written into a YAML only if the cell is actually choosing something, then the
package README updated. New behaviour ships default-off and byte-identical.

---

*Next: [02-models.md](02-models.md), the perception models this config points at.*
