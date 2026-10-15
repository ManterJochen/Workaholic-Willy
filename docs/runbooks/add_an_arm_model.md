# Runbook: adding an arm model

**Scope.** Taking an arm this stack does not describe yet, or one it describes and has not admitted, to an arm a
cell may name and a planner starts on: the description, the build under `scripts/arms/`, the judged retract,
admission, the evidence gate and a profile. The gripper half is [your_own_gripper.md](your_own_gripper.md), and the
Hand-E has its own, [hande_gripper_bringup.md](hande_gripper_bringup.md); the physical-cell procedure is
[real_cell_first_pick.md](real_cell_first_pick.md); the model-neutral procedure, with URSim and its traps, is
[cell_bringup.md](cell_bringup.md).

**Why it exists.** Until 2026-09-09 the stack answered "which UR can I drive" in four places that had never been
compared: the config gate, the DH table, the joint limits and the sim spec registry, and the family had been
drifting in both directions at once. Now an arm is one file, `config/arms/<model>.yaml`, and every reader asks it:
the joints, the DH rows, the limits, the home, the planner limits, the links and the base turns. The build under
`scripts/arms/` takes the arm by its model name and reads that file too.

**What still assumes a UR.** The meshes come through the one source the build has, Universal Robots' pinned ROS 2
description (`scripts/arms/_ur_description.py`), held by a manifest of UR meshes only
(`scripts/arms/ur_meshes.sha256`). The descriptor starts from cuRobo's UR template and writes into its
`ur_description` assets, `scripts/arms/check.py` finds `willy_ur*.yml` only, and the bake copies the 2F-85's arrays
from the ur5e bundle. An arm of another vendor needs its own source and those places first; the stages and their
order stay as they are.

---

## Trigger

Any of:

- a cell is being built on an arm the base tree does not describe (the base tree is a UR5e), or on a described
  arm that is not admitted;
- an existing cell is being swapped between arms, including between a UR CB-series arm and its e-series
  namesake, which is the swap nothing upstream can see (below);
- `python -m src.robot.safety.planning --doctor` reports the arm's collision mesh bundle missing (`no_bundle`), a
  missing retract row or missing combination evidence, or a cuRobo cell refuses to start its planner because no
  evidence file measured its arm, hand, plate, placement and margin;
- cuRobo refuses to start, or loads and then fails at the first plan;
- an arm's description changed (a joint row, a limit, a link), so everything built from it is built again.

## Diagnose

### What each described arm has today

| model | admitted | arm bundle | arm sphere map | descriptor | retract rows | evidence files | config profile |
|---|---|---|---|---|---|---|---|
| `ur3`   | yes | yes | yes | yes | 8 | 8 | `robot.ur3.yaml` (derived) |
| `ur3e`  | yes | yes | yes | yes | 8 | 8 | `robot.ur3e.yaml` (**measured cell**) |
| `ur5`   | yes | yes | yes | yes | 7 | 7 | `robot.ur5.yaml` (derived) |
| `ur5e`  | yes | yes | yes | yes | 8 | 8 | the base tree (**measured cell**) |
| `ur10`  | yes | yes | yes | yes | 11 | 18 | `robot.ur10.yaml` (derived, with the cable window) |
| `ur10e` | yes | yes | yes | yes | 8 | 8 | `robot.ur10e.yaml` (derived) |
| `ur16e` | no  | yes | yes | yes | 4 | none | none |
| `ur20`  | no  | no  | no  | no  | none | none | none |

The ur10 holds more than the others because it is the owner's cell: three rows for the Hand-E on its 23 mm plate at
+Z-Y, at planner margins of 0, 3 and 4 mm, each with its evidence file, and seven files for a hand that carries a part
(the `_a16` files, sixteen sphere slots reserved for the payload). The ur5 has one row less: at +Y+X the chooser's
planner could not be asked about the EGU-50 on it, and the table's rule records why (`pairs_with_no_retract`).

Every "yes" was measured by loading it, not by finding the file:

- `ext_deps/curobo_env/python.exe scripts/arms/check.py --hand robotiq_2f85` loaded all seven descriptors into
  cuRobo with the 2F-85 added and planned 21 waypoints with each, on 2026-10-14 (P0f F10): 4.2 s to 10.8 s each.
- The evidence files were measured again the same day, all 57 of them, `scripts/arms/gate.py` once per file with
  that file's own combination, and every count and hash came back as committed. The four ur16e rows cannot be
  measured, because a cell may not name an arm that is not admitted and the gate builds a cell.
- `scripts/arms/retract.py --check` judged the seven arms' pairs at +Y+X again the same day and found the table as
  committed.
- `tests/test_arm_bundles_reproduce_from_pinned_stl.py` bakes every arm's links again from the pinned meshes, where
  they are installed, and compares them array for array.

ur20 is described and nothing else: no pinned meshes, no bundle, no fit, no descriptor and no retract anchor, so
`install.ps1` builds descriptors for the other seven only.

---

## Mitigate

### Standing up an arm that already has everything

Set the profile and go:

```bash
WILLY_PROFILE=ur5 python -m src.config --print                                    # validates the tree
python -m src.robot.safety.planning --doctor --profile ur5,<your cell layers>     # reads the whole cell
```

The doctor prints its probes as separate rows, and they mean different things: the arm bundle, the gripper
geometry, the cuRobo descriptor, and on a cell whose planner is `curobo` the retract row and the combination
evidence for that cell's hand, plates, placement and planner margin. Those last two read a cell, so give the doctor
the cell's whole profile chain: the arm layer, a hand layer such as `hande` (`robot.gripper.model`, its plates and
tool frame), and a layer that declares `safety.self_collision.planner_margin_mm`, as `robot.ursim_curobo.yaml`
does. An arm profile alone declares neither a hand nor a margin, and the base tree names no hand: the doctor then
reads the bundle, the hand `--hand` names and the descriptor, reports the retract row and the evidence as missing
with what the cell has not declared, and exits 1. Once the cell declares them, a missing retract row or evidence
file names the `scripts/arms/retract.py` or `scripts/arms/gate.py` command that measures it, and a pair the rule
found no retract for says so instead.

### Adding an arm model

The order matters, because each step is gated on the one before it.

1. **Describe the arm.** One file, `config/arms/<model>.yaml`, copied from the nearest arm: its joints in metres
   from Universal Robots' pinned description (`scripts/arms/ur_ros2_description`), each joint's envelope and home
   angle, the datasheet reach and payload, and a `sim` block with the simulator asset and a workspace patch if the
   simulator ships the arm. Leave `admitted: false` until step 4. Keep the `upstream` block the copied arm carries:
   every UR arm is read from the same pin, and `tests/test_the_description_names_its_upstream.py` refuses a block
   that differs from `scripts/arms/fetch.py`. Each joint row names its `link` and bundle `part` as UR's description
   spells them; the frame is the row, and the flange is the last row. `base` stays `base_link` with both turns at
   180, which is UR's own macro for every model. Each row's `planner_limit_deg` is UR's `joint_limits.yaml` (the
   elbow +/-180), and `planner` keeps the 0.1 rad clip. An arm the simulator ships a Lula config for takes its
   `default_q` as `retract_anchor_rad` on every joint, `0.0` for the simulator's `-0.0`, with
   `retract_anchor_source` naming that file. An arm with none leaves both out, and then neither the seed build nor
   the retract chooser of step 3 has a pose to start from, so the arm cannot be judged (ur20 describes no anchor).
   `tests/test_arm_descriptions_are_urs_numbers.py` holds the numbers to UR's, `tests/test_ur_model_family.py`
   refuses a patch that reaches past its own arm, and `tests/test_arm_registry.py` lists every described arm by
   hand (`_DESCRIBED`), so add the new one there.

2. **Build it.** One command runs every stage in order, each under the interpreter it needs, and stops at the first
   that fails:
   ```bash
   python scripts/arms/build_arm.py <model>                          # verify: writes nothing
   python scripts/arms/build_arm.py <model> --write --hand <hand>    # fetch, bake, fit, the descriptor, the check
   ```
   A new arm stops at the descriptor, which refuses an arm with no judged retract row: step 3 is the way round, and
   then the same command runs to the end. The stages, for when one of them stops the build and has to be run by
   itself:
   - **fetch** (cuRobo interpreter) puts Universal Robots' meshes into the cuRobo asset root and holds each to its
     pin in `scripts/arms/ur_meshes.sha256`: `ext_deps/curobo_env/python.exe scripts/arms/fetch.py`. It fetches
     only what the manifest pins and has no mode that writes a pin. A model the manifest does not hold needs its
     rows first, each mesh hashed as UR's pinned commit holds it; until then the bake refuses the arm for having no
     pinned collision mesh.
   - **bake** writes the arm bundle `src/robot/safety/data/<model>_collision_meshes.npz` from UR's own collision
     STL files through the rendered description, with no simulator and no GPU:
     `python scripts/arms/bake.py <model> --write`. It compares the result with the committed bundle where there is
     one, in both directions and link by link, and refuses a difference that `--expect-change=<reason>` has not
     named; a rebake with the committed arrays leaves the file as it is. Run it without `--write` first. A new arm
     has nothing to compare with, and the bake says so. Record its row in `bundles.json` beside it by hand, copied
     from the nearest arm's (`model`, `source`, `upstream`, `note`, and the `arrays_sha256` that
     `src/robot/safety/planning/bundle_index.py` computes): no writer records an arm row,
     `tests/test_bundle_provenance.py` refuses a bundle nobody recorded, and the rebake test skips a row without
     `model`.
   - **fit** writes the arm's cover-fit sphere map `src/robot/safety/planning/robot/<model>_arm_spheres.yml`:
     `python scripts/arms/fit.py --arm <model> --write`. A refit of the same bundle writes the same file byte for
     byte.
   - **descriptor** (cuRobo interpreter) installs `willy_<model>.yml` and `<model>.urdf` into the ignored cuRobo
     content directory, one per arm with no hand in it:
     `ext_deps/curobo_env/python.exe scripts/arms/descriptor.py <model>`. It refuses an arm with no committed fit
     and an arm with no judged retract row, and if a link the template guards has no spheres it writes nothing.
   - **check** (cuRobo interpreter) loads the descriptor with the hand added as a body link and plans, because a yml
     that parses is not a robot that plans:
     `ext_deps/curobo_env/python.exe scripts/arms/check.py willy_<model> --hand <hand>`. `build_arm.py` runs it
     when it is given `--hand`, or a `--profile` whose cell names a hand.
   - **doctor** runs when `build_arm.py` is given `--profile`, which a new arm has only after steps 4 and 6.

   The descriptor carries no hand: the planner adds the cell's hand as a body link when it starts. The arm bundle
   carries the 2F-85's arrays, copied from the ur5e bundle, and the guard puts the cell's own hand in their place
   when it composes the arm. A hand nothing has baked yet follows [your_own_gripper.md](your_own_gripper.md). A
   hand bundle is a hand and nothing else and carries no list of arms: which arms it may go on is the evidence of
   step 5.

3. **Judge the retract.** `src/robot/safety/planning/robot/retract.yaml` holds one row per arm, hand, plate,
   placement and planner margin, each chosen by `scripts/arms/retract.py` on the exact meshes, and each row names
   the bundles it was judged on. A cell's driver reads the row of its own combination and refuses before a sidecar
   starts when there is none; the descriptor's `default_q` is the 2F-85's row at a bare flange at the rule's
   placement. The chooser asks a real planner, which needs a descriptor, so a seed build breaks that circle by
   writing the arm's anchor and saying so in its provenance:
   ```bash
   ext_deps/curobo_env/python.exe scripts/arms/descriptor.py <model> --retract-from-anchor   # a seed, unjudged
   python scripts/arms/retract.py <model>                                    # every registry hand, +Y+X (the simulator's)
   python scripts/arms/retract.py <model> --tool-rotation-xyzw 0 0 0 1       # the same at +Z+X, as real cells declare
   python scripts/arms/build_arm.py <model> --write --hand <hand>             # the judged descriptor, then the check
   ```
   Each run judges every registry hand, a hand with a mounting face at the shipped 0 and 20 mm plates, at the one
   placement it is given, and writes into the table rather than over it: every other row stays. A cell with other
   plates names them (`--plate-mm`), as the owner's UR10 did for the 23 mm plate under its Hand-E, and its doctor
   prints the whole command. A seed descriptor is never what a cell holds.

4. **Admit the arm.** Set `admitted: true` in its description. `tests/test_arm_registry.py` refuses that while the
   bundle or the sphere map is missing, and it writes out the admitted arms by hand
   (`test_the_admitted_arms_are_the_six_the_owner_admitted`): admitting an arm is the owner's decision, and that
   test is where it is recorded. Until the arm is admitted a cell may not name it at all: the config refuses it at
   load, and so does every command below that builds a cell.

5. **Measure each combination a cell will run.** A planner starts on an arm and hand only with a committed evidence
   file for the combination: the arm, the hand, its plates, its placement and the planner margin. The doctor of a
   cell names the whole command:
   ```bash
   python scripts/arms/gate.py --arm <model> --hand <hand> --planner-margin-mm 4 --write      # +Y+X, no plate
   python scripts/arms/gate.py --arm <model> --hand <hand> --planner-margin-mm 4 \
       --tool-rotation-xyzw 0 0 0 1 --write                                                   # +Z+X
   python scripts/arms/sweep.py --write                                                       # every table row
   ```
   A cell with plates adds `--plate` for each, and a cell that carries a part adds `--attach <slots>`; the sweep
   measures without a payload. The sweep exits 1 while the table holds rows of an arm that is not admitted (ur16e's
   four today) and writes the files of every row it could measure.

6. **Write the profile.** Copy the nearest `robot.<model>.yaml`. Every generated one is honest about what it does
   not know (below), and so should yours.

---

## Verify

Three checks, in this order, and each one asks a different question:

```bash
WILLY_PROFILE=<model> python -m src.config --print                                    # does the tree validate
python -m src.robot.safety.planning --doctor --profile <model>,<your cell layers>    # is everything the cell needs there
ext_deps/curobo_env/python.exe scripts/arms/check.py --hand <hand>                     # does it actually PLAN
```

The third is the one that cannot be faked. A descriptor that parses is not a robot cuRobo can plan for, and every
failure of that kind lands at the first motion rather than at load. Expect one line per descriptor, each planning
21 waypoints.

On the box, after any change to a description, a bundle or a bake recipe, also run:

```bash
python scripts/arms/retract.py --check        # exits 1 when the committed table is not what this box measures
python -m pytest tests/test_arm_registry.py tests/test_ur_model_family.py tests/test_arm_profiles.py \
                 tests/test_ur_series_twin.py tests/test_the_committed_bundles_hold_their_arrays.py \
                 tests/test_the_committed_evidence_recomputes.py
```

The last two tests are the ones worth understanding: every bundle has to be the arrays its index records, and every
planner evidence file has to recompute from the committed files and the descriptor installed on this box, so a
changed bundle says which evidence it moved.

## Rollback

Everything here is additive and reversible without touching a running cell:

- **A profile**: stop setting `WILLY_PROFILE`. The base tree is unchanged and is still a UR5e.
- **An admission**: set `admitted: false` again. A cell naming the arm is refused at load.
- **A bundle**: delete `src/robot/safety/data/<model>_collision_meshes.npz` and its `bundles.json` row. The guard
  reports `no_bundle` and drops to the capsule proxy, which is coarser and fail-closed, not unsafe.
- **A hand**: name the previous one in `robot.gripper.model`. The guard derives its bundle from that name, so there
  is no second key to unset, and a cell that names no hand refuses to build rather than planning against whatever
  hand the arm bundle carries; the doctor reports it `missing`.
- **A descriptor**: delete `willy_<model>.yml` from the cuRobo content directory. Nothing else reads it, and a cell
  whose descriptor is missing refuses to plan rather than loading another arm's. The shared scaffolding lives in
  `_ur_template.yml`, which builds never overwrite.
- **A retract row or an evidence file**: revert it through version control. A cell whose row or file is gone refuses
  to start its planner, and its doctor names the command that measures it again.

No step here can leave a cell in a state where it plans against wrong geometry: every artifact is either present
and gated, or absent and reported.

## ⚠ What a derived profile does NOT know

The four profiles generated on 2026-09-09 (`ur3`, `ur5`, `ur10`, `ur10e`) carry the model name three times, the
datasheet payload and a workspace box; `ur10` also keeps every full-turn joint within half a turn of home
(`safety.joint_limits.within_half_turn_of_home`), the cable window of the owner's UR10 cell. **The box is a reach
bound, not a cell.** It is the largest box the arm can serve, with its far corner at the same fraction of reach the
one standing UR3e cell actually uses, and it is symmetric about the base because a preferred direction cannot be
derived. Your cell is a room: it has a table, a wall and one direction you work in.

These are **deliberately absent** and inherit a tree authored for a UR5e:

- `home_joint_positions`: left unset, the home is the one the arm's description states, which for every UR is the
  pose authored for a UR5e. It puts a UR3e grasp centre at 93.4 % of its reach, past the 85 % this project treats
  as near-singular (measured, in `robot_schema.py`), and a home whose grasp centre lies outside `workspace_limits`
  is refused before the arm moves. Write it in radians, or in degrees as the pendant shows it with
  `home_joint_positions_deg` (one of the two, never both).
- `park_joint_positions`, the camera poses and a wrist camera's looks (`robot.look_joint_positions_deg`, in
  degrees), the scene layout.
- `robot.natural_closing_axis`, how the hand and its camera naturally stand: a name (`"-y"` on the owner's UR10,
  tool +X along base -y, where its wrist D415 image stands upright) or a taught pose's `quaternion_xyzw`, of which
  only the heading of its tool +X counts. Every camera grasp and every push then closes the way round nearer it,
  none left out or tilted, and `robot.tool_down` builds fixed poses along it. A program's `closing_axis` wins,
  beside `align_closing_to_base_x` no grasp is turned (a push still takes the natural way round), a value that names
  no axis is refused at load, and reading it loads no grasping package. Unset in every shipped profile: the cell
  behaves as before.

Set them from your own cell before a real pick. [real_cell_first_pick.md](real_cell_first_pick.md) is the
procedure.

## ⚠ The CB-series trap, and what catches it

A UR3 and a UR3e report the SAME model string to a controller dashboard (URSim answers `UR3` for a UR3e, measured
2026-08-19), so `ur.model` cannot be verified by asking the controller what it is. Measured over 20000 random joint
vectors: a `ur3`/`ur3e` swap moves the flange by as little as 8.50 mm and sits **under the default 10 mm tool-frame
tolerance in 12.2 % of poses**. The `ur5`/`ur5e` and `ur10`/`ur10e` pairs never do (59.17 mm and 28.43 mm minimum).
`ur16e` has no twin: no CB-series UR16 is described.

So the driver adds a RELATIVE check at connect, which no tolerance can swallow: it derives the controller's active
tool frame through the DH rows of both descriptions and refuses when the twin explains the controller better than
the configured model does. A correct cell reads about 0 mm for itself and 8.5 mm or more for its twin.

**If you see that refusal, check `ur.model` before you check the pendant**, but the message names both, because a
genuinely wrong TCP can also produce it.

## Known gap

There is **no validator tying `workspace_limits` to the configured model.** A cell that sets `ur.model: ur3` and
keeps the base box declares a far corner 837 mm from the shoulder on an arm that reaches 500 (`robot.ur3.yaml`
measures it). The profiles above are what keeps you out of it, and nothing refuses a hand-written tree that walks
back in. `tests/test_arm_profiles.py` checks the shipped profiles; it cannot check yours.
