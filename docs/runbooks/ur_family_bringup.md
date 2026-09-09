# Runbook: standing up any UR from a UR3 to a UR10

**Scope.** Going from "this stack drives a UR5e" to "this stack drives the arm I actually have". The
gripper half is [hande_gripper_bringup.md](hande_gripper_bringup.md); the physical-cell procedure is
[real_cell_first_pick.md](real_cell_first_pick.md); the worked UR3e example is
[ur3e_cell_bringup.md](ur3e_cell_bringup.md).

**Why it exists.** Until 2026-09-09 the stack answered "which UR can I drive" in four places that had
never been compared: the config gate, the DH table, the joint limits and the sim spec registry. The
DH table held seven models, the joint limits eight, and the other two held three. So `ur10` had
correct kinematics and could not be configured at all, and the family had been drifting in both
directions at once. All four now cover the same six keys and
[`tests/test_ur_model_family.py`](../../tests/test_ur_model_family.py) fails if they diverge again.

---

## Trigger

Any of:

- a cell is being built on a UR that is not the UR5e the base tree describes;
- an existing cell is being swapped between arms, including between a CB-series arm and its
  e-series namesake, which is the swap nothing upstream can see;
- `python -m src.robot.safety.planning --doctor` reports `no_bundle`, `primitive_colliders`
  or `variant_model_mismatch`;
- cuRobo refuses to start, or raises `Link tool0 not found in parent map` at the first plan.

## Diagnose

### What each arm can do today

| model | arm bundle | Hand-E variant | cuRobo descriptor | config profile |
|---|---|---|---|---|
| `ur3`   | yes | yes | yes | `robot.ur3.yaml` (derived) |
| `ur3e`  | yes | yes | yes | `robot.ur3e.yaml` (**measured cell**) |
| `ur5`   | yes | yes | yes | `robot.ur5.yaml` (derived) |
| `ur5e`  | yes | yes | yes | the base tree (**measured cell**) |
| `ur10`  | **no, and never** | no | **no** | `robot.ur10.yaml` (derived) |
| `ur10e` | yes | yes | yes | `robot.ur10e.yaml` (derived) |

Every "yes" was measured by loading it, not by finding the file:
`ext_deps/curobo_env/python.exe scripts/curobo/check_ur_descriptors.py` builds a planner for each
descriptor and plans a real motion with it. Five of five plan, 21 waypoints each.

### ⛔ ur10 is different in four ways, and three of them are its asset

An operator who reaches for a UR10 should know this before ordering one, so it is stated plainly
rather than left in a log:

1. **It ships as `ur10_robot.urdf`**, where all five siblings ship `{model}.urdf`. Handled.
2. **Its description has no `tool0`** — it is the older `ur_description` generation, with `world` and
   `ee_link` and no `flange`. cuRobo names `tool0` as its end effector, so a ur10 built without this
   loads fine and dies at the first plan. Handled: the fixed `wrist_3 -> flange -> tool0` chain is
   transcribed from a sibling, gated on the two `wrist_3` frames agreeing (measured 1.0e-07).
3. **Its Isaac asset collides the whole arm with thirteen cylinders**, with no collision mesh
   anywhere. So it gets no exact-mesh bundle, reports `primitive_colliders` rather than `no_bundle`,
   and plans against the capsule proxy permanently. That is not a missing file and no bake will
   produce one. Do not substitute the visual meshes: they are 31k to 65k vertices and are render
   assets, and a file named `*_collision_meshes.npz` promises exact geometry.
4. **Isaac ships ur10 in TWO incompatible link-frame families, and the obvious pairing is the wrong
   one.** This is what stops it, and it is a defect in Isaac's shipped files rather than in this
   repository. The ur10 Lula sphere map belongs to the **+z family**. `ur10_robot.urdf`, which sits
   in the same directory, and cuRobo's own shipped `ur_description/ur10.urdf` are both **-x family**.
   Every other UR on the box is -x on both sides, so ur10 is the single model where pairing the
   nearest sphere map with the nearest URDF is silently wrong.

   Measured 2026-09-09 through the full FK chain, as the signed distance of each sphere centre to the
   arm body, with two pairings that work today as calibration:

   | pairing | centres outside the arm | worst |
   |---|---|---|
   | sphere map + cuRobo's `ur10.urdf` (**the obvious one**) | 23 of 30 | **+341.5 mm** |
   | sphere map + the importer `ur10.urdf` | 6 of 30 | +38.3 mm |
   | *ur10e, which works today* | 5 of 33 | +45.3 mm |
   | *ur5e, which works today* | 4 of 39 | +51.0 mm |

   ⛔ **Read the calibration rows first.** 4 of 39 on a ur5e and 5 of 33 on a ur10e are what a
   CORRECT pairing looks like, so 23 of 30 is not a worse fit, it is a different arm. That is
   FAIL-OPEN by a third of a metre: a planner on the obvious pairing models the arm where it is
   not and leaves unguarded the space where it is, and nothing raises. `build_ur_config.py`
   refuses to write a ur10 descriptor and prints all of this when it does.

### What a real ur10 fix would have to author

The right frame family exists, so this is not hopeless, but it is four pieces of work rather than a
switch. All four were verified by seventeen independent agents on 2026-09-09, and the three "cheap"
claims among them were each measured and refuted:

1. **A loadable URDF.** The importer URDF
   (`isaacsim.asset.importer.urdf/data/urdf/robots/ur10/urdf/ur10.urdf`) is the correct frame family
   and carries real collision cylinders, but cuRobo's parser (`yourdfpy`) refuses to load it. The
   frame-identical `ur10_robot_suction.urdf` loads and carries `tool0`, and all 14 of its mesh
   references are missing from disk. So the URDF has to be composed: frames from one, geometry from
   the other.
2. **A `tool0`.** The importer URDF has none, and the transcription gate in `build_ur_config.py`
   **correctly refuses** to graft a sibling's chain onto it (its `wrist_3` rotation differs from the
   donor by 1.0 against a 1e-6 tolerance). The correct transform is measurable rather than
   remembered: relative to the importer URDF's `wrist_3_link` it is `xyz [0, 0.0922, 0]`,
   `rpy (-pi/2, 0, 0)`, constant over 300 random configurations.
3. **`shoulder_link` spheres.** Neither Lula file has any. `spheres_from_primitive_colliders.py`
   reads the USD and not a URDF, so it does not do this job as written. Naive spheres of the
   cylinder's own radius under-cover: 12151 of 20000 sampled surface points fall outside, worst gap
   5.2 mm, which is a thin false-CLEAR shell no current test would catch.
4. **A collision-mesh bundle**, baked from the importer URDF's frames, so the repository's own guard
   geometry and the cuRobo descriptor cannot drift apart.

⚠ **And none of that is proof.** Every measurement above is a static-file frame proof. The
load-bearing check is a cuRobo self-collision run on the assembled descriptor, which nothing here
has done.

**A `ur10` cell is therefore usable for config, kinematics, joint limits, workspace and the sim
asset, and is not cuRobo-plannable.** If you need a 1300 mm arm that plans, use the `ur10e`.

---

## Mitigate

### Standing up an arm that already has everything

Set the profile and go:

```bash
WILLY_PROFILE=ur5 python -m backend.config --print          # validates the tree
python -m src.robot.safety.planning --doctor        # bundle, gripper, descriptor
```

The doctor prints three separate probes, and they mean three different things. `collision mesh
bundle` reads `ok`, `missing` (bake one) or a `warn` naming the asset (nothing to bake, ever).

### Adding an arm that does not exist yet

The order matters, because each step is gated on the one before it.

1. **Add the key.** `UR_MODEL_KEYS` in `src/config/schema/robot/_ur_models.py`, a DH row in
   `safety/_ur_kinematics.py`, a joint-limit row, and a `URModelSpec` in `drivers/sim/robot_models.py`
   with reach, payload and workspace patch. `tests/test_ur_model_family.py` will tell you what is
   missing; it compares all four registries and refuses a spec whose patch reaches past its own arm.

2. **Bake the arm bundle** (needs Isaac):
   ```bash
   python.bat scripts/isaac/bake_ur_collision_meshes.py <model> --write
   ```
   It re-reads `ur5e` and diffs it against the committed bundle FIRST, every time, and stops before
   writing if that drifts past 0.15 mm. Measured on the day the four new arms were baked: 0.000653 mm.
   If it refuses with "carries N collision prims and not one of them is a mesh", that arm has no
   exact geometry and never will.

3. **Bake the gripper variant** (no Isaac, no GPU):
   ```bash
   python scripts/grippers/bake_gripper_variant.py robotiq_hande --arm <model> --write
   ```
   Same shape of gate: it reads the standalone 2F-85 and diffs it against the committed `ur5e` bundle
   before it writes anything (measured 0.05 mm against a 1.00 mm limit). A variant bundle is an ARM
   PLUS A HAND, so every arm needs its own, and a mismatch drops the whole cell to capsules.

4. **Build the cuRobo descriptor** (cuRobo sidecar interpreter):
   ```bash
   ext_deps/curobo_env/python.exe scripts/curobo/build_ur_config.py <model>
   ext_deps/curobo_env/python.exe scripts/curobo/check_ur_descriptors.py <model>
   ```
   The build refuses rather than writing a file that fails later: if a link the template guards has
   no spheres, nothing is written. The check is the one that matters — it loads the descriptor into
   cuRobo and plans, because a yml that parses is not a robot that plans.

5. **Write the profile.** Copy the nearest `robot.<model>.yaml`. Every generated one is honest about
   what it does not know, and so should yours.

---

## Verify

Three checks, in this order, and each one asks a different question:

```bash
WILLY_PROFILE=<model> python -m backend.config --print              # does the tree validate
python -m src.robot.safety.planning --doctor                 # are the three artifacts there
ext_deps/curobo_env/python.exe scripts/curobo/check_ur_descriptors.py  # does it actually PLAN
```

The third is the one that cannot be faked. A descriptor that parses is not a robot cuRobo can plan
for, and every failure of that kind lands at the first motion rather than at load. Expect one line
per model, each planning 21 waypoints.

On the box, after any change to a bundle or a bake recipe, also run:

```bash
python -m pytest tests/test_ur_model_family.py tests/test_arm_profiles.py \
                 tests/test_lula_and_bundle_agree.py tests/test_ur_series_twin.py
```

`test_lula_and_bundle_agree.py` is the one worth understanding: it checks that Isaac's sphere map and
this repository's baked meshes describe the same link in the same frame. That is what ur10 fails.

## Rollback

Everything here is additive and reversible without touching a running cell:

- **A profile**: stop setting `WILLY_PROFILE`. The base tree is unchanged and is still a UR5e.
- **A bundle**: delete `src/robot/safety/data/{model}_collision_meshes.npz`. The guard reports
  `no_bundle` and drops to the capsule proxy, which is coarser and fail-closed, not unsafe.
- **A gripper variant**: unset `safety.self_collision.collision_mesh_variant`. The cell then plans
  against whatever hand the arm bundle carries, and the doctor says so in a `warn`.
- **A descriptor**: delete `{model}.yml` from the cuRobo content directory. Nothing else reads it.
  The shared scaffolding lives in `_ur_template.yml`, which builds never overwrite.

No step here can leave a cell in a state where it plans against wrong geometry: every artifact is
either present and gated, or absent and reported.

## ⚠ What a derived profile does NOT know

The four profiles generated on 2026-09-09 (`ur3`, `ur5`, `ur10`, `ur10e`) carry the model name three
times, the datasheet payload, and a workspace box. **The box is a reach bound, not a cell.** It is the
largest box the arm can serve, with its far corner at the same fraction of reach the one standing
UR3e cell actually uses, and it is symmetric about the base because a preferred direction cannot be
derived. Your cell is a room: it has a table, a wall and one direction you work in.

These are **deliberately absent** and inherit a tree authored for a UR5e:

- `home_joint_positions` — the UR5e default puts a UR3e grasp centre at 93.4 % of its reach, past the
  85 % this project treats as near-singular. Measured, in `robot_schema.py`.
- `park_joint_positions`, the camera poses, the scene layout.

Set them from your own cell before a real pick. [real_cell_first_pick.md](real_cell_first_pick.md)
is the procedure.

## ⚠ The CB-series trap, and what catches it

A UR3 and a UR3e report the SAME model string to a controller dashboard (URSim answers `UR3` for a
UR3e, measured 2026-08-19), so `ur.model` cannot be verified by asking the controller what it is.
Measured over 20000 random joint vectors: a `ur3`/`ur3e` swap moves the flange by as little as
8.50 mm and sits **under the default 10 mm tool-frame tolerance in 12.2 % of poses**. The `ur5`/`ur5e`
and `ur10`/`ur10e` pairs never do (59.17 mm and 28.43 mm minimum).

So the driver adds a RELATIVE check at connect, which no tolerance can swallow: it derives the
controller's active tool frame through both DH tables and refuses when the twin explains the
controller better than the configured model does. A correct cell reads about 0 mm for itself and
8.5 mm or more for its twin.

**If you see that refusal, check `ur.model` before you check the pendant** — but the message names
both, because a genuinely wrong TCP can also produce it.

## Known gap

There is **no validator tying `workspace_limits` to the configured model.** A cell that sets
`ur.model: ur3` and keeps the base box declares a corner 707 mm from the shoulder on an arm that
reaches 500. The profiles above are what keeps you out of it, and nothing refuses a hand-written tree
that walks back in. `tests/test_arm_profiles.py` checks the shipped profiles; it cannot check yours.
