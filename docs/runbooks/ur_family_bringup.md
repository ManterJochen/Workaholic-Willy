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
- `python -m src.robot.safety.planning --doctor` reports `no_bundle` or
  `variant_model_mismatch`;
- cuRobo refuses to start, or raises `Link tool0 not found in parent map` at the first plan.

## Diagnose

### What each arm can do today

| model | arm bundle | Hand-E variant | cuRobo descriptor | config profile |
|---|---|---|---|---|
| `ur3`   | yes | yes | yes | `robot.ur3.yaml` (derived) |
| `ur3e`  | yes | yes | yes | `robot.ur3e.yaml` (**measured cell**) |
| `ur5`   | yes | yes | yes | `robot.ur5.yaml` (derived) |
| `ur5e`  | yes | yes | yes | the base tree (**measured cell**) |
| `ur10`  | yes (from its URDF) | yes | yes | `robot.ur10.yaml` (derived) |
| `ur10e` | yes | yes | yes | `robot.ur10e.yaml` (derived) |

Every "yes" was measured by loading it, not by finding the file:
`ext_deps/curobo_env/python.exe scripts/curobo/check_ur_descriptors.py` builds a planner for each
descriptor and plans a real motion with it. **Six of six plan, 21 waypoints each**, 2.7 s to 4.6 s.

### ⚠ ur10 needed four repairs the other five did not

It works now, and everything below is what it took. An operator who reaches for a UR10 should know
this, because every one of these presented as a property of the arm and was not.

1. **It ships as `ur10_robot.urdf`**, where all five siblings ship `{model}.urdf`.
2. **That file has NO GEOMETRY AT ALL** — 74 lines, 0 `<collision>`, 0 `<visual>`, 0 mesh
   references — and it belongs to a different link-frame family than Isaac's own sphere map for the
   same robot. Pairing them puts 23 of 30 sphere centres off the arm, worst 341.5 mm, fail-open and
   silent. The builder now picks by a rule instead: *a description with no geometry describes no
   body*, which excludes it without naming ur10 and changes nothing for the other five.
3. **Its USD collides the whole arm with thirteen cylinders**, so the Isaac bake cannot read it. But
   Isaac's URDF-importer package ships seven `.obj` link meshes for the same robot, and
   `scripts/isaac/bake_ur_meshes_from_urdf.py` bakes from those with no simulator at all. It gates
   itself by baking ur10e through the same path and diffing against the bundle Isaac produced:
   **0.000 mm**, two different readers, one answer.
4. **Its Lula sphere map is wrong about all three wrists.** Measured: their spheres sit about 61 mm
   from where the geometry is. Keeping them alongside the correct ones made each wrist a body twice
   its size spanning two positions, and cuRobo returned None from every plan — including a plan from
   a pose to itself — while loading perfectly. The builder drops a sphere whose centre lies further
   outside its own link mesh than its own radius, which is where a sphere and a body stop
   intersecting rather than a tuned threshold.

⛔ **AND ONE THAT WAS NEVER TRUE.** From 2026-09-09 to 2026-09-10 this runbook said cuRobo could not
load the importer description. It could not, because of **one stray `)` in an `xyz` attribute** on
line 28 of Isaac's file. A one-character parse failure had been written down as a capability of the
robot. That is worse than a check that stays silent: it says something plausible and wrong, and the
plausible thing gets believed. The builder repairs the character in its own copy and says so, and
never touches Isaac's tree.

### What the ur10 still does differently

Its collision bundle is baked from CONVEX HULLS of its visual meshes, because it declares no
collision mesh anywhere. That is not a compromise, and the reason is a control on an arm where both
halves exist: ur10e ships visual *and* collision geometry, its own collision meshes are **1.259x**
the volume of its visuals and every one of them is **exactly convex**. The ur10 hulls come out at
1.357x with the same per-link pattern — eight percentage points more conservative than what the
vendor ships, which is the correct direction for a fail-closed guard.

Its `shoulder_link` spheres are AUTHORED, because Isaac's map has none for that link. They come from
the two collision cylinders the description declares, and they cover **47.8 %** of that link's
surface against **31.4 %** for ur10e's own shoulder and 10.2 % for the vendor's weakest link.

**A `ur10` cell plans like any other now**, with exact mesh geometry, its own Hand-E variant and
a descriptor proved by planning rather than by existing.

---

## Mitigate

### Standing up an arm that already has everything

Set the profile and go:

```bash
WILLY_PROFILE=ur5 python -m backend.config --print          # validates the tree
python -m src.robot.safety.planning --doctor        # bundle, gripper, descriptor
```

The doctor prints three separate probes and they mean three different things: the arm bundle,
the gripper geometry and the cuRobo descriptor. All three read `ok` on all six models today.

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

   If it refuses with "carries N collision prims and not one of them is a mesh", that arm's USD has
   no mesh to read. That is not the end: try the URDF path instead, which needs no simulator at all
   and gates itself the same way.
   ```bash
   python scripts/isaac/bake_ur_meshes_from_urdf.py <model> --write
   ```
   It bakes `ur10e` from its own collision files through the same code first and diffs against the
   bundle Isaac produced; measured 0.000 mm. That is how `ur10` got its geometry.

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
