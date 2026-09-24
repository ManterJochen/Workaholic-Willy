# Runbook: standing up any UR arm

**Scope.** Going from "this stack drives a UR5e" to "this stack drives the arm I actually have". The
gripper half is [hande_gripper_bringup.md](hande_gripper_bringup.md); the physical-cell procedure is
[real_cell_first_pick.md](real_cell_first_pick.md); the model-neutral procedure, with URSim and its
traps, is [cell_bringup.md](cell_bringup.md).

**Why it exists.** Until 2026-09-09 the stack answered "which UR can I drive" in four places that had
never been compared: the config gate, the DH table, the joint limits and the sim spec registry. The
DH table held seven models, the joint limits eight, and the other two held three, so the family had
been drifting in both directions at once. All four now cover the same six keys and
[`tests/test_ur_model_family.py`](../../tests/test_ur_model_family.py) fails if they diverge again.

---

## Trigger

Any of:

- a cell is being built on a UR that is not the UR5e the base tree describes;
- an existing cell is being swapped between arms, including between a CB-series arm and its
  e-series namesake, which is the swap nothing upstream can see;
- `python -m src.robot.safety.planning --doctor` reports `no_bundle`, or a cuRobo cell refuses to
  start its planner because no evidence file measured its arm, hand, plate, placement and margin;
- cuRobo refuses to start, or loads and then fails at the first plan.

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
Those six descriptors were built under the earlier arm-and-hand names, and both drivers refuse every
one of them now: the descriptor column reads yes again for an arm once its `willy_{model}.yml` is
rebuilt (step 4 below) and the check has planned with it and a hand.

---

## Mitigate

### Standing up an arm that already has everything

Set the profile and go:

```bash
WILLY_PROFILE=ur5 python -m src.config --print          # validates the tree
python -m src.robot.safety.planning --doctor --profile ur5 --hand robotiq_2f85   # bundle, gripper, descriptor
```

The doctor prints three separate probes and they mean three different things: the arm bundle,
the gripper geometry and the cuRobo descriptor. The planning CLI takes the hand from `--hand` or
`robot.gripper.model`, and an arm profile names no hand, so without one `--check` exits 1 and the
doctor reports the descriptor MISSING, both naming `robot.gripper.model`. The descriptor itself is
`willy_{model}.yml` and carries no hand, so naming one is what the probe was missing, not a rebuild.

### Adding an arm that does not exist yet

The order matters, because each step is gated on the one before it.

1. **Add the key.** `UR_MODEL_KEYS` in `src/config/schema/robot/_ur_models.py`, a DH row in
   `safety/_ur_kinematics.py`, a joint-limit row, and a `URModelSpec` in `drivers/sim/robot_models.py`
   with reach, payload and workspace patch. `tests/test_ur_model_family.py` will tell you what is
   missing; it compares all four registries and refuses a spec whose patch reaches past its own arm.

2. **Bake the arm bundle** (no simulator, no GPU):
   ```bash
   python scripts/isaac/bake_ur_meshes_from_urdf.py <model> --write
   ```
   It reads the collision STL files of Universal Robots, pinned to one upstream commit, and places
   them through a URDF rendered from their vendored config. Before writing it compares the bake with
   the bundle already committed for that model, in both directions and link by link, and refuses a
   difference that `--expect-change=<reason>` has not named. Run it without `--write` first.

   The other reader takes the geometry out of a composed simulator articulation and needs the
   simulator:
   ```bash
   python.bat scripts/isaac/bake_ur_collision_meshes.py <model> --write
   ```
   It re-reads `ur5e` and diffs it against the committed bundle first, every time, and stops before
   writing if that drifts past 0.15 mm; measured 0.000653 mm. An arm whose asset carries no collision
   mesh cannot be read this way at all, which is what the vendor path above is for.

3. **Bake the hand** (no simulator, no GPU), only for a hand nothing has baked yet. A hand the
   catalogue does not hold, from its numbers, vendor mesh files or a USD, follows
   [your_own_gripper.md](your_own_gripper.md):
   ```bash
   python scripts/grippers/bake_gripper_variant.py robotiq_hande --write
   ```
   Same shape of gate: it reads the standalone 2F-85 and diffs it against the committed `ur5e` bundle
   before it writes anything (measured 0.05 mm against a 1.00 mm limit). A hand bundle is a hand and
   nothing else, composed onto whichever arm the cell has when the guard loads, and it carries no
   list of arms: a planner starts on an arm and hand only with a committed evidence file for the
   combination (`scripts/curobo/matrix_gate.py`), and an ik cell is decided by the exact-mesh guard
   alone.

4. **Build the cuRobo descriptor** (cuRobo sidecar interpreter), one per arm, with no hand in it:
   ```bash
   ext_deps/curobo_env/python.exe scripts/curobo/build_ur_config.py <model>
   ext_deps/curobo_env/python.exe scripts/curobo/check_ur_descriptors.py willy_<model> --hand <hand>
   ```
   The build refuses rather than writing a file that fails later: if a link the template guards has
   no spheres, nothing is written. The check is the one that matters: it loads the descriptor into
   cuRobo, adds the named hand and plans, because a yml that parses is not a robot that plans.

5. **Write the profile.** Copy the nearest `robot.<model>.yaml`. Every generated one is honest about
   what it does not know, and so should yours.

---

## Verify

Three checks, in this order, and each one asks a different question:

```bash
WILLY_PROFILE=<model> python -m src.config --print              # does the tree validate
python -m src.robot.safety.planning --doctor --profile <model> --hand <hand>   # are the three artifacts there
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
this repository's baked meshes describe the same link in the same frame.

## Rollback

Everything here is additive and reversible without touching a running cell:

- **A profile**: stop setting `WILLY_PROFILE`. The base tree is unchanged and is still a UR5e.
- **A bundle**: delete `src/robot/safety/data/{model}_collision_meshes.npz`. The guard reports
  `no_bundle` and drops to the capsule proxy, which is coarser and fail-closed, not unsafe.
- **A hand**: name the previous one in `robot.gripper.model`. The guard derives its bundle from that
  name, so there is no second key to unset, and a cell that names no hand refuses to build rather than
  planning against whatever hand the arm bundle carries; the doctor reports it `missing`.
- **A descriptor**: delete `willy_{model}.yml` from the cuRobo content directory. Nothing else reads it,
  and a cell whose descriptor is missing refuses to plan rather than loading another arm's.
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
  85 % this project treats as near-singular. Measured, in `robot_schema.py`. Write it in radians, or
  in degrees as the pendant shows it with `home_joint_positions_deg` (one of the two, never both).
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
