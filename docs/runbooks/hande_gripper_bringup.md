# Runbook: putting a Robotiq Hand-E on the cell

**Scope.** Swapping the end effector, from an unpacked gripper to a cell whose planner, whose
collision guard, whose grasp calculator and whose driver all model the hand that is actually bolted
on. It is the gripper half; the physical-arm procedure is
[real_cell_first_pick.md](real_cell_first_pick.md) and the robot half is
[ur3e_cell_bringup.md](ur3e_cell_bringup.md).

**Why it exists.** A cell describes its gripper three times, in three places, answering three
different questions, and until 2026-09-08 no validator named more than one of them:

| where | question it answers |
|---|---|
| `robot.gripper.*` | what the driver may command |
| `robot.grasping.gripper_geometry` | what shape the calculator sweeps |
| `robot.safety.self_collision` plus the cuRobo descriptor | what the planner and the guard refuse against |

Each is individually valid while the three disagree. That was reachable and undetectable: set
`max_width_mm: 50` for a Hand-E, leave the rest, and the cell commands a 50 mm stroke while the
collision filter sweeps a 2F-85's fingers and the planner refuses against a 2F-85's spheres. Nothing
warns, because until now every shipped cell was a 2F-85 and the defaults happened to describe it.

**The one line version.** One profile sets all three, and three numbers cannot be derived from any
file and have to come off the bench.

---

## Trigger

Any of:

- a Hand-E is being fitted to a cell that ran a different hand
- `python -m src.robot.safety.planning --doctor` reports `gripper geometry` or `gripper sphere map`
- a grasp contacts the object somewhere other than where the plan said, on a cell that recently
  changed grippers
- the planner refuses reachable poses, or accepts poses that collide, after a gripper swap

---

## Diagnose

Ask the artifacts, not your memory. The cuRobo descriptor states which hand it models:

```bash
python - <<'EOF'
import yaml, pathlib
cfg = yaml.safe_load(pathlib.Path("<curobo content>/configs/robot/ur3e_robotiq_hande.yml").read_text())
print(cfg.get("_provenance", "NO PROVENANCE: built before 2026-09-08, hand unknown"))
EOF
```

A descriptor with no `_provenance` block was built before the descriptor carried its own identity, so
which hand it models cannot be answered from the file and it has to be rebuilt.

Then ask the doctor, which now probes the hand and not only the arm:

```bash
python -m src.robot.safety.planning --doctor
```

Read three lines of its output:

- `collision mesh bundle (<arm>)` is the arm. It was green before this runbook existed even when the
  gripper geometry was missing entirely.
- `gripper bundle (<hand> on <arm>)` is the hand for this arm.
- `gripper sphere map (<hand>)` is what the planner will be given.

---

## Mitigate

### 1. Measure the coupling, once

Flange face to gripper mounting face, in millimetres, along the tool axis. Take the flange-to-TCP
transform in the same session: it is the same setup and the same reference surface, and doing them
apart is how the two end up describing different geometry.

The gripper's own contribution is already measured and does not need re-taking: **mounting face to
grasp centre is 135.75 mm**, off the asset. What you are measuring is the plate.

> **The rotational half is the dangerous half.** An offset wrong by 100 mm is a crash on run 1 and
> you fix it. A rotation wrong by 90 degrees is a cell that logs 10/10 successes while closing the
> jaws along the wrong object axis, and hand-eye calibration cannot catch it: it solves for whatever
> frame `get_tcp_pose()` reports and returns an excellent RMSE either way.

### 2. Select the hand in config

```bash
export WILLY_PROFILE=hande          # a UR5e cell
export WILLY_PROFILE=ur3e,hande     # a UR3e cell
```

That one profile sets the stroke, the collision envelope and the bundle the guard reads. It
deliberately leaves `robot.gripper.tool_frame.source` at `undeclared`, and the real UR driver refuses
to connect while it says that. Fill it in from step 1, in your own cell profile rather than in the
gripper one: the plate is a property of your cell and the gripper profile is shared.

### 3. Give the planner the hand

The guard reads config, and so does the planner's hand. **The descriptor is the arm alone** and is built
once per arm, on the box, with the cuRobo environment's interpreter:

```bash
ext_deps/curobo_env/python.exe scripts/curobo/build_ur_config.py ur3e
```

It refuses `--gripper` and `--coupling-mm` by name: the hand is not in the descriptor. When the planner
starts, its sidecar adds the hand this cell names as a fixed link under `tool0`, placed from
`robot.gripper.model`, `robot.gripper.coupling_plates_mm` and the declared tool frame. So the plate from
step 1 goes into the cell profile and nowhere else, and a plate nobody measured leaves the cell refusing
to build rather than modelling the hand one plate too close to the flange.

Check that it plans with the hand added:

```bash
ext_deps/curobo_env/python.exe scripts/curobo/check_ur_descriptors.py willy_ur3e --hand robotiq_hande --coupling-mm <the plate from step 1>
```

### 4. Declare the payload

`robot.safety.payload.mass_kg` and `cog_mm`, and the same numbers on the controller. The shipped tree
declares `mass_kg: 0.0`, which makes the payload guard's envelope check vacuous rather than wrong,
and a heavier hand eats the arm's rated payload with nothing noticing.

### 5. Declare the coupling plate, once, in two places

The plate between the arm flange and the gripper's own mounting face is a bench measurement, and it
is the same number every time:

```
robot.gripper.tool_frame.offset_mm            plate + 135.75 mm along the approach
robot.gripper.coupling_plates_mm              [plate]
```

Both are read at runtime, and both models of the hand are placed from them: the planner's hand link and
the exact-mesh guard's composed hand bundle. The check in step 3 takes the same number as
`--coupling-mm`, so a rehearsal models the cell being built rather than a bare flange.

`robotiq_hande_hand_meshes.npz` is stamped `gripper__origin = "mounting_face"`, which the bake module
defines as "whatever plate sits between that face and the flange has to be added before the planner sees
it". A Hand-E cell that writes no plate refuses to build: the hand's own arrays start at the mounting
face, and no plate written is no measurement rather than zero. `robot.hande.yaml` writes `[20.0]`, an
assumption to replace with this cell's bench number.

### 6. The planner does NOT know what it is carrying, and that is a decision

`robot.safety.planning_world.payload.enabled` ships `false`, so `attach_payload` returns `False` on
every pick and every lift, transit and place is planned with an EMPTY hand. The carried part is
absent from the planner's world model, and on a cell lifting out of a bin that part is the geometry
most likely to meet a wall.

⛔ **This is deliberate as of 2026-09-10, and the reason is worth more than the setting.** Turning it
on requires `length_mm`, how far a part extends beyond the grasp line, and the owner's answer was
that this cell handles SEVERAL parts, so no single length describes it. A wrong single number is
worse than none in both directions at once: too long refuses grasps that would work, too short lets
a part meet a wall the planner never modelled. The shipped default of 120 mm is plausible and
measured on nothing.

So the honest state is: this cell plans with an empty hand, on purpose, until the payload can be
described per part rather than per cell. Expect the first surprise at a bin wall, not in the logs.


---

## Verify

```bash
python -m src.robot.safety.planning --doctor
```

Everything in the gripper block reads `ok`, and the `gripper sphere map` warning about the coupling
is gone once the cell declares the plate.

```bash
python -c "
from src.config.loader import load_config
from src.robot.safety._fcl_self_collision import mesh_backend_status
from src.robot.safety.planning.hand import planner_hand
c = load_config(); h = planner_hand(c.robot)
print(mesh_backend_status(c.robot.ur.model, None, h.guard_variant))"
```

`ok`. Anything else means the whole cell is on the capsule proxy, arm included, not just the hand.

Then the arm, from [real_cell_first_pick.md](real_cell_first_pick.md), and one bench check of the
driver: command a width, read it back, and confirm the two agree within a millimetre.

---

## Rollback

Set `WILLY_PROFILE` back to the cell profile without `hande`. Nothing is rebuilt: the descriptor carries
no hand, so the planner adds whichever hand the profile then names.

The 2F-85's arrays are measured at the flange and need no plate, because they already include wherever
the arm put the hand. A profile naming it writes no `coupling_plates_mm`.

---

## What this cannot do for you

- **Nothing here has run on physical hardware.** Every number is measured off the vendor asset and
  against the committed bundles. A bench day is a different measurement.
- **Custom fingertips change `closed_width_mm`.** The profile ships `0.0`, which is the bare gripper:
  its carriages meet, measured. Fingertips that do not meet make that a real number, and it is now
  the single anchor of the driver's count map, so setting it is the whole fix rather than half of it.
- **The simulator's jaw direction is unconfirmed.** The asset's joint frames say a rising joint value
  opens the jaws; its geometry says that would open the hand to 100 mm, which no Hand-E does. The
  profile follows the geometry. If a sim run drives the jaws the wrong way, reverse
  `ROBOTIQ_HANDE_PROFILE.angle_width_table` and nothing else.
- **The collision envelope under-models one sliver, on purpose.** Between 24.6 and 34.7 mm behind the
  grasp centre the carriage shoulder narrows the passage to 36.7 mm, and a box model cannot say
  "thicker at the back". Covering it would declare the gap 13.3 mm narrower everywhere and refuse
  every object over 36.7 mm even for a shallow grasp. The box is a candidate filter; Coal and cuRobo
  refuse against the vertex-exact bundle, which has the shoulder in it, so the cost is a plan
  rejected later rather than a collision.
