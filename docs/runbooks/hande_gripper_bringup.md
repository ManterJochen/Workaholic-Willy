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
cfg = yaml.safe_load(pathlib.Path("<curobo content>/configs/robot/ur3e.yml").read_text())
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

The guard reads config. **The planner reads a descriptor**, so it does not change until this is run,
on the box, with the cuRobo environment's interpreter:

```bash
ext_deps/curobo_env/python.exe scripts/curobo/build_ur_config.py ur3e     --gripper robotiq_hande --coupling-mm <the plate from step 1>
```

It refuses without `--coupling-mm`, because the Hand-E's spheres are measured from the gripper's own
mounting face and assuming zero puts every one of them one plate too close to the flange. That is
optimistic in the one direction a planner must not be, and the resulting file looks entirely
reasonable. Passing `0` is a legitimate answer for a hand bolted straight to the flange; saying it
out loud is the point.

### 4. Declare the payload

`robot.safety.payload.mass_kg` and `cog_mm`, and the same numbers on the controller. The shipped tree
declares `mass_kg: 0.0`, which makes the payload guard's envelope check vacuous rather than wrong,
and a heavier hand eats the arm's rated payload with nothing noticing.

---

## Verify

```bash
python -m src.robot.safety.planning --doctor
```

Everything in the gripper block reads `ok`, and the `gripper sphere map` warning about the coupling
is gone once the descriptor was built with one.

```bash
python -c "
from src.config.loader import load_config
from src.robot.safety._fcl_self_collision import mesh_backend_status
c = load_config(); m = c.robot.ur.model
print(mesh_backend_status(m, None, c.robot.safety.self_collision.collision_mesh_variant))"
```

`ok`. Anything else means the whole cell is on the capsule proxy, arm included, not just the hand.

Then the arm, from [real_cell_first_pick.md](real_cell_first_pick.md), and one bench check of the
driver: command a width, read it back, and confirm the two agree within a millimetre.

---

## Rollback

Set `WILLY_PROFILE` back to the cell profile without `hande` and re-run step 3 with the old hand:

```bash
ext_deps/curobo_env/python.exe scripts/curobo/build_ur_config.py ur3e --gripper ur5e
```

The 2F-85's map is measured at the flange and takes no `--coupling-mm`; passing one is refused rather
than applied, because that map already includes wherever the arm put the hand.

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
